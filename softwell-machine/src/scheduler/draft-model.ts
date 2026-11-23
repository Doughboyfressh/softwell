/**
 * softwell-machine — scheduler: loading a draft model for speculative decoding.
 *
 * A draft model is a second loaded model: one shared StageExecutor (a
 * single pipeline stage holding all of the draft model's layers, with
 * embedding + lm_head) built from an already-opened WeightLoader, plus
 * one ModelDrafter per sequence, each with its own PagedKVCache. The
 * executor is the same reference transformer the target runs on — the
 * draft model drafts through the same virtual-hardware path as the
 * target; only the weights differ.
 *
 * Weight source is the caller's choice, and the loader is injected so
 * this module never decides where bytes come from:
 *  - serving path (src/cli.ts): the draft model is staged through the
 *    ModelRegistry (vSSD -> vRAM) like any model, and the loader is a
 *    VramWeightLoader reading tensor bytes from pooled vRAM — never
 *    from disk after staging.
 *  - tests / standalone: `loadDraftModelFromDir` opens the directory
 *    with the disk loader (openWeightLoader).
 *
 * Vocab contract: the draft and target models must share a tokenizer
 * and vocabulary — draft tokens are verified against the target's
 * logits position by position, so a vocab mismatch would verify
 * nonsense. The check is loud, not a silent fallback: `checkTargetVocab`
 * (called by the CLI at activation, before any sequence runs) and
 * `createDrafter` (per sequence, defence in depth) both throw a clear
 * error naming both vocab sizes. A draft model that cannot be loaded
 * (missing directory, unreadable weights) throws at load time with
 * the directory in the message.
 *
 * Deliberate boundary: the draft executor runs in-process on the
 * orchestrator. Sharding the draft model over mesh workers — its own
 * stages/workers in the mesh — needs ExecuteRequest to carry a model
 * discriminator, workers to hold a second executor + cache, and the
 * registry to hold a second plan; that is a follow-up, not this
 * module. ModelDrafter takes any StageExecutor, so a mesh-backed
 * executor plugs into the same DraftModel shape later without
 * changing the decoder or the scheduler.
 */

import type {
  EngineBackend,
  EngineOptions,
  ModelManifest,
  ShardAssignment,
  StageExecutor,
  WeightLoader,
} from "../contracts.ts";
import { ModelDrafter } from "../engine/novel-decoding.ts";
import { openWeightLoader } from "../model/loader.ts";
import { BlockPagedKVCache } from "./kvcache.ts";

/**
 * A loaded draft model: one shared draft executor plus per-sequence
 * drafters. The executor is built once (weights loaded once) and
 * shared by every sequence's drafter; each drafter owns its own KV
 * cache and seq id, so sequences never share draft state. Sharing the
 * executor is safe: TransformerStageExecutor keeps its per-sequence
 * incremental state keyed by `${seqId}:${layer}`, and every forward
 * takes the caller's cache.
 */
export interface DraftModel {
  readonly modelId: string;
  readonly manifest: ModelManifest;
  /**
   * Throw unless the draft model's vocab matches the target's. Called
   * at activation (before any sequence runs) so a mismatch fails the
   * activation loudly instead of failing the first sequence mid-run.
   */
  checkTargetVocab(targetVocabSize: number): void;
  /**
   * Build a drafter for one sequence. Re-checks the vocab (defence in
   * depth — the factory the scheduler holds is built per activation,
   * but a direct caller might skip checkTargetVocab).
   */
  createDrafter(seqId: string, targetVocabSize: number): ModelDrafter;
  /** Release the shared draft executor's weight loader (idempotent).
   *  Safe to call after load: the executor copied the tensors it needs
   *  into its own weight maps at load time. */
  close(): void;
}

export interface LoadDraftModelOptions {
  /** Already-opened weight loader for the draft model (disk or vRAM).
   *  The caller owns the error context for opening it; this function
   *  owns the load errors (missing tensors, shape mismatches). */
  loader: WeightLoader;
  backend: EngineBackend;
  engineOptions?: EngineOptions;
}

/** Load a draft model from an already-opened weight loader. */
export async function loadDraftModel(opts: LoadDraftModelOptions): Promise<DraftModel> {
  const manifest: ModelManifest = opts.loader.manifest;
  const cfg = manifest.config;
  if (cfg.vocabSize < 1) {
    throw new Error(
      `draft model ${manifest.modelId}: invalid vocabSize ${cfg.vocabSize} in its config`,
    );
  }
  if (cfg.numHiddenLayers < 1) {
    throw new Error(
      `draft model ${manifest.modelId}: invalid numHiddenLayers ${cfg.numHiddenLayers} in its config`,
    );
  }
  // Single pipeline stage: the whole draft model, embedding + lm_head.
  // The draft model is small by design (it must be cheap to run); a
  // sharded draft executor is the mesh follow-up noted above.
  const assignment: ShardAssignment = {
    workerId: "draft",
    stage: 0,
    layerStart: 0,
    layerEnd: cfg.numHiddenLayers,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 0,
  };
  const executor: StageExecutor = opts.backend.createStageExecutor(
    manifest,
    assignment,
    opts.engineOptions,
  );
  try {
    await executor.load(opts.loader);
  } catch (err) {
    throw new Error(
      `draft model ${manifest.modelId}: failed to load weights: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  let closed = false;
  const checkTargetVocab = (targetVocabSize: number): void => {
    if (targetVocabSize !== cfg.vocabSize) {
      throw new Error(
        `draft model ${manifest.modelId} has vocabSize ${cfg.vocabSize} but the target model ` +
          `has vocabSize ${targetVocabSize} — draft and target must share a tokenizer/vocabulary`,
      );
    }
  };
  return {
    modelId: manifest.modelId,
    manifest,
    checkTargetVocab,
    createDrafter(seqId: string, targetVocabSize: number): ModelDrafter {
      checkTargetVocab(targetVocabSize);
      const cache = new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
      return new ModelDrafter({
        executor,
        cache,
        seqId,
        vocabSize: cfg.vocabSize,
        eosTokenId: cfg.eosTokenId,
      });
    },
    close(): void {
      if (closed) return;
      closed = true;
      opts.loader.close();
    },
  };
}

/**
 * Load a draft model from a local directory with the disk loader.
 * A missing or unreadable directory throws a clear error naming the
 * directory (openWeightLoader's "model directory not found" is wrapped
 * with the draft-model context).
 */
export async function loadDraftModelFromDir(
  dir: string,
  backend: EngineBackend,
  engineOptions?: EngineOptions,
): Promise<DraftModel> {
  let loader: WeightLoader;
  try {
    loader = openWeightLoader(dir);
  } catch (err) {
    throw new Error(
      `draft model "${dir}": cannot load model directory: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  try {
    return await loadDraftModel({ loader, backend, engineOptions });
  } catch (err) {
    loader.close();
    throw err;
  }
}
