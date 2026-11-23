/**
 * Model-drafter tests — run with: node test/model-drafter.test.ts
 *
 * The real draft model is demo-10m drafting for the real target
 * demo-100m (both already in demo/models/, both share the 259-token
 * byte vocab). The properties under test:
 *  (1) identity — speculative decoding with a ModelDrafter is
 *      token-for-token identical to plain autoregressive decoding of
 *      the target (greedy, Theorem S1 exactness), even though the
 *      draft model is a different, smaller model;
 *  (2) acceptance accounting — draftProposed/draftAccepted/
 *      acceptanceRate are real counts from the model drafter (and the
 *      drafter's own forward counter proves it ran);
 *  (3) draft-model loading — a missing directory and a vocab mismatch
 *      vs the target fail loudly with clear errors, never a silent
 *      fallback to the n-gram drafter.
 *
 * NOTE on the numbers: both demo models have random weights and their
 * greedy output collapses to a single repeated token (demo-100m emits
 * 35s, demo-10m emits 57s for this prompt), so the draft model will
 * mostly be WRONG here — acceptance is expected to be low, and that
 * is the honest measurement for this pair. The claim under test is
 * exactness + real accounting, not a speedup. Real acceptance numbers
 * belong to the bench workstream, on the trained SmolLM pair.
 */

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import * as path from "node:path";
import type {
  PagedKVCache,
  SamplingParams,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../src/contracts.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import { ByteLevelTokenizer } from "../src/model/tokenizer.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  AutoregressiveDecoder,
  ModelDrafter,
  SpeculativeDecoder,
} from "../src/engine/novel-decoding.ts";
import { loadDraftModel, loadDraftModelFromDir } from "../src/scheduler/draft-model.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DRAFT_DIR = path.join(ROOT, "demo/models/demo-10m");
const TARGET_DIR = path.join(ROOT, "demo/models/demo-100m");
const backend = createReferenceBackend();

function fullAssignment(workerId: string, numLayers: number): ShardAssignment {
  return {
    workerId,
    stage: 0,
    layerStart: 0,
    layerEnd: numLayers,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 0,
  };
}

// ---------------------------------------------------------------------------
// Loading errors (fast: no forwards needed)
// ---------------------------------------------------------------------------

{
  // Missing directory: a clear error naming the directory, not a
  // silent fallback and not a bare ENOENT.
  await assert.rejects(
    () => loadDraftModelFromDir(path.join(ROOT, "demo/models/does-not-exist"), backend),
    /draft model ".*does-not-exist": cannot load model directory: model directory not found/,
    "missing draft-model dir throws a clear error",
  );
  console.log("ok draft-model loading: missing directory fails loudly with the dir in the message");

  // A loader whose manifest cannot load weights (empty dir has no
  // config): loadDraftModel wraps the failure with the model id.
  const emptyDir = path.join(ROOT, "demo/models");
  await assert.rejects(
    () => loadDraftModelFromDir(emptyDir, backend),
    /draft model/,
    "unreadable draft-model dir throws",
  );
  console.log("ok draft-model loading: unreadable directory throws");

  // Vocab mismatch vs the target: the draft model is demo-10m
  // (vocab 259); asking for a drafter against a 49152-vocab target
  // must throw naming both sizes — at check time AND at create time.
  const draft = await loadDraftModelFromDir(DRAFT_DIR, backend);
  assert.equal(draft.manifest.config.vocabSize, 259, "draft model vocab is 259");
  assert.throws(
    () => draft.checkTargetVocab(49152),
    /draft model .* has vocabSize 259 but the target model has vocabSize 49152/,
    "checkTargetVocab throws on mismatch, naming both vocabs",
  );
  assert.throws(
    () => draft.createDrafter("seq-x", 49152),
    /draft model .* has vocabSize 259 but the target model has vocabSize 49152/,
    "createDrafter throws on mismatch too",
  );
  draft.checkTargetVocab(259);
  const ok = draft.createDrafter("seq-ok", 259);
  assert.equal(ok.vocabSize, 259, "matching vocab creates a drafter");
  await ok.free();
  draft.close();
  console.log("ok draft-model loading: vocab mismatch vs target fails loudly, match succeeds");
}

// ---------------------------------------------------------------------------
// ModelDrafter unit behaviour (demo-10m drafting for itself)
// ---------------------------------------------------------------------------

const draftModel = await loadDraftModelFromDir(DRAFT_DIR, backend);
const draftCfg = draftModel.manifest.config;
console.log(
  `ok draft model loaded: ${draftModel.modelId}, hidden ${draftCfg.hiddenSize}, ${draftCfg.numHiddenLayers} layers, vocab ${draftCfg.vocabSize}`,
);

function draftCache(): PagedKVCache {
  return new BlockPagedKVCache(16, 1024, draftCfg.numKeyValueHeads, draftCfg.headDim);
}

{
  const tokenizer = new ByteLevelTokenizer({
    bosTokenId: draftCfg.bosTokenId,
    eosTokenId: draftCfg.eosTokenId,
    padTokenId: draftCfg.padTokenId,
  });
  const ctx = tokenizer.encode("Hello world");
  const d1 = draftModel.createDrafter("unit-1", draftCfg.vocabSize);
  const d2 = draftModel.createDrafter("unit-2", draftCfg.vocabSize);
  const tokens1 = await d1.draft(ctx, 4);
  const tokens2 = await d2.draft(ctx, 4);
  assert.equal(tokens1.length, 4, "draft() returns K tokens");
  assert.deepEqual(tokens1, tokens2, "greedy drafting is deterministic across fresh drafters");
  for (const t of tokens1) {
    assert.ok(t >= 0 && t < draftCfg.vocabSize, `draft token ${t} in vocab`);
  }
  assert.ok(d1.forwards > 0, "the draft model actually ran forwards");
  assert.deepEqual(await d1.draft(ctx, 0), [], "draft(…, 0) is empty");
  // A second draft() call continues the same sequence: it must not
  // re-feed the whole context, and its tokens must follow from the
  // extended context (prompt + first draft).
  const forwardsBefore = d1.forwards;
  const more = await d1.draft([...ctx, ...tokens1], 2);
  assert.equal(more.length, 2, "second draft() call returns K tokens");
  assert.ok(d1.forwards > forwardsBefore, "second draft() call ran more forwards");
  await d1.free();
  await d2.free();
  await assert.rejects(
    () => d1.draft(ctx, 1),
    /draft\(\) called after free/,
    "draft() after free() throws",
  );
  console.log(
    `ok ModelDrafter unit: deterministic drafts [${tokens1.join(", ")}] then [${more.join(", ")}], ${d1.forwards} forwards, free() enforced`,
  );
}

// ---------------------------------------------------------------------------
// Identity: demo-10m drafting for demo-100m == demo-100m autoregressive
// ---------------------------------------------------------------------------

const targetLoader: WeightLoader = openWeightLoader(TARGET_DIR);
const targetManifest = targetLoader.manifest;
const targetCfg = targetManifest.config;
assert.equal(
  targetCfg.vocabSize,
  draftCfg.vocabSize,
  "demo-10m and demo-100m share the 259-token vocab",
);
const targetExecutor: StageExecutor = backend.createStageExecutor(
  targetManifest,
  fullAssignment("target-worker", targetCfg.numHiddenLayers),
);
await targetExecutor.load(targetLoader);
console.log(
  `ok target loaded: ${targetManifest.modelId}, hidden ${targetCfg.hiddenSize}, ${targetCfg.numHiddenLayers} layers, vocab ${targetCfg.vocabSize}`,
);

function targetCache(): PagedKVCache {
  return new BlockPagedKVCache(16, 1024, targetCfg.numKeyValueHeads, targetCfg.headDim);
}

const tokenizer = new ByteLevelTokenizer({
  bosTokenId: targetCfg.bosTokenId,
  eosTokenId: targetCfg.eosTokenId,
  padTokenId: targetCfg.padTokenId,
});
const prompt = tokenizer.encode("Hello world");
const greedy: SamplingParams = { temperature: 0, topP: 1, topK: 0, maxTokens: 8 };

const base = await new AutoregressiveDecoder({
  executor: targetExecutor,
  cache: targetCache(),
  seqId: "target-base",
  promptTokenIds: prompt,
  sampling: greedy,
  vocabSize: targetCfg.vocabSize,
  eosTokenId: targetCfg.eosTokenId,
}).generate();
console.log(`target baseline (greedy, 8 tokens): [${base.tokens.join(", ")}] in ${base.forwards} forwards`);

{
  const drafter = draftModel.createDrafter("spec-1", targetCfg.vocabSize);
  const spec = await new SpeculativeDecoder({
    executor: targetExecutor,
    cache: targetCache(),
    seqId: "target-spec",
    promptTokenIds: prompt,
    sampling: greedy,
    vocabSize: targetCfg.vocabSize,
    eosTokenId: targetCfg.eosTokenId,
    draftLength: 4,
    drafter,
  }).generate();
  assert.deepEqual(
    spec.tokens,
    base.tokens,
    "speculative with ModelDrafter tokens must equal baseline token-for-token",
  );
  assert.equal(spec.finishReason, base.finishReason, "speculative finish reason matches baseline");
  // Acceptance accounting is real: the model drafter proposed drafts,
  // the counts cohere, and the drafter's own forward counter proves
  // the tokens came from the draft model, not the n-gram path.
  assert.ok(spec.draftProposed > 0, "model drafter actually proposed drafts");
  assert.ok(
    spec.draftAccepted >= 0 && spec.draftAccepted <= spec.draftProposed,
    `accepted ${spec.draftAccepted} within proposed ${spec.draftProposed}`,
  );
  assert.ok(
    Math.abs(spec.acceptanceRate - spec.draftAccepted / spec.draftProposed) < 1e-12,
    "acceptanceRate == accepted / proposed",
  );
  assert.ok(drafter.forwards > 0, `model drafter ran ${drafter.forwards} forwards`);
  console.log(
    `ok identity (demo-10m drafting for demo-100m): speculative == baseline [${spec.tokens.join(", ")}], ` +
      `${spec.forwards} target forwards vs ${base.forwards} baseline, ` +
      `acceptance ${spec.draftAccepted}/${spec.draftProposed} = ${spec.acceptanceRate.toFixed(3)} ` +
      `(draft model ran ${drafter.forwards} forwards)`,
  );
  await drafter.free();

  // Second prompt, same identity property (also exercises a fresh
  // drafter on a different context).
  const prompt2 = tokenizer.encode("The quick brown fox jumps");
  const base2 = await new AutoregressiveDecoder({
    executor: targetExecutor,
    cache: targetCache(),
    seqId: "target-base-2",
    promptTokenIds: prompt2,
    sampling: greedy,
    vocabSize: targetCfg.vocabSize,
    eosTokenId: targetCfg.eosTokenId,
  }).generate();
  const drafter2 = draftModel.createDrafter("spec-2", targetCfg.vocabSize);
  const spec2 = await new SpeculativeDecoder({
    executor: targetExecutor,
    cache: targetCache(),
    seqId: "target-spec-2",
    promptTokenIds: prompt2,
    sampling: greedy,
    vocabSize: targetCfg.vocabSize,
    eosTokenId: targetCfg.eosTokenId,
    draftLength: 4,
    drafter: drafter2,
  }).generate();
  assert.deepEqual(spec2.tokens, base2.tokens, "second prompt: speculative == baseline");
  assert.ok(spec2.draftProposed > 0, "second prompt: model drafter proposed drafts");
  console.log(
    `ok identity (second prompt): speculative == baseline [${spec2.tokens.join(", ")}], ` +
      `acceptance ${spec2.draftAccepted}/${spec2.draftProposed} = ${spec2.acceptanceRate.toFixed(3)}`,
  );
  await drafter2.free();
}

draftModel.close();
targetLoader.close();

console.log("ALL MODEL-DRAFTER TESTS PASSED");
