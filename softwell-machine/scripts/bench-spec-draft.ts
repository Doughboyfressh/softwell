/**
 * bench-spec-draft.ts — one short speculative run vs the baseline, same prompt.
 *
 * Target: SmolLM-135M (real trained model). Draft: draft-smollm (small model
 * trained on ~100k TinyStories tokens, same tokenizer/vocab). Both run
 * in-process on the reference backend. Temp 0, greedy — speculative output
 * must be token-for-token identical to the baseline (Theorem S1).
 *
 * Usage: node scripts/bench-spec-draft.ts [prompt] [maxTokens] [draftLength]
 * Writes spec-draft-bench.json next to this script's working directory.
 */
import * as fs from "node:fs";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { AutoregressiveDecoder, SpeculativeDecoder, forwardTokenRows } from "../src/engine/novel-decoding.ts";
import { loadDraftModelFromDir } from "../src/scheduler/draft-model.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import type { ForwardContext, ShardAssignment } from "../src/contracts.ts";

const TARGET_DIR = "./models/smollm-135m";
const DRAFT_DIR = "./models/draft-smollm";
const prompt = process.argv[2] ?? "The capital of France is";
const maxTokens = Number(process.argv[3] ?? 16);
const draftLength = Number(process.argv[4] ?? 4);

const backend = createReferenceBackend();

// ---- Load target ----
const targetManifest = loadManifest(TARGET_DIR, "smollm-135m");
const tokenizer = loadTokenizer(TARGET_DIR);
const targetLoader = openWeightLoader(TARGET_DIR, "smollm-135m");
const tCfg = targetManifest.config;
const targetAssignment: ShardAssignment = {
  workerId: "target", stage: 0, layerStart: 0, layerEnd: tCfg.numHiddenLayers,
  hasEmbedding: true, hasLmHead: true, tensorParallelRank: 0, tensorParallelSize: 1,
  experts: [], estimatedMemoryBytes: 0,
};
const targetExecutor = backend.createStageExecutor(targetManifest, targetAssignment);
const tLoad0 = performance.now();
await targetExecutor.load(targetLoader);
console.log(`target loaded in ${((performance.now() - tLoad0) / 1000).toFixed(1)}s`);

// ---- Load draft ----
const draftModel = await loadDraftModelFromDir(DRAFT_DIR, backend);
console.log(`draft loaded: ${draftModel.modelId}, vocab ${draftModel.manifest.config.vocabSize}`);
draftModel.checkTargetVocab(tCfg.vocabSize);

const promptIds = tokenizer.encode(prompt);
console.log(`prompt: ${JSON.stringify(prompt)} -> ids [${promptIds.join(", ")}] (${promptIds.length} tokens)`);
const sampling = { temperature: 0, topP: 1, topK: 0, maxTokens, seed: 1 };

// ---- Baseline: plain autoregressive ----
const baseCache = new BlockPagedKVCache(16, 1024, tCfg.numKeyValueHeads, tCfg.headDim);
const baseDecoder = new AutoregressiveDecoder({
  executor: targetExecutor, cache: baseCache, seqId: "base",
  promptTokenIds: promptIds, sampling, vocabSize: tCfg.vocabSize, eosTokenId: tCfg.eosTokenId,
});
const tBase0 = performance.now();
const baseResult = await baseDecoder.generate(maxTokens);
const baseMs = performance.now() - tBase0;
const baseTokPerSec = (baseResult.tokens.length / baseMs) * 1000;
console.log(`baseline: ${baseResult.tokens.length} tokens in ${baseMs.toFixed(0)} ms (${baseTokPerSec.toFixed(2)} tok/s), forwards=${baseResult.forwards}, rows=${baseResult.rowsProcessed}`);
console.log(`baseline tokens: [${baseResult.tokens.join(", ")}]`);
console.log(`baseline text: ${JSON.stringify(tokenizer.decode(baseResult.tokens))}`);

// ---- Speculative: real model drafter, K=4 ----
const specCache = new BlockPagedKVCache(16, 1024, tCfg.numKeyValueHeads, tCfg.headDim);
const drafter = draftModel.createDrafter("spec", tCfg.vocabSize);
const specDecoder = new SpeculativeDecoder({
  executor: targetExecutor, cache: specCache, seqId: "spec",
  promptTokenIds: promptIds, sampling, vocabSize: tCfg.vocabSize, eosTokenId: tCfg.eosTokenId,
  draftLength, drafter,
});
const tSpec0 = performance.now();
const specResult = await specDecoder.generate(maxTokens);
const specMs = performance.now() - tSpec0;
const specTokPerSec = (specResult.tokens.length / specMs) * 1000;
console.log(`speculative: ${specResult.tokens.length} tokens in ${specMs.toFixed(0)} ms (${specTokPerSec.toFixed(2)} tok/s), forwards=${specResult.forwards}, rows=${specResult.rowsProcessed}, rounds=${specResult.rounds}`);
console.log(`speculative tokens: [${specResult.tokens.join(", ")}]`);
console.log(`acceptance: ${specResult.draftAccepted}/${specResult.draftProposed} = ${specResult.acceptanceRate.toFixed(3)}`);
console.log(`draft forwards: ${drafter.forwards}`);
const identical = JSON.stringify(specResult.tokens) === JSON.stringify(baseResult.tokens);
console.log(`token identity vs baseline: ${identical ? "IDENTICAL" : "MISMATCH"}`);

// ---- Beta probe: 1-row vs 5-row forward cost on the target ----
const probeCache = new BlockPagedKVCache(16, 1024, tCfg.numKeyValueHeads, tCfg.headDim);
const probeSeq = "probe";
const prefill = await forwardTokenRows(targetExecutor, probeCache, probeSeq, promptIds, promptIds.map((_, i) => i), true);
void prefill;
const lastTok = promptIds[promptIds.length - 1] as number;
async function timeRows(rows: number, reps: number): Promise<number> {
  let total = 0;
  for (let r = 0; r < reps; r++) {
    const ids = Array.from({ length: rows }, () => lastTok);
    const positions = ids.map((_, i) => promptIds.length + r * rows + i);
    const t0 = performance.now();
    await forwardTokenRows(targetExecutor, probeCache, probeSeq, ids, positions, false);
    total += performance.now() - t0;
  }
  return total / reps;
}
const t1 = await timeRows(1, 3);
const t5 = await timeRows(5, 3);
const beta = (t5 / t1 - 1) / 4;
console.log(`beta probe: 1-row forward ${t1.toFixed(1)} ms, 5-row forward ${t5.toFixed(1)} ms -> beta ~= ${beta.toFixed(2)}`);

const out = {
  label: "spec-draft-short",
  timestamp: new Date().toISOString(),
  prompt,
  promptTokens: promptIds.length,
  maxTokens,
  draftLength,
  target: "smollm-135m",
  draft: "draft-smollm",
  baseline: {
    tokens: baseResult.tokens.length,
    ms: Math.round(baseMs),
    tokPerSec: Number(baseTokPerSec.toFixed(2)),
    forwards: baseResult.forwards,
    rowsProcessed: baseResult.rowsProcessed,
    tokenIds: baseResult.tokens,
    text: tokenizer.decode(baseResult.tokens),
  },
  speculative: {
    tokens: specResult.tokens.length,
    ms: Math.round(specMs),
    tokPerSec: Number(specTokPerSec.toFixed(2)),
    forwards: specResult.forwards,
    rowsProcessed: specResult.rowsProcessed,
    rounds: specResult.rounds,
    draftProposed: specResult.draftProposed,
    draftAccepted: specResult.draftAccepted,
    acceptanceRate: Number(specResult.acceptanceRate.toFixed(3)),
    draftForwards: drafter.forwards,
    tokenIds: specResult.tokens,
    text: tokenizer.decode(specResult.tokens),
    identicalToBaseline: identical,
  },
  speedup: Number((specTokPerSec / baseTokPerSec).toFixed(2)),
  betaProbe: { oneRowMs: Number(t1.toFixed(1)), fiveRowMs: Number(t5.toFixed(1)), beta: Number(beta.toFixed(2)) },
};
fs.writeFileSync("spec-draft-bench.json", JSON.stringify(out, null, 2) + "\n");
console.log(`wrote spec-draft-bench.json — speedup ${out.speedup}x, acceptance ${out.speculative.acceptanceRate}`);
draftModel.close();
