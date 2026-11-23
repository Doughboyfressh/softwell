/**
 * bench-forward-opt.ts — before/after bench for the forward-pass optimization piece.
 *
 * In-process SmolLM-135M on the reference backend, same protocol as
 * scripts/bench-spec-draft.ts (short prompt, temp 0, greedy): 3 reps of the
 * full 16-token generate (median tok/s), 3 reps of the prefill probe, one
 * decode-only run (median single-row forward), plus ONE long-prompt rep
 * (fox sentence x22, same construction as bench-kv.mjs; token count is
 * measured, not assumed — SmolLM's BPE tokenizes it differently than the
 * demo byte-level tokenizer).
 *
 * Usage: node scripts/bench-forward-opt.ts <before|after>
 * Writes forward-opt-bench-<phase>.json in the working directory.
 */
import * as fs from "node:fs";
import * as os from "node:os";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { AutoregressiveDecoder, forwardTokenRows } from "../src/engine/novel-decoding.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import type { ShardAssignment } from "../src/contracts.ts";

const phase = process.argv[2] ?? "before";
if (phase !== "before" && phase !== "after") throw new Error(`usage: bench-forward-opt.ts <before|after> [short-only], got ${phase}`);
const shortOnly = process.argv[3] === "short-only";

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
function hostInfo() {
  return {
    node: process.version, platform: process.platform, arch: process.arch,
    cpus: os.cpus().length, cpuModel: os.cpus()[0]?.model ?? null,
    totalMemMiB: Math.round(os.totalmem() / 1048576),
    freeMemMiB: Math.round(os.freemem() / 1048576),
    loadavg: os.loadavg(),
  };
}

const TARGET_DIR = "./models/smollm-135m";
const backend = createReferenceBackend();
const manifest = loadManifest(TARGET_DIR, "smollm-135m");
const tokenizer = loadTokenizer(TARGET_DIR);
const loader = openWeightLoader(TARGET_DIR, "smollm-135m");
const cfg = manifest.config;
const assignment: ShardAssignment = {
  workerId: "fwd-opt", stage: 0, layerStart: 0, layerEnd: cfg.numHiddenLayers,
  hasEmbedding: true, hasLmHead: true, tensorParallelRank: 0, tensorParallelSize: 1,
  experts: [], estimatedMemoryBytes: 0,
};
const executor = backend.createStageExecutor(manifest, assignment);
const tL0 = performance.now();
await executor.load(loader);
console.log(`[${phase}] target loaded in ${((performance.now() - tL0) / 1000).toFixed(1)}s, host ${JSON.stringify(hostInfo())}`);

const newCache = () => new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
const sampling = { temperature: 0, topP: 1, topK: 0, maxTokens: 16, seed: 1 };

// ---- Short prompt ----
const shortPrompt = "The capital of France is";
const shortIds = tokenizer.encode(shortPrompt);
console.log(`[${phase}] short prompt ${JSON.stringify(shortPrompt)} -> ${shortIds.length} tokens`);

// Prefill probe, 3 reps.
const prefillMs: number[] = [];
for (let r = 0; r < 3; r++) {
  const cache = newCache();
  const t0 = performance.now();
  await forwardTokenRows(executor, cache, `pre-${r}`, shortIds, shortIds.map((_, i) => i), true);
  prefillMs.push(performance.now() - t0);
  await cache.close();
}
console.log(`[${phase}] prefill ${shortIds.length} tok ms: [${prefillMs.map((v) => v.toFixed(0)).join(", ")}] median=${median(prefillMs).toFixed(0)}`);

// Full generate, 3 reps.
const genRuns: { rep: number; ms: number; tokPerSec: number; tokens: number[]; text: string }[] = [];
for (let r = 0; r < 3; r++) {
  const cache = newCache();
  const dec = new AutoregressiveDecoder({
    executor, cache, seqId: `gen-${r}`, promptTokenIds: shortIds, sampling,
    vocabSize: cfg.vocabSize, eosTokenId: cfg.eosTokenId,
  });
  const t0 = performance.now();
  const res = await dec.generate(16);
  const ms = performance.now() - t0;
  genRuns.push({
    rep: r + 1, ms: Math.round(ms),
    tokPerSec: Number(((res.tokens.length / ms) * 1000).toFixed(2)),
    tokens: res.tokens, text: tokenizer.decode(res.tokens),
  });
  console.log(`[${phase}] generate rep ${r + 1}: ${ms.toFixed(0)}ms = ${genRuns[r]!.tokPerSec} tok/s`);
  await cache.close();
}
const medianGenTokPerSec = median(genRuns.map((g) => g.tokPerSec));
const tokenIdentity = genRuns.every((g) => JSON.stringify(g.tokens) === JSON.stringify(genRuns[0]!.tokens));
console.log(`[${phase}] short median generate: ${medianGenTokPerSec} tok/s, token identity across reps: ${tokenIdentity}`);

// Decode-only: prefill once, then 16 single-row forwards, median ms.
const decCache = newCache();
await forwardTokenRows(executor, decCache, "dec", shortIds, shortIds.map((_, i) => i), true);
const decMs: number[] = [];
let tok = shortIds[shortIds.length - 1] as number;
for (let i = 0; i < 16; i++) {
  const t0 = performance.now();
  const out = await forwardTokenRows(executor, decCache, "dec", [tok], [shortIds.length + i], false);
  decMs.push(performance.now() - t0);
  const logits = out.data.subarray(0, cfg.vocabSize);
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  tok = best;
}
await decCache.close();
const medianDecMs = median(decMs);
console.log(`[${phase}] decode-only forwards ms: [${decMs.map((v) => v.toFixed(0)).join(", ")}] median=${medianDecMs.toFixed(1)}ms -> ${(1000 / medianDecMs).toFixed(2)} tok/s`);

// ---- Long prompt, 1 rep (skipped in short-only core mode) ----
let longSection: unknown = null;
if (!shortOnly) {
const longText = "The quick brown fox jumps over the lazy dog. ".repeat(22);
const longIds = tokenizer.encode(longText);
console.log(`[${phase}] long prompt chars=${longText.length} tokens=${longIds.length} (measured)`);
const longCache = newCache();
const tLong0 = performance.now();
await forwardTokenRows(executor, longCache, "long", longIds, longIds.map((_, i) => i), true);
const longPrefillMs = performance.now() - tLong0;
const longDecMs: number[] = [];
let ltok = longIds[longIds.length - 1] as number;
for (let i = 0; i < 8; i++) {
  const t0 = performance.now();
  const out = await forwardTokenRows(executor, longCache, "long", [ltok], [longIds.length + i], false);
  longDecMs.push(performance.now() - t0);
  const logits = out.data.subarray(0, cfg.vocabSize);
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  ltok = best;
}
await longCache.close();
const medianLongDecMs = median(longDecMs);
console.log(`[${phase}] long prefill ${longIds.length} tok: ${longPrefillMs.toFixed(0)}ms; decode median ${medianLongDecMs.toFixed(1)}ms -> ${(1000 / medianLongDecMs).toFixed(2)} tok/s`);
longSection = {
    promptConstruction: { sentence: "The quick brown fox jumps over the lazy dog. ", repeats: 22, chars: longText.length },
    promptTokensMeasured: longIds.length,
    reps: 1,
    prefillMs: Math.round(longPrefillMs),
    decodeMs: longDecMs.map((v) => Math.round(v)),
    medianDecodeMs: Number(medianLongDecMs.toFixed(1)),
    decodeTokPerSec: Number((1000 / medianLongDecMs).toFixed(2)),
  };
}

const out = {
  label: `forward-opt-${phase}`,
  phase,
  timestamp: new Date().toISOString(),
  model: "smollm-135m",
  params: 134015488,
  host: hostInfo(),
  short: {
    prompt: shortPrompt, promptTokens: shortIds.length, maxTokens: 16,
    prefillMs: prefillMs.map((v) => Math.round(v)),
    medianPrefillMs: Math.round(median(prefillMs)),
    generateRuns: genRuns,
    medianGenerateTokPerSec: medianGenTokPerSec,
    tokenIdentityAcrossReps: tokenIdentity,
    decodeOnlyMs: decMs.map((v) => Math.round(v)),
    medianDecodeMs: Number(medianDecMs.toFixed(1)),
    decodeOnlyTokPerSec: Number((1000 / medianDecMs).toFixed(2)),
  },
  long: longSection,
};
fs.writeFileSync(`forward-opt-bench-${phase}.json`, JSON.stringify(out, null, 2) + "\n");
console.log(`[${phase}] wrote forward-opt-bench-${phase}.json`);
