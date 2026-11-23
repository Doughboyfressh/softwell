/**
 * Engine-options wiring tests — run with: node test/engine-options.test.ts
 *
 * Covers: option parsing/validation/defaults, extendBlockStats vs
 * computeBlockStats, concatPackedTensors, and the TransformerStageExecutor
 * under each novel mode on the real demo-10m model (random weights).
 *
 * On random-weight keys the landmark/sparse paths degrade gracefully to
 * exact attention (every block is incoherent, so nothing is approximated
 * and nothing is skippable): their logits must therefore match exact
 * attention to float32 noise. That is the honest expectation for this
 * model — the speedup regimes need coherent/concentrated keys, which
 * random weights do not have. SW-code paths move the logits (measured
 * below); the assertions bound the movement, they do not claim quality.
 */

import assert from "node:assert/strict";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { EngineOptions, ShardAssignment } from "../src/contracts.ts";
import {
  describeEngineOptions,
  isBaselineOptions,
  parseEngineFlags,
  resolveEngineOptions,
  validateEngineOptions,
} from "../src/engine/options.ts";
import {
  computeBlockStats,
  extendBlockStats,
} from "../src/engine/novel-attention.ts";
import {
  concatPackedTensors,
  decodeMatrix,
  encodeMatrix,
} from "../src/engine/novel-numbers.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MODEL_DIR = path.join(ROOT, "demo/models/demo-10m");

// ---------------------------------------------------------------------------
// Options: parse / validate / resolve
// ---------------------------------------------------------------------------

{
  assert.deepEqual(resolveEngineOptions({}), {
    attention: "exact",
    weightFormat: "fp32",
    kvFormat: "fp32",
    decoding: "autoregressive",
    draftLength: 4,
    npu: "off",
  });
  assert.equal(isBaselineOptions({}), true);
  assert.equal(isBaselineOptions({ attention: "landmark" }), false);
  assert.equal(isBaselineOptions({ decoding: "speculative" }), false);
  assert.equal(isBaselineOptions({ npu: "int8" }), false);
  assert.equal(isBaselineOptions({ npu: "fp8" }), false);

  const parsed = parseEngineFlags(["run", "model", "--attention", "landmark", "--weights=sw3", "--draft-length", "8"]);
  assert.equal(parsed.options.attention, "landmark");
  assert.equal(parsed.options.weightFormat, "sw3");
  assert.equal(parsed.options.draftLength, 8);
  assert.deepEqual(parsed.rest, ["run", "model"]);

  const parsedNpu = parseEngineFlags(["run", "model", "--npu", "int8"]);
  assert.equal(parsedNpu.options.npu, "int8");
  assert.deepEqual(parsedNpu.rest, ["run", "model"]);
  const parsedNpuEq = parseEngineFlags(["--npu=fp8"]);
  assert.equal(parsedNpuEq.options.npu, "fp8");
  assert.equal(resolveEngineOptions({ npu: "fp8" }).npu, "fp8");
  assert.throws(() => parseEngineFlags(["--npu", "bogus"]), /invalid engine option --npu/);
  assert.throws(() => parseEngineFlags(["--npu"]), /missing value for --npu/);
  assert.throws(() => validateEngineOptions({ npu: "bf16" as never }), /invalid engine option npu/);
  validateEngineOptions({ npu: "int8" });

  assert.throws(() => parseEngineFlags(["--attention", "bogus"]), /invalid engine option --attention/);
  assert.throws(() => parseEngineFlags(["--weights"]), /missing value for --weights/);
  assert.throws(() => parseEngineFlags(["--draft-length", "0"]), /invalid --draft-length/);
  assert.throws(() => validateEngineOptions({ kvFormat: "fp8" as never }), /invalid engine option kvFormat/);
  validateEngineOptions({ attention: "sparse", weightFormat: "sw2", kvFormat: "sw4", decoding: "jacobi", draftLength: 2 });

  assert.equal(
    describeEngineOptions({ attention: "sparse", weightFormat: "sw3" }),
    "attention=sparse weights=sw3 kv=fp32 decoding=autoregressive draft-length=4 npu=off",
  );
  assert.equal(
    describeEngineOptions({ npu: "int8" }),
    "attention=exact weights=fp32 kv=fp32 decoding=autoregressive draft-length=4 npu=int8",
  );
  console.log("ok options: defaults, parsing, validation");
}

// ---------------------------------------------------------------------------
// extendBlockStats is bit-identical to a full rebuild
// ---------------------------------------------------------------------------

{
  const mulberry32 = (seed: number): (() => number) => {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  const rand = mulberry32(99);
  const kvHeads = 2;
  const headDim = 8;
  const blockSize = 4;
  const totalLen = 37;
  const k = new Float32Array(totalLen * kvHeads * headDim);
  const v = new Float32Array(totalLen * kvHeads * headDim);
  for (let i = 0; i < k.length; i++) {
    k[i] = rand() * 2 - 1;
    v[i] = rand() * 2 - 1;
  }
  // Build over the first 10 rows, then extend one row at a time to 37.
  let stats = computeBlockStats(k.subarray(0, 10 * kvHeads * headDim), v.subarray(0, 10 * kvHeads * headDim), kvHeads, headDim, blockSize).stats;
  for (let len = 11; len <= totalLen; len++) {
    const extended = extendBlockStats(stats, k.subarray(0, len * kvHeads * headDim), v.subarray(0, len * kvHeads * headDim), len);
    stats = extended.stats;
    const fresh = computeBlockStats(k.subarray(0, len * kvHeads * headDim), v.subarray(0, len * kvHeads * headDim), kvHeads, headDim, blockSize).stats;
    assert.deepEqual([...stats.meanK], [...fresh.meanK], `extend to ${len}: meanK identical`);
    assert.deepEqual([...stats.meanV], [...fresh.meanV], `extend to ${len}: meanV identical`);
    assert.deepEqual([...stats.radiusK], [...fresh.radiusK], `extend to ${len}: radiusK identical`);
    assert.deepEqual([...stats.radiusV], [...fresh.radiusV], `extend to ${len}: radiusV identical`);
    assert.deepEqual([...stats.maxVNorm], [...fresh.maxVNorm], `extend to ${len}: maxVNorm identical`);
  }
  // No-op extension and shrinking.
  assert.equal(extendBlockStats(stats, k, v, totalLen).stats, stats, "extending to the same length is a no-op");
  assert.throws(() => extendBlockStats(stats, k, v, totalLen - 1), /shrinking needs a rebuild/);
  console.log("ok extendBlockStats: bit-identical to computeBlockStats across 27 incremental extensions");
}

// ---------------------------------------------------------------------------
// concatPackedTensors decodes to the row-wise concatenation
// ---------------------------------------------------------------------------

{
  const a = new Float32Array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6]);
  const b = new Float32Array([-0.7, 0.8, 0.9, -1.0]);
  const pa = encodeMatrix(a, 3, 2, "sw3");
  const pb = encodeMatrix(b, 2, 2, "sw3");
  const joined = concatPackedTensors(pa, pb);
  assert.equal(joined.rows, 5);
  assert.equal(joined.cols, 2);
  const decoded = decodeMatrix(joined);
  const expected = decodeMatrix(encodeMatrix(Float32Array.from([...a, ...b]), 5, 2, "sw3"));
  assert.deepEqual([...decoded], [...expected], "concatenated tensor decodes like the joined matrix");
  assert.throws(
    () => concatPackedTensors(pa, encodeMatrix(b, 2, 2, "sw4")),
    /layout mismatch/,
  );
  console.log("ok concatPackedTensors: row-wise concatenation, layout mismatch rejected");
}

// ---------------------------------------------------------------------------
// Executor under each novel mode on demo-10m
// ---------------------------------------------------------------------------

function fullAssignment(manifest: { config: { numHiddenLayers: number } }): ShardAssignment {
  return {
    workerId: "test",
    stage: 0,
    layerStart: 0,
    layerEnd: manifest.config.numHiddenLayers,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 0,
  };
}

const PROMPT_128 = Array.from({ length: 128 }, (_, i) => (i * 7 + 11) % 259);

async function prefillLogits(
  options: EngineOptions,
  prompt: number[],
): Promise<{ lastLogits: Float32Array; loadMs: number; memBytes: number }> {
  const loader = openWeightLoader(MODEL_DIR);
  const executor = createReferenceBackend().createStageExecutor(
    loader.manifest,
    fullAssignment(loader.manifest),
    options,
  );
  const t0 = performance.now();
  await executor.load(loader);
  const loadMs = performance.now() - t0;
  const cache = new BlockPagedKVCache(
    16,
    1024,
    loader.manifest.config.numKeyValueHeads,
    loader.manifest.config.headDim,
  );
  const embedded = executor.embed(prompt);
  const out = await executor.forward(embedded, {
    seqIds: prompt.map(() => "t"),
    positions: prompt.map((_, i) => i),
    isPrefill: true,
    cache,
  });
  const vocab = loader.manifest.config.vocabSize;
  const lastLogits = Float32Array.from(out.data.subarray((prompt.length - 1) * vocab, prompt.length * vocab));
  const memBytes = executor.memoryBytes();
  loader.close();
  return { lastLogits, loadMs, memBytes };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

const exact = await prefillLogits({}, PROMPT_128);
console.log(`  exact baseline: load ${exact.loadMs.toFixed(0)} ms, mem ${(exact.memBytes / 1024 / 1024).toFixed(1)} MB`);

// Landmark / sparse on random-weight keys: graceful degradation to exact
// attention (nothing approximable / skippable), so logits must match.
for (const mode of ["landmark", "sparse"] as const) {
  const r = await prefillLogits({ attention: mode }, PROMPT_128);
  const diff = maxAbsDiff(r.lastLogits, exact.lastLogits);
  assert.ok(diff < 1e-3, `${mode} prefill logits match exact (max|Δ| ${diff}), expected graceful degradation on incoherent keys`);
  assert.equal(r.memBytes, exact.memBytes, `${mode} holds the same weights as exact`);
  console.log(`ok attention=${mode}: prefill logits match exact (max|Δ| ${diff.toExponential(1)})`);
}

// SW-code weights: smaller memory, moved logits (measured, bounded).
const weightBounds: Array<{ mode: "sw2" | "sw3" | "sw4"; maxDiff: number }> = [
  { mode: "sw2", maxDiff: 3 },
  { mode: "sw3", maxDiff: 2 },
  { mode: "sw4", maxDiff: 1 },
];
for (const { mode, maxDiff } of weightBounds) {
  const r = await prefillLogits({ weightFormat: mode }, PROMPT_128);
  const diff = maxAbsDiff(r.lastLogits, exact.lastLogits);
  assert.ok(Number.isFinite(diff) && diff < maxDiff, `${mode} logits move by ${diff.toFixed(3)} (< ${maxDiff})`);
  assert.ok(r.memBytes < exact.memBytes / 4, `${mode} memory ${(r.memBytes / 1024 / 1024).toFixed(1)} MB < 1/4 of fp32`);
  for (const x of r.lastLogits) assert.ok(Number.isFinite(x), `${mode} logits are finite`);
  console.log(
    `ok weights=${mode}: mem ${(r.memBytes / 1024 / 1024).toFixed(1)} MB ` +
      `(vs ${(exact.memBytes / 1024 / 1024).toFixed(1)} MB fp32), max|Δlogit| ${diff.toFixed(3)}`,
  );
}

// SW-code KV: same shape of check (attention stays exact).
const kvBounds: Array<{ mode: "sw2" | "sw3" | "sw4"; maxDiff: number }> = [
  { mode: "sw2", maxDiff: 2 },
  { mode: "sw3", maxDiff: 1 },
  { mode: "sw4", maxDiff: 1 },
];
for (const { mode, maxDiff } of kvBounds) {
  const r = await prefillLogits({ kvFormat: mode }, PROMPT_128);
  const diff = maxAbsDiff(r.lastLogits, exact.lastLogits);
  assert.ok(Number.isFinite(diff) && diff < maxDiff, `kv=${mode} logits move by ${diff.toFixed(3)} (< ${maxDiff})`);
  console.log(`ok kv=${mode}: max|Δlogit| ${diff.toFixed(3)}`);
}

// Decode steps under landmark/sparse use the incremental stats cache:
// logits must still match exact step by step.
async function decodeLogits(options: EngineOptions, prompt: number[], steps: number): Promise<Float32Array[]> {
  const loader = openWeightLoader(MODEL_DIR);
  const executor = createReferenceBackend().createStageExecutor(
    loader.manifest,
    fullAssignment(loader.manifest),
    options,
  );
  await executor.load(loader);
  const cache = new BlockPagedKVCache(
    16,
    1024,
    loader.manifest.config.numKeyValueHeads,
    loader.manifest.config.headDim,
  );
  const vocab = loader.manifest.config.vocabSize;
  const embedded = executor.embed(prompt);
  let out = await executor.forward(embedded, {
    seqIds: prompt.map(() => "t"),
    positions: prompt.map((_, i) => i),
    isPrefill: true,
    cache,
  });
  const rows: Float32Array[] = [];
  let pending = out.data.subarray((prompt.length - 1) * vocab, prompt.length * vocab);
  for (let s = 0; s < steps; s++) {
    let best = 0;
    for (let i = 1; i < vocab; i++) if (pending[i] > pending[best]) best = i;
    const tok = best;
    const emb = executor.embed([tok]);
    out = await executor.forward(emb, {
      seqIds: ["t"],
      positions: [prompt.length + s],
      isPrefill: false,
      cache,
    });
    pending = out.data.subarray(0, vocab);
    rows.push(Float32Array.from(pending));
  }
  loader.close();
  return rows;
}
{
  const prompt = PROMPT_128.slice(0, 64);
  const exactRows = await decodeLogits({}, prompt, 4);
  for (const mode of ["landmark", "sparse"] as const) {
    const rows = await decodeLogits({ attention: mode }, prompt, 4);
    for (let s = 0; s < rows.length; s++) {
      const diff = maxAbsDiff(rows[s] as Float32Array, exactRows[s] as Float32Array);
      assert.ok(diff < 1e-3, `${mode} decode step ${s} matches exact (max|Δ| ${diff})`);
    }
    console.log(`ok attention=${mode}: 4 decode steps match exact (incremental stats cache)`);
  }
  // Quantized KV decode steps: bounded movement, finite logits.
  for (const mode of ["sw3", "sw4"] as const) {
    const rows = await decodeLogits({ kvFormat: mode }, prompt, 4);
    for (let s = 0; s < rows.length; s++) {
      const diff = maxAbsDiff(rows[s] as Float32Array, exactRows[s] as Float32Array);
      assert.ok(Number.isFinite(diff) && diff < 2, `kv=${mode} decode step ${s} moves by ${diff.toFixed(3)}`);
    }
    console.log(`ok kv=${mode}: 4 decode steps bounded vs exact`);
  }
}

// Incompatible combination is a loud error, not a silent fallback.
{
  const loader = openWeightLoader(MODEL_DIR);
  assert.throws(
    () =>
      createReferenceBackend().createStageExecutor(loader.manifest, fullAssignment(loader.manifest), {
        attention: "landmark",
        kvFormat: "sw4",
      }),
    /cannot yet combine with quantized KV/,
  );
  loader.close();
  console.log("ok landmark + quantized KV is rejected loudly");
}

console.log("All engine-options tests passed.");
