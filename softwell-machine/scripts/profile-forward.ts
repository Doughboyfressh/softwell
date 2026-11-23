/**
 * profile-forward.ts — scratch profiler for the forward-pass optimization piece.
 * Microbenchmarks the engine kernels at SmolLM-135M decode/prefill shapes,
 * compares candidate matmul kernels against the current matmulTransposeB,
 * and times a full in-process autoregressive run. Interleaved medians so
 * host load drift cancels in the ratios. Not part of the shipped bench
 * protocol; results feed docs/FORWARD-OPT.md.
 * Usage: node scripts/profile-forward.ts
 */
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import {
  applyRope, groupedQueryAttention, matmulTransposeB, rmsNorm, swiglu,
} from "../src/engine/attention.ts";
import { AutoregressiveDecoder, forwardTokenRows } from "../src/engine/novel-decoding.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import type { ShardAssignment } from "../src/contracts.ts";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; };
}
function randArr(n: number, seed: number): Float32Array {
  const r = rng(seed); const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = r();
  return a;
}
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
async function timeIt(fn: () => unknown, reps: number, warmup = 2): Promise<number> {
  for (let i = 0; i < warmup; i++) fn();
  const ts: number[] = [];
  for (let i = 0; i < reps; i++) { const t0 = performance.now(); fn(); ts.push(performance.now() - t0); }
  return median(ts);
}
// Interleaved A/B: run fns in rotation, return median ms per fn.
async function timeAB(fns: Record<string, () => unknown>, rounds: number): Promise<Record<string, number>> {
  const acc: Record<string, number[]> = {};
  for (const k of Object.keys(fns)) acc[k] = [];
  for (const k of Object.keys(fns)) fns[k]!(); // warmup each once
  for (let r = 0; r < rounds; r++) {
    for (const [k, fn] of Object.entries(fns)) {
      const t0 = performance.now(); fn(); acc[k]!.push(performance.now() - t0);
    }
  }
  const out: Record<string, number> = {};
  for (const k of Object.keys(fns)) out[k] = median(acc[k]!);
  return out;
}

// ---- Candidate kernels (defined here first; the winner moves into attention.ts) ----

/** B: dot product unrolled by 4 with independent accumulators. */
function matmulTB_u4(a: Float32Array, aRows: number, aCols: number, b: Float32Array, bRows: number, bCols: number): Float32Array {
  const c = new Float32Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols, cRow = i * bRows;
    for (let j = 0; j < bRows; j++) {
      const bRow = j * bCols;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      let k = 0;
      const lim = aCols - (aCols % 4);
      for (; k < lim; k += 4) {
        s0 += a[aRow + k] * b[bRow + k];
        s1 += a[aRow + k + 1] * b[bRow + k + 1];
        s2 += a[aRow + k + 2] * b[bRow + k + 2];
        s3 += a[aRow + k + 3] * b[bRow + k + 3];
      }
      let s = (s0 + s1) + (s2 + s3);
      for (; k < aCols; k++) s += a[aRow + k] * b[bRow + k];
      c[cRow + j] = s;
    }
  }
  return c;
}
/** B8: dot product unrolled by 8. */
function matmulTB_u8(a: Float32Array, aRows: number, aCols: number, b: Float32Array, bRows: number, bCols: number): Float32Array {
  const c = new Float32Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols, cRow = i * bRows;
    for (let j = 0; j < bRows; j++) {
      const bRow = j * bCols;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
      let k = 0;
      const lim = aCols - (aCols % 8);
      for (; k < lim; k += 8) {
        s0 += a[aRow + k] * b[bRow + k];
        s1 += a[aRow + k + 1] * b[bRow + k + 1];
        s2 += a[aRow + k + 2] * b[bRow + k + 2];
        s3 += a[aRow + k + 3] * b[bRow + k + 3];
        s4 += a[aRow + k + 4] * b[bRow + k + 4];
        s5 += a[aRow + k + 5] * b[bRow + k + 5];
        s6 += a[aRow + k + 6] * b[bRow + k + 6];
        s7 += a[aRow + k + 7] * b[bRow + k + 7];
      }
      let s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
      for (; k < aCols; k++) s += a[aRow + k] * b[bRow + k];
      c[cRow + j] = s;
    }
  }
  return c;
}
/** C: pre-transposed weights Wt [in, out] row-major; C = A @ Wt, i,k,j loop. */
function transposeWeight(b: Float32Array, rows: number, cols: number): Float32Array {
  const t = new Float32Array(rows * cols);
  for (let j = 0; j < rows; j++) for (let k = 0; k < cols; k++) t[k * rows + j] = b[j * cols + k];
  return t;
}
function matmulT(a: Float32Array, aRows: number, aCols: number, wt: Float32Array, outCols: number): Float32Array {
  const c = new Float32Array(aRows * outCols);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols, cRow = i * outCols;
    for (let k = 0; k < aCols; k++) {
      const aik = a[aRow + k];
      if (aik === 0) continue;
      const wRow = k * outCols;
      for (let j = 0; j < outCols; j++) c[cRow + j] += aik * wt[wRow + j];
    }
  }
  return c;
}
/** C8: transposed weights, j unrolled by 8. */
function matmulT_u8(a: Float32Array, aRows: number, aCols: number, wt: Float32Array, outCols: number): Float32Array {
  const c = new Float32Array(aRows * outCols);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols, cRow = i * outCols;
    for (let k = 0; k < aCols; k++) {
      const aik = a[aRow + k];
      const wRow = k * outCols;
      let j = 0;
      const lim = outCols - (outCols % 8);
      for (; j < lim; j += 8) {
        c[cRow + j] += aik * wt[wRow + j];
        c[cRow + j + 1] += aik * wt[wRow + j + 1];
        c[cRow + j + 2] += aik * wt[wRow + j + 2];
        c[cRow + j + 3] += aik * wt[wRow + j + 3];
        c[cRow + j + 4] += aik * wt[wRow + j + 4];
        c[cRow + j + 5] += aik * wt[wRow + j + 5];
        c[cRow + j + 6] += aik * wt[wRow + j + 6];
        c[cRow + j + 7] += aik * wt[wRow + j + 7];
      }
      for (; j < outCols; j++) c[cRow + j] += aik * wt[wRow + j];
    }
  }
  return c;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

console.log(`host loadavg ${JSON.stringify((await import("node:os")).loadavg())} cpus ${(await import("node:os")).cpus().length}`);

// ---- Kernel microbenchmarks ----
const shapes = [
  { name: "q/o  [1,576]x[576,576]", m: 1, k: 576, n: 576 },
  { name: "gate [1,576]x[1536,576]", m: 1, k: 576, n: 1536 },
  { name: "down [1,1536]x[576,1536]", m: 1, k: 1536, n: 576 },
  { name: "lmhd [1,576]x[49152,576]", m: 1, k: 576, n: 49152 },
  { name: "prefill gate [5,576]x[1536,576]", m: 5, k: 576, n: 1536 },
  { name: "prefill gate [16,576]x[1536,576]", m: 16, k: 576, n: 1536 },
];
for (const s of shapes) {
  const a = randArr(s.m * s.k, 11);
  const b = randArr(s.n * s.k, 22);
  const wt = transposeWeight(b, s.n, s.k);
  const ref = matmulTransposeB(a, s.m, s.k, b, s.n, s.k);
  const d4 = maxAbsDiff(ref, matmulTB_u4(a, s.m, s.k, b, s.n, s.k));
  const d8 = maxAbsDiff(ref, matmulTB_u8(a, s.m, s.k, b, s.n, s.k));
  const dt = maxAbsDiff(ref, matmulT(a, s.m, s.k, wt, s.n));
  const dt8 = maxAbsDiff(ref, matmulT_u8(a, s.m, s.k, wt, s.n));
  console.log(`${s.name}: maxdiff u4=${d4.toExponential(1)} u8=${d8.toExponential(1)} T=${dt.toExponential(1)} T_u8=${dt8.toExponential(1)}`);
  const reps = s.n > 10000 ? 3 : 7;
  const t = await timeAB({
    cur: () => matmulTransposeB(a, s.m, s.k, b, s.n, s.k),
    u4: () => matmulTB_u4(a, s.m, s.k, b, s.n, s.k),
    u8: () => matmulTB_u8(a, s.m, s.k, b, s.n, s.k),
    T: () => matmulT(a, s.m, s.k, wt, s.n),
    T_u8: () => matmulT_u8(a, s.m, s.k, wt, s.n),
  }, reps);
  const gflops = (2 * s.m * s.k * s.n) / 1e9;
  const line = Object.entries(t).map(([k, v]) => `${k}=${v.toFixed(2)}ms(${(gflops / (v / 1000)).toFixed(2)} GFLOP/s)`).join(" ");
  console.log(`  ${line}`);
  console.log(`  ratios vs cur: ${Object.entries(t).map(([k, v]) => `${k}=${(t.cur! / v).toFixed(2)}x`).join(" ")}`);
}

// ---- RoPE: current vs cached cos/sin table ----
{
  const q = randArr(576, 31); const k = randArr(192, 32);
  const tCur = await timeIt(() => { const qq = q.slice(), kk = k.slice(); applyRope(qq, kk, [100], 9, 64, 10000, 3); }, 200);
  // cached-table candidate
  const half = 32;
  const table = new Map<number, { cos: Float32Array; sin: Float32Array }>();
  function ropeTable(pos: number): { cos: Float32Array; sin: Float32Array } {
    let e = table.get(pos);
    if (!e) {
      const cos = new Float32Array(half), sin = new Float32Array(half);
      for (let i = 0; i < half; i++) { const f = Math.pow(10000, -2 * i / 64); cos[i] = Math.cos(pos * f); sin[i] = Math.sin(pos * f); }
      e = { cos, sin }; table.set(pos, e);
    }
    return e;
  }
  function applyRopeCached(q: Float32Array, k: Float32Array, pos: number): void {
    const { cos, sin } = ropeTable(pos);
    const rot = (x: Float32Array, heads: number) => {
      for (let h = 0; h < heads; h++) for (let i = 0; i < half; i++) {
        const base = h * 64 + 2 * i, c = cos[i], s = sin[i];
        const x0 = x[base], x1 = x[base + 1];
        x[base] = x0 * c - x1 * s; x[base + 1] = x0 * s + x1 * c;
      }
    };
    rot(q, 9); rot(k, 3);
  }
  const tCached = await timeIt(() => { const qq = q.slice(), kk = k.slice(); applyRopeCached(qq, kk, 100); }, 200);
  // correctness
  const q1 = q.slice(), k1 = k.slice(); applyRope(q1, k1, [100], 9, 64, 10000, 3);
  const q2 = q.slice(), k2 = k.slice(); applyRopeCached(q2, k2, 100);
  console.log(`rope 1 row/layer: cur=${(tCur * 1000).toFixed(1)}us cached=${(tCached * 1000).toFixed(1)}us ratio=${(tCur / tCached).toFixed(2)}x maxdiff=${maxAbsDiff(q1, q2).toExponential(1)}; per-token x30 layers: cur=${(tCur * 30).toFixed(2)}ms cached=${(tCached * 30).toFixed(2)}ms`);
}

// ---- rmsNorm / GQA / swiglu at decode shapes ----
{
  const x = randArr(576, 41); const w = randArr(576, 42);
  const tNorm = await timeIt(() => rmsNorm(x, w, 1e-5), 2000);
  console.log(`rmsNorm[576]: ${(tNorm * 1000).toFixed(1)}us/call; x60/layer-token = ${(tNorm * 60 * 1000).toFixed(0)}us/token`);
  const q = randArr(576, 43); const kk = randArr(21 * 192, 44); const vv = randArr(21 * 192, 45);
  const pos = Array.from({ length: 21 }, (_, i) => i);
  const tAttn = await timeIt(() => groupedQueryAttention(q, kk, vv, 9, 3, 64, { causal: true, queryPosition: 20, kvPositions: pos }), 500);
  console.log(`GQA kvLen=21: ${(tAttn * 1000).toFixed(1)}us/call; x30 layers = ${(tAttn * 30).toFixed(2)}ms/token`);
  const gate = randArr(1536 * 576, 46); const up = randArr(1536 * 576, 47); const down = randArr(576 * 1536, 48);
  const tSwi = await timeIt(() => swiglu(x, gate, up, down), 100);
  console.log(`swiglu[576->1536]: ${tSwi.toFixed(2)}ms/call; x30 = ${(tSwi * 30).toFixed(1)}ms/token`);
}

// ---- KV cache read cost vs prefix length (host BlockPagedKVCache) ----
{
  const cache = new BlockPagedKVCache(16, 4096, 3, 64);
  const kv = randArr(192, 51);
  for (let p = 0; p < 200; p++) await cache.append("s", 0, p, kv, kv);
  for (const n of [5, 21, 100, 200]) {
    const prefix = Array.from({ length: n }, (_, i) => i);
    const t = await timeIt(() => cache.read("s", 0, prefix), 200);
    console.log(`cache.read prefix=${n}: ${(t * 1000).toFixed(1)}us/call; x30 layers = ${(t * 30).toFixed(2)}ms/token`);
  }
}

// ---- Full in-process forward: SmolLM baseline (same protocol as bench-spec-draft) ----
{
  const TARGET_DIR = "./models/smollm-135m";
  const backend = createReferenceBackend();
  const manifest = loadManifest(TARGET_DIR, "smollm-135m");
  const tokenizer = loadTokenizer(TARGET_DIR);
  const loader = openWeightLoader(TARGET_DIR, "smollm-135m");
  const cfg = manifest.config;
  const assignment: ShardAssignment = {
    workerId: "prof", stage: 0, layerStart: 0, layerEnd: cfg.numHiddenLayers,
    hasEmbedding: true, hasLmHead: true, tensorParallelRank: 0, tensorParallelSize: 1,
    experts: [], estimatedMemoryBytes: 0,
  };
  const executor = backend.createStageExecutor(manifest, assignment);
  const tL0 = performance.now();
  await executor.load(loader);
  console.log(`target loaded in ${((performance.now() - tL0) / 1000).toFixed(1)}s`);
  const prompt = "The capital of France is";
  const promptIds = tokenizer.encode(prompt);
  const sampling = { temperature: 0, topP: 1, topK: 0, maxTokens: 16, seed: 1 };
  // Prefill timing (single rep) + decode per-forward timings
  const cache = new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
  const tP0 = performance.now();
  const pre = await forwardTokenRows(executor, cache, "prof", promptIds, promptIds.map((_, i) => i), true);
  void pre;
  console.log(`prefill ${promptIds.length} tokens: ${(performance.now() - tP0).toFixed(0)}ms`);
  const decMs: number[] = [];
  let tok = promptIds[promptIds.length - 1] as number;
  for (let i = 0; i < 16; i++) {
    const t0 = performance.now();
    const out = await forwardTokenRows(executor, cache, "prof", [tok], [promptIds.length + i], false);
    decMs.push(performance.now() - t0);
    const logits = out.data.subarray(0, cfg.vocabSize);
    let best = 0; for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    tok = best;
  }
  console.log(`decode forwards ms: [${decMs.map((v) => v.toFixed(0)).join(", ")}] median=${median(decMs).toFixed(1)}ms -> ${(1000 / median(decMs)).toFixed(2)} tok/s (decode-only)`);
  // Full decoder run for the headline number (1 rep, same as spec bench baseline shape)
  const cache2 = new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
  const dec = new AutoregressiveDecoder({ executor, cache: cache2, seqId: "prof2", promptTokenIds: promptIds, sampling, vocabSize: cfg.vocabSize, eosTokenId: cfg.eosTokenId });
  const t0 = performance.now();
  const res = await dec.generate(16);
  const ms = performance.now() - t0;
  console.log(`full generate 16 tok: ${ms.toFixed(0)}ms = ${(res.tokens.length / ms * 1000).toFixed(2)} tok/s, text=${JSON.stringify(tokenizer.decode(res.tokens))}`);
}
console.log("profile done");
