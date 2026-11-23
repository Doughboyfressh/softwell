/**
 * profile-forward-3.ts — round-3 candidates for the forward-pass program.
 * Tiled fused SwiGLU, GQA with u4 dot + in-place softmax, and KV cache.read
 * cost at long prefixes (997). Interleaved medians. Scratch profiler.
 * Usage: node scripts/profile-forward-3.ts
 */
import {
  groupedQueryAttention, matmulTransposeB, silu, swiglu,
} from "../src/engine/attention.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; };
}
function randArr(n: number, seed: number): Float32Array {
  const r = rng(seed); const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = r();
  return a;
}
function median(xs: number[]): number { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }
async function timeAB(fns: Record<string, () => unknown>, rounds: number): Promise<Record<string, number>> {
  const acc: Record<string, number[]> = {};
  for (const k of Object.keys(fns)) acc[k] = [];
  for (const k of Object.keys(fns)) fns[k]!();
  for (let r = 0; r < rounds; r++) for (const [k, fn] of Object.entries(fns)) {
    const t0 = performance.now(); fn(); acc[k]!.push(performance.now() - t0);
  }
  const out: Record<string, number> = {};
  for (const k of Object.keys(fns)) out[k] = median(acc[k]!);
  return out;
}
function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m;
}

console.log(`loadavg ${JSON.stringify((await import("node:os")).loadavg())}`);

// ---- Tiled fused SwiGLU: per tile of J intermediate units, compute
// gate/up, apply silu*up, accumulate the down contribution immediately.
// Never materializes the full hidden vector or the gate/up arrays. ----
function swigluTiled(x: Float32Array, gateW: Float32Array, upW: Float32Array, downW: Float32Array, tile: number): Float32Array {
  const hiddenSize = x.length;
  const inter = gateW.length / hiddenSize;
  const out = new Float32Array(hiddenSize);
  const hidden = new Float32Array(tile);
  for (let j0 = 0; j0 < inter; j0 += tile) {
    const jt = Math.min(tile, inter - j0);
    for (let j = 0; j < jt; j++) {
      const gj = j0 + j;
      const gRow = gj * hiddenSize, uRow = gj * hiddenSize;
      let g0 = 0, g1 = 0, g2 = 0, g3 = 0, u0 = 0, u1 = 0, u2 = 0, u3 = 0, k = 0;
      const lim = hiddenSize - (hiddenSize % 4);
      for (; k < lim; k += 4) {
        const x0 = x[k], x1 = x[k + 1], x2 = x[k + 2], x3 = x[k + 3];
        g0 += x0 * gateW[gRow + k]; g1 += x1 * gateW[gRow + k + 1];
        g2 += x2 * gateW[gRow + k + 2]; g3 += x3 * gateW[gRow + k + 3];
        u0 += x0 * upW[uRow + k]; u1 += x1 * upW[uRow + k + 1];
        u2 += x2 * upW[uRow + k + 2]; u3 += x3 * upW[uRow + k + 3];
      }
      let g = (g0 + g1) + (g2 + g3), u = (u0 + u1) + (u2 + u3);
      for (; k < hiddenSize; k++) { g += x[k] * gateW[gRow + k]; u += x[k] * upW[uRow + k]; }
      hidden[j] = silu(g) * u;
    }
    // down tile: out += hidden_tile @ downW[j0:j0+jt]^T, u4 over k=jt
    for (let i = 0; i < hiddenSize; i++) {
      const dRow = i * inter + j0;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
      const lim = jt - (jt % 4);
      for (; k < lim; k += 4) {
        s0 += hidden[k] * downW[dRow + k];
        s1 += hidden[k + 1] * downW[dRow + k + 1];
        s2 += hidden[k + 2] * downW[dRow + k + 2];
        s3 += hidden[k + 3] * downW[dRow + k + 3];
      }
      let s = (s0 + s1) + (s2 + s3);
      for (; k < jt; k++) s += hidden[k] * downW[dRow + k];
      out[i] += s;
    }
  }
  return out;
}

{
  const x = randArr(576, 61);
  const gate = randArr(1536 * 576, 62), up = randArr(1536 * 576, 63), down = randArr(576 * 1536, 64);
  const ref = swiglu(x, gate, up, down);
  for (const tile of [128, 256, 512, 1536]) {
    console.log(`swiglu tiled-${tile} maxdiff=${maxAbsDiff(ref, swigluTiled(x, gate, up, down, tile)).toExponential(1)}`);
  }
  const t = await timeAB({
    cur: () => swiglu(x, gate, up, down),
    t128: () => swigluTiled(x, gate, up, down, 128),
    t256: () => swigluTiled(x, gate, up, down, 256),
    t512: () => swigluTiled(x, gate, up, down, 512),
    tFull: () => swigluTiled(x, gate, up, down, 1536),
  }, 15);
  console.log(`swiglu: ${Object.entries(t).map(([k, v]) => `${k}=${v.toFixed(2)}ms`).join(" ")} ratios vs cur: ${Object.entries(t).map(([k, v]) => `${k}=${(t.cur! / v).toFixed(2)}x`).join(" ")}`);
}

// ---- GQA: fused scores + u4 dot product ----
function gqaU4(q: Float32Array, k: Float32Array, v: Float32Array, numHeads: number, numKvHeads: number, headDim: number, kvLen: number): Float32Array {
  const groupSize = numHeads / numKvHeads;
  const scale = 1 / Math.sqrt(headDim);
  const out = new Float32Array(numHeads * headDim);
  const scores = new Float32Array(kvLen);
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const qOff = h * headDim;
    let max = -Infinity;
    for (let t = 0; t < kvLen; t++) {
      const kOff = (t * numKvHeads + kvHead) * headDim;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, d = 0;
      const lim = headDim - (headDim % 4);
      for (; d < lim; d += 4) {
        s0 += q[qOff + d] * k[kOff + d];
        s1 += q[qOff + d + 1] * k[kOff + d + 1];
        s2 += q[qOff + d + 2] * k[kOff + d + 2];
        s3 += q[qOff + d + 3] * k[kOff + d + 3];
      }
      let dot = (s0 + s1) + (s2 + s3);
      for (; d < headDim; d++) dot += q[qOff + d] * k[kOff + d];
      scores[t] = dot * scale;
      if (scores[t] > max) max = scores[t];
    }
    let sum = 0;
    for (let t = 0; t < kvLen; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
    const inv = 1 / sum;
    const oOff = h * headDim;
    for (let t = 0; t < kvLen; t++) {
      const w = scores[t] * inv;
      if (w === 0) continue;
      const vOff = (t * numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) out[oOff + d] += w * v[vOff + d];
    }
  }
  return out;
}

for (const kvLen of [21, 512, 997]) {
  const q = randArr(576, 71);
  const kk = randArr(kvLen * 192, 72); const vv = randArr(kvLen * 192, 73);
  const pos = Array.from({ length: kvLen }, (_, i) => i);
  const ref = groupedQueryAttention(q, kk, vv, 9, 3, 64, { causal: true, queryPosition: kvLen - 1, kvPositions: pos });
  console.log(`gqa u4 kvLen=${kvLen} maxdiff=${maxAbsDiff(ref, gqaU4(q, kk, vv, 9, 3, 64, kvLen)).toExponential(1)}`);
  const t = await timeAB({
    cur: () => groupedQueryAttention(q, kk, vv, 9, 3, 64, { causal: true, queryPosition: kvLen - 1, kvPositions: pos }),
    u4: () => gqaU4(q, kk, vv, 9, 3, 64, kvLen),
  }, kvLen > 100 ? 30 : 200);
  console.log(`GQA kvLen=${kvLen}: ${Object.entries(t).map(([k, v]) => `${k}=${(v * 1000).toFixed(1)}us`).join(" ")} ratio u4=${(t.cur! / t.u4!).toFixed(2)}x; x30 layers: cur=${(t.cur! * 30).toFixed(1)}ms u4=${(t.u4! * 30).toFixed(1)}ms per token`);
}

// ---- KV cache.read at long prefixes ----
{
  const cache = new BlockPagedKVCache(16, 4096, 3, 64);
  const kv = randArr(192, 51);
  for (let p = 0; p < 1000; p++) await cache.append("s", 0, p, kv, kv);
  for (const n of [100, 200, 500, 997]) {
    const prefix = Array.from({ length: n }, (_, i) => i);
    const ts: number[] = [];
    for (let r = 0; r < 30; r++) { const t0 = performance.now(); await cache.read("s", 0, prefix); ts.push(performance.now() - t0); }
    const med = median(ts);
    console.log(`cache.read prefix=${n}: ${med.toFixed(3)}ms/call; x30 layers = ${(med * 30).toFixed(1)}ms/token; bytes copied/call=${(n * 192 * 4 * 2 / 1024).toFixed(0)}KiB`);
  }
  await cache.close();
}
console.log("profile3 done");
