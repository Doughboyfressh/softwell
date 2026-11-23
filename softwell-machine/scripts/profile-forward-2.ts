/**
 * profile-forward-2.ts — second-round candidates: fused SwiGLU (gate+up share
 * the x pass), fused single-array GQA, and u4-swiglu. Interleaved medians.
 * Usage: node scripts/profile-forward-2.ts
 */
import {
  groupedQueryAttention, matmulTransposeB, silu, swiglu,
} from "../src/engine/attention.ts";

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

function matmulTB_u4(a: Float32Array, aRows: number, aCols: number, b: Float32Array, bRows: number, bCols: number): Float32Array {
  const c = new Float32Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols, cRow = i * bRows;
    for (let j = 0; j < bRows; j++) {
      const bRow = j * bCols;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
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

/** Fused SwiGLU: one pass over intermediate j computing gate_j and up_j
 *  with shared x loads, then the down projection (u4). */
function swigluFused(x: Float32Array, gateW: Float32Array, upW: Float32Array, downW: Float32Array): Float32Array {
  const hiddenSize = x.length;
  const inter = gateW.length / hiddenSize;
  const hidden = new Float32Array(inter);
  for (let j = 0; j < inter; j++) {
    const gRow = j * hiddenSize, uRow = j * hiddenSize;
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
  return matmulTB_u4(hidden, 1, inter, downW, hiddenSize, inter);
}
/** u4-only SwiGLU: same structure as current swiglu but with u4 matmuls. */
function swigluU4(x: Float32Array, gateW: Float32Array, upW: Float32Array, downW: Float32Array): Float32Array {
  const hiddenSize = x.length;
  const inter = gateW.length / hiddenSize;
  const gate = matmulTB_u4(x, 1, hiddenSize, gateW, inter, hiddenSize);
  const up = matmulTB_u4(x, 1, hiddenSize, upW, inter, hiddenSize);
  const hidden = new Float32Array(inter);
  for (let i = 0; i < inter; i++) hidden[i] = silu(gate[i]) * up[i];
  return matmulTB_u4(hidden, 1, inter, downW, hiddenSize, inter);
}

/** GQA with a single scores array (softmax in place, no weights array). */
function gqaFused(q: Float32Array, k: Float32Array, v: Float32Array, numHeads: number, numKvHeads: number, headDim: number, kvLen: number): Float32Array {
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
      let dot = 0;
      for (let d = 0; d < headDim; d++) dot += q[qOff + d] * k[kOff + d];
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

console.log(`loadavg ${JSON.stringify((await import("node:os")).loadavg())}`);
{
  const x = randArr(576, 61);
  const gate = randArr(1536 * 576, 62), up = randArr(1536 * 576, 63), down = randArr(576 * 1536, 64);
  const ref = swiglu(x, gate, up, down);
  console.log(`swiglu fused maxdiff=${maxAbsDiff(ref, swigluFused(x, gate, up, down)).toExponential(1)} u4 maxdiff=${maxAbsDiff(ref, swigluU4(x, gate, up, down)).toExponential(1)}`);
  const t = await timeAB({
    cur: () => swiglu(x, gate, up, down),
    u4: () => swigluU4(x, gate, up, down),
    fused: () => swigluFused(x, gate, up, down),
  }, 15);
  console.log(`swiglu: ${Object.entries(t).map(([k, v]) => `${k}=${v.toFixed(2)}ms`).join(" ")} ratios vs cur: ${Object.entries(t).map(([k, v]) => `${k}=${(t.cur! / v).toFixed(2)}x`).join(" ")}`);
}
{
  const q = randArr(576, 71); const kk = randArr(21 * 192, 72); const vv = randArr(21 * 192, 73);
  const pos = Array.from({ length: 21 }, (_, i) => i);
  const ref = groupedQueryAttention(q, kk, vv, 9, 3, 64, { causal: true, queryPosition: 20, kvPositions: pos });
  console.log(`gqa fused maxdiff=${maxAbsDiff(ref, gqaFused(q, kk, vv, 9, 3, 64, 21)).toExponential(1)}`);
  const t = await timeAB({
    cur: () => groupedQueryAttention(q, kk, vv, 9, 3, 64, { causal: true, queryPosition: 20, kvPositions: pos }),
    fused: () => gqaFused(q, kk, vv, 9, 3, 64, 21),
  }, 300);
  console.log(`GQA kvLen=21: ${Object.entries(t).map(([k, v]) => `${k}=${(v * 1000).toFixed(1)}us`).join(" ")} ratio=${(t.cur! / t.fused!).toFixed(2)}x`);
  // long context
  const kk2 = randArr(512 * 192, 74); const vv2 = randArr(512 * 192, 75);
  const pos2 = Array.from({ length: 512 }, (_, i) => i);
  const t2 = await timeAB({
    cur: () => groupedQueryAttention(q, kk2, vv2, 9, 3, 64, { causal: true, queryPosition: 511, kvPositions: pos2 }),
    fused: () => gqaFused(q, kk2, vv2, 9, 3, 64, 512),
  }, 100);
  console.log(`GQA kvLen=512: ${Object.entries(t2).map(([k, v]) => `${k}=${(v * 1000).toFixed(1)}us`).join(" ")} ratio=${(t2.cur! / t2.fused!).toFixed(2)}x`);
}
console.log("profile2 done");
