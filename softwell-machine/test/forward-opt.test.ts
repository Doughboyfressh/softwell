/**
 * forward-opt.test.ts — tests for the forward-pass optimization piece.
 *
 * Core change: matmulTransposeB uses a 4-way unrolled dot product with
 * independent accumulators. These tests check it against a naive serial
 * reference at the real SmolLM-135M shapes (including the lm_head shape
 * and aCols values that are not multiples of 4, which exercise the
 * remainder loop) and check the shape-mismatch guard still throws.
 * Run with: node --test test/forward-opt.test.ts
 */
import assert from "node:assert/strict";
import test from "node:test";
import {
  applyRope, groupedQueryAttention, matmulTransposeB, silu, swiglu,
} from "../src/engine/attention.ts";

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}
function randArr(n: number, seed: number): Float32Array {
  const r = rng(seed);
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = r();
  return a;
}
/** Naive serial reference: C = A @ B^T, one accumulator, no unrolling. */
function reference(a: Float32Array, aRows: number, aCols: number, b: Float32Array, bRows: number): Float32Array {
  const c = new Float32Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bRows; j++) {
      let sum = 0;
      for (let k = 0; k < aCols; k++) sum += a[i * aCols + k] * b[j * aCols + k];
      c[i * bRows + j] = sum;
    }
  }
  return c;
}
function assertClose(actual: Float32Array, expected: Float32Array, tol: number): void {
  assert.equal(actual.length, expected.length);
  let maxDiff = 0;
  for (let i = 0; i < actual.length; i++) maxDiff = Math.max(maxDiff, Math.abs(actual[i] - expected[i]));
  assert.ok(maxDiff <= tol, `max abs diff ${maxDiff} exceeds tol ${tol}`);
}

const shapes: { m: number; k: number; n: number; label: string }[] = [
  { m: 1, k: 576, n: 576, label: "q/o [1,576]x[576,576]" },
  { m: 1, k: 576, n: 1536, label: "gate/up [1,576]x[1536,576]" },
  { m: 1, k: 1536, n: 576, label: "down [1,1536]x[576,1536]" },
  { m: 1, k: 576, n: 49152, label: "lm_head [1,576]x[49152,576]" },
  { m: 5, k: 576, n: 1536, label: "prefill gate [5,576]x[1536,576]" },
  { m: 3, k: 7, n: 5, label: "odd k=7 remainder path [3,7]x[5,7]" },
  { m: 2, k: 1, n: 4, label: "k=1, below unroll width [2,1]x[4,1]" },
  { m: 2, k: 3, n: 4, label: "k=3, below unroll width [2,3]x[4,3]" },
];

for (const s of shapes) {
  test(`matmulTransposeB matches serial reference: ${s.label}`, () => {
    const a = randArr(s.m * s.k, 11);
    const b = randArr(s.n * s.k, 22);
    const got = matmulTransposeB(a, s.m, s.k, b, s.n, s.k);
    const want = reference(a, s.m, s.k, b, s.n);
    assertClose(got, want, 1e-5);
  });
}

test("matmulTransposeB is deterministic across calls", () => {
  const a = randArr(576, 31);
  const b = randArr(1536 * 576, 32);
  const r1 = matmulTransposeB(a, 1, 576, b, 1536, 576);
  const r2 = matmulTransposeB(a, 1, 576, b, 1536, 576);
  assert.deepEqual(r1, r2);
});

test("matmulTransposeB throws on shape mismatch", () => {
  const a = randArr(10, 41);
  const b = randArr(10, 42);
  assert.throws(() => matmulTransposeB(a, 1, 10, b, 2, 5), /shape mismatch/);
});

// ---- Round 2: fused tiled SwiGLU, u4 GQA, cached RoPE ----

/** Naive serial SwiGLU reference (three separate serial matmuls). */
function swigluReference(x: Float32Array, gateW: Float32Array, upW: Float32Array, downW: Float32Array): Float32Array {
  const hiddenSize = x.length;
  const inter = gateW.length / hiddenSize;
  const dot = (w: Float32Array, row: number): number => {
    let s = 0;
    for (let k = 0; k < hiddenSize; k++) s += x[k] * w[row * hiddenSize + k];
    return s;
  };
  const hidden = new Float32Array(inter);
  for (let j = 0; j < inter; j++) hidden[j] = silu(dot(gateW, j)) * dot(upW, j);
  const out = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    let s = 0;
    for (let k = 0; k < inter; k++) s += hidden[k] * downW[i * inter + k];
    out[i] = s;
  }
  return out;
}

for (const s of [
  { hidden: 576, inter: 1536, label: "SmolLM [576->1536]" },
  { hidden: 5, inter: 7, label: "odd sizes, below tile and unroll widths [5->7]" },
  { hidden: 16, inter: 300, label: "intermediate not a multiple of the 256 tile [16->300]" },
]) {
  test(`swiglu (fused tiled) matches serial reference: ${s.label}`, () => {
    const x = randArr(s.hidden, 51);
    const gate = randArr(s.inter * s.hidden, 52);
    const up = randArr(s.inter * s.hidden, 53);
    const down = randArr(s.hidden * s.inter, 54);
    const got = swiglu(x, gate, up, down);
    const want = swigluReference(x, gate, up, down);
    // Tile-partial down sums re-associate the accumulation, so the
    // tolerance is looser than the matmul tests (measured ~8e-6).
    assertClose(got, want, 2e-5);
  });
}

test("swiglu throws on shape mismatch", () => {
  const x = randArr(8, 61);
  assert.throws(() => swiglu(x, randArr(16, 62), randArr(24, 63), randArr(16, 64)), /shape mismatch/);
  assert.throws(() => swiglu(x, randArr(16, 62), randArr(16, 63), randArr(24, 64)), /shape mismatch/);
  assert.throws(() => swiglu(x, randArr(7, 62), randArr(7, 63), randArr(56, 64)), /not divisible/);
});

/** Naive serial GQA reference (one accumulator, separate weights array). */
function gqaReference(
  q: Float32Array, k: Float32Array, v: Float32Array,
  numHeads: number, numKvHeads: number, headDim: number,
  causal: boolean, queryPosition: number, kvPositions: number[],
): Float32Array {
  const kvLen = k.length / (numKvHeads * headDim);
  const groupSize = numHeads / numKvHeads;
  const scale = 1 / Math.sqrt(headDim);
  const out = new Float32Array(numHeads * headDim);
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const scores = new Float32Array(kvLen);
    let max = -Infinity;
    for (let t = 0; t < kvLen; t++) {
      if (causal && kvPositions[t] > queryPosition) { scores[t] = -Infinity; continue; }
      let dot = 0;
      for (let d = 0; d < headDim; d++) dot += q[h * headDim + d] * k[(t * numKvHeads + kvHead) * headDim + d];
      scores[t] = dot * scale;
      if (scores[t] > max) max = scores[t];
    }
    const weights = new Float32Array(kvLen);
    let sum = 0;
    for (let t = 0; t < kvLen; t++) {
      weights[t] = scores[t] === -Infinity ? 0 : Math.exp(scores[t] - max);
      sum += weights[t];
    }
    if (sum === 0) continue;
    for (let t = 0; t < kvLen; t++) {
      const w = weights[t] / sum;
      if (w === 0) continue;
      for (let d = 0; d < headDim; d++) out[h * headDim + d] += w * v[(t * numKvHeads + kvHead) * headDim + d];
    }
  }
  return out;
}

for (const s of [
  { kvLen: 21, queryPos: 20, label: "kvLen 21, causal, query at end" },
  { kvLen: 5, queryPos: 2, label: "kvLen 5, causal, query mid-prefix (masked tail)" },
  { kvLen: 1, queryPos: 0, label: "kvLen 1" },
]) {
  test(`groupedQueryAttention matches serial reference: ${s.label}`, () => {
    const q = randArr(576, 71);
    const k = randArr(s.kvLen * 192, 72);
    const v = randArr(s.kvLen * 192, 73);
    const pos = Array.from({ length: s.kvLen }, (_, i) => i);
    const got = groupedQueryAttention(q, k, v, 9, 3, 64, { causal: true, queryPosition: s.queryPos, kvPositions: pos });
    const want = gqaReference(q, k, v, 9, 3, 64, true, s.queryPos, pos);
    assertClose(got, want, 1e-5);
  });
}

test("groupedQueryAttention fully-masked query returns zeros", () => {
  const q = randArr(576, 81);
  const k = randArr(3 * 192, 82);
  const v = randArr(3 * 192, 83);
  // Query position -1: every KV position is after it, so all masked.
  const got = groupedQueryAttention(q, k, v, 9, 3, 64, { causal: true, queryPosition: -1, kvPositions: [0, 1, 2] });
  assert.deepEqual(got, new Float32Array(576));
});

/** Naive RoPE reference: cos/sin recomputed per element, no cache. */
function applyRopeReference(
  q: Float32Array, k: Float32Array, positions: number[],
  numHeads: number, headDim: number, theta: number, numKvHeads: number,
): void {
  const half = headDim / 2;
  const rotate = (x: Float32Array, rowSize: number, heads: number): void => {
    for (let r = 0; r < positions.length; r++) {
      const pos = positions[r] as number;
      for (let h = 0; h < heads; h++) {
        const base = r * rowSize + h * headDim;
        for (let i = 0; i < half; i++) {
          const angle = pos * Math.pow(theta, -2 * i / headDim);
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          const i0 = base + 2 * i;
          const x0 = x[i0], x1 = x[i0 + 1];
          x[i0] = x0 * c - x1 * s;
          x[i0 + 1] = x0 * s + x1 * c;
        }
      }
    }
  };
  rotate(q, numHeads * headDim, numHeads);
  rotate(k, numKvHeads * headDim, numKvHeads);
}

test("applyRope (cached tables) matches per-element reference, incl. repeated and large positions", () => {
  for (const pos of [0, 1, 100, 997, 5000]) {
    const q0 = randArr(576, 91), k0 = randArr(192, 92);
    const q1 = q0.slice(), k1 = k0.slice();
    const q2 = q0.slice(), k2 = k0.slice();
    applyRope(q1, k1, [pos], 9, 64, 10000, 3);
    applyRopeReference(q2, k2, [pos], 9, 64, 10000, 3);
    assertClose(q1, q2, 1e-6);
    assertClose(k1, k2, 1e-6);
    // Second call hits the cache — same result from the same input.
    const q3 = q0.slice(), k3 = k0.slice();
    applyRope(q3, k3, [pos], 9, 64, 10000, 3);
    assert.deepEqual(q3, q1);
    assert.deepEqual(k3, k1);
  }
});

test("applyRope multi-row matches per-element reference", () => {
  const rows = 3;
  const q1 = randArr(rows * 576, 95), k1 = randArr(rows * 192, 96);
  const q2 = q1.slice(), k2 = k1.slice();
  applyRope(q1, k1, [0, 7, 42], 9, 64, 10000, 3);
  applyRopeReference(q2, k2, [0, 7, 42], 9, 64, 10000, 3);
  assertClose(q1, q2, 1e-6);
  assertClose(k1, k2, 1e-6);
});
