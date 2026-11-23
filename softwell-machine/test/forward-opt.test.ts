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
import { matmulTransposeB } from "../src/engine/attention.ts";

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
