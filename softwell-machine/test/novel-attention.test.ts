/**
 * Novel attention tests — run with: node test/novel-attention.test.ts
 *
 * Theorems A1 (landmark / block-mean attention) and A2 (dynamic
 * sparsity by block screening) from src/engine/novel-attention.ts,
 * checked against exact groupedQueryAttention on the same inputs:
 *
 *   - correctness: measured per-head L2 error vs exact attention is
 *     asserted to be within the returned rigorous error bound;
 *   - scaling: counted multiply-add ops are asserted to grow
 *     sub-quadratically (doubling the length less than triples the
 *     ops; exact attention would quadruple them), with the measured
 *     ratios printed;
 *   - edge cases: single token, empty (fully masked) prefix, GQA,
 *     non-causal masking, partial final block, stats reuse.
 *
 * No files, no network. All randomness is seeded (mulberry32).
 */

import assert from "node:assert/strict";
import test from "node:test";
import { groupedQueryAttention } from "../src/engine/attention.ts";
import {
  computeBlockStats,
  landmarkAttention,
  prefillLandmarkAttention,
  prefillSparseAttention,
  sparseAttention,
} from "../src/engine/novel-attention.ts";
import { mulberry32 } from "../src/engine/sampling.ts";

function randn(rng: () => number): number {
  // Box-Muller.
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randMatrix(rng: () => number, rows: number, cols: number, scale = 1): Float32Array {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = randn(rng) * scale;
  return m;
}

/**
 * KV cache whose keys/values are constant within each block of
 * `blockSize` positions plus iid noise of std `noise`: the locally
 * coherent regime Theorem A1 is designed for. Block bases are drawn
 * per (block, KV head).
 */
function coherentKV(
  rng: () => number,
  kvLen: number,
  numKvHeads: number,
  headDim: number,
  blockSize: number,
  noise: number,
): { k: Float32Array; v: Float32Array } {
  const k = new Float32Array(kvLen * numKvHeads * headDim);
  const v = new Float32Array(kvLen * numKvHeads * headDim);
  const numBlocks = Math.ceil(kvLen / blockSize);
  for (let b = 0; b < numBlocks; b++) {
    for (let kh = 0; kh < numKvHeads; kh++) {
      const kBase = new Float32Array(headDim);
      const vBase = new Float32Array(headDim);
      for (let d = 0; d < headDim; d++) {
        kBase[d] = randn(rng);
        vBase[d] = randn(rng);
      }
      for (let t = b * blockSize; t < Math.min((b + 1) * blockSize, kvLen); t++) {
        for (let d = 0; d < headDim; d++) {
          k[(t * numKvHeads + kh) * headDim + d] = kBase[d] + randn(rng) * noise;
          v[(t * numKvHeads + kh) * headDim + d] = vBase[d] + randn(rng) * noise;
        }
      }
    }
  }
  return { k, v };
}

/**
 * KV cache with concentrated attention: every query equal to `u`
 * attends almost entirely to block 0, whose keys point along +u,
 * while every other block's keys point along -u. The regime Theorem
 * A2 is designed for.
 */
function concentratedKV(
  rng: () => number,
  kvLen: number,
  numKvHeads: number,
  headDim: number,
  blockSize: number,
  u: Float32Array,
  align: number,
  noise: number,
): { k: Float32Array; v: Float32Array } {
  const k = new Float32Array(kvLen * numKvHeads * headDim);
  const v = randMatrix(rng, kvLen * numKvHeads, headDim);
  const numBlocks = Math.ceil(kvLen / blockSize);
  for (let b = 0; b < numBlocks; b++) {
    const sign = b === 0 ? 1 : -1;
    for (let kh = 0; kh < numKvHeads; kh++) {
      for (let t = b * blockSize; t < Math.min((b + 1) * blockSize, kvLen); t++) {
        for (let d = 0; d < headDim; d++) {
          k[(t * numKvHeads + kh) * headDim + d] = sign * align * u[d] + randn(rng) * noise;
        }
      }
    }
  }
  return { k, v };
}

/**
 * Rescale a direction vector so ||u||^2 == headDim. The concentrated
 * test data needs a known separation between the aligned block and the
 * rest (landmark scores near +5.7 vs -5.7 at headDim 8); an unnormalized
 * Gaussian draw has ||u||^2 ~ chi^2_8, which can be small enough by
 * chance that nothing is separable and the test measures the wrong
 * regime. Normalizing pins the regime down.
 */
function normalizeDirection(u: Float32Array, headDim: number): void {
  let s = 0;
  for (let d = 0; d < headDim; d++) s += u[d] * u[d];
  const f = Math.sqrt(headDim / s);
  for (let d = 0; d < headDim; d++) u[d] *= f;
}

/** Max over (row, head) of the L2 norm of (a - b) for one head row. */
function maxHeadError(a: Float32Array, b: Float32Array, numHeads: number, headDim: number): number {
  assert.equal(a.length, b.length);
  const rows = a.length / (numHeads * headDim);
  assert.ok(Number.isInteger(rows));
  let max = 0;
  for (let r = 0; r < rows; r++) {
    for (let h = 0; h < numHeads; h++) {
      let s = 0;
      for (let d = 0; d < headDim; d++) {
        const diff = a[(r * numHeads + h) * headDim + d] - b[(r * numHeads + h) * headDim + d];
        s += diff * diff;
      }
      if (Math.sqrt(s) > max) max = Math.sqrt(s);
    }
  }
  return max;
}

/** Exact causal prefill via groupedQueryAttention, one row at a time. */
function exactPrefill(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
): Float32Array {
  const seqLen = q.length / (numHeads * headDim);
  const out = new Float32Array(seqLen * numHeads * headDim);
  const kvPositions = Array.from({ length: seqLen }, (_, i) => i);
  for (let i = 0; i < seqLen; i++) {
    const qRow = q.subarray(i * numHeads * headDim, (i + 1) * numHeads * headDim);
    const row = groupedQueryAttention(qRow, k, v, numHeads, numKvHeads, headDim, {
      causal: true,
      queryPosition: i,
      kvPositions,
    });
    out.set(row, i * numHeads * headDim);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Theorem A1: landmark / block-mean attention
// ---------------------------------------------------------------------------

test("A1: error within the proven bound on locally-coherent keys (GQA)", () => {
  const rng = mulberry32(101);
  const kvLen = 48;
  const numHeads = 4;
  const numKvHeads = 2;
  const headDim = 8;
  const blockSize = 8;
  const q = randMatrix(rng, numHeads, headDim);
  const { k, v } = coherentKV(rng, kvLen, numKvHeads, headDim, blockSize, 0.01);
  const { stats } = computeBlockStats(k, v, numKvHeads, headDim, blockSize);

  const approx = landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize,
    localWindow: 4,
    deltaMax: 0.05,
    stats,
  });
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
  const err = maxHeadError(approx.output, exact, numHeads, headDim);
  console.log(
    `A1 coherent: err=${err.toExponential(3)} bound=${approx.errorBound.toExponential(3)} ` +
      `ops=${approx.ops} exactOps=${approx.exactOps}`,
  );
  assert.ok(err <= approx.errorBound, `measured error ${err} exceeds proven bound ${approx.errorBound}`);
  assert.ok(approx.errorBound < 0.5, `bound ${approx.errorBound} is vacuous, not a real guarantee`);
  assert.ok(err > 1e-6, "error is ~0: the approximation path was not actually exercised");
  assert.ok(approx.ops < approx.exactOps, "landmark attention did no less work than exact");
});

test("A1: bound stays tight on incoherent keys via the exact fallback", () => {
  const rng = mulberry32(202);
  const kvLen = 48;
  const numHeads = 4;
  const numKvHeads = 2;
  const headDim = 8;
  const q = randMatrix(rng, numHeads, headDim);
  // iid keys/values: no local coherence, so every block's dispersion
  // exceeds deltaMax and must fall back to exact computation.
  const k = randMatrix(rng, kvLen * numKvHeads, headDim);
  const v = randMatrix(rng, kvLen * numKvHeads, headDim);
  const { stats } = computeBlockStats(k, v, numKvHeads, headDim, 8);

  const approx = landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize: 8,
    localWindow: 4,
    deltaMax: 0.05,
    stats,
  });
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
  const err = maxHeadError(approx.output, exact, numHeads, headDim);
  console.log(`A1 incoherent: err=${err.toExponential(3)} bound=${approx.errorBound.toExponential(3)} ops=${approx.ops} exactOps=${approx.exactOps}`);
  assert.ok(err <= approx.errorBound, `measured error ${err} exceeds proven bound ${approx.errorBound}`);
  assert.ok(approx.errorBound < 0.01, `bound ${approx.errorBound} should be ~float slack when everything is exact`);
  assert.ok(err < 1e-4, `error ${err} should be float32 reassociation only when everything is exact`);
  // No savings here by design: with stats reused the work is exact
  // attention plus one q-norm per head.
  assert.ok(
    approx.ops <= approx.exactOps + numHeads * headDim,
    `ops ${approx.ops} should be ~exactOps ${approx.exactOps} on incoherent data`,
  );
});

test("A1: deltaMax=0 reduces to exact attention", () => {
  const rng = mulberry32(303);
  const kvLen = 32;
  const numHeads = 2;
  const numKvHeads = 1;
  const headDim = 8;
  const q = randMatrix(rng, numHeads, headDim);
  const { k, v } = coherentKV(rng, kvLen, numKvHeads, headDim, 8, 0.01);
  const approx = landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize: 8,
    localWindow: 4,
    deltaMax: 0,
  });
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
  const err = maxHeadError(approx.output, exact, numHeads, headDim);
  assert.ok(err <= approx.errorBound);
  assert.ok(err < 1e-4, `deltaMax=0 must be exact up to float32, got err ${err}`);
});

test("A1 prefill: sub-quadratic op scaling and error within bound", () => {
  const numHeads = 2;
  const numKvHeads = 1;
  const headDim = 8;
  const lengths = [64, 128, 256];
  const opsByLen: number[] = [];
  const exactByLen: number[] = [];
  for (const n of lengths) {
    const rng = mulberry32(1000 + n);
    const blockSize = Math.max(4, Math.ceil(Math.sqrt(n)));
    const q = randMatrix(rng, n * numHeads, headDim);
    const { k, v } = coherentKV(rng, n, numKvHeads, headDim, blockSize, 0.01);
    const approx = prefillLandmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
      localWindow: 8,
      deltaMax: 0.05,
    });
    const exact = exactPrefill(q, k, v, numHeads, numKvHeads, headDim);
    const err = maxHeadError(approx.output, exact, numHeads, headDim);
    console.log(
      `A1 prefill n=${n}: err=${err.toExponential(3)} bound=${approx.errorBound.toExponential(3)} ` +
        `ops=${approx.ops} exactOps=${approx.exactOps} ratio=${(approx.ops / approx.exactOps).toFixed(3)}`,
    );
    assert.ok(err <= approx.errorBound, `n=${n}: measured error ${err} exceeds bound ${approx.errorBound}`);
    assert.ok(approx.errorBound < 0.5, `n=${n}: bound ${approx.errorBound} is vacuous`);
    opsByLen.push(approx.ops);
    exactByLen.push(approx.exactOps);
  }
  const r1 = opsByLen[1] / opsByLen[0];
  const r2 = opsByLen[2] / opsByLen[1];
  console.log(`A1 prefill op ratios: 128/64=${r1.toFixed(3)} 256/128=${r2.toFixed(3)} (exact would be ~4)`);
  assert.ok(r1 < 3, `doubling 64->128 less than triples ops, got ratio ${r1}`);
  assert.ok(r2 < 3, `doubling 128->256 less than triples ops, got ratio ${r2}`);
  assert.ok(opsByLen[2] < 0.5 * exactByLen[2], `n=256 ops ${opsByLen[2]} should be < half of exact ${exactByLen[2]}`);
});

// ---------------------------------------------------------------------------
// Theorem A2: dynamic sparsity by block screening
// ---------------------------------------------------------------------------

test("A2: error within the proven bound on concentrated attention (GQA)", () => {
  const rng = mulberry32(404);
  const kvLen = 64;
  const numHeads = 4;
  const numKvHeads = 2;
  const headDim = 8;
  const blockSize = 8;
  const u = randMatrix(rng, 1, headDim);
  normalizeDirection(u, headDim);
  const q = new Float32Array(numHeads * headDim);
  for (let h = 0; h < numHeads; h++) q.set(u, h * headDim);
  const { k, v } = concentratedKV(rng, kvLen, numKvHeads, headDim, blockSize, u, 2, 0.1);
  const { stats } = computeBlockStats(k, v, numKvHeads, headDim, blockSize);

  // Stats are built once per KV cache and reused across queries; the
  // work assertion below uses the reused-stats path, where the only
  // cost is the per-query screening and the surviving blocks.
  const approx = sparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize,
    localWindow: 4,
    dropThreshold: 1e-3,
    stats,
  });
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
  const err = maxHeadError(approx.output, exact, numHeads, headDim);
  console.log(
    `A2 concentrated: err=${err.toExponential(3)} bound=${approx.errorBound.toExponential(3)} ` +
      `droppedMass=${approx.droppedMassBound.toExponential(3)} ops=${approx.ops} exactOps=${approx.exactOps}`,
  );
  assert.ok(err <= approx.errorBound, `measured error ${err} exceeds proven bound ${approx.errorBound}`);
  assert.ok(approx.droppedMassBound > 0, "screening should have dropped some blocks");
  assert.ok(approx.droppedMassBound < 0.05, `dropped mass bound ${approx.droppedMassBound} is too large`);
  assert.ok(approx.errorBound < 0.5, `bound ${approx.errorBound} is vacuous`);
  assert.ok(approx.ops < 0.5 * approx.exactOps, "screening did not cut the work at least in half");
  // Building stats on the fly must give the same answer.
  const fresh = sparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize,
    localWindow: 4,
    dropThreshold: 1e-3,
  });
  assert.deepEqual([...fresh.output], [...approx.output]);
});

test("A2: uniform attention keeps everything (no false sparsity)", () => {
  const rng = mulberry32(505);
  const kvLen = 32;
  const numHeads = 2;
  const numKvHeads = 1;
  const headDim = 8;
  const q = randMatrix(rng, numHeads, headDim);
  // All keys identical: every block has the same landmark score, so no
  // block's mass bound can fall below the threshold.
  const k = new Float32Array(kvLen * numKvHeads * headDim);
  const one = randMatrix(rng, 1, headDim);
  for (let t = 0; t < kvLen; t++) k.set(one, t * headDim);
  const v = randMatrix(rng, kvLen * numKvHeads, headDim);
  const approx = sparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
    blockSize: 8,
    localWindow: 4,
    dropThreshold: 1e-3,
  });
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
  const err = maxHeadError(approx.output, exact, numHeads, headDim);
  assert.ok(err <= approx.errorBound);
  assert.ok(err < 1e-4, `uniform attention must be exact up to float32, got err ${err}`);
  assert.equal(approx.droppedMassBound, 0, "no block should be dropped when all scores tie");
});

test("A2 prefill: sub-quadratic op scaling and error within bound", () => {
  const numHeads = 2;
  const numKvHeads = 1;
  const headDim = 8;
  const lengths = [64, 128, 256];
  const opsByLen: number[] = [];
  const exactByLen: number[] = [];
  for (const n of lengths) {
    const rng = mulberry32(2000 + n);
    const blockSize = Math.max(4, Math.ceil(Math.sqrt(n)));
    const u = randMatrix(rng, 1, headDim);
    normalizeDirection(u, headDim);
    const q = new Float32Array(n * numHeads * headDim);
    for (let i = 0; i < n * numHeads; i++) q.set(u, i * headDim);
    const { k, v } = concentratedKV(rng, n, numKvHeads, headDim, blockSize, u, 2, 0.1);
    const approx = prefillSparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
      localWindow: 8,
      dropThreshold: 1e-3,
    });
    const exact = exactPrefill(q, k, v, numHeads, numKvHeads, headDim);
    const err = maxHeadError(approx.output, exact, numHeads, headDim);
    console.log(
      `A2 prefill n=${n}: err=${err.toExponential(3)} bound=${approx.errorBound.toExponential(3)} ` +
        `droppedMass=${approx.droppedMassBound.toExponential(3)} ops=${approx.ops} ` +
        `exactOps=${approx.exactOps} ratio=${(approx.ops / approx.exactOps).toFixed(3)}`,
    );
    assert.ok(err <= approx.errorBound, `n=${n}: measured error ${err} exceeds bound ${approx.errorBound}`);
    assert.ok(approx.errorBound < 0.5, `n=${n}: bound ${approx.errorBound} is vacuous`);
    opsByLen.push(approx.ops);
    exactByLen.push(approx.exactOps);
  }
  const r1 = opsByLen[1] / opsByLen[0];
  const r2 = opsByLen[2] / opsByLen[1];
  console.log(`A2 prefill op ratios: 128/64=${r1.toFixed(3)} 256/128=${r2.toFixed(3)} (exact would be ~4)`);
  assert.ok(r1 < 3, `doubling 64->128 less than triples ops, got ratio ${r1}`);
  assert.ok(r2 < 3, `doubling 128->256 less than triples ops, got ratio ${r2}`);
  assert.ok(opsByLen[2] < 0.5 * exactByLen[2], `n=256 ops ${opsByLen[2]} should be < half of exact ${exactByLen[2]}`);
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

test("edge cases: single token, empty prefix, GQA, non-causal, partial block", () => {
  const rng = mulberry32(606);

  // Single token: prefill of length 1 and a single query over kvLen 1.
  {
    const numHeads = 2;
    const numKvHeads = 1;
    const headDim = 8;
    const q = randMatrix(rng, 1 * numHeads, headDim);
    const k = randMatrix(rng, 1 * numKvHeads, headDim);
    const v = randMatrix(rng, 1 * numKvHeads, headDim);
    for (const fn of [prefillLandmarkAttention, prefillSparseAttention]) {
      const approx = fn(q, k, v, numHeads, numKvHeads, headDim, { localWindow: 4 });
      const exact = exactPrefill(q, k, v, numHeads, numKvHeads, headDim);
      assert.ok(maxHeadError(approx.output, exact, numHeads, headDim) <= approx.errorBound);
      assert.ok(maxHeadError(approx.output, exact, numHeads, headDim) < 1e-5);
    }
    const single = landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, { localWindow: 4 });
    const exactSingle = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
    assert.ok(maxHeadError(single.output, exactSingle, numHeads, headDim) < 1e-5);
  }

  // Empty prefix: causal masking hides every KV position.
  {
    const numHeads = 2;
    const numKvHeads = 2;
    const headDim = 8;
    const q = randMatrix(rng, numHeads, headDim);
    const k = randMatrix(rng, 3 * numKvHeads, headDim);
    const v = randMatrix(rng, 3 * numKvHeads, headDim);
    const opts = { causal: true, queryPosition: -1, kvPositions: [0, 1, 2], localWindow: 2 };
    for (const fn of [landmarkAttention, sparseAttention]) {
      const approx = fn(q, k, v, numHeads, numKvHeads, headDim, opts);
      const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim, opts);
      assert.deepEqual([...approx.output], [...exact]);
      assert.deepEqual([...approx.output], new Array(numHeads * headDim).fill(0));
      assert.equal(approx.errorBound, 0);
    }
  }

  // GQA with a single KV head shared by 4 query heads, causal prefix.
  {
    const numHeads = 4;
    const numKvHeads = 1;
    const headDim = 8;
    const kvLen = 24;
    const q = randMatrix(rng, numHeads, headDim);
    const { k, v } = coherentKV(rng, kvLen, numKvHeads, headDim, 8, 0.01);
    const kvPositions = Array.from({ length: kvLen }, (_, i) => i);
    for (const fn of [landmarkAttention, sparseAttention]) {
      const approx = fn(q, k, v, numHeads, numKvHeads, headDim, {
        causal: true,
        queryPosition: 17,
        kvPositions,
        blockSize: 8,
        localWindow: 4,
      });
      const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim, {
        causal: true,
        queryPosition: 17,
        kvPositions,
      });
      const err = maxHeadError(approx.output, exact, numHeads, headDim);
      assert.ok(err <= approx.errorBound, `GQA-1KV err ${err} exceeds bound ${approx.errorBound}`);
    }
  }

  // Non-causal: every KV position visible regardless of positions.
  {
    const numHeads = 2;
    const numKvHeads = 2;
    const headDim = 8;
    const kvLen = 20;
    const q = randMatrix(rng, numHeads, headDim);
    const { k, v } = coherentKV(rng, kvLen, numKvHeads, headDim, 8, 0.01);
    for (const fn of [landmarkAttention, sparseAttention]) {
      const approx = fn(q, k, v, numHeads, numKvHeads, headDim, {
        causal: false,
        blockSize: 8,
        localWindow: 4,
      });
      const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim, { causal: false });
      const err = maxHeadError(approx.output, exact, numHeads, headDim);
      assert.ok(err <= approx.errorBound, `non-causal err ${err} exceeds bound ${approx.errorBound}`);
    }
  }

  // Partial final block: kvLen 10 is not a multiple of blockSize 4.
  {
    const numHeads = 2;
    const numKvHeads = 1;
    const headDim = 8;
    const q = randMatrix(rng, numHeads, headDim);
    const { k, v } = coherentKV(rng, 10, numKvHeads, headDim, 4, 0.01);
    for (const fn of [landmarkAttention, sparseAttention]) {
      const approx = fn(q, k, v, numHeads, numKvHeads, headDim, { blockSize: 4, localWindow: 3 });
      const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);
      const err = maxHeadError(approx.output, exact, numHeads, headDim);
      assert.ok(err <= approx.errorBound, `partial-block err ${err} exceeds bound ${approx.errorBound}`);
    }
  }
});

test("edge cases: stats reuse is identical, shape errors match the reference", () => {
  const rng = mulberry32(707);
  const numHeads = 4;
  const numKvHeads = 2;
  const headDim = 8;
  const kvLen = 32;
  const q = randMatrix(rng, numHeads, headDim);
  const { k, v } = coherentKV(rng, kvLen, numKvHeads, headDim, 8, 0.01);
  const { stats } = computeBlockStats(k, v, numKvHeads, headDim, 8);

  for (const fn of [landmarkAttention, sparseAttention]) {
    const fresh = fn(q, k, v, numHeads, numKvHeads, headDim, { blockSize: 8, localWindow: 4 });
    const reused = fn(q, k, v, numHeads, numKvHeads, headDim, { localWindow: 4, stats });
    assert.deepEqual([...reused.output], [...fresh.output], "precomputed stats must not change the output");
    assert.equal(reused.errorBound, fresh.errorBound);
    // Reused stats cost no build ops, so reused work is strictly less.
    assert.ok(reused.ops < fresh.ops);
  }

  // Stats for a different blockSize are rejected.
  assert.throws(
    () => landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, { blockSize: 4, stats }),
    /blockSize/,
  );
  // Empty KV cache throws, like groupedQueryAttention.
  assert.throws(
    () => landmarkAttention(q, new Float32Array(0), new Float32Array(0), numHeads, numKvHeads, headDim),
    /whole number of KV rows/,
  );
  assert.throws(
    () => groupedQueryAttention(q, new Float32Array(0), new Float32Array(0), numHeads, numKvHeads, headDim),
    /whole number of KV rows/,
  );
  // Unsorted kvPositions are rejected: block methods need a prefix.
  const unsorted = Array.from({ length: kvLen }, (_, i) => i);
  unsorted[2] = 3;
  unsorted[3] = 2;
  assert.throws(
    () =>
      landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
        causal: true,
        queryPosition: 5,
        kvPositions: unsorted,
      }),
    /non-decreasing/,
  );
});
