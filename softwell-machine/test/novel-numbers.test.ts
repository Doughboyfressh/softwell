/**
 * Novel numbers tests — run with: node test/novel-numbers.test.ts
 *
 * SW-code (src/engine/novel-numbers.ts), checked the way the theorems
 * are stated — no fabricated measurements:
 *
 *   - (a) encode/decode error on realistic weight-like distributions
 *     (Gaussian, heavy-tailed spike mixture) vs the proven per-element
 *     bound from maxElementErrorBound; measured max/RMS error printed
 *     next to the bound;
 *   - (b) matmul through the packed format (float A x packed B^T, and
 *     packed A x packed B^T) vs a float64 reference, measured output
 *     error asserted within the Theorem N1 dot-product bound;
 *   - (c) KV quantization: attentionPackedKV vs exact
 *     groupedQueryAttention, measured per-head output error asserted
 *     within the Theorem N2 bound;
 *   - (d) measured bytes per parameter and traffic reduction vs BF16
 *     and FP32, on Gaussian and heavy-tailed weights and on KV cache.
 *
 * No files, no network. All randomness is seeded (mulberry32).
 */

import assert from "node:assert/strict";
import test from "node:test";
import { groupedQueryAttention, matmulTransposeB } from "../src/engine/attention.ts";
import type { PackedKV, PackedTensor, SwMode } from "../src/engine/novel-numbers.ts";
import {
  attentionPackedKV,
  decodeMatrix,
  describePacked,
  encodeMatrix,
  floatToFp16Bits,
  fp16BitsToFloat,
  kvAttentionErrorBound,
  matmulABErrorBound,
  matmulBTErrorBound,
  matmulPackedAB,
  matmulPackedBT,
  maxElementErrorBound,
  quantizeKV,
} from "../src/engine/novel-numbers.ts";
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

/** Heavy-tailed weight-like distribution: 99% N(0, sigma), 1% N(0, 10 sigma). */
function spikeMatrix(rng: () => number, rows: number, cols: number, sigma = 0.02): Float32Array {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) {
    m[i] = (rng() < 0.01 ? randn(rng) * 10 : randn(rng)) * sigma;
  }
  return m;
}

function maxAbs(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) {
    const a = Math.abs(x[i]);
    if (a > m) m = a;
  }
  return m;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function rmsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s / a.length);
}

/** Float64 reference C = A @ B^T (accumulation in JS numbers, no float32 rounding). */
function refMatmulBT(a: Float32Array, aRows: number, aCols: number, b: Float32Array, bRows: number): Float64Array {
  const c = new Float64Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bRows; j++) {
      let s = 0;
      for (let k = 0; k < aCols; k++) s += a[i * aCols + k] * b[j * aCols + k];
      c[i * bRows + j] = s;
    }
  }
  return c;
}

const MODES: SwMode[] = ["sw2", "sw3", "sw4"];

// ---------------------------------------------------------------------------
// fp16 conversion
// ---------------------------------------------------------------------------

test("fp16 conversion: known bit patterns and round-trip error bound", () => {
  assert.equal(floatToFp16Bits(1), 0x3c00);
  assert.equal(floatToFp16Bits(-1), 0xbc00);
  assert.equal(floatToFp16Bits(0.5), 0x3800);
  assert.equal(floatToFp16Bits(-2), 0xc000);
  assert.equal(floatToFp16Bits(65504), 0x7bff);
  assert.equal(floatToFp16Bits(6.103515625e-5), 0x0400); // smallest normal
  assert.equal(floatToFp16Bits(0), 0x0000);
  assert.equal(floatToFp16Bits(-0), 0x8000);
  assert.equal(fp16BitsToFloat(0x3c00), 1);
  assert.equal(fp16BitsToFloat(0x0400), 6.103515625e-5);
  assert.equal(fp16BitsToFloat(0x0001), 2 ** -24); // smallest subnormal
  assert.equal(fp16BitsToFloat(0x7c00), Infinity);
  assert.ok(Number.isNaN(fp16BitsToFloat(0x7e00)));

  // Random round-trip: |fl16(x) - x| <= 2^-11 |x| + 2^-25 (half-ulp).
  const rng = mulberry32(7);
  let worstRatio = 0;
  for (let i = 0; i < 20000; i++) {
    const x = randn(rng) * Math.pow(10, Math.floor(rng() * 9) - 4); // 1e-4 .. 1e4 scale
    const y = fp16BitsToFloat(floatToFp16Bits(x));
    const err = Math.abs(y - x);
    const allowed = 2 ** -11 * Math.abs(x) + 2 ** -25;
    assert.ok(err <= allowed, `fp16 round-trip exceeded half-ulp: x=${x} y=${y}`);
    if (Math.abs(x) > 1e-3) worstRatio = Math.max(worstRatio, err / Math.abs(x));
  }
  console.log(`fp16 round-trip: worst relative error ${worstRatio.toExponential(2)} (bound 2^-11 = ${(2 ** -11).toExponential(2)})`);
});

// ---------------------------------------------------------------------------
// (a) encode/decode error vs the proven bound
// ---------------------------------------------------------------------------

test("(a) encode/decode error within the proven per-element bound", () => {
  const rng = mulberry32(1234);
  const rows = 256;
  const cols = 512;
  for (const dist of ["gaussian", "spike"] as const) {
    const x = dist === "gaussian" ? randMatrix(rng, rows, cols, 0.02) : spikeMatrix(rng, rows, cols, 0.02);
    const maxX = maxAbs(x);
    for (const mode of MODES) {
      const packed = encodeMatrix(x, rows, cols, mode);
      const decoded = decodeMatrix(packed);
      const bound = maxElementErrorBound(packed);
      // The decoded output is float32, so comparing against the float32
      // input adds at most 2^-24 * |x| of float32 rounding on top of the
      // proven bound; that term is included in the assertion, not hidden.
      const slack = 2 ** -24 * maxX + 1e-12;
      const measuredMax = maxAbsDiff(decoded, x);
      const measuredRms = rmsDiff(decoded, x);
      const desc = describePacked(packed);
      console.log(
        `${dist} ${mode}: max err ${measuredMax.toExponential(3)} rms ${measuredRms.toExponential(3)} ` +
          `bound ${bound.toExponential(3)} outliers ${(100 * desc.outlierFraction).toFixed(2)}% ` +
          `${desc.bytesPerElement.toFixed(3)} B/param`,
      );
      assert.ok(
        measuredMax <= bound + slack,
        `${dist} ${mode}: measured max error ${measuredMax} exceeds bound ${bound} + slack ${slack}`,
      );
      assert.ok(measuredRms <= bound, `${dist} ${mode}: rms error exceeds the max-element bound`);
      assert.ok(desc.outlierFraction < 0.125, `${dist} ${mode}: outlier fraction above the 1/8 construction cap`);
    }
  }
});

test("encode/decode edge cases: zeros, constants, tiny and large values, errors", () => {
  // All zeros: exact, no outliers, minimal bytes.
  const zeros = new Float32Array(128);
  const pz = encodeMatrix(zeros, 2, 64, "sw4");
  assert.deepEqual(decodeMatrix(pz), zeros);
  assert.equal(pz.outlierTotal, 0);

  // Constant block: exact up to the step (all values equal the clip).
  const constant = new Float32Array(64).fill(0.5);
  const pc = encodeMatrix(constant, 1, 64, "sw4");
  const dc = decodeMatrix(pc);
  assert.ok(maxAbsDiff(dc, constant) <= maxElementErrorBound(pc) + 1e-12);

  // Single element and non-multiple-of-blockSize shapes.
  const single = new Float32Array([0.123]);
  assert.ok(Math.abs(decodeMatrix(encodeMatrix(single, 1, 1, "sw3"))[0] - 0.123) < 0.01);
  const odd = randMatrix(mulberry32(5), 3, 100, 0.02);
  const oddPacked = encodeMatrix(odd, 3, 100, "sw2");
  assert.ok(
    maxAbsDiff(decodeMatrix(oddPacked), odd) <= maxElementErrorBound(oddPacked) + 2 ** -24 * maxAbs(odd) + 1e-12,
  );

  // Tiny values (block scale underflows fp16): decode to ~0, error < 2^-24.
  const tiny = new Float32Array(64).fill(1e-9);
  tiny[0] = 5e-8; // one value above the tiny clip becomes an fp16 outlier
  const pt = encodeMatrix(tiny, 1, 64, "sw4");
  assert.ok(maxAbsDiff(decodeMatrix(pt), tiny) < 2 ** -23);

  // Large-but-in-range values.
  const big = new Float32Array(64).fill(60000);
  big[3] = -60000;
  const bigPacked = encodeMatrix(big, 1, 64, "sw4");
  assert.ok(maxAbsDiff(decodeMatrix(bigPacked), big) <= maxElementErrorBound(bigPacked) + 1e-9);

  // Determinism: same input -> identical bytes.
  const m = randMatrix(mulberry32(9), 16, 128, 0.02);
  const p1 = encodeMatrix(m, 16, 128, "sw3");
  const p2 = encodeMatrix(m, 16, 128, "sw3");
  assert.deepEqual(p1.data, p2.data);
  assert.deepEqual(p1.rowOffsets, p2.rowOffsets);

  // Invalid inputs throw instead of corrupting silently.
  const bad = new Float32Array([1, NaN, 3, 4]);
  assert.throws(() => encodeMatrix(bad, 1, 4, "sw4"), /finite/);
  assert.throws(() => encodeMatrix(new Float32Array([1, Infinity, 3, 4]), 1, 4, "sw4"), /finite/);
  assert.throws(() => encodeMatrix(new Float32Array([70000, 1, 2, 3]), 1, 4, "sw4"), /65504/);
  assert.throws(() => encodeMatrix(new Float32Array(8), 2, 3, "sw4"), /length/);
  assert.throws(() => encodeMatrix(new Float32Array(8), 2, 4, "sw4", { blockSize: 0 }), /blockSize/);
  assert.throws(() => encodeMatrix(new Float32Array(8), 2, 4, "sw4", { blockSize: 257 }), /blockSize/);
  assert.throws(() => encodeMatrix(new Float32Array(8), 2, 4, "sw9" as SwMode), /unknown SW mode/);
  const good: PackedTensor = encodeMatrix(new Float32Array(8), 2, 4, "sw4");
  assert.throws(() => matmulPackedBT(new Float32Array(8), 2, 4, { ...good, cols: 5 }), /shape mismatch/);
});

// ---------------------------------------------------------------------------
// (b) matmul through the packed format vs the Theorem N1 bound
// ---------------------------------------------------------------------------

test("(b) matmulPackedBT: output error within the N1 dot-product bound", () => {
  const rng = mulberry32(2024);
  const aRows = 16;
  const aCols = 256;
  const bRows = 128;
  const a = randMatrix(rng, aRows, aCols, 1); // activations ~ N(0,1)
  for (const dist of ["gaussian", "spike"] as const) {
    const b = dist === "gaussian" ? randMatrix(rng, bRows, aCols, 0.02) : spikeMatrix(rng, bRows, aCols, 0.02);
    const ref = refMatmulBT(a, aRows, aCols, b, bRows);
    for (const mode of MODES) {
      const packed = encodeMatrix(b, bRows, aCols, mode);
      const out = matmulPackedBT(a, aRows, aCols, packed);
      // Sanity: packed matmul agrees with dequantize-then-matmul (float32).
      const deq = matmulTransposeB(a, aRows, aCols, decodeMatrix(packed), bRows, aCols);
      assert.ok(maxAbsDiff(out, deq) <= 1e-4, `${dist} ${mode}: packed matmul disagrees with dequantize-then-matmul`);
      let worstRatio = 0;
      let worstMeasured = 0;
      let worstBound = 0;
      let sumSq = 0;
      for (let i = 0; i < aRows; i++) {
        const aRow = a.subarray(i * aCols, (i + 1) * aCols);
        for (let j = 0; j < bRows; j++) {
          const bound = matmulBTErrorBound(aRow, packed, j);
          const measured = Math.abs(out[i * bRows + j] - ref[i * bRows + j]);
          sumSq += measured * measured;
          // Slack: the packed output is float32 and the reference is
          // float64; 1e-6 absorbs float32 rounding of the output.
          assert.ok(
            measured <= bound * (1 + 1e-9) + 1e-6,
            `${dist} ${mode}: output (${i},${j}) error ${measured} exceeds N1 bound ${bound}`,
          );
          const ratio = measured / Math.max(bound, 1e-30);
          if (ratio > worstRatio) worstRatio = ratio;
          if (measured > worstMeasured) {
            worstMeasured = measured;
            worstBound = bound;
          }
        }
      }
      console.log(
        `matmul ${dist} ${mode}: worst measured err ${worstMeasured.toExponential(3)} ` +
          `rms ${Math.sqrt(sumSq / (aRows * bRows)).toExponential(3)} ` +
          `at bound ${worstBound.toExponential(3)} (worst measured/bound = ${worstRatio.toFixed(3)})`,
      );
    }
  }
});

test("(b) matmulPackedAB: quantized activations x quantized weights within the composed bound", () => {
  const rng = mulberry32(777);
  const aRows = 8;
  const aCols = 256;
  const bRows = 128;
  const aFloat = randMatrix(rng, aRows, aCols, 1);
  const bFloat = randMatrix(rng, bRows, aCols, 0.02);
  const ref = refMatmulBT(aFloat, aRows, aCols, bFloat, bRows);
  for (const [aMode, bMode] of [["sw4", "sw4"], ["sw4", "sw3"], ["sw3", "sw3"]] as Array<[SwMode, SwMode]>) {
    const aPacked = encodeMatrix(aFloat, aRows, aCols, aMode);
    const bPacked = encodeMatrix(bFloat, bRows, aCols, bMode);
    const out = matmulPackedAB(aPacked, bPacked);
    let worstRatio = 0;
    let worstMeasured = 0;
    let worstBound = 0;
    let sumSq = 0;
    for (let i = 0; i < aRows; i++) {
      for (let j = 0; j < bRows; j++) {
        const bound = matmulABErrorBound(aPacked, i, bPacked, j);
        const measured = Math.abs(out[i * bRows + j] - ref[i * bRows + j]);
        sumSq += measured * measured;
        assert.ok(
          measured <= bound * (1 + 1e-9) + 1e-6,
          `${aMode}x${bMode}: output (${i},${j}) error ${measured} exceeds composed bound ${bound}`,
        );
        const ratio = measured / Math.max(bound, 1e-30);
        if (ratio > worstRatio) worstRatio = ratio;
        if (measured > worstMeasured) {
          worstMeasured = measured;
          worstBound = bound;
        }
      }
    }
    console.log(
      `matmulAB ${aMode}x${bMode}: worst measured err ${worstMeasured.toExponential(3)} ` +
        `rms ${Math.sqrt(sumSq / (aRows * bRows)).toExponential(3)} ` +
        `at bound ${worstBound.toExponential(3)} (worst measured/bound = ${worstRatio.toFixed(3)})`,
    );
  }
  // Mismatched block sizes are a hard error, not silent misalignment.
  const aPacked = encodeMatrix(aFloat, aRows, aCols, "sw4", { blockSize: 32 });
  const bPacked = encodeMatrix(bFloat, bRows, aCols, "sw4", { blockSize: 64 });
  assert.throws(() => matmulPackedAB(aPacked, bPacked), /blockSize mismatch/);
});

// ---------------------------------------------------------------------------
// (c) KV-cache quantization vs the Theorem N2 bound
// ---------------------------------------------------------------------------

function checkKvAttention(
  label: string,
  keys: Float32Array,
  values: Float32Array,
  kvLen: number,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  mode: SwMode | { k: SwMode; v: SwMode },
  causal: boolean,
): void {
  const rng = mulberry32(4242);
  const q = randMatrix(rng, numHeads, headDim, 1);
  const exact = groupedQueryAttention(q, keys, values, numHeads, numKvHeads, headDim, { causal });
  const kv: PackedKV = quantizeKV(keys, values, kvLen, numKvHeads, headDim, mode);
  const packed = attentionPackedKV(q, kv, numHeads, { causal });
  let worstRatio = 0;
  let worstMeasured = 0;
  let worstBound = 0;
  for (let h = 0; h < numHeads; h++) {
    const bound = kvAttentionErrorBound(q, kv, numHeads, h, { causal });
    let sumSq = 0;
    for (let d = 0; d < headDim; d++) {
      const diff = packed[h * headDim + d] - exact[h * headDim + d];
      sumSq += diff * diff;
    }
    const measured = Math.sqrt(sumSq);
    // Slack: both attentions accumulate in float32; 1e-5 absorbs that.
    assert.ok(
      measured <= bound * (1 + 1e-9) + 1e-5,
      `${label} head ${h}: attention error ${measured} exceeds N2 bound ${bound}`,
    );
    const ratio = measured / Math.max(bound, 1e-30);
    if (ratio > worstRatio) worstRatio = ratio;
    if (measured > worstMeasured) {
      worstMeasured = measured;
      worstBound = bound;
    }
  }
  const kDesc = describePacked(kv.k);
  const vDesc = describePacked(kv.v);
  console.log(
    `KV ${label}: worst head L2 err ${worstMeasured.toExponential(3)} at bound ${worstBound.toExponential(3)} ` +
      `(worst measured/bound = ${worstRatio.toFixed(3)}); ` +
      `K ${kDesc.bytesPerElement.toFixed(3)} B/elem, V ${vDesc.bytesPerElement.toFixed(3)} B/elem`,
  );
}

test("(c) KV quantization: attention output error within the N2 bound", () => {
  const rng = mulberry32(31337);
  const kvLen = 128;
  const numHeads = 4;
  const numKvHeads = 2;
  const headDim = 64;

  // Plain Gaussian K/V.
  const k1 = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  const v1 = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  checkKvAttention("gaussian sw4/sw4", k1, v1, kvLen, numHeads, numKvHeads, headDim, "sw4", false);
  checkKvAttention("gaussian sw4/sw2", k1, v1, kvLen, numHeads, numKvHeads, headDim, { k: "sw4", v: "sw2" }, false);
  checkKvAttention("gaussian sw3/sw3 causal", k1, v1, kvLen, numHeads, numKvHeads, headDim, "sw3", true);

  // K with per-channel outliers (a few channels 8x larger, as in real
  // LLM key caches): the robust clip must keep the other channels exact.
  const k2 = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  for (let t = 0; t < kvLen * numKvHeads; t++) {
    k2[t * headDim + 0] *= 8;
    k2[t * headDim + 7] *= 8;
  }
  const v2 = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  checkKvAttention("K-channel-outliers sw4/sw4", k2, v2, kvLen, numHeads, numKvHeads, headDim, "sw4", false);
  checkKvAttention("K-channel-outliers sw2/sw2", k2, v2, kvLen, numHeads, numKvHeads, headDim, "sw2", false);
});

test("(c) KV causal masking matches the exact reference", () => {
  const rng = mulberry32(606);
  const kvLen = 16;
  const numHeads = 2;
  const numKvHeads = 1;
  const headDim = 16;
  const k = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  const v = randMatrix(rng, kvLen * numKvHeads, headDim, 1);
  const q = randMatrix(rng, numHeads, headDim, 1);
  const kvPositions = Array.from({ length: kvLen }, (_, t) => t);
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim, {
    causal: true,
    queryPosition: 5,
    kvPositions,
  });
  const kv = quantizeKV(k, v, kvLen, numKvHeads, headDim, "sw4");
  const packed = attentionPackedKV(q, kv, numHeads, { causal: true, queryPosition: 5, kvPositions });
  // Causal masking only removes terms; the N2 bound (computed without
  // masking) still applies, and the error must be small in absolute terms.
  for (let h = 0; h < numHeads; h++) {
    const bound = kvAttentionErrorBound(q, kv, numHeads, h, { causal: true, queryPosition: 5, kvPositions });
    let sumSq = 0;
    for (let d = 0; d < headDim; d++) {
      const diff = packed[h * headDim + d] - exact[h * headDim + d];
      sumSq += diff * diff;
    }
    assert.ok(Math.sqrt(sumSq) <= bound * (1 + 1e-9) + 1e-5, `causal head ${h} exceeds N2 bound ${bound}`);
  }
  assert.throws(() => quantizeKV(k, v, kvLen, numKvHeads, headDim + 1, "sw4"), /length/);
});

// ---------------------------------------------------------------------------
// (d) measured traffic: bytes per parameter vs BF16 / FP32
// ---------------------------------------------------------------------------

test("(d) traffic: measured bytes per parameter and reduction vs BF16/FP32", () => {
  const rng = mulberry32(2718);
  const rows = 1024;
  const cols = 2048;
  for (const dist of ["gaussian", "spike"] as const) {
    const x = dist === "gaussian" ? randMatrix(rng, rows, cols, 0.02) : spikeMatrix(rng, rows, cols, 0.02);
    console.log(`--- ${dist} weights ${rows}x${cols} (${((rows * cols) / 1e6).toFixed(1)}M params) ---`);
    for (const mode of MODES) {
      const packed = encodeMatrix(x, rows, cols, mode);
      const d = describePacked(packed);
      console.log(
        `${mode}: ${d.bytesPerElement.toFixed(4)} B/param (${d.bitsPerElement.toFixed(2)} bits), ` +
          `${d.ratioVsBf16.toFixed(2)}x vs BF16, ${d.ratioVsFp32.toFixed(2)}x vs FP32, ` +
          `outliers ${(100 * d.outlierFraction).toFixed(3)}%`,
      );
      if (dist === "gaussian") {
        if (mode === "sw2") assert.ok(d.ratioVsBf16 >= 6.0, `sw2 Gaussian ratio ${d.ratioVsBf16} < 6x vs BF16`);
        if (mode === "sw3") assert.ok(d.ratioVsBf16 >= 4.5, `sw3 Gaussian ratio ${d.ratioVsBf16} < 4.5x vs BF16`);
        if (mode === "sw4") assert.ok(d.ratioVsBf16 >= 3.5, `sw4 Gaussian ratio ${d.ratioVsBf16} < 3.5x vs BF16`);
        assert.ok(d.ratioVsFp32 >= 7.0, `${mode} Gaussian ratio ${d.ratioVsFp32} < 7x vs FP32`);
      } else {
        if (mode === "sw2") assert.ok(d.ratioVsBf16 >= 5.5, `sw2 spike ratio ${d.ratioVsBf16} < 5.5x vs BF16`);
        if (mode === "sw3") assert.ok(d.ratioVsBf16 >= 4.0, `sw3 spike ratio ${d.ratioVsBf16} < 4x vs BF16`);
        if (mode === "sw4") assert.ok(d.ratioVsBf16 >= 3.2, `sw4 spike ratio ${d.ratioVsBf16} < 3.2x vs BF16`);
      }
    }
  }

  // KV cache bytes: one (token, KV head) vector per row, blockSize = headDim.
  const rngKv = mulberry32(99);
  for (const headDim of [64, 128]) {
    const kvLen = 1024;
    const numKvHeads = 8;
    const k = randMatrix(rngKv, kvLen * numKvHeads, headDim, 1);
    const v = randMatrix(rngKv, kvLen * numKvHeads, headDim, 1);
    const kv = quantizeKV(k, v, kvLen, numKvHeads, headDim, "sw4");
    const kd = describePacked(kv.k);
    const vd = describePacked(kv.v);
    console.log(
      `KV headDim ${headDim} sw4: K ${kd.bytesPerElement.toFixed(3)} B/elem (${kd.ratioVsBf16.toFixed(2)}x vs BF16, ` +
        `${kd.ratioVsFp32.toFixed(2)}x vs FP32), V ${vd.bytesPerElement.toFixed(3)} B/elem`,
    );
    // KV rows are short (one head vector), so the 4-byte row offset per
    // row costs 4/headDim bytes per element on top of the block payload:
    // headDim 64 -> ~0.55 + 0.06 = 0.61 B/elem (3.25x), headDim 128 ->
    // ~0.52 + 0.03 = 0.55 B/elem (3.6x). Assert the measured values.
    if (headDim === 64) {
      assert.ok(kd.ratioVsBf16 >= 3.2, `KV headDim ${headDim} ratio ${kd.ratioVsBf16} < 3.2x vs BF16`);
      assert.ok(kd.ratioVsFp32 >= 6.4, `KV headDim ${headDim} ratio ${kd.ratioVsFp32} < 6.4x vs FP32`);
    } else {
      assert.ok(kd.ratioVsBf16 >= 3.5, `KV headDim ${headDim} ratio ${kd.ratioVsBf16} < 3.5x vs BF16`);
      assert.ok(kd.ratioVsFp32 >= 7.0, `KV headDim ${headDim} ratio ${kd.ratioVsFp32} < 7.0x vs FP32`);
    }
  }
});
