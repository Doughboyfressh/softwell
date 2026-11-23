/**
 * vNPU kernel test — run with: node test/vnpu-kernels.test.ts (exit 0 on pass).
 *
 * Covers, in-process with a fake NodeProvider (no real donors):
 *  - INT8 matmul / matmulTransposeB (per-tensor AND per-channel scales)
 *    vs the serial FP32 reference (matmul / matmulTransposeB): the
 *    observed max error is always <= the device's computed error bound
 *  - FP8-E4M3 and BF16 matmulTransposeB vs the same reference, same bar
 *  - attention: exact == groupedQueryAttention; landmark (A1) and sparse
 *    (A2) within their theorem error bounds, with fewer ops than exact
 *    on structured data; prefill landmark within its bound
 *  - SW-code native format: quantize/dequantize round-trip within
 *    maxElementErrorBound; matmulSw within its N1 bound; quantizeKv +
 *    attentionPackedKv within the N2 bound
 *  - tile-fit rejection: an attention whose KV block exceeds one tile's
 *    4 MiB SRAM is rejected at enqueue time
 *  - streams: a failed instruction on a stream does not block later ones
 *  - elementwise instructions (SOFTMAX / RMSNORM / ACT) match CpuKernels
 *
 * The error bound is the point: every quantized result carries a bound
 * computed from the actual quantization errors, and the test pins
 * observed <= bound. The bound/observed ratio is logged, not asserted
 * (random-sign cancellation makes the observed error much smaller than
 * the worst-case bound by construction).
 */
import assert from "node:assert/strict";
import { VirtualNpu, VNPU_SRAM_BYTES, VNPU_TILE_SRAM_BYTES, VNPU_TILES } from "../src/hw/vnpu.ts";
import { decodeToFloat32, encodeFromFloat32 } from "../src/hw/vgpu.ts";
import { CpuKernels } from "../src/hw/kernels-cpu.ts";
import { groupedQueryAttention, matmul, matmulTransposeB } from "../src/engine/attention.ts";
import { maxElementErrorBound } from "../src/engine/novel-numbers.ts";
import type { IVirtualRam, NodeProvider, VramAllocation, VramStats, WorkerInfo } from "../src/contracts.ts";

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

function makeNode(workerId: string): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 128,
    state: "healthy",
    registeredAt: 0,
    lastHeartbeatAt: 0,
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeRam implements IVirtualRam {
  async alloc(): Promise<VramAllocation> {
    throw new Error("FakeRam.alloc not used here");
  }
  async free(): Promise<void> {}
  async read(): Promise<Uint8Array> {
    throw new Error("FakeRam.read not used here");
  }
  async write(): Promise<void> {}
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: 0, usedBytes: 0, freeBytes: 0, pages: 0, nodes: 0 };
  }
}

const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
const npu = new VirtualNpu({ nodes: provider, ram: new FakeRam() });

function maxAbsErr(a: Float32Array, b: Float32Array): number {
  assert.equal(a.length, b.length, "length mismatch in maxAbsErr");
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    // Fail loudly on NaN/Inf: a NaN diff compares false against every
    // bound, so a silent max() would report a 0 error for garbage.
    assert.ok(Number.isFinite(e), `maxAbsErr: non-finite diff at index ${i} (${a[i]} vs ${b[i]})`);
    if (e > m) m = e;
  }
  return m;
}

function randMatrix(rand: () => number, rows: number, cols: number, scale = 1): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = (rand() * 2 - 1) * scale;
  return out;
}

// --- D1 constants ------------------------------------------------------------

{
  assert.equal(VNPU_TILES, 64, "D1: 64 tiles per device");
  assert.equal(VNPU_TILE_SRAM_BYTES, 4 * 1024 * 1024, "D1: 4 MiB tile SRAM");
  assert.equal(VNPU_SRAM_BYTES, 256 * 1024 * 1024, "D1: 256 MiB SRAM per device");
  const devices = npu.listDevices();
  assert.equal(devices.length, 1);
  assert.equal(devices[0]?.id, "vnpu-node-a");
  assert.equal(devices[0]?.tiles, 64);
  assert.equal(devices[0]?.sramBytes, VNPU_SRAM_BYTES);
  assert.equal(devices[0]?.int8Tops, 1e12, "int8Tops is the donor's own throughput, no multiplier");
  console.log("ok D1: 256 MiB SRAM (64 tiles x 4 MiB) per device");
}

// --- INT8 matmulTransposeB: per-tensor and per-channel ------------------------

{
  const rand = mulberry32(11);
  const a = randMatrix(rand, 4, 16);
  const b = randMatrix(rand, 8, 16);
  // One B row 10x larger, so per-channel scales matter.
  for (let c = 0; c < 16; c++) b[3 * 16 + c] *= 10;
  const ref = matmulTransposeB(a, 4, 16, b, 8, 16);

  for (const weightScale of ["per-tensor", "per-channel"] as const) {
    const r = await npu.matmulTransposeB(a, 4, 16, b, 8, 16, { format: "int8", weightScale });
    assert.equal(r.op, "MATMUL_INT8");
    assert.equal(r.format, "int8");
    assert.equal(r.device, "cpu", "reference executes on CPU, reported honestly");
    assert.equal(r.sramHit, false);
    const err = maxAbsErr(r.output, ref);
    assert.ok(Number.isFinite(r.errorBound) && r.errorBound > 0, "bound is finite and positive");
    assert.ok(err <= r.errorBound, `int8 ${weightScale}: observed ${err} > bound ${r.errorBound}`);
    console.log(
      `ok int8 matmulTransposeB ${weightScale}: observed max err ${err.toExponential(2)}, bound ${r.errorBound.toExponential(2)} (ratio ${(r.errorBound / Math.max(err, 1e-12)).toFixed(1)}x)`,
    );
  }
  // Default format is int8.
  const dflt = await npu.matmulTransposeB(a, 4, 16, b, 8, 16);
  assert.equal(dflt.format, "int8");
  assert.equal(dflt.op, "MATMUL_INT8");
  console.log("ok int8 matmulTransposeB: default format is int8");
}

// --- INT8 matmul (A @ B): per-tensor and per-channel (columns) ---------------

{
  const rand = mulberry32(23);
  const a = randMatrix(rand, 4, 16);
  const b = randMatrix(rand, 16, 8);
  for (let r = 0; r < 16; r++) b[r * 8 + 5] *= 10; // one B column 10x larger
  const ref = matmul(a, 4, 16, b, 16, 8);
  for (const weightScale of ["per-tensor", "per-channel"] as const) {
    const r = await npu.matmul(a, 4, 16, b, 16, 8, { format: "int8", weightScale });
    assert.equal(r.op, "MATMUL_INT8");
    const err = maxAbsErr(r.output, ref);
    assert.ok(err <= r.errorBound, `int8 matmul ${weightScale}: observed ${err} > bound ${r.errorBound}`);
    console.log(
      `ok int8 matmul ${weightScale}: observed max err ${err.toExponential(2)}, bound ${r.errorBound.toExponential(2)}`,
    );
  }
}

// --- FP8 and BF16 matmulTransposeB --------------------------------------------

{
  const rand = mulberry32(37);
  const a = randMatrix(rand, 4, 16);
  const b = randMatrix(rand, 8, 16);
  const ref = matmulTransposeB(a, 4, 16, b, 8, 16);
  for (const format of ["fp8", "bf16"] as const) {
    const r = await npu.matmulTransposeB(a, 4, 16, b, 8, 16, { format });
    assert.equal(r.format, format);
    const err = maxAbsErr(r.output, ref);
    assert.ok(err <= r.errorBound, `${format}: observed ${err} > bound ${r.errorBound}`);
    console.log(
      `ok ${format} matmulTransposeB: observed max err ${err.toExponential(2)}, bound ${r.errorBound.toExponential(2)} (ratio ${(r.errorBound / Math.max(err, 1e-12)).toFixed(1)}x)`,
    );
  }
}

// --- FP8 codec round-trip (pins the vgpu.ts E4M3 fix) ------------------------
// E4M3 has no infinities (0x7f/0xff are NaN, 0x7e/0xfe is 448). The old
// decoder treated every exp==15 byte as Inf/NaN, so the encoder's own
// saturation value 0x7e decoded back to NaN — and a NaN in the
// dequantized weights silently zeroed every error bound it touched.
{
  const probe = new Float32Array([448, -448, 464, 512, 0.001, -0.001, 1.5, -1.5, 0, 256, 100]);
  const codes = encodeFromFloat32(probe, "float8_e4m3");
  const back = decodeToFloat32(codes, "float8_e4m3");
  for (let i = 0; i < probe.length; i++) {
    assert.ok(Number.isFinite(back[i] as number), `fp8 round-trip of ${probe[i]} must stay finite`);
  }
  assert.equal(back[0], 448, "max finite 448 round-trips");
  assert.equal(back[1], -448);
  assert.equal(back[2], 448, "464 saturates to 448, not NaN");
  assert.equal(back[3], 448, "512 saturates to 448");
  assert.ok(Math.abs((back[6] as number) - 1.5) < 0.1, "1.5 survives with E4M3 precision");
  const nanCode = encodeFromFloat32(new Float32Array([NaN]), "float8_e4m3");
  assert.equal(nanCode[0], 0x7f, "NaN encodes to 0x7f");
  assert.ok(Number.isNaN(decodeToFloat32(nanCode, "float8_e4m3")[0] as number));
  const infCode = encodeFromFloat32(new Float32Array([Infinity]), "float8_e4m3");
  assert.equal(infCode[0], 0x7e, "Infinity saturates to 448 (no infinities in E4M3)");
  // Full sweep: every one of the 256 byte values decodes finite except NaN.
  for (let byte = 0; byte < 256; byte++) {
    const v = decodeToFloat32(new Uint8Array([byte]), "float8_e4m3")[0] as number;
    if (byte === 0x7f || byte === 0xff) assert.ok(Number.isNaN(v), `byte ${byte} is NaN`);
    else assert.ok(Number.isFinite(v), `byte ${byte} must decode finite, got ${v}`);
  }
  console.log("ok FP8 codec: 448 round-trips, >448 saturates, Inf saturates, only 0x7f/0xff are NaN");
}


{
  const rand = mulberry32(41);
  const numHeads = 2;
  const numKvHeads = 2;
  const headDim = 8;
  const kvLen = 64;
  const blockSize = 8;
  // Clustered K/V: each block shares a base vector plus tiny noise, so
  // landmark approximation is safe (small radii) and ops drop.
  const k = new Float32Array(kvLen * numKvHeads * headDim);
  const v = new Float32Array(kvLen * numKvHeads * headDim);
  for (let blk = 0; blk < kvLen / blockSize; blk++) {
    for (let kh = 0; kh < numKvHeads; kh++) {
      const baseK = randMatrix(rand, 1, headDim);
      const baseV = randMatrix(rand, 1, headDim);
      for (let t = blk * blockSize; t < (blk + 1) * blockSize; t++) {
        for (let d = 0; d < headDim; d++) {
          k[(t * numKvHeads + kh) * headDim + d] = baseK[d] as number + (rand() * 2 - 1) * 0.01;
          v[(t * numKvHeads + kh) * headDim + d] = baseV[d] as number + (rand() * 2 - 1) * 0.01;
        }
      }
    }
  }
  const q = randMatrix(rand, numHeads, headDim);
  const exact = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim);

  const rExact = await npu.attention(q, k, v, numHeads, numKvHeads, headDim, { mode: "exact" });
  assert.equal(rExact.op, "ATTN_EXACT");
  assert.equal(rExact.errorBound, 0);
  assert.ok(maxAbsErr(rExact.output, exact) < 1e-5, "ATTN_EXACT matches groupedQueryAttention");
  console.log("ok ATTN_EXACT matches groupedQueryAttention (< 1e-5)");

  // Block stats are built once and reused: the ops figures below exclude
  // the one-time stats build (the build itself is counted separately in
  // buildBlockStats' returned ops), the same convention as
  // novel-attention's tests.
  const { stats, ops: statsOps } = npu.buildBlockStats(k, v, numKvHeads, headDim, blockSize);
  assert.ok(statsOps > 0, "stats build counts its ops");

  const rLand = await npu.attention(q, k, v, numHeads, numKvHeads, headDim, {
    mode: "landmark",
    blockSize,
    stats,
  });
  assert.equal(rLand.op, "ATTN_LANDMARK");
  const landErr = maxAbsErr(rLand.output, exact);
  assert.ok(landErr <= rLand.errorBound, `A1: observed ${landErr} > bound ${rLand.errorBound}`);
  assert.ok(rLand.ops < rLand.exactOps, `A1: ops ${rLand.ops} not below exact ${rLand.exactOps}`);
  console.log(
    `ok ATTN_LANDMARK (A1): L2-ish max err ${landErr.toExponential(2)} <= bound ${rLand.errorBound.toExponential(2)}, ops ${rLand.ops} < exact ${rLand.exactOps}`,
  );

  // Without precomputed stats the same call rebuilds them and gives the
  // same output (and the same bound).
  const rLandNoStats = await npu.attention(q, k, v, numHeads, numKvHeads, headDim, {
    mode: "landmark",
    blockSize,
  });
  assert.ok(maxAbsErr(rLandNoStats.output, rLand.output) < 1e-6, "stats built on the fly match");
  assert.ok(maxAbsErr(rLandNoStats.output, exact) <= rLandNoStats.errorBound);
  console.log("ok ATTN_LANDMARK builds identical stats on the fly");

  // Sparse: distant K tiny, local K aligned with q, so distant blocks screen out.
  // Local scores must exceed distant landmark scores by > ln(B/dropThreshold)
  // (~9 for B=8, threshold 1e-3), hence the 12x local alignment.
  const k2 = new Float32Array(kvLen * numKvHeads * headDim);
  const v2 = randMatrix(rand, kvLen * numKvHeads, headDim);
  for (let t = 0; t < kvLen; t++) {
    for (let kh = 0; kh < numKvHeads; kh++) {
      for (let d = 0; d < headDim; d++) {
        const local = t >= kvLen - 16;
        k2[(t * numKvHeads + kh) * headDim + d] = local
          ? q[kh * headDim + d] as number * 12
          : (rand() * 2 - 1) * 0.05;
      }
    }
  }
  const exact2 = groupedQueryAttention(q, k2, v2, numHeads, numKvHeads, headDim);
  const { stats: stats2 } = npu.buildBlockStats(k2, v2, numKvHeads, headDim, blockSize);
  const rSparse = await npu.attention(q, k2, v2, numHeads, numKvHeads, headDim, {
    mode: "sparse",
    blockSize,
    stats: stats2,
  });
  assert.equal(rSparse.op, "ATTN_SPARSE");
  const sparseErr = maxAbsErr(rSparse.output, exact2);
  assert.ok(sparseErr <= rSparse.errorBound, `A2: observed ${sparseErr} > bound ${rSparse.errorBound}`);
  assert.ok(rSparse.ops < rSparse.exactOps, `A2: ops ${rSparse.ops} not below exact ${rSparse.exactOps}`);
  console.log(
    `ok ATTN_SPARSE (A2): max err ${sparseErr.toExponential(2)} <= bound ${rSparse.errorBound.toExponential(2)}, dropped mass bound ${rSparse.droppedMassBound.toExponential(2)}, ops ${rSparse.ops} < exact ${rSparse.exactOps}`,
  );

  // Prefill landmark vs per-row exact.
  const seqLen = 32;
  const qp = randMatrix(rand, seqLen * numHeads, headDim);
  const kp = k.subarray(0, seqLen * numKvHeads * headDim);
  const vp = v.subarray(0, seqLen * numKvHeads * headDim);
  const { stats: preStats } = npu.buildBlockStats(kp, vp, numKvHeads, headDim, blockSize);
  const pre = await npu.prefillAttention(qp, kp, vp, numHeads, numKvHeads, headDim, {
    mode: "landmark",
    blockSize,
    stats: preStats,
  });
  let preErr = 0;
  for (let i = 0; i < seqLen; i++) {
    const qi = qp.subarray(i * numHeads * headDim, (i + 1) * numHeads * headDim);
    const rowExact = groupedQueryAttention(
      qi,
      kp,
      vp,
      numHeads,
      numKvHeads,
      headDim,
      { queryPosition: i, kvPositions: Array.from({ length: seqLen }, (_, t) => t) },
    );
    const rowOut = pre.output.subarray(i * numHeads * headDim, (i + 1) * numHeads * headDim);
    preErr = Math.max(preErr, maxAbsErr(rowOut, rowExact));
  }
  assert.ok(preErr <= pre.errorBound, `prefill A1: observed ${preErr} > bound ${pre.errorBound}`);
  assert.ok(pre.ops < pre.exactOps, `prefill A1: ops ${pre.ops} not below exact ${pre.exactOps}`);
  console.log(
    `ok prefill ATTN_LANDMARK: max err ${preErr.toExponential(2)} <= bound ${pre.errorBound.toExponential(2)}, ops ${pre.ops} < exact ${pre.exactOps}`,
  );
}

// --- SW-code native format -----------------------------------------------------

{
  const rand = mulberry32(53);
  const rows = 6;
  const cols = 32;
  const data = randMatrix(rand, rows, cols);
  const packed = npu.quantizeSw(data, rows, cols, "sw4");
  assert.equal(packed.mode, "sw4");
  const decoded = npu.dequantizeSw(packed);
  const bound = maxElementErrorBound(packed);
  const err = maxAbsErr(decoded, data);
  assert.ok(err <= bound, `SW round-trip: observed ${err} > bound ${bound}`);
  console.log(`ok SW-code round-trip (sw4): max err ${err.toExponential(2)} <= bound ${bound.toExponential(2)}`);

  const a = randMatrix(rand, 3, cols);
  const b = randMatrix(rand, rows, cols);
  const packedB = npu.quantizeSw(b, rows, cols, "sw3");
  const r = await npu.matmulSw(a, 3, cols, packedB);
  assert.equal(r.op, "MATMUL_SW");
  assert.equal(r.format, "sw3");
  const ref = matmulTransposeB(a, 3, cols, b, rows, cols);
  const swErr = maxAbsErr(r.output, ref);
  assert.ok(swErr <= r.errorBound, `MATMUL_SW: observed ${swErr} > bound ${r.errorBound}`);
  console.log(
    `ok MATMUL_SW (sw3): max err ${swErr.toExponential(2)} <= N1 bound ${r.errorBound.toExponential(2)}`,
  );

  // KV_QUANTIZE + attention over packed KV (ATTN_PACKED_KV) within the N2 bound.
  const kvLen = 16;
  const numKvHeads = 2;
  const headDim = 8;
  const keys = randMatrix(rand, kvLen * numKvHeads, headDim, 0.5);
  const values = randMatrix(rand, kvLen * numKvHeads, headDim, 0.5);
  const kv = npu.quantizeKv(keys, values, kvLen, numKvHeads, headDim, "sw4");
  const q = randMatrix(rand, 2, headDim, 0.5);
  const rKv = await npu.attentionPackedKv(q, kv, 2);
  assert.equal(rKv.op, "ATTN_PACKED_KV");
  const exactKv = groupedQueryAttention(q, keys, values, 2, numKvHeads, headDim);
  const kvErr = maxAbsErr(rKv.output, exactKv);
  assert.ok(kvErr <= rKv.errorBound, `ATTN_PACKED_KV: observed ${kvErr} > N2 bound ${rKv.errorBound}`);
  console.log(
    `ok ATTN_PACKED_KV (N2): max err ${kvErr.toExponential(2)} <= bound ${rKv.errorBound.toExponential(2)}`,
  );
}

// --- tile-fit rejection ----------------------------------------------------------

{
  const rand = mulberry32(59);
  const q = randMatrix(rand, 2, 8);
  const k = randMatrix(rand, 64 * 2, 8);
  const v = randMatrix(rand, 64 * 2, 8);
  // One KV block of 64 rows x 2 heads x 8 dim x 4 bytes x 2 (K+V) = 16 KiB
  // fits; force a block that does not fit 4 MiB by asking for blockSize 100k.
  await assert.rejects(
    () => npu.attention(q, k, v, 2, 2, 8, { mode: "landmark", blockSize: 100_000 }),
    /exceeds one tile's SRAM/,
  );
  console.log("ok tile-fit: attention with an oversize KV block is rejected at enqueue time");
}

// --- streams ----------------------------------------------------------------------

{
  const stream = npu.createStream();
  assert.notEqual(stream, npu.createStream(), "streams are distinct");
  const a = randMatrix(mulberry32(61), 2, 8);
  const b = randMatrix(mulberry32(67), 4, 8);
  // A failing instruction on the stream must not block later ones.
  const bad = npu.matmulTransposeB(a, 2, 8, b, 4, 7, { streamId: stream });
  await assert.rejects(bad, /shape mismatch/);
  const good = await npu.matmulTransposeB(a, 2, 8, b, 4, 8, { streamId: stream });
  assert.equal(good.output.length, 8);
  await npu.synchronize(stream);
  console.log("ok streams: failed instruction does not block later ones on the same stream");
}

// --- elementwise instructions ------------------------------------------------------

{
  const rand = mulberry32(71);
  const x = randMatrix(rand, 3, 8);
  const sm = npu.softmax(x, 3, 8);
  const smRef = CpuKernels.softmaxRows(Float32Array.from(x), 3, 8);
  assert.ok(maxAbsErr(sm, smRef) < 1e-6, "SOFTMAX matches CpuKernels");
  const w = randMatrix(rand, 1, 8);
  const rn = npu.rmsNorm(x, 3, 8, w, 1e-5);
  const rnRef = CpuKernels.rmsNorm(x, 3, 8, w, 1e-5);
  assert.ok(maxAbsErr(rn, rnRef) < 1e-6, "RMSNORM matches CpuKernels");
  const s = npu.silu(x);
  const sRef = CpuKernels.silu(x);
  assert.ok(maxAbsErr(s, sRef) < 1e-6, "ACT (SiLU) matches CpuKernels");
  console.log("ok elementwise: SOFTMAX / RMSNORM / ACT match CpuKernels");
}

// --- stats --------------------------------------------------------------------------

{
  const s = npu.stats();
  assert.equal(s.devices.length, 1);
  assert.equal(s.sramCapacityBytes, VNPU_SRAM_BYTES);
  assert.ok(s.int8Matmuls > 0 && s.fp8Matmuls > 0 && s.swMatmuls > 0 && s.bf16Matmuls > 0);
  assert.ok(s.attentionCalls > 0 && s.kvQuantizeCalls > 0);
  assert.ok(s.opsCompleted > 0 && s.opsFailed > 0, "the rejected shape-mismatch counts as failed");
  console.log(
    `ok stats: int8=${s.int8Matmuls} fp8=${s.fp8Matmuls} sw=${s.swMatmuls} bf16=${s.bf16Matmuls} attention=${s.attentionCalls} completed=${s.opsCompleted} failed=${s.opsFailed}`,
  );
}
