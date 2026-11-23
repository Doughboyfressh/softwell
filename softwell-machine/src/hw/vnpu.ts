/**
 * vnpu.ts — the virtual NPU pool (the fifth virtual device).
 *
 * THE BOUNDARY: workloads see only VIRTUAL NPU devices (see
 * listDevices()) and instructions. A workload never learns which donor
 * backs a virtual device, and never sees a physical NPU. The mapping
 * virtual-device -> donor lives inside this file.
 *
 * What the vNPU is (docs/VNPU.md): the inference specialist. INT8/FP8
 * execution as first-class operations, attention and sparsity as
 * virtual-silicon primitives (Theorems A1/A2), and SW-code (Theorems
 * N1/N2) as the native number format. It is a real device with distinct
 * semantics, not a vGPU relabel: it issues different instructions
 * (MATMUL_INT8 / MATMUL_FP8 / MATMUL_SW / MATMUL_BF16, ATTN_LANDMARK /
 * ATTN_SPARSE / ATTN_EXACT, KV_QUANTIZE, QUANTIZE / DEQUANTIZE,
 * SOFTMAX / RMSNORM / ACT), against a different memory model, under a
 * different scheduler (per-sequence streams), than the vGPU.
 *
 * Design parameter D1 — SRAM capacity: 256 MiB per virtual device
 * (64 tiles x 4 MiB tile-local SRAM). A chosen design constant, in the
 * same class as TPU v5p VMEM (128 MB, secondary source) and Trainium2
 * SBUF (224 MiB); deliberately larger than either because the vNPU's
 * SRAM is pooled (N donors = N x 256 MiB aggregate). See docs/VNPU.md
 * §2.1. There is NO hardware cache and NO infinite cache: weights
 * stage from pooled vRAM into SRAM explicitly (stageWeights /
 * stageFromVram), a stage that does not fit evicts LRU residents, a
 * single weight larger than D1 spills (counted), and every use of a
 * non-resident weight counts a miss and streams the weight from vRAM.
 *
 * Reference implementation honesty: this reference executes every
 * instruction in-process on the coordinator (CPU fallback kernels,
 * like GpuNode), and every result reports device 'cpu' — the device
 * that ACTUALLY executed. It never claims NPU silicon it does not
 * have. The per-device throughput figures in listDevices() are the
 * donors' own reported sustained throughput with no multiplier. A
 * donor-side VnpuNode executing on real NPU silicon is future work
 * (docs/VNPU.md §4.2).
 *
 * Error bounds: every quantized matmul returns a rigorous bound on
 * |C_ij - C_ref_ij| against the serial FP32 reference
 * (matmulTransposeB / matmul in src/engine/attention.ts), computed
 * from the actual quantization errors by the triangle inequality —
 * never asserted, always computed. Attention instructions report the
 * theorems' own bounds (A1/A2/N2), evaluated on the actual data.
 */

import { decodeToFloat32, dtypeByteSize, encodeFromFloat32 } from "./vgpu.ts";
import { CpuKernels } from "./kernels-cpu.ts";
import { groupedQueryAttention, matmul, matmulTransposeB } from "../engine/attention.ts";
import {
  computeBlockStats,
  extendBlockStats,
  landmarkAttention,
  prefillLandmarkAttention,
  prefillSparseAttention,
  sparseAttention,
  type BlockStats,
} from "../engine/novel-attention.ts";
import {
  attentionPackedKV,
  decodeMatrix,
  encodeMatrix,
  kvAttentionErrorBound,
  matmulPackedBT,
  packedBytes,
  quantizeKV,
  rowErrorNormBound,
  type PackedKV,
  type PackedTensor,
  type SwMode,
} from "../engine/novel-numbers.ts";
import type {
  IVirtualNpu,
  IVirtualRam,
  NodeProvider,
  NpuAttentionMode,
  NpuAttentionOptions,
  NpuAttentionResult,
  NpuMatmulOptions,
  NpuMatmulResult,
  NpuOp,
  NpuPackedKvAttentionOptions,
  NpuSramStatus,
  NpuStageResult,
  NpuStats,
  NpuSwMatmulOptions,
  NpuWeightFormat,
  VAddr,
  VirtualNpuDevice,
  WeightDtype,
  WorkerDevice,
  WorkerInfo,
} from "../contracts.ts";

// ---------------------------------------------------------------------------
// Design parameters (docs/VNPU.md §2.1, §6 Q1/Q2)
// ---------------------------------------------------------------------------

/** Tiles per virtual device. */
export const VNPU_TILES = 64;
/** Tile-local SRAM per tile. An instruction's working set must fit here. */
export const VNPU_TILE_SRAM_BYTES = 4 * 1024 * 1024;
/** D1: SRAM capacity per virtual device = 64 tiles x 4 MiB = 256 MiB. */
export const VNPU_SRAM_BYTES = VNPU_TILES * VNPU_TILE_SRAM_BYTES;
/** Matmul tile dimension (the TPU MXU / Trainium2 Tensor Engine dimension). */
export const VNPU_TILE = 128;
/** Default execution format for matmuls that do not name one. */
export const VNPU_DEFAULT_FORMAT = "int8" as const;

// ---------------------------------------------------------------------------
// Quantization
//
// Every quantizer returns the dequantized fp32 values plus a per-element
// absolute error bound, from which the matmul error bounds below are
// computed. No bound is ever asserted without the data to back it.
// ---------------------------------------------------------------------------

/** FP8-E4M3: 3 mantissa bits -> half-ulp relative error 2^-4. */
const FP8_REL_ERR = 2 ** -4;
/** FP8-E4M3 subnormal half-step (2^-9 step / 2), times the scale. */
const FP8_SUBNORMAL_HALF_STEP = 2 ** -10;
/** BF16: 7 mantissa bits -> half-ulp relative error 2^-8. */
const BF16_REL_ERR = 2 ** -8;
/** Absolute floor for the BF16 bound (subnormal-scale values). */
const BF16_ABS_ERR = 2 ** -38;
/** INT8 codes are symmetric [-127, 127] (no -128), so the scale is exact. */
const INT8_MAX_CODE = 127;
/** FP8-E4M3 maximum finite value (saturation point in vgpu.ts). */
const FP8_MAX = 448;

export interface QuantAnalysis {
  /** L1 norm of each row of the dequantized values. */
  rowL1: Float64Array;
  /** L1 norm of each column of the dequantized values. */
  colL1: Float64Array;
  /** Max |x - x_hat| over each row. */
  rowErrMax: Float32Array;
  /** Max |x - x_hat| over each column. */
  colErrMax: Float32Array;
}

/** Row/column L1 norms and per-row/per-column max errors of a quantization. */
function analyzeQuantization(
  values: Float32Array,
  err: Float32Array,
  rows: number,
  cols: number,
): QuantAnalysis {
  const rowL1 = new Float64Array(rows);
  const colL1 = new Float64Array(cols);
  const rowErrMax = new Float32Array(rows);
  const colErrMax = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const a = Math.abs(values[i]);
      rowL1[r] += a;
      colL1[c] += a;
      // A NaN error must propagate (bound becomes NaN and the caller
      // fails loudly), never be dropped by a false comparison: NaN
      // once silently zeroed every bound it touched (see the FP8
      // codec fix in vgpu.ts).
      if (Number.isNaN(err[i]) || err[i] > rowErrMax[r]) rowErrMax[r] = err[i];
      if (Number.isNaN(err[i]) || err[i] > colErrMax[c]) colErrMax[c] = err[i];
    }
  }
  return { rowL1, colL1, rowErrMax, colErrMax };
}

function maxErr(analysis: QuantAnalysis): number {
  let m = 0;
  for (let r = 0; r < analysis.rowErrMax.length; r++) {
    if (analysis.rowErrMax[r] > m) m = analysis.rowErrMax[r];
  }
  return m;
}

export interface Int8Quantized {
  kind: "int8";
  codes: Int8Array;
  /** Per-channel scales: length 1 (per-tensor) or rows/cols (per-channel). */
  scales: Float32Array;
  perChannel: boolean;
  /** Which axis the per-channel scales run along ("row" or "col"). */
  channelAxis: "row" | "col";
  values: Float32Array;
  analysis: QuantAnalysis;
  rows: number;
  cols: number;
}

/**
 * Symmetric INT8 quantization. `scheme` "per-tensor" uses one scale
 * (max|x| / 127); "per-channel" uses one scale per row or per column.
 * No value is ever clamped: the scale is the block maximum / 127, so
 * the largest magnitude maps to exactly 127 and every element's error
 * is at most half a step (scale / 2).
 */
export function quantizeInt8(
  data: Float32Array,
  rows: number,
  cols: number,
  scheme: "per-tensor" | "per-channel",
  channelAxis: "row" | "col" = "row",
): Int8Quantized {
  if (data.length !== rows * cols) {
    throw new Error(`quantizeInt8: data length ${data.length} != ${rows}x${cols}`);
  }
  const numChannels = scheme === "per-channel" ? (channelAxis === "row" ? rows : cols) : 1;
  const scales = new Float32Array(numChannels);
  if (scheme === "per-tensor") {
    let m = 0;
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]);
      if (a > m) m = a;
    }
    scales[0] = m === 0 ? 1 : m / INT8_MAX_CODE;
  } else if (channelAxis === "row") {
    for (let r = 0; r < rows; r++) {
      let m = 0;
      for (let c = 0; c < cols; c++) {
        const a = Math.abs(data[r * cols + c]);
        if (a > m) m = a;
      }
      scales[r] = m === 0 ? 1 : m / INT8_MAX_CODE;
    }
  } else {
    for (let c = 0; c < cols; c++) {
      let m = 0;
      for (let r = 0; r < rows; r++) {
        const a = Math.abs(data[r * cols + c]);
        if (a > m) m = a;
      }
      scales[c] = m === 0 ? 1 : m / INT8_MAX_CODE;
    }
  }
  const codes = new Int8Array(rows * cols);
  const values = new Float32Array(rows * cols);
  const err = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const s = scheme === "per-channel" ? scales[channelAxis === "row" ? r : c] as number : (scales[0] as number);
      let q = Math.round(data[i] / s);
      if (q > INT8_MAX_CODE) q = INT8_MAX_CODE;
      if (q < -INT8_MAX_CODE) q = -INT8_MAX_CODE;
      codes[i] = q;
      values[i] = q * s;
      err[i] = s / 2;
    }
  }
  return {
    kind: "int8",
    codes,
    scales,
    perChannel: scheme === "per-channel",
    channelAxis,
    values,
    analysis: analyzeQuantization(values, err, rows, cols),
    rows,
    cols,
  };
}

export interface Fp8Quantized {
  kind: "fp8";
  codes: Uint8Array;
  scale: number;
  values: Float32Array;
  analysis: QuantAnalysis;
  rows: number;
  cols: number;
}

/**
 * FP8-E4M3 quantization with a per-tensor scale: values are scaled by
 * max|x| / 448 into the E4M3 range, encoded (vgpu.ts helpers), and
 * decoded back. Per-element error is at most 2^-4 * |x_hat| +
 * scale * 2^-10 (relative half-ulp, plus the subnormal half-step).
 */
export function quantizeFp8(data: Float32Array, rows: number, cols: number): Fp8Quantized {
  if (data.length !== rows * cols) {
    throw new Error(`quantizeFp8: data length ${data.length} != ${rows}x${cols}`);
  }
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (a > m) m = a;
  }
  const scale = m === 0 ? 1 : m / FP8_MAX;
  const scaled = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) scaled[i] = data[i] / scale;
  const codes = encodeFromFloat32(scaled, "float8_e4m3");
  const values = decodeToFloat32(codes, "float8_e4m3");
  const err = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    values[i] *= scale;
    err[i] = FP8_REL_ERR * Math.abs(values[i]) + scale * FP8_SUBNORMAL_HALF_STEP;
  }
  return {
    kind: "fp8",
    codes,
    scale,
    values,
    analysis: analyzeQuantization(values, err, rows, cols),
    rows,
    cols,
  };
}

export interface Bf16Quantized {
  kind: "bf16";
  bytes: Uint8Array;
  values: Float32Array;
  analysis: QuantAnalysis;
  rows: number;
  cols: number;
}

/** BF16 quantization (the compatibility path): round-to-nearest-even to
 *  7 mantissa bits; per-element error at most 2^-8 * |x_hat| + 2^-38. */
export function quantizeBf16(data: Float32Array, rows: number, cols: number): Bf16Quantized {
  if (data.length !== rows * cols) {
    throw new Error(`quantizeBf16: data length ${data.length} != ${rows}x${cols}`);
  }
  const bytes = encodeFromFloat32(data, "bfloat16");
  const values = decodeToFloat32(bytes, "bfloat16");
  const err = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    err[i] = BF16_REL_ERR * Math.abs(values[i]) + BF16_ABS_ERR;
  }
  return {
    kind: "bf16",
    bytes,
    values,
    analysis: analyzeQuantization(values, err, rows, cols),
    rows,
    cols,
  };
}

// ---------------------------------------------------------------------------
// Matmul error bounds.
//
// For C_ij = sum_k a_ik * b_kj and the quantized C_hat, the triangle
// inequality gives, per element,
//   |C_ij - C_hat_ij| <= sum_k (|a_ik| * eb_kj + |b_kj| * ea_ik
//                                + ea_ik * eb_kj)
//                      <= ebMax_j * ||a_i||_1 + eaMax * ||b_j||_1
//                         + K * eaMax * ebMax_j,
// where eaMax is the largest activation error and ebMax_j the largest
// weight error in output channel j. The bound is the max over (i, j).
// It is computed from the actual quantization errors, not asserted.
// ---------------------------------------------------------------------------

/** Bound for C = A @ B^T (B rows are the output channels). */
function boundTransposeB(
  aAnalysis: QuantAnalysis,
  bAnalysis: QuantAnalysis,
  aRows: number,
  bRows: number,
  k: number,
): number {
  const eaMax = maxErr(aAnalysis);
  let bound = 0;
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bRows; j++) {
      const ebMax = bAnalysis.rowErrMax[j] as number;
      const b =
        ebMax * (aAnalysis.rowL1[i] as number) +
        eaMax * (bAnalysis.rowL1[j] as number) +
        k * eaMax * ebMax;
      if (b > bound) bound = b;
    }
  }
  return bound;
}

/** Bound for C = A @ B (B columns are the output channels). */
function boundMatmul(
  aAnalysis: QuantAnalysis,
  bAnalysis: QuantAnalysis,
  aRows: number,
  bCols: number,
  k: number,
): number {
  const eaMax = maxErr(aAnalysis);
  let bound = 0;
  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      const ebMax = bAnalysis.colErrMax[j] as number;
      const b =
        ebMax * (aAnalysis.rowL1[i] as number) +
        eaMax * (bAnalysis.colL1[j] as number) +
        k * eaMax * ebMax;
      if (b > bound) bound = b;
    }
  }
  return bound;
}

// ---------------------------------------------------------------------------
// INT8 integer-MAC cores (exact integer arithmetic in float64 — the
// products and sums are exact for K <= 2^17 at |code| <= 127).
// ---------------------------------------------------------------------------

/** C = A @ B^T over INT8 codes with INT32-style accumulation. */
function int8MatmulTransposeBCore(
  a: Int8Quantized,
  b: Int8Quantized,
  aRows: number,
  bRows: number,
  k: number,
): Float32Array {
  const out = new Float32Array(aRows * bRows);
  const sA = a.scales[0] as number;
  for (let i = 0; i < aRows; i++) {
    const aBase = i * k;
    for (let j = 0; j < bRows; j++) {
      const sB = b.perChannel ? (b.scales[j] as number) : (b.scales[0] as number);
      const bBase = j * k;
      let sum = 0;
      for (let t = 0; t < k; t++) sum += a.codes[aBase + t] * b.codes[bBase + t];
      out[i * bRows + j] = sum * sA * sB;
    }
  }
  return out;
}

/** C = A @ B over INT8 codes. B's per-channel scales run along columns. */
function int8MatmulCore(
  a: Int8Quantized,
  b: Int8Quantized,
  aRows: number,
  bCols: number,
  k: number,
): Float32Array {
  const out = new Float32Array(aRows * bCols);
  const sA = a.scales[0] as number;
  for (let i = 0; i < aRows; i++) {
    const aBase = i * k;
    for (let j = 0; j < bCols; j++) {
      const sB = b.perChannel ? (b.scales[j] as number) : (b.scales[0] as number);
      let sum = 0;
      for (let t = 0; t < k; t++) sum += a.codes[aBase + t] * b.codes[t * bCols + j];
      out[i * bCols + j] = sum * sA * sB;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tile-fit check (docs/VNPU.md §2.1).
//
// An instruction's working set must fit one tile's 4 MiB SRAM. A matmul
// tile is 128x128x128 (VNPU_TILE): 16 KiB per INT8/FP8 operand tile and
// 64 KiB for the FP32/INT32 accumulator tile. An attention tile holds
// one KV block plus its block statistics plus the query row. A matmul
// always fits; an attention with a huge blockSize does not, and is
// rejected at enqueue time — never silently spilled to vRAM mid-op.
// ---------------------------------------------------------------------------

function tileWorkingSetMatmul(format: "int8" | "fp8" | "bf16" | SwMode): number {
  const tile = VNPU_TILE * VNPU_TILE;
  const acc = tile * 4; // FP32/INT32 accumulator tile
  switch (format) {
    case "int8":
    case "fp8":
      return tile * 1 + tile * 1 + acc;
    case "bf16":
      return tile * 2 + tile * 2 + acc;
    default: {
      // SW-code: A tile fp32 + packed B tile (~0.5 bytes/element at sw3)
      // + accumulator + lookup-table scratch.
      return tile * 4 + Math.ceil(tile / 2) + acc + 1024;
    }
  }
}

function tileWorkingSetAttention(
  blockSize: number,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
): number {
  // One KV block (K and V, fp32) + its block statistics (meanK, meanV,
  // radii) + the query row.
  const blockBytes = blockSize * numKvHeads * headDim * 4 * 2;
  const statsBytes = numKvHeads * headDim * 4 * 2 + numKvHeads * 4 * 2;
  const queryBytes = numHeads * headDim * 4;
  return blockBytes + statsBytes + queryBytes;
}

function checkTileFit(bytes: number, what: string): void {
  if (bytes > VNPU_TILE_SRAM_BYTES) {
    throw new Error(
      `vNPU: ${what} working set ${bytes} bytes exceeds one tile's SRAM ` +
        `(${VNPU_TILE_SRAM_BYTES} bytes); reduce the tile/block size — ` +
        `an instruction that cannot be tiled to 4 MiB is rejected at enqueue time`,
    );
  }
}

// ---------------------------------------------------------------------------
// SRAM residency: staged weights
// ---------------------------------------------------------------------------

type StagedPayload =
  | {
      kind: "int8";
      codes: Int8Array;
      scales: Float32Array;
      perChannel: boolean;
    }
  | { kind: "fp8"; codes: Uint8Array; scale: number }
  | { kind: "bf16"; bytes: Uint8Array }
  | { kind: "sw"; packed: PackedTensor };

interface SramEntry {
  key: string;
  format: NpuWeightFormat;
  rows: number;
  cols: number;
  bytes: number;
  payload: StagedPayload;
  lastUsed: number;
}

interface DeviceSram {
  entries: Map<string, SramEntry>;
  usedBytes: number;
}

export interface VirtualNpuOpts {
  nodes: NodeProvider;
  ram: IVirtualRam;
  /** Default execution format for matmuls that do not name one. */
  defaultFormat?: "int8" | "fp8";
  /**
   * Per-device SRAM capacity in bytes. Default D1 (256 MiB). The
   * override exists so tests can exercise spill/eviction without
   * staging hundreds of megabytes; production code never sets it.
   */
  sramBytes?: number;
}

function opForFormat(format: "int8" | "fp8" | "bf16" | SwMode): NpuOp {
  switch (format) {
    case "int8":
      return "MATMUL_INT8";
    case "fp8":
      return "MATMUL_FP8";
    case "bf16":
      return "MATMUL_BF16";
    default:
      return "MATMUL_SW";
  }
}

/** Visible KV length for one query, mirroring novel-attention's masking. */
function visibleLength(
  kvLen: number,
  causal: boolean,
  kvPositions: number[] | undefined,
  queryPosition: number | undefined,
): number {
  if (!causal || kvPositions === undefined || queryPosition === undefined) return kvLen;
  let vis = 0;
  for (let t = 0; t < kvLen; t++) {
    if ((kvPositions[t] as number) <= queryPosition) vis = t + 1;
  }
  return vis;
}

export class VirtualNpu implements IVirtualNpu {
  private nodes: NodeProvider;
  private ram: IVirtualRam;
  private defaultFormat: "int8" | "fp8";
  private sramBytes: number;
  /** Nodes lost via handleNodeLoss; never selected again. */
  private deadNodes = new Set<string>();
  /** Node ids ever presented as a virtual device (so handleNodeLoss can
   *  count a loss even if the provider already marked the node dead). */
  private seenNodes = new Set<string>();
  /** Per-stream promise chain; an instruction on a stream runs only
   *  after the previous one on that stream settles, so instructions
   *  on one stream execute in order. The stored chain swallows errors
   *  so one failed instruction does not block later ones. */
  private streamChains = new Map<string, Promise<void>>();
  private streamCounter = 0;
  /** Monotonic clock for LRU eviction. */
  private clock = 0;
  /** Weight key -> nodeId currently holding it resident. */
  private keyDevice = new Map<string, string>();
  /** Every key ever staged (resident, evicted, spilled, or lost with
   *  its donor). A later use of a known-but-not-resident key counts a
   *  miss; a never-staged key throws. */
  private knownKeys = new Set<string>();
  /** Per-node SRAM state. */
  private deviceSram = new Map<string, DeviceSram>();
  private sramHits = 0;
  private sramMisses = 0;
  private sramSpills = 0;
  private sramEvictions = 0;
  private int8Matmuls = 0;
  private fp8Matmuls = 0;
  private swMatmuls = 0;
  private bf16Matmuls = 0;
  private attentionCalls = 0;
  private kvQuantizeCalls = 0;
  private opsCompleted = 0;
  private opsFailed = 0;

  constructor(opts: VirtualNpuOpts) {
    this.nodes = opts.nodes;
    this.ram = opts.ram;
    this.defaultFormat = opts.defaultFormat ?? VNPU_DEFAULT_FORMAT;
    this.sramBytes = opts.sramBytes ?? VNPU_SRAM_BYTES;
  }

  // -- inventory ------------------------------------------------------------

  listDevices(): VirtualNpuDevice[] {
    const devices = this.nodes
      .listNodes()
      .filter((n) => n.state === "healthy" && !this.deadNodes.has(n.workerId))
      .map((n) => ({
        id: `vnpu-${n.workerId}`,
        tiles: VNPU_TILES,
        sramBytes: this.sramBytes,
        // Honest reporting: the donor's own sustained throughput, no
        // multiplier. The reference device claims no INT8/BF16 rate
        // the donor does not have.
        int8Tops: n.flopsPerSec,
        bf16Tflops: n.flopsPerSec,
        backedBy: n.device,
      }));
    for (const d of devices) this.seenNodes.add(d.id.slice("vnpu-".length));
    return devices;
  }

  createStream(): string {
    this.streamCounter += 1;
    return `npu-stream-${this.streamCounter}`;
  }

  async synchronize(streamId: string): Promise<void> {
    await (this.streamChains.get(streamId) ?? Promise.resolve());
  }

  // -- SRAM staging -----------------------------------------------------------

  /**
   * Stage fp32 weights into a device's SRAM as a quantized resident
   * copy. Weights are staged in [out, in] orientation (rows are
   * output channels), matching the engine. The fp32 source is never
   * retained — only the quantized payload and its scales live in SRAM.
   */
  stageWeights(
    key: string,
    data: Float32Array,
    rows: number,
    cols: number,
    format: NpuWeightFormat,
    opts?: { deviceId?: string; weightScale?: "per-tensor" | "per-channel" },
  ): NpuStageResult {
    if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(cols) || cols < 1) {
      throw new Error(`vNPU stageWeights: bad shape [${rows}, ${cols}] for key ${key}`);
    }
    if (data.length !== rows * cols) {
      throw new Error(
        `vNPU stageWeights: data length ${data.length} != ${rows}x${cols} for key ${key}`,
      );
    }
    const weightScale = opts?.weightScale ?? "per-channel";
    let payload: StagedPayload;
    let bytes: number;
    if (format === "int8") {
      const q = quantizeInt8(data, rows, cols, weightScale, "row");
      payload = { kind: "int8", codes: q.codes, scales: q.scales, perChannel: q.perChannel };
      bytes = q.codes.byteLength + q.scales.byteLength + 16;
    } else if (format === "fp8") {
      const q = quantizeFp8(data, rows, cols);
      payload = { kind: "fp8", codes: q.codes, scale: q.scale };
      bytes = q.codes.byteLength + 4 + 16;
    } else if (format === "bf16") {
      const q = quantizeBf16(data, rows, cols);
      payload = { kind: "bf16", bytes: q.bytes };
      bytes = q.bytes.byteLength + 16;
    } else {
      const packed = encodeMatrix(data, rows, cols, format);
      payload = { kind: "sw", packed };
      bytes = packedBytes(packed);
    }
    const nodeId = this.pickDevice(opts?.deviceId);
    // A single weight larger than the whole SRAM can never be resident:
    // record the spill and stay backed by vRAM.
    if (bytes > this.sramBytes) {
      this.sramSpills += 1;
      this.knownKeys.add(key);
      this.dropResident(key);
      return { key, deviceId: `vnpu-${nodeId}`, format, bytes, resident: false };
    }
    // Re-staging a key drops its old resident copy first.
    this.dropResident(key);
    const sram = this.sramFor(nodeId);
    // Evict LRU residents until the new weight fits.
    while (sram.usedBytes + bytes > this.sramBytes && sram.entries.size > 0) {
      let lruKey: string | null = null;
      let lruUsed = Infinity;
      for (const [k, e] of sram.entries) {
        if (e.lastUsed < lruUsed) {
          lruUsed = e.lastUsed;
          lruKey = k;
        }
      }
      if (lruKey === null) break;
      const evicted = sram.entries.get(lruKey);
      sram.entries.delete(lruKey);
      if (evicted !== undefined) sram.usedBytes -= evicted.bytes;
      this.keyDevice.delete(lruKey);
      this.sramEvictions += 1;
    }
    const entry: SramEntry = {
      key,
      format,
      rows,
      cols,
      bytes,
      payload,
      lastUsed: ++this.clock,
    };
    sram.entries.set(key, entry);
    sram.usedBytes += bytes;
    this.keyDevice.set(key, nodeId);
    this.knownKeys.add(key);
    return { key, deviceId: `vnpu-${nodeId}`, format, bytes, resident: true };
  }

  async stageFromVram(
    key: string,
    vaddr: VAddr,
    rows: number,
    cols: number,
    dtype: WeightDtype,
    format: NpuWeightFormat,
    opts?: { deviceId?: string; weightScale?: "per-tensor" | "per-channel" },
  ): Promise<NpuStageResult> {
    const bytes = await this.ram.read(vaddr, rows * cols * dtypeByteSize(dtype));
    const values = decodeToFloat32(bytes, dtype);
    return this.stageWeights(key, values, rows, cols, format, opts);
  }

  evictWeights(key: string): boolean {
    return this.dropResident(key);
  }

  resident(key: string): boolean {
    return this.keyDevice.has(key);
  }

  sramStatus(): NpuSramStatus {
    let usedBytes = 0;
    let entries = 0;
    for (const sram of this.deviceSram.values()) {
      usedBytes += sram.usedBytes;
      entries += sram.entries.size;
    }
    return {
      capacityBytes: this.sramBytes,
      usedBytes,
      entries,
      hits: this.sramHits,
      misses: this.sramMisses,
      spills: this.sramSpills,
      evictions: this.sramEvictions,
    };
  }

  /** Remove a key's resident copy from whichever device holds it. */
  private dropResident(key: string): boolean {
    const nodeId = this.keyDevice.get(key);
    if (nodeId === undefined) return false;
    const sram = this.deviceSram.get(nodeId);
    if (sram !== undefined) {
      const entry = sram.entries.get(key);
      if (entry !== undefined) {
        sram.entries.delete(key);
        sram.usedBytes -= entry.bytes;
      }
    }
    this.keyDevice.delete(key);
    return true;
  }

  private sramFor(nodeId: string): DeviceSram {
    let sram = this.deviceSram.get(nodeId);
    if (sram === undefined) {
      sram = { entries: new Map(), usedBytes: 0 };
      this.deviceSram.set(nodeId, sram);
    }
    return sram;
  }

  // -- instruction execution --------------------------------------------------

  /**
   * Run `fn` as one instruction, serialized on `streamId` when given.
   * Mirrors VirtualGpu.launch's stream chains: instructions on one
   * stream execute in order; the stored chain swallows errors so one
   * failed instruction does not block later ones on the stream.
   */
  private async execute<T>(streamId: string | undefined, fn: () => T): Promise<T> {
    const run = (): T => {
      try {
        const out = fn();
        this.opsCompleted += 1;
        return out;
      } catch (err) {
        this.opsFailed += 1;
        throw err;
      }
    };
    if (streamId === undefined) return run();
    const prev = this.streamChains.get(streamId) ?? Promise.resolve();
    const result = prev.then(run);
    this.streamChains.set(
      streamId,
      result.then(
        () => undefined,
        () => undefined,
      ),
    );
    return result;
  }

  /** Count a synchronous instruction (QUANTIZE / elementwise ops). */
  private countedSync<T>(fn: () => T): T {
    try {
      const out = fn();
      this.opsCompleted += 1;
      return out;
    } catch (err) {
      this.opsFailed += 1;
      throw err;
    }
  }

  /**
   * Pick the donor node for an instruction. An explicit deviceId is
   * honored (an unknown/dead device is an error, not a silent
   * reroute). Otherwise the device with the least resident SRAM wins,
   * keeping staged weights balanced across donors.
   */
  private pickDevice(requestedId?: string): string {
    const devices = this.listDevices();
    if (devices.length === 0) {
      throw new Error("vNPU: no healthy donor nodes available");
    }
    if (requestedId !== undefined) {
      const device = devices.find((d) => d.id === requestedId);
      if (device === undefined) {
        throw new Error(`vNPU: unknown or unavailable virtual device ${requestedId}`);
      }
      return device.id.slice("vnpu-".length);
    }
    let best = devices[0] as VirtualNpuDevice;
    let bestUsed = this.deviceSram.get(best.id.slice("vnpu-".length))?.usedBytes ?? 0;
    for (const device of devices.slice(1)) {
      const used = this.deviceSram.get(device.id.slice("vnpu-".length))?.usedBytes ?? 0;
      if (used < bestUsed) {
        best = device;
        bestUsed = used;
      }
    }
    return best.id.slice("vnpu-".length);
  }

  /**
   * Resolve a staged weight for a matmul. Returns the resident entry
   * (counting a hit) or null (counting a miss — the caller falls back
   * to quantizing the passed fp32 B on the fly). A key that was never
   * staged throws: a miss is a cache event, an unknown key is a bug.
   */
  private resolveWeight(key: string, deviceId?: string): SramEntry | null {
    const nodeId = this.keyDevice.get(key);
    if (nodeId === undefined) {
      if (this.knownKeys.has(key)) {
        this.sramMisses += 1;
        return null;
      }
      throw new Error(`vNPU: unknown staged weight ${key} — stage it with stageWeights() first`);
    }
    if (deviceId !== undefined && `vnpu-${nodeId}` !== deviceId) {
      throw new Error(`vNPU: weight ${key} is staged on vnpu-${nodeId}, not ${deviceId}`);
    }
    const entry = this.deviceSram.get(nodeId)?.entries.get(key);
    if (entry === undefined) {
      // The index said resident but the entry is gone: treat as a miss.
      this.keyDevice.delete(key);
      this.sramMisses += 1;
      return null;
    }
    entry.lastUsed = ++this.clock;
    this.sramHits += 1;
    return entry;
  }

  private stagedToInt8(entry: SramEntry): Int8Quantized {
    const p = entry.payload;
    if (p.kind !== "int8") throw new Error(`vNPU: staged weight ${entry.key} is ${p.kind}, not int8`);
    const values = new Float32Array(entry.rows * entry.cols);
    const err = new Float32Array(entry.rows * entry.cols);
    for (let r = 0; r < entry.rows; r++) {
      const s = p.perChannel ? (p.scales[r] as number) : (p.scales[0] as number);
      for (let c = 0; c < entry.cols; c++) {
        const i = r * entry.cols + c;
        values[i] = p.codes[i] * s;
        err[i] = s / 2;
      }
    }
    return {
      kind: "int8",
      codes: p.codes,
      scales: p.scales,
      perChannel: p.perChannel,
      channelAxis: "row",
      values,
      analysis: analyzeQuantization(values, err, entry.rows, entry.cols),
      rows: entry.rows,
      cols: entry.cols,
    };
  }

  private stagedToFp8(entry: SramEntry): Fp8Quantized {
    const p = entry.payload;
    if (p.kind !== "fp8") throw new Error(`vNPU: staged weight ${entry.key} is ${p.kind}, not fp8`);
    const values = decodeToFloat32(p.codes, "float8_e4m3");
    const err = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      values[i] *= p.scale;
      err[i] = FP8_REL_ERR * Math.abs(values[i]) + p.scale * FP8_SUBNORMAL_HALF_STEP;
    }
    return {
      kind: "fp8",
      codes: p.codes,
      scale: p.scale,
      values,
      analysis: analyzeQuantization(values, err, entry.rows, entry.cols),
      rows: entry.rows,
      cols: entry.cols,
    };
  }

  private stagedToBf16(entry: SramEntry): Bf16Quantized {
    const p = entry.payload;
    if (p.kind !== "bf16") throw new Error(`vNPU: staged weight ${entry.key} is ${p.kind}, not bf16`);
    const values = decodeToFloat32(p.bytes, "bfloat16");
    const err = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      err[i] = BF16_REL_ERR * Math.abs(values[i]) + BF16_ABS_ERR;
    }
    return {
      kind: "bf16",
      bytes: p.bytes,
      values,
      analysis: analyzeQuantization(values, err, entry.rows, entry.cols),
      rows: entry.rows,
      cols: entry.cols,
    };
  }

  // -- matmuls ----------------------------------------------------------------

  async matmul(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts: NpuMatmulOptions = {},
  ): Promise<NpuMatmulResult> {
    return this.execute(opts.streamId, () =>
      this.matmulSync(a, aRows, aCols, b, bRows, bCols, opts),
    );
  }

  private matmulSync(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts: NpuMatmulOptions,
  ): NpuMatmulResult {
    if (aCols !== bRows) {
      throw new Error(`vNPU matmul shape mismatch: A is [${aRows},${aCols}], B is [${bRows},${bCols}]`);
    }
    if (a.length !== aRows * aCols || b.length !== bRows * bCols) {
      throw new Error("vNPU matmul: input length does not match shape");
    }
    const staged = opts.weightKey !== undefined ? this.resolveWeight(opts.weightKey, opts.deviceId) : null;
    const format = staged !== null ? staged.format : (opts.format ?? this.defaultFormat);
    if (staged !== null && staged.format !== "int8" && staged.format !== "fp8" && staged.format !== "bf16") {
      throw new Error(`vNPU: weight ${staged.key} is staged as ${staged.format} — use matmulSw`);
    }
    if (staged !== null && (staged.rows !== bRows || staged.cols !== bCols)) {
      throw new Error(
        `vNPU matmul: staged weight ${staged.key} is [${staged.rows},${staged.cols}], B is [${bRows},${bCols}]`,
      );
    }
    checkTileFit(tileWorkingSetMatmul(format), opForFormat(format));
    // A staged per-channel weight is [out, in] (row channels): it can
    // only serve the transposed form. Per-tensor staged weights serve both.
    if (
      staged !== null &&
      staged.payload.kind === "int8" &&
      staged.payload.perChannel
    ) {
      throw new Error(
        `vNPU matmul: staged per-channel weight ${staged.key} is [out, in] orientation — use matmulTransposeB`,
      );
    }
    // Activations are always quantized per-tensor (a per-channel
    // activation scale would need a dynamic max pass per row).
    if (format === "int8") {
      const aQ = quantizeInt8(a, aRows, aCols, "per-tensor", "row");
      const bQ =
        staged !== null
          ? this.stagedToInt8(staged)
          : quantizeInt8(b, bRows, bCols, opts.weightScale ?? "per-channel", "col");
      const out = int8MatmulCore(aQ, bQ, aRows, bCols, aCols);
      this.int8Matmuls += 1;
      return {
        output: out,
        errorBound: boundMatmul(aQ.analysis, bQ.analysis, aRows, bCols, aCols),
        sramHit: staged !== null,
        format: "int8",
        op: "MATMUL_INT8",
        device: "cpu",
      };
    }
    if (format === "fp8") {
      const aQ = quantizeFp8(a, aRows, aCols);
      const bQ = staged !== null ? this.stagedToFp8(staged) : quantizeFp8(b, bRows, bCols);
      const out = matmul(aQ.values, aRows, aCols, bQ.values, bRows, bCols);
      this.fp8Matmuls += 1;
      return {
        output: out,
        errorBound: boundMatmul(aQ.analysis, bQ.analysis, aRows, bCols, aCols),
        sramHit: staged !== null,
        format: "fp8",
        op: "MATMUL_FP8",
        device: "cpu",
      };
    }
    const aQ = quantizeBf16(a, aRows, aCols);
    const bQ = staged !== null ? this.stagedToBf16(staged) : quantizeBf16(b, bRows, bCols);
    const out = matmul(aQ.values, aRows, aCols, bQ.values, bRows, bCols);
    this.bf16Matmuls += 1;
    return {
      output: out,
      errorBound: boundMatmul(aQ.analysis, bQ.analysis, aRows, bCols, aCols),
      sramHit: staged !== null,
      format: "bf16",
      op: "MATMUL_BF16",
      device: "cpu",
    };
  }

  async matmulTransposeB(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts: NpuMatmulOptions = {},
  ): Promise<NpuMatmulResult> {
    return this.execute(opts.streamId, () =>
      this.matmulTransposeBSync(a, aRows, aCols, b, bRows, bCols, opts),
    );
  }

  private matmulTransposeBSync(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts: NpuMatmulOptions,
  ): NpuMatmulResult {
    if (aCols !== bCols) {
      throw new Error(
        `vNPU matmulTransposeB shape mismatch: A is [${aRows},${aCols}], B is [${bRows},${bCols}]`,
      );
    }
    if (a.length !== aRows * aCols || b.length !== bRows * bCols) {
      throw new Error("vNPU matmulTransposeB: input length does not match shape");
    }
    const staged = opts.weightKey !== undefined ? this.resolveWeight(opts.weightKey, opts.deviceId) : null;
    const format = staged !== null ? staged.format : (opts.format ?? this.defaultFormat);
    if (staged !== null && staged.format !== "int8" && staged.format !== "fp8" && staged.format !== "bf16") {
      throw new Error(`vNPU: weight ${staged.key} is staged as ${staged.format} — use matmulSw`);
    }
    if (staged !== null && (staged.rows !== bRows || staged.cols !== bCols)) {
      throw new Error(
        `vNPU matmulTransposeB: staged weight ${staged.key} is [${staged.rows},${staged.cols}], B is [${bRows},${bCols}]`,
      );
    }
    checkTileFit(tileWorkingSetMatmul(format), opForFormat(format));
    if (format === "int8") {
      const aQ = quantizeInt8(a, aRows, aCols, "per-tensor", "row");
      const bQ =
        staged !== null
          ? this.stagedToInt8(staged)
          : quantizeInt8(b, bRows, bCols, opts.weightScale ?? "per-channel", "row");
      const out = int8MatmulTransposeBCore(aQ, bQ, aRows, bRows, aCols);
      this.int8Matmuls += 1;
      return {
        output: out,
        errorBound: boundTransposeB(aQ.analysis, bQ.analysis, aRows, bRows, aCols),
        sramHit: staged !== null,
        format: "int8",
        op: "MATMUL_INT8",
        device: "cpu",
      };
    }
    if (format === "fp8") {
      const aQ = quantizeFp8(a, aRows, aCols);
      const bQ = staged !== null ? this.stagedToFp8(staged) : quantizeFp8(b, bRows, bCols);
      const out = matmulTransposeB(aQ.values, aRows, aCols, bQ.values, bRows, bCols);
      this.fp8Matmuls += 1;
      return {
        output: out,
        errorBound: boundTransposeB(aQ.analysis, bQ.analysis, aRows, bRows, aCols),
        sramHit: staged !== null,
        format: "fp8",
        op: "MATMUL_FP8",
        device: "cpu",
      };
    }
    const aQ = quantizeBf16(a, aRows, aCols);
    const bQ = staged !== null ? this.stagedToBf16(staged) : quantizeBf16(b, bRows, bCols);
    const out = matmulTransposeB(aQ.values, aRows, aCols, bQ.values, bRows, bCols);
    this.bf16Matmuls += 1;
    return {
      output: out,
      errorBound: boundTransposeB(aQ.analysis, bQ.analysis, aRows, bRows, aCols),
      sramHit: staged !== null,
      format: "bf16",
      op: "MATMUL_BF16",
      device: "cpu",
    };
  }

  async matmulSw(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: PackedTensor,
    opts: NpuSwMatmulOptions = {},
  ): Promise<NpuMatmulResult> {
    return this.execute(opts.streamId, () => this.matmulSwSync(a, aRows, aCols, b, opts));
  }

  private matmulSwSync(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: PackedTensor,
    opts: NpuSwMatmulOptions,
  ): NpuMatmulResult {
    if (a.length !== aRows * aCols) {
      throw new Error(`vNPU matmulSw: A length ${a.length} != ${aRows}x${aCols}`);
    }
    let packed = b;
    let sramHit = false;
    if (opts.weightKey !== undefined) {
      const staged = this.resolveWeight(opts.weightKey, opts.deviceId);
      if (staged !== null) {
        if (staged.payload.kind !== "sw") {
          throw new Error(`vNPU matmulSw: weight ${staged.key} is staged as ${staged.format}, not SW-code`);
        }
        if (staged.rows !== b.rows || staged.cols !== b.cols) {
          throw new Error(
            `vNPU matmulSw: staged weight ${staged.key} is [${staged.rows},${staged.cols}], B is [${b.rows},${b.cols}]`,
          );
        }
        packed = staged.payload.packed;
        sramHit = true;
      }
    }
    checkTileFit(tileWorkingSetMatmul(packed.mode), "MATMUL_SW");
    const out = matmulPackedBT(a, aRows, aCols, packed);
    // Theorem N1, dot-product form: |a_i . b_j - a_i . b_hat_j| <=
    // ||a_i||_2 * E_j, so the bound is max_i ||a_i|| * max_j E_j.
    let aNormMax = 0;
    for (let i = 0; i < aRows; i++) {
      let s = 0;
      for (let k = 0; k < aCols; k++) s += a[i * aCols + k] * a[i * aCols + k];
      const n = Math.sqrt(s);
      if (n > aNormMax) aNormMax = n;
    }
    let eMax = 0;
    for (let j = 0; j < packed.rows; j++) {
      const e = rowErrorNormBound(packed, j);
      if (e > eMax) eMax = e;
    }
    this.swMatmuls += 1;
    return {
      output: out,
      errorBound: aNormMax * eMax,
      sramHit,
      format: packed.mode,
      op: "MATMUL_SW",
      device: "cpu",
    };
  }

  // -- attention ---------------------------------------------------------------

  async attention(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts: NpuAttentionOptions & { stats?: BlockStats } = {},
  ): Promise<NpuAttentionResult> {
    return this.execute(opts.streamId, () =>
      this.attentionSync(q, k, v, numHeads, numKvHeads, headDim, opts),
    );
  }

  private attentionSync(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts: NpuAttentionOptions & { stats?: BlockStats },
  ): NpuAttentionResult {
    const mode: NpuAttentionMode = opts.mode ?? "exact";
    const op: NpuOp = mode === "landmark" ? "ATTN_LANDMARK" : mode === "sparse" ? "ATTN_SPARSE" : "ATTN_EXACT";
    const causal = opts.causal ?? true;
    const kvLen = k.length / (numKvHeads * headDim);
    const visLen = visibleLength(kvLen, causal, opts.kvPositions, opts.queryPosition);
    const blockSize = opts.blockSize ?? Math.max(4, Math.ceil(Math.sqrt(Math.max(visLen, 1))));
    checkTileFit(
      tileWorkingSetAttention(blockSize, numHeads, numKvHeads, headDim),
      `${op} (blockSize ${blockSize})`,
    );
    const base = {
      scale: opts.scale,
      causal,
      queryPosition: opts.queryPosition,
      kvPositions: opts.kvPositions,
      localWindow: opts.localWindow,
      blockSize,
    };
    let result;
    if (mode === "landmark") {
      result = landmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
        ...base,
        deltaMax: opts.deltaMax,
        stats: opts.stats,
      });
    } else if (mode === "sparse") {
      result = sparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
        ...base,
        dropThreshold: opts.dropThreshold,
        stats: opts.stats,
      });
    } else {
      const output = groupedQueryAttention(q, k, v, numHeads, numKvHeads, headDim, {
        scale: opts.scale,
        causal,
        queryPosition: opts.queryPosition,
        kvPositions: opts.kvPositions,
      });
      result = {
        output,
        ops: 2 * visLen * headDim * numHeads,
        errorBound: 0,
        droppedMassBound: 0,
        exactOps: 2 * visLen * headDim * numHeads,
      };
    }
    this.attentionCalls += 1;
    return { ...result, mode, op, device: "cpu" };
  }

  async prefillAttention(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts: NpuAttentionOptions & { stats?: BlockStats } = {},
  ): Promise<NpuAttentionResult> {
    return this.execute(opts.streamId, () =>
      this.prefillAttentionSync(q, k, v, numHeads, numKvHeads, headDim, opts),
    );
  }

  private prefillAttentionSync(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts: NpuAttentionOptions & { stats?: BlockStats },
  ): NpuAttentionResult {
    const mode: NpuAttentionMode = opts.mode ?? "exact";
    const op: NpuOp = mode === "landmark" ? "ATTN_LANDMARK" : mode === "sparse" ? "ATTN_SPARSE" : "ATTN_EXACT";
    const seqLen = q.length / (numHeads * headDim);
    const blockSize = opts.blockSize ?? Math.max(4, Math.ceil(Math.sqrt(Math.max(seqLen, 1))));
    checkTileFit(
      tileWorkingSetAttention(blockSize, numHeads, numKvHeads, headDim),
      `${op} prefill (blockSize ${blockSize})`,
    );
    const base = {
      scale: opts.scale,
      localWindow: opts.localWindow,
      blockSize,
    };
    let result;
    if (mode === "landmark") {
      result = prefillLandmarkAttention(q, k, v, numHeads, numKvHeads, headDim, {
        ...base,
        deltaMax: opts.deltaMax,
        stats: opts.stats,
      });
    } else if (mode === "sparse") {
      result = prefillSparseAttention(q, k, v, numHeads, numKvHeads, headDim, {
        ...base,
        dropThreshold: opts.dropThreshold,
        stats: opts.stats,
      });
    } else {
      const causal = opts.causal ?? true;
      const output = CpuKernels.attention(q, k, v, {
        numHeads,
        headDim,
        numKvHeads,
        causal,
      });
      const exactOps = numHeads * headDim * seqLen * (seqLen + 1);
      result = { output, ops: exactOps, errorBound: 0, droppedMassBound: 0, exactOps };
    }
    this.attentionCalls += 1;
    return { ...result, mode, op, device: "cpu" };
  }

  async attentionPackedKv(
    q: Float32Array,
    kv: PackedKV,
    numHeads: number,
    opts: NpuPackedKvAttentionOptions = {},
  ): Promise<NpuAttentionResult> {
    return this.execute(opts.streamId, () => {
      const output = attentionPackedKV(q, kv, numHeads, {
        scale: opts.scale,
        causal: opts.causal,
        queryPosition: opts.queryPosition,
        kvPositions: opts.kvPositions,
      });
      // Theorem N2 bound, max over heads.
      let errorBound = 0;
      for (let h = 0; h < numHeads; h++) {
        const b = kvAttentionErrorBound(q, kv, numHeads, h, {
          scale: opts.scale,
          causal: opts.causal,
          queryPosition: opts.queryPosition,
          kvPositions: opts.kvPositions,
        });
        if (b > errorBound) errorBound = b;
      }
      const exactOps = 2 * kv.kvLen * kv.headDim * numHeads;
      this.attentionCalls += 1;
      return {
        output,
        ops: exactOps,
        errorBound,
        droppedMassBound: 0,
        exactOps,
        mode: "exact" as NpuAttentionMode,
        op: "ATTN_PACKED_KV" as NpuOp,
        device: "cpu" as WorkerDevice,
      };
    });
  }

  /**
   * Build the A1/A2 block statistics for a KV cache — the stats pass
   * the landmark/sparse instructions consume. In the device model
   * these statistics are device-resident state in tile SRAM, built
   * once per KV cache and reused per query (docs/VNPU.md §1); in this
   * reference they are returned to the caller, who passes them back
   * via the attention `stats` option.
   */
  buildBlockStats(
    k: Float32Array,
    v: Float32Array,
    numKvHeads: number,
    headDim: number,
    blockSize?: number,
  ): { stats: BlockStats; ops: number } {
    return this.countedSync(() => {
      const built = computeBlockStats(
        k,
        v,
        numKvHeads,
        headDim,
        blockSize ?? 32,
      );
      return { stats: built.stats, ops: built.ops };
    });
  }

  /** Extend block statistics to a longer KV cache (O(B*d), not O(n*d)). */
  extendBlockStats(
    stats: BlockStats,
    k: Float32Array,
    v: Float32Array,
    newKvLen: number,
  ): { stats: BlockStats; ops: number } {
    return this.countedSync(() => {
      const extended = extendBlockStats(stats, k, v, newKvLen);
      return { stats: extended.stats, ops: extended.ops };
    });
  }

  // -- SW-code native format ----------------------------------------------------

  quantizeSw(
    data: Float32Array,
    rows: number,
    cols: number,
    mode: SwMode,
    opts: { blockSize?: number } = {},
  ): PackedTensor {
    return this.countedSync(() => encodeMatrix(data, rows, cols, mode, opts));
  }

  dequantizeSw(packed: PackedTensor): Float32Array {
    return this.countedSync(() => decodeMatrix(packed));
  }

  quantizeKv(
    keys: Float32Array,
    values: Float32Array,
    kvLen: number,
    numKvHeads: number,
    headDim: number,
    mode: SwMode | { k: SwMode; v: SwMode },
  ): PackedKV {
    return this.countedSync(() => {
      const kv = quantizeKV(keys, values, kvLen, numKvHeads, headDim, mode);
      this.kvQuantizeCalls += 1;
      return kv;
    });
  }

  // -- elementwise instructions ---------------------------------------------------

  softmax(x: Float32Array, rows: number, cols: number): Float32Array {
    return this.countedSync(() => CpuKernels.softmaxRows(Float32Array.from(x), rows, cols));
  }

  rmsNorm(x: Float32Array, rows: number, cols: number, weight: Float32Array, eps: number): Float32Array {
    return this.countedSync(() => CpuKernels.rmsNorm(x, rows, cols, weight, eps));
  }

  silu(x: Float32Array): Float32Array {
    return this.countedSync(() => CpuKernels.silu(x));
  }

  // -- inventory / healing ---------------------------------------------------------

  stats(): NpuStats {
    const devices = this.listDevices();
    let sramUsedBytes = 0;
    const deviceStats = devices.map((d) => {
      const nodeId = d.id.slice("vnpu-".length);
      const used = this.deviceSram.get(nodeId)?.usedBytes ?? 0;
      sramUsedBytes += used;
      return { nodeId, device: d.backedBy, sramBytes: d.sramBytes, sramUsedBytes: used };
    });
    return {
      devices: deviceStats,
      sramCapacityBytes: this.sramBytes,
      sramUsedBytes,
      sramHits: this.sramHits,
      sramMisses: this.sramMisses,
      sramSpills: this.sramSpills,
      sramEvictions: this.sramEvictions,
      int8Matmuls: this.int8Matmuls,
      fp8Matmuls: this.fp8Matmuls,
      swMatmuls: this.swMatmuls,
      bf16Matmuls: this.bf16Matmuls,
      attentionCalls: this.attentionCalls,
      kvQuantizeCalls: this.kvQuantizeCalls,
      opsCompleted: this.opsCompleted,
      opsFailed: this.opsFailed,
    };
  }

  async handleNodeLoss(nodeId: string): Promise<{ devicesLost: number; sramBytesLost: number }> {
    const known =
      this.seenNodes.has(nodeId) ||
      this.nodes.listNodes().some((n: WorkerInfo) => n.workerId === nodeId);
    const devicesLost = known && !this.deadNodes.has(nodeId) ? 1 : 0;
    this.deadNodes.add(nodeId);
    // The device's SRAM dies with it: its resident weights are gone.
    // Their keys stay known, so a later use counts a miss and streams
    // the weight from vRAM instead of throwing.
    const sram = this.deviceSram.get(nodeId);
    const sramBytesLost = sram?.usedBytes ?? 0;
    if (sram !== undefined) {
      for (const key of sram.entries.keys()) this.keyDevice.delete(key);
      this.deviceSram.delete(nodeId);
    }
    return { devicesLost, sramBytesLost };
  }
}
