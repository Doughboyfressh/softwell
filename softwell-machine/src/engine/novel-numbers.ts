/**
 * softwell-machine — SW-code: a novel low-bit number system for weights,
 * activations, and KV cache.
 *
 * Governing principle (see docs/research/00-survey.md): a virtual machine
 * cannot exceed its hosts' physical FLOPS or bytes/second, so it wins only
 * by moving fewer bytes per token. The engine today moves every weight as
 * float32 (the loader dequantizes BF16/FP16/FP8 to Float32Array on load),
 * so the bytes that cross vRAM -> vGPU per decode token are 4 per
 * parameter plus 4 per KV element. SW-code cuts that to ~0.30-0.55 bytes
 * per element, with a *provable* per-element and per-dot-product error
 * bound (Theorem N1, docs/research/02-numbers.md) and a provable bound
 * on the attention output when the KV cache is quantized (Theorem N2).
 *
 * # The format
 *
 * A matrix is split into blocks of `blockSize` consecutive elements along
 * each row (default 64). Each block is encoded independently as:
 *
 *   [ codes: ceil(n * b / 8) bytes ][ scale: fp16, 2 bytes ]
 *   [ outlier count: 1 byte ][ outliers: 3 bytes each ]
 *
 * where b in {2, 3, 4} is the code width (modes "sw2", "sw3", "sw4") and
 * each code is one of 2L+1 uniform levels {-L, ..., +L} * delta
 * (L = 1, 3, 7), plus one escape code. The block scale (called the *clip*
 * level Q) is stored as fp16 and delta = Q / L.
 *
 * The twist is how Q is chosen. Almost every block quantizer sets the
 * scale from the block *maximum*, so one huge weight poisons the step size
 * for the other 63 values. SW-code instead sets the scale from a *robust
 * quantile* and stores the tail exactly:
 *
 *   Let a_(1) >= ... >= a_(n) be the sorted |x| in the block and
 *   r = ceil(n / 8). Then Q = min(a_(1), 2 * a_(r)).
 *   Outliers are the elements with |x| > Q; each is stored exactly as
 *   fp16 (1 index byte + 2 value bytes).
 *
 * Because Q >= a_(r), at most r-1 elements can exceed Q, so the outlier
 * count is bounded by construction (<= 7 for a 64-block) and the
 * worst-case bytes per element is bounded too. A block whose values are
 * all equal, all zero, or all tiny is handled by the same rule with no
 * special cases: if Q = 0 every code is 0; if the fp16 scale underflows
 * to 0 the non-outliers decode to 0 with error < 2^-24.
 *
 * Consequences:
 *  - Non-outlier error is at most delta / 2 (round to nearest).
 *  - Outlier error is at most the fp16 rounding error, 2^-10 * |value|
 *    + 2^-24 (the 2^-10, not 2^-11, absorbs converting between the true
 *    value and the stored fp16 value; see the proof in 02-numbers.md).
 *  - There is no tail term: an outlier can never blow up the bound,
 *    because it is never quantized on the coarse grid at all.
 *
 * # Access pattern
 *
 * Blocks are variable-length (outlier count varies), so a packed tensor
 * is a single byte stream plus one 4-byte row offset per row. Sequential
 * access (full decode, matmul) is O(1) amortized per block with a
 * streaming cursor; random single-element access would need a prefix
 * scan and is deliberately not offered. The matmuls below read the
 * packed stream directly: each block is unpacked into a small lookup
 * table (code -> value) and dotted against float32 activations without
 * ever materializing a float32 copy of the weight matrix. The traffic
 * saving is therefore realized at the vRAM -> vGPU transfer, not just
 * on disk.
 *
 * # Input domain
 *
 * Inputs must be finite with |x| <= 65504 (the fp16 maximum): every
 * value that can be stored as an outlier must fit in fp16, and the
 * block scale is stored as fp16. Real LLM weights, activations, and
 * K/V values are orders of magnitude inside this range; the encoder
 * throws a clear error outside it instead of silently corrupting data.
 *
 * # Status
 *
 * New parallel path, not wired into the loader or engine yet: encode a
 * tensor with encodeMatrix / quantizeKV, run matmulPackedBT /
 * matmulPackedAB / attentionPackedKV, and decode with decodeMatrix.
 * The existing loader and engine are untouched.
 */

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

export type SwMode = "sw2" | "sw3" | "sw4";

export interface SwModeInfo {
  mode: SwMode;
  /** Code width in bits. */
  bits: number;
  /** Levels L: codes represent {-L, ..., +L} * delta. */
  levels: number;
  /** Largest non-escape code (2L). */
  maxCode: number;
  /** Escape code (2L+1): value is in the outlier stream. */
  escapeCode: number;
}

/** Static facts about a mode. Throws on an unknown mode string. */
export function swModeInfo(mode: SwMode): SwModeInfo {
  switch (mode) {
    case "sw2":
      return { mode, bits: 2, levels: 1, maxCode: 2, escapeCode: 3 };
    case "sw3":
      return { mode, bits: 3, levels: 3, maxCode: 6, escapeCode: 7 };
    case "sw4":
      return { mode, bits: 4, levels: 7, maxCode: 14, escapeCode: 15 };
    default:
      throw new Error(`unknown SW mode: ${String(mode)}`);
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default block size (elements per block along a row). */
export const SW_DEFAULT_BLOCK_SIZE = 64;
/**
 * Maximum block size. Outlier indices are stored in one byte, so a block
 * may hold at most 256 elements.
 */
export const SW_MAX_BLOCK_SIZE = 256;
/** Clip level Q = min(max|x|, KAPPA * a_(r)); see the header. */
export const SW_OUTLIER_KAPPA = 2;
/** Largest finite fp16 value; the encoder rejects |x| above this. */
export const SW_FP16_MAX = 65504;
/** Safe bound on |x - fp16(x)| as FP16_REL * |stored| + FP16_ABS. */
export const SW_FP16_REL_ERR = 2 ** -10;
export const SW_FP16_ABS_ERR = 2 ** -24;
/** Smallest positive fp16 subnormal. A block whose fp16 scale rounds to
 *  0 has clip Q < this, so its non-outlier error is below it. */
export const SW_FP16_MIN_SUBNORMAL = 2 ** -24;
/** Row offsets are Uint32, so one packed tensor is capped at 4 GiB
 *  (~8B parameters at SW3). A production build would use 64-bit offsets;
 *  the encoder throws instead of wrapping. */
export const SW_MAX_PACKED_BYTES = 2 ** 32;

// ---------------------------------------------------------------------------
// fp16 bit conversion (no native deps; exact for all finite values)
// ---------------------------------------------------------------------------

const _f32bits = new Float32Array(1);
const _u32bits = new Uint32Array(_f32bits.buffer);

/**
 * Float32 -> fp16 bits, round to nearest even. Returns the 16-bit
 * pattern as a number in [0, 65535]. Non-finite inputs map to fp16
 * Inf (NaN keeps a payload bit); callers that need an error should
 * validate first.
 */
export function floatToFp16Bits(x: number): number {
  _f32bits[0] = x;
  const f = _u32bits[0];
  const sign = (f >>> 16) & 0x8000;
  const expField = (f >>> 23) & 0xff;
  const mant = f & 0x7fffff;
  if (expField === 0xff) {
    return sign | 0x7c00 | (mant === 0 ? 0 : 0x0200);
  }
  const e = expField - 127; // unbiased exponent
  if (e >= 16) return sign | 0x7c00; // overflow -> Inf
  if (e >= -14) {
    // Normal fp16: round the 23-bit mantissa to 10 bits, ties to even.
    const m = mant + 0xfff + ((mant >>> 13) & 1);
    if (m >= 0x800000) return sign | ((e + 16) << 10); // mantissa carry bumps the exponent
    return sign | ((e + 15) << 10) | (m >>> 13);
  }
  if (e >= -25) {
    // Subnormal fp16: value = m * 2^-24, m = round((2^23 + mant) * 2^(e+1)).
    const shift = -e - 1; // 14..24
    const full = 0x800000 + mant;
    const m = (full + (1 << (shift - 1))) >>> shift;
    return sign | m;
  }
  return sign; // underflows to signed zero
}

/** fp16 bits -> float32 value. Exact: every fp16 value is exactly representable in fp32. */
export function fp16BitsToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24; // zero or subnormal
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

// ---------------------------------------------------------------------------
// Packed tensor
// ---------------------------------------------------------------------------

export interface PackedTensor {
  mode: SwMode;
  rows: number;
  cols: number;
  blockSize: number;
  /**
   * Concatenated per-row block streams. One block is:
   * [codes: ceil(n*b/8) bytes][scale fp16 LE: 2 bytes]
   * [outlier count: 1 byte][outliers: 3 bytes each (index, fp16 LE)].
   */
  data: Uint8Array;
  /** Byte offset of each row's stream start; length rows+1, last entry == data.length. */
  rowOffsets: Uint32Array;
  /** Total number of blocks (rows * ceil(cols / blockSize)). */
  numBlocks: number;
  /** Total number of escaped outliers across all blocks. */
  outlierTotal: number;
}

export interface EncodeOptions {
  /** Elements per block along a row; default 64, must be in [1, 256]. */
  blockSize?: number;
}

export interface PackedDescription {
  mode: SwMode;
  rows: number;
  cols: number;
  blockSize: number;
  numBlocks: number;
  outlierTotal: number;
  outlierFraction: number;
  bytes: number;
  bytesPerElement: number;
  bitsPerElement: number;
  /** Bytes of the equivalent BF16 tensor divided by packed bytes. */
  ratioVsBf16: number;
  /** Bytes of the equivalent FP32 tensor divided by packed bytes. */
  ratioVsFp32: number;
}

// --- bit packing (LSB-first) -------------------------------------------------

function packCodes(codes: Uint8Array, n: number, bits: number, out: Uint8Array, pos: number): number {
  let acc = 0;
  let accBits = 0;
  for (let i = 0; i < n; i++) {
    acc |= codes[i] << accBits;
    accBits += bits;
    while (accBits >= 8) {
      out[pos++] = acc & 0xff;
      acc >>>= 8;
      accBits -= 8;
    }
  }
  if (accBits > 0) out[pos++] = acc & 0xff;
  return pos;
}

function unpackCodes(data: Uint8Array, byteOffset: number, n: number, bits: number, out: Uint8Array): void {
  const mask = (1 << bits) - 1;
  let acc = 0;
  let accBits = 0;
  let pos = byteOffset;
  for (let i = 0; i < n; i++) {
    while (accBits < bits) {
      acc |= data[pos++] << accBits;
      accBits += 8;
    }
    out[i] = acc & mask;
    acc >>>= bits;
    accBits -= bits;
  }
}

// --- block scratch: one parsed block ------------------------------------------

interface BlockScratch {
  codes: Uint8Array;
  /** lut[code] = (code - L) * delta for code <= 2L; lut[escape] = 0. */
  lut: Float32Array;
  outIdx: Uint8Array;
  outVal: Float32Array;
  blockLen: number;
  /** Decoded scale (clip level Q) for this block. */
  scale: number;
  /** Step size delta = scale / L (0 when the scale underflowed). */
  delta: number;
  outCount: number;
}

function makeBlockScratch(blockSize: number, info: SwModeInfo): BlockScratch {
  return {
    codes: new Uint8Array(blockSize),
    lut: new Float32Array(2 * info.levels + 2),
    outIdx: new Uint8Array(blockSize),
    outVal: new Float32Array(blockSize),
    blockLen: 0,
    scale: 0,
    delta: 0,
    outCount: 0,
  };
}

/**
 * Parse one block starting at `byteOffset` into `scratch`; returns the
 * byte offset just past the block. Throws on truncated or inconsistent
 * data (a corrupt packed tensor is a hard error, never silent garbage).
 */
function readBlock(
  packed: PackedTensor,
  byteOffset: number,
  blockLen: number,
  info: SwModeInfo,
  scratch: BlockScratch,
): number {
  const { data } = packed;
  const codeBytes = Math.ceil((blockLen * info.bits) / 8);
  if (byteOffset < 0 || byteOffset + codeBytes + 3 > data.length) {
    throw new Error("corrupt packed tensor: truncated block");
  }
  unpackCodes(data, byteOffset, blockLen, info.bits, scratch.codes);
  let p = byteOffset + codeBytes;
  const scaleBits = data[p] | (data[p + 1] << 8);
  const count = data[p + 2];
  p += 3;
  if (count > blockLen || p + 3 * count > data.length) {
    throw new Error("corrupt packed tensor: bad outlier count");
  }
  const scale = fp16BitsToFloat(scaleBits);
  const delta = scale / info.levels;
  const lut = scratch.lut;
  for (let code = 0; code <= 2 * info.levels; code++) lut[code] = (code - info.levels) * delta;
  lut[info.escapeCode] = 0;
  for (let t = 0; t < count; t++) {
    const idx = data[p];
    if (idx >= blockLen) throw new Error("corrupt packed tensor: outlier index out of block");
    scratch.outIdx[t] = idx;
    scratch.outVal[t] = fp16BitsToFloat(data[p + 1] | (data[p + 2] << 8));
    p += 3;
  }
  scratch.blockLen = blockLen;
  scratch.scale = scale;
  scratch.delta = delta;
  scratch.outCount = count;
  return p;
}

function validatePacked(packed: PackedTensor): void {
  if (!Number.isInteger(packed.rows) || packed.rows < 1) {
    throw new Error(`invalid packed tensor: rows ${packed.rows}`);
  }
  if (!Number.isInteger(packed.cols) || packed.cols < 1) {
    throw new Error(`invalid packed tensor: cols ${packed.cols}`);
  }
  if (!Number.isInteger(packed.blockSize) || packed.blockSize < 1 || packed.blockSize > SW_MAX_BLOCK_SIZE) {
    throw new Error(`invalid packed tensor: blockSize ${packed.blockSize}`);
  }
  if (packed.rowOffsets.length !== packed.rows + 1) {
    throw new Error("invalid packed tensor: rowOffsets length mismatch");
  }
  if (packed.rowOffsets[0] !== 0 || packed.rowOffsets[packed.rows] !== packed.data.length) {
    throw new Error("invalid packed tensor: rowOffsets do not span data");
  }
  for (let r = 0; r < packed.rows; r++) {
    if (packed.rowOffsets[r + 1] < packed.rowOffsets[r]) {
      throw new Error("invalid packed tensor: rowOffsets not monotonic");
    }
  }
  swModeInfo(packed.mode); // throws on an unknown mode
}

// --- encode / decode ---------------------------------------------------------

/**
 * Encode a row-major [rows, cols] Float32Array into SW-code.
 * The encoder is offline work (done once at load); it sorts each block's
 * |values| to find the robust clip level, so it is O(n log blockSize).
 * Throws on non-finite input, |x| > 65504, or a bad shape/blockSize.
 */
export function encodeMatrix(
  data: Float32Array,
  rows: number,
  cols: number,
  mode: SwMode,
  options: EncodeOptions = {},
): PackedTensor {
  const info = swModeInfo(mode);
  const blockSize = options.blockSize ?? SW_DEFAULT_BLOCK_SIZE;
  if (!Number.isInteger(rows) || rows < 1) throw new Error(`encodeMatrix: rows must be >= 1, got ${rows}`);
  if (!Number.isInteger(cols) || cols < 1) throw new Error(`encodeMatrix: cols must be >= 1, got ${cols}`);
  if (data.length !== rows * cols) {
    throw new Error(`encodeMatrix: data length ${data.length} != rows*cols ${rows * cols}`);
  }
  if (!Number.isInteger(blockSize) || blockSize < 1 || blockSize > SW_MAX_BLOCK_SIZE) {
    throw new Error(`encodeMatrix: blockSize must be in [1, ${SW_MAX_BLOCK_SIZE}], got ${blockSize}`);
  }
  const numBlocksPerRow = Math.ceil(cols / blockSize);
  // Worst-case byte size (every block at its max outlier count). The
  // outlier count of a block is at most ceil(n/8)-1 by construction.
  let worstBytes = 0;
  for (let b = 0; b < numBlocksPerRow; b++) {
    const n = Math.min(blockSize, cols - b * blockSize);
    worstBytes += Math.ceil((n * info.bits) / 8) + 3 + 3 * (Math.ceil(n / 8) - 1);
  }
  worstBytes *= rows;
  if (worstBytes >= SW_MAX_PACKED_BYTES) {
    throw new Error(
      `encodeMatrix: packed data would exceed the 4 GiB row-offset cap (${worstBytes} bytes); ` +
        `split the tensor or use 64-bit row offsets`,
    );
  }
  const out = new Uint8Array(worstBytes);
  const rowOffsets = new Uint32Array(rows + 1);
  const absScratch = new Float64Array(blockSize);
  const codesScratch = new Uint8Array(blockSize);
  const outIdxScratch = new Uint8Array(blockSize);
  const outBitsScratch = new Uint16Array(blockSize);
  let pos = 0;
  let outlierTotal = 0;
  let numBlocks = 0;
  for (let r = 0; r < rows; r++) {
    rowOffsets[r] = pos;
    const rowBase = r * cols;
    for (let b = 0; b < numBlocksPerRow; b++) {
      const k0 = b * blockSize;
      const n = Math.min(blockSize, cols - k0);
      for (let i = 0; i < n; i++) {
        const ax = Math.abs(data[rowBase + k0 + i]);
        if (!Number.isFinite(ax) || ax > SW_FP16_MAX) {
          throw new Error(
            `encodeMatrix: value at [${r}, ${k0 + i}] is ${data[rowBase + k0 + i]}; ` +
              `SW-code requires finite values with |x| <= ${SW_FP16_MAX}`,
          );
        }
        absScratch[i] = ax;
      }
      // TypedArray sort is numeric ascending; the top of the block is at the end.
      const sorted = absScratch.subarray(0, n);
      sorted.sort();
      const a1 = sorted[n - 1];
      const rank = Math.ceil(n / 8);
      const q0 = sorted[n - rank];
      const clip = Math.min(a1, SW_OUTLIER_KAPPA * q0);
      const scaleBits = floatToFp16Bits(clip);
      const scale = fp16BitsToFloat(scaleBits);
      const delta = scale / info.levels;
      let outCount = 0;
      for (let i = 0; i < n; i++) {
        const x = data[rowBase + k0 + i];
        if (Math.abs(x) > clip) {
          outIdxScratch[outCount] = i;
          outBitsScratch[outCount] = floatToFp16Bits(x);
          outCount++;
          codesScratch[i] = info.escapeCode;
        } else if (delta === 0) {
          codesScratch[i] = info.levels; // decodes to 0
        } else {
          let q = Math.round(x / delta);
          if (q > info.levels) q = info.levels;
          if (q < -info.levels) q = -info.levels;
          codesScratch[i] = q + info.levels;
        }
      }
      pos = packCodes(codesScratch, n, info.bits, out, pos);
      out[pos++] = scaleBits & 0xff;
      out[pos++] = (scaleBits >> 8) & 0xff;
      out[pos++] = outCount;
      for (let t = 0; t < outCount; t++) {
        out[pos++] = outIdxScratch[t];
        const vb = outBitsScratch[t];
        out[pos++] = vb & 0xff;
        out[pos++] = (vb >> 8) & 0xff;
      }
      outlierTotal += outCount;
      numBlocks++;
    }
  }
  rowOffsets[rows] = pos;
  return {
    mode,
    rows,
    cols,
    blockSize,
    data: out.slice(0, pos),
    rowOffsets,
    numBlocks,
    outlierTotal,
  };
}

/** Decode a whole packed tensor to a row-major Float32Array. */
export function decodeMatrix(packed: PackedTensor): Float32Array {
  validatePacked(packed);
  const out = new Float32Array(packed.rows * packed.cols);
  const info = swModeInfo(packed.mode);
  const scratch = makeBlockScratch(packed.blockSize, info);
  let o = 0;
  for (let r = 0; r < packed.rows; r++) {
    decodeRowInto(packed, r, out, o, info, scratch);
    o += packed.cols;
  }
  return out;
}

/**
 * Decode one row into `out` at `outOffset` (out must hold at least
 * outOffset + packed.cols floats). Reuses a caller scratch when given.
 */
export function decodeRowInto(
  packed: PackedTensor,
  row: number,
  out: Float32Array,
  outOffset: number,
  info?: SwModeInfo,
  scratch?: BlockScratch,
): void {
  if (row < 0 || row >= packed.rows) throw new Error(`decodeRowInto: row ${row} out of range`);
  const modeInfo = info ?? swModeInfo(packed.mode);
  const s = scratch ?? makeBlockScratch(packed.blockSize, modeInfo);
  let byteOffset = packed.rowOffsets[row];
  let k0 = 0;
  while (k0 < packed.cols) {
    const blockLen = Math.min(packed.blockSize, packed.cols - k0);
    byteOffset = readBlock(packed, byteOffset, blockLen, modeInfo, s);
    for (let k = 0; k < blockLen; k++) out[outOffset + k0 + k] = s.lut[s.codes[k]];
    for (let t = 0; t < s.outCount; t++) out[outOffset + k0 + s.outIdx[t]] = s.outVal[t];
    k0 += blockLen;
  }
  if (byteOffset !== packed.rowOffsets[row + 1]) {
    throw new Error("corrupt packed tensor: row byte length mismatch");
  }
}

// --- size / traffic accounting ------------------------------------------------

/** True serialized bytes of a packed tensor: stream + row offsets + a nominal 16-byte header. */
export function packedBytes(packed: PackedTensor): number {
  return packed.data.length + packed.rowOffsets.byteLength + 16;
}

export function bytesPerElement(packed: PackedTensor): number {
  return packedBytes(packed) / (packed.rows * packed.cols);
}

export function describePacked(packed: PackedTensor): PackedDescription {
  const bytes = packedBytes(packed);
  const n = packed.rows * packed.cols;
  return {
    mode: packed.mode,
    rows: packed.rows,
    cols: packed.cols,
    blockSize: packed.blockSize,
    numBlocks: packed.numBlocks,
    outlierTotal: packed.outlierTotal,
    outlierFraction: packed.outlierTotal / n,
    bytes,
    bytesPerElement: bytes / n,
    bitsPerElement: (bytes * 8) / n,
    ratioVsBf16: (2 * n) / bytes,
    ratioVsFp32: (4 * n) / bytes,
  };
}

/**
 * Concatenate two packed tensors row-wise (a's rows followed by b's).
 * Both must share mode, cols, and blockSize; the result decodes to the
 * row-wise concatenation of the two inputs. Used for incremental KV
 * appends: a one-token PackedKV is encoded per decode step and
 * concatenated onto the cached prefix instead of re-encoding the whole
 * prefix. Throws on mismatched layouts.
 */
export function concatPackedTensors(a: PackedTensor, b: PackedTensor): PackedTensor {
  validatePacked(a);
  validatePacked(b);
  if (a.mode !== b.mode || a.cols !== b.cols || a.blockSize !== b.blockSize) {
    throw new Error(
      `concatPackedTensors: layout mismatch (a: ${a.mode} ${a.rows}x${a.cols} block ${a.blockSize}; ` +
        `b: ${b.mode} ${b.rows}x${b.cols} block ${b.blockSize})`,
    );
  }
  const data = new Uint8Array(a.data.length + b.data.length);
  data.set(a.data, 0);
  data.set(b.data, a.data.length);
  const rowOffsets = new Uint32Array(a.rows + b.rows + 1);
  rowOffsets.set(a.rowOffsets.subarray(0, a.rows + 1), 0);
  for (let r = 0; r < b.rows + 1; r++) {
    rowOffsets[a.rows + r] = a.data.length + b.rowOffsets[r];
  }
  return {
    mode: a.mode,
    rows: a.rows + b.rows,
    cols: a.cols,
    blockSize: a.blockSize,
    data,
    rowOffsets,
    numBlocks: a.numBlocks + b.numBlocks,
    outlierTotal: a.outlierTotal + b.outlierTotal,
  };
}

// ---------------------------------------------------------------------------
// Error bounds — Theorem N1 (see docs/research/02-numbers.md)
// ---------------------------------------------------------------------------

/**
 * Worst-case |x_i - x_hat_i| over all elements of a packed tensor, from
 * the stored scales and outlier values alone (no access to the original).
 * Non-outliers contribute delta/2 (2^-24 when the scale underflowed to
 * 0); outliers contribute the fp16 rounding bound. The decoded output
 * is float32, so comparing against the original float32 input adds at
 * most a further 2^-24 * |x| of float32 rounding; see the tests.
 */
export function maxElementErrorBound(packed: PackedTensor): number {
  validatePacked(packed);
  const info = swModeInfo(packed.mode);
  const scratch = makeBlockScratch(packed.blockSize, info);
  let bound = 0;
  for (let r = 0; r < packed.rows; r++) {
    let byteOffset = packed.rowOffsets[r];
    let k0 = 0;
    while (k0 < packed.cols) {
      const blockLen = Math.min(packed.blockSize, packed.cols - k0);
      byteOffset = readBlock(packed, byteOffset, blockLen, info, scratch);
      const halfStep = scratch.delta === 0 ? SW_FP16_MIN_SUBNORMAL : scratch.delta / 2;
      if (halfStep > bound) bound = halfStep;
      for (let t = 0; t < scratch.outCount; t++) {
        const e = SW_FP16_REL_ERR * Math.abs(scratch.outVal[t]) + SW_FP16_ABS_ERR;
        if (e > bound) bound = e;
      }
      k0 += blockLen;
    }
  }
  return bound;
}

/**
 * Bound on ||x - x_hat||_2 for one row: sqrt(sum over blocks of
 * (n_b - o_b) * (delta_b/2)^2 + sum over outliers of the fp16 error^2).
 * This is the row error norm E used by the matmul bounds below.
 */
export function rowErrorNormBound(packed: PackedTensor, row: number): number {
  validatePacked(packed);
  if (row < 0 || row >= packed.rows) throw new Error(`rowErrorNormBound: row ${row} out of range`);
  const info = swModeInfo(packed.mode);
  const scratch = makeBlockScratch(packed.blockSize, info);
  let byteOffset = packed.rowOffsets[row];
  let sumSq = 0;
  let k0 = 0;
  while (k0 < packed.cols) {
    const blockLen = Math.min(packed.blockSize, packed.cols - k0);
    byteOffset = readBlock(packed, byteOffset, blockLen, info, scratch);
    const halfStep = scratch.delta === 0 ? SW_FP16_MIN_SUBNORMAL : scratch.delta / 2;
    sumSq += (blockLen - scratch.outCount) * halfStep * halfStep;
    for (let t = 0; t < scratch.outCount; t++) {
      const e = SW_FP16_REL_ERR * Math.abs(scratch.outVal[t]) + SW_FP16_ABS_ERR;
      sumSq += e * e;
    }
    k0 += blockLen;
  }
  return Math.sqrt(sumSq);
}

/** L2 norm of one decoded row (exact up to float64 accumulation). */
export function decodedRowNorm(packed: PackedTensor, row: number): number {
  validatePacked(packed);
  if (row < 0 || row >= packed.rows) throw new Error(`decodedRowNorm: row ${row} out of range`);
  const info = swModeInfo(packed.mode);
  const scratch = makeBlockScratch(packed.blockSize, info);
  const buf = new Float32Array(packed.cols);
  decodeRowInto(packed, row, buf, 0, info, scratch);
  let sumSq = 0;
  for (let i = 0; i < buf.length; i++) sumSq += buf[i] * buf[i];
  return Math.sqrt(sumSq);
}

/**
 * Theorem N1, dot-product form. |a . b_j - a . b_hat_j| <= ||a||_2 * E_j
 * where E_j is the row error norm bound of packed row j (Cauchy-Schwarz).
 * `aRow` is one float32 row of A (length == b.cols).
 */
export function matmulBTErrorBound(aRow: Float32Array, b: PackedTensor, bRow: number): number {
  if (aRow.length !== b.cols) {
    throw new Error(`matmulBTErrorBound: aRow length ${aRow.length} != b.cols ${b.cols}`);
  }
  let aNormSq = 0;
  for (let i = 0; i < aRow.length; i++) aNormSq += aRow[i] * aRow[i];
  return Math.sqrt(aNormSq) * rowErrorNormBound(b, bRow);
}

/**
 * Composed bound for packed-A times packed-B: with E_A, E_B the row error
 * norms and ||a_hat||, ||b_hat|| the decoded row norms,
 * |a.b - a_hat.b_hat| <= E_A * (||b_hat|| + E_B) + ||a_hat|| * E_B.
 * (Proof: split a.b - a_hat.b_hat = (a - a_hat).b + a_hat.(b - b_hat)
 * and use ||b|| <= ||b_hat|| + E_B.)
 */
export function matmulABErrorBound(a: PackedTensor, aRow: number, b: PackedTensor, bRow: number): number {
  const eA = rowErrorNormBound(a, aRow);
  const eB = rowErrorNormBound(b, bRow);
  const normA = decodedRowNorm(a, aRow);
  const normB = decodedRowNorm(b, bRow);
  return eA * (normB + eB) + normA * eB;
}

// ---------------------------------------------------------------------------
// Matmuls directly on the packed representation
// ---------------------------------------------------------------------------

/**
 * Dot one float32 vector against one packed row, reading the packed
 * stream block by block (lookup table per block, no float32 copy of
 * the row). `aOffset` is the start of the vector in `a`.
 */
export function dotPackedRow(
  a: Float32Array,
  aOffset: number,
  packed: PackedTensor,
  row: number,
  scratch?: BlockScratch,
): number {
  if (row < 0 || row >= packed.rows) throw new Error(`dotPackedRow: row ${row} out of range`);
  if (aOffset < 0 || aOffset + packed.cols > a.length) {
    throw new Error("dotPackedRow: vector out of range");
  }
  const info = swModeInfo(packed.mode);
  const s = scratch ?? makeBlockScratch(packed.blockSize, info);
  let byteOffset = packed.rowOffsets[row];
  let sum = 0;
  let k0 = 0;
  while (k0 < packed.cols) {
    const blockLen = Math.min(packed.blockSize, packed.cols - k0);
    byteOffset = readBlock(packed, byteOffset, blockLen, info, s);
    for (let k = 0; k < blockLen; k++) sum += a[aOffset + k0 + k] * s.lut[s.codes[k]];
    for (let t = 0; t < s.outCount; t++) sum += a[aOffset + k0 + s.outIdx[t]] * s.outVal[t];
    k0 += blockLen;
  }
  return sum;
}

/**
 * C = A @ B^T where A is float32 [aRows, aCols] and B is packed
 * [bRows, bCols] with bCols == aCols — the shape every weight projection
 * in the engine has (weights arrive [out, in]). Each packed block is
 * unpacked once per matmul into a lookup table and dotted against every
 * row of A; B is never materialized as float32. Accumulation is float64
 * (plain JS numbers), matching the error analysis.
 */
export function matmulPackedBT(
  a: Float32Array,
  aRows: number,
  aCols: number,
  b: PackedTensor,
): Float32Array {
  validatePacked(b);
  if (b.cols !== aCols) {
    throw new Error(`matmulPackedBT shape mismatch: A is [${aRows},${aCols}], B is [${b.rows},${b.cols}]`);
  }
  if (a.length !== aRows * aCols) {
    throw new Error(`matmulPackedBT: A length ${a.length} != ${aRows}*${aCols}`);
  }
  const c = new Float32Array(aRows * b.rows);
  const info = swModeInfo(b.mode);
  const scratch = makeBlockScratch(b.blockSize, info);
  for (let j = 0; j < b.rows; j++) {
    let byteOffset = b.rowOffsets[j];
    let k0 = 0;
    while (k0 < b.cols) {
      const blockLen = Math.min(b.blockSize, b.cols - k0);
      byteOffset = readBlock(b, byteOffset, blockLen, info, scratch);
      for (let i = 0; i < aRows; i++) {
        const aRow = i * aCols + k0;
        let sum = 0;
        for (let k = 0; k < blockLen; k++) sum += a[aRow + k] * scratch.lut[scratch.codes[k]];
        for (let t = 0; t < scratch.outCount; t++) {
          sum += a[aRow + scratch.outIdx[t]] * scratch.outVal[t];
        }
        c[i * b.rows + j] += sum;
      }
      k0 += blockLen;
    }
    if (byteOffset !== b.rowOffsets[j + 1]) {
      throw new Error("corrupt packed tensor: row byte length mismatch");
    }
  }
  return c;
}

/**
 * C = A @ B^T where both A [aRows, aCols] and B [bRows, bCols] are
 * packed (quantized activations times quantized weights) and
 * aCols == bCols. Requires both tensors to use the same blockSize so
 * blocks align; A is decoded one block at a time into a tile (one
 * block per row, never the whole matrix) and dotted against B's
 * lookup tables. Error bound: matmulABErrorBound.
 */
export function matmulPackedAB(a: PackedTensor, b: PackedTensor): Float32Array {
  validatePacked(a);
  validatePacked(b);
  if (a.cols !== b.cols) {
    throw new Error(`matmulPackedAB shape mismatch: A is [${a.rows},${a.cols}], B is [${b.rows},${b.cols}]`);
  }
  if (a.blockSize !== b.blockSize) {
    throw new Error(
      `matmulPackedAB: blockSize mismatch (${a.blockSize} vs ${b.blockSize}); ` +
        `encode both tensors with the same blockSize so blocks align`,
    );
  }
  const blockSize = a.blockSize;
  const numBlocks = Math.ceil(a.cols / blockSize);
  const c = new Float32Array(a.rows * b.rows);
  const infoA = swModeInfo(a.mode);
  const infoB = swModeInfo(b.mode);
  const scratchA = makeBlockScratch(blockSize, infoA);
  const scratchB = makeBlockScratch(blockSize, infoB);
  // Tile: one decoded block per A row for the current block index.
  const aTile = new Float32Array(a.rows * blockSize);
  const aOffsets = new Uint32Array(a.rows);
  const bOffsets = new Uint32Array(b.rows);
  for (let i = 0; i < a.rows; i++) aOffsets[i] = a.rowOffsets[i];
  for (let j = 0; j < b.rows; j++) bOffsets[j] = b.rowOffsets[j];
  for (let blk = 0; blk < numBlocks; blk++) {
    const k0 = blk * blockSize;
    const blockLen = Math.min(blockSize, a.cols - k0);
    for (let i = 0; i < a.rows; i++) {
      aOffsets[i] = readBlock(a, aOffsets[i], blockLen, infoA, scratchA);
      const base = i * blockSize;
      for (let k = 0; k < blockLen; k++) aTile[base + k] = scratchA.lut[scratchA.codes[k]];
      for (let t = 0; t < scratchA.outCount; t++) {
        aTile[base + scratchA.outIdx[t]] = scratchA.outVal[t];
      }
    }
    for (let j = 0; j < b.rows; j++) {
      bOffsets[j] = readBlock(b, bOffsets[j], blockLen, infoB, scratchB);
      for (let i = 0; i < a.rows; i++) {
        const base = i * blockSize;
        let sum = 0;
        for (let k = 0; k < blockLen; k++) sum += aTile[base + k] * scratchB.lut[scratchB.codes[k]];
        for (let t = 0; t < scratchB.outCount; t++) {
          sum += aTile[base + scratchB.outIdx[t]] * scratchB.outVal[t];
        }
        c[i * b.rows + j] += sum;
      }
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// KV cache quantization — Theorem N2 (see docs/research/02-numbers.md)
// ---------------------------------------------------------------------------

export interface PackedKV {
  k: PackedTensor;
  v: PackedTensor;
  kvLen: number;
  numKvHeads: number;
  headDim: number;
  kMode: SwMode;
  vMode: SwMode;
}

export interface PackedKVAttentionOptions {
  /** Mask out KV positions after the query position (default true). */
  causal?: boolean;
  /** Absolute position of the query token (needed for causal masking). */
  queryPosition?: number;
  /** Absolute positions of the KV entries, in order (needed for causal masking). */
  kvPositions?: number[];
  /** Attention scale; defaults to 1/sqrt(headDim). */
  scale?: number;
}

/**
 * Quantize a KV cache for one layer: keys and values are
 * [kvLen, numKvHeads, headDim] flattened, one block per (token, KV head)
 * vector of length headDim. K and V may use different modes (K is more
 * sensitive because it moves the softmax; V enters the output linearly).
 * Quantize *after* RoPE, matching the engine: the cache stores post-RoPE K.
 */
export function quantizeKV(
  keys: Float32Array,
  values: Float32Array,
  kvLen: number,
  numKvHeads: number,
  headDim: number,
  mode: SwMode | { k: SwMode; v: SwMode },
): PackedKV {
  const kMode: SwMode = typeof mode === "string" ? mode : mode.k;
  const vMode: SwMode = typeof mode === "string" ? mode : mode.v;
  if (!Number.isInteger(kvLen) || kvLen < 1) throw new Error(`quantizeKV: kvLen must be >= 1, got ${kvLen}`);
  if (!Number.isInteger(numKvHeads) || numKvHeads < 1) {
    throw new Error(`quantizeKV: numKvHeads must be >= 1, got ${numKvHeads}`);
  }
  if (!Number.isInteger(headDim) || headDim < 1) throw new Error(`quantizeKV: headDim must be >= 1, got ${headDim}`);
  const expected = kvLen * numKvHeads * headDim;
  if (keys.length !== expected) throw new Error(`quantizeKV: keys length ${keys.length} != ${expected}`);
  if (values.length !== expected) throw new Error(`quantizeKV: values length ${values.length} != ${expected}`);
  const rows = kvLen * numKvHeads;
  return {
    k: encodeMatrix(keys, rows, headDim, kMode, { blockSize: headDim }),
    v: encodeMatrix(values, rows, headDim, vMode, { blockSize: headDim }),
    kvLen,
    numKvHeads,
    headDim,
    kMode,
    vMode,
  };
}

/**
 * Grouped-query attention for a single query token against a quantized
 * KV cache. Mirrors groupedQueryAttention (src/engine/attention.ts):
 * q is [numHeads, headDim] flattened; returns [numHeads, headDim].
 * Scores are dotted directly against the packed K rows; V rows are
 * decoded one at a time into a headDim scratch vector and accumulated.
 * The cache itself stays packed throughout.
 */
export function attentionPackedKV(
  q: Float32Array,
  kv: PackedKV,
  numHeads: number,
  options: PackedKVAttentionOptions = {},
): Float32Array {
  const { kvLen, numKvHeads, headDim } = kv;
  if (q.length !== numHeads * headDim) {
    throw new Error(`attentionPackedKV: q length ${q.length}, expected ${numHeads * headDim}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(`attentionPackedKV: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`);
  }
  const groupSize = numHeads / numKvHeads;
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const causal = options.causal ?? true;
  const out = new Float32Array(numHeads * headDim);
  const kInfo = swModeInfo(kv.k.mode);
  const vInfo = swModeInfo(kv.v.mode);
  const scratchK = makeBlockScratch(kv.k.blockSize, kInfo);
  const scratchV = makeBlockScratch(kv.v.blockSize, vInfo);
  const vRow = new Float32Array(headDim);
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const qOff = h * headDim;
    const scores = new Float32Array(kvLen);
    let max = -Infinity;
    for (let t = 0; t < kvLen; t++) {
      if (causal && options.kvPositions && options.queryPosition !== undefined) {
        if (options.kvPositions[t] > options.queryPosition) {
          scores[t] = -Infinity;
          continue;
        }
      }
      const dot = dotPackedRow(q, qOff, kv.k, t * numKvHeads + kvHead, scratchK);
      scores[t] = dot * scale;
      if (scores[t] > max) max = scores[t];
    }
    let sum = 0;
    const weights = new Float32Array(kvLen);
    for (let t = 0; t < kvLen; t++) {
      weights[t] = scores[t] === -Infinity ? 0 : Math.exp(scores[t] - max);
      sum += weights[t];
    }
    if (sum === 0) continue; // fully masked: leave zeros
    const inv = 1 / sum;
    const oOff = h * headDim;
    for (let t = 0; t < kvLen; t++) {
      if (weights[t] === 0) continue;
      const w = weights[t] * inv;
      decodeRowInto(kv.v, t * numKvHeads + kvHead, vRow, 0, vInfo, scratchV);
      for (let d = 0; d < headDim; d++) out[oOff + d] += w * vRow[d];
    }
  }
  return out;
}

/**
 * Theorem N2 bound for one query head. Let epsK / epsV be the largest
 * per-row error norms over the KV rows this head reads, Vmax the largest
 * decoded V row norm, and deltaMax = ||q||_2 * epsK * scale the largest
 * logit error. Then
 *   ||o_hat - o||_2 <= 0.5 * deltaMax * Vmax + epsV.
 * The 0.5 is the softmax Lipschitz constant: the Jacobian of softmax
 * has infinity-to-1 operator norm <= 1/2 (see 02-numbers.md, Lemma N2.1).
 * The bound is per head; take the max over heads for the whole output.
 */
export function kvAttentionErrorBound(
  q: Float32Array,
  kv: PackedKV,
  numHeads: number,
  head: number,
  options: PackedKVAttentionOptions = {},
): number {
  const { kvLen, numKvHeads, headDim } = kv;
  if (head < 0 || head >= numHeads) throw new Error(`kvAttentionErrorBound: head ${head} out of range`);
  if (q.length !== numHeads * headDim) {
    throw new Error(`kvAttentionErrorBound: q length ${q.length}, expected ${numHeads * headDim}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(`kvAttentionErrorBound: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`);
  }
  const groupSize = numHeads / numKvHeads;
  const kvHead = Math.floor(head / groupSize);
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  let qNormSq = 0;
  for (let d = 0; d < headDim; d++) qNormSq += q[head * headDim + d] * q[head * headDim + d];
  const qNorm = Math.sqrt(qNormSq);
  let epsK = 0;
  let epsV = 0;
  let vMax = 0;
  for (let t = 0; t < kvLen; t++) {
    const row = t * numKvHeads + kvHead;
    const ek = rowErrorNormBound(kv.k, row);
    if (ek > epsK) epsK = ek;
    const ev = rowErrorNormBound(kv.v, row);
    if (ev > epsV) epsV = ev;
    const vn = decodedRowNorm(kv.v, row);
    if (vn > vMax) vMax = vn;
  }
  const deltaMax = qNorm * epsK * scale;
  return 0.5 * deltaMax * vMax + epsV;
}
