/**
 * Model layer — safetensors manifest helpers.
 *
 * A safetensors file is: 8-byte little-endian header length N, followed by
 * N bytes of JSON header, followed by the raw tensor data. The JSON header
 * maps each tensor name to { dtype, shape, data_offsets } where the offsets
 * are [start, end) byte ranges relative to the start of the data section.
 * A top-level "__metadata__" key (if present) holds string key/value pairs.
 *
 * This module is the read/write side of that format for softwell-machine.
 * It never touches the network and has no dependency on the softwell repo.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { ModelConfig, ModelManifest, TensorMeta, WeightDtype, WeightFile } from "../contracts.ts";

// ---------------------------------------------------------------------------
// Dtypes
// ---------------------------------------------------------------------------

export const DTYPE_BYTE_SIZES: Record<WeightDtype, number> = {
  bfloat16: 2,
  float16: 2,
  float32: 4,
  float8_e4m3: 1,
};

export function dtypeByteSize(dtype: WeightDtype): number {
  const size = DTYPE_BYTE_SIZES[dtype];
  if (size === undefined) throw new Error(`unsupported dtype: ${dtype}`);
  return size;
}

export function numElements(shape: number[]): number {
  let n = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) throw new Error(`invalid shape dimension: ${dim}`);
    n *= dim;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Float conversions (dequantize to float32)
// ---------------------------------------------------------------------------

/** Decode one bfloat16 (uint16) value to float32: shift into the high bits. */
export function bf16ToFloat32(bits: number): number {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint32(0, (bits << 16) >>> 0);
  return view.getFloat32(0);
}

/** Encode one float32 value to bfloat16 bits (round-to-nearest-even). */
export function float32ToBf16(value: number): number {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const roundingBias = 0x7fff + ((bits >> 16) & 1);
  return (bits + roundingBias) >>> 16;
}

/** Decode one float16 (uint16) value to float32. */
export function fp16ToFloat32(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x03ff;
  if (exp === 0) {
    // Zero or subnormal.
    if (frac === 0) return sign * 0;
    return sign * (frac / 1024) * Math.pow(2, -14);
  }
  if (exp === 31) {
    return frac === 0 ? sign * Infinity : NaN;
  }
  return sign * (1 + frac / 1024) * Math.pow(2, exp - 15);
}

/** Encode one float32 value to float16 bits (round-to-nearest-even). */
export function float32ToFp16(value: number): number {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  const sign = (bits >> 16) & 0x8000;
  let exp = ((bits >> 23) & 0xff) - 112; // re-bias 127 -> 15
  const frac = bits & 0x7fffff;
  if (exp >= 31) {
    // Overflow -> infinity (unless NaN).
    return Number.isNaN(value) ? sign | 0x7e00 : sign | 0x7c00;
  }
  if (exp <= 0) {
    if (exp < -10) return sign; // underflows to zero
    // Subnormal: shift the implicit leading 1 into the fraction.
    const mantissa = frac | 0x800000;
    const shift = 14 - exp;
    let halfFrac = mantissa >> shift;
    // Round-to-nearest-even on the dropped bits.
    const remainder = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (halfFrac & 1) === 1)) halfFrac += 1;
    return sign | halfFrac;
  }
  let halfFrac = frac >> 13;
  const remainder = frac & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfFrac & 1) === 1)) {
    halfFrac += 1;
    if (halfFrac === 0x400) {
      // Rounded up into the exponent.
      halfFrac = 0;
      exp += 1;
      if (exp >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exp << 10) | halfFrac;
}

/**
 * Decode `count` elements of `dtype` starting at `byteOffset` in `buffer`
 * into a fresh Float32Array. float8_e4m3 is decoded per the e4m3 spec
 * (no infinities; 0x7f/0xff are NaN).
 */
export function decodeToFloat32(
  dtype: WeightDtype,
  buffer: Buffer,
  byteOffset: number,
  count: number,
): Float32Array {
  const out = new Float32Array(count);
  if (dtype === "float32") {
    const view = new DataView(buffer.buffer, buffer.byteOffset + byteOffset, count * 4);
    for (let i = 0; i < count; i++) out[i] = view.getFloat32(i * 4, true);
    return out;
  }
  if (dtype === "bfloat16") {
    const view = new DataView(buffer.buffer, buffer.byteOffset + byteOffset, count * 2);
    for (let i = 0; i < count; i++) out[i] = bf16ToFloat32(view.getUint16(i * 2, true));
    return out;
  }
  if (dtype === "float16") {
    const view = new DataView(buffer.buffer, buffer.byteOffset + byteOffset, count * 2);
    for (let i = 0; i < count; i++) out[i] = fp16ToFloat32(view.getUint16(i * 2, true));
    return out;
  }
  // float8_e4m3
  for (let i = 0; i < count; i++) {
    const b = buffer[byteOffset + i] as number;
    const sign = (b & 0x80) !== 0 ? -1 : 1;
    const exp = (b >> 3) & 0x0f;
    const frac = b & 0x07;
    if (exp === 0) {
      out[i] = frac === 0 ? sign * 0 : sign * (frac / 8) * Math.pow(2, -6);
    } else if (exp === 15 && frac === 7) {
      out[i] = NaN;
    } else {
      out[i] = sign * (1 + frac / 8) * Math.pow(2, exp - 7);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Header parse / serialize
// ---------------------------------------------------------------------------

export interface SafetensorsHeader {
  tensors: Record<string, TensorMeta>;
  metadata: Record<string, string>;
}

interface RawTensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Canonical safetensors dtype codes for the internal WeightDtype names.
 * Real HF files use these codes (F32, F16, BF16, F8_E4M3); our own
 * writer used to emit the long names, which no spec reader accepts.
 */
const CANONICAL_DTYPE: Record<WeightDtype, string> = {
  bfloat16: "BF16",
  float16: "F16",
  float32: "F32",
  float8_e4m3: "F8_E4M3",
};

function parseTensorMeta(name: string, raw: unknown): TensorMeta {
  if (!isRecord(raw)) throw new Error(`safetensors header: tensor "${name}" is not an object`);
  const rawDtype = raw["dtype"];
  // Real safetensors files use the canonical dtype codes (F32, F16, BF16,
  // F8_E4M3); our own writer emits the long names (float32, ...). Accept
  // both, normalizing to the internal WeightDtype.
  const dtype: WeightDtype | undefined =
    rawDtype === "bfloat16" || rawDtype === "BF16"
      ? "bfloat16"
      : rawDtype === "float16" || rawDtype === "F16"
        ? "float16"
        : rawDtype === "float32" || rawDtype === "F32"
          ? "float32"
          : rawDtype === "float8_e4m3" || rawDtype === "F8_E4M3"
            ? "float8_e4m3"
            : undefined;
  if (dtype === undefined) {
    throw new Error(`safetensors header: tensor "${name}" has unsupported dtype ${String(rawDtype)}`);
  }
  const shape = raw["shape"];
  if (!Array.isArray(shape) || !shape.every((d) => Number.isInteger(d) && (d as number) >= 0)) {
    throw new Error(`safetensors header: tensor "${name}" has an invalid shape`);
  }
  const offsets = raw["data_offsets"];
  if (
    !Array.isArray(offsets) ||
    offsets.length !== 2 ||
    !Number.isInteger(offsets[0]) ||
    !Number.isInteger(offsets[1]) ||
    (offsets[0] as number) < 0 ||
    (offsets[1] as number) < (offsets[0] as number)
  ) {
    throw new Error(`safetensors header: tensor "${name}" has invalid data_offsets`);
  }
  return { dtype, shape: shape as number[], dataOffsets: [offsets[0] as number, offsets[1] as number] };
}

/**
 * Parse a safetensors header from a buffer holding at least the 8-byte
 * length prefix plus the JSON header. Returns the tensor map and metadata.
 * Throws on a truncated or malformed header.
 */
export function parseSafetensorsHeader(buffer: Buffer): SafetensorsHeader {
  if (buffer.length < 8) throw new Error("safetensors header: buffer too short for length prefix");
  const headerLen = Number(buffer.readBigUInt64LE(0));
  if (!Number.isSafeInteger(headerLen) || headerLen < 0) {
    throw new Error("safetensors header: invalid header length");
  }
  if (buffer.length < 8 + headerLen) {
    throw new Error(`safetensors header: truncated (need ${8 + headerLen} bytes, have ${buffer.length})`);
  }
  const jsonText = buffer.subarray(8, 8 + headerLen).toString("utf-8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch (err) {
    throw new Error(`safetensors header: invalid JSON: ${(err as Error).message}`);
  }
  if (!isRecord(parsed)) throw new Error("safetensors header: top level is not an object");
  const tensors: Record<string, TensorMeta> = {};
  const metadata: Record<string, string> = {};
  for (const [key, value] of Object.entries(parsed)) {
    if (key === "__metadata__") {
      if (!isRecord(value)) throw new Error("safetensors header: __metadata__ is not an object");
      for (const [mk, mv] of Object.entries(value)) {
        if (typeof mv !== "string") throw new Error(`safetensors header: metadata "${mk}" is not a string`);
        metadata[mk] = mv;
      }
    } else {
      tensors[key] = parseTensorMeta(key, value);
    }
  }
  return { tensors, metadata };
}

/** Header length (8-byte prefix + JSON) for a buffer, without parsing JSON. */
export function safetensorsHeaderLength(buffer: Buffer): number {
  if (buffer.length < 8) throw new Error("safetensors header: buffer too short for length prefix");
  return 8 + Number(buffer.readBigUInt64LE(0));
}

/**
 * Serialize a tensor map (+ optional metadata) into the on-disk header
 * bytes: 8-byte little-endian JSON length followed by the JSON itself.
 * Dtypes are written as the canonical safetensors codes (F32, F16, BF16,
 * F8_E4M3) — the same codes parseSafetensorsHeader accepts — so files
 * written here are valid safetensors for any reader, not just ours.
 */
export function serializeSafetensorsHeader(
  tensors: Record<string, TensorMeta>,
  metadata: Record<string, string> = {},
): Buffer {
  const obj: Record<string, unknown> = {};
  if (Object.keys(metadata).length > 0) obj["__metadata__"] = metadata;
  for (const [name, meta] of Object.entries(tensors)) {
    const header: RawTensorHeader = {
      dtype: CANONICAL_DTYPE[meta.dtype],
      shape: meta.shape,
      data_offsets: [meta.dataOffsets[0], meta.dataOffsets[1]],
    };
    obj[name] = header;
  }
  const json = Buffer.from(JSON.stringify(obj), "utf-8");
  const out = Buffer.alloc(8 + json.length);
  out.writeBigUInt64LE(BigInt(json.length), 0);
  json.copy(out, 8);
  return out;
}

// ---------------------------------------------------------------------------
// Totals / manifest construction
// ---------------------------------------------------------------------------

export function computeTotals(tensors: Record<string, TensorMeta>): {
  totalParams: number;
  totalBytes: number;
} {
  let totalParams = 0;
  let totalBytes = 0;
  for (const meta of Object.values(tensors)) {
    totalParams += numElements(meta.shape);
    totalBytes += meta.dataOffsets[1] - meta.dataOffsets[0];
  }
  return { totalParams, totalBytes };
}

/**
 * Assemble a ModelManifest from its parts, computing totalParams/totalBytes
 * from the tensor map. `files` is the shard list for the model directory.
 */
export function createManifest(
  modelId: string,
  config: ModelConfig,
  tensors: Record<string, TensorMeta>,
  files: WeightFile[],
  metadata: Record<string, string> = {},
): ModelManifest {
  const { totalParams, totalBytes } = computeTotals(tensors);
  return { format: "safetensors", modelId, config, metadata, tensors, files, totalBytes, totalParams };
}

// ---------------------------------------------------------------------------
// Writer (used by the demo to create a random-weight model file)
// ---------------------------------------------------------------------------

export interface TensorToWrite {
  name: string;
  dtype: WeightDtype;
  shape: number[];
  data: Float32Array;
}

/**
 * Write a single safetensors file. Tensor data is laid out contiguously in
 * the order given, with correct data_offsets in the header. Supports
 * float32, bfloat16 and float16 on the write path (values are converted
 * from the supplied float32 data).
 */
export function writeSafetensorsFile(filePath: string, tensors: TensorToWrite[]): void {
  // Validate shapes against data lengths and build the header first so
  // offsets are known before any bytes hit disk.
  const metas: Record<string, TensorMeta> = {};
  const encoded: Buffer[] = [];
  let offset = 0;
  for (const tensor of tensors) {
    const count = numElements(tensor.shape);
    if (tensor.data.length !== count) {
      throw new Error(
        `tensor "${tensor.name}": data has ${tensor.data.length} elements but shape ${JSON.stringify(tensor.shape)} needs ${count}`,
      );
    }
    const byteSize = dtypeByteSize(tensor.dtype);
    const bytes = Buffer.alloc(count * byteSize);
    if (tensor.dtype === "float32") {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
      for (let i = 0; i < count; i++) view.setFloat32(i * 4, tensor.data[i] as number, true);
    } else if (tensor.dtype === "bfloat16") {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
      for (let i = 0; i < count; i++) view.setUint16(i * 2, float32ToBf16(tensor.data[i] as number), true);
    } else if (tensor.dtype === "float16") {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
      for (let i = 0; i < count; i++) view.setUint16(i * 2, float32ToFp16(tensor.data[i] as number), true);
    } else {
      throw new Error(`writeSafetensorsFile: dtype ${tensor.dtype} is not supported on the write path`);
    }
    metas[tensor.name] = { dtype: tensor.dtype, shape: [...tensor.shape], dataOffsets: [offset, offset + bytes.length] };
    encoded.push(bytes);
    offset += bytes.length;
  }
  const header = serializeSafetensorsHeader(metas);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const fd = fs.openSync(filePath, "w");
  try {
    fs.writeSync(fd, header);
    for (const bytes of encoded) fs.writeSync(fd, bytes);
  } finally {
    fs.closeSync(fd);
  }
}
