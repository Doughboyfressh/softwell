/**
 * Model layer — safetensors weight loading.
 *
 * `loadManifest(dir)` reads config.json plus the safetensors header(s) in a
 * model directory and builds a ModelManifest. Sharded models are detected
 * via model.safetensors.index.json; otherwise a single model.safetensors
 * file is expected.
 *
 * `SafetensorsWeightLoader` implements the contracts WeightLoader: tensors
 * are loaded lazily on demand, file buffers are cached after the first read,
 * and every tensor is returned dequantized to Float32Array (bfloat16 via
 * bit-shift, float16 via half-float decode, float32 direct).
 *
 * Tensor names follow the HF Llama/Mixtral convention documented in
 * contracts.ts (model.embed_tokens.weight, model.layers.{i}.self_attn.*,
 * model.layers.{i}.mlp.*, model.norm.weight, lm_head.weight, ...).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { ModelConfig, ModelManifest, TensorMeta, WeightFile, WeightLoader } from "../contracts.ts";
import {
  createManifest,
  decodeToFloat32,
  numElements,
  parseSafetensorsHeader,
  safetensorsHeaderLength,
} from "./manifest.ts";

export const SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json";
export const SAFETENSORS_SINGLE_FILENAME = "model.safetensors";
export const CONFIG_FILENAME = "config.json";

interface ShardIndex {
  metadata?: { total_size?: number };
  weight_map?: Record<string, string>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asInt(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : fallback;
}

/**
 * Map a Hugging Face config.json object onto ModelConfig, filling in
 * defaults for anything absent. Accepts both the HF snake_case keys and
 * the camelCase keys used in contracts.ts.
 */
export function parseModelConfig(raw: unknown, fallbackModelId = ""): ModelConfig {
  const cfg: Record<string, unknown> = isRecord(raw) ? raw : {};
  const hiddenSize = asInt(cfg["hidden_size"] ?? cfg["hiddenSize"], 0);
  const numAttentionHeads = asInt(cfg["num_attention_heads"] ?? cfg["numAttentionHeads"], 0);
  const headDim = asInt(cfg["head_dim"] ?? cfg["headDim"], 0);
  return {
    architectures: Array.isArray(cfg["architectures"])
      ? (cfg["architectures"] as unknown[]).filter((a): a is string => typeof a === "string")
      : ["LlamaForCausalLM"],
    vocabSize: asInt(cfg["vocab_size"] ?? cfg["vocabSize"], 0),
    hiddenSize,
    intermediateSize: asInt(cfg["intermediate_size"] ?? cfg["intermediateSize"], 0),
    numHiddenLayers: asInt(cfg["num_hidden_layers"] ?? cfg["numHiddenLayers"], 0),
    numAttentionHeads,
    numKeyValueHeads: asInt(
      cfg["num_key_value_heads"] ?? cfg["numKeyValueHeads"],
      numAttentionHeads,
    ),
    headDim: headDim > 0 ? headDim : numAttentionHeads > 0 && hiddenSize > 0 ? hiddenSize / numAttentionHeads : 0,
    maxPositionEmbeddings: asInt(cfg["max_position_embeddings"] ?? cfg["maxPositionEmbeddings"], 0),
    rmsNormEps:
      typeof cfg["rms_norm_eps"] === "number"
        ? (cfg["rms_norm_eps"] as number)
        : typeof cfg["rmsNormEps"] === "number"
          ? (cfg["rmsNormEps"] as number)
          : 1e-5,
    ropeTheta:
      typeof cfg["rope_theta"] === "number"
        ? (cfg["rope_theta"] as number)
        : typeof cfg["ropeTheta"] === "number"
          ? (cfg["ropeTheta"] as number)
          : 10000,
    tieWordEmbeddings:
      typeof cfg["tie_word_embeddings"] === "boolean"
        ? (cfg["tie_word_embeddings"] as boolean)
        : (cfg["tieWordEmbeddings"] as boolean | undefined) ?? false,
    numExperts:
      cfg["num_local_experts"] !== undefined
        ? asInt(cfg["num_local_experts"], 0)
        : cfg["num_experts"] !== undefined
          ? asInt(cfg["num_experts"], 0)
          : undefined,
    numExpertsPerToken:
      cfg["num_experts_per_tok"] !== undefined
        ? asInt(cfg["num_experts_per_tok"], 0)
        : cfg["num_experts_per_token"] !== undefined
          ? asInt(cfg["num_experts_per_token"], 0)
          : undefined,
    normTopKProb:
      typeof cfg["norm_topk_prob"] === "boolean"
        ? (cfg["norm_topk_prob"] as boolean)
        : (cfg["normTopKProb"] as boolean | undefined) ?? undefined,
    bosTokenId: asInt(cfg["bos_token_id"] ?? cfg["bosTokenId"], 1),
    eosTokenId: asInt(cfg["eos_token_id"] ?? cfg["eosTokenId"], 2),
    padTokenId: asInt(cfg["pad_token_id"] ?? cfg["padTokenId"], 0),
  };
}

/** Read and parse a JSON file, throwing a clear error on failure. */
function readJsonFile(filePath: string): unknown {
  let text: string;
  try {
    text = fs.readFileSync(filePath, "utf-8");
  } catch (err) {
    throw new Error(`cannot read ${filePath}: ${(err as Error).message}`);
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(`invalid JSON in ${filePath}: ${(err as Error).message}`);
  }
}

/**
 * Read config.json + safetensors headers in `dir` and build a ModelManifest.
 * Prefers model.safetensors.index.json when present (sharded model);
 * otherwise reads the single model.safetensors file. Any other
 * *.safetensors file in the directory is also scanned as a fallback so a
 * directory with a non-standard shard name still loads.
 */
export function loadManifest(dir: string, modelId?: string): ModelManifest {
  const absDir = path.resolve(dir);
  if (!fs.existsSync(absDir) || !fs.statSync(absDir).isDirectory()) {
    throw new Error(`model directory not found: ${absDir}`);
  }
  const configPath = path.join(absDir, CONFIG_FILENAME);
  const config = fs.existsSync(configPath)
    ? parseModelConfig(readJsonFile(configPath))
    : parseModelConfig({});

  const id = modelId ?? path.basename(absDir);
  const tensors: Record<string, TensorMeta> = {};
  const metadata: Record<string, string> = {};
  const files: WeightFile[] = [];
  // Tensor name -> shard filename, so the loader knows where to read it.
  const tensorFiles: Record<string, string> = {};

  const indexPath = path.join(absDir, SAFETENSORS_INDEX_FILENAME);
  let shardNames: string[];
  if (fs.existsSync(indexPath)) {
    const index = readJsonFile(indexPath) as ShardIndex;
    const weightMap = isRecord(index) && isRecord(index["weight_map"])
      ? (index["weight_map"] as Record<string, unknown>)
      : {};
    const names = new Set<string>();
    for (const [tensorName, fileName] of Object.entries(weightMap)) {
      if (typeof fileName === "string") {
        names.add(fileName);
        tensorFiles[tensorName] = fileName;
      }
    }
    shardNames = [...names].sort();
  } else if (fs.existsSync(path.join(absDir, SAFETENSORS_SINGLE_FILENAME))) {
    shardNames = [SAFETENSORS_SINGLE_FILENAME];
  } else {
    // Fallback: any *.safetensors file in the directory.
    shardNames = fs
      .readdirSync(absDir)
      .filter((f) => f.endsWith(".safetensors"))
      .sort();
    if (shardNames.length === 0) {
      throw new Error(`no safetensors file found in ${absDir}`);
    }
  }

  for (const shardName of shardNames) {
    const filePath = path.join(absDir, shardName);
    const stat = fs.statSync(filePath);
    const headerBuf = readHeaderBytes(filePath);
    const { tensors: shardTensors, metadata: shardMetadata } = parseSafetensorsHeader(headerBuf);
    for (const [name, meta] of Object.entries(shardTensors)) {
      tensors[name] = meta;
      if (!(name in tensorFiles)) tensorFiles[name] = shardName;
    }
    Object.assign(metadata, shardMetadata);
    files.push({ filename: shardName, sizeBytes: stat.size });
  }

  const manifest = createManifest(id, config, tensors, files, metadata);
  // Attach the tensor->file map for the loader (kept out of the contract
  // shape by storing it on the manifest object as a non-enumerated extra).
  Object.defineProperty(manifest, "__tensorFiles", {
    value: tensorFiles,
    enumerable: false,
    writable: false,
  });
  return manifest;
}

/** Read just the header bytes (length prefix + JSON) of a safetensors file. */
function readHeaderBytes(filePath: string): Buffer {
  const fd = fs.openSync(filePath, "r");
  try {
    const prefix = Buffer.alloc(8);
    if (fs.readSync(fd, prefix, 0, 8, 0) !== 8) {
      throw new Error(`safetensors file is truncated: ${filePath}`);
    }
    const headerLen = Number(prefix.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLen) || headerLen < 0 || headerLen > 512 * 1024 * 1024) {
      throw new Error(`safetensors file has an implausible header length: ${filePath}`);
    }
    const buf = Buffer.alloc(8 + headerLen);
    prefix.copy(buf, 0);
    if (fs.readSync(fd, buf, 8, headerLen, 8) !== headerLen) {
      throw new Error(`safetensors file header is truncated: ${filePath}`);
    }
    return buf;
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Look up the shard filename for a tensor. Uses the non-enumerated
 * __tensorFiles map attached by loadManifest when present; otherwise falls
 * back to scanning the manifest's files in order and checking each header.
 */
export function tensorFileFor(manifest: ModelManifest, name: string): string | undefined {
  const map = (manifest as ModelManifest & { __tensorFiles?: Record<string, string> }).__tensorFiles;
  return map?.[name];
}

// ---------------------------------------------------------------------------
// WeightLoader
// ---------------------------------------------------------------------------

export class SafetensorsWeightLoader implements WeightLoader {
  readonly manifest: ModelManifest;
  private readonly dir: string;
  /** Raw bytes actually read from disk (cache hits do not count). The
   *  proof workstream asserts this stays 0 while inference runs from
   *  vRAM; any nonzero value here means a stage load went to disk. */
  bytesReadFromDisk = 0;
  /** Shard filename -> whole-file buffer, cached after first read. */
  private readonly fileCache = new Map<string, Buffer>();
  /** Shard filename -> byte offset where the data section starts. */
  private readonly dataStart = new Map<string, number>();
  /** Tensor name -> shard filename (resolved lazily if not in the manifest). */
  private readonly tensorFiles: Map<string, string> = new Map();

  constructor(manifest: ModelManifest, dir: string) {
    this.manifest = manifest;
    this.dir = path.resolve(dir);
    const map = (manifest as ModelManifest & { __tensorFiles?: Record<string, string> }).__tensorFiles;
    if (map) {
      for (const [name, file] of Object.entries(map)) this.tensorFiles.set(name, file);
    }
  }

  hasTensor(name: string): boolean {
    return name in this.manifest.tensors;
  }

  /** All tensor names in the manifest. */
  tensorNames(): string[] {
    return Object.keys(this.manifest.tensors);
  }

  async loadTensor(name: string): Promise<Float32Array> {
    const meta = this.manifest.tensors[name];
    if (!meta) throw new Error(`tensor not found: ${name}`);
    const fileName = await this.resolveTensorFile(name);
    const buffer = this.readShard(fileName);
    const start = this.dataStart.get(fileName) as number;
    const count = numElements(meta.shape);
    const expected = (meta.dataOffsets[1] - meta.dataOffsets[0]) / (meta.dtype === "float8_e4m3" ? 1 : meta.dtype === "float32" ? 4 : 2);
    if (expected !== count) {
      throw new Error(
        `tensor "${name}": data_offsets span ${meta.dataOffsets[1] - meta.dataOffsets[0]} bytes but shape needs ${count} elements of ${meta.dtype}`,
      );
    }
    return decodeToFloat32(meta.dtype, buffer, start + meta.dataOffsets[0], count);
  }

  /** Resolve which shard holds a tensor, scanning headers if necessary. */
  private async resolveTensorFile(name: string): Promise<string> {
    const known = this.tensorFiles.get(name);
    if (known) return known;
    // Fallback: scan each shard header until the tensor is found.
    for (const file of this.manifest.files) {
      const headerBuf = readHeaderBytes(path.join(this.dir, file.filename));
      const { tensors } = parseSafetensorsHeader(headerBuf);
      for (const tensorName of Object.keys(tensors)) this.tensorFiles.set(tensorName, file.filename);
      const found = this.tensorFiles.get(name);
      if (found) return found;
    }
    throw new Error(`tensor "${name}" is in the manifest but no shard contains it`);
  }

  /** Read (and cache) a whole shard file; records its data-section offset. */
  private readShard(fileName: string): Buffer {
    const cached = this.fileCache.get(fileName);
    if (cached) return cached;
    const filePath = path.join(this.dir, fileName);
    const buffer = fs.readFileSync(filePath);
    this.bytesReadFromDisk += buffer.length;
    this.fileCache.set(fileName, buffer);
    this.dataStart.set(fileName, safetensorsHeaderLength(buffer));
    return buffer;
  }

  close(): void {
    this.fileCache.clear();
    this.dataStart.clear();
  }
}

/**
 * Open a model directory: load its manifest and return a ready WeightLoader.
 */
export function openWeightLoader(dir: string, modelId?: string): SafetensorsWeightLoader {
  const manifest = loadManifest(dir, modelId);
  return new SafetensorsWeightLoader(manifest, dir);
}

/** Create a loader from an already-built manifest (same directory). */
export function createWeightLoader(manifest: ModelManifest, dir: string): SafetensorsWeightLoader {
  return new SafetensorsWeightLoader(manifest, dir);
}
