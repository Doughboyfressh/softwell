/**
 * softwell-machine — sharding planner cost model.
 *
 * Honest, explicit arithmetic for distributed LLM inference. The two facts
 * that drive everything:
 *
 *  1. Decode is memory-bandwidth bound. Generating one token reads every
 *     weight once (per sequence in the batch), so tokens/sec ≈ aggregate
 *     HBM read bandwidth / total weight bytes, times batch size.
 *  2. Prefill is compute bound. A forward pass costs ~2 FLOPs per
 *     (active) parameter per token, so prefill time ≈ prompt tokens *
 *     2 * active params / aggregate FLOPs/sec.
 *
 * No hype: a 1T-parameter BF16 model is 2 TB of weights, and at 3 TB/s
 * (one H100) that is ~1.5 tok/s per sequence. Frontier speed needs
 * frontier aggregate bandwidth, i.e. a lot of GPUs.
 */

import type { ModelConfig, ModelManifest, WeightDtype } from "../contracts.ts";

/** Bytes stored per parameter for a weight dtype. */
export function bytesPerParam(dtype: WeightDtype): number {
  switch (dtype) {
    case "bfloat16":
    case "float16":
      return 2;
    case "float32":
      return 4;
    case "float8_e4m3":
      return 1;
  }
}

/** Number of elements in a tensor shape (0 for a scalar/empty shape). */
export function numElements(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/**
 * Bytes occupied by one tensor. Prefer the safetensors data offsets
 * (authoritative on-disk size); fall back to shape * dtype size.
 */
export function tensorBytes(
  dtype: WeightDtype,
  shape: number[],
  dataOffsets?: [number, number],
): number {
  if (dataOffsets && dataOffsets[1] > dataOffsets[0]) {
    return dataOffsets[1] - dataOffsets[0];
  }
  return numElements(shape) * bytesPerParam(dtype);
}

/** Total weight bytes of a model, summed over every tensor in the manifest. */
export function modelWeightBytes(manifest: ModelManifest): number {
  let total = 0;
  for (const name of Object.keys(manifest.tensors)) {
    const t = manifest.tensors[name];
    total += tensorBytes(t.dtype, t.shape, t.dataOffsets);
  }
  return total;
}

/**
 * KV-cache bytes per generated token, across ALL layers.
 * 2 bytes/element (BF16) * 2 (K and V) * numLayers * numKvHeads * headDim.
 * Per-layer cost is this divided by numLayers.
 */
export function kvBytesPerToken(config: ModelConfig, bytesPerElement = 2): number {
  return 2 * bytesPerElement * config.numHiddenLayers * config.numKeyValueHeads * config.headDim;
}

/** KV-cache bytes per token for a single layer. */
export function kvBytesPerTokenPerLayer(config: ModelConfig, bytesPerElement = 2): number {
  return 2 * bytesPerElement * config.numKeyValueHeads * config.headDim;
}

/**
 * Parameter counts derived from the config (not the manifest).
 * `total` counts every parameter, including all MoE experts.
 * `active` counts the parameters touched by one token: attention + the
 * routed experts only (numExpertsPerToken of them), plus norms.
 */
export function paramsFromConfig(config: ModelConfig): { totalParams: number; activeParams: number } {
  const h = config.hiddenSize;
  const attnHeads = config.numAttentionHeads;
  const kvHeads = config.numKeyValueHeads;
  const d = config.headDim;
  // q, k, v, o projections per layer.
  const attnPerLayer = h * attnHeads * d + h * kvHeads * d + h * kvHeads * d + attnHeads * d * h;
  // SwiGLU MLP: gate + up + down.
  const mlpPerLayer = 3 * h * config.intermediateSize;
  const experts = config.numExperts ?? 0;
  const expertsPerToken = config.numExpertsPerToken ?? 0;
  const moePerLayer = experts > 0 ? experts * mlpPerLayer : mlpPerLayer;
  const moeActivePerLayer = experts > 0 ? expertsPerToken * mlpPerLayer : mlpPerLayer;
  const perLayer = attnPerLayer + moePerLayer;
  const perLayerActive = attnPerLayer + moeActivePerLayer;
  // Embedding (+ lm_head unless tied) and final norm; negligible for FLOPs
  // but counted here so totals roughly match the manifest.
  const embed = config.vocabSize * h;
  const lmHead = config.tieWordEmbeddings ? 0 : config.vocabSize * h;
  const norms = config.numHiddenLayers * 2 * h + h;
  return {
    totalParams: embed + lmHead + norms + config.numHiddenLayers * perLayer,
    activeParams: embed + lmHead + norms + config.numHiddenLayers * perLayerActive,
  };
}

/**
 * Prefill FLOPs per token ≈ 2 * (active) parameters. Dense models touch
 * every parameter; MoE models touch only the routed experts per token.
 */
export function prefillFlopsPerToken(config: ModelConfig): number {
  return 2 * paramsFromConfig(config).activeParams;
}

/** Prefill throughput (tokens/sec) at a given aggregate compute rate. */
export function prefillTokensPerSec(config: ModelConfig, aggregateFlopsPerSec: number): number {
  if (aggregateFlopsPerSec <= 0) return 0;
  return aggregateFlopsPerSec / prefillFlopsPerToken(config);
}

/**
 * Wall-clock time (ms) to prefill a prompt of `promptTokens` tokens.
 * Compute-bound: promptTokens * flopsPerToken / aggregateFlopsPerSec.
 */
export function prefillTimeMs(
  config: ModelConfig,
  promptTokens: number,
  aggregateFlopsPerSec: number,
): number {
  if (aggregateFlopsPerSec <= 0 || promptTokens <= 0) return Number.POSITIVE_INFINITY;
  return (promptTokens * prefillFlopsPerToken(config)) / aggregateFlopsPerSec * 1000;
}

/**
 * Decode throughput (tokens/sec). Memory-bandwidth bound: one full weight
 * read per generated token per sequence in the batch, so a batch of B
 * sequences produces B tokens per weight read.
 *
 * `batchSize` defaults to 1 — the pessimistic per-sequence bound. Pass a
 * larger batch for the batched (throughput) case.
 */
export function decodeTokensPerSec(
  manifest: ModelManifest,
  aggregateBandwidthBytesPerSec: number,
  batchSize = 1,
): number {
  const weights = modelWeightBytes(manifest);
  if (weights <= 0 || aggregateBandwidthBytesPerSec <= 0 || batchSize <= 0) return 0;
  return (batchSize * aggregateBandwidthBytesPerSec) / weights;
}

export interface LatencyEstimate {
  /** Decode throughput, tokens/sec (at the given batch size). */
  tokensPerSec: number;
  /** Time to first token for a `promptTokens` prompt, ms. */
  ttftMs: number;
  /** Prefill wall time alone, ms. */
  prefillMs: number;
  /** One decode step wall time, ms. */
  decodeStepMs: number;
  /** Human-readable bottleneck, e.g. "memory-bandwidth-bound decode". */
  bottleneck: string;
}

/**
 * Combined latency estimate. TTFT = prefill time (compute-bound) + one
 * decode step (one full weight read, bandwidth-bound).
 */
export function estimateLatency(
  manifest: ModelManifest,
  aggregateBandwidthBytesPerSec: number,
  aggregateFlopsPerSec: number,
  promptTokens = 1024,
  batchSize = 1,
): LatencyEstimate {
  const weights = modelWeightBytes(manifest);
  const tokensPerSec = decodeTokensPerSec(manifest, aggregateBandwidthBytesPerSec, batchSize);
  const prefillMs = prefillTimeMs(manifest.config, promptTokens, aggregateFlopsPerSec);
  const decodeStepMs =
    aggregateBandwidthBytesPerSec > 0
      ? (weights / aggregateBandwidthBytesPerSec) * 1000 / Math.max(1, batchSize)
      : Number.POSITIVE_INFINITY;
  const ttftMs = prefillMs + decodeStepMs;
  const prefillTps = prefillTokensPerSec(manifest.config, aggregateFlopsPerSec);
  const bottleneck =
    `memory-bandwidth-bound decode: ${formatNumber(tokensPerSec)} tok/s at batch ${batchSize} ` +
    `(one full ${formatBytes(weights)} weight read per token); ` +
    `prefill compute-bound at ~${formatNumber(prefillTps)} tok/s`;
  return { tokensPerSec, ttftMs, prefillMs, decodeStepMs, bottleneck };
}

// ---------------------------------------------------------------------------
// Human-readable hardware math
// ---------------------------------------------------------------------------

/** Reference H100 HBM read bandwidth used for the worked examples (3 TB/s). */
export const H100_BANDWIDTH_BYTES_PER_SEC = 3e12;

/** Format a byte count as e.g. "2.00 TB", "512.00 MB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "∞";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let v = bytes;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(2)} ${units[i]}`;
}

/** Format a number with thousands separators and up to 2 decimals. */
export function formatNumber(n: number): string {
  if (!Number.isFinite(n)) return "∞";
  return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

/**
 * Honest human-readable lines about what a model costs to serve.
 * Worked for the manifest's own size, plus a 1-H100 and a 64-H100 example.
 * No hype: the numbers are just weight bytes divided by bandwidth.
 */
export function describeHardwareMath(manifest: ModelManifest): string[] {
  const weights = modelWeightBytes(manifest);
  const params = manifest.totalParams;
  const dtype = mostCommonDtype(manifest);
  const h100 = H100_BANDWIDTH_BYTES_PER_SEC;
  const tps1 = decodeTokensPerSec(manifest, h100, 1);
  const tps64 = decodeTokensPerSec(manifest, 64 * h100, 1);
  const tps64Batch32 = decodeTokensPerSec(manifest, 64 * h100, 32);
  const bwFor100 = 100 * weights; // bytes/sec needed for 100 tok/s at batch 1
  const h100sFor100 = bwFor100 / h100;
  const kv = kvBytesPerToken(manifest.config);
  // Prefill FLOPs: 2 * active params. Scale the config-derived active
  // fraction by the manifest's own param count so the number stays
  // consistent with the manifest (dense: active == total; MoE: only the
  // routed experts count). This matters for manifests whose config is a
  // stand-in, e.g. a 1T-param model described by a small config.
  const fromConfig = paramsFromConfig(manifest.config);
  const activeFraction =
    fromConfig.totalParams > 0 ? fromConfig.activeParams / fromConfig.totalParams : 1;
  const activeParams = params * activeFraction;
  const flops = 2 * activeParams;
  const lines: string[] = [
    `${formatNumber(params)} params (${dtype}) = ${formatBytes(weights)} of weights.`,
    `Decode is memory-bandwidth bound: one full weight read per generated token per sequence, ` +
      `so tok/s ≈ aggregate HBM bandwidth / ${formatBytes(weights)}.`,
    `At 1x H100 (~${formatBytes(h100)}/s HBM): ~${formatNumber(tps1)} tok/s per sequence at batch 1.`,
    `At 64x H100 aggregate (~${formatBytes(64 * h100)}/s): ~${formatNumber(tps64)} tok/s per sequence at batch 1, ` +
      `~${formatNumber(tps64Batch32)} tok/s aggregate at batch 32.`,
    `Frontier 100+ tok/s per sequence needs ~${formatBytes(bwFor100)}/s aggregate bandwidth, ` +
      `i.e. ~${formatNumber(Math.ceil(h100sFor100))} H100-class GPUs (at ${formatBytes(h100)}/s each) or equivalent.`,
    `KV cache: ${formatBytes(kv)} per token across all ${manifest.config.numHiddenLayers} layers ` +
      `(${formatBytes(kvBytesPerTokenPerLayer(manifest.config))} per token per layer).`,
    `Prefill is compute bound: ~${formatNumber(flops)} FLOPs per token ` +
      `(2 FLOPs x ${formatNumber(activeParams)} active params).`,
  ];
  return lines;
}

function mostCommonDtype(manifest: ModelManifest): string {
  const counts = new Map<string, number>();
  for (const name of Object.keys(manifest.tensors)) {
    const dt = manifest.tensors[name].dtype;
    counts.set(dt, (counts.get(dt) ?? 0) + 1);
  }
  let best = "bfloat16";
  let bestCount = -1;
  for (const [dt, n] of counts) {
    if (n > bestCount) {
      best = dt;
      bestCount = n;
    }
  }
  return best;
}
