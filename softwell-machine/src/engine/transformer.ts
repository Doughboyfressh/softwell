/**
 * softwell-machine — reference transformer backend ("softwell-ts").
 *
 * A genuine transformer forward pass in plain TypeScript/Float32Array:
 * RMSNorm -> GQA self-attention with RoPE and a paged KV cache -> SwiGLU
 * MLP (or MoE routing) -> residual, per layer; final RMSNorm + lm_head on
 * the last stage. Correctness first — this is the reference other backends
 * are checked against.
 */

import type {
  EngineBackend,
  EngineOptions,
  ForwardContext,
  ModelManifest,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../contracts.ts";
import { resolveEngineOptions } from "./options.ts";
import {
  applyRope,
  groupedQueryAttention,
  matmulTransposeB,
  moeForward,
  rmsNorm,
  silu,
  swiglu,
  type MoeExpertWeights,
} from "./attention.ts";
import {
  computeBlockStats,
  extendBlockStats,
  landmarkAttention,
  prefillLandmarkAttention,
  prefillSparseAttention,
  sparseAttention,
  type BlockStats,
} from "./novel-attention.ts";
import {
  attentionPackedKV,
  concatPackedTensors,
  encodeMatrix,
  matmulPackedBT,
  quantizeKV,
  type PackedKV,
  type PackedTensor,
  type SwMode,
} from "./novel-numbers.ts";

export class ReferenceTransformerBackend implements EngineBackend {
  readonly name = "softwell-ts";
  createStageExecutor(
    manifest: ModelManifest,
    assignment: ShardAssignment,
    options?: EngineOptions,
  ): StageExecutor {
    return new TransformerStageExecutor(manifest, assignment, options);
  }
}

/** Fixed KV-block size for the executor's incrementally maintained stats. */
const STATS_BLOCK_SIZE = 32;

/** One layer's SW-code weights (parallel to LayerWeights, fp32 version). */
interface PackedLayerWeights {
  qProj: PackedTensor;
  kProj: PackedTensor;
  vProj: PackedTensor;
  oProj: PackedTensor;
  gateProj?: PackedTensor;
  upProj?: PackedTensor;
  downProj?: PackedTensor;
}

/** Incrementally maintained block stats for one (sequence, layer). */
interface StatsEntry {
  stats: BlockStats;
  /** True when a position inside the covered prefix was overwritten since
   *  the stats were built (speculative rejection); forces a rebuild. */
  dirty: boolean;
}

/** Incrementally maintained quantized KV for one (sequence, layer). */
interface PackedKvEntry {
  kv: PackedKV;
  /** Positions covered (== kv.kvLen). */
  kvLen: number;
  /** True when a position inside the covered prefix was overwritten;
   *  forces a rebuild from the float cache. */
  dirty: boolean;
}

/**
 * SwiGLU MLP for a single row over SW-code weights: down(silu(x @ gateW^T)
 * * (x @ upW^T)). Same shape contract as swiglu in attention.ts; the
 * packed matmuls read the packed stream directly, never materializing
 * a float32 copy of the weights.
 */
export function swigluPacked(
  x: Float32Array,
  gateW: PackedTensor,
  upW: PackedTensor,
  downW: PackedTensor,
): Float32Array {
  const hiddenSize = x.length;
  const intermediateSize = gateW.rows;
  if (upW.rows !== intermediateSize || downW.cols !== intermediateSize || downW.rows !== hiddenSize) {
    throw new Error("swigluPacked: weight shape mismatch");
  }
  const gate = matmulPackedBT(x, 1, hiddenSize, gateW);
  const up = matmulPackedBT(x, 1, hiddenSize, upW);
  const hidden = new Float32Array(intermediateSize);
  for (let i = 0; i < intermediateSize; i++) hidden[i] = silu(gate[i]) * up[i];
  return matmulPackedBT(hidden, 1, intermediateSize, downW);
}

interface LayerWeights {
  inputNorm: Float32Array;
  /** Dense-projection weights; absent when the layer is SW-code packed
   *  (see PackedLayerWeights) — the stage holds one copy, not two. */
  qProj?: Float32Array;
  kProj?: Float32Array;
  vProj?: Float32Array;
  oProj?: Float32Array;
  postNorm: Float32Array;
  // Dense MLP.
  gateProj?: Float32Array;
  upProj?: Float32Array;
  downProj?: Float32Array;
  // MoE MLP.
  router?: Float32Array;
  experts?: Map<number, MoeExpertWeights>;
}

export class TransformerStageExecutor implements StageExecutor {
  readonly assignment: ShardAssignment;
  readonly manifest: ModelManifest;
  /** Resolved engine options (novel paths); all-baseline by default. */
  readonly engineOptions: Required<EngineOptions>;
  private weights = new Map<string, Float32Array>();
  private layers = new Map<number, LayerWeights>();
  /** SW-code weights, present only when weightFormat != "fp32". */
  private packedLayers = new Map<number, PackedLayerWeights>();
  private embedW: Float32Array | null = null;
  private finalNormW: Float32Array | null = null;
  private lmHeadW: Float32Array | null = null;
  private packedLmHeadW: PackedTensor | null = null;
  /** Block stats cache for landmark/sparse attention, key `${seqId}:${layer}`. */
  private statsCache = new Map<string, StatsEntry>();
  /** Quantized KV cache for kvFormat != "fp32", key `${seqId}:${layer}`. */
  private packedKvCache = new Map<string, PackedKvEntry>();
  private weightBytes = 0;
  private loaded = false;

  constructor(manifest: ModelManifest, assignment: ShardAssignment, options: EngineOptions = {}) {
    this.manifest = manifest;
    this.assignment = assignment;
    this.engineOptions = resolveEngineOptions(options);
    if (this.engineOptions.attention !== "exact" && this.engineOptions.kvFormat !== "fp32") {
      throw new Error(
        "TransformerStageExecutor: landmark/sparse attention cannot yet combine with quantized KV " +
          "(attentionPackedKV has no block-mean variant) — use one or the other",
      );
    }
  }

  async load(loader: WeightLoader): Promise<void> {
    const cfg = this.manifest.config;
    const { layerStart, layerEnd, hasEmbedding, hasLmHead, experts } = this.assignment;
    const packedMode = this.engineOptions.weightFormat !== "fp32" ? (this.engineOptions.weightFormat as SwMode) : null;
    if (packedMode !== null && cfg.numExperts !== undefined && cfg.numExperts > 0) {
      throw new Error(
        `TransformerStageExecutor.load: SW-code weights (${packedMode}) are not implemented for MoE experts yet — ` +
          `use --weights fp32 for MoE models`,
      );
    }

    const take = async (name: string): Promise<Float32Array> => {
      const data = await loader.loadTensor(name);
      this.weights.set(name, data);
      this.weightBytes += data.byteLength;
      return data;
    };

    // Load a projection weight, SW-code it, and keep only the packed form.
    // The fp32 tensor is never retained: the packed matmuls never need it.
    const takePacked = async (name: string, rows: number, cols: number): Promise<PackedTensor> => {
      const data = await loader.loadTensor(name);
      if (data.length !== rows * cols) {
        throw new Error(`takePacked: ${name} has ${data.length} elements, expected ${rows}x${cols}`);
      }
      const packed = encodeMatrix(data, rows, cols, packedMode as SwMode);
      this.weightBytes += packed.data.length + packed.rowOffsets.byteLength + 16;
      return packed;
    };

    if (hasEmbedding) {
      this.embedW = await take("model.embed_tokens.weight");
    }

    for (let i = layerStart; i < layerEnd; i++) {
      const p = `model.layers.${i}`;
      const layer: LayerWeights = {
        inputNorm: await take(`${p}.input_layernorm.weight`),
        postNorm: await take(`${p}.post_attention_layernorm.weight`),
      };
      if (packedMode !== null) {
        const hidden = cfg.hiddenSize;
        const qDim = cfg.numAttentionHeads * cfg.headDim;
        const kvDim = cfg.numKeyValueHeads * cfg.headDim;
        const inter = cfg.intermediateSize;
        this.packedLayers.set(i, {
          qProj: await takePacked(`${p}.self_attn.q_proj.weight`, qDim, hidden),
          kProj: await takePacked(`${p}.self_attn.k_proj.weight`, kvDim, hidden),
          vProj: await takePacked(`${p}.self_attn.v_proj.weight`, kvDim, hidden),
          oProj: await takePacked(`${p}.self_attn.o_proj.weight`, hidden, qDim),
          gateProj: await takePacked(`${p}.mlp.gate_proj.weight`, inter, hidden),
          upProj: await takePacked(`${p}.mlp.up_proj.weight`, inter, hidden),
          downProj: await takePacked(`${p}.mlp.down_proj.weight`, hidden, inter),
        });
      } else {
        layer.qProj = await take(`${p}.self_attn.q_proj.weight`);
        layer.kProj = await take(`${p}.self_attn.k_proj.weight`);
        layer.vProj = await take(`${p}.self_attn.v_proj.weight`);
        layer.oProj = await take(`${p}.self_attn.o_proj.weight`);
      }
      if (cfg.numExperts !== undefined && cfg.numExperts > 0) {
        layer.router = await take(`${p}.mlp.router.weight`);
        // Load only this stage's assigned experts; an empty assignment list
        // means "all experts" (single-worker / full-model stage).
        const expertIds = experts.length > 0 ? experts : Array.from({ length: cfg.numExperts }, (_, e) => e);
        layer.experts = new Map<number, MoeExpertWeights>();
        for (const e of expertIds) {
          const ep = `${p}.mlp.experts.${e}`;
          layer.experts.set(e, {
            gate: await take(`${ep}.gate_proj.weight`),
            up: await take(`${ep}.up_proj.weight`),
            down: await take(`${ep}.down_proj.weight`),
          });
        }
      } else if (packedMode === null) {
        layer.gateProj = await take(`${p}.mlp.gate_proj.weight`);
        layer.upProj = await take(`${p}.mlp.up_proj.weight`);
        layer.downProj = await take(`${p}.mlp.down_proj.weight`);
      }
      this.layers.set(i, layer);
    }

    if (hasLmHead) {
      this.finalNormW = await take("model.norm.weight");
      if (loader.hasTensor("lm_head.weight")) {
        this.lmHeadW = await take("lm_head.weight");
      } else if (cfg.tieWordEmbeddings && this.embedW) {
        // Tied embeddings: lm_head shares the embedding matrix.
        this.lmHeadW = this.embedW;
      } else {
        throw new Error("TransformerStageExecutor.load: stage has lm_head but no lm_head.weight tensor");
      }
      if (packedMode !== null && this.lmHeadW !== null) {
        const vocab = cfg.vocabSize;
        const hidden = cfg.hiddenSize;
        this.packedLmHeadW = encodeMatrix(this.lmHeadW, vocab, hidden, packedMode);
        this.weightBytes += this.packedLmHeadW.data.length + this.packedLmHeadW.rowOffsets.byteLength + 16;
        // Keep the fp32 lm_head only when it is NOT the shared embedding
        // (tied embeddings still need it for embed()); otherwise drop it.
        if (this.lmHeadW !== this.embedW) {
          this.weightBytes -= this.lmHeadW.byteLength;
          this.weights.delete("lm_head.weight");
          this.lmHeadW = null;
        }
      }
    }

    this.loaded = true;
  }

  embed(tokenIds: number[]): Tensor {
    if (!this.embedW) {
      throw new Error("TransformerStageExecutor.embed: this stage has no embedding weights");
    }
    const hidden = this.manifest.config.hiddenSize;
    const vocab = this.manifest.config.vocabSize;
    const data = new Float32Array(tokenIds.length * hidden);
    for (let r = 0; r < tokenIds.length; r++) {
      const tok = tokenIds[r];
      if (tok < 0 || tok >= vocab) throw new Error(`embed: token id ${tok} out of vocab range [0, ${vocab})`);
      data.set(this.embedW.subarray(tok * hidden, (tok + 1) * hidden), r * hidden);
    }
    return { data, shape: [tokenIds.length, hidden] };
  }

  forward(input: Tensor, ctx: ForwardContext): Tensor {
    if (!this.loaded) throw new Error("TransformerStageExecutor.forward: load() has not been called");
    const cfg = this.manifest.config;
    const hidden = cfg.hiddenSize;
    const rows = input.shape[0];
    if (input.data.length !== rows * hidden) {
      throw new Error(`forward: input length ${input.data.length}, expected ${rows * hidden}`);
    }
    if (ctx.seqIds.length !== rows || ctx.positions.length !== rows) {
      throw new Error(
        `forward: ctx seqIds/positions length ${ctx.seqIds.length}/${ctx.positions.length}, expected ${rows}`,
      );
    }

    // Work on a copy of the hidden states.
    let x: Float32Array = new Float32Array(input.data);

    // Process rows grouped per sequence, in position order, so KV cache
    // appends for a sequence happen in order. `order` maps the sorted
    // iteration position back to the original row index.
    const order = Array.from({ length: rows }, (_, r) => r);
    order.sort((a, b) => {
      if (ctx.seqIds[a] !== ctx.seqIds[b]) return ctx.seqIds[a] < ctx.seqIds[b] ? -1 : 1;
      return ctx.positions[a] - ctx.positions[b];
    });

    for (let layerIdx = this.assignment.layerStart; layerIdx < this.assignment.layerEnd; layerIdx++) {
      const layer = this.layers.get(layerIdx);
      if (!layer) throw new Error(`forward: layer ${layerIdx} weights not loaded`);
      x = this.forwardLayer(x, rows, layer, layerIdx, ctx, order);
    }

    if (this.assignment.hasLmHead) {
      if (!this.finalNormW || (this.lmHeadW === null && this.packedLmHeadW === null)) {
        throw new Error("forward: stage has lm_head but final norm / lm_head weights not loaded");
      }
      const vocab = cfg.vocabSize;
      const normed = new Float32Array(rows * hidden);
      for (let r = 0; r < rows; r++) {
        const row = x.subarray(r * hidden, (r + 1) * hidden);
        normed.set(rmsNorm(row, this.finalNormW, cfg.rmsNormEps), r * hidden);
      }
      const logits =
        this.packedLmHeadW !== null
          ? matmulPackedBT(normed, rows, hidden, this.packedLmHeadW)
          : matmulTransposeB(normed, rows, hidden, this.lmHeadW as Float32Array, vocab, hidden);
      return { data: logits, shape: [rows, vocab] };
    }
    return { data: x, shape: [rows, hidden] };
  }

  /**
   * Record that a K/V row is about to be appended at `pos` for
   * (seqId, layer). An append *inside* the already-covered prefix is an
   * overwrite (speculative rejection / Jacobi repair): the incrementally
   * maintained stats and packed KV for that prefix are stale and must be
   * rebuilt on next use. Appends at the end only extend them.
   */
  private noteAppend(seqId: string, layerIdx: number, pos: number): void {
    const key = `${seqId}:${layerIdx}`;
    const s = this.statsCache.get(key);
    if (s !== undefined && pos < s.stats.kvLen) s.dirty = true;
    const p = this.packedKvCache.get(key);
    if (p !== undefined && pos < p.kvLen) p.dirty = true;
  }

  /**
   * Extend the incrementally maintained packed KV for (seqId, layer) with
   * one freshly appended row. No-op when kvFormat is fp32, when there is
   * no entry yet (built lazily at attention time), when the entry is dirty
   * (rebuilt lazily), or when the append is not sequential. The row's K/V
   * here are post-RoPE, matching quantizeKV's contract.
   */
  private noteAppendedKv(
    seqId: string,
    layerIdx: number,
    pos: number,
    kRow: Float32Array,
    vRow: Float32Array,
  ): void {
    if (this.engineOptions.kvFormat === "fp32") return;
    const key = `${seqId}:${layerIdx}`;
    const entry = this.packedKvCache.get(key);
    if (entry === undefined || entry.dirty) return;
    if (pos !== entry.kvLen) {
      entry.dirty = true; // non-sequential append: rebuild lazily
      return;
    }
    const cfg = this.manifest.config;
    const mode = this.engineOptions.kvFormat as SwMode;
    const one = quantizeKV(kRow, vRow, 1, cfg.numKeyValueHeads, cfg.headDim, mode);
    entry.kv = {
      k: concatPackedTensors(entry.kv.k, one.k),
      v: concatPackedTensors(entry.kv.v, one.v),
      kvLen: entry.kvLen + 1,
      numKvHeads: cfg.numKeyValueHeads,
      headDim: cfg.headDim,
      kMode: mode,
      vMode: mode,
    };
    entry.kvLen += 1;
  }

  /**
   * Block stats for the prefix `keys`/`values` (length kvLen rows),
   * reused across decode steps: a longer prefix is extended in O(B·d)
   * via extendBlockStats, and only an overwrite (dirty) or a shrink
   * forces a full rebuild. The block size is fixed (STATS_BLOCK_SIZE)
   * so the cached block boundaries stay valid as the prefix grows.
   */
  private blockStatsFor(seqId: string, layerIdx: number, keys: Float32Array, values: Float32Array): BlockStats {
    const cfg = this.manifest.config;
    const kvLen = keys.length / (cfg.numKeyValueHeads * cfg.headDim);
    const key = `${seqId}:${layerIdx}`;
    let entry = this.statsCache.get(key);
    if (
      entry === undefined ||
      entry.dirty ||
      entry.stats.kvLen > kvLen
    ) {
      const built = computeBlockStats(keys, values, cfg.numKeyValueHeads, cfg.headDim, STATS_BLOCK_SIZE);
      entry = { stats: built.stats, dirty: false };
      this.statsCache.set(key, entry);
      return entry.stats;
    }
    if (entry.stats.kvLen === kvLen) return entry.stats;
    const extended = extendBlockStats(entry.stats, keys, values, kvLen);
    entry = { stats: extended.stats, dirty: false };
    this.statsCache.set(key, entry);
    return entry.stats;
  }

  /** Quantized KV for the prefix `keys`/`values`, built lazily and
   *  extended incrementally by noteAppendedKv; rebuilt from the float
   *  cache (source of truth) when dirty or out of sync. */
  private packedKvFor(seqId: string, layerIdx: number, keys: Float32Array, values: Float32Array): PackedKV {
    const cfg = this.manifest.config;
    const kvLen = keys.length / (cfg.numKeyValueHeads * cfg.headDim);
    const key = `${seqId}:${layerIdx}`;
    let entry = this.packedKvCache.get(key);
    if (entry === undefined || entry.dirty || entry.kvLen !== kvLen) {
      const kv = quantizeKV(
        keys,
        values,
        kvLen,
        cfg.numKeyValueHeads,
        cfg.headDim,
        this.engineOptions.kvFormat as SwMode,
      );
      entry = { kv, kvLen, dirty: false };
      this.packedKvCache.set(key, entry);
    }
    return entry.kv;
  }

  /** One query row's attention under the configured attention/KV modes. */
  private attendRow(
    qRow: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    seqId: string,
    layerIdx: number,
    pos: number,
    prefix: number[],
  ): Float32Array {
    const cfg = this.manifest.config;
    const numHeads = cfg.numAttentionHeads;
    const numKvHeads = cfg.numKeyValueHeads;
    const headDim = cfg.headDim;
    const opts = { causal: true, queryPosition: pos, kvPositions: prefix };
    if (this.engineOptions.kvFormat !== "fp32") {
      const kv = this.packedKvFor(seqId, layerIdx, keys, values);
      return attentionPackedKV(qRow, kv, numHeads, opts);
    }
    switch (this.engineOptions.attention) {
      case "landmark": {
        const stats = this.blockStatsFor(seqId, layerIdx, keys, values);
        return landmarkAttention(qRow, keys, values, numHeads, numKvHeads, headDim, {
          ...opts,
          stats,
          blockSize: STATS_BLOCK_SIZE,
        }).output;
      }
      case "sparse": {
        const stats = this.blockStatsFor(seqId, layerIdx, keys, values);
        return sparseAttention(qRow, keys, values, numHeads, numKvHeads, headDim, {
          ...opts,
          stats,
          blockSize: STATS_BLOCK_SIZE,
        }).output;
      }
      default:
        return groupedQueryAttention(qRow, keys, values, numHeads, numKvHeads, headDim, opts);
    }
  }

  /**
   * Prefill attention for a whole batch under landmark/sparse mode: one
   * prefillLandmarkAttention / prefillSparseAttention call per sequence,
   * which builds the block stats once and reuses them across all rows —
   * that reuse is the O(n^(3/2) d) win. Rows are grouped by sequence
   * (order is already per-sequence, position-sorted). Sequences whose
   * prefix is longer than their prompt in this batch (a continued
   * prefill, not a fresh one) fall back to the per-row path.
   */
  private attendPrefillNovel(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    layer: LayerWeights,
    layerIdx: number,
    ctx: ForwardContext,
    order: number[],
    attnOut: Float32Array,
  ): void {
    const cfg = this.manifest.config;
    const hidden = cfg.hiddenSize;
    const numHeads = cfg.numAttentionHeads;
    const numKvHeads = cfg.numKeyValueHeads;
    const headDim = cfg.headDim;
    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;
    const packed = this.packedLayers.get(layerIdx);
    const bySeq = new Map<string, number[]>();
    for (const r of order) {
      const seqId = ctx.seqIds[r] as string;
      let list = bySeq.get(seqId);
      if (list === undefined) {
        list = [];
        bySeq.set(seqId, list);
      }
      list.push(r);
    }
    for (const [seqId, rs] of bySeq) {
      const n = rs.length;
      const qSeq = new Float32Array(n * qDim);
      const positions: number[] = [];
      for (let j = 0; j < n; j++) {
        const r = rs[j] as number;
        const pos = ctx.positions[r] as number;
        positions.push(pos);
        const qRow = q.subarray(r * qDim, (r + 1) * qDim);
        const kRow = k.subarray(r * kvDim, (r + 1) * kvDim);
        applyRope(qRow, kRow, [pos], numHeads, headDim, cfg.ropeTheta, numKvHeads);
        qSeq.set(qRow, j * qDim);
        this.noteAppend(seqId, layerIdx, pos);
        ctx.cache.append(seqId, layerIdx, pos, kRow, v.subarray(r * kvDim, (r + 1) * kvDim));
        this.noteAppendedKv(seqId, layerIdx, pos, kRow, v.subarray(r * kvDim, (r + 1) * kvDim));
      }
      const lastPos = positions[n - 1] as number;
      const prefix: number[] = [];
      for (let p = 0; p <= lastPos; p++) prefix.push(p);
      if (prefix.length !== n) {
        // Continued prefill, not a fresh one: the prefix is longer than
        // this batch's rows, so the whole-sequence prefill functions do
        // not apply — fall back to the per-row path for this sequence.
        for (const r of rs) {
          const pos = ctx.positions[r] as number;
          const rowPrefix: number[] = [];
          for (let p = 0; p <= pos; p++) rowPrefix.push(p);
          const { keys, values } = ctx.cache.read(seqId, layerIdx, rowPrefix);
          const headOut = this.attendRow(
            q.subarray(r * qDim, (r + 1) * qDim),
            keys,
            values,
            seqId,
            layerIdx,
            pos,
            rowPrefix,
          );
          const projected = packed
            ? matmulPackedBT(headOut, 1, hidden, packed.oProj)
            : matmulTransposeB(headOut, 1, hidden, layer.oProj as Float32Array, hidden, hidden);
          attnOut.set(projected, r * hidden);
        }
        continue;
      }
      const { keys, values } = ctx.cache.read(seqId, layerIdx, prefix);
      const result =
        this.engineOptions.attention === "landmark"
          ? prefillLandmarkAttention(qSeq, keys, values, numHeads, numKvHeads, headDim)
          : prefillSparseAttention(qSeq, keys, values, numHeads, numKvHeads, headDim);
      for (let j = 0; j < n; j++) {
        const r = rs[j] as number;
        const headOut = result.output.subarray(j * hidden, (j + 1) * hidden);
        const projected = packed
          ? matmulPackedBT(headOut, 1, hidden, packed.oProj)
          : matmulTransposeB(headOut, 1, hidden, layer.oProj as Float32Array, hidden, hidden);
        attnOut.set(projected, r * hidden);
      }
    }
  }

  /** One transformer layer over all rows; `order` is the per-sequence row order. */
  private forwardLayer(
    x: Float32Array,
    rows: number,
    layer: LayerWeights,
    layerIdx: number,
    ctx: ForwardContext,
    order: number[],
  ): Float32Array {
    const cfg = this.manifest.config;
    const hidden = cfg.hiddenSize;
    const numHeads = cfg.numAttentionHeads;
    const numKvHeads = cfg.numKeyValueHeads;
    const headDim = cfg.headDim;
    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;
    const packed = this.packedLayers.get(layerIdx);

    // --- input_layernorm (all rows at once) ---
    const normed = new Float32Array(rows * hidden);
    for (let r = 0; r < rows; r++) {
      normed.set(rmsNorm(x.subarray(r * hidden, (r + 1) * hidden), layer.inputNorm, cfg.rmsNormEps), r * hidden);
    }

    // --- q/k/v projections (batched matmuls; packed when SW-code is on) ---
    const q =
      packed !== undefined
        ? matmulPackedBT(normed, rows, hidden, packed.qProj)
        : matmulTransposeB(normed, rows, hidden, layer.qProj as Float32Array, qDim, hidden);
    const k =
      packed !== undefined
        ? matmulPackedBT(normed, rows, hidden, packed.kProj)
        : matmulTransposeB(normed, rows, hidden, layer.kProj as Float32Array, kvDim, hidden);
    const v =
      packed !== undefined
        ? matmulPackedBT(normed, rows, hidden, packed.vProj)
        : matmulTransposeB(normed, rows, hidden, layer.vProj as Float32Array, kvDim, hidden);

    // --- attention ---
    const attnOut = new Float32Array(rows * hidden);
    const usePrefillNovel =
      ctx.isPrefill && this.engineOptions.attention !== "exact" && this.engineOptions.kvFormat === "fp32";
    if (usePrefillNovel) {
      this.attendPrefillNovel(q, k, v, layer, layerIdx, ctx, order, attnOut);
    } else {
      // GQA attention, one row at a time (per-sequence order).
      for (const r of order) {
        const seqId = ctx.seqIds[r] as string;
        const pos = ctx.positions[r] as number;
        // The caller supplies absolute positions; the cache holds everything
        // appended in earlier forwards plus this forward's earlier rows.
        // We read the full prefix [0..pos] for this sequence.
        const prefix: number[] = [];
        for (let p = 0; p <= pos; p++) prefix.push(p);

        // RoPE on this row's q/k (in place, on the row slices).
        const qRow = q.subarray(r * qDim, (r + 1) * qDim);
        const kRow = k.subarray(r * kvDim, (r + 1) * kvDim);
        applyRope(qRow, kRow, [pos], numHeads, headDim, cfg.ropeTheta, numKvHeads);

        // Append this row's K/V to the cache, then read the prefix back.
        this.noteAppend(seqId, layerIdx, pos);
        ctx.cache.append(seqId, layerIdx, pos, kRow, v.subarray(r * kvDim, (r + 1) * kvDim));
        this.noteAppendedKv(seqId, layerIdx, pos, kRow, v.subarray(r * kvDim, (r + 1) * kvDim));
        const { keys, values } = ctx.cache.read(seqId, layerIdx, prefix);

        const headOut = this.attendRow(qRow, keys, values, seqId, layerIdx, pos, prefix);
        const projected =
          packed !== undefined
            ? matmulPackedBT(headOut, 1, hidden, packed.oProj)
            : matmulTransposeB(headOut, 1, hidden, layer.oProj as Float32Array, hidden, hidden);
        attnOut.set(projected, r * hidden);
      }
    }

    // --- residual 1 ---
    const afterAttn = new Float32Array(rows * hidden);
    for (let i = 0; i < rows * hidden; i++) afterAttn[i] = x[i] + attnOut[i];

    // --- post_attention_layernorm + MLP ---
    const out = new Float32Array(rows * hidden);
    for (let r = 0; r < rows; r++) {
      const row = afterAttn.subarray(r * hidden, (r + 1) * hidden);
      const h = rmsNorm(row, layer.postNorm, cfg.rmsNormEps);
      let mlpOut: Float32Array;
      if (layer.experts && layer.router) {
        mlpOut = moeForward(
          h,
          [...layer.experts.values()],
          layer.router,
          cfg.numExpertsPerToken ?? 2,
          cfg.normTopKProb ?? false,
          [...layer.experts.keys()],
        );
      } else if (packed !== undefined && packed.gateProj && packed.upProj && packed.downProj) {
        mlpOut = swigluPacked(h, packed.gateProj, packed.upProj, packed.downProj);
      } else if (layer.gateProj && layer.upProj && layer.downProj) {
        mlpOut = swiglu(h, layer.gateProj, layer.upProj, layer.downProj);
      } else {
        throw new Error(`forward: layer ${layerIdx} has no MLP weights`);
      }
      for (let i = 0; i < hidden; i++) out[r * hidden + i] = row[i] + mlpOut[i];
    }
    return out;
  }

  memoryBytes(): number {
    return this.weightBytes;
  }
}
