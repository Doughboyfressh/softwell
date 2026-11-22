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
  ForwardContext,
  ModelManifest,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../contracts.ts";
import {
  applyRope,
  groupedQueryAttention,
  matmulTransposeB,
  moeForward,
  rmsNorm,
  swiglu,
  type MoeExpertWeights,
} from "./attention.ts";

export class ReferenceTransformerBackend implements EngineBackend {
  readonly name = "softwell-ts";
  createStageExecutor(manifest: ModelManifest, assignment: ShardAssignment): StageExecutor {
    return new TransformerStageExecutor(manifest, assignment);
  }
}

interface LayerWeights {
  inputNorm: Float32Array;
  qProj: Float32Array;
  kProj: Float32Array;
  vProj: Float32Array;
  oProj: Float32Array;
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
  private weights = new Map<string, Float32Array>();
  private layers = new Map<number, LayerWeights>();
  private embedW: Float32Array | null = null;
  private finalNormW: Float32Array | null = null;
  private lmHeadW: Float32Array | null = null;
  private weightBytes = 0;
  private loaded = false;

  constructor(manifest: ModelManifest, assignment: ShardAssignment) {
    this.manifest = manifest;
    this.assignment = assignment;
  }

  async load(loader: WeightLoader): Promise<void> {
    const cfg = this.manifest.config;
    const { layerStart, layerEnd, hasEmbedding, hasLmHead, experts } = this.assignment;

    const take = async (name: string): Promise<Float32Array> => {
      const data = await loader.loadTensor(name);
      this.weights.set(name, data);
      this.weightBytes += data.byteLength;
      return data;
    };

    if (hasEmbedding) {
      this.embedW = await take("model.embed_tokens.weight");
    }

    for (let i = layerStart; i < layerEnd; i++) {
      const p = `model.layers.${i}`;
      const layer: LayerWeights = {
        inputNorm: await take(`${p}.input_layernorm.weight`),
        qProj: await take(`${p}.self_attn.q_proj.weight`),
        kProj: await take(`${p}.self_attn.k_proj.weight`),
        vProj: await take(`${p}.self_attn.v_proj.weight`),
        oProj: await take(`${p}.self_attn.o_proj.weight`),
        postNorm: await take(`${p}.post_attention_layernorm.weight`),
      };
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
      } else {
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
      if (!this.finalNormW || !this.lmHeadW) {
        throw new Error("forward: stage has lm_head but final norm / lm_head weights not loaded");
      }
      const vocab = cfg.vocabSize;
      const normed = new Float32Array(rows * hidden);
      for (let r = 0; r < rows; r++) {
        const row = x.subarray(r * hidden, (r + 1) * hidden);
        normed.set(rmsNorm(row, this.finalNormW, cfg.rmsNormEps), r * hidden);
      }
      const logits = matmulTransposeB(normed, rows, hidden, this.lmHeadW, vocab, hidden);
      return { data: logits, shape: [rows, vocab] };
    }
    return { data: x, shape: [rows, hidden] };
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

    // --- input_layernorm (all rows at once) ---
    const normed = new Float32Array(rows * hidden);
    for (let r = 0; r < rows; r++) {
      normed.set(rmsNorm(x.subarray(r * hidden, (r + 1) * hidden), layer.inputNorm, cfg.rmsNormEps), r * hidden);
    }

    // --- q/k/v projections (batched matmuls) ---
    const q = matmulTransposeB(normed, rows, hidden, layer.qProj, qDim, hidden);
    const k = matmulTransposeB(normed, rows, hidden, layer.kProj, kvDim, hidden);
    const v = matmulTransposeB(normed, rows, hidden, layer.vProj, kvDim, hidden);

    // --- GQA attention, one row at a time (per-sequence order) ---
    const attnOut = new Float32Array(rows * hidden);
    for (const r of order) {
      const seqId = ctx.seqIds[r];
      const pos = ctx.positions[r];
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
      ctx.cache.append(seqId, layerIdx, pos, kRow, v.subarray(r * kvDim, (r + 1) * kvDim));
      const { keys, values } = ctx.cache.read(seqId, layerIdx, prefix);

      const headOut = groupedQueryAttention(qRow, keys, values, numHeads, numKvHeads, headDim, {
        causal: true,
        queryPosition: pos,
        kvPositions: prefix,
      });
      const projected = matmulTransposeB(headOut, 1, hidden, layer.oProj, hidden, hidden);
      attnOut.set(projected, r * hidden);
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
