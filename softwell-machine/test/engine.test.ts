/**
 * Engine smoke test — run with: node test/engine.test.ts
 *
 * Builds a tiny random-weight Llama-style model entirely in memory
 * (hidden 64, 2 layers, 4 heads, 2 KV heads, vocab 128, intermediate 128),
 * runs embed + forward through a full-model stage executor, greedily
 * generates 5 tokens, and unit-checks RoPE, softmax, and matmul.
 * No files, no network.
 */

import assert from "node:assert/strict";
import type {
  ForwardContext,
  ModelConfig,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  TensorMeta,
  WeightLoader,
} from "../src/contracts.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import {
  applyRope,
  CpuKernels,
  groupedQueryAttention,
  matmul,
  rope,
  softmax,
} from "../src/engine/attention.ts";
import { mulberry32, sampleLogits } from "../src/engine/sampling.ts";

// ---------------------------------------------------------------------------
// Tiny random-weight model (deterministic seed)
// ---------------------------------------------------------------------------

const HIDDEN = 64;
const LAYERS = 2;
const HEADS = 4;
const KV_HEADS = 2;
const HEAD_DIM = 16;
const VOCAB = 128;
const INTERMEDIATE = 128;

const config: ModelConfig = {
  architectures: ["LlamaForCausalLM"],
  vocabSize: VOCAB,
  hiddenSize: HIDDEN,
  intermediateSize: INTERMEDIATE,
  numHiddenLayers: LAYERS,
  numAttentionHeads: HEADS,
  numKeyValueHeads: KV_HEADS,
  headDim: HEAD_DIM,
  maxPositionEmbeddings: 512,
  rmsNormEps: 1e-5,
  ropeTheta: 10000,
  tieWordEmbeddings: false,
  bosTokenId: 1,
  eosTokenId: 2,
  padTokenId: 0,
};

function randn(rng: () => number): number {
  // Box-Muller.
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randWeights(rng: () => number, rows: number, cols: number, scale: number): Float32Array {
  const w = new Float32Array(rows * cols);
  for (let i = 0; i < w.length; i++) w[i] = randn(rng) * scale;
  return w;
}

function buildModel(): { manifest: ModelManifest; loader: WeightLoader } {
  const rng = mulberry32(1234);
  const tensors = new Map<string, Float32Array>();
  const meta: Record<string, TensorMeta> = {};
  const put = (name: string, shape: number[], data: Float32Array): void => {
    tensors.set(name, data);
    meta[name] = { dtype: "float32", shape, dataOffsets: [0, data.byteLength] };
  };
  const w = (rows: number, cols: number): Float32Array =>
    randWeights(rng, rows, cols, 1 / Math.sqrt(cols));
  const ones = (n: number): Float32Array => new Float32Array(n).fill(1);

  put("model.embed_tokens.weight", [VOCAB, HIDDEN], w(VOCAB, HIDDEN));
  for (let i = 0; i < LAYERS; i++) {
    const p = `model.layers.${i}`;
    put(`${p}.input_layernorm.weight`, [HIDDEN], ones(HIDDEN));
    put(`${p}.self_attn.q_proj.weight`, [HEADS * HEAD_DIM, HIDDEN], w(HEADS * HEAD_DIM, HIDDEN));
    put(`${p}.self_attn.k_proj.weight`, [KV_HEADS * HEAD_DIM, HIDDEN], w(KV_HEADS * HEAD_DIM, HIDDEN));
    put(`${p}.self_attn.v_proj.weight`, [KV_HEADS * HEAD_DIM, HIDDEN], w(KV_HEADS * HEAD_DIM, HIDDEN));
    put(`${p}.self_attn.o_proj.weight`, [HIDDEN, HEADS * HEAD_DIM], w(HIDDEN, HEADS * HEAD_DIM));
    put(`${p}.post_attention_layernorm.weight`, [HIDDEN], ones(HIDDEN));
    put(`${p}.mlp.gate_proj.weight`, [INTERMEDIATE, HIDDEN], w(INTERMEDIATE, HIDDEN));
    put(`${p}.mlp.up_proj.weight`, [INTERMEDIATE, HIDDEN], w(INTERMEDIATE, HIDDEN));
    put(`${p}.mlp.down_proj.weight`, [HIDDEN, INTERMEDIATE], w(HIDDEN, INTERMEDIATE));
  }
  put("model.norm.weight", [HIDDEN], ones(HIDDEN));
  put("lm_head.weight", [VOCAB, HIDDEN], w(VOCAB, HIDDEN));

  let totalParams = 0;
  for (const t of tensors.values()) totalParams += t.length;

  const manifest: ModelManifest = {
    format: "safetensors",
    modelId: "test/tiny-llama",
    config,
    metadata: {},
    tensors: meta,
    files: [],
    totalBytes: totalParams * 4,
    totalParams,
  };
  const loader: WeightLoader = {
    manifest,
    hasTensor: (name: string) => tensors.has(name),
    loadTensor: async (name: string) => {
      const t = tensors.get(name);
      if (!t) throw new Error(`fake loader: no tensor ${name}`);
      return t;
    },
    close: () => {},
  };
  return { manifest, loader };
}

// ---------------------------------------------------------------------------
// Minimal in-memory PagedKVCache stub (interface only, per contracts.ts)
// ---------------------------------------------------------------------------

class StubCache implements PagedKVCache {
  readonly blockSize = 16;
  private store = new Map<string, { position: number; key: Float32Array; value: Float32Array }[]>();

  allocate(_seqId: string, _layer: number, _numTokens: number): number[] {
    return [0];
  }
  append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): void {
    const k = `${seqId}:${layer}`;
    const entries = this.store.get(k) ?? [];
    entries.push({ position, key: Float32Array.from(key), value: Float32Array.from(value) });
    this.store.set(k, entries);
  }
  read(seqId: string, layer: number, positions: number[]): { keys: Float32Array; values: Float32Array } {
    const entries = this.store.get(`${seqId}:${layer}`) ?? [];
    const headDim = entries.length > 0 ? entries[0].key.length : 0;
    const keys = new Float32Array(positions.length * headDim);
    const values = new Float32Array(positions.length * headDim);
    for (let i = 0; i < positions.length; i++) {
      const e = entries.find((x) => x.position === positions[i]);
      if (!e) throw new Error(`stub cache: no entry for position ${positions[i]}`);
      keys.set(e.key, i * headDim);
      values.set(e.value, i * headDim);
    }
    return { keys, values };
  }
  free(seqId: string): void {
    for (const k of [...this.store.keys()]) {
      if (k.startsWith(`${seqId}:`)) this.store.delete(k);
    }
  }
  usedBlocks(): number {
    return 0;
  }
  totalBlocks(): number {
    return 1024;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function assertFinite(a: Float32Array, label: string): void {
  for (let i = 0; i < a.length; i++) {
    assert.ok(Number.isFinite(a[i]), `${label}[${i}] is not finite: ${a[i]}`);
  }
}

function assertEqualArrays(a: Float32Array, b: Float32Array, label: string): void {
  assert.equal(a.length, b.length, `${label}: length ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    assert.equal(a[i], b[i], `${label}[${i}]: ${a[i]} vs ${b[i]}`);
  }
}

// ---------------------------------------------------------------------------
// Unit checks: matmul / softmax / RoPE / GQA
// ---------------------------------------------------------------------------

{
  // CpuKernels: the vGPU CPU-fallback bundle exposes every low-level op.
  const required = [
    "matmul",
    "rmsNorm",
    "rope",
    "softmax",
    "swiglu",
    "groupedQueryAttention",
    "moeForward",
  ] as const;
  for (const name of required) {
    assert.equal(typeof CpuKernels[name], "function", `CpuKernels.${name} is a function`);
  }
  // Spot-check the bundle actually works: CpuKernels.matmul == matmul.
  const y = CpuKernels.matmul(new Float32Array([1, 0, 0, 1]), 2, 2, new Float32Array([3, 4]), 2, 1);
  assertEqualArrays(y, new Float32Array([3, 4]), "CpuKernels.matmul");
  console.log("ok CpuKernels exports matmul, rmsNorm, rope, softmax, swiglu, groupedQueryAttention, moeForward");
}

{
  // matmul: identity * x == x
  const id = new Float32Array([1, 0, 0, 1]);
  const x = new Float32Array([3, 4]);
  const y = matmul(id, 2, 2, x, 2, 1);
  assertEqualArrays(y, new Float32Array([3, 4]), "matmul identity");
  console.log("ok matmul identity");
}

{
  // softmax sums to 1 and is stable for large logits.
  const p = softmax(new Float32Array([1000, 1001, 999]));
  assertFinite(p, "softmax large");
  let sum = 0;
  for (const v of p) sum += v;
  assert.ok(Math.abs(sum - 1) < 1e-6, `softmax sums to ${sum}`);
  console.log("ok softmax sums to 1 (stable for large logits)");
}

{
  // RoPE: position 0 is the identity; tables are cos=1, sin=0.
  const tables = rope([0], HEAD_DIM, 10000);
  for (let i = 0; i < tables.cos.length; i++) assert.equal(tables.cos[i], 1, `rope cos[${i}]`);
  for (let i = 0; i < tables.sin.length; i++) assert.equal(tables.sin[i], 0, `rope sin[${i}]`);
  const rng = mulberry32(7);
  const q = new Float32Array(HEADS * HEAD_DIM);
  const k = new Float32Array(KV_HEADS * HEAD_DIM);
  for (let i = 0; i < q.length; i++) q[i] = randn(rng);
  for (let i = 0; i < k.length; i++) k[i] = randn(rng);
  const qBefore = Float32Array.from(q);
  const kBefore = Float32Array.from(k);
  applyRope(q, k, [0], HEADS, HEAD_DIM, 10000, KV_HEADS);
  assertEqualArrays(q, qBefore, "rope pos0 q");
  assertEqualArrays(k, kBefore, "rope pos0 k");
  // And a non-zero position actually rotates (changes values, keeps norm).
  const q2 = Float32Array.from(qBefore);
  const k2 = Float32Array.from(kBefore);
  applyRope(q2, k2, [5], HEADS, HEAD_DIM, 10000, KV_HEADS);
  let changed = false;
  for (let i = 0; i < q2.length; i++) if (q2[i] !== qBefore[i]) changed = true;
  assert.ok(changed, "rope pos5 should rotate q");
  let n0 = 0;
  let n1 = 0;
  for (let i = 0; i < q.length; i++) {
    n0 += qBefore[i] * qBefore[i];
    n1 += q2[i] * q2[i];
  }
  assert.ok(Math.abs(n0 - n1) < 1e-4, "rope preserves norm");
  console.log("ok RoPE position 0 is identity; position 5 rotates, norm preserved");
}

{
  // GQA: one query head group attends over 2 KV positions; with a single
  // KV position the output must equal that position's V (softmax weight 1).
  const q = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0]); // 2 heads, headDim 4
  const k = new Float32Array([1, 0, 0, 0]); // 1 KV head, 1 position
  const v = new Float32Array([2, 3, 4, 5]);
  const out = groupedQueryAttention(q, k, v, 2, 1, 4, { causal: false });
  assertEqualArrays(out, new Float32Array([2, 3, 4, 5, 2, 3, 4, 5]), "gqa single-position");
  console.log("ok groupedQueryAttention single-position == V (GQA head repeat)");
}

// ---------------------------------------------------------------------------
// Full forward pass + greedy generation
// ---------------------------------------------------------------------------

const { manifest, loader } = buildModel();
const backend = createReferenceBackend();
assert.equal(backend.name, "softwell-ts", "backend name");

const assignment: ShardAssignment = {
  workerId: "test-worker",
  stage: 0,
  layerStart: 0,
  layerEnd: LAYERS,
  hasEmbedding: true,
  hasLmHead: true,
  tensorParallelRank: 0,
  tensorParallelSize: 1,
  experts: [],
  estimatedMemoryBytes: 0,
};

const executor = backend.createStageExecutor(manifest, assignment);
await executor.load(loader);
assert.ok(executor.memoryBytes() > 0, "memoryBytes > 0 after load");
console.log(`ok load: stage holds ${executor.memoryBytes()} bytes of weights`);

const prompt = [1, 5, 9, 13];
const embedded = executor.embed(prompt);
assert.deepEqual(embedded.shape, [4, HIDDEN], "embed shape");
assertFinite(embedded.data, "embed");

const cache = new StubCache();
const prefillCtx: ForwardContext = {
  seqIds: ["s1", "s1", "s1", "s1"],
  positions: [0, 1, 2, 3],
  isPrefill: true,
  cache,
};

const t0 = performance.now();
const logits = executor.forward(embedded, prefillCtx);
const prefillMs = performance.now() - t0;
assert.deepEqual(logits.shape, [4, VOCAB], "prefill logits shape");
assertFinite(logits.data, "prefill logits");
console.log(`ok prefill forward: shape [4, ${VOCAB}], ${(prefillMs / 4).toFixed(2)} ms/token`);

// Determinism: same weights + fresh cache => bitwise-identical logits.
const logits2 = executor.forward(embedded, {
  seqIds: ["s1", "s1", "s1", "s1"],
  positions: [0, 1, 2, 3],
  isPrefill: true,
  cache: new StubCache(),
});
assertEqualArrays(logits.data, logits2.data, "deterministic forward");
console.log("ok deterministic: repeated forward is bitwise identical");

// Greedy generation: 5 decode tokens, one position at a time.
const greedy = { temperature: 0, topP: 1, topK: 0, maxTokens: 5 };
const generated: number[] = [];
let lastLogits = logits.data.subarray(3 * VOCAB, 4 * VOCAB);
let decodeMsTotal = 0;
for (let step = 0; step < 5; step++) {
  const tok = sampleLogits(lastLogits, greedy);
  assert.ok(tok >= 0 && tok < VOCAB, `generated token ${tok} in vocab`);
  generated.push(tok);
  const emb = executor.embed([tok]);
  const d0 = performance.now();
  const out = executor.forward(emb, {
    seqIds: ["s1"],
    positions: [4 + step],
    isPrefill: false,
    cache,
  });
  decodeMsTotal += performance.now() - d0;
  assert.deepEqual(out.shape, [1, VOCAB], "decode logits shape");
  assertFinite(out.data, "decode logits");
  lastLogits = out.data;
}
console.log(`ok greedy generation: 5 tokens -> [${generated.join(", ")}]`);
console.log(`decode latency: ${(decodeMsTotal / 5).toFixed(2)} ms/token`);

// Batched rows for two sequences in one forward (grouped per sequence).
const cacheB = new StubCache();
const emb2 = executor.embed([1, 2]);
const out2 = executor.forward(emb2, {
  seqIds: ["a", "b"],
  positions: [0, 0],
  isPrefill: true,
  cache: cacheB,
});
assert.deepEqual(out2.shape, [2, VOCAB], "two-sequence forward shape");
assertFinite(out2.data, "two-sequence logits");
console.log("ok batched forward with two sequences in one call");

console.log("ALL ENGINE TESTS PASSED");
