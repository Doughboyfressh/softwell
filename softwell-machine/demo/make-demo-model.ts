/**
 * demo/make-demo-model.ts — generate the ~10M-param random-weight demo model.
 *
 * Writes a Llama-style model into demo/models/demo-10m/:
 *   config.json, tokenizer.json, model.safetensors (single shard, float32)
 *
 * Weights are random (seeded, deterministic) — generated text will be
 * incoherent. That is expected: the demo proves the pipeline end to end,
 * not that random weights know English.
 *
 * Usage: node demo/make-demo-model.ts [outDir]
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { writeSafetensorsFile } from "../src/model/manifest.ts";
import type { TensorToWrite } from "../src/model/manifest.ts";
import { ByteLevelTokenizer, saveTokenizer } from "../src/model/tokenizer.ts";

// Seeded PRNG (mulberry32) + Box-Muller normals, so the model is identical
// on every run.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRandn(seed: number): () => number {
  const rand = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    const t = 2 * Math.PI * u2;
    spare = r * Math.sin(t);
    return r * Math.cos(t);
  };
}

function randomTensor(name: string, shape: number[], std: number, randn: () => number): TensorToWrite {
  const count = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(count);
  for (let i = 0; i < count; i++) data[i] = randn() * std;
  return { name, dtype: "float32", shape, data };
}

function onesTensor(name: string, shape: number[]): TensorToWrite {
  const count = shape.reduce((a, b) => a * b, 1);
  return { name, dtype: "float32", shape, data: new Float32Array(count).fill(1) };
}

export const DEMO_CONFIG = {
  architectures: ["LlamaForCausalLM"],
  vocab_size: 259,
  hidden_size: 384,
  intermediate_size: 1024,
  num_hidden_layers: 6,
  num_attention_heads: 8,
  num_key_value_heads: 4,
  head_dim: 48,
  max_position_embeddings: 512,
  rms_norm_eps: 1e-5,
  rope_theta: 10000,
  tie_word_embeddings: true,
  bos_token_id: 256,
  eos_token_id: 257,
  pad_token_id: 258,
};

function main(): void {
  const outDir = path.resolve(process.argv[2] ?? "demo/models/demo-10m");
  fs.mkdirSync(outDir, { recursive: true });
  const randn = makeRandn(1337);
  const cfg = DEMO_CONFIG;
  const H = cfg.hidden_size;
  const I = cfg.intermediate_size;
  const V = cfg.vocab_size;
  const qDim = cfg.num_attention_heads * cfg.head_dim;
  const kvDim = cfg.num_key_value_heads * cfg.head_dim;

  const tensors: TensorToWrite[] = [];
  const embed = randomTensor("model.embed_tokens.weight", [V, H], 0.02, randn);
  tensors.push(embed);
  // Tied lm_head: a copy of the embedding matrix.
  tensors.push({ name: "lm_head.weight", dtype: "float32", shape: [V, H], data: Float32Array.from(embed.data) });

  for (let l = 0; l < cfg.num_hidden_layers; l++) {
    const p = `model.layers.${l}`;
    tensors.push(onesTensor(`${p}.input_layernorm.weight`, [H]));
    tensors.push(randomTensor(`${p}.self_attn.q_proj.weight`, [qDim, H], 0.02, randn));
    tensors.push(randomTensor(`${p}.self_attn.k_proj.weight`, [kvDim, H], 0.02, randn));
    tensors.push(randomTensor(`${p}.self_attn.v_proj.weight`, [kvDim, H], 0.02, randn));
    tensors.push(randomTensor(`${p}.self_attn.o_proj.weight`, [H, qDim], 0.02, randn));
    tensors.push(onesTensor(`${p}.post_attention_layernorm.weight`, [H]));
    tensors.push(randomTensor(`${p}.mlp.gate_proj.weight`, [I, H], 0.02, randn));
    tensors.push(randomTensor(`${p}.mlp.up_proj.weight`, [I, H], 0.02, randn));
    tensors.push(randomTensor(`${p}.mlp.down_proj.weight`, [H, I], 0.02, randn));
  }
  tensors.push(onesTensor("model.norm.weight", [H]));

  const totalParams = tensors.reduce((n, t) => n + t.data.length, 0);
  fs.writeFileSync(path.join(outDir, "config.json"), JSON.stringify(DEMO_CONFIG, null, 2));
  saveTokenizer(outDir, new ByteLevelTokenizer());
  writeSafetensorsFile(path.join(outDir, "model.safetensors"), tensors);
  const bytes = fs.statSync(path.join(outDir, "model.safetensors")).size;
  console.log(`demo model written to ${outDir}`);
  console.log(`  tensors: ${tensors.length}, params: ${(totalParams / 1e6).toFixed(2)}M, file: ${(bytes / 1024 / 1024).toFixed(1)} MB`);
}

main();
