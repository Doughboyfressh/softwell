/**
 * Smoke test for the model layer (src/model/*).
 * Run with: node test/model.test.ts  (exit 0 on success)
 *
 * Covers: safetensors write -> manifest load -> lazy tensor load
 * (float32, bfloat16, float16 round-trips), a sharded model via
 * model.safetensors.index.json, tokenizer encode/decode round-trip and
 * save/load, and hub URL construction / repo parsing (no network).
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  computeTotals,
  parseSafetensorsHeader,
  serializeSafetensorsHeader,
  writeSafetensorsFile,
} from "../src/model/manifest.ts";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { hubFileUrl, parseRepoId, resolveModelRef } from "../src/model/hub.ts";
import {
  ByteLevelTokenizer,
  loadTokenizer,
  saveTokenizer,
} from "../src/model/tokenizer.ts";

function approxEqual(actual: Float32Array, expected: number[], tol: number): void {
  assert.equal(actual.length, expected.length, "length mismatch");
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs((actual[i] as number) - (expected[i] as number));
    assert.ok(diff <= tol, `index ${i}: got ${actual[i]}, want ${expected[i]} (tol ${tol})`);
  }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-model-test-"));
try {
  // --- 1. Write a tiny safetensors file, parse its header back. ---
  const modelDir = path.join(tmp, "tiny");
  fs.mkdirSync(modelDir, { recursive: true });
  const embed = new Float32Array([0.5, -1.25, 3.0, 0.0, 2.5, -0.75]);
  const norm = new Float32Array([1, 2, 3, 4]);
  writeSafetensorsFile(path.join(modelDir, "model.safetensors"), [
    { name: "model.embed_tokens.weight", dtype: "float32", shape: [3, 2], data: embed },
    { name: "model.norm.weight", dtype: "float32", shape: [4], data: norm },
  ]);
  const raw = fs.readFileSync(path.join(modelDir, "model.safetensors"));
  const header = parseSafetensorsHeader(raw);
  assert.equal(Object.keys(header.tensors).length, 2);
  assert.deepEqual(header.tensors["model.embed_tokens.weight"]?.dataOffsets, [0, 24]);
  assert.deepEqual(header.tensors["model.norm.weight"]?.dataOffsets, [24, 40]);
  assert.deepEqual(computeTotals(header.tensors), { totalParams: 10, totalBytes: 40 });
  // Header serialization is stable: re-serializing the parsed header
  // yields the same bytes the writer produced.
  const reserialized = serializeSafetensorsHeader(header.tensors, header.metadata);
  assert.ok(reserialized.equals(raw.subarray(0, reserialized.length)));

  // --- 2. bfloat16 / float16 write + read round-trip (lossy, so tolerant). ---
  writeSafetensorsFile(path.join(modelDir, "model.safetensors"), [
    { name: "w.bf16", dtype: "bfloat16", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
    { name: "w.fp16", dtype: "float16", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
    { name: "w.f32", dtype: "float32", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
  ]);
  fs.writeFileSync(
    path.join(modelDir, "config.json"),
    JSON.stringify({
      architectures: ["LlamaForCausalLM"],
      vocab_size: 259,
      hidden_size: 4,
      intermediate_size: 8,
      num_hidden_layers: 1,
      num_attention_heads: 2,
      num_key_value_heads: 1,
      max_position_embeddings: 128,
      rms_norm_eps: 1e-5,
      rope_theta: 10000,
      tie_word_embeddings: false,
      bos_token_id: 256,
      eos_token_id: 257,
      pad_token_id: 258,
    }),
    "utf-8",
  );

  const manifest = loadManifest(modelDir);
  assert.equal(manifest.config.hiddenSize, 4);
  assert.equal(manifest.config.numKeyValueHeads, 1);
  assert.equal(manifest.totalParams, 12);
  assert.equal(manifest.files.length, 1);

  const loader = openWeightLoader(modelDir);
  try {
    assert.ok(loader.hasTensor("w.bf16"));
    assert.ok(!loader.hasTensor("nope"));
    approxEqual(await loader.loadTensor("w.f32"), [1.5, -2.25, 0.1, 100.0], 1e-7);
    approxEqual(await loader.loadTensor("w.bf16"), [1.5, -2.25, 0.1, 100.0], 0.01);
    approxEqual(await loader.loadTensor("w.fp16"), [1.5, -2.25, 0.1, 100.0], 0.001);
    // Second load hits the file cache and returns identical values.
    approxEqual(await loader.loadTensor("w.f32"), [1.5, -2.25, 0.1, 100.0], 1e-7);
    await assert.rejects(() => loader.loadTensor("missing.tensor"), /tensor not found/);
  } finally {
    loader.close();
  }

  // --- 3. Sharded model via model.safetensors.index.json. ---
  const shardDir = path.join(tmp, "sharded");
  fs.mkdirSync(shardDir, { recursive: true });
  writeSafetensorsFile(path.join(shardDir, "model-00001-of-00002.safetensors"), [
    { name: "a.weight", dtype: "float32", shape: [2], data: new Float32Array([7, 8]) },
  ]);
  writeSafetensorsFile(path.join(shardDir, "model-00002-of-00002.safetensors"), [
    { name: "b.weight", dtype: "float32", shape: [2], data: new Float32Array([9, 10]) },
  ]);
  fs.writeFileSync(
    path.join(shardDir, "model.safetensors.index.json"),
    JSON.stringify({
      metadata: { total_size: 16 },
      weight_map: {
        "a.weight": "model-00001-of-00002.safetensors",
        "b.weight": "model-00002-of-00002.safetensors",
      },
    }),
    "utf-8",
  );
  fs.writeFileSync(path.join(shardDir, "config.json"), JSON.stringify({ hidden_size: 2 }), "utf-8");
  const shardManifest = loadManifest(shardDir);
  assert.equal(shardManifest.files.length, 2);
  assert.equal(shardManifest.totalParams, 4);
  const shardLoader = openWeightLoader(shardDir);
  try {
    approxEqual(await shardLoader.loadTensor("a.weight"), [7, 8], 0);
    approxEqual(await shardLoader.loadTensor("b.weight"), [9, 10], 0);
  } finally {
    shardLoader.close();
  }

  // --- 4. Tokenizer round-trip + save/load. ---
  const tok = new ByteLevelTokenizer();
  assert.equal(tok.vocabSize, 259);
  assert.equal(tok.bosTokenId, 256);
  assert.equal(tok.eosTokenId, 257);
  const text = "Hello, softwell! 123 — ünïcödé ✓";
  const ids = tok.encode(text);
  assert.equal(ids[0], 256, "BOS first");
  assert.equal(tok.decode(ids), text);
  assert.equal(tok.decode([72, 105]), "Hi");
  assert.equal(tok.decode([256, 72, 257, 105, 258]), "Hi", "specials skipped");
  saveTokenizer(modelDir, tok);
  const loadedTok = loadTokenizer(modelDir);
  assert.equal(loadedTok.bosTokenId, 256);
  assert.equal(loadedTok.eosTokenId, 257);
  assert.equal(loadedTok.decode(loadedTok.encode(text)), text);
  // Missing tokenizer.json falls back to byte-level defaults.
  assert.equal(loadTokenizer(shardDir).vocabSize, 259);

  // --- 5. Hub helpers (no network). ---
  assert.deepEqual(parseRepoId("meta-llama/Llama-3.3-70B-Instruct"), {
    kind: "hub",
    repo: "meta-llama/Llama-3.3-70B-Instruct",
    owner: "meta-llama",
    name: "Llama-3.3-70B-Instruct",
  });
  assert.throws(() => parseRepoId("not-a-repo"), /not a valid Hugging Face repo id/);
  assert.equal(
    hubFileUrl("meta-llama/Llama-3.3-70B-Instruct", "config.json"),
    "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json",
  );
  assert.equal(
    hubFileUrl("meta-llama/Llama-3.3-70B-Instruct", "model-00001-of-00002.safetensors", "v1"),
    "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/v1/model-00001-of-00002.safetensors",
  );
  const hubRef = resolveModelRef("meta-llama/Llama-3.3-70B-Instruct");
  assert.equal(hubRef.kind, "hub");
  const localRef = resolveModelRef(modelDir);
  assert.equal(localRef.kind, "local");
  assert.equal((localRef as { dir: string }).dir, path.resolve(modelDir));

  console.log("model layer smoke test: all checks passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
