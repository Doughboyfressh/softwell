/**
 * vRAM weight loader tests (src/model/vram-loader.ts).
 * Run with: node --test test/vram-loader.test.ts
 *
 *  - decode correctness: VramWeightLoader values match the disk loader
 *    (SafetensorsWeightLoader) on the same demo-10m bytes
 *  - per-tensor cache + counters: a second loadTensor is a cache hit
 *    (no re-fetch, counters unchanged); close() drops the cache
 *  - nonexistent-directory proof: a manifest pointing at a directory
 *    that does not exist loads fine from canned shard bytes, and the
 *    loader source imports no node:fs at all
 *  - range-fetcher adapter: createTensorFetcherFromRange resolves each
 *    tensor to vaddr + headerLen + dataOffsets[0]
 *  - disk-loader counter: SafetensorsWeightLoader.bytesReadFromDisk
 *    counts only actual shard reads (cache hits do not count)
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  createManifest,
  parseSafetensorsHeader,
  safetensorsHeaderLength,
  serializeSafetensorsHeader,
  writeSafetensorsFile,
} from "../src/model/manifest.ts";
import { loadManifest, openWeightLoader, parseModelConfig } from "../src/model/loader.ts";
import {
  VramWeightLoader,
  createTensorFetcherFromRange,
} from "../src/model/vram-loader.ts";
import type { ModelManifest, VramWeightMap } from "../src/contracts.ts";

function approxEqual(actual: Float32Array, expected: number[], tol: number): void {
  assert.equal(actual.length, expected.length, "length mismatch");
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs((actual[i] as number) - (expected[i] as number));
    assert.ok(diff <= tol, `index ${i}: got ${actual[i]}, want ${expected[i]} (tol ${tol})`);
  }
}

const demo10mDir = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-vram-loader-test-"));
try {
  // --- Fixture: a tiny safetensors file with one tensor per dtype. ---
  const modelDir = path.join(tmp, "tiny");
  fs.mkdirSync(modelDir, { recursive: true });
  writeSafetensorsFile(path.join(modelDir, "model.safetensors"), [
    { name: "w.bf16", dtype: "bfloat16", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
    { name: "w.fp16", dtype: "float16", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
    { name: "w.f32", dtype: "float32", shape: [4], data: new Float32Array([1.5, -2.25, 0.1, 100.0]) },
  ]);
  const rawShard = fs.readFileSync(path.join(modelDir, "model.safetensors"));
  const headerLen = safetensorsHeaderLength(rawShard);
  const manifest = loadManifest(modelDir);

  // Weight map as the registry would build it: one alloc holds the shard.
  const weightMap: VramWeightMap = {};
  for (const [name, meta] of Object.entries(manifest.tensors)) {
    weightMap[name] = {
      allocId: "alloc-1",
      vaddr: 0x1000,
      headerLen,
      dataOffsets: [meta.dataOffsets[0], meta.dataOffsets[1]],
      dtype: meta.dtype,
      shape: [...meta.shape],
    };
  }

  // Fake vRAM: returns the raw tensor bytes for the requested tensor,
  // exactly as a vRAM read at vaddr + headerLen + dataOffsets[0] would.
  const fetchCalls: string[] = [];
  const fetchTensor = async (name: string): Promise<Uint8Array> => {
    fetchCalls.push(name);
    const meta = manifest.tensors[name];
    if (meta === undefined) throw new Error(`no such tensor: ${name}`);
    return rawShard.subarray(
      headerLen + meta.dataOffsets[0],
      headerLen + meta.dataOffsets[1],
    );
  };

  // --- 1. Decode correctness vs the disk loader on the same bytes. ---
  const diskLoader = openWeightLoader(modelDir);
  const vramLoader = new VramWeightLoader(manifest, weightMap, fetchTensor);
  try {
    assert.ok(vramLoader.hasTensor("w.f32"));
    assert.ok(!vramLoader.hasTensor("nope"));
    approxEqual(await vramLoader.loadTensor("w.f32"), [1.5, -2.25, 0.1, 100.0], 1e-7);
    approxEqual(await diskLoader.loadTensor("w.f32"), [1.5, -2.25, 0.1, 100.0], 1e-7);
    approxEqual(await vramLoader.loadTensor("w.bf16"), [1.5, -2.25, 0.1, 100.0], 0.01);
    approxEqual(await vramLoader.loadTensor("w.fp16"), [1.5, -2.25, 0.1, 100.0], 0.001);
    // Cross-check every tensor against the disk loader, value for value.
    for (const name of Object.keys(manifest.tensors)) {
      const fromVram = await vramLoader.loadTensor(name);
      const fromDisk = await diskLoader.loadTensor(name);
      assert.deepEqual(Array.from(fromVram), Array.from(fromDisk), `tensor ${name} differs`);
    }
    console.log("vRAM loader decodes identically to the disk loader: ok");
  } finally {
    diskLoader.close();
  }

  // --- 2. Counters + per-tensor cache. ---
  assert.equal(vramLoader.tensorsFetched, 3, "one fetch per distinct tensor");
  assert.equal(
    vramLoader.bytesFetched,
    8 + 8 + 16,
    "bytesFetched is the raw tensor bytes (bf16 8B + fp16 8B + f32 16B)",
  );
  assert.deepEqual(fetchCalls, ["w.f32", "w.bf16", "w.fp16"]);
  const callsBefore = fetchCalls.length;
  const first = await vramLoader.loadTensor("w.f32");
  const second = await vramLoader.loadTensor("w.f32");
  assert.equal(second, first, "second load returns the cached array");
  assert.equal(fetchCalls.length, callsBefore, "cache hit performs no fetch");
  assert.equal(vramLoader.tensorsFetched, 3, "cache hit does not count");
  assert.equal(vramLoader.bytesFetched, 8 + 8 + 16);
  // Concurrent loads of the same (uncached) tensor share one fetch.
  vramLoader.close();
  const [a, b] = await Promise.all([
    vramLoader.loadTensor("w.f32"),
    vramLoader.loadTensor("w.f32"),
  ]);
  assert.equal(a, b);
  assert.equal(vramLoader.tensorsFetched, 4, "close() drops the cache; one re-fetch");
  console.log("vRAM loader counters + cache + close(): ok");

  // --- 3. Unknown tensors fail loudly. ---
  await assert.rejects(() => vramLoader.loadTensor("missing.tensor"), /tensor not found/);
  assert.equal(vramLoader.hasTensor("missing.tensor"), false);
  const emptyMapLoader = new VramWeightLoader(manifest, {}, fetchTensor);
  assert.equal(emptyMapLoader.hasTensor("w.f32"), false);
  await assert.rejects(() => emptyMapLoader.loadTensor("w.f32"), /not in the vRAM weight map/);
  console.log("vRAM loader rejects unknown / unstaged tensors: ok");

  // --- 4. Nonexistent-directory proof. ---
  // A manifest for a model whose directory does not exist at all, with
  // canned shard bytes served by a fake fetch. The loader has no dir,
  // no fs import, and nothing to read from disk — it must still load.
  const ghostDir = path.join(tmp, "does-not-exist");
  assert.ok(!fs.existsSync(ghostDir), "fixture dir must not exist");
  const ghostHeader = serializeSafetensorsHeader({
    "w.f32": { dtype: "float32", shape: [4], dataOffsets: [0, 16] },
  });
  const ghostData = Buffer.alloc(16);
  const ghostView = new DataView(ghostData.buffer);
  [1.5, -2.25, 0.1, 100.0].forEach((v, i) => ghostView.setFloat32(i * 4, v, true));
  const ghostShard = Buffer.concat([ghostHeader, ghostData]);
  const ghostHeaderLen = ghostHeader.length;
  const ghostManifest: ModelManifest = createManifest(
    "ghost-model",
    parseModelConfig({
      hidden_size: 4,
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
    }),
    { "w.f32": { dtype: "float32", shape: [4], dataOffsets: [0, 16] } },
    [{ filename: "model.safetensors", sizeBytes: ghostShard.length }],
  );
  const ghostMap: VramWeightMap = {
    "w.f32": {
      allocId: "alloc-ghost",
      vaddr: 0x2000,
      headerLen: ghostHeaderLen,
      dataOffsets: [0, 16],
      dtype: "float32",
      shape: [4],
    },
  };
  const ghostLoader = new VramWeightLoader(ghostManifest, ghostMap, async (name) => {
    assert.equal(name, "w.f32");
    return ghostShard.subarray(ghostHeaderLen, ghostHeaderLen + 16);
  });
  approxEqual(await ghostLoader.loadTensor("w.f32"), [1.5, -2.25, 0.1, 100.0], 1e-7);
  assert.equal(ghostLoader.tensorsFetched, 1);
  assert.equal(ghostLoader.bytesFetched, 16);
  // Static proof: the loader module cannot touch disk — it has no fs
  // import (static or dynamic). The doc comment mentions node:fs, so
  // match import statements, not the bare string.
  const loaderSrc = fs.readFileSync(
    fileURLToPath(new URL("../src/model/vram-loader.ts", import.meta.url)),
    "utf-8",
  );
  assert.ok(
    !/from\s+["']node:fs["']/.test(loaderSrc),
    "vram-loader.ts must not import node:fs",
  );
  assert.ok(
    !/import\s*\(\s*["']node:fs["']\s*\)/.test(loaderSrc),
    "vram-loader.ts must not dynamically import node:fs",
  );
  assert.ok(!loaderSrc.includes("readFileSync"), "vram-loader.ts must not read files");
  console.log("vRAM loader loads from a nonexistent model directory: ok");

  // --- 5. Range-fetcher adapter. ---
  const rangeCalls: Array<{ vaddr: number; length: number }> = [];
  const rangeLoader = new VramWeightLoader(
    manifest,
    weightMap,
    createTensorFetcherFromRange(weightMap, async (vaddr, length) => {
      rangeCalls.push({ vaddr, length });
      // Simulate one flat vRAM holding the shard at 0x1000.
      const offset = vaddr - 0x1000;
      return rawShard.subarray(offset, offset + length);
    }),
  );
  approxEqual(await rangeLoader.loadTensor("w.fp16"), [1.5, -2.25, 0.1, 100.0], 0.001);
  const fp16Meta = manifest.tensors["w.fp16"];
  assert.deepEqual(rangeCalls, [
    {
      vaddr: 0x1000 + headerLen + (fp16Meta as { dataOffsets: [number, number] }).dataOffsets[0],
      length: 8,
    },
  ]);
  console.log("range-fetcher adapter resolves vaddr + headerLen + dataOffsets[0]: ok");

  // --- 6. Disk-loader counter (the proof workstream's zero-disk check). ---
  const countingLoader = openWeightLoader(modelDir);
  try {
    assert.equal(countingLoader.bytesReadFromDisk, 0, "nothing read before first load");
    await countingLoader.loadTensor("w.f32");
    assert.equal(
      countingLoader.bytesReadFromDisk,
      rawShard.length,
      "first load reads the whole shard file once",
    );
    await countingLoader.loadTensor("w.bf16");
    assert.equal(
      countingLoader.bytesReadFromDisk,
      rawShard.length,
      "second tensor in the same shard is a cache hit, not a disk read",
    );
  } finally {
    countingLoader.close();
  }
  // A fresh loader instance reads the shard again (its own cache is empty).
  const freshLoader = openWeightLoader(modelDir);
  try {
    await freshLoader.loadTensor("w.f32");
    assert.equal(freshLoader.bytesReadFromDisk, rawShard.length);
  } finally {
    freshLoader.close();
  }
  console.log("disk loader bytesReadFromDisk counts actual shard reads: ok");

  // --- 7. Sanity: the demo-10m manifest parses through the same path. ---
  const demoManifest = loadManifest(demo10mDir);
  assert.ok(Object.keys(demoManifest.tensors).length > 0);
  void parseSafetensorsHeader(rawShard);

  console.log("vRAM loader test: all checks passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
