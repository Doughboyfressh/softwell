/**
 * vNPU dequantization-cache test — run with: node test/vnpu-dequant-cache.test.ts
 *
 * The cache retains the dequantized FP32 values + QuantAnalysis per
 * staged weight key (populated at stage time from the quantize call
 * stageWeights already makes). These tests pin:
 *  - hits: a staged matmul is served from the cache, bit-identical
 *    to the uncached (dequantCacheBytes: 0) device's output and bound
 *  - invalidation on re-stage: staging different data under the same
 *    key never serves the old values
 *  - LRU cap: a cap that fits one weight evicts the other's
 *    dequantization; an oversized single dequantization is never
 *    cached; matmuls stay correct through eviction (miss + re-derive)
 *  - the cache dies with the resident copy: evictWeights, SRAM
 *    eviction, and donor loss all return the cache bytes to 0 for
 *    the affected keys (accounting invariant: usedBytes is exactly
 *    the sum of the retained dequantizations)
 *  - footprint: one lm_head-sized FP8 weight retains exactly
 *    values + analysis bytes (113,842,944)
 *
 * The deliberate INT8-divergence integration test is NOT touched
 * here (test/vnpu-integration.test.ts) — this file only checks the
 * cache mechanism, on synthetic weights.
 */
import assert from "node:assert/strict";
import {
  VirtualNpu,
  dequantBytes,
  quantizeFp8,
  quantizeInt8,
} from "../src/hw/vnpu.ts";
import { matmulTransposeB } from "../src/engine/attention.ts";
import type { IVirtualRam, NodeProvider, VramStats, WorkerInfo } from "../src/contracts.ts";

function makeNode(workerId: string): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 128,
    state: "healthy",
    registeredAt: 0,
    lastHeartbeatAt: 0,
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeRam implements IVirtualRam {
  async alloc(): Promise<never> {
    throw new Error("FakeRam.alloc not used here");
  }
  async free(): Promise<void> {}
  async read(): Promise<never> {
    throw new Error("FakeRam.read not used here");
  }
  async write(): Promise<void> {}
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: 0, usedBytes: 0, freeBytes: 0, pages: 0, nodes: 0 };
  }
}

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

function randomMatrix(rows: number, cols: number, rng: () => number): Float32Array {
  const a = new Float32Array(rows * cols);
  for (let i = 0; i < a.length; i++) a[i] = (rng() - 0.5) * 2;
  return a;
}

function makeNpu(opts: { sramBytes?: number; dequantCacheBytes?: number } = {}): VirtualNpu {
  const nodes: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  return new VirtualNpu({ nodes, ram: new FakeRam(), ...opts });
}

function assertSame(a: Float32Array, b: Float32Array): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.equal(a[i], b[i], `element ${i} differs`);
}

// --- 1. Hits: staged matmuls served from the cache, bit-identical to uncached ---

{
  const rng = mulberry32(7);
  const a = randomMatrix(2, 16, rng);
  const b = randomMatrix(8, 16, rng);
  const cached = makeNpu();
  const uncached = makeNpu({ dequantCacheBytes: 0 });
  for (const format of ["int8", "fp8", "bf16"] as const) {
    cached.stageWeights(`w-${format}`, b, 8, 16, format);
    uncached.stageWeights(`w-${format}`, b, 8, 16, format);
  }
  // Eager population at stage time: every first matmul is already a hit.
  for (const format of ["int8", "fp8", "bf16"] as const) {
    const r1 = await cached.matmulTransposeB(a, 2, 16, b, 8, 16, { format, weightKey: `w-${format}` });
    const r2 = await cached.matmulTransposeB(a, 2, 16, b, 8, 16, { format, weightKey: `w-${format}` });
    const ru = await uncached.matmulTransposeB(a, 2, 16, b, 8, 16, { format, weightKey: `w-${format}` });
    assertSame(r1.output, ru.output);
    assertSame(r2.output, ru.output);
    assert.equal(r1.errorBound, ru.errorBound, `${format}: cached bound != uncached bound`);
    assert.equal(r2.errorBound, ru.errorBound);
  }
  const st = cached.dequantCacheStatus();
  assert.equal(st.entries, 3);
  assert.equal(st.hits, 6);
  assert.equal(st.misses, 0);
  assert.equal(st.evictions, 0);
  assert.equal(uncached.dequantCacheStatus().hits, 0);
  assert.equal(uncached.dequantCacheStatus().misses, 3); // one uncached call per format
  // Non-transposed matmul uses the same cache (B staged as [16, 8]).
  const bT = randomMatrix(16, 8, rng);
  cached.stageWeights("w-fp8-t", bT, 16, 8, "fp8");
  uncached.stageWeights("w-fp8-t", bT, 16, 8, "fp8");
  const r = await cached.matmul(a, 2, 16, bT, 16, 8, { format: "fp8", weightKey: "w-fp8-t" });
  const ru = await uncached.matmul(a, 2, 16, bT, 16, 8, { format: "fp8", weightKey: "w-fp8-t" });
  assertSame(r.output, ru.output);
  console.log("ok 1: cache hits bit-identical to uncached (int8/fp8/bf16, both matmul forms)");
}

// --- 2. Re-stage under the same key invalidates: no stale values ---

{
  const rng = mulberry32(11);
  const a = randomMatrix(2, 16, rng);
  const b1 = randomMatrix(8, 16, rng);
  const b2 = randomMatrix(8, 16, rng);
  const npu = makeNpu();
  npu.stageWeights("w", b1, 8, 16, "fp8");
  const r1 = await npu.matmulTransposeB(a, 2, 16, b1, 8, 16, { format: "fp8", weightKey: "w" });
  // Re-stage different data under the same key.
  npu.stageWeights("w", b2, 8, 16, "fp8");
  const r2 = await npu.matmulTransposeB(a, 2, 16, b2, 8, 16, { format: "fp8", weightKey: "w" });
  const fresh = makeNpu();
  fresh.stageWeights("w", b2, 8, 16, "fp8");
  const rf = await fresh.matmulTransposeB(a, 2, 16, b2, 8, 16, { format: "fp8", weightKey: "w" });
  assertSame(r2.output, rf.output);
  assert.equal(r2.errorBound, rf.errorBound);
  let differs = false;
  for (let i = 0; i < r1.output.length; i++) if (r1.output[i] !== r2.output[i]) differs = true;
  assert.equal(differs, true, "re-staged weight still serves old values");
  console.log("ok 2: re-stage invalidates the cache (new values served, match a fresh device)");
}

// --- 3. LRU cap: one weight's dequant evicts the other's; matmuls stay correct ---

{
  const rng = mulberry32(13);
  const a = randomMatrix(2, 64, rng);
  const bA = randomMatrix(64, 64, rng);
  const bB = randomMatrix(64, 64, rng);
  const oneWeightBytes = dequantBytes(quantizeFp8(bA, 64, 64));
  const npu = makeNpu({ dequantCacheBytes: oneWeightBytes });
  npu.stageWeights("A", bA, 64, 64, "fp8");
  npu.stageWeights("B", bB, 64, 64, "fp8");
  let st = npu.dequantCacheStatus();
  assert.equal(st.entries, 1, "cap fits one weight, both dequants cannot be retained");
  assert.equal(st.usedBytes, oneWeightBytes);
  assert.equal(st.evictions, 1);
  // A was evicted: its matmul misses, re-derives, re-attaches (evicting B), stays correct.
  const rA = await npu.matmulTransposeB(a, 2, 64, bA, 64, 64, { format: "fp8", weightKey: "A" });
  const refA = matmulTransposeB(
    // FP32 reference over the dequantized A values.
    (() => {
      const q = quantizeFp8(a, 2, 64);
      return q.values;
    })(),
    2,
    64,
    quantizeFp8(bA, 64, 64).values,
    64,
    64,
  );
  assertSame(rA.output, refA);
  st = npu.dequantCacheStatus();
  assert.equal(st.entries, 1);
  assert.equal(st.misses, 1);
  assert.equal(st.evictions, 2);
  console.log("ok 3: LRU cap evicts, miss re-derives correctly");
}

// --- 4. Oversized single dequantization is never cached; disabled cache misses ---

{
  const rng = mulberry32(17);
  const a = randomMatrix(2, 64, rng);
  const b = randomMatrix(64, 64, rng);
  const tiny = makeNpu({ dequantCacheBytes: 1024 });
  tiny.stageWeights("w", b, 64, 64, "fp8");
  assert.equal(tiny.dequantCacheStatus().entries, 0);
  assert.equal(tiny.dequantCacheStatus().usedBytes, 0);
  const r = await tiny.matmulTransposeB(a, 2, 64, b, 64, 64, { format: "fp8", weightKey: "w" });
  assert.equal(tiny.dequantCacheStatus().misses, 1);
  assert.equal(tiny.dequantCacheStatus().hits, 0);
  assert.ok(r.errorBound > 0);

  const off = makeNpu({ dequantCacheBytes: 0 });
  off.stageWeights("w", b, 64, 64, "fp8");
  await off.matmulTransposeB(a, 2, 64, b, 64, 64, { format: "fp8", weightKey: "w" });
  await off.matmulTransposeB(a, 2, 64, b, 64, 64, { format: "fp8", weightKey: "w" });
  assert.equal(off.dequantCacheStatus().hits, 0);
  assert.equal(off.dequantCacheStatus().misses, 2);
  assert.equal(off.dequantCacheStatus().usedBytes, 0);
  console.log("ok 4: oversized dequant never cached; cap 0 disables the cache");
}

// --- 5. evictWeights / SRAM eviction / donor loss return the cache bytes ---

{
  const rng = mulberry32(19);
  const b = randomMatrix(32, 32, rng);
  const npu = makeNpu();
  npu.stageWeights("w", b, 32, 32, "int8");
  assert.ok(npu.dequantCacheStatus().usedBytes > 0);
  npu.evictWeights("w");
  assert.equal(npu.dequantCacheStatus().usedBytes, 0);
  assert.equal(npu.dequantCacheStatus().entries, 0);

  // SRAM eviction drops the dequant with the resident copy.
  const small = makeNpu({ sramBytes: 3000 });
  const wA = randomMatrix(32, 32, rng);
  const wB = randomMatrix(32, 32, rng);
  const wC = randomMatrix(32, 32, rng);
  small.stageWeights("A", wA, 32, 32, "int8");
  small.stageWeights("B", wB, 32, 32, "int8");
  small.stageWeights("C", wC, 32, 32, "int8");
  assert.equal(small.sramStatus().evictions, 1, "C evicts A from SRAM");
  const expectBytes =
    dequantBytes(quantizeInt8(wB, 32, 32, "per-channel", "row")) +
    dequantBytes(quantizeInt8(wC, 32, 32, "per-channel", "row"));
  assert.equal(small.dequantCacheStatus().entries, 2);
  assert.equal(small.dequantCacheStatus().usedBytes, expectBytes);

  // Donor loss drops every dequant on that donor.
  const npu2 = makeNpu();
  npu2.listDevices();
  npu2.stageWeights("w", b, 32, 32, "fp8");
  assert.ok(npu2.dequantCacheStatus().usedBytes > 0);
  const lost = await npu2.handleNodeLoss("node-a");
  assert.equal(lost.devicesLost, 1);
  assert.equal(npu2.dequantCacheStatus().usedBytes, 0);
  assert.equal(npu2.dequantCacheStatus().entries, 0);
  console.log("ok 5: evictWeights, SRAM eviction, and donor loss all drop cached dequants");
}

// --- 6. Footprint: one lm_head-sized FP8 weight retains exactly values + analysis ---

{
  const rng = mulberry32(23);
  const b = randomMatrix(49152, 576, rng);
  const npu = makeNpu();
  npu.stageWeights("lm_head", b, 49152, 576, "fp8");
  const st = npu.dequantCacheStatus();
  // values 49152*576*4 = 113,246,208; analysis rowL1 49152*8 + colL1
  // 576*8 + rowErrMax 49152*4 + colErrMax 576*4 = 596,736.
  const expected = 49152 * 576 * 4 + 49152 * 8 + 576 * 8 + 49152 * 4 + 576 * 4;
  assert.equal(st.usedBytes, expected, `footprint ${st.usedBytes} != ${expected}`);
  assert.equal(expected, 113842944);
  assert.equal(st.entries, 1);
  console.log(`ok 6: lm_head-sized FP8 dequant footprint = ${st.usedBytes} bytes (108.6 MiB)`);
}

console.log("vnpu-dequant-cache: all passed");
