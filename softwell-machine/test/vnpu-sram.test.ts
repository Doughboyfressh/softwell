/**
 * vNPU SRAM test — run with: node test/vnpu-sram.test.ts (exit 0 on pass).
 *
 * Covers the vNPU memory model (docs/VNPU.md §2, design parameter D1):
 *  - weights stage from pooled vRAM into SRAM (stageFromVram) and from
 *    fp32 arrays (stageWeights); a staged weight serves matmuls from
 *    SRAM (sramHit) with identical results to the on-the-fly path
 *  - fit: small weights are resident; the counters (hits / used bytes)
 *    match the staging
 *  - spill: a single weight larger than the whole SRAM is recorded as
 *    a spill (resident: false) and later uses count misses but still
 *    produce correct results by streaming from vRAM
 *  - eviction: staging past capacity evicts LRU residents (counted);
 *    an evicted weight's next use is a miss, still correct
 *  - node loss: a dead donor's device disappears and its resident SRAM
 *    bytes are reported lost; later uses of those keys are misses, not
 *    errors (the weights live in vRAM and re-fetch)
 *  - unknown keys throw; explicit eviction works
 *
 * The SRAM capacity is overridden per test (default D1 is 256 MiB, so
 * spill/eviction tests use a small override instead of staging
 * hundreds of megabytes). The override exists for tests only.
 */
import assert from "node:assert/strict";
import { VirtualNpu, VNPU_SRAM_BYTES } from "../src/hw/vnpu.ts";
import { encodeFromFloat32 } from "../src/hw/vgpu.ts";
import { matmulTransposeB } from "../src/engine/attention.ts";
import type { IVirtualRam, NodeProvider, VramAllocation, VramStats, WorkerInfo } from "../src/contracts.ts";

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

function makeNode(workerId: string, state: WorkerInfo["state"] = "healthy"): WorkerInfo {
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
    state,
    registeredAt: 0,
    lastHeartbeatAt: 0,
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

/** Fake vRAM: one flat byte store, bump-pointer allocations. */
class FakeRam implements IVirtualRam {
  private mem = new Uint8Array(4 * 1024 * 1024);
  private next = 0;
  private counter = 0;

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    const id = `alloc-${++this.counter}`;
    const alloc: VramAllocation = { id, vaddr: this.next, sizeBytes, pages: [], replicas: 1 };
    this.next += sizeBytes;
    return alloc;
  }
  async free(): Promise<void> {}
  async read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.mem.slice(vaddr, vaddr + length);
  }
  async write(vaddr: number, data: Uint8Array): Promise<void> {
    this.mem.set(data, vaddr);
  }
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: this.mem.length, usedBytes: this.next, freeBytes: this.mem.length - this.next, pages: 1, nodes: 1 };
  }
}

function randMatrix(rand: () => number, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < out.length; i++) out[i] = rand() * 2 - 1;
  return out;
}

function maxAbsErr(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.abs(a[i] - b[i]);
    // Fail loudly on NaN/Inf: a NaN diff compares false against every
    // bound, so a silent max() would report a 0 error for garbage.
    assert.ok(Number.isFinite(e), `maxAbsErr: non-finite diff at index ${i} (${a[i]} vs ${b[i]})`);
    if (e > m) m = e;
  }
  return m;
}

// --- fit: stage small weights, serve matmuls from SRAM -------------------------

{
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram });
  const rand = mulberry32(101);
  const w = randMatrix(rand, 8, 16);
  const staged = npu.stageWeights("layer0.q_proj", w, 8, 16, "int8");
  assert.equal(staged.resident, true);
  assert.equal(staged.deviceId, "vnpu-node-a");
  // int8: 8*16 codes + 8 per-channel scales * 4 bytes + 16 header.
  assert.equal(staged.bytes, 8 * 16 + 8 * 4 + 16);
  assert.equal(npu.resident("layer0.q_proj"), true);
  assert.equal(npu.sramStatus().usedBytes, staged.bytes);
  assert.equal(npu.sramStatus().entries, 1);

  const a = randMatrix(rand, 2, 16);
  const ref = matmulTransposeB(a, 2, 16, w, 8, 16);
  const r = await npu.matmulTransposeB(a, 2, 16, w, 8, 16, { weightKey: "layer0.q_proj" });
  assert.equal(r.sramHit, true, "staged weight served from SRAM");
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound);
  assert.equal(npu.stats().sramHits, 1);
  assert.equal(npu.stats().sramMisses, 0);
  console.log(`ok fit: staged int8 weight resident (${staged.bytes} bytes), matmul served from SRAM`);
}

// --- stageFromVram: vRAM -> SRAM fetch ------------------------------------------

{
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram });
  const rand = mulberry32(103);
  const w = randMatrix(rand, 8, 16);
  const alloc = await ram.alloc(w.byteLength);
  await ram.write(alloc.vaddr, encodeFromFloat32(w, "float32"));
  const staged = await npu.stageFromVram("layer0.k_proj", alloc.vaddr, 8, 16, "float32", "fp8");
  assert.equal(staged.resident, true);
  assert.equal(staged.format, "fp8");
  const a = randMatrix(rand, 2, 16);
  const r = await npu.matmulTransposeB(a, 2, 16, w, 8, 16, { weightKey: "layer0.k_proj" });
  assert.equal(r.sramHit, true);
  assert.equal(r.format, "fp8");
  const ref = matmulTransposeB(a, 2, 16, w, 8, 16);
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound);
  console.log("ok stageFromVram: fp32 weights read from vRAM, staged as fp8, matmul hits SRAM");
}

// --- spill: one weight larger than the whole SRAM --------------------------------

{
  // 1 KiB SRAM: a 64x64 int8 weight (4,368 bytes) can never be resident.
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram, sramBytes: 1024 });
  const rand = mulberry32(107);
  const w = randMatrix(rand, 64, 64);
  const staged = npu.stageWeights("huge", w, 64, 64, "int8");
  assert.equal(staged.resident, false, "oversize weight spills");
  assert.equal(staged.bytes, 64 * 64 + 64 * 4 + 16);
  assert.equal(npu.resident("huge"), false);
  assert.equal(npu.stats().sramSpills, 1);
  assert.equal(npu.sramStatus().usedBytes, 0, "a spill occupies no SRAM");

  const a = randMatrix(rand, 2, 64);
  const ref = matmulTransposeB(a, 2, 64, w, 64, 64);
  const r = await npu.matmulTransposeB(a, 2, 64, w, 64, 64, { weightKey: "huge" });
  assert.equal(r.sramHit, false, "spilled weight counts a miss");
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound, "miss path still correct (streams from vRAM)");
  assert.equal(npu.stats().sramMisses, 1);
  console.log("ok spill: oversize weight spills (counted), later use is a miss but still correct");
}

// --- eviction: staging past capacity evicts LRU -----------------------------------

{
  // 3,000-byte SRAM; each 32x32 per-channel int8 weight is 1,168 bytes.
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram, sramBytes: 3000 });
  const rand = mulberry32(109);
  const wA = randMatrix(rand, 32, 32);
  const wB = randMatrix(rand, 32, 32);
  const wC = randMatrix(rand, 32, 32);
  assert.equal(npu.stageWeights("A", wA, 32, 32, "int8").resident, true);
  assert.equal(npu.stageWeights("B", wB, 32, 32, "int8").resident, true);
  assert.equal(npu.sramStatus().usedBytes, 2 * 1168);
  // C does not fit alongside A+B: the LRU resident (A) is evicted.
  assert.equal(npu.stageWeights("C", wC, 32, 32, "int8").resident, true);
  assert.equal(npu.stats().sramEvictions, 1);
  assert.equal(npu.resident("A"), false, "LRU weight evicted");
  assert.equal(npu.resident("B"), true);
  assert.equal(npu.resident("C"), true);
  assert.equal(npu.sramStatus().usedBytes, 2 * 1168);

  // Using the evicted weight again is a miss, and still correct.
  const a = randMatrix(rand, 2, 32);
  const ref = matmulTransposeB(a, 2, 32, wA, 32, 32);
  const r = await npu.matmulTransposeB(a, 2, 32, wA, 32, 32, { weightKey: "A" });
  assert.equal(r.sramHit, false);
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound);
  assert.equal(npu.stats().sramMisses, 1);
  // Touching B makes it MRU; staging D then evicts C, not B.
  await npu.matmulTransposeB(a, 2, 32, wB, 32, 32, { weightKey: "B" });
  const wD = randMatrix(rand, 32, 32);
  npu.stageWeights("D", wD, 32, 32, "int8");
  assert.equal(npu.resident("C"), false, "second eviction takes the LRU (C), not the just-used B");
  assert.equal(npu.resident("B"), true);
  assert.equal(npu.stats().sramEvictions, 2);
  console.log("ok eviction: LRU eviction counted; evicted weights miss but stay correct");
}

// --- unknown keys and explicit eviction ----------------------------------------------

{
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram });
  const rand = mulberry32(113);
  const w = randMatrix(rand, 8, 16);
  const a = randMatrix(rand, 2, 16);
  await assert.rejects(
    () => npu.matmulTransposeB(a, 2, 16, w, 8, 16, { weightKey: "never-staged" }),
    /unknown staged weight/,
  );
  assert.equal(npu.evictWeights("never-staged"), false);
  npu.stageWeights("w", w, 8, 16, "int8");
  assert.equal(npu.resident("w"), true);
  assert.equal(npu.evictWeights("w"), true);
  assert.equal(npu.resident("w"), false);
  assert.equal(npu.evictWeights("w"), false);
  console.log("ok unknown keys throw; explicit eviction works");
}

// --- SW-code staging + matmulSw from SRAM ----------------------------------------------

{
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram });
  const rand = mulberry32(127);
  const w = randMatrix(rand, 8, 16);
  const staged = npu.stageWeights("layer0.down", w, 8, 16, "sw3");
  assert.equal(staged.resident, true);
  assert.equal(staged.format, "sw3");
  const packed = npu.quantizeSw(w, 8, 16, "sw3");
  const a = randMatrix(rand, 2, 16);
  const r = await npu.matmulSw(a, 2, 16, packed, { weightKey: "layer0.down" });
  assert.equal(r.sramHit, true);
  assert.equal(r.format, "sw3");
  const ref = matmulTransposeB(a, 2, 16, w, 8, 16);
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound);
  console.log("ok SW-code staging: matmulSw serves a staged sw3 weight from SRAM");
}

// --- node loss: resident SRAM dies with the donor ----------------------------------------

{
  const ram = new FakeRam();
  const nodes = [makeNode("node-a"), makeNode("node-b")];
  const provider: NodeProvider = { listNodes: () => nodes };
  const npu = new VirtualNpu({ nodes: provider, ram });
  const rand = mulberry32(131);
  const w = randMatrix(rand, 8, 16);
  // Least-used placement puts the first stage on node-a; pin it explicitly.
  const staged = npu.stageWeights("layer0.v_proj", w, 8, 16, "int8", { deviceId: "vnpu-node-a" });
  assert.equal(staged.deviceId, "vnpu-node-a");
  assert.equal(npu.listDevices().length, 2);

  const loss = await npu.handleNodeLoss("node-a");
  assert.deepEqual(loss, { devicesLost: 1, sramBytesLost: staged.bytes });
  assert.equal(npu.listDevices().length, 1);
  assert.equal(npu.listDevices()[0]?.id, "vnpu-node-b");
  assert.equal(npu.resident("layer0.v_proj"), false, "resident copy died with its donor");
  assert.equal(npu.sramStatus().usedBytes, 0);

  // The weight still lives in vRAM: the next use is a miss, not an error.
  const a = randMatrix(rand, 2, 16);
  const ref = matmulTransposeB(a, 2, 16, w, 8, 16);
  const r = await npu.matmulTransposeB(a, 2, 16, w, 8, 16, { weightKey: "layer0.v_proj" });
  assert.equal(r.sramHit, false);
  assert.ok(maxAbsErr(r.output, ref) <= r.errorBound);
  assert.equal(npu.stats().sramMisses, 1);
  // A second loss of the same node counts nothing again.
  const loss2 = await npu.handleNodeLoss("node-a");
  assert.deepEqual(loss2, { devicesLost: 0, sramBytesLost: 0 });
  // Staging again lands on the survivor.
  const restaged = npu.stageWeights("layer0.v_proj", w, 8, 16, "int8");
  assert.equal(restaged.deviceId, "vnpu-node-b");
  assert.equal(restaged.resident, true);
  console.log("ok node loss: device + resident SRAM lost (counted); later use misses, re-stage lands on survivor");
}

// --- D1 default capacity --------------------------------------------------------------------

{
  const ram = new FakeRam();
  const provider: NodeProvider = { listNodes: () => [makeNode("node-a")] };
  const npu = new VirtualNpu({ nodes: provider, ram });
  assert.equal(npu.sramStatus().capacityBytes, VNPU_SRAM_BYTES);
  assert.equal(npu.stats().sramCapacityBytes, VNPU_SRAM_BYTES);
  assert.equal(npu.listDevices()[0]?.sramBytes, VNPU_SRAM_BYTES);
  console.log("ok default capacity is D1 (256 MiB per device)");
}
