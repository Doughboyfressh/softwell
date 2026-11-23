/**
 * vNPU hypervisor test — run with: node test/vnpu-hypervisor.test.ts (exit 0 on pass).
 *
 * Covers the vNPU's registration with the hypervisor (docs/VNPU.md §3.1):
 *  - boot() with a VirtualNpu pool verifies its stats and returns a
 *    VirtualMachine whose .npu is the pool; getVmInfo().npu carries the
 *    pool stats (devices, SRAM capacity) alongside the other devices
 *  - handleNodeLoss() fans out to the vNPU: the lost device and its
 *    resident SRAM bytes appear in the summary's npu field
 *  - a hypervisor booted WITHOUT a vNPU omits the npu field from both
 *    getVmInfo() and the loss summary (existing summaries compare equal)
 *  - sweep() heals a disappeared node through the vNPU pool
 *
 * The other pools are minimal stubs; the vNPU itself is real.
 */
import assert from "node:assert/strict";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { VirtualNpu, VNPU_SRAM_BYTES } from "../src/hw/vnpu.ts";
import type {
  CpuStats,
  CpuTask,
  CpuTaskResult,
  GpuStats,
  IVirtualCpu,
  IVirtualGpu,
  IVirtualRam,
  IVirtualSsd,
  KernelLaunch,
  KernelResult,
  NodeProvider,
  SsdObjectMeta,
  SsdStats,
  VirtualGpuDevice,
  VramAllocation,
  VramStats,
  WorkerInfo,
} from "../src/contracts.ts";

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
  async alloc(sizeBytes: number): Promise<VramAllocation> {
    return { id: "alloc-0", vaddr: 0, sizeBytes, pages: [], replicas: 1 };
  }
  async free(): Promise<void> {}
  async read(): Promise<Uint8Array> {
    return new Uint8Array(0);
  }
  async write(): Promise<void> {}
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: 1 << 30, usedBytes: 0, freeBytes: 1 << 30, pages: 256, nodes: 1 };
  }
}

class FakeGpu implements IVirtualGpu {
  listDevices(): VirtualGpuDevice[] {
    return [];
  }
  createStream(): string {
    return "stream-0";
  }
  async launch(_launch: KernelLaunch): Promise<KernelResult> {
    throw new Error("FakeGpu.launch not used here");
  }
  async synchronize(): Promise<void> {}
  async transfer(): Promise<void> {}
  async handleNodeLoss(): Promise<{ streamsReassigned: number; devicesLost: number }> {
    return { streamsReassigned: 0, devicesLost: 0 };
  }
  stats(): GpuStats {
    return { devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 };
  }
}

class FakeCpu implements IVirtualCpu {
  registerHandler(): void {}
  async submit(task: CpuTask): Promise<CpuTaskResult> {
    return { taskId: task.id, ok: true, nodeId: "node-a", elapsedMs: 0 };
  }
  async handleNodeLoss(): Promise<{ tasksRequeued: number; tasksLost: number }> {
    return { tasksRequeued: 0, tasksLost: 0 };
  }
  stats(): CpuStats {
    return { nodes: 1, totalCores: 1, queued: 0, completed: 0, failed: 0 };
  }
}

class FakeSsd implements IVirtualSsd {
  async putObject(key: string, data: Uint8Array): Promise<SsdObjectMeta> {
    return { key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: 0 };
  }
  async getObject(): Promise<Uint8Array> {
    throw new Error("not found");
  }
  async getRange(): Promise<Uint8Array> {
    throw new Error("not found");
  }
  async list(): Promise<SsdObjectMeta[]> {
    return [];
  }
  async deleteObject(): Promise<void> {}
  async handleNodeLoss(): Promise<{ chunksRecovered: number; chunksLost: number }> {
    return { chunksRecovered: 0, chunksLost: 0 };
  }
  stats(): SsdStats {
    return { objects: 0, totalBytes: 0, nodes: 1 };
  }
}

function makeHypervisor(nodes: WorkerInfo[], npu?: VirtualNpu): Hypervisor {
  const provider: NodeProvider = { listNodes: () => nodes };
  const ram = new FakeRam();
  const gpu = new FakeGpu();
  const cpu = new FakeCpu();
  const ssd = new FakeSsd();
  return new Hypervisor({
    nodes: provider,
    ram,
    gpu,
    cpu,
    ssd,
    ...(npu !== undefined ? { npu } : {}),
    sweepIntervalMs: 60_000,
    bootTimeoutMs: 2000,
  });
}

// --- boot with a vNPU: vm.npu + getVmInfo().npu --------------------------------

{
  const nodes = [makeNode("node-a"), makeNode("node-b")];
  const provider: NodeProvider = { listNodes: () => nodes };
  const npu = new VirtualNpu({ nodes: provider, ram: new FakeRam() });
  const hypervisor = makeHypervisor(nodes, npu);
  const vm = await hypervisor.boot();
  assert.equal(vm.npu, npu, "vm.npu is the registered pool");
  const info = hypervisor.getVmInfo();
  assert.ok(info.npu !== undefined, "getVmInfo carries npu stats");
  assert.equal(info.npu?.sramCapacityBytes, VNPU_SRAM_BYTES);
  assert.equal(info.npu?.devices.length, 2);
  assert.equal(info.npu?.devices[0]?.nodeId, "node-a");
  assert.equal(info.npu?.devices[0]?.sramBytes, VNPU_SRAM_BYTES);
  // info() on the returned VM is the same view.
  assert.deepEqual(vm.info(), info);
  console.log("ok hypervisor boot: vm.npu registered, getVmInfo().npu present with 2 devices");

  // Stage a weight, then lose its donor through the hypervisor.
  const w = new Float32Array(8 * 16).fill(0.5);
  const staged = npu.stageWeights("layer0.q", w, 8, 16, "int8", { deviceId: "vnpu-node-a" });
  const summary = await hypervisor.handleNodeLoss("node-a");
  assert.deepEqual(summary.npu, { devicesLost: 1, sramBytesLost: staged.bytes });
  assert.deepEqual(summary.errors, []);
  assert.equal(hypervisor.getVmInfo().npu?.devices.length, 1, "lost device leaves the pool");
  console.log("ok handleNodeLoss fans out to the vNPU (device + resident SRAM bytes lost)");

  // A second loss of the same node counts nothing again.
  const summary2 = await hypervisor.handleNodeLoss("node-a");
  assert.deepEqual(summary2.npu, { devicesLost: 0, sramBytesLost: 0 });
  await hypervisor.shutdown();
}

// --- sweep heals a disappeared node through the vNPU -----------------------------

{
  const nodes = [makeNode("node-a"), makeNode("node-b")];
  const providerNodes = [...nodes];
  const provider: NodeProvider = { listNodes: () => providerNodes };
  const npu = new VirtualNpu({ nodes: provider, ram: new FakeRam() });
  const hypervisor = makeHypervisor(providerNodes, npu);
  await hypervisor.boot();
  providerNodes.splice(1, 1); // node-b disappears
  await hypervisor.sweep();
  assert.equal(hypervisor.getVmInfo().npu?.devices.length, 1);
  assert.equal(npu.listDevices().length, 1);
  await hypervisor.shutdown();
  console.log("ok sweep heals a disappeared node through the vNPU pool");
}

// --- no vNPU: the fields are omitted, not zeroed ----------------------------------

{
  const nodes = [makeNode("node-a")];
  const hypervisor = makeHypervisor(nodes);
  const vm = await hypervisor.boot();
  assert.equal(vm.npu, undefined, "no npu pool registered");
  const info = hypervisor.getVmInfo();
  assert.equal("npu" in info, false, "getVmInfo omits npu when no pool is registered");
  const summary = await hypervisor.handleNodeLoss("node-a");
  assert.equal("npu" in summary, false, "loss summary omits npu when no pool is registered");
  await hypervisor.shutdown();
  console.log("ok hypervisor without a vNPU omits the npu fields (existing summaries compare equal)");
}

// --- a throwing vNPU is isolated like any other pool --------------------------------

{
  const nodes = [makeNode("node-a")];
  const provider: NodeProvider = { listNodes: () => nodes };
  const npu = new VirtualNpu({ nodes: provider, ram: new FakeRam() });
  // A shallow copy that still reports stats/listDevices from the real
  // pool but throws from handleNodeLoss.
  const badNpu = Object.create(npu) as VirtualNpu;
  badNpu.handleNodeLoss = async () => {
    throw new Error("npu cannot heal node-a");
  };
  const hypervisor = makeHypervisor(nodes, badNpu);
  await hypervisor.boot();
  const summary = await hypervisor.handleNodeLoss("node-a");
  assert.equal(summary.errors.length, 1);
  assert.match(summary.errors[0] as string, /^npu: npu cannot heal node-a$/);
  assert.deepEqual(summary.npu, { devicesLost: 0, sramBytesLost: 0 });
  await hypervisor.shutdown();
  console.log("ok a throwing vNPU is isolated and recorded, like any other pool");
}
