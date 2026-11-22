/**
 * Hypervisor + hw node server smoke test — run with:
 *   node test/hw-hypervisor.test.ts   (exit 0 on pass)
 *
 * Covers, with fake pool stubs and a fake NodeProvider (no real donors):
 *  - boot() waits for a healthy node, verifies pool stats, returns a
 *    VirtualMachine; getVmInfo() shows 2 nodes and the stub stats
 *  - stageToVram() streams an object from the fake vSSD into fake vRAM
 *    in VRAM_PAGE_SIZE_BYTES pieces (incl. a >1-page object)
 *  - sweep() heals a disappeared node and a 'dead' node exactly once,
 *    calling handleNodeLoss on all four pools; a throwing pool is
 *    isolated and recorded in the summary's errors
 *  - methods throw before boot(); boot() times out with no healthy node
 *  - the hw node server over real HTTP: POST JSON body, GET empty body,
 *    404 for an unknown path, 500 for a throwing handler, and
 *    mergeRouteTables() later-tables-win
 */
import assert from "node:assert/strict";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { createHwNodeServer, mergeRouteTables } from "../src/hw/node-server.ts";
import { VRAM_PAGE_SIZE_BYTES } from "../src/contracts.ts";
import type {
  CpuStats,
  CpuTask,
  CpuTaskHandler,
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
  WorkerDevice,
  WorkerInfo,
  WorkerState,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

// --- Fake pools ------------------------------------------------------------

class FakeRam implements IVirtualRam {
  buf = new Uint8Array(16 * 1024 * 1024);
  nextVaddr = 0;
  handleNodeLossCalls = 0;
  lastLostNode: string | null = null;

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    const vaddr = this.nextVaddr;
    this.nextVaddr += sizeBytes;
    return { id: `alloc-${vaddr}`, vaddr, sizeBytes, pages: [], replicas: 0 };
  }
  async free(_allocId: string): Promise<void> {}
  async read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.buf.slice(vaddr, vaddr + length);
  }
  async write(vaddr: number, data: Uint8Array): Promise<void> {
    this.buf.set(data, vaddr);
  }
  async replicate(_allocId: string, _targetNodeId: string): Promise<void> {}
  async handleNodeLoss(nodeId: string): Promise<{ pagesRecovered: number; pagesLost: number }> {
    this.handleNodeLossCalls += 1;
    this.lastLostNode = nodeId;
    return { pagesRecovered: 3, pagesLost: 1 };
  }
  stats(): VramStats {
    return { totalBytes: 1 << 30, usedBytes: 1 << 20, freeBytes: (1 << 30) - (1 << 20), pages: 256, nodes: 2 };
  }
}

class FakeGpu implements IVirtualGpu {
  handleNodeLossCalls = 0;
  throwOnLoss = false;

  listDevices(): VirtualGpuDevice[] {
    return [
      { id: "vgpu-0", computeUnits: 64, memoryBytes: 1 << 30, flopsPerSec: 1e12, backedBy: "cuda" },
    ];
  }
  createStream(): string {
    return "stream-1";
  }
  async launch(_launch: KernelLaunch): Promise<KernelResult> {
    return { streamId: "stream-1", kernelId: "k-1", device: "cuda", nodeId: "node-b", elapsedMs: 1 };
  }
  async synchronize(_streamId: string): Promise<void> {}
  async transfer(_vaddr: number, _targetNodeId: string): Promise<void> {}
  async handleNodeLoss(nodeId: string): Promise<{ streamsReassigned: number; devicesLost: number }> {
    this.handleNodeLossCalls += 1;
    if (this.throwOnLoss) {
      throw new Error(`gpu cannot heal ${nodeId}`);
    }
    return { streamsReassigned: 2, devicesLost: 1 };
  }
  stats(): GpuStats {
    return {
      devices: [
        { nodeId: "node-a", device: "cpu", flopsPerSec: 1e11 },
        { nodeId: "node-b", device: "cuda", flopsPerSec: 1e12 },
      ],
      totalFlopsPerSec: 1.1e12,
      kernelsCompleted: 7,
      kernelsFailed: 0,
    };
  }
}

class FakeCpu implements IVirtualCpu {
  handleNodeLossCalls = 0;
  handlers = new Map<string, CpuTaskHandler>();

  registerHandler(kind: string, handler: CpuTaskHandler): void {
    this.handlers.set(kind, handler);
  }
  async submit(task: CpuTask): Promise<CpuTaskResult> {
    return { taskId: task.id, ok: true, result: null, nodeId: "node-a", elapsedMs: 0 };
  }
  async handleNodeLoss(_nodeId: string): Promise<{ tasksRequeued: number; tasksLost: number }> {
    this.handleNodeLossCalls += 1;
    return { tasksRequeued: 5, tasksLost: 0 };
  }
  stats(): CpuStats {
    return { nodes: 2, totalCores: 16, queued: 0, completed: 42, failed: 0 };
  }
}

class FakeSsd implements IVirtualSsd {
  handleNodeLossCalls = 0;
  objects = new Map<string, Uint8Array>();

  async putObject(key: string, data: Uint8Array): Promise<SsdObjectMeta> {
    this.objects.set(key, data);
    return { key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: Date.now() };
  }
  async getObject(key: string): Promise<Uint8Array> {
    const data = this.objects.get(key);
    if (!data) throw new Error(`no such object: ${key}`);
    return data;
  }
  async getRange(key: string, offset: number, length: number): Promise<Uint8Array> {
    const data = await this.getObject(key);
    return data.slice(offset, offset + length);
  }
  async list(prefix?: string): Promise<SsdObjectMeta[]> {
    const out: SsdObjectMeta[] = [];
    for (const [key, data] of this.objects) {
      if (prefix === undefined || key.startsWith(prefix)) {
        out.push({ key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: 0 });
      }
    }
    return out;
  }
  async deleteObject(key: string): Promise<void> {
    this.objects.delete(key);
  }
  async handleNodeLoss(_nodeId: string): Promise<{ chunksRecovered: number; chunksLost: number }> {
    this.handleNodeLossCalls += 1;
    return { chunksRecovered: 4, chunksLost: 0 };
  }
  stats(): SsdStats {
    let totalBytes = 0;
    for (const data of this.objects.values()) totalBytes += data.length;
    return { objects: this.objects.size, totalBytes, nodes: 2 };
  }
}

// --- Fake NodeProvider -----------------------------------------------------

function makeNode(
  workerId: string,
  device: WorkerDevice,
  state: WorkerState,
): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device,
    backend: "fake",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 256,
    state,
    registeredAt: 0,
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeNodeProvider implements NodeProvider {
  nodes: WorkerInfo[];
  constructor(nodes: WorkerInfo[]) {
    this.nodes = nodes;
  }
  listNodes(): WorkerInfo[] {
    return [...this.nodes];
  }
}

function assertAllHealed(
  ram: FakeRam,
  gpu: FakeGpu,
  cpu: FakeCpu,
  ssd: FakeSsd,
  expected: number,
): void {
  assert.equal(ram.handleNodeLossCalls, expected, "ram handleNodeLoss calls");
  assert.equal(gpu.handleNodeLossCalls, expected, "gpu handleNodeLoss calls");
  assert.equal(cpu.handleNodeLossCalls, expected, "cpu handleNodeLoss calls");
  assert.equal(ssd.handleNodeLossCalls, expected, "ssd handleNodeLoss calls");
}

async function main(): Promise<void> {
  // --- methods throw before boot; boot times out with no healthy node ---
  const emptyProvider = new FakeNodeProvider([]);
  const unbooted = new Hypervisor({
    nodes: emptyProvider,
    ram: new FakeRam(),
    gpu: new FakeGpu(),
    cpu: new FakeCpu(),
    ssd: new FakeSsd(),
    bootTimeoutMs: 600,
  });
  assert.equal(unbooted.booted, false);
  assert.throws(() => unbooted.getVmInfo(), /not booted/);
  await assert.rejects(() => unbooted.boot(), /no healthy nodes/);
  await assert.rejects(() => unbooted.stageToVram("x"), /not booted/);

  // --- boot with 2 healthy nodes -----------------------------------------
  const nodeA = makeNode("node-a", "cpu", "healthy");
  const nodeB = makeNode("node-b", "cuda", "healthy");
  const provider = new FakeNodeProvider([nodeA, nodeB]);
  const ram = new FakeRam();
  const gpu = new FakeGpu();
  const cpu = new FakeCpu();
  const ssd = new FakeSsd();
  const hypervisor = new Hypervisor({
    nodes: provider,
    ram,
    gpu,
    cpu,
    ssd,
    sweepIntervalMs: 60_000, // keep the background sweep out of the way
    bootTimeoutMs: 2000,
  });

  const vm = await hypervisor.boot();
  assert.equal(hypervisor.booted, true);
  assert.equal(vm.ram, ram);
  assert.equal(vm.gpu, gpu);
  assert.equal(vm.cpu, cpu);
  assert.equal(vm.ssd, ssd);

  const info = hypervisor.getVmInfo();
  assert.equal(info.nodes.length, 2);
  assert.deepEqual(info.nodes, [
    { nodeId: "node-a", device: "cpu", state: "healthy" },
    { nodeId: "node-b", device: "cuda", state: "healthy" },
  ]);
  assert.equal(info.vram.totalBytes, 1 << 30);
  assert.equal(info.vram.pages, 256);
  assert.equal(info.gpu.totalFlopsPerSec, 1.1e12);
  assert.equal(info.gpu.kernelsCompleted, 7);
  assert.equal(info.cpu.totalCores, 16);
  assert.equal(info.cpu.completed, 42);
  assert.equal(info.ssd.objects, 0);
  // info() on the returned VM is the same view.
  assert.deepEqual(vm.info(), info);

  // --- stageToVram: small object ------------------------------------------
  const pattern = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  await ssd.putObject("weights/blob", pattern);
  const alloc = await hypervisor.stageToVram("weights/blob");
  assert.equal(alloc.sizeBytes, pattern.length);
  assert.deepEqual(await ram.read(alloc.vaddr, pattern.length), pattern);
  await assert.rejects(
    () => hypervisor.stageToVram("weights/missing"),
    /not found in vSSD/,
  );

  // --- stageToVram: object spanning more than one VRAM page ---------------
  const bigSize = VRAM_PAGE_SIZE_BYTES + 7;
  const big = new Uint8Array(bigSize);
  for (let i = 0; i < bigSize; i++) big[i] = i % 251;
  await ssd.putObject("weights/big", big);
  const bigAlloc = await hypervisor.stageToVram("weights/big");
  assert.equal(bigAlloc.sizeBytes, bigSize);
  assert.deepEqual(await ram.read(bigAlloc.vaddr, bigSize), big);

  // --- sweep: node disappears ---------------------------------------------
  provider.nodes = [nodeA];
  await hypervisor.sweep();
  assertAllHealed(ram, gpu, cpu, ssd, 1);
  assert.equal(ram.lastLostNode, "node-b");
  // A second sweep does not re-heal the same disappearance.
  await hypervisor.sweep();
  assertAllHealed(ram, gpu, cpu, ssd, 1);

  // --- sweep: node goes 'dead' while still listed -------------------------
  provider.nodes = [makeNode("node-a", "cpu", "dead")];
  await hypervisor.sweep();
  assertAllHealed(ram, gpu, cpu, ssd, 2);
  // Still listed as dead: not healed again.
  await hypervisor.sweep();
  assertAllHealed(ram, gpu, cpu, ssd, 2);

  // --- handleNodeLoss summary shape ---------------------------------------
  const summary = await hypervisor.handleNodeLoss("node-a");
  assert.deepEqual(summary, {
    ram: { pagesRecovered: 3, pagesLost: 1 },
    gpu: { streamsReassigned: 2, devicesLost: 1 },
    cpu: { tasksRequeued: 5, tasksLost: 0 },
    ssd: { chunksRecovered: 4, chunksLost: 0 },
    errors: [],
  });
  assertAllHealed(ram, gpu, cpu, ssd, 3);

  // --- a throwing pool is isolated and recorded ---------------------------
  const badGpu = new FakeGpu();
  badGpu.throwOnLoss = true;
  const ram2 = new FakeRam();
  const cpu2 = new FakeCpu();
  const ssd2 = new FakeSsd();
  const hypervisor2 = new Hypervisor({
    nodes: new FakeNodeProvider([makeNode("node-a", "cpu", "healthy")]),
    ram: ram2,
    gpu: badGpu,
    cpu: cpu2,
    ssd: ssd2,
    sweepIntervalMs: 60_000,
    bootTimeoutMs: 2000,
  });
  await hypervisor2.boot();
  const badSummary = await hypervisor2.handleNodeLoss("node-a");
  assert.equal(badSummary.errors.length, 1);
  assert.match(badSummary.errors[0], /^gpu: gpu cannot heal node-a$/);
  assert.deepEqual(badSummary.gpu, { streamsReassigned: 0, devicesLost: 0 });
  assert.deepEqual(badSummary.ram, { pagesRecovered: 3, pagesLost: 1 });
  assert.equal(ram2.handleNodeLossCalls, 1);
  assert.equal(cpu2.handleNodeLossCalls, 1);
  assert.equal(ssd2.handleNodeLossCalls, 1);
  await hypervisor2.shutdown();
  assert.equal(hypervisor2.booted, false);

  await hypervisor.shutdown();
  assert.equal(hypervisor.booted, false);
  assert.throws(() => hypervisor.getVmInfo(), /not booted/);

  // --- hw node server over real HTTP --------------------------------------
  const seenBodies: Record<string, unknown>[] = [];
  const server = await createHwNodeServer({
    port: 0,
    routeTables: [
      {
        "/hw/ping": async (body) => {
          seenBodies.push(body);
          return { pong: true };
        },
        "/hw/vram/alloc": async (body) => {
          seenBodies.push(body);
          return { ok: true, sizeBytes: body["sizeBytes"] };
        },
      },
      // Later table wins on collision.
      {
        "/hw/vram/alloc": async () => ({ ok: true, overridden: true }),
        "/hw/boom": async () => {
          throw new Error("kaboom");
        },
      },
    ],
  });
  assert.ok(server.port > 0, "ephemeral port assigned");
  try {
    const base = `http://127.0.0.1:${server.port}`;

    const post = await fetch(`${base}/hw/vram/alloc`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sizeBytes: 123 }),
    });
    assert.equal(post.status, 200);
    assert.deepEqual(await post.json(), { ok: true, overridden: true });

    const get = await fetch(`${base}/hw/ping`);
    assert.equal(get.status, 200);
    assert.deepEqual(await get.json(), { pong: true });
    // GET routes receive an empty body.
    assert.deepEqual(seenBodies[seenBodies.length - 1], {});

    const boom = await fetch(`${base}/hw/boom`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    assert.equal(boom.status, 500);
    assert.deepEqual(await boom.json(), { error: "kaboom" });

    const missing = await fetch(`${base}/hw/nope`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    assert.equal(missing.status, 404);
    assert.deepEqual(await missing.json(), { error: "unknown route: /hw/nope" });
  } finally {
    await server.close();
  }

  // --- mergeRouteTables: later tables win ---------------------------------
  const merged = mergeRouteTables(
    { "/x": async () => "first" },
    { "/x": async () => "second", "/y": async () => "y" },
  );
  assert.equal(await merged["/x"]({}), "second");
  assert.equal(await merged["/y"]({}), "y");
}

try {
  await main();
  console.log("hw-hypervisor smoke test: PASS");
} catch (err) {
  console.error("hw-hypervisor smoke test: FAIL", err);
  process.exit(1);
}
