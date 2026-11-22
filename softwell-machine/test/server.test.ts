/**
 * Tests for the virtual server (src/server/server.ts).
 *
 * Small in-file fakes only: a fake VM (RAM/GPU/CPU/SSD), a fake model
 * registry, a fake serving-context factory, and a fake scheduler. The
 * real registry (src/server/registry.ts) is NOT imported.
 *
 * Run: node --test test/server.test.ts
 */

import assert from "node:assert/strict";
import test from "node:test";
import type {
  ClusterSnapshot,
  CpuStats,
  CpuTask,
  CpuTaskHandler,
  CpuTaskResult,
  EngineOptions,
  GeneratedToken,
  GenerateRequest,
  GpuStats,
  IVirtualCpu,
  IVirtualGpu,
  IVirtualRam,
  IVirtualSsd,
  KernelLaunch,
  KernelResult,
  ModelEntry,
  ModelManifest,
  Scheduler,
  SequenceState,
  ServerRequestRecord,
  ServingContext,
  SsdObjectMeta,
  SsdStats,
  Tokenizer,
  VirtualGpuDevice,
  VirtualMachine,
  VirtualMachineInfo,
  VramAllocation,
  VramStats,
  WorkerDevice,
} from "../src/contracts.ts";
import {
  DEFAULT_SERVER_MEMORY_BYTES,
  VirtualServer,
} from "../src/server/server.ts";
import type {
  ServingContextFactory,
  VirtualServerDeps,
} from "../src/server/server.ts";

// ---------------------------------------------------------------------------
// Fakes: virtual hardware
// ---------------------------------------------------------------------------

class FakeRam implements IVirtualRam {
  allocs: VramAllocation[] = [];
  freed: string[] = [];
  /** When set, alloc() throws on exactly this (1-based) call. */
  throwOnAllocCall: number | null = null;
  private allocCalls = 0;
  private nextVaddr = 0;

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    this.allocCalls += 1;
    if (this.throwOnAllocCall === this.allocCalls) {
      throw new Error(`fake RAM out of memory (alloc call ${this.allocCalls})`);
    }
    const alloc: VramAllocation = {
      id: `alloc-${this.allocCalls}`,
      vaddr: this.nextVaddr,
      sizeBytes,
      pages: [],
      replicas: 0,
    };
    this.nextVaddr += sizeBytes;
    this.allocs.push(alloc);
    return alloc;
  }

  async free(allocId: string): Promise<void> {
    this.freed.push(allocId);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    return new Uint8Array(length).fill(0).slice(vaddr, vaddr);
  }

  async write(_vaddr: number, _data: Uint8Array): Promise<void> {}

  async replicate(_allocId: string, _targetNodeId: string): Promise<void> {}

  async handleNodeLoss(_nodeId: string): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): VramStats {
    return { totalBytes: 1 << 30, usedBytes: 0, freeBytes: 1 << 30, pages: 256, nodes: 1 };
  }
}

class FakeGpu implements IVirtualGpu {
  streams: string[] = [];
  throwOnCreateStream = false;

  listDevices(): VirtualGpuDevice[] {
    return [
      { id: "vgpu-0", computeUnits: 64, memoryBytes: 1 << 30, flopsPerSec: 1e12, backedBy: "cpu" },
    ];
  }

  createStream(): string {
    if (this.throwOnCreateStream) {
      throw new Error("fake GPU cannot create a stream");
    }
    const id = `stream-${this.streams.length + 1}`;
    this.streams.push(id);
    return id;
  }

  async launch(_launch: KernelLaunch): Promise<KernelResult> {
    return { streamId: "stream-1", kernelId: "k-1", device: "cpu", nodeId: "node-1", elapsedMs: 1 };
  }

  async synchronize(_streamId: string): Promise<void> {}

  async transfer(_vaddr: number, _targetNodeId: string): Promise<void> {}

  async handleNodeLoss(_nodeId: string): Promise<{ streamsReassigned: number; devicesLost: number }> {
    return { streamsReassigned: 0, devicesLost: 0 };
  }

  stats(): GpuStats {
    return { devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 };
  }
}

class FakeCpu implements IVirtualCpu {
  handlers = new Map<string, CpuTaskHandler>();

  registerHandler(kind: string, handler: CpuTaskHandler): void {
    this.handlers.set(kind, handler);
  }

  async submit(task: CpuTask): Promise<CpuTaskResult> {
    const started = Date.now();
    const handler = this.handlers.get(task.kind);
    if (handler === undefined) {
      return { taskId: task.id, ok: false, error: `unknown kind ${task.kind}`, nodeId: "local", elapsedMs: 0 };
    }
    try {
      const result = await handler(task);
      return { taskId: task.id, ok: true, result, nodeId: "local", elapsedMs: Date.now() - started };
    } catch (err) {
      return {
        taskId: task.id,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
        nodeId: "local",
        elapsedMs: Date.now() - started,
      };
    }
  }

  /** Run a task through the registered handler; throws on handler error. */
  async run(kind: string, payload: unknown): Promise<unknown> {
    const result = await this.submit({ id: `task-${kind}`, kind, payload });
    if (!result.ok) {
      throw new Error(result.error ?? `cpu task ${kind} failed`);
    }
    return result.result;
  }

  async handleNodeLoss(_nodeId: string): Promise<{ tasksRequeued: number; tasksLost: number }> {
    return { tasksRequeued: 0, tasksLost: 0 };
  }

  stats(): CpuStats {
    return { nodes: 1, totalCores: 4, queued: 0, completed: 0, failed: 0 };
  }
}

class FakeSsd implements IVirtualSsd {
  async putObject(key: string, data: Uint8Array): Promise<SsdObjectMeta> {
    return { key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: Date.now() };
  }

  async getObject(_key: string): Promise<Uint8Array> {
    return new Uint8Array(0);
  }

  async getRange(_key: string, _offset: number, length: number): Promise<Uint8Array> {
    return new Uint8Array(length);
  }

  async list(_prefix?: string): Promise<SsdObjectMeta[]> {
    return [];
  }

  async deleteObject(_key: string): Promise<void> {}

  async handleNodeLoss(_nodeId: string): Promise<{ chunksRecovered: number; chunksLost: number }> {
    return { chunksRecovered: 0, chunksLost: 0 };
  }

  stats(): SsdStats {
    return { objects: 0, totalBytes: 0, nodes: 1 };
  }
}

class FakeVm implements VirtualMachine {
  ram = new FakeRam();
  gpu = new FakeGpu();
  cpu = new FakeCpu();
  ssd = new FakeSsd();
  /** When true, info() throws (simulates a VM that stopped responding). */
  infoThrows = false;

  info(): VirtualMachineInfo {
    if (this.infoThrows) {
      throw new Error("fake VM info failed");
    }
    return {
      vram: this.ram.stats(),
      gpu: this.gpu.stats(),
      cpu: this.cpu.stats(),
      ssd: this.ssd.stats(),
      nodes: [{ nodeId: "node-1", device: "cpu" as WorkerDevice, state: "healthy" }],
    };
  }
}

// ---------------------------------------------------------------------------
// Fakes: scheduler / tokenizer / manifest / registry / factory
// ---------------------------------------------------------------------------

const fakeTokenizer: Tokenizer = {
  vocabSize: 256,
  bosTokenId: 0,
  eosTokenId: 1,
  encode(text: string): number[] {
    return [...Buffer.from(text, "utf8")];
  },
  decode(tokenIds: number[]): string {
    return Buffer.from(tokenIds).toString("utf8");
  },
};

class FakeScheduler implements Scheduler {
  pending = 0;
  running = 0;

  submit(_req: GenerateRequest): string {
    return "seq-1";
  }

  async step(): Promise<GeneratedToken[]> {
    return [];
  }

  getSequence(_id: string): SequenceState | undefined {
    return undefined;
  }

  cancel(_id: string): boolean {
    return false;
  }

  pendingCount(): number {
    return this.pending;
  }

  runningCount(): number {
    return this.running;
  }
}

function makeManifest(modelId: string): ModelManifest {
  return {
    format: "safetensors",
    modelId,
    config: {
      architectures: ["LlamaForCausalLM"],
      vocabSize: 256,
      hiddenSize: 64,
      intermediateSize: 128,
      numHiddenLayers: 2,
      numAttentionHeads: 4,
      numKeyValueHeads: 2,
      headDim: 16,
      maxPositionEmbeddings: 1024,
      rmsNormEps: 1e-5,
      ropeTheta: 10000,
      tieWordEmbeddings: true,
      bosTokenId: 0,
      eosTokenId: 1,
      padTokenId: 2,
    },
    metadata: {},
    tensors: {},
    files: [],
    totalBytes: 1024,
    totalParams: 256,
  };
}

function makeEntry(id: string, status: ModelEntry["status"]): ModelEntry {
  return {
    id,
    source: { kind: "local", dir: `/models/${id}` },
    dir: `/models/${id}`,
    status,
    vramBytes: 0,
    loadedAt: 1_700_000_000_000,
    requestsServed: 0,
    tokensGenerated: 0,
  };
}

class FakeRegistry {
  entries = new Map<string, ModelEntry>();
  /** Model id the registry currently treats as active (for unload refusal). */
  activeId: string | null = null;
  unloaded: string[] = [];

  listAvailable(): Promise<ModelEntry[]> {
    return Promise.resolve([...this.entries.values()]);
  }

  list(): ModelEntry[] {
    return [...this.entries.values()];
  }

  get(id: string): ModelEntry | undefined {
    return this.entries.get(id);
  }

  async load(
    ref: string,
    _opts?: { engineOptions?: EngineOptions },
  ): Promise<ModelEntry> {
    const entry = this.entries.get(ref);
    if (entry === undefined) {
      throw new Error(`unknown model: ${ref}`);
    }
    entry.status = "ready";
    entry.loadedAt = Date.now();
    return entry;
  }

  async unload(id: string): Promise<void> {
    if (id === this.activeId) {
      throw new Error(`cannot unload the active model: ${id}`);
    }
    const entry = this.entries.get(id);
    if (entry === undefined) {
      throw new Error(`unknown model: ${id}`);
    }
    entry.status = "available";
    entry.vramBytes = 0;
    this.unloaded.push(id);
  }
}

class FakeFactory implements ServingContextFactory {
  built: string[] = [];
  disposed: ServingContext[] = [];
  /** When set, build() throws for this model id. */
  throwOnBuildId: string | null = null;
  /** Last scheduler built, so tests can drive queue counts. */
  lastScheduler: FakeScheduler | null = null;

  async build(entry: ModelEntry): Promise<ServingContext> {
    if (this.throwOnBuildId === entry.id) {
      throw new Error(`fake factory cannot build ${entry.id}`);
    }
    this.built.push(entry.id);
    const scheduler = new FakeScheduler();
    this.lastScheduler = scheduler;
    return {
      modelId: entry.id,
      scheduler,
      tokenizer: fakeTokenizer,
      manifest: makeManifest(entry.id),
    };
  }

  async dispose(ctx: ServingContext): Promise<void> {
    this.disposed.push(ctx);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function healthyCluster(modelId: string | null): ClusterSnapshot {
  return {
    modelId,
    plan: null,
    workers: [
      {
        workerId: "worker-1",
        host: "127.0.0.1",
        port: 9001,
        device: "cpu",
        backend: "softwell-ts",
        memoryBytes: 8 * 1024 ** 3,
        freeMemoryBytes: 6 * 1024 ** 3,
        flopsPerSec: 1e12,
        bandwidthBytesPerSec: 1e9,
        roles: ["mixed"],
        maxBatchTokens: 512,
        state: "healthy",
        registeredAt: Date.now(),
        lastHeartbeatAt: Date.now(),
        usedMemoryBytes: 0,
        activeSequences: 0,
      },
    ],
    sequences: { waiting: 0, running: 0, done: 0 },
    tokensPerSec: 0,
    avgLatencyMs: 0,
    uptimeSec: 0,
  };
}

function makeRecord(
  overrides: Partial<ServerRequestRecord> & { id: string },
): ServerRequestRecord {
  return {
    modelId: "model-a",
    endpoint: "/v1/chat/completions",
    startedAt: 1_700_000_000_000,
    promptTokens: 4,
    completionTokens: 0,
    status: "done",
    ...overrides,
  };
}

interface Harness {
  vm: FakeVm;
  registry: FakeRegistry;
  factory: FakeFactory;
  server: VirtualServer;
}

function makeHarness(
  opts: {
    serverOptions?: ConstructorParameters<typeof VirtualServer>[0]["options"];
    getCluster?: () => ClusterSnapshot;
    onShutdown?: () => void | Promise<void>;
  } = {},
): Harness {
  const registry = new FakeRegistry();
  registry.entries.set("model-a", makeEntry("model-a", "ready"));
  registry.entries.set("model-b", makeEntry("model-b", "ready"));
  registry.entries.set("model-c", makeEntry("model-c", "available"));
  const factory = new FakeFactory();
  // FakeRegistry implements the registry methods VirtualServer uses
  // (list/get/load/unload) but not the real ModelRegistry class's
  // private fields, so the deps are cast — the fake is never the real
  // registry, and no real registry is imported here.
  const server = new VirtualServer({
    registry,
    factory,
    options: opts.serverOptions,
    getCluster: opts.getCluster,
    onShutdown: opts.onShutdown,
  } as unknown as VirtualServerDeps);
  return { vm: new FakeVm(), registry, factory, server };
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

test("boot reserves vRAM and creates a stream; second boot throws", async (t) => {
  const { vm, server } = makeHarness();
  t.after(() => server.close());

  await server.boot(vm);

  assert.equal(vm.ram.allocs.length, 1);
  assert.equal(vm.ram.allocs[0]?.sizeBytes, DEFAULT_SERVER_MEMORY_BYTES);
  assert.equal(vm.gpu.streams.length, 1);
  assert.notEqual(server.getApiPort(), null);
  assert.equal(server.getServerStatus().state, "ready");
  assert.equal(server.getServerStatus().vm !== null, true);

  // The API server is actually listening (dashboard route).
  const port = server.getApiPort() as number;
  const res = await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /softwell/i);

  // The API server serves through this VirtualServer as its resolver:
  // /v1/models lists the registry, /v1/server/status and /healthz
  // reflect the live server state.
  await server.activateModel("model-a");
  const modelsRes = await fetch(`http://127.0.0.1:${port}/v1/models`);
  const modelsBody = (await modelsRes.json()) as { data: { id: string }[] };
  assert.equal(modelsRes.status, 200);
  assert.deepEqual(
    modelsBody.data.map((m) => m.id),
    ["model-a", "model-b", "model-c"],
  );
  const statusRes = await fetch(`http://127.0.0.1:${port}/v1/server/status`);
  const statusBody = (await statusRes.json()) as { activeModelId: string | null; state: string };
  assert.equal(statusRes.status, 200);
  assert.equal(statusBody.activeModelId, "model-a");
  assert.equal(statusBody.state, "ready");
  const healthRes = await fetch(`http://127.0.0.1:${port}/healthz`);
  const healthBody = (await healthRes.json()) as { status: string };
  assert.equal(healthRes.status, 200);
  // No getCluster was injected, so no healthy worker: degraded, not ok.
  assert.equal(healthBody.status, "degraded");

  await assert.rejects(() => server.boot(vm), /already booted/);
});

test("boot reserves the configured serverMemoryBytes", async (t) => {
  const { vm, server } = makeHarness({
    serverOptions: { serverMemoryBytes: 64 * 1024 * 1024 },
  });
  t.after(() => server.close());

  await server.boot(vm);

  assert.equal(vm.ram.allocs.length, 1);
  assert.equal(vm.ram.allocs[0]?.sizeBytes, 64 * 1024 * 1024);
});

test("boot failure frees partial reservations", async (t) => {
  const { vm, server } = makeHarness();
  t.after(() => server.close());

  // First attempt: the RAM alloc succeeds, then the GPU stream fails,
  // so boot must free the reservation it just made.
  vm.gpu.throwOnCreateStream = true;
  // Second attempt: the RAM alloc itself throws (second alloc call).
  vm.ram.throwOnAllocCall = 2;

  await assert.rejects(() => server.boot(vm), /cannot create a stream/);
  assert.equal(vm.ram.allocs.length, 1);
  assert.deepEqual(vm.ram.freed, [vm.ram.allocs[0]?.id]);
  assert.equal(server.getServerStatus().state, "starting");
  assert.equal(server.getServerStatus().vm, null);

  // Second attempt fails at the alloc; nothing new leaks.
  await assert.rejects(() => server.boot(vm), /out of memory/);
  assert.equal(vm.ram.allocs.length, 1);
  assert.deepEqual(vm.ram.freed, [vm.ram.allocs[0]?.id]);
});

test("boot failure at API start frees the vRAM reservation", async (t) => {
  const { vm, registry, factory } = makeHarness();
  // Occupy an ephemeral port first, then boot a server on that same
  // port: start() rejects with EADDRINUSE after the reservation was
  // made, so boot must free it.
  const { createServer } = await import("node:http");
  const squatter = createServer((_req, res) => res.end("squatter"));
  await new Promise<void>((resolve) => squatter.listen(0, resolve));
  t.after(() => new Promise<void>((resolve) => squatter.close(() => resolve())));
  const takenPort = (squatter.address() as { port: number }).port;

  const server = new VirtualServer({
    registry,
    factory,
    options: { port: takenPort },
  } as unknown as VirtualServerDeps);
  t.after(() => server.close());

  await assert.rejects(() => server.boot(vm), /EADDRINUSE/);
  assert.equal(vm.ram.allocs.length, 1);
  assert.deepEqual(vm.ram.freed, [vm.ram.allocs[0]?.id]);
  assert.equal(server.getServerStatus().state, "starting");
});

// ---------------------------------------------------------------------------
// load / activate / switch
// ---------------------------------------------------------------------------

test("load/activate/switch: activating the second model disposes the first", async (t) => {
  const { server, factory } = makeHarness();
  t.after(() => server.close());

  const entry = await server.loadModel("model-a");
  assert.equal(entry.id, "model-a");
  assert.equal(entry.status, "ready");

  const ctxA = await server.activateModel("model-a");
  assert.equal(server.getActive(), ctxA);
  assert.equal(server.getServerStatus().activeModelId, "model-a");
  assert.deepEqual(factory.built, ["model-a"]);
  assert.equal(factory.disposed.length, 0);

  const ctxB = await server.activateModel("model-b");
  assert.equal(server.getActive(), ctxB);
  assert.equal(server.getServerStatus().activeModelId, "model-b");
  assert.deepEqual(factory.built, ["model-a", "model-b"]);
  assert.deepEqual(factory.disposed, [ctxA]);

  assert.deepEqual(
    server.listModels().map((m) => m.id),
    ["model-a", "model-b", "model-c"],
  );
  assert.equal(server.listModels()[0]?.object, "model");
  assert.equal(server.listModels()[0]?.owned_by, "softwell");
});

test("activateModel throws for unknown and non-ready models", async (t) => {
  const { server } = makeHarness();
  t.after(() => server.close());

  await assert.rejects(() => server.activateModel("nope"), /unknown model: nope/);
  await assert.rejects(
    () => server.activateModel("model-c"),
    /model model-c is not ready \(status: available\)/,
  );
  assert.equal(server.getActive(), null);
});

test("activateModel keeps the previous model active when build throws", async (t) => {
  const { server, factory } = makeHarness();
  t.after(() => server.close());

  await server.activateModel("model-a");
  const before = server.getActive();
  factory.throwOnBuildId = "model-b";

  await assert.rejects(
    () => server.activateModel("model-b"),
    /fake factory cannot build model-b/,
  );
  assert.equal(server.getActive(), before);
  assert.equal(server.getServerStatus().activeModelId, "model-a");
  assert.equal(factory.disposed.length, 0);
});

test("unloadModel refuses the active model and unloads others", async (t) => {
  const { server, registry } = makeHarness();
  t.after(() => server.close());

  await server.activateModel("model-a");
  registry.activeId = "model-a";

  assert.throws(() => server.unloadModel("model-a"), /cannot unload the active model/);
  await server.unloadModel("model-b");
  assert.deepEqual(registry.unloaded, ["model-b"]);
  assert.equal(registry.get("model-b")?.status, "available");
});

// ---------------------------------------------------------------------------
// admit
// ---------------------------------------------------------------------------

test("admit: 429 with no active model, on mismatch, and when the queue is full", async (t) => {
  const { server, factory } = makeHarness();
  t.after(() => server.close());

  const noModel = server.admit("model-a");
  assert.equal(noModel.ok, false);
  if (!noModel.ok) {
    assert.equal(noModel.status, 429);
    assert.match(noModel.message, /no active model/);
  }
  assert.equal(server.counters().rejected, 1);

  await server.activateModel("model-a");

  const mismatch = server.admit("model-b");
  assert.equal(mismatch.ok, false);
  if (!mismatch.ok) {
    assert.equal(mismatch.status, 429);
    assert.match(mismatch.message, /not active/);
  }
  assert.equal(server.counters().rejected, 2);

  // Default maxQueue is 32: 20 pending + 12 running hits the limit.
  const scheduler = factory.lastScheduler as FakeScheduler;
  scheduler.pending = 20;
  scheduler.running = 12;
  const full = server.admit("model-a");
  assert.equal(full.ok, false);
  if (!full.ok) {
    assert.equal(full.status, 429);
    assert.match(full.message, /queue is full/);
    assert.match(full.message, /32/);
  }
  assert.equal(server.counters().rejected, 3);

  scheduler.pending = 10;
  scheduler.running = 10;
  assert.deepEqual(server.admit("model-a"), { ok: true });
  assert.equal(server.counters().rejected, 3);
});

test("admit honours a configured maxQueue", async (t) => {
  const { server, factory } = makeHarness({ serverOptions: { maxQueue: 2 } });
  t.after(() => server.close());

  await server.activateModel("model-a");
  const scheduler = factory.lastScheduler as FakeScheduler;
  scheduler.pending = 1;
  scheduler.running = 1;

  const full = server.admit("model-a");
  assert.equal(full.ok, false);
  if (!full.ok) assert.match(full.message, /limit 2/);
});

// ---------------------------------------------------------------------------
// record / counters / ring buffer
// ---------------------------------------------------------------------------

test("record updates counters and measures tokensPerSec (EMA)", async (t) => {
  const { server } = makeHarness();
  t.after(() => server.close());

  assert.equal(server.getServerStatus().tokensPerSec, 0);

  // 10 tokens in 1s -> 10 tok/s (seeds the EMA).
  server.record(makeRecord({ id: "r1", status: "done", completionTokens: 10, startedAt: 1000, endedAt: 2000 }));
  // 20 tokens in 1s -> 20 tok/s; EMA: 0.8*10 + 0.2*20 = 12.
  server.record(makeRecord({ id: "r2", status: "done", completionTokens: 20, startedAt: 2000, endedAt: 3000 }));
  // 20 tokens in 2s -> 10 tok/s; EMA: 0.8*12 + 0.2*10 = 11.6.
  server.record(makeRecord({ id: "r3", status: "done", completionTokens: 20, startedAt: 3000, endedAt: 5000 }));
  server.record(makeRecord({ id: "r4", status: "cancelled", completionTokens: 0 }));
  server.record(makeRecord({ id: "r5", status: "rejected", completionTokens: 0 }));

  const counters = server.counters();
  assert.deepEqual(counters, { active: 0, queued: 0, completed: 3, rejected: 1, cancelled: 1 });

  const status = server.getServerStatus();
  assert.ok(Math.abs(status.tokensPerSec - 11.6) < 1e-9, `tokensPerSec was ${status.tokensPerSec}`);
  assert.deepEqual(
    status.recentRequests.map((r) => r.id),
    ["r5", "r4", "r3", "r2", "r1"],
  );
});

test("record tracks live queued/running counts through transitions", async (t) => {
  const { server } = makeHarness();
  t.after(() => server.close());

  server.record(makeRecord({ id: "q1", status: "queued" }));
  server.record(makeRecord({ id: "q2", status: "queued" }));
  assert.deepEqual(server.counters(), { active: 0, queued: 2, completed: 0, rejected: 0, cancelled: 0 });

  server.record(makeRecord({ id: "q1", status: "running" }));
  assert.deepEqual(server.counters(), { active: 1, queued: 1, completed: 0, rejected: 0, cancelled: 0 });

  server.record(makeRecord({ id: "q1", status: "done", completionTokens: 5, startedAt: 1000, endedAt: 2000 }));
  assert.deepEqual(server.counters(), { active: 0, queued: 1, completed: 1, rejected: 0, cancelled: 0 });
});

test("recentRequests ring buffer caps at 200, newest first", async (t) => {
  const { server } = makeHarness();
  t.after(() => server.close());

  for (let i = 0; i < 201; i++) {
    server.record(makeRecord({ id: `r${i}`, status: "done" }));
  }
  const recent = server.getServerStatus().recentRequests;
  assert.equal(recent.length, 200);
  assert.equal(recent[0]?.id, "r200");
  assert.equal(recent[199]?.id, "r1");
  assert.equal(server.counters().completed, 201);
});

// ---------------------------------------------------------------------------
// health / readiness / status
// ---------------------------------------------------------------------------

test("health: down -> degraded -> ok transitions", async (t) => {
  // No boot: down.
  const plain = makeHarness();
  t.after(() => plain.server.close());
  assert.deepEqual(plain.server.getHealth(), {
    status: "down",
    checks: { vmBooted: false, activeModel: false, workersHealthy: false },
  });
  assert.equal(plain.server.isReady(), false);

  // Boot without a model and without a cluster: degraded.
  await plain.server.boot(plain.vm);
  assert.deepEqual(plain.server.getHealth(), {
    status: "degraded",
    checks: { vmBooted: true, activeModel: false, workersHealthy: false },
  });
  assert.equal(plain.server.isReady(), false);

  // Active model, still no healthy worker: degraded.
  await plain.server.activateModel("model-a");
  assert.equal(plain.server.getHealth().status, "degraded");
  assert.equal(plain.server.getHealth().checks.activeModel, true);
  assert.equal(plain.server.isReady(), false);

  // Active model + healthy worker: ok and ready.
  const withCluster = makeHarness({ getCluster: () => healthyCluster("model-a") });
  t.after(() => withCluster.server.close());
  await withCluster.server.boot(withCluster.vm);
  assert.equal(withCluster.server.getHealth().status, "degraded");
  await withCluster.server.activateModel("model-a");
  assert.deepEqual(withCluster.server.getHealth(), {
    status: "ok",
    checks: { vmBooted: true, activeModel: true, workersHealthy: true },
  });
  assert.equal(withCluster.server.isReady(), true);

  // VM stops responding to info(): back to down.
  withCluster.vm.infoThrows = true;
  assert.equal(withCluster.server.getHealth().status, "down");
  assert.equal(withCluster.server.getHealth().checks.vmBooted, false);
  assert.equal(withCluster.server.getServerStatus().vm, null);
});

test("getServerStatus reports queueDepth from the active scheduler", async (t) => {
  const { vm, server, factory } = makeHarness();
  t.after(() => server.close());

  assert.equal(server.getServerStatus().queueDepth, 0);
  assert.equal(server.getServerStatus().uptimeSec, 0);
  assert.equal(server.getServerStatus().vm, null);

  await server.boot(vm);
  assert.equal(server.getServerStatus().queueDepth, 0);
  assert.ok(server.getServerStatus().uptimeSec >= 0);
  assert.notEqual(server.getServerStatus().vm, null);

  await server.activateModel("model-a");
  const scheduler = factory.lastScheduler as FakeScheduler;
  scheduler.pending = 3;
  scheduler.running = 2;
  const status = server.getServerStatus();
  assert.equal(status.queueDepth, 3);
  assert.equal(status.activeModelId, "model-a");
  assert.equal(status.models.length, 3);
});

// ---------------------------------------------------------------------------
// close
// ---------------------------------------------------------------------------

test("close frees the vRAM reservation, disposes the active context, and is idempotent", async (t) => {
  const { vm, server, factory } = makeHarness();
  t.after(() => server.close());

  await server.boot(vm);
  const ctx = await server.activateModel("model-a");
  const allocId = vm.ram.allocs[0]?.id as string;

  await server.close();
  assert.deepEqual(vm.ram.freed, [allocId]);
  assert.deepEqual(factory.disposed, [ctx]);
  assert.equal(server.getServerStatus().state, "stopping");
  assert.equal(server.getActive(), null);

  // Second close is a no-op: no double free, no double dispose.
  await server.close();
  assert.deepEqual(vm.ram.freed, [allocId]);
  assert.deepEqual(factory.disposed, [ctx]);
});

test("close before boot is a safe no-op except for state", async () => {
  const { server } = makeHarness();
  await server.close();
  assert.equal(server.getServerStatus().state, "stopping");
  await server.close();
  assert.equal(server.getServerStatus().state, "stopping");
});

test("requestShutdown sets stopping and calls onShutdown once", async (t) => {
  let calls = 0;
  const { server } = makeHarness({ onShutdown: () => { calls += 1; } });
  t.after(() => server.close());

  server.requestShutdown();
  server.requestShutdown();
  assert.equal(calls, 1);
  assert.equal(server.getServerStatus().state, "stopping");
});

// ---------------------------------------------------------------------------
// vCPU handlers
// ---------------------------------------------------------------------------

test("vCPU handlers tokenize/detokenize with the active model", async (t) => {
  const { vm, server } = makeHarness();
  t.after(() => server.close());

  await server.boot(vm);
  assert.equal(vm.cpu.handlers.has("tokenize"), true);
  assert.equal(vm.cpu.handlers.has("detokenize"), true);

  // No active model: a clear error.
  await assert.rejects(
    () => vm.cpu.run("tokenize", { text: "hi" }),
    /no active model: cannot tokenize/,
  );
  await assert.rejects(
    () => vm.cpu.run("detokenize", { ids: [104, 105] }),
    /no active model: cannot detokenize/,
  );

  await server.activateModel("model-a");

  const ids = await vm.cpu.run("tokenize", { text: "hi" });
  assert.deepEqual(ids, [104, 105]);
  const text = await vm.cpu.run("detokenize", { ids: [104, 105] });
  assert.equal(text, "hi");

  // Switching models switches the tokenizer the handlers use.
  await server.activateModel("model-b");
  assert.deepEqual(await vm.cpu.run("tokenize", { text: "hi" }), [104, 105]);
});
