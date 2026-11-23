/**
 * vRAM weights tests (registry weight map + binary endpoint + worker).
 * Run with: node --test test/vram-weights.test.ts
 *
 * Uses a fake VM whose vRAM really stores bytes (alloc/write/read), so
 * the staged bytes can be compared against the on-disk shard bytes.
 *
 *  A. Registry: loading demo-100m builds a per-tensor weight map whose
 *     locations resolve to exactly the staged shard bytes; readWeightTensor
 *     serves those bytes and counts them; unknown model/tensor throw;
 *     unload clears the map. The readVram dep is exercised, plus the
 *     vm().ram.read fallback when it is omitted.
 *  B. Endpoint: GET /mesh/weights serves one tensor as
 *     application/octet-stream with bytes equal to the staged shard
 *     slice (404s for unknown model/tensor, 400 for missing params,
 *     405 for POST); GET /mesh/weights/map serves the manifest + map.
 *     Both routes use createWeightHandlers — the same construction the
 *     CLI wires into the real orchestrator — so the tested path is the
 *     shipped path.
 *  C. Worker: a real startWorker against a fake orchestrator backed by
 *     the registry loads a stage for the staged model through a
 *     VramWeightLoader (spawn disk loader reads 0 bytes, orchestrator
 *     serves bytes from vRAM), and falls back to the disk loader for a
 *     model the orchestrator has not staged.
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import type {
  SsdObjectMeta,
  SsdStats,
  VirtualMachine,
  VramAllocation,
  VramStats,
  VramTensorLocation,
} from "../src/contracts.ts";
import { loadManifest, openWeightLoader, SafetensorsWeightLoader } from "../src/model/loader.ts";
import { safetensorsHeaderLength } from "../src/model/manifest.ts";
import { VramWeightLoader } from "../src/model/vram-loader.ts";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import { ModelRegistry, createWeightHandlers } from "../src/server/registry.ts";
import type {
  EngineBackend,
  ForwardContext,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(
  cond: () => boolean,
  what: string,
  timeoutMs = 5000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cond()) return;
    if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
    await sleep(10);
  }
}

/** Fake vRAM that really stores bytes per allocation. */
class FakeRam {
  readonly totalBytes: number;
  usedBytes = 0;
  private nextVaddr = 0x1000;
  private nextId = 0;
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();

  constructor(totalBytes: number) {
    this.totalBytes = totalBytes;
  }

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    if (sizeBytes > this.totalBytes - this.usedBytes) {
      throw new Error(
        `fake vRAM out of memory: ${sizeBytes} bytes requested, ${this.totalBytes - this.usedBytes} free`,
      );
    }
    const id = `alloc-${++this.nextId}`;
    const vaddr = this.nextVaddr;
    this.nextVaddr += sizeBytes;
    this.allocs.set(id, { vaddr, data: new Uint8Array(sizeBytes) });
    this.usedBytes += sizeBytes;
    return { id, vaddr, sizeBytes, pages: [], replicas: 1 };
  }

  async free(allocId: string): Promise<void> {
    const alloc = this.allocs.get(allocId);
    if (alloc === undefined) throw new Error(`unknown allocation: ${allocId}`);
    this.allocs.delete(allocId);
    this.usedBytes -= alloc.data.length;
  }

  private find(vaddr: number): { base: number; data: Uint8Array } {
    for (const { vaddr: base, data } of this.allocs.values()) {
      if (vaddr >= base && vaddr < base + data.length) return { base, data };
    }
    throw new Error(`fake vRAM access outside any allocation: ${vaddr}`);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const { base, data } = this.find(vaddr);
    const off = vaddr - base;
    if (off + length > data.length) throw new Error("fake vRAM read past allocation end");
    return data.slice(off, off + length);
  }

  async write(vaddr: number, chunk: Uint8Array): Promise<void> {
    const { base, data } = this.find(vaddr);
    data.set(chunk, vaddr - base);
  }

  async replicate(): Promise<void> {}

  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): VramStats {
    return {
      totalBytes: this.totalBytes,
      usedBytes: this.usedBytes,
      freeBytes: this.totalBytes - this.usedBytes,
      pages: 0,
      nodes: 1,
    };
  }
}

class FakeSsd {
  private readonly objects = new Map<string, Uint8Array>();

  get(key: string): Uint8Array | undefined {
    return this.objects.get(key);
  }

  async putObject(key: string, data: Uint8Array): Promise<SsdObjectMeta> {
    this.objects.set(key, data);
    return { key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: Date.now() };
  }

  async getObject(key: string): Promise<Uint8Array> {
    const data = this.objects.get(key);
    if (!data) throw new Error(`object not found: ${key}`);
    return data;
  }

  async getRange(key: string, offset: number, length: number): Promise<Uint8Array> {
    const data = await this.getObject(key);
    return data.slice(offset, offset + length);
  }

  async list(prefix = ""): Promise<SsdObjectMeta[]> {
    const out: SsdObjectMeta[] = [];
    for (const [key, data] of this.objects) {
      if (key.startsWith(prefix)) {
        out.push({ key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: 0 });
      }
    }
    return out.sort((a, b) => (a.key < b.key ? -1 : 1));
  }

  async deleteObject(key: string): Promise<void> {
    this.objects.delete(key);
  }

  async handleNodeLoss(): Promise<{ chunksRecovered: number; chunksLost: number }> {
    return { chunksRecovered: 0, chunksLost: 0 };
  }

  stats(): SsdStats {
    let totalBytes = 0;
    for (const data of this.objects.values()) totalBytes += data.length;
    return { objects: this.objects.size, totalBytes, nodes: 1 };
  }
}

function makeFakeVm(totalVramBytes: number): { vm: VirtualMachine; ram: FakeRam; ssd: FakeSsd } {
  const ram = new FakeRam(totalVramBytes);
  const ssd = new FakeSsd();
  const vm = {
    ram,
    ssd,
    gpu: {
      listDevices: () => [],
      createStream: () => "stream-0",
      launch: async () => {
        throw new Error("fake vGPU cannot launch kernels");
      },
      synchronize: async () => {},
      transfer: async () => {},
      handleNodeLoss: async () => ({ streamsReassigned: 0, devicesLost: 0 }),
      stats: () => ({ devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 }),
    },
    cpu: {
      registerHandler: () => {},
      submit: async () => {
        throw new Error("fake vCPU cannot run tasks");
      },
      handleNodeLoss: async () => ({ tasksRequeued: 0, tasksLost: 0 }),
      stats: () => ({ nodes: 1, totalCores: 4, queued: 0, completed: 0, failed: 0 }),
    },
    info: () => ({
      vram: ram.stats(),
      gpu: { devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 },
      cpu: { nodes: 1, totalCores: 4, queued: 0, completed: 0, failed: 0 },
      ssd: ssd.stats(),
      nodes: [],
    }),
  } as unknown as VirtualMachine;
  return { vm, ram, ssd };
}

function makeStageToVram(ssd: FakeSsd, ram: FakeRam): (key: string) => Promise<VramAllocation> {
  return async (key: string) => {
    const data = ssd.get(key);
    if (!data) throw new Error(`stageToVram: object not found in vSSD: ${key}`);
    const alloc = await ram.alloc(data.length);
    await ram.write(alloc.vaddr, data);
    return alloc;
  };
}

const demo10mDir = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));
const demo100mDir = fileURLToPath(new URL("../demo/models/demo-100m", import.meta.url));
const demo100mShard = fs.readFileSync(path.join(demo100mDir, "model.safetensors"));
const demo100mHeaderLen = safetensorsHeaderLength(demo100mShard);
const manifest100m = loadManifest(demo100mDir, "demo-100m");
const firstTensor100m = Object.keys(manifest100m.tensors)[0] as string;
const firstMeta100m = manifest100m.tensors[firstTensor100m] as {
  dataOffsets: [number, number];
};
const firstBytes100m = demo100mShard.subarray(
  demo100mHeaderLen + firstMeta100m.dataOffsets[0],
  demo100mHeaderLen + firstMeta100m.dataOffsets[1],
);

function makeCache(): PagedKVCache {
  return {
    blockSize: 16,
    allocate: async () => [],
    append: async () => {},
    read: async () => ({ keys: new Float32Array(0), values: new Float32Array(0) }),
    free: async () => {},
    close: async () => {},
    usedBlocks: () => 0,
    totalBlocks: () => 0,
  };
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-vram-weights-test-"));
let endpointServer: Awaited<ReturnType<typeof createMeshServer>> | undefined;
let orchServer: Awaited<ReturnType<typeof createMeshServer>> | undefined;
let workerHandle: Awaited<ReturnType<typeof startWorker>> | undefined;
const spawnLoader = openWeightLoader(demo10mDir);
const copyDir = path.join(tmp, `softwell-demo-10m-copy-${process.pid}`);
try {
  const modelsDir = path.join(tmp, "models");
  fs.mkdirSync(modelsDir, { recursive: true });

  // --- A. Registry weight map for demo-100m. ---
  const { vm, ram, ssd } = makeFakeVm(4 * 1024 ** 3);
  const registry = new ModelRegistry({
    vm: () => vm,
    stageToVram: makeStageToVram(ssd, ram),
    readVram: (vaddr, length) => ram.read(vaddr, length),
    isActive: () => false,
    modelsDir,
  });
  const entry100m = await registry.load(demo100mDir);
  assert.equal(entry100m.status, "ready");
  assert.equal(registry.weightBytesServed, 0, "nothing served before any read");

  const weightMap = registry.getWeightMap("demo-100m");
  if (weightMap === undefined) throw new Error("staged model has no weight map");
  assert.deepEqual(
    Object.keys(weightMap).sort(),
    Object.keys(manifest100m.tensors).sort(),
    "weight map covers every tensor in the manifest",
  );
  let servedTotal = 0;
  for (const [name, meta] of Object.entries(manifest100m.tensors)) {
    // Explicit annotation: without it tsc reports TS7022 ('loc' implicitly
    // has type 'any' ...) for this indexed access inside the loop.
    const loc: VramTensorLocation | undefined = weightMap[name];
    assert.ok(loc !== undefined);
    assert.equal(loc.headerLen, demo100mHeaderLen, `tensor ${name}: headerLen`);
    assert.deepEqual(loc.dataOffsets, meta.dataOffsets, `tensor ${name}: dataOffsets`);
    assert.equal(loc.dtype, meta.dtype, `tensor ${name}: dtype`);
    assert.deepEqual(loc.shape, meta.shape, `tensor ${name}: shape`);
    assert.ok(loc.allocId.startsWith("alloc-"), `tensor ${name}: allocId`);
    assert.equal(typeof loc.vaddr, "number");
    // The bytes at the mapped location are exactly the staged shard bytes.
    const expected = demo100mShard.subarray(
      demo100mHeaderLen + meta.dataOffsets[0],
      demo100mHeaderLen + meta.dataOffsets[1],
    );
    const actual = await registry.readWeightTensor("demo-100m", name);
    assert.ok(Buffer.from(actual).equals(expected), `tensor ${name}: served bytes differ`);
    servedTotal += expected.length;
  }
  assert.equal(
    registry.weightBytesServed,
    servedTotal,
    "weightBytesServed counts every served tensor byte",
  );
  assert.equal(servedTotal, manifest100m.totalBytes, "all tensor bytes served once");

  // The map is a copy: mutating it does not corrupt the registry.
  const mapCopy = registry.getWeightMap("demo-100m") as Record<string, { vaddr: number }>;
  (mapCopy[firstTensor100m] as { vaddr: number }).vaddr = 999;
  assert.notEqual(
    (registry.getWeightMap("demo-100m") as Record<string, { vaddr: number }>)[firstTensor100m]?.vaddr,
    999,
  );

  const mapResponse = registry.getWeightMapResponse("demo-100m");
  assert.ok(mapResponse !== null);
  assert.equal(mapResponse.modelId, "demo-100m");
  assert.equal(mapResponse.manifest.modelId, "demo-100m");
  assert.deepEqual(Object.keys(mapResponse.weightMap).sort(), Object.keys(manifest100m.tensors).sort());

  await assert.rejects(
    () => registry.readWeightTensor("no-such-model", firstTensor100m),
    /no weights staged in vRAM/,
  );
  await assert.rejects(
    () => registry.readWeightTensor("demo-100m", "no.such.tensor"),
    /not staged in vRAM/,
  );
  assert.equal(registry.getWeightMap("no-such-model"), undefined);
  assert.equal(registry.getWeightMapResponse("no-such-model"), null);
  console.log("registry builds a correct per-tensor weight map: ok");

  // readVram fallback: a registry without the dep reads via vm().ram.read.
  const fallbackRegistry = new ModelRegistry({
    vm: () => vm,
    stageToVram: makeStageToVram(ssd, ram),
    isActive: () => false,
    modelsDir,
  });
  const entry10m = await fallbackRegistry.load(demo10mDir);
  assert.equal(entry10m.status, "ready");
  const manifest10m = loadManifest(demo10mDir, "demo-10m");
  const firstTensor10m = Object.keys(manifest10m.tensors)[0] as string;
  const firstMeta10m = manifest10m.tensors[firstTensor10m] as { dataOffsets: [number, number] };
  const demo10mShard = fs.readFileSync(path.join(demo10mDir, "model.safetensors"));
  const demo10mHeaderLen = safetensorsHeaderLength(demo10mShard);
  const expected10m = demo10mShard.subarray(
    demo10mHeaderLen + firstMeta10m.dataOffsets[0],
    demo10mHeaderLen + firstMeta10m.dataOffsets[1],
  );
  const actual10m = await fallbackRegistry.readWeightTensor("demo-10m", firstTensor10m);
  assert.ok(Buffer.from(actual10m).equals(expected10m), "fallback read returns the staged bytes");
  console.log("registry readVram dep + vm().ram.read fallback: ok");

  // --- B. Binary endpoint (same handlers the CLI wires in). ---
  const weightHandlers = createWeightHandlers(registry);
  endpointServer = createMeshServer({
    getWeightMap: weightHandlers.getWeightMap,
    getWeightTensor: weightHandlers.getWeightTensor,
  });
  await new Promise<void>((resolve, reject) => {
    endpointServer?.once("error", reject);
    endpointServer?.listen(0, "127.0.0.1", () => resolve());
  });
  const endpointPort = (endpointServer.address() as AddressInfo).port;
  const endpointUrl = `http://127.0.0.1:${endpointPort}`;

  const servedBeforeEndpoint = registry.weightBytesServed;
  const weightsRes = await fetch(
    `${endpointUrl}/mesh/weights?model=demo-100m&tensor=${encodeURIComponent(firstTensor100m)}`,
  );
  assert.equal(weightsRes.status, 200);
  assert.ok(
    (weightsRes.headers.get("content-type") ?? "").includes("application/octet-stream"),
    `weights content-type is octet-stream, got ${weightsRes.headers.get("content-type")}`,
  );
  const weightsBody = Buffer.from(await weightsRes.arrayBuffer());
  assert.ok(weightsBody.equals(firstBytes100m), "endpoint bytes equal the staged shard slice");
  assert.equal(
    registry.weightBytesServed,
    servedBeforeEndpoint + firstBytes100m.length,
    "endpoint read counts toward weightBytesServed",
  );

  const mapRes = await fetch(`${endpointUrl}/mesh/weights/map?model=demo-100m`);
  assert.equal(mapRes.status, 200);
  assert.ok((mapRes.headers.get("content-type") ?? "").includes("application/json"));
  const mapBody = (await mapRes.json()) as {
    modelId: string;
    manifest: { modelId: string };
    weightMap: Record<string, unknown>;
  };
  assert.equal(mapBody.modelId, "demo-100m");
  assert.equal(mapBody.manifest.modelId, "demo-100m");
  assert.ok(firstTensor100m in mapBody.weightMap);

  assert.equal(
    (await fetch(`${endpointUrl}/mesh/weights?model=nope&tensor=${encodeURIComponent(firstTensor100m)}`)).status,
    404,
    "unknown model -> 404",
  );
  assert.equal(
    (await fetch(`${endpointUrl}/mesh/weights?model=demo-100m&tensor=nope`)).status,
    404,
    "unknown tensor -> 404",
  );
  assert.equal(
    (await fetch(`${endpointUrl}/mesh/weights?model=demo-100m`)).status,
    400,
    "missing tensor param -> 400",
  );
  assert.equal(
    (await fetch(`${endpointUrl}/mesh/weights/map?model=nope`)).status,
    404,
    "unknown model map -> 404",
  );
  assert.equal(
    (await fetch(`${endpointUrl}/mesh/weights?model=demo-100m&tensor=x`, { method: "POST" })).status,
    405,
    "POST on the weights route -> 405",
  );
  console.log("binary weight endpoint (octet-stream, 404s, map): ok");

  // --- C. Worker: vRAM path when staged, disk fallback when not. ---
  const orchState: { version: number; assignment: ShardAssignment | null } = {
    version: 0,
    assignment: null,
  };
  orchServer = createMeshServer({
    register: () => ({ workerId: "worker-test-1", heartbeatIntervalMs: 50 }),
    heartbeat: () => ({ ok: true, assignmentVersion: orchState.version }),
    getAssignment: () => ({
      version: orchState.version,
      assignment: orchState.assignment,
    }),
    getWeightMap: weightHandlers.getWeightMap,
    getWeightTensor: weightHandlers.getWeightTensor,
  });
  await new Promise<void>((resolve, reject) => {
    orchServer?.once("error", reject);
    orchServer?.listen(0, "127.0.0.1", () => resolve());
  });
  const orchPort = (orchServer.address() as AddressInfo).port;
  const orchUrl = `http://127.0.0.1:${orchPort}`;
  const client = new MeshClient({ timeoutMs: 5000 });

  const builtFromManifests: string[] = [];
  const loadedLoaders: WeightLoader[] = [];
  const backend: EngineBackend = {
    name: "fake-recording",
    createStageExecutor(manifest: ModelManifest, assignment: ShardAssignment): StageExecutor {
      builtFromManifests.push(manifest.modelId);
      return {
        assignment,
        load: async (loader: WeightLoader) => {
          loadedLoaders.push(loader);
          // Load one real tensor through the loader, as the engine does.
          const first = Object.keys(loader.manifest.tensors)[0] as string;
          const data = await loader.loadTensor(first);
          assert.ok(data.length > 0, "loaded tensor must be non-empty");
        },
        embed: (tokenIds: number[]): Tensor => ({
          data: new Float32Array(tokenIds.length * 2),
          shape: [tokenIds.length, 2],
        }),
        forward: async (input: Tensor, _ctx: ForwardContext): Promise<Tensor> => input,
        memoryBytes: () => 4096,
      };
    },
  };

  const makeAssignment = (
    workerId: string,
    opts: { modelId?: string; modelDir?: string },
  ): ShardAssignment => ({
    workerId,
    stage: 0,
    layerStart: 0,
    layerEnd: 6,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 1024,
    ...(opts.modelId !== undefined ? { modelId: opts.modelId } : {}),
    ...(opts.modelDir !== undefined ? { modelDir: opts.modelDir } : {}),
  });

  workerHandle = await startWorker({
    orchestratorUrl: orchUrl,
    host: "127.0.0.1",
    port: 0,
    capability: {
      device: "cpu",
      backend: "fake-recording",
      memoryBytes: 1 << 30,
      freeMemoryBytes: 1 << 30,
      flopsPerSec: 1e12,
      bandwidthBytesPerSec: 1e9,
      roles: ["mixed"],
      maxBatchTokens: 256,
    },
    backend,
    loader: spawnLoader,
    createCache: makeCache,
    pollIntervalMs: 50,
  });
  const workerId = workerHandle.workerId;
  const workerUrl = `http://127.0.0.1:${workerHandle.port}`;
  const servedBeforeWorker = registry.weightBytesServed;
  assert.equal(spawnLoader.bytesReadFromDisk, 0, "spawn loader has read nothing yet");

  // (C1) Staged model: the worker must use the vRAM loader, not disk.
  orchState.assignment = makeAssignment(workerId, {
    modelId: "demo-100m",
    modelDir: demo100mDir,
  });
  orchState.version = 1;
  // Wait for the stage LOAD (not just executor creation): the fake
  // backend's load() fetches a tensor through the loader, so the
  // vRAM loader's counters only move once load() has run.
  await waitFor(
    () =>
      loadedLoaders.at(-1) instanceof VramWeightLoader &&
      (loadedLoaders.at(-1) as VramWeightLoader).tensorsFetched >= 1,
    "vRAM stage load",
  );
  const vramLoader = loadedLoaders.at(-1);
  assert.ok(vramLoader instanceof VramWeightLoader, "staged model loads via VramWeightLoader");
  assert.ok(vramLoader.tensorsFetched >= 1, "vRAM loader fetched at least one tensor");
  assert.ok(vramLoader.bytesFetched > 0, "vRAM loader fetched bytes");
  assert.equal(
    registry.weightBytesServed,
    servedBeforeWorker + firstBytes100m.length,
    "orchestrator served exactly the fetched tensor's bytes from vRAM",
  );
  assert.equal(
    spawnLoader.bytesReadFromDisk,
    0,
    "no disk read happened while loading the staged model",
  );
  const execResp = await client.execute(workerUrl, {
    workerId,
    seqIds: ["seq-1", "seq-1"],
    kind: "prefill",
    stage: 0,
    input: { data: [1, 2, 3, 4], shape: [2, 2] },
    positions: [0, 1],
  });
  assert.deepEqual(execResp.output, { data: [1, 2, 3, 4], shape: [2, 2] });
  console.log("worker loads a staged model from vRAM with zero disk reads: ok");

  // (C2) Unstaged model: the worker falls back to the disk loader.
  fs.cpSync(demo10mDir, copyDir, { recursive: true });
  const copyId = path.basename(copyDir);
  assert.notEqual(copyId, "demo-10m");
  orchState.assignment = makeAssignment(workerId, {
    modelId: copyId,
    modelDir: copyDir,
  });
  orchState.version = 2;
  await waitFor(
    () =>
      loadedLoaders.at(-1) instanceof SafetensorsWeightLoader &&
      (loadedLoaders.at(-1) as SafetensorsWeightLoader).bytesReadFromDisk > 0,
    "disk fallback stage load",
  );
  const diskLoader = loadedLoaders.at(-1);
  assert.ok(
    diskLoader instanceof SafetensorsWeightLoader,
    "unstaged model falls back to the disk loader",
  );
  assert.ok(
    diskLoader.bytesReadFromDisk > 0,
    "disk fallback really read the shard from disk",
  );
  assert.notEqual(diskLoader, spawnLoader, "fallback opened its own loader for the copy");
  await client.execute(workerUrl, {
    workerId,
    seqIds: ["seq-1", "seq-1"],
    kind: "prefill",
    stage: 0,
    input: { data: [1, 2, 3, 4], shape: [2, 2] },
    positions: [0, 1],
  });
  console.log("worker falls back to the disk loader for an unstaged model: ok");

  // --- A (continued). Unload clears the weight map. ---
  await registry.unload("demo-100m");
  assert.equal(registry.getWeightMap("demo-100m"), undefined);
  assert.equal(registry.getWeightMapResponse("demo-100m"), null);
  await assert.rejects(
    () => registry.readWeightTensor("demo-100m", firstTensor100m),
    /no weights staged in vRAM/,
  );
  console.log("unload clears the weight map: ok");

  console.log("vRAM weights test: all checks passed");
} finally {
  if (workerHandle !== undefined) await workerHandle.close();
  if (orchServer !== undefined) {
    await new Promise<void>((resolve) => orchServer?.close(() => resolve()));
  }
  if (endpointServer !== undefined) {
    await new Promise<void>((resolve) => endpointServer?.close(() => resolve()));
  }
  spawnLoader.close();
  fs.rmSync(tmp, { recursive: true, force: true });
}
