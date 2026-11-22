/**
 * Model registry test (src/server/registry.ts).
 * Run with: node --test test/model-registry.test.ts
 *
 * Uses an in-memory fake VM (byte-accounting vRAM, fake vSSD, fake
 * stageToVram delegating to the fake vRAM) plus a fake isActive. Loads
 * the real demo models from disk and checks: ready status, manifest
 * params/weightBytes, vRAM accounting and unload, fit failure with no
 * leaked allocation, unknown refs/ids throwing, load-twice identity,
 * active-model unload refusal, and listAvailable from disk + vSSD.
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  SsdObjectMeta,
  SsdStats,
  VirtualMachine,
  VramAllocation,
  VramStats,
} from "../src/contracts.ts";
import { loadManifest } from "../src/model/loader.ts";
import { ModelRegistry } from "../src/server/registry.ts";

class FakeRam {
  readonly totalBytes: number;
  usedBytes = 0;
  private nextVaddr = 0x1000;
  private nextId = 0;
  private readonly allocs = new Map<string, number>();

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
    this.allocs.set(id, sizeBytes);
    this.usedBytes += sizeBytes;
    return { id, vaddr, sizeBytes, pages: [], replicas: 1 };
  }

  async free(allocId: string): Promise<void> {
    const size = this.allocs.get(allocId);
    if (size === undefined) throw new Error(`unknown allocation: ${allocId}`);
    this.allocs.delete(allocId);
    this.usedBytes -= size;
  }

  async read(): Promise<Uint8Array> {
    return new Uint8Array(0);
  }

  async write(): Promise<void> {}

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
    return ram.alloc(data.length);
  };
}

const demo10mDir = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));
const demo100mDir = fileURLToPath(new URL("../demo/models/demo-100m", import.meta.url));
const demoModelsDir = fileURLToPath(new URL("../demo/models", import.meta.url));
const demo10mShardBytes = fs.statSync(path.join(demo10mDir, "model.safetensors")).size;
const demo100mShardBytes = fs.statSync(path.join(demo100mDir, "model.safetensors")).size;

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-registry-test-"));
try {
  const modelsDir = path.join(tmp, "models");
  fs.mkdirSync(modelsDir, { recursive: true });

  // --- Load demo-10m: ready, manifest matches, vRAM accounted. ---
  const { vm, ram, ssd } = makeFakeVm(4 * 1024 ** 3);
  const registry = new ModelRegistry({
    vm: () => vm,
    stageToVram: makeStageToVram(ssd, ram),
    isActive: () => false,
    modelsDir,
  });
  const manifest10m = loadManifest(demo10mDir, "demo-10m");
  const entry10m = await registry.load(demo10mDir);
  assert.equal(entry10m.id, "demo-10m");
  assert.equal(entry10m.status, "ready");
  assert.equal(entry10m.params, manifest10m.totalParams);
  assert.equal(entry10m.weightBytes, manifest10m.totalBytes);
  assert.equal(entry10m.numLayers, manifest10m.config.numHiddenLayers);
  assert.deepEqual(entry10m.architectures, manifest10m.config.architectures);
  assert.equal(entry10m.tokenizerKind, "byte-level");
  assert.equal(entry10m.vramBytes, demo10mShardBytes);
  assert.equal(typeof entry10m.weightVaddr, "number");
  assert.equal(ram.usedBytes, demo10mShardBytes);
  assert.equal(entry10m.source.kind, "local");
  assert.equal(registry.get("demo-10m"), entry10m);
  // Files staged on vSSD under models/<id>/<file>.
  assert.ok(ssd.get("models/demo-10m/model.safetensors") !== undefined);
  assert.ok(ssd.get("models/demo-10m/config.json") !== undefined);
  assert.ok(ssd.get("models/demo-10m/tokenizer.json") !== undefined);

  // --- Load twice returns the same entry (no re-staging). ---
  const again = await registry.load(demo10mDir);
  assert.equal(again, entry10m);
  assert.equal(ram.usedBytes, demo10mShardBytes);

  // --- Load demo-100m alongside. ---
  const manifest100m = loadManifest(demo100mDir, "demo-100m");
  const entry100m = await registry.load(demo100mDir);
  assert.equal(entry100m.status, "ready");
  assert.equal(entry100m.params, manifest100m.totalParams);
  assert.equal(entry100m.weightBytes, manifest100m.totalBytes);
  assert.equal(entry100m.tokenizerKind, "byte-level");
  assert.equal(entry100m.vramBytes, demo100mShardBytes);
  assert.equal(ram.usedBytes, demo10mShardBytes + demo100mShardBytes);
  assert.equal(registry.list().length, 2);

  // --- listAvailable merges loaded entries. ---
  const available = await registry.listAvailable();
  const availableIds = available.map((e) => e.id).sort();
  assert.ok(availableIds.includes("demo-10m"));
  assert.ok(availableIds.includes("demo-100m"));

  // --- Unload: vRAM freed, status available, vSSD copy kept. ---
  await registry.unload("demo-10m");
  assert.equal(entry10m.status, "available");
  assert.equal(entry10m.vramBytes, 0);
  assert.equal(entry10m.weightVaddr, undefined);
  assert.equal(ram.usedBytes, demo100mShardBytes);
  assert.ok(ssd.get("models/demo-10m/model.safetensors") !== undefined, "vSSD copy kept");

  // --- Unloading an "available" model is a no-op success. ---
  await registry.unload("demo-10m");

  // --- Unknown id / unknown ref throw clear errors. ---
  await assert.rejects(() => registry.unload("no-such-model"), /unknown model: no-such-model/);
  await assert.rejects(() => registry.load("not-a-model"), /not a valid Hugging Face repo id/);
  await assert.rejects(
    () => registry.load(path.join(tmp, "missing-dir")),
    /not a valid Hugging Face repo id|model directory not found/,
  );

  // --- Unload refuses the active model. ---
  const fake2 = makeFakeVm(4 * 1024 ** 3);
  const activeRegistry = new ModelRegistry({
    vm: () => fake2.vm,
    stageToVram: makeStageToVram(fake2.ssd, fake2.ram),
    isActive: (id) => id === "demo-10m",
    modelsDir,
  });
  await activeRegistry.load(demo10mDir);
  await assert.rejects(() => activeRegistry.unload("demo-10m"), /active model/);
  assert.equal(activeRegistry.get("demo-10m")?.status, "ready", "refused unload leaves it ready");
  assert.equal(fake2.ram.usedBytes, demo10mShardBytes);

  // --- Fit failure: status "error" naming the numbers, nothing leaked. ---
  const tiny = makeFakeVm(1024 * 1024);
  const tinyRegistry = new ModelRegistry({
    vm: () => tiny.vm,
    stageToVram: makeStageToVram(tiny.ssd, tiny.ram),
    isActive: () => false,
    modelsDir,
  });
  const failed = await tinyRegistry.load(demo10mDir);
  assert.equal(failed.status, "error");
  assert.match(failed.error ?? "", /does not fit in vRAM/);
  assert.ok(
    (failed.error ?? "").includes(String(manifest10m.totalBytes)),
    "error names the weight bytes",
  );
  assert.ok((failed.error ?? "").includes("1048576"), "error names the vRAM total");
  assert.equal(failed.vramBytes, 0);
  assert.equal(tiny.ram.usedBytes, 0, "no leaked vRAM allocation");
  assert.equal(tiny.ssd.stats().objects, 0, "nothing staged on fit failure");
  assert.equal(tinyRegistry.get("demo-10m")?.status, "error");

  // --- listAvailable finds vSSD-only models (empty local models dir). ---
  const emptyModelsDir = path.join(tmp, "empty-models");
  fs.mkdirSync(emptyModelsDir, { recursive: true });
  const ssdOnlyRegistry = new ModelRegistry({
    vm: () => vm,
    stageToVram: makeStageToVram(ssd, ram),
    isActive: () => false,
    modelsDir: emptyModelsDir,
  });
  const ssdAvailable = await ssdOnlyRegistry.listAvailable();
  const ssdEntry = ssdAvailable.find((e) => e.id === "demo-100m");
  assert.ok(ssdEntry, "vSSD-staged demo-100m is listed");
  assert.equal(ssdEntry.status, "available");
  assert.equal(ssdEntry.params, undefined, "params unknown until loaded");
  assert.equal(ssdEntry.weightBytes, undefined);

  // --- listAvailable finds local model dirs (no VM needed for that half). ---
  const localRegistry = new ModelRegistry({
    vm: () => null,
    stageToVram: async () => {
      throw new Error("no VM");
    },
    isActive: () => false,
    modelsDir: demoModelsDir,
  });
  const localAvailable = await localRegistry.listAvailable();
  const localIds = localAvailable.map((e) => e.id).sort();
  assert.deepEqual(localIds, ["demo-100m", "demo-10m"]);
  assert.ok(localAvailable.every((e) => e.status === "available" && e.params === undefined));

  // --- Loading with no VM booted throws a clear error. ---
  await assert.rejects(() => localRegistry.load(demo10mDir), /virtual machine is not booted/);

  console.log("model registry test: all checks passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
