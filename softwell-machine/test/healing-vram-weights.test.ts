/**
 * Failure semantics for vRAM-sourced weights (test/healing-vram-weights.test.ts).
 *
 * Workstream 4 of the "weights from vRAM" phase. These tests run
 * in-process against the real Hypervisor + VirtualRam + VirtualSsd +
 * ModelRegistry over real HTTP donor node servers (the same pieces the
 * demo and the CLI wire together), and answer one question: when a donor
 * dies, do staged model weights heal the same way they do today — by
 * re-replication from the surviving vRAM/vSSD copy — with no silent
 * re-read from the local model directory on disk?
 *
 * Coverage:
 *  1. Targeted kill of the donor holding the PRIMARY replica of weight
 *     pages: healing re-replicates from the surviving copy, all weight
 *     bytes read back byte-identical, and a later-added donor can be
 *     re-replicated onto (replicate()) so a second donor loss still
 *     leaves the weights readable.
 *  2. Hypervisor.sweep() (the timer path) heals a dead donor; the
 *     registry entry stays "ready" and the weight endpoint keeps
 *     serving bytes.
 *  3. Donor death DURING staging (mid stageToVram): staging still
 *     completes with correct bytes and no pages lost.
 *  4. Failed staging frees its vRAM allocation (no leak). Regression
 *     test for a real bug found by this file: Hypervisor.stageToVram
 *     used to leak the allocation when a write failed mid-stream.
 *  5. Permanent page loss (replicas=1): the registry leaves the model
 *     "ready" — it does NOT mark it error or re-stage from vSSD.
 *     Unload + load recovers it. (Known gap, documented, not fixed
 *     here: the registry has no hook into handleNodeLoss.)
 *  6. Healing to a single survivor is degraded: with no spare donor to
 *     re-replicate onto, a second donor loss loses the weight pages.
 *  7. The binary weight endpoint's read path (registry.readWeightTensor,
 *     the bytes GET /mesh/weights serves) survives primary-replica loss:
 *     reads fail over to the surviving replica before healing, and keep
 *     working after healing; the orchestrator-side counter
 *     weightBytesServed proves bytes were served from vRAM.
 *
 * Run: node --test test/healing-vram-weights.test.ts
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { VirtualGpu, GpuNode } from "../src/hw/vgpu.ts";
import { VirtualCpu, CpuNode } from "../src/hw/vcpu.ts";
import { VirtualSsd, SsdNode } from "../src/hw/vssd.ts";
import { loadManifest } from "../src/model/loader.ts";
import { ModelRegistry } from "../src/server/registry.ts";
import { VRAM_PAGE_SIZE_BYTES } from "../src/contracts.ts";
import type {
  NodeProvider,
  RouteHandler,
  VirtualMachine,
  VramAllocation,
  WorkerInfo,
} from "../src/contracts.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DEMO_10M_DIR = path.join(ROOT, "demo", "models", "demo-10m");
const DEMO_10M_WEIGHTS = path.join(DEMO_10M_DIR, "model.safetensors");
const MIB = 1024 * 1024;
const PAGE = VRAM_PAGE_SIZE_BYTES; // 4 MiB

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Deterministic pattern bytes (same helper shape as hw-vram.test.ts). */
function pattern(sizeBytes: number): Uint8Array {
  const buf = new Uint8Array(sizeBytes);
  for (let i = 0; i < sizeBytes; i++) buf[i] = (i * 31 + 7) % 256;
  return buf;
}

/**
 * Byte-compare two buffers. NOTE: assert.deepEqual on a multi-MB Buffer
 * enumerates every byte index as an object key and OOMs the test process
 * (observed: deepEqual on the 38 MB weights blows the 2 GiB heap in ~30 s);
 * Buffer.compare is a memcmp and is the only sane comparison for bulk bytes.
 */
function assertBytesEqual(actual: Uint8Array, expected: Uint8Array, message: string): void {
  assert.equal(
    actual.length,
    expected.length,
    `${message} (length ${actual.length} vs ${expected.length})`,
  );
  assert.equal(
    Buffer.compare(Buffer.from(actual), Buffer.from(expected)),
    0,
    message,
  );
}

function makeWorker(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "test",
    memoryBytes: 8 * 1024 * 1024 * 1024,
    freeMemoryBytes: 8 * 1024 * 1024 * 1024,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    cores: 4,
    roles: ["mixed"],
    maxBatchTokens: 256,
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeNodes implements NodeProvider {
  list: WorkerInfo[];
  constructor(list: WorkerInfo[]) {
    this.list = list;
  }
  listNodes(): WorkerInfo[] {
    return [...this.list];
  }
  markDead(workerId: string): void {
    const node = this.list.find((n) => n.workerId === workerId);
    if (node !== undefined) node.state = "dead";
  }
  add(node: WorkerInfo): void {
    this.list.push(node);
  }
}

type DonorServer = Awaited<ReturnType<typeof createHwNodeServer>>;

interface Donor {
  id: string;
  server: DonorServer;
  info: WorkerInfo;
}

interface Harness {
  tmp: string;
  provider: FakeNodes;
  vm: VirtualMachine;
  ram: VirtualRam;
  hypervisor: Hypervisor;
  registry: ModelRegistry;
  donors: Donor[];
  /** Close a donor's server and mark it dead in the provider. */
  killDonor(id: string): Promise<void>;
  /** Add a fresh healthy donor (for re-replication tests). */
  addDonor(id: string): Promise<Donor>;
  close(): Promise<void>;
}

/** Wrap a donor's store routes with a fixed delay (a slow donor widens
 *  the window for a deterministic mid-staging kill). */
function slowStores(
  routes: Record<string, RouteHandler>,
  delayMs: number,
): Record<string, RouteHandler> {
  const out: Record<string, RouteHandler> = { ...routes };
  for (const key of [
    "POST /hw/ram/store",
    "POST /hw/ram/store-bin",
    "POST /hw/ram/store-many-bin",
  ]) {
    const handler = out[key];
    if (handler !== undefined) {
      out[key] = async (body) => {
        await sleep(delayMs);
        return handler(body);
      };
    }
  }
  return out;
}

async function bootHarness(opts: {
  replicas: number;
  /** Delay (ms) on node-a's store routes; 0 = none. */
  slowDonorMs?: number;
  /** Delay (ms) on BOTH donors' store routes; 0 = none. */
  slowAllMs?: number;
}): Promise<Harness> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "healing-vram-"));
  const provider = new FakeNodes([]);
  const donors: Donor[] = [];

  const makeDonor = async (id: string): Promise<Donor> => {
    const ramNode = new RamNode({
      capacityBytes: 512 * MIB,
      spillDir: path.join(tmp, `${id}-spill`),
    });
    let routes = ramNode.routes();
    const delay =
      opts.slowAllMs ?? (id === "node-a" ? (opts.slowDonorMs ?? 0) : 0);
    if (delay > 0) routes = slowStores(routes, delay);
    const server = await createHwNodeServer({
      routeTables: [
        routes,
        new GpuNode({ device: "cpu" }).routes(),
        new CpuNode({}).routes(),
        new SsdNode({ dir: path.join(tmp, `${id}-ssd`) }).routes(),
      ],
    });
    const info = makeWorker(id, server.port);
    provider.add(info);
    const donor: Donor = { id, server, info };
    donors.push(donor);
    return donor;
  };

  await makeDonor("node-a");
  await makeDonor("node-b");

  const ram = new VirtualRam({ nodes: provider, defaultReplicas: opts.replicas });
  const gpu = new VirtualGpu({ nodes: provider, ram });
  const cpu = new VirtualCpu({ nodes: provider });
  const ssd = new VirtualSsd({
    nodes: provider,
    manifestPath: path.join(tmp, "vssd-manifest.json"),
    ram,
  });
  const hypervisor = new Hypervisor({
    nodes: provider,
    ram,
    gpu,
    cpu,
    ssd,
    // The timer path is exercised explicitly via sweep(); keep the
    // background timer out of the way.
    sweepIntervalMs: 60_000,
  });
  const vm = await hypervisor.boot();
  const registry = new ModelRegistry({
    vm: () => vm,
    stageToVram: (key: string) => hypervisor.stageToVram(key),
    readVram: (vaddr: number, length: number) => vm.ram.read(vaddr, length),
    isActive: () => false,
    modelsDir: tmp,
  });

  return {
    tmp,
    provider,
    vm,
    ram,
    hypervisor,
    registry,
    donors,
    async killDonor(id: string): Promise<void> {
      const donor = donors.find((d) => d.id === id);
      if (donor === undefined) throw new Error(`no such donor: ${id}`);
      await donor.server.close().catch(() => undefined);
      provider.markDead(id);
    },
    async addDonor(id: string): Promise<Donor> {
      return makeDonor(id);
    },
    async close(): Promise<void> {
      await hypervisor.shutdown().catch(() => undefined);
      for (const donor of donors) {
        await donor.server.close().catch(() => undefined);
      }
      fs.rmSync(tmp, { recursive: true, force: true });
    },
  };
}

/** Group an allocation's page refs by pageId, preserving placement
 *  order: the first ref for a page is its primary (copies[0]). */
function primariesByPage(alloc: VramAllocation): Map<string, string> {
  const out = new Map<string, string>();
  for (const ref of alloc.pages) {
    if (!out.has(ref.pageId)) out.set(ref.pageId, ref.nodeId);
  }
  return out;
}

/** Raw bytes of one tensor straight from the safetensors file on disk
 *  (the ground truth the vRAM reads are compared against). */
function tensorRawBytes(dir: string, tensorName: string): Buffer {
  const manifest = loadManifest(dir, path.basename(dir));
  const meta = manifest.tensors[tensorName];
  if (meta === undefined) throw new Error(`no such tensor: ${tensorName}`);
  const file = fs.readFileSync(path.join(dir, "model.safetensors"));
  const headerLen = Number(file.readBigUInt64LE(0));
  const start = 8 + headerLen + meta.dataOffsets[0];
  const end = 8 + headerLen + meta.dataOffsets[1];
  return Buffer.from(file.subarray(start, end));
}

async function donorRamPages(donor: Donor): Promise<number> {
  const stats = (await (
    await fetch(`http://127.0.0.1:${donor.server.port}/hw/ram/stats`)
  ).json()) as { pages: number };
  return stats.pages;
}

/** Stage the demo-10m weights through the real tiers (vSSD -> vRAM)
 *  and return the allocation (with its page refs for targeting). */
async function stageDemoWeights(h: Harness): Promise<VramAllocation> {
  const weights = fs.readFileSync(DEMO_10M_WEIGHTS);
  await h.vm.ssd.putObject("models/demo-10m/model.safetensors", weights);
  return h.hypervisor.stageToVram("models/demo-10m/model.safetensors");
}

test(
  "weight pages survive loss of the primary replica holder (targeted kill)",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 2 });
    try {
      const weights = fs.readFileSync(DEMO_10M_WEIGHTS);
      const alloc = await stageDemoWeights(h);
      const numPages = Math.ceil(weights.length / PAGE);
      assert.equal(
        new Set(alloc.pages.map((p) => p.pageId)).size,
        numPages,
        "one page per 4 MiB of weights",
      );

      // Target the donor holding the most PRIMARY replicas.
      const primaryCounts = new Map<string, number>();
      for (const nodeId of primariesByPage(alloc).values()) {
        primaryCounts.set(nodeId, (primaryCounts.get(nodeId) ?? 0) + 1);
      }
      const victim = [...primaryCounts.entries()].sort((a, b) => b[1] - a[1])[0] as [string, number];
      assert.ok(victim[1] > 0, "victim must hold at least one primary");
      const survivor = victim[0] === "node-a" ? "node-b" : "node-a";
      console.log(
        `  victim ${victim[0]} holds ${victim[1]}/${numPages} primaries; survivor ${survivor}`,
      );

      await h.killDonor(victim[0]);
      const healed = await h.hypervisor.handleNodeLoss(victim[0]);
      assert.deepEqual(healed.errors, [], "no pool may fail healing");
      assert.equal(healed.ram.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      // Recovered pages are the staged weight pages PLUS the vSSD tier-2
      // mirror: staging fetched the object through vSSD, which mirrored
      // its single 64 MiB chunk (the whole object fits in one chunk) into
      // its own vRAM allocation of ceil(weights.length / PAGE) pages.
      // Both allocations lose one copy when a donor dies, so both count.
      const mirrorPages = Math.ceil(weights.length / PAGE);
      assert.equal(
        healed.ram.pagesRecovered,
        numPages + mirrorPages,
        `expected ${numPages} weight pages + ${mirrorPages} vSSD mirror pages recovered, got ${JSON.stringify(healed)}`,
      );

      // Every weight byte still reads from vRAM, byte-identical to the
      // safetensors file — served from the surviving replica.
      const after = Buffer.from(await h.vm.ram.read(alloc.vaddr, weights.length));
      assertBytesEqual(after, weights, "weight bytes after healing must match the file");

      // Redundancy is NOT restored while only one donor is left: the
      // survivor already holds the only remaining copy of each page, so
      // there is nowhere to re-replicate onto. A newly joined donor
      // restores it via replicate().
      const donorC = await h.addDonor("node-c");
      await h.vm.ram.replicate(alloc.id, "node-c");
      assert.equal(
        await donorRamPages(donorC),
        numPages,
        "joined donor must hold a full replica after replicate()",
      );

      // Now losing the original survivor still leaves the weights
      // readable from the replica on node-c.
      await h.killDonor(survivor);
      const healed2 = await h.hypervisor.handleNodeLoss(survivor);
      assert.deepEqual(healed2.errors, [], "no pool may fail healing");
      // The weight pages survive via the node-c replica (pagesRecovered),
      // but the vSSD tier-2 mirror pages had their only copy on the
      // survivor, so they are lost — harmlessly: chunks re-fetch from
      // donor disk and the mirror rebuilds on next read.
      assert.equal(
        healed2.ram.pagesLost,
        Math.ceil(weights.length / PAGE),
        `second healing must lose only the ${Math.ceil(weights.length / PAGE)} vSSD mirror pages: ${JSON.stringify(healed2)}`,
      );
      const after2 = Buffer.from(await h.vm.ram.read(alloc.vaddr, weights.length));
      assertBytesEqual(after2, weights, "weight bytes after second loss must match the file");
      console.log("  weight pages survived two targeted donor losses via replicas");
    } finally {
      await h.close();
    }
  },
);

test(
  "hypervisor sweep heals a dead donor; registry stays ready, endpoint keeps serving",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 2 });
    try {
      const entry = await h.registry.load(DEMO_10M_DIR);
      assert.equal(entry.status, "ready", `registry load: ${entry.error ?? ""}`);
      const manifest = loadManifest(DEMO_10M_DIR, "demo-10m");
      const tensorNames = Object.keys(manifest.tensors);
      assert.ok(tensorNames.length > 0, "demo model has tensors");

      await h.killDonor("node-b");
      // The timer path: sweep() notices node-b is dead and heals it.
      await h.hypervisor.sweep();

      const after = await h.registry.get("demo-10m");
      assert.equal(after?.status, "ready", "registry must still report the model ready");
      for (const name of tensorNames) {
        const served = await h.registry.readWeightTensor("demo-10m", name);
        assertBytesEqual(
          served,
          tensorRawBytes(DEMO_10M_DIR, name),
          `tensor ${name} must still be served byte-identical after sweep healing`,
        );
      }
      assert.ok(
        h.registry.weightBytesServed > 0,
        "weightBytesServed must record bytes served from vRAM",
      );
      console.log(
        `  sweep healed node-b; ${tensorNames.length} tensors re-read byte-identical, ` +
          `${h.registry.weightBytesServed} weight bytes served from vRAM`,
      );
    } finally {
      await h.close();
    }
  },
);

test(
  "donor death during staging still completes with correct bytes",
  { timeout: 120_000 },
  async () => {
    // node-a is slow: its store batches take 100 ms each, so staging a
    // 128 MiB object (32 pages) takes well over a second and a kill at
    // 500 ms lands mid-stage (during alloc or during streaming writes).
    const h = await bootHarness({ replicas: 2, slowDonorMs: 100 });
    try {
      const data = pattern(128 * MIB);
      await h.vm.ssd.putObject("big/object.bin", data);
      let resolved = false;
      const staging = h.hypervisor
        .stageToVram("big/object.bin")
        .then((alloc) => {
          resolved = true;
          return alloc;
        });
      await sleep(500);
      assert.equal(resolved, false, "staging must still be in flight at kill time");
      await h.killDonor("node-a");
      const alloc = await staging;
      const readBack = Buffer.from(await h.vm.ram.read(alloc.vaddr, data.length));
      assertBytesEqual(readBack, data, "staged bytes must match after mid-stage kill");
      const healed = await h.hypervisor.handleNodeLoss("node-a");
      assert.deepEqual(healed.errors, [], "no pool may fail healing");
      assert.equal(healed.ram.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      const afterHeal = Buffer.from(await h.vm.ram.read(alloc.vaddr, data.length));
      assertBytesEqual(afterHeal, data, "bytes must survive healing too");
      console.log("  staging survived a mid-flight donor kill; bytes intact, 0 pages lost");
    } finally {
      await h.close();
    }
  },
);

test(
  "failed staging frees its vRAM allocation (no leak)",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 2, slowAllMs: 100 });
    try {
      const data = pattern(128 * MIB);
      await h.vm.ssd.putObject("big/object.bin", data);
      const staging = h.hypervisor.stageToVram("big/object.bin");
      // Wait until the allocation has placed pages, then kill BOTH
      // donors so every remaining write fails on all copies.
      const deadline = Date.now() + 10_000;
      for (;;) {
        if (h.vm.ram.stats().pages > 0) break;
        if (Date.now() > deadline) throw new Error("allocation never placed pages");
        await sleep(25);
      }
      const placedBefore = h.vm.ram.stats().pages;
      assert.ok(placedBefore > 0, "pages must be placed before the kill");
      await h.killDonor("node-a");
      await h.killDonor("node-b");
      // The kill can land during the alloc, the vSSD chunk read, or a
      // vRAM write, so the rejection is whichever of those fails first
      // — the point of this test is the allocation being freed, not
      // which layer reports the loss.
      await assert.rejects(
        staging,
        /failed on all copies|no healthy donor|could not place page|unavailable on all replicas/,
      );
      assert.equal(
        h.vm.ram.stats().pages,
        0,
        `failed staging must free its allocation (leaked ${h.vm.ram.stats().pages} pages)`,
      );
      console.log(`  failed staging freed its ${placedBefore}-page allocation`);
    } finally {
      await h.close();
    }
  },
);

test(
  "permanent page loss leaves the model 'ready' (registry gap); unload+load recovers",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 1 });
    try {
      const weights = fs.readFileSync(DEMO_10M_WEIGHTS);
      const entry = await h.registry.load(DEMO_10M_DIR);
      assert.equal(entry.status, "ready", `registry load: ${entry.error ?? ""}`);
      const vaddr = entry.weightVaddr as number;
      const weightBytes = entry.weightBytes as number;

      await h.killDonor("node-a");
      const healed = await h.hypervisor.handleNodeLoss("node-a");
      assert.deepEqual(healed.errors, [], "no pool may fail healing");
      assert.ok(
        healed.ram.pagesLost > 0,
        `replicas=1 must lose pages when their only donor dies: ${JSON.stringify(healed)}`,
      );
      console.log(`  pagesLost=${healed.ram.pagesLost} (replicas=1, donor node-a dead)`);

      // The gap: nothing marks the model error and nothing re-stages
      // it from vSSD. The entry still claims "ready" while its weights
      // are unreadable — a worker asking the weight endpoint for these
      // bytes would fail.
      const after = h.registry.get("demo-10m");
      assert.equal(
        after?.status,
        "ready",
        "registry leaves a permanently-damaged model 'ready' (known gap)",
      );
      await assert.rejects(
        h.vm.ram.read(vaddr, weightBytes),
        /unavailable/,
        "weight bytes must be unreadable after permanent page loss",
      );
      const firstTensor = Object.keys(loadManifest(DEMO_10M_DIR, "demo-10m").tensors)[0] as string;
      await assert.rejects(
        h.registry.readWeightTensor("demo-10m", firstTensor),
        /unavailable/,
        "weight endpoint must fail for the lost tensor",
      );

      // Manual recovery works: unload frees the broken allocations,
      // load re-stages from vSSD and the weights read back correctly.
      await h.registry.unload("demo-10m");
      const reloaded = await h.registry.load(DEMO_10M_DIR);
      assert.equal(reloaded.status, "ready", `reload: ${reloaded.error ?? ""}`);
      const readBack = Buffer.from(
        await h.vm.ram.read(reloaded.weightVaddr as number, weights.length),
      );
      assertBytesEqual(readBack, weights, "re-staged weights must match the file");
      console.log("  manual unload+load recovered the model from vSSD");
    } finally {
      await h.close();
    }
  },
);

test(
  "healing to a single survivor is degraded: a second loss loses weight pages",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 2 });
    try {
      const weights = fs.readFileSync(DEMO_10M_WEIGHTS);
      const alloc = await stageDemoWeights(h);
      const numPages = Math.ceil(weights.length / PAGE);

      await h.killDonor("node-a");
      const healed = await h.hypervisor.handleNodeLoss("node-a");
      assert.equal(healed.ram.pagesLost, 0);
      assertBytesEqual(
        await h.vm.ram.read(alloc.vaddr, weights.length),
        weights,
        "weights must survive the first donor loss",
      );

      // No spare donor exists to re-replicate onto, so the survivor now
      // holds the ONLY copy of every page. Losing it loses everything.
      await h.killDonor("node-b");
      const healed2 = await h.hypervisor.handleNodeLoss("node-b");
      assert.deepEqual(healed2.errors, [], "no pool may fail healing");
      // The second loss takes the last copy of every page: the staged
      // weight pages (numPages) plus the vSSD tier-2 mirror pages
      // (one 64 MiB chunk holding the whole object = numPages pages).
      assert.equal(
        healed2.ram.pagesLost,
        numPages + Math.ceil(weights.length / PAGE),
        `second loss must lose all weight + mirror pages: ${JSON.stringify(healed2)}`,
      );
      await assert.rejects(
        h.vm.ram.read(alloc.vaddr, weights.length),
        /unavailable/,
        "weights must be unreadable after the second donor loss",
      );
      console.log(
        "  single-survivor state confirmed degraded: redundancy returns only when a new donor joins (see test 1)",
      );
    } finally {
      await h.close();
    }
  },
);

test(
  "weight endpoint reads fail over to the surviving replica before and after healing",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness({ replicas: 2 });
    try {
      const entry = await h.registry.load(DEMO_10M_DIR);
      assert.equal(entry.status, "ready", `registry load: ${entry.error ?? ""}`);
      const manifest = loadManifest(DEMO_10M_DIR, "demo-10m");
      const tensorNames = Object.keys(manifest.tensors);

      // node-a is primary for roughly half the pages (round-robin
      // striping). Kill it WITHOUT healing first: endpoint reads must
      // fail over to the surviving replica on their own.
      await h.killDonor("node-a");
      for (const name of tensorNames) {
        const served = await h.registry.readWeightTensor("demo-10m", name);
        assertBytesEqual(
          served,
          tensorRawBytes(DEMO_10M_DIR, name),
          `tensor ${name} must be served from the surviving replica before healing`,
        );
      }
      const servedBeforeHeal = h.registry.weightBytesServed;

      await h.hypervisor.handleNodeLoss("node-a");
      for (const name of tensorNames) {
        const served = await h.registry.readWeightTensor("demo-10m", name);
        assertBytesEqual(
          served,
          tensorRawBytes(DEMO_10M_DIR, name),
          `tensor ${name} must still be served after healing`,
        );
      }
      assert.ok(
        h.registry.weightBytesServed > servedBeforeHeal,
        "weightBytesServed must keep growing after healing",
      );
      console.log(
        `  ${tensorNames.length} tensors served from the surviving replica pre- and post-heal; ` +
          `${h.registry.weightBytesServed} total weight bytes served from vRAM`,
      );
    } finally {
      await h.close();
    }
  },
);
