/**
 * Diagnostic scratch script (NOT part of the test suite): replicate the
 * live proof test's worker-restart sequence in-process with demo-10m
 * (40 MB, ~10 pages) to find out why a fresh worker's vRAM stage load
 * failed live with "page .../p77 unavailable on all 2 copies".
 *
 * Live sequence, in miniature:
 *   1. boot 2 donors, registry.load(demo-10m) — stages weights to vRAM;
 *   2. read every tensor via readWeightTensor (baseline, byte-identical);
 *   3. add a THIRD donor (the live replacement worker joining as a donor);
 *   4. kill a donor that holds primaries;
 *   5. hypervisor.sweep() (the timer path, as live);
 *   6. read every tensor again — byte-identical?
 *   7. probe each donor's RamNode directly for every page, and dump the
 *      allocation's page refs, to say exactly which copy is missing where.
 *
 * Usage: node scripts/diag-vram-live-sequence.ts
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
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
const MIB = 1024 * 1024;
const PAGE = VRAM_PAGE_SIZE_BYTES;

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

/** Does this donor's RamNode hold this page? (direct probe, no failover). */
async function donorHasPage(donor: Donor, pageId: string): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${donor.server.port}/hw/ram/fetch`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ pageId }),
    });
    if (!res.ok) return false;
    const body = (await res.json()) as { dataB64?: string };
    return typeof body.dataB64 === "string";
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "diag-vram-live-"));
  const provider = new FakeNodes([]);
  const donors: Donor[] = [];

  const makeDonor = async (id: string): Promise<Donor> => {
    const ramNode = new RamNode({
      capacityBytes: 512 * MIB,
      spillDir: path.join(tmp, `${id}-spill`),
    });
    const routes: Record<string, RouteHandler> = ramNode.routes();
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

  const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });
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
    sweepIntervalMs: 60_000,
  });
  const vm: VirtualMachine = await hypervisor.boot();
  const registry = new ModelRegistry({
    vm: () => vm,
    stageToVram: (key: string) => hypervisor.stageToVram(key),
    readVram: (vaddr: number, length: number) => vm.ram.read(vaddr, length),
    isActive: () => false,
    modelsDir: tmp,
  });

  try {
    // 1. Load demo-10m through the registry (vSSD -> vRAM staging).
    const entry = await registry.load(DEMO_10M_DIR);
    console.log(`registry load: status=${entry.status} vramBytes=${entry.vramBytes}`);
    if (entry.status !== "ready") throw new Error(`load failed: ${entry.error ?? ""}`);
    const manifest = loadManifest(DEMO_10M_DIR, "demo-10m");
    const tensorNames = Object.keys(manifest.tensors);
    const weightMap = registry.getWeightMap("demo-10m");
    if (weightMap === undefined) throw new Error("no weight map after load");
    const allocId = Object.values(weightMap)[0]?.vaddr !== undefined
      ? null
      : null;
    void allocId;

    // 2. Baseline: every tensor reads byte-identical from vRAM.
    for (const name of tensorNames) {
      const served = await registry.readWeightTensor("demo-10m", name);
      const truth = tensorRawBytes(DEMO_10M_DIR, name);
      if (!Buffer.from(served).equals(truth)) {
        throw new Error(`baseline tensor ${name} mismatch`);
      }
    }
    console.log(`baseline: ${tensorNames.length} tensors byte-identical from vRAM`);

    // Page refs for the model allocation: find it via the weight map's
    // vaddr range. The registry does not expose the alloc id, so recover
    // page placement by probing donors for each page id of the form
    // <allocId>/p<i>. Instead, list pages from the ram stats + probe the
    // first tensor's pages. Simpler: use vm stats + direct probes below
    // after the kill, keyed by page ids seen in the weight map.
    const pageIds = new Set<string>();
    for (const loc of Object.values(weightMap)) {
      const start = loc.vaddr + loc.headerLen + loc.dataOffsets[0];
      const end = loc.vaddr + loc.headerLen + loc.dataOffsets[1];
      // Page ids are <allocId>/p<i>; recover allocId from vaddr base:
      // the model's allocation starts at weightVaddr (entry.weightVaddr).
      const base = entry.weightVaddr ?? 0;
      for (let p = Math.floor((start - base) / PAGE); p < Math.ceil((end - base) / PAGE); p++) {
        pageIds.add(`p${p}`);
      }
    }
    console.log(`model spans pages: ${[...pageIds].join(", ")}`);

    // 3. Third donor joins (the live replacement worker).
    const donorC = await makeDonor("node-c");
    console.log(`donor node-c joined (port ${donorC.server.port})`);

    // 4. Kill node-a (a primary holder for ~half the pages).
    const victim = donors.find((d) => d.id === "node-a") as Donor;
    await victim.server.close().catch(() => undefined);
    provider.markDead("node-a");
    console.log("killed node-a");

    // 5. Timer path: sweep() heals (the live path; heals each lost node once).
    await hypervisor.sweep();
    console.log("sweep() done");

    // 6. Read every tensor again.
    let failed = 0;
    for (const name of tensorNames) {
      try {
        const served = await registry.readWeightTensor("demo-10m", name);
        const truth = tensorRawBytes(DEMO_10M_DIR, name);
        if (!Buffer.from(served).equals(truth)) {
          console.log(`MISMATCH ${name}: bytes differ`);
          failed += 1;
        }
      } catch (err) {
        console.log(`FAILED ${name}: ${err instanceof Error ? err.message : String(err)}`);
        failed += 1;
      }
    }
    console.log(`after kill+sweep: ${tensorNames.length - failed}/${tensorNames.length} tensors OK`);

    // 7. Direct per-donor page probes. Page ids need the alloc id; get it
    // from the registry status (weightVaddr is 0; the alloc id is not
    // exposed, so probe via the ram stats endpoint page list instead).
    // Fallback: probe with the page ids from the failing tensor's range
    // is not possible without the alloc id — instead list each donor's
    // pages via /hw/ram/stats and the orchestrator's page table via
    // vm.ram.stats (counts only). Report donor page counts.
    for (const donor of donors) {
      try {
        const stats = (await (
          await fetch(`http://127.0.0.1:${donor.server.port}/hw/ram/stats`)
        ).json()) as { pages: number; bytesUsed: number; spilled: number };
        console.log(`donor ${donor.id}: pages=${stats.pages} bytesUsed=${stats.bytesUsed} spilled=${stats.spilled}`);
      } catch (err) {
        console.log(`donor ${donor.id}: stats unreachable (${err instanceof Error ? err.message : String(err)})`);
      }
    }
    console.log(`vm ram stats: ${JSON.stringify(vm.ram.stats())}`);
    console.log(`weightBytesServed=${registry.weightBytesServed}`);
  } finally {
    await hypervisor.shutdown().catch(() => undefined);
    for (const donor of donors) {
      await donor.server.close().catch(() => undefined);
    }
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
