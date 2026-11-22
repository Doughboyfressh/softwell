/**
 * bench-vram.ts — vRAM benchmark: pooled sequential read/write bandwidth,
 * pooled capacity, and single-page read latency, vs HBM3e (B200).
 *
 * Method (all measured on this machine, nothing hardcoded):
 *  - Boot `opts.donors` donor node servers in-process (real HTTP on
 *    ephemeral loopback ports, RamNode + createHwNodeServer — the same
 *    pattern as demo/run-demo.ts). Each donor contributes 1 GiB of
 *    pooled vRAM; donors share one physical host, so the numbers are
 *    loopback-HTTP-bound, not HBM-bound — that is the honest limit
 *    this bench reports, and the donorsToBeat column says how many
 *    such donors it would take to beat the target.
 *  - Allocate one large buffer (256 MiB, or 16 MiB for size "small")
 *    with replicas=1 so each page lives on exactly one donor and the
 *    transfer is striped across donors by the coordinator's
 *    round-robin placement. A replicas=2 run would double the store
 *    traffic and is not what bandwidth means here.
 *  - READ: median of N timed full-buffer reads. WRITE: median of N
 *    timed full-buffer writes (full-page writes skip read-modify-write).
 *  - Capacity: stats().totalBytes (sum of donor memory) vs 192 GB.
 *  - Latency: median single-page (4 MiB) read, informational — the
 *    target is an aspirational reference, not a published HBM3e
 *    figure (no vendor publishes a single-page latency for HBM3e).
 *
 * The harness (src/bench/run.ts) calls runBench({ donors: N }) once
 * per N in the scaling curve 1..N, so runBench measures exactly
 * `opts.donors` donors. The standalone main below loops 1..N itself.
 *
 * Usage: node src/bench/bench-vram.ts [--donors N] [--size small|large] [--json <path>]
 * Exit code is 0 even when verdicts are FAIL — a FAIL is a
 * measurement, not a crash.
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { VRAM_PAGE_SIZE_BYTES } from "../contracts.ts";
import type { WorkerInfo } from "../contracts.ts";
import { RamNode, VirtualRam } from "../hw/vram.ts";
import { createHwNodeServer } from "../hw/node-server.ts";
import type { HwNodeServer } from "../hw/node-server.ts";
import { VRAM_TARGETS } from "./targets.ts";
import { formatResultsTable, makeResult, median, timeMs } from "./types.ts";
import type { BenchModule, BenchOptions, BenchResult } from "./types.ts";

const MIB = 1024 * 1024;
const GB = 1e9;

/** One donor's contribution to the pool. 1 GiB: the donors share one
 *  physical host in this bench, so claiming more per donor would
 *  double-count the same physical RAM. */
const DONOR_MEMORY_BYTES = 1 * 1024 * 1024 * 1024;

/** Workload sizes: "small" keeps `npm test` fast; "large" is the default. */
const SIZES = {
  small: { bytes: 16 * MIB, reps: 1, latencySamples: 3 },
  large: { bytes: 256 * MIB, reps: 3, latencySamples: 5 },
} as const;

function makeDonor(id: string, hwPort: number): WorkerInfo {
  return {
    workerId: id,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: DONOR_MEMORY_BYTES,
    freeMemoryBytes: DONOR_MEMORY_BYTES,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 1024,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

/** Boot `count` donor servers; caller must close them. */
async function bootDonors(count: number): Promise<{
  servers: HwNodeServer[];
  infos: WorkerInfo[];
}> {
  const servers: HwNodeServer[] = [];
  const infos: WorkerInfo[] = [];
  for (let i = 0; i < count; i++) {
    // No spillDir: evicted pages would hit disk and distort bandwidth.
    const ramNode = new RamNode({ capacityBytes: DONOR_MEMORY_BYTES });
    const server = await createHwNodeServer({ routeTables: [ramNode.routes()] });
    servers.push(server);
    infos.push(makeDonor(`bench-vram-donor-${i}`, server.port));
  }
  return { servers, infos };
}

function gbps(bytes: number, ms: number): number {
  return bytes / GB / (ms / 1000);
}

export async function runBench(opts: BenchOptions): Promise<BenchResult[]> {
  const donors = Math.max(1, opts.donors);
  const size = SIZES[opts.size === "small" ? "small" : "large"];
  const { servers, infos } = await bootDonors(donors);
  const targetName = VRAM_TARGETS.primary; // "HBM3e (NVIDIA B200)"
  const bwTarget = VRAM_TARGETS.primaryMetrics["memoryBandwidthGBps"] as number;
  const capTarget = VRAM_TARGETS.primaryMetrics["memoryCapacityGB"] as number;
  const notesBase =
    `loopback HTTP, binary page transport, 4 MiB pages, batched <=8 pages/req, concurrency 8, replicas=1, ` +
    `${donors} donor(s) sharing one physical host`;
  try {
    const ram = new VirtualRam({
      nodes: { listNodes: () => [...infos] },
      defaultReplicas: 1,
    });

    // One large allocation, striped across donors by round-robin placement.
    const alloc = await ram.alloc(size.bytes, { replicas: 1 });
    const data = Buffer.alloc(size.bytes, 0xa5);

    // Warmup: one full write + one full read (untimed), then verify a slice.
    await ram.write(alloc.vaddr, data);
    const warm = await ram.read(alloc.vaddr, size.bytes);
    if (warm.length !== size.bytes || warm[0] !== 0xa5 || warm[size.bytes - 1] !== 0xa5) {
      throw new Error("bench-vram: warmup read-back mismatch — refusing to report bandwidth");
    }

    // (b) Pooled sequential WRITE bandwidth, median of reps.
    const writeTimes: number[] = [];
    for (let i = 0; i < size.reps; i++) {
      writeTimes.push(await timeMs(async () => {
        await ram.write(alloc.vaddr, data);
      }));
    }
    const writeMs = median(writeTimes);
    const writeGBps = gbps(size.bytes, writeMs);

    // (a) Pooled sequential READ bandwidth, median of reps.
    const readTimes: number[] = [];
    for (let i = 0; i < size.reps; i++) {
      readTimes.push(await timeMs(async () => {
        await ram.read(alloc.vaddr, size.bytes);
      }));
    }
    const readMs = median(readTimes);
    const readGBps = gbps(size.bytes, readMs);

    // (d) Single-page read latency, median of samples (informational).
    const latencies: number[] = [];
    for (let i = 0; i < size.latencySamples; i++) {
      latencies.push(await timeMs(async () => {
        await ram.read(alloc.vaddr, VRAM_PAGE_SIZE_BYTES);
      }));
    }
    const latencyMs = median(latencies);
    // Built by hand, NOT via makeResult: latency is lower-is-better,
    // and makeResult computes PASS as measured >= target. More donors
    // never make one page fetch faster, so donorsToBeat is infinity.
    const latencyTargetMs = 0.05;
    const latencyRow: BenchResult = {
      device: "vram",
      metric: "single-page (4 MiB) read latency",
      measured: latencyMs,
      unit: "ms",
      // Aspirational reference only — no vendor publishes an HBM3e
      // single-page latency. The honest point is the measured value:
      // one page over loopback HTTP cannot approach HBM latency.
      target: latencyTargetMs,
      targetName: "HBM3e page access (aspirational reference, not a vendor spec)",
      donors,
      perDonor: latencyMs / donors,
      donorsToBeat: Number.POSITIVE_INFINITY,
      verdict: latencyMs <= latencyTargetMs ? "PASS" : "FAIL",
      notes: `informational; median of ${size.latencySamples} single-page read(s) over loopback HTTP; lower is better — PASS only when measured <= target; pooling never makes one op faster than one donor`,
    };

    // (c) Pooled capacity.
    const capacityGB = ram.stats().totalBytes / GB;

    await ram.free(alloc.id);

    return [
      makeResult({
        device: "vram",
        metric: `sequential READ bandwidth (${(size.bytes / MIB).toFixed(0)} MiB)`,
        measured: readGBps,
        unit: "GB/s",
        target: bwTarget,
        targetName,
        donors,
        notes: `${notesBase}; median of ${size.reps} timed read(s)`,
      }),
      makeResult({
        device: "vram",
        metric: `sequential WRITE bandwidth (${(size.bytes / MIB).toFixed(0)} MiB)`,
        measured: writeGBps,
        unit: "GB/s",
        target: bwTarget,
        targetName,
        donors,
        notes: `${notesBase}; median of ${size.reps} timed write(s); full-page writes skip read-modify-write`,
      }),
      makeResult({
        device: "vram",
        metric: "pooled capacity",
        measured: capacityGB,
        unit: "GB",
        target: capTarget,
        targetName,
        donors,
        notes: `${donors} donor(s) x 1 GiB each (stats().totalBytes); donors share one host`,
      }),
      latencyRow,
    ];
  } finally {
    for (const s of servers) await s.close().catch(() => undefined);
  }
}

export const benchVram: BenchModule = { name: "vram", run: runBench };

function usage(): string {
  return [
    "Usage: node src/bench/bench-vram.ts [--donors N] [--size small|large] [--json <path>]",
    "  --donors N   scaling curve 1..N donors (default 4)",
    "  --size S     small (16 MiB) or large (256 MiB, default)",
    "  --json PATH  also write the raw results JSON to PATH",
  ].join("\n");
}

async function main(): Promise<void> {
  let maxDonors = 4;
  let size: "small" | "large" = "large";
  let jsonOut: string | undefined;
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    const [flag, inlineValue] = arg.split("=", 2);
    const value = (name: string): string => {
      if (inlineValue !== undefined) return inlineValue;
      const next = argv[++i];
      if (next === undefined) throw new Error(`missing value for ${name}`);
      return next;
    };
    if (flag === "--help" || flag === "-h") {
      console.log(usage());
      process.exit(0);
    } else if (flag === "--donors") {
      const raw = value("--donors");
      const n = Number(raw);
      if (!Number.isInteger(n) || n < 1) throw new Error(`--donors must be a positive integer, got ${raw}`);
      maxDonors = n;
    } else if (flag === "--size") {
      const s = value("--size");
      if (s !== "small" && s !== "large") throw new Error(`--size must be small|large, got ${s}`);
      size = s;
    } else if (flag === "--json") {
      jsonOut = value("--json");
    } else {
      throw new Error(`unknown argument: ${arg}\n${usage()}`);
    }
  }

  const results: BenchResult[] = [];
  for (let donors = 1; donors <= maxDonors; donors++) {
    results.push(...(await runBench({ donors, size })));
  }
  console.log(formatResultsTable(results));
  console.log("");
  console.log(
    "Honest limits: donors are in-process servers sharing one host (one Node event loop, " +
      "1.5 shared CPUs here), so aggregate bandwidth is capped by that host's CPU, not by " +
      "placement — striping is verified working (see test/hw-vram.test.ts), but adding " +
      "donors on the same host adds no parallelism. Real scaling needs donors on separate " +
      "hosts. Pooling adds aggregate throughput/capacity; it never makes a single page " +
      "fetch faster than one donor.",
  );
  if (jsonOut !== undefined) {
    const outPath = resolve(jsonOut);
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, `${JSON.stringify(results, null, 2)}\n`, "utf8");
    console.log(`\nRaw results written to ${outPath}`);
  }
}

// Run main() only when executed directly, not when imported.
const isMain =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main().catch((err: unknown) => {
    console.error(`bench-vram error: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
