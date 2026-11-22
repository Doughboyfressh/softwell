/**
 * bench-vcpu.ts — vCPU benchmark: pooled virtual cores, measured FP64
 * TFLOPS from real "flops" tasks, and flops-task throughput, vs the
 * AMD EPYC 9965 (Turin).
 *
 * Method (all measured on this machine, nothing hardcoded):
 *  - Boot `opts.donors` donor node servers in-process (real HTTP on
 *    ephemeral loopback ports, CpuNode + createHwNodeServer — the
 *    same pattern as demo/run-demo.ts). Each donor reports its real
 *    core count (`capability.cores` = this host's os.cpus().length),
 *    so stats().totalCores is a sum of reported counts, not an
 *    assumption. The donors share one physical host, so the summed
 *    cores are pooled reported cores, not distinct silicon — and the
 *    TFLOPS do not scale with donor count for the same reason: every
 *    donor's FMA loop runs on the same host cores. That is the honest
 *    limit this bench reports, and the donorsToBeat column says how
 *    many such donors it would take to beat the target.
 *  - TFLOPS: N = 2 per donor independent "flops" tasks submitted
 *    concurrently through VirtualCpu and timed wall-clock by
 *    measureFlopsThroughput (src/hw/vcpu.ts). Every task's checksum
 *    is verified against expectedFlopsChecksum(size) before its
 *    FLOPs count — if a checksum fails, the bench throws instead of
 *    reporting a number. Aggregate TFLOPS = total measured FLOPs /
 *    wall time. The loop is FP64 FMA over Float64Array; the target
 *    is the EPYC FP32 figure, which is an ESTIMATE (see targets.ts),
 *    not a vendor number, and is labelled as such in every notes
 *    field. Measured FP64 vs an FP32 estimate is a like-for-scale
 *    comparison only: FP64 is the harder of the two to sustain.
 *  - Throughput: completed flops tasks per wall-clock second from
 *    the same batch, informational (target 0 — no enterprise
 *    tasks/s figure exists for the EPYC; its verdict is not
 *    meaningful and the notes say so).
 *
 * The harness (src/bench/run.ts) calls runBench({ donors: N }) once
 * per N in the scaling curve 1..N, so runBench measures exactly
 * `opts.donors` donors. The standalone main below loops 1..N itself
 * to print the scaling table.
 *
 * Usage: node src/bench/bench-vcpu.ts [--donors N] [--size small|large] [--json <path>]
 * Exit code is 0 even when verdicts are FAIL — a FAIL is a
 * measurement, not a crash.
 */
import { cpus } from "node:os";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkerInfo } from "../contracts.ts";
import {
  ASSUMED_CORES_PER_NODE,
  CpuNode,
  VirtualCpu,
  measureFlopsThroughput,
} from "../hw/vcpu.ts";
import { createHwNodeServer } from "../hw/node-server.ts";
import type { HwNodeServer } from "../hw/node-server.ts";
import { VCPU_TARGETS } from "./targets.ts";
import { formatResultsTable, makeResult } from "./types.ts";
import type { BenchModule, BenchOptions, BenchResult } from "./types.ts";

/** Workload sizes: "small" keeps `npm test` fast; "large" is the default. */
const SIZES = {
  small: { size: 10_000_000, tasksPerDonor: 2 },
  large: { size: 100_000_000, tasksPerDonor: 2 },
} as const;

function makeDonor(id: string, hwPort: number, cores: number): WorkerInfo {
  return {
    workerId: id,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    cores,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
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
async function bootDonors(count: number, cores: number): Promise<{
  servers: HwNodeServer[];
  infos: WorkerInfo[];
}> {
  const servers: HwNodeServer[] = [];
  const infos: WorkerInfo[] = [];
  for (let i = 0; i < count; i++) {
    const cpuNode = new CpuNode({});
    const server = await createHwNodeServer({ routeTables: [cpuNode.routes()] });
    servers.push(server);
    infos.push(makeDonor(`bench-vcpu-donor-${i}`, server.port, cores));
  }
  return { servers, infos };
}

export async function runBench(opts: BenchOptions): Promise<BenchResult[]> {
  const donors = Math.max(1, opts.donors);
  const size = SIZES[opts.size === "small" ? "small" : "large"];
  const hostCores = cpus().length;
  const { servers, infos } = await bootDonors(donors, hostCores);
  const cpu = new VirtualCpu({ nodes: { listNodes: () => [...infos] } });
  try {
    const stats = cpu.stats();
    const targetName = VCPU_TARGETS.primary; // "AMD EPYC 9965 (Turin)"
    const coresTarget = VCPU_TARGETS.primaryMetrics["cores"] as number;
    const fp32Target = VCPU_TARGETS.primaryMetrics[
      "fp32TflopsEstimate"
    ] as number;
    const coresNotes =
      `${stats.nodesReportingCores}/${donors} donor(s) reported real core counts ` +
      `(${hostCores} each); ${stats.nodesAssumedCores} fell back to the assumed ` +
      `${ASSUMED_CORES_PER_NODE}. In-process donors share one ${hostCores}-core ` +
      `host, so summed cores are pooled reported cores, not distinct silicon.`;

    const throughput = await measureFlopsThroughput(cpu, {
      taskCount: size.tasksPerDonor * donors,
      size: size.size,
      threads: 1,
    });
    if (!throughput.checksumOk || throughput.completed !== throughput.tasks) {
      throw new Error(
        `bench-vcpu: ${throughput.completed}/${throughput.tasks} flops tasks completed ` +
          `with valid checksums — refusing to report TFLOPS from unverified work`,
      );
    }

    const tflopsNotes =
      `measured FP64 FMA over Float64Array (${size.size.toLocaleString("en-US")} multiply-adds per task, ` +
      `${throughput.tasks} tasks, checksums verified); ` +
      `aggregate = total measured FLOPs / wall time (${throughput.elapsedMs.toFixed(0)} ms). ` +
      `Target ${fp32Target} TFLOPS is an ESTIMATE — 192 cores x 2.25 GHz x 32 FP32 FLOPs/cycle ` +
      `(2x AVX-512 FMA) — not a vendor number; measured FP64 is compared against that FP32 ` +
      `estimate for scale only. Donors share one ${hostCores}-core host, so aggregate TFLOPS ` +
      `does not scale with donor count here.`;

    return [
      makeResult({
        device: "vcpu",
        metric: "virtual cores",
        measured: stats.totalCores,
        unit: "cores",
        target: coresTarget,
        targetName,
        donors,
        notes: coresNotes,
      }),
      makeResult({
        device: "vcpu",
        metric: "FP64 TFLOPS (FMA, Float64Array)",
        measured: throughput.aggregateTflops,
        unit: "TFLOPS",
        target: fp32Target,
        targetName: `${targetName} — FP32 estimate`,
        donors,
        notes: tflopsNotes,
      }),
      makeResult({
        device: "vcpu",
        metric: "flops task throughput",
        measured: throughput.tasksPerSec,
        unit: "tasks/s",
        target: 0,
        targetName: "(informational — no enterprise target)",
        donors,
        notes:
          `informational row; verdict not meaningful (target 0). ` +
          `${throughput.completed}/${throughput.tasks} flops tasks ` +
          `(${size.size.toLocaleString("en-US")} multiply-adds each, checksums verified) in ` +
          `${throughput.elapsedMs.toFixed(0)} ms; per-task TFLOPS summed to ` +
          `${throughput.sumPerTaskTflops.toFixed(6)} TFLOPS.`,
      }),
    ];
  } finally {
    cpu.close();
    for (const s of servers) await s.close().catch(() => undefined);
  }
}

export const benchVcpu: BenchModule = { name: "vcpu", run: runBench };

function usage(): string {
  return [
    "Usage: node src/bench/bench-vcpu.ts [--donors N] [--size small|large] [--json <path>]",
    "  --donors N   scaling curve 1..N donors (default 4)",
    "  --size S     small (10M FMA/task) or large (100M FMA/task, default)",
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
    "Honest limits: donors are in-process on one host, so they share its cores — " +
      "summed virtual cores grow with donor count while measured TFLOPS stays flat " +
      "(one host's worth of JS FMA). Pooling adds aggregate throughput/capacity " +
      "across distinct donors; it never makes one FMA loop faster than one donor. " +
      "The 27.6 TFLOPS target is an FP32 estimate, not a vendor number.",
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
    console.error(`bench-vcpu error: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
