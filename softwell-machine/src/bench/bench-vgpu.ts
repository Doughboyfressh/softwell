/**
 * bench-vgpu.ts — vGPU benchmark: REAL timed execution, never asserted.
 *
 * Measures, on this machine, with real timed runs:
 *  (a) FP32 matmul TFLOPS at 1024^3 and 2048^3 (tiled + worker-thread
 *      parallel kernels, in-process on the local donor);
 *  (b) memory bandwidth via a timed copy/add kernel (GB/s);
 *  (c) scaling: 1..N local donor node servers booted in-process
 *      (RamNode/GpuNode/CpuNode/SsdNode + createHwNodeServer, the same
 *      pattern as demo/run-demo.ts), each running a matmul through the
 *      VirtualGpu coordinator over real HTTP, timed concurrently —
 *      aggregate TFLOPS vs donor count.
 *
 * Honesty notes (also printed by main()):
 *  - The measured rate is FP32 on a CPU fallback. It is compared against
 *    the BF16 tensor-core targets as the closest like-for-like line we
 *    can honestly draw — FP32 != FP4/BF16 tensor-core rates, and there
 *    is no FP4 hardware here. The gap is therefore a lower bound on the
 *    real gap, not a like-for-like defeat.
 *  - The scaling donors run in-process and share this machine's physical
 *    cores (and one worker pool), so aggregate TFLOPS in-process is
 *    capped by local cores; real donors on separate machines would each
 *    add their own cores. `donorsToBeat` is computed from per-donor
 *    throughput, which is the figure that transfers to real donors.
 *  - Every row is built with makeResult(): verdicts are computed from
 *    measured vs target, never asserted.
 *
 * Usage: node src/bench/bench-vgpu.ts [--donors N] [--size small|large]
 *        [--json-out <path>]
 */

import { fileURLToPath } from "node:url";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { matmulParallel } from "../hw/kernels-cpu.ts";
import { measureMemoryBandwidthBytesPerSec } from "../hw/calibrate.ts";
import { RamNode, VirtualRam } from "../hw/vram.ts";
import { GpuNode, VirtualGpu } from "../hw/vgpu.ts";
import { CpuNode } from "../hw/vcpu.ts";
import { SsdNode } from "../hw/vssd.ts";
import { createHwNodeServer } from "../hw/node-server.ts";
import type { HwNodeServer } from "../hw/node-server.ts";
import { VGPU_TARGETS } from "./targets.ts";
import {
  formatResultsTable,
  makeResult,
  median,
  type BenchOptions,
  type BenchResult,
} from "./types.ts";
import type { WorkerInfo } from "../contracts.ts";

const FP32_NOTE =
  "FP32 on CPU fallback (no tensor cores, no FP4 hardware here); " +
  "compared vs BF16 target as the closest honest line — FP32 != FP4/BF16 rates";

function randomMatrix(n: number): Float32Array {
  const a = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

function clampDonors(donors: number | undefined): number {
  const n = donors ?? 1;
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.min(4, Math.floor(n)));
}

/** Timed FP32 matmul TFLOPS at size n, in-process on the local donor. */
async function measureMatmulTflops(n: number, timedRuns: number): Promise<number> {
  const a = randomMatrix(n);
  const b = randomMatrix(n);
  await matmulParallel(a, n, n, b, n, n); // warmup: JIT + worker pool
  const flops = 2 * n * n * n;
  const elapsedMs: number[] = [];
  for (let run = 0; run < timedRuns; run++) {
    const start = performance.now();
    await matmulParallel(a, n, n, b, n, n);
    elapsedMs.push(performance.now() - start);
  }
  const med = median(elapsedMs);
  if (!(med > 0)) return 0;
  return flops / (med / 1000) / 1e12;
}

function makeDonorInfo(id: string, hwPort: number): WorkerInfo {
  return {
    workerId: id,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 512 * 1024 * 1024,
    freeMemoryBytes: 512 * 1024 * 1024,
    flopsPerSec: 0, // unknown until measured; never claim a number here
    bandwidthBytesPerSec: 0,
    roles: ["mixed"],
    maxBatchTokens: 1024,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

/**
 * Aggregate FP32 matmul TFLOPS across `donorCount` in-process donor node
 * servers, each running one n^3 matmul concurrently through the
 * VirtualGpu coordinator over real HTTP (JSON kernel route). Returns the
 * aggregate TFLOPS. Verifies the first output element against a locally
 * computed dot product so a fast-but-wrong run cannot pass as a result.
 */
async function measureScalingTflops(donorCount: number, n: number): Promise<number> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-bench-vgpu-"));
  const servers: HwNodeServer[] = [];
  try {
    const infos: WorkerInfo[] = [];
    for (let i = 0; i < donorCount; i++) {
      const ramNode = new RamNode({
        capacityBytes: 512 * 1024 * 1024,
        spillDir: path.join(tmp, `ram-spill-${i}`),
      });
      const gpuNode = new GpuNode({ device: "cpu" });
      const cpuNode = new CpuNode({});
      const ssdNode = new SsdNode({ dir: path.join(tmp, `ssd-${i}`) });
      const server = await createHwNodeServer({
        routeTables: [ramNode.routes(), gpuNode.routes(), cpuNode.routes(), ssdNode.routes()],
      });
      servers.push(server);
      infos.push(makeDonorInfo(`bench-donor-${i}`, server.port));
    }
    const nodes = { listNodes: () => [...infos] };
    const ram = new VirtualRam({ nodes, defaultReplicas: 1 });
    const gpu = new VirtualGpu({ nodes, ram });

    const aVals = randomMatrix(n);
    const bVals = randomMatrix(n);
    const aAlloc = await ram.alloc(n * n * 4, { replicas: 1 });
    const bAlloc = await ram.alloc(n * n * 4, { replicas: 1 });
    const outAllocs = [];
    for (let i = 0; i < donorCount; i++) outAllocs.push(await ram.alloc(n * n * 4, { replicas: 1 }));
    for (const alloc of [aAlloc, bAlloc, ...outAllocs]) gpu.registerAllocation(alloc);
    await ram.write(aAlloc.vaddr, new Uint8Array(aVals.buffer, aVals.byteOffset, aVals.byteLength));
    await ram.write(bAlloc.vaddr, new Uint8Array(bVals.buffer, bVals.byteOffset, bVals.byteLength));

    const flopsPerDonor = 2 * n * n * n;
    const start = performance.now();
    await Promise.all(
      outAllocs.map((outAlloc) =>
        gpu.launch({
          op: "matmul",
          inputs: [
            { vaddr: aAlloc.vaddr, shape: [n, n], dtype: "float32" },
            { vaddr: bAlloc.vaddr, shape: [n, n], dtype: "float32" },
          ],
          outputs: [{ vaddr: outAlloc.vaddr, shape: [n, n], dtype: "float32" }],
          params: {},
        }),
      ),
    );
    const elapsedMs = performance.now() - start;

    // Spot-check correctness: out[0][0] = sum_k a[0][k] * b[k][0].
    let expected = 0;
    for (let k = 0; k < n; k++) expected += aVals[k] * bVals[k * n];
    const gotBytes = await ram.read(outAllocs[0].vaddr, 4);
    const got = new Float32Array(gotBytes.buffer, gotBytes.byteOffset, 1)[0];
    const tolerance = Math.max(1e-2, Math.abs(expected) * 1e-4);
    if (!Number.isFinite(got) || Math.abs(got - expected) > tolerance) {
      throw new Error(
        `scaling correctness check failed: out[0][0] = ${got}, expected ${expected} (n=${n}, donors=${donorCount})`,
      );
    }

    if (!(elapsedMs > 0)) return 0;
    return (donorCount * flopsPerDonor) / (elapsedMs / 1000) / 1e12;
  } finally {
    for (const server of servers) await server.close().catch(() => undefined);
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

export async function runBench(opts: BenchOptions): Promise<BenchResult[]> {
  const donors = clampDonors(opts.donors);
  const small = opts.size === "small";
  // Small mode keeps the test fast; large mode is the real benchmark.
  const matmulSizes = small ? [256, 512] : [1024, 2048];
  const scalingSize = small ? 256 : 1024;
  const bwElements = small ? 4 * 1024 * 1024 : 16 * 1024 * 1024;

  const results: BenchResult[] = [];
  const bf16Rubin = VGPU_TARGETS.primaryMetrics.bf16Tflops as number;
  const bf16B200 = (VGPU_TARGETS.secondaryMetrics as Record<string, number>).bf16Tflops;
  const bwRubin = VGPU_TARGETS.primaryMetrics.memoryBandwidthGBps as number;
  const bwB200 = (VGPU_TARGETS.secondaryMetrics as Record<string, number>).memoryBandwidthGBps;

  // (a) FP32 matmul TFLOPS, in-process on the local donor.
  for (const n of matmulSizes) {
    const timedRuns = n >= 2048 ? 1 : small ? 3 : 2;
    const tflops = await measureMatmulTflops(n, timedRuns);
    results.push(
      makeResult({
        device: "vgpu",
        metric: `matmul TFLOPS (FP32, ${n}^3)`,
        measured: tflops,
        unit: "TFLOPS",
        target: bf16Rubin,
        targetName: `${VGPU_TARGETS.primary} (BF16)`,
        donors: 1,
        notes: `${FP32_NOTE}. Timed ${timedRuns} run(s) after warmup, median.`,
      }),
      makeResult({
        device: "vgpu",
        metric: `matmul TFLOPS (FP32, ${n}^3)`,
        measured: tflops,
        unit: "TFLOPS",
        target: bf16B200,
        targetName: `${VGPU_TARGETS.secondary} (BF16)`,
        donors: 1,
        notes: `${FP32_NOTE}. Timed ${timedRuns} run(s) after warmup, median.`,
      }),
    );
  }

  // (b) Memory bandwidth via a timed copy/add kernel.
  const bwBytesPerSec = await measureMemoryBandwidthBytesPerSec(bwElements);
  const bwGBps = bwBytesPerSec / 1e9;
  results.push(
    makeResult({
      device: "vgpu",
      metric: "memory bandwidth (copy/add, FP32)",
      measured: bwGBps,
      unit: "GB/s",
      target: bwRubin,
      targetName: `${VGPU_TARGETS.primary} (HBM4)`,
      donors: 1,
      notes:
        "Timed add kernel: 3 arrays x 4 bytes/elem moved per run, median of 3 after warmup. " +
        "Local DRAM via CPU fallback, not HBM; per-donor figure.",
    }),
    makeResult({
      device: "vgpu",
      metric: "memory bandwidth (copy/add, FP32)",
      measured: bwGBps,
      unit: "GB/s",
      target: bwB200,
      targetName: `${VGPU_TARGETS.secondary} (HBM3e)`,
      donors: 1,
      notes:
        "Timed add kernel: 3 arrays x 4 bytes/elem moved per run, median of 3 after warmup. " +
        "Local DRAM via CPU fallback, not HBM; per-donor figure.",
    }),
  );

  // (c) Scaling: 1..donors in-process donor node servers, concurrent
  // matmuls through the VirtualGpu coordinator over real HTTP.
  for (let d = 1; d <= donors; d++) {
    const aggregate = await measureScalingTflops(d, scalingSize);
    results.push(
      makeResult({
        device: "vgpu",
        metric: `aggregate matmul TFLOPS (FP32, ${scalingSize}^3 x ${d} donor${d === 1 ? "" : "s"})`,
        measured: aggregate,
        unit: "TFLOPS",
        target: bf16Rubin,
        targetName: `${VGPU_TARGETS.primary} (BF16)`,
        donors: d,
        notes:
          `${FP32_NOTE}. Concurrent ${scalingSize}^3 matmuls via VirtualGpu over HTTP JSON kernel route, ` +
          `one per donor. In-process donors share this machine's cores, so aggregate is capped by local cores.`,
      }),
    );
  }

  if (opts.jsonOut !== undefined) {
    fs.mkdirSync(path.dirname(opts.jsonOut), { recursive: true });
    fs.writeFileSync(opts.jsonOut, JSON.stringify(results, null, 2));
  }
  return results;
}

// ---------------------------------------------------------------------------
// Standalone main: `node src/bench/bench-vgpu.ts`
// ---------------------------------------------------------------------------

function argValue(argv: string[], name: string): string | undefined {
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i] as string;
    if (arg === `--${name}` && i + 1 < argv.length) return argv[i + 1];
    if (arg.startsWith(`--${name}=`)) return arg.slice(name.length + 3);
  }
  return undefined;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const donorsRaw = argValue(argv, "donors");
  const donors = donorsRaw !== undefined ? Number(donorsRaw) : 4;
  const sizeRaw = argValue(argv, "size");
  const size = sizeRaw === "small" ? "small" : "large";
  const jsonOut = argValue(argv, "json-out");
  const results = await runBench({
    donors: Number.isFinite(donors) ? donors : 4,
    size,
    jsonOut,
  });
  console.log(formatResultsTable(results));
  console.log();
  console.log(
    "Notes: FP32 on the CPU fallback is compared against BF16 tensor-core targets as the " +
      "closest honest line — FP32 != FP4/BF16 tensor-core rates, and there is no FP4 hardware " +
      "here, so the gap is a lower bound. Scaling donors run in-process and share this " +
      "machine's cores; real donors on separate machines would each add their own cores.",
  );
  if (jsonOut !== undefined) console.log(`Wrote ${jsonOut}`);
}

const isMain =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main().catch((err: unknown) => {
    console.error(`bench-vgpu failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 1;
  });
}
