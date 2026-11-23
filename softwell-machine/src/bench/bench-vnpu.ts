/**
 * bench-vnpu.ts — vNPU benchmark: REAL timed execution, never asserted.
 *
 * Measures, on this machine, with real timed runs:
 *  (a) INT8 TOPS, FP8 TFLOPS, BF16 TFLOPS at 512^3 through
 *      VirtualNpu.matmulTransposeB (staged weights — the forward-pass
 *      reality: per-call cost is activation quantization + the op);
 *  (b) memory bandwidth (vRAM -> tile) via a timed copy/add kernel;
 *  (c) pooled SRAM capacity (D1: 256 MiB per virtual device);
 *  (d) scaling: 1..N in-process VirtualNpu instances (one fake donor
 *      node each), concurrent 512^3 INT8 matmuls, aggregate TOPS.
 *
 * Honesty notes (also printed by main()):
 *  - The measured rates are INT8/FP8/BF16 on a CPU fallback — no NPU
 *    silicon here. The INT8 core is a plain triple loop in JS and the
 *    FP8/BF16 paths run the FP32 kernel on dequantized values, so the
 *    gap to TPU/Trainium INT8/FP8 tensor rates is a lower bound on
 *    the real gap, not a like-for-like defeat.
 *  - The scaling donors run in-process and share this machine's
 *    physical cores, so aggregate TOPS in-process is capped by local
 *    cores; real donors on separate machines would each add their own
 *    cores. `donorsToBeat` is computed from per-donor throughput,
 *    which is the figure that transfers to real donors. A donor-side
 *    VnpuNode (real NPU silicon, over-HTTP scaling like the vGPU
 *    module) is future work — docs/VNPU.md §4.2.
 *  - The SRAM capacity row is a design constant (D1), not a timed
 *    rate: the measured value is the enforced per-device capacity
 *    (a 256 MiB stage fits, a larger single weight spills — pinned
 *    by test/vnpu-sram.test.ts). It is the one row that can PASS.
 *
 * Usage: node src/bench/bench-vnpu.ts [--donors N] [--size small|large]
 *        [--json-out <path>]
 */

import { fileURLToPath } from "node:url";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { VirtualNpu, VNPU_SRAM_BYTES } from "../hw/vnpu.ts";
import { measureMemoryBandwidthBytesPerSec } from "../hw/calibrate.ts";
import { VNPU_TARGETS } from "./targets.ts";
import {
  formatResultsTable,
  makeResult,
  median,
  type BenchOptions,
  type BenchResult,
} from "./types.ts";
import type {
  IVirtualRam,
  NodeProvider,
  VramStats,
  WorkerInfo,
} from "../contracts.ts";

const INT8_NOTE =
  "INT8 on CPU fallback (plain JS triple-loop core, no NPU silicon); " +
  "compared vs INT8 tensor-core targets as the closest honest line";
const FP8_NOTE =
  "FP8 on CPU fallback (FP32 kernel over dequantized values, no NPU silicon); " +
  "compared vs FP8 tensor-core targets as the closest honest line";
const BF16_NOTE =
  "BF16 on CPU fallback (FP32 kernel over dequantized values, no NPU silicon); " +
  "compared vs BF16 tensor-core targets as the closest honest line";

function clampDonors(donors: number | undefined): number {
  const n = donors ?? 1;
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.min(4, Math.floor(n)));
}

function makeNode(workerId: string): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 0, // unknown until measured; never claim a number here
    bandwidthBytesPerSec: 0,
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
  async alloc(): Promise<never> {
    throw new Error("FakeRam.alloc not used here");
  }
  async free(): Promise<void> {}
  async read(): Promise<never> {
    throw new Error("FakeRam.read not used here");
  }
  async write(): Promise<void> {}
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: 0, usedBytes: 0, freeBytes: 0, pages: 0, nodes: 0 };
  }
}

function makeNpu(workerId: string): VirtualNpu {
  const nodes: NodeProvider = { listNodes: () => [makeNode(workerId)] };
  return new VirtualNpu({ nodes, ram: new FakeRam() });
}

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

function randomMatrix(n: number, rng: () => number): Float32Array {
  const a = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) a[i] = (rng() - 0.5) * 2;
  return a;
}

/** Timed matmulTransposeB TOPS at size n for one format, staged B. */
async function measureMatmulTops(
  npu: VirtualNpu,
  n: number,
  format: "int8" | "fp8" | "bf16",
  timedRuns: number,
): Promise<number> {
  const rng = mulberry32(20261122 + n);
  const a = randomMatrix(n, rng);
  const b = randomMatrix(n, rng);
  const key = `bench-${format}-${n}`;
  npu.stageWeights(key, b, n, n, format);
  try {
    await npu.matmulTransposeB(a, n, n, b, n, n, { format, weightKey: key }); // warmup
    const flops = 2 * n * n * n;
    const elapsedMs: number[] = [];
    for (let run = 0; run < timedRuns; run++) {
      const start = performance.now();
      await npu.matmulTransposeB(a, n, n, b, n, n, { format, weightKey: key });
      elapsedMs.push(performance.now() - start);
    }
    const med = median(elapsedMs);
    if (!(med > 0)) return 0;
    return flops / (med / 1000) / 1e12;
  } finally {
    npu.evictWeights(key);
  }
}

/**
 * Aggregate INT8 TOPS across `donorCount` in-process VirtualNpu
 * instances (one fake donor node each), one 512^3 matmul each,
 * run concurrently. Same in-process caveat as the vGPU module:
 * donors share this machine's cores.
 */
async function measureScalingTops(donorCount: number, n: number): Promise<number> {
  const rng = mulberry32(20261122 + n);
  const a = randomMatrix(n, rng);
  const b = randomMatrix(n, rng);
  const npus = Array.from({ length: donorCount }, (_, i) => makeNpu(`bench-npu-${i}`));
  for (const [i, npu] of npus.entries()) npu.stageWeights(`scale-${i}`, b, n, n, "int8");
  const flopsPerDonor = 2 * n * n * n;
  const start = performance.now();
  await Promise.all(
    npus.map((npu, i) =>
      npu.matmulTransposeB(a, n, n, b, n, n, { format: "int8", weightKey: `scale-${i}` }),
    ),
  );
  const elapsedMs = performance.now() - start;
  if (!(elapsedMs > 0)) return 0;
  return (donorCount * flopsPerDonor) / (elapsedMs / 1000) / 1e12;
}

export async function runBench(opts: BenchOptions): Promise<BenchResult[]> {
  const donors = clampDonors(opts.donors);
  const small = opts.size === "small";
  const matmulSize = small ? 256 : 512;
  const timedRuns = 2;
  const bwElements = small ? 4 * 1024 * 1024 : 16 * 1024 * 1024;

  const results: BenchResult[] = [];
  const npu = makeNpu("bench-npu");

  // (a) INT8 / FP8 / BF16 matmul TOPS/TFLOPS at 512^3 (256^3 small).
  const int8Tops = await measureMatmulTops(npu, matmulSize, "int8", timedRuns);
  const fp8Tflops = await measureMatmulTops(npu, matmulSize, "fp8", timedRuns);
  const bf16Tflops = await measureMatmulTops(npu, matmulSize, "bf16", timedRuns);
  const int8V6e = VNPU_TARGETS.primaryMetrics.int8Tops as number;
  const int8V5p = (VNPU_TARGETS.secondaryMetrics as Record<string, number>).int8Tops;
  const bf16V6e = VNPU_TARGETS.primaryMetrics.bf16Tflops as number;
  const bf16V5p = (VNPU_TARGETS.secondaryMetrics as Record<string, number>).bf16Tflops;
  const fp8V5p = (VNPU_TARGETS.secondaryMetrics as Record<string, number>).fp8Tflops;
  results.push(
    makeResult({
      device: "vnpu",
      metric: `INT8 TOPS (MATMUL_INT8, ${matmulSize}^3)`,
      measured: int8Tops,
      unit: "TOPS",
      target: int8V6e,
      targetName: `${VNPU_TARGETS.primary} (INT8)`,
      donors: 1,
      notes: `${INT8_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `INT8 TOPS (MATMUL_INT8, ${matmulSize}^3)`,
      measured: int8Tops,
      unit: "TOPS",
      target: int8V5p,
      targetName: `${VNPU_TARGETS.secondary} (INT8, Dec 2023 launch figure)`,
      donors: 1,
      notes: `${INT8_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `INT8 TOPS (MATMUL_INT8, ${matmulSize}^3)`,
      measured: int8Tops,
      unit: "TOPS",
      target: 380,
      targetName: "AWS Inferentia2 (INT8)",
      donors: 1,
      notes: `${INT8_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `FP8 TFLOPS (MATMUL_FP8, ${matmulSize}^3)`,
      measured: fp8Tflops,
      unit: "TFLOPS",
      target: 1_299,
      targetName: "AWS Trainium2 (FP8)",
      donors: 1,
      notes: `${FP8_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `FP8 TFLOPS (MATMUL_FP8, ${matmulSize}^3)`,
      measured: fp8Tflops,
      unit: "TFLOPS",
      target: fp8V5p,
      targetName: `${VNPU_TARGETS.secondary} (FP8)`,
      donors: 1,
      notes: `${FP8_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `BF16 TFLOPS (MATMUL_BF16, ${matmulSize}^3)`,
      measured: bf16Tflops,
      unit: "TFLOPS",
      target: bf16V6e,
      targetName: `${VNPU_TARGETS.primary} (BF16)`,
      donors: 1,
      notes: `${BF16_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
    makeResult({
      device: "vnpu",
      metric: `BF16 TFLOPS (MATMUL_BF16, ${matmulSize}^3)`,
      measured: bf16Tflops,
      unit: "TFLOPS",
      target: 667,
      targetName: "AWS Trainium2 (BF16/FP16)",
      donors: 1,
      notes: `${BF16_NOTE}. Staged weights, timed ${timedRuns} run(s) after warmup, median.`,
    }),
  );

  // (b) Memory bandwidth (vRAM -> tile) via a timed copy/add kernel.
  const bwBytesPerSec = await measureMemoryBandwidthBytesPerSec(bwElements);
  const bwGBps = bwBytesPerSec / 1e9;
  const bwV6e = VNPU_TARGETS.primaryMetrics.memoryBandwidthGBps as number;
  const bwV5p = (VNPU_TARGETS.secondaryMetrics as Record<string, number>).memoryBandwidthGBps;
  results.push(
    makeResult({
      device: "vnpu",
      metric: "memory bandwidth (vRAM -> tile, copy/add)",
      measured: bwGBps,
      unit: "GB/s",
      target: bwV5p,
      targetName: `${VNPU_TARGETS.secondary} (HBM, GB/s as published)`,
      donors: 1,
      notes:
        "Timed add kernel: 3 arrays x 4 bytes/elem moved per run, median of 3 after warmup. " +
        "Local DRAM via CPU fallback, not HBM; per-donor figure.",
    }),
    makeResult({
      device: "vnpu",
      metric: "memory bandwidth (vRAM -> tile, copy/add)",
      measured: bwGBps,
      unit: "GB/s",
      target: bwV6e,
      targetName: `${VNPU_TARGETS.primary} (HBM, GB/s as published)`,
      donors: 1,
      notes:
        "Timed add kernel: 3 arrays x 4 bytes/elem moved per run, median of 3 after warmup. " +
        "Local DRAM via CPU fallback, not HBM; per-donor figure.",
    }),
  );

  // (c) Pooled SRAM capacity (D1: 256 MiB per virtual device). A
  // capacity row, not a timed rate: the measured value is the enforced
  // per-device capacity (a 256 MiB stage fits, a larger single weight
  // spills — pinned by test/vnpu-sram.test.ts).
  const sramMiB = VNPU_SRAM_BYTES / 1024 / 1024;
  results.push(
    makeResult({
      device: "vnpu",
      metric: "pooled SRAM capacity",
      measured: sramMiB * donors,
      unit: "MiB",
      target: 224,
      targetName: "AWS Trainium2 (SBUF, 28 MiB x 8 NeuronCore-v3)",
      donors,
      notes:
        "Design constant D1 (256 MiB per virtual device), not a timed rate; " +
        "the enforced capacity the SRAM tests pin. Pooled aggregate is N x 256 MiB.",
    }),
  );

  // (d) Scaling: 1..donors in-process devices, concurrent INT8 matmuls.
  for (let d = 1; d <= donors; d++) {
    const aggregate = await measureScalingTops(d, matmulSize);
    results.push(
      makeResult({
        device: "vnpu",
        metric: `aggregate INT8 TOPS (MATMUL_INT8, ${matmulSize}^3 x ${d} donor${d === 1 ? "" : "s"})`,
        measured: aggregate,
        unit: "TOPS",
        target: int8V6e,
        targetName: `${VNPU_TARGETS.primary} (INT8)`,
        donors: d,
        notes:
          `${INT8_NOTE}. Concurrent ${matmulSize}^3 matmuls on one VirtualNpu per donor. ` +
          `In-process donors share this machine's cores, so aggregate is capped by local cores.`,
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
// Standalone main: `node src/bench/bench-vnpu.ts`
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
  const donors = donorsRaw !== undefined ? Number(donorsRaw) : 1;
  const sizeRaw = argValue(argv, "size");
  const size = sizeRaw === "small" ? "small" : "large";
  const jsonOut = argValue(argv, "json-out");
  const results = await runBench({
    donors: Number.isFinite(donors) ? donors : 1,
    size,
    jsonOut,
  });
  console.log(formatResultsTable(results));
  console.log();
  console.log(
    "Notes: INT8/FP8/BF16 rates are CPU-fallback rates with no NPU silicon here — " +
      "the gap to TPU/Trainium tensor rates is a lower bound. Scaling donors run " +
      "in-process and share this machine's cores; real donors on separate machines " +
      "would each add their own cores. The SRAM capacity row is design constant D1, " +
      "not a timed rate.",
  );
  if (jsonOut !== undefined) console.log(`Wrote ${jsonOut}`);
}

const isMain =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main().catch((err: unknown) => {
    console.error(`bench-vnpu failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 1;
  });
}
