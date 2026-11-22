/**
 * calibrate.ts — MEASURED donor capability, never invented.
 *
 * The worker used to advertise `flopsPerSec: 1e12` — a made-up number.
 * This module times a real 1024^3 FP32 matmul (tiled + worker-thread
 * parallel, the fastest path this donor has) and reports
 * 2*M*N*K / elapsed seconds. Median of 3 timed runs after a warmup.
 *
 * If calibration fails for any reason, callers must fall back to 0 and
 * log a warning — never silently claim a throughput the donor cannot
 * prove. A 0 means "unknown", and the pools that consume it (VirtualGpu
 * computeUnits, stats) treat 0 honestly.
 */

import { add, matmulParallel, matmulTiled } from "./kernels-cpu.ts";
import { median } from "../bench/types.ts";

function randomMatrix(n: number): Float32Array {
  const a = new Float32Array(n * n);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

/** Flat random vector of exactly `length` elements (not n x n). */
function randomVector(length: number): Float32Array {
  const a = new Float32Array(length);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 2 - 1;
  return a;
}

/**
 * Measure sustained FP32 matmul throughput on this donor in FLOPS/sec.
 * Timed 1024^3 tiled+parallel matmul, median of 3 runs after 1 warmup
 * run; TFLOPS = 2*M*N*K / elapsed. Returns 0 on any failure.
 */
export async function measureMatmulTflops(): Promise<number> {
  try {
    const n = 1024;
    const a = randomMatrix(n);
    const b = randomMatrix(n);
    // Warmup: JIT, worker-pool spin-up, page faults — untimed.
    await matmulParallel(a, n, n, b, n, n);
    const flops = 2 * n * n * n;
    const elapsedMs: number[] = [];
    for (let run = 0; run < 3; run++) {
      const start = performance.now();
      await matmulParallel(a, n, n, b, n, n);
      elapsedMs.push(performance.now() - start);
    }
    const med = median(elapsedMs);
    if (!(med > 0) || !Number.isFinite(med)) return 0;
    return flops / (med / 1000); // FLOPS/sec
  } catch {
    return 0;
  }
}

/**
 * Measure sustained memory bandwidth in bytes/sec with a timed
 * element-wise add kernel: each run reads two arrays and writes one,
 * so bytes moved = 3 * N * 4. Median of 3 runs after a warmup.
 * Returns 0 on any failure.
 */
export async function measureMemoryBandwidthBytesPerSec(
  numElements = 16 * 1024 * 1024,
): Promise<number> {
  try {
    const a = randomVector(numElements);
    const b = randomVector(numElements);
    add(a, b); // warmup
    const bytes = 3 * numElements * 4;
    const elapsedMs: number[] = [];
    for (let run = 0; run < 3; run++) {
      const start = performance.now();
      add(a, b);
      elapsedMs.push(performance.now() - start);
    }
    const med = median(elapsedMs);
    if (!(med > 0) || !Number.isFinite(med)) return 0;
    return bytes / (med / 1000);
  } catch {
    return 0;
  }
}

/** Single-thread reference: the tiled sequential path on 512^3, for
 *  callers that want a per-core number. Not used by the worker. */
export async function measureMatmulTflopsSingleThread(): Promise<number> {
  try {
    const n = 512;
    const a = randomMatrix(n);
    const b = randomMatrix(n);
    matmulTiled(a, n, n, b, n, n); // warmup
    const flops = 2 * n * n * n;
    const elapsedMs: number[] = [];
    for (let run = 0; run < 3; run++) {
      const start = performance.now();
      matmulTiled(a, n, n, b, n, n);
      elapsedMs.push(performance.now() - start);
    }
    const med = median(elapsedMs);
    if (!(med > 0) || !Number.isFinite(med)) return 0;
    return flops / (med / 1000);
  } catch {
    return 0;
  }
}
