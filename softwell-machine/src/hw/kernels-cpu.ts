import { Worker } from "node:worker_threads";
import os from "node:os";

/**
 * kernels-cpu.ts — CPU fallback kernel library for vGPU.
 *
 * Plain TypeScript, Float32Array math, no I/O. These are the kernels a donor
 * node runs when it has no real GPU (or when the reference implementation is
 * used): the virtual GPU still presents a virtual device, but the work
 * honestly executes on CPU and reports device 'cpu'.
 *
 * Every function is exported individually and also bundled in `CpuKernels`.
 * All inputs are row-major flat Float32Arrays. Functions that return a new
 * array are documented as such; `rope` and `softmaxRows` mutate in place
 * and return the same array for convenience.
 *
 * The engine owner's attention.ts may replace these later; the signatures
 * below are the stable contract — do not change them without updating
 * src/hw/vgpu.ts (GpuNode dispatches on them) and test/hw-vgpu.test.ts.
 */

export function matmul(
  a: Float32Array,
  aRows: number,
  aCols: number,
  b: Float32Array,
  bRows: number,
  bCols: number,
): Float32Array {
  if (aCols !== bRows) {
    throw new Error(
      `matmul: inner dimensions mismatch: a is ${aRows}x${aCols}, b is ${bRows}x${bCols}`,
    );
  }
  if (a.length !== aRows * aCols) {
    throw new Error(`matmul: a has ${a.length} elements, expected ${aRows * aCols}`);
  }
  if (b.length !== bRows * bCols) {
    throw new Error(`matmul: b has ${b.length} elements, expected ${bRows * bCols}`);
  }
  const out = new Float32Array(aRows * bCols);
  // i,k,j order: reuses a[i,k] across the row of b.
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i * aCols + k];
      if (aik === 0) continue;
      for (let j = 0; j < bCols; j++) {
        out[i * bCols + j] += aik * b[k * bCols + j];
      }
    }
  }
  return out;
}

export function add(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new Error(`add: length mismatch: ${a.length} vs ${b.length}`);
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

export function mul(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new Error(`mul: length mismatch: ${a.length} vs ${b.length}`);
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

/** SiLU (SwiGLU activation): x * sigmoid(x). Returns a new array. */
export function silu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return out;
}

/**
 * RMSNorm over the last dimension. `x` is [rows, cols] row-major, `weight`
 * is [cols]. Returns a new array: y = x / sqrt(mean(x^2) + eps) * weight,
 * computed per row.
 */
export function rmsNorm(
  x: Float32Array,
  rows: number,
  cols: number,
  weight: Float32Array,
  eps: number,
): Float32Array {
  if (x.length !== rows * cols) {
    throw new Error(`rmsNorm: x has ${x.length} elements, expected ${rows * cols}`);
  }
  if (weight.length !== cols) {
    throw new Error(`rmsNorm: weight has ${weight.length} elements, expected ${cols}`);
  }
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    let sumSq = 0;
    for (let c = 0; c < cols; c++) {
      const v = x[r * cols + c];
      sumSq += v * v;
    }
    const scale = 1 / Math.sqrt(sumSq / cols + eps);
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = x[r * cols + c] * scale * weight[c];
    }
  }
  return out;
}

/**
 * Rotary position embeddings, applied IN PLACE. Returns the same array.
 *
 * Layout assumed: `x` is [rows, heads, headDim] row-major, where `heads` is
 * inferred as x.length / (rows * headDim). `positions[r]` is the absolute
 * position of row r. Each consecutive pair (2i, 2i+1) within a head is
 * rotated by angle positions[r] * theta^(-2i / headDim).
 */
export function rope(
  x: Float32Array,
  rows: number,
  headDim: number,
  positions: ArrayLike<number>,
  theta: number,
): Float32Array {
  if (headDim % 2 !== 0) {
    throw new Error(`rope: headDim must be even, got ${headDim}`);
  }
  if (x.length % (rows * headDim) !== 0) {
    throw new Error(
      `rope: x has ${x.length} elements, not divisible into ${rows} rows of headDim ${headDim}`,
    );
  }
  if (positions.length !== rows) {
    throw new Error(`rope: positions has ${positions.length} entries, expected ${rows}`);
  }
  const heads = x.length / (rows * headDim);
  const half = headDim / 2;
  for (let r = 0; r < rows; r++) {
    const pos = positions[r];
    for (let h = 0; h < heads; h++) {
      const base = (r * heads + h) * headDim;
      for (let i = 0; i < half; i++) {
        const angle = pos * Math.pow(theta, -2 * i / headDim);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const x0 = x[base + 2 * i];
        const x1 = x[base + 2 * i + 1];
        x[base + 2 * i] = x0 * cos - x1 * sin;
        x[base + 2 * i + 1] = x0 * sin + x1 * cos;
      }
    }
  }
  return x;
}

/**
 * Row-wise softmax, numerically stable (subtracts the row max first).
 * Applied IN PLACE. Returns the same array. `x` is [rows, cols] row-major.
 */
export function softmaxRows(
  x: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  if (x.length !== rows * cols) {
    throw new Error(`softmaxRows: x has ${x.length} elements, expected ${rows * cols}`);
  }
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let max = -Infinity;
    for (let c = 0; c < cols; c++) {
      if (x[base + c] > max) max = x[base + c];
    }
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[base + c] - max);
      x[base + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) x[base + c] /= sum;
  }
  return x;
}

export interface AttentionOpts {
  numHeads: number;
  headDim: number;
  /** KV heads for grouped-query attention; defaults to numHeads (plain MHA). */
  numKvHeads?: number;
  /** When true, query at absolute position p attends only to keys at <= p. */
  causal?: boolean;
}

/**
 * Scaled dot-product attention.
 *
 * Layouts (all row-major, flat):
 * - `q`: [qLen, numHeads * headDim]
 * - `k`, `v`: [kvLen, numKvHeads * headDim]
 * - returns [qLen, numHeads * headDim]
 *
 * Scores are q.k^T / sqrt(headDim), softmaxed per query head, then
 * multiplied by v. GQA: KV heads are repeated — query head h uses KV head
 * floor(h / (numHeads / numKvHeads)), so numHeads must be divisible by
 * numKvHeads. With `causal`, query row i sits at absolute position
 * kvLen - qLen + i, so it attends to keys j <= kvLen - qLen + i; this
 * covers both prefill (qLen == kvLen) and single-token decode (qLen == 1,
 * attends to the whole cache).
 */
export function attention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  opts: AttentionOpts,
): Float32Array {
  const numHeads = opts.numHeads;
  const headDim = opts.headDim;
  const numKvHeads = opts.numKvHeads ?? numHeads;
  if (numHeads % numKvHeads !== 0) {
    throw new Error(
      `attention: numHeads ${numHeads} is not divisible by numKvHeads ${numKvHeads}`,
    );
  }
  if (q.length % (numHeads * headDim) !== 0) {
    throw new Error(
      `attention: q has ${q.length} elements, not a multiple of numHeads*headDim ${numHeads * headDim}`,
    );
  }
  if (k.length % (numKvHeads * headDim) !== 0 || v.length % (numKvHeads * headDim) !== 0) {
    throw new Error(
      `attention: k/v lengths ${k.length}/${v.length} are not multiples of numKvHeads*headDim ${numKvHeads * headDim}`,
    );
  }
  const qLen = q.length / (numHeads * headDim);
  const kvLen = k.length / (numKvHeads * headDim);
  if (v.length / (numKvHeads * headDim) !== kvLen) {
    throw new Error(`attention: k and v have different sequence lengths`);
  }
  const groupSize = numHeads / numKvHeads;
  const scale = 1 / Math.sqrt(headDim);
  const out = new Float32Array(qLen * numHeads * headDim);
  const scores = new Float32Array(kvLen);
  for (let qi = 0; qi < qLen; qi++) {
    // Absolute position of this query row (see doc comment above).
    const qPos = kvLen - qLen + qi;
    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / groupSize);
      const qBase = (qi * numHeads + h) * headDim;
      // Scores against every key, with the causal mask applied.
      let max = -Infinity;
      for (let kj = 0; kj < kvLen; kj++) {
        if (opts.causal === true && kj > qPos) {
          scores[kj] = -Infinity;
          continue;
        }
        const kBase = (kj * numKvHeads + kvHead) * headDim;
        let dot = 0;
        for (let d = 0; d < headDim; d++) dot += q[qBase + d] * k[kBase + d];
        scores[kj] = dot * scale;
        if (scores[kj] > max) max = scores[kj];
      }
      // Stable softmax over the unmasked scores.
      let sum = 0;
      for (let kj = 0; kj < kvLen; kj++) {
        if (scores[kj] === -Infinity) continue;
        const e = Math.exp(scores[kj] - max);
        scores[kj] = e;
        sum += e;
      }
      const oBase = (qi * numHeads + h) * headDim;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let kj = 0; kj < kvLen; kj++) {
          if (scores[kj] === 0) continue;
          const vBase = (kj * numKvHeads + kvHead) * headDim;
          acc += scores[kj] * v[vBase + d];
        }
        out[oBase + d] = acc / sum;
      }
    }
  }
  return out;
}

/**
 * Embedding lookup. `table` is [tableRows, dim] row-major; `ids` are row
 * indices. Returns [ids.length, dim] row-major. Throws on an out-of-range id.
 */
export function embeddingLookup(
  table: Float32Array,
  tableRows: number,
  dim: number,
  ids: ArrayLike<number>,
): Float32Array {
  if (table.length !== tableRows * dim) {
    throw new Error(
      `embeddingLookup: table has ${table.length} elements, expected ${tableRows * dim}`,
    );
  }
  const out = new Float32Array(ids.length * dim);
  for (let i = 0; i < ids.length; i++) {
    const id = Math.trunc(ids[i]);
    if (id < 0 || id >= tableRows) {
      throw new Error(`embeddingLookup: token id ${id} out of range [0, ${tableRows})`);
    }
    out.set(table.subarray(id * dim, (id + 1) * dim), i * dim);
  }
  return out;
}

/**
 * Cache-blocked / tiled matmul.
 *
 * Same contract as `matmul` (row-major, [aRows,aCols] x [bRows,bCols]).
 * Differences from the naive kernel, all deliberate:
 * - i/jj/kk tiling (64-wide tiles) so the working set of B stays in cache
 *   instead of streaming the whole B matrix once per row of A.
 * - Register-blocked inner loop: accumulators for 4 consecutive output
 *   columns are held in locals across the whole K tile and stored back
 *   once per tile, instead of a load-add-store per (k, j).
 * - NO zero-skip branch in the hot path (the naive kernel skips aik === 0,
 *   which is a branch per inner iteration and changes results when B
 *   holds infinities/NaNs: 0 * Inf is NaN, skipping keeps the old value).
 *
 * Accumulation order per output element is still k ascending, exactly as
 * in `matmul`, so on finite inputs the result is bit-identical to the
 * naive kernel. (Where the naive kernel's zero-skip would have skipped a
 * 0 * Inf = NaN term, this kernel honestly produces NaN.)
 *
 * BIT-IDENTITY DETAIL: the naive kernel rounds every partial sum back to
 * float32 on every store (`out[...] += ...` stores through the
 * Float32Array). Holding accumulators in double registers across k would
 * accumulate in double precision and differ in the last ulp, so every
 * step here applies Math.fround() — the same float32 rounding the store
 * would do — keeping the per-step rounding points identical to `matmul`.
 */
const TILE_M = 64;
const TILE_K = 64;
const TILE_N = 64;

export function matmulTiled(
  a: Float32Array,
  aRows: number,
  aCols: number,
  b: Float32Array,
  bRows: number,
  bCols: number,
): Float32Array {
  if (aCols !== bRows) {
    throw new Error(
      `matmul: inner dimensions mismatch: a is ${aRows}x${aCols}, b is ${bRows}x${bCols}`,
    );
  }
  if (a.length !== aRows * aCols) {
    throw new Error(`matmul: a has ${a.length} elements, expected ${aRows * aCols}`);
  }
  if (b.length !== bRows * bCols) {
    throw new Error(`matmul: b has ${b.length} elements, expected ${bRows * bCols}`);
  }
  const out = new Float32Array(aRows * bCols);
  for (let ii = 0; ii < aRows; ii += TILE_M) {
    const iEnd = Math.min(ii + TILE_M, aRows);
    for (let jj = 0; jj < bCols; jj += TILE_N) {
      const jEnd = Math.min(jj + TILE_N, bCols);
      for (let i = ii; i < iEnd; i++) {
        const aBase = i * aCols;
        const oBase = i * bCols;
        // Register-blocked groups of 4 output columns.
        let j0 = jj;
        for (; j0 + 4 <= jEnd; j0 += 4) {
          for (let kk = 0; kk < aCols; kk += TILE_K) {
            const kEnd = Math.min(kk + TILE_K, aCols);
            let o0 = out[oBase + j0];
            let o1 = out[oBase + j0 + 1];
            let o2 = out[oBase + j0 + 2];
            let o3 = out[oBase + j0 + 3];
            for (let k = kk; k < kEnd; k++) {
              const aik = a[aBase + k];
              const bBase = k * bCols;
              // Math.fround per step: same float32 rounding point as the
              // naive kernel's per-step store (see doc comment above).
              o0 = Math.fround(o0 + aik * b[bBase + j0]);
              o1 = Math.fround(o1 + aik * b[bBase + j0 + 1]);
              o2 = Math.fround(o2 + aik * b[bBase + j0 + 2]);
              o3 = Math.fround(o3 + aik * b[bBase + j0 + 3]);
            }
            out[oBase + j0] = o0;
            out[oBase + j0 + 1] = o1;
            out[oBase + j0 + 2] = o2;
            out[oBase + j0 + 3] = o3;
          }
        }
        // Tail columns (fewer than 4 left in the tile).
        for (; j0 < jEnd; j0++) {
          for (let kk = 0; kk < aCols; kk += TILE_K) {
            const kEnd = Math.min(kk + TILE_K, aCols);
            let acc = out[oBase + j0];
            for (let k = kk; k < kEnd; k++) {
              acc = Math.fround(acc + a[aBase + k] * b[k * bCols + j0]);
            }
            out[oBase + j0] = acc;
          }
        }
      }
    }
  }
  return out;
}

/**
 * Worker-thread parallel matmul.
 *
 * Splits the output rows across `threads` workers (default: all available
 * cores). Inputs/output live in SharedArrayBuffers; each worker owns a
 * disjoint row range, and every output element is accumulated in k
 * ascending order by exactly one worker — the same order as `matmul` /
 * `matmulTiled` — so the result is BIT-IDENTICAL to the sequential paths
 * on finite inputs (no zero-skip anywhere).
 *
 * Workers come from a small LAZY POOL, not one spawn per call: spawning a
 * worker costs ~90 ms on this machine, which would swamp any matmul
 * smaller than ~10^9 FLOPs. Pool workers are unref'd, so they never keep
 * a test run or CLI process alive after the work is done. Small matmuls
 * stay on the sequential tiled path below MATMUL_PARALLEL_MIN_FLOPS:
 * measured on this machine, the parallel path only clearly wins above
 * ~10^8 FLOPs (512^3 and up, ~1.2–2.5x); at 256^3 it is within noise of
 * the tiled path, so the threshold sits at 128M FLOPs.
 */
export const MATMUL_PARALLEL_MIN_FLOPS = 128 * 1024 * 1024;

/** Worker source (plain JS — workers run with eval, not TS stripping).
 *  Same cache-blocked, register-blocked algorithm as matmulTiled, over
 *  the worker's row range: per-element accumulation is k ascending, so
 *  the result is bit-identical to the sequential kernels. */
const MATMUL_WORKER_SOURCE = `
const { parentPort } = require("node:worker_threads");
parentPort.on("message", (job) => {
  const { jobId, aSab, bSab, outSab, aCols, bCols, startRow, endRow } = job;
  const a = new Float32Array(aSab);
  const b = new Float32Array(bSab);
  const out = new Float32Array(outSab);
  const TM = 64, TK = 64, TN = 64;
  for (let ii = startRow; ii < endRow; ii += TM) {
    const iEnd = Math.min(ii + TM, endRow);
    for (let jj = 0; jj < bCols; jj += TN) {
      const jEnd = Math.min(jj + TN, bCols);
      for (let i = ii; i < iEnd; i++) {
        const aBase = i * aCols;
        const oBase = i * bCols;
        let j0 = jj;
        for (; j0 + 4 <= jEnd; j0 += 4) {
          for (let kk = 0; kk < aCols; kk += TK) {
            const kEnd = Math.min(kk + TK, aCols);
            let o0 = out[oBase + j0];
            let o1 = out[oBase + j0 + 1];
            let o2 = out[oBase + j0 + 2];
            let o3 = out[oBase + j0 + 3];
            for (let k = kk; k < kEnd; k++) {
              const aik = a[aBase + k];
              const bBase = k * bCols;
              o0 = Math.fround(o0 + aik * b[bBase + j0]);
              o1 = Math.fround(o1 + aik * b[bBase + j0 + 1]);
              o2 = Math.fround(o2 + aik * b[bBase + j0 + 2]);
              o3 = Math.fround(o3 + aik * b[bBase + j0 + 3]);
            }
            out[oBase + j0] = o0;
            out[oBase + j0 + 1] = o1;
            out[oBase + j0 + 2] = o2;
            out[oBase + j0 + 3] = o3;
          }
        }
        for (; j0 < jEnd; j0++) {
          for (let kk = 0; kk < aCols; kk += TK) {
            const kEnd = Math.min(kk + TK, aCols);
            let acc = out[oBase + j0];
            for (let k = kk; k < kEnd; k++) acc = Math.fround(acc + a[aBase + k] * b[k * bCols + j0]);
            out[oBase + j0] = acc;
          }
        }
      }
    }
  }
  parentPort.postMessage({ jobId });
});
`;

/** One pooled worker plus its in-flight job resolvers. */
interface PooledWorker {
  worker: Worker;
  pending: Map<number, { resolve: () => void; reject: (err: Error) => void }>;
  /** True while a one-shot 'message' listener is armed on the worker. */
  listening: boolean;
}

/** Pools keyed by size, so concurrent callers asking for different
 *  thread counts never resize each other's pool mid-flight. */
const matmulPools = new Map<number, PooledWorker[]>();
let matmulJobCounter = 0;

function getMatmulPool(size: number): PooledWorker[] {
  const existing = matmulPools.get(size);
  if (existing !== undefined) return existing;
  const pool: PooledWorker[] = [];
  for (let i = 0; i < size; i++) {
    const worker = new Worker(MATMUL_WORKER_SOURCE, { eval: true });
    // Never keep the host process alive just for idle pool workers.
    worker.unref();
    const entry: PooledWorker = { worker, pending: new Map(), listening: false };
    // A persistent 'error' listener is safe (it does not re-ref the
    // worker); a persistent 'message' listener would — see below.
    worker.on("error", (err) => {
      const wrapped = err instanceof Error ? err : new Error(String(err));
      for (const { reject } of entry.pending.values()) reject(wrapped);
      entry.pending.clear();
    });
    pool.push(entry);
  }
  matmulPools.set(size, pool);
  return pool;
}

/**
 * Arm a one-shot 'message' listener on the worker, if none is armed.
 * A PERSISTENT 'message' listener re-refs the worker's MessagePort and
 * keeps the host process alive forever (verified: a pool with
 * worker.on('message') never lets `node test/*.test.ts` exit, while
 * one-shot listeners exit cleanly once the last job resolves). The
 * one-shot listener re-arms itself while jobs are still pending.
 */
function ensureListening(entry: PooledWorker): void {
  if (entry.listening) return;
  entry.listening = true;
  entry.worker.once("message", (msg: { jobId: number }) => {
    entry.listening = false;
    const job = entry.pending.get(msg.jobId);
    if (job !== undefined) {
      entry.pending.delete(msg.jobId);
      job.resolve();
    }
    if (entry.pending.size > 0) ensureListening(entry);
  });
}

export async function matmulParallel(
  a: Float32Array,
  aRows: number,
  aCols: number,
  b: Float32Array,
  bRows: number,
  bCols: number,
  threads?: number,
): Promise<Float32Array> {
  if (aCols !== bRows) {
    throw new Error(
      `matmul: inner dimensions mismatch: a is ${aRows}x${aCols}, b is ${bRows}x${bCols}`,
    );
  }
  if (a.length !== aRows * aCols) {
    throw new Error(`matmul: a has ${a.length} elements, expected ${aRows * aCols}`);
  }
  if (b.length !== bRows * bCols) {
    throw new Error(`matmul: b has ${b.length} elements, expected ${bRows * bCols}`);
  }
  const totalFlops = 2 * aRows * aCols * bCols;
  const available =
    typeof threads === "number" && Number.isFinite(threads) && threads > 0
      ? Math.floor(threads)
      : (os.availableParallelism?.() ?? os.cpus().length);
  const nThreads = Math.max(1, Math.min(available, aRows));
  if (nThreads <= 1 || totalFlops < MATMUL_PARALLEL_MIN_FLOPS) {
    return matmulTiled(a, aRows, aCols, b, bRows, bCols);
  }
  const aSab = new SharedArrayBuffer(a.byteLength);
  new Float32Array(aSab).set(a);
  const bSab = new SharedArrayBuffer(b.byteLength);
  new Float32Array(bSab).set(b);
  const outSab = new SharedArrayBuffer(aRows * bCols * 4);
  const rowsPerWorker = Math.ceil(aRows / nThreads);
  const pool = getMatmulPool(nThreads);
  const jobs: Array<Promise<void>> = [];
  for (let w = 0; w < nThreads; w++) {
    const startRow = w * rowsPerWorker;
    if (startRow >= aRows) break;
    const endRow = Math.min(startRow + rowsPerWorker, aRows);
    const entry = pool[w] as PooledWorker;
    const jobId = ++matmulJobCounter;
    jobs.push(
      new Promise<void>((resolve, reject) => {
        entry.pending.set(jobId, { resolve, reject });
        // Arm the reply listener BEFORE posting, so a fast reply cannot
        // arrive with no listener (all synchronous here, so no race).
        ensureListening(entry);
        entry.worker.postMessage({
          jobId,
          aSab,
          bSab,
          outSab,
          aCols,
          bCols,
          startRow,
          endRow,
        });
      }),
    );
  }
  await Promise.all(jobs);
  return new Float32Array(outSab);
}

/** All CPU fallback kernels, bundled under one object. */
export const CpuKernels = {
  matmul,
  matmulTiled,
  matmulParallel,
  add,
  mul,
  silu,
  rmsNorm,
  rope,
  softmaxRows,
  attention,
  embeddingLookup,
};
