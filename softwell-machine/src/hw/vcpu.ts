/**
 * vCPU — the virtual CPU pool.
 *
 * THE BOUNDARY: workloads submit a {@link CpuTask} and get a
 * {@link CpuTaskResult} back. The task ISA is the interface: `CpuTask.kind`
 * is the opcode, `payload` the operands. A workload never knows which donor
 * ran its task, how many cores that donor has, or where the handler code
 * lives. Donor nodes below the boundary execute tasks (see {@link CpuNode});
 * the coordinator above ({@link VirtualCpu}) only schedules.
 *
 * Approximations (now only for donors that do not report a core count):
 * - Each donor's real core count comes from `capability.cores`
 *   (WorkerCapability.cores, reported by the donor) when present.
 *   `stats().totalCores` sums the reported counts, falling back to
 *   {@link ASSUMED_CORES_PER_NODE} (4) only for donors that do not
 *   report, and `stats()` also reports how many donors reported a
 *   real count (`nodesReportingCores`) vs fell back
 *   (`nodesAssumedCores`). A donor's in-flight limit is likewise
 *   2x its own (reported or assumed) core count, never a global
 *   assumption applied to a donor that reported otherwise.
 * - A donor's real core count is unknowable when it does not report
 *   one, so 4 is a deliberately conservative stand-in for those
 *   donors only; it only affects scheduling pressure and the
 *   reported core count, never correctness.
 *
 * Delivery semantics: at-least-once. A task whose donor dies mid-flight is
 * requeued to a survivor and may therefore run twice (once on the dead
 * donor, once on the survivor). Handlers MUST be idempotent. A task that
 * fails a second time after its one retry is counted lost.
 *
 * Node built-ins only. Erasable TypeScript only.
 */

import { availableParallelism } from "node:os";
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from "node:worker_threads";
import type {
  CpuStats,
  CpuTask,
  CpuTaskHandler,
  CpuTaskResult,
  IVirtualCpu,
  NodeProvider,
  RouteHandler,
  WorkerInfo,
} from "../contracts.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Assumed virtual cores per donor (only when it does not report cores). */
export const ASSUMED_CORES_PER_NODE = 4;

/** A donor is busy at 2x its (reported or assumed) cores in flight. */
export const MAX_IN_FLIGHT_PER_NODE = 2 * ASSUMED_CORES_PER_NODE;

/** Did this donor report a real core count? */
export function nodeReportsCores(info: WorkerInfo): boolean {
  return (
    typeof info.cores === "number" &&
    Number.isFinite(info.cores) &&
    info.cores >= 1
  );
}

/** A donor's core count: reported `capability.cores`, else the assumed 4. */
export function coresForNode(info: WorkerInfo): number {
  return nodeReportsCores(info)
    ? Math.floor(info.cores as number)
    : ASSUMED_CORES_PER_NODE;
}

/** A donor's in-flight limit: 2x its own core count. */
export function maxInFlightForNode(info: WorkerInfo): number {
  return 2 * coresForNode(info);
}

// ---------------------------------------------------------------------------
// "flops" task — a real, measured, verifiable FMA workload
// ---------------------------------------------------------------------------

/**
 * FMA chunk length. The pattern arrays below repeat every
 * {@link FLOPS_CHUNK_LEN} elements, which is what makes the checksum
 * partition-independent: a worker's partial sum depends only on how
 * many multiply-adds it ran, never on where its slice started, so
 * parallel and sequential runs produce the exact same checksum.
 */
export const FLOPS_CHUNK_LEN = 65_536;

/** Default "flops" workload: 2e8 multiply-adds (4e8 FLOPs). */
export const FLOPS_DEFAULT_SIZE = 200_000_000;

/** Largest "flops" workload accepted (1e9 multiply-adds). */
export const FLOPS_MAX_SIZE = 1_000_000_000;

/** Result of a "flops" task: measured, never estimated. */
export interface FlopsResult {
  /** Total floating-point operations performed (2 per multiply-add). */
  flops: number;
  /** Wall time of the FMA loop only, in ms. */
  elapsedMs: number;
  /** Measured throughput: flops / elapsed, in TFLOPS (1e12 flop/s). */
  tflops: number;
  /** Deterministic checksum of the loop (see expectedFlopsChecksum). */
  checksum: number;
  /** Multiply-adds performed (payload `size`). */
  size: number;
  /** Worker threads actually used (1 = sequential). */
  threads: number;
}

let flopsPatternCache: { x: Float64Array; y: Float64Array } | null = null;

/**
 * The FMA pattern: x[j] = (j % 5) + 1, y[j] = (j % 7) + 1 — small
 * integers, so every product and every partial sum is an exact integer
 * in float64 (total < 2^53 for any accepted size) and summation order
 * cannot change the result. The loop is real work: the accumulator is
 * returned as the checksum, so it cannot be optimised away.
 */
function flopsPattern(): { x: Float64Array; y: Float64Array } {
  if (!flopsPatternCache) {
    const x = new Float64Array(FLOPS_CHUNK_LEN);
    const y = new Float64Array(FLOPS_CHUNK_LEN);
    for (let j = 0; j < FLOPS_CHUNK_LEN; j++) {
      x[j] = (j % 5) + 1;
      y[j] = (j % 7) + 1;
    }
    flopsPatternCache = { x, y };
  }
  return flopsPatternCache;
}

/** One chunk's worth of FMA work, as a checksum (exact integer). */
function flopsChunkSum(x: Float64Array, y: Float64Array): number {
  let sum = 0;
  for (let j = 0; j < FLOPS_CHUNK_LEN; j++) sum += x[j] * y[j];
  return sum;
}

/**
 * The checksum a correct "flops" run of `size` multiply-adds must
 * produce, computed from the pattern in O(chunk) time instead of
 * re-running the loop: full chunks x the per-chunk sum, plus the
 * partial chunk. Callers (bench, tests) use this to verify a result
 * without trusting the node that produced it.
 */
export function expectedFlopsChecksum(size: number): number {
  if (!Number.isInteger(size) || size < 1 || size > FLOPS_MAX_SIZE) {
    throw new Error(`size must be an integer in 1..${FLOPS_MAX_SIZE}`);
  }
  const { x, y } = flopsPattern();
  const fullChunks = Math.floor(size / FLOPS_CHUNK_LEN);
  const remainder = size % FLOPS_CHUNK_LEN;
  let checksum = fullChunks * flopsChunkSum(x, y);
  for (let j = 0; j < remainder; j++) checksum += x[j] * y[j];
  return checksum;
}

/** Run `size` multiply-adds over the pattern arrays; returns the checksum. */
function runFlopsLoop(
  size: number,
  x: Float64Array,
  y: Float64Array,
): number {
  let acc = 0;
  let remaining = size;
  while (remaining >= FLOPS_CHUNK_LEN) {
    for (let j = 0; j < FLOPS_CHUNK_LEN; j++) acc += x[j] * y[j];
    remaining -= FLOPS_CHUNK_LEN;
  }
  for (let j = 0; j < remaining; j++) acc += x[j] * y[j];
  return acc;
}

interface FlopsWorkerData {
  __softwellFlopsWorker: true;
  sabX: SharedArrayBuffer;
  sabY: SharedArrayBuffer;
  sabPartial: SharedArrayBuffer;
  index: number;
  count: number;
}

// Worker-thread entry point: this same module is re-executed as the
// worker (new Worker(new URL(import.meta.url))). When spawned that way
// with flops workerData, compute one slice, publish the partial sum
// into the SharedArrayBuffer, report it, and let the thread exit.
// The guard checks the marker so ordinary imports — including any
// test runner that happens to load this file in a worker — fall
// through to the normal module definitions below.
if (!isMainThread) {
  const wd = workerData as Partial<FlopsWorkerData> | undefined;
  if (wd && wd.__softwellFlopsWorker === true && parentPort) {
    const x = Float64Array.from(new Float64Array(wd.sabX as SharedArrayBuffer));
    const y = Float64Array.from(new Float64Array(wd.sabY as SharedArrayBuffer));
    const partial = runFlopsLoop(wd.count as number, x, y);
    new Float64Array(wd.sabPartial as SharedArrayBuffer)[wd.index as number] =
      partial;
    parentPort.postMessage({ index: wd.index, checksum: partial });
  }
}

/**
 * Parallel FMA: split `size` into `threads` chunk-aligned slices (each
 * slice starts on a chunk boundary, so the pattern phase matches the
 * sequential run exactly), run each slice in a worker thread over
 * SharedArrayBuffers holding the pattern, and sum the partial
 * checksums in slice order. Falls back to the sequential loop if
 * worker threads cannot be started — the checksum is identical either
 * way, and the returned `threadsUsed` says which path actually ran.
 */
async function runFlopsParallel(
  size: number,
  threads: number,
): Promise<{ checksum: number; threadsUsed: number }> {
  const fullChunks = Math.floor(size / FLOPS_CHUNK_LEN);
  const effectiveThreads = Math.max(
    1,
    Math.min(threads, availableParallelism(), fullChunks),
  );
  if (effectiveThreads <= 1) {
    const { x, y } = flopsPattern();
    return { checksum: runFlopsLoop(size, x, y), threadsUsed: 1 };
  }
  const { x, y } = flopsPattern();
  const sabX = new SharedArrayBuffer(FLOPS_CHUNK_LEN * 8);
  const sabY = new SharedArrayBuffer(FLOPS_CHUNK_LEN * 8);
  new Float64Array(sabX).set(x);
  new Float64Array(sabY).set(y);
  const sabPartial = new SharedArrayBuffer(effectiveThreads * 8);

  const remainder = size % FLOPS_CHUNK_LEN;
  const baseChunks = Math.floor(fullChunks / effectiveThreads);
  const extraChunks = fullChunks % effectiveThreads;
  const workers: Worker[] = [];
  try {
    const jobs: Array<Promise<{ index: number; checksum: number }>> = [];
    let chunkCursor = 0;
    for (let i = 0; i < effectiveThreads; i++) {
      const chunks = baseChunks + (i < extraChunks ? 1 : 0);
      // Slice starts are chunk-aligned, so the worker's pattern phase
      // matches the sequential run; slice 0 also takes the remainder.
      let count = chunks * FLOPS_CHUNK_LEN;
      if (i === 0) count += remainder;
      chunkCursor += chunks;
      const workerDataPayload: FlopsWorkerData = {
        __softwellFlopsWorker: true,
        sabX,
        sabY,
        sabPartial,
        index: i,
        count,
      };
      const worker = new Worker(new URL(import.meta.url), {
        workerData: workerDataPayload,
      });
      workers.push(worker);
      jobs.push(
        new Promise((resolve, reject) => {
          worker.once("message", (msg: unknown) => {
            const m = msg as { index: number; checksum: number };
            resolve({ index: m.index, checksum: m.checksum });
          });
          worker.once("error", reject);
          worker.once("exit", (code) => {
            if (code !== 0) {
              reject(new Error(`flops worker exited with code ${code}`));
            }
          });
        }),
      );
    }
    await Promise.all(jobs);
    const partials = new Float64Array(sabPartial);
    let checksum = 0;
    for (let i = 0; i < effectiveThreads; i++) checksum += partials[i] as number;
    return { checksum, threadsUsed: effectiveThreads };
  } catch {
    // Worker threads unavailable/broken in this environment: the
    // sequential loop gives the same checksum, just on one thread.
    for (const w of workers) {
      void w.terminate().catch(() => undefined);
    }
    return { checksum: runFlopsLoop(size, x, y), threadsUsed: 1 };
  } finally {
    for (const w of workers) {
      void w.terminate().catch(() => undefined);
    }
  }
}

/** How long a donor's advertised-kind list is cached (refreshed on use). */
export const KIND_CACHE_TTL_MS = 30_000;

/** Work-stealing rebalancer interval. */
export const REBALANCE_INTERVAL_MS = 250;

/** Per-dispatch fetch timeout (a donor that hangs this long is retried). */
const DISPATCH_TIMEOUT_MS = 30_000;

/** Node id reported by a node component for results it produced itself. */
const LOCAL_NODE_ID = "local";

// ---------------------------------------------------------------------------
// CpuNode — the donor-side component (runs on the worker)
// ---------------------------------------------------------------------------

export interface CpuNodeOpts {
  // No dependencies: a CpuNode is pure handler map + route table.
}

function payloadObject(task: CpuTask): Record<string, unknown> {
  if (
    typeof task.payload !== "object" ||
    task.payload === null ||
    Array.isArray(task.payload)
  ) {
    throw new Error(`kind ${task.kind}: payload must be an object`);
  }
  return task.payload as Record<string, unknown>;
}

function asNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function asString(value: unknown, name: string): string {
  if (typeof value !== "string") {
    throw new Error(`${name} must be a string`);
  }
  return value;
}

function asNumberArray(value: unknown, name: string): number[] {
  if (
    !Array.isArray(value) ||
    value.some((v) => typeof v !== "number" || !Number.isFinite(v))
  ) {
    throw new Error(`${name} must be an array of finite numbers`);
  }
  return value as number[];
}

/** mulberry32: small fast seeded PRNG (deterministic for a given seed). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder("utf-8", { fatal: false });

function echoHandler(task: CpuTask): unknown {
  return task.payload;
}

function addHandler(task: CpuTask): number {
  const p = payloadObject(task);
  return asNumber(p["a"], "a") + asNumber(p["b"], "b");
}

function mulHandler(task: CpuTask): number {
  const p = payloadObject(task);
  return asNumber(p["a"], "a") * asNumber(p["b"], "b");
}

/** Byte-level tokenize: UTF-8 byte ids (0..255), no BOS/EOS added. */
function tokenizeHandler(task: CpuTask): number[] {
  const p = payloadObject(task);
  const text = asString(p["text"], "text");
  return [...utf8Encoder.encode(text)];
}

/** Inverse of tokenize: byte ids -> string. */
function detokenizeHandler(task: CpuTask): string {
  const p = payloadObject(task);
  const ids = asNumberArray(p["ids"], "ids");
  for (const id of ids) {
    if (!Number.isInteger(id) || id < 0 || id > 255) {
      throw new Error("ids must be integers in 0..255");
    }
  }
  return utf8Decoder.decode(new Uint8Array(ids));
}

/**
 * Sample one token id from logits.
 * Greedy (argmax, first max on ties) when temperature <= 0; otherwise
 * temperature scaling + top-k / top-p filtering, sampled with mulberry32
 * when a seed is given (deterministic) or Math.random when it is not.
 */
function sampleHandler(task: CpuTask): number {
  const p = payloadObject(task);
  const logits = asNumberArray(p["logits"], "logits");
  if (logits.length === 0) {
    throw new Error("logits must be a non-empty array");
  }
  const temperature =
    p["temperature"] === undefined
      ? 1
      : asNumber(p["temperature"], "temperature");
  const topK =
    p["topK"] === undefined ? 0 : asNumber(p["topK"], "topK");
  const topP =
    p["topP"] === undefined ? 1 : asNumber(p["topP"], "topP");
  const seed =
    p["seed"] === undefined ? undefined : asNumber(p["seed"], "seed");

  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }
  if (topP <= 0 || topP > 1) {
    throw new Error("topP must be in (0, 1]");
  }

  // Indices sorted by logit, descending.
  const order = logits.map((_v, i) => i).sort((a, b) => logits[b] - logits[a]);
  let kept = order;
  if (topK > 0) {
    kept = order.slice(0, Math.min(Math.floor(topK), order.length));
  }

  // Softmax over the kept (temperature-scaled) logits.
  let max = -Infinity;
  for (const i of kept) max = Math.max(max, logits[i] / temperature);
  const weights = kept.map((i) => Math.exp(logits[i] / temperature - max));
  const total = weights.reduce((sum, w) => sum + w, 0);
  let probs = weights.map((w) => w / total);

  if (topP < 1) {
    let cumulative = 0;
    let cutoff = probs.length;
    for (let j = 0; j < probs.length; j++) {
      cumulative += probs[j];
      if (cumulative >= topP) {
        cutoff = j + 1;
        break;
      }
    }
    kept = kept.slice(0, cutoff);
    probs = probs.slice(0, cutoff);
    const subtotal = probs.reduce((sum, x) => sum + x, 0);
    probs = probs.map((x) => x / subtotal);
  }

  const rng = seed === undefined ? Math.random : mulberry32(seed);
  const r = rng();
  let acc = 0;
  for (let j = 0; j < kept.length; j++) {
    acc += probs[j];
    if (r < acc) return kept[j];
  }
  return kept[kept.length - 1];
}

/** FNV-1a (32-bit) over the UTF-8 bytes, as an 8-char lowercase hex string. */
function hashHandler(task: CpuTask): string {
  const p = payloadObject(task);
  const text = asString(p["text"], "text");
  let hash = 0x811c9dc5;
  for (const byte of utf8Encoder.encode(text)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

/**
 * Measured FMA workload. Payload: `{ size, threads? }` where `size` is
 * the number of multiply-adds (default {@link FLOPS_DEFAULT_SIZE}) and
 * `threads` the worker threads to spread them over (default 1 =
 * sequential). Returns {@link FlopsResult}: the measured TFLOPS plus
 * the checksum, so any caller can verify the work was really done
 * with {@link expectedFlopsChecksum} — a wrong count or a skipped
 * loop changes the checksum. Parallel (`threads` > 1) runs only when
 * the payload asks for it and the size is at least two chunks; the
 * checksum is identical to the sequential run by construction.
 */
async function flopsHandler(task: CpuTask): Promise<FlopsResult> {
  const p = payloadObject(task);
  const size =
    p["size"] === undefined
      ? FLOPS_DEFAULT_SIZE
      : asNumber(p["size"], "size");
  if (!Number.isInteger(size) || size < 1 || size > FLOPS_MAX_SIZE) {
    throw new Error(`size must be an integer in 1..${FLOPS_MAX_SIZE}`);
  }
  const threads =
    p["threads"] === undefined ? 1 : asNumber(p["threads"], "threads");
  if (!Number.isInteger(threads) || threads < 1 || threads > 64) {
    throw new Error("threads must be an integer in 1..64");
  }

  const started = performance.now();
  let checksum: number;
  let threadsUsed: number;
  if (threads > 1 && size >= 2 * FLOPS_CHUNK_LEN) {
    const parallel = await runFlopsParallel(size, threads);
    checksum = parallel.checksum;
    threadsUsed = parallel.threadsUsed;
  } else {
    const { x, y } = flopsPattern();
    checksum = runFlopsLoop(size, x, y);
    threadsUsed = 1;
  }
  const elapsedMs = performance.now() - started;
  const flops = 2 * size;
  return {
    flops,
    elapsedMs,
    tflops: elapsedMs > 0 ? flops / (elapsedMs / 1000) / 1e12 : 0,
    checksum,
    size,
    threads: threadsUsed,
  };
}

export class CpuNode {
  private handlers = new Map<string, CpuTaskHandler>();
  private completed = 0;
  private failed = 0;

  constructor(_opts: CpuNodeOpts = {}) {
    this.handlers.set("echo", echoHandler);
    this.handlers.set("add", addHandler);
    this.handlers.set("mul", mulHandler);
    this.handlers.set("tokenize", tokenizeHandler);
    this.handlers.set("detokenize", detokenizeHandler);
    this.handlers.set("sample", sampleHandler);
    this.handlers.set("hash", hashHandler);
    this.handlers.set("flops", flopsHandler);
  }

  /** Worker-side extension: add (or replace) a handler for a kind. */
  registerHandler(kind: string, handler: CpuTaskHandler): void {
    this.handlers.set(kind, handler);
  }

  /** Kinds this node advertises (its handler map keys). */
  listKinds(): string[] {
    return [...this.handlers.keys()].sort();
  }

  /** Execute one task through the handler map. */
  async execute(task: CpuTask): Promise<CpuTaskResult> {
    const started = Date.now();
    if (
      typeof task !== "object" ||
      task === null ||
      typeof task.kind !== "string"
    ) {
      this.failed++;
      return {
        taskId: "unknown",
        ok: false,
        error: "task must be an object with a string kind",
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      };
    }
    const handler = this.handlers.get(task.kind);
    if (!handler) {
      this.failed++;
      return {
        taskId: task.id,
        ok: false,
        error: `unknown kind ${task.kind}`,
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      };
    }
    try {
      const result = await handler(task);
      this.completed++;
      return {
        taskId: task.id,
        ok: true,
        result,
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      };
    } catch (err) {
      this.failed++;
      return {
        taskId: task.id,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      };
    }
  }

  /** Route table mounted by the worker's hw node server (see node-server). */
  routes(): Record<string, RouteHandler> {
    return {
      "/hw/cpu/task": async (body) => {
        const raw = body["task"];
        if (typeof raw !== "object" || raw === null) {
          return {
            taskId: "unknown",
            ok: false,
            error: "POST /hw/cpu/task: body.task must be an object",
            nodeId: LOCAL_NODE_ID,
            elapsedMs: 0,
          };
        }
        return this.execute(raw as CpuTask);
      },
      "/hw/cpu/kinds": async () => ({ kinds: this.listKinds() }),
      "/hw/cpu/stats": async () => ({
        completed: this.completed,
        failed: this.failed,
      }),
    };
  }
}

// ---------------------------------------------------------------------------
// VirtualCpu — the orchestrator-side coordinator (scheduling only)
// ---------------------------------------------------------------------------

export interface VirtualCpuOpts {
  nodes: NodeProvider;
}

interface PendingTask {
  task: CpuTask;
  resolve: (result: CpuTaskResult) => void;
  /** Failed dispatch attempts already consumed (max one retry). */
  attempts: number;
  state: "queued" | "in-flight" | "done";
  nodeId: string | undefined;
  abort: AbortController | undefined;
  enqueuedAt: number;
}

interface NodeState {
  info: WorkerInfo;
  /** Advertised kinds; null = not fetched yet (or invalidated). */
  kinds: Set<string> | null;
  kindsFetchedAt: number;
  kindsFetch: Promise<void> | null;
  inFlight: number;
  inFlightTasks: Set<PendingTask>;
  queue: PendingTask[];
}

/**
 * CpuStats plus core-reporting provenance. `stats()` returns this:
 * the base fields are unchanged, and the two additive fields say how
 * many donors reported a real core count vs fell back to the assumed
 * 4. (CpuStats itself lives in contracts.ts, which this workstream
 * does not own, so the additive fields live here as a subtype.)
 */
export interface CpuStatsWithCoreReporting extends CpuStats {
  /** Donors whose `capability.cores` was used for totalCores. */
  nodesReportingCores: number;
  /** Donors that did not report cores and fell back to the assumed 4. */
  nodesAssumedCores: number;
}

export class VirtualCpu implements IVirtualCpu {
  private nodes: NodeProvider;
  private localHandlers = new Map<string, CpuTaskHandler>();
  private nodeStates = new Map<string, NodeState>();
  /** Tombstoned by handleNodeLoss: never resurrected from a stale provider. */
  private removedNodes = new Set<string>();
  private globalQueue: PendingTask[] = [];
  private completed = 0;
  private failed = 0;
  private rebalanceTimer: ReturnType<typeof setInterval> | null;

  constructor(opts: VirtualCpuOpts) {
    this.nodes = opts.nodes;
    this.rebalanceTimer = setInterval(() => {
      this.rebalance();
      this.pump();
    }, REBALANCE_INTERVAL_MS);
    // Don't keep the process alive just for the rebalancer.
    this.rebalanceTimer.unref();
  }

  /**
   * Register a LOCAL fallback handler, run in the orchestrator process when
   * no donor advertises the kind. Donors are always preferred when one
   * advertises the kind.
   */
  registerHandler(kind: string, handler: CpuTaskHandler): void {
    this.localHandlers.set(kind, handler);
  }

  submit(task: CpuTask): Promise<CpuTaskResult> {
    return new Promise((resolve) => {
      const pending: PendingTask = {
        task,
        resolve,
        attempts: 0,
        state: "queued",
        nodeId: undefined,
        abort: undefined,
        enqueuedAt: Date.now(),
      };
      this.placeTask(pending);
    });
  }

  /**
   * Heal after a donor dies: requeue its queued + in-flight tasks to
   * survivors (at-least-once — handlers MUST be idempotent). A task that
   * had already used its one retry is counted lost instead of requeued.
   */
  async handleNodeLoss(
    nodeId: string,
  ): Promise<{ tasksRequeued: number; tasksLost: number }> {
    this.syncNodes();
    const node = this.nodeStates.get(nodeId);
    if (!node) {
      return { tasksRequeued: 0, tasksLost: 0 };
    }
    this.nodeStates.delete(nodeId);
    this.removedNodes.add(nodeId);

    let tasksRequeued = 0;
    let tasksLost = 0;
    const victims: PendingTask[] = [...node.queue, ...node.inFlightTasks];
    for (const pending of victims) {
      if (pending.state === "done") continue;
      const wasInFlight = pending.state === "in-flight";
      // Mark queued BEFORE aborting: the dispatch catch below sees the
      // state change and knows this task is already being handled here.
      pending.state = "queued";
      pending.nodeId = undefined;
      const abort = pending.abort;
      pending.abort = undefined;
      if (abort) abort.abort();
      if (wasInFlight && pending.attempts >= 1) {
        pending.state = "done";
        tasksLost++;
        this.failed++;
        pending.resolve({
          taskId: pending.task.id,
          ok: false,
          error: `cpu task lost: node ${nodeId} died after retry`,
          nodeId,
          elapsedMs: Date.now() - pending.enqueuedAt,
        });
      } else {
        // Requeueing an in-flight task consumes its one retry; a merely
        // queued task has not been attempted yet.
        if (wasInFlight) pending.attempts++;
        this.globalQueue.push(pending);
        tasksRequeued++;
      }
    }
    this.pump();
    return { tasksRequeued, tasksLost };
  }

  stats(): CpuStatsWithCoreReporting {
    this.syncNodes();
    let queued = this.globalQueue.length;
    for (const node of this.nodeStates.values()) {
      queued += node.queue.length;
    }
    let totalCores = 0;
    let nodesReportingCores = 0;
    for (const node of this.nodeStates.values()) {
      totalCores += coresForNode(node.info);
      if (nodeReportsCores(node.info)) nodesReportingCores++;
    }
    return {
      nodes: this.nodeStates.size,
      totalCores,
      queued,
      completed: this.completed,
      failed: this.failed,
      nodesReportingCores,
      nodesAssumedCores: this.nodeStates.size - nodesReportingCores,
    };
  }

  /**
   * Per-node queued depths, plus the global overflow queue under "global".
   * Test/diagnostics inspection (in-flight tasks are not queued).
   */
  queueDepths(): Record<string, number> {
    this.syncNodes();
    const depths: Record<string, number> = {};
    for (const [id, node] of this.nodeStates) {
      depths[id] = node.queue.length;
    }
    depths["global"] = this.globalQueue.length;
    return depths;
  }

  /**
   * Work stealing: move half the queued tasks from the busiest node's
   * queue to an idle node's queue (idle = empty queue + no in-flight).
   * No-op when there is no busiest queue of >= 2 or no idle node.
   * Does not dispatch; call pump() (or wait for the timer) to run them.
   */
  rebalance(): void {
    this.syncNodes();
    const nodes = [...this.nodeStates.values()];
    if (nodes.length < 2) return;
    let busiest: NodeState | undefined;
    for (const node of nodes) {
      if (!busiest || node.queue.length > busiest.queue.length) {
        busiest = node;
      }
    }
    if (!busiest || busiest.queue.length < 2) return;
    const idle = nodes.find(
      (n) => n !== busiest && n.queue.length === 0 && n.inFlight === 0,
    );
    if (!idle) return;
    const moveCount = Math.floor(busiest.queue.length / 2);
    if (moveCount < 1) return;
    const moved = busiest.queue.splice(0, moveCount);
    for (const pending of moved) {
      pending.nodeId = idle.info.workerId;
      idle.queue.push(pending);
    }
  }

  /**
   * Drive dispatch: run per-node queues up to the concurrency limit, then
   * assign global-overflow tasks to donors (or the local fallback).
   * Normally called automatically (submit, dispatch completion, timer).
   */
  pump(): void {
    this.syncNodes();
    for (const node of this.nodeStates.values()) {
      while (node.queue.length > 0 && node.inFlight < maxInFlightForNode(node.info)) {
        const pending = node.queue.shift();
        if (!pending) break;
        void this.dispatchToNode(node, pending);
      }
    }
    if (this.globalQueue.length > 0) {
      this.drainGlobalQueue();
    }
  }

  /**
   * Test seam: place a task directly on a node's queue without
   * dispatching it. Lets a test build a known queue shape (e.g. a busy
   * node + an idle node) and then exercise rebalance()/handleNodeLoss()
   * deterministically. Call pump() afterwards to run the queued tasks.
   */
  enqueueOnNode(nodeId: string, task: CpuTask): Promise<CpuTaskResult> {
    this.syncNodes();
    const node = this.nodeStates.get(nodeId);
    if (!node) {
      throw new Error(`unknown cpu node ${nodeId}`);
    }
    return new Promise((resolve) => {
      const pending: PendingTask = {
        task,
        resolve,
        attempts: 0,
        state: "queued",
        nodeId,
        abort: undefined,
        enqueuedAt: Date.now(),
      };
      node.queue.push(pending);
    });
  }

  /** Stop the periodic rebalancer (lets a test process exit cleanly). */
  close(): void {
    if (this.rebalanceTimer) {
      clearInterval(this.rebalanceTimer);
      this.rebalanceTimer = null;
    }
  }

  // -- internals ------------------------------------------------------------

  private syncNodes(): void {
    let list: WorkerInfo[];
    try {
      list = this.nodes.listNodes();
    } catch {
      return;
    }
    for (const info of list) {
      if (this.removedNodes.has(info.workerId)) continue;
      const existing = this.nodeStates.get(info.workerId);
      if (!existing) {
        this.nodeStates.set(info.workerId, {
          info,
          kinds: null,
          kindsFetchedAt: 0,
          kindsFetch: null,
          inFlight: 0,
          inFlightTasks: new Set(),
          queue: [],
        });
      } else {
        existing.info = info;
      }
    }
  }

  private candidatesFor(kind: string): NodeState[] {
    return [...this.nodeStates.values()]
      .filter((n) => n.kinds !== null && n.kinds.has(kind))
      .sort(
        (a, b) =>
          a.inFlight - b.inFlight || a.queue.length - b.queue.length,
      );
  }

  private placeTask(pending: PendingTask): void {
    this.syncNodes();
    const candidates = this.candidatesFor(pending.task.kind);
    if (candidates.length > 0) {
      const best = candidates[0];
      if (best.inFlight < maxInFlightForNode(best.info)) {
        best.queue.push(pending);
        pending.nodeId = best.info.workerId;
      } else {
        // All donors advertising the kind are busy: global overflow.
        this.globalQueue.push(pending);
      }
    } else if (
      this.nodeStates.size === 0 ||
      [...this.nodeStates.values()].some((n) => n.kinds === null)
    ) {
      // Kind support not yet known (or no donors at all): wait in the
      // global queue while kind discovery runs.
      this.globalQueue.push(pending);
      void this.ensureKinds().then(() => this.pump());
    } else if (this.localHandlers.has(pending.task.kind)) {
      void this.runLocal(pending);
    } else {
      this.failTask(
        pending,
        `unknown kind ${pending.task.kind}`,
        LOCAL_NODE_ID,
      );
    }
    this.pump();
  }

  private drainGlobalQueue(): void {
    const snapshot = [...this.globalQueue];
    for (const pending of snapshot) {
      if (pending.state !== "queued") continue;
      const candidates = this.candidatesFor(pending.task.kind);
      if (candidates.length > 0) {
        const best = candidates[0];
        if (best.inFlight < maxInFlightForNode(best.info)) {
          this.globalQueue = this.globalQueue.filter((p) => p !== pending);
          void this.dispatchToNode(best, pending);
        }
        // Else: all advertisers busy — stays in the global overflow queue.
        continue;
      }
      if ([...this.nodeStates.values()].some((n) => n.kinds === null)) {
        void this.ensureKinds().then(() => this.pump());
        continue;
      }
      if (this.localHandlers.has(pending.task.kind)) {
        this.globalQueue = this.globalQueue.filter((p) => p !== pending);
        void this.runLocal(pending);
        continue;
      }
      if (this.nodeStates.size === 0) {
        // No donors at all: keep waiting for one to appear.
        continue;
      }
      // Donors are known and none advertises the kind: fail fast.
      this.globalQueue = this.globalQueue.filter((p) => p !== pending);
      this.failTask(pending, `unknown kind ${pending.task.kind}`, LOCAL_NODE_ID);
    }
  }

  private async dispatchToNode(
    node: NodeState,
    pending: PendingTask,
  ): Promise<void> {
    node.inFlight++;
    node.inFlightTasks.add(pending);
    pending.state = "in-flight";
    pending.nodeId = node.info.workerId;
    const controller = new AbortController();
    pending.abort = controller;
    const started = Date.now();
    const timeout = setTimeout(() => controller.abort(), DISPATCH_TIMEOUT_MS);
    try {
      if (node.info.hwPort === undefined) {
        throw new Error(`node ${node.info.workerId} has no hwPort`);
      }
      const res = await fetch(
        `http://${node.info.host}:${node.info.hwPort}/hw/cpu/task`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ task: pending.task }),
          signal: controller.signal,
        },
      );
      if (!res.ok) {
        throw new Error(`POST /hw/cpu/task HTTP ${res.status}`);
      }
      const data = (await res.json()) as Partial<CpuTaskResult>;
      if (pending.state !== "in-flight") {
        // Node loss already requeued/lost this task; do not resolve twice.
        return;
      }
      pending.state = "done";
      pending.abort = undefined;
      node.inFlight--;
      node.inFlightTasks.delete(pending);
      if (data.ok === true) {
        this.completed++;
      } else {
        this.failed++;
      }
      pending.resolve({
        taskId: pending.task.id,
        ok: data.ok === true,
        result: data.result,
        error: typeof data.error === "string" ? data.error : undefined,
        nodeId: node.info.workerId,
        elapsedMs: Date.now() - started,
      });
      this.pump();
      if (node.queue.length === 0 && node.inFlight === 0) {
        // This node just drained: give it a chance to steal work.
        this.rebalance();
        this.pump();
      }
    } catch (err) {
      if (pending.state !== "in-flight") {
        // Already handled by handleNodeLoss (which aborted this fetch).
        return;
      }
      pending.state = "queued";
      pending.abort = undefined;
      node.inFlight--;
      node.inFlightTasks.delete(pending);
      // Dispatch failure invalidates the kind cache: re-discover on retry.
      node.kinds = null;
      node.kindsFetchedAt = 0;
      const message = err instanceof Error ? err.message : String(err);
      if (pending.attempts < 1) {
        pending.attempts++;
        pending.nodeId = undefined;
        this.globalQueue.push(pending);
        void this.ensureKinds().then(() => this.pump());
        this.pump();
      } else {
        pending.state = "done";
        this.failed++;
        pending.resolve({
          taskId: pending.task.id,
          ok: false,
          error: `cpu task failed after retry: ${message}`,
          nodeId: node.info.workerId,
          elapsedMs: Date.now() - started,
        });
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  private async runLocal(pending: PendingTask): Promise<void> {
    const handler = this.localHandlers.get(pending.task.kind);
    const started = Date.now();
    if (!handler) {
      this.failTask(pending, `unknown kind ${pending.task.kind}`, LOCAL_NODE_ID);
      return;
    }
    try {
      const result = await handler(pending.task);
      if (pending.state === "done") return;
      pending.state = "done";
      this.completed++;
      pending.resolve({
        taskId: pending.task.id,
        ok: true,
        result,
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      });
    } catch (err) {
      if (pending.state === "done") return;
      pending.state = "done";
      this.failed++;
      pending.resolve({
        taskId: pending.task.id,
        ok: false,
        error: err instanceof Error ? err.message : String(err),
        nodeId: LOCAL_NODE_ID,
        elapsedMs: Date.now() - started,
      });
    }
    this.pump();
  }

  private failTask(pending: PendingTask, error: string, nodeId: string): void {
    if (pending.state === "done") return;
    pending.state = "done";
    this.failed++;
    pending.resolve({
      taskId: pending.task.id,
      ok: false,
      error,
      nodeId,
      elapsedMs: Date.now() - pending.enqueuedAt,
    });
  }

  private ensureKinds(): Promise<void> {
    const jobs: Promise<void>[] = [];
    for (const node of this.nodeStates.values()) {
      if (
        node.kinds === null ||
        Date.now() - node.kindsFetchedAt > KIND_CACHE_TTL_MS
      ) {
        jobs.push(this.fetchKinds(node));
      }
    }
    return Promise.all(jobs).then(() => undefined);
  }

  private fetchKinds(node: NodeState): Promise<void> {
    if (node.kindsFetch) return node.kindsFetch;
    const job = (async () => {
      try {
        if (node.info.hwPort === undefined) {
          node.kinds = new Set();
          node.kindsFetchedAt = Date.now();
          return;
        }
        const res = await fetch(
          `http://${node.info.host}:${node.info.hwPort}/hw/cpu/kinds`,
          { signal: AbortSignal.timeout(5000) },
        );
        if (!res.ok) {
          throw new Error(`GET /hw/cpu/kinds HTTP ${res.status}`);
        }
        const data = (await res.json()) as { kinds?: unknown };
        const kinds = Array.isArray(data.kinds)
          ? data.kinds.filter((k): k is string => typeof k === "string")
          : [];
        node.kinds = new Set(kinds);
        node.kindsFetchedAt = Date.now();
      } catch {
        // Unreachable donor: advertise nothing for this cache window.
        // A real dispatch failure invalidates the cache sooner.
        node.kinds = new Set();
        node.kindsFetchedAt = Date.now();
      } finally {
        node.kindsFetch = null;
      }
    })();
    node.kindsFetch = job;
    return job;
  }
}

// ---------------------------------------------------------------------------
// Throughput measurement (used by src/bench/bench-vcpu.ts)
// ---------------------------------------------------------------------------

export interface FlopsThroughputOptions {
  /** Independent "flops" tasks to submit concurrently. */
  taskCount: number;
  /** Multiply-adds per task. */
  size: number;
  /** Worker threads per task (default 1 = sequential in the node). */
  threads?: number;
  /** Prefix for the submitted task ids (default "bench-flops"). */
  idPrefix?: string;
}

export interface FlopsThroughputResult {
  tasks: number;
  completed: number;
  failed: number;
  /** Every completed task's checksum matched expectedFlopsChecksum(size). */
  checksumOk: boolean;
  /** Wall time from first submit to last completion, in ms. */
  elapsedMs: number;
  /** Completed tasks per wall-clock second. */
  tasksPerSec: number;
  /** Sum of the completed tasks' measured FLOPs. */
  totalFlops: number;
  /** Aggregate throughput: totalFlops / wall time, in TFLOPS. */
  aggregateTflops: number;
  /** Sum of the per-task measured TFLOPS (each task's own flops/elapsed). */
  sumPerTaskTflops: number;
  perTask: Array<{
    taskId: string;
    tflops: number;
    elapsedMs: number;
    nodeId: string;
  }>;
}

function asFlopsResult(value: unknown): FlopsResult | null {
  if (typeof value !== "object" || value === null) return null;
  const r = value as Record<string, unknown>;
  if (
    typeof r["flops"] !== "number" ||
    typeof r["elapsedMs"] !== "number" ||
    typeof r["tflops"] !== "number" ||
    typeof r["checksum"] !== "number"
  ) {
    return null;
  }
  return {
    flops: r["flops"] as number,
    elapsedMs: r["elapsedMs"] as number,
    tflops: r["tflops"] as number,
    checksum: r["checksum"] as number,
    size: typeof r["size"] === "number" ? (r["size"] as number) : 0,
    threads: typeof r["threads"] === "number" ? (r["threads"] as number) : 1,
  };
}

/**
 * Submit `taskCount` independent "flops" tasks concurrently through
 * `cpu`, time the whole batch, and verify every checksum. Aggregate
 * TFLOPS is total measured FLOPs divided by wall time (not the sum
 * of per-task rates, which would double-count overlapped time);
 * `sumPerTaskTflops` is also returned so the two can be compared.
 * A checksum mismatch is reported as `checksumOk: false` — callers
 * that benchmark must refuse to report the TFLOPS, not silently
 * publish work that did not verify.
 */
export async function measureFlopsThroughput(
  cpu: IVirtualCpu,
  opts: FlopsThroughputOptions,
): Promise<FlopsThroughputResult> {
  const { taskCount, size } = opts;
  if (!Number.isInteger(taskCount) || taskCount < 1) {
    throw new Error("taskCount must be a positive integer");
  }
  if (!Number.isInteger(size) || size < 1 || size > FLOPS_MAX_SIZE) {
    throw new Error(`size must be an integer in 1..${FLOPS_MAX_SIZE}`);
  }
  const threads = opts.threads ?? 1;
  const idPrefix = opts.idPrefix ?? "bench-flops";
  const expectedChecksum = expectedFlopsChecksum(size);

  const started = performance.now();
  const results = await Promise.all(
    Array.from({ length: taskCount }, (_v, i) =>
      cpu.submit({
        id: `${idPrefix}-${i}`,
        kind: "flops",
        payload: { size, threads },
      }),
    ),
  );
  const elapsedMs = performance.now() - started;

  let completed = 0;
  let failed = 0;
  let checksumOk = true;
  let totalFlops = 0;
  let sumPerTaskTflops = 0;
  const perTask: FlopsThroughputResult["perTask"] = [];
  for (const r of results) {
    if (!r.ok) {
      failed++;
      continue;
    }
    const flopsResult = asFlopsResult(r.result);
    if (!flopsResult || flopsResult.checksum !== expectedChecksum) {
      checksumOk = false;
      failed++;
      continue;
    }
    completed++;
    totalFlops += flopsResult.flops;
    sumPerTaskTflops += flopsResult.tflops;
    perTask.push({
      taskId: r.taskId,
      tflops: flopsResult.tflops,
      elapsedMs: flopsResult.elapsedMs,
      nodeId: r.nodeId,
    });
  }

  return {
    tasks: taskCount,
    completed,
    failed,
    checksumOk,
    elapsedMs,
    tasksPerSec: elapsedMs > 0 ? completed / (elapsedMs / 1000) : 0,
    totalFlops,
    aggregateTflops:
      elapsedMs > 0 ? totalFlops / (elapsedMs / 1000) / 1e12 : 0,
    sumPerTaskTflops,
    perTask,
  };
}
