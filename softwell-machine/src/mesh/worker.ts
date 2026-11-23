import { fileURLToPath } from "node:url";
import os from "node:os";
import type { Server } from "node:http";
import { createReferenceBackend } from "../engine/backend.ts";
import { parseEngineFlags } from "../engine/options.ts";
import { openWeightLoader } from "../model/loader.ts";
import { VramWeightLoader } from "../model/vram-loader.ts";
import { BlockPagedKVCache } from "../scheduler/kvcache.ts";
import {
  DEFAULT_ORCHESTRATOR_PORT,
  HEARTBEAT_INTERVAL_MS,
} from "../contracts.ts";
import type {
  EngineBackend,
  EngineOptions,
  ExecuteRequest,
  ExecuteResponse,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
  WorkerCapability,
  WorkerDevice,
} from "../contracts.ts";
import { MeshClient, MeshError, createMeshServer } from "./protocol.ts";

/**
 * Worker process: registers with the orchestrator, serves POST /mesh/execute
 * on its own port, owns one StageExecutor plus a worker-local PagedKVCache,
 * and heartbeats until close().
 *
 * The PagedKVCache is built by an injected factory (`createCache`) — the
 * caller passes the scheduler's implementation. There is deliberately no
 * in-file fallback cache: a worker without a real cache must fail loudly
 * at construction time rather than silently run with a toy one.
 */

export interface StartWorkerOptions {
  orchestratorUrl: string;
  host: string;
  port: number;
  capability: Omit<WorkerCapability, "host" | "port">;
  backend: EngineBackend;
  loader: WeightLoader;
  /** Engine options for the stage executor (attention / weights / KV modes).
   *  Omit for the baseline. The decoding mode is a scheduler concern and
   *  does not affect the worker. */
  engineOptions?: EngineOptions;
  /** Factory for the worker-local PagedKVCache (scheduler's implementation).
   *  Required. It receives the manifest of the model the stage was just
   *  loaded from, so the cache is built from that model's GQA geometry —
   *  the geometry changes when the worker switches models. */
  createCache: (manifest: ModelManifest) => PagedKVCache;
  pollIntervalMs?: number;
  /**
   * Optional virtual-hardware node server factory. When provided it is
   * started before registration and its port is advertised as hwPort, so
   * the worker doubles as a virtual-hardware donor (vRAM/vGPU/vCPU/vSSD
   * routes). The factory is injected (rather than imported) so this module
   * stays ignorant of src/hw; the process entry point builds it.
   */
  hw?: { start: () => Promise<{ port: number; close(): Promise<void> }> };
}

export interface WorkerHandle {
  workerId: string;
  host: string;
  port: number;
  hwPort?: number;
  close(): Promise<void>;
}

export async function startWorker(opts: StartWorkerOptions): Promise<WorkerHandle> {
  if (typeof opts.createCache !== "function") {
    throw new Error("startWorker requires opts.createCache: (manifest: ModelManifest) => PagedKVCache");
  }
  const client = new MeshClient();
  const pollIntervalMs = opts.pollIntervalMs ?? HEARTBEAT_INTERVAL_MS;

  let workerId = "";
  let server: Server | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  let executor: StageExecutor | null = null;
  let cache: PagedKVCache | null = null;
  let loadedAssignment: ShardAssignment | null = null;
  let loadedVersion = 0;
  let knownVersion = 0;
  let loadPromise: Promise<boolean> | null = null;
  let activeExecutions = 0;
  let tokensSinceHeartbeat = 0;
  let closed = false;

  /**
   * Weight loaders by model key (modelId when the assignment carries one,
   * else the model dir). The spawn-time loader is seeded under its
   * manifest's modelId, so switching back to it is a cache hit. Loaders
   * are never evicted within a process lifetime; close() closes them all.
   */
  const loaders = new Map<string, WeightLoader>();
  loaders.set(opts.loader.manifest.modelId, opts.loader);
  /** Loader the current executor was built from (starts as spawn-time). */
  let currentLoader: WeightLoader = opts.loader;
  /** Cache key of `currentLoader`. */
  let currentModelKey: string = opts.loader.manifest.modelId;

  const log = (...args: unknown[]): void => {
    console.log(`[worker ${workerId || "starting"}]`, ...args);
  };

  /**
   * Resolve the loader for an assignment — vRAM first, disk only as a
   * fallback — and log which one was chosen, so a stage load's weight
   * source is always visible in the worker log.
   *
   *  1. No modelId and no modelDir: the v1 behaviour (spawn-time loader).
   *  2. Same model as the current stage, or a cached loader: reuse it
   *     (a cached VramWeightLoader is reused without another map fetch).
   *  3. modelId present: ask the orchestrator for the staged manifest +
   *     weight map (GET /mesh/weights/map). Staged -> build a
   *     VramWeightLoader whose tensor fetches hit the orchestrator's
   *     binary endpoint (GET /mesh/weights, octet-stream) and cache it.
   *     Not staged (404) -> fall through to the disk loader below.
   *     Any other fetch failure -> null: keep the old stage and retry
   *     on the next poll (do NOT silently serve a different model).
   *  4. Disk fallback, ONLY when the orchestrator has not staged the
   *     model (pre-activation / v1 path): openWeightLoader(modelDir).
   *     A model dir that cannot be opened -> null, keep the old stage.
   *     No modelDir at all -> the spawn-time loader (v1 behaviour).
   * Returns null when the new model cannot be loaded — the caller then
   * keeps serving the old stage and retries on the next poll.
   */
  async function loaderFor(assignment: ShardAssignment): Promise<WeightLoader | null> {
    if (assignment.modelDir === undefined && assignment.modelId === undefined) {
      return opts.loader;
    }
    // Step 1 above guarantees at least one of modelId/modelDir is set.
    const key = (assignment.modelId ?? assignment.modelDir) as string;
    if (key === currentModelKey) return currentLoader;
    const cached = loaders.get(key);
    if (cached !== undefined) return cached;
    if (assignment.modelId !== undefined) {
      let staged;
      try {
        staged = await client.getWeightMap(opts.orchestratorUrl, assignment.modelId);
      } catch (err) {
        log(
          `cannot fetch vRAM weight map for model ${assignment.modelId}: ${err instanceof Error ? err.message : String(err)}; keeping previous stage`,
        );
        return null;
      }
      if (staged !== null) {
        const modelId = assignment.modelId;
        const orchestratorUrl = opts.orchestratorUrl;
        const vramLoader = new VramWeightLoader(staged.manifest, staged.weightMap, (name) =>
          client.fetchWeightTensor(orchestratorUrl, modelId, name),
        );
        loaders.set(key, vramLoader);
        log(
          `using vRAM weight loader for model ${modelId} (${Object.keys(staged.weightMap).length} tensors staged on the orchestrator)`,
        );
        return vramLoader;
      }
      log(
        `model ${assignment.modelId} is not staged in vRAM on the orchestrator; falling back to the disk loader`,
      );
    }
    if (assignment.modelDir === undefined) return opts.loader;
    try {
      const opened = openWeightLoader(assignment.modelDir, assignment.modelId);
      loaders.set(key, opened);
      log(
        `using disk weight loader for model ${assignment.modelId ?? key} (orchestrator has not staged it)`,
      );
      return opened;
    } catch (err) {
      log(
        `cannot open model dir ${assignment.modelDir} for assignment (model ${assignment.modelId ?? key}): ${err instanceof Error ? err.message : String(err)}; keeping previous stage`,
      );
      return null;
    }
  }

  /** Fetch the current assignment and (re)load the stage if it changed. */
  async function loadAssignment(): Promise<boolean> {
    if (loadPromise !== null) return loadPromise;
    loadPromise = (async (): Promise<boolean> => {
      const resp = await client.getAssignment(opts.orchestratorUrl, workerId);
      if (resp.assignment === null) {
        knownVersion = resp.version;
        return false;
      }
      if (
        executor !== null &&
        loadedVersion === resp.version &&
        loadedAssignment !== null &&
        loadedAssignment.stage === resp.assignment.stage
      ) {
        knownVersion = resp.version;
        return true;
      }
      // Resolve the loader BEFORE building the stage. A model dir that
      // cannot be opened leaves the old executor, cache, assignment and
      // versions untouched — and knownVersion deliberately NOT advanced,
      // so the next heartbeat/assignment poll retries the switch.
      const loader = await loaderFor(resp.assignment);
      if (loader === null) return false;
      const next = opts.backend.createStageExecutor(
        loader.manifest,
        resp.assignment,
        opts.engineOptions,
      );
      await next.load(loader);
      executor = next;
      cache = opts.createCache(loader.manifest);
      currentLoader = loader;
      // The cache key of the loader actually used: the spawn loader's
      // manifest id when the assignment carries no modelDir (v1 behaviour),
      // else the assignment's modelId/modelDir key.
      currentModelKey =
        resp.assignment.modelDir === undefined
          ? loader.manifest.modelId
          : (resp.assignment.modelId ?? resp.assignment.modelDir);
      loadedAssignment = resp.assignment;
      loadedVersion = resp.version;
      knownVersion = resp.version;
      log(
        `loaded stage ${resp.assignment.stage} (layers ${resp.assignment.layerStart}-${resp.assignment.layerEnd}, model ${loader.manifest.modelId}, assignment v${resp.version}) via ${loader instanceof VramWeightLoader ? "vRAM" : "disk"} weights`,
      );
      return true;
    })();
    try {
      return await loadPromise;
    } finally {
      loadPromise = null;
    }
  }

  async function ensureExecutor(): Promise<StageExecutor> {
    if (executor !== null && cache !== null && loadedAssignment !== null) {
      // Fast path, unless the orchestrator has moved on since we last looked.
      if (knownVersion === loadedVersion) return executor;
    }
    const loaded = await loadAssignment();
    if (!loaded || executor === null || cache === null || loadedAssignment === null) {
      throw new Error(
        `no assignment for worker ${workerId} yet (assignment v${knownVersion})`,
      );
    }
    return executor;
  }

  async function executeHandler(req: ExecuteRequest): Promise<ExecuteResponse> {
    if (req.workerId !== workerId) {
      throw new Error(
        `execute routed to wrong worker: request is for ${req.workerId}, this worker is ${workerId}`,
      );
    }
    activeExecutions += 1;
    try {
      const stageExecutor = await ensureExecutor();
      const assignment = loadedAssignment as ShardAssignment;
      if (req.stage !== assignment.stage) {
        throw new Error(
          `stage mismatch: request is for stage ${req.stage}, this worker runs stage ${assignment.stage}`,
        );
      }
      const currentCache = cache as PagedKVCache;
      let input: Tensor;
      if (req.tokenIds !== undefined && assignment.hasEmbedding) {
        input = stageExecutor.embed(req.tokenIds);
      } else {
        input = {
          data: Float32Array.from(req.input.data),
          shape: [...req.input.shape],
        };
      }
      const output = stageExecutor.forward(input, {
        seqIds: [...req.seqIds],
        positions: [...req.positions],
        isPrefill: req.kind === "prefill",
        cache: currentCache,
      });
      const tokensProcessed = req.positions.length;
      tokensSinceHeartbeat += tokensProcessed;
      return {
        output: { data: Array.from(output.data), shape: [...output.shape] },
        isLogits: assignment.hasLmHead,
        tokensProcessed,
      };
    } finally {
      activeExecutions -= 1;
    }
  }

  // 1. Serve /mesh/execute on our own port first, so we know the real port
  //    (port 0 = ephemeral) before registering.
  server = createMeshServer({ execute: executeHandler });
  const actualPort: number = await new Promise<number>((resolve, reject) => {
    const onError = (err: Error): void => reject(err);
    server?.once("error", onError);
    server?.listen(opts.port, opts.host, () => {
      server?.removeListener("error", onError);
      const address = server?.address();
      resolve(
        typeof address === "object" && address !== null ? address.port : opts.port,
      );
    });
  });

  const capability: WorkerCapability = {
    ...opts.capability,
    host: opts.host,
    port: actualPort,
  };

  // 1b. Virtual-hardware donor server (if wired in by the caller): start
  //     before registering so hwPort can be advertised with the capability.
  let hwServer: { port: number; close(): Promise<void> } | undefined;
  if (opts.hw !== undefined) {
    hwServer = await opts.hw.start();
    capability.hwPort = hwServer.port;
  }

  // 2. Register with the orchestrator.
  let registered;
  try {
    registered = await client.register(opts.orchestratorUrl, capability);
  } catch (err) {
    await new Promise<void>((resolve) => server?.close(() => resolve()));
    if (hwServer !== undefined) await hwServer.close().catch(() => undefined);
    throw err;
  }
  workerId = registered.workerId;
  log(`registered with ${opts.orchestratorUrl} as ${workerId} on ${opts.host}:${actualPort}`);

  // 3. Heartbeat loop: report load, pick up newer assignments.
  async function doHeartbeat(): Promise<void> {
    if (closed) return;
    const tokens = tokensSinceHeartbeat;
    tokensSinceHeartbeat = 0;
    let resp;
    try {
      resp = await client.heartbeat(opts.orchestratorUrl, {
        workerId,
        usedMemoryBytes: executor !== null ? executor.memoryBytes() : 0,
        activeSequences: activeExecutions,
        tokensSinceHeartbeat: tokens,
      });
    } catch (err) {
      log(`heartbeat failed: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    if (!resp.ok) {
      log("heartbeat rejected by orchestrator (unknown worker id?)");
      return;
    }
    // loadAssignment() owns knownVersion: it advances it only once the
    // new assignment is actually loaded (or there is provably no
    // assignment). A failed load leaves knownVersion behind, so the next
    // heartbeat retries the switch instead of stranding the worker on a
    // version it never loaded.
    if (resp.assignmentVersion > knownVersion) {
      try {
        await loadAssignment();
      } catch (err) {
        log(
          `failed to load assignment v${resp.assignmentVersion}: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
  }

  // Initial heartbeat picks up any assignment already waiting for us.
  await doHeartbeat();
  timer = setInterval(() => {
    void doHeartbeat();
  }, pollIntervalMs);

  async function close(): Promise<void> {
    if (closed) return;
    closed = true;
    if (timer !== undefined) clearInterval(timer);
    timer = undefined;
    if (server !== undefined) {
      await new Promise<void>((resolve) => server?.close(() => resolve()));
      server = undefined;
    }
    if (hwServer !== undefined) {
      await hwServer.close().catch(() => undefined);
      hwServer = undefined;
    }
    // Loaders are never evicted while the worker runs; close them all now
    // (including the spawn-time loader seeded in the cache).
    for (const loader of loaders.values()) {
      try {
        loader.close();
      } catch {
        // Best-effort: a loader failing to close must not fail shutdown.
      }
    }
    loaders.clear();
    log("stopped");
  }

  return { workerId, host: opts.host, port: actualPort, hwPort: hwServer?.port, close };
}

// ---------------------------------------------------------------------------
// CLI entry point: `node src/mesh/worker.ts`
// ---------------------------------------------------------------------------

function argValue(argv: string[], name: string): string | undefined {
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i] as string;
    if (arg === `--${name}` && i + 1 < argv.length) return argv[i + 1];
    if (arg.startsWith(`--${name}=`)) return arg.slice(name.length + 3);
  }
  return undefined;
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const WORKER_DEVICES: WorkerDevice[] = ["cpu", "cuda", "metal", "rocm"];

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const orchestratorUrl =
    argValue(argv, "orchestrator") ??
    process.env.SOFTWELL_ORCHESTRATOR_URL ??
    `http://127.0.0.1:${DEFAULT_ORCHESTRATOR_PORT}`;
  const host =
    argValue(argv, "host") ?? process.env.SOFTWELL_WORKER_HOST ?? "127.0.0.1";
  const portRaw = argValue(argv, "port") ?? process.env.SOFTWELL_WORKER_PORT;
  const port = portRaw !== undefined ? Number(portRaw) : 0;
  const memoryBytes = envInt(
    "SOFTWELL_WORKER_MEMORY_BYTES",
    8 * 1024 * 1024 * 1024,
  );
  const modelDir =
    argValue(argv, "model-dir") ?? process.env.SOFTWELL_MODEL_DIR ?? "";
  const deviceRaw =
    argValue(argv, "device") ?? process.env.SOFTWELL_WORKER_DEVICE ?? "cpu";
  const device: WorkerDevice = (
    WORKER_DEVICES as string[]
  ).includes(deviceRaw)
    ? (deviceRaw as WorkerDevice)
    : "cpu";

  if (modelDir === "") {
    console.error(
      "usage: node src/mesh/worker.ts [--orchestrator <url>] [--host <host>] [--port <n>] [--model-dir <dir>]\n" +
        "  [--attention exact|landmark|sparse] [--weights fp32|sw2|sw3|sw4] [--kv fp32|sw2|sw3|sw4]\n" +
        "  env: SOFTWELL_ORCHESTRATOR_URL, SOFTWELL_WORKER_HOST, SOFTWELL_WORKER_PORT,\n" +
        "       SOFTWELL_WORKER_MEMORY_BYTES, SOFTWELL_WORKER_DEVICE, SOFTWELL_MODEL_DIR (required)",
    );
    process.exitCode = 2;
    return;
  }

  // Novel engine paths (attention / weight / KV modes); defaults are the
  // baseline. --decoding is a scheduler option and is not read here.
  const { options: engineOptions } = parseEngineFlags(argv);

  // The model loader and the PagedKVCache live with their owners
  // (src/model/loader.ts, src/scheduler/kvcache.ts). The cache factory
  // below builds one BlockPagedKVCache per stage load (block size 16,
  // 1024 blocks) from the loaded model's GQA geometry.
  let loader: WeightLoader;
  try {
    loader = openWeightLoader(modelDir);
  } catch (err) {
    console.error(
      `[worker] cannot open model dir ${modelDir}: ${err instanceof Error ? err.message : String(err)}`,
    );
    process.exitCode = 1;
    return;
  }
  const createCache = (manifest: ModelManifest): PagedKVCache =>
    new BlockPagedKVCache(
      16,
      1024,
      manifest.config.numKeyValueHeads,
      manifest.config.headDim,
    );

  const backend: EngineBackend = createReferenceBackend();

  // Virtual-hardware donor: this worker also serves /hw/* routes (vRAM,
  // vGPU, vCPU, vSSD node components) on an ephemeral port, advertised as
  // hwPort. Loaded via dynamic import so this module has no static
  // dependency on src/hw. Disable with SOFTWELL_HW=0.
  let hw: StartWorkerOptions["hw"];
  if (process.env.SOFTWELL_HW !== "0") {
    hw = {
      start: async () => {
        const { RamNode } = await import("../hw/vram.ts");
        const { GpuNode } = await import("../hw/vgpu.ts");
        const { CpuNode } = await import("../hw/vcpu.ts");
        const { SsdNode } = await import("../hw/vssd.ts");
        const { createHwNodeServer } = await import("../hw/node-server.ts");
        const ramNode = new RamNode({
          capacityBytes: memoryBytes,
          spillDir: `./data/ram-spill-${process.pid}`,
        });
        const gpuNode = new GpuNode({ device });
        const cpuNode = new CpuNode({});
        const ssdNode = new SsdNode({ dir: `./data/ssd-${process.pid}` });
        return createHwNodeServer({
          routeTables: [ramNode.routes(), gpuNode.routes(), cpuNode.routes(), ssdNode.routes()],
        });
      },
    };
  }

  // Calibrate once at startup: measure this donor's real sustained
  // matmul throughput instead of advertising an invented number.
  // Dynamic import so this module keeps no static dependency on src/hw.
  // On failure report 0 (unknown) and warn — never claim 1e12.
  let measuredFlopsPerSec = 0;
  try {
    const { measureMatmulTflops } = await import("../hw/calibrate.ts");
    measuredFlopsPerSec = await measureMatmulTflops();
    if (!(measuredFlopsPerSec > 0)) {
      console.warn("[worker] calibration returned no usable measurement; reporting flopsPerSec 0 (unknown)");
    } else {
      console.log(
        `[worker] calibrated flopsPerSec: ${(measuredFlopsPerSec / 1e9).toFixed(2)} GFLOP/s ` +
          `(${(measuredFlopsPerSec / 1e12).toFixed(4)} TFLOP/s, FP32 1024^3 matmul, median of 3)`,
      );
    }
  } catch (err) {
    console.warn(
      `[worker] calibration failed (${err instanceof Error ? err.message : String(err)}); reporting flopsPerSec 0 (unknown)`,
    );
    measuredFlopsPerSec = 0;
  }

  const handle = await startWorker({
    orchestratorUrl,
    host,
    port: Number.isFinite(port) ? port : 0,
    capability: {
      device,
      backend: backend.name,
      memoryBytes,
      freeMemoryBytes: memoryBytes,
      // MEASURED, not claimed: time a real 1024^3 matmul once at startup
      // (src/hw/calibrate.ts). If calibration fails we report 0
      // ("unknown") and warn — never the old invented 1e12.
      flopsPerSec: measuredFlopsPerSec,
      bandwidthBytesPerSec: 1e9,
      cores: os.cpus().length,
      roles: ["mixed"],
      maxBatchTokens: 1024,
    },
    backend,
    loader,
    createCache,
    engineOptions,
    hw,
  });

  const shutdown = (): void => {
    void handle.close().then(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

// Run main() only when executed directly, not when imported.
const isMain =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  main().catch((err: unknown) => {
    if (err instanceof MeshError) {
      console.error(`[worker] ${err.message}`);
    } else {
      console.error(`[worker] fatal: ${err instanceof Error ? err.message : String(err)}`);
    }
    process.exitCode = 1;
  });
}
