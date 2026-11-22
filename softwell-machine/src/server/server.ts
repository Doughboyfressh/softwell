/**
 * softwell-machine — the virtual server.
 *
 * The server is a workload of the virtual machine, not a host-side
 * sidecar: it boots with the VM, reserves its own slice of vRAM and a
 * vGPU stream from the hypervisor, registers its tokenize/detokenize
 * handlers on vCPU, and serves models using only virtual hardware.
 *
 * One active model at a time. The registry may stage several models,
 * but only one model has a serving context (scheduler + tokenizer)
 * driving requests; activating another model builds its context,
 * swaps it in, and disposes the previous one. In-flight requests on
 * the old model error out when their scheduler is disposed — the API
 * layer surfaces scheduler errors as 500s with the message.
 *
 * Throughput is measured, never invented: `tokensPerSec` is an
 * exponential moving average over the per-request decode rates of
 * completed requests only (see `observeCompletion`), and stays 0
 * until a request completes.
 *
 * Node built-ins only. Erasable TypeScript only.
 */

import type {
  AdmissionControl,
  ClusterSnapshot,
  CpuTask,
  HealthStatus,
  ModelEntry,
  ModelInfo,
  ServerRequestRecord,
  ServerStatus,
  ServingContext,
  ServingResolver,
  VirtualMachine,
  VirtualMachineInfo,
  VirtualServerOptions,
  VramAllocation,
} from "../contracts.ts";
import type { ModelRegistry } from "./registry.ts";
import { SoftwellServer } from "../api/server.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default vRAM reserved for the server's own buffers (256 MiB). */
export const DEFAULT_SERVER_MEMORY_BYTES = 256 * 1024 * 1024;

/** Default max waiting + running requests before new ones are rejected. */
export const DEFAULT_MAX_QUEUE = 32;

/** Request records kept in the ring buffer (newest first). */
export const RECENT_REQUESTS_CAP = 200;

/**
 * EMA weight for tokensPerSec: each completed request moves the average
 * 20% of the way toward that request's own measured decode rate.
 */
const TOKENS_PER_SEC_EMA_ALPHA = 0.2;

// ---------------------------------------------------------------------------
// Serving-context factory (implemented by the CLI wiring / another module)
// ---------------------------------------------------------------------------

export interface ServingContextFactory {
  /** Build a serving context for an already-loaded model entry: shard plan, scheduler, tokenizer. */
  build(entry: ModelEntry): Promise<ServingContext>;
  /** Tear down a serving context (stop driving its scheduler). */
  dispose(ctx: ServingContext): Promise<void>;
}

export interface VirtualServerDeps {
  registry: ModelRegistry;
  factory: ServingContextFactory;
  options?: VirtualServerOptions;
  getCluster?: () => ClusterSnapshot; // for health checks; absent => workersHealthy false
  onShutdown?: () => void | Promise<void>; // called by requestShutdown(); the CLI wires process exit
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Payload helpers for the vCPU tokenize/detokenize handlers. These follow
 * the payload convention of the vCPU "tokenize"/"detokenize" kinds in
 * src/hw/vcpu.ts: `{ text }` in, token ids out; `{ ids }` in, text out.
 * A bare string (tokenize) or bare number array (detokenize) is also
 * accepted, for callers that submit the operand directly.
 */
function tokenizeText(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (
    typeof payload === "object" &&
    payload !== null &&
    typeof (payload as Record<string, unknown>)["text"] === "string"
  ) {
    return (payload as Record<string, unknown>)["text"] as string;
  }
  throw new Error('tokenize: payload must be a string or { text: string }');
}

function detokenizeIds(payload: unknown): number[] {
  if (Array.isArray(payload)) {
    return assertTokenIds(payload);
  }
  if (
    typeof payload === "object" &&
    payload !== null &&
    Array.isArray((payload as Record<string, unknown>)["ids"])
  ) {
    return assertTokenIds(
      (payload as Record<string, unknown>)["ids"] as unknown[],
    );
  }
  throw new Error(
    "detokenize: payload must be a number[] or { ids: number[] }",
  );
}

function assertTokenIds(value: unknown[]): number[] {
  if (
    value.some((v) => typeof v !== "number" || !Number.isFinite(v))
  ) {
    throw new Error("detokenize: ids must be an array of finite numbers");
  }
  return value as number[];
}

export class VirtualServer implements ServingResolver, AdmissionControl {
  private deps: VirtualServerDeps;
  private options: VirtualServerOptions;

  private vm: VirtualMachine | null = null;
  private booted = false;
  private closed = false;
  private shutdownRequested = false;
  private state: ServerStatus["state"] = "starting";
  private bootedAt: number | null = null;

  private allocation: VramAllocation | null = null;
  private streamId: string | null = null;
  private apiServer: SoftwellServer | null = null;
  private apiPortValue: number | null = null;

  private active: ServingContext | null = null;
  private activeModelId: string | null = null;

  /** Ring buffer of request records, newest first (capped). */
  private recent: ServerRequestRecord[] = [];
  private counts = {
    active: 0,
    queued: 0,
    completed: 0,
    rejected: 0,
    cancelled: 0,
  };
  /** Measured decode throughput (tokens/sec); 0 until a request completes. */
  private tokensPerSecValue = 0;
  /** Completed requests that have contributed to the EMA (0 = unseeded). */
  private measuredCompletions = 0;

  constructor(deps: VirtualServerDeps) {
    this.deps = deps;
    this.options = deps.options ?? {};
  }

  // -- boot / shutdown ------------------------------------------------------

  /**
   * Boot the server on a virtual machine: reserve the server's own vRAM
   * slice, create a vGPU stream, register the tokenize/detokenize vCPU
   * handlers, and start the HTTP API server. State becomes "ready".
   *
   * Throws on a second boot. If any reservation fails, boot() throws
   * and frees whatever was already reserved, leaving the server
   * unbooted so boot() may be retried.
   */
  async boot(vm: VirtualMachine): Promise<void> {
    if (this.booted) {
      throw new Error("VirtualServer already booted");
    }
    // A previous close() is not terminal for boot: allow a fresh boot.
    this.closed = false;

    let allocation: VramAllocation | null = null;
    let apiServer: SoftwellServer | null = null;
    try {
      allocation = await vm.ram.alloc(
        this.options.serverMemoryBytes ?? DEFAULT_SERVER_MEMORY_BYTES,
      );
      const streamId = vm.gpu.createStream();
      vm.cpu.registerHandler("tokenize", (task) =>
        this.handleTokenize(task),
      );
      vm.cpu.registerHandler("detokenize", (task) =>
        this.handleDetokenize(task),
      );
      apiServer = new SoftwellServer({
        serving: this,
        admission: this,
        getCluster:
          this.deps.getCluster ?? (() => this.fallbackClusterSnapshot()),
        port: this.options.port,
        requestTimeoutMs: this.options.requestTimeoutMs,
        onShutdown: () => this.requestShutdown(),
      });
      const port = await apiServer.start();

      this.vm = vm;
      this.allocation = allocation;
      this.streamId = streamId;
      this.apiServer = apiServer;
      this.apiPortValue = port;
      this.booted = true;
      this.bootedAt = Date.now();
      this.state = "ready";
    } catch (err) {
      // Free whatever was already reserved; never mask the original error.
      if (apiServer !== null) {
        await apiServer.close().catch(() => undefined);
      }
      if (allocation !== null) {
        await vm.ram.free(allocation.id).catch(() => undefined);
      }
      throw err;
    }
  }

  /**
   * Request a graceful shutdown: state becomes "stopping" and the
   * wired onShutdown callback runs once (the CLI wires process exit
   * there). Resource cleanup itself happens in close().
   */
  requestShutdown(): void {
    this.state = "stopping";
    if (this.shutdownRequested) return;
    this.shutdownRequested = true;
    const cb = this.deps.onShutdown;
    if (cb === undefined) return;
    try {
      const result = cb();
      if (result instanceof Promise) {
        // Shutdown signalling must not surface an unhandled rejection.
        result.catch(() => undefined);
      }
    } catch {
      // Shutdown signalling must not fail the shutdown request itself.
    }
  }

  /**
   * Idempotent cleanup: stop the API server, dispose the active serving
   * context, and free the vRAM reservation. Safe to call before boot
   * (a no-op except that state becomes "stopping"). Cleanup is
   * best-effort: one failing step never prevents the others.
   */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.state = "stopping";
    const vm = this.vm;
    const allocation = this.allocation;
    const active = this.active;
    const apiServer = this.apiServer;
    this.vm = null;
    this.allocation = null;
    this.streamId = null;
    this.apiServer = null;
    this.apiPortValue = null;
    this.active = null;
    this.activeModelId = null;
    this.booted = false;
    if (apiServer !== null) {
      await apiServer.close().catch(() => undefined);
    }
    if (active !== null) {
      await this.deps.factory.dispose(active).catch(() => undefined);
    }
    if (vm !== null && allocation !== null) {
      await vm.ram.free(allocation.id).catch(() => undefined);
    }
  }

  // -- models ---------------------------------------------------------------

  /** Load (stage) a model through the registry. */
  loadModel(ref: string): Promise<ModelEntry> {
    return this.deps.registry.load(ref, {
      engineOptions: this.options.engineOptions,
    });
  }

  /**
   * Unload a model through the registry. Refuses the active model —
   * the registry enforces this too; the check here gives the clearer
   * error before delegating.
   */
  unloadModel(id: string): Promise<void> {
    if (id === this.activeModelId) {
      throw new Error(
        `cannot unload the active model: ${id} (activate another model first)`,
      );
    }
    return this.deps.registry.unload(id);
  }

  /**
   * Make a loaded model the active one: the entry must exist and have
   * status "ready". Builds the new context via the factory, swaps it
   * in, then disposes the previous context. If factory.build throws,
   * the previous active context (if any) stays active and the error
   * propagates. If disposing the previous context throws, the new
   * context is already active and the error propagates.
   */
  async activateModel(id: string): Promise<ServingContext> {
    const entry = this.deps.registry.get(id);
    if (entry === undefined) {
      throw new Error(`unknown model: ${id}`);
    }
    if (entry.status !== "ready") {
      throw new Error(
        `model ${id} is not ready (status: ${entry.status})`,
      );
    }
    const ctx = await this.deps.factory.build(entry);
    const previous = this.active;
    this.active = ctx;
    this.activeModelId = entry.id;
    if (previous !== null) {
      await this.deps.factory.dispose(previous);
    }
    return ctx;
  }

  /** All models the server knows about, as /v1/models entries. */
  listModels(): ModelInfo[] {
    return this.deps.registry.list().map((entry: ModelEntry) => ({
      id: entry.id,
      object: "model",
      created: Math.floor((entry.loadedAt ?? Date.now()) / 1000),
      owned_by: "softwell",
    }));
  }

  // -- ServingResolver ------------------------------------------------------

  getActive(): ServingContext | null {
    return this.active;
  }

  getServerStatus(): ServerStatus {
    return {
      state: this.state,
      uptimeSec:
        this.bootedAt === null ? 0 : (Date.now() - this.bootedAt) / 1000,
      activeModelId: this.activeModelId,
      models: this.deps.registry.list(),
      requests: this.counters(),
      tokensPerSec: this.tokensPerSecValue,
      queueDepth:
        this.active === null ? 0 : this.active.scheduler.pendingCount(),
      vm: this.vmInfoOrNull(),
      recentRequests: [...this.recent],
    };
  }

  getHealth(): HealthStatus {
    const vmBooted = this.vmBooted();
    const activeModel = this.active !== null;
    const workersHealthy = this.workersHealthy();
    const checks = { vmBooted, activeModel, workersHealthy };
    const status = !vmBooted
      ? "down"
      : activeModel && workersHealthy
        ? "ok"
        : "degraded";
    return { status, checks };
  }

  /** Readiness: true when a model is active and workers are healthy. */
  isReady(): boolean {
    return this.active !== null && this.workersHealthy();
  }

  /** Actual HTTP port the API server is listening on (null before boot). */
  getApiPort(): number | null {
    return this.apiPortValue;
  }

  // -- AdmissionControl -----------------------------------------------------

  /**
   * Admit a request for `modelId`, or reject it with 429: no active
   * model, the model is not the active one, or the active scheduler's
   * waiting + running count has reached maxQueue (default 32).
   * Every rejection is counted in counters().rejected.
   */
  admit(
    modelId: string,
  ): { ok: true } | { ok: false; status: number; message: string } {
    const active = this.active;
    if (active === null) {
      return this.reject(
        `no active model: cannot admit request for model ${modelId}`,
      );
    }
    if (modelId !== active.modelId) {
      return this.reject(
        `model ${modelId} is not active (active model is ${active.modelId})`,
      );
    }
    const maxQueue = this.options.maxQueue ?? DEFAULT_MAX_QUEUE;
    const depth =
      active.scheduler.pendingCount() + active.scheduler.runningCount();
    if (depth >= maxQueue) {
      return this.reject(
        `request queue is full (${depth} waiting + running, limit ${maxQueue})`,
      );
    }
    return { ok: true };
  }

  /**
   * Account for one request record. `active`/`queued` are live counts
   * driven by the queued/running/done/error/cancelled transitions;
   * `completed`/`rejected`/`cancelled` are cumulative. A terminal
   * record for a request that was never seen as queued/running (the
   * API layer records only the outcome) leaves the live counts at 0
   * rather than going negative. `rejected` records are counted here;
   * rejections decided by admit() are counted there instead — the API
   * layer must record a rejection through exactly one of the two
   * paths, not both.
   */
  record(record: ServerRequestRecord): void {
    this.recent.unshift(record);
    if (this.recent.length > RECENT_REQUESTS_CAP) {
      this.recent.length = RECENT_REQUESTS_CAP;
    }
    switch (record.status) {
      case "queued":
        this.counts.queued += 1;
        break;
      case "running":
        this.counts.active += 1;
        if (this.counts.queued > 0) this.counts.queued -= 1;
        break;
      case "done":
        this.counts.completed += 1;
        if (this.counts.active > 0) this.counts.active -= 1;
        this.observeCompletion(record);
        break;
      case "error":
        if (this.counts.active > 0) this.counts.active -= 1;
        break;
      case "cancelled":
        this.counts.cancelled += 1;
        if (this.counts.active > 0) this.counts.active -= 1;
        break;
      case "rejected":
        this.counts.rejected += 1;
        break;
    }
  }

  counters(): {
    active: number;
    queued: number;
    completed: number;
    rejected: number;
    cancelled: number;
  } {
    return { ...this.counts };
  }

  // -- internals ------------------------------------------------------------

  private reject(message: string): {
    ok: false;
    status: number;
    message: string;
  } {
    this.counts.rejected += 1;
    return { ok: false, status: 429, message };
  }

  /**
   * Fold one completed request into the measured tokens/sec.
   * Method: per-request decode rate = completionTokens / wall time
   * (endedAt - startedAt, seconds); the reported value is an
   * exponential moving average of those per-request rates (alpha 0.2),
   * seeded by the first completed request. Requests with no completion
   * tokens or no measurable wall time contribute nothing. No number is
   * ever invented: with no completed request the value stays 0.
   */
  private observeCompletion(record: ServerRequestRecord): void {
    if (record.completionTokens <= 0) return;
    const endedAt = record.endedAt ?? Date.now();
    const elapsedSec = (endedAt - record.startedAt) / 1000;
    if (!(elapsedSec > 0)) return;
    const instant = record.completionTokens / elapsedSec;
    if (this.measuredCompletions === 0) {
      this.tokensPerSecValue = instant;
    } else {
      this.tokensPerSecValue =
        (1 - TOKENS_PER_SEC_EMA_ALPHA) * this.tokensPerSecValue +
        TOKENS_PER_SEC_EMA_ALPHA * instant;
    }
    this.measuredCompletions += 1;
  }

  private vmBooted(): boolean {
    if (!this.booted || this.vm === null) return false;
    try {
      this.vm.info();
      return true;
    } catch {
      return false;
    }
  }

  private vmInfoOrNull(): VirtualMachineInfo | null {
    if (this.vm === null) return null;
    try {
      return this.vm.info();
    } catch {
      return null;
    }
  }

  private workersHealthy(): boolean {
    const getCluster = this.deps.getCluster;
    if (getCluster === undefined) return false;
    try {
      return getCluster().workers.some((w) => w.state === "healthy");
    } catch {
      return false;
    }
  }

  /**
   * Fallback cluster snapshot for the API server when no getCluster
   * was injected: no workers, no plan, and only measured values
   * (tokensPerSec from completed requests, 0 until one completes).
   */
  private fallbackClusterSnapshot(): ClusterSnapshot {
    return {
      modelId: this.activeModelId,
      plan: null,
      workers: [],
      sequences: { waiting: 0, running: 0, done: 0 },
      tokensPerSec: this.tokensPerSecValue,
      avgLatencyMs: 0,
      uptimeSec:
        this.bootedAt === null ? 0 : (Date.now() - this.bootedAt) / 1000,
    };
  }

  // -- vCPU handlers (registered on the VM at boot) --------------------------

  private handleTokenize(task: CpuTask): unknown {
    const active = this.active;
    if (active === null) {
      throw new Error(
        "no active model: cannot tokenize without an active model",
      );
    }
    return active.tokenizer.encode(tokenizeText(task.payload));
  }

  private handleDetokenize(task: CpuTask): unknown {
    const active = this.active;
    if (active === null) {
      throw new Error(
        "no active model: cannot detokenize without an active model",
      );
    }
    return active.tokenizer.decode(detokenizeIds(task.payload));
  }
}
