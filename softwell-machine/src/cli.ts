#!/usr/bin/env node
/**
 * src/cli.ts — `softwell` command line.
 *
 *   softwell up [--model-dir <dir>] [--workers N] [--port P] [--api-port A]
 *     Boot the cluster + virtual server and wait. --model-dir is
 *     optional: when given, that model is loaded and activated at boot;
 *     otherwise the server starts with no active model, and a model
 *     can be loaded later with POST /v1/models/load + /activate.
 *
 *   softwell run <model> [--workers N] [--port P] [--api-port A]
 *   softwell serve <model> [--workers N] [--port P] [--api-port A]
 *     Boot the cluster, load <model> (a local directory, or a Hugging Face
 *     repo id like "meta-llama/Llama-3.3-70B-Instruct" which is downloaded
 *     first), activate it and serve the OpenAI-compatible API + dashboard.
 *     `run` and `serve` are the same command under two names.
 *
 *   softwell models [--api-port A]
 *     List the models a running server knows about (HTTP client; no boot).
 *
 *   softwell stop [--api-port A]
 *     Stop a running server: SIGTERM the pid in data/softwell.pid when it
 *     is alive, otherwise POST /v1/server/shutdown on the API port.
 *
 *   softwell --help
 *
 * The CLI is the integration point: it owns the WorkerRegistry, the mesh
 * server, the workers, the virtual-hardware boot (hypervisor + pools) and
 * the ServingContextFactory that turns a ModelEntry into a running
 * scheduler. The serving plane itself (ModelRegistry, VirtualServer, the
 * HTTP API) lives in src/server/ and src/api/; this file only wires it.
 *
 * Reference-implementation note: workers open ONE weight loader from the
 * --model-dir they are spawned with, and open a new loader on demand
 * when an assignment names a different modelDir (see
 * src/mesh/worker.ts `loaderFor`), so activating a different model
 * re-plans and re-stages the workers without a restart. `softwell up`
 * without --model-dir starts workers on the demo model directory when
 * it exists. See docs/SERVER.md.
 */
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import type { Server } from "node:http";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_ORCHESTRATOR_PORT,
  type ClusterSnapshot,
  type EngineOptions,
  type GeneratedToken,
  type ModelEntry,
  type Scheduler,
  type ServingContext,
  type ShardPlan,
  type VirtualMachine,
} from "./contracts.ts";
import { describeEngineOptions, engineOptionsToFlags, parseEngineFlags } from "./engine/options.ts";
import { loadManifest } from "./model/loader.ts";
import { loadTokenizer } from "./model/tokenizer.ts";
import { downloadModel, resolveModelRef } from "./model/hub.ts";
import { plan, replan } from "./planner/planner.ts";
import { WorkerRegistry } from "./mesh/registry.ts";
import { createMeshServer, MeshClient } from "./mesh/protocol.ts";
import type { MeshServerHandlers } from "./mesh/protocol.ts";
import { ContinuousBatchingScheduler } from "./scheduler/batch.ts";
import { sampleLogits } from "./engine/sampling.ts";
import { ModelRegistry, createWeightHandlers } from "./server/registry.ts";
import { VirtualServer } from "./server/server.ts";
import type { ServingContextFactory } from "./server/server.ts";
import { VirtualRam } from "./hw/vram.ts";
import { VirtualGpu } from "./hw/vgpu.ts";
import { VirtualCpu } from "./hw/vcpu.ts";
import { VirtualSsd } from "./hw/vssd.ts";
import { Hypervisor } from "./hw/hypervisor.ts";

const WORKER_ENTRY = fileURLToPath(new URL("./mesh/worker.ts", import.meta.url));
const DEMO_MODEL_DIR = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));
const PIDFILE = fileURLToPath(new URL("../data/softwell.pid", import.meta.url));
const DEFAULT_API_PORT = 8420;
const DEFAULT_WORKERS = 2;

export interface CliOptions {
  command: "up" | "run" | "serve" | "models" | "stop" | "help";
  modelRef?: string;
  modelDir?: string;
  workers: number;
  port: number;
  apiPort: number;
  memoryBytes?: number;
  /** Novel engine paths (attention / weights / KV / decoding); all default
   *  to the baseline. Passed to workers (engine modes) and the scheduler
   *  (decoding mode). */
  engineOptions: EngineOptions;
}

function printHelp(): void {
  console.log(`softwell — a virtual machine for trillion-parameter open-weight LLMs

Usage:
  softwell up [--model-dir <dir>] [--workers N] [--port P] [--api-port A]
  softwell run <model> [--workers N] [--port P] [--api-port A]
  softwell serve <model> [--workers N] [--port P] [--api-port A]
  softwell models [--api-port A]
  softwell stop [--api-port A]
  softwell --help

  <model> is a local directory or a Hugging Face repo id
  (e.g. meta-llama/Llama-3.3-70B-Instruct).

  up starts the cluster and the virtual server. With --model-dir the
  model is loaded and activated at boot; without it the server starts
  with no model active, and a model can be loaded later with
  POST /v1/models/load followed by POST /v1/models/:id/activate
  (or start with --model-dir, or use run / serve.)
  run and serve are the same: boot, load <model>, activate it and serve
  until stopped. models lists what a running server knows about;
  stop shuts a running server down (pidfile first, API fallback).

  Novel engine paths (all default to the baseline behaviour):
    --attention exact|landmark|sparse   attention (Theorems A1/A2)
    --weights fp32|sw2|sw3|sw4          weight format (SW-code, Theorem N1)
    --kv fp32|sw2|sw3|sw4               KV-cache format (Theorem N2)
    --decoding autoregressive|speculative|jacobi
                                        token generation (Theorems S1/S2, greedy only)
    --draft-length K                    draft length / Jacobi block size (default 4)

Defaults: --workers ${DEFAULT_WORKERS}, --port ${DEFAULT_ORCHESTRATOR_PORT}, --api-port ${DEFAULT_API_PORT}.
The demo model (demo/models/demo-10m) is used for workers when no model is given and it exists.`);
}

export function parseArgs(argv: string[]): CliOptions {
  const { options: engineOptions, rest: argvRest } = parseEngineFlags(argv);
  const opts: CliOptions = {
    command: "help",
    workers: DEFAULT_WORKERS,
    port: DEFAULT_ORCHESTRATOR_PORT,
    apiPort: DEFAULT_API_PORT,
    engineOptions,
  };
  const rest: string[] = [];
  for (let i = 0; i < argvRest.length; i++) {
    const a = argvRest[i] as string;
    if (a === "up") opts.command = "up";
    else if (a === "run") opts.command = "run";
    else if (a === "serve") opts.command = "serve";
    else if (a === "models") opts.command = "models";
    else if (a === "stop") opts.command = "stop";
    else if (a === "--help" || a === "-h") opts.command = "help";
    else if (a === "--workers") opts.workers = Number(argvRest[++i]);
    else if (a === "--port") opts.port = Number(argvRest[++i]);
    else if (a === "--api-port") opts.apiPort = Number(argvRest[++i]);
    else if (a === "--model-dir") opts.modelDir = argvRest[++i];
    else if (a === "--memory") opts.memoryBytes = Number(argvRest[++i]);
    else if (a.startsWith("--")) throw new Error(`unknown flag: ${a}`);
    else rest.push(a);
  }
  if (opts.command === "run" || opts.command === "serve") {
    if (rest.length === 0) throw new Error(`softwell ${opts.command} needs a <model> argument`);
    opts.modelRef = rest[0];
  }
  return opts;
}

/** Scheduler wrapper that records step latency + completions for /v1/cluster. */
class InstrumentedScheduler implements Scheduler {
  private readonly inner: Scheduler;
  private latencies: number[] = [];
  private doneCount = 0;
  private submittedCount = 0;
  constructor(inner: Scheduler) {
    this.inner = inner;
  }
  submit(req: Parameters<Scheduler["submit"]>[0]): string {
    this.submittedCount++;
    return this.inner.submit(req);
  }
  async step(): Promise<GeneratedToken[]> {
    const t0 = performance.now();
    const tokens = await this.inner.step();
    this.latencies.push(performance.now() - t0);
    if (this.latencies.length > 100) this.latencies.shift();
    for (const t of tokens) if (t.finishReason !== null) this.doneCount++;
    return tokens;
  }
  getSequence(id: string) {
    return this.inner.getSequence(id);
  }
  cancel(id: string): boolean {
    return this.inner.cancel(id);
  }
  pendingCount(): number {
    return this.inner.pendingCount();
  }
  runningCount(): number {
    return this.inner.runningCount();
  }
  /** Swap in a new shard plan after a re-plan (delegates to the inner scheduler). */
  updatePlan(plan: ShardPlan): void {
    const inner = this.inner as Scheduler & { updatePlan?: (plan: ShardPlan) => void };
    inner.updatePlan?.(plan);
  }
  avgLatencyMs(): number {
    if (this.latencies.length === 0) return 0;
    return this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length;
  }
  done(): number {
    return this.doneCount;
  }
}

/** Mutable pointer to whatever is currently serving, shared by the
 *  factory (which sets it on activation), the cluster snapshot (which
 *  reads it) and the sweep/re-plan loop (which updates the plan). */
export interface ServingState {
  modelId: string | null;
  modelDir: string | null;
  plan: ShardPlan | null;
  scheduler: InstrumentedScheduler | null;
  inner: ContinuousBatchingScheduler | null;
}

function createServingState(): ServingState {
  return { modelId: null, modelDir: null, plan: null, scheduler: null, inner: null };
}

async function resolveModelDir(opts: CliOptions): Promise<string> {
  if (opts.modelDir) return path.resolve(opts.modelDir);
  if (opts.modelRef) {
    const ref = resolveModelRef(opts.modelRef);
    if (ref.kind === "local") return ref.dir;
    console.log(`downloading ${ref.repo} from the Hugging Face hub...`);
    const dest = path.resolve("models", ref.repo.replace("/", "__"));
    await downloadModel(opts.modelRef, dest);
    return dest;
  }
  if (fs.existsSync(DEMO_MODEL_DIR)) return DEMO_MODEL_DIR;
  throw new Error(
    "no model given and the demo model is not built yet — run `npm run make-demo-model` or pass a model directory / HF repo id",
  );
}

async function waitForWorkers(registry: WorkerRegistry, count: number, timeoutMs = 30_000): Promise<void> {
  const started = Date.now();
  for (;;) {
    const healthy = registry.listWorkers().filter((w) => w.state === "healthy").length;
    if (healthy >= count) return;
    if (Date.now() - started > timeoutMs) {
      throw new Error(`only ${healthy}/${count} workers became healthy within ${timeoutMs / 1000}s`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}

// ---------------------------------------------------------------------------
// Cluster boot
// ---------------------------------------------------------------------------

export interface BootClusterOptions {
  workers: number;
  port: number;
  memoryBytes?: number;
  engineOptions: EngineOptions;
  /** Local model directory workers load weights from at spawn. Workers
   *  open one weight loader at startup, so this pins what they can serve;
   *  it is NOT the active model — activation happens later, through the
   *  factory, and must name a model from this directory. */
  workerModelDir: string;
}

export interface BootedCluster {
  registry: WorkerRegistry;
  meshServer: Server;
  /**
   * The mesh handlers object the server was built from. createMeshServer
   * reads it at request time, so the weights routes (which need the
   * ModelRegistry, created only after boot) are attached to this same
   * object by serveCluster once the registry exists — no second server.
   */
  meshHandlers: MeshServerHandlers;
  children: ChildProcess[];
  hypervisor: Hypervisor;
  vm: VirtualMachine;
  servingState: ServingState;
  getCluster: () => ClusterSnapshot;
}

/**
 * Boot the cluster without any model: WorkerRegistry + mesh server +
 * N workers + waitForWorkers + Hypervisor boot (pools over the registry).
 * The virtual machine is up at the end; no model is loaded or active.
 */
export async function bootCluster(opts: BootClusterOptions): Promise<BootedCluster> {
  const servingState = createServingState();
  const startedAt = Date.now();

  // --- Orchestrator: mesh registry + mesh server ---------------------------
  // The handlers object is kept (and returned) so serveCluster can attach
  // the vRAM weights routes to it once the ModelRegistry exists; the
  // server reads handlers at request time, so late attachment works.
  const registry = new WorkerRegistry();
  const meshHandlers: MeshServerHandlers = {
    register: (req) => registry.register(req.capability),
    heartbeat: (req) => registry.heartbeat(req),
    getAssignment: (workerId) => registry.getAssignment(workerId),
  };
  const meshServer = createMeshServer(meshHandlers);
  await new Promise<void>((resolve) => meshServer.listen(opts.port, "127.0.0.1", resolve));
  const orchestratorUrl = `http://127.0.0.1:${opts.port}`;
  console.log(`softwell: orchestrator mesh on ${orchestratorUrl}`);

  // --- Workers --------------------------------------------------------------
  const children: ChildProcess[] = [];
  const engineFlags = engineOptionsToFlags(opts.engineOptions);
  const spawnWorker = (): void => {
    const env = { ...process.env };
    if (opts.memoryBytes) env.SOFTWELL_WORKER_MEMORY_BYTES = String(opts.memoryBytes);
    const child = spawn(
      process.execPath,
      [
        WORKER_ENTRY,
        "--orchestrator",
        orchestratorUrl,
        "--model-dir",
        opts.workerModelDir,
        "--port",
        "0",
        ...engineFlags,
      ],
      { env, stdio: "inherit" },
    );
    children.push(child);
  };
  for (let i = 0; i < opts.workers; i++) spawnWorker();
  console.log(`softwell: waiting for ${opts.workers} worker(s)...`);
  await waitForWorkers(registry, opts.workers);
  console.log(`softwell: ${opts.workers} worker(s) healthy`);

  // --- Virtual machine: boot the hypervisor over the donor workers --------
  // The VM *is* the hardware platform from here on: one global address
  // space (vRAM), one virtual GPU, pooled vCPUs, one object store (vSSD).
  // The LLM runtime below targets this VM; the mesh workers are its donors.
  const nodes = { listNodes: () => registry.listWorkers() };
  const ram = new VirtualRam({ nodes, defaultReplicas: 2 });
  const gpu = new VirtualGpu({ nodes, ram });
  const cpu = new VirtualCpu({ nodes });
  const ssd = new VirtualSsd({
    nodes,
    manifestPath: fileURLToPath(new URL("../data/vssd-manifest.json", import.meta.url)),
    ram,
  });
  const hypervisor = new Hypervisor({ nodes, ram, gpu, cpu, ssd });
  const vm: VirtualMachine = await hypervisor.boot();
  const vmInfo = vm.info();
  console.log(
    `softwell: virtual machine booted — vRAM ${(vmInfo.vram.totalBytes / 1024 ** 3).toFixed(1)} GiB pooled, ` +
      `vGPU ${(vmInfo.gpu.totalFlopsPerSec / 1e12).toFixed(1)} TFLOP/s across ${vmInfo.gpu.devices.length} virtual device(s), ` +
      `vCPU ${vmInfo.cpu.totalCores} cores, vSSD ${(vmInfo.ssd.totalBytes / 1024 ** 3).toFixed(1)} GiB`,
  );

  const getCluster = (): ClusterSnapshot => {
    const scheduler = servingState.scheduler;
    return {
      modelId: servingState.modelId,
      plan: servingState.plan,
      workers: registry.listWorkers(),
      sequences: {
        waiting: scheduler?.pendingCount() ?? 0,
        running: scheduler?.runningCount() ?? 0,
        done: scheduler?.done() ?? 0,
      },
      tokensPerSec: registry.listWorkers().reduce((n, w) => n + registry.workerThroughput(w.workerId), 0),
      avgLatencyMs: scheduler?.avgLatencyMs() ?? 0,
      uptimeSec: (Date.now() - startedAt) / 1000,
      vm: vm.info(),
    };
  };

  return { registry, meshServer, meshHandlers, children, hypervisor, vm, servingState, getCluster };
}

/**
 * Build the ServingContextFactory for a VirtualServer. Given a loaded
 * ModelEntry it reads the manifest, plans the model over the live workers
 * (throwing the planner's reason when infeasible), stamps modelId +
 * modelDir onto every assignment, installs the plan, and builds a
 * ContinuousBatchingScheduler (MeshClient transport, sampleLogits,
 * tokenizer decode, engine decoding options) wrapped in the
 * InstrumentedScheduler. Workers load their shards from pooled vRAM
 * over the mesh weight endpoint (falling back to the model directory
 * only before a model is staged); the CLI never opens a weight loader
 * itself.
 */
export function buildFactory(
  registry: WorkerRegistry,
  opts: CliOptions,
  servingState: ServingState,
): ServingContextFactory {
  const client = new MeshClient({ timeoutMs: 30_000 });
  return {
    async build(entry: ModelEntry): Promise<ServingContext> {
      const manifest = loadManifest(entry.dir, entry.id);
      const tokenizer = loadTokenizer(entry.dir);
      console.log(
        `softwell: ${manifest.modelId} — ${(manifest.totalParams / 1e6).toFixed(1)}M params, ` +
          `${(manifest.totalBytes / 1024 / 1024).toFixed(1)} MB, ${manifest.config.numHiddenLayers} layers`,
      );
      const workers = registry.listWorkers();
      const planned = plan({ manifest, workers });
      if (!planned.feasible) {
        throw new Error(`no feasible shard plan: ${planned.reason ?? "unknown"}`);
      }
      const stamped: ShardPlan = {
        ...planned,
        assignments: planned.assignments.map((a) => ({ ...a, modelId: entry.id, modelDir: entry.dir })),
      };
      registry.setAssignment(stamped);
      console.log(
        `softwell: plan — ${stamped.pipelineStages} pipeline stage(s), ` +
          `est. ${stamped.estimatedTokensPerSec.toFixed(1)} tok/s (${stamped.bottleneck})`,
      );
      // Novel decoding mode (speculative/Jacobi) is a scheduler option; the
      // engine modes (attention/weights/KV) already went to the workers.
      const inner = new ContinuousBatchingScheduler({
        plan: stamped,
        transport: async (workerId, req) => {
          const w = registry.getWorker(workerId);
          if (!w) throw new Error(`unknown worker ${workerId}`);
          return client.execute(`http://${w.host}:${w.port}`, { ...req, workerId });
        },
        sampler: sampleLogits,
        decode: (ids) => tokenizer.decode(ids),
        eosTokenId: manifest.config.eosTokenId,
        rolesByWorker: Object.fromEntries(workers.map((w) => [w.workerId, w.roles])),
        decoding: opts.engineOptions.decoding,
        draftLength: opts.engineOptions.draftLength,
      });
      const scheduler = new InstrumentedScheduler(inner);
      servingState.modelId = entry.id;
      servingState.modelDir = entry.dir;
      servingState.plan = stamped;
      servingState.scheduler = scheduler;
      servingState.inner = inner;
      return { modelId: entry.id, scheduler, tokenizer, manifest };
    },
    async dispose(ctx: ServingContext): Promise<void> {
      // Nothing to tear down for the scheduler itself: the next activation
      // re-plans and builds a fresh one. Only drop the CLI's pointer to
      // this context so the cluster snapshot and the re-plan loop stop
      // reporting it once it is gone.
      if (servingState.scheduler === ctx.scheduler) {
        servingState.modelId = null;
        servingState.modelDir = null;
        servingState.plan = null;
        servingState.scheduler = null;
        servingState.inner = null;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Pidfile (data/softwell.pid) — lets `softwell stop` find this process.
// ---------------------------------------------------------------------------

function writePidfile(): void {
  fs.mkdirSync(path.dirname(PIDFILE), { recursive: true });
  fs.writeFileSync(PIDFILE, `${process.pid}\n`, "utf-8");
}

function removePidfile(): void {
  try {
    fs.rmSync(PIDFILE, { force: true });
  } catch {
    // Best-effort: a missing pidfile is not an error.
  }
}

function readPidfile(): number | null {
  try {
    const raw = fs.readFileSync(PIDFILE, "utf-8").trim();
    const pid = Number(raw);
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    // ESRCH: no such process. EPERM: a process exists but we may not
    // signal it — treat as alive (SIGTERM below will report the error).
    return code !== "ESRCH";
  }
}

async function waitForPidExit(pid: number, timeoutMs: number): Promise<boolean> {
  const started = Date.now();
  for (;;) {
    if (!isPidAlive(pid)) return true;
    if (Date.now() - started > timeoutMs) return false;
    await new Promise((r) => setTimeout(r, 200));
  }
}

// ---------------------------------------------------------------------------
// Serve: bootCluster + ModelRegistry + VirtualServer, then serve forever.
// ---------------------------------------------------------------------------

/**
 * Boot the cluster and the virtual server, optionally load + activate one
 * model at boot, then serve until shutdown (SIGINT/SIGTERM or the API
 * shutdown endpoint). Returns only on fatal setup errors — once serving,
 * the process stays alive on its listening servers.
 */
async function serveCluster(opts: CliOptions, bootModelDir: string | undefined): Promise<void> {
  // Workers open one weight loader at spawn, so they need a model
  // directory even when no model is active yet. The demo model is the
  // fallback; without it the caller must name a directory.
  let workerModelDir: string;
  if (bootModelDir !== undefined) {
    workerModelDir = bootModelDir;
  } else if (fs.existsSync(DEMO_MODEL_DIR)) {
    workerModelDir = DEMO_MODEL_DIR;
    console.log(
      `softwell: no --model-dir given; workers start with the demo model directory (${DEMO_MODEL_DIR})`,
    );
  } else {
    throw new Error(
      "no model directory for workers: pass --model-dir <dir> or run `npm run make-demo-model` so the demo model exists",
    );
  }
  if (bootModelDir !== undefined) {
    console.log(`softwell: loading model from ${bootModelDir}`);
  }

  const cluster = await bootCluster({
    workers: opts.workers,
    port: opts.port,
    memoryBytes: opts.memoryBytes,
    engineOptions: opts.engineOptions,
    workerModelDir,
  });
  const { registry, meshServer, children, hypervisor, vm, servingState, getCluster } = cluster;
  const factory = buildFactory(registry, opts, servingState);

  let server: VirtualServer | null = null;
  let shuttingDown = false;
  let sweepTimer: ReturnType<typeof setInterval> | undefined;
  const shutdown = async (): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log("\nsoftwell: shutting down...");
    if (sweepTimer !== undefined) clearInterval(sweepTimer);
    await hypervisor.shutdown().catch(() => undefined);
    await server?.close().catch(() => undefined);
    await new Promise<void>((resolve) => meshServer.close(() => resolve()));
    for (const child of children) child.kill("SIGTERM");
    removePidfile();
    process.exit(0);
  };

  const modelRegistry = new ModelRegistry({
    vm: () => vm,
    stageToVram: (key: string) => hypervisor.stageToVram(key),
    readVram: (vaddr: number, length: number) => vm.ram.read(vaddr, length),
    isActive: (id: string) => server?.getActive()?.modelId === id,
    modelsDir: path.resolve("models"),
  });
  // Attach the binary weight endpoint to the orchestrator's mesh server:
  // workers fetch staged manifests + weight maps (JSON) and tensor bytes
  // (octet-stream) from pooled vRAM through these handlers. The registry
  // serves the bytes; no disk read happens on this path.
  const weightHandlers = createWeightHandlers(modelRegistry);
  cluster.meshHandlers.getWeightMap = weightHandlers.getWeightMap;
  cluster.meshHandlers.getWeightTensor = weightHandlers.getWeightTensor;
  server = new VirtualServer({
    registry: modelRegistry,
    factory,
    options: { port: opts.apiPort, engineOptions: opts.engineOptions },
    getCluster,
    onShutdown: () => void shutdown(),
  });
  // Load + activate the boot model BEFORE starting the API server, so
  // the API only opens once the model is actually serving (the demo and
  // clients treat "the API answers" as "ready to generate"). The
  // registry only needs the booted VM, not a booted server. `up` without
  // a model skips this and boots with nothing active.
  let bootEntry: ModelEntry | null = null;
  if (bootModelDir !== undefined) {
    bootEntry = await server.loadModel(bootModelDir);
    await server.activateModel(bootEntry.id);
  }
  await server.boot(vm);
  writePidfile();

  const actualApiPort = server.getApiPort() ?? opts.apiPort;
  console.log(`softwell: API on http://127.0.0.1:${actualApiPort}  (dashboard at /)`);
  console.log(`softwell: try: curl -s http://127.0.0.1:${actualApiPort}/v1/models`);

  if (bootEntry !== null) {
    console.log(`softwell: serving ${bootEntry.id} — press Ctrl-C to stop`);
  } else {
    console.log(
      "softwell: no model active — load one with POST /v1/models/load then " +
        "POST /v1/models/:id/activate, or restart with --model-dir <dir> " +
        "(or `softwell serve <model>`)",
    );
  }

  // --- Fault detection: sweep heartbeats, re-shard on worker loss -----------
  // Only the ACTIVE model is re-planned; with no model active there is
  // nothing to re-shard (a later activation plans over the survivors).
  sweepTimer = setInterval(() => {
    const changed = registry.sweep();
    for (const w of changed) {
      console.log(`softwell: worker ${w.workerId} -> ${w.state}`);
    }
    const newlyDead = changed.filter((w) => w.state === "dead");
    if (newlyDead.length === 0) return;
    const active = server?.getActive();
    if (active === null || active === undefined) {
      console.log("softwell: worker lost but no model is active — nothing to re-plan");
      return;
    }
    const modelDir = servingState.modelDir;
    const oldPlan = servingState.plan;
    if (modelDir === null || oldPlan === null) {
      console.log("softwell: worker lost but no shard plan is installed — nothing to re-plan");
      return;
    }
    const survivors = registry.listWorkers().filter((w) => w.state !== "dead");
    if (survivors.length === 0) {
      console.log("softwell: all workers lost — waiting for workers to rejoin");
      return;
    }
    try {
      const newPlan = replan(active.manifest, survivors, oldPlan);
      if (!newPlan.feasible) {
        console.log(`softwell: re-plan infeasible: ${newPlan.reason ?? "unknown"}`);
        return;
      }
      const stamped: ShardPlan = {
        ...newPlan,
        assignments: newPlan.assignments.map((a) => ({ ...a, modelId: active.modelId, modelDir })),
      };
      servingState.plan = stamped;
      registry.setAssignment(stamped);
      const scheduler = active.scheduler;
      if (scheduler instanceof InstrumentedScheduler) scheduler.updatePlan(stamped);
      else servingState.inner?.updatePlan(stamped);
      console.log(
        `softwell: re-sharded over ${survivors.length} survivor(s), ${stamped.pipelineStages} stage(s); ` +
          `sequences running on the lost worker will error and must be resubmitted (their KV cache is gone)`,
      );
    } catch (err) {
      console.log(`softwell: re-plan failed: ${(err as Error).message}`);
    }
  }, 2000);
  sweepTimer.unref();

  process.on("SIGINT", () => void shutdown());
  process.on("SIGTERM", () => void shutdown());
}

// ---------------------------------------------------------------------------
// `softwell models` / `softwell stop` — HTTP clients, no boot.
// ---------------------------------------------------------------------------

interface ModelsListResponse {
  data?: Array<{ id?: string }>;
}

interface ServerStatusResponse {
  state?: string;
  uptimeSec?: number;
  activeModelId?: string | null;
  models?: Array<{
    id: string;
    status: string;
    params?: number;
    weightBytes?: number;
    vramBytes: number;
    requestsServed?: number;
  }>;
}

async function runModelsCommand(apiPort: number): Promise<void> {
  const base = `http://127.0.0.1:${apiPort}`;
  let models: ModelsListResponse;
  try {
    const res = await fetch(`${base}/v1/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    models = (await res.json()) as ModelsListResponse;
  } catch {
    console.error(`no softwell server reachable on port ${apiPort} — start one with \`softwell up\``);
    process.exitCode = 1;
    return;
  }
  // The rich per-model fields (status, params, weight/vRAM bytes) live on
  // /v1/server/status; /v1/models only carries the ids. Fall back to the
  // bare list when the status route is unavailable.
  let status: ServerStatusResponse | null = null;
  try {
    const res = await fetch(`${base}/v1/server/status`);
    if (res.ok) status = (await res.json()) as ServerStatusResponse;
  } catch {
    status = null;
  }
  const rows = (status?.models ?? []).map((m) => ({
    id: m.id,
    status: m.status,
    paramsM: m.params !== undefined ? (m.params / 1e6).toFixed(1) : "—",
    weightMB: m.weightBytes !== undefined ? (m.weightBytes / 1024 / 1024).toFixed(1) : "—",
    vramMB: (m.vramBytes / 1024 / 1024).toFixed(1),
    active: status?.activeModelId === m.id ? "*" : "",
  }));
  if (rows.length === 0) {
    for (const m of models.data ?? []) {
      if (typeof m.id === "string") {
        rows.push({ id: m.id, status: "unknown", paramsM: "—", weightMB: "—", vramMB: "—", active: "" });
      }
    }
  }
  if (rows.length === 0) {
    console.log("no models known to the server");
  } else {
    const pad = (s: string, n: number): string => s.padEnd(n);
    console.log(
      `${pad("id", 24)}${pad("status", 12)}${pad("params (M)", 12)}${pad("weight (MB)", 13)}${pad("vRAM (MB)", 11)}active`,
    );
    for (const r of rows) {
      console.log(
        `${pad(r.id, 24)}${pad(r.status, 12)}${pad(r.paramsM, 12)}${pad(r.weightMB, 13)}${pad(r.vramMB, 11)}${r.active}`,
      );
    }
  }
  if (status !== null) {
    console.log(
      status.activeModelId !== null && status.activeModelId !== undefined
        ? `active model: ${status.activeModelId} (state: ${status.state ?? "?"}, uptime: ${Math.round(status.uptimeSec ?? 0)}s)`
        : `no model active (state: ${status.state ?? "?"}, uptime: ${Math.round(status.uptimeSec ?? 0)}s)`,
    );
  }
}

async function runStopCommand(apiPort: number): Promise<void> {
  const pid = readPidfile();
  if (pid !== null) {
    if (isPidAlive(pid)) {
      console.log(`softwell: stopping pid ${pid}...`);
      try {
        process.kill(pid, "SIGTERM");
      } catch (err) {
        console.error(`softwell: could not signal pid ${pid}: ${(err as Error).message}`);
        process.exitCode = 1;
        return;
      }
      const exited = await waitForPidExit(pid, 10_000);
      if (!exited) {
        console.log(`softwell: pid ${pid} did not exit in 10s — SIGKILL`);
        try {
          process.kill(pid, "SIGKILL");
        } catch {
          // Already gone.
        }
        await waitForPidExit(pid, 2000);
      }
      removePidfile();
      console.log(`softwell: stopped (pid ${pid})`);
      process.exitCode = 0;
      return;
    }
    // Stale pidfile: the pid is dead. Drop it and try the API instead.
    console.log(`softwell: pidfile pointed at dead pid ${pid} — trying the API instead`);
    removePidfile();
  }
  try {
    const res = await fetch(`http://127.0.0.1:${apiPort}/v1/server/shutdown`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    if (res.ok) {
      console.log(`softwell: shutdown requested via the API on port ${apiPort}`);
      process.exitCode = 0;
      return;
    }
    console.error(`softwell: API shutdown failed: HTTP ${res.status}`);
  } catch {
    console.error(`no softwell server reachable on port ${apiPort} — nothing to stop (start one with \`softwell up\`)`);
  }
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let opts: CliOptions;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`error: ${(err as Error).message}\n`);
    printHelp();
    process.exitCode = 2;
    return;
  }
  if (opts.command === "help") {
    printHelp();
    return;
  }
  if (opts.command === "models") {
    await runModelsCommand(opts.apiPort);
    return;
  }
  if (opts.command === "stop") {
    await runStopCommand(opts.apiPort);
    return;
  }
  try {
    if (opts.command === "up") {
      const bootModelDir = opts.modelDir ? path.resolve(opts.modelDir) : undefined;
      console.log(`softwell: engine: ${describeEngineOptions(opts.engineOptions)}`);
      await serveCluster(opts, bootModelDir);
    } else {
      // run | serve: same path — boot, load the model, activate, serve.
      const modelDir = await resolveModelDir(opts);
      console.log(`softwell: engine: ${describeEngineOptions(opts.engineOptions)}`);
      await serveCluster(opts, modelDir);
    }
  } catch (err) {
    removePidfile();
    console.error(`softwell: fatal: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 1;
  }
}

// Run main() only when executed directly, not when imported (tests import
// parseArgs from this module).
const isMain =
  process.argv[1] !== undefined &&
  fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  void main();
}
