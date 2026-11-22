#!/usr/bin/env node
/**
 * src/cli.ts — `softwell` command line.
 *
 *   softwell up [--model-dir <dir>] [--workers N] [--port P] [--api-port A]
 *     Boot the orchestrator + N workers and wait. (v1: a model dir is
 *     required so workers have weights to load; defaults to the demo model
 *     when it exists.)
 *
 *   softwell run <model> [--workers N] [--port P] [--api-port A]
 *     Boot the cluster, load <model> (a local directory, or a Hugging Face
 *     repo id like "meta-llama/Llama-3.3-70B-Instruct" which is downloaded
 *     first), and serve the OpenAI-compatible API + dashboard.
 *
 *   softwell --help
 *
 * The CLI is the integration point: it owns the WorkerRegistry, the mesh
 * server, the shard plan, the scheduler (with a MeshClient transport), and
 * the API server. The virtual-hardware boot (hypervisor + pools) slots in
 * here once src/hw lands: pools are constructed over the same registry and
 * the model is staged vSSD -> vRAM before serving.
 */
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_ORCHESTRATOR_PORT,
  type ClusterSnapshot,
  type ExecuteResponse,
  type GeneratedToken,
  type Scheduler,
  type ShardPlan,
} from "./contracts.ts";
import { openWeightLoader, loadManifest } from "./model/loader.ts";
import { loadTokenizer } from "./model/tokenizer.ts";
import { downloadModel, resolveModelRef } from "./model/hub.ts";
import { plan, replan } from "./planner/planner.ts";
import { WorkerRegistry } from "./mesh/registry.ts";
import { createMeshServer, MeshClient } from "./mesh/protocol.ts";
import { ContinuousBatchingScheduler } from "./scheduler/batch.ts";
import { sampleLogits } from "./engine/sampling.ts";
import { SoftwellServer } from "./api/server.ts";
import { VirtualRam } from "./hw/vram.ts";
import { VirtualGpu } from "./hw/vgpu.ts";
import { VirtualCpu } from "./hw/vcpu.ts";
import { VirtualSsd } from "./hw/vssd.ts";
import { Hypervisor } from "./hw/hypervisor.ts";
import type { VirtualMachine } from "./contracts.ts";

const WORKER_ENTRY = fileURLToPath(new URL("./mesh/worker.ts", import.meta.url));
const DEMO_MODEL_DIR = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));
const DEFAULT_API_PORT = 8420;
const DEFAULT_WORKERS = 2;

interface CliOptions {
  command: "up" | "run" | "help";
  modelRef?: string;
  modelDir?: string;
  workers: number;
  port: number;
  apiPort: number;
  memoryBytes?: number;
}

function printHelp(): void {
  console.log(`softwell — a virtual machine for trillion-parameter open-weight LLMs

Usage:
  softwell up [--model-dir <dir>] [--workers N] [--port P] [--api-port A]
  softwell run <model> [--workers N] [--port P] [--api-port A]
  softwell --help

  <model> is a local directory or a Hugging Face repo id
  (e.g. meta-llama/Llama-3.3-70B-Instruct).

Defaults: --workers ${DEFAULT_WORKERS}, --port ${DEFAULT_ORCHESTRATOR_PORT}, --api-port ${DEFAULT_API_PORT}.
The demo model (demo/models/demo-10m) is used when no model is given and it exists.`);
}

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    command: "help",
    workers: DEFAULT_WORKERS,
    port: DEFAULT_ORCHESTRATOR_PORT,
    apiPort: DEFAULT_API_PORT,
  };
  const rest: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a === "up") opts.command = "up";
    else if (a === "run") opts.command = "run";
    else if (a === "--help" || a === "-h") opts.command = "help";
    else if (a === "--workers") opts.workers = Number(argv[++i]);
    else if (a === "--port") opts.port = Number(argv[++i]);
    else if (a === "--api-port") opts.apiPort = Number(argv[++i]);
    else if (a === "--model-dir") opts.modelDir = argv[++i];
    else if (a === "--memory") opts.memoryBytes = Number(argv[++i]);
    else if (a.startsWith("--")) throw new Error(`unknown flag: ${a}`);
    else rest.push(a);
  }
  if (opts.command === "run") {
    if (rest.length === 0) throw new Error("softwell run needs a <model> argument");
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
  pendingCount(): number {
    return this.inner.pendingCount();
  }
  runningCount(): number {
    return this.inner.runningCount();
  }
  avgLatencyMs(): number {
    if (this.latencies.length === 0) return 0;
    return this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length;
  }
  done(): number {
    return this.doneCount;
  }
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

  const modelDir = await resolveModelDir(opts);
  console.log(`softwell: loading model from ${modelDir}`);
  const manifest = loadManifest(modelDir, path.basename(modelDir));
  const loader = openWeightLoader(modelDir);
  void loader; // workers load their own shards from disk; the CLI only needs the manifest
  const tokenizer = loadTokenizer(modelDir);
  console.log(
    `softwell: ${manifest.modelId} — ${(manifest.totalParams / 1e6).toFixed(1)}M params, ` +
      `${(manifest.totalBytes / 1024 / 1024).toFixed(1)} MB, ${manifest.config.numHiddenLayers} layers`,
  );

  // --- Orchestrator: mesh registry + mesh server ---------------------------
  const registry = new WorkerRegistry();
  const meshServer = createMeshServer({
    register: (req) => registry.register(req.capability),
    heartbeat: (req) => registry.heartbeat(req),
    getAssignment: (workerId) => registry.getAssignment(workerId),
  });
  await new Promise<void>((resolve) => meshServer.listen(opts.port, "127.0.0.1", resolve));
  const orchestratorUrl = `http://127.0.0.1:${opts.port}`;
  console.log(`softwell: orchestrator mesh on ${orchestratorUrl}`);

  // --- Workers --------------------------------------------------------------
  const children: ChildProcess[] = [];
  const spawnWorker = (): void => {
    const env = { ...process.env };
    if (opts.memoryBytes) env.SOFTWELL_WORKER_MEMORY_BYTES = String(opts.memoryBytes);
    const child = spawn(
      process.execPath,
      [WORKER_ENTRY, "--orchestrator", orchestratorUrl, "--model-dir", modelDir, "--port", "0"],
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

  // --- Stage the model: local files -> vSSD -> vRAM -------------------------
  // Weights live on the virtual SSD and stream up into pooled vRAM; the
  // vGPU reads tensors from vRAM by virtual address.
  const modelFiles = fs
    .readdirSync(modelDir)
    .filter((f) => f.endsWith(".json") || f.endsWith(".safetensors"))
    .sort();
  for (const file of modelFiles) {
    const bytes = fs.readFileSync(path.join(modelDir, file));
    await vm.ssd.putObject(`models/${manifest.modelId}/${file}`, bytes);
  }
  console.log(`softwell: staged ${modelFiles.length} model file(s) on vSSD`);
  const weightKey = `models/${manifest.modelId}/model.safetensors`;
  const weightAlloc = await hypervisor.stageToVram(weightKey);
  gpu.registerAllocation(weightAlloc);
  console.log(
    `softwell: weights streamed vSSD -> vRAM (${(weightAlloc.sizeBytes / 1024 / 1024).toFixed(1)} MB ` +
      `in ${weightAlloc.pages.length} page(s) at vaddr 0x${weightAlloc.vaddr.toString(16)})`,
  );

  // --- Shard plan -----------------------------------------------------------
  const workers = registry.listWorkers();
  let shardPlan: ShardPlan = plan({ manifest, workers });
  if ("feasible" in shardPlan && shardPlan.feasible === false) {
    throw new Error(`no feasible shard plan: ${(shardPlan as { reason?: string }).reason ?? "unknown"}`);
  }
  registry.setAssignment(shardPlan);
  console.log(
    `softwell: plan — ${shardPlan.pipelineStages} pipeline stage(s), ` +
      `est. ${shardPlan.estimatedTokensPerSec.toFixed(1)} tok/s (${shardPlan.bottleneck})`,
  );

  // --- Scheduler ------------------------------------------------------------
  const client = new MeshClient({ timeoutMs: 30_000 });
  const schedulerInner = new ContinuousBatchingScheduler({
    plan: shardPlan,
    transport: async (workerId, req) => {
      const w = registry.getWorker(workerId);
      if (!w) throw new Error(`unknown worker ${workerId}`);
      return client.execute(`http://${w.host}:${w.port}`, { ...req, workerId });
    },
    sampler: sampleLogits,
    decode: (ids) => tokenizer.decode(ids),
    eosTokenId: manifest.config.eosTokenId,
    rolesByWorker: Object.fromEntries(workers.map((w) => [w.workerId, w.roles])),
  });
  const scheduler = new InstrumentedScheduler(schedulerInner);

  // --- API server -----------------------------------------------------------
  const startedAt = Date.now();
  const getCluster = (): ClusterSnapshot => ({
    modelId: manifest.modelId,
    plan: shardPlan,
    workers: registry.listWorkers(),
    sequences: {
      waiting: scheduler.pendingCount(),
      running: scheduler.runningCount(),
      done: scheduler.done(),
    },
    tokensPerSec: registry.listWorkers().reduce((n, w) => n + registry.workerThroughput(w.workerId), 0),
    avgLatencyMs: scheduler.avgLatencyMs(),
    uptimeSec: (Date.now() - startedAt) / 1000,
    vm: vm.info(),
  });
  const api = new SoftwellServer({
    scheduler,
    tokenizer,
    modelId: manifest.modelId,
    getCluster,
    port: opts.apiPort,
  });
  const apiPort = await api.start();
  console.log(`softwell: API on http://127.0.0.1:${apiPort}  (dashboard at /)`);
  console.log(`softwell: try: curl -s http://127.0.0.1:${apiPort}/v1/models`);
  console.log(`softwell: serving ${manifest.modelId} — press Ctrl-C to stop`);

  // --- Fault detection: sweep heartbeats, re-shard on worker loss -----------
  const sweepTimer = setInterval(() => {
    const changed = registry.sweep();
    for (const w of changed) {
      console.log(`softwell: worker ${w.workerId} -> ${w.state}`);
    }
    const newlyDead = changed.filter((w) => w.state === "dead");
    if (newlyDead.length === 0) return;
    const survivors = registry.listWorkers().filter((w) => w.state !== "dead");
    if (survivors.length === 0) {
      console.log("softwell: all workers lost — waiting for workers to rejoin");
      return;
    }
    try {
      const newPlan = replan(manifest, survivors, shardPlan);
      if (!newPlan.feasible) {
        console.log(`softwell: re-plan infeasible: ${newPlan.reason ?? "unknown"}`);
        return;
      }
      shardPlan = newPlan;
      registry.setAssignment(newPlan);
      schedulerInner.updatePlan(newPlan);
      console.log(
        `softwell: re-sharded over ${survivors.length} survivor(s), ${newPlan.pipelineStages} stage(s); ` +
          `sequences running on the lost worker will error and must be resubmitted (their KV cache is gone)`,
      );
    } catch (err) {
      console.log(`softwell: re-plan failed: ${(err as Error).message}`);
    }
  }, 2000);
  sweepTimer.unref();

  // --- Shutdown ---------------------------------------------------------------
  let shuttingDown = false;
  const shutdown = async (): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log("\nsoftwell: shutting down...");
    clearInterval(sweepTimer);
    await hypervisor.shutdown().catch(() => undefined);
    await api.close().catch(() => undefined);
    await new Promise<void>((resolve) => meshServer.close(() => resolve()));
    for (const child of children) child.kill("SIGTERM");
    process.exit(0);
  };
  process.on("SIGINT", () => void shutdown());
  process.on("SIGTERM", () => void shutdown());
}

main().catch((err: unknown) => {
  console.error(`softwell: fatal: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
