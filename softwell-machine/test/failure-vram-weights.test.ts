/**
 * Failure semantics for vRAM-sourced weights at the WORKER level
 * (test/failure-vram-weights.test.ts).
 *
 * Workstream 4 of the "weights from vRAM" phase. Companion to
 * test/healing-vram-weights.test.ts, which covers the vRAM pool itself
 * (handleNodeLoss re-replication, sweep healing, mid-stage death,
 * permanent page loss). THIS file covers the worker/generation level:
 * a donor that is also a compute worker dying mid-inference, and a
 * weight-endpoint fetch failing mid-load.
 *
 *  1. Mid-generation donor+worker death (real ReferenceBackend,
 *     demo-100m, real Hypervisor + ModelRegistry + donor node servers):
 *     two workers serve a 2-stage pipeline from vRAM; a generation is
 *     in flight (prefill + decode tokens flowing) when worker-2 — which
 *     is also the vRAM donor holding half the weight replicas — is
 *     killed. The in-flight decode routed to the dead worker fails
 *     (it must be resubmitted; its KV cache is gone — the CLI logs the
 *     same contract). The survivor is re-planned to a single full
 *     stage; its re-load fetches the newly-owned layers from the
 *     weight endpoint WHILE the donor is still dead and unhealed (so
 *     the reads must fail over to the surviving replica), and only
 *     then is the pool healed. A fresh generation on the survivor
 *     produces byte-identical tokens to the pre-kill reference, the
 *     worker log shows the re-load "via vRAM weights" (never "disk
 *     weight loader"), the spawn disk loaders read 0 bytes, and the
 *     orchestrator's weightBytesServed grows across the re-load.
 *     The model directory is a temp COPY that is moved out of reach
 *     after staging, so any disk fallback would fail loudly instead
 *     of silently succeeding.
 *  2. Weight-endpoint fetch failure mid-load (fake backend, fake VM):
 *     the orchestrator's binary weight endpoint fails transiently
 *     while a worker loads a new stage. The worker keeps serving its
 *     old stage, retries on the next heartbeat poll, and eventually
 *     loads the new stage from vRAM — it never falls back to the disk
 *     loader. Counters (VramWeightLoader.tensorsFetched/bytesFetched,
 *     SafetensorsWeightLoader.bytesReadFromDisk) are asserted
 *     directly here because the worker is in-process.
 *
 * Covered elsewhere (not duplicated here):
 *  - test/healing-vram-weights.test.ts: targeted primary-replica kill
 *    + re-replication, sweep healing, donor death DURING staging,
 *    failed-staging allocation leak, permanent page loss (registry
 *    leaves the model "ready" — known gap), single-survivor
 *    degradation, endpoint replica failover before/after healing.
 *  - test/vram-weights-proof.test.ts: real spawned cluster, model dir
 *    moved out of reach, worker restart re-loads from vRAM.
 *  - demo/run-demo.ts: spawned 2-worker cluster, kill + re-shard +
 *    generate again.
 *
 * Deferred (with reasons):
 *  - A spawned-CLI variant of test 1 (kill a worker process
 *    mid-stream through the HTTP API): the kill timing and the
 *    vRAM re-load are covered here at the mesh level with stronger
 *    assertions (log lines + moved dir + token equality), and the
 *    spawned kill + regenerate path is already covered by the proof
 *    test and the demo. Spawning another 417 MB cluster concurrently
 *    on this shared host is also what OOM-killed the baseline runs
 *    (see the report).
 *  - Donor death DURING a single in-flight tensor fetch (a fetch
 *    issued to a donor that dies before answering): VirtualRam's
 *    fetchPageData already tries every copy in order (see
 *    src/hw/vram.ts fetchPageData), so the read fails over inside
 *    the same call; test 1 exercises the equivalent window by
 *    re-loading while the donor is dead and unhealed, and the
 *    healing file's test 7 proves the endpoint fails over
 *    pre-healing. A finer-grained in-flight kill is timing-only
 *    and adds no new semantic coverage.
 *
 * Notes on the environment: /tmp on this host is a 512 MiB tmpfs
 * that is usually nearly full, so ALL spill/tmp dirs here live
 * under <repo>/data/ (real disk) and are removed in finally blocks.
 * The shared demo/models/demo-100m directory is never renamed: a
 * sibling workstream renames it concurrently, so test 1 serves a
 * private temp copy instead.
 *
 * Run: node --test test/failure-vram-weights.test.ts
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { VirtualGpu, GpuNode } from "../src/hw/vgpu.ts";
import { VirtualCpu, CpuNode } from "../src/hw/vcpu.ts";
import { VirtualSsd, SsdNode } from "../src/hw/vssd.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import { loadManifest, openWeightLoader, SafetensorsWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { VramWeightLoader } from "../src/model/vram-loader.ts";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import { ModelRegistry, createWeightHandlers } from "../src/server/registry.ts";
import type {
  ModelManifest,
  NodeProvider,
  PagedKVCache,
  ShardAssignment,
  StageExecutor,
  Tensor,
  VirtualMachine,
  VramAllocation,
  WeightLoader,
  WorkerInfo,
} from "../src/contracts.ts";
import type { EngineBackend, ForwardContext } from "../src/contracts.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DEMO_10M_DIR = path.join(ROOT, "demo", "models", "demo-10m");
const DEMO_100M_DIR = path.join(ROOT, "demo", "models", "demo-100m");
const MIB = 1024 * 1024;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(
  cond: () => boolean,
  what: string,
  timeoutMs = 30_000,
  intervalMs = 25,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cond()) return;
    if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
    await sleep(intervalMs);
  }
}

/** Capture console.log lines (workers log their loader choice there)
 *  while still printing them, so failures stay debuggable. */
function captureLogs(): { lines: string[]; restore(): void } {
  const lines: string[] = [];
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    lines.push(
      args
        .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
        .join(" "),
    );
    origLog(...args);
  };
  return {
    lines,
    restore: () => {
      console.log = origLog;
    },
  };
}

function argmaxLastRow(output: { data: number[]; shape: number[] }): number {
  const rows = output.shape[0] as number;
  const cols = output.shape[1] as number;
  const start = (rows - 1) * cols;
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < cols; i++) {
    const v = output.data[start + i] as number;
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

interface PipelineStage {
  workerId: string;
  url: string;
  stage: number;
}

/**
 * Drive one greedy generation through a pipeline of stages, mirroring
 * the scheduler's execute conventions (one row per token; stage 0
 * embeds tokenIds, later stages take the previous stage's output;
 * prefill covers the whole prompt, decode feeds one token at a time).
 * Returns the sampled token ids (excluding the prompt).
 */
async function runPipeline(
  client: MeshClient,
  stages: PipelineStage[],
  promptIds: number[],
  maxNewTokens: number,
  seqId: string,
  eosTokenId: number,
): Promise<number[]> {
  const rows = promptIds.length;
  const seqIds = promptIds.map(() => seqId);
  const positions = promptIds.map((_, i) => i);
  let out: { data: number[]; shape: number[] } | null = null;
  for (let s = 0; s < stages.length; s++) {
    const st = stages[s] as PipelineStage;
    const resp = await client.execute(st.url, {
      workerId: st.workerId,
      seqIds: [...seqIds],
      kind: "prefill",
      stage: st.stage,
      input: out ?? { data: [], shape: [0] },
      positions: [...positions],
      ...(s === 0 ? { tokenIds: [...promptIds] } : {}),
    });
    if (s === stages.length - 1) {
      assert.equal(resp.isLogits, true, "last stage must return logits");
    }
    out = resp.output;
  }
  if (out === null) throw new Error("pipeline produced no output");
  const tokens: number[] = [];
  let next = argmaxLastRow(out);
  tokens.push(next);
  let pos = rows;
  while (tokens.length < maxNewTokens && next !== eosTokenId) {
    let stepOut: { data: number[]; shape: number[] } | null = null;
    for (let s = 0; s < stages.length; s++) {
      const st = stages[s] as PipelineStage;
      const resp = await client.execute(st.url, {
        workerId: st.workerId,
        seqIds: [seqId],
        kind: "decode",
        stage: st.stage,
        input: stepOut ?? { data: [], shape: [0] },
        positions: [pos],
        ...(s === 0 ? { tokenIds: [next] } : {}),
      });
      stepOut = resp.output;
    }
    if (stepOut === null) throw new Error("pipeline decode produced no output");
    next = argmaxLastRow(stepOut);
    tokens.push(next);
    pos += 1;
  }
  return tokens;
}

/* ------------------------------------------------------------------ */
/* Test 1 harness: real donors + hypervisor + registry + fake        */
/* orchestrator (register/heartbeat/getAssignment + the real binary  */
/* weight endpoint), two real in-process workers on the real         */
/* ReferenceBackend. The workers are ALSO the vRAM donors: donor    */
/* "worker-N" is the hw node server for the worker with that id, so  */
/* killing a worker kills its donor too — the production shape,     */
/* where one process serves both /mesh/execute and /hw/*.           */
/* ------------------------------------------------------------------ */

function makeWorkerInfo(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "test",
    memoryBytes: 8 * 1024 * 1024 * 1024,
    freeMemoryBytes: 8 * 1024 * 1024 * 1024,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    cores: 4,
    roles: ["mixed"],
    maxBatchTokens: 256,
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeNodes implements NodeProvider {
  list: WorkerInfo[];
  constructor(list: WorkerInfo[]) {
    this.list = list;
  }
  listNodes(): WorkerInfo[] {
    return [...this.list];
  }
  markDead(workerId: string): void {
    const node = this.list.find((n) => n.workerId === workerId);
    if (node !== undefined) node.state = "dead";
  }
  add(node: WorkerInfo): void {
    this.list.push(node);
  }
}

type DonorServer = Awaited<ReturnType<typeof createHwNodeServer>>;
type MeshServer = Awaited<ReturnType<typeof createMeshServer>>;
type WorkerHandle = Awaited<ReturnType<typeof startWorker>>;

interface Harness1 {
  tmp: string;
  provider: FakeNodes;
  vm: VirtualMachine;
  ram: VirtualRam;
  hypervisor: Hypervisor;
  registry: ModelRegistry;
  orchServer: MeshServer;
  orchUrl: string;
  orchState: {
    version: number;
    assignments: Map<string, ShardAssignment | null>;
    nextWorker: number;
  };
  donors: Map<string, DonorServer>;
  killDonor(id: string): Promise<void>;
  close(): Promise<void>;
}

async function bootHarness1(): Promise<Harness1> {
  // Disk-backed tmp (NOT os.tmpdir(): /tmp is a 512 MiB tmpfs here and
  // RamNode spill files would fill it).
  fs.mkdirSync(path.join(ROOT, "data"), { recursive: true });
  const tmp = fs.mkdtempSync(path.join(ROOT, "data", "failure-vram-weights-"));
  const provider = new FakeNodes([]);
  const donors = new Map<string, DonorServer>();

  for (const id of ["worker-1", "worker-2"]) {
    const ramNode = new RamNode({
      capacityBytes: 512 * MIB,
      spillDir: path.join(tmp, `${id}-spill`),
    });
    const server = await createHwNodeServer({
      routeTables: [
        ramNode.routes(),
        new GpuNode({ device: "cpu" }).routes(),
        new CpuNode({}).routes(),
        new SsdNode({ dir: path.join(tmp, `${id}-ssd`) }).routes(),
      ],
    });
    provider.add(makeWorkerInfo(id, server.port));
    donors.set(id, server);
  }

  const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });
  const gpu = new VirtualGpu({ nodes: provider, ram });
  const cpu = new VirtualCpu({ nodes: provider });
  const ssd = new VirtualSsd({
    nodes: provider,
    manifestPath: path.join(tmp, "vssd-manifest.json"),
    ram,
  });
  const hypervisor = new Hypervisor({
    nodes: provider,
    ram,
    gpu,
    cpu,
    ssd,
    sweepIntervalMs: 3_600_000, // effectively disabled: healing is
    // exercised explicitly via handleNodeLoss, and a 60 s timer raced
    // the test (a run lasting >60 s let the sweep heal first, so the
    // explicit call found 0 pages to recover and the assertion failed).
  });
  const vm = await hypervisor.boot();
  const registry = new ModelRegistry({
    vm: () => vm,
    stageToVram: (key: string) => hypervisor.stageToVram(key),
    readVram: (vaddr: number, length: number) => vm.ram.read(vaddr, length),
    isActive: () => false,
    modelsDir: tmp,
  });

  // Fake orchestrator: the real WorkerRegistry protocol (register /
  // heartbeat / getAssignment) plus the real binary weight endpoint
  // handlers. Assignment versions are bumped by the test to re-plan.
  const orchState: Harness1["orchState"] = {
    version: 0,
    assignments: new Map(),
    nextWorker: 1,
  };
  const weightHandlers = createWeightHandlers(registry);
  const orchServer = createMeshServer({
    register: () => ({
      workerId: `worker-${orchState.nextWorker++}`,
      heartbeatIntervalMs: 50,
    }),
    heartbeat: () => ({ ok: true, assignmentVersion: orchState.version }),
    getAssignment: (workerId: string) => ({
      version: orchState.version,
      assignment: orchState.assignments.get(workerId) ?? null,
    }),
    getWeightMap: weightHandlers.getWeightMap,
    getWeightTensor: weightHandlers.getWeightTensor,
  });
  await new Promise<void>((resolve, reject) => {
    orchServer.once("error", reject);
    orchServer.listen(0, "127.0.0.1", () => resolve());
  });
  const orchUrl = `http://127.0.0.1:${(orchServer.address() as AddressInfo).port}`;

  return {
    tmp,
    provider,
    vm,
    ram,
    hypervisor,
    registry,
    orchServer,
    orchUrl,
    orchState,
    donors,
    async killDonor(id: string): Promise<void> {
      const donor = donors.get(id);
      if (donor === undefined) throw new Error(`no such donor: ${id}`);
      await donor.close().catch(() => undefined);
      provider.markDead(id);
    },
    async close(): Promise<void> {
      await hypervisor.shutdown().catch(() => undefined);
      await new Promise<void>((resolve) => orchServer.close(() => resolve()));
      for (const donor of donors.values()) {
        await donor.close().catch(() => undefined);
      }
      fs.rmSync(tmp, { recursive: true, force: true });
    },
  };
}

function makeAssignment(
  workerId: string,
  opts: {
    stage: number;
    layerStart: number;
    layerEnd: number;
    hasEmbedding: boolean;
    hasLmHead: boolean;
    modelId: string;
    modelDir: string;
  },
): ShardAssignment {
  return {
    workerId,
    stage: opts.stage,
    layerStart: opts.layerStart,
    layerEnd: opts.layerEnd,
    hasEmbedding: opts.hasEmbedding,
    hasLmHead: opts.hasLmHead,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 1024,
    modelId: opts.modelId,
    modelDir: opts.modelDir,
  };
}

test(
  "mid-generation donor+worker death: survivor re-loads from vRAM, generation completes identically",
  { timeout: 300_000 },
  async () => {
    const logs = captureLogs();
    const h = await bootHarness1();
    const client = new MeshClient({ timeoutMs: 30_000 });
    let w1: WorkerHandle | undefined;
    let w2: WorkerHandle | undefined;
    const spawnLoaders: SafetensorsWeightLoader[] = [];
    // Private temp copy of the model: the shared demo/models/demo-100m
    // directory is renamed concurrently by a sibling workstream, so
    // this test never touches it — it serves its own copy and moves
    // THAT out of reach after staging.
    const copyDir = path.join(h.tmp, "demo-100m");
    const hiddenDir = path.join(h.tmp, "demo-100m-hidden");
    try {
      fs.cpSync(DEMO_100M_DIR, copyDir, { recursive: true });
      const manifest = loadManifest(copyDir, "demo-100m");
      assert.equal(manifest.config.numHiddenLayers, 12, "demo-100m has 12 layers");
      const eosTokenId = manifest.config.eosTokenId as number;
      const tokenizer = loadTokenizer(DEMO_100M_DIR);
      const promptIds = tokenizer.encode("Hello world");
      assert.ok(promptIds.length > 0, "prompt must tokenize to at least one id");

      // Stage demo-100m in vRAM (vSSD -> vRAM, weight map built).
      const entry = await h.registry.load(copyDir);
      assert.equal(entry.status, "ready", `registry load: ${entry.error ?? ""}`);
      const weightMap = h.registry.getWeightMap("demo-100m");
      assert.ok(weightMap !== undefined, "staged model must have a weight map");
      assert.equal(
        Object.keys(weightMap).length,
        Object.keys(manifest.tensors).length,
        "weight map must cover every tensor",
      );
      // Do NOT hold the 417 MB shard in memory for the whole test
      // (it plus the workers plus the donors OOM'd this host): stat
      // the size now, and compare bytes chunk-wise against the file
      // on disk at read-back time.
      const shardSize = fs.statSync(path.join(copyDir, "model.safetensors")).size;
      const weightVaddr = entry.weightVaddr as number;
      // entry.weightBytes is the manifest's tensor-bytes total, NOT
      // the shard file size (the file also carries the 8-byte length
      // prefix + JSON header). The staged vRAM allocation holds the
      // whole shard file, so its size is the file size.
      assert.equal(
        entry.vramBytes,
        shardSize,
        "staged vRAM bytes must be the shard size",
      );
      const weightBytes = shardSize;
      console.log(
        `  staged demo-100m: ${Object.keys(weightMap).length} tensors, ` +
          `${(weightBytes / MIB).toFixed(1)} MiB in vRAM`,
      );

      // Two workers. Their spawn-time disk loaders are demo-10m (a
      // DIFFERENT model id), so the demo-100m assignment can never
      // resolve to the spawn loader — it must go through the vRAM
      // path (or the disk fallback, which the log + moved dir expose).
      const backend = createReferenceBackend();
      const createCache = (m: ModelManifest): PagedKVCache =>
        new BlockPagedKVCache(
          16,
          1024,
          m.config.numKeyValueHeads,
          m.config.headDim,
        );
      const startOne = async (): Promise<WorkerHandle> => {
        const spawnLoader = openWeightLoader(DEMO_10M_DIR);
        spawnLoaders.push(spawnLoader);
        return startWorker({
          orchestratorUrl: h.orchUrl,
          host: "127.0.0.1",
          port: 0,
          capability: {
            device: "cpu",
            backend: backend.name,
            memoryBytes: 8 * 1024 * 1024 * 1024,
            freeMemoryBytes: 8 * 1024 * 1024 * 1024,
            flopsPerSec: 1e12,
            bandwidthBytesPerSec: 1e9,
            cores: 4,
            roles: ["mixed"],
            maxBatchTokens: 256,
          },
          backend,
          loader: spawnLoader,
          createCache,
          pollIntervalMs: 50,
        });
      };
      w1 = await startOne();
      w2 = await startOne();
      if (w1 === undefined || w2 === undefined) throw new Error("workers failed to start");
      const worker1: WorkerHandle = w1;
      const worker2: WorkerHandle = w2;
      // Capture the mesh URLs now: decodeOne() below must keep
      // addressing the dead worker by its (closed) port after the
      // kill, not throw a TypeError on an undefined handle.
      const w1Url = `http://127.0.0.1:${worker1.port}`;
      const w2Url = `http://127.0.0.1:${worker2.port}`;
      console.log(`  workers up: ${worker1.workerId} (stage 0) and ${worker2.workerId} (stage 1)`);

      // 2-stage plan, version 1.
      h.orchState.assignments.set(
        worker1.workerId,
        makeAssignment(worker1.workerId, {
          stage: 0, layerStart: 0, layerEnd: 6,
          hasEmbedding: true, hasLmHead: false,
          modelId: "demo-100m", modelDir: copyDir,
        }),
      );
      h.orchState.assignments.set(
        worker2.workerId,
        makeAssignment(worker2.workerId, {
          stage: 1, layerStart: 6, layerEnd: 12,
          hasEmbedding: false, hasLmHead: true,
          modelId: "demo-100m", modelDir: copyDir,
        }),
      );
      h.orchState.version = 1;
      await waitFor(
        () =>
          logs.lines.some((l) =>
            l.includes("loaded stage 0 (layers 0-6") && l.includes("via vRAM weights"),
          ) &&
          logs.lines.some((l) =>
            l.includes("loaded stage 1 (layers 6-12") && l.includes("via vRAM weights"),
          ),
        "both stages loaded via vRAM",
        120_000,
      );
      assert.ok(
        logs.lines.some((l) => l.includes("using vRAM weight loader for model demo-100m")),
        "workers must log the vRAM loader choice on first load",
      );

      // Move the model directory out of reach: from here on, any disk
      // fallback fails loudly (openWeightLoader throws ENOENT) instead
      // of silently succeeding.
      fs.renameSync(copyDir, hiddenDir);
      console.log("  model dir moved out of reach; all further loads must come from vRAM");

      const twoStage: PipelineStage[] = [
        { workerId: worker1.workerId, url: w1Url, stage: 0 },
        { workerId: worker2.workerId, url: w2Url, stage: 1 },
      ];
      const tokensRef = await runPipeline(
        client, twoStage, promptIds, 6, "seq-ref", eosTokenId,
      );
      assert.ok(tokensRef.length > 0, "reference generation must produce tokens");
      console.log(
        `  reference generation (2 stages, dir gone): ${tokensRef.length} tokens ` +
          `[${tokensRef.join(", ")}]`,
      );

      // --- Mid-generation kill. ---
      // Start a second generation, run prefill + 2 decode tokens, then
      // kill worker-2 (compute worker AND vRAM donor) mid-flight.
      const seqMid = "seq-mid";
      const pre0 = await client.execute(w1Url, {
        workerId: worker1.workerId,
        seqIds: promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 0,
        input: { data: [], shape: [0] },
        positions: promptIds.map((_, i) => i),
        tokenIds: [...promptIds],
      });
      const pre1 = await client.execute(w2Url, {
        workerId: worker2.workerId,
        seqIds: promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 1,
        input: pre0.output,
        positions: promptIds.map((_, i) => i),
      });
      assert.equal(pre1.isLogits, true);
      let next = argmaxLastRow(pre1.output);
      let pos = promptIds.length;
      const decodeOne = async (tok: number, p: number): Promise<number> => {
        const d0 = await client.execute(w1Url, {
          workerId: worker1.workerId,
          seqIds: [seqMid],
          kind: "decode",
          stage: 0,
          input: { data: [], shape: [0] },
          positions: [p],
          tokenIds: [tok],
        });
        const d1 = await client.execute(w2Url, {
          workerId: worker2.workerId,
          seqIds: [seqMid],
          kind: "decode",
          stage: 1,
          input: d0.output,
          positions: [p],
        });
        return argmaxLastRow(d1.output);
      };
      next = await decodeOne(next, pos);
      pos += 1;
      next = await decodeOne(next, pos);
      pos += 1;
      console.log(`  generation in flight (2 decode tokens done); killing ${worker2.workerId} now`);

      const victimId = worker2.workerId;
      await worker2.close();
      w2 = undefined;
      await h.killDonor(victimId);
      // The in-flight generation is interrupted: the next decode step
      // is routed to the dead worker and must fail (not hang). This is
      // the documented contract — the sequence must be resubmitted
      // after healing (its KV cache died with the worker).
      await assert.rejects(
        decodeOne(next, pos),
        "decode routed to the dead worker must reject, not hang",
      );
      console.log("  in-flight decode correctly rejected after the kill (must resubmit)");

      // --- Re-plan BEFORE healing. ---
      // The survivor's re-load fetches its newly-owned layers
      // (6-11 + lm_head + norm) from the weight endpoint while the
      // donor is still dead and unhealed: the reads must fail over to
      // the surviving replica inside the same call.
      const servedBeforeReplan = h.registry.weightBytesServed;
      h.orchState.assignments.set(
        worker1.workerId,
        makeAssignment(worker1.workerId, {
          stage: 0, layerStart: 0, layerEnd: 12,
          hasEmbedding: true, hasLmHead: true,
          modelId: "demo-100m", modelDir: copyDir, // copyDir no longer exists
        }),
      );
      h.orchState.version = 2;
      await waitFor(
        () =>
          logs.lines.some((l) =>
            l.includes("loaded stage 0 (layers 0-12") &&
            l.includes("assignment v2") &&
            l.includes("via vRAM weights"),
          ),
        "survivor re-loaded the full stage via vRAM (donor still dead)",
        120_000,
      );
      assert.ok(
        h.registry.weightBytesServed > servedBeforeReplan,
        `endpoint must have served the re-load from vRAM (served ${h.registry.weightBytesServed}, was ${servedBeforeReplan})`,
      );
      console.log(
        `  survivor re-loaded full stage via vRAM while donor dead; ` +
          `endpoint served ${h.registry.weightBytesServed - servedBeforeReplan} re-load bytes`,
      );

      // --- Heal the pool, then verify durability. ---
      const healed = await h.hypervisor.handleNodeLoss(victimId);
      assert.deepEqual(healed.errors, [], "no pool may fail healing");
      assert.equal(healed.ram.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      assert.ok(
        healed.ram.pagesRecovered > 0,
        `healing must recover pages: ${JSON.stringify(healed)}`,
      );
      // NOTE: no exact pagesRecovered assertion here. The staged
      // weights are ~105 pages, but vSSD's putObject also caches the
      // object in vRAM (~105 more pages), so a donor loss recovers
      // ~210 pages, not 105. The sibling healing file asserts the
      // exact count and fails on this (pre-existing; not this file).
      const afterHeal = await h.registry.get("demo-100m");
      assert.equal(afterHeal?.status, "ready", "registry must still report the model ready");
      // Chunked comparison (16 MiB at a time) against the shard file:
      // a single 417 MB vm.ram.read + Buffer copy peaked the host
      // into an OOM kill. Every weight byte is still compared.
      {
        const fd = fs.openSync(path.join(hiddenDir, "model.safetensors"), "r");
        try {
          const CHUNK = 16 * MIB;
          const fileBuf = Buffer.alloc(CHUNK);
          let offset = 0;
          while (offset < weightBytes) {
            const len = Math.min(CHUNK, weightBytes - offset);
            const read = fs.readSync(fd, fileBuf, 0, len, offset);
            assert.equal(read, len, `short read of shard file at offset ${offset}`);
            const fromVram = await h.vm.ram.read(weightVaddr + offset, len);
            assert.deepEqual(
              Buffer.from(fromVram),
              fileBuf.subarray(0, len),
              `weight bytes mismatch at offset ${offset}`,
            );
            offset += len;
          }
        } finally {
          fs.closeSync(fd);
        }
      }
      console.log(
        `  healed: pagesRecovered=${healed.ram.pagesRecovered}, pagesLost=0; ` +
          `weights byte-identical from vRAM`,
      );

      // --- Resubmitted generation completes, identically. ---
      const oneStage: PipelineStage[] = [
        { workerId: worker1.workerId, url: w1Url, stage: 0 },
      ];
      const tokensHealed = await runPipeline(
        client, oneStage, promptIds, 6, "seq-healed", eosTokenId,
      );
      assert.deepEqual(
        tokensHealed,
        tokensRef,
        "post-heal generation on the survivor must produce the same tokens as the pre-kill reference",
      );
      console.log(
        `  post-heal generation (1 stage, dir still gone): ${tokensHealed.length} tokens, ` +
          `identical to the reference`,
      );

      // The re-loaded stage never touched disk.
      assert.ok(
        !logs.lines.some((l) => l.includes("disk weight loader")),
        "no worker may ever log the disk loader fallback",
      );
      assert.ok(
        !logs.lines.some((l) => l.includes("cannot fetch vRAM weight map")),
        "no worker may fail to fetch the vRAM weight map",
      );
      for (const loader of spawnLoaders) {
        assert.equal(loader.bytesReadFromDisk, 0, "spawn disk loader must read 0 bytes");
      }
      console.log("  vRAM-only proof: log shows vRAM loads only; spawn disk loaders read 0 bytes");
    } finally {
      logs.restore();
      if (w2 !== undefined) await w2.close().catch(() => undefined);
      if (w1 !== undefined) await w1.close().catch(() => undefined);
      for (const loader of spawnLoaders) {
        try {
          loader.close();
        } catch {
          // best effort
        }
      }
      await h.close();
    }
  },
);

/* ------------------------------------------------------------------ */
/* Test 2 harness: fake VM (in-memory vRAM/vSSD that really store   */
/* bytes) + fake orchestrator whose binary weight endpoint can be    */
/* made to fail, one real worker on a recording fake backend. The    */
/* worker's loaders are observable here (loadedLoaders), so the     */
/* counters are asserted directly.                                   */
/* ------------------------------------------------------------------ */

/** Fake vRAM that really stores bytes per allocation. */
class FakeRam {
  readonly totalBytes: number;
  usedBytes = 0;
  private nextVaddr = 0x1000;
  private nextId = 0;
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();

  constructor(totalBytes: number) {
    this.totalBytes = totalBytes;
  }

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    if (sizeBytes > this.totalBytes - this.usedBytes) {
      throw new Error(
        `fake vRAM out of memory: ${sizeBytes} bytes requested, ${this.totalBytes - this.usedBytes} free`,
      );
    }
    const id = `alloc-${++this.nextId}`;
    const vaddr = this.nextVaddr;
    this.nextVaddr += sizeBytes;
    this.allocs.set(id, { vaddr, data: new Uint8Array(sizeBytes) });
    this.usedBytes += sizeBytes;
    return { id, vaddr, sizeBytes, pages: [], replicas: 1 };
  }

  async free(allocId: string): Promise<void> {
    const alloc = this.allocs.get(allocId);
    if (alloc === undefined) throw new Error(`unknown allocation: ${allocId}`);
    this.allocs.delete(allocId);
    this.usedBytes -= alloc.data.length;
  }

  private find(vaddr: number): { base: number; data: Uint8Array } {
    for (const { vaddr: base, data } of this.allocs.values()) {
      if (vaddr >= base && vaddr < base + data.length) return { base, data };
    }
    throw new Error(`fake vRAM access outside any allocation: ${vaddr}`);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const { base, data } = this.find(vaddr);
    const off = vaddr - base;
    if (off + length > data.length) throw new Error("fake vRAM read past allocation end");
    return data.slice(off, off + length);
  }

  async write(vaddr: number, chunk: Uint8Array): Promise<void> {
    const { base, data } = this.find(vaddr);
    data.set(chunk, vaddr - base);
  }

  async replicate(): Promise<void> {}

  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): { totalBytes: number; usedBytes: number; freeBytes: number; pages: number; nodes: number } {
    return {
      totalBytes: this.totalBytes,
      usedBytes: this.usedBytes,
      freeBytes: this.totalBytes - this.usedBytes,
      pages: 0,
      nodes: 1,
    };
  }
}

class FakeSsd {
  private readonly objects = new Map<string, Uint8Array>();

  get(key: string): Uint8Array | undefined {
    return this.objects.get(key);
  }

  async putObject(key: string, data: Uint8Array): Promise<{ key: string; sizeBytes: number; chunks: number; replicas: number; createdAt: number }> {
    this.objects.set(key, data);
    return { key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: Date.now() };
  }

  async getObject(key: string): Promise<Uint8Array> {
    const data = this.objects.get(key);
    if (!data) throw new Error(`object not found: ${key}`);
    return data;
  }

  async getRange(key: string, offset: number, length: number): Promise<Uint8Array> {
    const data = await this.getObject(key);
    return data.slice(offset, offset + length);
  }

  async list(prefix = ""): Promise<Array<{ key: string; sizeBytes: number; chunks: number; replicas: number; createdAt: number }>> {
    const out: Array<{ key: string; sizeBytes: number; chunks: number; replicas: number; createdAt: number }> = [];
    for (const [key, data] of this.objects) {
      if (key.startsWith(prefix)) {
        out.push({ key, sizeBytes: data.length, chunks: 1, replicas: 1, createdAt: 0 });
      }
    }
    return out.sort((a, b) => (a.key < b.key ? -1 : 1));
  }

  async deleteObject(key: string): Promise<void> {
    this.objects.delete(key);
  }

  async handleNodeLoss(): Promise<{ chunksRecovered: number; chunksLost: number }> {
    return { chunksRecovered: 0, chunksLost: 0 };
  }

  stats(): { objects: number; totalBytes: number; nodes: number } {
    let totalBytes = 0;
    for (const data of this.objects.values()) totalBytes += data.length;
    return { objects: this.objects.size, totalBytes, nodes: 1 };
  }
}

function makeFakeVm(totalVramBytes: number): { vm: VirtualMachine; ram: FakeRam; ssd: FakeSsd } {
  const ram = new FakeRam(totalVramBytes);
  const ssd = new FakeSsd();
  const vm = {
    ram,
    ssd,
    gpu: {
      listDevices: () => [],
      createStream: () => "stream-0",
      launch: async () => {
        throw new Error("fake vGPU cannot launch kernels");
      },
      synchronize: async () => {},
      transfer: async () => {},
      handleNodeLoss: async () => ({ streamsReassigned: 0, devicesLost: 0 }),
      stats: () => ({ devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 }),
    },
    cpu: {
      registerHandler: () => {},
      submit: async () => {
        throw new Error("fake vCPU cannot run tasks");
      },
      handleNodeLoss: async () => ({ tasksRequeued: 0, tasksLost: 0 }),
      stats: () => ({ nodes: 1, totalCores: 4, queued: 0, completed: 0, failed: 0 }),
    },
    info: () => ({
      vram: ram.stats(),
      gpu: { devices: [], totalFlopsPerSec: 0, kernelsCompleted: 0, kernelsFailed: 0 },
      cpu: { nodes: 1, totalCores: 4, queued: 0, completed: 0, failed: 0 },
      ssd: ssd.stats(),
      nodes: [],
    }),
  } as unknown as VirtualMachine;
  return { vm, ram, ssd };
}

function makeStageToVram(ssd: FakeSsd, ram: FakeRam): (key: string) => Promise<VramAllocation> {
  return async (key: string) => {
    const data = ssd.get(key);
    if (!data) throw new Error(`stageToVram: object not found in vSSD: ${key}`);
    const alloc = await ram.alloc(data.length);
    await ram.write(alloc.vaddr, data);
    return alloc;
  };
}

function makeStubCache(): PagedKVCache {
  return {
    blockSize: 16,
    allocate: () => [],
    append: () => {},
    read: () => ({ keys: new Float32Array(0), values: new Float32Array(0) }),
    free: () => {},
    usedBlocks: () => 0,
    totalBlocks: () => 0,
  };
}

test(
  "weight-endpoint fetch failure mid-load: worker keeps the old stage, retries, loads from vRAM",
  { timeout: 120_000 },
  async () => {
    const logs = captureLogs();
    const tmp = fs.mkdtempSync(path.join(ROOT, "data", "failure-vram-retry-"));
    let orchServer: MeshServer | undefined;
    let workerHandle: WorkerHandle | undefined;
    let spawnCopyLoader: SafetensorsWeightLoader | undefined;
    // The spawn loader must be a THIRD model id ("spawn-model"), so
    // neither staged model can resolve to the spawn-time disk loader.
    const spawnDir = path.join(tmp, "spawn-model");
    try {
      fs.cpSync(DEMO_10M_DIR, spawnDir, { recursive: true });
      spawnCopyLoader = openWeightLoader(spawnDir);
      assert.equal(
        spawnCopyLoader.manifest.modelId,
        "spawn-model",
        "spawn copy must carry its own model id",
      );

      const { vm, ram, ssd } = makeFakeVm(4 * 1024 ** 3);
      const registry = new ModelRegistry({
        vm: () => vm,
        stageToVram: makeStageToVram(ssd, ram),
        readVram: (vaddr, length) => ram.read(vaddr, length),
        isActive: () => false,
        modelsDir: tmp,
      });
      const entryA = await registry.load(DEMO_10M_DIR);
      assert.equal(entryA.status, "ready", `load demo-10m: ${entryA.error ?? ""}`);
      const entryB = await registry.load(DEMO_100M_DIR);
      assert.equal(entryB.status, "ready", `load demo-100m: ${entryB.error ?? ""}`);
      console.log("  staged demo-10m (A) and demo-100m (B) in the fake VM");

      // Fake orchestrator. The weight endpoint can be made to fail on
      // demand — the shape of a donor dying mid-fetch from the
      // worker's point of view (the fetch throws; the map fetch is
      // unaffected, so the worker must NOT treat this as "unstaged"
      // and fall back to disk).
      let endpointDown = false;
      const weightHandlers = createWeightHandlers(registry);
      const orchState: { version: number; assignment: ShardAssignment | null } = {
        version: 0,
        assignment: null,
      };
      orchServer = createMeshServer({
        register: () => ({ workerId: "worker-retry-1", heartbeatIntervalMs: 50 }),
        heartbeat: () => ({ ok: true, assignmentVersion: orchState.version }),
        getAssignment: () => ({
          version: orchState.version,
          assignment: orchState.assignment,
        }),
        getWeightMap: weightHandlers.getWeightMap,
        getWeightTensor: async (modelId: string, tensorName: string) => {
          if (endpointDown) {
            throw new Error(
              "simulated weight-endpoint failure (donor died mid-fetch)",
            );
          }
          return weightHandlers.getWeightTensor(modelId, tensorName);
        },
      });
      await new Promise<void>((resolve, reject) => {
        orchServer?.once("error", reject);
        orchServer?.listen(0, "127.0.0.1", () => resolve());
      });
      const orchUrl = `http://127.0.0.1:${(orchServer.address() as AddressInfo).port}`;
      const client = new MeshClient({ timeoutMs: 5000 });

      // Recording backend: load() records the loader ONLY after a
      // tensor fetch succeeds (a failed load records nothing), and
      // forward() returns a per-model marker so executes prove which
      // stage is actually serving.
      const loadedLoaders: WeightLoader[] = [];
      const backend: EngineBackend = {
        name: "fake-recording",
        createStageExecutor(manifest: ModelManifest, assignment: ShardAssignment): StageExecutor {
          const marker = manifest.modelId === "demo-10m" ? 10 : 100;
          return {
            assignment,
            load: async (loader: WeightLoader) => {
              const first = Object.keys(loader.manifest.tensors)[0] as string;
              const data = await loader.loadTensor(first);
              assert.ok(data.length > 0, "loaded tensor must be non-empty");
              loadedLoaders.push(loader);
            },
            embed: (tokenIds: number[]): Tensor => ({
              data: new Float32Array(tokenIds.length * 2),
              shape: [tokenIds.length, 2],
            }),
            forward: (_input: Tensor, _ctx: ForwardContext): Tensor => ({
              data: new Float32Array([marker]),
              shape: [1, 1],
            }),
            memoryBytes: () => 4096,
          };
        },
      };

      workerHandle = await startWorker({
        orchestratorUrl: orchUrl,
        host: "127.0.0.1",
        port: 0,
        capability: {
          device: "cpu",
          backend: "fake-recording",
          memoryBytes: 1 << 30,
          freeMemoryBytes: 1 << 30,
          flopsPerSec: 1e12,
          bandwidthBytesPerSec: 1e9,
          roles: ["mixed"],
          maxBatchTokens: 256,
        },
        backend,
        loader: spawnCopyLoader,
        createCache: makeStubCache,
        pollIntervalMs: 50,
      });
      const workerId = workerHandle.workerId;
      const workerUrl = `http://127.0.0.1:${workerHandle.port}`;
      const execMarker = async (): Promise<number> => {
        const resp = await client.execute(workerUrl, {
          workerId,
          seqIds: ["seq-1", "seq-1"],
          kind: "prefill",
          stage: 0,
          input: { data: [1, 2, 3, 4], shape: [2, 2] },
          positions: [0, 1],
          tokenIds: [1, 2],
        });
        return resp.output.data[0] as number;
      };

      // (1) Model A loads from vRAM and serves.
      orchState.assignment = makeAssignment(workerId, {
        stage: 0, layerStart: 0, layerEnd: 6,
        hasEmbedding: true, hasLmHead: true,
        modelId: "demo-10m", modelDir: DEMO_10M_DIR,
      });
      orchState.version = 1;
      await waitFor(
        () =>
          loadedLoaders.some(
            (l) => l instanceof VramWeightLoader && l.manifest.modelId === "demo-10m",
          ),
        "model A stage load via vRAM",
      );
      assert.equal(await execMarker(), 10, "model A must be serving");
      console.log("  model A serving from vRAM (marker 10)");

      // (2) Break the weight endpoint, then re-plan to model B. The
      // worker's load attempts fail; it must keep serving A.
      endpointDown = true;
      orchState.assignment = makeAssignment(workerId, {
        stage: 0, layerStart: 0, layerEnd: 12,
        hasEmbedding: true, hasLmHead: true,
        modelId: "demo-100m", modelDir: DEMO_100M_DIR,
      });
      orchState.version = 2;
      await sleep(400); // several heartbeat cycles of failed loads
      assert.ok(
        logs.lines.some((l) => l.includes("failed to load assignment v2")),
        "worker must log the failed load attempts (it retries, it does not give up)",
      );
      assert.equal(await execMarker(), 10, "old stage (A) must keep serving while B fails to load");
      assert.ok(
        !loadedLoaders.some(
          (l) => l instanceof VramWeightLoader && l.manifest.modelId === "demo-100m",
        ),
        "model B must not be recorded as loaded while the endpoint is down",
      );
      assert.ok(
        !loadedLoaders.some((l) => l instanceof SafetensorsWeightLoader),
        "no disk loader may be constructed while the vRAM fetch fails",
      );
      console.log("  endpoint down: worker kept serving A, logged retries, built no disk loader");

      // (3) Heal the endpoint: the worker retries and loads B from vRAM.
      endpointDown = false;
      await waitFor(
        () =>
          loadedLoaders.some(
            (l) =>
              l instanceof VramWeightLoader &&
              l.manifest.modelId === "demo-100m" &&
              (l as VramWeightLoader).tensorsFetched >= 1,
          ),
        "model B stage load via vRAM after the endpoint recovers",
      );
      assert.equal(await execMarker(), 100, "model B must be serving after recovery");
      const loaderB = loadedLoaders.find(
        (l) => l instanceof VramWeightLoader && l.manifest.modelId === "demo-100m",
      ) as VramWeightLoader;
      assert.ok(loaderB.bytesFetched > 0, "B's vRAM loader must have fetched bytes");
      assert.ok(
        !loadedLoaders.some((l) => l instanceof SafetensorsWeightLoader),
        "no disk loader may be constructed at any point",
      );
      assert.equal(
        spawnCopyLoader.bytesReadFromDisk,
        0,
        "spawn disk loader must read 0 bytes",
      );
      console.log(
        `  endpoint recovered: B serving from vRAM (marker 100), ` +
          `tensorsFetched=${loaderB.tensorsFetched}, bytesFetched=${loaderB.bytesFetched}`,
      );
    } finally {
      logs.restore();
      if (workerHandle !== undefined) await workerHandle.close().catch(() => undefined);
      if (orchServer !== undefined) {
        await new Promise<void>((resolve) => orchServer?.close(() => resolve()));
      }
      if (spawnCopyLoader !== undefined) {
        try {
          spawnCopyLoader.close();
        } catch {
          // best effort
        }
      }
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  },
);
