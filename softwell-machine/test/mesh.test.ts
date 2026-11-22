/**
 * Mesh smoke test — run with: node test/mesh.test.ts (exit 0 on pass).
 *
 * Covers, in-process and with fake engine pieces (the real backend,
 * loader, and KV cache have not landed yet):
 *  - registry + mesh server: register 2 fake workers via MeshClient,
 *    heartbeat them, set an assignment, verify getAssignment
 *  - sweep() with faked time: suspect after HEARTBEAT_TIMEOUT_MS, dead
 *    after HEARTBEAT_DEAD_MS
 *  - a worker process object (startWorker) with a fake echo backend:
 *    registers, heartbeats, picks up its assignment, and executes one
 *    fake stage through POST /mesh/execute
 */
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { WorkerRegistry } from "../src/mesh/registry.ts";
import { MeshClient, MeshError, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import {
  HEARTBEAT_DEAD_MS,
  HEARTBEAT_TIMEOUT_MS,
} from "../src/contracts.ts";
import type {
  EngineBackend,
  ForwardContext,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  ShardPlan,
  StageExecutor,
  Tensor,
  WeightLoader,
  WorkerCapability,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function makeCapability(port: number): WorkerCapability {
  return {
    host: "127.0.0.1",
    port,
    device: "cpu",
    backend: "fake-echo",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 256,
  };
}

function makeManifest(): ModelManifest {
  return {
    format: "safetensors",
    modelId: "test/fake",
    config: {
      architectures: ["LlamaForCausalLM"],
      vocabSize: 256,
      hiddenSize: 16,
      intermediateSize: 32,
      numHiddenLayers: 2,
      numAttentionHeads: 2,
      numKeyValueHeads: 1,
      headDim: 8,
      maxPositionEmbeddings: 1024,
      rmsNormEps: 1e-5,
      ropeTheta: 10000,
      tieWordEmbeddings: false,
      bosTokenId: 1,
      eosTokenId: 2,
      padTokenId: 0,
    },
    metadata: {},
    tensors: {},
    files: [],
    totalBytes: 0,
    totalParams: 0,
  };
}

function makeLoader(): WeightLoader {
  return {
    manifest: makeManifest(),
    hasTensor: () => true,
    loadTensor: async () => new Float32Array([0]),
    close: () => {},
  };
}

function makeCache(_manifest: ModelManifest): PagedKVCache {
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

/** StageExecutor that echoes its input hidden states back unchanged. */
function makeEchoBackend(): EngineBackend {
  return {
    name: "fake-echo",
    createStageExecutor(_manifest: ModelManifest, assignment: ShardAssignment): StageExecutor {
      return {
        assignment,
        load: async () => {},
        embed: (tokenIds: number[]): Tensor => ({
          data: new Float32Array(tokenIds.length * 2),
          shape: [tokenIds.length, 2],
        }),
        forward: (input: Tensor, _ctx: ForwardContext): Tensor => input,
        memoryBytes: () => 4096,
      };
    },
  };
}

function makePlan(workerId: string): ShardPlan {
  return {
    modelId: "test/fake",
    totalParams: 0,
    totalWeightBytes: 0,
    pipelineStages: 1,
    tensorParallelSize: 1,
    expertParallelSize: 1,
    assignments: [
      {
        workerId,
        stage: 0,
        layerStart: 0,
        layerEnd: 2,
        hasEmbedding: true,
        hasLmHead: false,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 1024,
      },
    ],
    estimatedTokensPerSec: 100,
    estimatedTtftMs: 10,
    bottleneck: "test",
    createdAt: new Date().toISOString(),
  };
}

async function main(): Promise<void> {
  const registry = new WorkerRegistry();
  const client = new MeshClient({ timeoutMs: 5000 });

  // Orchestrator-side mesh server backed by the registry. No execute
  // handler: the orchestrator must 404 on POST /mesh/execute.
  const server = createMeshServer({
    register: (req) => registry.register(req.capability),
    heartbeat: (req) => registry.heartbeat(req),
    getAssignment: (workerId) => registry.getAssignment(workerId),
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const orchPort = (server.address() as AddressInfo).port;
  const orchUrl = `http://127.0.0.1:${orchPort}`;

  let workerHandle: Awaited<ReturnType<typeof startWorker>> | undefined;
  try {
    // --- Register 2 fake workers via MeshClient ---
    const regA = await client.register(orchUrl, makeCapability(19001));
    const regB = await client.register(orchUrl, makeCapability(19002));
    assert.notEqual(regA.workerId, regB.workerId);
    assert.ok(regA.workerId.startsWith("worker-"));
    assert.equal(registry.getWorker(regA.workerId)?.state, "joining");

    // --- Heartbeat them (joining -> healthy) ---
    const hbA = await client.heartbeat(orchUrl, {
      workerId: regA.workerId,
      usedMemoryBytes: 1000,
      activeSequences: 1,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(hbA.ok, true);
    assert.equal(hbA.assignmentVersion, 0);
    assert.equal(registry.getWorker(regA.workerId)?.state, "healthy");
    assert.equal(registry.getWorker(regA.workerId)?.usedMemoryBytes, 1000);
    assert.equal(registry.getWorker(regA.workerId)?.activeSequences, 1);

    // Throughput: tokens over a real interval must register > 0.
    await sleep(20);
    await client.heartbeat(orchUrl, {
      workerId: regA.workerId,
      usedMemoryBytes: 1000,
      activeSequences: 1,
      tokensSinceHeartbeat: 50,
    });
    assert.ok(
      registry.workerThroughput(regA.workerId) > 0,
      `expected throughput > 0, got ${registry.workerThroughput(regA.workerId)}`,
    );
    await client.heartbeat(orchUrl, {
      workerId: regB.workerId,
      usedMemoryBytes: 2000,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(registry.listWorkers().length, 2);

    // Heartbeat for an unknown worker is rejected, not a crash.
    const hbUnknown = await client.heartbeat(orchUrl, {
      workerId: "worker-nope",
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(hbUnknown.ok, false);

    // --- Assignment ---
    registry.setAssignment(makePlan(regA.workerId));
    const assignA = await client.getAssignment(orchUrl, regA.workerId);
    assert.equal(assignA.version, 1);
    assert.equal(assignA.assignment?.stage, 0);
    assert.equal(assignA.assignment?.workerId, regA.workerId);
    const assignB = await client.getAssignment(orchUrl, regB.workerId);
    assert.equal(assignB.version, 1);
    assert.equal(assignB.assignment, null);

    // --- Sweep with faked time: suspect, then dead ---
    const now = Date.now();
    const changedSuspect = registry.sweep(now + HEARTBEAT_TIMEOUT_MS + 1000);
    assert.equal(changedSuspect.length, 2);
    assert.ok(changedSuspect.every((w) => w.state === "suspect"));
    assert.equal(registry.sweep(now + HEARTBEAT_TIMEOUT_MS + 1000).length, 0);
    const changedDead = registry.sweep(now + HEARTBEAT_DEAD_MS + 1000);
    assert.equal(changedDead.length, 2);
    assert.ok(changedDead.every((w) => w.state === "dead"));

    // --- Worker process object with a fake echo backend ---
    workerHandle = await startWorker({
      orchestratorUrl: orchUrl,
      host: "127.0.0.1",
      port: 0,
      capability: {
        device: "cpu",
        backend: "fake-echo",
        memoryBytes: 1 << 30,
        freeMemoryBytes: 1 << 30,
        flopsPerSec: 1e12,
        bandwidthBytesPerSec: 1e9,
        roles: ["mixed"],
        maxBatchTokens: 256,
      },
      backend: makeEchoBackend(),
      loader: makeLoader(),
      createCache: makeCache,
      pollIntervalMs: 100,
    });
    assert.ok(workerHandle.workerId.startsWith("worker-"));
    // startWorker's initial heartbeat already flipped it to healthy.
    assert.equal(registry.getWorker(workerHandle.workerId)?.state, "healthy");

    // Give it an assignment and let a heartbeat pick it up + load the stage.
    registry.setAssignment(makePlan(workerHandle.workerId));
    await sleep(300);
    const workerUrl = `http://127.0.0.1:${workerHandle.port}`;
    const execResp = await client.execute(workerUrl, {
      workerId: workerHandle.workerId,
      seqIds: ["seq-1", "seq-1"],
      kind: "prefill",
      stage: 0,
      input: { data: [1, 2, 3, 4], shape: [2, 2] },
      positions: [0, 1],
    });
    assert.deepEqual(execResp.output, { data: [1, 2, 3, 4], shape: [2, 2] });
    assert.equal(execResp.isLogits, false);
    assert.equal(execResp.tokensProcessed, 2);

    // The orchestrator server must not execute: 404, as a typed MeshError.
    await assert.rejects(
      client.execute(orchUrl, {
        workerId: workerHandle.workerId,
        seqIds: ["seq-1"],
        kind: "prefill",
        stage: 0,
        input: { data: [1, 2], shape: [1, 2] },
        positions: [0],
      }),
      (err: unknown) => err instanceof MeshError && err.status === 404,
    );

    // Connection errors are typed MeshErrors, not crashes.
    await assert.rejects(
      client.getAssignment("http://127.0.0.1:1", "worker-x"),
      (err: unknown) => err instanceof MeshError && err.code === "CONNECTION_ERROR",
    );

    console.log("mesh smoke test passed");
  } finally {
    if (workerHandle !== undefined) await workerHandle.close();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

main().catch((err: unknown) => {
  console.error("mesh smoke test FAILED:", err);
  process.exitCode = 1;
});
