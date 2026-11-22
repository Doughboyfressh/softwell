/**
 * Worker model-switching tests — run with: node test/worker-model-switch.test.ts
 * (exit 0 on pass).
 *
 * A real worker (startWorker) runs in-process against a fake orchestrator
 * (a real mesh server whose register/heartbeat/getAssignment handlers the
 * test controls) and a fake backend whose stage executor records which
 * manifest it was built from.
 *
 *  - first assignment loads model A (demo/models/demo-10m)
 *  - bumping the assignment version to model B (a copy of demo-10m under a
 *    different directory name, so its manifest modelId differs) makes the
 *    worker load the stage for model B, and execute() works on the new stage
 *  - switching back to model A reuses the cached loader
 *  - an assignment without modelDir keeps using the spawn-time loader
 *  - an assignment pointing at a nonexistent model dir does NOT crash the
 *    worker and the old stage keeps serving
 *
 * No real engine execution: the backend is a stub throughout.
 */
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import type {
  EngineBackend,
  ForwardContext,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

async function waitFor(
  cond: () => boolean,
  what: string,
  timeoutMs = 5000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cond()) return;
    if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
    await sleep(10);
  }
}

const DEMO_MODEL_A_DIR = fileURLToPath(
  new URL("../demo/models/demo-10m", import.meta.url),
);

function makeCache(): PagedKVCache {
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

function makeAssignment(
  workerId: string,
  opts?: { modelId?: string; modelDir?: string },
): ShardAssignment {
  return {
    workerId,
    stage: 0,
    layerStart: 0,
    layerEnd: 6,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 1024,
    ...(opts?.modelId !== undefined ? { modelId: opts.modelId } : {}),
    ...(opts?.modelDir !== undefined ? { modelDir: opts.modelDir } : {}),
  };
}

async function main(): Promise<void> {
  // --- Fake orchestrator state: the test drives the assignment version. ---
  const orchState: { version: number; assignment: ShardAssignment | null } = {
    version: 0,
    assignment: null,
  };
  const server = createMeshServer({
    register: () => ({ workerId: "worker-test-1", heartbeatIntervalMs: 50 }),
    heartbeat: () => ({ ok: true, assignmentVersion: orchState.version }),
    getAssignment: () => ({
      version: orchState.version,
      assignment: orchState.assignment,
    }),
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const orchPort = (server.address() as AddressInfo).port;
  const orchUrl = `http://127.0.0.1:${orchPort}`;
  const client = new MeshClient({ timeoutMs: 5000 });

  // --- Fake backend: records which manifest each stage was built from
  // and which loader instance load() received. ---
  const builtFromManifests: string[] = [];
  const loadedLoaders: WeightLoader[] = [];
  const cacheManifests: string[] = [];
  const backend: EngineBackend = {
    name: "fake-recording",
    createStageExecutor(manifest: ModelManifest, assignment: ShardAssignment): StageExecutor {
      builtFromManifests.push(manifest.modelId);
      return {
        assignment,
        load: async (loader: WeightLoader) => {
          loadedLoaders.push(loader);
        },
        embed: (tokenIds: number[]): Tensor => ({
          data: new Float32Array(tokenIds.length * 2),
          shape: [tokenIds.length, 2],
        }),
        forward: (input: Tensor, _ctx: ForwardContext): Tensor => input,
        memoryBytes: () => 4096,
      };
    },
  };

  const spawnLoader = openWeightLoader(DEMO_MODEL_A_DIR);
  assert.equal(spawnLoader.manifest.modelId, "demo-10m");

  // Model B: a copy of demo-10m under a different directory name, so
  // loadManifest derives a different modelId from the basename.
  const modelBDir = path.join(os.tmpdir(), `softwell-demo-10m-b-${process.pid}`);
  fs.cpSync(DEMO_MODEL_A_DIR, modelBDir, { recursive: true });
  const modelBId = path.basename(modelBDir);
  assert.notEqual(modelBId, "demo-10m");

  let workerHandle: Awaited<ReturnType<typeof startWorker>> | undefined;
  try {
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
      loader: spawnLoader,
      createCache: (manifest: ModelManifest) => {
        cacheManifests.push(manifest.modelId);
        return makeCache();
      },
      pollIntervalMs: 50,
    });
    const workerId = workerHandle.workerId;
    const workerUrl = `http://127.0.0.1:${workerHandle.port}`;

    const execute = async (): Promise<void> => {
      const resp = await client.execute(workerUrl, {
        workerId,
        seqIds: ["seq-1", "seq-1"],
        kind: "prefill",
        stage: 0,
        input: { data: [1, 2, 3, 4], shape: [2, 2] },
        positions: [0, 1],
      });
      assert.deepEqual(resp.output, { data: [1, 2, 3, 4], shape: [2, 2] });
      assert.equal(resp.isLogits, true);
      assert.equal(resp.tokensProcessed, 2);
    };

    // --- (1) First assignment: model A, via an explicit modelDir. ---
    orchState.assignment = makeAssignment(workerId, {
      modelId: "demo-10m",
      modelDir: DEMO_MODEL_A_DIR,
    });
    orchState.version = 1;
    await waitFor(() => builtFromManifests.at(-1) === "demo-10m", "model A stage load");
    assert.equal(cacheManifests.at(-1), "demo-10m");
    // Model A is the spawn-time model, so the spawn loader is reused.
    assert.equal(loadedLoaders.at(-1), spawnLoader);
    await execute();
    console.log("worker loads model A from assignment: ok");

    // --- (2) Bump to model B: the worker must load the new model's stage. ---
    orchState.assignment = makeAssignment(workerId, {
      modelId: modelBId,
      modelDir: modelBDir,
    });
    orchState.version = 2;
    await waitFor(() => builtFromManifests.at(-1) === modelBId, "model B stage load");
    assert.equal(cacheManifests.at(-1), modelBId);
    assert.notEqual(loadedLoaders.at(-1), spawnLoader);
    assert.equal(loadedLoaders.at(-1)?.manifest.modelId, modelBId);
    await execute();
    console.log("worker switches to model B and serves it: ok");

    // --- (3) Switch back to model A: the cached loader is reused. ---
    const loadsBeforeSwitchBack = loadedLoaders.length;
    orchState.assignment = makeAssignment(workerId, {
      modelId: "demo-10m",
      modelDir: DEMO_MODEL_A_DIR,
    });
    orchState.version = 3;
    await waitFor(
      () => loadedLoaders.length > loadsBeforeSwitchBack && builtFromManifests.at(-1) === "demo-10m",
      "switch back to model A",
    );
    assert.equal(loadedLoaders.at(-1), spawnLoader);
    await execute();
    console.log("worker switches back to model A reusing the cached loader: ok");

    // --- (4) Assignment without modelDir: keep the spawn-time loader. ---
    orchState.assignment = makeAssignment(workerId);
    orchState.version = 4;
    const loadsBeforeNoDir = loadedLoaders.length;
    await waitFor(
      () => loadedLoaders.length > loadsBeforeNoDir,
      "reload for assignment without modelDir",
    );
    assert.equal(builtFromManifests.at(-1), "demo-10m");
    assert.equal(loadedLoaders.at(-1), spawnLoader);
    await execute();
    console.log("assignment without modelDir keeps the spawn-time loader: ok");

    // --- (5) Assignment pointing at a nonexistent model dir: no crash,
    // the old stage keeps serving, and the worker retries on later polls. ---
    const loadsBeforeBad = loadedLoaders.length;
    orchState.assignment = makeAssignment(workerId, {
      modelId: "ghost-model",
      modelDir: path.join(os.tmpdir(), `softwell-no-such-model-${process.pid}`),
    });
    orchState.version = 5;
    // Several heartbeat polls go by (50ms interval); the worker must stay
    // alive and keep serving the old stage throughout.
    await sleep(400);
    assert.equal(
      loadedLoaders.length,
      loadsBeforeBad,
      "bad model dir must not trigger a stage load",
    );
    assert.equal(builtFromManifests.at(-1), "demo-10m");
    await execute();
    // The worker is still heartbeating (it would otherwise stop polling
    // and the execute above would have failed on a dead worker).
    console.log("nonexistent model dir does not crash the worker; old stage keeps serving: ok");

    // --- (6) Recovery: a good assignment after the bad one loads again. ---
    orchState.assignment = makeAssignment(workerId, {
      modelId: modelBId,
      modelDir: modelBDir,
    });
    orchState.version = 6;
    await waitFor(() => builtFromManifests.at(-1) === modelBId, "recovery load of model B");
    await execute();
    console.log("worker recovers with a good assignment after a bad one: ok");

    console.log("All worker model-switch tests passed.");
  } finally {
    if (workerHandle !== undefined) await workerHandle.close();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    spawnLoader.close();
    fs.rmSync(modelBDir, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error("worker model-switch test FAILED:", err);
  process.exitCode = 1;
});
