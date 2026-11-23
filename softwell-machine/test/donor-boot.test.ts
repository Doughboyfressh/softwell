/**
 * Donor boot tests — the core of the "add the user's computer as a
 * donor" piece (test/donor-boot.test.ts).
 *
 * A remote donor joins with no model on disk, so a worker must boot
 * with no --model-dir at all and load its first stage from the
 * orchestrator's staged vRAM weights. Proves, in two parts:
 *
 *  (1) In-process, fast: `startWorker` with no spawn-time loader
 *      registers healthy against a fake orchestrator, serves nothing
 *      yet (execute fails with "no assignment", not a disk error),
 *      and an assignment naming an unstaged model with no modelDir
 *      does not crash it — it keeps no stage and retries.
 *
 *  (2) End-to-end, real cluster: `node src/cli.ts up --workers 2`
 *      with NO --model-dir and the demo model directory moved out of
 *      reach boots both workers diskless (the log says so, no
 *      "cannot open model dir"); the demo model is then restored,
 *      loaded + activated over HTTP, and generation streams 8 tokens;
 *      finally a replacement worker spawned with NO --model-dir at
 *      all registers, one original worker is killed, the sweep
 *      re-plans over the survivors, and generation recovers with
 *      identical content — the replacement's stage came from vRAM,
 *      never from disk (zero "using disk weight loader" lines).
 *
 * The demo-10m model is used throughout (40 MB) to keep the test
 * fast; demo-100m would prove the same thing more slowly. The demo
 * directory is renamed away only while the cluster boots diskless,
 * and restored in a finally block even on failure.
 *
 * Run: node --test test/donor-boot.test.ts
 */

import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import type { AddressInfo } from "node:net";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import type {
  EngineBackend,
  ForwardContext,
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  StageExecutor,
  Tensor,
} from "../src/contracts.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const CLI = path.join(ROOT, "src", "cli.ts");
const WORKER_ENTRY = path.join(ROOT, "src", "mesh", "worker.ts");
const MODEL_DIR = path.join(ROOT, "demo", "models", "demo-10m");
const HIDDEN_DIR = path.join(ROOT, "demo", "models", `demo-10m-hidden-${process.pid}`);
const PIDFILE = path.join(ROOT, "data", "softwell.pid");
const PROMPT = "Hello world";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(
  label: string,
  check: () => boolean | Promise<boolean>,
  timeoutMs: number,
  intervalMs = 1000,
): Promise<void> {
  const started = Date.now();
  let lastError = "";
  for (;;) {
    try {
      if (await check()) return;
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`timed out waiting for ${label} (${timeoutMs} ms; last: ${lastError})`);
    }
    await sleep(intervalMs);
  }
}

// ---------------------------------------------------------------------------
// (1) In-process: a worker with no spawn-time loader.
// ---------------------------------------------------------------------------

function makeCache(): PagedKVCache {
  return {
    blockSize: 16,
    allocate: async () => [],
    append: async () => {},
    read: async () => ({ keys: new Float32Array(0), values: new Float32Array(0) }),
    free: async () => {},
    close: async () => {},
    usedBlocks: () => 0,
    totalBlocks: () => 0,
  };
}

test("diskless worker: startWorker with no loader registers and fails cleanly", async () => {
  const orchState: { version: number; assignment: ShardAssignment | null } = {
    version: 0,
    assignment: null,
  };
  const server = createMeshServer({
    register: () => ({ workerId: "worker-diskless-1", heartbeatIntervalMs: 50 }),
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
  const orchUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const client = new MeshClient({ timeoutMs: 5000 });
  const builtFrom: string[] = [];
  const backend: EngineBackend = {
    name: "fake-recording",
    createStageExecutor(manifest: ModelManifest): StageExecutor {
      builtFrom.push(manifest.modelId);
      return {
        assignment: orchState.assignment as ShardAssignment,
        load: async () => {},
        embed: (tokenIds: number[]): Tensor => ({
          data: new Float32Array(tokenIds.length * 2),
          shape: [tokenIds.length, 2],
        }),
        forward: async (input: Tensor, _ctx: ForwardContext): Promise<Tensor> => input,
        memoryBytes: () => 4096,
      };
    },
  };

  let handle: Awaited<ReturnType<typeof startWorker>> | undefined;
  try {
    // No `loader` at all: this is the diskless boot path.
    handle = await startWorker({
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
      createCache: () => makeCache(),
      pollIntervalMs: 50,
    });
    assert.equal(handle.workerId, "worker-diskless-1");
    const workerUrl = `http://127.0.0.1:${handle.port}`;

    // No assignment yet: execute fails with "no assignment", not a
    // disk error — the worker booted and is waiting, not broken.
    await assert.rejects(
      () =>
        client.execute(workerUrl, {
          workerId: handle?.workerId as string,
          seqIds: ["seq-1"],
          kind: "prefill",
          stage: 0,
          input: { data: [1, 2], shape: [1, 2] },
          positions: [0],
        }),
      /no assignment/,
    );

    // An assignment naming an unstaged model with no modelDir: the
    // worker must not crash and must not build a stage from nothing —
    // it keeps no stage (no executor was built) and retries on later
    // polls. getWeightMap 404s on the fake orchestrator (no handler),
    // so loaderFor returns null via the fetch failure path.
    orchState.assignment = {
      workerId: handle.workerId,
      stage: 0,
      layerStart: 0,
      layerEnd: 6,
      hasEmbedding: true,
      hasLmHead: true,
      tensorParallelRank: 0,
      tensorParallelSize: 1,
      experts: [],
      estimatedMemoryBytes: 1024,
      modelId: "no-such-model",
    };
    orchState.version = 1;
    await sleep(500);
    assert.equal(builtFrom.length, 0, "diskless worker built a stage with no staged weights");
    await assert.rejects(
      () =>
        client.execute(workerUrl, {
          workerId: handle?.workerId as string,
          seqIds: ["seq-1"],
          kind: "prefill",
          stage: 0,
          input: { data: [1, 2], shape: [1, 2] },
          positions: [0],
        }),
      /no assignment/,
    );
    console.log("diskless worker registers, waits, and survives an unstaged assignment: ok");
  } finally {
    await handle?.close();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

// ---------------------------------------------------------------------------
// (2) End-to-end: diskless `up` boot + diskless replacement worker.
// ---------------------------------------------------------------------------

async function freePort(): Promise<number> {
  return await new Promise<number>((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const addr = probe.address();
      const port = typeof addr === "object" && addr !== null ? addr.port : 0;
      probe.close(() => resolve(port));
    });
  });
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

async function postJson(
  url: string,
  body: unknown,
): Promise<{ status: number; body: unknown }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const parsed: unknown = await res.json().catch(() => null);
  return { status: res.status, body: parsed };
}

function workerPidsForMeshPort(meshPort: number): number[] {
  const pids: number[] = [];
  for (const name of fs.readdirSync("/proc")) {
    if (!/^\d+$/.test(name)) continue;
    try {
      const cmdline = fs.readFileSync(path.join("/proc", name, "cmdline"), "utf-8");
      if (cmdline.includes("mesh/worker.ts") && cmdline.includes(String(meshPort))) {
        pids.push(Number(name));
      }
    } catch {
      // Process exited between readdir and read; ignore.
    }
  }
  return pids.sort((a, b) => a - b);
}

test(
  "diskless donor boot: `up` with no model dir serves from vRAM, replacement worker boots with no --model-dir",
  { timeout: 300_000 },
  async () => {
    const meshPort = await freePort();
    const apiPort = await freePort();
    const apiBase = `http://127.0.0.1:${apiPort}`;
    console.log(`donor-boot test ports: mesh=${meshPort} api=${apiPort}`);
    assert.ok(
      fs.existsSync(path.join(MODEL_DIR, "model.safetensors")),
      `demo model not built: ${MODEL_DIR} — run \`npm run make-demo-model\``,
    );
    assert.ok(!fs.existsSync(HIDDEN_DIR), `stale hidden dir: ${HIDDEN_DIR}`);

    // Move the demo model out of reach BEFORE boot, so `up` has no
    // demo fallback and must boot its workers diskless.
    fs.renameSync(MODEL_DIR, HIDDEN_DIR);
    assert.ok(!fs.existsSync(MODEL_DIR), "model dir still present after rename");

    const output: string[] = [];
    const child: ChildProcess = spawn(
      process.execPath,
      [CLI, "up", "--workers", "2", "--port", String(meshPort), "--api-port", String(apiPort)],
      { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
    );
    const tail = (): string => output.slice(-40).join("");
    child.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    child.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    let replacement: ChildProcess | null = null;

    const streamGenerate = async (): Promise<{ content: string; contentChunks: number }> => {
      const res = await fetch(`${apiBase}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "demo-10m",
          messages: [{ role: "user", content: PROMPT }],
          max_tokens: 8,
          temperature: 0,
          stream: true,
        }),
        signal: AbortSignal.timeout(30_000),
      });
      assert.equal(res.status, 200, `chat completions: HTTP ${res.status}`);
      assert.ok(res.body !== null, "chat completions: empty response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let content = "";
      let contentChunks = 0;
      let done = false;
      for (;;) {
        const { done: streamDone, value } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") {
            done = true;
            continue;
          }
          const chunk = JSON.parse(data) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          const text = chunk.choices?.[0]?.delta?.content;
          if (typeof text === "string" && text !== "") {
            content += text;
            contentChunks += 1;
          }
        }
      }
      assert.ok(done, "stream ended without a [DONE] event");
      return { content, contentChunks };
    };

    try {
      // Boot: wait for the API server to answer at all (no model active yet).
      await waitFor(
        "API server to answer",
        async () => {
          try {
            const res = await fetch(`${apiBase}/v1/models`);
            return res.ok;
          } catch {
            return false;
          }
        },
        120_000,
      );
      // Both workers healthy, with no model directory anywhere.
      await waitFor(
        "2 diskless workers to become healthy",
        async () => {
          try {
            const res = await fetch(`${apiBase}/v1/cluster`);
            if (!res.ok) return false;
            const body = (await res.json()) as {
              workers?: Array<{ state?: string }>;
            };
            const workers = body.workers ?? [];
            return workers.length >= 2 && workers.every((w) => w.state === "healthy");
          } catch {
            return false;
          }
        },
        60_000,
      );
      const logText = output.join("");
      assert.ok(
        logText.includes("no model directory"),
        `workers did not log a diskless boot\n${tail()}`,
      );
      assert.equal(
        logText.split("cannot open model dir").length - 1,
        0,
        `a worker failed to open a model dir during diskless boot\n${tail()}`,
      );
      console.log("diskless boot: 2 workers healthy with no model directory: ok");

      // Restore the demo model so it can be ingested (disk is still
      // the entry point a model first comes in through), then load +
      // activate it over HTTP.
      fs.renameSync(HIDDEN_DIR, MODEL_DIR);
      assert.ok(fs.existsSync(path.join(MODEL_DIR, "model.safetensors")), "demo model was NOT restored");
      const loaded = await postJson(`${apiBase}/v1/models/load`, { ref: "demo/models/demo-10m" });
      assert.equal(loaded.status, 200, `load: HTTP ${loaded.status}\n${tail()}`);
      await waitFor(
        "demo-10m to reach status ready",
        async () => {
          const res = await fetch(`${apiBase}/v1/server/status`);
          if (!res.ok) return false;
          const body = (await res.json()) as {
            models?: Array<{ id?: string; status?: string }>;
          };
          return (body.models ?? []).some((m) => m.id === "demo-10m" && m.status === "ready");
        },
        120_000,
      );
      const activated = await postJson(`${apiBase}/v1/models/demo-10m/activate`, {});
      assert.equal(activated.status, 200, `activate: HTTP ${activated.status}\n${tail()}`);
      await waitFor(
        "server to report ready",
        async () => {
          try {
            return (await fetch(`${apiBase}/readyz`)).ok;
          } catch {
            return false;
          }
        },
        60_000,
      );
      const before = await streamGenerate();
      assert.equal(before.contentChunks, 8, `expected 8 content tokens, got ${before.contentChunks}`);
      console.log(`diskless workers serving: 8 tokens, content=${JSON.stringify(before.content)}`);

      const countOccurrences = (needle: string): number =>
        output.join("").split(needle).length - 1;
      assert.ok(
        countOccurrences("via vRAM weights") >= 2,
        `expected both diskless workers to load via vRAM, log shows ${countOccurrences("via vRAM weights")} "via vRAM weights" lines\n${tail()}`,
      );
      assert.equal(
        countOccurrences("using disk weight loader"),
        0,
        `a disk weight loader was constructed\n${tail()}`,
      );

      // Replacement worker: spawned with NO --model-dir at all.
      const vramLoadsBefore = countOccurrences("via vRAM weights");
      const originalPids = workerPidsForMeshPort(meshPort);
      assert.equal(originalPids.length, 2, `expected 2 workers, found PIDs ${originalPids.join(",")}`);
      const registeredBefore = countOccurrences("registered with");
      replacement = spawn(
        process.execPath,
        [
          WORKER_ENTRY,
          "--orchestrator",
          `http://127.0.0.1:${meshPort}`,
          "--port",
          "0",
        ],
        { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
      );
      replacement.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
      replacement.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
      await waitFor(
        "diskless replacement worker to register",
        () => countOccurrences("registered with") > registeredBefore,
        60_000,
      );
      assert.ok(
        output.join("").includes("no model directory"),
        `replacement worker did not log a diskless boot\n${tail()}`,
      );
      process.kill(originalPids[0] as number, "SIGTERM");
      console.log(`killed original worker ${originalPids[0]}; waiting for re-plan + fresh vRAM stage load`);
      let restarted: { content: string; contentChunks: number } | null = null;
      const genStarted = Date.now();
      let lastGenError = "";
      while (Date.now() - genStarted < 180_000) {
        try {
          const attempt = await streamGenerate();
          if (attempt.contentChunks === 8) {
            restarted = attempt;
            break;
          }
          lastGenError = `got ${attempt.contentChunks} tokens`;
        } catch (err) {
          lastGenError = err instanceof Error ? err.message : String(err);
        }
        await sleep(5000);
      }
      assert.ok(
        restarted !== null,
        `generation did not recover after diskless worker restart (last: ${lastGenError})\n${tail()}`,
      );
      assert.equal(
        restarted.content,
        before.content,
        `content changed after worker restart (before ${JSON.stringify(before.content)}, after ${JSON.stringify(restarted.content)})`,
      );
      console.log(
        `after diskless worker restart: ${restarted.contentChunks} tokens, content=${JSON.stringify(restarted.content)}`,
      );
      assert.ok(
        output.join("").includes("using vRAM weight loader for model demo-10m"),
        `no fresh vRAM loader construction in worker log\n${tail()}`,
      );
      assert.ok(
        countOccurrences("via vRAM weights") > vramLoadsBefore,
        `no new vRAM stage load after restart (before ${vramLoadsBefore}, after ${countOccurrences("via vRAM weights")})\n${tail()}`,
      );
      assert.equal(
        countOccurrences("using disk weight loader"),
        0,
        `a disk weight loader was constructed during the restart\n${tail()}`,
      );

      const shut = await postJson(`${apiBase}/v1/server/shutdown`, {});
      assert.ok(shut.status === 202, `shutdown endpoint: HTTP ${shut.status}\n${tail()}`);
      const exited = await waitForExit(child, 10_000);
      assert.ok(exited, `child did not exit within 10 s of shutdown\n${tail()}`);
      assert.equal(child.exitCode, 0, `child exited with code ${child.exitCode}\n${tail()}`);
    } finally {
      if (replacement !== null && replacement.exitCode === null) {
        replacement.kill("SIGTERM");
        const exited = await waitForExit(replacement, 3000);
        if (!exited) replacement.kill("SIGKILL");
        await waitForExit(replacement, 2000);
      }
      if (child.exitCode === null) {
        child.kill("SIGTERM");
        const exited = await waitForExit(child, 3000);
        if (!exited) child.kill("SIGKILL");
        await waitForExit(child, 2000);
      }
      for (const pid of workerPidsForMeshPort(meshPort)) {
        try {
          process.kill(pid, "SIGTERM");
        } catch {
          // Already gone.
        }
      }
      await sleep(2000);
      for (const pid of workerPidsForMeshPort(meshPort)) {
        try {
          process.kill(pid, "SIGKILL");
        } catch {
          // Already gone.
        }
      }
      // Belt and braces: the demo model must be back whatever happened.
      if (fs.existsSync(HIDDEN_DIR) && !fs.existsSync(MODEL_DIR)) {
        fs.renameSync(HIDDEN_DIR, MODEL_DIR);
      }
      fs.rmSync(PIDFILE, { force: true });
    }
  },
);
