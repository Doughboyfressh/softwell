/**
 * Disk-removed end-to-end proof for the weights-from-vRAM phase
 * (test/vram-weights-proof.test.ts).
 *
 * Boots the real cluster (`node src/cli.ts up`, 2 workers, ephemeral
 * ports), loads + activates demo-100m over HTTP, then moves the on-disk
 * model directory out of reach (rename, restored in a finally block even
 * on failure) and proves:
 *
 *   (a) generation still streams correctly with the disk dir gone;
 *   (b) no disk weight loader is ever constructed — the captured
 *       worker log shows the stage loaded "via vRAM weights" and never
 *       "using disk weight loader" — and the orchestrator's binary
 *       weight endpoint (GET /mesh/weights) still serves tensor bytes
 *       with the dir gone;
 *   (c) a worker restart also comes from vRAM: a fresh replacement
 *       worker process (spawned with a *different* model dir,
 *       demo-10m, so it can boot) registers, one original worker is
 *       killed, the fault-detection sweep re-plans over the survivors,
 *       and the replacement loads its demo-100m stage from vRAM —
 *       generation still works afterwards.
 *
 * Notes on what is NOT asserted here, and why:
 * - The in-process counters are now exposed on HTTP (2026-11-23,
 *   workstream 3): `GET /v1/server/status` carries
 *   `weights.bytesServed` (orchestrator side) and `GET /mesh/stats`
 *   on a worker carries its `tensorsFetched` / `bytesFetched`
 *   (SafetensorsWeightLoader.bytesReadFromDisk stays in-process
 *   only). They are asserted live in
 *   test/reactivation-disk-removed.test.ts and
 *   test/vram-weights.test.ts; this test still asserts the same fact
 *   behaviorally (successful generation with the directory gone *is*
 *   the zero-disk-read proof), and the worker log names the loader
 *   that was actually chosen.
 * - A re-activation after the move IS now the re-load trigger in
 *   test/reactivation-disk-removed.test.ts: activation takes the
 *   manifest from the staged copy and the tokenizer from staged
 *   vSSD, so it runs with the dir gone. This test keeps the
 *   worker-restart path as its re-load trigger instead.
 * - Workers are spawned with --model-dir demo-10m (the `up` default),
 *   so the spawn-time loader is demo-10m; the demo-100m stage loader
 *   is chosen per-assignment by the worker (vRAM first, disk fallback).
 *
 * The whole test is budgeted at ~420 s: load + stage of a 417 MB model
 * plus a full fresh-worker tensor fetch over loopback are slow on the
 * shared host, and the post-restart recovery window is 240 s. No
 * throughput is asserted — only correctness.
 *
 * Run: node --test test/vram-weights-proof.test.ts
 */

import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const CLI = path.join(ROOT, "src", "cli.ts");
const WORKER_ENTRY = path.join(ROOT, "src", "mesh", "worker.ts");
const MODEL_DIR = path.join(ROOT, "demo", "models", "demo-100m");
const HIDDEN_DIR = path.join(ROOT, "demo", "models", `demo-100m-hidden-${process.pid}`);
const FALLBACK_DIR = path.join(ROOT, "demo", "models", "demo-10m");
const PIDFILE = path.join(ROOT, "data", "softwell.pid");
// Ports are allocated fresh per run (ephemeral): the tree is shared with
// parallel workstreams, so fixed ports collide with orphaned clusters
// from earlier runs. Assigned at the top of the test, before any use.
let MESH_PORT = 0;
let API_PORT = 0;
let API_BASE = "";
let MESH_BASE = "";
const PROMPT = "Hello world";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Allocate a currently-free port from the OS (bind port 0, read it back). */
async function freePort(): Promise<number> {
  return await new Promise<number>((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const addr = probe.address();
      const port =
        typeof addr === "object" && addr !== null ? addr.port : 0;
      probe.close(() => resolve(port));
    });
  });
}

async function assertPortFree(port: number): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const probe = createServer();
    probe.once("error", (err: Error) => {
      reject(new Error(`port ${port} is not free (${err.message})`));
    });
    probe.listen(port, "127.0.0.1", () => {
      probe.close(() => resolve());
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

/** Poll until `check()` returns true, or throw on timeout. */
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

/** One streaming generation; returns the streamed content + chunk counts.
 * Each attempt is capped at 30 s: on a loaded host a generation queued
 * behind a worker re-load can hang until Node's 60 s headers timeout
 * destroys the socket, which would waste most of the recovery window
 * on a single attempt. A fast timeout turns that into a retryable
 * failure instead. */
async function streamGenerate(): Promise<{
  content: string;
  contentChunks: number;
  dataEvents: number;
}> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "demo-100m",
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
  let dataEvents = 0;
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
      dataEvents += 1;
      const text = chunk.choices?.[0]?.delta?.content;
      if (typeof text === "string" && text !== "") {
        content += text;
        contentChunks += 1;
      }
    }
  }
  assert.ok(done, "stream ended without a [DONE] event");
  return { content, contentChunks, dataEvents };
}

/** Worker PIDs whose command line names this test's mesh port. */
function workerPidsForMeshPort(): number[] {
  const pids: number[] = [];
  for (const name of fs.readdirSync("/proc")) {
    if (!/^\d+$/.test(name)) continue;
    try {
      const cmdline = fs.readFileSync(path.join("/proc", name, "cmdline"), "utf-8");
      if (cmdline.includes("mesh/worker.ts") && cmdline.includes(String(MESH_PORT))) {
        pids.push(Number(name));
      }
    } catch {
      // Process exited between readdir and read; ignore.
    }
  }
  return pids.sort((a, b) => a - b);
}

test(
  "weights from vRAM: generation survives disk removal and worker restart",
  { timeout: 420_000 },
  async () => {
    // Ephemeral ports: pick two free ones up front so this test never
    // collides with other workstreams or orphaned clusters.
    MESH_PORT = await freePort();
    API_PORT = await freePort();
    API_BASE = `http://127.0.0.1:${API_PORT}`;
    MESH_BASE = `http://127.0.0.1:${MESH_PORT}`;
    console.log(`proof test ports: mesh=${MESH_PORT} api=${API_PORT}`);
    await assertPortFree(MESH_PORT);
    await assertPortFree(API_PORT);
    assert.ok(
      fs.existsSync(path.join(MODEL_DIR, "model.safetensors")),
      `demo model not built: ${MODEL_DIR} — run \`npm run make-demo-model-100m\``,
    );
    assert.ok(
      fs.existsSync(path.join(FALLBACK_DIR, "model.safetensors")),
      `fallback model dir missing: ${FALLBACK_DIR}`,
    );
    assert.ok(!fs.existsSync(HIDDEN_DIR), `stale hidden dir: ${HIDDEN_DIR}`);

    const output: string[] = [];
    const child: ChildProcess = spawn(
      process.execPath,
      [CLI, "up", "--workers", "2", "--port", String(MESH_PORT), "--api-port", String(API_PORT)],
      { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
    );
    const tail = (): string => output.slice(-40).join("");
    child.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    child.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    let replacement: ChildProcess | null = null;

    try {
      // Boot: wait for the API server to answer at all (no model active yet).
      await waitFor(
        "API server to answer",
        async () => {
          try {
            const res = await fetch(`${API_BASE}/v1/models`);
            return res.ok;
          } catch {
            return false;
          }
        },
        120_000,
      );

      // Load + activate demo-100m over HTTP.
      const loaded = await postJson(`${API_BASE}/v1/models/load`, {
        ref: "demo/models/demo-100m",
      });
      assert.equal(loaded.status, 200, `load: HTTP ${loaded.status}\n${tail()}`);
      await waitFor(
        "demo-100m to reach status ready",
        async () => {
          const res = await fetch(`${API_BASE}/v1/server/status`);
          if (!res.ok) return false;
          const body = (await res.json()) as {
            models?: Array<{ id?: string; status?: string }>;
          };
          return (body.models ?? []).some((m) => m.id === "demo-100m" && m.status === "ready");
        },
        120_000,
      );
      const activated = await postJson(`${API_BASE}/v1/models/demo-100m/activate`, {});
      assert.equal(activated.status, 200, `activate: HTTP ${activated.status}\n${tail()}`);
      await waitFor(
        "server to report ready",
        async () => {
          try {
            return (await fetch(`${API_BASE}/readyz`)).ok;
          } catch {
            return false;
          }
        },
        120_000,
      );

      // Sanity generation with the disk dir still present.
      const before = await streamGenerate();
      assert.equal(before.contentChunks, 8, `expected 8 content tokens, got ${before.contentChunks}`);
      console.log(`sanity (disk present): ${before.contentChunks} tokens, content=${JSON.stringify(before.content)}`);

      const countOccurrences = (needle: string): number =>
        output.join("").split(needle).length - 1;
      assert.ok(
        countOccurrences("via vRAM weights") >= 2,
        `expected both workers to load via vRAM, log shows ${countOccurrences("via vRAM weights")} "via vRAM weights" lines\n${tail()}`,
      );
      assert.equal(
        countOccurrences("using disk weight loader"),
        0,
        `a disk weight loader was constructed before the move\n${tail()}`,
      );

      // The orchestrator's binary weight endpoint serves real bytes.
      const mapRes = await fetch(`${MESH_BASE}/mesh/weights/map?model=demo-100m`);
      assert.equal(mapRes.status, 200, `weight map: HTTP ${mapRes.status}`);
      const mapBody = (await mapRes.json()) as { weightMap?: Record<string, unknown> };
      const tensorNames = Object.keys(mapBody.weightMap ?? {});
      assert.ok(tensorNames.length > 0, "weight map is empty");
      const firstTensor = tensorNames[0] as string;
      const tensorRes = await fetch(
        `${MESH_BASE}/mesh/weights?model=demo-100m&tensor=${encodeURIComponent(firstTensor)}`,
      );
      assert.equal(tensorRes.status, 200, `weight tensor: HTTP ${tensorRes.status}`);
      const tensorBytes = new Uint8Array(await tensorRes.arrayBuffer());
      assert.ok(tensorBytes.byteLength > 0, "weight endpoint served 0 bytes");
      console.log(
        `weight endpoint: ${tensorNames.length} tensors staged, ` +
          `first tensor ${firstTensor} served ${tensorBytes.byteLength} bytes`,
      );

      // --- Move the on-disk model directory out of reach. ---------------
      fs.renameSync(MODEL_DIR, HIDDEN_DIR);
      assert.ok(!fs.existsSync(MODEL_DIR), "model dir still present after rename");
      try {
        // (a) Generation still streams correctly with the disk dir gone.
        const afterMove = await streamGenerate();
        assert.equal(afterMove.contentChunks, 8, `expected 8 content tokens after move, got ${afterMove.contentChunks}`);
        assert.equal(
          afterMove.content,
          before.content,
          `content changed after disk removal (before ${JSON.stringify(before.content)}, after ${JSON.stringify(afterMove.content)})`,
        );
        console.log(
          `after disk removal: ${afterMove.contentChunks} tokens, content=${JSON.stringify(afterMove.content)}`,
        );

        // (b) Still no disk loader; the weight endpoint still serves bytes.
        assert.equal(
          countOccurrences("using disk weight loader"),
          0,
          `a disk weight loader was constructed while the dir was gone\n${tail()}`,
        );
        const tensorRes2 = await fetch(
          `${MESH_BASE}/mesh/weights?model=demo-100m&tensor=${encodeURIComponent(firstTensor)}`,
        );
        assert.equal(tensorRes2.status, 200, `weight tensor after move: HTTP ${tensorRes2.status}`);
        const tensorBytes2 = new Uint8Array(await tensorRes2.arrayBuffer());
        assert.ok(tensorBytes2.byteLength > 0, "weight endpoint served 0 bytes after move");
        assert.equal(
          tensorBytes2.byteLength,
          tensorBytes.byteLength,
          "weight endpoint served different bytes after move",
        );
        console.log(
          `weight endpoint after move: still serving ${tensorBytes2.byteLength} bytes for ${firstTensor}`,
        );

        // (c) Worker restart: fresh process loads its stage from vRAM.
        const vramLoadsBefore = countOccurrences("via vRAM weights");
        const originalPids = workerPidsForMeshPort();
        assert.equal(originalPids.length, 2, `expected 2 workers, found PIDs ${originalPids.join(",")}`);
        // Spawn the replacement FIRST (it boots from the demo-10m dir,
        // which is still on disk), then kill one original worker so the
        // sweep re-plans over the survivors, including the replacement.
        const registeredBefore = countOccurrences("registered with");
        replacement = spawn(
          process.execPath,
          [
            WORKER_ENTRY,
            "--orchestrator",
            `http://127.0.0.1:${MESH_PORT}`,
            "--model-dir",
            FALLBACK_DIR,
            "--port",
            "0",
          ],
          { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
        );
        replacement.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
        replacement.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
        await waitFor(
          "replacement worker to register",
          () => countOccurrences("registered with") > registeredBefore,
          60_000,
        );
        // The replacement's "registered with" line is the newest one;
        // kill one of the ORIGINAL workers.
        process.kill(originalPids[0] as number, "SIGTERM");
        console.log(`killed original worker ${originalPids[0]}; waiting for re-plan + fresh vRAM stage load`);
        // Re-plan happens after the dead timeout (15 s) + sweep; the
        // replacement then fetches every tensor over the mesh. Poll
        // generation (retrying through the re-load window) rather than
        // guessing a fixed sleep.
        let restarted: { content: string; contentChunks: number; dataEvents: number } | null = null;
        const genStarted = Date.now();
        let lastGenError = "";
        while (Date.now() - genStarted < 240_000) {
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
          `generation did not recover after worker restart (last: ${lastGenError})\n${tail()}`,
        );
        assert.equal(
          restarted.content,
          before.content,
          `content changed after worker restart (before ${JSON.stringify(before.content)}, after ${JSON.stringify(restarted.content)})`,
        );
        console.log(
          `after worker restart: ${restarted.contentChunks} tokens, content=${JSON.stringify(restarted.content)}`,
        );
        // The replacement built a fresh VramWeightLoader (not the disk
        // fallback), and the survivor re-loaded its stage via vRAM.
        const logText = output.join("");
        assert.ok(
          logText.includes("using vRAM weight loader for model demo-100m"),
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
      } finally {
        // MUST restore the demo model even on failure.
        if (fs.existsSync(HIDDEN_DIR)) {
          fs.renameSync(HIDDEN_DIR, MODEL_DIR);
        }
        assert.ok(
          fs.existsSync(path.join(MODEL_DIR, "model.safetensors")),
          "demo model was NOT restored after the test",
        );
      }

      // Clean shutdown through the API (202 by design: the server
      // answers, then runs onShutdown — see src/api/server.ts).
      const shut = await postJson(`${API_BASE}/v1/server/shutdown`, {});
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
      // The orchestrator's mesh workers are NOT our children: killing
      // the orchestrator orphans them, and they inherit its stdio, so
      // they hold this test process's pipes open and hang the runner
      // on exit. Sweep any stragglers for this mesh port.
      for (const pid of workerPidsForMeshPort()) {
        try {
          process.kill(pid, "SIGTERM");
        } catch {
          // Already gone.
        }
      }
      await sleep(2000);
      for (const pid of workerPidsForMeshPort()) {
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
