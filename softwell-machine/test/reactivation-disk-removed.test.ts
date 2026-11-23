/**
 * Re-activation proof for the activation-side work (docs/DONOR.md):
 * `ModelRegistry.getTokenizer` reads the staged vSSD copy and
 * `buildFactory` (src/cli.ts) takes the manifest from the registry's
 * staged copy, so activation must never re-read the model directory
 * from disk.
 *
 * Boots the real cluster (`node src/cli.ts up`, 1 worker, ephemeral
 * ports), loads + activates demo-100m over HTTP, then moves the on-disk
 * model directory out of reach (rename, restored in a finally block
 * even on failure) and proves:
 *
 *   (a) re-activating the same model with the directory gone succeeds
 *       (POST /v1/models/demo-100m/activate -> 200, /readyz ok);
 *   (b) generation after the re-activation streams the same 8 tokens
 *       as before the move — the tokenizer (encode of the prompt,
 *       decode of the tokens) and the manifest both came from the
 *       staged copies, since the directory no longer exists to read;
 *   (c) no disk weight loader is ever constructed while the directory
 *       is gone (worker log shows no "using disk weight loader");
 *   (d) the orchestrator weight counter is visible over HTTP
 *       (GET /v1/server/status weights.bytesServed > 0 after the
 *       stage load — the (d) workstream item, asserted live here).
 *
 * The previous weights proof (test/vram-weights-proof.test.ts)
 * explicitly did NOT use re-activation as its re-load trigger, because
 * activation then re-read the manifest + tokenizer from the model
 * directory. That is the behaviour this test pins as fixed.
 *
 * Run: node --test test/reactivation-disk-removed.test.ts
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
const MODEL_DIR = path.join(ROOT, "demo", "models", "demo-100m");
const HIDDEN_DIR = path.join(ROOT, "demo", "models", `demo-100m-hidden-${process.pid}`);
const PIDFILE = path.join(ROOT, "data", "softwell.pid");
// Ports are allocated fresh per run (ephemeral): the tree is shared with
// parallel workstreams, so fixed ports collide with orphaned clusters
// from earlier runs. Assigned at the top of the test, before any use.
let MESH_PORT = 0;
let API_PORT = 0;
let API_BASE = "";
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

/** One streaming generation; returns the streamed content + chunk counts. */
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
    signal: AbortSignal.timeout(60_000),
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
  "re-activation works with the model directory moved away",
  { timeout: 420_000 },
  async () => {
    MESH_PORT = await freePort();
    API_PORT = await freePort();
    API_BASE = `http://127.0.0.1:${API_PORT}`;
    console.log(`reactivation proof ports: mesh=${MESH_PORT} api=${API_PORT}`);
    assert.ok(
      fs.existsSync(path.join(MODEL_DIR, "model.safetensors")),
      `demo model not built: ${MODEL_DIR} — run \`npm run make-demo-model-100m\``,
    );
    assert.ok(!fs.existsSync(HIDDEN_DIR), `stale hidden dir: ${HIDDEN_DIR}`);

    const output: string[] = [];
    const child: ChildProcess = spawn(
      process.execPath,
      [CLI, "up", "--workers", "1", "--port", String(MESH_PORT), "--api-port", String(API_PORT)],
      { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
    );
    const tail = (): string => output.slice(-40).join("");
    child.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    child.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));

    try {
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

      // (d) live: the orchestrator weight counter over HTTP.
      const statusRes = await fetch(`${API_BASE}/v1/server/status`);
      assert.equal(statusRes.status, 200, `server status: HTTP ${statusRes.status}`);
      const statusBody = (await statusRes.json()) as {
        weights?: { bytesServed?: number };
      };
      assert.ok(
        typeof statusBody.weights?.bytesServed === "number" &&
          statusBody.weights.bytesServed > 0,
        `expected weights.bytesServed > 0 after the stage load, got ${JSON.stringify(statusBody.weights)}`,
      );
      console.log(`weights.bytesServed over HTTP: ${statusBody.weights?.bytesServed}`);

      const countOccurrences = (needle: string): number =>
        output.join("").split(needle).length - 1;
      assert.equal(
        countOccurrences("using disk weight loader"),
        0,
        `a disk weight loader was constructed before the move\n${tail()}`,
      );

      // --- Move the on-disk model directory out of reach. ---------------
      fs.renameSync(MODEL_DIR, HIDDEN_DIR);
      assert.ok(!fs.existsSync(MODEL_DIR), "model dir still present after rename");
      try {
        // (a) Re-activate with the directory gone. This is the step
        // the old proof test could not run: activation used to re-read
        // the manifest + tokenizer from the directory.
        const reactivated = await postJson(`${API_BASE}/v1/models/demo-100m/activate`, {});
        assert.equal(
          reactivated.status,
          200,
          `re-activate with dir gone: HTTP ${reactivated.status} ${JSON.stringify(reactivated.body)}\n${tail()}`,
        );
        await waitFor(
          "server to report ready after re-activation",
          async () => {
            try {
              return (await fetch(`${API_BASE}/readyz`)).ok;
            } catch {
              return false;
            }
          },
          120_000,
        );
        console.log("re-activation with the model directory gone: ok");

        // (b) Generation after re-activation: same tokens as before.
        const after = await streamGenerate();
        assert.equal(after.contentChunks, 8, `expected 8 content tokens after re-activation, got ${after.contentChunks}`);
        assert.equal(
          after.content,
          before.content,
          `content changed after re-activation with dir gone (before ${JSON.stringify(before.content)}, after ${JSON.stringify(after.content)})`,
        );
        console.log(
          `after re-activation (dir gone): ${after.contentChunks} tokens, content=${JSON.stringify(after.content)}`,
        );

        // (c) Still no disk loader while the directory was gone.
        assert.equal(
          countOccurrences("using disk weight loader"),
          0,
          `a disk weight loader was constructed while the dir was gone\n${tail()}`,
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
