/**
 * End-to-end test for the softwell virtual server (test/server-e2e.test.ts).
 *
 * Spawns the real CLI as a child process — `node src/cli.ts serve
 * demo/models/demo-100m` on fixed high ports (18410 mesh / 18420 API) —
 * waits for readiness on GET /readyz, exercises the model + status +
 * streaming chat routes, measures tokens/sec (logged, never asserted —
 * the host is shared and slow), then shuts the server down through
 * POST /v1/server/shutdown and asserts the child exits promptly.
 *
 * The whole test is budgeted at ~120 s: 60 s for readiness, the rest for
 * one short generation and shutdown. The child is always cleaned up
 * (SIGTERM, then SIGKILL) in a finally block.
 *
 * Run: node --test test/server-e2e.test.ts
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
const PIDFILE = path.join(ROOT, "data", "softwell.pid");
const MESH_PORT = 18410;
const API_PORT = 18420;
const API_BASE = `http://127.0.0.1:${API_PORT}`;
const READY_TIMEOUT_MS = 60_000;
const SHUTDOWN_TIMEOUT_MS = 10_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Fail clearly when a port the test needs is already in use. */
async function assertPortFree(port: number): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const probe = createServer();
    probe.once("error", (err: Error) => {
      reject(
        new Error(
          `port ${port} is not free (${err.message}) — stop whatever is ` +
            `listening there before running the server e2e test`,
        ),
      );
    });
    probe.listen(port, "127.0.0.1", () => {
      probe.close(() => resolve());
    });
  });
}

async function getJson(url: string): Promise<{ status: number; body: unknown }> {
  const res = await fetch(url);
  const body: unknown = await res.json().catch(() => null);
  return { status: res.status, body };
}

/** Poll GET /readyz until it answers 200 (or the deadline passes). */
async function waitForReady(timeoutMs: number): Promise<void> {
  const started = Date.now();
  let lastStatus: string = "no response yet";
  for (;;) {
    try {
      const res = await fetch(`${API_BASE}/readyz`);
      if (res.ok) return;
      lastStatus = `HTTP ${res.status}`;
    } catch (err) {
      lastStatus = err instanceof Error ? err.message : String(err);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(
        `server did not become ready within ${timeoutMs / 1000}s ` +
          `(GET /readyz last: ${lastStatus})`,
      );
    }
    await sleep(500);
  }
}

/** Wait for the child to exit, or false on timeout. */
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

test(
  "virtual server end-to-end: serve demo-100m, stream, measure, shut down",
  { timeout: 120_000 },
  async () => {
    await assertPortFree(MESH_PORT);
    await assertPortFree(API_PORT);
    assert.ok(
      fs.existsSync(path.join(MODEL_DIR, "model.safetensors")),
      `demo model not built: ${MODEL_DIR} — run \`npm run make-demo-model-100m\``,
    );

    const output: string[] = [];
    const child: ChildProcess = spawn(
      process.execPath,
      [
        CLI,
        "serve",
        MODEL_DIR,
        "--port",
        String(MESH_PORT),
        "--api-port",
        String(API_PORT),
        "--workers",
        "1",
      ],
      { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
    );
    const tail = (): string => output.slice(-40).join("");
    child.stdout?.on("data", (d: Buffer) => output.push(d.toString("utf8")));
    child.stderr?.on("data", (d: Buffer) => output.push(d.toString("utf8")));

    try {
      await waitForReady(READY_TIMEOUT_MS);

      // GET /v1/models lists demo-100m.
      const models = await getJson(`${API_BASE}/v1/models`);
      assert.equal(models.status, 200, `GET /v1/models: HTTP ${models.status}\n${tail()}`);
      const listed = (models.body as { data?: Array<{ id?: string }> }).data ?? [];
      assert.ok(
        listed.some((m) => m.id === "demo-100m"),
        `demo-100m not in /v1/models: ${JSON.stringify(listed)}\n${tail()}`,
      );

      // GET /v1/server/status shows it active.
      const status = await getJson(`${API_BASE}/v1/server/status`);
      assert.equal(status.status, 200, `GET /v1/server/status: HTTP ${status.status}\n${tail()}`);
      const statusBody = status.body as { activeModelId?: string | null; state?: string };
      assert.equal(statusBody.activeModelId, "demo-100m", `active model: ${JSON.stringify(statusBody)}`);
      assert.equal(statusBody.state, "ready", `server state: ${JSON.stringify(statusBody)}`);

      // POST /v1/chat/completions, streaming: read the SSE incrementally
      // and prove the first chunk arrives before the response ends.
      const streamStarted = Date.now();
      const res = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "demo-100m",
          messages: [{ role: "user", content: "Hello world" }],
          max_tokens: 8,
          temperature: 0,
          stream: true,
        }),
      });
      assert.equal(res.status, 200, `chat completions: HTTP ${res.status}`);
      assert.ok(res.body !== null, "chat completions: empty response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let firstChunkAt: number | null = null;
      let lastTokenAt: number | null = null;
      let doneAt: number | null = null;
      let content = "";
      let contentChunks = 0;
      let dataEvents = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          const now = Date.now();
          if (data === "[DONE]") {
            doneAt = now;
            continue;
          }
          const chunk = JSON.parse(data) as {
            choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }>;
            usage?: unknown;
          };
          dataEvents += 1;
          if (firstChunkAt === null) firstChunkAt = now;
          // Streaming has no usage object — assert it stays absent.
          assert.equal(
            chunk.usage,
            undefined,
            `streaming chunk carried a usage object: ${data}`,
          );
          const text = chunk.choices?.[0]?.delta?.content;
          if (typeof text === "string" && text !== "") {
            content += text;
            contentChunks += 1;
            lastTokenAt = now;
          }
        }
      }
      const streamEndedAt = Date.now();
      assert.ok(firstChunkAt !== null, "no data chunk arrived before the stream ended");
      assert.ok(doneAt !== null, "stream ended without a [DONE] event");
      assert.ok(
        (firstChunkAt as number) < (doneAt as number),
        `first chunk did not arrive before the end (first: ${firstChunkAt}, [DONE]: ${doneAt})`,
      );
      assert.equal(typeof content, "string");
      // Content chunks plus the finish chunk.
      assert.ok(dataEvents >= 2, `expected >= 2 streamed chunks, got ${dataEvents}`);
      assert.ok(contentChunks >= 1, `expected >= 1 content chunk, got ${contentChunks}`);

      // Tokens/sec, measured only: completion tokens over wall time from
      // first token to last. Never asserted against a threshold — the
      // host is shared and slow.
      const wallSec = Math.max(((lastTokenAt as number) - (firstChunkAt as number)) / 1000, 0.001);
      const tokensPerSec = contentChunks / wallSec;
      console.log(
        `measured: ${contentChunks} completion tokens in ${(wallSec * 1000).toFixed(0)} ms ` +
          `from first to last token -> ${tokensPerSec.toFixed(2)} tok/s ` +
          `(demo-100m, 1 worker, streaming, max_tokens 8, ` +
          `total stream time ${streamEndedAt - streamStarted} ms)`,
      );

      // POST /v1/server/shutdown: the child must exit within 10 s.
      const shut = await fetch(`${API_BASE}/v1/server/shutdown`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
      assert.ok(shut.ok, `shutdown endpoint: HTTP ${shut.status}\n${tail()}`);
      const exited = await waitForExit(child, SHUTDOWN_TIMEOUT_MS);
      assert.ok(exited, `child did not exit within ${SHUTDOWN_TIMEOUT_MS / 1000}s of shutdown\n${tail()}`);
      assert.equal(child.exitCode, 0, `child exited with code ${child.exitCode}\n${tail()}`);
    } finally {
      if (child.exitCode === null) {
        child.kill("SIGTERM");
        const exited = await waitForExit(child, 3000);
        if (!exited) child.kill("SIGKILL");
        await waitForExit(child, 2000);
      }
      fs.rmSync(PIDFILE, { force: true });
    }
  },
);
