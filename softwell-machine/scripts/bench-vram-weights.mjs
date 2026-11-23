/**
 * Baseline / after benchmark for the weights-from-vRAM phase
 * (scripts/bench-vram-weights.mjs).
 *
 * Boots the real CLI (`node src/cli.ts serve demo/models/demo-100m`,
 * 1 worker) on given ports, streams a fixed prompt 3x, and records
 * TTFT, decode tok/s, total time and tokens per run, plus host
 * conditions. Writes a JSON file (default
 * bench-vram-weights-<label>.json in the repo root).
 *
 * Usage:
 *   node scripts/bench-vram-weights.mjs --label before \
 *     --mesh-port 18510 --api-port 18520 [--reps 3] [--out <file>]
 *
 * Prompt, max_tokens and temperature are fixed so before/after runs
 * are directly comparable. Nothing here is asserted against a
 * threshold — the host is shared and slow; numbers are reported.
 */
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CLI = path.join(ROOT, "src", "cli.ts");
const MODEL_DIR = path.join(ROOT, "demo", "models", "demo-100m");
const PIDFILE = path.join(ROOT, "data", "softwell.pid");

const PROMPT = "Hello world";
const MAX_TOKENS = 8;
const TEMPERATURE = 0;

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const label = arg("--label", "before");
const meshPort = Number(arg("--mesh-port", "18510"));
const apiPort = Number(arg("--api-port", "18520"));
const reps = Number(arg("--reps", "3"));
const outFile = arg("--out", path.join(ROOT, `bench-vram-weights-${label}.json`));
const API_BASE = `http://127.0.0.1:${apiPort}`;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForReady(timeoutMs = 90_000) {
  const started = Date.now();
  let last = "no response yet";
  for (;;) {
    try {
      const res = await fetch(`${API_BASE}/readyz`);
      if (res.ok) return;
      last = `HTTP ${res.status}`;
    } catch (err) {
      last = err instanceof Error ? err.message : String(err);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`server not ready within ${timeoutMs / 1000}s (last: ${last})`);
    }
    await sleep(500);
  }
}

function waitForExit(child, timeoutMs) {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve(true);
    });
  });
}

/** One streaming generation; returns TTFT, decode tok/s, total, tokens. */
async function oneRun() {
  const started = Date.now();
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "demo-100m",
      messages: [{ role: "user", content: PROMPT }],
      max_tokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      stream: true,
    }),
  });
  if (res.status !== 200 || res.body === null) {
    throw new Error(`chat completions: HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let firstContentAt = null;
  let lastContentAt = null;
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
      const now = Date.now();
      if (data === "[DONE]") {
        done = true;
        continue;
      }
      const chunk = JSON.parse(data);
      dataEvents += 1;
      const text = chunk.choices?.[0]?.delta?.content;
      if (typeof text === "string" && text !== "") {
        if (firstContentAt === null) firstContentAt = now;
        content += text;
        contentChunks += 1;
        lastContentAt = now;
      }
    }
  }
  const ended = Date.now();
  if (firstContentAt === null || !done) {
    throw new Error(`stream incomplete (contentChunks=${contentChunks}, done=${done})`);
  }
  const ttftMs = firstContentAt - started;
  const decodeWallSec = Math.max((lastContentAt - firstContentAt) / 1000, 0.001);
  const decodeTokPerSec = contentChunks / decodeWallSec;
  return {
    ttft_ms: ttftMs,
    decode_tok_per_s: Number(decodeTokPerSec.toFixed(2)),
    total_ms: ended - started,
    content_tokens: contentChunks,
    data_events: dataEvents,
    content,
  };
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function hostConditions() {
  let nodeVersion = process.version;
  let cpuModel = "unknown";
  try {
    cpuModel = os.cpus()[0]?.model ?? "unknown";
  } catch {}
  let mem = null;
  try {
    mem = execSync("free -m", { encoding: "utf-8" }).trim();
  } catch {}
  return {
    node: nodeVersion,
    platform: os.platform(),
    arch: os.arch(),
    cpus: os.cpus().length,
    cpu_model: cpuModel,
    total_mem_mib: Math.round(os.totalmem() / 1024 / 1024),
    free_mem_mib: Math.round(os.freemem() / 1024 / 1024),
    free_m_output: mem,
    loadavg: os.loadavg(),
  };
}

async function main() {
  if (!fs.existsSync(path.join(MODEL_DIR, "model.safetensors"))) {
    throw new Error(`demo model not built: ${MODEL_DIR}`);
  }
  const output = [];
  const child = spawn(
    process.execPath,
    [CLI, "serve", MODEL_DIR, "--port", String(meshPort), "--api-port", String(apiPort), "--workers", "1"],
    { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout?.on("data", (d) => output.push(d.toString("utf8")));
  child.stderr?.on("data", (d) => output.push(d.toString("utf8")));
  const tail = () => output.slice(-30).join("");
  const result = {
    label,
    date: new Date().toISOString(),
    model: "demo-100m",
    params: 104_365_056,
    workers: 1,
    mesh_port: meshPort,
    api_port: apiPort,
    prompt: PROMPT,
    max_tokens: MAX_TOKENS,
    temperature: TEMPERATURE,
    host: hostConditions(),
    runs: [],
    summary: null,
  };
  try {
    await waitForReady();
    const statusRes = await fetch(`${API_BASE}/v1/server/status`);
    const statusBody = await statusRes.json().catch(() => null);
    result.server_status = statusBody;
    for (let i = 0; i < reps; i++) {
      const run = await oneRun();
      result.runs.push({ rep: i + 1, ...run });
      console.log(
        `rep ${i + 1}: TTFT ${run.ttft_ms} ms, decode ${run.decode_tok_per_s} tok/s, ` +
          `total ${run.total_ms} ms, tokens ${run.content_tokens}`,
      );
    }
    result.summary = {
      median_ttft_ms: median(result.runs.map((r) => r.ttft_ms)),
      ttft_ms_range: [Math.min(...result.runs.map((r) => r.ttft_ms)), Math.max(...result.runs.map((r) => r.ttft_ms))],
      median_decode_tok_per_s: median(result.runs.map((r) => r.decode_tok_per_s)),
      decode_tok_per_s_range: [
        Math.min(...result.runs.map((r) => r.decode_tok_per_s)),
        Math.max(...result.runs.map((r) => r.decode_tok_per_s)),
      ],
      median_total_ms: median(result.runs.map((r) => r.total_ms)),
      total_ms_range: [Math.min(...result.runs.map((r) => r.total_ms)), Math.max(...result.runs.map((r) => r.total_ms))],
    };
    console.log("summary:", JSON.stringify(result.summary));
    fs.writeFileSync(outFile, JSON.stringify(result, null, 2) + "\n");
    console.log(`wrote ${outFile}`);
    const shut = await fetch(`${API_BASE}/v1/server/shutdown`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    if (!shut.ok) throw new Error(`shutdown endpoint: HTTP ${shut.status}\n${tail()}`);
    const exited = await waitForExit(child, 10_000);
    if (!exited) throw new Error(`child did not exit after shutdown\n${tail()}`);
  } finally {
    if (child.exitCode === null) {
      child.kill("SIGTERM");
      const exited = await waitForExit(child, 3000);
      if (!exited) child.kill("SIGKILL");
      await waitForExit(child, 2000);
    }
    fs.rmSync(PIDFILE, { force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
