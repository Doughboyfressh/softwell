/**
 * BEFORE baseline benchmark for the KV-cache-in-vRAM phase
 * (scripts/bench-kv.mjs).
 *
 * Same protocol as scripts/bench-vram-weights.mjs — boots the real CLI
 * (`node src/cli.ts serve demo/models/demo-100m`, 1 worker), streams a
 * fixed prompt 3x, records TTFT, decode tok/s, total time and tokens
 * per run, plus host conditions — but at TWO prompt lengths, because
 * the KV cache only matters once the prompt is long:
 *   - short: "Hello world" (the weights-bench prompt)
 *   - long:  a deterministic ~1k-token prompt (see LONG_* below)
 *
 * Both conditions run against the same server boot, short first.
 * The serving path under test uses the host BlockPagedKVCache
 * (src/mesh/worker.ts builds one per stage load, block size 16);
 * the vRAM cache (VramPagedKVCache) is NOT wired in yet. This file
 * is the BEFORE baseline: no "after" number exists yet.
 *
 * Usage:
 *   node scripts/bench-kv.mjs --label before \
 *     --mesh-port 18512 --api-port 18522 [--reps 3] [--out <file>] [--kv-store host|vram]
 *
 * --kv-store selects the workers' KV cache: host (default, the BEFORE
 * baseline) or vram (the AFTER run — KV pages in pooled vRAM). It is
 * passed through to `softwell serve` as --kv-store.
 *
 * Writes JSON (default bench-kv-<label>.json in the repo root).
 * Nothing here is asserted against a threshold — the host is shared
 * and slow; numbers are reported with medians and ranges.
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

const MAX_TOKENS = 8;
const TEMPERATURE = 0;

// --- Prompt conditions -------------------------------------------------
// Chat messages are concatenated as "user: <content>" and tokenized
// byte-level with a BOS prepended (src/api/openai.ts, byte-level
// tokenizer, add_bos_token=true), so prompt tokens = UTF-8 bytes of
// "user: " + content, plus 1 for BOS.
const SHORT_CONTENT = "Hello world";
const LONG_SENTENCE = "The quick brown fox jumps over the lazy dog. ";
const LONG_REPEATS = 22;
const LONG_CONTENT = LONG_SENTENCE.repeat(LONG_REPEATS); // 990 chars

function promptTokens(content) {
  return Buffer.byteLength(`user: ${content}`, "utf8") + 1; // +1 BOS
}

const CONDITIONS = [
  {
    name: "short",
    content: SHORT_CONTENT,
    prompt_construction: {
      content: SHORT_CONTENT,
      expected_prompt_tokens: promptTokens(SHORT_CONTENT),
    },
  },
  {
    name: "long",
    content: LONG_CONTENT,
    prompt_construction: {
      sentence: LONG_SENTENCE,
      repeats: LONG_REPEATS,
      content_chars: LONG_CONTENT.length,
      content_bytes: Buffer.byteLength(LONG_CONTENT, "utf8"),
      expected_prompt_tokens: promptTokens(LONG_CONTENT),
    },
  },
];

// --- CLI args ----------------------------------------------------------
function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const label = arg("--label", "before");
const meshPort = Number(arg("--mesh-port", "18512"));
const apiPort = Number(arg("--api-port", "18522"));
const reps = Number(arg("--reps", "3"));
const kvStore = arg("--kv-store", "host");
if (kvStore !== "host" && kvStore !== "vram") {
  throw new Error(`--kv-store must be "host" or "vram", got ${JSON.stringify(kvStore)}`);
}
const outFile = arg("--out", path.join(ROOT, `bench-kv-${label}.json`));
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
async function oneRun(content) {
  const started = Date.now();
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "demo-100m",
      messages: [{ role: "user", content }],
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
  let contentOut = "";
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
        contentOut += text;
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
    content: contentOut,
  };
}

/** Non-streaming probe to read the server-reported prompt token count. */
async function probePromptTokens(content) {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "demo-100m",
      messages: [{ role: "user", content }],
      max_tokens: 1,
      temperature: TEMPERATURE,
      stream: false,
    }),
  });
  if (res.status !== 200) throw new Error(`prompt-token probe: HTTP ${res.status}`);
  const body = await res.json();
  return body?.usage?.prompt_tokens ?? null;
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function summarize(runs) {
  const ok = runs.filter((r) => r.error === undefined);
  if (ok.length === 0) {
    return {
      median_ttft_ms: null,
      ttft_ms_range: null,
      median_decode_tok_per_s: null,
      decode_tok_per_s_range: null,
      median_total_ms: null,
      total_ms_range: null,
      successful_runs: 0,
      failed_runs: runs.length,
    };
  }
  return {
    median_ttft_ms: median(ok.map((r) => r.ttft_ms)),
    ttft_ms_range: [Math.min(...ok.map((r) => r.ttft_ms)), Math.max(...ok.map((r) => r.ttft_ms))],
    median_decode_tok_per_s: median(ok.map((r) => r.decode_tok_per_s)),
    decode_tok_per_s_range: [
      Math.min(...ok.map((r) => r.decode_tok_per_s)),
      Math.max(...ok.map((r) => r.decode_tok_per_s)),
    ],
    median_total_ms: median(ok.map((r) => r.total_ms)),
    total_ms_range: [Math.min(...ok.map((r) => r.total_ms)), Math.max(...ok.map((r) => r.total_ms))],
    successful_runs: ok.length,
    failed_runs: runs.length - ok.length,
  };
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
    [CLI, "serve", MODEL_DIR, "--port", String(meshPort), "--api-port", String(apiPort), "--workers", "1", "--kv-store", kvStore],
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
    kv_cache:
      kvStore === "vram"
        ? "vram VramPagedKVCache (block size 16, 4096 blocks, replicas 2, chunk 1024 tokens) over RemoteVramAccess"
        : "host BlockPagedKVCache (block size 16) — vRAM cache NOT wired in",
    // NOTE (2026-11-23): the ~1k-token "long" condition used to fail
    // 3/3 (see bench-kv-before.json): one prefill was a single mesh
    // execute, and a 997-token prefill takes ~224 s on this host while
    // the transport timeout is 30 s. Chunked prefill (16-row chunks,
    // src/scheduler/batch.ts) fixed it — each chunk is one execute
    // inside the timeout, so the long runs below are numbers now, not
    // failures. The in-process prefill time at 997 tokens was measured
    // separately with src/bench/novel-bench.ts Part B (baseline,
    // BlockPagedKVCache, no mesh): see bench-kv-before-inprocess.json.
    max_tokens: MAX_TOKENS,
    temperature: TEMPERATURE,
    host: hostConditions(),
    conditions: {},
  };
  try {
    await waitForReady();
    const statusRes = await fetch(`${API_BASE}/v1/server/status`);
    const statusBody = await statusRes.json().catch(() => null);
    result.server_status = statusBody;
    for (const cond of CONDITIONS) {
      const runs = [];
      for (let i = 0; i < reps; i++) {
        try {
          const run = await oneRun(cond.content);
          runs.push({ rep: i + 1, ...run });
          console.log(
            `${cond.name} rep ${i + 1}: TTFT ${run.ttft_ms} ms, decode ${run.decode_tok_per_s} tok/s, ` +
              `total ${run.total_ms} ms, tokens ${run.content_tokens}`,
          );
        } catch (err) {
          // A failed rep is a result, not a reason to abort the bench:
          // record it and keep going.
          const message = err instanceof Error ? err.message : String(err);
          runs.push({ rep: i + 1, error: message });
          console.log(`${cond.name} rep ${i + 1}: FAILED (${message})`);
        }
      }
      let measured = null;
      let probeError = null;
      try {
        measured = await probePromptTokens(cond.content);
      } catch (err) {
        probeError = err instanceof Error ? err.message : String(err);
      }
      console.log(
        `${cond.name}: server-reported prompt_tokens=${measured} (expected ${cond.prompt_construction.expected_prompt_tokens})` +
          (probeError ? ` probe error: ${probeError}` : ""),
      );
      result.conditions[cond.name] = {
        prompt: cond.name === "short" ? SHORT_CONTENT : undefined,
        prompt_construction: cond.prompt_construction,
        prompt_tokens_measured: measured,
        prompt_tokens_probe_error: probeError,
        host_before_condition: hostConditions(),
        runs,
        summary: summarize(runs),
      };
    }
    console.log("summary:", JSON.stringify(result.conditions, null, 2));
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
