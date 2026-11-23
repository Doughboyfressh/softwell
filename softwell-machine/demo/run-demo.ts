/**
 * demo/run-demo.ts — end-to-end demo of the softwell virtual machine.
 *
 * Part 1 — virtual hardware over real HTTP: boot a VM across 2 donor node
 *   servers (real HTTP on ephemeral ports), show pooled vRAM/vGPU/vCPU/vSSD
 *   capacity, put a model on vSSD, stream it to vRAM, run a real vGPU
 *   matmul kernel and real vCPU tokenize/sample tasks.
 * Part 2 — full cluster across 2 real worker processes: spawn
 *   `softwell run` with the ~10M demo model, show the live VM + shard
 *   placement from /v1/cluster, generate tokens through the OpenAI-compatible
 *   API (streaming), kill one worker, watch the fault detection + re-shard,
 *   and generate again on the survivor.
 *
 * Generated text is incoherent by design (random weights) — the demo proves
 * the pipeline, not the weights.
 *
 * Usage: node demo/run-demo.ts
 */
import { spawn, execSync } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkerInfo } from "../src/contracts.ts";
import { measureMatmulTflops } from "../src/hw/calibrate.ts";
import { VirtualRam, RamNode } from "../src/hw/vram.ts";
import { VirtualGpu, GpuNode } from "../src/hw/vgpu.ts";
import { VirtualCpu, CpuNode } from "../src/hw/vcpu.ts";
import { VirtualSsd, SsdNode } from "../src/hw/vssd.ts";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MODEL_DIR = path.join(ROOT, "demo/models/demo-10m");
const API_BASE = "http://127.0.0.1:8420";
const MESH_BASE = "http://127.0.0.1:8410";

let failures = 0;
function check(name: string, cond: boolean, detail = ""): void {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
  if (!cond) failures++;
}
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
function f32(values: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}
function asF32(bytes: Uint8Array): number[] {
  return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
}
async function waitFor(url: string, timeoutMs: number, label: string): Promise<void> {
  const started = Date.now();
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    if (Date.now() - started > timeoutMs) throw new Error(`timed out waiting for ${label} at ${url}`);
    await sleep(500);
  }
}

/**
 * Demo donors are in-process on one host, so they share one measured
 * calibration (never a hardcoded claim) and the host's real core count.
 */
function makeDonor(id: string, hwPort: number, flopsPerSec: number): WorkerInfo {
  return {
    workerId: id,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 512 * 1024 * 1024,
    freeMemoryBytes: 512 * 1024 * 1024,
    flopsPerSec,
    bandwidthBytesPerSec: 1e9,
    cores: os.cpus().length,
    roles: ["mixed"],
    maxBatchTokens: 1024,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

async function part1(): Promise<void> {
  console.log("\n=== Part 1: virtual hardware across 2 donor node servers (real HTTP) ===");
  const dataDir = path.join(ROOT, "data");
  fs.mkdirSync(dataDir, { recursive: true });
  // One measured calibration for both in-process donors (same host).
  const donorFlops = await measureMatmulTflops().catch(() => 0);
  console.log(`  donor calibration: ${(donorFlops / 1e9).toFixed(2)} GFLOP/s measured (1024^3 matmul)`);
  const donors: Array<{ server: Awaited<ReturnType<typeof createHwNodeServer>>; info: WorkerInfo }> = [];
  for (let i = 0; i < 2; i++) {
    const ramNode = new RamNode({ capacityBytes: 512 * 1024 * 1024, spillDir: path.join(dataDir, `demo-ram-spill-${i}`) });
    const gpuNode = new GpuNode({ device: "cpu" });
    const cpuNode = new CpuNode({});
    const ssdNode = new SsdNode({ dir: path.join(dataDir, `demo-ssd-${i}`) });
    const server = await createHwNodeServer({
      routeTables: [ramNode.routes(), gpuNode.routes(), cpuNode.routes(), ssdNode.routes()],
    });
    donors.push({ server, info: makeDonor(`demo-donor-${i}`, server.port, donorFlops) });
  }
  const nodes = { listNodes: () => donors.map((d) => d.info) };
  const ram = new VirtualRam({ nodes, defaultReplicas: 2 });
  const gpu = new VirtualGpu({ nodes, ram });
  const cpu = new VirtualCpu({ nodes });
  const ssd = new VirtualSsd({ nodes, manifestPath: path.join(dataDir, "demo-vssd-manifest.json"), ram });
  const hypervisor = new Hypervisor({ nodes, ram, gpu, cpu, ssd });
  try {
    const vm = await hypervisor.boot();
    const info = vm.info();
    console.log(
      `  VM booted: vRAM ${(info.vram.totalBytes / 1024 ** 2).toFixed(0)} MiB pooled, ` +
        `vGPU ${(info.gpu.totalFlopsPerSec / 1e9).toFixed(2)} GFLOP/s measured (${info.gpu.devices.length} virtual devices), ` +
        `vCPU ${info.cpu.totalCores} cores, vSSD ${info.ssd.nodes} node(s)`,
    );
    check("2 virtual GPU devices presented", info.gpu.devices.length === 2);
    check("vRAM pooled across 2 donors", info.vram.nodes === 2);

    // vSSD -> vRAM: stage the real demo weights through the tiers.
    const weights = fs.readFileSync(path.join(MODEL_DIR, "model.safetensors"));
    await vm.ssd.putObject("models/demo-10m/model.safetensors", weights);
    const alloc = await hypervisor.stageToVram("models/demo-10m/model.safetensors");
    const head = await vm.ram.read(alloc.vaddr, 16);
    check(
      "weights streamed vSSD -> vRAM (bytes match)",
      Buffer.from(head).equals(Buffer.from(weights.subarray(0, 16))),
      `${alloc.pages.length} pages, ${(alloc.sizeBytes / 1024 / 1024).toFixed(1)} MB`,
    );

    // vGPU: a real matmul kernel through the virtual GPU.
    const a = await vm.ram.alloc(6 * 4);
    const b = await vm.ram.alloc(6 * 4);
    const c = await vm.ram.alloc(4 * 4);
    await vm.ram.write(a.vaddr, f32([1, 2, 3, 4, 5, 6]));
    await vm.ram.write(b.vaddr, f32([7, 8, 9, 10, 11, 12]));
    const stream = vm.gpu.createStream();
    const result = await vm.gpu.launch({
      op: "matmul",
      inputs: [
        { vaddr: a.vaddr, shape: [2, 3], dtype: "float32" },
        { vaddr: b.vaddr, shape: [3, 2], dtype: "float32" },
      ],
      outputs: [{ vaddr: c.vaddr, shape: [2, 2], dtype: "float32" }],
      params: {},
      streamId: stream,
    });
    await vm.gpu.synchronize(stream);
    const cVals = asF32(await vm.ram.read(c.vaddr, 4 * 4));
    check("vGPU matmul [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]",
      cVals.join(",") === "58,64,139,154", `got [${cVals.join(", ")}] on ${result.device} in ${result.elapsedMs} ms`);

    // vCPU: tokenize + sample tasks through the virtual CPU pool.
    const tok = await vm.cpu.submit({ id: "demo-tok", kind: "tokenize", payload: { text: "Hello" } });
    check("vCPU tokenize('Hello')", tok.ok && JSON.stringify(tok.result) === JSON.stringify([72, 101, 108, 108, 111]),
      `node ${tok.nodeId}`);
    const sample = await vm.cpu.submit({
      id: "demo-sample",
      kind: "sample",
      payload: { logits: [0.1, 2.0, 0.5], temperature: 0 },
    });
    check("vCPU greedy sample picks argmax", sample.ok && sample.result === 1, `node ${sample.nodeId}`);
    console.log(`  vCPU stats: ${JSON.stringify(vm.cpu.stats())}, vGPU stats: ${JSON.stringify(vm.gpu.stats().kernelsCompleted)} kernel(s) completed`);
  } finally {
    await hypervisor.shutdown().catch(() => undefined);
    for (const d of donors) await d.server.close().catch(() => undefined);
    for (const dir of ["demo-ram-spill-0", "demo-ram-spill-1", "demo-ssd-0", "demo-ssd-1"]) {
      fs.rmSync(path.join(dataDir, dir), { recursive: true, force: true });
    }
    fs.rmSync(path.join(dataDir, "demo-vssd-manifest.json"), { force: true });
  }
}

async function generateStreaming(prompt: string, maxTokens: number): Promise<{ text: string; ttftMs: number; totalMs: number; tokens: number }> {
  const t0 = Date.now();
  let ttftMs = -1;
  let text = "";
  let tokens = 0;
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: "demo-10m",
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 0,
      stream: true,
    }),
  });
  if (!res.ok || res.body === null) throw new Error(`chat completions failed: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (ttftMs < 0) ttftMs = Date.now() - t0;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      const chunk = JSON.parse(data) as { choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }> };
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) {
        text += content;
        tokens++;
      }
    }
  }
  return { text, ttftMs, totalMs: Date.now() - t0, tokens };
}

async function part2(): Promise<void> {
  console.log("\n=== Part 2: full cluster — VM + serving across 2 worker processes ===");
  // Make sure no stale cluster is holding the ports.
  try {
    const probe = await fetch(`${API_BASE}/v1/models`);
    if (probe.ok) throw new Error("a cluster is already running on :8420 — stop it before running the demo");
  } catch (err) {
    if (err instanceof Error && err.message.includes("already running")) throw err;
  }
  const logPath = "/tmp/softwell-demo-cluster.log";
  const logStream = fs.createWriteStream(logPath);
  const cli: ChildProcess = spawn(
    process.execPath,
    ["src/cli.ts", "run", MODEL_DIR, "--workers", "2", "--port", "8410", "--api-port", "8420"],
    { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
  );
  cli.stdout?.pipe(logStream);
  cli.stderr?.pipe(logStream);
  try {
    await waitFor(`${API_BASE}/v1/models`, 60_000, "API server");
    const cluster = (await (await fetch(`${API_BASE}/v1/cluster`)).json()) as {
      vm?: { vram: { totalBytes: number }; gpu: { devices: unknown[]; totalFlopsPerSec: number }; cpu: { totalCores: number }; ssd: { objects: number } };
      plan?: { assignments: Array<{ stage: number; workerId: string; layerStart: number; layerEnd: number }> };
      workers: Array<{ workerId: string; state: string }>;
    };
    console.log(
      `  live VM: vRAM ${(cluster.vm!.vram.totalBytes / 1024 ** 3).toFixed(1)} GiB pooled, ` +
        `${cluster.vm!.gpu.devices.length} virtual GPU device(s), ${cluster.vm!.cpu.totalCores} vCPUs, ` +
        `${cluster.vm!.ssd.objects} vSSD object(s)`,
    );
    check("live cluster VM booted with 2 virtual GPU devices", cluster.vm!.gpu.devices.length === 2);
    check(
      "2 pipeline stages placed",
      (cluster.plan?.assignments.length ?? 0) === 2,
      (cluster.plan?.assignments ?? []).map((a) => `stage ${a.stage}: layers ${a.layerStart}-${a.layerEnd - 1}`).join(", "),
    );

    const gen1 = await generateStreaming("Hello world", 8);
    console.log(`  generation 1: ${gen1.tokens} tokens, TTFT ${gen1.ttftMs} ms, total ${gen1.totalMs} ms (${(gen1.tokens / (gen1.totalMs / 1000)).toFixed(1)} tok/s)`);
    console.log(`  generated text (random weights — incoherent by design): ${JSON.stringify(gen1.text)}`);
    check("streaming generation produced 8 tokens", gen1.tokens === 8);

    // Fault tolerance: kill one worker, watch the re-shard, generate again.
    // Scope pgrep to THIS demo's own orchestrator (port 8410): a bare
    // `pgrep -f "mesh/worker.ts"` matches every worker on the host, and
    // in this shared tree sibling workstreams run their own clusters
    // concurrently — an unscoped pgrep killed a sibling's worker and
    // left this demo's own cluster untouched (demo FAILED its re-shard
    // check). The demo's workers are spawned with
    // `--orchestrator http://127.0.0.1:8410`, so this pattern matches
    // only them.
    const pids = execSync('pgrep -f "mesh/worker.ts --orchestrator http://127.0.0.1:8410"').toString().trim().split("\n").filter(Boolean);
    console.log(`  killing worker pid ${pids[0]} ...`);
    process.kill(Number(pids[0]), "SIGKILL");
    await sleep(18_000);
    const after = (await (await fetch(`${API_BASE}/v1/cluster`)).json()) as {
      plan?: { assignments: Array<{ stage: number; workerId: string; layerStart: number; layerEnd: number }> };
      workers: Array<{ workerId: string; state: string }>;
    };
    console.log(`  after loss: workers [${after.workers.map((w) => w.state).join(", ")}], ` +
      `stages [${(after.plan?.assignments ?? []).map((a) => `stage ${a.stage}: layers ${a.layerStart}-${a.layerEnd - 1}`).join(", ")}]`);
    check("cluster re-sharded to 1 stage on the survivor", (after.plan?.assignments.length ?? 0) === 1);
    const gen2 = await generateStreaming("Hi", 4);
    check("generation still works after worker loss", gen2.tokens === 4, `${gen2.tokens} tokens in ${gen2.totalMs} ms`);
  } finally {
    cli.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => {
        try {
          cli.kill("SIGKILL");
        } catch {
          // already gone
        }
        resolve();
      }, 5000);
      cli.once("exit", () => {
        clearTimeout(t);
        resolve();
      });
    });
    logStream.end();
  }
}

async function main(): Promise<void> {
  if (!fs.existsSync(path.join(MODEL_DIR, "model.safetensors"))) {
    console.log("demo model not built yet — generating it...");
    execSync("node demo/make-demo-model.ts", { cwd: ROOT, stdio: "inherit" });
  }
  await part1();
  await part2();
  console.log(`\n=== Demo ${failures === 0 ? "PASSED" : `FAILED (${failures} check(s))`} ===`);
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((err: unknown) => {
  console.error(`demo failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
