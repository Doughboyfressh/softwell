/**
 * scripts/bench-vnpu.ts — vNPU measurements: perplexity, tok/s, kernel
 * throughput of SmolLM-135M through the vNPU path vs the FP32 baseline.
 *
 * Three sections, each writing its own JSON at the repo root (appended
 * as an array, same convention as sw-smollm-bench.json):
 *
 *   node scripts/bench-vnpu.ts [--modes off,int8,fp8] [--skip-ppl] [--skip-tok] [--skip-kernel]
 *
 * (a) Perplexity — the SW-code protocol, so the numbers are comparable
 *     (scripts/bench-sw-smollm.ts): same TinyStories slice (byte offset
 *     1,000,000,000), 4 chunks x seqLen 64 = 252 scored tokens,
 *     teacher-forced, BOS prepended per chunk. INT8/FP8 vs FP32.
 * (b) tok/s — short prompt "The capital of France is", 16 tokens,
 *     temp 0 greedy, 1 rep per mode (minimal reps: the vNPU decode
 *     forward is ~10-20 s/token on this fallback — see the note in
 *     benchTok — so 2 reps would double the section for no new
 *     information); prefill ms + full-generate tok/s + decode-only
 *     tok/s (8 single-row forwards, median), with the generated
 *     tokens recorded.
 * (c) Kernel throughput — INT8/FP8 matmulTransposeB vs FP32
 *     matmulTransposeB on the forward-opt shapes (q/o, gate, down,
 *     lm_head, prefill gate) plus square 256^3 / 512^3 for TOPS, with
 *     staged weights (the forward-pass reality) and a max-abs-error
 *     check against the FP32 reference.
 *
 * Single-run numbers with the load noted (the user is impatient):
 * minimal reps everywhere, 1-2 timed reps per kernel shape.
 * All modes load the model fresh; the vNPU modes stage weights into
 * SRAM at load (the executor does this itself).
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { TransformerStageExecutor } from "../src/engine/transformer.ts";
import { AutoregressiveDecoder, forwardTokenRows } from "../src/engine/novel-decoding.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import { sampleLogits } from "../src/engine/sampling.ts";
import { totalNll, pplFromMeanNll } from "../src/bench/perplexity.ts";
import { matmulTransposeB } from "../src/engine/attention.ts";
import { VirtualNpu } from "../src/hw/vnpu.ts";
import { median } from "../src/bench/types.ts";
import type {
  IVirtualRam,
  NodeProvider,
  NpuMode,
  ShardAssignment,
  VramStats,
  WorkerInfo,
} from "../src/contracts.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MODEL_DIR = path.join(ROOT, "models/smollm-135m");
const STORIES_PATH = path.join(ROOT, "training/data/TinyStories-train.txt");
const SEP = "<|endoftext|>";
const PPL_OUT = path.join(ROOT, "vnpu-perplexity-bench.json");
const TOK_OUT = path.join(ROOT, "vnpu-tok-bench.json");
const KERNEL_OUT = path.join(ROOT, "vnpu-kernel-bench.json");

const PROMPT = "The capital of France is";
const GEN_TOKENS = 16;

function usage(): never {
  console.log(
    "usage: node scripts/bench-vnpu.ts [--modes off,int8,fp8] [--skip-ppl] [--skip-tok] [--skip-kernel]\n" +
      "  [--ppl-out PATH] [--tok-out PATH] [--kernel-out PATH]",
  );
  process.exit(2);
}

function hostInfo() {
  return {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus().length,
    cpuModel: os.cpus()[0]?.model ?? null,
    totalMemMiB: Math.round(os.totalmem() / 1048576),
    loadavg: os.loadavg().map((x) => +x.toFixed(2)),
  };
}

function appendJson(outPath: string, result: unknown): void {
  let prior: unknown[] = [];
  if (fs.existsSync(outPath)) {
    const parsed: unknown = JSON.parse(fs.readFileSync(outPath, "utf8"));
    if (Array.isArray(parsed)) prior = parsed;
  }
  prior.push(result);
  fs.writeFileSync(outPath, JSON.stringify(prior, null, 2) + "\n");
}

/** Read a raw byte window of the (2.2 GiB) TinyStories file without loading it. */
function readTextSlice(byteOffset: number, byteLen: number): string {
  const fd = fs.openSync(STORIES_PATH, "r");
  try {
    const buf = Buffer.alloc(byteLen);
    const n = fs.readSync(fd, buf, 0, byteLen, byteOffset);
    return buf.subarray(0, n).toString("utf8");
  } finally {
    fs.closeSync(fd);
  }
}

/** Token stream: each complete story tokenized, joined by the EOS id. */
function buildTokenStream(byteOffset: number, byteLen: number): { tokens: number[]; stories: number } {
  const raw = readTextSlice(byteOffset, byteLen);
  const firstSep = raw.indexOf(SEP);
  if (firstSep < 0) throw new Error("no story separator found in text slice");
  const body = raw.slice(firstSep + SEP.length);
  const parts = body.split(SEP);
  parts.pop(); // last part may be a truncated story
  const tokenizer = loadTokenizer(MODEL_DIR);
  const eos = tokenizer.eosTokenId;
  const tokens: number[] = [];
  let stories = 0;
  for (const story of parts) {
    const trimmed = story.trim();
    if (trimmed.length < 20) continue;
    tokens.push(...tokenizer.encode(trimmed), eos);
    stories++;
  }
  return { tokens, stories };
}

function makeAssignment(cfg: { numHiddenLayers: number }): ShardAssignment {
  return {
    workerId: "vnpu-bench",
    stage: 0,
    layerStart: 0,
    layerEnd: cfg.numHiddenLayers,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 0,
  };
}

async function loadExecutor(mode: NpuMode): Promise<{ executor: TransformerStageExecutor; loadSec: number }> {
  const manifest = loadManifest(MODEL_DIR, "smollm-135m");
  const loader = openWeightLoader(MODEL_DIR, "smollm-135m");
  const executor = createReferenceBackend().createStageExecutor(
    manifest,
    makeAssignment(manifest.config),
    { npu: mode },
  ) as TransformerStageExecutor;
  const t0 = performance.now();
  await executor.load(loader);
  return { executor, loadSec: (performance.now() - t0) / 1000 };
}

// ---------------------------------------------------------------------------
// (a) Perplexity — SW-code protocol: 4 chunks x seqLen 64 = 252 tokens.
// ---------------------------------------------------------------------------

async function benchPerplexity(mode: NpuMode): Promise<void> {
  const chunks = 4;
  const seqLen = 64;
  const byteOffset = 1_000_000_000;
  const { tokens: stream, stories } = buildTokenStream(byteOffset, 256 * 1024);
  const need = chunks * (seqLen - 1);
  if (stream.length < need) throw new Error(`stream has ${stream.length} tokens, need ${need}`);
  const tokenizer = loadTokenizer(MODEL_DIR);
  const bos = tokenizer.bosTokenId;
  const chunkIds: number[][] = [];
  for (let c = 0; c < chunks; c++) {
    chunkIds.push([bos, ...stream.slice(c * (seqLen - 1), (c + 1) * (seqLen - 1))]);
  }
  const { executor, loadSec } = await loadExecutor(mode);
  const cfg = executor.manifest.config;
  const newCache = () => new BlockPagedKVCache(16, 4096, cfg.numKeyValueHeads, cfg.headDim);

  let totalNllSum = 0;
  let tokensScored = 0;
  let forwardSec = 0;
  const chunkPpls: number[] = [];
  for (let c = 0; c < chunks; c++) {
    const ids = chunkIds[c] as number[];
    const cache = newCache();
    const embedded = executor.embed(ids);
    const t0 = performance.now();
    const logits = await executor.forward(embedded, {
      seqIds: ids.map(() => `ppl-${c}`),
      positions: ids.map((_, i) => i),
      isPrefill: true,
      cache,
    });
    forwardSec += (performance.now() - t0) / 1000;
    const targets = ids.slice(1);
    const nll = totalNll(logits.data, cfg.vocabSize, targets);
    const mean = nll / targets.length;
    chunkPpls.push(pplFromMeanNll(mean));
    totalNllSum += nll;
    tokensScored += targets.length;
    console.log(`[ppl ${mode}] chunk ${c}: ${targets.length} tokens, mean NLL ${mean.toFixed(4)}, PPL ${pplFromMeanNll(mean).toFixed(2)}`);
    await cache.close();
  }
  const meanNll = totalNllSum / tokensScored;
  const perplexity = pplFromMeanNll(meanNll);
  const prefillTokPerSec = tokensScored / forwardSec;

  // Short greedy continuation from the first chunk's prompt: decode tok/s.
  const promptIds = (chunkIds[0] as number[]).slice(0, 32);
  const cache = newCache();
  const embedded = executor.embed(promptIds);
  const logits = await executor.forward(embedded, {
    seqIds: promptIds.map(() => "gen"),
    positions: promptIds.map((_, i) => i),
    isPrefill: true,
    cache,
  });
  let lastLogits = logits.data.subarray((promptIds.length - 1) * cfg.vocabSize, promptIds.length * cfg.vocabSize);
  const generated: number[] = [];
  let decodeSec = 0;
  for (let step = 0; step < 8; step++) {
    const tok = sampleLogits(lastLogits, { temperature: 0, topP: 1, topK: 0, maxTokens: 8 });
    generated.push(tok);
    if (tok === cfg.eosTokenId) break;
    const emb = executor.embed([tok]);
    const d0 = performance.now();
    const out = await executor.forward(emb, {
      seqIds: ["gen"],
      positions: [promptIds.length + step],
      isPrefill: false,
      cache,
    });
    decodeSec += (performance.now() - d0) / 1000;
    lastLogits = out.data;
  }
  await cache.close();
  const decodeTokPerSec = generated.length / Math.max(decodeSec, 1e-9);

  const result = {
    timestamp: new Date().toISOString(),
    setting: `npu=${mode}`,
    npu: mode,
    weights: "fp32",
    kv: "fp32",
    chunks,
    seqLen,
    textOffset: byteOffset,
    storiesInSlice: stories,
    tokensScored,
    meanNll: +meanNll.toFixed(5),
    perplexity: +perplexity.toFixed(3),
    chunkPpls: chunkPpls.map((p) => +p.toFixed(2)),
    prefillTokPerSec: +prefillTokPerSec.toFixed(3),
    decodeTokPerSec: +decodeTokPerSec.toFixed(3),
    loadSec: +loadSec.toFixed(1),
    weightBytes: executor.memoryBytes(),
    npuStats: executor.npuStats(),
    host: hostInfo(),
  };
  console.log(`[ppl ${mode}] PPL ${perplexity.toFixed(3)} prefill ${prefillTokPerSec.toFixed(3)} tok/s decode ${decodeTokPerSec.toFixed(3)} tok/s`);
  appendJson(PPL_OUT, result);
  console.log(`[ppl ${mode}] appended to ${PPL_OUT}`);
}

// ---------------------------------------------------------------------------
// (b) tok/s — short prompt, 16 tokens, temp 0, 2 reps per mode.
// ---------------------------------------------------------------------------

async function benchTok(mode: NpuMode): Promise<void> {
  const { executor, loadSec } = await loadExecutor(mode);
  const cfg = executor.manifest.config;
  const tokenizer = loadTokenizer(MODEL_DIR);
  const promptIds = tokenizer.encode(PROMPT);
  const sampling = { temperature: 0, topP: 1, topK: 0, maxTokens: GEN_TOKENS, seed: 1 };
  const newCache = () => new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);

  const prefillMs: number[] = [];
  for (let r = 0; r < 1; r++) {
    const cache = newCache();
    const t0 = performance.now();
    await forwardTokenRows(executor, cache, `pre-${r}`, promptIds, promptIds.map((_, i) => i), true);
    prefillMs.push(performance.now() - t0);
    await cache.close();
  }

  const genRuns: { rep: number; ms: number; tokPerSec: number; tokens: number[]; text: string }[] = [];
  for (let r = 0; r < 1; r++) {
    const cache = newCache();
    const dec = new AutoregressiveDecoder({
      executor,
      cache,
      seqId: `gen-${r}`,
      promptTokenIds: promptIds,
      sampling,
      vocabSize: cfg.vocabSize,
      eosTokenId: cfg.eosTokenId,
    });
    const t0 = performance.now();
    const res = await dec.generate(GEN_TOKENS);
    const ms = performance.now() - t0;
    genRuns.push({
      rep: r + 1,
      ms: Math.round(ms),
      tokPerSec: Number(((res.tokens.length / ms) * 1000).toFixed(2)),
      tokens: res.tokens,
      text: tokenizer.decode(res.tokens),
    });
    console.log(`[tok ${mode}] rep ${r + 1}: ${ms.toFixed(0)}ms = ${genRuns[r]!.tokPerSec} tok/s tokens=[${res.tokens.join(",")}]`);
    await cache.close();
  }

  // Decode-only: prefill once, then 8 single-row forwards, median ms.
  // (8, not 16: each vNPU decode forward re-derives the staged-weight
  // error bounds per call, so 16 would double this section's wall time
  // for no new information — the median is stable at 8.)
  const decCache = newCache();
  await forwardTokenRows(executor, decCache, "dec", promptIds, promptIds.map((_, i) => i), true);
  const decMs: number[] = [];
  let tok = promptIds[promptIds.length - 1] as number;
  for (let i = 0; i < 8; i++) {
    const t0 = performance.now();
    const out = await forwardTokenRows(executor, decCache, "dec", [tok], [promptIds.length + i], false);
    decMs.push(performance.now() - t0);
    const logits = out.data.subarray(0, cfg.vocabSize);
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    tok = best;
  }
  await decCache.close();
  const medianDecMs = median(decMs);

  const result = {
    timestamp: new Date().toISOString(),
    setting: `npu=${mode}`,
    npu: mode,
    model: "smollm-135m",
    prompt: PROMPT,
    promptTokens: promptIds.length,
    maxTokens: GEN_TOKENS,
    reps: 1,
    decodeOnlyForwards: 8,
    prefillMs: prefillMs.map((v) => Math.round(v)),
    medianPrefillMs: Math.round(median(prefillMs)),
    generateRuns: genRuns,
    medianGenerateTokPerSec: median(genRuns.map((g) => g.tokPerSec)),
    decodeOnlyMs: decMs.map((v) => Math.round(v)),
    medianDecodeMs: Number(medianDecMs.toFixed(1)),
    decodeOnlyTokPerSec: Number((1000 / medianDecMs).toFixed(2)),
    loadSec: +loadSec.toFixed(1),
    npuStats: executor.npuStats(),
    host: hostInfo(),
  };
  console.log(`[tok ${mode}] median generate ${result.medianGenerateTokPerSec} tok/s, decode-only ${result.decodeOnlyTokPerSec} tok/s`);
  appendJson(TOK_OUT, result);
  console.log(`[tok ${mode}] appended to ${TOK_OUT}`);
}

// ---------------------------------------------------------------------------
// (c) Kernel throughput — forward-opt shapes + square TOPS shapes.
// ---------------------------------------------------------------------------

function makeNode(workerId: string): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 0,
    bandwidthBytesPerSec: 0,
    roles: ["mixed"],
    maxBatchTokens: 128,
    state: "healthy",
    registeredAt: 0,
    lastHeartbeatAt: 0,
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeRam implements IVirtualRam {
  async alloc(): Promise<never> {
    throw new Error("FakeRam.alloc not used here");
  }
  async free(): Promise<void> {}
  async read(): Promise<never> {
    throw new Error("FakeRam.read not used here");
  }
  async write(): Promise<void> {}
  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: 0, usedBytes: 0, freeBytes: 0, pages: 0, nodes: 0 };
  }
}

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

function randomMatrix(rows: number, cols: number, rng: () => number): Float32Array {
  const a = new Float32Array(rows * cols);
  for (let i = 0; i < a.length; i++) a[i] = (rng() - 0.5) * 2;
  return a;
}

async function benchKernel(): Promise<void> {
  const nodes: NodeProvider = { listNodes: () => [makeNode("bench-npu")] };
  const npu = new VirtualNpu({ nodes, ram: new FakeRam() });
  const rng = mulberry32(20261122);

  // Forward-opt shapes (SmolLM-135M decode/prefill matmuls) + squares.
  const shapes: { m: number; k: number; n: number; label: string }[] = [
    { m: 1, k: 576, n: 576, label: "q/o [1,576]x[576,576]" },
    { m: 1, k: 576, n: 1536, label: "gate/up [1,576]x[1536,576]" },
    { m: 1, k: 1536, n: 576, label: "down [1,1536]x[576,1536]" },
    { m: 1, k: 576, n: 49152, label: "lm_head [1,576]x[49152,576]" },
    { m: 5, k: 576, n: 1536, label: "prefill gate [5,576]x[1536,576]" },
    { m: 256, k: 256, n: 256, label: "square 256^3" },
    { m: 512, k: 512, n: 512, label: "square 512^3" },
  ];
  const timedReps = 2;
  const rows: Record<string, unknown>[] = [];

  for (const s of shapes) {
    const a = randomMatrix(s.m, s.k, rng);
    const b = randomMatrix(s.n, s.k, rng);
    const flops = 2 * s.m * s.n * s.k;
    // FP32 baseline (the optimized kernel).
    const fp32Ms: number[] = [];
    for (let r = 0; r < timedReps; r++) {
      const t0 = performance.now();
      matmulTransposeB(a, s.m, s.k, b, s.n, s.k);
      fp32Ms.push(performance.now() - t0);
    }
    const fp32Med = median(fp32Ms);
    const ref = matmulTransposeB(a, s.m, s.k, b, s.n, s.k);
    const row: Record<string, unknown> = {
      shape: s.label,
      m: s.m,
      k: s.k,
      n: s.n,
      flops,
      fp32: { ms: +fp32Med.toFixed(2), tflops: +(flops / (fp32Med / 1000) / 1e12).toFixed(6) },
    };
    for (const format of ["int8", "fp8"] as const) {
      // Stage B once (the forward-pass reality: weights are resident,
      // the per-call cost is activation quantization + the op).
      const key = `bench-${format}-${s.m}x${s.k}x${s.n}`;
      npu.stageWeights(key, b, s.n, s.k, format);
      let maxErr = 0;
      let bound = 0;
      const ms: number[] = [];
      for (let r = 0; r < timedReps; r++) {
        const t0 = performance.now();
        const res = await npu.matmulTransposeB(a, s.m, s.k, b, s.n, s.k, { format, weightKey: key });
        ms.push(performance.now() - t0);
        if (r === 0) {
          bound = res.errorBound;
          for (let i = 0; i < res.output.length; i++) {
            const e = Math.abs(res.output[i] - ref[i]);
            if (e > maxErr) maxErr = e;
          }
        }
      }
      const med = median(ms);
      const tops = flops / (med / 1000) / 1e12;
      row[format] = {
        ms: +med.toFixed(2),
        tops: +tops.toFixed(6),
        maxAbsErrVsFp32: +maxErr.toExponential(3),
        errorBound: +bound.toExponential(3),
        withinBound: maxErr <= bound,
      };
      console.log(
        `[kernel] ${s.label} ${format}: ${med.toFixed(1)}ms = ${tops.toFixed(6)} TOPS (fp32 ${fp32Med.toFixed(1)}ms), maxErr ${maxErr.toExponential(2)} <= bound ${bound.toExponential(2)}`,
      );
      npu.evictWeights(key);
    }
    rows.push(row);
  }

  const result = {
    timestamp: new Date().toISOString(),
    note: "INT8/FP8 timed with staged (SRAM-resident) weights; per-call cost is activation quantization + the op. FP32 is the optimized matmulTransposeB kernel. Single host, 2 timed reps after warmup.",
    timedReps,
    shapes: rows,
    npuStats: npu.stats(),
    host: hostInfo(),
  };
  fs.writeFileSync(KERNEL_OUT, JSON.stringify([result], null, 2) + "\n");
  console.log(`[kernel] wrote ${KERNEL_OUT}`);
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  let modes: NpuMode[] = ["off", "int8", "fp8"];
  let skipPpl = false;
  let skipTok = false;
  let skipKernel = false;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a === "--modes") {
      const v = argv[++i];
      if (v === undefined) usage();
      modes = v.split(",") as NpuMode[];
      for (const m of modes) {
        if (m !== "off" && m !== "int8" && m !== "fp8") usage();
      }
    } else if (a === "--skip-ppl") skipPpl = true;
    else if (a === "--skip-tok") skipTok = true;
    else if (a === "--skip-kernel") skipKernel = true;
    else if (a === "--ppl-out" || a === "--tok-out" || a === "--kernel-out") {
      // Out paths are fixed at the repo root by convention; the flags
      // exist only to fail loudly instead of silently ignoring them.
      throw new Error(`${a} is not supported: outputs are vnpu-perplexity-bench.json, vnpu-tok-bench.json, vnpu-kernel-bench.json at the repo root`);
    } else usage();
  }
  console.log(`vnpu bench: modes=${modes.join(",")} host=${JSON.stringify(hostInfo())}`);
  if (!skipPpl) for (const mode of modes) await benchPerplexity(mode);
  if (!skipTok) for (const mode of modes) await benchTok(mode);
  if (!skipKernel) await benchKernel();
  console.log("vnpu bench done");
}

await main();
