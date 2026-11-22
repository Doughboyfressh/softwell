/**
 * bench/novel-bench.ts — benchmark the novel paths end to end.
 *
 * Two parts:
 *  (A) Attention-only scaling microbench (synthetic K/V, 100M-model
 *      geometry: 16 heads / 8 KV heads / headDim 64): exact prefill vs
 *      landmark (A1) vs sparse (A2) wall time at n = 256..4096, on three
 *      data regimes — coherent keys (A1's regime), concentrated attention
 *      (A2's regime), and incoherent keys (the honest worst case, where
 *      both methods degrade to exact work). Reports wall time, counted
 *      ops, and the measured error vs exact.
 *  (B) End-to-end model bench: prefill (TTFT) + decode tokens/sec +
 *      per-step latency + forwards/token for the baseline vs each novel
 *      path, on demo-10m and demo-100m, in-process (single-stage executor
 *      + BlockPagedKVCache + the decoders from novel-decoding.ts).
 *
 * Usage:
 *   node src/bench/novel-bench.ts [--models 10m,100m] [--prompt-lengths 128]
 *     [--gen-tokens 16] [--attention-lengths 256,512,1024,2048,4096]
 *     [--configs baseline,landmark,sparse,sw2,sw3,sw4,kv-sw3,kv-sw4,speculative,jacobi]
 *     [--no-attention] [--no-models] [--json <path>]
 *
 *   --models 10m,100m   which demo models to run part B on (default: both)
 *   --prompt-lengths     prefill lengths for part B (default: 128)
 *   --gen-tokens        decode tokens per config (default: 16)
 *   --attention-lengths lengths for part A (default: 256,512,1024,2048,4096)
 *   --configs           subset of part-B configs (default: all)
 *   --no-attention      skip part A
 *   --no-models         skip part B
 *   --json PATH         write the raw results JSON to PATH
 *
 * Exit code 0 when the run completes (a novel path measuring slower than
 * baseline is a result, not a failure). Non-zero only on a harness error.
 *
 * Honesty notes (read before quoting any number here):
 * - Both demo models have RANDOM weights. Their greedy output collapses
 *   to a constant token, so the n-gram drafter's measured acceptance rate
 *   (~0.95) is an artifact of that degeneracy, not drafter quality — see
 *   docs/research/03-decoding.md §4. Decode speedups are reported with
 *   the acceptance rate attached so the artifact is visible.
 * - The in-process executor is compute-bound (β ≈ 1): a batched forward
 *   costs ~one single-row forward per row. SW-code and speculative/Jacobi
 *   decoding win where bytes cross a slow boundary (vRAM -> vGPU over a
 *   network) or rows are cheap (β << 1); in-process they may measure
 *   slower, and the bench reports that plainly.
 * - Attention is a small fraction of a decode step at these prompt
 *   lengths (MLP/projections dominate); part A isolates the scaling that
 *   part B's end-to-end numbers dilute.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  EngineOptions,
  ForwardContext,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../contracts.ts";
import { groupedQueryAttention } from "../engine/attention.ts";
import {
  prefillLandmarkAttention,
  prefillSparseAttention,
} from "../engine/novel-attention.ts";
import { createReferenceBackend } from "../engine/backend.ts";
import { openWeightLoader } from "../model/loader.ts";
import { BlockPagedKVCache } from "../scheduler/kvcache.ts";
import {
  AutoregressiveDecoder,
  JacobiDecoder,
  SpeculativeDecoder,
} from "../engine/novel-decoding.ts";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));

export interface NovelBenchRow {
  part: "attention" | "model";
  /** Part A: data regime; part B: model id. */
  workload: string;
  /** Part A: sequence length; part B: prompt length. */
  promptLen: number;
  config: string;
  /** Part A: wall ms for the whole prefill. Part B: TTFT ms. */
  ttftMs: number;
  /** Part B only: decode wall ms. */
  decodeMs?: number;
  /** Part B only: tokens generated. */
  tokens?: number;
  /** Part B only: target forwards (prefill + decode). */
  forwards?: number;
  /** Part B only: rows fed to forward(). */
  rowsProcessed?: number;
  /** Part B only: measured draft acceptance (speculative) or 0. */
  acceptanceRate?: number;
  /** Part B only: generated tokens per decode second. */
  tokensPerSec?: number;
  /** Part B only: decode wall ms per generated token. */
  msPerToken?: number;
  /** Part A only: counted ops / exact ops. */
  opsRatio?: number;
  /** Part A only: max L2 error vs exact attention output. */
  maxErr?: number;
  /** Part A only: error bound returned by the theorem. */
  errorBound?: number;
  /** Part B only: executor load wall ms (includes SW-code encoding). */
  loadMs?: number;
  /** Part B only: executor weight bytes. */
  weightBytes?: number;
  notes: string;
}

// ---------------------------------------------------------------------------
// Small deterministic PRNG (mulberry32) + helpers
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Machine info, read from node:os at runtime — never hardcoded. */
function describeMachine(): string {
  const cpus = os.cpus();
  const model = cpus.length > 0 ? (cpus[0]?.model ?? "unknown") : "unknown";
  const cores = cpus.length;
  const ramGiB = (os.totalmem() / 1024 ** 3).toFixed(1);
  return `Machine: ${model} (${cores} cores), ${ramGiB} GiB RAM, Node ${process.version}, ${os.platform()} ${os.arch()}`;
}

function timeIt(fn: () => void): number {
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
}

// ---------------------------------------------------------------------------
// Part A: attention-only scaling
// ---------------------------------------------------------------------------

type Regime = "coherent" | "concentrated" | "incoherent";

function makeAttentionData(
  regime: Regime,
  n: number,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  seed: number,
): { q: Float32Array; k: Float32Array; v: Float32Array } {
  const rand = mulberry32(seed);
  const randn = (): number => {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const q = new Float32Array(n * numHeads * headDim);
  const k = new Float32Array(n * numKvHeads * headDim);
  const v = new Float32Array(n * numKvHeads * headDim);
  for (let i = 0; i < q.length; i++) q[i] = randn();
  for (let i = 0; i < v.length; i++) v[i] = randn();
  if (regime === "incoherent") {
    for (let i = 0; i < k.length; i++) k[i] = randn();
    return { q, k, v };
  }
  // Unit direction u shared by the coherent / concentrated structure.
  const u = new Float32Array(headDim);
  let norm = 0;
  for (let d = 0; d < headDim; d++) {
    u[d] = randn();
    norm += u[d] * u[d];
  }
  norm = Math.sqrt(norm);
  for (let d = 0; d < headDim; d++) u[d] /= norm;
  const blockSize = Math.max(4, Math.ceil(Math.sqrt(n)));
  if (regime === "coherent") {
    // Block-constant keys + tiny noise: every block is approximable.
    // Calibration matters: landmark approximates a block only when
    // delta_b = scale*||q||*R_b <= deltaMax (0.05). At headDim 64,
    // ||q|| ~= 8 and scale = 1/8, so this needs R_b ≲ 0.05, i.e.
    // per-element within-block noise ≲ 0.006. Noise 0.001 gives
    // R_b ~= 0.01 and delta_b ~= 0.01. (Noise 0.01, tried first, gives
    // delta_b ~= 0.1 and the landmark path falls back to exact
    // everywhere — the approximation never triggers and the bench
    // measures fallback, not the novel path.)
    for (let t = 0; t < n; t++) {
      const b = Math.floor(t / blockSize);
      const s = Math.sin(b * 1.7 + 1) * 2; // block-constant signal
      for (let kh = 0; kh < numKvHeads; kh++) {
        for (let d = 0; d < headDim; d++) {
          k[(t * numKvHeads + kh) * headDim + d] = s * u[d] + randn() * 0.001;
        }
      }
    }
    return { q, k, v };
  }
  // Concentrated: block 0 aligned with +u, all other blocks anti-aligned,
  // queries aligned with +u — attention concentrates on block 0.
  // Calibration matters: sparse skips a block only when its mass bound
  // B*exp(u_b - M) <= dropThreshold (1e-3), i.e. the logit gap to the
  // best block must exceed ln(B/1e-3) ~= 10.7. With scale = 1/8 and
  // q.u ~= 4, that needs key alignment of about +-12u (gap 12 in logit
  // units). Weaker separation (+-3u, tried first) gives a gap of ~1.5
  // and nothing is skipped. Key norms ~12 are large, but this is the
  // regime the screening mechanism is defined for; the parameters are
  // reported exactly and the incoherent regime measures the fallback.
  for (let i = 0; i < q.length; i++) q[i] = randn() * 0.1;
  for (let t = 0; t < n; t++) {
    const align = t < blockSize ? 12 : -12;
    for (let kh = 0; kh < numKvHeads; kh++) {
      for (let d = 0; d < headDim; d++) {
        k[(t * numKvHeads + kh) * headDim + d] = align * u[d] + randn() * 0.1;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        q[(i * numHeads + h) * headDim + d] += 4 * u[d];
      }
    }
  }
  return { q, k, v };
}

/** Exact prefill by looping groupedQueryAttention over rows (the O(n^2 d) reference). */
function exactPrefill(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
): Float32Array {
  const n = q.length / (numHeads * headDim);
  const out = new Float32Array(n * numHeads * headDim);
  for (let i = 0; i < n; i++) {
    const qRow = q.subarray(i * numHeads * headDim, (i + 1) * numHeads * headDim);
    const kPre = k.subarray(0, (i + 1) * numKvHeads * headDim);
    const vPre = v.subarray(0, (i + 1) * numKvHeads * headDim);
    const headOut = groupedQueryAttention(qRow, kPre, vPre, numHeads, numKvHeads, headDim);
    out.set(headOut, i * numHeads * headDim);
  }
  return out;
}

function maxL2Diff(a: Float32Array, b: Float32Array, rows: number, rowLen: number): number {
  let max = 0;
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let d = 0; d < rowLen; d++) {
      const diff = a[r * rowLen + d] - b[r * rowLen + d];
      sum += diff * diff;
    }
    const n = Math.sqrt(sum);
    if (n > max) max = n;
  }
  return max;
}

async function runAttentionPart(lengths: number[]): Promise<NovelBenchRow[]> {
  const rows: NovelBenchRow[] = [];
  const numHeads = 16;
  const numKvHeads = 8;
  const headDim = 64;
  const regimes: Regime[] = ["coherent", "concentrated", "incoherent"];
  for (const regime of regimes) {
    let prevExactMs: number | null = null;
    for (const n of lengths) {
      const { q, k, v } = makeAttentionData(regime, n, numHeads, numKvHeads, headDim, 1234 + n);
      // Exact (timed once; its output is the error reference).
      let exactOut: Float32Array = new Float32Array(0);
      const exactMs = timeIt(() => {
        exactOut = exactPrefill(q, k, v, numHeads, numKvHeads, headDim);
      });
      const exactOps = numHeads * headDim * n * (n + 1);
      const scaleNote =
        prevExactMs !== null ? ` (exact ${((exactMs / prevExactMs)).toFixed(2)}× vs n=${n / 2})` : "";
      rows.push({
        part: "attention",
        workload: regime,
        promptLen: n,
        config: "exact",
        ttftMs: exactMs,
        notes: `exact prefill, ops ${exactOps.toExponential(2)}${scaleNote}`,
      });
      prevExactMs = exactMs;
      for (const mode of ["landmark", "sparse"] as const) {
        const fn = mode === "landmark" ? prefillLandmarkAttention : prefillSparseAttention;
        let r: ReturnType<typeof fn> | undefined;
        const ms = timeIt(() => {
          r = fn(q, k, v, numHeads, numKvHeads, headDim);
        });
        if (r === undefined) throw new Error("unreachable: prefill function did not run");
        const err = maxL2Diff(r.output, exactOut, n, numHeads * headDim);
        rows.push({
          part: "attention",
          workload: regime,
          promptLen: n,
          config: mode,
          ttftMs: ms,
          opsRatio: r.ops / r.exactOps,
          maxErr: err,
          errorBound: r.errorBound,
          notes:
            `${mode} prefill: ${ms.toFixed(0)} ms vs exact ${exactMs.toFixed(0)} ms ` +
            `(${(exactMs / ms).toFixed(2)}×), ops ${r.ops / r.exactOps}x, ` +
            `max err ${err.toExponential(1)} vs bound ${r.errorBound.toExponential(1)}` +
            (err > r.errorBound ? " — BOUND VIOLATED" : ""),
        });
      }
      console.log(
        `  attention/${regime} n=${n}: exact ${exactMs.toFixed(0)} ms, ` +
          `landmark ${rows[rows.length - 2]?.ttftMs.toFixed(0)} ms, sparse ${rows[rows.length - 1]?.ttftMs.toFixed(0)} ms`,
      );
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Part B: end-to-end model bench
// ---------------------------------------------------------------------------

interface ModelConfig {
  id: string;
  dir: string;
}

const PART_B_CONFIGS: Array<{ name: string; options: EngineOptions; decoding: "autoregressive" | "speculative" | "jacobi" }> = [
  { name: "baseline", options: {}, decoding: "autoregressive" },
  { name: "landmark", options: { attention: "landmark" }, decoding: "autoregressive" },
  { name: "sparse", options: { attention: "sparse" }, decoding: "autoregressive" },
  { name: "sw2", options: { weightFormat: "sw2" }, decoding: "autoregressive" },
  { name: "sw3", options: { weightFormat: "sw3" }, decoding: "autoregressive" },
  { name: "sw4", options: { weightFormat: "sw4" }, decoding: "autoregressive" },
  { name: "kv-sw3", options: { kvFormat: "sw3" }, decoding: "autoregressive" },
  { name: "kv-sw4", options: { kvFormat: "sw4" }, decoding: "autoregressive" },
  { name: "speculative", options: {}, decoding: "speculative" },
  { name: "jacobi", options: {}, decoding: "jacobi" },
];

/**
 * StageExecutor wrapper that times prefill forwards and decode forwards
 * separately, inside one generate() call — so TTFT and decode tokens/sec
 * are measured directly, never derived by subtracting two separately
 * timed runs (whose warmup/noise would swamp the decode time at these
 * sizes). Delegates everything else to the wrapped executor.
 */
class TimingExecutor implements StageExecutor {
  readonly assignment: ShardAssignment;
  prefillMs = 0;
  decodeMs = 0;
  prefillForwards = 0;
  decodeForwards = 0;
  decodeRows = 0;
  private readonly inner: StageExecutor;

  constructor(inner: StageExecutor) {
    this.inner = inner;
    this.assignment = inner.assignment;
  }

  async load(loader: WeightLoader): Promise<void> {
    return this.inner.load(loader);
  }

  embed(tokenIds: number[]): Tensor {
    return this.inner.embed(tokenIds);
  }

  forward(input: Tensor, ctx: ForwardContext): Tensor {
    const t0 = performance.now();
    const out = this.inner.forward(input, ctx);
    const ms = performance.now() - t0;
    if (ctx.isPrefill) {
      this.prefillMs += ms;
      this.prefillForwards += 1;
    } else {
      this.decodeMs += ms;
      this.decodeForwards += 1;
      this.decodeRows += input.shape[0];
    }
    return out;
  }

  memoryBytes(): number {
    return this.inner.memoryBytes();
  }
}

async function runModelConfig(
  model: ModelConfig,
  configName: string,
  options: EngineOptions,
  decoding: "autoregressive" | "speculative" | "jacobi",
  promptLen: number,
  genTokens: number,
): Promise<NovelBenchRow> {
  const modelDir = path.join(ROOT, model.dir);
  const loader = openWeightLoader(modelDir);
  const manifest = loader.manifest;
  const assignment: ShardAssignment = {
    workerId: "bench",
    stage: 0,
    layerStart: 0,
    layerEnd: manifest.config.numHiddenLayers,
    hasEmbedding: true,
    hasLmHead: true,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 0,
  };
  const rawExecutor = createReferenceBackend().createStageExecutor(manifest, assignment, options);
  const executor = new TimingExecutor(rawExecutor);
  const tLoad0 = performance.now();
  await executor.load(loader);
  const loadMs = performance.now() - tLoad0;
  const weightBytes = executor.memoryBytes();
  // Deterministic prompt: seeded token ids in vocab range.
  const rand = mulberry32(7);
  const prompt = Array.from({ length: promptLen }, () => Math.floor(rand() * manifest.config.vocabSize));
  const sampling = { temperature: 0, topP: 1, topK: 0, maxTokens: genTokens };

  // Warmup (same executor, scratch cache/seqId): pays JIT + first-touch
  // costs so the timed run measures steady-state, not warmup.
  const warmCache = new BlockPagedKVCache(16, 4096, manifest.config.numKeyValueHeads, manifest.config.headDim);
  new AutoregressiveDecoder({
    executor,
    cache: warmCache,
    seqId: `warm-${model.id}-${configName}`,
    promptTokenIds: prompt.slice(0, Math.min(16, prompt.length)),
    sampling: { ...sampling, maxTokens: 2 },
    vocabSize: manifest.config.vocabSize,
    eosTokenId: manifest.config.eosTokenId,
  }).generate();
  executor.prefillMs = 0;
  executor.decodeMs = 0;
  executor.prefillForwards = 0;
  executor.decodeForwards = 0;
  executor.decodeRows = 0;

  const cache = new BlockPagedKVCache(
    16,
    4096,
    manifest.config.numKeyValueHeads,
    manifest.config.headDim,
  );
  const decoderOpts = {
    executor,
    cache,
    seqId: `bench-${model.id}-${configName}`,
    promptTokenIds: prompt,
    sampling,
    vocabSize: manifest.config.vocabSize,
    eosTokenId: manifest.config.eosTokenId,
  };
  let result;
  if (decoding === "speculative") {
    result = new SpeculativeDecoder({ ...decoderOpts, draftLength: 4 }).generate();
  } else if (decoding === "jacobi") {
    result = new JacobiDecoder({ ...decoderOpts, blockSize: 4 }).generate();
  } else {
    result = new AutoregressiveDecoder(decoderOpts).generate();
  }
  const ttftMs = executor.prefillMs;
  const decodeMs = executor.decodeMs;
  const tokens = result.tokens.length;
  loader.close();
  const tokensPerSec = decodeMs > 0 ? (tokens / decodeMs) * 1000 : 0;
  return {
    part: "model",
    workload: model.id,
    promptLen,
    config: configName,
    ttftMs,
    decodeMs,
    tokens,
    forwards: result.forwards,
    rowsProcessed: result.rowsProcessed,
    acceptanceRate: result.acceptanceRate,
    tokensPerSec,
    msPerToken: tokens > 0 ? decodeMs / tokens : 0,
    loadMs,
    weightBytes,
    notes:
      `${configName} on ${model.id} (prompt ${promptLen}, ${tokens} tokens): ` +
      `TTFT ${ttftMs.toFixed(0)} ms, decode ${decodeMs.toFixed(0)} ms ` +
      `(${tokensPerSec.toFixed(2)} tok/s), ${result.forwards} forwards / ${result.rowsProcessed} rows` +
      (decoding === "speculative" ? `, acceptance ${(result.acceptanceRate * 100).toFixed(0)}%` : ""),
  };
}

async function runModelPart(
  models: ModelConfig[],
  promptLengths: number[],
  genTokens: number,
  configNames: string[],
): Promise<NovelBenchRow[]> {
  const rows: NovelBenchRow[] = [];
  for (const model of models) {
    for (const promptLen of promptLengths) {
      for (const cfg of PART_B_CONFIGS) {
        if (!configNames.includes(cfg.name)) continue;
        const row = await runModelConfig(model, cfg.name, cfg.options, cfg.decoding, promptLen, genTokens);
        rows.push(row);
        console.log(`  ${row.notes}`);
      }
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function usage(): string {
  return [
    "Usage: node src/bench/novel-bench.ts [--models 10m,100m] [--prompt-lengths 128]",
    "  [--gen-tokens 16] [--attention-lengths 256,512,1024,2048,4096]",
    "  [--configs baseline,landmark,...] [--no-attention] [--no-models] [--json PATH]",
    `  configs: ${PART_B_CONFIGS.map((c) => c.name).join(",")}`,
  ].join("\n");
}

function parseList(value: string): string[] {
  return value.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
}

function parseInts(value: string, name: string): number[] {
  const out = parseList(value).map((s) => Number(s));
  if (out.some((n) => !Number.isInteger(n) || n < 1)) {
    throw new Error(`--${name} must be a comma-separated list of positive integers, got ${value}`);
  }
  return out;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  let models: ModelConfig[] = [
    { id: "demo-10m", dir: "demo/models/demo-10m" },
    { id: "demo-100m", dir: "demo/models/demo-100m" },
  ];
  let promptLengths = [128];
  let genTokens = 16;
  let attentionLengths = [256, 512, 1024, 2048, 4096];
  let configNames = PART_B_CONFIGS.map((c) => c.name);
  let runAttention = true;
  let runModels = true;
  let jsonOut: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    const [flag, inline] = arg.split("=", 2);
    const value = (name: string): string => {
      if (inline !== undefined) return inline;
      const next = argv[++i];
      if (next === undefined) throw new Error(`missing value for ${name}`);
      return next;
    };
    if (flag === "--help" || flag === "-h") {
      console.log(usage());
      process.exit(0);
    } else if (flag === "--models") {
      const ids = parseList(value("--models"));
      const all: ModelConfig[] = [
        { id: "demo-10m", dir: "demo/models/demo-10m" },
        { id: "demo-100m", dir: "demo/models/demo-100m" },
      ];
      models = ids.map((id) => {
        const m = all.find((a) => a.id === id);
        if (m === undefined) throw new Error(`--models: unknown model ${id} (expected demo-10m or demo-100m)`);
        return m;
      });
    } else if (flag === "--prompt-lengths") {
      promptLengths = parseInts(value("--prompt-lengths"), "prompt-lengths");
    } else if (flag === "--gen-tokens") {
      genTokens = Number(value("--gen-tokens"));
      if (!Number.isInteger(genTokens) || genTokens < 1) throw new Error(`--gen-tokens must be a positive integer, got ${genTokens}`);
    } else if (flag === "--attention-lengths") {
      attentionLengths = parseInts(value("--attention-lengths"), "attention-lengths");
    } else if (flag === "--configs") {
      configNames = parseList(value("--configs"));
      for (const name of configNames) {
        if (!PART_B_CONFIGS.some((c) => c.name === name)) {
          throw new Error(`--configs: unknown config ${name}`);
        }
      }
    } else if (flag === "--no-attention") {
      runAttention = false;
    } else if (flag === "--no-models") {
      runModels = false;
    } else if (flag === "--json") {
      jsonOut = value("--json");
    } else {
      throw new Error(`unknown argument: ${arg}\n${usage()}`);
    }
  }

  console.log("softwell novel-paths benchmark");
  console.log(describeMachine());
  console.log(`models: ${models.map((m) => m.id).join(", ")} | prompt lengths: ${promptLengths.join(", ")} | gen tokens: ${genTokens}`);
  console.log(`configs: ${configNames.join(", ")} | attention lengths: ${attentionLengths.join(", ")}`);
  console.log("");

  const rows: NovelBenchRow[] = [];
  if (runAttention) {
    console.log("=== Part A: attention-only scaling (16 heads / 8 KV heads / d=64) ===");
    rows.push(...(await runAttentionPart(attentionLengths)));
    console.log("");
  }
  if (runModels) {
    console.log("=== Part B: end-to-end model bench ===");
    rows.push(...(await runModelPart(models, promptLengths, genTokens, configNames)));
    console.log("");
  }

  // Summary table: speedup vs the baseline in the same (part, workload, promptLen).
  const baselineByKey = new Map<string, NovelBenchRow>();
  for (const r of rows) {
    if (r.config === "baseline" || r.config === "exact") {
      baselineByKey.set(`${r.part}|${r.workload}|${r.promptLen}`, r);
    }
  }
  console.log("=== Summary (speedup vs baseline in the same workload/length) ===");
  for (const r of rows) {
    const base = baselineByKey.get(`${r.part}|${r.workload}|${r.promptLen}`);
    if (r.part === "attention") {
      const speed = base !== undefined && base.ttftMs > 0 ? base.ttftMs / r.ttftMs : 0;
      console.log(
        `attention/${r.workload} n=${r.promptLen} ${r.config.padEnd(9)} ` +
          `${r.ttftMs.toFixed(0).padStart(8)} ms  ${speed.toFixed(2)}x vs exact` +
          (r.opsRatio !== undefined ? `  ops ${r.opsRatio.toFixed(2)}x` : "") +
          (r.maxErr !== undefined ? `  max err ${r.maxErr.toExponential(1)} (bound ${r.errorBound?.toExponential(1)})` : ""),
      );
    } else {
      const speed = base !== undefined && base.tokensPerSec !== undefined && base.tokensPerSec > 0 && r.tokensPerSec !== undefined
        ? r.tokensPerSec / base.tokensPerSec
        : 0;
      console.log(
        `model/${r.workload} p=${r.promptLen} ${r.config.padEnd(12)} ` +
          `TTFT ${r.ttftMs.toFixed(0).padStart(7)} ms  decode ${(r.tokensPerSec ?? 0).toFixed(2)} tok/s  ${speed.toFixed(2)}x vs baseline` +
          (r.forwards !== undefined ? `  forwards ${r.forwards}` : ""),
      );
    }
  }

  if (jsonOut !== undefined) {
    const payload = {
      date: new Date().toISOString(),
      machine: describeMachine(),
      node: process.version,
      rows,
    };
    fs.writeFileSync(jsonOut, JSON.stringify(payload, null, 2));
    console.log(`\nraw results written to ${jsonOut}`);
  }
  await new Promise<void>((resolve) => {
    process.stdout.write("", () => resolve());
  });
  process.exit(0);
}

main().catch((err: unknown) => {
  console.error(`novel bench harness error: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
