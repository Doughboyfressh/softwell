/**
 * scripts/bench-sw-smollm.ts — perplexity + tok/s of SmolLM-135M under
 * SW-code weights/KV vs the fp32 baseline, on real TinyStories text.
 *
 * One setting per run; results append to sw-smollm-bench.json at the repo
 * root so settings can be run (and re-run) independently:
 *
 *   node scripts/bench-sw-smollm.ts --weights sw4 --kv fp32 --chunks 8 --seq-len 96
 *
 * Method: take a slice of training/data/TinyStories-train.txt at a fixed
 * byte offset (past the head of the file, so the slice is held out from
 * anything trained on the file's start), split it into stories on the
 * <|endoftext|> separator, tokenize, and score chunks of `seqLen` tokens
 * teacher-forced: one prefill forward per chunk, logits at position i
 * score token i+1, BOS prepended per chunk. Perplexity is exp(mean NLL)
 * over all scored tokens. Prefill tok/s counts scored tokens per second
 * of forward time; decode tok/s is a short greedy continuation (default
 * 8 tokens) from the first chunk's prompt, timed on decode forwards only.
 *
 * The executor quantizes KV internally (packedKvCache) from the fp32
 * BlockPagedKVCache, so the same cache serves every --kv mode.
 */
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import { parseEngineFlags, resolveEngineOptions } from "../src/engine/options.ts";
import { sampleLogits } from "../src/engine/sampling.ts";
import { totalNll, pplFromMeanNll } from "../src/bench/perplexity.ts";
import type { ForwardContext, ShardAssignment } from "../src/contracts.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MODEL_DIR = path.join(ROOT, "models/smollm-135m");
const STORIES_PATH = path.join(ROOT, "training/data/TinyStories-train.txt");
const OUT_PATH = path.join(ROOT, "sw-smollm-bench.json");
const SEP = "<|endoftext|>";

function usage(): never {
  console.log(
    "usage: node scripts/bench-sw-smollm.ts [--weights fp32|sw2|sw3|sw4] [--kv fp32|sw2|sw3|sw4]\n" +
      "  [--chunks N] [--seq-len N] [--offset BYTES] [--gen-tokens N] [--out PATH]",
  );
  process.exit(2);
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

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const { options, rest } = parseEngineFlags(argv);
  const resolved = resolveEngineOptions(options);
  let chunks = 8;
  let seqLen = 96;
  let byteOffset = 1_000_000_000;
  let genTokens = 8;
  let outPath = OUT_PATH;
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i] as string;
    const val = (flag: string): number => {
      const v = rest[++i];
      if (v === undefined) usage();
      const n = Number(v);
      if (!Number.isInteger(n) || n < 1) throw new Error(`invalid ${flag}: ${v}`);
      return n;
    };
    if (a === "--chunks") chunks = val(a);
    else if (a === "--seq-len") seqLen = val(a);
    else if (a === "--offset") byteOffset = val(a);
    else if (a === "--gen-tokens") genTokens = val(a);
    else if (a === "--out") {
      const v = rest[++i];
      if (v === undefined) usage();
      outPath = path.isAbsolute(v) ? v : path.join(ROOT, v);
    } else usage();
  }
  if (seqLen < 8) throw new Error("--seq-len must be >= 8");

  const loadAvg = os.loadavg();
  console.log(
    `setting: weights=${resolved.weightFormat} kv=${resolved.kvFormat} chunks=${chunks} seqLen=${seqLen} ` +
      `offset=${byteOffset} loadavg=${loadAvg.map((x) => x.toFixed(2)).join("/")}`,
  );

  const { tokens: stream, stories } = buildTokenStream(byteOffset, 256 * 1024);
  console.log(`text: ${stories} complete stories in slice, ${stream.length} tokens in stream`);
  const need = chunks * (seqLen - 1);
  if (stream.length < need) {
    throw new Error(`stream has ${stream.length} tokens, need ${need} for ${chunks}x${seqLen}`);
  }
  const tokenizer = loadTokenizer(MODEL_DIR);
  const bos = tokenizer.bosTokenId;
  const chunkIds: number[][] = [];
  for (let c = 0; c < chunks; c++) {
    chunkIds.push([bos, ...stream.slice(c * (seqLen - 1), (c + 1) * (seqLen - 1))]);
  }

  const loader = openWeightLoader(MODEL_DIR);
  const manifest = loader.manifest;
  const cfg = manifest.config;
  const assignment: ShardAssignment = {
    workerId: "sw-ppl",
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
  const executor = createReferenceBackend().createStageExecutor(manifest, assignment, resolved);
  const tLoad0 = performance.now();
  await executor.load(loader);
  const loadSec = (performance.now() - tLoad0) / 1000;
  console.log(`loaded ${executor.memoryBytes()} weight bytes in ${loadSec.toFixed(1)}s`);

  const newCache = () =>
    new BlockPagedKVCache(16, 4096, cfg.numKeyValueHeads, cfg.headDim);

  let totalNllSum = 0;
  let tokensScored = 0;
  let forwardSec = 0;
  const chunkMeanNlls: number[] = [];
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
    } as ForwardContext);
    forwardSec += (performance.now() - t0) / 1000;
    const targets = ids.slice(1);
    const nll = totalNll(logits.data, cfg.vocabSize, targets);
    const mean = nll / targets.length;
    chunkMeanNlls.push(mean);
    chunkPpls.push(pplFromMeanNll(mean));
    totalNllSum += nll;
    tokensScored += targets.length;
    console.log(
      `chunk ${c}: ${targets.length} tokens, mean NLL ${mean.toFixed(4)}, PPL ${pplFromMeanNll(mean).toFixed(2)}`,
    );
  }
  const meanNll = totalNllSum / tokensScored;
  const perplexity = pplFromMeanNll(meanNll);
  const meanOfChunks = chunkMeanNlls.reduce((a, b) => a + b, 0) / chunkMeanNlls.length;
  const seMeanNll =
    chunkMeanNlls.length > 1
      ? Math.sqrt(
          chunkMeanNlls.reduce((a, b) => a + (b - meanOfChunks) ** 2, 0) / (chunkMeanNlls.length - 1),
        ) / Math.sqrt(chunkMeanNlls.length)
      : 0;
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
  } as ForwardContext);
  let lastLogits = logits.data.subarray((promptIds.length - 1) * cfg.vocabSize, promptIds.length * cfg.vocabSize);
  const generated: number[] = [];
  let decodeSec = 0;
  for (let step = 0; step < genTokens; step++) {
    const tok = sampleLogits(lastLogits, { temperature: 0, topP: 1, topK: 0, maxTokens: genTokens });
    generated.push(tok);
    if (tok === cfg.eosTokenId) break;
    const emb = executor.embed([tok]);
    const d0 = performance.now();
    const out = await executor.forward(emb, {
      seqIds: ["gen"],
      positions: [promptIds.length + step],
      isPrefill: false,
      cache,
    } as ForwardContext);
    decodeSec += (performance.now() - d0) / 1000;
    lastLogits = out.data;
  }
  const decodeTokPerSec = generated.length / Math.max(decodeSec, 1e-9);
  const genText = tokenizer.decode(generated);

  const result = {
    timestamp: new Date().toISOString(),
    setting: `weights=${resolved.weightFormat} kv=${resolved.kvFormat}`,
    weights: resolved.weightFormat,
    kv: resolved.kvFormat,
    chunks,
    seqLen,
    textOffset: byteOffset,
    storiesInSlice: stories,
    tokensScored,
    meanNll: +meanNll.toFixed(5),
    perplexity: +perplexity.toFixed(3),
    seMeanNll: +seMeanNll.toFixed(5),
    pplLow: +pplFromMeanNll(meanNll - seMeanNll).toFixed(3),
    pplHigh: +pplFromMeanNll(meanNll + seMeanNll).toFixed(3),
    chunkPpls: chunkPpls.map((p) => +p.toFixed(2)),
    prefillTokPerSec: +prefillTokPerSec.toFixed(3),
    decodeTokPerSec: +decodeTokPerSec.toFixed(3),
    loadSec: +loadSec.toFixed(1),
    weightBytes: executor.memoryBytes(),
    genTokens: generated.length,
    genText,
    loadAvg: loadAvg.map((x) => +x.toFixed(2)),
    node: process.version,
  };
  console.log(JSON.stringify(result, null, 2));

  let prior: unknown[] = [];
  if (fs.existsSync(outPath)) {
    const parsed: unknown = JSON.parse(fs.readFileSync(outPath, "utf8"));
    if (Array.isArray(parsed)) prior = parsed;
  }
  prior.push(result);
  fs.writeFileSync(outPath, JSON.stringify(prior, null, 2) + "\n");
  console.log(`appended to ${outPath}`);
}

await main();
