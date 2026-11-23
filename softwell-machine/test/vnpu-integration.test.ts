/**
 * vNPU integration test — run with: node test/vnpu-integration.test.ts
 * (exit 0 on pass).
 *
 * The forward pass issues its matmuls through the vNPU when --npu
 * int8|fp8 is set (TransformerStageExecutor, docs/VNPU.md §4.2 item 1):
 * the same model generating from the same short prompt through the
 * vNPU path vs the FP32 baseline path, on demo-100m AND SmolLM-135M.
 *
 * Token identity is the acceptance bar — but INT8/FP8 quantization
 * changes numerics, so identity may NOT hold, and this test does not
 * hide that: a divergence fails the test with the exact detail (first
 * differing position, how many of the 16 tokens differ, both token
 * lists and decoded texts). A passing run proves the quantized path
 * generates byte-identical tokens on these prompts; a failing run
 * proves quantized execution changes tokens, and says exactly how.
 * Both modes are checked before any assertion throws, so one run
 * reports the full picture (e.g. fp8 holds while int8 diverges).
 *
 * Note the demo-100m leg is weak evidence on its own: the random
 * demo weights collapse generation to one repeated token, so identity
 * there holds trivially. SmolLM-135M is the real check.
 *
 * The test also asserts the vNPU actually issued the matmuls (its
 * counters show matmuls run and staged weights hit in SRAM), so a
 * silent fallback to the FP32 path cannot pass as integration.
 */
import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { TransformerStageExecutor } from "../src/engine/transformer.ts";
import { AutoregressiveDecoder } from "../src/engine/novel-decoding.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import type { NpuMode, NpuStats, ShardAssignment } from "../src/contracts.ts";

const PROMPT = "The capital of France is";
const MAX_TOKENS = 16;

interface GenResult {
  tokens: number[];
  text: string;
  npuStats: NpuStats | null;
}

async function generate(modelDir: string, modelId: string, mode: NpuMode): Promise<GenResult> {
  const manifest = loadManifest(modelDir, modelId);
  const tokenizer = loadTokenizer(modelDir);
  const loader = openWeightLoader(modelDir, modelId);
  const cfg = manifest.config;
  const assignment: ShardAssignment = {
    workerId: "vnpu-int",
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
  const executor = createReferenceBackend().createStageExecutor(manifest, assignment, {
    npu: mode,
  }) as TransformerStageExecutor;
  await executor.load(loader);
  const cache = new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
  try {
    const promptIds = tokenizer.encode(PROMPT);
    const decoder = new AutoregressiveDecoder({
      executor,
      cache,
      seqId: "vnpu-int",
      promptTokenIds: promptIds,
      sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: MAX_TOKENS, seed: 1 },
      vocabSize: cfg.vocabSize,
      eosTokenId: cfg.eosTokenId,
    });
    const res = await decoder.generate(MAX_TOKENS);
    return {
      tokens: res.tokens,
      text: tokenizer.decode(res.tokens),
      npuStats: executor.npuStats(),
    };
  } finally {
    await cache.close();
  }
}

/** First differing position (-1 when identical) and differing count. */
function divergence(a: number[], b: number[]): { firstDiff: number; diffCount: number } {
  let firstDiff = -1;
  let diffCount = 0;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) {
      if (firstDiff < 0) firstDiff = i;
      diffCount += 1;
    }
  }
  return { firstDiff, diffCount };
}

function describeMode(mode: NpuMode, r: GenResult): string {
  const s = r.npuStats;
  const stats =
    s === null
      ? "npu=off"
      : `npu matmuls int8=${s.int8Matmuls} fp8=${s.fp8Matmuls} sramHits=${s.sramHits} ` +
        `sramMisses=${s.sramMisses} sramSpills=${s.sramSpills} sramEvictions=${s.sramEvictions}`;
  return `${mode}: tokens=[${r.tokens.join(", ")}] text=${JSON.stringify(r.text)} (${stats})`;
}

async function checkModel(modelDir: string, modelId: string): Promise<void> {
  const base = await generate(modelDir, modelId, "off");
  const int8 = await generate(modelDir, modelId, "int8");
  const fp8 = await generate(modelDir, modelId, "fp8");
  console.log(`--- ${modelId} ---`);
  console.log(`  ${describeMode("off", base)}`);
  console.log(`  ${describeMode("int8", int8)}`);
  console.log(`  ${describeMode("fp8", fp8)}`);

  // The vNPU path must actually have issued instructions — otherwise
  // this test would pass on a silent FP32 fallback.
  for (const [mode, r] of [
    ["int8", int8],
    ["fp8", fp8],
  ] as const) {
    const s = r.npuStats;
    assert.notEqual(s, null, `${modelId} ${mode}: expected vNPU stats, got null`);
    const matmuls = mode === "int8" ? (s as NpuStats).int8Matmuls : (s as NpuStats).fp8Matmuls;
    assert.ok(matmuls > 0, `${modelId} ${mode}: expected vNPU matmuls > 0, got ${matmuls}`);
    assert.ok(
      (s as NpuStats).sramHits > 0,
      `${modelId} ${mode}: expected staged-weight SRAM hits > 0, got ${(s as NpuStats).sramHits} ` +
        `(weights were not served from SRAM)`,
    );
    assert.equal(
      (s as NpuStats).sramMisses,
      0,
      `${modelId} ${mode}: expected 0 SRAM misses, got ${(s as NpuStats).sramMisses}`,
    );
  }

  // Both modes are checked before any assertion throws, so one run
  // reports the full picture instead of aborting at the first divergence.
  const failures: string[] = [];
  for (const [mode, r] of [
    ["int8", int8],
    ["fp8", fp8],
  ] as const) {
    const d = divergence(base.tokens, r.tokens);
    if (d.firstDiff >= 0) {
      failures.push(
        `${modelId} ${mode}: quantized execution changed tokens — ` +
          `first differing position ${d.firstDiff}, ${d.diffCount} of ${base.tokens.length} tokens differ.\n` +
          `  baseline: [${base.tokens.join(", ")}] ${JSON.stringify(base.text)}\n` +
          `  ${mode}:     [${r.tokens.join(", ")}] ${JSON.stringify(r.text)}`,
      );
    }
  }
  assert.equal(failures.length, 0, failures.join("\n"));
}

test("vNPU int8/fp8 token identity vs FP32 baseline (demo-100m)", async () => {
  await checkModel("demo/models/demo-100m", "demo-100m");
});

/**
 * True when the model directory holds at least one safetensors weight
 * file. SmolLM's weights (~489 MB) are never pushed to the repo, so a
 * checkout without them (e.g. CI) cannot run the SmolLM leg at all —
 * it skips instead of failing on a missing file.
 */
function hasSafetensors(modelDir: string): boolean {
  try {
    return readdirSync(modelDir).some((f) => f.endsWith(".safetensors"));
  } catch {
    return false;
  }
}

test("vNPU int8/fp8 token identity vs FP32 baseline (smollm-135m)", async (t) => {
  if (!hasSafetensors("models/smollm-135m")) {
    t.skip("SmolLM weights not present — skipping (weights are never pushed to the repo)");
    return;
  }
  await checkModel("models/smollm-135m", "smollm-135m");
});
