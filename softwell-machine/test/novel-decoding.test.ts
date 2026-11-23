/**
 * Novel-decoding tests — run with: node test/novel-decoding.test.ts
 *
 * Two layers:
 *  (1) A scripted fake executor (deterministic next-token function) for
 *      logic edge cases: EOS mid-draft, empty draft, draft longer than
 *      remaining maxTokens, Jacobi convergence. Fast and exact.
 *  (2) The real demo-10m model (random weights, hidden 384, 6 layers)
 *      for the properties that matter: speculative and Jacobi output must
 *      equal plain autoregressive output token-for-token (greedy), and the
 *      n-gram drafter's acceptance rate is MEASURED and reported honestly.
 *      Random weights mean the drafter will mostly be wrong — the test
 *      asserts the measurement is coherent with the cost model, not that
 *      speculation wins on this model. If it does not win, the test says
 *      so in its output; see docs/research/03-decoding.md §S1.4.
 */

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import * as path from "node:path";
import type {
  ForwardContext,
  PagedKVCache,
  SamplingParams,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../src/contracts.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import { ByteLevelTokenizer } from "../src/model/tokenizer.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  AutoregressiveDecoder,
  breakevenAcceptance,
  breakevenJacobiGamma,
  expectedForwardsPerToken,
  expectedJacobiForwardsPerBlock,
  expectedSpeculativeTokensPerRound,
  JacobiDecoder,
  jacobiSpeedup,
  NGramDrafter,
  SpeculativeDecoder,
  speculativeSpeedup,
  type Drafter,
} from "../src/engine/novel-decoding.ts";

// ---------------------------------------------------------------------------
// Cost-model unit checks (pure functions, no model needed)
// ---------------------------------------------------------------------------

{
  assert.equal(expectedSpeculativeTokensPerRound(0, 4), 1, "alpha=0: 1 token/round");
  assert.equal(expectedSpeculativeTokensPerRound(1, 4), 5, "alpha=1: K+1 tokens/round");
  assert.ok(
    Math.abs(expectedSpeculativeTokensPerRound(0.5, 2) - 1.75) < 1e-12,
    "alpha=0.5, K=2: 1 + 0.5 + 0.25",
  );
  assert.equal(expectedForwardsPerToken(0, 4), 1, "alpha=0: 1 forward/token");
  assert.ok(
    Math.abs(expectedForwardsPerToken(0.5, 4) - 1 / (1 + 0.5 + 0.25 + 0.125 + 0.0625)) < 1e-12,
    "forwards/token is 1 / E[tokens/round]",
  );
  assert.equal(breakevenAcceptance(4, 0), 0, "beta=0: any alpha > 0 breaks even");
  // At beta=1 the steady-state round costs K+1 row-units for at most K+1
  // tokens, so the breakeven alpha is exactly 1 (speedup <= 1 always).
  const be1 = breakevenAcceptance(4, 1) as number;
  assert.ok(Math.abs(be1 - 1) < 1e-9, `beta=1 breakeven is 1, got ${be1}`);
  assert.ok(
    Math.abs(speculativeSpeedup(be1, 4, 1) - 1) < 1e-9,
    "speedup at the breakeven alpha is 1",
  );
  assert.ok(speculativeSpeedup(0, 4, 1) < 1, "alpha=0 loses at beta=1 (verification costs K+1)");
  assert.ok(
    Math.abs(speculativeSpeedup(1, 4, 1) - 1) < 1e-12,
    "alpha=1 ties at beta=1 (K+1 tokens for K+1 rows of work)",
  );
  assert.ok(
    Math.abs(speculativeSpeedup(1, 4, 0.5) - 5 / 3) < 1e-12,
    "alpha=1, beta=0.5: 5 tokens / 3 row-costs",
  );

  assert.equal(expectedJacobiForwardsPerBlock(1, 4), 1, "gamma=1: 1 forward/block");
  assert.equal(expectedJacobiForwardsPerBlock(0, 4), 5, "gamma=0: K+1 forwards/block");
  assert.equal(jacobiSpeedup(1, 4, 0), 4, "gamma=1, beta=0: 4x (K tokens / 1 forward)");
  // NOTE: unlike speculative decoding, Jacobi at gamma=0 is *worse* than
  // baseline even at beta=0 (K+1 forwards for K tokens: the commit/detection
  // forward), so its beta=0 breakeven is strictly positive.
  const beJ0 = breakevenJacobiGamma(4, 0) as number;
  assert.ok(beJ0 > 0 && beJ0 < 1, `Jacobi beta=0 breakeven in (0,1), got ${beJ0}`);
  assert.ok(Math.abs(jacobiSpeedup(beJ0, 4, 0) - 1) < 1e-9, "Jacobi speedup at beta=0 breakeven is 1");
  // At beta=1 Jacobi speedup is 1/E[T] <= 1 with equality only at gamma=1:
  // a K-row forward costs K, so Jacobi can never beat baseline on a
  // compute-bound backend — it needs beta < 1 (bandwidth-bound) to win.
  const beJ = breakevenJacobiGamma(4, 1) as number;
  assert.ok(Math.abs(beJ - 1) < 1e-9, `Jacobi beta=1 breakeven is 1, got ${beJ}`);
  assert.ok(Math.abs(jacobiSpeedup(1, 4, 1) - 1) < 1e-12, "Jacobi gamma=1, beta=1: speedup exactly 1");
  assert.ok(jacobiSpeedup(0.5, 4, 1) < 1, "Jacobi gamma=0.5, beta=1: loses");
  console.log("ok cost model: speculative + Jacobi expectations and breakevens cohere");
}

// ---------------------------------------------------------------------------
// NGramDrafter unit checks
// ---------------------------------------------------------------------------

{
  const d = new NGramDrafter(8);
  // Suffix [1,2] occurred at position 0, followed by 3.
  assert.equal(d.nextToken([1, 2, 3, 1, 2]), 3, "n-gram copies the token after the most recent match");
  // Longest suffix wins: [2,3] occurred (followed by 1), [3] also occurred.
  assert.equal(d.nextToken([1, 2, 3, 1, 2, 3]), 1, "longest suffix match wins");
  assert.deepEqual(d.draft([5, 5], 3), [5, 5, 5], "no match: repeat the last token");
  assert.deepEqual(d.draft([1, 2, 3], 0), [], "draft(…, 0) is empty");
  console.log("ok NGramDrafter: longest-suffix copy, repeat fallback");
}

// ---------------------------------------------------------------------------
// Scripted fake executor: next(token) is a fixed function, EOS on token 7.
// next(t) = (5t + 11) mod 14 for t != 7; next(7) = EOS (2). Vocab 16.
// Prompt [1, 3] therefore generates greedily [12, 1, 2(EOS)].
// The fake ignores the KV cache (it reads the token id out of the hidden
// state), so it tests decoder *logic*; cache correctness is tested on the
// real executor below.
// ---------------------------------------------------------------------------

const FAKE_VOCAB = 16;
const FAKE_EOS = 2;

function fakeNext(token: number): number {
  if (token === 7) return FAKE_EOS;
  return (5 * token + 11) % 14;
}

class FakeExecutor implements StageExecutor {
  readonly assignment: ShardAssignment;
  constructor(assignment: ShardAssignment) {
    this.assignment = assignment;
  }
  async load(_loader: WeightLoader): Promise<void> {}
  embed(tokenIds: number[]): Tensor {
    const data = new Float32Array(tokenIds.length * 4);
    for (let r = 0; r < tokenIds.length; r++) data[r * 4] = tokenIds[r] as number;
    return { data, shape: [tokenIds.length, 4] };
  }
  async forward(input: Tensor, ctx: ForwardContext): Promise<Tensor> {
    const rows = input.shape[0];
    const logits = new Float32Array(rows * FAKE_VOCAB);
    for (let r = 0; r < rows; r++) {
      const token = Math.round(input.data[r * 4] as number);
      const next = fakeNext(token);
      logits[r * FAKE_VOCAB + next] = 10;
      void ctx;
    }
    return { data: logits, shape: [rows, FAKE_VOCAB] };
  }
  memoryBytes(): number {
    return 0;
  }
}

const fakeAssignment: ShardAssignment = {
  workerId: "fake",
  stage: 0,
  layerStart: 0,
  layerEnd: 0,
  hasEmbedding: true,
  hasLmHead: true,
  tensorParallelRank: 0,
  tensorParallelSize: 1,
  experts: [],
  estimatedMemoryBytes: 0,
};

function fakeCache(): PagedKVCache {
  return new BlockPagedKVCache(16, 64, 2, 8);
}

const greedy: SamplingParams = { temperature: 0, topP: 1, topK: 0, maxTokens: 8 };
const fakePrompt = [1, 3];

{
  const fake = new FakeExecutor(fakeAssignment);
  const base = await new AutoregressiveDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "base",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
  }).generate();
  assert.deepEqual(base.tokens, [12, 1, 2], "fake baseline generates [12, 1, EOS]");
  assert.equal(base.finishReason, "eos", "fake baseline finishes on EOS");
  assert.equal(base.forwards, 3, "fake baseline: prefill + 2 decode forwards");
  console.log("ok fake executor baseline: [12, 1, EOS] in 3 forwards");
}

class NullDrafter implements Drafter {
  draft(_context: number[], _count: number): number[] {
    return [];
  }
}

class FixedDrafter implements Drafter {
  private readonly seq: number[];
  constructor(seq: number[]) {
    this.seq = seq;
  }
  draft(_context: number[], count: number): number[] {
    return this.seq.slice(0, count);
  }
}

{
  const fake = new FakeExecutor(fakeAssignment);
  // Empty draft: must fall back to plain autoregressive steps, exactly.
  const spec = await new SpeculativeDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "spec-empty",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
    draftLength: 4,
    drafter: new NullDrafter(),
  }).generate();
  assert.deepEqual(spec.tokens, [12, 1, 2], "empty draft: tokens equal baseline");
  assert.equal(spec.forwards, 3, "empty draft: forwards equal baseline (no wasted verification)");
  assert.equal(spec.draftProposed, 0, "empty draft: nothing proposed");
  console.log("ok edge case: empty draft falls back to plain decoding, token-identical");

  // Draft longer than remaining maxTokens (K=8, maxTokens=3): exact, no overrun.
  const specLong = await new SpeculativeDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "spec-long",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
    draftLength: 8,
  }).generate(3);
  assert.deepEqual(specLong.tokens, [12, 1, 2], "draft longer than maxTokens: exact, capped at 3");
  console.log("ok edge case: draftLength 8 > maxTokens 3, exact and capped");

  // EOS mid-draft: drafter proposes [12, 1, EOS, 9]; the EOS is accepted
  // as the third draft token and generation must stop there — the 9 and
  // the bonus token must never appear.
  const specEos = await new SpeculativeDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "spec-eos",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
    draftLength: 4,
    drafter: new FixedDrafter([12, 1, 2, 9]),
  }).generate();
  assert.deepEqual(specEos.tokens, [12, 1, 2], "EOS mid-draft: stops at EOS, drops later drafts + bonus");
  assert.equal(specEos.finishReason, "eos", "EOS mid-draft: finish reason is eos");
  console.log("ok edge case: EOS accepted mid-draft stops generation, later drafts dropped");

  // Jacobi on the fake: converges to the same tokens, EOS included.
  const jac = await new JacobiDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "jac-eos",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
    blockSize: 4,
    drafter: new FixedDrafter([12, 1, 2, 9]),
  }).generate();
  assert.deepEqual(jac.tokens, [12, 1, 2], "Jacobi: tokens equal baseline, EOS stops the block");
  assert.equal(jac.finishReason, "eos", "Jacobi: finish reason is eos");
  console.log("ok edge case: Jacobi with EOS in the initial guess, token-identical");

  // Jacobi with a fully wrong initial guess still converges (worst case:
  // blockSize + 1 forwards for the block, by Theorem S2).
  const jacWrong = await new JacobiDecoder({
    executor: fake,
    cache: fakeCache(),
    seqId: "jac-wrong",
    promptTokenIds: fakePrompt,
    sampling: greedy,
    vocabSize: FAKE_VOCAB,
    eosTokenId: FAKE_EOS,
    blockSize: 4,
    drafter: new FixedDrafter([9, 9, 9, 9]),
  }).generate();
  assert.deepEqual(jacWrong.tokens, [12, 1, 2], "Jacobi from an all-wrong guess still converges exactly");
  assert.ok(jacWrong.forwards <= 6, `Jacobi worst case <= K+2 forwards, got ${jacWrong.forwards}`);
  console.log(`ok Jacobi worst case: all-wrong guess converges in ${jacWrong.forwards} forwards, ${jacWrong.iterations} iterations`);

  // Sampled speculative decoding is deterministic under a fixed seed and
  // stays in-vocab (distributional exactness is a theorem, not a
  // token-equality claim, so the test asserts determinism, not equality
  // with the greedy baseline).
  const sampled: SamplingParams = { temperature: 0.7, topP: 1, topK: 0, maxTokens: 6, seed: 42 };
  const mkSpec = async () =>
    await new SpeculativeDecoder({
      executor: fake,
      cache: fakeCache(),
      seqId: "spec-sampled",
      promptTokenIds: fakePrompt,
      sampling: sampled,
      vocabSize: FAKE_VOCAB,
      eosTokenId: FAKE_EOS,
      draftLength: 3,
    }).generate();
  const s1 = await mkSpec();
  const s2 = await mkSpec();
  assert.deepEqual(s1.tokens, s2.tokens, "sampled speculative: same seed => same tokens");
  for (const t of s1.tokens) assert.ok(t >= 0 && t < FAKE_VOCAB, `sampled token ${t} in vocab`);
  console.log(`ok sampled speculative decoding is deterministic under a fixed seed: [${s1.tokens.join(", ")}]`);
}

// ---------------------------------------------------------------------------
// Real demo-10m model: exactness + honest acceptance measurement
// ---------------------------------------------------------------------------

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MODEL_DIR = path.join(ROOT, "demo/models/demo-10m");
const loader = openWeightLoader(MODEL_DIR);
const manifest = loader.manifest;
const cfg = manifest.config;
const backend = createReferenceBackend();
const demoAssignment: ShardAssignment = {
  workerId: "demo-worker",
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
const demoExecutor = backend.createStageExecutor(manifest, demoAssignment);
await demoExecutor.load(loader);
console.log(
  `ok demo model loaded: ${manifest.modelId}, hidden ${cfg.hiddenSize}, ${cfg.numHiddenLayers} layers, vocab ${cfg.vocabSize}`,
);

function demoCache(): PagedKVCache {
  return new BlockPagedKVCache(16, 1024, cfg.numKeyValueHeads, cfg.headDim);
}

const tokenizer = new ByteLevelTokenizer({
  bosTokenId: cfg.bosTokenId,
  eosTokenId: cfg.eosTokenId,
  padTokenId: cfg.padTokenId,
});
const demoPrompt = tokenizer.encode("Hello world");
const demoGreedy: SamplingParams = { temperature: 0, topP: 1, topK: 0, maxTokens: 12 };

const base = await new AutoregressiveDecoder({
  executor: demoExecutor,
  cache: demoCache(),
  seqId: "demo-base",
  promptTokenIds: demoPrompt,
  sampling: demoGreedy,
  vocabSize: cfg.vocabSize,
  eosTokenId: cfg.eosTokenId,
}).generate();
console.log(`demo baseline (greedy, 12 tokens): [${base.tokens.join(", ")}] in ${base.forwards} forwards`);

{
  const spec = await new SpeculativeDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-spec",
    promptTokenIds: demoPrompt,
    sampling: demoGreedy,
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
    draftLength: 4,
  }).generate();
  assert.deepEqual(spec.tokens, base.tokens, "speculative (K=4) tokens must equal baseline token-for-token");
  assert.equal(spec.finishReason, base.finishReason, "speculative finish reason matches baseline");
  assert.ok(
    spec.forwards <= base.forwards,
    `speculative forwards (${spec.forwards}) must not exceed baseline (${base.forwards})`,
  );
  console.log(
    `ok exactness (demo): speculative K=4 == baseline, ${spec.forwards} forwards vs ${base.forwards} baseline, ` +
      `acceptance ${spec.draftAccepted}/${spec.draftProposed} = ${spec.acceptanceRate.toFixed(3)}`,
  );

  const jac = await new JacobiDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-jac",
    promptTokenIds: demoPrompt,
    sampling: demoGreedy,
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
    blockSize: 4,
  }).generate();
  assert.deepEqual(jac.tokens, base.tokens, "Jacobi (K=4) tokens must equal baseline token-for-token");
  assert.equal(jac.finishReason, base.finishReason, "Jacobi finish reason matches baseline");
  console.log(
    `ok exactness (demo): Jacobi K=4 == baseline, ${jac.forwards} forwards, ${jac.iterations} iterations, ${jac.blocks} blocks`,
  );

  // Draft longer than remaining maxTokens on the real model.
  const specLong = await new SpeculativeDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-spec-long",
    promptTokenIds: demoPrompt,
    sampling: demoGreedy,
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
    draftLength: 8,
  }).generate(3);
  const base3 = await new AutoregressiveDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-base-3",
    promptTokenIds: demoPrompt,
    sampling: demoGreedy,
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
  }).generate(3);
  assert.deepEqual(specLong.tokens, base3.tokens, "demo: draftLength 8 > maxTokens 3, exact");
  console.log("ok edge case (demo): draftLength 8 > maxTokens 3, exact");
}

// --- Acceptance measurement (24 tokens, K=4): reported, not tuned. ---
// WARNING — read before quoting the numbers below: the demo model's greedy
// output collapses to a single repeated token (57, 57, 57, …) for this
// prompt. The n-gram drafter's repeat-last fallback therefore predicts
// almost every draft correctly, and the measured alpha is an artifact of
// that collapse, NOT evidence that n-gram drafting works on real text.
// A second prompt is measured below to check whether the collapse (and
// hence the high alpha) is prompt-specific. On a trained model with varied
// output, alpha would be far lower; the honest claim here is exactness +
// the cost-model arithmetic, not a speedup on this model.
{
  const measured = await new SpeculativeDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-measure",
    promptTokenIds: demoPrompt,
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 24 },
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
    draftLength: 4,
  }).generate();
  const alpha = measured.acceptanceRate;
  const modelTokensPerRound = expectedSpeculativeTokensPerRound(alpha, 4);
  const measuredTokensPerRound =
    measured.rounds > 0 ? measured.tokens.length / measured.rounds : 0;
  console.log(
    `MEASURED (demo-10m, prompt "Hello world", greedy, 24 tokens, K=4, n-gram drafter):\n` +
      `  tokens=[${measured.tokens.join(", ")}]\n` +
      `  acceptance alpha = ${measured.draftAccepted}/${measured.draftProposed} = ${alpha.toFixed(4)}\n` +
      `  rounds=${measured.rounds} forwards=${measured.forwards} (baseline would be ${measured.tokens.length} decode forwards + prefill)\n` +
      `  measured tokens/round = ${measuredTokensPerRound.toFixed(3)}; ` +
      `cost-model E[tokens/round | alpha] = ${modelTokensPerRound.toFixed(3)}\n` +
      `  expected forwards/token = ${expectedForwardsPerToken(alpha, 4).toFixed(4)}\n` +
      `  breakeven alpha: beta=0 -> ${breakevenAcceptance(4, 0)}, beta=0.5 -> ${(breakevenAcceptance(4, 0.5) as number).toFixed(4)}, ` +
      `beta=1 -> ${(breakevenAcceptance(4, 1) as number).toFixed(4)}\n` +
      `  expected speedup: beta=0 -> ${speculativeSpeedup(alpha, 4, 0).toFixed(3)}x, ` +
      `beta=0.5 -> ${speculativeSpeedup(alpha, 4, 0.5).toFixed(3)}x, ` +
      `beta=1 -> ${speculativeSpeedup(alpha, 4, 1).toFixed(3)}x`,
  );
  // Coherence checks on the measurement (these assert the *model*, not a win):
  assert.ok(alpha >= 0 && alpha <= 1, "measured alpha in [0, 1]");
  assert.ok(measured.draftProposed > 0, "measurement actually proposed drafts");
  assert.ok(
    measuredTokensPerRound >= 1 && measuredTokensPerRound <= 5,
    `tokens/round in [1, K+1], got ${measuredTokensPerRound}`,
  );
  // The independence model overestimates slightly when acceptance is
  // correlated / the last round is partial (small sample: 5-7 rounds), so
  // allow 25% slack rather than asserting equality.
  assert.ok(
    Math.abs(measuredTokensPerRound - modelTokensPerRound) / modelTokensPerRound < 0.25,
    `measured tokens/round (${measuredTokensPerRound.toFixed(3)}) within 25% of model (${modelTokensPerRound.toFixed(3)})`,
  );
  // The breakeven implication must hold: below the beta=1 breakeven the
  // beta=1 speedup is < 1; at/above it, >= 1.
  const be1 = breakevenAcceptance(4, 1) as number;
  if (alpha < be1) {
    assert.ok(speculativeSpeedup(alpha, 4, 1) < 1, "below breakeven at beta=1, speedup < 1");
    console.log(
      `  HONEST RESULT: alpha=${alpha.toFixed(4)} is BELOW the beta=1 breakeven (${be1.toFixed(4)}), ` +
        `so on this compute-bound CPU backend speculation LOSES wall-clock ` +
        `(${speculativeSpeedup(alpha, 4, 1).toFixed(3)}x). It still saves target forwards ` +
        `(${measured.forwards} vs ${measured.tokens.length + 1} baseline incl. prefill) — the win appears only when beta < 1 (bandwidth-bound hardware) or alpha rises (trained model). ` +
        `Note the alpha itself is inflated by the model's collapse to a constant token (see WARNING above).`,
    );
  } else {
    assert.ok(speculativeSpeedup(alpha, 4, 1) >= 1, "above breakeven at beta=1, speedup >= 1");
    console.log(`  RESULT: alpha=${alpha.toFixed(4)} is above the beta=1 breakeven (${be1.toFixed(4)}).`);
  }
  // Exactness must hold on the measurement run too.
  const base24 = await new AutoregressiveDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-base-24",
    promptTokenIds: demoPrompt,
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 24 },
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
  }).generate();
  assert.deepEqual(measured.tokens, base24.tokens, "measurement run tokens equal baseline");
  console.log("ok measurement run is exact vs baseline and coherent with the cost model");

  // Second prompt: same measurement, to see whether the constant-output
  // collapse (and the high alpha it causes) is prompt-specific.
  const prompt2 = tokenizer.encode("The quick brown fox jumps");
  const measured2 = await new SpeculativeDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-measure-2",
    promptTokenIds: prompt2,
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 24 },
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
    draftLength: 4,
  }).generate();
  const base2 = await new AutoregressiveDecoder({
    executor: demoExecutor,
    cache: demoCache(),
    seqId: "demo-base-2",
    promptTokenIds: prompt2,
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 24 },
    vocabSize: cfg.vocabSize,
    eosTokenId: cfg.eosTokenId,
  }).generate();
  assert.deepEqual(measured2.tokens, base2.tokens, "second-prompt measurement run tokens equal baseline");
  console.log(
    `MEASURED (demo-10m, prompt "The quick brown fox jumps", greedy, 24 tokens, K=4):\n` +
      `  baseline tokens=[${base2.tokens.join(", ")}]\n` +
      `  acceptance alpha = ${measured2.draftAccepted}/${measured2.draftProposed} = ${measured2.acceptanceRate.toFixed(4)}, ` +
      `rounds=${measured2.rounds} forwards=${measured2.forwards}`,
  );
}

console.log("ALL NOVEL-DECODING TESTS PASSED");
