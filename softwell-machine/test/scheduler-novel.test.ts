/**
 * Scheduler novel-decoding tests — run with: node test/scheduler-novel.test.ts
 *
 * The fake transport returns logits whose argmax is always token 3, so the
 * n-gram drafter predicts the stream perfectly (acceptance 1.0) and both
 * novel modes must produce exactly the autoregressive token stream [3×8]
 * in fewer stage-chain forwards. The test asserts token identity AND the
 * forward saving — the two properties Theorems S1/S2 actually promise.
 * Wall-clock is not asserted here (on this CPU backend the novel modes
 * are expected to lose wall-clock; see docs/research/03-decoding.md §5
 * and docs/BENCHMARKS.md).
 */

import assert from "node:assert/strict";
import { ContinuousBatchingScheduler } from "../src/scheduler/batch.ts";
import type {
  ExecuteRequest,
  ExecuteResponse,
  Sampler,
  ShardPlan,
} from "../src/contracts.ts";

const VOCAB = 10;
const EOS = 9;
const HIDDEN = 8;

function fakePlan(): ShardPlan {
  return {
    modelId: "test/fake-2-stage",
    totalParams: 1000,
    totalWeightBytes: 4000,
    pipelineStages: 2,
    tensorParallelSize: 1,
    expertParallelSize: 1,
    assignments: [
      {
        workerId: "worker-0",
        stage: 0,
        layerStart: 0,
        layerEnd: 1,
        hasEmbedding: true,
        hasLmHead: false,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 1000,
      },
      {
        workerId: "worker-1",
        stage: 1,
        layerStart: 1,
        layerEnd: 2,
        hasEmbedding: false,
        hasLmHead: true,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 1000,
      },
    ],
    estimatedTokensPerSec: 10,
    estimatedTtftMs: 100,
    bottleneck: "test",
    createdAt: new Date().toISOString(),
  };
}

const greedySampler: Sampler = (logits) => {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if ((logits[i] as number) > (logits[best] as number)) best = i;
  }
  return best;
};

/** Fake transport: stage 0 embeds; last stage returns logits whose argmax
 *  is always token 3 (EOS 9 loses badly). Records every call. */
function makeFakeTransport(calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[]) {
  return async (
    workerId: string,
    req: Omit<ExecuteRequest, "workerId">,
  ): Promise<ExecuteResponse> => {
    calls.push({ workerId, req });
    const rows = req.seqIds.length;
    if (req.stage === 0) {
      const data: number[] = [];
      for (const tokenId of req.tokenIds ?? []) {
        for (let h = 0; h < HIDDEN; h++) data.push(tokenId + 0.01);
      }
      return { output: { data, shape: [rows, HIDDEN] }, isLogits: false, tokensProcessed: rows };
    }
    const data: number[] = [];
    for (let r = 0; r < rows; r++) {
      for (let v = 0; v < VOCAB; v++) {
        data.push(v === 3 ? 10 : v === EOS ? -100 : -Math.abs(v - 3));
      }
    }
    return { output: { data, shape: [rows, VOCAB] }, isLogits: true, tokensProcessed: rows };
  };
}

function countDecodeForwards(calls: { req: Omit<ExecuteRequest, "workerId"> }[]): number {
  return calls.filter((c) => c.req.kind === "decode" && c.req.stage === 0).length;
}

async function runToDone(
  scheduler: ContinuousBatchingScheduler,
  maxSteps: number,
): Promise<number[]> {
  const tokens: number[] = [];
  for (let i = 0; i < maxSteps; i++) {
    const produced = await scheduler.step();
    tokens.push(...produced.map((t) => t.tokenId));
    const seq = scheduler.getSequence("seq-1");
    if (seq?.status === "done" || seq?.status === "error") break;
  }
  return tokens;
}

function submitEight(scheduler: ContinuousBatchingScheduler, temperature: number): void {
  scheduler.submit({
    id: "seq-1",
    prompt: "hello",
    tokenIds: [1, 2, 3],
    sampling: { temperature, topP: 1, topK: 0, maxTokens: 8 },
    role: "mixed",
  });
}

{
  // Baseline: 1 prefill + 7 decode forwards for 8 tokens.
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: makeFakeTransport(calls),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
  });
  submitEight(scheduler, 0);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "baseline generates [3×8]");
  assert.equal(countDecodeForwards(calls), 7, "baseline: 7 decode forwards for 8 tokens");
  console.log("ok baseline: [3×8] in 7 decode forwards");
}

{
  // Speculative, K=4, acceptance 1.0: prefill + 2 verification forwards.
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: makeFakeTransport(calls),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    decoding: "speculative",
    draftLength: 4,
  });
  submitEight(scheduler, 0);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "speculative generates [3×8], token-identical to baseline");
  assert.equal(countDecodeForwards(calls), 2, `speculative: 2 verification forwards for 8 tokens, got ${countDecodeForwards(calls)}`);
  const info = scheduler.lastStepInfo();
  assert.ok(info?.routing.includes("decoding mode: speculative (K=4)"), "routing info names the decoding mode");
  console.log("ok speculative: [3×8] in 2 verification forwards (7 -> 2)");
}

{
  // Jacobi, K=4: prefill + 2 block forwards (1 iteration each).
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: makeFakeTransport(calls),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    decoding: "jacobi",
    draftLength: 4,
  });
  submitEight(scheduler, 0);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "jacobi generates [3×8], token-identical to baseline");
  assert.equal(countDecodeForwards(calls), 2, `jacobi: 2 block forwards for 8 tokens, got ${countDecodeForwards(calls)}`);
  console.log("ok jacobi: [3×8] in 2 block forwards (7 -> 2)");
}

{
  // Sampled sequences fall back to autoregressive even in speculative mode.
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: makeFakeTransport(calls),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    decoding: "speculative",
    draftLength: 4,
  });
  submitEight(scheduler, 0.7);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "sampled sequence still generates [3×8]");
  assert.equal(countDecodeForwards(calls), 7, "sampled sequence falls back to 7 autoregressive decode forwards");
  console.log("ok sampled fallback: temperature > 0 decodes autoregressively under --decoding speculative");
}

{
  // Invalid draftLength is rejected at construction.
  assert.throws(
    () =>
      new ContinuousBatchingScheduler({
        plan: fakePlan(),
        transport: makeFakeTransport([]),
        sampler: greedySampler,
        decode: (ids) => ids.join(","),
        eosTokenId: EOS,
        decoding: "jacobi",
        draftLength: 0,
      }),
    /draftLength must be a positive integer/,
  );
  console.log("ok invalid draftLength rejected");
}

console.log("All scheduler novel-decoding tests passed.");
