/**
 * Scheduler draft-model tests — run with: node test/scheduler-draft-model.test.ts
 *
 * The scheduler's speculative path with a per-sequence model drafter:
 * a fake StageExecutor whose argmax is always token 3 backs a
 * ModelDrafter, and the fake mesh transport's argmax is also always
 * token 3, so every draft is accepted and the scheduler must produce
 * exactly the autoregressive token stream [3×8] in 2 verification
 * forwards — with the acceptance counters showing real counts from
 * the model drafter (not the n-gram path). A second case proves a
 * broken draft factory (vocab mismatch) fails the sequence loudly
 * with the mismatch in the error, instead of silently falling back
 * to n-gram drafting.
 *
 * The fake executor ignores the KV cache (it reads the token id out
 * of the hidden state), so this file tests scheduler plumbing, not
 * cache correctness — cache correctness with a real draft model is
 * covered by test/model-drafter.test.ts.
 */

import assert from "node:assert/strict";
import { ContinuousBatchingScheduler } from "../src/scheduler/batch.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import { ModelDrafter } from "../src/engine/novel-decoding.ts";
import type {
  ExecuteRequest,
  ExecuteResponse,
  ForwardContext,
  PagedKVCache,
  Sampler,
  ShardAssignment,
  StageExecutor,
  Tensor,
  WeightLoader,
} from "../src/contracts.ts";

const VOCAB = 10;
const EOS = 9;
const HIDDEN = 8;

function fakePlan() {
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

/**
 * Fake draft executor: every row's logits have argmax 3, whatever the
 * input token — so the ModelDrafter always proposes 3s, which the fake
 * target (argmax 3) always accepts. Ignores the cache, like the fake
 * executor in test/novel-decoding.test.ts.
 */
class FakeDraftExecutor implements StageExecutor {
  readonly assignment: ShardAssignment;
  forwards = 0;
  constructor(assignment: ShardAssignment) {
    this.assignment = assignment;
  }
  async load(_loader: WeightLoader): Promise<void> {}
  embed(tokenIds: number[]): Tensor {
    const data = new Float32Array(tokenIds.length * 4);
    for (let r = 0; r < tokenIds.length; r++) data[r * 4] = tokenIds[r] as number;
    return { data, shape: [tokenIds.length, 4] };
  }
  async forward(input: Tensor, _ctx: ForwardContext): Promise<Tensor> {
    this.forwards += 1;
    const rows = input.shape[0];
    const logits = new Float32Array(rows * VOCAB);
    for (let r = 0; r < rows; r++) {
      for (let v = 0; v < VOCAB; v++) {
        logits[r * VOCAB + v] = v === 3 ? 10 : v === EOS ? -100 : -Math.abs(v - 3);
      }
    }
    return { data: logits, shape: [rows, VOCAB] };
  }
  memoryBytes(): number {
    return 0;
  }
}

const fakeDraftAssignment: ShardAssignment = {
  workerId: "draft",
  stage: 0,
  layerStart: 0,
  layerEnd: 1,
  hasEmbedding: true,
  hasLmHead: true,
  tensorParallelRank: 0,
  tensorParallelSize: 1,
  experts: [],
  estimatedMemoryBytes: 0,
};

function draftCache(): PagedKVCache {
  return new BlockPagedKVCache(16, 64, 2, 8);
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

function submitEight(scheduler: ContinuousBatchingScheduler): void {
  scheduler.submit({
    id: "seq-1",
    prompt: "hello",
    tokenIds: [1, 2, 3],
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 8 },
    role: "mixed",
  });
}

{
  // Speculative with a model drafter: [3×8] in 2 verification forwards,
  // acceptance counters real (all drafts accepted), and the drafter's
  // own forward counter proves the model drafter ran.
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const draftExecutor = new FakeDraftExecutor(fakeDraftAssignment);
  const created: ModelDrafter[] = [];
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
    draftDrafterFactory: (seqId: string) => {
      const d = new ModelDrafter({
        executor: draftExecutor,
        cache: draftCache(),
        seqId: `${seqId}:draft`,
        vocabSize: VOCAB,
        eosTokenId: EOS,
      });
      created.push(d);
      return d;
    },
  });
  submitEight(scheduler);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "model drafter: generates [3×8], token-identical to baseline");
  assert.equal(
    countDecodeForwards(calls),
    2,
    `model drafter: 2 verification forwards for 8 tokens, got ${countDecodeForwards(calls)}`,
  );
  assert.equal(created.length, 1, "one per-sequence drafter was built for the sequence");
  const drafter = created[0] as ModelDrafter;
  assert.ok(drafter.forwards > 0, `model drafter ran ${drafter.forwards} forwards`);
  const info = scheduler.lastStepInfo();
  assert.ok(info !== null, "lastStepInfo is present");
  // The last step ran a speculative round (the second verification
  // forward); its counters are real counts from the model drafter.
  assert.ok((info as { specRounds: number }).specRounds >= 1, "step info records speculative rounds");
  assert.ok((info as { specProposed: number }).specProposed > 0, "step info records proposed drafts");
  assert.equal(
    (info as { specAccepted: number }).specAccepted,
    (info as { specProposed: number }).specProposed,
    "fake target accepts every draft: accepted == proposed",
  );
  // The finished sequence's draft KV was freed (the drafter's free()
  // ran at finish): drafting again after free throws.
  await assert.rejects(
    () => drafter.draft([1, 2, 3], 1),
    /draft\(\) called after free/,
    "sequence finish frees the per-sequence drafter",
  );
  console.log(
    `ok scheduler + model drafter: [3×8] in 2 verification forwards, ` +
      `draft model ran ${drafter.forwards} forwards, step acceptance ` +
      `${(info as { specAccepted: number }).specAccepted}/${(info as { specProposed: number }).specProposed}`,
  );
}

{
  // A draft factory that throws (e.g. vocab mismatch vs the target)
  // fails the sequence loudly with the mismatch in the error — it
  // must NOT silently fall back to the n-gram drafter.
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
    draftDrafterFactory: () => {
      throw new Error(
        "draft model demo/draft has vocabSize 259 but the target model has vocabSize 49152 — draft and target must share a tokenizer/vocabulary",
      );
    },
  });
  submitEight(scheduler);
  await runToDone(scheduler, 12);
  const seq = scheduler.getSequence("seq-1");
  assert.equal(seq?.status, "error", "broken draft factory: sequence errors, no silent fallback");
  assert.match(seq?.error ?? "", /vocabSize 259.*vocabSize 49152/, "error names the vocab mismatch");
  console.log(`ok scheduler: broken draft factory fails the sequence loudly: "${seq?.error}"`);
}

{
  // No factory (default): the n-gram path is unchanged — the
  // scheduler-novel test covers it in full; here just check the
  // factory-less speculative run still produces [3×8].
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
  submitEight(scheduler);
  const tokens = await runToDone(scheduler, 12);
  assert.deepEqual(tokens, [3, 3, 3, 3, 3, 3, 3, 3], "no factory: n-gram default still generates [3×8]");
  console.log("ok scheduler: no draft factory keeps the n-gram default");
}

console.log("ALL SCHEDULER DRAFT-MODEL TESTS PASSED");
