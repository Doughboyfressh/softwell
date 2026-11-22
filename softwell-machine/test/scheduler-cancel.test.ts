/**
 * Cancel tests for ContinuousBatchingScheduler (src/scheduler/batch.ts).
 * Run with: node test/scheduler-cancel.test.ts  (exit 0 = pass)
 *
 * The coordinator already implemented Scheduler.cancel() with
 * step-boundary semantics plus guards in step() and appendToken(); these
 * tests pin that behaviour down with a fake transport (canned logits,
 * argmax sampler) — no real engine.
 *
 *  (1) cancel a waiting sequence
 *  (2) cancel a running (decoding) sequence mid-generation
 *  (3) cancel an unknown id
 *  (4) cancel an already done/error sequence
 *  (5) cancel during an in-flight prefill (deferred transport)
 */
import assert from "node:assert/strict";
import { ContinuousBatchingScheduler } from "../src/scheduler/batch.ts";
import type {
  ExecuteRequest,
  ExecuteResponse,
  GenerateRequest,
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

/** Canned stage outputs: stage 0 echoes tokens, last stage returns logits
 *  whose argmax is token 3 (EOS 9 is very unlikely). */
function cannedResponse(req: Omit<ExecuteRequest, "workerId">): ExecuteResponse {
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
}

function makeScheduler(freed: string[]) {
  return new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: async (_workerId, req) => cannedResponse(req),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    freeSequence: async (seqId) => {
      freed.push(seqId);
    },
  });
}

function makeRequest(id: string, maxTokens = 3): GenerateRequest {
  return {
    id,
    prompt: "hello",
    tokenIds: [1, 2, 3],
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens },
    role: "mixed",
  };
}

// ---------------------------------------------------------------------------
// (1) Cancel a waiting sequence
// ---------------------------------------------------------------------------

{
  const freed: string[] = [];
  const scheduler = makeScheduler(freed);
  scheduler.submit(makeRequest("seq-wait-a"));
  scheduler.submit(makeRequest("seq-wait-b"));
  assert.equal(scheduler.pendingCount(), 2);

  assert.equal(scheduler.cancel("seq-wait-a"), true);
  assert.equal(scheduler.pendingCount(), 1);
  const cancelled = scheduler.getSequence("seq-wait-a");
  assert.equal(cancelled?.status, "done");
  assert.equal(cancelled?.finishReason, "cancelled");
  assert.deepEqual(cancelled?.generatedTokenIds, []);
  assert.deepEqual(freed, ["seq-wait-a"]);

  // A subsequent step() produces no token for the cancelled sequence —
  // and the surviving sequence still generates normally.
  const produced = await scheduler.step();
  assert.ok(produced.length > 0, "expected the surviving sequence to produce a token");
  assert.ok(
    produced.every((t) => t.seqId === "seq-wait-b"),
    `expected only seq-wait-b tokens, got ${JSON.stringify(produced)}`,
  );
  assert.equal(scheduler.getSequence("seq-wait-a")?.generatedTokenIds.length, 0);

  console.log("cancel waiting sequence: ok");
}

// ---------------------------------------------------------------------------
// (2) Cancel a running (decoding) sequence mid-generation
// ---------------------------------------------------------------------------

{
  const freed: string[] = [];
  const scheduler = makeScheduler(freed);
  scheduler.submit(makeRequest("seq-run", 5));

  const first = await scheduler.step();
  assert.deepEqual(
    first.map((t) => t.seqId),
    ["seq-run"],
  );
  assert.equal(scheduler.getSequence("seq-run")?.status, "decoding");
  assert.deepEqual(scheduler.getSequence("seq-run")?.generatedTokenIds, [3]);

  assert.equal(scheduler.cancel("seq-run"), true);
  const after = scheduler.getSequence("seq-run");
  assert.equal(after?.status, "done");
  assert.equal(after?.finishReason, "cancelled");
  assert.deepEqual(after?.generatedTokenIds, [3]);
  assert.deepEqual(freed, ["seq-run"]);

  // No further tokens are appended after the cancel, across two more steps.
  const second = await scheduler.step();
  assert.ok(
    second.every((t) => t.seqId !== "seq-run"),
    `cancelled sequence produced a token: ${JSON.stringify(second)}`,
  );
  const third = await scheduler.step();
  assert.deepEqual(third, []);
  assert.deepEqual(scheduler.getSequence("seq-run")?.generatedTokenIds, [3]);

  console.log("cancel running sequence mid-generation: ok");
}

// ---------------------------------------------------------------------------
// (3) Cancel an unknown id
// ---------------------------------------------------------------------------

{
  const freed: string[] = [];
  const scheduler = makeScheduler(freed);
  scheduler.submit(makeRequest("seq-known"));
  assert.equal(scheduler.cancel("seq-nope"), false);
  assert.equal(scheduler.cancel(""), false);
  // The known sequence is untouched.
  assert.equal(scheduler.pendingCount(), 1);
  assert.equal(scheduler.getSequence("seq-known")?.status, "waiting");
  assert.deepEqual(freed, []);

  console.log("cancel unknown id: ok");
}

// ---------------------------------------------------------------------------
// (4) Cancel an already done / error sequence
// ---------------------------------------------------------------------------

{
  const freed: string[] = [];
  const scheduler = makeScheduler(freed);

  // Done via maxTokens: one step generates the single allowed token.
  scheduler.submit(makeRequest("seq-done", 1));
  const produced = await scheduler.step();
  assert.equal(produced.length, 1);
  assert.equal(produced[0]?.finishReason, "length");
  assert.equal(scheduler.getSequence("seq-done")?.status, "done");
  assert.equal(scheduler.cancel("seq-done"), false);
  assert.equal(scheduler.getSequence("seq-done")?.finishReason, "length");

  // Error via a failing transport.
  const errorScheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: async () => {
      throw new Error("worker-0 unreachable");
    },
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
  });
  errorScheduler.submit(makeRequest("seq-err", 2));
  assert.deepEqual(await errorScheduler.step(), []);
  assert.equal(errorScheduler.getSequence("seq-err")?.status, "error");
  assert.equal(errorScheduler.cancel("seq-err"), false);
  assert.equal(errorScheduler.getSequence("seq-err")?.status, "error");

  // Cancelling the same sequence twice: the second call is false.
  assert.equal(scheduler.cancel("seq-done"), false);

  console.log("cancel done/error sequence: ok");
}

// ---------------------------------------------------------------------------
// (5) Cancel during an in-flight prefill
// ---------------------------------------------------------------------------

{
  const freed: string[] = [];
  let resolveGate!: () => void;
  const gate = new Promise<void>((resolve) => {
    resolveGate = resolve;
  });
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    // The transport waits on a deferred promise: the prefill forward is
    // in flight until the test resolves it.
    transport: async (_workerId, req) => {
      await gate;
      return cannedResponse(req);
    },
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    freeSequence: async (seqId) => {
      freed.push(seqId);
    },
  });
  scheduler.submit(makeRequest("seq-prefill", 3));

  // Start the step but do not await it: the prefill forward is now
  // awaiting the gate, and the sequence is in "prefill" status.
  const stepPromise = scheduler.step();
  assert.equal(scheduler.getSequence("seq-prefill")?.status, "prefill");

  // Cancel while the forward is in flight.
  assert.equal(scheduler.cancel("seq-prefill"), true);
  assert.equal(scheduler.getSequence("seq-prefill")?.status, "done");
  assert.equal(scheduler.getSequence("seq-prefill")?.finishReason, "cancelled");

  // Let the forward finish; its logits must be discarded.
  resolveGate();
  const produced = await stepPromise;
  assert.ok(
    produced.every((t) => t.seqId !== "seq-prefill"),
    `cancelled prefill produced a token: ${JSON.stringify(produced)}`,
  );
  const seq = scheduler.getSequence("seq-prefill");
  assert.equal(seq?.status, "done");
  assert.equal(seq?.finishReason, "cancelled");
  assert.deepEqual(seq?.generatedTokenIds, []);
  assert.deepEqual(freed, ["seq-prefill"]);

  // A later step produces nothing for it either.
  assert.deepEqual(await scheduler.step(), []);

  console.log("cancel during in-flight prefill: ok");
}

console.log("All scheduler cancel tests passed.");
