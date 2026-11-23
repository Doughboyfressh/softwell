/**
 * Smoke test for the scheduler: BlockPagedKVCache + ContinuousBatchingScheduler.
 * Run with: node test/scheduler.test.ts  (exit 0 = pass)
 */
import assert from "node:assert/strict";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import { ContinuousBatchingScheduler } from "../src/scheduler/batch.ts";
import type {
  ExecuteRequest,
  ExecuteResponse,
  Sampler,
  ShardPlan,
} from "../src/contracts.ts";

// ---------------------------------------------------------------------------
// BlockPagedKVCache
// ---------------------------------------------------------------------------

{
  const cache = new BlockPagedKVCache(4, 4, 2, 3); // blockSize 4, 4 blocks, kvDim 6
  assert.equal(cache.totalBlocks(), 4);
  assert.equal(cache.usedBlocks(), 0);

  // 5 tokens need 2 blocks.
  const table = await cache.allocate("a", 0, 5);
  assert.equal(table.length, 2);
  assert.equal(cache.usedBlocks(), 2);

  // Append 5 tokens with distinct K/V, then read them back out of order.
  // (Values use halves so they are exact in float32.)
  for (let pos = 0; pos < 5; pos++) {
    const key = new Float32Array([pos, pos + 0.5, pos + 1, pos + 1.5, pos + 2, pos + 2.5]);
    const value = new Float32Array([100 + pos, 101 + pos, 102 + pos, 103 + pos, 104 + pos, 105 + pos]);
    await cache.append("a", 0, pos, key, value);
  }
  const { keys, values } = await cache.read("a", 0, [4, 0, 2]);
  assert.deepEqual([...keys.slice(0, 6)], [4, 4.5, 5, 5.5, 6, 6.5]);
  assert.deepEqual([...keys.slice(6, 12)], [0, 0.5, 1, 1.5, 2, 2.5]);
  assert.deepEqual([...keys.slice(12, 18)], [2, 2.5, 3, 3.5, 4, 4.5]);
  assert.deepEqual([...values.slice(0, 6)], [104, 105, 106, 107, 108, 109]);

  // A second layer for the same sequence allocates its own blocks.
  await cache.allocate("a", 1, 1);
  assert.equal(cache.usedBlocks(), 3);

  // Free releases everything.
  await cache.free("a");
  assert.equal(cache.usedBlocks(), 0);

  // Out of memory: 20 tokens need 5 blocks but only 4 exist.
  await assert.rejects(
    async () => cache.allocate("oom", 0, 20),
    /out of memory/,
  );
  // The partial allocation is still held (2 blocks were taken before OOM
  // on the 5th); free it and the pool is whole again.
  await cache.free("oom");
  assert.equal(cache.usedBlocks(), 0);

  console.log("BlockPagedKVCache: ok");
}

// ---------------------------------------------------------------------------
// ContinuousBatchingScheduler with a fake 2-stage mesh
// ---------------------------------------------------------------------------

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

/** Fake transport: stage 0 embeds (echoes token + 0.01); last stage returns
 *  deterministic logits whose argmax is token 3, with EOS (9) very unlikely. */
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
    // Last stage: logits. Token 3 wins, EOS (9) loses badly.
    const data: number[] = [];
    for (let r = 0; r < rows; r++) {
      for (let v = 0; v < VOCAB; v++) {
        data.push(v === 3 ? 10 : v === EOS ? -100 : -Math.abs(v - 3));
      }
    }
    return { output: { data, shape: [rows, VOCAB] }, isLogits: true, tokensProcessed: rows };
  };
}

{
  const calls: { workerId: string; req: Omit<ExecuteRequest, "workerId"> }[] = [];
  const freed: string[] = [];
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: makeFakeTransport(calls),
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
    maxBatchTokens: 100,
    maxSequences: 4,
    freeSequence: async (seqId) => {
      freed.push(seqId);
    },
  });

  const id = scheduler.submit({
    id: "seq-1",
    prompt: "hello",
    tokenIds: [1, 2, 3],
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 3 },
    role: "mixed",
  });
  assert.equal(id, "seq-1");
  assert.equal(scheduler.pendingCount(), 1);
  assert.equal(scheduler.runningCount(), 0);

  const allTokens: { tokenId: number; text: string; finishReason: string | null }[] = [];
  for (let i = 0; i < 3; i++) {
    const tokens = await scheduler.step();
    allTokens.push(...tokens);
  }

  // 3 steps -> 3 generated tokens, all the fake argmax (3), sequence done.
  assert.equal(allTokens.length, 3);
  assert.deepEqual(
    allTokens.map((t) => t.tokenId),
    [3, 3, 3],
  );
  assert.equal(allTokens[2]?.finishReason, "length");
  assert.deepEqual(
    allTokens.map((t) => t.text),
    ["<3>", "<3>", "<3>"],
  );

  const seq = scheduler.getSequence("seq-1");
  assert.equal(seq?.status, "done");
  assert.equal(seq?.finishReason, "length");
  assert.deepEqual(seq?.generatedTokenIds, [3, 3, 3]);
  assert.equal(scheduler.pendingCount(), 0);
  assert.equal(scheduler.runningCount(), 0);
  assert.deepEqual(freed, ["seq-1"]);

  // Prefill went to stage 0 with the full prompt; decode fed the last token
  // at absolute position 3 (prompt length 3).
  const prefillStage0 = calls.find((c) => c.req.kind === "prefill" && c.req.stage === 0);
  assert.deepEqual(prefillStage0?.req.tokenIds, [1, 2, 3]);
  assert.deepEqual(prefillStage0?.req.positions, [0, 1, 2]);
  assert.equal(prefillStage0?.workerId, "worker-0");
  const decodeStage0 = calls.find((c) => c.req.kind === "decode" && c.req.stage === 0);
  assert.deepEqual(decodeStage0?.req.tokenIds, [3]);
  assert.deepEqual(decodeStage0?.req.positions, [3]);
  assert.equal(decodeStage0?.workerId, "worker-0");
  // Both stages were hit, in order, for every forward.
  assert.ok(calls.some((c) => c.req.stage === 1 && c.req.kind === "prefill"));
  assert.ok(calls.some((c) => c.req.stage === 1 && c.req.kind === "decode"));

  // Routing info is recorded on the last step.
  const info = scheduler.lastStepInfo();
  assert.ok(info !== null);
  assert.ok(info?.routing.includes("worker-0"));
  assert.ok(info?.routing.includes("worker-1"));

  console.log("ContinuousBatchingScheduler 3-token generation: ok");
}

// ---------------------------------------------------------------------------
// Worker execute failure -> sequence 'error', scheduler survives
// ---------------------------------------------------------------------------

{
  const scheduler = new ContinuousBatchingScheduler({
    plan: fakePlan(),
    transport: async () => {
      throw new Error("worker-0 unreachable");
    },
    sampler: greedySampler,
    decode: (ids) => ids.map((id) => `<${id}>`).join(""),
    eosTokenId: EOS,
  });
  scheduler.submit({
    id: "seq-err",
    prompt: "boom",
    tokenIds: [1, 2],
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 2 },
    role: "mixed",
  });
  const tokens = await scheduler.step();
  assert.deepEqual(tokens, []);
  const seq = scheduler.getSequence("seq-err");
  assert.equal(seq?.status, "error");
  assert.match(seq?.error ?? "", /worker-0 unreachable/);
  // Scheduler still works afterwards.
  assert.equal(scheduler.pendingCount(), 0);
  assert.equal(scheduler.runningCount(), 0);

  console.log("ContinuousBatchingScheduler worker-failure handling: ok");
}

// ---------------------------------------------------------------------------
// Chunked prefill: a long prompt goes out as 16-row chunks, not one execute
// ---------------------------------------------------------------------------

{
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

  // 40-token prompt + a 5-token prompt admitted in the same step:
  // 45 rows = 16 + 16 + 13. One unchunked execute would carry all 45
  // rows; the mesh timeout caps a single execute, so the scheduler must
  // split it (the ~1k-token API prompt is the same shape, 63 chunks).
  // The short sequence's last row is in chunk 1 and the long one's in
  // chunk 3 — runStageChain must return each sequence's own last row,
  // not just the last chunk's rows.
  const prompt40 = Array.from({ length: 40 }, (_, i) => (i % 9) + 1);
  scheduler.submit({
    id: "seq-short",
    prompt: "short",
    tokenIds: [1, 2, 3, 4, 5],
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 3 },
    role: "mixed",
  });
  scheduler.submit({
    id: "seq-long",
    prompt: "long",
    tokenIds: prompt40,
    sampling: { temperature: 0, topP: 1, topK: 0, maxTokens: 3 },
    role: "mixed",
  });
  const bySeq = new Map<string, number[]>();
  for (let i = 0; i < 3; i++) {
    const tokens = await scheduler.step();
    for (const t of tokens) {
      const list = bySeq.get(t.seqId) ?? [];
      list.push(t.tokenId);
      bySeq.set(t.seqId, list);
    }
  }
  assert.deepEqual(
    bySeq.get("seq-short"),
    [3, 3, 3],
    "short sequence (last row in chunk 1) generates the same tokens",
  );
  assert.deepEqual(
    bySeq.get("seq-long"),
    [3, 3, 3],
    "long sequence (last row in chunk 3) generates the same tokens",
  );
  assert.equal(scheduler.getSequence("seq-short")?.status, "done");
  assert.equal(scheduler.getSequence("seq-long")?.status, "done");

  const prefill0 = calls.filter((c) => c.req.kind === "prefill" && c.req.stage === 0);
  assert.equal(prefill0.length, 3, "45 rows go out as 3 prefill chunks");
  assert.deepEqual(
    prefill0.map((c) => c.req.positions.length),
    [16, 16, 13],
    "chunk sizes are 16, 16, 13",
  );
  assert.deepEqual(
    prefill0.flatMap((c) => c.req.positions ?? []),
    [...[0, 1, 2, 3, 4], ...Array.from({ length: 40 }, (_, i) => i)],
    "chunks cover the short prompt then the long prompt, contiguously, in order",
  );
  assert.deepEqual(
    prefill0[2]?.req.tokenIds,
    prompt40.slice(27),
    "the last chunk carries the long prompt's tail — its last row is what gets sampled for it",
  );
  const prefill1 = calls.filter((c) => c.req.kind === "prefill" && c.req.stage === 1);
  assert.equal(prefill1.length, 3, "every chunk runs through every stage");

  console.log("ContinuousBatchingScheduler chunked prefill: ok");
}

console.log("All scheduler tests passed.");
