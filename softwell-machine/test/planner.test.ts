/**
 * Smoke test for the sharding planner (src/planner/cost.ts, planner.ts).
 * Builds a fake ~13M-param Llama-style manifest and fake workers, runs
 * plan(), and asserts the invariants that matter: contiguous layer
 * coverage, no stage over its worker's memory, expert distribution for
 * MoE. Run with: node test/planner.test.ts (exit 0 = pass).
 */
import assert from "node:assert/strict";
import { describeHardwareMath, decodeTokensPerSec } from "../src/planner/cost.ts";
import { plan, replan } from "../src/planner/planner.ts";
import type { ModelConfig, ModelManifest, TensorMeta, WorkerInfo } from "../src/contracts.ts";

const GB = 1024 ** 3;

function makeConfig(moe: boolean): ModelConfig {
  return {
    architectures: moe ? ["MixtralForCausalLM"] : ["LlamaForCausalLM"],
    vocabSize: 4096,
    hiddenSize: 512,
    intermediateSize: 1024,
    numHiddenLayers: 4,
    numAttentionHeads: 4,
    numKeyValueHeads: 2,
    headDim: 128,
    maxPositionEmbeddings: 4096,
    rmsNormEps: 1e-5,
    ropeTheta: 10000,
    tieWordEmbeddings: false,
    ...(moe ? { numExperts: 2, numExpertsPerToken: 1 } : {}),
    bosTokenId: 1,
    eosTokenId: 2,
    padTokenId: 0,
  };
}

function makeManifest(moe: boolean): ModelManifest {
  const config = makeConfig(moe);
  const tensors: Record<string, TensorMeta> = {};
  let offset = 0;
  const add = (name: string, shape: number[]) => {
    const elems = shape.reduce((a, b) => a * b, 1);
    const bytes = elems * 2; // bfloat16
    tensors[name] = { dtype: "bfloat16", shape, dataOffsets: [offset, offset + bytes] };
    offset += bytes;
  };
  const H = 512;
  const I = 1024;
  add("model.embed_tokens.weight", [4096, H]);
  for (let i = 0; i < 4; i++) {
    const p = `model.layers.${i}`;
    add(`${p}.input_layernorm.weight`, [H]);
    add(`${p}.self_attn.q_proj.weight`, [512, H]);
    add(`${p}.self_attn.k_proj.weight`, [256, H]);
    add(`${p}.self_attn.v_proj.weight`, [256, H]);
    add(`${p}.self_attn.o_proj.weight`, [H, 512]);
    add(`${p}.post_attention_layernorm.weight`, [H]);
    if (moe) {
      for (let e = 0; e < 2; e++) {
        add(`${p}.mlp.experts.${e}.gate_proj.weight`, [I, H]);
        add(`${p}.mlp.experts.${e}.up_proj.weight`, [I, H]);
        add(`${p}.mlp.experts.${e}.down_proj.weight`, [H, I]);
      }
      add(`${p}.mlp.router.weight`, [2, H]);
    } else {
      add(`${p}.mlp.gate_proj.weight`, [I, H]);
      add(`${p}.mlp.up_proj.weight`, [I, H]);
      add(`${p}.mlp.down_proj.weight`, [H, I]);
    }
  }
  add("model.norm.weight", [H]);
  add("lm_head.weight", [4096, H]);
  const totalParams = Object.values(tensors).reduce(
    (a, t) => a + t.shape.reduce((x, y) => x * y, 1),
    0,
  );
  return {
    format: "safetensors",
    modelId: moe ? "fake/moe-20m" : "fake/llama-13m",
    config,
    metadata: {},
    tensors,
    files: [{ filename: "model.safetensors", sizeBytes: offset }],
    totalBytes: offset,
    totalParams,
  };
}

function makeWorker(
  workerId: string,
  freeMemoryBytes: number,
  bandwidthBytesPerSec: number,
  flopsPerSec: number,
): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 9000,
    device: "cuda",
    backend: "softwell-ts",
    memoryBytes: freeMemoryBytes,
    freeMemoryBytes,
    flopsPerSec,
    bandwidthBytesPerSec,
    roles: ["mixed"],
    maxBatchTokens: 4096,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

/** Assert assignments cover every layer exactly once, contiguously. */
function assertContiguousCoverage(plan_: ReturnType<typeof plan>, numLayers: number): void {
  const sorted = [...plan_.assignments].sort((a, b) => a.stage - b.stage);
  assert.equal(sorted.length, plan_.pipelineStages, "one assignment per stage");
  assert.equal(sorted[0].layerStart, 0, "first stage starts at layer 0");
  for (let i = 0; i < sorted.length; i++) {
    assert.equal(sorted[i].stage, i, `stage index ${i}`);
    assert.ok(sorted[i].layerEnd > sorted[i].layerStart, "non-empty stage");
    if (i + 1 < sorted.length) {
      assert.equal(sorted[i + 1].layerStart, sorted[i].layerEnd, "contiguous ranges");
    }
  }
  assert.equal(sorted[sorted.length - 1].layerEnd, numLayers, "last stage ends at numLayers");
  assert.ok(sorted[0].hasEmbedding, "stage 0 has the embedding");
  assert.ok(sorted[sorted.length - 1].hasLmHead, "last stage has the lm_head");
}

function assertMemoryFits(plan_: ReturnType<typeof plan>, workers: WorkerInfo[]): void {
  const byId = new Map(workers.map((w) => [w.workerId, w]));
  for (const a of plan_.assignments) {
    const w = byId.get(a.workerId);
    assert.ok(w, `assignment references known worker ${a.workerId}`);
    assert.ok(
      a.estimatedMemoryBytes <= w!.freeMemoryBytes,
      `stage ${a.stage} needs ${a.estimatedMemoryBytes} bytes but worker ${a.workerId} ` +
        `has ${w!.freeMemoryBytes} free`,
    );
  }
}

// --- 1. Dense model, 2 workers with different memory/bandwidth -----------
const dense = makeManifest(false);
console.log(`dense manifest: ${dense.totalParams.toLocaleString()} params, ${(dense.totalBytes / 1e6).toFixed(1)} MB`);
const wA = makeWorker("worker-a", 8 * GB, 3e12, 1e15);
const wB = makeWorker("worker-b", 4 * GB, 1.5e12, 5e14);
const p2 = plan({ manifest: dense, workers: [wA, wB] });
assert.equal(p2.feasible, true, "2-worker plan is feasible");
assert.equal(p2.pipelineStages, 2, "2 stages for 2 workers");
assertContiguousCoverage(p2, 4);
assertMemoryFits(p2, [wA, wB]);
assert.ok(p2.estimatedTokensPerSec > 0, "positive tok/s estimate");
assert.ok(p2.estimatedTtftMs > 0, "positive TTFT estimate");
console.log("2-worker plan:");
for (const a of [...p2.assignments].sort((x, y) => x.stage - y.stage)) {
  console.log(
    `  stage ${a.stage}: worker=${a.workerId} layers=[${a.layerStart},${a.layerEnd}) ` +
      `mem=${(a.estimatedMemoryBytes / 1e6).toFixed(1)} MB embed=${a.hasEmbedding} lm_head=${a.hasLmHead}`,
  );
}
console.log(`  estimatedTokensPerSec=${p2.estimatedTokensPerSec.toFixed(2)} ttftMs=${p2.estimatedTtftMs.toFixed(1)}`);
console.log(`  bottleneck: ${p2.bottleneck}`);

// --- 2. One worker: whole model on one stage ------------------------------
const p1 = plan({ manifest: dense, workers: [wA] });
assert.equal(p1.feasible, true);
assert.equal(p1.pipelineStages, 1);
assert.equal(p1.assignments[0].layerStart, 0);
assert.equal(p1.assignments[0].layerEnd, 4);
assert.ok(p1.assignments[0].hasEmbedding && p1.assignments[0].hasLmHead);
assertContiguousCoverage(p1, 4);
assertMemoryFits(p1, [wA]);
console.log("1-worker plan: OK (1 stage, layers [0,4), embed + lm_head)");

// --- 3. More workers than layers: stages capped at numLayers --------------
const many = [wA, wB, makeWorker("worker-c", 8 * GB, 3e12, 1e15), makeWorker("worker-d", 8 * GB, 3e12, 1e15), makeWorker("worker-e", 8 * GB, 3e12, 1e15), makeWorker("worker-f", 8 * GB, 3e12, 1e15)];
const p6 = plan({ manifest: dense, workers: many });
assert.equal(p6.feasible, true);
assert.equal(p6.pipelineStages, 4, "stages capped at numLayers");
assertContiguousCoverage(p6, 4);
assertMemoryFits(p6, many);
assert.ok(p6.bottleneck.includes("unused"), "bottleneck notes unused workers");
console.log("6-workers/4-layers plan: OK (4 stages, 2 workers unused)");

// --- 4. Worker with too little memory is skipped --------------------------
const tiny = makeWorker("worker-tiny", 1 * 1024 * 1024, 1e9, 1e12); // 1 MB free
const pSkip = plan({ manifest: dense, workers: [wA, wB, tiny] });
assert.equal(pSkip.feasible, true);
assert.equal(pSkip.pipelineStages, 2, "tiny worker skipped, 2 stages remain");
assert.ok(!pSkip.assignments.some((a) => a.workerId === "worker-tiny"), "tiny worker got no stage");
assert.ok(pSkip.bottleneck.includes("worker-tiny"), "bottleneck notes the skipped worker");
assertContiguousCoverage(pSkip, 4);
assertMemoryFits(pSkip, [wA, wB]);
console.log("tiny-worker plan: OK (worker-tiny skipped)");

// --- 5. MoE manifest: experts distributed ---------------------------------
const moe = makeManifest(true);
console.log(`moe manifest: ${moe.totalParams.toLocaleString()} params, ${(moe.totalBytes / 1e6).toFixed(1)} MB`);
const pMoe = plan({ manifest: moe, workers: [wA, wB] });
assert.equal(pMoe.feasible, true);
assert.equal(pMoe.pipelineStages, 2);
assertContiguousCoverage(pMoe, 4);
assertMemoryFits(pMoe, [wA, wB]);
for (const a of pMoe.assignments) {
  assert.deepEqual(a.experts, [0, 1], `stage ${a.stage} owns experts [0,1]`);
}
assert.ok(pMoe.expertParallelSize >= 1, "expertParallelSize recorded");
console.log("MoE plan: OK (every stage owns experts [0,1])");

// --- 6. replan: worker loss, same stage count keeps boundaries ------------
const wC = makeWorker("worker-c", 8 * GB, 3e12, 1e15);
const p3 = plan({ manifest: dense, workers: [wA, wB, wC] });
assert.equal(p3.pipelineStages, 3);
const wD = makeWorker("worker-d", 4 * GB, 1.5e12, 5e14);
const rp = replan(dense, [wA, wB, wD], p3);
assert.equal(rp.feasible, true);
assert.equal(rp.pipelineStages, 3, "same stage count after replacing one worker");
assertContiguousCoverage(rp, 4);
assertMemoryFits(rp, [wA, wB, wD]);
const oldRanges = [...p3.assignments].sort((a, b) => a.stage - b.stage).map((a) => [a.layerStart, a.layerEnd]);
const newRanges = [...rp.assignments].sort((a, b) => a.stage - b.stage).map((a) => [a.layerStart, a.layerEnd]);
assert.deepEqual(newRanges, oldRanges, "stage boundaries preserved across replan");
console.log("replan (worker replaced, 3 stages): OK, boundaries kept:", JSON.stringify(newRanges));

// --- 7. replan: worker loss shrinking the mesh re-balances ----------------
const rp2 = replan(dense, [wA], p2);
assert.equal(rp2.feasible, true);
assert.equal(rp2.pipelineStages, 1);
assertContiguousCoverage(rp2, 4);
console.log("replan (2 -> 1 worker): OK, re-balanced to 1 stage");

// --- 8. Infeasible: model fits nowhere ------------------------------------
const noMem = makeWorker("worker-empty", 1024, 1e9, 1e12);
const pBad = plan({ manifest: dense, workers: [noMem] });
assert.equal(pBad.feasible, false, "plan infeasible when nothing fits");
assert.ok(pBad.reason, "infeasible plan carries a reason");
console.log(`infeasible plan: OK (reason: ${pBad.reason})`);

// --- 9. Headline hardware math for a 1T-param BF16 model ------------------
const oneT: ModelManifest = {
  ...dense,
  modelId: "fake/llama-1t",
  totalParams: 1e12,
  totalBytes: 2e12,
  tensors: {
    "model.embed_tokens.weight": {
      dtype: "bfloat16",
      shape: [1000000000000],
      dataOffsets: [0, 2000000000000],
    },
  },
  config: { ...dense.config, numHiddenLayers: 80, numKeyValueHeads: 8 },
};
console.log("\n1T-param BF16 hardware math:");
for (const line of describeHardwareMath(oneT)) console.log(`  ${line}`);
const tps1 = decodeTokensPerSec(oneT, 3e12, 1);
const tps64 = decodeTokensPerSec(oneT, 64 * 3e12, 1);
assert.ok(Math.abs(tps1 - 1.5) < 0.01, `1x H100 gives ~1.5 tok/s, got ${tps1}`);
assert.ok(Math.abs(tps64 - 96) < 0.01, `64x H100 gives ~96 tok/s, got ${tps64}`);
console.log(`\nheadline: 1x H100 -> ${tps1} tok/s; 64x H100 -> ${tps64} tok/s (batch 1)`);

console.log("\nAll planner smoke tests passed.");
