/**
 * softwell-machine — sharding planner.
 *
 * Splits a model across workers with pipeline parallelism first:
 * contiguous, byte-balanced ranges of layers, one stage per worker,
 * embedding on stage 0, lm_head (+ final norm) on the last stage.
 * Tensor parallelism is NOT implemented in the reference engine, so the
 * planner prefers TP=1 and records a TP recommendation (and marks the
 * plan infeasible) when a single layer cannot fit on any one worker.
 *
 * MoE: experts are distributed round-robin across the workers that own
 * a stage. In this pipeline-only design each stage has exactly one
 * worker, so the round-robin degenerates to "the stage's worker owns all
 * experts of its layers" — recorded explicitly in assignment.experts.
 *
 * Roles are implicit in stage order (never mutating WorkerInfo): with
 * >= 2 workers the stage-0 worker is prefill-capable and the remaining
 * stages run mixed/decode. This is documented in the plan's bottleneck
 * string.
 */

import {
  decodeTokensPerSec,
  estimateLatency,
  kvBytesPerTokenPerLayer,
  tensorBytes,
} from "./cost.ts";
import type {
  ModelManifest,
  PlannerInput,
  ShardAssignment,
  ShardPlan,
  WorkerInfo,
} from "../contracts.ts";

/** A ShardPlan plus the feasibility verdict. contracts.ts is not edited. */
export type PlannedShardPlan = ShardPlan & {
  feasible: boolean;
  /** Why the plan is infeasible (only set when feasible is false). */
  reason?: string;
};

/** KV-cache headroom assumed per sequence when sizing a stage (tokens). */
export const KV_HEADROOM_TOKENS = 4096;

/** Bytes of one layer's weights, plus the standalone embedding/lm_head. */
export interface LayerWeightBytes {
  /** Weight bytes per layer index, length == numHiddenLayers. */
  perLayer: number[];
  embedding: number;
  lmHead: number;
  norm: number;
}

const LAYER_RE = /^model\.layers\.(\d+)\./;

/** Split a manifest's tensors into per-layer byte counts. */
export function layerWeightBytes(manifest: ModelManifest): LayerWeightBytes {
  const n = manifest.config.numHiddenLayers;
  const perLayer: number[] = new Array<number>(n).fill(0);
  let embedding = 0;
  let lmHead = 0;
  let norm = 0;
  for (const name of Object.keys(manifest.tensors)) {
    const t = manifest.tensors[name];
    const bytes = tensorBytes(t.dtype, t.shape, t.dataOffsets);
    const m = LAYER_RE.exec(name);
    if (m) {
      const i = Number(m[1]);
      if (i >= 0 && i < n) perLayer[i] += bytes;
    } else if (name === "model.embed_tokens.weight") {
      embedding += bytes;
    } else if (name === "lm_head.weight") {
      lmHead += bytes;
    } else if (name === "model.norm.weight") {
      norm += bytes;
    }
  }
  return { perLayer, embedding, lmHead, norm };
}

/**
 * Partition `numLayers` layers into `numStages` contiguous ranges,
 * balanced by weight bytes (not layer count — MoE layers are heavier).
 * Returns stage boundaries as [start, end) pairs.
 */
export function partitionLayers(layerBytes: number[], numStages: number): Array<[number, number]> {
  const n = layerBytes.length;
  const stages = Math.max(1, Math.min(numStages, n));
  const total = layerBytes.reduce((a, b) => a + b, 0);
  const prefix: number[] = [0];
  for (const b of layerBytes) prefix.push(prefix[prefix.length - 1] + b);
  const bounds: Array<[number, number]> = [];
  let start = 0;
  for (let s = 0; s < stages; s++) {
    const end = s === stages - 1 ? n : closestCut(prefix, total, s, stages, start);
    bounds.push([start, end]);
    start = end;
  }
  return bounds;
}

/** Layer index whose prefix sum is closest to (s+1)/stages of the total. */
function closestCut(
  prefix: number[],
  total: number,
  s: number,
  stages: number,
  start: number,
): number {
  const target = (total * (s + 1)) / stages;
  let best = start + 1;
  let bestDiff = Number.POSITIVE_INFINITY;
  for (let i = start + 1; i < prefix.length; i++) {
    const diff = Math.abs(prefix[i] - target);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = i;
    }
    if (prefix[i] > target && i > start + 1) break;
  }
  return best;
}

function range(n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(i);
  return out;
}

/**
 * Weight bytes one stage holds: its layers + embedding (stage 0) +
 * lm_head and final norm (last stage) + KV-cache headroom for the
 * expected concurrent sequences.
 */
export function stageMemoryBytes(
  stageBytes: number,
  hasEmbedding: boolean,
  hasLmHead: boolean,
  embeddingBytes: number,
  lmHeadBytes: number,
  normBytes: number,
  numLayersInStage: number,
  kvPerTokenPerLayer: number,
  expectedSequences: number,
): number {
  return (
    stageBytes +
    (hasEmbedding ? embeddingBytes : 0) +
    (hasLmHead ? lmHeadBytes + normBytes : 0) +
    kvPerTokenPerLayer * numLayersInStage * expectedSequences * KV_HEADROOM_TOKENS
  );
}

function basePlan(manifest: ModelManifest): ShardPlan {
  return {
    modelId: manifest.modelId,
    totalParams: manifest.totalParams,
    totalWeightBytes: manifest.totalBytes,
    pipelineStages: 0,
    tensorParallelSize: 1,
    expertParallelSize: 1,
    assignments: [],
    estimatedTokensPerSec: 0,
    estimatedTtftMs: 0,
    bottleneck: "",
    createdAt: new Date().toISOString(),
  };
}

function infeasible(manifest: ModelManifest, reason: string, extra?: Partial<ShardPlan>): PlannedShardPlan {
  return { ...basePlan(manifest), feasible: false, reason, bottleneck: reason, ...extra };
}

/**
 * Build a shard plan for `manifest` over `workers`.
 *
 * - 1 worker: the whole model on one stage.
 * - More workers than layers: stages are capped at numLayers; the extra
 *   workers get no stage and are noted in the bottleneck string.
 * - A worker whose free memory cannot hold even its byte-balanced share
 *   is skipped (noted in the bottleneck string) and planning retries
 *   over the remaining workers.
 * - A single layer that fits on no one worker needs tensor parallelism,
 *   which the reference engine does not implement: the plan is marked
 *   infeasible with the recommended TP size.
 */
export function plan(input: PlannerInput): PlannedShardPlan {
  const { manifest } = input;
  const config = manifest.config;
  const numLayers = config.numHiddenLayers;
  const expectedSequences = input.expectedSequences ?? 1;
  const kvPerTokenPerLayer =
    input.kvBytesPerTokenPerLayer ?? kvBytesPerTokenPerLayer(config);

  // Eligible workers: alive and with some free memory. Dead / zero-memory
  // workers are skipped up front.
  const skipped: string[] = [];
  let eligible = input.workers.filter((w) => {
    if (w.state === "dead" || w.freeMemoryBytes <= 0) {
      skipped.push(`${w.workerId} (${w.state}, ${w.freeMemoryBytes} free bytes)`);
      return false;
    }
    return true;
  });
  if (eligible.length === 0) {
    return infeasible(manifest, "no usable workers: all workers are dead or report no free memory");
  }
  if (numLayers <= 0) {
    return infeasible(manifest, "manifest has no hidden layers");
  }

  const weights = layerWeightBytes(manifest);
  const totalLayerBytes = weights.perLayer.reduce((a, b) => a + b, 0);
  const maxLayerBytes = Math.max(...weights.perLayer);
  const maxFree = Math.max(...eligible.map((w) => w.freeMemoryBytes));
  const totalFree = eligible.reduce((a, w) => a + w.freeMemoryBytes, 0);

  // Tensor-parallel check: a single layer must fit on one worker, because
  // the reference engine runs TP=1. Record the recommendation either way.
  if (maxLayerBytes > maxFree) {
    const tpSize = Math.ceil(maxLayerBytes / maxFree);
    if (maxLayerBytes > totalFree) {
      return infeasible(
        manifest,
        `largest layer needs ${maxLayerBytes} bytes but the whole mesh only has ` +
          `${totalFree} free bytes: model cannot fit anywhere`,
        { tensorParallelSize: tpSize },
      );
    }
    return infeasible(
      manifest,
      `largest layer needs ${maxLayerBytes} bytes but the largest worker has only ` +
        `${maxFree} free bytes: needs tensor-parallel size ${tpSize}, ` +
        `which the reference engine (TP=1) does not implement`,
      { tensorParallelSize: tpSize },
    );
  }

  // Retry loop: drop workers that cannot hold their assigned stage and
  // re-partition over the survivors. Terminates because each iteration
  // either succeeds or removes at least one worker.
  for (let attempt = 0; attempt <= eligible.length; attempt++) {
    const numStages = Math.min(eligible.length, numLayers);
    const ranges = partitionLayers(weights.perLayer, numStages);
    const stageBytes = ranges.map(([s, e]) =>
      weights.perLayer.slice(s, e).reduce((a, b) => a + b, 0),
    );

    // Pair the biggest stages with the biggest memories, then re-order
    // by stage index so the pipeline runs in layer order.
    const stageOrder = range(numStages).sort((a, b) => stageBytes[b] - stageBytes[a]);
    const workerOrder = [...eligible].sort((a, b) => b.freeMemoryBytes - a.freeMemoryBytes);
    const paired: Array<{ stage: number; worker: WorkerInfo }> = stageOrder.map((stage, i) => ({
      stage,
      worker: workerOrder[i],
    }));
    paired.sort((a, b) => a.stage - b.stage);

    const assignments: ShardAssignment[] = [];
    const tooSmall: WorkerInfo[] = [];
    for (const { stage, worker } of paired) {
      const [layerStart, layerEnd] = ranges[stage];
      const hasEmbedding = stage === 0;
      const hasLmHead = stage === numStages - 1;
      const mem = stageMemoryBytes(
        stageBytes[stage],
        hasEmbedding,
        hasLmHead,
        weights.embedding,
        weights.lmHead,
        weights.norm,
        layerEnd - layerStart,
        kvPerTokenPerLayer,
        expectedSequences,
      );
      if (mem > worker.freeMemoryBytes) {
        tooSmall.push(worker);
        continue;
      }
      assignments.push({
        workerId: worker.workerId,
        stage,
        layerStart,
        layerEnd,
        hasEmbedding,
        hasLmHead,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: config.numExperts ? range(config.numExperts) : [],
        estimatedMemoryBytes: mem,
      });
    }

    if (tooSmall.length === 0) {
      // Success: every stage fits its worker.
      const unused = eligible.length - numStages;
      const notes: string[] = [];
      if (skipped.length > 0) notes.push(`skipped unusable worker(s): ${skipped.join(", ")}`);
      if (unused > 0) {
        notes.push(
          `${unused} worker(s) unused: more workers (${eligible.length}) than layers (${numLayers}), ` +
            `stages capped at ${numLayers}`,
        );
      }
      return finishPlan(manifest, assignments, eligible, numStages, notes, expectedSequences);
    }

    // Drop the workers that could not hold their stage and try again.
    for (const w of tooSmall) {
      skipped.push(`${w.workerId} (too little free memory for its stage share)`);
    }
    eligible = eligible.filter((w) => !tooSmall.includes(w));
    if (eligible.length === 0) {
      return infeasible(
        manifest,
        `no worker can hold its stage share: total weights ${totalLayerBytes} bytes exceed ` +
          `usable memory; skipped: ${skipped.join("; ")}`,
      );
    }
  }
  return infeasible(manifest, "could not find a feasible partition");
}

function finishPlan(
  manifest: ModelManifest,
  assignments: ShardAssignment[],
  assignedWorkers: WorkerInfo[],
  numStages: number,
  notes: string[],
  _expectedSequences: number,
): PlannedShardPlan {
  const config = manifest.config;
  const byId = new Map(assignedWorkers.map((w) => [w.workerId, w]));
  let aggBandwidth = 0;
  let aggFlops = 0;
  for (const a of assignments) {
    const w = byId.get(a.workerId);
    if (w) {
      aggBandwidth += w.bandwidthBytesPerSec;
      aggFlops += w.flopsPerSec;
    }
  }
  const latency = estimateLatency(manifest, aggBandwidth, aggFlops, 1024, 1);
  const tps = decodeTokensPerSec(manifest, aggBandwidth, 1);

  const bottleneckParts: string[] = [latency.bottleneck];
  if (assignments.length >= 2) {
    const first = assignments[0];
    bottleneckParts.push(
      `prefill/disaggregation: stage-0 worker ${first.workerId} is prefill-capable; ` +
        `stages 1..${assignments.length - 1} run mixed/decode (roles implicit in stage order)`,
    );
  } else {
    bottleneckParts.push("single worker: whole model on one stage, prefill and decode share it");
  }
  if (config.numExperts) {
    bottleneckParts.push(
      `MoE: ${config.numExperts} experts/layer, ${config.numExpertsPerToken ?? 0} routed per token; ` +
        `each stage's worker owns all experts of its layers (round-robin across stages)`,
    );
  }
  bottleneckParts.push(...notes);

  return {
    modelId: manifest.modelId,
    totalParams: manifest.totalParams,
    totalWeightBytes: manifest.totalBytes,
    pipelineStages: numStages,
    tensorParallelSize: 1,
    expertParallelSize: config.numExperts
      ? Math.min(config.numExperts, numStages)
      : 1,
    assignments,
    estimatedTokensPerSec: tps,
    estimatedTtftMs: latency.ttftMs,
    bottleneck: bottleneckParts.join(" | "),
    createdAt: new Date().toISOString(),
    feasible: true,
  };
}

/**
 * Re-shard after worker loss.
 *
 * Approach: re-run `plan()` over the survivors, then — when the survivor
 * count still yields the same number of stages — snap the new assignments
 * back onto the old plan's layer ranges, so in-flight sequences and
 * per-stage KV caches move as little as possible. Snapping is only kept
 * if every snapped stage still fits its worker's free memory; otherwise
 * the fresh byte-balanced partition is used and the moved boundaries are
 * reported in the bottleneck string.
 */
export function replan(
  manifest: ModelManifest,
  survivingWorkers: WorkerInfo[],
  oldPlan: ShardPlan,
): PlannedShardPlan {
  const fresh = plan({ manifest, workers: survivingWorkers });
  if (!fresh.feasible) return fresh;
  if (fresh.pipelineStages !== oldPlan.pipelineStages) {
    fresh.bottleneck +=
      ` | re-plan after worker loss: stage count changed ${oldPlan.pipelineStages} -> ` +
      `${fresh.pipelineStages}, stage boundaries re-balanced by weight bytes`;
    return fresh;
  }

  const weights = layerWeightBytes(manifest);
  const oldRanges = [...oldPlan.assignments].sort((a, b) => a.stage - b.stage);
  const newByStage = [...fresh.assignments].sort((a, b) => a.stage - b.stage);
  if (oldRanges.length !== newByStage.length) {
    fresh.bottleneck += " | re-plan after worker loss: boundaries re-balanced by weight bytes";
    return fresh;
  }

  // Try to snap onto the old ranges.
  const kvPerLayer = kvBytesPerTokenPerLayer(manifest.config);
  const snapped: ShardAssignment[] = [];
  let moved = 0;
  for (let i = 0; i < newByStage.length; i++) {
    const a = newByStage[i];
    const old = oldRanges[i];
    const worker = survivingWorkers.find((w) => w.workerId === a.workerId);
    if (!worker) return fresh;
    const stageBytes = weights.perLayer
      .slice(old.layerStart, old.layerEnd)
      .reduce((x, y) => x + y, 0);
    const mem = stageMemoryBytes(
      stageBytes,
      old.hasEmbedding,
      old.hasLmHead,
      weights.embedding,
      weights.lmHead,
      weights.norm,
      old.layerEnd - old.layerStart,
      kvPerLayer,
      1,
    );
    if (mem > worker.freeMemoryBytes) {
      fresh.bottleneck +=
        " | re-plan after worker loss: old stage boundaries do not fit the survivors' " +
        "memory, boundaries re-balanced by weight bytes";
      return fresh;
    }
    if (a.layerStart !== old.layerStart || a.layerEnd !== old.layerEnd) moved++;
    snapped.push({
      ...a,
      layerStart: old.layerStart,
      layerEnd: old.layerEnd,
      hasEmbedding: old.hasEmbedding,
      hasLmHead: old.hasLmHead,
      estimatedMemoryBytes: mem,
    });
  }
  snapped.sort((a, b) => a.stage - b.stage);
  fresh.assignments = snapped;
  fresh.bottleneck +=
    ` | re-plan after worker loss: kept ${newByStage.length - moved}/${newByStage.length} ` +
    `stage boundaries from the previous plan` +
    (moved > 0 ? ` (${moved} moved to fit survivors)` : "");
  return fresh;
}
