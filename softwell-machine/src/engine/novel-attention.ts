/**
 * softwell-machine — novel attention mathematics.
 *
 * Two approximate attention variants with *provable* error bounds,
 * implemented as pure functions on Float32Array in the style of
 * src/engine/attention.ts. The existing engine is untouched: these are
 * parallel paths to be wired in later behind a flag.
 *
 * Shared idea. Partition the KV prefix into consecutive blocks of size B.
 * For each block b and KV head, precompute once (per KV cache, reused by
 * every query) the block mean key kbar_b, the block mean value vbar_b,
 * and the block radii
 *
 *   R_b = max_t ||k_t - kbar_b||_2,   S_b = max_t ||v_t - vbar_b||_2.
 *
 * All bounds below are worst-case over the actual data — no
 * distributional assumptions, no learned parameters, no training.
 *
 * Theorem A1 (landmark / block-mean attention). Distant positions are
 * replaced by their block means: one score dot and one value vector per
 * block instead of B. The last `localWindow` positions are always exact,
 * and any block whose keys are too dispersed to approximate safely is
 * computed exactly instead. With B = ceil(sqrt(n)) the prefill cost is
 * O(n^(3/2) d) instead of O(n^2 d); see docs/research/01-attention.md.
 *
 * Theorem A2 (dynamic sparsity by block screening). A block is skipped
 * entirely when a rigorous upper bound on its total attention mass is
 * below `dropThreshold`. The bound on the skipped mass is computed from
 * the data, and the output error is at most twice the skipped mass
 * times the largest value norm.
 *
 * Op-counting convention (used by every `ops` figure returned here and
 * by `exactOps`): one multiply-add on one vector element counts as one
 * op. A length-d dot product costs d ops; scaling-and-adding one
 * length-d value vector costs d ops. Block-mean accumulation costs d
 * ops per position; a block radius costs 2d ops per position (subtract
 * + square-accumulate). Softmax exponentials, scalar max/sum work and
 * index arithmetic are O(n) per row — lower order than the O(n d)
 * matmul-like work — and are not counted, for the exact and the
 * approximate paths alike.
 */

/** Float32 reassociation slack: the theorems are proved in exact
 * arithmetic; the implementation reorders the same sums, so the
 * returned bound adds this fraction of (max value norm + 1). At
 * n <= 512 it is orders of magnitude above observed float32 drift
 * (~1e-6), and it is the only non-rigorous term in either bound. */
const FLOAT_SLACK = 1e-4;

/** Per-KV-cache block summaries shared by Theorems A1 and A2. */
export interface BlockStats {
  blockSize: number;
  numBlocks: number;
  kvLen: number;
  numKvHeads: number;
  headDim: number;
  /** Block mean keys, [numBlocks, numKvHeads, headDim] row-major. */
  meanK: Float32Array;
  /** Block mean values, [numBlocks, numKvHeads, headDim] row-major. */
  meanV: Float32Array;
  /** Key radii, [numBlocks, numKvHeads]: max_t ||k_t - meanK_b||_2. */
  radiusK: Float32Array;
  /** Value radii, [numBlocks, numKvHeads]: max_t ||v_t - meanV_b||_2. */
  radiusV: Float32Array;
  /** Largest value norm per KV head, [numKvHeads]. */
  maxVNorm: Float32Array;
}

export interface BlockStatsResult {
  stats: BlockStats;
  /** Ops spent building the stats, same counting convention. */
  ops: number;
}

/** Mutable op counter threaded through the per-head cores. */
interface OpCounter {
  ops: number;
}

interface OneBlockStats {
  /** meanK/meanV for this block: [numKvHeads, headDim] row-major. */
  meanK: Float32Array;
  meanV: Float32Array;
  /** Radii for this block: [numKvHeads]. */
  radiusK: Float32Array;
  radiusV: Float32Array;
  /** Largest value norm among this block's rows, per KV head. */
  maxVNorm: Float32Array;
  ops: number;
}

/**
 * Block summaries for a single block `b` of the KV cache, using exactly
 * the same arithmetic (and op counting) as computeBlockStats. Factored
 * out so extendBlockStats can recompute one block without rebuilding
 * the whole cache.
 */
function computeSingleBlock(
  k: Float32Array,
  v: Float32Array,
  numKvHeads: number,
  headDim: number,
  blockSize: number,
  b: number,
  kvLen: number,
): OneBlockStats {
  const start = b * blockSize;
  const end = Math.min(start + blockSize, kvLen);
  const count = end - start;
  const meanK = new Float32Array(numKvHeads * headDim);
  const meanV = new Float32Array(numKvHeads * headDim);
  const radiusK = new Float32Array(numKvHeads);
  const radiusV = new Float32Array(numKvHeads);
  const maxVNorm = new Float32Array(numKvHeads);
  let ops = 0;
  for (let kh = 0; kh < numKvHeads; kh++) {
    for (let t = start; t < end; t++) {
      const kBase = (t * numKvHeads + kh) * headDim;
      const mBase = kh * headDim;
      for (let d = 0; d < headDim; d++) {
        meanK[mBase + d] += k[kBase + d];
        meanV[mBase + d] += v[kBase + d];
      }
      ops += 2 * headDim;
    }
    const mBase = kh * headDim;
    const inv = 1 / count;
    for (let d = 0; d < headDim; d++) {
      meanK[mBase + d] *= inv;
      meanV[mBase + d] *= inv;
    }
    let rK = 0;
    let rV = 0;
    for (let t = start; t < end; t++) {
      const kBase = (t * numKvHeads + kh) * headDim;
      let dk = 0;
      let dv = 0;
      let vv = 0;
      for (let d = 0; d < headDim; d++) {
        const ek = k[kBase + d] - meanK[mBase + d];
        const ev = v[kBase + d] - meanV[mBase + d];
        dk += ek * ek;
        dv += ev * ev;
        vv += v[kBase + d] * v[kBase + d];
      }
      ops += 5 * headDim;
      if (dk > rK) rK = dk;
      if (dv > rV) rV = dv;
      const vNorm = Math.sqrt(vv);
      if (vNorm > maxVNorm[kh]) maxVNorm[kh] = vNorm;
    }
    radiusK[kh] = Math.sqrt(rK);
    radiusV[kh] = Math.sqrt(rV);
  }
  return { meanK, meanV, radiusK, radiusV, maxVNorm, ops };
}

/**
 * Extend previously computed block stats to a longer KV cache.
 *
 * `stats` describes the first `stats.kvLen` rows of `k`/`v`; `k`/`v`
 * now hold `newKvLen >= stats.kvLen` rows. Only the blocks that gained
 * rows are recomputed — the last partial old block (if any) plus any
 * new blocks — so extending by one decode token costs O(B·d), not
 * O(n·d). Blocks before the first changed block are copied verbatim,
 * and maxVNorm can only grow, so the result is bit-identical to
 * computeBlockStats over the full new cache (same arithmetic, same
 * order). Throws when newKvLen < stats.kvLen (shrinking is a rebuild:
 * call computeBlockStats) or when the shapes do not match.
 */
export function extendBlockStats(
  stats: BlockStats,
  k: Float32Array,
  v: Float32Array,
  newKvLen: number,
): BlockStatsResult {
  const { numKvHeads, headDim, blockSize } = stats;
  if (!Number.isInteger(newKvLen) || newKvLen < stats.kvLen) {
    throw new Error(
      `extendBlockStats: newKvLen ${newKvLen} < stats.kvLen ${stats.kvLen}; shrinking needs a rebuild`,
    );
  }
  if (k.length !== newKvLen * numKvHeads * headDim) {
    throw new Error(
      `extendBlockStats: k length ${k.length} != newKvLen ${newKvLen} * ${numKvHeads} * ${headDim}`,
    );
  }
  if (v.length !== k.length) {
    throw new Error(`extendBlockStats: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  const newNumBlocks = Math.ceil(newKvLen / blockSize);
  if (newNumBlocks === stats.numBlocks && newKvLen === stats.kvLen) {
    return { stats, ops: 0 }; // nothing changed
  }
  // First block that may have changed: the last old block, when it was
  // partial and has now grown; when the old cache ended on a block
  // boundary, every old block is final and only new blocks are computed.
  // In both cases that index is floor(kvLen / blockSize).
  const firstChanged = Math.floor(stats.kvLen / blockSize);
  const meanK = new Float32Array(newNumBlocks * numKvHeads * headDim);
  const meanV = new Float32Array(newNumBlocks * numKvHeads * headDim);
  const radiusK = new Float32Array(newNumBlocks * numKvHeads);
  const radiusV = new Float32Array(newNumBlocks * numKvHeads);
  const maxVNorm = new Float32Array(numKvHeads);
  meanK.set(stats.meanK.subarray(0, firstChanged * numKvHeads * headDim));
  meanV.set(stats.meanV.subarray(0, firstChanged * numKvHeads * headDim));
  radiusK.set(stats.radiusK.subarray(0, firstChanged * numKvHeads));
  radiusV.set(stats.radiusV.subarray(0, firstChanged * numKvHeads));
  for (let kh = 0; kh < numKvHeads; kh++) maxVNorm[kh] = stats.maxVNorm[kh];
  let ops = 0;
  for (let b = firstChanged; b < newNumBlocks; b++) {
    const one = computeSingleBlock(k, v, numKvHeads, headDim, blockSize, b, newKvLen);
    meanK.set(one.meanK, b * numKvHeads * headDim);
    meanV.set(one.meanV, b * numKvHeads * headDim);
    radiusK.set(one.radiusK, b * numKvHeads);
    radiusV.set(one.radiusV, b * numKvHeads);
    for (let kh = 0; kh < numKvHeads; kh++) {
      if (one.maxVNorm[kh] > maxVNorm[kh]) maxVNorm[kh] = one.maxVNorm[kh];
    }
    ops += one.ops;
  }
  return {
    stats: {
      blockSize,
      numBlocks: newNumBlocks,
      kvLen: newKvLen,
      numKvHeads,
      headDim,
      meanK,
      meanV,
      radiusK,
      radiusV,
      maxVNorm,
    },
    ops,
  };
}

/**
 * Build block summaries for a KV cache.
 *
 * k, v are [kvLen, numKvHeads, headDim] row-major. blockSize must be
 * >= 1. The summaries are exact functions of k and v (up to float32
 * rounding); nothing is approximated here.
 */
export function computeBlockStats(
  k: Float32Array,
  v: Float32Array,
  numKvHeads: number,
  headDim: number,
  blockSize: number,
): BlockStatsResult {
  const kvLen = k.length / (numKvHeads * headDim);
  if (!Number.isInteger(kvLen) || kvLen === 0) {
    throw new Error(`computeBlockStats: k length ${k.length} is not a whole number of KV rows`);
  }
  if (v.length !== k.length) {
    throw new Error(`computeBlockStats: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  if (!Number.isInteger(blockSize) || blockSize < 1) {
    throw new Error(`computeBlockStats: blockSize must be a positive integer, got ${blockSize}`);
  }
  const numBlocks = Math.ceil(kvLen / blockSize);
  const meanK = new Float32Array(numBlocks * numKvHeads * headDim);
  const meanV = new Float32Array(numBlocks * numKvHeads * headDim);
  const radiusK = new Float32Array(numBlocks * numKvHeads);
  const radiusV = new Float32Array(numBlocks * numKvHeads);
  const maxVNorm = new Float32Array(numKvHeads);
  let ops = 0;

  for (let b = 0; b < numBlocks; b++) {
    const one = computeSingleBlock(k, v, numKvHeads, headDim, blockSize, b, kvLen);
    meanK.set(one.meanK, b * numKvHeads * headDim);
    meanV.set(one.meanV, b * numKvHeads * headDim);
    radiusK.set(one.radiusK, b * numKvHeads);
    radiusV.set(one.radiusV, b * numKvHeads);
    for (let kh = 0; kh < numKvHeads; kh++) {
      if (one.maxVNorm[kh] > maxVNorm[kh]) maxVNorm[kh] = one.maxVNorm[kh];
    }
    ops += one.ops;
  }
  return {
    stats: { blockSize, numBlocks, kvLen, numKvHeads, headDim, meanK, meanV, radiusK, radiusV, maxVNorm },
    ops,
  };
}

export interface NovelAttentionOptions {
  /** Mask out KV positions after the query position (default true). */
  causal?: boolean;
  /** Absolute position of the query token (needed for causal masking). */
  queryPosition?: number;
  /** Absolute positions of the KV entries, in order (needed for causal masking). */
  kvPositions?: number[];
  /** Attention scale; defaults to 1/sqrt(headDim). */
  scale?: number;
  /** Trailing positions always computed exactly (default 16). */
  localWindow?: number;
  /** KV block size; defaults to ceil(sqrt(visible length)), min 4. */
  blockSize?: number;
  /** Precomputed block stats for this exact k/v; built on the fly if omitted. */
  stats?: BlockStats;
}

export interface LandmarkAttentionOptions extends NovelAttentionOptions {
  /**
   * Largest tolerated per-logit perturbation from replacing keys by
   * their block mean (default 0.05). A block whose
   * scale * ||q|| * R_b exceeds this is computed exactly, so the
   * returned error bound stays tight on incoherent data at the cost
   * of exact work for those blocks.
   */
  deltaMax?: number;
}

export interface SparseAttentionOptions extends NovelAttentionOptions {
  /**
   * Per-block mass threshold (default 1e-3). A block is skipped only
   * when a rigorous upper bound on its total attention mass is at
   * most this large; the sum of the skipped bounds is reported as
   * `droppedMassBound`.
   */
  dropThreshold?: number;
}

export interface NovelAttentionResult {
  output: Float32Array;
  /** Counted multiply-add-equivalent operations (see file header). */
  ops: number;
  /** Rigorous upper bound on ||output - exact||_2, max over heads (and rows, for prefill). */
  errorBound: number;
  /** Rigorous upper bound on total dropped attention mass (A2; 0 for A1). */
  droppedMassBound: number;
  /** What exact grouped-query attention would cost, same counting convention. */
  exactOps: number;
}

/**
 * Number of visible KV positions for one query, mirroring
 * groupedQueryAttention's masking: mask only when causal masking is on
 * and both kvPositions and queryPosition are given. Block methods need
 * the visible set to be a prefix of the KV cache, so kvPositions must
 * be non-decreasing when masking is used.
 */
function visibleLength(
  kvLen: number,
  options: NovelAttentionOptions,
): number {
  const causal = options.causal ?? true;
  if (!causal || !options.kvPositions || options.queryPosition === undefined) return kvLen;
  const pos = options.kvPositions;
  if (pos.length !== kvLen) {
    throw new Error(`novel attention: kvPositions length ${pos.length} != kvLen ${kvLen}`);
  }
  let vis = 0;
  for (let t = 0; t < kvLen; t++) {
    if (t > 0 && pos[t] < pos[t - 1]) {
      throw new Error("novel attention: kvPositions must be non-decreasing");
    }
    if (pos[t] <= options.queryPosition) vis = t + 1;
  }
  return vis;
}

function validateWindow(localWindow: number): void {
  if (!Number.isInteger(localWindow) || localWindow < 1) {
    throw new Error(`novel attention: localWindow must be a positive integer, got ${localWindow}`);
  }
}

function validateHeads(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
): number {
  const kvLen = k.length / (numKvHeads * headDim);
  if (!Number.isInteger(kvLen) || kvLen === 0) {
    throw new Error(`novel attention: k length ${k.length} is not a whole number of KV rows`);
  }
  if (q.length !== numHeads * headDim) {
    throw new Error(`novel attention: q length ${q.length}, expected ${numHeads * headDim}`);
  }
  if (v.length !== k.length) {
    throw new Error(`novel attention: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(`novel attention: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`);
  }
  return kvLen;
}

function checkStats(
  stats: BlockStats,
  kvLen: number,
  numKvHeads: number,
  headDim: number,
  blockSize?: number,
): void {
  if (stats.kvLen !== kvLen || stats.numKvHeads !== numKvHeads || stats.headDim !== headDim) {
    throw new Error(
      "novel attention: supplied stats do not describe this k/v " +
        `(stats kvLen ${stats.kvLen}, numKvHeads ${stats.numKvHeads}, headDim ${stats.headDim})`,
    );
  }
  if (blockSize !== undefined && blockSize !== stats.blockSize) {
    throw new Error(
      `novel attention: blockSize ${blockSize} != stats blockSize ${stats.blockSize}`,
    );
  }
}

function resolveStats(
  k: Float32Array,
  v: Float32Array,
  numKvHeads: number,
  headDim: number,
  visLen: number,
  options: NovelAttentionOptions,
  counter: OpCounter,
): BlockStats {
  if (options.stats) {
    checkStats(options.stats, k.length / (numKvHeads * headDim), numKvHeads, headDim, options.blockSize);
    return options.stats;
  }
  const blockSize = options.blockSize ?? Math.max(4, Math.ceil(Math.sqrt(Math.max(visLen, 1))));
  const built = computeBlockStats(k, v, numKvHeads, headDim, blockSize);
  counter.ops += built.ops;
  return built.stats;
}

interface HeadOutcome {
  out: Float32Array;
  bound: number;
  droppedMass: number;
}

interface BlockItem {
  kind: "block";
  block: number;
}

interface PosItem {
  kind: "pos";
  pos: number;
}

type Item = BlockItem | PosItem;

/**
 * Theorem A1, one query head. Distant full blocks are replaced by
 * their block mean (one score dot, one value vector per block) unless
 * their key dispersion makes the logit perturbation exceed deltaMax,
 * in which case the block is computed exactly. The trailing
 * localWindow positions and the partial block at the distant boundary
 * are always exact. Returns the head output and its error bound.
 */
function landmarkHead(
  q: Float32Array,
  qOff: number,
  k: Float32Array,
  v: Float32Array,
  kvHead: number,
  headDim: number,
  visLen: number,
  stats: BlockStats,
  scale: number,
  localWindow: number,
  deltaMax: number,
  counter: OpCounter,
): HeadOutcome {
  const bSize = stats.blockSize;
  const out = new Float32Array(headDim);
  let qNormSq = 0;
  for (let d = 0; d < headDim; d++) qNormSq += q[qOff + d] * q[qOff + d];
  const qNorm = Math.sqrt(qNormSq);
  counter.ops += headDim;

  const localStart = Math.max(0, visLen - localWindow);
  const distantEnd = localStart;
  const numFullBlocks = Math.floor(distantEnd / bSize);

  // Build the collapsed item list: approximate blocks plus exact positions.
  const items: Item[] = [];
  let delta = 0;
  let sRadius = 0;
  for (let b = 0; b < numFullBlocks; b++) {
    const deltaB = scale * qNorm * stats.radiusK[b * stats.numKvHeads + kvHead];
    if (deltaB <= deltaMax) {
      items.push({ kind: "block", block: b });
      if (deltaB > delta) delta = deltaB;
      const sB = stats.radiusV[b * stats.numKvHeads + kvHead];
      if (sB > sRadius) sRadius = sB;
    } else {
      for (let t = b * bSize; t < (b + 1) * bSize; t++) items.push({ kind: "pos", pos: t });
    }
  }
  for (let t = numFullBlocks * bSize; t < distantEnd; t++) items.push({ kind: "pos", pos: t });
  for (let t = localStart; t < visLen; t++) items.push({ kind: "pos", pos: t });
  if (items.length === 0) return { out, bound: 0, droppedMass: 0 }; // fully masked

  // Scores: one dot per item (a block item scores B positions at once).
  const logits = new Float32Array(items.length);
  let maxLogit = -Infinity;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    let dot = 0;
    if (it.kind === "block") {
      const base = (it.block * stats.numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) dot += q[qOff + d] * stats.meanK[base + d];
    } else {
      const base = (it.pos * stats.numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) dot += q[qOff + d] * k[base + d];
    }
    counter.ops += headDim;
    logits[i] = dot * scale;
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  // Softmax over the collapsed distribution: a block item carries
  // multiplicity bSize (its B positions share one approximate logit).
  let z = 0;
  const weights = new Float32Array(items.length);
  for (let i = 0; i < items.length; i++) {
    const mult = items[i].kind === "block" ? bSize : 1;
    weights[i] = Math.exp(logits[i] - maxLogit) * mult;
    z += weights[i];
  }
  const invZ = 1 / z;
  for (let i = 0; i < items.length; i++) {
    const w = weights[i] * invZ;
    if (w === 0) continue;
    const it = items[i];
    if (it.kind === "block") {
      const base = (it.block * stats.numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) out[d] += w * stats.meanV[base + d];
    } else {
      const base = (it.pos * stats.numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) out[d] += w * v[base + d];
    }
    counter.ops += headDim;
  }

  // Theorem A1 bound: (e^(2 delta) - 1) * Vmax + S, plus float slack.
  const vMax = stats.maxVNorm[kvHead];
  const bound = (Math.exp(2 * delta) - 1) * vMax + sRadius + FLOAT_SLACK * (vMax + 1);
  return { out, bound, droppedMass: 0 };
}

/**
 * Theorem A2, one query head. Every distant full block gets a rigorous
 * upper bound on its total attention mass from its landmark score plus
 * its key radius; blocks whose bound is at most dropThreshold are
 * skipped outright. The trailing localWindow positions, the partial
 * block at the distant boundary, and every surviving block are scored
 * exactly, and the kept weights are renormalized. Returns the head
 * output, its error bound, and the dropped-mass bound.
 */
function sparseHead(
  q: Float32Array,
  qOff: number,
  k: Float32Array,
  v: Float32Array,
  kvHead: number,
  headDim: number,
  visLen: number,
  stats: BlockStats,
  scale: number,
  localWindow: number,
  dropThreshold: number,
  counter: OpCounter,
): HeadOutcome {
  const bSize = stats.blockSize;
  const out = new Float32Array(headDim);
  let qNormSq = 0;
  for (let d = 0; d < headDim; d++) qNormSq += q[qOff + d] * q[qOff + d];
  const qNorm = Math.sqrt(qNormSq);
  counter.ops += headDim;

  const localStart = Math.max(0, visLen - localWindow);
  const distantEnd = localStart;
  const numFullBlocks = Math.floor(distantEnd / bSize);
  if (visLen === 0) return { out, bound: 0, droppedMass: 0 }; // fully masked

  // Exact local scores (always kept). Their max lower-bounds the true
  // max score, which is what makes the mass bounds below rigorous.
  const localCount = visLen - localStart;
  const localScores = new Float32Array(localCount);
  let m = -Infinity;
  for (let j = 0; j < localCount; j++) {
    const t = localStart + j;
    const base = (t * stats.numKvHeads + kvHead) * headDim;
    let dot = 0;
    for (let d = 0; d < headDim; d++) dot += q[qOff + d] * k[base + d];
    counter.ops += headDim;
    localScores[j] = dot * scale;
    if (localScores[j] > m) m = localScores[j];
  }

  // Landmark scores for distant full blocks; max landmark score is
  // also a lower bound on the true max score (max >= mean per block).
  const landmark = new Float32Array(numFullBlocks);
  for (let b = 0; b < numFullBlocks; b++) {
    const base = (b * stats.numKvHeads + kvHead) * headDim;
    let dot = 0;
    for (let d = 0; d < headDim; d++) dot += q[qOff + d] * stats.meanK[base + d];
    counter.ops += headDim;
    landmark[b] = dot * scale;
    if (landmark[b] > m) m = landmark[b];
  }

  // Screen: skip a block only when its total-mass upper bound
  // B * exp(landmark + scale*||q||*R_b - m) is at most dropThreshold.
  // Kept positions go first in the list, local positions last, so the
  // local scores computed above can be reused without recomputation.
  const keptPos: number[] = [];
  let dropped = 0;
  for (let b = 0; b < numFullBlocks; b++) {
    const upper = landmark[b] + scale * qNorm * stats.radiusK[b * stats.numKvHeads + kvHead];
    const massBound = bSize * Math.exp(upper - m);
    if (massBound <= dropThreshold) {
      dropped += massBound;
    } else {
      for (let t = b * bSize; t < (b + 1) * bSize; t++) keptPos.push(t);
    }
  }
  for (let t = numFullBlocks * bSize; t < distantEnd; t++) keptPos.push(t);
  const keptCount = keptPos.length;
  for (let j = 0; j < localCount; j++) keptPos.push(localStart + j);

  // Exact scores for kept non-local positions; local scores reused.
  const logits = new Float32Array(keptPos.length);
  for (let i = 0; i < keptCount; i++) {
    const base = (keptPos[i] * stats.numKvHeads + kvHead) * headDim;
    let dot = 0;
    for (let d = 0; d < headDim; d++) dot += q[qOff + d] * k[base + d];
    counter.ops += headDim;
    logits[i] = dot * scale;
  }
  for (let j = 0; j < localCount; j++) logits[keptCount + j] = localScores[j];

  // Renormalized softmax over kept positions only.
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
  let z = 0;
  const weights = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    weights[i] = Math.exp(logits[i] - maxLogit);
    z += weights[i];
  }
  const invZ = 1 / z;
  for (let i = 0; i < logits.length; i++) {
    const w = weights[i] * invZ;
    if (w === 0) continue;
    const base = (keptPos[i] * stats.numKvHeads + kvHead) * headDim;
    for (let d = 0; d < headDim; d++) out[d] += w * v[base + d];
    counter.ops += headDim;
  }

  // Theorem A2 bound: 2 * droppedMass * Vmax, plus float slack.
  const vMax = stats.maxVNorm[kvHead];
  const bound = 2 * dropped * vMax + FLOAT_SLACK * (vMax + 1);
  return { out, bound, droppedMass: dropped };
}

/**
 * Theorem A1 for a single query token: landmark / block-mean attention.
 * Same signature and masking semantics as groupedQueryAttention in
 * attention.ts; see the file header for the bound and the op counting.
 */
export function landmarkAttention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  options: LandmarkAttentionOptions = {},
): NovelAttentionResult {
  const kvLen = validateHeads(q, k, v, numHeads, numKvHeads, headDim);
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const visLen = visibleLength(kvLen, options);
  const localWindow = options.localWindow ?? 16;
  validateWindow(localWindow);
  const deltaMax = options.deltaMax ?? 0.05;
  const counter: OpCounter = { ops: 0 };
  const stats = resolveStats(k, v, numKvHeads, headDim, visLen, options, counter);
  const groupSize = numHeads / numKvHeads;
  const out = new Float32Array(numHeads * headDim);
  let errorBound = 0;
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const r = landmarkHead(
      q, h * headDim, k, v, kvHead, headDim, visLen, stats, scale, localWindow, deltaMax, counter,
    );
    out.set(r.out, h * headDim);
    if (r.bound > errorBound) errorBound = r.bound;
  }
  return {
    output: out,
    ops: counter.ops,
    errorBound,
    droppedMassBound: 0,
    exactOps: 2 * visLen * headDim * numHeads,
  };
}

/**
 * Theorem A2 for a single query token: dynamic sparsity by block
 * screening. Same signature and masking semantics as
 * groupedQueryAttention; see the file header for the bound.
 */
export function sparseAttention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  options: SparseAttentionOptions = {},
): NovelAttentionResult {
  const kvLen = validateHeads(q, k, v, numHeads, numKvHeads, headDim);
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const visLen = visibleLength(kvLen, options);
  const localWindow = options.localWindow ?? 16;
  validateWindow(localWindow);
  const dropThreshold = options.dropThreshold ?? 1e-3;
  const counter: OpCounter = { ops: 0 };
  const stats = resolveStats(k, v, numKvHeads, headDim, visLen, options, counter);
  const groupSize = numHeads / numKvHeads;
  const out = new Float32Array(numHeads * headDim);
  let errorBound = 0;
  let droppedMass = 0;
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const r = sparseHead(
      q, h * headDim, k, v, kvHead, headDim, visLen, stats, scale, localWindow, dropThreshold, counter,
    );
    out.set(r.out, h * headDim);
    if (r.bound > errorBound) errorBound = r.bound;
    if (r.droppedMass > droppedMass) droppedMass = r.droppedMass;
  }
  return {
    output: out,
    ops: counter.ops,
    errorBound,
    droppedMassBound: droppedMass,
    exactOps: 2 * visLen * headDim * numHeads,
  };
}

export interface PrefillOptions {
  /** Attention scale; defaults to 1/sqrt(headDim). */
  scale?: number;
  /** Trailing positions always computed exactly (default 16). */
  localWindow?: number;
  /** KV block size; defaults to ceil(sqrt(seqLen)), min 4. */
  blockSize?: number;
  /** Precomputed block stats for this exact k/v; built on the fly if omitted. */
  stats?: BlockStats;
}

export interface LandmarkPrefillOptions extends PrefillOptions {
  /** Largest tolerated per-logit perturbation (default 0.05). */
  deltaMax?: number;
}

export interface SparsePrefillOptions extends PrefillOptions {
  /** Per-block mass threshold (default 1e-3). */
  dropThreshold?: number;
}

/**
 * Theorem A1 for a whole prefill: q is [seqLen, numHeads, headDim],
 * k/v are [seqLen, numKvHeads, headDim]; row i attends causally over
 * [0..i]. Block stats are built once for the whole KV cache and reused
 * by every row — that reuse is what makes the total cost
 * O(n^(3/2) d) with B = ceil(sqrt(n)) instead of O(n^2 d).
 * errorBound / droppedMassBound are the maxima over rows and heads.
 */
export function prefillLandmarkAttention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  options: LandmarkPrefillOptions = {},
): NovelAttentionResult {
  const seqLen = q.length / (numHeads * headDim);
  if (!Number.isInteger(seqLen) || seqLen === 0) {
    throw new Error(`prefillLandmarkAttention: q length ${q.length} is not a whole number of query rows`);
  }
  const kvLen = k.length / (numKvHeads * headDim);
  if (kvLen !== seqLen) {
    throw new Error(`prefillLandmarkAttention: q rows ${seqLen} != k rows ${kvLen}`);
  }
  if (v.length !== k.length) {
    throw new Error(`prefillLandmarkAttention: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(
      `prefillLandmarkAttention: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`,
    );
  }
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const localWindow = options.localWindow ?? 16;
  validateWindow(localWindow);
  const deltaMax = options.deltaMax ?? 0.05;
  const counter: OpCounter = { ops: 0 };
  const stats = resolveStats(k, v, numKvHeads, headDim, seqLen, options, counter);
  const groupSize = numHeads / numKvHeads;
  const out = new Float32Array(seqLen * numHeads * headDim);
  let errorBound = 0;
  for (let i = 0; i < seqLen; i++) {
    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / groupSize);
      const r = landmarkHead(
        q, (i * numHeads + h) * headDim, k, v, kvHead, headDim, i + 1,
        stats, scale, localWindow, deltaMax, counter,
      );
      out.set(r.out, (i * numHeads + h) * headDim);
      if (r.bound > errorBound) errorBound = r.bound;
    }
  }
  return {
    output: out,
    ops: counter.ops,
    errorBound,
    droppedMassBound: 0,
    exactOps: numHeads * headDim * seqLen * (seqLen + 1),
  };
}

/**
 * Theorem A2 for a whole prefill: same layout as
 * prefillLandmarkAttention, but each row screens distant blocks by
 * their rigorous mass upper bound and scores only the survivors
 * exactly. errorBound / droppedMassBound are maxima over rows/heads.
 */
export function prefillSparseAttention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  options: SparsePrefillOptions = {},
): NovelAttentionResult {
  const seqLen = q.length / (numHeads * headDim);
  if (!Number.isInteger(seqLen) || seqLen === 0) {
    throw new Error(`prefillSparseAttention: q length ${q.length} is not a whole number of query rows`);
  }
  const kvLen = k.length / (numKvHeads * headDim);
  if (kvLen !== seqLen) {
    throw new Error(`prefillSparseAttention: q rows ${seqLen} != k rows ${kvLen}`);
  }
  if (v.length !== k.length) {
    throw new Error(`prefillSparseAttention: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(
      `prefillSparseAttention: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`,
    );
  }
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const localWindow = options.localWindow ?? 16;
  validateWindow(localWindow);
  const dropThreshold = options.dropThreshold ?? 1e-3;
  const counter: OpCounter = { ops: 0 };
  const stats = resolveStats(k, v, numKvHeads, headDim, seqLen, options, counter);
  const groupSize = numHeads / numKvHeads;
  const out = new Float32Array(seqLen * numHeads * headDim);
  let errorBound = 0;
  let droppedMass = 0;
  for (let i = 0; i < seqLen; i++) {
    for (let h = 0; h < numHeads; h++) {
      const kvHead = Math.floor(h / groupSize);
      const r = sparseHead(
        q, (i * numHeads + h) * headDim, k, v, kvHead, headDim, i + 1,
        stats, scale, localWindow, dropThreshold, counter,
      );
      out.set(r.out, (i * numHeads + h) * headDim);
      if (r.bound > errorBound) errorBound = r.bound;
      if (r.droppedMass > droppedMass) droppedMass = r.droppedMass;
    }
  }
  return {
    output: out,
    ops: counter.ops,
    errorBound,
    droppedMassBound: droppedMass,
    exactOps: numHeads * headDim * seqLen * (seqLen + 1),
  };
}
