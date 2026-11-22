/**
 * softwell-machine — engine math (reference backend).
 *
 * Pure math, no I/O. Everything operates on Float32Array, row-major.
 * Correctness first: this is the reference implementation other backends
 * (cuda-fp8, metal, ...) are checked against.
 */

/** RMSNorm of a single row: x / sqrt(mean(x^2) + eps) * weight. */
export function rmsNorm(x: Float32Array, weight: Float32Array, eps: number): Float32Array {
  const n = x.length;
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
  const inv = 1 / Math.sqrt(sumSq / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * weight[i];
  return out;
}

/**
 * Row-major matmul C = A @ B.
 * A is [aRows, aCols], B is [bRows, bCols], aCols === bRows.
 * Loop order i,k,j keeps the inner loop over contiguous B rows.
 */
export function matmul(
  a: Float32Array, aRows: number, aCols: number,
  b: Float32Array, bRows: number, bCols: number,
): Float32Array {
  if (aCols !== bRows) {
    throw new Error(`matmul shape mismatch: A is [${aRows},${aCols}], B is [${bRows},${bCols}]`);
  }
  const c = new Float32Array(aRows * bCols);
  for (let i = 0; i < aRows; i++) {
    const cRow = i * bCols;
    const aRow = i * aCols;
    for (let k = 0; k < aCols; k++) {
      const aik = a[aRow + k];
      if (aik === 0) continue;
      const bRow = k * bCols;
      for (let j = 0; j < bCols; j++) {
        c[cRow + j] += aik * b[bRow + j];
      }
    }
  }
  return c;
}

/**
 * Row-major matmul C = A @ B^T, where B is stored [bRows, bCols] and
 * aCols === bCols. This is the shape weights arrive in from safetensors
 * ([out, in]): logits = hidden @ lm_head^T, q = hidden @ Wq^T, etc.
 * Loop order i,j,k keeps the inner loop over contiguous rows of A and B.
 */
export function matmulTransposeB(
  a: Float32Array, aRows: number, aCols: number,
  b: Float32Array, bRows: number, bCols: number,
): Float32Array {
  if (aCols !== bCols) {
    throw new Error(`matmulTransposeB shape mismatch: A is [${aRows},${aCols}], B is [${bRows},${bCols}]`);
  }
  const c = new Float32Array(aRows * bRows);
  for (let i = 0; i < aRows; i++) {
    const aRow = i * aCols;
    const cRow = i * bRows;
    for (let j = 0; j < bRows; j++) {
      const bRow = j * bCols;
      let sum = 0;
      for (let k = 0; k < aCols; k++) sum += a[aRow + k] * b[bRow + k];
      c[cRow + j] = sum;
    }
  }
  return c;
}

export interface RopeTables {
  /** cos table, [numPositions, headDim/2] flattened row-major. */
  cos: Float32Array;
  /** sin table, [numPositions, headDim/2] flattened row-major. */
  sin: Float32Array;
  headDim: number;
}

/**
 * RoPE cos/sin tables for the given absolute positions.
 * Frequencies follow the Llama convention: theta^(-2i/headDim).
 */
export function rope(positions: number[], headDim: number, theta: number): RopeTables {
  if (headDim % 2 !== 0) throw new Error(`rope: headDim must be even, got ${headDim}`);
  const half = headDim / 2;
  const cos = new Float32Array(positions.length * half);
  const sin = new Float32Array(positions.length * half);
  for (let p = 0; p < positions.length; p++) {
    const pos = positions[p];
    for (let i = 0; i < half; i++) {
      const freq = Math.pow(theta, -2 * i / headDim);
      const angle = pos * freq;
      cos[p * half + i] = Math.cos(angle);
      sin[p * half + i] = Math.sin(angle);
    }
  }
  return { cos, sin, headDim };
}

/**
 * Apply RoPE in place to q and k.
 *
 * q is [rows, numHeads, headDim] and k is [rows, numKvHeads, headDim]
 * (both row-major, flattened); `positions` has one absolute position per
 * row. numKvHeads may be < numHeads (GQA). Rotation uses interleaved pairs
 * (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos), the HF Llama convention.
 * Rotating by position 0 is the identity.
 */
export function applyRope(
  q: Float32Array,
  k: Float32Array,
  positions: number[],
  numHeads: number,
  headDim: number,
  theta: number = 10000,
  numKvHeads: number = numHeads,
): void {
  const rows = positions.length;
  if (q.length !== rows * numHeads * headDim) {
    throw new Error(
      `applyRope shape mismatch: q length ${q.length}, expected ${rows * numHeads * headDim}`,
    );
  }
  if (k.length !== rows * numKvHeads * headDim) {
    throw new Error(
      `applyRope shape mismatch: k length ${k.length}, expected ${rows * numKvHeads * headDim}`,
    );
  }
  const half = headDim / 2;
  const rotate = (x: Float32Array, rowSize: number, heads: number): void => {
    for (let r = 0; r < rows; r++) {
      const pos = positions[r];
      for (let h = 0; h < heads; h++) {
        const base = r * rowSize + h * headDim;
        for (let i = 0; i < half; i++) {
          const freq = Math.pow(theta, -2 * i / headDim);
          const angle = pos * freq;
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          const i0 = base + 2 * i;
          const i1 = i0 + 1;
          const x0 = x[i0];
          const x1 = x[i1];
          x[i0] = x0 * c - x1 * s;
          x[i1] = x0 * s + x1 * c;
        }
      }
    }
  };
  rotate(q, numHeads * headDim, numHeads);
  rotate(k, numKvHeads * headDim, numKvHeads);
}

/** Numerically stable softmax over a flat vector; returns a new array. */
export function softmax(x: Float32Array): Float32Array {
  const n = x.length;
  const out = new Float32Array(n);
  if (n === 0) return out;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(x[i] - max);
    sum += out[i];
  }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) out[i] *= inv;
  return out;
}

export interface GroupedQueryAttentionOptions {
  /** Mask out KV positions after the query position (default true). */
  causal?: boolean;
  /** Absolute position of the query token (needed for causal masking). */
  queryPosition?: number;
  /** Absolute positions of the KV entries, in order (needed for causal masking). */
  kvPositions?: number[];
  /** Attention scale; defaults to 1/sqrt(headDim). */
  scale?: number;
}

/**
 * Grouped-query attention for a single query token.
 *
 * q: [numHeads, headDim] (flattened). k, v: [kvLen, numKvHeads, headDim]
 * (flattened). numKvHeads may be < numHeads (GQA): each KV head is shared
 * by numHeads/numKvHeads query heads. Returns [numHeads, headDim].
 */
export function groupedQueryAttention(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  numHeads: number,
  numKvHeads: number,
  headDim: number,
  options: GroupedQueryAttentionOptions = {},
): Float32Array {
  const kvLen = k.length / (numKvHeads * headDim);
  if (!Number.isInteger(kvLen) || kvLen === 0) {
    throw new Error(`groupedQueryAttention: k length ${k.length} is not a whole number of KV rows`);
  }
  if (q.length !== numHeads * headDim) {
    throw new Error(`groupedQueryAttention: q length ${q.length}, expected ${numHeads * headDim}`);
  }
  if (v.length !== k.length) {
    throw new Error(`groupedQueryAttention: k/v length mismatch ${k.length} vs ${v.length}`);
  }
  if (numHeads % numKvHeads !== 0) {
    throw new Error(`groupedQueryAttention: numHeads ${numHeads} not divisible by numKvHeads ${numKvHeads}`);
  }
  const groupSize = numHeads / numKvHeads;
  const scale = options.scale ?? 1 / Math.sqrt(headDim);
  const causal = options.causal ?? true;
  const out = new Float32Array(numHeads * headDim);

  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / groupSize);
    const qOff = h * headDim;
    // Scores over KV positions.
    const scores = new Float32Array(kvLen);
    let max = -Infinity;
    for (let t = 0; t < kvLen; t++) {
      if (causal && options.kvPositions && options.queryPosition !== undefined) {
        if (options.kvPositions[t] > options.queryPosition) {
          scores[t] = -Infinity;
          continue;
        }
      }
      const kOff = (t * numKvHeads + kvHead) * headDim;
      let dot = 0;
      for (let d = 0; d < headDim; d++) dot += q[qOff + d] * k[kOff + d];
      scores[t] = dot * scale;
      if (scores[t] > max) max = scores[t];
    }
    // Stable softmax + weighted sum of V.
    let sum = 0;
    const weights = new Float32Array(kvLen);
    for (let t = 0; t < kvLen; t++) {
      weights[t] = scores[t] === -Infinity ? 0 : Math.exp(scores[t] - max);
      sum += weights[t];
    }
    const oOff = h * headDim;
    if (sum === 0) continue; // fully masked: leave zeros
    const inv = 1 / sum;
    for (let t = 0; t < kvLen; t++) {
      if (weights[t] === 0) continue;
      const w = weights[t] * inv;
      const vOff = (t * numKvHeads + kvHead) * headDim;
      for (let d = 0; d < headDim; d++) out[oOff + d] += w * v[vOff + d];
    }
  }
  return out;
}

/** SiLU (sigmoid linear unit): x * sigmoid(x). */
export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/**
 * SwiGLU MLP for a single row: down(silu(x @ gateW^T) * (x @ upW^T)).
 * gateW/upW are [intermediateSize, hiddenSize], downW is [hiddenSize, intermediateSize].
 */
export function swiglu(
  x: Float32Array,
  gateW: Float32Array,
  upW: Float32Array,
  downW: Float32Array,
): Float32Array {
  const hiddenSize = x.length;
  const intermediateSize = gateW.length / hiddenSize;
  if (!Number.isInteger(intermediateSize)) {
    throw new Error(`swiglu: gateW length ${gateW.length} not divisible by hiddenSize ${hiddenSize}`);
  }
  if (upW.length !== gateW.length || downW.length !== hiddenSize * intermediateSize) {
    throw new Error("swiglu: weight shape mismatch");
  }
  const gate = matmulTransposeB(x, 1, hiddenSize, gateW, intermediateSize, hiddenSize);
  const up = matmulTransposeB(x, 1, hiddenSize, upW, intermediateSize, hiddenSize);
  const hidden = new Float32Array(intermediateSize);
  for (let i = 0; i < intermediateSize; i++) hidden[i] = silu(gate[i]) * up[i];
  return matmulTransposeB(hidden, 1, intermediateSize, downW, hiddenSize, intermediateSize);
}

export interface MoeExpertWeights {
  gate: Float32Array;
  up: Float32Array;
  down: Float32Array;
}

/**
 * CPU fallback kernel library for the vGPU.
 *
 * The virtual-hardware layer (src/hw) calls these when a kernel has no
 * accelerated implementation on the current node. Stable names, plain
 * Float32Array in / out, no I/O — the same ops the reference transformer
 * backend is built from.
 */
export const CpuKernels = {
  matmul,
  rmsNorm,
  rope,
  softmax,
  swiglu,
  groupedQueryAttention,
  moeForward,
  // Extras the reference backend also relies on:
  applyRope,
  matmulTransposeB,
};

/**
 * MoE forward for a single row (Mixtral-style).
 * Router logits = x @ routerW^T (routerW is [numExperts, hiddenSize]),
 * softmaxed; the top-k experts run and their outputs are summed weighted
 * by router probability. When normTopKProb is set, the top-k probabilities
 * are renormalized to sum to 1 before weighting.
 * `experts` is parallel to `expertIds` (the global expert ids these
 * weights belong to); when omitted the experts are ids 0..N-1. A stage
 * that owns only a subset of experts routes among its own experts.
 */
export function moeForward(
  x: Float32Array,
  experts: MoeExpertWeights[],
  routerW: Float32Array,
  topK: number,
  normTopKProb: boolean = false,
  expertIds?: number[],
): Float32Array {
  const hiddenSize = x.length;
  const numExperts = routerW.length / hiddenSize;
  if (!Number.isInteger(numExperts)) {
    throw new Error(`moeForward: routerW length ${routerW.length} not divisible by hiddenSize ${hiddenSize}`);
  }
  const ids = expertIds ?? experts.map((_, i) => i);
  if (ids.length !== experts.length) {
    throw new Error(`moeForward: expertIds length ${ids.length} != experts length ${experts.length}`);
  }
  const k = Math.min(topK, ids.length);
  if (k <= 0) throw new Error("moeForward: topK must be >= 1");
  const routerLogits = matmulTransposeB(x, 1, hiddenSize, routerW, numExperts, hiddenSize);
  const routerProbs = softmax(routerLogits);
  // Pick top-k among the owned experts by router probability.
  const topIdx: number[] = [];
  const used = new Array<boolean>(ids.length).fill(false);
  for (let n = 0; n < k; n++) {
    let best = -1;
    let bestP = -Infinity;
    for (let j = 0; j < ids.length; j++) {
      if (!used[j] && routerProbs[ids[j]] > bestP) {
        bestP = routerProbs[ids[j]];
        best = j;
      }
    }
    used[best] = true;
    topIdx.push(best);
  }
  let weightSum = 0;
  for (const j of topIdx) weightSum += routerProbs[ids[j]];
  const out = new Float32Array(hiddenSize);
  for (const j of topIdx) {
    let w = routerProbs[ids[j]];
    if (normTopKProb) w /= weightSum;
    const expertOut = swiglu(x, experts[j].gate, experts[j].up, experts[j].down);
    for (let i = 0; i < hiddenSize; i++) out[i] += w * expertOut[i];
  }
  return out;
}
