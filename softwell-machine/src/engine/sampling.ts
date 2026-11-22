/**
 * softwell-machine — token sampling.
 *
 * Greedy / temperature / top-k / top-p (nucleus) sampling over a logits
 * vector. Sampling uses a seeded PRNG (mulberry32) when params.seed is
 * set, so runs are reproducible; otherwise Math.random.
 */

import type { Sampler, SamplingParams } from "../contracts.ts";

/** Deterministic PRNG (mulberry32). Returns floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Argmax of a logits vector. */
export function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

/**
 * Sample one token id from `logits` under `params`.
 * temperature <= 0 (or NaN) is greedy argmax. Otherwise: temperature
 * scaling, top-k filtering (topK > 0), top-p filtering (topP < 1), then a
 * single categorical draw from the renormalized distribution.
 */
export function sampleLogits(logits: Float32Array, params: SamplingParams): number {
  if (logits.length === 0) throw new Error("sampleLogits: empty logits");
  const temperature = params.temperature;
  if (!(temperature > 0)) return argmax(logits);

  // Work on a copy so the caller's logits are untouched.
  const scaled = new Float32Array(logits.length);
  const invT = 1 / temperature;
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] * invT;

  // Top-k: keep the k largest logits, mask the rest.
  if (params.topK > 0 && params.topK < scaled.length) {
    const sorted = Float32Array.from(scaled).sort((a, b) => b - a);
    const threshold = sorted[params.topK - 1];
    // Count how many are strictly above the threshold so ties at the
    // boundary are all kept (at most k survive in the common case).
    let kept = 0;
    for (let i = 0; i < scaled.length; i++) {
      if (scaled[i] < threshold || (scaled[i] === threshold && kept >= params.topK)) {
        scaled[i] = -Infinity;
      } else {
        kept++;
      }
    }
  }

  // Softmax (stable) over the surviving logits.
  let max = -Infinity;
  for (let i = 0; i < scaled.length; i++) if (scaled[i] > max) max = scaled[i];
  const probs = new Float32Array(scaled.length);
  let sum = 0;
  for (let i = 0; i < scaled.length; i++) {
    probs[i] = scaled[i] === -Infinity ? 0 : Math.exp(scaled[i] - max);
    sum += probs[i];
  }
  const invSum = 1 / sum;
  for (let i = 0; i < probs.length; i++) probs[i] *= invSum;

  // Top-p (nucleus): sort by probability desc, keep the smallest set whose
  // cumulative probability reaches topP (always keep at least the top token).
  if (params.topP < 1) {
    const order = Array.from({ length: probs.length }, (_, i) => i);
    order.sort((a, b) => probs[b] - probs[a]);
    let cumulative = 0;
    const keep = new Set<number>();
    for (const idx of order) {
      keep.add(idx);
      cumulative += probs[idx];
      if (cumulative >= params.topP) break;
    }
    let keptSum = 0;
    for (let i = 0; i < probs.length; i++) {
      if (!keep.has(i)) probs[i] = 0;
      keptSum += probs[i];
    }
    const inv = 1 / keptSum;
    for (let i = 0; i < probs.length; i++) probs[i] *= inv;
  }

  // Categorical draw.
  const rng = params.seed !== undefined ? mulberry32(params.seed) : Math.random;
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1; // floating-point fallback: last non-zero token
}

/** Sampler that always takes the argmax, ignoring temperature/top-k/top-p. */
export const greedySampler: Sampler = (logits: Float32Array, _params: SamplingParams): number => {
  return argmax(logits);
};

export const defaultSampler: Sampler = sampleLogits;
