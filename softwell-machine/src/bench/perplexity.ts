/**
 * src/bench/perplexity.ts — teacher-forced perplexity helpers.
 *
 * Pure math over a logits tensor: no model, no I/O. The eval script
 * (scripts/bench-sw-smollm.ts) runs one prefill forward per text chunk
 * and scores token i+1 under the logits at position i; these helpers
 * turn those logits into a negative log-likelihood.
 */

/** Negative log-likelihood (natural log) of `target` under one logits row. */
export function rowNll(
  logits: Float32Array,
  rowStart: number,
  vocabSize: number,
  target: number,
): number {
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[rowStart + i] as number;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < vocabSize; i++) {
    sum += Math.exp((logits[rowStart + i] as number) - max);
  }
  return Math.log(sum) + max - (logits[rowStart + target] as number);
}

/**
 * Total NLL of `targets[i]` under logits row i, for i in [0, targets.length).
 * The caller owns the offset convention: for a prefill of token ids
 * `t[0..n-1]`, pass rows 0..n-2 of the logits and targets `t[1..n-1]`.
 */
export function totalNll(logits: Float32Array, vocabSize: number, targets: number[]): number {
  let total = 0;
  for (let i = 0; i < targets.length; i++) {
    total += rowNll(logits, i * vocabSize, vocabSize, targets[i] as number);
  }
  return total;
}

/** Perplexity from a mean per-token NLL. */
export function pplFromMeanNll(meanNll: number): number {
  return Math.exp(meanNll);
}
