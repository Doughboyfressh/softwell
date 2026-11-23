/**
 * softwell-machine — speculative / parallel token generation.
 *
 * Two new decoding schemes that live *alongside* the existing scheduler and
 * engine (nothing in src/scheduler or src/engine/transformer.ts is touched).
 * Both are wrappers around a StageExecutor: they call the same
 * `forward()` the autoregressive scheduler calls, but they arrange the rows
 * so that one forward pass can produce several tokens. The win is tokens
 * per forward pass, never FLOPS per token — a virtual machine cannot exceed
 * its host's physical limits; it can only do less work per accepted token.
 *
 * Schemes
 * -------
 * 1. Speculative decoding (Theorem S1, docs/research/03-decoding.md):
 *    a cheap drafter proposes K tokens, one batched target forward verifies
 *    them all, the longest matching prefix is kept plus one correction /
 *    bonus token. Greedy output is token-for-token identical to plain
 *    autoregressive decoding; sampled output is identically distributed
 *    (speculative sampling, Leviathan et al. 2023; Chen et al. 2023).
 *    There is no separate draft model in this repo, so the drafter is an
 *    n-gram / context-copy drafter (NGramDrafter): it copies continuations
 *    it has already seen in the prompt + generated text. It costs no
 *    forward pass at all — which is exactly why its acceptance rate, not
 *    its cost, decides whether speculation wins.
 * 2. Parallel multi-token / Jacobi decoding (Theorem S2): guess a whole
 *    block of K tokens at once, refine every position in parallel with one
 *    batched forward per iteration, and stop at the fixed point. The fixed
 *    point *is* the greedy autoregressive block, so output is again
 *    identical to plain decoding (greedy only).
 *
 * KV-cache contract (read this before wiring these decoders in)
 * ---------------------------------------------------------------
 * Both schemes feed tokens into the cache that later turn out to be wrong
 * (rejected drafts, intermediate Jacobi guesses). Correctness therefore
 * relies on one property of the cache: `append(seqId, layer, position, …)`
 * *overwrites* the K/V slot for that position, so feeding the correct token
 * at a position repairs the slot before any later forward reads it as part
 * of a prefix. BlockPagedKVCache (src/scheduler/kvcache.ts) has this
 * property — its append writes into the position's slot. An append-only
 * cache would NOT work: it would need a truncate/rollback primitive, or
 * the decoders would have to run verification against a throwaway cache
 * and copy the accepted prefix's K/V back. Every forward in this file
 * appends a row's K/V *before* reading the prefix that contains it (that
 * ordering is inside TransformerStageExecutor.forwardLayer), so a row
 * always attends to the correct token at its own position.
 *
 * Cost model
 * ----------
 * A batched forward with R rows is not free on a compute-bound CPU
 * backend: its cost is modelled as `1 + (R-1) * beta` single-row forwards,
 * where beta in [0, 1] is the marginal cost of one extra row
 * (beta ≈ 0 when the forward is memory-bandwidth-bound and weights are
 * streamed once per forward, beta ≈ 1 when it is compute-bound and every
 * row pays full matmul FLOPS). All speedup / breakeven functions take beta
 * explicitly; nothing in this file assumes a value for it.
 * A speculative round drafts K tokens. The first round's verification
 * forward has K rows (the drafts); every later round's has K+1 rows (the
 * not-yet-fed carry token plus K drafts). The cost model uses the
 * steady-state round: cost `1 + K * beta` per round. A hard consequence
 * worth stating up front: at beta = 1 the best possible speedup is 1
 * (K+1 tokens from K+1 rows of work, attained only at acceptance 1), so
 * speculation can only win wall-clock when beta < 1 or, equivalently,
 * when the hardware is bandwidth-bound enough that extra rows are cheap.
 */

import type {
  ForwardContext,
  PagedKVCache,
  SamplingParams,
  StageExecutor,
  Tensor,
} from "../contracts.ts";
import { argmax, mulberry32 } from "./sampling.ts";

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/** One batched forward: embed `tokenIds`, run the stage, return logits. */
export async function forwardTokenRows(
  executor: StageExecutor,
  cache: PagedKVCache,
  seqId: string,
  tokenIds: number[],
  positions: number[],
  isPrefill: boolean,
): Promise<Tensor> {
  const embedded = executor.embed(tokenIds);
  const ctx: ForwardContext = { seqIds: tokenIds.map(() => seqId), positions, isPrefill, cache };
  return executor.forward(embedded, ctx);
}

/** Copy of row `row` of a [rows, vocab] logits tensor. */
export function logitsRow(logits: Tensor, row: number, vocabSize: number): Float32Array {
  return Float32Array.from(logits.data.subarray(row * vocabSize, (row + 1) * vocabSize));
}

/** Numerically stable softmax over a logits vector. */
export function softmaxProbs(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.exp(logits[i] - max);
    sum += probs[i];
  }
  const inv = 1 / sum;
  for (let i = 0; i < probs.length; i++) probs[i] *= inv;
  return probs;
}

/** One categorical draw from `probs` (which must sum to ~1). */
export function sampleFromProbs(probs: Float32Array, rng: () => number): number {
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

function makeRng(seed: number | undefined): () => number {
  if (seed !== undefined) return mulberry32(seed);
  return mulberry32((Math.random() * 4294967296) >>> 0);
}

/** The target model's own token at one position: argmax, or a seeded draw. */
export function sampleTarget(logits: Float32Array, params: SamplingParams, rng: () => number): number {
  if (!(params.temperature > 0)) return argmax(logits);
  return sampleFromProbs(softmaxProbs(logits), rng);
}

function assertPositiveInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`novel-decoding: ${name} must be a positive integer, got ${value}`);
  }
}

function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Cost model (pure functions — see docs/research/03-decoding.md, §S1.3/S2.3)
// ---------------------------------------------------------------------------

/**
 * Expected tokens produced by one speculative round with draft length K
 * when each draft token is independently accepted with probability alpha.
 * One correction/bonus token is always produced, plus the accepted prefix:
 * E = 1 + alpha + alpha^2 + … + alpha^K. This holds for every round: the
 * first round verifies K drafts in K rows, and every later round verifies
 * K drafts in K+1 rows (carry token + drafts) — the draft count, and hence
 * this expectation, is the same either way.
 * (Independence is a modelling assumption — real acceptance is correlated
 * along the draft; see the research note for the correlated bound.)
 */
export function expectedSpeculativeTokensPerRound(alpha: number, draftLength: number): number {
  assertPositiveInt(draftLength, "draftLength");
  if (!(alpha >= 0 && alpha <= 1)) throw new Error(`expectedSpeculativeTokensPerRound: alpha ${alpha} not in [0, 1]`);
  let sum = 1;
  let term = 1;
  for (let i = 1; i <= draftLength; i++) {
    term *= alpha;
    sum += term;
  }
  return sum;
}

/** Expected target forwards per accepted token: 1 / E[tokens per round]. */
export function expectedForwardsPerToken(alpha: number, draftLength: number): number {
  return 1 / expectedSpeculativeTokensPerRound(alpha, draftLength);
}

/**
 * Expected wall-clock speedup of speculative decoding over plain
 * autoregressive decoding, as a multiple of baseline tokens/sec.
 * A steady-state verification forward has K+1 rows (carry + K drafts),
 * costing `1 + K * beta` single-row forwards (beta = marginal cost of one
 * extra row, 0..1). Drafting itself is free (n-gram lookup, no forward).
 * Speedup = E[tokens/round] / cost. beta = 0 is the memory-bandwidth-bound
 * GPU regime (verification ~free); beta = 1 is the compute-bound CPU
 * regime (a K+1-row forward costs K+1). At beta = 1 the speedup is at
 * most 1, attained only at alpha = 1 — see the module header.
 */
export function speculativeSpeedup(alpha: number, draftLength: number, rowCostBeta: number): number {
  if (!(rowCostBeta >= 0 && rowCostBeta <= 1)) {
    throw new Error(`speculativeSpeedup: rowCostBeta ${rowCostBeta} not in [0, 1]`);
  }
  const cost = 1 + draftLength * rowCostBeta;
  return expectedSpeculativeTokensPerRound(alpha, draftLength) / cost;
}

/**
 * Breakeven acceptance rate: the smallest alpha with speculativeSpeedup >= 1,
 * or null if no alpha in [0, 1] breaks even (impossible for beta <= 1, but
 * kept total for callers that pass a hypothetical beta > 1). Found by
 * bisection; E[tokens] is monotone increasing in alpha.
 */
export function breakevenAcceptance(draftLength: number, rowCostBeta: number): number | null {
  assertPositiveInt(draftLength, "draftLength");
  if (speculativeSpeedup(1, draftLength, rowCostBeta) < 1) return null;
  if (speculativeSpeedup(0, draftLength, rowCostBeta) >= 1) return 0;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 64; i++) {
    const mid = (lo + hi) / 2;
    if (speculativeSpeedup(mid, draftLength, rowCostBeta) >= 1) hi = mid;
    else lo = mid;
  }
  return hi;
}

/**
 * Expected forwards per K-token Jacobi block when each initial guess is
 * independently correct with probability gamma. If J is the first wrong
 * guess index (J = K when all are right), the block takes K - J + 1
 * forwards (Theorem S2), so E[T] = sum_j (K-j+1) * P(J=j).
 */
export function expectedJacobiForwardsPerBlock(gamma: number, blockSize: number): number {
  assertPositiveInt(blockSize, "blockSize");
  if (!(gamma >= 0 && gamma <= 1)) throw new Error(`expectedJacobiForwardsPerBlock: gamma ${gamma} not in [0, 1]`);
  let expected = 0;
  let probAllCorrectSoFar = 1;
  for (let j = 0; j < blockSize; j++) {
    // P(J = j) = gamma^j * (1 - gamma); T = K - j + 1.
    expected += (blockSize - j + 1) * probAllCorrectSoFar * (1 - gamma);
    probAllCorrectSoFar *= gamma;
  }
  expected += 1 * probAllCorrectSoFar; // P(J = K) = gamma^K; T = 1.
  return expected;
}

/**
 * Expected wall-clock speedup of Jacobi decoding over plain decoding.
 * Baseline: K forwards for K tokens. Jacobi: E[T] forwards, each a K-row
 * forward costing `1 + (K-1) * beta`. Speedup = K / (E[T] * cost).
 */
export function jacobiSpeedup(gamma: number, blockSize: number, rowCostBeta: number): number {
  if (!(rowCostBeta >= 0 && rowCostBeta <= 1)) {
    throw new Error(`jacobiSpeedup: rowCostBeta ${rowCostBeta} not in [0, 1]`);
  }
  const costPerForward = 1 + (blockSize - 1) * rowCostBeta;
  return blockSize / (expectedJacobiForwardsPerBlock(gamma, blockSize) * costPerForward);
}

/** Smallest gamma with jacobiSpeedup >= 1, or null if none breaks even. */
export function breakevenJacobiGamma(blockSize: number, rowCostBeta: number): number | null {
  assertPositiveInt(blockSize, "blockSize");
  if (jacobiSpeedup(1, blockSize, rowCostBeta) < 1) return null;
  if (jacobiSpeedup(0, blockSize, rowCostBeta) >= 1) return 0;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 64; i++) {
    const mid = (lo + hi) / 2;
    if (jacobiSpeedup(mid, blockSize, rowCostBeta) >= 1) hi = mid;
    else lo = mid;
  }
  return hi;
}

// ---------------------------------------------------------------------------
// Drafters
// ---------------------------------------------------------------------------

/** A drafter proposes `count` continuation tokens for `context` (prompt + generated so far). It must be cheap: no target forward. */
export interface Drafter {
  draft(context: number[], count: number): number[];
}

/**
 * N-gram / context-copy drafter.
 *
 * To propose the next token after a context, find the longest suffix of
 * the context (up to `maxN` tokens) that has occurred before in the
 * context, and copy the token that followed its most recent occurrence.
 * Fall back to repeating the last token when no suffix of length >= 1 has
 * ever occurred before (e.g. at the very start of a generation). Drafting
 * is sequential: each proposed token extends the working context for the
 * next proposal. Cost is O(count * maxN * contextLength) integer compares
 * — no forward pass, no learned weights.
 *
 * This is the drafter the demo model gets: there is no trained draft model
 * in the repo, and a random-weight target has almost no predictable
 * continuations, so its acceptance rate is expected to be low. That is a
 * property of the (untrained) model, not of the decoding scheme — the
 * scheme is exact regardless, and the acceptance rate is measured, not
 * assumed (see test/novel-decoding.test.ts).
 */
export class NGramDrafter implements Drafter {
  readonly maxN: number;

  constructor(maxN = 8) {
    if (!Number.isInteger(maxN) || maxN < 1) {
      throw new Error(`NGramDrafter: maxN must be a positive integer, got ${maxN}`);
    }
    this.maxN = maxN;
  }

  draft(context: number[], count: number): number[] {
    if (count <= 0) return [];
    const working = [...context];
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
      const next = this.nextToken(working);
      out.push(next);
      working.push(next);
    }
    return out;
  }

  /** Longest-suffix copy; most recent occurrence wins; else repeat last. */
  nextToken(context: number[]): number {
    if (context.length === 0) throw new Error("NGramDrafter: empty context");
    const last = context[context.length - 1] as number;
    const maxN = Math.min(this.maxN, context.length - 1);
    for (let n = maxN; n >= 1; n--) {
      const suffixStart = context.length - n;
      // Most recent earlier occurrence first (i + n < context.length so a
      // following token exists).
      for (let i = context.length - n - 1; i >= 0; i--) {
        let match = true;
        for (let j = 0; j < n; j++) {
          if (context[i + j] !== context[suffixStart + j]) {
            match = false;
            break;
          }
        }
        if (match) return context[i + n] as number;
      }
    }
    return last;
  }
}

// ---------------------------------------------------------------------------
// Shared result / options shapes
// ---------------------------------------------------------------------------

export interface GenerateResult {
  tokens: number[];
  finishReason: "eos" | "length";
  /** Target forward() calls (prefill + every decode/verification forward). */
  forwards: number;
  /** Total rows fed to forward() (prefill rows + decode/verification rows). */
  rowsProcessed: number;
  /** Speculative/Jacobi rounds (0 for the plain baseline). */
  rounds: number;
  /** Speculative only: draft tokens proposed / accepted. */
  draftProposed: number;
  draftAccepted: number;
  /** Speculative only: draftAccepted / draftProposed (0 when none proposed). */
  acceptanceRate: number;
  /** Tokens per target forward (forwards counted, not FLOPS). */
  tokensPerForward: number;
  /** Jacobi only: total Jacobi update iterations (excluding commit forwards). */
  iterations: number;
  /** Jacobi only: blocks attempted. */
  blocks: number;
}

function emptyResult(): GenerateResult {
  return {
    tokens: [],
    finishReason: "length",
    forwards: 0,
    rowsProcessed: 0,
    rounds: 0,
    draftProposed: 0,
    draftAccepted: 0,
    acceptanceRate: 0,
    tokensPerForward: 0,
    iterations: 0,
    blocks: 0,
  };
}

export interface DecoderOptions {
  executor: StageExecutor;
  cache: PagedKVCache;
  seqId: string;
  promptTokenIds: number[];
  sampling: SamplingParams;
  vocabSize: number;
  eosTokenId: number;
}

function validateOptions(opts: DecoderOptions): void {
  if (opts.promptTokenIds.length === 0) throw new Error("novel-decoding: promptTokenIds must not be empty");
  if (!Number.isInteger(opts.vocabSize) || opts.vocabSize < 1) {
    throw new Error(`novel-decoding: vocabSize must be a positive integer, got ${opts.vocabSize}`);
  }
  for (const t of opts.promptTokenIds) {
    if (!Number.isInteger(t) || t < 0 || t >= opts.vocabSize) {
      throw new Error(`novel-decoding: prompt token ${t} out of vocab range [0, ${opts.vocabSize})`);
    }
  }
}

/**
 * Speculative sampling, one draft token (Leviathan et al. 2023; Chen et al.
 * 2023), specialised here to a *deterministic* drafter (draft distribution
 * q is a point mass at `draftToken`, so q(draftToken) = 1):
 * accept with probability min(1, p(draftToken)); on rejection, resample
 * from the residual norm(max(0, p - q)) — i.e. p with the draft token
 * zeroed out and renormalised. Greedy (temperature <= 0) is the special
 * case: accept iff draftToken is the argmax, else the argmax is the
 * correction token. In both cases the returned token is distributed
 * exactly as the target model's own token at this position (proof in
 * docs/research/03-decoding.md, Theorem S1).
 */
function verifyOneDraft(
  predLogits: Float32Array,
  draftToken: number,
  params: SamplingParams,
  rng: () => number,
): { accepted: boolean; token: number } {
  if (!(params.temperature > 0)) {
    const target = argmax(predLogits);
    return { accepted: draftToken === target, token: target };
  }
  const probs = softmaxProbs(predLogits);
  const pDraft = probs[draftToken] as number;
  if (rng() < pDraft) return { accepted: true, token: draftToken };
  // Residual: p with the draft token removed, renormalised.
  const residual = Float32Array.from(probs);
  residual[draftToken] = 0;
  let sum = 0;
  for (let i = 0; i < residual.length; i++) sum += residual[i];
  if (!(sum > 0)) {
    // p(draftToken) was 1 (up to float error) yet rejection happened; fall
    // back to the draft token rather than dividing by ~0. Unreachable in
    // exact arithmetic because then rng() < 1 always accepts.
    return { accepted: false, token: draftToken };
  }
  const inv = 1 / sum;
  for (let i = 0; i < residual.length; i++) residual[i] *= inv;
  return { accepted: false, token: sampleFromProbs(residual, rng) };
}

// ---------------------------------------------------------------------------
// Baseline: plain autoregressive decoding (the reference the new schemes
// must match token-for-token under greedy sampling).
// ---------------------------------------------------------------------------

export class AutoregressiveDecoder {
  private readonly opts: DecoderOptions;

  constructor(opts: DecoderOptions) {
    validateOptions(opts);
    this.opts = opts;
  }

  async generate(maxTokens?: number): Promise<GenerateResult> {
    const { executor, cache, seqId, promptTokenIds, sampling, vocabSize, eosTokenId } = this.opts;
    const limit = maxTokens ?? sampling.maxTokens;
    const result = emptyResult();
    if (limit <= 0) return result;
    const rng = makeRng(sampling.seed);
    await cache.free(seqId);
    const promptLen = promptTokenIds.length;
    const positions = promptTokenIds.map((_, i) => i);
    const prefill = await forwardTokenRows(executor, cache, seqId, promptTokenIds, positions, true);
    result.forwards += 1;
    result.rowsProcessed += promptLen;
    let pending = logitsRow(prefill, promptLen - 1, vocabSize);
    while (result.tokens.length < limit) {
      const tok = sampleTarget(pending, sampling, rng);
      result.tokens.push(tok);
      if (tok === eosTokenId) {
        result.finishReason = "eos";
        break;
      }
      if (result.tokens.length >= limit) break;
      const pos = promptLen + result.tokens.length - 1;
      const out = await forwardTokenRows(executor, cache, seqId, [tok], [pos], false);
      result.forwards += 1;
      result.rowsProcessed += 1;
      pending = logitsRow(out, 0, vocabSize);
    }
    result.finishReason = result.tokens[result.tokens.length - 1] === eosTokenId ? "eos" : "length";
    result.tokensPerForward = result.forwards > 0 ? result.tokens.length / result.forwards : 0;
    return result;
  }
}

// ---------------------------------------------------------------------------
// Speculative decoding (Theorem S1)
// ---------------------------------------------------------------------------

export interface SpeculativeDecoderOptions extends DecoderOptions {
  /** Draft length K (tokens proposed per round). Default 4. */
  draftLength?: number;
  /** Drafter. Default: NGramDrafter(8). */
  drafter?: Drafter;
}

export class SpeculativeDecoder {
  private readonly opts: DecoderOptions;
  private readonly draftLength: number;
  private readonly drafter: Drafter;

  constructor(opts: SpeculativeDecoderOptions) {
    validateOptions(opts);
    assertPositiveInt(opts.draftLength ?? 4, "draftLength");
    this.opts = opts;
    this.draftLength = opts.draftLength ?? 4;
    this.drafter = opts.drafter ?? new NGramDrafter(8);
  }

  async generate(maxTokens?: number): Promise<GenerateResult> {
    const { executor, cache, seqId, promptTokenIds, sampling, vocabSize, eosTokenId } = this.opts;
    const limit = maxTokens ?? sampling.maxTokens;
    const result = emptyResult();
    if (limit <= 0) return result;
    const rng = makeRng(sampling.seed);
    const promptLen = promptTokenIds.length;
    await cache.free(seqId);

    // Prefill: logits for the first generated token.
    const prefill = await forwardTokenRows(
      executor,
      cache,
      seqId,
      promptTokenIds,
      promptTokenIds.map((_, i) => i),
      true,
    );
    result.forwards += 1;
    result.rowsProcessed += promptLen;
    let pending: Float32Array | null = logitsRow(prefill, promptLen - 1, vocabSize);
    // The last accepted token, not yet fed into the model (its K/V is not
    // in the cache). It is fed as row 0 of the next round's verification
    // forward. Null until the first round has produced a token.
    let carry: { token: number; pos: number } | null = null;

    // Append one token; returns false when generation must stop (EOS or
    // maxTokens reached).
    const pushToken = (tok: number): boolean => {
      result.tokens.push(tok);
      if (tok === eosTokenId) {
        result.finishReason = "eos";
        return false;
      }
      if (result.tokens.length >= limit) {
        result.finishReason = "length";
        return false;
      }
      return true;
    };
    const context = (): number[] => [...promptTokenIds, ...result.tokens];

    while (result.tokens.length < limit) {
      const remaining = limit - result.tokens.length;
      result.rounds += 1;

      if (pending !== null) {
        // ---- Round type A: pending logits predict the next token. ----
        const nextPos = promptLen + result.tokens.length;
        const k = Math.min(this.draftLength, remaining);
        const draft = this.drafter.draft(context(), k);
        if (draft.length === 0) {
          // Empty draft: plain autoregressive step (edge case, e.g. a
          // drafter that declines to propose). Exact by construction.
          if (!pushToken(sampleTarget(pending, sampling, rng))) break;
          const pos = promptLen + result.tokens.length - 1;
          const out = await forwardTokenRows(executor, cache, seqId, [result.tokens[result.tokens.length - 1] as number], [pos], false);
          result.forwards += 1;
          result.rowsProcessed += 1;
          pending = logitsRow(out, 0, vocabSize);
          continue;
        }
        const positions = draft.map((_, i) => nextPos + i);
        const out = await forwardTokenRows(executor, cache, seqId, draft, positions, false);
        result.forwards += 1;
        result.rowsProcessed += draft.length;
        // Verify draft[i] against the logits that predict it: pending for
        // i = 0, verification row i-1 for i >= 1.
        let accepted = 0;
        let endToken: number | null = null;
        let allAccepted = true;
        for (let i = 0; i < draft.length; i++) {
          const pred = i === 0 ? (pending as Float32Array) : logitsRow(out, i - 1, vocabSize);
          const v = verifyOneDraft(pred, draft[i] as number, sampling, rng);
          result.draftProposed += 1;
          if (v.accepted) {
            result.draftAccepted += 1;
            accepted += 1;
          } else {
            endToken = v.token;
            allAccepted = false;
            break;
          }
        }
        let stopped = false;
        for (let i = 0; i < accepted; i++) {
          if (!pushToken(draft[i] as number)) {
            stopped = true;
            break;
          }
        }
        if (stopped) break;
        if (allAccepted) {
          // Bonus token from the last verification row (predicts the token
          // after the last draft). Only if room remains.
          if (result.tokens.length < limit) {
            const bonus = sampleTarget(logitsRow(out, draft.length - 1, vocabSize), sampling, rng);
            if (!pushToken(bonus)) break;
            carry = { token: bonus, pos: nextPos + draft.length };
          }
          pending = null;
        } else {
          // Correction token at the first rejected position.
          const correction = endToken as number;
          if (!pushToken(correction)) break;
          carry = { token: correction, pos: nextPos + accepted };
          pending = null;
        }
      } else {
        // ---- Round type B: `carry` was accepted but never fed. Feed it
        // as row 0; its logits verify the drafts that follow it. ----
        const carried: { token: number; pos: number } = carry as { token: number; pos: number };
        const carryToken = carried.token;
        const carryPos = carried.pos;
        const nextDraftPos = carryPos + 1;
        // Draft the full K tokens after the carry (rows = 1 carry + K
        // drafts), so every round drafts the same number of tokens.
        const k = Math.min(this.draftLength, remaining);
        const draft = this.drafter.draft(context(), k);
        const rows = [carryToken, ...draft];
        const positions = rows.map((_, i) => carryPos + i);
        const out = await forwardTokenRows(executor, cache, seqId, rows, positions, false);
        result.forwards += 1;
        result.rowsProcessed += rows.length;
        if (draft.length === 0) {
          // Nothing drafted after the carry: plain step from its logits.
          const tok = sampleTarget(logitsRow(out, 0, vocabSize), sampling, rng);
          if (!pushToken(tok)) break;
          carry = { token: tok, pos: carryPos + 1 };
          pending = null;
          continue;
        }
        // Draft[i] sits at nextDraftPos + i and is predicted by row i
        // (row i is the forward of the token at carryPos + i).
        let accepted = 0;
        let endToken: number | null = null;
        let allAccepted = true;
        for (let i = 0; i < draft.length; i++) {
          const pred = logitsRow(out, i, vocabSize);
          const v = verifyOneDraft(pred, draft[i] as number, sampling, rng);
          result.draftProposed += 1;
          if (v.accepted) {
            result.draftAccepted += 1;
            accepted += 1;
          } else {
            endToken = v.token;
            allAccepted = false;
            break;
          }
        }
        let stopped = false;
        for (let i = 0; i < accepted; i++) {
          if (!pushToken(draft[i] as number)) {
            stopped = true;
            break;
          }
        }
        if (stopped) break;
        if (allAccepted) {
          if (result.tokens.length < limit) {
            const bonus = sampleTarget(logitsRow(out, draft.length, vocabSize), sampling, rng);
            if (!pushToken(bonus)) break;
            carry = { token: bonus, pos: nextDraftPos + draft.length };
          }
          pending = null;
        } else {
          const correction = endToken as number;
          if (!pushToken(correction)) break;
          carry = { token: correction, pos: nextDraftPos + accepted };
          pending = null;
        }
      }
    }

    if (result.tokens.length === 0) result.finishReason = "length";
    else if (result.tokens[result.tokens.length - 1] !== eosTokenId && result.finishReason !== "eos") {
      result.finishReason = "length";
    }
    result.acceptanceRate =
      result.draftProposed > 0 ? result.draftAccepted / result.draftProposed : 0;
    result.tokensPerForward = result.forwards > 0 ? result.tokens.length / result.forwards : 0;
    return result;
  }
}

// ---------------------------------------------------------------------------
// Jacobi / parallel multi-token decoding (Theorem S2, greedy only)
// ---------------------------------------------------------------------------

export interface JacobiDecoderOptions extends DecoderOptions {
  /** Block size K (tokens guessed per block). Default 4. */
  blockSize?: number;
  /** Guesser for the initial block. Default: NGramDrafter(8). */
  drafter?: Drafter;
}

export class JacobiDecoder {
  private readonly opts: DecoderOptions;
  private readonly blockSize: number;
  private readonly drafter: Drafter;

  constructor(opts: JacobiDecoderOptions) {
    validateOptions(opts);
    if (opts.sampling.temperature > 0) {
      throw new Error(
        "JacobiDecoder: only greedy sampling (temperature 0) is supported — " +
          "a Jacobi fixed point is defined by argmax, not by a sampled draw",
      );
    }
    assertPositiveInt(opts.blockSize ?? 4, "blockSize");
    this.opts = opts;
    this.blockSize = opts.blockSize ?? 4;
    this.drafter = opts.drafter ?? new NGramDrafter(8);
  }

  async generate(maxTokens?: number): Promise<GenerateResult> {
    const { executor, cache, seqId, promptTokenIds, vocabSize, eosTokenId } = this.opts;
    const limit = maxTokens ?? this.opts.sampling.maxTokens;
    const result = emptyResult();
    if (limit <= 0) return result;
    const promptLen = promptTokenIds.length;
    await cache.free(seqId);

    const prefill = await forwardTokenRows(
      executor,
      cache,
      seqId,
      promptTokenIds,
      promptTokenIds.map((_, i) => i),
      true,
    );
    result.forwards += 1;
    result.rowsProcessed += promptLen;
    let pending = logitsRow(prefill, promptLen - 1, vocabSize);

    const pushToken = (tok: number): boolean => {
      result.tokens.push(tok);
      if (tok === eosTokenId) {
        result.finishReason = "eos";
        return false;
      }
      if (result.tokens.length >= limit) {
        result.finishReason = "length";
        return false;
      }
      return true;
    };

    while (result.tokens.length < limit) {
      const remaining = limit - result.tokens.length;
      const blockLen = Math.min(this.blockSize, remaining);
      const nextPos = promptLen + result.tokens.length;

      if (blockLen === 1) {
        // A 1-token block is plain autoregressive decoding; short-circuit
        // it instead of paying a Jacobi iteration + commit forward for it.
        const tok = argmax(pending);
        if (!pushToken(tok)) break;
        if (result.tokens.length >= limit) break;
        const out = await forwardTokenRows(executor, cache, seqId, [tok], [nextPos], false);
        result.forwards += 1;
        result.rowsProcessed += 1;
        pending = logitsRow(out, 0, vocabSize);
        continue;
      }

      result.blocks += 1;
      // Initial guess: entirely from the (cheap, target-free) drafter.
      let guess = this.drafter.draft([...promptTokenIds, ...result.tokens], blockLen);
      while (guess.length < blockLen) {
        guess.push(guess.length > 0 ? (guess[guess.length - 1] as number) : (promptTokenIds[promptTokenIds.length - 1] as number));
      }
      guess = guess.slice(0, blockLen);

      let converged = false;
      let out: Tensor | null = null;
      for (let iter = 0; iter < blockLen; iter++) {
        result.iterations += 1;
        const positions = guess.map((_, i) => nextPos + i);
        out = await forwardTokenRows(executor, cache, seqId, guess, positions, false);
        result.forwards += 1;
        result.rowsProcessed += guess.length;
        const refined: number[] = [argmax(pending)];
        for (let i = 1; i < blockLen; i++) {
          refined.push(argmax(logitsRow(out as Tensor, i - 1, vocabSize)));
        }
        if (arraysEqual(refined, guess)) {
          converged = true;
          break;
        }
        guess = refined;
      }
      if (!converged) {
        // After `blockLen` iterations the guess is correct by Theorem S2
        // (position i is fixed after iteration i+1), but the cache still
        // holds the K/V of the *previous* (wrong) guess at some positions.
        // One commit forward feeds the correct tokens, repairing those
        // slots and yielding the logits for the next block.
        const positions = guess.map((_, i) => nextPos + i);
        out = await forwardTokenRows(executor, cache, seqId, guess, positions, false);
        result.forwards += 1;
        result.rowsProcessed += guess.length;
      }
      pending = logitsRow(out as Tensor, blockLen - 1, vocabSize);
      let stopped = false;
      for (let i = 0; i < blockLen; i++) {
        if (!pushToken(guess[i] as number)) {
          stopped = true;
          break;
        }
      }
      if (stopped) break;
    }

    if (result.tokens.length === 0) result.finishReason = "length";
    else if (result.tokens[result.tokens.length - 1] !== eosTokenId && result.finishReason !== "eos") {
      result.finishReason = "length";
    }
    result.tokensPerForward = result.forwards > 0 ? result.tokens.length / result.forwards : 0;
    return result;
  }
}
