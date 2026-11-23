# Speculative and Parallel Token Generation for the softwell Virtual Machine

**Research note 03 — decoding.** Implementation: `src/engine/novel-decoding.ts`
(`SpeculativeDecoder`, `JacobiDecoder`, `NGramDrafter`, cost model).
Tests: `test/novel-decoding.test.ts`. Status: implemented, tested, **not yet
wired into the scheduler** — both decoders are parallel paths behind the
existing `StageExecutor` interface, to be enabled behind a flag (see §7).

## 0. Abstract

Autoregressive decoding spends one target forward pass per token. This note
gives two schemes that spend one forward pass per *several* tokens, proves
they produce exactly the tokens plain greedy decoding would produce (and, for
speculative decoding, exactly the target's sampling distribution), and derives
the cost model that decides whether they win. The headline result is a
conditional one: on the reference TypeScript/CPU backend both schemes
**lose wall-clock** (measured 0.91× and 0.53×), because a batched forward
there costs almost exactly one single-row forward per row (measured
β ≈ 1.1, §5). The schemes are nevertheless correct and exactly as fast in
*forwards per token* as the theory predicts (7 forwards vs 25 for 24
tokens); they win wall-clock only where extra rows are cheap — i.e. on
bandwidth-bound hardware (β ≪ 1) — and, for speculative decoding, where the
drafter's acceptance rate clears the breakeven in §2.3. Nothing here claims a
win the measurements do not support.

**Notation.** Target model M, vocabulary V, prompt x₁…xₙ, generated tokens
y₁…y_m at absolute positions n+1…n+m. `F(S)` is one target forward over a
batch of rows S, returning one logits row per input row; the logits row for
an input token at position p is M's distribution for position p+1. Greedy
decoding is y_{t} = argmax M(· | x, y_{<t}). A *round* is one verification /
Jacobi forward (plus, for speculative decoding, the drafting that precedes
it). Draft length / block size is K. β ∈ [0, 1] is the marginal cost of one
extra row in a batched forward, as a fraction of a single-row forward (§5).

**Assumptions.** (A1) The target is deterministic given its cache state
(the reference executor is bitwise deterministic; verified in
`test/engine.test.ts`). (A2) KV-cache `append` at a position *overwrites*
that position's K/V slot — true of `BlockPagedKVCache`, false of an
append-only cache; see the module header of `novel-decoding.ts`. (A3) The
drafter never calls the target and never sees target logits (it sees only
prompt + generated tokens). (A4) One forward's cost is `1 + (R−1)·β` for R
rows, with a single β for the hardware at hand. (A5) Single sequence,
batch size 1. Argmax ties are broken by first index (the `argmax` in
`src/engine/sampling.ts`); the theorems hold for any fixed tie-break shared
by both decoders.

---

## 1. The drafter: n-gram / context-copy drafting

The default drafter is built from what exists: the text itself.
`NGramDrafter` proposes the next token by
finding the longest suffix of the context (up to 8 tokens) that has occurred
before in the context and copying the token that followed its most recent
occurrence; if no suffix of length ≥ 1 has occurred before, it repeats the
last token. Drafting is sequential (each proposal extends the working
context) and costs O(K · 8 · |context|) integer comparisons — no forward
pass, no weights, no training.

A real draft model is also supported: `ModelDrafter`
(src/engine/novel-decoding.ts) runs a second, smaller loaded model
autoregressively through its own `StageExecutor` (see
src/scheduler/draft-model.ts; `--draft-model <dir>` on the CLI, which
requires `--decoding speculative`). It replaces the n-gram drafter
wherever the decoder accepts an `AsyncDrafter`, and changes α and
nothing else in the theorems below — the verification logic and the
exactness proofs are drafter-agnostic, provided (A3) holds. The draft
model must share the target's tokenizer/vocabulary (checked loudly at
activation); a mismatch is an error, never a silent fallback.

This drafter is deliberately weak. It can only predict continuations that
have already been seen, so its acceptance rate on a trained model generating
varied text is expected to be low (single-digit percent is the honest prior;
we did not measure it on a trained model because the repo has none). Its
virtue is that it is *free*, which makes the cost model of §2.3 a clean
function of one measured number, the acceptance rate α. Replacing it with a
trained draft model (a small transformer, or a draft head on the target's
hidden states in the style of EAGLE) changes α and nothing else in the
theorems below — the verification logic and the exactness proofs are
drafter-agnostic, provided (A3) holds.

---

## 2. Speculative decoding

### 2.1 Algorithm

State is kept as one of two kinds of round, alternating as tokens are
accepted (see `SpeculativeDecoder.generate`):

- **Round A (pending logits known).** Prefill — or the previous round's
  bonus/correction token, once fed — has produced logits L predicting the
  next token. Draft K tokens d₁…d_K from the drafter, feed them at the next
  K positions in *one* forward, and verify left to right: dᵢ is checked
  against the logits that predict it (L for i = 1; verification row i−1 for
  i ≥ 2). Let a be the length of the accepted prefix.
  - If a < K, the first rejection at position a+1 yields a *correction*
    token c sampled from the predicting distribution (§2.2); the round
    outputs d₁…d_a, c. Token c sits at its position but has not been fed —
    it becomes the next round's *carry*.
  - If a = K, a *bonus* token b is sampled from the last verification row
    (which predicts the token after d_K); the round outputs d₁…d_K, b, and
    b becomes the carry.
- **Round B (carry token c at position p, not yet fed).** Feed c as row 0
  of the verification forward, followed by K fresh drafts for positions
  p+1…p+K (K+1 rows total). Row 0's logits verify the first draft, row i
  verifies draft i+1, and the last row yields the bonus token. Acceptance,
  correction, and carry work exactly as in round A.

EOS: tokens are appended in order and generation stops the moment an EOS
is appended — including an EOS accepted mid-draft; later drafts and the
bonus token are dropped. `maxTokens` clamps the draft length and suppresses
a bonus token that would overrun it. An empty draft (a drafter that declines
to propose) degenerates to one plain autoregressive step. Draft length K is
the number of *drafts* per round in both round types, so the per-round
token expectation below is uniform.

The carry row is the price of KV-cache honesty: a correction/bonus token
must be fed before its successor's logits exist, and feeding it inside the
next verification forward costs one row. It is also what repairs the cache:
the rejected draft that previously occupied the carry's position is
overwritten when the carry is fed (A2), and any stale draft K/V beyond the
rejection point is overwritten when those positions are (re-)fed later,
before any forward reads them as prefix — because each forward appends a
row's K/V *before* reading the prefix containing it.

### 2.2 Theorem S1 — correctness and expected speedup

**Theorem S1a (greedy exactness).** *Under (A1)–(A3), greedy speculative
decoding (temperature 0) outputs exactly the token sequence greedy
autoregressive decoding outputs, for any drafter and any K ≥ 1.*

*Proof sketch (induction on output positions).* Fix the output position t
and assume all earlier outputs equal the greedy sequence. The token the
speculative decoder emits at t is either (i) a draft token that was verified
equal to the target's argmax at t — hence equal to the greedy token; or
(ii) a correction/bonus token defined as the target's argmax at t — hence
equal to the greedy token. In both cases the emitted token is the greedy
token, and the cache state it leaves behind is the greedy state: accepted
drafts were fed at their positions with the greedy tokens' K/V, and the
correction/bonus token is fed at its position at the start of the next
round, overwriting the rejected draft's K/V there (A2) before any later
row attends to it. ∎

**Theorem S1b (sampled distributional exactness).** *Under (A1)–(A3), with
temperature > 0, each token speculative decoding emits is distributed
exactly as the target model's own token at that position (and hence the
whole sequence is distributed as the target's autoregressive sequence).*

*Proof sketch.* The drafter is deterministic, so its draft distribution q
is a point mass at the draft token d. The verification step is speculative
sampling (Leviathan et al. 2023; Chen et al. 2023) specialised to q(d) = 1:
accept d with probability min(1, p(d)) where p is the target distribution
at that position; on rejection, resample from the residual
norm(max(0, p − q)) — here p with d zeroed and renormalised. The output
distribution is then, for any token x: if x = d, P = p(d) (accepted) +
(1 − p(d)) · 0 (residual excludes d) = p(d); if x ≠ d, P = 0 (never
accepted as a draft) + (1 − p(d)) · p(x)/(1 − p(d)) = p(x). So the emitted
token ∼ p, exactly. The bonus token is sampled directly from p. Induction
over positions, as in S1a, extends this to the sequence — with one caveat:
the *joint* statement needs the RNG draws at different positions to be
independent, which holds because the decoder owns one persistent seeded
RNG stream (it deliberately does not reuse `sampleLogits`, whose per-call
seed would repeat the same draw). The bonus/correction token's feeding and
cache repair are as in S1a. ∎

*What is proven vs assumed.* S1a is a complete proof given (A1)–(A3); its
only gap is (A2), which is a property of the cache implementation, not of
the algorithm — it is verified by construction for `BlockPagedKVCache` and
would fail for an append-only cache (which would need a truncate/rollback
primitive). S1b's per-position calculation is exact; the sequence-level
claim additionally assumes the target's conditional distributions are the
ones the verification rows compute, i.e. that the cache state at each
position is the greedy/sampled prefix state — which follows from S1a's
cache argument applied position by position. The implementation's sampled
path is exercised only for determinism under a fixed seed, not for
distributional equality (which no finite test can establish); see §6.

**Theorem S1c (expected speedup).** *Let each draft token be accepted
independently with probability α. Then one round yields
E[T] = 1 + α + α² + … + α^K tokens, i.e. 1/E[T] forwards per token, and the
expected wall-clock speedup over plain decoding is*

$$S(\alpha, K, \beta) = \frac{1 + \alpha + \cdots + \alpha^K}{1 + K\beta}.$$

*The breakeven acceptance rate α\* (S = 1) is 0 at β = 0, and 1 at β = 1.*

*Derivation.* A round always emits one correction/bonus token, plus the
accepted prefix. The prefix has length ≥ i iff the first i drafts are
accepted, probability αⁱ under independence, so E[prefix] = Σᵢ₌₁^K αⁱ.
A round is one target forward; a steady-state round's forward has K+1 rows
(carry + K drafts), costing 1 + Kβ by (A4) (the first round has K rows and
is a one-off). Dividing gives S. At β = 0, S = E[T] ≥ 1 with equality only
at α = 0, so α\* = 0. At β = 1, S = E[T]/(K+1) ≤ 1 with equality only at
α = 1, so α\* = 1. ∎

*Remark (without independence).* If each draft is accepted with conditional
probability ≥ α given the earlier drafts in its round, the chain rule gives
P(prefix ≥ i) ≥ αⁱ, so E[T] ≥ 1 + Σ αⁱ: the independence formula is a
*lower bound* under a uniform conditional acceptance floor, not merely an
approximation. Real acceptance is positively correlated along a draft
(repeated text is either all predictable or all surprising), which is why
the measured tokens/round in §5 (4.00) sits *below* the independence
prediction (4.52) at the same α̂: correlation concentrates rejections, and
one rejection truncates the whole round. The independence formula should
therefore be read as an upper-ish estimate for bursty text and a lower
bound only under the conditional-floor reading above.

*Corollary (β = 1 is unwinnable).* At β = 1 no speculative scheme of this
form beats plain decoding in wall-clock: a round produces at most K+1
tokens from K+1 rows of work. The win, when it exists, is entirely a
β < 1 phenomenon — extra rows must be cheaper than a full forward.

### 2.3 Breakeven table (K = 4)

| β (row cost) | α\* breakeven | S at α = 0.95 |
|---|---|---|
| 0.0 (weights streamed once) | 0.000 | 4.52× |
| 0.5 | 0.741 | 1.51× |
| 1.0 (compute-bound) | 1.000 | 0.905× |

(Computed by `breakevenAcceptance` / `speculativeSpeedup`; the β = 1
breakeven of exactly 1 is the corollary above, not a numerical artifact.)

---

## 3. Parallel multi-token (Jacobi) decoding

### 3.1 Algorithm

Fix a block of K positions. Start from an initial guess g⁽⁰⁾ supplied
entirely by the drafter (no target logits are used to seed it). One Jacobi
iteration feeds the whole guess in one batched forward and replaces every
position by the target's argmax conditioned on the current guess prefix:
g⁽ᵗ⁺¹⁾₀ = argmax(L) where L are the pending logits from the true prefix,
g⁽ᵗ⁺¹⁾ᵢ = argmax(row i−1 of the forward of g⁽ᵗ⁾) for i ≥ 1. Iterate until
g⁽ᵗ⁺¹⁾ = g⁽ᵗ⁾ (fixed point — the guess is then the greedy block), or for K
iterations. A 1-token block short-circuits to plain decoding.

If K iterations pass without a detected fixed point, the guess is
nevertheless correct (Theorem S2a) but the cache holds the K/V of the
*previous* guess at some positions, so one extra *commit* forward feeds the
correct tokens, repairing those slots (A2) and yielding the next block's
pending logits. EOS inside a block stops generation at the EOS, as in §2.1.

### 3.2 Theorem S2 — convergence bound

**Theorem S2a (convergence in ≤ K+1 forwards).** *Under (A1)–(A2), greedy
Jacobi decoding of a K-token block converges to exactly the greedy
autoregressive block. Let J be the first index where the initial guess is
wrong (J = K if the guess is entirely right). Then the block takes exactly
K − J + 1 target forwards (iterations plus a commit forward when the fixed
point is not detected within K iterations), and never more than K+1.*

*Proof sketch (induction on positions).* Position 0 of the block depends
only on the true prefix, so after iteration 1 it is the greedy token, and
it never changes again. Suppose positions 0…i−1 are greedy after iteration
i. Position i's update in iteration i+1 is the argmax conditioned on those
positions, hence greedy; so position i is greedy after iteration i+1.
Thus after t iterations the first t positions are greedy, and after K
iterations the whole block is greedy. For the forward count: the output of
iteration s has its first J+s positions greedy (the first J were already
right, and each iteration fixes one more), so the output is entirely
greedy after s = K−J iterations, and the *input* to iteration K−J+1 is
entirely greedy, making that iteration's output equal its input — the
fixed point is detected at iteration K−J+1 — except when J = 0, where
detection would need iteration K+1; in that case the loop stops after K
iterations with a greedy guess and the commit forward is the (K+1)th
forward. In all cases the accepted block is the greedy block, and the
cache ends in the greedy state: on detection the last forward fed exactly
the accepted tokens; on commit the commit forward did. ∎

**Theorem S2b (expected forwards).** *If each initial guess is
independently correct with probability γ, the expected forwards per block
are E[T] = Σⱼ₌₀^{K−1} (K−j+1)·γʲ(1−γ) + 1·γ^K, and the expected wall-clock
speedup is S = K / (E[T]·(1 + (K−1)β)). At β = 1, S = 1/E[T] ≤ 1, with
equality only at γ = 1.*

*Derivation.* J is geometric: P(J = j) = γʲ(1−γ) for j < K, P(J = K) =
γ^K; apply S2a. Each forward is K rows, costing 1+(K−1)β; the baseline
spends K forwards for K tokens. ∎

*Remarks.* (i) Jacobi is greedy-only in this implementation: a fixed
point is defined by argmax, and there is no sampled analogue that preserves
the target distribution without a verification step (which would turn it
back into speculative decoding). The constructor rejects temperature > 0.
(ii) The worst case K+1 forwards for K tokens is *worse* than baseline in
forward count, but it is bounded and rare — it needs the very first guess
token to be wrong. (iii) A cheap optimisation, not implemented: seed
g⁽⁰⁾₀ with argmax(L), which is already known and always right; it shifts
J ≥ 1 and saves at most one forward per block. It is omitted so the
theorem's J refers to the drafter's guess alone. (iv) Unlike speculative
decoding, Jacobi re-attends the same block prefix on every iteration, so
its wall-clock cost has an extra term the β model does not capture
(repeated attention over the block prefix); §5 measures the gap.

### 3.3 Breakeven table (K = 4)

| β | γ\* breakeven | S at γ = 0.95 |
|---|---|---|
| 0.0 | 0.519 | 2.71× |
| 0.5 | 0.936 | 1.08× |
| 1.0 | 1.000 | 0.68× |

(Computed by `breakevenJacobiGamma` / `jacobiSpeedup`. Note the β = 0
breakeven is *not* 0: even with free rows, a fully-wrong guess costs K+1
forwards for K tokens.)

---

## 4. What the demo model actually measures

The demo model (10M parameters, random weights, hidden 384, 6 layers,
vocab 259) is untrained, and its greedy output collapses to a single
repeated token. Both prompts measured produce a constant stream:

| prompt | greedy baseline (24 tokens) |
|---|---|
| "Hello world" | 57 × 24 |
| "The quick brown fox jumps" | 130 × 24 |

That collapse is the whole story of the measured acceptance rate: the
n-gram drafter's repeat-last fallback predicts a constant stream perfectly,
so α̂ = 19/20 = 0.95 on *both* prompts. **This number must not be quoted
as the drafter's quality.** It is an artifact of the target's degeneracy,
and on a trained model generating varied text the same drafter would score
far lower. What the demo *does* establish is exactness (both schemes match
the baseline token-for-token on both prompts, including the 24-token
measurement runs) and the arithmetic of the cost model:

| quantity (K = 4, 24 tokens, "Hello world") | measured | cost-model prediction |
|---|---|---|
| speculative acceptance α̂ | 19/20 = 0.950 | — |
| speculative rounds / forwards | 6 / 7 (baseline 25 incl. prefill) | — |
| speculative tokens / round | 4.00 | 4.52 at α̂ = 0.95 |
| speculative forwards / token | 0.29 (7/24) | 0.221 (1/E[T]) |
| speculative speedup, β = 0 / 0.5 / 1 | — | 4.52× / 1.51× / 0.905× |
| Jacobi forwards / iterations / blocks | 10 / 9 / 6 | E[T] ≈ 1.48/block at γ = 0.95 |
| Jacobi speedup at β = 1 | — | 0.68× |

The tokens/round gap (4.00 vs 4.52) is the correlation effect of §2.2
plus the final partial round: with only 6 rounds, one rejection truncates
a whole round. The test asserts the gap is within 25%, not equality.

## 5. Wall-clock: the β ≈ 1 regime

Timing on this machine (demo-10m, 24 tokens, median of 3 runs):

| scheme | median wall-clock | speedup vs baseline |
|---|---|---|
| baseline autoregressive | 993 ms (848, 993, 994) | 1.00× |
| speculative, K = 4 | 1094 ms (1062, 1094, 1117) | **0.91×** |
| Jacobi, K = 4 | 1877 ms (1319, 1877, 2017) | **0.53×** |

The β measurement behind this: after prefilling "Hello world", a 1-row
decode forward takes 17.4 ms, a 2-row forward 36.6 ms, a 4-row forward
73.4 ms — i.e. cost(4 rows)/cost(1 row) = 4.23, implied β̂ ≈ 1.08
(2-row: β̂ ≈ 1.11). Rows cost *slightly more* than linear — the excess
over 1.0 is partly attention over growing prefixes inside the batch, so
β̂ ≈ 1 is the honest summary: this backend is compute-bound, and a
batched forward is (nearly) one single-row forward per row.

Consequences, all consistent with the theorems:

- Speculative measured 0.91× vs the β = 1 prediction of 0.905× — the
  model predicts the wall-clock outcome to within measurement noise.
  Speculation still wins on *forwards* (7 vs 25) and would win wall-clock
  on any hardware with β < 0.88 (the breakeven at α = 0.95 is β\* ≈ 0.88
  for K = 4 — solve S = 1 for β).
- Jacobi measured 0.53× vs the β = 1 prediction of 0.68× at γ = 0.95.
  The model is optimistic here: it does not charge for re-attending the
  block prefix on every iteration, and Jacobi's run-to-run variance is
  large (1319–2017 ms). Direction and order of magnitude are right;
  precision is not claimed.
- Neither scheme can win wall-clock at β = 1 (§2.2 corollary, §3.2) —
  and the measurements confirm both lose. The actionable conclusion is
  not "speculation is useless" but "speculation's win lives in β": it
  needs bandwidth-bound hardware (real GPUs serving large models, where
  decode is weight-streaming-bound and β ≈ 0) and/or a drafter with a
  high α on real text.

## 6. What is proven, what is conjectured, what is not done

**Proven (modulo the stated assumptions).** Greedy exactness of both
schemes (S1a, S2a); per-position distributional exactness of sampled
speculative decoding (S1b); the expected-tokens, speedup, and breakeven
formulas under independence (S1c, S2b); the Jacobi forward bound K−J+1
(S2a); the β = 1 impossibility corollaries.

**Conjectured / assumed, not proven.** The independence model for α and γ
(real acceptance is correlated; §2.2's remark gives the one-sided bound
that survives without it). That β is a single constant per hardware (it
drifts with prefix length and batch composition; §5's β̂ ≈ 1.1 already
shows superlinearity the model ignores). That a trained draft model can
reach α ≈ 0.6–0.9 on real text — the literature reports this range for
small draft models (Leviathan et al. 2023), but it is not measured here.

**Proof gaps.** (A2) is load-bearing and implementation-specific: if the
cache ever stops overwriting on append, both schemes silently produce
wrong tokens (rejected drafts would shadow correct ones). A regression
test that asserts overwrite semantics on `BlockPagedKVCache` would close
this; it does not exist yet. (A4) is a modelling choice, not a theorem.
S1b's sequence-level claim relies on the cache argument of S1a applied
per position; a fully formal treatment would need the joint distribution
over cache states, which the proof sketch does not give. The sampled path
has no distributional test — only determinism under a fixed seed.

**What did not work.** (1) Wall-clock on this backend: both schemes lose
(§5) — expected, given β ≈ 1, but worth stating plainly. (2) The n-gram
drafter is not a serious drafter for varied text: it only copies seen
continuations, and its headline α = 0.95 is a degeneracy artifact (§4).
A real draft model (or a draft head on target hidden states) is the
missing piece for a real speedup. (3) Jacobi's commit forward and repeated
prefix attention make it the weaker of the two schemes here; its
advantage — needing no drafter at all — does not compensate on this
backend. (4) The first speculative implementation drafted only K−1 tokens
in carry rounds (one row spent on the carry); it was changed to draft the
full K (K+1 rows) so the per-round expectation is uniform — the earlier
variant's steady-state breakeven at β = 1 was α = 1 with strictly worse
constants, and the change is recorded here so the git history is not
misread as a measurement.

## 7. Wiring notes (for the scheduler integration)

- Both decoders take the same `StageExecutor` + `PagedKVCache` the
  scheduler already owns; they are single-sequence, batch-size-1, and
  assume the cache is theirs for `seqId` (they call `cache.free(seqId)`
  on entry). Multi-sequence batching would need one decoder per sequence
  sharing a batched verification forward — not implemented.
- The natural integration point is a flag on the decode step: when
  enabled, replace the scheduler's one-token-per-step decode for a
  sequence with a `SpeculativeDecoder` (or `JacobiDecoder`) driving the
  same stage chain, feeding it the sequence's prompt + generated tokens.
  The pipeline-stage chain complicates this: verification rows must flow
  through stages 0…N−1 as one batch, which the scheduler's `runStageChain`
  already supports for multi-row batches (prefill does exactly this).
- Before enabling by default: measure β on the target hardware; enable
  only when β is below the breakeven for the measured α (e.g. β < 0.88 at
  α = 0.95, K = 4 — and α must be measured on real traffic, not the demo
  model). Log α, forwards/token, and wall-clock per sequence so the
  breakeven can be re-checked in production.
- The drafter is the highest-leverage follow-up: a 1-layer draft
  transformer on the target's hidden states (EAGLE-style) or a small
  standalone draft model would raise α on real text; the verification
  machinery would not change.

## References (verified via web search, 2026-11-22)

- Leviathan, Kalman & Matias (2023). *Fast Inference from Transformers
  via Speculative Decoding.* arXiv:2211.17192. Introduces speculative
  decoding and speculative sampling; 2–3× on T5-XXL with identical outputs.
- Chen et al. (2023). *Accelerating Large Language Model Decoding with
  Speculative Sampling.* Independent, concurrent formulation of the same
  acceptance/rejection scheme.
- Fu, Bailis, Stoica & Zhang (2024). *Break the Sequential Dependency of
  LLM Inference Using Lookahead Decoding.* arXiv:2402.02057 (ICML 2024).
  Jacobi-iteration parallel decoding with no draft model; the formulation
  §3 builds on, and cites Jacobi decoding as in Santilli et al. (2023).
