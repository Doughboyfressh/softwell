# Novel attention mathematics for the softwell virtual machine

**Status:** implemented and tested, 2026-11-22.
**Code:** `src/engine/novel-attention.ts` — pure functions on `Float32Array`,
in the style of `src/engine/attention.ts`. The existing engine
(`attention.ts`, `transformer.ts`, the scheduler) is untouched; these are
parallel paths to be wired in later behind a flag.
**Tests:** `test/novel-attention.test.ts` (9 tests, all passing).
**Run:** `node --test test/novel-attention.test.ts`.

## 0. What this note claims — and what it does not

The softwell VM cannot exceed its host's physical FLOPS or bytes. The
only way to beat enterprise hardware is to do *less work per token*.
This note gives two attention variants that do less work, each with a
**worst-case error bound against exact grouped-query attention that is
computed from the actual data** — no distributional assumptions, no
learned parameters, no training. The bounds are proved in exact
arithmetic (Section 3); the implementation adds one empirical term, a
`1e-4` relative float32 reassociation slack, which is flagged wherever
it appears.

What is **not** claimed:

- No claim about downstream model quality (perplexity, benchmark
  scores). The theorems bound the attention *output* error against
  exact attention on given inputs. Whether that error is tolerable to a
  real model is an empirical question for real-model validation, not
  proved here.
- No wall-clock speedup claim. The measurements below are *counted
  multiply-add operations* under a stated convention, not timings; the
  implementation is unoptimized reference TypeScript.
- The sub-quadratic *cost* claims are conditional on properties of the
  data (local key coherence for A1, concentrated attention for A2).
  Worst case, both methods degrade gracefully to exact attention plus
  a one-off `O(n d)` statistics pass — the error bound never breaks,
  only the speedup does. Section 5 states exactly where each proof
  ends and each empirical assumption begins.

## 1. Setup and shared machinery

Fix one query head. `q ∈ R^d` is the query, `k_0..k_{n-1}` and
`v_0..v_{n-1}` are the visible KV prefix (causal masking makes the
visible set a prefix; `kvPositions` must be non-decreasing, which the
code validates). Scale is `α > 0` (default `1/√d`). Exact attention is

```
z_t = α q·k_t,   p = softmax(z),   o = Σ_t p_t v_t.
```

Let `Vmax = max_t ||v_t||₂`.

**Blocks.** Partition the distant positions `[0, n−W)` into consecutive
blocks of size `B`, where `W` is the *local window*: the last `W`
positions are always computed exactly, as is the partial block at the
distant boundary. For each block `b` and KV head, precompute once per
KV cache and reuse for every query:

```
k̄_b = mean of k_t over b,   v̄_b = mean of v_t over b,
R_b = max_t ||k_t − k̄_b||₂,  S_b = max_t ||v_t − v̄_b||₂.
```

`computeBlockStats` builds these in `O(n d)` per KV head (7 counted
ops per element: two means, two radii, one value norm). GQA is handled
by keeping stats per KV head: query heads in a group share their KV
head's stats, and each head's bound uses its own `||q||` with its KV
head's radii and `Vmax`. Reported bounds are maxima over heads (and
rows, for prefill).

**Op counting.** One multiply-add on one vector element counts as one
op. A length-`d` dot product costs `d` ops; scaling-and-adding one
length-`d` value vector costs `d` ops. Softmax exponentials and scalar
max/sum work are `O(n)` per row — lower order than the `O(n d)`
matmul-like work — and are not counted, for the exact and the
approximate paths alike. `exactOps` is what exact grouped-query
attention costs under the same convention: `2·n·d` per (query, head).

## 2. Theorem A1 — landmark / block-mean attention

**Algorithm** (`landmarkAttention`, `prefillLandmarkAttention`).
For each distant full block `b`, compute the logit perturbation the
block-mean approximation would introduce,

```
δ_b = α ||q|| R_b.
```

If `δ_b ≤ δmax` (default `0.05`), replace the whole block by its
landmark: one score dot `α q·k̄_b`, and one value vector `v̄_b` carrying
multiplicity `B` in the softmax (all `B` positions in the block share
one approximate logit, so the block contributes
`B·exp(α q·k̄_b)/Z · v̄_b`). Otherwise compute the block exactly.
The trailing `W` positions and the partial boundary block are always
exact.

**Theorem A1.** Let `δ` be the largest `δ_b` over approximated blocks
(`0` if none) and `S` the largest `S_b` over approximated blocks.
Let `õ` be the landmark-attention output. Then

```
||õ − o||₂ ≤ (e^{2δ} − 1) · Vmax + S.
```

*Proof.* Write `z̃_t` for the approximate logits (`z̃_t = α q·k̄_b`
for `t` in an approximated block, `z̃_t = z_t` elsewhere) and `ṽ_t`
for the approximate values (`ṽ_t = v̄_b` in an approximated block,
`ṽ_t = v_t` elsewhere), and `p̃ = softmax(z̃)`. Then

```
õ − o = Σ_t p̃_t (ṽ_t − v_t) + Σ_t (p̃_t − p_t) v_t.
```

*Value term.* `||Σ_t p̃_t (ṽ_t − v_t)|| ≤ Σ_t p̃_t ||ṽ_t − v_t||
≤ S · Σ_{t approx} p̃_t ≤ S`, since `||ṽ_t − v_t|| ≤ S_b ≤ S` in
approximated blocks and `0` elsewhere.

*Weight term.* Lemma 1 (below) gives `|z_t − z̃_t| ≤ δ_b ≤ δ` for
`t` in block `b`, and `0` for exact positions. Lemma 2 gives
`||p̃ − p||₁ ≤ e^{2δ} − 1`, so
`||Σ_t (p̃_t − p_t) v_t|| ≤ Σ_t |p̃_t − p_t| · ||v_t||
≤ (e^{2δ} − 1) · Vmax`. Adding the two terms proves the theorem. ∎

**Lemma 1 (logit perturbation).** `|α q·k_t − α q·k̄_b| ≤ α ||q|| R_b`.
*Proof.* Cauchy–Schwarz: `|q·(k_t − k̄_b)| ≤ ||q|| · ||k_t − k̄_b||
≤ ||q|| R_b`. ∎

**Lemma 2 (softmax perturbation).** If `|z_t − z̃_t| ≤ δ` for all `t`,
then `||softmax(z̃) − softmax(z)||₁ ≤ e^{2δ} − 1`.
*Proof.* Write `Z = Σ e^{z_t}`, `Z̃ = Σ e^{z̃_t}`. Each
`e^{z̃_t} ∈ [e^{−δ} e^{z_t}, e^{δ} e^{z_t}]`, so
`Z̃ ∈ [e^{−δ} Z, e^{δ} Z]`, hence `p̃_t / p_t ∈ [e^{−2δ}, e^{2δ}]`.
Then `|p̃_t − p_t| ≤ (e^{2δ} − 1) p_t` (the upper side dominates,
since `e^{2δ} − 1 ≥ 1 − e^{−2δ}`), and summing over `t` gives the
claim. ∎

**Corollary (cost).** With `B = ⌈√n⌉`, prefill over `n` rows costs
`O(n^{3/2} d)` per head when every block is approximated: per row `i`
there are `≈ i/B` landmark dots plus `W` exact local dots, and
`Σ_i i/B = n²/(2B) = O(n^{3/2})`; block stats add one `O(n d)` pass.
If a block is incoherent (`δ_b > δmax`) it costs `B` dots instead of
one, so the worst case — every block incoherent — is `O(n² d)`,
i.e. exact attention plus the statistics pass. The bound holds in
both cases; only the speedup is data-dependent.

**Remark.** The exact-fallback decision uses only the key dispersion
`δ_b`, not the value radius `S_b`. If keys are coherent but values are
dispersed inside a block, the block is still approximated and `S`
enters the bound as stated — the bound stays valid, but can be loose.
Adding an `S_b` threshold to the fallback is a natural refinement
(see Section 5).

## 3. Theorem A2 — dynamic sparsity by block screening

**Algorithm** (`sparseAttention`, `prefillSparseAttention`). Score
the local window exactly and every distant full block's landmark
`ℓ_b = α q·k̄_b`. Let `M = max(max local score, max_b ℓ_b)`. For each
distant block, an upper bound on its *total* attention mass is

```
mass_b ≤ B · exp(u_b − M),   u_b = ℓ_b + α ||q|| R_b.
```

If that bound is at most `dropThreshold` (default `10⁻³`), skip the
block entirely — never score its positions. Otherwise score the block
exactly. Renormalize the softmax over the kept positions. The local
window, the partial boundary block, and surviving blocks are always
exact, so the kept set is never empty when the prefix is non-empty.

**Theorem A2.** Let `m` be the sum of the mass upper bounds over the
skipped blocks. Let `õ` be the sparse-attention output. Then

```
||õ − o||₂ ≤ 2 m · Vmax.
```

*Proof.* First, the mass bound is rigorous. For `t` in block `b`,
Lemma 1 gives `z_t ≤ ℓ_b + α||q||R_b = u_b`. The denominator
`Z = Σ_t e^{z_t} ≥ e^{max_t z_t}`, and `M ≤ max_t z_t`: the local
scores are true scores, and each `ℓ_b` is the *mean* of its block's
true scores, hence `≤` the block's max. So
`Σ_{t∈b} p_t = Σ_{t∈b} e^{z_t}/Z ≤ B·e^{u_b}/e^M`, as claimed.
Summing over skipped blocks, the actual dropped mass
`m_a = Σ_{t dropped} p_t` satisfies `m_a ≤ m`.

Now write `A = Σ_{kept} p_t v_t`, `D = Σ_{dropped} p_t v_t`,
so `o = A + D` and `õ = A/(1 − m_a)`. Then

```
||õ − o|| = ||A·m_a/(1−m_a) − D||
          ≤ ||A||·m_a/(1−m_a) + ||D||
          ≤ (1−m_a)·Vmax·m_a/(1−m_a) + m_a·Vmax = 2 m_a Vmax ≤ 2 m Vmax,
```

using `||A|| ≤ (1−m_a)·Vmax` and `||D|| ≤ m_a·Vmax`. ∎

**Corollary (cost).** Per (row, head) the cost is
`O((n/B + s·B + W)·d)` where `s` is the number of surviving blocks:
landmark dots for all blocks, exact work only for survivors and the
local window. With `B = ⌈√n⌉` and `s` bounded by a constant
(concentrated attention), prefill is `O(n^{3/2} d)` per head. Worst
case — nothing skippable — is `O(n² d)`, exact attention plus the
statistics pass and one landmark dot per block per row.

**Remark.** A2 never evicts KV entries: screening is per query, from
a rigorous mass bound, and a block skipped for one query can be kept
for the next. That is the difference from KV-cache eviction policies
such as H2O, which drop entries permanently on the basis of
accumulated scores.

## 4. Measurements

All numbers below are from `test/novel-attention.test.ts`, run
2026-11-22 on this machine (`node --test`). Errors are max over
heads (and rows, for prefill) of the L2 norm `||approx − exact||₂`
against `groupedQueryAttention`; bounds are the theorems' bounds
evaluated on the actual inputs, plus the `1e-4` float32 slack.
`ops` counts multiply-adds under the Section 1 convention.

### A1, single query, GQA (n = 48, 4 heads, 2 KV heads, d = 8, B = 8, W = 4)

| keys | err | bound | ops / exactOps |
|---|---|---|---|
| locally coherent (block-constant + noise σ = 0.01) | 7.7e-5 | 4.3e-1 | 1200 / 3072 (0.39×) |
| incoherent (iid) — exact fallback | 0 | 6.8e-4 | 3104 / 3072 (1.01×) |

The incoherent row is the honest worst case: no block is approximable,
so the method does exact work (plus one `||q||` per head) and the
bound collapses to the float32 slack. With stats built fresh instead
of reused, the one-off `O(n d)` statistics pass would make a single
incoherent query cost *more* than exact attention; the statistics only
pay when reused across queries (prefill) or decode steps.

### A1, prefill (2 heads, 1 KV head, d = 8, B = ⌈√n⌉, W = 8, coherent keys)

| n | err | bound | ops / exactOps | ops ratio vs previous n |
|---|---|---|---|---|
| 64 | 1.7e-4 | 4.8e-1 | 35,664 / 66,560 (0.54×) | — |
| 128 | 1.2e-4 | 4.2e-1 | 98,464 / 264,192 (0.37×) | 2.76 |
| 256 | 1.2e-4 | 4.7e-1 | 275,984 / 1,052,672 (0.26×) | 2.80 |

Doubling the length multiplies counted ops by ≈ 2.8, not ≈ 4 as exact
attention would. Measured errors are ~3 orders of magnitude below the
bound: the bound is worst-case over the largest block radius, while
typical deviations are far smaller and partially cancel.

### A2, single query, GQA (n = 64, 4 heads, 2 KV heads, d = 8, B = 8, W = 4)

Concentrated data (block 0 keys aligned with the query, all other
blocks anti-aligned): err `1.0e-4`, bound `7.8e-3`, dropped-mass bound
`8.5e-4`, ops `1280 / 4096` (0.31×, stats reused).

### A2, prefill (2 heads, 1 KV head, d = 8, B = ⌈√n⌉, W = 8, concentrated)

| n | err | bound | dropped mass | ops / exactOps | ops ratio |
|---|---|---|---|---|---|
| 64 | 7.6e-5 | 8.5e-3 | 8.6e-4 | 41,712 / 66,560 (0.63×) | — |
| 128 | 1.2e-4 | 1.9e-2 | 1.9e-3 | 112,864 / 264,192 (0.43×) | 2.71 |
| 256 | 1.3e-4 | 4.2e-2 | 4.1e-3 | 290,160 / 1,052,672 (0.28×) | 2.57 |

### A2, uniform attention (all keys identical)

Nothing is skippable — every block's mass bound exceeds the threshold
— so dropped mass is exactly `0` and the output equals exact
attention up to float32 (`err < 1e-4`). No false sparsity.

### Edge cases (all in the test file)

Single token (prefill length 1 and single query over `kvLen = 1`);
empty prefix (causal masking hides everything: output zeros, bound
`0`, matching exact); GQA with 1 and 2 KV heads; non-causal masking;
`kvLen` not a multiple of `B` (partial final block); stats reuse is
bit-identical to building stats on the fly; `δmax = 0` reduces A1 to
exact attention; empty KV cache throws like the reference; unsorted
`kvPositions` are rejected (block methods need the visible set to be
a prefix).

## 5. What is proven, what is conjectured, where the gaps are

**Proven** (in exact arithmetic; see Sections 2–3): the two error
bounds, given the inputs. Both are worst-case over the actual data
and are checked by the tests on every run: measured error never
exceeded the returned bound in any test above.

**Empirical, not proved:**

1. *The speedup is conditional.* A1's sub-quadratic cost needs most
   blocks to satisfy `δ_b ≤ δmax` (locally coherent keys); A2's needs
   most blocks to be skippable (concentrated attention). Real LLM KV
   caches may or may not satisfy these properties, and to what degree
   is a property of trained models, not of this note. The fallback
   guarantees are the point: when the properties fail, the method
   degrades to exact work, never to an unbounded error.
2. *The bounds are loose.* Measured errors run ~10³× below the
   bounds. The looseness comes from using max radii (`R_b`, `S_b`)
   and the worst-case softmax perturbation `e^{2δ} − 1`. A tighter
   average-case bound (e.g. in terms of RMS rather than max radii, or
   a bound on `||p̃ − p||₁` tighter than Lemma 2's) is open work.
   The `S` term in Theorem A1 is the main source of looseness on
   value-dispersed data; the fallback currently thresholds only on
   `δ_b`, not on `S_b`.
3. *Float32.* The theorems are proved in exact arithmetic. The
   implementation adds `1e-4 · (Vmax + 1)` to each bound as a
   reassociation slack. Observed float32 drift vs exact attention is
   ~`1e-6`–`1e-4` in these tests, inside the slack, but no roundoff
   analysis is given; the slack constant is a safety margin, not a
   proved roundoff bound.
4. *No model-quality claim.* Nothing here says a real model keeps its
   perplexity under these approximations. That needs evaluation on a
   real model with real KV caches — the necessary next step before
   wiring either path into the transformer behind a flag.
5. *Scope.* The bounds are per attention head per query; no claim is
   made about error accumulation across layers or across generated
   tokens. Composition of the per-layer bounds through a deep network
   is future work.

**What did not work.** The first version of the A2 scaling test used
unnormalized Gaussian directions, so the aligned/anti-aligned
separation depended on a `χ²₈` draw: on two of three seeds nothing
was separable, nothing was skipped, and prefill ops *exceeded* exact
(`1.11×`, `1.07×`) while the bound still held. That failure was
instructive — it is exactly the graceful-degradation regime — but it
was the wrong test of the concentrated regime. The test now normalizes
the direction to `||u||² = d`, pinning the separation. The lesson is
recorded here because it is also a property of the method: A2's win
is only as reliable as the attention's concentration, and a deployment
would want to monitor the achieved skip rate, not assume it.

## 6. Relation to prior art

- **Nyströmformer** (Xiong et al., arXiv:2102.03902): uses
  segment-means as landmark points to reconstruct the softmax matrix
  in `O(n)` time. A1 shares the segment-mean landmark idea but
  approximates the *keys themselves* by their block means, keeps an
  exact local window, and — unlike Nyströmformer — carries a
  per-instance worst-case error bound with an exact fallback for
  incoherent blocks.
- **Linformer** (Wang et al., arXiv:2006.04768): shows the attention
  matrix is approximately low-rank and projects keys/values to length
  `k`. A1's block-mean step is a low-rank-plus-local decomposition of
  the same spirit, but the projection here is fixed (block means) and
  the error is bounded per instance rather than in expectation over a
  random projection.
- **FlashAttention** (Dao et al., arXiv:2205.14135): exact attention
  with `O(n)` memory and fewer HBM accesses — still `O(n²)` FLOPs.
  A1/A2 are orthogonal: they reduce FLOPs, and could in principle be
  combined with IO-aware tiling.
- **H2O** (Zhang et al., arXiv:2306.14048): KV-cache eviction that
  keeps heavy-hitter and recent tokens, with a submodular guarantee.
  A2 is per-query screening with a rigorous mass bound and no eviction.

## 7. References

Only works actually consulted (via web search, 2026-11-22) are listed.
No other prior-art claims are made.

- Wang, Li, Khabsa, Fang, Ma. *Linformer: Self-Attention with Linear
  Complexity.* arXiv:2006.04768. https://arxiv.org/abs/2006.04768
- Xiong, Zeng, Chakraborty, Tan, Fung, Li, Singh. *Nyströmformer: A
  Nyström-Based Algorithm for Approximating Self-Attention.*
  arXiv:2102.03902. https://arxiv.org/abs/2102.03902
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness.*
  arXiv:2205.14135. https://arxiv.org/abs/2205.14135
- Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett,
  Wang, Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative
  Inference of Large Language Models.* arXiv:2306.14048.
  https://arxiv.org/abs/2306.14048
