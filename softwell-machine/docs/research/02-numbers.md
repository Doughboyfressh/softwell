# SW-code: a low-bit number system for weights, activations, and KV cache

**Softwell research note 02 — 2026-11-22.**

**Scope.** A novel number representation for the softwell virtual machine
that cuts the bytes moved per token by 4–8× versus BF16 (6–13× versus the
FP32 the engine actually moves today), with *provable* error bounds: a
per-element and per-dot-product bound for the representation itself
(Theorem N1) and a bound on the attention output when the KV cache is
quantized (Theorem N2). Implementation:
`src/engine/novel-numbers.ts`; tests: `test/novel-numbers.test.ts`.
All numbers in §5–§6 are measured by those tests on this machine
(`node --test test/novel-numbers.test.ts`); nothing in this note is
asserted from memory or from theory alone. Where a number is arithmetic
on measured quantities rather than a direct measurement, it is labelled
as such.

**Standing assumptions.** The machine is 100% virtual; decode is
memory-bandwidth-bound (survey note 00, §1.4), so time per token is
dominated by bytes moved, not FLOPs. A VM cannot exceed its hosts'
physical FLOPS or bytes/second — the win here is entirely *fewer bytes
per token*, not faster arithmetic. The engine today moves every weight
as float32 (the loader dequantizes BF16/FP16/FP8 to `Float32Array` on
load), so the status quo is 4 bytes per parameter and 4 bytes per KV
element crossing vRAM → vGPU.

---

## 1. Definitions and the format

**Definition 1 (SW-code).** A matrix is split into blocks of `n`
consecutive elements along each row (default `n = 64`). A block is
encoded independently as

```
[codes: ceil(n*b/8) bytes][scale: fp16, 2 bytes][outlier count: 1 byte]
[outliers: 3 bytes each = 1 index byte + fp16 value]
```

where `b ∈ {2, 3, 4}` is the code width (modes `"sw2"`, `"sw3"`,
`"sw4"`) and each code is one of `2L+1` uniform levels
`{-L, …, +L} · Δ` with `L = 1, 3, 7`, plus one escape code. The block
scale — the *clip level* `Q` — is stored as fp16 and `Δ = Q / L`.

The scale is **not** the block maximum. Let `a_(1) ≥ … ≥ a_(n)` be the
sorted `|x|` in a block and `r = ⌈n/8⌉`. Then

```
Q  = min(a_(1), 2 · a_(r))
```

and the *outliers* are the elements with `|x| > Q`, each stored exactly
as fp16. This is the whole twist, and it has two consequences that
drive the theorems:

- The scale is a **robust quantile** (the 87.5th percentile of `|x|`
  for `n = 64`), so one huge weight cannot poison the step size for
  the other 63 values. With a max-based scale, a single 10σ weight
  makes `Δ` ten times coarser for the whole block; here it is escaped
  and the block keeps its tight step.
- Because `Q ≥ a_(r)`, at most `r - 1` elements can exceed `Q`, so the
  outlier count is bounded **by construction** (`≤ 7` for a 64-block)
  and the worst-case bytes per element is bounded too
  (Lemma N1.0 below). There is no data-dependent blow-up.

**Definition 2 (packed tensor).** Blocks are variable-length (the
outlier count varies), so a packed tensor is one byte stream plus one
4-byte row offset per row (`rowOffsets`, length `rows+1`). Sequential
access — full decode, matmul — is O(1) amortized per block with a
streaming cursor. Random single-element access would need a prefix
scan and is deliberately not offered: the access patterns the engine
needs are all sequential.

**Definition 3 (input domain).** Inputs must be finite with
`|x| ≤ 65504` (the fp16 maximum): every value that can be stored as an
outlier must fit in fp16, and the block scale is stored as fp16. Real
LLM weights, activations, and K/V values are orders of magnitude
inside this range. The encoder throws a clear error outside it instead
of silently corrupting data.

**What the matmul does.** `matmulPackedBT` (float32 A × packed Bᵀ,
the shape of every weight projection in the engine) reads the packed
stream directly: each block is unpacked into a small lookup table
(code → value) and dotted against the float32 activations. B is never
materialized as float32. `matmulPackedAB` (packed × packed, for
quantized activations) decodes A one block at a time into a tile — one
block per row, never the whole matrix — and dots it against B's lookup
tables. The traffic saving is therefore realized at the vRAM → vGPU
transfer, not just on disk. Accumulation is float64 (plain JS numbers).

---

## 2. Theorem N1 — representation and dot-product error

**Lemma N0.1 (fp16 rounding).** For finite `x` with `|x| ≤ 65504`,
let `fl16(x)` be round-to-nearest-even fp16. Then
`|fl16(x) − x| ≤ 2⁻¹¹·|x| + 2⁻²⁵`.
*Proof sketch.* If `|x| ≥ 2⁻¹⁴` (normal range), the fp16 ulp at `x`
is at most `2⁻¹⁰·|x|`, so the rounding error is at most half that.
If `|x| < 2⁻¹⁴` (subnormal range), the ulp is `2⁻²⁴`, so the error
is at most `2⁻²⁵`. ∎

The code needs the bound in terms of the *stored* value `v = fl16(x)`,
since the true `x` is gone after encoding. From the lemma,
`|x| ≤ (|v| + 2⁻²⁵)/(1 − 2⁻¹¹) < 1.001·|v| + 2⁻²⁴`, hence the safe
form used everywhere below:

```
|fl16(x) − x| ≤ 2⁻¹⁰·|v| + 2⁻²⁴.                          (1)
```

**Lemma N1.0 (outlier count and worst-case size).** A block of `n`
elements has at most `⌈n/8⌉ − 1` outliers, and at most
`⌈n·b/8⌉ + 3 + 3·(⌈n/8⌉ − 1)` bytes.
*Proof.* `Q = min(a_(1), 2·a_(r)) ≥ a_(r)`, so any element with
`|x| > Q` is among the `r − 1` largest. The byte count is codes +
3 header bytes + 3 bytes per outlier. ∎

For `n = 64`: worst case 4.375 + 2.63 = 7.00 bits/element (sw4),
3.375 + 2.63 = 6.00 (sw3), 2.375 + 2.63 = 5.00 (sw2) — the
worst case is reached only if every block carries all 7 outliers,
which never happens on the measured distributions (§5).

**Theorem N1 (per-element error).** Let a block have stored scale
`s = fl16(Q)` and step `Δ = s/L`. For every element `x` in the block,
with `x̂` its decoded value:

- non-outlier (`|x| ≤ Q`): `|x − x̂| ≤ Δ/2`, except that if `Δ = 0`
  then `|x − x̂| < 2⁻²⁴`;
- outlier (`|x| > Q`): `|x − x̂| ≤ 2⁻¹⁰·|v| + 2⁻²⁴`, where `v` is
  the stored fp16 value.

*Proof.* Non-outlier: `x̂ = round(x/Δ)·Δ`, and round-to-nearest is
within `1/2` of its argument, so `|x̂ − x| ≤ Δ/2`. If `Δ = 0` then
`s = 0`, so `fl16(Q) = 0`, which implies `Q ≤ 2⁻²⁵ < 2⁻²⁴`
(round-to-nearest sends only values below the midpoint `2⁻²⁵` to
zero); non-outliers have `|x| ≤ Q` and decode to `0`. Outlier:
`x̂ = fl16(x)` exactly (fp16 values are exact in fp32), and (1)
applies. ∎

Note what is *absent*: there is no tail term. An outlier can never
blow up the bound, because it is never quantized on the coarse grid
at all — it is stored exactly as fp16. The bound is also essentially
tight: on Gaussian weights the measured max error is within 0.2% of
the bound (§5, Table 1), because some element in a large tensor
always sits near a rounding boundary.

**Corollary N1.1 (dot-product error).** Let `E` be the row error
norm bound of a packed row, `E² = Σ_blocks [(n_b − o_b)·(Δ_b/2)²
+ Σ_outliers (2⁻¹⁰·|v| + 2⁻²⁴)²]` (with `Δ_b = 0` contributing
`(2⁻²⁴)²` per non-outlier). Then for any float32 vector `a`,
`|a·x̂ − a·x| ≤ ‖a‖₂ · E`.
*Proof.* `a·(x̂ − x) ≤ ‖a‖₂·‖x̂ − x‖₂` by Cauchy–Schwarz, and
`‖x̂ − x‖₂² ≤ E²` by summing the per-element bounds of Theorem N1
over the row. ∎

**Corollary N1.2 (packed × packed).** With `E_A`, `E_B` the row error
norms and `‖â‖`, `‖b̂‖` the decoded row norms,
`|a·b − â·b̂| ≤ E_A·(‖b̂‖ + E_B) + ‖â‖·E_B`.
*Proof.* Split `a·b − â·b̂ = (a − â)·b + â·(b − b̂)`, apply
Cauchy–Schwarz to each term, and use `‖b‖ ≤ ‖b̂‖ + E_B`. ∎

**Typical case (remark, not a theorem).** For Gaussian weights the
clip is the block max, `Q ≈ 2.5σ` for `n = 64`, so `Δ ≈ 2.5σ/L`
and, if rounding errors are treated as independent uniform on
`[−Δ/2, Δ/2]`, the RMS error is `Δ/√12` and a dot product over `K`
terms has RMS error `‖a‖₂·Δ/√12`, versus the worst-case bound
`‖a‖₂·√K·Δ/2` — a factor `√(3K)` smaller. Measured RMS errors match
the `Δ/√12` prediction to within 3% (§5). The worst-case bound is
the honest guarantee; the typical error is what a model actually
feels, and §5 reports both.

---

## 3. Theorem N2 — KV-cache quantization error on attention output

Quantize K and V per (token, KV head) vector of length `headDim`
(one block per vector; `blockSize = headDim`), after RoPE, matching
the engine: the cache stores post-RoPE K. K and V may use different
modes — K is more sensitive because it moves the softmax, V enters
the output linearly. Let `ε_K`, `ε_V` be the largest per-row error
norms (Corollary N1.1) over the KV rows a head reads, `Vmax` the
largest decoded V row norm, and `δmax = ‖q‖₂·ε_K/√d` the largest
logit error at the default scale `1/√d` (for a general scale `s`,
`δmax = ‖q‖₂·ε_K·s`).

**Lemma N2.1 (softmax is ½-Lipschitz from ℓ∞ to ℓ₁).**
`‖softmax(u) − softmax(v)‖₁ ≤ (1/2)·‖u − v‖∞`.
*Proof.* By the integral form of the mean value theorem,
`softmax(u) − softmax(v) = ∫₀¹ J(v + t(u−v))·(u−v) dt` where
`J = diag(p) − ppᵀ` is the softmax Jacobian. The `∞→1` operator
norm of `J` is its largest column sum: column `j` sums to
`p_j(1−p_j) + Σ_{i≠j} p_i·p_j = 2·p_j(1−p_j) ≤ 1/2`, attained at
`p_j = 1/2`. Hence `‖softmax(u) − softmax(v)‖₁ ≤ (1/2)·‖u−v‖∞`. ∎

**Theorem N2 (attention output error).** For one query head,
with `o` the exact attention output and `ô` the output computed
from quantized K and V,

```
‖ô − o‖₂ ≤ (1/2) · δmax · Vmax + ε_V.                  (2)
```

*Proof.* Write `ô − o = Σ_t (p̂_t − p_t)·v̂_t + Σ_t p_t·(v̂_t − v_t)`.
First term: `‖Σ_t Δp_t·v̂_t‖₂ ≤ Σ_t |Δp_t|·‖v̂_t‖₂ ≤ ‖Δp‖₁·Vmax`,
and by Lemma N2.1 `‖Δp‖₁ ≤ (1/2)·‖δ‖∞ ≤ (1/2)·δmax`, since each
logit error satisfies `|δ_t| = |q·(k̂_t − k_t)|/√d ≤ ‖q‖₂·ε_K/√d`.
Second term: `‖Σ_t p_t·(v̂_t − v_t)‖₂ ≤ Σ_t p_t·ε_V = ε_V`, using
`Σ_t p_t = 1`. ∎

Two notes. First, the bound is stated without causal masking: masked
positions have `p_t = p̂_t = 0` in both, so they contribute nothing to
either output, and the unmasked bound still applies — the tests
verify the causal case against the same bound. Second, the bound is
deliberately worst-case and therefore loose: it takes the max over
all tokens of `ε` and `Vmax`, and the `‖Δp‖₁ ≤ (1/2)‖δ‖∞` step
assumes the whole probability mass moves onto the single largest
V row. Measured errors are 30–60× below the bound (§5, Table 3).
Tightening it would need a distributional assumption (e.g. near-
orthogonal V rows, or a bound on `Σ_t p_t²`); that is listed as a
gap in §7, not claimed here.

---

## 4. What is new, and what is not

The ingredients are not new: block scaling with a shared exponent is
the standard low-bit format (GPTQ uses per-group scales [1]; AWQ uses
per-channel scaling to protect salient weights [2]; QLoRA's NF4 uses
a distribution-matched codebook with one fp32 scale per 64 weights
[3]); asymmetric KV quantization with different treatment for K and V
is KIVI's core finding [4]. What is new here as a package is:

1. **the robust-quantile scale with exact outlier escape** — the
   scale comes from the 87.5th percentile of `|x|`, not the max, and
   the tail is stored exactly as fp16, so the per-element bound has
   no tail term (Theorem N1) and the outlier count is bounded by
   construction (Lemma N1.0);
2. **one uniform bit-packing + lookup-table decoder serving 2/3/4-bit
   modes for weights, activations, and KV** with a single streaming
   layout;
3. **matmuls that consume the packed stream directly** (lookup table
   per block, no float32 materialization of the weight matrix), so
   the traffic saving is realized at the vRAM → vGPU transfer;
4. **end-to-end bounds in closed form** (N1, N2) that are computed
   from the packed data alone and checked by the tests.

Two designs were considered and rejected. A **non-uniform
(NF4-style) Gaussian-quantile codebook** would cut RMS error by
~25%, but the max-error bound would become the largest codebook
gap instead of a single clean `Δ/2`, and the decode would need the
codebook rather than one multiply — not worth it for a first version.
A **power-of-two scale** (1-byte header instead of fp16) would save
1 byte per block but coarsens the step by up to 2×, doubling the
error bound — a bad trade at these block sizes.

---

## 5. Measured results

Reproduce with `node --test test/novel-numbers.test.ts`. All
randomness is seeded. Weights are synthetic: Gaussian `N(0, 0.02)`
and a heavy-tailed "spike" mixture (99% `N(0, 0.02)`, 1%
`N(0, 0.2)`), which mimics the outlier weights seen in real LLMs.

### Table 1 — Encode/decode error vs the Theorem N1 bound (256×512, σ = 0.02)

| distribution | mode | max err | RMS err | bound | outliers |
|---|---|---|---|---|---|
| Gaussian | sw2 | 3.680e-2 | 1.425e-2 | 3.680e-2 | 0.21% |
| Gaussian | sw3 | 1.226e-2 | 4.934e-3 | 1.227e-2 | 0.21% |
| Gaussian | sw4 | 5.254e-3 | 2.113e-3 | 5.258e-3 | 0.21% |
| spike | sw2 | 4.637e-2 | 1.540e-2 | 4.639e-2 | 0.89% |
| spike | sw3 | 1.545e-2 | 5.560e-3 | 1.546e-2 | 0.89% |
| spike | sw4 | 6.616e-3 | 2.373e-3 | 6.627e-3 | 0.89% |

The bound is in absolute units (σ = 0.02, so sw4's RMS error is
0.106σ). Measured max error is within 0.2% of the bound in every
cell — the bound is essentially tight. RMS error matches the
`Δ/√12` typical-case prediction to within 3% (sw4: predicted
0.103σ, measured 0.106σ). The heavy-tailed distribution costs
almost nothing: the spike outliers are escaped (0.89% of elements),
and the max error rises only 26% over Gaussian at the same mode,
where a max-scaled quantizer would have its step blown up ~10× in
the ~half of blocks that contain a spike.

### Table 2 — Matmul error vs the Theorem N1 dot-product bound

A ~ N(0,1), 16×256; B weight-like, 128×256; reference is a float64
matmul. Output std is 0.32, for scale.

| distribution | mode | worst err | RMS err | bound at worst | worst/bound |
|---|---|---|---|---|---|
| Gaussian | sw2 | 0.765 | 0.229 | 7.07 | 0.108 |
| Gaussian | sw3 | 0.365 | 0.081 | 2.61 | 0.140 |
| Gaussian | sw4 | 0.123 | 0.034 | 1.00 | 0.126 |
| spike | sw2 | 0.824 | 0.245 | 8.65 | 0.111 |
| spike | sw3 | 0.321 | 0.090 | 3.15 | 0.112 |
| spike | sw4 | 0.144 | 0.039 | 1.25 | 0.116 |

Packed × packed (activations × weights, Corollary N1.2):
sw4×sw4 worst 0.181, RMS 0.048, worst/bound 0.079;
sw4×sw3 worst 0.301, RMS 0.085, 0.075;
sw3×sw3 worst 0.405, RMS 0.114, 0.068.

The bound holds everywhere with 7–14× slack, as expected: the
Cauchy–Schwarz step assumes the error vector aligns perfectly with
the activation row, while real rounding errors are near-independent.
The RMS column is the number a model would feel: relative to the
0.32 output std, per-layer RMS error is 11% (sw4), 25% (sw3), 72%
(sw2). That is the honest price of the format — §7 discusses it.

### Table 3 — KV-cache quantization vs the Theorem N2 bound

kvLen 128, 4 heads / 2 KV heads, headDim 64; error is per-head L2
over 64 dims (per-element RMS = L2/8; exact output per-element
std ≈ 1).

| setup | worst head L2 err | bound | worst/bound | per-elem RMS |
|---|---|---|---|---|
| Gaussian sw4/sw4 | 0.167 | 12.45 | 0.016 | 0.021 |
| Gaussian sw4/sw2 | 0.776 | 25.74 | 0.030 | 0.097 |
| Gaussian sw3/sw3, causal | 0.459 | 30.55 | 0.015 | 0.057 |
| K-channel-outliers sw4/sw4 | 0.258 | 15.42 | 0.020 | 0.032 |
| K-channel-outliers sw2/sw2 | 2.047 | 130.7 | 0.016 | 0.256 |

The bound holds in every case, including causal masking and K with
per-channel outliers (channels ×8). The recommended KV operating
point is sw4/sw4 (~2% per-element output error) or sw4/sw2 (~10%,
V only). sw2/sw2 keeps the bound but the measured 26% per-element
error is too large for quality — the bound is not a quality
certificate, only an error guarantee.

### Table 4 — Measured traffic (2.1M parameters, 1024×2048)

| distribution | mode | B/param | bits | vs BF16 | vs FP32 |
|---|---|---|---|---|---|
| Gaussian | sw2 | 0.305 | 2.44 | **6.55×** | 13.10× |
| Gaussian | sw3 | 0.430 | 3.44 | **4.65×** | 9.30× |
| Gaussian | sw4 | 0.555 | 4.44 | 3.60× | 7.20× |
| spike | sw2 | 0.326 | 2.61 | **6.13×** | 12.26× |
| spike | sw3 | 0.451 | 3.61 | **4.43×** | 8.86× |
| spike | sw4 | 0.576 | 4.61 | 3.47× | 6.94× |

KV cache, sw4: headDim 64 → 0.616 B/elem (3.25× vs BF16,
6.50× vs FP32); headDim 128 → 0.561 B/elem (3.56×, 7.13×).
The KV numbers are worse than the weight numbers at the same mode
because KV rows are short (one head vector), so the 4-byte row
offset costs 4/headDim bytes per element on top of the block
payload — 0.0625 B/elem at headDim 64.

**Reading the headline.** The 4–8× band versus BF16 is the SW3/SW2
operating range (4.43–6.55× measured). SW4 alone is 3.47–3.60×
versus BF16 — this is arithmetic, not a failure of the
implementation: 4-bit codes are exactly 4× before any header, so
no 4-bit format with per-block metadata can beat 4× versus BF16.
Versus the FP32 the engine actually moves today, all three modes
are 6.9–13.1×.

---

## 6. Traffic model for a decode step (arithmetic, not a benchmark)

The numbers below are arithmetic on the measured bytes/parameter
from Table 4, not a live benchmark: they assume batch-1 decode,
where each weight is read once per token and the full KV cache is
read once per token. Take a 70B model (Llama-3.3-70B shape: 80
layers, 8 KV heads, headDim 128 — the repo's own planner test
quotes 327.68 KB of BF16 KV per token across all layers, which
matches 2·8·128·80 = 163,840 elements/token).

| per decode token | BF16 | SW3 weights + SW4 KV | reduction |
|---|---|---|---|
| weights (70B) | 140 GB | 30.1 GB | 4.65× |
| KV read, 4k ctx | 1.31 GB | 0.38 GB | 3.56× |
| KV read, 32k ctx | 10.5 GB | 3.0 GB | 3.56× |
| **total, 4k ctx** | 141.3 GB | 30.5 GB | **4.63×** |
| **total, 32k ctx** | 150.5 GB | 33.1 GB | **4.55×** |

Weights dominate, so the blended reduction tracks the weight mode:
~4.6× with SW3, ~6.4× with SW2. Versus the FP32 the engine moves
today (280 GB of weights), the same step is ~9.2×. At batch > 1 the
weight read is amortized across the batch and KV dominates, so the
blended number moves toward the KV ratio (3.3–3.6×) — the format
still wins, but the headline number is a batch-1 number. (Sparse
attention from note 01 would change the KV term further; the two
techniques compose.)

---

## 7. Limitations, gaps, and what did not work

- **The error bound is not a quality guarantee.** The theorems bound
  numeric error, not perplexity or downstream task metrics. The
  measured per-layer relative RMS errors (11% sw4, 25% sw3, 72%
  sw2 on synthetic weights) say sw2 weights would badly damage a
  model and sw3 is marginal, whatever the bound says. Measuring
  perplexity impact on a real model is future work and needs the
  loader wiring (§8) plus a real model.
- **Theorem N2 is 30–60× loose.** The looseness comes from taking
  maxima over all tokens and the worst-case `‖Δp‖₁` step. A tighter
  bound needs a distributional assumption (near-orthogonal V rows,
  or a bound on the attention concentration `Σ_t p_t²`). **Gap:**
  no such tightened bound is proved here; the measured/bound ratios
  in Table 3 are reported so the gap is visible.
- **SW4 cannot reach 4× vs BF16** (see §5). If a future requirement
  is "4× at 4-bit fidelity", the options are larger blocks (worse
  error adaptation), a 1-byte scale (up to 2× worse step), or
  entropy coding (kills the direct packed matmul). None is free.
- **KV row-offset overhead.** Short KV rows pay 4 bytes/row. A
  dedicated KV layout (one offset per block-group, or offsets only
  per sequence) would recover most of the 0.06 B/elem at headDim 64.
- **4 GiB cap.** Row offsets are Uint32, so one packed tensor is
  capped at 4 GiB (~8B parameters at SW3). The fix is 64-bit offsets;
  the encoder throws instead of wrapping.
- **Block-size alignment.** `matmulPackedAB` requires both tensors
  to share a blockSize and throws otherwise. A production kernel
  would handle misaligned blocks by iterating over the finer
  partition.
- **Encoder cost.** The encoder sorts each block's `|x|` (O(n log n)
  per block). That is fine for offline load-time encoding, but a
  streaming encoder should use nth-element selection for the
  87.5th percentile instead of a full sort.
- **No error compensation.** GPTQ-style Hessian-aware compensation
  would cut the effective error substantially (it is the reason 3–4
  bit GPTQ keeps perplexity), but it needs calibration data and a
  second-order pass; SW-code is a pure post-training format with no
  calibration. Combining the two — SW-code as the storage format
  under a GPTQ-style compensated quantizer — is the most promising
  follow-up.
- **Assumptions restated.** The theorems assume the input domain
  (finite, `|x| ≤ 65504`), round-to-nearest encoding, and the exact
  byte layout of Definition 1. The typical-case remarks (§2) assume
  independent uniform rounding errors and Gaussian weights; they are
  labelled remarks, not theorems, and the measured RMS values next
  to them are the check.

## 8. Integration notes (for the wiring pass)

SW-code is a new parallel path; the loader and engine are untouched.
To wire it in behind a flag: encode each weight tensor with
`encodeMatrix` at load (from the dequantized float32 the loader
already produces), replace the `matmulTransposeB` calls in the
transformer projections with `matmulPackedBT`, quantize K/V with
`quantizeKV` on the paged-cache append path (after RoPE), and run
`attentionPackedKV` in place of `groupedQueryAttention`. Activations
use `encodeMatrix` per token-row plus `matmulPackedAB`. The error
bounds are computable from the packed data alone
(`matmulBTErrorBound`, `kvAttentionErrorBound`), so a deployment can
assert them at runtime per layer instead of trusting them.

## References

All references below were verified by web search on 2026-11-22
(title, authors, venue/year, arXiv identifier confirmed against the
arXiv record). No reference is cited from memory alone.

[1] Frantar, Ashkboos, Hoefler, Alistarh. "GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers."
ICLR 2023. arXiv:2210.17323. One-shot 3–4 bit weight quantization
using approximate second-order (Hessian) information; per-group
scales.

[2] Lin, Tang, Tang, Yang, Dang, Han, et al. "AWQ: Activation-aware
Weight Quantization for LLM Compression and Acceleration." MLSys
2024 (Best Paper). arXiv:2306.00978. Per-channel scaling that
protects the ~1% of salient weights identified via activations.

[3] Dettmers, Pagnoni, Holtzman, Zettlemoyer. "QLoRA: Efficient
Finetuning of Quantized LLMs." NeurIPS 2023. arXiv:2305.14314.
Introduces NF4, a 4-bit NormalFloat codebook with levels at normal
quantiles, one fp32 scale per 64 weights, plus double quantization.

[4] Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. "KIVI: A
Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML 2024.
arXiv:2402.02750. Key cache quantized per-channel, value cache
per-token, at 2 bits; the asymmetric treatment is motivated by
per-channel outliers in keys.
