# softwell Research: Doing Less Work per Token

**Status:** research phase, 2026-11-22. Theorems A1, A2, N1, N2, S1, S2, P1,
P1b, P3, P2 are implemented and tested; the novel paths are wired into the
runtime behind flags (see §9) and benchmarked end to end (see §8 and
`docs/BENCHMARKS.md`).

**Scope.** This paper assembles the five research notes in
`docs/research/` into one document: the survey (`00-survey.md`), the
placement physics (`05-placement-physics.md`), novel attention
(`01-attention.md`), the SW-code number system (`02-numbers.md`), and
speculative/parallel decoding (`03-decoding.md`). It states what is
proven, what is conjectured, and what is unproven — and reports measured
results honestly, including the inventions that do not measure faster.

**Notation.** $P$ = parameter count, $b$ = bits per parameter, $S$ =
context length (tokens), $d$ = model width, $L$ = layers, $B$ = batch
size. "Bytes per token" means bytes that must cross a memory boundary
per generated token.

---

## 1. The governing principle

A virtual machine cannot exceed its hosts' physical FLOPS or
bytes/second. Pooling donors adds their FLOPS and bytes/second together,
but it never makes one operation faster than the fastest single device
that could run it, and it adds a network hop to every cross-donor
transfer. The machine therefore wins only by **doing less work per
token**: fewer FLOPs and fewer bytes moved per generated token.

That principle decides everything downstream. The roofline model
(Williams, Waterman, and Patterson [1]) is the lens: attainable
performance is $\min(\text{peak FLOPS},\ \text{peak bandwidth} \times
\text{operational intensity})$, and autoregressive decode at batch size
1 is the canonical memory-bound case — each step is a matrix–vector
product with operational intensity $\sim$1 FLOP per weight byte, far
below any modern ridge point. Every technique in this paper is, in
roofline terms, either raising operational intensity (batching,
speculative decoding) or lowering the bytes on the x-axis
(quantization, sparsity, KV compression). Nothing here raises the roof
itself — that is the hosts' job.

Amdahl's law [2] caps the other direction: the sequential fraction of
the computation is the autoregressive chain itself — token $t{+}1$
cannot begin until token $t$ exists. Speculative decoding (§7) is the
one technique that attacks this directly, by parallelising *across* the
sequential chain with verification. Everything else parallelises
*within* a step and inherits Amdahl's cap on latency, however good its
throughput.

Landauer's principle [3] closes the door on free lunches: erasing one
bit must dissipate at least $kT\ln 2$ of heat ($\approx 2.9\times
10^{-21}$ J at room temperature). Real hardware dissipates $\sim$9
orders of magnitude more per bit moved, so Landauer's bound never binds
on any machine softwell will run on, and no design here may claim to
approach or beat it. Any claimed win must be traceable to fewer bits
moved or fewer operations performed, never to a new physical effect.

The single most important fact for softwell, due to Shazeer [4]: at
batch size 1, time per token $\approx$ bytes of weights read per token
$\div$ available bandwidth, and the FLOPs are essentially free. That is
why a 100% virtual machine with slow donors can still win — by
shrinking the numerator. (At large batch the workload becomes
compute-bound and the bottleneck moves to FLOPS and KV-cache capacity;
the scheduler must treat batch size as the knob that moves the workload
along the roofline, not as a free throughput multiplier.)

---

## 2. What the state of the art says

The survey (note 00) verified 23 references by web search on
2026-11-22 — title, authors, venue/year, and arXiv identifier confirmed
against the publisher or arXiv record. Reported speedups below are the
papers' own numbers on their own hardware, not softwell measurements;
softwell has measured none of them. Where a claim is the paper's
empirical result rather than a theorem, it is described as such.

**I/O-aware attention.** FlashAttention (Dao et al. [7]) tiles attention,
keeps tiles in SRAM, and recomputes the softmax online, reducing HBM
traffic to $O(Nd)$ and proving a matching lower bound — no exact
attention algorithm can asymptotically beat its HBM access count. It is
*exact*: it does not reduce FLOPs or the asymptotic $O(N^2)$ compute,
only the memory traffic. FlashAttention-2 (Dao [8]) adds
sequence-length parallelism — the same idea as splitting the KV
dimension across donors. Flash-Decoding (Dao et al. [9], a blog post,
not a peer-reviewed paper) splits the KV sequence across parallel
workers for single-query decode, reporting up to 8× faster generation
for very long sequences on CodeLlama-34B; it is the closest published
analogue of softwell's core scheduling problem.

**Speculative decoding.** Leviathan, Kalman, and Matias [10] and,
independently, Chen et al. [11] proved the same result: a small draft
model proposes $\gamma$ tokens, the target scores them all in one
parallel forward, and a modified rejection-sampling rule accepts a
prefix — with the accepted tokens distributed *exactly* as if sampled
from the target alone. Reported: 2–3× (T5-XXL) and 2–2.5× (Chinchilla
70B). The win is governed entirely by the acceptance rate $\alpha$,
and it needs a draft model and a bandwidth-bound target pass.

**Quantization.** FP8 (Micikevicius et al. [12]) is a format standard.
GPTQ (Frantar et al. [13]) quantizes weights to 3–4 bits in one shot
using Hessian information, reporting ~3.25× end-to-end inference
speedup over FP16 on A100. AWQ (Lin et al. [14]) protects the ~1% of
salient weight channels and reports >3× speedup. SmoothQuant (Xiao et
al. [15]) enables W8A8 with negligible loss. All three are weight-only
or need calibration data and dequantization hardware; on a plain CPU
donor, naive INT4 can be *slower* than FP16 because of the unpack cost
— a risk softwell must measure, not assume.

**Sparsity.** SparseGPT (Frantar and Alistarh [16]) prunes large GPT
models to 50–60% unstructured sparsity in one shot without retraining,
with negligible perplexity increase — but unstructured sparsity saves
*time* only if the engine skips zeros efficiently. Mixture-of-Experts
(Shazeer et al. [17]; DeepSeek-V2 [18], 236B total, 21B activated per
token) changes the model, not the runtime: softwell cannot turn a dense
model into an MoE, but it can serve MoE models well, sharding by expert.

**KV-cache compression.** MQA (Shazeer [4]) and GQA (Ainslie et al.
[19]) cut KV traffic by sharing KV heads — architectural, requiring the
model to be built that way. MLA (DeepSeek-V2 [18]) compresses keys and
values into a low-rank latent vector, cutting the KV cache by 93.3%
and boosting maximum generation throughput 5.76× — the strongest
single published result for bytes per token, and the bar softwell's
KV-compression work should be judged against. KIVI (Liu et al. [20])
quantizes the KV cache itself to 2 bits (keys per-channel, values
per-token, full-precision sliding window), reporting 2.6× less peak
memory and 2.35–3.47× throughput — the most directly implementable KV
technique for softwell, needing no retraining. H2O (Zhang et al. [21])
evicts all but heavy-hitter and recent tokens, with a submodular
guarantee about the eviction objective, not about output quality —
useful as an opt-in scheduler policy, dangerous as a default.

**Low-rank methods.** The Linformer (Wang et al. [22]) proved via
Johnson–Lindenstrauss that the attention matrix can be approximated
by a low-rank matrix, reducing self-attention from $O(n^2)$ to $O(n)$.
LoRA (Hu et al. [23]) is a training method, but it has a serving
consequence: the base model is served once and small adapters swap per
request — a genuine bytes-per-token win at the system level.

**Serving and scheduling.** Continuous batching (Orca, Yu et al. [5])
replaced request-level with iteration-level scheduling: 36.9×
throughput over FasterTransformer on GPT-3 175B at equal latency.
Paged KV cache (vLLM, Kwon et al. [6]) applied OS paging to the KV
cache: 2–4× throughput, with the gain growing with sequence length and
model size. Softwell already implements both
(`src/scheduler/batch.ts`, `src/scheduler/kvcache.ts`).

**Synthesis.** No single technique here is a breakthrough for softwell
— each is a constant-factor reduction in bytes or FLOPs per token,
proven on hardware softwell does not have. Their product is what
matters. A 7B model at INT4 weights (8×), KIVI KV (8×), speculative
decoding with $\alpha\approx0.7$ (~2.5× tokens per weight read), and
batching ($B$× amortisation) multiplies to a 100×+ reduction in bytes
per *served* token against the naive FP32 batch-1 baseline — before
any new theorem. That product, not any one paper, is the softwell
thesis. The theorems in §3 state the ceiling that product is
approaching.

---

## 3. Placement physics: the theorems that govern the planner

The machine is 100% virtual: $n$ donor machines, each with local read
bandwidth $B_i$, pairwise link bandwidth $L_{ij}$, and link latency
$\lambda_{ij}$. There is no physical GPU backend. A decode step at
batch size 1 is memory-bandwidth-bound (§1). All theorems below are
stated under explicit assumptions; anything not proved is labelled
**conjecture** or **gap** (see §10).

### 3.1 Theorem P1 — placement lower bound for a sharded matmul

Let $y = Wx$ with $W \in \mathbb{R}^{m \times k}$ at $b$ bits per
element be computed by $n$ donors under Assumption A1 (one-dimensional
sharding: a weight matrix is sharded as whole rows or whole columns,
not arbitrary 2-D tiles), with the input $x$ assembled at one donor
and the output $y$ collected at one donor. Then the network traffic per
evaluation of the matmul is at least

$$\min(nk + m,\;\; k + nm)\cdot \frac{b}{8}\ \text{bytes}.$$

The bound is tight: the pure row-shard achieves $(nk+m)\,b/8$ and the
pure column-shard achieves $(k+nm)\,b/8$. (Proof: a row-sharded donor
needs the entire input; a column-sharded donor emits a full-length
partial sum; any mixed plan is no better than the better pure plan.
Full counting argument in note 05, §2.)

**What this proves.** Splitting one matmul across donors has a
communication cost that grows *linearly in the number of donors*,
while the local read cost falls only as $1/n$. There is therefore a
finite optimal donor count for any tensor-parallel split, and the
planner must choose row- vs column-sharding by the aspect ratio
($m \ge k$ favours rows).

**Corollary P3 (optimal donor count, sequential model).** Under the
cost model $T \ge \max(\text{local bytes}/B,\ \text{network
bytes}/L)$ with uniform $B$ and $L$, the per-token time of the matmul
in Theorem P1 is at least

$$T(n) \ge \max\!\left(\frac{mk\,b}{8nB},\;\; \frac{\min(nk+m,\;k+nm)\,b}{8L}\right).$$

For $m \ge k$ (row shard optimal) the two terms cross at

$$n^* = \sqrt{\frac{mL}{kB}},$$

and $T(n)$ increases for $n > n^*$. *Proven under the sequential cost
model* (no communication/computation overlap). With overlap the
optimum shifts upward but remains finite, because the traffic term is
a lower bound on bytes that must cross the link regardless of
scheduling. (Proving the exact optimum under overlap is a gap.)

$n^*$ is small whenever the link is slow relative to local memory
($L \ll B$) or the matrix is near-square — the usual case for a donor
pool on a commodity network. The planner should compute $n^*$ per
layer and refuse to shard a matmul across more donors than that.

**Proposition P1b (layer-pipelined placement).** If instead whole
layers are placed on single donors (pipeline parallelism), the
per-token network traffic is $2 \cdot q \cdot d \cdot b/8$ bytes,
where $q$ is the number of layer boundaries whose two sides sit on
different donors — there is no factor of $n$ and no per-matmul
broadcast. The cost is latency: one step takes the *sum* of per-donor
layer times, with no parallelism within a step. *Proven by counting.*

**Planner consequence.** P1 and P1b are the two poles the planner
chooses between: tensor-parallel placement parallelises within a step
but pays traffic linear in $n$; pipeline placement pays almost no
traffic but serialises the step. The optimal plan is in general a
hybrid — tensor-parallel within a fast local group, pipeline across
slow links — and the planner's cost function
(`src/planner/cost.ts`) should evaluate exactly these two terms per
layer.

### 3.2 Theorem P2 — information-theoretic lower bound on bytes per token

Consider a dense transformer with $P$ parameters at $b$ bits each,
decoded at batch size 1, under Assumption A2 (aggregate fast storage
directly attached to arithmetic units is $F$ bytes with $F < P\cdot
b/8$ — true for every real donor). Any execution of one decode step
must deliver at least

$$\frac{P\cdot b}{8}\ \text{bytes of parameter data}$$

to arithmetic units, plus at least $2\cdot L \cdot S \cdot d_{kv}\cdot
b_{kv}/8$ bytes of KV-cache data. Consequently, with aggregate usable
read bandwidth $B_{\mathrm{agg}}$ across the donor pool, the token
rate is bounded by

$$\text{tokens/s} \;\le\; \frac{B_{\mathrm{agg}}}{\,P\cdot b/8 \;+\; 2\,L\,S\,d_{kv}\,b_{kv}/8\,}.$$

*Proof sketch.* Every parameter participates in at least one
multiply-add against the single activation vector of the decode step,
so each weight is read at least once per token; by A2 the weights
cannot reside in fast storage between steps, so each weight byte must
cross a memory boundary at least once per token. The KV term is the
same argument applied to attention. The bound is tight for dense
models — a straightforward implementation reads each weight exactly
once.

**What this proves.** The token-rate ceiling is set by *bytes*, and the
only quantities in it are the model size, the precision, the context,
and the pool's bandwidth. No scheduling, no kernel, no placement can
beat it for a dense model at fixed $b$. Every technique in §2 is an
attack on one term of the denominator: quantization shrinks $b$ and
$b_{kv}$; KV compression shrinks $d_{kv}$ or $S$ (eviction); MoE
replaces $P$ by $P_{\mathrm{active}}$; speculative decoding divides
the whole fraction by the expected number of accepted tokens per
target pass; batching by $B$ divides the *weight* term by $B$.

**Worked illustration (arithmetic, not a measurement).** A $P =
7\times10^9$ model at $b = 16$ needs $14$ GB of parameter traffic per
token. At an aggregate pool bandwidth of $100$ GB/s the ceiling is
$\approx 7$ tokens/s at batch 1; at batch $32$ the weight term falls
to $0.44$ GB per token and the ceiling rises until the KV term or
compute takes over. For scale, the bandwidth softwell actually
measured on this machine is $0.189$ GB/s of vRAM read (one donor,
2026-11-22, `docs/BENCHMARKS.md`): at that bandwidth the same model
would be capped at $\approx 0.014$ tokens/s. The gap between $0.014$
and $7$ is entirely bandwidth.

**What P2 does not prove.** It does not bound *latency*; it does not
apply to a model that has been structurally changed (MoE, MLA,
pruning — each changes $P$ or the traffic pattern, and the bound must
be restated for the new structure); and it assumes the weights are
actually read from memory each step. A hypothetical machine with $F
\ge P\cdot b/8$ of fast storage would evade it — which is why A2 is
stated, and why the bound is a statement about real donors, not about
all possible hardware.

### 3.3 "Novel physics": what it is, honestly

There is no new physical law in softwell. The honest translation of
the phrase "novel physics" is (a) information theory and (b) real,
measured host physics. Anything that cannot be cashed out in those
terms is not claimed.

**Information theory.** The content of Theorems P1 and P2 is counting:
how many bits must move, and across which boundary. That is
Shannon-style reasoning applied to a memory hierarchy, not new
physics. The one idea that deserves the name "novel" in this project
is the *virtual* part: the memory hierarchy gains an extra, very slow
level — the donor network — and the placement theorems are the
consequences of optimising over a hierarchy with that level included.
The roofline model already contains the idea; softwell extends it by
one level and by one cost term (network traffic).

**Real host physics.** Three real effects govern the machine, all
standard computer architecture: (1) **Locality** — bytes that stay on
a donor cost that donor's local bandwidth; bytes that cross the
network cost link bandwidth plus latency, and energy per bit grows
with distance and protocol layers crossed (stated qualitatively: exact
per-bit energies are hardware-dependent and have not been measured for
softwell's donors, so no numbers are claimed); (2) **Bandwidth, not
latency, sets throughput** — decode is a streaming workload, and
latency only matters where it serialises steps (pipeline depth, §3.1);
(3) **Landauer's floor is irrelevant in practice** — a decode step
erases of order $10^{5}$–$10^{7}$ bits, so the Landauer floor per
token is of order $10^{-16}$–$10^{-14}$ J, while real hardware
dissipates roughly nine orders of magnitude more per bit moved.

**What is NOT claimed.** No violation of Landauer's principle, the
speed of light, or host memory bandwidth. No new physical effect — no
"virtual" bandwidth beyond the sum of donor bandwidths, no latency
below the speed-of-light delay of the actual links, no energy below
the hosts' real dissipation. No circumvention of Theorem P1 — pooling
more donors into one matmul eventually *increases* per-token latency
(Corollary P3).

---

## 4. Novel attention: Theorems A1 and A2

Two approximate attention variants with **worst-case error bounds
against exact grouped-query attention computed from the actual
data** — no distributional assumptions, no learned parameters, no
training. Implementation: `src/engine/novel-attention.ts`; tests:
`test/novel-attention.test.ts` (9 tests, all passing).

**Shared machinery.** Partition the KV prefix into consecutive blocks
of size $B$. For each block $b$ and KV head, precompute once per KV
cache and reuse for every query the block mean key $\bar{k}_b$, the
block mean value $\bar{v}_b$, and the block radii $R_b = \max_t
\lVert k_t - \bar{k}_b\rVert_2$, $S_b = \max_t \lVert v_t -
\bar{v}_b\rVert_2$. Building the stats costs $O(nd)$ per KV head.
The last $W$ positions (local window, default 16) are always computed
exactly, as is the partial block at the distant boundary.

**Op counting.** One multiply-add on one vector element counts as one
op. Softmax exponentials and scalar max/sum work are $O(n)$ per row —
lower order than the $O(nd)$ matmul-like work — and are not counted,
for the exact and the approximate paths alike.

### 4.1 Theorem A1 — landmark / block-mean attention

For each distant full block $b$, compute the logit perturbation the
block-mean approximation would introduce, $\delta_b = \alpha
\lVert q\rVert R_b$. If $\delta_b \le \delta_{\max}$ (default
$0.05$), replace the whole block by its landmark: one score dot
$\alpha q\cdot\bar{k}_b$ and one value vector $\bar{v}_b$ carrying
multiplicity $B$ in the softmax. Otherwise compute the block exactly.

**Theorem A1.** Let $\delta$ be the largest $\delta_b$ over
approximated blocks ($0$ if none) and $S$ the largest $S_b$ over
approximated blocks. Let $\tilde{o}$ be the landmark-attention
output. Then

$$\lVert\tilde{o} - o\rVert_2 \le (e^{2\delta} - 1) \cdot V_{\max} + S,$$

where $V_{\max} = \max_t \lVert v_t\rVert_2$.

*Proof.* Write $\tilde{z}_t$ for the approximate logits and
$\tilde{v}_t$ for the approximate values, $\tilde{p} =
\mathrm{softmax}(\tilde{z})$. Then $\tilde{o} - o = \sum_t
\tilde{p}_t (\tilde{v}_t - v_t) + \sum_t (\tilde{p}_t - p_t) v_t$.
The value term is at most $S$. For the weight term, Lemma 1
(Cauchy–Schwarz) gives $|z_t - \tilde{z}_t| \le \delta_b \le
\delta$, and Lemma 2 (softmax perturbation) gives
$\lVert\tilde{p} - p\rVert_1 \le e^{2\delta} - 1$, so the weight
term is at most $(e^{2\delta} - 1)\cdot V_{\max}$. ∎

**Corollary (cost).** With $B = \lceil\sqrt{n}\rceil$, prefill over
$n$ rows costs $O(n^{3/2} d)$ per head when every block is
approximated. If a block is incoherent ($\delta_b >
\delta_{\max}$) it costs $B$ dots instead of one, so the worst case
— every block incoherent — is $O(n^2 d)$, i.e. exact attention plus
the statistics pass. The bound holds in both cases; only the speedup
is data-dependent.

### 4.2 Theorem A2 — dynamic sparsity by block screening

Score the local window exactly and every distant full block's
landmark $\ell_b = \alpha q\cdot\bar{k}_b$. Let $M = \max(\max$
local score$, \max_b \ell_b)$. For each distant block, an upper
bound on its *total* attention mass is $\mathrm{mass}_b \le B \cdot
\exp(u_b - M)$ with $u_b = \ell_b + \alpha\lVert q\rVert R_b$. If
that bound is at most `dropThreshold` (default $10^{-3}$), skip the
block entirely. Otherwise score the block exactly. Renormalize the
softmax over the kept positions.

**Theorem A2.** Let $m$ be the sum of the mass upper bounds over the
skipped blocks. Let $\tilde{o}$ be the sparse-attention output. Then

$$\lVert\tilde{o} - o\rVert_2 \le 2 m \cdot V_{\max}.$$

*Proof.* The mass bound is rigorous: for $t$ in block $b$, Lemma 1
gives $z_t \le u_b$, and $M \le \max_t z_t$ (the local scores are
true scores, and each $\ell_b$ is the *mean* of its block's true
scores, hence $\le$ the block's max), so $\sum_{t\in b} p_t \le
B\cdot e^{u_b}/e^M$. Summing over skipped blocks, the actual dropped
mass $m_a$ satisfies $m_a \le m$. Writing $o = A + D$ (kept +
dropped) and $\tilde{o} = A/(1-m_a)$ gives $\lVert\tilde{o} - o\rVert
\le 2 m_a V_{\max} \le 2 m V_{\max}$. ∎

**Corollary (cost).** Per (row, head) the cost is $O((n/B + s\cdot B
+ W)\cdot d)$ where $s$ is the number of surviving blocks. With $B =
\lceil\sqrt{n}\rceil$ and $s$ bounded by a constant (concentrated
attention), prefill is $O(n^{3/2} d)$ per head. Worst case — nothing
skippable — is $O(n^2 d)$.

**Remark.** A2 never evicts KV entries: screening is per query, from
a rigorous mass bound, and a block skipped for one query can be kept
for the next. That is the difference from KV-cache eviction policies
such as H2O, which drop entries permanently on the basis of
accumulated scores.

### 4.3 Measured results (wave 1, `test/novel-attention.test.ts`)

All numbers are counted multiply-add operations under the §4
convention, not wall-clock timings; the implementation is unoptimized
reference TypeScript. Errors are max over heads (and rows, for
prefill) of $\lVert\text{approx} - \text{exact}\rVert_2$; bounds are
the theorems' bounds evaluated on the actual inputs, plus a $10^{-4}$
float32 reassociation slack (the theorems are proved in exact
arithmetic; the slack is the only non-rigorous term, and observed
float32 drift is ~$10^{-6}$–$10^{-4}$, inside it).

A1, single query, GQA ($n=48$, 4 heads, 2 KV heads, $d=8$, $B=8$,
$W=4$): locally coherent keys — error $7.7\times10^{-5}$, bound
$4.3\times10^{-1}$, ops $1200/3072$ ($0.39\times$). Incoherent keys
(iid) — exact fallback: error $0$, bound $6.8\times10^{-4}$, ops
$3104/3072$ ($1.01\times$).

A1, prefill (2 heads, 1 KV head, $d=8$, $B=\lceil\sqrt{n}\rceil$,
$W=8$, coherent keys): $n=64$ — ops $0.54\times$ exact; $n=128$ —
$0.37\times$; $n=256$ — $0.26\times$. Doubling the length multiplies
counted ops by $\approx 2.8$, not $\approx 4$ as exact attention
would. Measured errors are ~3 orders of magnitude below the bound.

A2, single query, concentrated data ($n=64$): error $1.0\times
10^{-4}$, bound $7.8\times10^{-3}$, dropped-mass bound $8.5\times
10^{-4}$, ops $1280/4096$ ($0.31\times$, stats reused).

A2, prefill, concentrated: $n=64$ — ops $0.63\times$; $n=128$ —
$0.43\times$; $n=256$ — $0.28\times$ (ops ratio vs previous $n$:
$2.71$, $2.57$). A2, uniform attention (all keys identical): nothing
is skippable, dropped mass is exactly $0$, output equals exact
attention up to float32 — no false sparsity.

**What did not work.** The first version of the A2 scaling test used
unnormalized Gaussian directions, so on two of three seeds nothing
was separable, nothing was skipped, and prefill ops *exceeded* exact
($1.11\times$, $1.07\times$) while the bound still held. That failure
was instructive — it is exactly the graceful-degradation regime — but
it was the wrong test of the concentrated regime. The lesson is also
a property of the method: A2's win is only as reliable as the
attention's concentration, and a deployment would want to monitor
the achieved skip rate, not assume it.

**Limits.** The speedup is conditional: A1 needs most blocks to
satisfy $\delta_b \le \delta_{\max}$ (locally coherent keys); A2
needs most blocks to be skippable (concentrated attention). Real LLM
KV caches may or may not satisfy these properties. The bounds are
loose — measured errors run ~$10^3\times$ below them — because they
use max radii and the worst-case softmax perturbation. The
$S$-term in Theorem A1 is the main source of looseness on
value-dispersed data; the fallback currently thresholds only on
$\delta_b$, not on $S_b$. No model-quality claim is made: nothing here
says a real model keeps its perplexity under these approximations.
The bounds are per attention head per query; no claim is made about
error accumulation across layers or across generated tokens.

**Relation to prior art.** A1 shares the segment-mean landmark idea
with Nyströmformer (Xiong et al., arXiv:2102.03902) but approximates
the *keys themselves* by their block means, keeps an exact local
window, and carries a per-instance worst-case error bound with an
exact fallback. It is a low-rank-plus-local decomposition in the
spirit of the Linformer (Wang et al., arXiv:2006.04768), but with a
fixed projection and a per-instance bound rather than an expectation
over a random projection. A1/A2 are orthogonal to FlashAttention
(Dao et al., arXiv:2205.14135): they reduce FLOPs, and could in
principle be combined with IO-aware tiling.

---

## 5. SW-code: a low-bit number system (Theorems N1 and N2)

A novel number representation that cuts the bytes moved per token by
4–8× versus BF16 (6–13× versus the FP32 the engine actually moves
today), with *provable* error bounds. Implementation:
`src/engine/novel-numbers.ts`; tests:
`test/novel-numbers.test.ts`. The engine today moves every weight as
float32 (the loader dequantizes BF16/FP16/FP8 to `Float32Array` on
load), so the status quo is 4 bytes per parameter and 4 bytes per KV
element crossing vRAM → vGPU.

### 5.1 The format

A matrix is split into blocks of $n$ consecutive elements along each
row (default $n = 64$). A block is encoded independently as

```
[codes: ceil(n*b/8) bytes][scale: fp16, 2 bytes][outlier count: 1 byte]
[outliers: 3 bytes each = 1 index byte + fp16 value]
```

where $b \in \{2, 3, 4\}$ is the code width (modes `"sw2"`, `"sw3"`,
`"sw4"`) and each code is one of $2L+1$ uniform levels $\{-L, \dots,
+L\} \cdot \Delta$ with $L = 1, 3, 7$, plus one escape code. The block
scale — the *clip* level $Q$ — is stored as fp16 and $\Delta = Q /
L$.

The twist is how $Q$ is chosen. Almost every block quantizer sets the
scale from the block *maximum*, so one huge weight poisons the step
size for the other 63 values. SW-code instead sets the scale from a
*robust quantile* and stores the tail exactly: let $a_{(1)} \ge \dots
\ge a_{(n)}$ be the sorted $|x|$ in the block and $r = \lceil n/8
\rceil$; then $Q = \min(a_{(1)}, 2 \cdot a_{(r)})$, and the outliers
are the elements with $|x| > Q$, each stored exactly as fp16. Because
$Q \ge a_{(r)}$, at most $r - 1$ elements can exceed $Q$, so the
outlier count is bounded by construction ($\le 7$ for a 64-block) and
the worst-case bytes per element is bounded too. Blocks are
variable-length, so a packed tensor is one byte stream plus one
4-byte row offset per row; sequential access is O(1) amortized per
block with a streaming cursor. The matmuls (`matmulPackedBT`,
`matmulPackedAB`) read the packed stream directly — each block is
unpacked into a small lookup table and dotted against float32
activations without ever materializing a float32 copy of the weight
matrix — so the traffic saving is realized at the vRAM → vGPU
transfer, not just on disk. Inputs must be finite with $|x| \le
65504$ (the fp16 maximum); the encoder throws outside that range
instead of silently corrupting data.

### 5.2 Theorem N1 — representation and dot-product error

**Theorem N1 (per-element error).** Let a block have stored scale
$s = \mathrm{fl16}(Q)$ and step $\Delta = s/L$. For every element
$x$ in the block, with $\hat{x}$ its decoded value: non-outlier
($|x| \le Q$): $|x - \hat{x}| \le \Delta/2$ (or $< 2^{-24}$ if
$\Delta = 0$); outlier ($|x| > Q$): $|x - \hat{x}| \le 2^{-10}
\cdot |v| + 2^{-24}$, where $v$ is the stored fp16 value.

*Proof.* Non-outlier: round-to-nearest is within $1/2$ of its
argument. Outlier: $\hat{x} = \mathrm{fl16}(x)$ exactly, and the
fp16 rounding lemma gives the bound. ∎

Note what is *absent*: there is no tail term. An outlier can never
blow up the bound, because it is never quantized on the coarse grid
at all. The bound is essentially tight: on Gaussian weights the
measured max error is within 0.2% of the bound.

**Corollary N1.1 (dot-product error).** Let $E$ be the row error
norm bound of a packed row. Then for any float32 vector $a$,
$|a\cdot\hat{x} - a\cdot x| \le \lVert a\rVert_2 \cdot E$
(Cauchy–Schwarz, summing the per-element bounds over the row).

**Corollary N1.2 (packed × packed).** With $E_A$, $E_B$ the row
error norms and $\lVert\hat{a}\rVert$, $\lVert\hat{b}\rVert$ the
decoded row norms, $|a\cdot b - \hat{a}\cdot\hat{b}| \le E_A\cdot
(\lVert\hat{b}\rVert + E_B) + \lVert\hat{a}\rVert\cdot E_B$.

**Typical case (remark, not a theorem).** For Gaussian weights the
clip is the block max, $Q \approx 2.5\sigma$ for $n = 64$, so if
rounding errors are treated as independent uniform on
$[-\Delta/2, \Delta/2]$, the RMS error is $\Delta/\sqrt{12}$ and a
dot product over $K$ terms has RMS error $\lVert a\rVert_2\cdot
\Delta/\sqrt{12}$, versus the worst-case bound
$\lVert a\rVert_2\cdot\sqrt{K}\cdot\Delta/2$ — a factor
$\sqrt{3K}$ smaller. Measured RMS errors match the $\Delta/\sqrt{12}$
prediction to within 3%.

### 5.3 Theorem N2 — KV-cache quantization error on attention output

Quantize K and V per (token, KV head) vector of length `headDim`
(one block per vector), after RoPE. K and V may use different modes
— K is more sensitive because it moves the softmax, V enters the
output linearly. Let $\varepsilon_K$, $\varepsilon_V$ be the largest
per-row error norms over the KV rows a head reads, $V_{\max}$ the
largest decoded V row norm, and $\delta_{\max} = \lVert q\rVert_2
\cdot \varepsilon_K/\sqrt{d}$ the largest logit error.

**Lemma N2.1 (softmax is ½-Lipschitz from $\ell_\infty$ to
$\ell_1$).** $\lVert\mathrm{softmax}(u) -
\mathrm{softmax}(v)\rVert_1 \le (1/2)\cdot\lVert u - v\rVert_\infty$.
*Proof.* The softmax Jacobian $J = \mathrm{diag}(p) - pp^T$ has
$\infty\to1$ operator norm equal to its largest column sum,
$2p_j(1-p_j) \le 1/2$. ∎

**Theorem N2 (attention output error).** For one query head, with
$o$ the exact attention output and $\hat{o}$ the output computed
from quantized K and V,

$$\lVert\hat{o} - o\rVert_2 \le (1/2) \cdot \delta_{\max} \cdot V_{\max} + \varepsilon_V.$$

*Proof.* Split $\hat{o} - o = \sum_t (\hat{p}_t - p_t)\hat{v}_t +
\sum_t p_t(\hat{v}_t - v_t)$; the first term is bounded by Lemma
N2.1 and the second by $\sum_t p_t = 1$. ∎

### 5.4 Measured results (wave 1, `test/novel-numbers.test.ts`)

Weights are synthetic: Gaussian $N(0, 0.02)$ and a heavy-tailed
"spike" mixture (99% $N(0, 0.02)$, 1% $N(0, 0.2)$), which mimics
the outlier weights seen in real LLMs.

**Encode/decode error vs the Theorem N1 bound** (256×512,
$\sigma = 0.02$): the bound is essentially tight — measured max
error within 0.2% of the bound in every cell (e.g. Gaussian sw4:
max err $5.254\times10^{-3}$, bound $5.258\times10^{-3}$; sw2:
$3.680\times10^{-2}$ vs $3.680\times10^{-2}$). The heavy-tailed
distribution costs almost nothing: spike outliers are escaped
(0.89% of elements) and the max error rises only 26% over Gaussian,
where a max-scaled quantizer would have its step blown up ~10× in
the ~half of blocks that contain a spike.

**Matmul error vs the dot-product bound** (A ~ N(0,1), 16×256; B
weight-like, 128×256; output std 0.32): the bound holds everywhere
with 7–14× slack (worst/bound 0.108–0.140). The RMS column is the
number a model would feel: relative to the 0.32 output std,
per-layer RMS error is 11% (sw4), 25% (sw3), 72% (sw2). That is the
honest price of the format.

**KV-cache quantization vs the Theorem N2 bound** (kvLen 128, 4
heads / 2 KV heads, headDim 64; exact output per-element std
≈ 1): the bound holds in every case, including causal masking and
K with per-channel outliers — but it is 30–60× loose (worst/bound
0.015–0.030). The recommended KV operating point is sw4/sw4 (~2%
per-element output error) or sw4/sw2 (~10%, V only). sw2/sw2 keeps
the bound but the measured 26% per-element error is too large for
quality — the bound is not a quality certificate, only an error
guarantee.

**Measured traffic** (2.1M parameters, 1024×2048): sw2 6.55×,
sw3 4.65×, sw4 3.60× vs BF16 (Gaussian); spike: 6.13×, 4.43×,
3.47×. KV cache, sw4: headDim 64 → 3.25× vs BF16; headDim 128 →
3.56×. The KV numbers are worse than the weight numbers at the same
mode because KV rows are short, so the 4-byte row offset costs
4/headDim bytes per element on top of the block payload.

**Perplexity on a real model (SmolLM-135M, TinyStories).** First
real-model measurement (`scripts/bench-sw-smollm.ts`,
`sw-smollm-bench.json`): 252 tokens, 4×64-token chunks teacher-forced
from a 256 KiB slice at byte offset 1,000,000,000 of
`training/data/TinyStories-train.txt` (322 complete stories in the
slice; BOS prepended per chunk, fp32 KV throughout, identical tokens
for every setting). Baseline (fp32 weights): mean NLL 3.862, PPL
47.54. SW-code weights: sw4 mean NLL 4.164, PPL 64.32 (1.35× baseline,
+0.30 nats; worse on all 4 chunks, paired ΔNLL +0.46/+0.09/+0.28/+0.37);
sw3 mean NLL 6.499, PPL 664.38 (14.0×); sw2 mean NLL 21.112, PPL
1.47×10⁹ (31M×) — the model is destroyed, worse than uniform
(ln 49152 ≈ 10.8), and its greedy continuation is gibberish
(".academthon follows") where sw3 already degenerates to repetition
(" the day the day"). The verdict is plain: **sw2 and sw3 weights are
unusable, and even sw4 costs +35% perplexity** — the synthetic
per-layer RMS errors (11%/25%/72%) understated the damage, because the
error compounds over 30 layers and the lm_head is quantized too.
SW-code weights need error compensation (GPTQ-style, §5.5) or a
different operating point before they are viable as a weight format;
sw4 is the only mode that still produces coherent text. Throughput on
this backend moved the wrong way too: prefill 1.88 tok/s baseline vs
0.59 (sw4), 1.01 (sw3), 1.02 (sw2); decode ~1.00 tok/s baseline vs
0.73–0.75 for all three SW modes (4-token greedy sample; host
contended, loadavg 1.3–2.9, so treat the absolute tok/s as noisy and
the ratios as the signal). Weight bytes: 538 MB fp32 vs 189/172/156 MB
(sw4/sw3/sw2); load including encode 3.1 s vs 52.8/15.4/14.6 s (sw4's
52.8 s is partly first-run JIT and contention, not a mode cost).
Caveats: 252 tokens only (SE of mean NLL ≈ 0.16–0.21 nats, so sw4's
+0.30 nats is ~1.5–2 SE — the direction is solid from the paired
chunks, the exact ratio is not); chunks cut mid-story with BOS
prepended, so the absolute PPL is slightly pessimistic while the
paired comparison is fair. KV-cache perplexity impact is still
unmeasured.

**Reading the headline.** The 4–8× band versus BF16 is the SW3/SW2
operating range (4.43–6.55× measured). SW4 alone is 3.47–3.60×
versus BF16 — this is arithmetic, not a failure of the
implementation: 4-bit codes are exactly 4× before any header, so no
4-bit format with per-block metadata can beat 4× versus BF16.
Versus the FP32 the engine actually moves today, all three modes
are 6.9–13.1×.

**Traffic model for a decode step (arithmetic, not a benchmark).**
For a 70B model (Llama-3.3-70B shape, 80 layers, 8 KV heads,
headDim 128): SW3 weights + SW4 KV give 30.5 GB per token at 4k
context vs 141.3 GB BF16 — a 4.63× reduction; 33.1 GB vs 150.5 GB
at 32k context (4.55×). Weights dominate, so the blended reduction
tracks the weight mode. At batch > 1 the weight read is amortized
and KV dominates, so the blended number moves toward the KV ratio
(3.3–3.6×).

### 5.5 What is new, and what is not

The ingredients are not new: block scaling with a shared exponent is
the standard low-bit format (GPTQ uses per-group scales [13]; AWQ
uses per-channel scaling [14]; QLoRA's NF4 uses a
distribution-matched codebook [24]); asymmetric KV quantization is
KIVI's core finding [20]. What is new here as a package is: (1) the
robust-quantile scale with exact outlier escape — the scale comes
from the 87.5th percentile of $|x|$, not the max, and the tail is
stored exactly as fp16, so the per-element bound has no tail term
and the outlier count is bounded by construction; (2) one uniform
bit-packing + lookup-table decoder serving 2/3/4-bit modes for
weights, activations, and KV with a single streaming layout; (3)
matmuls that consume the packed stream directly, so the traffic
saving is realized at the vRAM → vGPU transfer; (4) end-to-end
bounds in closed form (N1, N2) computed from the packed data alone
and checked by the tests.

Two designs were considered and rejected. A non-uniform (NF4-style)
Gaussian-quantile codebook would cut RMS error by ~25%, but the
max-error bound would become the largest codebook gap instead of a
single clean $\Delta/2$. A power-of-two scale (1-byte header instead
of fp16) would save 1 byte per block but coarsens the step by up to
2×, doubling the error bound.

**Limits.** The error bound is not a quality guarantee: the measured
per-layer relative RMS errors (11% sw4, 25% sw3, 72% sw2 on synthetic
weights) say sw2 weights would badly damage a model and sw3 is
marginal, whatever the bound says — and the real-model measurement
(§5.4) is worse than that: sw4 +35% perplexity, sw3 14×, sw2
destroyed. KV-cache perplexity impact on a real model is still
unmeasured. Theorem N2 is 30–60× loose. SW4 cannot
reach 4× vs BF16. KV row-offset overhead (4 bytes/row) costs 0.06
B/elem at headDim 64. Row offsets are Uint32, so one packed tensor is
capped at 4 GiB (~8B parameters at SW3). `matmulPackedAB` requires
both tensors to share a blockSize. The encoder sorts each block's
$|x|$ (O(n log n) per block) — fine for offline load-time encoding.
No error compensation: GPTQ-style Hessian-aware compensation would
cut the effective error substantially, but it needs calibration data
and a second-order pass; combining the two — SW-code as the storage
format under a GPTQ-style compensated quantizer — is the most
promising follow-up.

---

## 6. (Reserved — see §7 for decoding; numbering follows the research notes)

---

## 7. Speculative and parallel decoding: Theorems S1 and S2

Autoregressive decoding spends one target forward pass per token.
This section gives two schemes that spend one forward pass per
*several* tokens, proves they produce exactly the tokens plain greedy
decoding would produce (and, for speculative decoding, exactly the
target's sampling distribution), and derives the cost model that
decides whether they win. Implementation:
`src/engine/novel-decoding.ts` (`SpeculativeDecoder`,
`JacobiDecoder`, `NGramDrafter`, cost model); tests:
`test/novel-decoding.test.ts`. Wired into the scheduler behind the
`--decoding` flag (see §9).

**Cost model.** A batched forward with $R$ rows costs $1 + (R-1)
\cdot \beta$ single-row forwards, where $\beta \in [0, 1]$ is the
marginal cost of one extra row ($\beta \approx 0$ when the forward
is memory-bandwidth-bound and weights are streamed once per forward,
$\beta \approx 1$ when it is compute-bound and every row pays full
matmul FLOPS). All speedup/breakeven functions take $\beta$
explicitly; nothing assumes a value for it.

**Assumptions.** (A1) The target is deterministic given its cache
state. (A2) KV-cache `append` at a position *overwrites* that
position's K/V slot — true of `BlockPagedKVCache`, false of an
append-only cache. (A3) The drafter never calls the target and never
sees target logits. (A4) One forward's cost is $1 + (R-1)\cdot\beta$
for $R$ rows, with a single $\beta$ for the hardware at hand. (A5)
Single sequence, batch size 1.

**The drafter.** The default drafter is built from what exists: the
text itself. `NGramDrafter`
proposes the next token by finding the longest suffix of the context
(up to 8 tokens) that has occurred before and copying the token that
followed its most recent occurrence; if no suffix has occurred before,
it repeats the last token. It costs O(K · 8 · |context|) integer
comparisons — no forward pass, no weights, no training. A real draft
model is also supported (`ModelDrafter`, `--draft-model <dir>`); it
changes only the acceptance rate α, not the theorems. It is
deliberately weak: it can only predict continuations that have
already been seen, so its acceptance rate on a trained model
generating varied text is expected to be low (single-digit percent is
the honest prior; not measured on a trained model because the repo
has none). Its virtue is that it is *free*, which makes the cost
model a clean function of one measured number, the acceptance rate
$\alpha$. Replacing it with a trained draft model changes $\alpha$
and nothing else in the theorems — the verification logic and the
exactness proofs are drafter-agnostic, provided (A3) holds.

### 7.1 Theorem S1 — speculative decoding

A cheap drafter proposes $K$ tokens, one batched target forward
verifies them all, the longest matching prefix is kept plus one
correction/bonus token. State alternates between two round types: in
round A the pending logits predict the next token and $K$ drafts are
verified; in round B an accepted-but-unfed *carry* token is row 0 of
the verification forward, followed by $K$ fresh drafts. The carry row
is the price of KV-cache honesty: a correction/bonus token must be
fed before its successor's logits exist, and feeding it inside the
next verification forward repairs the cache — the rejected draft that
previously occupied the carry's position is overwritten when the
carry is fed (A2).

**Theorem S1a (greedy exactness).** Under (A1)–(A3), greedy
speculative decoding (temperature 0) outputs exactly the token
sequence greedy autoregressive decoding outputs, for any drafter and
any $K \ge 1$. *Proof sketch (induction on output positions).* The
token emitted at each position is either a draft verified equal to
the target's argmax, or a correction/bonus token defined as the
target's argmax — hence the greedy token in both cases — and the
cache state left behind is the greedy state. ∎

**Theorem S1b (sampled distributional exactness).** Under (A1)–(A3),
with temperature > 0, each token speculative decoding emits is
distributed exactly as the target model's own token at that
position. *Proof sketch.* The drafter is deterministic, so its draft
distribution $q$ is a point mass at the draft token $d$; the
verification step is speculative sampling (Leviathan et al. [10];
Chen et al. [11]) specialised to $q(d) = 1$: accept $d$ with
probability $\min(1, p(d))$; on rejection, resample from the residual
$\mathrm{norm}(\max(0, p - q))$. ∎

**Theorem S1c (expected speedup).** Let each draft token be accepted
independently with probability $\alpha$. Then one round yields
$E[T] = 1 + \alpha + \alpha^2 + \cdots + \alpha^K$ tokens, and the
expected wall-clock speedup over plain decoding is

$$S(\alpha, K, \beta) = \frac{1 + \alpha + \cdots + \alpha^K}{1 + K\beta}.$$

The breakeven acceptance rate $\alpha^*$ is 0 at $\beta = 0$ and 1
at $\beta = 1$. *Corollary ($\beta = 1$ is unwinnable).* At $\beta =
1$ no speculative scheme of this form beats plain decoding in
wall-clock: a round produces at most $K+1$ tokens from $K+1$ rows of
work. The win, when it exists, is entirely a $\beta < 1$ phenomenon.

Breakeven table ($K = 4$): $\beta=0$: $\alpha^*=0.000$, $S=4.52\times$
at $\alpha=0.95$; $\beta=0.5$: $\alpha^*=0.741$, $S=1.51\times$;
$\beta=1$: $\alpha^*=1.000$, $S=0.905\times$.

*Remark (without independence).* If each draft is accepted with
conditional probability $\ge \alpha$ given the earlier drafts, the
independence formula is a *lower bound* under a uniform conditional
acceptance floor. Real acceptance is positively correlated along a
draft, which is why measured tokens/round (4.00) sits *below* the
independence prediction (4.52) at the same $\hat{\alpha}$.

### 7.2 Theorem S2 — Jacobi (parallel multi-token) decoding

Fix a block of $K$ positions. Start from an initial guess supplied
entirely by the drafter. One Jacobi iteration feeds the whole guess
in one batched forward and replaces every position by the target's
argmax conditioned on the current guess prefix. Iterate until the
guess is a fixed point, or for $K$ iterations; a 1-token block
short-circuits to plain decoding. If $K$ iterations pass without a
detected fixed point, the guess is nevertheless correct (Theorem
S2a) but the cache holds the previous guess's K/V at some positions,
so one extra *commit* forward feeds the correct tokens.

**Theorem S2a (convergence in ≤ K+1 forwards).** Under (A1)–(A2),
greedy Jacobi decoding of a $K$-token block converges to exactly the
greedy autoregressive block. Let $J$ be the first index where the
initial guess is wrong ($J = K$ if entirely right). Then the block
takes exactly $K - J + 1$ target forwards, and never more than $K+1$.
*Proof sketch (induction on positions).* Position 0 depends only on
the true prefix, so after iteration 1 it is the greedy token and
never changes; suppose positions $0\ldots i-1$ are greedy after
iteration $i$; position $i$'s update in iteration $i+1$ is the argmax
conditioned on those positions, hence greedy. ∎

**Theorem S2b (expected forwards).** If each initial guess is
independently correct with probability $\gamma$, the expected
forwards per block are $E[T] = \sum_{j=0}^{K-1} (K-j+1)\gamma^j
(1-\gamma) + \gamma^K$, and the expected wall-clock speedup is $S =
K / (E[T]\cdot(1 + (K-1)\beta))$. At $\beta = 1$, $S = 1/E[T] \le 1$,
with equality only at $\gamma = 1$.

Breakeven table ($K = 4$): $\beta=0$: $\gamma^*=0.519$,
$S=2.71\times$ at $\gamma=0.95$; $\beta=0.5$: $\gamma^*=0.936$,
$S=1.08\times$; $\beta=1$: $\gamma^*=1.000$, $S=0.68\times$. Note
the $\beta = 0$ breakeven is *not* 0: even with free rows, a
fully-wrong guess costs $K+1$ forwards for $K$ tokens.

Jacobi is greedy-only in this implementation: a fixed point is
defined by argmax, and there is no sampled analogue that preserves
the target distribution without a verification step (which would turn
it back into speculative decoding). Unlike speculative decoding,
Jacobi re-attends the same block prefix on every iteration, so its
wall-clock cost has an extra term the $\beta$ model does not capture.

### 7.3 Measured results (wave 1, `test/novel-decoding.test.ts`)

The demo model (10M parameters, random weights) is untrained, and its
greedy output collapses to a single repeated token ("Hello world" →
57 × 24; "The quick brown fox jumps" → 130 × 24). That collapse is
the whole story of the measured acceptance rate: the n-gram drafter's
repeat-last fallback predicts a constant stream perfectly, so
$\hat{\alpha}$ = 19/20 = 0.95 on *both* prompts. **This number must
not be quoted as the drafter's quality.** It is an artifact of the
target's degeneracy, and on a trained model generating varied text
the same drafter would score far lower. What the demo *does*
establish is exactness (both schemes match the baseline token-for-token
on both prompts, including the 24-token measurement runs) and the
arithmetic of the cost model: speculative 7 forwards vs 25 for 24
tokens (0.29 forwards/token vs the 0.221 prediction — the gap is the
correlation effect plus the final partial round); Jacobi 10 forwards
/ 9 iterations / 6 blocks.

**Wall-clock: the β ≈ 1 regime.** Timing on this machine (demo-10m,
24 tokens, median of 3 runs): baseline autoregressive 993 ms;
speculative, K = 4: 1094 ms (**0.91×**); Jacobi, K = 4: 1877 ms
(**0.53×**). The β measurement behind this: after prefilling "Hello
world", a 1-row decode forward takes 17.4 ms, a 2-row forward 36.6
ms, a 4-row forward 73.4 ms — cost(4 rows)/cost(1 row) = 4.23,
implied $\hat{\beta} \approx 1.08$. Rows cost *slightly more* than
linear — the excess over 1.0 is partly attention over growing
prefixes inside the batch — so $\hat{\beta} \approx 1$ is the honest
summary: this backend is compute-bound, and a batched forward is
(nearly) one single-row forward per row.

Consequences, all consistent with the theorems: speculative measured
0.91× vs the β = 1 prediction of 0.905× — the model predicts the
wall-clock outcome to within measurement noise. Jacobi measured 0.53×
vs the β = 1 prediction of 0.68× at γ = 0.95 — the model is
optimistic here (it does not charge for re-attending the block prefix
on every iteration), and Jacobi's run-to-run variance is large
(1319–2017 ms). Neither scheme can win wall-clock at β = 1 — and
the measurements confirm both lose. The actionable conclusion is not
"speculation is useless" but "speculation's win lives in β": it needs
bandwidth-bound hardware (real GPUs serving large models, where decode
is weight-streaming-bound and β ≈ 0) and/or a drafter with a high α
on real text.

**Measured with a real draft model (2026-11-22, `spec-draft-bench.json`).**
Target SmolLM-135M (real trained model, 49152 vocab); draft a 17.3M
Llama model trained on ~100k TinyStories tokens (final loss 6.25,
top-1 next-token accuracy 7.56% — lightly trained, by design for this
proof run). Single short run, prompt "The capital of France is",
16 tokens, temp 0, K = 4, quiet host: baseline 3.45 tok/s (16 tokens
in 4,634 ms, 16 forwards); speculative 0.89 tok/s (16 tokens in
18,002 ms, 15 forwards, 69 rows, 14 rounds); **acceptance 2/16 =
0.125**; draft ran 51 forwards. Tokens identical to the baseline
(S1 exactness holds with a real drafter). β probe on the target:
1-row forward 235 ms, 5-row 1,156 ms, β̂ ≈ 0.98. S1 predicts
0.23× at α = 0.125, K = 4, β = 0.98 (E = 1.14 tokens/round over a
cost of 4.92 forwards/round); measured **0.26×** — consistent.
**Correction:** the wave-1 α̂ = 0.95 above is superseded — the first
real acceptance on real text is 0.125, and it is low because the draft
is lightly trained, not because the scheme is broken. But the β ≈ 1
cap is the binding constraint: even at α = 1 this backend caps
speculative speedup at ~1×, so a better-trained draft alone cannot
make it win wall-clock here; it needs β < 1 (bandwidth-bound
hardware) as well. Caveats: one run only (16 draft proposals, so α̂
is noisy), short prompt only — no long-prompt bench was run for this
report, and no medians.

**Proof gaps.** (A2) is load-bearing and implementation-specific: if
the cache ever stops overwriting on append, both schemes silently
produce wrong tokens. A regression test that asserts overwrite
semantics on `BlockPagedKVCache` would close this; it does not exist
yet. (A4) is a modelling choice, not a theorem. S1b's sequence-level
claim relies on the cache argument of S1a applied per position; a
fully formal treatment would need the joint distribution over cache
states. The sampled path has no distributional test — only
determinism under a fixed seed. The independence model for α and γ is
a modelling assumption (real acceptance is correlated). That β is a
single constant per hardware is assumed; it drifts with prefix length
and batch composition (§7.3's $\hat{\beta} \approx 1.1$ already shows
superlinearity the model ignores).

---

## 8. Measured end-to-end (wave 2)

Wave-2 wiring and benchmarks: `src/bench/novel-bench.ts`
(`npm run bench:novel`), raw results in
`novel-bench-results-2026-11-22.json` (65 rows), full tables in
`docs/BENCHMARKS.md`. Machine: AMD EPYC 9D25 126-Core Processor
(4 cores), 7.7 GiB RAM, Node v24.22.1, linux x64. Models: demo-10m
and demo-100m (104.37M params, hidden 1024, 12 layers), both random
weights. Part B: prompt 128, 16 generated tokens, greedy, one rep
per config. Part A: synthetic K/V, 16 heads / 8 KV heads / d=64,
one rep per cell.

**Attention scaling (Part A, prefill wall-clock vs exact).** Each
mode wins only on its own regime and falls back to exact elsewhere:

| regime / mode | n=256 | n=512 | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|---|
| coherent / landmark | 1.47× | 4.66× | 9.84× | 15.41× | 20.41× |
| concentrated / sparse | 1.92× | 5.17× | 6.64× | 10.46× | 9.04× |
| incoherent / either | 0.91–0.96× | 0.90–0.96× | 0.90–0.96× | 0.92–0.94× | 0.72–1.15× |

Counted ops fall to 0.04× (landmark/coherent) and 0.12×
(sparse/concentrated) at n=4096. Exact prefill scales ~4× per
doubling; landmark on coherent data ~2.4×, sparse on concentrated
data ~2.7×. Errors stay ~10³× below the bounds (landmark/coherent
max err 3.3–3.9e-3 vs bound ~11; sparse/concentrated 0.75–1.8e-3
vs bound 0.1–1.3); no bound was violated in any run; fallback cells
have max err exactly 0.0.

The regimes are calibrated to data the theorems cover: coherent =
block-constant keys + N(0, 0.001) noise (δ_b ≈ 0.01 ≤ δ_max = 0.05);
concentrated = block 0 at +12u, others at −12u, queries at +4u
(logit gap ≈ 12, above the ln(B/1e-3) ≈ 10.7 the mass bound
needs). The first attempt (noise 0.01, alignment ±3u) measured
~1.0× everywhere because the thresholds never triggered — at d=64,
δ_b grows as √d for fixed per-element noise, so the approximation
condition is a strong one on real KV caches, whose within-block
dispersion is unmeasured. One implementation cost is visible:
landmark on incoherent data at n=4096 measures 0.72×, because the
reference implementation builds a per-item object list for every
position in the fallback path.

**End-to-end (Part B, decode tokens/s vs baseline, prompt 128).**

| config | demo-10m | demo-100m |
|---|---|---|
| baseline | 52.08 tok/s (TTFT 2,471 ms) | 5.17 tok/s (TTFT 24,656 ms) |
| landmark | 1.04× | 0.99× |
| sparse | 1.00× | 0.96× |
| sw2 / sw3 / sw4 weights | 0.35× / 0.21× / 0.20× | 0.14× / 0.14× / 0.14× |
| kv-sw3 / kv-sw4 | 0.82× / 0.79× | 0.59× / 0.58× |
| speculative (K=4, α=92%) | 0.80×, 5 forwards vs 16 | 0.56×, 5 forwards vs 16 |
| jacobi (K=4) | 0.76×, 6 forwards vs 16 | 0.52×, 6 forwards vs 16 |

Attention end-to-end is a wash at prompt 128 (0.96–1.04×):
attention is a small fraction of a step at that length, and the demo
models' keys are incoherent, so the paths degrade to exact. SW-code
weights measure 5–7× slower in-process (packed-matmul unpack cost,
no slow link to save bytes across) — the realized saving is memory
(10m stage 37.9 MB → 3.3–5.7 MB). Quantized KV measures 0.58–0.82×.
Speculative/Jacobi cut forwards but lose wall-clock, as Theorems
S1c/S2b predict at β ≈ 1; the 92% acceptance rate is an artifact of
the random-weight models collapsing to a constant token. Part B
configs ran sequentially and the host drifted during the 100m run,
so 100m TTFT for later configs is inflated ~1.5×; decode ratios are
the more reliable comparison. Absolute times vary ~40% between runs;
ratios are stable.

The honest summary of wave 2 is the same as wave 1: the theorems
hold, the error bounds hold, the cost models predict the wall-clock
outcomes — and on this compute-bound CPU backend the inventions that
save *bytes* do not yet save *time*, because time here is spent on
FLOPs, not on moving bytes across a slow link. That is a property of
the backend, not a refutation of the theorems — and it is exactly
what the benchmarks are for. The missing measurement is end-to-end
at 1k–4k prompts, where the Part A attention win should show in the
full model.

---

## 9. What is wired, behind which flags

Defaults are the baseline everywhere. Flags are parsed by
`src/engine/options.ts` (`parseEngineFlags`, `validateEngineOptions`,
`resolveEngineOptions`) and flow from `softwell` (CLI) to workers
(engine modes) and the scheduler (decoding mode).

```
softwell run <model> [--attention exact|landmark|sparse]
                     [--weights fp32|sw2|sw3|sw4]
                     [--kv fp32|sw2|sw3|sw4]
                     [--decoding autoregressive|speculative|jacobi]
                     [--draft-length K]   (default 4)
```

| Flag | Effect | Where |
|---|---|---|
| `--attention landmark` | Theorem A1 in the transformer: prefill uses `prefillLandmarkAttention` (stats built once per prefill, reused across rows); decode uses `landmarkAttention` with an incrementally maintained block-stats cache (`extendBlockStats`, O(B·d) per step) | `src/engine/transformer.ts`, `src/engine/novel-attention.ts` |
| `--attention sparse` | Theorem A2, same structure | same |
| `--weights sw2\|sw3\|sw4` | Weights SW-code encoded at load; projections and lm_head use `matmulPackedBT` (packed stream, no fp32 copy held) | `src/engine/transformer.ts`, `src/engine/novel-numbers.ts` |
| `--kv sw2\|sw3\|sw4` | KV cache quantized per (token, KV head) after RoPE; attention via `attentionPackedKV`; the packed KV is maintained incrementally (one-token encode + `concatPackedTensors` per step) and rebuilt from the float cache on overwrite | same |
| `--decoding speculative` | Theorem S1 in the scheduler: one batched verification forward per round across all greedy sequences; sampled sequences fall back to autoregressive | `src/scheduler/batch.ts`, `src/engine/novel-decoding.ts` |
| `--decoding jacobi` | Theorem S2 in the scheduler: one batched forward per Jacobi iteration, plus a commit forward for unconverged blocks; greedy only | same |

**Invariants.** With all flags at their defaults, behaviour is
exactly v1: the existing scheduler/engine tests pass unchanged, and
`npm run demo` is unaffected. The engine rejects
landmark/sparse + quantized KV loudly at construction (no packed
variant of the block-mean attention exists yet) instead of silently
falling back. SW-code weights are rejected for MoE models at load
(no packed MoE kernel yet). `EngineOptions` is additive in
`src/contracts.ts`: `createStageExecutor(manifest, assignment,
options?)` — existing callers pass nothing and get the baseline.

**What is NOT wired.** The scheduler's speculative/Jacobi modes are
single-sequence-per-round batched across sequences, but they assume
the stage chain can carry multi-row decode batches (it can — prefill
already does). Multi-sequence speculative verification shares one
forward; per-sequence drafters share one `NGramDrafter`. The
`--decoding` flag does not change the API server's contract; it
changes how the scheduler generates. Landmark/sparse attention does
not yet combine with quantized KV. SW-code does not yet cover MoE
experts, the embedding lookup (a row gather, not a matmul — packing
would not help), or activations (`matmulPackedAB` exists but is not
wired into the transformer).

---

## 10. What is proven, what is conjectured, what is unproven

| Item | Status |
|---|---|
| Theorem A1 (landmark attention error bound) | **Proven** in exact arithmetic; checked by tests on every run (measured error never exceeded the bound). Float32 slack ($10^{-4}$) is a safety margin, not a proved roundoff bound. |
| Theorem A2 (sparse attention error bound) | **Proven** in exact arithmetic; checked by tests. Same float32 caveat. |
| A1/A2 sub-quadratic prefill cost | **Proven conditional** on data properties (coherent keys / concentrated attention). Worst case degrades to exact + stats pass. |
| Theorem N1 (SW-code per-element + dot-product error) | **Proven**; bound essentially tight (measured max error within 0.2% of bound). |
| Theorem N2 (KV attention output bound) | **Proven**; bound 30–60× loose. Tightening it needs a distributional assumption — **gap**. |
| SW-code traffic ratios (4.65× sw3, 6.55× sw2 vs BF16) | **Measured** on synthetic weights (Gaussian + spike). Perplexity impact on a real model **unmeasured**. |
| Theorem S1 (speculative exactness + expected speedup) | **Proven** modulo (A1)–(A3); expected speedup under the independence model. Sampled path has no distributional test. |
| Theorem S2 (Jacobi convergence + expected forwards) | **Proven** modulo (A1)–(A2); expected forwards under independence. |
| Speculative/Jacobi wall-clock win | **Conditional**: needs β < 1 and (for speculative) α above breakeven. **Measured to lose** on this CPU backend (0.91× / 0.53×, β ≈ 1). |
| Theorem P1 (communication floor, 1-D sharding) | **Proven** (complete counting argument under A1). Tight. |
| Corollary P3 (optimal donor count) | **Proven under the sequential cost model**. Exact optimum under overlap is a **gap**. |
| Proposition P1b (pipeline traffic) | **Proven** by counting. |
| Theorem P2 (bytes-per-token floor) | **Proven** under A2. Tight for dense models; must be restated per architecture for MoE/MLA/pruned models. |
| Hybrid placement (tensor-parallel within fast groups, pipeline across slow links) is optimal | **Conjecture.** P1 and P1b give the two poles; optimality of the hybrid, and an algorithm to find it, are not proven. The planner currently uses a heuristic cost model. |
| 2-D (block) sharding beats the P1 bound | **Open / gap.** A1 excludes 2-D tiles. The corresponding 2-D lower bound (of the form $\Omega((\sqrt{n}\,k + \sqrt{n}\,m)b/8)$ by a surface-to-volume argument) is **conjectured, not proven here**. The most important theoretical gap in the placement work. |
| Speculative decoding across donors (draft on a cheap donor) | **Conjecture** about the system design; the underlying algorithm's correctness is proven in the cited papers. |
| A trained draft model can reach α ≈ 0.6–0.9 on real text | **Conjecture** (literature reports this range for small draft models; not measured here). The n-gram drafter's α = 0.95 is a degeneracy artifact of the random-weight demo model. |
| SW-code perplexity impact on a real model | **Unmeasured.** Per-layer RMS errors (11% sw4, 25% sw3, 72% sw2 on synthetic weights) say sw2 would badly damage a model and sw3 is marginal — needs a real model to settle. |
| Attention error accumulation across layers / tokens | **Unproven.** Bounds are per head per query; composition through a deep network is future work. |
| Energy-per-bit figures for softwell's donors | **Unmeasured.** Qualitative scaling only (§3.3). |
| Per-donor $B_i$, pairwise $L_{ij}$ and $\lambda_{ij}$ | **Unmeasured** on real donors. Needed before the placement bounds stop being arithmetic and start being predictions. |

**Next steps.** (1) Prove or refute the conjectured 2-D sharding bound
— it decides whether the planner should ever use 2-D tiles. (2)
Extend Corollary P3 to the overlap model and to non-uniform $B_i$,
$L_{ij}$. (3) Restate Theorem P2 for the architectures softwell will
actually serve (MoE, MLA, INT4). (4) Measure perplexity impact of
SW-code and of A1/A2 on a real model with real KV caches. (5) Measure
β and α on real traffic and on bandwidth-bound hardware, where the
decoding theorems predict a win. (6) A real draft model (or an
EAGLE-style draft head) — the highest-leverage follow-up for
speculative decoding. (7) GPTQ-style error compensation under SW-code
as the storage format. (8) A tighter Theorem N2 bound under a
distributional assumption.

---

## 11. The novel math as vNPU instructions

> The instruction names, operand formats, and API surface below
> are the core device workstream's API as landed
> (`src/hw/vnpu.ts`, `IVirtualNpu` in `src/contracts.ts`,
> design in `docs/VNPU.md`). Nothing here changes the status
> of the theorems themselves (§10).

The vNPU's instruction set (`docs/VNPU.md` §1) is the theorems
in silicon form. The mapping, as implemented:

| Theorem | vNPU instruction(s) | What changes vs the engine implementation |
|---|---|---|
| A1 (landmark attention, §4.1) | `ATTN_LANDMARK` (`vnpu.attention` / `vnpu.prefillAttention`, `mode: "landmark"`) | The block statistics ($\bar{k}_b$, $\bar{v}_b$, $R_b$, $S_b$) become device-resident state, built once per KV cache (`vnpu.buildBlockStats`, extended incrementally by `vnpu.extendBlockStats`) and reused per query; the $\delta_b \le \delta_{\max}$ gate is a per-block bound check inside the instruction, with `deltaMax` carried per instruction, not a host-side branch. The error bound is the same bound, evaluated on the same data, and reported with the result. |
| A2 (dynamic sparsity, §4.2) | `ATTN_SPARSE` (`vnpu.attention` / `vnpu.prefillAttention`, `mode: "sparse"`) | The mass bound $B \cdot \exp(u_b - M)$ is computed on-device from resident landmark scores, with `dropThreshold` carried per instruction; skipped blocks are never fetched from vRAM at all (today the reference implementation still touches them to decide). The per-query screening and the no-eviction guarantee are unchanged. |
| N1 (SW-code representation, §5.2) | `MATMUL_SW` (`vnpu.matmulSw`), `QUANTIZE` / `DEQUANTIZE` (`vnpu.quantizeSw` / `vnpu.dequantizeSw`) | The packed stream becomes the device's native weight format: `MATMUL_SW` consumes it directly with no FP32 materialisation, as `matmulPackedBT` does today — the difference is that the format is now the *storage* format in SRAM/vRAM, not an encoding applied at load. The per-element bound and the no-tail-term property are unchanged; `matmulSw` reports the N1 dot-product bound with the result. |
| N2 (KV quantization, §5.3) | `KV_QUANTIZE` (`vnpu.quantizeKv`), `ATTN_PACKED_KV` (`vnpu.attentionPackedKv`) | Quantizing K/V per (token, KV head) after RoPE becomes a device instruction issued at append time, writing packed KV straight to vRAM. The bound is the same bound. Exact attention over the packed KV exists as `ATTN_PACKED_KV`, so the exact path does not inherit the engine's landmark/sparse + quantized-KV rejection (§9) — but landmark/sparse *over packed KV* remains open (see below). |

What does **not** move into silicon:

- **S1/S2 (speculative/Jacobi decoding, §7)** stay in the
  scheduler. They are policies over forward passes, and their
  exactness proofs are about the *sequence* of forwards, not
  about any one op. The vNPU exposes no `SPECULATE` op.
- **P1/P2/P3 (placement physics, §3)** stay in the planner.
  They govern *where* instructions run, not *what* they do;
  the vNPU's scheduler applies them to instruction placement
  (VNPU.md §3.2) but does not re-prove them.

Open questions remaining (see also VNPU.md §6):

1. Whether `ATTN_LANDMARK` / `ATTN_SPARSE` can operate over
   packed KV (`KV_QUANTIZE` output). The vNPU answers the
   exact case with `ATTN_PACKED_KV`; the approximate cases
   still need block statistics over quantized rows.
2. Settled in v1: the A1/A2 thresholds (`delta_max`,
   `drop_threshold`) are per-instruction immediates, passed
   per call in the attention options — not per-device
   registers. The theorems do not care; the API does, and it
   chose immediates.
3. SW-code at sw2/sw3 remains an issuable instruction
   (`vnpu.matmulSw` accepts any `SwMode`), because the device
   must not silently refuse a valid instruction — but the
   scheduler must never select it for a real model without
   error compensation, and the benchmark must record the
   perplexity cost alongside any throughput win.

---

## References

All references below were verified by web search on 2026-11-22
(title, authors, venue/year, arXiv identifier confirmed against the
arXiv or publisher record). No reference is cited from memory alone.
Flash-Decoding [9] is the only non-peer-reviewed entry and is labelled
as such. Reported speedups are the cited papers' own figures.

[1] S. Williams, A. Waterman, and D. Patterson. "Roofline: An
Insightful Visual Performance Model for Multicore Architectures."
*Communications of the ACM* 52(4), 2009. DOI:
10.1145/1498765.1498785.

[2] G. M. Amdahl. "Validity of the Single Processor Approach to
Achieving Large Scale Computing Capabilities." *AFIPS Conference
Proceedings* 30, 1967, pp. 483–485.

[3] R. Landauer. "Irreversibility and Heat Generation in the Computing
Process." *IBM Journal of Research and Development* 5(3), July 1961,
pp. 183–191.

[4] N. Shazeer. "Fast Transformer Decoding: One Write-Head is All You
Need." arXiv:1911.02150, 2019.

[5] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun. "Orca: A
Distributed Serving System for Transformer-Based Generative Models."
*Proceedings of OSDI '22*, 2022.

[6] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E.
Gonzalez, H. Zhang, and I. Stoica. "Efficient Memory Management for
Large Language Model Serving with PagedAttention." *Proceedings of
SOSP '23*, 2023. arXiv:2309.06180.

[7] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. "FlashAttention:
Fast and Memory-Efficient Exact Attention with IO-Awareness."
*NeurIPS 2022*. arXiv:2205.14135.

[8] T. Dao. "FlashAttention-2: Faster Attention with Better
Parallelism and Work Partitioning." *ICLR 2024*. arXiv:2307.08691.

[9] T. Dao, D. Haziza, F. Massa, and G. Sizov. "Flash-Decoding for
long-context inference." Blog post, Stanford CRFM / Princeton NLP,
October 2023. https://crfm.stanford.edu/2023/10/12/flashdecoding.html
— *not peer-reviewed; figures are the authors' benchmarks.*

[10] Y. Leviathan, M. Kalman, and Y. Matias. "Fast Inference from
Transformers via Speculative Decoding." *ICML 2023*.
arXiv:2211.17192.

[11] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J.
Jumper. "Accelerating Large Language Model Decoding with Speculative
Sampling." arXiv:2302.01318, 2023.

[12] P. Micikevicius et al. "FP8 Formats for Deep Learning."
arXiv:2209.05433, 2022.

[13] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. "GPTQ:
Accurate Post-Training Quantization for Generative Pre-trained
Transformers." *ICLR 2023*. arXiv:2210.17323.

[14] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G.
Xiao, X. Dang, C. Gan, and S. Han. "AWQ: Activation-aware Weight
Quantization for LLM Compression and Acceleration." *MLSys 2024*
(Best Paper). arXiv:2306.00978.

[15] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han.
"SmoothQuant: Accurate and Efficient Post-Training Quantization for
Large Language Models." *ICML 2023*. arXiv:2211.10438.

[16] E. Frantar and D. Alistarh. "SparseGPT: Massive Language Models
Can Be Accurately Pruned in One-Shot." *ICML 2023*.
arXiv:2301.00774.

[17] N. Shazeer, A. Mirhoshini, K. Maziarz, A. Davis, Q. Le, G.
Hinton, and J. Dean. "Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017*.
arXiv:1701.06538.

[18] DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model." arXiv:2405.04434, 2024.

[19] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón,
and S. Sanghai. "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints." *EMNLP 2023*. arXiv:2305.13245.

[20] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen,
and X. Hu. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
Cache." *ICML 2024*. arXiv:2402.02750.

[21] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y. Tian, C. Ré, C. Barrett, Z. Wang, and B. Chen. "H₂O: Heavy-Hitter
Oracle for Efficient Generative Inference of Large Language Models."
*NeurIPS 2023*. arXiv:2306.14048.

[22] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma. "Linformer:
Self-Attention with Linear Complexity." arXiv:2006.04768, 2020.

[23] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L.
Wang, and W. Chen. "LoRA: Low-Rank Adaptation of Large Language
Models." *ICLR 2022*. arXiv:2106.09685.

[24] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. "QLoRA:
Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
arXiv:2305.14314.

[25] Xiong, Zeng, Chakraborty, Tan, Fung, Li, Singh. "Nyströmformer:
A Nyström-Based Algorithm for Approximating Self-Attention."
arXiv:2102.03902.

[26] Y. Fu, P. Bailis, I. Stoica, and H. Zhang. "Break the Sequential
Dependency of LLM Inference Using Lookahead Decoding."
arXiv:2402.02057 (ICML 2024). Jacobi-iteration parallel decoding with
no draft model; the formulation of §7.2 builds on it.

---

*Source notes.* This paper assembles `docs/research/00-survey.md`,
`01-attention.md`, `02-numbers.md`, `03-decoding.md`, and
`05-placement-physics.md` (all 2026-11-22). The theorems and proofs
are stated in full in those notes; this paper states each theorem and
its proof idea and points at the note for the complete proof. Section
numbering for the decoding material follows note 03 (§7 here = §2–3
there); the empty §6 above preserves the research-note numbering so
cross-references stay stable.
