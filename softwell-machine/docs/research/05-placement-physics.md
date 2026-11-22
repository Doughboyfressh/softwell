# Placement and the Physics of the Virtual Machine

**Softwell research note 05 — 2026-11-22.**
This note states the two theorems that govern softwell's planner, and then says
plainly what "novel physics" means here — and what it does not mean.

**Standing assumptions.** The machine is 100% virtual: $n$ donor machines,
each with local read bandwidth $B_i$ bytes/s, pairwise link bandwidth
$L_{ij}$ bytes/s, and link latency $\lambda_{ij}$. There is no physical GPU
backend. A decode step at batch size 1 is memory-bandwidth-bound (survey
note 00, §1.4): time per token is dominated by bytes moved, not FLOPs.
All theorems below are stated under explicit assumptions; anything not
proved is labelled **conjecture** or **gap**.

---

## 1. Definitions and cost model

**Definition 1 (donor pool).** A set $D = \{1,\dots,n\}$ of donors. Donor $i$
can read its own memory at $B_i$ bytes/s and can send/receive over a link of
bandwidth $L_{ij}$ bytes/s to donor $j$. For the theorems we use the uniform
case $B_i = B$, $L_{ij} = L$; the non-uniform case only weakens the bounds by
replacing $B, L$ with $\min_i B_i$, $\min_{ij} L_{ij}$.

**Definition 2 (work per decode token).** A dense transformer with $P$
parameters at $b$ bits each, $L$ layers, width $d$, context $S$, KV head
dimension $d_{kv}$ per layer at $b_{kv}$ bits. One decode step (batch size 1)
performs $\approx 2P$ FLOPs and touches every parameter at least once.

**Definition 3 (traffic).** The *network traffic* of a placement is the total
bytes sent between distinct donors during one decode step. *Local traffic*
is bytes read from a donor's own memory. Time is modelled, conservatively,
as sequential: $T \ge \max(\text{local bytes}/B,\ \text{network bytes}/L)$
per donor/link, with no overlap between communication and computation. Any
overlap can only improve on these bounds, never worsen them, so all lower
bounds below remain valid under overlap.

**Assumption A1 (one-dimensional sharding).** A weight matrix is sharded
across donors as whole rows or whole columns (tensor parallelism as
practised), not as arbitrary 2-D tiles. This matches every real
tensor-parallel implementation; 2-D sharding is discussed in §5 as a gap.

**Assumption A2 (fast storage is smaller than the model).** The aggregate
fast storage directly attached to arithmetic units (registers, CPU caches)
is $F$ bytes with $F < P\cdot b/8$. True for every real donor (megabytes of
cache against gigabytes of weights). It is what forces parameters to cross a
memory boundary once per token instead of living next to the arithmetic.

---

## 2. Theorem P1 — placement lower bound for a sharded matmul

**Theorem P1 (communication floor of tensor-parallel placement).**
Let $y = Wx$ with $W \in \mathbb{R}^{m \times k}$ at $b$ bits per element be
computed by $n$ donors under Assumption A1, with the input $x$ assembled at
one donor and the output $y$ collected at one donor. Then the network traffic
per evaluation of the matmul is at least

$$\min(nk + m,\;\; k + nm)\cdot \frac{b}{8}\ \text{bytes}.$$

The bound is tight: the pure row-shard achieves $(nk+m)\,b/8$ and the pure
column-shard achieves $(k+nm)\,b/8$.

*Proof sketch.* Under A1 each donor holds either a set of whole rows or a
set of whole columns. Let $r$ donors be row-sharded and $c$ column-sharded,
$r + c = n$. A row-sharded donor needs the entire input $x$ ($k\cdot b/8$
bytes) to compute its rows; the row-sharded donors together emit all $m$
output rows ($m\cdot b/8$ bytes). A column-sharded donor needs only its
column slice of $x$ (slices sum to $k\cdot b/8$ bytes across all such donors)
but emits a full-length partial sum ($m\cdot b/8$ bytes each). Total traffic
is therefore at least $(rk + k + m + cm)\,b/8$. If $m \ge k$, this is at
least the pure-row figure: $(rk + k + m + cm) - (nk + m) = k + c(m-k) \ge 0$.
If $m < k$, it is at least the pure-column figure:
$(rk + k + m + cm) - (k + nm) = r(k-m) + m > 0$. Hence any mixed plan is
no better than the better pure plan, and the bound follows. ∎

**What this proves.** Splitting one matmul across donors has a communication
cost that grows *linearly in the number of donors*, while the local read cost
falls only as $1/n$. There is therefore a finite optimal donor count for any
tensor-parallel split, and the planner must choose row- vs column-sharding
by the aspect ratio ($m \ge k$ favours rows). This is the precise form of
"pooling helps, but not without limit".

**Corollary P3 (optimal donor count, sequential model).** Under the cost
model of Definition 3, with uniform $B$ and $L$, the per-token time of the
matmul in Theorem P1 is at least

$$T(n) \ge \max\!\left(\frac{mk\,b}{8nB},\;\; \frac{\min(nk+m,\;k+nm)\,b}{8L}\right).$$

For $m \ge k$ (row shard optimal) the two terms cross at

$$n^* = \sqrt{\frac{mL}{kB}},$$

and $T(n)$ increases for $n > n^*$. *Proof.* The first term is the local
weight read, split evenly over $n$ donors; the second is Theorem P1's
traffic over a link of bandwidth $L$. Equating them for the row case,
$mk/(nB) = nk/L$, gives $n^2 = mL/(kB)$. ∎

*Remark.* $n^*$ is small whenever the link is slow relative to local memory
($L \ll B$) or the matrix is near-square — which is the usual case for a
donor pool on a commodity network. The planner should compute $n^*$ per
layer and refuse to shard a matmul across more donors than that. Note the
corollary assumes no overlap of communication and computation; with overlap
the optimum shifts upward but remains finite, because the traffic term is a
lower bound on bytes that must cross the link regardless of scheduling.
(Proving the exact optimum under overlap is a gap; see §5.)

**Proposition P1b (layer-pipelined placement).** If instead whole layers are
placed on single donors (pipeline parallelism), the per-token network traffic
is $2 \cdot q \cdot d \cdot b/8$ bytes, where $q$ is the number of layer
boundaries whose two sides sit on different donors — there is no factor of
$n$ and no per-matmul broadcast. The cost is latency: one step takes the
*sum* of per-donor layer times, with no parallelism within a step.
*Proof.* Each boundary crossing carries one activation vector of $d$ elements
in each direction; counting boundaries gives the figure. ∎

**Planner consequence.** P1 and P1b are the two poles the planner chooses
between: tensor-parallel placement (P1) parallelises within a step but pays
traffic linear in $n$; pipeline placement (P1b) pays almost no traffic but
serialises the step. The optimal plan is in general a hybrid — tensor-parallel
within a fast local group, pipeline across slow links — and the planner's
cost function (`src/planner/cost.ts`) should evaluate exactly these two
terms per layer.

---

## 3. Theorem P2 — information-theoretic lower bound on bytes per token

**Theorem P2 (minimum memory traffic per decode token).** Consider a dense
transformer with $P$ parameters at $b$ bits each, decoded at batch size 1,
under Assumption A2. Any execution of one decode step must deliver at least

$$\frac{P\cdot b}{8}\ \text{bytes of parameter data}$$

to arithmetic units, plus at least $2\cdot L \cdot S \cdot d_{kv}\cdot
b_{kv}/8$ bytes of KV-cache data. Consequently, with aggregate usable read
bandwidth $B_{\mathrm{agg}}$ (bytes/s) across the donor pool, the token rate
is bounded by

$$\text{tokens/s} \;\le\; \frac{B_{\mathrm{agg}}}{\,P\cdot b/8 \;+\; 2\,L\,S\,d_{kv}\,b_{kv}/8\,}.$$

*Proof sketch.* Every parameter of a dense transformer participates in at
least one multiply-add against the single activation vector of the decode
step: each weight matrix multiplies the activation, so each weight is read
at least once per token. By A2 the weights cannot reside in fast storage
between steps, so each weight byte must cross a memory boundary at least
once per token — that is the $P\cdot b/8$ term, and it is tight, since a
straightforward implementation reads each weight exactly once. The KV term
is the same argument applied to attention: the step must read the keys and
values of all $S$ context tokens in every layer to compute attention scores.
Dividing the per-token byte count by the available bytes per second gives
the rate bound. ∎

**What this proves.** The token-rate ceiling is set by *bytes*, and the only
quantities in it are the model size, the precision, the context, and the
pool's bandwidth. No scheduling, no kernel, no placement can beat it for a
dense model at fixed $b$ — it is a property of the workload, not of softwell.
Every technique in survey note 00 is, in this light, an attack on one term
of the denominator: quantization shrinks $b$ and $b_{kv}$; KV compression
shrinks $d_{kv}$ or $S$ (eviction); MoE replaces $P$ by $P_{\mathrm{active}}$;
speculative decoding divides the whole fraction by the expected number of
accepted tokens per target pass; batching by $B$ divides the *weight* term
by $B$ (weights are read once per batch step, not once per token — the KV
term is not amortised the same way, since each sequence has its own KV).

**Worked illustration (arithmetic, not a measurement).** A $P = 7\times10^9$
model at $b = 16$ needs $14$ GB of parameter traffic per token. At an
aggregate pool bandwidth of $100$ GB/s the ceiling is $\approx 7$ tokens/s
at batch 1; at batch $32$ the weight term falls to $0.44$ GB per token and
the ceiling rises until the KV term or compute takes over. For scale, the
bandwidth softwell actually measured on this machine is $0.189$ GB/s of
vRAM read (one donor, 2026-11-22, `docs/BENCHMARKS.md`): at that bandwidth
the same model would be capped at $\approx 0.014$ tokens/s. The gap between
$0.014$ and $7$ is entirely bandwidth — which is why the survey's techniques
matter and why no claim about softwell's speed can be made until the
bandwidth is measured on real donors.

**What P2 does not prove.** It does not bound *latency* (a single token can
be slow even if throughput is high), it does not apply to a model that has
been structurally changed (MoE, MLA, pruning — each changes $P$ or the
traffic pattern, and the bound must be restated for the new structure), and
it assumes the weights are actually read from memory each step. A
hypothetical machine with $F \ge P\cdot b/8$ of fast storage would evade it
— which is why A2 is stated, and why the bound is a statement about real
donors, not about all possible hardware.

---

## 4. "Novel physics": what it is, honestly

There is no new physical law in softwell. What follows is the honest
translation of the phrase "novel physics" into (a) information theory and
(b) real, measured host physics. Anything that cannot be cashed out in those
terms is not claimed.

### 4.1 Information theory

The content of Theorems P1 and P2 is counting: how many bits must move, and
across which boundary. That is Shannon-style reasoning applied to a memory
hierarchy, not new physics. The one idea that deserves the name "novel" in
this project is the *virtual* part: the memory hierarchy gains an extra,
very slow level — the donor network — and the placement theorems are the
consequences of optimising over a hierarchy with that level included. The
roofline model (survey note 00, §1.1) already contains the idea; softwell
extends it by one level and by one cost term (network traffic).

### 4.2 Real host physics

Three real effects govern the machine, all standard computer architecture:

1. **Locality.** Bytes that stay on a donor cost that donor's local bandwidth
   ($B_i$); bytes that cross the network cost link bandwidth ($L_{ij}$) plus
   latency ($\lambda_{ij}$). Energy per bit moved likewise grows with the
   distance and the number of protocol layers crossed — on-chip movement is
   cheapest, DRAM access costs far more, network transit costs far more
   again. (Stated qualitatively: the exact per-bit energies are
   hardware-dependent and have not been measured for softwell's donors, so no
   numbers are claimed here.)
2. **Bandwidth, not latency, sets throughput.** Decode is a streaming
   workload: the binding constraint is bytes per second, and latency only
   matters where it serialises steps (pipeline depth, §2, Proposition P1b).
3. **Landauer's floor is irrelevant in practice.** Erasing one bit costs at
   least $kT\ln 2 \approx 2.9\times10^{-21}$ J at room temperature
   (Landauer [3]). A decode step erases at least the logits buffer and the
   per-layer activation buffers — of order $10^{5}$–$10^{7}$ bits — so the
   Landauer floor per token is of order $10^{-16}$–$10^{-14}$ J. Real
   hardware dissipates roughly nine orders of magnitude more per bit moved.
   Landauer's principle therefore never binds on softwell, and it is cited
   here only to close the door on free-lunch claims, not as a design target.

### 4.3 What is NOT claimed

- **No violation of Landauer's principle, the speed of light, or host memory
  bandwidth.** A virtual device cannot move a byte faster than the slowest
  physical link it crosses, and cannot compute a FLOP the hosts do not
  perform. Every speedup in this project is a reduction in bytes or FLOPs
  per token, as bounded by Theorem P2.
- **No new physical effect.** There is no claimed phenomenon in which
  virtualisation changes the physics of the hosts — no "virtual" bandwidth
  beyond the sum of donor bandwidths, no latency below the speed-of-light
  delay of the actual links, no energy below the hosts' real dissipation.
- **No circumvention of Theorem P1.** Pooling more donors into one matmul
  eventually *increases* per-token latency (Corollary P3). Any benchmark
  showing otherwise would be measuring something else (e.g. batching, or a
  different sharding).
- **No measured numbers in this note.** The worked illustration in §3 is
  arithmetic from stated assumptions plus one cited in-repo measurement
  (0.189 GB/s vRAM read). No new measurement is claimed.

---

## 5. What is proven, what is conjectured, what is missing

| Item | Status |
|---|---|
| Theorem P1 (communication floor, 1-D sharding) | **Proven** (proof sketch above; full proof is the counting argument, which is complete under A1). Tight. |
| Corollary P3 (optimal donor count) | **Proven under the sequential cost model** (no comm/compute overlap). With overlap the optimum shifts up; the exact optimum under overlap is a **gap**. |
| Proposition P1b (pipeline traffic) | **Proven** by counting. |
| Theorem P2 (bytes-per-token floor) | **Proven** under A2 (fast storage < model size). Tight for dense models; must be restated per architecture for MoE/MLA/pruned models. |
| Hybrid placement (tensor-parallel within fast groups, pipeline across slow links) is optimal | **Conjecture.** P1 and P1b give the two poles; optimality of the hybrid, and an algorithm to find it, are not proven. The planner currently uses a heuristic cost model. |
| 2-D (block) sharding beats the P1 bound | **Open / gap.** A1 excludes 2-D tiles. A 2-D shard needs only a row-slice of $x$ and emits only a column-slice of partials per donor, so the P1 floor does not apply as stated; the corresponding 2-D lower bound (of the form $\Omega((\sqrt{n}\,k + \sqrt{n}\,m)b/8)$ by a surface-to-volume argument) is **conjectured, not proven here**. This is the most important theoretical gap in the note. |
| Speculative decoding across donors (draft on a cheap donor) | **Conjecture** about the system design; the underlying algorithm's correctness is proven in the cited papers (survey note 00, §3). |
| Energy-per-bit figures for softwell's donors | **Unmeasured.** Qualitative scaling only (§4.2). |

**Next steps for this line of work.** (1) Prove or refute the conjectured 2-D
sharding bound — it decides whether the planner should ever use 2-D tiles.
(2) Extend Corollary P3 to the overlap model and to non-uniform $B_i$,
$L_{ij}$, which is what the planner actually faces. (3) Restate Theorem P2
for the architectures softwell will actually serve (MoE, MLA, INT4), so the
ceiling the implementation is judged against is the right one. (4) Measure:
per-donor $B_i$, pairwise $L_{ij}$ and $\lambda_{ij}$, and the real
bytes-per-token of the reference engine, so the bounds stop being arithmetic
and start being predictions.

---

## References

[1] S. Williams, A. Waterman, and D. Patterson. "Roofline: An Insightful
Visual Performance Model for Multicore Architectures." *Communications of
the ACM* 52(4), 2009. (Survey note 00, [1].)

[2] G. M. Amdahl. "Validity of the Single Processor Approach to Achieving
Large Scale Computing Capabilities." *AFIPS Conference Proceedings* 30,
1967, pp. 483–485. (Survey note 00, [2].)

[3] R. Landauer. "Irreversibility and Heat Generation in the Computing
Process." *IBM Journal of Research and Development* 5(3), July 1961,
pp. 183–191. (Survey note 00, [3].)

All other background references are in survey note 00; no additional
citations are introduced here, and no citation in this note is unverified.
The theorems P1, P2, P3, and P1b are original to this note and are stated
with their assumptions and proof status in §5.
