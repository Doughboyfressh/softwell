# Survey: Doing Less Work per Token

**Softwell research note 00 — 2026-11-22.**
Scope: the real state of the art relevant to softwell's governing principle —
a virtual machine cannot exceed its hosts' physical FLOPS or bytes/second, so it
wins only by reducing FLOPs and bytes per generated token. This survey covers
attention I/O, speculative decoding, quantization, sparsity, KV-cache
compression, low-rank methods, and serving/scheduling, plus the theoretical
backdrop that makes those techniques the right ones.

**Method.** Every reference below was verified to exist by web search on
2026-11-22: title, authors, venue/year, and (where given) arXiv identifier
were confirmed against the publisher or arXiv record. No reference is cited
from memory alone. Reported speedups are the papers' own numbers on their own
hardware, not softwell measurements; softwell has measured none of them.
Where a claim is the paper's empirical result rather than a theorem, it is
described as such.

**Notation.** $P$ = parameter count, $b$ = bits per parameter, $S$ = context
length (tokens), $d$ = model width, $L$ = layers, $B$ = batch size. "Bytes per
token" means bytes that must cross a memory boundary per generated token.

---

## 1. Theoretical backdrop

### 1.1 The roofline model

Williams, Waterman, and Patterson [1] model attainable performance as
$\min(\text{peak FLOPS},\ \text{peak bandwidth} \times \text{operational intensity})$,
where operational intensity is FLOPs per byte of DRAM traffic. A kernel is
either compute-bound (flat part of the roof) or memory-bound (sloped part).
Autoregressive decode at batch size 1 is the canonical memory-bound case:
each step is a matrix–vector product, so operational intensity is $\sim$1 FLOP
per weight byte, far below any modern ridge point. The roofline is therefore
the right lens for softwell: every technique in this survey is, in roofline
terms, either raising operational intensity (batching, speculative decoding)
or lowering the bytes on the x-axis (quantization, sparsity, KV compression).
Nothing here raises the roof itself — that is the hosts' job.

**Limits.** The roofline is a bound, not a prediction: it ignores latency,
cache behaviour, and network effects, all of which matter on a donor pool
(see §1.4 and research note 05).

**Build on.** Use the roofline as the planner's cost model: the planner
(`src/planner/cost.ts`) should estimate per-token time as
$\max(\text{FLOPs}/\text{FLOPS},\ \text{bytes}/\text{bandwidth})$ per donor,
plus a network term the original model does not have.

### 1.2 Amdahl's law

Amdahl [2] argued that the sequential fraction of a computation caps the
speedup from parallelism, however many processors are added. For softwell the
sequential fraction is the autoregressive chain itself: token $t{+}1$ cannot
begin until token $t$ exists. Speculative decoding (§3) is the one technique
that attacks this directly, by parallelising *across* the sequential chain
with verification. Everything else in this survey parallelises *within* a
step and therefore inherits Amdahl's cap on latency, however good its
throughput.

### 1.3 Landauer's principle

Landauer [3] showed that erasing one bit of information must dissipate at
least $kT\ln 2$ of heat — about $2.9\times10^{-21}$ J at room temperature.
Two consequences for softwell. First, it is a floor, not a target: real
hardware dissipates $\sim$9 orders of magnitude more per bit moved, so
Landauer's bound never binds on any machine we will run on, and no softwell
design may claim to approach or beat it. Second, it is the correct answer to
"is there a physical free lunch?": no. Any claimed win must be traceable to
fewer bits moved or fewer operations performed, never to a new physical
effect. (See research note 05, §4, for the full statement of what is *not*
claimed.)

### 1.4 Decode is memory-bandwidth-bound

Shazeer [4] gave the first precise analysis: incremental (decode) inference
is slow not because of arithmetic but because of the memory-bandwidth cost of
repeatedly loading the keys, values, and weights. His multi-query attention
proposal (§6) follows directly from that diagnosis. Yu et al. [5] and Kwon et
al. [6] built the serving systems on top of it. The practical form of the
claim, now uncontroversial: at batch size 1, time per token $\approx$
bytes of weights read per token $\div$ available bandwidth, and the FLOPs are
essentially free. This is the single most important fact for softwell: it is
why a 100% virtual machine with slow donors can still win — by shrinking the
numerator.

**Limits.** The claim holds at small batch. At large batch, decode becomes
compute-bound (operational intensity rises with $B$), and the bottleneck
moves to FLOPS and to KV-cache capacity. Softwell's scheduler must therefore
treat batch size as the knob that moves the workload along the roofline, not
as a free throughput multiplier.

---

## 2. I/O-aware attention

### 2.1 FlashAttention

Dao et al. [7] observed that standard attention is I/O-bound, not
compute-bound: it materialises the $N \times N$ attention matrix in HBM. Their
algorithm tiles the computation, keeps tiles in SRAM, and recomputes the
softmax online, reducing HBM traffic to $O(Nd)$ and proving a matching lower
bound — no exact-attention algorithm can asymptotically beat its HBM access
count for a range of SRAM sizes. Reported: up to 7.6× faster attention on
GPT-2, 3× end-to-end training speedup, memory linear in sequence length.

**Limits.** FlashAttention is *exact*: it does not reduce FLOPs or the
asymptotic $O(N^2)$ compute, only the memory traffic. Its proof is about the
GPU HBM/SRAM hierarchy; on softwell's donors the "SRAM" is the host CPU
cache and the "HBM" is host RAM plus the network, so the constants change
completely even though the tiling idea transfers.

**Build on.** The tiling + online-softmax structure is directly implementable
in softwell's attention kernel (`src/engine/attention.ts`): never materialise
the $S \times S$ score matrix; stream KV pages in blocks. The lower-bound
proof is the template for note 05's Theorem P2.

### 2.2 FlashAttention-2

Dao [8] reworked the algorithm's parallelism: fewer non-matmul FLOPs,
parallelisation along the sequence-length dimension (not just batch and
heads), and warp-level work partitioning. Reported: ~2× over FlashAttention,
50–73% of theoretical peak FLOPS on A100, up to 225 TFLOPS/s training
GPT-style models.

**Limits.** The gains are about GPU occupancy and warp scheduling — hardware
structure softwell does not have. The transferable part is the sequence-length
parallelism, which is exactly what a donor pool needs: it is the same idea as
splitting the KV dimension across donors.

**Build on.** Sequence-parallel attention decomposition as the primitive for
distributing one head's KV across multiple donors (see note 05, §2).

### 2.3 Flash-Decoding

Dao, Haziza, Massa, and Sizov [9] (blog post, October 2023, not a peer-reviewed
paper) extended FlashAttention to the decode regime, where the query length is
1 and the batch × heads parallelism leaves most of the device idle. The fix:
split the KV sequence itself across parallel workers and combine partial
results with a log-sum-exp reduction. Reported: up to 8× faster generation
for very long sequences on CodeLlama-34B.

**Limits.** It is an engineering report, not a theorem; the 8× figure is for
long contexts on A100. At short context the extra reduction step costs more
than the parallelism saves — the split factor must be chosen adaptively.

**Build on.** This is the closest published analogue of softwell's core
scheduling problem: how to parallelise a single-query decode across many
workers. Softwell's planner should implement adaptive KV-splitting with the
same split-and-reduce shape, choosing the split count from batch size, head
count, and donor count.

---

## 3. Speculative decoding

Leviathan, Kalman, and Matias [10] and, independently, Chen et al. [11]
introduced the same idea: a small draft model proposes $\gamma$ tokens
autoregressively; the large target model scores all $\gamma$ proposals in one
parallel forward pass; a modified rejection-sampling rule accepts a prefix of
the draft and resamples at the first rejection. Both papers *prove* the
accepted tokens are distributed exactly as if sampled from the target model
alone — the speedup is lossless by construction, not approximate. Reported:
2–3× (Leviathan et al., T5-XXL) and 2–2.5× (Chen et al., Chinchilla 70B).

**What it proves.** The correctness theorem is the valuable part: expected
tokens per target pass $= (1-\alpha^{\gamma+1})/(1-\alpha)$ for acceptance rate
$\alpha$, so the win is governed entirely by $\alpha$, the draft model's
fidelity. The cost model is also the point: verification costs one weight read
no matter how many tokens are verified, because decode is bandwidth-bound —
speculative decoding converts the bandwidth cost of $\gamma$ tokens into the
bandwidth cost of one.

**Limits.** It needs a draft model (training cost, or a smaller model in the
same family), it helps only at small batch (at large batch the target pass is
already compute-bound and verification is no longer free), and the speedup
collapses when $\alpha$ is low. It does not reduce total FLOPs — it increases
them; it reduces *serial* weight reads.

**Build on.** This is softwell's highest-leverage decoding theorem (handled in
detail by the companion decoding note). For the virtual machine it has a
second benefit the papers do not discuss: the draft model can run on a *cheap*
donor while the target weights sit on the pool, so the draft's serial latency
hides behind the target's single parallel pass. The open question for softwell
is what draft to use when the target is an arbitrary open-weight model the
user supplies — a small model in the same family, or a distilled draft, or
the target itself with early exits (unproven; treat as conjecture).

---

## 4. Quantization

Quantization is the most direct attack on bytes per token: it shrinks $b$ in
Theorem P2 of note 05, and every halving of $b$ is a doubling of the token-rate
ceiling at fixed bandwidth.

### 4.1 FP8

Micikevicius et al. [12] specified the FP8 interchange format (E4M3 and E5M2)
and showed training up to 175B parameters matching 16-bit quality with
unchanged hyperparameters. This is a format standard, not a compression
algorithm: its value for softwell is that 8-bit weights are a *supported*
target, not a hack.

### 4.2 INT4 weight quantization: GPTQ and AWQ

Frantar et al. [13] (GPTQ, ICLR 2023) quantize weights to 3–4 bits in one shot
using approximate second-order (Hessian) information, with negligible accuracy
loss — 175B models in ~4 GPU-hours — and report ~3.25× end-to-end inference
speedup over FP16 on A100. Lin et al. [14] (AWQ, MLSys 2024, Best Paper)
observe that only ~1% of weight channels are salient, protect them by
activation-aware per-channel scaling, and report >3× speedup over FP16
inference with a 4-bit model, including the first deployment of Llama-2-70B
on mobile GPUs.

**Limits.** Both are weight-only: activations stay FP16, so the compute stays
FP16 and only the weight *traffic* shrinks. Both need calibration data and a
quantization pass per model; neither is free at load time. INT4 needs
dequantization hardware (or a dequant step) to realise the speedup — on a
plain CPU donor, naive INT4 can be *slower* than FP16 because of the unpack
cost. This is a real risk for softwell's TypeScript/Float32Array engine and
must be measured before any claim is made.

### 4.3 INT8 activations: SmoothQuant

Xiao et al. [15] (ICML 2023) migrate quantization difficulty from activations
to weights with a mathematically equivalent per-channel scaling, enabling W8A8
(INT8 weights *and* activations) with negligible loss — up to 1.56× speedup
and 2× memory reduction, including serving a 530B model on a single node.

**Limits.** W8A8 needs INT8 matmul support to pay off; on donors without it,
the benefit is memory only. Activation outliers are model-dependent, so the
scaling must be calibrated per model.

**Build on.** Quantization is the first thing softwell should implement after
the FP32 reference path is correct, in this order: (1) FP16/BF16 weight
storage with FP32 compute (halves weight traffic, no accuracy risk beyond
rounding); (2) INT8 weights with FP32 dequant on load (halves traffic again);
(3) INT4 via GPTQ/AWQ only after measuring dequant cost on real donors.
Each step must be validated by perplexity, not assumed. The companion
number-system note owns the details.

---

## 5. Sparsity

### 5.1 Structured and dynamic sparsity

The cleanest result is SparseGPT [16] (Frantar and Alistarh, ICML 2023):
large GPT-family models can be pruned to 50–60% *unstructured* sparsity in
one shot, without retraining, with negligible perplexity increase — over 100B
weights of OPT-175B can simply be ignored at inference. It also supports
semi-structured 2:4 and 4:8 patterns, which are the ones hardware can
actually accelerate.

**Limits.** Unstructured 50% sparsity saves *storage* but saves *time* only
if the engine skips zeros efficiently; on dense hardware it often saves
nothing. Structured sparsity (whole rows/columns/heads) is immediately useful
but harder to achieve without accuracy loss. Dynamic sparsity — skipping
work per token based on the input — is the most promising direction for
softwell and the least settled: there is no published theorem giving a
per-token sparsity guarantee for arbitrary prompts.

**Build on.** Structured pruning of attention heads and FFN rows is directly
usable: a pruned weight is a weight never read. Treat unstructured sparsity as
a storage win only, until a sparse kernel exists.

### 5.2 Mixture-of-Experts

Shazeer et al. [17] (ICML 2017) introduced the sparsely-gated MoE layer:
thousands of feed-forward experts, a trainable gate routing each token to a
few. Result: >1000× more capacity at near-constant compute per token.
DeepSeek-V2 [18] is the modern form: 236B total parameters, 21B activated per
token, with fine-grained expert segmentation.

**Limits.** MoE changes the *model*, not the runtime: softwell cannot turn a
dense model into an MoE. What it can do is *serve* MoE models well, which is a
placement problem: experts are the natural unit of sharding across donors,
and the gate's routing is known one layer ahead, so expert weights can be
prefetched. The risk is load imbalance — a popular expert becomes a hot donor.

**Build on.** Expert-parallel placement is the strongest form of Theorem P1
in note 05: when the model is an MoE, the planner should shard by expert and
co-locate experts that the gate tends to select together. For dense models,
the analogue is layer-pipelined placement (§2 of note 05).

---

## 6. KV-cache compression

The KV cache is the second bytes-per-token term (after weights) and the one
that grows with context. For a decode step at context $S$, the attention
layer must read $2 \cdot S \cdot d_{kv}$ values per layer; at long context
this rivals the weight read.

### 6.1 Fewer KV heads: MQA and GQA

Shazeer [4] (MQA, 2019) shares one key/value head across all query heads,
cutting KV traffic by a factor of $h$ (the head count) at small quality cost —
12× faster incremental decoding in his experiments. Ainslie et al. [19]
(GQA, EMNLP 2023) interpolate: $G$ KV heads shared across groups of query
heads, with an uptraining recipe converting an MHA checkpoint to GQA with 5%
of original training compute, at near-MHA quality and near-MQA speed.

**Limits.** Both are architectural: they require the model to have been built
(or uptrained) that way. Softwell cannot apply them to an arbitrary MHA
checkpoint without retraining.

### 6.2 Low-rank KV: MLA

DeepSeek-AI [18] (DeepSeek-V2, 2024) compresses keys and values jointly into
a low-rank latent vector, cutting the KV cache by 93.3% against their
baseline and boosting maximum generation throughput 5.76× — while *improving*
quality over MHA. MLA is both a low-rank method (§7) and a KV-cache method;
it is the strongest single published result in this survey for bytes per
token.

**Limits.** Same as MQA/GQA: architectural, needs training. But it sets the
target: a 10–15× KV reduction is achievable in principle, so softwell's
KV-compression work should be judged against that bar, not against 2×.

### 6.3 Quantized KV: KIVI

Liu et al. [20] (ICML 2024) quantize the KV cache itself to 2 bits, with an
asymmetric scheme — keys per-channel, values per-token — motivated by a
careful study of outlier structure, keeping a full-precision sliding window
for recent tokens. Reported: 2.6× less peak memory, up to 4× larger batch,
2.35–3.47× throughput, with near-identical quality on Llama, Falcon, Mistral.

**Limits.** KV quantization interacts with attention numerics: keys feed a
softmax, so key error is amplified in a way weight error is not. The
full-precision residual window is load-bearing — the method is not "2-bit
everywhere".

**Build on.** KIVI is the most directly implementable KV technique for
softwell: it needs no retraining, it composes with the paged KV cache
(`src/scheduler/kvcache.ts`) by quantizing per page, and it attacks exactly
the term that dominates at long context. It should be the first KV
compression implemented, after FP16 KV storage.

### 6.4 Eviction: H2O

Zhang et al. [21] (NeurIPS 2023) observe that a small set of "heavy-hitter"
tokens carries most of the attention mass, formulate eviction as a dynamic
submodular problem, prove a guarantee under mild assumptions, and retain a
mix of heavy hitters and recent tokens. Reported: up to 29× throughput over
baseline serving systems on OPT models (a systems comparison, not a pure
algorithmic speedup — read the number accordingly).

**Limits.** Eviction is lossy and prompt-dependent; there is no guarantee for
adversarial prompts (e.g. needle-in-haystack retrieval, where the evicted
token is exactly the one needed). The submodular guarantee is about the
eviction objective, not about output quality.

**Build on.** Eviction is a *policy* the scheduler can offer, not a default:
useful for long-context summarisation, dangerous for retrieval. Softwell
should implement it as an opt-in per-request policy with the H2O
heavy-hitter-plus-recent rule.

---

## 7. Low-rank methods

### 7.1 Attention is low-rank: Linformer

Wang et al. [22] proved (via Johnson–Lindenstrauss) that the attention matrix
can be approximated by a low-rank matrix and built the Linformer, reducing
self-attention from $O(n^2)$ to $O(n)$ in time and space, with near-RoBERTa
quality. Reported: up to 20× faster and 60× more memory-efficient at
sequence length 65,536.

**Limits.** The projection fixes the sequence length and the method needs
training; the theorems guarantee less than the headline suggests (the
required rank grows with the error tolerance). It is a prefill/training
technique more than a decode technique: at decode, the attention matrix is
$1 \times S$, already linear.

**Build on.** The low-rank *observation* matters more than the Linformer
itself: it justifies MLA (§6.2) and justifies trying low-rank structure in
softwell's attention path. But for decode, the rank that matters is the KV
rank, not the attention-matrix rank.

### 7.2 Low-rank adaptation: LoRA

Hu et al. [23] (ICLR 2022) showed that fine-tuning updates have low intrinsic
rank: freeze the base weights, learn $\Delta W = BA$ with $r \ll \min(d,k)$,
cutting trainable parameters ~10,000× on GPT-3 175B with no inference-latency
cost after merging.

**Limits.** LoRA is a *training* method. It does not make inference cheaper:
the merged weight is the same size as the original. Its relevance to softwell
is indirect but real: if users fine-tune with LoRA, softwell can serve the
base model once and swap small adapters per request — the adapter is a tiny
amount of extra traffic per token, and the base weights are read once and
shared across all adapters. That is a genuine bytes-per-token win at the
*system* level, and it fits the donor-pool model well (base on the pool,
adapter on the requesting donor).

---

## 8. Serving and scheduling

### 8.1 Continuous batching: Orca

Yu et al. [5] (OSDI 2022) replaced request-level scheduling with
iteration-level scheduling: after every decode step, finished sequences leave
and new ones join. Reported: 36.9× throughput over FasterTransformer on
GPT-3 175B at equal latency. The technique is now universal (vLLM, TGI,
TensorRT-LLM, SGLang).

**Limits.** Continuous batching raises *throughput* by keeping the batch full;
it does not reduce per-token latency and does not reduce bytes per token. It
is the mechanism that lets the other techniques' savings turn into served
tokens.

**Build on.** Softwell already has a continuous-batching scheduler
(`src/scheduler/batch.ts`); the survey confirms this is the right primitive.
The open softwell question is batching *across donors*: the batch must be
large enough to amortise a weight read, but the donors holding the weights
must all be ready at the same step — a synchronisation problem the
single-machine literature does not have.

### 8.2 Paged KV cache: vLLM

Kwon et al. [6] (SOSP 2023) applied OS virtual-memory paging to the KV cache:
fixed-size blocks, a block table, near-zero fragmentation, and sharing of
blocks across requests (prefix caching). Reported: 2–4× throughput over
FasterTransformer and Orca at equal latency, with the gain growing with
sequence length and model size.

**Limits.** Paging saves *memory* (fragmentation), not *traffic*: the KV bytes
still have to be read every step. Its value is enabling larger batches, which
is where the throughput comes from.

**Build on.** Softwell already has a paged KV cache (`src/scheduler/kvcache.ts`).
The donor-pool extension is the research problem of note 05: pages are the
natural unit of placement, and the planner's job is to put the pages a
sequence is about to read on donors that are close to the donor computing
that sequence's attention.

---

## 9. Synthesis: what softwell can build on

| Technique | Bytes/token saved | Needs retraining? | Softwell status |
|---|---|---|---|
| FlashAttention tiling | $O(S^2)\to O(S)$ attention traffic | No | Implement in attention kernel |
| Flash-Decoding split-KV | Parallelises single-query decode | No | Planner primitive (note 05) |
| Speculative decoding | $\div$ expected accepted run length | Draft model needed | Companion decoding note |
| FP16 weights | 2× weight traffic | No | First quantization step |
| INT8 / INT4 weights | 4–8× weight traffic | Calibration only | After measuring dequant cost |
| KIVI (2-bit KV) | ~8× KV traffic | No | First KV compression |
| MLA / GQA / MQA | Up to ~15× KV traffic | Yes (architectural) | Serve such models; can't convert |
| H2O eviction | Bounded KV size | No | Opt-in scheduler policy |
| SparseGPT pruning | Up to 60% of weights unread | No (one-shot) | Storage win now; time win needs sparse kernel |
| MoE serving | Only active experts read | No (serve only) | Expert-parallel placement (note 05) |
| LoRA adapters | Base weights shared across requests | No (serve only) | Adapter-per-request serving |
| Continuous batching | Amortises weight reads over $B$ | No | Already implemented |
| Paged KV cache | Removes fragmentation; enables batch | No | Already implemented |

The honest summary: no single technique here is a breakthrough for softwell
— each is a constant-factor reduction in bytes or FLOPs per token, proven on
hardware softwell does not have. Their product is what matters. A 7B model at
INT4 weights (8×), KIVI KV (8×), speculative decoding with $\alpha\approx0.7$
(~2.5× tokens per weight read), and batching ($B$× amortisation) multiplies
to a 100×+ reduction in bytes per *served* token against the naive FP32
batch-1 baseline — before any new theorem. That product, not any one paper,
is the softwell thesis. The theorems in note 05 state the ceiling that
product is approaching.

---

## References

[1] S. Williams, A. Waterman, and D. Patterson. "Roofline: An Insightful
Visual Performance Model for Multicore Architectures." *Communications of
the ACM* 52(4), 2009. DOI: 10.1145/1498765.1498785.

[2] G. M. Amdahl. "Validity of the Single Processor Approach to Achieving
Large Scale Computing Capabilities." *AFIPS Conference Proceedings* 30,
1967, pp. 483–485.

[3] R. Landauer. "Irreversibility and Heat Generation in the Computing
Process." *IBM Journal of Research and Development* 5(3), July 1961,
pp. 183–191.

[4] N. Shazeer. "Fast Transformer Decoding: One Write-Head is All You Need."
arXiv:1911.02150, 2019.

[5] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun. "Orca: A
Distributed Serving System for Transformer-Based Generative Models."
*Proceedings of OSDI '22*, 2022. (36.9× throughput over FasterTransformer.)

[6] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez,
H. Zhang, and I. Stoica. "Efficient Memory Management for Large Language
Model Serving with PagedAttention." *Proceedings of SOSP '23*, 2023.
arXiv:2309.06180.

[7] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*.
arXiv:2205.14135.

[8] T. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and
Work Partitioning." *ICLR 2024*. arXiv:2307.08691.

[9] T. Dao, D. Haziza, F. Massa, and G. Sizov. "Flash-Decoding for long-context
inference." Blog post, Stanford CRFM / Princeton NLP, October 2023.
https://crfm.stanford.edu/2023/10/12/flashdecoding.html — *not peer-reviewed;
figures are the authors' benchmarks.*

[10] Y. Leviathan, M. Kalman, and Y. Matias. "Fast Inference from Transformers
via Speculative Decoding." *ICML 2023*. arXiv:2211.17192.

[11] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper.
"Accelerating Large Language Model Decoding with Speculative Sampling."
arXiv:2302.01318, 2023.

[12] P. Micikevicius et al. "FP8 Formats for Deep Learning." arXiv:2209.05433,
2022.

[13] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. "GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers."
*ICLR 2023*. arXiv:2210.17323.

[14] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao,
X. Dang, C. Gan, and S. Han. "AWQ: Activation-aware Weight Quantization for
On-Device LLM Compression and Acceleration." *MLSys 2024* (Best Paper).
arXiv:2306.00978.

[15] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han. "SmoothQuant:
Accurate and Efficient Post-Training Quantization for Large Language Models."
*ICML 2023*. arXiv:2211.10438.

[16] E. Frantar and D. Alistarh. "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot." *ICML 2023*. arXiv:2301.00774.

[17] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and
J. Dean. "Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer." *ICLR 2017*. arXiv:1701.06538.

[18] DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model." arXiv:2405.04434, 2024.

[19] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and
S. Sanghai. "GQA: Training Generalized Multi-Query Transformer Models from
Multi-Head Checkpoints." *EMNLP 2023*. arXiv:2305.13245.

[20] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and
X. Hu. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache."
*ICML 2024*. arXiv:2402.02750.

[21] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y. Tian, C. Ré, C. Barrett, Z. Wang, and B. Chen. "H₂O: Heavy-Hitter Oracle
for Efficient Generative Inference of Large Language Models." *NeurIPS 2023*.
arXiv:2306.14048.

[22] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma. "Linformer:
Self-Attention with Linear Complexity." arXiv:2006.04768, 2020.

[23] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
arXiv:2106.09685.

---

*Verification note.* All 23 references above were confirmed by web search on
2026-11-22 against arXiv or publisher records (title, authors, venue/year).
No reference was dropped as unverifiable; every citation attempted was
confirmed. Flash-Decoding [9] is the only non-peer-reviewed entry and is
labelled as such. Reported speedups are the cited papers' own figures.
