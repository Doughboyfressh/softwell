# softwell Virtual NPU (vNPU)

**Status:** core device implemented, 2026-11-22 (`src/hw/vnpu.ts`,
`VirtualNpu` + `IVirtualNpu` in `src/contracts.ts`); forward-pass
integration implemented 2026-11-22 (`src/engine/transformer.ts` —
with `--npu int8|fp8` the executor stages weights into SRAM at load
and issues its matmuls as MATMUL_INT8 / MATMUL_FP8; `--npu off`
leaves the FP32 path byte-identical), tested in
`test/vnpu-integration.test.ts`; measurements filled 2026-11-22
(§5.3, `src/bench/bench-vnpu.ts`, `npm run bench -- --only vnpu`).
The instruction set (§1), memory model (§2, D1 confirmed as built),
streams (§3.1), and hypervisor registration (§3.1) exist and are
tested (`test/vnpu-kernels.test.ts`, `test/vnpu-sram.test.ts`,
`test/vnpu-hypervisor.test.ts`). What is NOT built yet: the
donor-side `VnpuNode` (§4.2).

**What the vNPU is.** A fifth virtual device alongside vGPU, vCPU,
vSSD, and vRAM — the inference specialist. Where the vGPU is the
general dense-compute device, the vNPU is the device that runs
transformer inference natively: INT8/FP8 execution as first-class
operations, attention and sparsity as virtual-silicon primitives, and
the novel math of `docs/RESEARCH.md` as instructions rather than
library calls. It must be a real device with distinct semantics, not
a vGPU relabel: a workload that runs on the vNPU issues different
instructions, against a different memory model, under a different
scheduler, than the same workload on the vGPU.

**What it is not.** It is not a physical NPU backend. Like every
other device in this machine, the vNPU stays 100% virtual: its first
implementation executes on CPU donors through the same fallback-kernel
pattern as the vGPU (`src/hw/kernels-cpu.ts`), and it wins only by
pooling donors and by doing less work per token — never by claiming
silicon it does not have.

## 1. Instruction set

The vNPU executes a small, fixed instruction set. An instruction is
a record `{ op, dst, a, b, shape, precision, flags }` issued against
vRAM addresses; there are no dynamic shapes at issue time — every
tile dimension is fixed when the instruction is enqueued, so the
scheduler can cost it exactly (cf. Theorem P2 in
`docs/RESEARCH.md`: the cost of an instruction is its bytes moved,
and bytes moved is computable from the instruction alone).

In v1 the instruction set is the typed method API of `VirtualNpu`
(`src/hw/vnpu.ts`, interface `IVirtualNpu` in `src/contracts.ts`);
§1.2's 64-byte binary encoding is deferred to the donor-node
transport work. The mapping from op name to method is:

| Op | Method |
|---|---|
| `MATMUL_INT8` | `vnpu.matmulTransposeB` / `vnpu.matmul` with `format: "int8"` |
| `MATMUL_FP8` | `vnpu.matmulTransposeB` / `vnpu.matmul` with `format: "fp8"` |
| `MATMUL_SW` | `vnpu.matmulSw` |
| `MATMUL_BF16` | `vnpu.matmulTransposeB` / `vnpu.matmul` with `format: "bf16"` |
| `ATTN_LANDMARK` | `vnpu.attention` / `vnpu.prefillAttention` with `mode: "landmark"` |
| `ATTN_SPARSE` | `vnpu.attention` / `vnpu.prefillAttention` with `mode: "sparse"` |
| `ATTN_EXACT` | `vnpu.attention` / `vnpu.prefillAttention` with `mode: "exact"` |
| `KV_QUANTIZE` | `vnpu.quantizeKv` |
| `SOFTMAX` | `vnpu.softmax` |
| `RMSNORM` | `vnpu.rmsNorm` |
| `ACT` | `vnpu.silu` |
| `QUANTIZE` / `DEQUANTIZE` | `vnpu.quantizeSw` / `vnpu.dequantizeSw` |

Two additions beyond the §1 table, both landed with the core
device: `ATTN_PACKED_KV` (`vnpu.attentionPackedKv`) — exact
attention against a quantized KV cache, the v1 answer to open
question 3 in §6 — and per-sequence streams
(`vnpu.createStream` / `vnpu.synchronize`), which serialize
instructions on one stream in order, like the vGPU's streams.

### 1.1 Compute instructions

| Op | Semantics | Precision |
|---|---|---|
| `MATMUL_INT8` | Dense $C = A \times B$ over INT8 operands, INT32 accumulation, INT8/FP16 out | INT8 |
| `MATMUL_FP8` | Dense $C = A \times B$ over FP8-E4M3 operands, FP32 accumulation | FP8 |
| `MATMUL_SW` | Dense $C = A \times B$ where $B$ (and optionally $A$) is SW-code packed (Theorem N1) — the packed stream is consumed directly, never materialised as FP32 | sw2 / sw3 / sw4 |
| `MATMUL_BF16` | Dense $C = A \times B$ over BF16, FP32 accumulation — the compatibility path, not the fast path | BF16 |
| `ATTN_LANDMARK` | Theorem A1 as one instruction: block-mean landmark attention over a KV block range, with per-block $\delta_b$ gating and exact fallback per block | FP32 accumulate |
| `ATTN_SPARSE` | Theorem A2 as one instruction: block-screened sparse attention over a KV block range, with the mass bound computed on-device | FP32 accumulate |
| `ATTN_EXACT` | Exact grouped-query attention (FlashAttention-style tiling over SRAM, §2) — the fallback both approximate ops degrade to | FP32 accumulate |
| `KV_QUANTIZE` | Quantize a KV row to SW-code after RoPE (Theorem N2) | sw2 / sw3 / sw4 |
| `SOFTMAX` | Row softmax over an SRAM-resident tile | FP32 |
| `RMSNORM` | RMSNorm over an SRAM-resident tile, with weights | FP32 |
| `ACT` | Elementwise activation: SiLU, GELU, SwiGLU gate | FP32 |
| `QUANTIZE` / `DEQUANTIZE` | Convert between FP32/BF16/FP8/INT8/SW-code at a tile boundary | — |

Design rules:

- **Low precision is native, not emulated.** `MATMUL_INT8` and
  `MATMUL_FP8` are first-class ops with their own cost model; they
  are not `MATMUL_BF16` with a cast before and after. The cast, if
  any, is a separate `QUANTIZE` instruction and is charged as one.
- **Attention is silicon, not a kernel call.** `ATTN_LANDMARK` and
  `ATTN_SPARSE` are single instructions whose error bounds are the
  theorems' bounds (A1, A2) evaluated on the actual data — the
  device reports the bound alongside the result, the same way the
  reference implementations do today in
  `src/engine/novel-attention.ts`.
- **SW-code is the native number format.** `MATMUL_SW` and
  `KV_QUANTIZE` make the format of `docs/RESEARCH.md` §5 the
  device's storage format for weights and KV, not a compression
  layer on top of FP32. The known limit carries over unchanged:
  SW-code weights at sw2/sw3 destroy a real model (measured:
  sw4 +35% perplexity, sw3 14×, sw2 destroyed — see RESEARCH.md
  §5.4), so `MATMUL_SW` at sw2/sw3 is a valid instruction that
  produces a wrong model, and the device must not hide that.
  Error compensation (GPTQ-style) is future work, not part of
  this instruction set.
- **Speculative/Jacobi decoding stay in the scheduler.** Theorems
  S1/S2 are scheduling policies over forward passes, not device
  instructions. The vNPU exposes no `SPECULATE` op; the scheduler
  (§3) issues batched `MATMUL_*` / `ATTN_*` instructions and
  verifies drafts itself, exactly as `src/scheduler/batch.ts`
  does today.

### 1.2 Instruction encoding (design parameter)

Fixed 64-byte instruction header, then operand descriptors:

```
[op: u8][precision: u8][flags: u16][tile_m: u16][tile_n: u16][tile_k: u16]
[dst: vram_addr][a: vram_addr][b: vram_addr][scratch: sram_addr]
[scale_a: fp16][scale_b: fp16][delta_max: fp32][drop_threshold: fp32]
```

`delta_max` and `drop_threshold` are the A1/A2 gating thresholds
carried per instruction, so a single stream can mix exact and
approximate attention row by row. Operand addresses are vRAM
addresses (the global virtual address space); `scratch` is an SRAM
address inside the issuing tile (§2). The binary encoding is
deferred: v1 exposes instructions as typed method calls
(`src/hw/vnpu.ts`), which carry the same fields (fixed tile
dimensions via `VNPU_TILE = 128`, operand shapes, precision,
thresholds as per-call options) and satisfy the invariant — an
instruction is self-describing and costable without executing it
(every result reports its counted `ops` and, for attention, its
error bound). The 64-byte header below is the encoding the
donor-node transport work should implement; it is a design
parameter, not a wire format yet.

## 2. Memory model

The vNPU has a three-level, compiler-managed memory hierarchy.
There is **no hardware cache** anywhere in it: placement is explicit,
like TPU VMEM and the XDNA 2 cache-less design, and a tile that
needs data must name where it comes from. This is a deliberate
choice — a cache would hide the bytes-moved accounting that
Theorem P2 is built on.

### 2.1 Levels

| Level | What lives there | Capacity (per virtual device) | Bandwidth class |
|---|---|---|---|
| Tile SRAM | Activations, attention tiles, softmax rows, quantization scratch — everything an instruction touches while it runs | **256 MiB total: 64 tiles × 4 MiB tile-local SRAM** (design parameter D1, see below) | On-tile; the fast path, no network |
| Device vRAM | Weights, KV cache, activations between instructions — the vNPU's working set, allocated from pooled vRAM | Unbounded in the design (pooled vRAM capacity); a device's share is its allocation | Pooled vRAM bandwidth |
| vSSD | Model files, checkpoints, cold KV — anything not currently executing | Pooled vSSD capacity | Pooled vSSD bandwidth |

**Design parameter D1 — SRAM capacity: 256 MiB per virtual device
(64 tiles × 4 MiB).** This is a chosen design constant, not a
vendor figure and not a measurement. It is in the same class as
the on-chip memories of the target chips — TPU v5p carries a
128 MB vector memory (VMEM) per chip (secondary source: MLSysBook
appendix, not verified against a Google primary document in this
pass), and Trainium2 carries 224 MiB of SBUF per chip (28 MiB × 8
NeuronCore-v3, AWS Neuron documentation) — and it is deliberately
larger than either, because the vNPU's SRAM is *pooled*: the
aggregate SRAM of a vNPU with $N$ donors is $N \times 256$ MiB,
and the "better" bar (§5) is the pooled aggregate, same as every
other device. **D1 is confirmed as built**: the core device
implements exactly 256 MiB per virtual device
(`VNPU_SRAM_BYTES` in `src/hw/vnpu.ts`, pinned by
`test/vnpu-kernels.test.ts`). What D1 must never become is a number
quoted as if a vendor published it.

How v1 models the SRAM (see `src/hw/vnpu.ts` for the code):
weights stage from pooled vRAM into SRAM explicitly
(`stageWeights` / `stageFromVram`) as quantized resident copies;
a stage that does not fit evicts LRU residents (counted as
`sramEvictions`); a single weight larger than D1 spills (counted
as `sramSpills`, resident: false); every use of a non-resident
weight counts a miss (`sramMisses`) and streams the weight from
vRAM instead. In v1 the 256 MiB is accounted as one pool per
device with per-tensor LRU eviction — the tile-private
subdivision (which tile holds which bytes) is a scheduler concern
for later work, not a v1 accounting distinction. The per-tile
4 MiB limit IS enforced in v1: every instruction's working set
(one 128×128×128 matmul tile, or one KV block plus its block
statistics plus the query row for attention) is checked against
`VNPU_TILE_SRAM_BYTES`, and an instruction that cannot be tiled
to 4 MiB is rejected at enqueue time (tested in
`test/vnpu-kernels.test.ts`).

Per-tile rules:

- A tile's 4 MiB SRAM is private to that tile. Tiles do not read
  each other's SRAM; inter-tile data moves through vRAM (named,
  costed) — this is the vNPU analogue of Theorem P1's traffic
  accounting: every cross-tile byte is a byte that crosses a
  boundary, and the scheduler charges for it.
- An instruction's working set must fit its tile's SRAM. A matmul
  tile of $128 \times 128 \times 128$ at INT8 is 16 KiB per
  operand tile; an attention tile holds one KV block plus its
  block statistics ($\bar{k}_b$, $\bar{v}_b$, $R_b$, $S_b$ — see
  RESEARCH.md §4) plus the query row. The compiler (scheduler)
  tiles every instruction to fit; an instruction that cannot be
  tiled to 4 MiB is rejected at enqueue time, not at runtime.
- SRAM is not persistent across instructions except as explicitly
  named scratch: a tile may keep block statistics resident across
  the rows of one prefill (the A1/A2 stats pass is built once per
  KV cache and reused per query — RESEARCH.md §4), but that
  residency is part of the instruction stream's declared scratch,
  and a donor loss drops it with the tile (rebuilt from vRAM, not
  recovered).

### 2.2 Addressing and coherence

- All vRAM addresses are global virtual addresses in the pooled
  vRAM space (`src/hw/vram.ts`); the vNPU allocates weights and KV
  through the same allocator as the vGPU, so a weight staged once
  serves both devices.
- Coherence is single-writer + version invalidation, same as the
  rest of the machine. KV append is the one multi-writer case:
  `KV_QUANTIZE` writes are appends at a named position and
  overwrite that position's slot (the same overwrite semantics
  speculative decoding depends on — RESEARCH.md §7, assumption
  A2). A tile never writes a KV slot another tile is reading in
  the same step; the scheduler serialises append vs read per
  sequence.
- Weights are read-only during a generation. A weight write (a
  model load or re-stage) invalidates every tile's SRAM copy by
  version bump; tiles re-fetch from vRAM on next use.

## 3. Scheduling and hypervisor

### 3.1 Devices, streams, and queues

- The hypervisor (`src/hw/hypervisor.ts`) owns vNPU devices the
  way it owns vGPU devices: a vNPU device is created at boot,
  registered with its tile count (64) and SRAM capacity (D1), and
  assigned to donors by the same pool coordinator.
- Each device exposes **streams** — ordered instruction queues,
  one per sequence (decode) plus shared prefill streams. A stream
  is the unit of ordering: instructions in one stream execute in
  order; instructions across streams execute concurrently on
  different tiles, subject to SRAM residency.
- The scheduler is the existing continuous-batching scheduler
  (`src/scheduler/batch.ts`) extended with a vNPU queue: prefill
  for a new sequence becomes a set of `MATMUL_*` + `ATTN_*`
  instructions on a prefill stream; one decode step per running
  sequence becomes one instruction group on that sequence's
  stream. Batching across sequences is instruction-level: the
  scheduler packs same-shape matmuls from different sequences
  into one batched instruction where the shapes agree, and falls
  back to per-sequence instructions where they do not.
- Admission control is the same as the API server's: a sequence
  is admitted only if its KV allocation fits in pooled vRAM and a
  stream is free. Backpressure propagates as it does today.

### 3.2 Placement: which tile, which donor

Placement follows the placement physics of RESEARCH.md §3, with
one vNPU-specific addition:

- **Tensor-parallel splits pay P1 traffic.** Splitting one
  matmul across tiles on different donors costs
  $\min(nk+m,\ k+nm)\cdot b/8$ bytes (Theorem P1) — the scheduler
  computes the optimal donor count $n^*$ per layer (Corollary
  P3) and refuses to shard past it, exactly as the planner does
  for the vGPU.
- **Pipeline placement is the default across slow links.**
  Whole layers on single donors (Proposition P1b) pay only
  $2qd\,b/8$ bytes per token across $q$ layer boundaries — the
  vNPU's default for donors on a commodity network.
- **Attention tiles stay with their KV.** An `ATTN_*`
  instruction executes on a tile whose donor holds the KV blocks
  it reads, in vRAM or in SRAM if the blocks were prefetched.
  Moving KV to the compute is charged; moving compute to the KV
  is free by construction. This is the rule that makes A1/A2
  cheap: the landmark statistics live with the blocks, and the
  per-query work is one dot per block against resident stats.
- **SW-code weights stay packed until the tile.** A `MATMUL_SW`
  instruction's weight operand is fetched as the packed byte
  stream (3.6–6.6× fewer bytes than BF16 — RESEARCH.md §5.4) and
  unpacked into the tile's SRAM lookup table on arrival. The
  traffic saving is realised at the vRAM → tile transfer, which
  is the only place it can be realised (in-process unpacking on
  a single donor measures 5–7× *slower* — BENCHMARKS.md wave 2).

### 3.3 Fault tolerance

- Instructions are idempotent and re-issuable: a tile that dies
  mid-instruction loses its SRAM scratch, and the stream re-issues
  the instruction on a surviving tile. This is the vCPU's
  at-least-once delivery applied to NPU instructions; handlers
  (KV appends) must be idempotent — an append re-issued after a
  donor loss overwrites the same slot with the same bytes.
- A donor loss triggers the same sweep as the other devices:
  suspect at 6 s, dead at 15 s, re-plan over survivors. Sequences
  whose KV lives in pooled vRAM survive a donor loss (replicas +
  healing, as proven for the KV-in-vRAM work); sequences whose
  in-flight tile SRAM held unflushed state re-issue from the last
  flushed step. The chunk-table-dies-with-the-worker caveat from
  the KV work applies unchanged.
- Determinism: re-issuing an instruction produces bit-identical
  output (same inputs, same thresholds, same fallback decisions),
  so a retry is invisible to the sequence — the same exactness
  argument as Theorems S1a/S2a, applied to one instruction
  instead of one decode round.

## 4. Execution model

### 4.1 How an instruction runs

1. **Enqueue.** The scheduler emits an instruction with fixed
   tile dimensions, operand vRAM addresses, precision, and
   thresholds. The hypervisor assigns it to a stream and a tile.
2. **Fetch.** The tile DMA-fetches its operands from vRAM into
   SRAM (weights as packed SW-code where applicable, KV blocks
   plus their statistics for attention). Fetch is the charged
   traffic; the bytes fetched are the instruction's cost.
3. **Execute.** The tile runs the op against SRAM: systolic-style
   matmul over INT8/FP8/SW tiles, or attention over resident KV
   blocks with A1/A2 gating per block (approximate where the
   thresholds allow, exact fallback where they do not — the
   fallback decision is per block, per query, from the actual
   data, never from a static flag).
4. **Write back.** Results go to SRAM scratch, then to their
   destination vRAM address. KV appends write through to vRAM
   immediately (the survival contract depends on it); transient
   activations may stay in SRAM until the next instruction needs
   the space.
5. **Report.** The tile reports completion plus, for attention
   instructions, the achieved error bound and the skip/fallback
   rate — the numbers the benchmarks need and the numbers a
   deployment would monitor (RESEARCH.md §4.3: A2's win is only
   as reliable as the achieved skip rate).

### 4.2 First implementation (what runs on day one)

The first vNPU executes on CPU donors, exactly like the vGPU's
CPU fallback: in v1 the coordinator-side `VirtualNpu` runs each
instruction in-process against its modeled tile SRAM (a real
4 MiB working-set budget per tile, enforced by the tile-fit
check — not a fiction) using the existing kernel math
(`src/hw/kernels-cpu.ts`, `src/engine/novel-attention.ts`,
`src/engine/novel-numbers.ts`). Results report the device that
actually executed (`cpu`), never an NPU it does not have — the
same honesty rule as `GpuNode`. A per-donor `VnpuNode` executing
on real NPU silicon is future work, not v1; the instruction set
and SRAM accounting are written so the node can plug in under
the same interface.

What the first implementation must prove, in order:

1. An instruction stream runs end to end and produces the same
   tokens as the vGPU path for the same model and prompt
   (token identity, as in every phase so far). **Proven in part,
   2026-11-22** (`test/vnpu-integration.test.ts`, prompt "The
   capital of France is", 16 tokens, temp 0): FP8 holds exactly on
   both models (demo-100m and SmolLM-135M, 16/16 tokens identical
   to the FP32 baseline); INT8 holds on demo-100m but DIVERGES on
   real SmolLM-135M — first differing position 2, 13 of 16 tokens
   differ. The INT8 output stays coherent English (" Paris. It is
   the largest city in the world. It is the capital of") but it is
   not the baseline's tokens. Quantized execution changes tokens
   at INT8; that is the measured result, not a wiring failure
   (3,856 INT8 matmuls issued, 3,856 SRAM hits, 0 misses). The
   demo-100m leg is weak evidence on its own: the random demo
   weights collapse generation to one repeated token, so identity
   there holds trivially.
2. The SRAM accounting is real: a tile's residency is measured,
   an instruction that exceeds 4 MiB is rejected, and the
   reported SRAM usage matches the allocation. **Done** —
   `test/vnpu-sram.test.ts` pins fit/spill/LRU-eviction/node-loss
   against the counters, and `test/vnpu-kernels.test.ts` pins the
   tile-fit rejection. Every quantized matmul additionally
   returns a computed error bound against the FP32 reference
   (measured: INT8 observed max err 1.54e-1 vs bound 6.19e-1;
   FP8 1.32e-1 vs 1.21e+0; BF16 5.57e-3 vs 7.15e-2 — see the
   kernel test output).
3. The benchmark rows in §5 fill in with measured numbers —
   including, per the established convention, FAILs recorded
   plainly. **Done 2026-11-22:** §5.3 is filled (11 rows: 10 FAIL,
   1 PASS on the SRAM capacity row — a design constant, not a
   timed rate) plus the perplexity / tok/s / kernel tables. The
   attention-throughput row is deliberately unmeasured — no
   enterprise NPU publishes a comparable figure, and the attention
   instructions were not wired into the forward pass in this
   workstream (only the matmuls were).

### 4.3 What the vNPU deliberately does not do

- No dynamic control flow inside an instruction stream: no
  data-dependent branching other than the A1/A2 per-block
  fallback, which is a bound check, not a branch on model
  output. Sampling stays on the vCPU (`sample` task), as today.
- No cross-sequence attention: attention is per sequence, per
  head. Batching is across sequences at the instruction level,
  never inside one attention op.
- No MoE expert routing in silicon (v1): MoE dispatch stays in
  the scheduler/engine, as with SW-code today (MoE + SW-code is
  rejected at load). An `EXPERT_ROUTE` instruction is a named
  future extension, not part of this set.
- No claim about single-op latency. Pooling never makes one
  matmul or one attention step faster than the fastest single
  device that could run it (BENCHMARKS.md, "Honest limits").

## 5. Benchmarks

### 5.1 What "better" means for the vNPU

Same definition as every other device (BENCHMARKS.md): **pooled
aggregate throughput or capacity**. One donor is one machine; $N$
donors contribute $N$ machines' worth of INT8/FP8 operations,
bytes, and bytes/second to the virtual device, and the virtual
device's measured aggregate is compared against a single
enterprise NPU. A PASS means the pool, at the measured donor
count, already out-performs that one chip on that metric.

What is NOT claimed: single-op latency. One matmul, one attention
step, or one SRAM fetch on one donor is bounded by that donor's
hardware plus the network hop to reach it. The claim is that the
pool is a bigger machine, not a faster single operation.

Verdicts are computed, never asserted: PASS when
`measured >= target`, else FAIL — same rule as
`src/bench/types.ts`. Every row also reports `perDonor`
(`measured / donors`) and `donorsToBeat`.

### 5.2 Targets

The numbers the vNPU must beat. Verified 2026-11-22 from the
sources listed; see the notes below the table for unit and
provenance caveats.

**Primary target: Google TPU v6e (Trillium)** — the current
generation: highest BF16/INT8 compute per chip of the enterprise
NPUs, and the inference-shaped part (v6e is the efficiency-class
chip Google sells for serving; there is no separate v6p).

**Secondary target: Google TPU v5p** — the performance-class
chip, still the memory/bandwidth leader of the TPU line (95 GiB,
2,765 GB/s) and the largest pod (8,960 chips).

**Secondary targets: AWS Trainium2** (training-class, the
highest FP8 compute of the AWS line) and **AWS Inferentia2**
(inference-class, the direct analogue of what the vNPU is).

| Device | Target | Metric | Target value | Source |
|---|---|---|---|---|
| vNPU | Google TPU v6e (Trillium) — primary | INT8 peak | 1,836 TOPS | Google Cloud docs, TPU v6e |
| vNPU | Google TPU v6e (Trillium) — primary | BF16 peak | 918 TFLOPS | Google Cloud docs, TPU v6e |
| vNPU | Google TPU v6e (Trillium) — primary | memory bandwidth | 1,638 GB/s | Google Cloud docs, TPU v6e |
| vNPU | Google TPU v6e (Trillium) — primary | memory capacity | 32 GB | Google Cloud docs, TPU v6e |
| vNPU | Google TPU v6e (Trillium) — primary | on-chip SRAM (VMEM) | — (not published by Google) | — |
| vNPU | Google TPU v5p — secondary | INT8 peak | 918 TOPS | Google Cloud launch materials, Dec 2023 |
| vNPU | Google TPU v5p — secondary | BF16 peak | 459 TFLOPS | Google Cloud docs, TPU v5p |
| vNPU | Google TPU v5p — secondary | FP8 peak | 459 TFLOPS | Google Cloud docs, TPU v5p |
| vNPU | Google TPU v5p — secondary | memory bandwidth | 2,765 GB/s | Google Cloud docs, TPU v5p |
| vNPU | Google TPU v5p — secondary | memory capacity | 95 GiB | Google Cloud docs, TPU v5p |
| vNPU | AWS Trainium2 — secondary | FP8 peak | 1,299 TFLOPS | AWS Neuron docs, Trainium2 architecture |
| vNPU | AWS Trainium2 — secondary | BF16 / FP16 peak | 667 TFLOPS | AWS Neuron docs, Trainium2 architecture |
| vNPU | AWS Trainium2 — secondary | memory bandwidth | 2,900 GB/s (2.9 TB/s) | AWS Neuron docs, Trainium2 architecture |
| vNPU | AWS Trainium2 — secondary | memory capacity | 96 GiB | AWS Neuron docs, Trainium2 architecture |
| vNPU | AWS Trainium2 — secondary | on-chip SRAM (SBUF) | 224 MiB (28 MiB × 8 NeuronCore-v3) | AWS Neuron docs, Trainium2 architecture |
| vNPU | AWS Inferentia2 — secondary | INT8 peak | 380 TOPS | AWS Neuron docs, Inferentia2 architecture |
| vNPU | AWS Inferentia2 — secondary | FP16 / BF16 / cFP8 peak | 190 TFLOPS | AWS Neuron docs, Inferentia2 architecture |
| vNPU | AWS Inferentia2 — secondary | memory bandwidth | 820 GiB/s (as published) | AWS Neuron docs, Inferentia2 architecture |
| vNPU | AWS Inferentia2 — secondary | memory capacity | 32 GiB | AWS Neuron docs, Inferentia2 architecture |

Sources:

- https://docs.cloud.google.com/tpu/docs/v6e (Google Cloud
  documentation, TPU v6e / Trillium: 918 BF16 TFLOPS, 1,836 INT8
  TOPS, 32 GB HBM, 1,638 GB/s, 800 GB/s ICI)
- https://docs.cloud.google.com/tpu/docs/v5p (Google Cloud
  documentation, TPU v5p: 459 BF16 TFLOPS, 459 FP8 TFLOPS,
  95 GiB HBM, 2,765 GB/s)
- https://www.theregister.com/on-prem/2023/12/06/google-unveils-tpu-v5p-pods-to-accelerate-ai-training/
  (The Register, Dec 2023, reporting Google's v5p launch: 918
  INT8 TOPS — the INT8 figure is from the launch materials, not
  from the current v5p docs page, which lists FP8 459 instead;
  see notes)
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html
  (AWS Neuron documentation, Trainium2 architecture: 1,299 FP8
  TFLOPS, 667 BF16/FP16/TF32 TFLOPS, 2,563 sparse TFLOPS,
  96 GiB, 2.9 TB/s, 224 MiB SBUF, 1.28 TB/s NeuronLink-v3)
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inferentia2.html
  (AWS Neuron documentation, Inferentia2 architecture: 380 INT8
  TOPS, 190 FP16/BF16/cFP8/TF32 TFLOPS, 47.5 FP32 TFLOPS,
  32 GiB HBM, 820 GiB/s)

Target notes and caveats:

- **Units are quoted exactly as the vendor publishes them.**
  Google quotes v6e HBM bandwidth in GB/s (1,638) and v5p in
  GB/s (2,765); AWS quotes Trainium2 in TB/s (2.9) and
  Inferentia2 in GiB/s (820). GiB/s and GB/s differ by ~7%;
  the table does not normalise, and any bench that compares
  against these targets must state which unit its measured
  number is in.
- **TPU v5p INT8 vs FP8.** Google's current v5p docs page lists
  FP8 peak as 459 TFLOPS — the same as BF16 — and no INT8 row.
  The 918 INT8 TOPS figure is from Google's December 2023
  launch announcement, reported by The Register. Both are
  recorded here rather than silently picking one: the target
  table uses 918 INT8 TOPS (the launch figure, matching the
  2× BF16 pattern of v6e), and the FP8 459 row is kept
  alongside it. If Google's docs are taken as authoritative
  over the launch materials, the v5p INT8 target is
  unverified — flag it, do not silently drop it.
- **Conflicting secondary figures for Trainium2 exist and were
  not used.** Some aggregators list Trainium2 as 756 BF16 /
  1,512 FP8 TFLOPS with 3,200 GB/s (e.g. InferenceBench) or
  1,334 FP8 (BenchGecko). Those conflict with the AWS Neuron
  documentation (667 / 1,299 / 2.9 TB/s). The vendor doc is
  authoritative; the aggregator figures are not used anywhere
  in this table.
- **TPU v5p VMEM (on-chip SRAM) is not a target.** Google does
  not publish a VMEM capacity on the v5p docs page. A secondary
  source (MLSysBook appendix) lists 128 MB; that was not
  verified against a Google primary document in this pass, so
  it is not used as a target — it is mentioned only as context
  for design parameter D1 (§2.1).
- **TPU v7 (Ironwood) is a newer generation and is not a
  target here.** It was announced at Google Cloud Next 2025
  (April 2025) and reached general availability in November
  2025. Figures seen in secondary sources (2,307 BF16 TFLOPS,
  192 GiB HBM per chip) were not verified against Google Cloud
  documentation in this pass, and the task scoped the
  enterprise targets to v5p/v6e and Trainium2/Inferentia2.
  Adopting Ironwood as the primary target is a one-search
  follow-up, not a redesign — the table structure already
  supports it.
- **Consumer reference points (context only, not targets).**
  Apple M4 Neural Engine: 38 TOPS (Apple press release, May
  2024, https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/).
  Qualcomm Hexagon NPU (Snapdragon X Elite): 45 TOPS
  (Qualcomm product brief,
  https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/SnapdragonXEliteProductBrief.pdf).
  AMD XDNA 2 NPU (Ryzen AI 300): 50 TOPS (AMD press release,
  Computex 2024,
  https://ir.amd.com/news-events/press-releases/detail/1202/amd-unveils-next-gen-zen-5-ryzen-processors-to-power-advanced-ai-experiences).
  These are the consumer NPUs the vNPU concept is modelled
  against; the bar it must beat is the enterprise table above,
  not these. (All three consumer figures are INT8-class TOPS as
  published by the vendors; precision basis differs by vendor
  and is not normalised here.)

### 5.3 Measured results — filled 2026-11-22

Run: `npm run bench -- --only vnpu --json vnpu-bench-results-2026-11-22.json`
(11 rows, exit 0). Machine: AMD EPYC 9D25 126-Core Processor
(4 cores visible), 7.7 GiB RAM, Node v24.22.1, linux x64; loadavg
~1.0–1.7 during the run (quiet host). Raw JSON:
`vnpu-bench-results-2026-11-22.json`. The perplexity / tok/s /
kernel numbers below it come from `scripts/bench-vnpu.ts`
(`vnpu-perplexity-bench.json`, `vnpu-tok-bench.json`,
`vnpu-kernel-bench.json`), same machine, same day (timestamps in
the JSONs are UTC 2026-11-23; the run was the evening of
2026-11-22 PST). All single-run numbers with the load noted;
per the BENCHMARKS.md convention, a later run appends its own
table with its own machine line and date rather than overwriting
this one.

| device | metric | measured | target | margin | donors | donors-to-beat | verdict |
|---|---|---|---|---|---|---|---|
| vnpu | INT8 TOPS (MATMUL_INT8, 512^3) | 0.000898 TOPS | 1,836 (v6e) | 0.00005% | 1 | 2,043,548 | FAIL |
| vnpu | INT8 TOPS (MATMUL_INT8, 512^3) | 0.000898 TOPS | 918 (v5p) | 0.00010% | 1 | 1,021,774 | FAIL |
| vnpu | INT8 TOPS (MATMUL_INT8, 512^3) | 0.000898 TOPS | 380 (Inferentia2) | 0.00024% | 1 | 422,957 | FAIL |
| vnpu | FP8 TFLOPS (MATMUL_FP8, 512^3) | 0.001194 TFLOPS | 1,299 (Trainium2) | 0.00009% | 1 | 1,087,485 | FAIL |
| vnpu | FP8 TFLOPS (MATMUL_FP8, 512^3) | 0.001194 TFLOPS | 459 (v5p) | 0.00026% | 1 | 384,262 | FAIL |
| vnpu | BF16 TFLOPS (MATMUL_BF16, 512^3) | 0.001469 TFLOPS | 918 (v6e) | 0.00016% | 1 | 624,887 | FAIL |
| vnpu | BF16 TFLOPS (MATMUL_BF16, 512^3) | 0.001469 TFLOPS | 667 (Trainium2) | 0.00022% | 1 | 454,031 | FAIL |
| vnpu | memory bandwidth (vRAM → tile) | 1.332 GB/s | 2,765 (v5p) | 0.05% | 1 | 2,077 | FAIL |
| vnpu | memory bandwidth (vRAM → tile) | 1.332 GB/s | 1,638 (v6e) | 0.08% | 1 | 1,230 | FAIL |
| vnpu | pooled SRAM capacity | 256 MiB | 224 MiB (Trainium2 SBUF) | 114.3% | 1 | 1 | PASS |
| vnpu | aggregate INT8 TOPS (MATMUL_INT8, 512^3 x 1 donor) | 0.000897 TOPS | 1,836 (v6e) | 0.00005% | 1 | 2,047,064 | FAIL |

As predicted in advance, the throughput rows FAIL by 5–6 orders
of magnitude on CPU donors — exactly as the vGPU's did
(BENCHMARKS.md: vGPU matmul 0.000459 TFLOPS vs a 4,000 TFLOPS
target). The one PASS is the SRAM capacity row, and it is a
design constant (D1), not a timed rate: 256 MiB per virtual
device already exceeds Trainium2's 224 MiB SBUF, so one donor
passes and the pooled aggregate is N × 256 MiB. A FAIL is a
measurement, not a crash; the bench exits 0.

#### Perplexity — the question SW-code failed (SmolLM-135M, real text)

Same protocol as the SW-code bench (`scripts/bench-sw-smollm.ts`):
TinyStories slice at byte offset 1,000,000,000, 4 chunks ×
seqLen 64 = 252 scored tokens, teacher-forced, BOS prepended
per chunk. FP32 baseline reproduces the SW bench exactly
(PPL 47.539, same as `sw-smollm-bench.json`).

| mode | perplexity | mean NLL | vs FP32 | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| FP32 (baseline) | 47.539 | 3.86154 | — | 5.678 | 5.844 |
| INT8 (vNPU) | 61.610 | 4.12082 | +29.6% | 0.813 (0.14×) | 0.546 (0.09×) |
| FP8 (vNPU) | 50.092 | 3.91386 | +5.4% | 0.194 (0.034×) | 0.129 (0.022×) |

**FP8 costs little perplexity (+5.4%) — the key question is
answered yes for FP8.** INT8 costs +29.6%, about the same as
SW-code sw4 (+35%). But neither saves wall-clock time on this
CPU fallback: INT8 runs at 0.09–0.14× the FP32 rate and FP8 at
0.02–0.03×. The byte-savers save bytes, not time — the same
verdict as SW-code (RESEARCH.md §5.4). Model load also slows:
2.5 s (FP32) vs 6.6 s (INT8 staging) vs 24.5 s (FP8 staging —
the FP8 codec over 135M weights is slow). Both quantized runs
issued 36,703 vNPU matmuls with 36,703 SRAM hits, 0 misses,
0 spills, 0 evictions; staged weights occupy ~129 MiB of the
256 MiB SRAM.

#### tok/s through the vNPU path (short prompt, 16 tokens, temp 0, 1 rep)

Prompt "The capital of France is", greedy, SmolLM-135M.
Tokens match the integration test exactly (same prompt, same
decoding): FP8 identical to baseline, INT8 diverged (see §4.2
item 1).

| mode | median generate tok/s | vs FP32 | prefill ms | decode-only tok/s |
|---|---|---|---|---|
| FP32 (baseline) | 4.29 | — | 961 | 5.87 |
| INT8 (vNPU) | 0.45 | 0.10× | 7,559 | 0.56 |
| FP8 (vNPU) | 0.11 | 0.026× | 28,464 | 0.13 |

#### Kernel throughput (staged weights, 2 timed reps, median)

INT8/FP8 `matmulTransposeB` vs the FP32 `matmulTransposeB`
kernel, forward-opt shapes plus square TOPS shapes. Every
quantized result is within its computed error bound
(maxErr ≤ bound in all 14 cases).

| shape | FP32 ms | INT8 ms | FP8 ms |
|---|---|---|---|
| q/o [1,576]x[576,576] | 2.93 | 10.18 | 16.88 |
| gate/up [1,576]x[1536,576] | 1.61 | 12.99 | 41.49 |
| down [1,1536]x[576,1536] | 1.09 | 23.18 | 42.44 |
| lm_head [1,576]x[49152,576] | 35.3 | 506.94 | 1,353.86 |
| prefill gate [5,576]x[1536,576] | 5.52 | 19.64 | 44.84 |
| square 256^3 | 20.75 | 38.62 (0.000869 TOPS) | 34.06 (0.000985 TFLOPS) |
| square 512^3 | 168.47 | 287.86 (0.000933 TOPS) | 235.09 (0.001142 TFLOPS) |

**FP32 beats INT8/FP8 on every shape on this fallback**
(1.4× at 512^3, 14× at lm_head m=1, 27× at down m=1).
INT8 wins over FP8 at decode shapes (m=1: the naive INT8 core
is cheaper than FP8's dequantize + FP32 kernel); FP8 wins
over INT8 at large squares (512^3: the FP32 kernel's MAC speed
finally outweighs the dequantize cost). Neither wins over FP32
anywhere here.

#### Why the quantized path is slow — the measured bottleneck

The per-call cost is not the MACs. Every `matmulTransposeB`
with a staged weight re-derives the dequantized weight and the
full error-bound analysis from the staged payload
(`stagedToInt8` / `stagedToFp8` in `src/hw/vnpu.ts`) — an
O(weight-size) pass per call, with fresh multi-hundred-megabyte
allocations for the largest weights. The transformer's
per-row decode path issues ~210 such calls per token
(7 matmuls × 30 layers), each paying O(weight) analysis for
an O(1-row) matmul. Measured per m=1 call: ~9 ms (INT8),
~37 ms (FP8 — FP8 decode costs ~4× INT8's per element), which
accounts for the observed decode rates (INT8 0.56 tok/s,
FP8 0.13 tok/s) and for FP8 being ~4× slower than INT8 at
prefill/decode despite running the faster FP32 kernel for the
MACs. The fix is a follow-up for the core-device workstream,
not this one: cache the dequantized values + `QuantAnalysis`
per staged weight key (they are pure functions of the staged
payload, which is immutable between stages), so a call pays
only the MACs plus activation quantization. Until then, the
vNPU's INT8/FP8 win is bytes moved, not wall-clock time —
same as SW-code.

#### Attention throughput row — not measured

| device | metric | measured | target | margin | donors | donors-to-beat | verdict |
|---|---|---|---|---|---|---|---|
| vnpu | attention throughput (ATTN_LANDMARK / ATTN_SPARSE, tokens/s at n=4096) | — | — (no enterprise NPU publishes a comparable attention figure; informational row) | — | — | — | — |

No enterprise NPU publishes a comparable attention figure, and
the attention instructions were not wired into the forward pass
in this workstream (only the matmuls were). Measuring
ATTN_LANDMARK / ATTN_SPARSE throughput at n=4096 against the
vGPU's attention path is a named follow-up, not a gap hidden
by omission.

## 6. Open questions for the core device workstream

1. **D1 (SRAM capacity).** 256 MiB per device (64 × 4 MiB
   tiles) is a chosen constant. **Resolved as built** — the core
   device implements D1 exactly (§2.1); revising it now means
   changing `VNPU_SRAM_BYTES` and this section together.
2. **Tile geometry.** The instruction set assumes a
   $128 \times 128$ matmul tile (the TPU MXU dimension and the
   Trainium2 Tensor Engine dimension). **Built as assumed:**
   `VNPU_TILE = 128` in `src/hw/vnpu.ts`, and the tile-fit check
   uses it. If a later workstream picks a different tile, the
   SRAM fit analysis in §2.1 must be redone.
3. **Attention + quantized KV.** Today the engine rejects
   landmark/sparse attention combined with quantized KV
   (RESEARCH.md §9). The vNPU needs `ATTN_LANDMARK` /
   `ATTN_SPARSE` over packed KV, or it inherits the rejection.
   **Partially resolved:** the vNPU provides exact attention
   over packed KV as `ATTN_PACKED_KV`
   (`vnpu.attentionPackedKv`, Theorem N2 bound reported with
   the result), so the exact path does not inherit the
   rejection. Landmark/sparse over packed KV is still open —
   it needs block statistics over quantized rows, which the
   current `buildBlockStats` does not produce.
4. **API surface.** The instruction format (§1.2), the device
   registration, and the stream/queue interface are drafts.
   The final wording depends on the core workstream's API;
   nothing here should be presented as final until that lands.
   **Resolved:** the API landed as the typed `IVirtualNpu`
   interface (`src/contracts.ts`) and `VirtualNpu`
   (`src/hw/vnpu.ts`); the method table in §1 is the final
   v1 wording. The 64-byte binary encoding (§1.2) remains a
   draft for the donor-node transport work.
5. **Ironwood as primary target.** See §5.2 notes: verifying
   TPU v7 figures against Google Cloud docs and promoting it to
   primary is a small follow-up.
