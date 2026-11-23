# Forward-pass optimization — core result (2026-11-22)

Core cut of the forward-pass piece: profile + the single biggest
optimization, one short-prompt before/after. The full program (fused
SwiGLU, cached RoPE tables, in-place-softmax GQA, long-prompt bench,
demo-100m) is deferred to follow-up. Nothing here changes any default,
touches the scheduler/server/model code, or adds a physical backend.

## Round 2 — top 3 hot spots beyond matmulTransposeB (2026-11-23)

The core (u4 matmul) was already in place, so this round profiled again
and optimized the next-biggest pieces. All three changes are in one file,
`src/engine/attention.ts`; no call-site, default, scheduler, server, or
model-code changes. Nothing pushed or merged.

### What changed

1. **Fused tiled SwiGLU** (`swiglu`, tile 256). For each tile of
   intermediate units, the gate and up dots share one pass over x, the
   silu*up product goes into a 256-float tile buffer, and the down
   projection for the tile accumulates straight into the output. The
   full gate/up/hidden arrays are never materialized. Also benefits
   `moeForward`, which calls `swiglu` per expert.
2. **GQA: u4 Q.K dot + in-place softmax** (`groupedQueryAttention`).
   One scores array reused across heads and softmaxed in place (no
   second weights array); the per-head dot uses the same 4-accumulator
   unroll as `matmulTransposeB`. Masking semantics unchanged, including
   the fully-masked query returning zeros.
3. **Cached RoPE tables** (`applyRope` via `ropeRow`). cos/sin per
   (theta, headDim, position) built lazily on first use, tables grown by
   doubling; positions past 2^20 are computed on the fly so the cache
   cannot grow without bound. Values identical to per-element
   computation (max abs diff 6e-8, Float32 rounding).

### Measured kernel gains (interleaved medians, load-cancelled)

| Kernel | Before | After | Ratio |
|---|---|---|---|
| swiglu [576→1536] | 3.35–3.40 ms/call | 2.70–2.85 ms/call | **1.19–1.25x** |
| GQA kvLen 21 | 52.1 µs/call | 40.9 µs/call | **1.13–1.16x** |
| GQA kvLen 512 | 1127 µs/call | 973 µs/call | **1.19x** |
| GQA kvLen 997 | 2246 µs/call | 1939 µs/call | **1.10x** |
| RoPE 1 row/layer | 26.2 µs | 3.9 µs | **6.7x** |

Re-measured rejections in the same runs: u8 unroll 0.92–1.05x vs u4,
pre-transposed-weights T/T_u8 0.64–0.80x — both still rejected.
The tiled SwiGLU re-associates the down accumulation across tiles:
max abs diff vs the unfused version 7.6e-6 (tile 256; tile 128 was
1.1e-5). GQA is bit-identical (maxdiff 0.0 at all kvLens).

### End-to-end (SmolLM-135M, short prompt) — noisy, read with the caveat

`scripts/bench-forward-opt.ts` before/after (full JSON in
`forward-opt-bench-before.json` / `forward-opt-bench-after.json`;
the core-phase before-file is backed up in `/tmp/fwd-opt-core-phase/`):

| Metric (median) | Before | After run 1 | After run 2 |
|---|---|---|---|
| Generate, tok/s (3 reps) | 4.56 | 3.38 | 3.33 |
| Decode-only, tok/s | 5.89 (169.7 ms) | 4.41 (227.0 ms) | 4.60 (217.2 ms) |
| Prefill 5 tok, ms | 874 | 1219 | 1138 |
| Long (221 tok) prefill, s | 51.9 | 69.7 | 54.0 |
| Long decode median, ms | 386.5 | 273.4 | 254.3 |

The absolute after numbers look worse because the host got much busier
between the runs: loadavg 0.83 at the before run vs 4.41 / 1.61 at the
after runs, with two sibling workstreams running concurrently. The
before/after bench therefore does **not** measure the change — it
measures the load. The honest end-to-end check is the alternating A/B
in scratch trees under `/tmp/fwd-ab-old` (pre-change kernels) and
`/tmp/fwd-ab-new` (current code), same minimal decode bench
(24 single-row forwards, median ms), run old/new/old/new…:

| Pair | Old | New | Ratio |
|---|---|---|---|
| 1 | 402.8 ms | 389.9 ms | 1.03x |
| 2 | 413.4 ms | 371.2 ms | 1.11x |
| 3 | 270.3 ms | 389.7 ms | 0.69x (load swung mid-pair: loadavg hit 6.0) |
| 4 | 408.3 ms | 388.7 ms | 1.05x |

Pooled medians: old ~405.5 ms vs new ~389.2 ms ≈ **1.04x**. Kernel math
predicts ~1.10x on decode (swiglu 1.2x on ~57% of a decode token, GQA /
RoPE negligible at short context), and the A/B is directionally
consistent with that but too noisy to quote a clean end-to-end number.
Do not quote the bench's 4.56 → 3.33 tok/s as a regression — it is a
load artifact, contradicted by the interleaved kernel numbers and the
A/B pairs.

**Token identity: byte-for-byte.** The 16 generated tokens (and text
`" Paris.\nThe capital of France is the capital of France.\nThe capital"`)
are identical between the before and after bench runs, and the 24
tokens from every A/B run (old and new trees) are identical.

### Long context (997-token prompt)

`scripts/profile-forward-997.ts` (fox sentence x100, 1001 tokens
measured): prefill 317.2 s (316.9 ms/token) and decode median 306.6 ms
→ 3.26 tok/s, text `" the. The. The. The."` — **new code only, under
loadavg 5.69; there is no old-code run at 997 tokens, so this is a
characterization, not a before/after.** The 221-token long decode in the
bench needs a correction to the round-1 reading: its 8 samples declined
steadily (420 → 191 ms), i.e. the 386 ms median caught a warmup/GC
transient after the 52 s prefill; steady-state long decode is only
~20–30 ms above short decode, consistent with the measured GQA
(~15 ms/token at kvLen 221) + cache.read (~6 ms/token) — there is no
unexplained 2x long-decode cost.

`cache.read` at prefix 997 costs 0.27–0.33 ms/call (8–10 ms/token
×30 layers) for 1.5 MB copied per call — near the memcpy floor, so a
block-copy fast path inside `read()` was **not** done. The bigger
long-context cost is allocation churn: every decode token allocates
~10 MB of short-lived prefix copies at kvLen 221 (~45 MB at kvLen 997)
plus per-layer intermediates, which is the likely GC pressure behind
the transient above. Fixing it needs the deferred cache-interface
change (zero-copy attention / `readAppend`), not a faster copy.

### β / bandwidth-bound status — no direct β measurement

β was **not** measured directly (no new β number is claimed). What was
measured: a decode token moves ~536 MB of weights (134M params × 4 B)
for ~268 MFLOP of work — arithmetic intensity 0.5 FLOP/byte — at
~1.6 GFLOP/s effective, i.e. ~1.6–2.7 GB/s of weight streaming at
3–5 tok/s. Host RAM bandwidth is far above that, so the forward is
**scalar-execution-bound in JS** (per-operation / FP-latency cost),
not RAM-bandwidth-bound: the binding constraint is what each scalar
multiply-add costs, not the bytes moved. That is consistent with the
quantized paths (vNPU INT8/FP8, SW-code) being slower in-process
despite moving fewer bytes — shrinking bytes does not help wall-clock
until the per-op cost drops. The three changes in this round reduce
per-op and allocation overhead; none of them moves the backend toward
bandwidth-bound, because bytes were never the binding constraint here.

### Verification

- `npx tsc --noEmit`: clean.
- Targeted tests only (coordinator runs the full suite once at the
  end): `test/forward-opt.test.ts` **20/20 pass** (10 matmul + 10 new:
  fused SwiGLU vs serial reference at 3 shapes incl. odd/tile-remainder
  sizes, GQA vs serial reference at kvLen 21/5/1 + fully-masked query,
  RoPE cached vs per-element reference incl. positions 0/1/100/997/5000
  and multi-row); `test/engine.test.ts` + `test/vnpu-kernels.test.ts`
  pass; `test/novel-attention.test.ts` + `test/novel-numbers.test.ts`
  **17/17 pass**. 39 tests total, 0 fail.
- No push/merge. No change to defaults, scheduler, server, model, or
  `docs/RESEARCH.md`.

### Files

- Changed: `src/engine/attention.ts` (swiglu, groupedQueryAttention,
  applyRope/ropeRow only), `test/forward-opt.test.ts` (+10 tests),
  `docs/FORWARD-OPT.md` (this section)
- Added: `scripts/profile-forward-3.ts` (tiled-SwiGLU / GQA-u4 /
  cache.read profilers), `scripts/profile-forward-997.ts` (long-context
  characterization), `forward-opt-bench-before.json` /
  `forward-opt-bench-after.json` (re-run for this round; core-phase
  files backed up to `/tmp/fwd-opt-core-phase/`)
- Scratch only (not in the tree): `/tmp/fwd-ab-old`, `/tmp/fwd-ab-new`
  (alternating A/B trees; old tree has the three kernels reverted)

### Not checked / deferred to follow-up

- lm_head for last row only on prefill (21% of decode matmuls):
  still needs a `ForwardContext` contract decision (speculative/Jacobi
  verification reads every row's logits) — unchanged from round 1.
- Zero-copy attention / cache `readAppend`: the real long-context fix
  (kills the ~10–45 MB/token of prefix-copy allocation churn); needs a
  cache-interface change, sequenced next if long context is the
  priority.
- Allocation churn in the forward loop generally (per-layer
  intermediates: normed/q/k/v/attnOut/afterAttn/out arrays): unmeasured
  as a separate item; buffer reuse would be a bigger, riskier refactor.
- demo-100m in-process bench — not run this round.
- A quiet-host end-to-end re-run: the A/B pairs above are the best
  available while sibling workstreams share the host; re-run
  `bench-forward-opt.ts` when the host is quiet for a cleaner headline
  number.

## Profile: where a decode token goes (SmolLM-135M, in-process)

SmolLM-135M: 30 layers, hidden 576, 9 heads / 3 KV heads, headDim 64,
intermediate 1536, vocab 49152 (tied embeddings). Per decode token
(1 row) the forward does ~134M multiply-accumulates: ~106M in the 30
layers (q/k/v/o + SwiGLU MLP) and ~28M in lm_head (21% of the total).

Measured kernel shares per decode token (interleaved medians,
`scripts/profile-forward.ts`, host load ~1.6):

| Component | Per-token cost | Share |
|---|---|---|
| Matmuls (`matmulTransposeB`: q/k/v/o, gate/up/down, lm_head) | ~253 ms of ~290 ms | ~85% |
| GQA attention, kvLen 21 | 1.7 ms | ~0.6% |
| RoPE (`applyRope`, 30 layers) | 0.8 ms | ~0.3% |
| RMSNorm (60 calls) | 0.15 ms | ~0.05% |
| KV cache `read` (prefix 5, ×30 layers) | 0.15 ms | ~0.05% |
| Rest (residuals, copies, loop/await overhead) | ~35 ms | ~12% |

The engine is pure TypeScript with no native BLAS, and that caps it:
the serial dot product in `matmulTransposeB` runs at ~0.8–1.2 GFLOP/s
on this host — the FP-add dependency chain in `sum += a*b` stalls on
add latency. At the measured ~3.45 tok/s baseline that is ~0.9 GFLOP/s
of useful work. No TypeScript-level change can reach BLAS speeds; the
win available here is removing stalls inside the dot product, not
changing the algorithm. Attention, RoPE, norms, and cache reads are
all <1% each at short context, so they cannot move the headline
number — they are follow-up items, not the core.

Two things the profile also showed, for the record:

- The pre-transposed-weights matvec (`C = A @ Wt`, i,k,j loop) was
  **slower** (0.80–1.00x): the `c[j] +=` load/store traffic per
  inner iteration costs more in JS than the dot product's streaming
  reads. Rejected.
- KV `read` grows with context (48 µs/call at prefix 100 vs 5 µs at
  prefix 5, ×30 layers = ~1.4 ms/token at kvLen 100) and the
  per-row full-prefix read is O(n²) over a generation. At short
  context it is negligible; fixing it needs a cache-interface change
  (engine-side prefix snapshot or a cache `readAppend` op) and is
  deferred — it is not the short-prompt bottleneck.

## The change (one kernel, one file)

`src/engine/attention.ts`, `matmulTransposeB` only: the inner dot
product is unrolled by 4 with four independent accumulators
(`s0..s3`, combined at the end, remainder loop for `aCols % 4`).
Everything that calls it — q/k/v/o projections, SwiGLU gate/up/down
(via `swiglu`), MoE router/experts, and lm_head — gets the faster
kernel with no call-site changes and no new options. No other file
changed in the engine.

Why it works: four independent FP-add chains let V8 keep the
pipeline full instead of serializing on one add latency. Measured
(interleaved medians in the same process, so host load cancels):
**1.32–1.54x** on every SmolLM shape — q/o 1.41x, gate 1.34x, down
1.54x, lm_head 1.32x, prefill gate (5 and 16 rows) 1.37x/1.35x.
Outputs are bit-identical to the serial loop at these magnitudes
(max abs diff 0.0 in the profile; the test asserts ≤1e-5 vs a naive
reference, including odd `aCols` and `aCols < 4`).

## Measured before/after (short prompt, SmolLM-135M, in-process)

Protocol: prompt `"The capital of France is"` (5 tokens), 16 tokens,
temp 0 greedy, `scripts/bench-forward-opt.ts`, full JSON in
`forward-opt-bench.json` (per-run files `forward-opt-bench-before.json`
/ `forward-opt-bench-after.json`).

| Metric (median) | Before run 1 | Before run 2 | After run 1 |
|---|---|---|---|
| Generate, tok/s (3 reps) | 1.92 | 1.46 | **2.89** |
| Decode-only, tok/s (16 single-row forwards) | 1.63 | 1.38 | **3.20** |
| Prefill 5 tok, ms | 1899 | 3191 | 2881 |
| Generated tokens | identical in all runs | | identical |

Generated text is byte-identical before/after and matches the
spec-draft baseline (` Paris.\nThe capital of France is the capital
of France.\nThe capital`).

**Read these numbers with the load caveat.** The host is shared with
three other agents and absolute times swing ~40%+ run to run: the two
"before" runs alone differ by 1.92 vs 1.46 tok/s, and the after run
happened as load fell (loadavg at end of run: 4.53 / — / 0.77). So the
end-to-end ratios (generate 1.5–2.0x, decode-only 2.0–2.3x) overstate
the kernel gain — part of them is a quieter host. The defensible
claim is the load-cancelled one: the matmul itself is 1.32–1.54x
faster, it is ~85% of a decode forward, so the real decode gain is
roughly **1.3–1.5x**; the end-to-end runs are consistent with that
once load is accounted for, but they do not prove a bigger number.
Prefill shows no clean signal (1899 / 3191 / 2881 ms — noise
dominates at 5 rows); do not quote a prefill gain.

For reference, the pre-existing spec-draft baseline measured under
quieter conditions was 3.45 tok/s; the after-run 2.89 tok/s was
measured under heavier load, so the two are not directly comparable.

## Verification

- `npx tsc --noEmit`: **not clean tree-wide** — 1 remaining error in
  `test/heartbeat-p7.test.ts:542`, a file owned by a concurrently
  running agent (it had many errors mid-edit; it is down to 1). Zero
  errors in any file this piece touched. I did not touch that file.
- `node --test test/forward-opt.test.ts`: **10/10 pass** (matmul vs
  serial reference at 8 shapes incl. lm_head and remainder paths,
  determinism, shape-mismatch throw).
- Full test suite: not run (coordinator runs it once at the end, per
  brief). Targeted tests only while working.
- No push/merge. No change to defaults, scheduler, server, model, or
  `docs/RESEARCH.md`.

## Not checked / deferred to follow-up

- Long prompt (997-token) before/after — partially done in round 2:
  a 1001-token characterization run exists (`scripts/profile-forward-997.ts`,
  317 s prefill / 3.26 tok/s decode under loadavg 5.69) but there is no
  old-code run at that length, so it is not a before/after.
- demo-100m in-process bench — deferred.
- Fused SwiGLU, cached RoPE tables, in-place-softmax GQA: **done in
  round 2** (see above) — 1.19–1.25x, 6.7x, 1.10–1.19x respectively.
- lm_head for last row only: would save most of the 21% lm_head
  share on prefill, but needs a signal through `ForwardContext`
  (speculative/Jacobi verification reads every row's logits), so it
  is a contract decision, not a kernel tweak.
- Zero-copy attention / cache `readAppend` and general allocation
  churn: the remaining long-context work, sequenced as follow-up
  (see round 2 section).

## Files

- Changed: `src/engine/attention.ts` (matmulTransposeB only)
- Added: `test/forward-opt.test.ts`, `scripts/bench-forward-opt.ts`,
  `scripts/profile-forward.ts`, `scripts/profile-forward-2.ts`
  (profilers; the -2 script holds the deferred fused-SwiGLU/GQA
  candidates), `docs/FORWARD-OPT.md` (this file),
  `forward-opt-bench.json` (+ per-run `forward-opt-bench-before.json`,
  `forward-opt-bench-after.json`)
