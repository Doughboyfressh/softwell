# Forward-pass optimization — core result (2026-11-22)

Core cut of the forward-pass piece: profile + the single biggest
optimization, one short-prompt before/after. The full program (fused
SwiGLU, cached RoPE tables, in-place-softmax GQA, long-prompt bench,
demo-100m) is deferred to follow-up. Nothing here changes any default,
touches the scheduler/server/model code, or adds a physical backend.

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

- Long prompt (997-token) before/after — deferred; the O(n²)
  attention + full-prefix cache read dominates there and is a
  separate change.
- demo-100m in-process bench — deferred.
- Fused SwiGLU (gate+up sharing one x pass): profiled at 2.24x vs
  current `swiglu` (1.83x for u4 matmuls alone), i.e. ~1.2x on top of
  this change. Next in line.
- Cached RoPE cos/sin tables: profiled at 6.8x on `applyRope`
  (27.2 → 4.0 µs/row/layer) but only ~0.3% of a short decode token;
  worth doing for prefill, not for the headline.
- In-place-softmax GQA (drop the per-head weights array): 1.17x at
  kvLen 21, 1.05x at kvLen 512 — small, cheap, follow-up.
- lm_head for last row only: would save most of the 21% lm_head
  share on prefill, but needs a signal through `ForwardContext`
  (speculative/Jacobi verification reads every row's logits), so it
  is a contract decision, not a kernel tweak.

## Files

- Changed: `src/engine/attention.ts` (matmulTransposeB only)
- Added: `test/forward-opt.test.ts`, `scripts/bench-forward-opt.ts`,
  `scripts/profile-forward.ts`, `scripts/profile-forward-2.ts`
  (profilers; the -2 script holds the deferred fused-SwiGLU/GQA
  candidates), `docs/FORWARD-OPT.md` (this file),
  `forward-opt-bench.json` (+ per-run `forward-opt-bench-before.json`,
  `forward-opt-bench-after.json`)
