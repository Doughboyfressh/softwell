# softwell Benchmarks

How the virtual machine is measured against the best enterprise
hardware, what "better" means, and the honest results. Verdicts are
computed from real measurements taken by `npm run bench` on the
machine that ran the bench — never asserted, never hardcoded.

```sh
npm run bench                    # all devices, 1 donor
npm run bench -- --donors 4      # scaling curve 1..4 donors
npm run bench -- --only vgpu     # one device only
npm run bench -- --json results.json
```

Exit code is 0 even when verdicts are FAIL: a FAIL is a measurement,
not a crash. Non-zero only on a harness error (a bench module missing
or throwing, bad arguments).

## Targets

The numbers the virtual devices must beat, from
`src/bench/targets.ts` (verified 2026-11-22).

| Device | Target (primary) | Metric | Target value |
|---|---|---|---|
| vGPU | NVIDIA Rubin R100 | FP4 dense | 35,000 TFLOPS |
| vGPU | NVIDIA Rubin R100 | FP8 | 17,500 TFLOPS |
| vGPU | NVIDIA Rubin R100 | FP16 / BF16 | 4,000 TFLOPS |
| vGPU | NVIDIA Rubin R100 | memory bandwidth | 16,250 GB/s (130 TB/s / 8 GPUs) |
| vGPU | NVIDIA Rubin R100 | memory capacity | 288 GB |
| vGPU | NVIDIA B200 (secondary) | FP4 dense | 9,000 TFLOPS |
| vGPU | NVIDIA B200 (secondary) | FP8 | 4,500 TFLOPS |
| vGPU | NVIDIA B200 (secondary) | FP16 / BF16 | 2,250 TFLOPS |
| vGPU | NVIDIA B200 (secondary) | memory bandwidth | 8,000 GB/s |
| vGPU | NVIDIA B200 (secondary) | memory capacity | 192 GB |
| vCPU | AMD EPYC 9965 (Turin) | cores / threads | 192 / 384 |
| vCPU | AMD EPYC 9965 (Turin) | L3 cache | 384 MB |
| vCPU | AMD EPYC 9965 (Turin) | memory bandwidth | 614 GB/s (DDR5-6400, AMD up-to figure) |
| vCPU | AMD EPYC 9965 (Turin) | FP32 peak | ~27.6 TFLOPS (ESTIMATE, not a vendor number: 192 cores x 2.25 GHz x 32 FP32 FLOPs/cycle) |
| vRAM | HBM3e (NVIDIA B200) | memory bandwidth | 8,000 GB/s |
| vRAM | HBM3e (NVIDIA B200) | memory capacity | 192 GB |
| vSSD | Micron 9650 (PCIe Gen6) | sequential read | 28 GB/s |
| vSSD | Micron 9650 (PCIe Gen6) | sequential write | 14 GB/s |
| vSSD | Micron 9650 (PCIe Gen6) | random read | 5,500,000 IOPS |
| vSSD | Micron 9650 (PCIe Gen6) | random write | 900,000 IOPS |
| vNPU | Google TPU v6e (Trillium) (primary) | INT8 peak | 1,836 TOPS |
| vNPU | Google TPU v6e (Trillium) (primary) | BF16 peak | 918 TFLOPS |
| vNPU | Google TPU v6e (Trillium) (primary) | memory bandwidth | 1,638 GB/s |
| vNPU | Google TPU v6e (Trillium) (primary) | memory capacity | 32 GB |
| vNPU | Google TPU v5p (secondary) | INT8 peak | 918 TOPS |
| vNPU | Google TPU v5p (secondary) | BF16 peak | 459 TFLOPS |
| vNPU | Google TPU v5p (secondary) | FP8 peak | 459 TFLOPS |
| vNPU | Google TPU v5p (secondary) | memory bandwidth | 2,765 GB/s |
| vNPU | Google TPU v5p (secondary) | memory capacity | 95 GiB |
| vNPU | AWS Trainium2 (secondary) | FP8 peak | 1,299 TFLOPS |
| vNPU | AWS Trainium2 (secondary) | BF16 / FP16 peak | 667 TFLOPS |
| vNPU | AWS Trainium2 (secondary) | memory bandwidth | 2,900 GB/s |
| vNPU | AWS Trainium2 (secondary) | memory capacity | 96 GiB |
| vNPU | AWS Trainium2 (secondary) | on-chip SRAM (SBUF) | 224 MiB |
| vNPU | AWS Inferentia2 (secondary) | INT8 peak | 380 TOPS |
| vNPU | AWS Inferentia2 (secondary) | FP16 / BF16 / cFP8 peak | 190 TFLOPS |
| vNPU | AWS Inferentia2 (secondary) | memory bandwidth | 820 GiB/s (as published) |
| vNPU | AWS Inferentia2 (secondary) | memory capacity | 32 GiB |

Why these chips: Rubin R100 is the genuinely current best enterprise
GPU (in full production since early 2026); B200 is kept as the
secondary target because it is the widely deployed reference. EPYC
9965 is still the shipping core-count leader (EPYC 9006 "Venice" is
announced but has no shipping specs to benchmark against). Micron
9650 is still the fastest shipping data-center SSD. TPU v6e
(Trillium) is the vNPU's primary target — the current
inference-shaped TPU; v5p, Trainium2 and Inferentia2 are
secondaries. The full vNPU target table, sources, and the v5p
INT8-vs-FP8 provenance note are in docs/VNPU.md §5.2.

Sources:

- https://www.nvidia.com/en-us/data-center/hgx/
- https://www.spheron.network/blog/nvidia-b200-complete-guide/
- https://www.techpowerup.com/cpu-specs/epyc-9965.c3904
- https://www.cpu-monkey.com/en/cpu-amd_epyc_9965
- https://my.micron.com/products/storage/ssd/data-center-ssd/9650-ssd
- https://www.tomshardware.com/pc-components/ssds/microns-industry-first-pci-6-0-ssd-promises-sequential-reads-up-to-28-000-mb-s-245-tb-ssd-also-coming-for-those-who-need-capacity-more-than-cutting-edge-speed

## How "better" is defined

"Better" means **aggregate throughput or capacity by pooling donors**.
One donor is one machine; N donors contribute N machines' worth of
FLOPS, bytes, and bytes/second to the virtual device, and the virtual
device's measured aggregate is compared against a single enterprise
chip. A PASS means the pool, at the measured donor count, already
out-performs that one chip on that metric.

What is NOT claimed: **single-op latency**. One operation on one donor
is bounded by that donor's hardware plus the network hop to reach it.
Pooling never makes a single matmul, a single page read, or a single
IOPS faster than the fastest single device that could run it. The
claim is that the pool is a bigger machine, not a faster single
operation. See "Honest limits" below.

Verdicts are computed, never asserted: `PASS` when
`measured >= target`, else `FAIL` (`makeResult` in
`src/bench/types.ts`). Every row also reports `perDonor`
(`measured / donors`) and `donorsToBeat`.

## Results

**Measured 2026-11-22** with `npm run bench -- --donors 1` — one
`npm run bench` invocation, 1 donor, 18 rows, exit 0. The numbers
below are rounded to 3–4 significant figures from that run's raw
JSON output (re-run `npm run bench` to reproduce; expect run-to-run
variation — see the conditions note).

> **Read these numbers with their conditions attached.** The run
> happened on a shared 4-core VM while other agents were running
> test suites and benches on the same host (the machine had just
> rebooted; load was falling through the run). Absolute rates move
> a lot between runs: an earlier `npm run bench` the same day
> measured vRAM read 0.189 GB/s vs 0.128 GB/s here, vSSD write
> 0.120 GB/s vs 0.030 GB/s here, and vGPU memory bandwidth 0 GB/s
> vs 0.652 GB/s here. Treat the *verdicts* (all FAIL except the
> two flagged artifacts below) and the *orders of magnitude* as the
> finding; treat the third significant figure as noise. Re-run on a
> quiet machine before quoting any single rate. The vSSD large
> workload also needed `TMPDIR` pointed at a disk-backed directory:
> the default `/tmp` is a 512 MB tmpfs and the 256 MiB + 64 MiB
> objects do not fit in it alongside other agents' temp dirs (the
> first vSSD attempts failed with `could not store chunk ... on any
> donor node` for exactly that reason — an environment limit, not
> a vSSD bug).

| device | metric | measured | target | margin | donors | donors-to-beat | verdict |
|---|---|---|---|---|---|---|---|
| vgpu | matmul TFLOPS (FP32, 1024^3) | 0.000459 TFLOPS | 4,000 (Rubin BF16) | 0.00001% | 1 | 8,713,611 | FAIL |
| vgpu | matmul TFLOPS (FP32, 1024^3) | 0.000459 TFLOPS | 2,250 (B200 BF16) | 0.00002% | 1 | 4,901,407 | FAIL |
| vgpu | matmul TFLOPS (FP32, 2048^3) | 0.00105 TFLOPS | 4,000 (Rubin BF16) | 0.00003% | 1 | 3,821,783 | FAIL |
| vgpu | matmul TFLOPS (FP32, 2048^3) | 0.00105 TFLOPS | 2,250 (B200 BF16) | 0.00005% | 1 | 2,149,753 | FAIL |
| vgpu | memory bandwidth (copy/add, FP32) | 0.652 GB/s | 16,250 (Rubin) | 0.004% | 1 | 24,906 | FAIL |
| vgpu | memory bandwidth (copy/add, FP32) | 0.652 GB/s | 8,000 (B200) | 0.008% | 1 | 12,262 | FAIL |
| vgpu | aggregate matmul TFLOPS (FP32, 1024^3 x 1 donor) | 0.000247 TFLOPS | 4,000 (Rubin BF16) | 0.000006% | 1 | 16,207,180 | FAIL |
| vcpu | virtual cores | 4 cores | 192 (EPYC 9965) | 2.1% | 1 | 48 | FAIL |
| vcpu | FP64 TFLOPS (FMA, Float64Array) | 0.000284 TFLOPS | 27.6 (EPYC FP32 est.) | 0.001% | 1 | 97,138 | FAIL |
| vcpu | flops task throughput | 1.42 tasks/s | 0 (no target) | n/a | 1 | 1 | PASS* |
| vram | sequential READ bandwidth (256 MiB) | 0.128 GB/s | 8,000 (HBM3e) | 0.002% | 1 | 62,477 | FAIL |
| vram | sequential WRITE bandwidth (256 MiB) | 0.0924 GB/s | 8,000 (HBM3e) | 0.001% | 1 | 86,583 | FAIL |
| vram | pooled capacity | 1.07 GB | 192 (HBM3e) | 0.56% | 1 | 179 | FAIL |
| vram | single-page (4 MiB) read latency | 8.99 ms | 0.05 ms (aspirational) | 17983% | 1 | inf | FAIL |
| vssd | sequential read | 0.0447 GB/s | 28 (Micron 9650) | 0.16% | 1 | 627 | FAIL |
| vssd | sequential write | 0.0305 GB/s | 14 (Micron 9650) | 0.22% | 1 | 460 | FAIL |
| vssd | random 4 KiB read | 608 IOPS | 5,500,000 (Micron 9650) | 0.011% | 1 | 9,044 | FAIL |
| vssd | capacity stored | 0.3125 GiB | 0.3125 (informational) | 100% | 1 | 1 | PASS* |

\* **The two PASS rows are verdict artifacts, not wins — read
them as informational:**
- *vCPU flops task throughput* has target 0 (no enterprise
  tasks/s figure exists for the EPYC); `measured >= 0` is trivially
  true and the module's own notes say the verdict is not meaningful.
- *vSSD capacity stored* sets target = measured by construction
  (informational row; pool capacity was not probed), so it passes
  by definition.

Two more notes from the runs:
- *vRAM latency is now judged the right way round.* An earlier
  run the same day reported this row as PASS because the generic
  `measured >= target` rule points the wrong way for latency
  (bigger is worse). The vRAM module has since been fixed to build
  the latency row by hand — PASS only when measured <= target,
  donors-to-beat infinite — and this run correctly reports FAIL
  (8.99 ms vs the 0.05 ms aspirational reference, which is not a
  vendor spec).
- *vGPU memory bandwidth was 0 GB/s in the earlier run and
  0.652 GB/s in this one.* The 0 was a calibration quirk under
  load, not a physical rate; 0.652 GB/s is the current measured
  value and the one to quote, with the usual variance warning.
- *Scaling curve measured — see "Scaling curve (measured
  2026-11-22)" below.* The `--donors 2` run shows capacity and
  cores scaling exactly linearly, throughput flat-to-modestly-up on
  this shared host. `donorsToBeat` remains a linear projection from
  per-donor rates, not a measured multi-host curve.

### What the numbers actually say

With one donor on this small VM, every substantive metric fails by
2–6 orders of magnitude — which is exactly what the honest math
predicts: one 4-core CPU donor's FP32 matmul is ~10⁶x short of a
Rubin's BF16 tensor cores, and loopback-HTTP page transfers are
~10⁴x short of HBM3e. The interesting columns are `donors-to-beat`
(capacity: 179 donors of 1 GiB; vSSD sequential read: 627 donors;
vCPU cores: 48 donors) and the scaling curve below: those are the
numbers that decide whether pooling can ever close the gap, and at
what donor count the network stops the curve from being linear.

## Scaling curve (measured 2026-11-22)

`npm run bench -- --donors 2 --json bench-results-2026-11-22-donors2.json`
(37 rows, exit 0, 195 s). Machine: AMD EPYC 9D64, 4 cores visible,
7.7 GiB RAM, Node v24.22.1, linux x64. Same conditions warning as
above: shared host, run-to-run variance of 2–3x on throughput rows.
The 1-donor column below is from this same run (the vGPU module
measures in-process and over-HTTP rows; the HTTP aggregate row is
the one quoted).

| metric | 1 donor | 2 donors | scaling |
|---|---|---|---|
| vGPU aggregate matmul TFLOPS (FP32, 1024^3, over HTTP) | 0.00045 | 0.00150 | ~3x (noisy; the two 1-donor measurements in this run were 0.00154 in-process vs 0.00045 over HTTP) |
| vGPU memory bandwidth (copy/add, FP32) | 2.75 GB/s | 2.83 GB/s | flat |
| vCPU virtual cores | 4 | 8 | exact linear (2/2 donors reported real core counts) |
| vCPU FP64 TFLOPS (FMA) | 0.00148 | 0.00170 | flat |
| vRAM sequential read (256 MiB) | 0.260 GB/s | 0.314 GB/s | modest gain |
| vRAM sequential write (256 MiB) | 0.311 GB/s | 0.257 GB/s | flat / noise |
| vRAM pooled capacity | 1.07 GB | 2.15 GB | exact linear |
| vRAM single-page read latency | 9.7 ms | 7.5 ms | noise |
| vSSD sequential read | 0.226 GB/s | 0.203 GB/s | flat |
| vSSD sequential write | 0.182 GB/s | 0.190 GB/s | flat |
| vSSD random 4 KiB read | 3,900 IOPS | 3,778 IOPS | flat |

What the curve says: **capacity and cores scale exactly linearly;
throughput does not move on this host.** Every donor here is an
in-process server sharing the same 4 host cores and one event loop,
so a second donor adds no new silicon — aggregate throughput is
capped by the host, and per-donor rates fall as donors grow. That
is why `donorsToBeat` is computed from the per-donor rate and
labelled a linear projection: it is the donor count that would beat
the target *if* each donor were a distinct machine contributing its
own cores, memory, and disk. A multi-host scaling curve (donors on
separate machines) is still unmeasured and is the next thing to run
before quoting any donor count as a plan.

## Scaling math

Each bench module measures an aggregate rate across `donors` donors
and reports the per-donor rate:

```
perDonor     = measured / donors
donorsToBeat = ceil(target / perDonor)        (at least 1; inf if perDonor is 0)
margin       = measured / target             (as a percentage in the table)
```

`donorsToBeat` is the honest form of a FAIL: it says how big the donor
pool would have to be — at the measured per-donor rate — for the
virtual device to beat the target. It assumes linear scaling, which
is the thing the scaling curve (`--donors N`, which runs 1..N and
prints a row per donor count) exists to check: if per-donor rates
fall as donors grow, the network or the coordinator is the bottleneck
and the linear projection is optimistic.

Example of the arithmetic (illustrative, not a measurement): if one
donor delivers 2 GB/s of vRAM bandwidth against the 8,000 GB/s HBM3e
target, `donorsToBeat = ceil(8000 / 2) = 4,000` donors.

## Honest limits

- **Single-op latency vs a B200/Rubin.** A single kernel launch, a
  single page read, or a single small I/O on one donor cannot beat
  the same op on one B200 or Rubin R100. The donor adds a network hop
  and runs on weaker hardware; pooling only helps when the work splits
  across donors. Decode latency for one sequence is the sharpest case:
  it is memory-bandwidth bound per token, and batching helps aggregate
  throughput, not per-sequence latency (see "The honest hardware math"
  in ARCHITECTURE.md).
- **Network-bound cases.** Anything whose working set does not fit on
  one donor, or that shuffles data between donors per step (tensor
  parallel splits, KV cache reads across nodes, vRAM page fetches
  over HTTP), pays the interconnect on every step. On a 1 GbE or
  Wi-Fi pool the network, not the donors, sets the ceiling; the
  scaling curve will show it as falling per-donor rates.
- **FP32 vs FP4/BF16 tensor cores.** The vGPU's CPU fallback kernels
  run FP32 math in TypeScript. The Rubin/B200 targets are FP4/FP8/BF16
  tensor-core numbers — a different kind of arithmetic on dedicated
  silicon. Comparing FP32 CPU throughput against FP4 tensor-core
  throughput is honest only as an aggregate-throughput statement
  ("this many donors' FP32 FLOPS equal one Rubin's FP4 FLOPS"); it
  says nothing about running an FP4-quantized model at tensor-core
  speed. Real GPU donors with a CUDA/Metal backend are what close
  that gap, and they do not exist yet.
- **HTTP+JSON overhead where it remains.** Hidden states, pages, and
  chunks still travel as JSON (base64 for bytes) over HTTP between
  donors. At demo scale that is fine; at benchmark scale it is a
  measurable tax on every transfer, and it inflates the per-donor
  rates the benches report. A binary/gRPC transport would change the
  numbers without changing the platform interfaces.
- **What still needs real GPU donors.** There is no CUDA/Metal
  backend: `GpuNode` reports the device that actually executed
  (`cpu`), never a GPU it does not have. Until real GPU donors join
  the pool, every vGPU number is a CPU-fallback number, and the
  donors-to-beat for compute will be large. That is the measurement
  doing its job, not a reason to adjust it.
- **Decoding loses on this CPU backend.** Speculative decoding
  measures 0.80× (demo-10m) / 0.56× (demo-100m) and Jacobi 0.76× /
  0.52× vs autoregressive, at β ≈ 1 (a batched forward costs ~one
  single-row forward per row). The theorems predict exactly this;
  the win needs bandwidth-bound hardware (β < 1) and a drafter with
  a real acceptance rate. The 92% acceptance in the bench is an
  artifact of random-weight models collapsing to a constant token.
- **Attention wins are data-conditional.** Landmark/sparse win big
  only where the data satisfies their thresholds (coherent keys /
  concentrated attention), and at headDim 64 those thresholds are
  strict (see "Novel paths" above). On incoherent data they fall
  back to exact attention at 0.72–1.15× the cost. No perplexity
  impact has been measured on a real model.
- **SW-code saves bytes, not time, in-process.** Packed weights
  measure 0.14–0.35× tokens/s and quantized KV 0.58–0.82× on this
  backend: the unpack cost dominates because there is no slow link
  for the traffic saving to pay against. SW2's 72% per-layer RMS
  error on synthetic weights would badly damage a real model;
  perplexity impact of all SW modes on a real model is unmeasured.
- **All novel-path speedups are measured on random-weight models.**
  Greedy output collapses to a constant token; acceptance rates,
  and any quality claim, are artifacts until measured on a trained
  model. Single-rep timings on a shared host; ratios are stable,
  absolute times are not.

## Hardware the numbers were measured on

The 2026-11-22 run above was measured on this machine (the machine
line the harness prints at the top of every run):

- CPU: AMD EPYC 9D64 88-Core Processor — 4 cores visible to this VM
- RAM: 7.7 GiB total
- Node: v24.22.1, linux x64
- Date: 2026-11-22

Do not copy numbers from one machine's run into another machine's
row. When the bench is re-run (especially the `--donors N` scaling
curve on a quiet machine), append the new table here with its own
machine line and date rather than overwriting this one.

Harness note: the first full run completed all 18 measurements and
wrote its JSON, but the process did not exit — bench modules leave
handles behind (pooled worker threads) that keep the event loop
alive. The runner now flushes stdout and exits explicitly after a
completed run (see `src/bench/run.ts`); the measurements themselves
were unaffected.

## Re-run, later 2026-11-22

Re-ran `npm run bench -- --donors 1 --json bench-results-2026-11-22.json`
the same day (raw JSON in `bench-results-2026-11-22.json`; 16 tests
also pass). Machine line this time: AMD EPYC 9D25 126-Core Processor
(4 cores), 7.7 GiB RAM, Node v24.22.1, linux x64 — a different host
model than the first run's 9D64, so cross-run differences are partly
host, partly contention. Totals: 2 PASS (both informational artifacts),
16 FAIL. Same verdicts as the first run, with two corrections:

- vGPU memory bandwidth measured **1.41 GB/s** this time (Rubin:
  11,524 donors to beat; B200: 5,674) — the earlier 0 GB/s does not
  reproduce, so do not quote 0 GB/s.
- vRAM single-page latency (18.1 ms vs 0.05 ms reference) now
  correctly verdicts **FAIL** — the earlier PASS artifact is fixed.

Headline donors-to-beat (1 donor, linear projection): vCPU cores 48;
vRAM capacity 179; vSSD sequential read 103, sequential write 65,
random 4 KiB read 1,807; vRAM read bandwidth 21,394; vGPU matmul
vs B200 BF16 1,308,060 (1024^3) and vs Rubin 2,325,439. Closest
gaps remain capacity/cores/SSD throughput; compute vs tensor-core
targets remains 6 orders of magnitude out. The `--donors 2`
scaling curve has now been measured — see the section below.

## Novel paths (wave 2, measured 2026-11-22)

`npm run bench:novel` (`src/bench/novel-bench.ts`) — Part A:
attention-only prefill scaling; Part B: end-to-end prefill/decode
per config on the demo models. Reproduce with
`npm run bench:novel -- --json novel-bench-results-2026-11-22.json`.
Raw JSON: `novel-bench-results-2026-11-22.json` (65 rows; Part A and
Part B were run as two invocations of the same harness after a
service restart killed the first combined run mid–Part B; the rows
are concatenated unchanged, and the Part A re-run reproduced the
killed run's ratios).

Machine: AMD EPYC 9D25 126-Core Processor (4 cores), 7.7 GiB RAM,
Node v24.22.1, linux x64. Models: demo-10m (10M params, random
weights) and demo-100m (104.37M params, hidden 1024, 12 layers,
random weights, `demo/models/demo-100m/`). Part B: prompt 128, 16
generated tokens, greedy, one rep per config. Part A: synthetic K/V,
16 heads / 8 KV heads / headDim 64, one rep per cell.

> **Read these with their conditions attached, as above.** Absolute
> times move ~40% between runs on this shared host (Part A exact
> n=2048: 8,689 ms in one run, 12,562 ms in the next); the
> *ratios* are stable across runs and are the finding. Part B
> configs ran sequentially and the host drifted during the 100m run:
> TTFT for the configs run after the SW-weight configs is inflated
> ~1.5× (kv/spec/jacobi TTFT 36–40 s vs baseline 24.7 s, while
> landmark/sparse, run first, show TTFT *below* baseline) — decode
> tokens/s is the more reliable comparison for those rows. All
> speedups are on random-weight models whose greedy output collapses
> to a constant token.

### Part A — attention-only prefill (wall ms, speedup vs exact)

Regimes (exact parameters in `src/bench/novel-bench.ts`):
*coherent* — block-constant keys + N(0, 0.001) noise per element
(within-block radius R_b ≈ 0.01, δ_b ≈ 0.01 ≤ δ_max = 0.05);
*concentrated* — block 0 keys at +12u, all other blocks at −12u,
queries at +4u (logit gap ≈ 12); *incoherent* — iid N(0,1) keys.
`ops` is counted multiply-adds vs exact; `err` is max L2 error vs
exact prefill, with the theorem bound in parentheses. No bound was
violated in any run; measured errors run ~10³× below the bounds.

| regime / mode | n=256 | n=512 | n=1024 | n=2048 | n=4096 |
|---|---|---|---|---|---|
| coherent / landmark | 1.47× (ops 0.24×) | 4.66× (0.15×) | 9.84× (0.09×) | 15.41× (0.06×) | 20.41× (0.04×) |
| coherent / sparse | 0.76× (1.04×) | 0.99× (1.03×) | 1.01× (1.02×) | 1.00× (1.01×) | 0.90× (1.01×) |
| concentrated / sparse | 1.92× (0.33×) | 5.17× (0.21×) | 6.64× (0.14×) | 10.46× (0.10×) | 9.04× (0.12×) |
| concentrated / landmark | 0.68× (1.02×) | 0.98× (1.01×) | 0.91× (1.00×) | 0.94× (1.00×) | 1.02× (1.00×) |
| incoherent / landmark | 0.91× (1.02×) | 0.90× (1.01×) | 0.90× (1.00×) | 0.92× (1.00×) | 0.72× (1.00×) |
| incoherent / sparse | 0.96× (1.04×) | 0.96× (1.03×) | 0.96× (1.02×) | 0.94× (1.01×) | 1.15× (1.01×) |

Exact prefill scales ~4× per length doubling (coherent: 245 → 730 →
3,184 → 12,562 → 45,906 ms); landmark on coherent data scales ~2.4×
per doubling and sparse on concentrated data ~2.7×. Errors:
landmark/coherent max err 3.3–3.9e-3 vs bound ~11; sparse/
concentrated max err 0.75–1.8e-3 vs bound 0.1–1.3; every fallback
cell has max err exactly 0.0 (the fallback *is* exact attention).

**The calibration lesson, stated plainly.** The first version of
this bench used coherent noise 0.01 and concentrated alignment ±3u
— and measured ~1.0× everywhere, because the approximation
thresholds never triggered: at headDim 64, δ_b = scale·‖q‖·R_b
grows as √d for fixed per-element noise (‖q‖ ≈ 8, R_b ≈ 0.08 at
noise 0.01, so δ_b ≈ 0.08 > δ_max = 0.05), and the sparse mass
bound needs a logit gap above ln(B/1e-3) ≈ 10.7, which ±3u does not
provide (gap ≈ 1.5). The regimes were recalibrated to data the
theorems actually cover (noise 0.001; alignment ±12u) — the win is
real, but it is *conditional on data satisfying the thresholds*, and
at d=64 that is a strong condition on real KV caches, whose
within-block dispersion is unmeasured. The fallback cost is small
(ops 1.00–1.04×) except for one case: landmark on incoherent data
at n=4096 measures 0.72×, because the reference implementation
builds a per-item object list for every position in the fallback
path — an implementation cost, not a theorem cost.

### Part B — end-to-end, prompt 128, 16 tokens

| config | demo-10m TTFT | demo-10m tok/s | demo-100m TTFT | demo-100m tok/s |
|---|---|---|---|---|
| baseline | 2,471 ms | 52.08 (1.00×) | 24,656 ms | 5.17 (1.00×) |
| landmark | 2,308 ms | 54.38 (1.04×) | 22,642 ms | 5.10 (0.99×) |
| sparse | 2,309 ms | 51.98 (1.00×) | 23,103 ms | 4.98 (0.96×) |
| sw2 weights | 6,388 ms | 18.24 (0.35×) | 127,195 ms | 0.74 (0.14×) |
| sw3 weights | 10,766 ms | 10.70 (0.21×) | 148,101 ms | 0.74 (0.14×) |
| sw4 weights | 10,787 ms | 10.53 (0.20×) | 150,473 ms | 0.73 (0.14×) |
| kv-sw3 | 2,671 ms | 42.72 (0.82×) | 38,169 ms | 3.06 (0.59×) |
| kv-sw4 | 2,799 ms | 40.88 (0.79×) | 40,263 ms | 2.99 (0.58×) |
| speculative (K=4) | 2,471 ms | 41.82 (0.80×), 5 forwards, α=92% | 36,712 ms | 2.91 (0.56×), 5 forwards, α=92% |
| jacobi (K=4) | 2,371 ms | 39.79 (0.76×), 6 forwards | 36,364 ms | 2.71 (0.52×), 6 forwards |

What this says, without flattery:

- **Attention end-to-end is a wash at prompt 128** (0.96–1.04×).
  Attention is a small fraction of a step at that length, and the
  demo models' keys are incoherent, so landmark/sparse degrade to
  exact attention (their logits match exact to 0.0 on demo-10m).
  The Part A win only shows end-to-end at long context, which this
  run did not measure — a 1k–4k-prompt end-to-end run is the missing
  measurement.
- **SW-code weights measure 5–7× slower in-process** (0.14–0.35×
  tok/s), not faster. The packed matmul's unpack cost dominates
  because there is no slow link in-process for the traffic saving
  to pay against — the weights never leave this machine's memory.
  The saving that *is* realized is memory: the 10m stage shrinks
  from 37.9 MB to 3.3 MB (sw2) / 4.5 MB (sw3) / 5.7 MB (sw4). On
  100m all three SW modes measure identically (~0.14×), i.e. the
  packed-matmul cost is bit-width-insensitive at this scale; the
  10m sw2-vs-sw3 gap (0.35× vs 0.21×) is within run-to-run noise
  and is not claimed as a property of sw2.
- **Quantized KV measures slower too** (0.58–0.82×): the packed-KV
  attention path pays unpack cost per step.
- **Speculative/Jacobi lose wall-clock, as the theorems predict at
  β ≈ 1** (0.80×/0.76× on 10m, 0.56×/0.52× on 100m — worse than
  wave 1's 0.91×/0.53× on a short prompt, because longer prompts
  make each batched row more expensive). They do cut target
  forwards (5–6 vs 16 for 16 tokens). The 92% acceptance rate is
  an artifact of the random-weight models collapsing to a constant
  token, not a property of the n-gram drafter — do not quote it as
  drafter quality.
- **Nothing here refutes the theorems.** The error bounds held
  (Part A), the op-count scaling held (Part A), the β≈1 cost model
  predicted the decoding losses (Part B), and the traffic ratios
  held (wave 1). What the end-to-end numbers show is that on this
  compute-bound, single-machine, in-process backend, *time is spent
  on FLOPs, not on moving bytes across a slow link* — so inventions
  that save bytes do not save time here. Their wins live where
  bytes cross a slow boundary (multi-donor serving) or where FLOPs
  are actually removed (attention at long context).

## vNPU (measured 2026-11-22)

`npm run bench -- --only vnpu --json vnpu-bench-results-2026-11-22.json`
(11 rows, exit 0). Machine: AMD EPYC 9D25 126-Core Processor
(4 cores visible), 7.7 GiB RAM, Node v24.22.1, linux x64; loadavg
~1.0–1.7 during the run (quiet host). Raw JSON:
`vnpu-bench-results-2026-11-22.json`. Perplexity / tok/s / kernel
numbers are in `docs/VNPU.md` §5.3 (same machine, same day;
`vnpu-perplexity-bench.json`, `vnpu-tok-bench.json`,
`vnpu-kernel-bench.json`). Timestamps in the JSONs are UTC
2026-11-23; the run was the evening of 2026-11-22 PST.

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
| vnpu | pooled SRAM capacity | 256 MiB | 224 (Trainium2 SBUF) | 114.3% | 1 | 1 | PASS |
| vnpu | aggregate INT8 TOPS (MATMUL_INT8, 512^3 x 1 donor) | 0.000897 TOPS | 1,836 (v6e) | 0.00005% | 1 | 2,047,064 | FAIL |

What the numbers actually say:

- **Every throughput row FAILs by 5–6 orders of magnitude**,
  as predicted — one 4-core CPU donor's INT8/FP8 matmul is
  ~10⁶× short of a TPU v6e's INT8 tensor cores, and local DRAM
  copy bandwidth is ~10³× short of HBM. Same shape as the vGPU's
  first numbers.
- **The one PASS is the SRAM capacity row, and it is a design
  constant (D1), not a timed rate:** 256 MiB per virtual device
  already exceeds Trainium2's 224 MiB SBUF, so one donor passes
  and the pooled aggregate is N × 256 MiB. Read it as "the
  capacity bar is cleared by construction", not as a speed win.
- **Perplexity (the question SW-code failed): FP8 costs little
  (+5.4%: 50.092 vs 47.539 FP32 on 252 TinyStories tokens,
  SmolLM-135M); INT8 costs +29.6% (61.610).** But neither saves
  wall-clock time on this fallback: INT8 generates at 0.45 tok/s
  vs 4.29 FP32 (0.10×), FP8 at 0.11 tok/s (0.026×). Full tables
  in docs/VNPU.md §5.3.
- **Why the quantized path is slow (measured, not assumed):**
  every vNPU matmul re-derives the dequantized staged weight and
  the full error-bound analysis from the staged payload — an
  O(weight-size) pass per call — and the transformer's per-row
  decode path issues ~210 such calls per token. FP32 beats
  INT8/FP8 on every kernel shape measured (1.4× at 512^3, 14×
  at lm_head m=1). The fix — caching the dequantized values +
  analysis per staged weight — is a named follow-up for the
  core-device workstream; until then the vNPU's INT8/FP8 win is
  bytes moved, not wall-clock time, same as SW-code.
- **Token identity (docs/VNPU.md §4.2 item 1): FP8 holds exactly**
  on both models (16/16 tokens); INT8 holds on demo-100m but
  diverges on real SmolLM-135M (first differing position 2,
  13 of 16 tokens differ). Quantized execution changes tokens
  at INT8 — recorded plainly, not hidden.
