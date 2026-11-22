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

Why these chips: Rubin R100 is the genuinely current best enterprise
GPU (in full production since early 2026); B200 is kept as the
secondary target because it is the widely deployed reference. EPYC
9965 is still the shipping core-count leader (EPYC 9006 "Venice" is
announced but has no shipping specs to benchmark against). Micron
9650 is still the fastest shipping data-center SSD.

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
- *No scaling curve was measured yet.* The run used `--donors 1`
  only, so `donorsToBeat` is a linear projection from one donor's
  rate, not a measured curve. The `--donors N` curve (which the
  harness supports) still needs a quiet-machine run to check
  whether per-donor rates hold as donors grow.

### What the numbers actually say

With one donor on this small VM, every substantive metric fails by
2–6 orders of magnitude — which is exactly what the honest math
predicts: one 4-core CPU donor's FP32 matmul is ~10⁶x short of a
Rubin's BF16 tensor cores, and loopback-HTTP page transfers are
~10⁴x short of HBM3e. The interesting columns are `donors-to-beat`
(capacity: 179 donors of 1 GiB; vSSD sequential read: 627 donors;
vCPU cores: 48 donors) and the scaling curve that still needs
measuring: those are the numbers that decide whether pooling can
ever close the gap, and at what donor count the network stops the
curve from being linear.

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
targets remains 6 orders of magnitude out. No scaling curve
(`--donors N`) has been measured yet.
