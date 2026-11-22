# softwell-machine — Architecture

Softwell is a **virtual computer**. The virtual machine *is* the hardware
platform: one global virtual address space (vRAM), one virtual GPU (vGPU),
a pool of virtual CPUs (vCPU), and one virtual object store (vSSD).
Workloads — including the LLM inference runtime — are written against the
virtual-hardware interfaces in `src/contracts.ts` only. They never touch a
physical device. Physical machines are **resource donors** underneath: they
contribute cycles, bytes, and bandwidth to the pools. The mapping
virtual-device → donor lives inside each pool coordinator and its
node components, never above the boundary.

A trillion-parameter model sees one giant machine: one giant GPU, terabytes
of vRAM, petabytes of vSSD. The donors needed to actually deliver that are
a separate question — see "The honest hardware math" below.

## Component map

```
src/contracts.ts          The platform. Every interface lives here.
src/model/                Weights in: safetensors manifest/loader,
                          Hugging Face hub download, byte-level tokenizer.
src/planner/              Shard planner (pipeline/tensor/expert parallel)
                          + cost model. cost.ts is where the hardware math lives.
src/mesh/                 Donor registry: register, heartbeat, fault
                          detection (sweep), stage assignments, /mesh/execute.
src/scheduler/            Continuous batching, paged KV cache (worker-local),
                          prefill/decode routing by worker role.
src/engine/               Reference transformer: GQA attention, RoPE,
                          SwiGLU, MoE routing, sampling. The math here is
                          also the vGPU's CPU fallback kernel library.
src/hw/vram.ts            vRAM: global address space, 4 MiB pages,
                          replication, version-based coherence, donor spill.
src/hw/vgpu.ts            vGPU: virtual devices, streams, kernel dispatch.
src/hw/kernels-cpu.ts     CPU fallback kernels (matmul, attention, ...).
src/hw/vcpu.ts            vCPU: the task ISA (kind = opcode), dispatch,
                          work stealing, at-least-once retries.
src/hw/vssd.ts            vSSD: chunked/replicated objects, streaming range
                          reads, 3-tier cache (hot LRU -> vRAM -> node disk).
src/hw/hypervisor.ts      Boots the VM, sweeps for dead donors, fans out
                          handleNodeLoss, stages models vSSD -> vRAM.
src/hw/node-server.ts     One HTTP server per donor mounting all /hw/* routes.
src/api/                  OpenAI-compatible /v1/chat/completions (SSE),
                          /v1/models, /v1/cluster, dashboard.
src/cli.ts                `softwell up` / `softwell run`: boots the
                          orchestrator + workers + VM, stages the model,
                          serves the API. Owns the re-shard-on-loss loop.
src/dashboard/            Cluster UI (polls /v1/cluster).
demo/                     ~10M-param random-weight model + end-to-end demo.
```

## How a request flows

1. **Boot.** `softwell run <model>` starts the orchestrator (mesh
   registry + hypervisor + API) and N worker processes. Each worker
   registers as a donor and serves both `/mesh/execute` (stage execution)
   and `/hw/*` (virtual-hardware donor routes) on its hwPort.
2. **Stage.** Model files are `putObject` onto vSSD, then
   `hypervisor.stageToVram()` streams the weights up into pooled vRAM
   (4 MiB pages, 2 replicas by default). The vGPU is told which
   allocation holds the weights (`registerAllocation`).
3. **Plan.** The planner cuts the model into pipeline stages balanced by
   *bytes* (MoE layers are heavier), one stage per worker, embedding on
   stage 0, lm_head on the last stage.
4. **Serve.** The scheduler admits requests, runs prefill for new
   sequences and one decode step per running sequence per step, chaining
   hidden states through the stages over `/mesh/execute`. The last stage
   returns logits; sampling happens per sequence; tokens stream back as
   SSE. Tokenize/detokenize/sample are also available as vCPU tasks
   (`tokenize`, `detokenize`, `sample`).
5. **Lose a donor.** The CLI sweep (2 s) marks missed heartbeats
   suspect (6 s) then dead (15 s), re-plans over the survivors, and
   re-assigns stages; the hypervisor sweep (5 s) heals the pools
   (vRAM/vSSD re-replicate, vGPU reassigns streams, vCPU requeues tasks).
   Sequences whose KV cache lived on the dead worker error out and must
   be resubmitted — their cache is gone with the donor.

## The honest hardware math (from `src/planner/cost.ts`)

Decode is **memory-bandwidth bound**: every generated token requires one
full read of the weights (plus KV cache), so

```
tokens/sec (batch 1) ≈ aggregate read bandwidth / weight bytes
```

| Model | Weights (BF16) | 1× H100 (~3 TB/s) | 64× H100 (~192 TB/s) |
|---|---|---|---|
| 10M demo | 39.7 MB | ~75k tok/s (toy) | — |
| 70B | 140 GB | ~21 tok/s | ~1,370 tok/s |
| 1T | **2.0 TB** | **~1.5 tok/s** | **~96 tok/s** |

Frontier speed — 100+ tok/s per sequence — needs ~200 TB/s of aggregate
read bandwidth, i.e. **~67 H100-class GPUs** (or equivalent) just for the
weights, before KV cache, network, and prefill. Batching helps aggregate
throughput (one weight read serves a whole batch) but not per-sequence
latency. KV cache for a 1T model is ~328 KB per token (80 layers), so
1M tokens of context is ~328 GB of vRAM on top of the weights. Prefill is
compute-bound instead: ~2 FLOPs per active parameter per token (~2×10¹²
FLOPs/token for a dense 1T model; MoE pays only for its routed experts).

What this means: **"frontier speeds" is a statement about the donor pool,
not about this code.** This machine makes a pool of donors look like one
computer; it cannot conjure bandwidth the donors do not have. On the two
CPU donors in this repo's demo, the same 10M model does ~7 tok/s — the
pipeline is real, the speed is whatever the hardware underneath is.

## What is real, and what is not yet

**Real, tested end to end (`npm test`, `node demo/run-demo.ts`):**
- The virtual machine: pooled vRAM with paging/replication/coherence,
  virtual GPU devices with streams and kernel dispatch, vCPU task pool
  with work stealing, chunked/replicated vSSD with tiered caching.
- The LLM runtime: safetensors loading (BF16/FP16/FP32/FP8-E4M3),
  HF hub download, a genuine transformer forward pass (GQA, RoPE,
  SwiGLU, MoE), continuous batching, paged KV cache, prefill/decode
  routing, OpenAI-compatible streaming API, dashboard.
- Fault tolerance: heartbeat sweep, re-shard on donor loss, pool healing,
  generation continuing on the survivor (proven in the demo).
- The demo model actually generates tokens across 2 workers.

**Reference-quality (works, but not production):**
- **Compute is CPU fallback.** Every kernel runs through the TypeScript
  `kernels-cpu.ts` math. There is no CUDA/Metal backend; `GpuNode`
  reports the device that *actually* executed (`cpu`), never a GPU it
  does not have. A real device backend plugs in at `GpuNode.handleKernel`.
- **Transport is JSON + base64 over HTTP.** Hidden states, pages, and
  chunks travel as JSON. Fine at demo scale; a production transport is
  binary/gRPC and would not change the platform interfaces.
- **Coherence is single-writer + version invalidation.** No distributed
  transactions, no multi-writer conflict resolution.
- **vCPU delivery is at-least-once.** Tasks retried after donor loss may
  run twice; handlers must be idempotent.
- **No auth, no TLS.** Every endpoint trusts localhost. Do not expose
  these ports to a network you do not control.
- **The vSSD manifest is coordinator-local** (persisted to a JSON file,
  best-effort). Losing the orchestrator loses the object index even
  though the chunks survive on donors.
- **Workers load stage weights from a local model directory.** The
  vSSD → vRAM staging path is implemented and proven, but the stage
  executor's `WeightLoader` still reads local disk. The production path
  is a `VramWeightLoader` (weights served from pooled vRAM) — the
  interface for it already exists (`WeightLoader`); it is not written yet.
- **Stage execution bypasses the vGPU coordinator.** The mesh worker
  runs the engine's math in-process — which *is* the vGPU's donor-side
  kernel execution (the same `CpuKernels` code) — but the per-op
  `vGPU.launch()` path is not the hot loop. Routing every matmul through
  the coordinator is the identified next step; it needs `StageExecutor`
  to become async and is documented as such in the code, not done here.
- **The scheduler's KV cache is worker-local**, not allocated out of
  pooled vRAM. The block size matches the vRAM page model, but the
  integration (cache blocks as vRAM allocations) is future work.

## Mapping onto the existing softwell repo

The existing softwell repo (Doughboyfressh/softwell) is a TypeScript web
app (Vite + TanStack Start + React 19 + Tailwind, Postgres via Pglite,
Vercel deploy). This machine is a **new backend service**, not a change
to that app — no repo files were touched or copied to build it.

The intended fit:
- `softwell-machine` runs as a service (`softwell run <model>`) wherever
  the donors live; the existing web app becomes the console for it,
  calling the OpenAI-compatible API (`/v1/chat/completions`,
  `/v1/models`) and the cluster API (`/v1/cluster`) that this repo
  already serves — the dashboard in `src/dashboard/` is the reference
  version of that UI.
- The repo's tagline, "Pi's software GPU. The program is the chip," is
  now literal: the program *is* the virtual GPU, CPU, RAM, and SSD.
- A trillion-parameter model is served by pointing the machine at enough
  donors; the web app never sees the difference between 2 donors and 200.

## Running it

See `README.md`. The short version:

```sh
npm install
npm test                 # all 11 smoke-test files
npm run make-demo-model  # ~10M random-weight demo model (once)
npm run demo             # boots the VM, serves, kills a worker, proves it
node src/cli.ts run demo/models/demo-10m   # serve it yourself
```
