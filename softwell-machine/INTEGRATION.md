# softwell-machine — integration notes

Read `src/contracts.ts` first. It is the single source of truth. Do not invent
parallel types — import from `../contracts.ts` (or the right relative path).

## Rules for every module

- Node built-ins only. No npm dependencies beyond the dev `typescript`.
- Erasable TypeScript only (Node 24 runs `.ts` directly via type stripping):
  no enums, no parameter properties, no namespaces, no `const enum`.
- Import with explicit `.ts` extensions (NodeNext), e.g.
  `import { ... } from "../contracts.ts"`.
- `npm run typecheck` must pass for your files. Run it before you finish.
- Write a small runnable smoke test under `test/` for your piece and run it
  with `node` (Node 24 strips types natively).

## File ownership (non-overlapping)

| Owner | Files |
|---|---|
| Model layer | `src/model/manifest.ts`, `src/model/loader.ts`, `src/model/hub.ts`, `src/model/tokenizer.ts` |
| Planner | `src/planner/planner.ts`, `src/planner/cost.ts` |
| Mesh | `src/mesh/registry.ts`, `src/mesh/protocol.ts`, `src/mesh/worker.ts` |
| Scheduler | `src/scheduler/batch.ts`, `src/scheduler/kvcache.ts` |
| Engine | `src/engine/backend.ts`, `src/engine/transformer.ts`, `src/engine/attention.ts`, `src/engine/sampling.ts` |
| API + CLI + dashboard | `src/api/server.ts`, `src/api/openai.ts`, `src/cli.ts`, `src/dashboard/index.html` |
| Demo | `demo/make-demo-model.ts`, `demo/run-demo.ts`, `demo/models/` (generated) |

## How the pieces connect (the data flow)

```
HF hub / local dir                demo/make-demo-model.ts
  │  (safetensors files)            │ writes config.json, tokenizer.json,
  ▼                                 │ model.safetensors (random weights)
src/model/loader.ts  ── WeightLoader ──► engine StageExecutor
src/model/hub.ts     ── resolve + download HF repo files
src/model/tokenizer.ts ── byte-level Tokenizer (vocab 256 + specials)

src/planner/planner.ts ── plan(manifest, workers) ──► ShardPlan
src/planner/cost.ts    ── memory-bandwidth decode model

src/mesh/registry.ts   ── WorkerRegistry: register/heartbeat/fault detect
src/mesh/protocol.ts   ── HTTP client+server helpers for mesh messages
src/mesh/worker.ts     ── worker process: registers, serves /mesh/execute,
                          owns a StageExecutor + worker-local PagedKVCache

src/scheduler/kvcache.ts ── PagedKVCache (block-paged, worker-local)
src/scheduler/batch.ts   ── ContinuousBatchingScheduler implements Scheduler:
                          submit/step; prefill vs decode routing by role;
                          drives stages in order via mesh execute calls.

src/engine/transformer.ts ── ReferenceTransformerBackend implements
                          EngineBackend; StageExecutor runs its layer range
                          (embed on stage 0, lm_head on last stage).
src/engine/attention.ts  ── GQA attention + RoPE + RMSNorm + SwiGLU + MoE
src/engine/sampling.ts   ── temperature / top-p / top-k / greedy sampler

src/api/server.ts  ── HTTP server: /v1/chat/completions (SSE streaming),
                      /v1/models, /v1/cluster (snapshot), / (dashboard)
src/api/openai.ts  ── request/response mapping + SSE chunk formatting
src/cli.ts         ── `softwell up` (orchestrator + N workers),
                      `softwell run <model>` (load + serve)
src/dashboard/index.html ── static UI polling /v1/cluster

demo/run-demo.ts ── spawns orchestrator + 2 workers as child processes,
                    loads the ~10M demo model, generates tokens, prints
                    throughput/latency. Must exit 0 on success.
```

## Generation loop (one decode step, orchestrator side)

1. Scheduler picks the batch (waiting -> prefill, running -> 1 decode token).
2. For a prefill: tokenize prompt, stage 0 embeds all prompt tokens, each
   stage forwards hidden states in order; the last stage returns logits for
   the final position; sample token 1.
3. For a decode step: each running sequence contributes its single last
   token; stages forward one position each; sample the next token.
4. Tokens stream back to API clients as SSE chunks; sequences finish on EOS
   or maxTokens.

## Worker protocol (HTTP JSON, see contracts.ts)

- `POST /mesh/register` { capability } -> { workerId, heartbeatIntervalMs }
- `POST /mesh/heartbeat` { workerId, ... } -> { ok, assignmentVersion }
- `GET /mesh/assignment?workerId=...` -> { version, assignment }
- `POST /mesh/execute` { workerId, seqId, kind, stage, input, positions }
  -> { output, isLogits, tokensProcessed }

The orchestrator is the source of truth for assignments; workers poll for a
new assignment version on heartbeat. On worker loss the registry marks it
dead and the planner re-plans over the survivors (re-sharding).

## Demo model

`demo/make-demo-model.ts` writes a ~10M-param random-weight Llama-style
model into `demo/models/demo-10m/`: `config.json`, `tokenizer.json`,
`model.safetensors` (single shard). Vocab is byte-level (256 tokens) plus
BOS/EOS/PAD. The demo proves the pipeline end to end; generated text will
be incoherent (random weights) — that is expected and must be stated in
the demo output and docs.

## Virtual hardware layer (the foundation — read this before the LLM runtime)

The machine itself is a VIRTUAL COMPUTER, and the VM REPLACES the
hardware. The virtual-hardware API in `src/contracts.ts` is the complete
hardware platform: the virtual CPU task ISA, the virtual GPU kernel/stream
model, the virtual memory hierarchy (one global address space), and
virtual storage. Workloads — including the LLM runtime — are written
against these interfaces ONLY, never against a physical device. Physical
machines are RESOURCE DONORS underneath: they contribute cycles, bytes,
and bandwidth to the pools. The mapping virtual-device -> donor lives
INSIDE each pool coordinator and its node components, never above the
boundary. If a donor dies, the pool heals itself through `handleNodeLoss`
and the platform above never sees it.

| Owner | Files |
|---|---|
| vRAM | `src/hw/vram.ts` |
| vGPU | `src/hw/vgpu.ts`, `src/hw/kernels-cpu.ts` |
| vCPU | `src/hw/vcpu.ts` |
| vSSD | `src/hw/vssd.ts` |
| Hypervisor | `src/hw/hypervisor.ts`, `src/hw/node-server.ts` |

- Each pool has a **coordinator** (orchestrator side, implements the
  `IVirtual*` interface from contracts.ts) and a **node component**
  (worker side, the donor adapter) exposing `routes():
  Record<string, RouteHandler>` under `/hw/<pool>/*`. One HTTP server
  per worker (see `node-server.ts`) mounts all four route tables on the
  worker's `hwPort` (WorkerCapability.hwPort).
- Coordinators discover donors through the injected `NodeProvider`
  (`listNodes(): WorkerInfo[]`) — the mesh registry in production, a
  fake in tests. They call node routes with plain `fetch`. No hw module
  imports a sibling hw module or the mesh/engine/scheduler.
- Binary payloads over the JSON routes travel as base64. That is fine
  at demo scale; a production transport would be binary/gRPC without
  changing the platform interfaces.
- The engine's raw math (attention.ts etc.) is the **CPU fallback
  kernel library** for vGPU: vGPU's `kernels-cpu.ts` implements the same
  ops in plain TypeScript; the engine's StageExecutor runs ON the VM in
  the integrated system (its matmuls become vGPU kernel launches, its
  KV-cache blocks are vRAM allocations, its weights stream from vSSD).
  The reference engine still works standalone for testing.
- The scheduler's `BlockPagedKVCache` blocks are the unit the engine
  allocates out of vRAM in the integrated system; in the reference
  demo the cache is worker-local and the scheduler only tracks counts.

## Virtual-machine boot order (CLI / demo)

1. Start orchestrator: mesh registry + hypervisor + pools + API server.
2. Start N workers: each runs the mesh worker (registers capability,
   serves /mesh/execute) AND the hw node server (serves /hw/* routes).
3. Hypervisor boots the VM: `hypervisor.boot()` returns a VirtualMachine.
4. Load a model: weights `putObject` on vSSD, stream ranges into vRAM
   allocations, vGPU reads tensors from vRAM by address.
5. Serve: scheduler drives generation; engine stages execute via vGPU
   kernels; vCPU handles tokenize/sample/detokenize tasks.
