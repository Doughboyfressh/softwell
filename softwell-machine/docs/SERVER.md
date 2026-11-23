# The softwell virtual server

The virtual server is the serving plane that runs **on** the virtual
machine — it is a workload of the VM, not a host-side sidecar. It boots
with the VM, reserves its own slice of virtual hardware from the
hypervisor, and serves models using only virtual hardware. The code
lives in `src/server/` (`VirtualServer`, `ModelRegistry`) and `src/api/`
(the HTTP layer); the CLI in `src/cli.ts` only wires it together.

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │  virtual machine (Hypervisor over donors)    │
                │  vRAM · vGPU · vCPU · vSSD                    │
                │                                              │
                │  ┌────────────────────────────────────────┐  │
                │  │  VirtualServer (a VM workload)         │  │
                │  │  - 256 MiB vRAM for its own buffers    │  │
                │  │  - one vGPU stream                     │  │
                │  │  - tokenize/detokenize handlers on vCPU│  │
                │  │  - ModelRegistry + one ServingContext  │  │
                │  └────────────────────────────────────────┘  │
                │        │ load / activate (factory)           │
                │        ▼                                     │
                │  scheduler ──mesh──▶ workers (donors)        │
                └──────────────────────────────────────────────┘
```

### What the server reserves from the hypervisor

At `boot(vm)` the server reserves, through the VM interfaces only:

- **vRAM**: `serverMemoryBytes` (default 256 MiB) for its own buffers,
  freed again on `close()`. Model weights are separate allocations
  owned by the registry (see below).
- **vGPU**: one stream (`createStream()`), used for the server's own
  GPU work.
- **vCPU**: `tokenize` / `detokenize` task handlers, which encode/decode
  through the *active* model's tokenizer. With no active model they
  throw — tokenization is a property of the active model, not of the
  server.

The server never touches a physical device, a donor, or a worker
directly. Execution runs on the mesh workers, which are the VM's
donors; the CLI's factory builds the scheduler that drives them.

### How models flow: vSSD → vRAM → workers

1. **Load** (`ModelRegistry.load(ref)`). A reference is a local
   directory or a Hugging Face repo id. The registry reads the manifest
   and tokenizer, then stages **every file** of the model directory to
   vSSD under `models/<id>/…`, and stages each `*.safetensors` shard
   into vRAM (one allocation per shard) via `hypervisor.stageToVram`.
   The entry records `vramBytes` and the first shard's `weightVaddr`.
2. **Fit check, before staging anything.** Weights + 512 MiB of KV-cache
   headroom must fit in total vRAM. A model that does not fit is **not**
   an exception: the entry is returned with status `"error"` and a
   message naming the weights, the headroom and the vRAM total, and no
   allocation is left behind. Activating it is refused (see below).
3. **Activate** (`VirtualServer.activateModel(id)`). The entry must
   exist and have status `"ready"`. The CLI's factory builds the serving
   context for it: it reads the manifest, plans the model over the live
   workers (throwing the planner's reason when no feasible plan exists),
   stamps `modelId` + `modelDir` onto every assignment, installs the
   plan in the mesh registry, and builds a `ContinuousBatchingScheduler`
   (mesh transport, `sampleLogits`, tokenizer decode, the configured
   decoding mode) wrapped in the instrumented scheduler that feeds
   `/v1/cluster`. The new context swaps in and the previous one is
   disposed.
4. **Serve.** Requests are admitted against the active model's
   scheduler, run prefill + decode across the pipeline stages on the
   workers, and stream back as SSE or JSON. Workers load their stage
   weights from pooled vRAM through the orchestrator's binary weight
   endpoint (`GET /mesh/weights`); no weight bytes come from disk at
   inference time. (The model directory is still read once at ingest —
   manifest, tokenizer, staging to vSSD — and a fresh worker process
   still needs a `--model-dir` to boot. See `docs/VRAM-WEIGHTS.md`.)
5. **Unload** (`ModelRegistry.unload(id)`). Refuses the active model
   and a model still loading. Frees every vRAM allocation and leaves
   the model `"available"` — the vSSD copy and the local directory stay,
   so it can be loaded again without re-downloading or re-staging
   from disk.

### One active model at a time — and why

The registry may stage several models, but only one model is **active**
per cluster. The reasons are structural, not policy:

- Each worker holds **one weight loader per active model** (a
  `VramWeightLoader` fetching tensors from pooled vRAM for a staged
  model; the spawn `--model-dir` disk loader is only the starting
  point) and one stage executor; switching models means reloading
  every worker's stage.
- The KV cache on the workers belongs to the active model's geometry
  (layers, heads, head dim). A second model would need a second cache
  and a second plan over the same workers.
- The scheduler, tokenizer and the vCPU tokenize/detokenize handlers
  all belong to one serving context.

Activating another model builds its context, swaps it in, and disposes
the previous one. In-flight requests on the old model error out when
their scheduler is disposed — the API layer surfaces those as 500s.
Clients should treat a model switch as a restart of serving, not as a
seamless handover.

## How to run

All commands run from the repository root.

### `softwell up` — boot the cluster and wait

```sh
# Cluster + server, no model active yet.
softwell up

# Cluster + server, with a model loaded and activated at boot.
softwell up --model-dir demo/models/demo-100m

# Same, with 1 worker and explicit ports.
softwell up --model-dir demo/models/demo-100m --workers 1 --port 8410 --api-port 8420
```

`up` boots the WorkerRegistry + mesh server, spawns N workers, waits
for them to become healthy, boots the hypervisor (vRAM / vGPU / vCPU /
vSSD over the workers), then boots the VirtualServer on the VM and
starts the API + dashboard. It writes `data/softwell.pid` (this
process's pid) once the server is up, and removes it on shutdown.

With `--model-dir` the model is loaded (staged vSSD → vRAM) and
activated at boot. Without it the server starts in state `ready` with
no active model: the dashboard shows "no model active", `/v1/models`
is empty, and chat-completion requests are rejected with 429 until a
model is active. A model can be loaded into the running server with
`POST /v1/models/load` followed by `POST /v1/models/:id/activate`
(see [Model management](#model-management)) — no restart needed.

Workers open one weight loader from the `--model-dir` they are spawned
with, so `up` without `--model-dir` still starts workers on the demo
model directory (`demo/models/demo-10m`) when it exists. Activating a
different model re-plans the workers and each worker loads the new
model's weights from pooled vRAM on demand — it asks the orchestrator
for the staged weight map and builds a `VramWeightLoader` (see
`src/mesh/worker.ts` `loaderFor`); the disk loader is used only when
the orchestrator has not staged the model. The spawn directory is only
the starting point, and a fresh worker process still needs one to boot.

### `softwell run` / `softwell serve` — boot, load, serve

```sh
softwell run demo/models/demo-100m
softwell serve demo/models/demo-100m --workers 1 --port 18410 --api-port 18420
softwell run meta-llama/Llama-3.3-70B-Instruct   # HF repo id: downloaded first
```

`run` and `serve` are the same command under two names: boot the
cluster, load `<model>` (a local directory, or a Hugging Face repo id,
downloaded into `models/<owner>__<name>` first), activate it, and serve
until stopped (Ctrl-C / SIGTERM / `POST /v1/server/shutdown`). The demo
model is used when no model is given and it exists — for `run`/`serve`
a `<model>` argument is required.

### `softwell models` — what a running server knows

```sh
softwell models
softwell models --api-port 8420
```

An HTTP client; it does not boot anything. It prints a table from
`GET /v1/models` + `GET /v1/server/status`:

```
id                      status      params (M)  weight (MB)  vRAM (MB)  active
demo-100m               ready       104.9       398.1        398.1      *
active model: demo-100m (state: ready, uptime: 42s)
```

`*` marks the active model. If no server is reachable on the port it
prints `no softwell server reachable on port <A> — start one with
'softwell up'` and exits 1.

### `softwell stop` — shut a running server down

```sh
softwell stop
softwell stop --api-port 8420
```

Reads `data/softwell.pid`; if that pid is alive it SIGTERMs it and
waits up to 10 s for exit (SIGKILL after that), removes the pidfile and
reports stopped. If there is no pidfile, or the pid is dead, it falls
back to `POST /v1/server/shutdown` on the API port. Exit 0 when
something was stopped, 1 with a clear message when nothing was running.

### Flags

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--workers N` | 2 | worker processes to spawn |
| `--port P` | 8410 | orchestrator mesh port |
| `--api-port A` | 8420 | serving API + dashboard port |
| `--model-dir <dir>` | — | model directory (`up` only; `run`/`serve` take it positionally) |
| `--memory <bytes>` | — | per-worker memory budget (`SOFTWELL_WORKER_MEMORY_BYTES`) |
| `--attention exact\|landmark\|sparse` | exact | attention path (Theorems A1/A2) |
| `--weights fp32\|sw2\|sw3\|sw4` | fp32 | weight format (SW-code, Theorem N1) |
| `--kv fp32\|sw2\|sw3\|sw4` | fp32 | KV-cache format (Theorem N2) |
| `--decoding autoregressive\|speculative\|jacobi` | autoregressive | token generation (Theorems S1/S2, greedy only) |
| `--draft-length K` | 4 | draft length / Jacobi block size |

Engine flags go to the workers (attention/weights/KV) and to the
scheduler (decoding). See `docs/RESEARCH.md` for what the novel paths
do — and for the honest warning that on this CPU backend they save
bytes, not wall-clock time.

## API reference

Base URL: `http://127.0.0.1:<api-port>` (default 8420). All responses
are JSON unless noted. The dashboard is served at `/`.

### `GET /v1/models`

```sh
curl -s http://127.0.0.1:8420/v1/models
```

```json
{ "object": "list", "data": [{ "id": "demo-100m", "object": "model", "created": 1793000000, "owned_by": "softwell" }] }
```

Lists the models the registry has **loaded** (available-but-unloaded
models are not listed here; see `GET /v1/server/status`).

### `GET /v1/models/:id`

```sh
curl -s http://127.0.0.1:8420/v1/models/demo-100m
```

One model, or `404 { "error": { "message": "model not found: …" } }`.

### `POST /v1/models/load`

Stage a model into the running server. Body: `{ "ref": "<local
directory | model id | Hugging Face repo id>" }`. Returns the
`ModelEntry` (`200`, status `"ready"`). A model that does not fit in
vRAM returns `422` with the entry at status `"error"` and the reason
in `error`; an unknown reference is `404`, a malformed `ref` is `400`.

```sh
curl -s -X POST http://127.0.0.1:8420/v1/models/load \
  -H 'content-type: application/json' \
  -d '{"ref": "demo/models/demo-100m"}'
```

### `POST /v1/models/:id/activate`

Make a loaded (`"ready"`) model the active one. Workers are
re-planned onto it and fetch its weights from vRAM on demand; the previous
model's context is disposed. Returns the model entry (`200`).
Unknown id is `404`, a model that is not ready is `409`.

```sh
curl -s -X POST http://127.0.0.1:8420/v1/models/demo-100m/activate
```

### `POST /v1/models/:id/unload`

Free a staged model's vRAM (its vSSD copy and local directory are
kept). Returns `{ "id", "status": "available" }` (`200`). The active
model cannot be unloaded (`409` — activate another model first);
unknown id is `404`.

```sh
curl -s -X POST http://127.0.0.1:8420/v1/models/demo-10m/unload
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions, JSON or SSE streaming. The `model`
must be the active model; anything else (or no active model) is
rejected with 429.

```sh
# JSON (one response at the end)
curl -s http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "demo-100m", "messages": [{"role": "user", "content": "Hello world"}], "max_tokens": 8, "temperature": 0}'

# Streaming (SSE: one data: event per token, then a finish chunk, then data: [DONE])
curl -sN http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "demo-100m", "messages": [{"role": "user", "content": "Hello world"}], "max_tokens": 8, "temperature": 0, "stream": true}'
```

Streaming chunks carry `choices[0].delta.content` and a final chunk
with `finish_reason`; there is **no `usage` object in the stream**.
The non-streaming response includes
`usage: { prompt_tokens, completion_tokens, total_tokens }` and
`choices[0].finish_reason` (`"stop"`, `"length"`, `"eos"` or `"error"`).
`temperature`, `top_p`, `top_k`, `stop` and `seed` are honoured as
documented in `src/contracts.ts`; novel decoding modes apply to greedy
sequences only.

### `POST /v1/completions`

Plain-text completions (`{ "model", "prompt", … }`), JSON or SSE, same
admission and streaming rules as chat completions.

### `GET /v1/cluster`

The live cluster snapshot: active `modelId`, shard `plan`, `workers`,
sequence counts (`waiting` / `running` / `done`), measured
`tokensPerSec`, `avgLatencyMs`, `uptimeSec` and the pooled `vm` view
(vRAM / vGPU / vCPU / vSSD). This is what the dashboard's lower panels
poll.

```sh
curl -s http://127.0.0.1:8420/v1/cluster | head -c 600
```

### `GET /v1/server/status`

```sh
curl -s http://127.0.0.1:8420/v1/server/status
```

```json
{
  "state": "ready",
  "uptimeSec": 42.1,
  "activeModelId": "demo-100m",
  "models": [ { "id": "demo-100m", "status": "ready", "params": 104857600,
                "weightBytes": 417473018, "vramBytes": 417473018,
                "requestsServed": 3, "tokensGenerated": 24 } ],
  "requests": { "active": 0, "queued": 0, "completed": 3, "rejected": 0, "cancelled": 0 },
  "tokensPerSec": 4.2,
  "queueDepth": 0,
  "vm": { "vram": { "totalBytes": 8589934592, "usedBytes": 675282944, "...": "..." } },
  "recentRequests": [ { "id": "…", "modelId": "demo-100m", "status": "done",
                        "promptTokens": 12, "completionTokens": 8, "ttftMs": 812 } ]
}
```

`state` is `starting` / `ready` / `degraded` / `stopping`.
`tokensPerSec` is an exponential moving average over the decode rates
of **completed** requests only, and stays 0 until a request completes —
it is measured, never a cost-model estimate. `models` is the registry's
full entry list, including `available` (unloaded) and `error` entries.

### `GET /healthz` and `GET /readyz`

```sh
curl -s http://127.0.0.1:8420/healthz
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8420/readyz
```

- `/healthz` → `200 { status: "ok"|"degraded", checks: { vmBooted,
  activeModel, workersHealthy } }`, or `503` when `status` is `"down"`
  (VM not booted).
- `/readyz` → `200 { "ready": true }` when a model is active **and**
  workers are healthy, else `503 { "ready": false }`. This is the
  readiness signal to poll after starting a server.

### `POST /v1/server/shutdown`

```sh
curl -s -X POST http://127.0.0.1:8420/v1/server/shutdown \
  -H 'content-type: application/json' -d '{}'
```

Returns `202` and triggers the graceful shutdown: state becomes
`stopping`, the CLI's shutdown runs (hypervisor shutdown, API close,
mesh close, workers SIGTERMed, pidfile removed) and the process exits
0. This is what `softwell stop` falls back to, and what the e2e test
uses to stop the child it spawned.

## Model management

Models are managed through `VirtualServer` (`loadModel` /
`unloadModel` / `activateModel`) and `ModelRegistry` (`load` /
`unload`), reachable over HTTP as documented in the
[API reference](#api-reference) (`POST /v1/models/load`,
`POST /v1/models/:id/activate`, `POST /v1/models/:id/unload`) as well
as at boot time through the CLI (`up --model-dir`, `run`, `serve`).

- **Load.** `softwell up --model-dir <dir>` or
  `softwell serve <ref>`. `<ref>` is a local directory or an HF repo
  id (`owner/repo`); a hub repo is downloaded into
  `models/<owner>__<name>` (skipping files already present at the
  expected size) and needs `HF_TOKEN` in the environment for gated
  repos. Loading is idempotent: loading a model that is already `ready`
  returns the same entry.
- **Fit errors.** When weights + 512 MiB KV headroom exceed total
  vRAM, load succeeds with the entry at status `"error"` and a message
  like `model "…" does not fit in vRAM: weights … + 524288000 bytes KV
  headroom = … exceeds vRAM total …`. The model is visible in
  `/v1/server/status` (and `softwell models`) with that status, and
  activating it is refused with `model … is not ready (status: error)`.
  The fix is a smaller model or more donors (more pooled vRAM).
- **Activate.** Only a `ready` entry can be activated. Activation
  re-plans over the live workers; if the plan is infeasible the
  planner's reason is thrown and the previous active model (if any)
  stays active.
- **Unload.** Refuses the active model (`activate another model
  first`) and a model still loading. Frees the vRAM allocations and
  returns the entry to `"available"`; the vSSD copy and local directory
  stay, so a later load is cheap. Unloading an unknown id throws
  `unknown model: …`.
- **Tokenizer.** The registry records `tokenizerKind`: `"bpe"` when the
  model's `tokenizer.json` parses into a usable BPE model, otherwise
  `"byte-level"` — the fallback tokenizer, where every UTF-8 byte is a
  token (ids 0–255, plus BOS 256 / EOS 257 / PAD 258) and any text
  round-trips exactly. The fallback keeps any model servable, but it
  does **not** match the tokenization a BPE-trained model was trained
  with: prompts are encoded as raw bytes, so generation quality and
  token counts will differ from the same model served with its real
  tokenizer. The demo models are byte-level by construction.

## The dashboard

`http://127.0.0.1:<api-port>/` serves `src/dashboard/index.html`
(plain HTML/JS, no dependencies). It polls two endpoints every 2 s:

- `/v1/server/status` — the Server panel (state + uptime, active model,
  live request counts, tokens/sec, queue depth, vRAM used/total, vGPU
  devices + total TFLOP/s, vCPU cores) and the Models table (id, status,
  params, vRAM, requests served; `★` marks the active model). When no
  model is active it says so explicitly.
- `/v1/cluster` — throughput, the virtual-machine panel, the workers
  table and the shard-placement table (unchanged from before).

The "Try it" box sends a non-streaming chat completion to the active
model and is disabled with an explanation while no model is active.

## Fault tolerance

Every 2 s the CLI sweeps worker heartbeats (`suspect` after 6 s
silent, `dead` after 15 s). On worker loss it re-plans the **active**
model's manifest over the survivors (keeping stage boundaries where
they still fit), stamps `modelId`/`modelDir` on the new assignments,
installs the plan and calls `scheduler.updatePlan`. Sequences that
were running on the lost worker error on their next step and must be
resubmitted — their KV cache is gone with the worker. With
`--kv-store vram` the flushed KV *bytes* survive in pooled vRAM, but
the chunk table (which allocIds belong to which seq/layer) dies with
the worker, so the replacement still cannot resume the sequence — it
rejects loudly and the sequence is resubmitted (proven end-to-end by
`test/kvcache-vram-survival.test.ts`). A pure donor loss, by contrast,
does not kill the sequence: flushed KV fails over to the surviving
replica and generation continues. With no model
active there is nothing to re-plan; a later activation plans over
whatever workers are alive. If all workers are lost the server waits
for workers to rejoin.

## Limits

Honest limits of this version. See also `docs/RESEARCH.md` and
`docs/BENCHMARKS.md`.

- **One active model per cluster.** See "One active model at a time"
  above. Switching models disposes the old serving context; requests
  already in flight keep the old context's scheduler and are not
  migrated to the new model.
- **Model management is HTTP + boot-time.** Load / unload / activate
  are exposed over HTTP (`POST /v1/models/load`,
  `POST /v1/models/:id/activate`, `POST /v1/models/:id/unload`) and
  at boot (`up --model-dir`, `run`, `serve`). `up` without a model
  starts a healthy server with nothing to serve until one is loaded.
- **Workers fetch weights from pooled vRAM, not from a shared
  filesystem.** At load time the model is staged to vSSD and then to
  vRAM (`ModelRegistry.load`); at activation each worker builds a
  `VramWeightLoader` from the orchestrator's weight map and fetches
  tensor bytes over the mesh (`GET /mesh/weights`). A worker no longer
  needs to see the model directory to serve a staged model — the
  directory is read once at ingest, and only a fresh worker process
  still needs some `--model-dir` to boot (its spawn-time loader). The
  disk loader remains only as a fallback for models the orchestrator
  has not staged. See `docs/VRAM-WEIGHTS.md` for the proof.
- **The CPU backend is compute-bound, so the novel byte-saving paths
  save bytes, not wall-clock time.** Measured on this backend
  (`docs/RESEARCH.md`): speculative decoding 0.91× and Jacobi 0.53×
  wall-clock vs autoregressive (β ≈ 1), and SW-code weights run slower
  in-process because unpacking costs more than the bytes saved. The
  wins — landmark attention up to 20.41× at n=4096, SW-code cutting
  bytes per token 3.6–6.6× vs BF16 — are real, but they pay off on
  bandwidth-bound hardware, not here. Do not quote throughput numbers
  from the cost model as measured performance; `/v1/server/status`
  reports the measured EMA.
- **The byte-level fallback tokenizer** (above) keeps models servable
  without a usable BPE model, at the cost of tokenization fidelity.
- **Request timeouts / backpressure defaults.** New requests are
  rejected with 429 once waiting + running sequences reach `maxQueue`
  (default 32), when no model is active, or when the requested model
  is not the active one. Each request has a 300 s deadline
  (`requestTimeoutMs`, default 300_000) after which it is cancelled.
  The scheduler admits at most 16 concurrent sequences (its default).
  Rejected, cancelled and completed counts are visible in
  `/v1/server/status` and on the dashboard.
- **Demo weights are random.** The demo models generate incoherent
  text by design; they prove the pipeline (staging, planning,
  streaming, re-sharding), not the weights.
- **Single-op latency is the tradeoff.** "Better than enterprise
  hardware" is defined as pooled aggregate throughput/capacity; a
  single virtual op still pays mesh + HTTP round trips.
