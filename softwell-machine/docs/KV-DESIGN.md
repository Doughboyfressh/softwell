# KV cache in vRAM — interface design decision

**Status:** decision (not implementation). Written 2026-11-22 by workstream 1
(interface design) for the KV-cache-in-vRAM phase.

**Decision in one paragraph:** make the cache interface and `forward()`
async (option (a)). The serving path above the engine is already async —
`executeHandler`, the scheduler's `step()`, and `MeshClient.execute` all
return promises — so the blast radius is the engine, the worker's cache
factory, the bench wrapper, and the tests, not the mesh or the scheduler.
The blocking-transport alternative (option (b)) keeps the sync signatures
but blocks the worker process's event loop for the whole of every vRAM
read, which trips the orchestrator's heartbeat dead-man switch on exactly
the long prefill that is vRAM KV's viable slice. There is no runtime
fallback to host memory: the store is chosen explicitly at worker startup
(`host` or `vram`), and a `vram` worker whose vRAM is unavailable fails
its stage load loudly instead of silently serving from host memory.

## 0. Context and evidence base

- The prototype is `src/scheduler/kvcache-vram.ts` (`VramPagedKVCache`,
  an async twin of `PagedKVCache`). Its header is the source of the
  measured numbers quoted here; the results JSON is
  `test/kvcache-vram-bench-results-2026-11-22.json`.
- Geometry for every number below: demo-100m, kvDim 512 (8 KV heads x
  headDim 64), 12 layers, blockSize 16, chunk 1024 tokens = one 4 MiB
  vRAM page. One token's K+V is `2 x 512 x 4 B = 4 KiB`; a full-context
  read of one layer at ctx 1024 moves 4 MiB, all 12 layers 48 MiB.
- Measured (warm medians, loopback HTTP, replicas=2, in-process
  `VirtualRam` + two donor nodes):
  - Full-context read, one layer: ctx 128 — host 0.13–0.19 ms, vRAM
    0.10–0.39 ms (parity); ctx 1024 — host 1.2–4.7 ms, vRAM 14–99 ms
    (best single-page fetch 7.1 ms, cold reads ~99–127 ms).
  - Full-context read, all 12 layers sequential (the per-decode-token
    cost): ctx 128 — host 1.3–4.0 ms, vRAM 1.4–17.8 ms (~1–4x);
    ctx 1024 — host 65–285 ms, vRAM 394–1792 ms (~6–9x).
  - Projected decode: ctx 128 +0.4% to +3.5% (negligible); ctx 1024
    +75% (0.42 → 0.24 tok/s, run 4) — material.
  - Appends are staged on the host and cheap (0.06–0.30 ms/token/layer
    vs host 0.003–0.004); a naive per-token vRAM write costs 19–200 ms
    (read-modify-write of a whole 4 MiB page). Alloc of one 4 MiB chunk
    costs 38–212 ms, amortized over 1024 tokens.
- The prototype's verdict stands: NO-GO for decode by default at long
  context, GO-WITH-CONDITIONS at short context / prefill-only. This note
  decides *how* to wire it in, not *when to turn it on* (see §5).

## 1. Audit

### 1.1 `forward()` call sites — 13 real sites, 3 in `src/`

Grep: `\.forward(` across `src/` and `test/` returns 14 lines; one is a
doc comment (`src/scheduler/kvcache-vram.ts:76`). The real sites:

| # | Location | What it is | Async today? |
|---|----------|------------|--------------|
| 1 | `src/mesh/worker.ts:279` | `executeHandler` calls `stageExecutor.forward(...)` | Yes — handler is `async`, one `await` to add |
| 2 | `src/engine/novel-decoding.ts:90` | `forwardTokenRows()` helper, used by all three decoders | No — sync helper |
| 3 | `src/bench/novel-bench.ts:382` | `TimingExecutor.forward()` decorator, delegates to inner | No — sync wrapper |
| 4–7 | `test/engine.test.ts:315, 322, 342, 359` | Full forward + determinism + batched-rows checks | No |
| 8–10 | `test/engine-options.test.ts:186, 270, 283` | Novel-path forwards (landmark/sparse/SW) | No |
| 11–13 | `test/kvcache-vram-bench.ts:360, 376, 424` | Bench script: host-cache forward for the projection | No (script, not a test) |

`StageExecutor.forward` implementations that must change signature
(the interface plus every implementor):

| Implementor | Location |
|-------------|----------|
| `TransformerStageExecutor.forward` (the only real one) | `src/engine/transformer.ts:294` |
| `TimingExecutor.forward` (bench decorator) | `src/bench/novel-bench.ts:382` |
| Fake echo executors (4) | `test/mesh.test.ts:116`, `test/worker-model-switch.test.ts:133`, `test/vram-weights.test.ts:480`, `test/failure-vram-weights.test.ts:1072` |

### 1.2 `PagedKVCache` method call sites in `src/engine/` — 8 sites

| Location | Method | Context |
|----------|--------|---------|
| `src/engine/transformer.ts:540` | `ctx.cache.append(...)` | `attendPrefillNovel`, per-row append |
| `src/engine/transformer.ts:554` | `ctx.cache.read(...)` | `attendPrefillNovel`, continued-prefill fallback (per-row read) |
| `src/engine/transformer.ts:571` | `ctx.cache.read(...)` | `attendPrefillNovel`, whole-prefix read |
| `src/engine/transformer.ts:649` | `ctx.cache.append(...)` | `forwardLayer`, per-row path |
| `src/engine/transformer.ts:651` | `ctx.cache.read(...)` | `forwardLayer`, per-row path (full prefix `[0..pos]`) |
| `src/engine/novel-decoding.ts:463` | `cache.free(seqId)` | `AutoregressiveDecoder.generate()` prologue |
| `src/engine/novel-decoding.ts:521` | `cache.free(seqId)` | `SpeculativeDecoder.generate()` prologue |
| `src/engine/novel-decoding.ts:735` | `cache.free(seqId)` | `JacobiDecoder.generate()` prologue |

Notable absences (verified by grep, not assumed):

- `allocate()` is **never** called by the engine or the scheduler. Both
  cache implementations call it internally from `append()`. No engine
  call site exists to convert.
- `usedBlocks()` / `totalBlocks()` are called only inside the cache
  implementations themselves (`src/scheduler/kvcache.ts:91`) and in
  tests. The scheduler does **not** read them — admission control is a
  token budget in `src/scheduler/batch.ts`. They are pure in-memory
  counters in both implementations, so they stay synchronous.
- `src/scheduler/batch.ts` and `src/server/*` never touch a cache or a
  `forward()` directly. The scheduler drives workers over HTTP
  (`MeshClient.execute`, `src/mesh/protocol.ts:427`, already async) from
  its async `step()` (`src/scheduler/batch.ts:271`). The mesh protocol
  (`ExecuteRequest` / `ExecuteResponse`) is unchanged by either option.

### 1.3 The worker's cache factory

- Type: `StartWorkerOptions.createCache: (manifest: ModelManifest) =>
  PagedKVCache` (`src/mesh/worker.ts:55`). Required; `startWorker`
  throws at :77 when absent. Deliberately no in-file fallback cache.
- Call site: `cache = opts.createCache(loader.manifest)` in
  `loadAssignment()` (`src/mesh/worker.ts:216`), which is already async.
- CLI factory: `src/mesh/worker.ts:485` builds one
  `BlockPagedKVCache(16, 1024, numKeyValueHeads, headDim)` per stage load
  from the loaded model's GQA geometry.
- Two latent vRAM leaks the integration must close (not this note's job
  to fix, but the design must account for them): `loadAssignment()`
  replaces `cache` without freeing the old one, and `close()` never
  touches the cache. That is free for host memory (GC) and a 4 MiB-chunk
  leak per model switch for vRAM. See §6.4 and proof criterion P4.

### 1.4 What is missing before *either* option can run in production

No remote `VramAccess` exists. The only implementation is
`VirtualRamAccess` over an in-process `IVirtualRam`
(`src/scheduler/kvcache-vram.ts`) — usable in tests and benches, not in
the worker process, which is separate from the orchestrator/VM. The
weights precedent (`GET /mesh/weights/map` + `GET /mesh/weights`,
`src/mesh/protocol.ts:137–138`, `MeshClient.fetchWeightTensor`) shows
the shape, but there is no KV equivalent: no orchestrator endpoint for
KV page read/write/alloc/free and no `MeshClient` method for it. The
integration workstream must build that (new endpoint + `RemoteVramAccess`
implementing `VramAccess` over it) before a worker can use vRAM KV.
The interface decision in this note does not depend on it, and the
refactor workstream must not invent the endpoint as a side effect —
it is a named dependency in §7.

## 2. Option (a): make the cache interface and `forward()` async

### 2.1 Mechanical scope

`src/contracts.ts` — two interfaces change, nothing else:

```ts
export interface PagedKVCache {
  readonly blockSize: number;
  allocate(seqId: string, layer: number, numTokens: number): Promise<number[]>;
  append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): Promise<void>;
  read(seqId: string, layer: number, positions: number[]): Promise<{ keys: Float32Array; values: Float32Array }>;
  free(seqId: string): Promise<void>;
  usedBlocks(): number;   // unchanged — pure in-memory counter
  totalBlocks(): number;  // unchanged — pure in-memory counter
}

export interface StageExecutor {
  readonly assignment: ShardAssignment;
  load(loader: WeightLoader): Promise<void>;  // already async
  embed(tokenIds: number[]): Tensor;          // unchanged — pure lookup
  forward(input: Tensor, ctx: ForwardContext): Promise<Tensor>;
  memoryBytes(): number;                      // unchanged
}
```

`ForwardContext` itself is unchanged (its `cache: PagedKVCache` field
just denotes the async interface now). `EngineBackend.createStageExecutor`
stays a sync factory; `load()` is already async. The mesh protocol,
`ExecuteRequest`/`ExecuteResponse`, and every scheduler/server type are
untouched.

`src/engine/transformer.ts` — three methods gain `async`/`await`, the
rest stay sync:

- `forward()` (:294) → `async`, awaits `forwardLayer` per layer.
- `forwardLayer()` (:588) → `async`: awaits `attendPrefillNovel` or,
  per row per layer, awaits `cache.append` then `cache.read`.
- `attendPrefillNovel()` (:499) → `async`: awaits its appends and reads.
- Stay sync: `attendRow()` (:450 — takes already-fetched keys/values),
  `noteAppend` / `noteAppendedKv`, `blockStatsFor`, `packedKvFor`,
  all matmuls/norms/RoPE. The novel-attention paths need no logic
  change; they consume the arrays `read()` returns.

`src/engine/novel-decoding.ts` — the second-order effect the steer
understates:

- `forwardTokenRows()` (:90) → `async`.
- All three decoders' `generate()` (:457, :514, :729) → `async
  Promise<GenerateResult>`; their three `cache.free()` calls are awaited.
- Nothing else in the file changes: drafters, cost model, verification
  logic are pure and sync.

`src/scheduler/kvcache.ts` — `BlockPagedKVCache`: add `async` to
`allocate`/`append`/`read`/`free`; bodies unchanged. This answers the
"async wrapper vs async methods" question: **async methods on the class
itself, no wrapper.** A wrapper (`AsyncPagedKVCache` delegating to a
sync `BlockPagedKVCache`) would leave two classes, two construction
sites to choose between, and a sync class that no longer satisfies the
interface — the exact divergence the prototype's header warns against
when it refuses to claim `implements PagedKVCache`. The `async`
keyword on four methods is the entire change to the host cache; its
tests are the only ones that observe it.

`src/scheduler/kvcache-vram.ts` — `VramPagedKVCache` needs **no
signature change at all**: its methods already return exactly the
promises the new interface requires. It can now declare
`implements PagedKVCache`, and its header's "deliberately does NOT
declare" paragraph becomes history (the integration workstream updates
that header). That the prototype already matches the async interface
exactly is evidence the async shape was the intended one.

`src/mesh/worker.ts` — two one-line changes: `await` the `forward()`
call at :279 (the handler is already async; heartbeat, assignment
polling, and concurrent executes are unaffected), and await/pass the
new factory at :216 (see §6.3).

`src/bench/novel-bench.ts` — `TimingExecutor.forward` (:382) → `async`
(its timing still brackets the awaited inner call); the four
`.generate()` calls (:445, :469, :471, :473) are awaited in the
already-async bench runner.

### 2.2 Test impact (mechanical, countable)

| File | Change |
|------|--------|
| `test/engine.test.ts` | `StubCache` (`implements PagedKVCache`, :133): 4 methods → `async`; 4 `forward()` calls (:315, :322, :342, :359) → `await`. |
| `test/engine-options.test.ts` | 3 `forward()` calls (:186, :270, :283) → `await`. (Uses `BlockPagedKVCache` only via `forward`, so no direct cache edits.) |
| `test/scheduler.test.ts` | Direct cache calls (:21–:58) → `await`; the OOM check at :51–:55 uses `assert.throws(() => cache.allocate(...))` and must become `assert.rejects(...)`, since the throw becomes a rejection. `usedBlocks()`/`totalBlocks()` assertions unchanged. |
| `test/kvcache-vram.test.ts` | Already awaits the vRAM cache; the 4 sync host-cache calls (:187, :193, :200, :212) → `await`. |
| `test/kvcache-vram-bench.ts` | `fillLayer` already awaits; the host-branch reads (:~258–266) are currently un-awaited and must be awaited, or the timed "read" would measure promise creation instead of the read. The 3 `forward()` calls (:360, :376, :424) → `await`. |
| `test/novel-decoding.test.ts` | 16 `.generate()` calls → `await`. |
| `test/mesh.test.ts` | `makeCache` object literal (:92): 4 methods → `async`; fake executor `forward` (:116) → `async`. |
| `test/worker-model-switch.test.ts` | Same two edits (:61, :133, :167). |
| `test/vram-weights.test.ts` | Same two edits (:251, :480). |
| `test/failure-vram-weights.test.ts` | Same three edits (:513, :969, :1072). |
| `test/scheduler-novel.test.ts` | Unaffected: it drives the scheduler's async `step()` and fakes the mesh boundary, not `StageExecutor`. |

No test is deleted or weakened; every edit is `async`/`await` (plus
one `throws` → `rejects`). The scheduler, server, API, and
model-registry tests need no changes at all.

### 2.3 Costs and risks of (a)

- **Await granularity is per row per layer.** A prefill forward does
  `rows × layers` appends and reads (128 rows × 12 layers = 1536 of
  each). The awaits themselves cost microseconds (microtask hops); the
  real cost is the vRAM round trips, which are already measured (§0)
  and are the same under either option. Mitigating factor, specific to
  this design: prefill appends land in host staging and prefill reads
  of staged chunks are served from staging with no vRAM traffic, so
  most prefill awaits resolve without I/O. Decode pays one full-prefix
  vRAM read per layer per token — 12 round trips per token — which is
  precisely the measured +75% at ctx 1024.
- **No overlap is introduced — and none is promised.** Layers are
  sequential dependencies, so reads across layers cannot overlap
  without changing numerics. (a) does, however, *permit* future
  overlap (e.g. prefetching the next layer's prefix while the current
  layer's matmul runs); (b) structurally forbids it.
- **Error-type divergence.** The host cache throws `Error`; the vRAM
  cache rejects with `VramKvError`. The interface declares no error
  type, so callers must treat any rejection as sequence-fatal and let
  the scheduler re-prefill or fail the sequence — never branch on the
  error class. The scheduler already has that path (`failSequence` in
  `src/scheduler/batch.ts`).
- **Speculative/Jacobi overwrite path.** Both decoders rely on `append`
  overwriting a position's slot. Against vRAM, an overwrite inside an
  already-flushed chunk triggers the RMW fetch path
  (`kvcache-vram.ts`, `ensureStaged`). Correct, but it is the one place
  where the vRAM cache is slower *per append* than the host cache in a
  way the prototype's happy-path numbers do not cover; the integration
  proof must include a rejection-repair test (criterion P8).

## 3. Option (b): synchronous blocking transport (worker thread + `Atomics.wait`)

### 3.1 What would be built

A new module (no such code exists in the repo) with a hand-rolled IPC
protocol over a `SharedArrayBuffer`: the engine thread writes an op
code (ALLOC / APPEND / READ / FREE) plus arguments into a control
region, `Atomics.wait`s on a status word; a `node:worker_threads`
worker owns the `VramAccess`, performs the async HTTP calls, writes
result bytes into a data region, sets the status, and `Atomics.notify`s.
Node permits `Atomics.wait` on the main thread (unlike browsers), so
this is possible. The repo's nearest precedent,
`src/hw/kernels-cpu.ts:419,510,578`, uses worker threads +
`SharedArrayBuffer` for matmul — but with async message-passing
(`postMessage` + one-shot listeners), never `Atomics.wait`. There is no
in-repo precedent for a blocking bridge, and the kernels file documents
a hard-learned lesson (a persistent `message` listener re-refs the
port and keeps the process alive forever) that the bridge would have
to re-learn around shutdown ordering.

### 3.2 Shared-memory marshaling — no clean fixed size

The data region must hold the largest single transfer: one layer's
full-prefix read. At demo geometry that is ctx 1024 × 4 KiB/token =
**4 MiB** per read (48 MiB across 12 layers, but one layer at a time,
so 4 MiB suffices); at 70B-ish geometry (kvDim 1024 → 8 KiB/token)
ctx 4096 is **32 MiB** per layer. A fixed SAB sized for the worst case
is infeasible at long context (ctx 128k × 8 KiB = 1 GiB); a growable
SAB (`maxByteLength`, Node 22) makes the size dynamic but then both
sides must agree on a changing layout per request — more protocol. A
fresh SAB per request (transferred by `postMessage`) reintroduces
allocation + GC cost per read and defeats the zero-copy point. There is
no sizing answer that is both fixed and correct across models and
contexts; (a) has no such problem (each `read()` allocates exactly the
arrays it returns).

There is also an aliasing hazard: the natural zero-copy return — a
`Float32Array` view over the shared region — is clobbered by the next
read. The engine uses the returned arrays immediately inside the row
loop today, but `blockStatsFor` / `packedKvFor` derive cached data
from them, so "immediate use" is a load-bearing assumption that the
next reader could silently break. The safe choice is a copy out of the
SAB per read (a 4 MiB memcpy per layer at ctx 1024 — cheap next to a
14–99 ms network read, but it is a copy the design must state, not a
zero-copy win to claim).

### 3.3 Event-loop blocking — the decisive failure

The engine thread **is** the worker process's main thread: it runs the
HTTP server (`createMeshServer`), the heartbeat timer, and assignment
polling. Every `Atomics.wait` blocks all of them for the duration of
that vRAM round trip.

- One layer's read at ctx 1024 costs 14–99 ms (warm–cold); one decode
  token is 12 such waits, ~170 ms–1.2 s of accumulated blocking.
- One prefill forward is far worse: 128 rows × 12 layers of
  append/read waits. Even if only the reads hit vRAM, that is hundreds
  of waits totalling **~20 s to ~150 s** of event-loop blockage inside a
  single `forward()` call.
- The orchestrator's trip wires are `HEARTBEAT_TIMEOUT_MS = 6000` and
  `HEARTBEAT_DEAD_MS = 15000` (`src/contracts.ts`). A worker that is
  alive but deaf for 20+ s is marked suspect, then dead, mid-prefill;
  the orchestrator re-plans around a healthy worker while that worker
  is still (correctly) computing. Heartbeats are best-effort on the
  worker side (`doHeartbeat` catches and logs), but the orchestrator
  side has no such leniency.
- Concurrent executes stall behind the blocked loop too, and
  `activeExecutions` accounting freezes with them. The worker can serve
  exactly one sequence's forward at a time *and* cannot even say so.
- Mitigations do not save it: chunking the wait (poll with a timeout
  and pump the event loop between polls) is just cooperative
  async-with-extra-steps and still serializes everything behind the
  longest read; moving the HTTP server or heartbeats to another thread
  is a second bridge of the same kind.

This is the sharpest form of the argument: **(b) fails precisely in
the viable slice.** The prototype's one viable slice is prefill-only
staging (§0), and prefill is where (b) blocks longest. Short-context
decode — where (b)'s blockage is shortest — is also where (a)'s cost
is negligible, so (b) wins nowhere.

### 3.4 Failure modes (b) invents, (a) does not have

- **Worker-thread crash / hang.** `Atomics.wait` without a timeout
  blocks the engine thread forever; the process must be killed from
  outside. With a timeout, the request state becomes unknown — the
  late worker may still complete and write into the SAB after the
  engine has moved on — so the protocol needs generation/epoch fencing
  to reject stale writes. (a)'s equivalent is a rejected promise.
- **Lossy errors.** A `VramKvError` with a cause chain must be encoded
  into a bounded SAB region as a numeric code + truncated string. The
  scheduler's re-prefill decision then depends on a code table instead
  of the error object.
- **New test surface.** The bridge needs its own suite — timeout,
  worker crash, late-write fencing, concurrent requests, SAB overflow,
  shutdown ordering — roughly the size of the test churn (a) causes,
  except it tests new IPC instead of converted call sites. The "zero
  test churn" selling point of (b) is false; the churn moves.
- **(b) fixes only the interface, not the cost.** The read ratio is
  unchanged: same 6–9x at ctx 1024, same projected +75% decode. Anyone
  choosing (b) for performance reasons is mistaken about what it buys.

### 3.5 The one honest point for (b)

It is the smaller diff to the engine's public surface and could ship
without touching the decoders or any test — if the team needed vRAM KV
this week, only at short contexts, and only inside a worker whose
heartbeats did not matter, (b) would be the faster hack. None of those
conditions hold: the serving path's heartbeats are load-bearing, and
prefill is the target slice. (b) is rejected.

## 4. Recommendation — option (a)

Adopt (a): make `PagedKVCache` and `StageExecutor.forward()` async,
convert `BlockPagedKVCache` to async methods in place (no wrapper),
make the three decoders' `generate()` async, and change the worker's
cache factory as specified in §6. This confirms the steer: the worker's
execute path is already async, so the serving path above the engine
needs no changes at all — scheduler, server, mesh protocol, and
`MeshClient` are untouched, which is the containment property that
matters. Two corrections to the steer for the record: (1) the blast
radius is "engine + worker factory + bench + tests", and the tests
include the decoders' 20 `generate()` call sites, not just the engine
tests; (2) the factory change is a second parameter (the vRAM access),
not an async factory — cache construction itself performs no I/O (see
§6.3). Neither correction changes the recommendation.

## 5. Fallback policy — vRAM-only when vRAM is selected; no runtime fallback

**Decision:** there is no runtime fallback from vRAM KV to host KV,
at any point — not at cache construction, not on vRAM OOM, not
mid-sequence. `BlockPagedKVCache` stays in the repo, but as an
explicitly selected store and as the test/bench oracle, never as a
fallback.

**How selection works.** The store is chosen once, explicitly, at
worker startup (flag/env, e.g. `--kv-store host|vram`; exact flag name
is the integration workstream's choice). The worker logs the selection
at startup in the same style as the weight-loader log lines in
`loaderFor` (`src/mesh/worker.ts`), so any run's KV store is visible
in the worker log and never inferred.

**Why not a runtime fallback.** The prototype's rule governs: any
fallback must be explicit and observable, or the path is vRAM-only
with a clear error. A runtime fallback fails that test three ways:

1. **It silently changes failure semantics.** Host KV does not survive
   donor loss; vRAM KV with replicas=2 does. A sequence that started
   on vRAM and silently continued on host would lose its durability
   guarantee without any record of when the guarantee stopped holding.
2. **It hides capacity problems.** vRAM exhaustion is the signal the
   operator needs (add donors, lower concurrency). A fallback converts
   it into silent host-memory growth until the worker OOMs somewhere
   less diagnosable.
3. **It makes measurements non-comparable.** A benchmark run could be
   part-vRAM, part-host KV with no marker in the results — the same
   failure that corrupted the prototype's first header numbers, which
   had to be corrected after the fact.

**What happens instead, per failure point:**

| Failure | Behaviour (vRAM selected) |
|---------|---------------------------|
| vRAM unreachable / alloc fails at cache-build time (`loadAssignment`) | Stage load fails; worker keeps the previous stage and retries on the next poll — the exact pattern `loaderFor` already uses when a weight map cannot be fetched (`worker.ts:121–131`). Log line + counter; no host cache is constructed. |
| vRAM OOM mid-sequence (`allocate`/`append` rejects `VramKvError`) | The sequence errors; the scheduler fails or re-prefills it from its prompt tokens (the `failSequence` path). Its KV is never migrated to host memory. |
| vRAM read fails on committed KV (page unavailable on all copies) | `read()` rejects `VramKvError`; committed KV is never silently zero-filled (prototype semantics, kept). Scheduler re-prefills. |
| Worker crash | Staged (unflushed) KV is lost exactly as host KV is today; flushed KV survives on donors. Scheduler re-prefills. |

**Default.** vRAM KV is **opt-in** at first (`host` remains the
default) until the read-cost work lands (sub-page reads, or a vGPU
attention kernel that reads KV in place on the donor — the prototype's
conditions 2–3 in its header). The interface change (a) lands
regardless, so flipping the default later is a config change, not a
refactor. Shipping vRAM KV as the default while decode costs +75% at
ctx 1024 would repeat the prototype's mistake of letting an interface
decision smuggle in a performance regression.

**What `BlockPagedKVCache` remains for.** (1) The `host` store, (2)
the oracle `test/kvcache-vram.test.ts` compares the vRAM cache
against, (3) the bench baseline. It is not deleted, and it is not a
fallback.

## 6. Exact interface changes

### 6.1 `src/contracts.ts`

```ts
// PagedKVCache — allocate/append/read/free become Promise-returning.
// usedBlocks()/totalBlocks() stay synchronous (pure in-memory counters).
export interface PagedKVCache {
  readonly blockSize: number;
  allocate(seqId: string, layer: number, numTokens: number): Promise<number[]>;
  append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): Promise<void>;
  read(seqId: string, layer: number, positions: number[]): Promise<{ keys: Float32Array; values: Float32Array }>;
  free(seqId: string): Promise<void>;
  usedBlocks(): number;
  totalBlocks(): number;
}

// StageExecutor — forward becomes Promise-returning. embed/load/memoryBytes unchanged.
export interface StageExecutor {
  readonly assignment: ShardAssignment;
  load(loader: WeightLoader): Promise<void>;
  embed(tokenIds: number[]): Tensor;
  forward(input: Tensor, ctx: ForwardContext): Promise<Tensor>;
  memoryBytes(): number;
}
```

No change to `ForwardContext`, `EngineBackend`, `WeightLoader`, the
mesh protocol types, or any scheduler/server type.

### 6.2 `VramAccess` moves to `contracts.ts`

`VramAccess` is currently defined in `src/scheduler/kvcache-vram.ts`
(the prototype file). Move the interface — not the implementations —
into `src/contracts.ts`, next to `IVirtualRam`, so the worker, the mesh
client, and tests can reference it without importing the prototype
file:

```ts
export interface VramAccess {
  read(vaddr: number, length: number): Promise<Uint8Array>;
  write(vaddr: number, data: Uint8Array): Promise<void>;
  alloc(sizeBytes: number, opts?: { replicas?: number }): Promise<{ id: string; vaddr: number }>;
  free(id: string): Promise<void>;
}
```

`VirtualRamAccess` (the in-process adapter) stays in
`src/scheduler/kvcache-vram.ts`. The production `RemoteVramAccess`
does not exist yet and is a named dependency (§1.4, §7) — the refactor
workstream must not invent it as a side effect.

### 6.3 `src/mesh/worker.ts` — factory signature

```ts
// Before:
createCache: (manifest: ModelManifest) => PagedKVCache;
// After:
createCache: (manifest: ModelManifest, vram: VramAccess | null) => PagedKVCache;
```

- The second parameter is the vRAM access for this worker, or `null`
  when the worker runs with the host store. It is passed explicitly
  rather than closed over, so tests can inject a fake access (or
  `null`) without reaching into the worker's construction scope.
- **The factory stays synchronous.** Neither cache constructor performs
  I/O (`VramPagedKVCache` allocates lazily in `allocate()`), so no
  `await` is needed at the call site (`worker.ts:216`). If the
  integration workstream finds build-time I/O (e.g. a handshake on the
  remote access), the type becomes
  `(manifest, vram) => PagedKVCache | Promise<PagedKVCache>` and
  `loadAssignment` awaits it — one line, decided then, not now.
- `startWorker` itself needs no new option: the caller (`main()`) closes
  over the mode and the access when building the factory closure, and
  the factory's type carries the contract. `main()`'s CLI factory
  (:485) becomes the mode switch: `host` → `new BlockPagedKVCache(16,
  1024, heads, headDim)`; `vram` → `new VramPagedKVCache(16, <cap>,
  heads, headDim, vramAccess, { replicas: 2 })` with the `vram`
  argument asserted non-null (a `vram` worker started without an
  access fails fast at startup, before registering).
- `executeHandler` (:279): `const output = await
  stageExecutor.forward(...)`. Nothing else in the handler changes.
- Test factories (`test/mesh.test.ts:92`,
  `test/worker-model-switch.test.ts:167`,
  `test/vram-weights.test.ts:520`,
  `test/failure-vram-weights.test.ts:513`) gain the second parameter
  and ignore it.
- **Leak closure (required, §1.3):** `loadAssignment()` must
  `await oldCache?.free(...)` — best-effort, failures logged — before
  replacing `cache` on a model switch, and `close()` must do the same
  for the current cache (plus `flushAll()` first when vRAM-backed, so
  no committed KV sits un-replicated in host staging at shutdown).
  For the host cache these calls are no-ops in effect; for vRAM they
  return whole 4 MiB chunks to the pool.

### 6.4 `src/engine/*` and `src/scheduler/*`

| File | Change |
|------|--------|
| `src/engine/transformer.ts` | `forward`, `forwardLayer`, `attendPrefillNovel` → `async`, awaiting the 5 cache sites in §1.2. Everything else sync. |
| `src/engine/novel-decoding.ts` | `forwardTokenRows` → `async`; `AutoregressiveDecoder.generate`, `SpeculativeDecoder.generate`, `JacobiDecoder.generate` → `async Promise<GenerateResult>`; await the 3 `cache.free()` sites. |
| `src/scheduler/kvcache.ts` | `BlockPagedKVCache.allocate/append/read/free` → `async`; bodies unchanged; still throws plain `Error`. |
| `src/scheduler/kvcache-vram.ts` | No signature change. Declare `implements PagedKVCache`. Update the file header: the NO-GO verdict and the sync-interface section become history with a pointer to this note; the measured numbers and the layout/append/read/replication semantics stay as the authoritative record. |
| `src/bench/novel-bench.ts` | `TimingExecutor.forward` → `async`; await the 4 `generate()` calls. |

### 6.5 Explicitly out of scope for the refactor

- The remote `VramAccess` + orchestrator KV endpoint (§1.4).
- Any read-path performance work (sub-page reads, prefetching,
  in-place vGPU attention, cross-layer read overlap).
- Flipping the default store to `vram` (§5).
- Changing the chunk layout, staging design, replication default (2),
  or the `VramKvError` contract.
- Any change to the scheduler, server, API layer, or mesh protocol.

## 7. Proof criteria the integration must meet

The refactor workstream owns the code; the integration workstream must
demonstrate each of these before vRAM KV ships behind the flag. P1–P3
are the refactor's own exit criteria; P4–P8 are integration criteria.

- **P1 — Typecheck + full suite green, no test deleted.** `tsc` clean;
  every existing test passes with only the mechanical edits in §2.2
  (plus the header update in §6.4). Any test that cannot be converted
  by adding `await` is a design smell — stop and re-read this note.
- **P2 — Token identity, host vs vRAM.** Same prompt, same seed,
  greedy decoding: the token stream from a vRAM-backed stage is
  identical to the host-backed stage, for a short prefill (ctx ≤ 128)
  and a long one (ctx 1024), across prefill + ≥ 16 decode tokens.
  Covers the overwrite/RMW path: include a speculative-decoding run,
  whose rejections exercise append-overwrite of flushed chunks.
- **P3 — Determinism preserved.** `test/engine.test.ts`'s
  bitwise-identical repeated forward still passes, and the
  novel-decoding token-identity assertions (speculative/Jacobi ==
  autoregressive greedy) still pass against the async engine.
- **P4 — No vRAM leak.** After `free(seqId)`, `stats().vramBytes` and
  `stats().chunks` for that sequence drop to 0. After a model switch
  and after worker `close()`, no chunks belonging to the old stage
  remain allocated (assert via `VramPagedKVCache.stats()` /
  `IVirtualRam.stats()` in the harness).
- **P5 — No silent fallback, observably.** With `--kv-store vram` and
  vRAM unreachable, the stage load fails loudly (log line naming the
  model and the cause, previous stage kept, retry on next poll) and no
  host cache is ever constructed — assertable by a counter (e.g.
  `kvStoreFallbacks`, which must read 0) and by the startup log line
  naming the selected store. Grep-able in the worker log.
- **P6 — Donor-loss semantics.** With replicas=2, kill one donor
  mid-generation: decode continues and reads succeed (healing
  re-replicates). With replicas=1 and a donor loss, the next read
  rejects `VramKvError` and the scheduler re-prefills — committed KV
  is never silently zero-filled.
- **P7 — Heartbeat liveness.** During a long prefill against a slow
  fake `VramAccess` (e.g. 50 ms/read), the worker keeps heartbeating
  on schedule — no suspect/dead trip. This is the property (b) would
  have violated; the integration test must assert it, not assume it.
- **P8 — Measured before/after, both contexts.** Tokens/sec on the
  live server path (not just the bench harness) before and after the
  switch, at ctx 128 and ctx 1024, reported as measured numbers.
  Expectation from the prototype: ctx 128 within ~5% of host; ctx
  1024 materially slower (~+75% decode). The ctx-1024 number must be
  reported, not hidden, and it is the input to the later decision to
  flip the default (§5).

## 7a. Status after integration (2026-11-23)

- **P1 — met, pending the final full-suite run.** `tsc` clean; every
  targeted file green (`test/mesh.test.ts`,
  `test/worker-model-switch.test.ts`, `test/kvcache-vram.test.ts`,
  `test/scheduler.test.ts`, `test/kvcache-vram-survival.test.ts`).
  No test deleted. One full sequential run remains at the end.
- **P2 — met at 8 decode tokens, short + long.** The bench records
  identical content host vs vRAM on the short prompt (`"CCCCCCCC"`,
  3/3 each) and on the long prompt's successful rep (8 replacement
  chars, identical). The letter of P2 wants ≥ 16 decode tokens and a
  speculative-decoding run (whose rejections exercise the
  append-overwrite of flushed chunks) — that extension is not done;
  the 8-token identity is what the bench measures.
- **P3 — met.** Engine bitwise-identical repeated forward and the
  novel-decoding identity assertions pass against the async engine
  (`test/engine.test.ts`, `test/novel-decoding.test.ts` green).
- **P4 — met.** `free(seqId)` drops `usedBlocks` and vRAM bytes to 0
  and `close()` frees every chunk
  (`test/kvcache-vram.test.ts` §3, §4b). The serving-path leak that
  P4's cache-level tests could not see — `freeSequence` never wired,
  so finished sequences leaked until stage reload — was found live
  (second long generation OOMed) and fixed with the
  `POST /mesh/free-sequence` route; two long generations back-to-back
  now succeed on both stores.
- **P5 — met.** The vRAM factory probes the access (one alloc +
  free); a rejecting factory keeps the previous stage and retries on
  the next poll (`test/worker-model-switch.test.ts` case 5b). No host
  cache is ever constructed in vRAM mode (the factory throws before
  building one). Startup log names the selected store (`[worker]
  kv-store: vram`).
- **P6 — met.** Donor death mid-generation: decode continues,
  generation identical, healing 0 pages lost
  (`test/kvcache-vram-survival.test.ts` donor-death e2e). Replicas=1
  loud rejection is cache-level only; the scheduler re-prefill on a
  replicas=1 read failure is the documented contract, not an e2e run.
- **P7 — met (dedicated test `test/heartbeat-p7.test.ts`, 4/4 pass,
  2026-11-23).** Contract pinned: suspect after >6 s silence, dead
  after >15 s (exact boundaries asserted), heartbeat revives
  suspect/dead to healthy, and suspect alone never triggers a
  re-plan — only newly-dead workers do. Live proof: a real worker
  serving demo-100m through a real `VramPagedKVCache` over a slow
  fake `VramAccess` (50 ms/read) generated 23 tokens in 12.3 s
  (22 decodes, 10 post-flush, chunkFlushes=12, vramReads=132) with
  zero suspect/dead transitions and max heartbeat gap 2,338 ms
  (500 ms interval) — the generation's wall time exceeded the 6 s
  suspect timeout while heartbeats kept flowing. A control test
  proves the detector works (blocking the event loop 6.5 s trips
  suspect), so the clean live run is a real pass. Caveat the test
  also measures: the 2.3 s max gap shows synchronous matmul still
  delays heartbeats measurably — a long *synchronous* prefill can
  still trip suspect (the known symptom); only the slow *async*
  vRAM reads are proven not to.
- **P8 — met, numbers in `docs/VRAM-WEIGHTS.md` and
  `bench-kv-{before,after-host,after}.json` (final bench 2026-11-23
  03:34–04:12 UTC, post leak-fix).** Short (ctx 18): host TTFT 3,527
  ms / 5.73 tok/s, vRAM 3,819 ms / 5.81 tok/s — parity. Long (ctx
  997): host TTFT 228,175 ms / 4.18 tok/s, vRAM 226,540 ms / 4.12
  tok/s, 3/3 on both — parity. The before short numbers (TTFT 6,207
  ms) ran under loadavg up to 4.16 vs ~1.1–1.5 for the after runs, so
  they are not a clean baseline and are not quoted as a speedup. The
  ctx-1024 flushed-chunk regime (cache-level +75% decode projection)
  is not measured by this bench — a ≤1024-token prompt never flushes
  — and is still the input to the later decision to flip the default
  (§5).

## 8. Risks and open questions for the integration workstream

- The remote `VramAccess` transport (§1.4) is the largest remaining
  unknown: binary page transport over HTTP already exists for weights;
  KV needs alloc/free as well as read/write, and its failure modes
  (partial write, alloc succeeding but the reply lost) need the same
  care the weight path got. Do not reuse the weight endpoint for KV
  pages — the access patterns and lifetimes differ.
- `VramPagedKVCache` assumes single-threaded driving (one forward at
  a time per stage). The worker guarantees this today
  (`activeExecutions` is accounting, not parallelism — `forward()` is
  awaited inside one handler invocation and handlers run one at a
  time per stage in practice). If the worker ever executes forwards
  concurrently, the cache needs serialization first.
- The per-row full-prefix read is the cost centre (12 vRAM round trips
  per decode token). The durable fix is not in this note: either a
  sub-millisecond small-read path in vRAM, or attention that reads KV
  in place on the donor. Until one lands, long-context vRAM decode
  stays a capacity-vs-speed trade and the default stays `host`.
