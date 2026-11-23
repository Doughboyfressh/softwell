# Weights from vRAM — design, proof, and numbers

Date: 2026-11-22. Phase: "weights from vRAM" — the engine used to load
weights from local disk at inference time (`SafetensorsWeightLoader`,
`fs.readFileSync`); vRAM staging was allocation/accounting only. After
this phase, weights come from pooled vRAM allocations.

## How weights flow now: vSSD → vRAM → worker

1. **Ingest (the only disk reads on the weight path).**
   `POST /v1/models/load` (`ModelRegistry.load`, `src/server/registry.ts`)
   reads the manifest + tokenizer from the model directory, fit-checks
   weights + KV headroom against vRAM, then stages every file to vSSD
   (`vm.ssd.putObject`) and stages each `*.safetensors` shard into vRAM
   (`hypervisor.stageToVram`, one allocation per shard, 4 MiB pages,
   2 replicas by default). The registry records each shard's safetensors
   header length while the bytes are in hand — the weight map needs it to
   locate tensors inside the raw staged bytes.
2. **Serving.** The orchestrator serves the staged manifest + per-tensor
   weight map as JSON (`GET /mesh/weights/map?model=<id>`) and each
   tensor's raw bytes as octet-stream (`GET /mesh/weights?model=<id>&
   tensor=<name>`, `ModelRegistry.readWeightTensor` → `vm.ram.read`).
   The map locates a tensor inside its shard's staged bytes
   (`vaddr + headerLen + dataOffsets`); the read goes through pooled
   vRAM with replica failover, never through disk.
3. **Worker load.** On a new shard assignment the worker asks the
   orchestrator for the weight map and builds a `VramWeightLoader`
   (`src/model/vram-loader.ts`) whose tensor fetches hit the binary
   endpoint. `VramWeightLoader` imports no `node:fs` — it cannot read a
   file even by accident. The disk loader (`SafetensorsWeightLoader`)
   remains only as a fallback for models the orchestrator has not staged
   (pre-activation / v1 path); the worker logs which loader it chose
   (`using vRAM weight loader …` / `using disk weight loader …` and
   `loaded stage … via vRAM|disk weights`).
4. **Healing.** vRAM pages are replicated (default 2 copies, round-robin
   across healthy donors). The hypervisor sweep heals a dead donor by
   re-replicating surviving copies onto healthy donors
   (`VirtualRam.handleNodeLoss`); reads fail over to the surviving
   replica first. Writes store the new bytes to **every** copy before
   returning, so a replica never holds stale bytes and failover can
   never serve pre-write data (see "Integration fixes" below).
   See `test/healing-vram-weights.test.ts`.

## Counters

- `VramWeightLoader.tensorsFetched` / `bytesFetched` — distinct tensors
  / raw bytes actually fetched from vRAM (per-tensor cache hits excluded).
- `SafetensorsWeightLoader.bytesReadFromDisk` — raw bytes actually read
  from disk (file-cache hits excluded). The proof asserts this stays 0
  behaviorally (see below).
- `ModelRegistry.weightBytesServed` — raw weight bytes the orchestrator
  has served to workers over `GET /mesh/weights`.

Gap (2026-11-22): these counters live in-process and are **not exposed
on any HTTP endpoint** — `weightBytesServed` has a getter on the
registry but no route surfaces it, and the workers' loader counters are
inside worker processes. The proof test therefore asserts the same fact
behaviorally (disk-removed generation + loader-choice log lines) instead
of reading a counter over HTTP. Surfacing the counters (e.g. on
`/v1/server/status` / `/v1/cluster`) is still to do.

## Proof: inference with the disk removed

`test/vram-weights-proof.test.ts` — boots the real cluster
(`node src/cli.ts up`, 2 workers), loads + activates demo-100m over
HTTP, then renames `demo/models/demo-100m` out of reach (restored in a
`finally`, even on failure) and proves:

- **(a) Generation still streams correctly.** 8 tokens, content
  identical to before the move (`"CCCCCCCC"` — random demo weights,
  temperature 0). Run 2026-11-22: before-move and after-move content
  both `"CCCCCCCC"`, 8/8 content tokens, `[DONE]` received.
- **(b) No disk loader is ever constructed.** The captured worker log
  shows both workers loading `via vRAM weights` and contains zero
  `using disk weight loader` lines — before the move, after the move,
  and during the restart below. The binary weight endpoint keeps
  serving bytes with the dir gone: `GET /mesh/weights?model=demo-100m
  &tensor=model.embed_tokens.weight` returned 1,060,864 bytes before
  and after the move (111 tensors staged). With the directory renamed
  away, any disk read would throw — successful generation *is* the
  zero-disk-read proof.
- **(c) A worker restart also comes from vRAM — PASSED on a quiet
  host (2026-11-22, resumed run).** A fresh replacement worker
  (spawned with `--model-dir demo/models/demo-10m` so it can boot —
  see "Still not done") registers, one original worker is killed, the
  fault-detection sweep re-plans over the survivors, and the
  replacement builds a fresh `VramWeightLoader`
  (`using vRAM weight loader for model demo-100m (111 tensors staged
  on the orchestrator)`) and loads its stage
  (`loaded stage 1 (layers 6-12, model demo-100m, assignment v2) via
  vRAM weights`). Generation afterwards is again 8 tokens, content
  identical (`"CCCCCCCC"`). Passed twice: standalone in 107.2 s
  (ports mesh 36771 / api 43947) and inside the full `npm test` run
  in 105.3 s (ports mesh 33469 / api 33347). The test also asserts
  zero `using disk weight loader` lines across the whole run.

  History — (c) **failed once live, under heavy contention**, before
  the resume (see below). Its stage load then failed repeatedly:
  `failed to load assignment v2: mesh request failed with HTTP 500:
  vRAM: page <alloc>/p77 unavailable on all 2 copies`, and generation
  did not recover within the 240 s retry window.

  What is known about that earlier (c) failure (2026-11-22):
  - The vRAM failover/healing mechanism itself works: the in-process
    equivalent passes — `test/healing-vram-weights.test.ts` test 2
    (kill a donor, `sweep()`, re-read every tensor byte-identical) and
    a scratch reproduction of the live sequence with a third donor
    joining (`scripts/diag-vram-live-sequence.ts`, demo-10m, 57/57
    tensors byte-identical after kill + sweep, healing re-replicated
    onto the joined donor) both pass. `pagesLost` was 0 in both.
  - The live failure happened while the shared host was heavily
    contended: two full test suites and several live clusters were
    running concurrently, load average ~5.6 on 4 cores, and the
    kernel OOM-killed processes twice in the same window. A
    CPU-starved donor failing to answer page fetches (or a store
    failing under memory pressure) produces exactly this symptom:
    both copies "unavailable" from the reader's side, with healing
    unable to re-replicate from the same unanswering copy.
  - The resumed run confirms the contention reading: on a quiet host
    (load ~1.5, >4 GiB free) (c) passes, twice. The failure is
    therefore treated as environmental, not a product bug — but it is
    a real robustness note: a worker restart under host memory
    pressure can fail to re-load its stage, and the test's 240 s
    retry window did not save it that time.

Three test-only fixes landed while resuming this workstream (no `src/`
changes): ephemeral ports per run (fixed ports collided with an
orphaned cluster from the pre-restart attempt); a `finally` sweep
that kills any remaining mesh workers for the test's port (orphaned
workers inherit the orchestrator's stdio and held the runner's pipes
open, so the test process never exited); and the shutdown assertion
corrected from 200 to 202 (`POST /v1/server/shutdown` answers 202 by
design).

## Before / after numbers

Model: demo-100m (104,365,056 params, random weights). Prompt:
`"Hello world"`, `max_tokens` 8, `temperature` 0, streaming, 1 worker
for the bench (the proof test uses 2 workers — see above). Script:
`scripts/bench-vram-weights.mjs`; raw JSON:
`bench-vram-weights-before.json` (before) and
`bench-vram-weights-after.json` (after, when measured).

Host (both runs): AMD EPYC 9D25, 4 cores, 7.7 GiB RAM, Node v24.22.1.
TTFT is measured from request send to first content token; decode
tok/s is content tokens over first-to-last content token. Medians of
3 runs; range in brackets.

| Metric | Before (disk weights) | After (vRAM weights) | Change |
|---|---|---|---|
| TTFT, median ms | 9,491 [5,413–10,180] | _pending_ | — |
| Decode tok/s, median | 2.29 [2.09–3.11] | _pending_ | — |
| Total stream time, median ms | 13,319 [7,985–13,679] | _pending_ | — |
| Tokens / run | 8 / 8 / 8 | _pending_ | — |

Before-run conditions (2026-11-22, `bench-vram-weights-before.json`):
load average 5.63 at run start (other workstreams running benches in
parallel), 6,695 MiB free of 7,934 MiB. Decode is far below the
coordinator's earlier live proof on the same host (5.10 tok/s, TTFT
5,882 ms) — the host was loaded, and the earlier run's TTFT was
measured differently. The before numbers above are this workstream's
own fresh measurement with the fixed script (timer starts before the
request is sent; an earlier script version started the timer after
headers arrived and understated TTFT — those numbers were discarded,
not reported).

After numbers are **pending integration** (2026-11-22, resumed run).
They are re-measured with the identical script, prompt, and repetition
count once the tree is green with all workstreams landed and the host
is quiet enough for a fair comparison. The proof test passes, and the
integration fixes below landed green — healing 7/7 plus
binary-transport 1/1, `npx tsc --noEmit` clean — but the full suite
has not completed yet: the default parallel `npm test` run was killed
at ~60 min by host contention (35 test files in parallel on 4 cores /
7.7 GiB), so the suite is being re-run with
`node --test --test-concurrency=2`. The earlier 75/77 state is
superseded: the two failures then were the healing count expectations
(now fixed — see below) and the `vram-weights.test.ts:294` type error
(now fixed).

When the suite lands green, run the same script with `--label after`
(`node scripts/bench-vram-weights.mjs --label after --mesh-port <p>
--api-port <q>`, same prompt, same 3 repeats; it writes
`bench-vram-weights-after.json`) and fill in the table. If inference
got slower, the table will say so plainly — extra vRAM read hops are
a real possibility on this host. Do not quote after numbers until that
run exists, and do not present the incidental 5.71 tok/s decode seen
in the standalone `server-e2e` run as the "after" number: that run
used 1 worker via `softwell serve` under different conditions, not
the baseline protocol.

## Integration fixes (2026-11-22)

Three real bugs surfaced while landing the workstreams; all are fixed
in `src/` or `test/`, with tests pinning the new behavior.

- **Stale replicas served wrong bytes on failover (the big one).**
  `VirtualRam.write` used to store new bytes to the primary copy
  only and mark the other replicas `dirty`, trusting the next read to
  repair them (the read's lazy-repair loop re-stored the fresh bytes).
  If the primary donor died *before* any read repaired the replicas,
  failover served the replicas' stale bytes — zeros for freshly
  staged weights — as if they were current, and `handleNodeLoss`
  then re-replicated those stale bytes as the "recovered" page, so
  the corruption became permanent. The healing tests exposed it:
  post-kill reads returned zeros for pages whose primary had died,
  and the failure only appeared when no read had happened before the
  kill (a pre-kill read repaired the replicas and masked it — which
  is why the failure workstream's re-load test, which reads before
  killing, passed). Fix (`src/hw/vram.ts`): a write now stores to
  **every** copy (batched per donor, as before) and only then bumps
  the version and marks all surviving copies resident at the new
  version; copies that still fail are dropped from the page table.
  Defense in depth: `fetchPageData` now skips any copy whose version
  lags the page table, so a stale copy can never be served even if
  some future path leaves one behind — it fails loudly instead.
  Cost: a write sends replicas × the bytes (2× at the default
  replica count); reads and healing are unchanged, and staging-time
  cost only — inference reads weights, it does not write them.
  `test/binary-transport.test.ts` previously *asserted* the
  lazy-repair re-store on reads; it now asserts the opposite (the
  write replicated eagerly, so a read performs no repair store).
  Note: this bug is distinct from the live proof-(c) failure above —
  that failure was pages *unavailable* on all copies (starved donors
  answering nothing), while this bug served stale bytes *silently*.
  A (c)-style recovery in the stale window would have come back
  wrong without any error, not failed loudly.
- **Failed staging leaked its vRAM allocation.**
  `Hypervisor.stageToVram` allocated, then streamed chunks with no
  cleanup on failure — a donor death mid-stage left the whole
  allocation in the page table. It now frees the allocation
  (best-effort) and rethrows. The healing test's "failed staging
  frees its vRAM allocation" also had its expected-error regex
  broadened: the kill can land during the alloc, the vSSD chunk
  read, or a vRAM write, so the rejection is whichever layer fails
  first — the freed allocation is the point, not the error text.
- **Large-buffer `assert.deepEqual` OOMs the test process.**
  `assert.deepEqual` on a multi-MB Buffer enumerates every byte
  index as an object key; comparing the 38 MB staged weights blew
  the 2 GiB heap in ~30 s (and 128 MB pattern buffers likewise).
  `test/healing-vram-weights.test.ts` now compares bulk bytes with
  `Buffer.compare` via an `assertBytesEqual` helper (memcmp, no
  allocation). The same hazard exists in
  `test/binary-transport.test.ts:571,573` (8 MiB reads — passes, but
  worth converting). The healing test's count expectations were also
  corrected in the same pass: staging mirrors each vSSD chunk into
  vRAM, so a donor loss recovers staged weight pages *plus* vSSD
  mirror pages (`pagesRecovered` 20, not 10, for demo-10m).

## Still not done (2026-11-22)

- **KV cache is still worker-local host memory.** `VramPagedKVCache`
  (`src/scheduler/kvcache-vram.ts`) exists with correctness tests, but
  the serving path still builds `BlockPagedKVCache` per stage load
  (`src/mesh/worker.ts`); the vRAM KV cache is not wired in.
- **Activation transport is still JSON.** Stage inputs travel as
  `ExecuteRequest.input: { data: number[]; shape: number[] }` over
  `POST /mesh/execute`. Only weights (and vRAM page transfers) are
  binary now.
- **Activation still reads the manifest + tokenizer from disk.**
  `ModelRegistry.load` and the activation factory
  (`buildFactory`, `src/cli.ts`) call `loadManifest(entry.dir)` /
  `loadTokenizer(entry.dir)`. Re-activating a model after its directory
  is gone fails on the orchestrator side — the proof test therefore
  uses a worker restart, not a re-activation, as its re-load trigger.
- **A fresh worker process still needs a model directory to boot.**
  Worker `main` opens the spawn-time loader from `--model-dir` before
  registering. A replacement worker must boot from some existing
  directory (the proof test uses demo-10m); it then switches to the
  vRAM loader per-assignment. A worker cannot yet boot with no model
  directory at all.
- **Weight counters are not exposed over HTTP** (see Counters above).
- **A worker restart under host memory pressure can fail to
  re-load.** The one live (c) failure happened under OOM contention
  (see Proof); on a quiet host (c) passes. The failure mode — page
  fetches timing out on a starved donor while healing cannot
  re-replicate from the same copy — is worth a follow-up, not a
  blocker for this phase.
