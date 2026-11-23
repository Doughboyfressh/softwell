/**
 * P7 heartbeat-liveness test (test/heartbeat-p7.test.ts).
 *
 * Run with: node --test test/heartbeat-p7.test.ts
 *
 * Pins the heartbeat contract (docs/KV-DESIGN.md §7 / §7a P7):
 *
 *   - A worker that stops heartbeating goes `joining -> healthy ->
 *     suspect` after HEARTBEAT_TIMEOUT_MS (6 s) of silence and
 *     `suspect -> dead` after HEARTBEAT_DEAD_MS (15 s). A heartbeat
 *     from a suspect (or dead) worker revives it to `healthy`.
 *   - ONLY `dead` triggers a re-plan: a worker that merely trips
 *     `suspect` — the known symptom during long prefills, where the
 *     worker is alive but its event loop is busy — must NOT cause a
 *     re-plan. The orchestrator's sweep loop re-plans on newly-dead
 *     workers only (src/cli.ts); this test pins the `dead`-only rule
 *     at the registry level, where the transition is decided.
 *   - The async design's promise (the property option (b) in
 *     KV-DESIGN §3 would have violated): during a long generation
 *     against a slow fake VramAccess (50 ms/read), the worker keeps
 *     heartbeating on schedule — no suspect/dead trip — even though
 *     the generation's wall time exceeds the 6 s suspect timeout.
 *     Slow vRAM reads are awaits, so they yield the event loop and
 *     the heartbeat timer fires between them; only synchronous
 *     compute blocks it.
 *
 * The last point is proven live, not argued: a real worker serves
 * demo-100m through a real VramPagedKVCache over a slow in-memory
 * VramAccess while a real WorkerRegistry sweeps every 250 ms. The
 * test asserts the slow path was actually exercised (chunkFlushes >=
 * 1 and vramReads > 0 — otherwise "no suspect" would be vacuous)
 * and that the generation's wall time exceeded the suspect timeout
 * while the worker never left `healthy`.
 *
 * A control test proves the detector itself works: block the event
 * loop for 6.5 s and the sweep marks the worker suspect — so the
 * live test's clean run is a real pass, not a sweep that never ran.
 */

import assert from "node:assert/strict";
import * as path from "node:path";
import test from "node:test";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";
import {
  HEARTBEAT_DEAD_MS,
  HEARTBEAT_TIMEOUT_MS,
} from "../src/contracts.ts";
import type {
  ModelManifest,
  PagedKVCache,
  ShardAssignment,
  VramAccess,
  WorkerCapability,
  WorkerInfo,
} from "../src/contracts.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { WorkerRegistry } from "../src/mesh/registry.ts";
import { startWorker } from "../src/mesh/worker.ts";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { VramPagedKVCache } from "../src/scheduler/kvcache-vram.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** The exact re-plan trigger the orchestrator uses (src/cli.ts sweep
 *  loop): re-plan only for workers the sweep just marked `dead`. */
function newlyDead(changed: WorkerInfo[]): WorkerInfo[] {
  return changed.filter((w) => w.state === "dead");
}

function makeCapability(): WorkerCapability {
  return {
    host: "127.0.0.1",
    port: 0,
    device: "cpu",
    backend: "softwell-ts",
    memoryBytes: 8 * 1024 * 1024 * 1024,
    freeMemoryBytes: 8 * 1024 * 1024 * 1024,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    cores: 4,
    roles: ["mixed"],
    maxBatchTokens: 256,
  };
}

test(
  "P7 contract: suspect after 6 s silence, dead after 15 s, heartbeat revives",
  () => {
    const registry = new WorkerRegistry();
    const { workerId } = registry.register(makeCapability());
    assert.equal(registry.getWorker(workerId)?.state, "joining");

    // First heartbeat flips joining -> healthy.
    let res = registry.heartbeat({
      workerId,
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(res.ok, true);
    assert.equal(registry.getWorker(workerId)?.state, "healthy");

    const lastBeat = registry.getWorker(workerId)?.lastHeartbeatAt as number;

    // Boundary: exactly HEARTBEAT_TIMEOUT_MS of silence is NOT suspect
    // (sweep uses `>`); 1 ms past it is.
    assert.deepEqual(registry.sweep(lastBeat + HEARTBEAT_TIMEOUT_MS), []);
    assert.equal(registry.getWorker(workerId)?.state, "healthy");
    const toSuspect = registry.sweep(lastBeat + HEARTBEAT_TIMEOUT_MS + 1);
    assert.equal(toSuspect.length, 1);
    assert.equal(toSuspect[0]?.state, "suspect");
    assert.equal(registry.getWorker(workerId)?.state, "suspect");

    // A suspect worker that heartbeats again is healthy, not dead.
    registry.heartbeat({
      workerId,
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(registry.getWorker(workerId)?.state, "healthy");
    const beat2 = registry.getWorker(workerId)?.lastHeartbeatAt as number;

    // Boundary: exactly HEARTBEAT_DEAD_MS is suspect, not dead; 1 ms
    // past it is dead.
    const toSuspect2 = registry.sweep(beat2 + HEARTBEAT_DEAD_MS);
    assert.equal(toSuspect2.length, 1);
    assert.equal(toSuspect2[0]?.state, "suspect");
    const toDead = registry.sweep(beat2 + HEARTBEAT_DEAD_MS + 1);
    assert.equal(toDead.length, 1);
    assert.equal(toDead[0]?.state, "dead");
    assert.equal(registry.getWorker(workerId)?.state, "dead");

    // Dead is terminal for the sweep (no repeat change reported)...
    assert.deepEqual(registry.sweep(beat2 + HEARTBEAT_DEAD_MS + 60_000), []);
    // ...but a heartbeat from a dead worker still revives it.
    registry.heartbeat({
      workerId,
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(registry.getWorker(workerId)?.state, "healthy");
  },
);

test("P7 contract: suspect alone never triggers a re-plan; dead does", async () => {
  const registry = new WorkerRegistry();
  const a = registry.register(makeCapability()).workerId;
  const b = registry.register(makeCapability()).workerId;
  for (const id of [a, b]) {
    registry.heartbeat({
      workerId: id,
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
  }
  const t0 = Math.max(
    registry.getWorker(a)?.lastHeartbeatAt as number,
    registry.getWorker(b)?.lastHeartbeatAt as number,
  );

  // Both workers go silent for 7 s: suspect only — no re-plan.
  const changed = registry.sweep(t0 + 7_000);
  assert.ok(changed.length >= 1, "silence past 6 s must trip suspect");
  assert.ok(
    changed.every((w) => w.state === "suspect"),
    `7 s of silence must trip suspect only, got ${JSON.stringify(changed)}`,
  );
  assert.deepEqual(
    newlyDead(changed),
    [],
    "a suspect worker must not trigger a re-plan",
  );

  // Separate B's heartbeat from A's by 9 real seconds, so that at the
  // dead-check sweep A is ~16 s silent (dead) while B is only ~7 s
  // silent (suspect): the two beats must be seconds apart, not the
  // milliseconds apart that back-to-back heartbeat() calls produce.
  await sleep(9_000);
  registry.heartbeat({
    workerId: b,
    usedMemoryBytes: 0,
    activeSequences: 0,
    tokensSinceHeartbeat: 0,
  });
  const tB = registry.getWorker(b)?.lastHeartbeatAt as number;
  const changed2 = registry.sweep(tB + 7_000);
  assert.ok(
    changed2.some((w) => w.workerId === a && w.state === "dead"),
    `A must be dead after ~16 s silence, got ${JSON.stringify(changed2)}`,
  );
  assert.deepEqual(
    newlyDead(changed2).map((w) => w.workerId),
    [a],
    "only the newly-dead worker triggers a re-plan",
  );
  // B was refreshed 7 s ago: suspect, not dead — and not re-planned.
  assert.equal(registry.getWorker(b)?.state, "suspect");
});

test(
  "P7 control: blocking the event loop for 6.5 s trips suspect (the detector works)",
  () => {
    const registry = new WorkerRegistry();
    const { workerId } = registry.register(makeCapability());
    registry.heartbeat({
      workerId,
      usedMemoryBytes: 0,
      activeSequences: 0,
      tokensSinceHeartbeat: 0,
    });
    assert.equal(registry.getWorker(workerId)?.state, "healthy");

    // Block this thread — no heartbeat timer, no sweep timer can run —
    // for longer than the suspect timeout. Atomics.wait is the honest
    // way to block: a busy Date.now() loop would do the same work.
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 6_500);

    const changed = registry.sweep();
    assert.equal(changed.length, 1);
    assert.equal(changed[0]?.state, "suspect");
    assert.equal(registry.getWorker(workerId)?.state, "suspect");
  },
);

/* ------------------------------------------------------------------ */
/* Live P7: slow vRAM reads must not trip heartbeats.                 */
/* ------------------------------------------------------------------ */

/** In-memory VramAccess whose reads/writes cost a fixed 50 ms — the
 *  slow fake access KV-DESIGN §7 P7 specifies. Every await yields
 *  the event loop, so the worker's heartbeat timer keeps firing. */
class SlowVramAccess implements VramAccess {
  readonly readDelayMs: number;
  reads = 0;
  writes = 0;
  private readonly allocs = new Map<
    string,
    { vaddr: number; data: Uint8Array }
  >();
  private nextId = 0;
  private nextVaddr = 0x1000;

  constructor(readDelayMs = 50) {
    this.readDelayMs = readDelayMs;
  }

  private find(vaddr: number, length: number): { base: number; data: Uint8Array } {
    for (const { vaddr: base, data } of this.allocs.values()) {
      if (vaddr >= base && vaddr + length <= base + data.length) {
        return { base, data };
      }
    }
    throw new Error(`SlowVramAccess: read/write outside any allocation at vaddr ${vaddr}`);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const { base, data } = this.find(vaddr, length);
    await sleep(this.readDelayMs);
    this.reads += 1;
    return data.slice(vaddr - base, vaddr - base + length);
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    const { base, data: stored } = this.find(vaddr, data.length);
    await sleep(this.readDelayMs);
    stored.set(data, vaddr - base);
    this.writes += 1;
  }

  async alloc(
    sizeBytes: number,
    _opts?: { replicas?: number },
  ): Promise<{ id: string; vaddr: number }> {
    const id = `slow-alloc-${++this.nextId}`;
    const vaddr = this.nextVaddr;
    this.nextVaddr += sizeBytes;
    this.allocs.set(id, { vaddr, data: new Uint8Array(sizeBytes) });
    return { id, vaddr };
  }

  async free(id: string): Promise<void> {
    this.allocs.delete(id);
  }
}

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DEMO_100M_DIR = path.join(ROOT, "demo", "models", "demo-100m");
/** Tokens per vRAM chunk in the live test: a 16-token context flushes
 *  chunk 0 per layer, so decode reads after that hit slow vRAM. */
const CHUNK_TOKENS = 16;

function argmaxLastRow(output: { data: number[]; shape: number[] }): number {
  const rows = output.shape[0] as number;
  const cols = output.shape[1] as number;
  const start = (rows - 1) * cols;
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < cols; i++) {
    const v = output.data[start + i] as number;
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

async function waitFor(
  cond: () => boolean,
  what: string,
  timeoutMs = 120_000,
  intervalMs = 25,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cond()) return;
    if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
    await sleep(intervalMs);
  }
}

/** Capture console.log lines (workers log their stage loads there)
 *  while still printing them, so failures stay debuggable. */
function captureLogs(): { lines: string[]; restore(): void } {
  const lines: string[] = [];
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    lines.push(
      args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" "),
    );
    origLog(...args);
  };
  return {
    lines,
    restore: () => {
      console.log = origLog;
    },
  };
}

test(
  "P7 live: a long generation against a slow (50 ms/read) vRAM keeps heartbeating — no suspect/dead trip",
  { timeout: 300_000 },
  async () => {
    const logs = captureLogs();
    const registry = new WorkerRegistry();
    const slow = new SlowVramAccess(50);
    // Holder object (not a bare `let`): the cache is installed by the
    // createCache callback, and TS narrows a bare variable to null
    // across the await boundary — the holder keeps the read honest.
    const liveCacheHolder: { current: VramPagedKVCache | null } = { current: null };

    // Fake orchestrator: the real registry behind the real mesh server.
    // The test sweeps it every 250 ms and records every transition.
    const transitions: Array<{ at: number; workerId: string; state: string }> = [];
    const heartbeatSeenAt: number[] = [];
    const orchState: {
      version: number;
      assignments: Map<string, ShardAssignment | null>;
    } = { version: 0, assignments: new Map() };
    const orchServer = createMeshServer({
      register: (req) => registry.register(req.capability),
      heartbeat: (req) => {
        const res = registry.heartbeat(req);
        heartbeatSeenAt.push(Date.now());
        // The registry owns worker state, but the ASSIGNMENT version
        // is the test's (orchState): registry.setAssignment is never
        // called here, so returning res.assignmentVersion (0) would
        // hide the version-1 assignment and the worker would never
        // load its stage.
        return { ok: res.ok, assignmentVersion: orchState.version };
      },
      getAssignment: (workerId: string) => ({
        version: orchState.version,
        assignment: orchState.assignments.get(workerId) ?? null,
      }),
      getWeightMap: async () => null,
    });
    await new Promise<void>((resolve, reject) => {
      orchServer.once("error", reject);
      orchServer.listen(0, "127.0.0.1", () => resolve());
    });
    const orchUrl = `http://127.0.0.1:${(orchServer.address() as AddressInfo).port}`;
    const sweepTimer = setInterval(() => {
      for (const w of registry.sweep()) {
        transitions.push({ at: Date.now(), workerId: w.workerId, state: w.state });
      }
    }, 250);

    const client = new MeshClient({ timeoutMs: 30_000 });
    const backend = createReferenceBackend();
    let worker: Awaited<ReturnType<typeof startWorker>> | undefined;
    try {
      const manifest = loadManifest(DEMO_100M_DIR, "demo-100m");
      const tokenizer = loadTokenizer(DEMO_100M_DIR);
      const eos = manifest.config.eosTokenId;
      worker = await startWorker({
        orchestratorUrl: orchUrl,
        host: "127.0.0.1",
        port: 0,
        capability: {
          device: "cpu",
          backend: backend.name,
          memoryBytes: 8 * 1024 * 1024 * 1024,
          freeMemoryBytes: 8 * 1024 * 1024 * 1024,
          flopsPerSec: 1e12,
          bandwidthBytesPerSec: 1e9,
          cores: 4,
          roles: ["mixed"],
          maxBatchTokens: 256,
        },
        backend,
        loader: openWeightLoader(DEMO_100M_DIR),
        createCache: async (
          m: ModelManifest,
          vram: VramAccess | null,
        ): Promise<PagedKVCache> => {
          if (vram === null) throw new Error("P7 worker started without a VramAccess");
          const cache = new VramPagedKVCache(
            16,
            4096,
            m.config.numKeyValueHeads,
            m.config.headDim,
            vram,
            { chunkTokens: CHUNK_TOKENS, replicas: 1 },
          );
          liveCacheHolder.current = cache;
          return cache;
        },
        vramAccess: slow,
        // Heartbeat every 500 ms: 12 chances to miss before suspect.
        pollIntervalMs: 500,
      });
      const workerId = worker.workerId;
      const workerUrl = `http://127.0.0.1:${worker.port}`;

      // Single stage holding all 12 layers + embedding + lm_head.
      orchState.assignments.set(workerId, {
        workerId,
        stage: 0,
        layerStart: 0,
        layerEnd: 12,
        hasEmbedding: true,
        hasLmHead: true,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 1024,
        modelId: "demo-100m",
        modelDir: DEMO_100M_DIR,
      });
      orchState.version = 1;
      await waitFor(
        () =>
          logs.lines.some((l) => l.includes("loaded stage 0 (layers 0-12")),
        "stage loaded",
      );
      assert.equal(
        registry.getWorker(workerId)?.state,
        "healthy",
        "worker must be healthy before the generation starts",
      );

      // Short prompt: a 4-token prefill keeps synchronous matmul far
      // below the 6 s suspect window; the long wall time comes from
      // slow vRAM reads during decode, which must NOT trip it.
      const sentence = "The city was quiet in the morning light. ";
      const promptIds = tokenizer.encode(sentence).slice(0, 4);
      assert.equal(promptIds.length, 4);

      const startedAt = Date.now();
      let seqNo = 0;
      let postFlushDecodes = 0;
      let totalDecodes = 0;
      let tokensProduced = 0;
      // Decode until 10 decode steps have run AFTER the first chunk
      // flushed (those steps re-read the flushed chunk from slow
      // vRAM on every layer). A sequence that hits EOS early is
      // abandoned for a fresh one — the cache stats accumulate, and
      // the heartbeat watch spans every attempt.
      while (postFlushDecodes < 10 && seqNo < 3) {
        seqNo += 1;
        const seqId = `seq-p7-${seqNo}`;
        const pre = await client.execute(workerUrl, {
          workerId,
          seqIds: promptIds.map(() => seqId),
          kind: "prefill",
          stage: 0,
          input: { data: [], shape: [0] },
          positions: promptIds.map((_, i) => i),
          tokenIds: [...promptIds],
        });
        assert.equal(pre.isLogits, true);
        let next = argmaxLastRow(pre.output);
        tokensProduced += 1;
        let pos = promptIds.length;
        while (next !== eos && totalDecodes < 40) {
          const flushedBefore = liveCacheHolder.current?.stats().chunkFlushes ?? 0;
          const dec = await client.execute(workerUrl, {
            workerId,
            seqIds: [seqId],
            kind: "decode",
            stage: 0,
            input: { data: [], shape: [0] },
            positions: [pos],
            tokenIds: [next],
          });
          next = argmaxLastRow(dec.output);
          tokensProduced += 1;
          totalDecodes += 1;
          pos += 1;
          // Count only decodes that ran AFTER a chunk had already
          // flushed: those re-read the flushed chunk from slow vRAM
          // on every layer. The decode that causes the flush itself
          // does not count.
          if (flushedBefore > 0) {
            postFlushDecodes += 1;
          }
          if (postFlushDecodes >= 10) break;
        }
        if (postFlushDecodes >= 10) break;
      }
      const wallMs = Date.now() - startedAt;

      const stats = liveCacheHolder.current?.stats();
      console.log(
        `  P7 live: ${tokensProduced} tokens in ${(wallMs / 1000).toFixed(1)} s ` +
          `(${totalDecodes} decodes, ${postFlushDecodes} post-flush), ` +
          `chunkFlushes=${stats?.chunkFlushes}, vramReads=${stats?.vramReads}, ` +
          `slowAccessReads=${slow.reads}, heartbeats seen=${heartbeatSeenAt.length}`,
      );

      // The slow path was actually exercised — otherwise "no suspect"
      // would be vacuous.
      assert.ok(
        (stats?.chunkFlushes ?? 0) >= 1,
        `a chunk must have flushed (got ${stats?.chunkFlushes})`,
      );
      assert.ok(
        (stats?.vramReads ?? 0) > 0,
        `decode must have read flushed KV from slow vRAM (got ${stats?.vramReads})`,
      );
      assert.ok(slow.reads > 0, "the slow access must have served reads");
      // The generation took longer than the suspect timeout — the
      // worker was "busy" past the point where silence would trip it.
      assert.ok(
        wallMs > HEARTBEAT_TIMEOUT_MS,
        `generation wall time ${wallMs} ms must exceed the ${HEARTBEAT_TIMEOUT_MS} ms suspect timeout`,
      );
      // ...yet it never left healthy: no suspect, no dead, no re-plan
      // trigger at any point during the generation. (Capture the
      // dead-only subset BEFORE asserting: assert.deepEqual narrows
      // `transitions` to `[]` afterwards, so a later .filter would
      // see `never`.)
      const deadTransitions = transitions.filter((t) => t.state === "dead");
      assert.deepEqual(
        transitions,
        [],
        `worker must never trip suspect/dead during the slow generation, got ${JSON.stringify(transitions)}`,
      );
      assert.deepEqual(
        deadTransitions,
        [],
        "no re-plan trigger during the slow generation",
      );
      assert.equal(registry.getWorker(workerId)?.state, "healthy");
      // Heartbeats kept arriving on schedule throughout: the largest
      // gap between heartbeats the orchestrator saw must stay well
      // under the 6 s timeout (500 ms interval + scheduling noise).
      let maxGap = 0;
      for (let i = 1; i < heartbeatSeenAt.length; i++) {
        maxGap = Math.max(maxGap, (heartbeatSeenAt[i] as number) - (heartbeatSeenAt[i - 1] as number));
      }
      console.log(`  P7 live: max heartbeat gap ${maxGap} ms over ${heartbeatSeenAt.length} heartbeats`);
      assert.ok(
        maxGap > 0 && maxGap < HEARTBEAT_TIMEOUT_MS,
        `max heartbeat gap ${maxGap} ms must be < ${HEARTBEAT_TIMEOUT_MS} ms`,
      );
    } finally {
      clearInterval(sweepTimer);
      logs.restore();
      if (worker !== undefined) await worker.close().catch(() => undefined);
      await new Promise<void>((resolve) => orchServer.close(() => resolve()));
    }
  },
);
