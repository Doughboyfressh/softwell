/**
 * KV cache in vRAM — survival proof tests
 * (test/kvcache-vram-survival.test.ts).
 *
 * Workstream 4 of the "KV cache in pooled vRAM" phase. These tests pin
 * the survival contract for `VramPagedKVCache`
 * (src/scheduler/kvcache-vram.ts): what survives a DONOR dying, what
 * survives a WORKER dying, and where staged and flushed KV differ.
 *
 * What runs here (cache level, real donors, no engine/worker):
 * the cache is driven directly over a real in-process VirtualRam +
 * two real donor RamNodes served over loopback HTTP (the same shape
 * as test/kvcache-vram-bench.ts), with a recording VramAccess wrapper
 * so the test can read the raw vRAM bytes of a chunk after its owning
 * cache instance is gone. A host BlockPagedKVCache fed the identical
 * appends is the byte ground truth throughout.
 *
 * What ALSO runs here now (end-to-end, at the bottom): the same
 * experiments through real workers serving demo-100m with
 * VramPagedKVCache over the remote KV transport (RemoteVramAccess
 * over the orchestrator's /mesh/vram/* endpoints). Those were
 * skeletons until the async-forward refactor and the remote transport
 * landed (2026-11-23); they are real tests now. Geometry is small on
 * purpose: chunkTokens 32, so a 40-token prompt flushes the first
 * chunk per layer (a >1024-token prompt would be needed at demo
 * geometry, ~4 minutes of prefill per run). Survival semantics do
 * not depend on geometry — the bench file covers demo-100m geometry
 * for performance — and the cache-level tests above use the same
 * 32-token chunks.
 *
 * Harness notes (follow the healing/failure files):
 *  - No Hypervisor here, so there is NO background sweep timer at
 *    all: VirtualRam has no timer, healing is always explicit via
 *    ram.handleNodeLoss(). The race the weights phase fixed (a 60 s
 *    sweep healing before the explicit call) cannot occur.
 *  - killDonor closes the donor's HTTP server and marks it dead in
 *    the provider, exactly like the healing file.
 *  - Geometry is deliberately small (kvDim 32, chunkTokens 32):
 *    a chunk is 8 KiB of K/V inside one 4 MiB vRAM page, so flushes
 *    happen after 32 tokens and every test runs in seconds. The
 *    survival semantics do not depend on geometry (the bench file
 *    covers demo-100m geometry for performance).
 *
 * Proven contract (2026-11-22, this file):
 *  - DONOR death: flushed KV survives. With replicas=2, killing a
 *    donor mid-generation leaves every flushed chunk readable
 *    byte-identical BEFORE healing (failover to the surviving
 *    replica) and after explicit handleNodeLoss (pagesLost 0).
 *    Generation can continue on the same cache afterwards: further
 *    appends land and the full prefix still reads byte-identical
 *    to the host cache.
 *  - DONOR death during a flush: the flush SUCCEEDS on the
 *    surviving replica (VirtualRam.write drops the dead copy and
 *    completes while one copy survives); the flushed chunk reads
 *    back byte-identical, and healing afterwards loses 0 pages.
 *  - WORKER death: the replacement worker CANNOT resume the
 *    sequence from vRAM KV — it must resubmit (re-prefill). The
 *    chunk table (which allocIds belong to which seq/layer) lives
 *    only in the dead worker's cache instance; a fresh cache over
 *    the same vRAM rejects reads with "no blocks allocated". The
 *    flushed BYTES still sit in vRAM byte-identical (proven by raw
 *    reads at the recorded vaddrs), but they are unreachable
 *    without the table. Even flushAll() before the death does not
 *    help: it makes the bytes durable, not the index.
 *  - STAGED vs FLUSHED: staged KV (chunk never filled) survives a
 *    donor death — it lives in host staging, so no donor holds it
 *    — but dies with the worker: it never reached vRAM (the raw
 *    page still reads all zeros) and a fresh cache cannot see it.
 *    A sequence whose KV is entirely staged behaves exactly like
 *    today on worker death. Flushed KV survives donor death and
 *    (as bytes) worker death, but is only resumable with a chunk
 *    table the replacement worker does not have.
 *  - PERMANENT loss is loud: with replicas=1 a donor death loses
 *    pages, and reads of the lost chunks reject with VramKvError
 *    (never silent zeros). Placement quirk pinned by the test:
 *    single-page chunk allocations all land their only copy on the
 *    SAME donor (the candidate rotation and the per-page start
 *    cancel out), so killing that donor loses ALL flushed chunks
 *    while killing the other loses none.
 *
 * Run: node --test test/kvcache-vram-survival.test.ts
 */

import assert from "node:assert/strict";
import * as path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type { AddressInfo } from "node:net";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { startWorker } from "../src/mesh/worker.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  VirtualRamAccess,
  VramKvError,
  VramPagedKVCache,
} from "../src/scheduler/kvcache-vram.ts";
import { RemoteVramAccess } from "../src/scheduler/kvcache-vram-remote.ts";
import { createKvHandlers } from "../src/server/vram-kv.ts";
import { VRAM_PAGE_SIZE_BYTES } from "../src/contracts.ts";
import type { VramAccess } from "../src/contracts.ts";
import type {
  ModelManifest,
  NodeProvider,
  PagedKVCache,
  ShardAssignment,
  WorkerInfo,
} from "../src/contracts.ts";

const BLOCK = 16;
const KV_HEADS = 2;
const HEAD_DIM = 16;
const KV_DIM = KV_HEADS * HEAD_DIM; // 32
const CHUNK_TOKENS = 32;
const TOTAL_BLOCKS = 4096;
const MIB = 1024 * 1024;
/** Bytes of K+V for one chunk (32 tokens x 2 x 32 floats x 4 B). */
const CHUNK_BYTES = CHUNK_TOKENS * 2 * KV_DIM * 4;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Deterministic pseudo-random floats in [-1, 1) (same shape as
 *  test/kvcache-vram.test.ts). */
function pseudoRandom(seed: number, n: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

function prefix(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

function makeWorker(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "test",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    cores: 4,
    roles: ["mixed"],
    maxBatchTokens: 256,
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

class FakeNodes implements NodeProvider {
  list: WorkerInfo[];
  constructor(list: WorkerInfo[]) {
    this.list = list;
  }
  listNodes(): WorkerInfo[] {
    return [...this.list];
  }
  markDead(workerId: string): void {
    const node = this.list.find((n) => n.workerId === workerId);
    if (node !== undefined) node.state = "dead";
  }
  add(node: WorkerInfo): void {
    this.list.push(node);
  }
}

/**
 * VramAccess wrapper that records every allocation (id, vaddr, size)
 * in creation order, so the test can read the raw vRAM bytes of a
 * chunk after the cache instance that owned it is gone (worker-death
 * simulation). The cache itself exposes no allocIds — that absence
 * is exactly what the worker-death tests prove.
 */
class RecordingVramAccess implements VramAccess {
  private readonly inner: VramAccess;
  readonly allocs: Array<{ id: string; vaddr: number; sizeBytes: number }> = [];
  readonly freed: string[] = [];

  constructor(inner: VramAccess) {
    this.inner = inner;
  }

  read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.inner.read(vaddr, length);
  }

  write(vaddr: number, data: Uint8Array): Promise<void> {
    return this.inner.write(vaddr, data);
  }

  async alloc(
    sizeBytes: number,
    opts?: { replicas?: number },
  ): Promise<{ id: string; vaddr: number }> {
    const alloc = await this.inner.alloc(sizeBytes, opts);
    this.allocs.push({ id: alloc.id, vaddr: alloc.vaddr, sizeBytes });
    return alloc;
  }

  async free(id: string): Promise<void> {
    this.freed.push(id);
    return this.inner.free(id);
  }
}

type DonorServer = Awaited<ReturnType<typeof createHwNodeServer>>;

interface Harness {
  provider: FakeNodes;
  ram: VirtualRam;
  access: RecordingVramAccess;
  raw: VirtualRamAccess;
  donors: Map<string, DonorServer>;
  killDonor(id: string): Promise<void>;
  heal(id: string): Promise<{ pagesRecovered: number; pagesLost: number }>;
  close(): Promise<void>;
}

async function bootHarness(replicas: number): Promise<Harness> {
  const provider = new FakeNodes([]);
  const donors = new Map<string, DonorServer>();
  for (const id of ["donor-a", "donor-b"]) {
    // Pure in-memory RamNodes (no spill dir): closing the server
    // loses the donor's pages, like a real donor process dying.
    const node = new RamNode({ capacityBytes: 1 << 30 });
    const server = await createHwNodeServer({ routeTables: [node.routes()] });
    provider.add(makeWorker(id, server.port));
    donors.set(id, server);
  }
  const ram = new VirtualRam({ nodes: provider, defaultReplicas: replicas });
  const raw = new VirtualRamAccess(ram);
  const access = new RecordingVramAccess(raw);
  return {
    provider,
    ram,
    access,
    raw,
    donors,
    async killDonor(id: string): Promise<void> {
      const donor = donors.get(id);
      if (donor === undefined) throw new Error(`no such donor: ${id}`);
      await donor.close().catch(() => undefined);
      provider.markDead(id);
    },
    heal(id: string): Promise<{ pagesRecovered: number; pagesLost: number }> {
      return ram.handleNodeLoss(id);
    },
    async close(): Promise<void> {
      for (const donor of donors.values()) {
        await donor.close().catch(() => undefined);
      }
    },
  };
}

function makeVramCache(access: VramAccess, replicas = 2): VramPagedKVCache {
  return new VramPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM, access, {
    chunkTokens: CHUNK_TOKENS,
    replicas,
  });
}

function makeHostCache(): BlockPagedKVCache {
  return new BlockPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM);
}

function kvFor(layer: number, pos: number): { k: Float32Array; v: Float32Array } {
  return {
    k: pseudoRandom(layer * 10000 + pos * 2 + 1, KV_DIM),
    v: pseudoRandom(layer * 10000 + pos * 2 + 2, KV_DIM),
  };
}

/** Append `count` tokens (positions [start, start+count)) for one
 *  layer to BOTH caches, with the same deterministic K/V. */
async function appendBoth(
  vcache: VramPagedKVCache,
  host: BlockPagedKVCache,
  seq: string,
  layer: number,
  start: number,
  count: number,
): Promise<void> {
  for (let pos = start; pos < start + count; pos++) {
    const { k, v } = kvFor(layer, pos);
    await vcache.append(seq, layer, pos, k, v);
    await host.append(seq, layer, pos, k, v);
  }
}

/** The vRAM cache's full-prefix read for one layer must equal the
 *  host cache's, bit for bit. */
async function assertPrefixEqual(
  vcache: VramPagedKVCache,
  host: BlockPagedKVCache,
  seq: string,
  layer: number,
  n: number,
  what: string,
): Promise<void> {
  const got = await vcache.read(seq, layer, prefix(n));
  const want = await host.read(seq, layer, prefix(n));
  assert.deepEqual(got.keys, want.keys, `${what}: keys layer ${layer} n=${n}`);
  assert.deepEqual(got.values, want.values, `${what}: values layer ${layer} n=${n}`);
}

/**
 * Raw vRAM bytes of one recorded chunk allocation, as floats.
 * `chunkIndex` is the allocation order (fillSeq allocates layer 0's
 * chunks first, then layer 1's, in chunk order within a layer).
 * A fresh Uint8Array copy keeps the Float32Array view 4-byte aligned.
 */
async function rawChunkFloats(
  access: RecordingVramAccess,
  chunkIndex: number,
): Promise<Float32Array> {
  const alloc = access.allocs[chunkIndex];
  if (alloc === undefined) throw new Error(`no recorded alloc #${chunkIndex}`);
  const bytes = await access.read(alloc.vaddr, CHUNK_BYTES);
  const copy = Uint8Array.from(bytes);
  return new Float32Array(copy.buffer, copy.byteOffset, copy.length / 4);
}

/** Expected K/V floats for one token inside its chunk's raw page. */
function assertTokenInRaw(
  raw: Float32Array,
  tokenInChunk: number,
  layer: number,
  pos: number,
  what: string,
): void {
  const { k, v } = kvFor(layer, pos);
  const base = tokenInChunk * 2 * KV_DIM;
  assert.deepEqual(
    raw.subarray(base, base + KV_DIM),
    k,
    `${what}: raw K for layer ${layer} pos ${pos}`,
  );
  assert.deepEqual(
    raw.subarray(base + KV_DIM, base + 2 * KV_DIM),
    v,
    `${what}: raw V for layer ${layer} pos ${pos}`,
  );
}

test(
  "donor death mid-generation: flushed KV reads back byte-identical before and after healing, generation continues identically",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness(2);
    try {
      const vcache = makeVramCache(h.access);
      const host = makeHostCache();
      const seq = "seq-midgen";
      const layers = [0, 1];

      // Prefill 96 tokens per layer (3 chunks flushed per layer) +
      // 2 decode tokens (staged in chunk 3): the sequence is
      // in flight when the donor dies.
      for (const layer of layers) {
        await appendBoth(vcache, host, seq, layer, 0, 96);
        await appendBoth(vcache, host, seq, layer, 96, 2);
      }
      const s = vcache.stats();
      assert.equal(s.chunks, 8, "2 layers x 4 chunks (3 flushed + 1 staged)");
      assert.equal(s.chunkFlushes, 6, "3 flushed chunks per layer");
      assert.equal(s.stagedBytes, 2 * CHUNK_BYTES, "one staged chunk per layer");
      console.log(
        `  in flight: 98 tokens/layer, ${s.chunkFlushes} chunks flushed, ` +
          `${s.stagedBytes / 1024} KiB staged; killing donor-a now`,
      );

      await h.killDonor("donor-a");

      // BEFORE healing: flushed chunks must fail over to the surviving
      // replica inside the same read — byte-identical, no error.
      for (const layer of layers) {
        await assertPrefixEqual(vcache, host, seq, layer, 98, "pre-heal read");
      }
      console.log("  pre-heal: full prefix reads byte-identical from the surviving replica");

      // Generation continues on the SAME cache while the donor is
      // still dead and unhealed: 8 more decode tokens per layer
      // (still staged — no flush is attempted against the dead donor).
      for (const layer of layers) {
        await appendBoth(vcache, host, seq, layer, 98, 8);
      }
      for (const layer of layers) {
        await assertPrefixEqual(vcache, host, seq, layer, 106, "mid-death decode");
      }
      console.log("  generation continued while donor dead: 8 more decode tokens, prefix identical");

      // Explicit healing: 0 pages lost, all 8 chunk pages recovered.
      const healed = await h.heal("donor-a");
      assert.equal(healed.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      assert.equal(
        healed.pagesRecovered,
        8,
        `healing must recover all 8 chunk pages: ${JSON.stringify(healed)}`,
      );
      for (const layer of layers) {
        await assertPrefixEqual(vcache, host, seq, layer, 106, "post-heal read");
      }
      console.log(
        `  healed: pagesRecovered=${healed.pagesRecovered}, pagesLost=0; ` +
          `106-token prefix byte-identical after healing`,
      );

      // And generation still continues identically after healing.
      for (const layer of layers) {
        await appendBoth(vcache, host, seq, layer, 106, 22);
      }
      for (const layer of layers) {
        await assertPrefixEqual(vcache, host, seq, layer, 128, "post-heal decode");
      }
      assert.equal(
        vcache.stats().chunkFlushes,
        8,
        "chunk 3 of each layer flushed at 128 tokens",
      );
      console.log("  post-heal: generation continued to 128 tokens, prefix identical");
    } finally {
      await h.close();
    }
  },
);

test(
  "flush landing while a donor is dead succeeds on the surviving replica",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness(2);
    try {
      const vcache = makeVramCache(h.access);
      const host = makeHostCache();
      const seq = "seq-flush-death";
      // 64 tokens = 2 flushed chunks; chunk 2 staged with 0 tokens.
      await appendBoth(vcache, host, seq, 0, 0, 64);
      assert.equal(vcache.stats().chunkFlushes, 2);

      await h.killDonor("donor-a");
      // Filling chunk 2 (positions 64..95) flushes it WHILE donor-a
      // is dead. VirtualRam.write drops the dead copy and completes
      // on the survivor — the append must NOT reject, and the chunk
      // must not stay staged.
      await appendBoth(vcache, host, seq, 0, 64, 32);
      const s = vcache.stats();
      assert.equal(s.chunkFlushes, 3, "chunk 2 must have flushed during the donor death");
      assert.equal(s.stagedBytes, 0, "no chunk may stay staged after the flush");
      await assertPrefixEqual(vcache, host, seq, 0, 96, "flush-during-death read");
      console.log("  chunk flushed while donor dead; 96-token prefix byte-identical pre-heal");

      // Healing afterwards: the flushed chunk's dead copy was already
      // dropped by the write, so only the 2 older pages count as
      // recovered; nothing is lost.
      const healed = await h.heal("donor-a");
      assert.equal(healed.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      assert.equal(
        healed.pagesRecovered,
        2,
        `only the 2 pre-kill pages still list the dead copy: ${JSON.stringify(healed)}`,
      );
      await assertPrefixEqual(vcache, host, seq, 0, 96, "post-heal read");
      console.log(
        `  healed: pagesRecovered=${healed.pagesRecovered}, pagesLost=0; ` +
          `flushed-during-death chunk still byte-identical`,
      );
    } finally {
      await h.close();
    }
  },
);

test(
  "worker death: replacement worker cannot resume from vRAM KV — it must resubmit (re-prefill)",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness(2);
    try {
      // Worker 1's cache: 96 tokens flushed + 8 staged, one layer.
      let vcache: VramPagedKVCache | null = makeVramCache(h.access);
      const host = makeHostCache();
      const seq = "seq-worker-death";
      await appendBoth(vcache, host, seq, 0, 0, 96);
      await appendBoth(vcache, host, seq, 0, 96, 8);
      assert.equal(vcache.stats().chunkFlushes, 3, "3 chunks flushed before the death");
      assert.equal(h.access.allocs.length, 4, "4 chunk allocations recorded");
      assert.deepEqual(h.access.freed, [], "nothing freed before the death");
      console.log("  worker-1 cache: 104 tokens (3 chunks flushed, 1 staged); killing the worker");

      // --- Worker death: the instance is dropped. No free() runs
      // (a crash does not free), no flushAll() runs, no table is
      // exported — the chunk table dies with the instance. ---
      vcache = null;
      await sleep(0); // let the reference drop; nothing async to await

      // The replacement worker builds a FRESH cache over the SAME
      // vRAM (same VirtualRam, same donors, same allocations).
      const replacement = makeVramCache(h.access);
      await assert.rejects(
        replacement.read(seq, 0, prefix(104)),
        (err: unknown) =>
          err instanceof VramKvError && /no blocks allocated/.test(err.message),
        "replacement cache must reject: it never knew the sequence's chunks",
      );
      console.log("  replacement worker read rejects (no blocks allocated): cannot resume");

      // But the flushed BYTES still sit in vRAM, byte-identical:
      // raw reads at the recorded vaddrs prove the data survived
      // the worker death — what died is the index, not the bytes.
      // Allocs 0-2 are the flushed chunks (layer 0, chunks 0-2).
      for (let c = 0; c < 3; c++) {
        const raw = await rawChunkFloats(h.access, c);
        assertTokenInRaw(raw, 0, 0, c * CHUNK_TOKENS, `flushed chunk ${c} raw bytes`);
        assertTokenInRaw(raw, 31, 0, c * CHUNK_TOKENS + 31, `flushed chunk ${c} raw bytes`);
      }
      console.log("  flushed bytes still in vRAM byte-identical (raw reads at recorded vaddrs)");

      // The staged chunk's page is still all zeros: staged KV never
      // reached vRAM, so it died with the worker at the byte level.
      const stagedRaw = await rawChunkFloats(h.access, 3);
      assert.deepEqual(
        stagedRaw,
        new Float32Array(stagedRaw.length),
        "staged chunk's vRAM page must still be all zeros",
      );
      console.log("  staged chunk's vRAM page is all zeros: staged KV never reached vRAM");

      // The orphaned allocations are still held: no owner, no reaper.
      assert.equal(
        h.ram.stats().pages,
        4,
        "orphaned chunk pages are still allocated after worker death (no owner frees them)",
      );
      assert.deepEqual(h.access.freed, [], "no free ran for the dead worker's chunks");

      // The resubmit path works: the replacement re-prefills the same
      // sequence from scratch and reads it back identically.
      for (let pos = 0; pos < 104; pos++) {
        const { k, v } = kvFor(0, pos);
        await replacement.append(seq, 0, pos, k, v);
      }
      const got = await replacement.read(seq, 0, prefix(104));
      const want = await host.read(seq, 0, prefix(104));
      assert.deepEqual(got.keys, want.keys, "re-prefilled keys must match");
      assert.deepEqual(got.values, want.values, "re-prefilled values must match");
      console.log("  replacement re-prefilled the sequence from scratch: reads identical");
    } finally {
      await h.close();
    }
  },
);

test(
  "staged-only sequence behaves exactly like today on worker death",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness(2);
    try {
      // Short context: 8 tokens, chunk never filled, zero flushes.
      let vcache: VramPagedKVCache | null = makeVramCache(h.access);
      const host = makeHostCache();
      const seq = "seq-short";
      await appendBoth(vcache, host, seq, 0, 0, 8);
      assert.equal(vcache.stats().chunkFlushes, 0, "no chunk may flush for 8 tokens");
      assert.equal(vcache.stats().stagedBytes, CHUNK_BYTES, "the whole sequence is staged");
      assert.equal(vcache.stats().vramBytes, 0, "no bytes in vRAM for a staged-only sequence");
      assert.equal(h.access.allocs.length, 1, "one chunk allocation (eager, zero-filled)");
      await assertPrefixEqual(vcache, host, seq, 0, 8, "pre-death read");
      console.log("  short sequence: 8 tokens, 0 flushes, entirely staged");

      // A donor death does NOT touch staged KV (it lives in host
      // staging): the sequence still reads fine with a donor dead.
      await h.killDonor("donor-a");
      await assertPrefixEqual(vcache, host, seq, 0, 8, "staged read with donor dead");
      console.log("  staged KV survives donor death (host staging, no donor holds it)");

      // But the WORKER dying loses it exactly like today: the
      // replacement cannot see it, and vRAM holds only zeros.
      vcache = null;
      const replacement = makeVramCache(h.access);
      await assert.rejects(
        replacement.read(seq, 0, prefix(8)),
        (err: unknown) =>
          err instanceof VramKvError && /no blocks allocated/.test(err.message),
        "staged-only sequence must be invisible to the replacement worker",
      );
      const raw = await rawChunkFloats(h.access, 0);
      assert.deepEqual(
        raw,
        new Float32Array(raw.length),
        "staged-only chunk's vRAM page must be all zeros",
      );
      console.log("  worker death loses the staged-only sequence (vRAM page all zeros): like today");
    } finally {
      await h.close();
    }
  },
);

test(
  "flushAll before handover makes bytes durable but still not resumable without the chunk table",
  { timeout: 120_000 },
  async () => {
    const h = await bootHarness(2);
    try {
      let vcache: VramPagedKVCache | null = makeVramCache(h.access);
      const host = makeHostCache();
      const seq = "seq-handover";
      await appendBoth(vcache, host, seq, 0, 0, 40);
      assert.equal(vcache.stats().chunkFlushes, 1, "chunk 0 flushed on fill");
      assert.equal(vcache.stats().stagedBytes, CHUNK_BYTES, "chunk 1 staged (8 tokens)");

      // Graceful handover: flush everything, so ALL bytes are in vRAM.
      await vcache.flushAll();
      assert.equal(vcache.stats().chunkFlushes, 2, "flushAll flushed the partial chunk");
      assert.equal(vcache.stats().stagedBytes, 0, "nothing staged after flushAll");
      assert.equal(vcache.stats().vramBytes, 2 * CHUNK_BYTES, "both chunks in vRAM");
      console.log("  flushAll: both chunks durable in vRAM before the handover");

      // The worker still dies (graceful or not): the table dies with it.
      vcache = null;
      const replacement = makeVramCache(h.access);
      await assert.rejects(
        replacement.read(seq, 0, prefix(40)),
        (err: unknown) =>
          err instanceof VramKvError && /no blocks allocated/.test(err.message),
        "flushAll does not hand the chunk table to the replacement",
      );
      // The bytes ARE there this time — even the partial chunk's.
      const raw1 = await rawChunkFloats(h.access, 1);
      assertTokenInRaw(raw1, 0, 0, 32, "flushed partial chunk raw bytes");
      assertTokenInRaw(raw1, 7, 0, 39, "flushed partial chunk raw bytes");
      console.log(
        "  replacement still cannot resume: flushAll durably stored the bytes, " +
          "but the chunk table (allocIds per seq/layer) is what is missing",
      );
    } finally {
      await h.close();
    }
  },
);

test(
  "permanent page loss is loud: replicas=1 donor death makes reads throw VramKvError, never silent zeros",
  { timeout: 120_000 },
  async () => {
    // Placement quirk this test pins (vRAM, not the cache):
    // VirtualRam.placementCandidates rotates the candidate list by
    // `rr`, and alloc()'s per-page start adds `rr` again, so for
    // single-page allocations the two rotations cancel (2*rr mod 2
    // = 0) and EVERY chunk page's only copy lands on the same donor
    // (donor-a here). With replicas=1, killing donor-a therefore
    // loses ALL flushed chunks, not half; killing donor-b loses
    // none. Both sides are asserted below. (Multi-page allocations
    // do stripe: within one alloc, page i lands on rest[i%2].)
    {
      const h = await bootHarness(1);
      try {
        const vcache = makeVramCache(h.access, 1);
        const host = makeHostCache();
        const seq = "seq-permanent-loss";
        await appendBoth(vcache, host, seq, 0, 0, 128);
        assert.equal(vcache.stats().chunkFlushes, 4, "4 chunks flushed");

        await h.killDonor("donor-a");
        const healed = await h.heal("donor-a");
        assert.equal(healed.pagesLost, 4, `donor-a holds all 4 single-page chunks: ${JSON.stringify(healed)}`);
        assert.equal(healed.pagesRecovered, 0, "replicas=1 has no surviving copy to recover from");
        console.log(`  killed donor-a: pagesLost=${healed.pagesLost}, pagesRecovered=0 (replicas=1)`);

        // The full-prefix read must REJECT (loud), never return zeros
        // or partial data for the lost chunks.
        await assert.rejects(
          vcache.read(seq, 0, prefix(128)),
          (err: unknown) =>
            err instanceof VramKvError &&
            /re-prefill/.test(err.message) &&
            /seq-permanent-loss/.test(err.message),
          "read of a prefix with lost pages must reject with VramKvError naming the seq",
        );
        // Every per-chunk read rejects too — no chunk survived to
        // be silently half-right.
        for (let c = 0; c < 4; c++) {
          const range = prefix(CHUNK_TOKENS).map((i) => c * CHUNK_TOKENS + i);
          await assert.rejects(
            vcache.read(seq, 0, range),
            (err: unknown) =>
              err instanceof VramKvError && /re-prefill/.test(err.message),
            `lost chunk ${c} must reject with VramKvError`,
          );
        }
        console.log("  all 4 chunks lost, every read rejects loudly (never silent zeros)");
      } finally {
        await h.close();
      }
    }
    {
      // Flip side: killing the OTHER donor loses nothing, because
      // no chunk page was ever placed on it.
      const h = await bootHarness(1);
      try {
        const vcache = makeVramCache(h.access, 1);
        const host = makeHostCache();
        const seq = "seq-permanent-loss-b";
        await appendBoth(vcache, host, seq, 0, 0, 128);
        await h.killDonor("donor-b");
        const healed = await h.heal("donor-b");
        assert.equal(healed.pagesLost, 0, `donor-b holds no chunk pages: ${JSON.stringify(healed)}`);
        await assertPrefixEqual(vcache, host, seq, 0, 128, "read after donor-b loss");
        console.log("  killed donor-b: pagesLost=0, 128-token prefix still byte-identical");
      } finally {
        await h.close();
      }
    }
  },
);

/* ------------------------------------------------------------------ */
/* End-to-end generation versions (unblocked 2026-11-23 by the       */
/* async-forward refactor + the remote KV transport). The harness    */
/* mirrors test/failure-vram-weights.test.ts test 1 — real donors +  */
/* fake orchestrator + real workers — except the orchestrator also   */
/* serves the KV endpoints (createKvHandlers over the same pooled    */
/* VirtualRam the donors back) and the workers' createCache builds   */
/* VramPagedKVCache over RemoteVramAccess (wrapped in the recording  */
/* wrapper, so the test can name the chunk allocs). Weights come     */
/* from disk here (the spawn loader IS demo-100m, so the assignment  */
/* resolves to it directly); weights-from-vRAM is proven by the      */
/* failure file, and this file's variable under test is the KV cache.*/
/* ------------------------------------------------------------------ */

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DEMO_100M_DIR = path.join(ROOT, "demo", "models", "demo-100m");
/** Tokens per vRAM chunk in the e2e tests: a 40-token prompt flushes chunk 0 per layer. */
const E2E_CHUNK_TOKENS = 32;

async function waitForE2e(
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
      args
        .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
        .join(" "),
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

interface PipelineStage {
  workerId: string;
  url: string;
  stage: number;
}

/**
 * Drive one greedy generation through a pipeline of stages, mirroring
 * the scheduler's execute conventions (one row per token; stage 0
 * embeds tokenIds, later stages take the previous stage's output;
 * prefill covers the whole prompt, decode feeds one token at a time).
 * Returns the sampled token ids (excluding the prompt).
 */
async function runPipeline(
  client: MeshClient,
  stages: PipelineStage[],
  promptIds: number[],
  maxNewTokens: number,
  seqId: string,
  eosTokenId: number,
): Promise<number[]> {
  const rows = promptIds.length;
  const seqIds = promptIds.map(() => seqId);
  const positions = promptIds.map((_, i) => i);
  let out: { data: number[]; shape: number[] } | null = null;
  for (let s = 0; s < stages.length; s++) {
    const st = stages[s] as PipelineStage;
    const resp = await client.execute(st.url, {
      workerId: st.workerId,
      seqIds: [...seqIds],
      kind: "prefill",
      stage: st.stage,
      input: out ?? { data: [], shape: [0] },
      positions: [...positions],
      ...(s === 0 ? { tokenIds: [...promptIds] } : {}),
    });
    if (s === stages.length - 1) {
      assert.equal(resp.isLogits, true, "last stage must return logits");
    }
    out = resp.output;
  }
  if (out === null) throw new Error("pipeline produced no output");
  const tokens: number[] = [];
  let next = argmaxLastRow(out);
  tokens.push(next);
  let pos = rows;
  while (tokens.length < maxNewTokens && next !== eosTokenId) {
    let stepOut: { data: number[]; shape: number[] } | null = null;
    for (let s = 0; s < stages.length; s++) {
      const st = stages[s] as PipelineStage;
      const resp = await client.execute(st.url, {
        workerId: st.workerId,
        seqIds: [seqId],
        kind: "decode",
        stage: st.stage,
        input: stepOut ?? { data: [], shape: [0] },
        positions: [pos],
        ...(s === 0 ? { tokenIds: [next] } : {}),
      });
      stepOut = resp.output;
    }
    if (stepOut === null) throw new Error("pipeline decode produced no output");
    next = argmaxLastRow(stepOut);
    tokens.push(next);
    pos += 1;
  }
  return tokens;
}

function makeAssignment(
  workerId: string,
  opts: {
    stage: number;
    layerStart: number;
    layerEnd: number;
    hasEmbedding: boolean;
    hasLmHead: boolean;
    modelId: string;
    modelDir: string;
  },
): ShardAssignment {
  return {
    workerId,
    stage: opts.stage,
    layerStart: opts.layerStart,
    layerEnd: opts.layerEnd,
    hasEmbedding: opts.hasEmbedding,
    hasLmHead: opts.hasLmHead,
    tensorParallelRank: 0,
    tensorParallelSize: 1,
    experts: [],
    estimatedMemoryBytes: 1024,
    modelId: opts.modelId,
    modelDir: opts.modelDir,
  };
}

type E2eWorkerHandle = Awaited<ReturnType<typeof startWorker>>;

interface E2eWorker {
  handle: E2eWorkerHandle;
  url: string;
  recording: RecordingVramAccess;
}

interface E2eHarness extends Harness {
  orchServer: Awaited<ReturnType<typeof createMeshServer>>;
  orchUrl: string;
  orchState: {
    version: number;
    assignments: Map<string, ShardAssignment | null>;
    nextWorker: number;
  };
  /** Start one real worker serving demo-100m with vRAM KV. */
  startVramWorker(): Promise<E2eWorker>;
  /** The 40-token prompt (2 flushed chunks per layer: chunk 0 fills). */
  promptIds: number[];
  eosTokenId: number;
  manifest: ModelManifest;
  /** Bytes of K+V in one e2e chunk (32 tokens). */
  chunkBytes: number;
}

async function bootE2e(): Promise<E2eHarness> {
  const h = await bootHarness(2);
  const manifest = loadManifest(DEMO_100M_DIR, "demo-100m");
  const eosTokenId = manifest.config.eosTokenId as number;
  const tokenizer = loadTokenizer(DEMO_100M_DIR);
  const sentence =
    "The city was quiet in the morning light, and the river moved slowly past the old stone bridge. ";
  let promptIds: number[] = [];
  while (promptIds.length < 40) promptIds = [...promptIds, ...tokenizer.encode(sentence)];
  promptIds = promptIds.slice(0, 40);
  const chunkBytes =
    E2E_CHUNK_TOKENS *
    2 *
    manifest.config.numKeyValueHeads *
    manifest.config.headDim *
    4;

  const kvHandlers = createKvHandlers(h.ram);
  const orchState: E2eHarness["orchState"] = {
    version: 0,
    assignments: new Map(),
    nextWorker: 1,
  };
  const orchServer = createMeshServer({
    register: () => ({
      workerId: `worker-${orchState.nextWorker++}`,
      heartbeatIntervalMs: 50,
    }),
    heartbeat: () => ({ ok: true, assignmentVersion: orchState.version }),
    getAssignment: (workerId: string) => ({
      version: orchState.version,
      assignment: orchState.assignments.get(workerId) ?? null,
    }),
    // Nothing staged: workers fall back to their disk spawn loader.
    getWeightMap: async () => null,
    ...kvHandlers,
  });
  await new Promise<void>((resolve, reject) => {
    orchServer.once("error", reject);
    orchServer.listen(0, "127.0.0.1", () => resolve());
  });
  const orchUrl = `http://127.0.0.1:${(orchServer.address() as AddressInfo).port}`;
  const backend = createReferenceBackend();

  const startVramWorker = async (): Promise<E2eWorker> => {
    const spawnLoader = openWeightLoader(DEMO_100M_DIR);
    const recording = new RecordingVramAccess(new RemoteVramAccess(orchUrl));
    const createCache = async (
      m: ModelManifest,
      vram: VramAccess | null,
    ): Promise<PagedKVCache> => {
      if (vram === null) {
        throw new Error("e2e vRAM worker started without a VramAccess");
      }
      // Same probe as the production CLI factory (src/mesh/worker.ts
      // main()): one page alloc + free, so an unreachable vRAM fails
      // the stage load loudly instead of failing every sequence.
      const probe = await vram.alloc(VRAM_PAGE_SIZE_BYTES, { replicas: 2 });
      await vram.free(probe.id);
      return new VramPagedKVCache(
        16,
        1024,
        m.config.numKeyValueHeads,
        m.config.headDim,
        vram,
        { chunkTokens: E2E_CHUNK_TOKENS, replicas: 2 },
      );
    };
    const handle = await startWorker({
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
      loader: spawnLoader,
      createCache,
      vramAccess: recording,
      pollIntervalMs: 50,
    });
    return { handle, url: `http://127.0.0.1:${handle.port}`, recording };
  };

  return {
    ...h,
    orchServer,
    orchUrl,
    orchState,
    startVramWorker,
    promptIds,
    eosTokenId,
    manifest,
    chunkBytes,
    async close(): Promise<void> {
      await new Promise<void>((resolve) => orchServer.close(() => resolve()));
      await h.close();
    },
  };
}

/** Chunk allocs of one worker that are still live (not freed). */
function liveChunkAllocs(w: E2eWorker, chunkBytes: number): Array<{ id: string; vaddr: number }> {
  return w.recording.allocs.filter(
    (a) => a.sizeBytes === chunkBytes && !w.recording.freed.includes(a.id),
  );
}

test(
  "e2e: donor death mid-generation through the worker — generation completes identically",
  { timeout: 300_000 },
  async () => {
    const logs = captureLogs();
    const h = await bootE2e();
    const client = new MeshClient({ timeoutMs: 30_000 });
    let w1: E2eWorker | undefined;
    let w2: E2eWorker | undefined;
    try {
      w1 = await h.startVramWorker();
      w2 = await h.startVramWorker();
      const worker1 = w1;
      const worker2 = w2;
      console.log(
        `  workers up: ${worker1.handle.workerId} (stage 0) and ${worker2.handle.workerId} (stage 1), vRAM KV`,
      );

      // 2-stage plan, version 1.
      h.orchState.assignments.set(
        worker1.handle.workerId,
        makeAssignment(worker1.handle.workerId, {
          stage: 0, layerStart: 0, layerEnd: 6,
          hasEmbedding: true, hasLmHead: false,
          modelId: "demo-100m", modelDir: DEMO_100M_DIR,
        }),
      );
      h.orchState.assignments.set(
        worker2.handle.workerId,
        makeAssignment(worker2.handle.workerId, {
          stage: 1, layerStart: 6, layerEnd: 12,
          hasEmbedding: false, hasLmHead: true,
          modelId: "demo-100m", modelDir: DEMO_100M_DIR,
        }),
      );
      h.orchState.version = 1;
      await waitForE2e(
        () =>
          logs.lines.some((l) =>
            l.includes("loaded stage 0 (layers 0-6") && l.includes("via disk weights"),
          ) &&
          logs.lines.some((l) =>
            l.includes("loaded stage 1 (layers 6-12") && l.includes("via disk weights"),
          ),
        "both stages loaded",
      );

      const twoStage: PipelineStage[] = [
        { workerId: worker1.handle.workerId, url: worker1.url, stage: 0 },
        { workerId: worker2.handle.workerId, url: worker2.url, stage: 1 },
      ];
      const tokensRef = await runPipeline(
        client, twoStage, h.promptIds, 6, "seq-ref", h.eosTokenId,
      );
      assert.ok(tokensRef.length > 0, "reference generation must produce tokens");
      console.log(`  reference generation (vRAM KV): ${tokensRef.length} tokens [${tokensRef.join(", ")}]`);

      // PROOF the KV pages live in pooled vRAM, not worker-local memory:
      // the stage-1 worker's cache allocated 2 chunks per layer (chunk 0
      // flushed at 32 tokens, chunk 1 staged) through the remote access,
      // and the pooled ram holds those pages.
      const kvAllocs = liveChunkAllocs(worker2, h.chunkBytes);
      assert.equal(
        kvAllocs.length,
        12,
        `stage-1 cache must hold 2 chunks x 6 layers in vRAM, got ${kvAllocs.length}`,
      );
      assert.ok(
        h.ram.stats().usedBytes >= kvAllocs.length * 1024 * 1024,
        `pooled vRAM must hold the KV pages (used ${h.ram.stats().usedBytes})`,
      );
      console.log(
        `  KV lives in pooled vRAM: ${kvAllocs.length} chunk allocs, ` +
          `${(h.ram.stats().usedBytes / 1024 / 1024).toFixed(1)} MiB used`,
      );

      // --- Mid-generation kill. ---
      // Second generation: prefill + 2 decode tokens, then kill the
      // donor holding KV replicas mid-flight. The donor is NOT a
      // worker here (donors are the RamNode servers) — the workers
      // stay alive, only their KV pages lose a replica.
      const seqMid = "seq-mid";
      const pre0 = await client.execute(worker1.url, {
        workerId: worker1.handle.workerId,
        seqIds: h.promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 0,
        input: { data: [], shape: [0] },
        positions: h.promptIds.map((_, i) => i),
        tokenIds: [...h.promptIds],
      });
      const pre1 = await client.execute(worker2.url, {
        workerId: worker2.handle.workerId,
        seqIds: h.promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 1,
        input: pre0.output,
        positions: h.promptIds.map((_, i) => i),
      });
      assert.equal(pre1.isLogits, true);
      let next = argmaxLastRow(pre1.output);
      let pos = h.promptIds.length;
      const tokensMid: number[] = [next];
      const decodeOne = async (tok: number, p: number): Promise<number> => {
        const d0 = await client.execute(worker1.url, {
          workerId: worker1.handle.workerId,
          seqIds: [seqMid],
          kind: "decode",
          stage: 0,
          input: { data: [], shape: [0] },
          positions: [p],
          tokenIds: [tok],
        });
        const d1 = await client.execute(worker2.url, {
          workerId: worker2.handle.workerId,
          seqIds: [seqMid],
          kind: "decode",
          stage: 1,
          input: d0.output,
          positions: [p],
        });
        return argmaxLastRow(d1.output);
      };
      next = await decodeOne(next, pos);
      tokensMid.push(next);
      pos += 1;
      next = await decodeOne(next, pos);
      tokensMid.push(next);
      pos += 1;
      console.log("  generation in flight (2 decode tokens done); killing donor-a now");

      await h.killDonor("donor-a");

      // Generation continues on the SAME workers while the donor is
      // still dead and unhealed: every decode step re-reads the full
      // prefix, including the flushed chunk 0 of every layer, so each
      // step must fail over to the surviving replica inside the read.
      next = await decodeOne(next, pos);
      tokensMid.push(next);
      pos += 1;
      next = await decodeOne(next, pos);
      tokensMid.push(next);
      pos += 1;
      console.log("  generation continued while donor dead: 2 more decode tokens");

      // Explicit healing: 0 pages lost; the flushed chunks re-replicate.
      const healed = await h.heal("donor-a");
      assert.equal(healed.pagesLost, 0, `healing lost pages: ${JSON.stringify(healed)}`);
      assert.ok(
        healed.pagesRecovered > 0,
        `healing must recover pages: ${JSON.stringify(healed)}`,
      );
      console.log(
        `  healed: pagesRecovered=${healed.pagesRecovered}, pagesLost=0`,
      );

      // Finish the generation (6 tokens total) and compare.
      while (tokensMid.length < 6 && next !== h.eosTokenId) {
        next = await decodeOne(next, pos);
        tokensMid.push(next);
        pos += 1;
      }
      assert.deepEqual(
        tokensMid,
        tokensRef,
        "generation through a donor death must produce the same tokens as the uninterrupted reference",
      );
      console.log(
        `  post-kill generation: ${tokensMid.length} tokens, identical to the reference`,
      );
    } finally {
      logs.restore();
      if (w2 !== undefined) await w2.handle.close().catch(() => undefined);
      if (w1 !== undefined) await w1.handle.close().catch(() => undefined);
      await h.close();
    }
  },
);

test(
  "e2e: worker death — replacement worker cannot resume from vRAM KV, resubmission produces the reference tokens",
  { timeout: 300_000 },
  async () => {
    const logs = captureLogs();
    const h = await bootE2e();
    const client = new MeshClient({ timeoutMs: 30_000 });
    let w1: E2eWorker | undefined;
    let w2: E2eWorker | undefined;
    let w3: E2eWorker | undefined;
    let victimHandle: E2eWorkerHandle | undefined;
    try {
      w1 = await h.startVramWorker();
      w2 = await h.startVramWorker();
      const worker1 = w1;
      const worker2 = w2;
      h.orchState.assignments.set(
        worker1.handle.workerId,
        makeAssignment(worker1.handle.workerId, {
          stage: 0, layerStart: 0, layerEnd: 6,
          hasEmbedding: true, hasLmHead: false,
          modelId: "demo-100m", modelDir: DEMO_100M_DIR,
        }),
      );
      h.orchState.assignments.set(
        worker2.handle.workerId,
        makeAssignment(worker2.handle.workerId, {
          stage: 1, layerStart: 6, layerEnd: 12,
          hasEmbedding: false, hasLmHead: true,
          modelId: "demo-100m", modelDir: DEMO_100M_DIR,
        }),
      );
      h.orchState.version = 1;
      await waitForE2e(
        () =>
          logs.lines.some((l) =>
            l.includes("loaded stage 0 (layers 0-6") && l.includes("via disk weights"),
          ) &&
          logs.lines.some((l) =>
            l.includes("loaded stage 1 (layers 6-12") && l.includes("via disk weights"),
          ),
        "both stages loaded",
      );

      const twoStage: PipelineStage[] = [
        { workerId: worker1.handle.workerId, url: worker1.url, stage: 0 },
        { workerId: worker2.handle.workerId, url: worker2.url, stage: 1 },
      ];
      const tokensRef = await runPipeline(
        client, twoStage, h.promptIds, 6, "seq-ref", h.eosTokenId,
      );
      assert.ok(tokensRef.length > 0, "reference generation must produce tokens");
      console.log(`  reference generation (vRAM KV): ${tokensRef.length} tokens [${tokensRef.join(", ")}]`);

      // Mid-generation run on a second sequence: prefill + 2 decodes.
      const seqMid = "seq-mid";
      const pre0 = await client.execute(worker1.url, {
        workerId: worker1.handle.workerId,
        seqIds: h.promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 0,
        input: { data: [], shape: [0] },
        positions: h.promptIds.map((_, i) => i),
        tokenIds: [...h.promptIds],
      });
      const pre1 = await client.execute(worker2.url, {
        workerId: worker2.handle.workerId,
        seqIds: h.promptIds.map(() => seqMid),
        kind: "prefill",
        stage: 1,
        input: pre0.output,
        positions: h.promptIds.map((_, i) => i),
      });
      let next = argmaxLastRow(pre1.output);
      let pos = h.promptIds.length;
      const decodeVia = async (
        w1url: string, w1id: string,
        w2url: string, w2id: string,
        tok: number, p: number,
      ): Promise<number> => {
        const d0 = await client.execute(w1url, {
          workerId: w1id,
          seqIds: [seqMid],
          kind: "decode",
          stage: 0,
          input: { data: [], shape: [0] },
          positions: [p],
          tokenIds: [tok],
        });
        const d1 = await client.execute(w2url, {
          workerId: w2id,
          seqIds: [seqMid],
          kind: "decode",
          stage: 1,
          input: d0.output,
          positions: [p],
        });
        return argmaxLastRow(d1.output);
      };
      next = await decodeVia(worker1.url, worker1.handle.workerId, worker2.url, worker2.handle.workerId, next, pos);
      pos += 1;
      next = await decodeVia(worker1.url, worker1.handle.workerId, worker2.url, worker2.handle.workerId, next, pos);
      pos += 1;
      console.log("  generation in flight (2 decode tokens done); killing worker-2 now");

      // ABANDON worker-2 (do not close it): close() would free its
      // vRAM chunks (graceful shutdown), while a real crash orphans
      // them. The replacement's fresh cache cannot read them either
      // way — the chunk table died with the worker — and abandoning
      // models the crash: the bytes stay in vRAM, the index is gone.
      // The victim handle is kept in `victimHandle` so the finally
      // block can still close it at teardown.
      const victim = worker2;
      victimHandle = victim.handle;
      w2 = undefined;
      const victimChunks = liveChunkAllocs(victim, h.chunkBytes);
      assert.ok(
        victimChunks.length > 0,
        "victim must hold live KV chunks before the kill",
      );
      // The flushed bytes survive the worker's death (raw vRAM read):
      // byte-identity of surviving bytes is proven at cache level in
      // this file ("worker death"); here the check is that the bytes
      // are still there at all — a freed chunk would read zeros.
      const first = victimChunks[0] as { id: string; vaddr: number };
      const rawBytes = await h.ram.read(first.vaddr, 1024);
      assert.ok(
        rawBytes.some((b) => b !== 0),
        "the victim's flushed chunk bytes must still sit in vRAM after the worker dies",
      );
      console.log(
        `  victim abandoned with ${victimChunks.length} live KV chunks; ` +
          `its flushed bytes still read back from vRAM`,
      );

      // Replacement worker registers, takes the stage-1 assignment.
      w3 = await h.startVramWorker();
      const worker3 = w3;
      h.orchState.assignments.set(
        worker3.handle.workerId,
        makeAssignment(worker3.handle.workerId, {
          stage: 1, layerStart: 6, layerEnd: 12,
          hasEmbedding: false, hasLmHead: true,
          modelId: "demo-100m", modelDir: DEMO_100M_DIR,
        }),
      );
      h.orchState.version = 2;
      await waitForE2e(
        () =>
          logs.lines.some((l) =>
            l.includes("loaded stage 1 (layers 6-12") &&
            l.includes("assignment v2") &&
            l.includes("via disk weights"),
          ),
        "replacement loaded stage 1",
      );

      // The replacement CANNOT resume the sequence: its fresh cache
      // has no K/V for seq-mid, so the decode must reject loudly
      // (never silently zero-fill or wrong-token). The message is the
      // unwritten-blocks check, not "no blocks allocated": the
      // decode's append allocates the blocks for positions 0..pos
      // first, so the prefix read fails because position 0 was never
      // written in this cache. The pure-read "no blocks allocated"
      // case is pinned at cache level in this file ("worker death").
      await assert.rejects(
        decodeVia(
          worker1.url, worker1.handle.workerId,
          worker3.url, worker3.handle.workerId,
          next, pos,
        ),
        /no K\/V written yet for seq seq-mid layer 6 position 0/,
        "replacement decode of a dead worker's sequence must reject",
      );
      console.log("  replacement correctly rejects the dead sequence (no K/V written yet)");

      // Resubmit the generation from scratch on the replacement: the
      // tokens must equal the uninterrupted reference.
      const healedTwoStage: PipelineStage[] = [
        { workerId: worker1.handle.workerId, url: worker1.url, stage: 0 },
        { workerId: worker3.handle.workerId, url: worker3.url, stage: 1 },
      ];
      const tokensResubmitted = await runPipeline(
        client, healedTwoStage, h.promptIds, 6, "seq-resubmit", h.eosTokenId,
      );
      assert.deepEqual(
        tokensResubmitted,
        tokensRef,
        "resubmitted generation on the replacement worker must produce the same tokens as the reference",
      );
      console.log(
        `  resubmitted generation: ${tokensResubmitted.length} tokens, identical to the reference`,
      );
    } finally {
      logs.restore();
      if (w3 !== undefined) await w3.handle.close().catch(() => undefined);
      if (w2 !== undefined) await w2.handle.close().catch(() => undefined);
      if (w1 !== undefined) await w1.handle.close().catch(() => undefined);
      await victimHandle?.close().catch(() => undefined);
      await h.close();
    }
  },
);
