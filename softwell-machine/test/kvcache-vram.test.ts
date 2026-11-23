/**
 * vRAM-backed paged KV cache — correctness tests.
 * Run with: node test/kvcache-vram.test.ts (exit 0 on pass).
 *
 * All tests run against a Map-backed fake VramAccess (byte-addressed,
 * zero latency, optional capacity limit for OOM) plus a fault-injecting
 * wrapper that throws on read. The real-Transport path (VirtualRam +
 * donor HTTP servers) is measured in test/kvcache-vram-bench.ts, not
 * here, so this file stays fast and hermetic.
 *
 * Coverage:
 *  - allocate/append/read round-trip matches BlockPagedKVCache exactly
 *    (same random K/V, same reads, bit-for-bit equal).
 *  - Multi-chunk growth: small chunkTokens forces flushes and
 *    multi-chunk reads; out-of-order append to a flushed chunk
 *    (read-modify-write path) returns the rewritten value.
 *  - free/reuse: freed blocks are reusable and a freed sequence reads
 *    as unallocated; a new sequence reuses the same address range.
 *  - Multi-layer and multi-sequence isolation: interleaved appends
 *    never cross-talk.
 *  - Out-of-memory: a capacity-limited fake rejects with VramKvError
 *    and leaves the cache usable afterwards; the block cap also rejects.
 *  - Fault injection: a read failure rejects (never silently returns
 *    zeros) and a failed flush rejects while keeping staged data.
 *  - Unwritten reads: untouched chunk throws; unwritten position in a
 *    touched chunk reads as zeros.
 *  - Counters: chunk flushes and vRAM bytes match the chunking model.
 *  - 11. Block ids: globally unique across sequences and layers,
 *    stable across re-allocate, fresh (never reused) after free;
 *    the shrinking re-allocate divergence from the host is pinned.
 *  - 12. Block-granular unwritten reads: a read in an untouched block
 *    of a touched chunk throws (host parity), including after
 *    flushAll() on an untouched chunk.
 *  - 13. Concurrency: interleaved appends across sequences/layers on
 *    one cache stay correct; 32 concurrent appends to one chunk
 *    create exactly one chunk; concurrent allocate()s do not
 *    double-allocate.
 *  - 14. Free-then-reallocate: freed blocks are reusable, a new
 *    sequence gets fresh block ids, usedBlocks returns to 0.
 *  - 15. Allocate beyond the block cap rejects (VramKvError, no
 *    partial growth, cache stays usable, never falls back to host);
 *    a totalBlocks smaller than one chunk is a constructor error.
 *  - 16. Read after a chunk flushed mid-sequence: prefix reads stay
 *    correct across flushed + staged chunks as the sequence grows.
 *  - 17. Append-after-free: the same seq id starts clean, with no
 *    stale K/V surviving the free.
 *  - 18. free() never rejects: with a backend whose free always
 *    fails, free() resolves, stats().freeFailures counts the
 *    failures, and the cache stays usable.
 *  - 19. Stats scripted verification: every counter
 *    (chunks/stagedBytes/vramBytes/vramReads/vramWrites/bytesRead/
 *    bytesWritten/chunkFlushes/rmwFetches/freeFailures) checked
 *    against an exact scripted op sequence, including
 *    usedBlocks() vs logicalBlocks() and short-read rejection.
 */
import assert from "node:assert/strict";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  VramKvError,
  VramPagedKVCache,
} from "../src/scheduler/kvcache-vram.ts";
import type { VramAccess } from "../src/contracts.ts";

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/** Byte-addressed Map backend. Alloc rounds up to whole 4 MiB pages
 *  (like real vRAM) so OOM accounting is honest; `capacityBytes`
 *  limits total allocated bytes (undefined = unbounded). */
class MapVramAccess implements VramAccess {
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();
  private bump = 0;
  private nextId = 0;
  private usedBytes = 0;
  readonly capacityBytes: number | undefined;
  /** Allocations in creation order (for the reuse test). */
  readonly allocLog: Array<{ id: string; vaddr: number }> = [];

  constructor(capacityBytes?: number) {
    this.capacityBytes = capacityBytes;
  }

  private find(vaddr: number, length: number): { vaddr: number; data: Uint8Array } {
    for (const alloc of this.allocs.values()) {
      if (vaddr >= alloc.vaddr && vaddr + length <= alloc.vaddr + alloc.data.length) {
        return alloc;
      }
    }
    throw new Error(`MapVramAccess: range [${vaddr}, ${vaddr + length}) outside any allocation`);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const alloc = this.find(vaddr, length);
    return new Uint8Array(alloc.data.subarray(vaddr - alloc.vaddr, vaddr - alloc.vaddr + length));
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    const alloc = this.find(vaddr, data.length);
    alloc.data.set(data, vaddr - alloc.vaddr);
  }

  async alloc(sizeBytes: number): Promise<{ id: string; vaddr: number }> {
    const PAGE = 4 * 1024 * 1024;
    const rounded = Math.ceil(sizeBytes / PAGE) * PAGE;
    if (this.capacityBytes !== undefined && this.usedBytes + rounded > this.capacityBytes) {
      throw new Error(
        `MapVramAccess: out of memory (used=${this.usedBytes}, requested=${rounded}, capacity=${this.capacityBytes})`,
      );
    }
    const id = `alloc-${this.nextId++}`;
    const vaddr = this.bump;
    this.bump += rounded;
    this.usedBytes += rounded;
    this.allocs.set(id, { vaddr, data: new Uint8Array(rounded) });
    this.allocLog.push({ id, vaddr });
    return { id, vaddr };
  }

  async free(id: string): Promise<void> {
    const alloc = this.allocs.get(id);
    if (alloc === undefined) throw new Error(`MapVramAccess: unknown allocation ${id}`);
    this.allocs.delete(id);
    this.usedBytes -= alloc.data.length;
  }

  allocatedBytes(): number {
    return this.usedBytes;
  }
}

/** Wraps another VramAccess and throws on reads (or on the Nth read)
 *  to prove error propagation. */
class FaultyVramAccess implements VramAccess {
  private readonly inner: VramAccess;
  /** When set, only the Nth read (1-based) throws; otherwise all reads throw. */
  private readonly failReadN: number | undefined;
  private reads = 0;
  failWrites = false;

  constructor(inner: VramAccess, failReadN?: number) {
    this.inner = inner;
    this.failReadN = failReadN;
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    this.reads += 1;
    if (this.failReadN === undefined || this.reads === this.failReadN) {
      throw new Error(`FaultyVramAccess: injected read failure (read #${this.reads})`);
    }
    return this.inner.read(vaddr, length);
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    if (this.failWrites) throw new Error("FaultyVramAccess: injected write failure");
    return this.inner.write(vaddr, data);
  }

  alloc(sizeBytes: number, opts?: { replicas?: number }): Promise<{ id: string; vaddr: number }> {
    return this.inner.alloc(sizeBytes, opts);
  }

  free(id: string): Promise<void> {
    return this.inner.free(id);
  }
}

/** Wraps another VramAccess and always throws on free, to prove
 *  free() never rejects (failures are counted, not thrown). */
class FailingFreeVramAccess implements VramAccess {
  private readonly inner: VramAccess;

  constructor(inner: VramAccess) {
    this.inner = inner;
  }

  read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.inner.read(vaddr, length);
  }

  write(vaddr: number, data: Uint8Array): Promise<void> {
    return this.inner.write(vaddr, data);
  }

  alloc(sizeBytes: number, opts?: { replicas?: number }): Promise<{ id: string; vaddr: number }> {
    return this.inner.alloc(sizeBytes, opts);
  }

  async free(id: string): Promise<void> {
    throw new Error(`FailingFreeVramAccess: injected free failure for ${id}`);
  }
}

/** Wraps another VramAccess and returns one byte short on every read,
 *  to prove short reads reject instead of serving partial data. */
class ShortReadVramAccess implements VramAccess {
  private readonly inner: VramAccess;

  constructor(inner: VramAccess) {
    this.inner = inner;
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const data = await this.inner.read(vaddr, length);
    return data.subarray(0, Math.max(0, data.length - 1));
  }

  write(vaddr: number, data: Uint8Array): Promise<void> {
    return this.inner.write(vaddr, data);
  }

  alloc(sizeBytes: number, opts?: { replicas?: number }): Promise<{ id: string; vaddr: number }> {
    return this.inner.alloc(sizeBytes, opts);
  }

  free(id: string): Promise<void> {
    return this.inner.free(id);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random floats in [-1, 1). */
function pseudoRandom(seed: number, n: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

const BLOCK = 16;
const KV_HEADS = 2;
const HEAD_DIM = 16;
const KV_DIM = KV_HEADS * HEAD_DIM; // 32
const TOTAL_BLOCKS = 4096;

function makeCache(vram: VramAccess, chunkTokens = 64): VramPagedKVCache {
  return new VramPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM, vram, {
    chunkTokens,
    replicas: 2,
  });
}

function prefix(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

async function main(): Promise<void> {
  // --- 1. Round-trip matches the host cache exactly ----------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram);
    const host = new BlockPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM);
    const seq = "seq-roundtrip";
    const layers = [0, 1, 2];
    const tokens = 100; // spans two 64-token chunks
    for (const layer of layers) {
      for (let pos = 0; pos < tokens; pos++) {
        const k = pseudoRandom(layer * 10000 + pos * 2 + 1, KV_DIM);
        const v = pseudoRandom(layer * 10000 + pos * 2 + 2, KV_DIM);
        await vcache.append(seq, layer, pos, k, v);
        await host.append(seq, layer, pos, k, v);
      }
    }
    for (const layer of layers) {
      for (const n of [1, 16, 17, 64, 65, 100]) {
        const got = await vcache.read(seq, layer, prefix(n));
        const want = await host.read(seq, layer, prefix(n));
        assert.deepEqual(got.keys, want.keys, `round-trip keys layer ${layer} n=${n}`);
        assert.deepEqual(got.values, want.values, `round-trip values layer ${layer} n=${n}`);
      }
      // Non-contiguous + out-of-order reads must also match.
      const positions = [99, 0, 63, 64, 17, 42];
      const got = await vcache.read(seq, layer, positions);
      const want = await host.read(seq, layer, positions);
      assert.deepEqual(got.keys, want.keys, `round-trip keys layer ${layer} scattered`);
      assert.deepEqual(got.values, want.values, `round-trip values layer ${layer} scattered`);
    }
    // Block tables: same length as the host's, ids unique within a
    // layer, and stable across repeated allocate() calls. Exact ids
    // differ by design — the host pops its free-list LIFO (so a fresh
    // cache hands out 4095, 4094, ...) while the vRAM cache numbers
    // chunk slots in order. The contract only promises a stable
    // logical->physical mapping, not specific ids.
    for (const layer of layers) {
      const vTable = await vcache.allocate(seq, layer, tokens);
      const hTable = await host.allocate(seq, layer, tokens);
      assert.equal(vTable.length, hTable.length, `block table length layer ${layer}`);
      assert.equal(new Set(vTable).size, vTable.length, `block ids unique layer ${layer}`);
      assert.deepEqual(await vcache.allocate(seq, layer, tokens), vTable, `block table stable layer ${layer}`);
    }
    assert.equal(vcache.usedBlocks(), 3 * 8, "usedBlocks counts whole reserved chunks");
    assert.equal(host.usedBlocks(), 3 * 7, "host usedBlocks counts needed blocks");
    console.log("1. round-trip matches BlockPagedKVCache exactly ok");
  }

  // --- 2. Multi-chunk growth + out-of-order rewrite of a flushed chunk --
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32); // 32 tokens/chunk = 2 blocks
    const seq = "seq-chunks";
    assert.equal(vcache.getChunkTokens(), 32);
    for (let pos = 0; pos < 100; pos++) {
      await vcache.append(seq, 0, pos, pseudoRandom(pos * 2 + 1, KV_DIM), pseudoRandom(pos * 2 + 2, KV_DIM));
    }
    // 100 tokens -> 4 chunks; the first 3 flushed on fill, the last staged.
    const stats = vcache.stats();
    assert.equal(stats.chunks, 4, "expected 4 chunks");
    assert.equal(stats.chunkFlushes, 3, "expected 3 flushes (chunks 0-2 filled)");
    assert.equal(stats.stagedBytes, 32 * 2 * KV_DIM * 4, "one partial chunk staged");
    // Full-prefix read across flushed + staged chunks.
    const { keys, values } = await vcache.read(seq, 0, prefix(100));
    for (let pos = 0; pos < 100; pos++) {
      assert.deepEqual(keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(pos * 2 + 1, KV_DIM));
      assert.deepEqual(values.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(pos * 2 + 2, KV_DIM));
    }
    // Rewrite position 5 in flushed chunk 0 (read-modify-write path).
    const k5 = pseudoRandom(999001, KV_DIM);
    const v5 = pseudoRandom(999002, KV_DIM);
    await vcache.append(seq, 0, 5, k5, v5);
    const after = await vcache.read(seq, 0, [5, 6, 0, 31, 32]);
    assert.deepEqual(after.keys.subarray(0, KV_DIM), k5, "rewritten K at pos 5");
    assert.deepEqual(after.values.subarray(0, KV_DIM), v5, "rewritten V at pos 5");
    assert.deepEqual(after.keys.subarray(KV_DIM, 2 * KV_DIM), pseudoRandom(6 * 2 + 1, KV_DIM));
    assert.equal(vcache.stats().rmwFetches, 1, "expected one RMW fetch");
    console.log("2. multi-chunk growth + flushed-chunk rewrite ok");
  }

  // --- 3. free / reuse ----------------------------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    for (let pos = 0; pos < 40; pos++) {
      await vcache.append("seq-a", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
      await vcache.append("seq-a", 1, pos, pseudoRandom(500 + pos, KV_DIM), pseudoRandom(600 + pos, KV_DIM));
    }
    assert.equal(vcache.usedBlocks(), 2 * 2 * 2, "2 layers x 2 chunks x 2 blocks");
    assert.ok(vram.allocatedBytes() > 0);
    await vcache.free("seq-a");
    assert.equal(vcache.usedBlocks(), 0, "usedBlocks must drop to 0 after free");
    assert.equal(vram.allocatedBytes(), 0, "vRAM bytes must drop to 0 after free");
    await assert.rejects(vcache.read("seq-a", 0, [0]), /no blocks allocated/);
    await vcache.free("seq-a"); // double free is a no-op, like the host cache
    // A new sequence works after the free and starts clean (zeros, not stale data).
    await vcache.append("seq-b", 0, 0, pseudoRandom(7, KV_DIM), pseudoRandom(8, KV_DIM));
    const { keys } = await vcache.read("seq-b", 0, [0, 1]);
    assert.deepEqual(keys.subarray(0, KV_DIM), pseudoRandom(7, KV_DIM));
    assert.deepEqual(keys.subarray(KV_DIM), new Float32Array(KV_DIM), "unwritten pos in touched chunk reads as zeros");
    console.log("3. free/reuse ok");
  }

  // --- 4. Multi-layer + multi-sequence isolation --------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    const seqs = ["s1", "s2", "s3"];
    const layers = [0, 1, 2, 3];
    for (let round = 0; round < 50; round++) {
      for (const seq of seqs) {
        for (const layer of layers) {
          const seed = seq.charCodeAt(1) * 100000 + layer * 1000 + round;
          await vcache.append(seq, layer, round, pseudoRandom(seed + 1, KV_DIM), pseudoRandom(seed + 2, KV_DIM));
        }
      }
    }
    for (const seq of seqs) {
      for (const layer of layers) {
        const { keys, values } = await vcache.read(seq, layer, prefix(50));
        for (let pos = 0; pos < 50; pos++) {
          const seed = seq.charCodeAt(1) * 100000 + layer * 1000 + pos;
          assert.deepEqual(keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(seed + 1, KV_DIM));
          assert.deepEqual(values.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(seed + 2, KV_DIM));
        }
      }
    }
    console.log("4. multi-layer + multi-sequence isolation ok");
  }

  // --- 4b. close() frees every chunk (P4: no vRAM leak on worker close) ---
  // The worker calls cache.close() on stage reload and on shutdown; a
  // chunk left allocated there is a leak per model switch (the latent
  // leak docs/KV-DESIGN.md §6.4 flagged). Staged chunks flush first,
  // then every chunk is freed; the underlying vRAM drops to 0 too.
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32); // 32 tokens/chunk = 2 blocks
    for (let pos = 0; pos < 40; pos++) {
      await vcache.append("seq-close", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
      await vcache.append("seq-close", 1, pos, pseudoRandom(500 + pos, KV_DIM), pseudoRandom(600 + pos, KV_DIM));
    }
    assert.equal(vcache.stats().chunks, 4, "2 layers x 2 chunks before close");
    assert.ok(vram.allocatedBytes() > 0);
    await vcache.close();
    assert.equal(vcache.stats().chunks, 0, "no chunks remain after close");
    assert.equal(vcache.stats().vramBytes, 0, "no vRAM bytes remain after close");
    assert.equal(vcache.stats().stagedBytes, 0, "no staged bytes remain after close");
    assert.equal(vram.allocatedBytes(), 0, "underlying vRAM freed after close");
    await vcache.close(); // double close is a no-op and never rejects
    console.log("4b. close() frees every chunk ok");
  }

  // --- 5. Out-of-memory ----------------------------------------------------
  {
    // Capacity for exactly one 4 MiB page: one chunk fits, the second does not.
    const PAGE = 4 * 1024 * 1024;
    const vram = new MapVramAccess(PAGE);
    const vcache = makeCache(vram, 32);
    for (let pos = 0; pos < 32; pos++) {
      await vcache.append("seq-oom", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    await assert.rejects(
      vcache.append("seq-oom", 0, 32, pseudoRandom(1, KV_DIM), pseudoRandom(2, KV_DIM)),
      (err: unknown) => err instanceof VramKvError && /out of memory/.test(err.message),
      "second chunk must reject with VramKvError/out of memory",
    );
    // The cache is still usable afterwards: reads of committed data work
    // and a different sequence that fits in the existing chunk works.
    const { keys } = await vcache.read("seq-oom", 0, prefix(32));
    assert.deepEqual(keys.subarray(0, KV_DIM), pseudoRandom(1, KV_DIM));
    // Block-cap OOM (before touching vRAM at all).
    const tiny = new VramPagedKVCache(BLOCK, 2, KV_HEADS, HEAD_DIM, new MapVramAccess(), {
      chunkTokens: 32,
    });
    await assert.rejects(
      tiny.allocate("seq-cap", 0, 64),
      (err: unknown) => err instanceof VramKvError && /out of memory/.test(err.message),
    );
    console.log("5. out-of-memory errors ok");
  }

  // --- 6. Fault injection: read failures propagate, never silent zeros ---
  {
    const inner = new MapVramAccess();
    const faulty = new FaultyVramAccess(inner);
    const vcache = makeCache(faulty, 32);
    for (let pos = 0; pos < 40; pos++) {
      await vcache.append("seq-fault", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    // Chunk 0 is flushed, so this read must hit vRAM — and the fault
    // must surface as a rejection naming the sequence.
    await assert.rejects(
      vcache.read("seq-fault", 0, prefix(40)),
      (err: unknown) =>
        err instanceof VramKvError &&
        /seq-fault/.test(err.message) &&
        /re-prefill/.test(err.message),
      "vRAM read failure must reject with a VramKvError naming the seq",
    );
    // A failed flush rejects but keeps the staged data (append is not lost).
    const inner2 = new MapVramAccess();
    const faulty2 = new FaultyVramAccess(inner2);
    faulty2.failWrites = true;
    const vcache2 = makeCache(faulty2, 32);
    for (let pos = 0; pos < 31; pos++) {
      await vcache2.append("seq-fw", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    await assert.rejects(
      vcache2.append("seq-fw", 0, 31, pseudoRandom(63, KV_DIM), pseudoRandom(64, KV_DIM)),
      (err: unknown) => err instanceof VramKvError && /failed to flush/.test(err.message),
    );
    // Staged data is still readable after the failed flush.
    const { keys } = await vcache2.read("seq-fw", 0, [31]);
    assert.deepEqual(keys, pseudoRandom(63, KV_DIM), "staged data survives a failed flush");
    console.log("6. fault injection propagates errors ok");
  }

  // --- 7. Unwritten reads --------------------------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    await assert.rejects(vcache.read("nope", 0, [0]), /no blocks allocated/);
    await vcache.allocate("seq-u", 0, 32);
    await assert.rejects(
      vcache.read("seq-u", 0, [0]),
      (err: unknown) => err instanceof VramKvError && /no K\/V written yet/.test(err.message),
      "untouched chunk must throw like the host cache",
    );
    await vcache.append("seq-u", 0, 3, pseudoRandom(1, KV_DIM), pseudoRandom(2, KV_DIM));
    const { keys } = await vcache.read("seq-u", 0, [0, 1, 2, 3]);
    assert.deepEqual(keys.subarray(0, 3 * KV_DIM), new Float32Array(3 * KV_DIM));
    assert.deepEqual(keys.subarray(3 * KV_DIM), pseudoRandom(1, KV_DIM));
    console.log("7. unwritten reads ok");
  }

  // --- 8. Counters match the chunking model -------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    const CHUNK_BYTES = 32 * 2 * KV_DIM * 4;
    for (let pos = 0; pos < 96; pos++) {
      await vcache.append("seq-c", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    // 96 tokens = 3 full chunks, each flushed once with one whole-page write.
    let s = vcache.stats();
    assert.equal(s.chunkFlushes, 3);
    assert.equal(s.vramWrites, 3);
    assert.equal(s.bytesWritten, 3 * CHUNK_BYTES);
    assert.equal(s.vramReads, 0, "appends alone must not read from vRAM");
    await vcache.read("seq-c", 0, prefix(96));
    s = vcache.stats();
    assert.equal(s.vramReads, 3, "one coalesced read per flushed chunk");
    assert.equal(s.bytesRead, 3 * CHUNK_BYTES);
    // A second identical read re-fetches (no read cache in this prototype).
    await vcache.read("seq-c", 0, prefix(96));
    assert.equal(vcache.stats().vramReads, 6);
    console.log("8. counters match the chunking model ok");
  }

  // --- 9. flushAll makes staged data durable in vRAM -----------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    await vcache.append("seq-f", 0, 0, pseudoRandom(1, KV_DIM), pseudoRandom(2, KV_DIM));
    assert.equal(vcache.stats().chunkFlushes, 0);
    await vcache.flushAll();
    assert.equal(vcache.stats().chunkFlushes, 1);
    // After the flush the chunk is served from vRAM, not staging.
    const before = vcache.stats().vramReads;
    const { keys } = await vcache.read("seq-f", 0, [0]);
    assert.deepEqual(keys, pseudoRandom(1, KV_DIM));
    assert.equal(vcache.stats().vramReads, before + 1);
    console.log("9. flushAll ok");
  }

  // --- 10. Default chunking is one vRAM page at demo-100m geometry -------
  {
    const vram = new MapVramAccess();
    // demo-100m: numKeyValueHeads 8, headDim 64 -> kvDim 512.
    const demo = new VramPagedKVCache(16, 1 << 20, 8, 64, vram);
    assert.equal(demo.getChunkTokens(), 1024, "1024 tokens per chunk at demo geometry");
    assert.equal(demo.getChunkBytes(), 4 * 1024 * 1024, "one 4 MiB page per chunk");
    console.log("10. demo-100m chunking is one page ok");
  }

  // --- 11. Block ids: globally unique, stable, fresh after free --------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32); // 2 blocks per chunk
    const a0 = await vcache.allocate("A", 0, 32);
    const b0 = await vcache.allocate("B", 0, 32);
    const a1 = await vcache.allocate("A", 1, 32);
    assert.deepEqual(a0, [0, 1]);
    assert.deepEqual(b0, [2, 3], "a second sequence must not reuse block ids");
    assert.deepEqual(a1, [4, 5], "a second layer must not reuse block ids");
    assert.equal(new Set([...a0, ...b0, ...a1]).size, 6, "ids unique across seqs and layers");
    assert.deepEqual(await vcache.allocate("A", 0, 32), a0, "block table stable on re-allocate");
    await vcache.free("A");
    const c0 = await vcache.allocate("C", 0, 32);
    assert.deepEqual(c0, [6, 7], "ids are never reused after free (host reuses via free-list; fresh ids avoid ABA staleness)");
    console.log("11. block ids globally unique + stable + fresh after free ok");
  }

  // --- 11b. Shrinking re-allocate: pinned divergence from the host -----
  {
    const host = new BlockPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM);
    const vcache = makeCache(new MapVramAccess(), 32);
    const vTable = await vcache.allocate("s", 0, 100);
    const hTable = await host.allocate("s", 0, 100);
    assert.equal(vTable.length, hTable.length, "growing allocate lengths match the host");
    assert.equal((await host.allocate("s", 0, 16)).length, 7, "host returns its FULL table on a shrinking re-allocate");
    assert.equal((await vcache.allocate("s", 0, 16)).length, 1, "vRAM returns the shorter logical prefix on a shrinking re-allocate (documented divergence; the engine only grows)");
    assert.equal((await vcache.allocate("s", 0, 0)).length, 0, "allocate(0) returns an empty table");
    assert.equal((await host.allocate("s", 0, 0)).length, 7, "host allocate(0) returns the full table (same divergence)");
    console.log("11b. shrinking re-allocate divergence pinned ok");
  }

  // --- 12. Block-granular unwritten reads (host parity) -----------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 64); // 4 blocks per chunk
    const host = new BlockPagedKVCache(BLOCK, TOTAL_BLOCKS, KV_HEADS, HEAD_DIM);
    await vcache.append("s", 0, 48, pseudoRandom(1, KV_DIM), pseudoRandom(2, KV_DIM));
    await host.append("s", 0, 48, pseudoRandom(1, KV_DIM), pseudoRandom(2, KV_DIM));
    // Position 0 is in block 0, never touched: both must throw.
    await assert.rejects(vcache.read("s", 0, [0]), /no K\/V written yet/);
    await assert.rejects(host.read("s", 0, [0]), /no K\/V written yet/);
    // A run spanning an unwritten block throws, even though a later
    // position in the run is written.
    await assert.rejects(vcache.read("s", 0, prefix(49)), /no K\/V written yet/);
    await assert.rejects(vcache.read("s", 0, [32, 33]), /no K\/V written yet/);
    // Position 49 is in the touched block (48-63): zeros, like the host.
    const got = await vcache.read("s", 0, [48, 49]);
    assert.deepEqual(got.keys.subarray(0, KV_DIM), pseudoRandom(1, KV_DIM));
    assert.deepEqual(got.keys.subarray(KV_DIM), new Float32Array(KV_DIM));
    assert.deepEqual((await host.read("s", 0, [48, 49])).keys, got.keys);
    // flushAll() on an untouched chunk must NOT turn a throw into zeros
    // (the prototype bug: flushed=true bypassed the written check).
    const vcache2 = makeCache(new MapVramAccess(), 32);
    await vcache2.allocate("s", 0, 32);
    await vcache2.flushAll();
    await assert.rejects(vcache2.read("s", 0, [0]), /no K\/V written yet/);
    console.log("12. block-granular unwritten reads ok");
  }

  // --- 13. Concurrent sequences interleaved on one cache ----------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    const seqs = ["s1", "s2", "s3"];
    const layers = [0, 1];
    // Interleaved appends across sequences and layers, all at once.
    await Promise.all(
      seqs.flatMap((seq) =>
        layers.flatMap((layer) =>
          Array.from({ length: 40 }, (_, pos) =>
            vcache.append(seq, layer, pos, pseudoRandom(seq.charCodeAt(1) * 1000 + layer * 100 + pos * 2 + 1, KV_DIM), pseudoRandom(seq.charCodeAt(1) * 1000 + layer * 100 + pos * 2 + 2, KV_DIM)),
          ),
        ),
      ),
    );
    for (const seq of seqs) {
      for (const layer of layers) {
        const { keys, values } = await vcache.read(seq, layer, prefix(40));
        for (let pos = 0; pos < 40; pos++) {
          const seed = seq.charCodeAt(1) * 1000 + layer * 100 + pos * 2;
          assert.deepEqual(keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(seed + 1, KV_DIM));
          assert.deepEqual(values.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(seed + 2, KV_DIM));
        }
      }
    }
    // 3 seqs x 2 layers x 40 tokens, 32-token chunks: 2 chunks each.
    assert.equal(vcache.stats().chunks, 12, "3 seqs x 2 layers x 2 chunks");
    // 32 concurrent appends to ONE chunk must create exactly one chunk
    // (the prototype created 32 — one per racing allocate()).
    const vcache2 = makeCache(new MapVramAccess(), 32);
    await Promise.all(
      Array.from({ length: 32 }, (_, pos) =>
        vcache2.append("race", 0, pos, pseudoRandom(pos * 2 + 1, KV_DIM), pseudoRandom(pos * 2 + 2, KV_DIM)),
      ),
    );
    assert.equal(vcache2.stats().chunks, 1, "concurrent appends to one chunk must not double-allocate");
    assert.equal(vcache2.stats().chunkFlushes, 1, "the chunk flushes exactly once on fill");
    const { keys } = await vcache2.read("race", 0, prefix(32));
    for (let pos = 0; pos < 32; pos++) {
      assert.deepEqual(keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(pos * 2 + 1, KV_DIM));
    }
    // Concurrent allocate()s of the same (seq, layer) must not
    // double-allocate either.
    const vcache3 = makeCache(new MapVramAccess(), 32);
    await Promise.all([vcache3.allocate("r", 0, 32), vcache3.allocate("r", 0, 32), vcache3.allocate("r", 0, 32)]);
    assert.equal(vcache3.stats().chunks, 1, "concurrent allocate()s must create one chunk");
    assert.equal(vcache3.usedBlocks(), 2);
    console.log("13. concurrent sequences interleaved on one cache ok");
  }

  // --- 14. Free-then-reallocate -------------------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    for (let pos = 0; pos < 40; pos++) {
      await vcache.append("seq-a", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
      await vcache.append("seq-a", 1, pos, pseudoRandom(500 + pos, KV_DIM), pseudoRandom(600 + pos, KV_DIM));
    }
    const beforeIds = await vcache.allocate("seq-a", 0, 40);
    assert.equal(vcache.usedBlocks(), 8, "2 layers x 2 chunks x 2 blocks");
    await vcache.free("seq-a");
    assert.equal(vcache.usedBlocks(), 0, "usedBlocks must drop to 0 after free");
    assert.equal(vcache.logicalBlocks(), 0, "logicalBlocks must drop to 0 after free");
    assert.equal(vram.allocatedBytes(), 0, "vRAM bytes must drop to 0 after free");
    await assert.rejects(vcache.read("seq-a", 0, [0]), /no blocks allocated/);
    await vcache.free("seq-a"); // double free is a no-op, like the host cache
    const afterIds = await vcache.allocate("seq-b", 0, 40);
    assert.ok(afterIds.every((id) => !beforeIds.includes(id)), "reallocated ids must be fresh, never reused");
    assert.equal(vcache.usedBlocks(), 4, "2 chunks x 2 blocks after reallocate");
    assert.equal(vcache.logicalBlocks(), 3, "ceil(40/16) = 3 logical blocks");
    console.log("14. free-then-reallocate ok");
  }

  // --- 15. Allocate beyond the block cap: reject, never fall back -------
  {
    const vram = new MapVramAccess();
    const tiny = new VramPagedKVCache(BLOCK, 2, KV_HEADS, HEAD_DIM, vram, { chunkTokens: 32 });
    await assert.rejects(
      tiny.allocate("seq-cap", 0, 64),
      (err: unknown) => err instanceof VramKvError && /out of memory/.test(err.message),
      "64 tokens need 2 chunks (4 blocks) over a 2-block cap",
    );
    assert.equal(tiny.usedBlocks(), 0, "a rejected allocate must not partially grow the table");
    assert.equal(tiny.stats().chunks, 0, "a rejected allocate must not allocate a chunk");
    // The cache stays usable: a fitting allocate works afterwards.
    assert.deepEqual(await tiny.allocate("seq-cap", 0, 32), [0, 1]);
    for (let pos = 0; pos < 32; pos++) {
      await tiny.append("seq-cap", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    await assert.rejects(
      tiny.allocate("seq-cap", 0, 64),
      (err: unknown) => err instanceof VramKvError && /out of memory/.test(err.message),
      "growing past the cap rejects even when one chunk already exists",
    );
    assert.equal(tiny.stats().chunks, 1, "the rejected growth added no chunk");
    assert.equal(tiny.usedBlocks(), 2, "usedBlocks unchanged by the rejection");
    // Committed data still reads, and an overwrite inside the existing
    // (flushed) chunk works — nothing silently fell back to host.
    const { keys } = await tiny.read("seq-cap", 0, prefix(32));
    assert.deepEqual(keys.subarray(0, KV_DIM), pseudoRandom(1, KV_DIM));
    assert.equal(tiny.stats().vramReads, 1, "the read came from vRAM, not a host fallback");
    // A totalBlocks smaller than one chunk is a loud constructor error,
    // not a cache that rejects every allocate with a confusing OOM.
    assert.throws(
      () => new VramPagedKVCache(BLOCK, 1, KV_HEADS, HEAD_DIM, new MapVramAccess(), { chunkTokens: 32 }),
      (err: unknown) => err instanceof VramKvError && /smaller than one chunk/.test(err.message),
    );
    assert.throws(
      () => new VramPagedKVCache(16, 63, 8, 64, new MapVramAccess()),
      (err: unknown) => err instanceof VramKvError && /smaller than one chunk/.test(err.message),
      "63 blocks cannot hold one 64-block demo-geometry chunk",
    );
    console.log("15. allocate beyond the block cap rejects, never falls back ok");
  }

  // --- 16. Read after a chunk flushed mid-sequence ------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    for (let pos = 0; pos < 40; pos++) {
      await vcache.append("seq-mid", 0, pos, pseudoRandom(pos * 2 + 1, KV_DIM), pseudoRandom(pos * 2 + 2, KV_DIM));
    }
    // Chunk 0 flushed at 32 tokens; chunk 1 (8 tokens) still staged.
    assert.equal(vcache.stats().chunkFlushes, 1, "chunk 0 flushed mid-sequence");
    assert.equal(vcache.stats().vramReads, 0, "no reads yet");
    const first = await vcache.read("seq-mid", 0, prefix(40));
    for (let pos = 0; pos < 40; pos++) {
      assert.deepEqual(first.keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(pos * 2 + 1, KV_DIM));
    }
    assert.equal(vcache.stats().vramReads, 1, "only the flushed chunk costs a vRAM read; the staged chunk is free");
    // Grow past the second chunk boundary and read the whole prefix again.
    for (let pos = 40; pos < 70; pos++) {
      await vcache.append("seq-mid", 0, pos, pseudoRandom(pos * 2 + 1, KV_DIM), pseudoRandom(pos * 2 + 2, KV_DIM));
    }
    assert.equal(vcache.stats().chunkFlushes, 2, "chunk 1 flushed at 64 tokens");
    const second = await vcache.read("seq-mid", 0, prefix(70));
    for (let pos = 0; pos < 70; pos++) {
      assert.deepEqual(second.keys.subarray(pos * KV_DIM, (pos + 1) * KV_DIM), pseudoRandom(pos * 2 + 1, KV_DIM));
    }
    assert.equal(vcache.stats().vramReads, 3, "two flushed chunks re-fetched, staged tail free");
    // A read straddling the chunk boundary returns the right values.
    const straddle = await vcache.read("seq-mid", 0, [31, 32, 63, 64, 65]);
    assert.deepEqual(straddle.keys.subarray(0, KV_DIM), pseudoRandom(31 * 2 + 1, KV_DIM));
    assert.deepEqual(straddle.keys.subarray(KV_DIM, 2 * KV_DIM), pseudoRandom(32 * 2 + 1, KV_DIM));
    assert.deepEqual(straddle.keys.subarray(2 * KV_DIM, 3 * KV_DIM), pseudoRandom(63 * 2 + 1, KV_DIM));
    assert.deepEqual(straddle.keys.subarray(4 * KV_DIM), pseudoRandom(65 * 2 + 1, KV_DIM));
    console.log("16. read after a chunk flushed mid-sequence ok");
  }

  // --- 17. Append-after-free starts clean ---------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    for (let pos = 0; pos < 4; pos++) {
      await vcache.append("seq-x", 0, pos, pseudoRandom(100 + pos, KV_DIM), pseudoRandom(200 + pos, KV_DIM));
    }
    const stale = pseudoRandom(101, KV_DIM);
    assert.ok(stale.some((x) => x !== 0), "the stale value at pos 1 is non-zero, so zeros prove it is gone");
    await vcache.free("seq-x");
    // Same seq id, new data: only pos 0 is written this time.
    await vcache.append("seq-x", 0, 0, pseudoRandom(999, KV_DIM), pseudoRandom(998, KV_DIM));
    const { keys, values } = await vcache.read("seq-x", 0, [0, 1]);
    assert.deepEqual(keys.subarray(0, KV_DIM), pseudoRandom(999, KV_DIM), "new value at pos 0");
    assert.deepEqual(keys.subarray(KV_DIM), new Float32Array(KV_DIM), "pos 1 is zeros, not the stale pre-free value");
    assert.deepEqual(values.subarray(KV_DIM), new Float32Array(KV_DIM));
    assert.equal(vcache.stats().chunks, 1, "one fresh chunk after the free");
    console.log("17. append-after-free starts clean ok");
  }

  // --- 18. free() never rejects, even when vRAM free fails ---------------
  {
    const inner = new MapVramAccess();
    const failing = new FailingFreeVramAccess(inner);
    const vcache = makeCache(failing, 32);
    for (let pos = 0; pos < 32; pos++) {
      await vcache.append("seq-f", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
      await vcache.append("seq-f", 1, pos, pseudoRandom(500 + pos, KV_DIM), pseudoRandom(600 + pos, KV_DIM));
    }
    assert.equal(vcache.stats().chunks, 2, "two chunks (one per layer) to free");
    // Must resolve, not reject: the decoders call free() at generate()
    // start with no try/catch, so a throw here would abort generation.
    await vcache.free("seq-f");
    assert.equal(vcache.stats().freeFailures, 2, "both failed vRAM frees are counted");
    assert.equal(vcache.usedBlocks(), 0, "local state is dropped even when the vRAM frees fail");
    assert.equal(vcache.stats().chunks, 0);
    await assert.rejects(vcache.read("seq-f", 0, [0]), /no blocks allocated/);
    // The cache stays usable afterwards.
    await vcache.append("seq-g", 0, 0, pseudoRandom(7, KV_DIM), pseudoRandom(8, KV_DIM));
    const { keys } = await vcache.read("seq-g", 0, [0]);
    assert.deepEqual(keys, pseudoRandom(7, KV_DIM));
    // A second free of the same (now unknown) seq is a no-op that
    // adds no failures.
    await vcache.free("seq-f");
    assert.equal(vcache.stats().freeFailures, 2, "double free adds no failures");
    console.log("18. free() never rejects; failures counted ok");
  }

  // --- 19. Stats scripted verification ------------------------------------
  {
    const vram = new MapVramAccess();
    const vcache = makeCache(vram, 32);
    const CHUNK_BYTES = 32 * 2 * KV_DIM * 4; // 8192
    const TOKEN_BYTES = 2 * KV_DIM * 4; // 256
    const eq = (s: ReturnType<typeof vcache.stats>, want: Partial<ReturnType<typeof vcache.stats>>, step: string): void => {
      for (const [k, v] of Object.entries(want)) {
        assert.equal(s[k as keyof typeof s], v, `${step}: stats.${k} expected ${v}, got ${s[k as keyof typeof s]}`);
      }
    };
    eq(vcache.stats(), { chunks: 0, stagedBytes: 0, vramBytes: 0, vramReads: 0, vramWrites: 0, bytesRead: 0, bytesWritten: 0, chunkFlushes: 0, rmwFetches: 0, freeFailures: 0 }, "fresh");
    // Fill chunk 0 (32 tokens) -> one flush, one whole-page write.
    for (let pos = 0; pos < 32; pos++) {
      await vcache.append("seq-c", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    eq(vcache.stats(), { chunks: 1, stagedBytes: 0, vramBytes: CHUNK_BYTES, vramReads: 0, vramWrites: 1, bytesRead: 0, bytesWritten: CHUNK_BYTES, chunkFlushes: 1, rmwFetches: 0 }, "chunk 0 filled");
    assert.equal(vcache.usedBlocks(), 2, "one chunk = 2 blocks reserved");
    assert.equal(vcache.logicalBlocks(), 2, "ceil(32/16) = 2 logical blocks");
    // One more token stages chunk 1; reserved and logical counts diverge.
    await vcache.append("seq-c", 0, 32, pseudoRandom(65, KV_DIM), pseudoRandom(66, KV_DIM));
    eq(vcache.stats(), { chunks: 2, stagedBytes: CHUNK_BYTES, vramBytes: CHUNK_BYTES, vramWrites: 1, bytesWritten: CHUNK_BYTES }, "chunk 1 staged");
    assert.equal(vcache.usedBlocks(), 4, "two chunks reserved");
    assert.equal(vcache.logicalBlocks(), 3, "ceil(33/16) = 3 logical blocks");
    // Prefix read: flushed chunk costs one coalesced read, staged is free.
    await vcache.read("seq-c", 0, prefix(33));
    eq(vcache.stats(), { vramReads: 1, bytesRead: CHUNK_BYTES }, "prefix read across flushed + staged");
    // Overwrite pos 5 in the flushed chunk: one RMW fetch + one re-flush.
    await vcache.append("seq-c", 0, 5, pseudoRandom(999001, KV_DIM), pseudoRandom(999002, KV_DIM));
    eq(vcache.stats(), { vramReads: 2, bytesRead: 2 * CHUNK_BYTES, vramWrites: 2, bytesWritten: 2 * CHUNK_BYTES, chunkFlushes: 2, rmwFetches: 1 }, "RMW overwrite of flushed pos 5");
    const afterRmw = await vcache.read("seq-c", 0, [5]);
    assert.deepEqual(afterRmw.keys, pseudoRandom(999001, KV_DIM), "rewritten value survives the RMW round-trip");
    eq(vcache.stats(), { vramReads: 3, bytesRead: 2 * CHUNK_BYTES + TOKEN_BYTES }, "single-token read of flushed chunk");
    // flushAll flushes the partial chunk 1 (whole page, zero tail).
    await vcache.flushAll();
    eq(vcache.stats(), { chunks: 2, stagedBytes: 0, vramBytes: 2 * CHUNK_BYTES, vramWrites: 3, bytesWritten: 3 * CHUNK_BYTES, chunkFlushes: 3 }, "flushAll");
    // Free: live snapshots drop to zero; cumulative counters are untouched.
    await vcache.free("seq-c");
    eq(vcache.stats(), { chunks: 0, stagedBytes: 0, vramBytes: 0, freeFailures: 0, chunkFlushes: 3, vramWrites: 3, bytesWritten: 3 * CHUNK_BYTES, vramReads: 3 }, "free");
    assert.equal(vcache.usedBlocks(), 0);
    assert.equal(vcache.logicalBlocks(), 0);
    // Short reads reject and are NOT counted as traffic.
    const shortCache = makeCache(new ShortReadVramAccess(new MapVramAccess()), 32);
    for (let pos = 0; pos < 32; pos++) {
      await shortCache.append("seq-s", 0, pos, pseudoRandom(pos + 1, KV_DIM), pseudoRandom(pos + 2, KV_DIM));
    }
    await assert.rejects(
      shortCache.read("seq-s", 0, prefix(32)),
      (err: unknown) => err instanceof VramKvError && /returned \d+ bytes, expected/.test(err.message),
      "a short vRAM read must reject, not serve partial data",
    );
    assert.equal(shortCache.stats().vramReads, 0, "a short read counts no vRAM traffic");
    assert.equal(shortCache.stats().bytesRead, 0);
    console.log("19. stats scripted verification ok");
  }

  console.log("kvcache-vram tests passed");
}

main().catch((err: unknown) => {
  console.error("kvcache-vram tests FAILED:", err);
  process.exitCode = 1;
});
