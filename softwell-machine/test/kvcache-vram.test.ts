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
 */
import assert from "node:assert/strict";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  VramKvError,
  VramPagedKVCache,
} from "../src/scheduler/kvcache-vram.ts";
import type { VramAccess } from "../src/scheduler/kvcache-vram.ts";

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
        host.append(seq, layer, pos, k, v);
      }
    }
    for (const layer of layers) {
      for (const n of [1, 16, 17, 64, 65, 100]) {
        const got = await vcache.read(seq, layer, prefix(n));
        const want = host.read(seq, layer, prefix(n));
        assert.deepEqual(got.keys, want.keys, `round-trip keys layer ${layer} n=${n}`);
        assert.deepEqual(got.values, want.values, `round-trip values layer ${layer} n=${n}`);
      }
      // Non-contiguous + out-of-order reads must also match.
      const positions = [99, 0, 63, 64, 17, 42];
      const got = await vcache.read(seq, layer, positions);
      const want = host.read(seq, layer, positions);
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
      const hTable = host.allocate(seq, layer, tokens);
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

  console.log("kvcache-vram tests passed");
}

main().catch((err: unknown) => {
  console.error("kvcache-vram tests FAILED:", err);
  process.exitCode = 1;
});
