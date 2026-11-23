/**
 * Chunk-table handover: resubmit is the contract — run with:
 * node test/kv-handover-resubmit.test.ts
 *
 * After a worker dies, its VramPagedKVCache chunk table dies with it:
 * the table (which allocId/vaddr belongs to which seq/layer, which
 * blocks are written, how far each chunk is filled) lives only in the
 * cache instance. This test pins that contract and proves the
 * replacement path — re-prefill (resubmit) — reproduces the KV exactly:
 *
 *  1. A cache with small chunkTokens (32) appends 100 tokens x 2 layers.
 *     Chunks 0-2 flush to vRAM; chunk 3 stays staged on the host.
 *  2. The flushed bytes are read back DIRECTLY from the vRAM access
 *     (bypassing any cache): they survive — the bytes are not the
 *     problem, the table is.
 *  3. The cache is dropped WITHOUT close() (a crash: no flush, no
 *     free). A fresh cache over the SAME access cannot read the old
 *     sequence at all (VramKvError — "no blocks allocated"): it has
 *     no table to re-attach with, even though the bytes are still there.
 *  4. Resubmit: the same 100 tokens are re-prefilled into the fresh
 *     cache, and the full-prefix reads are bit-identical to the
 *     original cache's reads, both layers. Re-prefill reproduces KV.
 *
 * Why resubmit is right and handover is not built (docs/KV-DESIGN.md
 * §7b): at production geometry chunkTokens is 1024 (one 4 MiB page),
 * so a sequence only flushes past 1024 tokens — and only flushed
 * chunks survive a worker death; the staged tail is host memory and
 * is lost with the worker either way. In the regime where vRAM KV is
 * viable (short context, the default-stays-host regime), NOTHING is
 * ever flushed, so a handover would save exactly nothing. Above 1024
 * tokens it would save at most the flushed prefix while still needing
 * a snapshot-distribution protocol (the table must reach the
 * orchestrator BEFORE the death, since a dead worker cannot send it),
 * an attach protocol, block-id continuity, and a tail re-prefill for
 * the lost staged chunk — for a regime that already costs +75%
 * decode. The scheduler already owns the resubmit path (sequences
 * whose KV lived on a lost worker fail their next step and are marked
 * "error"; the caller resubmits — src/scheduler/batch.ts updatePlan).
 */
import assert from "node:assert/strict";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  VramKvError,
  VramPagedKVCache,
} from "../src/scheduler/kvcache-vram.ts";
import type { VramAccess } from "../src/contracts.ts";

/** Byte-addressed Map backend (same shape as test/kvcache-vram.test.ts). */
class MapVramAccess implements VramAccess {
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();
  private bump = 0;
  private nextId = 0;
  readonly allocLog: Array<{ id: string; vaddr: number }> = [];

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
    const id = `alloc-${this.nextId++}`;
    const vaddr = this.bump;
    this.bump += rounded;
    this.allocs.set(id, { vaddr, data: new Uint8Array(rounded) });
    this.allocLog.push({ id, vaddr });
    return { id, vaddr };
  }

  async free(id: string): Promise<void> {
    if (!this.allocs.delete(id)) {
      throw new Error(`MapVramAccess: unknown allocation ${id}`);
    }
  }

  liveAllocations(): number {
    return this.allocs.size;
  }
}

const BLOCK_SIZE = 16;
const TOTAL_BLOCKS = 4096;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 8;
const KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 16
const CHUNK_TOKENS = 32; // test-only override: forces flushes at small sizes
const TOKENS = 100;
const LAYERS = [0, 1];

/** Deterministic K/V for (layer, position): K = pos, V = pos + 1000. */
function kv(layer: number, position: number): { key: Float32Array; value: Float32Array } {
  const key = new Float32Array(KV_DIM).fill(position + layer * 10_000);
  const value = new Float32Array(KV_DIM).fill(position + 1000 + layer * 10_000);
  return { key, value };
}

async function prefill(
  cache: VramPagedKVCache,
  seqId: string,
): Promise<Array<{ keys: Float32Array; values: Float32Array }>> {
  const reads: Array<{ keys: Float32Array; values: Float32Array }> = [];
  for (const layer of LAYERS) {
    for (let pos = 0; pos < TOKENS; pos++) {
      const { key, value } = kv(layer, pos);
      await cache.append(seqId, layer, pos, key, value);
    }
    reads.push(
      await cache.read(seqId, layer, Array.from({ length: TOKENS }, (_, i) => i)),
    );
  }
  return reads;
}

async function main(): Promise<void> {
  const access = new MapVramAccess();
  let cache: VramPagedKVCache | null = new VramPagedKVCache(
    BLOCK_SIZE,
    TOTAL_BLOCKS,
    NUM_KV_HEADS,
    HEAD_DIM,
    access,
    { chunkTokens: CHUNK_TOKENS, replicas: 1 },
  );

  // 1. Prefill 100 tokens x 2 layers. Chunks 0-2 (positions 0-95) flush;
  //    chunk 3 (96-99) stays staged on the host.
  const originalReads = await prefill(cache, "seq-1");
  const stats = cache.stats();
  assert.equal(stats.chunkFlushes, 6, `expected 3 flushed chunks x 2 layers, got ${stats.chunkFlushes}`);
  assert.equal(stats.chunks, 8, `expected 4 chunks x 2 layers, got ${stats.chunks}`);
  assert.ok(stats.vramBytes > 0, "flushed chunks must hold vRAM bytes");
  assert.ok(stats.stagedBytes > 0, "the tail chunk must still be staged on the host");
  console.log(
    `prefill: ${TOKENS} tokens x ${LAYERS.length} layers, ` +
      `chunkFlushes=${stats.chunkFlushes}, vramBytes=${stats.vramBytes}, stagedBytes=${stats.stagedBytes}`,
  );

  // 2. The flushed bytes survive in vRAM, readable without any cache:
  //    chunk 0 of layer 0 is the first allocation; its first token's K
  //    is position 0 (value 0.0) and its V is 1000.0, interleaved.
  const firstChunk = access.allocLog[0] as { id: string; vaddr: number };
  const raw = await access.read(firstChunk.vaddr, CHUNK_TOKENS * 2 * KV_DIM * 4);
  const rawF32 = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
  assert.equal(rawF32[0], 0, "flushed chunk must hold token 0's K");
  assert.equal(rawF32[KV_DIM], 1000, "flushed chunk must hold token 0's V");
  assert.equal(rawF32[2 * KV_DIM], 1, "flushed chunk must hold token 1's K");
  console.log("flushed chunk bytes survive in vRAM, readable without a cache: ok");

  // 3. Worker death: drop the cache WITHOUT close() (a crash — no flush
  //    of the staged tail, no free of anything). The allocations stay
  //    live in vRAM (that leak is the reaper's job — see
  //    test/kv-reaper.test.ts), but the table is gone.
  const liveBefore = access.liveAllocations();
  cache = null;
  const replacement = new VramPagedKVCache(
    BLOCK_SIZE,
    TOTAL_BLOCKS,
    NUM_KV_HEADS,
    HEAD_DIM,
    access,
    { chunkTokens: CHUNK_TOKENS, replicas: 1 },
  );
  assert.equal(
    access.liveAllocations(),
    liveBefore,
    "dropping the cache must not free vRAM (crash path)",
  );
  await assert.rejects(
    replacement.read("seq-1", 0, [0]),
    (err: unknown) =>
      err instanceof VramKvError && /no blocks allocated/.test(err.message),
    "a fresh cache must not be able to read the dead worker's sequence",
  );
  // free() of the unknown sequence is a clean no-op, so the replacement
  // starts from a known-empty state.
  await replacement.free("seq-1");
  console.log("replacement cache cannot re-attach to the dead worker's KV (VramKvError): ok");

  // 4. Resubmit: re-prefill the same tokens into the replacement cache
  //    and compare full-prefix reads bit-for-bit with the original.
  const resubmittedReads = await prefill(replacement, "seq-1");
  for (let l = 0; l < LAYERS.length; l++) {
    assert.deepEqual(
      Array.from((resubmittedReads[l] as { keys: Float32Array }).keys),
      Array.from((originalReads[l] as { keys: Float32Array }).keys),
      `layer ${LAYERS[l]} keys must be identical after resubmit`,
    );
    assert.deepEqual(
      Array.from((resubmittedReads[l] as { values: Float32Array }).values),
      Array.from((originalReads[l] as { values: Float32Array }).values),
      `layer ${LAYERS[l]} values must be identical after resubmit`,
    );
  }
  console.log("resubmit reproduces the KV bit-for-bit (both layers): ok");

  // 5. The quantitative point: at production geometry nothing ever
  //    flushes below 1024 tokens, so there is nothing to hand over in
  //    the viable regime. Demo-100m geometry (kvDim 512) must give
  //    chunkTokens 1024, and the host cache agrees the workload fits.
  const prodCache = new VramPagedKVCache(16, 4096, 8, 64, access);
  assert.equal(prodCache.getChunkTokens(), 1024, "demo-100m geometry must chunk at 1024 tokens");
  const host = new BlockPagedKVCache(16, 1024, 8, 64);
  assert.equal(host.totalBlocks(), 1024);
  console.log(
    "production chunkTokens = 1024: sequences shorter than 1024 tokens never flush, " +
      "so a handover would save nothing where vRAM KV is viable",
  );

  console.log("kv-handover-resubmit test passed (resubmit is the contract)");
}

main().catch((err: unknown) => {
  console.error("kv-handover-resubmit test FAILED:", err);
  process.exitCode = 1;
});
