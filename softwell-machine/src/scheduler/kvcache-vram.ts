/**
 * softwell-machine — scheduler: vRAM-backed block-paged KV cache.
 *
 * STATUS (2026-11-23, KV-cache-in-vRAM phase): WIRED IN behind the
 * `--kv-store vram` worker flag (default stays `host`). This class now
 * declares `implements PagedKVCache`: the interface went async
 * (docs/KV-DESIGN.md), so the async shape this file always had is now
 * the interface itself. HARDENED 2026-11-22 (workstream 5): globally-
 * unique block ids, block-granular unwritten-read throws, an infallible
 * free(), serialized concurrent operations, short-read rejection, and
 * a constructor guard against a totalBlocks smaller than one chunk.
 * What remains: long-context decode still costs ~+75% at ctx 1024
 * (item 2 below), so the default stays `host` until the read-cost
 * work lands (sub-page reads, or a vGPU attention kernel that reads
 * KV in place on the donor).
 *
 * CORRECTED 2026-11-22: the previous header's numbers
 * ("host ~0.02 ms, vRAM ~19 ms, ~1,000x slower", "~230 ms per decode
 * token", "2-4.5x decode slowdown") were written before any bench run
 * completed — no results file existed. Three real runs now exist (runs
 * 1, 2, 4; run 3 was killed by a daemon restart; run 4's results JSON
 * is test/kvcache-vram-bench-results-2026-11-22.json, run 1/2 numbers
 * survive in the bench logs quoted in the workstream report). The real
 * picture is much less extreme — and the no-go now rests on the
 * sync-interface blocker plus a material long-context cost, not on a
 * 1,000x read slowdown.
 *
 * Measured on demo-100m geometry (kvDim 512, 12 layers, blockSize 16,
 * chunk 1024 tokens = one 4 MiB vRAM page) with a real in-process
 * VirtualRam + two real donor RamNodes over loopback HTTP (binary page
 * transport, replicas=2; see test/kvcache-vram-bench.ts). All timings
 * are warm medians after 1 warmup; ranges are across the 3 runs.
 *   - Full-context read, ONE layer:
 *       ctx 128:  host 0.13-0.19 ms, vRAM 0.10-0.39 ms (~1x, parity)
 *       ctx 1024: host 1.2-4.7 ms,   vRAM 14-99 ms    (~6-60x; best
 *                 single-page fetch 7.1 ms, cold reads ~99-127 ms)
 *   - Full-context read, ALL 12 layers sequential (the per-decode-token
 *     cost, as the engine runs them):
 *       ctx 128:  host 1.3-4.0 ms,   vRAM 1.4-17.8 ms (~1-4x)
 *       ctx 1024: host 65-285 ms,    vRAM 394-1792 ms (~6-9x — the
 *                 ratio is stable across runs even as absolutes swing
 *                 4x with ambient load; both sides inflate together)
 *   - Appends (staged design): 0.06-0.30 ms/token/layer vs host
 *     0.003-0.004 (15-80x slower, but 0.8-3.6 ms/token across 12
 *     layers — negligible next to decode). A naive per-token vRAM
 *     write costs 19-200 ms per 4 KiB token (a read-modify-write of
 *     a whole 4 MiB page: fetch + store); staging is mandatory and
 *     works. Alloc of one 4 MiB chunk (replicas=2): 38-212 ms,
 *     amortized over 1024 tokens — negligible per token.
 *   - Projected decode (host decode + measured vRAM read per token;
 *     a projection — end-to-end numbers are measured in the KV phase
 *     with the async engine driving this cache; see bench-kv-after.json):
 *       ctx 128:  +0.4% (run 4) to +3.5% (run 2) — negligible.
 *       ctx 1024: +75% in run 4 (0.42 -> 0.24 tok/s) — material.
 *     Absolute tok/s in the bench (0.9-1.9 tok/s at ctx 128) is far
 *     below the coordinator's live-server proof (5.10 tok/s): the
 *     bench shares a 1.5-CPU cell with other workstreams running
 *     heavy benches concurrently, so absolute times are load-dominated
 *     and ratios/projections within a run are the reliable signal.
 * Cost that remains (why the default is still `host`):
 *   1. Long-context decode pays ~+75% (up to ~2x) for vRAM KV — too
 *      much to accept by default at ctx 1024. Short contexts are
 *      nearly free, so performance alone would be GO-WITH-CONDITIONS
 *      (short contexts, or prefill-only), not an outright rejection.
 * What would have to change for the default to flip:
 *   1. A sub-millisecond small-read path in vRAM (sub-page transfers
 *      or a KV side-channel that does not move whole 4 MiB pages), or
 *      attention that does not re-read the full prefix every token
 *      (e.g. a vGPU attention kernel that reads KV in-place on the
 *      donor, so KV never crosses HTTP per token).
 *   2. For long contexts, the ~6-9x read ratio has to come down or be
 *      accepted as a capacity-vs-speed trade.
 *
 * HISTORY — the sync-interface problem, solved 2026-11-23
 * (docs/KV-DESIGN.md): PagedKVCache used to be synchronous and
 * TransformerStageExecutor.forward() with it, so no remote (async)
 * backing store could satisfy the interface. The interface and
 * forward() went async instead of building a blocking transport
 * (a worker-thread + Atomics.wait bridge was evaluated and rejected:
 * it would block the worker's event loop for ~20-150 s on a single
 * prefill, tripping the orchestrator's heartbeat timeouts).
 *
 * DESIGN
 * ------
 * Layout — one vRAM allocation per (sequence, layer) CHUNK, not per
 * block and not one giant allocation:
 *   - Per-block allocations are unusable: vRAM allocates in whole
 *     4 MiB pages, so one 64 KiB block (16 tokens x kvDim 512 x 4 B x
 *     K+V) per allocation wastes 64x the bytes, and alloc() zero-fills
 *     every page on every donor (replicated), so block-granular alloc
 *     would also pay a full page store per 16 tokens.
 *   - One giant allocation per (sequence, layer) cannot grow: the
 *     cache grows block by block as a sequence decodes, and vRAM has
 *     no realloc. Growing would mean a new bigger allocation plus a
 *     full copy of everything written so far.
 *   - Chunks split the difference. chunkTokens is the largest multiple
 *     of blockSize whose bytes (chunkTokens x 2 x kvDim x 4) fit in
 *     one 4 MiB vRAM page — 1024 tokens at demo-100m geometry, so one
 *     chunk is exactly one page: zero waste for full chunks, at most
 *     one page of waste per (sequence, layer) tail, and alloc cost is
 *     amortized over 1024 tokens. Chunks are independent allocations,
 *     so growth is one alloc per 1024 tokens and free() returns whole
 *     chunks to vRAM (the coordinator reuses freed address ranges).
 *   - Within a chunk, K and V are INTERLEAVED per token:
 *     token t lives at [t*2*kvDim*4, (t+1)*2*kvDim*4), K first then V.
 *     A contiguous run of positions is therefore one contiguous byte
 *     range and one vRAM read — the engine always reads a full prefix,
 *     so a full-context read is one read per chunk, not one per token.
 *
 * Appends — host-side write-back staging, never a per-token vRAM write:
 *   - A per-token vRAM write is a read-modify-write of a whole 4 MiB
 *     page (VirtualRam.write fetches the page, patches 4 KiB, stores
 *     it back): measured ~40 ms per token per layer. Unusable.
 *   - Appends therefore land in a host staging buffer for the chunk.
 *     When a chunk fills (1024 tokens), the whole page is flushed with
 *     ONE full-page vRAM write (no read-modify-write: the write covers
 *     the entire page). Amortized append cost is then ~0.02 ms/token
 *     at demo geometry. The price is bounded host staging memory:
 *     at most one chunk per (sequence, layer) is staged at a time
 *     (<= 4 MiB x 12 layers = 48 MiB per sequence at demo geometry,
 *     transient — it drops to ~0 once the chunk fills and flushes).
 *   - Out-of-order appends to an already-flushed chunk fetch the page
 *     back into staging first (one RMW, only on that rare path); the
 *     engine appends in position order, so this never happens in
 *     normal decode/prefill.
 *   - Staged data is NOT yet replicated: a worker crash loses staged
 *     KV exactly as a crash loses the host cache today (the scheduler
 *     re-prefills the sequence). Only flushed chunks enjoy vRAM
 *     replication. This matches the host cache's failure semantics;
 *     vRAM adds donor-loss tolerance for flushed data, not worker-loss
 *     tolerance for in-flight data. And the tolerance stops at the
 *     chunk TABLE: it lives only in this instance (which allocIds
 *     belong to which seq/layer), so a replacement worker cannot
 *     re-attach to flushed chunks even though their bytes survive in
 *     vRAM — it must resubmit. Proven by
 *     test/kvcache-vram-survival.test.ts; resuming would need a
 *     snapshot/attach handover that does not exist yet.
 *
 * Reads — coalesced per chunk, staging-aware:
 *   - Requested positions are grouped by chunk; consecutive positions
 *     in one chunk become one byte range and one vRAM read. Reads of
 *     staged chunks are served from the host staging buffer (no vRAM
 *     traffic). Reads of flushed chunks fetch from vRAM.
 *   - A read of a position in a chunk no append ever touched throws.
 *     A read of an unwritten position inside a touched chunk returns
 *     zeros (the chunk's pages are zero-filled on alloc and a flush
 *     writes the zero tail with them). This is BLOCK-granular since
 *     2026-11-22 (workstream 5): each chunk tracks a per-block
 *     written flag, so reads throw exactly when BlockPagedKVCache
 *     throws. The prototype was chunk-granular (zeros for any
 *     position in a touched chunk, and flushAll() on an untouched
 *     chunk turned a throw into zeros) — verified by probe, fixed,
 *     and covered by tests 12 and 19. The two caches now agree on
 *     every read the engine can issue, not just on appended prefixes.
 *   - A read that comes back short (fewer bytes than requested) or a
 *     re-fetch that comes back short throws instead of serving
 *     partial data.
 *
 * Replication / failure semantics — inherited from vRAM, not re-invented:
 *   - Every chunk allocation is created with `replicas` copies
 *     (default 2, matching the CLI default). VirtualRam places copies
 *     on different donors and handleNodeLoss() re-replicates pages
 *     that lost a copy from a surviving copy. A chunk therefore
 *     survives any single donor loss with replicas=2.
 *   - A read that finds a page unavailable on ALL copies throws a
 *     VramKvError (cause: the underlying VramError) — committed KV is
 *     never silently replaced with zeros or dropped. Recovery belongs
 *     to the scheduler, which owns the prompt: drop the sequence's KV
 *     (free) and re-prefill from its prompt tokens (recompute).
 *   - With replicas=1 a donor loss CAN lose committed KV: handleNodeLoss
 *     reports pagesLost > 0 and the next read throws. Do not run this
 *     cache with replicas=1 unless the scheduler already re-prefills
 *     on KV errors.
 *
 * Exhaustion — no silent fallback:
 *   - When vRAM cannot allocate (no donors, capacity exhausted) or
 *     the configured totalBlocks cap is reached, allocate()/append()
 *     reject with a VramKvError naming the sequence, layer and cause.
 *     There is NO silent fallback to host memory: a fallback would
 *     silently change failure semantics (host KV does not survive
 *     donor loss) and hide capacity problems. If the coordinator ever
 *     wants a fallback it must be an explicit, observable policy
 *     (counter + log line per fallback), not a default.
 *
 * Concurrency: the engine drives one cache single-threaded (one
 * forward at a time per stage), so appends/reads to the same chunk
 * are not concurrent in practice — but the prototype did not
 * serialize concurrent operations at all, and a probe showed 32
 * concurrent appends to one chunk creating 32 chunks (and 3
 * concurrent allocate()s creating 3). Since 2026-11-22 (workstream
 * 5) every public operation runs through a FIFO promise chain, so
 * concurrent callers are serialized and the cap check, chunk growth
 * and flush cannot interleave. Do not drive it from two forwards at
 * once and expect parallelism — serialization is a correctness
 * guard, not a concurrency design.
 *
 * HARDENING AUDIT 2026-11-22 (workstream 5) — method by method vs
 * BlockPagedKVCache (src/scheduler/kvcache.ts). Fixed:
 *   - allocate(): block ids were chunk-local (0,1,...) and collided
 *     across sequences AND layers; now globally unique, monotonic,
 *     never reused after free. Verified by probe (P1) and test 11.
 *   - allocate(): the returned table is sliced to the logical blocks
 *     requested, so its length matches the host on growing
 *     allocates; on a SHRINKING re-allocate the host returns its
 *     full table while this returns the shorter prefix — STANDING
 *     divergence, pinned by test 11. The engine only grows.
 *   - read(): unwritten-block reads returned zeros (chunk-granular)
 *     and flushAll() on an untouched chunk turned a throw into
 *     zeros; now block-granular throws, host-identical. Tests 12, 19.
 *   - read()/ensureStaged(): short reads were served as partial
 *     data; now they throw. Test 19.
 *   - free(): threw when a vRAM free failed (after dropping local
 *     state, leaking the alloc id) — and the decoders call free()
 *     at generate() start with no try/catch, so a throw would abort
 *     generation; the scheduler's freeSequence sites swallow
 *     failures, so only the decoder path was exposed. free() now
 *     never rejects; failures are counted in stats().freeFailures.
 *     Test 18.
 *   - allocate()/append(): concurrent callers double-allocated
 *     chunks (32 chunks for 32 concurrent appends to one chunk);
 *     now serialized. Test 13.
 *   - Constructor: totalBlocks smaller than one chunk made every
 *     allocate reject with a confusing OOM; now a loud error at
 *     construction. Test 15.
 * Standing (not fixable in this class, or deliberately kept):
 *   - usedBlocks() counts whole reserved chunks (up to 63 extra
 *     blocks per (seq, layer) at demo geometry) while the host
 *     counts logical blocks. The scheduler (batch.ts) does not use
 *     usedBlocks() for admission today (maxSequences + token
 *     budget), so no live caller is misled; logicalBlocks() is the
 *     host-equivalent count for future admission, usedBlocks() the
 *     capacity backstop. Test 19 quantifies both.
 *   - allocate() still reserves a whole chunk per (seq, layer) the
 *     moment one token is appended — minimum allocation is one
 *     4 MiB page; that is the chunking design, not a bug.
 *   - The async twin still does NOT implement PagedKVCache (the
 *     sync-interface problem below) — wiring still needs the bridge.
 * Out-of-order appends (verified against the real paths): the
 * transformer sorts rows per sequence by position
 * (transformer.ts forward()), so prefill and autoregressive decode
 * never append out of order. Speculative decoding
 * (batch.ts decodeSpeculative, novel-decoding.ts SpeculativeDecoder)
 * and Jacobi (JacobiDecoder) overwrite a rejected draft's K/V at its
 * position via append(); that is in-order except when the overwritten
 * position's chunk already flushed — once per 1024-token chunk
 * boundary per layer at demo geometry — and then it costs one RMW
 * fetch + one re-flush, which is correct (test 10 / probe P10) but
 * slow, so speculative/Jacobi at a chunk boundary pays ~40 ms per
 * affected layer. Append-after-free re-creates the sequence cleanly
 * (test 14). Allocate beyond the cap rejects with VramKvError and
 * leaves the cache usable, never falling back to host (test 15).
 */

import { VRAM_PAGE_SIZE_BYTES } from "../contracts.ts";
import type { IVirtualRam, PagedKVCache, VramAccess } from "../contracts.ts";

// ---------------------------------------------------------------------------
// VramAccess — the minimal remote-vRAM surface this cache needs.
// ---------------------------------------------------------------------------
// The interface lives in src/contracts.ts (next to IVirtualRam) so the
// worker, the mesh client, and tests can reference it without importing
// this file. Implementations: VirtualRamAccess below (in-process, used
// by the measurement harness) and RemoteVramAccess
// (src/scheduler/kvcache-vram-remote.ts, HTTP to the orchestrator).

/**
 * VramAccess adapter over the in-process IVirtualRam coordinator
 * (src/hw/vram.ts). Used by the measurement harness; the production
 * wiring replaces this with the remote HTTP client.
 */
export class VirtualRamAccess implements VramAccess {
  private readonly ram: IVirtualRam;

  constructor(ram: IVirtualRam) {
    this.ram = ram;
  }

  read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.ram.read(vaddr, length);
  }

  write(vaddr: number, data: Uint8Array): Promise<void> {
    return this.ram.write(vaddr, data);
  }

  async alloc(
    sizeBytes: number,
    opts?: { replicas?: number },
  ): Promise<{ id: string; vaddr: number }> {
    const alloc = await this.ram.alloc(sizeBytes, opts);
    return { id: alloc.id, vaddr: alloc.vaddr };
  }

  free(id: string): Promise<void> {
    return this.ram.free(id);
  }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** All failures raised by VramPagedKVCache (OOM, vRAM errors, misuse). */
export class VramKvError extends Error {
  constructor(message: string, opts?: { cause?: unknown }) {
    super(message, opts?.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = "VramKvError";
  }
}

// ---------------------------------------------------------------------------
// VramPagedKVCache — async twin of PagedKVCache, backed by VramAccess.
// ---------------------------------------------------------------------------

export interface VramPagedKVCacheOptions {
  /** Override the computed chunk size (tokens per chunk). Tests only:
   *  forces multi-chunk behaviour at small geometries. */
  chunkTokens?: number;
  /** Copies per chunk allocation. Default 2 (matches the CLI default). */
  replicas?: number;
}

interface Chunk {
  allocId: string;
  vaddr: number;
  /** Physical block ids for this chunk's slots, globally unique across
   *  the cache (assigned from a monotonic counter at chunk creation,
   *  never reused after free — the host cache reuses ids via its
   *  free-list, but reuse is not part of the contract and fresh ids
   *  avoid ABA staleness if a caller ever holds a table past free). */
  blockIds: number[];
  /** Per-block written flags (1 = some position in the block was
   *  appended). Read throws for a position in an unwritten block,
   *  exactly like BlockPagedKVCache — chunk-granular zeros were a
   *  prototype divergence, fixed 2026-11-22 (workstream 5). */
  writtenBlocks: Uint8Array;
  /** Host staging buffer (chunkBytes), null once flushed to vRAM. */
  staging: Uint8Array | null;
  /** Float32 view over `staging` (null when staging is null). */
  stagingF32: Float32Array | null;
  /** Highest position+1 appended into this chunk (0 = untouched). */
  writtenUpTo: number;
  /** True once the chunk's contents live in vRAM (staging dropped). */
  flushed: boolean;
}

export interface VramKvStats {
  /** Chunks currently allocated (all sequences, all layers). */
  chunks: number;
  /** Bytes currently staged on the host (unflushed chunks). */
  stagedBytes: number;
  /** Bytes currently held in vRAM (flushed chunks). */
  vramBytes: number;
  /** vRAM reads issued (one per coalesced run). */
  vramReads: number;
  /** vRAM writes issued (one per flushed chunk, plus RMW fetches). */
  vramWrites: number;
  /** Bytes read from vRAM (fetched pages, incl. RMW fetches). */
  bytesRead: number;
  /** Bytes written to vRAM (full-chunk flushes). */
  bytesWritten: number;
  /** Chunk flushes (full-page writes). */
  chunkFlushes: number;
  /** Read-modify-write fetches (out-of-order appends to flushed chunks). */
  rmwFetches: number;
  /** vRAM frees that failed inside free() (local state was still
   *  dropped; the allocation id is unrecoverable, so the bytes may
   *  leak until the donor heals or restarts — counted, never thrown,
   *  so free() cannot break generation). */
  freeFailures: number;
}

export class VramPagedKVCache implements PagedKVCache {
  readonly blockSize: number;
  private readonly total: number;
  private readonly numKvHeads: number;
  private readonly headDim: number;
  private readonly kvDim: number;
  private readonly vram: VramAccess;
  private readonly replicas: number;
  /** Bytes of K+V for one token (interleaved layout). */
  private readonly bytesPerToken: number;
  /** Tokens per vRAM chunk (one page at demo geometry). */
  private readonly chunkTokens: number;
  /** Bytes per chunk. */
  private readonly chunkBytes: number;
  /** Blocks per chunk (chunkTokens / blockSize). */
  private readonly blocksPerChunk: number;
  /** seqId -> layer -> ordered list of chunks (chunk i covers logical
   *  positions [i * chunkTokens, (i+1) * chunkTokens)). */
  private readonly tables: Map<string, Map<number, Chunk[]>>;
  /** seqId -> layer -> largest numTokens ever requested by allocate()
   *  (drives logicalBlocks(), the host-equivalent admission count). */
  private readonly logicalTokens: Map<string, Map<number, number>>;
  private readonly counters: VramKvStats;
  /** Next globally-unique physical block id (monotonic, never reused). */
  private nextBlockId = 0;
  /** FIFO serialization chain: every public operation runs inside
   *  runSerialized(), so concurrent appends/allocates to the same
   *  chunk cannot double-allocate or lose a write to a chunk another
   *  append just flushed (verified by probe 2026-11-22: 32 concurrent
   *  appends to one chunk created 32 chunks pre-fix, 1 post-fix).
   *  The engine drives one forward at a time, so this costs nothing
   *  on the serving path; it only guards against concurrent callers. */
  private queue: Promise<void> = Promise.resolve();

  constructor(
    blockSize: number,
    totalBlocks: number,
    numKvHeads: number,
    headDim: number,
    vram: VramAccess,
    opts?: VramPagedKVCacheOptions,
  ) {
    if (!Number.isInteger(blockSize) || blockSize <= 0) {
      throw new VramKvError(`VramPagedKVCache: blockSize must be a positive integer, got ${blockSize}`);
    }
    if (!Number.isInteger(totalBlocks) || totalBlocks <= 0) {
      throw new VramKvError(`VramPagedKVCache: totalBlocks must be a positive integer, got ${totalBlocks}`);
    }
    if (!Number.isInteger(numKvHeads) || numKvHeads <= 0) {
      throw new VramKvError(`VramPagedKVCache: numKvHeads must be a positive integer, got ${numKvHeads}`);
    }
    if (!Number.isInteger(headDim) || headDim <= 0) {
      throw new VramKvError(`VramPagedKVCache: headDim must be a positive integer, got ${headDim}`);
    }
    if (opts?.replicas !== undefined && (!Number.isInteger(opts.replicas) || opts.replicas < 1)) {
      throw new VramKvError(`VramPagedKVCache: replicas must be a positive integer, got ${opts.replicas}`);
    }
    this.blockSize = blockSize;
    this.total = totalBlocks;
    this.numKvHeads = numKvHeads;
    this.headDim = headDim;
    this.kvDim = numKvHeads * headDim;
    this.vram = vram;
    this.replicas = opts?.replicas ?? 2;
    this.bytesPerToken = 2 * this.kvDim * 4;
    // Largest block-multiple of tokens whose bytes fit in one vRAM page.
    // At demo-100m geometry (kvDim 512) this is exactly 1024 tokens =
    // one 4 MiB page per chunk, zero waste for full chunks.
    const tokensPerPage = Math.floor(VRAM_PAGE_SIZE_BYTES / this.bytesPerToken);
    const computed =
      tokensPerPage >= blockSize
        ? Math.floor(tokensPerPage / blockSize) * blockSize
        : blockSize;
    this.chunkTokens =
      opts?.chunkTokens !== undefined ? opts.chunkTokens : computed;
    if (
      !Number.isInteger(this.chunkTokens) ||
      this.chunkTokens <= 0 ||
      this.chunkTokens % blockSize !== 0
    ) {
      throw new VramKvError(
        `VramPagedKVCache: chunkTokens must be a positive multiple of blockSize ${blockSize}, got ${this.chunkTokens}`,
      );
    }
    this.chunkBytes = this.chunkTokens * this.bytesPerToken;
    this.blocksPerChunk = this.chunkTokens / blockSize;
    if (this.total < this.blocksPerChunk) {
      throw new VramKvError(
        `VramPagedKVCache: totalBlocks ${this.total} is smaller than one chunk ` +
          `(${this.blocksPerChunk} blocks of ${blockSize} tokens) — no allocation could ever fit; ` +
          `raise totalBlocks to at least ${this.blocksPerChunk}`,
      );
    }
    this.tables = new Map();
    this.logicalTokens = new Map();
    this.counters = {
      chunks: 0,
      stagedBytes: 0,
      vramBytes: 0,
      vramReads: 0,
      vramWrites: 0,
      bytesRead: 0,
      bytesWritten: 0,
      chunkFlushes: 0,
      rmwFetches: 0,
      freeFailures: 0,
    };
  }

  /** Tokens per vRAM chunk (one page at demo-100m geometry). */
  getChunkTokens(): number {
    return this.chunkTokens;
  }

  /** Bytes per chunk (one vRAM page at demo-100m geometry). */
  getChunkBytes(): number {
    return this.chunkBytes;
  }

  /** Run `fn` after all previously queued operations finish, in
   *  FIFO order. Errors propagate to the caller but never break the
   *  chain. Public methods serialize through this; private `*Inner`
   *  methods call each other directly and must never re-enter it. */
  private runSerialized<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.queue.then(() => fn());
    this.queue = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /** Observable counters (see VramKvStats): read/write counts and bytes,
   *  so the coordinator can see the amplification this cache causes.
   *  chunks/stagedBytes/vramBytes are live snapshots; stagedBytes and
   *  vramBytes count LOGICAL chunk bytes (chunkBytes per chunk) — the
   *  physical vRAM holding is whole 4 MiB pages per chunk, which is
   *  equal at demo-100m geometry (chunkBytes is exactly one page) and
   *  larger for small test chunkTokens. */
  stats(): VramKvStats {
    let staged = 0;
    let vramBytes = 0;
    let chunks = 0;
    for (const perLayer of this.tables.values()) {
      for (const table of perLayer.values()) {
        for (const chunk of table) {
          chunks += 1;
          if (chunk.staging !== null) staged += this.chunkBytes;
          else if (chunk.flushed) vramBytes += this.chunkBytes;
        }
      }
    }
    return { ...this.counters, chunks, stagedBytes: staged, vramBytes };
  }

  /**
   * Grow the sequence's chunk list for `layer` so it can hold
   * `numTokens` tokens, allocating new chunks from vRAM as needed.
   * Returns the block table for that layer: physical block ids, in
   * order, globally unique across sequences and layers (unlike the
   * prototype, which numbered chunk slots 0..n per (seq, layer) and
   * collided across both). The table is stable: a block id never
   * changes while the sequence lives. Rejects with VramKvError on
   * out-of-memory (cap reached or vRAM alloc failed) — no silent
   * fallback to host memory.
   *
   * The returned table is sliced to the LOGICAL blocks the caller
   * asked for (ceil(numTokens / blockSize)), so its length matches
   * BlockPagedKVCache on every growing allocate — but the host cache
   * returns its FULL table, so on a shrinking re-allocate (fewer
   * tokens than before) the host returns the full table while this
   * returns the shorter prefix. The engine only ever grows
   * (positions increase), so this is a documented divergence, not a
   * live one. usedBlocks() still counts the whole reserved chunks —
   * that is what vRAM actually holds; logicalBlocks() is the
   * host-equivalent count for admission control.
   */
  async allocate(seqId: string, layer: number, numTokens: number): Promise<number[]> {
    return this.runSerialized(() => this.allocateInner(seqId, layer, numTokens));
  }

  private async allocateInner(seqId: string, layer: number, numTokens: number): Promise<number[]> {
    if (!Number.isInteger(numTokens) || numTokens < 0) {
      throw new VramKvError(
        `VramPagedKVCache.allocate: numTokens must be a non-negative integer, got ${numTokens}`,
      );
    }
    let perLayer = this.tables.get(seqId);
    if (perLayer === undefined) {
      perLayer = new Map();
      this.tables.set(seqId, perLayer);
    }
    let table = perLayer.get(layer);
    if (table === undefined) {
      table = [];
      perLayer.set(layer, table);
    }
    // Track the largest request per (seq, layer) for logicalBlocks().
    let perLayerLogical = this.logicalTokens.get(seqId);
    if (perLayerLogical === undefined) {
      perLayerLogical = new Map();
      this.logicalTokens.set(seqId, perLayerLogical);
    }
    perLayerLogical.set(layer, Math.max(perLayerLogical.get(layer) ?? 0, numTokens));
    const needed = Math.ceil(numTokens / this.chunkTokens);
    if ((this.usedBlocks() + (needed - table.length) * this.blocksPerChunk) > this.total) {
      throw new VramKvError(
        `VramPagedKVCache out of memory: block cap reached (total=${this.total}, used=${this.usedBlocks()}) ` +
          `while allocating for seq ${seqId} layer ${layer}`,
      );
    }
    while (table.length < needed) {
      const chunkIndex = table.length;
      let alloc: { id: string; vaddr: number };
      try {
        alloc = await this.vram.alloc(this.chunkBytes, { replicas: this.replicas });
      } catch (err) {
        throw new VramKvError(
          `VramPagedKVCache out of memory: vRAM alloc failed for seq ${seqId} layer ${layer} chunk ${chunkIndex} ` +
            `(${this.chunkBytes} bytes, replicas=${this.replicas})`,
          { cause: err },
        );
      }
      const staging = new Uint8Array(this.chunkBytes);
      const blockIds: number[] = [];
      for (let b = 0; b < this.blocksPerChunk; b++) blockIds.push(this.nextBlockId++);
      table.push({
        allocId: alloc.id,
        vaddr: alloc.vaddr,
        blockIds,
        writtenBlocks: new Uint8Array(this.blocksPerChunk),
        staging,
        stagingF32: new Float32Array(staging.buffer),
        writtenUpTo: 0,
        flushed: false,
      });
    }
    return this.blockTable(table).slice(0, Math.ceil(numTokens / this.blockSize));
  }

  private blockTable(table: Chunk[]): number[] {
    const out: number[] = [];
    for (const chunk of table) out.push(...chunk.blockIds);
    return out;
  }

  /** Fetch a flushed chunk back into host staging (read-modify-write
   *  path, only for out-of-order appends to a flushed chunk). */
  private async ensureStaged(chunk: Chunk, seqId: string, layer: number): Promise<void> {
    if (chunk.staging !== null) return;
    let data: Uint8Array;
    try {
      data = await this.vram.read(chunk.vaddr, this.chunkBytes);
    } catch (err) {
      throw new VramKvError(
        `VramPagedKVCache: failed to re-fetch flushed chunk for seq ${seqId} layer ${layer} ` +
          `(alloc ${chunk.allocId}) — committed KV may be unrecoverable; re-prefill the sequence`,
        { cause: err },
      );
    }
    this.counters.vramReads += 1;
    this.counters.bytesRead += data.length;
    this.counters.rmwFetches += 1;
    if (data.length !== this.chunkBytes) {
      throw new VramKvError(
        `VramPagedKVCache: re-fetch of flushed chunk for seq ${seqId} layer ${layer} ` +
          `(alloc ${chunk.allocId}) returned ${data.length} bytes, expected ${this.chunkBytes} — ` +
          `committed KV may be unrecoverable; re-prefill the sequence`,
      );
    }
    const staging = new Uint8Array(this.chunkBytes);
    staging.set(data);
    chunk.staging = staging;
    chunk.stagingF32 = new Float32Array(staging.buffer);
    chunk.flushed = false;
  }

  /** Flush a full staged chunk to vRAM with one whole-page write. */
  private async flushChunk(chunk: Chunk, seqId: string, layer: number): Promise<void> {
    if (chunk.staging === null) return;
    const staging = chunk.staging;
    try {
      await this.vram.write(chunk.vaddr, staging);
    } catch (err) {
      throw new VramKvError(
        `VramPagedKVCache: failed to flush chunk for seq ${seqId} layer ${layer} ` +
          `(alloc ${chunk.allocId}); staged data is still held in host memory and the append is not lost, ` +
          `but the chunk is not replicated — retry the flush or free the sequence`,
        { cause: err },
      );
    }
    this.counters.vramWrites += 1;
    this.counters.bytesWritten += staging.length;
    this.counters.chunkFlushes += 1;
    chunk.staging = null;
    chunk.stagingF32 = null;
    chunk.flushed = true;
  }

  /**
   * Flush every staged chunk (all sequences, all layers) to vRAM.
   * Call before a graceful shutdown or before handing a sequence's KV
   * to another worker, so no committed KV sits un-replicated in host
   * staging. Partial chunks are flushed whole (their zero tail goes
   * with them, matching the read semantics for unwritten positions).
   */
  async flushAll(): Promise<void> {
    return this.runSerialized(() => this.flushAllInner());
  }

  private async flushAllInner(): Promise<void> {
    for (const [seqId, perLayer] of this.tables) {
      for (const [layer, table] of perLayer) {
        for (const chunk of table) {
          await this.flushChunk(chunk, seqId, layer);
        }
      }
    }
  }

  /**
   * Write one token's K/V for `layer` at absolute `position`.
   * Appends go to host staging; a chunk is flushed to vRAM exactly
   * once, when it fills. Rejects with VramKvError on OOM or on a
   * failed chunk flush. Overwriting a position in an already-flushed
   * chunk is correct (read-modify-write fetch, then re-flush) — the
   * engine only does this for speculative/Jacobi corrections, and
   * only when the corrected position sits in a chunk that already
   * filled, i.e. at most once per chunk boundary per layer.
   */
  async append(
    seqId: string,
    layer: number,
    position: number,
    key: Float32Array,
    value: Float32Array,
  ): Promise<void> {
    return this.runSerialized(() => this.appendInner(seqId, layer, position, key, value));
  }

  private async appendInner(
    seqId: string,
    layer: number,
    position: number,
    key: Float32Array,
    value: Float32Array,
  ): Promise<void> {
    if (!Number.isInteger(position) || position < 0) {
      throw new VramKvError(`VramPagedKVCache.append: position must be a non-negative integer, got ${position}`);
    }
    if (key.length !== this.kvDim || value.length !== this.kvDim) {
      throw new VramKvError(
        `VramPagedKVCache.append: key/value must have ${this.kvDim} floats ` +
          `(numKvHeads=${this.numKvHeads} * headDim=${this.headDim}), ` +
          `got key.length=${key.length}, value.length=${value.length}`,
      );
    }
    await this.allocateInner(seqId, layer, position + 1);
    const perLayer = this.tables.get(seqId) as Map<number, Chunk[]>;
    const chunks = perLayer.get(layer) as Chunk[];
    const chunkIndex = Math.floor(position / this.chunkTokens);
    const chunk = chunks[chunkIndex] as Chunk;
    const offsetInChunk = position % this.chunkTokens;
    await this.ensureStaged(chunk, seqId, layer);
    const stagingF32 = chunk.stagingF32 as Float32Array;
    const base = offsetInChunk * 2 * this.kvDim;
    stagingF32.set(key, base);
    stagingF32.set(value, base + this.kvDim);
    chunk.writtenBlocks[Math.floor(offsetInChunk / this.blockSize)] = 1;
    if (offsetInChunk + 1 > chunk.writtenUpTo) {
      chunk.writtenUpTo = offsetInChunk + 1;
    }
    if (chunk.writtenUpTo === this.chunkTokens) {
      await this.flushChunk(chunk, seqId, layer);
    }
  }

  /**
   * Concatenated K and V for `positions` (in the requested order) for
   * one layer. Each returned array has shape [positions.length, kvDim].
   * Consecutive positions in one chunk are fetched with a single
   * coalesced vRAM read; staged chunks are served from host memory.
   * A read of a position in a block no append ever touched throws,
   * exactly like BlockPagedKVCache (block-granular since 2026-11-22;
   * the prototype returned chunk-granular zeros, and flushAll() on an
   * untouched chunk even turned a throw into zeros). A read of an
   * unwritten position inside a TOUCHED block returns zeros, like the
   * host cache. Rejects with VramKvError when a needed vRAM page is
   * unavailable on all copies, or when a read comes back short —
   * committed KV is never silently zero-filled.
   */
  async read(
    seqId: string,
    layer: number,
    positions: number[],
  ): Promise<{ keys: Float32Array; values: Float32Array }> {
    return this.runSerialized(() => this.readInner(seqId, layer, positions));
  }

  private async readInner(
    seqId: string,
    layer: number,
    positions: number[],
  ): Promise<{ keys: Float32Array; values: Float32Array }> {
    const perLayer = this.tables.get(seqId);
    const chunks = perLayer?.get(layer);
    if (chunks === undefined) {
      throw new VramKvError(
        `VramPagedKVCache.read: no blocks allocated for seq ${seqId} layer ${layer}`,
      );
    }
    const keys = new Float32Array(positions.length * this.kvDim);
    const values = new Float32Array(positions.length * this.kvDim);
    // Group into runs: consecutive positions in the same chunk, in
    // increasing order, form one contiguous byte range.
    let i = 0;
    while (i < positions.length) {
      const startPos = positions[i] as number;
      if (!Number.isInteger(startPos) || startPos < 0) {
        throw new VramKvError(
          `VramPagedKVCache.read: position must be a non-negative integer, got ${startPos}`,
        );
      }
      const chunkIndex = Math.floor(startPos / this.chunkTokens);
      const chunk = chunks[chunkIndex];
      if (chunk === undefined) {
        throw new VramKvError(
          `VramPagedKVCache.read: position ${startPos} not allocated for seq ${seqId} layer ${layer}`,
        );
      }
      // Extend the run while positions stay consecutive in this chunk.
      let j = i + 1;
      while (j < positions.length) {
        const p = positions[j] as number;
        if (
          !Number.isInteger(p) ||
          p < 0 ||
          p !== (positions[j - 1] as number) + 1 ||
          Math.floor(p / this.chunkTokens) !== chunkIndex
        ) {
          break;
        }
        j += 1;
      }
      const runLength = j - i;
      const firstOffset = startPos % this.chunkTokens;
      // Block-granular written check, before any vRAM traffic: every
      // position in the run must sit in a block an append touched.
      // (Checked per position, not per run, so a run spanning a
      // written block, an unwritten block and a written block throws.)
      for (let t = 0; t < runLength; t++) {
        const blockInChunk = Math.floor((firstOffset + t) / this.blockSize);
        if (chunk.writtenBlocks[blockInChunk] !== 1) {
          throw new VramKvError(
            `VramPagedKVCache.read: no K/V written yet for seq ${seqId} layer ${layer} position ${startPos + t}`,
          );
        }
      }
      if (chunk.staging !== null) {
        // Served from host staging: no vRAM traffic.
        const stagingF32 = chunk.stagingF32 as Float32Array;
        for (let t = 0; t < runLength; t++) {
          const src = (firstOffset + t) * 2 * this.kvDim;
          keys.set(stagingF32.subarray(src, src + this.kvDim), (i + t) * this.kvDim);
          values.set(
            stagingF32.subarray(src + this.kvDim, src + 2 * this.kvDim),
            (i + t) * this.kvDim,
          );
        }
      } else {
        // One coalesced vRAM read for the whole run.
        const byteStart = firstOffset * this.bytesPerToken;
        const byteLength = runLength * this.bytesPerToken;
        let data: Uint8Array;
        try {
          data = await this.vram.read(chunk.vaddr + byteStart, byteLength);
        } catch (err) {
          throw new VramKvError(
            `VramPagedKVCache.read: failed to fetch KV for seq ${seqId} layer ${layer} ` +
              `positions ${startPos}..${startPos + runLength - 1} (alloc ${chunk.allocId}) — ` +
              `the page is unavailable on all copies; re-prefill the sequence`,
            { cause: err },
          );
        }
        if (data.length !== byteLength) {
          throw new VramKvError(
            `VramPagedKVCache.read: vRAM read for seq ${seqId} layer ${layer} ` +
              `positions ${startPos}..${startPos + runLength - 1} returned ${data.length} bytes, ` +
              `expected ${byteLength} — re-prefill the sequence`,
          );
        }
        this.counters.vramReads += 1;
        this.counters.bytesRead += data.length;
        // Float32Array views need a 4-byte-aligned offset; a backing
        // store that hands back an unaligned slice is copied first.
        const aligned =
          data.byteOffset % 4 === 0
            ? data
            : Uint8Array.from(data);
        const f32 = new Float32Array(
          aligned.buffer,
          aligned.byteOffset,
          aligned.length / 4,
        );
        for (let t = 0; t < runLength; t++) {
          const src = t * 2 * this.kvDim;
          keys.set(f32.subarray(src, src + this.kvDim), (i + t) * this.kvDim);
          values.set(f32.subarray(src + this.kvDim, src + 2 * this.kvDim), (i + t) * this.kvDim);
        }
      }
      i = j;
    }
    return { keys, values };
  }

  /**
   * Release every chunk owned by `seqId`, across all layers, back to
   * vRAM. No-op if unknown. Local state is dropped even when a vRAM
   * free fails, and free() NEVER rejects: the host cache's free() is
   * infallible, and the decoders call free() at the start of
   * generate() (novel-decoding.ts) with no try/catch — a throw there
   * would abort the whole generation — while the scheduler's
   * freeSequence call sites already swallow failures. A failed vRAM
   * free is counted in stats().freeFailures instead (the allocation
   * id is gone, so the bytes may leak until donor healing/restart).
   * The prototype threw here; that was a wiring hazard, fixed
   * 2026-11-22 (workstream 5).
   */
  async free(seqId: string): Promise<void> {
    return this.runSerialized(() => this.freeInner(seqId));
  }

  private async freeInner(seqId: string): Promise<void> {
    const perLayer = this.tables.get(seqId);
    if (perLayer === undefined) return;
    this.tables.delete(seqId);
    this.logicalTokens.delete(seqId);
    const frees: Array<Promise<void>> = [];
    for (const table of perLayer.values()) {
      for (const chunk of table) {
        frees.push(
          this.vram.free(chunk.allocId).catch(() => {
            this.counters.freeFailures += 1;
          }),
        );
      }
    }
    await Promise.all(frees);
  }

  /**
   * Release everything this cache holds: flush staged chunks to vRAM
   * first (so no committed KV sits un-replicated in host staging at
   * shutdown / model switch), then free every chunk of every sequence.
   * Best-effort like free(): a flush or free that fails is counted in
   * stats().freeFailures and the rest still release — close() never
   * rejects, so the worker's shutdown path cannot fail on it.
   */
  async close(): Promise<void> {
    return this.runSerialized(() => this.closeInner());
  }

  private async closeInner(): Promise<void> {
    // Flush first; a failing flush must not stop the frees (the staged
    // bytes belong to sequences that are being abandoned — the scheduler
    // re-prefills them — so dropping them is consistent with free()).
    try {
      await this.flushAllInner();
    } catch {
      this.counters.freeFailures += 1;
    }
    const seqIds = [...this.tables.keys()];
    for (const seqId of seqIds) {
      await this.freeInner(seqId);
    }
  }

  /**
   * Blocks currently reserved (whole chunks count fully). This is
   * what vRAM actually holds, and what allocate() enforces the cap
   * against — so it is the honest number for capacity. It OVER-counts
   * against the host cache: one 17-token sequence reserves a whole
   * 64-block chunk per layer at demo geometry (1024 tokens/chunk),
   * where the host counts 2 blocks. The scheduler (batch.ts) does
   * not consult usedBlocks() for admission today — admission is
   * maxSequences + token budget, despite contracts.ts claiming the
   * scheduler "only needs usedBlocks() for admission control" — so
   * no live caller is misled; any future admission control must use
   * logicalBlocks() for host-like admission and treat usedBlocks()
   * as the hard capacity backstop, or it will under-admit by up to
   * (blocksPerChunk - 1) blocks per (sequence, layer).
   */
  usedBlocks(): number {
    let used = 0;
    for (const perLayer of this.tables.values()) {
      for (const table of perLayer.values()) {
        used += table.length * this.blocksPerChunk;
      }
    }
    return used;
  }

  /** Host-equivalent block count: ceil(tokens / blockSize) per
   *  (sequence, layer), from the largest allocate() request — what
   *  BlockPagedKVCache.usedBlocks() would report for the same
   *  workload. Use this for admission decisions that assume host
   *  accounting; usedBlocks() remains the capacity truth. */
  logicalBlocks(): number {
    let logical = 0;
    for (const perLayer of this.logicalTokens.values()) {
      for (const numTokens of perLayer.values()) {
        logical += Math.ceil(numTokens / this.blockSize);
      }
    }
    return logical;
  }

  totalBlocks(): number {
    return this.total;
  }
}
