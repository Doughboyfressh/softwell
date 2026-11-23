/**
 * softwell-machine — scheduler: vRAM-backed block-paged KV cache (prototype).
 *
 * VERDICT: NO-GO for decode in this phase (prototype only — do not wire
 * into the worker). CORRECTED 2026-11-22: the previous header's numbers
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
 *     a projection — the engine's forward() is synchronous and cannot
 *     drive the async cache, so this was not run end-to-end):
 *       ctx 128:  +0.4% (run 4) to +3.5% (run 2) — negligible.
 *       ctx 1024: +75% in run 4 (0.42 -> 0.24 tok/s) — material.
 *     Absolute tok/s in the bench (0.9-1.9 tok/s at ctx 128) is far
 *     below the coordinator's live-server proof (5.10 tok/s): the
 *     bench shares a 1.5-CPU cell with other workstreams running
 *     heavy benches concurrently, so absolute times are load-dominated
 *     and ratios/projections within a run are the reliable signal.
 * Why still NO-GO:
 *   1. The sync-interface problem below is decisive on its own: the
 *      engine cannot drive this cache without an async forward() or
 *      a synchronous blocking transport. No performance number fixes
 *      that in this phase.
 *   2. Long-context decode pays ~+75% (up to ~2x) for vRAM KV — too
 *      much to accept by default at ctx 1024. Short contexts are
 *      nearly free, so performance alone would be GO-WITH-CONDITIONS
 *      (short contexts, or prefill-only), not an outright rejection.
 * What would have to change for this to become viable:
 *   1. The async-interface work (engine forward() async, or a
 *      worker-thread + Atomics.wait bridge) — see below. The bridge
 *      alone only fixes the interface, not the cost.
 *   2. A sub-millisecond small-read path in vRAM (sub-page transfers
 *      or a KV side-channel that does not move whole 4 MiB pages), or
 *      attention that does not re-read the full prefix every token
 *      (e.g. a vGPU attention kernel that reads KV in-place on the
 *      donor, so KV never crosses HTTP per token).
 *   3. For long contexts, the ~6-9x read ratio has to come down or be
 *      accepted as a capacity-vs-speed trade.
 * The one viable slice is PREFILL-ONLY staging: prefill appends are
 * already batched (one flush per full chunk, ~19-103 ms per 4 MiB
 * page, amortized ~0.02-0.10 ms/token) and prefill reads the prefix
 * once, not once per token. Even that needs the async-interface work
 * below, so it is a follow-up, not this phase.
 *
 * THE SYNC-INTERFACE PROBLEM (read this before integrating):
 * PagedKVCache (src/contracts.ts) is synchronous: append()/read()
 * return values, and TransformerStageExecutor.forward() — which is
 * itself synchronous — calls them inline, once per row per layer.
 * Any remote vRAM access is inherently async (HTTP to donors). There
 * is no way to satisfy the synchronous PagedKVCache interface with a
 * remote backing store without either
 *   (a) making the cache interface and forward() async (a cross-cutting
 *       engine change owned by the coordinator), or
 *   (b) a synchronous blocking transport (e.g. a worker thread that
 *       performs the async vRAM calls while the engine thread blocks
 *       on Atomics.wait over a SharedArrayBuffer — sketched, not built).
 * This file therefore implements the ASYNC twin of the interface: the
 * same method names and semantics, but every method returns a Promise.
 * It deliberately does NOT declare `implements PagedKVCache`, because
 * claiming that would be a lie the typechecker would catch. The class
 * is a faithful prototype of the async design and its measured costs;
 * the sync bridge is out of scope for this workstream.
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
 *     tolerance for in-flight data.
 *
 * Reads — coalesced per chunk, staging-aware:
 *   - Requested positions are grouped by chunk; consecutive positions
 *     in one chunk become one byte range and one vRAM read. Reads of
 *     staged chunks are served from the host staging buffer (no vRAM
 *     traffic). Reads of flushed chunks fetch from vRAM.
 *   - A read of a position in a chunk no append ever touched throws.
 *     A read of an unwritten position inside a touched chunk returns
 *     zeros (the chunk's pages are zero-filled on alloc and a flush
 *     writes the zero tail with them). This is chunk-granular, where
 *     BlockPagedKVCache is block-granular (it throws when the *block*
 *     was never touched): the two agree whenever reads stay within
 *     appended prefixes, which is all the engine ever does.
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
 * are not concurrent in practice. This class does not serialize
 * concurrent appends to the same chunk; do not drive it from two
 * forwards at once.
 */

import { VRAM_PAGE_SIZE_BYTES } from "../contracts.ts";
import type { IVirtualRam } from "../contracts.ts";

// ---------------------------------------------------------------------------
// VramAccess — the minimal remote-vRAM surface this cache needs.
// ---------------------------------------------------------------------------

/**
 * Minimal remote-vRAM access. Deliberately small and independent of
 * workstream 1's weight loader: the coordinator wires this to the real
 * remote client (HTTP binary weight/KV endpoint on the orchestrator)
 * during integration. All methods are async: vRAM lives across the
 * network from the worker.
 */
export interface VramAccess {
  /** Read `length` bytes starting at virtual address `vaddr`. */
  read(vaddr: number, length: number): Promise<Uint8Array>;
  /** Write `data` starting at virtual address `vaddr`. */
  write(vaddr: number, data: Uint8Array): Promise<void>;
  /**
   * Allocate `sizeBytes` (rounded up to whole vRAM pages by the
   * implementation) and return its id and base virtual address.
   * `opts.replicas` overrides the implementation default when given.
   */
  alloc(
    sizeBytes: number,
    opts?: { replicas?: number },
  ): Promise<{ id: string; vaddr: number }>;
  /** Release an allocation created by alloc(). */
  free(id: string): Promise<void>;
}

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
}

export class VramPagedKVCache {
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
  private readonly counters: VramKvStats;

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
    this.tables = new Map();
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

  /** Observable counters (see VramKvStats): read/write counts and bytes,
   *  so the coordinator can see the amplification this cache causes. */
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
   * order, where block b lives in chunk floor(b / blocksPerChunk).
   * The table is stable: a block id never changes while the sequence
   * lives. Rejects with VramKvError on out-of-memory (cap reached or
   * vRAM alloc failed) — no silent fallback to host memory.
   */
  async allocate(seqId: string, layer: number, numTokens: number): Promise<number[]> {
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
      table.push({
        allocId: alloc.id,
        vaddr: alloc.vaddr,
        staging,
        stagingF32: new Float32Array(staging.buffer),
        writtenUpTo: 0,
        flushed: false,
      });
    }
    // Return exactly the logical blocks the caller asked for (like the
    // host cache), even though whole chunks are reserved underneath:
    // the tail block slots of the last chunk exist but are not listed
    // until a later allocate() needs them. usedBlocks() still counts
    // the whole reserved chunks — that is what vRAM actually holds.
    return this.blockTable(table).slice(0, Math.ceil(numTokens / this.blockSize));
  }

  private blockTable(table: Chunk[]): number[] {
    const out: number[] = [];
    for (let c = 0; c < table.length; c++) {
      for (let b = 0; b < this.blocksPerChunk; b++) {
        out.push(c * this.blocksPerChunk + b);
      }
    }
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
   * failed chunk flush.
   */
  async append(
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
    const table = await this.allocate(seqId, layer, position + 1);
    void table; // allocate() grows the table in place; re-read it below.
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
   * Rejects with VramKvError when a needed vRAM page is unavailable
   * on all copies — committed KV is never silently zero-filled.
   */
  async read(
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
      if (chunk.writtenUpTo === 0 && !chunk.flushed) {
        throw new VramKvError(
          `VramPagedKVCache.read: no K/V written yet for seq ${seqId} layer ${layer} position ${startPos}`,
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
   * free fails; failures are reported (the allocation may already be
   * gone with a dead donor) rather than silently swallowed.
   */
  async free(seqId: string): Promise<void> {
    const perLayer = this.tables.get(seqId);
    if (perLayer === undefined) return;
    this.tables.delete(seqId);
    const failures: string[] = [];
    const frees: Array<Promise<void>> = [];
    for (const table of perLayer.values()) {
      for (const chunk of table) {
        frees.push(
          this.vram.free(chunk.allocId).catch((err: unknown) => {
            failures.push(
              `${chunk.allocId}: ${err instanceof Error ? err.message : String(err)}`,
            );
          }),
        );
      }
    }
    await Promise.all(frees);
    if (failures.length > 0) {
      throw new VramKvError(
        `VramPagedKVCache.free: ${failures.length} vRAM free(s) failed for seq ${seqId} ` +
          `(local state dropped anyway): ${failures.join("; ")}`,
      );
    }
  }

  /** Blocks currently allocated (whole chunks count fully, as with the host cache). */
  usedBlocks(): number {
    let used = 0;
    for (const perLayer of this.tables.values()) {
      for (const table of perLayer.values()) {
        used += table.length * this.blocksPerChunk;
      }
    }
    return used;
  }

  totalBlocks(): number {
    return this.total;
  }
}
