/**
 * vSSD — the virtual SSD pool for softwell-machine.
 *
 * The virtual computer's storage: ONE object store. Callers above the
 * boundary see only object keys and byte ranges. Chunking, placement,
 * replication, and tiering are internal — a workload never knows which
 * donor holds a chunk or which tier served it.
 *
 * Layout:
 *  - `VirtualSsd` is the coordinator (orchestrator side, implements
 *    `IVirtualSsd`). It discovers donors through the injected
 *    `NodeProvider`, splits objects into chunks, places replicas
 *    round-robin over distinct donors, and talks to donors with plain
 *    `fetch` against their `/hw/ssd/*` routes.
 *  - `SsdNode` is the donor adapter (worker side). It stores one file
 *    per chunk under its directory (or in memory when `memoryOnly`) and
 *    exposes its chunk store as `routes()`.
 *
 * Read tiers, hottest first:
 *  1. Hot: an in-coordinator LRU of recently fetched chunks.
 *  2. vRAM: when a `ram` pool is injected, hot chunks are mirrored into
 *     vRAM (one small allocation per chunk). The vRAM mirror has its own
 *     LRU and outlives hot eviction, so an evicted chunk can still be
 *     served from vRAM via `ram.read` instead of going to disk. Each
 *     cache entry tracks the chunk's vaddr.
 *  3. Node disk: the donor's local chunk store, first-listed holder
 *     first, then replicas.
 *
 * Node built-ins only. Erasable TypeScript only.
 */
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { randomUUID } from "node:crypto";
import { dirname, join } from "node:path";
import { VSSD_CHUNK_SIZE_BYTES } from "../contracts.ts";
import type {
  IVirtualRam,
  IVirtualSsd,
  NodeProvider,
  RouteHandler,
  SsdObjectMeta,
  SsdStats,
  VAddr,
  VramAllocation,
  WorkerInfo,
} from "../contracts.ts";

// ---------------------------------------------------------------------------
// Manifest + cache shapes (internal; the manifest is what gets persisted)
// ---------------------------------------------------------------------------

/** One chunk's placement: which donors hold a copy. */
export interface SsdChunkRef {
  chunkId: string;
  nodes: string[];
}

/** One object's manifest entry (persisted as JSON when manifestPath set). */
export interface SsdManifestEntry {
  key: string;
  sizeBytes: number;
  chunks: SsdChunkRef[];
  replicas: number;
  createdAt: number;
}

/** Tier counters for reads. */
export interface SsdCacheStats {
  hotHits: number;
  vramHits: number;
  diskFetches: number;
}

export interface VirtualSsdOpts {
  nodes: NodeProvider;
  chunkSizeBytes?: number;
  defaultReplicas?: number;
  manifestPath?: string;
  ram?: IVirtualRam;
  /** Hot LRU capacity in chunks (default 64). */
  hotCacheChunks?: number;
  /** vRAM mirror LRU capacity in chunks (default 256). */
  vramCacheChunks?: number;
  /** Max chunk fetches/stores in flight (default 8). */
  chunkIoConcurrency?: number;
}

export interface SsdNodeOpts {
  /** Chunk directory. Default `./data/ssd-<random>`. Ignored when memoryOnly. */
  dir?: string;
  /** Keep chunks in a Map instead of files (for tests). */
  memoryOnly?: boolean;
}

interface HotEntry {
  data: Uint8Array;
  allocId?: string;
  vaddr?: VAddr;
}

interface VramEntry {
  allocId: string;
  vaddr: VAddr;
  sizeBytes: number;
}

const MANIFEST_VERSION = 1;

function toMeta(entry: SsdManifestEntry): SsdObjectMeta {
  return {
    key: entry.key,
    sizeBytes: entry.sizeBytes,
    chunks: entry.chunks.length,
    replicas: entry.replicas,
    createdAt: entry.createdAt,
  };
}

/** Thrown when a node answers 404 to a binary route: it predates binary support. */
class BinaryRouteMissingError extends Error {}

/** Thrown when the binary get-chunk route answers JSON `{ error }`: the chunk is absent on that node. */
class ChunkMissingError extends Error {}

/** Max chunk fetches/stores in flight (default; see VirtualSsdOpts.chunkIoConcurrency). */
const DEFAULT_CHUNK_IO_CONCURRENCY = 8;

/**
 * Run `fn` over `items` with at most `limit` in flight, preserving input
 * order in the returned array. Single-threaded index hand-out keeps the
 * assignment race-free.
 */
async function mapLimit<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from(
    { length: Math.max(1, Math.min(limit, items.length)) },
    async () => {
      while (next < items.length) {
        const i = next++;
        results[i] = await fn(items[i], i);
      }
    },
  );
  await Promise.all(workers);
  return results;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asManifestEntry(raw: unknown): SsdManifestEntry | null {
  if (!isRecord(raw)) return null;
  if (typeof raw["key"] !== "string") return null;
  if (typeof raw["sizeBytes"] !== "number") return null;
  if (typeof raw["replicas"] !== "number") return null;
  if (typeof raw["createdAt"] !== "number") return null;
  if (!Array.isArray(raw["chunks"])) return null;
  const chunks: SsdChunkRef[] = [];
  for (const c of raw["chunks"] as unknown[]) {
    if (!isRecord(c)) return null;
    if (typeof c["chunkId"] !== "string") return null;
    if (!Array.isArray(c["nodes"])) return null;
    if (!c["nodes"].every((n) => typeof n === "string")) return null;
    chunks.push({ chunkId: c["chunkId"], nodes: [...(c["nodes"] as string[])] });
  }
  return {
    key: raw["key"] as string,
    sizeBytes: raw["sizeBytes"] as number,
    chunks,
    replicas: raw["replicas"] as number,
    createdAt: raw["createdAt"] as number,
  };
}

// ---------------------------------------------------------------------------
// VirtualSsd — coordinator
// ---------------------------------------------------------------------------

export class VirtualSsd implements IVirtualSsd {
  private readonly nodes: NodeProvider;
  private readonly chunkSizeBytes: number;
  private readonly defaultReplicas: number;
  private readonly manifestPath?: string;
  private readonly ram?: IVirtualRam;
  private readonly hotCacheChunks: number;
  private readonly vramCacheChunks: number;
  private readonly chunkIoConcurrency: number;
  /** Per-node binary-route support. Absent = untested, try binary first.
   *  Set to false when a node 404s a binary route (pre-binary node). */
  private readonly binaryByNode = new Map<string, boolean>();

  private readonly manifest = new Map<string, SsdManifestEntry>();
  private readonly hotCache = new Map<string, HotEntry>();
  private readonly vramIndex = new Map<string, VramEntry>();
  private hotHits = 0;
  private vramHits = 0;
  private diskFetches = 0;
  /** Round-robin cursor over healthy donors. */
  private cursor = 0;

  constructor(opts: VirtualSsdOpts) {
    this.nodes = opts.nodes;
    this.chunkSizeBytes = Math.max(
      1,
      Math.floor(opts.chunkSizeBytes ?? VSSD_CHUNK_SIZE_BYTES),
    );
    this.defaultReplicas = Math.max(1, Math.floor(opts.defaultReplicas ?? 2));
    this.manifestPath = opts.manifestPath;
    this.ram = opts.ram;
    this.hotCacheChunks = Math.max(1, Math.floor(opts.hotCacheChunks ?? 64));
    this.vramCacheChunks = Math.max(1, Math.floor(opts.vramCacheChunks ?? 256));
    this.chunkIoConcurrency = Math.max(
      1,
      Math.floor(opts.chunkIoConcurrency ?? DEFAULT_CHUNK_IO_CONCURRENCY),
    );
    this.loadManifest();
  }

  // --- IVirtualSsd ---------------------------------------------------------

  async putObject(
    key: string,
    data: Uint8Array,
    opts?: { replicas?: number },
  ): Promise<SsdObjectMeta> {
    const pool = this.healthyNodes();
    if (pool.length === 0) {
      throw new Error("vSSD: no donor nodes available to store object");
    }
    const replicas = Math.max(
      1,
      Math.min(
        Math.floor(opts?.replicas ?? this.defaultReplicas),
        pool.length,
      ),
    );

    // Overwrite: drop the old object's chunks first so stale chunk ids
    // (chunk ids are key-derived) can never be read back.
    if (this.manifest.has(key)) {
      await this.deleteObject(key);
    }

    const numChunks = Math.ceil(data.byteLength / this.chunkSizeBytes);
    // Placement is pure (round-robin cursor) and computed up front, so the
    // stores below can run in parallel without racing the cursor.
    const placements: Array<{ chunkId: string; chunk: Uint8Array; holders: string[] }> = [];
    for (let i = 0; i < numChunks; i++) {
      const chunkId = `${key}:${i}`;
      const start = i * this.chunkSizeBytes;
      const chunk = data.subarray(
        start,
        Math.min(data.byteLength, start + this.chunkSizeBytes),
      );
      placements.push({ chunkId, chunk, holders: this.placeChunk(replicas) });
    }

    // Store all chunks in parallel (concurrency-limited); holders within
    // one chunk are tried in listed order.
    const stored = await mapLimit(
      placements,
      this.chunkIoConcurrency,
      async (p) => {
        const ok: string[] = [];
        for (const holderId of p.holders) {
          const node = this.findNode(holderId);
          if (!node) continue;
          if (await this.tryPostChunk(node, p.chunkId, p.chunk)) {
            ok.push(holderId);
          }
        }
        return ok;
      },
    );

    // Repair pass: any chunk that landed under-replicated (a donor failed
    // mid-put) gets substitute donors from the healthy pool.
    const refs: SsdChunkRef[] = [];
    for (let i = 0; i < placements.length; i++) {
      const p = placements[i];
      const ok = stored[i];
      if (ok.length < replicas) {
        for (const node of this.healthyNodes()) {
          if (ok.length >= replicas) break;
          if (ok.includes(node.workerId)) continue;
          if (await this.tryPostChunk(node, p.chunkId, p.chunk)) {
            ok.push(node.workerId);
          }
        }
      }
      if (ok.length === 0) {
        throw new Error(
          `vSSD: could not store chunk ${p.chunkId} on any donor node`,
        );
      }
      if (ok.length < replicas) {
        console.warn(
          `vSSD: chunk ${p.chunkId} stored with ${ok.length}/${replicas} replicas`,
        );
      }
      refs.push({ chunkId: p.chunkId, nodes: ok });
    }

    const entry: SsdManifestEntry = {
      key,
      sizeBytes: data.byteLength,
      chunks: refs,
      replicas,
      createdAt: Date.now(),
    };
    this.manifest.set(key, entry);
    this.persistManifest();
    return toMeta(entry);
  }

  async getObject(key: string): Promise<Uint8Array> {
    const entry = this.requireEntry(key);
    // Parallel chunk fetches (concurrency-limited), order preserved by
    // mapLimit — the assembled bytes are identical to a sequential read.
    const parts = await mapLimit(
      entry.chunks,
      this.chunkIoConcurrency,
      (ref) => this.fetchChunk(ref.chunkId, ref.nodes),
    );
    return Buffer.concat(parts);
  }

  async getRange(
    key: string,
    offset: number,
    length: number,
  ): Promise<Uint8Array> {
    const entry = this.requireEntry(key);
    if (offset < 0 || length < 0 || offset > entry.sizeBytes) {
      throw new Error(
        `vSSD: getRange out of bounds for ${key}: offset=${offset} length=${length} size=${entry.sizeBytes}`,
      );
    }
    const end = Math.min(entry.sizeBytes, offset + length);
    if (end <= offset) {
      return new Uint8Array(0);
    }
    // Fetch ONLY the overlapping chunks — the streaming-read path.
    // Parallel (concurrency-limited), order preserved by mapLimit.
    const first = Math.floor(offset / this.chunkSizeBytes);
    const last = Math.floor((end - 1) / this.chunkSizeBytes);
    const wanted: SsdChunkRef[] = [];
    for (let i = first; i <= last; i++) {
      const ref = entry.chunks[i];
      if (!ref) {
        throw new Error(`vSSD: missing chunk ${i} for ${key}`);
      }
      wanted.push(ref);
    }
    const parts = await mapLimit(
      wanted,
      this.chunkIoConcurrency,
      (ref) => this.fetchChunk(ref.chunkId, ref.nodes),
    );
    const joined = Buffer.concat(parts);
    const startInJoined = offset - first * this.chunkSizeBytes;
    return joined.subarray(startInJoined, startInJoined + (end - offset));
  }

  async list(prefix?: string): Promise<SsdObjectMeta[]> {
    const out: SsdObjectMeta[] = [];
    for (const entry of this.manifest.values()) {
      if (prefix !== undefined && !entry.key.startsWith(prefix)) continue;
      out.push(toMeta(entry));
    }
    out.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    return out;
  }

  async deleteObject(key: string): Promise<void> {
    const entry = this.manifest.get(key);
    if (!entry) return; // idempotent
    for (const ref of entry.chunks) {
      for (const nodeId of ref.nodes) {
        const node = this.findNode(nodeId);
        if (node) {
          await this.dropChunkOnNode(node, ref.chunkId);
        }
      }
      await this.evictCached(ref.chunkId);
    }
    this.manifest.delete(key);
    this.persistManifest();
  }

  async handleNodeLoss(
    nodeId: string,
  ): Promise<{ chunksRecovered: number; chunksLost: number }> {
    let chunksRecovered = 0;
    let chunksLost = 0;
    let changed = false;
    // Healthy donors that could take a replacement replica.
    const healthy = this.healthyNodes().filter((n) => n.workerId !== nodeId);
    for (const entry of this.manifest.values()) {
      for (const ref of entry.chunks) {
        if (!ref.nodes.includes(nodeId)) continue;
        const survivors = ref.nodes.filter((id) => id !== nodeId);
        if (survivors.length === 0) {
          // No surviving copy anywhere: the chunk is gone.
          chunksLost++;
          ref.nodes = [];
          changed = true;
          continue;
        }
        // Re-replicate from a surviving copy onto a healthy donor that
        // does not already hold this chunk.
        const candidates = healthy.filter((n) => !survivors.includes(n.workerId));
        let recovered = false;
        if (candidates.length > 0) {
          let data: Uint8Array | null = null;
          try {
            data = await this.fetchChunk(ref.chunkId, survivors);
          } catch {
            data = null; // survivors unreachable; chunk not lost, just unhealed
          }
          if (data !== null) {
            for (const target of candidates) {
              try {
                await this.postChunk(target, ref.chunkId, data);
                ref.nodes = [...survivors, target.workerId];
                chunksRecovered++;
                recovered = true;
                changed = true;
                break;
              } catch {
                // Try the next healthy donor.
              }
            }
          }
        }
        if (!recovered) {
          // Surviving copy exists but no donor could take a new replica
          // (e.g. every survivor already holds it): drop the dead holder
          // from the list, keep the survivors.
          ref.nodes = survivors;
          changed = true;
        }
      }
    }
    if (changed) this.persistManifest();
    return { chunksRecovered, chunksLost };
  }

  stats(): SsdStats {
    let totalBytes = 0;
    for (const entry of this.manifest.values()) {
      totalBytes += entry.sizeBytes;
    }
    return {
      objects: this.manifest.size,
      totalBytes,
      nodes: this.nodes.listNodes().length,
    };
  }

  /** Tier counters for reads (extra method, not part of IVirtualSsd). */
  cacheStats(): SsdCacheStats {
    return {
      hotHits: this.hotHits,
      vramHits: this.vramHits,
      diskFetches: this.diskFetches,
    };
  }

  // --- chunk placement / node calls ----------------------------------------

  /**
   * Pick `replicas` distinct donors for one chunk, round-robin. Pure:
   * no I/O, so putObject can place every chunk up front and then store
   * them all in parallel.
   */
  private placeChunk(replicas: number): string[] {
    const pool = this.healthyNodes();
    const holders: string[] = [];
    for (
      let attempt = 0;
      attempt < pool.length && holders.length < replicas;
      attempt++
    ) {
      const node = pool[(this.cursor + attempt) % pool.length];
      if (!node || holders.includes(node.workerId)) continue;
      holders.push(node.workerId);
    }
    this.cursor = (this.cursor + 1) % Math.max(1, pool.length);
    return holders;
  }

  /** Store one chunk on `replicas` distinct donors, round-robin. */
  private async storeChunk(
    chunkId: string,
    data: Uint8Array,
    replicas: number,
  ): Promise<string[]> {
    const holders = this.placeChunk(replicas);
    const ok: string[] = [];
    for (const holderId of holders) {
      const node = this.findNode(holderId);
      if (!node) continue;
      if (await this.tryPostChunk(node, chunkId, data)) ok.push(holderId);
    }
    if (ok.length === 0) {
      throw new Error(
        `vSSD: could not store chunk ${chunkId} on any donor node`,
      );
    }
    if (ok.length < replicas) {
      console.warn(
        `vSSD: chunk ${chunkId} stored with ${ok.length}/${replicas} replicas`,
      );
    }
    return ok;
  }

  /** postChunk that reports success instead of throwing (parallel put path). */
  private async tryPostChunk(
    node: WorkerInfo,
    chunkId: string,
    data: Uint8Array,
  ): Promise<boolean> {
    try {
      await this.postChunk(node, chunkId, data);
      return true;
    } catch {
      return false;
    }
  }

  private nodeBaseUrl(node: WorkerInfo): string {
    return `http://${node.host}:${node.hwPort ?? node.port}`;
  }

  private async callNode(
    node: WorkerInfo,
    path: string,
    body: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const url = `${this.nodeBaseUrl(node)}${path}`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      throw new Error(
        `vSSD: cannot reach node ${node.workerId} at ${url}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    if (!res.ok) {
      throw new Error(
        `vSSD: node ${node.workerId} ${path} returned HTTP ${res.status}`,
      );
    }
    const parsed: unknown = await res.json();
    if (!isRecord(parsed)) {
      throw new Error(
        `vSSD: node ${node.workerId} ${path} returned a malformed response`,
      );
    }
    return parsed;
  }

  /**
   * Store one chunk, binary route first with JSON fallback. A node that
   * 404s the binary route is remembered as JSON-only; any other binary
   * failure falls back to JSON for this call only (the JSON result is
   * authoritative — a failed binary attempt never loses a chunk, since
   * the JSON retry stores the same chunk id).
   */
  private async postChunk(
    node: WorkerInfo,
    chunkId: string,
    data: Uint8Array,
  ): Promise<void> {
    if (this.binaryByNode.get(node.workerId) !== false) {
      try {
        await this.postChunkBin(node, chunkId, data);
        return;
      } catch (err) {
        if (err instanceof BinaryRouteMissingError) {
          this.binaryByNode.set(node.workerId, false);
        }
        // Otherwise fall through to the JSON route for this call.
      }
    }
    const res = await this.callNode(node, "/hw/ssd/put-chunk", {
      chunkId,
      dataB64: Buffer.from(data).toString("base64"),
    });
    if (res["ok"] !== true) {
      throw new Error(
        `vSSD: node ${node.workerId} rejected chunk ${chunkId}`,
      );
    }
  }

  /**
   * Binary put: POST /hw/ssd/put-chunk-bin with the raw chunk bytes.
   * The chunk id travels in the X-Chunk-Id header AND as a 4-byte
   * big-endian length prefix inside the body — the node server hands
   * handlers only `body.__raw` (no headers), so the in-body prefix is
   * what the donor actually reads; the header is set for observability
   * and for a future node server that passes headers through.
   */
  private async postChunkBin(
    node: WorkerInfo,
    chunkId: string,
    data: Uint8Array,
  ): Promise<void> {
    const url = `${this.nodeBaseUrl(node)}/hw/ssd/put-chunk-bin`;
    const idBytes = Buffer.from(chunkId, "utf8");
    const body = Buffer.alloc(4 + idBytes.byteLength + data.byteLength);
    body.writeUInt32BE(idBytes.byteLength, 0);
    idBytes.copy(body, 4);
    Buffer.from(data).copy(body, 4 + idBytes.byteLength);
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/octet-stream",
          "x-chunk-id": chunkId,
        },
        body,
      });
    } catch (err) {
      throw new Error(
        `vSSD: cannot reach node ${node.workerId} at ${url}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    if (res.status === 404) {
      throw new BinaryRouteMissingError(
        `vSSD: node ${node.workerId} has no binary put-chunk route`,
      );
    }
    if (!res.ok) {
      throw new Error(
        `vSSD: node ${node.workerId} put-chunk-bin returned HTTP ${res.status}`,
      );
    }
    const parsed: unknown = await res.json();
    if (!isRecord(parsed) || parsed["ok"] !== true) {
      throw new Error(
        `vSSD: node ${node.workerId} rejected binary chunk ${chunkId}`,
      );
    }
  }

  /**
   * Fetch one chunk, binary route first with JSON fallback. A JSON
   * `{ error }` answer from the binary route means the chunk is absent
   * on this node — that is thrown as ChunkMissingError so the caller
   * tries the next replica instead of pointlessly retrying JSON here.
   */
  private async getChunkFromNode(
    node: WorkerInfo,
    chunkId: string,
  ): Promise<Uint8Array> {
    if (this.binaryByNode.get(node.workerId) !== false) {
      try {
        return await this.getChunkFromNodeBin(node, chunkId);
      } catch (err) {
        if (err instanceof BinaryRouteMissingError) {
          this.binaryByNode.set(node.workerId, false);
        } else if (err instanceof ChunkMissingError) {
          throw err;
        }
        // Other binary failures: fall through to JSON for this call.
      }
    }
    return this.getChunkFromNodeJson(node, chunkId);
  }

  /** Binary get: POST /hw/ssd/get-chunk-bin, raw chunk id as the body. */
  private async getChunkFromNodeBin(
    node: WorkerInfo,
    chunkId: string,
  ): Promise<Uint8Array> {
    const url = `${this.nodeBaseUrl(node)}/hw/ssd/get-chunk-bin`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: Buffer.from(chunkId, "utf8"),
      });
    } catch (err) {
      throw new Error(
        `vSSD: cannot reach node ${node.workerId} at ${url}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    if (res.status === 404) {
      throw new BinaryRouteMissingError(
        `vSSD: node ${node.workerId} has no binary get-chunk route`,
      );
    }
    if (!res.ok) {
      throw new Error(
        `vSSD: node ${node.workerId} get-chunk-bin returned HTTP ${res.status}`,
      );
    }
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.startsWith("application/octet-stream")) {
      return new Uint8Array(await res.arrayBuffer());
    }
    // JSON answer: either `{ error }` (chunk missing on this node) or a
    // malformed response. Handlers cannot set a 404 status, so a missing
    // chunk arrives as 200 JSON — see SsdNode.routes().
    const parsed: unknown = await res.json();
    if (isRecord(parsed) && typeof parsed["error"] === "string") {
      throw new ChunkMissingError(
        `vSSD: chunk ${chunkId} not found on node ${node.workerId}: ${parsed["error"]}`,
      );
    }
    throw new Error(
      `vSSD: node ${node.workerId} returned a malformed get-chunk-bin response for ${chunkId}`,
    );
  }

  private async getChunkFromNodeJson(
    node: WorkerInfo,
    chunkId: string,
  ): Promise<Uint8Array> {
    const res = await this.callNode(node, "/hw/ssd/get-chunk", { chunkId });
    if (typeof res["error"] === "string") {
      throw new Error(
        `vSSD: chunk ${chunkId} not found on node ${node.workerId}: ${res["error"]}`,
      );
    }
    const dataB64 = res["dataB64"];
    if (typeof dataB64 !== "string") {
      throw new Error(
        `vSSD: node ${node.workerId} returned a malformed get-chunk response for ${chunkId}`,
      );
    }
    return Buffer.from(dataB64, "base64");
  }

  private async dropChunkOnNode(
    node: WorkerInfo,
    chunkId: string,
  ): Promise<void> {
    try {
      await this.callNode(node, "/hw/ssd/drop-chunk", { chunkId });
    } catch {
      // Best effort: a dead donor's chunks are gone with it anyway.
    }
  }

  // --- tiered chunk fetch ----------------------------------------------------

  /**
   * Fetch one chunk through the tiers: hot LRU -> vRAM mirror -> donor
   * disk (first-listed holder first, then replicas). A disk fetch
   * repopulates the hotter tiers.
   */
  private async fetchChunk(
    chunkId: string,
    nodeIds: string[],
  ): Promise<Uint8Array> {
    // Tier 1: hot in-coordinator LRU.
    const hot = this.hotCache.get(chunkId);
    if (hot) {
      this.hotCache.delete(chunkId);
      this.hotCache.set(chunkId, hot);
      this.hotHits++;
      return hot.data;
    }

    // Tier 2: vRAM mirror (reads check vRAM before going to disk).
    if (this.ram) {
      const vramEntry = this.vramIndex.get(chunkId);
      if (vramEntry) {
        try {
          const data = await this.ram.read(
            vramEntry.vaddr,
            vramEntry.sizeBytes,
          );
          this.vramIndex.delete(chunkId);
          this.vramIndex.set(chunkId, vramEntry);
          this.vramHits++;
          this.storeHot(chunkId, data, vramEntry.allocId, vramEntry.vaddr);
          return data;
        } catch {
          // Stale mirror (freed underneath us): drop it, fall to disk.
          this.vramIndex.delete(chunkId);
        }
      }
    }

    // Tier 3: donor node disk, trying holders in listed order.
    let lastError: Error | null = null;
    for (const nodeId of nodeIds) {
      const node = this.findNode(nodeId);
      if (!node) continue;
      try {
        const data = await this.getChunkFromNode(node, chunkId);
        this.diskFetches++;
        const mirrored = await this.mirrorToVram(chunkId, data);
        this.storeHot(chunkId, data, mirrored?.allocId, mirrored?.vaddr);
        return data;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
      }
    }
    throw new Error(
      `vSSD: chunk ${chunkId} unavailable on all replicas (${nodeIds.join(", ") || "none"}): ${lastError?.message ?? "no holders"}`,
    );
  }

  private storeHot(
    chunkId: string,
    data: Uint8Array,
    allocId?: string,
    vaddr?: VAddr,
  ): void {
    this.hotCache.delete(chunkId);
    this.hotCache.set(chunkId, { data, allocId, vaddr });
    while (this.hotCache.size > this.hotCacheChunks) {
      const oldest = this.hotCache.keys().next().value;
      if (oldest === undefined) break;
      // Hot eviction drops only the in-coordinator copy; the vRAM mirror
      // survives under its own LRU (see vramIndex) so the chunk can still
      // be served from tier 2.
      this.hotCache.delete(oldest);
    }
  }

  /**
   * Mirror a hot chunk into vRAM: one small allocation per chunk. On
   * allocation failure the coldest mirror is freed (ram.free) and the
   * allocation retried. Returns the allocation, or null when vRAM is
   * unavailable. Best effort — a failed mirror never fails the read.
   */
  private async mirrorToVram(
    chunkId: string,
    data: Uint8Array,
  ): Promise<{ allocId: string; vaddr: VAddr } | null> {
    const ram = this.ram;
    if (!ram) return null;
    const existing = this.vramIndex.get(chunkId);
    if (existing) {
      this.vramIndex.delete(chunkId);
      this.vramIndex.set(chunkId, existing);
      return { allocId: existing.allocId, vaddr: existing.vaddr };
    }
    for (let attempt = 0; attempt < 8; attempt++) {
      let alloc: VramAllocation | null = null;
      try {
        alloc = await ram.alloc(data.byteLength);
        await ram.write(alloc.vaddr, data);
        this.vramIndex.set(chunkId, {
          allocId: alloc.id,
          vaddr: alloc.vaddr,
          sizeBytes: data.byteLength,
        });
        await this.evictVramOverflow();
        return { allocId: alloc.id, vaddr: alloc.vaddr };
      } catch {
        if (alloc) {
          try {
            await ram.free(alloc.id);
          } catch {
            // Best effort.
          }
        }
        if (!(await this.evictOldestVram())) return null;
      }
    }
    return null;
  }

  private async evictVramOverflow(): Promise<void> {
    while (this.vramIndex.size > this.vramCacheChunks) {
      if (!(await this.evictOldestVram())) break;
    }
  }

  /** Evict the coldest vRAM mirror, freeing its allocation. */
  private async evictOldestVram(): Promise<boolean> {
    const oldest = this.vramIndex.keys().next().value;
    if (oldest === undefined) return false;
    const entry = this.vramIndex.get(oldest);
    this.vramIndex.delete(oldest);
    if (entry && this.ram) {
      try {
        await this.ram.free(entry.allocId);
      } catch {
        // Best effort.
      }
    }
    return true;
  }

  /** Drop a chunk from both cache tiers (used on delete). */
  private async evictCached(chunkId: string): Promise<void> {
    this.hotCache.delete(chunkId);
    const entry = this.vramIndex.get(chunkId);
    if (entry) {
      this.vramIndex.delete(chunkId);
      if (this.ram) {
        try {
          await this.ram.free(entry.allocId);
        } catch {
          // Best effort.
        }
      }
    }
  }

  // --- helpers ---------------------------------------------------------------

  private requireEntry(key: string): SsdManifestEntry {
    const entry = this.manifest.get(key);
    if (!entry) {
      throw new Error(`vSSD: object not found: ${key}`);
    }
    return entry;
  }

  private findNode(nodeId: string): WorkerInfo | undefined {
    return this.nodes.listNodes().find((n) => n.workerId === nodeId);
  }

  /** Donors usable for placement: healthy ones, or all listed if none report healthy. */
  private healthyNodes(): WorkerInfo[] {
    const all = this.nodes.listNodes();
    const healthy = all.filter((n) => n.state === "healthy");
    return healthy.length > 0 ? healthy : all;
  }

  // --- manifest persistence (best effort) ------------------------------------

  private persistManifest(): void {
    if (!this.manifestPath) return;
    try {
      mkdirSync(dirname(this.manifestPath), { recursive: true });
      const payload = {
        version: MANIFEST_VERSION,
        objects: [...this.manifest.values()],
      };
      writeFileSync(this.manifestPath, JSON.stringify(payload, null, 2), "utf8");
    } catch {
      // Best effort: the in-memory manifest stays authoritative.
    }
  }

  private loadManifest(): void {
    if (!this.manifestPath || !existsSync(this.manifestPath)) return;
    try {
      const parsed: unknown = JSON.parse(
        readFileSync(this.manifestPath, "utf8"),
      );
      if (!isRecord(parsed) || !Array.isArray(parsed["objects"])) return;
      for (const raw of parsed["objects"] as unknown[]) {
        const entry = asManifestEntry(raw);
        if (entry) this.manifest.set(entry.key, entry);
      }
    } catch (err) {
      console.warn(
        `vSSD: could not load manifest from ${this.manifestPath}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// SsdNode — donor adapter
// ---------------------------------------------------------------------------

export class SsdNode {
  private readonly dir: string | null;
  private readonly memory: Map<string, Uint8Array> | null;
  private chunkCount = 0;
  private bytesUsed = 0;

  constructor(opts: SsdNodeOpts = {}) {
    if (opts.memoryOnly) {
      this.memory = new Map();
      this.dir = null;
    } else {
      this.dir =
        opts.dir ?? join(process.cwd(), "data", `ssd-${randomUUID().slice(0, 8)}`);
      this.memory = null;
      mkdirSync(this.dir, { recursive: true });
      // Pick up chunks left on disk from a previous run.
      for (const name of readdirSync(this.dir)) {
        if (!name.endsWith(".chunk")) continue;
        try {
          const st = statSync(join(this.dir, name));
          this.chunkCount++;
          this.bytesUsed += st.size;
        } catch {
          // Ignore unreadable entries.
        }
      }
    }
  }

  private chunkPath(chunkId: string): string {
    // chunkIds are key-derived ("<key>:<index>"); encode so any key is a
    // safe single filename.
    return join(this.dir as string, `${encodeURIComponent(chunkId)}.chunk`);
  }

  private putChunk(chunkId: string, data: Uint8Array): void {
    if (this.memory) {
      const prev = this.memory.get(chunkId);
      if (prev) this.bytesUsed -= prev.byteLength;
      else this.chunkCount++;
      this.memory.set(chunkId, Buffer.from(data));
      this.bytesUsed += data.byteLength;
      return;
    }
    const path = this.chunkPath(chunkId);
    const existed = existsSync(path);
    const oldSize = existed ? statSync(path).size : 0;
    writeFileSync(path, data);
    if (!existed) this.chunkCount++;
    this.bytesUsed += data.byteLength - oldSize;
  }

  private getChunk(chunkId: string): Uint8Array | null {
    if (this.memory) {
      return this.memory.get(chunkId) ?? null;
    }
    const path = this.chunkPath(chunkId);
    if (!existsSync(path)) return null;
    return readFileSync(path);
  }

  private dropChunk(chunkId: string): void {
    if (this.memory) {
      const prev = this.memory.get(chunkId);
      if (prev) {
        this.memory.delete(chunkId);
        this.chunkCount--;
        this.bytesUsed -= prev.byteLength;
      }
      return;
    }
    const path = this.chunkPath(chunkId);
    if (!existsSync(path)) return;
    const size = statSync(path).size;
    unlinkSync(path);
    this.chunkCount--;
    this.bytesUsed -= size;
  }

  /**
   * Route table for the worker's hw node server. Keys are plain paths —
   * the node server dispatches by exact path (see src/hw/node-server.ts).
   * Methods: POST put/get/drop-chunk, GET stats.
   *
   * Binary routes (throughput path, no base64):
   *  - POST /hw/ssd/put-chunk-bin: raw chunk bytes as the body. The
   *    chunk id is a 4-byte big-endian length prefix + UTF-8 id at the
   *    head of the body. It is NOT read from the X-Chunk-Id header:
   *    the node server hands handlers only `body.__raw` and never
   *    passes headers, so the in-body prefix is the only channel the
   *    handler can see. The coordinator still sets X-Chunk-Id for
   *    observability. Returns JSON { ok: true }.
   *  - POST /hw/ssd/get-chunk-bin: the chunk id as raw UTF-8 bytes in
   *    the body; returns the raw chunk bytes. A handler cannot set an
   *    HTTP status, so a missing chunk comes back as 200 JSON
   *    { error } instead of a 404 — the coordinator treats that
   *    payload as "absent on this node" and tries the next replica.
   * The JSON routes below stay for back-compat (and for nodes whose
   * coordinator predates binary support).
   */
  routes(): Record<string, RouteHandler> {
    return {
      "/hw/ssd/put-chunk-bin": async (body) => {
        const raw = body["__raw"];
        if (!(raw instanceof Uint8Array) || raw.byteLength < 4) {
          throw new Error(
            "put-chunk-bin requires a binary body: 4-byte id length + chunk id + chunk bytes",
          );
        }
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        const idLen = view.getUint32(0, false);
        if (idLen > raw.byteLength - 4) {
          throw new Error("put-chunk-bin body has an invalid chunk id length");
        }
        const chunkId = Buffer.from(raw.subarray(4, 4 + idLen)).toString("utf8");
        if (chunkId.length === 0) {
          throw new Error("put-chunk-bin body has an empty chunk id");
        }
        this.putChunk(chunkId, raw.subarray(4 + idLen));
        return { ok: true };
      },
      "/hw/ssd/get-chunk-bin": async (body) => {
        const raw = body["__raw"];
        if (!(raw instanceof Uint8Array)) {
          throw new Error("get-chunk-bin requires a binary body (chunk id as UTF-8)");
        }
        const chunkId = Buffer.from(raw).toString("utf8");
        const data = this.getChunk(chunkId);
        if (!data) {
          return { error: `chunk not found: ${chunkId}` };
        }
        return Buffer.from(data);
      },
      "/hw/ssd/put-chunk": async (body) => {
        const chunkId = body["chunkId"];
        const dataB64 = body["dataB64"];
        if (typeof chunkId !== "string" || typeof dataB64 !== "string") {
          throw new Error("put-chunk requires { chunkId, dataB64 }");
        }
        this.putChunk(chunkId, Buffer.from(dataB64, "base64"));
        return { ok: true };
      },
      "/hw/ssd/get-chunk": async (body) => {
        const chunkId = body["chunkId"];
        if (typeof chunkId !== "string") {
          throw new Error("get-chunk requires { chunkId }");
        }
        const data = this.getChunk(chunkId);
        if (!data) {
          return { error: `chunk not found: ${chunkId}` };
        }
        return { dataB64: Buffer.from(data).toString("base64") };
      },
      "/hw/ssd/drop-chunk": async (body) => {
        const chunkId = body["chunkId"];
        if (typeof chunkId !== "string") {
          throw new Error("drop-chunk requires { chunkId }");
        }
        this.dropChunk(chunkId);
        return { ok: true };
      },
      "/hw/ssd/stats": async () => {
        return { chunks: this.chunkCount, bytesUsed: this.bytesUsed };
      },
    };
  }
}
