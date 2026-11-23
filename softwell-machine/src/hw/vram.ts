/**
 * vRAM — pooled virtual RAM for the softwell virtual machine.
 *
 * THE BOUNDARY: this module presents ONE global virtual address space.
 * Callers above the boundary see only virtual addresses (VAddr) and
 * allocations (VramAllocation). Donor nodes below the boundary hold
 * physical pages. The coordinator (VirtualRam) owns the global page table
 * and the virtual-address allocator; node ids appear only inside PageRef
 * records (placement metadata), never as something callers must handle.
 *
 * Two pieces live here:
 *   - RamNode: the donor-side adapter. An in-memory page store (with LRU
 *     eviction + spill to disk) behind the /hw/ram/* JSON routes.
 *   - VirtualRam: the orchestrator-side coordinator implementing
 *     IVirtualRam. It talks to donor nodes with plain fetch() against
 *     their hwPort; it never imports a sibling hw module.
 *
 * Page transfer uses binary routes (/hw/ram/fetch-bin, /hw/ram/store-bin)
 * over the node server's octet-stream path — no base64 overhead. The
 * JSON routes (/hw/ram/fetch, /hw/ram/store) stay for back-compat and
 * as the fallback when a donor does not serve the binary routes.
 * Transfers run with bounded concurrency (VRAM_IO_CONCURRENCY in
 * flight) and in batches of <= VRAM_BATCH_PAGES pages per request
 * (/hw/ram/fetch-many, /hw/ram/store-many-bin): the per-request
 * latency floor dominates 4 MiB transfers on loopback, so batching
 * is what makes aggregate bandwidth real. A failed batch falls back
 * to the per-page path with replica failover.
 *
 * Runtime: Node built-ins only, erasable TypeScript only.
 */

import { randomUUID } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { VRAM_PAGE_SIZE_BYTES } from "../contracts.ts";
import type {
  IVirtualRam,
  NodeProvider,
  PageRef,
  RouteHandler,
  VAddr,
  VramAllocation,
  VramStats,
  WorkerInfo,
} from "../contracts.ts";

// ---------------------------------------------------------------------------
// RamNode — donor-side page store
// ---------------------------------------------------------------------------

export interface RamNodeOpts {
  capacityBytes: number;
  /** Directory evicted pages spill to (one file per page). Without it,
   *  evicted pages are dropped. */
  spillDir?: string;
}

/** Shape of GET /hw/ram/stats. */
export interface RamNodeStats {
  pages: number; // pages held in memory
  bytesUsed: number;
  capacityBytes: number;
  spilled: number; // pages currently spilled to disk
  spillReads: number; // fetches served from disk (slower path)
}

function sanitizePageId(pageId: string): string {
  return pageId.replace(/[^A-Za-z0-9._-]/g, "_");
}

export class RamNode {
  readonly capacityBytes: number;
  readonly spillDir?: string;
  /** pageId -> page bytes. Insertion order IS the LRU order: the front
   *  is the least-recently-used page. Access re-inserts at the back. */
  private pages = new Map<string, Buffer>();
  private bytesUsed = 0;
  private spilled = new Set<string>();
  private spillReads = 0;

  constructor(opts: RamNodeOpts) {
    if (opts.capacityBytes <= 0) {
      throw new Error("RamNode: capacityBytes must be > 0");
    }
    this.capacityBytes = opts.capacityBytes;
    this.spillDir = opts.spillDir;
    if (opts.spillDir !== undefined) {
      mkdirSync(opts.spillDir, { recursive: true });
    }
  }

  private spillPath(pageId: string): string {
    // spillDir is set whenever this is called (guarded by callers).
    return join(this.spillDir as string, `${sanitizePageId(pageId)}.page`);
  }

  /** Store a page, evicting LRU pages while over capacity. */
  private storePage(pageId: string, data: Buffer): void {
    const existing = this.pages.get(pageId);
    if (existing !== undefined) {
      this.bytesUsed -= existing.length;
      this.pages.delete(pageId);
    }
    // A re-stored page is no longer spilled.
    if (this.spilled.delete(pageId) && this.spillDir !== undefined) {
      rmSync(this.spillPath(pageId), { force: true });
    }
    this.pages.set(pageId, data);
    this.bytesUsed += data.length;
    this.evictWhileOverCapacity();
  }

  private evictWhileOverCapacity(): void {
    while (this.bytesUsed > this.capacityBytes && this.pages.size > 0) {
      const oldest = this.pages.keys().next().value as string;
      const data = this.pages.get(oldest) as Buffer;
      this.pages.delete(oldest);
      this.bytesUsed -= data.length;
      if (this.spillDir !== undefined) {
        writeFileSync(this.spillPath(oldest), data);
        this.spilled.add(oldest);
      }
      // Without a spillDir the evicted page is simply dropped.
    }
  }

  /** Fetch a page: memory first, spill file second. Null when unknown. */
  private fetchPage(pageId: string): { data: Buffer; fromSpill: boolean } | null {
    const inMemory = this.pages.get(pageId);
    if (inMemory !== undefined) {
      // Touch: move to the back (most-recently-used).
      this.pages.delete(pageId);
      this.pages.set(pageId, inMemory);
      return { data: inMemory, fromSpill: false };
    }
    if (this.spillDir !== undefined && this.spilled.has(pageId)) {
      const path = this.spillPath(pageId);
      if (existsSync(path)) {
        this.spillReads += 1;
        return { data: readFileSync(path), fromSpill: true };
      }
      // Spill file vanished; forget the record.
      this.spilled.delete(pageId);
    }
    return null;
  }

  private dropPage(pageId: string): void {
    const data = this.pages.get(pageId);
    if (data !== undefined) {
      this.pages.delete(pageId);
      this.bytesUsed -= data.length;
    }
    if (this.spilled.delete(pageId) && this.spillDir !== undefined) {
      rmSync(this.spillPath(pageId), { force: true });
    }
  }

  private nodeStats(): RamNodeStats {
    return {
      pages: this.pages.size,
      bytesUsed: this.bytesUsed,
      capacityBytes: this.capacityBytes,
      spilled: this.spilled.size,
      spillReads: this.spillReads,
    };
  }

  /**
   * Route table for this donor's hw server. Keys are "METHOD /path";
   * the node server splits on the first space to dispatch.
   */
  routes(): Record<string, RouteHandler> {
    return {
      "POST /hw/ram/fetch": async (body) => {
        const pageId = body["pageId"];
        if (typeof pageId !== "string" || pageId === "") {
          throw new Error("RamNode: fetch requires a non-empty string pageId");
        }
        const hit = this.fetchPage(pageId);
        if (hit === null) {
          throw new Error(`RamNode: page not found: ${pageId}`);
        }
        return { dataB64: hit.data.toString("base64"), fromSpill: hit.fromSpill };
      },
      "POST /hw/ram/store": async (body) => {
        const pageId = body["pageId"];
        const dataB64 = body["dataB64"];
        if (typeof pageId !== "string" || pageId === "") {
          throw new Error("RamNode: store requires a non-empty string pageId");
        }
        if (typeof dataB64 !== "string") {
          throw new Error("RamNode: store requires a base64 string dataB64");
        }
        this.storePage(pageId, Buffer.from(dataB64, "base64"));
        return { ok: true };
      },
      /**
       * Binary page transport (no base64): the node server hands the
       * raw request bytes to the handler as body.__raw and sends a
       * Buffer result back raw with Content-Type: application/octet-stream.
       *
       * fetch-bin: body.__raw is the pageId as UTF-8; returns the page
       * bytes raw (exactly VRAM_PAGE_SIZE_BYTES).
       *
       * store-bin: body.__raw is [4-byte big-endian pageId length]
       * [pageId UTF-8] [page bytes]. The length prefix carries the
       * pageId because route handlers receive only the body, not the
       * request headers (node-server.ts is intentionally untouched).
       */
      "POST /hw/ram/fetch-bin": async (body) => {
        const raw = body["__raw"];
        if (!(raw instanceof Uint8Array)) {
          throw new Error(
            "RamNode: fetch-bin requires a binary body (the pageId as UTF-8)",
          );
        }
        const pageId = Buffer.from(raw).toString("utf8");
        if (pageId === "") {
          throw new Error("RamNode: fetch-bin requires a non-empty pageId");
        }
        const hit = this.fetchPage(pageId);
        if (hit === null) {
          throw new Error(`RamNode: page not found: ${pageId}`);
        }
        return hit.data;
      },
      /**
       * Batched fetch: JSON body { pageIds: string[] }; returns the
       * pages concatenated in order as raw bytes (exactly one page
       * per id). One request per batch instead of one per page: the
       * per-request latency floor dominates 4 MiB transfers on
       * loopback, so batching is what makes bandwidth real. Any
       * missing page fails the whole batch (500); the coordinator
       * then falls back to per-page fetches with replica failover.
       */
      "POST /hw/ram/fetch-many": async (body) => {
        const pageIds = body["pageIds"];
        if (
          !Array.isArray(pageIds) ||
          pageIds.length === 0 ||
          pageIds.some((id) => typeof id !== "string" || id === "")
        ) {
          throw new Error(
            "RamNode: fetch-many requires a non-empty string[] pageIds",
          );
        }
        const chunks: Buffer[] = [];
        for (const pageId of pageIds as string[]) {
          const hit = this.fetchPage(pageId);
          if (hit === null) {
            throw new Error(`RamNode: page not found: ${pageId}`);
          }
          if (hit.data.length !== VRAM_PAGE_SIZE_BYTES) {
            throw new Error(
              `RamNode: page ${pageId} has ${hit.data.length} bytes, expected one ${VRAM_PAGE_SIZE_BYTES}-byte page`,
            );
          }
          chunks.push(hit.data);
        }
        return Buffer.concat(chunks);
      },
      "POST /hw/ram/store-bin": async (body) => {
        const raw = body["__raw"];
        if (!(raw instanceof Uint8Array) || raw.length < 4) {
          throw new Error(
            "RamNode: store-bin requires a binary body ([u32 pageId length][pageId][page bytes])",
          );
        }
        const buf = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
        const pageIdLen = buf.readUInt32BE(0);
        if (pageIdLen === 0 || 4 + pageIdLen > buf.length) {
          throw new Error("RamNode: store-bin has a malformed pageId prefix");
        }
        const pageId = buf.subarray(4, 4 + pageIdLen).toString("utf8");
        const data = Buffer.from(buf.subarray(4 + pageIdLen));
        this.storePage(pageId, data);
        return { ok: true };
      },
      /**
       * Batched store: body.__raw is [u32 count] then per page
       * [u32 pageId length][pageId UTF-8][exactly one page of bytes].
       * The whole batch is parsed before anything is stored, so a
       * malformed body stores nothing. Returns { ok, stored }.
       */
      "POST /hw/ram/store-many-bin": async (body) => {
        const raw = body["__raw"];
        if (!(raw instanceof Uint8Array) || raw.length < 4) {
          throw new Error(
            "RamNode: store-many-bin requires a binary body ([u32 count][u32 pageId length][pageId][page bytes]...)",
          );
        }
        const buf = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
        const count = buf.readUInt32BE(0);
        if (count === 0) {
          throw new Error("RamNode: store-many-bin requires count > 0");
        }
        let offset = 4;
        const entries: Array<{ pageId: string; data: Buffer }> = [];
        for (let i = 0; i < count; i++) {
          if (offset + 4 > buf.length) {
            throw new Error("RamNode: store-many-bin truncated pageId length");
          }
          const pageIdLen = buf.readUInt32BE(offset);
          offset += 4;
          if (
            pageIdLen === 0 ||
            offset + pageIdLen + VRAM_PAGE_SIZE_BYTES > buf.length
          ) {
            throw new Error("RamNode: store-many-bin truncated page entry");
          }
          const pageId = buf.subarray(offset, offset + pageIdLen).toString("utf8");
          offset += pageIdLen;
          const data = Buffer.from(
            buf.subarray(offset, offset + VRAM_PAGE_SIZE_BYTES),
          );
          offset += VRAM_PAGE_SIZE_BYTES;
          entries.push({ pageId, data });
        }
        if (offset !== buf.length) {
          throw new Error("RamNode: store-many-bin has trailing bytes");
        }
        for (const { pageId, data } of entries) {
          this.storePage(pageId, data);
        }
        return { ok: true, stored: entries.length };
      },
      "POST /hw/ram/drop": async (body) => {
        const pageId = body["pageId"];
        if (typeof pageId !== "string" || pageId === "") {
          throw new Error("RamNode: drop requires a non-empty string pageId");
        }
        this.dropPage(pageId);
        return { ok: true };
      },
      "GET /hw/ram/stats": async () => {
        return this.nodeStats();
      },
    };
  }
}

// ---------------------------------------------------------------------------
// VirtualRam — coordinator: one global virtual address space
// ---------------------------------------------------------------------------

export interface VirtualRamOpts {
  nodes: NodeProvider;
  /** Copies per page when alloc() does not specify replicas. */
  defaultReplicas?: number;
}

interface PageEntry {
  allocId: string;
  version: number;
  /** copies[0] is the primary. */
  copies: PageRef[];
}

interface AllocRecord {
  id: string;
  vaddr: VAddr;
  sizeBytes: number;
  pageIds: string[];
  replicas: number;
}

interface FreeRange {
  vaddr: VAddr;
  sizeBytes: number;
}

const FETCH_TIMEOUT_MS = 30_000;

/** Max page transfers in flight for one read()/write()/alloc(). */
const VRAM_IO_CONCURRENCY = 8;

/** Max pages per batched transfer request (8 x 4 MiB = 32 MiB).
 *  Batching amortizes the per-request latency floor (~20-45 ms on
 *  loopback here), which otherwise dominates 4 MiB page transfers;
 *  batches stay small enough to avoid multi-hundred-MiB single
 *  responses and their GC pressure. */
const VRAM_BATCH_PAGES = 8;

/** Split items into chunks of at most `size` (order preserved). */
function batchesOf<T>(items: readonly T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

/**
 * Run `fn` over `items` with at most `limit` tasks in flight.
 * Results are collected by index, so callers can preserve order
 * (page order in a read buffer, page order in an allocation).
 */
async function runLimited<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from(
    { length: Math.min(limit, items.length) },
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

/** Error type for coordinator-side failures (fetch errors, missing pages). */
export class VramError extends Error {
  /** HTTP status of the failed donor call, when the failure was an HTTP response. */
  readonly status?: number;
  constructor(message: string, opts?: { cause?: unknown; status?: number }) {
    super(
      message,
      opts?.cause !== undefined ? { cause: opts.cause } : undefined,
    );
    this.name = "VramError";
    this.status = opts?.status;
  }
}

export class VirtualRam implements IVirtualRam {
  private readonly nodes: NodeProvider;
  private readonly defaultReplicas: number;
  /** Global page table: pageId -> version + copies (primary first). */
  private readonly pageTable = new Map<string, PageEntry>();
  private readonly allocs = new Map<string, AllocRecord>();
  /** Bump pointer for fresh virtual addresses. */
  private bumpVaddr = 0;
  /** Freed ranges available for reuse, kept sorted by vaddr. */
  private freeList: FreeRange[] = [];
  /** Round-robin cursor for copy placement. */
  private rr = 0;

  constructor(opts: VirtualRamOpts) {
    this.nodes = opts.nodes;
    this.defaultReplicas =
      opts.defaultReplicas !== undefined ? opts.defaultReplicas : 2;
    if (this.defaultReplicas < 1) {
      throw new Error("VirtualRam: defaultReplicas must be >= 1");
    }
  }

  // --- node discovery + transport ------------------------------------------

  /** Donors the coordinator may place pages on: healthy + serving hw. */
  private healthyNodes(): WorkerInfo[] {
    return this.nodes
      .listNodes()
      .filter((n) => n.state === "healthy" && typeof n.hwPort === "number");
  }

  private findNode(nodeId: string): WorkerInfo | undefined {
    return this.nodes.listNodes().find((n) => n.workerId === nodeId);
  }

  /** One JSON call to a donor's /hw/ram/* route. Fetch failures become
   *  VramErrors, never crashes. */
  private async callNode(
    node: WorkerInfo,
    path: string,
    body: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const url = `http://${node.host}:${node.hwPort}${path}`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
    } catch (err) {
      throw new VramError(
        `vRAM: ${path} to node ${node.workerId} (${url}) failed: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new VramError(
        `vRAM: ${path} to node ${node.workerId} returned HTTP ${res.status}: ${text.slice(0, 200)}`,
        { status: res.status },
      );
    }
    return (await res.json()) as Record<string, unknown>;
  }

  /** One binary call to a donor's /hw/ram/*-bin route. Returns the raw
   *  response bytes. Fetch failures become VramErrors, never crashes. */
  private async callNodeBinary(
    node: WorkerInfo,
    path: string,
    body: Buffer,
  ): Promise<Buffer> {
    const url = `http://${node.host}:${node.hwPort}${path}`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body,
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
    } catch (err) {
      throw new VramError(
        `vRAM: ${path} to node ${node.workerId} (${url}) failed: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new VramError(
        `vRAM: ${path} to node ${node.workerId} returned HTTP ${res.status}: ${text.slice(0, 200)}`,
        { status: res.status },
      );
    }
    return Buffer.from(await res.arrayBuffer());
  }

  /** Batched fetch: POST { pageIds } as JSON, get the pages
   *  concatenated in order as raw bytes. Throws VramError (with
   *  status when the donor answered HTTP) on any failure. */
  private async callNodeFetchMany(
    node: WorkerInfo,
    pageIds: string[],
  ): Promise<Buffer> {
    const url = `http://${node.host}:${node.hwPort}/hw/ram/fetch-many`;
    let res: Response;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ pageIds }),
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
    } catch (err) {
      throw new VramError(
        `vRAM: /hw/ram/fetch-many to node ${node.workerId} (${url}) failed: ${err instanceof Error ? err.message : String(err)}`,
        { cause: err },
      );
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new VramError(
        `vRAM: /hw/ram/fetch-many to node ${node.workerId} returned HTTP ${res.status}: ${text.slice(0, 200)}`,
        { status: res.status },
      );
    }
    return Buffer.from(await res.arrayBuffer());
  }

  /** Store several pages on one donor in one binary request.
   *  A 404 means the donor predates the batch route: fall back to
   *  per-page stores. Any other failure throws (callers fall back
   *  to the per-segment path). */
  private async storePagesToNode(
    nodeId: string,
    pages: Array<{ pageId: string; data: Buffer }>,
  ): Promise<void> {
    const node = this.findNode(nodeId);
    if (node === undefined || typeof node.hwPort !== "number") {
      throw new VramError(`vRAM: no reachable donor node ${nodeId}`);
    }
    try {
      const parts: Buffer[] = [Buffer.alloc(4)];
      (parts[0] as Buffer).writeUInt32BE(pages.length, 0);
      for (const { pageId, data } of pages) {
        const prefix = Buffer.alloc(4);
        prefix.writeUInt32BE(Buffer.byteLength(pageId), 0);
        parts.push(prefix, Buffer.from(pageId, "utf8"), data);
      }
      await this.callNodeBinary(
        node,
        "/hw/ram/store-many-bin",
        Buffer.concat(parts),
      );
      return;
    } catch (err) {
      if (!(err instanceof VramError) || err.status !== 404) throw err;
    }
    for (const { pageId, data } of pages) {
      await this.storePageToNode(nodeId, pageId, data);
    }
  }

  private async storePageToNode(
    nodeId: string,
    pageId: string,
    data: Buffer,
  ): Promise<void> {
    const node = this.findNode(nodeId);
    if (node === undefined || typeof node.hwPort !== "number") {
      throw new VramError(`vRAM: no reachable donor node ${nodeId}`);
    }
    // Binary first; a 404 means the donor predates the binary routes,
    // so fall back to the JSON route (base64) for that donor.
    try {
      const prefix = Buffer.alloc(4);
      prefix.writeUInt32BE(Buffer.byteLength(pageId), 0);
      await this.callNodeBinary(
        node,
        "/hw/ram/store-bin",
        Buffer.concat([prefix, Buffer.from(pageId, "utf8"), data]),
      );
      return;
    } catch (err) {
      if (!(err instanceof VramError) || err.status !== 404) throw err;
    }
    await this.callNode(node, "/hw/ram/store", {
      pageId,
      dataB64: data.toString("base64"),
    });
  }

  /** Fetch a page's bytes from its primary, falling back to replicas in
   *  order. Throws when no copy answers. */
  private async fetchPageData(entry: PageEntry, pageId: string): Promise<Buffer> {
    for (const copy of entry.copies) {
      // Never serve a stale copy: a replica whose version lags the page
      // table holds pre-write bytes. Serving it on failover would
      // silently return old data; skipping it fails loudly instead.
      // (Writes now store to every copy, so a listed copy should always
      // be current — this guard exists so any future path that leaves a
      // stale copy behind degrades to an error, not corruption.)
      if (copy.version !== entry.version) continue;
      const node = this.findNode(copy.nodeId);
      if (node === undefined || typeof node.hwPort !== "number") continue;
      // Binary first; a 404 on the binary route falls back to the JSON
      // route on the same node. Any other failure tries the next copy.
      let data: Buffer | null = null;
      try {
        data = await this.callNodeBinary(
          node,
          "/hw/ram/fetch-bin",
          Buffer.from(pageId, "utf8"),
        );
      } catch (err) {
        if (err instanceof VramError && err.status === 404) {
          try {
            const out = await this.callNode(node, "/hw/ram/fetch", { pageId });
            if (typeof out["dataB64"] !== "string") continue;
            data = Buffer.from(out["dataB64"] as string, "base64");
          } catch {
            continue;
          }
        } else {
          continue;
        }
      }
      if (data === null) continue;
      if (data.length === VRAM_PAGE_SIZE_BYTES) return data;
      // Tolerate a short/long page: normalize to exactly one page.
      const page = Buffer.alloc(VRAM_PAGE_SIZE_BYTES);
      data.copy(page, 0, 0, Math.min(data.length, VRAM_PAGE_SIZE_BYTES));
      return page;
    }
    throw new VramError(
      `vRAM: page ${pageId} unavailable on all ${entry.copies.length} copies`,
    );
  }

  /**
   * Fetch one full page per segment, batched per primary donor
   * (<= VRAM_BATCH_PAGES pages per request). A failed batch falls
   * back to per-page fetches with replica failover, so a missing
   * page or a dead primary never loses the read — it just costs
   * the per-page path for that batch.
   */
  private async fetchSegmentPages(
    segments: Array<{
      pageId: string;
      pageOffset: number;
      rangeStart: number;
      rangeEnd: number;
    }>,
  ): Promise<Map<string, Buffer>> {
    const fetched = new Map<string, Buffer>();
    const byNode = new Map<
      string,
      Array<{
        seg: (typeof segments)[number];
        entry: PageEntry;
      }>
    >();
    for (const seg of segments) {
      const entry = this.pageTable.get(seg.pageId);
      if (entry === undefined) {
        throw new VramError(`vRAM: page table has no entry for ${seg.pageId}`);
      }
      const primary = entry.copies[0];
      if (primary === undefined) {
        // Same message fetchPageData throws for a copy-less page, so
        // callers (and tests) see the same failure either way.
        throw new VramError(
          `vRAM: page ${seg.pageId} unavailable on all ${entry.copies.length} copies`,
        );
      }
      const list = byNode.get(primary.nodeId) ?? [];
      list.push({ seg, entry });
      byNode.set(primary.nodeId, list);
    }
    await runLimited(
      [...byNode.entries()],
      VRAM_IO_CONCURRENCY,
      async ([nodeId, items]) => {
        const node = this.findNode(nodeId);
        if (node !== undefined && typeof node.hwPort === "number") {
          for (const batch of batchesOf(items, VRAM_BATCH_PAGES)) {
            try {
              const raw = await this.callNodeFetchMany(
                node,
                batch.map((b) => b.seg.pageId),
              );
              if (raw.length !== batch.length * VRAM_PAGE_SIZE_BYTES) {
                throw new VramError(
                  `vRAM: fetch-many from node ${nodeId} returned ${raw.length} bytes for ${batch.length} pages`,
                );
              }
              batch.forEach((b, i) => {
                fetched.set(
                  b.seg.pageId,
                  raw.subarray(
                    i * VRAM_PAGE_SIZE_BYTES,
                    (i + 1) * VRAM_PAGE_SIZE_BYTES,
                  ),
                );
              });
              continue;
            } catch {
              // Fall through to per-page fetches for this batch.
            }
            for (const b of batch) {
              fetched.set(
                b.seg.pageId,
                await this.fetchPageData(b.entry, b.seg.pageId),
              );
            }
          }
        } else {
          for (const b of items) {
            fetched.set(
              b.seg.pageId,
              await this.fetchPageData(b.entry, b.seg.pageId),
            );
          }
        }
      },
    );
    return fetched;
  }

  // --- placement ------------------------------------------------------------

  /**
   * Candidate donors in placement order: nodeAffinity nodes first (in the
   * given order), then the remaining healthy donors starting at the
   * round-robin cursor so consecutive pages spread across donors.
   */
  private placementCandidates(affinity?: string[]): WorkerInfo[] {
    const healthy = this.healthyNodes();
    if (healthy.length === 0) return [];
    const byId = new Map(healthy.map((n) => [n.workerId, n]));
    const ordered: WorkerInfo[] = [];
    for (const id of affinity ?? []) {
      const node = byId.get(id);
      if (node !== undefined && !ordered.includes(node)) ordered.push(node);
    }
    const rest = healthy.filter((n) => !ordered.includes(n));
    const start = rest.length === 0 ? 0 : this.rr % rest.length;
    for (let i = 0; i < rest.length; i++) {
      ordered.push(rest[(start + i) % rest.length]);
    }
    return ordered;
  }

  // --- virtual-address allocator --------------------------------------------

  /** Carve `sizeBytes` (already page-rounded) from the free list or the
   *  bump pointer. */
  private carveAddress(sizeBytes: number): VAddr {
    for (let i = 0; i < this.freeList.length; i++) {
      const range = this.freeList[i];
      if (range.sizeBytes >= sizeBytes) {
        const vaddr = range.vaddr;
        if (range.sizeBytes === sizeBytes) {
          this.freeList.splice(i, 1);
        } else {
          this.freeList[i] = {
            vaddr: range.vaddr + sizeBytes,
            sizeBytes: range.sizeBytes - sizeBytes,
          };
        }
        return vaddr;
      }
    }
    const vaddr = this.bumpVaddr;
    this.bumpVaddr += sizeBytes;
    return vaddr;
  }

  private releaseAddress(vaddr: VAddr, sizeBytes: number): void {
    this.freeList.push({ vaddr, sizeBytes });
    this.freeList.sort((a, b) => a.vaddr - b.vaddr);
    const merged: FreeRange[] = [];
    for (const range of this.freeList) {
      const last = merged[merged.length - 1];
      if (
        last !== undefined &&
        last.vaddr + last.sizeBytes >= range.vaddr
      ) {
        last.sizeBytes = Math.max(
          last.sizeBytes,
          range.vaddr + range.sizeBytes - last.vaddr,
        );
      } else {
        merged.push({ ...range });
      }
    }
    this.freeList = merged;
  }

  private findAllocForRange(
    vaddr: VAddr,
    length: number,
  ): AllocRecord | undefined {
    for (const alloc of this.allocs.values()) {
      if (vaddr >= alloc.vaddr && vaddr + length <= alloc.vaddr + alloc.sizeBytes) {
        return alloc;
      }
    }
    return undefined;
  }

  /** Split [vaddr, vaddr+length) into per-page segments. */
  private pageSegments(
    alloc: AllocRecord,
    vaddr: VAddr,
    length: number,
  ): Array<{
    pageId: string;
    pageOffset: number;
    rangeStart: number;
    rangeEnd: number;
  }> {
    const segments = [];
    let cursor = vaddr;
    const end = vaddr + length;
    while (cursor < end) {
      const offset = cursor - alloc.vaddr;
      const pageIndex = Math.floor(offset / VRAM_PAGE_SIZE_BYTES);
      const pageOffset = offset % VRAM_PAGE_SIZE_BYTES;
      const segEnd = Math.min(end, alloc.vaddr + (pageIndex + 1) * VRAM_PAGE_SIZE_BYTES);
      segments.push({
        pageId: alloc.pageIds[pageIndex],
        pageOffset,
        rangeStart: cursor,
        rangeEnd: segEnd,
      });
      cursor = segEnd;
    }
    return segments;
  }

  // --- IVirtualRam ------------------------------------------------------------

  async alloc(
    sizeBytes: number,
    opts?: { replicas?: number; nodeAffinity?: string[] },
  ): Promise<VramAllocation> {
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      throw new VramError("vRAM: alloc sizeBytes must be > 0");
    }
    const replicas = opts?.replicas ?? this.defaultReplicas;
    if (replicas < 1) {
      throw new VramError("vRAM: replicas must be >= 1");
    }
    const numPages = Math.ceil(sizeBytes / VRAM_PAGE_SIZE_BYTES);
    const footprint = numPages * VRAM_PAGE_SIZE_BYTES;
    const candidates = this.placementCandidates(opts?.nodeAffinity);
    if (candidates.length === 0) {
      throw new VramError("vRAM: no healthy donor nodes with hwPort to place pages on");
    }

    const id = randomUUID();
    const vaddr = this.carveAddress(footprint);
    const zeroPage = Buffer.alloc(VRAM_PAGE_SIZE_BYTES);
    const pageIds: string[] = [];
    const allRefs: PageRef[] = [];
    // Pages placed so far (filled as tasks complete, in completion
    // order) — the rollback below drops exactly these, even when a
    // parallel task fails midway.
    const placedIds: string[] = [];
    // Fast path: batch the zero-page stores per donor (<=8 pages per
    // request) before the per-page loop. Pairs the batch confirmed are
    // counted as copies without another store call; anything the batch
    // missed is placed by the per-page loop below, which keeps the
    // exact "skip a dead donor, place fewer copies" semantics.
    const planned = Array.from({ length: numPages }, (_, i) => {
      const pageId = `${id}/p${i}`;
      const start = (this.rr + i) % candidates.length;
      const nodes = Array.from(
        { length: Math.min(replicas, candidates.length) },
        (_, j) => candidates[(start + j) % candidates.length].workerId,
      );
      return { pageId, nodes };
    });
    const storedPairs = new Set<string>();
    const zeroByNode = new Map<string, string[]>();
    for (const p of planned) {
      for (const nodeId of p.nodes) {
        const list = zeroByNode.get(nodeId) ?? [];
        list.push(p.pageId);
        zeroByNode.set(nodeId, list);
      }
    }
    await runLimited(
      [...zeroByNode.entries()],
      VRAM_IO_CONCURRENCY,
      async ([nodeId, pageIds]) => {
        for (const batch of batchesOf(pageIds, VRAM_BATCH_PAGES)) {
          try {
            await this.storePagesToNode(
              nodeId,
              batch.map((pageId) => ({ pageId, data: zeroPage })),
            );
            for (const pageId of batch) {
              storedPairs.add(`${pageId}@${nodeId}`);
            }
          } catch {
            // The per-page loop below retries these pages.
          }
        }
      },
    );
    try {
      // Place pages concurrently (bounded): consecutive pages stripe
      // across donors via the per-page round-robin start.
      const placed = await runLimited(
        Array.from({ length: numPages }, (_, i) => i),
        VRAM_IO_CONCURRENCY,
        async (i) => {
          const pageId = `${id}/p${i}`;
          // Round-robin start per page so copies spread across donors.
          const start = (this.rr + i) % candidates.length;
          const copies: PageRef[] = [];
          for (let j = 0; j < candidates.length && copies.length < replicas; j++) {
            const node = candidates[(start + j) % candidates.length];
            if (storedPairs.has(`${pageId}@${node.workerId}`)) {
              copies.push({
                pageId,
                nodeId: node.workerId,
                version: 0,
                state: "resident",
              });
              continue;
            }
            try {
              await this.storePageToNode(node.workerId, pageId, zeroPage);
              copies.push({
                pageId,
                nodeId: node.workerId,
                version: 0,
                state: "resident",
              });
            } catch {
              // Donor unreachable mid-alloc: skip it, place fewer copies.
            }
          }
          if (copies.length === 0) {
            throw new VramError(
              `vRAM: could not place page ${pageId} on any donor node`,
            );
          }
          this.pageTable.set(pageId, { allocId: id, version: 0, copies });
          placedIds.push(pageId);
          return { pageId, copies };
        },
      );
      // runLimited preserves index order, so pageIds stay page-ordered.
      for (const p of placed) {
        pageIds.push(p.pageId);
        allRefs.push(...p.copies);
      }
    } catch (err) {
      // Roll back: drop whatever was placed and release the address range.
      for (const pageId of placedIds) {
        const entry = this.pageTable.get(pageId);
        if (entry !== undefined) {
          for (const copy of entry.copies) {
            try {
              const node = this.findNode(copy.nodeId);
              if (node !== undefined) {
                await this.callNode(node, "/hw/ram/drop", { pageId });
              }
            } catch {
              // Best effort only.
            }
          }
          this.pageTable.delete(pageId);
        }
      }
      this.releaseAddress(vaddr, footprint);
      throw err;
    }
    this.rr += numPages;
    this.allocs.set(id, { id, vaddr, sizeBytes, pageIds, replicas });
    return { id, vaddr, sizeBytes, pages: allRefs, replicas };
  }

  async free(allocId: string): Promise<void> {
    const alloc = this.allocs.get(allocId);
    if (alloc === undefined) {
      throw new VramError(`vRAM: unknown allocation ${allocId}`);
    }
    for (const pageId of alloc.pageIds) {
      const entry = this.pageTable.get(pageId);
      if (entry === undefined) continue;
      for (const copy of entry.copies) {
        const node = this.findNode(copy.nodeId);
        if (node === undefined) continue;
        try {
          await this.callNode(node, "/hw/ram/drop", { pageId });
        } catch {
          // Best effort: a dead donor's pages are gone already.
        }
      }
      this.pageTable.delete(pageId);
    }
    this.allocs.delete(allocId);
    this.releaseAddress(alloc.vaddr, alloc.pageIds.length * VRAM_PAGE_SIZE_BYTES);
  }

  async write(vaddr: VAddr, data: Uint8Array): Promise<void> {
    if (data.length === 0) return;
    const alloc = this.findAllocForRange(vaddr, data.length);
    if (alloc === undefined) {
      throw new VramError(
        `vRAM: write of ${data.length} bytes at vaddr ${vaddr} is outside any allocation`,
      );
    }
    const segments = this.pageSegments(alloc, vaddr, data.length);
    // Segments touch distinct pages, so they run concurrently (bounded);
    // per-page version handling below is unchanged.
    //
    // Phase 1: fetch the pages partial writes need (read-modify-write).
    // Full-page writes need no fetch. Fetches are batched per donor.
    const partialSegs = segments.filter(
      (seg) =>
        !(seg.pageOffset === 0 && seg.rangeEnd - seg.rangeStart === VRAM_PAGE_SIZE_BYTES),
    );
    const fetchedPartial =
      partialSegs.length > 0
        ? await this.fetchSegmentPages(partialSegs)
        : new Map<string, Buffer>();

    // Build each segment's new full page contents.
    const newPages = new Map<string, Buffer>();
    for (const seg of segments) {
      const segLen = seg.rangeEnd - seg.rangeStart;
      const chunk = data.subarray(seg.rangeStart - vaddr, seg.rangeEnd - vaddr);
      if (seg.pageOffset === 0 && segLen === VRAM_PAGE_SIZE_BYTES) {
        // A write covering a whole page needs no read-modify-write.
        // When the chunk is already a Buffer, pass the view through:
        // storePagesToNode copies exactly the view's bytes, so an
        // extra Buffer.from here would be a wasted 4 MiB copy per page.
        newPages.set(
          seg.pageId,
          Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk),
        );
      } else {
        const page = fetchedPartial.get(seg.pageId);
        if (page === undefined) {
          throw new VramError(`vRAM: page ${seg.pageId} was not fetched`);
        }
        page.set(chunk, seg.pageOffset);
        newPages.set(seg.pageId, page);
      }
    }

    // Phase 2: batch-store each page to EVERY copy in <=8-page requests
    // per donor. A replica that only holds stale bytes is worse than
    // useless: if the primary dies before a read repairs it, failover
    // serves the stale bytes as if they were current, and healing then
    // re-replicates the stale bytes as the recovered page — the new
    // data is silently lost. (The previous design stored only the
    // primary and repaired replicas lazily on read; that left exactly
    // this window, and the healing tests exposed it: post-kill reads
    // returned zeros for pages whose primary had died.) A write is
    // therefore not complete until every surviving copy holds the new
    // bytes. Cost: a write sends replicas x the bytes (2x at the
    // default replica count); reads and healing stay unchanged.
    const storedCopies = new Set<string>(); // `${pageId}@${nodeId}`
    const byNode = new Map<string, Array<{ pageId: string; data: Buffer }>>();
    for (const seg of segments) {
      const entry = this.pageTable.get(seg.pageId);
      if (entry === undefined) {
        throw new VramError(`vRAM: page table has no entry for ${seg.pageId}`);
      }
      if (entry.copies.length === 0) {
        throw new VramError(
          `vRAM: page ${seg.pageId} unavailable on all 0 copies`,
        );
      }
      const page = newPages.get(seg.pageId);
      if (page === undefined) {
        throw new VramError(`vRAM: page ${seg.pageId} has no new contents`);
      }
      for (const copy of entry.copies) {
        const list = byNode.get(copy.nodeId) ?? [];
        list.push({ pageId: seg.pageId, data: page });
        byNode.set(copy.nodeId, list);
      }
    }
    await runLimited(
      [...byNode.entries()],
      VRAM_IO_CONCURRENCY,
      async ([nodeId, items]) => {
        const node = this.findNode(nodeId);
        if (node === undefined || typeof node.hwPort !== "number") return;
        for (const batch of batchesOf(items, VRAM_BATCH_PAGES)) {
          try {
            await this.storePagesToNode(nodeId, batch);
            for (const item of batch) {
              storedCopies.add(`${item.pageId}@${nodeId}`);
            }
          } catch {
            // Sequential store below retries this batch's pages.
          }
        }
      },
    );

    // Phase 3: per-segment finalization — retry any copy the batch did
    // not confirm, drop copies that still fail, then bump the version
    // and mark every surviving copy resident at the new version. The
    // first surviving copy stays the primary (copies[0]); if the old
    // primary died mid-write, the first surviving replica is promoted
    // by the filter below.
    await runLimited(segments, VRAM_IO_CONCURRENCY, async (seg) => {
      const entry = this.pageTable.get(seg.pageId);
      if (entry === undefined) {
        throw new VramError(`vRAM: page table has no entry for ${seg.pageId}`);
      }
      const page = newPages.get(seg.pageId);
      if (page === undefined) {
        throw new VramError(`vRAM: page ${seg.pageId} has no new contents`);
      }
      const failedNodeIds: string[] = [];
      for (const copy of entry.copies) {
        if (storedCopies.has(`${seg.pageId}@${copy.nodeId}`)) continue;
        try {
          await this.storePageToNode(copy.nodeId, seg.pageId, page);
          storedCopies.add(`${seg.pageId}@${copy.nodeId}`);
        } catch {
          failedNodeIds.push(copy.nodeId);
        }
      }
      const surviving = entry.copies.filter(
        (c) => !failedNodeIds.includes(c.nodeId),
      );
      if (surviving.length === 0) {
        throw new VramError(
          `vRAM: write to page ${seg.pageId} failed on all copies`,
        );
      }
      entry.copies = surviving;
      entry.version += 1;
      for (const copy of entry.copies) {
        copy.version = entry.version;
        copy.state = "resident";
      }
    });
  }

  async read(vaddr: VAddr, length: number): Promise<Uint8Array> {
    if (length === 0) return new Uint8Array(0);
    const alloc = this.findAllocForRange(vaddr, length);
    if (alloc === undefined) {
      throw new VramError(
        `vRAM: read of ${length} bytes at vaddr ${vaddr} is outside any allocation`,
      );
    }
    const segments = this.pageSegments(alloc, vaddr, length);
    const out = new Uint8Array(length);
    // Batched fetch (one request per <=8 pages per donor); segments
    // are assembled in address order below.
    const fetched = await this.fetchSegmentPages(segments);
    for (const seg of segments) {
      const page = fetched.get(seg.pageId);
      if (page === undefined) {
        throw new VramError(`vRAM: page ${seg.pageId} was not fetched`);
      }
      out.set(
        page.subarray(seg.pageOffset, seg.pageOffset + (seg.rangeEnd - seg.rangeStart)),
        seg.rangeStart - vaddr,
      );
    }
    // Lazy repair: refresh any stale replicas from the fresh pages,
    // concurrently (bounded). A dead donor just stays dirty.
    const repairs: Array<{ pageId: string; entry: PageEntry; copy: PageRef; page: Buffer }> = [];
    for (const seg of segments) {
      const entry = this.pageTable.get(seg.pageId);
      const page = fetched.get(seg.pageId);
      if (entry === undefined || page === undefined) continue;
      for (const copy of entry.copies) {
        if (copy.state !== "resident") {
          repairs.push({ pageId: seg.pageId, entry, copy, page });
        }
      }
    }
    await runLimited(repairs, VRAM_IO_CONCURRENCY, async (repair) => {
      try {
        await this.storePageToNode(repair.copy.nodeId, repair.pageId, repair.page);
        repair.copy.state = "resident";
        repair.copy.version = repair.entry.version;
      } catch {
        // Dead donor: leave it dirty; handleNodeLoss heals it.
      }
    });
    return out;
  }

  async replicate(allocId: string, targetNodeId: string): Promise<void> {
    const alloc = this.allocs.get(allocId);
    if (alloc === undefined) {
      throw new VramError(`vRAM: unknown allocation ${allocId}`);
    }
    const target = this.findNode(targetNodeId);
    if (target === undefined || typeof target.hwPort !== "number") {
      throw new VramError(`vRAM: no reachable donor node ${targetNodeId}`);
    }
    for (const pageId of alloc.pageIds) {
      const entry = this.pageTable.get(pageId);
      if (entry === undefined) continue;
      const page = await this.fetchPageData(entry, pageId);
      await this.storePageToNode(targetNodeId, pageId, page);
      const existing = entry.copies.find((c) => c.nodeId === targetNodeId);
      if (existing !== undefined) {
        existing.state = "resident";
        existing.version = entry.version;
      } else {
        entry.copies.push({
          pageId,
          nodeId: targetNodeId,
          version: entry.version,
          state: "resident",
        });
      }
    }
  }

  async handleNodeLoss(
    nodeId: string,
  ): Promise<{ pagesRecovered: number; pagesLost: number }> {
    let pagesRecovered = 0;
    let pagesLost = 0;
    for (const [pageId, entry] of this.pageTable) {
      if (!entry.copies.some((c) => c.nodeId === nodeId)) {
        continue; // Unaffected by this loss.
      }
      const surviving = entry.copies.filter((c) => c.nodeId !== nodeId);
      if (surviving.length === 0) {
        // Every copy was on the dead node: the page is gone.
        entry.copies = [];
        pagesLost += 1;
        continue;
      }
      // Promote the first surviving copy to primary (copies[0]).
      entry.copies = surviving;
      // Re-replicate back toward the allocation's replica count on
      // healthy donors that do not already hold a copy.
      const alloc = this.allocs.get(entry.allocId);
      const desired = alloc?.replicas ?? entry.copies.length + 1;
      const donors = this.healthyNodes().filter(
        (n) =>
          n.workerId !== nodeId &&
          !entry.copies.some((c) => c.nodeId === n.workerId),
      );
      for (const donor of donors) {
        if (entry.copies.length >= desired) break;
        try {
          const page = await this.fetchPageData(entry, pageId);
          await this.storePageToNode(donor.workerId, pageId, page);
          entry.copies.push({
            pageId,
            nodeId: donor.workerId,
            version: entry.version,
            state: "resident",
          });
        } catch {
          // Donor unreachable: try the next one, or recover with
          // fewer copies. The page itself survives via its primary.
        }
      }
      pagesRecovered += 1;
    }
    return { pagesRecovered, pagesLost };
  }

  stats(): VramStats {
    const nodes = this.nodes.listNodes();
    const totalBytes = nodes.reduce((sum, n) => sum + n.memoryBytes, 0);
    const pages = this.pageTable.size;
    const usedBytes = pages * VRAM_PAGE_SIZE_BYTES;
    return {
      totalBytes,
      usedBytes,
      freeBytes: Math.max(0, totalBytes - usedBytes),
      pages,
      nodes: nodes.length,
    };
  }
}
