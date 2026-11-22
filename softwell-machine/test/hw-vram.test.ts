/**
 * vRAM smoke test — run with: node test/hw-vram.test.ts (exit 0 on pass).
 *
 * Covers:
 *  - RamNode routes() called directly (in-process): store/fetch/drop/stats,
 *    LRU eviction with spill to disk and fetch served from spill.
 *  - RamNode routes() over real HTTP: a tiny node:http server built from
 *    the route table serves the second node. (The VirtualRam coordinator
 *    itself always speaks HTTP to donors, so both nodes get an HTTP server
 *    for the coordinator part; node A is additionally exercised directly.)
 *  - VirtualRam over a fake NodeProvider: alloc 10 MiB with replicas=2,
 *    write a known pattern, read it back, kill one node (handleNodeLoss)
 *    and verify pagesRecovered > 0 and reads still work, then free and
 *    verify the address range is reused.
 *  - Repeat with replicas=1 and verify pagesLost > 0 after a node loss.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import { VRAM_PAGE_SIZE_BYTES } from "../src/contracts.ts";
import type {
  NodeProvider,
  RouteHandler,
  WorkerInfo,
} from "../src/contracts.ts";

const MIB = 1024 * 1024;
const PAGE = VRAM_PAGE_SIZE_BYTES; // 4 MiB

function pattern(sizeBytes: number): Uint8Array {
  const buf = new Uint8Array(sizeBytes);
  for (let i = 0; i < sizeBytes; i++) buf[i] = (i * 31 + 7) % 256;
  return buf;
}

async function readRawBody(req: IncomingMessage): Promise<Buffer[]> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return chunks;
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks = await readRawBody(req);
  const text = Buffer.concat(chunks).toString("utf8");
  if (text.trim() === "") return {};
  return JSON.parse(text) as Record<string, unknown>;
}

/** Minimal HTTP server that dispatches "METHOD /path" route-table keys.
 *  Mirrors createHwNodeServer's binary handling: an
 *  application/octet-stream request arrives as body.__raw, and a
 *  Buffer/Uint8Array result is sent back raw. */
async function serveRoutes(routes: Record<string, RouteHandler>): Promise<{
  port: number;
  close: () => Promise<void>;
}> {
  const server: Server = createServer(async (req, res) => {
    try {
      const pathname = new URL(req.url ?? "/", "http://127.0.0.1").pathname;
      const handler = routes[`${req.method} ${pathname}`];
      if (handler === undefined) {
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: `no route for ${req.method} ${pathname}` }));
        return;
      }
      const isBinary =
        (req.headers["content-type"] ?? "").split(";")[0].trim() ===
        "application/octet-stream";
      const body = isBinary
        ? { __raw: Buffer.concat(await readRawBody(req)) }
        : await readJsonBody(req);
      const out = await handler(body);
      if (out instanceof Uint8Array) {
        res.writeHead(200, {
          "content-type": "application/octet-stream",
          "content-length": out.byteLength,
        });
        res.end(out);
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(out));
    } catch (err) {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(
        JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
      );
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  return {
    port,
    close: () =>
      new Promise<void>((resolve) => {
        // Drop keep-alive connections so close() does not wait them out.
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

function makeWorker(
  workerId: string,
  hwPort: number,
  state: WorkerInfo["state"],
): WorkerInfo {
  return {
    workerId,
    state,
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "test",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
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
}

async function main(): Promise<void> {
  const spillDirs: string[] = [];
  const servers: Array<{ close: () => Promise<void> }> = [];
  try {
    // --- RamNode routes() called directly (node A, in-process) -----------
    const spillA = mkdtempSync(join(tmpdir(), "vram-a-"));
    const spillB = mkdtempSync(join(tmpdir(), "vram-b-"));
    spillDirs.push(spillA, spillB);
    // Node A holds only 8 MiB in memory: 3 pages (12 MiB) force a spill.
    const nodeA = new RamNode({ capacityBytes: 8 * MIB, spillDir: spillA });
    const nodeB = new RamNode({ capacityBytes: 64 * MIB, spillDir: spillB });
    const routesA = nodeA.routes();

    const fill = Buffer.alloc(PAGE, 0xab);
    assert.deepEqual(
      await routesA["POST /hw/ram/store"]({
        pageId: "direct/p0",
        dataB64: fill.toString("base64"),
      }),
      { ok: true },
    );
    const fetched = (await routesA["POST /hw/ram/fetch"]({
      pageId: "direct/p0",
    })) as { dataB64: string; fromSpill: boolean };
    assert.equal(fetched.fromSpill, false);
    assert.deepEqual(Buffer.from(fetched.dataB64, "base64"), fill);

    // Two more pages (12 MiB total > 8 MiB capacity) evict p0 to disk.
    for (const id of ["direct/p1", "direct/p2"]) {
      await routesA["POST /hw/ram/store"]({
        pageId: id,
        dataB64: Buffer.alloc(PAGE, 0xcd).toString("base64"),
      });
    }
    const spilledFetch = (await routesA["POST /hw/ram/fetch"]({
      pageId: "direct/p0",
    })) as { dataB64: string; fromSpill: boolean };
    assert.equal(spilledFetch.fromSpill, true);
    assert.deepEqual(Buffer.from(spilledFetch.dataB64, "base64"), fill);
    const statsA = (await routesA["GET /hw/ram/stats"]({})) as {
      pages: number;
      spilled: number;
      spillReads: number;
    };
    assert.ok(statsA.spilled >= 1, "expected >= 1 spilled page");
    assert.ok(statsA.spillReads >= 1, "expected >= 1 spill-served fetch");
    for (const id of ["direct/p0", "direct/p1", "direct/p2"]) {
      assert.deepEqual(await routesA["POST /hw/ram/drop"]({ pageId: id }), {
        ok: true,
      });
    }
    const statsClean = (await routesA["GET /hw/ram/stats"]({})) as {
      pages: number;
      spilled: number;
    };
    assert.equal(statsClean.pages, 0);
    assert.equal(statsClean.spilled, 0);
    console.log("direct RamNode routes (incl. spill) ok");

    // --- RamNode routes() over real HTTP (node B) -------------------------
    // The coordinator always speaks HTTP to donors, so node A gets a
    // server too for the VirtualRam part below.
    const srvA = await serveRoutes(nodeA.routes());
    const srvB = await serveRoutes(nodeB.routes());
    servers.push(srvA, srvB);
    const httpStats = (await (
      await fetch(`http://127.0.0.1:${srvB.port}/hw/ram/stats`)
    ).json()) as { pages: number; capacityBytes: number };
    assert.equal(httpStats.pages, 0);
    assert.equal(httpStats.capacityBytes, 64 * MIB);
    console.log("HTTP RamNode routes ok");

    // --- VirtualRam, replicas=2 ------------------------------------------
    const provider = new FakeNodes([
      makeWorker("node-a", srvA.port, "healthy"),
      makeWorker("node-b", srvB.port, "healthy"),
    ]);
    const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });

    const alloc = await ram.alloc(10 * MIB, { replicas: 2 });
    assert.equal(alloc.sizeBytes, 10 * MIB);
    assert.equal(alloc.vaddr, 0);
    assert.equal(alloc.pages.length, 6, "3 pages x 2 copies");
    assert.equal(new Set(alloc.pages.map((p) => p.pageId)).size, 3);

    const data = pattern(10 * MIB);
    await ram.write(alloc.vaddr, data);
    assert.deepEqual(await ram.read(alloc.vaddr, 10 * MIB), data);

    // A straddling read/write across a page boundary.
    const straddle = pattern(64);
    await ram.write(alloc.vaddr + PAGE - 32, straddle);
    assert.deepEqual(await ram.read(alloc.vaddr + PAGE - 32, 64), straddle);

    const stats = ram.stats();
    assert.equal(stats.pages, 3);
    assert.equal(stats.nodes, 2);
    assert.equal(stats.usedBytes, 3 * PAGE);
    assert.equal(stats.totalBytes, 2 * (1 << 30));
    assert.equal(stats.freeBytes, 2 * (1 << 30) - 3 * PAGE);

    // Kill node B: stop its server and mark it dead, then heal.
    await srvB.close();
    provider.markDead("node-b");
    const healed = await ram.handleNodeLoss("node-b");
    assert.ok(healed.pagesRecovered > 0, "expected pages recovered");
    assert.equal(healed.pagesRecovered, 3);
    assert.equal(healed.pagesLost, 0);
    // Reads still work from the surviving donor (node A, incl. spill).
    const afterLoss = await ram.read(alloc.vaddr, 10 * MIB);
    const expected = pattern(10 * MIB);
    expected.set(straddle, PAGE - 32);
    assert.deepEqual(afterLoss, expected);
    console.log("replicas=2 node loss healed ok:", healed);

    // Free releases the address range for reuse.
    await ram.free(alloc.id);
    assert.equal(ram.stats().pages, 0);
    await assert.rejects(
      ram.read(alloc.vaddr, 16),
      /outside any allocation/,
    );
    const alloc2 = await ram.alloc(10 * MIB, { replicas: 2 });
    assert.equal(alloc2.vaddr, 0, "freed range should be reused");
    await ram.free(alloc2.id);

    // --- VirtualRam, replicas=1: a node loss loses pages -----------------
    const spillC = mkdtempSync(join(tmpdir(), "vram-c-"));
    const spillD = mkdtempSync(join(tmpdir(), "vram-d-"));
    spillDirs.push(spillC, spillD);
    const nodeC = new RamNode({ capacityBytes: 64 * MIB, spillDir: spillC });
    const nodeD = new RamNode({ capacityBytes: 64 * MIB, spillDir: spillD });
    const srvC = await serveRoutes(nodeC.routes());
    const srvD = await serveRoutes(nodeD.routes());
    servers.push(srvC, srvD);
    const provider2 = new FakeNodes([
      makeWorker("node-c", srvC.port, "healthy"),
      makeWorker("node-d", srvD.port, "healthy"),
    ]);
    const ram2 = new VirtualRam({ nodes: provider2, defaultReplicas: 1 });
    const alloc3 = await ram2.alloc(10 * MIB, { replicas: 1 });
    assert.equal(alloc3.pages.length, 3, "3 pages x 1 copy");
    await ram2.write(alloc3.vaddr, pattern(10 * MIB));
    assert.deepEqual(
      await ram2.read(alloc3.vaddr, 10 * MIB),
      pattern(10 * MIB),
    );

    await srvD.close();
    provider2.markDead("node-d");
    const healed2 = await ram2.handleNodeLoss("node-d");
    assert.ok(healed2.pagesLost > 0, "expected pages lost with replicas=1");
    assert.equal(healed2.pagesLost, 1, "round-robin put 1 of 3 pages on node-d");
    assert.equal(healed2.pagesRecovered, 0);
    await assert.rejects(
      ram2.read(alloc3.vaddr, 10 * MIB),
      /unavailable/,
    );
    console.log("replicas=1 node loss ok:", healed2);

    // --- Concurrent multi-page read/write + binary transport --------------
    // Two fresh donors behind the REAL node server (binary routes served).
    const spillE = mkdtempSync(join(tmpdir(), "vram-e-"));
    const spillF = mkdtempSync(join(tmpdir(), "vram-f-"));
    spillDirs.push(spillE, spillF);
    const nodeE = new RamNode({ capacityBytes: 64 * MIB, spillDir: spillE });
    const nodeF = new RamNode({ capacityBytes: 64 * MIB, spillDir: spillF });
    const srvE = await createHwNodeServer({ routeTables: [nodeE.routes()] });
    const srvF = await createHwNodeServer({ routeTables: [nodeF.routes()] });
    servers.push(srvE, srvF);
    const provider3 = new FakeNodes([
      makeWorker("node-e", srvE.port, "healthy"),
      makeWorker("node-f", srvF.port, "healthy"),
    ]);
    const ram3 = new VirtualRam({ nodes: provider3, defaultReplicas: 1 });

    // Binary routes round-trip, called directly: store-bin then fetch-bin.
    const routesE = nodeE.routes();
    const binPage = Buffer.alloc(PAGE, 0x5a);
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32BE(Buffer.byteLength("bin/p0"), 0);
    assert.deepEqual(
      await routesE["POST /hw/ram/store-bin"]({
        __raw: Buffer.concat([prefix, Buffer.from("bin/p0", "utf8"), binPage]),
      }),
      { ok: true },
    );
    const binFetched = (await routesE["POST /hw/ram/fetch-bin"]({
      __raw: Buffer.from("bin/p0", "utf8"),
    })) as Buffer;
    assert.ok(Buffer.isBuffer(binFetched), "fetch-bin must return raw bytes");
    assert.deepEqual(binFetched, binPage);
    await assert.rejects(
      routesE["POST /hw/ram/store-bin"]({ __raw: Buffer.from([0, 0]) }),
      /malformed|requires a binary body/,
    );
    await assert.rejects(
      routesE["POST /hw/ram/fetch-bin"]({ __raw: Buffer.from("nope/p0", "utf8") }),
      /page not found/,
    );
    await routesE["POST /hw/ram/drop"]({ pageId: "bin/p0" });
    console.log("binary routes round-trip ok (direct)");

    // Batched routes, called directly: store-many-bin then fetch-many.
    const binPages = [Buffer.alloc(PAGE, 0x11), Buffer.alloc(PAGE, 0x22)];
    const manyBody = (() => {
      const parts: Buffer[] = [Buffer.alloc(4)];
      (parts[0] as Buffer).writeUInt32BE(2, 0);
      for (const [i, data] of binPages.entries()) {
        const id = `bin/m${i}`;
        const prefix = Buffer.alloc(4);
        prefix.writeUInt32BE(Buffer.byteLength(id), 0);
        parts.push(prefix, Buffer.from(id, "utf8"), data);
      }
      return Buffer.concat(parts);
    })();
    assert.deepEqual(
      await routesE["POST /hw/ram/store-many-bin"]({ __raw: manyBody }),
      { ok: true, stored: 2 },
    );
    const manyFetched = (await routesE["POST /hw/ram/fetch-many"]({
      pageIds: ["bin/m0", "bin/m1"],
    })) as Buffer;
    assert.ok(Buffer.isBuffer(manyFetched), "fetch-many must return raw bytes");
    assert.equal(manyFetched.length, 2 * PAGE);
    assert.deepEqual(manyFetched.subarray(0, PAGE), binPages[0]);
    assert.deepEqual(manyFetched.subarray(PAGE), binPages[1]);
    // A batch with a missing page fails as a whole (coordinator falls
    // back to per-page fetches); a malformed batch stores nothing.
    await assert.rejects(
      routesE["POST /hw/ram/fetch-many"]({ pageIds: ["bin/m0", "nope/m9"] }),
      /page not found/,
    );
    await assert.rejects(
      routesE["POST /hw/ram/store-many-bin"]({ __raw: Buffer.from([0, 0, 0, 1, 0]) }),
      /truncated/,
    );
    await assert.rejects(
      routesE["POST /hw/ram/fetch-many"]({ pageIds: [] }),
      /non-empty/,
    );
    for (const id of ["bin/m0", "bin/m1"]) {
      await routesE["POST /hw/ram/drop"]({ pageId: id });
    }
    console.log("batched routes round-trip ok (direct)");

    // Binary routes over real HTTP: raw bytes out, no base64 anywhere.
    const httpStore = await fetch(`http://127.0.0.1:${srvE.port}/hw/ram/store-bin`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: Buffer.concat([prefix, Buffer.from("bin/p0", "utf8"), binPage]),
    });
    assert.equal(httpStore.status, 200);
    const httpFetch = await fetch(`http://127.0.0.1:${srvE.port}/hw/ram/fetch-bin`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: Buffer.from("bin/p0", "utf8"),
    });
    assert.equal(httpFetch.status, 200);
    assert.equal(httpFetch.headers.get("content-type"), "application/octet-stream");
    assert.deepEqual(Buffer.from(await httpFetch.arrayBuffer()), binPage);
    await routesE["POST /hw/ram/drop"]({ pageId: "bin/p0" });
    console.log("binary routes round-trip ok (HTTP)");

    // 32 MiB = 8 pages, replicas=1: concurrent write/read must round-trip
    // exactly, and placement must stripe primaries across both donors.
    const alloc4 = await ram3.alloc(32 * MIB, { replicas: 1 });
    assert.equal(alloc4.pages.length, 8, "8 pages x 1 copy");
    const primaryCounts = new Map<string, number>();
    for (const ref of alloc4.pages) {
      primaryCounts.set(ref.nodeId, (primaryCounts.get(ref.nodeId) ?? 0) + 1);
    }
    assert.equal(primaryCounts.get("node-e"), 4, "striping: 4 primaries on node-e");
    assert.equal(primaryCounts.get("node-f"), 4, "striping: 4 primaries on node-f");
    const big = pattern(32 * MIB);
    await ram3.write(alloc4.vaddr, big);
    assert.deepEqual(await ram3.read(alloc4.vaddr, 32 * MIB), big);
    // Partial writes at several page boundaries (read-modify-write).
    for (const boundary of [PAGE - 16, 3 * PAGE - 1, 7 * PAGE + 1024]) {
      const patch = pattern(128);
      await ram3.write(alloc4.vaddr + boundary, patch);
      big.set(patch, boundary);
      assert.deepEqual(await ram3.read(alloc4.vaddr + boundary, 128), patch);
    }
    assert.deepEqual(await ram3.read(alloc4.vaddr, 32 * MIB), big);
    await ram3.free(alloc4.id);
    console.log("concurrent multi-page read/write + striping ok");

    // JSON fallback: donors serving ONLY the JSON routes (no binary
    // routes -> 404) must still work end to end via the fallback.
    const nodeG = new RamNode({ capacityBytes: 64 * MIB });
    const nodeH = new RamNode({ capacityBytes: 64 * MIB });
    // serveRoutes (above) serves exactly the route table it is given,
    // so strip the binary routes to simulate a pre-binary donor.
    const jsonOnly = (routes: Record<string, RouteHandler>) => {
      const out: Record<string, RouteHandler> = {};
      for (const [key, handler] of Object.entries(routes)) {
        if (!key.endsWith("-bin")) out[key] = handler;
      }
      return out;
    };
    const srvG = await serveRoutes(jsonOnly(nodeG.routes()));
    const srvH = await serveRoutes(jsonOnly(nodeH.routes()));
    servers.push(srvG, srvH);
    const provider4 = new FakeNodes([
      makeWorker("node-g", srvG.port, "healthy"),
      makeWorker("node-h", srvH.port, "healthy"),
    ]);
    const ram4 = new VirtualRam({ nodes: provider4, defaultReplicas: 1 });
    const alloc5 = await ram4.alloc(12 * MIB, { replicas: 1 });
    const small = pattern(12 * MIB);
    await ram4.write(alloc5.vaddr, small);
    assert.deepEqual(await ram4.read(alloc5.vaddr, 12 * MIB), small);
    await ram4.free(alloc5.id);
    console.log("JSON fallback (pre-binary donors) ok");

    console.log("hw-vram smoke test passed");
  } finally {
    for (const srv of servers) {
      try {
        await srv.close();
      } catch {
        // Already closed (the killed nodes).
      }
    }
    for (const dir of spillDirs) rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error("hw-vram smoke test FAILED:", err);
  process.exitCode = 1;
});
