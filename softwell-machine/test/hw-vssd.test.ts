/**
 * vSSD smoke test — run with: node test/hw-vssd.test.ts (exit 0 on pass).
 *
 * Covers, in-process with fake donors and a fake vRAM pool:
 *  - 2 memoryOnly SsdNodes served over real HTTP (their routes() tables
 *    dispatched directly by a tiny in-test server — no node-server import)
 *  - put a 2500-byte object with 1 KiB chunks (3 chunks, 2 replicas),
 *    get it back byte-identical
 *  - getRange() a middle slice; only the overlapping chunks are fetched
 *  - the hot cache serves the second getRange (hotHits increases)
 *  - the vRAM tier serves a chunk evicted from a 1-chunk hot cache
 *    (vramHits increases)
 *  - kill a node, heal with a replacement donor (handleNodeLoss):
 *    chunksRecovered > 0, nothing lost, the object still reads fully
 *  - list + delete, and the manifest surviving in a second coordinator
 *    constructed from the same manifestPath
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SsdNode, VirtualSsd } from "../src/hw/vssd.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import type { HwNodeServer } from "../src/hw/node-server.ts";
import type {
  IVirtualRam,
  NodeProvider,
  RouteHandler,
  VramAllocation,
  VramStats,
  WorkerInfo,
} from "../src/contracts.ts";

// --- tiny HTTP server dispatching a node's routes() table directly --------

async function serveNode(node: SsdNode): Promise<{ server: Server; port: number }> {
  const routes = node.routes();
  const server: Server = createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const handler: RouteHandler | undefined = routes[url.pathname];
      let body: Record<string, unknown> = {};
      if (req.method !== "GET") {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk as Buffer);
        const raw = Buffer.concat(chunks).toString("utf8");
        if (raw.length > 0) {
          body = JSON.parse(raw) as Record<string, unknown>;
        }
      }
      if (!handler) {
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: `unknown route: ${url.pathname}` }));
        return;
      }
      try {
        const result = await handler(body);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(result ?? {}));
      } catch (err) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: err instanceof Error ? err.message : String(err),
          }),
        );
      }
    })().catch((err: unknown) => {
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: err instanceof Error ? err.message : String(err),
          }),
        );
      }
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address() as AddressInfo;
  return { server, port: address.port };
}

function closeServer(server: Server): Promise<void> {
  return new Promise<void>((resolve) => server.close(() => resolve()));
}

// --- fake NodeProvider ------------------------------------------------------

class FakeProvider implements NodeProvider {
  nodes: WorkerInfo[];
  constructor(nodes: WorkerInfo[]) {
    this.nodes = nodes;
  }
  listNodes(): WorkerInfo[] {
    return [...this.nodes];
  }
}

function makeWorkerInfo(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "fake",
    memoryBytes: 1 << 30,
    freeMemoryBytes: 1 << 30,
    flopsPerSec: 1e12,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 256,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

// --- fake vRAM: bump-allocated byte store behind the IVirtualRam shape ------

class FakeRam implements IVirtualRam {
  private readonly store = new Map<number, Uint8Array>();
  private readonly ids = new Map<string, number>();
  private nextVaddr = 0x10000000;
  private nextId = 0;

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    const id = `fake-alloc-${this.nextId++}`;
    const vaddr = this.nextVaddr;
    this.nextVaddr += Math.max(sizeBytes, 1);
    this.store.set(vaddr, new Uint8Array(sizeBytes));
    this.ids.set(id, vaddr);
    return { id, vaddr, sizeBytes, pages: [], replicas: 1 };
  }

  async free(allocId: string): Promise<void> {
    const vaddr = this.ids.get(allocId);
    if (vaddr !== undefined) {
      this.store.delete(vaddr);
      this.ids.delete(allocId);
    }
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const buf = this.store.get(vaddr);
    if (!buf) throw new Error(`FakeRam: no allocation at vaddr ${vaddr}`);
    return buf.subarray(0, length);
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    const buf = this.store.get(vaddr);
    if (!buf) throw new Error(`FakeRam: no allocation at vaddr ${vaddr}`);
    buf.set(data);
  }

  async replicate(): Promise<void> {
    // No-op for the fake.
  }

  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): VramStats {
    let used = 0;
    for (const buf of this.store.values()) used += buf.byteLength;
    return {
      totalBytes: 1 << 30,
      usedBytes: used,
      freeBytes: (1 << 30) - used,
      pages: this.store.size,
      nodes: 1,
    };
  }
}

// --- the test ----------------------------------------------------------------

async function main(): Promise<void> {
  const tmp = mkdtempSync(join(tmpdir(), "vssd-test-"));
  const manifestPath = join(tmp, "manifest.json");
  const nodeA = new SsdNode({ memoryOnly: true });
  const nodeB = new SsdNode({ memoryOnly: true });
  const srvA = await serveNode(nodeA);
  const srvB = await serveNode(nodeB);
  let srvC: { server: Server; port: number } | null = null;
  let binSrv: HwNodeServer | null = null;
  const provider = new FakeProvider([
    makeWorkerInfo("node-a", srvA.port),
    makeWorkerInfo("node-b", srvB.port),
  ]);
  const ram = new FakeRam();
  const ssd = new VirtualSsd({
    nodes: provider,
    chunkSizeBytes: 1024,
    defaultReplicas: 2,
    manifestPath,
    ram,
  });

  try {
    // Deterministic 2500-byte object: 3 chunks of 1024 / 1024 / 452.
    const data = new Uint8Array(2500);
    for (let i = 0; i < data.length; i++) data[i] = (i * 7 + 3) % 256;

    const meta = await ssd.putObject("models/demo.bin", data);
    assert.equal(meta.key, "models/demo.bin");
    assert.equal(meta.sizeBytes, 2500);
    assert.equal(meta.chunks, 3);
    assert.equal(meta.replicas, 2);

    // Donor-side stats over HTTP: each node holds all 3 chunks (2 nodes,
    // 2 replicas).
    const statsA = (await (
      await fetch(`http://127.0.0.1:${srvA.port}/hw/ssd/stats`)
    ).json()) as { chunks: number; bytesUsed: number };
    assert.equal(statsA.chunks, 3);
    assert.equal(statsA.bytesUsed, 2500);

    // Full read back, byte-identical.
    const back = await ssd.getObject("models/demo.bin");
    assert.ok(Buffer.from(back).equals(Buffer.from(data)), "getObject bytes differ");

    // Middle slice via getRange: offset 1000, length 500 spans chunks 0-1.
    const before = ssd.cacheStats();
    assert.equal(before.diskFetches, 3, "first getObject should fetch 3 chunks from disk");
    const slice = await ssd.getRange("models/demo.bin", 1000, 500);
    assert.ok(
      Buffer.from(slice).equals(Buffer.from(data.subarray(1000, 1500))),
      "getRange slice differs",
    );
    const after = ssd.cacheStats();
    assert.ok(
      after.hotHits > before.hotHits,
      `hot cache should serve the second getRange: ${JSON.stringify(before)} -> ${JSON.stringify(after)}`,
    );
    assert.equal(after.diskFetches, before.diskFetches, "getRange must not re-fetch from disk");

    // Manifest persistence: a second coordinator on the same manifestPath
    // sees the object. With a 1-chunk hot cache, the first chunk is evicted
    // from hot after the full read — but mirrored in vRAM — so reading it
    // back must hit the vRAM tier, not disk.
    const ssdSmall = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: 1024,
      defaultReplicas: 2,
      manifestPath,
      ram,
      hotCacheChunks: 1,
    });
    const loaded = await ssdSmall.list();
    assert.equal(loaded.length, 1);
    assert.equal(loaded[0].key, "models/demo.bin");
    await ssdSmall.getObject("models/demo.bin"); // 3 disk fetches; hot keeps only chunk 2
    const vBefore = ssdSmall.cacheStats();
    const head = await ssdSmall.getRange("models/demo.bin", 0, 100);
    assert.ok(
      Buffer.from(head).equals(Buffer.from(data.subarray(0, 100))),
      "vRAM-tier getRange slice differs",
    );
    const vAfter = ssdSmall.cacheStats();
    assert.ok(
      vAfter.vramHits > vBefore.vramHits,
      `vRAM tier should serve the hot-evicted chunk: ${JSON.stringify(vBefore)} -> ${JSON.stringify(vAfter)}`,
    );
    assert.equal(
      vAfter.diskFetches,
      vBefore.diskFetches,
      "vRAM hit must not re-fetch from disk",
    );

    // Kill node-b. A replacement donor (node-c) joins; healing must
    // re-replicate every chunk from its surviving copy on node-a.
    const nodeC = new SsdNode({ memoryOnly: true });
    srvC = await serveNode(nodeC);
    provider.nodes = [
      makeWorkerInfo("node-a", srvA.port),
      makeWorkerInfo("node-c", srvC.port),
    ];
    await closeServer(srvB.server); // node-b is dead
    const healed = await ssd.handleNodeLoss("node-b");
    assert.ok(
      healed.chunksRecovered > 0,
      `expected chunksRecovered > 0, got ${JSON.stringify(healed)}`,
    );
    assert.equal(healed.chunksLost, 0);
    assert.equal(healed.chunksRecovered, 3);

    // The object still reads fully after the loss.
    const afterLoss = await ssd.getObject("models/demo.bin");
    assert.ok(
      Buffer.from(afterLoss).equals(Buffer.from(data)),
      "object unreadable after node loss",
    );

    // The replacement donor now holds all 3 chunks.
    const statsC = (await (
      await fetch(`http://127.0.0.1:${srvC.port}/hw/ssd/stats`)
    ).json()) as { chunks: number; bytesUsed: number };
    assert.equal(statsC.chunks, 3);
    assert.equal(statsC.bytesUsed, 2500);

    // Parallel getObject: a 100 KiB object is 100 x 1 KiB chunks, far more
    // than the 8-in-flight limit, so the parallel path is exercised. The
    // bytes must be identical to the sequential read, and every chunk must
    // still be fetched from disk exactly once (no duplicates, no skips).
    const big = new Uint8Array(100 * 1024);
    for (let i = 0; i < big.length; i++) big[i] = (i * 11 + 5) % 256;
    await ssd.putObject("models/big.bin", big);
    const fetchBefore = ssd.cacheStats();
    const bigBack = await ssd.getObject("models/big.bin");
    assert.ok(
      Buffer.from(bigBack).equals(Buffer.from(big)),
      "parallel getObject bytes differ",
    );
    const fetchAfter = ssd.cacheStats();
    assert.equal(
      fetchAfter.diskFetches - fetchBefore.diskFetches,
      100,
      `parallel getObject should fetch 100 chunks from disk, got ${JSON.stringify(fetchAfter)}`,
    );
    // Parallel getRange over the same object: a slice spanning many chunks.
    const bigSlice = await ssd.getRange("models/big.bin", 10 * 1024, 50 * 1024);
    assert.ok(
      Buffer.from(bigSlice).equals(Buffer.from(big.subarray(10 * 1024, 60 * 1024))),
      "parallel getRange slice differs",
    );
    await ssd.deleteObject("models/big.bin");

    // Binary chunk round-trip: a real node server serving ONLY the binary
    // routes (plus stats) — the JSON routes 404, so a successful put/get
    // proves the coordinator used the binary path end to end.
    const binNode = new SsdNode({ memoryOnly: true });
    const binRoutes = Object.fromEntries(
      Object.entries(binNode.routes()).filter(
        ([path]) => path.endsWith("-bin") || path === "/hw/ssd/stats",
      ),
    );
    binSrv = await createHwNodeServer({ routeTables: [binRoutes] });
    const binProvider = new FakeProvider([
      makeWorkerInfo("node-bin", binSrv.port),
    ]);
    const binSsd = new VirtualSsd({
      nodes: binProvider,
      chunkSizeBytes: 1024,
      defaultReplicas: 1,
    });
    const binProbe = await fetch(
      `http://127.0.0.1:${binSrv.port}/hw/ssd/put-chunk`,
      { method: "POST", headers: { "content-type": "application/json" }, body: "{}" },
    );
    assert.equal(binProbe.status, 404, "binary-only server must 404 the JSON route");
    const binData = new Uint8Array(8192);
    for (let i = 0; i < binData.length; i++) binData[i] = (i * 13 + 1) % 256;
    await binSsd.putObject("bin/obj.bin", binData);
    const binBack = await binSsd.getObject("bin/obj.bin");
    assert.ok(
      Buffer.from(binBack).equals(Buffer.from(binData)),
      "binary round-trip bytes differ",
    );
    const binStats = (await (
      await fetch(`http://127.0.0.1:${binSrv.port}/hw/ssd/stats`)
    ).json()) as { chunks: number; bytesUsed: number };
    assert.equal(binStats.chunks, 8);
    assert.equal(binStats.bytesUsed, 8192);
    // A missing chunk over the binary route must surface as not-found,
    // not as a crash or garbage bytes.
    const missingBin = await fetch(
      `http://127.0.0.1:${binSrv.port}/hw/ssd/get-chunk-bin`,
      {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: "nope:0",
      },
    );
    assert.equal(missingBin.status, 200);
    assert.deepEqual(await missingBin.json(), { error: "chunk not found: nope:0" });

    // JSON fallback: the tiny in-test servers above have NO binary routes
    // (they 404/500 them), so this put/get only succeeds if the
    // coordinator falls back to the JSON routes.
    const fbData = new Uint8Array(3000);
    for (let i = 0; i < fbData.length; i++) fbData[i] = (i * 3 + 9) % 256;
    await ssd.putObject("models/fallback.bin", fbData);
    const fbBack = await ssd.getObject("models/fallback.bin");
    assert.ok(
      Buffer.from(fbBack).equals(Buffer.from(fbData)),
      "JSON-fallback round-trip bytes differ",
    );
    await ssd.deleteObject("models/fallback.bin");

    // List + delete.
    const listed = await ssd.list("models/");
    assert.equal(listed.length, 1);
    assert.equal(listed[0].key, "models/demo.bin");
    assert.equal((await ssd.list("other/")).length, 0);
    await ssd.deleteObject("models/demo.bin");
    assert.deepEqual(await ssd.list(), []);
    await assert.rejects(
      ssd.getObject("models/demo.bin"),
      /object not found/,
    );
    const stats = ssd.stats();
    assert.equal(stats.objects, 0);
    assert.equal(stats.totalBytes, 0);
    assert.equal(stats.nodes, 2);

    console.log("hw-vssd smoke test passed");
  } finally {
    await closeServer(srvA.server);
    await closeServer(srvB.server).catch(() => undefined);
    if (srvC) await closeServer(srvC.server);
    if (binSrv) await binSrv.close().catch(() => undefined);
    rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error("hw-vssd smoke test FAILED:", err);
  process.exitCode = 1;
});
