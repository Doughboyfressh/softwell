/**
 * Binary-transport regression test — run with: node test/binary-transport.test.ts
 *
 * Weight/page/chunk bytes must cross the wire as binary
 * (application/octet-stream), never as base64 inside JSON. This test
 * proves it at the wire level instead of trusting the code or the
 * comments:
 *
 *  - Real RamNode + SsdNode donors served by the real node server sit
 *    behind a recording reverse proxy. The proxy logs every request's
 *    content-type and whether any JSON body/response carries a `dataB64`
 *    field, then forwards the request untouched to the real donor.
 *  - The REAL VirtualRam / VirtualSsd coordinators (and a real
 *    Hypervisor for stageToVram) are driven through every path that
 *    moves weight/page bytes: alloc, full + partial write, read
 *    (writes replicate eagerly to every copy, so a read needs no
 *    repair re-store), replicate, handleNodeLoss re-replication,
 *    putObject, getObject, getRange, and stageToVram.
 *  - Every phase asserts: zero base64 byte transfers, and the binary
 *    routes actually used (store-bin / store-many-bin / fetch-bin /
 *    fetch-many / put-chunk-bin / get-chunk-bin).
 *  - A second, binary-only donor set has the JSON byte routes
 *    (POST /hw/ram/fetch, /hw/ram/store, /hw/ssd/put-chunk,
 *    /hw/ssd/get-chunk) 404'd by the proxy. End-to-end success there
 *    proves the coordinators never need the JSON routes at all; the
 *    test also asserts zero attempts were made at them. Control-plane
 *    JSON (drop, drop-chunk, stats, fetch-many's pageId list) still
 *    passes through untouched.
 *
 * The JSON base64 routes stay as a fallback for pre-binary donors
 * (see vram.ts / vssd.ts); this test pins the hot path to binary so a
 * regression to base64 fails loudly.
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { request as httpRequest } from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { SsdNode, VirtualSsd } from "../src/hw/vssd.ts";
import { Hypervisor } from "../src/hw/hypervisor.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import type { HwNodeServer } from "../src/hw/node-server.ts";
import { VRAM_PAGE_SIZE_BYTES } from "../src/contracts.ts";
import type {
  IVirtualCpu,
  IVirtualGpu,
  NodeProvider,
  WorkerInfo,
} from "../src/contracts.ts";

const MIB = 1024 * 1024;
const PAGE = VRAM_PAGE_SIZE_BYTES; // 4 MiB

/** JSON byte routes: base64 carriers. Binary-only proxies 404 these. */
const JSON_BYTE_ROUTES = new Set([
  "POST /hw/ram/fetch",
  "POST /hw/ram/store",
  "POST /hw/ssd/put-chunk",
  "POST /hw/ssd/get-chunk",
]);

interface WireRecord {
  phase: string;
  method: string;
  path: string;
  reqContentType: string;
  reqHasB64: boolean;
  resStatus: number;
  resContentType: string;
  resHasB64: boolean;
  /** True when the proxy refused a JSON byte route (binary-only mode). */
  blocked: boolean;
}

let currentPhase = "setup";
const records: WireRecord[] = [];

function hasB64Field(contentType: string, buf: Buffer): boolean {
  if (contentType !== "application/json" || buf.length === 0) return false;
  try {
    const parsed: unknown = JSON.parse(buf.toString("utf8"));
    return (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof (parsed as Record<string, unknown>)["dataB64"] === "string"
    );
  } catch {
    return false;
  }
}

/**
 * Recording reverse proxy: coordinator -> proxy -> real donor server.
 * Logs what actually crossed the wire (content-types + any dataB64
 * field), forwards the request untouched. With `blockJsonByteRoutes`,
 * the JSON byte routes are answered 404 by the proxy itself — a
 * binary-only donor — and the attempt is recorded as blocked.
 */
async function startProxy(
  donorId: string,
  targetPort: number,
  opts?: { blockJsonByteRoutes?: boolean },
): Promise<{ port: number; close: () => Promise<void> }> {
  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    void (async () => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const reqBody = Buffer.concat(chunks);
      const pathname = new URL(req.url ?? "/", "http://127.0.0.1").pathname;
      const method = req.method ?? "";
      const reqCt = (req.headers["content-type"] ?? "").split(";")[0].trim();
      const key = `${method} ${pathname}`;
      if (opts?.blockJsonByteRoutes === true && JSON_BYTE_ROUTES.has(key)) {
        records.push({
          phase: currentPhase,
          method,
          path: pathname,
          reqContentType: reqCt,
          reqHasB64: hasB64Field(reqCt, reqBody),
          resStatus: 404,
          resContentType: "application/json",
          resHasB64: false,
          blocked: true,
        });
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: `binary-only donor: no route ${pathname}` }));
        return;
      }
      const fwdHeaders: Record<string, string> = {};
      for (const [k, v] of Object.entries(req.headers)) {
        if (k === "host" || k === "content-length" || k === "connection") continue;
        if (typeof v === "string") fwdHeaders[k] = v;
      }
      const fwd = httpRequest(
        {
          hostname: "127.0.0.1",
          port: targetPort,
          path: req.url,
          method,
          headers: fwdHeaders,
          // No keep-alive to the donor: the donor server's close()
          // then resolves immediately instead of waiting out idle
          // sockets.
          agent: false,
        },
        (fwdRes) => {
          const resChunks: Buffer[] = [];
          fwdRes.on("data", (c: Buffer) => resChunks.push(c));
          fwdRes.on("end", () => {
            const resBody = Buffer.concat(resChunks);
            const resCt = (fwdRes.headers["content-type"] ?? "").split(";")[0].trim();
            records.push({
              phase: currentPhase,
              method,
              path: pathname,
              reqContentType: reqCt,
              reqHasB64: hasB64Field(reqCt, reqBody),
              resStatus: fwdRes.statusCode ?? 0,
              resContentType: resCt,
              resHasB64: hasB64Field(resCt, resBody),
              blocked: false,
            });
            const outHeaders: Record<string, string> = {};
            for (const [k, v] of Object.entries(fwdRes.headers)) {
              if (
                k === "content-length" ||
                k === "transfer-encoding" ||
                k === "connection"
              ) {
                continue;
              }
              if (typeof v === "string") outHeaders[k] = v;
            }
            res.writeHead(fwdRes.statusCode ?? 500, outHeaders);
            res.end(resBody);
          });
        },
      );
      fwd.on("error", (err) => {
        records.push({
          phase: currentPhase,
          method,
          path: pathname,
          reqContentType: reqCt,
          reqHasB64: hasB64Field(reqCt, reqBody),
          resStatus: 502,
          resContentType: "",
          resHasB64: false,
          blocked: false,
        });
        res.writeHead(502, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: String(err) }));
      });
      fwd.end(reqBody);
    })().catch((err: unknown) => {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;
  return {
    port,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

// --- wire-log assertions -----------------------------------------------------

function inPhase(phase: string): WireRecord[] {
  return records.filter((r) => r.phase === phase);
}

function base64Records(recs: WireRecord[]): WireRecord[] {
  return recs.filter((r) => r.reqHasB64 || r.resHasB64);
}

/** Fail if any page/chunk bytes crossed as base64 in this phase. */
function assertNoBase64(phase: string): void {
  const bad = base64Records(inPhase(phase));
  assert.equal(
    bad.length,
    0,
    `${phase}: expected zero base64 page/chunk transfers, got ${bad.length}: ${JSON.stringify(bad.slice(0, 3))}`,
  );
}

function countRoute(recs: WireRecord[], method: string, path: string): number {
  return recs.filter((r) => r.method === method && r.path === path).length;
}

/** The route must have been used, and at least once with raw bytes. */
function assertRouteBinary(phase: string, method: string, path: string): void {
  const recs = inPhase(phase).filter(
    (r) => r.method === method && r.path === path,
  );
  assert.ok(
    recs.length > 0,
    `${phase}: expected ${method} ${path} to be used, but it was never called`,
  );
  const binary = recs.filter(
    (r) =>
      r.reqContentType === "application/octet-stream" ||
      r.resContentType === "application/octet-stream",
  );
  assert.ok(
    binary.length > 0,
    `${phase}: ${method} ${path} was called ${recs.length}x but never carried raw bytes ` +
      `(req/res content-types: ${recs.map((r) => `${r.reqContentType}->${r.resContentType}`).join(", ")})`,
  );
}

/** No request at all may have targeted the JSON byte routes. */
function assertNoJsonByteRouteAttempts(phase: string): void {
  const attempts = inPhase(phase).filter((r) =>
    JSON_BYTE_ROUTES.has(`${r.method} ${r.path}`),
  );
  assert.equal(
    attempts.length,
    0,
    `${phase}: coordinator attempted JSON byte route(s) ${attempts.map((r) => `${r.method} ${r.path}`).join(", ")} — the binary path must not need them`,
  );
}

// --- harness -----------------------------------------------------------------

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

function pattern(sizeBytes: number): Uint8Array {
  const buf = new Uint8Array(sizeBytes);
  for (let i = 0; i < sizeBytes; i++) buf[i] = (i * 31 + 7) % 256;
  return buf;
}

async function main(): Promise<void> {
  const tmp = mkdtempSync(join(tmpdir(), "binary-transport-"));
  const manifestPath = join(tmp, "manifest.json");
  const proxies: Array<{ close: () => Promise<void> }> = [];
  const donorServers: HwNodeServer[] = [];
  try {
    // --- Three donors, each serving merged RamNode + SsdNode routes -------
    const donorIds = ["node-a", "node-b", "node-c"];
    const proxyPorts: number[] = [];
    for (const id of donorIds) {
      const ramNode = new RamNode({
        capacityBytes: 64 * MIB,
        spillDir: join(tmp, id),
      });
      const ssdNode = new SsdNode({ memoryOnly: true });
      const srv = await createHwNodeServer({
        routeTables: [ramNode.routes(), ssdNode.routes()],
      });
      donorServers.push(srv);
      const proxy = await startProxy(id, srv.port);
      proxies.push(proxy);
      proxyPorts.push(proxy.port);
    }
    const provider = new FakeNodes(
      donorIds.map((id, i) => makeWorker(id, proxyPorts[i] as number)),
    );
    const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });
    const ssd = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: 1 * MIB,
      defaultReplicas: 2,
      manifestPath,
    });

    // --- vRAM: alloc / write / read / repair / replicate ------------------
    currentPhase = "vram:alloc";
    const alloc = await ram.alloc(8 * MIB, { replicas: 2 });
    assertRouteBinary("vram:alloc", "POST", "/hw/ram/store-many-bin");
    assertNoBase64("vram:alloc");

    currentPhase = "vram:write";
    const data = pattern(8 * MIB);
    await ram.write(alloc.vaddr, data);
    assertRouteBinary("vram:write", "POST", "/hw/ram/store-many-bin");
    assertNoBase64("vram:write");

    currentPhase = "vram:read";
    const back = await ram.read(alloc.vaddr, 8 * MIB);
    assert.ok(Buffer.from(back).equals(Buffer.from(data)), "read bytes differ");
    // Full-page read: batched binary fetch. No repair re-store: the
    // write stored the bytes to EVERY copy, so no replica is dirty.
    // (The old design stored only the primary and repaired replicas
    // lazily here; that left a window where a donor death before the
    // first read served stale bytes on failover — see vram.ts write().)
    assertRouteBinary("vram:read", "POST", "/hw/ram/fetch-many");
    assert.equal(
      countRoute(inPhase("vram:read"), "POST", "/hw/ram/store-bin"),
      0,
      "vram:read: no repair re-store expected — the write replicated eagerly",
    );
    assertNoBase64("vram:read");

    currentPhase = "vram:write-partial";
    const patch = pattern(128);
    await ram.write(alloc.vaddr + PAGE - 64, patch);
    // Partial write is read-modify-write: a binary fetch of the page,
    // then a binary store of the patched page.
    assertRouteBinary("vram:write-partial", "POST", "/hw/ram/fetch-many");
    assertRouteBinary("vram:write-partial", "POST", "/hw/ram/store-many-bin");
    assertNoBase64("vram:write-partial");

    currentPhase = "vram:read-repair";
    const back2 = await ram.read(alloc.vaddr + PAGE - 64, 128);
    assert.ok(Buffer.from(back2).equals(Buffer.from(patch)), "patch bytes differ");
    assertRouteBinary("vram:read-repair", "POST", "/hw/ram/fetch-many");
    // No repair re-store here either: the partial write stored the
    // patched page to every copy (see the vram:read note above).
    assert.equal(
      countRoute(inPhase("vram:read-repair"), "POST", "/hw/ram/store-bin"),
      0,
      "vram:read-repair: no repair re-store expected — the write replicated eagerly",
    );
    assertNoBase64("vram:read-repair");

    currentPhase = "vram:replicate";
    await ram.replicate(alloc.id, "node-c");
    assertRouteBinary("vram:replicate", "POST", "/hw/ram/fetch-bin");
    assertRouteBinary("vram:replicate", "POST", "/hw/ram/store-bin");
    assertNoBase64("vram:replicate");

    // --- vSSD: put / getRange / getObject (cold coordinators) -------------
    const ssdData = pattern(2 * MIB + 123);
    currentPhase = "vssd:put";
    await ssd.putObject("models/w.bin", ssdData);
    assert.equal(
      countRoute(inPhase("vssd:put"), "POST", "/hw/ssd/put-chunk-bin"),
      6,
      "vssd:put: 3 chunks x 2 replicas must all go via put-chunk-bin",
    );
    assertRouteBinary("vssd:put", "POST", "/hw/ssd/put-chunk-bin");
    assertNoBase64("vssd:put");

    currentPhase = "vssd:getRange";
    const ssdCold = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: 1 * MIB,
      defaultReplicas: 2,
      manifestPath,
    });
    const slice = await ssdCold.getRange("models/w.bin", MIB - 100, 500);
    assert.ok(
      Buffer.from(slice).equals(
        Buffer.from(ssdData.subarray(MIB - 100, MIB + 400)),
      ),
      "getRange slice differs",
    );
    assertRouteBinary("vssd:getRange", "POST", "/hw/ssd/get-chunk-bin");
    // Every get-chunk-bin response must be raw bytes, not JSON.
    for (const r of inPhase("vssd:getRange")) {
      if (r.path === "/hw/ssd/get-chunk-bin") {
        assert.equal(
          r.resContentType,
          "application/octet-stream",
          `vssd:getRange: get-chunk-bin answered ${r.resContentType}, expected octet-stream`,
        );
      }
    }
    assertNoBase64("vssd:getRange");

    currentPhase = "vssd:getObject";
    const ssdCold2 = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: 1 * MIB,
      defaultReplicas: 2,
      manifestPath,
    });
    const whole = await ssdCold2.getObject("models/w.bin");
    assert.ok(Buffer.from(whole).equals(Buffer.from(ssdData)), "getObject bytes differ");
    assert.equal(
      countRoute(inPhase("vssd:getObject"), "POST", "/hw/ssd/get-chunk-bin"),
      3,
      "vssd:getObject: 3 chunks must come via get-chunk-bin",
    );
    assertNoBase64("vssd:getObject");

    // --- Node loss: kill node-b, heal both pools --------------------------
    // Placement is deterministic: vRAM pages are [a,b] and [b,c] (the
    // replicate above added node-c to the first page only), and vSSD
    // chunks are [a,b], [b,c], [c,a] — so both pools must move real
    // bytes to heal.
    currentPhase = "vram:heal";
    await donorServers[1]!.close();
    provider.markDead("node-b");
    const healedRam = await ram.handleNodeLoss("node-b");
    assert.equal(healedRam.pagesLost, 0);
    assert.equal(healedRam.pagesRecovered, 2);
    assertRouteBinary("vram:heal", "POST", "/hw/ram/fetch-bin");
    assertRouteBinary("vram:heal", "POST", "/hw/ram/store-bin");
    assertNoBase64("vram:heal");

    currentPhase = "vram:read-after-loss";
    const back3 = await ram.read(alloc.vaddr, 8 * MIB);
    const expected = pattern(8 * MIB);
    expected.set(patch, PAGE - 64);
    assert.ok(Buffer.from(back3).equals(Buffer.from(expected)), "post-loss bytes differ");
    assertNoBase64("vram:read-after-loss");
    await ram.free(alloc.id);

    currentPhase = "vssd:heal";
    const healedSsd = await ssd.handleNodeLoss("node-b");
    assert.equal(healedSsd.chunksLost, 0);
    assert.equal(healedSsd.chunksRecovered, 2);
    assertRouteBinary("vssd:heal", "POST", "/hw/ssd/get-chunk-bin");
    assertRouteBinary("vssd:heal", "POST", "/hw/ssd/put-chunk-bin");
    assertNoBase64("vssd:heal");

    currentPhase = "vssd:get-after-loss";
    const ssdCold3 = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: 1 * MIB,
      defaultReplicas: 2,
      manifestPath,
    });
    const afterLoss = await ssdCold3.getObject("models/w.bin");
    assert.ok(
      Buffer.from(afterLoss).equals(Buffer.from(ssdData)),
      "post-loss getObject bytes differ",
    );
    assertNoBase64("vssd:get-after-loss");

    // --- Staging: ssd.getRange -> ram.write, end to end -------------------
    // Hypervisor only calls stats()/handleNodeLoss() on gpu/cpu during
    // boot/heal, so minimal stubs suffice here.
    const fakeGpu = {
      stats: () => ({
        devices: [],
        totalFlopsPerSec: 0,
        kernelsCompleted: 0,
        kernelsFailed: 0,
      }),
      handleNodeLoss: async () => ({ streamsReassigned: 0, devicesLost: 0 }),
    } as unknown as IVirtualGpu;
    const fakeCpu = {
      stats: () => ({ nodes: 0, totalCores: 0, queued: 0, completed: 0, failed: 0 }),
      handleNodeLoss: async () => ({ tasksRequeued: 0, tasksLost: 0 }),
    } as unknown as IVirtualCpu;
    const hypervisor = new Hypervisor({
      nodes: provider,
      ram,
      gpu: fakeGpu,
      cpu: fakeCpu,
      ssd,
    });
    await hypervisor.boot();
    try {
      currentPhase = "stage:stageToVram";
      const staged = await hypervisor.stageToVram("models/w.bin");
      const stagedBack = await ram.read(staged.vaddr, 2 * MIB + 123);
      assert.ok(
        Buffer.from(stagedBack).equals(Buffer.from(ssdData)),
        "staged bytes differ",
      );
      assertRouteBinary("stage:stageToVram", "POST", "/hw/ssd/get-chunk-bin");
      assertRouteBinary("stage:stageToVram", "POST", "/hw/ram/store-many-bin");
      assertNoBase64("stage:stageToVram");
      await ram.free(staged.id);
    } finally {
      await hypervisor.shutdown();
    }

    // --- Binary-only donors: JSON byte routes must never be needed --------
    const binProxies: Array<{ close: () => Promise<void> }> = [];
    const binServers: HwNodeServer[] = [];
    try {
      const binPorts: number[] = [];
      for (const id of ["node-x", "node-y"]) {
        const ramNode = new RamNode({ capacityBytes: 64 * MIB });
        const ssdNode = new SsdNode({ memoryOnly: true });
        const srv = await createHwNodeServer({
          routeTables: [ramNode.routes(), ssdNode.routes()],
        });
        binServers.push(srv);
        const proxy = await startProxy(id, srv.port, { blockJsonByteRoutes: true });
        binProxies.push(proxy);
        binPorts.push(proxy.port);
      }
      const binProvider = new FakeNodes(
        ["node-x", "node-y"].map((id, i) => makeWorker(id, binPorts[i] as number)),
      );
      const binRam = new VirtualRam({ nodes: binProvider, defaultReplicas: 2 });
      const binSsd = new VirtualSsd({
        nodes: binProvider,
        chunkSizeBytes: 1 * MIB,
        defaultReplicas: 2,
        manifestPath: join(tmp, "bin-manifest.json"),
      });

      currentPhase = "binary-only:vram";
      const binAlloc = await binRam.alloc(8 * MIB, { replicas: 2 });
      const binData = pattern(8 * MIB);
      await binRam.write(binAlloc.vaddr, binData);
      assert.deepEqual(await binRam.read(binAlloc.vaddr, 8 * MIB), binData);
      await binRam.replicate(binAlloc.id, "node-y");
      assert.deepEqual(await binRam.read(binAlloc.vaddr, 8 * MIB), binData);
      await binRam.free(binAlloc.id); // control-plane JSON (drop) still works
      assertNoJsonByteRouteAttempts("binary-only:vram");
      assertNoBase64("binary-only:vram");

      currentPhase = "binary-only:vssd";
      const binObj = pattern(2 * MIB + 123);
      await binSsd.putObject("models/bin.bin", binObj);
      const binCold = new VirtualSsd({
        nodes: binProvider,
        chunkSizeBytes: 1 * MIB,
        defaultReplicas: 2,
        manifestPath: join(tmp, "bin-manifest.json"),
      });
      assert.ok(
        Buffer.from(await binCold.getRange("models/bin.bin", MIB - 50, 200)).equals(
          Buffer.from(binObj.subarray(MIB - 50, MIB + 150)),
        ),
        "binary-only getRange bytes differ",
      );
      assert.ok(
        Buffer.from(await binCold.getObject("models/bin.bin")).equals(
          Buffer.from(binObj),
        ),
        "binary-only getObject bytes differ",
      );
      await binSsd.deleteObject("models/bin.bin"); // control-plane JSON still works
      assertNoJsonByteRouteAttempts("binary-only:vssd");
      assertNoBase64("binary-only:vssd");
    } finally {
      for (const p of binProxies) await p.close().catch(() => undefined);
      for (const s of binServers) await s.close().catch(() => undefined);
    }

    console.log(
      `binary-transport regression test passed ` +
        `(${records.length} wire requests observed, ` +
        `${base64Records(records).length} base64 byte transfers)`,
    );
  } finally {
    for (const p of proxies) await p.close().catch(() => undefined);
    for (const s of donorServers) await s.close().catch(() => undefined);
    rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err: unknown) => {
  console.error("binary-transport test FAILED:", err);
  process.exitCode = 1;
});
