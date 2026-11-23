/**
 * Remote VramAccess over HTTP — unit + integration tests.
 * Run with: node --test test/vram-kv-http.test.ts
 *
 *  A. RemoteVramAccess against a fake HTTP server (Map backend, same
 *     status contract as the real endpoints): alloc/write/read/free
 *     round-trip byte-identical, partial ranges, unknown alloc/id
 *     rejects (never zero-fills), client-side validation, zero-length
 *     ops make no request, counters, replicas passthrough.
 *  B. Real mesh routes + real createKvHandlers over a FakeRam:
 *     the shipped route table (POST /mesh/vram/alloc, GET
 *     /mesh/vram/read, POST /mesh/vram/write, POST /mesh/vram/free)
 *     with 400s for malformed requests, 404 for unknown addresses /
 *     ids, 405 for the wrong method, and byte-identical round-trips
 *     through MeshClient. Same construction the CLI wires in.
 *  C. Integration: real VirtualRam + TWO REAL donor RamNodes over
 *     loopback HTTP (binary page transport, replicas=2) behind a real
 *     mesh server, driven through RemoteVramAccess. Alloc, write and
 *     read back byte-identical; kill the donor holding the PRIMARY
 *     copy, and reads still return byte-identical data (failover to
 *     the surviving replica, before any healing); a write after the
 *     death lands on the survivor and reads back the NEW bytes (no
 *     stale bytes); handleNodeLoss heals with pagesLost === 0 and the
 *     new bytes still read back (healing re-replicates current bytes,
 *     not stale zeros); free, then a read rejects instead of
 *     zero-filling.
 *
 * Placement note for C: VirtualRam places a one-page alloc's copies
 * round-robin starting at its internal counter, which starts at 0, so
 * the first alloc's copies are [bench-a, bench-b] (primary: bench-a)
 * and the second's are [bench-b, bench-a] (primary: bench-b). Killing
 * bench-a therefore forces a primary failover for the first alloc's
 * reads; both allocs are read back, so the failover path is
 * necessarily exercised.
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { RemoteVramAccess } from "../src/scheduler/kvcache-vram-remote.ts";
import { MeshClient, MeshError, createMeshServer } from "../src/mesh/protocol.ts";
import { createKvHandlers } from "../src/server/vram-kv.ts";
import { RamNode, VirtualRam, VramError } from "../src/hw/vram.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import type {
  IVirtualRam,
  NodeProvider,
  VramAllocation,
  VramStats,
  WorkerInfo,
} from "../src/contracts.ts";

const MIB = 1024 * 1024;
const PAGE = 4 * MIB;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Deterministic pseudo-random bytes (LCG) — same bytes every run. */
function patternBytes(seed: number, n: number): Uint8Array {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s >>> 24) & 0xff;
  }
  return out;
}

function assertRejectsMesh(
  p: Promise<unknown>,
  status: number,
  what: string,
): Promise<void> {
  return assert.rejects(
    p,
    (err: unknown) =>
      err instanceof MeshError && err.status === status,
    what,
  );
}

// ---------------------------------------------------------------------------
// A. Fake HTTP server (Map backend, real status contract)
// ---------------------------------------------------------------------------

interface FakeRequestLog {
  method: string;
  path: string;
}

class FakeKvHttp {
  readonly requests: FakeRequestLog[] = [];
  /** Last replicas value received on alloc (undefined = omitted). */
  lastReplicas: number | undefined = undefined;
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();
  private bump = 0x1000;
  private nextId = 0;
  private server: Server | undefined;
  url = "";

  async start(): Promise<void> {
    const server = createServer((req, res) => {
      void this.handle(req, res).catch((err: unknown) => {
        if (!res.headersSent) {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: String(err) }));
        } else res.end();
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    this.server = server;
    this.url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  }

  async stop(): Promise<void> {
    if (this.server !== undefined) {
      await new Promise<void>((resolve) => this.server?.close(() => resolve()));
    }
  }

  private sendJson(res: ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, {
      "content-type": "application/json",
      "content-length": Buffer.byteLength(text),
    });
    res.end(text);
  }

  private find(vaddr: number, length: number): { base: number; data: Uint8Array } {
    for (const { vaddr: base, data } of this.allocs.values()) {
      if (vaddr >= base && vaddr + length <= base + data.length) return { base, data };
    }
    throw new VramError(
      `vRAM: read of ${length} bytes at vaddr ${vaddr} is outside any allocation`,
    );
  }

  private async readBody(req: IncomingMessage): Promise<Buffer> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks);
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const path = url.pathname;
    this.requests.push({ method: req.method ?? "", path });
    if (path === "/mesh/vram/alloc" && req.method === "POST") {
      const body = JSON.parse((await this.readBody(req)).toString("utf8")) as {
        sizeBytes: number;
        replicas?: number;
      };
      this.lastReplicas = body.replicas;
      const id = `fake-${this.nextId++}`;
      const vaddr = this.bump;
      this.bump += body.sizeBytes;
      this.allocs.set(id, { vaddr, data: new Uint8Array(body.sizeBytes) });
      this.sendJson(res, 200, { id, vaddr, sizeBytes: body.sizeBytes });
      return;
    }
    if (path === "/mesh/vram/read" && req.method === "GET") {
      const vaddr = Number(url.searchParams.get("vaddr"));
      const length = Number(url.searchParams.get("length"));
      try {
        const { base, data } = this.find(vaddr, length);
        const slice = data.slice(vaddr - base, vaddr - base + length);
        res.writeHead(200, {
          "content-type": "application/octet-stream",
          "content-length": slice.length,
        });
        res.end(slice);
      } catch (err) {
        this.sendJson(res, 404, { error: err instanceof Error ? err.message : String(err) });
      }
      return;
    }
    if (path === "/mesh/vram/write" && req.method === "POST") {
      const vaddr = Number(url.searchParams.get("vaddr"));
      const data = await this.readBody(req);
      try {
        const { base, data: stored } = this.find(vaddr, data.length);
        stored.set(data, vaddr - base);
        this.sendJson(res, 200, { bytesWritten: data.length });
      } catch (err) {
        this.sendJson(res, 404, { error: err instanceof Error ? err.message : String(err) });
      }
      return;
    }
    if (path === "/mesh/vram/free" && req.method === "POST") {
      const body = JSON.parse((await this.readBody(req)).toString("utf8")) as { id: string };
      if (!this.allocs.has(body.id)) {
        this.sendJson(res, 404, { error: `vRAM: unknown allocation ${body.id}` });
        return;
      }
      this.allocs.delete(body.id);
      this.sendJson(res, 200, { freed: body.id });
      return;
    }
    this.sendJson(res, 404, { error: `unknown route: ${req.method} ${path}` });
  }
}

async function testClientAgainstFake(): Promise<void> {
  const fake = new FakeKvHttp();
  await fake.start();
  try {
    const access = new RemoteVramAccess(fake.url);

    // Alloc / write / read round-trip, byte-identical; partial range.
    const alloc = await access.alloc(PAGE, { replicas: 2 });
    assert.equal(fake.lastReplicas, 2, "replicas reaches the server");
    const pattern = patternBytes(7, PAGE);
    await access.write(alloc.vaddr, pattern);
    const back = await access.read(alloc.vaddr, PAGE);
    assert.deepEqual(Buffer.from(back), Buffer.from(pattern), "full-page read is byte-identical");
    const partial = await access.read(alloc.vaddr + MIB, 8192);
    assert.deepEqual(
      Buffer.from(partial),
      Buffer.from(pattern.subarray(MIB, MIB + 8192)),
      "partial read matches the written bytes",
    );
    // Overwrite a sub-range; the rest is untouched.
    const patch = patternBytes(99, 4096);
    await access.write(alloc.vaddr + 2 * MIB, patch);
    const patched = await access.read(alloc.vaddr, PAGE);
    const expected = Uint8Array.from(pattern);
    expected.set(patch, 2 * MIB);
    assert.deepEqual(Buffer.from(patched), Buffer.from(expected), "sub-range overwrite is exact");

    // Counters count actual HTTP ops.
    assert.equal(access.allocs, 1);
    assert.equal(access.bytesWritten, PAGE + 4096);
    assert.equal(access.bytesRead, PAGE + 8192 + PAGE);
    assert.equal(access.frees, 0);

    // Zero-length ops make no request (VirtualRam semantics).
    const before = fake.requests.length;
    assert.deepEqual(await access.read(alloc.vaddr, 0), new Uint8Array(0));
    await access.write(alloc.vaddr, new Uint8Array(0));
    assert.equal(fake.requests.length, before, "zero-length ops issue no HTTP request");

    // Client-side validation rejects before any request.
    await assert.rejects(access.read(-1, 8), /vaddr must be a non-negative integer/);
    await assert.rejects(access.read(0, -1), /length must be a non-negative integer/);
    await assert.rejects(access.alloc(0), /sizeBytes must be a positive integer/);
    await assert.rejects(access.alloc(8, { replicas: 0 }), /replicas must be a positive integer/);
    await assert.rejects(access.free(""), /id must be a non-empty string/);
    assert.throws(() => new RemoteVramAccess(""), /orchestratorUrl must be a non-empty string/);

    // Unknown address / id reject with the HTTP status — never zero-fill.
    // (0xdead would still land inside the 4 MiB alloc above; use an
    // address far past every allocation the fake ever handed out.)
    const OUTSIDE = 1 << 30;
    await assertRejectsMesh(access.read(OUTSIDE, 16), 404, "read outside any allocation is 404");
    await assertRejectsMesh(
      access.write(OUTSIDE, new Uint8Array(16)),
      404,
      "write outside any allocation is 404",
    );
    await assertRejectsMesh(access.free("no-such-id"), 404, "free of unknown id is 404");

    // Free, then the range is gone: reads reject, double free rejects.
    await access.free(alloc.id);
    assert.equal(access.frees, 1);
    await assertRejectsMesh(access.read(alloc.vaddr, 16), 404, "read after free is 404");
    await assertRejectsMesh(access.free(alloc.id), 404, "double free is 404");

    // Trailing-slash base URL is tolerated.
    const access2 = new RemoteVramAccess(`${fake.url}/`);
    const a2 = await access2.alloc(64);
    await access2.write(a2.vaddr, patternBytes(3, 64));
    assert.deepEqual(
      Buffer.from(await access2.read(a2.vaddr, 64)),
      Buffer.from(patternBytes(3, 64)),
    );
    await access2.free(a2.id);

    console.log("A. RemoteVramAccess against fake HTTP server: passed");
  } finally {
    await fake.stop();
  }
}

// ---------------------------------------------------------------------------
// B. Real mesh routes + createKvHandlers over a FakeRam
// ---------------------------------------------------------------------------

/** Map-backed IVirtualRam that throws real VramErrors, like VirtualRam. */
class FakeRam implements IVirtualRam {
  readonly totalBytes: number;
  private nextVaddr = 0x1000;
  private nextId = 0;
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();

  constructor(totalBytes: number) {
    this.totalBytes = totalBytes;
  }

  async alloc(sizeBytes: number, opts?: { replicas?: number }): Promise<VramAllocation> {
    void opts;
    if (sizeBytes > this.totalBytes) {
      throw new VramError(
        `vRAM: could not place page: out of memory (${sizeBytes} > ${this.totalBytes})`,
      );
    }
    const id = `alloc-${++this.nextId}`;
    const vaddr = this.nextVaddr;
    this.nextVaddr += sizeBytes;
    this.allocs.set(id, { vaddr, data: new Uint8Array(sizeBytes) });
    return { id, vaddr, sizeBytes, pages: [], replicas: opts?.replicas ?? 1 };
  }

  async free(allocId: string): Promise<void> {
    if (!this.allocs.has(allocId)) {
      throw new VramError(`vRAM: unknown allocation ${allocId}`);
    }
    this.allocs.delete(allocId);
  }

  private find(vaddr: number, length: number): { base: number; data: Uint8Array } {
    for (const { vaddr: base, data } of this.allocs.values()) {
      if (vaddr >= base && vaddr + length <= base + data.length) return { base, data };
    }
    throw new VramError(
      `vRAM: read of ${length} bytes at vaddr ${vaddr} is outside any allocation`,
    );
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    const { base, data } = this.find(vaddr, length);
    return data.slice(vaddr - base, vaddr - base + length);
  }

  async write(vaddr: number, chunk: Uint8Array): Promise<void> {
    const { base, data } = this.find(vaddr, chunk.length);
    data.set(chunk, vaddr - base);
  }

  async replicate(): Promise<void> {}
  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }
  stats(): VramStats {
    return { totalBytes: this.totalBytes, usedBytes: 0, freeBytes: this.totalBytes, pages: 0, nodes: 1 };
  }
}

async function testRoutesOverFakeRam(): Promise<void> {
  const ram = new FakeRam(64 * MIB);
  const server = createMeshServer({ ...createKvHandlers(ram) });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const client = new MeshClient();
  try {
    // Round-trip through MeshClient (the methods RemoteVramAccess uses).
    const alloc = await client.allocVram(url, 8192, { replicas: 2 });
    assert.equal(typeof alloc.id, "string");
    assert.equal(typeof alloc.vaddr, "number");
    assert.equal(alloc.sizeBytes, 8192);
    const bytes = patternBytes(11, 8192);
    const written = await client.writeVram(url, alloc.vaddr, bytes);
    assert.equal(written.bytesWritten, 8192);
    assert.deepEqual(
      Buffer.from(await client.readVram(url, alloc.vaddr, 8192)),
      Buffer.from(bytes),
      "mesh route round-trip is byte-identical",
    );
    const freed = await client.freeVram(url, alloc.id);
    assert.equal(freed.freed, alloc.id);

    // Unknown address / id -> 404; OOM alloc -> 503.
    await assertRejectsMesh(client.readVram(url, 0xdead, 16), 404, "read outside alloc is 404");
    await assertRejectsMesh(
      client.writeVram(url, 0xdead, new Uint8Array(16)),
      404,
      "write outside alloc is 404",
    );
    await assertRejectsMesh(client.freeVram(url, "no-such"), 404, "free unknown id is 404");
    await assertRejectsMesh(
      client.allocVram(url, 1024 * MIB),
      503,
      "alloc past FakeRam capacity is 503",
    );

    // Malformed requests -> 400 (raw fetch, so the JSON error is visible).
    const badAlloc = await fetch(`${url}/mesh/vram/alloc`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sizeBytes: 0 }),
    });
    assert.equal(badAlloc.status, 400, "alloc sizeBytes 0 is 400");
    const badRead = await fetch(`${url}/mesh/vram/read?vaddr=16`);
    assert.equal(badRead.status, 400, "read without length is 400");
    const badRead2 = await fetch(`${url}/mesh/vram/read?vaddr=16&length=abc`);
    assert.equal(badRead2.status, 400, "read with non-integer length is 400");
    const badFree = await fetch(`${url}/mesh/vram/free`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(badFree.status, 400, "free without id is 400");
    const emptyWrite = await fetch(`${url}/mesh/vram/write?vaddr=16`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: new Uint8Array(0),
    });
    assert.equal(emptyWrite.status, 400, "empty write body is 400");

    // Wrong method -> 405.
    const wrongMethod = await fetch(`${url}/mesh/vram/read?vaddr=16&length=16`, {
      method: "POST",
    });
    assert.equal(wrongMethod.status, 405, "POST on the read route is 405");

    console.log("B. Real mesh routes + createKvHandlers over FakeRam: passed");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

// ---------------------------------------------------------------------------
// C. Integration: real VirtualRam + two real donors, one donor dies
// ---------------------------------------------------------------------------

function makeWorker(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "integration",
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
}

async function testIntegrationTwoDonors(): Promise<void> {
  const nodeA = new RamNode({ capacityBytes: 1 << 30 });
  const nodeB = new RamNode({ capacityBytes: 1 << 30 });
  const srvA = await createHwNodeServer({ routeTables: [nodeA.routes()] });
  const srvB = await createHwNodeServer({ routeTables: [nodeB.routes()] });
  const provider = new FakeNodes([
    makeWorker("bench-a", srvA.port),
    makeWorker("bench-b", srvB.port),
  ]);
  const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });

  const mesh = createMeshServer({ ...createKvHandlers(ram) });
  await new Promise<void>((resolve) => mesh.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(mesh.address() as AddressInfo).port}`;
  const access = new RemoteVramAccess(url);

  try {
    // Two one-page allocs: round-robin placement puts the first's
    // primary on bench-a and the second's on bench-b (see header).
    const a1 = await access.alloc(PAGE, { replicas: 2 });
    const a2 = await access.alloc(PAGE, { replicas: 2 });
    assert.equal(ram.stats().pages, 2, "two pages placed in pooled vRAM");
    const p1 = patternBytes(21, PAGE);
    const p2 = patternBytes(22, PAGE);
    await access.write(a1.vaddr, p1);
    await access.write(a2.vaddr, p2);
    assert.deepEqual(Buffer.from(await access.read(a1.vaddr, PAGE)), Buffer.from(p1));
    assert.deepEqual(Buffer.from(await access.read(a2.vaddr, PAGE)), Buffer.from(p2));

    // Kill bench-a — the PRIMARY copy holder for the first alloc.
    // No healing call yet: VirtualRam.read must fail over on its own.
    await srvA.close();
    await sleep(50);
    assert.deepEqual(
      Buffer.from(await access.read(a1.vaddr, PAGE)),
      Buffer.from(p1),
      "read after primary donor death returns byte-identical data (failover, no healing yet)",
    );
    assert.deepEqual(
      Buffer.from(await access.read(a2.vaddr, PAGE)),
      Buffer.from(p2),
      "second alloc still reads byte-identical after the death",
    );

    // Write AFTER the death: the new bytes must land on the survivor
    // and read back as the new bytes — never the pre-death bytes.
    const p1b = patternBytes(23, PAGE);
    await access.write(a1.vaddr, p1b);
    assert.deepEqual(
      Buffer.from(await access.read(a1.vaddr, PAGE)),
      Buffer.from(p1b),
      "post-death write reads back the new bytes, not stale pre-death bytes",
    );

    // Heal: the dead donor's copies are pruned; nothing is lost, and
    // the healed pages still hold the CURRENT bytes. pagesRecovered is
    // 1, not 2: the post-death write already pruned bench-a's copy of
    // the first alloc's page (writes drop dead copies), so only the
    // second alloc's page still has a bench-a copy for the heal to
    // recover.
    const healed = await ram.handleNodeLoss("bench-a");
    assert.equal(healed.pagesLost, 0, "no pages lost when one of two donors dies");
    assert.equal(healed.pagesRecovered, 1, "the page still holding a dead copy is recovered");
    assert.deepEqual(
      Buffer.from(await access.read(a1.vaddr, PAGE)),
      Buffer.from(p1b),
      "post-heal read still returns the post-death bytes",
    );
    assert.deepEqual(
      Buffer.from(await access.read(a2.vaddr, PAGE)),
      Buffer.from(p2),
      "post-heal read of the second alloc is unchanged",
    );

    // Free, then reads reject instead of zero-filling.
    await access.free(a1.id);
    await access.free(a2.id);
    await assertRejectsMesh(access.read(a1.vaddr, 16), 404, "read after free is 404");
    await assertRejectsMesh(access.read(a2.vaddr, 16), 404, "read after free is 404");

    console.log("C. Integration (real VirtualRam + 2 donors, donor death): passed");
  } finally {
    await new Promise<void>((resolve) => mesh.close(() => resolve()));
    await srvB.close();
  }
}

async function main(): Promise<void> {
  await testClientAgainstFake();
  await testRoutesOverFakeRam();
  await testIntegrationTwoDonors();
  console.log("vram-kv-http tests passed");
}

main().catch((err: unknown) => {
  console.error("vram-kv-http tests FAILED:", err);
  process.exitCode = 1;
});
