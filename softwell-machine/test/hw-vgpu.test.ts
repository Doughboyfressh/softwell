/**
 * vGPU smoke test — run with: node test/hw-vgpu.test.ts (exit 0 on pass).
 *
 * Covers, in-process with fake vRAM and a fake NodeProvider:
 *  - two fake GpuNodes served over real HTTP on ephemeral ports, mounted
 *    from their routes() tables (the way node-server.ts will mount them)
 *  - listDevices() returns 2 virtual devices, one per healthy donor
 *  - a matmul kernel (2x3 times 3x2, known values) launched through the
 *    VirtualGpu coordinator, with the correct output written back to vRAM
 *  - honest fallback reporting: a donor advertising 'cuda' still reports
 *    device 'cpu' on the KernelResult, because the reference node runs the
 *    CPU fallback kernels
 *  - stream ordering: two launches fired on one stream without awaiting
 *    run in submission order (the second consumes the first's output)
 *  - node loss: killing a node (server closed + provider marks it dead +
 *    handleNodeLoss) leaves the survivor serving later launches
 *  - transfer() replicates a registered allocation to another node
 *  - an unknown op is rejected with a clear error (routes() called directly)
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo, Server } from "node:net";
import { GpuNode, VirtualGpu } from "../src/hw/vgpu.ts";
import { matmul, matmulParallel, matmulTiled } from "../src/hw/kernels-cpu.ts";
import type {
  IVirtualRam,
  IVirtualGpu,
  KernelLaunch,
  NodeProvider,
  RouteHandler,
  TensorRef,
  VramAllocation,
  VramStats,
  WorkerInfo,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

// --- Fake vRAM: one flat byte store, bump-pointer allocations -------------

class FakeRam implements IVirtualRam {
  private mem = new Uint8Array(1 << 20);
  private next = 0;
  private allocs = new Map<string, VramAllocation>();
  private counter = 0;
  replicated: Array<{ allocId: string; targetNodeId: string }> = [];

  async alloc(sizeBytes: number, opts?: { replicas?: number; nodeAffinity?: string[] }): Promise<VramAllocation> {
    void opts;
    const id = `alloc-${++this.counter}`;
    const alloc: VramAllocation = {
      id,
      vaddr: this.next,
      sizeBytes,
      pages: [],
      replicas: opts?.replicas ?? 1,
    };
    this.next += sizeBytes;
    this.allocs.set(id, alloc);
    return alloc;
  }

  async free(allocId: string): Promise<void> {
    this.allocs.delete(allocId);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    return this.mem.slice(vaddr, vaddr + length);
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    this.mem.set(data, vaddr);
  }

  async replicate(allocId: string, targetNodeId: string): Promise<void> {
    this.replicated.push({ allocId, targetNodeId });
  }

  async handleNodeLoss(_nodeId: string): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): VramStats {
    return {
      totalBytes: this.mem.length,
      usedBytes: this.next,
      freeBytes: this.mem.length - this.next,
      pages: 0,
      nodes: 1,
    };
  }
}

// --- Fake NodeProvider ------------------------------------------------------

function makeWorker(workerId: string, hwPort: number, device: "cpu" | "cuda", flopsPerSec: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    registeredAt: Date.now(),
    lastHeartbeatAt: Date.now(),
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device,
    backend: "softwell-ts",
    memoryBytes: 8 * 2 ** 30,
    freeMemoryBytes: 4 * 2 ** 30,
    flopsPerSec,
    bandwidthBytesPerSec: 1e9,
    roles: ["mixed"],
    maxBatchTokens: 512,
    usedMemoryBytes: 0,
    activeSequences: 0,
  };
}

// --- Serve a GpuNode's routes() table over HTTP on an ephemeral port ------

async function serveNode(node: GpuNode): Promise<{ server: Server; port: number }> {
  const routes = node.routes();
  const server = createServer(async (req, res) => {
    const key = `${req.method} ${new URL(req.url ?? "/", "http://127.0.0.1").pathname}`;
    const handler: RouteHandler | undefined = routes[key];
    if (handler === undefined) {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: `no route ${key}` }));
      return;
    }
    try {
      let body: Record<string, unknown> = {};
      if (req.method === "POST") {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk as Buffer);
        const text = Buffer.concat(chunks).toString("utf-8");
        body = text.length > 0 ? (JSON.parse(text) as Record<string, unknown>) : {};
      }
      const result = await handler(body);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const port = (server.address() as AddressInfo).port;
  return { server, port };
}

// --- Helpers ----------------------------------------------------------------

function f32(values: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function readF32(ram: IVirtualRam, vaddr: number, count: number): Promise<number[]> {
  return ram.read(vaddr, count * 4).then((bytes) => [...new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))]);
}

function ref(vaddr: number, shape: number[]): TensorRef {
  return { vaddr, shape, dtype: "float32" };
}

async function main(): Promise<void> {
  const node1 = new GpuNode({ device: "cpu" });
  const node2 = new GpuNode({ device: "cuda" }); // advertises cuda, still CPU fallback
  const srv1 = await serveNode(node1);
  const srv2 = await serveNode(node2);

  const workers: WorkerInfo[] = [
    makeWorker("node-1", srv1.port, "cpu", 2e12),
    makeWorker("node-2", srv2.port, "cuda", 5e12),
  ];
  const provider: NodeProvider = { listNodes: () => [...workers] };
  const ram = new FakeRam();
  const gpu: IVirtualGpu & VirtualGpu = new VirtualGpu({ nodes: provider, ram });

  try {
    // --- listDevices: one virtual device per healthy donor -----------------
    const devices = gpu.listDevices();
    assert.equal(devices.length, 2);
    assert.deepEqual(
      devices.map((d) => d.id),
      ["vgpu-node-1", "vgpu-node-2"],
    );
    assert.equal(devices[0].computeUnits, 2); // round(2e12 / 1e12)
    assert.equal(devices[1].computeUnits, 5);
    assert.equal(devices[1].backedBy, "cuda");
    assert.equal(devices[0].memoryBytes, 4 * 2 ** 30);

    // --- matmul 2x3 * 3x2 through the coordinator --------------------------
    // a = [[1,2,3],[4,5,6]], b = [[7,8],[9,10],[11,12]]
    // a*b = [[58,64],[139,154]]
    const a = await ram.alloc(6 * 4);
    const b = await ram.alloc(6 * 4);
    const out = await ram.alloc(4 * 4);
    for (const alloc of [a, b, out]) gpu.registerAllocation(alloc);
    await ram.write(a.vaddr, f32([1, 2, 3, 4, 5, 6]));
    await ram.write(b.vaddr, f32([7, 8, 9, 10, 11, 12]));

    const stream = gpu.createStream();
    const result = await gpu.launch({
      op: "matmul",
      inputs: [ref(a.vaddr, [2, 3]), ref(b.vaddr, [3, 2])],
      outputs: [ref(out.vaddr, [2, 2])],
      params: {},
      streamId: stream,
    } satisfies KernelLaunch);
    assert.equal(result.streamId, stream);
    assert.equal(result.device, "cpu"); // CPU fallback ran, honestly reported
    assert.ok(result.elapsedMs >= 0);
    assert.deepEqual(await readF32(ram, out.vaddr, 4), [58, 64, 139, 154]);

    // Explicit placement on the 'cuda'-backed virtual device still runs the
    // CPU fallback and says so.
    const cudaResult = await gpu.launch({
      op: "matmul",
      inputs: [ref(a.vaddr, [2, 3]), ref(b.vaddr, [3, 2])],
      outputs: [ref(out.vaddr, [2, 2])],
      params: {},
      deviceId: "vgpu-node-2",
    } satisfies KernelLaunch);
    assert.equal(cudaResult.nodeId, "node-2");
    assert.equal(cudaResult.device, "cpu");
    assert.deepEqual(await readF32(ram, out.vaddr, 4), [58, 64, 139, 154]);

    // --- transfer() replicates a registered allocation ----------------------
    await gpu.transfer(a.vaddr, "node-2");
    assert.deepEqual(ram.replicated, [{ allocId: a.id, targetNodeId: "node-2" }]);
    await assert.rejects(
      gpu.transfer(999_999, "node-2"),
      (err: unknown) => err instanceof Error && /no registered allocation/.test(err.message),
    );

    // --- stream ordering: second launch consumes the first's output --------
    // t = a*b (written above), then out2 = t * I2 == t. Both launches are
    // fired without awaiting; if the stream ran them out of order, out2
    // would read the still-zero t and come out all zeros.
    const t = await ram.alloc(4 * 4);
    const ident = await ram.alloc(4 * 4);
    const out2 = await ram.alloc(4 * 4);
    for (const alloc of [t, ident, out2]) gpu.registerAllocation(alloc);
    await ram.write(ident.vaddr, f32([1, 0, 0, 1]));
    const stream2 = gpu.createStream();
    const p1 = gpu.launch({
      op: "matmul",
      inputs: [ref(a.vaddr, [2, 3]), ref(b.vaddr, [3, 2])],
      outputs: [ref(t.vaddr, [2, 2])],
      params: {},
      streamId: stream2,
    } satisfies KernelLaunch);
    const p2 = gpu.launch({
      op: "matmul",
      inputs: [ref(t.vaddr, [2, 2]), ref(ident.vaddr, [2, 2])],
      outputs: [ref(out2.vaddr, [2, 2])],
      params: {},
      streamId: stream2,
    } satisfies KernelLaunch);
    await Promise.all([p1, p2]);
    await gpu.synchronize(stream2);
    assert.deepEqual(await readF32(ram, out2.vaddr, 4), [58, 64, 139, 154]);

    // --- unknown op is rejected with a clear error (routes() direct) -------
    const kernelRoute = node1.routes()["POST /hw/gpu/kernel"];
    assert.ok(kernelRoute !== undefined);
    await assert.rejects(
      kernelRoute({ op: "bogus-op", inputs: [], outputs: [], params: {} }),
      (err: unknown) => err instanceof Error && /unknown kernel op: "bogus-op"/.test(err.message),
    );

    // --- node loss: kill node-2, survivor still serves launches ------------
    await new Promise<void>((resolve) => srv2.server.close(() => resolve()));
    workers[1].state = "dead";
    const loss = await gpu.handleNodeLoss("node-2");
    assert.equal(loss.devicesLost, 1);
    assert.equal(gpu.listDevices().length, 1);
    assert.equal(gpu.listDevices()[0].id, "vgpu-node-1");
    // Give the dead node's in-flight bookkeeping a moment to settle, then
    // launch again — it must land on the survivor.
    await sleep(10);
    const afterLoss = await gpu.launch({
      op: "matmul",
      inputs: [ref(a.vaddr, [2, 3]), ref(b.vaddr, [3, 2])],
      outputs: [ref(out.vaddr, [2, 2])],
      params: {},
    } satisfies KernelLaunch);
    assert.equal(afterLoss.nodeId, "node-1");
    assert.equal(afterLoss.device, "cpu");
    assert.deepEqual(await readF32(ram, out.vaddr, 4), [58, 64, 139, 154]);

    // --- stats ---------------------------------------------------------------
    const stats = gpu.stats();

    // --- kernels: tiled == naive, parallel == sequential (bit-identical) ---
    // Deterministic pseudo-random values (seeded, so the test is stable).
    let seed = 0x12345678;
    const nextRand = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 2 ** 32 * 2 - 1;
    };
    const randMatrix = (rows: number, cols: number): Float32Array => {
      const m = new Float32Array(rows * cols);
      for (let i = 0; i < m.length; i++) m[i] = nextRand();
      return m;
    };
    const assertBitIdentical = (x: Float32Array, y: Float32Array, label: string): void => {
      assert.equal(x.length, y.length, `${label}: length`);
      for (let i = 0; i < x.length; i++) {
        assert.equal(x[i], y[i], `${label}: element ${i} differs (${x[i]} vs ${y[i]})`);
      }
    };
    for (const [m, k, n] of [
      [3, 5, 2],
      [37, 53, 41],
      [64, 64, 64],
      [1, 7, 1],
      [17, 1, 23],
    ] as const) {
      const a = randMatrix(m, k);
      const b = randMatrix(k, n);
      const naive = matmul(a, m, k, b, k, n);
      assertBitIdentical(matmulTiled(a, m, k, b, k, n), naive, `matmulTiled ${m}x${k}x${n}`);
      assertBitIdentical(await matmulParallel(a, m, k, b, k, n), naive, `matmulParallel ${m}x${k}x${n}`);
      assertBitIdentical(
        await matmulParallel(a, m, k, b, k, n, 2),
        matmulTiled(a, m, k, b, k, n),
        `matmulParallel(threads=2) ${m}x${k}x${n}`,
      );
    }
    // Parallel above the dispatch threshold still matches exactly.
    {
      const a = randMatrix(128, 128);
      const b = randMatrix(128, 128);
      assertBitIdentical(
        await matmulParallel(a, 128, 128, b, 128, 128),
        matmulTiled(a, 128, 128, b, 128, 128),
        "matmulParallel 128^3",
      );
    }
    // Dimension errors are identical across all three paths.
    for (const fn of [
      (x: Float32Array, y: Float32Array) => matmul(x, 2, 3, y, 4, 2),
      (x: Float32Array, y: Float32Array) => matmulTiled(x, 2, 3, y, 4, 2),
      (x: Float32Array, y: Float32Array) => matmulParallel(x, 2, 3, y, 4, 2),
    ]) {
      await assert.rejects(
        (async () => {
          await fn(new Float32Array(6), new Float32Array(8));
        })(),
        (err: unknown) => err instanceof Error && /inner dimensions mismatch/.test(err.message),
      );
    }
    // GpuNode routes a large matmul through the parallel kernel over the
    // JSON route and returns the same values as the sequential kernel.
    {
      const n = 256;
      const a = randMatrix(n, n);
      const b = randMatrix(n, n);
      const expected = matmulTiled(a, n, n, b, n, n);
      const route = node1.routes()["POST /hw/gpu/kernel"];
      assert.ok(route !== undefined);
      const body = (await route({
        op: "matmul",
        inputs: [
          { dataB64: Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64"), shape: [n, n], dtype: "float32" },
          { dataB64: Buffer.from(b.buffer, b.byteOffset, b.byteLength).toString("base64"), shape: [n, n], dtype: "float32" },
        ],
        outputs: [{ shape: [n, n], dtype: "float32" }],
        params: {},
      })) as { outputs: Array<{ dataB64: string }>; device: string };
      assert.equal(body.device, "cpu");
      const raw = Buffer.from(body.outputs[0].dataB64, "base64");
      const got = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
      assertBitIdentical(got, expected, "GpuNode 256^3 matmul via JSON route");
    }

    assert.equal(stats.devices.length, 1);
    assert.equal(stats.totalFlopsPerSec, 2e12);
    assert.ok(stats.kernelsCompleted >= 4, `expected >=4 completed, got ${stats.kernelsCompleted}`);
    assert.equal(stats.kernelsFailed, 0);
    const nodeStats = (await node1.routes()["GET /hw/gpu/stats"]?.({})) as {
      kernelsCompleted: number;
      kernelsFailed: number;
      device: string;
    };
    assert.equal(nodeStats.device, "cpu");
    assert.ok(nodeStats.kernelsCompleted >= 3);

    console.log("hw-vgpu smoke test passed");
  } finally {
    if (srv1.server.listening) await new Promise<void>((resolve) => srv1.server.close(() => resolve()));
    if (srv2.server.listening) await new Promise<void>((resolve) => srv2.server.close(() => resolve()));
  }
}

main().catch((err: unknown) => {
  console.error("hw-vgpu smoke test FAILED:", err);
  process.exitCode = 1;
});
