/**
 * vCPU smoke test — run with: node test/hw-vcpu.test.ts (exit 0 on pass).
 *
 * Covers, with 2 in-process CpuNodes served over real HTTP (their routes()
 * tables mounted on tiny node:http servers, as the hw node server does):
 *  - submit 'add', 'tokenize', 'detokenize' (round-trip incl. multi-byte),
 *    'sample' (greedy + seeded determinism), 'hash', 'echo' via VirtualCpu
 *  - local fallback handler when no donor advertises the kind, and
 *    fail-fast { ok: false } for a kind nobody offers
 *  - work stealing: a node with 4 queued tasks + an idle node ends with
 *    the tasks split 2/2 (asserted via queueDepths() after rebalance())
 *  - node loss: tasks queued on a killed node complete on the survivor
 *    via handleNodeLoss (at-least-once requeue)
 */

import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { CpuNode, VirtualCpu, expectedFlopsChecksum } from "../src/hw/vcpu.ts";
import type {
  CpuTask,
  NodeProvider,
  RouteHandler,
  WorkerInfo,
} from "../src/contracts.ts";

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Mount a CpuNode's routes() table on an ephemeral HTTP server. */
async function serveNode(
  node: CpuNode,
): Promise<{ server: Server; port: number; close(): Promise<void> }> {
  const routes: Record<string, RouteHandler> = node.routes();
  const server: Server = createServer((req, res) => {
    void handle(req, res, routes);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("could not determine server port");
  }
  return {
    server,
    port: address.port,
    close(): Promise<void> {
      return new Promise((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  routes: Record<string, RouteHandler>,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const handler = routes[url.pathname];
  const send = (status: number, payload: unknown): void => {
    const text = JSON.stringify(payload);
    res.writeHead(status, {
      "content-type": "application/json",
      "content-length": Buffer.byteLength(text),
    });
    res.end(text);
  };
  if (!handler) {
    send(404, { error: `unknown route: ${url.pathname}` });
    return;
  }
  let body: Record<string, unknown> = {};
  if (req.method !== "GET") {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString("utf8");
    if (raw.length > 0) body = JSON.parse(raw) as Record<string, unknown>;
  }
  try {
    send(200, await handler(body));
  } catch (err) {
    send(500, { error: err instanceof Error ? err.message : String(err) });
  }
}

function makeWorkerInfo(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device: "cpu",
    backend: "softwell-ts",
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

async function main(): Promise<void> {
  const nodeA = new CpuNode({});
  const nodeB = new CpuNode({});

  // routes() called directly: kind advertisement + stats shape.
  const kindsA = (await nodeA.routes()["/hw/cpu/kinds"]!({})) as {
    kinds: string[];
  };
  for (const kind of ["echo", "add", "mul", "tokenize", "detokenize", "sample", "hash", "flops"]) {
    assert.ok(kindsA.kinds.includes(kind), `node should advertise ${kind}`);
  }

  const serverA = await serveNode(nodeA);
  const serverB = await serveNode(nodeB);
  const infoA = makeWorkerInfo("worker-a", serverA.port);
  const infoB = makeWorkerInfo("worker-b", serverB.port);

  const listed: WorkerInfo[] = [infoA, infoB];
  const provider: NodeProvider = { listNodes: () => [...listed] };
  const vcpu = new VirtualCpu({ nodes: provider });

  try {
    // -- basic dispatch over fetch -------------------------------------
    const add = await vcpu.submit({
      id: "t-add",
      kind: "add",
      payload: { a: 2, b: 3 },
    });
    assert.equal(add.ok, true, `add should succeed: ${add.error ?? ""}`);
    assert.equal(add.result, 5);
    assert.ok(add.nodeId === "worker-a" || add.nodeId === "worker-b");

    const echo = await vcpu.submit({
      id: "t-echo",
      kind: "echo",
      payload: { hello: "softwell" },
    });
    assert.equal(echo.ok, true);
    assert.deepEqual(echo.result, { hello: "softwell" });

    // -- tokenize / detokenize round-trip (incl. multi-byte UTF-8) -----
    const text = "Hello, softwell! 🌍";
    const tokenized = await vcpu.submit({
      id: "t-tok",
      kind: "tokenize",
      payload: { text },
    });
    assert.equal(tokenized.ok, true);
    assert.deepEqual(
      tokenized.result,
      [...new TextEncoder().encode(text)],
      "tokenize must return raw UTF-8 byte ids",
    );
    const detokenized = await vcpu.submit({
      id: "t-detok",
      kind: "detokenize",
      payload: { ids: tokenized.result },
    });
    assert.equal(detokenized.ok, true);
    assert.equal(detokenized.result, text, "detokenize must invert tokenize");
    const hi = await vcpu.submit({
      id: "t-hi",
      kind: "tokenize",
      payload: { text: "hi" },
    });
    assert.deepEqual(hi.result, [104, 105]);

    // -- sample: greedy + seeded determinism ---------------------------
    const greedy = await vcpu.submit({
      id: "t-greedy",
      kind: "sample",
      payload: { logits: [0.1, 2.0, 1.0], temperature: 0 },
    });
    assert.equal(greedy.ok, true);
    assert.equal(greedy.result, 1, "greedy sample must pick the argmax");

    const sampled1 = await vcpu.submit({
      id: "t-s1",
      kind: "sample",
      payload: {
        logits: [1, 2, 3, 4],
        temperature: 1,
        topK: 0,
        topP: 1,
        seed: 42,
      },
    });
    const sampled2 = await vcpu.submit({
      id: "t-s2",
      kind: "sample",
      payload: {
        logits: [1, 2, 3, 4],
        temperature: 1,
        topK: 0,
        topP: 1,
        seed: 42,
      },
    });
    assert.equal(sampled1.ok, true);
    assert.equal(
      sampled1.result,
      sampled2.result,
      "same seed must give the same token",
    );
    assert.ok(
      typeof sampled1.result === "number" &&
        (sampled1.result as number) >= 0 &&
        (sampled1.result as number) < 4,
    );

    // -- hash: FNV-1a vectors -------------------------------------------
    const hashA = await vcpu.submit({
      id: "t-hash",
      kind: "hash",
      payload: { text: "a" },
    });
    assert.equal(hashA.result, "e40c292c");
    const hashEmpty = await vcpu.submit({
      id: "t-hash0",
      kind: "hash",
      payload: { text: "" },
    });
    assert.equal(hashEmpty.result, "811c9dc5");

    // -- flops: real measured FMA work, verifiable checksum --------------
    const flopsSize = 1_000_000;
    const flops = await vcpu.submit({
      id: "t-flops",
      kind: "flops",
      payload: { size: flopsSize, threads: 1 },
    });
    assert.equal(flops.ok, true, `flops should succeed: ${flops.error ?? ""}`);
    const flopsResult = flops.result as {
      flops: number;
      elapsedMs: number;
      tflops: number;
      checksum: number;
      size: number;
      threads: number;
    };
    assert.equal(flopsResult.flops, 2 * flopsSize, "flops = 2 per multiply-add");
    assert.equal(flopsResult.size, flopsSize);
    assert.equal(flopsResult.threads, 1);
    assert.equal(
      flopsResult.checksum,
      expectedFlopsChecksum(flopsSize),
      "flops checksum must match the independently computed value",
    );
    assert.ok(flopsResult.tflops > 0, `tflops must be positive, got ${flopsResult.tflops}`);
    assert.ok(flopsResult.elapsedMs >= 0);

    // Parallel run must produce the exact same checksum as sequential.
    const flopsPar = await vcpu.submit({
      id: "t-flops-par",
      kind: "flops",
      payload: { size: flopsSize, threads: 2 },
    });
    assert.equal(flopsPar.ok, true, `parallel flops should succeed: ${flopsPar.error ?? ""}`);
    const flopsParResult = flopsPar.result as {
      checksum: number;
      tflops: number;
      threads: number;
    };
    assert.equal(
      flopsParResult.checksum,
      flopsResult.checksum,
      "parallel checksum must equal sequential checksum for the same size",
    );
    assert.equal(
      flopsParResult.checksum,
      expectedFlopsChecksum(flopsSize),
    );
    assert.ok(flopsParResult.tflops > 0);
    assert.equal(flopsParResult.threads, 2, "parallel run must actually use 2 worker threads, not fall back to sequential");

    // -- real core reporting: stats() sums reported capability.cores ---
    // A separate coordinator over the same two donors, with cores
    // reported (6 + 2), must report 8 total cores and 2 reporting
    // nodes — not 2 x the assumed 4.
    infoA.cores = 6;
    infoB.cores = 2;
    const reportingCpu = new VirtualCpu({ nodes: provider });
    try {
      const reportingStats = reportingCpu.stats();
      assert.equal(reportingStats.nodes, 2);
      assert.equal(reportingStats.totalCores, 8, "6 + 2 reported cores");
      assert.equal(reportingStats.nodesReportingCores, 2);
      assert.equal(reportingStats.nodesAssumedCores, 0);
    } finally {
      reportingCpu.close();
    }
    // Unreported again for the rest of the test (assumed 4 each).
    delete infoA.cores;
    delete infoB.cores;
    const assumedStats = vcpu.stats();
    assert.equal(assumedStats.totalCores, 8, "2 donors x 4 assumed cores");
    assert.equal(assumedStats.nodesReportingCores, 0);
    assert.equal(assumedStats.nodesAssumedCores, 2);

    // -- local fallback + unknown kind ----------------------------------
    vcpu.registerHandler("double", (task: CpuTask) => {
      const p = task.payload as { a: number };
      return p.a * 2;
    });
    const doubled = await vcpu.submit({
      id: "t-double",
      kind: "double",
      payload: { a: 21 },
    });
    assert.equal(doubled.ok, true);
    assert.equal(doubled.result, 42);
    assert.equal(doubled.nodeId, "local", "fallback must run locally");

    const unknown = await vcpu.submit({
      id: "t-nope",
      kind: "nope",
      payload: {},
    });
    assert.equal(unknown.ok, false);
    assert.match(unknown.error ?? "", /unknown kind nope/);

    // -- work stealing ---------------------------------------------------
    // Build the queue shape synchronously (no await between enqueue and
    // rebalance, so the 250 ms timer cannot interleave): 4 tasks on
    // worker-a, worker-b idle.
    const stealTasks: Array<Promise<unknown>> = [];
    for (let i = 0; i < 4; i++) {
      stealTasks.push(
        vcpu.enqueueOnNode("worker-a", {
          id: `t-steal-${i}`,
          kind: "add",
          payload: { a: i, b: 10 },
        }),
      );
    }
    vcpu.rebalance();
    const depths = vcpu.queueDepths();
    assert.equal(depths["worker-a"], 2, `stealing must halve the busy queue, got ${JSON.stringify(depths)}`);
    assert.equal(depths["worker-b"], 2, `idle node must receive half, got ${JSON.stringify(depths)}`);
    vcpu.pump();
    const stealResults = await Promise.all(stealTasks);
    assert.deepEqual(
      (stealResults as Array<{ result: unknown }>).map((r) => r.result),
      [10, 11, 12, 13],
    );

    // -- node loss: queued tasks complete on the survivor ---------------
    const lossTasks = [
      vcpu.enqueueOnNode("worker-a", {
        id: "t-loss-0",
        kind: "add",
        payload: { a: 40, b: 2 },
      }),
      vcpu.enqueueOnNode("worker-a", {
        id: "t-loss-1",
        kind: "mul",
        payload: { a: 6, b: 7 },
      }),
    ];
    // The registry drops the dead donor from the provider list, too.
    listed.splice(0, 1);
    const loss = await vcpu.handleNodeLoss("worker-a");
    assert.deepEqual(loss, { tasksRequeued: 2, tasksLost: 0 });
    const lossResults = await Promise.all(lossTasks);
    assert.equal(lossResults[0].ok, true);
    assert.equal(lossResults[0].result, 42);
    assert.equal(lossResults[0].nodeId, "worker-b");
    assert.equal(lossResults[1].ok, true);
    assert.equal(lossResults[1].result, 42);
    assert.equal(lossResults[1].nodeId, "worker-b");

    const stats = vcpu.stats();
    assert.equal(stats.nodes, 1, "only the survivor should remain");
    assert.equal(stats.totalCores, 4, "one donor x 4 assumed cores");
    assert.equal(stats.queued, 0);
    assert.ok(stats.completed >= 10, `completed=${stats.completed}`);
    assert.ok(stats.failed >= 1, `failed=${stats.failed} (unknown kind)`);

    // Node-side stats route reflects the work its donor did.
    const nodeStats = (await nodeB.routes()["/hw/cpu/stats"]!({})) as {
      completed: number;
      failed: number;
    };
    assert.ok(nodeStats.completed > 0);

    // Let the rebalancer tick once more to prove the timer is harmless.
    await sleep(300);

    console.log("hw-vcpu: all checks passed");
  } finally {
    vcpu.close();
    await serverA.close();
    await serverB.close();
  }
}

const watchdog = setTimeout(() => {
  console.error("hw-vcpu: timed out");
  process.exit(1);
}, 20_000);

main()
  .then(() => {
    clearTimeout(watchdog);
  })
  .catch((err) => {
    clearTimeout(watchdog);
    console.error(err);
    process.exitCode = 1;
  });
