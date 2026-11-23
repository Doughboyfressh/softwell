/**
 * Orphaned-chunk reaper — run with: node test/kv-reaper.test.ts
 *
 * The leak: a worker that crashes never frees its KV chunks. Its chunk
 * table dies with it (see test/kv-handover-resubmit.test.ts), so nobody
 * else knows the alloc ids, and the orchestrator's VirtualRam holds the
 * pages forever — until the orchestrator restarts. One crashed
 * 1024-token sequence leaks up to 12 layers x 4 MiB = 48 MiB per stage;
 * a crash loop leaks without bound.
 *
 * The fix: allocations made through POST /mesh/vram/alloc carry an
 * owner tag (the worker id — RemoteVramAccess.setOwner, called by the
 * worker once registered). The orchestrator's KV handlers track every
 * tagged allocation, and reapOwner(owner) frees a dead worker's
 * allocations. The CLI's fault-detection sweep calls it for every
 * newly-dead worker (src/cli.ts).
 *
 * Coverage:
 *  A. Handler level (Map-backed FakeRam): allocs tagged per owner +
 *     one untagged; kvAllocStats census; reapOwner frees exactly the
 *     dead owner's allocations (bytes drop by exactly the reaped
 *     amount, live reads of the survivor still work); reaping is
 *     idempotent; untagged allocations are never reaped; an explicit
 *     free removes the entry from tracking.
 *  B. Wire level (real mesh server + RemoteVramAccess with setOwner):
 *     the owner tag crosses POST /mesh/vram/alloc and shows up in the
 *     handlers' census, and reapOwner frees it.
 */
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { RemoteVramAccess } from "../src/scheduler/kvcache-vram-remote.ts";
import { MeshClient, createMeshServer } from "../src/mesh/protocol.ts";
import { createKvHandlers } from "../src/server/vram-kv.ts";
import { VramError } from "../src/hw/vram.ts";
import type { IVirtualRam, VramAllocation, VramStats } from "../src/contracts.ts";

const MIB = 1024 * 1024;
const PAGE = 4 * MIB;

/** Minimal Map-backed IVirtualRam (alloc rounds up to whole pages). */
class FakeRam implements IVirtualRam {
  private readonly allocs = new Map<string, { vaddr: number; data: Uint8Array }>();
  private bump = 0;
  private nextId = 0;

  async alloc(sizeBytes: number): Promise<VramAllocation> {
    const rounded = Math.ceil(sizeBytes / PAGE) * PAGE;
    const id = `alloc-${this.nextId++}`;
    const vaddr = this.bump;
    this.bump += rounded;
    this.allocs.set(id, { vaddr, data: new Uint8Array(rounded) });
    return { id, vaddr, sizeBytes: rounded, pages: [], replicas: 1 };
  }

  async free(allocId: string): Promise<void> {
    if (!this.allocs.delete(allocId)) {
      throw new VramError(`vRAM: unknown allocation ${allocId}`);
    }
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    for (const alloc of this.allocs.values()) {
      if (vaddr >= alloc.vaddr && vaddr + length <= alloc.vaddr + alloc.data.length) {
        return new Uint8Array(
          alloc.data.subarray(vaddr - alloc.vaddr, vaddr - alloc.vaddr + length),
        );
      }
    }
    throw new VramError(`vRAM: read of ${length} bytes at vaddr ${vaddr} is outside any allocation`);
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    for (const alloc of this.allocs.values()) {
      if (vaddr >= alloc.vaddr && vaddr + data.length <= alloc.vaddr + alloc.data.length) {
        alloc.data.set(data, vaddr - alloc.vaddr);
        return;
      }
    }
    throw new VramError(`vRAM: write of ${data.length} bytes at vaddr ${vaddr} is outside any allocation`);
  }

  async replicate(): Promise<void> {
    throw new Error("FakeRam.replicate: not implemented");
  }

  async handleNodeLoss(): Promise<{ pagesRecovered: number; pagesLost: number }> {
    return { pagesRecovered: 0, pagesLost: 0 };
  }

  stats(): VramStats {
    let used = 0;
    for (const alloc of this.allocs.values()) used += alloc.data.length;
    return { totalBytes: 1 << 30, usedBytes: used, freeBytes: (1 << 30) - used, pages: this.allocs.size, nodes: 1 };
  }
}

function byOwner(
  stats: ReturnType<ReturnType<typeof createKvHandlers>["kvAllocStats"]>,
  owner: string | null,
): { allocations: number; bytes: number } {
  const found = stats.byOwner.find((b) => b.owner === owner);
  if (found === undefined) return { allocations: 0, bytes: 0 };
  return { allocations: found.allocations, bytes: found.bytes };
}

async function main(): Promise<void> {
  // --- A. Handler level ----------------------------------------------------
  const ram = new FakeRam();
  const handlers = createKvHandlers(ram);

  const dead1 = await handlers.allocVram({ sizeBytes: PAGE, owner: "worker-dead" });
  const dead2 = await handlers.allocVram({ sizeBytes: PAGE, owner: "worker-dead" });
  const alive = await handlers.allocVram({ sizeBytes: PAGE, owner: "worker-alive" });
  const untagged = await handlers.allocVram({ sizeBytes: PAGE });
  // A live sequence's bytes, to prove the survivor keeps working.
  const marker = new Uint8Array([7, 7, 7, 7]);
  await handlers.writeVram(alive.vaddr, marker);

  const before = handlers.kvAllocStats();
  assert.equal(before.allocations, 4, `expected 4 tracked allocs, got ${before.allocations}`);
  assert.equal(before.bytes, 4 * PAGE, `expected ${4 * PAGE} tracked bytes, got ${before.bytes}`);
  assert.deepEqual(byOwner(before, "worker-dead"), { allocations: 2, bytes: 2 * PAGE });
  assert.deepEqual(byOwner(before, "worker-alive"), { allocations: 1, bytes: PAGE });
  assert.deepEqual(byOwner(before, null), { allocations: 1, bytes: PAGE });
  assert.equal(ram.stats().usedBytes, 4 * PAGE, "FakeRam must hold all 4 pages before the reap");
  console.log("census before reap: 4 allocs (dead 2, alive 1, untagged 1), 16 MiB");

  // Reap the dead worker: exactly its 2 allocations, 8 MiB.
  const reaped = await handlers.reapOwner("worker-dead");
  assert.deepEqual(reaped, { freed: 2, bytes: 2 * PAGE });
  assert.equal(ram.stats().usedBytes, 2 * PAGE, "reap must return exactly the dead worker's bytes");
  const after = handlers.kvAllocStats();
  assert.equal(after.allocations, 2, `expected 2 tracked allocs after reap, got ${after.allocations}`);
  assert.deepEqual(byOwner(after, "worker-dead"), { allocations: 0, bytes: 0 });
  assert.deepEqual(byOwner(after, "worker-alive"), { allocations: 1, bytes: PAGE });
  assert.deepEqual(byOwner(after, null), { allocations: 1, bytes: PAGE });
  // The survivor's bytes are untouched by the reap.
  assert.deepEqual(
    Array.from(await handlers.readVram(alive.vaddr, 4)),
    [7, 7, 7, 7],
    "reaping a dead worker must not disturb a live worker's chunks",
  );
  // The reaped allocations are really gone: reads reject, not zero-fill.
  await assert.rejects(
    handlers.readVram(dead1.vaddr, 4),
    (err: unknown) => err instanceof Error && /outside any allocation/.test(err.message),
  );
  console.log("reapOwner(worker-dead): freed 2 allocs / 8 MiB; survivor + untagged untouched");

  // Idempotent: a second reap frees nothing.
  assert.deepEqual(await handlers.reapOwner("worker-dead"), { freed: 0, bytes: 0 });
  // Reaping an unknown owner frees nothing.
  assert.deepEqual(await handlers.reapOwner("worker-never-existed"), { freed: 0, bytes: 0 });
  // Reaping the live worker frees only its own allocation; the untagged
  // one is never reaped by owner.
  assert.deepEqual(await handlers.reapOwner("worker-alive"), { freed: 1, bytes: PAGE });
  assert.equal(handlers.kvAllocStats().allocations, 1, "only the untagged alloc must remain");
  assert.equal(ram.stats().usedBytes, PAGE, "only the untagged page must remain in vRAM");

  // An explicit free removes the entry from tracking (no double-reap).
  await handlers.freeVram(untagged.id);
  assert.equal(handlers.kvAllocStats().allocations, 0, "explicit free must untrack the alloc");
  assert.deepEqual(await handlers.reapOwner("worker-alive"), { freed: 0, bytes: 0 });
  // Freeing an already-freed id still rejects (the 404 contract).
  await assert.rejects(handlers.freeVram(untagged.id), /unknown allocation/);
  assert.equal(dead2.id.startsWith("alloc-"), true, "alloc ids are tracked (dead2 unused otherwise)");
  console.log("idempotence + untagged-never-reaped + explicit-free untracking: ok");

  // --- B. Wire level: owner tag crosses POST /mesh/vram/alloc -------------
  const ram2 = new FakeRam();
  const handlers2 = createKvHandlers(ram2);
  const server = createMeshServer({
    allocVram: handlers2.allocVram,
    readVram: handlers2.readVram,
    writeVram: handlers2.writeVram,
    freeVram: handlers2.freeVram,
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const client = new MeshClient({ timeoutMs: 5000 });
    const access = new RemoteVramAccess(base);
    access.setOwner("worker-wire");
    const alloc = await access.alloc(PAGE, { replicas: 1 });
    assert.equal(
      byOwner(handlers2.kvAllocStats(), "worker-wire").allocations,
      1,
      "the owner tag set on the client must arrive in the handlers' census",
    );
    // Without setOwner, allocs are untagged.
    const untaggedAccess = new RemoteVramAccess(base);
    await untaggedAccess.alloc(PAGE, { replicas: 1 });
    assert.equal(
      byOwner(handlers2.kvAllocStats(), null).allocations,
      1,
      "allocs without setOwner must stay untagged",
    );
    // Reap through the handlers frees the tagged alloc over the wire.
    const reapedWire = await handlers2.reapOwner("worker-wire");
    assert.deepEqual(reapedWire, { freed: 1, bytes: PAGE });
    await assert.rejects(
      access.read(alloc.vaddr, 4),
      (err: unknown) => err instanceof Error && /404/.test(err.message),
      "a reaped chunk must reject on read (404), never zero-fill",
    );
    console.log("wire: owner tag crosses alloc, reap frees it (read then 404s): ok");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  console.log("kv-reaper test passed");
}

main().catch((err: unknown) => {
  console.error("kv-reaper test FAILED:", err);
  process.exitCode = 1;
});
