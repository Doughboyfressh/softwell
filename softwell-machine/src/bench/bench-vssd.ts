/**
 * bench-vssd.ts — vSSD benchmark: sequential read/write GB/s and random
 * 4 KiB read IOPS across 1..N local donor node servers, compared against
 * the Micron 9650 (PCIe Gen6).
 *
 * Method (all measured on this machine, nothing hardcoded):
 *  - Boot `opts.donors` disk-backed SsdNode servers in-process over real
 *    HTTP (the same createHwNodeServer + VirtualSsd path the demo uses).
 *  - Sequential: a large object (256 MiB, 1 MiB chunks, 1 replica) is
 *    written with timed putObject, then read back with timed getObject
 *    from a FRESH coordinator sharing the manifest (cold hot cache, no
 *    vRAM tier) and byte-compared. Two runs each, median taken.
 *  - Random 4 KiB read: a second object with 4 KiB chunks (so one
 *    getRange = one chunk fetch, no read amplification) is read at
 *    seeded-random offsets, 32 getRange calls in flight, timed;
 *    IOPS = completed ops / wall seconds. Every returned slice is
 *    byte-checked against the written data.
 *  - Capacity: informational row — GiB actually stored in this run.
 *    Donor-disk capacity is not probed; the note says so.
 *
 * Honest caveats: random IOPS over one HTTP round trip per 4 KiB op is
 * latency-bound (expect thousands, not millions) — the donorsToBeat
 * column says how many donors that rate implies. Sequential throughput
 * is bounded by local-disk speed and loopback HTTP, not by the pool
 * design; replicas=1 is used so replication does not halve the write
 * number (noted on every row).
 *
 * Usage: node src/bench/bench-vssd.ts [--donors N] [--size small|large]
 *        [--json-out path]
 */
import { randomBytes } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkerInfo } from "../contracts.ts";
import { createHwNodeServer } from "../hw/node-server.ts";
import type { HwNodeServer } from "../hw/node-server.ts";
import { SsdNode, VirtualSsd } from "../hw/vssd.ts";
import { VSSD_TARGETS } from "./targets.ts";
import { formatResultsTable, makeResult, median } from "./types.ts";
import type { BenchOptions, BenchResult } from "./types.ts";

const SEQ_CHUNK_BYTES = 1024 * 1024; // 1 MiB — parallelism shows
const RAND_CHUNK_BYTES = 4 * 1024; // 4 KiB — one op = one chunk fetch
const RAND_CONCURRENCY = 32;
const REPLICAS = 1; // max-throughput config; replication would halve writes

interface BenchSizes {
  seqBytes: number;
  randBytes: number;
  randOps: number;
}

function sizesFor(size?: string): BenchSizes {
  if (size === "small") {
    return { seqBytes: 16 * 1024 * 1024, randBytes: 4 * 1024 * 1024, randOps: 200 };
  }
  return { seqBytes: 256 * 1024 * 1024, randBytes: 64 * 1024 * 1024, randOps: 2000 };
}

/** Deterministic PRNG (mulberry32) so the random offsets are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Concurrency-limited map preserving order (same shape as vssd's). */
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

function makeWorkerInfo(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    host: "127.0.0.1",
    port: 0,
    hwPort,
    device: "cpu",
    backend: "bench",
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

const GB = 1024 ** 3;

export async function runBench(opts: BenchOptions): Promise<BenchResult[]> {
  const donors = Math.max(1, Math.floor(opts.donors ?? 1));
  const sizes = sizesFor(opts.size);
  const target = VSSD_TARGETS.primaryMetrics;
  const targetName = VSSD_TARGETS.primary;
  const tmp = mkdtempSync(join(tmpdir(), "vssd-bench-"));
  const manifestPath = join(tmp, "manifest.json");
  const servers: HwNodeServer[] = [];
  const infos: WorkerInfo[] = [];
  try {
    for (let i = 0; i < donors; i++) {
      const node = new SsdNode({ dir: join(tmp, `donor-${i}`) });
      const server = await createHwNodeServer({ routeTables: [node.routes()] });
      servers.push(server);
      infos.push(makeWorkerInfo(`bench-donor-${i}`, server.port));
    }
    const provider = { listNodes: () => [...infos] };
    const common = `donors=${donors}, disk-backed nodes, chunk 1 MiB, replicas=${REPLICAS}, binary chunk transport, 8 chunk fetches/stores in flight`;

    // --- sequential read / write -------------------------------------------
    const seqData = randomBytes(sizes.seqBytes);
    const seqKey = "bench/seq.bin";
    const writer = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: SEQ_CHUNK_BYTES,
      defaultReplicas: REPLICAS,
      manifestPath,
      hotCacheChunks: 64,
    });
    const writeMs: number[] = [];
    for (let run = 0; run < 2; run++) {
      const start = performance.now();
      await writer.putObject(seqKey, seqData, { replicas: REPLICAS });
      writeMs.push(performance.now() - start);
    }
    const writeGBps = sizes.seqBytes / (median(writeMs) / 1000) / GB;

    // Fresh coordinator => cold hot cache, no vRAM tier. Two timed reads.
    const readMs: number[] = [];
    for (let run = 0; run < 2; run++) {
      const reader = new VirtualSsd({
        nodes: provider,
        chunkSizeBytes: SEQ_CHUNK_BYTES,
        defaultReplicas: REPLICAS,
        manifestPath,
        hotCacheChunks: 64,
      });
      const start = performance.now();
      const back = await reader.getObject(seqKey);
      readMs.push(performance.now() - start);
      if (!Buffer.from(back).equals(seqData)) {
        throw new Error("vSSD bench: sequential readback bytes differ from written data");
      }
    }
    const readGBps = sizes.seqBytes / (median(readMs) / 1000) / GB;

    // --- random 4 KiB read IOPS ---------------------------------------------
    const randData = randomBytes(sizes.randBytes);
    const randKey = "bench/rand.bin";
    const randWriter = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: RAND_CHUNK_BYTES,
      defaultReplicas: REPLICAS,
      manifestPath,
      hotCacheChunks: 4,
    });
    await randWriter.putObject(randKey, randData, { replicas: REPLICAS });
    // Fresh coordinator with a tiny hot cache so random reads go to disk.
    const randReader = new VirtualSsd({
      nodes: provider,
      chunkSizeBytes: RAND_CHUNK_BYTES,
      defaultReplicas: REPLICAS,
      manifestPath,
      hotCacheChunks: 4,
    });
    const rand = mulberry32(0x9e3779b9);
    const maxOffset = sizes.randBytes - RAND_CHUNK_BYTES;
    const offsets: number[] = [];
    for (let i = 0; i < sizes.randOps; i++) {
      offsets.push(Math.floor(rand() * (maxOffset / RAND_CHUNK_BYTES)) * RAND_CHUNK_BYTES);
    }
    const start = performance.now();
    const slices = await mapLimit(offsets, RAND_CONCURRENCY, (off) =>
      randReader.getRange(randKey, off, RAND_CHUNK_BYTES),
    );
    const randSecs = (performance.now() - start) / 1000;
    for (let i = 0; i < offsets.length; i++) {
      if (!Buffer.from(slices[i]).equals(Buffer.from(randData.subarray(offsets[i], offsets[i] + RAND_CHUNK_BYTES)))) {
        throw new Error(`vSSD bench: random readback differs at offset ${offsets[i]}`);
      }
    }
    const randIops = sizes.randOps / randSecs;

    const storedGiB = (sizes.seqBytes + sizes.randBytes) / GB;
    const results: BenchResult[] = [
      makeResult({
        device: "vssd",
        metric: "sequential read",
        measured: readGBps,
        unit: "GB/s",
        target: target["seqReadGBps"] ?? 28,
        targetName,
        donors,
        notes: `timed getObject of ${(sizes.seqBytes / 1024 / 1024).toFixed(0)} MiB, cold cache, median of 2 runs; ${common}`,
      }),
      makeResult({
        device: "vssd",
        metric: "sequential write",
        measured: writeGBps,
        unit: "GB/s",
        target: target["seqWriteGBps"] ?? 14,
        targetName,
        donors,
        notes: `timed putObject of ${(sizes.seqBytes / 1024 / 1024).toFixed(0)} MiB, median of 2 runs; ${common}`,
      }),
      makeResult({
        device: "vssd",
        metric: "random 4 KiB read",
        measured: randIops,
        unit: "IOPS",
        target: target["randomReadIops"] ?? 5_500_000,
        targetName,
        donors,
        notes: `${sizes.randOps} seeded-random 4 KiB getRange ops, ${RAND_CONCURRENCY} in flight, 4 KiB chunks, cold-ish cache; one HTTP round trip per op, so this is latency-bound, not disk-bound`,
      }),
      makeResult({
        device: "vssd",
        metric: "capacity stored",
        measured: storedGiB,
        unit: "GiB",
        target: storedGiB,
        targetName: "informational — no capacity target",
        donors,
        notes: `Informational: GiB stored in this run (${(sizes.seqBytes / 1024 / 1024).toFixed(0)} MiB seq + ${(sizes.randBytes / 1024 / 1024).toFixed(0)} MiB random objects). Pool capacity is bounded by donor disks and was not probed.`,
      }),
    ];

    if (opts.jsonOut) {
      writeFileSync(opts.jsonOut, JSON.stringify(results, null, 2), "utf8");
    }
    return results;
  } finally {
    for (const s of servers) await s.close().catch(() => undefined);
    rmSync(tmp, { recursive: true, force: true });
  }
}

// --- standalone main: scaling table over 1..4 donors ------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const flag = (name: string): string | undefined => {
    const i = args.indexOf(name);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const onlyDonors = flag("--donors") ? Number(flag("--donors")) : undefined;
  const size = flag("--size") ?? "large";
  const jsonOut = flag("--json-out");
  const donorCounts =
    onlyDonors && Number.isFinite(onlyDonors) && onlyDonors >= 1
      ? [Math.floor(onlyDonors)]
      : [1, 2, 3, 4];
  const all: BenchResult[] = [];
  for (const donors of donorCounts) {
    console.log(`\n--- vSSD bench: ${donors} donor(s), size=${size} ---`);
    const results = await runBench({ donors, size });
    console.log(formatResultsTable(results));
    all.push(...results);
  }
  if (jsonOut) {
    writeFileSync(jsonOut, JSON.stringify(all, null, 2), "utf8");
    console.log(`\nwrote ${jsonOut}`);
  }
}

const isMain = process.argv[1] === fileURLToPath(import.meta.url);
if (isMain) {
  main().catch((err: unknown) => {
    console.error(`bench-vssd failed: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
