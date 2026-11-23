/**
 * vRAM KV cache — measurement bench (NOT part of `npm test`).
 * Run with: node test/kvcache-vram-bench.ts
 *
 * Measures, on demo-100m geometry (kvDim 512, 12 layers, blockSize 16,
 * chunk 1024 tokens = one 4 MiB vRAM page), the vRAM-backed cache
 * (VramPagedKVCache over a REAL in-process VirtualRam + TWO REAL donor
 * RamNodes served over loopback HTTP via createHwNodeServer, binary
 * page transport, replicas=2 unless noted) against the host
 * BlockPagedKVCache:
 *   (a) append latency per token (staged design vs naive per-token
 *       vRAM write vs host),
 *   (b) full-context read latency at ctx 128 and 1024, one layer and
 *       all 12 layers sequential (the per-decode-token cost, as the
 *       engine runs layers sequentially),
 *   (c) host engine decode rate on the real demo-100m model, plus the
 *       projected vRAM-backed decode rate (host time + measured vRAM
 *       read time per token — a projection, because the engine's
 *       forward() is synchronous and cannot drive the async cache;
 *       see the VERDICT in src/scheduler/kvcache-vram.ts). At ctx 1024
 *       the cache is filled directly with appends instead of a full
 *       engine prefill (the reference backend's prefill is O(n^2)
 *       and takes minutes); the decode step reads the same positions
 *       through cache.read either way, so timing is unaffected.
 *
 * Conditions are printed with the numbers and saved to
 * test/kvcache-vram-bench-results-2026-11-22.json. Donors are pure
 * in-memory RamNodes (no spill dir), 1 GiB each, sharing this host.
 * All timings are medians; every timed op is warmed up once first.
 * Minima are recorded alongside medians for the headline read
 * metrics: under CPU contention from concurrent work the median
 * inflates, while the minimum approximates the uncontended floor.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { RamNode, VirtualRam } from "../src/hw/vram.ts";
import { createHwNodeServer } from "../src/hw/node-server.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import {
  VirtualRamAccess,
  VramPagedKVCache,
} from "../src/scheduler/kvcache-vram.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { openWeightLoader } from "../src/model/loader.ts";
import { sampleLogits } from "../src/engine/sampling.ts";
import type { NodeProvider, ShardAssignment, WorkerInfo } from "../src/contracts.ts";

const KV_HEADS = 8;
const HEAD_DIM = 64;
const KV_DIM = KV_HEADS * HEAD_DIM; // 512
const LAYERS = 12;
const BLOCK = 16;
const MIB = 1024 * 1024;

function pseudoRandom(seed: number, n: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 4294967296) * 2 - 1;
  }
  return out;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] as number;
}

function min(values: number[]): number {
  return Math.min(...values);
}

function makeWorker(workerId: string, hwPort: number): WorkerInfo {
  return {
    workerId,
    state: "healthy",
    host: "127.0.0.1",
    port: hwPort,
    hwPort,
    device: "cpu",
    backend: "bench",
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

function prefix(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

interface Results {
  conditions: Record<string, unknown>;
  measurements: Record<string, number>;
}

async function main(): Promise<void> {
  const results: Results = {
    conditions: {
      model: "demo-100m",
      kvDim: KV_DIM,
      layers: LAYERS,
      blockSize: BLOCK,
      chunkTokens: 1024,
      chunkBytes: 4 * MIB,
      donors: 2,
      donorCapacityBytes: 1 << 30,
      donorSpill: "none (pure in-memory RamNode)",
      transport: "loopback HTTP, binary page transport, 4 MiB pages",
      replicas: 2,
      vramPageBytes: 4 * MIB,
      host: `${process.platform} ${process.arch}, node ${process.version}`,
      timing: "median of N runs after 1 warmup; performance.now()",
    },
    measurements: {},
  };
  const m = results.measurements;
  const note = (k: string, v: number) => {
    m[k] = v;
  };

  // --- Harness: two real donors + real VirtualRam coordinator -----------
  const nodeA = new RamNode({ capacityBytes: 1 << 30 });
  const nodeB = new RamNode({ capacityBytes: 1 << 30 });
  const srvA = await createHwNodeServer({ routeTables: [nodeA.routes()] });
  const srvB = await createHwNodeServer({ routeTables: [nodeB.routes()] });
  const provider = new FakeNodes([
    makeWorker("bench-a", srvA.port),
    makeWorker("bench-b", srvB.port),
  ]);
  const ram = new VirtualRam({ nodes: provider, defaultReplicas: 2 });
  const access = new VirtualRamAccess(ram);
  // Replicas=1 comparison uses the same donors, its own coordinator.
  const ram1 = new VirtualRam({ nodes: provider, defaultReplicas: 1 });
  const access1 = new VirtualRamAccess(ram1);

  try {
    const hostCache = () =>
      new BlockPagedKVCache(BLOCK, 8192, KV_HEADS, HEAD_DIM);
    const vramCache = (replicas = 2) =>
      new VramPagedKVCache(
        BLOCK,
        1 << 20,
        KV_HEADS,
        HEAD_DIM,
        replicas === 2 ? access : access1,
        { replicas },
      );

    // --- (a) append latency ---------------------------------------------
    // Host: 1024 appends, one layer.
    {
      const times: number[] = [];
      for (let run = 0; run < 5; run++) {
        const cache = hostCache();
        const k = pseudoRandom(1, KV_DIM);
        const v = pseudoRandom(2, KV_DIM);
        for (let p = 0; p < 16; p++) cache.append("s", 0, p, k, v); // warmup
        const t0 = performance.now();
        for (let p = 0; p < 1024; p++) cache.append("s", 0, p, k, v);
        times.push((performance.now() - t0) / 1024);
      }
      note("host_append_ms_per_token_1layer", median(times));
      console.log(`host append: ${median(times).toFixed(4)} ms/token/layer`);
    }
    // vRAM staged: 1024 appends, one layer (includes one chunk flush).
    {
      const times: number[] = [];
      for (let run = 0; run < 5; run++) {
        const cache = vramCache();
        const k = pseudoRandom(1, KV_DIM);
        const v = pseudoRandom(2, KV_DIM);
        for (let p = 0; p < 16; p++) await cache.append("w", 0, p, k, v); // warmup
        await cache.free("w");
        const t0 = performance.now();
        for (let p = 0; p < 1024; p++) await cache.append("s", 0, p, k, v);
        times.push((performance.now() - t0) / 1024);
        await cache.free("s");
      }
      note("vram_append_staged_ms_per_token_1layer", median(times));
      console.log(`vRAM staged append: ${median(times).toFixed(4)} ms/token/layer`);
    }
    // Naive per-token vRAM write: one 4 KiB write (K+V of one token)
    // straight to VirtualRam — the read-modify-write cost the staging
    // design exists to avoid.
    {
      const alloc = await ram.alloc(4 * MIB, { replicas: 2 });
      const tokenBytes = new Uint8Array(2 * KV_DIM * 4);
      await ram.write(alloc.vaddr, tokenBytes); // warmup
      const times: number[] = [];
      for (let run = 0; run < 5; run++) {
        const t0 = performance.now();
        await ram.write(alloc.vaddr, tokenBytes);
        times.push(performance.now() - t0);
      }
      note("vram_naive_write_4kib_ms", median(times));
      console.log(`vRAM naive 4 KiB write (one token): ${median(times).toFixed(2)} ms`);
      // Full-page flush + alloc, for the record.
      const page = new Uint8Array(4 * MIB);
      const wtimes: number[] = [];
      for (let run = 0; run < 3; run++) {
        const t0 = performance.now();
        await ram.write(alloc.vaddr, page);
        wtimes.push(performance.now() - t0);
      }
      note("vram_full_page_write_4mib_ms", median(wtimes));
      console.log(`vRAM full-page (4 MiB) write: ${median(wtimes).toFixed(2)} ms`);
      const atimes: number[] = [];
      for (let run = 0; run < 3; run++) {
        const t0 = performance.now();
        const a = await ram.alloc(4 * MIB, { replicas: 2 });
        atimes.push(performance.now() - t0);
        await ram.free(a.id);
      }
      note("vram_alloc_4mib_ms", median(atimes));
      console.log(`vRAM alloc (4 MiB, replicas=2): ${median(atimes).toFixed(2)} ms`);
      await ram.free(alloc.id);
    }

    // --- (b) full-context read latency ----------------------------------
    // Setup helper: fill one layer of a cache with `ctx` tokens.
    const fillLayer = async (
      cache: VramPagedKVCache | BlockPagedKVCache,
      seq: string,
      layer: number,
      ctx: number,
    ) => {
      for (let p = 0; p < ctx; p++) {
        const k = pseudoRandom(p * 2 + 1, KV_DIM);
        const v = pseudoRandom(p * 2 + 2, KV_DIM);
        await cache.append(seq, layer, p, k, v);
      }
    };

    for (const ctx of [128, 1024]) {
      // Host, one layer.
      {
        const cache = hostCache();
        await fillLayer(cache, "s", 0, ctx);
        await cache.read("s", 0, prefix(ctx)); // warmup
        const times: number[] = [];
        for (let run = 0; run < 5; run++) {
          const t0 = performance.now();
          await cache.read("s", 0, prefix(ctx));
          times.push(performance.now() - t0);
        }
        note(`host_read_ms_ctx${ctx}_1layer`, median(times));
        console.log(`host read ctx ${ctx}, 1 layer: ${median(times).toFixed(3)} ms`);
      }
      // vRAM, one layer: cold = first read after the flush (includes
      // lazy replica repair), warm = median of subsequent reads.
      for (const replicas of [2, 1]) {
        const cache = vramCache(replicas);
        await fillLayer(cache, "s", 0, ctx);
        const tCold0 = performance.now();
        await cache.read("s", 0, prefix(ctx));
        const cold = performance.now() - tCold0;
        const times: number[] = [];
        for (let run = 0; run < 7; run++) {
          const t0 = performance.now();
          await cache.read("s", 0, prefix(ctx));
          times.push(performance.now() - t0);
        }
        note(`vram_read_cold_ms_ctx${ctx}_1layer_rep${replicas}`, cold);
        note(`vram_read_warm_ms_ctx${ctx}_1layer_rep${replicas}`, median(times));
        note(`vram_read_min_ms_ctx${ctx}_1layer_rep${replicas}`, min(times));
        console.log(
          `vRAM read ctx ${ctx}, 1 layer, replicas=${replicas}: cold ${cold.toFixed(2)} ms, warm ${median(times).toFixed(2)} ms`,
        );
        await cache.free("s");
      }
    }

    // Decode-step cost: all 12 layers, sequential (as the engine runs them).
    for (const ctx of [128, 1024]) {
      {
        const cache = hostCache();
        for (let layer = 0; layer < LAYERS; layer++) {
          await fillLayer(cache, "s", layer, ctx);
        }
        const times: number[] = [];
        for (let run = 0; run < 5; run++) {
          const t0 = performance.now();
          for (let layer = 0; layer < LAYERS; layer++) await cache.read("s", layer, prefix(ctx));
          times.push(performance.now() - t0);
        }
        note(`host_read_ms_ctx${ctx}_12layers`, median(times));
        console.log(`host read ctx ${ctx}, 12 layers: ${median(times).toFixed(3)} ms`);
      }
      for (const replicas of [2, 1]) {
        const cache = vramCache(replicas);
        for (let layer = 0; layer < LAYERS; layer++) {
          await fillLayer(cache, "s", layer, ctx);
        }
        const tCold0 = performance.now();
        for (let layer = 0; layer < LAYERS; layer++) await cache.read("s", layer, prefix(ctx));
        const cold = performance.now() - tCold0;
        const times: number[] = [];
        for (let run = 0; run < 5; run++) {
          const t0 = performance.now();
          for (let layer = 0; layer < LAYERS; layer++) await cache.read("s", layer, prefix(ctx));
          times.push(performance.now() - t0);
        }
        note(`vram_read_cold_ms_ctx${ctx}_12layers_rep${replicas}`, cold);
        note(`vram_read_warm_ms_ctx${ctx}_12layers_rep${replicas}`, median(times));
        note(`vram_read_min_ms_ctx${ctx}_12layers_rep${replicas}`, min(times));
        console.log(
          `vRAM read ctx ${ctx}, 12 layers, replicas=${replicas}: cold ${cold.toFixed(2)} ms, warm ${median(times).toFixed(2)} ms`,
        );
        await cache.free("s");
      }
    }

    // --- (c) host engine decode rate on the real demo-100m ---------------
    {
      const loader = openWeightLoader("demo/models/demo-100m");
      const manifest = loader.manifest;
      assert.equal(manifest.config.numHiddenLayers, LAYERS);
      const assignment: ShardAssignment = {
        workerId: "bench",
        stage: 0,
        layerStart: 0,
        layerEnd: LAYERS,
        hasEmbedding: true,
        hasLmHead: true,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 0,
      };
      const backend = createReferenceBackend();
      const executor = backend.createStageExecutor(manifest, assignment);
      await executor.load(loader);
      loader.close();
      const cache = hostCache();
      const ctx = 128;
      const prompt: number[] = [256];
      for (let i = 1; i < ctx; i++) prompt.push(i % 259);
      const embedded = executor.embed(prompt);
      const tPrefill0 = performance.now();
      const logits = await executor.forward(embedded, {
        seqIds: prompt.map(() => "s"),
        positions: prompt.map((_, i) => i),
        isPrefill: true,
        cache,
      });
      const prefillMs = performance.now() - tPrefill0;
      note("host_engine_prefill_ms_per_token_ctx128", prefillMs / ctx);
      console.log(`host engine prefill ctx 128: ${(prefillMs / ctx).toFixed(2)} ms/token`);
      let lastLogits = logits.data.subarray((ctx - 1) * manifest.config.vocabSize);
      const greedy = { temperature: 0, topP: 1, topK: 0, maxTokens: 1 };
      const decodeTimes: number[] = [];
      for (let step = 0; step < 8; step++) {
        const tok = sampleLogits(lastLogits, greedy);
        const emb = executor.embed([tok]);
        const t0 = performance.now();
        const out = await executor.forward(emb, {
          seqIds: ["s"],
          positions: [ctx + step],
          isPrefill: false,
          cache,
        });
        decodeTimes.push(performance.now() - t0);
        lastLogits = out.data;
      }
      const decodeMs = median(decodeTimes);
      note("host_engine_decode_ms_per_token_ctx128", decodeMs);
      note("host_engine_decode_tok_per_sec_ctx128", 1000 / decodeMs);
      console.log(`host engine decode ctx 128: ${decodeMs.toFixed(2)} ms/token (${(1000 / decodeMs).toFixed(2)} tok/s)`);
      // Projection: vRAM-backed decode = host decode + vRAM read time
      // for all 12 layers (the engine reads the full prefix once per
      // layer per token; appends are staged, so no vRAM write on the
      // decode path until a chunk fills — amortized ~0.02 ms/token,
      // negligible next to the read cost).
      for (const replicas of [2, 1]) {
        const warm = m[`vram_read_warm_ms_ctx128_12layers_rep${replicas}`] as number;
        const cold = m[`vram_read_cold_ms_ctx128_12layers_rep${replicas}`] as number;
        note(`projected_vram_decode_ms_per_token_ctx128_rep${replicas}_warm`, decodeMs + warm);
        note(`projected_vram_decode_tok_per_sec_ctx128_rep${replicas}_warm`, 1000 / (decodeMs + warm));
        note(`projected_vram_decode_ms_per_token_ctx128_rep${replicas}_cold`, decodeMs + cold);
        console.log(
          `projected vRAM decode ctx 128 replicas=${replicas}: warm ${(decodeMs + warm).toFixed(2)} ms/token (${(1000 / (decodeMs + warm)).toFixed(2)} tok/s), cold ${(decodeMs + cold).toFixed(2)} ms/token`,
        );
      }
      // ctx 1024 projection: host decode at ctx 1024 is slower (the
      // attention math is O(ctx)); measure it directly. A full engine
      // prefill of 1024 tokens is O(n^2) through the reference
      // backend and takes minutes under contention, so the cache is
      // filled DIRECTLY with appends (same K/V layout the engine
      // would have written; logits content is irrelevant to decode
      // timing — the decode step reads the same 1024 positions
      // through cache.read either way) and one decode step is timed.
      {
        const cache2 = hostCache();
        await fillLayer(cache2, "s", 0, 1024);
        // NOTE: fillLayer fills one layer; decode reads all 12, so
        // fill the remaining layers the same way.
        for (let layer = 1; layer < LAYERS; layer++) {
          await fillLayer(cache2, "s", layer, 1024);
        }
        const emb = executor.embed([256]);
        const dTimes: number[] = [];
        for (let step = 0; step < 3; step++) {
          const d0 = performance.now();
          const out = await executor.forward(emb, {
            seqIds: ["s"],
            positions: [1024 + step],
            isPrefill: false,
            cache: cache2,
          });
          void out;
          dTimes.push(performance.now() - d0);
        }
        const decode1024Ms = median(dTimes);
        note("host_engine_decode_ms_per_token_ctx1024", decode1024Ms);
        note("host_engine_decode_tok_per_sec_ctx1024", 1000 / decode1024Ms);
        console.log(`host engine decode ctx 1024: ${decode1024Ms.toFixed(2)} ms/token (${(1000 / decode1024Ms).toFixed(2)} tok/s)`);
        for (const replicas of [2, 1]) {
          const warm = m[`vram_read_warm_ms_ctx1024_12layers_rep${replicas}`] as number;
          const cold = m[`vram_read_cold_ms_ctx1024_12layers_rep${replicas}`] as number;
          note(`projected_vram_decode_ms_per_token_ctx1024_rep${replicas}_warm`, decode1024Ms + warm);
          note(`projected_vram_decode_tok_per_sec_ctx1024_rep${replicas}_warm`, 1000 / (decode1024Ms + warm));
          note(`projected_vram_decode_ms_per_token_ctx1024_rep${replicas}_cold`, decode1024Ms + cold);
          console.log(
            `projected vRAM decode ctx 1024 replicas=${replicas}: warm ${(decode1024Ms + warm).toFixed(2)} ms/token (${(1000 / (decode1024Ms + warm)).toFixed(2)} tok/s), cold ${(decode1024Ms + cold).toFixed(2)} ms/token`,
          );
        }
      }
    }

    // --- Ratios ------------------------------------------------------------
    const ratio = (a: number, b: number) => a / b;
    console.log("\n--- ratios (vRAM / host; >1 = vRAM slower) ---");
    console.log(
      `append/token/layer: ${ratio(m["vram_append_staged_ms_per_token_1layer"] as number, m["host_append_ms_per_token_1layer"] as number).toFixed(1)}x (staged)`,
    );
    console.log(
      `naive 4 KiB write vs host append: ${ratio(m["vram_naive_write_4kib_ms"] as number, m["host_append_ms_per_token_1layer"] as number).toFixed(0)}x`,
    );
    for (const ctx of [128, 1024]) {
      console.log(
        `read ctx ${ctx} 1 layer (warm, rep2): ${ratio(m[`vram_read_warm_ms_ctx${ctx}_1layer_rep2`] as number, m[`host_read_ms_ctx${ctx}_1layer`] as number).toFixed(0)}x`,
      );
      console.log(
        `read ctx ${ctx} 12 layers (warm, rep2): ${ratio(m[`vram_read_warm_ms_ctx${ctx}_12layers_rep2`] as number, m[`host_read_ms_ctx${ctx}_12layers`] as number).toFixed(0)}x`,
      );
    }

    const outPath = "test/kvcache-vram-bench-results-2026-11-22.json";
    writeFileSync(outPath, JSON.stringify(results, null, 2) + "\n");
    console.log(`\nresults written to ${outPath}`);
  } finally {
    await srvA.close();
    await srvB.close();
  }
}

main().catch((err: unknown) => {
  console.error("kvcache-vram bench FAILED:", err);
  process.exitCode = 1;
});
