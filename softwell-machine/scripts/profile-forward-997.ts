/**
 * profile-forward-997.ts — one long-context characterization run on the
 * current code: prefill ~997 tokens (fox sentence x100), then 8 decode
 * forwards. No before/after claim — characterizes the kv~997 regime
 * (GQA + cache.read share) and checks end-to-end correctness there.
 * Usage: node scripts/profile-forward-997.ts
 */
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { forwardTokenRows } from "../src/engine/novel-decoding.ts";
import { BlockPagedKVCache } from "../src/scheduler/kvcache.ts";
import type { ShardAssignment } from "../src/contracts.ts";

function median(xs: number[]): number { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }
const backend = createReferenceBackend();
const manifest = loadManifest("./models/smollm-135m", "smollm-135m");
const tokenizer = loadTokenizer("./models/smollm-135m");
const loader = openWeightLoader("./models/smollm-135m", "smollm-135m");
const cfg = manifest.config;
const assignment: ShardAssignment = {
  workerId: "long997", stage: 0, layerStart: 0, layerEnd: cfg.numHiddenLayers,
  hasEmbedding: true, hasLmHead: true, tensorParallelRank: 0, tensorParallelSize: 1,
  experts: [], estimatedMemoryBytes: 0,
};
const executor = backend.createStageExecutor(manifest, assignment);
await executor.load(loader);
const text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
const ids = tokenizer.encode(text);
console.log(`long997: prompt chars=${text.length} tokens=${ids.length} loadavg=${JSON.stringify((await import("node:os")).loadavg())}`);
const cache = new BlockPagedKVCache(16, 4096, cfg.numKeyValueHeads, cfg.headDim);
const t0 = performance.now();
await forwardTokenRows(executor, cache, "long997", ids, ids.map((_, i) => i), true);
console.log(`long997: prefill ${ids.length} tok: ${(performance.now() - t0).toFixed(0)}ms = ${((performance.now() - t0) / ids.length).toFixed(1)}ms/token`);
const ms: number[] = [];
let tok = ids[ids.length - 1] as number;
const tokens: number[] = [];
for (let i = 0; i < 8; i++) {
  const t1 = performance.now();
  const out = await forwardTokenRows(executor, cache, "long997", [tok], [ids.length + i], false);
  ms.push(performance.now() - t1);
  const logits = out.data.subarray(0, cfg.vocabSize);
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  tok = best;
  tokens.push(tok);
}
await cache.close();
console.log(`long997: decode ms [${ms.map((v) => v.toFixed(0)).join(", ")}] median=${median(ms).toFixed(1)}ms -> ${(1000 / median(ms)).toFixed(2)} tok/s tokens=${JSON.stringify(tokens)} text=${JSON.stringify(tokenizer.decode(tokens))}`);
