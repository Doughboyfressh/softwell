/** In-process generation proof for SmolLM-135M on the reference backend. */
import { loadManifest, openWeightLoader } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";
import { createReferenceBackend } from "../src/engine/backend.ts";
import { sampleLogits } from "../src/engine/sampling.ts";
import type { ForwardContext, PagedKVCache, ShardAssignment } from "../src/contracts.ts";

class StubCache implements PagedKVCache {
  readonly blockSize = 16;
  private store = new Map<string, { position: number; key: Float32Array; value: Float32Array }[]>();
  async allocate(_seqId: string, _layer: number, _numTokens: number): Promise<number[]> { return [0]; }
  async append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): Promise<void> {
    const k = `${seqId}:${layer}`;
    const entries = this.store.get(k) ?? [];
    entries.push({ position, key: Float32Array.from(key), value: Float32Array.from(value) });
    this.store.set(k, entries);
  }
  async read(seqId: string, layer: number, positions: number[]): Promise<{ keys: Float32Array; values: Float32Array }> {
    const entries = this.store.get(`${seqId}:${layer}`) ?? [];
    const headDim = entries.length > 0 ? entries[0]!.key.length : 0;
    const keys = new Float32Array(positions.length * headDim);
    const values = new Float32Array(positions.length * headDim);
    for (let i = 0; i < positions.length; i++) {
      const e = entries.find((x) => x.position === positions[i]);
      if (!e) throw new Error(`stub cache: no entry for position ${positions[i]}`);
      keys.set(e.key, i * headDim);
      values.set(e.value, i * headDim);
    }
    return { keys, values };
  }
  async free(seqId: string): Promise<void> {
    for (const k of [...this.store.keys()]) if (k.startsWith(`${seqId}:`)) this.store.delete(k);
  }
}

const dir = "./models/smollm-135m";
const prompt = process.argv[2] ?? "The capital of France is";
const maxTokens = Number(process.argv[3] ?? 24);

const manifest = loadManifest(dir, "smollm-135m");
const tokenizer = loadTokenizer(dir);
const loader = openWeightLoader(dir, "smollm-135m");
const backend = createReferenceBackend();
const cfg = manifest.config;
const assignment: ShardAssignment = {
  workerId: "gen", stage: 0, layerStart: 0, layerEnd: cfg.numHiddenLayers,
  hasEmbedding: true, hasLmHead: true, tensorParallelRank: 0, tensorParallelSize: 1,
  experts: [], estimatedMemoryBytes: 0,
};
const executor = backend.createStageExecutor(manifest, assignment);
const tLoad0 = performance.now();
await executor.load(loader);
console.log(`loaded ${executor.memoryBytes()} bytes of weights in ${((performance.now() - tLoad0) / 1000).toFixed(1)}s`);

const promptIds = tokenizer.encode(prompt);
console.log(`prompt: ${JSON.stringify(prompt)} -> ids [${promptIds.join(", ")}]`);
const cache = new StubCache();
const t0 = performance.now();
const embedded = executor.embed(promptIds);
const logits = await executor.forward(embedded, {
  seqIds: promptIds.map(() => "s1"),
  positions: promptIds.map((_, i) => i),
  isPrefill: true,
  cache,
} as ForwardContext);
const ttftMs = performance.now() - t0;
console.log(`prefill: ${promptIds.length} tokens, TTFT ${ttftMs.toFixed(0)} ms`);

const greedy = { temperature: 0, topP: 1, topK: 0, maxTokens };
const generated: number[] = [];
let lastLogits = logits.data.subarray((promptIds.length - 1) * cfg.vocabSize, promptIds.length * cfg.vocabSize);
let decodeMs = 0;
for (let step = 0; step < maxTokens; step++) {
  const tok = sampleLogits(lastLogits, greedy);
  generated.push(tok);
  if (tok === cfg.eosTokenId) { console.log(`(stopped on EOS ${tok} at step ${step})`); break; }
  const emb = executor.embed([tok]);
  const d0 = performance.now();
  const out = await executor.forward(emb, {
    seqIds: ["s1"], positions: [promptIds.length + step], isPrefill: false, cache,
  } as ForwardContext);
  decodeMs += performance.now() - d0;
  lastLogits = out.data;
}
const totalMs = performance.now() - t0;
console.log(`generated ids: [${generated.join(", ")}]`);
console.log(`generated text: ${JSON.stringify(tokenizer.decode(generated))}`);
console.log(`full text: ${JSON.stringify(tokenizer.decode([...promptIds, ...generated]))}`);
console.log(`decode: ${(decodeMs / Math.max(1, generated.length)).toFixed(1)} ms/token, total ${totalMs.toFixed(0)} ms (${(generated.length / (totalMs / 1000)).toFixed(2)} tok/s incl prefill)`);
