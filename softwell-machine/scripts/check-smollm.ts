import * as fs from "node:fs";
import { loadManifest } from "../src/model/loader.ts";
import { loadTokenizer, BpeTokenizer } from "../src/model/tokenizer.ts";
import { computeTotals } from "../src/model/manifest.ts";

const dir = "./models/smollm-135m";
const manifest = loadManifest(dir, "smollm-135m");
console.log("config:", JSON.stringify(manifest.config, null, 2));
console.log("totalParams:", manifest.totalParams, "totalBytes:", manifest.totalBytes);
console.log("files:", JSON.stringify(manifest.files));
console.log("metadata:", JSON.stringify(manifest.metadata));
const dtypes = new Map<string, number>();
for (const m of Object.values(manifest.tensors)) dtypes.set(m.dtype, (dtypes.get(m.dtype) ?? 0) + 1);
console.log("dtypes:", JSON.stringify([...dtypes]));
console.log("tensor count:", Object.keys(manifest.tensors).length);
console.log("has lm_head:", "lm_head.weight" in manifest.tensors, "has embed:", "model.embed_tokens.weight" in manifest.tensors);
console.log("embed shape:", JSON.stringify(manifest.tensors["model.embed_tokens.weight"]?.shape));
console.log("layer0 q shape:", JSON.stringify(manifest.tensors["model.layers.0.self_attn.q_proj.weight"]?.shape));
console.log("layer29 down shape:", JSON.stringify(manifest.tensors["model.layers.29.mlp.down_proj.weight"]?.shape));
console.log("norm shape:", JSON.stringify(manifest.tensors["model.norm.weight"]?.shape));

const tok = loadTokenizer(dir);
console.log("tokenizer class:", tok.constructor.name, "vocabSize:", tok.vocabSize, "bos:", tok.bosTokenId, "eos:", tok.eosTokenId);
if (tok instanceof BpeTokenizer) {
  const raw = JSON.parse(fs.readFileSync(dir + "/tokenizer.json", "utf-8"));
  console.log("pre_tokenizer:", JSON.stringify(raw.pre_tokenizer));
  console.log("normalizer:", JSON.stringify(raw.normalizer));
  console.log("model type:", raw.model?.type, "vocab entries:", Object.keys(raw.model?.vocab ?? {}).length, "merges:", (raw.model?.merges ?? []).length);
  console.log("added_tokens:", JSON.stringify(raw.added_tokens));
}
const ids = tok.encode("The capital of France is");
console.log("encode('The capital of France is'):", JSON.stringify(ids));
console.log("decode back:", JSON.stringify(tok.decode(ids)));
console.log("encode('Hello world'):", JSON.stringify(tok.encode("Hello world")));
