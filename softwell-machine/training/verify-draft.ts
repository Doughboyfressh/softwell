/** Verify the exported draft model loads in the engine: manifest + tokenizer. */
import { loadManifest } from "../src/model/loader.ts";
import { loadTokenizer } from "../src/model/tokenizer.ts";

const dir = process.argv[2] ?? "models/draft-smollm";
const manifest = loadManifest(dir);
console.log("manifest id:", manifest.id);
console.log("config:", JSON.stringify(manifest.config));
console.log("tensors:", Object.keys(manifest.tensors).length);
const needed = [
  "model.embed_tokens.weight",
  "model.layers.0.self_attn.q_proj.weight",
  "model.layers.5.mlp.down_proj.weight",
  "model.norm.weight",
  "lm_head.weight",
];
for (const n of needed) {
  const t = manifest.tensors[n];
  console.log(`  ${t ? "OK  " : "MISS"} ${n}${t ? ` ${t.dtype} [${t.shape.join(",")}]` : ""}`);
}
const tok = loadTokenizer(dir);
const ids = tok.encode("Once upon a time there was a little boy named Ben.");
console.log("tokenizer encode:", JSON.stringify(ids));
console.log("tokenizer decode:", JSON.stringify(tok.decode(ids)));
console.log("VERIFY-LOAD OK");
