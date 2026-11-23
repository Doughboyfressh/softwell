/**
 * BPE tokenizer test (src/model/tokenizer.ts).
 * Run with: node --test test/tokenizer-bpe.test.ts
 *
 * Builds a tiny synthetic HF tokenizer.json fixture (vocab + merges +
 * GPT-2-style pre-tokenizer + Strip/Lowercase normalizer + added tokens)
 * in a temp dir and checks: merges apply, encode/decode round-trips,
 * added tokens and specials are respected, the normalizer runs, and
 * loadTokenizer picks BPE for the fixture but byte-level for the demo
 * model dir and for a missing file.
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  BpeTokenizer,
  ByteLevelTokenizer,
  loadTokenizer,
} from "../src/model/tokenizer.ts";

const GPT2_PATTERN =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

const VOCAB: Record<string, number> = {
  "<s>": 0,
  "</s>": 1,
  "<pad>": 2,
  h: 3,
  e: 4,
  l: 5,
  o: 6,
  he: 7,
  hel: 8,
  hell: 9,
  hello: 10,
  "Ġ": 11,
  w: 12,
  r: 13,
  d: 14,
  wo: 15,
  wor: 16,
  worl: 17,
  world: 18,
  "Ġworld": 19,
  "Ġw": 20,
  "Ġwo": 21,
  "Ġwor": 22,
  "Ġworl": 23,
  "!": 24,
  ",": 25,
  H: 26,
  He: 27,
  Hel: 28,
  Hell: 29,
  Hello: 30,
  "[TEST]": 31,
};

const MERGES = [
  "h e",
  "he l",
  "hel l",
  "hell o",
  "Ġ w",
  "Ġw o",
  "Ġwo r",
  "Ġwor l",
  "Ġworl d",
  "w o",
  "wo r",
  "wor l",
  "worl d",
  "H e",
  "He l",
  "Hel l",
  "Hell o",
];

const ADDED_TOKENS = [
  { id: 0, content: "<s>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
  { id: 1, content: "</s>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
  { id: 2, content: "<pad>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
  { id: 31, content: "[TEST]", single_word: false, lstrip: false, rstrip: false, normalized: false, special: false },
];

function fixtureJson(preTokenizer: unknown): Record<string, unknown> {
  return {
    version: "1.0",
    added_tokens: ADDED_TOKENS,
    normalizer: {
      type: "Sequence",
      normalizers: [
        { type: "Strip", strip_left: true, strip_right: true },
        { type: "Lowercase" },
      ],
    },
    pre_tokenizer: preTokenizer,
    model: { type: "BPE", vocab: VOCAB, merges: MERGES, unk_token: "<unk>" },
    decoder: { type: "ByteLevel" },
    add_bos_token: true,
    add_eos_token: false,
  };
}

const GPT2_PRE_TOKENIZER = {
  type: "Split",
  pattern: { String: GPT2_PATTERN },
  behavior: "Isolated",
  invert: false,
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-bpe-test-"));
try {
  const fixtureDir = path.join(tmp, "bpe-model");
  fs.mkdirSync(fixtureDir, { recursive: true });
  fs.writeFileSync(
    path.join(fixtureDir, "tokenizer.json"),
    JSON.stringify(fixtureJson(GPT2_PRE_TOKENIZER)),
    "utf-8",
  );

  // --- loadTokenizer picks BPE for the fixture. ---
  const tok = loadTokenizer(fixtureDir);
  assert.ok(tok instanceof BpeTokenizer, "fixture should load as BpeTokenizer");
  assert.equal(tok.vocabSize, 32);
  assert.equal(tok.bosTokenId, 0);
  assert.equal(tok.eosTokenId, 1);

  // --- Merges apply: "hello" is one token, not five byte tokens. ---
  assert.deepEqual(tok.encode("hello"), [0, 10]);
  assert.deepEqual(tok.encode("hello world"), [0, 10, 19]);
  assert.deepEqual(tok.encode("world"), [0, 18]);

  // --- Encode/decode round-trips (lowercase text: the fixture lowercases). ---
  assert.equal(tok.decode(tok.encode("hello world")), "hello world");
  assert.equal(tok.decode(tok.encode("hello, hello!")), "hello, hello!");
  assert.equal(tok.decode(tok.encode("world hello world")), "world hello world");

  // --- Specials: BOS is prepended, EOS/PAD/BOS are skipped on decode. ---
  assert.equal(tok.encode("hello")[0], 0, "BOS first");
  assert.equal(tok.decode([0, 10, 1, 2]), "hello");

  // --- Added tokens: longest-match, never BPE'd, decoded literally. ---
  assert.deepEqual(tok.encode("hello [TEST] world"), [0, 10, 11, 31, 19]);
  assert.equal(tok.decode(tok.encode("hello [TEST] world")), "hello [TEST] world");

  // --- Normalizer: Strip + Lowercase. ---
  assert.deepEqual(tok.encode("  HELLO "), tok.encode("hello"));
  assert.equal(tok.decode(tok.encode("  HELLO ")), "hello");

  // --- Direct construction: no normalizer, capital-H merges, EOS on. ---
  const direct = new BpeTokenizer({
    vocab: VOCAB,
    merges: MERGES,
    addedTokens: [
      { id: 0, content: "<s>", special: true },
      { id: 1, content: "</s>", special: true },
      { id: 2, content: "<pad>", special: true },
    ],
    addBos: false,
    addEos: true,
    preTokenizerKind: "regex",
    preTokenizerPattern: GPT2_PATTERN,
  });
  assert.deepEqual(direct.encode("Hello"), [30, 1]);
  assert.equal(direct.decode(direct.encode("Hello")), "Hello");

  // --- Fallback pre-tokenizer (no pattern) still splits and round-trips. ---
  const fallback = new BpeTokenizer({
    vocab: VOCAB,
    merges: MERGES,
    addedTokens: [{ id: 0, content: "<s>", special: true }],
    addBos: false,
    preTokenizerKind: "fallback",
  });
  assert.deepEqual(fallback.encode("hello world"), [10, 19]);
  assert.equal(fallback.decode(fallback.encode("hello world")), "hello world");

  // --- Unrecognised pre-tokenizer shape falls back instead of failing. ---
  const weirdDir = path.join(tmp, "weird-model");
  fs.mkdirSync(weirdDir, { recursive: true });
  fs.writeFileSync(
    path.join(weirdDir, "tokenizer.json"),
    JSON.stringify(fixtureJson({ type: "SomethingElse" })),
    "utf-8",
  );
  const weird = loadTokenizer(weirdDir);
  assert.ok(weird instanceof BpeTokenizer, "unrecognised pre-tokenizer still loads as BPE");
  assert.equal(weird.decode(weird.encode("hello world")), "hello world");

  // --- GPT-2-style tokenizer.json carries no bos/eos: tokenizer_config.json
  // + config.json supply them (SmolLM-135M shape). Without them the
  // tokenizer falls back to the demo defaults 256/257/258 — which are real
  // vocab tokens here ("q"/"y"/"x"), so decode() would drop them.
  const SMOL_VOCAB: Record<string, number> = {
    "<|endoftext|>": 0,
    h: 1,
    e: 2,
    l: 3,
    o: 4,
    he: 5,
    hel: 6,
    hell: 7,
    hello: 8,
    "Ġ": 9,
    q: 256,
    y: 257,
    x: 258,
  };
  const SMOL_MERGES = ["h e", "he l", "hel l", "hell o"];
  const smolJson = (): Record<string, unknown> => ({
    version: "1.0",
    added_tokens: [
      { id: 0, content: "<|endoftext|>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    ],
    pre_tokenizer: GPT2_PRE_TOKENIZER,
    model: { type: "BPE", vocab: SMOL_VOCAB, merges: SMOL_MERGES, unk_token: "<|endoftext|>" },
    decoder: { type: "ByteLevel" },
  });
  const smolDir = path.join(tmp, "smol-style");
  fs.mkdirSync(smolDir, { recursive: true });
  fs.writeFileSync(path.join(smolDir, "tokenizer.json"), JSON.stringify(smolJson()), "utf-8");
  // bos/eos come from tokenizer_config.json names; pad only from config.json.
  fs.writeFileSync(
    path.join(smolDir, "tokenizer_config.json"),
    JSON.stringify({ bos_token: "<|endoftext|>", eos_token: "<|endoftext|>" }),
    "utf-8",
  );
  fs.writeFileSync(
    path.join(smolDir, "config.json"),
    JSON.stringify({ pad_token_id: 0 }),
    "utf-8",
  );
  const smolTok = loadTokenizer(smolDir);
  assert.ok(smolTok instanceof BpeTokenizer, "smol-style fixture loads as BPE");
  assert.equal(smolTok.bosTokenId, 0, "bos from tokenizer_config.json");
  assert.equal(smolTok.eosTokenId, 0, "eos from tokenizer_config.json");
  assert.equal(smolTok.padTokenId, 0, "pad from config.json");
  assert.equal(smolTok.decode([256, 257, 258]), "qyx", "real tokens survive decode");
  assert.equal(smolTok.decode([0, 8]), "hello", "EOS still skipped on decode");
  assert.deepEqual(smolTok.encode("hello"), [8]);

  // Negative control: same tokenizer.json, no sibling files -> demo
  // defaults 256/257, and the real tokens behind those ids drop on decode.
  // (Pad falls back to EOS here, so 258 is no longer treated as special.)
  const bareDir = path.join(tmp, "smol-bare");
  fs.mkdirSync(bareDir, { recursive: true });
  fs.writeFileSync(path.join(bareDir, "tokenizer.json"), JSON.stringify(smolJson()), "utf-8");
  const bareTok = loadTokenizer(bareDir);
  assert.ok(bareTok instanceof BpeTokenizer, "bare fixture still loads as BPE");
  assert.equal(bareTok.bosTokenId, 256);
  assert.equal(bareTok.eosTokenId, 257);
  assert.equal(bareTok.padTokenId, 257, "pad falls back to EOS when unspecified");
  assert.equal(bareTok.decode([256, 257]), "", "default specials drop real tokens");

  // --- loadTokenizer falls back to byte-level for the demo model + missing files. ---
  const demoDir = fileURLToPath(new URL("../demo/models/demo-10m", import.meta.url));
  const demoTok = loadTokenizer(demoDir);
  assert.ok(demoTok instanceof ByteLevelTokenizer, "demo model stays byte-level");
  assert.ok(!(demoTok instanceof BpeTokenizer));
  assert.equal(demoTok.vocabSize, 259);
  assert.equal(demoTok.decode(demoTok.encode("Hello, softwell!")), "Hello, softwell!");

  const missingTok = loadTokenizer(path.join(tmp, "no-such-dir"));
  assert.ok(missingTok instanceof ByteLevelTokenizer, "missing file falls back to byte-level");
  assert.equal(missingTok.vocabSize, 259);

  // --- A tokenizer.json without merges is not BPE. ---
  const noMergesDir = path.join(tmp, "no-merges");
  fs.mkdirSync(noMergesDir, { recursive: true });
  fs.writeFileSync(
    path.join(noMergesDir, "tokenizer.json"),
    JSON.stringify({ model: { type: "BPE", vocab: VOCAB, merges: [] } }),
    "utf-8",
  );
  assert.ok(loadTokenizer(noMergesDir) instanceof ByteLevelTokenizer, "empty merges -> byte-level");

  console.log("BPE tokenizer test: all checks passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
