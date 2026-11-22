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
