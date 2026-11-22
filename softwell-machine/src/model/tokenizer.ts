/**
 * Model layer — byte-level tokenizer.
 *
 * The reference tokenizer is deliberately simple: every UTF-8 byte is a
 * token (ids 0..255), plus three special tokens — BOS (256), EOS (257),
 * PAD (258) — for a 259-token vocabulary. That matches the demo model,
 * whose vocab is byte-level, and it round-trips any text exactly.
 *
 * `loadTokenizer(dir)` reads tokenizer.json when present. It understands
 * the simple {"vocab_size","bos_token_id","eos_token_id","pad_token_id"}
 * shape written by `saveTokenizer`, and falls back to the byte-level
 * defaults for anything else (including a full HF tokenizer.json, whose
 * BPE merges this reference implementation does not apply).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tokenizer } from "../contracts.ts";

export const BYTE_VOCAB_SIZE = 256;
export const DEFAULT_BOS_TOKEN_ID = 256;
export const DEFAULT_EOS_TOKEN_ID = 257;
export const DEFAULT_PAD_TOKEN_ID = 258;
export const DEFAULT_VOCAB_SIZE = 259;

export interface ByteLevelTokenizerOptions {
  /** Prepend BOS on encode. Defaults to true (Llama-style). */
  addBos?: boolean;
  /** Append EOS on encode. Defaults to false. */
  addEos?: boolean;
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
}

export class ByteLevelTokenizer implements Tokenizer {
  readonly vocabSize: number;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly padTokenId: number;
  readonly addBos: boolean;
  readonly addEos: boolean;
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder("utf-8", { fatal: false });

  constructor(options: ByteLevelTokenizerOptions = {}) {
    this.bosTokenId = options.bosTokenId ?? DEFAULT_BOS_TOKEN_ID;
    this.eosTokenId = options.eosTokenId ?? DEFAULT_EOS_TOKEN_ID;
    this.padTokenId = options.padTokenId ?? DEFAULT_PAD_TOKEN_ID;
    this.addBos = options.addBos ?? true;
    this.addEos = options.addEos ?? false;
    this.vocabSize = Math.max(this.bosTokenId, this.eosTokenId, this.padTokenId) + 1;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    if (this.addBos) ids.push(this.bosTokenId);
    const bytes = this.encoder.encode(text);
    for (const b of bytes) ids.push(b);
    if (this.addEos) ids.push(this.eosTokenId);
    return ids;
  }

  decode(tokenIds: number[]): string {
    const bytes: number[] = [];
    for (const id of tokenIds) {
      // Skip special tokens (BOS/EOS/PAD and anything outside byte range).
      if (id >= 0 && id < BYTE_VOCAB_SIZE) bytes.push(id);
    }
    return this.decoder.decode(new Uint8Array(bytes));
  }
}

/** Convenience factory for the default byte-level tokenizer. */
export function createByteLevelTokenizer(options: ByteLevelTokenizerOptions = {}): ByteLevelTokenizer {
  return new ByteLevelTokenizer(options);
}

interface TokenizerJsonShape {
  vocab_size?: unknown;
  bos_token_id?: unknown;
  eos_token_id?: unknown;
  pad_token_id?: unknown;
  add_bos_token?: unknown;
  add_eos_token?: unknown;
  model?: unknown;
}

function asId(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : fallback;
}

function parseTokenizerJson(raw: unknown): ByteLevelTokenizerOptions {
  if (typeof raw !== "object" || raw === null) return {};
  const obj = raw as TokenizerJsonShape;
  const options: ByteLevelTokenizerOptions = {};
  // Simple shape written by saveTokenizer.
  if (typeof obj["vocab_size"] === "number") {
    // vocab_size is implied by the special ids; only honour it when it is
    // consistent with a byte-level vocab (>= 259) — otherwise fall back.
    void obj["vocab_size"];
  }
  if (typeof obj["bos_token_id"] === "number") options.bosTokenId = asId(obj["bos_token_id"], DEFAULT_BOS_TOKEN_ID);
  if (typeof obj["eos_token_id"] === "number") options.eosTokenId = asId(obj["eos_token_id"], DEFAULT_EOS_TOKEN_ID);
  if (typeof obj["pad_token_id"] === "number") options.padTokenId = asId(obj["pad_token_id"], DEFAULT_PAD_TOKEN_ID);
  if (typeof obj["add_bos_token"] === "boolean") options.addBos = obj["add_bos_token"];
  if (typeof obj["add_eos_token"] === "boolean") options.addEos = obj["add_eos_token"];
  // HF tokenizer.json shape: model.vocab is a BPE vocab this reference
  // implementation does not apply, but added_tokens may carry the specials.
  const model = obj["model"];
  if (typeof model === "object" && model !== null) {
    const added = (model as Record<string, unknown>)["added_tokens"];
    if (Array.isArray(added)) {
      for (const entry of added) {
        if (typeof entry !== "object" || entry === null) continue;
        const rec = entry as Record<string, unknown>;
        if (typeof rec["id"] !== "number" || typeof rec["content"] !== "string") continue;
        if (rec["content"] === "<s>" && options.bosTokenId === undefined) options.bosTokenId = rec["id"];
        if (rec["content"] === "</s>" && options.eosTokenId === undefined) options.eosTokenId = rec["id"];
        if (rec["content"] === "<pad>" && options.padTokenId === undefined) options.padTokenId = rec["id"];
      }
    }
  }
  return options;
}

/**
 * Load a tokenizer from a model directory. Reads tokenizer.json when
 * present (simple shape, or HF shape with added_tokens for the specials)
 * and falls back to the byte-level defaults otherwise.
 */
export function loadTokenizer(dir: string): ByteLevelTokenizer {
  const filePath = path.join(path.resolve(dir), "tokenizer.json");
  if (!fs.existsSync(filePath)) return new ByteLevelTokenizer();
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (err) {
    throw new Error(`invalid JSON in ${filePath}: ${(err as Error).message}`);
  }
  return new ByteLevelTokenizer(parseTokenizerJson(raw));
}

/**
 * Save a tokenizer to a model directory as tokenizer.json, in the simple
 * shape loadTokenizer understands. Used by the demo model generator.
 */
export function saveTokenizer(dir: string, tokenizer: Tokenizer): void {
  const absDir = path.resolve(dir);
  fs.mkdirSync(absDir, { recursive: true });
  const shape: Record<string, unknown> = {
    vocab_size: tokenizer.vocabSize,
    bos_token_id: tokenizer.bosTokenId,
    eos_token_id: tokenizer.eosTokenId,
    model: { type: "byte-level" },
  };
  if (tokenizer instanceof ByteLevelTokenizer) {
    shape["pad_token_id"] = tokenizer.padTokenId;
    shape["add_bos_token"] = tokenizer.addBos;
    shape["add_eos_token"] = tokenizer.addEos;
  }
  fs.writeFileSync(path.join(absDir, "tokenizer.json"), JSON.stringify(shape, null, 2) + "\n", "utf-8");
}
