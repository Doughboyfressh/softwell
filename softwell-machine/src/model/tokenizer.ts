/**
 * Model layer — byte-level tokenizer + BPE tokenizer.
 *
 * The reference tokenizer is deliberately simple: every UTF-8 byte is a
 * token (ids 0..255), plus three special tokens — BOS (256), EOS (257),
 * PAD (258) — for a 259-token vocabulary. That matches the demo model,
 * whose vocab is byte-level, and it round-trips any text exactly.
 *
 * `loadTokenizer(dir)` reads tokenizer.json when present. It understands
 * the simple {"vocab_size","bos_token_id","eos_token_id","pad_token_id"}
 * shape written by `saveTokenizer`, and falls back to the byte-level
 * defaults for anything else. When tokenizer.json carries a usable
 * Hugging Face BPE model (non-empty `model.vocab` AND `model.merges`),
 * it returns a `BpeTokenizer` instead: normalizer, pre-tokenizer splits
 * (including the GPT-2-style byte-level regex), BPE merges by rank,
 * added tokens (longest-match first), and a byte-level decoder.
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

// ---------------------------------------------------------------------------
// BPE tokenizer (Hugging Face tokenizer.json)
// ---------------------------------------------------------------------------

/**
 * GPT-2-style byte-level pre-tokenizer regex (the pattern shipped in most
 * HF `tokenizer.json` files as a Split pattern or a ByteLevel
 * pre-tokenizer with `use_regex`). Kept as the default when a file's
 * pre-tokenizer is a ByteLevel pre-tokenizer.
 */
export const GPT2_PRE_TOKENIZER_PATTERN =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/** Fallback split when a file's pre-tokenizer shape is unrecognised:
 * words (with an optional leading space), punctuation runs, and
 * whitespace runs. Covers every character, so nothing is dropped. */
const FALLBACK_SPLIT_PATTERN = " ?[\\p{L}\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+";

/** GPT-2 bytes_to_unicode maps: every byte 0..255 maps to one unicode
 * char; printable bytes map to themselves, the rest to codepoints >= 256. */
function buildByteMaps(): { byteToChar: Map<number, string>; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteToChar = new Map<number, string>();
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCharCode(cs[i] as number);
    byteToChar.set(bs[i] as number, ch);
    charToByte.set(ch, bs[i] as number);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteMaps();

export interface BpeAddedToken {
  id: number;
  content: string;
  special?: boolean;
}

export type BpePreTokenizerKind = "regex" | "byte-level" | "metaspace" | "fallback";

export interface BpeTokenizerOptions {
  vocab: Record<string, number>;
  merges: Array<string | [string, string]>;
  addedTokens?: BpeAddedToken[];
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
  /** Prepend BOS on encode. Defaults to false (HF files say so explicitly). */
  addBos?: boolean;
  /** Append EOS on encode. Defaults to false. */
  addEos?: boolean;
  /** Normalizer: strip surrounding whitespace (left/right) and/or lowercase. */
  stripLeft?: boolean;
  stripRight?: boolean;
  lowercase?: boolean;
  /** Pre-tokenizer shape. "regex" uses `preTokenizerPattern`;
   * "byte-level" uses the GPT-2 pattern; "metaspace" splits on the
   * SentencePiece replacement char; "fallback" splits on whitespace +
   * punctuation boundaries. Defaults to "fallback" when no pattern given. */
  preTokenizerKind?: BpePreTokenizerKind;
  preTokenizerPattern?: string | null;
  /** ByteLevel pre-tokenizers may ask for a leading space. */
  addPrefixSpace?: boolean;
  /** Metaspace replacement character (default "▁"). */
  metaspaceReplacement?: string;
  /** Metaspace prepend_scheme "always" (default false). */
  metaspacePrepend?: boolean;
}

/**
 * A BPE tokenizer parsed from a Hugging Face tokenizer.json: vocab +
 * ordered merges, with normalizer, pre-tokenizer, added tokens and a
 * byte-level decoder. Encode applies the normalizer, splits on added
 * tokens (longest match first, never normalized or BPE'd), pre-tokenizes
 * each segment, then runs the standard merge-rank loop per piece.
 * Decode maps ids back to byte-level strings, byte-decodes them to
 * UTF-8, skips specials, and emits added tokens literally.
 */
export class BpeTokenizer implements Tokenizer {
  readonly vocabSize: number;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly padTokenId: number;
  readonly addBos: boolean;
  readonly addEos: boolean;
  private readonly vocab = new Map<string, number>();
  private readonly idToToken = new Map<number, string>();
  private readonly mergeRanks = new Map<string, number>();
  private readonly addedTokens: BpeAddedToken[] = [];
  private readonly addedByContent = new Map<string, BpeAddedToken>();
  private readonly addedById = new Map<number, BpeAddedToken>();
  private readonly specialIds = new Set<number>();
  private readonly unkTokenId: number | undefined;
  private readonly stripLeft: boolean;
  private readonly stripRight: boolean;
  private readonly lowercase: boolean;
  private readonly preTokenizerKind: BpePreTokenizerKind;
  private readonly preTokenizerPattern: string | null;
  private readonly addPrefixSpace: boolean;
  private readonly metaspaceReplacement: string;
  private readonly metaspacePrepend: boolean;
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder("utf-8", { fatal: false });

  constructor(options: BpeTokenizerOptions) {
    let maxId = -1;
    for (const [token, id] of Object.entries(options.vocab)) {
      this.vocab.set(token, id);
      this.idToToken.set(id, token);
      if (id > maxId) maxId = id;
    }
    this.vocabSize = maxId + 1;
    for (const [a, b] of options.merges.map(normalizeMerge)) {
      const key = `${a}\u0000${b}`;
      if (!this.mergeRanks.has(key)) this.mergeRanks.set(key, this.mergeRanks.size);
    }
    const added = [...(options.addedTokens ?? [])].sort((x, y) => y.content.length - x.content.length);
    for (const token of added) {
      if (token.content.length === 0) continue;
      this.addedTokens.push(token);
      this.addedByContent.set(token.content, token);
      this.addedById.set(token.id, token);
      if (token.special) this.specialIds.add(token.id);
    }
    const bosFromAdded = this.addedByContent.get("<s>")?.id;
    const eosFromAdded = this.addedByContent.get("</s>")?.id;
    const padFromAdded = this.addedByContent.get("<pad>")?.id;
    this.bosTokenId = options.bosTokenId ?? bosFromAdded ?? DEFAULT_BOS_TOKEN_ID;
    this.eosTokenId = options.eosTokenId ?? eosFromAdded ?? DEFAULT_EOS_TOKEN_ID;
    this.padTokenId = options.padTokenId ?? padFromAdded ?? DEFAULT_PAD_TOKEN_ID;
    this.specialIds.add(this.bosTokenId);
    this.specialIds.add(this.eosTokenId);
    this.specialIds.add(this.padTokenId);
    this.unkTokenId = options.unkTokenId;
    this.addBos = options.addBos ?? false;
    this.addEos = options.addEos ?? false;
    this.stripLeft = options.stripLeft ?? false;
    this.stripRight = options.stripRight ?? false;
    this.lowercase = options.lowercase ?? false;
    this.preTokenizerKind = options.preTokenizerKind ?? (options.preTokenizerPattern ? "regex" : "fallback");
    this.preTokenizerPattern = options.preTokenizerPattern ?? null;
    this.addPrefixSpace = options.addPrefixSpace ?? false;
    this.metaspaceReplacement = options.metaspaceReplacement ?? "▁";
    this.metaspacePrepend = options.metaspacePrepend ?? false;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    if (this.addBos) ids.push(this.bosTokenId);
    // Added tokens are split out first, so the normalizer never touches
    // them; Strip then applies only at the whole input's edges (the first
    // segment's left, the last segment's right) while Lowercase applies
    // to every segment. Stripping every segment would eat the spaces
    // around added tokens and break encode/decode round-trips.
    const segments = this.splitAddedTokens(text);
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i] as string | number;
      if (typeof segment === "number") {
        ids.push(segment);
        continue;
      }
      const normalized = this.applyNormalizer(segment, i === 0, i === segments.length - 1);
      for (const piece of this.preTokenize(normalized)) {
        ids.push(...this.encodePiece(piece));
      }
    }
    if (this.addEos) ids.push(this.eosTokenId);
    return ids;
  }

  decode(tokenIds: number[]): string {
    let text = "";
    const bytes: number[] = [];
    const flushBytes = (): void => {
      if (bytes.length > 0) {
        text += this.decoder.decode(new Uint8Array(bytes));
        bytes.length = 0;
      }
    };
    for (const id of tokenIds) {
      if (this.specialIds.has(id)) continue;
      const added = this.addedById.get(id);
      if (added) {
        flushBytes();
        text += added.content;
        continue;
      }
      const token = this.idToToken.get(id);
      if (token === undefined) continue;
      for (const ch of token) {
        const byte = CHAR_TO_BYTE.get(ch);
        if (byte !== undefined) {
          bytes.push(byte);
        } else {
          // A vocab char outside the byte-level map is a literal char
          // (e.g. a plain unicode token); emit it as-is.
          flushBytes();
          text += ch;
        }
      }
    }
    flushBytes();
    return text;
  }

  /** Split text on added tokens (longest match first); numbers are added-token ids. */
  private splitAddedTokens(text: string): Array<string | number> {
    if (this.addedTokens.length === 0) return text.length > 0 ? [text] : [];
    const out: Array<string | number> = [];
    let pending = "";
    let i = 0;
    while (i < text.length) {
      let matched: BpeAddedToken | undefined;
      for (const token of this.addedTokens) {
        if (text.startsWith(token.content, i)) {
          matched = token;
          break;
        }
      }
      if (matched) {
        if (pending.length > 0) {
          out.push(pending);
          pending = "";
        }
        out.push(matched.id);
        i += matched.content.length;
      } else {
        pending += text[i] as string;
        i += 1;
      }
    }
    if (pending.length > 0) out.push(pending);
    return out;
  }

  private applyNormalizer(text: string, isFirst: boolean, isLast: boolean): string {
    let out = text;
    if (this.stripLeft && isFirst) out = out.replace(/^\s+/, "");
    if (this.stripRight && isLast) out = out.replace(/\s+$/, "");
    if (this.lowercase) out = out.toLowerCase();
    return out;
  }

  private preTokenize(text: string): string[] {
    if (text.length === 0) return [];
    if (this.preTokenizerKind === "metaspace") {
      let replaced = text.split(" ").join(this.metaspaceReplacement);
      if (this.metaspacePrepend && !replaced.startsWith(this.metaspaceReplacement)) {
        replaced = this.metaspaceReplacement + replaced;
      }
      return replaced.split(new RegExp(`(?=${escapeRegExp(this.metaspaceReplacement)})`, "u")).filter((p) => p.length > 0);
    }
    let pattern: string | null = null;
    if (this.preTokenizerKind === "regex") pattern = this.preTokenizerPattern;
    else if (this.preTokenizerKind === "byte-level") pattern = GPT2_PRE_TOKENIZER_PATTERN;
    else if (this.preTokenizerKind === "fallback") pattern = FALLBACK_SPLIT_PATTERN;
    if (pattern === null) return [text];
    let input = text;
    if (this.preTokenizerKind === "byte-level" && this.addPrefixSpace && !input.startsWith(" ")) {
      input = " " + input;
    }
    const pieces = matchAll(input, pattern);
    if (pieces === null) return [text];
    // A pattern that does not cover the whole input would silently drop
    // characters; fall back to the covering split instead.
    if (pieces.join("") !== input) {
      const fallback = matchAll(input, FALLBACK_SPLIT_PATTERN);
      return fallback ?? [text];
    }
    return pieces;
  }

  /** BPE-encode one pre-tokenized piece via the merge-rank loop. */
  private encodePiece(piece: string): number[] {
    const bytes = this.encoder.encode(piece);
    let word: string[] = [];
    for (const b of bytes) word.push(BYTE_TO_CHAR.get(b) as string);
    if (word.length === 0) return [];
    while (word.length >= 2) {
      let bestRank = Infinity;
      let bestPair: [string, string] | null = null;
      for (let i = 0; i + 1 < word.length; i++) {
        const a = word[i] as string;
        const b = word[i + 1] as string;
        const rank = this.mergeRanks.get(`${a}\u0000${b}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = [a, b];
        }
      }
      if (bestPair === null) break;
      const [a, b] = bestPair;
      const merged = a + b;
      const next: string[] = [];
      let i = 0;
      while (i < word.length) {
        if (i + 1 < word.length && word[i] === a && word[i + 1] === b) {
          next.push(merged);
          i += 2;
        } else {
          next.push(word[i] as string);
          i += 1;
        }
      }
      word = next;
    }
    const ids: number[] = [];
    for (const symbol of word) {
      const id = this.vocab.get(symbol);
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      // A merged symbol missing from the vocab (should not happen for a
      // well-formed file): fall back to its single-byte chars, then unk.
      let resolved = false;
      for (const ch of symbol) {
        const charId = this.vocab.get(ch);
        if (charId !== undefined) {
          ids.push(charId);
          resolved = true;
        }
      }
      if (!resolved && this.unkTokenId !== undefined) ids.push(this.unkTokenId);
    }
    return ids;
  }
}

function normalizeMerge(merge: string | [string, string]): [string, string] {
  if (Array.isArray(merge)) return merge;
  const idx = merge.indexOf(" ");
  if (idx < 0) return [merge, ""];
  return [merge.slice(0, idx), merge.slice(idx + 1)];
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** All matches of `pattern` in `text`, or null when the pattern is invalid. */
function matchAll(text: string, pattern: string): string[] | null {
  let re: RegExp;
  try {
    re = new RegExp(pattern, "gu");
  } catch {
    return null;
  }
  const out: string[] = [];
  for (const m of text.matchAll(re)) {
    if (m[0].length > 0) out.push(m[0]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Parsing a Hugging Face tokenizer.json into a BpeTokenizer
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseAddedTokens(raw: unknown): BpeAddedToken[] {
  const model = isRecord(raw) ? raw["model"] : undefined;
  const sources: unknown[] = [
    isRecord(raw) ? raw["added_tokens"] : undefined,
    isRecord(model) ? model["added_tokens"] : undefined,
  ];
  for (const source of sources) {
    if (!Array.isArray(source)) continue;
    const out: BpeAddedToken[] = [];
    for (const entry of source) {
      if (!isRecord(entry)) continue;
      if (typeof entry["id"] !== "number" || typeof entry["content"] !== "string") continue;
      out.push({
        id: entry["id"],
        content: entry["content"],
        special: entry["special"] === true,
      });
    }
    if (out.length > 0) return out;
  }
  return [];
}

function parseNormalizer(raw: unknown): { stripLeft: boolean; stripRight: boolean; lowercase: boolean } {
  let stripLeft = false;
  let stripRight = false;
  let lowercase = false;
  const walk = (node: unknown): void => {
    if (!isRecord(node)) return;
    const type = node["type"];
    if (type === "Strip") {
      if (node["strip_left"] !== false) stripLeft = true;
      if (node["strip_right"] !== false) stripRight = true;
    } else if (type === "Lowercase") {
      lowercase = true;
    } else if (type === "Sequence") {
      const inner = node["normalizers"];
      if (Array.isArray(inner)) for (const child of inner) walk(child);
    }
    // Any other normalizer (Replace, NFKC, BertNormalizer, ...) is ignored.
  };
  walk(isRecord(raw) ? raw["normalizer"] : undefined);
  return { stripLeft, stripRight, lowercase };
}

function parsePreTokenizer(raw: unknown): {
  kind: BpePreTokenizerKind;
  pattern: string | null;
  addPrefixSpace: boolean;
  metaspaceReplacement: string;
  metaspacePrepend: boolean;
} {
  const fallback = {
    kind: "fallback" as BpePreTokenizerKind,
    pattern: null as string | null,
    addPrefixSpace: false,
    metaspaceReplacement: "▁",
    metaspacePrepend: false,
  };
  const pt = isRecord(raw) ? raw["pre_tokenizer"] : undefined;
  if (!isRecord(pt)) return fallback;
  const items: unknown[] =
    pt["type"] === "Sequence" && Array.isArray(pt["pretokenizers"]) ? pt["pretokenizers"] : [pt];
  for (const item of items) {
    if (!isRecord(item)) continue;
    if (item["type"] === "Split") {
      const pattern = item["pattern"];
      let text: string | undefined;
      if (typeof pattern === "string") text = pattern;
      else if (isRecord(pattern)) {
        if (typeof pattern["String"] === "string") text = pattern["String"];
        else if (typeof pattern["Regex"] === "string") text = pattern["Regex"];
      }
      if (text !== undefined) {
        return { ...fallback, kind: "regex", pattern: text };
      }
    } else if (item["type"] === "ByteLevel") {
      return { ...fallback, kind: "byte-level", addPrefixSpace: item["add_prefix_space"] === true };
    } else if (item["type"] === "Metaspace") {
      return {
        ...fallback,
        kind: "metaspace",
        metaspaceReplacement: typeof item["replacement"] === "string" ? item["replacement"] : "▁",
        metaspacePrepend: item["prepend_scheme"] === "always",
      };
    }
    // WhitespaceSplit / Whitespace / Punctuation / Digits etc. fall through
    // to the fallback split below.
  }
  return fallback;
}

function asId(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

/**
 * Build a BpeTokenizer from a parsed tokenizer.json object, or null when
 * it does not carry a usable BPE model (model.type is not "BPE", or the
 * vocab or merges are empty). Unknown normalizer / pre-tokenizer shapes
 * are ignored (with the fallback split) rather than failing the parse.
 */
export function bpeTokenizerFromJson(raw: unknown): BpeTokenizer | null {
  if (!isRecord(raw)) return null;
  const model = raw["model"];
  if (!isRecord(model) || model["type"] !== "BPE") return null;
  const vocabRaw = model["vocab"];
  if (!isRecord(vocabRaw) || Object.keys(vocabRaw).length === 0) return null;
  const vocab: Record<string, number> = {};
  for (const [token, id] of Object.entries(vocabRaw)) {
    if (typeof id === "number" && Number.isInteger(id) && id >= 0) vocab[token] = id;
  }
  if (Object.keys(vocab).length === 0) return null;
  const mergesRaw = model["merges"];
  if (!Array.isArray(mergesRaw) || mergesRaw.length === 0) return null;
  const merges: Array<string | [string, string]> = [];
  for (const entry of mergesRaw) {
    if (typeof entry === "string") merges.push(entry);
    else if (
      Array.isArray(entry) &&
      entry.length === 2 &&
      typeof entry[0] === "string" &&
      typeof entry[1] === "string"
    ) {
      merges.push([entry[0], entry[1]]);
    }
  }
  if (merges.length === 0) return null;

  const addedTokens = parseAddedTokens(raw);
  const addedByContent = new Map(addedTokens.map((t) => [t.content, t.id]));
  const normalizer = parseNormalizer(raw);
  const preTokenizer = parsePreTokenizer(raw);
  const unkContent = typeof model["unk_token"] === "string" ? model["unk_token"] : undefined;
  const unkTokenId =
    unkContent !== undefined ? (addedByContent.get(unkContent) ?? vocab[unkContent]) : undefined;
  const bosFromString =
    typeof raw["bos_token"] === "string" ? addedByContent.get(raw["bos_token"]) : undefined;
  const eosFromString =
    typeof raw["eos_token"] === "string" ? addedByContent.get(raw["eos_token"]) : undefined;
  const padFromString =
    typeof raw["pad_token"] === "string" ? addedByContent.get(raw["pad_token"]) : undefined;

  return new BpeTokenizer({
    vocab,
    merges,
    addedTokens,
    bosTokenId: asId(raw["bos_token_id"]) ?? bosFromString,
    eosTokenId: asId(raw["eos_token_id"]) ?? eosFromString,
    padTokenId: asId(raw["pad_token_id"]) ?? padFromString,
    unkTokenId,
    addBos: typeof raw["add_bos_token"] === "boolean" ? raw["add_bos_token"] : undefined,
    addEos: typeof raw["add_eos_token"] === "boolean" ? raw["add_eos_token"] : undefined,
    stripLeft: normalizer.stripLeft,
    stripRight: normalizer.stripRight,
    lowercase: normalizer.lowercase,
    preTokenizerKind: preTokenizer.kind,
    preTokenizerPattern: preTokenizer.pattern,
    addPrefixSpace: preTokenizer.addPrefixSpace,
    metaspaceReplacement: preTokenizer.metaspaceReplacement,
    metaspacePrepend: preTokenizer.metaspacePrepend,
  });
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

function asIdWithFallback(value: unknown, fallback: number): number {
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
  if (typeof obj["bos_token_id"] === "number") options.bosTokenId = asIdWithFallback(obj["bos_token_id"], DEFAULT_BOS_TOKEN_ID);
  if (typeof obj["eos_token_id"] === "number") options.eosTokenId = asIdWithFallback(obj["eos_token_id"], DEFAULT_EOS_TOKEN_ID);
  if (typeof obj["pad_token_id"] === "number") options.padTokenId = asIdWithFallback(obj["pad_token_id"], DEFAULT_PAD_TOKEN_ID);
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
 * present: a usable HF BPE model (non-empty vocab AND merges) yields a
 * BpeTokenizer; otherwise the simple shape, or HF shape with added_tokens
 * for the specials, yields a ByteLevelTokenizer; a missing file falls
 * back to the byte-level defaults.
 */
export function loadTokenizer(dir: string): Tokenizer {
  const filePath = path.join(path.resolve(dir), "tokenizer.json");
  if (!fs.existsSync(filePath)) return new ByteLevelTokenizer();
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (err) {
    throw new Error(`invalid JSON in ${filePath}: ${(err as Error).message}`);
  }
  return bpeTokenizerFromJson(raw) ?? new ByteLevelTokenizer(parseTokenizerJson(raw));
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
