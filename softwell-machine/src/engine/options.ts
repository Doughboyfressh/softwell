/**
 * src/engine/options.ts — EngineOptions: defaults, validation, parsing.
 *
 * The novel paths (landmark/sparse attention, SW-code weights/KV,
 * speculative/Jacobi decoding) are all selected through one options
 * object, defined in src/contracts.ts. Every field defaults to the
 * baseline, so omitting options — or passing an empty object — reproduces
 * exactly the v1 behaviour. Nothing here changes any default.
 */

import type {
  AttentionMode,
  DecodingMode,
  EngineOptions,
  KvFormat,
  WeightFormat,
} from "../contracts.ts";

export const DEFAULT_ATTENTION: AttentionMode = "exact";
export const DEFAULT_WEIGHT_FORMAT: WeightFormat = "fp32";
export const DEFAULT_KV_FORMAT: KvFormat = "fp32";
export const DEFAULT_DECODING: DecodingMode = "autoregressive";
export const DEFAULT_DRAFT_LENGTH = 4;

export const ATTENTION_MODES: AttentionMode[] = ["exact", "landmark", "sparse"];
export const WEIGHT_FORMATS: WeightFormat[] = ["fp32", "sw2", "sw3", "sw4"];
export const KV_FORMATS: KvFormat[] = ["fp32", "sw2", "sw3", "sw4"];
export const DECODING_MODES: DecodingMode[] = ["autoregressive", "speculative", "jacobi"];

/** Fill in every default. Pure: never mutates its input. */
export function resolveEngineOptions(options: EngineOptions = {}): Required<EngineOptions> {
  return {
    attention: options.attention ?? DEFAULT_ATTENTION,
    weightFormat: options.weightFormat ?? DEFAULT_WEIGHT_FORMAT,
    kvFormat: options.kvFormat ?? DEFAULT_KV_FORMAT,
    decoding: options.decoding ?? DEFAULT_DECODING,
    draftLength: options.draftLength ?? DEFAULT_DRAFT_LENGTH,
  };
}

/** True when the options select only baseline paths (nothing novel). */
export function isBaselineOptions(options: EngineOptions = {}): boolean {
  const r = resolveEngineOptions(options);
  return (
    r.attention === "exact" &&
    r.weightFormat === "fp32" &&
    r.kvFormat === "fp32" &&
    r.decoding === "autoregressive"
  );
}

function checkEnum<T extends string>(value: unknown, allowed: T[], name: string): T {
  if (typeof value !== "string" || !(allowed as string[]).includes(value)) {
    throw new Error(
      `invalid engine option ${name}: ${JSON.stringify(value)} (expected one of ${allowed.join(" | ")})`,
    );
  }
  return value as T;
}

/** Validate a (possibly partial) EngineOptions; throws on any bad value. */
export function validateEngineOptions(options: EngineOptions): void {
  if (options.attention !== undefined) checkEnum(options.attention, ATTENTION_MODES, "attention");
  if (options.weightFormat !== undefined) checkEnum(options.weightFormat, WEIGHT_FORMATS, "weightFormat");
  if (options.kvFormat !== undefined) checkEnum(options.kvFormat, KV_FORMATS, "kvFormat");
  if (options.decoding !== undefined) checkEnum(options.decoding, DECODING_MODES, "decoding");
  if (options.draftLength !== undefined) {
    if (!Number.isInteger(options.draftLength) || options.draftLength < 1) {
      throw new Error(`invalid engine option draftLength: ${options.draftLength} (expected a positive integer)`);
    }
  }
}

/**
 * Parse `--attention`, `--weights`, `--kv`, `--decoding`, `--draft-length`
 * out of an argv array (values `--flag value` or `--flag=value`). Unknown
 * flags are left alone — this parser only consumes the five it knows, so
 * callers must still reject anything else. Returns the options plus the
 * argv entries that were not consumed.
 */
export function parseEngineFlags(argv: string[]): { options: EngineOptions; rest: string[] } {
  const options: EngineOptions = {};
  const rest: string[] = [];
  const takeValue = (argv: string[], i: number, flag: string): { value: string; next: number } => {
    const inline = argv[i]?.split("=", 2)[1];
    if (inline !== undefined) return { value: inline, next: i + 1 };
    const next = argv[i + 1];
    if (next === undefined) throw new Error(`missing value for ${flag}`);
    return { value: next, next: i + 2 };
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    const [flag] = arg.split("=", 2);
    if (flag === "--attention") {
      const { value, next } = takeValue(argv, i, flag);
      options.attention = checkEnum(value, ATTENTION_MODES, "--attention");
      i = next - 1;
    } else if (flag === "--weights") {
      const { value, next } = takeValue(argv, i, flag);
      options.weightFormat = checkEnum(value, WEIGHT_FORMATS, "--weights");
      i = next - 1;
    } else if (flag === "--kv") {
      const { value, next } = takeValue(argv, i, flag);
      options.kvFormat = checkEnum(value, KV_FORMATS, "--kv");
      i = next - 1;
    } else if (flag === "--decoding") {
      const { value, next } = takeValue(argv, i, flag);
      options.decoding = checkEnum(value, DECODING_MODES, "--decoding");
      i = next - 1;
    } else if (flag === "--draft-length") {
      const { value, next } = takeValue(argv, i, flag);
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) {
        throw new Error(`invalid --draft-length: ${value} (expected a positive integer)`);
      }
      options.draftLength = n;
      i = next - 1;
    } else {
      rest.push(arg);
    }
  }
  return { options, rest };
}

/** Render the resolved options back as CLI flags (for worker spawn + logs). */
export function engineOptionsToFlags(options: EngineOptions): string[] {
  const r = resolveEngineOptions(options);
  return [
    "--attention",
    r.attention,
    "--weights",
    r.weightFormat,
    "--kv",
    r.kvFormat,
    "--decoding",
    r.decoding,
    "--draft-length",
    String(r.draftLength),
  ];
}

/** One-line summary for logs, e.g. "attention=landmark weights=sw3 kv=fp32 decoding=autoregressive". */
export function describeEngineOptions(options: EngineOptions): string {
  const r = resolveEngineOptions(options);
  return (
    `attention=${r.attention} weights=${r.weightFormat} kv=${r.kvFormat} ` +
    `decoding=${r.decoding} draft-length=${r.draftLength}`
  );
}
