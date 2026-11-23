/**
 * bench/types.ts — the shared benchmark result contract.
 *
 * Every bench module (bench-vgpu, bench-vcpu, bench-vram, bench-vssd)
 * returns BenchResult[] in exactly this shape. The runner
 * (src/bench/run.ts) only prints and compares; it never re-measures.
 *
 * RULE: `measured` must come from a real timed execution on this
 * machine. Never hardcode a passing number. If a target cannot be
 * beaten on the hardware at hand, report the honest measured value
 * and the donor count that WOULD beat it in `donorsToBeat`.
 */

export interface BenchResult {
  /** Which virtual device was measured. */
  device: "vgpu" | "vcpu" | "vram" | "vssd" | "vnpu";
  /** What was measured, e.g. "matmul TFLOPS (FP32, 2048^3)". */
  metric: string;
  /** Aggregate measured value across `donors` donors. */
  measured: number;
  /** Unit of `measured`, `target`, and `perDonor`. */
  unit: string;
  /** Enterprise target in the same unit. */
  target: number;
  /** Human name of the target hardware, e.g. "NVIDIA Rubin R100". */
  targetName: string;
  /** Number of donors the measurement ran across. */
  donors: number;
  /** Measured value per donor (measured / donors). */
  perDonor: number;
  /** Donors needed to beat the target: ceil(target / perDonor). */
  donorsToBeat: number;
  /** PASS when measured >= target, else FAIL. Computed, never asserted. */
  verdict: "PASS" | "FAIL";
  /** One line of honest context (hardware, method, caveat). */
  notes: string;
}

export interface BenchOptions {
  /** Donor count for the scaling measurement (default 1). */
  donors: number;
  /** Optional workload size hint, e.g. "small" | "large". */
  size?: string;
  /** Write machine-readable JSON to this path when set. */
  jsonOut?: string;
}

export interface BenchModule {
  name: string;
  run(opts: BenchOptions): Promise<BenchResult[]>;
}

/** Build a BenchResult from a measured aggregate; verdict is computed. */
export function makeResult(
  input: Omit<BenchResult, "perDonor" | "donorsToBeat" | "verdict">,
): BenchResult {
  const perDonor = input.donors > 0 ? input.measured / input.donors : input.measured;
  const donorsToBeat =
    perDonor > 0 ? Math.max(1, Math.ceil(input.target / perDonor)) : Number.POSITIVE_INFINITY;
  return {
    ...input,
    perDonor,
    donorsToBeat,
    verdict: input.measured >= input.target ? "PASS" : "FAIL",
  };
}

/** Median of a sample (robust to one slow first run). */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/** Time an async function in ms (performance.now, sub-ms precision). */
export async function timeMs(fn: () => Promise<void>): Promise<number> {
  const start = performance.now();
  await fn();
  return performance.now() - start;
}

/** Render benchmark results as a plain-text table (shared by the runner
 *  and by each module's standalone main()). */
export function formatResultsTable(results: BenchResult[]): string {
  const lines: string[] = [];
  const header = "DEVICE  METRIC                                    MEASURED      TARGET        MARGIN    DONORS  TO-BEAT  VERDICT";
  lines.push(header);
  lines.push("-".repeat(header.length));
  for (const r of results) {
    const margin =
      r.target > 0 ? `${((r.measured / r.target) * 100).toFixed(1)}%` : "n/a";
    const toBeat =
      r.donorsToBeat === Number.POSITIVE_INFINITY
        ? "inf"
        : String(r.donorsToBeat);
    lines.push(
      `${r.device.padEnd(7)} ${r.metric.padEnd(41)} ${String(r.measured.toFixed(3)).padEnd(13)} ${String(r.target).padEnd(13)} ${margin.padEnd(9)} ${String(r.donors).padEnd(7)} ${toBeat.padEnd(8)} ${r.verdict}`,
    );
    lines.push(`        target: ${r.targetName} | per-donor: ${r.perDonor.toFixed(3)} ${r.unit} | ${r.notes}`);
  }
  return lines.join("\n");
}
