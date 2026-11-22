/**
 * bench/report.ts — print benchmark results and the honest-limits block.
 *
 * The runner (src/bench/run.ts) only prints and compares; it never
 * re-measures. Verdicts come from the BenchResult rows themselves
 * (computed by makeResult in types.ts from real measurements).
 *
 * Usage:
 *   printReport(results);                    // table + verdicts + limits
 *   printReport(results, { jsonOut: path }); // also writes raw JSON
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { formatResultsTable } from "./types.ts";
import type { BenchResult } from "./types.ts";

/** Fixed honest-limits text printed with every report. */
export const HONEST_LIMITS = [
  "Honest limits:",
  "  - Aggregate throughput/capacity is the claim. Pooling donors adds",
  "    throughput and capacity; that is what these numbers measure.",
  "  - Single-op latency cannot beat a single B200/Rubin. One op on one",
  "    donor is bounded by that donor's hardware plus the network hop;",
  "    pooling never makes a single operation faster than the fastest",
  "    single device that could run it.",
  "  - Numbers are measured on the machine that ran the bench (see the",
  "    machine line at the top of the run). They are not transferable to",
  "    other hardware, and they say nothing about a donor pool that",
  "    does not exist yet.",
].join("\n");

export interface PrintReportOptions {
  /** When set, also write the raw results JSON to this path. */
  jsonOut?: string;
}

/**
 * Print the results table, a per-device PASS/FAIL summary, and the
 * honest-limits block. Optionally writes the raw results as JSON.
 */
export function printReport(
  results: BenchResult[],
  opts: PrintReportOptions = {},
): void {
  console.log(formatResultsTable(results));
  console.log("");

  // Per-device verdict summary.
  const devices = ["vgpu", "vcpu", "vram", "vssd"] as const;
  let totalPass = 0;
  let totalFail = 0;
  console.log("Verdicts:");
  for (const device of devices) {
    const rows = results.filter((r) => r.device === device);
    if (rows.length === 0) continue;
    const pass = rows.filter((r) => r.verdict === "PASS").length;
    const fail = rows.length - pass;
    totalPass += pass;
    totalFail += fail;
    console.log(`  ${device}: ${pass} PASS, ${fail} FAIL (${rows.length} metric(s))`);
  }
  if (results.length === 0) {
    console.log("  (no results)");
  }
  console.log(`  total: ${totalPass} PASS, ${totalFail} FAIL (${results.length} metric(s))`);
  console.log("");
  console.log(HONEST_LIMITS);

  if (opts.jsonOut) {
    const outPath = path.resolve(opts.jsonOut);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, `${JSON.stringify(results, null, 2)}\n`, "utf8");
    console.log(`\nRaw results written to ${outPath}`);
  }
}
