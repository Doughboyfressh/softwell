/**
 * Smoke test for the bench report printer (src/bench/report.ts).
 * Run with: node test/bench-runner.test.ts  (exit 0 on success)
 *
 * Covers: printReport with synthetic BenchResult fixtures — the table
 * contains PASS and FAIL rows, the per-device verdict summary counts
 * them, the honest-limits block is printed, and --json output is
 * written and parses. No real bench execution happens here; the
 * device modules are never imported, so tests stay fast.
 */

import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { makeResult } from "../src/bench/types.ts";
import type { BenchResult } from "../src/bench/types.ts";
import { HONEST_LIMITS, printReport } from "../src/bench/report.ts";

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "softwell-bench-test-"));

/** Capture everything printReport writes to console.log. */
function capture(fn: () => void): string {
  const lines: string[] = [];
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    lines.push(args.map(String).join(" "));
  };
  try {
    fn();
  } finally {
    console.log = origLog;
  }
  return lines.join("\n");
}

const fixtures: BenchResult[] = [
  makeResult({
    device: "vgpu",
    metric: "matmul TFLOPS (FP32, 2048^3)",
    measured: 1.5,
    unit: "TFLOPS",
    target: 9000,
    targetName: "NVIDIA B200 (Blackwell)",
    donors: 1,
    notes: "synthetic fixture",
  }),
  makeResult({
    device: "vram",
    metric: "capacity (GB)",
    measured: 512,
    unit: "GB",
    target: 192,
    targetName: "HBM3e (NVIDIA B200)",
    donors: 1,
    notes: "synthetic fixture",
  }),
  makeResult({
    device: "vram",
    metric: "bandwidth (GB/s)",
    measured: 12,
    unit: "GB/s",
    target: 8000,
    targetName: "HBM3e (NVIDIA B200)",
    donors: 1,
    notes: "synthetic fixture",
  }),
];

try {
  // --- 1. Table contains PASS and FAIL rows, verdicts computed. ---
  assert.equal(fixtures[0]?.verdict, "FAIL");
  assert.equal(fixtures[1]?.verdict, "PASS");
  assert.equal(fixtures[2]?.verdict, "FAIL");
  assert.equal(fixtures[1]?.donorsToBeat, 1);
  assert.equal(fixtures[0]?.donorsToBeat, 6000);

  const out = capture(() => printReport(fixtures));
  assert.ok(out.includes("vgpu"), "table lists the vgpu row");
  assert.ok(out.includes("FAIL"), "table contains a FAIL row");
  assert.ok(out.includes("PASS"), "table contains a PASS row");
  assert.ok(out.includes("matmul TFLOPS"), "table shows the metric name");
  assert.ok(out.includes("NVIDIA B200"), "table shows the target name");

  // --- 2. Per-device verdict summary counts. ---
  assert.ok(out.includes("vgpu: 0 PASS, 1 FAIL"), "vgpu summary counts 0/1");
  assert.ok(out.includes("vram: 1 PASS, 1 FAIL"), "vram summary counts 1/1");
  assert.ok(out.includes("total: 1 PASS, 2 FAIL"), "total summary counts 1/2");

  // --- 3. Honest-limits block is printed verbatim. ---
  assert.ok(out.includes(HONEST_LIMITS), "honest-limits block is printed");
  assert.ok(
    out.includes("Single-op latency cannot beat a single B200/Rubin"),
    "honest-limits names the latency limit",
  );

  // --- 4. JSON output is written and parses back to the same rows. ---
  const jsonPath = path.join(tmp, "results.json");
  capture(() => printReport(fixtures, { jsonOut: jsonPath }));
  assert.ok(fs.existsSync(jsonPath), "JSON file was written");
  const parsed = JSON.parse(fs.readFileSync(jsonPath, "utf8")) as BenchResult[];
  assert.equal(parsed.length, 3, "JSON has all three rows");
  assert.equal(parsed[1]?.verdict, "PASS");
  assert.equal(parsed[1]?.measured, 512);
  assert.equal(parsed[0]?.donorsToBeat, 6000);

  // --- 5. Empty results still print a summary, not a crash. ---
  const emptyOut = capture(() => printReport([]));
  assert.ok(emptyOut.includes("total: 0 PASS, 0 FAIL"), "empty run summarises 0/0");
  assert.ok(emptyOut.includes("Honest limits:"), "empty run still prints limits");

  console.log("bench-runner: all checks passed");
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
