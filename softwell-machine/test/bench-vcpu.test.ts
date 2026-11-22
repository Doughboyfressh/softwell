/**
 * bench-vcpu test — run with: node test/bench-vcpu.test.ts (exit 0 on pass).
 *
 * A small bench run returns well-formed BenchResult rows: one cores
 * row, one FP64 TFLOPS row (whose notes label the 27.6 TFLOPS target
 * as an estimate, never a vendor number), and one informational
 * throughput row. A FAIL verdict on this small machine is EXPECTED —
 * this test asserts shape, never PASS.
 */
import assert from "node:assert/strict";
import { cpus } from "node:os";
import { runBench } from "../src/bench/bench-vcpu.ts";
import type { BenchResult } from "../src/bench/types.ts";

function assertWellFormed(results: BenchResult[]): void {
  assert.ok(results.length >= 3, `expected >= 3 rows, got ${results.length}`);
  for (const r of results) {
    assert.equal(r.device, "vcpu");
    assert.ok(typeof r.metric === "string" && r.metric.length > 0);
    assert.ok(Number.isFinite(r.measured) && r.measured >= 0, `${r.metric}: measured=${r.measured}`);
    assert.ok(typeof r.unit === "string" && r.unit.length > 0);
    assert.ok(Number.isFinite(r.target), `${r.metric}: target=${r.target}`);
    assert.ok(typeof r.targetName === "string" && r.targetName.length > 0);
    assert.ok(r.donors >= 1);
    assert.ok(Number.isFinite(r.perDonor) && r.perDonor >= 0);
    assert.ok(r.donorsToBeat >= 1, `${r.metric}: donorsToBeat=${r.donorsToBeat}`);
    assert.ok(r.verdict === "PASS" || r.verdict === "FAIL");
    assert.equal(
      r.verdict,
      r.measured >= r.target ? "PASS" : "FAIL",
      `${r.metric}: verdict must be computed from measured vs target`,
    );
    assert.ok(typeof r.notes === "string" && r.notes.length > 0);
  }
}

async function main(): Promise<void> {
  const hostCores = cpus().length;

  // One donor, small workload.
  const one = await runBench({ donors: 1, size: "small" });
  assertWellFormed(one);
  const coresRow = one.find((r) => r.metric === "virtual cores");
  assert.ok(coresRow, "cores row present");
  assert.equal(coresRow.measured, hostCores, `1 donor must report the host's ${hostCores} cores`);
  assert.equal(coresRow.target, 192, "cores target is the EPYC 9965's 192");
  assert.equal(coresRow.targetName, "AMD EPYC 9965 (Turin)");
  assert.equal(coresRow.donors, 1);
  // FAIL is expected here; asserting the shape, not a pass.
  assert.equal(coresRow.verdict, "FAIL");

  const tflopsRow = one.find((r) => r.metric.startsWith("FP64 TFLOPS"));
  assert.ok(tflopsRow, "TFLOPS row present");
  assert.ok(tflopsRow.measured > 0, `TFLOPS must be a real positive measurement, got ${tflopsRow.measured}`);
  assert.equal(tflopsRow.target, 27.6);
  assert.match(tflopsRow.targetName, /estimate/i, "TFLOPS target must be labelled an estimate");
  assert.match(tflopsRow.notes, /estimate/i, "TFLOPS notes must label the target as an estimate");
  assert.match(tflopsRow.notes, /not a vendor number/i);

  const throughputRow = one.find((r) => r.metric === "flops task throughput");
  assert.ok(throughputRow, "throughput row present");
  assert.ok(throughputRow.measured > 0, `tasks/s must be positive, got ${throughputRow.measured}`);
  assert.equal(throughputRow.target, 0, "throughput row is informational (target 0)");
  assert.match(throughputRow.notes, /informational/i);

  // Two donors: the scaling table — one row per donor count for the
  // measured metrics, with cores summing the reported counts.
  const two = await runBench({ donors: 2, size: "small" });
  assertWellFormed(two);
  const coresRow2 = two.find((r) => r.metric === "virtual cores");
  assert.ok(coresRow2, "cores row present at 2 donors");
  assert.equal(coresRow2.measured, 2 * hostCores, "2 donors must report 2x the host cores");
  assert.equal(coresRow2.donors, 2);
  const tflopsRow2 = two.find((r) => r.metric.startsWith("FP64 TFLOPS"));
  assert.ok(tflopsRow2, "TFLOPS row present at 2 donors");
  assert.ok(tflopsRow2.measured > 0);

  console.log("bench-vcpu: all checks passed (FAIL verdicts expected on this machine)");
}

const watchdog = setTimeout(() => {
  console.error("bench-vcpu: timed out");
  process.exit(1);
}, 60_000);

main()
  .then(() => {
    clearTimeout(watchdog);
  })
  .catch((err) => {
    clearTimeout(watchdog);
    console.error(err);
    process.exitCode = 1;
  });
