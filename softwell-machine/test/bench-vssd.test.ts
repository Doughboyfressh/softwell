/**
 * bench-vssd smoke test — run with: node test/bench-vssd.test.ts (exit 0 on pass).
 *
 * Runs a SMALL bench (1 donor, 16 MiB sequential object, 200 random ops)
 * and asserts only SHAPE: every row is a well-formed BenchResult for
 * device "vssd" with finite numbers and a computed verdict. It does NOT
 * assert PASS — on this small machine FAIL vs the Micron 9650 is
 * EXPECTED, and the honest measured numbers are what matter.
 */
import assert from "node:assert/strict";
import { runBench } from "../src/bench/bench-vssd.ts";

async function main(): Promise<void> {
  const results = await runBench({ donors: 1, size: "small" });
  assert.equal(results.length, 4, `expected 4 bench rows, got ${results.length}`);

  const metrics = results.map((r) => r.metric);
  assert.ok(metrics.includes("sequential read"), `missing sequential read row: ${metrics}`);
  assert.ok(metrics.includes("sequential write"), `missing sequential write row: ${metrics}`);
  assert.ok(metrics.includes("random 4 KiB read"), `missing random read row: ${metrics}`);
  assert.ok(metrics.includes("capacity stored"), `missing capacity row: ${metrics}`);

  for (const r of results) {
    assert.equal(r.device, "vssd");
    assert.equal(typeof r.metric, "string");
    assert.equal(typeof r.unit, "string");
    assert.ok(r.unit.length > 0, `empty unit for ${r.metric}`);
    assert.ok(Number.isFinite(r.measured), `measured not finite for ${r.metric}: ${r.measured}`);
    assert.ok(r.measured >= 0, `measured negative for ${r.metric}: ${r.measured}`);
    assert.ok(Number.isFinite(r.target) && r.target > 0, `bad target for ${r.metric}: ${r.target}`);
    assert.equal(typeof r.targetName, "string");
    assert.ok(r.targetName.length > 0, `empty targetName for ${r.metric}`);
    assert.equal(r.donors, 1);
    assert.ok(Number.isFinite(r.perDonor), `perDonor not finite for ${r.metric}`);
    assert.ok(
      Number.isFinite(r.donorsToBeat) && r.donorsToBeat >= 1,
      `bad donorsToBeat for ${r.metric}: ${r.donorsToBeat}`,
    );
    assert.ok(r.verdict === "PASS" || r.verdict === "FAIL", `bad verdict for ${r.metric}`);
    assert.equal(r.verdict, r.measured >= r.target ? "PASS" : "FAIL", `verdict mismatch for ${r.metric}`);
    assert.equal(typeof r.notes, "string");
    assert.ok(r.notes.length > 0, `empty notes for ${r.metric}`);
  }

  // The real Micron targets must be the comparison point, not a straw man.
  const byMetric = Object.fromEntries(results.map((r) => [r.metric, r]));
  assert.equal(byMetric["sequential read"].target, 28);
  assert.equal(byMetric["sequential write"].target, 14);
  assert.equal(byMetric["random 4 KiB read"].target, 5_500_000);
  assert.ok(byMetric["random 4 KiB read"].measured > 0, "random IOPS must be a real measured value > 0");
  assert.ok(byMetric["sequential read"].measured > 0, "sequential read must be a real measured value > 0");
  assert.ok(byMetric["sequential write"].measured > 0, "sequential write must be a real measured value > 0");

  // Shape only: FAIL vs the Micron 9650 is expected on this machine.
  console.log(
    "bench-vssd smoke test passed (shape only — verdicts: " +
      results.map((r) => `${r.metric}=${r.verdict}`).join(", ") +
      ")",
  );
}

main().catch((err: unknown) => {
  console.error("bench-vssd smoke test FAILED:", err);
  process.exitCode = 1;
});
