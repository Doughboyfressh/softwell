/**
 * bench-vgpu test — run with: node test/bench-vgpu.test.ts (exit 0 on pass).
 *
 * Runs the real vGPU benchmark at small sizes and asserts the SHAPE of
 * the results, not a PASS: on this small CPU-only machine every row is
 * expected to FAIL against Rubin/B200 targets (a few GFLOP/s of FP32
 * against 2,250–4,000 BF16 TFLOPS). Asserting FAIL-here would bake the
 * machine into the test, so the test asserts well-formedness instead:
 * every row has finite measured/per-donor numbers, donorsToBeat >= 1,
 * a computed verdict, and the expected rows are present.
 */
import assert from "node:assert/strict";
import { runBench } from "../src/bench/bench-vgpu.ts";
import type { BenchResult } from "../src/bench/types.ts";

async function main(): Promise<void> {
  const results: BenchResult[] = await runBench({ donors: 2, size: "small" });

  // Rows: 2 matmul sizes x 2 targets (Rubin + B200), 2 bandwidth rows,
  // 2 scaling rows (1 and 2 donors).
  assert.equal(results.length, 8, `expected 8 rows, got ${results.length}`);

  const matmulRows = results.filter((r) => r.metric.startsWith("matmul TFLOPS"));
  const bwRows = results.filter((r) => r.metric.startsWith("memory bandwidth"));
  const scalingRows = results.filter((r) => r.metric.startsWith("aggregate matmul TFLOPS"));
  assert.equal(matmulRows.length, 4);
  assert.equal(bwRows.length, 2);
  assert.equal(scalingRows.length, 2);
  assert.deepEqual(
    scalingRows.map((r) => r.donors),
    [1, 2],
  );

  for (const r of results) {
    assert.equal(r.device, "vgpu");
    assert.ok(r.metric.length > 0);
    assert.ok(Number.isFinite(r.measured) && r.measured > 0, `${r.metric}: measured=${r.measured}`);
    assert.ok(r.unit.length > 0);
    assert.ok(Number.isFinite(r.target) && r.target > 0, `${r.metric}: target=${r.target}`);
    assert.ok(r.targetName.length > 0);
    assert.ok(Number.isInteger(r.donors) && r.donors >= 1);
    assert.ok(Number.isFinite(r.perDonor) && r.perDonor > 0, `${r.metric}: perDonor=${r.perDonor}`);
    assert.ok(
      Number.isFinite(r.donorsToBeat) && r.donorsToBeat >= 1,
      `${r.metric}: donorsToBeat=${r.donorsToBeat}`,
    );
    // Verdict is computed from measured vs target, never asserted by the
    // bench: re-derive it here.
    assert.equal(r.verdict, r.measured >= r.target ? "PASS" : "FAIL");
    assert.ok(r.notes.length > 0);
    // FP32-vs-tensor-core honesty: compute rows must say so.
    if (r.metric.includes("matmul")) {
      assert.match(r.notes, /FP32/);
    }
  }

  // On this small CPU-only machine every row is expected to FAIL — the
  // test asserts the shape, not a PASS. If a row ever passes here it
  // means the machine got much faster, not that the test is wrong; the
  // verdict derivation above already covers both cases.
  const fails = results.filter((r) => r.verdict === "FAIL").length;
  console.log(`bench-vgpu test passed (${results.length} rows, ${fails} FAIL as expected on this machine)`);
  for (const r of results) {
    console.log(
      `  ${r.metric} — measured ${r.measured.toFixed(3)} ${r.unit} vs ${r.targetName} ${r.target} ` +
        `-> ${r.verdict} (donors to beat: ${r.donorsToBeat})`,
    );
  }
}

main().catch((err: unknown) => {
  console.error("bench-vgpu test FAILED:", err);
  process.exitCode = 1;
});
