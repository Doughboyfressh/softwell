/**
 * bench-vram smoke test — run with: node test/bench-vram.test.ts (exit 0 on pass).
 *
 * Runs a SMALL bench (16 MiB, 1-2 donors) and asserts only the SHAPE of
 * the results: well-formed BenchResult rows, computed fields consistent
 * (perDonor, donorsToBeat, verdict), capacity equal to the donors'
 * configured memory. FAIL verdicts are EXPECTED on this machine (loopback
 * HTTP vs HBM3e) and are NOT asserted against — a FAIL is a measurement.
 */
import assert from "node:assert/strict";
import { runBench } from "../src/bench/bench-vram.ts";

async function main(): Promise<void> {
  for (const donors of [1, 2]) {
    const results = await runBench({ donors, size: "small" });
    assert.equal(results.length, 4, `expected 4 rows for ${donors} donor(s)`);
    const metrics = results.map((r) => r.metric);
    assert.ok(metrics.some((m) => m.startsWith("sequential READ bandwidth")), "missing READ row");
    assert.ok(metrics.some((m) => m.startsWith("sequential WRITE bandwidth")), "missing WRITE row");
    assert.ok(metrics.some((m) => m === "pooled capacity"), "missing capacity row");
    assert.ok(metrics.some((m) => m.startsWith("single-page")), "missing latency row");

    for (const row of results) {
      assert.equal(row.device, "vram");
      assert.equal(row.donors, donors);
      assert.ok(Number.isFinite(row.measured) && row.measured >= 0, `measured must be finite >= 0: ${row.metric}`);
      assert.ok(row.target > 0, `target must be > 0: ${row.metric}`);
      assert.ok(row.targetName.length > 0, `targetName must be set: ${row.metric}`);
      assert.ok(row.unit.length > 0, `unit must be set: ${row.metric}`);
      assert.ok(row.notes.length > 0, `notes must be set: ${row.metric}`);
      assert.ok(row.verdict === "PASS" || row.verdict === "FAIL", `verdict must be PASS|FAIL: ${row.metric}`);
      assert.ok(Math.abs(row.perDonor - row.measured / donors) < 1e-9, "perDonor must be measured / donors");
      if (row.metric.startsWith("single-page")) {
        // Latency is lower-is-better: verdict is PASS only when
        // measured <= target, and more donors never beat a latency
        // target, so donorsToBeat is infinity.
        assert.equal(row.verdict, row.measured <= row.target ? "PASS" : "FAIL", "latency verdict must be PASS only when measured <= target");
        assert.equal(row.donorsToBeat, Number.POSITIVE_INFINITY, "latency donorsToBeat must be infinity");
      } else {
        assert.equal(row.verdict, row.measured >= row.target ? "PASS" : "FAIL", "verdict must be computed from measured vs target");
        assert.ok(
          row.donorsToBeat === Math.max(1, Math.ceil(row.target / row.perDonor)),
          "donorsToBeat must be ceil(target / perDonor)",
        );
      }
    }

    const read = results.find((r) => r.metric.startsWith("sequential READ bandwidth"))!;
    const write = results.find((r) => r.metric.startsWith("sequential WRITE bandwidth"))!;
    const capacity = results.find((r) => r.metric === "pooled capacity")!;
    const latency = results.find((r) => r.metric.startsWith("single-page"))!;
    assert.equal(read.unit, "GB/s");
    assert.equal(read.target, 8000, "READ target must be HBM3e 8 TB/s");
    assert.equal(write.unit, "GB/s");
    assert.equal(write.target, 8000, "WRITE target must be HBM3e 8 TB/s");
    assert.ok(read.measured > 0, "READ bandwidth must be > 0 (real timed transfer)");
    assert.ok(write.measured > 0, "WRITE bandwidth must be > 0 (real timed transfer)");
    assert.equal(capacity.unit, "GB");
    assert.equal(capacity.target, 192, "capacity target must be HBM3e 192 GB");
    // Each bench donor contributes 1 GiB; capacity is reported in
    // decimal GB, so 1 GiB = 2^30 / 1e9 GB per donor.
    const expectedCapacityGB = (donors * 2 ** 30) / 1e9;
    assert.ok(
      Math.abs(capacity.measured - expectedCapacityGB) < 1e-9,
      `capacity must be ${expectedCapacityGB} GB for ${donors} donor(s), got ${capacity.measured}`,
    );
    assert.equal(latency.unit, "ms");
    assert.ok(latency.measured > 0, "single-page latency must be > 0 (real timed read)");
    console.log(
      `bench-vram shape ok (${donors} donor(s)): ` +
        `read ${read.measured.toFixed(3)} GB/s, write ${write.measured.toFixed(3)} GB/s, ` +
        `capacity ${capacity.measured} GB, latency ${latency.measured.toFixed(2)} ms ` +
        `(verdicts: ${results.map((r) => r.verdict).join(",")} — FAIL expected vs HBM3e)`,
    );
  }
  console.log("bench-vram smoke test passed");
}

main().catch((err: unknown) => {
  console.error("bench-vram smoke test FAILED:", err);
  process.exitCode = 1;
});
