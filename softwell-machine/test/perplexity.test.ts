import test from "node:test";
import assert from "node:assert/strict";
import { rowNll, totalNll, pplFromMeanNll } from "../src/bench/perplexity.ts";

test("rowNll: uniform 2-way logits give ln 2", () => {
  const logits = new Float32Array([0, 0]);
  assert.ok(Math.abs(rowNll(logits, 0, 2, 0) - Math.LN2) < 1e-6);
  assert.ok(Math.abs(rowNll(logits, 0, 2, 1) - Math.LN2) < 1e-6);
});

test("rowNll: hand-computed 3-way case", () => {
  // logits [1, 0, -1], target 0: logsumexp = ln(e + 1 + e^-1) = 1.40761...,
  // NLL = 1.40761... - 1 = 0.40761...
  const logits = new Float32Array([1, 0, -1]);
  const expected = Math.log(Math.E + 1 + 1 / Math.E) - 1;
  assert.ok(Math.abs(rowNll(logits, 0, 3, 0) - expected) < 1e-6);
  assert.ok(Math.abs(expected - 0.40760596) < 1e-6);
});

test("rowNll: large logits stay finite (max-subtraction)", () => {
  const logits = new Float32Array([1000, 999, -1000]);
  const nll = rowNll(logits, 0, 3, 0);
  assert.ok(Number.isFinite(nll));
  // e^1000 would overflow without the max subtraction; NLL ≈ ln(1 + e^-1 + e^-2000).
  assert.ok(Math.abs(nll - Math.log(1 + Math.exp(-1))) < 1e-6);
});

test("totalNll: sums rows in order, honours the row offset", () => {
  // Row 0: [0, 0] target 1 -> ln 2. Row 1: [2, 0] target 0 -> ln(1 + e^-2).
  const logits = new Float32Array([0, 0, 2, 0]);
  const total = totalNll(logits, 2, [1, 0]);
  const expected = Math.LN2 + Math.log(1 + Math.exp(-2));
  assert.ok(Math.abs(total - expected) < 1e-6);
});

test("pplFromMeanNll: exp of the mean NLL", () => {
  assert.ok(Math.abs(pplFromMeanNll(Math.LN2) - 2) < 1e-9);
  assert.ok(Math.abs(pplFromMeanNll(0) - 1) < 1e-9);
});
