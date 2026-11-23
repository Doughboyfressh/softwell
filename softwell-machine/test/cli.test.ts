/**
 * CLI arg-parsing coverage (test/cli.test.ts).
 *
 * Imports parseArgs from src/cli.ts (the module only runs main() when
 * executed directly, so importing it is side-effect free).
 *
 * Run: node --test test/cli.test.ts
 */

import test from "node:test";
import assert from "node:assert/strict";
import { parseArgs } from "../src/cli.ts";

test("parseArgs: up defaults", () => {
  const opts = parseArgs(["up"]);
  assert.equal(opts.command, "up");
  assert.equal(opts.workers, 2);
  assert.equal(opts.port, 8410);
  assert.equal(opts.apiPort, 8420);
  assert.equal(opts.modelDir, undefined);
});

test("parseArgs: up with flags", () => {
  const opts = parseArgs([
    "up",
    "--model-dir",
    "demo/models/demo-10m",
    "--workers",
    "1",
    "--port",
    "18410",
    "--api-port",
    "18420",
    "--memory",
    "1073741824",
  ]);
  assert.equal(opts.command, "up");
  assert.equal(opts.modelDir, "demo/models/demo-10m");
  assert.equal(opts.workers, 1);
  assert.equal(opts.port, 18410);
  assert.equal(opts.apiPort, 18420);
  assert.equal(opts.memoryBytes, 1073741824);
});

test("parseArgs: run and serve take a model", () => {
  assert.equal(parseArgs(["run", "demo/models/demo-10m"]).command, "run");
  assert.equal(parseArgs(["run", "demo/models/demo-10m"]).modelRef, "demo/models/demo-10m");
  assert.equal(parseArgs(["serve", "meta-llama/Llama-3.3-70B-Instruct"]).command, "serve");
  assert.equal(
    parseArgs(["serve", "meta-llama/Llama-3.3-70B-Instruct"]).modelRef,
    "meta-llama/Llama-3.3-70B-Instruct",
  );
});

test("parseArgs: run/serve without a model throw", () => {
  assert.throws(() => parseArgs(["run"]), /softwell run needs a <model> argument/);
  assert.throws(() => parseArgs(["serve"]), /softwell serve needs a <model> argument/);
});

test("parseArgs: models and stop take no model", () => {
  assert.equal(parseArgs(["models"]).command, "models");
  assert.equal(parseArgs(["models", "--api-port", "18420"]).apiPort, 18420);
  assert.equal(parseArgs(["stop"]).command, "stop");
  assert.equal(parseArgs(["stop", "--api-port", "18420"]).apiPort, 18420);
});

test("parseArgs: --help", () => {
  assert.equal(parseArgs(["--help"]).command, "help");
  assert.equal(parseArgs([]).command, "help");
});

test("parseArgs: unknown flags throw", () => {
  assert.throws(() => parseArgs(["up", "--nope"]), /unknown flag: --nope/);
});

test("parseArgs: novel engine flags pass through", () => {
  const opts = parseArgs([
    "run",
    "demo/models/demo-10m",
    "--attention",
    "landmark",
    "--weights",
    "sw3",
    "--kv",
    "sw2",
    "--decoding",
    "speculative",
    "--draft-length",
    "8",
  ]);
  assert.equal(opts.engineOptions.attention, "landmark");
  assert.equal(opts.engineOptions.weightFormat, "sw3");
  assert.equal(opts.engineOptions.kvFormat, "sw2");
  assert.equal(opts.engineOptions.decoding, "speculative");
  assert.equal(opts.engineOptions.draftLength, 8);
});

test("parseArgs: invalid engine flag values throw", () => {
  assert.throws(() => parseArgs(["run", "x", "--attention", "bogus"]), /invalid engine option/);
});

test("parseArgs: --draft-model with speculative decoding", () => {
  const opts = parseArgs([
    "serve",
    "demo/models/demo-100m",
    "--decoding",
    "speculative",
    "--draft-model",
    "demo/models/demo-10m",
  ]);
  assert.equal(opts.draftModelDir, "demo/models/demo-10m");
  assert.equal(opts.engineOptions.decoding, "speculative");
});

test("parseArgs: --draft-model without speculative decoding throws", () => {
  assert.throws(
    () => parseArgs(["serve", "demo/models/demo-100m", "--draft-model", "demo/models/demo-10m"]),
    /--draft-model requires --decoding speculative/,
    "draft model without speculative decoding fails fast, not a silent n-gram fallback",
  );
  assert.throws(
    () => parseArgs(["serve", "demo/models/demo-100m", "--decoding", "jacobi", "--draft-model", "demo/models/demo-10m"]),
    /--draft-model requires --decoding speculative/,
    "draft model with Jacobi also fails fast",
  );
});

test("parseArgs: no --draft-model leaves the default (no draft model)", () => {
  const opts = parseArgs(["serve", "demo/models/demo-100m", "--decoding", "speculative"]);
  assert.equal(opts.draftModelDir, undefined);
});
