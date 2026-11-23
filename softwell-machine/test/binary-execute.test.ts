/**
 * Binary activation transport — run with: node test/binary-execute.test.ts
 *
 * POST /mesh/execute used to carry hidden states as JSON number arrays.
 * It now carries them as raw float32 LE bytes with a small JSON header
 * ([u32 LE header length][header JSON][float32 payload]); see the
 * framing in src/mesh/protocol.ts. This test proves it at the wire level:
 *
 *  - MeshClient.execute round-trips a 64x128 hidden-state tensor through
 *    a real createMeshServer echo handler, bit-exact as float32, and the
 *    tokenIds placeholder path ({ data: [], shape: [n, 0] }) still works.
 *  - A raw binary fetch gets an application/octet-stream response that
 *    decodes (with the exported decoder) to the same values.
 *  - The binary body is less than half the JSON body for the same
 *    request (long-decimal values, the realistic case).
 *  - A raw JSON request still works and gets a JSON response
 *    (backward compatibility with older clients).
 *  - Malformed frames are rejected with 400: truncated payload,
 *    payload length mismatch, bad header JSON, absurd header length.
 */
import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { MeshClient, createMeshServer, decodeExecuteResponse, encodeExecuteRequest } from "../src/mesh/protocol.ts";
import type { ExecuteRequest, ExecuteResponse } from "../src/contracts.ts";

/** Deterministic long-decimal values (the realistic JSON-bloat case). */
function makeInput(rows: number, cols: number): number[] {
  const out: number[] = new Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    out[i] = (((i * 2654435761) % 1000) / 1000) * 0.123456789 - 0.0617283945;
  }
  return out;
}

async function main(): Promise<void> {
  const client = new MeshClient({ timeoutMs: 5000 });
  // Echo handler: returns the input hidden states unchanged; on the
  // tokenIds path it mimics embed() (shape [n, 2], constant 0.5).
  const server = createMeshServer({
    execute: async (req: ExecuteRequest): Promise<ExecuteResponse> => {
      if (req.tokenIds !== undefined) {
        const n = req.tokenIds.length;
        return {
          output: { data: new Array(n * 2).fill(0.5), shape: [n, 2] },
          isLogits: false,
          tokensProcessed: n,
        };
      }
      return {
        output: { data: [...req.input.data], shape: [...req.input.shape] },
        isLogits: false,
        tokensProcessed: req.positions.length,
      };
    },
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    // --- 1. Client round-trip, bit-exact as float32 ----------------------
    const input = makeInput(64, 128);
    const expected = Array.from(new Float32Array(input));
    const resp = await client.execute(base, {
      workerId: "worker-1",
      seqIds: new Array(64).fill("seq-1"),
      kind: "prefill",
      stage: 0,
      input: { data: input, shape: [64, 128] },
      positions: Array.from({ length: 64 }, (_, i) => i),
    });
    assert.deepEqual(resp.output, { data: expected, shape: [64, 128] });
    assert.equal(resp.isLogits, false);
    assert.equal(resp.tokensProcessed, 64);
    console.log("client binary round-trip (64x128, f32-exact): ok");

    // --- 2. tokenIds placeholder path ------------------------------------
    const tokResp = await client.execute(base, {
      workerId: "worker-1",
      seqIds: ["seq-1", "seq-1", "seq-1"],
      kind: "prefill",
      stage: 0,
      input: { data: [], shape: [3, 0] },
      positions: [0, 1, 2],
      tokenIds: [1, 2, 3],
    });
    assert.deepEqual(tokResp.output, { data: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], shape: [3, 2] });
    assert.equal(tokResp.tokensProcessed, 3);
    console.log("tokenIds placeholder path: ok");

    // --- 3. Wire size: binary < half of JSON ------------------------------
    const req: ExecuteRequest = {
      workerId: "worker-1",
      seqIds: new Array(64).fill("seq-1"),
      kind: "prefill",
      stage: 0,
      input: { data: input, shape: [64, 128] },
      positions: Array.from({ length: 64 }, (_, i) => i),
    };
    const binaryBytes = encodeExecuteRequest(req).byteLength;
    const jsonBytes = Buffer.byteLength(JSON.stringify(req), "utf8");
    assert.ok(
      binaryBytes * 2 < jsonBytes,
      `binary body ${binaryBytes} B should be < half of JSON body ${jsonBytes} B`,
    );
    console.log(`wire size: binary ${binaryBytes} B vs JSON ${jsonBytes} B (${(jsonBytes / binaryBytes).toFixed(2)}x smaller)`);

    // --- 4. Raw binary fetch: octet-stream response, decodable ------------
    const rawBinary = await fetch(`${base}/mesh/execute`, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: encodeExecuteRequest(req),
    });
    assert.equal(rawBinary.status, 200, `raw binary fetch: HTTP ${rawBinary.status}`);
    assert.equal(
      (rawBinary.headers.get("content-type") ?? "").split(";")[0].trim(),
      "application/octet-stream",
      "binary request must get a binary response",
    );
    const decoded = decodeExecuteResponse(new Uint8Array(await rawBinary.arrayBuffer()));
    assert.deepEqual(decoded.output, { data: expected, shape: [64, 128] });
    console.log("raw binary fetch + decode: ok");

    // --- 5. Raw JSON fetch: backward compatibility ------------------------
    const rawJson = await fetch(`${base}/mesh/execute`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        workerId: "worker-1",
        seqIds: ["seq-1", "seq-1"],
        kind: "prefill",
        stage: 0,
        input: { data: [1, 2, 3, 4], shape: [2, 2] },
        positions: [0, 1],
      }),
    });
    assert.equal(rawJson.status, 200, `raw JSON fetch: HTTP ${rawJson.status}`);
    assert.equal(
      (rawJson.headers.get("content-type") ?? "").split(";")[0].trim(),
      "application/json",
      "JSON request must get a JSON response",
    );
    assert.deepEqual((await rawJson.json()) as unknown, {
      output: { data: [1, 2, 3, 4], shape: [2, 2] },
      isLogits: false,
      tokensProcessed: 2,
    });
    console.log("raw JSON request (backward compat): ok");

    // --- 6. Malformed frames -> 400 ---------------------------------------
    const good = encodeExecuteRequest(req);
    async function expect400(label: string, body: Uint8Array): Promise<void> {
      const res = await fetch(`${base}/mesh/execute`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body,
      });
      assert.equal(res.status, 400, `${label}: expected HTTP 400, got ${res.status}`);
    }
    // Truncated payload (drop the last 4 bytes of one float).
    await expect400("truncated payload", good.subarray(0, good.length - 4));
    // Payload length mismatch (one extra float appended).
    const longer = new Uint8Array(good.length + 4);
    longer.set(good);
    await expect400("payload length mismatch", longer);
    // Bad header JSON: header length 2, body "xx", then a valid payload.
    const badHeader = new Uint8Array(4 + 2 + 4);
    new DataView(badHeader.buffer).setUint32(0, 2, true);
    badHeader.set(new TextEncoder().encode("xx"), 4);
    await expect400("bad header JSON", badHeader);
    // Absurd header length (claims 1 GiB).
    const hugeHeader = new Uint8Array(4);
    new DataView(hugeHeader.buffer).setUint32(0, 1 << 30, true);
    await expect400("absurd header length", hugeHeader);
    // Empty body.
    await expect400("empty body", new Uint8Array(0));
    console.log("malformed frames -> 400 (5 cases): ok");

    console.log("binary-execute test passed");
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

main().catch((err: unknown) => {
  console.error("binary-execute test FAILED:", err);
  process.exitCode = 1;
});
