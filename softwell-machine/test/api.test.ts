/**
 * Smoke test for the API server + OpenAI mapping (src/api/).
 *
 * Builds a fake Scheduler (one fake token per step, done after 3 tokens)
 * and a fake byte-level tokenizer stub, starts SoftwellServer on an
 * ephemeral port, and asserts the public routes behave.
 *
 * Run: node test/api.test.ts   (exit 0 on success)
 */

import type {
  ClusterSnapshot,
  GeneratedToken,
  GenerateRequest,
  Scheduler,
  SequenceState,
  Tokenizer,
} from "../src/contracts.ts";
import { SoftwellServer } from "../src/api/server.ts";

// --- fakes -----------------------------------------------------------------

const fakeTokenizer: Tokenizer = {
  vocabSize: 256,
  bosTokenId: 0,
  eosTokenId: 1,
  encode(text: string): number[] {
    return [...Buffer.from(text, "utf8")];
  },
  decode(tokenIds: number[]): string {
    return Buffer.from(tokenIds).toString("utf8");
  },
};

class FakeScheduler implements Scheduler {
  private seqs = new Map<string, SequenceState>();
  private counter = 0;

  submit(req: GenerateRequest): string {
    const id = `seq-${++this.counter}`;
    this.seqs.set(id, {
      id,
      promptTokenIds: req.tokenIds,
      generatedTokenIds: [],
      status: "waiting",
      createdAt: Date.now(),
      finishReason: null,
    });
    return id;
  }

  async step(): Promise<GeneratedToken[]> {
    const out: GeneratedToken[] = [];
    for (const seq of this.seqs.values()) {
      if (seq.status === "done" || seq.status === "error") continue;
      seq.status = "decoding";
      const tokenId = 72; // "H"
      seq.generatedTokenIds.push(tokenId);
      const done = seq.generatedTokenIds.length >= 3;
      const finishReason = done ? "stop" : null;
      if (done) {
        seq.status = "done";
        seq.finishReason = "stop";
      }
      out.push({ seqId: seq.id, tokenId, text: "H", finishReason });
    }
    return out;
  }

  getSequence(id: string): SequenceState | undefined {
    return this.seqs.get(id);
  }

  pendingCount(): number {
    let n = 0;
    for (const s of this.seqs.values()) if (s.status === "waiting") n += 1;
    return n;
  }

  runningCount(): number {
    let n = 0;
    for (const s of this.seqs.values()) {
      if (s.status === "prefill" || s.status === "decoding") n += 1;
    }
    return n;
  }
}

const snapshot: ClusterSnapshot = {
  modelId: "test-model",
  plan: {
    modelId: "test-model",
    totalParams: 1_000_000,
    totalWeightBytes: 4_000_000,
    pipelineStages: 1,
    tensorParallelSize: 1,
    expertParallelSize: 1,
    assignments: [
      {
        workerId: "worker-1",
        stage: 0,
        layerStart: 0,
        layerEnd: 2,
        hasEmbedding: true,
        hasLmHead: true,
        tensorParallelRank: 0,
        tensorParallelSize: 1,
        experts: [],
        estimatedMemoryBytes: 4_000_000,
      },
    ],
    estimatedTokensPerSec: 42,
    estimatedTtftMs: 120,
    bottleneck: "memory-bandwidth-bound decode",
    createdAt: new Date().toISOString(),
  },
  workers: [
    {
      workerId: "worker-1",
      host: "127.0.0.1",
      port: 9001,
      device: "cpu",
      backend: "softwell-ts",
      memoryBytes: 8 * 1024 ** 3,
      freeMemoryBytes: 6 * 1024 ** 3,
      flopsPerSec: 1e12,
      bandwidthBytesPerSec: 1e9,
      roles: ["mixed"],
      maxBatchTokens: 512,
      state: "healthy",
      registeredAt: Date.now(),
      lastHeartbeatAt: Date.now(),
      usedMemoryBytes: 2 * 1024 ** 3,
      activeSequences: 0,
    },
  ],
  sequences: { waiting: 0, running: 0, done: 0 },
  tokensPerSec: 42,
  avgLatencyMs: 12.5,
  uptimeSec: 60,
};

// --- tiny assertion harness -----------------------------------------------

const failures: string[] = [];

function check(name: string, cond: boolean, detail = ""): void {
  if (cond) {
    console.log(`PASS ${name}`);
  } else {
    failures.push(name);
    console.log(`FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

// --- test ------------------------------------------------------------------

const server = new SoftwellServer({
  scheduler: new FakeScheduler(),
  tokenizer: fakeTokenizer,
  modelId: "test-model",
  getCluster: () => snapshot,
});

const port = await server.start();
const base = `http://127.0.0.1:${port}`;

try {
  // GET /v1/models lists the model
  {
    const res = await fetch(`${base}/v1/models`);
    const body = (await res.json()) as { object: string; data: { id: string }[] };
    check("GET /v1/models is 200", res.status === 200, `got ${res.status}`);
    check(
      "GET /v1/models lists test-model",
      body.object === "list" && body.data.some((m) => m.id === "test-model"),
      JSON.stringify(body),
    );
  }

  // GET /v1/models/:id
  {
    const res = await fetch(`${base}/v1/models/test-model`);
    const body = (await res.json()) as { id: string };
    check("GET /v1/models/:id is 200", res.status === 200, `got ${res.status}`);
    check("GET /v1/models/:id returns the model", body.id === "test-model");
    const missing = await fetch(`${base}/v1/models/nope`);
    check("GET /v1/models/:id unknown is 404", missing.status === 404, `got ${missing.status}`);
  }

  // POST /v1/chat/completions, non-streaming
  {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "test-model",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
      }),
    });
    const body = (await res.json()) as {
      choices: { message: { content: string }; finish_reason: string | null }[];
      usage: { completion_tokens: number };
    };
    check("POST non-streaming is 200", res.status === 200, `got ${res.status}`);
    check(
      "POST non-streaming returns content",
      body.choices[0]?.message.content === "HHH",
      JSON.stringify(body),
    );
    check(
      "POST non-streaming usage counts 3 tokens",
      body.usage.completion_tokens === 3,
      JSON.stringify(body.usage),
    );
    check(
      "POST non-streaming finish_reason is stop",
      body.choices[0]?.finish_reason === "stop",
      JSON.stringify(body.choices),
    );
  }

  // POST /v1/chat/completions, streaming (SSE)
  {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "test-model",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    const text = await res.text();
    check("POST streaming is 200", res.status === 200, `got ${res.status}`);
    check(
      "POST streaming is SSE",
      (res.headers.get("content-type") ?? "").includes("text/event-stream"),
      res.headers.get("content-type") ?? "",
    );
    const events = text.split("\n\n").filter((e) => e.startsWith("data: "));
    check("POST streaming emits token chunks", events.length >= 4, text);
    const chunks = events
      .filter((e) => !e.includes("[DONE]"))
      .map((e) => JSON.parse(e.slice("data: ".length)) as {
        choices: { delta: { content?: string }; finish_reason: string | null }[];
      });
    const streamed = chunks
      .map((c) => c.choices[0]?.delta.content ?? "")
      .join("");
    check("POST streaming content is HHH", streamed === "HHH", JSON.stringify(streamed));
    check("POST streaming ends with [DONE]", text.trimEnd().endsWith("data: [DONE]"), text);
  }

  // POST with bad JSON / missing messages -> 400
  {
    const badJson = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{not json",
    });
    check("POST bad JSON is 400", badJson.status === 400, `got ${badJson.status}`);
    const noMessages = await fetch(`${base}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "test-model", messages: [] }),
    });
    check("POST empty messages is 400", noMessages.status === 400, `got ${noMessages.status}`);
  }

  // GET /v1/cluster returns the injected snapshot
  {
    const res = await fetch(`${base}/v1/cluster`);
    const body = (await res.json()) as ClusterSnapshot;
    check("GET /v1/cluster is 200", res.status === 200, `got ${res.status}`);
    check("GET /v1/cluster returns modelId", body.modelId === "test-model");
    check(
      "GET /v1/cluster returns workers",
      body.workers.length === 1 && body.workers[0]?.workerId === "worker-1",
      JSON.stringify(body.workers),
    );
    check("GET /v1/cluster returns tokensPerSec", body.tokensPerSec === 42);
  }

  // GET / serves the dashboard HTML
  {
    const res = await fetch(`${base}/`);
    const text = await res.text();
    check("GET / is 200", res.status === 200, `got ${res.status}`);
    check("GET / serves HTML containing softwell", text.toLowerCase().includes("softwell"), text.slice(0, 120));
    check("GET / HTML polls /v1/cluster", text.includes("/v1/cluster"));
  }
} finally {
  await server.close();
}

if (failures.length > 0) {
  console.log(`\n${failures.length} failure(s): ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nAll API smoke tests passed.");
