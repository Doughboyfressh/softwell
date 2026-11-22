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
  AdmissionControl,
  ClusterSnapshot,
  GeneratedToken,
  GenerateRequest,
  HealthStatus,
  ModelEntry,
  ModelInfo,
  ModelManifest,
  Scheduler,
  SequenceState,
  ServerRequestRecord,
  ServerStatus,
  ServingContext,
  ServingResolver,
  Tokenizer,
} from "../src/contracts.ts";
import { SoftwellServer } from "../src/api/server.ts";

// --- fakes -----------------------------------------------------------------

const fakeTokenizer: Tokenizer = {
  vocabSize: 256,
  bosTokenId: 0,
  eosTokenId: 1,
  // Models the real byte-level tokenizer: encode() owns BOS and
  // prepends it exactly once (see src/api/openai.ts).
  encode(text: string): number[] {
    return [0, ...Buffer.from(text, "utf8")];
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

  cancel(id: string): boolean {
    const seq = this.seqs.get(id);
    if (seq === undefined || seq.status === "done" || seq.status === "error") return false;
    seq.status = "done";
    seq.finishReason = "cancelled";
    return true;
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Scheduler with a per-step delay, a configurable token budget, and a
 *  cancel spy. produceTokens=false models a scheduler that never emits. */
class ConfigurableScheduler implements Scheduler {
  private seqs = new Map<string, SequenceState>();
  private counter = 0;
  cancelCalls: string[] = [];

  private stepDelayMs: number;
  private tokensToDone: number;
  private produceTokens: boolean;

  constructor(stepDelayMs: number, tokensToDone: number, produceTokens = true) {
    this.stepDelayMs = stepDelayMs;
    this.tokensToDone = tokensToDone;
    this.produceTokens = produceTokens;
  }

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
    if (this.stepDelayMs > 0) await sleep(this.stepDelayMs);
    if (!this.produceTokens) return [];
    const out: GeneratedToken[] = [];
    for (const seq of this.seqs.values()) {
      if (seq.status === "done" || seq.status === "error") continue;
      seq.status = "decoding";
      const tokenId = 72; // "H"
      seq.generatedTokenIds.push(tokenId);
      const done = seq.generatedTokenIds.length >= this.tokensToDone;
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

  cancel(id: string): boolean {
    this.cancelCalls.push(id);
    const seq = this.seqs.get(id);
    if (seq === undefined || seq.status === "done" || seq.status === "error") return false;
    seq.status = "done";
    seq.finishReason = "cancelled";
    return true;
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

class FakeAdmission implements AdmissionControl {
  records: ServerRequestRecord[] = [];
  reject: { status: number; message: string } | null = null;

  admit(
    _modelId: string,
  ): { ok: true } | { ok: false; status: number; message: string } {
    if (this.reject !== null) {
      return { ok: false, status: this.reject.status, message: this.reject.message };
    }
    return { ok: true };
  }

  record(record: ServerRequestRecord): void {
    this.records.push(record);
  }

  counters(): {
    active: number;
    queued: number;
    completed: number;
    rejected: number;
    cancelled: number;
  } {
    return {
      active: 0,
      queued: 0,
      completed: this.records.filter((r) => r.status === "done").length,
      rejected: this.records.filter((r) => r.status === "rejected").length,
      cancelled: this.records.filter((r) => r.status === "cancelled").length,
    };
  }
}

const fakeManifest = {
  format: "safetensors",
  modelId: "model-a",
} as unknown as ModelManifest;

class FakeServingResolver implements ServingResolver {
  activeId: string | null = "model-a";
  ready = true;
  healthy = true;
  shutdownRequested = false;
  scheduler: Scheduler;

  constructor(scheduler: Scheduler) {
    this.scheduler = scheduler;
  }

  getActive(): ServingContext | null {
    if (this.activeId === null) return null;
    return {
      modelId: this.activeId,
      scheduler: this.scheduler,
      tokenizer: fakeTokenizer,
      manifest: fakeManifest,
    };
  }

  listModels(): ModelInfo[] {
    return ["model-a", "model-b"].map((id) => ({
      id,
      object: "model",
      created: 1234567890,
      owned_by: "softwell",
    }));
  }

  getServerStatus(): ServerStatus {
    return {
      state: "ready",
      uptimeSec: 10,
      activeModelId: this.activeId,
      models: [],
      requests: { active: 0, queued: 0, completed: 0, rejected: 0, cancelled: 0 },
      tokensPerSec: 0,
      queueDepth: 0,
      vm: null,
      recentRequests: [],
    };
  }

  isReady(): boolean {
    return this.ready;
  }

  getHealth(): HealthStatus {
    return {
      status: this.healthy ? "ok" : "down",
      checks: {
        vmBooted: true,
        activeModel: this.activeId !== null,
        workersHealthy: this.healthy,
      },
    };
  }

  requestShutdown(): void {
    this.shutdownRequested = true;
  }

  // -- model management fake: a small in-memory registry ----------------
  knownModels = new Set<string>(["model-a", "model-b"]);
  notReadyModels = new Set<string>(["model-not-ready"]);
  loadCalls: string[] = [];

  private fakeEntry(id: string, status: ModelEntry["status"]): ModelEntry {
    return {
      id,
      source: { kind: "local", dir: `/tmp/${id}` },
      dir: `/tmp/${id}`,
      status,
      params: 1000,
      weightBytes: 4000,
      vramBytes: status === "ready" ? 4000 : 0,
      loadedAt: Date.now(),
      requestsServed: 0,
      tokensGenerated: 0,
    };
  }

  async loadModel(ref: string): Promise<ModelEntry> {
    this.loadCalls.push(ref);
    if (ref === "too-big") {
      const entry = this.fakeEntry("too-big", "error");
      entry.error =
        'model "too-big" does not fit in vRAM: weights 999 + 524288000 bytes KV headroom > 100 total';
      return entry;
    }
    if (!this.knownModels.has(ref) && ref !== "model-c") {
      throw new Error(`unknown model: ${ref}`);
    }
    this.knownModels.add(ref);
    return this.fakeEntry(ref, "ready");
  }

  async unloadModel(id: string): Promise<void> {
    if (id === this.activeId) {
      throw new Error(`cannot unload model "${id}": it is the active model`);
    }
    if (!this.knownModels.has(id)) {
      throw new Error(`unknown model: ${id}`);
    }
    this.knownModels.delete(id);
  }

  async activateModel(id: string): Promise<ServingContext> {
    if (this.notReadyModels.has(id)) {
      throw new Error(`model ${id} is not ready (status: error)`);
    }
    if (!this.knownModels.has(id)) {
      throw new Error(`unknown model: ${id}`);
    }
    this.activeId = id;
    const ctx = this.getActive();
    if (ctx === null) throw new Error("no active model after activation");
    return ctx;
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

// --- legacy health / ready / status / shutdown (no resolver, no hook) ----

{
  const legacy = new SoftwellServer({
    scheduler: new FakeScheduler(),
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
  });
  const legacyPort = await legacy.start();
  const legacyBase = `http://127.0.0.1:${legacyPort}`;
  try {
    {
      const res = await fetch(`${legacyBase}/healthz`);
      const body = (await res.json()) as {
        status: string;
        checks: { vmBooted: boolean; activeModel: boolean; workersHealthy: boolean };
      };
      check("GET /healthz (legacy) is 200", res.status === 200, `got ${res.status}`);
      check("GET /healthz (legacy) status is ok", body.status === "ok", JSON.stringify(body));
      check(
        "GET /healthz (legacy) is honest: vmBooted is false",
        body.checks.vmBooted === false,
        JSON.stringify(body.checks),
      );
      check(
        "GET /healthz (legacy) activeModel is true",
        body.checks.activeModel === true,
        JSON.stringify(body.checks),
      );
    }
    {
      const res = await fetch(`${legacyBase}/readyz`);
      const body = (await res.json()) as { ready: boolean };
      check("GET /readyz (legacy) is 200", res.status === 200, `got ${res.status}`);
      check("GET /readyz (legacy) ready is true", body.ready === true, JSON.stringify(body));
    }
    {
      const res = await fetch(`${legacyBase}/v1/server/status`);
      check(
        "GET /v1/server/status (legacy) is 404",
        res.status === 404,
        `got ${res.status}`,
      );
    }
    {
      const res = await fetch(`${legacyBase}/v1/server/shutdown`, { method: "POST" });
      check(
        "POST /v1/server/shutdown without onShutdown is 404",
        res.status === 404,
        `got ${res.status}`,
      );
    }
  } finally {
    await legacy.close();
  }
}

// --- POST /v1/completions ---------------------------------------------------

{
  const srv = new SoftwellServer({
    scheduler: new FakeScheduler(),
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    // Non-streaming, string prompt: BOS + "hi" (2 bytes) = 3 prompt tokens.
    {
      const res = await fetch(`${b}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "test-model", prompt: "hi", stream: false }),
      });
      const body = (await res.json()) as {
        id: string;
        object: string;
        choices: { text: string; finish_reason: string | null; index: number }[];
        usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
      };
      check("POST /v1/completions is 200", res.status === 200, `got ${res.status}`);
      check(
        "POST /v1/completions object is text_completion",
        body.object === "text_completion",
        JSON.stringify(body),
      );
      check(
        "POST /v1/completions returns HHH",
        body.choices[0]?.text === "HHH",
        JSON.stringify(body),
      );
      check(
        "POST /v1/completions choice index is 0",
        body.choices[0]?.index === 0,
        JSON.stringify(body.choices),
      );
      check(
        "POST /v1/completions prompt_tokens counts BOS + bytes",
        body.usage.prompt_tokens === 3,
        JSON.stringify(body.usage),
      );
      check(
        "POST /v1/completions completion_tokens is 3",
        body.usage.completion_tokens === 3,
        JSON.stringify(body.usage),
      );
      check(
        "POST /v1/completions total_tokens is 6",
        body.usage.total_tokens === 6,
        JSON.stringify(body.usage),
      );
      check(
        "POST /v1/completions id is a cmpl- id",
        body.id.startsWith("cmpl-"),
        body.id,
      );
    }

    // Non-streaming, string[] prompt: "hi\nyo" is 5 bytes + BOS = 6.
    {
      const res = await fetch(`${b}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "test-model", prompt: ["hi", "yo"] }),
      });
      const body = (await res.json()) as {
        choices: { text: string }[];
        usage: { prompt_tokens: number };
      };
      check("POST /v1/completions string[] is 200", res.status === 200, `got ${res.status}`);
      check(
        "POST /v1/completions string[] returns HHH",
        body.choices[0]?.text === "HHH",
        JSON.stringify(body),
      );
      check(
        "POST /v1/completions string[] prompt_tokens is 6",
        body.usage.prompt_tokens === 6,
        JSON.stringify(body.usage),
      );
    }

    // Streaming: delta.text chunks, ends with [DONE].
    {
      const res = await fetch(`${b}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "test-model", prompt: "hi", stream: true }),
      });
      const text = await res.text();
      check("POST /v1/completions streaming is 200", res.status === 200, `got ${res.status}`);
      check(
        "POST /v1/completions streaming is SSE",
        (res.headers.get("content-type") ?? "").includes("text/event-stream"),
        res.headers.get("content-type") ?? "",
      );
      const events = text.split("\n\n").filter((e) => e.startsWith("data: "));
      const chunks = events
        .filter((e) => !e.includes("[DONE]"))
        .map((e) => JSON.parse(e.slice("data: ".length)) as {
          object: string;
          choices: { delta: { text?: string }; finish_reason: string | null }[];
        });
      check(
        "POST /v1/completions streaming chunks are text_completion",
        chunks.length > 0 && chunks.every((c) => c.object === "text_completion"),
        text,
      );
      const streamed = chunks.map((c) => c.choices[0]?.delta.text ?? "").join("");
      check(
        "POST /v1/completions streaming delta.text is HHH",
        streamed === "HHH",
        JSON.stringify(streamed),
      );
      check(
        "POST /v1/completions streaming ends with [DONE]",
        text.trimEnd().endsWith("data: [DONE]"),
        text,
      );
    }

    // Invalid prompts -> 400.
    {
      const missing = await fetch(`${b}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "test-model" }),
      });
      check("POST /v1/completions missing prompt is 400", missing.status === 400, `got ${missing.status}`);
      const empty = await fetch(`${b}/v1/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "test-model", prompt: [] }),
      });
      check("POST /v1/completions empty prompt array is 400", empty.status === 400, `got ${empty.status}`);
    }
  } finally {
    await srv.close();
  }
}

// --- resolver form: model switching, 400 / 503, health, ready, status ------

{
  const admission = new FakeAdmission();
  const resolver = new FakeServingResolver(new FakeScheduler());
  const srv = new SoftwellServer({
    getCluster: () => snapshot,
    serving: resolver,
    admission,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    {
      const res = await fetch(`${b}/v1/models`);
      const body = (await res.json()) as { data: { id: string }[] };
      check("resolver GET /v1/models is 200", res.status === 200, `got ${res.status}`);
      check(
        "resolver GET /v1/models lists both models",
        body.data.map((m) => m.id).sort().join(",") === "model-a,model-b",
        JSON.stringify(body),
      );
    }
    {
      const res = await fetch(`${b}/v1/models/model-b`);
      check("resolver GET /v1/models/model-b is 200", res.status === 200, `got ${res.status}`);
      const missing = await fetch(`${b}/v1/models/nope`);
      check("resolver GET /v1/models/nope is 404", missing.status === 404, `got ${missing.status}`);
    }
    // Inactive model -> 400 naming both models.
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "model-b",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      const body = (await res.json()) as { error: { message: string } };
      check("resolver chat for inactive model is 400", res.status === 400, `got ${res.status}`);
      check(
        "resolver 400 names the requested model",
        body.error.message.includes("model-b"),
        JSON.stringify(body),
      );
      check(
        "resolver 400 names the active model",
        body.error.message.includes("model-a"),
        JSON.stringify(body),
      );
    }
    // Active model -> 200, and the done record matches the response id.
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "model-a",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      const body = (await res.json()) as {
        id: string;
        choices: { message: { content: string } }[];
        usage: { prompt_tokens: number; completion_tokens: number };
      };
      check("resolver chat for active model is 200", res.status === 200, `got ${res.status}`);
      check(
        "resolver chat returns HHH",
        body.choices[0]?.message.content === "HHH",
        JSON.stringify(body),
      );
      const rec = admission.records[admission.records.length - 1];
      check(
        "resolver chat records a done record",
        rec?.status === "done",
        JSON.stringify(rec),
      );
      check(
        "resolver done record id matches the response id",
        rec?.id === body.id,
        `${rec?.id} vs ${body.id}`,
      );
      check(
        "resolver done record counts tokens",
        rec?.promptTokens === body.usage.prompt_tokens &&
          rec?.completionTokens === body.usage.completion_tokens,
        JSON.stringify(rec),
      );
      check(
        "resolver done record has a ttftMs number",
        typeof rec?.ttftMs === "number" && (rec?.ttftMs ?? -1) >= 0,
        JSON.stringify(rec),
      );
      check(
        "resolver done record endpoint is chat completions",
        rec?.endpoint === "/v1/chat/completions",
        JSON.stringify(rec),
      );
    }
    // Switch the active model between requests.
    resolver.activeId = "model-b";
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "model-b",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      check("resolver chat after switch to model-b is 200", res.status === 200, `got ${res.status}`);
      const stale = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "model-a",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      const staleBody = (await stale.json()) as { error: { message: string } };
      check("resolver chat for now-inactive model-a is 400", stale.status === 400, `got ${stale.status}`);
      check(
        "resolver 400 after switch names both models",
        staleBody.error.message.includes("model-a") &&
          staleBody.error.message.includes("model-b"),
        JSON.stringify(staleBody),
      );
    }
    // Server status reflects the active model.
    {
      const res = await fetch(`${b}/v1/server/status`);
      const body = (await res.json()) as ServerStatus;
      check("resolver GET /v1/server/status is 200", res.status === 200, `got ${res.status}`);
      check(
        "resolver server status activeModelId is model-b",
        body.activeModelId === "model-b",
        JSON.stringify(body.activeModelId),
      );
    }
    // Readiness: controllable both ways.
    {
      resolver.ready = false;
      const notReady = await fetch(`${b}/readyz`);
      const notReadyBody = (await notReady.json()) as { ready: boolean };
      check("resolver GET /readyz not ready is 503", notReady.status === 503, `got ${notReady.status}`);
      check("resolver GET /readyz not ready body", notReadyBody.ready === false, JSON.stringify(notReadyBody));
      resolver.ready = true;
      const ready = await fetch(`${b}/readyz`);
      const readyBody = (await ready.json()) as { ready: boolean };
      check("resolver GET /readyz ready is 200", ready.status === 200, `got ${ready.status}`);
      check("resolver GET /readyz ready body", readyBody.ready === true, JSON.stringify(readyBody));
    }
    // Health: controllable both ways.
    {
      const ok = await fetch(`${b}/healthz`);
      const okBody = (await ok.json()) as HealthStatus;
      check("resolver GET /healthz ok is 200", ok.status === 200, `got ${ok.status}`);
      check("resolver GET /healthz ok status", okBody.status === "ok", JSON.stringify(okBody));
      resolver.healthy = false;
      const down = await fetch(`${b}/healthz`);
      const downBody = (await down.json()) as HealthStatus;
      check("resolver GET /healthz down is 503", down.status === 503, `got ${down.status}`);
      check("resolver GET /healthz down status", downBody.status === "down", JSON.stringify(downBody));
      resolver.healthy = true;
    }
    // No active model -> 503, but the model list still works.
    resolver.activeId = null;
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "model-a",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      const body = (await res.json()) as { error: { message: string; type: string } };
      check("resolver chat with no active model is 503", res.status === 503, `got ${res.status}`);
      check(
        "resolver 503 message is no model is active",
        body.error.message === "no model is active",
        JSON.stringify(body),
      );
      check(
        "resolver 503 type is server_error",
        body.error.type === "server_error",
        JSON.stringify(body),
      );
      const models = await fetch(`${b}/v1/models`);
      check("resolver GET /v1/models still 200 with no active model", models.status === 200, `got ${models.status}`);
    }
  } finally {
    await srv.close();
  }
}

// --- model management: POST /v1/models/load, :id/unload, :id/activate ------

{
  const admission = new FakeAdmission();
  const resolver = new FakeServingResolver(new FakeScheduler());
  const srv = new SoftwellServer({
    getCluster: () => snapshot,
    serving: resolver,
    admission,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  const post = (path: string, body: unknown) =>
    fetch(`${b}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  try {
    // Load a new model.
    {
      const res = await post("/v1/models/load", { ref: "model-c" });
      const body = (await res.json()) as ModelEntry;
      check("POST /v1/models/load is 200", res.status === 200, `got ${res.status}`);
      check(
        "POST /v1/models/load returns the entry",
        body.id === "model-c" && body.status === "ready",
        JSON.stringify(body),
      );
    }
    // Load validation: missing/empty ref, bad JSON.
    {
      const missing = await post("/v1/models/load", {});
      check("POST /v1/models/load without ref is 400", missing.status === 400, `got ${missing.status}`);
      const empty = await post("/v1/models/load", { ref: "  " });
      check("POST /v1/models/load with empty ref is 400", empty.status === 400, `got ${empty.status}`);
      const badJson = await fetch(`${b}/v1/models/load`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{not json",
      });
      check("POST /v1/models/load with bad JSON is 400", badJson.status === 400, `got ${badJson.status}`);
    }
    // Unknown ref -> 404; model that does not fit -> 422 with error entry.
    {
      const unknown = await post("/v1/models/load", { ref: "missing-model" });
      check("POST /v1/models/load unknown ref is 404", unknown.status === 404, `got ${unknown.status}`);
      const big = await post("/v1/models/load", { ref: "too-big" });
      const bigBody = (await big.json()) as ModelEntry;
      check("POST /v1/models/load too-big is 422", big.status === 422, `got ${big.status}`);
      check(
        "POST /v1/models/load too-big entry is error",
        bigBody.status === "error" && typeof bigBody.error === "string",
        JSON.stringify(bigBody),
      );
    }
    // Unload: unknown -> 404, active -> 409, staged -> 200 then gone.
    {
      const unknown = await post("/v1/models/model-z/unload", {});
      check("POST /v1/models/:id/unload unknown is 404", unknown.status === 404, `got ${unknown.status}`);
      const active = await post("/v1/models/model-a/unload", {});
      check("POST /v1/models/:id/unload active is 409", active.status === 409, `got ${active.status}`);
      const ok = await post("/v1/models/model-b/unload", {});
      const okBody = (await ok.json()) as { id: string; status: string };
      check("POST /v1/models/:id/unload staged is 200", ok.status === 200, `got ${ok.status}`);
      check(
        "POST /v1/models/:id/unload body",
        okBody.id === "model-b" && okBody.status === "available",
        JSON.stringify(okBody),
      );
      const again = await post("/v1/models/model-b/unload", {});
      check("POST /v1/models/:id/unload twice is 404", again.status === 404, `got ${again.status}`);
    }
    // Activate: unknown -> 404, not-ready -> 409, success switches the
    // active model (chat for the old model is now 400, for the new 200).
    {
      const unknown = await post("/v1/models/model-z/activate", {});
      check("POST /v1/models/:id/activate unknown is 404", unknown.status === 404, `got ${unknown.status}`);
      const notReady = await post("/v1/models/model-not-ready/activate", {});
      check("POST /v1/models/:id/activate not-ready is 409", notReady.status === 409, `got ${notReady.status}`);
      const ok = await post("/v1/models/model-c/activate", {});
      const okBody = (await ok.json()) as { id: string };
      check("POST /v1/models/:id/activate is 200", ok.status === 200, `got ${ok.status}`);
      check(
        "POST /v1/models/:id/activate body names the model",
        okBody.id === "model-c",
        JSON.stringify(okBody),
      );
      const oldChat = await post("/v1/chat/completions", {
        model: "model-a",
        messages: [{ role: "user", content: "hi" }],
      });
      check("chat for the old active model is now 400", oldChat.status === 400, `got ${oldChat.status}`);
      const newChat = await post("/v1/chat/completions", {
        model: "model-c",
        messages: [{ role: "user", content: "hi" }],
      });
      check("chat for the newly active model is 200", newChat.status === 200, `got ${newChat.status}`);
    }
  } finally {
    await srv.close();
  }
}

// --- model management without a resolver: legacy form is 404 ---------------

{
  const srv = new SoftwellServer({
    scheduler: new FakeScheduler(),
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    const load = await fetch(`${b}/v1/models/load`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ref: "model-a" }),
    });
    check("legacy POST /v1/models/load is 404", load.status === 404, `got ${load.status}`);
    const unload = await fetch(`${b}/v1/models/model-a/unload`, { method: "POST" });
    check("legacy POST /v1/models/:id/unload is 404", unload.status === 404, `got ${unload.status}`);
    const activate = await fetch(`${b}/v1/models/model-a/activate`, { method: "POST" });
    check("legacy POST /v1/models/:id/activate is 404", activate.status === 404, `got ${activate.status}`);
  } finally {
    await srv.close();
  }
}

// --- admission rejection ----------------------------------------------------

{
  const admission = new FakeAdmission();
  admission.reject = { status: 429, message: "queue is full" };
  const srv = new SoftwellServer({
    scheduler: new FakeScheduler(),
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
    admission,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    const res = await fetch(`${b}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "test-model",
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    const body = (await res.json()) as { error: { message: string; type: string } };
    check("admission rejection is 429", res.status === 429, `got ${res.status}`);
    check(
      "admission rejection message is the admission message",
      body.error.message === "queue is full",
      JSON.stringify(body),
    );
    check(
      "admission rejection type is invalid_request_error",
      body.error.type === "invalid_request_error",
      JSON.stringify(body),
    );
    check(
      "admission rejection recorded one record",
      admission.records.length === 1,
      JSON.stringify(admission.records),
    );
    const rec = admission.records[0];
    check(
      "admission rejection record status is rejected",
      rec?.status === "rejected",
      JSON.stringify(rec),
    );
    check(
      "admission rejection record has the model and endpoint",
      rec?.modelId === "test-model" && rec?.endpoint === "/v1/chat/completions",
      JSON.stringify(rec),
    );
    check(
      "admission rejection record has the error message",
      rec?.error === "queue is full",
      JSON.stringify(rec),
    );
  } finally {
    await srv.close();
  }
}

// --- timeout (non-streaming 408 + streaming error chunk) --------------------

{
  const admission = new FakeAdmission();
  const scheduler = new ConfigurableScheduler(0, 3, false); // never produces
  const srv = new SoftwellServer({
    scheduler,
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
    admission,
    requestTimeoutMs: 50,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "test-model",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
      const body = (await res.json()) as { error: { message: string; type: string } };
      check("timeout non-streaming is 408", res.status === 408, `got ${res.status}`);
      check(
        "timeout message is request timed out",
        body.error.message === "request timed out",
        JSON.stringify(body),
      );
      check(
        "timeout type is timeout",
        body.error.type === "timeout",
        JSON.stringify(body),
      );
      check(
        "timeout cancelled the sequence",
        scheduler.cancelCalls.includes("seq-1"),
        JSON.stringify(scheduler.cancelCalls),
      );
      const rec = admission.records[admission.records.length - 1];
      check(
        "timeout recorded an error record",
        rec?.status === "error" && rec?.error === "request timed out",
        JSON.stringify(rec),
      );
    }
    {
      const res = await fetch(`${b}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "test-model",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      });
      const text = await res.text();
      check("timeout streaming is 200 (headers already sent)", res.status === 200, `got ${res.status}`);
      check(
        "timeout streaming ends with an error chunk",
        text.includes("request timed out"),
        text,
      );
      check(
        "timeout streaming ends with [DONE]",
        text.trimEnd().endsWith("data: [DONE]"),
        text,
      );
      const rec = admission.records[admission.records.length - 1];
      check(
        "timeout streaming recorded an error record",
        rec?.status === "error",
        JSON.stringify(rec),
      );
    }
  } finally {
    await srv.close();
  }
}

// --- cancellation on client disconnect --------------------------------------

{
  const admission = new FakeAdmission();
  const scheduler = new ConfigurableScheduler(50, 100); // slow, never finishes
  const srv = new SoftwellServer({
    scheduler,
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
    admission,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    const controller = new AbortController();
    const res = await fetch(`${b}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "test-model",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
      signal: controller.signal,
    });
    check("cancellation stream starts 200", res.status === 200, `got ${res.status}`);
    const reader = res.body?.getReader();
    let firstChunk = "";
    if (reader !== undefined) {
      const first = await reader.read();
      firstChunk = new TextDecoder().decode(first.value);
      // Disconnect mid-stream.
      controller.abort();
      try {
        await reader.read();
      } catch {
        // Aborted reads reject; that is the disconnect under test.
      }
      try {
        reader.releaseLock();
      } catch {
        // Already released by the abort.
      }
    }
    check(
      "cancellation first chunk arrived before disconnect",
      firstChunk.includes("data: "),
      firstChunk,
    );
    await sleep(300);
    check(
      "cancellation called scheduler.cancel with the sequence id",
      scheduler.cancelCalls.includes("seq-1"),
      JSON.stringify(scheduler.cancelCalls),
    );
    const rec = admission.records[admission.records.length - 1];
    check(
      "cancellation recorded a cancelled record",
      rec?.status === "cancelled",
      JSON.stringify(rec),
    );
    check(
      "cancellation record is for the same sequence's completion",
      rec?.endpoint === "/v1/chat/completions" && rec?.modelId === "test-model",
      JSON.stringify(rec),
    );
  } finally {
    await srv.close();
  }
}

// --- incremental streaming: first chunk well before the response ends -------

{
  const srv = new SoftwellServer({
    scheduler: new ConfigurableScheduler(50, 3), // one token every 50 ms
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    const started = Date.now();
    const res = await fetch(`${b}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "test-model",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    check("incremental streaming is 200", res.status === 200, `got ${res.status}`);
    const reader = res.body?.getReader();
    let firstAt = -1;
    let text = "";
    if (reader !== undefined) {
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (firstAt < 0) firstAt = Date.now() - started;
        text += decoder.decode(value, { stream: true });
      }
      text += new TextDecoder().decode();
    }
    const totalMs = Date.now() - started;
    check(
      "incremental streaming first chunk arrives before completion",
      firstAt >= 0 && firstAt < totalMs,
      `firstAt=${firstAt} totalMs=${totalMs}`,
    );
    check(
      "incremental streaming leaves at least 50 ms after the first chunk",
      firstAt >= 0 && totalMs - firstAt >= 50,
      `firstAt=${firstAt} totalMs=${totalMs}`,
    );
    check(
      "incremental streaming total takes at least 100 ms (3 x 50 ms tokens)",
      totalMs >= 100,
      `totalMs=${totalMs}`,
    );
    const streamed = text
      .split("\n\n")
      .filter((e) => e.startsWith("data: ") && !e.includes("[DONE]"))
      .map((e) => JSON.parse(e.slice("data: ".length)) as {
        choices: { delta: { content?: string } }[];
      })
      .map((c) => c.choices[0]?.delta.content ?? "")
      .join("");
    check("incremental streaming content is HHH", streamed === "HHH", JSON.stringify(streamed));
  } finally {
    await srv.close();
  }
}

// --- shutdown with a hook ----------------------------------------------------

{
  let shutdownCalled: boolean = false;
  const srv = new SoftwellServer({
    scheduler: new FakeScheduler(),
    tokenizer: fakeTokenizer,
    modelId: "test-model",
    getCluster: () => snapshot,
    onShutdown: () => {
      shutdownCalled = true;
    },
  });
  const p = await srv.start();
  const b = `http://127.0.0.1:${p}`;
  try {
    const res = await fetch(`${b}/v1/server/shutdown`, { method: "POST" });
    const body = (await res.json()) as { shutting_down: boolean };
    check("POST /v1/server/shutdown is 202", res.status === 202, `got ${res.status}`);
    check(
      "POST /v1/server/shutdown body is shutting_down",
      body.shutting_down === true,
      JSON.stringify(body),
    );
    check("POST /v1/server/shutdown called onShutdown", shutdownCalled);
  } finally {
    await srv.close();
  }
}

if (failures.length > 0) {
  console.log(`\n${failures.length} failure(s): ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nAll API smoke tests passed.");
