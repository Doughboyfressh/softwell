/**
 * softwell-machine — HTTP API server (OpenAI-compatible) + dashboard host.
 *
 * Node built-ins only (node:http, node:fs). The server takes its
 * dependencies injected — the CLI wires the real scheduler, tokenizer and
 * cluster snapshot; this module never imports the mesh registry, planner
 * or engine.
 *
 * Two dependency forms:
 *   - Legacy fixed-context form: a fixed scheduler/tokenizer/modelId.
 *   - Resolver form (preferred; wins when `serving` is present): the
 *     active ServingContext is resolved per request from a ServingResolver,
 *     so the active model can switch between requests. An AdmissionControl
 *     gates submissions and records every terminal request outcome.
 *
 * Routes:
 *   GET  /v1/models               -> { object: "list", data: [ModelInfo] }
 *   GET  /v1/models/:id           -> one ModelInfo (404 if unknown)
 *   POST /v1/models/load          -> stage a model (body { ref }); 422 if it does not fit
 *   POST /v1/models/:id/unload    -> free a staged model (409 for the active model)
 *   POST /v1/models/:id/activate  -> make a loaded model the active one
 *   POST /v1/chat/completions     -> JSON or SSE stream
 *   POST /v1/completions          -> JSON or SSE stream (plain text)
 *   GET  /v1/cluster              -> ClusterSnapshot JSON
 *   GET  /healthz                  -> HealthStatus (200 ok/degraded, 503 down)
 *   GET  /readyz                   -> { ready: boolean } (200 / 503)
 *   GET  /v1/server/status        -> ServerStatus (404 in legacy form)
 *   POST /v1/server/shutdown      -> 202, then onShutdown() (404 without one)
 *   GET  /                        -> dashboard (src/dashboard/index.html)
 *   GET  /dashboard/*             -> static files from src/dashboard
 */

import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  AdmissionControl,
  ChatCompletionRequest,
  ClusterSnapshot,
  GeneratedToken,
  GenerateRequest,
  HealthStatus,
  ModelEntry,
  ModelInfo,
  Scheduler,
  ServerRequestRecord,
  ServerRequestStatus,
  ServingResolver,
  Tokenizer,
} from "../contracts.ts";
import {
  chatRequestToGenerate,
  completionsRequestToGenerate,
  formatChunk,
  formatCompletionChunk,
  formatCompletionFinishChunk,
  formatCompletionResponse,
  formatFinalResponse,
  formatFinishChunk,
  formatModel,
  formatModelsList,
  newChatCompletionId,
  newCompletionId,
  sseDone,
  sseEncode,
  type CompletionRequest,
} from "./openai.ts";

export interface SoftwellServerDeps {
  // Legacy fixed-context form (kept: test/api.test.ts uses it):
  scheduler?: Scheduler;
  tokenizer?: Tokenizer;
  modelId?: string;
  getCluster: () => ClusterSnapshot;
  port?: number;
  // Resolver form (preferred; when `serving` is present it wins):
  serving?: ServingResolver;
  admission?: AdmissionControl;
  onShutdown?: () => void | Promise<void>;
  /** Per-request deadline in ms (default 300_000). */
  requestTimeoutMs?: number;
}

const DASHBOARD_DIR = fileURLToPath(new URL("../dashboard/", import.meta.url));
const MAX_BODY_BYTES = 10 * 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT_MS = 300_000;
const TIMEOUT_MESSAGE = "request timed out";

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sendJson(
  res: http.ServerResponse,
  status: number,
  body: unknown,
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: http.ServerResponse, status: number, message: string): void {
  sendJson(res, status, { error: { message, type: "invalid_request_error" } });
}

function sendErrorTyped(
  res: http.ServerResponse,
  status: number,
  message: string,
  type: string,
): void {
  sendJson(res, status, { error: { message, type } });
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on("data", (chunk: Buffer | string) => {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      size += buf.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(buf);
    });
    req.on("end", () => {
      if (tooLarge) {
        reject(new Error("request body too large"));
        return;
      }
      resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", reject);
  });
}

function isValidChatRequest(body: unknown): body is ChatCompletionRequest {
  if (typeof body !== "object" || body === null) return false;
  const req = body as ChatCompletionRequest;
  if (!Array.isArray(req.messages) || req.messages.length === 0) return false;
  return req.messages.every(
    (m) =>
      typeof m === "object" &&
      m !== null &&
      typeof m.role === "string" &&
      typeof m.content === "string",
  );
}

function isValidCompletionRequest(body: unknown): body is CompletionRequest {
  if (typeof body !== "object" || body === null) return false;
  const req = body as CompletionRequest;
  if (typeof req.prompt === "string") return true;
  if (Array.isArray(req.prompt)) {
    return (
      req.prompt.length > 0 && req.prompt.every((p) => typeof p === "string")
    );
  }
  return false;
}

/** What one generation request needs from whichever context serves it. */
interface ServingContextLike {
  modelId: string;
  scheduler: Scheduler;
  tokenizer: Tokenizer;
}

/** One submitted generation request, tracked for recording/timeout. */
interface ActiveRequest {
  id: string;
  modelId: string;
  endpoint: "/v1/chat/completions" | "/v1/completions";
  startedAt: number;
  created: number;
  promptTokens: number;
  seqId: string;
  submitAt: number;
  deadlineMs: number;
}

/** How to format one generation's output (chat vs plain-text completions). */
interface GenerationFormat {
  chunk(
    id: string,
    model: string,
    created: number,
    token: GeneratedToken,
    index: number,
  ): object;
  finishChunk(
    id: string,
    model: string,
    created: number,
    finishReason: string | null,
    index: number,
  ): object;
  final(
    id: string,
    model: string,
    created: number,
    content: string,
    promptTokens: number,
    completionTokens: number,
    finishReason: string | null,
  ): object;
}

const CHAT_FORMAT: GenerationFormat = {
  chunk: formatChunk,
  finishChunk: formatFinishChunk,
  final: formatFinalResponse,
};

const COMPLETION_FORMAT: GenerationFormat = {
  chunk: formatCompletionChunk,
  finishChunk: formatCompletionFinishChunk,
  final: formatCompletionResponse,
};

export class SoftwellServer {
  private scheduler?: Scheduler;
  private tokenizer?: Tokenizer;
  private modelId?: string;
  private getCluster: () => ClusterSnapshot;
  private port: number;
  private serving?: ServingResolver;
  private admission?: AdmissionControl;
  private onShutdown?: () => void | Promise<void>;
  private requestTimeoutMs: number;
  private server: http.Server | null = null;

  constructor(deps: SoftwellServerDeps) {
    this.scheduler = deps.scheduler;
    this.tokenizer = deps.tokenizer;
    this.modelId = deps.modelId;
    this.getCluster = deps.getCluster;
    this.port = deps.port ?? 0;
    this.serving = deps.serving;
    this.admission = deps.admission;
    this.onShutdown = deps.onShutdown;
    this.requestTimeoutMs = deps.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  }

  /** Start listening; resolves with the actual port (ephemeral if 0). */
  start(): Promise<number> {
    return new Promise((resolve, reject) => {
      const server = http.createServer((req, res) => {
        void this.handle(req, res).catch((err: unknown) => {
          const message = err instanceof Error ? err.message : "internal error";
          if (!res.headersSent) sendError(res, 500, message);
          else res.destroy();
        });
      });
      server.once("error", reject);
      server.listen(this.port, () => {
        this.server = server;
        const addr = server.address();
        const actualPort =
          typeof addr === "object" && addr !== null ? addr.port : this.port;
        resolve(actualPort);
      });
    });
  }

  /** Stop listening and close all connections. */
  close(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server === null) {
        resolve();
        return;
      }
      const server = this.server;
      this.server = null;
      server.close(() => resolve());
    });
  }

  private modelInfo(): ModelInfo {
    return {
      id: this.modelId ?? "",
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "softwell",
    };
  }

  private listModels(): ModelInfo[] {
    if (this.serving !== undefined) return this.serving.listModels();
    if (this.modelId === undefined) return [];
    return [this.modelInfo()];
  }

  /**
   * Resolve the context for one request. Resolver form: the active context
   * (null when no model is active). Legacy form: the fixed context (null
   * when it was never configured).
   */
  private resolveContext(): ServingContextLike | null {
    if (this.serving !== undefined) {
      const active = this.serving.getActive();
      if (active === null) return null;
      return {
        modelId: active.modelId,
        scheduler: active.scheduler,
        tokenizer: active.tokenizer,
      };
    }
    if (
      this.scheduler === undefined ||
      this.tokenizer === undefined ||
      this.modelId === undefined
    ) {
      return null;
    }
    return {
      modelId: this.modelId,
      scheduler: this.scheduler,
      tokenizer: this.tokenizer,
    };
  }

  private record(record: ServerRequestRecord): void {
    if (this.admission !== undefined) this.admission.record(record);
  }

  private recordTerminal(
    active: ActiveRequest,
    status: ServerRequestStatus,
    completionTokens: number,
    ttftMs: number | undefined,
    error?: string,
  ): void {
    this.record({
      id: active.id,
      modelId: active.modelId,
      endpoint: active.endpoint,
      startedAt: active.startedAt,
      endedAt: Date.now(),
      status,
      promptTokens: active.promptTokens,
      completionTokens,
      ttftMs,
      error,
    });
  }

  private async handle(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const pathname = url.pathname;
    const method = req.method ?? "GET";

    if (method === "GET" && pathname === "/v1/models") {
      sendJson(res, 200, formatModelsList(this.listModels()));
      return;
    }

    if (method === "GET" && pathname.startsWith("/v1/models/")) {
      const id = decodeURIComponent(pathname.slice("/v1/models/".length));
      const found = this.listModels().find((m) => m.id === id);
      if (found !== undefined) {
        sendJson(res, 200, formatModel(found));
      } else {
        sendError(res, 404, `model not found: ${id}`);
      }
      return;
    }

    if (method === "POST" && pathname === "/v1/models/load") {
      await this.handleModelLoad(req, res);
      return;
    }

    if (method === "POST" && pathname.startsWith("/v1/models/")) {
      const rest = pathname.slice("/v1/models/".length);
      const slash = rest.indexOf("/");
      if (slash > 0) {
        const id = decodeURIComponent(rest.slice(0, slash));
        const action = rest.slice(slash + 1);
        if (action === "unload") {
          await this.handleModelUnload(res, id);
          return;
        }
        if (action === "activate") {
          await this.handleModelActivate(res, id);
          return;
        }
      }
      sendError(res, 404, `not found: ${pathname}`);
      return;
    }

    if (method === "POST" && pathname === "/v1/chat/completions") {
      await this.handleChatCompletions(req, res);
      return;
    }

    if (method === "POST" && pathname === "/v1/completions") {
      await this.handleCompletions(req, res);
      return;
    }

    if (method === "GET" && pathname === "/healthz") {
      this.handleHealthz(res);
      return;
    }

    if (method === "GET" && pathname === "/readyz") {
      this.handleReadyz(res);
      return;
    }

    if (method === "GET" && pathname === "/v1/server/status") {
      this.handleServerStatus(res);
      return;
    }

    if (method === "POST" && pathname === "/v1/server/shutdown") {
      this.handleShutdown(res);
      return;
    }

    if (method === "GET" && pathname === "/v1/cluster") {
      sendJson(res, 200, this.getCluster());
      return;
    }

    if (method === "GET" && (pathname === "/" || pathname.startsWith("/dashboard/"))) {
      await this.handleDashboard(pathname, res);
      return;
    }

    sendError(res, 404, `not found: ${pathname}`);
  }

  private handleHealthz(res: http.ServerResponse): void {
    if (this.serving !== undefined) {
      const health: HealthStatus = this.serving.getHealth();
      sendJson(res, health.status === "down" ? 503 : 200, health);
      return;
    }
    // Legacy fixed-model form: there is no VM to boot, so be honest about
    // that — the server itself is up and a model is configured.
    sendJson(res, 200, {
      status: "ok",
      checks: { vmBooted: false, activeModel: true, workersHealthy: true },
    });
  }

  private handleReadyz(res: http.ServerResponse): void {
    const ready = this.serving !== undefined ? this.serving.isReady() : true;
    if (ready) {
      sendJson(res, 200, { ready: true });
    } else {
      sendJson(res, 503, { ready: false });
    }
  }

  private handleServerStatus(res: http.ServerResponse): void {
    if (this.serving === undefined) {
      sendError(
        res,
        404,
        "server status is not available: this server was started in the " +
          "legacy fixed-model form without a serving resolver",
      );
      return;
    }
    sendJson(res, 200, this.serving.getServerStatus());
  }

  private handleShutdown(res: http.ServerResponse): void {
    if (this.onShutdown === undefined) {
      sendError(
        res,
        404,
        "shutdown is not available: no shutdown handler is configured",
      );
      return;
    }
    sendJson(res, 202, { shutting_down: true });
    // Respond first, then shut down. Synchronous invocation (after res.end)
    // keeps the ordering deterministic for callers awaiting the 202.
    try {
      const result = this.onShutdown();
      if (result instanceof Promise) {
        result.catch(() => undefined);
      }
    } catch {
      // Already responded; a failing shutdown hook must not rewrite it.
    }
  }

  /**
   * Map a model-management throw to an HTTP status. The registry and
   * the server throw plain Errors: something unknown or missing is
   * 404, a malformed reference is 400, refusing the active model or a
   * model that is not ready is 409, anything else is 500.
   */
  private modelErrorStatus(err: unknown): number {
    const message = err instanceof Error ? err.message : String(err);
    if (/unknown model|not found|ENOENT|no such file/i.test(message)) return 404;
    if (/invalid|empty model reference/i.test(message)) return 400;
    if (/active model|not ready|still loading/i.test(message)) return 409;
    return 500;
  }

  private static readonly MODEL_MANAGEMENT_LEGACY_MESSAGE =
    "model management is not available: this server was started in the " +
    "legacy fixed-model form without a serving resolver";

  /**
   * POST /v1/models/load — stage a model into the running server.
   * Body: { ref } where ref is a local directory, a loaded model id,
   * or a Hugging Face repo id. A model that does not fit is a handled
   * outcome, not a crash: the entry comes back with status "error"
   * and the reason, under 422.
   */
  private async handleModelLoad(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
    const serving = this.serving;
    if (serving === undefined) {
      sendError(res, 404, SoftwellServer.MODEL_MANAGEMENT_LEGACY_MESSAGE);
      return;
    }
    if (serving.loadModel === undefined) {
      sendError(res, 404, "model loading is not supported by this server");
      return;
    }
    let raw: string;
    try {
      raw = await readBody(req);
    } catch (err) {
      sendError(res, 400, err instanceof Error ? err.message : "bad request");
      return;
    }
    let body: unknown;
    try {
      body = JSON.parse(raw) as unknown;
    } catch {
      sendError(res, 400, "request body is not valid JSON");
      return;
    }
    const ref = (body as { ref?: unknown } | null)?.ref;
    if (typeof ref !== "string" || ref.trim().length === 0) {
      sendError(
        res,
        400,
        "body.ref must be a non-empty string (local directory, model id, " +
          "or Hugging Face repo id)",
      );
      return;
    }
    try {
      const entry: ModelEntry = await serving.loadModel(ref);
      sendJson(res, entry.status === "error" ? 422 : 200, entry);
    } catch (err) {
      sendError(
        res,
        this.modelErrorStatus(err),
        err instanceof Error ? err.message : "model load failed",
      );
    }
  }

  /** POST /v1/models/:id/unload — free a staged model. */
  private async handleModelUnload(
    res: http.ServerResponse,
    id: string,
  ): Promise<void> {
    const serving = this.serving;
    if (serving === undefined) {
      sendError(res, 404, SoftwellServer.MODEL_MANAGEMENT_LEGACY_MESSAGE);
      return;
    }
    if (serving.unloadModel === undefined) {
      sendError(res, 404, "model unloading is not supported by this server");
      return;
    }
    try {
      await serving.unloadModel(id);
      sendJson(res, 200, { id, status: "available" });
    } catch (err) {
      sendError(
        res,
        this.modelErrorStatus(err),
        err instanceof Error ? err.message : "model unload failed",
      );
    }
  }

  /** POST /v1/models/:id/activate — make a loaded model the active one. */
  private async handleModelActivate(
    res: http.ServerResponse,
    id: string,
  ): Promise<void> {
    const serving = this.serving;
    if (serving === undefined) {
      sendError(res, 404, SoftwellServer.MODEL_MANAGEMENT_LEGACY_MESSAGE);
      return;
    }
    if (serving.activateModel === undefined) {
      sendError(res, 404, "model activation is not supported by this server");
      return;
    }
    try {
      await serving.activateModel(id);
      const info = this.listModels().find((m) => m.id === id);
      sendJson(
        res,
        200,
        info !== undefined ? formatModel(info) : { id, object: "model" },
      );
    } catch (err) {
      sendError(
        res,
        this.modelErrorStatus(err),
        err instanceof Error ? err.message : "model activation failed",
      );
    }
  }

  private async handleDashboard(
    pathname: string,
    res: http.ServerResponse,
  ): Promise<void> {
    const rel = pathname === "/" ? "index.html" : pathname.slice("/dashboard/".length);
    const resolved = path.resolve(DASHBOARD_DIR, decodeURIComponent(rel));
    if (!resolved.startsWith(path.resolve(DASHBOARD_DIR))) {
      sendError(res, 403, "forbidden");
      return;
    }
    try {
      const data = await readFile(resolved);
      const ext = path.extname(resolved);
      res.writeHead(200, {
        "content-type": MIME[ext] ?? "application/octet-stream",
        "content-length": data.length,
      });
      res.end(data);
    } catch {
      sendError(res, 404, `not found: ${pathname}`);
    }
  }

  /**
   * Shared request prologue for generation endpoints: read + parse the body,
   * resolve the serving context, and enforce the active-model check.
   * Returns null after already responding (parse/validation/503/400).
   */
  private async readGenerationRequest(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    isValid: (body: unknown) => boolean,
    invalidMessage: string,
  ): Promise<{ body: unknown; ctx: ServingContextLike } | null> {
    let raw: string;
    try {
      raw = await readBody(req);
    } catch (err) {
      sendError(res, 400, err instanceof Error ? err.message : "bad request");
      return null;
    }
    let body: unknown;
    try {
      body = JSON.parse(raw) as unknown;
    } catch {
      sendError(res, 400, "request body is not valid JSON");
      return null;
    }
    if (!isValid(body)) {
      sendError(res, 400, invalidMessage);
      return null;
    }
    const ctx = this.resolveContext();
    if (ctx === null) {
      sendErrorTyped(res, 503, "no model is active", "server_error");
      return null;
    }
    // Resolver form only: a named model must be the active one.
    const requestedModel = (body as { model?: unknown }).model;
    if (
      this.serving !== undefined &&
      typeof requestedModel === "string" &&
      requestedModel.length > 0 &&
      requestedModel !== ctx.modelId
    ) {
      sendError(
        res,
        400,
        `model "${requestedModel}" is not active; active model is "${ctx.modelId}"`,
      );
      return null;
    }
    return { body, ctx };
  }

  private async handleChatCompletions(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
    const startedAt = Date.now();
    const parsed = await this.readGenerationRequest(
      req,
      res,
      isValidChatRequest,
      "request must include a non-empty messages array",
    );
    if (parsed === null) return;
    const body = parsed.body as ChatCompletionRequest;
    const ctx = parsed.ctx;
    const model = body.model || ctx.modelId;
    const genReq = chatRequestToGenerate(body, ctx.tokenizer);
    const id = newChatCompletionId();
    const created = Math.floor(Date.now() / 1000);
    await this.executeGeneration(
      req,
      res,
      ctx,
      genReq,
      "/v1/chat/completions",
      id,
      created,
      startedAt,
      model,
      body.stream === true,
      CHAT_FORMAT,
    );
  }

  private async handleCompletions(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
    const startedAt = Date.now();
    const parsed = await this.readGenerationRequest(
      req,
      res,
      isValidCompletionRequest,
      "request must include a prompt string or a non-empty array of prompt strings",
    );
    if (parsed === null) return;
    const body = parsed.body as CompletionRequest;
    const ctx = parsed.ctx;
    const model = body.model || ctx.modelId;
    const genReq = completionsRequestToGenerate(body, ctx.tokenizer);
    const id = newCompletionId();
    const created = Math.floor(Date.now() / 1000);
    await this.executeGeneration(
      req,
      res,
      ctx,
      genReq,
      "/v1/completions",
      id,
      created,
      startedAt,
      model,
      body.stream === true,
      COMPLETION_FORMAT,
    );
  }

  /**
   * Admission-gate, submit, then drive the sequence to a terminal outcome
   * (streaming or not). Every terminal outcome of a submitted request is
   * recorded via admission.record() when admission is configured.
   */
  private async executeGeneration(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    ctx: ServingContextLike,
    genReq: GenerateRequest,
    endpoint: "/v1/chat/completions" | "/v1/completions",
    id: string,
    created: number,
    startedAt: number,
    model: string,
    stream: boolean,
    format: GenerationFormat,
  ): Promise<void> {
    if (this.admission !== undefined) {
      const decision = this.admission.admit(model);
      if (!decision.ok) {
        sendErrorTyped(res, decision.status, decision.message, "invalid_request_error");
        this.record({
          id,
          modelId: model,
          endpoint,
          startedAt,
          endedAt: Date.now(),
          status: "rejected",
          promptTokens: genReq.tokenIds.length,
          completionTokens: 0,
          error: decision.message,
        });
        return;
      }
    }
    const submitAt = Date.now();
    const seqId = ctx.scheduler.submit(genReq);
    const active: ActiveRequest = {
      id,
      modelId: model,
      endpoint,
      startedAt,
      created,
      promptTokens: genReq.tokenIds.length,
      seqId,
      submitAt,
      deadlineMs: submitAt + this.requestTimeoutMs,
    };
    if (stream) {
      await this.streamGeneration(req, res, ctx, active, genReq, format);
    } else {
      await this.completeGeneration(req, res, ctx, active, genReq, format);
    }
  }

  /**
   * Drive the scheduler until the sequence finishes, collecting tokens.
   * Returns the generated token ids, the finish reason, and whether the
   * run ended by timeout or by client disconnect. On timeout the sequence
   * is cancelled before returning.
   */
  private async runUntilDone(
    scheduler: Scheduler,
    seqId: string,
    maxTokens: number,
    deadlineMs: number,
    onToken: (token: GeneratedToken) => void,
    isDisconnected: () => boolean,
  ): Promise<{
    tokenIds: number[];
    finishReason: string | null;
    timedOut: boolean;
    disconnected: boolean;
  }> {
    const tokenIds: number[] = [];
    let finishReason: string | null = null;
    let timedOut = false;
    // Generous safety cap: prefill steps may produce no token for this
    // sequence, so allow well past maxTokens iterations.
    const maxSteps = maxTokens + 1000;
    for (let step = 0; step < maxSteps; step += 1) {
      if (Date.now() >= deadlineMs) {
        timedOut = true;
        scheduler.cancel(seqId);
        break;
      }
      const seq = scheduler.getSequence(seqId);
      if (seq === undefined) break;
      if (seq.status === "done") {
        finishReason = seq.finishReason ?? "stop";
        break;
      }
      if (seq.status === "error") {
        finishReason = "error";
        break;
      }
      if (isDisconnected()) break;
      const tokens = await scheduler.step();
      if (Date.now() >= deadlineMs) {
        timedOut = true;
        scheduler.cancel(seqId);
        break;
      }
      let produced = false;
      for (const token of tokens) {
        if (token.seqId !== seqId) continue;
        produced = true;
        tokenIds.push(token.tokenId);
        onToken(token);
        if (token.finishReason !== null) finishReason = token.finishReason;
      }
      const after = scheduler.getSequence(seqId);
      if (after !== undefined && after.status === "done") {
        finishReason = after.finishReason ?? finishReason ?? "stop";
        break;
      }
      if (after !== undefined && after.status === "error") {
        finishReason = "error";
        break;
      }
      // Don't busy-spin: yield when a step produced nothing for us.
      if (!produced) await sleep(0);
    }
    return { tokenIds, finishReason, timedOut, disconnected: isDisconnected() };
  }

  private async streamGeneration(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    ctx: ServingContextLike,
    active: ActiveRequest,
    genReq: GenerateRequest,
    format: GenerationFormat,
  ): Promise<void> {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    let disconnected = false;
    const onClose = (): void => {
      disconnected = true;
    };
    req.on("close", onClose);
    res.on("close", onClose);

    let firstTokenAt: number | null = null;
    const result = await this.runUntilDone(
      ctx.scheduler,
      active.seqId,
      genReq.sampling.maxTokens,
      active.deadlineMs,
      (token) => {
        if (firstTokenAt === null) firstTokenAt = Date.now();
        if (!disconnected) {
          res.write(
            sseEncode(format.chunk(active.id, active.modelId, active.created, token, 0)),
          );
        }
      },
      () => disconnected,
    );
    const ttftMs =
      firstTokenAt === null ? undefined : firstTokenAt - active.submitAt;
    const completionTokens = result.tokenIds.length;

    if (result.timedOut) {
      this.recordTerminal(active, "error", completionTokens, ttftMs, TIMEOUT_MESSAGE);
      if (!disconnected) {
        if (res.headersSent) {
          res.write(
            sseEncode({
              error: { message: TIMEOUT_MESSAGE, type: "timeout" },
            }),
          );
          res.write(sseDone());
          res.end();
        } else {
          sendErrorTyped(res, 408, TIMEOUT_MESSAGE, "timeout");
        }
      } else {
        res.destroy();
      }
      return;
    }

    if (result.disconnected || disconnected) {
      ctx.scheduler.cancel(active.seqId);
      this.recordTerminal(active, "cancelled", completionTokens, ttftMs, "request cancelled");
      res.destroy();
      return;
    }

    if (result.finishReason === "error") {
      this.recordTerminal(active, "error", completionTokens, ttftMs, "sequence error");
    } else {
      this.recordTerminal(active, "done", completionTokens, ttftMs);
    }
    if (!disconnected) {
      res.write(
        sseEncode(
          format.finishChunk(
            active.id,
            active.modelId,
            active.created,
            result.finishReason ?? "stop",
            0,
          ),
        ),
      );
      res.write(sseDone());
      res.end();
    } else {
      res.destroy();
    }
  }

  private async completeGeneration(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    ctx: ServingContextLike,
    active: ActiveRequest,
    genReq: GenerateRequest,
    format: GenerationFormat,
  ): Promise<void> {
    let disconnected = false;
    const onClose = (): void => {
      disconnected = true;
    };
    req.on("close", onClose);
    res.on("close", onClose);

    let firstTokenAt: number | null = null;
    const result = await this.runUntilDone(
      ctx.scheduler,
      active.seqId,
      genReq.sampling.maxTokens,
      active.deadlineMs,
      () => {
        if (firstTokenAt === null) firstTokenAt = Date.now();
      },
      () => disconnected,
    );
    const ttftMs =
      firstTokenAt === null ? undefined : firstTokenAt - active.submitAt;
    const completionTokens = result.tokenIds.length;

    if (result.timedOut) {
      this.recordTerminal(active, "error", completionTokens, ttftMs, TIMEOUT_MESSAGE);
      if (!disconnected && !res.headersSent) {
        sendErrorTyped(res, 408, TIMEOUT_MESSAGE, "timeout");
      } else if (disconnected) {
        res.destroy();
      }
      return;
    }

    if (result.disconnected || disconnected) {
      ctx.scheduler.cancel(active.seqId);
      this.recordTerminal(active, "cancelled", completionTokens, ttftMs, "request cancelled");
      res.destroy();
      return;
    }

    if (result.finishReason === "error") {
      this.recordTerminal(active, "error", completionTokens, ttftMs, "sequence error");
    } else {
      this.recordTerminal(active, "done", completionTokens, ttftMs);
    }
    const content = ctx.tokenizer.decode(result.tokenIds);
    sendJson(
      res,
      200,
      format.final(
        active.id,
        active.modelId,
        active.created,
        content,
        active.promptTokens,
        completionTokens,
        result.finishReason ?? "stop",
      ),
    );
  }
}
