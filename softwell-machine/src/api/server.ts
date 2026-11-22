/**
 * softwell-machine — HTTP API server (OpenAI-compatible) + dashboard host.
 *
 * Node built-ins only (node:http, node:fs). The server takes its
 * dependencies injected — the CLI wires the real scheduler, tokenizer and
 * cluster snapshot; this module never imports the mesh registry, planner
 * or engine.
 *
 * Routes:
 *   GET  /v1/models               -> { object: "list", data: [ModelInfo] }
 *   GET  /v1/models/:id           -> one ModelInfo (404 if unknown)
 *   POST /v1/chat/completions     -> JSON or SSE stream
 *   GET  /v1/cluster              -> ClusterSnapshot JSON
 *   GET  /                        -> dashboard (src/dashboard/index.html)
 *   GET  /dashboard/*             -> static files from src/dashboard
 */

import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  ChatCompletionRequest,
  ClusterSnapshot,
  GeneratedToken,
  ModelInfo,
  Scheduler,
  Tokenizer,
} from "../contracts.ts";
import {
  chatRequestToGenerate,
  formatChunk,
  formatFinalResponse,
  formatFinishChunk,
  formatModel,
  formatModelsList,
  newChatCompletionId,
  sseDone,
  sseEncode,
} from "./openai.ts";

export interface SoftwellServerDeps {
  scheduler: Scheduler;
  tokenizer: Tokenizer;
  modelId: string;
  getCluster: () => ClusterSnapshot;
  port?: number;
}

const DASHBOARD_DIR = fileURLToPath(new URL("../dashboard/", import.meta.url));
const MAX_BODY_BYTES = 10 * 1024 * 1024;

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

export class SoftwellServer {
  private scheduler: Scheduler;
  private tokenizer: Tokenizer;
  private modelId: string;
  private getCluster: () => ClusterSnapshot;
  private port: number;
  private server: http.Server | null = null;

  constructor(deps: SoftwellServerDeps) {
    this.scheduler = deps.scheduler;
    this.tokenizer = deps.tokenizer;
    this.modelId = deps.modelId;
    this.getCluster = deps.getCluster;
    this.port = deps.port ?? 0;
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
      id: this.modelId,
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "softwell",
    };
  }

  private async handle(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const pathname = url.pathname;
    const method = req.method ?? "GET";

    if (method === "GET" && pathname === "/v1/models") {
      sendJson(res, 200, formatModelsList([this.modelInfo()]));
      return;
    }

    if (method === "GET" && pathname.startsWith("/v1/models/")) {
      const id = decodeURIComponent(pathname.slice("/v1/models/".length));
      if (id === this.modelId) {
        sendJson(res, 200, formatModel(this.modelInfo()));
      } else {
        sendError(res, 404, `model not found: ${id}`);
      }
      return;
    }

    if (method === "POST" && pathname === "/v1/chat/completions") {
      await this.handleChatCompletions(req, res);
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

  private async handleChatCompletions(
    req: http.IncomingMessage,
    res: http.ServerResponse,
  ): Promise<void> {
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
    if (!isValidChatRequest(body)) {
      sendError(res, 400, "request must include a non-empty messages array");
      return;
    }

    const genReq = chatRequestToGenerate(body, this.tokenizer);
    const model = body.model || this.modelId;
    const seqId = this.scheduler.submit(genReq);
    const id = newChatCompletionId();
    const created = Math.floor(Date.now() / 1000);

    if (body.stream) {
      await this.streamSequence(req, res, seqId, id, model, created, genReq.sampling.maxTokens);
    } else {
      await this.completeSequence(res, seqId, id, model, created, genReq.tokenIds.length, genReq.sampling.maxTokens);
    }
  }

  /**
   * Drive the scheduler until the sequence finishes, collecting tokens.
   * Returns the generated token ids and the finish reason.
   */
  private async runUntilDone(
    seqId: string,
    maxTokens: number,
    onToken: (token: GeneratedToken) => void,
    isDisconnected: () => boolean,
  ): Promise<{ tokenIds: number[]; finishReason: string | null }> {
    const tokenIds: number[] = [];
    let finishReason: string | null = null;
    // Generous safety cap: prefill steps may produce no token for this
    // sequence, so allow well past maxTokens iterations.
    const maxSteps = maxTokens + 1000;
    for (let step = 0; step < maxSteps; step += 1) {
      if (isDisconnected()) break;
      const seq = this.scheduler.getSequence(seqId);
      if (seq === undefined) break;
      if (seq.status === "done") {
        finishReason = seq.finishReason ?? "stop";
        break;
      }
      if (seq.status === "error") {
        finishReason = "error";
        break;
      }
      const tokens = await this.scheduler.step();
      let produced = false;
      for (const token of tokens) {
        if (token.seqId !== seqId) continue;
        produced = true;
        tokenIds.push(token.tokenId);
        onToken(token);
        if (token.finishReason !== null) finishReason = token.finishReason;
      }
      const after = this.scheduler.getSequence(seqId);
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
    return { tokenIds, finishReason };
  }

  private async streamSequence(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    seqId: string,
    id: string,
    model: string,
    created: number,
    maxTokens: number,
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

    const { finishReason } = await this.runUntilDone(
      seqId,
      maxTokens,
      (token) => {
        if (!disconnected) res.write(sseEncode(formatChunk(id, model, created, token, 0)));
      },
      () => disconnected,
    );
    if (!disconnected) {
      res.write(sseEncode(formatFinishChunk(id, model, created, finishReason ?? "stop", 0)));
      res.write(sseDone());
      res.end();
    } else {
      res.destroy();
    }
  }

  private async completeSequence(
    res: http.ServerResponse,
    seqId: string,
    id: string,
    model: string,
    created: number,
    promptTokens: number,
    maxTokens: number,
  ): Promise<void> {
    const { tokenIds, finishReason } = await this.runUntilDone(
      seqId,
      maxTokens,
      () => undefined,
      () => false,
    );
    const content = this.tokenizer.decode(tokenIds);
    sendJson(
      res,
      200,
      formatFinalResponse(
        id,
        model,
        created,
        content,
        promptTokens,
        tokenIds.length,
        finishReason ?? "stop",
      ),
    );
  }
}
