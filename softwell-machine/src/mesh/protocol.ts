import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import { MESH_PROTOCOL_VERSION } from "../contracts.ts";
import type {
  AssignmentResponse,
  ExecuteRequest,
  ExecuteResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  RegisterRequest,
  RegisterResponse,
  WorkerCapability,
} from "../contracts.ts";

/**
 * Mesh HTTP helpers (node:http + fetch only).
 *
 * createMeshServer() serves the mesh routes and delegates to handlers
 * supplied by the caller. The orchestrator passes registry-backed handlers
 * and no execute handler (POST /mesh/execute then answers 404); a worker
 * passes only an execute handler (register/heartbeat/assignment then
 * answer 404 on the worker's own server).
 */

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class MeshError extends Error {
  readonly status?: number;
  readonly code?: string;
  constructor(
    message: string,
    opts?: { status?: number; code?: string; cause?: unknown },
  ) {
    super(message, opts?.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = "MeshError";
    this.status = opts?.status;
    this.code = opts?.code;
  }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/** Runs one pipeline stage on a worker. Supplied by the worker process. */
export type ExecuteHandler = (req: ExecuteRequest) => Promise<ExecuteResponse>;

export interface MeshServerHandlers {
  register?: (
    req: RegisterRequest,
  ) => RegisterResponse | Promise<RegisterResponse>;
  heartbeat?: (
    req: HeartbeatRequest,
  ) => HeartbeatResponse | Promise<HeartbeatResponse>;
  getAssignment?: (
    workerId: string,
  ) => AssignmentResponse | Promise<AssignmentResponse>;
  /** If absent, POST /mesh/execute answers 404 (e.g. the orchestrator). */
  execute?: ExecuteHandler;
}

const MAX_BODY_BYTES = 64 * 1024 * 1024;

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > MAX_BODY_BYTES) {
      throw new MeshError("request body too large", { status: 413, code: "BODY_TOO_LARGE" });
    }
    chunks.push(buf);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  if (text.trim() === "") return {};
  try {
    return JSON.parse(text) as unknown;
  } catch (err) {
    throw new MeshError("request body is not valid JSON", {
      status: 400,
      code: "INVALID_JSON",
      cause: err,
    });
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function normalizePath(pathname: string): string {
  if (pathname.length > 1 && pathname.endsWith("/")) {
    return pathname.slice(0, -1);
  }
  return pathname;
}

/**
 * Create an HTTP server for the mesh routes:
 *   POST /mesh/register
 *   POST /mesh/heartbeat
 *   GET  /mesh/assignment?workerId=...
 *   POST /mesh/execute
 * The caller owns listen()/close(). Unknown routes answer 404, a known
 * route with the wrong method answers 405, a route whose handler was not
 * supplied answers 404.
 */
export function createMeshServer(handlers: MeshServerHandlers): Server {
  const server = createServer((req, res) => {
    void handleRequest(req, res, handlers).catch((err: unknown) => {
      if (!res.headersSent) {
        const message = err instanceof Error ? err.message : String(err);
        const status = err instanceof MeshError && err.status !== undefined ? err.status : 500;
        sendJson(res, status, { error: message });
      } else {
        res.end();
      }
    });
  });
  return server;
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  handlers: MeshServerHandlers,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const path = normalizePath(url.pathname);
  const method = req.method ?? "GET";

  const knownPaths = ["/mesh/register", "/mesh/heartbeat", "/mesh/assignment", "/mesh/execute"];
  if (!knownPaths.includes(path)) {
    sendJson(res, 404, { error: `unknown route: ${method} ${path}` });
    return;
  }

  if (path === "/mesh/register") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.register === undefined) {
      return sendJson(res, 404, { error: "register not available on this server" });
    }
    const body = (await readJsonBody(req)) as RegisterRequest;
    sendJson(res, 200, await handlers.register(body));
    return;
  }

  if (path === "/mesh/heartbeat") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.heartbeat === undefined) {
      return sendJson(res, 404, { error: "heartbeat not available on this server" });
    }
    const body = (await readJsonBody(req)) as HeartbeatRequest;
    sendJson(res, 200, await handlers.heartbeat(body));
    return;
  }

  if (path === "/mesh/assignment") {
    if (method !== "GET") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.getAssignment === undefined) {
      return sendJson(res, 404, { error: "assignment not available on this server" });
    }
    const workerId = url.searchParams.get("workerId");
    if (workerId === null || workerId === "") {
      return sendJson(res, 400, { error: "missing query parameter: workerId" });
    }
    sendJson(res, 200, await handlers.getAssignment(workerId));
    return;
  }

  // path === "/mesh/execute"
  if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
  if (handlers.execute === undefined) {
    return sendJson(res, 404, { error: "execute not available on this server" });
  }
  const body = (await readJsonBody(req)) as ExecuteRequest;
  sendJson(res, 200, await handlers.execute(body));
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export interface MeshClientOptions {
  /** Per-request timeout in ms. Default 10_000. */
  timeoutMs?: number;
}

function joinUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/+$/, "")}${path}`;
}

/**
 * HTTP client for the mesh protocol. Every method throws MeshError (never
 * a raw fetch crash) on connection errors, timeouts, and non-2xx statuses;
 * check `err.status` / `err.code` to tell them apart.
 */
export class MeshClient {
  private timeoutMs: number;

  constructor(opts?: MeshClientOptions) {
    this.timeoutMs = opts?.timeoutMs ?? 10_000;
  }

  private async request<T>(url: string, init: RequestInit): Promise<T> {
    let res: Response;
    try {
      res = await fetch(url, {
        ...init,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      if (err instanceof Error && err.name === "TimeoutError") {
        throw new MeshError(`mesh request timed out after ${this.timeoutMs}ms: ${url}`, {
          code: "TIMEOUT",
          cause: err,
        });
      }
      throw new MeshError(`mesh request failed: ${url}`, {
        code: "CONNECTION_ERROR",
        cause: err,
      });
    }
    if (!res.ok) {
      let detail = "";
      try {
        const body = (await res.json()) as { error?: unknown };
        if (typeof body.error === "string") detail = `: ${body.error}`;
      } catch {
        // Non-JSON error body; keep the status text only.
      }
      throw new MeshError(`mesh request failed with HTTP ${res.status}${detail}`, {
        status: res.status,
        code: "HTTP_ERROR",
      });
    }
    return (await res.json()) as T;
  }

  async register(
    orchestratorUrl: string,
    capability: WorkerCapability,
  ): Promise<RegisterResponse> {
    const body: RegisterRequest = {
      protocolVersion: MESH_PROTOCOL_VERSION,
      capability,
    };
    return this.request<RegisterResponse>(joinUrl(orchestratorUrl, "/mesh/register"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async heartbeat(
    orchestratorUrl: string,
    req: HeartbeatRequest,
  ): Promise<HeartbeatResponse> {
    return this.request<HeartbeatResponse>(joinUrl(orchestratorUrl, "/mesh/heartbeat"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
  }

  async getAssignment(
    orchestratorUrl: string,
    workerId: string,
  ): Promise<AssignmentResponse> {
    const url = `${joinUrl(orchestratorUrl, "/mesh/assignment")}?workerId=${encodeURIComponent(workerId)}`;
    return this.request<AssignmentResponse>(url, { method: "GET" });
  }

  async execute(workerUrl: string, req: ExecuteRequest): Promise<ExecuteResponse> {
    return this.request<ExecuteResponse>(joinUrl(workerUrl, "/mesh/execute"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
  }
}
