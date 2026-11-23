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
  VramAllocRequest,
  VramAllocResponse,
  VramFreeResponse,
  VramWriteResponse,
  WeightMapResponse,
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
  /**
   * Release a finished sequence's KV cache on this worker
   * (POST /mesh/free-sequence {seqId} -> {freed}). The scheduler calls
   * it when a sequence finishes (done/error/cancelled) so the worker's
   * per-stage cache returns the blocks — without it the cache leaks
   * every finished sequence until the stage reloads (found 2026-11-23:
   * a 997-token generation consumed 756 of the host cache's 1024
   * blocks and the next one OOMed with "no free blocks"). Idempotent:
   * freeing an unknown seqId is a no-op that still answers 200, and
   * the scheduler treats failures as best-effort (never fails a step).
   */
  freeSequence?: (seqId: string) => Promise<{ freed: boolean }>;
  /**
   * Binary weight serving (orchestrator only). getWeightMap answers
   * GET /mesh/weights/map?model=<id> with the manifest + per-tensor
   * vRAM locations, or null when the model is not staged (404).
   * getWeightTensor answers GET /mesh/weights?model=<id>&tensor=<name>
   * with exactly that tensor's raw bytes as application/octet-stream;
   * it throws MeshError 404 for an unknown model/tensor and must never
   * read disk — bytes come from pooled vRAM via the registry.
   */
  getWeightMap?: (
    modelId: string,
  ) => WeightMapResponse | null | Promise<WeightMapResponse | null>;
  getWeightTensor?: (
    modelId: string,
    tensorName: string,
  ) => Promise<Uint8Array>;
  /**
   * Remote vRAM chunk ops (orchestrator only; see src/server/vram-kv.ts
   * for the registry-free handlers over the orchestrator's VirtualRam).
   * The handler layer translates vRAM failures into MeshError: 404 for
   * an unknown allocation id / an address outside any allocation, 503
   * for a page unavailable on all copies or an alloc that cannot be
   * placed (OOM / no donors). This layer only validates request shape
   * (400) and renders the handlers' MeshErrors.
   *   allocVram -> POST /mesh/vram/alloc {sizeBytes, replicas?} (JSON)
   *   readVram  -> GET  /mesh/vram/read?vaddr=...&length=... (octet-stream)
   *   writeVram -> POST /mesh/vram/write?vaddr=... (octet-stream body)
   *   freeVram  -> POST /mesh/vram/free {id} (JSON)
   */
  allocVram?: (req: VramAllocRequest) => Promise<VramAllocResponse>;
  readVram?: (vaddr: number, length: number) => Promise<Uint8Array>;
  writeVram?: (vaddr: number, data: Uint8Array) => Promise<VramWriteResponse>;
  freeVram?: (id: string) => Promise<VramFreeResponse>;
}

const MAX_BODY_BYTES = 64 * 1024 * 1024;

/** Largest single KV read/write (16 vRAM pages — a chunk is one page). */
export const MAX_KV_RANGE_BYTES = 64 * 1024 * 1024;

/**
 * Parse a query parameter as a non-negative integer, or throw a 400
 * MeshError naming the parameter. `max` (when given) bounds the value.
 */
function queryInt(
  url: URL,
  name: string,
  opts?: { min?: number; max?: number },
): number {
  const raw = url.searchParams.get(name);
  if (raw === null || raw === "") {
    throw new MeshError(`missing query parameter: ${name}`, { status: 400 });
  }
  const value = Number(raw);
  const min = opts?.min ?? 0;
  if (!Number.isInteger(value) || value < min) {
    throw new MeshError(
      `query parameter ${name} must be an integer >= ${min}, got ${JSON.stringify(raw)}`,
      { status: 400 },
    );
  }
  if (opts?.max !== undefined && value > opts.max) {
    throw new MeshError(
      `query parameter ${name} must be <= ${opts.max}, got ${value}`,
      { status: 400 },
    );
  }
  return value;
}

/** Read a request body as raw bytes, capped (413 past the cap). */
async function readRawBody(req: IncomingMessage, maxBytes: number): Promise<Uint8Array> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > maxBytes) {
      throw new MeshError("request body too large", { status: 413, code: "BODY_TOO_LARGE" });
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks);
}

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

function sendOctet(res: ServerResponse, bytes: Uint8Array): void {
  res.writeHead(200, {
    "content-type": "application/octet-stream",
    "content-length": bytes.byteLength,
  });
  res.end(bytes);
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
 *   POST /mesh/free-sequence             (JSON {seqId} -> {freed})
 *   GET  /mesh/weights/map?model=...        (manifest + vRAM weight map, JSON)
 *   GET  /mesh/weights?model=...&tensor=... (one tensor's raw bytes, octet-stream)
 *   POST /mesh/vram/alloc                   (JSON {sizeBytes, replicas?} -> {id, vaddr, sizeBytes})
 *   GET  /mesh/vram/read?vaddr=...&length=... (byte range, octet-stream)
 *   POST /mesh/vram/write?vaddr=...         (octet-stream body -> {bytesWritten})
 *   POST /mesh/vram/free                    (JSON {id} -> {freed})
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

  const knownPaths = [
    "/mesh/register",
    "/mesh/heartbeat",
    "/mesh/assignment",
    "/mesh/execute",
    "/mesh/free-sequence",
    "/mesh/weights",
    "/mesh/weights/map",
    "/mesh/vram/alloc",
    "/mesh/vram/read",
    "/mesh/vram/write",
    "/mesh/vram/free",
  ];
  if (!knownPaths.includes(path)) {
    sendJson(res, 404, { error: `unknown route: ${method} ${path}` });
    return;
  }

  if (path === "/mesh/vram/alloc") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.allocVram === undefined) {
      return sendJson(res, 404, { error: "vram alloc not available on this server" });
    }
    const body = (await readJsonBody(req)) as Partial<VramAllocRequest>;
    if (
      typeof body !== "object" ||
      body === null ||
      !Number.isInteger(body.sizeBytes) ||
      (body.sizeBytes as number) <= 0
    ) {
      return sendJson(res, 400, { error: "body.sizeBytes must be a positive integer" });
    }
    if (
      body.replicas !== undefined &&
      (!Number.isInteger(body.replicas) || (body.replicas as number) < 1)
    ) {
      return sendJson(res, 400, { error: "body.replicas must be a positive integer when given" });
    }
    sendJson(res, 200, await handlers.allocVram(body as VramAllocRequest));
    return;
  }

  if (path === "/mesh/vram/read") {
    if (method !== "GET") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.readVram === undefined) {
      return sendJson(res, 404, { error: "vram read not available on this server" });
    }
    const vaddr = queryInt(url, "vaddr");
    const length = queryInt(url, "length", { min: 1, max: MAX_KV_RANGE_BYTES });
    sendOctet(res, await handlers.readVram(vaddr, length));
    return;
  }

  if (path === "/mesh/vram/write") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.writeVram === undefined) {
      return sendJson(res, 404, { error: "vram write not available on this server" });
    }
    const vaddr = queryInt(url, "vaddr");
    const data = await readRawBody(req, MAX_KV_RANGE_BYTES);
    if (data.length === 0) {
      return sendJson(res, 400, { error: "write body must not be empty" });
    }
    sendJson(res, 200, await handlers.writeVram(vaddr, data));
    return;
  }

  if (path === "/mesh/vram/free") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.freeVram === undefined) {
      return sendJson(res, 404, { error: "vram free not available on this server" });
    }
    const body = (await readJsonBody(req)) as { id?: unknown };
    if (typeof body !== "object" || body === null || typeof body.id !== "string" || body.id === "") {
      return sendJson(res, 400, { error: "body.id must be a non-empty string" });
    }
    sendJson(res, 200, await handlers.freeVram(body.id));
    return;
  }

  if (path === "/mesh/weights/map") {
    if (method !== "GET") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.getWeightMap === undefined) {
      return sendJson(res, 404, { error: "weights map not available on this server" });
    }
    const modelId = url.searchParams.get("model");
    if (modelId === null || modelId === "") {
      return sendJson(res, 400, { error: "missing query parameter: model" });
    }
    const result = await handlers.getWeightMap(modelId);
    if (result === null) {
      return sendJson(res, 404, { error: `unknown model or model not staged in vRAM: ${modelId}` });
    }
    sendJson(res, 200, result);
    return;
  }

  if (path === "/mesh/weights") {
    if (method !== "GET") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.getWeightTensor === undefined) {
      return sendJson(res, 404, { error: "weights not available on this server" });
    }
    const modelId = url.searchParams.get("model");
    const tensorName = url.searchParams.get("tensor");
    if (modelId === null || modelId === "") {
      return sendJson(res, 400, { error: "missing query parameter: model" });
    }
    if (tensorName === null || tensorName === "") {
      return sendJson(res, 400, { error: "missing query parameter: tensor" });
    }
    sendOctet(res, await handlers.getWeightTensor(modelId, tensorName));
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

  // path === "/mesh/free-sequence"
  if (path === "/mesh/free-sequence") {
    if (method !== "POST") return sendJson(res, 405, { error: "method not allowed" });
    if (handlers.freeSequence === undefined) {
      return sendJson(res, 404, { error: "free-sequence not available on this server" });
    }
    const body = (await readJsonBody(req)) as { seqId?: unknown };
    if (typeof body !== "object" || body === null || typeof body.seqId !== "string" || body.seqId === "") {
      return sendJson(res, 400, { error: "body.seqId must be a non-empty string" });
    }
    sendJson(res, 200, await handlers.freeSequence(body.seqId));
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

  /**
   * Raw fetch with the same timeout/connection error mapping as
   * request(), but returning the Response so callers can handle binary
   * bodies and 404s themselves.
   */
  private async fetchRaw(url: string, init: RequestInit): Promise<Response> {
    try {
      return await fetch(url, {
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
  }

  /** Turn a non-2xx response into a MeshError, preferring its JSON error. */
  private async errorFromResponse(res: Response, url: string): Promise<MeshError> {
    let detail = "";
    try {
      const body = (await res.json()) as { error?: unknown };
      if (typeof body.error === "string") detail = `: ${body.error}`;
    } catch {
      // Non-JSON error body; keep the status text only.
    }
    return new MeshError(`mesh request failed with HTTP ${res.status}${detail}`, {
      status: res.status,
      code: "HTTP_ERROR",
    });
  }

  /**
   * Fetch the manifest + vRAM weight map for a staged model
   * (GET /mesh/weights/map?model=<id>). Returns null when the
   * orchestrator has not staged the model (HTTP 404) — the worker
   * treats that as "fall back to the disk loader". Any other failure
   * throws MeshError.
   */
  async getWeightMap(
    orchestratorUrl: string,
    modelId: string,
  ): Promise<WeightMapResponse | null> {
    const url = `${joinUrl(orchestratorUrl, "/mesh/weights/map")}?model=${encodeURIComponent(modelId)}`;
    const res = await this.fetchRaw(url, { method: "GET" });
    if (res.status === 404) return null;
    if (!res.ok) throw await this.errorFromResponse(res, url);
    return (await res.json()) as WeightMapResponse;
  }

  /**
   * Fetch one tensor's raw bytes from pooled vRAM
   * (GET /mesh/weights?model=<id>&tensor=<name>), as octet-stream —
   * never JSON/base64. Throws MeshError (with status 404 for an
   * unknown model/tensor).
   */
  async fetchWeightTensor(
    orchestratorUrl: string,
    modelId: string,
    tensorName: string,
  ): Promise<Uint8Array> {
    const url =
      `${joinUrl(orchestratorUrl, "/mesh/weights")}` +
      `?model=${encodeURIComponent(modelId)}&tensor=${encodeURIComponent(tensorName)}`;
    const res = await this.fetchRaw(url, { method: "GET" });
    if (!res.ok) throw await this.errorFromResponse(res, url);
    return new Uint8Array(await res.arrayBuffer());
  }

  // --- Remote vRAM chunk ops (POST /mesh/vram/*) ---------------------------
  // Same timeout / MeshError mapping as the weight methods above; the
  // bodies are raw bytes, never JSON/base64. A 404 means the address or
  // allocation id is unknown to the orchestrator; a 503 means vRAM
  // could not serve the op (OOM, no donors, or a page lost on all
  // copies) — callers must treat that as "bytes unavailable", never as
  // zeros. See src/server/vram-kv.ts for the server-side mapping.

  /**
   * Allocate `sizeBytes` in pooled vRAM (POST /mesh/vram/alloc).
   * Returns the allocation id, its base virtual address, and the bytes
   * actually reserved (rounded up to whole 4 MiB pages).
   */
  async allocVram(
    orchestratorUrl: string,
    sizeBytes: number,
    opts?: { replicas?: number },
  ): Promise<VramAllocResponse> {
    const body: VramAllocRequest = { sizeBytes, ...(opts?.replicas !== undefined ? { replicas: opts.replicas } : {}) };
    return this.request<VramAllocResponse>(joinUrl(orchestratorUrl, "/mesh/vram/alloc"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  /**
   * Read `length` bytes from pooled vRAM (GET /mesh/vram/read), as
   * octet-stream. Throws MeshError when the range is unavailable.
   */
  async readVram(
    orchestratorUrl: string,
    vaddr: number,
    length: number,
  ): Promise<Uint8Array> {
    const url =
      `${joinUrl(orchestratorUrl, "/mesh/vram/read")}` +
      `?vaddr=${encodeURIComponent(String(vaddr))}&length=${encodeURIComponent(String(length))}`;
    const res = await this.fetchRaw(url, { method: "GET" });
    if (!res.ok) throw await this.errorFromResponse(res, url);
    return new Uint8Array(await res.arrayBuffer());
  }

  /**
   * Write raw bytes to pooled vRAM (POST /mesh/vram/write, octet-stream
   * body). Returns the bytes the orchestrator stored.
   */
  async writeVram(
    orchestratorUrl: string,
    vaddr: number,
    data: Uint8Array,
  ): Promise<VramWriteResponse> {
    const url = `${joinUrl(orchestratorUrl, "/mesh/vram/write")}?vaddr=${encodeURIComponent(String(vaddr))}`;
    const res = await this.fetchRaw(url, {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      // fetch accepts Uint8Array as a body; pass the view through so a
      // 4 MiB page is not copied.
      body: data,
    });
    if (!res.ok) throw await this.errorFromResponse(res, url);
    return (await res.json()) as VramWriteResponse;
  }

  /** Release an allocation created by allocVram (POST /mesh/vram/free). */
  async freeVram(orchestratorUrl: string, id: string): Promise<VramFreeResponse> {
    return this.request<VramFreeResponse>(joinUrl(orchestratorUrl, "/mesh/vram/free"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id }),
    });
  }

  async execute(workerUrl: string, req: ExecuteRequest): Promise<ExecuteResponse> {
    return this.request<ExecuteResponse>(joinUrl(workerUrl, "/mesh/execute"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
  }

  /**
   * Release a finished sequence's KV cache on a worker
   * (POST /mesh/free-sequence). Best-effort by design: the caller
   * swallows failures — a worker that is gone took its cache with it.
   */
  async freeSequence(workerUrl: string, seqId: string): Promise<{ freed: boolean }> {
    return this.request<{ freed: boolean }>(joinUrl(workerUrl, "/mesh/free-sequence"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ seqId }),
    });
  }
}
