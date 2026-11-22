/**
 * One HTTP server per worker node for the virtual-hardware routes.
 *
 * Mounts every pool's route table (`/hw/<pool>/*`) on a single node:http
 * server — the worker's hwPort (see WorkerCapability.hwPort in
 * contracts.ts).
 *
 * Semantics:
 *  - Routes are matched by exact path (query string ignored).
 *  - POST (and other non-GET) requests carry a JSON body, parsed with a
 *    512 MiB size cap. GET routes receive an empty body {}.
 *  - Binary: a request with `Content-Type: application/octet-stream`
 *    is NOT parsed as JSON. Its raw bytes arrive as `body.__raw`
 *    (a Buffer), and a handler that returns a Buffer/Uint8Array gets
 *    it sent back raw with `Content-Type: application/octet-stream`.
 *    This is the bandwidth path (page/chunk transfer without base64).
 *  - Unknown path -> 404 { error }. Handler throwing -> 500 { error }.
 *  - Malformed JSON -> 400 { error }. Body over the cap -> 413 { error }.
 */
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { RouteHandler } from "../contracts.ts";

/** Maximum JSON request body: 512 MiB. */
export const MAX_HW_BODY_BYTES = 512 * 1024 * 1024;

export interface HwNodeServerOpts {
  port?: number;
  routeTables: Record<string, RouteHandler>[];
}

export interface HwNodeServer {
  port: number;
  close(): Promise<void>;
}

/**
 * Merge route tables into one. Later tables win on path collision — the
 * table listed last owns any path it defines.
 */
export function mergeRouteTables(
  ...tables: Record<string, RouteHandler>[]
): Record<string, RouteHandler> {
  const merged: Record<string, RouteHandler> = {};
  for (const table of tables) {
    for (const [path, handler] of Object.entries(table)) {
      merged[path] = handler;
    }
  }
  return merged;
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const text = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let failed = false;
    req.on("data", (chunk: Buffer) => {
      if (failed) {
        return;
      }
      total += chunk.length;
      if (total > MAX_HW_BODY_BYTES) {
        failed = true;
        reject(
          new Error(`request body exceeds ${MAX_HW_BODY_BYTES} bytes`),
        );
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (!failed) {
        resolve(Buffer.concat(chunks));
      }
    });
    req.on("error", (err) => {
      if (!failed) {
        failed = true;
        reject(err);
      }
    });
  });
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  routes: Record<string, RouteHandler>,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const path = url.pathname;
  // Route tables are keyed "METHOD /path" (e.g. "POST /hw/ram/store");
  // fall back to the bare path for tables that omit the method.
  const handler = routes[`${req.method ?? "GET"} ${path}`] ?? routes[path];
  if (!handler) {
    sendJson(res, 404, { error: `unknown route: ${path}` });
    return;
  }

  let body: Record<string, unknown> = {};
  const isBinary =
    (req.headers["content-type"] ?? "").split(";")[0].trim() ===
    "application/octet-stream";
  if (req.method !== "GET") {
    let raw: Buffer;
    try {
      raw = await readBody(req);
    } catch (err) {
      sendJson(res, 413, {
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }
    if (isBinary) {
      // Binary path: raw bytes in body.__raw; no JSON parsing.
      body = { __raw: raw };
    } else if (raw.length > 0) {
      try {
        const parsed: unknown = JSON.parse(raw.toString("utf8"));
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          sendJson(res, 400, { error: "JSON body must be an object" });
          return;
        }
        body = parsed as Record<string, unknown>;
      } catch {
        sendJson(res, 400, { error: "invalid JSON body" });
        return;
      }
    }
  }

  try {
    const result = await handler(body);
    if (result instanceof Uint8Array) {
      // Binary response: send the bytes raw.
      res.writeHead(200, {
        "content-type": "application/octet-stream",
        "content-length": result.byteLength,
      });
      res.end(result);
      return;
    }
    sendJson(res, 200, result === undefined ? {} : result);
  } catch (err) {
    sendJson(res, 500, {
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

/**
 * Start the node server. `port` 0 (the default) picks an ephemeral port;
 * the actual port is returned. `close()` stops the server.
 */
export async function createHwNodeServer(
  opts: HwNodeServerOpts,
): Promise<HwNodeServer> {
  const routes = mergeRouteTables(...opts.routeTables);
  const server: Server = createServer((req, res) => {
    void handleRequest(req, res, routes).catch((err) => {
      if (!res.headersSent) {
        sendJson(res, 500, {
          error: err instanceof Error ? err.message : String(err),
        });
      } else {
        res.destroy();
      }
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(opts.port ?? 0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("hw node server: could not determine listening port");
  }

  let closed = false;
  return {
    port: address.port,
    close(): Promise<void> {
      if (closed) {
        return Promise.resolve();
      }
      closed = true;
      return new Promise<void>((resolve, reject) => {
        server.close((err) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      });
    },
  };
}
