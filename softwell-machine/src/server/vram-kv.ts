/**
 * Server layer — remote vRAM chunk ops for the KV cache.
 *
 * The orchestrator-side half of the remote VramAccess (the worker side
 * is RemoteVramAccess in src/scheduler/kvcache-vram-remote.ts). These
 * handlers are deliberately NOT model-scoped: they expose the
 * orchestrator's pooled VirtualRam as allocate / read / write / free
 * over HTTP, so a worker's VramPagedKVCache can hold its chunks in
 * pooled vRAM exactly as if it held a local IVirtualRam — same
 * semantics, same failure modes, one network hop in between.
 *
 * Failover / staleness — inherited from VirtualRam, not re-invented:
 * reads go through `ram.read`, which tries each replica in turn and
 * throws when a page is unavailable on ALL copies; writes store to
 * every surviving copy before completing. Nothing here zero-fills,
 * synthesizes, or caches bytes: a read that cannot be served fails
 * (503) and the worker's cache turns that into a re-prefill, exactly
 * as it does for a local VramError.
 *
 * Error mapping (the wire contract the client relies on):
 *   400  malformed request — validated by the mesh protocol layer,
 *        not here (missing/invalid params, empty write body, body too
 *        large).
 *   404  the address range is outside any allocation, or the
 *        allocation id is unknown (free). Detected by matching the
 *        VramError message — VirtualRam has no "has allocation"
 *        query, and these two messages are the only ones it raises
 *        for a bad address/id. If VirtualRam ever gains a typed
 *        error, this match is the one place to change.
 *   503  every other vRAM failure: alloc that cannot be placed (OOM,
 *        no healthy donors), a page unavailable on all copies, a
 *        write that failed on all copies. "vRAM cannot serve this
 *        right now" — never a silent success.
 *
 * The mesh protocol layer (src/mesh/protocol.ts) renders a thrown
 * MeshError as JSON { error } with its status; anything else becomes
 * a 500, so every failure path here throws MeshError.
 */

import type { IVirtualRam } from "../contracts.ts";
import type {
  VramAllocRequest,
  VramAllocResponse,
  VramFreeResponse,
  VramWriteResponse,
} from "../contracts.ts";
import { MeshError } from "../mesh/protocol.ts";
import { VramError } from "../hw/vram.ts";

/** A vRAM failure naming a bad address or id: the client asked for
 *  something the orchestrator does not hold. */
function isNotFoundVramError(err: unknown): boolean {
  if (!(err instanceof VramError)) return false;
  return (
    err.message.includes("outside any allocation") ||
    err.message.includes("unknown allocation")
  );
}

/** Translate a VirtualRam failure into the wire error the client sees.
 *  Non-vRAM errors (programming bugs, not storage failures) are
 *  rethrown so the protocol layer renders them as 500, not 503. */
function toMeshError(err: unknown, what: string): MeshError {
  if (err instanceof MeshError) return err;
  if (err instanceof VramError) {
    if (isNotFoundVramError(err)) {
      return new MeshError(`${what}: ${err.message}`, { status: 404, cause: err });
    }
    return new MeshError(`${what}: ${err.message}`, { status: 503, cause: err });
  }
  throw err;
}

/**
 * Mesh handlers for the orchestrator's remote-vRAM endpoints, backed by
 * one IVirtualRam (the orchestrator's pooled VirtualRam). This is the
 * single shared construction used by the CLI wiring (src/cli.ts) and
 * the endpoint tests, so the tested path is the shipped path:
 *   allocVram -> POST /mesh/vram/alloc {sizeBytes, replicas?}
 *   readVram  -> GET  /mesh/vram/read?vaddr=...&length=... (bytes)
 *   writeVram -> POST /mesh/vram/write?vaddr=... (bytes in, JSON out)
 *   freeVram  -> POST /mesh/vram/free {id}
 * No disk is involved on any of these paths; alloc rounds up to whole
 * 4 MiB pages inside VirtualRam, exactly as a local alloc would.
 */
export function createKvHandlers(ram: IVirtualRam): {
  allocVram: (req: VramAllocRequest) => Promise<VramAllocResponse>;
  readVram: (vaddr: number, length: number) => Promise<Uint8Array>;
  writeVram: (vaddr: number, data: Uint8Array) => Promise<VramWriteResponse>;
  freeVram: (id: string) => Promise<VramFreeResponse>;
} {
  return {
    allocVram: async (req: VramAllocRequest): Promise<VramAllocResponse> => {
      try {
        const alloc = await ram.alloc(
          req.sizeBytes,
          req.replicas !== undefined ? { replicas: req.replicas } : undefined,
        );
        return { id: alloc.id, vaddr: alloc.vaddr, sizeBytes: alloc.sizeBytes };
      } catch (err) {
        throw toMeshError(err, `vRAM alloc of ${req.sizeBytes} bytes failed`);
      }
    },

    readVram: async (vaddr: number, length: number): Promise<Uint8Array> => {
      try {
        const bytes = await ram.read(vaddr, length);
        if (bytes.length !== length) {
          throw new MeshError(
            `vRAM read of ${length} bytes at vaddr ${vaddr} returned ${bytes.length} bytes`,
            { status: 503 },
          );
        }
        return bytes;
      } catch (err) {
        throw toMeshError(err, `vRAM read of ${length} bytes at vaddr ${vaddr} failed`);
      }
    },

    writeVram: async (vaddr: number, data: Uint8Array): Promise<VramWriteResponse> => {
      try {
        await ram.write(vaddr, data);
        return { bytesWritten: data.length };
      } catch (err) {
        throw toMeshError(err, `vRAM write of ${data.length} bytes at vaddr ${vaddr} failed`);
      }
    },

    freeVram: async (id: string): Promise<VramFreeResponse> => {
      try {
        await ram.free(id);
        return { freed: id };
      } catch (err) {
        throw toMeshError(err, `vRAM free of allocation ${id} failed`);
      }
    },
  };
}
