/**
 * Scheduler layer — remote VramAccess over HTTP.
 *
 * `RemoteVramAccess` is the production VramAccess for a worker whose KV
 * cache lives in the orchestrator's pooled vRAM: it speaks the
 * /mesh/vram/* endpoints (src/server/vram-kv.ts) through the shared
 * MeshClient, so a VramPagedKVCache can allocate, write, read and free
 * chunks on the orchestrator exactly as if it held a local
 * IVirtualRam — same interface, same failure contract, one HTTP hop
 * in between.
 *
 * Location — src/scheduler, next to the VramAccess interface it
 * implements (src/scheduler/kvcache-vram.ts) and the cache that
 * consumes it. The mesh layer (src/mesh/protocol.ts) only carries the
 * protocol — the endpoint verbs and their error mapping — the same
 * split as VramWeightLoader (src/model) over MeshClient (src/mesh).
 * This module imports NO src/hw: the worker receives the access, it
 * does not import the hardware layer (the orchestrator owns VirtualRam
 * and its donor topology; the worker only knows an orchestrator URL).
 *
 * Failure semantics — identical to VirtualRamAccess, by construction:
 * every method rejects on failure, never returns partial or
 * synthesized bytes. A read that the orchestrator cannot serve (HTTP
 * 404: address outside any allocation; 503: page unavailable on all
 * copies) rejects with the MeshError, and VramPagedKVCache turns that
 * into a re-prefill. A free of an unknown id rejects (the allocation
 * may already be gone — the cache reports that rather than assuming
 * success). Zero-length reads return an empty array and zero-length
 * writes are a no-op without a request, matching VirtualRam.read /
 * VirtualRam.write, which the protocol layer would otherwise reject
 * with 400.
 *
 * Counters (bytesRead / bytesWritten / allocs / frees) count actual
 * HTTP ops, so the coordinator can see the traffic the remote cache
 * causes; they are informational only and never affect behaviour.
 */

import type { VramAccess } from "../contracts.ts";
import { MeshClient } from "../mesh/protocol.ts";
import type { MeshClientOptions } from "../mesh/protocol.ts";

export class RemoteVramAccess implements VramAccess {
  /** Raw bytes actually read from the orchestrator (this client). */
  bytesRead = 0;
  /** Raw bytes actually written to the orchestrator (this client). */
  bytesWritten = 0;
  /** Allocations created through this client. */
  allocs = 0;
  /** Frees issued through this client. */
  frees = 0;
  /** Owner tag for future allocs (set by the worker once registered). */
  private owner: string | undefined;

  private readonly orchestratorUrl: string;
  private readonly client: MeshClient;

  /**
   * @param orchestratorUrl the orchestrator's mesh base URL — the same
   *   URL the worker already uses to register and fetch weight maps
   *   (e.g. `http://127.0.0.1:8080`).
   * @param opts per-request timeout, as for MeshClient (default 10 s).
   *   A 4 MiB chunk read/write over loopback is normally tens of ms;
   *   raise it only on a slow or heavily loaded mesh.
   */
  constructor(orchestratorUrl: string, opts?: MeshClientOptions) {
    if (typeof orchestratorUrl !== "string" || orchestratorUrl === "") {
      throw new Error("RemoteVramAccess: orchestratorUrl must be a non-empty string");
    }
    this.orchestratorUrl = orchestratorUrl.replace(/\/+$/, "");
    this.client = new MeshClient(opts);
  }

  async read(vaddr: number, length: number): Promise<Uint8Array> {
    if (!Number.isInteger(vaddr) || vaddr < 0) {
      throw new Error(`RemoteVramAccess.read: vaddr must be a non-negative integer, got ${vaddr}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`RemoteVramAccess.read: length must be a non-negative integer, got ${length}`);
    }
    if (length === 0) return new Uint8Array(0);
    const bytes = await this.client.readVram(this.orchestratorUrl, vaddr, length);
    if (bytes.length !== length) {
      throw new Error(
        `RemoteVramAccess.read: orchestrator returned ${bytes.length} bytes for ${length} requested at vaddr ${vaddr}`,
      );
    }
    this.bytesRead += bytes.length;
    return bytes;
  }

  async write(vaddr: number, data: Uint8Array): Promise<void> {
    if (!Number.isInteger(vaddr) || vaddr < 0) {
      throw new Error(`RemoteVramAccess.write: vaddr must be a non-negative integer, got ${vaddr}`);
    }
    if (data.length === 0) return;
    const res = await this.client.writeVram(this.orchestratorUrl, vaddr, data);
    if (res.bytesWritten !== data.length) {
      throw new Error(
        `RemoteVramAccess.write: orchestrator stored ${res.bytesWritten} of ${data.length} bytes at vaddr ${vaddr}`,
      );
    }
    this.bytesWritten += data.length;
  }

  /**
   * Tag future allocs with an owner (the worker's id) for the
   * orphaned-chunk reaper (docs/KV-DESIGN.md §7b). The worker calls
   * this once it knows its id (after registration); allocs made
   * before that (none on the serving path — the probe alloc runs at
   * stage load, after registration) stay untagged and are never
   * reaped by owner.
   */
  setOwner(owner: string): void {
    if (typeof owner !== "string" || owner === "") {
      throw new Error("RemoteVramAccess.setOwner: owner must be a non-empty string");
    }
    this.owner = owner;
  }

  async alloc(
    sizeBytes: number,
    opts?: { replicas?: number; owner?: string },
  ): Promise<{ id: string; vaddr: number }> {
    if (!Number.isInteger(sizeBytes) || sizeBytes <= 0) {
      throw new Error(`RemoteVramAccess.alloc: sizeBytes must be a positive integer, got ${sizeBytes}`);
    }
    if (
      opts?.replicas !== undefined &&
      (!Number.isInteger(opts.replicas) || opts.replicas < 1)
    ) {
      throw new Error(
        `RemoteVramAccess.alloc: replicas must be a positive integer when given, got ${opts.replicas}`,
      );
    }
    const alloc = await this.client.allocVram(this.orchestratorUrl, sizeBytes, {
      ...(opts?.replicas !== undefined ? { replicas: opts.replicas } : {}),
      ...(opts?.owner !== undefined
        ? { owner: opts.owner }
        : this.owner !== undefined
          ? { owner: this.owner }
          : {}),
    });
    this.allocs += 1;
    return { id: alloc.id, vaddr: alloc.vaddr };
  }

  async free(id: string): Promise<void> {
    if (typeof id !== "string" || id === "") {
      throw new Error("RemoteVramAccess.free: id must be a non-empty string");
    }
    await this.client.freeVram(this.orchestratorUrl, id);
    this.frees += 1;
  }
}
