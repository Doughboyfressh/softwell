/**
 * Model layer — weight loading from pooled vRAM.
 *
 * `VramWeightLoader` implements the contracts WeightLoader, but its
 * bytes come from the pooled vRAM allocations that hold a model's staged
 * weights — never from local disk. This module imports no `node:fs`: it
 * cannot read a file even by accident. The model directory need not
 * exist at all (see test/vram-loader.test.ts, which loads from a manifest
 * pointing at a nonexistent directory with a fake fetch function).
 *
 * Fetch function (the one design decision): the constructor takes a
 * TENSOR fetcher — `(tensorName) => Promise<Uint8Array>` returning
 * exactly that tensor's RAW bytes (the `dataOffsets` span, no header,
 * no whole shard). Rationale: the tensor is the unit the engine asks
 * for, the unit the orchestrator's binary endpoint serves
 * (GET /mesh/weights?model=<id>&tensor=<name>), and the unit the
 * counters account (`bytesFetched` is raw tensor bytes served). A
 * RANGE fetcher `(vaddr, length) => Promise<Uint8Array>` — the shape
 * `vm.ram.read` has — is adapted to a tensor fetcher with
 * `createTensorFetcherFromRange`, using the weight map to compute the
 * absolute address (`vaddr + headerLen + dataOffsets[0]`).
 *
 * Semantics (same as the disk loader, minus the disk):
 *  - loadTensor fetches the tensor's raw bytes once and decodes them
 *    with the existing decodeToFloat32 (src/model/manifest.ts).
 *  - Per-tensor cache: the engine loads each tensor once at stage load;
 *    a second loadTensor for the same name returns the cached array
 *    without another fetch. Concurrent loads of the same tensor share
 *    one in-flight fetch.
 *  - Counters: `tensorsFetched` / `bytesFetched` count actual vRAM
 *    fetches only (cache hits do not count). close() drops the cache.
 *  - The vRAM allocations remain the source of truth: nothing here is
 *    persisted to disk, and after close() a tensor can be re-fetched
 *    from vRAM (e.g. after the pool healed a lost replica).
 */

import type {
  ModelManifest,
  VramWeightMap,
  WeightLoader,
} from "../contracts.ts";
import { decodeToFloat32, dtypeByteSize, numElements } from "./manifest.ts";

/** Fetch exactly one tensor's raw bytes (the dataOffsets span). */
export type VramTensorFetch = (tensorName: string) => Promise<Uint8Array>;

/** Fetch an arbitrary byte range from the vRAM address space. */
export type VramRangeFetch = (vaddr: number, length: number) => Promise<Uint8Array>;

/**
 * Adapt a range fetcher (e.g. `vm.ram.read`) to the tensor fetcher a
 * VramWeightLoader takes, using the weight map to resolve each tensor
 * to its absolute vRAM address.
 */
export function createTensorFetcherFromRange(
  weightMap: VramWeightMap,
  fetchRange: VramRangeFetch,
): VramTensorFetch {
  return async (tensorName: string): Promise<Uint8Array> => {
    const loc = weightMap[tensorName];
    if (loc === undefined) {
      throw new Error(`tensor not in vRAM weight map: ${tensorName}`);
    }
    return fetchRange(
      loc.vaddr + loc.headerLen + loc.dataOffsets[0],
      loc.dataOffsets[1] - loc.dataOffsets[0],
    );
  };
}

export class VramWeightLoader implements WeightLoader {
  readonly manifest: ModelManifest;
  /** The weight map this loader was built from (tensor -> vRAM location). */
  readonly weightMap: VramWeightMap;
  /** Distinct tensors actually fetched from vRAM (cache hits excluded). */
  tensorsFetched = 0;
  /** Raw tensor bytes actually fetched from vRAM (cache hits excluded). */
  bytesFetched = 0;

  private readonly fetchTensor: VramTensorFetch;
  /** Tensor name -> decoded Float32Array, cached after the first fetch. */
  private readonly cache = new Map<string, Float32Array>();
  /** Tensor name -> in-flight fetch, so concurrent loads share one fetch. */
  private readonly inflight = new Map<string, Promise<Float32Array>>();

  constructor(
    manifest: ModelManifest,
    weightMap: VramWeightMap,
    fetchTensor: VramTensorFetch,
  ) {
    this.manifest = manifest;
    this.weightMap = weightMap;
    this.fetchTensor = fetchTensor;
  }

  /** True only when the tensor is in the manifest AND staged in vRAM. */
  hasTensor(name: string): boolean {
    return name in this.manifest.tensors && name in this.weightMap;
  }

  /** All tensor names in the manifest. */
  tensorNames(): string[] {
    return Object.keys(this.manifest.tensors);
  }

  async loadTensor(name: string): Promise<Float32Array> {
    const cached = this.cache.get(name);
    if (cached !== undefined) return cached;
    const pending = this.inflight.get(name);
    if (pending !== undefined) return pending;
    const task = this.fetchAndDecode(name);
    this.inflight.set(name, task);
    try {
      const decoded = await task;
      this.cache.set(name, decoded);
      return decoded;
    } finally {
      this.inflight.delete(name);
    }
  }

  private async fetchAndDecode(name: string): Promise<Float32Array> {
    const meta = this.manifest.tensors[name];
    if (meta === undefined) throw new Error(`tensor not found: ${name}`);
    const loc = this.weightMap[name];
    if (loc === undefined) {
      throw new Error(`tensor "${name}" is not in the vRAM weight map`);
    }
    if (loc.dtype !== meta.dtype) {
      throw new Error(
        `tensor "${name}": weight map says dtype ${loc.dtype} but manifest says ${meta.dtype}`,
      );
    }
    if (
      loc.shape.length !== meta.shape.length ||
      loc.shape.some((d, i) => d !== meta.shape[i])
    ) {
      throw new Error(
        `tensor "${name}": weight map says shape [${loc.shape.join(", ")}] but manifest says [${meta.shape.join(", ")}]`,
      );
    }
    const count = numElements(meta.shape);
    const expectedBytes = count * dtypeByteSize(meta.dtype);
    const span = loc.dataOffsets[1] - loc.dataOffsets[0];
    if (span !== expectedBytes) {
      throw new Error(
        `tensor "${name}": data_offsets span ${span} bytes but shape needs ${count} elements of ${meta.dtype} (${expectedBytes} bytes)`,
      );
    }
    const raw = await this.fetchTensor(name);
    if (raw.length !== expectedBytes) {
      throw new Error(
        `vRAM fetch for tensor "${name}" returned ${raw.length} bytes, expected ${expectedBytes}`,
      );
    }
    this.tensorsFetched += 1;
    this.bytesFetched += raw.length;
    return decodeToFloat32(meta.dtype, Buffer.from(raw), 0, count);
  }

  close(): void {
    this.cache.clear();
    this.inflight.clear();
  }
}
