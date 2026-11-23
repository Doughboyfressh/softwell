/**
 * Server layer — model registry.
 *
 * The ModelRegistry is the server's catalogue of models. It knows about
 * models in three places: the local models directory, vSSD under the
 * "models/" prefix, and models it has loaded itself (staged weights in
 * vRAM). Loading a model resolves the reference (local directory stays
 * local; a Hugging Face repo id downloads into
 * models/<owner>__<name>), reads its manifest and tokenizer, stages
 * every file to vSSD, checks the weights fit in vRAM with KV headroom,
 * and stages each *.safetensors shard into vRAM. Unloading frees the
 * vRAM allocations but keeps the vSSD copy and the local directory, so
 * the model stays "available".
 *
 * State lives in a module-level WeakMap keyed by instance — deliberately
 * not in `private` fields — so the class stays structurally assignable:
 * the server's tests substitute a plain FakeRegistry for this type, and
 * a class with private members would reject that substitution at
 * compile time. The allocation ids are still not part of the public API.
 *
 * No network access happens at import time (a hub download only happens
 * inside load(), and only when the model is not already on disk).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type {
  EngineOptions,
  ModelEntry,
  ModelManifest,
  VirtualMachine,
  VramAllocation,
  VramWeightMap,
  WeightMapResponse,
} from "../contracts.ts";
import { downloadModel, resolveModelRef } from "../model/hub.ts";
import { loadManifest, tensorFileFor } from "../model/loader.ts";
import { safetensorsHeaderLength } from "../model/manifest.ts";
import { BpeTokenizer, loadTokenizer } from "../model/tokenizer.ts";
import { MeshError } from "../mesh/protocol.ts";

/** KV-cache headroom reserved alongside the weights in the fit check. */
export const KV_HEADROOM_BYTES = 512 * 1024 * 1024;

export interface RegistryDeps {
  /** Lazily resolved VM (null before boot). */
  vm: () => VirtualMachine | null;
  /** hypervisor.stageToVram: stage one vSSD object into vRAM. */
  stageToVram: (key: string) => Promise<VramAllocation>;
  /**
   * Read bytes from pooled vRAM (the orchestrator's path for serving
   * weight bytes to workers). Defaults to `vm().ram.read` when omitted;
   * the CLI passes `(vaddr, length) => vm.ram.read(vaddr, length)`
   * explicitly so the serving path does not depend on the fallback.
   */
  readVram?: (vaddr: number, length: number) => Promise<Uint8Array>;
  /** Unload refuses the active model. */
  isActive: (modelId: string) => boolean;
  /** Directory scanned for locally available models. Default "./models". */
  modelsDir?: string;
}

interface RegistryState {
  deps: RegistryDeps;
  modelsDir: string;
  entries: Map<string, ModelEntry>;
  /** Model id -> vRAM allocation ids holding its staged weights. */
  allocIds: Map<string, string[]>;
  /** Model id -> per-tensor vRAM locations (built at stage time). */
  weightMaps: Map<string, VramWeightMap>;
  /** Model id -> manifest read at load time (served to workers so they
   *  can build a VramWeightLoader without touching disk). */
  manifests: Map<string, ModelManifest>;
  /** Raw weight bytes served to workers via readWeightTensor. */
  weightBytesServed: number;
}

const states = new WeakMap<ModelRegistry, RegistryState>();

function stateOf(registry: ModelRegistry): RegistryState {
  const state = states.get(registry);
  if (state === undefined) throw new Error("ModelRegistry is not initialized");
  return state;
}

/** A directory counts as a model when it holds a config or any weights. */
function isModelDir(dir: string): boolean {
  try {
    if (fs.existsSync(path.join(dir, "config.json"))) return true;
    return fs.readdirSync(dir).some((f) => f.endsWith(".safetensors"));
  } catch {
    return false;
  }
}

/** All regular files under `dir`, as forward-slash relative paths, sorted. */
function listFilesRecursive(dir: string): string[] {
  const out: string[] = [];
  const walk = (current: string, prefix: string): void => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const rel = prefix === "" ? entry.name : `${prefix}/${entry.name}`;
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full, rel);
      else if (entry.isFile()) out.push(rel);
    }
  };
  walk(dir, "");
  return out.sort();
}

function baseEntry(
  id: string,
  source: ModelEntry["source"],
  dir: string,
): ModelEntry {
  return {
    id,
    source,
    dir,
    status: "available",
    vramBytes: 0,
    requestsServed: 0,
    tokensGenerated: 0,
  };
}

/** True when `id` is a model directory on disk or staged on vSSD. */
async function isKnownAvailable(state: RegistryState, id: string): Promise<boolean> {
  const localDir = path.join(state.modelsDir, id);
  try {
    if (fs.existsSync(localDir) && fs.statSync(localDir).isDirectory()) return true;
  } catch {
    // Fall through to the vSSD check.
  }
  const vm = state.deps.vm();
  if (vm !== null) {
    try {
      const metas = await vm.ssd.list(`models/${id}/`);
      if (metas.length > 0) return true;
    } catch {
      // Unknown: not available.
    }
  }
  return false;
}

/** Free every tracked vRAM allocation for a model (best effort). Also
 *  drops the model's weight map and stored manifest: both describe
 *  bytes that are no longer staged. */
async function freeAllocations(state: RegistryState, id: string, vm: VirtualMachine): Promise<void> {
  const allocs = state.allocIds.get(id) ?? [];
  state.allocIds.delete(id);
  state.weightMaps.delete(id);
  state.manifests.delete(id);
  for (const allocId of allocs) {
    try {
      await vm.ram.free(allocId);
    } catch {
      // Best effort: one failed free must not hide the others.
    }
  }
}

export class ModelRegistry {
  constructor(deps: RegistryDeps) {
    states.set(this, {
      deps,
      modelsDir: path.resolve(deps.modelsDir ?? "./models"),
      entries: new Map<string, ModelEntry>(),
      allocIds: new Map<string, string[]>(),
      weightMaps: new Map<string, VramWeightMap>(),
      manifests: new Map<string, ModelManifest>(),
      weightBytesServed: 0,
    });
  }

  /**
   * Models available without loading: local model directories plus
   * vSSD-staged models not on local disk. Loaded entries win over
   * either source. Params/weightBytes stay undefined until loaded.
   * Does not mutate the registry: use load() to make one known.
   */
  async listAvailable(): Promise<ModelEntry[]> {
    const state = stateOf(this);
    const merged = new Map<string, ModelEntry>();
    for (const entry of state.entries.values()) merged.set(entry.id, entry);

    if (fs.existsSync(state.modelsDir)) {
      let names: string[] = [];
      try {
        names = fs.readdirSync(state.modelsDir).sort();
      } catch {
        names = [];
      }
      for (const name of names) {
        if (merged.has(name)) continue;
        const full = path.join(state.modelsDir, name);
        let stat: fs.Stats | undefined;
        try {
          stat = fs.statSync(full);
        } catch {
          continue;
        }
        if (!stat.isDirectory() || !isModelDir(full)) continue;
        merged.set(name, baseEntry(name, { kind: "local", dir: full }, full));
      }
    }

    const vm = state.deps.vm();
    if (vm !== null) {
      let metas: Array<{ key: string }> = [];
      try {
        metas = await vm.ssd.list("models/");
      } catch {
        metas = [];
      }
      const ids = new Set<string>();
      for (const meta of metas) {
        const parts = meta.key.split("/");
        if (parts.length >= 3 && parts[0] === "models" && parts[1] !== "") {
          ids.add(parts[1] as string);
        }
      }
      for (const id of [...ids].sort()) {
        if (merged.has(id)) continue;
        const dir = path.join(state.modelsDir, id);
        merged.set(id, baseEntry(id, { kind: "local", dir }, dir));
      }
    }

    return [...merged.values()];
  }

  /** Models the registry has loaded (or tried to load). */
  list(): ModelEntry[] {
    return [...stateOf(this).entries.values()];
  }

  get(id: string): ModelEntry | undefined {
    return stateOf(this).entries.get(id);
  }

  /**
   * Per-tensor vRAM locations for a staged model (a copy; mutating it
   * does not affect the registry). Undefined when the model is not
   * staged — i.e. the orchestrator has no vRAM weights to serve for it
   * and workers must fall back to the disk loader.
   */
  getWeightMap(modelId: string): VramWeightMap | undefined {
    const map = stateOf(this).weightMaps.get(modelId);
    if (map === undefined) return undefined;
    const copy: VramWeightMap = {};
    for (const [name, loc] of Object.entries(map)) {
      copy[name] = {
        allocId: loc.allocId,
        vaddr: loc.vaddr,
        headerLen: loc.headerLen,
        dataOffsets: [loc.dataOffsets[0], loc.dataOffsets[1]],
        dtype: loc.dtype,
        shape: [...loc.shape],
      };
    }
    return copy;
  }

  /** Manifest read at load time, or undefined when the model is not staged. */
  getManifest(modelId: string): ModelManifest | undefined {
    return stateOf(this).manifests.get(modelId);
  }

  /**
   * Everything a worker needs to build a VramWeightLoader for a staged
   * model, or null when the model is not staged. Served as JSON by
   * GET /mesh/weights/map?model=<id>.
   */
  getWeightMapResponse(modelId: string): WeightMapResponse | null {
    const state = stateOf(this);
    const manifest = state.manifests.get(modelId);
    const weightMap = this.getWeightMap(modelId);
    if (manifest === undefined || weightMap === undefined) return null;
    return { modelId, manifest, weightMap };
  }

  /** Raw weight bytes served to workers so far (orchestrator-side counter). */
  get weightBytesServed(): number {
    return stateOf(this).weightBytesServed;
  }

  /**
   * Read one tensor's raw bytes from pooled vRAM (never from disk):
   * the weight map locates the tensor inside its shard's staged bytes
   * and the read goes through `deps.readVram` (or `vm().ram.read` when
   * no readVram was supplied). Counts the bytes served. Throws a clear
   * Error for an unstaged model or tensor; the mesh handler maps those
   * to 404 (see createWeightHandlers).
   */
  async readWeightTensor(modelId: string, tensorName: string): Promise<Uint8Array> {
    const state = stateOf(this);
    const map = state.weightMaps.get(modelId);
    if (map === undefined) {
      throw new Error(`model "${modelId}" has no weights staged in vRAM`);
    }
    const loc = map[tensorName];
    if (loc === undefined) {
      throw new Error(`tensor "${tensorName}" is not staged in vRAM for model "${modelId}"`);
    }
    const start = loc.vaddr + loc.headerLen + loc.dataOffsets[0];
    const length = loc.dataOffsets[1] - loc.dataOffsets[0];
    const read = state.deps.readVram;
    let bytes: Uint8Array;
    if (read !== undefined) {
      bytes = await read(start, length);
    } else {
      const vm = state.deps.vm();
      if (vm === null) {
        throw new Error(`cannot read vRAM for model "${modelId}": virtual machine is not booted`);
      }
      bytes = await vm.ram.read(start, length);
    }
    if (bytes.length !== length) {
      throw new Error(
        `vRAM read for tensor "${tensorName}" returned ${bytes.length} bytes, expected ${length}`,
      );
    }
    state.weightBytesServed += bytes.length;
    return bytes;
  }

  /**
   * Load a model: resolve the reference, read its manifest and tokenizer,
   * fit-check the weights against vRAM, stage every file to vSSD, then
   * stage each *.safetensors shard into vRAM (one allocation per shard).
   *
   * A fit failure never throws: the entry is returned with status "error"
   * and a message naming the weights, the headroom and the vRAM total, and
   * no allocation is left behind. Unknown references, unreadable
   * directories and download failures throw a clear Error. Loading a
   * model that is already ready returns the same entry.
   */
  async load(ref: string, opts?: { engineOptions?: EngineOptions }): Promise<ModelEntry> {
    void opts; // Engine options are honoured by the server at activation, not here.
    const state = stateOf(this);
    const parsed = resolveModelRef(ref);
    const vm = state.deps.vm();
    if (vm === null) {
      throw new Error(`cannot load model "${ref}": virtual machine is not booted`);
    }

    let id: string;
    let dir: string;
    let source: ModelEntry["source"];
    if (parsed.kind === "local") {
      dir = parsed.dir;
      id = path.basename(dir);
      source = { kind: "local", dir };
    } else {
      id = `${parsed.owner}__${parsed.name}`;
      dir = path.join(state.modelsDir, id);
      source = { kind: "hub", repo: parsed.repo };
    }

    const existing = state.entries.get(id);
    if (existing !== undefined && (existing.status === "ready" || existing.status === "loading")) {
      return existing;
    }

    // Hub: download unless the model is already on disk.
    if (parsed.kind === "hub" && !fs.existsSync(path.join(dir, "config.json"))) {
      dir = await downloadModel(parsed.repo, dir);
    }

    const manifest = loadManifest(dir, id);
    const tokenizer = loadTokenizer(dir);
    const tokenizerKind = tokenizer instanceof BpeTokenizer ? "bpe" : "byte-level";
    const weightBytes = manifest.totalBytes;

    const entry: ModelEntry = existing ?? baseEntry(id, source, dir);
    entry.source = source;
    entry.dir = dir;
    entry.status = "loading";
    entry.error = undefined;
    entry.params = manifest.totalParams;
    entry.weightBytes = weightBytes;
    entry.numLayers = manifest.config.numHiddenLayers;
    entry.architectures = [...manifest.config.architectures];
    entry.tokenizerKind = tokenizerKind;
    entry.vramBytes = 0;
    entry.weightVaddr = undefined;
    state.entries.set(id, entry);

    // Free any stale allocations from a previous attempt before staging.
    await freeAllocations(state, id, vm);

    // Fit check BEFORE staging anything: weights + KV headroom vs vRAM.
    const totalVram = vm.info().vram.totalBytes;
    const needed = weightBytes + KV_HEADROOM_BYTES;
    if (needed > totalVram) {
      entry.status = "error";
      entry.error =
        `model "${id}" does not fit in vRAM: weights ${weightBytes} bytes + ` +
        `${KV_HEADROOM_BYTES} bytes KV headroom = ${needed} bytes exceeds ` +
        `vRAM total ${totalVram} bytes`;
      console.log(`softwell: model ${id} failed to load: ${entry.error}`);
      return entry;
    }

    const staged: string[] = [];
    const stagedAllocs: VramAllocation[] = [];
    try {
      // Stage every file to vSSD. While the shard bytes are in hand,
      // record each shard's safetensors header length — the weight map
      // needs it to locate tensors inside the raw staged bytes, and
      // reading it here (ingest, from disk) is the only disk read the
      // weight path ever does.
      const headerLens = new Map<string, number>();
      for (const rel of listFilesRecursive(dir)) {
        const data = fs.readFileSync(path.join(dir, rel));
        await vm.ssd.putObject(`models/${id}/${rel}`, data);
        if (rel.endsWith(".safetensors")) {
          const headerLen = safetensorsHeaderLength(data);
          headerLens.set(rel, headerLen);
          headerLens.set(path.basename(rel), headerLen);
        }
      }
      let totalStaged = 0;
      for (const file of manifest.files) {
        const alloc = await state.deps.stageToVram(`models/${id}/${file.filename}`);
        staged.push(alloc.id);
        stagedAllocs.push(alloc);
        totalStaged += alloc.sizeBytes;
        if (entry.weightVaddr === undefined) entry.weightVaddr = alloc.vaddr;
      }
      entry.vramBytes = totalStaged;
      state.allocIds.set(id, staged);
      // Per-tensor weight map: shard alloc vaddr + header length + the
      // tensor's dataOffsets from the manifest. Staged bytes are the raw
      // shard file bytes, so tensor bytes live at
      // vaddr + headerLen + dataOffsets[0].
      const allocByFile = new Map<string, VramAllocation>();
      for (let i = 0; i < manifest.files.length; i++) {
        const file = manifest.files[i] as { filename: string };
        allocByFile.set(file.filename, stagedAllocs[i] as VramAllocation);
      }
      const weightMap: VramWeightMap = {};
      for (const [name, meta] of Object.entries(manifest.tensors)) {
        const shard =
          tensorFileFor(manifest, name) ??
          (manifest.files.length === 1
            ? (manifest.files[0] as { filename: string }).filename
            : undefined);
        if (shard === undefined) {
          throw new Error(`tensor "${name}" has no shard file in the manifest`);
        }
        const alloc = allocByFile.get(shard);
        const headerLen = headerLens.get(shard);
        if (alloc === undefined || headerLen === undefined) {
          throw new Error(`shard "${shard}" was not staged to vRAM`);
        }
        weightMap[name] = {
          allocId: alloc.id,
          vaddr: alloc.vaddr,
          headerLen,
          dataOffsets: [meta.dataOffsets[0], meta.dataOffsets[1]],
          dtype: meta.dtype,
          shape: [...meta.shape],
        };
      }
      state.weightMaps.set(id, weightMap);
      state.manifests.set(id, manifest);
    } catch (err) {
      for (const allocId of staged) {
        try {
          await vm.ram.free(allocId);
        } catch {
          // Best effort: one failed free must not hide the others.
        }
      }
      state.allocIds.delete(id);
      state.weightMaps.delete(id);
      state.manifests.delete(id);
      entry.status = "error";
      entry.error = err instanceof Error ? err.message : String(err);
      entry.vramBytes = 0;
      entry.weightVaddr = undefined;
      console.log(`softwell: model ${id} failed to load: ${entry.error}`);
      throw err;
    }

    entry.status = "ready";
    entry.loadedAt = Date.now();
    console.log(
      `softwell: model ${id} loaded (${entry.params} params, ` +
        `${(entry.vramBytes / 1024 / 1024).toFixed(1)} MiB in vRAM)`,
    );
    return entry;
  }

  /**
   * Unload a model: refuse the active model and a model still loading,
   * free every vRAM allocation, and leave the model "available" (the
   * vSSD copy and local directory stay). Unloading an "available" model
   * — including one only known from disk or vSSD — is a no-op success.
   * Unloading an unknown id throws a clear Error.
   */
  async unload(id: string): Promise<void> {
    const state = stateOf(this);
    const entry = state.entries.get(id);
    if (entry === undefined) {
      if (await isKnownAvailable(state, id)) return;
      throw new Error(`unknown model: ${id}`);
    }
    if (state.deps.isActive(id)) {
      throw new Error(`cannot unload model "${id}": it is the active model`);
    }
    if (entry.status === "loading" || entry.status === "unloading") {
      throw new Error(`cannot unload model "${id}": it is still ${entry.status}`);
    }
    if (entry.status === "available") return;

    const vm = state.deps.vm();
    const allocs = state.allocIds.get(id) ?? [];
    if (allocs.length > 0 && vm === null) {
      throw new Error(`cannot unload model "${id}": virtual machine is not booted`);
    }
    if (vm !== null) await freeAllocations(state, id, vm);

    entry.status = "available";
    entry.error = undefined;
    entry.vramBytes = 0;
    entry.weightVaddr = undefined;
    console.log(`softwell: model ${id} unloaded`);
  }
}

/**
 * Mesh handlers for the orchestrator's binary weight endpoint, backed by
 * one registry. This is the single shared construction used by the CLI
 * wiring (src/cli.ts) and the endpoint tests, so the tested path is the
 * shipped path:
 *   getWeightMap    -> GET /mesh/weights/map?model=<id> (JSON)
 *   getWeightTensor -> GET /mesh/weights?model=<id>&tensor=<name> (bytes)
 * Unknown model / unstaged tensor -> MeshError 404 (the protocol layer
 * renders it as JSON { error }); the byte path never reads disk.
 */
export function createWeightHandlers(registry: ModelRegistry): {
  getWeightMap: (modelId: string) => WeightMapResponse | null;
  getWeightTensor: (modelId: string, tensorName: string) => Promise<Uint8Array>;
} {
  return {
    getWeightMap: (modelId: string) => registry.getWeightMapResponse(modelId),
    getWeightTensor: async (modelId: string, tensorName: string) => {
      const map = registry.getWeightMap(modelId);
      if (map === undefined) {
        throw new MeshError(`unknown model or model not staged in vRAM: ${modelId}`, {
          status: 404,
        });
      }
      if (!(tensorName in map)) {
        throw new MeshError(`tensor not staged in vRAM for model ${modelId}: ${tensorName}`, {
          status: 404,
        });
      }
      return registry.readWeightTensor(modelId, tensorName);
    },
  };
}
