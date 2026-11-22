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
  VirtualMachine,
  VramAllocation,
} from "../contracts.ts";
import { downloadModel, resolveModelRef } from "../model/hub.ts";
import { loadManifest } from "../model/loader.ts";
import { BpeTokenizer, loadTokenizer } from "../model/tokenizer.ts";

/** KV-cache headroom reserved alongside the weights in the fit check. */
export const KV_HEADROOM_BYTES = 512 * 1024 * 1024;

export interface RegistryDeps {
  /** Lazily resolved VM (null before boot). */
  vm: () => VirtualMachine | null;
  /** hypervisor.stageToVram: stage one vSSD object into vRAM. */
  stageToVram: (key: string) => Promise<VramAllocation>;
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

/** Free every tracked vRAM allocation for a model (best effort). */
async function freeAllocations(state: RegistryState, id: string, vm: VirtualMachine): Promise<void> {
  const allocs = state.allocIds.get(id) ?? [];
  state.allocIds.delete(id);
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
    try {
      for (const rel of listFilesRecursive(dir)) {
        const data = fs.readFileSync(path.join(dir, rel));
        await vm.ssd.putObject(`models/${id}/${rel}`, data);
      }
      let totalStaged = 0;
      for (const file of manifest.files) {
        const alloc = await state.deps.stageToVram(`models/${id}/${file.filename}`);
        staged.push(alloc.id);
        totalStaged += alloc.sizeBytes;
        if (entry.weightVaddr === undefined) entry.weightVaddr = alloc.vaddr;
      }
      entry.vramBytes = totalStaged;
      state.allocIds.set(id, staged);
    } catch (err) {
      for (const allocId of staged) {
        try {
          await vm.ram.free(allocId);
        } catch {
          // Best effort: one failed free must not hide the others.
        }
      }
      state.allocIds.delete(id);
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
