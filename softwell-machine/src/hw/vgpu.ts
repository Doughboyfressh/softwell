/**
 * vgpu.ts — the virtual GPU pool.
 *
 * THE BOUNDARY: workloads see only VIRTUAL GPU devices (see listDevices())
 * and kernel launches. A workload never learns which donor backs a virtual
 * device, and never sees a physical GPU. The mapping virtual-device ->
 * donor lives inside this file: `VirtualGpu` (coordinator, orchestrator
 * side) picks a donor and POSTs the kernel to that donor's node server;
 * `GpuNode` (donor side) translates the virtual op into whatever the donor
 * physically has.
 *
 * Reference implementation honesty: this reference `GpuNode` always runs
 * the CPU fallback kernels from ./kernels-cpu.ts, even when the donor
 * reports device 'cuda'/'metal' — a real device backend would plug in at
 * the marked spot in GpuNode.handleKernel. Kernel results therefore report
 * device 'cpu' (the device that ACTUALLY executed), and stats never claim
 * GPU flops a donor does not have: listDevices() reports the donor's own
 * flopsPerSec, and computeUnits is documented as an approximation below.
 *
 * Transport: coordinators speak only to donor nodes through /hw/* routes
 * via plain fetch; binary payloads travel as base64 inside JSON. Node
 * inputs/outputs are normalized to float32 on the wire; the coordinator
 * converts to/from the tensor's declared vRAM dtype at the boundary.
 */

import { CpuKernels, MATMUL_PARALLEL_MIN_FLOPS, matmulParallel, matmulTiled } from "./kernels-cpu.ts";
import type {
  GpuStats,
  IVirtualGpu,
  IVirtualRam,
  KernelLaunch,
  KernelOp,
  KernelResult,
  NodeProvider,
  RouteHandler,
  TensorRef,
  VAddr,
  VirtualGpuDevice,
  VramAllocation,
  WeightDtype,
  WorkerDevice,
  WorkerInfo,
} from "../contracts.ts";

// ---------------------------------------------------------------------------
// Dtype conversion (vRAM bytes <-> float32 values). The kernel library works
// in float32; tensors in vRAM may be stored in any WeightDtype.
// ---------------------------------------------------------------------------

export function dtypeByteSize(dtype: WeightDtype): number {
  switch (dtype) {
    case "float32":
      return 4;
    case "float16":
    case "bfloat16":
      return 2;
    case "float8_e4m3":
      return 1;
  }
}

function numel(shape: number[]): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

/** Decode raw vRAM bytes of the given dtype into float32 values. */
export function decodeToFloat32(bytes: Uint8Array, dtype: WeightDtype): Float32Array {
  switch (dtype) {
    case "float32": {
      const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      return new Float32Array(buf);
    }
    case "float16": {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const out = new Float32Array(bytes.byteLength / 2);
      for (let i = 0; i < out.length; i++) out[i] = halfToFloat(view.getUint16(i * 2, true));
      return out;
    }
    case "bfloat16": {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const out = new Float32Array(bytes.byteLength / 2);
      const bits = new Uint32Array(1);
      const asFloat = new Float32Array(bits.buffer);
      for (let i = 0; i < out.length; i++) {
        bits[0] = view.getUint16(i * 2, true) << 16;
        out[i] = asFloat[0];
      }
      return out;
    }
    case "float8_e4m3": {
      const out = new Float32Array(bytes.byteLength);
      for (let i = 0; i < out.length; i++) out[i] = fp8E4M3ToFloat(bytes[i]);
      return out;
    }
  }
}

/** Encode float32 values into raw bytes of the given dtype. */
export function encodeFromFloat32(values: Float32Array, dtype: WeightDtype): Uint8Array {
  switch (dtype) {
    case "float32": {
      return new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
    }
    case "float16": {
      const out = new Uint8Array(values.length * 2);
      const view = new DataView(out.buffer);
      for (let i = 0; i < values.length; i++) view.setUint16(i * 2, floatToHalf(values[i]), true);
      return out;
    }
    case "bfloat16": {
      const out = new Uint8Array(values.length * 2);
      const view = new DataView(out.buffer);
      const bits = new Uint32Array(new Float32Array([0]).buffer);
      const asFloat = new Float32Array(bits.buffer);
      for (let i = 0; i < values.length; i++) {
        asFloat[0] = values[i];
        // Round-to-nearest-even on the truncated low 16 bits.
        const rounded = (bits[0] + 0x7fff + ((bits[0] >> 16) & 1)) & 0xffff0000;
        view.setUint16(i * 2, rounded >>> 16, true);
      }
      return out;
    }
    case "float8_e4m3": {
      const out = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) out[i] = floatToFp8E4M3(values[i]);
      return out;
    }
  }
}

function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x03ff;
  if (exp === 0) return sign * mant * Math.pow(2, -24);
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * Math.pow(2, exp - 15);
}

function floatToHalf(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  const sign = value < 0 ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (!Number.isFinite(abs)) return sign | 0x7c00;
  const exp = Math.floor(Math.log2(abs));
  if (exp > 15) return sign | 0x7c00; // overflow -> infinity
  if (exp < -24) return sign; // underflow -> zero
  if (exp < -14) {
    // Subnormal half.
    return sign | Math.round(abs * Math.pow(2, 24));
  }
  const mant = Math.round((abs / Math.pow(2, exp) - 1) * 1024);
  return sign | ((exp + 15) << 10) | mant;
}

/**
 * E4M3 decode, per the OCP spec (no infinities; 0x7f/0xff are NaN,
 * 0x7e/0xfe is the max finite value 448) — matching the spec-correct
 * decoder in src/model/manifest.ts. NOTE: this previously treated
 * every exp==15 byte as Inf/NaN, so the encoder's own saturation
 * value 0x7e decoded back to NaN; a max-magnitude element quantized
 * to FP8 came back as NaN and silently zeroed every error bound
 * that touched it (NaN comparisons are false).
 */
function fp8E4M3ToFloat(byte: number): number {
  const sign = (byte & 0x80) !== 0 ? -1 : 1;
  const exp = (byte >> 3) & 0x0f;
  const mant = byte & 0x07;
  if (exp === 0) return sign * (mant / 8) * Math.pow(2, -6);
  if (exp === 15) return mant === 7 ? NaN : sign * (1 + mant / 8) * Math.pow(2, 8);
  return sign * (1 + mant / 8) * Math.pow(2, exp - 7);
}

function floatToFp8E4M3(value: number): number {
  if (Number.isNaN(value)) return 0x7f;
  const sign = value < 0 ? 0x80 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  // No infinity in E4M3: saturate at the max finite value (448, 0x7e).
  if (!Number.isFinite(abs)) return sign | 0x7e;
  const exp = Math.floor(Math.log2(abs));
  if (exp > 8) return sign | 0x7e; // saturate at max finite (448)
  if (exp < -10) return sign;
  const biased = exp + 7;
  if (biased <= 0) return sign | Math.round(abs * Math.pow(2, 9));
  let mant = Math.round((abs / Math.pow(2, exp) - 1) * 8);
  let b = biased;
  if (mant > 7) {
    // Round-up carries into the exponent (as in floatToHalf above).
    mant = 0;
    b += 1;
  }
  // 0x7f is NaN, not 480: saturate instead of emitting it.
  if (b > 15 || (b === 15 && mant === 7)) return sign | 0x7e;
  return sign | (b << 3) | mant;
}

// ---------------------------------------------------------------------------
// VirtualGpu — coordinator (orchestrator side)
// ---------------------------------------------------------------------------

export interface VirtualGpuOpts {
  nodes: NodeProvider;
  ram: IVirtualRam;
}

interface NodeTarget {
  device: VirtualGpuDevice;
  node: WorkerInfo;
}

export class VirtualGpu implements IVirtualGpu {
  private nodes: NodeProvider;
  private ram: IVirtualRam;
  /** Nodes lost via handleNodeLoss; never selected again. */
  private deadNodes = new Set<string>();
  /** Node ids ever presented as a virtual device (so handleNodeLoss can
   *  count a loss even if the provider already marked the node dead). */
  private seenNodes = new Set<string>();
  /** Per-stream promise chain; a launch runs only after the previous one
   *  on the same stream settles, so kernels on one stream execute in order.
   *  The stored chain swallows errors so one failed kernel does not block
   *  later kernels on the stream. */
  private streamChains = new Map<string, Promise<void>>();
  /** Launches queued or running per stream (for handleNodeLoss accounting). */
  private streamPending = new Map<string, number>();
  /** Node currently executing work for a stream (for handleNodeLoss). */
  private streamTargets = new Map<string, Map<string, number>>();
  /** In-flight kernels per node id (placement: least-loaded device). */
  private inFlight = new Map<string, number>();
  private kernelsCompleted = 0;
  private kernelsFailed = 0;
  private kernelCounter = 0;
  private streamCounter = 0;
  /**
   * Allocations registered via registerAllocation(), keyed by allocation id.
   * IVirtualRam has no lookup-by-address, so transfer() resolves the
   * allocation containing a vaddr from this registry. The CLI registers
   * every allocation it creates for weight tensors after allocating them.
   */
  private allocations = new Map<string, VramAllocation>();

  constructor(opts: VirtualGpuOpts) {
    this.nodes = opts.nodes;
    this.ram = opts.ram;
  }

  /** Register a vRAM allocation so transfer() can find it by address. */
  registerAllocation(alloc: VramAllocation): void {
    this.allocations.set(alloc.id, alloc);
  }

  listDevices(): VirtualGpuDevice[] {
    const devices = this.nodes
      .listNodes()
      .filter((n) => n.state === "healthy" && !this.deadNodes.has(n.workerId))
      .map((n) => ({
        id: `vgpu-${n.workerId}`,
        // Approximation: one compute unit per TFLOP/s of sustained donor
        // throughput, at least 1. This is a coarse sizing hint for the
        // platform above, not a hardware CU count — donors report
        // flopsPerSec themselves and stats never inflate it.
        computeUnits: Math.max(1, Math.round(n.flopsPerSec / 1e12)),
        memoryBytes: n.freeMemoryBytes,
        flopsPerSec: n.flopsPerSec,
        backedBy: n.device,
      }));
    for (const d of devices) this.seenNodes.add(d.id.slice("vgpu-".length));
    return devices;
  }

  createStream(): string {
    this.streamCounter += 1;
    return `stream-${this.streamCounter}`;
  }

  async launch(launch: KernelLaunch): Promise<KernelResult> {
    if (launch.streamId !== undefined) {
      const streamId = launch.streamId;
      const prev = this.streamChains.get(streamId) ?? Promise.resolve();
      this.streamPending.set(streamId, (this.streamPending.get(streamId) ?? 0) + 1);
      const run = prev.then(() => this.executeLaunch(launch, streamId));
      const settle = (): void => {
        this.streamPending.set(streamId, Math.max(0, (this.streamPending.get(streamId) ?? 1) - 1));
      };
      run.then(settle, settle);
      // The chain swallows errors so a failed kernel does not block the
      // stream; the caller still sees this launch's rejection via `run`.
      this.streamChains.set(
        streamId,
        run.then(
          () => undefined,
          () => undefined,
        ),
      );
      return run;
    }
    // No stream: run immediately, unserialized.
    return this.executeLaunch(launch, "default");
  }

  async synchronize(streamId: string): Promise<void> {
    await (this.streamChains.get(streamId) ?? Promise.resolve());
  }

  async transfer(vaddr: VAddr, targetNodeId: string): Promise<void> {
    for (const alloc of this.allocations.values()) {
      if (vaddr >= alloc.vaddr && vaddr < alloc.vaddr + alloc.sizeBytes) {
        await this.ram.replicate(alloc.id, targetNodeId);
        return;
      }
    }
    throw new Error(
      `vGPU: no registered allocation contains vaddr ${vaddr} — ` +
        `register weight-tensor allocations with registerAllocation() first`,
    );
  }

  async handleNodeLoss(nodeId: string): Promise<{ streamsReassigned: number; devicesLost: number }> {
    // A device counts as lost if this pool ever presented it, even when the
    // provider already marked the node dead before this call (the registry
    // usually flips state first, so listDevices() alone would miss it).
    const known =
      this.seenNodes.has(nodeId) ||
      this.nodes.listNodes().some((n) => n.workerId === nodeId);
    const devicesLost = known && !this.deadNodes.has(nodeId) ? 1 : 0;
    this.deadNodes.add(nodeId);
    // Streams with work currently executing on the dead node are the ones
    // being reassigned: kernels already POSTed to it may fail, but every
    // kernel still queued on those streams picks its device lazily at
    // execution time (see executeLaunch), so it lands on a survivor.
    let streamsReassigned = 0;
    for (const targets of this.streamTargets.values()) {
      if ((targets.get(nodeId) ?? 0) > 0) streamsReassigned += 1;
    }
    return { streamsReassigned, devicesLost };
  }

  stats(): GpuStats {
    const devices = this.listDevices();
    return {
      // Inventory view: `device` here is what backs the virtual device on
      // the donor. KernelResult.device is different — it reports the device
      // that ACTUALLY executed ('cpu' for the fallback).
      devices: devices.map((d) => ({
        nodeId: d.id.slice("vgpu-".length),
        device: d.backedBy,
        flopsPerSec: d.flopsPerSec,
      })),
      totalFlopsPerSec: devices.reduce((sum, d) => sum + d.flopsPerSec, 0),
      kernelsCompleted: this.kernelsCompleted,
      kernelsFailed: this.kernelsFailed,
    };
  }

  /**
   * Pick the donor for a launch. An explicit deviceId is honored (and an
   * unknown/dead device is an error, not a silent reroute). Otherwise the
   * least-loaded device by in-flight kernels wins. Placement by input-data
   * locality is not possible here — this coordinator does not track vRAM
   * page placement — so load is the documented tiebreaker.
   */
  private pickDevice(requestedId?: string): NodeTarget {
    const devices = this.listDevices();
    if (devices.length === 0) {
      throw new Error("vGPU: no healthy donor nodes available for kernel launch");
    }
    const byId = new Map<string, WorkerInfo>();
    for (const n of this.nodes.listNodes()) {
      if (n.state === "healthy" && !this.deadNodes.has(n.workerId)) byId.set(n.workerId, n);
    }
    if (requestedId !== undefined) {
      const device = devices.find((d) => d.id === requestedId);
      if (device === undefined) {
        throw new Error(`vGPU: unknown or unavailable virtual device ${requestedId}`);
      }
      const node = byId.get(device.id.slice("vgpu-".length));
      if (node === undefined) throw new Error(`vGPU: no donor node for device ${requestedId}`);
      return { device, node };
    }
    let best = devices[0];
    let bestLoad = this.inFlight.get(best.id.slice("vgpu-".length)) ?? 0;
    for (const device of devices.slice(1)) {
      const load = this.inFlight.get(device.id.slice("vgpu-".length)) ?? 0;
      if (load < bestLoad) {
        best = device;
        bestLoad = load;
      }
    }
    const node = byId.get(best.id.slice("vgpu-".length));
    if (node === undefined) throw new Error(`vGPU: no donor node for device ${best.id}`);
    return { device: best, node };
  }

  /**
   * Execute one launch: resolve input TensorRefs to bytes by reading them
   * from vRAM (coordinator-side resolution — the node never sees virtual
   * addresses), POST the kernel to the chosen donor's /hw/gpu/kernel
   * route with inputs inline as base64, and write returned outputs back
   * to vRAM at the output TensorRefs.
   */
  private async executeLaunch(launch: KernelLaunch, streamId: string): Promise<KernelResult> {
    const { node } = this.pickDevice(launch.deviceId);
    const nodeId = node.workerId;
    if (node.hwPort === undefined) {
      throw new Error(`vGPU: donor node ${nodeId} does not serve virtual hardware (no hwPort)`);
    }
    this.inFlight.set(nodeId, (this.inFlight.get(nodeId) ?? 0) + 1);
    const targets = this.streamTargets.get(streamId) ?? new Map<string, number>();
    targets.set(nodeId, (targets.get(nodeId) ?? 0) + 1);
    this.streamTargets.set(streamId, targets);
    const kernelId = `kernel-${++this.kernelCounter}`;
    try {
      const inputs: Array<{ dataB64: string; shape: number[]; dtype: WeightDtype }> = [];
      for (const ref of launch.inputs) {
        const bytes = await this.ram.read(ref.vaddr, numel(ref.shape) * dtypeByteSize(ref.dtype));
        const values = decodeToFloat32(bytes, ref.dtype);
        inputs.push({
          dataB64: Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString("base64"),
          shape: ref.shape,
          dtype: "float32",
        });
      }
      const outputs = launch.outputs.map((o) => ({ shape: o.shape, dtype: "float32" as WeightDtype }));
      const url = `http://${node.host}:${node.hwPort}/hw/gpu/kernel`;
      let res: Response;
      try {
        res = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ op: launch.op, inputs, outputs, params: launch.params }),
        });
      } catch (err) {
        throw new Error(`vGPU: could not reach donor node ${nodeId} at ${url}: ${err instanceof Error ? err.message : String(err)}`);
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`vGPU: donor node ${nodeId} rejected kernel ${launch.op}: HTTP ${res.status} ${text}`);
      }
      const body = (await res.json()) as {
        outputs: Array<{ dataB64: string; shape: number[] }>;
        elapsedMs: number;
        device: WorkerDevice;
      };
      if (!Array.isArray(body.outputs) || body.outputs.length !== launch.outputs.length) {
        throw new Error(
          `vGPU: donor node ${nodeId} returned ${Array.isArray(body.outputs) ? body.outputs.length : 0} outputs, expected ${launch.outputs.length}`,
        );
      }
      for (let i = 0; i < launch.outputs.length; i++) {
        const outRef: TensorRef = launch.outputs[i];
        const raw = Buffer.from(body.outputs[i].dataB64, "base64");
        const values = new Float32Array(
          raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
        );
        if (values.length !== numel(outRef.shape)) {
          throw new Error(
            `vGPU: donor node ${nodeId} returned ${values.length} values for output ${i}, expected ${numel(outRef.shape)}`,
          );
        }
        await this.ram.write(outRef.vaddr, encodeFromFloat32(values, outRef.dtype));
      }
      this.kernelsCompleted += 1;
      return { streamId, kernelId, device: body.device, nodeId, elapsedMs: body.elapsedMs };
    } catch (err) {
      this.kernelsFailed += 1;
      throw err;
    } finally {
      this.inFlight.set(nodeId, Math.max(0, (this.inFlight.get(nodeId) ?? 1) - 1));
      const remaining = (targets.get(nodeId) ?? 1) - 1;
      if (remaining <= 0) targets.delete(nodeId);
      else targets.set(nodeId, remaining);
      if (targets.size === 0) this.streamTargets.delete(streamId);
    }
  }
}

// ---------------------------------------------------------------------------
// GpuNode — donor side (worker side)
// ---------------------------------------------------------------------------

const KERNEL_OPS: ReadonlySet<KernelOp> = new Set([
  "matmul",
  "add",
  "mul",
  "silu",
  "rmsnorm",
  "rope",
  "softmax",
  "attention",
  "embedding",
]);

interface KernelInputWire {
  dataB64: string;
  shape: number[];
  dtype: WeightDtype;
}

interface KernelOutputWire {
  shape: number[];
  dtype: WeightDtype;
}

export interface GpuNodeOpts {
  device: WorkerDevice;
}

export class GpuNode {
  private device: WorkerDevice;
  private kernelsCompleted = 0;
  private kernelsFailed = 0;

  constructor(opts: GpuNodeOpts) {
    this.device = opts.device;
  }

  routes(): Record<string, RouteHandler> {
    return {
      "POST /hw/gpu/kernel": async (body) => this.handleKernel(body),
      "GET /hw/gpu/stats": async () => ({
        kernelsCompleted: this.kernelsCompleted,
        kernelsFailed: this.kernelsFailed,
        // The device that ACTUALLY executed kernels on this node. This
        // reference always runs the CPU fallback (see handleKernel), so
        // this is 'cpu' even when the donor reports cuda/metal.
        device: "cpu" as WorkerDevice,
      }),
    };
  }

  private async handleKernel(body: Record<string, unknown>): Promise<unknown> {
    const started = Date.now();
    try {
      const op = body["op"];
      if (typeof op !== "string" || !KERNEL_OPS.has(op as KernelOp)) {
        throw new Error(
          `unknown kernel op: ${JSON.stringify(op)} (expected one of: ${[...KERNEL_OPS].join(", ")})`,
        );
      }
      const inputs = parseKernelInputs(body["inputs"]);
      const outputs = parseKernelOutputs(body["outputs"]);
      const params = parseParams(body["params"]);
      if (outputs.length === 0) {
        throw new Error(`kernel ${op}: at least one output shape is required`);
      }
      // REFERENCE BACKEND: every op runs through the CPU fallback kernels
      // below, whatever this.device is. A real device backend (CUDA/Metal)
      // would plug in here — dispatching on this.device and running the op
      // natively — and the response's `device` field would then report the
      // device that actually ran it. Until then the honest report is 'cpu'.
      // Matmul specifically runs the tiled kernel, and the worker-thread
      // parallel kernel above MATMUL_PARALLEL_MIN_FLOPS (measured: the
      // parallel path only wins once the work dwarfs the dispatch cost).
      void this.device;
      const result = await this.runOp(op as KernelOp, inputs, params);
      const expected = numel(outputs[0].shape);
      if (result.length !== expected) {
        throw new Error(
          `kernel ${op}: produced ${result.length} values, expected ${expected} for output shape [${outputs[0].shape.join(", ")}]`,
        );
      }
      this.kernelsCompleted += 1;
      return {
        outputs: [
          {
            dataB64: Buffer.from(result.buffer, result.byteOffset, result.byteLength).toString("base64"),
            shape: outputs[0].shape,
          },
        ],
        elapsedMs: Date.now() - started,
        device: "cpu" as WorkerDevice,
      };
    } catch (err) {
      this.kernelsFailed += 1;
      throw err;
    }
  }

  private async runOp(
    op: KernelOp,
    inputs: Array<{ values: Float32Array; shape: number[] }>,
    params: Record<string, number | number[]>,
  ): Promise<Float32Array> {
    const need = (index: number, name: string): Float32Array => {
      if (inputs[index] === undefined) {
        throw new Error(`kernel ${op}: missing input ${index} (${name})`);
      }
      return inputs[index].values;
    };
    const paramNumber = (name: string, fallback?: number): number => {
      const v = params[name];
      if (typeof v === "number") return v;
      if (fallback !== undefined) return fallback;
      throw new Error(`kernel ${op}: missing numeric param ${name}`);
    };
    switch (op) {
      case "matmul": {
        const a = need(0, "a");
        const b = need(1, "b");
        const aShape = inputs[0].shape;
        const bShape = inputs[1].shape;
        if (aShape.length !== 2 || bShape.length !== 2) {
          throw new Error(`kernel matmul: inputs must be 2-D, got [${aShape.join(", ")}] and [${bShape.join(", ")}]`);
        }
        // Large matmuls go parallel (worker threads, bit-identical to the
        // sequential path); everything else runs the cache-blocked tiled
        // kernel. Both are faster than the naive kernel and produce the
        // same values on finite inputs.
        const flops = 2 * aShape[0] * aShape[1] * bShape[1];
        if (flops >= MATMUL_PARALLEL_MIN_FLOPS) {
          return matmulParallel(a, aShape[0], aShape[1], b, bShape[0], bShape[1]);
        }
        return matmulTiled(a, aShape[0], aShape[1], b, bShape[0], bShape[1]);
      }
      case "add":
        return CpuKernels.add(need(0, "a"), need(1, "b"));
      case "mul":
        return CpuKernels.mul(need(0, "a"), need(1, "b"));
      case "silu":
        return CpuKernels.silu(need(0, "x"));
      case "rmsnorm": {
        const x = need(0, "x");
        const xShape = inputs[0].shape;
        const rows = paramNumber("rows", xShape.length >= 2 ? xShape[0] : 1);
        const cols = paramNumber(
          "cols",
          xShape.length >= 2 ? xShape[1] : x.length,
        );
        const eps = paramNumber("eps", 1e-5);
        const weight =
          inputs[1] !== undefined ? inputs[1].values : new Float32Array(cols).fill(1);
        return CpuKernels.rmsNorm(x, rows, cols, weight, eps);
      }
      case "rope": {
        const x = need(0, "x");
        const xShape = inputs[0].shape;
        const rows = paramNumber("rows", xShape.length >= 1 ? xShape[0] : 1);
        const headDim = paramNumber("headDim", x.length / rows);
        const theta = paramNumber("theta", 10000);
        const positionsRaw = params["positions"];
        const positions: number[] =
          Array.isArray(positionsRaw) && positionsRaw.every((v) => typeof v === "number")
            ? (positionsRaw as number[])
            : Array.from({ length: rows }, (_v, i) => i);
        // Copy: rope is in place, and the decoded input buffer is ours, but
        // copying keeps the handler's input handling obviously side-effect free.
        return CpuKernels.rope(Float32Array.from(x), rows, headDim, positions, theta);
      }
      case "softmax": {
        const x = need(0, "x");
        const xShape = inputs[0].shape;
        const rows = paramNumber("rows", xShape.length >= 2 ? xShape[0] : 1);
        const cols = paramNumber("cols", xShape.length >= 2 ? xShape[1] : x.length);
        return CpuKernels.softmaxRows(Float32Array.from(x), rows, cols);
      }
      case "attention": {
        const q = need(0, "q");
        const k = need(1, "k");
        const v = need(2, "v");
        return CpuKernels.attention(q, k, v, {
          numHeads: paramNumber("numHeads"),
          headDim: paramNumber("headDim"),
          numKvHeads:
            typeof params["numKvHeads"] === "number" ? params["numKvHeads"] : undefined,
          causal: paramNumber("causal", 0) === 1,
        });
      }
      case "embedding": {
        const table = need(0, "table");
        const ids = need(1, "ids");
        const tableShape = inputs[0].shape;
        const tableRows = paramNumber("tableRows", tableShape.length >= 2 ? tableShape[0] : 1);
        const dim = paramNumber("dim", tableShape.length >= 2 ? tableShape[1] : table.length);
        return CpuKernels.embeddingLookup(table, tableRows, dim, ids);
      }
    }
  }
}

function parseKernelInputs(value: unknown): Array<{ values: Float32Array; shape: number[] }> {
  if (!Array.isArray(value)) {
    throw new Error("kernel: inputs must be an array of {dataB64, shape, dtype}");
  }
  return value.map((entry, i) => {
    if (typeof entry !== "object" || entry === null) {
      throw new Error(`kernel: input ${i} must be an object`);
    }
    const input = entry as unknown as Partial<KernelInputWire>;
    if (typeof input.dataB64 !== "string") {
      throw new Error(`kernel: input ${i} is missing dataB64`);
    }
    if (!Array.isArray(input.shape) || !input.shape.every((d) => typeof d === "number")) {
      throw new Error(`kernel: input ${i} is missing a numeric shape`);
    }
    if (typeof input.dtype !== "string") {
      throw new Error(`kernel: input ${i} is missing dtype`);
    }
    const raw = Buffer.from(input.dataB64, "base64");
    return { values: decodeToFloat32(raw, input.dtype), shape: input.shape };
  });
}

function parseKernelOutputs(value: unknown): KernelOutputWire[] {
  if (!Array.isArray(value)) {
    throw new Error("kernel: outputs must be an array of {shape, dtype}");
  }
  return value.map((entry, i) => {
    if (typeof entry !== "object" || entry === null) {
      throw new Error(`kernel: output ${i} must be an object`);
    }
    const output = entry as unknown as Partial<KernelOutputWire>;
    if (!Array.isArray(output.shape) || !output.shape.every((d) => typeof d === "number")) {
      throw new Error(`kernel: output ${i} is missing a numeric shape`);
    }
    if (typeof output.dtype !== "string") {
      throw new Error(`kernel: output ${i} is missing dtype`);
    }
    return { shape: output.shape, dtype: output.dtype };
  });
}

function parseParams(value: unknown): Record<string, number | number[]> {
  if (value === undefined) return {};
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("kernel: params must be an object");
  }
  const params: Record<string, number | number[]> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (typeof entry === "number") params[key] = entry;
    else if (Array.isArray(entry) && entry.every((v) => typeof v === "number")) {
      params[key] = entry as number[];
    } else {
      throw new Error(`kernel: param ${key} must be a number or number[]`);
    }
  }
  return params;
}
