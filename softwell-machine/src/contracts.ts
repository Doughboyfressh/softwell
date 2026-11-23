/**
 * softwell-machine — shared contracts.
 *
 * Every module in this repo builds against this file. It is the single source
 * of truth for the worker protocol, the shard-plan schema, the engine/backend
 * interfaces, the scheduler interface, and the public API shapes.
 *
 * Transport note: the reference implementation speaks plain HTTP + JSON
 * (Node built-ins only). Hidden states travel as JSON number arrays. That is
 * fine at demo scale (hidden size in the hundreds) and deliberately simple;
 * a production deployment would swap the transport for binary/gRPC without
 * changing these shapes.
 *
 * Runtime note: code runs under Node >= 22 with type stripping, so this file
 * (and all code in the repo) must use erasable TypeScript only: no enums,
 * no parameter properties, no namespaces, no `const enum`.
 */

import type { PackedKV, PackedTensor, SwMode } from "./engine/novel-numbers.ts";
import type { BlockStats } from "./engine/novel-attention.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const DEFAULT_ORCHESTRATOR_PORT = 8410;
export const DEFAULT_WORKER_PORT = 0; // 0 = ephemeral, chosen by the OS
export const HEARTBEAT_INTERVAL_MS = 2000;
export const HEARTBEAT_TIMEOUT_MS = 6000; // missed heartbeats before "suspect"
export const HEARTBEAT_DEAD_MS = 15000; // missed heartbeats before "dead"
export const MESH_PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------------
// Model layer — safetensors-compatible manifest + config
// ---------------------------------------------------------------------------

export type WeightDtype = "bfloat16" | "float16" | "float32" | "float8_e4m3";

export interface ModelConfig {
  architectures: string[]; // e.g. ["LlamaForCausalLM"] or ["MixtralForCausalLM"]
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number; // GQA: < numAttentionHeads
  headDim: number; // usually hiddenSize / numAttentionHeads
  maxPositionEmbeddings: number;
  rmsNormEps: number;
  ropeTheta: number;
  tieWordEmbeddings: boolean;
  // MoE (Mixtral-style). Absent for dense models.
  numExperts?: number;
  numExpertsPerToken?: number;
  normTopKProb?: boolean;
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number;
}

export interface TensorMeta {
  dtype: WeightDtype;
  shape: number[];
  /** Byte offsets [start, end) into the weight file's data section. */
  dataOffsets: [number, number];
}

export interface WeightFile {
  filename: string;
  sizeBytes: number;
  sha256?: string;
}

export interface ModelManifest {
  format: "safetensors";
  modelId: string; // e.g. "meta-llama/Llama-3.3-70B-Instruct" or a local path id
  config: ModelConfig;
  metadata: Record<string, string>;
  /** Tensor name -> metadata. Names follow the HF Llama/Mixtral convention:
   *  model.embed_tokens.weight
   *  model.layers.{i}.input_layernorm.weight
   *  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
   *  model.layers.{i}.post_attention_layernorm.weight
   *  model.layers.{i}.mlp.{gate,up,down}_proj.weight          (dense)
   *  model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight (MoE)
   *  model.layers.{i}.mlp.router.weight                        (MoE)
   *  model.norm.weight
   *  lm_head.weight
   */
  tensors: Record<string, TensorMeta>;
  files: WeightFile[];
  totalBytes: number;
  totalParams: number;
}

/** A loaded tensor: always Float32Array in memory (dequantized on load). */
export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export interface WeightLoader {
  readonly manifest: ModelManifest;
  hasTensor(name: string): boolean;
  /** Load one tensor, dequantized to float32. */
  loadTensor(name: string): Promise<Float32Array>;
  close(): void;
}

/**
 * Where one tensor's raw bytes live in pooled vRAM. Staged bytes are the
 * RAW shard file bytes (hypervisor.stageToVram streams vSSD -> vRAM), so
 * a tensor's bytes sit at `vaddr + headerLen + dataOffsets[0]` for
 * `dataOffsets[1] - dataOffsets[0]` bytes, where `headerLen` is the
 * safetensors header length (8-byte prefix + JSON) of the shard that
 * holds the tensor. `allocId` names the vRAM allocation (one per shard)
 * so the bytes can be re-fetched after healing. `dtype`/`shape` repeat
 * the manifest's TensorMeta so a loader can validate what it fetched.
 */
export interface VramTensorLocation {
  allocId: string;
  vaddr: VAddr;
  headerLen: number;
  dataOffsets: [number, number];
  dtype: WeightDtype;
  shape: number[];
}

/** Tensor name -> its location in pooled vRAM (one entry per tensor). */
export type VramWeightMap = Record<string, VramTensorLocation>;

/**
 * What a worker needs to build a VramWeightLoader for a staged model:
 * the manifest (config + tensor metadata) plus the weight map. Served
 * as JSON by GET /mesh/weights/map?model=<id>; the tensor bytes
 * themselves travel separately as octet-stream (GET /mesh/weights).
 */
export interface WeightMapResponse {
  modelId: string;
  manifest: ModelManifest;
  weightMap: VramWeightMap;
}

// --- vRAM chunk ops (remote VramAccess) ------------------------------------
// JSON shapes for POST /mesh/vram/alloc and POST /mesh/vram/free. Reads
// (GET /mesh/vram/read) and writes (POST /mesh/vram/write) carry raw
// bytes as application/octet-stream — never JSON/base64 — with the
// address in the query string (read) or as `vaddr` alongside the body
// (write). See src/server/vram-kv.ts.

/** POST /mesh/vram/alloc body: bytes to allocate, copies per page. */
export interface VramAllocRequest {
  sizeBytes: number;
  replicas?: number;
}

/** POST /mesh/vram/alloc response: the allocation the worker holds. */
export interface VramAllocResponse {
  id: string;
  vaddr: number;
  /** Bytes actually reserved (sizeBytes rounded up to whole 4 MiB pages). */
  sizeBytes: number;
}

/** POST /mesh/vram/write response. */
export interface VramWriteResponse {
  bytesWritten: number;
}

/** POST /mesh/vram/free response. */
export interface VramFreeResponse {
  freed: string;
}

/** Minimal tokenizer contract. The reference impl is byte-level (see model/tokenizer.ts). */
export interface Tokenizer {
  readonly vocabSize: number;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  encode(text: string): number[];
  decode(tokenIds: number[]): string;
}

// ---------------------------------------------------------------------------
// Execution engine — transformer forward pass + pluggable backends
// ---------------------------------------------------------------------------

export interface ForwardContext {
  /** One seq id per input row (parallel to `positions`). */
  seqIds: string[];
  /** Absolute positions of the tokens in `input` (length == batch tokens). */
  positions: number[];
  isPrefill: boolean;
  /** Worker-local paged KV cache for this stage's layers. */
  cache: PagedKVCache;
}

/**
 * One pipeline stage: a contiguous range of transformer layers (plus the
 * embedding on stage 0 and the lm_head on the last stage) owned by one
 * worker. `forward` consumes hidden states and returns hidden states — or
 * logits when this stage has the lm_head.
 */
export interface StageExecutor {
  readonly assignment: ShardAssignment;
  load(loader: WeightLoader): Promise<void>;
  embed(tokenIds: number[]): Tensor;
  forward(input: Tensor, ctx: ForwardContext): Promise<Tensor>;
  /** Bytes of weight memory this stage holds. */
  memoryBytes(): number;
}

export interface EngineBackend {
  readonly name: string; // e.g. "softwell-ts" (reference) — "cuda-fp8" later
  createStageExecutor(
    manifest: ModelManifest,
    assignment: ShardAssignment,
    options?: EngineOptions,
  ): StageExecutor;
}

/**
 * Engine options: which novel paths a stage executor uses. All fields
 * default to the baseline (exact attention, fp32 weights and KV,
 * autoregressive decoding) when omitted — see DEFAULT_ENGINE_OPTIONS in
 * src/engine/options.ts. These are additive: existing callers that pass
 * no options get exactly the v1 behaviour.
 *
 * - attention: "exact" (grouped-query attention), "landmark" (Theorem A1,
 *   docs/research/01-attention.md), "sparse" (Theorem A2).
 * - weightFormat: how projection weights are stored and multiplied:
 *   "fp32" (baseline) or SW-code "sw2"/"sw3"/"sw4" (Theorem N1,
 *   docs/research/02-numbers.md).
 * - kvFormat: how the KV cache is stored for attention: "fp32" (baseline)
 *   or SW-code "sw2"/"sw3"/"sw4" (Theorem N2).
 * - decoding: how the scheduler generates tokens for a sequence:
 *   "autoregressive" (baseline), "speculative" (Theorem S1) or "jacobi"
 *   (Theorem S2, docs/research/03-decoding.md). Greedy only in the
 *   scheduler: sequences with temperature > 0 fall back to autoregressive.
 * - draftLength: K, the draft length (speculative) / block size (Jacobi).
 *   Default 4.
 * - npu: which vNPU execution format a stage executor uses when it
 *   runs on the virtual NPU: "off" (baseline — the executor never
 *   issues vNPU instructions), "int8" or "fp8". Default "off". With
 *   "int8"/"fp8" the forward pass issues its matmuls as MATMUL_INT8 /
 *   MATMUL_FP8 instructions on a VirtualNpu (an injected device, or
 *   a local in-process one built by the executor) against weights
 *   staged into the device's SRAM at load (docs/VNPU.md).
 */
export type AttentionMode = "exact" | "landmark" | "sparse";
export type WeightFormat = "fp32" | "sw2" | "sw3" | "sw4";
export type KvFormat = "fp32" | "sw2" | "sw3" | "sw4";
export type DecodingMode = "autoregressive" | "speculative" | "jacobi";
export type NpuMode = "int8" | "fp8" | "off";

export interface EngineOptions {
  attention?: AttentionMode;
  weightFormat?: WeightFormat;
  kvFormat?: KvFormat;
  decoding?: DecodingMode;
  draftLength?: number;
  npu?: NpuMode;
}

export interface SamplingParams {
  temperature: number; // 0 = greedy
  topP: number;
  topK: number; // 0 = disabled
  maxTokens: number;
  stop?: string[];
  seed?: number;
}

export type Sampler = (logits: Float32Array, params: SamplingParams) => number;

// ---------------------------------------------------------------------------
// Paged KV cache
// ---------------------------------------------------------------------------

/**
 * Block-paged KV cache, one instance per pipeline stage (worker-local).
 * Physical K/V for a layer lives in fixed-size blocks; a sequence's blocks
 * are tracked by a block table. The scheduler only needs `usedBlocks()` for
 * admission control; the engine owns allocation.
 *
 * allocate/append/read/free are async: a vRAM-backed cache is remote
 * (HTTP to donors), and the interface is one shape for both stores
 * (docs/KV-DESIGN.md §2). usedBlocks()/totalBlocks() stay synchronous —
 * pure in-memory counters in both implementations.
 */
export interface PagedKVCache {
  readonly blockSize: number; // tokens per block
  allocate(seqId: string, layer: number, numTokens: number): Promise<number[]>;
  append(seqId: string, layer: number, position: number, key: Float32Array, value: Float32Array): Promise<void>;
  /** Concatenated K and V for `positions` (in order) for one layer. */
  read(seqId: string, layer: number, positions: number[]): Promise<{ keys: Float32Array; values: Float32Array }>;
  free(seqId: string): Promise<void>;
  /**
   * Release everything this cache holds — all sequences, all layers.
   * For a vRAM-backed cache this flushes staged chunks first (so no
   * committed KV sits un-replicated in host staging) and returns every
   * chunk to the pool; for the host cache it resets the tables. The
   * worker calls it on model switch and on shutdown so a vRAM cache
   * cannot leak chunks (docs/KV-DESIGN.md §6.3, P4). Best-effort: a
   * close that cannot free a chunk counts the failure and still
   * releases the rest rather than rejecting.
   */
  close(): Promise<void>;
  usedBlocks(): number;
  totalBlocks(): number;
}

// ---------------------------------------------------------------------------
// Worker mesh — capabilities, registration, heartbeat, fault detection
// ---------------------------------------------------------------------------

export type WorkerRole = "prefill" | "decode" | "mixed";
export type WorkerDevice = "cpu" | "cuda" | "metal" | "rocm";
export type WorkerState = "joining" | "healthy" | "suspect" | "dead";

export interface WorkerCapability {
  host: string;
  port: number;
  /** Port of this worker's virtual-hardware node server (vRAM/vGPU/vCPU/vSSD
   *  routes). Absent when the worker does not serve virtual hardware. */
  hwPort?: number;
  /** Physical CPU cores on the donor (os.cpus().length). Absent when the
   *  donor does not report it; vCPU must never assume a value. */
  cores?: number;
  device: WorkerDevice;
  backend: string; // engine backend name this worker runs, e.g. "softwell-ts"
  memoryBytes: number;
  freeMemoryBytes: number;
  /** Sustained matmul throughput the worker can deliver. */
  flopsPerSec: number;
  /** Inter-worker link bandwidth (bytes/sec) this worker can sustain. */
  bandwidthBytesPerSec: number;
  roles: WorkerRole[];
  maxBatchTokens: number;
}

export interface WorkerInfo extends WorkerCapability {
  workerId: string;
  state: WorkerState;
  registeredAt: number;
  lastHeartbeatAt: number;
  usedMemoryBytes: number;
  activeSequences: number;
}

// Mesh HTTP messages (JSON bodies). Orchestrator base URL is passed to workers
// on the command line / env (SOFTWELL_ORCHESTRATOR_URL).

export interface RegisterRequest {
  protocolVersion: number;
  capability: WorkerCapability;
}
export interface RegisterResponse {
  workerId: string;
  heartbeatIntervalMs: number;
}
export interface HeartbeatRequest {
  workerId: string;
  usedMemoryBytes: number;
  activeSequences: number;
  /** Tokens generated by this worker since last heartbeat (for throughput). */
  tokensSinceHeartbeat: number;
}
export interface HeartbeatResponse {
  ok: boolean;
  /** Assignment version the worker should be running; if newer than the
   *  worker's, it must GET /mesh/assignment. */
  assignmentVersion: number;
}
export interface AssignmentResponse {
  version: number;
  assignment: ShardAssignment | null; // null = no assignment yet
}

// A single pipeline-stage execution request (orchestrator -> worker).
export interface ExecuteRequest {
  workerId: string;
  /** One seq id per input row (parallel to `input` rows and `positions`). */
  seqIds: string[];
  kind: "prefill" | "decode";
  stage: number;
  /** Hidden states in, as a flat row-major array. Shape [tokens, hidden]. */
  input: { data: number[]; shape: number[] };
  /** Absolute position of each input row. */
  positions: number[];
  /** When set and this stage has the embedding, embed these token ids
   *  instead of using `input` (length == number of rows). */
  tokenIds?: number[];
}
export interface ExecuteResponse {
  /** Hidden states out — or logits when this stage has the lm_head. */
  output: { data: number[]; shape: number[] };
  isLogits: boolean;
  tokensProcessed: number;
}

// ---------------------------------------------------------------------------
// Sharding planner
// ---------------------------------------------------------------------------

export interface ShardAssignment {
  workerId: string;
  stage: number; // pipeline stage index, 0..N-1
  layerStart: number; // inclusive
  layerEnd: number; // exclusive
  hasEmbedding: boolean;
  hasLmHead: boolean;
  tensorParallelRank: number;
  tensorParallelSize: number;
  /** Expert-parallel: expert ids this worker owns (MoE only). */
  experts: number[];
  estimatedMemoryBytes: number;
  /** Model this assignment serves. When it changes, the worker swaps its
   *  weight loader to this model (see src/mesh/worker.ts). Absent on
   *  assignments built before model switching existed. */
  modelId?: string;
  /** Local directory holding this model's files (same filesystem as the
   *  worker in the reference implementation; multi-host model
   *  distribution via vSSD is a documented follow-up). */
  modelDir?: string;
}

export interface ShardPlan {
  modelId: string;
  totalParams: number;
  totalWeightBytes: number;
  pipelineStages: number;
  tensorParallelSize: number;
  expertParallelSize: number;
  assignments: ShardAssignment[];
  /** Cost-model estimate for decode throughput (tokens/sec). */
  estimatedTokensPerSec: number;
  /** Cost-model estimate for time-to-first-token (ms) for a 1k-token prompt. */
  estimatedTtftMs: number;
  /** Human-readable bottleneck, e.g. "memory-bandwidth-bound decode". */
  bottleneck: string;
  createdAt: string;
}

export interface PlannerInput {
  manifest: ModelManifest;
  workers: WorkerInfo[];
  /** Bytes per token of KV cache per layer (derived from config if omitted). */
  kvBytesPerTokenPerLayer?: number;
  /** Assumed concurrent sequences, for KV-cache headroom. */
  expectedSequences?: number;
}

// ---------------------------------------------------------------------------
// Scheduler — continuous batching, prefill/decode disaggregation
// ---------------------------------------------------------------------------

export interface GenerateRequest {
  id: string;
  prompt: string;
  tokenIds: number[]; // already tokenized prompt
  sampling: SamplingParams;
  /** Preferred role routing; "mixed" lets the scheduler decide. */
  role: WorkerRole;
}

export type SequenceStatus = "waiting" | "prefill" | "decoding" | "done" | "error";

export interface SequenceState {
  id: string;
  promptTokenIds: number[];
  generatedTokenIds: number[];
  status: SequenceStatus;
  createdAt: number;
  prefillDoneAt?: number;
  finishReason?: string | null;
  error?: string;
}

export interface GeneratedToken {
  seqId: string;
  tokenId: number;
  text: string;
  finishReason: string | null;
}

/**
 * Continuous-batching scheduler. The orchestrator owns one scheduler; each
 * step() runs one decode step for the current batch (prefill for newly
 * admitted sequences, one decode token for running ones) by driving the mesh.
 * Prefill and decode are routed to different worker roles when the plan
 * provides dedicated prefill/decode workers; otherwise both run on "mixed".
 */
export interface Scheduler {
  submit(req: GenerateRequest): string;
  /** Run one scheduling step; returns tokens produced this step. */
  step(): Promise<GeneratedToken[]>;
  getSequence(id: string): SequenceState | undefined;
  pendingCount(): number;
  runningCount(): number;
  /**
   * Cancel a waiting or running sequence: it stops generating, its
   * worker-side KV cache is freed, and its state becomes "done" with
   * finishReason "cancelled". Returns false when no such sequence exists
   * (already done/error, or unknown id). Never throws for an unknown id.
   */
  cancel(id: string): boolean;
}

// ---------------------------------------------------------------------------
// Public API — OpenAI-compatible
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  stop?: string[];
  seed?: number;
}

export interface ChatCompletionChoice {
  index: number;
  message?: ChatMessage;
  delta?: Partial<ChatMessage>;
  finish_reason: string | null;
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
}

export interface ModelInfo {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

// ---------------------------------------------------------------------------
// Cluster snapshot (dashboard + status endpoints)
// ---------------------------------------------------------------------------

export interface ClusterSnapshot {
  modelId: string | null;
  plan: ShardPlan | null;
  workers: WorkerInfo[];
  sequences: { waiting: number; running: number; done: number };
  tokensPerSec: number;
  avgLatencyMs: number;
  uptimeSec: number;
  /** Pooled virtual-machine view (present once the hypervisor has booted). */
  vm?: VirtualMachineInfo;
}

// ---------------------------------------------------------------------------
// Virtual hardware — the virtual computer (vRAM / vGPU / vCPU / vSSD).
// THE BOUNDARY: this section is the complete hardware platform. Workloads —
// including the LLM runtime — are written against THESE interfaces only:
// the virtual CPU task ISA, the virtual GPU kernel/stream model, the
// virtual memory hierarchy (one global address space), and virtual storage.
// They never touch a physical device. Physical machines are RESOURCE DONORS
// underneath: they contribute cycles, bytes, and bandwidth to the pools.
// The mapping virtual-device -> donor lives INSIDE each pool coordinator
// and its node components, never above this boundary. A pool coordinator
// speaks only to donor nodes through /hw/* routes; a node component
// translates virtual ops into whatever its donor physically has (a real
// GPU, or the CPU fallback kernels). If a donor dies, the pool heals
// itself through handleNodeLoss and the platform above never sees it.
// ---------------------------------------------------------------------------

export const VRAM_PAGE_SIZE_BYTES = 4 * 1024 * 1024; // 4 MiB pages
export const VSSD_CHUNK_SIZE_BYTES = 64 * 1024 * 1024; // 64 MiB chunks

/** Byte offset in the global virtual address space. */
export type VAddr = number;

/** One HTTP route handler on a worker's virtual-hardware node server. */
export type RouteHandler = (body: Record<string, unknown>) => Promise<unknown>;

// --- vRAM: pooled, unified memory across nodes ------------------------------

export interface PageRef {
  pageId: string;
  nodeId: string;
  version: number;
  state: "resident" | "loading" | "dirty" | "replicating";
}

export interface VramAllocation {
  id: string;
  vaddr: VAddr;
  sizeBytes: number;
  pages: PageRef[];
  replicas: number;
}

export interface VramStats {
  totalBytes: number;
  usedBytes: number;
  freeBytes: number;
  pages: number;
  nodes: number;
}

export interface IVirtualRam {
  alloc(sizeBytes: number, opts?: { replicas?: number; nodeAffinity?: string[] }): Promise<VramAllocation>;
  free(allocId: string): Promise<void>;
  read(vaddr: VAddr, length: number): Promise<Uint8Array>;
  write(vaddr: VAddr, data: Uint8Array): Promise<void>;
  replicate(allocId: string, targetNodeId: string): Promise<void>;
  /** Heal after a donor dies: re-replicate pages that lost their last copy. */
  handleNodeLoss(nodeId: string): Promise<{ pagesRecovered: number; pagesLost: number }>;
  stats(): VramStats;
}

/**
 * Minimal remote-vRAM access: the surface `VramPagedKVCache`
 * (src/scheduler/kvcache-vram.ts) needs. Lives here (not in the
 * prototype file) so the worker, the mesh client, and tests can
 * reference it without importing the cache implementation.
 * Implementations: `VirtualRamAccess` (in-process, in the prototype
 * file) and `RemoteVramAccess` (HTTP, src/scheduler/kvcache-vram-remote.ts).
 * All methods are async: vRAM lives across the network from the worker.
 */
export interface VramAccess {
  /** Read `length` bytes starting at virtual address `vaddr`. */
  read(vaddr: number, length: number): Promise<Uint8Array>;
  /** Write `data` starting at virtual address `vaddr`. */
  write(vaddr: number, data: Uint8Array): Promise<void>;
  /**
   * Allocate `sizeBytes` (rounded up to whole vRAM pages by the
   * implementation) and return its id and base virtual address.
   * `opts.replicas` overrides the implementation default when given.
   */
  alloc(
    sizeBytes: number,
    opts?: { replicas?: number },
  ): Promise<{ id: string; vaddr: number }>;
  /** Release an allocation created by alloc(). */
  free(id: string): Promise<void>;
}

// --- vGPU: one virtual GPU dispatching across real devices -----------------

export type KernelOp =
  | "matmul"
  | "add"
  | "mul"
  | "silu"
  | "rmsnorm"
  | "rope"
  | "softmax"
  | "attention"
  | "embedding";

/** A tensor living in vRAM, referenced by virtual address. */
export interface TensorRef {
  vaddr: VAddr;
  shape: number[];
  dtype: WeightDtype;
}

export interface KernelLaunch {
  op: KernelOp;
  inputs: TensorRef[];
  outputs: TensorRef[];
  params: Record<string, number | number[]>;
  streamId?: string;
  /** Virtual GPU device id (see listDevices). Omit for automatic placement. */
  deviceId?: string;
}

/** A VIRTUAL GPU device: the platform's unit of GPU compute. Backed by one
 *  donor's real GPU, or by a slice of donor CPU (fallback). Workloads see
 *  only virtual devices; which donor backs each one is internal to vGPU. */
export interface VirtualGpuDevice {
  id: string;
  computeUnits: number;
  memoryBytes: number;
  flopsPerSec: number;
  /** What backs this virtual device on the donor side (informational). */
  backedBy: WorkerDevice;
}

export interface KernelResult {
  streamId: string;
  kernelId: string;
  device: WorkerDevice;
  nodeId: string;
  elapsedMs: number;
}

export interface GpuStats {
  devices: Array<{ nodeId: string; device: WorkerDevice; flopsPerSec: number }>;
  totalFlopsPerSec: number;
  kernelsCompleted: number;
  kernelsFailed: number;
}

export interface IVirtualGpu {
  /** The virtual devices this GPU presents — the platform's GPU inventory. */
  listDevices(): VirtualGpuDevice[];
  createStream(): string;
  launch(launch: KernelLaunch): Promise<KernelResult>;
  synchronize(streamId: string): Promise<void>;
  /** Migrate pages backing `vaddr` to another node (vRAM <-> vGPU transfer). */
  transfer(vaddr: VAddr, targetNodeId: string): Promise<void>;
  /** Heal after a donor dies: reassign its streams/devices to survivors. */
  handleNodeLoss(nodeId: string): Promise<{ streamsReassigned: number; devicesLost: number }>;
  stats(): GpuStats;
}

// --- vCPU: pooled general compute ------------------------------------------

export interface CpuTask {
  id: string;
  kind: string;
  payload: unknown;
  cpuMillisEstimate?: number;
}

export interface CpuTaskResult {
  taskId: string;
  ok: boolean;
  result?: unknown;
  error?: string;
  nodeId: string;
  elapsedMs: number;
}

export type CpuTaskHandler = (task: CpuTask) => Promise<unknown> | unknown;

export interface CpuStats {
  nodes: number;
  totalCores: number;
  queued: number;
  completed: number;
  failed: number;
}

export interface IVirtualCpu {
  registerHandler(kind: string, handler: CpuTaskHandler): void;
  submit(task: CpuTask): Promise<CpuTaskResult>;
  /** Heal after a donor dies: requeue its in-flight tasks to survivors. */
  handleNodeLoss(nodeId: string): Promise<{ tasksRequeued: number; tasksLost: number }>;
  stats(): CpuStats;
}

// --- vSSD: pooled distributed object store ---------------------------------

export interface SsdObjectMeta {
  key: string;
  sizeBytes: number;
  chunks: number;
  replicas: number;
  createdAt: number;
}

export interface SsdStats {
  objects: number;
  totalBytes: number;
  nodes: number;
}

export interface IVirtualSsd {
  putObject(key: string, data: Uint8Array, opts?: { replicas?: number }): Promise<SsdObjectMeta>;
  getObject(key: string): Promise<Uint8Array>;
  getRange(key: string, offset: number, length: number): Promise<Uint8Array>;
  list(prefix?: string): Promise<SsdObjectMeta[]>;
  deleteObject(key: string): Promise<void>;
  /** Heal after a donor dies: re-replicate chunks that lost their last copy. */
  handleNodeLoss(nodeId: string): Promise<{ chunksRecovered: number; chunksLost: number }>;
  stats(): SsdStats;
}

// --- vNPU: the virtual inference accelerator -------------------------------
//
// The fifth virtual device (docs/VNPU.md): INT8/FP8 execution as
// first-class operations, attention and sparsity as virtual-silicon
// primitives (Theorems A1/A2), and SW-code (Theorems N1/N2) as the
// native number format. Same boundary as the other pools: workloads
// issue instructions against virtual devices and never see a donor.
// The reference implementation executes its instructions in-process
// on the coordinator (CPU fallback, like GpuNode) and reports device
// 'cpu' on every result — it never claims NPU silicon it does not
// have. A donor-side VnpuNode (real NPU silicon) is future work.

/** vNPU instruction mnemonics (docs/VNPU.md §1). */
export type NpuOp =
  | "MATMUL_INT8"
  | "MATMUL_FP8"
  | "MATMUL_SW"
  | "MATMUL_BF16"
  | "ATTN_LANDMARK"
  | "ATTN_SPARSE"
  | "ATTN_EXACT"
  | "ATTN_PACKED_KV"
  | "KV_QUANTIZE"
  | "SOFTMAX"
  | "RMSNORM"
  | "ACT"
  | "QUANTIZE"
  | "DEQUANTIZE";

/** Weight formats a vNPU instruction can stage or multiply. */
export type NpuWeightFormat = "int8" | "fp8" | "bf16" | SwMode;

/** Attention mode for the vNPU attention instructions. */
export type NpuAttentionMode = "exact" | "landmark" | "sparse";

/** A VIRTUAL NPU device: the platform's unit of NPU compute. One per
 *  healthy donor, like the vGPU. Each device carries 64 tiles of
 *  4 MiB tile-local SRAM (design parameter D1, docs/VNPU.md §2.1). */
export interface VirtualNpuDevice {
  id: string;
  /** Tiles per device (64). */
  tiles: number;
  /** On-chip SRAM analogue, bytes (256 MiB — D1). */
  sramBytes: number;
  /** Peak INT8 ops/sec the device model sustains. The reference
   *  implementation reports the donor's own sustained throughput
   *  here with no multiplier — it claims no INT8 rate the donor
   *  does not have. */
  int8Tops: number;
  /** Peak BF16 FLOPS/sec, same honesty rule as int8Tops. */
  bf16Tflops: number;
  /** What backs this virtual device on the donor side (informational). */
  backedBy: WorkerDevice;
}

export interface NpuStats {
  devices: Array<{
    nodeId: string;
    device: WorkerDevice;
    sramBytes: number;
    sramUsedBytes: number;
  }>;
  /** SRAM capacity per device, bytes (D1: 256 MiB). */
  sramCapacityBytes: number;
  /** Resident staged-weight bytes across all devices. */
  sramUsedBytes: number;
  /** Uses of a staged weight that was resident in SRAM. */
  sramHits: number;
  /** Uses of a staged weight that was NOT resident (spilled, evicted,
   *  or lost with its donor) — the weight was streamed from vRAM and
   *  quantized on the fly instead. */
  sramMisses: number;
  /** stageWeights calls that could not fit even an empty SRAM (a
   *  single weight larger than D1). */
  sramSpills: number;
  /** Resident weights evicted (LRU) to make room for a new stage. */
  sramEvictions: number;
  int8Matmuls: number;
  fp8Matmuls: number;
  swMatmuls: number;
  bf16Matmuls: number;
  attentionCalls: number;
  kvQuantizeCalls: number;
  opsCompleted: number;
  opsFailed: number;
}

export interface NpuMatmulOptions {
  /** Execution format. Defaults to the device's configured default
   *  ("int8" unless the device was built with another). */
  format?: "int8" | "fp8" | "bf16";
  /** INT8 weight scale scheme: "per-tensor" (one scale for B) or
   *  "per-channel" (one scale per B output channel — the default).
   *  Ignored for fp8/bf16 (per-tensor scale). */
  weightScale?: "per-tensor" | "per-channel";
  /** Key of a weight staged with stageWeights()/stageFromVram().
   *  When given and resident, the staged quantized copy is used
   *  (sramHit); when given but not resident, the call counts a miss
   *  and quantizes the passed fp32 B on the fly. Unknown keys throw. */
  weightKey?: string;
  /** Virtual NPU device id (see listDevices). Omit for automatic placement. */
  deviceId?: string;
  /** Stream to serialize this instruction on (see createStream).
   *  Omit to run immediately, unserialized. */
  streamId?: string;
}

export interface NpuMatmulResult {
  output: Float32Array;
  /** Rigorous bound on |C_ij - C_ref_ij| for every output element,
   *  where C_ref is the serial FP32 reference (matmulTransposeB /
   *  matmul in src/engine/attention.ts). See docs/VNPU.md §1. */
  errorBound: number;
  /** True when the weight came from resident SRAM. */
  sramHit: boolean;
  format: "int8" | "fp8" | "bf16" | SwMode;
  op: NpuOp;
  /** The device that ACTUALLY executed: 'cpu' for the reference
   *  implementation (same honesty rule as the vGPU fallback). */
  device: WorkerDevice;
}

export interface NpuSwMatmulOptions {
  weightKey?: string;
  deviceId?: string;
  streamId?: string;
}

export interface NpuAttentionOptions {
  /** Which attention instruction: "exact" (ATTN_EXACT), "landmark"
   *  (ATTN_LANDMARK, Theorem A1), "sparse" (ATTN_SPARSE, Theorem A2).
   *  Default "exact". */
  mode?: NpuAttentionMode;
  /** Attention scale; defaults to 1/sqrt(headDim). */
  scale?: number;
  /** Mask out KV positions after the query position (default true). */
  causal?: boolean;
  /** Absolute position of the query token (needed for causal masking). */
  queryPosition?: number;
  /** Absolute positions of the KV entries, in order (needed for causal masking). */
  kvPositions?: number[];
  /** Trailing positions always computed exactly (default 16). */
  localWindow?: number;
  /** KV block size; defaults to ceil(sqrt(visible length)), min 4. */
  blockSize?: number;
  /** A1: largest tolerated per-logit perturbation (default 0.05). */
  deltaMax?: number;
  /** A2: per-block mass threshold (default 1e-3). */
  dropThreshold?: number;
  deviceId?: string;
  streamId?: string;
}

export interface NpuAttentionResult {
  output: Float32Array;
  /** Counted multiply-add-equivalent operations (same convention as
   *  src/engine/novel-attention.ts). */
  ops: number;
  /** Rigorous upper bound on ||output - exact||_2, max over heads
   *  (and rows, for prefill). 0 for ATTN_EXACT. */
  errorBound: number;
  /** Rigorous upper bound on total dropped attention mass (A2; 0 otherwise). */
  droppedMassBound: number;
  /** What exact grouped-query attention would cost, same counting convention. */
  exactOps: number;
  mode: NpuAttentionMode;
  op: NpuOp;
  /** The device that ACTUALLY executed: 'cpu' for the reference implementation. */
  device: WorkerDevice;
}

export interface NpuPackedKvAttentionOptions {
  scale?: number;
  causal?: boolean;
  queryPosition?: number;
  kvPositions?: number[];
  deviceId?: string;
  streamId?: string;
}

export interface NpuStageResult {
  key: string;
  deviceId: string;
  format: NpuWeightFormat;
  /** SRAM bytes the staged (quantized) weight occupies. */
  bytes: number;
  /** False when the weight did not fit even an empty SRAM (spill). */
  resident: boolean;
}

export interface NpuSramStatus {
  /** SRAM capacity per device, bytes (D1). */
  capacityBytes: number;
  /** Resident staged-weight bytes across all devices. */
  usedBytes: number;
  entries: number;
  hits: number;
  misses: number;
  spills: number;
  evictions: number;
}

export interface IVirtualNpu {
  /** The virtual devices this NPU presents — the platform's NPU inventory. */
  listDevices(): VirtualNpuDevice[];
  /** An ordered instruction queue (one per sequence, cf. docs/VNPU.md §3.1). */
  createStream(): string;
  /** Wait for every instruction on a stream to settle. */
  synchronize(streamId: string): Promise<void>;
  /**
   * Stage fp32 weights into a device's SRAM as a quantized resident
   * copy. `format` selects the storage format (int8/fp8/bf16/SW-code).
   * When the weight does not fit, LRU residents are evicted; when it
   * exceeds total SRAM it is recorded as spilled (resident: false)
   * and later uses count misses. The fp32 source is never retained.
   */
  stageWeights(
    key: string,
    data: Float32Array,
    rows: number,
    cols: number,
    format: NpuWeightFormat,
    opts?: { deviceId?: string; weightScale?: "per-tensor" | "per-channel" },
  ): NpuStageResult;
  /**
   * Stage weights from pooled vRAM: read `rows*cols` elements of
   * `dtype` at `vaddr`, decode to fp32, and stage as stageWeights()
   * does. This is the vRAM -> SRAM fetch path (docs/VNPU.md §2).
   */
  stageFromVram(
    key: string,
    vaddr: VAddr,
    rows: number,
    cols: number,
    dtype: WeightDtype,
    format: NpuWeightFormat,
    opts?: { deviceId?: string; weightScale?: "per-tensor" | "per-channel" },
  ): Promise<NpuStageResult>;
  /** Drop a staged weight from SRAM (returns false when unknown). */
  evictWeights(key: string): boolean;
  /** True when `key` is currently resident in SRAM on some device. */
  resident(key: string): boolean;
  sramStatus(): NpuSramStatus;
  /**
   * C = A @ B (MATMUL_INT8 / MATMUL_FP8 / MATMUL_BF16). The engine's
   * matmul shape. A and B are fp32; the device quantizes them per
   * `opts.format` (or a staged weight when `opts.weightKey` is given).
   */
  matmul(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts?: NpuMatmulOptions,
  ): Promise<NpuMatmulResult>;
  /**
   * C = A @ B^T (same instruction set). The shape every projection
   * in the engine has: weights arrive [out, in] and hidden @ W^T is
   * the call (cf. matmulTransposeB in src/engine/attention.ts).
   * Per-channel INT8 scales are per B row (per output channel).
   */
  matmulTransposeB(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: Float32Array,
    bRows: number,
    bCols: number,
    opts?: NpuMatmulOptions,
  ): Promise<NpuMatmulResult>;
  /**
   * C = A @ B^T where B is SW-code packed (MATMUL_SW, Theorem N1).
   * The packed stream is consumed directly, never materialized as
   * fp32 — cf. matmulPackedBT in src/engine/novel-numbers.ts.
   */
  matmulSw(
    a: Float32Array,
    aRows: number,
    aCols: number,
    b: PackedTensor,
    opts?: NpuSwMatmulOptions,
  ): Promise<NpuMatmulResult>;
  /**
   * Single-query attention (ATTN_EXACT / ATTN_LANDMARK / ATTN_SPARSE).
   * Same layouts and masking semantics as groupedQueryAttention in
   * src/engine/attention.ts; q is [numHeads, headDim], k/v are
   * [kvLen, numKvHeads, headDim]. Returns the result plus the
   * theorem error bound for the approximate modes.
   */
  attention(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts?: NpuAttentionOptions,
  ): Promise<NpuAttentionResult>;
  /**
   * Whole-prefill attention: q/k/v are [seqLen, ...], row i attends
   * causally over [0..i]. Same semantics as prefillLandmarkAttention /
   * prefillSparseAttention (or exact multi-row attention).
   */
  prefillAttention(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    numHeads: number,
    numKvHeads: number,
    headDim: number,
    opts?: NpuAttentionOptions,
  ): Promise<NpuAttentionResult>;
  /** QUANTIZE to SW-code (Theorem N1): the native number format. */
  quantizeSw(
    data: Float32Array,
    rows: number,
    cols: number,
    mode: SwMode,
    opts?: { blockSize?: number },
  ): PackedTensor;
  /** DEQUANTIZE from SW-code back to fp32. */
  dequantizeSw(packed: PackedTensor): Float32Array;
  /**
   * KV_QUANTIZE (Theorem N2): quantize a KV cache for one layer —
   * keys and values [kvLen, numKvHeads, headDim], one SW-code block
   * per (token, KV head) vector. Quantize *after* RoPE, as the engine does.
   */
  quantizeKv(
    keys: Float32Array,
    values: Float32Array,
    kvLen: number,
    numKvHeads: number,
    headDim: number,
    mode: SwMode | { k: SwMode; v: SwMode },
  ): PackedKV;
  /**
   * Exact attention against a quantized KV cache (ATTN_PACKED_KV).
   * The v1 answer to VNPU.md §6 Q3: attention over packed KV exists
   * as an instruction, so the vNPU does not inherit the engine's
   * landmark/sparse + quantized-KV rejection for the exact path.
   * Landmark/sparse over packed KV remains open.
   */
  attentionPackedKv(
    q: Float32Array,
    kv: PackedKV,
    numHeads: number,
    opts?: NpuPackedKvAttentionOptions,
  ): Promise<NpuAttentionResult>;
  /** SOFTMAX: row softmax over an SRAM-resident tile (fp32). */
  softmax(x: Float32Array, rows: number, cols: number): Float32Array;
  /** RMSNORM over an SRAM-resident tile, with weights (fp32). */
  rmsNorm(x: Float32Array, rows: number, cols: number, weight: Float32Array, eps: number): Float32Array;
  /** ACT (SiLU): x * sigmoid(x), elementwise (fp32). */
  silu(x: Float32Array): Float32Array;
  /**
   * Build the A1/A2 block statistics for a KV cache — the stats pass
   * the landmark/sparse instructions consume. Device-resident state
   * in the device model, built once per KV cache and reused per query;
   * pass the result back via the attention `stats` option.
   */
  buildBlockStats(
    k: Float32Array,
    v: Float32Array,
    numKvHeads: number,
    headDim: number,
    blockSize?: number,
  ): { stats: BlockStats; ops: number };
  /** Extend block statistics to a longer KV cache (O(B*d), not O(n*d)). */
  extendBlockStats(
    stats: BlockStats,
    k: Float32Array,
    v: Float32Array,
    newKvLen: number,
  ): { stats: BlockStats; ops: number };
  /**
   * Heal after a donor dies: its virtual device is lost and every
   * SRAM-resident weight it held is gone with it (their keys stay
   * known, so later uses count misses and re-fetch from vRAM rather
   * than throwing). Returns the devices lost and the resident SRAM
   * bytes lost with them.
   */
  handleNodeLoss(nodeId: string): Promise<{ devicesLost: number; sramBytesLost: number }>;
  stats(): NpuStats;
}

// --- Hypervisor / the single virtual machine --------------------------------

export interface NodeProvider {
  listNodes(): WorkerInfo[];
}

export interface VirtualMachineInfo {
  vram: VramStats;
  gpu: GpuStats;
  cpu: CpuStats;
  ssd: SsdStats;
  /** Present once a vNPU pool is registered with the hypervisor. */
  npu?: NpuStats;
  nodes: Array<{ nodeId: string; device: WorkerDevice; state: WorkerState }>;
}

/** The single virtual-machine API the LLM runtime targets. */
export interface VirtualMachine {
  ram: IVirtualRam;
  gpu: IVirtualGpu;
  cpu: IVirtualCpu;
  ssd: IVirtualSsd;
  /** Present once a vNPU pool is registered with the hypervisor. */
  npu?: IVirtualNpu;
  info(): VirtualMachineInfo;
}

// ---------------------------------------------------------------------------
// Virtual server — the serving plane that runs ON the virtual machine.
//
// The server is a workload of the VM, not a host-side sidecar: it boots
// with the VM (see src/server/server.ts), reserves its memory from vRAM,
// its stream from vGPU and its task handlers from vCPU through the
// hypervisor, and serves models using only virtual hardware. Model weights
// live on vSSD and are staged through vRAM; execution runs on the mesh
// workers, which are the VM's donors. One model is ACTIVE at a time per
// cluster (workers hold one weight loader); the registry may stage
// several models, and activating another re-plans and reloads workers.
// ---------------------------------------------------------------------------

export type ModelEntryStatus = "available" | "loading" | "ready" | "error" | "unloading";

/** One model known to the server, whether or not it is loaded. */
export interface ModelEntry {
  /** Model id (HF repo id or local directory basename). */
  id: string;
  /** Where the model came from. */
  source: { kind: "local"; dir: string } | { kind: "hub"; repo: string };
  /** Local directory holding the model files (after any HF download). */
  dir: string;
  status: ModelEntryStatus;
  /** Human-readable error when status is "error". */
  error?: string;
  /** Manifest summary (present once the manifest has been read). */
  params?: number;
  weightBytes?: number;
  numLayers?: number;
  architectures?: string[];
  /** Which tokenizer the model uses: "byte-level" (fallback) or "bpe"
   *  (parsed from the model's tokenizer.json). */
  tokenizerKind?: "byte-level" | "bpe";
  /** vRAM held by this model's staged weights (0 when not staged). */
  vramBytes: number;
  /** Virtual address of the staged weights (undefined when not staged). */
  weightVaddr?: VAddr;
  loadedAt?: number;
  requestsServed: number;
  tokensGenerated: number;
}

export type ServerRequestStatus =
  | "queued"
  | "running"
  | "done"
  | "error"
  | "cancelled"
  | "rejected";

/** One served request, for the request log and /v1/server/status. */
export interface ServerRequestRecord {
  id: string;
  modelId: string;
  endpoint: "/v1/chat/completions" | "/v1/completions";
  startedAt: number;
  endedAt?: number;
  status: ServerRequestStatus;
  promptTokens: number;
  completionTokens: number;
  /** ms from submit to first generated token (streaming + non-streaming). */
  ttftMs?: number;
  error?: string;
}

/** Live server status (GET /v1/server/status, dashboard). */
export interface ServerStatus {
  state: "starting" | "ready" | "degraded" | "stopping";
  uptimeSec: number;
  activeModelId: string | null;
  models: ModelEntry[];
  requests: {
    active: number;
    queued: number;
    completed: number;
    rejected: number;
    cancelled: number;
  };
  /** Measured decode throughput across the active model (tokens/sec). */
  tokensPerSec: number;
  /** Waiting requests (scheduler waiting count). */
  queueDepth: number;
  vm: VirtualMachineInfo | null;
  /** Last few request records (newest first). */
  recentRequests: ServerRequestRecord[];
}

export interface HealthStatus {
  status: "ok" | "degraded" | "down";
  checks: {
    vmBooted: boolean;
    activeModel: boolean;
    workersHealthy: boolean;
  };
}

/** What the HTTP layer needs to serve one request. */
export interface ServingContext {
  modelId: string;
  scheduler: Scheduler;
  tokenizer: Tokenizer;
  manifest: ModelManifest;
}

/**
 * The HTTP layer's view of the server (implemented by VirtualServer in
 * src/server/server.ts). The API layer never touches the registry, the
 * hypervisor or the scheduler directly — only this interface.
 */
export interface ServingResolver {
  /** Active serving context, or null when no model is active. */
  getActive(): ServingContext | null;
  /** All models the server knows about, as /v1/models entries. */
  listModels(): ModelInfo[];
  /** Live server status for /v1/server/status and the dashboard. */
  getServerStatus(): ServerStatus;
  /** Readiness: true when a model is active and workers are healthy. */
  isReady(): boolean;
  /** Health snapshot for /healthz. */
  getHealth(): HealthStatus;
  /** Request the server to shut down gracefully (admin endpoint). */
  requestShutdown(): void;
  /**
   * Optional model management (implemented by VirtualServer; absent in
   * the legacy fixed-model form and in minimal fakes, where the API
   * answers 404). loadModel resolves a local directory, a loaded
   * model id, or an HF repo id; a model that does not fit comes back
   * as an entry with status "error" rather than throwing.
   */
  loadModel?(ref: string): Promise<ModelEntry>;
  /** Unload a staged model. Throws for unknown ids and the active model. */
  unloadModel?(id: string): Promise<void>;
  /**
   * Make a loaded ("ready") model the active one. Throws for unknown
   * ids and models that are not ready.
   */
  activateModel?(id: string): Promise<ServingContext>;
}

/**
 * Admission control + request accounting, implemented by VirtualServer.
 * The API layer calls admit() before submitting and record() when a
 * request ends; the server decides 429s, timeouts and cancellation.
 */
export interface AdmissionControl {
  /** Called before a request is submitted. ok:false rejects the request. */
  admit(modelId: string): { ok: true } | { ok: false; status: number; message: string };
  /** Called when a request finishes, errors, or is cancelled. */
  record(record: ServerRequestRecord): void;
  counters(): {
    active: number;
    queued: number;
    completed: number;
    rejected: number;
    cancelled: number;
  };
}

/** Options for booting the virtual server. */
export interface VirtualServerOptions {
  /** HTTP port for the serving API (0 = ephemeral). */
  port?: number;
  /** Max waiting + running requests before new ones are rejected (429).
   *  Default 32. */
  maxQueue?: number;
  /** Per-request deadline in ms; a request past it is cancelled.
   *  Default 300_000. */
  requestTimeoutMs?: number;
  /** Max concurrently running sequences in the active model's scheduler.
   *  Default 16. */
  maxSequences?: number;
  /** vRAM reserved for the server's own buffers (bytes). Default 256 MiB. */
  serverMemoryBytes?: number;
  /** Directory scanned for locally available models (default "./models"). */
  modelsDir?: string;
  /** Novel engine paths for loaded models (defaults to baseline). */
  engineOptions?: EngineOptions;
}
