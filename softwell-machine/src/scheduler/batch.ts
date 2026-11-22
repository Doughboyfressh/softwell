/**
 * softwell-machine — scheduler: continuous batching.
 *
 * One scheduler lives on the orchestrator. Each step():
 *   1. admits waiting sequences (FIFO) while the running count is below
 *      maxSequences and the step's token budget (maxBatchTokens) fits;
 *   2. runs PREFILL for newly admitted sequences — one batched forward per
 *      pipeline stage, chaining hidden states stage 0 -> ... -> N-1, and
 *      samples the first token from the last stage's logits;
 *   3. runs one DECODE token for every already-running sequence through the
 *      same stage chain, and samples the next token.
 *
 * Prefill/decode disaggregation: when the plan has more than one worker per
 * stage, prefill executes prefer a worker whose roles include "prefill"
 * and decode executes prefer one whose roles include "decode"; "mixed"
 * workers serve either. In v1 there is exactly one worker per stage, so
 * the assigned worker is used for both — the routing decision for each
 * step is recorded in `lastStepInfo()` (see `SchedulerStepInfo.routing`).
 *
 * KV cache lives on the workers, so freeing a finished sequence's cache is
 * the worker's job. The scheduler calls the injected `freeSequence`
 * callback when a sequence finishes (the CLI wires it to a worker call);
 * it defaults to a no-op. Failures there are best-effort and never fail
 * the step.
 *
 * Worker execute failures mark the affected sequences "error" with the
 * message; the scheduler itself never throws out of step().
 */

import type {
  ExecuteRequest,
  ExecuteResponse,
  GenerateRequest,
  GeneratedToken,
  Sampler,
  Scheduler,
  SequenceState,
  ShardAssignment,
  ShardPlan,
  WorkerRole,
} from "../contracts.ts";

/** Transport injected by the coordinator: one execute call to one worker. */
export type SchedulerTransport = (
  workerId: string,
  req: Omit<ExecuteRequest, "workerId">,
) => Promise<ExecuteResponse>;

export interface ContinuousBatchingSchedulerOptions {
  plan: ShardPlan;
  transport: SchedulerTransport;
  sampler: Sampler;
  /** Decode token ids to text (the tokenizer's decode). */
  decode: (tokenIds: number[]) => string;
  /** Token id that ends a sequence. */
  eosTokenId: number;
  /** Max tokens (prefill + decode rows) in one step. Default 4096. */
  maxBatchTokens?: number;
  /** Max concurrently running sequences. Default 16. */
  maxSequences?: number;
  /**
   * Called when a sequence finishes (done or error) so the caller can free
   * its worker-side KV cache. Defaults to a no-op.
   */
  freeSequence?: (seqId: string) => Promise<void>;
  /**
   * Worker roles by worker id, when the coordinator knows them (from the
   * mesh registry). Used to prefer dedicated prefill/decode workers when a
   * stage has more than one candidate worker. Omit in v1 (one worker per
   * stage); every worker is then treated as "mixed".
   */
  rolesByWorker?: Record<string, WorkerRole[]>;
}

export interface StageRoute {
  stage: number;
  workerId: string;
  /** Role this worker was chosen for: "prefill" | "decode" | "mixed" (fallback). */
  role: WorkerRole;
  /** Why this worker was picked, e.g. "dedicated prefill worker" / "only worker for stage". */
  reason: string;
}

/** Diagnostic info about the most recent step(); see `lastStepInfo()`. */
export interface SchedulerStepInfo {
  prefilled: string[];
  decoded: string[];
  finished: string[];
  errored: string[];
  tokensProduced: number;
  prefillRoutes: StageRoute[];
  decodeRoutes: StageRoute[];
  /** Human-readable summary of the routing decisions this step. */
  routing: string;
}

interface SequenceRecord {
  state: SequenceState;
  request: GenerateRequest;
  /** Decoded text of generated tokens so far (for stop-string checks). */
  generatedText: string;
}

const DEFAULT_MAX_BATCH_TOKENS = 4096;
const DEFAULT_MAX_SEQUENCES = 16;

export class ContinuousBatchingScheduler implements Scheduler {
  private plan: ShardPlan;
  private readonly transport: SchedulerTransport;
  private readonly sampler: Sampler;
  private readonly decodeText: (tokenIds: number[]) => string;
  private readonly eosTokenId: number;
  private readonly maxBatchTokens: number;
  private readonly maxSequences: number;
  private readonly freeSequence: (seqId: string) => Promise<void>;
  private readonly rolesByWorker: Record<string, WorkerRole[]>;
  private readonly sequences: Map<string, SequenceRecord>;
  private readonly waitingOrder: string[];
  private stepInfo: SchedulerStepInfo | null;

  constructor(opts: ContinuousBatchingSchedulerOptions) {
    this.plan = opts.plan;
    this.transport = opts.transport;
    this.sampler = opts.sampler;
    this.decodeText = opts.decode;
    this.eosTokenId = opts.eosTokenId;
    this.maxBatchTokens = opts.maxBatchTokens ?? DEFAULT_MAX_BATCH_TOKENS;
    this.maxSequences = opts.maxSequences ?? DEFAULT_MAX_SEQUENCES;
    this.freeSequence = opts.freeSequence ?? (async () => undefined);
    this.rolesByWorker = opts.rolesByWorker ?? {};
    this.sequences = new Map();
    this.waitingOrder = [];
    this.stepInfo = null;
  }

  /**
   * Swap in a new shard plan (e.g. after the orchestrator re-shards over
   * surviving workers). Running sequences keep their generated tokens; the
   * next step routes stages through the new assignments. Sequences whose
   * KV cache lived on a lost worker will fail on their next step and are
   * marked 'error' — the caller must resubmit them (re-prefill).
   */
  updatePlan(plan: ShardPlan): void {
    this.plan = plan;
  }

  submit(req: GenerateRequest): string {
    if (this.sequences.has(req.id)) {
      throw new Error(`ContinuousBatchingScheduler.submit: duplicate sequence id ${req.id}`);
    }
    const state: SequenceState = {
      id: req.id,
      promptTokenIds: [...req.tokenIds],
      generatedTokenIds: [],
      status: "waiting",
      createdAt: Date.now(),
      finishReason: null,
    };
    this.sequences.set(req.id, { state, request: req, generatedText: "" });
    this.waitingOrder.push(req.id);
    return req.id;
  }

  getSequence(id: string): SequenceState | undefined {
    const record = this.sequences.get(id);
    if (record === undefined) return undefined;
    return {
      ...record.state,
      promptTokenIds: [...record.state.promptTokenIds],
      generatedTokenIds: [...record.state.generatedTokenIds],
    };
  }

  pendingCount(): number {
    let count = 0;
    for (const record of this.sequences.values()) {
      if (record.state.status === "waiting") count++;
    }
    return count;
  }

  runningCount(): number {
    let count = 0;
    for (const record of this.sequences.values()) {
      if (record.state.status === "prefill" || record.state.status === "decoding") count++;
    }
    return count;
  }

  /** Diagnostic info about the most recent step(), or null before any step. */
  lastStepInfo(): SchedulerStepInfo | null {
    return this.stepInfo === null ? null : { ...this.stepInfo };
  }

  async step(): Promise<GeneratedToken[]> {
    const info: SchedulerStepInfo = {
      prefilled: [],
      decoded: [],
      finished: [],
      errored: [],
      tokensProduced: 0,
      prefillRoutes: [],
      decodeRoutes: [],
      routing: "",
    };
    this.stepInfo = info;
    const produced: GeneratedToken[] = [];

    // -- (a) Admission: FIFO, bounded by maxSequences and the token budget.
    // A newly admitted sequence costs its whole prompt this step; each
    // already-running sequence costs one decode row.
    const running = [...this.sequences.values()].filter(
      (r) => r.state.status === "prefill" || r.state.status === "decoding",
    );
    let budgetUsed = running.length; // one decode token per running sequence
    const newlyAdmitted: SequenceRecord[] = [];
    while (this.waitingOrder.length > 0) {
      if (running.length + newlyAdmitted.length >= this.maxSequences) break;
      const nextId = this.waitingOrder[0] as string;
      const record = this.sequences.get(nextId);
      if (record === undefined) {
        this.waitingOrder.shift();
        continue;
      }
      if (budgetUsed + record.state.promptTokenIds.length > this.maxBatchTokens) break;
      this.waitingOrder.shift();
      budgetUsed += record.state.promptTokenIds.length;
      record.state.status = "prefill";
      newlyAdmitted.push(record);
    }

    // -- (b) Prefill: one batched forward per stage for all newly admitted.
    if (newlyAdmitted.length > 0) {
      const emptyPrompt = newlyAdmitted.filter((r) => r.state.promptTokenIds.length === 0);
      const toPrefill = newlyAdmitted.filter((r) => r.state.promptTokenIds.length > 0);
      for (const record of emptyPrompt) {
        this.failSequence(record, "empty prompt: no tokens to prefill", info);
      }
      if (toPrefill.length > 0) {
        const routes = this.routesFor("prefill");
        info.prefillRoutes = routes;
        try {
          const logits = await this.runStageChain(toPrefill, "prefill", routes);
          // Logits rows are in the same order as the batched rows; each
          // sequence's last prompt token is its sampling position.
          let row = 0;
          for (const record of toPrefill) {
            const promptLen = record.state.promptTokenIds.length;
            const lastRow = row + promptLen - 1;
            const vocab = logits.shape[1] as number;
            const rowLogits = new Float32Array(logits.data.slice(lastRow * vocab, (lastRow + 1) * vocab));
            const tokenId = this.sampler(rowLogits, record.request.sampling);
            row += promptLen;
            record.state.status = "decoding";
            record.state.prefillDoneAt = Date.now();
            info.prefilled.push(record.state.id);
            this.appendToken(record, tokenId, produced, info);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          for (const record of toPrefill) {
            if (record.state.status === "prefill") {
              this.failSequence(record, message, info);
            }
          }
        }
      }
    }

    // -- (c) Decode: one token for every sequence already running before
    // this step (newly prefilled sequences sampled their first token above
    // and decode on the next step).
    const toDecode = [...this.sequences.values()].filter(
      (r) => r.state.status === "decoding" && !newlyAdmitted.includes(r),
    );
    if (toDecode.length > 0) {
      const routes = this.routesFor("decode");
      info.decodeRoutes = routes;
      try {
        const logits = await this.runStageChain(toDecode, "decode", routes);
        const vocab = logits.shape[1] as number;
        for (let i = 0; i < toDecode.length; i++) {
          const record = toDecode[i] as SequenceRecord;
          const rowLogits = new Float32Array(logits.data.slice(i * vocab, (i + 1) * vocab));
          const tokenId = this.sampler(rowLogits, record.request.sampling);
          info.decoded.push(record.state.id);
          this.appendToken(record, tokenId, produced, info);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        for (const record of toDecode) {
          if (record.state.status === "decoding") {
            this.failSequence(record, message, info);
          }
        }
      }
    }

    info.tokensProduced = produced.length;
    info.routing = this.describeRouting(info);
    return produced;
  }

  /**
   * Run one batched forward through every pipeline stage in order.
   * Returns the last stage's output (logits). `records` are the sequences
   * in this batch, in row order.
   */
  private async runStageChain(
    records: SequenceRecord[],
    kind: "prefill" | "decode",
    routes: StageRoute[],
  ): Promise<{ data: number[]; shape: number[] }> {
    const stageCount = this.plan.pipelineStages;
    if (stageCount <= 0) {
      throw new Error(`ContinuousBatchingScheduler: plan has no pipeline stages`);
    }
    if (routes.length !== stageCount) {
      throw new Error(
        `ContinuousBatchingScheduler: plan has ${stageCount} stages but ${routes.length} routes`,
      );
    }

    // Build the batched rows: prefill rows are the full prompt per sequence,
    // decode rows are each sequence's single last token.
    const seqIds: string[] = [];
    const positions: number[] = [];
    const tokenIds: number[] = [];
    for (const record of records) {
      const promptLen = record.state.promptTokenIds.length;
      const generated = record.state.generatedTokenIds.length;
      if (kind === "prefill") {
        for (let i = 0; i < promptLen; i++) {
          seqIds.push(record.state.id);
          positions.push(i);
          tokenIds.push(record.state.promptTokenIds[i] as number);
        }
      } else {
        // The last generated token sits at absolute position
        // promptLen + generated - 1; feed it to predict the next token.
        const lastToken = record.state.generatedTokenIds[generated - 1] as number;
        seqIds.push(record.state.id);
        positions.push(promptLen + generated - 1);
        tokenIds.push(lastToken);
      }
    }

    // Stage 0 embeds tokenIds; its `input` is a placeholder (no hidden
    // states exist yet) with shape [rows, 0]. Later stages consume the
    // previous stage's output as their input.
    let input: { data: number[]; shape: number[] } = { data: [], shape: [seqIds.length, 0] };
    let output: { data: number[]; shape: number[] } = input;
    for (let stage = 0; stage < stageCount; stage++) {
      const route = routes[stage] as StageRoute;
      const req: Omit<ExecuteRequest, "workerId"> = {
        seqIds: [...seqIds],
        kind,
        stage,
        input,
        positions: [...positions],
        // tokenIds are only meaningful to the stage with the embedding
        // (stage 0); later stages ignore them.
        ...(stage === 0 ? { tokenIds: [...tokenIds] } : {}),
      };
      const res = await this.transport(route.workerId, req);
      output = res.output;
      input = output;
    }
    if (output.shape.length !== 2) {
      throw new Error(
        `ContinuousBatchingScheduler: last stage returned unexpected output shape ${JSON.stringify(output.shape)}`,
      );
    }
    return output;
  }

  /**
   * Pick one worker per stage for `kind`. With several assignments for a
   * stage (tensor-parallel / replicated stages), prefer a worker whose
   * roles include the requested kind, else a "mixed" worker, else the first
   * assignment. In v1 there is one assignment per stage, so the assigned
   * worker is always used.
   */
  private routesFor(kind: "prefill" | "decode"): StageRoute[] {
    const routes: StageRoute[] = [];
    for (let stage = 0; stage < this.plan.pipelineStages; stage++) {
      const candidates = this.plan.assignments.filter((a) => a.stage === stage);
      if (candidates.length === 0) {
        throw new Error(`ContinuousBatchingScheduler: no worker assigned to pipeline stage ${stage}`);
      }
      const ranked = [...candidates].sort((a, b) => this.roleScore(a, kind) - this.roleScore(b, kind));
      const chosen = ranked[0] as ShardAssignment;
      const roles = this.rolesByWorker[chosen.workerId] ?? ["mixed"];
      const dedicated = roles.includes(kind);
      routes.push({
        stage,
        workerId: chosen.workerId,
        role: dedicated ? kind : "mixed",
        reason:
          candidates.length === 1
            ? "only worker for stage"
            : dedicated
              ? `dedicated ${kind} worker`
              : "mixed worker (no dedicated worker for stage)",
      });
    }
    return routes;
  }

  /** Lower is better: dedicated role first, then mixed, then anything else. */
  private roleScore(assignment: ShardAssignment, kind: "prefill" | "decode"): number {
    const roles = this.rolesByWorker[assignment.workerId];
    if (roles === undefined) return 1; // unknown roles: treat as mixed
    if (roles.includes(kind)) return 0;
    if (roles.includes("mixed")) return 1;
    return 2;
  }

  private describeRouting(info: SchedulerStepInfo): string {
    const parts: string[] = [];
    const summarize = (routes: StageRoute[], kind: string): string => {
      if (routes.length === 0) return "";
      const dedicated = routes.filter((r) => r.role === kind).length;
      return (
        `${kind}: ${routes.length} stage(s) via ` +
        routes.map((r) => `${r.workerId} (stage ${r.stage}, ${r.reason})`).join(", ") +
        (dedicated > 0 ? ` — ${dedicated} dedicated ${kind} worker(s)` : " — no dedicated workers, using assigned/mixed workers")
      );
    };
    const prefill = summarize(info.prefillRoutes, "prefill");
    const decode = summarize(info.decodeRoutes, "decode");
    if (prefill !== "") parts.push(prefill);
    if (decode !== "") parts.push(decode);
    if (parts.length === 0) parts.push("no forwards this step (nothing admitted or running)");
    return parts.join("; ");
  }

  /** Append a sampled token, emit its GeneratedToken, finish if EOS/maxTokens/stop. */
  private appendToken(
    record: SequenceRecord,
    tokenId: number,
    produced: GeneratedToken[],
    info: SchedulerStepInfo,
  ): void {
    const state = record.state;
    state.generatedTokenIds.push(tokenId);
    const text = this.decodeText([tokenId]);
    record.generatedText += text;
    let finishReason: string | null = null;
    if (tokenId === this.eosTokenId) {
      finishReason = "eos";
    } else if (state.generatedTokenIds.length >= record.request.sampling.maxTokens) {
      finishReason = "length";
    } else {
      const stop = record.request.sampling.stop ?? [];
      if (stop.some((s) => s !== "" && record.generatedText.endsWith(s))) {
        finishReason = "stop";
      }
    }
    produced.push({ seqId: state.id, tokenId, text, finishReason });
    if (finishReason !== null) {
      state.status = "done";
      state.finishReason = finishReason;
      info.finished.push(state.id);
      // Best-effort: free the worker-side KV cache; never fail the step.
      void this.freeSequence(state.id).catch(() => undefined);
    }
  }

  /** Mark a sequence errored and best-effort free its worker-side cache. */
  private failSequence(record: SequenceRecord, message: string, info: SchedulerStepInfo): void {
    record.state.status = "error";
    record.state.error = message;
    info.errored.push(record.state.id);
    void this.freeSequence(record.state.id).catch(() => undefined);
  }
}
