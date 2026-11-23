/**
 * softwell-machine — scheduler: continuous batching.
 *
 * One scheduler lives on the orchestrator. Each step():
 *   1. admits waiting sequences (FIFO) while the running count is below
 *      maxSequences and the step's token budget (maxBatchTokens) fits;
 *   2. runs PREFILL for newly admitted sequences — the prompt is split
 *      into 16-row chunks (PREFILL_CHUNK_ROWS), each chunk one batched
 *      forward per pipeline stage, chaining hidden states stage 0 ->
 *      ... -> N-1, and samples the first token from the last chunk's
 *      logits (the last row of the last chunk is the prompt's last
 *      token);
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
  DecodingMode,
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
import { argmax } from "../engine/sampling.ts";
import { NGramDrafter } from "../engine/novel-decoding.ts";

/**
 * Rows per prefill chunk (runStageChain). See the chunked-prefill comment
 * there: 16 rows ≈ 4 s per stage-chain execute at ctx ~1000 on this host,
 * comfortably inside the server transport's 30 s mesh timeout.
 */
const PREFILL_CHUNK_ROWS = 16;

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
  /**
   * Token-generation mode for greedy sequences: "autoregressive" (one
   * forward per token, baseline), "speculative" (Theorem S1) or "jacobi"
   * (Theorem S2) — see src/engine/novel-decoding.ts and
   * docs/research/03-decoding.md. Default "autoregressive". Sequences
   * with temperature > 0 always use autoregressive decoding (Jacobi is
   * greedy-only by construction; speculative sampling needs the target
   * distribution, which the scheduler's Sampler does not expose).
   * On this CPU backend both novel modes are expected to LOSE wall-clock
   * (β ≈ 1); the flag exists so the loss — or a win on bandwidth-bound
   * hardware — is measured, not assumed.
   */
  decoding?: DecodingMode;
  /** Draft length K for speculative decoding / block size for Jacobi. Default 4. */
  draftLength?: number;
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
  /** Logits predicting the next token (from prefill or the last
   *  verification forward); the speculative/Jacobi rounds start here. */
  pendingLogits: Float32Array | null;
  /** Accepted token not yet fed into the model (its K/V is not in the
   *  cache); fed as row 0 of the next speculative verification forward. */
  carry: { token: number; pos: number } | null;
  /** Speculative-decoding counters (measured, per sequence). */
  specProposed: number;
  specAccepted: number;
  specRounds: number;
  /** Jacobi counters (measured, per sequence). */
  jacobiBlocks: number;
  jacobiIterations: number;
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
  private readonly decoding: DecodingMode;
  private readonly draftLength: number;
  /** Free n-gram drafter (no target forward); shared by all sequences. */
  private readonly drafter = new NGramDrafter(8);
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
    this.decoding = opts.decoding ?? "autoregressive";
    this.draftLength = opts.draftLength ?? 4;
    if (!Number.isInteger(this.draftLength) || this.draftLength < 1) {
      throw new Error(
        `ContinuousBatchingScheduler: draftLength must be a positive integer, got ${opts.draftLength}`,
      );
    }
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
    this.sequences.set(req.id, {
      state,
      request: req,
      generatedText: "",
      pendingLogits: null,
      carry: null,
      specProposed: 0,
      specAccepted: 0,
      specRounds: 0,
      jacobiBlocks: 0,
      jacobiIterations: 0,
    });
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

  /**
   * Cancel a waiting or running sequence: it stops generating, its
   * worker-side KV cache is freed (best-effort), and its state becomes
   * "done" with finishReason "cancelled". Step-boundary semantics: a
   * sequence already admitted to the step currently in flight may still
   * have its forward run, but its logits are discarded and no further
   * token is appended (see the guards in step() and appendToken()).
   * Returns false for an unknown id or one already done/error. Never
   * throws for an unknown id.
   */
  cancel(id: string): boolean {
    const record = this.sequences.get(id);
    if (record === undefined) return false;
    const status = record.state.status;
    if (status === "done" || status === "error") return false;
    const waitingIndex = this.waitingOrder.indexOf(id);
    if (waitingIndex >= 0) this.waitingOrder.splice(waitingIndex, 1);
    record.state.status = "done";
    record.state.finishReason = "cancelled";
    void this.freeSequence(id).catch(() => undefined);
    return true;
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
          // runStageChain returns one row per sequence (in toPrefill
          // order): the logits for each sequence's last prompt token.
          // (Decode does the same — one row per sequence — so the two
          // call sites read rows the same way.)
          let row = 0;
          for (const record of toPrefill) {
            // The row index advances for every record, including one
            // cancelled mid-prefill: runStageChain returns one row per
            // record, so skipping the increment would misalign the rest.
            const thisRow = row;
            row += 1;
            // A sequence cancelled while its prefill forward was in
            // flight must not be resurrected: discard its logits.
            if (record.state.status !== "prefill") continue;
            const vocab = logits.shape[1] as number;
            const rowLogits = new Float32Array(logits.data.slice(thisRow * vocab, (thisRow + 1) * vocab));
            const tokenId = this.sampler(rowLogits, record.request.sampling);
            record.state.status = "decoding";
            record.state.prefillDoneAt = Date.now();
            // Logits predicting the token after the first one: the starting
            // point for speculative/Jacobi rounds on the next step.
            record.pendingLogits = rowLogits;
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

    // -- (c) Decode: one token (autoregressive) or one speculative/Jacobi
    // round for every sequence already running before this step (newly
    // prefilled sequences sampled their first token above and decode on
    // the next step). Novel modes apply only to greedy sequences; sampled
    // sequences always decode autoregressively (see the `decoding` option).
    const toDecode = [...this.sequences.values()].filter(
      (r) => r.state.status === "decoding" && !newlyAdmitted.includes(r),
    );
    if (toDecode.length > 0) {
      const routes = this.routesFor("decode");
      info.decodeRoutes = routes;
      const novelMode = this.decoding !== "autoregressive";
      const greedy = novelMode
        ? toDecode.filter((r) => !(r.request.sampling.temperature > 0))
        : [];
      const autoregressive = novelMode ? toDecode.filter((r) => !greedy.includes(r)) : toDecode;
      try {
        if (autoregressive.length > 0) {
          const logits = await this.runStageChain(autoregressive, "decode", routes);
          const vocab = logits.shape[1] as number;
          for (let i = 0; i < autoregressive.length; i++) {
            const record = autoregressive[i] as SequenceRecord;
            const rowLogits = new Float32Array(logits.data.slice(i * vocab, (i + 1) * vocab));
            const tokenId = this.sampler(rowLogits, record.request.sampling);
            record.pendingLogits = rowLogits;
            info.decoded.push(record.state.id);
            this.appendToken(record, tokenId, produced, info);
          }
        }
        if (greedy.length > 0) {
          if (this.decoding === "speculative") {
            await this.decodeSpeculative(greedy, routes, info, produced);
          } else if (this.decoding === "jacobi") {
            await this.decodeJacobi(greedy, routes, info, produced);
          }
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
    if (this.decoding !== "autoregressive") {
      info.routing += `; decoding mode: ${this.decoding} (K=${this.draftLength})`;
    }
    return produced;
  }

  /** One row of a batched stage-chain forward: a token for one sequence
   *  at one absolute position. Rows for different sequences may share a
   *  forward; rows for one sequence must be in position order. */
  private async runRows(
    rows: Array<{ seqId: string; position: number; tokenId: number }>,
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
    if (rows.length === 0) {
      throw new Error("ContinuousBatchingScheduler.runRows: no rows");
    }
    const seqIds = rows.map((r) => r.seqId);
    const positions = rows.map((r) => r.position);
    const tokenIds = rows.map((r) => r.tokenId);

    // Stage 0 embeds tokenIds; its `input` is a placeholder (no hidden
    // states exist yet) with shape [rows, 0]. Later stages consume the
    // previous stage's output as their input.
    let input: { data: number[]; shape: number[] } = { data: [], shape: [rows.length, 0] };
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
   * Run one batched forward through every pipeline stage in order.
   * Returns one logits row per record, in record order: for decode,
   * each record's single row; for prefill, each record's last-prompt-
   * token row (prefill is chunked internally — see below — so the
   * returned rows are assembled from the chunks that held them).
   * `records` are the sequences in this batch, in row order.
   */
  private async runStageChain(
    records: SequenceRecord[],
    kind: "prefill" | "decode",
    routes: StageRoute[],
  ): Promise<{ data: number[]; shape: number[] }> {
    // Build the batched rows: prefill rows are the full prompt per sequence,
    // decode rows are each sequence's single last token.
    const rows: Array<{ seqId: string; position: number; tokenId: number }> = [];
    for (const record of records) {
      const promptLen = record.state.promptTokenIds.length;
      const generated = record.state.generatedTokenIds.length;
      if (kind === "prefill") {
        for (let i = 0; i < promptLen; i++) {
          rows.push({
            seqId: record.state.id,
            position: i,
            tokenId: record.state.promptTokenIds[i] as number,
          });
        }
      } else {
        // The last generated token sits at absolute position
        // promptLen + generated - 1; feed it to predict the next token.
        const lastToken = record.state.generatedTokenIds[generated - 1] as number;
        rows.push({
          seqId: record.state.id,
          position: promptLen + generated - 1,
          tokenId: lastToken,
        });
      }
    }
    // Chunked prefill: a long prompt must not go out as one mesh execute.
    // A 997-token prefill on this host takes ~224 s (bench-kv-before,
    // in-process supplement) while the server's scheduler transport uses
    // a MeshClient with a 30 s timeout (src/cli.ts buildFactory) — so a
    // single execute always timed out, the scheduler marked the sequence
    // "error", and the worker kept grinding the orphaned prefill for
    // minutes. Each chunk is one stage-chain execute with absolute
    // positions preserved; the engine treats chunks after the first as a
    // continued prefill (its per-row fallback path), so chunking changes
    // timing, not numerics. The returned tensor has ONE row per record
    // (in record order) — the logits for that record's last prompt
    // token, lifted out of whichever chunk contained it — never the
    // last chunk's rows, which would lose every sequence whose last
    // token fell in an earlier chunk. Sizing: ~0.22 s/row at ctx ~1000
    // on this host, so 16 rows ≈ 4 s per chunk (~7x headroom under the
    // 30 s timeout, ~2x under load noise). A 4k prompt's late rows cost
    // ~2x more per row, so 4k is near the edge of this size — a
    // longer-context follow-up should re-measure before raising it.
    // Decode rows are single tokens and never chunked.
    if (kind === "prefill") {
      // Per-record row ranges in the flattened rows array.
      const ranges: Array<{ start: number; len: number }> = [];
      let cursor = 0;
      for (const record of records) {
        const len = record.state.promptTokenIds.length;
        ranges.push({ start: cursor, len });
        cursor += len;
      }
      const lastRows: Array<number[] | undefined> = records.map(() => undefined);
      let vocab: number | null = null;
      for (let start = 0; start < rows.length; start += PREFILL_CHUNK_ROWS) {
        const chunk = rows.slice(start, start + PREFILL_CHUNK_ROWS);
        const output = await this.runRows(chunk, kind, routes);
        if (vocab === null) vocab = output.shape[1] as number;
        for (let i = 0; i < ranges.length; i++) {
          if (lastRows[i] !== undefined) continue;
          const range = ranges[i] as { start: number; len: number };
          const lastRow = range.start + range.len - 1;
          if (lastRow >= start && lastRow < start + chunk.length) {
            const rowInChunk = lastRow - start;
            lastRows[i] = output.data.slice(
              rowInChunk * (vocab as number),
              (rowInChunk + 1) * (vocab as number),
            );
          }
        }
      }
      // Every record's last row is in exactly one chunk, so all rows are
      // captured; rows is non-empty here (step() only prefills sequences
      // with a non-empty prompt), so vocab is always set.
      const data: number[] = [];
      for (const row of lastRows) data.push(...(row as number[]));
      return { data, shape: [records.length, vocab as number] };
    }
    return this.runRows(rows, kind, routes);
  }
  private remainingTokens(record: SequenceRecord): number {
    return record.request.sampling.maxTokens - record.state.generatedTokenIds.length;
  }

  /** Next absolute position for a sequence (prompt + generated so far). */
  private nextPosition(record: SequenceRecord): number {
    return record.state.promptTokenIds.length + record.state.generatedTokenIds.length;
  }

  /** Prompt + generated tokens: the drafter's context for a sequence. */
  private draftContext(record: SequenceRecord): number[] {
    return [...record.state.promptTokenIds, ...record.state.generatedTokenIds];
  }

  /**
   * One speculative round (Theorem S1) for every greedy sequence, batched
   * into a single verification forward through the stage chain. Each
   * sequence is in round A (pending logits predict the next token) or
   * round B (an accepted-but-unfed carry token is row 0). Verification is
   * greedy: a draft is accepted iff it equals the target's argmax at its
   * position. Output is token-for-token identical to autoregressive
   * decoding; the cache stays correct because a correction/bonus token is
   * fed at its position at the start of the next round, overwriting the
   * rejected draft's K/V (BlockPagedKVCache.append overwrites).
   */
  private async decodeSpeculative(
    records: SequenceRecord[],
    routes: StageRoute[],
    info: SchedulerStepInfo,
    produced: GeneratedToken[],
  ): Promise<void> {
    interface Plan {
      record: SequenceRecord;
      /** "A": pending logits known; "B": carry token is row 0. */
      kind: "A" | "B";
      drafts: number[];
      /** First verification row for this sequence in the batched output. */
      rowStart: number;
      /** Position of the first draft token. */
      draftPos: number;
    }
    const rows: Array<{ seqId: string; position: number; tokenId: number }> = [];
    const plans: Plan[] = [];
    for (const record of records) {
      const remaining = this.remainingTokens(record);
      if (remaining <= 0) continue;
      const k = Math.min(this.draftLength, remaining);
      const drafts = this.drafter.draft(this.draftContext(record), k);
      if (record.carry !== null) {
        const carry = record.carry;
        const rowStart = rows.length;
        rows.push({ seqId: record.state.id, position: carry.pos, tokenId: carry.token });
        for (let i = 0; i < drafts.length; i++) {
          rows.push({ seqId: record.state.id, position: carry.pos + 1 + i, tokenId: drafts[i] as number });
        }
        plans.push({ record, kind: "B", drafts, rowStart, draftPos: carry.pos + 1 });
      } else if (record.pendingLogits !== null) {
        const rowStart = rows.length;
        const nextPos = this.nextPosition(record);
        for (let i = 0; i < drafts.length; i++) {
          rows.push({ seqId: record.state.id, position: nextPos + i, tokenId: drafts[i] as number });
        }
        plans.push({ record, kind: "A", drafts, rowStart, draftPos: nextPos });
      } else {
        // No logits to verify against (should not happen after prefill):
        // fall back to one plain autoregressive step for this sequence.
        const lastToken = record.state.generatedTokenIds[record.state.generatedTokenIds.length - 1] as number;
        const pos = this.nextPosition(record) - 1;
        rows.push({ seqId: record.state.id, position: pos, tokenId: lastToken });
        plans.push({ record, kind: "A", drafts: [], rowStart: rows.length - 1, draftPos: pos + 1 });
      }
    }
    if (rows.length === 0) return;
    const logits = await this.runRows(rows, "decode", routes);
    const vocab = logits.shape[1] as number;
    const logitsRow = (row: number): Float32Array =>
      new Float32Array(logits.data.slice(row * vocab, (row + 1) * vocab));

    for (const plan of plans) {
      const { record } = plan;
      if (record.state.status !== "decoding") continue;
      record.specRounds += 1;
      if (plan.drafts.length === 0) {
        // Plain fallback step: sample from the single row's logits.
        const rowLogits = logitsRow(plan.rowStart);
        const tokenId = argmax(rowLogits);
        record.pendingLogits = rowLogits;
        record.carry = null;
        info.decoded.push(record.state.id);
        this.appendToken(record, tokenId, produced, info);
        continue;
      }
      // Verify left to right. Draft i is predicted by the pending logits
      // (round A, i = 0) or by verification row i-1 (round A) / row i
      // (round B, where row 0 is the carry).
      let accepted = 0;
      let correction: number | null = null;
      for (let i = 0; i < plan.drafts.length; i++) {
        const pred =
          plan.kind === "A"
            ? i === 0
              ? (record.pendingLogits as Float32Array)
              : logitsRow(plan.rowStart + i - 1)
            : logitsRow(plan.rowStart + i);
        record.specProposed += 1;
        if ((plan.drafts[i] as number) === argmax(pred)) {
          record.specAccepted += 1;
          accepted += 1;
        } else {
          correction = argmax(pred);
          break;
        }
      }
      let stopped = false;
      for (let i = 0; i < accepted; i++) {
        info.decoded.push(record.state.id);
        this.appendToken(record, plan.drafts[i] as number, produced, info);
        if (record.state.status !== "decoding") {
          stopped = true;
          break;
        }
      }
      if (stopped) continue;
      record.pendingLogits = null;
      if (correction !== null) {
        // Rejection at draft `accepted`: the correction token.
        info.decoded.push(record.state.id);
        this.appendToken(record, correction, produced, info);
        if (record.state.status === "decoding") {
          record.carry = { token: correction, pos: plan.draftPos + accepted };
        }
      } else {
        // All drafts accepted: bonus token from the last verification row.
        const bonusRow = plan.kind === "A" ? plan.rowStart + plan.drafts.length - 1 : plan.rowStart + plan.drafts.length;
        const bonus = argmax(logitsRow(bonusRow));
        if (this.remainingTokens(record) > 0) {
          info.decoded.push(record.state.id);
          this.appendToken(record, bonus, produced, info);
          if (record.state.status === "decoding") {
            record.carry = { token: bonus, pos: plan.draftPos + plan.drafts.length };
          }
        } else {
          // No room for the bonus (maxTokens reached exactly on the last
          // draft); the sequence is already done, nothing to carry.
          record.carry = null;
        }
      }
    }
  }

  /**
   * One Jacobi block (Theorem S2) for every greedy sequence, iterated to
   * the fixed point with one batched forward per iteration through the
   * stage chain, plus a commit forward for blocks that do not converge
   * within K iterations. Greedy only. Output is token-for-token identical
   * to autoregressive decoding.
   */
  private async decodeJacobi(
    records: SequenceRecord[],
    routes: StageRoute[],
    info: SchedulerStepInfo,
    produced: GeneratedToken[],
  ): Promise<void> {
    const K = this.draftLength;
    interface JState {
      record: SequenceRecord;
      blockLen: number;
      guess: number[];
      nextPos: number;
      pending: Float32Array;
      converged: boolean;
      /** Logits predicting the token after this block. */
      nextPending: Float32Array | null;
    }
    const states: JState[] = [];
    for (const record of records) {
      const remaining = this.remainingTokens(record);
      if (remaining <= 0) continue;
      if (record.pendingLogits === null) {
        // No logits to start from: one plain autoregressive step instead.
        const lastToken = record.state.generatedTokenIds[record.state.generatedTokenIds.length - 1] as number;
        const pos = this.nextPosition(record) - 1;
        const logits = await this.runRows(
          [{ seqId: record.state.id, position: pos, tokenId: lastToken }],
          "decode",
          routes,
        );
        const vocab = logits.shape[1] as number;
        const rowLogits = new Float32Array(logits.data.slice(0, vocab));
        const tokenId = argmax(rowLogits);
        record.pendingLogits = rowLogits;
        info.decoded.push(record.state.id);
        this.appendToken(record, tokenId, produced, info);
        continue;
      }
      const blockLen = Math.min(K, remaining);
      if (blockLen === 1) {
        // A 1-token block is plain autoregressive decoding.
        const tokenId = argmax(record.pendingLogits);
        info.decoded.push(record.state.id);
        this.appendToken(record, tokenId, produced, info);
        if (record.state.status === "decoding") {
          // Feed the token to get the next block's pending logits.
          const pos = this.nextPosition(record) - 1;
          const logits = await this.runRows(
            [{ seqId: record.state.id, position: pos, tokenId }],
            "decode",
            routes,
          );
          const vocab = logits.shape[1] as number;
          record.pendingLogits = new Float32Array(logits.data.slice(0, vocab));
        }
        continue;
      }
      let guess = this.drafter.draft(this.draftContext(record), blockLen);
      while (guess.length < blockLen) {
        guess.push(guess.length > 0 ? (guess[guess.length - 1] as number) : (record.state.promptTokenIds[record.state.promptTokenIds.length - 1] as number));
      }
      states.push({
        record,
        blockLen,
        guess: guess.slice(0, blockLen),
        nextPos: this.nextPosition(record),
        pending: record.pendingLogits,
        converged: false,
        nextPending: null,
      });
      record.jacobiBlocks += 1;
    }
    if (states.length === 0) return;

    const vocabOf = (logits: { data: number[]; shape: number[] }): number => logits.shape[1] as number;
    const rowOf = (logits: { data: number[]; shape: number[] }, row: number): Float32Array => {
      const vocab = vocabOf(logits);
      return new Float32Array(logits.data.slice(row * vocab, (row + 1) * vocab));
    };

    // Jacobi iterations: one batched forward per iteration over the
    // still-unconverged blocks.
    for (let iter = 0; iter < K; iter++) {
      const active = states.filter((s) => !s.converged);
      if (active.length === 0) break;
      const rows: Array<{ seqId: string; position: number; tokenId: number }> = [];
      for (const s of active) {
        for (let i = 0; i < s.blockLen; i++) {
          rows.push({ seqId: s.record.state.id, position: s.nextPos + i, tokenId: s.guess[i] as number });
        }
      }
      const logits = await this.runRows(rows, "decode", routes);
      let rowStart = 0;
      for (const s of active) {
        s.record.jacobiIterations += 1;
        const refined: number[] = [argmax(s.pending)];
        for (let i = 1; i < s.blockLen; i++) {
          refined.push(argmax(rowOf(logits, rowStart + i - 1)));
        }
        s.nextPending = rowOf(logits, rowStart + s.blockLen - 1);
        if (refined.every((t, i) => t === s.guess[i])) {
          s.converged = true;
        } else {
          s.guess = refined;
        }
        rowStart += s.blockLen;
      }
    }

    // Commit forward for blocks that did not converge within K iterations:
    // the guess is correct by Theorem S2a, but the cache still holds the
    // previous guess's K/V at some positions; feeding the correct tokens
    // repairs those slots and yields the next block's pending logits.
    const needCommit = states.filter((s) => !s.converged);
    if (needCommit.length > 0) {
      const rows: Array<{ seqId: string; position: number; tokenId: number }> = [];
      for (const s of needCommit) {
        for (let i = 0; i < s.blockLen; i++) {
          rows.push({ seqId: s.record.state.id, position: s.nextPos + i, tokenId: s.guess[i] as number });
        }
      }
      const logits = await this.runRows(rows, "decode", routes);
      let rowStart = 0;
      for (const s of needCommit) {
        s.nextPending = rowOf(logits, rowStart + s.blockLen - 1);
        s.converged = true;
        rowStart += s.blockLen;
      }
    }

    // Emit the blocks in order; EOS (or maxTokens) stops a block early.
    for (const s of states) {
      if (s.record.state.status !== "decoding") continue;
      let stopped = false;
      for (let i = 0; i < s.blockLen; i++) {
        info.decoded.push(s.record.state.id);
        this.appendToken(s.record, s.guess[i] as number, produced, info);
        if (s.record.state.status !== "decoding") {
          stopped = true;
          break;
        }
      }
      if (!stopped && s.nextPending !== null) {
        s.record.pendingLogits = s.nextPending;
      } else {
        s.record.pendingLogits = null;
      }
      s.record.carry = null;
    }
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
    // A sequence cancelled (or errored) while its forward was in flight
    // must not gain another token: its logits are discarded.
    if (record.state.status === "done" || record.state.status === "error") return;
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
