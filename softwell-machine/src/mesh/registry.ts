import {
  HEARTBEAT_DEAD_MS,
  HEARTBEAT_INTERVAL_MS,
  HEARTBEAT_TIMEOUT_MS,
} from "../contracts.ts";
import type {
  AssignmentResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  RegisterResponse,
  ShardAssignment,
  ShardPlan,
  WorkerCapability,
  WorkerInfo,
} from "../contracts.ts";

/**
 * WorkerRegistry — pure in-memory registry for the worker mesh.
 *
 * No HTTP here: the API/orchestrator server calls this class and the
 * protocol layer (./protocol.ts) only translates HTTP <-> method calls.
 *
 * Lifecycle: register() -> state "joining"; the first heartbeat flips the
 * worker to "healthy". sweep() marks workers "suspect" after
 * HEARTBEAT_TIMEOUT_MS without a heartbeat and "dead" after
 * HEARTBEAT_DEAD_MS. A heartbeat from a suspect/dead worker revives it to
 * "healthy" (it is alive again; the planner may still re-plan around it).
 */
export class WorkerRegistry {
  private workers = new Map<string, WorkerInfo>();
  private assignments = new Map<string, ShardAssignment>();
  private assignmentVersion = 0;
  /** Last computed tokens/sec per worker, from heartbeat deltas. */
  private throughput = new Map<string, number>();

  register(capability: WorkerCapability): RegisterResponse {
    const workerId =
      `worker-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    const now = Date.now();
    const info: WorkerInfo = {
      ...capability,
      workerId,
      state: "joining",
      registeredAt: now,
      lastHeartbeatAt: now,
      usedMemoryBytes: 0,
      activeSequences: 0,
    };
    this.workers.set(workerId, info);
    this.throughput.set(workerId, 0);
    return { workerId, heartbeatIntervalMs: HEARTBEAT_INTERVAL_MS };
  }

  heartbeat(req: HeartbeatRequest): HeartbeatResponse {
    const worker = this.workers.get(req.workerId);
    if (worker === undefined) {
      return { ok: false, assignmentVersion: this.assignmentVersion };
    }
    const now = Date.now();
    const elapsedMs = now - worker.lastHeartbeatAt;
    // Tokens/sec from this heartbeat interval. Only update on intervals
    // that actually produced tokens; an idle worker keeps its last rate
    // instead of reporting 0 for doing nothing.
    if (elapsedMs > 0 && req.tokensSinceHeartbeat > 0) {
      this.throughput.set(
        req.workerId,
        (req.tokensSinceHeartbeat * 1000) / elapsedMs,
      );
    }
    worker.lastHeartbeatAt = now;
    worker.usedMemoryBytes = req.usedMemoryBytes;
    worker.activeSequences = req.activeSequences;
    if (
      worker.state === "joining" ||
      worker.state === "suspect" ||
      worker.state === "dead"
    ) {
      worker.state = "healthy";
    }
    return { ok: true, assignmentVersion: this.assignmentVersion };
  }

  getWorker(id: string): WorkerInfo | undefined {
    const worker = this.workers.get(id);
    return worker === undefined ? undefined : { ...worker };
  }

  listWorkers(): WorkerInfo[] {
    return [...this.workers.values()].map((worker) => ({ ...worker }));
  }

  /** Install a new shard plan: bumps the version and maps workerId -> assignment. */
  setAssignment(plan: ShardPlan): void {
    this.assignmentVersion += 1;
    this.assignments.clear();
    for (const assignment of plan.assignments) {
      this.assignments.set(assignment.workerId, assignment);
    }
  }

  currentAssignmentVersion(): number {
    return this.assignmentVersion;
  }

  getAssignment(workerId: string): AssignmentResponse {
    return {
      version: this.assignmentVersion,
      assignment: this.assignments.get(workerId) ?? null,
    };
  }

  /**
   * Fault detection. `now` defaults to Date.now() and is injectable so
   * tests can fake time. Returns the workers whose state changed.
   */
  sweep(now: number = Date.now()): WorkerInfo[] {
    const changed: WorkerInfo[] = [];
    for (const worker of this.workers.values()) {
      if (worker.state === "dead") continue;
      const silentMs = now - worker.lastHeartbeatAt;
      if (silentMs > HEARTBEAT_DEAD_MS) {
        worker.state = "dead";
        changed.push({ ...worker });
      } else if (silentMs > HEARTBEAT_TIMEOUT_MS && worker.state !== "suspect") {
        worker.state = "suspect";
        changed.push({ ...worker });
      }
    }
    return changed;
  }

  /** Remove a worker (and its assignment, if any). Returns true if it existed. */
  removeWorker(id: string): boolean {
    this.assignments.delete(id);
    this.throughput.delete(id);
    return this.workers.delete(id);
  }

  /** Last measured tokens/sec for a worker, 0 when unknown / never produced tokens. */
  workerThroughput(workerId: string): number {
    return this.throughput.get(workerId) ?? 0;
  }
}
