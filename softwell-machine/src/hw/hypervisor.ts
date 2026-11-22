/**
 * Hypervisor — the only component that knows donors exist AND pools exist.
 *
 * It wires the four hardware pools (vRAM, vGPU, vCPU, vSSD) together with a
 * NodeProvider and presents ONE VirtualMachine. Nothing above this module
 * ever sees a donor, a node id, or a pool coordinator: the VirtualMachine
 * object returned by boot() is the entire hardware platform for the LLM
 * runtime (see src/contracts.ts, "Virtual hardware").
 *
 * Pools are injected through HypervisorOpts and spoken to only through the
 * IVirtual* interfaces from contracts.ts. This module never imports a
 * sibling hw module.
 */
import { VRAM_PAGE_SIZE_BYTES } from "../contracts.ts";
import type {
  IVirtualCpu,
  IVirtualGpu,
  IVirtualRam,
  IVirtualSsd,
  NodeProvider,
  VirtualMachine,
  VirtualMachineInfo,
  VramAllocation,
  WorkerInfo,
} from "../contracts.ts";

export interface HypervisorOpts {
  nodes: NodeProvider;
  ram: IVirtualRam;
  gpu: IVirtualGpu;
  cpu: IVirtualCpu;
  ssd: IVirtualSsd;
  /** How often sweep() runs on its own timer. Default 5000 ms. */
  sweepIntervalMs?: number;
  /** How long boot() waits for a healthy node before giving up. Default 10000 ms. */
  bootTimeoutMs?: number;
}

/**
 * Summary of healing after one donor disappears. `errors` collects one
 * entry per pool whose handleNodeLoss threw (format "<pool>: <message>");
 * a throwing pool never prevents the other pools from healing, and its
 * counters stay at zero.
 */
export interface NodeLossSummary {
  ram: { pagesRecovered: number; pagesLost: number };
  gpu: { streamsReassigned: number; devicesLost: number };
  cpu: { tasksRequeued: number; tasksLost: number };
  ssd: { chunksRecovered: number; chunksLost: number };
  errors: string[];
}

const POLL_INTERVAL_MS = 500;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export class Hypervisor {
  booted = false;

  private opts: HypervisorOpts;
  private sweepTimer: ReturnType<typeof setInterval> | null = null;
  private lastSeen = new Set<string>();
  private vm: VirtualMachine | null = null;

  constructor(opts: HypervisorOpts) {
    this.opts = opts;
  }

  /**
   * Wait for at least one healthy node, verify every pool reports stats,
   * then return the single VirtualMachine and start the sweep timer.
   * Throws if no healthy node appears within bootTimeoutMs (default 10 s),
   * or if any pool's stats() throws.
   */
  async boot(): Promise<VirtualMachine> {
    if (this.booted) {
      throw new Error("Hypervisor already booted");
    }
    const timeoutMs = this.opts.bootTimeoutMs ?? 10000;
    const deadline = Date.now() + timeoutMs;
    let nodes: WorkerInfo[] = [];
    for (;;) {
      nodes = this.opts.nodes.listNodes();
      if (nodes.some((n) => n.state === "healthy")) {
        break;
      }
      if (Date.now() >= deadline) {
        throw new Error(
          `Hypervisor boot timed out after ${timeoutMs} ms: ` +
            `NodeProvider reported no healthy nodes`,
        );
      }
      await sleep(Math.min(POLL_INTERVAL_MS, Math.max(0, deadline - Date.now())));
    }

    // Verify each pool reports sane stats; a throwing pool fails the boot
    // with that pool's error.
    this.opts.ram.stats();
    this.opts.gpu.stats();
    this.opts.cpu.stats();
    this.opts.ssd.stats();

    this.lastSeen = new Set(nodes.map((n) => n.workerId));
    this.vm = {
      ram: this.opts.ram,
      gpu: this.opts.gpu,
      cpu: this.opts.cpu,
      ssd: this.opts.ssd,
      info: () => this.getVmInfo(),
    };
    this.booted = true;

    const sweepIntervalMs = this.opts.sweepIntervalMs ?? 5000;
    this.sweepTimer = setInterval(() => {
      void this.sweep().catch(() => {
        // The timer must never surface an unhandled rejection; a failed
        // sweep is retried on the next tick. Direct sweep() calls still
        // surface their errors to the caller.
      });
    }, sweepIntervalMs);
    // Don't keep the process alive just for the sweep timer.
    this.sweepTimer.unref();

    return this.vm;
  }

  /** Aggregate the four pools' stats plus the current node list. */
  getVmInfo(): VirtualMachineInfo {
    this.assertBooted();
    const nodes = this.opts.nodes.listNodes();
    return {
      vram: this.opts.ram.stats(),
      gpu: this.opts.gpu.stats(),
      cpu: this.opts.cpu.stats(),
      ssd: this.opts.ssd.stats(),
      nodes: nodes.map((n) => ({
        nodeId: n.workerId,
        device: n.device,
        state: n.state,
      })),
    };
  }

  /**
   * Compare the current node list against the last-seen set. Any node that
   * disappeared, or whose state is now 'dead', is healed via
   * handleNodeLoss(). A node is healed at most once per disappearance: once
   * handled it leaves the last-seen set, so a node that stays listed as
   * 'dead' is not re-healed on every sweep.
   */
  async sweep(): Promise<void> {
    this.assertBooted();
    const current = this.opts.nodes.listNodes();
    const currentIds = new Set(current.map((n) => n.workerId));
    const lost: string[] = [];
    for (const id of this.lastSeen) {
      if (!currentIds.has(id)) {
        lost.push(id);
      }
    }
    for (const n of current) {
      if (n.state === "dead" && this.lastSeen.has(n.workerId)) {
        lost.push(n.workerId);
      }
    }
    // Survivors (anything still listed and not dead) stay in the last-seen
    // set; lost and dead nodes leave it.
    this.lastSeen = new Set(
      current.filter((n) => n.state !== "dead").map((n) => n.workerId),
    );
    for (const nodeId of lost) {
      await this.handleNodeLoss(nodeId);
    }
  }

  /**
   * Fan out healing to all four pools and collect the results. A pool that
   * throws does not prevent the others from healing; its error is recorded
   * in the returned summary's `errors` array and its counters stay zero.
   */
  async handleNodeLoss(nodeId: string): Promise<NodeLossSummary> {
    this.assertBooted();
    const summary: NodeLossSummary = {
      ram: { pagesRecovered: 0, pagesLost: 0 },
      gpu: { streamsReassigned: 0, devicesLost: 0 },
      cpu: { tasksRequeued: 0, tasksLost: 0 },
      ssd: { chunksRecovered: 0, chunksLost: 0 },
      errors: [],
    };
    try {
      summary.ram = await this.opts.ram.handleNodeLoss(nodeId);
    } catch (err) {
      summary.errors.push(`ram: ${errorMessage(err)}`);
    }
    try {
      summary.gpu = await this.opts.gpu.handleNodeLoss(nodeId);
    } catch (err) {
      summary.errors.push(`gpu: ${errorMessage(err)}`);
    }
    try {
      summary.cpu = await this.opts.cpu.handleNodeLoss(nodeId);
    } catch (err) {
      summary.errors.push(`cpu: ${errorMessage(err)}`);
    }
    try {
      summary.ssd = await this.opts.ssd.handleNodeLoss(nodeId);
    } catch (err) {
      summary.errors.push(`ssd: ${errorMessage(err)}`);
    }
    return summary;
  }

  /**
   * The streaming load path: look the object's size up in vSSD, allocate
   * that many bytes in vRAM, then stream it in VRAM_PAGE_SIZE_BYTES pieces
   * (ssd.getRange + ram.write) and return the allocation.
   */
  async stageToVram(key: string): Promise<VramAllocation> {
    this.assertBooted();
    const metas = await this.opts.ssd.list();
    const meta = metas.find((m) => m.key === key);
    if (!meta) {
      throw new Error(`stageToVram: object not found in vSSD: ${key}`);
    }
    const alloc = await this.opts.ram.alloc(meta.sizeBytes);
    let offset = 0;
    while (offset < meta.sizeBytes) {
      const length = Math.min(VRAM_PAGE_SIZE_BYTES, meta.sizeBytes - offset);
      const chunk = await this.opts.ssd.getRange(key, offset, length);
      await this.opts.ram.write(alloc.vaddr + offset, chunk);
      offset += length;
    }
    return alloc;
  }

  /** Stop the sweep timer. The VirtualMachine must not be used afterwards. */
  async shutdown(): Promise<void> {
    this.assertBooted();
    if (this.sweepTimer !== null) {
      clearInterval(this.sweepTimer);
      this.sweepTimer = null;
    }
    this.booted = false;
    this.vm = null;
  }

  private assertBooted(): void {
    if (!this.booted) {
      throw new Error("Hypervisor is not booted: call boot() first");
    }
  }
}
