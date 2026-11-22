export type Phase = "hbm" | "tmem" | "mma" | "epilogue";

export type DieCell = { die: 0 | 1; sm: number };

export type DieEvent = {
  phase: Phase;
  cells: DieCell[];
  caption: string;
  cycles: number;
  /** Clock after this phase. */
  cycle: number;
  bytes: number;
  addr: number | null;
};

export const PHASES: { id: Phase; label: string }[] = [
  { id: "hbm", label: "LOAD" },
  { id: "tmem", label: "TMEM" },
  { id: "mma", label: "MMA" },
  { id: "epilogue", label: "STORE" },
];

/** Logical SMs. A B300 ships 160. Nothing here is fused off — there is no die to bin. */
export const SMS_PER_DIE = 128;

/** Placement alternates the two software panes. */
export function place(count: number): DieCell[] {
  const n = Math.max(0, Math.min(count, SMS_PER_DIE * 2));
  const cells: DieCell[] = [];
  for (let i = 0; i < n; i++) {
    cells.push({ die: (i % 2) as 0 | 1, sm: Math.floor(i / 2) % SMS_PER_DIE });
  }
  return cells;
}

const PHASE_ORDER: Phase[] = ["hbm", "tmem", "mma", "epilogue"];

export function spanEvents(opts: {
  cells: number;
  from: number;
  spans: Record<Phase, { cycles: number; bytes: number; addr: number | null }>;
  captions: Record<Phase, string>;
}): { events: DieEvent[]; cycles: number } {
  const cells = place(opts.cells);
  let clock = opts.from;
  let total = 0;
  const events: DieEvent[] = [];
  for (const phase of PHASE_ORDER) {
    const span = opts.spans[phase];
    if (span.cycles <= 0 && span.bytes <= 0) continue;
    clock += span.cycles;
    total += span.cycles;
    events.push({
      phase,
      cells,
      caption: opts.captions[phase],
      cycles: span.cycles,
      cycle: clock,
      bytes: span.bytes,
      addr: span.addr,
    });
  }
  return { events, cycles: total };
}

export function launch(opts: {
  cells: number;
  captions: Record<Phase, string>;
  kblocks: number;
  macs: number;
}): { events: DieEvent[]; cycles: number } {
  const k = Math.max(1, opts.kblocks);
  const costs: Record<Phase, number> = {
    hbm: 80 * k * Math.max(1, Math.ceil(opts.cells / 16)),
    tmem: 24 * k * Math.max(1, Math.ceil(opts.cells / 8)),
    mma: Math.max(4, Math.round(opts.macs / 8)),
    epilogue: 12 * Math.max(1, opts.cells),
  };
  return spanEvents({
    cells: opts.cells,
    from: 0,
    spans: {
      hbm: { cycles: costs.hbm, bytes: 0, addr: null },
      tmem: { cycles: costs.tmem, bytes: 0, addr: null },
      mma: { cycles: costs.mma, bytes: 0, addr: null },
      epilogue: { cycles: costs.epilogue, bytes: 0, addr: null },
    },
    captions: opts.captions,
  });
}
