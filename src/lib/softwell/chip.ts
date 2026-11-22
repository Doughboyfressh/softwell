import { PI_BLOCK, dotNvfp4, gemmNvfp4, type GemmResult } from "./nvfp4";
import { RAM_BYTES } from "./board";
import { SMS_PER_DIE, type Phase } from "./silicon";

/** Bytes moved in one cycle. A software beat, not an HBM pin. */
export const BUS_BYTES = 32;
/** On-chip tensor memory. A line that does not fit is evicted, oldest first. */
export const TMEM_BYTES = 4096;
const WORD = 4;

export type Beat = {
  cycle: number;
  cycles: number;
  phase: Phase;
  dir: "R" | "W";
  addr: number;
  bytes: number;
};

export type Stall = {
  cycle: number;
  cycles: number;
  phase: Phase;
};

export type Span = { cycles: number; bytes: number; addr: number | null };

export type GridJob = { a: number[]; b: number[]; value: number };

export class Chip {
  cycle = 0;
  bytes = 0;
  beats: Beat[] = [];
  stalls: Stall[] = [];
  tmemHits = 0;
  tmemMisses = 0;
  /** Bytes threads read from TMEM instead of the bus. */
  tmemServed = 0;
  private words = new Map<number, number>();
  private u64 = new Map<number, number>();
  private next = 0;
  private tmem = new Map<number, { values: number[]; bytes: number; used: number }>();
  private tmemUsed = 0;
  private tmemStamp = 0;

  /** Bytes occupied in RAM, including the program that was paged in. */
  get used(): number {
    return this.next;
  }

  /** The program occupies [0, base). The heap starts at base. */
  seat(base: number) {
    if (base < 0 || base > RAM_BYTES) throw new Error("RAM: program does not fit");
    this.next = base;
  }

  alloc(nFloats: number): number {
    return this.claim(nFloats * WORD);
  }

  allocBytes(n: number): number {
    if (this.next % 8) this.claim(8 - (this.next % 8));
    return this.claim(n);
  }

  private claim(n: number): number {
    if (n < 0 || this.next + n > RAM_BYTES) throw new Error("RAM: allocation does not fit");
    const addr = this.next;
    this.next += n;
    return addr;
  }

  peek(addr: number): number {
    return this.words.get(addr / WORD) ?? Number.NaN;
  }

  peekU64(addr: number): number {
    return this.u64.get(addr) ?? Number.NaN;
  }

  write(addr: number, values: number[], phase: Phase) {
    this.touch(addr, values.length * WORD, "W", phase);
    for (let i = 0; i < values.length; i++) this.words.set(addr / WORD + i, values[i]);
  }

  writeU64(addr: number, values: number[], phase: Phase) {
    this.touch(addr, values.length * 8, "W", phase);
    for (let i = 0; i < values.length; i++) this.u64.set(addr + i * 8, values[i]);
  }

  read(addr: number, n: number, phase: Phase): number[] {
    this.touch(addr, n * WORD, "R", phase);
    const out: number[] = [];
    for (let i = 0; i < n; i++) out.push(this.words.get(addr / WORD + i) ?? 0);
    return out;
  }

  /** One bus beat covering already-resident bytes. Does not copy them again. */
  touch(addr: number, bytes: number, dir: "R" | "W", phase: Phase) {
    const cycles = Math.max(1, Math.ceil(bytes / BUS_BYTES));
    this.beats.push({ cycle: this.cycle, cycles, phase, dir, addr, bytes });
    this.bytes += bytes;
    this.cycle += cycles;
  }

  stall(cycles: number, phase: Phase) {
    const n = Math.max(0, cycles);
    if (n === 0) return;
    this.stalls.push({ cycle: this.cycle, cycles: n, phase });
    this.cycle += n;
  }

  clearTmem() {
    this.tmem.clear();
    this.tmemUsed = 0;
  }

  /**
   * One line, one SIMT broadcast. `threads` is how many lanes share this address.
   * A miss fills from the bus. A hit does not.
   */
  fetch(addr: number, n: number, threads: number): number[] {
    const bytes = n * WORD;
    const line = this.tmem.get(addr);
    if (line && line.values.length === n) {
      line.used = ++this.tmemStamp;
      this.tmemHits += threads;
      this.tmemServed += bytes * threads;
      this.stall(1, "tmem");
      return line.values;
    }
    const values = this.read(addr, n, "tmem");
    this.install(addr, values);
    this.tmemMisses += 1;
    const rest = Math.max(0, threads - 1);
    this.tmemHits += rest;
    this.tmemServed += bytes * rest;
    this.stall(1, "tmem");
    return values;
  }

  private install(addr: number, values: number[]) {
    const bytes = values.length * WORD;
    if (bytes > TMEM_BYTES) return;
    if (this.tmem.has(addr)) {
      this.tmemUsed -= this.tmem.get(addr)!.bytes;
      this.tmem.delete(addr);
    }
    while (this.tmemUsed + bytes > TMEM_BYTES && this.tmem.size > 0) {
      let oldestAddr = -1;
      let oldestUsed = Infinity;
      for (const [key, line] of this.tmem) {
        if (line.used < oldestUsed) {
          oldestUsed = line.used;
          oldestAddr = key;
        }
      }
      const evicted = this.tmem.get(oldestAddr);
      if (!evicted) break;
      this.tmem.delete(oldestAddr);
      this.tmemUsed -= evicted.bytes;
    }
    this.tmem.set(addr, { values: values.slice(), bytes, used: ++this.tmemStamp });
    this.tmemUsed += bytes;
  }
}

export function tally(chip: Chip, beat0: number, stall0: number): Record<Phase, Span> {
  const spans: Record<Phase, Span> = {
    hbm: { cycles: 0, bytes: 0, addr: null },
    tmem: { cycles: 0, bytes: 0, addr: null },
    mma: { cycles: 0, bytes: 0, addr: null },
    epilogue: { cycles: 0, bytes: 0, addr: null },
  };
  for (const beat of chip.beats.slice(beat0)) {
    const span = spans[beat.phase];
    span.cycles += beat.cycles;
    span.bytes += beat.bytes;
    span.addr = beat.addr;
  }
  for (const stall of chip.stalls.slice(stall0)) spans[stall.phase].cycles += stall.cycles;
  return spans;
}

type Op =
  | { op: "mov.u32"; dst: string }
  | { op: "ld.param.u64"; dst: string; param: string }
  | { op: "ld.param.u32"; dst: string; param: string }
  | { op: "mul.wide.u32"; dst: string; a: string; imm: number }
  | { op: "add.u64"; dst: string; a: string; b: string }
  | { op: "ld.global.u64"; dst: string; addr: string }
  | { op: "pi.dot.block8.f32"; dst: string; a: string; b: string; k: string }
  | { op: "pi.mma.block8.f32"; c: string; mapA: string; mapB: string; tid: string; k: string; tm: string; tn: string }
  | { op: "st.global.f32"; addr: string; src: string }
  | { op: "ret" };

/** One grid. `%tid.x` selects the pair. `pi.dot.block8.f32` is this chip's opcode, not NVIDIA's. */
export const PTX_KERNEL = `.version 8.5
.target sm_100
.address_size 64

// One grid. %tid.x selects the pair. Each operand is stored once.
// pi.dot.block8.f32 keeps that pair in TMEM. A shared address is not read from the bus again.
.visible .entry pi_dot(.param .u64 MapA, .param .u64 MapB, .param .u64 C, .param .u32 K)
{
    .reg .b32 %tid, %k;
    .reg .b64 %mapa, %mapb, %rc, %ra, %rb, %addr;
    .reg .f32 %acc;
    mov.u32 %tid, %tid.x;
    ld.param.u64 %mapa, [MapA];
    ld.param.u64 %mapb, [MapB];
    ld.param.u64 %rc, [C];
    ld.param.u32 %k, [K];
    mul.wide.u32 %addr, %tid, 8;
    add.u64 %addr, %mapa, %addr;
    ld.global.u64 %ra, [%addr];
    mul.wide.u32 %addr, %tid, 8;
    add.u64 %addr, %mapb, %addr;
    ld.global.u64 %rb, [%addr];
    pi.dot.block8.f32 %acc, [%ra], [%rb], %k;
    mul.wide.u32 %addr, %tid, 4;
    add.u64 %rc, %rc, %addr;
    st.global.f32 [%rc], %acc;
    ret;
}
`;

/** One thread, one 4×4 tile, one issue. `pi.mma.block8.f32` is this chip's opcode, not NVIDIA's. */
export const MMA_M = 4;
export const MMA_N = 4;

export type TileJob = { rows: number[][]; cols: number[][]; values: number[][] };

export const PTX_MMA = `.version 8.5
.target sm_100
.address_size 64

// One thread owns a 4x4 tile. Tiles that fit on the SMs are one wave: the pipe runs once.
// The bus is shared, so the fills stay in order. pi.mma.block8.f32 is this chip's opcode.
.visible .entry pi_mma(.param .u64 MapA, .param .u64 MapB, .param .u64 C, .param .u32 K, .param .u32 TM, .param .u32 TN)
{
    .reg .b32 %tid, %k, %tm, %tn;
    .reg .b64 %mapa, %mapb, %rc;
    mov.u32 %tid, %tid.x;
    ld.param.u64 %mapa, [MapA];
    ld.param.u64 %mapb, [MapB];
    ld.param.u64 %rc, [C];
    ld.param.u32 %k, [K];
    ld.param.u32 %tm, [TM];
    ld.param.u32 %tn, [TN];
    pi.mma.block8.f32 [%rc], [%mapa], [%mapb], %tid, %k, %tm, %tn;
    ret;
}
`;

export function parsePtx(src: string): Op[] {
  const ops: Op[] = [];
  for (const raw of src.split("\n")) {
    const line = raw.replace(/\/\/.*$/, "").trim();
    if (!line || line.startsWith(".") || line === "{" || line === "}" || line === "(" || line === ")") continue;
    const s = line.endsWith(";") ? line.slice(0, -1).trim() : line;
    if (s === "ret") {
      ops.push({ op: "ret" });
      continue;
    }
    if (s === "mov.u32 %tid, %tid.x") {
      ops.push({ op: "mov.u32", dst: "%tid" });
      continue;
    }
    let m = /^ld\.param\.u64\s+(%\w+),\s+\[(\w+)\]$/.exec(s);
    if (m) {
      ops.push({ op: "ld.param.u64", dst: m[1], param: m[2] });
      continue;
    }
    m = /^ld\.param\.u32\s+(%\w+),\s+\[(\w+)\]$/.exec(s);
    if (m) {
      ops.push({ op: "ld.param.u32", dst: m[1], param: m[2] });
      continue;
    }
    m = /^mul\.wide\.u32\s+(%\w+),\s+(%\w+),\s+(\d+)$/.exec(s);
    if (m) {
      ops.push({ op: "mul.wide.u32", dst: m[1], a: m[2], imm: Number(m[3]) });
      continue;
    }
    m = /^add\.u64\s+(%\w+),\s+(%\w+),\s+(%\w+)$/.exec(s);
    if (m) {
      ops.push({ op: "add.u64", dst: m[1], a: m[2], b: m[3] });
      continue;
    }
    m = /^ld\.global\.u64\s+(%\w+),\s+\[(%\w+)\]$/.exec(s);
    if (m) {
      ops.push({ op: "ld.global.u64", dst: m[1], addr: m[2] });
      continue;
    }
    m = /^pi\.dot\.block8\.f32\s+(%\w+),\s+\[(%\w+)\],\s+\[(%\w+)\],\s+(%\w+)$/.exec(s);
    if (m) {
      ops.push({ op: "pi.dot.block8.f32", dst: m[1], a: m[2], b: m[3], k: m[4] });
      continue;
    }
    m = /^pi\.mma\.block8\.f32\s+\[(%\w+)\],\s+\[(%\w+)\],\s+\[(%\w+)\],\s+(%\w+),\s+(%\w+),\s+(%\w+),\s+(%\w+)$/.exec(s);
    if (m) {
      ops.push({
        op: "pi.mma.block8.f32",
        c: m[1],
        mapA: m[2],
        mapB: m[3],
        tid: m[4],
        k: m[5],
        tm: m[6],
        tn: m[7],
      });
      continue;
    }
    m = /^st\.global\.f32\s+\[(%\w+)\],\s+(%\w+)$/.exec(s);
    if (m) {
      ops.push({ op: "st.global.f32", addr: m[1], src: m[2] });
      continue;
    }
    throw new Error(`PTX: cannot run '${s}'`);
  }
  return ops;
}

export const PTX_OPS = parsePtx(PTX_KERNEL);
export const PTX_MMA_OPS = parsePtx(PTX_MMA);

function reg(regs: Map<string, number>, name: string): number {
  const v = regs.get(name);
  if (v == null || !Number.isFinite(v)) throw new Error(`PTX: ${name} is empty`);
  return v;
}

/** Contiguous addresses in one SIMT instruction share a beat. Identical reloads do not. */
function touchGather(chip: Chip, addrs: number[], width: number, phase: Phase) {
  const order = addrs.map((addr, i) => ({ addr, i })).sort((a, b) => a.addr - b.addr);
  let start = order[0].addr;
  let end = order[0].addr + width;
  const flush = () => chip.touch(start, end - start, "R", phase);
  for (let n = 1; n < order.length; n++) {
    const addr = order[n].addr;
    if (addr === order[n - 1].addr) {
      chip.touch(addr, width, "R", phase);
      continue;
    }
    if (addr === end) end = addr + width;
    else {
      flush();
      start = addr;
      end = addr + width;
    }
  }
  flush();
}

/**
 * One launch, one kernel. Unique operands are written once.
 * pi.dot.block8.f32 fills TMEM on the first use of an address and reuses it.
 */
export function issueGrid(chip: Chip, jobs: GridJob[]): { unique: number } {
  if (jobs.length === 0) return { unique: 0 };
  chip.clearTmem();
  const k = jobs[0].a.length;
  for (const job of jobs) {
    if (job.a.length !== k || job.b.length !== k) throw new Error("PTX: one grid cannot mix lengths");
  }
  const pool = new Map<string, number>();
  const intern = (v: number[]) => {
    const key = v.join(",");
    const hit = pool.get(key);
    if (hit != null) return hit;
    const addr = chip.alloc(v.length);
    chip.write(addr, v, "hbm");
    pool.set(key, addr);
    return addr;
  };
  const pairs = jobs.map((job) => [intern(job.a), intern(job.b)] as const);
  const mapA = chip.allocBytes(jobs.length * 8);
  const mapB = chip.allocBytes(jobs.length * 8);
  chip.writeU64(mapA, pairs.map((p) => p[0]), "hbm");
  chip.writeU64(mapB, pairs.map((p) => p[1]), "hbm");
  const C = chip.alloc(jobs.length);
  const params: Record<string, number> = { MapA: mapA, MapB: mapB, C, K: k };
  const regs = jobs.map(() => new Map<string, number>());
  const n = jobs.length;

  for (const op of PTX_OPS) {
    if (op.op === "mov.u32") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, t);
    } else if (op.op === "ld.param.u64" || op.op === "ld.param.u32") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, params[op.param]);
    } else if (op.op === "mul.wide.u32") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, reg(regs[t], op.a) * op.imm);
    } else if (op.op === "add.u64") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, reg(regs[t], op.a) + reg(regs[t], op.b));
    } else if (op.op === "ld.global.u64") {
      const addrs = regs.map((r) => reg(r, op.addr));
      touchGather(chip, addrs, 8, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, chip.peekU64(addrs[t]));
    } else if (op.op === "pi.dot.block8.f32") {
      const kLen = reg(regs[0], op.k);
      const loadSide = (name: string) => {
        const addrs = regs.map((r) => reg(r, name));
        const data = new Map<number, number[]>();
        const count = new Map<number, number>();
        for (const addr of addrs) count.set(addr, (count.get(addr) ?? 0) + 1);
        for (const [addr, threads] of count) data.set(addr, chip.fetch(addr, kLen, threads));
        return addrs.map((addr) => data.get(addr)!);
      };
      const as = loadSide(op.a);
      const bs = loadSide(op.b);
      for (let t = 0; t < n; t++) {
        const gemm: GemmResult = dotNvfp4(as[t], bs[t], PI_BLOCK);
        chip.stall(Math.max(1, Math.ceil(k / PI_BLOCK)), "mma");
        if (gemm.c[0][0] !== jobs[t].value) throw new Error("PTX: bus data does not match the Pi value");
        regs[t].set(op.dst, gemm.c[0][0]);
      }
    } else if (op.op === "st.global.f32") {
      const addrs = regs.map((r) => reg(r, op.addr));
      const values = regs.map((r) => reg(r, op.src));
      const packed = addrs.every((addr, i) => addr === addrs[0] + i * WORD);
      if (packed) chip.write(addrs[0], values, "epilogue");
      else for (let t = 0; t < n; t++) chip.write(addrs[t], [values[t]], "epilogue");
    } else if (op.op === "ret") {
      break;
    }
  }

  for (let t = 0; t < n; t++) {
    if (chip.peek(C + t * WORD) !== jobs[t].value) throw new Error("PTX: st.global.f32 did not land");
  }
  return { unique: pool.size };
}

/** One thread, one tile. Tiles that fit on the SMs share one pipe. Fills stay on the shared bus. */
export function issueTiles(chip: Chip, jobs: TileJob[]): { unique: number; waves: number } {
  if (jobs.length === 0) return { unique: 0, waves: 0 };
  chip.clearTmem();
  const tm = jobs[0].rows.length;
  const tn = jobs[0].cols.length;
  const k = jobs[0].rows[0].length;
  if (tm < 1 || tn < 1 || tm > MMA_M || tn > MMA_N) throw new Error("PTX: tile is not 4x4 or smaller");
  for (const job of jobs) {
    if (job.rows.length !== tm || job.cols.length !== tn || job.rows[0].length !== k) {
      throw new Error("PTX: one grid cannot mix tile shapes");
    }
  }
  const pool = new Map<string, number>();
  const intern = (v: number[]) => {
    const key = v.join(",");
    const hit = pool.get(key);
    if (hit != null) return hit;
    const addr = chip.alloc(v.length);
    chip.write(addr, v, "hbm");
    pool.set(key, addr);
    return addr;
  };
  const rowAddrs = jobs.map((job) => job.rows.map(intern));
  const colAddrs = jobs.map((job) => job.cols.map(intern));
  const mapA = chip.allocBytes(jobs.length * tm * 8);
  const mapB = chip.allocBytes(jobs.length * tn * 8);
  chip.writeU64(mapA, rowAddrs.flat(), "hbm");
  chip.writeU64(mapB, colAddrs.flat(), "hbm");
  const C = chip.alloc(jobs.length * tm * tn);
  const params: Record<string, number> = { MapA: mapA, MapB: mapB, C, K: k, TM: tm, TN: tn };
  const regs = jobs.map(() => new Map<string, number>());
  const n = jobs.length;
  const waves = Math.max(1, Math.ceil(n / (SMS_PER_DIE * 2)));
  const flat: number[] = [];

  for (const op of PTX_MMA_OPS) {
    if (op.op === "mov.u32") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, t);
    } else if (op.op === "ld.param.u64" || op.op === "ld.param.u32") {
      chip.stall(1, "tmem");
      for (let t = 0; t < n; t++) regs[t].set(op.dst, params[op.param]);
    } else if (op.op === "pi.mma.block8.f32") {
      chip.touch(mapA, n * tm * 8, "R", "tmem");
      chip.touch(mapB, n * tn * 8, "R", "tmem");
      const uses = new Map<number, number>();
      for (const addrs of [...rowAddrs, ...colAddrs]) {
        for (const addr of addrs) uses.set(addr, (uses.get(addr) ?? 0) + 1);
      }
      const data = new Map<number, number[]>();
      for (const [addr, threads] of uses) data.set(addr, chip.fetch(addr, k, threads));
      const pipe = Math.max(1, Math.ceil(k / PI_BLOCK));
      for (let t = 0; t < n; t++) {
        const rows = rowAddrs[t].map((addr) => data.get(addr)!);
        const cols = colAddrs[t].map((addr) => data.get(addr)!);
        const b = rows[0].map((_, ki) => cols.map((col) => col[ki]));
        const gemm = gemmNvfp4(rows, b, PI_BLOCK);
        for (let i = 0; i < tm; i++) {
          for (let j = 0; j < tn; j++) {
            if (gemm.c[i][j] !== jobs[t].values[i][j]) throw new Error("PTX: tile does not match the Pi value");
            flat.push(gemm.c[i][j]);
          }
        }
      }
      chip.stall(waves * pipe, "mma");
      chip.write(C, flat, "epilogue");
    } else if (op.op === "ret") {
      break;
    } else {
      throw new Error(`PTX: pi_mma cannot run ${op.op}`);
    }
  }

  for (let i = 0; i < flat.length; i++) {
    if (chip.peek(C + i * WORD) !== flat[i]) throw new Error("PTX: tile store did not land");
  }
  return { unique: pool.size, waves };
}
