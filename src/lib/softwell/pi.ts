import {
  B300_BLOCK,
  PI_BLOCK,
  gemmFp,
  gemmNvfp4,
  dotNvfp4,
  maxAbs,
  reduceNvfp4,
  softmaxRows,
  transpose,
  type GemmResult,
} from "./nvfp4";
import { launch, spanEvents, type DieEvent, type Phase } from "./silicon";
import { Chip, MMA_M, MMA_N, PTX_KERNEL, PTX_MMA, issueGrid, issueTiles, tally } from "./chip";

export type Proof = {
  name: string;
  measured: number;
  bound: number;
  holds: boolean;
};

export type ProgramResult =
  | {
      ok: true;
      prints: string[];
      valueText: string;
      kind: "matrix" | "scalar" | "text";
      matrix?: number[][];
      defect?: number[][];
      scalar?: number;
      reference?: number;
      proofs: Proof[];
      events: DieEvent[];
      ptx: string;
      isa: string[];
      macs: number;
      cycles: number;
      busBytes: number;
      busBeats: number;
      tmemHits: number;
      tmemMisses: number;
      tmemServed: number;
      /** Max |Δ| of the same multiply quantized as B300 NVFP4, block of 16. Null if no tensor word ran. */
      b300Measured: number | null;
      fp8MaxAbs: number | null;
      note: string;
    }
  | { ok: false; error: string; line: number };

class PiFail extends Error {
  line: number;
  constructor(message: string, line = 1) {
    super(message);
    this.line = line;
  }
}

type Tok =
  | { k: "num"; v: number; line: number }
  | { k: "str"; v: string; line: number }
  | { k: "id"; v: string; line: number }
  | { k: "sym"; v: string; line: number }
  | { k: "eof"; line: number };

type Expr =
  | { k: "num"; v: number; line: number }
  | { k: "str"; v: string; line: number }
  | { k: "name"; v: string; line: number }
  | { k: "list"; items: Expr[]; line: number }
  | { k: "unary"; op: string; e: Expr; line: number }
  | { k: "bin"; op: string; l: Expr; r: Expr; line: number }
  | { k: "idx"; list: Expr; i: Expr; line: number }
  | { k: "call"; name: string; args: Expr[]; line: number };

type Stmt =
  | { k: "let"; name: string; e: Expr; line: number }
  | { k: "return"; e: Expr | null; line: number }
  | { k: "if"; c: Expr; t: Stmt[]; e: Stmt[] | null; line: number }
  | { k: "for"; name: string; iter: Expr; body: Stmt[]; line: number }
  | { k: "gpu"; name: string; iter: Expr; out: string; body: Stmt[]; line: number }
  | { k: "while"; c: Expr; body: Stmt[]; line: number }
  | { k: "expr"; e: Expr; line: number };

type Skill = { params: string[]; body: Stmt[] };

class ReturnSignal {
  value: unknown;
  constructor(value: unknown) {
    this.value = value;
  }
}

function lex(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  let line = 1;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === "\n") {
      line++;
      i++;
      continue;
    }
    if (c === " " || c === "\t" || c === "\r") {
      i++;
      continue;
    }
    if (c === "#") {
      while (i < n && src[i] !== "\n") i++;
      continue;
    }
    if (c === '"') {
      i++;
      let s = "";
      const start = line;
      while (i < n && src[i] !== '"') {
        if (src[i] === "\n") throw new PiFail("unterminated string", start);
        s += src[i];
        i++;
      }
      if (i >= n) throw new PiFail("unterminated string", start);
      i++;
      out.push({ k: "str", v: s, line: start });
      continue;
    }
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(src[i + 1] ?? ""))) {
      const start = i;
      const startLine = line;
      while (i < n && /[0-9]/.test(src[i])) i++;
      if (src[i] === ".") {
        i++;
        while (i < n && /[0-9]/.test(src[i])) i++;
      }
      out.push({ k: "num", v: Number(src.slice(start, i)), line: startLine });
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      const start = i;
      const startLine = line;
      while (i < n && /[A-Za-z0-9_]/.test(src[i])) i++;
      out.push({ k: "id", v: src.slice(start, i), line: startLine });
      continue;
    }
    const two = src.slice(i, i + 2);
    if (["==", "!=", "<=", ">=", "&&", "||", "++", "//", "->"].includes(two)) {
      out.push({ k: "sym", v: two, line });
      i += 2;
      continue;
    }
    if ("(){}[],+-*/%=<>!^".includes(c)) {
      out.push({ k: "sym", v: c, line });
      i++;
      continue;
    }
    throw new PiFail(`unexpected '${c}'`, line);
  }
  out.push({ k: "eof", line });
  return out;
}

class Parser {
  i = 0;
  toks: Tok[];
  constructor(toks: Tok[]) {
    this.toks = toks;
  }

  cur(): Tok {
    return this.toks[this.i] ?? { k: "eof", line: 1 };
  }

  line(): number {
    return this.cur().line;
  }

  eatSym(s: string): boolean {
    const t = this.cur();
    if (t.k === "sym" && t.v === s) {
      this.i++;
      return true;
    }
    return false;
  }

  eatId(s?: string): string | null {
    const t = this.cur();
    if (t.k === "id" && (s === undefined || t.v === s)) {
      this.i++;
      return t.v;
    }
    return null;
  }

  expectSym(s: string) {
    if (!this.eatSym(s)) throw new PiFail(`expected '${s}'`, this.line());
  }

  parseProgram(): Map<string, Skill> {
    const skills = new Map<string, Skill>();
    while (this.cur().k !== "eof") {
      if (this.eatId("import")) throw new PiFail("import is the compiler's word; this page runs one file", this.line());
      if (!this.eatId("skill")) throw new PiFail("expected skill", this.line());
      const nameTok = this.cur();
      const name = this.eatId();
      if (!name) throw new PiFail("expected skill name", this.line());
      this.expectSym("(");
      const params: string[] = [];
      if (!this.eatSym(")")) {
        for (;;) {
          const p = this.eatId();
          if (!p) throw new PiFail("expected parameter", this.line());
          params.push(p);
          if (this.eatSym(")")) break;
          this.expectSym(",");
        }
      }
      if (this.eatSym("->")) {
        if (!this.eatId()) throw new PiFail("expected a type after ->", this.line());
      }
      const body = this.parseBlock();
      if (skills.has(name)) throw new PiFail(`skill '${name}' defined twice`, nameTok.line);
      skills.set(name, { params, body });
    }
    return skills;
  }

  parseBlock(): Stmt[] {
    this.expectSym("{");
    const body: Stmt[] = [];
    while (!this.eatSym("}")) {
      if (this.cur().k === "eof") throw new PiFail("unclosed block", this.line());
      body.push(this.parseStmt());
    }
    return body;
  }

  parseStmt(): Stmt {
    const line = this.line();
    if (this.eatId("let")) {
      const name = this.eatId();
      if (!name) throw new PiFail("expected name after let", line);
      this.expectSym("=");
      return { k: "let", name, e: this.parseExpr(), line };
    }
    if (this.eatId("return")) {
      const t = this.cur();
      if (t.k === "sym" && (t.v === "}" || t.v === "{")) return { k: "return", e: null, line };
      if (t.k === "id" && (t.v === "skill" || t.v === "let" || t.v === "if" || t.v === "for" || t.v === "while" || t.v === "return")) {
        return { k: "return", e: null, line };
      }
      return { k: "return", e: this.parseExpr(), line };
    }
    if (this.eatId("if")) {
      const c = this.parseExpr();
      const t = this.parseBlock();
      let e: Stmt[] | null = null;
      if (this.eatId("else")) {
        const nxt = this.cur();
        if (nxt.k === "id" && nxt.v === "if") {
          e = [this.parseStmt()];
        } else {
          e = this.parseBlock();
        }
      }
      return { k: "if", c, t, e, line };
    }
    if (this.eatId("do")) return { k: "expr", e: this.parseExpr(), line };
    if (this.eatId("gpu")) {
      if (!this.eatId("for")) throw new PiFail("expected 'gpu for'", line);
      const name = this.eatId();
      if (!name) throw new PiFail("expected loop name", line);
      if (!this.eatId("in")) throw new PiFail("expected 'in'", line);
      const iter = this.parseExpr();
      this.expectSym("->");
      const out = this.eatId();
      if (!out) throw new PiFail("expected the output name after ->", line);
      return { k: "gpu", name, iter, out, body: this.parseBlock(), line };
    }
    if (this.eatId("for")) {
      const name = this.eatId();
      if (!name) throw new PiFail("expected loop name", line);
      if (!this.eatId("in")) throw new PiFail("expected 'in'", line);
      const iter = this.parseExpr();
      return { k: "for", name, iter, body: this.parseBlock(), line };
    }
    if (this.eatId("while")) {
      const c = this.parseExpr();
      return { k: "while", c, body: this.parseBlock(), line };
    }
    return { k: "expr", e: this.parseExpr(), line };
  }

  parseExpr(): Expr {
    return this.parseOr();
  }

  parseOr(): Expr {
    let l = this.parseAnd();
    while (this.eatSym("||") || this.eatId("or")) {
      const line = this.line();
      l = { k: "bin", op: "||", l, r: this.parseAnd(), line };
    }
    return l;
  }

  parseAnd(): Expr {
    let l = this.parseCmp();
    while (this.eatSym("&&") || this.eatId("and")) {
      const line = this.line();
      l = { k: "bin", op: "&&", l, r: this.parseCmp(), line };
    }
    return l;
  }

  parseCmp(): Expr {
    let l = this.parseSum();
    for (;;) {
      const t = this.cur();
      if (t.k === "sym" && ["==", "!=", "<", ">", "<=", ">="].includes(t.v)) {
        this.i++;
        l = { k: "bin", op: t.v, l, r: this.parseSum(), line: t.line };
      } else break;
    }
    return l;
  }

  parseSum(): Expr {
    let l = this.parseProd();
    for (;;) {
      if (this.eatSym("+")) l = { k: "bin", op: "+", l, r: this.parseProd(), line: this.line() };
      else if (this.eatSym("++")) l = { k: "bin", op: "++", l, r: this.parseProd(), line: this.line() };
      else if (this.eatSym("-")) l = { k: "bin", op: "-", l, r: this.parseProd(), line: this.line() };
      else break;
    }
    return l;
  }

  parseProd(): Expr {
    let l = this.parsePow();
    for (;;) {
      if (this.eatSym("*")) l = { k: "bin", op: "*", l, r: this.parsePow(), line: this.line() };
      else if (this.eatSym("//")) l = { k: "bin", op: "//", l, r: this.parsePow(), line: this.line() };
      else if (this.eatSym("/")) l = { k: "bin", op: "/", l, r: this.parsePow(), line: this.line() };
      else if (this.eatSym("%")) l = { k: "bin", op: "%", l, r: this.parsePow(), line: this.line() };
      else break;
    }
    return l;
  }

  parsePow(): Expr {
    const l = this.parseUnary();
    if (!this.eatSym("^")) return l;
    const r = this.parsePow();
    return { k: "bin", op: "^", l, r, line: l.line };
  }

  parseUnary(): Expr {
    if (this.eatSym("-")) return { k: "unary", op: "-", e: this.parseUnary(), line: this.line() };
    if (this.eatSym("!") || this.eatId("not")) return { k: "unary", op: "!", e: this.parseUnary(), line: this.line() };
    return this.parsePost();
  }

  parsePost(): Expr {
    let e = this.parsePrimary();
    for (;;) {
      if (this.eatSym("[")) {
        const i = this.parseExpr();
        this.expectSym("]");
        e = { k: "idx", list: e, i, line: e.line };
      } else break;
    }
    return e;
  }

  parsePrimary(): Expr {
    const t = this.cur();
    if (t.k === "num") {
      this.i++;
      return { k: "num", v: t.v, line: t.line };
    }
    if (t.k === "str") {
      this.i++;
      return { k: "str", v: t.v, line: t.line };
    }
    if (t.k === "id") {
      this.i++;
      if (t.v === "true") return { k: "num", v: 1, line: t.line };
      if (t.v === "false") return { k: "num", v: 0, line: t.line };
      if (t.v === "null") return { k: "name", v: "null", line: t.line };
      if (this.eatSym("(")) {
        const args: Expr[] = [];
        if (!this.eatSym(")")) {
          for (;;) {
            args.push(this.parseExpr());
            if (this.eatSym(")")) break;
            this.expectSym(",");
          }
        }
        return { k: "call", name: t.v, args, line: t.line };
      }
      return { k: "name", v: t.v, line: t.line };
    }
    if (this.eatSym("[")) {
      const items: Expr[] = [];
      if (!this.eatSym("]")) {
        for (;;) {
          items.push(this.parseExpr());
          if (this.eatSym("]")) break;
          this.expectSym(",");
        }
      }
      return { k: "list", items, line: t.line };
    }
    if (this.eatSym("(")) {
      const e = this.parseExpr();
      this.expectSym(")");
      return e;
    }
    throw new PiFail("expected expression", this.line());
  }
}

type Frame = Map<string, unknown>;

type Machine = {
  skills: Map<string, Skill>;
  prints: string[];
  proofs: Proof[];
  events: DieEvent[];
  isa: string[];
  issues: string[];
  chip: Chip;
  macs: number;
  cycles: number;
  b300Measured: number | null;
  fp8MaxAbs: number | null;
  note: string;
  loops: number;
  matrix?: number[][];
  defect?: number[][];
  scalar?: number;
  reference?: number;
  kind: "matrix" | "scalar" | "text";
  inGpu: number;
  dots: { measured: number; bound: number; holds: boolean; b300: number; fp: number; a: number[]; b: number[]; value: number }[];
  tiles: { rows: number[][]; cols: number[][]; values: number[][]; defect: number[][] }[];
  entry: "dot" | "mma" | "none";
};

const MAX_LOOPS = 200_000;
const MAX_DIM = 32;

function truthy(v: unknown): boolean {
  if (v == null) return false;
  if (typeof v === "number") return v !== 0 && !Number.isNaN(v);
  if (typeof v === "string") return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  return true;
}

function asNum(v: unknown, line: number): number {
  if (typeof v !== "number" || !Number.isFinite(v)) throw new PiFail("expected a number", line);
  return v;
}

function placeTiles(
  matrix: number[][],
  tiles: { values: number[][]; defect: number[][] }[],
): number[][] | null {
  if (tiles.length === 0) return null;
  const tm = tiles[0].values.length;
  const tn = tiles[0].values[0].length;
  if (tiles.some((tile) => tile.values.length !== tm || tile.values[0].length !== tn)) return null;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const tr = Math.ceil(rows / tm);
  const tc = Math.ceil(cols / tn);
  if (tr * tc !== tiles.length) return null;
  const defect = matrix.map((row) => row.map(() => 0));
  for (let t = 0; t < tiles.length; t++) {
    const ti = Math.floor(t / tc);
    const tj = t % tc;
    const tile = tiles[t];
    for (let i = 0; i < tm; i++) {
      for (let j = 0; j < tn; j++) {
        const r = ti * tm + i;
        const c = tj * tn + j;
        if (r >= rows || c >= cols || matrix[r][c] !== tile.values[i][j]) return null;
        defect[r][c] = tile.defect[i][j];
      }
    }
  }
  return defect;
}

function isMatrix(v: unknown): v is number[][] {
  if (!Array.isArray(v) || v.length === 0) return false;
  const w = Array.isArray(v[0]) ? (v[0] as unknown[]).length : -1;
  if (w <= 0) return false;
  return v.every(
    (row) =>
      Array.isArray(row) &&
      row.length === w &&
      (row as unknown[]).every((x) => typeof x === "number" && Number.isFinite(x as number)),
  );
}

function isVec(v: unknown): v is number[] {
  return Array.isArray(v) && v.every((x) => typeof x === "number" && Number.isFinite(x));
}

function requireMatrix(v: unknown, name: string, line: number): number[][] {
  if (!isMatrix(v)) throw new PiFail(`${name} must be a rectangular matrix of numbers`, line);
  if (v.length > MAX_DIM || v[0].length > MAX_DIM) throw new PiFail(`${name} exceeds ${MAX_DIM} on a side`, line);
  return v;
}

function fmtNum(n: number): string {
  if (!Number.isFinite(n)) return "nan";
  const r = Math.round(n);
  if (Math.abs(n - r) < 1e-9 && Math.abs(n) < 1e12) return String(r);
  return n.toFixed(4);
}

function fmtVal(v: unknown): string {
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "string") return v;
  if (v == null) return "null";
  if (isMatrix(v)) return v.map((row) => row.map(fmtNum).join("  ")).join("\n");
  if (Array.isArray(v)) return "[" + v.map(fmtVal).join(", ") + "]";
  return String(v);
}

function bumpLoop(m: Machine, line: number) {
  m.loops++;
  if (m.loops > MAX_LOOPS) throw new PiFail("loop ran too long", line);
}

function addProof(m: Machine, name: string, measured: number, bound: number, holds: boolean) {
  m.proofs.push({ name, measured, bound, holds });
}

function noteB300(m: Machine, measured: number) {
  m.b300Measured = m.b300Measured == null ? measured : Math.max(m.b300Measured, measured);
}

function recordGemm(m: Machine, label: string, g: GemmResult, baseline: GemmResult, shape: string) {
  addProof(m, label, g.measured, g.boundMax, g.holds);
  noteB300(m, baseline.measured);
  m.macs += g.macs;
  m.fp8MaxAbs = m.fp8MaxAbs == null ? g.fp8MaxAbs : Math.max(m.fp8MaxAbs, g.fp8MaxAbs);
  const cells = g.c.length * (g.c[0]?.length ?? 0);
  const k = g.macs / Math.max(1, cells);
  const kblocks = Math.ceil(k / PI_BLOCK);
  const sched = launch({
    cells,
    kblocks,
    macs: g.macs,
    captions: {
      hbm: `${label} — load A and B onto the software panes`,
      tmem: `${label} — Pi block ${PI_BLOCK}, E2M1 × E4M3 scale`,
      mma: `${label} — tensor MMA, FP32 accumulate`,
      epilogue: `${label} — store, then check the residual against B300's block ${B300_BLOCK}`,
    },
  });
  m.events.push(...sched.events);
  m.cycles += sched.cycles;
  m.isa.push(
    `// ${label}  ${shape}`,
    `pi.quant.block${PI_BLOCK}       kblocks=${kblocks}  bits=5.0`,
    `b300.nvfp4.block${B300_BLOCK}   |Δ|=${baseline.measured.toExponential(3)}  bits=4.5  not scheduled`,
    `pane.split             outputs=${cells}`,
    `mma.pi.acc.f32         macs=${g.macs}`,
    `prove.residual        ${g.holds ? "holds" : "FAIL"}  |Δ|=${g.measured.toExponential(3)}  bound=${g.boundMax.toExponential(3)}`,
  );
}

function captionsReduce(): Record<Phase, string> {
  return {
    hbm: "Reduce — stripe the vector across the software panes",
    tmem: `Reduce — Pi blocks of ${PI_BLOCK}`,
    mma: "Reduce — sum reconstructed values",
    epilogue: "Reduce — commit the scalar",
  };
}

function evalExpr(e: Expr, frame: Frame, m: Machine): unknown {
  switch (e.k) {
    case "num":
      return e.v;
    case "str":
      return e.v;
    case "name":
      if (e.v === "null") return null;
      if (!frame.has(e.v)) throw new PiFail(`unknown name '${e.v}'`, e.line);
      return frame.get(e.v);
    case "list":
      return e.items.map((it) => evalExpr(it, frame, m));
    case "unary": {
      const v = asNum(evalExpr(e.e, frame, m), e.line);
      if (e.op === "-") return -v;
      return truthy(v) ? 0 : 1;
    }
    case "bin": {
      if (e.op === "&&") {
        const l = evalExpr(e.l, frame, m);
        if (!truthy(l)) return 0;
        return truthy(evalExpr(e.r, frame, m)) ? 1 : 0;
      }
      if (e.op === "||") {
        const l = evalExpr(e.l, frame, m);
        if (truthy(l)) return 1;
        return truthy(evalExpr(e.r, frame, m)) ? 1 : 0;
      }
      const l = evalExpr(e.l, frame, m);
      const r = evalExpr(e.r, frame, m);
      if (e.op === "+") {
        if (typeof l === "string" || typeof r === "string") return String(l) + String(r);
        if (Array.isArray(l) && Array.isArray(r)) return [...l, ...r];
        return asNum(l, e.line) + asNum(r, e.line);
      }
      const ln = asNum(l, e.line);
      const rn = asNum(r, e.line);
      if (e.op === "-") return ln - rn;
      if (e.op === "*") return ln * rn;
      if (e.op === "/") {
        if (rn === 0) throw new PiFail("division by zero", e.line);
        return ln / rn;
      }
      if (e.op === "//") {
        if (rn === 0) throw new PiFail("division by zero", e.line);
        return Math.floor(ln / rn);
      }
      if (e.op === "%") {
        if (rn === 0) throw new PiFail("modulo by zero", e.line);
        return ln % rn;
      }
      if (e.op === "^") {
        const exp = Math.trunc(rn);
        if (exp < 0 || exp > 64) throw new PiFail("power exponent is out of range", e.line);
        let acc = 1;
        for (let i = 0; i < exp; i++) acc *= ln;
        return acc;
      }
      if (e.op === "++") {
        if (Array.isArray(l) && Array.isArray(r)) return [...l, ...r];
        return fmtVal(l) + fmtVal(r);
      }
      if (e.op === "==") return ln === rn ? 1 : 0;
      if (e.op === "!=") return ln !== rn ? 1 : 0;
      if (e.op === "<") return ln < rn ? 1 : 0;
      if (e.op === ">") return ln > rn ? 1 : 0;
      if (e.op === "<=") return ln <= rn ? 1 : 0;
      if (e.op === ">=") return ln >= rn ? 1 : 0;
      throw new PiFail(`unknown operator ${e.op}`, e.line);
    }
    case "idx": {
      const list = evalExpr(e.list, frame, m);
      let i = asNum(evalExpr(e.i, frame, m), e.line);
      if (!Array.isArray(list)) throw new PiFail("index of a non-list", e.line);
      if (i < 0) i = list.length + i;
      if (!Number.isInteger(i) || i < 0 || i >= list.length) return null;
      return list[i];
    }
    case "call":
      return call(e.name, e.args.map((a) => evalExpr(a, frame, m)), frame, m, e.line);
    default:
      throw new PiFail("bad expression", 1);
  }
}

function call(name: string, args: unknown[], _frame: Frame, m: Machine, line: number): unknown {
  const skill = m.skills.get(name);
  if (skill) {
    if (args.length !== skill.params.length) {
      throw new PiFail(`${name} expects ${skill.params.length} args`, line);
    }
    const next: Frame = new Map();
    skill.params.forEach((p, i) => next.set(p, args[i]));
    try {
      exec(skill.body, next, m);
    } catch (err) {
      if (err instanceof ReturnSignal) return err.value;
      throw err;
    }
    return null;
  }
  const builtin = BUILTINS[name];
  if (!builtin) throw new PiFail(`unknown word '${name}'`, line);
  return builtin(args, m, line);
}

function exec(body: Stmt[], frame: Frame, m: Machine) {
  for (const s of body) execOne(s, frame, m);
}

function runGpu(s: Extract<Stmt, { k: "gpu" }>, frame: Frame, m: Machine) {
  const iter = evalExpr(s.iter, frame, m);
  if (!Array.isArray(iter)) throw new PiFail("gpu for expects a list", s.line);
  const dot0 = m.dots.length;
  const tile0 = m.tiles.length;
  const beat0 = m.chip.beats.length;
  const stall0 = m.chip.stalls.length;
  const cycle0 = m.chip.cycle;
  m.inGpu++;
  const results: unknown[] = [];
  try {
    for (const item of iter) {
      bumpLoop(m, s.line);
      const inner: Frame = new Map(frame);
      inner.set(s.name, item);
      exec(s.body, inner, m);
      results.push(inner.has(s.out) ? inner.get(s.out) : null);
    }
  } finally {
    m.inGpu--;
  }
  frame.set(s.out, results);
  const dots = m.dots.slice(dot0);
  const tiles = m.tiles.slice(tile0);
  const tiledCells = tiles.reduce((sum, tile) => sum + tile.values.length * tile.values[0].length, 0);
  if (tiles.length > 0 && dots.length !== tiledCells) {
    throw new PiFail("gpu for cannot mix pi_dot and pi_mma", s.line);
  }
  let measured = 0;
  let bound = 0;
  let b300 = 0;
  let holds = true;
  for (const d of dots) {
    if (d.measured > measured) measured = d.measured;
    if (d.bound > bound) bound = d.bound;
    if (d.b300 > b300) b300 = d.b300;
    if (!d.holds) holds = false;
  }
  if (dots.length > 0) {
    addProof(m, `gpu for ${s.out}:${s.line}`, measured, bound, holds);
    noteB300(m, b300);
  }
  const cells = results.length;
  const hits0 = m.chip.tmemHits;
  const miss0 = m.chip.tmemMisses;
  const grid = tiles.length > 0 ? issueTiles(m.chip, tiles) : dots.length > 0 ? issueGrid(m.chip, dots) : { unique: 0 };
  if (tiles.length > 0) m.entry = "mma";
  else if (dots.length > 0) m.entry = "dot";
  const hits = m.chip.tmemHits - hits0;
  const misses = m.chip.tmemMisses - miss0;
  if (dots.length === 0) m.chip.stall(Math.max(1, cells), "tmem");
  const shape = tiles.length > 0 ? `${tiles[0].values.length}x${tiles[0].values[0].length}` : "";
  const sched = spanEvents({
    cells,
    from: cycle0,
    spans: tally(m.chip, beat0, stall0),
    captions: {
      hbm: `bus write — ${grid.unique} operands, once`,
      tmem: tiles.length > 0 ? `%tid.x — ${tiles.length} tiles in TMEM` : `%tid.x — TMEM holds ${grid.unique} operands`,
      mma:
        tiles.length > 0
          ? `pi.mma.block8.f32 — ${shape}, one wave`
          : dots.length > 0
            ? `pi.dot.block8.f32 — from TMEM, block ${PI_BLOCK}`
            : `gpu for — no tensor opcode`,
      epilogue: `st.global.f32 — ${tiles.length > 0 ? tiledCells : dots.length} words`,
    },
  });
  m.events.push(...sched.events);
  m.cycles = m.chip.cycle;
  m.kind = "text";
  m.note =
    tiles.length > 0
      ? `One gpu for, one kernel. Each thread owns a ${shape} tile. The tiles that fit on the SMs are one wave, so the pipe runs once. The bus is shared, so the fills stay in order. Block of 8. The B300 column is the same products in blocks of 16.`
      : dots.length > 0
        ? "One gpu for, one kernel. %tid.x picks the pair. Each operand is written once, then kept in TMEM. A shared row is not read from the bus again. Block of 8. The B300 column is the same products in blocks of 16."
        : "gpu for ran. The body did not call pi_dot, so the bus stayed quiet and there is no residual to check.";
  const mark = dots.length > 0 && holds ? "holds" : dots.length > 0 ? "FAIL" : "no pi_dot";
  const moved = m.chip.beats.slice(beat0).reduce((sum, beat) => sum + beat.bytes, 0);
  const threads = tiles.length > 0 ? tiles.length : dots.length;
  m.issues.push(
    tiles.length > 0
      ? `// grid %tid.x  threads ${threads}  tile ${shape}  wave ${"waves" in grid ? grid.waves : 1}  pipe once  operands ${grid.unique} stored once  tmem hits ${hits}  misses ${misses}  clock ${m.chip.cycle}  bus ${moved} B`
      : `// grid %tid.x  threads ${threads}  operands ${grid.unique} stored once  tmem hits ${hits}  misses ${misses}  clock ${m.chip.cycle}  bus ${moved} B`,
    `// residual ${mark}${dots.length > 0 ? `  |Δ|=${measured.toExponential(3)}  bound=${bound.toExponential(3)}  b300=${b300.toExponential(3)}` : ""}`,
  );
}

function execOne(s: Stmt, frame: Frame, m: Machine) {
  switch (s.k) {
    case "let":
      frame.set(s.name, evalExpr(s.e, frame, m));
      return;
    case "return":
      throw new ReturnSignal(s.e ? evalExpr(s.e, frame, m) : null);
    case "expr":
      evalExpr(s.e, frame, m);
      return;
    case "if":
      if (truthy(evalExpr(s.c, frame, m))) exec(s.t, frame, m);
      else if (s.e) exec(s.e, frame, m);
      return;
    case "for": {
      const iter = evalExpr(s.iter, frame, m);
      if (!Array.isArray(iter)) throw new PiFail("for expects a list", s.line);
      for (const item of iter) {
        bumpLoop(m, s.line);
        frame.set(s.name, item);
        exec(s.body, frame, m);
      }
      return;
    }
    case "gpu":
      runGpu(s, frame, m);
      return;
    case "while":
      while (truthy(evalExpr(s.c, frame, m))) {
        bumpLoop(m, s.line);
        exec(s.body, frame, m);
      }
      return;
    default:
      return;
  }
}

const BUILTINS: Record<string, (args: unknown[], m: Machine, line: number) => unknown> = {
  print(args, m) {
    m.prints.push(args.map(fmtVal).join(" "));
    return null;
  },
  range(args, _m, line) {
    if (args.length !== 2) throw new PiFail("range expects (start, end)", line);
    const a = Math.trunc(asNum(args[0], line));
    const b = Math.trunc(asNum(args[1], line));
    if (b - a > 4096 || a > b) throw new PiFail("range is empty or too long", line);
    const out: number[] = [];
    for (let i = a; i < b; i++) out.push(i);
    return out;
  },
  len(args, _m, line) {
    const v = args[0];
    if (typeof v === "string" || Array.isArray(v)) return v.length;
    throw new PiFail("len expects a list or text", line);
  },
  push(args, _m, line) {
    const list = args[0];
    if (!Array.isArray(list)) throw new PiFail("push expects a list", line);
    if (list.length > 4096) throw new PiFail("list too long", line);
    return [...list, args[1]];
  },
  append(args, m, line) {
    return BUILTINS.push(args, m, line);
  },
  slice(args, _m, line) {
    const list = args[0];
    if (!Array.isArray(list)) throw new PiFail("slice expects a list", line);
    const a = Math.trunc(asNum(args[1], line));
    const b = Math.trunc(asNum(args[2], line));
    return list.slice(a, b);
  },
  sqrt(args, _m, line) {
    const n = asNum(args[0], line);
    if (n < 0) return null;
    return Math.sqrt(n);
  },
  exp(args, _m, line) {
    return Math.exp(asNum(args[0], line));
  },
  pi_dot(args, m, line) {
    if (m.inGpu === 0) throw new PiFail("pi_dot is a GPU word — write it inside gpu for", line);
    if (!isVec(args[0]) || !isVec(args[1])) throw new PiFail("pi_dot expects two lists of numbers", line);
    if (args[0].length !== args[1].length || args[0].length === 0) {
      throw new PiFail("pi_dot lists must be the same non-empty length", line);
    }
    if (args[0].length > 512) throw new PiFail("pi_dot list is too long", line);
    const g = dotNvfp4(args[0], args[1], PI_BLOCK);
    const baseline = dotNvfp4(args[0], args[1], B300_BLOCK);
    const value = g.c[0][0];
    m.dots.push({
      measured: g.measured,
      bound: g.boundMax,
      holds: g.holds,
      b300: baseline.measured,
      fp: g.fp[0][0],
      a: args[0].slice(),
      b: args[1].slice(),
      value,
    });
    m.macs += args[0].length;
    return value;
  },
  pi_mma(args, m, line) {
    if (m.inGpu === 0) throw new PiFail("pi_mma is a GPU word — write it inside gpu for", line);
    if (!Array.isArray(args[0]) || !Array.isArray(args[1]) || args[0].length === 0 || args[1].length === 0) {
      throw new PiFail("pi_mma expects two lists of lists", line);
    }
    if (!args[0].every(isVec) || !args[1].every(isVec)) throw new PiFail("pi_mma expects lists of numbers", line);
    const rows = args[0].map((row) => row.slice());
    const cols = args[1].map((col) => col.slice());
    if (rows.length > MMA_M || cols.length > MMA_N) {
      throw new PiFail(`pi_mma tile is at most ${MMA_M} by ${MMA_N}`, line);
    }
    const k = rows[0].length;
    if (k === 0 || k > 512 || rows.some((row) => row.length !== k) || cols.some((col) => col.length !== k)) {
      throw new PiFail("pi_mma rows and columns must share one non-empty length", line);
    }
    const b = Array.from({ length: k }, (_, ki) => cols.map((col) => col[ki]));
    const g = gemmNvfp4(rows, b, PI_BLOCK);
    const baseline = gemmNvfp4(rows, b, B300_BLOCK);
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < cols.length; j++) {
        const cellBound = g.bound[i][j];
        const cell = g.defect[i][j];
        m.dots.push({
          measured: cell,
          bound: cellBound,
          holds: cell <= cellBound + 1e-6 * Math.max(1, Math.abs(cellBound)) + 1e-8,
          b300: baseline.defect[i][j],
          fp: g.fp[i][j],
          a: rows[i],
          b: cols[j],
          value: g.c[i][j],
        });
      }
    }
    m.tiles.push({ rows, cols, values: g.c, defect: g.defect });
    m.macs += rows.length * cols.length * k;
    return g.c;
  },
  wave(args, _m, line) {
    const n = Math.trunc(asNum(args[0], line));
    if (n < 1 || n > 512) throw new PiFail("wave length must be 1..512", line);
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = Math.sin(i * 0.55) * 2.4 + ((i % 5) - 2) * 0.15;
      out.push(Math.round(x * 1000) / 1000);
    }
    return out;
  },
  tc_gemm(args, m, line) {
    const a = requireMatrix(args[0], "A", line);
    const b = requireMatrix(args[1], "B", line);
    if (a[0].length !== b.length) throw new PiFail(`K mismatch: ${a[0].length} vs ${b.length}`, line);
    const g = gemmNvfp4(a, b, PI_BLOCK);
    const baseline = gemmNvfp4(a, b, B300_BLOCK);
    recordGemm(m, "tc_gemm", g, baseline, `${a.length}x${a[0].length} · ${b.length}x${b[0].length}`);
    m.matrix = g.c;
    m.defect = g.defect;
    m.kind = "matrix";
    m.note =
      "Pi GPU, block of 8. The B300 number is this same multiply in NVFP4 blocks of 16. No module is in the loop.";
    return g.c;
  },
  tc_reduce(args, m, line) {
    if (!isVec(args[0]) || args[0].length === 0) throw new PiFail("tc_reduce expects a non-empty list of numbers", line);
    if (args[0].length > 512) throw new PiFail("tc_reduce list is too long", line);
    const r = reduceNvfp4(args[0], PI_BLOCK);
    const baseline = reduceNvfp4(args[0], B300_BLOCK);
    addProof(m, "tc_reduce", r.measured, r.bound, r.holds);
    noteB300(m, baseline.measured);
    m.macs += args[0].length;
    m.fp8MaxAbs = r.fp8MaxAbs;
    const sched = launch({
      cells: Math.min(args[0].length, 256),
      kblocks: Math.ceil(args[0].length / PI_BLOCK),
      macs: args[0].length,
      captions: captionsReduce(),
    });
    m.events.push(...sched.events);
    m.cycles += sched.cycles;
    m.isa.push(
      `// tc_reduce  n=${args[0].length}`,
      `pi.quant.block${PI_BLOCK}       blocks=${Math.ceil(args[0].length / PI_BLOCK)}`,
      `b300.nvfp4.block${B300_BLOCK}   |Δ|=${baseline.measured.toExponential(3)}  not scheduled`,
      `acc.f32.sum            reconstructed blocks`,
      `prove.residual        ${r.holds ? "holds" : "FAIL"}  |Δ|=${r.measured.toExponential(3)}  bound=${r.bound.toExponential(3)}`,
    );
    m.scalar = r.sum;
    m.reference = r.fp;
    m.kind = "scalar";
    m.note = "Pi reduction, blocks of 8. The bound is the sum of absolute defects. The B300 figure uses blocks of 16 on the same vector.";
    return r.sum;
  },
  tc_attn(args, m, line) {
    const q = requireMatrix(args[0], "Q", line);
    const k = requireMatrix(args[1], "K", line);
    const v = requireMatrix(args[2], "V", line);
    if (q[0].length !== k[0].length) throw new PiFail("Q and K must share a head width", line);
    if (k.length !== v.length) throw new PiFail("K and V must share a sequence length", line);
    const kt = transpose(k);
    const qk = gemmNvfp4(q, kt, PI_BLOCK);
    const qkB = gemmNvfp4(q, kt, B300_BLOCK);
    recordGemm(m, "tc_attn QKᵀ", qk, qkB, `${q.length}x${q[0].length} · ${kt.length}x${kt[0].length}`);
    const scale = 1 / Math.sqrt(q[0].length);
    const scores = qk.c.map((row) => row.map((x) => x * scale));
    const scoresFp = qk.fp.map((row) => row.map((x) => x * scale));
    const probs = softmaxRows(scores);
    const probsFp = softmaxRows(scoresFp);
    const av = gemmNvfp4(probs, v, PI_BLOCK);
    const avB = gemmNvfp4(probs, v, B300_BLOCK);
    recordGemm(m, "tc_attn PV", av, avB, `${probs.length}x${probs[0].length} · ${v.length}x${v[0].length}`);
    const outFp = gemmFp(probsFp, v);
    const drift = maxAbs(av.c, outFp);
    m.matrix = av.c;
    m.defect = av.c.map((row, i) => row.map((val, j) => Math.abs(val - outFp[i][j])));
    m.kind = "matrix";
    m.note = `Two Pi GEMMs, blocks of 8. Softmax stays FP32. End-to-end drift vs a pure FP32 attention is ${drift.toExponential(3)}, outside the linear bound on purpose. The B300 column is NVFP4 block 16 on the same products.`;
    m.isa.push(`epilogue.softmax.f32   scale=1/sqrt(${q[0].length})`);
    m.isa.push(`note.drift             ${drift.toExponential(3)}  not part of the residual proof`);
    return av.c;
  },
};

export function runPi(source: string): ProgramResult {
  try {
    const skills = new Parser(lex(source)).parseProgram();
    const main = skills.get("main");
    if (!main) throw new PiFail("no skill main", 1);
    if (main.params.length !== 0) throw new PiFail("main takes no arguments", 1);
    const machine: Machine = {
      skills,
      prints: [],
      proofs: [],
      events: [],
      isa: [],
      issues: [],
      chip: new Chip(),
      macs: 0,
      cycles: 0,
      b300Measured: null,
      fp8MaxAbs: null,
      note: "Host interpreter only. No SM was scheduled.",
      loops: 0,
      kind: "text",
      inGpu: 0,
      dots: [],
      tiles: [],
      entry: "none",
    };
    let value: unknown = null;
    try {
      exec(main.body, new Map(), machine);
    } catch (err) {
      if (err instanceof ReturnSignal) value = err.value;
      else throw err;
    }
    if (machine.kind === "text") {
      if (typeof value === "number") {
        machine.kind = "scalar";
        machine.scalar = value;
        if (machine.dots.length === 1) machine.reference = machine.dots[0].fp;
      } else if (isMatrix(value)) {
        machine.kind = "matrix";
        machine.matrix = value;
        const n = value.length * value[0].length;
        if (machine.dots.length >= n && machine.tiles.length === 0) {
          const tail = machine.dots.slice(-n);
          const defect: number[][] = [];
          let t = 0;
          for (const row of value) {
            const d: number[] = [];
            for (let j = 0; j < row.length; j++) d.push(tail[t++].measured);
            defect.push(d);
          }
          machine.defect = defect;
        }
        const placed = placeTiles(value, machine.tiles);
        if (placed) machine.defect = placed;
      }
    }
    const ptx =
      machine.issues.length > 0
        ? `${machine.entry === "mma" ? PTX_MMA : PTX_KERNEL}\n${machine.issues.join("\n")}\n`
        : "// no gpu for\n// the clock stayed at 0\n";
    machine.isa = ptx.split("\n");
    return {
      ok: true,
      prints: machine.prints,
      valueText: fmtVal(value),
      kind: machine.kind,
      matrix: machine.matrix,
      defect: machine.defect,
      scalar: machine.scalar,
      reference: machine.reference,
      proofs: machine.proofs,
      events: machine.events,
      ptx,
      isa: machine.isa,
      macs: machine.macs,
      cycles: machine.chip.cycle || machine.cycles,
      busBytes: machine.chip.bytes,
      busBeats: machine.chip.beats.length,
      tmemHits: machine.chip.tmemHits,
      tmemMisses: machine.chip.tmemMisses,
      tmemServed: machine.chip.tmemServed,
      b300Measured: machine.b300Measured,
      fp8MaxAbs: machine.fp8MaxAbs,
      note: machine.note,
    };
  } catch (err) {
    if (err instanceof PiFail) return { ok: false, error: err.message, line: err.line };
    return { ok: false, error: err instanceof Error ? err.message : "run failed", line: 1 };
  }
}
