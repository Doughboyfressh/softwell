/** NVFP4 as Blackwell publishes it: E2M1 values, one E4M3 scale per block of 16. */

export const PI_BLOCK = 8;
export const B300_BLOCK = 16;
export const NVFP4_BLOCK = B300_BLOCK;
const E2M1_MAG = [0, 0.5, 1, 1.5, 2, 3, 4, 6] as const;
const E4M3_MAX = 448;
const E4M3_MIN_POS = 2 ** -9;

export function nearestE2M1(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  let best = 0;
  let bestD = Infinity;
  for (const v of E2M1_MAG) {
    const d = Math.abs(ax - v);
    if (d < bestD) {
      bestD = d;
      best = v;
    }
  }
  return best === 0 ? 0 : sign * best;
}

function decodeE4M3Bits(bits: number): number {
  const exp = (bits >> 3) & 0xf;
  const mant = bits & 0x7;
  if (exp === 0) return mant * E4M3_MIN_POS;
  if (exp === 15 && mant === 7) return Number.NaN;
  return (1 + mant / 8) * 2 ** (exp - 7);
}

/** Round-to-nearest E4M3 finite value. Ties in the mantissa round away from zero via Math.round. */
export function quantizeE4M3(x: number): number {
  if (!Number.isFinite(x) || x === 0) return 0;
  const sign = x < 0 ? -1 : 1;
  const ax0 = Math.abs(x);
  if (ax0 >= E4M3_MAX) return sign * E4M3_MAX;
  if (ax0 < E4M3_MIN_POS / 2) return 0;

  if (ax0 < 2 ** -6) {
    const mant = Math.min(7, Math.max(0, Math.round(ax0 / E4M3_MIN_POS)));
    return sign * decodeE4M3Bits(mant);
  }

  let e = Math.floor(Math.log2(ax0));
  let norm = ax0 / 2 ** e;
  if (norm >= 2) {
    norm /= 2;
    e += 1;
  } else if (norm < 1) {
    norm *= 2;
    e -= 1;
  }
  let m3 = Math.round((norm - 1) * 8);
  if (m3 === 8) {
    m3 = 0;
    e += 1;
  }
  if (e > 8) return sign * E4M3_MAX;
  if (e < -6) {
    const mant = Math.min(7, Math.max(0, Math.round(ax0 / E4M3_MIN_POS)));
    return sign * decodeE4M3Bits(mant);
  }
  const expField = e + 7;
  if (expField === 15 && m3 >= 7) return sign * E4M3_MAX;
  const mag = decodeE4M3Bits((expField << 3) | m3);
  if (!Number.isFinite(mag)) return sign * E4M3_MAX;
  return sign * mag;
}

export function quantizeBlock(values: number[]): { scale: number; recon: number[] } {
  let amax = 0;
  for (const v of values) amax = Math.max(amax, Math.abs(v));
  let scale = 0;
  if (amax > 0) {
    scale = quantizeE4M3(amax / 6);
    if (scale === 0) scale = E4M3_MIN_POS;
  }
  const recon = values.map((v) => (scale === 0 ? 0 : nearestE2M1(v / scale) * scale));
  return { scale, recon };
}

function quantizeRows(matrix: number[][], block: number): { recon: number[][]; delta: number[][] } {
  const recon: number[][] = [];
  const delta: number[][] = [];
  for (const row of matrix) {
    const rq = new Array<number>(row.length);
    const dq = new Array<number>(row.length);
    for (let k = 0; k < row.length; k += block) {
      const n = Math.min(block, row.length - k);
      const slice = row.slice(k, k + n);
      const q = quantizeBlock(slice);
      for (let t = 0; t < n; t++) {
        rq[k + t] = q.recon[t];
        dq[k + t] = q.recon[t] - slice[t];
      }
    }
    recon.push(rq);
    delta.push(dq);
  }
  return { recon, delta };
}

export function transpose(m: number[][]): number[][] {
  if (m.length === 0) return [];
  const rows = m.length;
  const cols = m[0].length;
  const out: number[][] = [];
  for (let j = 0; j < cols; j++) {
    const row = new Array<number>(rows);
    for (let i = 0; i < rows; i++) row[i] = m[i][j];
    out.push(row);
  }
  return out;
}

export type GemmResult = {
  c: number[][];
  fp: number[][];
  defect: number[][];
  bound: number[][];
  measured: number;
  boundMax: number;
  holds: boolean;
  macs: number;
  fp8MaxAbs: number;
};

function within(measured: number, bound: number): boolean {
  return measured <= bound + 1e-6 * Math.max(1, Math.abs(bound)) + 1e-8;
}

export function dotNvfp4(a: number[], b: number[], block = B300_BLOCK): GemmResult {
  return gemmNvfp4(
    [a],
    b.map((v) => [v]),
    block,
  );
}

export function gemmFp(a: number[][], b: number[][]): number[][] {
  const m = a.length;
  const k = a[0]?.length ?? 0;
  const n = b[0]?.length ?? 0;
  const c: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row = new Array<number>(n).fill(0);
    for (let t = 0; t < k; t++) {
      const av = a[i][t];
      const brow = b[t];
      for (let j = 0; j < n; j++) row[j] += av * brow[j];
    }
    c.push(row);
  }
  return c;
}

export function gemmNvfp4(a: number[][], b: number[][], block = B300_BLOCK): GemmResult {
  const m = a.length;
  const k = a[0].length;
  const n = b[0].length;
  const aq = quantizeRows(a, block);
  const bt = transpose(b);
  const bq = quantizeRows(bt, block);
  const bRecon = transpose(bq.recon);
  const bDelta = transpose(bq.delta);

  const c: number[][] = [];
  const fp: number[][] = [];
  const defect: number[][] = [];
  const bound: number[][] = [];
  let measured = 0;
  let boundMax = 0;
  let holds = true;

  for (let i = 0; i < m; i++) {
    const cr = new Array<number>(n);
    const fr = new Array<number>(n);
    const dr = new Array<number>(n);
    const br = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      let acc = 0;
      let ref = 0;
      let bd = 0;
      for (let t = 0; t < k; t++) {
        const av = a[i][t];
        const bv = b[t][j];
        const da = aq.delta[i][t];
        const db = bDelta[t][j];
        acc += aq.recon[i][t] * bRecon[t][j];
        ref += av * bv;
        bd += Math.abs(av) * Math.abs(db) + Math.abs(bv) * Math.abs(da) + Math.abs(da) * Math.abs(db);
      }
      const d = Math.abs(acc - ref);
      cr[j] = acc;
      fr[j] = ref;
      dr[j] = d;
      br[j] = bd;
      if (d > measured) measured = d;
      if (bd > boundMax) boundMax = bd;
      if (!within(d, bd)) holds = false;
    }
    c.push(cr);
    fp.push(fr);
    defect.push(dr);
    bound.push(br);
  }

  const a8 = a.map((row) => row.map(quantizeE4M3));
  const b8 = b.map((row) => row.map(quantizeE4M3));
  const c8 = gemmFp(a8, b8);
  let fp8MaxAbs = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      fp8MaxAbs = Math.max(fp8MaxAbs, Math.abs(c8[i][j] - fp[i][j]));
    }
  }

  return { c, fp, defect, bound, measured, boundMax, holds, macs: m * n * k, fp8MaxAbs };
}

export type ReduceResult = {
  sum: number;
  fp: number;
  measured: number;
  bound: number;
  holds: boolean;
  fp8Sum: number;
  fp8MaxAbs: number;
};

export function reduceNvfp4(values: number[], block = B300_BLOCK): ReduceResult {
  const { recon } = quantizeRows([values], block);
  const row = recon[0] ?? [];
  let sum = 0;
  let fp = 0;
  let bound = 0;
  let fp8 = 0;
  for (let i = 0; i < values.length; i++) {
    sum += row[i];
    fp += values[i];
    bound += Math.abs(row[i] - values[i]);
    fp8 += quantizeE4M3(values[i]);
  }
  const measured = Math.abs(sum - fp);
  return {
    sum,
    fp,
    measured,
    bound,
    holds: within(measured, bound),
    fp8Sum: fp8,
    fp8MaxAbs: Math.abs(fp8 - fp),
  };
}

export function softmaxRows(m: number[][]): number[][] {
  return m.map((row) => {
    let max = -Infinity;
    for (const v of row) if (v > max) max = v;
    const ex = row.map((v) => Math.exp(v - max));
    let s = 0;
    for (const v of ex) s += v;
    return ex.map((v) => (s === 0 ? 0 : v / s));
  });
}

export function maxAbs(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  }
  return m;
}
