export type Sample = {
  id: string;
  label: string;
  blurb: string;
  source: string;
};

export const SAMPLES: Sample[] = [
  {
    id: "gemm",
    label: "Pi GEMM",
    blurb: "one pi.mma per 4×4 tile",
    source: `# Pi. gpu for is the GPU word. pi_mma is the tile word.
# A is large, then small. B is small, then large along K.
# One thread owns a 4 by 4 tile and issues it once.

skill col(m, j) -> Any {
  let r = []
  for row in m { let r = append(r, row[j]) }
  return r
}

skill row(i, seed) -> Any {
  let r = []
  for j in range(0, 16) {
    let v = 1.2 + (i % 3) * 0.4 + seed * 0.05
    if j >= 8 { let v = v * 0.03 }
    let r = append(r, v)
  }
  return r
}

skill mat(rows, seed) -> Any {
  let m = []
  for i in range(0, rows) { let m = append(m, row(i, seed)) }
  return m
}

skill brow(i, seed) -> Any {
  let r = []
  for j in range(0, 8) {
    let v = 1.1 + (j % 3) * 0.35 + seed * 0.04
    if i < 8 { let v = v * 0.03 }
    let r = append(r, v)
  }
  return r
}

skill bmat(seed) -> Any {
  let m = []
  for i in range(0, 16) { let m = append(m, brow(i, seed)) }
  return m
}

skill gemm(a, b) -> Any {
  let n = len(b[0])
  let tc = (n + 3) // 4
  let tr = (len(a) + 3) // 4
  gpu for t in range(0, tr * tc) -> tiles {
    let ti = t // tc
    let tj = t % tc
    let rows = slice(a, ti * 4, ti * 4 + 4)
    let cols = []
    let j = 0
    while j < 4 {
      let jj = tj * 4 + j
      if jj < n { let cols = append(cols, col(b, jj)) }
      let j = j + 1
    }
    let tiles = pi_mma(rows, cols)
  }
  let out = []
  let i = 0
  while i < len(a) {
    let row = []
    let j = 0
    while j < n {
      let tile = tiles[(i // 4) * tc + (j // 4)]
      let row = append(row, tile[i % 4][j % 4])
      let j = j + 1
    }
    let out = append(out, row)
    let i = i + 1
  }
  return out
}

skill main() -> Any {
  print("pi gemm committed")
  return gemm(mat(8, 3), bmat(7))
}
`,
  },
  {
    id: "attn",
    label: "Attention tile",
    blurb: "gpu for the dots, softmax stays Pi",
    source: `# Q is large then small. K is small then large.
# The dots are pi_dot. Softmax is ordinary Pi.

skill col(m, j) -> Any {
  let r = []
  for row in m { let r = append(r, row[j]) }
  return r
}

skill seq(n, seed, flip) -> Any {
  let r = []
  for j in range(0, n) {
    let v = 0.8 + (seed % 4) * 0.25
    if flip == 0 and j >= 8 { let v = v * 0.03 }
    if flip == 1 and j < 8 { let v = v * 0.03 }
    let r = append(r, v)
  }
  return r
}

skill mat(rows, cols, seed, flip) -> Any {
  let m = []
  for i in range(0, rows) { let m = append(m, seq(cols, seed + i, flip)) }
  return m
}

skill softmax(row) -> Any {
  let m = row[0]
  for x in row { if x > m { let m = x } }
  let ex = []
  let s = 0
  for x in row {
    let e = exp(x - m)
    let ex = append(ex, e)
    let s = s + e
  }
  let p = []
  for e in ex { let p = append(p, e / s) }
  return p
}

skill attn(q, k, v) -> Any {
  let scale = 1 / sqrt(len(q[0]))
  let probs = []
  for qrow in q {
    gpu for krow in k -> scores {
      let scores = pi_dot(qrow, krow) * scale
    }
    let probs = append(probs, softmax(scores))
  }
  let n = len(v[0])
  let out = []
  let i = 0
  while i < len(probs) {
    gpu for j in range(0, n) -> row {
      let row = pi_dot(probs[i], col(v, j))
    }
    let out = append(out, row)
    let i = i + 1
  }
  return out
}

skill main() -> Any {
  return attn(mat(4, 16, 1, 0), mat(4, 16, 2, 1), mat(4, 16, 3, 0))
}
`,
  },
  {
    id: "reduce",
    label: "Block reduce",
    blurb: "one gpu for, pi_dot against ones",
    source: `# A spike every 16 values. pi_dot blocks by 8.
# A block of 16 spends its scale on the spike.

skill ones(n) -> Any {
  let r = []
  for i in range(0, n) { let r = append(r, 1) }
  return r
}

skill main() -> Any {
  let v = []
  for i in range(0, 48) {
    let x = 0.05
    if i % 16 == 0 { let x = 6 }
    let v = append(v, x * 0.1)
  }
  print(len(v))
  gpu for t in [0] -> s {
    let s = pi_dot(v, ones(len(v)))
  }
  return s
}
`,
  },
  {
    id: "host",
    label: "Host-only Pi",
    blurb: "No gpu for, the chip stays idle",
    source: `# No gpu for. This is ordinary Pi.
# The chip is the program, and nothing is scheduled.

skill fact(n) -> Any {
  if n <= 1 { return 1 }
  return n * fact(n - 1)
}

skill main() -> Any {
  let total = 0
  for x in [1, 2, 3, 4, 5] {
    let total = total + fact(x)
  }
  print(total)
  return total
}
`,
  },
];
