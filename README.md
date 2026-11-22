# softwell

**Pi's software GPU. The program is the chip.**

softwell is a virtual computer for running trillion-parameter open-weight
LLMs at frontier speeds — and the web app that shows it off. The machine
is 100% virtual: a virtual GPU, virtual CPU, virtual SSD, and virtual
RAM pooled across donor machines, with an OpenAI-compatible inference
runtime and server on top. Physical machines only donate cycles and
memory underneath; workloads target the virtual hardware and nothing
else.

## softwell-machine — the virtual computer

`softwell-machine/` is the machine. It boots across donor nodes, pools
their resources into four virtual devices, and runs LLMs on them:

- **vGPU** — virtual devices, streams, kernel dispatch (CPU fallback)
- **vCPU** — task ISA with work stealing
- **vRAM** — paged, replicated, pooled memory
- **vSSD** — chunked, replicated storage with a tiered cache
- **Hypervisor** — boots the VM, heals donor loss by re-sharding

On top: a real transformer runtime (safetensors loader, sharding
planner, continuous batching, paged KV cache), novel algorithms that do
less work per token, and a virtual server that serves models over an
OpenAI-compatible API.

### Quickstart

```sh
cd softwell-machine
npm install
npm test                  # full suite, 69 tests, all must pass
npm run make-demo-model   # build the ~10M random-weight demo model
npm run demo              # full end-to-end demo (VM boot, serving, worker kill)
```

### Serve a model

```sh
node src/cli.ts serve demo/models/demo-100m   # boot VM + server + load model
node src/cli.ts up --model-dir ./my-model --workers 2
node src/cli.ts models                        # list loaded/available models
node src/cli.ts stop                          # graceful shutdown
```

Then:

```sh
curl -s http://127.0.0.1:8420/v1/models
curl -s -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo-100m","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

Dashboard: http://127.0.0.1:8420/ (cluster state, shard placement, VM pools,
server status, live requests).

Any Hugging Face safetensors model loads the same way
(`softwell serve meta-llama/Llama-3.3-70B-Instruct`); the server
fit-checks weights against pooled vRAM before staging anything.

### Benchmarks — the honest version

`npm run bench` measures the four virtual devices against the best
enterprise hardware (NVIDIA Rubin R100 / B200, AMD EPYC 9965, HBM3e,
Micron 9650). On one donor, every device FAILs by 2–6 orders of
magnitude — a VM cannot exceed its host's physics on raw FLOPS or
bytes, and the bench says so plainly. "Better" means pooled aggregate
throughput and capacity as donors join, not single-op latency. Full
results, targets, sources, and limits: [`softwell-machine/docs/BENCHMARKS.md`](softwell-machine/docs/BENCHMARKS.md).

### Novel algorithms — win by doing less work per token

Instead of adding silicon, the machine does fundamentally less work
per token. Theorems with proofs in
[`softwell-machine/docs/RESEARCH.md`](softwell-machine/docs/RESEARCH.md):

- **A1 landmark / block-mean attention** — up to **20.41×** prefill at
  n=4096, errors ~1,000× below the proven bound
- **A2 dynamic sparsity** — up to **10.46×** at n=2048
- **N1/N2 SW-code low-bit numbers** — 3.6–6.6× fewer bytes vs BF16
- **S1/S2 speculative and Jacobi decoding** — fewer forwards per token

Measured with `npm run bench:novel` (65 rows in
`softwell-machine/novel-bench-results-2026-11-22.json`). The honest
caveat: the byte-savers save bytes, not yet wall-clock time on this
compute-bound CPU backend — SW-code weights run 5–7× slower
in-process and speculative/Jacobi run 0.52–0.80×, because unpacking
and verification cost more than the bytes save here. The win needs
bandwidth-bound hardware to show up as time saved. Demo-100m decodes
at ~5 tok/s on the reference host.

### Docs

- [`softwell-machine/ARCHITECTURE.md`](softwell-machine/ARCHITECTURE.md) — design and what is real vs. what is not
- [`softwell-machine/docs/BENCHMARKS.md`](softwell-machine/docs/BENCHMARKS.md) — measured results vs enterprise hardware
- [`softwell-machine/docs/RESEARCH.md`](softwell-machine/docs/RESEARCH.md) — theorems, proofs, and measurements
- [`softwell-machine/docs/SERVER.md`](softwell-machine/docs/SERVER.md) — the virtual server
- [`softwell-machine/INTEGRATION.md`](softwell-machine/INTEGRATION.md) — how the pieces connect

## The web app

The root of this repo is the original softwell web app (Vite +
TanStack Start + React 19, Postgres/PGlite, Vercel) — the interactive
face of Pi's software GPU, where the CPU, RAM, and SSD share one
clock and the GPU waits on the same register.

```sh
npm install
npm run dev      # http://localhost:8080
npm run build
npm test
```
