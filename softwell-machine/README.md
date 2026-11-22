# softwell-machine

A virtual computer for running trillion-parameter open-weight LLMs: a
virtual GPU, virtual CPU, virtual SSD, and virtual RAM pooled across
donor machines, with an OpenAI-compatible inference runtime on top.
The VM *is* the hardware platform — workloads target the virtual
hardware interfaces only; physical machines are just resource donors.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, the honest
hardware math, and what is real vs. what is still a limit. The machine
stays 100% virtual by design — no physical GPU backend. It wins by
doing less work per token: novel algorithms with proofs in
[docs/RESEARCH.md](docs/RESEARCH.md) (landmark attention up to 20.41×
at n=4096, sparse attention up to 10.46× at n=2048).

## Quickstart

```sh
npm install
npm test                  # full suite, 69 tests, all must pass
npm run bench             # benchmark the virtual devices vs enterprise hardware
npm run bench:novel       # benchmark the novel algorithms (attention, SW-code, decoding)
npm run make-demo-model   # build the ~10M random-weight demo model
npm run demo              # full end-to-end demo (VM boot, serving, worker kill)
```

## Benchmarks

`npm run bench` measures the four virtual devices (vGPU, vCPU, vRAM,
vSSD) against the best enterprise hardware — NVIDIA Rubin R100 / B200,
AMD EPYC 9965, HBM3e, Micron 9650 — and prints a results table with a
PASS/FAIL verdict per metric. Verdicts are computed from real
measurements taken on the machine that ran the bench, never asserted:
a FAIL is a measurement, not a crash, so the harness still exits 0.

```sh
npm run bench                    # all devices, 1 donor
npm run bench -- --donors 4      # scaling curve 1..4 donors
npm run bench -- --only vgpu     # one device only
npm run bench -- --json results.json
```

The claim is aggregate throughput/capacity by pooling donors, not
single-op latency. Full results, targets, sources, and honest limits:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Serve a model

The virtual server boots as a workload of the VM itself — it reserves
vRAM and a vGPU stream from the hypervisor and serves models over an
OpenAI-compatible API. Full details: [docs/SERVER.md](docs/SERVER.md).

```sh
node src/cli.ts serve demo/models/demo-100m   # boot VM + server + load model
node src/cli.ts serve meta-llama/Llama-3.3-70B-Instruct --workers 4   # from the HF hub
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

## Layout

- `src/contracts.ts` — the platform interfaces (virtual hardware + runtime)
- `src/hw/` — vRAM, vGPU (+ CPU kernels), vCPU, vSSD, hypervisor
- `src/model/`, `src/planner/`, `src/mesh/`, `src/scheduler/`, `src/engine/` — the LLM runtime
- `src/server/` — the virtual server (model registry, serving lifecycle)
- `src/api/`, `src/cli.ts`, `src/dashboard/` — OpenAI-compatible API, CLI, UI
- `demo/` — demo model generator + end-to-end demo
- `INTEGRATION.md` — how the pieces connect (for contributors)
