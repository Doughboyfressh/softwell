# Adding your computer as a softwell donor

Status (2026-11-22): **the code side is done, the live join is NOT done.**
A worker can now boot with no model directory at all and load its stage
from staged vRAM weights — proven in `test/donor-boot.test.ts` (2/2).
No real remote worker has joined yet. This procedure is **held until
you are at your computer**; do not treat the donor as working until a
remote worker has actually registered and served tokens.

Free only: everything below is free — Node.js, git, Tailscale personal
use, no paid hosting anywhere. Your computer needs no model download
and no disk space for weights: it donates CPU/RAM only, and weights
come from the orchestrator's pooled vRAM over the mesh.

## What changed in the code (so you know what you're running)

- `src/mesh/worker.ts`: `--model-dir` is now optional. With no model
  directory the worker logs
  `[worker] no model directory — booting without a spawn-time loader;
  stages load from vRAM on assignment`, registers, waits, and loads
  its first stage from the orchestrator's staged vRAM weights
  (`using vRAM weight loader …`, `loaded stage … via vRAM weights`).
  A directory that *is* given but cannot be opened still fails
  startup loudly, as before.
- `src/cli.ts`: `softwell up` with no `--model-dir` and no demo model
  on disk now boots its workers diskless instead of refusing to start.
  The demo model is still used as the spawn directory when it exists;
  defaults are otherwise unchanged.
- Activation-side work also landed in this tree (`src/server/registry.ts`
  `getTokenizer` reads the staged vSSD copy; `src/cli.ts`
  `buildFactory` takes the manifest from the registry's staged copy):
  its dedicated end-to-end proof is DONE —
  `test/reactivation-disk-removed.test.ts` (2026-11-23, 102 s):
  demo-100m loaded + activated, the model directory renamed away,
  re-activation succeeds, generation streams the same 8 tokens
  ("CCCCCCCC") as before the move, zero "using disk weight loader"
  lines. Activation never reads the model directory from disk.

## Before the join: blockers that must be cleared first

1. **The orchestrator's mesh port only listens on loopback.**
   `bootCluster` binds the mesh server to `127.0.0.1`
   (`src/cli.ts`). A remote worker cannot register until the
   follow-up binds it to the tailnet address (or all interfaces).
   Do not attempt the live join until that fix is in.
2. **This VM is not on your Tailscale network right now.**
   `tailscale status` reports `Not connected`. Joining needs your
   approval of the link `tailscale up` prints — that is a join-time
   step, done with you present.
3. **Inbound Tailscale to this VM is unverified.** Per the Tailscale
   connector doc, this VM's Tailscale client makes outgoing TCP
   connections and accepts none. The donor worker must open a TCP
   connection *to* the orchestrator, so before the donor is declared
   working the follow-up must demonstrate a real connection from
   the tailnet to the mesh port. If that fails, the join needs a
   different reachability path — to be decided from the test result,
   not guessed now.

## Join procedure (run when you are at your computer)

### 0. What you need on your computer

- Node.js 24 (check with `node --version`; anything below 22.18 will
  not run this repo — the repo requires Node >= 22.18).
- git.
- Tailscale (free personal use).
- A checkout of the softwell repo. Muse will give you the exact clone
  command at join time — it is the same repo this machine pushes to,
  so do not guess the URL from memory.
- No model files. Nothing to download beyond the repo itself.

### 1. Check your OS (only if you are unsure)

Run `node --version` and `git --version` in a terminal. If either is
missing, install it first: Node from nodejs.org (LTS 24), git from
git-scm.com, or your OS package manager. There are no OS-specific
softwell steps beyond installing those two tools and Tailscale.

### 2. Install Tailscale and join the same tailnet

- Install Tailscale from tailscale.com/download (pick your OS) and
  sign in with the same account as your other devices.
- Confirm it with `tailscale ip -4` — it prints your computer's
  tailnet IPv4 address (a `100.x` address). Write it down; the
  worker command in step 5 needs it.
- Keep Tailscale running while donating. If you stop it, the worker
  loses the orchestrator and the cluster treats it as a dead donor.

### 3. Get the code

```sh
git clone <clone-url — Muse supplies the exact command at join time>
cd softwell-machine
git pull
node --version   # must be >= 22.18 (24 expected)
```

No `npm install` is needed for the worker: it uses Node built-ins
only. (The repo has a `node_modules` with TypeScript for development;
the worker itself does not need it.)

### 4. Orchestrator side (Muse does this first, before step 5)

1. Muse runs `tailscale up` on this VM and you approve the device
   from the link it shows you. Muse confirms with
   `tailscale status` and notes this VM's tailnet IP.
2. Muse starts the cluster with a known mesh port and tells you two
   values: the orchestrator's tailnet IP and the mesh port, e.g.
   `100.x.y.z` and `18410`. (The mesh-port loopback blocker in
   "Before the join" must be fixed before this step.)
3. Muse loads and activates a model, so there is a stage for your
   worker to pick up. A worker that joins before any model is active
   just waits — that is normal, not an error.

### 5. Start the donor worker (your computer)

In the repo directory, run exactly this — note there is **no**
`--model-dir`:

```sh
node src/mesh/worker.ts \
  --orchestrator http://<orchestrator-tailnet-ip>:<mesh-port> \
  --host <your-tailnet-ip-from-step-2> \
  --port 0
```

Leave that terminal window open while donating. What success looks
like, in order:

1. `[worker] no model directory — booting without a spawn-time loader; stages load from vRAM on assignment`
2. `[worker] registered with http://<orchestrator-tailnet-ip>:<mesh-port> as worker-…`
3. Once a model is active: `using vRAM weight loader for model <id> (… tensors staged on the orchestrator)` and `loaded stage … via vRAM weights`.

If you see `using disk weight loader` instead, stop and tell Muse:
that means the worker fell back to disk, which a donor must never do.

### 6. Verify from the orchestrator side (Muse does this)

- `GET /v1/cluster` shows the new worker `healthy`.
- A generation streams tokens with the donor in the plan.
- Only then is the donor declared working. Registration alone is
  not proof — serving tokens is.

### 7. Stopping

- Press Ctrl-C in the worker terminal. The cluster re-plans over
  the remaining workers; in-flight sequences on your worker error
  and must be resubmitted (their KV cache is gone with the worker).
- Optional: `tailscale down` on your computer if you want to leave
  the tailnet entirely.

## If it does not work

- `node: command not found` / Node < 22.18 — install Node 24 (step 1).
- Worker prints `cannot open model dir …` — you passed a
  `--model-dir` (or `SOFTWELL_MODEL_DIR` is set in your environment).
  Donors pass no model directory; unset the variable and re-run
  without the flag.
- Worker cannot reach the orchestrator (connection refused / timeout)
  — check `tailscale status` on both sides, confirm both machines
  are on the same tailnet, and confirm the orchestrator is actually
  listening (blockers 1–3 above). A failed `ping` proves nothing:
  Tailscale here carries TCP only, so test with the worker itself,
  not ping.
- Worker registers but never loads a stage — no model is active on
  the orchestrator yet (step 4.3), or the model is not staged in
  vRAM. Wait for Muse to activate a model, then watch for the
  `using vRAM weight loader` line.
- The worker's log shows `failed to load assignment` repeatedly —
  copy the exact lines and send them to Muse; do not keep
  restarting it.
