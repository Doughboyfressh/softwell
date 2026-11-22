import { useEffect, useMemo, useState } from "react";
import { Play } from "lucide-react";
import { SAMPLES } from "@/lib/softwell/samples";
import { runPi, type ProgramResult } from "@/lib/softwell/pi";
import { PHASES, SMS_PER_DIE } from "@/lib/softwell/silicon";
import { BUS_BYTES, TMEM_BYTES } from "@/lib/softwell/chip";

function cn(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

function sci(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (Object.is(n, -0) || n === 0) return "0";
  const r = Math.round(n);
  if (Math.abs(n - r) < 1e-9 && Math.abs(n) < 1e12) return String(r);
  const a = Math.abs(n);
  if (a >= 1000 || a < 0.001) return n.toExponential(3);
  return n.toFixed(4);
}

function hex(n: number): string {
  return "0x" + n.toString(16);
}

function fixed(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const r = Math.round(n);
  if (Math.abs(n - r) < 1e-9 && Math.abs(n) < 1e12) return String(r);
  return n.toFixed(3);
}

const COMPARE = [
  ["SMs", "256", "160"],
  ["Tensor", "Pi block 8 · 5.0 bits", "NVFP4 block 16 · 4.5 bits"],
  ["Bus", "32 B/cycle", "HBM3e · 8 TB/s"],
  ["TMEM", "4 KB on the chip", "not this TMEM"],
  ["Clock", "cycle register", "not this clock"],
  ["ISA", "PTX + pi.mma 4×4", "SASS"],
  ["Power", "the program", "up to 1,400 W"],
  ["Chip", "this program", "a module"],
];

export function SoftwellApp() {
  const [sampleId, setSampleId] = useState(SAMPLES[0].id);
  const [drafts, setDrafts] = useState<Record<string, string>>(() =>
    Object.fromEntries(SAMPLES.map((s) => [s.id, s.source])),
  );
  const [result, setResult] = useState<ProgramResult | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  const source = drafts[sampleId] ?? "";

  function choose(id: string) {
    setSampleId(id);
  }

  function run() {
    const next = runPi(source);
    setResult(next);
    setFrame(0);
    setPlaying(next.ok && next.events.length > 0);
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        const next = runPi(source);
        setResult(next);
        setFrame(0);
        setPlaying(next.ok && next.events.length > 0);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [source]);

  useEffect(() => {
    if (!playing || !result?.ok || result.events.length === 0) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setFrame(result.events.length - 1);
      setPlaying(false);
      return;
    }
    const id = window.setInterval(() => {
      setFrame((f) => f + 1);
    }, 520);
    return () => window.clearInterval(id);
  }, [playing, result]);

  useEffect(() => {
    if (!result?.ok) return;
    if (result.events.length === 0) return;
    if (frame >= result.events.length - 1) setPlaying(false);
  }, [frame, result]);

  const event = result?.ok && result.events.length > 0 ? result.events[Math.min(frame, result.events.length - 1)] : null;

  const hot = useMemo(() => {
    const sets: [Set<number>, Set<number>] = [new Set(), new Set()];
    if (!event) return sets;
    for (const cell of event.cells) sets[cell.die].add(cell.sm);
    return sets;
  }, [event]);

  const allHold = result?.ok && result.proofs.length > 0 && result.proofs.every((p) => p.holds);

  return (
    <div className="min-h-screen bg-bg text-fg">
      <header className="sticky top-0 z-20 border-b border-line bg-bg">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3">
          <div className="min-w-0">
            <h1 className="text-xl font-semibold leading-none tracking-tight">Softwell</h1>
            <p className="mt-1 truncate font-mono text-xs text-muted">The software is the chip.</p>
          </div>
          <button
            type="button"
            onClick={run}
            className="inline-flex h-11 shrink-0 items-center gap-2 rounded-md bg-copper px-4 font-sans text-sm font-semibold text-ink"
          >
            <Play className="size-4" aria-hidden="true" />
            Run the chip
          </button>
        </div>
      </header>

      <main className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6">
        <section className="flex flex-col gap-4">
          <p className="max-w-3xl text-base leading-relaxed text-fg">
            The software is the chip. One gpu for is one kernel. %tid.x picks the pair, and each operand is
            written on the {BUS_BYTES}-byte bus once. A shared row stays in the {TMEM_BYTES / 1024} KB of TMEM.
            A thread owns a 4×4 tile. The tiles that fit on the SMs are one wave, so the pipe runs once. The block
            is 8, checked against FP32. A B300 is a different device — 160 SMs, NVFP4 in blocks of 16, 288 GB, up
            to 1,400 W. The same multiply is scored against that format. It is not what is running.
          </p>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="bg-surface">
                  <th className="px-3 py-2 font-mono text-xs font-medium text-faint"> </th>
                  <th className="px-3 py-2 font-mono text-xs font-medium text-copper">Pi GPU</th>
                  <th className="px-3 py-2 font-mono text-xs font-medium text-muted">B300</th>
                </tr>
              </thead>
              <tbody>
                {COMPARE.map(([k, pi, b300]) => (
                  <tr key={k} className="border-t border-line">
                    <th className="px-3 py-2 font-mono text-xs font-medium text-faint">{k}</th>
                    <td className="px-3 py-2 font-mono text-sm text-fg">{pi}</td>
                    <td className="px-3 py-2 font-mono text-sm text-muted">{b300}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Programs">
          {SAMPLES.map((s) => {
            const on = s.id === sampleId;
            return (
              <button
                key={s.id}
                type="button"
                role="tab"
                aria-selected={on}
                onClick={() => choose(s.id)}
                className={cn(
                  "min-h-11 rounded-md border px-3 py-2 text-left",
                  on ? "border-copper bg-surface text-fg" : "border-line bg-bg text-muted",
                )}
              >
                <span className="block text-sm font-semibold">{s.label}</span>
                <span className="block font-mono text-xs text-faint">{s.blurb}</span>
              </button>
            );
          })}
        </div>

        <div className="grid min-w-0 gap-4 lg:grid-cols-5">
          <section className="flex flex-col gap-2 lg:col-span-2">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="text-sm font-semibold">Pi source</h2>
              <span className="font-mono text-xs text-faint">skill, gpu for, pi_dot</span>
            </div>
            <label className="sr-only" htmlFor="pi-source">
              Pi source
            </label>
            <textarea
              id="pi-source"
              value={source}
              onChange={(e) => setDrafts((d) => ({ ...d, [sampleId]: e.target.value }))}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
              className="min-h-80 w-full flex-1 resize-y rounded-md border border-line bg-ink p-3 font-mono text-sm leading-relaxed text-fg"
            />
          </section>

          <section className="flex min-w-0 flex-col gap-3 rounded-md border border-line bg-surface p-3 lg:col-span-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-semibold">Pi GPU</h2>
              <p className="font-mono text-xs text-faint">
                clock {event ? event.cycle : result?.ok ? result.cycles : 0} · 256 SMs · {SMS_PER_DIE} on each half
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Die id={0} hot={hot[0]} />
              <Die id={1} hot={hot[1]} />
            </div>
            <div className="flex flex-wrap gap-2" aria-hidden="true">
              {PHASES.map((p) => (
                <span
                  key={p.id}
                  className={cn(
                    "rounded-sm border px-2 py-1 font-mono text-xs",
                    event?.phase === p.id ? "border-copper text-copper" : "border-line text-faint",
                  )}
                >
                  {p.label}
                </span>
              ))}
              <span className="self-center font-mono text-xs text-muted">
                {event ? `${event.cells.length} SMs in this phase` : "Idle — nothing scheduled"}
              </span>
            </div>
            <p className="min-h-10 font-mono text-xs leading-relaxed text-muted">
              {event?.caption ?? "Idle. The clock is at 0. The bus has not moved."}
            </p>
            <p className="font-mono text-xs leading-relaxed text-faint">
              bus {BUS_BYTES} B/cycle
              {event && event.bytes > 0 ? ` · ${event.bytes} B this phase` : ""}
              {event?.addr != null ? ` · last ${hex(event.addr)}` : ""}
              {result?.ok && result.busBeats > 0 ? ` · ${result.busBeats} beats` : ""}
            </p>
            {result?.ok && result.tmemMisses > 0 ? (
              <p className="font-mono text-xs leading-relaxed text-faint">
                tmem {TMEM_BYTES / 1024} KB · {result.tmemHits} hits · {result.tmemServed} B kept on the chip
              </p>
            ) : null}
            <div>
              <h3 className="font-mono text-xs text-faint">PTX</h3>
              <p className="mt-1 font-mono text-xs text-faint">
                The interpreter runs this listing. pi.mma.block8.f32 is the tile opcode. pi.dot.block8.f32 remains for one product.
              </p>
              <pre className="mt-1 max-h-40 min-w-0 max-w-full overflow-auto font-mono text-xs leading-relaxed text-muted">
                {result?.ok ? result.ptx : "// waiting"}
              </pre>
            </div>
          </section>
        </div>

        <section className="grid gap-4 lg:grid-cols-5">
          <div className="flex flex-col gap-3 rounded-md border border-line bg-surface p-4 lg:col-span-2">
            <h2 className="text-sm font-semibold">Block residual</h2>
            <p className="font-mono text-xs leading-relaxed text-muted">
              |c_ij − q_ij| ≤ Σ_k (|a_ik|·|δb_kj| + |b_kj|·|δa_ik| + |δa_ik|·|δb_kj|)
            </p>
            <p className="text-sm leading-relaxed text-muted">
              δ is what pi_dot records: the E2M1 code times its E4M3 scale, minus the original. A gpu for runs one thread
              per output. Scales cover 8 values, not the 16 a B300 uses. The inequality is the triangle bound on
              (a+δa)(b+δb) − ab. Both sides are computed. The bound is not assumed.
            </p>
            {result?.ok && result.proofs.length > 0 ? (
              <ul className="flex flex-col gap-2">
                {result.proofs.map((p) => (
                  <li key={p.name} className="border-t border-line pt-2">
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="font-mono text-xs text-fg">{p.name}</span>
                      <span className={cn("font-mono text-xs", p.holds ? "text-copper" : "text-danger")}>
                        {p.holds ? "holds" : "broke"}
                      </span>
                    </div>
                    <p className="mt-1 font-mono text-xs tabular-nums text-muted">
                      |Δ| {sci(p.measured)} · bound {sci(p.bound)} · slack {sci(p.bound - p.measured)}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="font-mono text-xs text-faint">No tensor word has been checked yet.</p>
            )}
            {result?.ok && result.proofs.length > 0 ? (
              <p className={cn("font-mono text-xs", allHold ? "text-copper" : "text-danger")}>
                {allHold ? "Every checked product is inside its bound." : "A product escaped its bound. The model is wrong."}
              </p>
            ) : null}
          </div>

          <div className="flex flex-col gap-4 lg:col-span-3">
            <Output result={result} />
          </div>
        </section>

        <footer className="border-t border-line pt-4 text-sm leading-relaxed text-muted">
          <p>
            The clock is the cycle register this run advances. The bus moves {BUS_BYTES} bytes per cycle. The
            listing is PTX; pi.dot.block8.f32 is this chip's opcode, and the interpreter runs that listing. A B300,
            for the score only, is published silicon: Blackwell Ultra, 160 SMs, 288 GB HBM3e at 8 TB/s, dense
            NVFP4 around 1.5× a B200, up to 1,400 W. This clock is not that module's clock, and this listing is
            not its SASS.
          </p>
          <p className="mt-2">
            Pi, the language, is{" "}
            <a
              className="text-copper underline decoration-line underline-offset-2"
              href="https://github.com/Doughboyfressh/CJ3"
              target="_blank"
              rel="noreferrer"
            >
              Doughboyfressh/CJ3
            </a>
            . The language’s GPU form is <span className="font-mono text-xs">{"gpu for x in xs -> v"}</span>, one
            thread per item. Tensors were the piece still missing from it.{" "}
            <span className="font-mono text-xs">pi_dot</span> is that piece, and this page is the GPU it runs on.
          </p>
        </footer>
      </main>
    </div>
  );
}

function Die({ id, hot }: { id: 0 | 1; hot: Set<number> }) {
  return (
    <div className="min-w-0 rounded-sm border border-line bg-ink p-2">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="font-mono text-xs text-copper">PI {id}</span>
        <span className="font-mono text-xs text-faint">{hot.size} lit</span>
      </div>
      <div className="grid grid-cols-8 gap-1">
        {Array.from({ length: SMS_PER_DIE }, (_, i) => {
          const on = hot.has(i);
          return (
            <div
              key={i}
              title={`SM ${i}${on ? " live" : ""}`}
              className={cn(
                "size-2.5 rounded-sm motion-reduce:transition-none sm:size-3 md:size-3.5",
                "transition-colors duration-200",
                on ? "bg-copper" : "bg-raised",
              )}
            />
          );
        })}
      </div>
    </div>
  );
}

function Output({ result }: { result: ProgramResult | null }) {
  if (!result) {
    return (
      <div className="rounded-md border border-line bg-surface p-4">
        <h2 className="text-sm font-semibold">Output</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Nothing committed. Run the open program. The die returns the Pi result. Beside it: how far that moved from
          FP32, and how far B300’s NVFP4 block of 16 moved on the same numbers.
        </p>
      </div>
    );
  }

  if (!result.ok) {
    return (
      <div className="rounded-md border border-line bg-surface p-4">
        <h2 className="text-sm font-semibold">Output</h2>
        <p className="mt-3 font-mono text-sm text-danger">
          Line {result.line}: {result.error}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 rounded-md border border-line bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold">Output</h2>
        <p className="font-mono text-xs tabular-nums text-faint">
          {result.macs} macs · clock {result.cycles} · bus {result.busBytes} B
        </p>
      </div>
      <p className="text-sm leading-relaxed text-muted">{result.note}</p>

      {result.b300Measured != null ? (
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-sm border border-line bg-line sm:grid-cols-3">
          <Stat k="Pi |Δ|" v={sci(result.proofs.reduce((m, p) => Math.max(m, p.measured), 0))} />
          <Stat k="B300 NVFP4 |Δ|" v={sci(result.b300Measured)} />
          <Stat k="Pi residual bound" v={sci(result.proofs.reduce((m, p) => Math.max(m, p.bound), 0))} />
        </dl>
      ) : null}
      {result.b300Measured != null ? <Beat pi={result.proofs.reduce((m, p) => Math.max(m, p.measured), 0)} b300={result.b300Measured} /> : null}

      {result.kind === "scalar" ? (
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="font-mono text-xs text-faint">{result.reference != null ? "Pi value" : "Returned"}</p>
            <p className="mt-1 font-mono text-2xl tabular-nums text-fg">{sci(result.scalar ?? Number.NaN)}</p>
          </div>
          {result.reference != null ? (
            <div>
              <p className="font-mono text-xs text-faint">FP32 value</p>
              <p className="mt-1 font-mono text-2xl tabular-nums text-muted">{sci(result.reference)}</p>
            </div>
          ) : null}
        </div>
      ) : null}

      {result.kind === "matrix" && result.matrix ? (
        <div className="grid gap-4 xl:grid-cols-2">
          <Matrix title="Pi result" m={result.matrix} emphasize={false} />
          {result.defect ? <Matrix title="Absolute defect vs reference" m={result.defect} emphasize /> : null}
        </div>
      ) : null}

      {result.kind === "text" ? (
        <pre className="overflow-auto font-mono text-sm leading-relaxed text-fg">{result.valueText}</pre>
      ) : null}

      {result.prints.length > 0 ? (
        <div>
          <h3 className="font-mono text-xs text-faint">print</h3>
          <pre className="mt-1 overflow-auto font-mono text-sm leading-relaxed text-fg">{result.prints.join("\n")}</pre>
        </div>
      ) : null}
    </div>
  );
}

function Beat({ pi, b300 }: { pi: number; b300: number }) {
  const tol = 1e-8 + 1e-6 * Math.max(1, Math.abs(b300));
  const lower = pi < b300 - tol;
  const higher = pi > b300 + tol;
  const text = lower
    ? "Pi |Δ| is under the B300 format on this run."
    : higher
      ? "This tile did not beat NVFP4 block 16."
      : "Same residual as NVFP4 block 16 on this run.";
  return <p className={cn("font-mono text-xs", higher ? "text-danger" : "text-copper")}>{text}</p>;
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-ink px-3 py-2">
      <dt className="font-mono text-xs text-faint">{k}</dt>
      <dd className="mt-1 font-mono text-sm tabular-nums text-fg">{v}</dd>
    </div>
  );
}

function Matrix({ title, m, emphasize }: { title: string; m: number[][]; emphasize: boolean }) {
  let peak = 0;
  if (emphasize) {
    for (const row of m) for (const v of row) peak = Math.max(peak, v);
  }
  return (
    <div className="min-w-0">
      <h3 className="font-mono text-xs text-faint">{title}</h3>
      <div className="mt-2 overflow-x-auto">
        <table className="border-collapse font-mono text-xs tabular-nums">
          <tbody>
            {m.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => {
                  const hot = emphasize && peak > 0 && v > peak * 0.65;
                  return (
                    <td key={j} className={cn("px-2 py-1 text-right", hot ? "text-copper" : "text-fg")}>
                      {fixed(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
