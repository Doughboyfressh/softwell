/**
 * bench/run.ts — the `npm run bench` harness.
 *
 * Dynamically imports the four device bench modules
 * (bench-vgpu, bench-vcpu, bench-vram, bench-vssd), each exporting
 * `runBench(opts: BenchOptions): Promise<BenchResult[]>`, runs them
 * across the scaling curve 1..N donors, and prints the results via
 * printReport (src/bench/report.ts).
 *
 * Usage:
 *   node src/bench/run.ts [--donors N] [--json <path>] [--only vgpu|vcpu|vram|vssd]
 *
 *   --donors N   run the scaling curve 1..N donors (default 1)
 *   --json PATH  also write the raw results JSON to PATH
 *   --only DEV   run only one device bench (default: all four)
 *
 * Exit code: 0 even when verdicts are FAIL — a FAIL is a measurement,
 * not a crash. Non-zero only on a harness error (import fails, a
 * module throws, bad arguments).
 */
import * as os from "node:os";
import { printReport } from "./report.ts";
import type { BenchOptions, BenchResult } from "./types.ts";

type Device = "vgpu" | "vcpu" | "vram" | "vssd";

const DEVICES: Device[] = ["vgpu", "vcpu", "vram", "vssd"];

interface BenchModuleExports {
  runBench: (opts: BenchOptions) => Promise<BenchResult[]>;
}

/** Machine info, read from node:os at runtime — never hardcoded. */
export function describeMachine(): string {
  const cpus = os.cpus();
  const model = cpus.length > 0 ? (cpus[0]?.model ?? "unknown") : "unknown";
  const cores = cpus.length;
  const ramGiB = (os.totalmem() / 1024 ** 3).toFixed(1);
  return `Machine: ${model} (${cores} cores), ${ramGiB} GiB RAM, Node ${process.version}, ${os.platform()} ${os.arch()}`;
}

interface CliArgs {
  donors: number;
  jsonOut?: string;
  only?: Device;
}

function usage(): string {
  return [
    "Usage: node src/bench/run.ts [--donors N] [--json <path>] [--only vgpu|vcpu|vram|vssd]",
    "  --donors N   run the scaling curve 1..N donors (default 1)",
    "  --json PATH  also write the raw results JSON to PATH",
    "  --only DEV   run only one device bench (default: all four)",
  ].join("\n");
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = { donors: 1 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i] as string;
    const [flag, inlineValue] = arg.split("=", 2);
    const value = (name: string): string => {
      if (inlineValue !== undefined) return inlineValue;
      const next = argv[++i];
      if (next === undefined) throw new Error(`missing value for ${name}`);
      return next;
    };
    if (flag === "--help" || flag === "-h") {
      console.log(usage());
      process.exit(0);
    } else if (flag === "--donors") {
      const raw = value("--donors");
      const n = Number(raw);
      if (!Number.isInteger(n) || n < 1) {
        throw new Error(`--donors must be a positive integer, got ${raw}`);
      }
      args.donors = n;
    } else if (flag === "--json") {
      args.jsonOut = value("--json");
    } else if (flag === "--only") {
      const only = value("--only");
      if (!(DEVICES as string[]).includes(only)) {
        throw new Error(`--only must be one of ${DEVICES.join("|")}, got ${only}`);
      }
      args.only = only as Device;
    } else {
      throw new Error(`unknown argument: ${arg}\n${usage()}`);
    }
  }
  return args;
}

async function loadModule(device: Device): Promise<BenchModuleExports> {
  const mod = (await import(`./bench-${device}.ts`)) as Partial<BenchModuleExports>;
  if (typeof mod.runBench !== "function") {
    throw new Error(`bench module bench-${device}.ts does not export runBench(opts)`);
  }
  return mod as BenchModuleExports;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const selected = args.only ? [args.only] : DEVICES;

  console.log("softwell benchmark harness");
  console.log(describeMachine());
  console.log(`Donors: scaling curve 1..${args.donors} | devices: ${selected.join(", ")}`);
  console.log("");

  const modules = new Map<Device, BenchModuleExports>();
  for (const device of selected) {
    modules.set(device, await loadModule(device));
  }

  const results: BenchResult[] = [];
  for (let donors = 1; donors <= args.donors; donors++) {
    for (const device of selected) {
      const mod = modules.get(device) as BenchModuleExports;
      const rows = await mod.runBench({ donors });
      results.push(...rows);
    }
  }

  printReport(results, { jsonOut: args.jsonOut });

  // All measurement work is done. Bench modules may leave handles
  // behind (pooled worker threads, keep-alive sockets) that keep the
  // event loop alive long after the work finished; the harness must
  // still terminate. Flush stdout first so piped output is not lost,
  // then exit explicitly. A FAIL verdict is a measurement, not a
  // crash, so a completed run always exits 0.
  await new Promise<void>((resolve) => {
    process.stdout.write("", () => resolve());
  });
  process.exit(0);
}

main().catch((err: unknown) => {
  console.error(`bench harness error: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
