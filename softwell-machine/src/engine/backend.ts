import type { EngineBackend, IVirtualNpu } from "../contracts.ts";
import { ReferenceTransformerBackend } from "./transformer.ts";

/**
 * Factory for the reference TypeScript backend.
 * The mesh worker imports this; keep the exported name and signature.
 * An optional vNPU device is passed through to every stage executor
 * it creates; when none is given, an executor with --npu int8|fp8
 * builds its own local in-process device (see transformer.ts).
 */
export function createReferenceBackend(opts?: { npu?: IVirtualNpu }): EngineBackend {
  return new ReferenceTransformerBackend(opts ?? {});
}
