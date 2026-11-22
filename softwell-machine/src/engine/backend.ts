import type { EngineBackend } from "../contracts.ts";
import { ReferenceTransformerBackend } from "./transformer.ts";

/**
 * Factory for the reference TypeScript backend.
 * The mesh worker imports this; keep the exported name and signature.
 */
export function createReferenceBackend(): EngineBackend {
  return new ReferenceTransformerBackend();
}
