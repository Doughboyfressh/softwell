/**
 * Model layer — Hugging Face hub-compatible model references.
 *
 * A model reference is either a Hugging Face repo id ("owner/repo", e.g.
 * "meta-llama/Llama-3.3-70B-Instruct") or a local directory path.
 * `resolveModelRef` tells the two apart; `downloadModel` fetches a hub
 * repo's files over HTTPS (global fetch, Node built-ins only), streams
 * them to disk, and verifies sizes against the repo metadata when the
 * API reports them.
 *
 * Auth: if HF_TOKEN is set in the environment it is sent as a Bearer
 * token, which is required for gated repos.
 *
 * No network access happens at import time; the smoke test only exercises
 * URL construction and parsing.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const HF_BASE_URL = "https://huggingface.co";
export const HF_API_BASE_URL = "https://huggingface.co/api";

export type ModelRefKind = "hub" | "local";

export interface HubModelRef {
  kind: "hub";
  /** Full repo id, e.g. "meta-llama/Llama-3.3-70B-Instruct". */
  repo: string;
  owner: string;
  name: string;
}

export interface LocalModelRef {
  kind: "local";
  /** Absolute local directory path. */
  dir: string;
}

export type ModelRef = HubModelRef | LocalModelRef;

const REPO_ID_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

/**
 * Parse a "owner/repo" Hugging Face repo id. Throws on anything that is
 * not a bare repo id (local paths are not repo ids — use resolveModelRef
 * for those).
 */
export function parseRepoId(ref: string): HubModelRef {
  const trimmed = ref.trim();
  if (!REPO_ID_RE.test(trimmed)) {
    throw new Error(`not a valid Hugging Face repo id ("owner/repo"): ${ref}`);
  }
  const [owner, name] = trimmed.split("/") as [string, string];
  return { kind: "hub", repo: trimmed, owner, name };
}

/** True when `ref` looks like a local path rather than a hub repo id. */
export function isLocalPath(ref: string): boolean {
  const trimmed = ref.trim();
  if (
    trimmed.startsWith("/") ||
    trimmed.startsWith("./") ||
    trimmed.startsWith("../") ||
    trimmed.startsWith("~/") ||
    trimmed === "." ||
    trimmed === ".."
  ) {
    return true;
  }
  // An existing directory on disk is local even if it happens to look like
  // "owner/repo" (e.g. a relative "models/demo-10m" path).
  try {
    return fs.existsSync(trimmed) && fs.statSync(trimmed).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Resolve a model reference: a local directory path stays local, anything
 * shaped like "owner/repo" is a hub repo. Throws on anything else.
 */
export function resolveModelRef(ref: string): ModelRef {
  const trimmed = ref.trim();
  if (trimmed.length === 0) throw new Error("empty model reference");
  if (isLocalPath(trimmed)) {
    const expanded = trimmed.startsWith("~/")
      ? path.join(process.env["HOME"] ?? "~", trimmed.slice(2))
      : trimmed;
    return { kind: "local", dir: path.resolve(expanded) };
  }
  return parseRepoId(trimmed);
}

/** Download URL for one file in a hub repo (branch defaults to "main"). */
export function hubFileUrl(repo: string, file: string, revision = "main"): string {
  const { repo: repoId } = parseRepoId(repo);
  const encodedPath = file
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  return `${HF_BASE_URL}/${repoId}/resolve/${encodeURIComponent(revision)}/${encodedPath}`;
}

/** API URL for a repo's metadata (includes the `siblings` file list). */
export function hubRepoApiUrl(repo: string): string {
  const { repo: repoId } = parseRepoId(repo);
  return `${HF_API_BASE_URL}/models/${repoId}`;
}

/** API URL for a repo's file tree (recursive). */
export function hubRepoTreeApiUrl(repo: string, revision = "main"): string {
  const { repo: repoId } = parseRepoId(repo);
  return `${HF_API_BASE_URL}/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=true`;
}

function authHeaders(): Record<string, string> {
  const token = process.env["HF_TOKEN"];
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export interface HubSibling {
  rfilename: string;
  size?: number;
  sha?: string;
}

interface RepoApiResponse {
  siblings?: unknown;
}

function parseSiblings(value: unknown): HubSibling[] {
  if (!Array.isArray(value)) return [];
  const out: HubSibling[] = [];
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null) continue;
    const rec = entry as Record<string, unknown>;
    if (typeof rec["rfilename"] !== "string") continue;
    const sibling: HubSibling = { rfilename: rec["rfilename"] as string };
    if (typeof rec["size"] === "number") sibling.size = rec["size"] as number;
    if (typeof rec["lfs"] === "object" && rec["lfs"] !== null) {
      const lfs = rec["lfs"] as Record<string, unknown>;
      if (typeof lfs["sha256"] === "string") sibling.sha = lfs["sha256"] as string;
    }
    out.push(sibling);
  }
  return out;
}

/**
 * List the files in a hub repo (via the repo metadata endpoint, falling
 * back to the recursive tree endpoint). Returns rfilename + size when the
 * API reports it.
 */
export async function listRepoFiles(repo: string): Promise<HubSibling[]> {
  const headers = authHeaders();
  const res = await fetch(hubRepoApiUrl(repo), { headers });
  if (res.ok) {
    const json = (await res.json()) as RepoApiResponse;
    const siblings = parseSiblings(json.siblings);
    if (siblings.length > 0) return siblings;
  }
  // Fallback: tree endpoint.
  const treeRes = await fetch(hubRepoTreeApiUrl(repo), { headers });
  if (!treeRes.ok) {
    throw new Error(
      `failed to list files for ${repo}: repo API ${res.status}, tree API ${treeRes.status}`,
    );
  }
  const tree = (await treeRes.json()) as unknown;
  if (!Array.isArray(tree)) throw new Error(`unexpected tree response for ${repo}`);
  const out: HubSibling[] = [];
  for (const entry of tree) {
    if (typeof entry !== "object" || entry === null) continue;
    const rec = entry as Record<string, unknown>;
    if (rec["type"] !== "file" || typeof rec["path"] !== "string") continue;
    const sibling: HubSibling = { rfilename: rec["path"] as string };
    if (typeof rec["size"] === "number") sibling.size = rec["size"] as number;
    if (typeof rec["oid"] === "string") sibling.sha = rec["oid"] as string;
    out.push(sibling);
  }
  return out;
}

const MODEL_FILE_RE = /(^|\/)(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|model\.safetensors\.index\.json|model\.safetensors|.*\.safetensors)$/;

/** Keep only the files a softwell worker needs from a repo listing. */
export function filterModelFiles(siblings: HubSibling[]): HubSibling[] {
  return siblings.filter((s) => MODEL_FILE_RE.test(s.rfilename));
}

async function downloadOneFile(
  repo: string,
  file: HubSibling,
  destDir: string,
  revision = "main",
): Promise<string> {
  const url = hubFileUrl(repo, file.rfilename, revision);
  const res = await fetch(url, { headers: authHeaders(), redirect: "follow" });
  if (!res.ok) {
    throw new Error(`failed to download ${url}: HTTP ${res.status}`);
  }
  if (!res.body) throw new Error(`no response body for ${url}`);
  const destPath = path.join(destDir, file.rfilename);
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const writeStream = fs.createWriteStream(destPath);
  await pipeline(Readable.fromWeb(res.body as never), writeStream);
  if (typeof file.size === "number") {
    const actual = fs.statSync(destPath).size;
    if (actual !== file.size) {
      throw new Error(
        `size mismatch for ${file.rfilename}: expected ${file.size} bytes, got ${actual}`,
      );
    }
  }
  return destPath;
}

export interface DownloadModelOptions {
  revision?: string;
  /** When set, only these files are downloaded (otherwise the model file set). */
  files?: string[];
}

/**
 * Download a Hugging Face repo into `destDir` and return the local dir.
 * Downloads the model file set (config, tokenizer files, safetensors
 * index + shards); skips files already present at the expected size.
 * `ref` must be a hub repo id — local paths are returned as-is by
 * resolveModelRef and should not be passed here.
 */
export async function downloadModel(
  ref: string,
  destDir: string,
  options: DownloadModelOptions = {},
): Promise<string> {
  const { repo } = parseRepoId(ref);
  const absDest = path.resolve(destDir);
  fs.mkdirSync(absDest, { recursive: true });
  const revision = options.revision ?? "main";
  const siblings = await listRepoFiles(repo);
  let wanted = filterModelFiles(siblings);
  if (options.files) {
    const only = new Set(options.files);
    wanted = wanted.filter((s) => only.has(s.rfilename));
  }
  if (wanted.length === 0) {
    throw new Error(`no downloadable model files found for ${repo}`);
  }
  for (const file of wanted) {
    const destPath = path.join(absDest, file.rfilename);
    if (
      typeof file.size === "number" &&
      fs.existsSync(destPath) &&
      fs.statSync(destPath).size === file.size
    ) {
      continue; // already downloaded and verified by size
    }
    await downloadOneFile(repo, file, absDest, revision);
  }
  return absDest;
}
