/**
 * softwell-machine — OpenAI-compatible request/response mapping.
 *
 * Pure functions, no I/O. Converts between the public Chat Completions
 * shapes (see src/contracts.ts) and the scheduler's GenerateRequest /
 * GeneratedToken shapes, and formats SSE chunks.
 */

import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  GeneratedToken,
  GenerateRequest,
  ModelInfo,
  SamplingParams,
  Tokenizer,
} from "../contracts.ts";

let requestCounter = 0;

/** Deterministic-enough request id for chat completions. */
export function newChatCompletionId(): string {
  requestCounter += 1;
  return `chatcmpl-${Date.now().toString(36)}-${requestCounter}`;
}

/**
 * Map an OpenAI chat request onto a scheduler GenerateRequest.
 *
 * Messages are concatenated as "system: ...\nuser: ...\nassistant:" style
 * prompt text, then tokenized. Sampling params fall back to sane defaults
 * (temperature 0.7, topP 0.9, topK 0, maxTokens 128).
 */
export function chatRequestToGenerate(
  req: ChatCompletionRequest,
  tokenizer: Tokenizer,
): GenerateRequest {
  const prompt = req.messages
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");
  const tokenIds = [tokenizer.bosTokenId, ...tokenizer.encode(prompt)];
  const sampling: SamplingParams = {
    temperature: req.temperature ?? 0.7,
    topP: req.top_p ?? 0.9,
    topK: req.top_k ?? 0,
    maxTokens: req.max_tokens ?? 128,
    stop: req.stop,
    seed: req.seed,
  };
  return {
    id: newChatCompletionId(),
    prompt,
    tokenIds,
    sampling,
    role: "mixed",
  };
}

/** Format one generated token as a streaming chat-completion chunk. */
export function formatChunk(
  id: string,
  model: string,
  created: number,
  token: GeneratedToken,
  index: number,
): ChatCompletionChunk {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index,
        delta: { role: "assistant", content: token.text },
        finish_reason: token.finishReason,
      },
    ],
  };
}

/** Final chunk of a stream: no delta content, just the finish reason. */
export function formatFinishChunk(
  id: string,
  model: string,
  created: number,
  finishReason: string | null,
  index: number,
): ChatCompletionChunk {
  return {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [
      {
        index,
        delta: {},
        finish_reason: finishReason,
      },
    ],
  };
}

/** Format a completed (non-streaming) chat completion response. */
export function formatFinalResponse(
  id: string,
  model: string,
  created: number,
  content: string,
  promptTokens: number,
  completionTokens: number,
  finishReason: string | null,
): ChatCompletionResponse {
  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

/** Format the /v1/models list response. */
export function formatModelsList(models: ModelInfo[]): {
  object: "list";
  data: ModelInfo[];
} {
  return { object: "list", data: models };
}

/** Format a single model entry (already in ModelInfo shape; validated copy). */
export function formatModel(model: ModelInfo): ModelInfo {
  return {
    id: model.id,
    object: "model",
    created: model.created,
    owned_by: model.owned_by,
  };
}

/** Encode one chunk as a single SSE event. */
export function sseEncode(chunk: object): string {
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/** The terminal SSE event for a stream. */
export function sseDone(): string {
  return "data: [DONE]\n\n";
}
