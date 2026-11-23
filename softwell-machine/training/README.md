# Draft model for SmolLM-135M speculative decoding

## What this is
A small real Llama-arch draft model sharing SmolLM-135M's tokenizer
(49152 vocab) and the engine's weight format, trained on TinyStories.
Proof-of-speculative-decoding scale per user direction: ~100k tokens,
1 epoch (NOT the originally planned 1M).

## Files
- `train.py` — trainer + exporter (prep / smoke / train / export)
- `verify-draft.ts` — engine load check (loadManifest + loadTokenizer)
- `data/` — tokenizer.json, TinyStories text, tokens.npy (100,256 tokens)
- `draft.ckpt.pt` — training checkpoint (state_dict + loss history)
- `train.log` — training log
- Export: `../models/draft-smollm/` (model.safetensors float32,
  config.json, tokenizer.json)

## Model
hidden 256, 6 layers, 8 heads, 4 KV heads, head_dim 32,
intermediate 768, vocab 49152, max_pos 2048, rope_theta 10000,
rms_norm_eps 1e-5, tie_word_embeddings true. 17.30M unique params
(29.9M on disk incl. tied lm_head copy).

## Training (2026-11-22)
- Torch 2.15.0+cpu, tokenizers 0.23.3, safetensors 0.8.0
  (`pip install --user --break-system-packages`; PEP 668 blocks plain pip)
- Smoke: 228 tok/s (batch 4x256) -> ETA 7.3 min @ 100k tokens
- Actual: 97 steps, 100,256 tokens, 11.9 min wall clock (~140 tok/s
  average under load from other workstreams)
- AdamW lr 3e-4, cosine decay, weight_decay 0.01, grad clip 1.0,
  seq 256, batch 4, chunked CE (64-token chunks, avoids full logits)
- Loss: 10.84 -> 6.25 final (step 25: 8.07, step 50: 6.62, step 75: 6.30);
  mean 7.35. Held-out TinyStories loss 6.21 (ppl 498), top-1 next-token
  accuracy 7.56% (chance 0.002%). Still descending steeply at the end.

## Quality caveat (read this)
100k tokens is far too little to converge: loss 6.25 vs ~3.5-4.5 a
1M-token run would plausibly reach, and greedy decoding collapses to
the "." attractor (40 dots). Sampled (temp 0.8) output is real English
words but garbled syntax. The model has learned real signal
(7.56% top-1 vs 0.002% chance) and the engine plumbing is proven, but
expect only single-digit/low-teens % acceptance against SmolLM-135M.
If acceptance is too low for a meaningful spec-decoding demo, the fix
is more tokens (1M), not a bigger draft.

## Engine verification (2026-11-22)
- `node training/verify-draft.ts models/draft-smollm`:
  loadManifest OK (57 tensors, float32), loadTokenizer OK,
  encode/decode round-trip OK.
- `softwell serve models/draft-smollm` (1 worker): healthz ok,
  /v1/models lists draft-smollm, /v1/completions generates.
  Greedy, prompt "Once upon a time there was a little boy named Ben.",
  40 tokens: "........................................" (identical to
  the PyTorch reference generation — export/engine are consistent;
  the dots are the undertrained model, not a format bug).
  Sampled temp 0.8, prompt "The cat sat on the", 30 tokens:
  " pictures became he the. lots went drink the the.\n\n have. an.
  want.One trees to to.. he.\n many was"
- Server stopped after verification; no stray processes left.
