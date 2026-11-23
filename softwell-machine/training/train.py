#!/usr/bin/env python3
"""Train a small real draft model for SmolLM-135M speculative decoding.

Llama-arch decoder (RMSNorm, RoPE, GQA, SwiGLU) matching the softwell
engine's config schema and HF Llama safetensors layout. Trains on
TinyStories, exports float32 safetensors + config.json + tokenizer.json
to models/draft-smollm/.

Usage:
  python3 train.py --smoke [seconds]     # throughput smoke test, random tokens
  python3 train.py --prep                 # tokenize TinyStories -> data/tokens.npy
  python3 train.py --train                # full training run (backgrounded)
  python3 train.py --export               # export checkpoint -> models/draft-smollm/
"""
import argparse, json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "training", "data")
OUT = os.path.join(ROOT, "models", "draft-smollm")

# Draft config: hidden 256, 6 layers, 8 heads, 4 KV heads, head_dim 32.
# lm_head (49152 x 256) dominates a small draft, so hidden stays small.
CFG = dict(
    vocab_size=49152, hidden_size=256, intermediate_size=768,
    num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=4,
    head_dim=32, max_position_embeddings=2048, rms_norm_eps=1e-5,
    rope_theta=10000.0, tie_word_embeddings=True,
)
SEQ_LEN = 256
BATCH = 4
LR = 3e-4
MAX_TOKENS = 100_000  # ~100k tokens, 1 epoch (proof-of-spec-decoding scale)


def build_model(cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RMSNorm(nn.Module):
        def __init__(self, d, eps):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

    def rope_freqs(seq, head_dim, theta, device):
        pos = torch.arange(seq, device=device, dtype=torch.float32)
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        ang = pos[:, None] * inv[None, :]
        return torch.cos(ang), torch.sin(ang)  # [T, D/2]

    def apply_rope(x, cos, sin):
        # x: [B, H, T, D]; cos/sin: [T, D/2]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c = cos[None, None, :, :]
        s = sin[None, None, :, :]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)

    class Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            d, h, kv, hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            self.h, self.kv, self.hd = h, kv, hd
            self.q = nn.Linear(d, h * hd, bias=False)
            self.k = nn.Linear(d, kv * hd, bias=False)
            self.v = nn.Linear(d, kv * hd, bias=False)
            self.o = nn.Linear(h * hd, d, bias=False)
        def forward(self, x, cos, sin, mask):
            B, T, _ = x.shape
            q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
            k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
            v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if self.kv != self.h:
                rep = self.h // self.kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
            return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.hd))

    class MLP(nn.Module):
        def __init__(self, c):
            super().__init__()
            d, m = c["hidden_size"], c["intermediate_size"]
            self.gate = nn.Linear(d, m, bias=False)
            self.up = nn.Linear(d, m, bias=False)
            self.down = nn.Linear(m, d, bias=False)
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ln1 = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
            self.attn = Attn(c)
            self.ln2 = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
            self.mlp = MLP(c)
        def forward(self, x, cos, sin, mask):
            x = x + self.attn(self.ln1(x), cos, sin, mask)
            return x + self.mlp(self.ln2(x))

    class Llama(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c
            self.embed = nn.Embedding(c["vocab_size"], c["hidden_size"])
            self.blocks = nn.ModuleList([Block(c) for _ in range(c["num_hidden_layers"])])
            self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
            self._freqs = {}
        def forward(self, ids):
            import torch
            B, T = ids.shape
            key = (T, str(ids.device))
            if key not in self._freqs:
                self._freqs[key] = rope_freqs(T, self.c["head_dim"], self.c["rope_theta"], ids.device)
            cos, sin = self._freqs[key]
            x = self.embed(ids)
            for b in self.blocks:
                x = b(x, cos, sin, None)  # causal via is_causal=True
            return self.norm(x)

    model = Llama(CFG)
    # HF-style init (std 0.02); default nn.Embedding std=1.0 blows logits up
    # (initial loss ~240 instead of ~ln(49152)=10.8) and wastes a short run.
    def _init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    model.apply(_init)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def step_loss(model, ids, labels):
    """Forward + chunked-CE loss (avoids materializing full [B,T,49152] logits)."""
    import torch
    import torch.nn.functional as F
    h = model(ids)  # [B, T, H]
    B, T, _ = h.shape
    loss = 0.0
    n = 0
    # predict token t+1 from position t; chunk over T to bound logits memory
    for s in range(0, T - 1, 64):
        e = min(s + 64, T - 1)
        logits = F.linear(h[:, s:e, :], model.embed.weight)  # tied lm_head
        loss = loss + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels[:, s + 1:e + 1].reshape(-1))
        n += 1
    return loss / n


def load_tokens():
    path = os.path.join(DATA, "tokens.npy")
    return np.load(path, mmap_mode="r")


def do_smoke(seconds):
    import torch
    torch.set_num_threads(4)
    model = build_model(CFG)
    print(f"draft params: {count_params(model)/1e6:.2f}M "
          f"(embed {CFG['vocab_size']*CFG['hidden_size']/1e6:.1f}M tied)")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ids = torch.randint(0, CFG["vocab_size"], (BATCH, SEQ_LEN))
    # warmup
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        step_loss(model, ids, ids).backward()
        opt.step()
    t0 = time.time()
    steps = 0
    toks = 0
    while time.time() - t0 < seconds:
        opt.zero_grad(set_to_none=True)
        loss = step_loss(model, ids, ids)
        loss.backward()
        opt.step()
        steps += 1
        toks += BATCH * SEQ_LEN
    dt = time.time() - t0
    tps = toks / dt
    print(f"SMOKE: {steps} steps in {dt:.1f}s -> {tps:.0f} tok/s "
          f"(batch {BATCH}x{SEQ_LEN})")
    need = MAX_TOKENS
    print(f"SMOKE ETA for {need/1e6:.1f}M tokens 1 epoch: {need/tps/60:.1f} min")
    print(f"SMOKE loss (random tokens): {loss.item():.3f}")
    return tps


def do_prep(max_bytes=10_000_000):
    from tokenizers import Tokenizer
    tok_path = os.path.join(DATA, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = os.path.join(ROOT, "models", "smollm-135m", "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    print("tokenizer vocab:", tok.get_vocab_size())
    src = os.path.join(DATA, "TinyStories-train.txt")
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_bytes)
    print(f"read {len(text)/1e6:.1f} MB of TinyStories")
    # Stories are separated by <|endoftext|>; encode story by story, append EOS(0).
    stories = text.split("<|endoftext|>")
    print(f"{len(stories)} stories in slice")
    ids = []
    eos = 0
    for s in stories:
        s = s.strip()
        if not s:
            continue
        ids.extend(tok.encode(s).ids)
        ids.append(eos)
        if len(ids) > MAX_TOKENS + SEQ_LEN:
            break
    arr = np.array(ids[:MAX_TOKENS + SEQ_LEN], dtype=np.int32)
    np.save(os.path.join(DATA, "tokens.npy"), arr)
    print(f"saved {len(arr)} tokens -> {os.path.join(DATA, 'tokens.npy')}")
    # quick decode sanity check
    print("decode sample:", repr(tok.decode(arr[:40].tolist()))[:300])


def do_train():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    toks = load_tokens()
    n_tok = min(len(toks) - SEQ_LEN, MAX_TOKENS)
    n_blocks = n_tok // SEQ_LEN
    print(f"training on {n_blocks*SEQ_LEN/1e6:.2f}M tokens "
          f"({n_blocks} blocks of {SEQ_LEN}), batch {BATCH}, lr {LR}")
    model = build_model(CFG)
    print(f"params: {count_params(model)/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = n_blocks // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    order = np.random.permutation(n_blocks)
    log_path = os.path.join(ROOT, "training", "train.log")
    ckpt = os.path.join(ROOT, "training", "draft.ckpt.pt")
    losses = []
    t0 = time.time()
    step = 0
    done_tok = 0
    for bi in range(0, n_blocks - BATCH + 1, BATCH):
        idx = order[bi:bi + BATCH]
        batch = np.stack([toks[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in idx])
        ids = torch.from_numpy(batch.astype(np.int64))
        opt.zero_grad(set_to_none=True)
        loss = step_loss(model, ids, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        done_tok += BATCH * SEQ_LEN
        losses.append(loss.item())
        if step % 25 == 0 or step == 1:
            el = time.time() - t0
            msg = (f"step {step}/{total_steps} loss {loss.item():.4f} "
                   f"avg100 {np.mean(losses[-100:]):.4f} "
                   f"{done_tok/el:.0f} tok/s elapsed {el/60:.1f}m "
                   f"eta {(total_steps-step)*(el/max(step,1))/60:.1f}m")
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
        if step % 200 == 0:
            torch.save({"state": model.state_dict(), "step": step,
                        "losses": losses}, ckpt)
    torch.save({"state": model.state_dict(), "step": step, "losses": losses}, ckpt)
    el = time.time() - t0
    summary = (f"TRAIN DONE: {done_tok/1e6:.2f}M tokens, {step} steps, "
               f"{el/60:.1f} min, final loss {losses[-1]:.4f}, "
               f"avg last100 {np.mean(losses[-100:]):.4f}, "
               f"avg first100 {np.mean(losses[:100]):.4f}")
    print(summary, flush=True)
    with open(log_path, "a") as f:
        f.write(summary + "\n")


def do_export():
    import torch
    from safetensors.torch import save_file
    ckpt = torch.load(os.path.join(ROOT, "training", "draft.ckpt.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["state"]
    print(f"exporting checkpoint step {ckpt['step']}, "
          f"losses first/last: {ckpt['losses'][0]:.3f} -> {ckpt['losses'][-1]:.3f}")
    out = {}
    out["model.embed_tokens.weight"] = sd["embed.weight"].float().contiguous()
    for l in range(CFG["num_hidden_layers"]):
        p = f"blocks.{l}"
        out[f"model.layers.{l}.self_attn.q_proj.weight"] = sd[f"{p}.attn.q.weight"].float().contiguous()
        out[f"model.layers.{l}.self_attn.k_proj.weight"] = sd[f"{p}.attn.k.weight"].float().contiguous()
        out[f"model.layers.{l}.self_attn.v_proj.weight"] = sd[f"{p}.attn.v.weight"].float().contiguous()
        out[f"model.layers.{l}.self_attn.o_proj.weight"] = sd[f"{p}.attn.o.weight"].float().contiguous()
        out[f"model.layers.{l}.mlp.gate_proj.weight"] = sd[f"{p}.mlp.gate.weight"].float().contiguous()
        out[f"model.layers.{l}.mlp.up_proj.weight"] = sd[f"{p}.mlp.up.weight"].float().contiguous()
        out[f"model.layers.{l}.mlp.down_proj.weight"] = sd[f"{p}.mlp.down.weight"].float().contiguous()
        out[f"model.layers.{l}.input_layernorm.weight"] = sd[f"{p}.ln1.w"].float().contiguous()
        out[f"model.layers.{l}.post_attention_layernorm.weight"] = sd[f"{p}.ln2.w"].float().contiguous()
    out["model.norm.weight"] = sd["norm.w"].float().contiguous()
    out["lm_head.weight"] = sd["embed.weight"].float().clone()  # tied copy (clone: safetensors rejects shared storage)
    os.makedirs(OUT, exist_ok=True)
    save_file(out, os.path.join(OUT, "model.safetensors"),
              metadata={"format": "pt"})
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": CFG["vocab_size"],
        "hidden_size": CFG["hidden_size"],
        "intermediate_size": CFG["intermediate_size"],
        "num_hidden_layers": CFG["num_hidden_layers"],
        "num_attention_heads": CFG["num_attention_heads"],
        "num_key_value_heads": CFG["num_key_value_heads"],
        "head_dim": CFG["head_dim"],
        "max_position_embeddings": CFG["max_position_embeddings"],
        "rms_norm_eps": CFG["rms_norm_eps"],
        "rope_theta": CFG["rope_theta"],
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32",
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    import shutil
    src_tok = os.path.join(DATA, "tokenizer.json")
    if not os.path.exists(src_tok):
        src_tok = os.path.join(ROOT, "models", "smollm-135m", "tokenizer.json")
    shutil.copy(src_tok, os.path.join(OUT, "tokenizer.json"))
    total = sum(t.numel() for t in out.values())
    print(f"exported {len(out)} tensors, {total/1e6:.1f}M params -> {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(" ", f, os.path.getsize(os.path.join(OUT, f)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=float, default=None)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.smoke is not None:
        do_smoke(args.smoke)
    elif args.prep:
        do_prep()
    elif args.train:
        do_train()
    elif args.export:
        do_export()
    else:
        ap.print_help()
