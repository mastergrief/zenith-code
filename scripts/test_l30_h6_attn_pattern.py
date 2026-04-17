"""Round 26: L30 H6 attention pattern inspection.

R25 showed L30 H6 is the fd-gatherer (Δ=-1.528). Which positions
does it attend to? If it concentrates on a-digit or b-digit tokens
we can reconstruct its lookup structure as a LookUpExact gate.

Mechanism: capture the input to L30's attention (post-attn-norm of
the residual at L30 entry). Compute Q, K, apply RoPE, compute
attention weights for H6 at the last query position. This matches
what _forward_layer does internally.

For L30 (SWA, d_head=256, n_heads_q=8, GQA=8→2 KV groups):
  H6 is in KV group 1 (heads 4-7 share V columns 512-1023).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]
TARGET_LAYER = 30
TARGET_HEAD = 6


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class InputCapture:
    """Wraps attn_q to capture its INPUT (not output)."""
    def __init__(self, inner):
        self.inner = inner
        self.captured = None
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        self.captured = x.detach().clone()
        return self.inner(x)


def get_h6_weights(m, token_ids):
    """Run forward up to L30, capture its attention input, reconstruct
    H6's attention weights at the last query position. Matches the
    real computation in _forward_layer (NEOX RoPE, no /sqrt(d_head),
    Gemma 4 score scale=1.0)."""
    from calm.llm_computer.gemma_substrate import (KVCache, _apply_rope)
    cfg = m.config
    S = token_ids.shape[1]
    cache = KVCache(cfg.n_layers, device="cuda")

    h = m.token_embd[token_ids].to("cuda") * math.sqrt(cfg.d_model)
    m._per_layer_embd = None
    if m.per_layer_token_embd is not None:
        pl_embd = m.per_layer_token_embd[token_ids] * math.sqrt(cfg.d_per_layer)
        pl_embd = pl_embd.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
        if m.per_layer_model_proj is not None:
            h_proj = h @ m.per_layer_model_proj * (1.0 / math.sqrt(cfg.d_model))
            h_proj = h_proj.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
            if m.per_layer_proj_norm_w is not None:
                h_proj = _rms_norm(h_proj, m.per_layer_proj_norm_w, cfg.rms_norm_eps)
            pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
        m._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

    target = m.layers[TARGET_LAYER]
    cap = InputCapture(target.attn_q)
    target.attn_q = cap
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
                if i == TARGET_LAYER:
                    break
    finally:
        target.attn_q = cap.inner

    # Reconstruct H6 attention
    target = m.layers[TARGET_LAYER]
    with torch.no_grad():
        x_attn = cap.captured  # (1, S, d_model) — post-attn-norm input
        q_raw = cap.inner(x_attn)  # redo Q projection
        n_heads_q = cfg.n_heads_q
        d_head_q = q_raw.shape[-1] // n_heads_q
        q = q_raw.reshape(1, S, n_heads_q, d_head_q).transpose(1, 2)  # (1, H, S, D)
        if target.attn_q_norm_w is not None:
            q = _rms_norm(q, target.attn_q_norm_w, cfg.rms_norm_eps)

        is_global = d_head_q > cfg.d_head
        freqs = m.rope_freqs_global if is_global else m.rope_freqs_swa
        q = _apply_rope(q, freqs[:S])

        # K: for SWA L30, need own K (target.attn_k is the unwrapped one)
        # Check if L30 uses own KV or shared
        kv_src = cfg.kv_source_layer(TARGET_LAYER, is_swa=not is_global)
        if kv_src == TARGET_LAYER:
            k_raw = target.attn_k(x_attn)
            n_heads_kv = cfg.n_heads_kv
            d_head_kv = k_raw.shape[-1] // n_heads_kv
            k_new = k_raw.reshape(1, S, n_heads_kv, d_head_kv).transpose(1, 2)
            if target.attn_k_norm_w is not None:
                k_new = _rms_norm(k_new, target.attn_k_norm_w, cfg.rms_norm_eps)
            k = _apply_rope(k_new, freqs[:S])
        else:
            # Read cached K
            k = cache.k_cache[kv_src].float()[..., :S, :]

        # GQA expand
        if cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)

        # For H6, last query position
        q_h = q[0, TARGET_HEAD, -1, :]   # (d_head,)
        k_h = k[0, TARGET_HEAD, :, :]    # (S, d_head)
        scores = q_h.unsqueeze(0) @ k_h.T  # (1, S)
        scores = scores.squeeze(0)
        weights_h6 = F.softmax(scores, dim=-1)

        # Also compute for H0 (control) and H4 (R25 also weak-hurter)
        other_weights = {}
        for H in (0, 4):
            q_o = q[0, H, -1, :]
            k_o = k[0, H, :, :]
            s_o = (q_o.unsqueeze(0) @ k_o.T).squeeze(0)
            other_weights[H] = F.softmax(s_o, dim=-1).cpu()

    return weights_h6.cpu(), other_weights


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l30-h6-attn] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n=== L30 H6 attention pattern at last query position ===\n")

    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids_list = tok.encode(prompt)
        token_ids = torch.tensor([token_ids_list], device="cuda")
        tokens = [tok.id_to_token.get(tid, f"?{tid}") for tid in token_ids_list]

        w_h6, others = get_h6_weights(m, token_ids)

        # Show full distribution
        print(f"\n{a}×{b}={a*b}  ({len(tokens)} tokens)")
        print(f"  {'pos':>3}  {'token':>10}   {'H6':>7}  {'H4':>7}  {'H0':>7}")
        for i, t in enumerate(tokens):
            h6_v = w_h6[i].item()
            h4_v = others[4][i].item()
            h0_v = others[0][i].item()
            # Mark high attention
            mark = ""
            if h6_v >= 0.10:
                mark = " ← H6"
            print(f"  [{i:>2}] {t!r:>10}   {h6_v:>7.3f}  {h4_v:>7.3f}  {h0_v:>7.3f}{mark}")


if __name__ == "__main__":
    sys.exit(main())
