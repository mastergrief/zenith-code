"""Round 41: L23 H1/H4 attention pattern on ARITHMETIC prompts.

Closes the loop: R17 showed L23 H1/H4 are arithmetic's primary heads
(Δ = -4.85, -4.30). R21 showed L23 V encodes operand content. R40
showed the same heads on SV prompts attend to subject (H4) and
distractor (H1) respectively.

What do H1 and H4 at L23 attend to on arithmetic prompts? If the
"same heads, task-specific Q routing" hypothesis holds, they should
attend to operand-related positions. Predicted split: H4 → one
operand, H1 → the other.

Prompts: "17 times 23 equals " (10 tokens, positions 2/3 = a_tens/
a_ones, 6/7 = b_tens/b_ones).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 23
HEADS = [0, 1, 4]  # H0 control, H1, H4

PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class InputCapture:
    def __init__(self, inner):
        self.inner = inner
        self.captured = None
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        self.captured = x.detach().clone()
        return self.inner(x)


def get_head_weights(m, token_ids, target_layer, target_head):
    from calm.llm_computer.gemma_substrate import KVCache, _apply_rope
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

    target = m.layers[target_layer]
    cap = InputCapture(target.attn_q)
    target.attn_q = cap
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
                if i == target_layer:
                    break
    finally:
        target.attn_q = cap.inner

    target = m.layers[target_layer]
    with torch.no_grad():
        x_attn = cap.captured
        q_raw = cap.inner(x_attn)
        n_heads_q = cfg.n_heads_q
        d_head_q = q_raw.shape[-1] // n_heads_q
        q = q_raw.reshape(1, S, n_heads_q, d_head_q).transpose(1, 2)
        if target.attn_q_norm_w is not None:
            q = _rms_norm(q, target.attn_q_norm_w, cfg.rms_norm_eps)

        is_global = d_head_q > cfg.d_head
        freqs = m.rope_freqs_global if is_global else m.rope_freqs_swa
        q = _apply_rope(q, freqs[:S])

        kv_src = cfg.kv_source_layer(target_layer, is_swa=not is_global)
        if kv_src == target_layer:
            k_raw = target.attn_k(x_attn)
            n_heads_kv = cfg.n_heads_kv
            d_head_kv = k_raw.shape[-1] // n_heads_kv
            k_new = k_raw.reshape(1, S, n_heads_kv, d_head_kv).transpose(1, 2)
            if target.attn_k_norm_w is not None:
                k_new = _rms_norm(k_new, target.attn_k_norm_w, cfg.rms_norm_eps)
            k = _apply_rope(k_new, freqs[:S])
        else:
            k = cache.k_cache[kv_src].float()[..., :S, :]

        if cfg.n_heads_kv < cfg.n_heads_q:
            repeat = cfg.n_heads_q // cfg.n_heads_kv
            k = k.repeat_interleave(repeat, dim=1)

        q_h = q[0, target_head, -1, :]
        k_h = k[0, target_head, :, :]
        scores = (q_h.unsqueeze(0) @ k_h.T).squeeze(0)
        weights = F.softmax(scores, dim=-1)
    return weights.cpu()


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l23-arith-pattern] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n=== L{TARGET_LAYER} H1/H4 attention on ARITHMETIC prompts ===")
    # Tokens: [bos, ▁, a_tens, a_ones, ▁times, ▁, b_tens, b_ones, ▁equals, ▁]
    POS_A_TENS, POS_A_ONES = 2, 3
    POS_B_TENS, POS_B_ONES = 6, 7

    # Aggregate weights per position across all pairs
    agg = {h: torch.zeros(10) for h in HEADS}
    n_same_len = 0

    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids_list = tok.encode(prompt)
        if len(token_ids_list) != 10:
            continue
        token_ids = torch.tensor([token_ids_list], device="cuda")
        tokens = [tok.id_to_token.get(tid, f"?{tid}") for tid in token_ids_list]

        ws = {h: get_head_weights(m, token_ids, TARGET_LAYER, h) for h in HEADS}
        for h in HEADS:
            agg[h] += ws[h]
        n_same_len += 1

        print(f"\n{a}×{b}")
        print(f"  {'pos':>3} {'tok':>10}   {'H1':>7}  {'H4':>7}  {'H0':>7}  note")
        for i, t in enumerate(tokens):
            note = ""
            if i == POS_A_TENS:
                note = " ← a_tens"
            elif i == POS_A_ONES:
                note = " ← a_ones"
            elif i == POS_B_TENS:
                note = " ← b_tens"
            elif i == POS_B_ONES:
                note = " ← b_ones"
            print(f"  [{i:>2}] {t!r:>10}   {ws[1][i]:>7.3f}  {ws[4][i]:>7.3f}  {ws[0][i]:>7.3f}{note}")

    if n_same_len:
        print(f"\n\n=== average attention across {n_same_len} prompts ===")
        # Using last prompt's tokens for display
        avg_tok = [tok.id_to_token.get(tid, '?') for tid in tok.encode("10 times 10 equals ")]
        print(f"  {'pos':>3} {'tok':>10}   {'H1':>7}  {'H4':>7}  {'H0':>7}")
        for i in range(10):
            h1 = agg[1][i].item() / n_same_len
            h4 = agg[4][i].item() / n_same_len
            h0 = agg[0][i].item() / n_same_len
            note = ""
            if i == POS_A_TENS: note = " ← a_tens"
            elif i == POS_A_ONES: note = " ← a_ones"
            elif i == POS_B_TENS: note = " ← b_tens"
            elif i == POS_B_ONES: note = " ← b_ones"
            label = avg_tok[i] if i < len(avg_tok) else "?"
            print(f"  [{i:>2}] {label!r:>10}   {h1:>7.3f}  {h4:>7.3f}  {h0:>7.3f}{note}")


if __name__ == "__main__":
    sys.exit(main())
