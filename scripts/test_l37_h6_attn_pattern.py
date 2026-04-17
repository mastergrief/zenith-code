"""Round 33: L37 H6 attention pattern on induction prompts.

R32 identified L37 H6 as the concentrated induction head. The classic
induction-head pattern (Olsson 2022): at position of current token X,
attend to position AFTER the previous occurrence of X. If prev X was
followed by Y, that's what gets copied.

Prompt: "A B C D A B C D A B" (10 tokens). Last token is "B" (at pos 9).
The answer the model should produce is "C". Induction head at pos 9
should attend to pos 6 (the previous "C" — which follows the previous
"B"). Or more precisely: at final "B" (pos 9), H6 should attend to
where "C" will be predicted from — i.e., position 2 (where first "C"
appeared, right after first "B").

Actually the canonical induction head: at a query position predicting
"C", attends to the PREVIOUS occurrence of the same context pattern.
Measuring this means looking at H6's attention from the last query
position to where in the sequence it's concentrated.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 37
TARGET_HEAD = 6
CONTROL_HEAD = 0


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
    """Reconstruct attention weights for target_head at target_layer's
    last query position, via Q/K capture + manual recompute with RoPE."""
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

        # SWA causal + window mask: last query (pos S-1) can attend to
        # positions max(0, S-1 - window_size) .. S-1. Window for Gemma
        # SWA = 512, our S ≤ 15, so no window constraint. Causal only.
        # At last query pos, all prior positions are visible.

        weights = F.softmax(scores, dim=-1)
    return weights.cpu()


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l37-h6-attn] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    random.seed(0)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Pattern: "A B C D A B C D A B" (10 tokens). Tokenized as
    # [bos, ▁A, ▁B, ▁C, ▁D, ▁A, ▁B, ▁C, ▁D, ▁A, ▁B] → 11 tokens.
    # After " A B C D A B C D A B", last pos = "▁B" (pos 10).
    # Model should predict "▁C" (the letter that followed B in the
    # first cycle). Classic induction: last-B's attention should
    # target the position AFTER the first "▁B" (which is first "▁C"
    # at pos 3). That position would then provide the C to copy.
    n_prompts = 5
    print(f"\n=== L{TARGET_LAYER} H{TARGET_HEAD} attention pattern ===")
    for _ in range(n_prompts):
        chosen = random.sample(letters, 4)
        seq = chosen + chosen + chosen[:2]
        prompt = " ".join(seq)
        token_ids_list = tok.encode(prompt)
        token_ids = torch.tensor([token_ids_list], device="cuda")
        tokens = [tok.id_to_token.get(tid, f"?{tid}") for tid in token_ids_list]

        w_h6 = get_head_weights(m, token_ids, TARGET_LAYER, TARGET_HEAD)
        w_ctrl = get_head_weights(m, token_ids, TARGET_LAYER, CONTROL_HEAD)

        print(f"\n{prompt} → expected='{chosen[2]}'  ({len(tokens)} tokens)")
        print(f"  {'pos':>3}  {'token':>8}   {'H6':>7}  {'H0':>7}  note")
        for i, t in enumerate(tokens):
            h6_v = w_h6[i].item()
            h0_v = w_ctrl[i].item()
            # Annotate which positions match "candidate induction targets"
            note = ""
            # We expect H6 to attend to the position AFTER first "▁B"
            # in the sequence. First occurrence of chosen[1] in tokens:
            # find first '▁' + chosen[1] match position, then i+1 is
            # "where C is" — the induction target.
            target_letter = "▁" + chosen[1]   # e.g. "▁B"
            answer_letter = "▁" + chosen[2]   # e.g. "▁C"
            if t == answer_letter:
                note += " (answer letter)"
            if t == target_letter:
                note += " (query-matching)"
            mark = ""
            if h6_v >= 0.15:
                mark = " ← H6"
            print(f"  [{i:>2}] {t!r:>8}   {h6_v:>7.3f}  {h0_v:>7.3f}{note}{mark}")


if __name__ == "__main__":
    sys.exit(main())
