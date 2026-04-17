"""Round 40: L23 H1+H4 attention pattern on subject-verb prompts.

R39: L23 H1 and H4 hurt SV agreement strongly (-0.91, -1.05). R17:
same two heads are the arithmetic circuit (H1=-4.85, H4=-4.30).
Hypothesis: L23 H1/H4 are general content-carrier heads that read
from content positions regardless of task.

Test: do H1/H4 at L23 attend to the SUBJECT TOKEN in SV prompts?
The parallel to arithmetic: R26 showed L30 H6 attends to operand-
digit positions (a_ones at pos 3). If L23 H1/H4 attend to subject-
noun positions on SV prompts, the "general content carrier"
hypothesis is directly confirmed.

Cross-capability validation: same heads, same mechanism, different
downstream consumers.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 23
TARGET_HEADS = [1, 4]
CONTROL_HEAD = 0

# SV prompts chosen so the subject is a clearly-identified noun at
# a known position. Structure: "The <subject> <modifier>". Subject
# is the second token after "The".
PROMPTS = [
    ("The cat that sits near the window", "cat"),
    ("The cats that sit near the window", "cats"),
    ("The dog with the red collar", "dog"),
    ("The dogs with the red collar", "dogs"),
    ("The teacher with the students", "teacher"),
    ("The teachers with the student", "teachers"),
    ("The key to the cabinets", "key"),
    ("The keys to the cabinet", "keys"),
    ("The farmer beside the horses", "farmer"),
    ("The farmers beside the horse", "farmers"),
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
    print("[l23-sv-pattern] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n=== L{TARGET_LAYER} H1/H4 attention at last query position on SV prompts ===")

    for prompt, subject_word in PROMPTS:
        token_ids_list = tok.encode(prompt)
        token_ids = torch.tensor([token_ids_list], device="cuda")
        tokens = [tok.id_to_token.get(tid, f"?{tid}") for tid in token_ids_list]

        w_h1 = get_head_weights(m, token_ids, TARGET_LAYER, 1)
        w_h4 = get_head_weights(m, token_ids, TARGET_LAYER, 4)
        w_ctrl = get_head_weights(m, token_ids, TARGET_LAYER, CONTROL_HEAD)

        # Find the subject-token position
        subj_tok = "▁" + subject_word
        subj_pos = None
        for i, t in enumerate(tokens):
            if t == subj_tok:
                subj_pos = i
                break

        print(f"\n{prompt!r}  subject={subject_word!r} at pos={subj_pos}")
        print(f"  {'pos':>3}  {'token':>12}   {'H1':>7}  {'H4':>7}  {'H0':>7}  note")
        for i, t in enumerate(tokens):
            h1_v = w_h1[i].item()
            h4_v = w_h4[i].item()
            h0_v = w_ctrl[i].item()
            note = ""
            if i == subj_pos:
                note = " ← SUBJECT"
            mark_h1 = " ← H1" if h1_v >= 0.10 else ""
            mark_h4 = " ← H4" if h4_v >= 0.10 else ""
            print(f"  [{i:>2}] {t!r:>12}   {h1_v:>7.3f}  {h4_v:>7.3f}  {h0_v:>7.3f}{note}{mark_h1}{mark_h4}")


if __name__ == "__main__":
    sys.exit(main())
