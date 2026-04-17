"""Round 43: L23 H1/H4 forced attention on comparison + counting.

Extends R42 (validated L23 H1/H4 as position-selectors on SV) to
the other two capabilities that R36/R34 identified as using L23.
If both preserve under forced attention, the "4-for-1 compilation"
claim is fully validated:
  - Arithmetic (R17 shows L23 H1/H4 central; R28 validated via L30)
  - SV agreement (R42 validated)
  - Comparison (R36 shows L23 secondary; R37 shows diffuse at L23)
  - Counting (R34 shows L23 secondary)

Predicted outcomes:
  Comparison → likely preserves (H1/H4 appear top at L23 per R37)
  Counting → preserves if L23 contribution is position-based
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 23

COMPARISON_PROMPTS = []
random.seed(0)
for _ in range(20):
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    if a == b:
        continue
    COMPARISON_PROMPTS.append((
        f"Which is larger, {a} or {b}? Answer: ",
        max(a, b),
        "comparison",
    ))

COUNTING_PROMPTS = []
random.seed(0)
for _ in range(20):
    length = random.randint(4, 7)
    start = random.randint(1, 9)
    nums = list(range(start, start + length))
    nxt = start + length
    if nxt > 9:
        continue
    COUNTING_PROMPTS.append((
        "Count: " + ", ".join(str(x) for x in nums) + ", ",
        nxt,
        "counting",
    ))


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


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


def get_natural_top_positions(m, token_ids, target_layer, heads=(1, 4)):
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

        top_positions = {}
        for H in heads:
            q_h = q[0, H, -1, :]
            k_h = k[0, H, :, :]
            scores = (q_h.unsqueeze(0) @ k_h.T).squeeze(0)
            weights = F.softmax(scores, dim=-1)
            top_positions[H] = int(weights.argmax().item())
    return top_positions


class ForcedAttentionOutput:
    H1_SLICE = slice(1 * 512, 2 * 512)
    H4_SLICE = slice(4 * 512, 5 * 512)

    def __init__(self, inner, kv_cache, kv_src_layer, pos_for_h1, pos_for_h4):
        self.inner = inner
        self.kv_cache = kv_cache
        self.kv_src_layer = kv_src_layer
        self.pos_h1 = pos_for_h1
        self.pos_h4 = pos_for_h4
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        v_cache = self.kv_cache.v_cache[self.kv_src_layer]
        v_h1 = v_cache[0, 0, self.pos_h1, :].to(x.dtype).to(x.device)
        v_h4 = v_cache[0, 1, self.pos_h4, :].to(x.dtype).to(x.device)
        x_mod = x.clone()
        x_mod[0, -1, self.H1_SLICE] = v_h1
        x_mod[0, -1, self.H4_SLICE] = v_h4
        return self.inner(x_mod)


def forward_with_forced_attn(m, token_ids, pos_h1=None, pos_h4=None):
    from calm.llm_computer.gemma_substrate import KVCache
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
    saved = target.attn_output
    if pos_h1 is not None and pos_h4 is not None:
        target.attn_output = ForcedAttentionOutput(
            saved, cache, TARGET_LAYER, pos_h1, pos_h4)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_output = saved


def run_capability(m, tok, prompts, cap_name):
    print(f"\n=== {cap_name} ({len(prompts)} prompts) ===")
    print(f"  {'prompt':>42}  {'base':>8}  {'forced':>8}  {'match':>5}  {'Δ':>8}")

    matches = 0
    sum_abs_delta = 0.0
    n_clean = 0
    for prompt, expected, _ in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")

        logits_base = forward_with_forced_attn(m, token_ids)
        base_argmax = int(logits_base[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_logit = logits_base[0, -1, base_argmax].item()

        # Only test on prompts where Gemma is correct at baseline
        stripped = base_tok.lstrip('▁')
        if stripped != str(expected):
            continue
        n_clean += 1

        top_pos = get_natural_top_positions(m, token_ids, TARGET_LAYER)
        logits_forced = forward_with_forced_attn(
            m, token_ids, pos_h1=top_pos[1], pos_h4=top_pos[4])
        force_argmax = int(logits_forced[0, -1].argmax())
        force_tok = tok.id_to_token.get(force_argmax, '?')
        force_logit = logits_forced[0, -1, base_argmax].item()

        match = force_argmax == base_argmax
        delta = force_logit - base_logit
        sum_abs_delta += abs(delta)
        if match:
            matches += 1

        short = prompt[:40] if len(prompt) > 40 else prompt
        print(f"  {short!r:>42}  {base_tok!r:>8}  {force_tok!r:>8}  "
              f"{'Y' if match else 'N':>5}  {delta:>+8.2f}")

    if n_clean:
        print(f"\n  mean |Δ|: {sum_abs_delta/n_clean:.3f}")
        print(f"  matches:  {matches}/{n_clean}")
    return matches, n_clean, sum_abs_delta


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l23-forced-cross] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n[l23-forced-cross] Testing L23 H1/H4 forced attention on:")
    print(f"  1. Comparison ({len(COMPARISON_PROMPTS)} prompts)")
    print(f"  2. Counting    ({len(COUNTING_PROMPTS)} prompts)")

    m_cmp, n_cmp, d_cmp = run_capability(m, tok, COMPARISON_PROMPTS, "COMPARISON")
    m_cnt, n_cnt, d_cnt = run_capability(m, tok, COUNTING_PROMPTS, "COUNTING")

    print(f"\n\n=== SUMMARY ===")
    print(f"  R28 (arithmetic / L30): mean|Δ|=0.407, 9/10  ✓")
    print(f"  R42 (SV agree / L23):   mean|Δ|=0.467, 8/10  ✓")
    print(f"  R43 comparison / L23:   mean|Δ|={d_cmp/max(n_cmp,1):.3f}, {m_cmp}/{n_cmp}")
    print(f"  R43 counting / L23:     mean|Δ|={d_cnt/max(n_cnt,1):.3f}, {m_cnt}/{n_cnt}")

    pass_cmp = m_cmp / max(n_cmp, 1) >= 0.7 and d_cmp / max(n_cmp, 1) < 3.0
    pass_cnt = m_cnt / max(n_cnt, 1) >= 0.7 and d_cnt / max(n_cnt, 1) < 3.0

    if pass_cmp and pass_cnt:
        print(f"\n  ✓✓ 4-FOR-1 COMPILATION CONFIRMED.")
        print(f"     L23 H1/H4 compiled replacement would benefit:")
        print(f"     arithmetic (R28) + SV agreement (R42) + comparison + counting")
    elif pass_cmp or pass_cnt:
        which = "comparison" if pass_cmp else "counting"
        print(f"\n  ~ PARTIAL. {which} preserves but not the other. L23 hub")
        print(f"    benefits 3 of 4 tested capabilities. Still strong ROI.")
    else:
        print(f"\n  ✗ L23 forced attention doesn't generalize to numeric tasks")
        print(f"    beyond arithmetic + SV. Hub serves fewer than expected.")


if __name__ == "__main__":
    sys.exit(main())
