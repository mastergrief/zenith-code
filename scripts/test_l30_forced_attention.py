"""Round 28: forced-attention intervention at L30 H4/H6.

R27 validated V at L22 at pos 3 perfectly encodes a, pos 7 perfectly
encodes b. R26 showed L30 H6 attends 0.61 to pos 3 (and 0.18 to pos 7),
H4 attends 0.42 to pos 7 (and 0.21 to pos 3). Our hypothesis: these
heads are just position selectors — H6 → read pos 3, H4 → read pos 7.

This round tests the hypothesis causally. At L30, override H6's
attention output at the last query position with V[L22, KV group 1,
pos=3] (100% pos 3), and H4's with V[L22, KV group 1, pos=7]. If fd
preserves on test pairs, the circuit interpretation is mechanistically
confirmed — we know we can replace L30 H4/H6 with compiled LookUpExact
gates at positions 3 and 7.

Measure: baseline fd-logit, forced-attention fd-logit, diff. If |diff|
is small (< 2.0 logit units) for all 10 pairs, interpretation holds.
If |diff| is large, attention is doing more than we modeled.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}
PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]
TARGET_LAYER = 30
KV_SOURCE_LAYER = 22  # L30 SWA reuses L22's V cache
H4_SLICE = slice(4 * 256, 5 * 256)   # H4 in attn_output's input: cols 1024-1279
H6_SLICE = slice(6 * 256, 7 * 256)   # H6: cols 1536-1791


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class ForcedAttentionOutput:
    """Wrap L30's attn_output to override H4+H6 slices at the last query
    position with V from L22 cache at specified positions.

    attn_output's input has shape (1, S, n_heads_q * d_head_q) = (1, S, 2048)
    for L30 SWA. We override slices corresponding to H4 and H6 at the last
    query position (S-1) with the forced V content from L22 cache.
    """
    def __init__(self, inner, kv_cache, kv_src_layer, pos_for_h4, pos_for_h6):
        self.inner = inner
        self.kv_cache = kv_cache
        self.kv_src_layer = kv_src_layer
        self.pos_for_h4 = pos_for_h4
        self.pos_for_h6 = pos_for_h6
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        # x: (1, S, 2048) — the attention output in attn_output's input format
        # Read L22's V cache: shape (B, n_heads_kv, S, d_head_kv) = (1, 2, S, 256)
        v_cache = self.kv_cache.v_cache[self.kv_src_layer]
        # KV group 1 (H4-H7 share): index 1 in n_heads_kv dim.
        group_idx = 1
        v_at_h4_pos = v_cache[0, group_idx, self.pos_for_h4, :].to(x.dtype).to(x.device)  # (256,)
        v_at_h6_pos = v_cache[0, group_idx, self.pos_for_h6, :].to(x.dtype).to(x.device)  # (256,)

        x_mod = x.clone()
        # Override last query position's H4 and H6 slices
        x_mod[0, -1, H4_SLICE] = v_at_h4_pos
        x_mod[0, -1, H6_SLICE] = v_at_h6_pos
        return self.inner(x_mod)


def forward_with_forced_attn(m, token_ids, force_h4_pos=None, force_h6_pos=None):
    """Run forward, optionally forcing L30 H4/H6 at last position to read
    from specified positions via V cache. force_*_pos=None means no
    intervention."""
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

    # Intervention at L30: wrap attn_output AFTER L22 has populated the cache
    # (so KV_SOURCE_LAYER's cache is available when L30 runs).
    target = m.layers[TARGET_LAYER]
    saved = target.attn_output
    if force_h4_pos is not None and force_h6_pos is not None:
        target.attn_output = ForcedAttentionOutput(
            saved, cache, KV_SOURCE_LAYER, force_h4_pos, force_h6_pos)

    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_output = saved


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l30-forced] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Tokens: [bos, ▁, a_tens, a_ones, ▁times, ▁, b_tens, b_ones, ▁equals, ▁]
    #         [ 0,  1,  2,      3,      4,       5, 6,      7,      8,        9]
    # H6 should read pos 3 (a_ones), H4 should read pos 7 (b_ones)
    POS_A_ONES = 3
    POS_B_ONES = 7

    print(f"\n=== baseline vs forced attention (H6→pos{POS_A_ONES}, H4→pos{POS_B_ONES}) ===\n")
    print(f"  {'pair':>8} {'fd':>3}  {'base_fd':>8}  {'forced_fd':>10}  "
          f"{'Δfd':>8}  {'base_argmax':>12}  {'forced_argmax':>14}")

    sum_abs_delta = 0.0
    matches = 0
    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        fd = str(a * b)[0]

        # Baseline
        logits_base = forward_with_forced_attn(m, token_ids)
        base_fd = logits_base[0, -1, DIGIT_IDS[fd]].item()
        base_argmax = int(logits_base[0, -1].argmax())
        base_argmax_tok = tok.id_to_token.get(base_argmax, '?')

        # Forced
        logits_forced = forward_with_forced_attn(m, token_ids,
                                                   force_h4_pos=POS_B_ONES,
                                                   force_h6_pos=POS_A_ONES)
        forced_fd = logits_forced[0, -1, DIGIT_IDS[fd]].item()
        forced_argmax = int(logits_forced[0, -1].argmax())
        forced_argmax_tok = tok.id_to_token.get(forced_argmax, '?')

        delta = forced_fd - base_fd
        sum_abs_delta += abs(delta)
        if forced_argmax == base_argmax:
            matches += 1
        print(f"  {a:>2}×{b:<2}={a*b:<4} '{fd}'  {base_fd:+8.2f}  {forced_fd:+10.2f}  "
              f"{delta:+8.2f}  {base_argmax_tok!r:>12}  {forced_argmax_tok!r:>14}")

    n = len(PAIRS)
    mean_abs_delta = sum_abs_delta / n
    print(f"\n  mean |Δfd| across {n} pairs: {mean_abs_delta:.3f}")
    print(f"  argmax matches baseline: {matches}/{n}")

    if mean_abs_delta < 2.0 and matches >= 8:
        print(f"\n  ✓ HYPOTHESIS CONFIRMED: forced attention ≈ learned attention.")
        print(f"    L30 H4+H6 can be replaced by position-selecting LookUpExact gates.")
    elif mean_abs_delta < 5.0:
        print(f"\n  ~ PARTIAL: some disagreement — attention is doing more than pure")
        print(f"    position selection but the hypothesis captures most of the signal.")
    else:
        print(f"\n  ✗ HYPOTHESIS WEAKENED: attention is doing significantly more than")
        print(f"    position-selection. Needs deeper investigation.")


if __name__ == "__main__":
    sys.exit(main())
