"""Round 42: forced-attention at L23 H1/H4 on SV agreement.

Mirror of R28 which validated forced attention at L30 H4/H6 preserves
arithmetic. This tests the hub-sharing hypothesis: can the SAME kind
of forced one-hot position-selection at L23 preserve a DIFFERENT
capability (SV agreement)?

Mechanism: R40 showed L23 H4 attends to subject complex (~0.76 on
subject+preposition region), L23 H1 attends to distractor noun
(~0.50). For each SV prompt, identify the natural top-attended
position for each head (H1, H4 separately). Then replace the
head's attn_output input at last query position with V_cache[23]
at that top position — i.e., force one-hot attention at the
natural peak.

If SV argmax preserves (matches baseline) → L23 H1/H4 operate as
position-selectors at L23 in the same way L30 H4/H6 do at L30.
Hub-sharing mechanism confirmed cross-capability.

Note: L23 is a GLOBAL layer that OWNS its KV. Unlike R28 which
read V from L22 cache, R42 reads V from L23's own cache after the
forward pass reaches L23.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 23

PROMPTS = [
    ("The cat that sits near the window", "sing"),
    ("The cats that sit near the window", "plur"),
    ("The dog with the red collar", "sing"),
    ("The dogs with the red collar", "plur"),
    ("The teacher with the students", "sing"),
    ("The teachers with the student", "plur"),
    ("The key to the cabinets", "sing"),
    ("The keys to the cabinet", "plur"),
    ("The farmer beside the horses", "sing"),
    ("The farmers beside the horse", "plur"),
]

SING_VERB_TOKENS = {"▁is", "▁was", "▁has", "▁does", "▁seems", "▁sits", "▁sells"}
PLUR_VERB_TOKENS = {"▁are", "▁were", "▁have", "▁do", "▁seem", "▁sit"}


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class InputCapture:
    """Capture input to attn_q (post-attn-norm residual) for Q/K
    reconstruction."""
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
    """Capture L23 attention weights, return {head: top_attended_pos}."""
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
    """Force L23's H1 and H4 at last query position to attend 100% to
    specified positions via V_cache read. Other heads pass through
    unchanged.

    attn_output's input shape for global d_head=512: (1, S, 8*512=4096).
    H1 slice: [512:1024]. H4 slice: [2048:2560].
    L23 KV groups: H1 in group 0 (heads 0-3), H4 in group 1 (heads 4-7).
    """
    H1_SLICE = slice(1 * 512, 2 * 512)   # cols 512-1023
    H4_SLICE = slice(4 * 512, 5 * 512)   # cols 2048-2559

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
        # V cache shape: (B, n_heads_kv, S, d_head_kv)
        v_cache = self.kv_cache.v_cache[self.kv_src_layer]
        # H1 is in KV group 0 (heads 0-3), H4 in KV group 1 (heads 4-7)
        v_h1 = v_cache[0, 0, self.pos_h1, :].to(x.dtype).to(x.device)
        v_h4 = v_cache[0, 1, self.pos_h4, :].to(x.dtype).to(x.device)
        x_mod = x.clone()
        x_mod[0, -1, self.H1_SLICE] = v_h1
        x_mod[0, -1, self.H4_SLICE] = v_h4
        return self.inner(x_mod)


def forward_with_forced_attn(m, token_ids, pos_h1=None, pos_h4=None):
    """Forward with L23 H1 and H4 forced to specified positions. None
    means no intervention."""
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


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l23-forced-sv] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n=== R42: forced L23 H1/H4 at natural top positions on SV ===\n")
    print(f"  {'pair':>42}  {'base_tok':>10}  {'force_tok':>10}  {'pos_h1':>6}  {'pos_h4':>6}  {'match':>5}  {'Δlogit':>8}")

    matches = 0
    sum_abs_delta = 0.0
    for prompt, expected_num in PROMPTS:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")

        # Baseline
        logits_base = forward_with_forced_attn(m, token_ids)
        base_argmax = int(logits_base[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_logit = logits_base[0, -1, base_argmax].item()

        # Natural top positions
        top_pos = get_natural_top_positions(m, token_ids, TARGET_LAYER)
        pos_h1 = top_pos[1]
        pos_h4 = top_pos[4]

        # Forced
        logits_forced = forward_with_forced_attn(
            m, token_ids, pos_h1=pos_h1, pos_h4=pos_h4)
        force_argmax = int(logits_forced[0, -1].argmax())
        force_tok = tok.id_to_token.get(force_argmax, '?')
        force_logit = logits_forced[0, -1, base_argmax].item()

        match = force_argmax == base_argmax
        delta = force_logit - base_logit
        sum_abs_delta += abs(delta)
        if match:
            matches += 1

        print(f"  {prompt!r:>42}  {base_tok!r:>10}  {force_tok!r:>10}  "
              f"{pos_h1:>6}  {pos_h4:>6}  {'Y' if match else 'N':>5}  "
              f"{delta:>+8.2f}")

    n = len(PROMPTS)
    print(f"\n  mean |Δ|: {sum_abs_delta/n:.3f}")
    print(f"  argmax matches: {matches}/{n}")
    if matches >= 8 and sum_abs_delta/n < 3.0:
        print(f"\n  ✓ HUB-SHARING CONFIRMED. L23 H1/H4 work as position-")
        print(f"    selectors on SV agreement (same mechanism as R28")
        print(f"    validated for L30 H4/H6 on arithmetic). The L23")
        print(f"    hub is compilable as a Tier-2 target benefiting")
        print(f"    multiple capabilities simultaneously.")
    else:
        print(f"\n  ~ PARTIAL / NEGATIVE. L23 H1/H4 at natural top positions")
        print(f"    alone don't fully preserve SV. Softmax-over-multi-")
        print(f"    positions (not just argmax) carries meaningful info.")


if __name__ == "__main__":
    sys.exit(main())
