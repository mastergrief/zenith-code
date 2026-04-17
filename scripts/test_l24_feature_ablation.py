"""Round 50.5: causal validation — ablate composition features.

R50.4 identified N composition-specific features at L24 (firing ≥ 3×
more on multi-step than matched single-step). R50.5 tests whether
those features are CAUSALLY responsible for Gemma's multi-step
computation.

Method: at inference, hook L24's residual contribution at non-last
positions. Subtract the decoder contribution of the composition
features from the raw output, then let forward continue.

  ablation_delta(x) = SAE.decoder(z_comp_only(x)) * std
    where z_comp_only = SAE.encoder(normed_x) with all non-composition
    features zeroed
  y_ablated = y - ablation_delta

Gate:
  - Multi-step argmax correct drops by ≥ 30%
  - Single-step argmax correct drops by ≤ 10% (task-specific)
  - Both are required: if both drop equally, the features are general
    not composition-specific.

Baseline (no intervention) established first for each prompt set.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 24
D_MODEL = 2560
D_SAE_HIDDEN = 5120
TOPK = 100
N_HELD_OUT = 30

DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_hidden, k):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        self.k = k

    def encode(self, x):
        z_pre = F.relu(self.enc(x))
        if self.k >= z_pre.shape[1]:
            return z_pre
        vals, idx = torch.topk(z_pre, self.k, dim=1)
        z = torch.zeros_like(z_pre)
        z.scatter_(1, idx, vals)
        return z


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


def forward_with_feature_ablation(m, token_ids, sae=None, comp_idx=None,
                                    mean=None, std=None):
    """Forward through Gemma. At L24, subtract composition features'
    contribution from the residual write at all non-last positions.
    If sae/comp_idx is None, runs baseline."""
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

    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER and sae is not None:
                # L24's contribution: h - h_before
                contribution = h - h_before  # (1, S, d_model)
                # Ablate composition features at all NON-last positions.
                S_len = contribution.shape[1]
                if S_len > 1:
                    # Non-last slice
                    contrib_nl = contribution[0, :S_len-1, :]
                    # Normalize
                    contrib_nl_n = (contrib_nl - mean) / std
                    z = sae.encode(contrib_nl_n)
                    # Zero out non-composition features
                    z_comp = torch.zeros_like(z)
                    z_comp[:, comp_idx] = z[:, comp_idx]
                    # Decode composition-only contribution
                    decoded_comp = sae.dec(z_comp) * std
                    # Subtract it from contribution at non-last positions
                    new_nl = contrib_nl - decoded_comp
                    h[0, :S_len-1, :] = h_before[0, :S_len-1, :] + new_nl
    return project_to_logits(m, h)


def build_multi_prompt(a, b, c):
    return f"What is ({a} * {b}) + {c}? Answer: "


def build_single_prompt(a, b):
    return f"What is ({a} * {b})? Answer: "


def sample_multi(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        c = rng.randint(1, 99)
        out.append((build_multi_prompt(a, b, c), a, b, c, a * b + c))
    return out


def sample_single(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        out.append((build_single_prompt(a, b), a, b, None, a * b))
    return out


def score_prompts(m, tok, sae, comp_idx, mean, std, prompts, label):
    n = len(prompts)
    base_correct = 0
    ablated_correct = 0
    print(f"\n=== {label} ({n} prompts) ===")
    print(f"  {'prompt':>38}  {'expected':>8}  {'base':>8}  {'ablated':>8}  base  abl")
    for prompt, _a, _b, _c, answer in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(answer)[0]
        base_logits = forward_with_feature_ablation(m, token_ids)
        base_argmax = int(base_logits[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_ok = base_tok.lstrip('▁') == correct_d
        if base_ok:
            base_correct += 1

        abl_logits = forward_with_feature_ablation(
            m, token_ids, sae=sae, comp_idx=comp_idx,
            mean=mean, std=std)
        abl_argmax = int(abl_logits[0, -1].argmax())
        abl_tok = tok.id_to_token.get(abl_argmax, '?')
        abl_ok = abl_tok.lstrip('▁') == correct_d
        if abl_ok:
            ablated_correct += 1

        short = prompt[:36]
        b_mk = "✓" if base_ok else " "
        a_mk = "✓" if abl_ok else " "
        print(f"  {short!r:>38}  {str(answer):>8}  "
              f"{base_tok!r:>8}  {abl_tok!r:>8}  "
              f" {b_mk}    {a_mk}")

    return base_correct, ablated_correct


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    # Load composition-features + SAE
    if not os.path.exists("/tmp/r50_4_comp_features.pt"):
        print("ERROR: run R50.4 first")
        return 1
    feat_data = torch.load("/tmp/r50_4_comp_features.pt",
                            weights_only=False)
    sae_data = torch.load("/tmp/r50_3_topk.pt", weights_only=False)

    k_idx = [i for i, r in enumerate(sae_data["results"])
             if r["k"] == TOPK]
    if not k_idx:
        print(f"ERROR: no K={TOPK} SAE")
        return 1
    sae_ckpt = sae_data["results"][k_idx[0]]
    sae = TopKSAE(D_MODEL, D_SAE_HIDDEN, TOPK)
    sae.load_state_dict(sae_ckpt["state_dict"])

    device = "cuda"
    sae = sae.to(device).eval()
    mean = feat_data["X_mean"].to(device)
    std_val = feat_data["X_std"]
    if isinstance(std_val, (int, float)):
        std = torch.tensor(std_val, device=device)
    else:
        std = std_val.to(device)

    # How many composition features to ablate?
    ratio = feat_data["ratio"]
    fire_multi = feat_data["fire_multi"]
    live_mask = fire_multi > 0.01
    ratio_masked = torch.where(live_mask, ratio, torch.zeros_like(ratio))
    # Top 50 composition features, or all with ratio ≥ 3×, whichever
    n_comp_3x = int(feat_data["n_comp_3x"])
    n_ablate = min(50, max(20, n_comp_3x))
    top_idx = torch.argsort(ratio_masked, descending=True)[:n_ablate]
    comp_idx = top_idx.to(device)
    print(f"[r50.5] ablating top-{n_ablate} composition features")
    print(f"  top 5 ratios: {[f'{ratio_masked[i].item():.2f}' for i in top_idx[:5]]}")

    # Load Gemma
    enable_triton_tq4(True)
    print(f"\n[r50.5] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Held-out prompts
    rng = random.Random(999)
    multi_prompts = sample_multi(rng, N_HELD_OUT)
    single_prompts = sample_single(rng, N_HELD_OUT)

    multi_base, multi_abl = score_prompts(
        m, tok, sae, comp_idx, mean, std, multi_prompts, "MULTI-STEP")
    single_base, single_abl = score_prompts(
        m, tok, sae, comp_idx, mean, std, single_prompts, "SINGLE-STEP")

    # Report
    multi_drop = (multi_base - multi_abl) / max(multi_base, 1)
    single_drop = (single_base - single_abl) / max(single_base, 1)

    print(f"\n========== R50.5 RESULTS ==========")
    print(f"  MULTI-STEP baseline:  {multi_base}/{N_HELD_OUT}")
    print(f"  MULTI-STEP ablated:   {multi_abl}/{N_HELD_OUT}   "
          f"({multi_drop*100:+.0f}% drop)")
    print(f"  SINGLE-STEP baseline: {single_base}/{N_HELD_OUT}")
    print(f"  SINGLE-STEP ablated:  {single_abl}/{N_HELD_OUT}   "
          f"({single_drop*100:+.0f}% drop)")

    print(f"\n========== R50.5 GATE ==========")
    gate_multi = multi_drop >= 0.30
    gate_single = single_drop <= 0.10
    print(f"  multi drop ≥ 30%:    "
          f"{'PASS' if gate_multi else 'FAIL'} ({multi_drop*100:.0f}%)")
    print(f"  single drop ≤ 10%:   "
          f"{'PASS' if gate_single else 'FAIL'} ({single_drop*100:.0f}%)")

    if gate_multi and gate_single:
        print(f"\n  ✓ Composition features CAUSALLY validated.")
        print(f"    Ablating {n_ablate} features selectively degrades multi-step")
        print(f"    without affecting single-step. R50.6 ready.")
    elif gate_multi:
        print(f"\n  ~ Multi drop OK but single-step also affected. Features")
        print(f"    are not purely composition-specific.")
    elif gate_single:
        print(f"\n  ~ Single-step unchanged (good) but multi-step also unaffected.")
        print(f"    Selected features aren't causally important.")
    else:
        print(f"\n  ✗ Ablation has no selective effect. Composition circuit")
        print(f"    may be elsewhere, or our features aren't task-aligned.")

    torch.save({
        "multi_base": multi_base, "multi_abl": multi_abl,
        "single_base": single_base, "single_abl": single_abl,
        "comp_idx": comp_idx.cpu(), "n_ablate": n_ablate,
    }, "/tmp/r50_5_ablation.pt")


if __name__ == "__main__":
    sys.exit(main())
