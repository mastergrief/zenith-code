"""Round 50.6: install SAE as L24 FFN replacement.

R50.3 trained a K=100 TopK SAE that reconstructs L24 residual
contributions with 99.6% variance from ~80 active features per sample.
R50.6 tests whether this reconstruction preserves Gemma's task
behavior.

Install: wrap L24's residual contribution at non-last positions with
  y_new = SAE.decode(SAE.topk_encode((y - mean) / std)) * std + mean
Last position passes through (R49.5 showed it's task-rank 1 anyway).

Gates:
  - Multi-step accuracy ≥ 80% of baseline: SAE preserves composition
  - Single-step accuracy ≥ 80% of baseline: SAE preserves single-step
    (both should hold — the SAE was trained on MULTI data but the 99%
    reconstruction should generalize)

If both pass: the SAE IS a faithful rank-100 compile target for L24.
If only multi passes: SAE learned multi-specific, not general.
If neither: the 1% lost variance carries critical task info.

This is the substrate-compilation step: replacing a full
transformer FFN+attn+proj layer's cooperative behavior with a
rank-100 linear map (encoder) + top-K threshold + rank-100 linear
map (decoder) = ~1M compiled params vs ~65M in the real L24.
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

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z), z


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


def forward_with_sae_install(m, token_ids, sae=None, mean=None, std=None):
    """Run Gemma's forward. At L24, replace the residual contribution
    at non-last positions with the SAE's reconstruction. Last position
    (which is task-rank 1, per R49.5) passes through.
    If sae is None, runs baseline."""
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
                # L24's contribution at all positions
                contribution = h - h_before  # (1, S, d_model)
                S_len = contribution.shape[1]
                if S_len > 1:
                    # Replace non-last positions with SAE reconstruction
                    contrib_nl = contribution[0, :S_len-1, :]
                    contrib_nl_n = (contrib_nl - mean) / std
                    contrib_reconstructed, _ = sae(contrib_nl_n)
                    contrib_nl_new = contrib_reconstructed * std + mean
                    h[0, :S_len-1, :] = h_before[0, :S_len-1, :] + contrib_nl_new
    return project_to_logits(m, h)


def build_multi(a, b, c):
    return f"What is ({a} * {b}) + {c}? Answer: "


def build_single(a, b):
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
        out.append((build_multi(a, b, c), a * b + c))
    return out


def sample_single(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        out.append((build_single(a, b), a * b))
    return out


def score_prompts(m, tok, sae, mean, std, prompts, label):
    print(f"\n=== {label} ({len(prompts)} prompts) ===")
    print(f"  {'prompt':>38}  {'expected':>8}  {'base':>8}  {'sae':>8}  base  sae")
    base_c = 0
    sae_c = 0
    for prompt, answer in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(answer)[0]
        base_logits = forward_with_sae_install(m, token_ids)
        base_argmax = int(base_logits[0, -1].argmax())
        base_tok = tok.id_to_token.get(base_argmax, '?')
        base_ok = base_tok.lstrip('▁') == correct_d
        if base_ok:
            base_c += 1

        sae_logits = forward_with_sae_install(
            m, token_ids, sae=sae, mean=mean, std=std)
        sae_argmax = int(sae_logits[0, -1].argmax())
        sae_tok = tok.id_to_token.get(sae_argmax, '?')
        sae_ok = sae_tok.lstrip('▁') == correct_d
        if sae_ok:
            sae_c += 1

        short = prompt[:36]
        b_mk = "✓" if base_ok else " "
        s_mk = "✓" if sae_ok else " "
        print(f"  {short!r:>38}  {str(answer):>8}  "
              f"{base_tok!r:>8}  {sae_tok!r:>8}  "
              f" {b_mk}    {s_mk}")
    return base_c, sae_c


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    # Load SAE
    if not os.path.exists("/tmp/r50_3_topk.pt"):
        print("ERROR: run R50.3 first")
        return 1
    sae_data = torch.load("/tmp/r50_3_topk.pt", weights_only=False)
    k_idx = [i for i, r in enumerate(sae_data["results"])
             if r["k"] == TOPK]
    if not k_idx:
        print(f"ERROR: no K={TOPK} SAE")
        return 1
    sae_ckpt = sae_data["results"][k_idx[0]]
    sae = TopKSAE(D_MODEL, D_SAE_HIDDEN, TOPK)
    sae.load_state_dict(sae_ckpt["state_dict"])
    print(f"[r50.6] K={TOPK} SAE: var_expl {sae_ckpt['var_expl']*100:.1f}%, "
          f"L0 {sae_ckpt['l0']:.1f}")

    device = "cuda"
    sae = sae.to(device).eval()
    mean = sae_data["X_mean"].to(device)
    std_val = sae_data["X_std"]
    if isinstance(std_val, (int, float)):
        std = torch.tensor(std_val, device=device)
    else:
        std = std_val.to(device)

    # Load Gemma
    enable_triton_tq4(True)
    print(f"\n[r50.6] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    rng = random.Random(999)
    multi_prompts = sample_multi(rng, N_HELD_OUT)
    single_prompts = sample_single(rng, N_HELD_OUT)

    multi_base, multi_sae = score_prompts(
        m, tok, sae, mean, std, multi_prompts, "MULTI-STEP")
    single_base, single_sae = score_prompts(
        m, tok, sae, mean, std, single_prompts, "SINGLE-STEP")

    multi_pres = multi_sae / max(multi_base, 1)
    single_pres = single_sae / max(single_base, 1)

    print(f"\n========== R50.6 RESULTS ==========")
    print(f"  MULTI-STEP baseline: {multi_base}/{N_HELD_OUT}")
    print(f"  MULTI-STEP SAE-inst: {multi_sae}/{N_HELD_OUT}  "
          f"(preserves {multi_pres*100:.0f}%)")
    print(f"  SINGLE-STEP base:    {single_base}/{N_HELD_OUT}")
    print(f"  SINGLE-STEP SAE:     {single_sae}/{N_HELD_OUT}  "
          f"(preserves {single_pres*100:.0f}%)")

    print(f"\n========== R50.6 GATE ==========")
    g_multi = multi_pres >= 0.80
    g_single = single_pres >= 0.80
    print(f"  multi preserves ≥ 80%:    "
          f"{'PASS' if g_multi else 'FAIL'} ({multi_pres*100:.0f}%)")
    print(f"  single preserves ≥ 80%:   "
          f"{'PASS' if g_single else 'FAIL'} ({single_pres*100:.0f}%)")

    if g_multi and g_single:
        print(f"\n  ✓ SAE is a FAITHFUL compile target for L24.")
        print(f"    99% reconstruction transfers to ≥80% task accuracy.")
        print(f"    Compile card: {D_MODEL}→{D_SAE_HIDDEN} encoder + topK + "
              f"{D_SAE_HIDDEN}→{D_MODEL} decoder")
        print(f"    ≈ {D_MODEL * D_SAE_HIDDEN * 2 / 1e6:.1f}M params vs "
              f"L24's full ~65M params. "
              f"{65 / (D_MODEL * D_SAE_HIDDEN * 2 / 1e6):.1f}x reduction.")
    elif g_multi and not g_single:
        print(f"\n  ~ Multi preserved, single degraded. SAE is task-specific;")
        print(f"    training included only multi-step.")
    elif not g_multi and g_single:
        print(f"\n  ~ Single preserved, multi lost. SAE is missing composition.")
    else:
        print(f"\n  ✗ SAE loses critical info. Either more features needed")
        print(f"    (K=200) or reconstruction threshold wasn't high enough.")

    torch.save({
        "multi_base": multi_base, "multi_sae": multi_sae,
        "single_base": single_base, "single_sae": single_sae,
    }, "/tmp/r50_6_sae_install.pt")


if __name__ == "__main__":
    sys.exit(main())
