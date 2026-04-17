"""Round 50.4: identify composition-specific SAE features.

R50.3 trained a K=100 TopK SAE on L24 residual contributions across
500 multi-step prompts (8680 samples). ~3400 live features out of
5120. This round figures out WHICH features are composition-specific.

Method:
  1. Capture L24 activations on ~500 matched-format SINGLE-step
     prompts (same prompt shape, just no +c).
     Format: "What is ({a} * {b})? Answer: "
     (vs multi-step:  "What is ({a} * {b}) + {c}? Answer: ")
  2. Load best K=100 SAE from R50.3.
  3. Encode multi-step (cached) + single-step activations with the SAE.
  4. For each of 5120 features, compute:
     - mean activation (when firing) on multi vs single
     - fire rate (how often it activates) on multi vs single
     - ratio: (mean_multi × fire_multi) / (mean_single × fire_single)
  5. Rank features by ratio. Top-N ratio features = composition-specific.
  6. Save top ~50 for R50.5/50.6.

Gate:
  At least 20 features have ratio ≥ 3× — enough for a meaningful
  composition-specific basis. If 0 features separate tasks, SAE's
  99% reconstruction is capturing shared signal, not composition.

Cost: ~500 new single-step captures (~8 min without daemon) + analysis.
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
N_PROMPTS = 500
D_MODEL = 2560
D_SAE_HIDDEN = 5120
SINGLE_CACHE = "/tmp/r50_captures_single.pt"
MULTI_CACHE = "/tmp/r50_captures.pt"
TOPK_SAE_CACHE = "/tmp/r50_3_topk.pt"


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def forward_and_capture_all_positions(m, token_ids):
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

    contribution = None
    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER:
                contribution = (h - h_before).detach()
    return contribution


def sample_single_step(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        prompt = f"What is ({a} * {b})? Answer: "
        out.append((prompt, a * b))
    return out


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

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        return self.dec(z), z


def load_sae(k: int = 100) -> TopKSAE:
    """Load the trained SAE from R50.3 for the given K."""
    if not os.path.exists(TOPK_SAE_CACHE):
        raise FileNotFoundError(
            f"Run R50.3 first — expected {TOPK_SAE_CACHE}")
    data = torch.load(TOPK_SAE_CACHE, weights_only=False)
    results = data["results"]
    matching = [r for r in results if r["k"] == k]
    if not matching:
        raise ValueError(
            f"No K={k} checkpoint in {TOPK_SAE_CACHE}. "
            f"Available: {[r['k'] for r in results]}")
    ckpt = matching[0]
    sae = TopKSAE(D_MODEL, D_SAE_HIDDEN, k)
    sae.load_state_dict(ckpt["state_dict"])
    return sae, data["X_mean"], data["X_std"]


def capture_activations(tok, m, prompts):
    samples = []
    for i, (prompt, _) in enumerate(prompts):
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        contrib = forward_and_capture_all_positions(m, token_ids)
        S = contrib.shape[1]
        if S > 1:
            samples.append(contrib[0, :S-1, :].cpu())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(prompts)}")
    return torch.cat(samples, dim=0).float()


def main():
    # --- Load or capture single-step activations ---
    if os.path.exists(SINGLE_CACHE):
        print(f"[r50.4] loading cached single-step captures...")
        X_single = torch.load(SINGLE_CACHE, weights_only=False)["X"]
        print(f"  cached {X_single.shape[0]} samples")
    else:
        from calm.llm_computer.gemma_substrate import (
            GemmaSubstrate, enable_triton_tq4,
        )
        from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

        enable_triton_tq4(True)
        print("[r50.4] loading substrate...")
        m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
        m.preload_gpu("cuda")
        m.warmup(seq_lens=(1, 20))
        tok = GemmaTokenizer.from_gguf(GGUF_PATH)

        rng = random.Random(42)
        prompts = sample_single_step(rng, N_PROMPTS)
        print(f"\n=== capturing L{TARGET_LAYER} SINGLE-step contributions "
              f"({N_PROMPTS} prompts) ===")
        X_single = capture_activations(tok, m, prompts)
        torch.save({"X": X_single, "n_prompts": N_PROMPTS}, SINGLE_CACHE)
        print(f"  saved: {SINGLE_CACHE}")

    # --- Load multi-step captures ---
    X_multi = torch.load(MULTI_CACHE, weights_only=False)["X"]
    print(f"\n  multi-step samples: {X_multi.shape[0]}")
    print(f"  single-step samples: {X_single.shape[0]}")

    # --- Load SAE ---
    print(f"\n[r50.4] loading K=100 SAE...")
    sae, X_mean, X_std = load_sae(k=100)
    device = "cuda"
    sae = sae.to(device).eval()
    X_mean = X_mean.to(device)
    # X_std was saved as a Python float, not a tensor
    if isinstance(X_std, (int, float)):
        X_std_t = torch.tensor(X_std, device=device)
    else:
        X_std_t = X_std.to(device)

    # --- Encode both corpora ---
    print(f"\n=== encoding activations with SAE ===")
    with torch.no_grad():
        X_multi_n = ((X_multi.to(device) - X_mean) / X_std_t)
        X_single_n = ((X_single.to(device) - X_mean) / X_std_t)
        z_multi = sae.encode(X_multi_n)   # (N_multi, 5120)
        z_single = sae.encode(X_single_n)  # (N_single, 5120)

    # --- Per-feature stats ---
    print(f"\n=== per-feature composition scores ===")
    # Fire rate: fraction of samples where feature > 0
    fire_multi = (z_multi > 0).float().mean(dim=0)    # (5120,)
    fire_single = (z_single > 0).float().mean(dim=0)  # (5120,)

    # Mean activation when firing
    sum_multi = z_multi.sum(dim=0)
    sum_single = z_single.sum(dim=0)
    n_multi = (z_multi > 0).sum(dim=0).clamp_min(1)
    n_single = (z_single > 0).sum(dim=0).clamp_min(1)
    mean_when_fire_multi = sum_multi / n_multi
    mean_when_fire_single = sum_single / n_single

    # Composition score = mean firing strength × fire rate (~ mean over all samples)
    strength_multi = mean_when_fire_multi * fire_multi  # expected activation
    strength_single = mean_when_fire_single * fire_single

    # Ratio (with smoothing to avoid div/zero)
    eps = 1e-3
    ratio = (strength_multi + eps) / (strength_single + eps)

    # Only consider features alive on multi-step (fire rate > 1%)
    live_mask = fire_multi > 0.01
    n_live = live_mask.sum().item()
    print(f"  live features (fire >1% on multi): {n_live}/{D_SAE_HIDDEN}")

    # Rank composition features by ratio (multi-specific)
    ratio_masked = torch.where(live_mask, ratio,
                                 torch.zeros_like(ratio))
    top_comp_idx = torch.argsort(ratio_masked, descending=True)[:50]

    print(f"\n=== TOP 20 composition-specific features (multi-step/single-step ratio) ===")
    print(f"  {'feat':>6} {'ratio':>8} {'mult_str':>10} {'single_str':>10} "
          f"{'mult_fire':>10} {'single_fire':>12}")
    for i, fidx in enumerate(top_comp_idx[:20]):
        f = fidx.item()
        print(f"  {f:>6} {ratio[f].item():>8.2f} "
              f"{strength_multi[f].item():>10.4f} "
              f"{strength_single[f].item():>10.4f} "
              f"{fire_multi[f].item()*100:>9.1f}% "
              f"{fire_single[f].item()*100:>11.1f}%")

    # How many features have ratio ≥ 3×?
    n_comp_3x = ((ratio_masked >= 3.0)).sum().item()
    n_comp_5x = ((ratio_masked >= 5.0)).sum().item()
    n_comp_10x = ((ratio_masked >= 10.0)).sum().item()
    print(f"\n  features with ratio ≥  3×: {n_comp_3x}")
    print(f"  features with ratio ≥  5×: {n_comp_5x}")
    print(f"  features with ratio ≥ 10×: {n_comp_10x}")

    # Gate
    print(f"\n========== R50.4 GATE ==========")
    gate = n_comp_3x >= 20
    print(f"  ≥ 20 features with ratio ≥ 3×: "
          f"{'PASS' if gate else 'FAIL'} ({n_comp_3x})")
    if gate:
        print(f"\n  ✓ Composition subspace identified. {n_comp_3x} features")
        print(f"    fire specifically on multi-step prompts. Next R50.5:")
        print(f"    ablate these features' contribution to L24 output,")
        print(f"    measure multi-step accuracy drop (should be task-specific).")
    else:
        print(f"\n  ~ Too few composition-specific features. Possible reasons:")
        print(f"    - K=100 SAE is too broad (try K=50)")
        print(f"    - Matched-format single-step prompts already activate")
        print(f"      the composition circuit (unlikely)")
        print(f"    - The composition signal is in ALL features at low level")

    # Save for R50.5/50.6
    torch.save({
        "top_comp_idx": top_comp_idx.cpu(),
        "ratio": ratio.cpu(),
        "fire_multi": fire_multi.cpu(),
        "fire_single": fire_single.cpu(),
        "strength_multi": strength_multi.cpu(),
        "strength_single": strength_single.cpu(),
        "n_comp_3x": n_comp_3x,
        "X_mean": X_mean.cpu(),
        "X_std": X_std,
    }, "/tmp/r50_4_comp_features.pt")
    print(f"\n  saved: /tmp/r50_4_comp_features.pt")


if __name__ == "__main__":
    sys.exit(main())
