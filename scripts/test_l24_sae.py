"""Round 50.1: Sparse Autoencoder on L24 residual contributions.

R49.5 showed L24's per-prompt composition info lives at non-last
positions (not at position -1). R50.1 tests whether that information
is sparsifiable — i.e., whether a small number of interpretable
feature directions can reconstruct L24's per-position contributions.

Minimum viable SAE (Anthropic recipe, scaled down for session):
  input: L24 residual contribution at non-last positions, dim 2560
  arch: Linear(2560 → 5120) + ReLU + Linear(5120 → 2560)
  loss: MSE(x̂, x) + λ * ||enc_act||_1
  train: Adam lr=1e-3, λ=5e-4, ~1000 steps

Gate:
  (a) Reconstruction explained variance ≥ 80% → SAE is learning
  (b) Mean L0 (active features per sample) < 300 → genuine sparsity
      (5120 features, so L0 < 300 = < 5% active, Anthropic typically
      gets 50-150 active out of 1M+; scaling down our target)

If both pass: circuit IS sparsifiable, feature analysis next.
If (a) fails: composition info doesn't fit a linear feature basis.
If (b) fails: no sparsity, SAE degenerated to dense coding.

Cost:
  ~500 prompt captures at ~16 non-last positions each ≈ 8000 samples
  ~15 min forwards + ~3 min SAE training.
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
L1_LAMBDA = 5e-4
LR = 1e-3
N_STEPS = 1500
BATCH_SIZE = 256


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def forward_and_capture_all_positions(m, token_ids):
    """Capture L24 residual contribution at ALL positions (B=1, S, d)."""
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
                # Contribution at all positions: (1, S, d)
                contribution = (h - h_before).detach()
    return contribution  # (1, S, d_model)


def sample_multi_step(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        c = rng.randint(1, 99)
        prompt = f"What is ({a} * {b}) + {c}? Answer: "
        out.append((prompt, a * b + c))
    return out


class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        # Normalize decoder columns to unit norm per Anthropic recipe.
        with torch.no_grad():
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)

    def encode(self, x):
        return F.relu(self.enc(x))

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r50.1] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    rng = random.Random(42)
    prompts = sample_multi_step(rng, N_PROMPTS)
    print(f"\n=== capturing L{TARGET_LAYER} contributions at ALL positions ===")
    print(f"  {N_PROMPTS} prompts, ~16 positions each ≈ 8000 samples")

    all_samples = []
    for i, (prompt, _) in enumerate(prompts):
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        contrib = forward_and_capture_all_positions(m, token_ids)
        # Keep only non-last positions (exclude position S-1)
        S = contrib.shape[1]
        if S > 1:
            non_last = contrib[0, :S-1, :].cpu()  # (S-1, d_model)
            all_samples.append(non_last)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_PROMPTS}")

    X = torch.cat(all_samples, dim=0).float()  # (N_samples, d_model)
    print(f"\n  total samples: {X.shape[0]}")
    print(f"  mean ||x||: {X.norm(dim=1).mean().item():.2f}")
    print(f"  std: {X.std().item():.4f}")

    # Normalize: subtract mean, unit variance (Anthropic trains on
    # normalized activations to stabilize).
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std()
    X_norm = (X - X_mean) / X_std

    # --- Train SAE ---
    device = "cuda"
    X_norm = X_norm.to(device)
    sae = SAE(D_MODEL, D_SAE_HIDDEN).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=LR)

    print(f"\n=== training SAE (d_hidden={D_SAE_HIDDEN}, λ={L1_LAMBDA}, "
          f"{N_STEPS} steps) ===")
    n_samples = X_norm.shape[0]

    for step in range(N_STEPS):
        idx = torch.randint(0, n_samples, (BATCH_SIZE,), device=device)
        x = X_norm[idx]
        x_hat, z = sae(x)
        mse = F.mse_loss(x_hat, x)
        l1 = z.abs().mean()  # L1 per-feature per-sample
        loss = mse + L1_LAMBDA * l1
        optim.zero_grad()
        loss.backward()
        # Renormalize decoder weights (Anthropic recipe)
        optim.step()
        with torch.no_grad():
            sae.dec.weight.data = F.normalize(sae.dec.weight.data, dim=0)

        if step % 100 == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=1).mean().item()
                var_explained = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"  step {step:>5}: mse={mse.item():.4f}  "
                      f"l1={l1.item():.4f}  L0={l0:.1f}  "
                      f"var_expl={var_explained.item():.3f}")

    # Final eval
    print(f"\n=== final evaluation ===")
    sae.eval()
    with torch.no_grad():
        # Full dataset metrics
        x_hat, z = sae(X_norm)
        mse_final = F.mse_loss(x_hat, X_norm).item()
        l0_final = (z > 0).float().sum(dim=1).mean().item()
        var_expl_final = 1 - (X_norm - x_hat).pow(2).sum() / X_norm.pow(2).sum()
        frac_dead = ((z > 0).any(dim=0) == False).float().mean().item()

    print(f"  final MSE:            {mse_final:.4f}")
    print(f"  final variance expl:  {var_expl_final.item():.3f}")
    print(f"  final mean L0:        {l0_final:.1f}  "
          f"(out of {D_SAE_HIDDEN} features, "
          f"{l0_final/D_SAE_HIDDEN*100:.1f}%)")
    print(f"  fraction dead features: {frac_dead*100:.1f}%")

    # Gate
    print(f"\n========== R50.1 GATE ==========")
    gate_var = var_expl_final.item() >= 0.80
    gate_l0 = l0_final < 300
    print(f"  var_explained ≥ 80%:   "
          f"{'PASS' if gate_var else 'FAIL'}  ({var_expl_final.item()*100:.1f}%)")
    print(f"  mean L0 < 300:         "
          f"{'PASS' if gate_l0 else 'FAIL'}  ({l0_final:.1f})")

    if gate_var and gate_l0:
        print(f"\n  ✓ SAE trained successfully. L24 composition info IS")
        print(f"    sparsifiable in a feature dictionary.")
        print(f"    Next R50.2: feature analysis — find multi-step-specific")
        print(f"    features by comparing activations on multi-step vs")
        print(f"    single-step prompts.")
    else:
        print(f"\n  ~ SAE didn't reach both gates.")
        if not gate_var:
            print(f"    Reconstruction quality too low — composition info")
            print(f"    may not lie in a linear feature basis, or more")
            print(f"    training / tuning needed.")
        if not gate_l0:
            print(f"    Sparsity insufficient — may need higher L1 λ or")
            print(f"    more training.")

    torch.save({
        "sae_state_dict": sae.state_dict(),
        "X_mean": X_mean.cpu(),
        "X_std": X_std.item(),
        "metrics": {
            "mse": mse_final, "var_explained": var_expl_final.item(),
            "l0": l0_final, "frac_dead": frac_dead,
        },
    }, "/tmp/r50_1_sae.pt")
    print(f"\n  saved: /tmp/r50_1_sae.pt")


if __name__ == "__main__":
    sys.exit(main())
