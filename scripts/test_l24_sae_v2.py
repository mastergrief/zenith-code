"""Round 50.2: retuned SAE — higher λ, warmup, grad clip, checkpointing.

R50.1 trained a working SAE (99% var_explained in 100 steps) but
failed sparsity (L0 ~ 2000 out of 5120, 40% active). Fixes here:

  1. λ bumped 10× to 5e-3
  2. Warmup: λ ramps 0 → 5e-3 over first 200 steps (reconstruction
     first, sparsity second, avoids early collapse)
  3. Gradient clipping at max_norm=1.0 (prevents step-1499-style
     explosions)
  4. Best-L0 checkpointing: save when L0 first drops below 300 AND
     var_expl still ≥ 70%. Fall back to best var_expl * L0_penalty
     product.
  5. Caches captured activations to disk (/tmp/r50_captures.pt) so
     subsequent iterations skip Gemma re-load.

Gate:
  var_expl ≥ 70% (slightly relaxed vs R50.1's 80% — sparsity tradeoff)
  L0 < 300 (out of 5120 features, < 6% active)

If both pass: composition info IS sparsifiable, R50.3 starts feature
analysis (which features fire multi-step specifically).
If only var_expl passes: λ still too low, try higher.
If neither: L24 activations may not be sparsifiable, deeper
methodology change needed.
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
L1_TARGET_LAMBDA = 5e-3       # 10× R50.1
WARMUP_STEPS = 200
LR = 1e-3
N_STEPS = 2000
BATCH_SIZE = 256
GRAD_CLIP = 1.0
CAPTURES_PATH = "/tmp/r50_captures.pt"


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
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
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
    # --- Load or capture activations ---
    if os.path.exists(CAPTURES_PATH):
        print(f"[r50.2] loading cached captures from {CAPTURES_PATH}...")
        data = torch.load(CAPTURES_PATH, weights_only=False)
        X = data["X"]
        print(f"  cached {X.shape[0]} samples, dim {X.shape[1]}")
    else:
        from calm.llm_computer.gemma_substrate import (
            GemmaSubstrate, enable_triton_tq4,
        )
        from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

        enable_triton_tq4(True)
        print("[r50.2] loading substrate...")
        m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
        m.preload_gpu("cuda")
        m.warmup(seq_lens=(1, 20))
        tok = GemmaTokenizer.from_gguf(GGUF_PATH)

        rng = random.Random(42)
        prompts = sample_multi_step(rng, N_PROMPTS)
        print(f"\n=== capturing L{TARGET_LAYER} contributions "
              f"({N_PROMPTS} prompts) ===")
        X = capture_activations(tok, m, prompts)
        torch.save({"X": X, "n_prompts": N_PROMPTS},
                   CAPTURES_PATH)
        print(f"  saved captures: {CAPTURES_PATH}")
        del m

    print(f"\n  total samples: {X.shape[0]}")
    print(f"  mean ||x||: {X.norm(dim=1).mean().item():.2f}")

    # Normalize
    device = "cuda"
    X = X.to(device)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std()
    X_norm = (X - X_mean) / X_std

    # --- Train SAE ---
    sae = SAE(D_MODEL, D_SAE_HIDDEN).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=LR)

    print(f"\n=== SAE training (hidden={D_SAE_HIDDEN}, "
          f"target λ={L1_TARGET_LAMBDA}, warmup={WARMUP_STEPS}, "
          f"{N_STEPS} steps) ===")
    n_samples = X_norm.shape[0]
    best_ckpt = None
    best_score = -float("inf")

    def evaluate_full():
        with torch.no_grad():
            x_hat, z = sae(X_norm)
            mse = F.mse_loss(x_hat, X_norm).item()
            l0 = (z > 0).float().sum(dim=1).mean().item()
            var_expl = 1 - ((X_norm - x_hat).pow(2).sum()
                              / X_norm.pow(2).sum()).item()
            frac_dead = ((z > 0).any(dim=0) == False).float().mean().item()
        return mse, l0, var_expl, frac_dead

    for step in range(N_STEPS):
        # Linear λ warmup
        lam = L1_TARGET_LAMBDA * min(1.0, step / max(WARMUP_STEPS, 1))
        idx = torch.randint(0, n_samples, (BATCH_SIZE,), device=device)
        x = X_norm[idx]
        x_hat, z = sae(x)
        mse_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().mean()
        loss = mse_loss + lam * l1_loss
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), GRAD_CLIP)
        optim.step()
        with torch.no_grad():
            sae.dec.weight.data = F.normalize(sae.dec.weight.data, dim=0)

        if step % 100 == 0 or step == N_STEPS - 1:
            mse_full, l0_full, var_full, frac_dead = evaluate_full()
            # Composite score: favor low L0 and high var_expl
            # (penalize L0 with exponential, var_expl linear, require both)
            score = var_full - 0.001 * max(l0_full - 300, 0)
            if score > best_score and var_full >= 0.70:
                best_score = score
                best_ckpt = {
                    "step": step, "mse": mse_full, "l0": l0_full,
                    "var_expl": var_full, "frac_dead": frac_dead,
                    "state_dict": {k: v.clone().detach()
                                    for k, v in sae.state_dict().items()},
                }
            print(f"  step {step:>5}: λ={lam:.4f}  mse={mse_full:.4f}  "
                  f"L0={l0_full:.0f}  var_expl={var_full:.3f}  "
                  f"dead={frac_dead*100:.1f}%")

    # Final metrics from best checkpoint
    print(f"\n=== BEST CHECKPOINT ===")
    if best_ckpt is None:
        print(f"  ✗ No checkpoint met var_expl ≥ 0.70 minimum.")
        print(f"    Reconstruction failed at this λ setting.")
        return

    sae.load_state_dict(best_ckpt["state_dict"])
    print(f"  step:           {best_ckpt['step']}")
    print(f"  MSE:            {best_ckpt['mse']:.4f}")
    print(f"  var_explained:  {best_ckpt['var_expl']*100:.1f}%")
    print(f"  mean L0:        {best_ckpt['l0']:.1f}  "
          f"(of {D_SAE_HIDDEN}, "
          f"{best_ckpt['l0']/D_SAE_HIDDEN*100:.1f}%)")
    print(f"  dead features:  {best_ckpt['frac_dead']*100:.1f}%")

    # Gate
    print(f"\n========== R50.2 GATE ==========")
    gate_var = best_ckpt["var_expl"] >= 0.70
    gate_l0 = best_ckpt["l0"] < 300
    print(f"  var_explained ≥ 70%:   "
          f"{'PASS' if gate_var else 'FAIL'}  "
          f"({best_ckpt['var_expl']*100:.1f}%)")
    print(f"  mean L0 < 300:         "
          f"{'PASS' if gate_l0 else 'FAIL'}  "
          f"({best_ckpt['l0']:.1f})")

    if gate_var and gate_l0:
        print(f"\n  ✓ Sparse dictionary learned. L24 composition info IS")
        print(f"    sparsifiable. Next R50.3: find multi-step-specific")
        print(f"    features by comparing fire-rate on multi-step vs")
        print(f"    single-step prompts.")
    else:
        if gate_var and not gate_l0:
            print(f"\n  ~ Good reconstruction, bad sparsity.")
            print(f"    L24 activations may be fundamentally dense —")
            print(f"    each sample requires many features. Try λ=1e-2")
            print(f"    or conclude circuit isn't sparse.")
        elif not gate_var and gate_l0:
            print(f"\n  ~ Good sparsity but lost reconstruction. λ too high.")
        else:
            print(f"\n  ✗ Both gates fail.")

    torch.save({
        "best_ckpt": best_ckpt,
        "X_mean": X_mean.cpu(), "X_std": X_std.item(),
    }, "/tmp/r50_2_sae.pt")


if __name__ == "__main__":
    sys.exit(main())
