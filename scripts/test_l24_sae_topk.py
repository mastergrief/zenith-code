"""Round 50.3: TopK SAE on cached L24 captures.

R50.1 (λ=5e-4) and R50.2 (λ=5e-3) both plateaued at L0 ~1700-2000
due to L1 regularization's soft sparsity. Modern SAE literature
(OpenAI's k-sparse, DeepMind's JumpReLU) uses hard TopK constraints:
keep only the K highest-activating features per sample, zero the
rest. Guarantees L0 ≤ K exactly.

Hypothesis: with hard sparsity at K=100 or K=200, L24's composition
info can be reconstructed from a sparse dictionary. If K=100 gives
var_expl ≥ 70%, composition IS sparsifiable. If K=200 fails too,
the activation distribution is genuinely dense.

TopK SAE:
  z = ReLU(encoder(x))
  z_sparse = keep top K of z per sample, zero rest
  x_hat = decoder(z_sparse)
  loss = MSE(x_hat, x)       # no L1, sparsity is hard

Train 2 variants in parallel: K=100 and K=200. Pick the one with
better reconstruction.

Uses cached activations from R50.2 at /tmp/r50_captures.pt. No
Gemma needed. Trains in ~2 min on GPU.

Gates:
  Primary: K=100 var_expl ≥ 70%
  Fallback: K=200 var_expl ≥ 70%
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


CAPTURES_PATH = "/tmp/r50_captures.pt"
D_MODEL = 2560
D_SAE_HIDDEN = 5120
LR = 1e-3
N_STEPS = 3000
BATCH_SIZE = 256
GRAD_CLIP = 1.0
K_VALUES = [50, 100, 200, 500]


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_hidden, k):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)
        self.k = k
        with torch.no_grad():
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)

    def encode(self, x):
        z_pre = F.relu(self.enc(x))
        # Per-sample top-K: keep only K highest values, zero the rest.
        # For (B, D), find top-K along dim=1, mask others to zero.
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
        x_hat = self.decode(z)
        return x_hat, z


def train_topk(X_norm, k, device, log_prefix=""):
    sae = TopKSAE(D_MODEL, D_SAE_HIDDEN, k).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=LR)
    n_samples = X_norm.shape[0]

    print(f"\n{log_prefix}=== TopK SAE k={k} ===")
    best_ckpt = None
    best_score = -float("inf")

    for step in range(N_STEPS):
        idx = torch.randint(0, n_samples, (BATCH_SIZE,), device=device)
        x = X_norm[idx]
        x_hat, _ = sae(x)
        loss = F.mse_loss(x_hat, x)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), GRAD_CLIP)
        optim.step()
        with torch.no_grad():
            sae.dec.weight.data = F.normalize(sae.dec.weight.data, dim=0)

        if step % 200 == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                x_hat, z = sae(X_norm)
                mse = F.mse_loss(x_hat, X_norm).item()
                l0 = (z > 0).float().sum(dim=1).mean().item()
                var_expl = 1 - ((X_norm - x_hat).pow(2).sum()
                                 / X_norm.pow(2).sum()).item()
                frac_dead = ((z > 0).any(dim=0) == False).float().mean().item()
            # Save best-var_expl checkpoint
            if var_expl > best_score:
                best_score = var_expl
                best_ckpt = {
                    "step": step, "mse": mse, "l0": l0,
                    "var_expl": var_expl, "frac_dead": frac_dead,
                    "k": k,
                    "state_dict": {kk: vv.clone().detach()
                                    for kk, vv in sae.state_dict().items()},
                }
            print(f"{log_prefix}  step {step:>5}: mse={mse:.4f}  "
                  f"L0={l0:.1f}  var_expl={var_expl:.3f}  "
                  f"dead={frac_dead*100:.1f}%")

    return best_ckpt


def main():
    if not os.path.exists(CAPTURES_PATH):
        print(f"ERROR: no cached captures at {CAPTURES_PATH}.")
        print(f"Run R50.2 first or re-capture.")
        return 1
    print(f"[r50.3] loading captures from {CAPTURES_PATH}...")
    data = torch.load(CAPTURES_PATH, weights_only=False)
    X = data["X"]
    print(f"  {X.shape[0]} samples, dim {X.shape[1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std()
    X_norm = (X - X_mean) / X_std

    results = []
    for k in K_VALUES:
        ckpt = train_topk(X_norm, k, device, log_prefix=f"[k={k}] ")
        results.append(ckpt)

    # Summary
    print(f"\n========== R50.3 SUMMARY ==========")
    print(f"{'k':>5} {'var_expl':>10} {'L0':>8} {'dead':>8}")
    for r in results:
        print(f"{r['k']:>5} {r['var_expl']*100:>9.1f}% {r['l0']:>8.1f} "
              f"{r['frac_dead']*100:>7.1f}%")

    # Gate: smallest K meeting var_expl >= 0.70
    passing = [r for r in results if r["var_expl"] >= 0.70]
    print(f"\n========== R50.3 GATE ==========")
    if passing:
        best = min(passing, key=lambda r: r["k"])
        print(f"  ✓ K={best['k']} achieves var_expl={best['var_expl']*100:.1f}%")
        print(f"    Composition IS sparsifiable at K≤{best['k']}.")
        print(f"    Next R50.4: feature analysis — find multi-step-")
        print(f"    specific features by comparing to single-step captures.")
    else:
        print(f"  ✗ No K reached var_expl ≥ 70%.")
        print(f"    Even hard-TopK can't reconstruct with ≤ 500 features.")
        print(f"    L24 activation distribution is fundamentally dense.")
        print(f"    Stop SAE approach; consider:")
        print(f"      - Train on FFN hidden (10240-d) instead of residual")
        print(f"      - Try different layer (L25 or L29)")
        print(f"      - Accept null; shipping path is R46 parse-verify-bias")

    torch.save({
        "results": results,
        "X_mean": X_mean.cpu(), "X_std": X_std.item(),
    }, "/tmp/r50_3_topk.pt")
    print(f"\n  saved: /tmp/r50_3_topk.pt")


if __name__ == "__main__":
    sys.exit(main())
