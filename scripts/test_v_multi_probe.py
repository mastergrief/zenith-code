"""Round 21: multi-label linear probe on L23 V (KV group 1).

R19 showed V → first-digit-of-product probes at 2x chance (0.22 vs 0.11).
Weak but real. R20 showed the signal is distributed across the 512-d
V subspace (not sparse in d_head=2).

This round: probe V for MANY labels simultaneously. What does V actually
encode? If some labels decode cleanly and others don't, that tells us
the semantic contents of V and gives R22 concrete feature targets.

Candidate labels (per prompt a × b = p):
  Operand encodings:   a, b, a%10, a//10, b%10, b//10
  Product encodings:   p, p%10, (p//10)%10, fd(p) [=R19 baseline]
  Intermediate:        (a%10)*(b%10), ((a%10)*(b%10))//10  [the carry]

Classification probes: train logistic on V → label. Measure test accuracy
vs chance. Any label >3x chance is an interpretable feature direction.

Regression probes: train linear on V → continuous target. Measure test R².
Any R² > 0.3 means V has a strong linear direction for that value.

Cost: 270 V captures (reuse R19's setup) + instant probes offline.
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


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class VCapture:
    def __init__(self, inner):
        self.inner = inner
        self.captured = None
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        self.captured = out.detach().clone()
        return out


def forward_capture_v(m, token_ids, layer_idx):
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

    target = m.layers[layer_idx]
    saved_v = target.attn_v
    capture = VCapture(saved_v)
    target.attn_v = capture
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return capture.captured
    finally:
        target.attn_v = saved_v


def classification_probe(X, y, n_classes, name, epochs=500):
    """Train logistic regression on X→y via SGD, return test accuracy."""
    N, D = X.shape
    torch.manual_seed(42)
    perm = torch.randperm(N)
    split = int(N * 0.8)
    tr, te = perm[:split], perm[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    clf = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(X_tr), y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_acc = (clf(X_tr).argmax(-1) == y_tr).float().mean().item()
        te_acc = (clf(X_te).argmax(-1) == y_te).float().mean().item()
    chance = 1.0 / n_classes
    return {
        "name": name,
        "n_classes": n_classes,
        "chance": chance,
        "train_acc": tr_acc,
        "test_acc": te_acc,
        "ratio": te_acc / chance,
    }


def regression_probe(X, y, name, epochs=500):
    """Train linear regression on X→y via SGD, return test R²."""
    N, D = X.shape
    y_mean = y.mean()
    y_std = y.std().clamp_min(1e-8)
    y_norm = (y - y_mean) / y_std  # normalize

    torch.manual_seed(42)
    perm = torch.randperm(N)
    split = int(N * 0.8)
    tr, te = perm[:split], perm[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_norm[tr], y_norm[te]

    reg = nn.Linear(D, 1)
    opt = torch.optim.Adam(reg.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred = reg(X_tr).squeeze(-1)
        loss = F.mse_loss(pred, y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_te = reg(X_te).squeeze(-1)
        ss_res = ((y_te - pred_te) ** 2).sum().item()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-8)
        # Pearson r
        r = torch.corrcoef(torch.stack([pred_te, y_te]))[0, 1].item()
    return {"name": name, "r2": r2, "pearson_r": r}


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[v-multi-probe] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Generate pairs balanced across first-digit of product
    pairs_by_fd = {d: [] for d in range(1, 10)}
    random.seed(0)
    target_per_digit = 30
    attempts = 0
    while sum(len(v) for v in pairs_by_fd.values()) < target_per_digit * 9 and attempts < 5000:
        attempts += 1
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        p = a * b
        fd = int(str(p)[0])
        if fd in pairs_by_fd and len(pairs_by_fd[fd]) < target_per_digit:
            pairs_by_fd[fd].append((a, b, p))

    pairs = []
    for fd, ps in pairs_by_fd.items():
        pairs.extend(ps)
    print(f"[v-multi-probe] {len(pairs)} pairs, balanced across fd ∈ [1,9]")

    # Capture V at L23, last position, KV group 1 (cols 512-1023)
    L = 23
    print(f"\n[v-multi-probe] capturing attn_v at L{L} last position...")
    Vs = []
    rows = []  # dicts of labels
    for idx, (a, b, p) in enumerate(pairs):
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        v_out = forward_capture_v(m, token_ids, L)
        v_last = v_out[0, -1, 512:1024].float().cpu()
        Vs.append(v_last)
        rows.append({
            "a": a, "b": b, "p": p,
            "a_ones": a % 10, "a_tens": a // 10,
            "b_ones": b % 10, "b_tens": b // 10,
            "p_ones": p % 10, "p_tens": (p // 10) % 10, "p_huns": (p // 100) % 10,
            "fd": int(str(p)[0]),
            "ones_prod": (a % 10) * (b % 10),
            "ones_carry": ((a % 10) * (b % 10)) // 10,
        })
        if (idx + 1) % 30 == 0:
            print(f"  [{idx+1}/{len(pairs)}] last: {a}×{b}={p}")

    X = torch.stack(Vs)  # (N, 512)
    N = len(rows)
    print(f"\n[v-multi-probe] X: {X.shape}")

    # Classification probes
    print(f"\n=== classification probes (linear, 80/20 split, Adam 500 epochs) ===\n")
    print(f"  {'label':>14} {'classes':>8} {'chance':>8} {'train':>8} {'test':>8} {'ratio':>8}  interp")

    cls_specs = [
        ("a",         "a", 100, lambda r: r["a"] - 10),    # 10..99 → 0..89 but sparse; use class=a-10
        ("b",         "b", 100, lambda r: r["b"] - 10),
        ("a_ones",    "a_ones", 10, lambda r: r["a_ones"]),
        ("a_tens",    "a_tens", 9,  lambda r: r["a_tens"] - 1),  # 1..9 → 0..8
        ("b_ones",    "b_ones", 10, lambda r: r["b_ones"]),
        ("b_tens",    "b_tens", 9,  lambda r: r["b_tens"] - 1),
        ("p_ones",    "p_ones", 10, lambda r: r["p_ones"]),
        ("p_tens",    "p_tens", 10, lambda r: r["p_tens"]),
        ("p_huns",    "p_huns", 10, lambda r: r["p_huns"]),
        ("fd(R19)",   "fd",     9,  lambda r: r["fd"] - 1),
        ("ones_prod", "ones_prod", 82, lambda r: r["ones_prod"]),  # 0..81
        ("ones_carry","ones_carry",  9, lambda r: r["ones_carry"]),
    ]

    cls_results = []
    for disp, key, ncls, fn in cls_specs:
        y = torch.tensor([fn(r) for r in rows], dtype=torch.long)
        # For sparse labels (a, b, ones_prod with 82-100 classes but only 270 samples),
        # chance is 1/n_present_classes not 1/n_declared_classes. Use the
        # uniform chance over empirical class count.
        unique = int(y.unique().numel())
        r = classification_probe(X, y, ncls, disp)
        r["n_unique"] = unique
        r["chance_emp"] = 1.0 / max(unique, 1)
        r["ratio_emp"] = r["test_acc"] / r["chance_emp"]
        cls_results.append(r)
        interp = ""
        if r["ratio_emp"] >= 3.0:
            interp = "✓ STRONG"
        elif r["ratio_emp"] >= 1.5:
            interp = "~ weak"
        else:
            interp = "✗ none"
        print(f"  {disp:>14} {unique:>8} {r['chance_emp']:>8.3f} "
              f"{r['train_acc']:>8.3f} {r['test_acc']:>8.3f} {r['ratio_emp']:>8.2f}x  {interp}")

    # Regression probes
    print(f"\n=== regression probes (linear, 80/20, Adam 500 epochs) ===\n")
    print(f"  {'label':>14} {'R²':>8} {'Pearson r':>12}  interp")

    reg_specs = [
        ("a",         lambda r: r["a"]),
        ("b",         lambda r: r["b"]),
        ("p",         lambda r: r["p"]),
        ("ones_prod", lambda r: r["ones_prod"]),
    ]

    for disp, fn in reg_specs:
        y = torch.tensor([fn(r) for r in rows], dtype=torch.float32)
        rr = regression_probe(X, y, disp)
        interp = ""
        if rr["r2"] >= 0.3:
            interp = "✓ STRONG"
        elif rr["r2"] >= 0.1:
            interp = "~ weak"
        else:
            interp = "✗ none"
        print(f"  {disp:>14} {rr['r2']:>+8.3f} {rr['pearson_r']:>+12.3f}  {interp}")

    # Summary
    print(f"\n=== summary ===\n")
    strong = [r for r in cls_results if r["ratio_emp"] >= 3.0]
    weak = [r for r in cls_results if 1.5 <= r["ratio_emp"] < 3.0]
    print(f"  STRONG (≥3x chance): {[r['name'] for r in strong]}")
    print(f"  weak  (1.5-3x):      {[r['name'] for r in weak]}")
    print(f"\n  R19 baseline (fd): ratio 2.0x — same data here should reproduce.")
    print(f"  If any operand label decodes cleanly, V contains operand encodings")
    print(f"  → R22 SAE on V will recover operand feature directions.")

    torch.save({
        "X": X, "rows": rows,
        "cls_results": cls_results,
    }, "/tmp/r21_v_probes.pt")
    print(f"\n[saved] /tmp/r21_v_probes.pt")


if __name__ == "__main__":
    sys.exit(main())
