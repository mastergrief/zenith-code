"""Round 23: fine-grained layer trace across the L29→L35 "finalizer" transition.

R22 showed fd jumps 0.39 (L29) → 0.78 (L35) and p_huns jumps 0.19 → 0.33
across this 6-layer span. Question: is it a cumulative climb or a
sharp transition at one specific layer?

Methodology hardened from R22:
  - Capture at L28, L29, L30, L31, L32, L33, L34, L35, L36 (9 layers)
  - Probe fd, p_huns, p_tens, p_ones at each
  - Shuffled-label control at each layer (catch broken probes)
  - wd=1e-2 (stronger than R22's 1e-3, since R22 showed high-class
    probes need regularization)

If one specific layer does the heavy lifting → compilable locus.
If distributed → need deeper investigation (per-head, FFN, etc).
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
LAYERS = [28, 29, 30, 31, 32, 33, 34, 35, 36]


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def forward_capture_residuals(m, token_ids, layer_indices):
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

    captured = {}
    want = set(layer_indices)
    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i in want:
                captured[i] = h[0, -1, :].detach().clone()
    return captured


def probe(X, y, n_classes, wd=1e-2, epochs=500, seed=42):
    N, D = X.shape
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    split = int(N * 0.8)
    tr, te = perm[:split], perm[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    clf = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(X_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        te_acc = (clf(X_te).argmax(-1) == y_te).float().mean().item()
    return te_acc


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[finalizer-trace] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    pairs_by_fd = {d: [] for d in range(1, 10)}
    random.seed(0)
    while sum(len(v) for v in pairs_by_fd.values()) < 270:
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        p = a * b
        fd = int(str(p)[0])
        if len(pairs_by_fd[fd]) < 30:
            pairs_by_fd[fd].append((a, b, p))
    pairs = [x for v in pairs_by_fd.values() for x in v]

    t0 = time.time()
    print(f"[finalizer-trace] capturing residual at L{LAYERS[0]}..L{LAYERS[-1]} × {len(pairs)} pairs...")
    per_layer_X = {L: [] for L in LAYERS}
    rows = []
    for idx, (a, b, p) in enumerate(pairs):
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        caps = forward_capture_residuals(m, token_ids, LAYERS)
        for L in LAYERS:
            per_layer_X[L].append(caps[L].float().cpu())
        rows.append({
            "a": a, "b": b, "p": p,
            "p_ones": p % 10,
            "p_tens": (p // 10) % 10,
            "p_huns": (p // 100) % 10,
            "fd": int(str(p)[0]),
        })
        if (idx + 1) % 60 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {time.time()-t0:.0f}s")
    per_layer_X = {L: torch.stack(v) for L, v in per_layer_X.items()}

    label_specs = [
        ("fd",      9,  lambda r: r["fd"] - 1),
        ("p_huns", 10,  lambda r: r["p_huns"]),
        ("p_tens", 10,  lambda r: r["p_tens"]),
        ("p_ones", 10,  lambda r: r["p_ones"]),
    ]
    labels = {n: torch.tensor([fn(r) for r in rows], dtype=torch.long)
              for n, _, fn in label_specs}

    # Main trace
    print(f"\n=== FINALIZER TRACE (wd=1e-2) ===\n")
    print("  label     " + " ".join(f"L{L:>3}" for L in LAYERS) + "  chance")
    for name, ncls, _ in label_specs:
        y = labels[name]
        chance = 1.0 / max(y.unique().numel(), 1)
        row = []
        for L in LAYERS:
            te = probe(per_layer_X[L], y, ncls)
            row.append(te)
        print(f"  {name:>8}  " + " ".join(f"{v:>4.2f}" for v in row) + f"  {chance:>5.3f}")

    # Shuffled controls at each layer
    print(f"\n=== SHUFFLED CONTROL at each layer (should be ≈ chance) ===\n")
    print("  label     " + " ".join(f"L{L:>3}" for L in LAYERS))
    for name, ncls, _ in label_specs:
        torch.manual_seed(123)
        y_shuf = labels[name].clone()[torch.randperm(len(labels[name]))]
        chance = 1.0 / max(y_shuf.unique().numel(), 1)
        row = []
        for L in LAYERS:
            te = probe(per_layer_X[L], y_shuf, ncls)
            row.append(te)
        row_str = " ".join(f"{v:>4.2f}" for v in row)
        max_r = max(row) / chance
        flag = " ⚠" if max_r > 1.5 else ""
        print(f"  {name:>8}  {row_str}  (chance={chance:.3f}){flag}")

    # Transition detection: per-layer delta
    print(f"\n=== PER-LAYER DELTA (layer L vs L-1) ===\n")
    print("  label    " + " ".join(f"L{L-1}→L{L}" for L in LAYERS[1:]))
    for name, ncls, _ in label_specs:
        y = labels[name]
        accs = [probe(per_layer_X[L], y, ncls) for L in LAYERS]
        deltas = [accs[i] - accs[i-1] for i in range(1, len(accs))]
        print(f"  {name:>7}  " + " ".join(f"{d:>+6.2f}" for d in deltas))

    torch.save({
        "per_layer_X": per_layer_X,
        "rows": rows,
    }, "/tmp/r23_finalizer.pt")
    print(f"\n[saved] /tmp/r23_finalizer.pt")


if __name__ == "__main__":
    sys.exit(main())
