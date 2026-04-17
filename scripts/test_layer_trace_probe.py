"""Round 22: layer-trace linear probe + methodology controls.

R21 showed V at L23 encodes operand digits + ones-place multiplication
+ carry, but not the product's high digits. Question: WHERE in depth
does each feature emerge? Operand digits are probably in L1 (input).
ones_prod + ones_carry are COMPUTED — should emerge at some layer.

Three things in one script:

(1) Layer trace: capture RESIDUAL at L1, L5, L11, L17, L23, L29, L35,
    L41 (the global layers + layer 1) for 270 arithmetic pairs. Probe
    each layer for all labels. The transition from chance → strong
    for ones_prod/ones_carry localizes the multiplication circuit.

(2) Control: shuffle labels, probe L23. Should give chance — if it
    gives high test acc, our probe methodology is broken.

(3) Regularization sweep at L23: weight_decay ∈ {0, 1e-3, 1e-1, 1.0}.
    If the strong-signal labels survive wd=1.0, the probe isn't
    merely overfitting.

Why residual and not V specifically: at every layer the residual
contains everything the model has computed so far. V at L23 is a
PROJECTION of the residual for KV group 1. If we want to know where
features EMERGE, the residual is the native substrate.

Cost: 270 prompts × 8 layers = 2160 captures ≈ 7 min. Probes offline.
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
LAYERS = [1, 5, 11, 17, 23, 29, 35, 41]


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def forward_capture_residuals(m, token_ids, layer_indices):
    """Run forward, capture residual AFTER each specified layer.
    Returns dict {layer_idx: tensor (B, S, d_model)}."""
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


def classification_probe(X, y, n_classes, wd=1e-3, epochs=500, seed=42):
    """Train logistic regression via SGD, return train & test acc."""
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
        loss = F.cross_entropy(clf(X_tr), y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_acc = (clf(X_tr).argmax(-1) == y_tr).float().mean().item()
        te_acc = (clf(X_te).argmax(-1) == y_te).float().mean().item()
    return tr_acc, te_acc


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[layer-trace] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Same 270 pairs as R19/R21
    pairs_by_fd = {d: [] for d in range(1, 10)}
    random.seed(0)
    target = 30
    attempts = 0
    while sum(len(v) for v in pairs_by_fd.values()) < target * 9 and attempts < 5000:
        attempts += 1
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        p = a * b
        fd = int(str(p)[0])
        if fd in pairs_by_fd and len(pairs_by_fd[fd]) < target:
            pairs_by_fd[fd].append((a, b, p))
    pairs = [x for v in pairs_by_fd.values() for x in v]
    print(f"[layer-trace] {len(pairs)} pairs")

    # Capture residual at all target layers
    import time
    t0 = time.time()
    print(f"\n[layer-trace] capturing residual at layers {LAYERS} × {len(pairs)} pairs...")
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
            "a_ones": a % 10, "a_tens": a // 10,
            "b_ones": b % 10, "b_tens": b // 10,
            "p_ones": p % 10, "p_tens": (p // 10) % 10, "p_huns": (p // 100) % 10,
            "fd": int(str(p)[0]),
            "ones_prod": (a % 10) * (b % 10),
            "ones_carry": ((a % 10) * (b % 10)) // 10,
        })
        if (idx + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(pairs)}] {elapsed:.0f}s elapsed")

    per_layer_X = {L: torch.stack(v) for L, v in per_layer_X.items()}
    D = per_layer_X[LAYERS[0]].shape[1]
    print(f"[layer-trace] residual dim = {D}")

    # Labels + class counts
    label_specs = [
        ("a_ones",     10, lambda r: r["a_ones"]),
        ("a_tens",      9, lambda r: r["a_tens"] - 1),
        ("b_ones",     10, lambda r: r["b_ones"]),
        ("b_tens",      9, lambda r: r["b_tens"] - 1),
        ("p_ones",     10, lambda r: r["p_ones"]),
        ("p_tens",     10, lambda r: r["p_tens"]),
        ("p_huns",     10, lambda r: r["p_huns"]),
        ("fd",          9, lambda r: r["fd"] - 1),
        ("ones_prod",  82, lambda r: r["ones_prod"]),
        ("ones_carry",  9, lambda r: r["ones_carry"]),
    ]

    labels = {name: torch.tensor([fn(r) for r in rows], dtype=torch.long)
              for (name, _, fn) in label_specs}

    # Main trace: test acc per (label, layer) at wd=1e-3
    print(f"\n=== (1) LAYER TRACE — test acc per label per layer (wd=1e-3) ===\n")
    headers = ["label"] + [f"L{L}" for L in LAYERS]
    print("  " + " ".join(f"{h:>10}" for h in headers) + f"  {'chance':>8}")
    trace = {}
    for name, ncls, _ in label_specs:
        y = labels[name]
        chance = 1.0 / max(y.unique().numel(), 1)
        row_vals = []
        for L in LAYERS:
            _, te = classification_probe(per_layer_X[L], y, ncls, wd=1e-3)
            row_vals.append(te)
            trace[(name, L)] = te
        cells = [f"{v:>10.3f}" for v in row_vals]
        print(f"  {name:>10} {' '.join(cells)}  {chance:>8.3f}")

    # (2) Control: shuffle labels at L23, all labels. Should give chance.
    print(f"\n=== (2) CONTROL — shuffled labels at L23 (expect ≈ chance) ===\n")
    torch.manual_seed(123)
    for name, ncls, _ in label_specs:
        y = labels[name].clone()
        y = y[torch.randperm(len(y))]   # shuffle
        chance = 1.0 / max(y.unique().numel(), 1)
        _, te = classification_probe(per_layer_X[23], y, ncls, wd=1e-3)
        ratio = te / chance
        flag = " ⚠ PROBE BROKEN" if ratio > 1.5 else ""
        print(f"  {name:>10}  test={te:.3f}  chance={chance:.3f}  ratio={ratio:.2f}x{flag}")

    # (3) Regularization sweep: L23 × the strong labels from R21
    print(f"\n=== (3) REGULARIZATION SWEEP at L23 (does signal survive strong wd?) ===\n")
    wd_grid = [0.0, 1e-3, 1e-1, 1.0]
    sweep_labels = ["a_ones", "a_tens", "ones_prod", "ones_carry", "p_huns", "fd"]
    print(f"  {'label':>10} " + " ".join(f"{'wd=' + str(w):>10}" for w in wd_grid) + f"  {'chance':>8}")
    for name in sweep_labels:
        y = labels[name]
        chance = 1.0 / max(y.unique().numel(), 1)
        row_vals = []
        for wd in wd_grid:
            _, te = classification_probe(per_layer_X[23], y, dict(label_specs)[name][0]
                                          if False else next(n for lbl, n, _ in label_specs if lbl == name),
                                          wd=wd)
            row_vals.append(te)
        cells = [f"{v:>10.3f}" for v in row_vals]
        print(f"  {name:>10} {' '.join(cells)}  {chance:>8.3f}")

    # Key finding: where does each computed feature emerge?
    print(f"\n=== key emergences ===\n")
    for name in ["ones_prod", "ones_carry", "p_ones", "p_huns", "fd"]:
        y = labels[name]
        chance = 1.0 / max(y.unique().numel(), 1)
        emerge_L = None
        for L in LAYERS:
            if trace[(name, L)] >= 3 * chance:
                emerge_L = L
                break
        note = f"emerges at L{emerge_L}" if emerge_L is not None else "never reaches 3x chance"
        print(f"  {name:>10}: {note}  (final L23 = {trace[(name, 23)]:.3f})")

    torch.save({
        "per_layer_X": per_layer_X,
        "rows": rows,
        "trace": trace,
    }, "/tmp/r22_layer_trace.pt")
    print(f"\n[saved] /tmp/r22_layer_trace.pt")


if __name__ == "__main__":
    sys.exit(main())
