"""Round 24: attention vs FFN ablation at L30, L31, L32 — which writes fd?

R23 showed fd builds across L30-L32 as a 3-layer pipeline with the
biggest step at L31→L32. Each layer has both attention and FFN
sub-modules. Which one at each layer writes the product-digit info?

Mechanism:
  For each layer L ∈ {29, 30, 31, 32}:
    baseline:    run normal forward, capture h at L's output. Probe fd.
    attn-zero:   at layer L, replace attn_output with zero-returning
                 stub → attn_out = 0 + inpL. Capture h at L's output.
    ffn-zero:    at layer L, replace ffn_down with zero-returning
                 stub → h = 0 + attn_out. Capture h at L's output.
    Probe fd on each capture. Drop in fd when zeroing = what that
    sub-module contributed.

If attention contributes more → product digits are built by pulling
info from prior positions (look-up mechanism, easy to compile).
If FFN contributes more → product digits are built by register-to-
register nonlinear computation (ReGLU-like, also compilable but
requires more gates).
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
TARGET_LAYERS = [29, 30, 31, 32]


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class ZeroReturning:
    """Wrap a linear to return all-zeros of its output shape.
    Matches MmapTq4Linear's call convention."""
    def __init__(self, inner):
        self.inner = inner
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        return torch.zeros_like(out)


def forward_capture_at_layer(m, token_ids, probe_layer,
                              ablate_layer=None, ablate_what=None):
    """Run forward, capture residual at the output of `probe_layer`.
    Optionally ablate attn_output or ffn_down of `ablate_layer`."""
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

    saved = None
    if ablate_layer is not None:
        tgt = m.layers[ablate_layer]
        if ablate_what == "attn":
            saved = ("attn_output", tgt.attn_output)
            tgt.attn_output = ZeroReturning(tgt.attn_output)
        elif ablate_what == "ffn":
            saved = ("ffn_down", tgt.ffn_down)
            tgt.ffn_down = ZeroReturning(tgt.ffn_down)
        else:
            raise ValueError(ablate_what)

    captured = None
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
                if i == probe_layer:
                    captured = h[0, -1, :].detach().clone()
                    break
        return captured
    finally:
        if saved is not None:
            attr_name, original = saved
            setattr(m.layers[ablate_layer], attr_name, original)


def probe(X, y, n_classes, wd=1e-2, epochs=500, seed=42):
    N, D = X.shape
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    split = int(N * 0.8)
    tr, te = perm[:split], perm[split:]
    clf = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(X[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        te = (clf(X[te]).argmax(-1) == y[te]).float().mean().item()
    return te


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[attn-vs-ffn] loading substrate...")
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

    # Capture 3 variants at each target layer output
    # variants: baseline, no-attn-at-L, no-ffn-at-L
    # For each L, probe fd on the 3 variants
    print(f"\n[attn-vs-ffn] capturing baselines + ablations at layers {TARGET_LAYERS}")

    # Collect: per_layer_variant[L][variant] = list of (N, D) tensors
    per_layer_variant = {L: {"baseline": [], "no_attn": [], "no_ffn": []}
                          for L in TARGET_LAYERS}
    rows = []

    t0 = time.time()
    for idx, (a, b, p) in enumerate(pairs):
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")

        for L in TARGET_LAYERS:
            # Baseline
            h_base = forward_capture_at_layer(m, token_ids, probe_layer=L)
            per_layer_variant[L]["baseline"].append(h_base.float().cpu())

            # No-attn at L
            h_no_attn = forward_capture_at_layer(m, token_ids, probe_layer=L,
                                                   ablate_layer=L, ablate_what="attn")
            per_layer_variant[L]["no_attn"].append(h_no_attn.float().cpu())

            # No-ffn at L
            h_no_ffn = forward_capture_at_layer(m, token_ids, probe_layer=L,
                                                  ablate_layer=L, ablate_what="ffn")
            per_layer_variant[L]["no_ffn"].append(h_no_ffn.float().cpu())

        rows.append({
            "a": a, "b": b, "p": p,
            "fd": int(str(p)[0]),
            "p_ones": p % 10,
            "p_huns": (p // 100) % 10,
        })
        if (idx + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) * len(TARGET_LAYERS) * 3 / elapsed
            total = len(pairs) * len(TARGET_LAYERS) * 3
            eta = (total - (idx + 1) * len(TARGET_LAYERS) * 3) / rate
            print(f"  [{idx+1}/{len(pairs)}] {elapsed:.0f}s  rate={rate:.1f} fwd/s  ETA={eta:.0f}s")

    # Stack
    for L in TARGET_LAYERS:
        for variant in ["baseline", "no_attn", "no_ffn"]:
            per_layer_variant[L][variant] = torch.stack(per_layer_variant[L][variant])

    # Probe fd, p_huns, p_ones
    y_fd = torch.tensor([r["fd"] - 1 for r in rows], dtype=torch.long)
    y_huns = torch.tensor([r["p_huns"] for r in rows], dtype=torch.long)
    y_ones = torch.tensor([r["p_ones"] for r in rows], dtype=torch.long)

    print(f"\n=== fd decoding at each layer (test acc) ===\n")
    print(f"  {'layer':>5}  {'baseline':>10}  {'no_attn':>10}  {'no_ffn':>10}  "
          f"{'Δattn':>8}  {'Δffn':>8}  {'winner':>10}")
    for L in TARGET_LAYERS:
        b = probe(per_layer_variant[L]["baseline"], y_fd, 9)
        na = probe(per_layer_variant[L]["no_attn"], y_fd, 9)
        nf = probe(per_layer_variant[L]["no_ffn"], y_fd, 9)
        d_attn = b - na   # drop when attn zeroed = attn contribution
        d_ffn = b - nf    # drop when ffn zeroed = ffn contribution
        winner = "ATTN" if d_attn > d_ffn + 0.02 else ("FFN" if d_ffn > d_attn + 0.02 else "tie")
        print(f"  L{L:>4}  {b:>10.3f}  {na:>10.3f}  {nf:>10.3f}  "
              f"{d_attn:>+8.3f}  {d_ffn:>+8.3f}  {winner:>10}")

    print(f"\n=== p_huns decoding at each layer (test acc) ===\n")
    print(f"  {'layer':>5}  {'baseline':>10}  {'no_attn':>10}  {'no_ffn':>10}  "
          f"{'Δattn':>8}  {'Δffn':>8}  {'winner':>10}")
    for L in TARGET_LAYERS:
        b = probe(per_layer_variant[L]["baseline"], y_huns, 10)
        na = probe(per_layer_variant[L]["no_attn"], y_huns, 10)
        nf = probe(per_layer_variant[L]["no_ffn"], y_huns, 10)
        d_attn = b - na
        d_ffn = b - nf
        winner = "ATTN" if d_attn > d_ffn + 0.02 else ("FFN" if d_ffn > d_attn + 0.02 else "tie")
        print(f"  L{L:>4}  {b:>10.3f}  {na:>10.3f}  {nf:>10.3f}  "
              f"{d_attn:>+8.3f}  {d_ffn:>+8.3f}  {winner:>10}")

    print(f"\n=== p_ones decoding at each layer (test acc) ===\n")
    print(f"  {'layer':>5}  {'baseline':>10}  {'no_attn':>10}  {'no_ffn':>10}  "
          f"{'Δattn':>8}  {'Δffn':>8}  {'winner':>10}")
    for L in TARGET_LAYERS:
        b = probe(per_layer_variant[L]["baseline"], y_ones, 10)
        na = probe(per_layer_variant[L]["no_attn"], y_ones, 10)
        nf = probe(per_layer_variant[L]["no_ffn"], y_ones, 10)
        d_attn = b - na
        d_ffn = b - nf
        winner = "ATTN" if d_attn > d_ffn + 0.02 else ("FFN" if d_ffn > d_attn + 0.02 else "tie")
        print(f"  L{L:>4}  {b:>10.3f}  {na:>10.3f}  {nf:>10.3f}  "
              f"{d_attn:>+8.3f}  {d_ffn:>+8.3f}  {winner:>10}")

    torch.save({
        "per_layer_variant": {L: {k: v for k, v in per_layer_variant[L].items()}
                               for L in TARGET_LAYERS},
        "rows": rows,
    }, "/tmp/r24_attn_vs_ffn.pt")
    print(f"\n[saved] /tmp/r24_attn_vs_ffn.pt")


if __name__ == "__main__":
    sys.exit(main())
