"""Round 19: inspect V-vectors at L23 KV group 1 across arithmetic pairs.

Capture the V output (after attn_v projection, pre-attention-op) at
L23 for 50 multiplication prompts. For each, collect the V at the
LAST prompt position (which is about to determine the next token =
the first digit of the product).

Analyses:
  1. Is V clustered by first-digit of the product? PCA + color by
     first digit.
  2. Cosine similarity between Vs for same first-digit products.
  3. Is there a specific direction in V that linearly predicts the
     first digit? Linear probe (logistic regression) on V → digit.

If probe accuracy >> chance (1/9 or so), interpretable features
live in V — confirmable via SAE later.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


# Collect V vectors via instrumentation. Patch attn_v to also save output.
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
    """Run forward, capture attn_v output at layer_idx."""
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
        return capture.captured  # (B, S, n_kv * d_kv)
    finally:
        target.attn_v = saved_v


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[v-analysis] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Generate 50 multiplication prompts spanning products with different
    # first digits. We want representatives of first_digit ∈ {1..9} for
    # the probe.
    pairs_by_fd = {d: [] for d in range(1, 10)}  # first digit of product
    import random
    random.seed(0)
    attempts = 0
    target_per_digit = 30  # ~270 total
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
    print(f"[v-analysis] {len(pairs)} pairs, first-digit distribution:")
    for fd in range(1, 10):
        print(f"  fd={fd}: {len(pairs_by_fd[fd])} pairs")

    # Collect V at L23, last position
    L = 23
    print(f"\n[v-analysis] capturing attn_v output at L{L} last position ...")
    # V shape per call: (1, S, n_kv * d_kv) = (1, S, 1024)
    # KV group 1 = columns [512:1024]
    Vs = []   # (N, 512)
    labels = []  # first digit of product

    for idx, (a, b, p) in enumerate(pairs):
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        v_out = forward_capture_v(m, token_ids, L)  # (1, S, 1024)
        # Take last position, KV group 1 (cols 512-1023)
        v_last = v_out[0, -1, 512:1024].float().cpu()
        Vs.append(v_last)
        labels.append(int(str(p)[0]))
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(pairs)}] last pair: {a}×{b}={p}, fd={labels[-1]}")

    V_mat = torch.stack(Vs)  # (N, 512)
    labels_t = torch.tensor(labels)
    print(f"\n[v-analysis] V matrix: {V_mat.shape}, labels: {labels_t.shape}")

    # Analysis 1: mean cosine similarity within-class vs between-class
    V_norm = V_mat / V_mat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_sim = V_norm @ V_norm.T  # (N, N)
    same_mask = labels_t.unsqueeze(0) == labels_t.unsqueeze(1)
    # Exclude diagonal
    eye = torch.eye(len(labels_t), dtype=torch.bool)
    within_cos = cos_sim[same_mask & ~eye].mean().item()
    between_cos = cos_sim[~same_mask].mean().item()
    print(f"\n  mean within-fd cosine: {within_cos:.4f}")
    print(f"  mean between-fd cosine: {between_cos:.4f}")
    print(f"  separation: {within_cos - between_cos:+.4f}")

    # Analysis 2: linear probe (logistic regression via sklearn-style closed-form)
    # Simple: train a one-vs-rest logistic regression with torch.linalg.lstsq
    # Actually easier: use torch's built-in functional
    import torch.nn as nn
    import torch.nn.functional as F

    # Normalize features
    X = V_mat
    y = labels_t - 1  # 0..8
    N, D = X.shape
    n_classes = 9  # first digit 1-9

    # Train-test split (80-20 random)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    split = int(N * 0.8)
    train_idx = perm[:split]
    test_idx = perm[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Small linear classifier trained via SGD
    clf = nn.Linear(D, n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01)
    for epoch in range(500):
        opt.zero_grad()
        logits = clf(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (clf(X_train).argmax(-1) == y_train).float().mean().item()
        test_acc = (clf(X_test).argmax(-1) == y_test).float().mean().item()
    chance = 1.0 / n_classes

    print(f"\n  linear probe (V-group-1 → first digit):")
    print(f"    train acc: {train_acc:.3f}")
    print(f"    test acc:  {test_acc:.3f}  (chance = {chance:.3f})")
    print(f"    vs chance: {test_acc / chance:.1f}x")

    if test_acc > 2 * chance:
        print(f"    ✓ STRONG: V linearly encodes the product's first digit")
    elif test_acc > 1.3 * chance:
        print(f"    ~ WEAK: some information present but not cleanly encoded")
    else:
        print(f"    ✗ NONE: first digit not linearly decodable from V")


if __name__ == "__main__":
    sys.exit(main())
