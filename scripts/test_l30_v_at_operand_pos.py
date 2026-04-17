"""Round 27: probe V at L30 positions 3 and 7 — do they literally contain
a_ones and b_ones?

R26 showed L30 H6 attends 60% to a_ones token (pos 3) and L30 H4
attends to b_ones token (pos 7). The INTERPRETATION is that these
heads select content already at those positions.

This round TESTS the interpretation. Capture V at L30 at every position.
Probe V[pos=3] for a_ones, V[pos=7] for b_ones. If both decode cleanly,
H6/H4 are position selectors over content already present.

Also probe V at pos=3 for b_ones (control — should fail or be weak)
and V at pos=7 for a_ones (control — also weak).

If confirmed: the compilable structure is
  LookUpExact(pos=3) → a_ones (into H6's V slot)
  LookUpExact(pos=7) → b_ones (into H4's V slot)
→ L31-L32 FFN (ReGLU transforms of a_ones, b_ones → fd)
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
# L30 SWA reuses V from L22 (shared_kv_layers=18, n_layer_kv_from_start=24,
# SWA → cutoff - 2 = 22). So H6's attention at L30 reads L22's V content.
TARGET_LAYER = 22


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


def forward_capture_v_at_layer(m, token_ids, layer_idx):
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
    saved = target.attn_v
    cap = VCapture(saved)
    target.attn_v = cap
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
                if i == layer_idx:
                    break
    finally:
        target.attn_v = saved

    return cap.captured  # (1, S, n_kv * d_kv)


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
        te_acc = (clf(X[te]).argmax(-1) == y[te]).float().mean().item()
    return te_acc


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l30-v-operand-pos] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # 270 pairs balanced by fd
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

    # For each pair, capture V at L30, extract pos=3 (a_ones) and pos=7 (b_ones)
    # Both operands are 2-digit so prompts tokenize same:
    # [bos, ▁, a_tens, a_ones, ▁times, ▁, b_tens, b_ones, ▁equals, ▁]
    POS_A_ONES = 3
    POS_B_ONES = 7
    POS_A_TENS = 2
    POS_B_TENS = 6

    V_at_pos3, V_at_pos7, V_at_pos2, V_at_pos6, V_at_pos9 = [], [], [], [], []
    rows = []

    t0 = time.time()
    print(f"[l30-v-operand-pos] capturing V at L{TARGET_LAYER} × {len(pairs)} pairs")
    for idx, (a, b, p) in enumerate(pairs):
        prompt = f"{a} times {b} equals "
        token_ids_list = tok.encode(prompt)
        # Confirm tokenization: should have len 10 with operand-digit positions
        if idx == 0:
            tokens = [tok.id_to_token.get(tid, f"?{tid}") for tid in token_ids_list]
            print(f"  pair example: {a}×{b} tokens = {tokens}")
            print(f"  expected: pos={POS_A_ONES} is a_ones, pos={POS_B_ONES} is b_ones")
            print(f"  actual:   pos={POS_A_ONES}={tokens[POS_A_ONES]!r}, pos={POS_B_ONES}={tokens[POS_B_ONES]!r}")

        # Require len=10 for uniform positions
        if len(token_ids_list) != 10:
            continue

        token_ids = torch.tensor([token_ids_list], device="cuda")
        v_out = forward_capture_v_at_layer(m, token_ids, TARGET_LAYER)  # (1, 10, 1024)
        # V KV group 1 (cols 512-1023) — H4 + H6 share this
        # Actually L30 is SWA with n_heads_kv=2, d_head_kv=256 → V total 512
        # Let's check the actual shape
        v_total = v_out.shape[-1]
        v_group_0 = v_out[0, :, :v_total // 2].float().cpu()  # cols 0..256 (H0-H3)
        v_group_1 = v_out[0, :, v_total // 2:].float().cpu()  # cols 256..512 (H4-H7)

        V_at_pos3.append(v_group_1[POS_A_ONES])  # KV group 1 (H4/H6) at a_ones pos
        V_at_pos7.append(v_group_1[POS_B_ONES])  # KV group 1 at b_ones pos
        V_at_pos2.append(v_group_1[POS_A_TENS])
        V_at_pos6.append(v_group_1[POS_B_TENS])
        V_at_pos9.append(v_group_1[-1])  # last pos (the ▁)

        rows.append({
            "a": a, "b": b, "p": p,
            "a_ones": a % 10, "a_tens": a // 10,
            "b_ones": b % 10, "b_tens": b // 10,
        })
        if (idx + 1) % 30 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {time.time()-t0:.0f}s")

    # Stack
    V_at_pos3 = torch.stack(V_at_pos3)
    V_at_pos7 = torch.stack(V_at_pos7)
    V_at_pos2 = torch.stack(V_at_pos2)
    V_at_pos6 = torch.stack(V_at_pos6)
    V_at_pos9 = torch.stack(V_at_pos9)
    print(f"\n[l30-v-operand-pos] V[pos] shapes: {V_at_pos3.shape}")

    y_a_ones = torch.tensor([r["a_ones"] for r in rows], dtype=torch.long)
    y_a_tens = torch.tensor([r["a_tens"] for r in rows], dtype=torch.long)
    y_b_ones = torch.tensor([r["b_ones"] for r in rows], dtype=torch.long)
    y_b_tens = torch.tensor([r["b_tens"] for r in rows], dtype=torch.long)

    # Main hypothesis test: does V at pos 3 encode a_ones? V at pos 7 encode b_ones?
    print(f"\n=== V at L{TARGET_LAYER} KV-group-1 decoding test ===\n")
    print(f"  (chance 10-class = 0.1; test=0.80/0.20 split)\n")

    # V at pos 3 (a_ones position) probed for various targets
    for pos_name, V_tensor in [
        ("pos=3 (a_ones tok)", V_at_pos3),
        ("pos=7 (b_ones tok)", V_at_pos7),
        ("pos=2 (a_tens tok)", V_at_pos2),
        ("pos=6 (b_tens tok)", V_at_pos6),
        ("pos=9 (last ▁)   ", V_at_pos9),
    ]:
        print(f"  {pos_name}:")
        for lbl_name, y, ncls in [
            ("a_ones", y_a_ones, 10),
            ("a_tens", y_a_tens, 10),
            ("b_ones", y_b_ones, 10),
            ("b_tens", y_b_tens, 10),
        ]:
            acc = probe(V_tensor, y, ncls)
            flag = ""
            if acc > 0.5:
                flag = " ✓ STRONG"
            elif acc > 0.25:
                flag = " ~ moderate"
            print(f"    → {lbl_name:>7}  test={acc:.3f}{flag}")


if __name__ == "__main__":
    sys.exit(main())
