"""HullKVCache as a drop-in for Small2DTransformer's hard-max attention.

For each real program, extract per-head (q, k, v) and run the same
argmax-select-values operation two ways:
  1. The batched einsum + argmax + scatter path (production).
  2. Per-position: build a HullKVCache of past (k, v), query with q.

The two must produce identical outputs (or equivalent up to tie-breaking).

This is a proof-of-correctness for the cache at the head level,
not a full incremental-decoding rewrite — that would require refactoring
forward() to generate token-by-token, which is a perf-only change and
only pays off at long sequences (S >> hull_size). Our programs use S ≤ 5.
"""

from __future__ import annotations

import torch

from calm.llm_computer.hull_cache import HullKVCache
from calm.llm_computer.programs.retrieve_by_index import build_retrieve_by_index
from calm.llm_computer.programs.read_by_key import build_read_by_key


def _extract_qkv_per_head(model, x: torch.Tensor, layer: int):
    """Return (q, k, v) shaped (H, S, 2) for `layer`'s attention input."""
    B, S = x.shape
    cfg = model.config
    with torch.no_grad():
        pos_idx = torch.arange(S, device=x.device)
        resid = model.tok(x) + model.pos(pos_idx)
        # Run prior layers to compute the correct residual at `layer`.
        for L in range(layer):
            qkv = model.W_qkv[L](resid).reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            idx = scores.argmax(dim=-1, keepdim=True)
            w = torch.zeros_like(scores)
            w.scatter_(-1, idx, 1.0)
            attn = torch.einsum("bhij,bhjd->bhid", w, v)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            resid = resid + model.W_out[L](attn)
            gate, val = model.ff_in[L](resid).chunk(2, dim=-1)
            resid = resid + model.ff_out[L](torch.relu(gate) * val)
        qkv = model.W_qkv[layer](resid).reshape(B, S, 3, cfg.n_heads, cfg.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, Dh)
    # Return B=0 slice: (H, S, 2) for each
    return q[0], k[0], v[0]  # (H, S, 2)


def _hull_attention(q, k, v) -> torch.Tensor:
    """Compute hard-max attention using HullKVCache, per head.

    Shapes:
      q, k: (H, S, 2)  —  the 2D query and key per head/position
      v:    (H, S, 2)  —  the value

    Returns (H, S, 2) — same as the batched hard-max attention output.
    """
    H, S, _ = q.shape
    out = torch.zeros_like(v)
    for h in range(H):
        cache = HullKVCache()
        for i in range(S):
            cache.insert(
                (float(k[h, i, 0].item()), float(k[h, i, 1].item())),
                v[h, i],
            )
            q_i = (float(q[h, i, 0].item()), float(q[h, i, 1].item()))
            out[h, i] = cache.query(q_i)
    return out


def _reference_hard_max_attention(q, k, v) -> torch.Tensor:
    """The production path from Small2DTransformer._attention, per head."""
    H, S, Dh = q.shape
    scores = torch.einsum("hid,hjd->hij", q, k)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device),
                      diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    idx = scores.argmax(dim=-1, keepdim=True)
    w = torch.zeros_like(scores)
    w.scatter_(-1, idx, 1.0)
    out = torch.einsum("hij,hjd->hid", w, v)
    return out


def test_hull_matches_reference_retrieve_by_index():
    model = build_retrieve_by_index(vocab_size=4, max_len=5)
    inp = torch.tensor([[2, 0, 3, 1, 2]], dtype=torch.long)
    q, k, v = _extract_qkv_per_head(model, inp, layer=0)
    hull_out = _hull_attention(q, k, v)
    ref_out = _reference_hard_max_attention(q, k, v)
    assert torch.equal(hull_out, ref_out), \
        "hull attention diverges from reference on retrieve_by_index"


def test_hull_matches_reference_read_by_key():
    model = build_read_by_key(vocab_size=4, max_len=5)
    inp = torch.tensor([[2, 0, 3, 1, 2]], dtype=torch.long)
    # Layer 0 attention is zeroed (LookUpExact is at layer 1). Test layer 1.
    q, k, v = _extract_qkv_per_head(model, inp, layer=1)
    hull_out = _hull_attention(q, k, v)
    ref_out = _reference_hard_max_attention(q, k, v)
    assert torch.equal(hull_out, ref_out), \
        "hull attention diverges from reference on read_by_key layer 1"


if __name__ == "__main__":
    test_hull_matches_reference_retrieve_by_index()
    print("[ok] hull attention matches reference on retrieve_by_index (layer 0)")
    test_hull_matches_reference_read_by_key()
    print("[ok] hull attention matches reference on read_by_key (layer 1)")
