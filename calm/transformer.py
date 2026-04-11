"""
Minimal reference transformer forward pass for CALM-compiled weights.

Pure NumPy, no layer norm, no batching, no KV cache. Designed to be
read end-to-end in five minutes. If you're looking for performance,
you are in the wrong file — this exists to verify correctness of
hand-compiled weight dictionaries, not to serve traffic.

Weight dict schema (all np.float32 unless noted):

    tok_embed  : (V, D)              token embedding lookup table
    pos_embed  : (T_max, D)          position embedding (may be all zero
                                     if the compiler writes positions
                                     into the residual stream via
                                     tok_embed slots instead)

    W_Q        : (H, D, D_head)      per-head query projection
    W_K        : (H, D, D_head)      per-head key projection
    W_V        : (H, D, D_head)      per-head value projection
    W_O        : (H * D_head, D)     output projection for attention
    attn_scale : float               softmax temperature (1/sqrt(D_head)
                                     by convention, but the compiler
                                     may override to sharpen lookup)

    W_ffn1     : (D, D_ffn)          FFN first linear
    b_ffn1     : (D_ffn,)            FFN first bias
    W_ffn2     : (D_ffn, D)          FFN second linear
    b_ffn2     : (D,)                FFN second bias

    W_out      : (D, V)              output projection to vocab logits
    b_out      : (V,)                output bias

The forward pass is a single transformer block (1 attention + 1 FFN)
followed by the output projection. Add more blocks here if a program
needs them — the compiler is expected to produce a compatible weights
dict shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int          # residual stream dimension
    d_head: int           # per-head attention dimension
    n_heads: int
    d_ffn: int            # hidden dimension of the FFN
    max_seq_len: int


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def forward(
    tokens: np.ndarray,           # (T,) int ids
    cfg: TransformerConfig,
    weights: Dict[str, np.ndarray],
) -> np.ndarray:                   # returns (T, vocab_size) logits
    """
    One forward pass: embedding → 1 block → output projection.

    Returns logits for every position. Callers sample/argmax whichever
    position they care about (usually the last one).
    """
    T = tokens.shape[0]
    assert T <= cfg.max_seq_len

    # --- embedding ---------------------------------------------------
    x = weights["tok_embed"][tokens]                         # (T, D)
    x = x + weights["pos_embed"][:T]                         # (T, D)

    # --- attention (causal) -----------------------------------------
    H, D, Dh = cfg.n_heads, cfg.d_model, cfg.d_head

    # Per-head projections.
    Q = np.einsum("td,hde->hte", x, weights["W_Q"])          # (H, T, Dh)
    K = np.einsum("td,hde->hte", x, weights["W_K"])          # (H, T, Dh)
    V = np.einsum("td,hde->hte", x, weights["W_V"])          # (H, T, Dh)

    # Scaled dot-product attention with a causal mask.
    scores = np.einsum("htd,hsd->hts", Q, K) * weights["attn_scale"]  # (H, T, T)
    causal = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(causal[None, :, :], scores, -1e9)
    attn = _softmax(scores, axis=-1)                          # (H, T, T)
    head_out = np.einsum("hts,hsd->htd", attn, V)             # (H, T, Dh)

    # Concatenate heads and project back to residual dim.
    concat = head_out.transpose(1, 0, 2).reshape(T, H * Dh)   # (T, H*Dh)
    attn_out = concat @ weights["W_O"]                        # (T, D)
    x = x + attn_out                                          # residual

    # --- FFN ---------------------------------------------------------
    hidden = np.maximum(0.0, x @ weights["W_ffn1"] + weights["b_ffn1"])  # ReLU
    ffn_out = hidden @ weights["W_ffn2"] + weights["b_ffn2"]  # (T, D)
    x = x + ffn_out                                           # residual

    # --- output projection ------------------------------------------
    logits = x @ weights["W_out"] + weights["b_out"]          # (T, V)
    return logits


def generate(
    prompt: np.ndarray,           # (T,) int ids
    max_new_tokens: int,
    cfg: TransformerConfig,
    weights: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Autoregressive greedy decode. Returns the full sequence (prompt +
    generated tokens). No sampling — this is for deterministic
    compiled programs, not stochastic language models.
    """
    seq = list(prompt.tolist())
    for _ in range(max_new_tokens):
        logits = forward(np.asarray(seq, dtype=np.int64), cfg, weights)
        next_token = int(np.argmax(logits[-1]))
        seq.append(next_token)
    return np.asarray(seq, dtype=np.int64)


# --- smoke test: random weights should not crash --------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfg = TransformerConfig(
        vocab_size=12, d_model=24, d_head=3, n_heads=2, d_ffn=32, max_seq_len=16
    )
    W = {
        "tok_embed": rng.standard_normal((cfg.vocab_size, cfg.d_model)).astype(np.float32),
        "pos_embed": np.zeros((cfg.max_seq_len, cfg.d_model), dtype=np.float32),
        "W_Q": rng.standard_normal((cfg.n_heads, cfg.d_model, cfg.d_head)).astype(np.float32),
        "W_K": rng.standard_normal((cfg.n_heads, cfg.d_model, cfg.d_head)).astype(np.float32),
        "W_V": rng.standard_normal((cfg.n_heads, cfg.d_model, cfg.d_head)).astype(np.float32),
        "W_O": rng.standard_normal((cfg.n_heads * cfg.d_head, cfg.d_model)).astype(np.float32),
        "attn_scale": np.float32(1.0),
        "W_ffn1": rng.standard_normal((cfg.d_model, cfg.d_ffn)).astype(np.float32),
        "b_ffn1": np.zeros(cfg.d_ffn, dtype=np.float32),
        "W_ffn2": rng.standard_normal((cfg.d_ffn, cfg.d_model)).astype(np.float32),
        "b_ffn2": np.zeros(cfg.d_model, dtype=np.float32),
        "W_out": rng.standard_normal((cfg.d_model, cfg.vocab_size)).astype(np.float32),
        "b_out": np.zeros(cfg.vocab_size, dtype=np.float32),
    }
    out = forward(np.array([0, 10, 3, 11], dtype=np.int64), cfg, W)
    print("smoke test logits shape:", out.shape, "dtype:", out.dtype)
    assert out.shape == (4, 12)
    print("OK")
