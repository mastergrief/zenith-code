"""HRM-Text-1.58 single-row (B=1) KV cache for inference/probe decode.

Per codex +1 γ1 implement at msg 1779530833485-eb9296ca. Companion design
proposal at msg 1779530825108-86d50e8a closes the recurrence-aliasing hole
codex flagged at msg 1779530491974-01905b3c.

Cache contract (load-bearing):
- Keys: (level: "L"|"H", rec_idx: int, layer_idx: int).
- Each HRM forward visits 32 distinct keys per row:
  L: H_cycles × L_cycles iterations × n_layers_per_level (typically 6×4=24)
  H: H_cycles iterations × n_layers_per_level (typically 2×4=8)
- Buffers store post-RoPE, transposed K/V: (B=1, num_kv_heads, max_seq, head_dim).
- Runtime-only object; NOT registered as buffer/parameter; never in state_dict.
- Resets between rows; never persists across decode sessions.

Mask semantics (corrected per codex msg 1779530776100-9a914271):
- Prefill: caller passes full prompt; cache stores full K/V for length S;
  caller applies existing PrefixLM mask exactly.
- Single-token decode: caller passes 1 new position; cache appends K/V at
  length S → S+1; caller uses attn_mask=None, is_causal=False (cache
  truncation already enforces causality — no future K/V are ever stored).
- `is_causal=True` with non-square q_len=1, k_len=S+1 is BANNED because
  SDPA aligns it to query-local index 0 (Q[0] would attend only to K[0]).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


CacheKey = tuple[str, int, int]  # (level, rec_idx, layer_idx)


@dataclass
class _CacheBuffer:
    """Per-key K/V storage + current valid length."""
    K: Tensor  # (1, num_kv_heads, max_seq, head_dim)
    V: Tensor
    length: int


class KVCache:
    """Single-row KV cache keyed by (level, rec_idx, layer_idx).

    Lazy allocation: a buffer is materialized on first `update()` for a given
    key. This avoids paying memory for unused cycles if recurrence depth is
    smaller than the configured max.

    Memory at HRM-Text-1.58 defaults (max_seq=384, num_kv_heads=4,
    head_dim=128, fp32):
      Per buffer: K = 4*384*128*4 bytes = 768 KB; V same; total 1.5 MB.
      32 buffers (24 L + 8 H) = 48 MB total.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> None:
        self._max_seq = max_seq_len
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._device = device
        self._buffers: dict[CacheKey, _CacheBuffer] = {}
        # Test/diagnostic: per-call access log of (level, rec_idx, layer_idx).
        # Reset by `reset()` along with buffers.
        self._access_log: list[CacheKey] = []

    @property
    def max_seq_len(self) -> int:
        return self._max_seq

    def reset(self) -> None:
        """Clear all buffers and access log. Call between rows / new prompts."""
        self._buffers.clear()
        self._access_log.clear()

    def update(
        self,
        level: str,
        rec_idx: int,
        layer_idx: int,
        new_k: Tensor,
        new_v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new_k, new_v at the current length for this key; return full (K, V).

        Args:
            level: "L" or "H" (which RecurrentBlock).
            rec_idx: recurrence iteration index within the HRM forward.
            layer_idx: layer index inside the level's Transformer.
            new_k, new_v: shape (1, num_kv_heads, S_new, head_dim). For prefill,
                S_new = full prompt length; for decode, S_new = 1.

        Returns:
            (K_full, V_full): views of shape (1, num_kv_heads, length, head_dim)
            where length = previous_length + S_new.

        Raises:
            RuntimeError: if appending overflows max_seq_len.
            ValueError: on shape mismatches or invalid arguments.
        """
        if level not in ("L", "H"):
            raise ValueError(f"level must be 'L' or 'H', got {level!r}")
        if new_k.shape != new_v.shape:
            raise ValueError(
                f"new_k.shape {new_k.shape} != new_v.shape {new_v.shape}"
            )
        if new_k.dim() != 4:
            raise ValueError(
                f"expected new_k.dim()==4 (B, num_kv_heads, S_new, head_dim), "
                f"got {new_k.dim()}"
            )
        B, H, S_new, D = new_k.shape
        if B != 1:
            raise ValueError(
                f"γ1 KVCache is single-row only (B=1); got B={B}. "
                "Batched decode is γ2 (deferred)."
            )
        if H != self._num_kv_heads:
            raise ValueError(
                f"num_kv_heads mismatch: cache={self._num_kv_heads}, new_k={H}"
            )
        if D != self._head_dim:
            raise ValueError(
                f"head_dim mismatch: cache={self._head_dim}, new_k={D}"
            )

        key: CacheKey = (level, rec_idx, layer_idx)
        self._access_log.append(key)

        if key not in self._buffers:
            shape = (1, self._num_kv_heads, self._max_seq, self._head_dim)
            self._buffers[key] = _CacheBuffer(
                K=torch.zeros(shape, dtype=self._dtype, device=self._device),
                V=torch.zeros(shape, dtype=self._dtype, device=self._device),
                length=0,
            )

        buf = self._buffers[key]
        S_start = buf.length
        S_end = S_start + S_new
        if S_end > self._max_seq:
            raise RuntimeError(
                f"KVCache overflow at key={key}: requested length {S_end} "
                f"exceeds max_seq_len={self._max_seq}"
            )
        buf.K[:, :, S_start:S_end, :] = new_k
        buf.V[:, :, S_start:S_end, :] = new_v
        buf.length = S_end
        return buf.K[:, :, :S_end, :], buf.V[:, :, :S_end, :]

    def get_buffer(self, level: str, rec_idx: int, layer_idx: int):
        """Return (K_view, V_view, length) for the given key, or None if unset.

        Read-only access for tests / diagnostics. Does NOT log access.
        """
        key: CacheKey = (level, rec_idx, layer_idx)
        if key not in self._buffers:
            return None
        buf = self._buffers[key]
        L = buf.length
        return buf.K[:, :, :L, :], buf.V[:, :, :L, :], L

    def get_access_log(self) -> list[CacheKey]:
        """Return a copy of the access log for test inspection."""
        return list(self._access_log)

    def num_buffers(self) -> int:
        return len(self._buffers)

    def total_memory_bytes(self) -> int:
        """Sum of allocated buffer bytes (K + V, all keys)."""
        total = 0
        for buf in self._buffers.values():
            total += buf.K.numel() * buf.K.element_size()
            total += buf.V.numel() * buf.V.element_size()
        return total
