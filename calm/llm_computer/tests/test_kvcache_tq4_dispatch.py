"""CPU dispatch tests for KVCacheTq4.write_only — the path the Phase 2
fused tq4 flash-attn integration uses to skip the eager dequant in
KVCacheTq4.update.

write_only must:
  - write the same tq4 bytes update() would have written
  - advance layer_pos identically
  - invalidate the memo (next dequant rebuilds against new bytes)
  - return None (signals to caller: bytes only, no fp32 materialization)
"""

from __future__ import annotations

import torch


class _Cfg:
    def __init__(self):
        self.n_layers = 4
        self.n_heads_kv = 2


class _Lin:
    out_features = 2 * 256


class _Lay:
    attn_k = _Lin()


class _M:
    config = _Cfg()
    layers = [_Lay() for _ in range(4)]


def test_write_only_returns_none():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    cache = KVCacheTq4(_M(), max_len=64, device="cpu")
    k = torch.randn(1, 2, 1, 256)
    v = torch.randn(1, 2, 1, 256)
    out = cache.write_only(0, k, v, is_swa=False)
    assert out is None, "write_only must not return fp32 K/V"
    assert cache.layer_pos[0] == 1


def test_write_only_writes_same_bytes_as_update():
    """write_only and update must produce byte-identical cache state."""
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(0)
    k = torch.randn(1, 2, 4, 256)
    v = torch.randn(1, 2, 4, 256)

    cache_a = KVCacheTq4(_M(), max_len=64, device="cpu")
    cache_a.write_only(0, k, v)

    cache_b = KVCacheTq4(_M(), max_len=64, device="cpu")
    cache_b.update(0, k, v)

    assert torch.equal(cache_a.k_qs[0], cache_b.k_qs[0]), (
        "write_only and update produce different K bytes")
    assert torch.equal(cache_a.k_d[0], cache_b.k_d[0])
    assert torch.equal(cache_a.v_qs[0], cache_b.v_qs[0])
    assert torch.equal(cache_a.v_d[0], cache_b.v_d[0])
    assert cache_a.layer_pos[0] == cache_b.layer_pos[0] == 4


def test_write_only_invalidates_memo():
    """If memo holds a stale dequant, write_only must invalidate it so the
    next read sees the new bytes."""
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    cache = KVCacheTq4(_M(), max_len=64, device="cpu")
    cache.update(0, torch.randn(1, 2, 4, 256), torch.randn(1, 2, 4, 256))
    pre = cache.k_cache[0]   # populates memo with pos=4
    assert ("k", 0) in cache._memo

    cache.write_only(0, torch.randn(1, 2, 1, 256),
                     torch.randn(1, 2, 1, 256))
    # Memo entry must be gone; next read rebuilds at pos=5
    assert ("k", 0) not in cache._memo
    post = cache.k_cache[0]
    assert post.shape == (1, 2, 5, 256)
    assert post is not pre  # new tensor, different sequence length


def test_write_only_multi_token_prefill():
    """Prefill writes S>=1 tokens via write_only; bytes must match
    update()'s write for the same input."""
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(1)
    k = torch.randn(1, 2, 8, 256)
    v = torch.randn(1, 2, 8, 256)

    cache_a = KVCacheTq4(_M(), max_len=64, device="cpu")
    cache_a.write_only(0, k, v)

    cache_b = KVCacheTq4(_M(), max_len=64, device="cpu")
    cache_b.update(0, k, v)

    # Per-byte equality across the entire prefill region
    assert torch.equal(cache_a.k_qs[0][:, :8, :], cache_b.k_qs[0][:, :8, :])
    assert torch.equal(cache_a.v_d[0][:, :8], cache_b.v_d[0][:, :8])
