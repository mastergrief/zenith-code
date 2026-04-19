"""Unit tests for the per-step dequant memo introduced in Phase 1.

CPU-only — uses the same _MockModel pattern as scripts/test_kvcache_tq4_multitoken.py.
The memo cuts shared-KV consumer reads from O(N) per layer to O(1) per
source layer per decode step, which is the actual perf fix behind the
R53.33 USE_TQ4_KV revert. These tests pin the memoization contract so a
future refactor can't silently break it.
"""

from __future__ import annotations

import pytest
import torch


class _MockConfig:
    def __init__(self, n_layers: int = 4, n_heads_kv: int = 2):
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv


class _MockLinear:
    def __init__(self, out_features: int):
        self.out_features = out_features


class _MockLayer:
    def __init__(self, n_heads_kv: int = 2, d_head: int = 256):
        self.attn_k = _MockLinear(n_heads_kv * d_head)


class _MockModel:
    def __init__(self, n_layers: int = 4, n_heads_kv: int = 2,
                 d_head: int = 256):
        self.config = _MockConfig(n_layers=n_layers, n_heads_kv=n_heads_kv)
        self.layers = [_MockLayer(n_heads_kv=n_heads_kv, d_head=d_head)
                       for _ in range(n_layers)]


def _make_cache(max_len: int = 32):
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    model = _MockModel(n_layers=4)
    return KVCacheTq4(model, max_len=max_len, device="cpu")


def test_update_returns_same_tensor_as_proxy_read():
    """update() and kv_cache.k_cache[il] must hit the same memo entry —
    they're called on opposite sides of the own-vs-shared-KV split in
    _forward_layer but should return the exact tensor on the same step."""
    torch.manual_seed(0)
    cache = _make_cache()
    k = torch.randn(1, 2, 1, 256)
    v = torch.randn(1, 2, 1, 256)
    k_full_from_update, _ = cache.update(0, k, v)
    k_from_proxy = cache.k_cache[0]
    assert k_full_from_update is k_from_proxy, (
        "update() return and proxy read must share the memoized tensor")


def test_repeated_proxy_reads_hit_memo():
    """Shared-KV layers in _forward_layer call kv_cache.k_cache[kv_src] once
    per consumer layer (18 times for layers 24-41 reading layer 17). All
    those calls must share one cached fp32 tensor."""
    torch.manual_seed(0)
    cache = _make_cache()
    cache.update(0, torch.randn(1, 2, 4, 256), torch.randn(1, 2, 4, 256))
    a = cache.k_cache[0]
    b = cache.k_cache[0]
    c = cache.v_cache[0]
    d = cache.v_cache[0]
    assert a is b, "consecutive k reads should be memo hits"
    assert c is d, "consecutive v reads should be memo hits"
    # k and v memos are independent
    assert a is not c


def test_update_invalidates_memo():
    """After a second update() on the same layer, the memo must point at the
    NEW dequant, not the stale prefix-only one."""
    torch.manual_seed(0)
    cache = _make_cache()
    cache.update(0, torch.randn(1, 2, 4, 256), torch.randn(1, 2, 4, 256))
    k_after_first = cache.k_cache[0]
    assert k_after_first.shape[2] == 4

    cache.update(0, torch.randn(1, 2, 1, 256), torch.randn(1, 2, 1, 256))
    k_after_second = cache.k_cache[0]
    assert k_after_second.shape[2] == 5
    assert k_after_first is not k_after_second


def test_clear_drops_all_memo_entries():
    torch.manual_seed(0)
    cache = _make_cache()
    cache.update(0, torch.randn(1, 2, 4, 256), torch.randn(1, 2, 4, 256))
    cache.update(1, torch.randn(1, 2, 4, 256), torch.randn(1, 2, 4, 256))
    _ = cache.k_cache[0]
    _ = cache.v_cache[1]
    assert len(cache._memo) > 0
    cache.clear()
    assert len(cache._memo) == 0
    assert all(p == 0 for p in cache.layer_pos)


def test_trim_swa_invalidates_memo():
    """After trim_swa_storage the bytes 0..keep_blocks now hold a different
    sequence; old memo entries pointing at the pre-trim dequant must be
    dropped."""
    torch.manual_seed(0)
    cache = _make_cache(max_len=128)
    # Write 100 tokens to layer 0 marked SWA with window=64
    cache.update(0, torch.randn(1, 2, 100, 256), torch.randn(1, 2, 100, 256),
                 is_swa=True, window_size=64)
    pre_trim = cache.k_cache[0]
    assert pre_trim.shape[2] == 100
    cache.trim_swa_storage()
    assert cache.layer_pos[0] == 64
    post_trim = cache.k_cache[0]
    assert post_trim.shape[2] == 64
    # Different tensor object — memo was rebuilt against trimmed bytes.
    assert post_trim is not pre_trim


def test_empty_layer_returns_zero_length_tensor():
    """Layers that haven't been written must return (1, n_kv_h, 0, d_head)
    — needed by GQA expand and shared-KV reads on cold layers."""
    cache = _make_cache()
    out = cache.k_cache[0]
    assert out.shape == (1, 2, 0, 256)
    assert out.dtype == torch.float32


def test_memo_round_trip_matches_full_dequant():
    """Memo must produce bit-identical output to a fresh full-prefix dequant
    (no caching effects on values, only on whether we recompute)."""
    from calm.llm_computer.tq4_torch import Tq4Tensor, dequantize_tq4
    torch.manual_seed(0)
    cache = _make_cache()
    k = torch.randn(1, 2, 8, 256)
    v = torch.randn(1, 2, 8, 256)
    cache.update(0, k, v)
    memo_k = cache.k_cache[0]

    # Recompute by hand from the raw bytes, no memo path. Phase 2 head-major:
    # k_qs[0] is (n_kv_h, max_len*bpr, 128); slice per-head [:, :pos*bpr, :].
    bpr = cache._bpr[0]
    n_kv_h = cache.cfg.n_heads_kv
    pos = 8
    qs = cache.k_qs[0][:, :pos * bpr, :].contiguous().reshape(-1, 128)
    d = cache.k_d[0][:, :pos * bpr].contiguous().reshape(-1)
    flat = dequantize_tq4(Tq4Tensor(
        qs=qs, d=d, shape=(qs.shape[0] * 256,),
    ), pi=cache._pi, centroids=cache._centroids)
    fresh_k = flat.reshape(n_kv_h, pos, 256).unsqueeze(0).contiguous().float()

    assert torch.equal(memo_k, fresh_k), (
        "memoized dequant diverges from fresh dequant — memo is leaking state")
