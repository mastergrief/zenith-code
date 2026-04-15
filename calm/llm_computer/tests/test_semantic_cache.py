"""Tests for semantic cache."""

from __future__ import annotations

import torch

from calm.llm_computer.semantic_cache import (
    CacheStats, SemanticCache, cached_forward, fingerprint,
)


def test_fingerprint_deterministic():
    x = torch.tensor([1.5, 2.3, -0.7])
    a = fingerprint(x)
    b = fingerprint(x.clone())
    assert a == b


def test_fingerprint_differs_for_different_inputs():
    a = fingerprint(torch.tensor([1.0, 2.0]))
    b = fingerprint(torch.tensor([1.0, 2.1]))
    assert a != b


def test_fingerprint_distinguishes_layers():
    x = torch.tensor([1.0, 2.0])
    a = fingerprint(x, layer_idx=0)
    b = fingerprint(x, layer_idx=1)
    assert a != b


def test_cache_hit_on_repeat():
    cache = SemanticCache(max_entries=4)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = torch.tensor([[0.5, 0.6, 0.7]])
    cache.put(x, out)
    retrieved = cache.get(x)
    assert retrieved is not None
    assert torch.equal(retrieved, out)
    assert cache.stats().hits == 1
    assert cache.stats().misses == 0


def test_cache_miss_on_first_lookup():
    cache = SemanticCache()
    x = torch.tensor([[1.0, 2.0]])
    assert cache.get(x) is None
    assert cache.stats().misses == 1
    assert cache.stats().hits == 0


def test_cache_evicts_lru():
    cache = SemanticCache(max_entries=2)
    x1 = torch.tensor([[1.0]])
    x2 = torch.tensor([[2.0]])
    x3 = torch.tensor([[3.0]])
    cache.put(x1, torch.tensor([[10.0]]))
    cache.put(x2, torch.tensor([[20.0]]))
    # Access x1 → move to end (x2 becomes LRU)
    _ = cache.get(x1)
    cache.put(x3, torch.tensor([[30.0]]))
    # x2 should be evicted (was LRU), x1 and x3 should remain
    assert cache.get(x2) is None
    assert cache.get(x1) is not None
    assert cache.get(x3) is not None
    assert cache.stats().evictions == 1


def test_hit_rate():
    cache = SemanticCache()
    cache.put(torch.tensor([1.0]), torch.tensor([2.0]))
    cache.get(torch.tensor([1.0]))     # hit
    cache.get(torch.tensor([1.0]))     # hit
    cache.get(torch.tensor([99.0]))    # miss
    assert cache.stats().hit_rate == 2 / 3


def test_cached_forward_calls_once():
    """Second invocation with same input must NOT call compute_fn."""
    cache = SemanticCache()
    call_count = [0]

    def compute(x):
        call_count[0] += 1
        return x * 2

    x = torch.tensor([[1.0, 2.0]])
    out1 = cached_forward(cache, 0, x, compute)
    out2 = cached_forward(cache, 0, x, compute)
    assert call_count[0] == 1
    assert torch.equal(out1, out2)


def test_cached_forward_recomputes_different_input():
    cache = SemanticCache()
    call_count = [0]
    def compute(x):
        call_count[0] += 1
        return x + 1
    cached_forward(cache, 0, torch.tensor([[1.0]]), compute)
    cached_forward(cache, 0, torch.tensor([[2.0]]), compute)
    assert call_count[0] == 2


def test_cache_returns_detached_clone():
    """Modifying returned tensor must not corrupt cache entry."""
    cache = SemanticCache()
    x = torch.tensor([[1.0]])
    out = torch.tensor([[5.0]])
    cache.put(x, out)
    retrieved = cache.get(x)
    retrieved[0, 0] = 999.0  # modify
    # Cache entry should still be 5.0
    again = cache.get(x)
    assert again[0, 0].item() == 5.0


def test_clear_resets_size_not_stats():
    cache = SemanticCache()
    cache.put(torch.tensor([1.0]), torch.tensor([2.0]))
    cache.get(torch.tensor([1.0]))  # 1 hit
    cache.clear()
    assert cache.stats().size == 0
    # Lifetime hit count preserved
    assert cache.stats().hits == 1


def test_precision_rounding_for_float_fuzziness():
    """Small float drift should hit the same fingerprint at low precision."""
    cache = SemanticCache(precision=2)
    x1 = torch.tensor([[1.234567]])
    x2 = torch.tensor([[1.234999]])  # within 0.01
    cache.put(x1, torch.tensor([[1.0]]))
    # At precision=2, both round to same int → hit
    retrieved = cache.get(x2)
    assert retrieved is not None


if __name__ == "__main__":
    test_fingerprint_deterministic()
    print("[ok] fingerprint deterministic")
    test_fingerprint_differs_for_different_inputs()
    print("[ok] different inputs = different fingerprints")
    test_fingerprint_distinguishes_layers()
    print("[ok] per-layer fingerprints")
    test_cache_hit_on_repeat()
    print("[ok] repeat access hits cache")
    test_cache_miss_on_first_lookup()
    print("[ok] first access misses")
    test_cache_evicts_lru()
    print("[ok] LRU eviction")
    test_hit_rate()
    print("[ok] hit_rate computation")
    test_cached_forward_calls_once()
    print("[ok] cached_forward avoids recomputation")
    test_cached_forward_recomputes_different_input()
    print("[ok] cached_forward recomputes new inputs")
    test_cache_returns_detached_clone()
    print("[ok] returned tensor is detached clone")
    test_clear_resets_size_not_stats()
    print("[ok] clear resets size, preserves lifetime stats")
    test_precision_rounding_for_float_fuzziness()
    print("[ok] precision rounding tolerates float drift")
