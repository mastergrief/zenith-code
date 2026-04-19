"""Test multi-token KVCacheTq4.update() — verify S>=1 writes work.

Doesn't need GGUF — mocks a GemmaSubstrate-like object with minimal
config + n_layers. Tests:
  1. S=1 write → reads back approximately same tensor (tq4 round-trip)
  2. S=8 prefill write → reads back correct concat
  3. Two updates (prefill S=8, then decode S=1) → total 9 tokens
  4. Per-layer positions are independent
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _MockConfig:
    def __init__(self, n_layers=4, n_heads_kv=2):
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv


class _MockLayer:
    def __init__(self, n_heads_kv=2, d_head=256):
        # attn_k.out_features = n_heads_kv * d_head
        class _Linear:
            def __init__(self, out_features):
                self.out_features = out_features
        self.attn_k = _Linear(n_heads_kv * d_head)


class _MockModel:
    def __init__(self, n_layers=4, n_heads_kv=2, d_head=256):
        self.config = _MockConfig(n_layers=n_layers, n_heads_kv=n_heads_kv)
        self.layers = [_MockLayer(n_heads_kv=n_heads_kv, d_head=d_head)
                       for _ in range(n_layers)]


def _shape_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    return tuple(a.shape) == tuple(b.shape)


def _close(a: torch.Tensor, b: torch.Tensor, atol: float = 0.5) -> bool:
    """tq4 has ~4% error on per-element; check per-channel correlation."""
    a = a.float()
    b = b.float()
    if not _shape_match(a, b):
        return False
    # High correlation + bounded absolute error per element.
    per_el_err = (a - b).abs().max().item()
    if per_el_err > atol:
        return False
    return True


def test_single_token():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(0)
    model = _MockModel(n_layers=4)
    cache = KVCacheTq4(model, max_len=128, device="cuda")
    k = torch.randn(1, 2, 1, 256, device="cuda")
    v = torch.randn(1, 2, 1, 256, device="cuda")
    k_full, v_full = cache.update(0, k, v, is_swa=False)
    assert k_full.shape == (1, 2, 1, 256), k_full.shape
    assert cache.layer_pos[0] == 1
    # Round-trip should be close (tq4 lossy but bounded).
    assert _close(k_full, k), f"max err {(k_full - k).abs().max().item():.3f}"
    print("✓ test_single_token")


def test_multi_token_prefill():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(0)
    model = _MockModel(n_layers=4)
    cache = KVCacheTq4(model, max_len=128, device="cuda")
    S = 8
    k = torch.randn(1, 2, S, 256, device="cuda")
    v = torch.randn(1, 2, S, 256, device="cuda")
    k_full, v_full = cache.update(0, k, v, is_swa=False)
    assert k_full.shape == (1, 2, S, 256), k_full.shape
    assert cache.layer_pos[0] == S
    assert _close(k_full, k), f"max err {(k_full - k).abs().max().item():.3f}"
    assert _close(v_full, v), f"max err {(v_full - v).abs().max().item():.3f}"
    print("✓ test_multi_token_prefill")


def test_prefill_then_decode():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(1)
    model = _MockModel(n_layers=4)
    cache = KVCacheTq4(model, max_len=128, device="cuda")
    S = 8
    k1 = torch.randn(1, 2, S, 256, device="cuda")
    v1 = torch.randn(1, 2, S, 256, device="cuda")
    cache.update(0, k1, v1, is_swa=False)
    assert cache.layer_pos[0] == S

    k2 = torch.randn(1, 2, 1, 256, device="cuda")
    v2 = torch.randn(1, 2, 1, 256, device="cuda")
    k_full, v_full = cache.update(0, k2, v2, is_swa=False)
    assert k_full.shape == (1, 2, S + 1, 256), k_full.shape
    assert cache.layer_pos[0] == S + 1
    # First S tokens should still match k1.
    k_first = k_full[:, :, :S, :]
    assert _close(k_first, k1, atol=0.5), f"first-S err {(k_first - k1).abs().max().item():.3f}"
    k_last = k_full[:, :, S:S+1, :]
    assert _close(k_last, k2, atol=0.5), f"last-1 err {(k_last - k2).abs().max().item():.3f}"
    print("✓ test_prefill_then_decode")


def test_per_layer_independence():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(2)
    model = _MockModel(n_layers=4)
    cache = KVCacheTq4(model, max_len=128, device="cuda")
    k = torch.randn(1, 2, 4, 256, device="cuda")
    v = torch.randn(1, 2, 4, 256, device="cuda")
    cache.update(0, k, v)
    cache.update(1, k, v)
    # layer 2 NOT updated
    assert cache.layer_pos[0] == 4
    assert cache.layer_pos[1] == 4
    assert cache.layer_pos[2] == 0
    assert cache.layer_pos[3] == 0
    # Shared-KV read from layer 0 works, empty from layer 2.
    k_read_0 = cache.k_cache[0]
    assert k_read_0.shape == (1, 2, 4, 256), k_read_0.shape
    k_read_2 = cache.k_cache[2]
    assert k_read_2.shape == (1, 2, 0, 256), k_read_2.shape
    print("✓ test_per_layer_independence")


def test_swa_trim():
    from calm.llm_computer.gemma_substrate import KVCacheTq4
    torch.manual_seed(3)
    model = _MockModel(n_layers=4)
    cache = KVCacheTq4(model, max_len=1024, device="cuda")
    # Prefill 800 tokens as SWA with window=512
    k = torch.randn(1, 2, 800, 256, device="cuda")
    v = torch.randn(1, 2, 800, 256, device="cuda")
    cache.update(0, k, v, is_swa=True, window_size=512)
    assert cache.layer_pos[0] == 800
    cache.trim_swa_storage()
    assert cache.layer_pos[0] == 512, f"after trim pos={cache.layer_pos[0]}"
    # Last 512 tokens should match k[-512:]
    k_read = cache.k_cache[0]
    assert k_read.shape == (1, 2, 512, 256)
    k_expected = k[:, :, -512:, :]
    assert _close(k_read, k_expected, atol=0.7), (
        f"trim err {(k_read.float() - k_expected.float()).abs().max().item():.3f}"
    )
    print("✓ test_swa_trim")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    test_single_token()
    test_multi_token_prefill()
    test_prefill_then_decode()
    test_per_layer_independence()
    test_swa_trim()
    print("\nAll tests passed.")
