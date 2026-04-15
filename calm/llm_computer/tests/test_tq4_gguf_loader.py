"""Tests for tq4 GGUF loader — reads our Gemma 4 E4B tq4 file directly."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.tq4_gguf_loader import (
    extract_fp_tensor, extract_tq4_tensor, ggml_gemma_tensor_map,
    patch_gguf_for_turboquant, read_turboquant_gguf, summarize_gguf,
)
from calm.llm_computer.tq4_torch import Tq4Tensor


GGUF_PATH = Path(
    os.environ.get(
        "ZENITH_GEMMA_GGUF",
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
    )
)
GGUF_AVAILABLE = GGUF_PATH.exists()


def test_patch_is_idempotent():
    patch_gguf_for_turboquant()
    patch_gguf_for_turboquant()  # second call must not crash


def test_gemma_tensor_map_structure():
    m = ggml_gemma_tensor_map(n_layers=2)
    assert "_meta" in m
    assert "token_embd" in m["_meta"]
    assert m["_meta"]["token_embd"] == "token_embd.weight"
    for i in range(2):
        layer = m[f"layer_{i}"]
        assert layer["q"] == f"blk.{i}.attn_q.weight"
        assert layer["k"] == f"blk.{i}.attn_k.weight"
        assert layer["ffn_norm"] == f"blk.{i}.ffn_norm.weight"


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_read_real_gguf():
    reader = read_turboquant_gguf(GGUF_PATH)
    assert len(reader.tensors) > 0


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_summarize_reports_gemma4_structure():
    reader = read_turboquant_gguf(GGUF_PATH)
    summary = summarize_gguf(reader)
    assert summary["n_tensors"] > 300
    assert summary["type_counts"].get("TQ4_K256", 0) > 0
    # Gemma 4 E4B has 42 blocks
    meta = summary["metadata"]
    for key, val in meta.items():
        if "block_count" in key:
            assert val == 42


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_extract_tq4_first_layer_q_proj():
    """Load blk.0.attn_q.weight from the real Gemma 4 E4B GGUF."""
    reader = read_turboquant_gguf(GGUF_PATH)
    q = extract_tq4_tensor(reader, "blk.0.attn_q.weight")
    assert isinstance(q, Tq4Tensor)
    # From our probe: shape is (2560, 2048) so 2560*2048 = 5,242,880 elements
    # = 5242880 / 256 = 20480 blocks
    assert q.qs.shape == (20480, 128)
    assert q.d.shape == (20480,)
    assert q.qs.dtype == torch.uint8
    assert q.d.dtype == torch.float32
    # L2 norms should be finite and positive-ish
    assert torch.isfinite(q.d).all()


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_extract_tq4_all_layer_projections_shapes():
    """Every tq4 projection in the first 3 layers should load cleanly."""
    reader = read_turboquant_gguf(GGUF_PATH)
    for layer in range(3):
        for tensor_name in [
            f"blk.{layer}.attn_q.weight",
            f"blk.{layer}.attn_k.weight",
            f"blk.{layer}.attn_v.weight",
            f"blk.{layer}.attn_output.weight",
            f"blk.{layer}.ffn_gate.weight",
            f"blk.{layer}.ffn_up.weight",
            f"blk.{layer}.ffn_down.weight",
        ]:
            q = extract_tq4_tensor(reader, tensor_name)
            assert isinstance(q, Tq4Tensor)
            assert q.qs.shape[1] == 128
            assert q.qs.shape[0] == q.d.shape[0]


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_extract_tq4_dequantize_produces_finite_output():
    """Loaded tq4 tensor must dequantize to finite values."""
    from calm.llm_computer.tq4_torch import dequantize_tq4, build_pi
    reader = read_turboquant_gguf(GGUF_PATH)
    q = extract_tq4_tensor(reader, "blk.0.attn_q.weight")
    # Use the C-reference Pi for bit-exact dequant match with GGUF
    pi = build_pi(source="c_header")
    fp = dequantize_tq4(q, pi=pi)
    assert fp.shape == (2560, 2048)
    assert torch.isfinite(fp).all(), "dequant produced non-finite values"
    # Weights should have reasonable magnitude (not all zero)
    assert fp.abs().mean().item() > 1e-6


@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_extract_f32_norm_tensor():
    """Load an F32 norm tensor (no tq4 unpacking needed)."""
    reader = read_turboquant_gguf(GGUF_PATH)
    n = extract_fp_tensor(reader, "blk.0.attn_norm.weight")
    assert n.shape == (2560,)
    assert n.dtype in (torch.float32, torch.float16)


def test_extract_unknown_tensor_raises():
    """Without GGUF access, verify the error path."""
    class FakeReader:
        tensors = []
    with pytest.raises(KeyError, match="not in GGUF"):
        extract_tq4_tensor(FakeReader(), "nonexistent")


if __name__ == "__main__":
    test_patch_is_idempotent()
    print("[ok] patch idempotent")
    test_gemma_tensor_map_structure()
    print("[ok] tensor map structure")
    test_extract_unknown_tensor_raises()
    print("[ok] unknown tensor raises")
    if GGUF_AVAILABLE:
        test_read_real_gguf()
        print("[ok] read real GGUF")
        test_summarize_reports_gemma4_structure()
        print("[ok] summary reports Gemma 4 structure")
        test_extract_tq4_first_layer_q_proj()
        print("[ok] extract first layer Q projection")
        test_extract_tq4_all_layer_projections_shapes()
        print("[ok] extract all projection shapes (3 layers)")
        test_extract_tq4_dequantize_produces_finite_output()
        print("[ok] dequantize produces finite weights")
        test_extract_f32_norm_tensor()
        print("[ok] extract F32 norm")
    else:
        print("[SKIP] GGUF not available on this machine")
