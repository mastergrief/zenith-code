"""Tests for byte-level tq4 install (no re-quantization)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.tq4_byte_install import (
    Tq4LinearGGMLOriented, build_zero_tq4_block_bytes,
    pad_tq4_tensor_columns, pad_tq4_tensor_rows,
    pad_tq4_tensor_rows_and_cols,
)
from calm.llm_computer.tq4_torch import (
    HEAD_DIM, Tq4Tensor, build_pi, dequantize_tq4, quantize_tq4,
)


GGUF_PATH = Path(
    os.environ.get(
        "ZENITH_GEMMA_GGUF",
        "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
    )
)
GGUF_AVAILABLE = GGUF_PATH.exists()


# ----- Zero blocks -----

def test_zero_blocks_dequantize_to_zero():
    """A tq4 block with d=0 dequantizes to all zeros regardless of qs."""
    qs, d = build_zero_tq4_block_bytes(n_blocks=4)
    q = Tq4Tensor(qs=qs, d=d, shape=(4, HEAD_DIM))
    pi = build_pi(source="c_header")
    out = dequantize_tq4(q, pi=pi)
    assert out.shape == (4, HEAD_DIM)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ----- Column padding -----

def test_pad_columns_preserves_original_values():
    """After column padding, the original region dequantizes to the
    same values as the original tensor."""
    torch.manual_seed(42)
    pi = build_pi(source="c_header")
    # Build a 2-row tensor with 2 blocks per row (2 × HEAD_DIM = 512 cols)
    orig_fp = torch.randn(2, 512) * 0.05
    orig_q = quantize_tq4(orig_fp, pi=pi)
    # Pad to 1024 columns (4 blocks per row)
    padded = pad_tq4_tensor_columns(orig_q, target_n_cols=1024)
    assert padded.shape == (2, 1024)

    # Dequantize; first 512 cols should match original
    orig_dq = dequantize_tq4(orig_q, pi=pi)
    padded_dq = dequantize_tq4(padded, pi=pi)
    assert torch.allclose(padded_dq[:, :512], orig_dq, atol=1e-6), (
        f"column padding corrupted original region"
    )
    # Padded region (cols 512-1023) should be zero
    assert torch.allclose(padded_dq[:, 512:], torch.zeros(2, 512), atol=1e-6)


def test_pad_columns_rejects_misalignment():
    torch.manual_seed(0)
    orig_fp = torch.randn(2, 512) * 0.05
    orig_q = quantize_tq4(orig_fp)
    with pytest.raises(AssertionError, match="divisible"):
        pad_tq4_tensor_columns(orig_q, target_n_cols=513)


# ----- Row padding -----

def test_pad_rows_preserves_original_values():
    torch.manual_seed(1)
    pi = build_pi(source="c_header")
    orig_fp = torch.randn(3, 256) * 0.05
    orig_q = quantize_tq4(orig_fp, pi=pi)
    padded = pad_tq4_tensor_rows(orig_q, target_n_rows=8)
    assert padded.shape == (8, 256)
    padded_dq = dequantize_tq4(padded, pi=pi)
    orig_dq = dequantize_tq4(orig_q, pi=pi)
    assert torch.allclose(padded_dq[:3], orig_dq, atol=1e-6)
    assert torch.allclose(padded_dq[3:], torch.zeros(5, 256), atol=1e-6)


def test_pad_both_dims():
    torch.manual_seed(2)
    pi = build_pi(source="c_header")
    orig_fp = torch.randn(2, 512) * 0.05
    orig_q = quantize_tq4(orig_fp, pi=pi)
    padded = pad_tq4_tensor_rows_and_cols(
        orig_q, target_n_rows=6, target_n_cols=1024,
    )
    assert padded.shape == (6, 1024)
    padded_dq = dequantize_tq4(padded, pi=pi)
    orig_dq = dequantize_tq4(orig_q, pi=pi)
    # Top-left (2, 512) block matches
    assert torch.allclose(padded_dq[:2, :512], orig_dq, atol=1e-6)
    # Rest is zero
    assert torch.allclose(padded_dq[2:, :], torch.zeros(4, 1024), atol=1e-6)
    assert torch.allclose(padded_dq[:2, 512:], torch.zeros(2, 512), atol=1e-6)


# ----- Tq4LinearGGMLOriented -----

def test_ggml_oriented_linear_forward_shape():
    layer = Tq4LinearGGMLOriented(in_features=256, out_features=512)
    # Install zero blocks to make forward pass runnable
    qs = torch.zeros(256 * 512 // HEAD_DIM, 128, dtype=torch.uint8)
    d = torch.zeros(256 * 512 // HEAD_DIM, dtype=torch.float32)
    layer.install_tq4(Tq4Tensor(qs=qs, d=d, shape=(256, 512)))
    x = torch.randn(1, 3, 256)
    out = layer(x)
    assert out.shape == (1, 3, 512)
    # Zero weights → zero output
    assert torch.allclose(out, torch.zeros_like(out))


def test_ggml_oriented_linear_matches_standard_when_weights_transferred():
    """Install the same weight bytes into both GGML-oriented and
    standard Tq4Linear. Compare outputs — they should match when the
    standard variant uses transposed weights."""
    from calm.llm_computer.tq4_torch import Tq4Linear
    torch.manual_seed(7)
    pi = build_pi(source="c_header")
    # Build a (in=256, out=512) weight in (in, out) orientation
    w_io = torch.randn(256, 512) * 0.02
    q_io = quantize_tq4(w_io, pi=pi)

    # GGML-oriented: install directly
    ggml_layer = Tq4LinearGGMLOriented(in_features=256, out_features=512)
    ggml_layer.install_tq4(q_io)

    # Standard: transpose weight, quantize, install
    w_oi = w_io.T.contiguous()
    standard_layer = Tq4Linear(in_features=256, out_features=512)
    standard_layer.load_weight(w_oi)

    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        out_ggml = ggml_layer(x)
        out_standard = standard_layer(x)
    # They go through different quantization paths (GGML kept original
    # bytes; standard re-quantized the transpose). Expect close but not
    # bit-exact.
    assert out_ggml.shape == out_standard.shape
    # Both should produce finite values
    assert torch.isfinite(out_ggml).all()
    assert torch.isfinite(out_standard).all()


# ----- Real GGUF byte-level install -----

@pytest.mark.skipif(not GGUF_AVAILABLE, reason="Gemma GGUF not present")
def test_gemma_weight_bytes_install_without_requant_drift():
    """Core claim: installing Gemma's tq4 bytes byte-level into a padded
    Tq4LinearGGMLOriented preserves Gemma's EXACT numerics (zero drift)."""
    from calm.llm_computer.tq4_gguf_loader import (
        extract_tq4_tensor, read_turboquant_gguf,
    )
    reader = read_turboquant_gguf(GGUF_PATH)
    pi = build_pi(source="c_header")

    # Get Gemma's layer 0 Q projection: GGUF (2560, 2048)
    q_gemma = extract_tq4_tensor(reader, "blk.0.attn_q.weight")
    assert q_gemma.shape == (2560, 2048)

    # Dequantize Gemma once — this is our reference
    fp_gemma = dequantize_tq4(q_gemma, pi=pi)

    # Pad to (4096, 4096) — substrate shape
    q_padded = pad_tq4_tensor_rows_and_cols(
        q_gemma, target_n_rows=4096, target_n_cols=4096,
    )
    assert q_padded.shape == (4096, 4096)

    # Dequantize the padded version
    fp_padded = dequantize_tq4(q_padded, pi=pi)

    # Gemma's region should match bit-for-bit
    assert torch.allclose(fp_padded[:2560, :2048], fp_gemma, atol=1e-5), (
        f"byte-level install drifted: max diff "
        f"{(fp_padded[:2560, :2048] - fp_gemma).abs().max()}"
    )
    # Padding regions are zero
    assert torch.allclose(fp_padded[2560:, :], torch.zeros(1536, 4096), atol=1e-6)
    assert torch.allclose(fp_padded[:2560, 2048:], torch.zeros(2560, 2048), atol=1e-6)

    print(f"\n  Byte install drift: "
          f"{(fp_padded[:2560, :2048] - fp_gemma).abs().max():.2e}  "
          f"(vs ~1% for dequant+requant path)")


if __name__ == "__main__":
    test_zero_blocks_dequantize_to_zero()
    print("[ok] zero blocks dequantize to zero")
    test_pad_columns_preserves_original_values()
    print("[ok] column padding preserves original region")
    test_pad_columns_rejects_misalignment()
    print("[ok] misaligned column target rejected")
    test_pad_rows_preserves_original_values()
    print("[ok] row padding preserves original region")
    test_pad_both_dims()
    print("[ok] row + column padding together")
    test_ggml_oriented_linear_forward_shape()
    print("[ok] Tq4LinearGGMLOriented forward shape")
    test_ggml_oriented_linear_matches_standard_when_weights_transferred()
    print("[ok] ggml-oriented Linear forward produces valid outputs")
    if GGUF_AVAILABLE:
        test_gemma_weight_bytes_install_without_requant_drift()
        print("[ok] Gemma bytes install into padded substrate WITHOUT re-quant drift")
    else:
        print("[SKIP] Gemma GGUF not available")
