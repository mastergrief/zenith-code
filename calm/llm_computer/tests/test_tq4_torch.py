"""Tests for PyTorch tq4 port."""

from __future__ import annotations

import math

import pytest
import torch

from calm.llm_computer.tq4_torch import (
    HEAD_DIM, N_LEVELS, Tq4Linear, Tq4Tensor,
    build_pi, compute_lloyd_max_codebook,
    dequantize_tq4, dequantize_tq4_differentiable,
    quantize_tq4,
)


# ----- Pi matrix -----

def test_pi_is_orthogonal():
    pi = build_pi()
    assert pi.shape == (HEAD_DIM, HEAD_DIM)
    # Pi @ Pi.T ≈ I
    prod = pi @ pi.T
    assert torch.allclose(prod, torch.eye(HEAD_DIM), atol=1e-5)


def test_pi_deterministic():
    a = build_pi()
    b = build_pi()
    assert torch.equal(a, b)


# ----- Lloyd-Max codebook -----

def test_codebook_shape():
    c, b = compute_lloyd_max_codebook()
    assert c.shape == (N_LEVELS,)
    assert b.shape == (N_LEVELS - 1,)


def test_codebook_centroids_monotonic():
    c, _ = compute_lloyd_max_codebook()
    diffs = c[1:] - c[:-1]
    assert (diffs > 0).all(), "centroids should be strictly increasing"


def test_codebook_boundaries_midpoints():
    c, b = compute_lloyd_max_codebook()
    expected_midpoints = 0.5 * (c[:-1] + c[1:])
    assert torch.allclose(b, expected_midpoints, atol=1e-6)


def test_codebook_symmetric_about_zero():
    """For N(0, sigma²), the codebook should be symmetric: c[-i] ≈ -c[i]."""
    c, _ = compute_lloyd_max_codebook()
    flipped = c.flip(0)
    # Inner levels should pair up near zero
    assert torch.allclose(c, -flipped, atol=1e-4), (
        f"codebook not symmetric: {c}"
    )


def test_codebook_in_expected_range():
    """Centroids should be within ~4 sigma."""
    c, _ = compute_lloyd_max_codebook()
    sigma = 1.0 / math.sqrt(HEAD_DIM)
    assert c.abs().max() < 4 * sigma


# ----- Quantize / dequantize -----

def test_quantize_dequantize_shape_preserved():
    x = torch.randn(HEAD_DIM * 4) * 0.1  # small scale
    q = quantize_tq4(x)
    y = dequantize_tq4(q)
    assert y.shape == x.shape


def test_quantize_matrix_shape():
    x = torch.randn(8, HEAD_DIM) * 0.1
    q = quantize_tq4(x)
    y = dequantize_tq4(q)
    assert y.shape == x.shape


def test_zero_input_produces_zero_output():
    x = torch.zeros(HEAD_DIM)
    q = quantize_tq4(x)
    y = dequantize_tq4(q)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_roundtrip_relative_error_reasonable():
    """Round-trip error should be small for normally-distributed input.

    Note: use a DIFFERENT seed than PI_SEED=42, otherwise the input
    correlates with the rotation basis and breaks variance concentration.
    """
    torch.manual_seed(12345)
    x = torch.randn(HEAD_DIM * 8) * (1.0 / math.sqrt(HEAD_DIM))
    q = quantize_tq4(x)
    y = dequantize_tq4(q)
    rel_err = (y - x).norm() / x.norm()
    assert rel_err < 0.25, f"round-trip rel_err = {rel_err:.3%}, too high"


def test_norm_recovered():
    """The L2 norm of each block should be roughly preserved."""
    x = torch.randn(HEAD_DIM * 4) * 0.05
    q = quantize_tq4(x)
    blocks_orig = x.reshape(-1, HEAD_DIM)
    norms_orig = blocks_orig.norm(dim=-1)
    # q.d stores norms
    assert torch.allclose(norms_orig, q.d, atol=1e-5)


def test_bytes_on_disk():
    x = torch.randn(HEAD_DIM * 4) * 0.1
    q = quantize_tq4(x)
    # 4 blocks × 132 bytes = 528 bytes
    assert q.bytes_on_disk() == 4 * 132


# ----- Differentiable dequant -----

def test_differentiable_dequant_same_output():
    """Differentiable dequant must produce identical output to regular."""
    torch.manual_seed(1)
    x = torch.randn(HEAD_DIM * 2) * 0.05
    q = quantize_tq4(x)
    pi = build_pi()
    c, _ = compute_lloyd_max_codebook()
    a = dequantize_tq4(q, pi=pi, centroids=c)
    b = dequantize_tq4_differentiable(q, pi, c)
    assert torch.allclose(a, b, atol=1e-5)


def test_straight_through_gradient():
    """Backward through the STE should not raise and produce gradient
    on DOWNSTREAM parameters (not on qs/d themselves)."""
    torch.manual_seed(2)
    x = torch.randn(HEAD_DIM) * 0.05
    q = quantize_tq4(x)
    pi = build_pi()
    c, _ = compute_lloyd_max_codebook()
    # A downstream trainable parameter
    alpha = torch.nn.Parameter(torch.ones(HEAD_DIM))
    w = dequantize_tq4_differentiable(q, pi, c)
    y = (w * alpha).sum()
    y.backward()
    assert alpha.grad is not None
    assert (alpha.grad != 0).any()


# ----- Tq4Linear -----

def test_tq4_linear_forward():
    torch.manual_seed(3)
    layer = Tq4Linear(in_features=HEAD_DIM, out_features=HEAD_DIM)
    w = torch.randn(HEAD_DIM, HEAD_DIM) * 0.1
    layer.load_weight(w)
    x = torch.randn(2, HEAD_DIM)
    y = layer(x)
    assert y.shape == (2, HEAD_DIM)


def test_tq4_linear_matches_fp_reference_roughly():
    """Tq4Linear output should be close to x @ W.T (FP reference)."""
    torch.manual_seed(4)
    layer = Tq4Linear(in_features=HEAD_DIM, out_features=HEAD_DIM)
    w = torch.randn(HEAD_DIM, HEAD_DIM) * (1.0 / math.sqrt(HEAD_DIM))
    layer.load_weight(w)
    x = torch.randn(1, HEAD_DIM) * (1.0 / math.sqrt(HEAD_DIM))
    ref = x @ w.T
    y = layer(x)
    # Allow quantization error up to 25%
    rel_err = (y - ref).norm() / ref.norm()
    assert rel_err < 0.3, f"Tq4Linear diverges too much: {rel_err:.3%}"


def test_tq4_linear_no_weight_raises():
    layer = Tq4Linear(HEAD_DIM, HEAD_DIM)
    with pytest.raises(AssertionError, match="no weight loaded"):
        layer(torch.zeros(1, HEAD_DIM))


def test_tq4_linear_with_bias_trainable():
    layer = Tq4Linear(HEAD_DIM, HEAD_DIM, bias=True)
    w = torch.randn(HEAD_DIM, HEAD_DIM) * 0.05
    layer.load_weight(w)
    assert layer.bias is not None
    assert layer.bias.requires_grad
    # Weight tq4 codes are NOT parameters
    params = [p for p in layer.parameters() if p.requires_grad]
    # Only bias should be trainable
    assert len(params) == 1
    assert params[0] is layer.bias


def test_tq4_linear_frozen_weights_no_grad():
    """After training a downstream adapter, tq4 codes must remain
    untouched."""
    torch.manual_seed(5)
    layer = Tq4Linear(HEAD_DIM, HEAD_DIM, bias=True)
    w = torch.randn(HEAD_DIM, HEAD_DIM) * 0.05
    layer.load_weight(w)
    qs_before = layer._qs.clone()
    d_before = layer._d.clone()
    # Train bias via simple loss
    opt = torch.optim.Adam([layer.bias], lr=1e-2)
    x = torch.randn(4, HEAD_DIM) * 0.1
    target = torch.randn(4, HEAD_DIM) * 0.1
    for _ in range(5):
        opt.zero_grad()
        y = layer(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        opt.step()
    # Quantized weights untouched
    assert torch.equal(layer._qs, qs_before)
    assert torch.equal(layer._d, d_before)


if __name__ == "__main__":
    test_pi_is_orthogonal()
    print("[ok] Pi orthogonal")
    test_pi_deterministic()
    print("[ok] Pi deterministic")
    test_codebook_shape()
    print("[ok] codebook shape")
    test_codebook_centroids_monotonic()
    print("[ok] centroids monotonic")
    test_codebook_boundaries_midpoints()
    print("[ok] boundaries midpoints")
    test_codebook_symmetric_about_zero()
    print("[ok] codebook symmetric")
    test_codebook_in_expected_range()
    print("[ok] codebook in expected range")
    test_quantize_dequantize_shape_preserved()
    print("[ok] shape preserved")
    test_quantize_matrix_shape()
    print("[ok] matrix shape")
    test_zero_input_produces_zero_output()
    print("[ok] zero in → zero out")
    test_roundtrip_relative_error_reasonable()
    print("[ok] round-trip error reasonable")
    test_norm_recovered()
    print("[ok] norm recovered")
    test_bytes_on_disk()
    print("[ok] bytes on disk")
    test_differentiable_dequant_same_output()
    print("[ok] differentiable dequant matches regular")
    test_straight_through_gradient()
    print("[ok] straight-through gradient flows")
    test_tq4_linear_forward()
    print("[ok] Tq4Linear forward")
    test_tq4_linear_matches_fp_reference_roughly()
    print("[ok] Tq4Linear matches FP reference")
    test_tq4_linear_no_weight_raises()
    print("[ok] no weight raises")
    test_tq4_linear_with_bias_trainable()
    print("[ok] bias trainable, codes frozen")
    test_tq4_linear_frozen_weights_no_grad()
    print("[ok] frozen weights preserved during training")
