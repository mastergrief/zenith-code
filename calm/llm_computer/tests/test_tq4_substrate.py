"""Tests for Tq4GroupedSmall2DTransformer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.grouped_small2d import GroupedSmall2DConfig
from calm.llm_computer.tq4_byte_install import Tq4LinearGGMLOriented
from calm.llm_computer.tq4_substrate import (
    Tq4GroupedSmall2DTransformer,
    install_ffn_in_from_parts,
    install_qkv_from_parts,
    install_simple_tq4_corner,
)
from calm.llm_computer.tq4_torch import (
    HEAD_DIM, Tq4Tensor, build_pi, dequantize_tq4, quantize_tq4,
)


def _tiny_cfg():
    return GroupedSmall2DConfig(
        vocab_size=HEAD_DIM,
        d_model=HEAD_DIM,
        n_heads=HEAD_DIM // 2,  # d_head=2 invariant
        n_layers=1,
        d_ffn=HEAD_DIM,
        max_len=4,
        use_hard_max=False,
    )


def test_tq4_substrate_builds():
    cfg = _tiny_cfg()
    m = Tq4GroupedSmall2DTransformer(cfg)
    assert len(m.W_qkv) == cfg.n_layers
    assert isinstance(m.W_qkv[0], Tq4LinearGGMLOriented)


def test_tq4_substrate_zero_init_runs_forward():
    cfg = _tiny_cfg()
    m = Tq4GroupedSmall2DTransformer(cfg)
    m.initialize_all_zero_tq4()
    # Zero token embedding too
    with torch.no_grad():
        m.tok.weight.zero_()
        m.pos.weight.zero_()
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 4, cfg.vocab_size)
    # Zero weights → zero logits
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_tq4_substrate_forward_produces_finite_with_real_weights():
    """Install non-zero tq4 into each layer; forward must run and
    produce finite logits."""
    cfg = _tiny_cfg()
    m = Tq4GroupedSmall2DTransformer(cfg)
    torch.manual_seed(1)
    pi = build_pi(source="c_header")
    # Build tq4 weights for each layer
    for layer in m.W_qkv:
        w_fp = torch.randn(layer.in_features, layer.out_features) * 0.05
        layer.install_tq4(quantize_tq4(w_fp, pi=pi))
    for layer in m.W_out:
        w_fp = torch.randn(layer.in_features, layer.out_features) * 0.05
        layer.install_tq4(quantize_tq4(w_fp, pi=pi))
    for layer in m.ff_in:
        w_fp = torch.randn(layer.in_features, layer.out_features) * 0.05
        layer.install_tq4(quantize_tq4(w_fp, pi=pi))
    for layer in m.ff_out:
        w_fp = torch.randn(layer.in_features, layer.out_features) * 0.05
        layer.install_tq4(quantize_tq4(w_fp, pi=pi))
    w_fp = torch.randn(m.head.in_features, m.head.out_features) * 0.05
    m.head.install_tq4(quantize_tq4(w_fp, pi=pi))
    with torch.no_grad():
        m.tok.weight.normal_(0, 0.02)
        m.pos.weight.normal_(0, 0.02)
    m.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (1, 4, cfg.vocab_size)
    assert torch.isfinite(out).all()


def test_install_qkv_from_parts_shapes():
    """Byte-install Q, K, V into a stacked W_qkv and verify shapes."""
    D_s = 512                       # substrate d_model (multiple of HEAD_DIM)
    gemma_d_model = 256
    gemma_q_out = 256
    gemma_kv_out = HEAD_DIM          # 256 == HEAD_DIM

    pi = build_pi(source="c_header")
    q_fp = torch.randn(gemma_d_model, gemma_q_out) * 0.02
    k_fp = torch.randn(gemma_d_model, gemma_kv_out) * 0.02
    v_fp = torch.randn(gemma_d_model, gemma_kv_out) * 0.02
    q_tq4 = quantize_tq4(q_fp, pi=pi)
    k_tq4 = quantize_tq4(k_fp, pi=pi)
    v_tq4 = quantize_tq4(v_fp, pi=pi)

    target = Tq4LinearGGMLOriented(D_s, 3 * D_s)
    install_qkv_from_parts(target, q_tq4, k_tq4, v_tq4, substrate_d_model=D_s)

    # Target should be loaded
    assert target.is_loaded()
    # Forward a test input through it
    x = torch.randn(1, 2, D_s) * 0.02
    with torch.no_grad():
        out = target(x)
    assert out.shape == (1, 2, 3 * D_s)
    assert torch.isfinite(out).all()


def test_install_qkv_preserves_gemma_region():
    """The installed W_qkv should, when dequantized, have Gemma's Q in
    the top-left corner of its Q region, Gemma's K in the K region,
    Gemma's V in the V region, and zeros everywhere else."""
    D_s = 512
    gemma_d_model = 256
    gemma_q_out = 256
    gemma_kv_out = HEAD_DIM

    pi = build_pi(source="c_header")
    q_fp = torch.randn(gemma_d_model, gemma_q_out) * 0.02
    k_fp = torch.randn(gemma_d_model, gemma_kv_out) * 0.02
    v_fp = torch.randn(gemma_d_model, gemma_kv_out) * 0.02
    q_tq4 = quantize_tq4(q_fp, pi=pi)
    k_tq4 = quantize_tq4(k_fp, pi=pi)
    v_tq4 = quantize_tq4(v_fp, pi=pi)

    target = Tq4LinearGGMLOriented(D_s, 3 * D_s)
    install_qkv_from_parts(target, q_tq4, k_tq4, v_tq4, substrate_d_model=D_s)

    # Dequantize and inspect the full weight
    from calm.llm_computer.tq4_torch import dequantize_tq4_differentiable
    with torch.no_grad():
        w_full = dequantize_tq4_differentiable(
            Tq4Tensor(qs=target._qs, d=target._d, shape=(D_s, 3*D_s)),
            target._pi, target._centroids,
        )

    # Original dequants
    q_dq = dequantize_tq4(q_tq4, pi=pi)
    k_dq = dequantize_tq4(k_tq4, pi=pi)
    v_dq = dequantize_tq4(v_tq4, pi=pi)

    # Q region: w_full[:gemma_d_model, :gemma_q_out] should match q_dq
    assert torch.allclose(
        w_full[:gemma_d_model, :gemma_q_out], q_dq, atol=1e-5,
    ), "Q region drifted after byte install"
    # K region: w_full[:gemma_d_model, D_s : D_s+gemma_kv_out] should match k_dq
    assert torch.allclose(
        w_full[:gemma_d_model, D_s : D_s + gemma_kv_out], k_dq, atol=1e-5,
    ), "K region drifted"
    # V region:
    assert torch.allclose(
        w_full[:gemma_d_model, 2*D_s : 2*D_s + gemma_kv_out], v_dq, atol=1e-5,
    ), "V region drifted"
    # Padding (rows beyond gemma_d_model) should be zero
    assert torch.allclose(
        w_full[gemma_d_model:, :], torch.zeros(D_s - gemma_d_model, 3*D_s), atol=1e-6,
    )
    # Q padding (cols gemma_q_out..D_s in Q region)
    assert torch.allclose(
        w_full[:gemma_d_model, gemma_q_out:D_s],
        torch.zeros(gemma_d_model, D_s - gemma_q_out), atol=1e-6,
    )


def test_install_ffn_from_parts():
    """Stack gate and up projections into ff_in."""
    D_s = 512
    D_ffn_s = 512
    gemma_d_model = 256
    gemma_d_ffn = 256

    pi = build_pi(source="c_header")
    gate_fp = torch.randn(gemma_d_model, gemma_d_ffn) * 0.02
    up_fp = torch.randn(gemma_d_model, gemma_d_ffn) * 0.02
    gate_tq4 = quantize_tq4(gate_fp, pi=pi)
    up_tq4 = quantize_tq4(up_fp, pi=pi)

    target = Tq4LinearGGMLOriented(D_s, 2 * D_ffn_s)
    install_ffn_in_from_parts(
        target, gate_tq4, up_tq4, substrate_d_model=D_s, substrate_d_ffn=D_ffn_s,
    )
    assert target.is_loaded()
    x = torch.randn(1, 2, D_s) * 0.02
    with torch.no_grad():
        out = target(x)
    assert out.shape == (1, 2, 2 * D_ffn_s)
    assert torch.isfinite(out).all()


def test_install_simple_tq4_corner():
    """For W_out / ff_out that just need corner padding."""
    from calm.llm_computer.tq4_byte_install import Tq4LinearGGMLOriented
    target = Tq4LinearGGMLOriented(512, 512)
    pi = build_pi(source="c_header")
    src_fp = torch.randn(256, 256) * 0.02
    src_tq4 = quantize_tq4(src_fp, pi=pi)
    install_simple_tq4_corner(target, src_tq4, target_in=512, target_out=512)
    assert target.is_loaded()


if __name__ == "__main__":
    test_tq4_substrate_builds()
    print("[ok] Tq4GroupedSmall2DTransformer builds")
    test_tq4_substrate_zero_init_runs_forward()
    print("[ok] zero-init forward produces zero logits")
    test_tq4_substrate_forward_produces_finite_with_real_weights()
    print("[ok] forward with real tq4 weights produces finite logits")
    test_install_qkv_from_parts_shapes()
    print("[ok] install_qkv_from_parts shape works")
    test_install_qkv_preserves_gemma_region()
    print("[ok] install_qkv preserves Gemma region bit-accurate")
    test_install_ffn_from_parts()
    print("[ok] install_ffn gate+up stack works")
    test_install_simple_tq4_corner()
    print("[ok] corner install for W_out / ff_out")
