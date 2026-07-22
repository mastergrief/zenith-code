"""Phase 0 gate tests for the ternary-rotor runtime quant facade.

CPU-safe (no training loop, no GPU): round-trip/orthonormality
characterization of the turbo2/turbo3 port + bits-ledger audit.
"""

import math

import pytest
import torch

from calm.hrm_text_158.native_full_stack.rotor_runtime_quant import (
    ROTOR_GROUP,
    ROTOR_TARGET_SURFACES,
    _signed_fwht,
    _signed_fwht_inverse,
    rotor_bits_ledger,
    rotor_fake_quant,
)


def _gauss(shape, seed=17):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


class TestSignedFWHT:
    def test_orthonormal_norm_preserving(self):
        x = _gauss((64, ROTOR_GROUP))
        y = _signed_fwht(x)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)

    def test_inverse_round_trip(self):
        x = _gauss((64, ROTOR_GROUP))
        back = _signed_fwht_inverse(_signed_fwht(x))
        assert torch.allclose(x, back, atol=1e-5)


class TestRotorFakeQuant:
    def test_shape_and_dtype_preserved(self):
        x = _gauss((3, 7, 512))  # hidden=512 = 4 rotation groups
        out = rotor_fake_quant(x, bits=2)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_zero_input_maps_to_zero(self):
        x = torch.zeros(2, ROTOR_GROUP)
        out = rotor_fake_quant(x, bits=2)
        assert torch.all(out == 0)

    def test_reconstruction_quality_gaussian(self):
        # Characterization: cosine similarity per group on N(0,1) data.
        # Values are empirical anchors for THIS port (regression guards),
        # not quality claims about HRM activations — that is Phase 1.
        x = _gauss((256, ROTOR_GROUP))
        cos2 = torch.cosine_similarity(x, rotor_fake_quant(x, bits=2), dim=-1)
        cos3 = torch.cosine_similarity(x, rotor_fake_quant(x, bits=3), dim=-1)
        assert cos2.mean() > 0.85
        assert cos3.mean() > 0.94
        # 3-bit must strictly beat 2-bit.
        assert cos3.mean() > cos2.mean()

    def test_group_norm_preserved_by_corrected_scale(self):
        # The corrected-norm construction reproduces each group's L2 norm
        # up to fp16 scale rounding.
        x = _gauss((128, ROTOR_GROUP))
        out = rotor_fake_quant(x, bits=2)
        rel = (out.norm(dim=-1) - x.norm(dim=-1)).abs() / x.norm(dim=-1)
        assert rel.max() < 2e-3

    def test_scale_invariance_of_direction(self):
        # Polar form: scaling input scales output (same codes, scaled norm).
        x = _gauss((32, ROTOR_GROUP))
        a = rotor_fake_quant(x, bits=2)
        b = rotor_fake_quant(x * 4.0, bits=2)
        assert torch.allclose(a * 4.0, b, rtol=2e-3, atol=1e-5)

    def test_rejects_bad_last_dim_and_bits(self):
        with pytest.raises(ValueError):
            rotor_fake_quant(torch.zeros(2, 100), bits=2)
        with pytest.raises(ValueError):
            rotor_fake_quant(torch.zeros(2, ROTOR_GROUP), bits=5)


class TestBitsLedger:
    def test_turbo2_fp16_is_2p125_and_not_sub2(self):
        # Hand audit: 128 codes * 2 bits + 16-bit scale = 272 bits / 128
        led = rotor_bits_ledger(
            128, 2, surface="attention_kv_attention_buffers"
        )
        assert led.total_bits == 272
        assert led.bpw_scale_inclusive == pytest.approx(2.125)
        assert led.sub2_scale_inclusive is False

    def test_turbo3_fp16_is_3p125(self):
        # Hand audit: 128 * (2-bit codes + 1-bit sign plane) + 16-bit scale
        # = 400 bits / 128 = 3.125 (block_turbo3_0: 2+32+16 = 50 bytes).
        led = rotor_bits_ledger(128, 3, surface="activations_residuals")
        assert led.bpw_scale_inclusive == pytest.approx(3.125)
        assert led.sign_plane_bits_per_value == 1
        assert led.sub2_scale_inclusive is False

    def test_int8_scale_packing_step_would_clear_sub2_for_2bit(self):
        # Phase 4 packing arithmetic: 2-bit codes + int8 group scale
        # = 2.0625 -> still NOT sub2 (strict < 2.0). Honest accounting.
        led = rotor_bits_ledger(128, 2, surface="activations_residuals",
                                scale_dtype="int8")
        assert led.bpw_scale_inclusive == pytest.approx(2.0625)
        assert led.sub2_scale_inclusive is False

    def test_multi_group_tensor(self):
        # hidden=512 activations, 1024 tokens: 512*1024 values, 4096 groups.
        n = 512 * 1024
        led = rotor_bits_ledger(n, 2, surface="backward_saved_tensors_transients")
        assert led.n_groups == n // 128
        assert led.bpw_scale_inclusive == pytest.approx(2.125)

    def test_surfaces_enum_matches_lane_targets(self):
        assert set(ROTOR_TARGET_SURFACES) == {
            "activations_residuals",
            "attention_kv_attention_buffers",
            "backward_saved_tensors_transients",
        }

    def test_ternary_q175_geometry_is_sub2(self):
        # THE sub-2 route: 3-level codes base-3 packed (26B/128) + fp16
        # scale = (208+16)/128 = 1.75 bpw — exactly the Q1_75 geometry.
        led = rotor_bits_ledger(128, "ternary",
                                surface="attention_kv_attention_buffers")
        assert led.bpw_scale_inclusive == pytest.approx(1.75)
        assert led.sub2_scale_inclusive is True
        # int8 scale variant: 1.6875.
        led8 = rotor_bits_ledger(128, "ternary", surface="x",
                                 scale_dtype="int8")
        assert led8.bpw_scale_inclusive == pytest.approx(1.6875)

    def test_ternary_fake_quant_round_trip(self):
        x = _gauss((256, ROTOR_GROUP))
        out = rotor_fake_quant(x, bits="ternary")
        cos = torch.cosine_similarity(x, out, dim=-1)
        # 3-level polar quant on rotated gaussian: empirical anchor for this
        # port; must be worse than 2-bit (fewer levels).
        cos2 = torch.cosine_similarity(x, rotor_fake_quant(x, bits=2), dim=-1)
        assert cos.mean() > 0.75
        assert cos.mean() < cos2.mean()

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            rotor_bits_ledger(100, 2, surface="x")
        with pytest.raises(ValueError):
            rotor_bits_ledger(128, 5, surface="x")
        with pytest.raises(ValueError):
            rotor_bits_ledger(128, 2, surface="x", scale_dtype="fp8")
        with pytest.raises(ValueError):
            rotor_bits_ledger(256, "ternary", surface="x", group_size=256)
