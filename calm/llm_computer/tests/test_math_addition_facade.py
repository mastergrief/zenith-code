"""Unit tests for MathAdditionFacade — everything that doesn't require
prod Gemma (so these run fast on CPU)."""

import os

import pytest
import torch


CKPT = "calm/hrm/checkpoints/copy_augmented_hrm_best.pt"

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason="PT checkpoint not available in this environment",
)


def _facade(device="cpu", **kw):
    from calm.llm_computer.facades import MathAdditionFacade
    return MathAdditionFacade(pt_ckpt_path=CKPT, device=device, **kw)


def test_allocation_ranges_disjoint():
    f = _facade()
    lo1, hi1 = f.alloc.pt_ch
    lo2, hi2 = f.alloc.adder_ch
    assert hi1 <= lo2, f"PT ch {f.alloc.pt_ch} overlaps adder ch {f.alloc.adder_ch}"
    assert (hi1 - lo1) == f.PT_VOCAB
    assert (hi2 - lo2) == f.ADDER_VOCAB


def test_construction_loads_models():
    f = _facade()
    # PT was loaded and wrapped
    pt_params = sum(p.numel() for p in f.pt.pt.parameters())
    assert pt_params == 181_313, f"PT params {pt_params} != 181,313"
    # adder_tiny compiled
    assert f.adder.config.vocab_size == 8
    # Router has one registered route
    assert len(f.router.routes) == 1
    assert f.router.routes[0].operator == "+"


def test_set_prompt_produces_pt_ids():
    f = _facade()
    f.set_prompt("what is 2 plus 3")
    from calm.hrm.data import _CHAR_TO_ID
    ids = f._pt_input_ids[0].tolist()
    assert ids[0] == _CHAR_TO_ID["<bos>"]
    assert ids[-1] == _CHAR_TO_ID["<sep>"]
    # Mid-sequence should contain digits 2 and 3
    mid = ids[1:-1]
    assert _CHAR_TO_ID["2"] in mid
    assert _CHAR_TO_ID["3"] in mid


def test_pt_card_input_requires_set_prompt():
    f = _facade()
    with pytest.raises(RuntimeError, match="set_prompt"):
        f._pt_card_input(torch.zeros(1, 1, 2560))


def test_translate_operands_clamps():
    f = _facade()
    # In-range
    assert f._translate_adder_operands([2, 3]).tolist() == [[2, 3]]
    # Out-of-range clamps to [0, 7]
    assert f._translate_adder_operands([10, -5]).tolist() == [[7, 0]]


def test_adder_card_input_sets_parse_flag():
    """Synthesize a residual containing a one-hot-encoded '2 + 3' in
    PT's channel range, check that _adder_card_input extracts (2, 3)
    and sets parse_ok=True."""
    from calm.hrm.data import _CHAR_TO_ID
    f = _facade()
    pt_lo, pt_hi = f.alloc.pt_ch
    d_model = 2560
    S = 5
    h = torch.zeros(1, S, d_model)
    # Make <pad> (ch 2400) the default across all positions
    h[..., pt_lo] = 1.0
    # Overwrite positions 0..4 with one-hot for '2', ' ', '+', ' ', '3'
    chars = ["2", " ", "+", " ", "3"]
    for i, c in enumerate(chars):
        h[..., i, pt_lo] = 0.0
        h[..., i, pt_lo + _CHAR_TO_ID[c]] = 1.0

    inp = f._adder_card_input(h)
    assert f._parse_ok is True
    assert inp.tolist() == [[2, 3]]


def test_adder_card_input_parse_failure_flags():
    """PT output has no '+' — parse should fail, flag should be False."""
    from calm.hrm.data import _CHAR_TO_ID
    f = _facade()
    pt_lo, _ = f.alloc.pt_ch
    d_model = 2560
    h = torch.zeros(1, 3, d_model)
    h[..., pt_lo] = 1.0  # all <pad>
    # Write 'the' (no '+')
    for i, c in enumerate(["t", "h", "e"]):
        h[..., i, pt_lo] = 0.0
        h[..., i, pt_lo + _CHAR_TO_ID[c]] = 1.0

    f._adder_card_input(h)
    assert f._parse_ok is False


def test_pt_writer_one_hots_in_bounds():
    """PT writer must keep residual values in {0, 1} — bounded to avoid
    warping output_norm."""
    f = _facade()
    pt_lo, pt_hi = f.alloc.pt_ch
    d_model = 2560
    h = torch.zeros(1, 5, d_model)
    # Fake card_out: G=3 tokens with log-probs up to -100
    card_out = torch.full((1, 3, f.PT_VOCAB), -100.0)
    card_out[0, 0, 8] = 0.0   # argmax token 8 = '2'
    card_out[0, 1, 16] = 0.0  # argmax token 16 = '+'
    card_out[0, 2, 9] = 0.0   # argmax token 9 = '3'

    h = f._pt_writer(h, card_out, pt_lo, pt_hi)
    pt_slice = h[..., pt_lo:pt_hi]
    assert pt_slice.max().item() <= 1.0
    assert pt_slice.min().item() >= 0.0
    # Each position's channel values sum to 1 (one-hot)
    per_pos = pt_slice.sum(dim=-1)
    assert torch.allclose(per_pos, torch.ones_like(per_pos))


def test_adder_writer_zeros_card_out_on_parse_failure():
    """Writer must in-place zero card_out so VerificationHook.last_output
    reads margin=0 and the hook stays silent."""
    f = _facade()
    _, _ = f.alloc.adder_ch
    d_model = 2560
    h = torch.zeros(1, 5, d_model)
    card_out = torch.zeros(1, 2, f.ADDER_VOCAB)
    card_out[0, -1, 0] = 1.0  # simulate confident answer '0'

    f._parse_ok = False
    f._adder_writer(h, card_out, *f.alloc.adder_ch)
    # card_out must be zeroed IN-PLACE (the CardSlot saved this ref to
    # slot.last_output — the hook reads it there)
    assert card_out.abs().max().item() == 0.0


def test_adder_writer_preserves_card_out_when_parse_ok():
    f = _facade()
    d_model = 2560
    h = torch.zeros(1, 5, d_model)
    card_out = torch.zeros(1, 2, f.ADDER_VOCAB)
    card_out[0, -1, 5] = 1.0  # adder says 5

    f._parse_ok = True
    f._adder_writer(h, card_out, *f.alloc.adder_ch)
    # card_out untouched so the hook can read the confident '5'
    assert card_out[0, -1, 5].item() == 1.0


def test_install_without_gemma_raises_on_double_install():
    """Reinstall without detach should refuse."""

    class FakeGemma:
        def __init__(self):
            self.layers = [type("L", (), {})() for _ in range(42)]
            self.verification_hooks = []
            self.reserved_channels = []

    f = _facade()
    g = FakeGemma()
    f.install(g)
    with pytest.raises(RuntimeError, match="already installed"):
        f.install(g)
    f.detach(g)
    f.install(g)  # now OK


def test_detach_removes_hooks_and_slots():
    class FakeGemma:
        def __init__(self):
            self.layers = [type("L", (), {})() for _ in range(42)]
            self.verification_hooks = []
            self.reserved_channels = []

    f = _facade()
    g = FakeGemma()
    f.install(g)
    assert len(g.verification_hooks) == 1
    assert len(g.layers[f.layer].card_slots) == 2
    assert any(r[2] == f.layer for r in g.reserved_channels)

    f.detach(g)
    assert g.verification_hooks == []
    assert g.layers[f.layer].card_slots == []
    assert not any(r[2] == f.layer for r in g.reserved_channels)
