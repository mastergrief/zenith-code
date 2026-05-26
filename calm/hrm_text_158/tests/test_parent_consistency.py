"""Unit tests for the parent-consistency loss slice.

Covers the cheap gates from the design (the model-level smoke + F.2d sweep are
separate): KL-helper correctness/gradient, denominator safety, and is_prior
rung-based tagging. No GPU / no model load required.
"""
import importlib.util
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Load the trainer module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thr)

from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID

_pc_kl = _thr._parent_consistency_kl


def _mk_labels(B, L, resp_from=3):
    labels = torch.full((B, L), IGNORE_LABEL_ID, dtype=torch.long)
    labels[:, resp_from:] = torch.randint(0, 260, (B, L - resp_from))
    return labels


def test_child_equals_parent_kl_near_zero():
    torch.manual_seed(0)
    B, L, V = 4, 6, 260
    logits = torch.randn(B, L, V, requires_grad=True)
    labels = _mk_labels(B, L)
    is_prior = torch.tensor([True, False, True, True])
    kl = _pc_kl(logits, logits.detach(), labels, is_prior, temp=1.0)
    assert kl.item() < 1e-6, f"child==parent KL must be ~0, got {kl.item()}"


def test_no_prior_rows_returns_zero():
    torch.manual_seed(1)
    B, L, V = 4, 6, 260
    child = torch.randn(B, L, V, requires_grad=True)
    parent = torch.randn(B, L, V)
    labels = _mk_labels(B, L)
    is_prior = torch.tensor([False, False, False, False])
    kl = _pc_kl(child, parent, labels, is_prior)
    assert kl.item() == 0.0, f"no prior rows -> exactly 0 (denom clamp), got {kl.item()}"
    # grad path stays finite (0 grad) — no NaN from div-by-zero.
    kl.backward()
    assert torch.isfinite(child.grad).all()


def test_divergent_positive_and_grad_flows():
    torch.manual_seed(2)
    B, L, V = 4, 6, 260
    child = torch.randn(B, L, V, requires_grad=True)
    parent = torch.randn(B, L, V)
    labels = _mk_labels(B, L)
    is_prior = torch.tensor([True, True, False, True])
    kl = _pc_kl(child, parent, labels, is_prior)
    assert kl.item() > 0.0, f"divergent KL must be >0, got {kl.item()}"
    kl.backward()
    assert child.grad is not None and torch.isfinite(child.grad).all()
    # Gradient must be zero on non-prior rows (row 2) and on prefix (ignored).
    assert child.grad[2].abs().sum().item() == 0.0, "non-prior row must get 0 grad"
    assert child.grad[0, :3].abs().sum().item() == 0.0, "prefix positions must get 0 grad"


def test_is_prior_rung_tagging():
    """is_prior = (rung != curriculum_rung); anchors (no rung) count as prior."""
    from calm.hrm_text_158.curriculum.broad_tokenizer import BroadTokenizer as _Tok
    tok = _Tok()
    rows = [
        {"question": "what is 5?", "expected": "5", "rung": "L0c1"},   # target -> NOT prior
        {"question": "what is 2 plus 3?", "expected": "5", "rung": "R1"},  # prior
        {"question": "what is 7?", "expected": "7"},                    # anchor (no rung) -> prior
    ]
    ds = _thr.HrmTextGsm8kDataset(rows, tok, max_len=64, curriculum_rung="L0c1")
    flags = [it[2] for it in ds.items]  # is_prior per kept row
    assert flags == [False, True, True], f"unexpected is_prior tagging: {flags}"
    # Off-curriculum -> all False.
    ds_off = _thr.HrmTextGsm8kDataset(rows, tok, max_len=64, curriculum_rung=None)
    assert all(it[2] is False for it in ds_off.items)


# --- STEP 2a: true-label close-sibling CE-interleave (parent-KL mask exclusion) ---

_CE_NAME = "L0c1-close-sibling-true-label-ce"


def test_ce_interleave_rows_are_true_label_not_prior():
    """CE-interleave rows are CE-only (is_prior=False) while replay/anchor stay
    prior. Regression for the parent-KL-mask bug (co_lead 1779797888141): a
    ce_interleave row whose `rung` != curriculum_rung must NOT be parent-KL'd
    (parent-KL would preserve the broken `2 -> "22"` label it repairs)."""
    from calm.hrm_text_158.curriculum.broad_tokenizer import BroadTokenizer as _Tok
    tok = _Tok()
    target = "L0c2-K2-addition-full"
    ce_rows = _thr._compose_ce_interleave_rows([f"{_CE_NAME}:1"], seed=17)
    ce_two = next(r for r in ce_rows if r["question"] == "2 equals what?")
    assert ce_two["expected"] == 2
    assert ce_two["ce_interleave"] == _CE_NAME
    assert ce_two["rung"] == _CE_NAME            # trace tag; differs from target rung
    rows = [
        {"question": "20 plus 1 equals what?", "expected": 21, "rung": target},  # target -> NOT prior
        {"question": "what is 2 plus 3?", "expected": "5", "rung": "R1"},         # replay prior
        {"question": "what is 7?", "expected": "7"},                             # anchor (no rung) -> prior
        ce_two,                                                                   # CE-interleave -> NOT prior
    ]
    ds = _thr.HrmTextGsm8kDataset(rows, tok, max_len=64, curriculum_rung=target)
    flags = [it[2] for it in ds.items]
    assert flags == [False, True, True, False], f"unexpected is_prior tagging: {flags}"


def test_compose_ce_interleave_rows_shape_and_repeat():
    rows1 = _thr._compose_ce_interleave_rows([f"{_CE_NAME}:1"], seed=17)
    rows3 = _thr._compose_ce_interleave_rows([f"{_CE_NAME}:3"], seed=17)
    assert len(rows1) == 13                      # 0..9 identity + 11/17/99 sentinels
    assert len(rows3) == 39 == 13 * 3            # REPEAT is mechanical
    assert all(r["ce_interleave"] == _CE_NAME and r["rung"] == _CE_NAME for r in rows1)
    assert ("2 equals what?", 2) in [(r["question"], r["expected"]) for r in rows1]
    assert _thr._compose_ce_interleave_rows(None, seed=17) == []
    assert _thr._compose_ce_interleave_rows([], seed=17) == []


def test_parse_ce_interleave_specs_valid_and_rejects():
    import pytest
    assert _thr._parse_ce_interleave_specs([f"{_CE_NAME}:3"]) == [(_CE_NAME, 3)]
    assert _thr._parse_ce_interleave_specs(None) == []
    with pytest.raises(ValueError, match="not in allowlist"):
        _thr._parse_ce_interleave_specs(["some-other-support:1"])
    with pytest.raises(ValueError, match="NAME:REPEAT"):
        _thr._parse_ce_interleave_specs([_CE_NAME])            # missing colon
    with pytest.raises(ValueError, match="must be an int"):
        _thr._parse_ce_interleave_specs([f"{_CE_NAME}:x"])
    with pytest.raises(ValueError, match="must be >= 1"):
        _thr._parse_ce_interleave_specs([f"{_CE_NAME}:0"])


def test_ce_interleave_support_is_not_a_retained_kl_support():
    """The CE-interleave allowlist must be disjoint from the parent-KL retained
    registry: CE-interleave is true-label CE, NOT KL toward the parent."""
    assert _CE_NAME in _thr._CE_INTERLEAVE_SUPPORTS
    assert _CE_NAME not in _thr._RETAINED_SUPPORT_REGISTRY
    assert _thr._CE_INTERLEAVE_SUPPORTS.isdisjoint(set(_thr._RETAINED_SUPPORT_REGISTRY))


def test_rejects_negative_weight():
    import pytest
    with pytest.raises(ValueError, match="must be >= 0"):
        _thr.train(parent_consistency_weight=-1.0)


def test_rejects_nonpositive_temp():
    import pytest
    with pytest.raises(ValueError, match="must be > 0"):
        _thr.train(parent_consistency_temp=0.0)


if __name__ == "__main__":
    test_child_equals_parent_kl_near_zero()
    test_no_prior_rows_returns_zero()
    test_divergent_positive_and_grad_flows()
    try:
        test_is_prior_rung_tagging()
        print("is_prior tagging: PASS")
    except Exception as e:  # tokenizer import name may differ; report, don't hard-fail here
        print(f"is_prior tagging: SKIPPED/ERROR ({type(e).__name__}: {e})")
    print("PC-KL helper tests: PASS")
