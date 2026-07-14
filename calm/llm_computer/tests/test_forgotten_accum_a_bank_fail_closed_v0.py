"""Fail-closed A-BANK scope-(A) safety landing tests + early-refuse spies."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch import (
    EXIT_RUN_ARMS_BANK_INPUTS,
    apply_claim_coupling,
    launch_run_arms,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_measure import (
    BANK_INPUTS_INVALID,
    BankInputsRefuse,
    parse_complete_bank_inputs,
    parse_required_arm_bank_blob,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    FailClosedClass,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
    run_forgotten_accum_training_equivalence_arms,
)


def _complete_blob(**overrides):
    blob = {
        "acquire_pct": 100.0,
        "retain_pct_by_support": {"L0b": 100.0, "math_a0": 100.0},
        "clears_by_save": {250: True, 500: True, 1500: True},
        "parent_consistency_ok": True,
        "close_sibling_ok": True,
    }
    blob.update(overrides)
    return blob


def _full_synthetic_bank():
    return {a: _complete_blob() for a in ("U", "E", "R0", "RW")}


def test_parse_refuses_missing_field_no_default():
    blob = _complete_blob()
    del blob["acquire_pct"]
    with pytest.raises(BankInputsRefuse, match="missing required fields"):
        parse_required_arm_bank_blob("U", blob)


def test_parse_complete_refuses_empty_bank_inputs():
    with pytest.raises(BankInputsRefuse, match="MISSING"):
        parse_complete_bank_inputs(None)
    with pytest.raises(BankInputsRefuse, match="MISSING"):
        parse_complete_bank_inputs({})


def test_driver_formal_absent_refuses_before_runner(tmp_path: Path):
    calls: list[int] = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("runner must not be called on formal")

    result = run_forgotten_accum_training_equivalence_arms(
        runner=boom,
        model=object(),
        batch={},
        tensor_states={"A": make_bounded_tensor_state(
            "A", torch.zeros(4, dtype=torch.int8), 1.0,
            torch.zeros(4, dtype=torch.int16),
        )},
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        developer_validation=False,
        bank_inputs=None,
    )
    assert result.status == "FAILURE"
    assert result.fail_closed_class == FailClosedClass.BANK_INPUTS_INVALID.value
    assert BANK_INPUTS_INVALID in str(result.error)
    assert result.notes.get("bank_refuse_kind") == "MISSING"
    assert calls == []
    assert result.arm_call_counts == {"U": 0, "E": 0, "R0": 0, "RW": 0}
    assert not any((tmp_path / "arms").rglob("*.pt"))


def test_driver_formal_full_synthetic_also_refuses_before_runner(tmp_path: Path):
    calls: list[int] = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("runner must not be called")

    result = run_forgotten_accum_training_equivalence_arms(
        runner=boom,
        model=object(),
        batch={},
        tensor_states={"A": make_bounded_tensor_state(
            "A", torch.zeros(4, dtype=torch.int8), 1.0,
            torch.zeros(4, dtype=torch.int16),
        )},
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        developer_validation=False,
        bank_inputs=_full_synthetic_bank(),
    )
    assert result.status == "FAILURE"
    assert result.fail_closed_class == FailClosedClass.BANK_INPUTS_INVALID.value
    assert "UNRESOLVED_POLICY" in str(result.error)
    assert calls == []
    assert result.arm_call_counts == {"U": 0, "E": 0, "R0": 0, "RW": 0}


def test_launch_formal_refuses_before_materialize(monkeypatch, tmp_path: Path):
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization as mat_mod
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver as drv_mod

    mat_calls: list[int] = []
    driver_calls: list[int] = []

    def fake_materialize(**_kwargs):
        mat_calls.append(1)
        raise AssertionError("materialize must not run on formal scope-A")

    def fake_driver(**_kwargs):
        driver_calls.append(1)
        raise AssertionError("driver must not run on formal scope-A")

    monkeypatch.setattr(mat_mod, "materialize_run_arms_live_bundle", fake_materialize)
    monkeypatch.setattr(
        drv_mod, "run_forgotten_accum_training_equivalence_arms", fake_driver
    )
    monkeypatch.setattr(drv_mod, "assert_carrier_preflight", lambda **_k: None)

    args = SimpleNamespace(
        allow_gpu_launch=True,
        i_have_claude_run_arms_smoke_authority=False,
        formal_science=True,
        live_acc_carrier_selector="NONE",
        global_cap_contract="c1_banked_faithful_long_run_global_cap",
        eligible_scope="all-bitlinear",
        event_coded_flags_present=False,
        t_cut=500,
        runway_steps=1500,
        W=32,
        parent=str(tmp_path / "parent.pt"),
        parent_sha256="a" * 64,
        device="cpu",
        scratch_root=str(tmp_path / "scratch"),
        receipt_out=None,
    )
    receipt, code = launch_run_arms(args)
    assert code == EXIT_RUN_ARMS_BANK_INPUTS
    assert receipt["fail_closed_class"] == "BANK_INPUTS_INVALID"
    assert receipt["claimable_science"] is False
    assert receipt["bankable"] is False
    assert mat_calls == []
    assert driver_calls == []


def test_claim_coupling_blocks_unconditional_formal_true():
    receipt = {
        "status": "OK",
        "claimable_science": True,
        "bankable": True,
        "bank_receipts": None,
        "notes": {
            "bank_section": "suppressed",
            "ledger_claimable": False,
            "ledger_field_provenance": {"gpu_time_seconds": "SYNTHETIC"},
        },
    }
    out = apply_claim_coupling(receipt, mode="formal")
    assert out["claimable_science"] is False
    assert out["bankable"] is False
    assert out["ledger_synthetic"] is True


def test_exit_26_constant():
    assert EXIT_RUN_ARMS_BANK_INPUTS == 26
