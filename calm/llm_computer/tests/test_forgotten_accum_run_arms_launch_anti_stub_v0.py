"""Anti-stub: REAL launch_run_arms must reach materialize -> driver."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
RUN_PY = REPO / "scripts/forgotten_accum_training_equivalence_run.py"


def _load_run_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module("scripts.forgotten_accum_training_equivalence_run")


def test_launch_run_arms_ast_not_unconditional_raise_stub():
    src = RUN_PY.read_text()
    tree = ast.parse(src)
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "launch_run_arms":
            func = node
            break
    assert func is not None
    # Supplement only: body must not be solely raise RuntimeError(...materialization deferred...)
    raises = [
        n
        for n in func.body
        if isinstance(n, ast.Raise)
    ]
    deferred = []
    for node in raises:
        if isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                if isinstance(arg, ast.Constant) and "materialization deferred" in str(
                    arg.value
                ):
                    deferred.append(node)
    assert deferred == [], "launch_run_arms still contains materialization-deferred raise stub"


def test_real_launch_reaches_materialize_then_driver(monkeypatch, tmp_path: Path):
    mod = _load_run_mod()
    calls = {"materialize": 0, "driver": 0}

    class _Bundle:
        model = object()
        batch = {"inputs": None}
        tensor_states = {"m": object()}
        eligible_modules = {"m": object()}
        device = "cpu"
        identity_inventory = {"ok": True}
        cadence_saver = lambda **kw: tmp_path / "x.pt"
        config = {"eligible_scope": "all-bitlinear", "use_ternary_bulk": True}

    def fake_materialize(**kwargs):
        calls["materialize"] += 1
        assert "parent_path" in kwargs
        return _Bundle()

    class _Result:
        def as_dict(self):
            return {"status": "OK", "fail_closed_class": None, "science_label": None}

    def fake_driver(**kwargs):
        calls["driver"] += 1
        assert kwargs["model"] is _Bundle.model
        assert kwargs["developer_validation"] is True  # smoke mode
        return _Result()

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization.materialize_run_arms_live_bundle",
        fake_materialize,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver.run_forgotten_accum_training_equivalence_arms",
        fake_driver,
    )
    # Also patch symbols imported inside launch_run_arms via local imports — patch modules.
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization as mat
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver as drv

    monkeypatch.setattr(mat, "materialize_run_arms_live_bundle", fake_materialize)
    monkeypatch.setattr(drv, "run_forgotten_accum_training_equivalence_arms", fake_driver)
    monkeypatch.setattr(drv, "assert_carrier_preflight", lambda **kw: None)

    args = SimpleNamespace(
        allow_gpu_launch=True,
        i_have_claude_run_arms_smoke_authority=True,
        formal_science=False,
        live_acc_carrier_selector="NONE",
        global_cap_contract="c1_banked_faithful_long_run_global_cap",
        eligible_scope="all-bitlinear",
        event_coded_flags_present=False,
        parent=str(tmp_path / "p.pt"),
        parent_sha256="abc",
        device="cpu",
        scratch_root=str(tmp_path / "scratch"),
        t_cut=2,
        runway_steps=4,
        W=1,
    )
    receipt, code = mod.launch_run_arms(args)
    assert calls["materialize"] == 1
    assert calls["driver"] == 1
    assert code == 0
    assert receipt["run_kind"] == "REAL_DEVICE_SMOKE"
    assert receipt["claimable_science"] is False
    assert receipt["bankable"] is False
    assert receipt["science_label"] is None


def test_real_launch_propagates_materialize_identity_refuse(monkeypatch, tmp_path: Path):
    """Negative control: materialize raises → REAL launch must not swallow into OK."""

    mod = _load_run_mod()
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization as mat
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver as drv
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
        IdentityRefuse,
    )

    def boom_materialize(**_kwargs):
        raise IdentityRefuse("injected materialize failure for swallow-hazard check")

    driver_calls = []

    def boom_if_driver(**_kwargs):
        driver_calls.append(1)
        raise AssertionError("driver must not run when materialize raises")

    monkeypatch.setattr(mat, "materialize_run_arms_live_bundle", boom_materialize)
    monkeypatch.setattr(drv, "assert_carrier_preflight", lambda **kw: None)
    monkeypatch.setattr(drv, "run_forgotten_accum_training_equivalence_arms", boom_if_driver)

    args = SimpleNamespace(
        allow_gpu_launch=True,
        i_have_claude_run_arms_smoke_authority=True,
        formal_science=False,
        live_acc_carrier_selector="NONE",
        global_cap_contract="c1_banked_faithful_long_run_global_cap",
        eligible_scope="all-bitlinear",
        event_coded_flags_present=False,
        parent=str(tmp_path / "p.pt"),
        parent_sha256="abc",
        device="cpu",
        scratch_root=str(tmp_path / "scratch"),
        t_cut=2,
        runway_steps=4,
        W=1,
    )
    with pytest.raises(IdentityRefuse, match="injected materialize failure"):
        mod.launch_run_arms(args)
    assert driver_calls == []
