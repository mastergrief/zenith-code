"""Option A C-b: geometry-gated UNVERIFIED refuse — production boundary attacks."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch as launch_mod
import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver as drv_mod
from calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch import (
    EXIT_RUN_ARMS_FAILURE,
    launch_run_arms,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    OPTION_A_ADMITTED_CHARACTERIZATION_GEOMETRIES,
    FailClosedClass,
    is_option_a_admitted_characterization_geometry,
    notes_indicate_unverified_ledger,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
    run_forgotten_accum_training_equivalence_arms,
)
from calm.llm_computer.tests.test_forgotten_accum_training_equivalence_science_driver_v0 import (
    _cpu_saver,
    _fake_runner_factory,
    _states,
)


def test_helper_marks_synthetic_and_forged_shapes_unverified():
    assert notes_indicate_unverified_ledger(None) is True
    assert notes_indicate_unverified_ledger({}) is True
    assert notes_indicate_unverified_ledger({"ledger_claimable": False}) is True
    forged = {
        "ledger_claimable": True,
        "ledger_field_provenance": {"update_count": "MEASURED"},
    }
    assert notes_indicate_unverified_ledger(forged) is True


def test_closed_set_cardinality_and_neighbors():
    assert len(OPTION_A_ADMITTED_CHARACTERIZATION_GEOMETRIES) == 2
    assert is_option_a_admitted_characterization_geometry(
        t_cut=2, runway_steps=4, rewarm_window_steps=1
    )
    assert is_option_a_admitted_characterization_geometry(
        t_cut=2, runway_steps=6, rewarm_window_steps=2
    )
    for triple in ((2, 4, 2), (2, 6, 1), (2, 5, 1), (1, 4, 1), (500, 1500, 32)):
        assert not is_option_a_admitted_characterization_geometry(
            t_cut=triple[0],
            runway_steps=triple[1],
            rewarm_window_steps=triple[2],
        )


def test_launch_and_driver_bind_same_predicate_object():
    """Object identity — not equal tuple values."""
    assert (
        launch_mod.is_option_a_admitted_characterization_geometry
        is drv_mod.is_option_a_admitted_characterization_geometry
    )


def test_admitted_262_runs_arms_then_unverified(tmp_path: Path):
    """Admitted (2,6,2): characterization executes; terminal OK unreachable."""
    resolved: list = []
    result = run_forgotten_accum_training_equivalence_arms(
        runner=_fake_runner_factory(resolved),
        model=object(),
        batch={},
        tensor_states=_states(),
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        t_cut=2,
        runway_steps=6,
        W=2,
        save_cadence=(2, 4, 6),
        cadence_saver=_cpu_saver,
        developer_validation=True,
    )
    assert result.status == "FAILURE"
    assert (
        result.fail_closed_class
        == FailClosedClass.A_LEDGER_ACCOUNTING_UNVERIFIED.value
    )
    assert result.arm_call_counts == {"U": 1, "E": 1, "R0": 1, "RW": 1}
    assert result.runner_invocations
    assert resolved  # runner seam invoked
    budget = 6 + 4 + 4 + 4
    assert budget == 18
    assert result.notes.get("ledger_claimable") is False


def test_rejected_geometries_zero_arm_calls(tmp_path: Path):
    for t_cut, runway, W in ((2, 4, 2), (2, 6, 1), (2, 5, 1), (1, 4, 1), (500, 1500, 32)):
        resolved: list = []
        result = run_forgotten_accum_training_equivalence_arms(
            runner=_fake_runner_factory(resolved),
            model=object(),
            batch={},
            tensor_states=_states(),
            eligible_modules={},
            device="cpu",
            experiment_root=tmp_path / f"g_{t_cut}_{runway}_{W}",
            t_cut=t_cut,
            runway_steps=runway,
            W=W,
            save_cadence=(min(t_cut, runway), runway),
            cadence_saver=_cpu_saver,
            developer_validation=True,
        )
        assert result.status == "FAILURE"
        assert (
            result.fail_closed_class
            == FailClosedClass.A_LEDGER_ACCOUNTING_UNVERIFIED.value
        )
        assert result.arm_call_counts == {"U": 0, "E": 0, "R0": 0, "RW": 0}
        assert result.runner_invocations == []
        assert resolved == []
        assert result.ledger is None


def test_launch_refuses_non_admitted_before_materialize(monkeypatch, tmp_path: Path):
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization as mat_mod
    import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver as drv

    mat_calls: list[int] = []
    drv_calls: list[int] = []

    def boom_mat(**_k):
        mat_calls.append(1)
        raise AssertionError("materialize must not run for non-admitted geometry")

    def boom_drv(**_k):
        drv_calls.append(1)
        raise AssertionError("driver must not run for non-admitted geometry")

    monkeypatch.setattr(mat_mod, "materialize_run_arms_live_bundle", boom_mat)
    monkeypatch.setattr(drv, "run_forgotten_accum_training_equivalence_arms", boom_drv)
    monkeypatch.setattr(drv, "assert_carrier_preflight", lambda **_k: None)

    args = SimpleNamespace(
        allow_gpu_launch=True,
        i_have_claude_run_arms_smoke_authority=True,
        formal_science=False,
        live_acc_carrier_selector="NONE",
        global_cap_contract="c1_banked_faithful_long_run_global_cap",
        eligible_scope="all-bitlinear",
        event_coded_flags_present=False,
        t_cut=2,
        runway_steps=4,
        W=2,  # neighbor of (2,4,1) — must refuse (RW vacuous)
        parent=str(tmp_path / "parent.pt"),
        parent_sha256="a" * 64,
        device="cpu",
        scratch_root=str(tmp_path / "scratch"),
        receipt_out=None,
    )
    receipt, code = launch_run_arms(args)
    assert code == EXIT_RUN_ARMS_FAILURE
    assert (
        receipt["fail_closed_class"]
        == FailClosedClass.A_LEDGER_ACCOUNTING_UNVERIFIED.value
    )
    assert mat_calls == []
    assert drv_calls == []


def test_no_verified_valid_symbol_in_option_a_surface():
    roots = [
        Path(
            "calm/hrm_text_158/native_full_stack/"
            "forgotten_accum_training_equivalence_contracts.py"
        ),
        Path(
            "calm/hrm_text_158/native_full_stack/"
            "forgotten_accum_training_equivalence_science_driver.py"
        ),
        Path(
            "calm/hrm_text_158/native_full_stack/"
            "forgotten_accum_run_arms_launch.py"
        ),
    ]
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "VERIFIED_VALID", path
            if isinstance(node, ast.Attribute):
                assert node.attr != "VERIFIED_VALID", path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == "VERIFIED_VALID":
                    raise AssertionError(f"VERIFIED_VALID string literal in {path}")
    params = list(inspect.signature(notes_indicate_unverified_ledger).parameters)
    assert params == ["notes"]
    # No mode/smoke/bool bypass parameters on the predicate.
    pred_params = list(
        inspect.signature(is_option_a_admitted_characterization_geometry).parameters
    )
    assert pred_params == ["t_cut", "runway_steps", "rewarm_window_steps"]
