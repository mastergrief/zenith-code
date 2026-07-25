"""PLAN_v10 pure-contract + CLI wiring tests (dedicated seam).

Bound by frozen PLAN_v10r4 + +1 1784891014883 + sixth-path amendment
+ defect-cycle 1784892185413. All new v10 reducer/bind/bar/wiring tests live HERE.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM1,
    ARM2,
    ARM3,
    FAMILY_F3,
    FAMILY_F4,
)
from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
    FORMAL150_CONTROL_SHA256,
    load_and_validate_control_baseline,
    pin_and_load_formal_control_baseline,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (
    backlog_bar_v10,
    build_v10_terminal_receipt,
    classify_forgetting_family_screen_v10,
    g0_valid_v10,
    h_bar_v10,
    pressure_bar_v10,
    recompute_deferred_survival_class,
    regime_exit_v10,
    validate_control_baseline_bind,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
)

REPO = Path(__file__).resolve().parents[3]
FORMAL = REPO / "artifacts/acc_entropy/pressure_metric_formal150_censor_null.json"
FORMAL_SHA = FORMAL150_CONTROL_SHA256

# Exact C2 echoes from formal artifact (bit-exact).
_C2_ECHO = {
    "measurements.demand.mean_ratio": 21563.959075520834,
    "measurements.demand.max_ratio": 26549.3818359375,
    "measurements.demand.frac_steps_ratio_ge_2": 0.94,
    "measurements.demand.n_steps": 150,
    "measurements.deferred_survival.N_events_evaluable": 91614389,
    "measurements.deferred_survival.N_survived_applied_within_H": 134048,
    "measurements.deferred_survival.N_never_applied_within_H": 91480341,
    "measurements.deferred_survival.N_events_censored_insufficient_followup": 26345583,
    "measurements.deferred_survival.N_events_evaluable_early": 51145534,
    "measurements.deferred_survival.N_events_evaluable_late": 40468855,
    "measurements.deferred_survival.N_never_applied_within_H_early": 51077741,
    "measurements.deferred_survival.N_never_applied_within_H_late": 40402600,
    "measurements.deferred_survival.deferred_never_apply_within_H_frac": 0.998536823729731,
    "measurements.deferred_survival.deferred_never_apply_within_H_frac_early": 0.998674507924778,
    "measurements.deferred_survival.deferred_never_apply_within_H_frac_late": 0.998362815058642,
    "measurements.deferred_survival.delta_never_apply": -0.0003116928661359708,
    "measurements.deferred_survival.deferred_survival_class": "stable_high",
    "measurements.H_bits_per_weight": 7.394114888250829,
    "measurements.H_trajectory[0].H_bits_per_weight": 5.277003270954126,
    "probes.acq_step0_count": 64,
    "probes.acq_final_count": 22,
    "probes.acq_delta_count": -42,
    "probes.ret_step0_count": 64,
    "probes.ret_final_count": 10,
    "probes.retention_ok": False,
    "classifier.family": "pressure_source_backlog",
    "classifier.stop_reason": "R1_pressure",
}


def _arm_ok(**over):
    # counts → collapsing (late-early delta <= -0.10) with consistent fractions
    base = {
        "H_final": 2.0,
        "n_flips": 5000,
        "q_changed_count": 200,
        "n_applied_drains": 10000,
        "lifetime_censored_frac": 0.99,  # non-gating under v10
        "retention_ok": True,
        "acq_delta_count": 1,
        "mean_ratio": 1.0,
        "max_ratio": 2.0,
        "frac_steps_ratio_ge_2": 0.2,
        "n_steps": 150,
        "receipt_steps": 150,
        "N_events_evaluable": 1000,
        "N_survived_applied_within_H": 400,
        "N_never_applied_within_H": 600,
        "N_events_censored_insufficient_followup": 10,
        "N_events_evaluable_early": 500,
        "N_events_evaluable_late": 500,
        "N_never_applied_within_H_early": 350,
        "N_never_applied_within_H_late": 250,
        "never_frac": 0.6,
        "early_never_frac": 0.7,
        "late_never_frac": 0.5,
        "delta_never_apply": -0.2,
        "deferred_survival_class": "collapsing",
    }
    base.update(over)
    return base


def test_contract_module_has_no_file_io():
    src = (
        REPO / "calm/hrm_text_158/native_full_stack/forgetting_screen_v10_contract.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "pathlib" not in imports
    assert "json" not in imports
    assert "hashlib" not in imports
    assert "Path(" not in src
    assert "read_bytes" not in src
    assert "read_text" not in src


def test_control_bind_formal_artifact_bit_exact():
    bind = load_and_validate_control_baseline(
        FORMAL, expected_sha256=FORMAL_SHA, require_exact_c2_echo=_C2_ECHO
    )
    assert bind["ok"] is True
    assert bind["c2_key_count"] == 27
    assert bind["H_control_final"] == 7.394114888250829


def test_control_bind_sha_mismatch_fail_closed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(FORMAL.read_text())
    bind = load_and_validate_control_baseline(p, expected_sha256="0" * 64)
    assert bind["ok"] is False
    assert bind["reason"] == "control_baseline_sha_mismatch"


def test_pin_refuses_alternate_self_consistent_sha(tmp_path):
    """Self-consistent alternate sha+artifact MUST fail at pin (not after load)."""
    obj = json.loads(FORMAL.read_text())
    obj["batch"] = 7  # mutate
    alt = tmp_path / "alt.json"
    raw = json.dumps(obj).encode("utf-8")
    alt.write_bytes(raw)
    alt_sha = hashlib.sha256(raw).hexdigest()
    assert alt_sha != FORMAL_SHA
    bind = pin_and_load_formal_control_baseline(alt, supplied_sha256=alt_sha)
    assert bind["ok"] is False
    assert bind["reason"] == "control_baseline_sha_not_pinned"


def test_control_bind_missing_c2_key_fail_closed():
    obj = json.loads(FORMAL.read_text())
    del obj["measurements"]["demand"]["mean_ratio"]
    bind = validate_control_baseline_bind(
        obj, expected_sha256=FORMAL_SHA, actual_sha256=FORMAL_SHA
    )
    assert bind["ok"] is False
    assert bind["reason"] == "c2_key_missing"


def test_geometry_only_real_artifact_paths():
    obj = json.loads(FORMAL.read_text())
    assert "device" not in obj and "seed" not in obj
    bind = validate_control_baseline_bind(
        obj, expected_sha256=FORMAL_SHA, actual_sha256=FORMAL_SHA
    )
    assert bind["ok"] is True
    obj2 = dict(obj)
    obj2["batch"] = 1
    bad = validate_control_baseline_bind(
        obj2, expected_sha256=FORMAL_SHA, actual_sha256=FORMAL_SHA
    )
    assert bad["ok"] is False


def test_lcf_alone_does_not_invalidate_g0():
    m = _arm_ok(lifetime_censored_frac=0.99)
    ok, reason = g0_valid_v10(m)
    assert ok is True
    assert reason is None


def test_g0_cohort_and_conservation():
    ok, _ = g0_valid_v10(_arm_ok(N_events_evaluable_early=50))
    assert ok is False
    ok2, reason = g0_valid_v10(
        _arm_ok(N_survived_applied_within_H=100, N_never_applied_within_H=100)
    )
    assert ok2 is False
    assert reason == "conservation_eval_surv_never"


def test_g0_vacuous_and_unknown_class():
    ok, reason = g0_valid_v10(_arm_ok(deferred_survival_class="vacuous"))
    assert ok is False and reason == "class_vacuous"
    ok2, reason2 = g0_valid_v10(_arm_ok(deferred_survival_class="not_a_class"))
    assert ok2 is False and reason2 == "class_unrecognized"


def test_g0_class_recompute_mismatch_fail_closed():
    # Raw counts imply collapsing, but receipt claims stable_high → fail.
    ok, reason = g0_valid_v10(_arm_ok(deferred_survival_class="stable_high"))
    assert ok is False
    assert reason == "class_recompute_mismatch"
    assert (
        recompute_deferred_survival_class(
            n_eval=1000, n_early=500, n_late=500, never_frac=0.6, delta=-0.2
        )
        == "collapsing"
    )


def test_g0_full_surface_requires_max_ratio_and_delta():
    m = _arm_ok()
    del m["max_ratio"]
    ok, reason = g0_valid_v10(m)
    assert ok is False and reason == "missing_field:max_ratio"


def test_g0_n_steps_must_equal_150_and_receipt_steps():
    ok, reason = g0_valid_v10(_arm_ok(n_steps=64, receipt_steps=150))
    assert ok is False and reason == "n_steps_not_150"
    ok2, reason2 = g0_valid_v10(_arm_ok(n_steps=150, receipt_steps=64))
    assert ok2 is False and reason2 == "n_steps_receipt_steps_mismatch"
    ok3, _ = g0_valid_v10(_arm_ok(n_steps=150, receipt_steps=150))
    assert ok3 is True


def test_pressure_bar_or_regime_exit_boundaries():
    assert regime_exit_v10(mean_ratio_arm=10.0, frac_ge2_arm=0.49) is True
    assert pressure_bar_v10(
        mean_ratio_arm=10.0, frac_ge2_arm=0.49, mean_ratio_control=20.0
    )
    assert pressure_bar_v10(
        mean_ratio_arm=15.0, frac_ge2_arm=0.4, mean_ratio_control=20.0
    ) is True
    assert regime_exit_v10(mean_ratio_arm=2.0, frac_ge2_arm=0.50) is False
    assert regime_exit_v10(mean_ratio_arm=1.9, frac_ge2_arm=0.50) is True
    assert regime_exit_v10(mean_ratio_arm=2.0, frac_ge2_arm=0.49) is True


def test_backlog_bar_enum_first_unknown_cannot_win():
    assert (
        backlog_bar_v10(
            never_frac_arm=0.99, never_frac_control=0.99, klass="weird"
        )
        is False
    )
    assert backlog_bar_v10(
        never_frac_arm=0.99, never_frac_control=0.99, klass="collapsing"
    )
    assert backlog_bar_v10(
        never_frac_arm=0.80, never_frac_control=0.99, klass="stable_high"
    )


def test_h_alone_cannot_win():
    metrics = {
        ARM1: _arm_ok(
            H_final=1.0,
            mean_ratio=20000.0,
            max_ratio=25000.0,
            frac_steps_ratio_ge_2=0.9,
            never_frac=0.6,
            early_never_frac=0.6,
            late_never_frac=0.6,
            delta_never_apply=0.0,
            N_never_applied_within_H_early=300,
            N_never_applied_within_H_late=300,
            deferred_survival_class="stable_high",
        ),
        ARM2: _arm_ok(H_final=6.0),
        ARM3: _arm_ok(H_final=6.0),
    }
    out = classify_forgetting_family_screen_v10(
        control_baseline_ok=True,
        H_control_final=7.394114888250829,
        mean_ratio_control=21563.959075520834,
        never_frac_control=0.998536823729731,
        arm_metrics=metrics,
    )
    assert ARM1 in out["E"]
    assert ARM1 not in out["W"]
    assert out["family"] == FAMILY_F4
    assert "R0_null" in str(out["stop_reason"])


def test_winner_requires_all_three_bars():
    metrics = {
        ARM1: _arm_ok(H_final=2.0),
        ARM2: _arm_ok(H_final=6.0),
        ARM3: _arm_ok(H_final=1.5),
    }
    out = classify_forgetting_family_screen_v10(
        control_baseline_ok=True,
        H_control_final=7.394114888250829,
        mean_ratio_control=21563.959075520834,
        never_frac_control=0.998536823729731,
        arm_metrics=metrics,
    )
    assert ARM3 in out["W"]
    assert out["family"] == FAMILY_F3


# ---- CLI wiring ----


def _load_screen_module():
    import importlib.util

    path = REPO / "scripts/hrm_text_158_forgetting_mechanism_screen.py"
    spec = importlib.util.spec_from_file_location("forget_screen_cli_v10", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj))


def _r1_surface(*, klass="collapsing"):
    """Count-consistent collapsing R1 surface for synthetic mechanism receipts."""
    return {
        "demand": {
            "mean_ratio": 1.0,
            "max_ratio": 2.0,
            "frac_steps_ratio_ge_2": 0.2,
            "n_steps": 150,
        },
        "deferred_survival": {
            "N_events_evaluable": 1000,
            "N_survived_applied_within_H": 400,
            "N_never_applied_within_H": 600,
            "N_events_censored_insufficient_followup": 10,
            "N_events_evaluable_early": 500,
            "N_events_evaluable_late": 500,
            "N_never_applied_within_H_early": 350,
            "N_never_applied_within_H_late": 250,
            "deferred_never_apply_within_H_frac": 0.6,
            "deferred_never_apply_within_H_frac_early": 0.7,
            "deferred_never_apply_within_H_frac_late": 0.5,
            "delta_never_apply": -0.2,
            "deferred_survival_class": klass,
        },
    }


def _minimal_arm(arm: str, **meas_over):
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ACQUISITION_SELECTION_SHA256,
        DEFAULT_PARENT_SHA256,
        IDENTITY_SELECTION_SHA256,
        PHASE0_SCREEN_ID,
    )
    from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
        AUTHORITY_DISPATCH,
        PLAN_SHA256,
    )

    r1 = _r1_surface()
    m = {
        "H_bits_per_weight": 2.0,
        "n_flips": 5000,
        "q_changed_count": 200,
        "n_applied_drains": 10000,
        "lifetime_censored_frac": 0.99,
        "demand": r1["demand"],
        "deferred_survival": r1["deferred_survival"],
    }
    m.update(meas_over)
    return {
        "arm": arm,
        "screen": PHASE0_SCREEN_ID,
        "schema_only": False,
        "correctness_smoke": False,
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "steps": 150,
        "batch": 8,
        "topk": 1024,
        "banked_sha": {
            "before": DEFAULT_PARENT_SHA256,
            "after": DEFAULT_PARENT_SHA256,
            "match": True,
        },
        "frozen_scale_sha": {
            "before": "scale_shared_abc",
            "after": "scale_shared_abc",
            "match": True,
        },
        "q_sha": {"before": "q_shared_def", "after": "q_shared_def"},
        "route_counters": {
            "n_fixed_qscale_forwards": 10,
            "n_bitlinear_dynamic_forwards": 0,
            "n_eligible_keys": 32,
            "n_credit_grads_present": 32,
        },
        "measurements": m,
        "probes": {
            "skipped": False,
            "acquisition_selection_sha256": ACQUISITION_SELECTION_SHA256,
            "identity_selection_sha256": IDENTITY_SELECTION_SHA256,
            "acquisition_n": 64,
            "retention_n": 64,
            "acq_step0_count": 64,
            "acq_final_count": 65,
            "acq_delta_count": 1,
            "retention_step0_count": 64,
            "retention_final_count": 64,
            "retention_ok": True,
        },
    }


def test_cli_positive_baseline_reaches_v10_classification(tmp_path):
    mod = _load_screen_module()
    arms = []
    for arm, H in ((ARM1, 2.0), (ARM2, 2.5), (ARM3, 1.5)):
        p = tmp_path / f"{arm}.json"
        _write_json(p, _minimal_arm(arm, H_bits_per_weight=H))
        arms.append(str(p))
    out = tmp_path / "out.json"
    ns = argparse.Namespace(
        control_baseline_json=str(FORMAL),
        control_baseline_sha256=FORMAL_SHA,
        arm_receipts=",".join(arms),
        output_json=str(out),
        phase0_receipt=None,
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
    )
    assert mod._run_aggregate_phase1(ns) == 0
    rec = json.loads(out.read_text())
    assert rec["screen"] == "forgetting_mechanism_phase1/v10"
    assert rec["control_baseline_ok"] is True
    assert rec["arms_classified"] is True
    assert rec["transition"] is None
    assert rec["control_arm0_receipt_present"] is False
    assert rec["family"] in {FAMILY_F3, FAMILY_F4, "F1_decay_leak", "F2_ttl_age_drain"}


@pytest.mark.parametrize(
    "mutate",
    [
        "omit_json",
        "omit_sha",
        "wrong_sha",
        "legacy_phase0_receipt",
        "legacy_predecessor",
        "legacy_censor_cleared",
        "four_receipts",
    ],
)
def test_cli_fail_closed_negatives(tmp_path, mutate):
    mod = _load_screen_module()
    arms = []
    for arm in (ARM1, ARM2, ARM3):
        p = tmp_path / f"{arm}.json"
        _write_json(p, _minimal_arm(arm))
        arms.append(str(p))
    out = tmp_path / "out.json"
    ns = argparse.Namespace(
        control_baseline_json=str(FORMAL),
        control_baseline_sha256=FORMAL_SHA,
        arm_receipts=",".join(arms),
        output_json=str(out),
        phase0_receipt=None,
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
    )
    if mutate == "omit_json":
        ns.control_baseline_json = None
        with pytest.raises(SystemExit):
            mod._run_aggregate_phase1(ns)
    elif mutate == "omit_sha":
        ns.control_baseline_sha256 = None
        with pytest.raises(SystemExit):
            mod._run_aggregate_phase1(ns)
    elif mutate == "wrong_sha":
        ns.control_baseline_sha256 = "0" * 64
        assert mod._run_aggregate_phase1(ns) == 0
        rec = json.loads(out.read_text())
        assert rec["arms_classified"] is False
        assert rec["family"] == FAMILY_F4
        assert rec["stop_reason"] == "control_baseline_sha_not_pinned"
    elif mutate == "four_receipts":
        from calm.hrm_text_158.native_full_stack.family_classifier import ARM0

        p0 = tmp_path / "arm0.json"
        _write_json(p0, _minimal_arm(ARM0))
        ns.arm_receipts = ",".join([str(p0)] + arms)
        with pytest.raises(SystemExit, match="exactly 3"):
            mod._run_aggregate_phase1(ns)
    elif mutate == "legacy_phase0_receipt":
        ns.phase0_receipt = str(tmp_path / "p0.json")
        with pytest.raises(SystemExit, match="HARD-REFUSE"):
            mod._run_aggregate_phase1(ns)
    elif mutate == "legacy_predecessor":
        ns.phase0_predecessor_receipt = str(tmp_path / "pred.json")
        with pytest.raises(SystemExit, match="HARD-REFUSE"):
            mod._run_aggregate_phase1(ns)
    else:
        ns.phase0_censor_cleared = 1
        with pytest.raises(SystemExit, match="HARD-REFUSE"):
            mod._run_aggregate_phase1(ns)


def test_cli_alternate_sha_pin_negative(tmp_path):
    mod = _load_screen_module()
    obj = json.loads(FORMAL.read_text())
    obj["batch"] = 7
    alt = tmp_path / "alt.json"
    raw = json.dumps(obj).encode("utf-8")
    alt.write_bytes(raw)
    alt_sha = hashlib.sha256(raw).hexdigest()
    arms = []
    for arm in (ARM1, ARM2, ARM3):
        p = tmp_path / f"{arm}.json"
        _write_json(p, _minimal_arm(arm))
        arms.append(str(p))
    out = tmp_path / "out.json"
    ns = argparse.Namespace(
        control_baseline_json=str(alt),
        control_baseline_sha256=alt_sha,
        arm_receipts=",".join(arms),
        output_json=str(out),
        phase0_receipt=None,
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
    )
    assert mod._run_aggregate_phase1(ns) == 0
    rec = json.loads(out.read_text())
    assert rec["stop_reason"] == "control_baseline_sha_not_pinned"
    assert rec["arms_classified"] is False


def test_schema_only_is_non_authoritative_no_control_bind(tmp_path, monkeypatch):
    mod = _load_screen_module()
    ns = argparse.Namespace(
        schema_only=True,
        aggregate_phase1=False,
        ckpt_path=None,
        phase0_receipt=None,
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
        control_baseline_json=None,
        control_baseline_sha256=None,
    )
    mod._refuse_legacy_phase0_flags(ns)
    assert ns.control_baseline_json is None


def test_v10_receipt_identity_rejects_v9_accepts_v10():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        V10ArmReceiptContractError,
        validate_three_mechanism_arm_receipts_v10,
    )
    from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
        AUTHORITY_DISPATCH,
        AUTHORITY_DISPATCH_V9,
        COMMIT_SURFACE_FILES,
        PLAN_SHA256,
        PLAN_V9_SHA256,
    )

    assert PLAN_SHA256 != PLAN_V9_SHA256
    assert AUTHORITY_DISPATCH != AUTHORITY_DISPATCH_V9
    assert len(COMMIT_SURFACE_FILES) == 8
    assert (
        "artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10.json"
        in COMMIT_SURFACE_FILES
    )    # Frozen constant matches on-disk plan bytes (no self-ref inside plan JSON).
    plan_bytes = (REPO / "artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10.json").read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == PLAN_SHA256
    plan_obj = json.loads(plan_bytes.decode("utf-8"))
    assert "plan_sha256" not in plan_obj  # no self-reference

    good = {arm: _minimal_arm(arm) for arm in (ARM1, ARM2, ARM3)}
    shared = validate_three_mechanism_arm_receipts_v10(
        good,
        expected_plan_sha256=PLAN_SHA256,
        expected_parent_sha256=good[ARM1]["banked_sha"]["before"],
        expected_authority_dispatch=AUTHORITY_DISPATCH,
    )
    assert shared["control"] == "formal150_artifact_sole"

    bad_v9 = {arm: _minimal_arm(arm) for arm in (ARM1, ARM2, ARM3)}
    for r in bad_v9.values():
        r["plan_sha256"] = PLAN_V9_SHA256
        r["authority_dispatch"] = AUTHORITY_DISPATCH_V9
    with pytest.raises(V10ArmReceiptContractError, match="plan_sha256"):
        validate_three_mechanism_arm_receipts_v10(
            bad_v9,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=bad_v9[ARM1]["banked_sha"]["before"],
            expected_authority_dispatch=AUTHORITY_DISPATCH,
        )
    # v9 authority alone also rejected when plan is v10 but authority is v9
    bad_auth = {arm: _minimal_arm(arm) for arm in (ARM1, ARM2, ARM3)}
    for r in bad_auth.values():
        r["authority_dispatch"] = AUTHORITY_DISPATCH_V9
    with pytest.raises(V10ArmReceiptContractError, match="authority_dispatch"):
        validate_three_mechanism_arm_receipts_v10(
            bad_auth,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=bad_auth[ARM1]["banked_sha"]["before"],
            expected_authority_dispatch=AUTHORITY_DISPATCH,
        )


def _drive_cpu_lifecycle_store(*, steps: int = 150) -> DeviceLifecycleStore:
    """CPU/static producer path — DeviceLifecycleStore driven with synthetic masks."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
        run_full_per_step_lifecycle,
    )

    shapes = {"layer.w": (32, 16)}
    store = DeviceLifecycleStore.from_arm_shapes(shapes, steps=steps, device="cpu")
    n = 32 * 16
    for t in range(1, steps + 1):
        cand = {"layer.w": torch.ones(shapes["layer.w"], dtype=torch.bool)}
        if t <= steps // 4:
            applied = {"layer.w": torch.zeros(shapes["layer.w"], dtype=torch.bool)}
        elif t <= steps // 2:
            applied = {"layer.w": torch.ones(shapes["layer.w"], dtype=torch.bool)}
        else:
            applied = {"layer.w": (torch.arange(n).reshape(32, 16) % 3 == 0)}
        ep_before = {
            "layer.w": torch.full(shapes["layer.w"], max(0, t - 5), dtype=torch.int32)
        }
        ep_after = {
            "layer.w": torch.full(
                shapes["layer.w"], t if t % 2 else max(0, t - 5), dtype=torch.int32
            )
        }
        run_full_per_step_lifecycle(
            store,
            candidate_masks=cand,
            applied_masks=applied,
            episode_start_before=ep_before,
            episode_start_after=ep_after,
            step=t,
            n_candidates=int(cand["layer.w"].sum().item()),
            n_applied=int(applied["layer.w"].sum().item()),
        )
    store.finalize_window(final_step=steps)
    return store


def test_assemble_arm_receipt_emits_live_r1_passes_v10_contract(tmp_path):
    """Real assemble_arm_receipt path — no manual demand/deferred_survival injection."""
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ACQUISITION_SELECTION_SHA256,
        DEFAULT_PARENT_SHA256,
        IDENTITY_SELECTION_SHA256,
        V10ArmReceiptContractError,
        validate_three_mechanism_arm_receipts_v10,
    )
    from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
        build_phase1_probe_sets,
    )
    from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
        AUTHORITY_DISPATCH,
        PLAN_SHA256,
        PLAN_V9_SHA256,
        assemble_arm_receipt,
    )

    steps = 150
    store = _drive_cpu_lifecycle_store(steps=steps)
    shape = (8, 8)
    # Tiny residency tensors for receipt assembly (H/lcf path).
    acc = {"layer.w": torch.ones(shape, dtype=torch.int16)}
    episode_start = {"layer.w": torch.ones(shape, dtype=torch.int32)}
    flip_count = {"layer.w": torch.ones(shape, dtype=torch.int32)}
    q_levels = {"layer.w": torch.zeros(shape, dtype=torch.int8)}
    frozen_scales = {"layer.w": torch.ones(shape, dtype=torch.float32)}
    ckpt = tmp_path / "parent.pt"
    ckpt.write_bytes(b"v10-assemble-characterization-ckpt")
    parent_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    loop_out = {
        "acc": acc,
        "episode_start": episode_start,
        "flip_count": flip_count,
        "lifetimes": [1, 2, 3],
        "credited_mass": 10,
        "n_flips": 5000,
        "q_changed_count": 200,
        "n_applied_drains": 10000,
        "batch_rng_base": 1000,
        "excluded_hit_count": 0,
        "H_trajectory": [
            {
                "step": steps,
                "H_bits_per_weight": 2.0,
                "support": "test",
                "denominator": "acc.numel()",
                "estimator": "shannon_unique_counts",
            }
        ],
        "train_route_counters": {
            "n_fixed_qscale_forwards": 10,
            "n_bitlinear_dynamic_forwards": 0,
            "n_eligible_keys": 32,
            "n_credit_grads_present": 32,
        },
        "pressure_telemetry": store,
    }
    probe_sets = build_phase1_probe_sets()
    # Match assemble's scale/q sha formula (hash of concatenated per-tensor hex digests).
    def _tsha(t):
        return hashlib.sha256(
            t.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    scale_before = hashlib.sha256(
        b"".join(_tsha(frozen_scales[n]).encode() for n in sorted(frozen_scales))
    ).hexdigest()
    q_before = hashlib.sha256(
        b"".join(_tsha(q_levels[n]).encode() for n in sorted(q_levels))
    ).hexdigest()
    args = argparse.Namespace(
        arm=ARM1,
        steps=steps,
        batch=8,
        topk=1024,
        correctness_smoke=False,
        skip_probes=False,
    )
    emitted = assemble_arm_receipt(
        args=args,
        device="cpu",
        sha_before=parent_sha,
        scale_sha_before=scale_before,
        q_sha_before=q_before,
        frozen_scales=frozen_scales,
        q_levels=q_levels,
        ckpt_path=str(ckpt),
        probe_sets=probe_sets,
        acq_step0=64,
        ret_step0=64,
        acq_final=65,
        ret_final=64,
        loop_out=loop_out,
    )
    # EMITTED by assemble — no manual R1 injection.
    assert "demand" in emitted["measurements"]
    assert "deferred_survival" in emitted["measurements"]
    assert emitted["measurements"]["demand"]["n_steps"] == 150
    assert emitted["plan_sha256"] == PLAN_SHA256
    assert emitted["plan_sha256"] != PLAN_V9_SHA256
    assert emitted["authority_dispatch"] == AUTHORITY_DISPATCH
    assert emitted["launch_authority_dispatch"] is None
    assert emitted["defect_cycle_authority"] == "1784892185413-a4f0e9bb"
    # Align banked parent to DEFAULT for three-arm shared contract (geometry/R1 focus).
    # Re-bind banked to expected parent constant used by live aggregate.
    emitted["banked_sha"] = {
        "before": DEFAULT_PARENT_SHA256,
        "after": DEFAULT_PARENT_SHA256,
        "match": True,
    }
    # Probe pins already from build_phase1_probe_sets; assert present.
    assert emitted["probes"]["acquisition_selection_sha256"] == ACQUISITION_SELECTION_SHA256
    assert emitted["probes"]["identity_selection_sha256"] == IDENTITY_SELECTION_SHA256

    # Clone into three mechanism arms — R1 fields remain assemble-emitted (not hand-built).
    by_arm = {}
    for arm in (ARM1, ARM2, ARM3):
        r = json.loads(json.dumps(emitted))  # deep copy JSON-safe
        r["arm"] = arm
        by_arm[arm] = r
    shared = validate_three_mechanism_arm_receipts_v10(
        by_arm,
        expected_plan_sha256=PLAN_SHA256,
        expected_parent_sha256=DEFAULT_PARENT_SHA256,
        expected_authority_dispatch=AUTHORITY_DISPATCH,
    )
    assert shared["steps"] == 150
    assert shared["mechanism_arms"] == [ARM1, ARM2, ARM3]
    # v9 identity on emitted receipt must fail the v10 validator
    by_arm_v9 = {a: dict(r) for a, r in by_arm.items()}
    for r in by_arm_v9.values():
        r["plan_sha256"] = PLAN_V9_SHA256
    with pytest.raises(V10ArmReceiptContractError):
        validate_three_mechanism_arm_receipts_v10(
            by_arm_v9,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=DEFAULT_PARENT_SHA256,
            expected_authority_dispatch=AUTHORITY_DISPATCH,
        )
