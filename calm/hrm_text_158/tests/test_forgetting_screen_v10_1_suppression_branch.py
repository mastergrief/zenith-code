"""PLAN_v10.1r8 suppression-branch + hotpath + adversarial tests (focused).

DO NOT put these into test_forgetting_screen_v10_contract.py (VALIDATE_ONLY).
Producer/CLI/cost live in test_forgetting_screen_v10_1_producer_toggle.py.
"""
from __future__ import annotations

import hashlib
import inspect

import pytest
import torch

from calm.hrm_text_158.native_full_stack.forgetting_screen_pre_post_telemetry import (
    PrePostTransformAccumulator,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_1_contract import (
    AUTHORITY_DISPATCH_V10_1,
    CONTROL_CREDITED_MASS,
    FORMAL_BATCH,
    FORMAL_DEVICE,
    FORMAL_STEPS,
    FORMAL_TOPK,
    H_SAMPLE_STEPS,
    PLAN_V10_1_PATH,
    PLAN_V10_1_SHA256,
    PRE_POST_SCHEMA,
    SUPPRESSION_ARM,
    TERMINAL_LABEL_CANONICAL,
    TRANSFER_CARDINALITY,
    TRANSFER_LAW,
    build_exhaustive_transfer_table,
    build_terminal_label_canonical,
    classify_discriminator_branch,
    credited_mass_ratio_in_band,
    g0_valid_v10_1,
    pre_post_evidence_schema_valid,
    suppression_diagnostic_match,
    suppression_disposition,
    transfer_pair,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (
    g0_valid_v10,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
    assert_hotpath_sync_allowlist,
    assert_pre_post_telemetry_single_d2h,
)
from calm.hrm_text_158.native_full_stack.screen_receipt_output import PLAN_SHA256


def _valid_pre_post(**overrides):
    base = {
        "schema": PRE_POST_SCHEMA,
        "law": TRANSFER_LAW,
        "move_abs_bins": {"1": 100},
        "move_nonzero_count": 100,
        "post_projection": {"nonzero": 80, "abs_max": 1, "abs_p50": 1.0, "abs_p90": 1.0},
        "post_decay": {"nonzero": 0, "abs_max": 0, "abs_p50": 0.0, "abs_p90": 0.0},
        "pre_nonzero_to_post_zero_count": 80,
        "pre_nonzero_to_post_zero_frac": 1.0,
        "post_decay_candidate_count": 0,
        "law_mismatch_count": 0,
        "steps_accumulated": 150,
    }
    base.update(overrides)
    return base


def _zero_ds():
    return {
        "N_events_evaluable": 0,
        "N_survived_applied_within_H": 0,
        "N_never_applied_within_H": 0,
        "N_events_evaluable_early": 0,
        "N_events_evaluable_late": 0,
        "N_never_applied_within_H_early": 0,
        "N_never_applied_within_H_late": 0,
        "N_events_censored_insufficient_followup": 0,
        "deferred_never_apply_within_H_frac": None,
        "deferred_survival_frac": None,
        "deferred_never_apply_within_H_frac_early": None,
        "deferred_never_apply_within_H_frac_late": None,
        "delta_never_apply": None,
        "deferred_survival_class": "vacuous",
    }


def _qhex(ch: str = "a") -> str:
    return (ch * 64)[:64]


def _suppression_receipt(**overrides):
    r = {
        "arm": SUPPRESSION_ARM,
        "steps": 150,
        "batch": 8,
        "topk": 1024,
        "device": "cuda:0",
        "plan_v10_1_path": PLAN_V10_1_PATH,
        "plan_v10_1_sha256": PLAN_V10_1_SHA256,
        "authority_dispatch_v10_1": AUTHORITY_DISPATCH_V10_1,
        "pinned_control_sha256": "5e593454f0ddffb946692e09913da5df1ddfe0f2f11aaaf3fb663a2f00fbcfdb",
        "pre_post_telemetry": True,
        "q_sha": {"before": _qhex("a"), "after": _qhex("a")},
        "route_counters": {
            "n_fixed_qscale_forwards": 1024,
            "n_bitlinear_dynamic_forwards": 0,
        },
        "probes": {
            "skipped": False,
            "acq_delta_count": 0,
            "acq_step0_count": 64,
            "acq_final_count": 64,
            "retention_ok": True,
            "retention_step0_count": 64,
            "retention_final_count": 64,
        },
        "measurements": {
            "n_flips": 0,
            "n_applied_drains": 0,
            "q_changed_count": 0,
            "credited_mass": int(0.9971 * CONTROL_CREDITED_MASS),
            "H_trajectory": [
                {"step": s, "H_bits_per_weight": -0.0} for s in H_SAMPLE_STEPS
            ],
            "demand": {
                "mean_ratio": 0.0,
                "max_ratio": 0.0,
                "frac_steps_ratio_ge_2": 0.0,
                "n_steps": 150,
            },
            "deferred_survival": _zero_ds(),
            "pre_post_transform": _valid_pre_post(),
        },
    }
    for k, v in overrides.items():
        if k in r["measurements"]:
            r["measurements"][k] = v
        elif k in (
            "deferred_survival",
            "demand",
            "pre_post_transform",
            "probes",
            "route_counters",
            "q_sha",
        ):
            if k in ("probes", "route_counters", "q_sha"):
                r[k] = v
            else:
                r["measurements"][k] = v
        else:
            r[k] = v
    return r


def test_hotpath_sync_allowlist_passes_with_telemetry_helper():
    observed = assert_hotpath_sync_allowlist()
    assert "forgetting_screen_pre_post_telemetry.py" in observed
    assert observed["screen_execution_loop.py"][".item("] == 0
    assert observed["forgetting_screen_pre_post_telemetry.py"][".item("] == 0


def test_d2h_ast_single_transfer_in_finalize_only():
    info = assert_pre_post_telemetry_single_d2h()
    assert info["finalize_cpu"] == 1
    assert info["accumulate_d2h"] == 0
    assert info["module_totals"][".cpu("] == 1


def test_transfer_table_cardinality_765_and_boundaries():
    rows = build_exhaustive_transfer_table()
    assert len(rows) == TRANSFER_CARDINALITY == 765
    by = {(r["acc"], r["move"]): r for r in rows}
    assert by[(127, 1)]["pre"] == 127
    assert by[(-127, -1)]["pre"] == -127
    assert transfer_pair(1, 0)[1] == 0


def test_mass_band_fail_closed_on_none():
    ok, ratio, reason = credited_mass_ratio_in_band(None)
    assert ok is False and ratio is None and reason is not None
    ok, _, reason = credited_mass_ratio_in_band(int(0.89 * CONTROL_CREDITED_MASS))
    assert not ok and reason == "credited_mass_ratio_outside_band"


def test_suppression_positive_and_g0():
    r = _suppression_receipt()
    assert suppression_diagnostic_match(r) == (True, None)
    ok, reason, meta = g0_valid_v10_1(r)
    assert ok and reason is None and meta["branch"] == "suppression_diagnostic"
    assert meta["disposition"]["family_winner"] is False
    assert meta["disposition"]["terminal_label"] == TERMINAL_LABEL_CANONICAL


def test_terminal_label_example_vs_rule():
    joined = build_terminal_label_canonical(
        arm=SUPPRESSION_ARM,
        law=TRANSFER_LAW,
        steps=FORMAL_STEPS,
        batch=FORMAL_BATCH,
        topk=FORMAL_TOPK,
        device=FORMAL_DEVICE,
    )
    assert joined == TERMINAL_LABEL_CANONICAL
    disp = suppression_disposition(
        arm=SUPPRESSION_ARM,
        law=TRANSFER_LAW,
        steps=FORMAL_STEPS,
        batch=FORMAL_BATCH,
        topk=FORMAL_TOPK,
        device=FORMAL_DEVICE,
    )
    assert disp["terminal_label"] == TERMINAL_LABEL_CANONICAL
    assert disp["arm"] == SUPPRESSION_ARM
    assert disp["receipt_valid_for_diagnosis"] is True
    assert "arm1_decay_leak" in disp["terminal_label"]
    assert "lambda_1_32" in disp["terminal_label"]
    assert "steps150" in disp["terminal_label"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm": "arm2_ttl_age_drain"},
        {"law": "trunc_other"},
        {"steps": 64},
        {"batch": 4},
        {"topk": 512},
        {"device": "cpu"},
    ],
)
def test_terminal_label_mismatch_not_canonical(kwargs):
    """Wrong arm/law/geometry must NOT silently mint the arm1 canonical label."""
    label = build_terminal_label_canonical(**kwargs)
    assert label != TERMINAL_LABEL_CANONICAL
    disp = suppression_disposition(**kwargs)
    assert disp["terminal_label"] != TERMINAL_LABEL_CANONICAL
    assert disp["receipt_valid_for_diagnosis"] is False
    if "arm" in kwargs:
        assert disp["arm"] == kwargs["arm"]
        assert disp["arm"] != SUPPRESSION_ARM


def test_build_terminal_label_wrong_arm_does_not_return_arm1_canonical():
    assert (
        build_terminal_label_canonical(arm="arm2_ttl_age_drain")
        != TERMINAL_LABEL_CANONICAL
    )
    assert build_terminal_label_canonical(arm="arm2_ttl_age_drain").startswith(
        "arm2_ttl_age_drain__"
    )


@pytest.mark.parametrize(
    "mut,substr",
    [
        (lambda r: r.__setitem__("arm", "arm2_ttl_age_drain"), "arm_not"),
        (lambda r: r.__setitem__("steps", 25), "steps_not"),
        (lambda r: r.__setitem__("batch", 4), "batch_not"),
        (
            lambda r: r["measurements"].__setitem__(
                "H_trajectory", [{"step": 25, "H_bits_per_weight": 0.0}]
            ),
            "H_trajectory",
        ),
        (
            lambda r: r["probes"].__setitem__("acq_final_count", 63),
            "acq_step0_final",
        ),
        (
            lambda r: r.__setitem__("q_sha", {"before": "zz", "after": "zz"}),
            "q_sha_not_hex64",
        ),
        (
            lambda r: r["measurements"].__setitem__(
                "pre_post_transform",
                _valid_pre_post(
                    post_projection={
                        "nonzero": 80,
                        "abs_max": 2,
                        "abs_p50": 1.0,
                        "abs_p90": 2.0,
                    }
                ),
            ),
            "abs_max_not_1",
        ),
        (
            lambda r: r["measurements"].__setitem__("credited_mass", None),
            "credited_mass",
        ),
        (
            lambda r: r["measurements"].__setitem__(
                "pre_post_transform", _valid_pre_post(steps_accumulated=25)
            ),
            "steps_accumulated",
        ),
        (
            lambda r: r["measurements"].__setitem__(
                "pre_post_transform",
                _valid_pre_post(move_abs_bins={"1": 50}, move_nonzero_count=100),
            ),
            "bin_sum",
        ),
        (lambda r: r.__setitem__("authority_dispatch_v10_1", "wrong"), "authority_dispatch"),
        (lambda r: r.__setitem__("plan_v10_1_path", "wrong/path.json"), "plan_v10_1_path"),
        (lambda r: r.__setitem__("pre_post_telemetry", False), "pre_post_telemetry"),
        (
            lambda r: r["measurements"].__setitem__(
                "pre_post_transform",
                _valid_pre_post(move_abs_bins={"0": 5, "1": 95}, move_nonzero_count=100),
            ),
            "zero_bin",
        ),
        # co_lead executed probes (exact-type / conservation)
        (
            lambda r: r["measurements"].__setitem__(
                "pre_post_transform",
                _valid_pre_post(move_abs_bins={}, move_nonzero_count=100),
            ),
            "bin_sum",
        ),
        (lambda r: r["probes"].__setitem__("retention_ok", "true"), "retention_not_ok"),
        (
            lambda r: r["measurements"]["demand"].__setitem__("mean_ratio", "0.0"),
            "demand_nonzero",
        ),
        (
            lambda r: r["measurements"].__setitem__("credited_mass", "2910513700"),
            "credited_mass",
        ),
        # exact skipped boolean (r12)
        (lambda r: r["probes"].pop("skipped", None), "probes_skipped_missing"),
        (lambda r: r["probes"].__setitem__("skipped", None), "probes_skipped_not_false"),
        (lambda r: r["probes"].__setitem__("skipped", 0), "probes_skipped_not_false"),
        (lambda r: r["probes"].__setitem__("skipped", ""), "probes_skipped_not_false"),
        (lambda r: r["probes"].__setitem__("skipped", "false"), "probes_skipped_not_false"),
    ],
)
def test_adversarial_matrix(mut, substr):
    r = _suppression_receipt()
    mut(r)
    ok, reason = suppression_diagnostic_match(r)
    assert not ok
    assert substr in str(reason)


@pytest.mark.parametrize(
    "skipped_val,reason_token",
    [
        ("__MISSING__", "probes_skipped_missing"),
        (None, "probes_skipped_not_false"),
        (0, "probes_skipped_not_false"),
        ("", "probes_skipped_not_false"),
        ("false", "probes_skipped_not_false"),
    ],
)
def test_probes_skipped_exact_false_negatives(skipped_val, reason_token):
    r = _suppression_receipt()
    if skipped_val == "__MISSING__":
        del r["probes"]["skipped"]
    else:
        r["probes"]["skipped"] = skipped_val
    ok, reason = suppression_diagnostic_match(r)
    assert not ok
    assert reason == reason_token


def test_empty_bins_positive_moves_schema_fail():
    """bins={} + move_nonzero_count=100 must fail conservation unconditionally."""
    pp = _valid_pre_post(move_abs_bins={}, move_nonzero_count=100)
    ok, reason = pre_post_evidence_schema_valid(pp)
    assert not ok and "bin_sum" in str(reason)


@pytest.mark.parametrize("bad", [64.9, "64", True, 1.0])
def test_exact_json_int_adversaries(bad):
    pp = _valid_pre_post(move_nonzero_count=bad)
    ok, reason = pre_post_evidence_schema_valid(pp)
    assert not ok
    assert reason is not None


def test_law_inconsistent_absmax2_routes_wiring():
    pp = _valid_pre_post(
        post_projection={"nonzero": 10, "abs_max": 2, "abs_p50": 1.0, "abs_p90": 2.0},
        pre_nonzero_to_post_zero_count=10,
        pre_nonzero_to_post_zero_frac=1.0,
        law_mismatch_count=0,
    )
    assert classify_discriminator_branch(pp) == "wiring_or_representation"
    r = _suppression_receipt(pre_post_transform=pp)
    assert suppression_diagnostic_match(r)[0] is False


def test_ordinary_g0_source_hash_stable():
    src = inspect.getsource(g0_valid_v10)
    assert (
        hashlib.sha256(src.encode()).hexdigest()
        == "d66916923f90b120957a704591cf66f2d7c7b9f507ca218d262076ffc3709f24"
    )
    assert PLAN_SHA256.startswith("2cb92e50")


def test_real_helper_schema_and_strong_s1_with_zero_moves():
    """Law-consistent fixture: zeros+ones moves; pre mag1; post zeros → strong_S1."""
    acc = PrePostTransformAccumulator(device="cpu")
    pre = {"w": torch.tensor([1, 1, 0, 1], dtype=torch.int16)}
    post = {"w": torch.tensor([0, 0, 0, 0], dtype=torch.int16)}
    moves = {"w": torch.tensor([1, 1, 0, 1], dtype=torch.int8)}
    acc.accumulate_step(
        moves=moves, acc_pre_decay=pre, acc_post_decay=post, n_cand_after_decay=0
    )
    out = acc.finalize()
    assert "0" not in out["move_abs_bins"]
    assert sum(out["move_abs_bins"].values()) == out["move_nonzero_count"] == 3
    ok, reason = pre_post_evidence_schema_valid(out)
    assert ok, reason
    assert classify_discriminator_branch(out) == "strong_S1"


def test_one_chain_helper_schema_classifier_receipt_g0():
    helper = PrePostTransformAccumulator(device="cpu")
    for _ in range(150):
        helper.accumulate_step(
            moves={"w": torch.tensor([1, 0, 1], dtype=torch.int8)},
            acc_pre_decay={"w": torch.tensor([1, 0, 1], dtype=torch.int16)},
            acc_post_decay={"w": torch.tensor([0, 0, 0], dtype=torch.int16)},
            n_cand_after_decay=0,
        )
    ppt = helper.finalize()
    assert ppt["steps_accumulated"] == 150
    assert pre_post_evidence_schema_valid(ppt)[0]
    assert classify_discriminator_branch(ppt) == "strong_S1"
    r = _suppression_receipt(pre_post_transform=ppt)
    assert suppression_diagnostic_match(r) == (True, None)
    ok, reason, meta = g0_valid_v10_1(r)
    assert ok and meta["branch"] == "suppression_diagnostic"
    assert meta["disposition"]["terminal_label"] == TERMINAL_LABEL_CANONICAL
