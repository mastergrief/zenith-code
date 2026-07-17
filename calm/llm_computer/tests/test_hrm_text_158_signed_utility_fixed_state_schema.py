"""CPU-static tests for signed_utility_fixed_state_schema (PLAN v6 D1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    classify_signed_utility,
    epsilon_from_noop,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema import (
    REQUIRED_PHASE_MARKER_NAMES,
    SCHEMA_PREFLIGHT,
    SCHEMA_SCIENCE,
    SCHEMA_UNVERIFIED,
    SCIENCE_REQUIRED,
    SchemaValidationError,
    TERMINAL_CLASSES,
    build_non_authoritative_developer_payload,
    validate_authoritative_result_payload_v3,
    validate_authoritative_result_schema_v4_min,
    validate_preflight_execution_receipt,
    validate_science_payload,
    validate_unverified_payload,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_schema.py"
_HEX = "a" * 64
_LEAK = {
    "pass": True,
    "row_id_overlap": 0,
    "normalized_prompt_hash_overlap": 0,
    "normalized_target_hash_overlap": 0,
    "response_token_hash_overlap": 0,
}


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 250


def _markers(value: bool = True):
    return {n: value for n in REQUIRED_PHASE_MARKER_NAMES}


def _nll(mean: float = 1.0):
    row = {"numerator_f64": mean, "denominator": 1, "mean": mean}
    return {k: dict(row) for k in ("prod", "inv", "noop", "noop_repeat")}


def _proof():
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
        AGG_FLOOR, ESTIMAND_NAME, PER_KEY_FLOOR, SKEW_MAX,
    )
    per = {
        "k0": {"original_count": 2, "retained_count": 2, "dropped_count": 0, "retained_fraction": 1.0},
        "k1": {"original_count": 2, "retained_count": 2, "dropped_count": 0, "retained_fraction": 1.0},
    }
    return {
        "estimand": ESTIMAND_NAME,
        "legal_subset": {
            "estimand": ESTIMAND_NAME,
            "original_applied_total": 4, "retained_total": 4, "dropped_total": 0,
            "aggregate_retained_fraction": 1.0, "all_keys_nonempty": True, "replay_veto_total": 0,
            "boundary_q_acc_by_direction_counts": {"q0_acc0_d1": 4}, "per_key": per,
            "support_floors": {
                "pass": True, "per_key_min": PER_KEY_FLOOR, "aggregate_min": AGG_FLOOR,
                "skew_max": SKEW_MAX, "skew_observed": 1.0, "skew_defined": True,
            },
            "retained_stream_sha256": _HEX, "dropped_stream_sha256": _HEX,
            "applied_plan_index_direction_sha256": _HEX,
        },
        "observer_public_apply_calibration": {"ok": True},
        "current_weights_sha256_by_arm": {a: _HEX for a in ("prod", "inv", "noop", "noop_repeat")},
        "eval_row_ids_sha256": _HEX,
        "eval_batch_count": 2,
        "leakage_report_compact": dict(_LEAK),
        "mutation_parity": {
            "pass": True,
            "q_levels": {"pass": True, "per_key": {"k0": {}, "k1": {}}},
            "exact_accumulator_shadow": {"pass": True, "per_key": {"k0": {}, "k1": {}}},
            "frozen_scale": {"pass": True, "per_key": {
                "k0": {"pass": True, "shape": [], "dtype": "float32",
                       "base_sha256": _HEX, "prod_sha256": _HEX, "inv_sha256": _HEX},
                "k1": {"pass": True, "shape": [], "dtype": "float32",
                       "base_sha256": _HEX, "prod_sha256": _HEX, "inv_sha256": _HEX},
            }},
        },
        "terminal_precedence_path": ["integrity_clear", "asymmetry_clear", "science"],
    }


def _science(L: float = 1.0, **over):
    cls, eps = classify_signed_utility(L, L, L)
    payload = {
        "schema": SCHEMA_SCIENCE,
        "classifier": cls,
        "L_prod": L,
        "L_inv": L,
        "L_noop": L,
        "L_noop_repeat": L,
        "epsilon": eps,
        "parent_sha256_pre": _HEX,
        "parent_sha256_post": _HEX,
        "phase_markers": _markers(),
        "nll_per_arm": _nll(L),
        "apply_integer_vote_update_from_frozen_plan_calls": 4,
        "eligible_state_key_count": 2,
        **_proof(),
    }
    payload.update(over)
    return payload


def test_terminal_classes_and_developer_payload():
    assert "UNVERIFIED_ASYMMETRIC_INTERVENTION" in TERMINAL_CLASSES
    payload = build_non_authoritative_developer_payload({"classifier": "X"})
    assert payload["non_authoritative"] is True


def test_science_schema_requires_all_L_and_nll_fields():
    validate_science_payload(_science())
    with pytest.raises(SchemaValidationError, match="missing_fields"):
        validate_science_payload({"schema": SCHEMA_SCIENCE, "classifier": "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL"})
    with pytest.raises(SchemaValidationError, match="nll_per_arm_missing"):
        bad = _science()
        bad["nll_per_arm"] = {"prod": bad["nll_per_arm"]["prod"]}
        validate_science_payload(bad)
    with pytest.raises(SchemaValidationError, match="call_count_not_two_times"):
        validate_science_payload(_science(apply_integer_vote_update_from_frozen_plan_calls=3))


def test_science_rejects_missing_canonical_proof_fields():
    for key in (
        "estimand",
        "legal_subset",
        "observer_public_apply_calibration",
        "current_weights_sha256_by_arm",
        "eval_row_ids_sha256",
        "eval_batch_count",
        "leakage_report_compact",
        "mutation_parity",
        "terminal_precedence_path",
    ):
        bad = _science()
        del bad[key]
        with pytest.raises(SchemaValidationError, match="missing_fields"):
            validate_science_payload(bad)
    assert "legal_subset" in SCIENCE_REQUIRED


def test_science_rejects_empty_proofs_false_markers_and_bad_hex():
    with pytest.raises(SchemaValidationError):
        validate_science_payload(
            _science(
                observer_public_apply_calibration={},
                current_weights_sha256_by_arm={},
                leakage_report_compact={},
                mutation_parity={},
                terminal_precedence_path=[],
                eval_row_ids_sha256="z" * 64,
            )
        )
    with pytest.raises(SchemaValidationError, match="phase_marker_not_true"):
        validate_science_payload(_science(phase_markers=_markers(False)))
    with pytest.raises(SchemaValidationError, match="eval_row_ids_sha256_not_lowercase_hex64"):
        validate_science_payload(_science(eval_row_ids_sha256="Z" * 64))
    with pytest.raises(SchemaValidationError, match="eval_batch_count_not_2"):
        validate_science_payload(_science(eval_batch_count=1))


def test_science_proof_semantics_negative_cases():
    with pytest.raises(SchemaValidationError, match="calibration_pass_or_ok_not_true"):
        validate_science_payload(_science(observer_public_apply_calibration={"ok": False}))
    with pytest.raises(SchemaValidationError, match="current_weights_arm_keys_invalid"):
        validate_science_payload(_science(current_weights_sha256_by_arm={"prod": _HEX}))
    with pytest.raises(SchemaValidationError, match="leakage_pass_not_true"):
        validate_science_payload(_science(leakage_report_compact={**_LEAK, "pass": False}))
    with pytest.raises(SchemaValidationError, match="leakage_overlap_nonzero"):
        validate_science_payload(_science(leakage_report_compact={**_LEAK, "row_id_overlap": 1}))
    with pytest.raises(SchemaValidationError, match="mutation_parity_pass_not_true"):
        validate_science_payload(_science(mutation_parity={"pass": False}))
    with pytest.raises(SchemaValidationError, match="raw_index_arrays_forbidden"):
        scale = {
            "k0": {"pass": True, "shape": [], "dtype": "float32",
                   "base_sha256": _HEX, "prod_sha256": _HEX, "inv_sha256": _HEX},
            "k1": {"pass": True, "shape": [], "dtype": "float32",
                   "base_sha256": _HEX, "prod_sha256": _HEX, "inv_sha256": _HEX},
        }
        validate_science_payload(_science(mutation_parity={
            "pass": True,
            "q_levels": {"pass": True, "per_key": {"k0": {}, "k1": {}}, "changed_prod": [1]},
            "exact_accumulator_shadow": {"pass": True, "per_key": {"k0": {}, "k1": {}}},
            "frozen_scale": {"pass": True, "per_key": scale},
        }))
    with pytest.raises(SchemaValidationError, match="raw_index_arrays_forbidden"):
        bad_ls = _science()
        bad_ls["legal_subset"] = dict(bad_ls["legal_subset"])
        bad_ls["legal_subset"]["candidate_indices"] = [1, 2, 3]
        validate_science_payload(bad_ls)
    with pytest.raises(SchemaValidationError, match="support_floor_constants_mismatch"):
        forged = _science()
        forged["legal_subset"] = {
            "estimand": forged["estimand"], "support_floors": {"pass": True},
            "retained_stream_sha256": _HEX, "dropped_stream_sha256": _HEX,
            "applied_plan_index_direction_sha256": _HEX,
        }
        validate_science_payload(forged)
    with pytest.raises(SchemaValidationError, match="legal_subset_support_not_pass"):
        bad_floor = _science()
        bad_floor["legal_subset"] = dict(bad_floor["legal_subset"])
        bad_floor["legal_subset"]["support_floors"] = dict(bad_floor["legal_subset"]["support_floors"])
        bad_floor["legal_subset"]["support_floors"]["pass"] = False
        validate_science_payload(bad_floor)
    with pytest.raises(SchemaValidationError, match="per_key_global_totals_mismatch"):
        forged_global = _science()
        forged_global["legal_subset"] = dict(forged_global["legal_subset"])
        forged_global["legal_subset"]["original_applied_total"] = 100
        forged_global["legal_subset"]["retained_total"] = 100
        forged_global["legal_subset"]["dropped_total"] = 0
        forged_global["legal_subset"]["aggregate_retained_fraction"] = 1.0
        forged_global["legal_subset"]["boundary_q_acc_by_direction_counts"] = {"q0_acc0_d1": 100}
        validate_science_payload(forged_global)
    with pytest.raises(SchemaValidationError, match="legal_subset_parity_keyset_mismatch"):
        empty_scale = _science()
        empty_scale["mutation_parity"] = dict(empty_scale["mutation_parity"])
        empty_scale["mutation_parity"]["frozen_scale"] = {"pass": True, "per_key": {}}
        validate_science_payload(empty_scale)
    with pytest.raises(SchemaValidationError, match="frozen_scale_row_pass|frozen_scale_row_meta"):
        empty_rows = _science()
        empty_rows["mutation_parity"] = dict(empty_rows["mutation_parity"])
        empty_rows["mutation_parity"]["frozen_scale"] = {
            "pass": True, "per_key": {"k0": {}, "k1": {}},
        }
        validate_science_payload(empty_rows)
    with pytest.raises(SchemaValidationError, match="boundary_count_invalid"):
        bad_b = _science()
        bad_b["legal_subset"] = dict(bad_b["legal_subset"])
        bad_b["legal_subset"]["boundary_q_acc_by_direction_counts"] = {"q0_acc0_d1": True}
        validate_science_payload(bad_b)
    with pytest.raises(SchemaValidationError, match="boundary_count_invalid"):
        bad_bf = _science()
        bad_bf["legal_subset"] = dict(bad_bf["legal_subset"])
        bad_bf["legal_subset"]["boundary_q_acc_by_direction_counts"] = {"q0_acc0_d1": 4.0}
        validate_science_payload(bad_bf)
    with pytest.raises(SchemaValidationError, match="nonfinite:aggregate_retained_fraction"):
        nan_agg = _science()
        nan_agg["legal_subset"] = dict(nan_agg["legal_subset"])
        nan_agg["legal_subset"]["aggregate_retained_fraction"] = float("nan")
        validate_science_payload(nan_agg)
    with pytest.raises(SchemaValidationError, match="skew_not_defined"):
        undef = _science()
        undef["legal_subset"] = dict(undef["legal_subset"])
        undef["legal_subset"]["support_floors"] = dict(undef["legal_subset"]["support_floors"])
        undef["legal_subset"]["support_floors"]["skew_defined"] = False
        undef["legal_subset"]["support_floors"]["skew_observed"] = None
        validate_science_payload(undef)
    with pytest.raises(SchemaValidationError, match="frozen_scale_hash_unequal|frozen_scale_row_meta"):
        bogus = _science()
        bogus["mutation_parity"] = dict(bogus["mutation_parity"])
        bogus["mutation_parity"]["frozen_scale"] = {
            "pass": True,
            "per_key": {
                "k0": {"pass": True, "shape": ["bogus"], "dtype": "bogus",
                       "base_sha256": _HEX, "prod_sha256": "b" * 64, "inv_sha256": "c" * 64},
                "k1": {"pass": True, "shape": [], "dtype": "float32",
                       "base_sha256": _HEX, "prod_sha256": _HEX, "inv_sha256": _HEX},
            },
        }
        validate_science_payload(bogus)
    with pytest.raises(SchemaValidationError, match="estimand_mismatch"):
        validate_science_payload(_science(estimand="full_production_signed_utility"))
    with pytest.raises(SchemaValidationError, match="terminal_precedence_path_invalid"):
        validate_science_payload(_science(terminal_precedence_path=["science"]))


def test_science_epsilon_and_classifier_bound_to_losses():
    present = _science(L_prod=0.0, L_inv=1.0, L_noop=1.0, L_noop_repeat=1.0)
    present["nll_per_arm"] = {
        "prod": {"numerator_f64": 0.0, "denominator": 1, "mean": 0.0},
        "inv": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
        "noop": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
        "noop_repeat": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
    }
    present["epsilon"] = epsilon_from_noop(1.0)
    present["classifier"] = "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"
    validate_science_payload(present)
    with pytest.raises(SchemaValidationError, match="epsilon_mismatch"):
        validate_science_payload(_science(epsilon=9e-7))
    with pytest.raises(SchemaValidationError, match="classifier_mismatch"):
        validate_science_payload(_science(classifier="SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"))


def test_science_nll_L_parent_and_proof_type_checks():
    with pytest.raises(SchemaValidationError, match="parent_pre_post_mismatch"):
        validate_science_payload(_science(parent_sha256_post="c" * 64))
    with pytest.raises(SchemaValidationError, match="L_arm_nll_inconsistent"):
        validate_science_payload(_science(L_prod=2.0))
    with pytest.raises(SchemaValidationError, match="nll_mean_mismatch"):
        bad = _science()
        bad["nll_per_arm"]["prod"]["mean"] = 9.0
        bad["L_prod"] = 9.0
        validate_science_payload(bad)


def test_science_rejects_nonfinite_and_noop_repeat_epsilon_boundary():
    nan, inf = float("nan"), float("inf")
    for bad_L in (nan, inf, -inf):
        bad = _science(L_prod=bad_L)
        bad["nll_per_arm"]["prod"] = {"numerator_f64": bad_L, "denominator": 1, "mean": bad_L}
        with pytest.raises(SchemaValidationError, match="nonfinite:"):
            validate_science_payload(bad)
    bad_eps = _science(epsilon=nan)
    with pytest.raises(SchemaValidationError, match="nonfinite:epsilon"):
        validate_science_payload(bad_eps)
    # Claude BLOCK repro: large noop-repeat drift must reject science.
    drift = _science(L_noop=1.0, L_noop_repeat=100.0)
    drift["nll_per_arm"]["noop"] = {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0}
    drift["nll_per_arm"]["noop_repeat"] = {"numerator_f64": 100.0, "denominator": 1, "mean": 100.0}
    drift["epsilon"] = epsilon_from_noop(1.0)
    with pytest.raises(SchemaValidationError, match="noop_repeat_drift_crosses_epsilon"):
        validate_science_payload(drift)
    eps = epsilon_from_noop(1.0)
    # Pass: drift strictly below epsilon; Fail: drift at/above epsilon.
    ok = _science(L_noop=1.0, L_noop_repeat=1.0 + 0.5 * eps)
    ok["nll_per_arm"]["noop"] = {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0}
    ok["nll_per_arm"]["noop_repeat"] = {
        "numerator_f64": 1.0 + 0.5 * eps, "denominator": 1, "mean": 1.0 + 0.5 * eps
    }
    ok["epsilon"] = eps
    validate_science_payload(ok)
    at = _science(L_noop=1.0, L_noop_repeat=1.0 + eps)
    at["nll_per_arm"]["noop"] = {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0}
    at["nll_per_arm"]["noop_repeat"] = {
        "numerator_f64": 1.0 + eps, "denominator": 1, "mean": 1.0 + eps
    }
    at["epsilon"] = eps
    with pytest.raises(SchemaValidationError, match="noop_repeat_drift_crosses_epsilon"):
        validate_science_payload(at)


def test_science_rejects_nonexact_counts_parent_hex_and_zero_eligible():
    # Claude c3 BLOCK: fractional denom with consistent mean must reject.
    frac = _science()
    for arm in frac["nll_per_arm"]:
        frac["nll_per_arm"][arm] = {"numerator_f64": 1.5, "denominator": 1.5, "mean": 1.0}
    with pytest.raises(SchemaValidationError, match="not_exact_int:prod.denominator"):
        validate_science_payload(frac)
    with pytest.raises(SchemaValidationError, match="parent_sha256_pre_not_lowercase_hex64"):
        validate_science_payload(_science(parent_sha256_pre="x", parent_sha256_post="x"))
    with pytest.raises(SchemaValidationError, match="eligible_state_key_count_not_positive"):
        validate_science_payload(
            _science(eligible_state_key_count=0, apply_integer_vote_update_from_frozen_plan_calls=0)
        )
    with pytest.raises(SchemaValidationError, match="not_exact_int:apply_integer_vote_update"):
        validate_science_payload(
            _science(apply_integer_vote_update_from_frozen_plan_calls=4.0, eligible_state_key_count=2)
        )
    with pytest.raises(SchemaValidationError, match="not_exact_int:eligible_state_key_count"):
        validate_science_payload(
            _science(eligible_state_key_count="2", apply_integer_vote_update_from_frozen_plan_calls=4)
        )
    with pytest.raises(SchemaValidationError, match="not_exact_int:eligible_state_key_count"):
        validate_science_payload(
            _science(eligible_state_key_count=True, apply_integer_vote_update_from_frozen_plan_calls=2)
        )
    with pytest.raises(SchemaValidationError, match="not_exact_int:eval_batch_count"):
        validate_science_payload(_science(eval_batch_count=2.0))
    with pytest.raises(SchemaValidationError, match="not_exact_int:leakage.row_id_overlap"):
        validate_science_payload(_science(leakage_report_compact={**_LEAK, "row_id_overlap": 0.0}))



def test_missing_nll_uses_unverified_schema_not_science_schema():
    payload = {
        "schema": SCHEMA_UNVERIFIED,
        "classifier": "UNVERIFIED_INTEGRITY_OR_EXECUTION",
        "reason": "missing_nll",
        "failed_stage": "EVAL",
        "phase_markers": _markers(False),
        "parent_sha256_pre": "b" * 64,
        "compact_diagnostics": {"note": "no nll"},
    }
    validate_unverified_payload(payload)
    validate_authoritative_result_payload_v3(payload)
    with pytest.raises(SchemaValidationError):
        validate_science_payload(payload)
    with pytest.raises(SchemaValidationError, match="phase_marker_missing_or_not_bool"):
        bad = dict(payload)
        bad["phase_markers"] = {n: True for n in REQUIRED_PHASE_MARKER_NAMES[:-1]}
        validate_unverified_payload(bad)


def test_preflight_pin_failure_forbids_parent_sha_and_routes():
    payload = {
        "schema": SCHEMA_PREFLIGHT,
        "classifier": "UNVERIFIED_INTEGRITY_OR_EXECUTION",
        "failed_stage": "source_pins",
        "observed": {"head": "x"},
        "expected": {"head": "y"},
        "ts_utc": "2026-07-17T00:00:00Z",
    }
    validate_preflight_execution_receipt(payload)
    with pytest.raises(SchemaValidationError, match="preflight_parent_sha_must_be_null"):
        validate_preflight_execution_receipt({**payload, "parent_sha256_pre": "c" * 64})


def test_legacy_v4_min_still_accepts_augmented_payload():
    markers = {n: True for n in REQUIRED_PHASE_MARKER_NAMES}
    validate_authoritative_result_schema_v4_min(
        {
            "schema": "v4_min",
            "classifier": "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL",
            "L_prod": 1.0,
            "L_inv": 1.0,
            "L_noop": 1.0,
            "epsilon": 1e-7,
            "parent_sha256_pre": "a" * 64,
            "parent_sha256_post": "a" * 64,
            "phase_markers": markers,
            "nll_per_arm": {
                "prod": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
                "inv": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
                "noop": {"numerator_f64": 1.0, "denominator": 1, "mean": 1.0},
            },
            "apply_integer_vote_update_from_frozen_plan_calls": 4,
            "eligible_state_key_count": 2,
        }
    )
