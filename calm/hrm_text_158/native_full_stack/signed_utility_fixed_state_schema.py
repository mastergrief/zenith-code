"""Result/telemetry schema for fixed-state signed-utility (PLAN v6 D1 + D2c7)."""
from __future__ import annotations
import json, math, re
from typing import Any, Mapping
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    AGG_FLOOR, ESTIMAND_NAME, MAX_AUTHORITATIVE_RESULT_BYTES, PER_KEY_FLOOR, SKEW_MAX,
    payload_has_raw_index_arrays)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    classify_signed_utility, epsilon_from_noop)
TERMINAL_CLASSES = (
    "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN", "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL",
    "UNVERIFIED_ASYMMETRIC_INTERVENTION", "UNVERIFIED_INTEGRITY_OR_EXECUTION")
REQUIRED_PHASE_MARKER_NAMES = (
    "PHASE_MATERIALIZE_BEGIN", "PHASE_MATERIALIZE_END", "PHASE_CAPTURE_BACKWARD_VOTE_BEGIN",
    "PHASE_CAPTURE_BACKWARD_VOTE_END", "PHASE_THREE_ARM_APPLY_WRITEBACK_BEGIN",
    "PHASE_THREE_ARM_APPLY_WRITEBACK_END", "PHASE_THREE_ARM_EVAL_NLL_BEGIN", "PHASE_THREE_ARM_EVAL_NLL_END",
    "PHASE_EMIT_FLUSH_BEGIN", "PHASE_EMIT_FLUSH_END")
SCHEMA_PREFLIGHT = "post_seam_signed_utility_preflight_execution_receipt_v4"
SCHEMA_UNVERIFIED = "post_seam_signed_utility_authoritative_result_unverified_v4"
SCHEMA_SCIENCE = "post_seam_signed_utility_authoritative_result_science_v4"
SCIENCE_REQUIRED = (
    "schema", "classifier", "estimand", "legal_subset", "L_prod", "L_inv", "L_noop", "L_noop_repeat", "epsilon",
    "nll_per_arm", "parent_sha256_pre", "parent_sha256_post", "phase_markers",
    "apply_integer_vote_update_from_frozen_plan_calls", "eligible_state_key_count",
    "observer_public_apply_calibration", "current_weights_sha256_by_arm", "eval_row_ids_sha256",
    "eval_batch_count", "leakage_report_compact", "mutation_parity", "terminal_precedence_path")
_ARMS = (("prod", "L_prod"), ("inv", "L_inv"), ("noop", "L_noop"), ("noop_repeat", "L_noop_repeat"))
_W, _HEX = ("prod", "inv", "noop", "noop_repeat"), re.compile(r"^[0-9a-f]{64}$")
_PATH = ("integrity_clear", "asymmetry_clear", "science")
_LEAK = ("row_id_overlap", "normalized_prompt_hash_overlap", "normalized_target_hash_overlap", "response_token_hash_overlap")
class SchemaValidationError(ValueError): pass
def required_phase_marker_names(): return REQUIRED_PHASE_MARKER_NAMES
def build_non_authoritative_developer_payload(diag: Mapping[str, Any]) -> dict[str, Any]:
    return {"mode": "developer_check", "non_authoritative": True, "schema": "post_seam_signed_utility_developer_payload_v0", "diag": dict(diag)}
def _req(payload, keys):
    missing = [k for k in keys if k not in payload]
    if missing: raise SchemaValidationError(f"missing_fields:{missing}")
def _finite(x, name):
    if not math.isfinite(float(x)): raise SchemaValidationError(f"nonfinite:{name}")
def _exact_int(x, name):
    if type(x) is not int: raise SchemaValidationError(f"not_exact_int:{name}")
    return x
def _hex64(x, name):
    if not isinstance(x, str) or _HEX.fullmatch(x) is None:
        raise SchemaValidationError(f"{name}_not_lowercase_hex64")
def _phase_markers(payload, *, require_true=False):
    m = payload["phase_markers"]
    if not isinstance(m, Mapping): raise SchemaValidationError("phase_markers_not_mapping")
    for name in REQUIRED_PHASE_MARKER_NAMES:
        if name not in m or not isinstance(m[name], bool):
            raise SchemaValidationError(f"phase_marker_missing_or_not_bool:{name}")
        if require_true and m[name] is not True: raise SchemaValidationError(f"phase_marker_not_true:{name}")
def _nll_and_L(payload):
    nll = payload["nll_per_arm"]
    for arm, lkey in _ARMS:
        if arm not in nll: raise SchemaValidationError(f"nll_per_arm_missing:{arm}")
        row = nll[arm]
        for field in ("numerator_f64", "denominator", "mean"):
            if field not in row: raise SchemaValidationError(f"nll_arm_field_missing:{arm}.{field}")
            _finite(row[field], f"{arm}.{field}")
        _finite(payload[lkey], lkey)
        den = _exact_int(row["denominator"], f"{arm}.denominator")
        if den <= 0: raise SchemaValidationError(f"nll_denominator_nonpositive:{arm}")
        mean, num = float(row["mean"]), float(row["numerator_f64"])
        if abs(mean - (num / den)) > 1e-12: raise SchemaValidationError(f"nll_mean_mismatch:{arm}")
        if abs(float(payload[lkey]) - mean) > 1e-12: raise SchemaValidationError(f"L_arm_nll_inconsistent:{arm}")
def _science_proofs(payload):
    if payload.get("estimand") != ESTIMAND_NAME:
        raise SchemaValidationError("estimand_mismatch")
    ls = payload["legal_subset"]
    if not isinstance(ls, Mapping) or ls.get("estimand") != ESTIMAND_NAME:
        raise SchemaValidationError("legal_subset_estimand_invalid")
    floors = ls.get("support_floors")
    if not isinstance(floors, Mapping) or floors.get("pass") is not True:
        raise SchemaValidationError("legal_subset_support_not_pass")
    if floors.get("per_key_min") != PER_KEY_FLOOR or floors.get("aggregate_min") != AGG_FLOOR or floors.get("skew_max") != SKEW_MAX:
        raise SchemaValidationError("support_floor_constants_mismatch")
    if floors.get("skew_defined") is not True:
        raise SchemaValidationError("skew_not_defined")
    _finite(floors.get("skew_observed"), "skew_observed")
    orig = _exact_int(ls.get("original_applied_total", -1), "original_applied_total")
    ret = _exact_int(ls.get("retained_total", -1), "retained_total")
    drop = _exact_int(ls.get("dropped_total", -1), "dropped_total")
    if orig <= 0 or ret < 0 or drop < 0 or ret + drop != orig:
        raise SchemaValidationError("support_totals_inconsistent")
    _finite(ls.get("aggregate_retained_fraction"), "aggregate_retained_fraction")
    agg = float(ls["aggregate_retained_fraction"])
    if abs(agg - (ret / float(orig))) > 1e-12:
        raise SchemaValidationError("aggregate_retained_fraction_mismatch")
    if ls.get("all_keys_nonempty") is not True:
        raise SchemaValidationError("all_keys_nonempty_not_true")
    if _exact_int(ls.get("replay_veto_total", -1), "replay_veto_total") != 0:
        raise SchemaValidationError("replay_veto_total_nonzero")
    bcounts = ls.get("boundary_q_acc_by_direction_counts")
    if not isinstance(bcounts, Mapping):
        raise SchemaValidationError("boundary_classes_missing")
    per = ls.get("per_key")
    if not isinstance(per, Mapping) or not per:
        raise SchemaValidationError("legal_subset_per_key_missing")
    for hx in ("retained_stream_sha256", "dropped_stream_sha256", "applied_plan_index_direction_sha256"):
        _hex64(ls.get(hx), hx)
    mp = payload["mutation_parity"]
    if not isinstance(mp, Mapping) or mp.get("pass") is not True:
        raise SchemaValidationError("mutation_parity_pass_not_true")
    for carrier in ("q_levels", "exact_accumulator_shadow", "frozen_scale"):
        sub = mp.get(carrier)
        if not isinstance(sub, Mapping) or sub.get("pass") is not True:
            raise SchemaValidationError(f"mutation_parity_carrier_fail:{carrier}")
    q_keys = set((mp.get("q_levels") or {}).get("per_key") or {})
    a_keys = set((mp.get("exact_accumulator_shadow") or {}).get("per_key") or {})
    s_rows = (mp.get("frozen_scale") or {}).get("per_key") or {}
    if not isinstance(s_rows, Mapping) or set(per) != q_keys or set(per) != a_keys or set(per) != set(s_rows):
        raise SchemaValidationError("legal_subset_parity_keyset_mismatch")
    for key, row in s_rows.items():
        if not isinstance(row, Mapping) or row.get("pass") is not True:
            raise SchemaValidationError(f"frozen_scale_row_pass:{key}")
        shape, dtype = row.get("shape"), row.get("dtype")
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape) or dtype != "float32":
            raise SchemaValidationError(f"frozen_scale_row_meta:{key}")
        for hx in ("base_sha256", "prod_sha256", "inv_sha256"):
            _hex64(row.get(hx), f"frozen_scale.{key}.{hx}")
        if not (row["base_sha256"] == row["prod_sha256"] == row["inv_sha256"]):
            raise SchemaValidationError(f"frozen_scale_hash_unequal:{key}")
    if len(per) != _exact_int(payload["eligible_state_key_count"], "eligible_state_key_count"):
        raise SchemaValidationError("legal_subset_key_count_mismatch")
    fracs, sum_o, sum_r, sum_d, bsum = [], 0, 0, 0, 0
    for key, row in per.items():
        o = _exact_int(row.get("original_count", -1), f"{key}.original_count")
        r = _exact_int(row.get("retained_count", -1), f"{key}.retained_count")
        d = _exact_int(row.get("dropped_count", -1), f"{key}.dropped_count")
        if o <= 0 or r < 1 or d < 0 or r + d != o:
            raise SchemaValidationError(f"per_key_counts_inconsistent:{key}")
        rf = float(row.get("retained_fraction", -1))
        if abs(rf - (r / float(o))) > 1e-12 or rf + 1e-15 < PER_KEY_FLOOR:
            raise SchemaValidationError(f"per_key_fraction_invalid:{key}")
        fracs.append(rf); sum_o += o; sum_r += r; sum_d += d
    if sum_o != orig or sum_r != ret or sum_d != drop:
        raise SchemaValidationError("per_key_global_totals_mismatch")
    for bk, bv in bcounts.items():
        if type(bv) is not int or bv < 0:
            raise SchemaValidationError(f"boundary_count_invalid:{bk}")
        bsum += bv
    if bsum != orig:
        raise SchemaValidationError("boundary_count_sum_mismatch")
    skew = max(fracs) / min(fracs) if fracs and min(fracs) > 0 else None
    if skew is None or abs(float(floors["skew_observed"]) - skew) > 1e-12 or skew > SKEW_MAX + 1e-15:
        raise SchemaValidationError("skew_recompute_mismatch")
    if agg + 1e-15 < AGG_FLOOR:
        raise SchemaValidationError("aggregate_floor_failed")
    cal = payload["observer_public_apply_calibration"]
    if not isinstance(cal, Mapping) or not (cal.get("pass") is True or cal.get("ok") is True):
        raise SchemaValidationError("calibration_pass_or_ok_not_true")
    w = payload["current_weights_sha256_by_arm"]
    if not isinstance(w, Mapping) or set(w) != set(_W):
        raise SchemaValidationError("current_weights_arm_keys_invalid")
    for arm in _W: _hex64(w[arm], f"current_weights_{arm}")
    leak = payload["leakage_report_compact"]
    if not isinstance(leak, Mapping) or leak.get("pass") is not True:
        raise SchemaValidationError("leakage_pass_not_true")
    for key in _LEAK:
        if _exact_int(leak.get(key, -1), f"leakage.{key}") != 0:
            raise SchemaValidationError(f"leakage_overlap_nonzero:{key}")
    if payload_has_raw_index_arrays(mp) or payload_has_raw_index_arrays(ls):
        raise SchemaValidationError("raw_index_arrays_forbidden")
    if tuple(payload["terminal_precedence_path"]) != _PATH:
        raise SchemaValidationError("terminal_precedence_path_invalid")
    _hex64(payload["eval_row_ids_sha256"], "eval_row_ids_sha256")
    if _exact_int(payload["eval_batch_count"], "eval_batch_count") != 2:
        raise SchemaValidationError("eval_batch_count_not_2")
def validate_preflight_execution_receipt(payload: Mapping[str, Any]) -> None:
    _req(payload, ("schema", "classifier", "failed_stage", "observed", "expected", "ts_utc"))
    if payload["schema"] != SCHEMA_PREFLIGHT: raise SchemaValidationError("schema_id_mismatch_preflight")
    if payload["classifier"] != "UNVERIFIED_INTEGRITY_OR_EXECUTION":
        raise SchemaValidationError("preflight_classifier_must_be_integrity")
    if "parent_sha256_pre" in payload and payload["parent_sha256_pre"] is not None:
        raise SchemaValidationError("preflight_parent_sha_must_be_null")
def validate_unverified_payload(payload: Mapping[str, Any]) -> None:
    _req(payload, ("schema", "classifier", "reason", "failed_stage", "phase_markers",
                   "parent_sha256_pre", "compact_diagnostics"))
    if payload["schema"] != SCHEMA_UNVERIFIED: raise SchemaValidationError("schema_id_mismatch_unverified")
    if payload["classifier"] not in TERMINAL_CLASSES[2:]:
        raise SchemaValidationError("unverified_classifier_invalid")
    _phase_markers(payload)
    if payload_has_raw_index_arrays(payload.get("compact_diagnostics")):
        raise SchemaValidationError("raw_index_arrays_forbidden")
def validate_science_payload(payload: Mapping[str, Any]) -> None:
    _req(payload, SCIENCE_REQUIRED)
    if payload["schema"] != SCHEMA_SCIENCE: raise SchemaValidationError("schema_id_mismatch_science")
    if payload["classifier"] not in TERMINAL_CLASSES[:2]:
        raise SchemaValidationError("science_classifier_invalid")
    _hex64(payload["parent_sha256_pre"], "parent_sha256_pre"); _hex64(payload["parent_sha256_post"], "parent_sha256_post")
    if payload["parent_sha256_pre"] != payload["parent_sha256_post"]:
        raise SchemaValidationError("parent_pre_post_mismatch")
    _phase_markers(payload, require_true=True); _nll_and_L(payload)
    n = _exact_int(payload["eligible_state_key_count"], "eligible_state_key_count")
    calls = _exact_int(payload["apply_integer_vote_update_from_frozen_plan_calls"], "apply_integer_vote_update_from_frozen_plan_calls")
    if n <= 0: raise SchemaValidationError("eligible_state_key_count_not_positive")
    if calls != 2 * n: raise SchemaValidationError("call_count_not_two_times_eligible_keys")
    _science_proofs(payload)
    nbytes = len(json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode())
    if nbytes > MAX_AUTHORITATIVE_RESULT_BYTES:
        raise SchemaValidationError(f"authoritative_result_overflow:{nbytes}")
    L0, Lr, eps = float(payload["L_noop"]), float(payload["L_noop_repeat"]), float(payload["epsilon"]); _finite(eps, "epsilon")
    if abs(eps - epsilon_from_noop(L0)) > 1e-18: raise SchemaValidationError("epsilon_mismatch")
    if abs(L0 - Lr) >= eps: raise SchemaValidationError("noop_repeat_drift_crosses_epsilon")
    exp, _ = classify_signed_utility(float(payload["L_prod"]), float(payload["L_inv"]), L0)
    if payload["classifier"] != exp: raise SchemaValidationError("classifier_mismatch")
def validate_authoritative_result_payload_v3(payload: Mapping[str, Any]) -> None:
    s = payload.get("schema")
    if s == SCHEMA_PREFLIGHT: validate_preflight_execution_receipt(payload)
    elif s == SCHEMA_UNVERIFIED: validate_unverified_payload(payload)
    elif s == SCHEMA_SCIENCE: validate_science_payload(payload)
    else: raise SchemaValidationError(f"unknown_schema:{s}")
def validate_authoritative_result_schema_v4_min(payload: Mapping[str, Any]) -> None:
    p, hx = dict(payload), "a" * 64
    if p.get("schema") in (None, "v4_min"): p["schema"] = SCHEMA_SCIENCE
    p.setdefault("L_noop_repeat", p.get("L_noop"))
    nll = dict(p.get("nll_per_arm") or {})
    if "noop_repeat" not in nll and "noop" in nll:
        nll["noop_repeat"] = dict(nll["noop"]); p["nll_per_arm"] = nll
    p.setdefault("estimand", ESTIMAND_NAME)
    n_keys = int(p.get("eligible_state_key_count") or 1)
    if n_keys < 1: n_keys = 1
    per = {f"k{i}": {"original_count": 4, "retained_count": 4, "dropped_count": 0, "retained_fraction": 1.0}
           for i in range(n_keys)}
    tot = 4 * n_keys
    p.setdefault("legal_subset", {
        "estimand": ESTIMAND_NAME, "original_applied_total": tot, "retained_total": tot, "dropped_total": 0,
        "aggregate_retained_fraction": 1.0, "all_keys_nonempty": True, "replay_veto_total": 0,
        "boundary_q_acc_by_direction_counts": {"q0_acc0_d1": tot}, "per_key": per,
        "support_floors": {"pass": True, "per_key_min": PER_KEY_FLOOR, "aggregate_min": AGG_FLOOR,
                          "skew_max": SKEW_MAX, "skew_observed": 1.0, "skew_defined": True},
        "retained_stream_sha256": hx, "dropped_stream_sha256": hx, "applied_plan_index_direction_sha256": hx,
    })
    mp_keys = {k: {} for k in per}
    scale_rows = {k: {"pass": True, "shape": [], "dtype": "float32",
                      "base_sha256": hx, "prod_sha256": hx, "inv_sha256": hx} for k in per}
    defs = {"observer_public_apply_calibration": {"ok": True}, "current_weights_sha256_by_arm": {a: hx for a in _W},
            "eval_row_ids_sha256": hx, "eval_batch_count": 2,
            "leakage_report_compact": {"pass": True, **{k: 0 for k in _LEAK}},
            "mutation_parity": {"pass": True, "q_levels": {"pass": True, "per_key": dict(mp_keys)},
                                "exact_accumulator_shadow": {"pass": True, "per_key": dict(mp_keys)},
                                "frozen_scale": {"pass": True, "per_key": scale_rows}},
            "terminal_precedence_path": list(_PATH)}
    p.setdefault("eligible_state_key_count", n_keys)
    p.setdefault("apply_integer_vote_update_from_frozen_plan_calls", 2 * n_keys)
    for k, v in defs.items(): p.setdefault(k, v)
    p.setdefault("epsilon", epsilon_from_noop(float(p["L_noop"]))); validate_science_payload(p)
