"""Fail-closed readiness + DEEP trajectory schema validation (PLAN_v6 rev4).

Owns: evaluate_readiness, validate_trajectory_schemas (nested p10/p50/p90/n,
finite numerics, probe step0/final fields). Dependency: readiness → telemetry
constants only. Never imports CLI/loop/benchmark.
Bound by PLAN_v6 sha 346b67d8…; rev4 re-scope 1784829182373.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    PARENT_SHA256,
    PLAN_SHA256,
    expected_trajectory_boundaries,
    optional_json_float,
)

MARGIN_ROW_KEYS = (
    "step",
    "residual_margin_pre_cap_crossers",
    "residual_margin_applied_topk",
)
EPISODE_ROW_KEYS = (
    "step",
    "active_episode_count",
    "episode_age_quantiles_p10_p50_p90",
)
DEMAND_ROW_KEYS = (
    "step",
    "candidate_crossers_before_cap",
    "applied_count",
    "demand_applied_ratio",
)
QUANTILE_KEYS = ("p10", "p50", "p90", "n")


def _finite_or_none(x: Any) -> bool:
    if x is None:
        return True
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


def _finite_number(x: Any) -> bool:
    """Numeric finite REQUIRED — None/non-numeric/NaN/inf all fail (rev5)."""
    if x is None or isinstance(x, bool):
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


def _validate_quantile_block(name: str, block: Any) -> str | None:
    if not isinstance(block, Mapping):
        return f"readiness_trajectory_{name}_quantile_not_mapping"
    for k in QUANTILE_KEYS:
        if k not in block:
            return f"readiness_trajectory_{name}_quantile_missing:{k}"
    if not isinstance(block["n"], int) or int(block["n"]) < 0:
        return f"readiness_trajectory_{name}_quantile_n_invalid"
    for k in ("p10", "p50", "p90"):
        if not _finite_or_none(block[k]):
            return f"readiness_trajectory_{name}_quantile_nonfinite:{k}"
        # if n==0 all quantiles must be None; if n>0 quantiles must be numeric
        if int(block["n"]) == 0 and block[k] is not None:
            return f"readiness_trajectory_{name}_quantile_nonzero_empty:{k}"
        if int(block["n"]) > 0 and block[k] is None:
            return f"readiness_trajectory_{name}_quantile_missing_value:{k}"
    return None


def validate_trajectory_schemas(
    *,
    steps: int,
    margin_trajectory: list,
    episode_trajectory: list,
    demand_per_25: list,
    H_trajectory: list,
) -> dict[str, Any]:
    """Fail-closed: exact boundaries + DEEP nested population/quantile validation."""
    expected = expected_trajectory_boundaries(steps)
    checks: dict[str, Any] = {"expected_boundaries": expected}

    def _check_margin(rows: list) -> str | None:
        if not isinstance(rows, list) or not rows:
            return "readiness_trajectory_margin_empty"
        got = []
        for row in rows:
            if not isinstance(row, Mapping):
                return "readiness_trajectory_margin_row_not_mapping"
            for k in MARGIN_ROW_KEYS:
                if k not in row:
                    return f"readiness_trajectory_margin_missing_key:{k}"
            for pop in (
                "residual_margin_pre_cap_crossers",
                "residual_margin_applied_topk",
            ):
                err = _validate_quantile_block(f"margin.{pop}", row[pop])
                if err:
                    return err
            # rev5: bind quantile population n to the row's own counts when present
            for pop, cnt_key in (
                ("residual_margin_pre_cap_crossers", "n_candidates"),
                ("residual_margin_applied_topk", "n_applied"),
            ):
                if cnt_key in row:
                    cnt = row[cnt_key]
                    if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 0:
                        return f"readiness_trajectory_margin_count_invalid:{cnt_key}"
                    if int(row[pop]["n"]) != cnt:
                        return f"readiness_trajectory_margin_count_population_mismatch:{cnt_key}"
            got.append(int(row["step"]))
        if got != expected:
            return "readiness_trajectory_margin_boundaries_mismatch"
        return None

    def _check_episode(rows: list) -> str | None:
        if not isinstance(rows, list) or not rows:
            return "readiness_trajectory_episode_empty"
        got = []
        for row in rows:
            if not isinstance(row, Mapping):
                return "readiness_trajectory_episode_row_not_mapping"
            for k in EPISODE_ROW_KEYS:
                if k not in row:
                    return f"readiness_trajectory_episode_missing_key:{k}"
            cnt = row["active_episode_count"]
            if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 0:
                return "readiness_trajectory_episode_count_invalid"
            err = _validate_quantile_block(
                "episode.age", row["episode_age_quantiles_p10_p50_p90"]
            )
            if err:
                return err
            # rev5: active count must equal its quantile population n
            if int(row["episode_age_quantiles_p10_p50_p90"]["n"]) != cnt:
                return "readiness_trajectory_episode_count_population_mismatch"
            got.append(int(row["step"]))
        if got != expected:
            return "readiness_trajectory_episode_boundaries_mismatch"
        return None

    def _check_demand(rows: list) -> str | None:
        if not isinstance(rows, list) or not rows:
            return "readiness_trajectory_demand_empty"
        got = []
        for row in rows:
            if not isinstance(row, Mapping):
                return "readiness_trajectory_demand_row_not_mapping"
            for k in DEMAND_ROW_KEYS:
                if k not in row:
                    return f"readiness_trajectory_demand_missing_key:{k}"
            for k in ("candidate_crossers_before_cap", "applied_count"):
                if not isinstance(row[k], int) or int(row[k]) < 0:
                    return f"readiness_trajectory_demand_invalid:{k}"
            # rev5: ratio is produced as n_candidates/max(1,n_applied) — always
            # a finite float; null/non-numeric = instrumentation failure.
            if not _finite_number(row["demand_applied_ratio"]) or float(row["demand_applied_ratio"]) < 0:
                return "readiness_trajectory_demand_ratio_nonfinite"
            got.append(int(row["step"]))
        if got != expected:
            return "readiness_trajectory_demand_boundaries_mismatch"
        return None

    for name, fn, rows in (
        ("margin", _check_margin, margin_trajectory),
        ("episode", _check_episode, episode_trajectory),
        ("demand", _check_demand, demand_per_25),
    ):
        err = fn(rows)
        checks[f"{name}_ok"] = err is None
        if err:
            return {"ok": False, "stop_reason": err, "checks": checks}

    if not isinstance(H_trajectory, list) or not H_trajectory:
        return {"ok": False, "stop_reason": "readiness_trajectory_H_empty", "checks": checks}
    h_steps = []
    for row in H_trajectory:
        if not isinstance(row, Mapping) or "step" not in row or "H_bits_per_weight" not in row:
            return {"ok": False, "stop_reason": "readiness_trajectory_H_malformed", "checks": checks}
        # rev5: H is produced as float(entropy_bits(...)) — always numeric;
        # null = instrumentation failure, not an acceptable science row.
        if not _finite_number(row["H_bits_per_weight"]):
            return {"ok": False, "stop_reason": "readiness_trajectory_H_nonfinite", "checks": checks}
        h_steps.append(int(row["step"]))
    if h_steps != expected:
        return {"ok": False, "stop_reason": "readiness_trajectory_H_boundaries_mismatch", "checks": checks}
    checks["H_ok"] = True
    return {"ok": True, "stop_reason": None, "checks": checks}


def evaluate_readiness(
    *,
    expected_parent_sha: str | None,
    banked_sha: Mapping[str, Any] | None,
    frozen_scale_sha: Mapping[str, Any] | None,
    route_counters: Mapping[str, Any] | None,
    paired_proof: Mapping[str, Any] | None,
    paired_determinism_cost_ok: bool | None,
    H_step25: float | None,
    required_telemetry: Mapping[str, Any] | None,
    schema_only: bool = False,
    steps: int | None = None,
    probes: Mapping[str, Any] | None = None,
    require_probes: bool = True,
) -> dict[str, Any]:
    """Fail-closed readiness gate — ANY false/missing → ok=False + named stop_reason."""
    if schema_only:
        return {"ok": True, "stop_reason": None, "checks": {"schema_only": True}}

    checks: dict[str, Any] = {}

    parent_ok = (
        expected_parent_sha is not None and str(expected_parent_sha) == PARENT_SHA256
    )
    checks["expected_parent_sha_match"] = parent_ok
    if not parent_ok:
        return {"ok": False, "stop_reason": "readiness_parent_sha_mismatch", "checks": checks}

    if not isinstance(banked_sha, Mapping):
        return {"ok": False, "stop_reason": "readiness_banked_sha_missing", "checks": checks}
    before = banked_sha.get("before")
    after = banked_sha.get("after")
    match = banked_sha.get("match")
    bank_ok = before == PARENT_SHA256 and after == PARENT_SHA256 and match is True and before == after
    checks["banked_sha_match"] = bank_ok
    if not bank_ok:
        return {"ok": False, "stop_reason": "readiness_banked_sha_fail", "checks": checks}

    if not isinstance(frozen_scale_sha, Mapping):
        return {"ok": False, "stop_reason": "readiness_frozen_scale_sha_missing", "checks": checks}
    fs_before = frozen_scale_sha.get("before")
    fs_after = frozen_scale_sha.get("after")
    fs_match = frozen_scale_sha.get("match")
    scale_ok = fs_before is not None and fs_after is not None and fs_before == fs_after and fs_match is True
    checks["frozen_scale_sha_match"] = scale_ok
    if not scale_ok:
        return {"ok": False, "stop_reason": "readiness_frozen_scale_sha_fail", "checks": checks}

    if not isinstance(route_counters, Mapping):
        return {"ok": False, "stop_reason": "readiness_route_counters_missing", "checks": checks}
    n_fixed_raw = route_counters.get("n_fixed_qscale_forwards")
    n_dyn_raw = route_counters.get("n_bitlinear_dynamic_forwards")
    if n_fixed_raw is None or n_dyn_raw is None:
        return {"ok": False, "stop_reason": "readiness_route_counters_missing_fields", "checks": checks}
    n_fixed = int(n_fixed_raw)
    n_dyn = int(n_dyn_raw)
    route_ok = n_fixed > 0 and n_dyn == 0
    checks["route_ok"] = route_ok
    checks["n_fixed_qscale_forwards"] = n_fixed
    checks["n_bitlinear_dynamic_forwards"] = n_dyn
    if not route_ok:
        return {"ok": False, "stop_reason": "readiness_route_counters_fail", "checks": checks}

    if not isinstance(paired_proof, Mapping):
        return {"ok": False, "stop_reason": "readiness_paired_proof_missing", "checks": checks}
    required_paired = (
        "path", "sha256", "protocol", "overhead_frac_AB", "overhead_frac_BA",
        "determinism_prefix_match", "accepted", "plan_sha256", "authority_dispatch",
        "device", "N", "steps", "batch", "topk", "is_proof",
        "two_tier_threshold_assert_pass",
    )
    missing_paired = [k for k in required_paired if k not in paired_proof]
    if missing_paired:
        return {
            "ok": False,
            "stop_reason": f"readiness_paired_proof_fields_missing:{','.join(missing_paired)}",
            "checks": checks,
        }
    if paired_proof.get("is_proof") is not True:
        return {"ok": False, "stop_reason": "readiness_paired_proof_marked_non_proof", "checks": checks}
    if paired_proof.get("plan_sha256") != PLAN_SHA256:
        return {"ok": False, "stop_reason": "readiness_paired_proof_plan_sha_mismatch", "checks": checks}
    if paired_proof.get("authority_dispatch") != AUTHORITY_DISPATCH:
        return {"ok": False, "stop_reason": "readiness_paired_proof_authority_mismatch", "checks": checks}
    if paired_proof.get("accepted") is not True:
        return {"ok": False, "stop_reason": "readiness_paired_proof_not_accepted", "checks": checks}
    if paired_proof.get("determinism_prefix_match") is not True:
        return {"ok": False, "stop_reason": "readiness_paired_determinism_prefix_mismatch", "checks": checks}
    if paired_proof.get("two_tier_threshold_assert_pass") is not True:
        return {"ok": False, "stop_reason": "readiness_paired_threshold_proof_fail", "checks": checks}
    try:
        o_ab = float(paired_proof["overhead_frac_AB"])
        o_ba = float(paired_proof["overhead_frac_BA"])
    except (TypeError, ValueError, KeyError):
        return {"ok": False, "stop_reason": "readiness_paired_overhead_unparseable", "checks": checks}
    if o_ab > 0.15 or o_ba > 0.15:
        return {"ok": False, "stop_reason": "readiness_paired_overhead_exceeded", "checks": checks}
    if paired_determinism_cost_ok is not True:
        return {
            "ok": False,
            "stop_reason": (
                "readiness_paired_determinism_cost_missing"
                if paired_determinism_cost_ok is None
                else "readiness_paired_determinism_cost_fail"
            ),
            "checks": checks,
        }

    if H_step25 is None:
        return {"ok": False, "stop_reason": "readiness_H_step25_missing", "checks": checks}

    if require_probes:
        if not isinstance(probes, Mapping):
            return {"ok": False, "stop_reason": "readiness_probes_missing", "checks": checks}
        for k in (
            "acq_step0_count", "acq_final_count",
            "ret_step0_count", "ret_final_count",
            "step0_taken_before_train",
        ):
            if k not in probes:
                return {"ok": False, "stop_reason": f"readiness_probes_missing_field:{k}", "checks": checks}
        if probes.get("step0_taken_before_train") is not True:
            return {"ok": False, "stop_reason": "readiness_probes_step0_not_before_train", "checks": checks}

    if not isinstance(required_telemetry, Mapping):
        return {"ok": False, "stop_reason": "readiness_required_telemetry_missing", "checks": checks}
    req_keys = (
        "demand", "deferred_survival", "margin_trajectory",
        "episode_trajectory", "demand_per_25", "H_trajectory",
    )
    for k in req_keys:
        val = required_telemetry.get(k)
        if val is None or val == [] or val == {}:
            return {"ok": False, "stop_reason": f"readiness_required_telemetry_absent:{k}", "checks": checks}

    traj_steps = int(steps) if steps is not None else 150
    traj = validate_trajectory_schemas(
        steps=traj_steps,
        margin_trajectory=list(required_telemetry.get("margin_trajectory") or []),
        episode_trajectory=list(required_telemetry.get("episode_trajectory") or []),
        demand_per_25=list(required_telemetry.get("demand_per_25") or []),
        H_trajectory=list(required_telemetry.get("H_trajectory") or []),
    )
    checks["trajectory_schema"] = traj.get("checks")
    if not traj["ok"]:
        return {"ok": False, "stop_reason": traj["stop_reason"], "checks": checks}

    return {"ok": True, "stop_reason": None, "checks": checks}
