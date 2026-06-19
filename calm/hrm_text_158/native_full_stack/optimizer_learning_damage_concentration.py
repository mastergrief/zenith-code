"""Fail-closed learning-damage concentration audit for HRM-Text-1.58.

CPU/receipt-only lane. Classifies whether drain-induced learning damage is
concentrated (temporal and/or key-local) or diffuse. Picks mechanism branch only;
does not build veto/trust-region or flip readiness.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_learning_damage_concentration_audit/v0.fail_closed"
)
OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_TARGET_NAME = (
    "optimizer_learning_damage_concentration_audit"
)

PARENT_SHA_LOCKED_ARM_A_9DB27EE4 = (
    "9db27ee4543dac49954873fe586ba1d6769000e4081fbb8b155ef5bdc7ef45ef"
)
PARENT_SHA_HISTORICAL_V8C2_4DDEACC8 = (
    "4ddeacc84a4bca05e1a75307af967500c39d4491189141c6749d21e1372bc5be"
)

BRANCH_MEASUREMENT_INVALID = "BR-DAMAGE-MEASUREMENT-INVALID"
BRANCH_CONCENTRATED_KEYS = "BR-DAMAGE-CONCENTRATED-KEYS"
BRANCH_CONCENTRATED_TEMPORAL = "BR-DAMAGE-CONCENTRATED-TEMPORAL"
BRANCH_DIFFUSE = "BR-DAMAGE-DIFFUSE"
BRANCH_UNRESOLVED = "BR-DAMAGE-UNRESOLVED"

TEMPORAL_TOP_K_SHARE = 0.70
TEMPORAL_TOP_K_FRACTION = 0.25
DIFFUSE_FAIL_STEP_FRACTION = 0.50
DIFFUSE_MAX_BUCKET_SHARE = 0.40
KEY_TOP_BUCKET_SHARE = 0.70
KEY_RATE_TOP_BUCKET_SHARE = 0.70
STABILITY_MIN_FAIL_STEPS = 2
STABILITY_PASS_ABSENCE_RATIO = 0.50
STABILITY_CONTROL_ABSENCE_RATIO = 0.50

CONCENTRATION_NON_CLAIMS = (
    "learning damage concentration audit picks mechanism branch only",
    "ready_to_flip and optimizer_credit_state_sub2_claim remain false",
    "this receipt does not build functional-window-veto or trust-region",
    "control parent is stability-only contrast, not a classifier co-predicate",
    "no GPU launch, readiness flip, checkpoint mutation, or mint authority",
)


@dataclass(frozen=True)
class OptimizerLearningDamageConcentrationAuditReceipt:
    schema_version: str
    target_name: str
    parent_receipt_sha256: str
    control_receipt_sha256: str | None
    branch_id: str
    ce_proxy_eps_ce: float
    fail_step_count: int
    pass_step_count: int
    total_excess: float
    top1_excess_share: float
    top3_excess_share: float
    top_k_excess_share: float
    hhi_excess: float
    gini_excess: float
    longest_contiguous_fail_run: int
    temporal_concentrated: bool
    dominant_family: str | None
    lift_dominant_family: str | None
    rate_dominant_family: str | None
    dominant_family_lift_share: float
    dominant_family_rate_lift_share: float
    keys_co_predicate_pass: bool
    stability_recurrence_pass: bool
    stability_passing_absence_pass: bool
    stability_control_absence_pass: bool
    stability_predicate_pass: bool
    diffuse_candidate: bool
    ready_to_flip: bool
    optimizer_credit_state_sub2_claim: bool
    readiness_row_flip_authorized: bool
    mechanism_built: bool
    mint_authority: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            if field.name != "non_claims"
            else list(self.non_claims)
            for field in fields(self)
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_family(module_key: str) -> str:
    level = "H" if ".H_level." in module_key else "L"
    block = "attn" if ".attn." in module_key else "mlp"
    projection = module_key.rsplit(".", 1)[-1]
    return f"{level}/{block}/{projection}"


def _gini(values: Sequence[float]) -> float:
    positives = [float(v) for v in values if float(v) > 0.0]
    if not positives:
        return 0.0
    positives.sort()
    total = sum(positives)
    if total <= 0.0:
        return 0.0
    n = len(positives)
    weighted = sum((2 * i - n - 1) * v for i, v in enumerate(positives, start=1))
    return weighted / (n * total)


def _hhi(shares: Sequence[float]) -> float:
    return sum(float(s) ** 2 for s in shares)


def _extract_pressure_raw(per_key_report: Mapping[str, Any]) -> tuple[float, float, int]:
    if "numel" not in per_key_report:
        raise ValueError("missing numel")
    numel = int(per_key_report["numel"])
    if numel <= 0:
        raise ValueError("numel must be positive")
    pressure = per_key_report.get("pressure_diagnostics")
    if not isinstance(pressure, dict):
        raise ValueError("missing pressure_diagnostics")
    if "unapplied_crossing_count" not in pressure:
        raise ValueError("missing unapplied_crossing_count")
    if "cold_exception_row_count_delta" not in pressure:
        raise ValueError("missing cold_exception_row_count_delta")
    ucc = int(pressure["unapplied_crossing_count"])
    cold_signed = int(pressure["cold_exception_row_count_delta"])
    cold_nonneg = max(0, cold_signed)
    pressure_raw = float(ucc + cold_nonneg)
    pressure_rate = pressure_raw / float(numel)
    return pressure_raw, pressure_rate, cold_signed


def _parse_parent_steps(parent: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    if "ce_proxy_eps_ce" not in parent:
        raise ValueError("missing ce_proxy_eps_ce")
    eps = float(parent["ce_proxy_eps_ce"])
    reports = parent.get("per_step_reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("missing per_step_reports")
    steps: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("per_step_reports entries must be objects")
        if "step_id" not in report:
            raise ValueError("missing step_id")
        if "ce_proxy_delta_rel" not in report:
            raise ValueError("missing ce_proxy_delta_rel")
        if "ce_proxy_delta_within_tolerance" not in report:
            raise ValueError("missing ce_proxy_delta_within_tolerance")
        per_key = report.get("per_key")
        if not isinstance(per_key, dict) or not per_key:
            raise ValueError("missing per_key")
        step_id = int(report["step_id"])
        delta = float(report["ce_proxy_delta_rel"])
        within = bool(report["ce_proxy_delta_within_tolerance"])
        excess = max(0.0, delta - eps)
        fail_step = not within
        key_pressure: dict[str, tuple[float, float, int]] = {}
        for module_key, key_report in per_key.items():
            if not isinstance(key_report, dict):
                raise ValueError("per_key entries must be objects")
            key_pressure[str(module_key)] = _extract_pressure_raw(key_report)
        steps.append(
            {
                "step_id": step_id,
                "delta": delta,
                "within": within,
                "excess": excess,
                "fail_step": fail_step,
                "key_pressure": key_pressure,
            }
        )
    steps.sort(key=lambda row: row["step_id"])
    return steps, eps


def _family_lift_rate_aggregates(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    keys = sorted({k for step in steps for k in step["key_pressure"]})
    fail_steps = [s for s in steps if s["fail_step"]]
    pass_steps = [s for s in steps if not s["fail_step"]]

    failing_mean_pressure: dict[str, float] = {}
    passing_mean_pressure: dict[str, float] = {}
    failing_mean_rate: dict[str, float] = {}
    passing_mean_rate: dict[str, float] = {}

    for key in keys:
        fail_raw = [s["key_pressure"][key][0] for s in fail_steps if key in s["key_pressure"]]
        pass_raw = [s["key_pressure"][key][0] for s in pass_steps if key in s["key_pressure"]]
        fail_rate = [s["key_pressure"][key][1] for s in fail_steps if key in s["key_pressure"]]
        pass_rate = [s["key_pressure"][key][1] for s in pass_steps if key in s["key_pressure"]]
        failing_mean_pressure[key] = sum(fail_raw) / len(fail_raw) if fail_raw else 0.0
        passing_mean_pressure[key] = sum(pass_raw) / len(pass_raw) if pass_raw else 0.0
        failing_mean_rate[key] = sum(fail_rate) / len(fail_rate) if fail_rate else 0.0
        passing_mean_rate[key] = sum(pass_rate) / len(pass_rate) if pass_rate else 0.0

    lift_pass = {
        key: max(0.0, failing_mean_pressure[key] - passing_mean_pressure[key])
        for key in keys
    }
    rate_lift = {
        key: max(0.0, failing_mean_rate[key] - passing_mean_rate[key]) for key in keys
    }

    family_lift: dict[str, float] = {}
    family_rate: dict[str, float] = {}
    family_step_lift: dict[str, dict[int, float]] = {}
    family_step_rate: dict[str, dict[int, float]] = {}

    for step in fail_steps:
        sid = int(step["step_id"])
        excess = float(step["excess"])
        for key, (raw, rate, _cold) in step["key_pressure"].items():
            family = key_family(key)
            lift_score = excess * lift_pass[key]
            rate_score = excess * rate_lift[key]
            family_lift[family] = family_lift.get(family, 0.0) + lift_score
            family_rate[family] = family_rate.get(family, 0.0) + rate_score
            family_step_lift.setdefault(family, {})[sid] = (
                family_step_lift.setdefault(family, {}).get(sid, 0.0) + lift_score
            )
            family_step_rate.setdefault(family, {})[sid] = (
                family_step_rate.setdefault(family, {}).get(sid, 0.0) + rate_score
            )

    return (
        lift_pass,
        rate_lift,
        family_lift,
        family_rate,
        family_step_lift,
        family_step_rate,
    )


def _pick_dominant_family(family_lift: Mapping[str, float], family_rate: Mapping[str, float]) -> str | None:
    if not family_lift:
        return None
    return max(
        family_lift.keys(),
        key=lambda family: (
            float(family_lift.get(family, 0.0)),
            float(family_rate.get(family, 0.0)),
            family,
        ),
    )


def _share_of_total(value: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return float(value) / float(total)


def _temporal_metrics(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_steps = len(steps)
    fail_steps = [s for s in steps if s["fail_step"]]
    pass_steps = [s for s in steps if not s["fail_step"]]
    total_excess = sum(float(s["excess"]) for s in fail_steps)
    fail_excess = sorted(
        ((int(s["step_id"]), float(s["excess"])) for s in fail_steps),
        key=lambda item: (-item[1], item[0]),
    )
    excess_values = [excess for _, excess in fail_excess]
    if total_excess > 0.0:
        shares = [excess / total_excess for excess in excess_values]
        top1 = shares[0] if shares else 0.0
        top3 = sum(shares[:3]) if shares else 0.0
    else:
        shares = []
        top1 = 0.0
        top3 = 0.0
    top_k = max(1, math.ceil(TEMPORAL_TOP_K_FRACTION * n_steps))
    top_k_excess = sum(excess for _, excess in fail_excess[:top_k])
    top_k_share = _share_of_total(top_k_excess, total_excess)

    longest = 0
    current = 0
    for step in steps:
        if step["fail_step"]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    temporal_concentrated = (
        top_k_share >= TEMPORAL_TOP_K_SHARE and longest >= 2 and bool(fail_steps)
    )
    return {
        "fail_step_count": len(fail_steps),
        "pass_step_count": len(pass_steps),
        "total_excess": total_excess,
        "top1_excess_share": top1,
        "top3_excess_share": top3,
        "top_k_excess_share": top_k_share,
        "hhi_excess": _hhi(shares),
        "gini_excess": _gini(excess_values),
        "longest_contiguous_fail_run": longest,
        "temporal_concentrated": temporal_concentrated,
        "top_k": top_k,
        "fail_excess": fail_excess,
    }


def _stability_for_bucket(
    *,
    failing_scores: Sequence[float],
    passing_scores: Sequence[float],
    control_scores: Sequence[float] | None,
) -> dict[str, bool]:
    positive_fail = [float(v) for v in failing_scores if float(v) > 0.0]
    recurrence = len(positive_fail) >= STABILITY_MIN_FAIL_STEPS
    if not positive_fail:
        return {
            "recurrence": False,
            "passing_absence": False,
            "control_absence": True,
            "predicate": False,
        }
    failing_mean = sum(positive_fail) / len(positive_fail)
    passing_mean = (
        sum(float(v) for v in passing_scores) / len(passing_scores)
        if passing_scores
        else 0.0
    )
    passing_absence = passing_mean <= STABILITY_PASS_ABSENCE_RATIO * failing_mean
    if control_scores is None:
        control_absence = True
    else:
        control_mean = (
            sum(float(v) for v in control_scores) / len(control_scores)
            if control_scores
            else 0.0
        )
        control_absence = (
            control_mean <= STABILITY_CONTROL_ABSENCE_RATIO * failing_mean
        )
    predicate = recurrence and passing_absence and control_absence
    return {
        "recurrence": recurrence,
        "passing_absence": passing_absence,
        "control_absence": control_absence,
        "predicate": predicate,
    }


def _control_family_lift_scores(
    control_parent: Mapping[str, Any], family: str
) -> list[float]:
    steps, _eps = _parse_parent_steps(control_parent)
    control_fail = [s for s in steps if s["fail_step"]]
    if not control_fail:
        return []
    _, _, family_lift, _, family_step_lift, _ = _family_lift_rate_aggregates(steps)
    if family not in family_step_lift:
        return [0.0 for _ in control_fail]
    return [float(family_step_lift[family].get(int(s["step_id"]), 0.0)) for s in control_fail]


def build_optimizer_learning_damage_concentration_audit_receipt(
    *,
    parent_receipt_sha256: str,
    steps: Sequence[Mapping[str, Any]],
    ce_proxy_eps_ce: float,
    control_receipt_sha256: str | None = None,
    control_parent: Mapping[str, Any] | None = None,
) -> OptimizerLearningDamageConcentrationAuditReceipt:
    temporal = _temporal_metrics(steps)
    (
        _lift_pass,
        _rate_lift,
        family_lift,
        family_rate,
        family_step_lift,
        _family_step_rate,
    ) = _family_lift_rate_aggregates(steps)

    total_lift = sum(family_lift.values())
    total_rate = sum(family_rate.values())
    dominant_family = _pick_dominant_family(family_lift, family_rate)
    lift_dominant_family = (
        max(family_lift, key=lambda f: (family_lift[f], f)) if family_lift else None
    )
    rate_dominant_family = (
        max(family_rate, key=lambda f: (family_rate[f], f)) if family_rate else None
    )

    dominant_lift_share = (
        _share_of_total(family_lift[dominant_family], total_lift)
        if dominant_family is not None
        else 0.0
    )
    dominant_rate_share_for_lift_family = (
        _share_of_total(family_rate[dominant_family], total_rate)
        if dominant_family is not None
        else 0.0
    )

    keys_co_predicate = bool(
        dominant_family is not None
        and dominant_lift_share >= KEY_TOP_BUCKET_SHARE
        and dominant_rate_share_for_lift_family >= KEY_RATE_TOP_BUCKET_SHARE
    )

    # Stability for keys path uses lift-dominant family bucket scores.
    if dominant_family is not None:
        fail_scores = [
            float(family_step_lift[dominant_family].get(int(s["step_id"]), 0.0))
            for s in steps
            if s["fail_step"]
        ]
        pass_scores = [
            float(family_step_lift[dominant_family].get(int(s["step_id"]), 0.0))
            for s in steps
            if not s["fail_step"]
        ]
        control_scores = (
            _control_family_lift_scores(control_parent, dominant_family)
            if control_parent is not None
            else None
        )
        key_stability = _stability_for_bucket(
            failing_scores=fail_scores,
            passing_scores=pass_scores,
            control_scores=control_scores,
        )
    else:
        key_stability = {
            "recurrence": False,
            "passing_absence": False,
            "control_absence": True,
            "predicate": False,
        }

    # Temporal stability uses per-step excess as bucket score on top-k failing steps.
    top_k = int(temporal["top_k"])
    fail_excess = list(temporal["fail_excess"])
    top_ids = {step_id for step_id, _ in fail_excess[:top_k]}
    temporal_fail_scores = [
        float(s["excess"]) for s in steps if s["fail_step"] and int(s["step_id"]) in top_ids
    ]
    temporal_pass_scores = [
        float(s["excess"]) for s in steps if not s["fail_step"]
    ]
    temporal_stability = _stability_for_bucket(
        failing_scores=temporal_fail_scores,
        passing_scores=temporal_pass_scores,
        control_scores=None,
    )

    n_steps = len(steps)
    fail_fraction = temporal["fail_step_count"] / n_steps if n_steps else 0.0
    max_bucket_share = max(
        float(temporal["top1_excess_share"]),
        float(dominant_lift_share),
        float(dominant_rate_share_for_lift_family),
    )
    diffuse_candidate = (
        fail_fraction >= DIFFUSE_FAIL_STEP_FRACTION
        and max_bucket_share <= DIFFUSE_MAX_BUCKET_SHARE
    )

    branch_id = BRANCH_UNRESOLVED
    if keys_co_predicate and key_stability["predicate"]:
        branch_id = BRANCH_CONCENTRATED_KEYS
    elif bool(temporal["temporal_concentrated"]) and temporal_stability["predicate"]:
        branch_id = BRANCH_CONCENTRATED_TEMPORAL
    elif diffuse_candidate:
        branch_id = BRANCH_DIFFUSE
    elif keys_co_predicate or bool(temporal["temporal_concentrated"]):
        branch_id = BRANCH_UNRESOLVED

    receipt = OptimizerLearningDamageConcentrationAuditReceipt(
        schema_version=OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_SCHEMA_VERSION,
        target_name=OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_TARGET_NAME,
        parent_receipt_sha256=str(parent_receipt_sha256).lower(),
        control_receipt_sha256=(
            str(control_receipt_sha256).lower() if control_receipt_sha256 else None
        ),
        branch_id=branch_id,
        ce_proxy_eps_ce=float(ce_proxy_eps_ce),
        fail_step_count=int(temporal["fail_step_count"]),
        pass_step_count=int(temporal["pass_step_count"]),
        total_excess=float(temporal["total_excess"]),
        top1_excess_share=float(temporal["top1_excess_share"]),
        top3_excess_share=float(temporal["top3_excess_share"]),
        top_k_excess_share=float(temporal["top_k_excess_share"]),
        hhi_excess=float(temporal["hhi_excess"]),
        gini_excess=float(temporal["gini_excess"]),
        longest_contiguous_fail_run=int(temporal["longest_contiguous_fail_run"]),
        temporal_concentrated=bool(temporal["temporal_concentrated"]),
        dominant_family=dominant_family,
        lift_dominant_family=lift_dominant_family,
        rate_dominant_family=rate_dominant_family,
        dominant_family_lift_share=float(dominant_lift_share),
        dominant_family_rate_lift_share=float(dominant_rate_share_for_lift_family),
        keys_co_predicate_pass=bool(keys_co_predicate),
        stability_recurrence_pass=bool(key_stability["recurrence"]),
        stability_passing_absence_pass=bool(key_stability["passing_absence"]),
        stability_control_absence_pass=bool(key_stability["control_absence"]),
        stability_predicate_pass=bool(key_stability["predicate"]),
        diffuse_candidate=bool(diffuse_candidate),
        ready_to_flip=False,
        optimizer_credit_state_sub2_claim=False,
        readiness_row_flip_authorized=False,
        mechanism_built=False,
        mint_authority=False,
        non_claims=CONCENTRATION_NON_CLAIMS,
    )
    validate_optimizer_learning_damage_concentration_audit_receipt(receipt)
    return receipt


def build_measurement_invalid_receipt(
    *,
    parent_receipt_sha256: str,
    reason: str,
) -> OptimizerLearningDamageConcentrationAuditReceipt:
    receipt = OptimizerLearningDamageConcentrationAuditReceipt(
        schema_version=OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_SCHEMA_VERSION,
        target_name=OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_TARGET_NAME,
        parent_receipt_sha256=str(parent_receipt_sha256).lower(),
        control_receipt_sha256=None,
        branch_id=BRANCH_MEASUREMENT_INVALID,
        ce_proxy_eps_ce=0.0,
        fail_step_count=0,
        pass_step_count=0,
        total_excess=0.0,
        top1_excess_share=0.0,
        top3_excess_share=0.0,
        top_k_excess_share=0.0,
        hhi_excess=0.0,
        gini_excess=0.0,
        longest_contiguous_fail_run=0,
        temporal_concentrated=False,
        dominant_family=None,
        lift_dominant_family=None,
        rate_dominant_family=None,
        dominant_family_lift_share=0.0,
        dominant_family_rate_lift_share=0.0,
        keys_co_predicate_pass=False,
        stability_recurrence_pass=False,
        stability_passing_absence_pass=False,
        stability_control_absence_pass=False,
        stability_predicate_pass=False,
        diffuse_candidate=False,
        ready_to_flip=False,
        optimizer_credit_state_sub2_claim=False,
        readiness_row_flip_authorized=False,
        mechanism_built=False,
        mint_authority=False,
        non_claims=CONCENTRATION_NON_CLAIMS,
    )
    if reason:
        _ = reason
    validate_optimizer_learning_damage_concentration_audit_receipt(receipt)
    return receipt


def classify_from_parent_receipt_file(
    path: Path,
    *,
    control_path: Path | None = None,
) -> OptimizerLearningDamageConcentrationAuditReceipt:
    parent_sha = _sha256_file(path) if path.is_file() else ""
    try:
        parent = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parent, dict):
            raise ValueError("parent receipt must be a JSON object")
        steps, eps = _parse_parent_steps(parent)
        control_parent = None
        control_sha = None
        if control_path is not None:
            control_parent = json.loads(control_path.read_text(encoding="utf-8"))
            if not isinstance(control_parent, dict):
                raise ValueError("control receipt must be a JSON object")
            control_sha = _sha256_file(control_path)
        return build_optimizer_learning_damage_concentration_audit_receipt(
            parent_receipt_sha256=parent_sha,
            steps=steps,
            ce_proxy_eps_ce=eps,
            control_receipt_sha256=control_sha,
            control_parent=control_parent,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return build_measurement_invalid_receipt(
            parent_receipt_sha256=parent_sha,
            reason=str(exc),
        )


def validate_optimizer_learning_damage_concentration_audit_receipt(
    receipt: OptimizerLearningDamageConcentrationAuditReceipt,
) -> None:
    if (
        receipt.schema_version
        != OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_SCHEMA_VERSION
    ):
        raise ValueError("concentration audit schema mismatch")
    if receipt.target_name != OPTIMIZER_LEARNING_DAMAGE_CONCENTRATION_AUDIT_TARGET_NAME:
        raise ValueError("concentration audit target mismatch")
    if receipt.ready_to_flip or receipt.optimizer_credit_state_sub2_claim:
        raise ValueError("concentration audit forbids flip/sub2 claims")
    if receipt.readiness_row_flip_authorized or receipt.mechanism_built:
        raise ValueError("concentration audit forbids readiness/mechanism claims")
    if receipt.mint_authority:
        raise ValueError("concentration audit forbids mint authority")
    if receipt.branch_id == BRANCH_MEASUREMENT_INVALID:
        return
    if receipt.branch_id == BRANCH_CONCENTRATED_KEYS:
        if not receipt.keys_co_predicate_pass:
            raise ValueError("CONCENTRATED-KEYS requires keys co-predicate")
        if not receipt.stability_predicate_pass:
            raise ValueError("CONCENTRATED-KEYS requires stability predicate")
        if receipt.lift_dominant_family != receipt.dominant_family:
            raise ValueError("dominant family must match lift-dominant family")
    if receipt.non_claims != CONCENTRATION_NON_CLAIMS:
        raise ValueError("concentration audit non_claims must be exact")
