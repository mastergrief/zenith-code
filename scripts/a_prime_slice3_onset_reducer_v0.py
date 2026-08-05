"""Pure A′ slice-3 onset/shape reducers — stdlib only, no IO side effects.

Plan authority: A_prime_slice3_dense_collapse_onset_PLAN_v5.json
(sha 7ba78320eab8f0d5cc92b5af86a6d158ee2e473d5185547d95aeba178d43d567).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 35, 50)
BASELINE_COUNT = 1484
BASELINE_TOTAL = 1485
ENDPOINT_ANCHOR_COUNT = 62
N50_ACCEPT_LO = 47
N50_ACCEPT_HI = 77
START_AGG_LO = 1483
START_AGG_HI = 1485
GLOBAL_HORIZON_REQUIRED = 50
PARENT_SHA_EXPECTED = (
    "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
)
RECOVERY_ABS_COUNT = 30  # ceil(0.02 * 1485)
ELIGIBLE_MODULE_DEFAULT = "model.H_level.core.layers.0.attn.gqkv_proj"

CLASS_PRIORITY: tuple[str, ...] = (
    "LIVENESS_OR_INSTRUMENT_FAIL",
    "PREFIX_EQUIVALENCE_FAIL",
    "NO_REPRODUCTION_OR_ENDPOINT_DRIFT",
    "NONMONOTONE_OR_MULTI_CLIFF",
    "COLLAPSE_AT_STEP_1",
    "THRESHOLD_CLIFF",
    "GRADUAL_DRIFT",
    "UNCLASSIFIED_SHAPE",
)

PREFIX_TOP_FIELDS = (
    "bp_steps",
    "global_horizon",
    "loss",
    "metrics",
    "q_changed_count",
)
PREFIX_SUPPORT_FIELDS = (
    "batch_content_hash16",
    "row_ids",
)
PREFIX_TENSOR_FIELDS = (
    "q_sha256_before",
    "q_sha256_after",
    "votes_sha256",
    "applied_flat_indices_hash16",
)


def _strict_count(report: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(report, Mapping):
        return None, None
    if "strict_exact_count" in report and "strict_exact_total" in report:
        return int(report["strict_exact_count"]), int(report["strict_exact_total"])
    se = report.get("strict_exact")
    if isinstance(se, str) and "/" in se:
        a, b = se.split("/", 1)
        return int(a), int(b)
    return None, None


def aggregate_strict_exact(
    reports: Mapping[str, Any] | None, *, supports: Sequence[str] = ("L0b", "math_a0")
) -> tuple[int | None, int | None]:
    if not isinstance(reports, Mapping):
        return None, None
    total_c = 0
    total_t = 0
    for name in supports:
        c, t = _strict_count(reports.get(name) if isinstance(reports.get(name), Mapping) else None)
        if c is None or t is None:
            return None, None
        total_c += c
        total_t += t
    return total_c, total_t


def extract_horizon_point(receipt: Mapping[str, Any], *, expected_n: int) -> dict[str, Any]:
    """Extract liveness fields + final aggregate count from one dense receipt."""
    out: dict[str, Any] = {
        "expected_n": int(expected_n),
        "ok": False,
        "reasons": [],
        "start_count": None,
        "start_total": None,
        "final_count": None,
        "final_total": None,
        "steps_completed": receipt.get("steps_completed"),
        "parent_hash_before": receipt.get("parent_hash_before"),
        "parent_hash_after": receipt.get("parent_hash_after"),
        "parent_hash_unchanged": receipt.get("parent_hash_unchanged"),
        "global_horizon": None,
        "prior_audit_enabled": None,
    }
    # C1: nested terminal_status authority (fail-closed)
    ts_status = receipt.get("terminal_status")
    if not isinstance(ts_status, Mapping):
        out["reasons"].append("terminal_status_missing_or_not_dict")
    else:
        if ts_status.get("planned_return_code") != 0:
            out["reasons"].append(
                f"terminal_status.planned_return_code!={ts_status.get('planned_return_code')!r}"
            )
        if ts_status.get("producer_clean_completion") is not True:
            out["reasons"].append(
                f"terminal_status.producer_clean_completion!={ts_status.get('producer_clean_completion')!r}"
            )
    pa = receipt.get("prior_audit")
    if not isinstance(pa, Mapping) or not pa.get("enabled"):
        out["reasons"].append("prior_audit_missing_or_disabled")
        out["ok"] = False
        return out
    out["prior_audit_enabled"] = True
    start_c, start_t = aggregate_strict_exact(pa.get("start_reports"))
    final_c, final_t = aggregate_strict_exact(pa.get("final_reports"))
    out["start_count"] = start_c
    out["start_total"] = start_t
    out["final_count"] = final_c
    out["final_total"] = final_t
    if start_c is None or final_c is None:
        out["reasons"].append("strict_exact_unparseable")
        out["ok"] = False
        return out
    steps = receipt.get("steps_completed")
    if steps != expected_n:
        out["reasons"].append(f"steps_completed!={expected_n}:{steps!r}")
    if receipt.get("parent_hash_unchanged") is not True:
        out["reasons"].append("parent_hash_changed_or_missing")
    phb = receipt.get("parent_hash_before")
    pha = receipt.get("parent_hash_after")
    if phb != pha:
        out["reasons"].append("parent_hash_before_ne_after")
    if phb != PARENT_SHA_EXPECTED:
        out["reasons"].append("parent_hash_ne_expected")
    # C2: exact step coverage {"1"..str(N)} + global_horizon on EVERY step
    sr = receipt.get("step_reports")
    if not isinstance(sr, Mapping):
        out["reasons"].append("step_reports_missing_or_not_dict")
        out["ok"] = len(out["reasons"]) == 0
        return out
    expected_keys = {str(k) for k in range(1, int(expected_n) + 1)}
    # accept int keys as well as str for coverage check
    actual_keys = set()
    for k in sr.keys():
        actual_keys.add(str(k))
    if actual_keys != expected_keys:
        out["reasons"].append(
            f"step_reports_coverage!={sorted(expected_keys)!r}:got={sorted(actual_keys)!r}"
        )
    gh_ref = None
    for k in range(1, int(expected_n) + 1):
        sk = str(k)
        step = sr.get(sk) if sk in sr else sr.get(k)
        if not isinstance(step, Mapping):
            out["reasons"].append(f"step_{k}_missing")
            continue
        gh_k = step.get("global_horizon")
        if gh_k != GLOBAL_HORIZON_REQUIRED:
            out["reasons"].append(
                f"step_{k}_global_horizon!={GLOBAL_HORIZON_REQUIRED}:{gh_k!r}"
            )
        if gh_ref is None:
            gh_ref = gh_k
    out["global_horizon"] = gh_ref
    if not (START_AGG_LO <= int(start_c) <= START_AGG_HI):
        out["reasons"].append(f"start_aggregate_out_of_band:{start_c}")
    if start_t != BASELINE_TOTAL or final_t != BASELINE_TOTAL:
        out["reasons"].append(f"denom_ne_{BASELINE_TOTAL}:{start_t}/{final_t}")
    req = pa.get("requested_supports") or []
    if set(req) < {"L0b", "math_a0"} and "L0b" not in (pa.get("start_reports") or {}):
        out["reasons"].append("supports_missing")
    out["ok"] = len(out["reasons"]) == 0
    return out


def _step_prefix_payload(
    step_report: Mapping[str, Any], *, eligible_module: str
) -> dict[str, Any] | None:
    try:
        payload: dict[str, Any] = {}
        for f in PREFIX_TOP_FIELDS:
            if f not in step_report:
                return None
            payload[f] = step_report[f]
        sb = step_report.get("support_batch")
        if not isinstance(sb, Mapping):
            return None
        for f in PREFIX_SUPPORT_FIELDS:
            if f not in sb:
                return None
            payload[f"support_batch.{f}"] = sb[f]
        ts = (step_report.get("step_result") or {}).get("tensor_stats") or {}
        if not isinstance(ts, Mapping) or eligible_module not in ts:
            return None
        mod = ts[eligible_module]
        if not isinstance(mod, Mapping):
            return None
        for f in PREFIX_TENSOR_FIELDS:
            if f not in mod:
                return None
            payload[f"tensor.{f}"] = mod[f]
        return payload
    except Exception:
        return None


def check_prefix_equivalence(
    horizon_receipts: Mapping[int, Mapping[str, Any]],
    *,
    reference_n: int = 50,
    eligible_module: str = ELIGIBLE_MODULE_DEFAULT,
) -> tuple[bool, list[str]]:
    """Compare steps 1..N of each truncated receipt to reference N=50 receipt."""
    reasons: list[str] = []
    ref = horizon_receipts.get(reference_n)
    if not isinstance(ref, Mapping):
        return False, ["missing_reference_receipt"]
    ref_sr = ref.get("step_reports")
    if not isinstance(ref_sr, Mapping):
        return False, ["reference_missing_step_reports"]
    for n in HORIZONS:
        if n == reference_n:
            continue
        rec = horizon_receipts.get(n)
        if not isinstance(rec, Mapping):
            reasons.append(f"missing_horizon_{n}")
            continue
        sr = rec.get("step_reports")
        if not isinstance(sr, Mapping):
            reasons.append(f"horizon_{n}_missing_step_reports")
            continue
        for k in range(1, n + 1):
            sk = str(k)
            a = sr.get(sk) or sr.get(k)
            b = ref_sr.get(sk) or ref_sr.get(k)
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                reasons.append(f"N{n}_step{k}_missing")
                continue
            pa = _step_prefix_payload(a, eligible_module=eligible_module)
            pb = _step_prefix_payload(b, eligible_module=eligible_module)
            if pa is None or pb is None:
                reasons.append(f"N{n}_step{k}_field_missing")
                continue
            if pa != pb:
                reasons.append(f"N{n}_step{k}_mismatch")
    return (len(reasons) == 0), reasons


def classify_from_counts(
    counts_by_n: Mapping[int, int],
    *,
    baseline_count: int = BASELINE_COUNT,
    liveness_ok: bool = True,
    prefix_ok: bool = True,
) -> dict[str, Any]:
    """Pure morphology classifier over horizon final counts (keys must include all HORIZONS)."""
    details: dict[str, Any] = {
        "counts_by_n": {str(k): int(counts_by_n[k]) for k in sorted(counts_by_n)},
        "baseline_count": int(baseline_count),
    }
    if not liveness_ok:
        return {
            "branch": "LIVENESS_OR_INSTRUMENT_FAIL",
            "details": details,
        }
    if not prefix_ok:
        return {
            "branch": "PREFIX_EQUIVALENCE_FAIL",
            "details": details,
        }
    missing = [n for n in HORIZONS if n not in counts_by_n]
    if missing:
        return {
            "branch": "LIVENESS_OR_INSTRUMENT_FAIL",
            "details": {**details, "missing_horizons": missing},
        }
    c50 = int(counts_by_n[50])
    details["count_n50"] = c50
    if c50 < N50_ACCEPT_LO or c50 > N50_ACCEPT_HI:
        return {
            "branch": "NO_REPRODUCTION_OR_ENDPOINT_DRIFT",
            "details": {
                **details,
                "accepted_inclusive": [N50_ACCEPT_LO, N50_ACCEPT_HI],
                "anchor": ENDPOINT_ANCHOR_COUNT,
            },
        }
    d_total = int(baseline_count) - c50
    details["total_drop_D"] = d_total
    ordered = list(HORIZONS)
    drops: list[tuple[int, int, int]] = []
    recoveries: list[tuple[int, int, int]] = []
    for a, b in zip(ordered, ordered[1:]):
        ca = int(counts_by_n[a])
        cb = int(counts_by_n[b])
        drop_ab = max(0, ca - cb)
        drops.append((a, b, drop_ab))
        rec = cb - ca
        if rec >= RECOVERY_ABS_COUNT:
            recoveries.append((a, b, rec))
    details["adjacent_drops"] = [
        {"from": a, "to": b, "drop": drop} for a, b, drop in drops
    ]
    details["recoveries"] = [
        {"from": a, "to": b, "gain": g} for a, b, g in recoveries
    ]

    multi_cliff = sum(1 for _, _, drop in drops if d_total > 0 and drop >= 0.30 * d_total)
    if recoveries or multi_cliff >= 2:
        return {
            "branch": "NONMONOTONE_OR_MULTI_CLIFF",
            "details": {**details, "multi_cliff_bins": multi_cliff},
        }

    c1 = int(counts_by_n[1])
    if d_total > 0 and c1 <= baseline_count - 0.50 * d_total:
        return {
            "branch": "COLLAPSE_AT_STEP_1",
            "details": {**details, "count_n1": c1},
        }

    if d_total > 0:
        big = [(a, b, drop) for a, b, drop in drops if drop >= 0.50 * d_total]
        mid = [(a, b, drop) for a, b, drop in drops if drop >= 0.30 * d_total]
        if len(big) == 1 and len(mid) == 1:
            return {
                "branch": "THRESHOLD_CLIFF",
                "details": {
                    **details,
                    "cliff": {"from": big[0][0], "to": big[0][1], "drop": big[0][2]},
                },
            }

    vals = [int(counts_by_n[n]) for n in ordered]
    non_increasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    long_enough = len(ordered) >= 3
    every_small = d_total > 0 and all(drop < 0.50 * d_total for _, _, drop in drops)
    if d_total > 0 and non_increasing and long_enough and every_small:
        return {
            "branch": "GRADUAL_DRIFT",
            "details": details,
        }

    return {
        "branch": "UNCLASSIFIED_SHAPE",
        "details": details,
    }


def classify_suite(
    horizon_receipts: Mapping[int, Mapping[str, Any]],
    *,
    skip_prefix: bool = False,
    eligible_module: str = ELIGIBLE_MODULE_DEFAULT,
) -> dict[str, Any]:
    """Full suite classification from N→receipt mapping.

    Note: input sha binding is external (launch packet + finalize source_shas).
    """
    points: dict[str, Any] = {}
    liveness_ok = True
    liveness_reasons: list[str] = []
    counts: dict[int, int] = {}
    for n in HORIZONS:
        rec = horizon_receipts.get(n)
        if not isinstance(rec, Mapping):
            liveness_ok = False
            liveness_reasons.append(f"missing_receipt_N{n}")
            points[str(n)] = {"ok": False, "reasons": ["missing"]}
            continue
        pt = extract_horizon_point(rec, expected_n=n)
        points[str(n)] = pt
        if not pt["ok"]:
            liveness_ok = False
            liveness_reasons.extend([f"N{n}:{r}" for r in pt["reasons"]])
        else:
            counts[n] = int(pt["final_count"])

    prefix_ok = True
    prefix_reasons: list[str] = []
    if not skip_prefix and liveness_ok:
        prefix_ok, prefix_reasons = check_prefix_equivalence(
            horizon_receipts, eligible_module=eligible_module
        )
    elif skip_prefix:
        prefix_ok = True
        prefix_reasons = ["skipped"]

    if not liveness_ok or len(counts) != len(HORIZONS):
        result = {
            "branch": "LIVENESS_OR_INSTRUMENT_FAIL",
            "details": {
                "points": points,
                "liveness_reasons": liveness_reasons,
                "prefix_ok": prefix_ok,
                "prefix_reasons": prefix_reasons,
            },
        }
    else:
        result = classify_from_counts(
            counts, liveness_ok=True, prefix_ok=prefix_ok
        )
        result["details"] = {
            **result.get("details", {}),
            "points": points,
            "prefix_ok": prefix_ok,
            "prefix_reasons": prefix_reasons,
        }
        if not prefix_ok:
            result["branch"] = "PREFIX_EQUIVALENCE_FAIL"

    assert result["branch"] in CLASS_PRIORITY
    return {
        "schema": "a_prime_slice3_onset_classification/v0",
        "branch": result["branch"],
        "class_priority": list(CLASS_PRIORITY),
        "horizons": list(HORIZONS),
        "details": result.get("details", {}),
        "claim_boundary": {
            "morphology_only": True,
            "pre_cause": True,
            "pre_carrier": True,
        },
    }
