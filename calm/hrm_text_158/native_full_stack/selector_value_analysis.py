"""Receipt-only selector-value identity and outcome analysis for paired probe runs."""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_STATE_KEY = "model.H_level.core.layers.0.attn.gqkv_proj"
REQUIRED_CURRICULUM_SEED = 44
REQUIRED_SUPPORT_ORDER_SEED = 44
OUTCOME_EPSILON = 1e-6
PRIMARY_STEP_MIN = 3
PRIMARY_STEP_MAX = 10
TRAJECTORY_STEP_MIN = 2
TRAJECTORY_STEP_MAX = 10
CONTEXT_STEP = 1
DIRECTIONAL_PERSISTENCE_MIN_STEPS = 6
TRAJECTORY_STEP_COUNT = TRAJECTORY_STEP_MAX - TRAJECTORY_STEP_MIN + 1

FORBIDDEN_CLAIMS = (
    "Positive selector-value claim without dual science read",
    "Rank-order claims from topK hash16 fields",
    "Cross-arm raw selector-score comparison (within-arm semantics differ by arm)",
    "Population-level or statistically independent claims from N=10 trajectory verdicts",
)

ON_SCORE_SEMANTICS = "local_loss_delta_at_applied_flat_index"
OFF_SCORE_SEMANTICS = "abs_new_acc_at_applied_flat_index"


@dataclass(frozen=True)
class ScheduleGuardResult:
    ok: bool
    issues: tuple[str, ...]


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def load_paired_receipts(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    on = json.loads((run_root / "on" / "receipt.json").read_text(encoding="utf-8"))
    off = json.loads((run_root / "off" / "receipt.json").read_text(encoding="utf-8"))
    return on, off


def _tensor_stats(step_entry: Mapping[str, Any], state_key: str) -> dict[str, Any]:
    return (
        step_entry.get("step_result", {})
        .get("tensor_stats", {})
        .get(state_key, {})
    )


def extract_cap_window_steps(
    receipt: Mapping[str, Any],
    *,
    state_key: str = DEFAULT_STATE_KEY,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for key, entry in receipt.get("step_reports", {}).items():
        step = int(key)
        ts = _tensor_stats(entry, state_key)
        ingress = entry.get("grad_proxy_ingress") or {}
        crossing = ingress.get("grad_proxy_ingress_crossing_eligible_count")
        if crossing is None:
            crossing = (ingress.get("crossing_count_by_state_key") or {}).get(state_key)
        applied_indices = list(ts.get("applied_indices") or [])
        out[step] = {
            "applied_indices": applied_indices,
            "applied_flat_indices_hash16": ts.get("applied_flat_indices_hash16"),
            "top8_hash16": ts.get("top8_flat_indices_hash16"),
            "top64_hash16": ts.get("top64_flat_indices_hash16"),
            "top4096_hash16": ts.get("top4096_flat_indices_hash16"),
            "cap_window_jaccard_vs_prior_step": ts.get("cap_window_jaccard_vs_prior_step"),
            "score_p50": ts.get("applied_selection_score_p50"),
            "score_p95": ts.get("applied_selection_score_p95"),
            "score_semantics": ts.get("applied_selection_score_semantics"),
            "q_sha256_after": ts.get("q_sha256_after"),
            "votes_sha256": ts.get("votes_sha256"),
            "crossing": crossing,
            "candidate_count": ts.get("candidate_count"),
            "applied_count": ts.get("applied_count") or len(applied_indices),
        }
    return out


def build_identity_tables(
    on_steps: Mapping[int, Mapping[str, Any]],
    off_steps: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    steps = sorted(set(on_steps) | set(off_steps))
    c1: list[dict[str, Any]] = []
    c2: list[dict[str, Any]] = []
    c3_on: list[dict[str, Any]] = []
    c3_off: list[dict[str, Any]] = []
    c5: list[dict[str, Any]] = []

    for step in steps:
        on_row = on_steps.get(step, {})
        off_row = off_steps.get(step, {})
        on_applied = list(on_row.get("applied_indices") or [])
        off_applied = list(off_row.get("applied_indices") or [])
        cross_j = (
            jaccard(on_applied, off_applied)
            if on_applied and off_applied
            else None
        )
        c1.append(
            {
                "step": step,
                "cross_arm_jaccard": cross_j,
                "on_applied_count": len(on_applied),
                "off_applied_count": len(off_applied),
                "on_crossing": on_row.get("crossing"),
                "off_candidate": off_row.get("candidate_count"),
                "on_hash16": on_row.get("applied_flat_indices_hash16"),
                "off_hash16": off_row.get("applied_flat_indices_hash16"),
            }
        )
        c2.append(
            {
                "step": step,
                "q_match": (
                    on_row.get("q_sha256_after") == off_row.get("q_sha256_after")
                    and on_row.get("q_sha256_after") is not None
                ),
                "votes_match": (
                    on_row.get("votes_sha256") == off_row.get("votes_sha256")
                    and on_row.get("votes_sha256") is not None
                ),
                "on_q": on_row.get("q_sha256_after"),
                "off_q": off_row.get("q_sha256_after"),
            }
        )
        if on_row.get("score_p50") is not None:
            c3_on.append(
                {
                    "step": step,
                    "p50": on_row.get("score_p50"),
                    "p95": on_row.get("score_p95"),
                    "semantics": on_row.get("score_semantics"),
                }
            )
        if off_row.get("score_p50") is not None:
            c3_off.append(
                {
                    "step": step,
                    "p50": off_row.get("score_p50"),
                    "p95": off_row.get("score_p95"),
                    "semantics": off_row.get("score_semantics"),
                }
            )
        if on_row.get("cap_window_jaccard_vs_prior_step") is not None:
            c5.append(
                {
                    "step": step,
                    "on_cap_jaccard_vs_prior": on_row.get("cap_window_jaccard_vs_prior_step"),
                }
            )

    primary = [
        row
        for row in c1
        if PRIMARY_STEP_MIN <= row["step"] <= PRIMARY_STEP_MAX
        and row["on_applied_count"] > 0
        and row["off_applied_count"] > 0
    ]
    cross_vals = [
        row["cross_arm_jaccard"]
        for row in primary
        if row["cross_arm_jaccard"] is not None
    ]
    q_vote_by_step = {row["step"]: row for row in c2}
    verdict, verdict_meta = identity_verdict(primary, q_vote_by_step)

    return {
        "verdict": verdict,
        "verdict_meta": verdict_meta,
        "c1_cross_arm_jaccard": c1,
        "c1_primary_steps_3_10": primary,
        "c2_q_votes": c2,
        "c3_within_arm_scores": {"on": c3_on, "off": c3_off},
        "c5_on_cap_rotation": c5,
        "cross_arm_jaccard_stats_steps_3_10": {
            "min": min(cross_vals) if cross_vals else None,
            "max": max(cross_vals) if cross_vals else None,
            "mean": statistics.fmean(cross_vals) if cross_vals else None,
            "all_zero": all(v == 0.0 for v in cross_vals) if cross_vals else None,
        },
        "score_semantics_guard": score_semantics_guard(on_steps, off_steps),
        "cannot": list(FORBIDDEN_CLAIMS),
    }


def identity_verdict(
    primary_rows: Sequence[Mapping[str, Any]],
    q_vote_rows_by_step: Mapping[int, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not primary_rows:
        return "selector_value_different_unresolved", {"partial_overlap_present": False}

    cross_vals = [
        float(row["cross_arm_jaccard"])
        for row in primary_rows
        if row.get("cross_arm_jaccard") is not None
    ]
    if len(cross_vals) != len(primary_rows):
        return "selector_value_different_unresolved", {"partial_overlap_present": False}

    q_matches = [
        bool(q_vote_rows_by_step[int(row["step"])]["q_match"])
        for row in primary_rows
        if int(row["step"]) in q_vote_rows_by_step
    ]
    if cross_vals and q_matches and all(q_matches) and all(v == 1.0 for v in cross_vals):
        return "selector_value_committed_null", {"partial_overlap_present": False}

    if all(0.0 < v < 1.0 for v in cross_vals):
        return "selector_value_measurable_signal", {"partial_overlap_present": False}

    partial_overlap_present = any(v > 0.0 for v in cross_vals)
    return "selector_value_different_unresolved", {
        "partial_overlap_present": partial_overlap_present,
    }


def score_semantics_guard(
    on_steps: Mapping[int, Mapping[str, Any]],
    off_steps: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    on_sem = {
        row.get("score_semantics")
        for row in on_steps.values()
        if row.get("score_semantics") is not None
    }
    off_sem = {
        row.get("score_semantics")
        for row in off_steps.values()
        if row.get("score_semantics") is not None
    }
    return {
        "on_semantics": sorted(on_sem),
        "off_semantics": sorted(off_sem),
        "cross_arm_score_compare_forbidden": True,
        "expected_on_semantics": ON_SCORE_SEMANTICS,
        "expected_off_semantics": OFF_SCORE_SEMANTICS,
        "on_semantics_ok": ON_SCORE_SEMANTICS in on_sem or not on_sem,
        "off_semantics_ok": OFF_SCORE_SEMANTICS in off_sem or not off_sem,
    }


def overlap_band_characterization(
    on_steps: Mapping[int, Mapping[str, Any]],
    off_steps: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in sorted(set(on_steps) | set(off_steps)):
        if not (PRIMARY_STEP_MIN <= step <= PRIMARY_STEP_MAX):
            continue
        on_applied = list(on_steps.get(step, {}).get("applied_indices") or [])
        off_applied = list(off_steps.get(step, {}).get("applied_indices") or [])
        if not on_applied or not off_applied:
            continue
        intersection = sorted(set(on_applied) & set(off_applied))
        cap = max(len(on_applied), len(off_applied), 1)
        row: dict[str, Any] = {
            "step": step,
            "intersection_count": len(intersection),
            "intersection_fraction_of_cap": len(intersection) / cap,
            "subordinate_non_headline": True,
        }
        if intersection:
            row["intersection_index_min"] = min(intersection)
            row["intersection_index_max"] = max(intersection)
            row["intersection_index_median"] = statistics.median(intersection)
        rows.append(row)
    return rows


def extract_outcome_step(
    receipt: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    entry = receipt.get("step_reports", {}).get(str(step), {})
    support_batch = dict(entry.get("support_batch") or {})
    metrics = dict(entry.get("metrics") or {})
    return {
        "step": step,
        "loss": float(entry.get("loss", 0.0)),
        "loss_finite": bool(entry.get("loss_finite", True)),
        "metrics": metrics,
        "exact_accuracy": list(metrics.get("exact_accuracy") or []),
        "support_batch_hash16": support_batch.get("batch_content_hash16"),
        "support_row_ids": list(support_batch.get("row_ids") or []),
    }


def _receipt_seed_guard(receipt: Mapping[str, Any], arm: str) -> list[str]:
    issues: list[str] = []
    batch = receipt.get("batch") or {}
    if batch.get("seed") != REQUIRED_CURRICULUM_SEED:
        issues.append(f"{arm}_batch_seed_not_{REQUIRED_CURRICULUM_SEED}")
    if batch.get("support_order_seed") != REQUIRED_SUPPORT_ORDER_SEED:
        issues.append(f"{arm}_support_order_seed_not_{REQUIRED_SUPPORT_ORDER_SEED}")
    return issues


def check_schedule_guards(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
) -> ScheduleGuardResult:
    issues: list[str] = []
    for receipt, arm in ((on, "on"), (off, "off")):
        if receipt.get("steps_completed") != 10:
            issues.append(f"{arm}_steps_completed_not_10")
        issues.extend(_receipt_seed_guard(receipt, arm))

    for step in range(1, 11):
        on_step = on.get("step_reports", {}).get(str(step))
        off_step = off.get("step_reports", {}).get(str(step))
        if on_step is None:
            issues.append(f"on_missing_step_{step}")
            continue
        if off_step is None:
            issues.append(f"off_missing_step_{step}")
            continue
        if not on_step.get("loss_finite", True) or not off_step.get("loss_finite", True):
            issues.append(f"step_{step}_loss_not_finite")
        on_support = on_step.get("support_batch") or {}
        off_support = off_step.get("support_batch") or {}
        on_hash = on_support.get("batch_content_hash16")
        off_hash = off_support.get("batch_content_hash16")
        if on_hash != off_hash:
            issues.append(f"step_{step}_support_batch_hash_mismatch")
        on_rows = on_support.get("row_ids")
        off_rows = off_support.get("row_ids")
        if on_rows is not None and off_rows is not None and list(on_rows) != list(off_rows):
            issues.append(f"step_{step}_support_row_ids_mismatch")

    return ScheduleGuardResult(ok=not issues, issues=tuple(issues))


def build_outcome_tables(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
) -> dict[str, Any]:
    guards = check_schedule_guards(on, off)
    per_step: list[dict[str, Any]] = []
    for step in range(1, 11):
        on_row = extract_outcome_step(on, step)
        off_row = extract_outcome_step(off, step)
        delta = on_row["loss"] - off_row["loss"]
        exact_accuracy_match = on_row["exact_accuracy"] == off_row["exact_accuracy"]
        per_step.append(
            {
                "step": step,
                "on_loss": on_row["loss"],
                "off_loss": off_row["loss"],
                "delta_on_minus_off": delta,
                "support_batch_hash_match": (
                    on_row["support_batch_hash16"] == off_row["support_batch_hash16"]
                ),
                "support_row_ids_match": on_row["support_row_ids"] == off_row["support_row_ids"],
                "exact_accuracy_on": on_row["exact_accuracy"],
                "exact_accuracy_off": off_row["exact_accuracy"],
                "exact_accuracy_match": exact_accuracy_match,
                "loss_finite": on_row["loss_finite"] and off_row["loss_finite"],
            }
        )

    trajectory_rows = [
        row for row in per_step if TRAJECTORY_STEP_MIN <= row["step"] <= TRAJECTORY_STEP_MAX
    ]
    cumulative_delta = sum(row["delta_on_minus_off"] for row in trajectory_rows)
    mean_delta = (
        statistics.fmean(row["delta_on_minus_off"] for row in trajectory_rows)
        if trajectory_rows
        else 0.0
    )

    verdict_payload = outcome_verdict(
        guards=guards,
        trajectory_rows=trajectory_rows,
        mean_delta=mean_delta,
        cumulative_delta=cumulative_delta,
    )

    return {
        "guards": {"ok": guards.ok, "issues": list(guards.issues)},
        "per_step": per_step,
        "context_step_1": next((row for row in per_step if row["step"] == CONTEXT_STEP), None),
        "trajectory_steps_2_10": trajectory_rows,
        "secondary_non_headline": {
            "cumulative_delta_on_minus_off_steps_2_10": cumulative_delta,
            "mean_delta_on_minus_off_steps_2_10": mean_delta,
        },
        **verdict_payload,
        "cannot": list(FORBIDDEN_CLAIMS),
        "trajectory_scope": (
            "Descriptive verdict for fixed N=10 matched-support trajectory only; "
            "not a population selector-value claim."
        ),
    }


def _accuracy_primary_delta(row: Mapping[str, Any]) -> float | None:
    on_acc = row.get("exact_accuracy_on") or []
    off_acc = row.get("exact_accuracy_off") or []
    if not on_acc or not off_acc:
        return None
    return float(on_acc[0]) - float(off_acc[0])


def _has_opposite_metric_direction(trajectory_rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in trajectory_rows:
        loss_delta = float(row["delta_on_minus_off"])
        acc_delta = _accuracy_primary_delta(row)
        if acc_delta is None:
            continue
        if abs(loss_delta) <= OUTCOME_EPSILON or abs(acc_delta) <= OUTCOME_EPSILON:
            continue
        # Higher ON loss => ON worse; higher ON accuracy => ON better.
        loss_favors_off = loss_delta > OUTCOME_EPSILON
        loss_favors_on = loss_delta < -OUTCOME_EPSILON
        accuracy_favors_on = acc_delta > OUTCOME_EPSILON
        accuracy_favors_off = acc_delta < -OUTCOME_EPSILON
        if (loss_favors_off and accuracy_favors_on) or (loss_favors_on and accuracy_favors_off):
            return True
    return False


def outcome_verdict(
    *,
    guards: ScheduleGuardResult,
    trajectory_rows: Sequence[Mapping[str, Any]],
    mean_delta: float,
    cumulative_delta: float,
) -> dict[str, Any]:
    if not guards.ok:
        return {
            "verdict": "outcome_analysis_insufficient_surface",
            "accuracy_tie_caveat": False,
            "directional_persistence_steps": 0,
        }

    if _has_opposite_metric_direction(trajectory_rows):
        return {
            "verdict": "outcome_diverges_direction_unresolved",
            "accuracy_tie_caveat": False,
            "directional_persistence_steps": 0,
            "unresolved_reason": "opposite_metric_direction",
        }

    loss_indistinguishable = all(
        abs(float(row["delta_on_minus_off"])) <= OUTCOME_EPSILON for row in trajectory_rows
    )
    metrics_match = all(bool(row["exact_accuracy_match"]) for row in trajectory_rows)
    if loss_indistinguishable and metrics_match:
        return {
            "verdict": "outcome_indistinguishable",
            "accuracy_tie_caveat": False,
            "directional_persistence_steps": 0,
        }
    if loss_indistinguishable and not metrics_match:
        return {
            "verdict": "outcome_diverges_direction_unresolved",
            "accuracy_tie_caveat": False,
            "directional_persistence_steps": 0,
            "unresolved_reason": "metrics_differ_with_indistinguishable_loss",
        }

    positive_steps = sum(
        1 for row in trajectory_rows if float(row["delta_on_minus_off"]) > OUTCOME_EPSILON
    )
    negative_steps = sum(
        1 for row in trajectory_rows if float(row["delta_on_minus_off"]) < -OUTCOME_EPSILON
    )
    accuracy_tie_caveat = all(bool(row["exact_accuracy_match"]) for row in trajectory_rows)

    favors_off = (
        mean_delta > OUTCOME_EPSILON
        and cumulative_delta > OUTCOME_EPSILON
        and positive_steps >= DIRECTIONAL_PERSISTENCE_MIN_STEPS
    )
    favors_on = (
        mean_delta < -OUTCOME_EPSILON
        and cumulative_delta < -OUTCOME_EPSILON
        and negative_steps >= DIRECTIONAL_PERSISTENCE_MIN_STEPS
    )

    if favors_off and not favors_on:
        return {
            "verdict": "outcome_trajectory_favors_OFF",
            "accuracy_tie_caveat": accuracy_tie_caveat,
            "directional_persistence_steps": positive_steps,
        }
    if favors_on and not favors_off:
        return {
            "verdict": "outcome_trajectory_favors_ON",
            "accuracy_tie_caveat": accuracy_tie_caveat,
            "directional_persistence_steps": negative_steps,
        }

    return {
        "verdict": "outcome_diverges_direction_unresolved",
        "accuracy_tie_caveat": accuracy_tie_caveat,
        "directional_persistence_steps": max(positive_steps, negative_steps),
        "unresolved_reason": "loss_diverges_without_symmetric_directional_agreement",
    }


def run_identity_analysis(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
    *,
    state_key: str = DEFAULT_STATE_KEY,
    include_overlap_band: bool = True,
) -> dict[str, Any]:
    on_steps = extract_cap_window_steps(on, state_key=state_key)
    off_steps = extract_cap_window_steps(off, state_key=state_key)
    summary = build_identity_tables(on_steps, off_steps)
    if include_overlap_band:
        summary["overlap_band_subordinate"] = overlap_band_characterization(on_steps, off_steps)
    summary["steps_completed"] = {
        "on": on.get("steps_completed"),
        "off": off.get("steps_completed"),
    }
    return summary


def run_outcome_analysis(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
) -> dict[str, Any]:
    return build_outcome_tables(on, off)


def run_full_analysis(
    on: Mapping[str, Any],
    off: Mapping[str, Any],
    *,
    state_key: str = DEFAULT_STATE_KEY,
) -> dict[str, Any]:
    identity = run_identity_analysis(on, off, state_key=state_key, include_overlap_band=True)
    outcome = run_outcome_analysis(on, off)
    return {"identity": identity, "outcome": outcome}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_analysis_memo(
    path: Path,
    *,
    identity: Mapping[str, Any] | None,
    outcome: Mapping[str, Any] | None,
) -> None:
    lines = ["# Selector-Value Analysis Memo\n\n"]
    if identity is not None:
        lines.append(f"## Identity verdict: `{identity['verdict']}`\n\n")
        if identity.get("verdict_meta"):
            lines.append(f"- verdict_meta: `{identity['verdict_meta']}`\n")
        stats = identity.get("cross_arm_jaccard_stats_steps_3_10") or {}
        lines.append(f"- cross_arm_jaccard_stats_steps_3_10: `{stats}`\n\n")
    if outcome is not None:
        lines.append(f"## Outcome verdict: `{outcome['verdict']}`\n\n")
        lines.append(f"- trajectory_scope: {outcome.get('trajectory_scope')}\n")
        if outcome.get("accuracy_tie_caveat"):
            lines.append("- accuracy_tie_caveat: true\n")
        secondary = outcome.get("secondary_non_headline") or {}
        lines.append(
            "- secondary_non_headline cumulative_delta: "
            f"`{secondary.get('cumulative_delta_on_minus_off_steps_2_10')}`\n\n"
        )
        lines.append("## Trajectory steps 2-10\n\n")
        for row in outcome.get("trajectory_steps_2_10") or []:
            lines.append(
                f"- step {row['step']}: delta={row['delta_on_minus_off']}, "
                f"on_loss={row['on_loss']}, off_loss={row['off_loss']}\n"
            )
    lines.append("\n## Cannot claim\n\n")
    for claim in FORBIDDEN_CLAIMS:
        lines.append(f"- {claim}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_run_manifest(
    path: Path,
    *,
    run_root: Path,
    mode: str,
    argv: Sequence[str],
    repo_head: str | None,
    script_path: Path | None,
    on_receipt: Path,
    off_receipt: Path,
    output_paths: Sequence[Path],
) -> dict[str, Any]:
    manifest = {
        "run_root": str(run_root),
        "mode": mode,
        "argv": list(argv),
        "repo_head": repo_head,
        "script_sha256": sha256_file(script_path) if script_path and script_path.exists() else None,
        "input_receipt_sha256": {
            "on": sha256_file(on_receipt),
            "off": sha256_file(off_receipt),
        },
        "output_paths": [str(item) for item in output_paths],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
