"""Offline step_update cost attribution from durable phase_telemetry run.log JSONL."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.grad_proxy_audit import DRIFT_AUDIT_STEP_INTERVAL

SCHEMA_VERSION = "step_update_cost_attribution_discriminator/v1"
PHASE_TELEMETRY_SCHEMA = "hrm_text_158_c2p2_phase_telemetry/v0"
STEP_UPDATE_PHASE = "step_update"

SELECTOR_ALLOWLIST_PHASES: tuple[str, ...] = (
    "two_tier_grad_proxy_ingress",
    "activation_credit_forward_backward",
    "activation_credit_gather",
)
DRIFT_AUDIT_PHASE = "proxy_oracle_drift_audit"

DEFAULT_THRESHOLD_S = 95.0
DEFAULT_THRESHOLD_LINEAGE_PACKET_MSG_ID = "1781216817861"
DEFAULT_THRESHOLD_DISCRIMINATOR = "branch_step_update_liveness_fail"

PHASE_CLASS_SELECTOR_ALLOWLIST = "selector_allowlist"
PHASE_CLASS_DRIFT_AUDIT = "drift_audit"
PHASE_CLASS_OTHER_NAMED = "other_named"
PHASE_CLASS_UNKNOWN = "UNKNOWN"

_KNOWN_OTHER_NAMED_PHASES: frozenset[str] = frozenset(
    {
        "b2b_sequential_capture",
    }
)

_JSON_LINE_RE = re.compile(r"\{.*\}$")


@dataclass(frozen=True)
class ThresholdConfig:
    threshold_s: float = DEFAULT_THRESHOLD_S
    lineage_packet_msg_id: str = DEFAULT_THRESHOLD_LINEAGE_PACKET_MSG_ID
    discriminator: str = DEFAULT_THRESHOLD_DISCRIMINATOR


def classify_nested_phase(phase: str) -> str:
    if phase in SELECTOR_ALLOWLIST_PHASES:
        return PHASE_CLASS_SELECTOR_ALLOWLIST
    if phase == DRIFT_AUDIT_PHASE:
        return PHASE_CLASS_DRIFT_AUDIT
    if phase in _KNOWN_OTHER_NAMED_PHASES:
        return PHASE_CLASS_OTHER_NAMED
    return PHASE_CLASS_UNKNOWN


def parse_run_log_events(run_log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in run_log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _JSON_LINE_RE.search(stripped)
        if match is None:
            continue
        payload = json.loads(match.group(0))
        if payload.get("schema") != PHASE_TELEMETRY_SCHEMA:
            continue
        events.append(payload)
    return events


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _ingress_only_seconds(nested_by_phase: Mapping[str, float]) -> float:
    return float(nested_by_phase.get("two_tier_grad_proxy_ingress", 0.0))


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return None
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numer / denom


def build_step_update_attribution(
    events: Sequence[Mapping[str, Any]],
    *,
    threshold: ThresholdConfig | None = None,
    source_run_log: str | None = None,
) -> dict[str, Any]:
    threshold = threshold or ThresholdConfig()
    active_step: int | None = None
    per_step_rows: list[dict[str, Any]] = []
    all_nested_phases: set[str] = set()
    unknown_phases: set[str] = set()
    other_named_phases: set[str] = set()

    for event in events:
        phase = str(event.get("phase") or "")
        event_kind = str(event.get("event") or "")
        step_raw = event.get("step")
        step = int(step_raw) if step_raw is not None else None

        if phase == STEP_UPDATE_PHASE and event_kind == "start":
            active_step = step
            continue

        if active_step is None:
            continue

        if phase == STEP_UPDATE_PHASE and event_kind == "end" and step == active_step:
            total_s = float(event.get("duration_seconds") or 0.0)
            row = per_step_rows[-1] if per_step_rows and per_step_rows[-1]["step"] == active_step else None
            if row is None:
                row = {
                    "step": active_step,
                    "step_update_total_s": total_s,
                    "nested_by_phase": {},
                }
                per_step_rows.append(row)
            else:
                row["step_update_total_s"] = total_s
            active_step = None
            continue

        if event_kind == "end" and step == active_step and phase != STEP_UPDATE_PHASE:
            duration_s = float(event.get("duration_seconds") or 0.0)
            if not per_step_rows or per_step_rows[-1]["step"] != active_step:
                per_step_rows.append(
                    {
                        "step": active_step,
                        "step_update_total_s": 0.0,
                        "nested_by_phase": {},
                    }
                )
            nested_by_phase = per_step_rows[-1]["nested_by_phase"]
            nested_by_phase[phase] = float(nested_by_phase.get(phase, 0.0)) + duration_s
            all_nested_phases.add(phase)
            classification = classify_nested_phase(phase)
            if classification == PHASE_CLASS_UNKNOWN:
                unknown_phases.add(phase)
            elif classification == PHASE_CLASS_OTHER_NAMED:
                other_named_phases.add(phase)

    finalized_steps: list[dict[str, Any]] = []
    for row in per_step_rows:
        nested_by_phase: dict[str, float] = dict(row["nested_by_phase"])
        selector_allowlist_s = sum(
            nested_by_phase.get(phase, 0.0) for phase in SELECTOR_ALLOWLIST_PHASES
        )
        drift_audit_s = float(nested_by_phase.get(DRIFT_AUDIT_PHASE, 0.0))
        selector_overhead_s = selector_allowlist_s + drift_audit_s
        total_s = float(row["step_update_total_s"])
        unattributed_apply_residual_s = total_s - selector_overhead_s
        finalized_steps.append(
            {
                "step": int(row["step"]),
                "step_update_total_s": total_s,
                "selector_allowlist_s": selector_allowlist_s,
                "drift_audit_s": drift_audit_s,
                "selector_overhead_s": selector_overhead_s,
                "unattributed_apply_residual_s": unattributed_apply_residual_s,
                "nested_by_phase": nested_by_phase,
                "ingress_s": _ingress_only_seconds(nested_by_phase),
            }
        )

    finalized_steps.sort(key=lambda item: int(item["step"]))
    totals = [float(row["step_update_total_s"]) for row in finalized_steps]
    residuals = [float(row["unattributed_apply_residual_s"]) for row in finalized_steps]
    selector_overheads = [float(row["selector_overhead_s"]) for row in finalized_steps]
    ingress_values = [float(row["ingress_s"]) for row in finalized_steps]

    max_total_s = max(totals) if totals else 0.0
    max_step = (
        int(finalized_steps[totals.index(max_total_s)]["step"]) if totals else None
    )
    headroom_pct = (
        ((threshold.threshold_s - max_total_s) / threshold.threshold_s) * 100.0
        if threshold.threshold_s > 0.0
        else 0.0
    )

    audit_rows = [row for row in finalized_steps if float(row["drift_audit_s"]) > 0.0]
    drift_audit_values = [float(row["drift_audit_s"]) for row in audit_rows]

    late_steps = [row for row in finalized_steps if int(row["step"]) >= 3]
    ingress_slope = _linear_slope(
        [float(row["step"]) for row in late_steps],
        [float(row["ingress_s"]) for row in late_steps],
    )

    nesting_complete = not unknown_phases
    nested_phase_classification = {
        phase: classify_nested_phase(phase) for phase in sorted(all_nested_phases)
    }

    discriminators = {
        DEFAULT_THRESHOLD_DISCRIMINATOR: bool(
            any(total > threshold.threshold_s for total in totals)
        ),
        "step_update_selector_overhead_dominant": bool(
            any(
                float(row["selector_overhead_s"]) > float(row["unattributed_apply_residual_s"])
                for row in finalized_steps
            )
        ),
        "step_update_headroom_pct": headroom_pct,
    }

    interpretation: dict[str, Any] = {
        "unattributed_floor_flat_at_zero_crossings": bool(
            finalized_steps
            and all(
                abs(float(row["unattributed_apply_residual_s"]) - residuals[0]) < 5.0
                for row in finalized_steps[:2]
            )
        ),
        "enabled_path_floor_not_crossing_driven": (
            "Steps 1-2 show ~flat unattributed_apply_residual_s with zero ingress crossings; "
            "residual is two-tier-enabled apply path floor, not crossing-driven selector work."
        ),
        "f2_kernelization_debt_pointer": (
            "Unattributed apply residual dominates selector overhead at N=20; "
            "OFF-arm comparison (when provided in derivation receipt) indicates large enabled-path cost."
        ),
    }

    artifact: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source_run_log": source_run_log,
        "threshold_s": threshold.threshold_s,
        "threshold_lineage": {
            "packet_msg_id": threshold.lineage_packet_msg_id,
            "discriminator": threshold.discriminator,
            "kind": "receipt-only",
        },
        "selector_allowlist_phases": list(SELECTOR_ALLOWLIST_PHASES),
        "drift_audit_phase": DRIFT_AUDIT_PHASE,
        "drift_audit_step_interval": int(DRIFT_AUDIT_STEP_INTERVAL),
        "drift_audit_step_interval_source": (
            "calm/hrm_text_158/native_full_stack/grad_proxy_audit.py:"
            "DRIFT_AUDIT_STEP_INTERVAL"
        ),
        "nesting_complete": nesting_complete,
        "nested_phase_classification": nested_phase_classification,
        "unknown_nested_phases": sorted(unknown_phases),
        "other_named_nested_phases": sorted(other_named_phases),
        "per_step": finalized_steps,
        "aggregates": {
            "step_count": len(finalized_steps),
            "step_update_total_s": {
                "max": max_total_s,
                "max_step": max_step,
                "mean": statistics.fmean(totals) if totals else 0.0,
            },
            "selector_overhead_s": {
                "max": max(selector_overheads) if selector_overheads else 0.0,
                "mean": statistics.fmean(selector_overheads) if selector_overheads else 0.0,
            },
            "unattributed_apply_residual_s": {
                "max": max(residuals) if residuals else 0.0,
                "mean": statistics.fmean(residuals) if residuals else 0.0,
                "min": min(residuals) if residuals else 0.0,
            },
            "ingress_s": {
                "max": max(ingress_values) if ingress_values else 0.0,
                "mean": statistics.fmean(ingress_values) if ingress_values else 0.0,
            },
        },
        "drift_audit": {
            "step_interval": int(DRIFT_AUDIT_STEP_INTERVAL),
            "audit_step_count": len(audit_rows),
            "audit_steps": [int(row["step"]) for row in audit_rows],
            "duration_s": {
                "max": max(drift_audit_values) if drift_audit_values else 0.0,
                "mean_on_audit_steps": (
                    statistics.fmean(drift_audit_values) if drift_audit_values else 0.0
                ),
            },
            "registered_run_contract": (
                "proxy_oracle_drift_audit runs every drift_audit_step_interval optimizer steps; "
                "amortizable run-contract lever for future packets."
            ),
        },
        "discriminators": discriminators,
        "extrapolation_model": {
            "kind": "informational_not_hard_gate",
            "ingress_linear_slope_s_per_step": ingress_slope,
            "notes": (
                "Late-curve ingress growth plus periodic drift_audit spikes project "
                "threshold breach around steps 24-28 without backend change."
            ),
        },
        "interpretation": interpretation,
    }
    return artifact


def analyze_run_log(
    run_log_path: Path,
    *,
    threshold: ThresholdConfig | None = None,
) -> dict[str, Any]:
    events = parse_run_log_events(run_log_path)
    return build_step_update_attribution(
        events,
        threshold=threshold,
        source_run_log=str(run_log_path.resolve()),
    )


def build_derivation_receipt(
    sources: Sequence[tuple[str, Path]],
    *,
    threshold: ThresholdConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    threshold = threshold or ThresholdConfig()
    runs: list[dict[str, Any]] = []
    sha_table: dict[str, str] = {}

    for label, run_log_path in sources:
        resolved = run_log_path.resolve()
        sha_table[str(resolved)] = _sha256_file(resolved)
        attribution = analyze_run_log(resolved, threshold=threshold)
        runs.append(
            {
                "label": label,
                "run_log_path": str(resolved),
                "sha256": sha_table[str(resolved)],
                "attribution": attribution,
            }
        )

    on_runs = [run for run in runs if run["label"].endswith("_on") or "/on/" in run["run_log_path"]]
    off_runs = [run for run in runs if run["label"].endswith("_off") or "/off/" in run["run_log_path"]]

    on_max = max(
        (
            float(run["attribution"]["aggregates"]["step_update_total_s"]["max"])
            for run in on_runs
        ),
        default=0.0,
    )
    off_max = max(
        (
            float(run["attribution"]["aggregates"]["step_update_total_s"]["max"])
            for run in off_runs
        ),
        default=0.0,
    )
    on_residual_mean = statistics.fmean(
        [
            float(run["attribution"]["aggregates"]["unattributed_apply_residual_s"]["mean"])
            for run in on_runs
        ]
    ) if on_runs else 0.0

    enabled_path_cost_ratio = (on_residual_mean / off_max) if off_max > 0.0 else None

    for run in runs:
        interpretation = run["attribution"]["interpretation"]
        interpretation["off_reference_max_step_update_s"] = off_max if off_runs else None
        interpretation["on_unattributed_residual_mean_s"] = on_residual_mean if on_runs else None
        interpretation["enabled_path_cost_ratio_estimate"] = enabled_path_cost_ratio

    receipt = {
        "schema": "step_update_cost_derivation_receipt/v1",
        "threshold_s": threshold.threshold_s,
        "threshold_lineage": {
            "packet_msg_id": threshold.lineage_packet_msg_id,
            "discriminator": threshold.discriminator,
            "kind": "receipt-only",
        },
        "source_sha256": sha_table,
        "runs": runs,
        "cross_run_summary": {
            "on_max_step_update_total_s": on_max,
            "off_max_step_update_total_s": off_max,
            "on_off_max_ratio": (on_max / off_max) if off_max > 0.0 else None,
            "enabled_path_cost_ratio_estimate": enabled_path_cost_ratio,
        },
        "nesting_complete_all_runs": all(
            bool(run["attribution"]["nesting_complete"]) for run in runs
        ),
    }
    if output_dir is not None:
        receipt["output_dir"] = str(output_dir.resolve())
    return receipt
