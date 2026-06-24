"""CPU band-sweep harness for V4-LIVE event-coded carrier feasibility."""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    DEFAULT_VERDICT_NUMEL,
    DenseOracleState,
    EventCodedAccLiveState,
    decisive_surface_drift_count,
    decisive_surface_drift_details,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.votes_emit_dynamics_replay import (
    CLASSIFIER_INTRINSIC_WIDE_CONFIRMED,
    CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
    CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS,
    CLASSIFIER_STATIC_PROXY_ARTIFACT,
)

V4_LIVE_BAND_SWEEP_SCHEMA_VERSION = "hrm_text_158_v4_live_band_sweep/v0"
DEFAULT_DEMOTION_BAND_SWEEP = tuple(range(1, 7))
DYNAMICS_CLASS_SYNTHETIC = "synthetic_adversarial"
HOT_RISK_PROXY_LABEL = "numeric"

CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE = "qualify_proceed_to_live_gpu"
CPU_VERDICT_NO_QUALIFY_SYNTHETIC = "no_qualify_on_adversarial_synthetic"
CPU_VERDICT_MISSING_DECISIVE_ARM_SIGNAL = "missing_decisive_arm_signal_synthetic"


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    steps: tuple[Mapping[int, int], ...]


@dataclass(frozen=True)
class BandSweepRow:
    demotion_band: int
    hot_exact_row_count_p95: float
    ledger_pass: bool
    decisive_surface_drift_count: int
    r4v_acc_inclusive_physical_bits_per_weight: float


@dataclass(frozen=True)
class BandSweepResult:
    rows: tuple[BandSweepRow, ...]
    qualifying_bands: tuple[int, ...]
    cpu_verdict: str
    verdict_numel: int
    dynamics_class: str = DYNAMICS_CLASS_SYNTHETIC
    hot_risk_proxy: str = HOT_RISK_PROXY_LABEL
    reducible_banking_deferred_to_live: bool = True
    drift_root_cause: str = ""
    band_knob_live: bool = False


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def default_adversarial_scenarios(*, numel: int) -> tuple[SyntheticScenario, ...]:
    del numel  # scenarios are lane-local; numel only bounds index validity.
    delayed = (
        {0: 5},
        {},
        {},
        {0: 5},
    )
    decay_only = (
        {0: 6},
        {},
        {},
        {},
        {},
    )
    oscillation = (
        {0: 5, 1: -5},
        {},
        {0: -5, 1: 5},
        {},
        {0: 6, 1: -6},
        {},
    )
    demotion_band_sensitivity = (
        {0: 3},
        {},
        {},
        {0: 7},
        {0: 3},
    )
    delayed_demotion_gap = (
        {0: 5},
        {},
        {0: 5},
    )
    return (
        SyntheticScenario(name="delayed_crossing_sparse_votes", steps=delayed),
        SyntheticScenario(name="decay_without_vote", steps=decay_only),
        SyntheticScenario(name="oscillation_two_lane", steps=oscillation),
        SyntheticScenario(name="demotion_band_sensitivity", steps=demotion_band_sensitivity),
        SyntheticScenario(name="delayed_demotion_gap", steps=delayed_demotion_gap),
    )


def _run_scenario_pair(
    *,
    scenario: SyntheticScenario,
    numel: int,
    demotion_band: int,
) -> tuple[list[int], int, list[dict[str, object]]]:
    carrier = EventCodedAccLiveState(
        logical_numel=int(numel),
        demotion_band=int(demotion_band),
    )
    oracle = DenseOracleState.zeros(int(numel))
    hot_counts: list[int] = []
    for step_index, votes in enumerate(scenario.steps):
        carrier.apply_step(step_index, votes=votes)
        oracle.apply_step(step_index, votes=votes)
        hot_counts.append(int(carrier.step_records[-1].hot_exact_row_count))
    drift = decisive_surface_drift_count(carrier.step_records, oracle.step_records)
    details = decisive_surface_drift_details(carrier.step_records, oracle.step_records)
    return hot_counts, int(drift), details


def _ledger_pass_for_payload(
    *,
    numel: int,
    carrier: EventCodedAccLiveState,
) -> tuple[bool, float, float, float]:
    payload = carrier.to_checkpoint_payload()
    qstate = QScaleWeightState(
        q_levels=torch.zeros((int(numel), 1), dtype=torch.int8),
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    report = measure_r4v_event_coded_acc_budget(
        [qstate],
        [payload],
        state_keys=["acc"],
    )
    payload_bytes = (
        int(report.r4v_actual_events_payload_bytes)
        + int(report.r4v_actual_backlog_payload_bytes)
        + int(report.r4v_actual_hot_exact_payload_bytes)
    )
    metadata_bytes = int(report.r4v_actual_acc_metadata_bytes)
    payload_bpw = float(payload_bytes * 8 / int(numel))
    metadata_bpw = float(metadata_bytes * 8 / int(numel))
    return (
        bool(report.r4v_ledger_pass),
        float(report.r4v_acc_inclusive_physical_bits_per_weight),
        float(payload_bpw),
        float(metadata_bpw),
    )


def _ledger_pass_for_carrier(
    carrier: EventCodedAccLiveState,
) -> tuple[bool, float]:
    ledger_pass, inclusive_bpw, _payload_bpw, _metadata_bpw = _ledger_pass_for_payload(
        numel=int(carrier.logical_numel),
        carrier=carrier,
    )
    return ledger_pass, inclusive_bpw


def demotion_band_knob_is_live(rows: Sequence[BandSweepRow]) -> bool:
    drift_values = {int(row.decisive_surface_drift_count) for row in rows}
    hot_values = {float(row.hot_exact_row_count_p95) for row in rows}
    return len(drift_values) > 1 or len(hot_values) > 1


def classify_drift_root_cause(
    *,
    rows: Sequence[BandSweepRow],
    sample_details: Sequence[dict[str, object]],
) -> str:
    if not sample_details:
        return "no_decisive_surface_drift_observed"
    mismatch_fields = {
        str(field)
        for detail in sample_details
        for field in detail.get("mismatch_fields", ())
    }
    if mismatch_fields == {"decisive_q_snapshot"}:
        return "harness_sparse_vs_dense_q_comparison_bug_fixed"
    if "crossing_indices" in mismatch_fields or "applied_indices" in mismatch_fields:
        if demotion_band_knob_is_live(rows):
            return "demotion_lossy_delayed_crossing_drift"
        return "structural_crossing_or_applied_mismatch"
    return "mixed_decisive_surface_drift"


def run_band_sweep(
    *,
    numel: int | None = None,
    demotion_bands: Iterable[int] = DEFAULT_DEMOTION_BAND_SWEEP,
    scenarios: Sequence[SyntheticScenario] | None = None,
) -> BandSweepResult:
    verdict_numel = int(numel if numel is not None else DEFAULT_VERDICT_NUMEL)
    scenario_list = tuple(scenarios or default_adversarial_scenarios(numel=verdict_numel))
    rows: list[BandSweepRow] = []
    qualifying: list[int] = []
    sample_details: list[dict[str, object]] = []
    for demotion_band in demotion_bands:
        hot_counts: list[int] = []
        total_drift = 0
        for scenario in scenario_list:
            counts, drift, details = _run_scenario_pair(
                scenario=scenario,
                numel=verdict_numel,
                demotion_band=int(demotion_band),
            )
            hot_counts.extend(counts)
            total_drift += int(drift)
            if demotion_band == 1 and not sample_details:
                sample_details.extend(details)
        carrier = EventCodedAccLiveState(
            logical_numel=verdict_numel,
            demotion_band=int(demotion_band),
        )
        for scenario in scenario_list:
            for step_index, votes in enumerate(scenario.steps):
                carrier.apply_step(step_index, votes=votes)
        ledger_pass, inclusive_bpw, payload_bpw, metadata_bpw = _ledger_pass_for_payload(
            numel=verdict_numel,
            carrier=carrier,
        )
        del payload_bpw, metadata_bpw
        row = BandSweepRow(
            demotion_band=int(demotion_band),
            hot_exact_row_count_p95=float(_percentile([float(item) for item in hot_counts], 95.0)),
            ledger_pass=bool(ledger_pass),
            decisive_surface_drift_count=int(total_drift),
            r4v_acc_inclusive_physical_bits_per_weight=float(inclusive_bpw),
        )
        rows.append(row)
        if row.ledger_pass and row.decisive_surface_drift_count == 0:
            qualifying.append(int(demotion_band))
    band_live = demotion_band_knob_is_live(rows)
    drift_root_cause = classify_drift_root_cause(rows=rows, sample_details=sample_details)
    if qualifying:
        cpu_verdict = CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE
    elif all(not row.ledger_pass for row in rows):
        cpu_verdict = CPU_VERDICT_MISSING_DECISIVE_ARM_SIGNAL
    else:
        cpu_verdict = CPU_VERDICT_NO_QUALIFY_SYNTHETIC
    return BandSweepResult(
        rows=tuple(rows),
        qualifying_bands=tuple(qualifying),
        cpu_verdict=str(cpu_verdict),
        verdict_numel=int(verdict_numel),
        drift_root_cause=str(drift_root_cause),
        band_knob_live=bool(band_live),
    )


def pareto_frontier_rows(rows: Sequence[BandSweepRow]) -> list[dict[str, Any]]:
    frontier: list[BandSweepRow] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other.hot_exact_row_count_p95 <= row.hot_exact_row_count_p95
                and other.decisive_surface_drift_count <= row.decisive_surface_drift_count
                and (
                    other.hot_exact_row_count_p95 < row.hot_exact_row_count_p95
                    or other.decisive_surface_drift_count < row.decisive_surface_drift_count
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return [
        {
            "demotion_band": int(item.demotion_band),
            "hot_exact_row_count_p95": float(item.hot_exact_row_count_p95),
            "ledger_pass": bool(item.ledger_pass),
            "decisive_surface_drift_count": int(item.decisive_surface_drift_count),
            "r4v_acc_inclusive_physical_bits_per_weight": float(
                item.r4v_acc_inclusive_physical_bits_per_weight
            ),
        }
        for item in frontier
    ]


def build_sweep_table_payload(result: BandSweepResult) -> dict[str, Any]:
    rows = [
        {
            "demotion_band": int(row.demotion_band),
            "hot_exact_row_count_p95": float(row.hot_exact_row_count_p95),
            "ledger_pass": bool(row.ledger_pass),
            "decisive_surface_drift_count": int(row.decisive_surface_drift_count),
            "r4v_acc_inclusive_physical_bits_per_weight": float(
                row.r4v_acc_inclusive_physical_bits_per_weight
            ),
        }
        for row in result.rows
    ]
    return {
        "schema_version": V4_LIVE_BAND_SWEEP_SCHEMA_VERSION,
        "dynamics_class": str(result.dynamics_class),
        "hot_risk_proxy": str(result.hot_risk_proxy),
        "verdict_numel": int(result.verdict_numel),
        "band_knob_live": bool(result.band_knob_live),
        "drift_root_cause": str(result.drift_root_cause),
        "cpu_verdict": str(result.cpu_verdict),
        "qualifying_bands": [int(item) for item in result.qualifying_bands],
        "reducible_banking_deferred_to_live": bool(result.reducible_banking_deferred_to_live),
        "rows": rows,
        "pareto_frontier": pareto_frontier_rows(result.rows),
        "summary": {
            "qualifying_band_exists": bool(result.qualifying_bands),
            "necessary_not_sufficient_note": (
                "A qualifying synthetic band is necessary-but-not-sufficient; "
                "REDUCIBLE_UNDER_DYNAMICS banks only on the live GPU run."
            ),
            "no_qualify_is_strong_infeasibility_signal": (
                "If no band achieves ledger_pass and drift==0 on adversarial-synthetic, "
                "treat as a strong MISSING_DECISIVE_ARM signal before GPU spend."
            ),
        },
    }


def write_sweep_table_json(*, run_root: Path, result: BandSweepResult) -> Path:
    out_dir = Path(run_root) / "v4_live_band_sweep" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_table.json"
    out_path.write_text(
        json.dumps(build_sweep_table_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def classify_cpu_synthetic_verdict(result: BandSweepResult) -> str:
    return str(result.cpu_verdict)


def map_cpu_verdict_to_terminal_classifier(
    *,
    cpu_verdict: str,
    replay_only: bool,
) -> str:
    if replay_only:
        return CLASSIFIER_STATIC_PROXY_ARTIFACT
    if cpu_verdict == CPU_VERDICT_QUALIFY_PROCEED_TO_LIVE:
        return CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    if cpu_verdict in (
        CPU_VERDICT_NO_QUALIFY_SYNTHETIC,
        CPU_VERDICT_MISSING_DECISIVE_ARM_SIGNAL,
    ):
        return CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    return CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW


def assert_cpu_never_banks_reducible_or_intrinsic(
    *,
    cpu_verdict: str,
    mapped_classifier: str,
) -> None:
    if mapped_classifier in (
        CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS,
        CLASSIFIER_INTRINSIC_WIDE_CONFIRMED,
    ):
        raise AssertionError(
            f"CPU synthetic sweep must not bank {mapped_classifier}; got verdict={cpu_verdict!r}"
        )


def run_representative_ms_scenario(*, numel: int, demotion_band: int = 3) -> EventCodedAccLiveState:
    carrier = EventCodedAccLiveState(logical_numel=int(numel), demotion_band=int(demotion_band))
    carrier.apply_step(0, votes={0: 6})
    for step in range(1, 6):
        carrier.apply_step(step, votes={})
    carrier.apply_step(6, votes={0: 6})
    return carrier
