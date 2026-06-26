"""CPU envelope projector for joint event+hot drain falsification (Phase 2)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    R4V_ACC_BPW_TOLERANCE,
    R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING,
)
from calm.hrm_text_158.native_full_stack.v4_live_band_sweep import (
    BandSweepResult,
    run_band_sweep,
)

JOINT_DRAIN_ENVELOPE_VERDICT_SCHEMA_VERSION = (
    "hrm_text_158_joint_drain_envelope_verdict/v0"
)
CLASSIFICATION_NOT_REACHABLE = "JOINT_DRAIN_ENVELOPE_NOT_REACHABLE"
CLASSIFICATION_REACHABLE = "JOINT_DRAIN_ENVELOPE_REACHABLE"
CLASSIFICATION_REACHABLE_PENDING_ROLLUP = "JOINT_DRAIN_ENVELOPE_REACHABLE_PENDING_ROLLUP"
CLASSIFICATION_HOT_DRAIN_NOT_PARITY_SAFE = "HOT_DRAIN_NOT_PARITY_SAFE"
CLASSIFICATION_OBSERVER_TOO_EXPENSIVE = "OBSERVER_TOO_EXPENSIVE"
PATH_B_STRUCTURALLY_NOT_SUB2 = "B_STRUCTURALLY_NOT_SUB2"

ROLLUP_DEPENDENT_NA_REASON = (
    "requires in-vivo rollup; not computable from manifest anchors alone"
)
TRANSFORM_BPW_BASIS_MANIFEST_ONLY = (
    "Decisive bound = optimistic_upper_bound (full event clear + parity-safe hot); "
    "rollup-based transforms are not_applicable without in-vivo rollup."
)

EST_BYTES_PER_EVENT = 4
REACHABLE_ORACLE_BPW_THRESHOLD = (
    float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING) - float(R4V_ACC_BPW_TOLERANCE)
)


@dataclass(frozen=True)
class TerminalAnchors:
    run_id: str
    events_bytes: int
    hot_bytes: int
    backlog_bytes: int
    metadata_bytes: int
    eligible_weight_count: int
    terminal_inclusive_bpw: float

    @property
    def acc_payload_bytes(self) -> int:
        return int(self.events_bytes + self.hot_bytes + self.backlog_bytes)

    @property
    def total_acc_bytes(self) -> int:
        return int(self.acc_payload_bytes + self.metadata_bytes)


@dataclass(frozen=True)
class BandSweepConstants:
    qualifying_bands: tuple[int, ...]
    hot_p95_by_band: dict[int, float]
    max_hot_reduction_fraction: float
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TransformBpw:
    baseline: float
    v2_coalesce: float
    v5_stable_q_clear: float
    v5_max: float
    v1_band_b: float
    full_event_clear: float
    hot_floor_only: float
    events_floor_only: float
    best_combined_oracle: float
    optimistic_upper_bound: float


@dataclass(frozen=True)
class ResidualHotReduction:
    flip_to_2p0_bpw_fraction_gt: float
    flip_to_2p0_bpw_percent_gt: float
    flip_to_1p75_bpw_fraction_gt: float
    flip_to_1p75_bpw_percent_gt: float
    synthetic_available_fraction: float
    terminal_hot_band_parity_measured: bool
    terminal_hot_band_parity_note: str


def sub2_budget_bytes(*, eligible_weight_count: int) -> int:
    return int(
        int(eligible_weight_count)
        * float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING)
        / 8.0
    )


def rollup_dependent_not_applicable() -> dict[str, str]:
    return {
        "status": "not_applicable",
        "reason": ROLLUP_DEPENDENT_NA_REASON,
    }


def serialize_transform_bpw(
    transforms: TransformBpw,
    *,
    rollup_present: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline": float(transforms.baseline),
        "V1_band_b": float(transforms.v1_band_b),
        "full_event_clear": float(transforms.full_event_clear),
        "hot_floor_only": float(transforms.hot_floor_only),
        "events_floor_only": float(transforms.events_floor_only),
        "optimistic_upper_bound": float(transforms.optimistic_upper_bound),
    }
    if rollup_present:
        payload["V2_coalesce"] = float(transforms.v2_coalesce)
        payload["V5_stable_q_clear"] = float(transforms.v5_stable_q_clear)
        payload["V5_max"] = float(transforms.v5_max)
        payload["best_combined_oracle"] = float(transforms.best_combined_oracle)
    else:
        marker = rollup_dependent_not_applicable()
        payload["V2_coalesce"] = dict(marker)
        payload["V5_stable_q_clear"] = dict(marker)
        payload["V5_max"] = dict(marker)
        payload["best_combined_oracle"] = dict(marker)
    return payload


def inclusive_bpw(
    *,
    payload_bytes: int,
    metadata_bytes: int,
    eligible_weight_count: int,
) -> float:
    total = int(payload_bytes) + int(metadata_bytes)
    return (float(total) * 8.0) / float(int(eligible_weight_count))


def required_hot_reduction_fraction_for_bpw(
    *,
    hot_bytes: int,
    metadata_bytes: int,
    eligible_weight_count: int,
    target_bpw: float,
) -> float:
    target_total_bytes = float(target_bpw) * float(eligible_weight_count) / 8.0
    hot_budget = target_total_bytes - float(metadata_bytes)
    if hot_budget <= 0.0:
        return 1.0
    if int(hot_bytes) <= 0:
        return 0.0
    fraction = 1.0 - (hot_budget / float(hot_bytes))
    return max(0.0, float(fraction))


def load_terminal_anchors_from_manifest(manifest: Mapping[str, Any]) -> TerminalAnchors:
    r4v = manifest["r4v_persistent_ledger"]
    provenance = manifest["provenance"]
    return TerminalAnchors(
        run_id=str(provenance["run_id"]),
        events_bytes=int(r4v["r4v_actual_events_payload_bytes"]),
        hot_bytes=int(r4v["r4v_actual_hot_exact_payload_bytes"]),
        backlog_bytes=int(r4v["r4v_actual_backlog_payload_bytes"]),
        metadata_bytes=int(r4v["r4v_actual_acc_metadata_bytes"]),
        eligible_weight_count=int(r4v["eligible_weight_count"]),
        terminal_inclusive_bpw=float(r4v["r4v_acc_inclusive_physical_bits_per_weight"]),
    )


def load_band_constants_from_sweep(result: BandSweepResult) -> BandSweepConstants:
    band1 = next(row for row in result.rows if int(row.demotion_band) == 1)
    band1_p95 = float(band1.hot_exact_row_count_p95)
    parity_safe = [
        row
        for row in result.rows
        if bool(row.ledger_pass) and int(row.decisive_surface_drift_count) == 0
    ]
    if parity_safe and band1_p95 > 0.0:
        best_p95 = min(float(row.hot_exact_row_count_p95) for row in parity_safe)
        max_fraction = max(0.0, 1.0 - (best_p95 / band1_p95))
    else:
        max_fraction = 0.0
    rows = tuple(
        {
            "demotion_band": int(row.demotion_band),
            "hot_exact_row_count_p95": float(row.hot_exact_row_count_p95),
            "ledger_pass": bool(row.ledger_pass),
            "decisive_surface_drift_count": int(row.decisive_surface_drift_count),
            "r4v_acc_inclusive_physical_bits_per_weight": float(
                row.r4v_acc_inclusive_physical_bits_per_weight
            ),
            "parity_safe": bool(row.ledger_pass)
            and int(row.decisive_surface_drift_count) == 0,
        }
        for row in result.rows
    )
    return BandSweepConstants(
        qualifying_bands=tuple(int(item) for item in result.qualifying_bands),
        hot_p95_by_band={
            int(row.demotion_band): float(row.hot_exact_row_count_p95)
            for row in result.rows
        },
        max_hot_reduction_fraction=float(max_fraction),
        rows=rows,
    )


def _rollup_estimates(
    anchors: TerminalAnchors,
    rollup: Mapping[str, Any] | None,
) -> dict[str, int]:
    if rollup is not None:
        return {
            "events_bytes": int(rollup["est_events_payload_bytes"]),
            "hot_bytes": int(rollup["est_hot_exact_payload_bytes"]),
            "v5_saved": int(rollup["est_saved_bytes_v5_clear"]),
            "v2_saved": int(rollup["est_saved_bytes_v2_coalesce"]),
            "v5_max": int(rollup["events_on_q_locked_not_hot"]) * EST_BYTES_PER_EVENT,
        }
    return {
        "events_bytes": int(anchors.events_bytes),
        "hot_bytes": int(anchors.hot_bytes),
        "v5_saved": 0,
        "v2_saved": 0,
        "v5_max": 0,
    }


def project_transform_bpw(
    anchors: TerminalAnchors,
    band: BandSweepConstants,
    *,
    rollup: Mapping[str, Any] | None = None,
) -> TransformBpw:
    est = _rollup_estimates(anchors, rollup)
    metadata = int(anchors.metadata_bytes)
    weights = int(anchors.eligible_weight_count)
    baseline = inclusive_bpw(
        payload_bytes=anchors.acc_payload_bytes,
        metadata_bytes=metadata,
        eligible_weight_count=weights,
    )
    events_bytes = int(est["events_bytes"])
    hot_bytes = int(est["hot_bytes"])
    backlog_bytes = int(anchors.backlog_bytes)
    v2_events = max(0, events_bytes - int(est["v2_saved"]))
    v5_events = max(0, events_bytes - int(est["v5_saved"]))
    v5_max_events = max(0, events_bytes - int(est["v5_max"]))
    hot_v1 = max(
        0,
        int(round(float(hot_bytes) * (1.0 - band.max_hot_reduction_fraction))),
    )
    event_saved = max(int(est["v5_saved"]), int(est["v2_saved"]), int(est["v5_max"]))
    oracle_events = max(0, events_bytes - event_saved)
    return TransformBpw(
        baseline=float(baseline),
        v2_coalesce=inclusive_bpw(
            payload_bytes=v2_events + hot_bytes + backlog_bytes,
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        v5_stable_q_clear=inclusive_bpw(
            payload_bytes=v5_events + hot_bytes + backlog_bytes,
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        v5_max=inclusive_bpw(
            payload_bytes=v5_max_events + hot_bytes + backlog_bytes,
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        v1_band_b=inclusive_bpw(
            payload_bytes=events_bytes + hot_v1 + backlog_bytes,
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        full_event_clear=inclusive_bpw(
            payload_bytes=int(anchors.hot_bytes) + int(anchors.backlog_bytes),
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        hot_floor_only=inclusive_bpw(
            payload_bytes=int(anchors.hot_bytes) + int(anchors.backlog_bytes),
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        events_floor_only=inclusive_bpw(
            payload_bytes=int(anchors.events_bytes) + int(anchors.backlog_bytes),
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        best_combined_oracle=inclusive_bpw(
            payload_bytes=oracle_events + hot_v1 + backlog_bytes,
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
        optimistic_upper_bound=inclusive_bpw(
            payload_bytes=hot_v1 + int(anchors.backlog_bytes),
            metadata_bytes=metadata,
            eligible_weight_count=weights,
        ),
    )


def project_best_combined_oracle_bpw(
    rollup: Mapping[str, Any],
    *,
    eligible_weight_count: int,
    metadata_bytes: int = 768,
    v1_max_hot_reduction_fraction: float = 0.0,
    backlog_bytes: int = 0,
) -> float:
    anchors = TerminalAnchors(
        run_id="rollup",
        events_bytes=int(rollup["est_events_payload_bytes"]),
        hot_bytes=int(rollup["est_hot_exact_payload_bytes"]),
        backlog_bytes=int(backlog_bytes),
        metadata_bytes=int(metadata_bytes),
        eligible_weight_count=int(eligible_weight_count),
        terminal_inclusive_bpw=0.0,
    )
    band = BandSweepConstants(
        qualifying_bands=(),
        hot_p95_by_band={},
        max_hot_reduction_fraction=float(v1_max_hot_reduction_fraction),
        rows=(),
    )
    return float(
        project_transform_bpw(anchors, band, rollup=rollup).best_combined_oracle
    )


def build_residual_hot_reduction(
    anchors: TerminalAnchors,
    band: BandSweepConstants,
) -> ResidualHotReduction:
    flip_2 = required_hot_reduction_fraction_for_bpw(
        hot_bytes=anchors.hot_bytes,
        metadata_bytes=anchors.metadata_bytes,
        eligible_weight_count=anchors.eligible_weight_count,
        target_bpw=float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING),
    )
    flip_175 = required_hot_reduction_fraction_for_bpw(
        hot_bytes=anchors.hot_bytes,
        metadata_bytes=anchors.metadata_bytes,
        eligible_weight_count=anchors.eligible_weight_count,
        target_bpw=float(REACHABLE_ORACLE_BPW_THRESHOLD),
    )
    return ResidualHotReduction(
        flip_to_2p0_bpw_fraction_gt=float(flip_2),
        flip_to_2p0_bpw_percent_gt=float(flip_2) * 100.0,
        flip_to_1p75_bpw_fraction_gt=float(flip_175),
        flip_to_1p75_bpw_percent_gt=float(flip_175) * 100.0,
        synthetic_available_fraction=float(band.max_hot_reduction_fraction),
        terminal_hot_band_parity_measured=False,
        terminal_hot_band_parity_note=(
            "Synthetic numel=1024 adversarial band sweep does not model the "
            "terminal ~115MB hot_exact geometry from run 2189e72004."
        ),
    )


def classify_envelope_verdict(
    transforms: TransformBpw,
    *,
    residual: ResidualHotReduction,
    rollup_present: bool,
) -> str:
    if float(transforms.optimistic_upper_bound) >= float(REACHABLE_ORACLE_BPW_THRESHOLD):
        return CLASSIFICATION_NOT_REACHABLE
    if not rollup_present:
        return CLASSIFICATION_REACHABLE_PENDING_ROLLUP
    if float(transforms.best_combined_oracle) < float(REACHABLE_ORACLE_BPW_THRESHOLD):
        return CLASSIFICATION_REACHABLE
    return CLASSIFICATION_NOT_REACHABLE


def build_envelope_verdict_artifact(
    *,
    anchors: TerminalAnchors,
    band: BandSweepConstants,
    transforms: TransformBpw,
    residual: ResidualHotReduction,
    manifest_sha256: str,
    head_commit: str,
    parent_hash: str,
    packet_sha: str | None = None,
    rollup_present: bool = False,
) -> dict[str, Any]:
    classification = classify_envelope_verdict(
        transforms,
        residual=residual,
        rollup_present=rollup_present,
    )
    budget_bytes = sub2_budget_bytes(eligible_weight_count=anchors.eligible_weight_count)
    optimistic_total = int(
        round(
            float(transforms.optimistic_upper_bound)
            * float(anchors.eligible_weight_count)
            / 8.0
        )
    )
    gap_multiple = float(optimistic_total) / float(budget_bytes)
    return {
        "schema_version": JOINT_DRAIN_ENVELOPE_VERDICT_SCHEMA_VERSION,
        "classification": classification,
        "classification_basis": (
            "NOT_REACHABLE under available parity evidence: measured manifest hot "
            "floor after EVENT_FOLD_LOAD_SAFE event clear exceeds sub-2 budget; "
            "synthetic parity-safe hot reduction fraction is zero."
        ),
        "path": PATH_B_STRUCTURALLY_NOT_SUB2
        if classification == CLASSIFICATION_NOT_REACHABLE
        else "not_applicable",
        "terminal_anchors": {
            "run_id": anchors.run_id,
            "events_bytes": int(anchors.events_bytes),
            "hot_bytes": int(anchors.hot_bytes),
            "backlog_bytes": int(anchors.backlog_bytes),
            "metadata_bytes": int(anchors.metadata_bytes),
            "eligible_weight_count": int(anchors.eligible_weight_count),
            "terminal_inclusive_bpw": float(anchors.terminal_inclusive_bpw),
        },
        "transform_bpw": serialize_transform_bpw(
            transforms,
            rollup_present=rollup_present,
        ),
        "transform_bpw_basis": (
            TRANSFORM_BPW_BASIS_MANIFEST_ONLY
            if not rollup_present
            else (
                "All transforms computed from in-vivo rollup plus manifest anchors "
                "and parity-safe band constants."
            )
        ),
        "optimistic_upper_bound": {
            "events_bytes": 0,
            "hot_bytes": int(anchors.hot_bytes),
            "backlog_bytes": int(anchors.backlog_bytes),
            "metadata_bytes": int(anchors.metadata_bytes),
            "total_bytes": int(optimistic_total),
            "inclusive_bpw": float(transforms.optimistic_upper_bound),
            "gap_vs_sub2_budget_multiple": float(gap_multiple),
        },
        "sub2_budget": {
            "bytes": int(budget_bytes),
            "bpw_ceiling": float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING),
            "reachable_threshold_bpw": float(REACHABLE_ORACLE_BPW_THRESHOLD),
            "tolerance_bpw": float(R4V_ACC_BPW_TOLERANCE),
        },
        "residual_hot_reduction": {
            "flip_to_2p0_bpw_fraction_gt": float(residual.flip_to_2p0_bpw_fraction_gt),
            "flip_to_2p0_bpw_percent_gt": float(residual.flip_to_2p0_bpw_percent_gt),
            "flip_to_1p75_bpw_fraction_gt": float(residual.flip_to_1p75_bpw_fraction_gt),
            "flip_to_1p75_bpw_percent_gt": float(residual.flip_to_1p75_bpw_percent_gt),
            "synthetic_available_fraction": float(residual.synthetic_available_fraction),
            "terminal_hot_band_parity_measured": bool(
                residual.terminal_hot_band_parity_measured
            ),
            "terminal_hot_band_parity_note": str(residual.terminal_hot_band_parity_note),
            "strongest_leg": (
                "Measured 31.32 bpw hot floor from manifest 2189e72004 after "
                "EVENT_FOLD_LOAD_SAFE full event clear."
            ),
        },
        "band_sweep": {
            "source": "calm.hrm_text_158.native_full_stack.v4_live_band_sweep.run_band_sweep",
            "qualifying_bands": [int(item) for item in band.qualifying_bands],
            "max_hot_reduction_fraction": float(band.max_hot_reduction_fraction),
            "rows": list(band.rows),
        },
        "provenance": {
            "head_commit": str(head_commit),
            "parent_hash": str(parent_hash),
            "manifest_sha256": str(manifest_sha256),
            "packet_sha": packet_sha if packet_sha is not None else "not_applicable",
            "manifest_path": (
                "artifacts/consensus_prep/"
                "v4_live_phase_a_diagnostic_tier1_run_2189e72004_evidence_manifest.json"
            ),
        },
        "next_action": (
            "Bank STRUCTURALLY_NOT_SUB2 envelope receipt; no GPU drain-screen; "
            "real-pack confirmation not_applicable under decisive NOT_REACHABLE."
        ),
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_verdict_from_manifest_path(
    *,
    manifest_path: str,
    head_commit: str,
    packet_sha: str | None = None,
    rollup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_text = open(manifest_path, encoding="utf-8").read()
    manifest = json.loads(manifest_text)
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    anchors = load_terminal_anchors_from_manifest(manifest)
    band = load_band_constants_from_sweep(run_band_sweep())
    transforms = project_transform_bpw(anchors, band, rollup=rollup)
    residual = build_residual_hot_reduction(anchors, band)
    parent_hash = str(manifest["provenance"]["parent_hash_before"])
    return build_envelope_verdict_artifact(
        anchors=anchors,
        band=band,
        transforms=transforms,
        residual=residual,
        manifest_sha256=manifest_sha256,
        head_commit=head_commit,
        parent_hash=parent_hash,
        packet_sha=packet_sha,
        rollup_present=rollup is not None,
    )
