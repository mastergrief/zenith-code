"""Offline replay scorer for votes-emitting dynamics-proof runs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r4v_event_coded_acc_budget,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import crossing_bool_w6
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VOTES_EMIT_SCHEMA_VERSION,
    VOTES_EMIT_SECTION6_CONTRACT_FIELDS,
)

CLASSIFIER_INTRINSIC_WIDE_CONFIRMED = "INTRINSIC_WIDE_CONFIRMED"
CLASSIFIER_STATIC_PROXY_ARTIFACT = "STATIC_PROXY_ARTIFACT"
CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS = "REDUCIBLE_UNDER_DYNAMICS"
CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW = "MISSING_OBSERVABLES_OR_INVALID_WINDOW"
CLASSIFIER_RUN_HEALTH_FAIL = "RUN_HEALTH_FAIL"

REPLAY_MODE_R_STATIC = "r-static"
REPLAY_MODE_R_DYNAMICS = "r-dynamics"

ARM_MODE_REPLAY_ONLY = "replay-only"
ARM_MODE_LIVE_ALTERED_DYNAMICS = "live-altered-dynamics"

DYNAMICS_PROOF_REPLAY_SCHEMA_VERSION = "hrm_text_158_votes_emit_dynamics_replay/v0"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def section6_contract_field_names() -> tuple[str, ...]:
    return tuple(VOTES_EMIT_SECTION6_CONTRACT_FIELDS)


def missing_section6_fields(record: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in section6_contract_field_names():
        if field_name not in record:
            missing.append(field_name)
            continue
        value = record[field_name]
        if value is None:
            missing.append(field_name)
    return missing


def load_votes_emit_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = Path(run_root) / "votes_emit" / "v1" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"votes emit manifest missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_votes_emit_step_records(run_root: Path) -> list[dict[str, Any]]:
    per_step_dir = Path(run_root) / "votes_emit" / "v1" / "per_step"
    if not per_step_dir.is_dir():
        raise FileNotFoundError(f"votes emit per_step dir missing: {per_step_dir}")
    manifest = load_votes_emit_manifest(run_root)
    per_step_hashes = dict(manifest.get("per_step_hashes", {}))
    records: list[dict[str, Any]] = []
    for step_name in sorted(per_step_hashes):
        step_path = per_step_dir / f"{step_name}.json"
        if not step_path.is_file():
            raise FileNotFoundError(f"missing per-step record: {step_path}")
        payload = json.loads(step_path.read_text(encoding="utf-8"))
        canonical = _canonical_json(payload)
        expected_hash = str(per_step_hashes[step_name])
        actual_hash = _sha256_text(canonical)
        if actual_hash != expected_hash:
            raise ValueError(
                f"manifest hash mismatch for step {step_name}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        records.append(payload)
    return records


def _threshold_abs_from_record(record: Mapping[str, Any]) -> int:
    threshold_semantics = record.get("threshold_semantics")
    if not isinstance(threshold_semantics, Mapping):
        raise ValueError("threshold_semantics block missing")
    return int(threshold_semantics["crossing_threshold_abs"])


def reconstruct_sampled_crossing_mask_hash(record: Mapping[str, Any]) -> str:
    """Spot-verification hash from the 32-row sample only (non-authoritative)."""

    threshold_abs = _threshold_abs_from_record(record)
    rows: list[dict[str, Any]] = []
    for row in record.get("sampled_candidate_table", []):
        pre_acc = int(row["pre_accumulator_i16"])
        vote_value = int(row["vote_value"])
        q_level = int(row["current_q_level"])
        new_acc = int(pre_acc) + int(vote_value)
        rows.append(
            {
                "state_key": str(row["state_key"]),
                "flat_index": int(row["flat_index"]),
                "crossing_bool_w6": bool(
                    crossing_bool_w6(new_acc, q_level, threshold_abs=threshold_abs)
                ),
            }
        )
    rows.sort(
        key=lambda item: (str(item["state_key"]), int(item["flat_index"]))
    )
    return _sha256_text(_canonical_json(rows))


def verify_section6_internal_consistency(record: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    cap_summary = record.get("cap_order_summary")
    if not isinstance(cap_summary, Mapping):
        failures.append("cap_order_summary_not_mapping")
        return failures
    accepted_hash = str(cap_summary.get("accepted_flat_indices_hash", ""))
    applied_hash = str(record.get("applied_flat_indices_hash", ""))
    if accepted_hash != applied_hash:
        failures.append("applied_flat_indices_hash_mismatch_vs_cap_order_summary")
    if not str(record.get("pre_update_state_hash", "")):
        failures.append("pre_update_state_hash_empty")
    return failures


@dataclass(frozen=True)
class DynamicsReplayScoreInputs:
    replay_mode: str
    arm_id: str
    arm_mode: str
    from_clean_contiguous: bool
    run_health_ok: bool
    section6_complete: bool
    section6_consistent: bool
    live_evidence: bool
    r4v_ledger_pass: bool | None
    static_proxy_gap_falsified: bool
    all_live_variants_failed_sub2: bool


def classify_dynamics_proof_verdict(inputs: DynamicsReplayScoreInputs) -> str:
    if not inputs.run_health_ok:
        return CLASSIFIER_RUN_HEALTH_FAIL
    if not inputs.section6_complete or not inputs.section6_consistent:
        return CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW
    if inputs.replay_mode in {REPLAY_MODE_R_STATIC, REPLAY_MODE_R_DYNAMICS}:
        if inputs.arm_mode == ARM_MODE_REPLAY_ONLY:
            return CLASSIFIER_STATIC_PROXY_ARTIFACT
    if (
        inputs.arm_mode == ARM_MODE_LIVE_ALTERED_DYNAMICS
        and inputs.arm_id == "V4"
        and inputs.live_evidence
        and inputs.r4v_ledger_pass is True
    ):
        return CLASSIFIER_REDUCIBLE_UNDER_DYNAMICS
    if (
        inputs.arm_mode == ARM_MODE_LIVE_ALTERED_DYNAMICS
        and inputs.from_clean_contiguous
        and inputs.live_evidence
        and inputs.all_live_variants_failed_sub2
    ):
        return CLASSIFIER_INTRINSIC_WIDE_CONFIRMED
    if inputs.arm_mode == ARM_MODE_LIVE_ALTERED_DYNAMICS and inputs.live_evidence:
        return CLASSIFIER_STATIC_PROXY_ARTIFACT
    return CLASSIFIER_STATIC_PROXY_ARTIFACT


def score_votes_emit_replay_run(
    run_root: Path,
    *,
    replay_mode: str,
    arm_id: str,
    arm_mode: str,
    from_clean_contiguous: bool,
    live_evidence: bool,
    run_health_ok: bool = True,
    r4v_ledger_pass: bool | None = None,
    static_proxy_gap_falsified: bool = False,
    all_live_variants_failed_sub2: bool = False,
    event_payloads: Sequence[Any] | None = None,
    qscale_states: Sequence[Any] | None = None,
    state_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    records = load_votes_emit_step_records(run_root)
    if not records:
        return {
            "schema_version": DYNAMICS_PROOF_REPLAY_SCHEMA_VERSION,
            "classifier_verdict": CLASSIFIER_MISSING_OBSERVABLES_OR_INVALID_WINDOW,
            "step_count": 0,
            "section6_complete": False,
            "section6_consistent": False,
            "replay_mode": str(replay_mode),
            "arm_id": str(arm_id),
            "arm_mode": str(arm_mode),
        }

    missing_fields: list[str] = []
    consistency_failures: list[str] = []
    sampled_crossing_hashes: list[str] = []
    for record in records:
        missing_fields.extend(missing_section6_fields(record))
        consistency_failures.extend(verify_section6_internal_consistency(record))
        sampled_crossing_hashes.append(reconstruct_sampled_crossing_mask_hash(record))

    section6_complete = not missing_fields
    section6_consistent = not consistency_failures

    measured_r4v_pass: bool | None = r4v_ledger_pass
    if event_payloads is not None and qscale_states is not None:
        report = measure_r4v_event_coded_acc_budget(
            qscale_states,
            event_payloads,
            state_keys=state_keys,
        )
        measured_r4v_pass = bool(report.r4v_ledger_pass)

    verdict = classify_dynamics_proof_verdict(
        DynamicsReplayScoreInputs(
            replay_mode=str(replay_mode),
            arm_id=str(arm_id),
            arm_mode=str(arm_mode),
            from_clean_contiguous=bool(from_clean_contiguous),
            run_health_ok=bool(run_health_ok),
            section6_complete=bool(section6_complete),
            section6_consistent=bool(section6_consistent),
            live_evidence=bool(live_evidence),
            r4v_ledger_pass=measured_r4v_pass,
            static_proxy_gap_falsified=bool(static_proxy_gap_falsified),
            all_live_variants_failed_sub2=bool(all_live_variants_failed_sub2),
        )
    )

    return {
        "schema_version": DYNAMICS_PROOF_REPLAY_SCHEMA_VERSION,
        "votes_emit_schema_version": VOTES_EMIT_SCHEMA_VERSION,
        "classifier_verdict": verdict,
        "step_count": int(len(records)),
        "section6_complete": bool(section6_complete),
        "section6_consistent": bool(section6_consistent),
        "missing_section6_fields": sorted(set(missing_fields)),
        "section6_consistency_failures": sorted(set(consistency_failures)),
        "sampled_crossing_reconstruction_hashes": sampled_crossing_hashes,
        "replay_mode": str(replay_mode),
        "replay_mode_label": (
            "R-static"
            if replay_mode == REPLAY_MODE_R_STATIC
            else "R-dynamics"
            if replay_mode == REPLAY_MODE_R_DYNAMICS
            else str(replay_mode)
        ),
        "arm_id": str(arm_id),
        "arm_mode": str(arm_mode),
        "from_clean_contiguous": bool(from_clean_contiguous),
        "live_evidence": bool(live_evidence),
        "r4v_ledger_pass": measured_r4v_pass,
        "reducible_or_intrinsic_from_replay_only": False,
    }
