"""CPU synthetic shadow-policy contract for the B7b accumulator screen.

This is a measurement-contract harness only. It does not run training, mutate a
model, launch GPU work, or claim full-sub2 runtime readiness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    FIXTURE_CURRENT_REPO,
    fixture_full_sub2_runtime_ready_for_science,
)


ACCUMULATOR_POLICY_SHADOW_SCREEN_SCHEMA_VERSION = (
    "hrm_text_158_accumulator_policy_shadow_screen/v0.cpu_synthetic_contract"
)
B7B_SCREEN_DIAGNOSTIC_CONTRACT_ID = (
    "measurement_only_pre_full_stack_diagnostic/b7b_screen_a/v0"
)
PRE_FULL_STACK_DIAGNOSTIC_ONLY = "pre_full_stack_diagnostic_only"

CLAIM_SUB2 = "sub2"
CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2 = "algorithmic_proxy_not_physical_sub2"
CLAIM_INT16_REFERENCE = "int16_reference"
CLAIM_TRANSIENT_FP_DEBT = "transient_fp_debt"
PERSISTENT_STATE_CLAIM_CLASSES = (
    CLAIM_SUB2,
    CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2,
    CLAIM_INT16_REFERENCE,
    CLAIM_TRANSIENT_FP_DEBT,
)

ARM_INT16_BASELINE = "int16_baseline"
ARM_ACCUMULATOR_ONLY = "accumulator_only"
ARM_TRANSIENT_RESOLVER_ONLY = "transient_resolver_only"
ARM_ACCUMULATOR_PLUS_TRANSIENT = "accumulator_plus_transient"
REQUIRED_SHADOW_ARMS = (
    ARM_INT16_BASELINE,
    ARM_ACCUMULATOR_ONLY,
    ARM_TRANSIENT_RESOLVER_ONLY,
    ARM_ACCUMULATOR_PLUS_TRANSIENT,
)

LABEL_ACCUMULATOR_TRACKS_INT16_POLICY = "accumulator_tracks_int16_policy"
LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH = "accumulator_improves_but_not_enough"
LABEL_TRANSIENT_CARRIES_SELECTION = "transient_carries_selection"
LABEL_ACCUMULATOR_NO_TRACKING_NULL = "accumulator_no_tracking_null"
LABEL_SCREEN_HARNESS_OR_GATE_FAIL = "screen_harness_or_gate_fail"
ACCUMULATOR_POLICY_TAXONOMY_LABELS = (
    LABEL_ACCUMULATOR_TRACKS_INT16_POLICY,
    LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH,
    LABEL_TRANSIENT_CARRIES_SELECTION,
    LABEL_ACCUMULATOR_NO_TRACKING_NULL,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
)

DEFAULT_PREREG_THRESHOLDS: dict[str, float | int | bool] = {
    "min_jaccard_vs_int16": 0.90,
    "min_regret_capture_vs_oracle": 0.90,
    "max_transient_only_advantage_allowed": 0.05,
    "min_steps_for_verdict": 50,
    "n20_liveness_only": True,
}
REQUIRED_THRESHOLD_FIELDS = tuple(DEFAULT_PREREG_THRESHOLDS)


@dataclass(frozen=True)
class ShadowArmLedger:
    persistent_state_claim_class: str
    fp_transient_used_for_update: bool
    fp_transient_used_for_selection: bool
    selection_state_source: str
    selection_reads_decoded_int16: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_state_claim_class": self.persistent_state_claim_class,
            "fp_transient_used_for_update": bool(self.fp_transient_used_for_update),
            "fp_transient_used_for_selection": bool(
                self.fp_transient_used_for_selection
            ),
            "selection_state_source": self.selection_state_source,
            "selection_reads_decoded_int16": bool(self.selection_reads_decoded_int16),
        }


DEFAULT_ARM_LEDGERS: dict[str, ShadowArmLedger] = {
    ARM_INT16_BASELINE: ShadowArmLedger(
        persistent_state_claim_class=CLAIM_INT16_REFERENCE,
        fp_transient_used_for_update=False,
        fp_transient_used_for_selection=False,
        selection_state_source="int16_reference_accumulator",
        selection_reads_decoded_int16=True,
    ),
    ARM_ACCUMULATOR_ONLY: ShadowArmLedger(
        persistent_state_claim_class=CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2,
        fp_transient_used_for_update=True,
        fp_transient_used_for_selection=False,
        selection_state_source="bounded_accumulator_proxy",
        selection_reads_decoded_int16=False,
    ),
    ARM_TRANSIENT_RESOLVER_ONLY: ShadowArmLedger(
        persistent_state_claim_class=CLAIM_TRANSIENT_FP_DEBT,
        fp_transient_used_for_update=False,
        fp_transient_used_for_selection=True,
        selection_state_source="transient_fp_resolver",
        selection_reads_decoded_int16=False,
    ),
    ARM_ACCUMULATOR_PLUS_TRANSIENT: ShadowArmLedger(
        persistent_state_claim_class=CLAIM_TRANSIENT_FP_DEBT,
        fp_transient_used_for_update=True,
        fp_transient_used_for_selection=True,
        selection_state_source="bounded_accumulator_proxy_plus_transient_fp",
        selection_reads_decoded_int16=False,
    ),
}


def _stable_hash16(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_synthetic_shadow_candidate_stream(
    *,
    steps: int = 50,
    candidates_per_step: int = 4,
    mode: str = "accumulator_tracks",
) -> tuple[dict[str, Any], ...]:
    """Build a deterministic candidate stream for CPU contract tests."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if candidates_per_step < 2:
        raise ValueError("candidates_per_step must be at least 2")
    valid_modes = {
        "accumulator_tracks",
        "accumulator_improves",
        "transient_carries",
        "accumulator_null",
    }
    if mode not in valid_modes:
        raise ValueError(f"unknown synthetic stream mode {mode!r}; valid={sorted(valid_modes)}")

    stream: list[dict[str, Any]] = []
    for step in range(steps):
        oracle_idx = step % candidates_per_step
        decoy_idx = (oracle_idx + 1) % candidates_per_step
        candidates: list[dict[str, Any]] = []
        for idx in range(candidates_per_step):
            oracle_gain = 1.0 if idx == oracle_idx else 0.45 - 0.05 * idx
            if idx == decoy_idx:
                oracle_gain = 0.62
            int16_score = oracle_gain + (0.001 * (candidates_per_step - idx))
            if mode == "accumulator_tracks":
                accumulator_score = int16_score
                transient_score = 0.95 if idx == decoy_idx else 0.20 + 0.01 * idx
            elif mode == "accumulator_improves":
                accumulator_score = 0.80 if idx == oracle_idx else 0.82 if idx == decoy_idx else 0.10
                transient_score = 0.20 + 0.01 * idx
            elif mode == "transient_carries":
                accumulator_score = 0.90 if idx == decoy_idx else 0.10 + 0.01 * idx
                transient_score = int16_score
            else:
                accumulator_score = 1.0 if idx == 0 else 0.05 * idx
                transient_score = 0.30 + 0.01 * idx
            candidates.append(
                {
                    "candidate_id": f"s{step:04d}:c{idx}",
                    "candidate_index": idx,
                    "oracle_gain": round(float(oracle_gain), 6),
                    "int16_score": round(float(int16_score), 6),
                    "accumulator_score": round(float(accumulator_score), 6),
                    "transient_score": round(float(transient_score), 6),
                }
            )
        stream.append({"step": step, "candidates": tuple(candidates)})
    return tuple(stream)


def _current_repo_readiness_summary() -> dict[str, Any]:
    readiness = fixture_full_sub2_runtime_ready_for_science(FIXTURE_CURRENT_REPO)
    payload = readiness.to_dict()
    return {
        "fixture_name": FIXTURE_CURRENT_REPO,
        "schema_version": payload["schema_version"],
        "target_name": payload["target_name"],
        "ready_for_main_science": bool(payload["ready_for_main_science"]),
        "ready_for_pre_full_stack_diagnostic": bool(
            payload["ready_for_pre_full_stack_diagnostic"]
        ),
        "main_science_launch_blocked": bool(payload["main_science_launch_blocked"]),
        "counts_by_class": dict(payload["counts_by_class"]),
        "blocker_surface_names": list(payload["blocker_surface_names"]),
        "surface_names_by_class": dict(payload["surface_names_by_class"]),
    }


def _merge_ledgers(
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, ShadowArmLedger]:
    ledgers = dict(DEFAULT_ARM_LEDGERS)
    for arm, override in dict(overrides or {}).items():
        base = ledgers.get(arm)
        if base is None:
            raise ValueError(f"unknown shadow arm override {arm!r}")
        fields = base.to_dict()
        fields.update(dict(override))
        ledgers[arm] = ShadowArmLedger(
            persistent_state_claim_class=str(fields["persistent_state_claim_class"]),
            fp_transient_used_for_update=bool(fields["fp_transient_used_for_update"]),
            fp_transient_used_for_selection=bool(
                fields["fp_transient_used_for_selection"]
            ),
            selection_state_source=str(fields["selection_state_source"]),
            selection_reads_decoded_int16=bool(fields["selection_reads_decoded_int16"]),
        )
    return ledgers


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    rate_cap: int,
) -> tuple[str, ...]:
    if arm == ARM_INT16_BASELINE:
        score_field = "int16_score"
    elif arm == ARM_ACCUMULATOR_ONLY:
        score_field = "accumulator_score"
    elif arm == ARM_TRANSIENT_RESOLVER_ONLY:
        score_field = "transient_score"
    elif arm == ARM_ACCUMULATOR_PLUS_TRANSIENT:
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row["accumulator_score"]) + float(row["transient_score"]),
                str(row["candidate_id"]),
            ),
            reverse=True,
        )
        return tuple(str(row["candidate_id"]) for row in ranked[:rate_cap])
    else:
        raise ValueError(f"unknown shadow arm {arm!r}")
    ranked = sorted(
        candidates,
        key=lambda row: (float(row[score_field]), str(row["candidate_id"])),
        reverse=True,
    )
    return tuple(str(row["candidate_id"]) for row in ranked[:rate_cap])


def _arm_metrics(
    *,
    stream: Sequence[Mapping[str, Any]],
    selected_by_step: Sequence[tuple[str, ...]],
    int16_by_step: Sequence[tuple[str, ...]],
    rate_cap: int,
) -> dict[str, Any]:
    id_to_gain_by_step: list[dict[str, float]] = []
    oracle_top_by_step: list[str] = []
    for step in stream:
        gains = {
            str(row["candidate_id"]): float(row["oracle_gain"])
            for row in step["candidates"]
        }
        id_to_gain_by_step.append(gains)
        oracle_top_by_step.append(max(gains.items(), key=lambda item: item[1])[0])

    jaccards: list[float] = []
    regret_capture: list[float] = []
    oracle_top_hit: list[float] = []
    for idx, selected in enumerate(selected_by_step):
        selected_set = set(selected)
        int16_set = set(int16_by_step[idx])
        union = selected_set | int16_set
        jaccards.append(len(selected_set & int16_set) / len(union) if union else 1.0)
        gains = id_to_gain_by_step[idx]
        oracle_best = max(gains.values())
        selected_gain = sum(gains[candidate_id] for candidate_id in selected_set)
        regret_capture.append(
            min(1.0, selected_gain / max(oracle_best * rate_cap, 1e-12))
        )
        oracle_top_hit.append(1.0 if oracle_top_by_step[idx] in selected_set else 0.0)

    return {
        "jaccard_vs_int16": sum(jaccards) / len(jaccards),
        "regret_capture_vs_oracle": sum(regret_capture) / len(regret_capture),
        "oracle_top1_in_applied_set_rate": sum(oracle_top_hit) / len(oracle_top_hit),
        "applied_set_count": len(selected_by_step),
    }


def _state_hash_payload(
    arm: str,
    selected_by_step: Sequence[tuple[str, ...]],
    ledger: ShadowArmLedger,
) -> str:
    return _stable_hash16(
        {
            "arm": arm,
            "selected_by_step": list(selected_by_step),
            "ledger": ledger.to_dict(),
        }
    )


def _classify(
    *,
    contract_satisfied: bool,
    steps: int,
    thresholds: Mapping[str, Any],
    metrics_by_arm: Mapping[str, Mapping[str, Any]],
) -> str:
    if not contract_satisfied:
        return LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    min_steps = int(thresholds["min_steps_for_verdict"])
    min_jaccard = float(thresholds["min_jaccard_vs_int16"])
    min_regret = float(thresholds["min_regret_capture_vs_oracle"])
    max_transient_advantage = float(thresholds["max_transient_only_advantage_allowed"])
    accumulator = metrics_by_arm[ARM_ACCUMULATOR_ONLY]
    transient = metrics_by_arm[ARM_TRANSIENT_RESOLVER_ONLY]
    accumulator_jaccard = float(accumulator["jaccard_vs_int16"])
    accumulator_regret = float(accumulator["regret_capture_vs_oracle"])
    transient_advantage = (
        float(transient["regret_capture_vs_oracle"]) - accumulator_regret
    )
    if steps < min_steps:
        return (
            LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH
            if accumulator_regret > 0.0
            else LABEL_ACCUMULATOR_NO_TRACKING_NULL
        )
    if (
        accumulator_jaccard >= min_jaccard
        and accumulator_regret >= min_regret
        and transient_advantage <= max_transient_advantage
    ):
        return LABEL_ACCUMULATOR_TRACKS_INT16_POLICY
    if (
        float(transient["jaccard_vs_int16"]) >= min_jaccard
        and float(transient["regret_capture_vs_oracle"]) >= min_regret
        and transient_advantage > max_transient_advantage
    ):
        return LABEL_TRANSIENT_CARRIES_SELECTION
    if accumulator_regret >= 0.50 or accumulator_jaccard >= 0.50:
        return LABEL_ACCUMULATOR_IMPROVES_BUT_NOT_ENOUGH
    return LABEL_ACCUMULATOR_NO_TRACKING_NULL


def _diagnostic_contract_state(
    *,
    receipt: Mapping[str, Any],
    ledgers: Mapping[str, ShadowArmLedger],
    candidate_stream_hashes_by_arm: Mapping[str, str],
) -> dict[str, Any]:
    thresholds = dict(receipt["thresholds"])
    arms = dict(receipt["arms"])
    readiness = dict(receipt["readiness_current_repo"])
    failure_reasons: list[str] = []
    checks: dict[str, bool] = {}

    checks["no_q_mutation"] = not bool(receipt["q_mutation_applied_to_model"])
    checks["same_candidate_stream"] = (
        len(set(candidate_stream_hashes_by_arm.values())) == 1
    )
    checks["compact_outputs"] = bool(receipt["compact_receipt"])
    checks["update_selection_ledger"] = all(
        field in arms[arm]
        for arm in REQUIRED_SHADOW_ARMS
        for field in (
            "fp_transient_used_for_update",
            "fp_transient_used_for_selection",
            "selection_state_source",
        )
        if arm in arms
    )
    checks["readiness_embedded"] = {
        "ready_for_main_science",
        "ready_for_pre_full_stack_diagnostic",
        "main_science_launch_blocked",
        "counts_by_class",
        "blocker_surface_names",
    }.issubset(readiness)
    checks["current_repo_fail_closed"] = (
        checks["readiness_embedded"]
        and readiness["ready_for_main_science"] is False
        and readiness["ready_for_pre_full_stack_diagnostic"] is False
        and readiness["main_science_launch_blocked"] is True
    )
    checks["no_runtime_claims"] = (
        receipt["pre_full_stack_diagnostic_only"] is True
        and receipt["runtime_readiness_claim"] is False
        and receipt["training_or_acquisition_claim"] is False
    )
    checks["threshold_fields_present"] = all(
        field in thresholds for field in REQUIRED_THRESHOLD_FIELDS
    )
    checks["required_arms_present"] = set(REQUIRED_SHADOW_ARMS).issubset(arms)
    checks["claim_classes_valid"] = all(
        ledgers[arm].persistent_state_claim_class in PERSISTENT_STATE_CLAIM_CLASSES
        for arm in REQUIRED_SHADOW_ARMS
    )

    accumulator = ledgers[ARM_ACCUMULATOR_ONLY]
    checks["accumulator_physical_sub2_selection_clean"] = not (
        accumulator.persistent_state_claim_class == CLAIM_SUB2
        and (
            accumulator.fp_transient_used_for_selection
            or accumulator.selection_reads_decoded_int16
        )
    )
    checks["accumulator_proxy_selection_not_overclaimed"] = not (
        accumulator.selection_reads_decoded_int16
        and accumulator.persistent_state_claim_class
        != CLAIM_ALGORITHMIC_PROXY_NOT_PHYSICAL_SUB2
    )

    for check_name, passed in checks.items():
        if not passed:
            failure_reasons.append(check_name)
    return {
        "contract_id": B7B_SCREEN_DIAGNOSTIC_CONTRACT_ID,
        "satisfied": not failure_reasons,
        "failure_reasons": failure_reasons,
        "checks": checks,
    }


def run_accumulator_policy_shadow_screen(
    *,
    candidate_stream: Sequence[Mapping[str, Any]] | None = None,
    steps: int = 50,
    rate_cap: int = 1,
    thresholds: Mapping[str, Any] | None = None,
    arm_ledger_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_stream_hash_overrides: Mapping[str, str] | None = None,
    q_mutation_applied_to_model: bool = False,
    synthetic_mode: str = "accumulator_tracks",
) -> dict[str, Any]:
    """Run the CPU synthetic four-arm shadow-policy screen.

    The returned receipt is intentionally compact and diagnostic-only. Any
    contract violation leaves the measured payload intact but classifies the run
    as ``screen_harness_or_gate_fail``.
    """

    if rate_cap <= 0:
        raise ValueError("rate_cap must be positive")
    stream = tuple(
        candidate_stream
        if candidate_stream is not None
        else build_synthetic_shadow_candidate_stream(steps=steps, mode=synthetic_mode)
    )
    if not stream:
        raise ValueError("candidate_stream must not be empty")
    thresholds_payload = dict(DEFAULT_PREREG_THRESHOLDS)
    thresholds_payload.update(dict(thresholds or {}))
    ledgers = _merge_ledgers(arm_ledger_overrides)

    candidate_stream_hash = _stable_hash16(stream)
    candidate_stream_hashes_by_arm = {
        arm: candidate_stream_hash for arm in REQUIRED_SHADOW_ARMS
    }
    candidate_stream_hashes_by_arm.update(dict(candidate_stream_hash_overrides or {}))

    selections_by_arm: dict[str, tuple[tuple[str, ...], ...]] = {}
    for arm in REQUIRED_SHADOW_ARMS:
        selections_by_arm[arm] = tuple(
            _select_candidate(step["candidates"], arm=arm, rate_cap=rate_cap)
            for step in stream
        )

    int16_by_step = selections_by_arm[ARM_INT16_BASELINE]
    metrics_by_arm = {
        arm: _arm_metrics(
            stream=stream,
            selected_by_step=selected_by_step,
            int16_by_step=int16_by_step,
            rate_cap=rate_cap,
        )
        for arm, selected_by_step in selections_by_arm.items()
    }
    transient_only_advantage = (
        metrics_by_arm[ARM_TRANSIENT_RESOLVER_ONLY]["regret_capture_vs_oracle"]
        - metrics_by_arm[ARM_ACCUMULATOR_ONLY]["regret_capture_vs_oracle"]
    )

    arms_payload: dict[str, dict[str, Any]] = {}
    for arm, ledger in ledgers.items():
        selected_by_step = selections_by_arm[arm]
        arms_payload[arm] = {
            **ledger.to_dict(),
            "metrics": metrics_by_arm[arm],
            "selected_candidate_ids_hash16": _stable_hash16(selected_by_step),
            "arm_state_hash": _state_hash_payload(arm, selected_by_step, ledger),
        }

    receipt: dict[str, Any] = {
        "schema_version": ACCUMULATOR_POLICY_SHADOW_SCREEN_SCHEMA_VERSION,
        "contract_id": B7B_SCREEN_DIAGNOSTIC_CONTRACT_ID,
        "receipt_kind": "cpu_synthetic_shadow_policy_harness",
        "pre_full_stack_diagnostic_only": True,
        "runtime_readiness_claim": False,
        "training_or_acquisition_claim": False,
        "q_mutation_applied_to_model": bool(q_mutation_applied_to_model),
        "compact_receipt": True,
        "steps": len(stream),
        "rate_cap": int(rate_cap),
        "verdict_allowed": len(stream) >= int(thresholds_payload["min_steps_for_verdict"]),
        "thresholds": thresholds_payload,
        "readiness_current_repo": _current_repo_readiness_summary(),
        "candidate_stream_hash": candidate_stream_hash,
        "candidate_stream_hashes_by_arm": candidate_stream_hashes_by_arm,
        "divergent_arm_state_hashes_allowed": True,
        "arms": arms_payload,
        "aggregate_metrics": {
            "transient_only_advantage_vs_accumulator": transient_only_advantage,
            "candidate_stream_step_count": len(stream),
            "candidate_stream_candidate_count": sum(
                len(step["candidates"]) for step in stream
            ),
        },
    }
    diagnostic_contract = _diagnostic_contract_state(
        receipt=receipt,
        ledgers=ledgers,
        candidate_stream_hashes_by_arm=candidate_stream_hashes_by_arm,
    )
    primary_label = _classify(
        contract_satisfied=bool(diagnostic_contract["satisfied"]),
        steps=len(stream),
        thresholds=thresholds_payload,
        metrics_by_arm=metrics_by_arm,
    )
    receipt["diagnostic_contract"] = diagnostic_contract
    receipt["primary_label"] = primary_label
    receipt["taxonomy_labels"] = [PRE_FULL_STACK_DIAGNOSTIC_ONLY, primary_label]
    receipt["screen_harness_or_gate_fail"] = (
        primary_label == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    )
    receipt["failure_reasons"] = list(diagnostic_contract["failure_reasons"])
    return receipt
