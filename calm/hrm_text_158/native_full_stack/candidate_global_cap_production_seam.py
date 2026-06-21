"""B2-5c Step-1b-(1) candidate→global-cap production routing seam (CPU/read-only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import _tensor_sha256
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_receipt import (
    MagnitudeRegime,
    composition_path_exists,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_reference import (
    MaterializedBridgeArtifacts,
    build_vote_update_plan_from_bridge_artifacts,
    materialize_bridge_artifacts_from_candidate_result,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GlobalRateCapResult,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec, VoteUpdateState

PRODUCTION_SEAM_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-1b-(1) is CPU/read-only production-shaped routing seam only",
    "B2-5c Step-1b-(1) does NOT wire candidate_mode + global_cap_spec in the trainer loop",
    "B2-5c Step-1b-(1) does NOT un-raise bounded_delta_learner.py:1646-1647",
    "B2-5c Step-1b-(1) does NOT mint selection_parity_pass",
    "B2-5c Step-1b-(1) does NOT flip global_cap_margin_only_reference",
    "B2-5c Step-1b-(1) does NOT flip optimizer_credit_state / readiness rows",
    "Single-state regression vs Step-1a compose is faithful-wrap regression, not independent equivalence",
)

PRODUCTION_SEAM_HARD_FALSE_FIELDS: tuple[str, ...] = (
    "selection_parity_pass",
    "native_selector_wired",
    "readiness_flip_authorized",
    "global_cap_margin_only_reference_flipped",
    "optimizer_credit_state_sub2_claim",
    "wiring_authorized",
    "trainer_guard_unraised",
)


@dataclass(frozen=True)
class CandidateGlobalCapSeamEntry:
    prior_state: VoteUpdateState
    candidate_result: Any
    vote_spec: VoteUpdateSpec


@dataclass(frozen=True)
class CandidateGlobalCapProductionSeamResult:
    artifacts_by_key: dict[str, MaterializedBridgeArtifacts]
    cap_inputs: tuple[GlobalRateCapTensorInput, ...]
    cap_result: GlobalRateCapResult
    q_acc_by_key: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]
    summary: dict[str, Any]
    magnitude_regime_by_key: dict[str, MagnitudeRegime]
    fidelity_lattice_pass_by_key: dict[str, bool]
    prior_state_q_sha256_by_key: dict[str, str]
    prior_state_acc_sha256_by_key: dict[str, str]
    non_claims: tuple[str, ...]
    composition_path_exists: bool
    selection_parity_pass: bool = False
    native_selector_wired: bool = False
    readiness_flip_authorized: bool = False
    global_cap_margin_only_reference_flipped: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    wiring_authorized: bool = False
    trainer_guard_unraised: bool = False


def _magnitude_regime_for_applied(
    *,
    residual_after_threshold: Mapping[int, int],
    threshold: int,
) -> MagnitudeRegime:
    clip_edge = int(threshold) - 1
    for residual in residual_after_threshold.values():
        if abs(int(residual)) >= clip_edge:
            return "clip_boundary_reconciliation"
    return "no_clip_exact_add_back"


def _validate_seam_admission(
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
    tie_rule_mode: str,
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None,
    front_c_identity_observer: object | None,
    local_selection_ordering_mode: str | None,
) -> None:
    if deferred_backlog is not None:
        raise ValueError(
            "production seam admission rejects deferred backlog in candidate scope",
        )
    if tie_rule_mode != EXACT_GLOBAL_CAP_TIE_RULE_MODE:
        raise ValueError(
            "production seam admission requires exact_global_cap tie-rule scope",
        )
    if replay_ce_veto_votes_by_key is not None or replay_ce_veto_moves_by_key is not None:
        raise ValueError(
            "production seam admission rejects replay/pc auxiliary paths",
        )
    if pc_aux_votes_by_key is not None or pc_aux_moves_by_key is not None:
        raise ValueError(
            "production seam admission rejects replay/pc auxiliary paths",
        )
    if front_c_identity_observer is not None:
        raise ValueError(
            "production seam admission rejects front_c live identity observation",
        )
    if local_selection_ordering_mode is not None:
        raise ValueError(
            "production seam admission rejects alternate local ordering",
        )


def apply_candidate_global_cap_production_seam(
    entries: Mapping[str, CandidateGlobalCapSeamEntry],
    global_cap_spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None = None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None = None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None = None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None = None,
    front_c_identity_observer: object | None = None,
    local_selection_ordering_mode: str | None = None,
    contract_name: str | None = None,
) -> CandidateGlobalCapProductionSeamResult:
    if not entries:
        raise ValueError("production seam requires at least one entry")
    _validate_seam_admission(
        deferred_backlog=deferred_backlog,
        tie_rule_mode=tie_rule_mode,
        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
        pc_aux_votes_by_key=pc_aux_votes_by_key,
        pc_aux_moves_by_key=pc_aux_moves_by_key,
        front_c_identity_observer=front_c_identity_observer,
        local_selection_ordering_mode=local_selection_ordering_mode,
    )

    prior_state_q_sha256_by_key: dict[str, str] = {}
    prior_state_acc_sha256_by_key: dict[str, str] = {}
    artifacts_by_key: dict[str, MaterializedBridgeArtifacts] = {}
    magnitude_regime_by_key: dict[str, MagnitudeRegime] = {}
    fidelity_lattice_pass_by_key: dict[str, bool] = {}
    cap_inputs: list[GlobalRateCapTensorInput] = []

    for state_key in sorted(entries):
        entry = entries[state_key]
        prior_state_q_sha256_by_key[state_key] = _tensor_sha256(entry.prior_state.q_levels)
        prior_state_acc_sha256_by_key[state_key] = _tensor_sha256(
            entry.prior_state.accumulators,
        )
        artifacts = materialize_bridge_artifacts_from_candidate_result(
            state_key=state_key,
            prior_state=entry.prior_state,
            candidate_result=entry.candidate_result,
            vote_spec=entry.vote_spec,
        )
        bridge_plan = build_vote_update_plan_from_bridge_artifacts(
            prior_state=entry.prior_state,
            artifacts=artifacts,
        )
        artifacts_by_key[state_key] = artifacts
        fidelity_lattice_pass_by_key[state_key] = True
        magnitude_regime_by_key[state_key] = _magnitude_regime_for_applied(
            residual_after_threshold=artifacts.residual_after_threshold,
            threshold=artifacts.threshold,
        )
        cap_inputs.append(
            GlobalRateCapTensorInput(
                state_key=state_key,
                state=entry.prior_state,
                plan=bridge_plan,
                vote_inputs=None,
            ),
        )

    cap_result = apply_global_rate_cap_reference(
        cap_inputs,
        global_cap_spec,
        deferred_backlog=deferred_backlog,
        tie_rule_mode=tie_rule_mode,
        contract_name=contract_name,
    )
    q_acc_by_key = {
        item.state_key: (item.q_levels, item.accumulators, item.stats)
        for item in cap_result.tensor_results
    }
    summary = dict(cap_result.step_summary)
    summary["global_rate_cap_enabled"] = True
    summary["production_seam_state_keys"] = sorted(entries)
    summary["production_seam_cap_input_count"] = len(cap_inputs)
    summary["composition_path_exists"] = composition_path_exists()

    return CandidateGlobalCapProductionSeamResult(
        artifacts_by_key=artifacts_by_key,
        cap_inputs=tuple(cap_inputs),
        cap_result=cap_result,
        q_acc_by_key=q_acc_by_key,
        summary=summary,
        magnitude_regime_by_key=magnitude_regime_by_key,
        fidelity_lattice_pass_by_key=fidelity_lattice_pass_by_key,
        prior_state_q_sha256_by_key=prior_state_q_sha256_by_key,
        prior_state_acc_sha256_by_key=prior_state_acc_sha256_by_key,
        non_claims=PRODUCTION_SEAM_NON_CLAIMS,
        composition_path_exists=composition_path_exists(),
    )


def apply_candidate_global_cap_production_seam_single(
    state_key: str,
    entry: CandidateGlobalCapSeamEntry,
    global_cap_spec: GlobalRateCapSpec,
    **kwargs: Any,
) -> CandidateGlobalCapProductionSeamResult:
    return apply_candidate_global_cap_production_seam(
        {state_key: entry},
        global_cap_spec,
        **kwargs,
    )
