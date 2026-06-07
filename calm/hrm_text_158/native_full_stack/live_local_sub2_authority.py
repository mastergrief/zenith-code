"""2B0 local-only live authority receipt.

This receipt formalizes the executable bounded local vote-update seam without
claiming trainer integration or broad runtime readiness row flips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_sidecar_runtime import (
    HybridSidecarPersistentStateReport,
    hybrid_sidecar_persistent_state_report,
    make_applied_crossing_direction_residual_persistent_state,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


LIVE_LOCAL_SUB2_AUTHORITY_SCHEMA_VERSION = (
    "hrm_text_158_2b0_live_local_sub2_authority/v0.local_only_receipt"
)
LIVE_LOCAL_SUB2_AUTHORITY_TARGET_NAME = "step2b0_live_local_sub2_authority"
LIVE_LOCAL_SUB2_AUTHORITY_ENTRYPOINT = (
    "bounded_delta_learner.apply_bounded_delta_vote_step"
    "(candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE)"
)
LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS = (
    "global_cap",
    "replay_ce_veto",
    "pc_aux",
    "global_backlog",
    "trainer_integration",
    "checkpoint_load_save_in_trainer",
)
LIVE_LOCAL_SUB2_AUTHORITY_NON_CLAIMS = (
    "2B0 receipt covers local bounded-authority seam only; not broad Step-2B completion",
    "live_runtime_authority_converted=false for trainer-used path",
    "trainer_entrypoint_uses_candidate=false",
    "readiness_row_flip_authorized=false; FIXTURE_CURRENT_REPO q/acc rows must not flip",
    "global_cap/replay/PC/backlog/trainer integration are blockers, not covered parity surfaces",
    "not learning, acquisition, throughput, GPU residency, training launch, or .pt mutation",
)


@dataclass(frozen=True)
class LiveLocalSub2AuthorityReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    entrypoint: str
    local_authority_seam_executable: bool
    exact_local_parity_pass: bool
    local_persistent_core_sub2: bool
    no_dense_int16_counted_authority_local: bool
    dense_oracle_control_used_for_comparison: bool
    oracle_control_authority: str
    production_authority_claim_authorized: bool
    live_runtime_authority_converted: bool
    trainer_entrypoint_uses_candidate: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    current_repo_readiness_rows_may_flip: bool
    physical_persistent_bits_per_weight: float
    effective_persistent_bits_per_weight: float
    target_bits_per_weight: float
    declared_coverage_domain: dict[str, Any]
    uncovered_blockers: tuple[str, ...]
    sidecar_report: HybridSidecarPersistentStateReport
    local_step_summary: dict[str, Any]
    proof_by_key: dict[str, dict[str, Any]]
    proof_anchors: tuple[str, ...]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_row_flip_authorized_surface_names"] = list(
            self.readiness_row_flip_authorized_surface_names
        )
        payload["uncovered_blockers"] = list(self.uncovered_blockers)
        payload["sidecar_report"] = self.sidecar_report.to_dict()
        payload["proof_anchors"] = list(self.proof_anchors)
        payload["non_claims"] = list(self.non_claims)
        return payload


def _default_fixture() -> tuple[
    dict[str, BoundedDeltaTensorState],
    dict[str, torch.Tensor],
    dict[str, VoteUpdateSpec],
    dict[str, dict[int, int]],
]:
    state_key = "step2b0.local.proj"
    q_levels = torch.zeros((128, 128), dtype=torch.int8)
    accumulators = torch.zeros_like(q_levels, dtype=torch.int16)
    flat = accumulators.flatten()
    flat[3] = 9
    flat[197] = -9
    votes = torch.zeros_like(q_levels, dtype=torch.int16)
    votes.flatten()[3] = 2
    votes.flatten()[197] = -2
    state = make_bounded_tensor_state(
        state_key,
        q_levels,
        1.0,
        accumulators,
        hot_exact_indices=(3, 197),
        cold_default_value=0,
    )
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )
    return (
        {state_key: state},
        {state_key: votes},
        {state_key: spec},
        {state_key: {3: 2, 197: -2}},
    )


def _sidecar_report_from_local_states(
    *,
    prior_states: Mapping[str, BoundedDeltaTensorState],
    next_states: Mapping[str, BoundedDeltaTensorState],
) -> HybridSidecarPersistentStateReport:
    sidecar_states = {}
    for key, next_state in sorted(next_states.items()):
        prior = prior_states[key]
        before = prior.q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
        after = next_state.q_levels.detach().cpu().to(torch.int8).flatten().contiguous()
        decoded = next_state.decoded_accumulators().flatten().contiguous()
        changed_indices = []
        changed_directions = []
        residual_values = []
        for index, (before_value, after_value) in enumerate(zip(before.tolist(), after.tolist())):
            delta = int(after_value) - int(before_value)
            if delta == 0:
                continue
            changed_indices.append(int(index))
            changed_directions.append(1 if delta > 0 else -1)
            residual_values.append(int(decoded[int(index)].item()))
        sidecar_states[key] = make_applied_crossing_direction_residual_persistent_state(
            key,
            next_state.q_levels,
            next_state.frozen_scale,
            applied_indices=tuple(changed_indices),
            applied_directions=tuple(changed_directions),
            residual_values=tuple(residual_values),
        )
    return hybrid_sidecar_persistent_state_report(sidecar_states)


def _local_exact_parity_pass(proof_by_key: Mapping[str, Mapping[str, Any]]) -> bool:
    for proof in proof_by_key.values():
        if not bool(proof.get("pass")) or not bool(proof.get("parity_pass")):
            return False
        if proof.get("candidate_q_sha256_after") != proof.get("oracle_q_sha256_after"):
            return False
        if proof.get("candidate_bounded_decode_sha256_after") != proof.get("oracle_acc_sha256_after"):
            return False
        if proof.get("applied_row_identities_sha256") != proof.get("oracle_applied_row_identities_sha256"):
            return False
        if proof.get("residual_after_threshold_sha256") != proof.get("oracle_residual_after_threshold_sha256"):
            return False
    return bool(proof_by_key)


def build_live_local_sub2_authority_receipt(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState] | None = None,
    votes_by_key: Mapping[str, torch.Tensor] | None = None,
    vote_specs_by_key: Mapping[str, VoteUpdateSpec] | None = None,
    candidate_sparse_vote_events_by_key: Mapping[str, Mapping[int, int]] | None = None,
) -> LiveLocalSub2AuthorityReceipt:
    """Build the local-only 2B0 receipt without changing trainer authority."""

    if (
        tensor_states is None
        or votes_by_key is None
        or vote_specs_by_key is None
        or candidate_sparse_vote_events_by_key is None
    ):
        (
            default_states,
            default_votes,
            default_specs,
            default_sparse_events,
        ) = _default_fixture()
        tensor_states = default_states if tensor_states is None else tensor_states
        votes_by_key = default_votes if votes_by_key is None else votes_by_key
        vote_specs_by_key = default_specs if vote_specs_by_key is None else vote_specs_by_key
        candidate_sparse_vote_events_by_key = (
            default_sparse_events
            if candidate_sparse_vote_events_by_key is None
            else candidate_sparse_vote_events_by_key
        )

    step = apply_bounded_delta_vote_step(
        tensor_states,
        votes_by_key,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=candidate_sparse_vote_events_by_key,
        candidate_oracle_control_enabled=True,
    )
    proof_by_key = {
        str(key): dict(value)
        for key, value in dict(step.global_summary["candidate_local_update_proof_by_key"]).items()
    }
    sidecar_report = _sidecar_report_from_local_states(
        prior_states=tensor_states,
        next_states=step.tensor_states,
    )
    coverage_domains = {
        str(key): dict(proof.get("coverage_domain") or {})
        for key, proof in proof_by_key.items()
    }
    coverage_domain = next(iter(coverage_domains.values())) if len(coverage_domains) == 1 else {
        "per_key": coverage_domains
    }
    ledger = dict(sidecar_report.movement_overlay.persistent_sidecar_ledger)
    inclusive_bpw = float(ledger.get("inclusive_bits_per_weight", 0.0))
    exact_parity_pass = _local_exact_parity_pass(proof_by_key)
    no_dense_authority = bool(
        not sidecar_report.persistent_dense_shadow_present
        and int(sidecar_report.persistent_dense_shadow_bytes) == 0
        and all(state.exact_accumulator_shadow is None for state in step.tensor_states.values())
    )
    local_sub2 = bool(sidecar_report.pass_report and sidecar_report.budget_guard.pass_guard and inclusive_bpw < 2.0)
    pass_receipt = bool(
        step.global_summary.get("candidate_local_update_pass")
        and exact_parity_pass
        and local_sub2
        and no_dense_authority
    )
    receipt = LiveLocalSub2AuthorityReceipt(
        schema_version=LIVE_LOCAL_SUB2_AUTHORITY_SCHEMA_VERSION,
        target_name=LIVE_LOCAL_SUB2_AUTHORITY_TARGET_NAME,
        pass_receipt=pass_receipt,
        entrypoint=LIVE_LOCAL_SUB2_AUTHORITY_ENTRYPOINT,
        local_authority_seam_executable=True,
        exact_local_parity_pass=exact_parity_pass,
        local_persistent_core_sub2=local_sub2,
        no_dense_int16_counted_authority_local=no_dense_authority,
        dense_oracle_control_used_for_comparison=all(
            bool(proof.get("dense_oracle_control_used"))
            for proof in proof_by_key.values()
        ),
        oracle_control_authority="transient_comparison_only_not_persisted_or_counted",
        production_authority_claim_authorized=False,
        live_runtime_authority_converted=False,
        trainer_entrypoint_uses_candidate=False,
        readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        current_repo_readiness_rows_may_flip=False,
        physical_persistent_bits_per_weight=inclusive_bpw,
        effective_persistent_bits_per_weight=inclusive_bpw,
        target_bits_per_weight=2.0,
        declared_coverage_domain=coverage_domain,
        uncovered_blockers=LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS,
        sidecar_report=sidecar_report,
        local_step_summary=dict(step.global_summary),
        proof_by_key=proof_by_key,
        proof_anchors=(
            "bounded_delta_learner.py:1025",
            "bounded_delta_learner.py:1064",
            "bounded_delta_learner.py:1110",
            "bounded_delta_accumulator.py:837",
            "sub2_native_birth_sidecar_runtime.py:304",
        ),
        non_claims=LIVE_LOCAL_SUB2_AUTHORITY_NON_CLAIMS,
    )
    validate_live_local_sub2_authority_receipt(receipt)
    return receipt


def validate_live_local_sub2_authority_receipt(receipt: LiveLocalSub2AuthorityReceipt) -> None:
    if receipt.schema_version != LIVE_LOCAL_SUB2_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("live-local sub2 authority schema version mismatch")
    if receipt.target_name != LIVE_LOCAL_SUB2_AUTHORITY_TARGET_NAME:
        raise ValueError("live-local sub2 authority target name mismatch")
    if receipt.production_authority_claim_authorized:
        raise ValueError("2B0 cannot authorize production authority claims")
    if receipt.live_runtime_authority_converted:
        raise ValueError("2B0 cannot claim trainer-used live runtime authority conversion")
    if receipt.trainer_entrypoint_uses_candidate:
        raise ValueError("2B0 cannot claim trainer entrypoint use")
    if receipt.readiness_row_flip_authorized or receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("2B0 cannot authorize broad readiness row flips")
    if receipt.current_repo_readiness_rows_may_flip:
        raise ValueError("2B0 cannot flip FIXTURE_CURRENT_REPO q/acc readiness rows")
    if tuple(receipt.uncovered_blockers) != LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS:
        raise ValueError("2B0 uncovered blocker list must name trainer/global/replay/PC debt")
    domain = receipt.declared_coverage_domain
    if not bool(domain.get("no_global_cap")):
        raise ValueError("2B0 coverage domain must be local/no-global-cap")
    for unsupported in (
        "supports_replay_ce_veto",
        "supports_pc_aux",
        "supports_global_backlog",
        "supports_dense_vote_authority",
        "supports_dense_shadow_authority",
        "supports_dense_decode_candidate_path",
    ):
        if bool(domain.get(unsupported)):
            raise ValueError(f"2B0 coverage cannot support {unsupported}")
    if not receipt.exact_local_parity_pass:
        raise ValueError("2B0 exact local parity must pass")
    if not receipt.local_authority_seam_executable:
        raise ValueError("2B0 local authority seam must be executable")
    if not receipt.dense_oracle_control_used_for_comparison:
        raise ValueError("2B0 receipt must include transient dense oracle comparison proof")
    if receipt.oracle_control_authority != "transient_comparison_only_not_persisted_or_counted":
        raise ValueError("2B0 dense oracle must be comparison-only")
    if receipt.sidecar_report.persistent_dense_shadow_present:
        raise ValueError("2B0 sidecar report cannot contain dense shadow authority")
    if int(receipt.sidecar_report.persistent_dense_shadow_bytes) != 0:
        raise ValueError("2B0 sidecar dense shadow bytes must be zero")
    if not receipt.sidecar_report.pass_report or not receipt.sidecar_report.budget_guard.pass_guard:
        raise ValueError("2B0 sidecar persistent budget must pass")
    if float(receipt.physical_persistent_bits_per_weight) >= float(receipt.target_bits_per_weight):
        raise ValueError("2B0 physical persistent bits/weight must stay < 2")
    if abs(
        float(receipt.physical_persistent_bits_per_weight)
        - float(receipt.effective_persistent_bits_per_weight)
    ) > 1e-12:
        raise ValueError("2B0 physical/effective bits must match the sidecar ledger")
    if not receipt.no_dense_int16_counted_authority_local:
        raise ValueError("2B0 local receipt must exclude counted dense-int16 authority")
    if not receipt.local_persistent_core_sub2:
        raise ValueError("2B0 local persistent core must be sub2")
    for key, proof in receipt.proof_by_key.items():
        if not bool(proof.get("pass")) or not bool(proof.get("parity_pass")):
            raise ValueError(f"2B0 proof for {key} does not pass exact parity")
        if proof.get("candidate_q_sha256_after") != proof.get("oracle_q_sha256_after"):
            raise ValueError(f"2B0 q hash parity failed for {key}")
        if proof.get("candidate_bounded_decode_sha256_after") != proof.get("oracle_acc_sha256_after"):
            raise ValueError(f"2B0 residual decode hash parity failed for {key}")
        if proof.get("applied_row_identities_sha256") != proof.get("oracle_applied_row_identities_sha256"):
            raise ValueError(f"2B0 applied row identity hash parity failed for {key}")
        if proof.get("residual_after_threshold_sha256") != proof.get("oracle_residual_after_threshold_sha256"):
            raise ValueError(f"2B0 residual-after-threshold hash parity failed for {key}")
        if bool(proof.get("candidate_dense_decode_used")):
            raise ValueError(f"2B0 candidate path used dense decode for {key}")
        if bool(proof.get("candidate_dense_vote_authority_used")):
            raise ValueError(f"2B0 candidate path used dense vote authority for {key}")
    serialized_non_claims = " ".join(receipt.non_claims)
    for required in (
        "live_runtime_authority_converted=false",
        "trainer_entrypoint_uses_candidate=false",
        "readiness_row_flip_authorized=false",
        "global_cap/replay/PC/backlog/trainer integration",
    ):
        if required not in serialized_non_claims:
            raise ValueError(f"2B0 non-claims must include {required}")
    if bool(receipt.pass_receipt) != bool(
        receipt.local_authority_seam_executable
        and receipt.exact_local_parity_pass
        and receipt.local_persistent_core_sub2
        and receipt.no_dense_int16_counted_authority_local
        and not receipt.production_authority_claim_authorized
        and not receipt.live_runtime_authority_converted
        and not receipt.trainer_entrypoint_uses_candidate
        and not receipt.readiness_row_flip_authorized
        and not receipt.current_repo_readiness_rows_may_flip
    ):
        raise ValueError("2B0 pass_receipt must be computed from explicit local-only gates")


__all__ = [
    "LIVE_LOCAL_SUB2_AUTHORITY_ENTRYPOINT",
    "LIVE_LOCAL_SUB2_AUTHORITY_NON_CLAIMS",
    "LIVE_LOCAL_SUB2_AUTHORITY_SCHEMA_VERSION",
    "LIVE_LOCAL_SUB2_AUTHORITY_TARGET_NAME",
    "LIVE_LOCAL_SUB2_AUTHORITY_UNCOVERED_BLOCKERS",
    "LiveLocalSub2AuthorityReceipt",
    "build_live_local_sub2_authority_receipt",
    "validate_live_local_sub2_authority_receipt",
]
