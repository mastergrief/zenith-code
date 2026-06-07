"""2C1 trainer-facing sub-2 authority construction/counting proof.

This module is deliberately construction-only. It lets the real trainer inspect
its built ``LMHead``/``BitLinear`` modules and validate a q+scale+bounded-acc
authority payload without changing the trainer's checkpoint format or claiming
update parity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Callable, Mapping

import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    BoundedDeltaTensorState,
    apply_bounded_delta_vote_step,
    authoritative_forward_context,
    build_authoritative_checkpoint_payload,
    build_optimizer_excluding_eligible_masters,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    derive_bounded_tensor_state_from_weight,
    make_candidate_authority_tensor_state,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    tensor_sha256,
    validate_authoritative_resume_payload,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateResult,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
)


TRAINER_SUB2_AUTHORITY_SCHEMA_VERSION = (
    "hrm_text_158_2c1_trainer_sub2_authority/v0.construction_counting_only"
)
TRAINER_SUB2_AUTHORITY_TARGET_NAME = "step2c1_trainer_sub2_authority_construction"
TRAINER_SUB2_LOCAL_UPDATE_SCHEMA_VERSION = (
    "hrm_text_158_2c2_trainer_sub2_authority/v0.local_update_proof"
)
TRAINER_SUB2_LOCAL_UPDATE_TARGET_NAME = "step2c2_trainer_local_qacc_update_proof"
TRAINER_SUB2_AUTHORITY_NON_CLAIMS = (
    "2C1 proves default-off construction/counting/payload validation only",
    "trainer_entrypoint_uses_candidate=false; no learner update is invoked",
    "live_runtime_authority_converted=false; no production trainer authority conversion",
    "readiness_row_flip_authorized=false; FIXTURE_CURRENT_REPO q/acc rows remain unchanged",
    "strict model_state load/save semantics are not altered; checkpoint versioning is deferred to 2C4",
    "not learning, acquisition, update parity, throughput, GPU residency, training launch, or .pt mutation",
)
TRAINER_SUB2_LOCAL_UPDATE_NON_CLAIMS = (
    "2C2 proves a default-off trainer local qacc update proof only",
    "dense weighted_grad/credit/projected-move/rank-vote tensors are proof-only transient over-2 tensors",
    "exact oracle decode/comparison is proof-only transient and never persisted as authority",
    "trainer_entrypoint_uses_candidate=false; production/broad runtime flags remain false until 2C4",
    "global cap, replay CE veto, PC auxiliary, backlog, checkpoint resume, and readiness row flips are deferred",
    "not learning, acquisition, throughput, GPU residency, training launch, full-sub2 runtime, or .pt mutation",
)


def _identity_sha256(state_key: str, indices: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for index in sorted(int(item) for item in indices):
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(index).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_identity_sha256(state_key: str, indices: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for index in indices:
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _ordered_value_sha256(state_key: str, label: str, values: Mapping[int, int]) -> str:
    h = hashlib.sha256()
    for index, value in values.items():
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(label).encode("utf-8"))
        h.update(b":")
        h.update(str(int(index)).encode("utf-8"))
        h.update(b"=")
        h.update(str(int(value)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _sparse_value_sha256(state_key: str, values: Mapping[int, int]) -> str:
    h = hashlib.sha256()
    for index, value in sorted((int(k), int(v)) for k, v in values.items()):
        h.update(str(state_key).encode("utf-8"))
        h.update(b":")
        h.update(str(index).encode("utf-8"))
        h.update(b"=")
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class TrainerSub2AuthorityConstructionReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    dry_run: bool
    gpu_launched: bool
    checkpoint_written: bool
    learner_update_called: bool
    optimizer_step_called: bool
    trainer_entrypoint_can_construct_sub2_authority: bool
    trainer_entrypoint_uses_candidate: bool
    live_runtime_authority_converted: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    use_ternary_bulk_required: bool
    use_ternary_bulk_observed: bool
    eligible_scope: str
    eligible_module_count: int
    eligible_state_keys: tuple[str, ...]
    eligible_weight_count: int
    optimizer_exclusion_proof: dict[str, Any]
    checkpoint_payload_validated: bool
    checkpoint_payload_summary: dict[str, Any]
    countability_ledger: dict[str, Any]
    persistent_authority_bits_per_weight: float
    target_bits_per_weight: float
    dense_int16_persistent_authority_bits_counted: int
    fp_master_persistent_authority_bits_counted: int
    proof_anchors: tuple[str, ...]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_row_flip_authorized_surface_names"] = list(
            self.readiness_row_flip_authorized_surface_names
        )
        payload["eligible_state_keys"] = list(self.eligible_state_keys)
        payload["proof_anchors"] = list(self.proof_anchors)
        payload["non_claims"] = list(self.non_claims)
        return payload


@dataclass(frozen=True)
class TrainerSub2AuthorityLocalUpdateReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    dry_run: bool
    gpu_launched: bool
    checkpoint_written: bool
    learner_update_called: bool
    optimizer_step_called: bool
    default_off_trainer_local_qacc_update_proof_exercised: bool
    trainer_entrypoint_can_construct_sub2_authority: bool
    trainer_entrypoint_uses_candidate: bool
    live_runtime_authority_converted: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    use_ternary_bulk_required: bool
    use_ternary_bulk_observed: bool
    eligible_scope: str
    eligible_module_count: int
    eligible_state_keys: tuple[str, ...]
    eligible_weight_count: int
    optimizer_exclusion_proof: dict[str, Any]
    forward_backward_capture_proof: dict[str, Any]
    transient_over2_tensors: tuple[str, ...]
    vote_projection_proof: dict[str, Any]
    candidate_step_summary: dict[str, Any]
    exact_local_parity_proof_by_key: dict[str, Any]
    total_sparse_vote_event_count: int
    q_changed_count: int
    authority_state_shadow_free_after: bool
    eligible_fp_masters_byte_identical: bool
    checkpoint_payload_written: bool
    checkpoint_payload_contains_oracle: bool
    proof_anchors: tuple[str, ...]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_row_flip_authorized_surface_names"] = list(
            self.readiness_row_flip_authorized_surface_names
        )
        payload["eligible_state_keys"] = list(self.eligible_state_keys)
        payload["transient_over2_tensors"] = list(self.transient_over2_tensors)
        payload["proof_anchors"] = list(self.proof_anchors)
        payload["non_claims"] = list(self.non_claims)
        return payload


def select_trainer_eligible_bitlinears(
    model: torch.nn.Module,
    *,
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
) -> dict[str, BitLinear]:
    """Find trainer-built BitLinear modules and fail closed outside ternary bulk."""

    if not bool(use_ternary_bulk):
        raise RuntimeError(
            "2C1 trainer sub2 authority proof requires --use-ternary-bulk; "
            "the FP-master Linear path has no eligible BitLinear authority seam"
        )
    modules = {
        str(name): module
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }
    if not modules:
        raise RuntimeError("2C1 trainer sub2 authority proof found no eligible BitLinear modules")
    if eligible_scope == "first-bitlinear":
        key = sorted(modules)[0]
        return {key: modules[key]}
    if eligible_scope == "all-bitlinear":
        return dict(sorted(modules.items()))
    raise ValueError(f"unsupported 2C1 eligible_scope {eligible_scope!r}")


def derive_trainer_sub2_authority_states(
    eligible_modules: Mapping[str, BitLinear],
) -> dict[str, BoundedDeltaTensorState]:
    """Derive q+scale+bounded-acc candidate authority states from trainer weights."""

    states: dict[str, BoundedDeltaTensorState] = {}
    for key, module in sorted(eligible_modules.items()):
        with_shadow = derive_bounded_tensor_state_from_weight(
            str(key),
            module.weight.detach(),
            scale_eps=module._SCALE_EPS,
        )
        states[str(key)] = make_candidate_authority_tensor_state(
            with_shadow,
            with_shadow.q_levels,
            with_shadow.bounded_accumulator,
        )
    return states


def trainer_authoritative_forward_context(
    eligible_modules: Mapping[str, BitLinear],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool,
) -> Any:
    """Trainer-facing wrapper for the reusable q-state forward seam."""

    return authoritative_forward_context(
        eligible_modules,
        tensor_states,
        device=device,
        requires_grad=requires_grad,
    )


def _authority_countability_ledger(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
) -> dict[str, Any]:
    total_weights = sum(int(state.q_levels.numel()) for state in tensor_states.values())
    if total_weights <= 0:
        raise ValueError("2C1 authority countability needs at least one eligible weight")
    q_bits = math.log2(3.0) * float(total_weights)
    scale_bits = 32 * len(tensor_states)
    bounded_acc_bits = 0
    tensor_rows = {}
    for key, state in sorted(tensor_states.items()):
        bounded = state.bounded_accumulator
        hot = len(bounded.hot_exact_indices)
        cold = len(bounded.cold_exception_indices)
        if hot or cold:
            index_bits = max(1, math.ceil(math.log2(float(max(1, bounded.logical_numel)))))
            bounded_acc_bits += hot * (index_bits + 16) + cold * (index_bits + 16)
        tensor_rows[str(key)] = {
            "shape": list(state.q_levels.shape),
            "weight_count": int(state.q_levels.numel()),
            "q_sha256": tensor_sha256(state.q_levels),
            "exact_accumulator_shadow_available": state.exact_accumulator_shadow is not None,
            "hot_exact_row_count": hot,
            "cold_exception_row_count": cold,
        }
    total_bits = q_bits + float(scale_bits) + float(bounded_acc_bits)
    bits_per_weight = float(total_bits) / float(total_weights)
    return {
        "schema": "hrm_text_158_2c1_trainer_sub2_authority_countability/v0",
        "q_regime": "base3_ternary_entropy_pack_ready",
        "eligible_weight_count": int(total_weights),
        "tensor_count": len(tensor_states),
        "q_bits_per_weight": math.log2(3.0),
        "scale_bits_total": int(scale_bits),
        "bounded_accumulator_bits_total": int(bounded_acc_bits),
        "dense_int16_persistent_authority_bits_counted": 0,
        "fp_master_persistent_authority_bits_counted": 0,
        "persistent_authority_bits_per_weight": bits_per_weight,
        "target_bits_per_weight": 2.0,
        "under_target": bits_per_weight < 2.0,
        "tensor_rows": tensor_rows,
        "counting_boundary": (
            "2C1 counts only the q+scale+bounded authority payload; FP masters "
            "remain present in legacy model_state but are non-authoritative and "
            "checkpoint exclusion/versioning is deferred to 2C4"
        ),
    }


def _checkpoint_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "artifact_role": payload.get("artifact_role"),
        "authoritative_state_source": payload.get("authoritative_state_source"),
        "step": int(payload.get("step", -1)),
        "dry_run": bool(payload.get("dry_run")),
        "checkpoint_written": bool(payload.get("checkpoint_written")),
        "q_codec": payload.get("q_codec"),
        "bounded_accumulator_schema": payload.get("bounded_accumulator_schema"),
        "authoritative_state_sha256": payload.get("authoritative_state_sha256"),
        "tensor_summary_count": len(payload.get("tensor_summaries") or {}),
        "tensor_keys": sorted((payload.get("tensor_summaries") or {}).keys()),
    }


def _default_local_vote_update_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=2,
        fraction_per_tensor=1.0,
    )


def _sparse_vote_events(votes: torch.Tensor) -> dict[int, int]:
    flat = votes.detach().cpu().to(torch.int16).flatten()
    return {
        int(index): int(flat[int(index)].item())
        for index in torch.nonzero(flat != 0, as_tuple=False).flatten().tolist()
    }


def _eligible_master_hashes(eligible_modules: Mapping[str, BitLinear]) -> dict[str, str]:
    return {
        str(key): tensor_sha256(module.weight.detach())
        for key, module in sorted(eligible_modules.items())
    }


def _oracle_parity_proof(
    *,
    state_key: str,
    prior_state: BoundedDeltaTensorState,
    next_state: BoundedDeltaTensorState,
    votes: torch.Tensor,
    vote_spec: VoteUpdateSpec,
    candidate_proof: Mapping[str, Any],
) -> dict[str, Any]:
    oracle_state = VoteUpdateState(
        q_levels=prior_state.q_levels.detach().cpu().contiguous(),
        accumulators=prior_state.decoded_accumulators().detach().cpu().contiguous(),
    )
    oracle_result: VoteUpdateResult = apply_integer_vote_update_reference(
        oracle_state,
        VoteUpdateInputs(votes=votes.detach().cpu().to(torch.int16).contiguous()),
        vote_spec,
    )
    oracle_applied = tuple(
        int(index)
        for index in oracle_result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
    )
    oracle_directions = {
        int(index): int(direction)
        for index, direction in zip(
            oracle_applied,
            oracle_result.plan.applied_directions.detach().cpu().to(torch.int16).tolist(),
        )
    }
    oracle_thresholds = {
        int(index): int(threshold)
        for index, threshold in zip(
            oracle_applied,
            oracle_result.plan.applied_thresholds.detach().cpu().to(torch.int32).tolist(),
        )
    }
    oracle_residuals = {
        int(index): int(oracle_result.accumulators.flatten()[int(index)].item())
        for index in oracle_applied
    }
    decoded_next = next_state.decoded_accumulators().detach().cpu().contiguous()
    oracle_hashes = {
        "oracle_q_sha256_after": tensor_sha256(oracle_result.q_levels),
        "oracle_acc_sha256_after": tensor_sha256(oracle_result.accumulators),
        "oracle_applied_row_identities_sha256": _identity_sha256(state_key, oracle_applied),
        "oracle_ordered_applied_row_identities_sha256": _ordered_identity_sha256(
            state_key,
            oracle_applied,
        ),
        "oracle_applied_directions_sha256": _ordered_value_sha256(
            state_key,
            "direction",
            oracle_directions,
        ),
        "oracle_applied_thresholds_sha256": _ordered_value_sha256(
            state_key,
            "threshold",
            oracle_thresholds,
        ),
        "oracle_residual_after_threshold_sha256": _sparse_value_sha256(
            state_key,
            oracle_residuals,
        ),
    }
    candidate_hashes = {
        "candidate_q_sha256_after": tensor_sha256(next_state.q_levels),
        "candidate_bounded_decode_sha256_after": tensor_sha256(decoded_next),
        "candidate_applied_row_identities_sha256": candidate_proof.get(
            "applied_row_identities_sha256"
        ),
        "candidate_ordered_applied_row_identities_sha256": candidate_proof.get(
            "ordered_applied_row_identities_sha256"
        ),
        "candidate_applied_directions_sha256": candidate_proof.get("applied_directions_sha256"),
        "candidate_applied_thresholds_sha256": candidate_proof.get("applied_thresholds_sha256"),
        "candidate_residual_after_threshold_sha256": candidate_proof.get(
            "residual_after_threshold_sha256"
        ),
    }
    counters_match = (
        int(candidate_proof.get("candidate_count", -1))
        == int(oracle_result.plan.candidate_indices.numel())
        and int(candidate_proof.get("applied_row_count", -1))
        == int(oracle_result.plan.applied_indices.numel())
        and int(candidate_proof.get("q_changed_count", -1))
        == int(oracle_result.stats.get("q_changed_count", -2))
        and int(candidate_proof.get("event_vote_count", -1))
        == int(oracle_result.stats.get("vote_nonzero_count", -2))
    )
    parity_pass = bool(
        candidate_hashes["candidate_q_sha256_after"] == oracle_hashes["oracle_q_sha256_after"]
        and candidate_hashes["candidate_bounded_decode_sha256_after"]
        == oracle_hashes["oracle_acc_sha256_after"]
        and candidate_hashes["candidate_applied_row_identities_sha256"]
        == oracle_hashes["oracle_applied_row_identities_sha256"]
        and candidate_hashes["candidate_ordered_applied_row_identities_sha256"]
        == oracle_hashes["oracle_ordered_applied_row_identities_sha256"]
        and candidate_hashes["candidate_applied_directions_sha256"]
        == oracle_hashes["oracle_applied_directions_sha256"]
        and candidate_hashes["candidate_applied_thresholds_sha256"]
        == oracle_hashes["oracle_applied_thresholds_sha256"]
        and candidate_hashes["candidate_residual_after_threshold_sha256"]
        == oracle_hashes["oracle_residual_after_threshold_sha256"]
        and counters_match
    )
    return {
        "state_key": str(state_key),
        "parity_pass": parity_pass,
        "counters_match": bool(counters_match),
        "oracle_candidate_count": int(oracle_result.plan.candidate_indices.numel()),
        "oracle_applied_count": int(oracle_result.plan.applied_indices.numel()),
        "oracle_q_changed_count": int(oracle_result.stats.get("q_changed_count", 0)),
        "oracle_vote_nonzero_count": int(oracle_result.stats.get("vote_nonzero_count", 0)),
        **oracle_hashes,
        **candidate_hashes,
    }


def build_trainer_sub2_authority_construction_receipt(
    model: torch.nn.Module,
    *,
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    lr: float = 0.0,
    weight_decay: float = 0.0,
    step: int = 0,
) -> TrainerSub2AuthorityConstructionReceipt:
    """Build the gated 2C1 receipt without invoking learner update/training."""

    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    _optimizer, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model,
        eligible,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    countability = _authority_countability_ledger(states)
    payload = build_authoritative_checkpoint_payload(
        states,
        step=int(step),
        updater_config={
            "scope": "2C1_construction_counting_only",
            "eligible_scope": str(eligible_scope),
            "learner_update_called": False,
            "optimizer_step_called": False,
        },
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
    )
    validate_authoritative_resume_payload(payload)
    summary = _checkpoint_payload_summary(payload)
    pass_receipt = bool(
        eligible
        and states
        and optimizer_checks.get("pass")
        and int(optimizer_checks.get("eligible_params_in_optimizer", -1)) == 0
        and int(optimizer_checks.get("eligible_optimizer_state_entries", -1)) == 0
        and bool(countability.get("under_target"))
        and int(countability.get("dense_int16_persistent_authority_bits_counted", -1)) == 0
        and int(countability.get("fp_master_persistent_authority_bits_counted", -1)) == 0
        and summary["dry_run"] is True
        and summary["checkpoint_written"] is False
    )
    receipt = TrainerSub2AuthorityConstructionReceipt(
        schema_version=TRAINER_SUB2_AUTHORITY_SCHEMA_VERSION,
        target_name=TRAINER_SUB2_AUTHORITY_TARGET_NAME,
        pass_receipt=pass_receipt,
        dry_run=True,
        gpu_launched=False,
        checkpoint_written=False,
        learner_update_called=False,
        optimizer_step_called=False,
        trainer_entrypoint_can_construct_sub2_authority=pass_receipt,
        trainer_entrypoint_uses_candidate=False,
        live_runtime_authority_converted=False,
        readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        use_ternary_bulk_required=True,
        use_ternary_bulk_observed=bool(use_ternary_bulk),
        eligible_scope=str(eligible_scope),
        eligible_module_count=len(eligible),
        eligible_state_keys=tuple(sorted(states)),
        eligible_weight_count=int(countability["eligible_weight_count"]),
        optimizer_exclusion_proof=dict(optimizer_checks),
        checkpoint_payload_validated=True,
        checkpoint_payload_summary=summary,
        countability_ledger=countability,
        persistent_authority_bits_per_weight=float(countability["persistent_authority_bits_per_weight"]),
        target_bits_per_weight=2.0,
        dense_int16_persistent_authority_bits_counted=0,
        fp_master_persistent_authority_bits_counted=0,
        proof_anchors=(
            "scripts/train_hrm_text_158.py:1357",
            "bounded_delta_learner.py:680",
            "bounded_delta_learner.py:729",
            "bounded_delta_learner.py:1404",
        ),
        non_claims=TRAINER_SUB2_AUTHORITY_NON_CLAIMS,
    )
    validate_trainer_sub2_authority_construction_receipt(receipt)
    return receipt


def build_trainer_sub2_authority_local_update_receipt(
    model: torch.nn.Module,
    *,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    lr: float = 0.0,
    weight_decay: float = 0.0,
    step: int = 0,
    vote_update_spec: VoteUpdateSpec | None = None,
) -> TrainerSub2AuthorityLocalUpdateReceipt:
    """Run the gated 2C2 default-off trainer local qacc proof."""

    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    _optimizer, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model,
        eligible,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    before_master_hashes = _eligible_master_hashes(eligible)
    rank_spec = default_dry_run_rank_vote_spec()
    update_spec = vote_update_spec or _default_local_vote_update_spec()
    vote_specs_by_key = {key: update_spec for key in states}
    dense_votes_by_key: dict[str, torch.Tensor] = {}
    sparse_events_by_key: dict[str, dict[int, int]] = {}
    weighted_grad_stats: dict[str, dict[str, Any]] = {}
    prior_training = bool(model.training)
    loss_finite = False
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device=device,
            requires_grad=True,
        ) as handle:
            loss = forward_loss_fn(model, batch)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("2C2 forward_loss_fn must return a torch.Tensor loss")
            loss_to_backward = loss if loss.numel() == 1 else loss.mean()
            loss_finite = bool(torch.isfinite(loss_to_backward.detach()).item())
            loss_to_backward.backward()
            for key, state in sorted(states.items()):
                weighted_grad = handle.weighted_grad(key)
                credit = credit_from_weighted_grad(weighted_grad)
                moves = project_s1_gradient_to_moves(weighted_grad, state.q_levels)
                votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
                dense_votes_by_key[key] = votes.detach().cpu().to(torch.int16).contiguous()
                sparse_events_by_key[key] = _sparse_vote_events(votes)
                weighted_grad_stats[key] = {
                    "weighted_grad_shape": list(weighted_grad.shape),
                    "weighted_grad_nonzero_count": int((weighted_grad != 0).sum().item()),
                    "credit_nonzero_count": int((credit != 0).sum().item()),
                    "projected_move_nonzero_count": int((moves != 0).sum().item()),
                    "dense_rank_vote_nonzero_count": int((votes != 0).sum().item()),
                    "sparse_vote_event_count": int(len(sparse_events_by_key[key])),
                    "weighted_grad_finite": bool(torch.isfinite(weighted_grad).all().item()),
                }
    finally:
        model.train(prior_training)

    step_result = apply_bounded_delta_vote_step(
        states,
        dense_votes_by_key,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
    )
    proof_by_key = step_result.global_summary["candidate_local_update_proof_by_key"]
    parity_by_key = {
        key: _oracle_parity_proof(
            state_key=key,
            prior_state=states[key],
            next_state=step_result.tensor_states[key],
            votes=dense_votes_by_key[key],
            vote_spec=vote_specs_by_key[key],
            candidate_proof=proof_by_key[key],
        )
        for key in sorted(states)
    }
    after_master_hashes = _eligible_master_hashes(eligible)
    fp_masters_byte_identical = before_master_hashes == after_master_hashes
    shadow_free_after = all(
        state.exact_accumulator_shadow is None
        for state in step_result.tensor_states.values()
    )
    total_sparse_events = sum(len(events) for events in sparse_events_by_key.values())
    q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
    pass_receipt = bool(
        loss_finite
        and optimizer_checks.get("pass")
        and int(optimizer_checks.get("eligible_params_in_optimizer", -1)) == 0
        and int(optimizer_checks.get("eligible_optimizer_state_entries", -1)) == 0
        and total_sparse_events > 0
        and q_changed_count > 0
        and all(bool(proof.get("pass")) for proof in proof_by_key.values())
        and all(bool(proof.get("parity_pass")) for proof in parity_by_key.values())
        and shadow_free_after
        and fp_masters_byte_identical
    )
    receipt = TrainerSub2AuthorityLocalUpdateReceipt(
        schema_version=TRAINER_SUB2_LOCAL_UPDATE_SCHEMA_VERSION,
        target_name=TRAINER_SUB2_LOCAL_UPDATE_TARGET_NAME,
        pass_receipt=pass_receipt,
        dry_run=True,
        gpu_launched=False,
        checkpoint_written=False,
        learner_update_called=True,
        optimizer_step_called=False,
        default_off_trainer_local_qacc_update_proof_exercised=pass_receipt,
        trainer_entrypoint_can_construct_sub2_authority=True,
        trainer_entrypoint_uses_candidate=False,
        live_runtime_authority_converted=False,
        readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        use_ternary_bulk_required=True,
        use_ternary_bulk_observed=bool(use_ternary_bulk),
        eligible_scope=str(eligible_scope),
        eligible_module_count=len(eligible),
        eligible_state_keys=tuple(sorted(states)),
        eligible_weight_count=sum(int(state.q_levels.numel()) for state in states.values()),
        optimizer_exclusion_proof=dict(optimizer_checks),
        forward_backward_capture_proof={
            "trainer_authoritative_forward_context_requires_grad": True,
            "forward_backward_loss_finite": bool(loss_finite),
            "weighted_grad_capture_by_key": weighted_grad_stats,
        },
        transient_over2_tensors=(
            "weighted_grad",
            "credit",
            "projected_moves",
            "dense_rank_votes_before_sparse_event_extraction",
            "decoded_bounded_accumulator_for_exact_oracle_control",
            "dense_oracle_qacc_reference_result",
        ),
        vote_projection_proof={
            "rank_vote_spec": rank_spec.to_live_dict(),
            "vote_update_spec": asdict(update_spec),
            "candidate_mode": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            "candidate_sparse_vote_events_only": True,
            "dense_vote_authority_persisted": False,
            "total_sparse_vote_event_count": int(total_sparse_events),
        },
        candidate_step_summary={
            "candidate_local_update_pass": bool(
                step_result.global_summary.get("candidate_local_update_pass")
            ),
            "q_changed_count": q_changed_count,
            "candidate_dense_decode_used": bool(
                step_result.global_summary.get("candidate_dense_decode_used")
            ),
            "candidate_accumulator_transient_over2_used": bool(
                step_result.global_summary.get("candidate_accumulator_transient_over2_used")
            ),
            "candidate_vote_transient_over2_used": bool(
                step_result.global_summary.get("candidate_vote_transient_over2_used")
            ),
            "candidate_dense_vote_authority_used": bool(
                step_result.global_summary.get("candidate_dense_vote_authority_used")
            ),
            "candidate_local_update_proof_by_key": dict(proof_by_key),
        },
        exact_local_parity_proof_by_key=parity_by_key,
        total_sparse_vote_event_count=int(total_sparse_events),
        q_changed_count=q_changed_count,
        authority_state_shadow_free_after=shadow_free_after,
        eligible_fp_masters_byte_identical=fp_masters_byte_identical,
        checkpoint_payload_written=False,
        checkpoint_payload_contains_oracle=False,
        proof_anchors=(
            "scripts/train_hrm_text_158.py:1466",
            "bounded_delta_learner.py:211",
            "bounded_delta_learner.py:226",
            "bounded_delta_learner.py:307",
            "bounded_delta_learner.py:1025",
            "vote_update.py:269",
            "vote_update.py:409",
        ),
        non_claims=TRAINER_SUB2_LOCAL_UPDATE_NON_CLAIMS,
    )
    validate_trainer_sub2_authority_local_update_receipt(receipt)
    return receipt


def validate_trainer_sub2_authority_construction_receipt(
    receipt: TrainerSub2AuthorityConstructionReceipt,
) -> None:
    if receipt.schema_version != TRAINER_SUB2_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("2C1 trainer authority schema version mismatch")
    if receipt.target_name != TRAINER_SUB2_AUTHORITY_TARGET_NAME:
        raise ValueError("2C1 trainer authority target name mismatch")
    if not receipt.dry_run or receipt.gpu_launched or receipt.checkpoint_written:
        raise ValueError("2C1 must stay CPU/dry-run and must not write checkpoints")
    if receipt.learner_update_called or receipt.optimizer_step_called:
        raise ValueError("2C1 cannot call learner update or optimizer step")
    if not receipt.trainer_entrypoint_can_construct_sub2_authority:
        raise ValueError("2C1 trainer entrypoint construction proof did not pass")
    if receipt.trainer_entrypoint_uses_candidate:
        raise ValueError("2C1 cannot claim trainer-used candidate authority")
    if receipt.live_runtime_authority_converted:
        raise ValueError("2C1 cannot claim live runtime authority conversion")
    if receipt.readiness_row_flip_authorized or receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("2C1 cannot authorize readiness row flips")
    if not receipt.use_ternary_bulk_required or not receipt.use_ternary_bulk_observed:
        raise ValueError("2C1 requires observed ternary bulk")
    if receipt.eligible_module_count <= 0 or receipt.eligible_weight_count <= 0:
        raise ValueError("2C1 needs at least one eligible BitLinear module")
    checks = dict(receipt.optimizer_exclusion_proof)
    if not checks.get("pass"):
        raise ValueError("2C1 optimizer exclusion proof failed")
    if int(checks.get("eligible_params_in_optimizer", -1)) != 0:
        raise ValueError("2C1 eligible masters entered optimizer")
    if int(checks.get("eligible_optimizer_state_entries", -1)) != 0:
        raise ValueError("2C1 eligible masters have optimizer state entries")
    if not receipt.checkpoint_payload_validated:
        raise ValueError("2C1 checkpoint payload was not validated")
    if bool(receipt.checkpoint_payload_summary.get("checkpoint_written")):
        raise ValueError("2C1 checkpoint payload must be sidecar/dry-run only")
    if receipt.persistent_authority_bits_per_weight >= receipt.target_bits_per_weight:
        raise ValueError("2C1 q+scale+bounded authority is not countable under sub-2")
    if receipt.dense_int16_persistent_authority_bits_counted != 0:
        raise ValueError("2C1 cannot count dense int16 as persistent authority")
    if receipt.fp_master_persistent_authority_bits_counted != 0:
        raise ValueError("2C1 cannot count FP master as persistent authority")
    if tuple(receipt.non_claims) != TRAINER_SUB2_AUTHORITY_NON_CLAIMS:
        raise ValueError("2C1 non-claims changed")


def validate_trainer_sub2_authority_local_update_receipt(
    receipt: TrainerSub2AuthorityLocalUpdateReceipt,
) -> None:
    if receipt.schema_version != TRAINER_SUB2_LOCAL_UPDATE_SCHEMA_VERSION:
        raise ValueError("2C2 trainer local update schema version mismatch")
    if receipt.target_name != TRAINER_SUB2_LOCAL_UPDATE_TARGET_NAME:
        raise ValueError("2C2 trainer local update target name mismatch")
    if not receipt.dry_run or receipt.gpu_launched or receipt.checkpoint_written:
        raise ValueError("2C2 must stay CPU/dry-run and must not write checkpoints")
    if not receipt.learner_update_called:
        raise ValueError("2C2 must exercise the local qacc learner update")
    if receipt.optimizer_step_called:
        raise ValueError("2C2 must exit before optimizer step")
    if not receipt.default_off_trainer_local_qacc_update_proof_exercised:
        raise ValueError("2C2 local qacc proof was not exercised")
    if receipt.trainer_entrypoint_uses_candidate:
        raise ValueError("2C2 cannot flip broad trainer_entrypoint_uses_candidate")
    if receipt.live_runtime_authority_converted:
        raise ValueError("2C2 cannot claim live runtime authority conversion")
    if receipt.readiness_row_flip_authorized or receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("2C2 cannot authorize readiness row flips")
    if not receipt.use_ternary_bulk_required or not receipt.use_ternary_bulk_observed:
        raise ValueError("2C2 requires observed ternary bulk")
    if receipt.eligible_module_count <= 0 or receipt.eligible_weight_count <= 0:
        raise ValueError("2C2 needs at least one eligible BitLinear module")
    checks = dict(receipt.optimizer_exclusion_proof)
    if not checks.get("pass"):
        raise ValueError("2C2 optimizer exclusion proof failed")
    if int(checks.get("eligible_params_in_optimizer", -1)) != 0:
        raise ValueError("2C2 eligible masters entered optimizer")
    if int(checks.get("eligible_optimizer_state_entries", -1)) != 0:
        raise ValueError("2C2 eligible masters have optimizer state entries")
    if receipt.total_sparse_vote_event_count <= 0:
        raise ValueError("2C2 needs nonzero sparse vote events")
    if receipt.q_changed_count <= 0:
        raise ValueError("2C2 must prove q changed in the authority state")
    if not receipt.authority_state_shadow_free_after:
        raise ValueError("2C2 candidate authority state retained an exact shadow")
    if not receipt.eligible_fp_masters_byte_identical:
        raise ValueError("2C2 eligible FP masters changed")
    if receipt.checkpoint_payload_written or receipt.checkpoint_payload_contains_oracle:
        raise ValueError("2C2 cannot write checkpoint payloads or persist oracle controls")
    if "weighted_grad" not in receipt.transient_over2_tensors:
        raise ValueError("2C2 receipt must name proof-only transient over-2 tensors")
    if not bool(receipt.candidate_step_summary.get("candidate_local_update_pass")):
        raise ValueError("2C2 candidate local update did not pass")
    if bool(receipt.candidate_step_summary.get("candidate_dense_vote_authority_used")):
        raise ValueError("2C2 cannot use dense vote authority in the candidate path")
    if bool(receipt.candidate_step_summary.get("candidate_dense_decode_used")):
        raise ValueError("2C2 cannot dense-decode in the candidate path")
    for key, proof in receipt.exact_local_parity_proof_by_key.items():
        if not bool(proof.get("parity_pass")):
            raise ValueError(f"2C2 exact local parity failed for {key}")
    if tuple(receipt.non_claims) != TRAINER_SUB2_LOCAL_UPDATE_NON_CLAIMS:
        raise ValueError("2C2 non-claims changed")
    if not receipt.pass_receipt:
        raise ValueError("2C2 trainer local update proof did not pass")
