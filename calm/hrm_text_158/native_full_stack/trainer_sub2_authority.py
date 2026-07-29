"""2C1 trainer-facing sub-2 authority construction/counting proof.

This module is deliberately construction-only. It lets the real trainer inspect
its built ``LMHead``/``BitLinear`` modules and validate a q+scale+bounded-acc
authority payload without changing the trainer's checkpoint format or claiming
update parity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from typing import Any, Callable, Mapping, Sequence

import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    decode_bounded_accumulator_to_i16,
)
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
    sparse_rank_bucketed_int16_vote_events_from_weighted_grad,
    tensor_sha256,
    validate_authoritative_resume_payload,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    PackedEventCodedAccState,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    event_coded_live_carrier_enabled,
    hydrate_event_coded_live_carrier_from_packed,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
    PACKED_EVENT_CODED_ACC_FORMAT,
)
from calm.hrm_text_158.native_full_stack.named_receipt_binding import (
    build_named_receipt_path_bindings,
    logical_shape_by_key_from_q_levels,
    require_finite_nonnegative_interval,
    require_lowercase_sha256_hex,
    validate_named_receipt_evidence_maps,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    PackedW5AccumulatorPayload,
    PackedW6AccumulatorPayload,
    W5_BYTE_PACKED_SCHEMA,
    W6_BYTE_PACKED_SCHEMA,
    pack_w5_lanes_to_bytes,
    pack_w6_lanes_to_bytes,
    reject_int16_tensor_as_packed_acc,
    unpack_w5_lanes_from_bytes,
    unpack_w6_lanes_from_bytes,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PACKED_TERNARY_Q_FORMAT,
    PackedTernaryQState,
    _reject_int8_tensor_as_packed_q,
    pack_ternary_q_2bit_reference,
    unpack_ternary_q_2bit_reference,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    BASE3_Q_FORMAT,
    PackedBase3TernaryQState,
    pack_ternary_q_base3_5perbyte_reference,
    unpack_ternary_q_base3_5perbyte_reference,
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
TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION = (
    "hrm_text_158_2c4a_trainer_sub2_authority/v0.roundtrip_resume_update_proof"
)
TRAINER_SUB2_ROUNDTRIP_TARGET_NAME = "step2c4a_trainer_authority_checkpoint_roundtrip"
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
TRAINER_SUB2_ROUNDTRIP_NON_CLAIMS = (
    "2C4a proves reconstructable trainer-used sidecar checkpoint/resume/update proof only",
    "eligible FP masters are excluded from authoritative model_state and explicitly non-authoritative",
    "poisoned FP-master falsification covers the resumed proof path, not broad production runtime",
    "dense int16 accumulators remain proof-only transient and are not saved/loaded as authority",
    "normal BitLinear weight forward remains the legacy path and is not claimed as sub2",
    "readiness row flips, Step 3, GPU launch, native-birth, acquisition, retention, learning, and .pt commits are deferred",
)
TRAINER_SUB2_ACTIVE_CONTROL_PARAMETER_NAMES = frozenset(
    {
        "global_cap_spec",
        "deferred_backlog",
        "replay_ce_veto_votes_by_key",
        "replay_ce_veto_moves_by_key",
        "pc_aux_votes_by_key",
        "pc_aux_moves_by_key",
        "pc_aux_mode",
        "front_c_identity_observer",
    }
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
    default_off_trainer_active_controls_inactive_proven: bool
    global_cap_spec_passed: bool
    global_rate_cap_enabled: bool
    deferred_backlog_input_present: bool
    deferred_backlog_output_entry_count: int
    replay_ce_veto_maps_present: bool
    pc_aux_maps_present: bool
    pc_aux_mode_effective: str
    front_c_identity_observer_present: bool
    candidate_mode_rejects_active_controls: bool
    trainer_builder_has_no_active_control_parameters: bool
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


@dataclass(frozen=True)
class TrainerSub2AuthorityRoundtripReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    dry_run: bool
    gpu_launched: bool
    checkpoint_written: bool
    learner_update_called: bool
    optimizer_step_called: bool
    persistent_authority_state_roundtrip_pass: bool
    trainer_state_mutation_uses_sub2_authority: bool
    resumed_forward_uses_sidecar_authority: bool
    poisoned_fp_master_bypass_falsified: bool
    eligible_fp_masters_authoritative: bool
    eligible_fp_master_keys_excluded_from_authoritative_model_state: bool
    raw_state_dict_eligible_weight_fallback_rejected: bool
    normal_bitlinear_weight_forward_not_claimed: bool
    dense_int16_persistent_accumulator_saved: bool
    dense_int16_persistent_accumulator_loaded: bool
    q_scale_sidecar_bounded_hash_roundtrip_pass: bool
    post_resume_update_mutated_resumed_sub2_authority: bool
    update_law_quality_claim: bool
    learning_claim: bool
    optimizer_credit_state_resolved: bool
    credit_ranking_uninformative_update_law_pivot_deferred: bool
    trainer_entrypoint_can_construct_sub2_authority: bool
    trainer_entrypoint_uses_candidate: bool
    live_runtime_authority_converted: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    broad_runtime_authority_converted: bool
    full_sub2_runtime_readiness_claim: bool
    use_ternary_bulk_required: bool
    use_ternary_bulk_observed: bool
    eligible_scope: str
    eligible_module_count: int
    eligible_state_keys: tuple[str, ...]
    eligible_weight_count: int
    checkpoint_payload_summary: dict[str, Any]
    checkpoint_load_proof: dict[str, Any]
    poison_forward_proof: dict[str, Any]
    post_resume_update_proof: dict[str, Any]
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


def trainer_local_update_builder_active_control_parameters() -> tuple[str, ...]:
    """List forbidden active-control params if the default-off builder grows any."""

    signature = inspect.signature(build_trainer_sub2_authority_local_update_receipt)
    return tuple(
        sorted(
            name
            for name in signature.parameters
            if name in TRAINER_SUB2_ACTIVE_CONTROL_PARAMETER_NAMES
        )
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


def build_checkpoint_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, bankable receipt view of an authoritative checkpoint payload.

    Preserves resume/audit provability fields without embedding tensor state.
    Superset of the historical private helper: adds ``updater_config_sha256``.
    """
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
        "updater_config_sha256": payload.get("updater_config_sha256"),
        "tensor_summary_count": len(payload.get("tensor_summaries") or {}),
        "tensor_keys": sorted((payload.get("tensor_summaries") or {}).keys()),
    }


def _checkpoint_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_checkpoint_payload_summary(payload)


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


def _sparse_vote_events(votes: torch.Tensor) -> SparseVoteEvents:
    return SparseVoteEvents.from_dense_votes(votes)


SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY = "fused_only"
SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON = "oracle_on"
_SPARSE_VOTE_AUTHORITY_MODES = frozenset(
    {
        SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    }
)


def normalize_sparse_vote_authority_mode(
    sparse_vote_authority_mode: object = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> str:
    """Fail-closed mode normalizer. Omitted default is fused_only; None is REJECT."""

    if sparse_vote_authority_mode is None:
        raise TypeError(
            "sparse_vote_authority_mode must not be None; "
            "omit the keyword for default fused_only"
        )
    if not isinstance(sparse_vote_authority_mode, str):
        raise TypeError(
            "sparse_vote_authority_mode must be str, "
            f"got {type(sparse_vote_authority_mode).__name__}"
        )
    if sparse_vote_authority_mode not in _SPARSE_VOTE_AUTHORITY_MODES:
        raise ValueError(
            "sparse_vote_authority_mode must be one of "
            f"{sorted(_SPARSE_VOTE_AUTHORITY_MODES)}, got {sparse_vote_authority_mode!r}"
        )
    return sparse_vote_authority_mode


def _dense_votes_from_sparse_events(
    events: SparseVoteEvents,
    *,
    shape: Sequence[int] | torch.Size,
) -> torch.Tensor:
    """Votes-only reconstruction for oracle proof — NEVER production apply input."""

    dense = torch.zeros(tuple(int(x) for x in shape), dtype=torch.int16).reshape(-1)
    if events.event_count() > 0:
        dense[events.indices] = events.values
    return dense.reshape(tuple(int(x) for x in shape)).contiguous()


def resolve_sparse_vote_authority_path(
    *,
    weighted_grad_by_key: Mapping[str, torch.Tensor],
    q_levels_by_key: Mapping[str, torch.Tensor],
    rank_spec: Any,
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> dict[str, Any]:
    """Shared B1/B2/B3 mode facade: fused producer authoritative; oracle_on opt-in only.

    Returns path-derived mode discriminator fields. Callers MUST pass
    votes_by_key=None + sparse_vote_authority_only=True into apply and MUST set
    receipt sparse_vote_authority_mode from resolved_mode (never independently).
    """

    resolved_mode = normalize_sparse_vote_authority_mode(sparse_vote_authority_mode)
    if set(weighted_grad_by_key) != set(q_levels_by_key):
        raise ValueError("weighted_grad_by_key and q_levels_by_key must have identical keys")

    sparse_events_by_key: dict[str, SparseVoteEvents] = {}
    for key in sorted(weighted_grad_by_key):
        sparse_events_by_key[key] = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
            weighted_grad_by_key[key],
            q_levels_by_key[key],
            rank_spec,
        )

    oracle_only: dict[str, Any] | None = None
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON:
        dense_derived_by_key: dict[str, SparseVoteEvents] = {}
        events_equal_by_key: dict[str, bool] = {}
        for key in sorted(weighted_grad_by_key):
            weighted_grad = weighted_grad_by_key[key]
            credit = credit_from_weighted_grad(weighted_grad)
            moves = project_s1_gradient_to_moves(weighted_grad, q_levels_by_key[key])
            votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
            dense_derived = _sparse_vote_events(votes)
            dense_derived_by_key[key] = dense_derived
            fused = sparse_events_by_key[key]
            events_equal_by_key[key] = bool(
                fused.event_count() == dense_derived.event_count()
                and (
                    fused.event_count() == 0
                    or (
                        torch.equal(fused.indices, dense_derived.indices)
                        and torch.equal(fused.values, dense_derived.values)
                    )
                )
            )
        oracle_only = {
            "dense_derived_sparse_events_by_key": dense_derived_by_key,
            "events_equal_fused_vs_dense_derived": all(events_equal_by_key.values()),
            "events_equal_by_key": events_equal_by_key,
            "dense_reference_tagged": "oracle_only",
        }

    # PLAN_v7 S1: centralized binding maps + diagnostic interval on path (option 2).
    shape_by_key = logical_shape_by_key_from_q_levels(q_levels_by_key)
    named_bindings = build_named_receipt_path_bindings(
        sparse_events_by_key=sparse_events_by_key,
        logical_shape_by_key=shape_by_key,
        oracle_only=oracle_only,
        resolved_mode=resolved_mode,
    )

    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_path_resolved_mode(resolved_mode)
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_named_receipt_bindings(named_bindings)

    return {
        "resolved_mode": resolved_mode,
        "sparse_vote_authority_mode": resolved_mode,  # path-derived discriminator
        "sparse_events_by_key": sparse_events_by_key,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "candidate_oracle_control_enabled": False,
        "votes_by_key_applied": None,
        "oracle_only": oracle_only,
        "transient_over2_tensors_fused_only": ("weighted_grad",),
        "sparse_event_map_binding_sha256_by_key": named_bindings[
            "sparse_event_map_binding_sha256_by_key"
        ],
        "sparse_event_count_by_key": named_bindings["sparse_event_count_by_key"],
        "sparse_event_logical_shape_by_key": named_bindings[
            "sparse_event_logical_shape_by_key"
        ],
        "s1_binding_interval_seconds": named_bindings["s1_binding_interval_seconds"],
        "oracle_only_serializable": named_bindings["oracle_only_serializable"],
    }


def _fused_or_oracle_transient_over2(resolved_mode: str) -> tuple[str, ...]:
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY:
        return ("weighted_grad",)
    return (
        "weighted_grad",
        "decoded_bounded_accumulator_for_exact_oracle_control",
        "dense_oracle_qacc_reference_result",
    )


def _build_vote_projection_proof(
    *,
    rank_spec: Any,
    update_spec: VoteUpdateSpec,
    resolved_mode: str,
    total_sparse_events: int,
    oracle_only: Any,
    sparse_event_map_binding_sha256_by_key: Mapping[str, str] | None = None,
    sparse_event_count_by_key: Mapping[str, int] | None = None,
    sparse_event_logical_shape_by_key: Mapping[str, Any] | None = None,
    s1_binding_interval_seconds: float | None = None,
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "rank_vote_spec": rank_spec.to_live_dict(),
        "vote_update_spec": asdict(update_spec),
        "candidate_mode": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "candidate_sparse_vote_events_only": True,
        "dense_vote_authority_persisted": False,
        "sparse_vote_authority_mode": resolved_mode,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "votes_by_key_applied": None,
        "candidate_oracle_control_enabled": False,
        "total_sparse_vote_event_count": int(total_sparse_events),
        "transient_over2_tensors": list(_fused_or_oracle_transient_over2(resolved_mode)),
    }
    # PLAN_v7 B1: consume exact path-returned maps (no recompute).
    if sparse_event_map_binding_sha256_by_key is None:
        raise ValueError(
            "vote_projection_proof requires path-returned sparse_event_map_binding_sha256_by_key"
        )
    if sparse_event_count_by_key is None or sparse_event_logical_shape_by_key is None:
        raise ValueError("vote_projection_proof requires path-returned count/shape maps")
    if s1_binding_interval_seconds is None:
        raise ValueError("vote_projection_proof requires path-returned s1_binding_interval_seconds")
    binding = {
        str(k): require_lowercase_sha256_hex(
            v, field=f"B1.sparse_event_map_binding_sha256_by_key[{k}]"
        )
        for k, v in sorted(sparse_event_map_binding_sha256_by_key.items())
    }
    counts = {str(k): int(v) for k, v in sorted(sparse_event_count_by_key.items())}
    shapes = {
        str(k): [int(d) for d in v]
        for k, v in sorted(sparse_event_logical_shape_by_key.items())
    }
    if not (set(binding) == set(counts) == set(shapes)):
        raise ValueError("B1 path map key-set invariant failed")
    proof["sparse_event_map_binding_sha256_by_key"] = binding
    proof["sparse_event_count_by_key"] = counts
    proof["sparse_event_logical_shape_by_key"] = shapes
    proof["s1_binding_interval_seconds_diagnostic"] = require_finite_nonnegative_interval(
        s1_binding_interval_seconds, field="s1_binding_interval_seconds_diagnostic"
    )
    # ABSENCE policy: oracle_only key present only under oracle_on.
    # Consume JSON-safe serializable projection (never raw SparseVoteEvents/tensors).
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON:
        if not isinstance(oracle_only, Mapping):
            raise ValueError(
                "B1 oracle_on requires path-returned oracle_only_serializable mapping"
            )
        proof["oracle_only"] = {
            "events_equal_by_key": {
                str(k): bool(v)
                for k, v in sorted(dict(oracle_only.get("events_equal_by_key") or {}).items())
            },
            "events_equal_fused_vs_dense_derived": bool(
                oracle_only["events_equal_fused_vs_dense_derived"]
            ),
            "dense_reference_tagged": "oracle_only",
        }
    validate_named_receipt_evidence_maps(
        proof, resolved_mode=resolved_mode, require_oracle_only_key=True
    )
    return proof


def validate_sparse_vote_authority_mode_matches_execution_path(
    receipt_fields: Mapping[str, Any],
    *,
    resolved_mode: str,
) -> None:
    """Reject caller-authored discriminator that disagrees with path-resolved mode.

    resolved_mode MUST come from an independent path witness / facade result — never
    from the claimed receipt field being validated (no self-compare).
    """

    claimed = receipt_fields.get("sparse_vote_authority_mode")
    if claimed != resolved_mode:
        raise ValueError(
            "sparse_vote_authority_mode discriminator mismatch: "
            f"claimed={claimed!r} resolved={resolved_mode!r}"
        )
    if receipt_fields.get("sparse_vote_authority_only") is not True:
        raise ValueError("sparse_vote_authority_only must be True on production path")
    if receipt_fields.get("votes_by_key_applied") is not None:
        raise ValueError("votes_by_key_applied must be None on sparse authority path")
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY:
        if "oracle_only" in receipt_fields and receipt_fields.get("oracle_only") is not None:
            raise ValueError("oracle_only must be ABSENT/None under fused_only")
        tot = receipt_fields.get("transient_over2_tensors")
        if list(tot) != ["weighted_grad"]:
            raise ValueError(
                "fused_only transient_over2_tensors must be exactly ['weighted_grad'], "
                f"got {tot!r}"
            )


def _path_witness_token(
    *,
    resolved_mode: str,
    execution_nonce: str,
    weighted_grad_capture_sha256_by_key: Mapping[str, str],
) -> str:
    """Non-caller-authored path witness bound at execution time."""

    payload = {
        "resolved_mode": resolved_mode,
        "execution_nonce": execution_nonce,
        "weighted_grad_capture_sha256_by_key": {
            str(k): str(v)
            for k, v in sorted(weighted_grad_capture_sha256_by_key.items())
        },
        "schema": "sparse_vote_path_witness_v1",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


class SparseVoteExecutionWitness:
    """Execution-scoped counters + path-mode observation (findings 2/A/B).

    Validity requires forward_backward_count==1 AND update_count==1 (no max()).
    resolved_mode is observed only at resolve_sparse_vote_authority_path.
    """

    def __init__(self) -> None:
        self.forward_backward_count = 0
        self.update_count = 0
        self.weighted_grad_capture_sha256_by_key: dict[str, str] = {}
        self.path_resolved_mode_observations: list[str] = []
        # PLAN_v7 named-receipt relay (copied from path maps at resolve when active).
        self.named_receipt_bindings_observed: bool = False
        self.sparse_event_map_binding_sha256_by_key: dict[str, str] = {}
        self.sparse_event_count_by_key: dict[str, int] = {}
        self.sparse_event_logical_shape_by_key: dict[str, list[int]] = {}
        self.oracle_only_serializable: dict[str, Any] | None = None
        self.s1_binding_interval_seconds: float | None = None

    def note_named_receipt_bindings(self, named_bindings: Mapping[str, Any]) -> None:
        """Copy path-returned named receipt maps for B3 subproof assembly.

        Second note is hard-rejected (fail-closed; no silent overwrite of any field).
        """
        if self.named_receipt_bindings_observed:
            raise ValueError(
                "named_receipt_bindings already noted on witness "
                "(second note forbidden; no overwrite)"
            )
        binding = {
            str(k): require_lowercase_sha256_hex(
                v, field=f"witness.sparse_event_map_binding_sha256_by_key[{k}]"
            )
            for k, v in sorted(
                dict(named_bindings["sparse_event_map_binding_sha256_by_key"]).items()
            )
        }
        counts = {
            str(k): int(v)
            for k, v in sorted(dict(named_bindings["sparse_event_count_by_key"]).items())
        }
        shapes = {
            str(k): [int(d) for d in v]
            for k, v in sorted(
                dict(named_bindings["sparse_event_logical_shape_by_key"]).items()
            )
        }
        if not (set(binding) == set(counts) == set(shapes)):
            raise ValueError("witness named-receipt key-set invariant failed")
        interval = require_finite_nonnegative_interval(
            named_bindings["s1_binding_interval_seconds"],
            field="s1_binding_interval_seconds",
        )
        serializable = named_bindings.get("oracle_only_serializable")
        if serializable is not None and not isinstance(serializable, Mapping):
            raise ValueError("oracle_only_serializable must be mapping or None")
        self.named_receipt_bindings_observed = True
        self.sparse_event_map_binding_sha256_by_key = binding
        self.sparse_event_count_by_key = counts
        self.sparse_event_logical_shape_by_key = shapes
        self.oracle_only_serializable = (
            dict(serializable) if isinstance(serializable, Mapping) else None
        )
        self.s1_binding_interval_seconds = interval

    def note_forward_backward(self, weighted_grad_by_key: Mapping[str, torch.Tensor]) -> None:
        self.forward_backward_count += 1
        for key, tensor in sorted(weighted_grad_by_key.items()):
            self.weighted_grad_capture_sha256_by_key[str(key)] = tensor_sha256(
                tensor.detach()
            )

    def note_update(self) -> None:
        self.update_count += 1

    def note_path_resolved_mode(self, resolved_mode: str) -> None:
        mode = str(resolved_mode)
        if self.path_resolved_mode_observations and mode != self.path_resolved_mode_observations[-1]:
            raise ValueError(
                "conflicting path_resolved_mode observations: "
                f"{self.path_resolved_mode_observations[-1]!r} vs {mode!r}"
            )
        self.path_resolved_mode_observations.append(mode)

    def witnessed_path_resolved_mode(self) -> str:
        if not self.path_resolved_mode_observations:
            raise ValueError("path_resolved_mode never observed at resolve site")
        if len(self.path_resolved_mode_observations) != 1:
            # multiple identical ok if same; conflict already rejected
            # require exactly one observation event for one-execution claim
            if len(set(self.path_resolved_mode_observations)) != 1:
                raise ValueError("multiple distinct path_resolved_mode observations")
            # still require exactly one observation for strict one-execution
            if len(self.path_resolved_mode_observations) != 1:
                raise ValueError(
                    "path_resolved_mode observation count must be 1 for one-execution, "
                    f"got {len(self.path_resolved_mode_observations)}"
                )
        return self.path_resolved_mode_observations[0]

    @property
    def forward_backward_update_call_count(self) -> int:
        # kept for legacy field; validity uses both counts == 1 independently
        if self.forward_backward_count == 1 and self.update_count == 1:
            return 1
        return int(self.forward_backward_count + self.update_count)


# process-local active witness (tests may install); core always creates fresh
_ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS: SparseVoteExecutionWitness | None = None
_PHASE_EMITTER: Any | None = None  # optional (kind, phase) -> None for GPU phase hooks

def _emit_phase(kind: str, phase: str) -> None:
    emitter = _PHASE_EMITTER
    if emitter is not None:
        emitter(kind, phase)


def p1b_receipt_canonical_sha256(receipt: "TrainerSub2AuthorityLiveConversionReceipt") -> str:
    """Canonical digest of unchanged P1b receipt (S2 digest-bound one-execution)."""

    payload = receipt.to_dict() if hasattr(receipt, "to_dict") else asdict(receipt)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class SparseVoteAuthorityLandingReceipt:
    """Task-local wrapper around unchanged P1b live conversion receipt (Q2/S2)."""

    schema_version: str
    slice_readiness_claim: bool
    sparse_vote_authority_subproof: dict[str, Any]
    p1b_live_conversion_receipt: TrainerSub2AuthorityLiveConversionReceipt
    p1b_receipt_sha256: str
    core_execution_identity: dict[str, Any]
    plan_sha256: str
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slice_readiness_claim": bool(self.slice_readiness_claim),
            "sparse_vote_authority_subproof": dict(self.sparse_vote_authority_subproof),
            "p1b_live_conversion_receipt": self.p1b_live_conversion_receipt.to_dict(),
            "p1b_receipt_sha256": self.p1b_receipt_sha256,
            "core_execution_identity": dict(self.core_execution_identity),
            "plan_sha256": self.plan_sha256,
            "task_id": self.task_id,
        }


def validate_sparse_vote_authority_landing_receipt(
    receipt: SparseVoteAuthorityLandingReceipt,
    *,
    allow_legacy_without_named_evidence: bool = False,
) -> None:
    if receipt.schema_version != "sparse_vote_authority_landing_receipt_v1":
        raise ValueError("landing receipt schema_version mismatch")
    if receipt.slice_readiness_claim is not False:
        raise ValueError("slice_readiness_claim must be hard-false on landing receipt")
    recomputed = p1b_receipt_canonical_sha256(receipt.p1b_live_conversion_receipt)
    if recomputed != receipt.p1b_receipt_sha256:
        raise ValueError("p1b_receipt_sha256 mismatch vs embedded P1b payload")
    core_id = dict(receipt.core_execution_identity)
    sub_id = dict(receipt.sparse_vote_authority_subproof.get("execution_identity") or {})
    for bag, label in ((core_id, "core"), (sub_id, "subproof")):
        if str(bag.get("p1b_receipt_sha256")) != recomputed:
            raise ValueError(f"{label} execution_identity.p1b_receipt_sha256 mismatch")
        if int(bag.get("forward_backward_count", -1)) != 1:
            raise ValueError(f"{label} forward_backward_count must be 1")
        if int(bag.get("update_count", -1)) != 1:
            raise ValueError(f"{label} update_count must be 1")
        if int(bag.get("forward_backward_update_call_count", -1)) != 1:
            raise ValueError(f"{label} forward_backward_update_call_count must be 1")
    # Independent resolved mode from path observation at resolve site — NOT from kwarg/claim
    resolved_mode = str(core_id.get("path_resolved_mode") or "")
    if not resolved_mode:
        raise ValueError("core_execution_identity missing path_resolved_mode witness")
    if int(core_id.get("path_resolved_mode_observation_count", 0)) != 1:
        raise ValueError("path_resolved_mode observation count must be 1")
    expected_token = _path_witness_token(
        resolved_mode=resolved_mode,
        execution_nonce=str(core_id.get("execution_nonce") or ""),
        weighted_grad_capture_sha256_by_key=dict(
            core_id.get("weighted_grad_capture_sha256_by_key") or {}
        ),
    )
    if str(core_id.get("path_witness_token") or "") != expected_token:
        raise ValueError("path_witness_token mismatch vs execution artifacts")
    if str(sub_id.get("path_witness_token") or "") != expected_token:
        raise ValueError("subproof path_witness_token mismatch")
    # claimed discriminator vs independent path witness
    validate_sparse_vote_authority_mode_matches_execution_path(
        receipt.sparse_vote_authority_subproof,
        resolved_mode=resolved_mode,
    )
    wg = core_id.get("weighted_grad_capture_sha256_by_key")
    if not isinstance(wg, dict) or not wg:
        raise ValueError("weighted_grad_capture_sha256_by_key must be non-empty per-key map")
    forbidden_pollution = {"total_sparse_vote_event_count", "post_resume_payload_sha256_after"}
    if forbidden_pollution & set(wg):
        raise ValueError(
            "weighted_grad_capture_sha256_by_key must not contain pollution keys "
            f"{forbidden_pollution}"
        )
    # PLAN_v7 / gate-2 C3+D3: named-receipt evidence on DESERIALIZED subproof.
    # Default requires maps; legacy fixtures must opt in explicitly.
    validate_named_receipt_evidence_maps(
        receipt.sparse_vote_authority_subproof,
        resolved_mode=resolved_mode,
        require_oracle_only_key=True,
        allow_legacy_without_named_evidence=bool(allow_legacy_without_named_evidence),
    )


def assemble_b2_post_resume_update_proof(
    *,
    path: Mapping[str, Any],
    loss_finite: bool,
    total_sparse_events: int,
    step_result_global_summary: Mapping[str, Any],
    post_resume_mutated: bool,
) -> dict[str, Any]:
    """Pure B2 post_resume_update_proof assembly (gate-2 D1 production helper).

    Consumes path-returned maps + oracle_only_serializable; no SparseVoteEvents/tensors.
    """
    resolved_mode = str(path["resolved_mode"])
    proof: dict[str, Any] = {
        "loss_finite": bool(loss_finite),
        "candidate_mode": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "total_sparse_vote_event_count": int(total_sparse_events),
        "q_changed_count": int(step_result_global_summary.get("q_changed_count", 0)),
        "candidate_local_update_pass": bool(
            step_result_global_summary.get("candidate_local_update_pass")
        ),
        "candidate_dense_decode_used": bool(
            step_result_global_summary.get("candidate_dense_decode_used")
        ),
        "candidate_dense_vote_authority_used": bool(
            step_result_global_summary.get("candidate_dense_vote_authority_used")
        ),
        "post_resume_update_mutated_resumed_sub2_authority": bool(post_resume_mutated),
        # full frozen discriminator / compartment set (plan b_site_receipt_propagation.B2)
        "sparse_vote_authority_mode": resolved_mode,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "votes_by_key_applied": None,
        "candidate_oracle_control_enabled": False,
        "transient_over2_tensors": list(_fused_or_oracle_transient_over2(resolved_mode)),
        # PLAN_v7 B2: exact path-returned maps (no recompute).
        "sparse_event_map_binding_sha256_by_key": {
            str(k): require_lowercase_sha256_hex(
                v, field=f"B2.sparse_event_map_binding_sha256_by_key[{k}]"
            )
            for k, v in sorted(
                dict(path["sparse_event_map_binding_sha256_by_key"]).items()
            )
        },
        "sparse_event_count_by_key": {
            str(k): int(v)
            for k, v in sorted(dict(path["sparse_event_count_by_key"]).items())
        },
        "sparse_event_logical_shape_by_key": {
            str(k): [int(d) for d in v]
            for k, v in sorted(
                dict(path["sparse_event_logical_shape_by_key"]).items()
            )
        },
        "s1_binding_interval_seconds_diagnostic": require_finite_nonnegative_interval(
            path["s1_binding_interval_seconds"],
            field="s1_binding_interval_seconds_diagnostic",
        ),
    }
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON:
        serializable = path.get("oracle_only_serializable")
        if not isinstance(serializable, Mapping):
            raise ValueError("B2 oracle_on requires path oracle_only_serializable")
        # JSON-safe serializable only (no SparseVoteEvents/tensors)
        proof["oracle_only"] = dict(serializable)
    validate_named_receipt_evidence_maps(
        proof,
        resolved_mode=resolved_mode,
        require_oracle_only_key=True,
    )
    return proof


def assemble_b3_named_receipt_subproof_fields(
    witness: SparseVoteExecutionWitness,
    *,
    resolved_mode: str,
) -> dict[str, Any]:
    """Pure B3 named-receipt fields from execution witness (gate-2 D2 production helper).

    Fail-closed if witness never noted bindings. Used by run_sparse_vote_authority_b3_core
    and hostile missing-relay tests.
    """
    if not bool(witness.named_receipt_bindings_observed):
        raise ValueError(
            "B3 landing subproof missing named_receipt_bindings on execution witness "
            "(resolve must note bindings under active witness)"
        )
    fields: dict[str, Any] = {
        "sparse_event_map_binding_sha256_by_key": dict(
            witness.sparse_event_map_binding_sha256_by_key
        ),
        "sparse_event_count_by_key": dict(witness.sparse_event_count_by_key),
        "sparse_event_logical_shape_by_key": dict(
            witness.sparse_event_logical_shape_by_key
        ),
    }
    if witness.s1_binding_interval_seconds is None:
        raise ValueError("B3 missing s1_binding_interval_seconds on witness")
    fields["s1_binding_interval_seconds_diagnostic"] = (
        require_finite_nonnegative_interval(
            witness.s1_binding_interval_seconds,
            field="s1_binding_interval_seconds_diagnostic",
        )
    )
    mode = str(resolved_mode)
    if mode == SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON:
        serializable = witness.oracle_only_serializable
        if not isinstance(serializable, Mapping) or "events_equal_by_key" not in serializable:
            raise ValueError(
                "B3 oracle_on requires real events_equal_by_key map on witness "
                "(tag-only stub forbidden)"
            )
        eq_map = {
            str(k): bool(v)
            for k, v in sorted(dict(serializable["events_equal_by_key"]).items())
        }
        if set(eq_map) != set(witness.sparse_event_map_binding_sha256_by_key):
            raise ValueError("B3 oracle events_equal_by_key key-set must match binding keys")
        fields["oracle_only"] = {
            "events_equal_by_key": eq_map,
            "events_equal_fused_vs_dense_derived": bool(
                serializable.get("events_equal_fused_vs_dense_derived")
            ),
            "dense_reference_tagged": "oracle_only",
        }
    elif witness.oracle_only_serializable is not None:
        raise ValueError(
            "fused_only ABSENCE violated: oracle_only_serializable present on witness"
        )
    validate_named_receipt_evidence_maps(
        fields,
        resolved_mode=mode,
        require_oracle_only_key=True,
    )
    return fields


def run_sparse_vote_authority_b3_core(
    **kwargs: Any,
) -> tuple[TrainerSub2AuthorityLiveConversionReceipt, dict[str, Any], dict[str, Any]]:
    """ONE execution core: returns (p1b_receipt, sparse_subproof, core_execution_identity).

    Public P1b builder returns only the first element; wrapper wraps all three.
    """

    global _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS
    witness = SparseVoteExecutionWitness()
    prior = _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS
    _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS = witness
    try:
        sparse_mode = kwargs.get(
            "sparse_vote_authority_mode", SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY
        )
        p1b = _build_trainer_sub2_authority_live_conversion_receipt_impl(**kwargs)
    finally:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS = prior

    digest = p1b_receipt_canonical_sha256(p1b)
    # executed mode from path observation at resolve site (B) — kwarg is request only
    resolved_mode = witness.witnessed_path_resolved_mode()
    wg_map = dict(witness.weighted_grad_capture_sha256_by_key)
    fb = int(witness.forward_backward_count)
    upd = int(witness.update_count)
    call_count = 1 if (fb == 1 and upd == 1) else int(fb + upd)
    execution_nonce = hashlib.sha256(
        f"{digest}:{p1b.post_resume_payload_sha256_after}:{resolved_mode}:{fb}:{upd}".encode()
    ).hexdigest()
    path_token = _path_witness_token(
        resolved_mode=resolved_mode,
        execution_nonce=execution_nonce,
        weighted_grad_capture_sha256_by_key=wg_map,
    )
    core_execution_identity = {
        "execution_nonce": execution_nonce,
        "forward_backward_update_call_count": call_count,
        "forward_backward_count": fb,
        "update_count": upd,
        "weighted_grad_capture_sha256_by_key": wg_map,
        "post_update_payload_sha256": str(p1b.post_resume_payload_sha256_after),
        "p1b_receipt_sha256": digest,
        "path_resolved_mode": resolved_mode,
        "path_resolved_mode_observation_count": len(witness.path_resolved_mode_observations),
        "path_witness_token": path_token,
        "requested_sparse_vote_authority_mode": normalize_sparse_vote_authority_mode(
            sparse_mode
        ),
    }
    subproof: dict[str, Any] = {
        "sparse_vote_authority_mode": resolved_mode,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "votes_by_key_applied": None,
        "candidate_oracle_control_enabled": False,
        "fused_sparse_event_count_total": int(p1b.total_sparse_vote_event_count),
        "mutation_witness": {
            "post_resume_update_mutated": bool(p1b.post_resume_update_mutated),
            "q_changed_count": int(p1b.q_changed_count),
        },
        "transient_over2_tensors": ["weighted_grad"],
        "execution_identity": dict(core_execution_identity),
    }
    # PLAN_v7 B3 / gate-2 D2: pure helper owns witness→subproof named-receipt assembly.
    subproof.update(
        assemble_b3_named_receipt_subproof_fields(
            witness, resolved_mode=resolved_mode
        )
    )
    return p1b, subproof, core_execution_identity


def build_sparse_vote_authority_landing_receipt(
    *,
    plan_sha256: str,
    task_id: str,
    **kwargs: Any,
) -> SparseVoteAuthorityLandingReceipt:
    # R2: single B3 emission pair — started after update in
    # _run_live_p1_vote_carrier_subproof; closed here after full receipt
    # construct + validate (covers P1b receipt/validation too).
    p1b, subproof, core_id = run_sparse_vote_authority_b3_core(**kwargs)
    receipt = SparseVoteAuthorityLandingReceipt(
        schema_version="sparse_vote_authority_landing_receipt_v1",
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=subproof,
        p1b_live_conversion_receipt=p1b,
        p1b_receipt_sha256=str(core_id["p1b_receipt_sha256"]),
        core_execution_identity=core_id,
        plan_sha256=str(plan_sha256),
        task_id=str(task_id),
    )
    validate_sparse_vote_authority_landing_receipt(receipt)
    _emit_phase("PHASE_END", "emission")
    return receipt


def _eligible_master_hashes(eligible_modules: Mapping[str, BitLinear]) -> dict[str, str]:
    return {
        str(key): tensor_sha256(module.weight.detach())
        for key, module in sorted(eligible_modules.items())
    }


def _eligible_weight_state_keys(eligible_modules: Mapping[str, BitLinear]) -> tuple[str, ...]:
    return tuple(f"{key}.weight" for key in sorted(str(item) for item in eligible_modules))


def _tensor_sha_or_none(value: torch.Tensor | None) -> str | None:
    return tensor_sha256(value) if value is not None else None


def _roundtrip_tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    if cpu.dtype == torch.bfloat16:
        h.update(cpu.view(torch.int16).numpy().tobytes())
    else:
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _roundtrip_canonicalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": _roundtrip_tensor_sha256(tensor),
        }
    if isinstance(value, Mapping):
        return {str(key): _roundtrip_canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_roundtrip_canonicalize(item) for item in value]
    return value


def _roundtrip_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = _roundtrip_canonicalize(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


TRAINER_SUB2_ROUNDTRIP_TARGET_NAME = "step2c4a_trainer_authority_checkpoint_roundtrip"
PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV = (
    "HRM_TEXT_158_PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED"
)
PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV = (
    "HRM_TEXT_158_PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED"
)
W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY = "w6_byte_packed_accumulator_persisted"
W6_BYTE_PACKED_PAYLOAD_KEY = "w6_byte_packed_payload"
W6_BYTE_PACKED_SCHEMA_KEY = "w6_byte_packed_schema"
W6_BYTE_PACKED_LOGICAL_SHAPE_KEY = "w6_byte_packed_logical_shape"
W6_BYTE_PACKED_LOGICAL_NUMEL_KEY = "w6_byte_packed_logical_numel"
W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY = "w5_byte_packed_accumulator_persisted"
W5_BYTE_PACKED_PAYLOAD_KEY = "w5_byte_packed_payload"
W5_BYTE_PACKED_SCHEMA_KEY = "w5_byte_packed_schema"
W5_BYTE_PACKED_LOGICAL_SHAPE_KEY = "w5_byte_packed_logical_shape"
W5_BYTE_PACKED_LOGICAL_NUMEL_KEY = "w5_byte_packed_logical_numel"
EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY = "event_coded_live_carrier_persisted"
EVENT_CODED_LIVE_CARRIER_SAVED_KEY = "event_coded_live_carrier_saved"
EVENT_CODED_LIVE_CARRIER_SCHEMA_KEY = "event_coded_live_carrier_schema"
EVENT_CODED_LIVE_CARRIER_FORMAT_KEY = "event_coded_live_carrier_format"
EVENT_CODED_EVENTS_PACKED_KEY = "event_coded_events_packed"
EVENT_CODED_BACKLOG_PACKED_KEY = "event_coded_backlog_packed"
EVENT_CODED_HOT_EXACT_PACKED_KEY = "event_coded_hot_exact_packed"
EVENT_CODED_EVENT_COUNT_KEY = "event_coded_event_count"
EVENT_CODED_BACKLOG_COUNT_KEY = "event_coded_backlog_entry_count"
EVENT_CODED_HOT_EXACT_ROW_COUNT_KEY = "event_coded_hot_exact_row_count"
EVENT_CODED_LOGICAL_NUMEL_KEY = "event_coded_logical_numel"
EVENT_CODED_DEMOTION_BAND_KEY = "event_coded_demotion_band"
PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV = "HRM_TEXT_158_PERSISTENT_Q_TERNARY_BYTE_PACKED"
Q_TERNARY_BYTE_PACKED_PERSISTED_KEY = "q_ternary_byte_packed_persisted"
Q_TERNARY_PACKED_PAYLOAD_KEY = "q_ternary_packed_payload"
Q_TERNARY_PACKED_SCHEMA_KEY = "q_ternary_packed_schema"
Q_TERNARY_LOGICAL_SHAPE_KEY = "q_ternary_logical_shape"
Q_TERNARY_LOGICAL_NUMEL_KEY = "q_ternary_logical_numel"
Q_TERNARY_PADDING_VALUES_KEY = "q_ternary_padding_values"
Q_TERNARY_BYTE_PACKED_PERSISTED_SAVED_KEY = "q_ternary_byte_packed_persisted_saved"
PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV = "HRM_TEXT_158_PERSISTENT_Q_TERNARY_BASE3_CODEC"
Q_CODEC_SELECTOR_2BIT = "2bit"
Q_CODEC_SELECTOR_BASE3 = "base3"
RAW_Q_LEVELS_FIELD = "q_levels"


def persistent_w6_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV) == "1"


def persistent_w5_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV) == "1"


def persistent_q_ternary_byte_packed_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV) == "1"


def persistent_q_ternary_base3_codec_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV) == "1"


def _event_coded_live_carrier_payload_from_state(
    carrier: EventCodedAccLiveState,
) -> dict[str, Any]:
    packed = carrier.to_checkpoint_payload()
    if str(packed.schema) != EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1:
        raise ValueError(
            "event-coded live carrier checkpoint requires schema "
            f"{EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1!r}"
        )
    return {
        EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY: True,
        EVENT_CODED_LIVE_CARRIER_SCHEMA_KEY: str(packed.schema),
        EVENT_CODED_LIVE_CARRIER_FORMAT_KEY: str(packed.format),
        EVENT_CODED_EVENTS_PACKED_KEY: packed.events_packed.detach().cpu().contiguous(),
        EVENT_CODED_BACKLOG_PACKED_KEY: packed.backlog_packed.detach().cpu().contiguous(),
        EVENT_CODED_HOT_EXACT_PACKED_KEY: packed.hot_exact_packed.detach().cpu().contiguous(),
        EVENT_CODED_EVENT_COUNT_KEY: int(packed.event_count),
        EVENT_CODED_BACKLOG_COUNT_KEY: int(packed.backlog_entry_count),
        EVENT_CODED_HOT_EXACT_ROW_COUNT_KEY: int(packed.hot_exact_row_count),
        EVENT_CODED_LOGICAL_NUMEL_KEY: int(packed.logical_numel),
        EVENT_CODED_DEMOTION_BAND_KEY: int(carrier.demotion_band),
    }


def _packed_event_coded_from_roundtrip_payload(
    payload: Mapping[str, Any],
) -> PackedEventCodedAccState:
    events_packed = payload.get(EVENT_CODED_EVENTS_PACKED_KEY)
    backlog_packed = payload.get(EVENT_CODED_BACKLOG_PACKED_KEY)
    hot_exact_packed = payload.get(EVENT_CODED_HOT_EXACT_PACKED_KEY)
    if not isinstance(events_packed, torch.Tensor):
        raise ValueError("2C4a event-coded sidecar missing event_coded_events_packed tensor")
    if not isinstance(backlog_packed, torch.Tensor):
        raise ValueError("2C4a event-coded sidecar missing event_coded_backlog_packed tensor")
    if not isinstance(hot_exact_packed, torch.Tensor):
        raise ValueError("2C4a event-coded sidecar missing event_coded_hot_exact_packed tensor")
    return PackedEventCodedAccState(
        events_packed=events_packed.detach().cpu().contiguous(),
        backlog_packed=backlog_packed.detach().cpu().contiguous(),
        logical_numel=int(payload[EVENT_CODED_LOGICAL_NUMEL_KEY]),
        event_count=int(payload[EVENT_CODED_EVENT_COUNT_KEY]),
        backlog_entry_count=int(payload[EVENT_CODED_BACKLOG_COUNT_KEY]),
        schema=str(payload[EVENT_CODED_LIVE_CARRIER_SCHEMA_KEY]),
        format=str(
            payload.get(EVENT_CODED_LIVE_CARRIER_FORMAT_KEY, PACKED_EVENT_CODED_ACC_FORMAT)
        ),
        hot_exact_packed=hot_exact_packed.detach().cpu().contiguous(),
        hot_exact_row_count=int(payload[EVENT_CODED_HOT_EXACT_ROW_COUNT_KEY]),
    )


def resolve_q_codec_selector(*, q_codec_selector: str | None = None) -> str:
    if q_codec_selector is not None:
        selector = str(q_codec_selector)
        if selector not in (Q_CODEC_SELECTOR_2BIT, Q_CODEC_SELECTOR_BASE3):
            raise ValueError(
                f"q_codec_selector must be {Q_CODEC_SELECTOR_2BIT!r} or "
                f"{Q_CODEC_SELECTOR_BASE3!r}, got {selector!r}"
            )
        return selector
    if persistent_q_ternary_base3_codec_enabled():
        return Q_CODEC_SELECTOR_BASE3
    return Q_CODEC_SELECTOR_2BIT


def _pack_q_for_checkpoint(
    q_levels_logical: torch.Tensor,
    *,
    q_packed_enabled: bool | None,
    q_codec_selector: str | None = None,
) -> dict[str, Any]:
    selector = resolve_q_codec_selector(q_codec_selector=q_codec_selector)
    master_enabled = persistent_q_ternary_byte_packed_enabled(enabled=q_packed_enabled)
    if selector == Q_CODEC_SELECTOR_BASE3 and not master_enabled:
        raise ValueError(
            "base-3 q codec selector requires "
            f"{PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV}=1 before checkpoint save"
        )
    if not master_enabled:
        return {
            RAW_Q_LEVELS_FIELD: q_levels_logical,
            Q_TERNARY_BYTE_PACKED_PERSISTED_KEY: False,
        }
    if selector == Q_CODEC_SELECTOR_BASE3:
        packed_q = pack_ternary_q_base3_5perbyte_reference(q_levels_logical)
    else:
        packed_q = pack_ternary_q_2bit_reference(q_levels_logical)
    return {
        Q_TERNARY_BYTE_PACKED_PERSISTED_KEY: True,
        Q_TERNARY_PACKED_SCHEMA_KEY: str(packed_q.format),
        Q_TERNARY_PACKED_PAYLOAD_KEY: packed_q.packed.detach().cpu().contiguous(),
        Q_TERNARY_LOGICAL_SHAPE_KEY: list(packed_q.logical_shape),
        Q_TERNARY_LOGICAL_NUMEL_KEY: int(packed_q.logical_numel),
        Q_TERNARY_PADDING_VALUES_KEY: int(packed_q.padding_values),
    }


def _unpack_q_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    q_packed_enabled: bool | None,
) -> torch.Tensor:
    q_flag_enabled = persistent_q_ternary_byte_packed_enabled(enabled=q_packed_enabled)
    q_packed_saved = bool(payload.get(Q_TERNARY_BYTE_PACKED_PERSISTED_KEY))
    if q_packed_saved and not q_flag_enabled:
        raise ValueError(
            "2C4a sidecar contains byte-packed q payload but "
            f"{PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if q_flag_enabled and q_packed_saved:
        if RAW_Q_LEVELS_FIELD in payload:
            raise ValueError(
                "2C4a byte-packed q sidecar must not retain raw int8 q_levels "
                "alongside packed payload"
            )
        packed_tensor = payload.get(Q_TERNARY_PACKED_PAYLOAD_KEY)
        if not isinstance(packed_tensor, torch.Tensor):
            raise ValueError("2C4a byte-packed q sidecar missing q_ternary_packed_payload tensor")
        packed_cpu = packed_tensor.detach().cpu().contiguous()
        _reject_int8_tensor_as_packed_q(
            packed_cpu,
            context="2C4a byte-packed q sidecar payload",
        )
        format_tag = payload.get(Q_TERNARY_PACKED_SCHEMA_KEY)
        if not isinstance(format_tag, str) or not format_tag:
            raise ValueError("2C4a byte-packed q sidecar missing q packed format tag")
        logical_shape = tuple(int(dim) for dim in payload[Q_TERNARY_LOGICAL_SHAPE_KEY])
        logical_numel = int(payload[Q_TERNARY_LOGICAL_NUMEL_KEY])
        padding_values = int(payload[Q_TERNARY_PADDING_VALUES_KEY])
        if format_tag == PACKED_TERNARY_Q_FORMAT:
            packed_q = PackedTernaryQState(
                packed=packed_cpu,
                logical_shape=logical_shape,
                logical_numel=logical_numel,
                padding_values=padding_values,
                format=format_tag,
            )
            return unpack_ternary_q_2bit_reference(packed_q)
        if format_tag == BASE3_Q_FORMAT:
            packed_q = PackedBase3TernaryQState(
                packed=packed_cpu,
                logical_shape=logical_shape,
                logical_numel=logical_numel,
                padding_values=padding_values,
                format=format_tag,
            )
            return unpack_ternary_q_base3_5perbyte_reference(packed_q)
        raise ValueError(f"2C4a byte-packed q sidecar has unknown format tag {format_tag!r}")
    if any(
        key in payload
        for key in (
            Q_TERNARY_PACKED_PAYLOAD_KEY,
            Q_TERNARY_PACKED_SCHEMA_KEY,
            Q_TERNARY_LOGICAL_SHAPE_KEY,
            Q_TERNARY_LOGICAL_NUMEL_KEY,
            Q_TERNARY_PADDING_VALUES_KEY,
        )
    ):
        raise ValueError("2C4a byte-packed q metadata present without persisted flag")
    if RAW_Q_LEVELS_FIELD not in payload:
        raise ValueError("2C4a sidecar missing authoritative int8 q_levels payload")
    return payload[RAW_Q_LEVELS_FIELD].detach().cpu().to(torch.int8).contiguous()


def _distinct_q_format_tags_from_tensor_payloads(
    tensor_payloads: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    tags: set[str] = set()
    for payload in tensor_payloads.values():
        if not bool(payload.get(Q_TERNARY_BYTE_PACKED_PERSISTED_KEY)):
            continue
        format_tag = payload.get(Q_TERNARY_PACKED_SCHEMA_KEY)
        if not isinstance(format_tag, str) or not format_tag:
            raise ValueError("2C4a byte-packed q sidecar missing q packed format tag")
        tags.add(format_tag)
    return tags


def _assert_uniform_checkpoint_q_formats(
    tensor_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    tags = _distinct_q_format_tags_from_tensor_payloads(tensor_payloads)
    if len(tags) > 1:
        raise ValueError(
            "2C4a checkpoint contains mixed q packed formats across modules: "
            f"{sorted(tags)}"
        )


def packed_q_state_from_roundtrip_q_payload(
    payload: Mapping[str, Any],
) -> PackedTernaryQState | PackedBase3TernaryQState:
    if not bool(payload.get(Q_TERNARY_BYTE_PACKED_PERSISTED_KEY)):
        raise ValueError("roundtrip payload is not byte-packed q")
    packed_tensor = payload.get(Q_TERNARY_PACKED_PAYLOAD_KEY)
    if not isinstance(packed_tensor, torch.Tensor):
        raise ValueError("roundtrip payload missing q_ternary_packed_payload tensor")
    packed_cpu = packed_tensor.detach().cpu().contiguous()
    _reject_int8_tensor_as_packed_q(
        packed_cpu,
        context="roundtrip byte-packed q payload",
    )
    format_tag = payload.get(Q_TERNARY_PACKED_SCHEMA_KEY)
    if not isinstance(format_tag, str) or not format_tag:
        raise ValueError("roundtrip payload missing q packed format tag")
    logical_shape = tuple(int(dim) for dim in payload[Q_TERNARY_LOGICAL_SHAPE_KEY])
    logical_numel = int(payload[Q_TERNARY_LOGICAL_NUMEL_KEY])
    padding_values = int(payload[Q_TERNARY_PADDING_VALUES_KEY])
    if format_tag == PACKED_TERNARY_Q_FORMAT:
        return PackedTernaryQState(
            packed=packed_cpu,
            logical_shape=logical_shape,
            logical_numel=logical_numel,
            padding_values=padding_values,
            format=format_tag,
        )
    if format_tag == BASE3_Q_FORMAT:
        return PackedBase3TernaryQState(
            packed=packed_cpu,
            logical_shape=logical_shape,
            logical_numel=logical_numel,
            padding_values=padding_values,
            format=format_tag,
        )
    raise ValueError(f"roundtrip payload has unknown q packed format tag {format_tag!r}")


def packed_w6_acc_payload_from_roundtrip_bounded_payload(
    bounded_payload: Mapping[str, Any],
) -> PackedW6AccumulatorPayload:
    if not bool(bounded_payload.get(W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY)):
        raise ValueError("roundtrip bounded payload is not W6 byte-packed")
    packed_tensor = bounded_payload.get(W6_BYTE_PACKED_PAYLOAD_KEY)
    if not isinstance(packed_tensor, torch.Tensor):
        raise ValueError("roundtrip bounded payload missing w6_byte_packed_payload tensor")
    packed_cpu = packed_tensor.detach().cpu().contiguous()
    reject_int16_tensor_as_packed_acc(
        packed_cpu,
        context="roundtrip W6 byte-packed accumulator payload",
    )
    return PackedW6AccumulatorPayload(
        packed=packed_cpu,
        logical_shape=tuple(
            int(dim) for dim in bounded_payload[W6_BYTE_PACKED_LOGICAL_SHAPE_KEY]
        ),
        logical_numel=int(bounded_payload[W6_BYTE_PACKED_LOGICAL_NUMEL_KEY]),
        schema=str(bounded_payload.get(W6_BYTE_PACKED_SCHEMA_KEY, W6_BYTE_PACKED_SCHEMA)),
    )


def _bounded_accumulator_from_decoded_i16(
    acc: torch.Tensor,
    *,
    cold_default_value: int,
    candidate_name: str,
) -> BoundedDeltaAccumulatorState:
    flat = acc.detach().cpu().flatten().to(torch.int16)
    numel = int(flat.numel())
    default = int(cold_default_value)
    hot_indices: list[int] = []
    hot_values: list[int] = []
    for index in range(numel):
        value = int(flat[index].item())
        if value != default:
            hot_indices.append(index)
            hot_values.append(value)
    return BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(dim) for dim in acc.shape),
        cold_default_value=default,
        hot_exact_indices=tuple(hot_indices),
        hot_exact_values=tuple(hot_values),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name=str(candidate_name),
        raw_arrays_included=False,
    )


def _tensor_state_roundtrip_payload(
    state: BoundedDeltaTensorState,
    *,
    byte_packed_enabled: bool | None = None,
    w5_byte_packed_enabled: bool | None = None,
    q_packed_enabled: bool | None = None,
    q_codec_selector: str | None = None,
) -> dict[str, Any]:
    bounded = state.bounded_accumulator
    bounded_payload: dict[str, Any] = {
        "logical_shape": tuple(int(dim) for dim in bounded.logical_shape),
        "cold_default_value": int(bounded.cold_default_value),
        "hot_exact_indices": tuple(int(item) for item in bounded.hot_exact_indices),
        "hot_exact_values": tuple(int(item) for item in bounded.hot_exact_values),
        "cold_exception_indices": tuple(int(item) for item in bounded.cold_exception_indices),
        "cold_exception_values": tuple(int(item) for item in bounded.cold_exception_values),
        "candidate_name": str(bounded.candidate_name),
        "raw_arrays_serialized_for_resume_only": True,
        "dense_int16_accumulator_persisted": False,
        W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY: False,
        W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY: False,
        "hot_exact_indices_sha256": _identity_sha256(
            state.state_key,
            tuple(int(item) for item in bounded.hot_exact_indices),
        ),
        "hot_exact_values_sha256": _ordered_value_sha256(
            state.state_key,
            "hot_exact_value",
            {
                int(index): int(value)
                for index, value in zip(bounded.hot_exact_indices, bounded.hot_exact_values)
            },
        ),
        "cold_exception_indices_sha256": _identity_sha256(
            state.state_key,
            tuple(int(item) for item in bounded.cold_exception_indices),
        ),
        "cold_exception_values_sha256": _ordered_value_sha256(
            state.state_key,
            "cold_exception_value",
            {
                int(index): int(value)
                for index, value in zip(
                    bounded.cold_exception_indices,
                    bounded.cold_exception_values,
                )
            },
        ),
    }
    use_w5 = persistent_w5_byte_packed_enabled(enabled=w5_byte_packed_enabled)
    use_w6 = persistent_w6_byte_packed_enabled(enabled=byte_packed_enabled)
    if use_w5 and use_w6:
        raise ValueError("W5 and W6 persistent accumulator byte-packing are mutually exclusive")
    if state.event_coded_live_carrier is not None and (use_w5 or use_w6):
        raise ValueError(
            "event-coded live carrier is mutually exclusive with W5/W6 byte-packed accumulators"
        )
    decoded_i16 = decode_bounded_accumulator_to_i16(bounded)
    event_coded_payload: dict[str, Any] = {}
    if state.event_coded_live_carrier is not None:
        event_coded_payload = _event_coded_live_carrier_payload_from_state(
            state.event_coded_live_carrier
        )
    elif use_w5:
        packed_payload = pack_w5_lanes_to_bytes(decoded_i16)
        bounded_payload[W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY] = True
        bounded_payload[W5_BYTE_PACKED_SCHEMA_KEY] = str(packed_payload.schema)
        bounded_payload[W5_BYTE_PACKED_PAYLOAD_KEY] = packed_payload.packed.detach().cpu().contiguous()
        bounded_payload[W5_BYTE_PACKED_LOGICAL_SHAPE_KEY] = list(packed_payload.logical_shape)
        bounded_payload[W5_BYTE_PACKED_LOGICAL_NUMEL_KEY] = int(packed_payload.logical_numel)
    elif use_w6:
        packed_payload = pack_w6_lanes_to_bytes(decoded_i16)
        bounded_payload[W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY] = True
        bounded_payload[W6_BYTE_PACKED_SCHEMA_KEY] = str(packed_payload.schema)
        bounded_payload[W6_BYTE_PACKED_PAYLOAD_KEY] = packed_payload.packed.detach().cpu().contiguous()
        bounded_payload[W6_BYTE_PACKED_LOGICAL_SHAPE_KEY] = list(packed_payload.logical_shape)
        bounded_payload[W6_BYTE_PACKED_LOGICAL_NUMEL_KEY] = int(packed_payload.logical_numel)
    q_levels_logical = state.q_levels.detach().cpu().to(torch.int8).contiguous()
    payload: dict[str, Any] = {
        "state_key": str(state.state_key),
        "q_sha256": tensor_sha256(q_levels_logical),
        "frozen_scale": state.frozen_scale.detach().cpu().to(torch.float32).contiguous(),
        "frozen_scale_sha256": tensor_sha256(
            state.frozen_scale.detach().cpu().to(torch.float32).contiguous()
        ),
        "bounded_accumulator": bounded_payload,
        "exact_accumulator_shadow_saved": False,
        "exact_accumulator_shadow_sha256": _tensor_sha_or_none(state.exact_accumulator_shadow),
        Q_TERNARY_BYTE_PACKED_PERSISTED_KEY: False,
    }
    payload.update(
        _pack_q_for_checkpoint(
            q_levels_logical,
            q_packed_enabled=q_packed_enabled,
            q_codec_selector=q_codec_selector,
        )
    )
    payload.update(event_coded_payload)
    return payload


def _state_from_roundtrip_payload(
    payload: Mapping[str, Any],
    *,
    byte_packed_enabled: bool | None = None,
    w5_byte_packed_enabled: bool | None = None,
    q_packed_enabled: bool | None = None,
    event_coded_enabled: bool | None = None,
) -> BoundedDeltaTensorState:
    bounded_payload = dict(payload.get("bounded_accumulator") or {})
    w5_saved = bool(bounded_payload.get(W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY))
    w6_saved = bool(bounded_payload.get(W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY))
    w5_flag = persistent_w5_byte_packed_enabled(enabled=w5_byte_packed_enabled)
    w6_flag = persistent_w6_byte_packed_enabled(enabled=byte_packed_enabled)
    if w5_saved and w6_saved:
        raise ValueError("2C4a sidecar contains both W5 and W6 byte-packed accumulator payloads")
    if w5_saved and not w5_flag:
        raise ValueError(
            "2C4a sidecar contains byte-packed W5 accumulator payload but "
            f"{PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if w6_saved and not w6_flag:
        raise ValueError(
            "2C4a sidecar contains byte-packed W6 accumulator payload but "
            f"{PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if w5_flag and w5_saved:
        packed_tensor = bounded_payload.get(W5_BYTE_PACKED_PAYLOAD_KEY)
        if not isinstance(packed_tensor, torch.Tensor):
            raise ValueError("2C4a byte-packed sidecar missing w5_byte_packed_payload tensor")
        packed_cpu = packed_tensor.detach().cpu().contiguous()
        reject_int16_tensor_as_packed_acc(
            packed_cpu,
            context="2C4a byte-packed W5 accumulator sidecar payload",
        )
        packed_payload = PackedW5AccumulatorPayload(
            packed=packed_cpu,
            logical_shape=tuple(
                int(dim) for dim in bounded_payload[W5_BYTE_PACKED_LOGICAL_SHAPE_KEY]
            ),
            logical_numel=int(bounded_payload[W5_BYTE_PACKED_LOGICAL_NUMEL_KEY]),
            schema=str(bounded_payload.get(W5_BYTE_PACKED_SCHEMA_KEY, W5_BYTE_PACKED_SCHEMA)),
        )
        decoded_i16 = unpack_w5_lanes_from_bytes(packed_payload)
        bounded = _bounded_accumulator_from_decoded_i16(
            decoded_i16,
            cold_default_value=int(bounded_payload["cold_default_value"]),
            candidate_name=str(bounded_payload["candidate_name"]),
        )
    elif w6_flag and w6_saved:
        packed_tensor = bounded_payload.get(W6_BYTE_PACKED_PAYLOAD_KEY)
        if not isinstance(packed_tensor, torch.Tensor):
            raise ValueError("2C4a byte-packed sidecar missing w6_byte_packed_payload tensor")
        packed_cpu = packed_tensor.detach().cpu().contiguous()
        reject_int16_tensor_as_packed_acc(
            packed_cpu,
            context="2C4a byte-packed accumulator sidecar payload",
        )
        packed_payload = PackedW6AccumulatorPayload(
            packed=packed_cpu,
            logical_shape=tuple(
                int(dim) for dim in bounded_payload[W6_BYTE_PACKED_LOGICAL_SHAPE_KEY]
            ),
            logical_numel=int(bounded_payload[W6_BYTE_PACKED_LOGICAL_NUMEL_KEY]),
            schema=str(bounded_payload.get(W6_BYTE_PACKED_SCHEMA_KEY, W6_BYTE_PACKED_SCHEMA)),
        )
        decoded_i16 = unpack_w6_lanes_from_bytes(packed_payload)
        bounded = _bounded_accumulator_from_decoded_i16(
            decoded_i16,
            cold_default_value=int(bounded_payload["cold_default_value"]),
            candidate_name=str(bounded_payload["candidate_name"]),
        )
    else:
        bounded = BoundedDeltaAccumulatorState(
            logical_shape=tuple(int(dim) for dim in bounded_payload["logical_shape"]),
            cold_default_value=int(bounded_payload["cold_default_value"]),
            hot_exact_indices=tuple(int(item) for item in bounded_payload["hot_exact_indices"]),
            hot_exact_values=tuple(int(item) for item in bounded_payload["hot_exact_values"]),
            cold_exception_indices=tuple(
                int(item) for item in bounded_payload["cold_exception_indices"]
            ),
            cold_exception_values=tuple(
                int(item) for item in bounded_payload["cold_exception_values"]
            ),
            candidate_name=str(bounded_payload["candidate_name"]),
            raw_arrays_included=False,
        )
    q_levels = _unpack_q_from_checkpoint(payload, q_packed_enabled=q_packed_enabled)
    frozen_scale = payload["frozen_scale"].detach().cpu().to(torch.float32).contiguous()
    if tensor_sha256(q_levels) != str(payload.get("q_sha256")):
        raise ValueError("2C4a q sidecar hash mismatch on load")
    if tensor_sha256(frozen_scale) != str(payload.get("frozen_scale_sha256")):
        raise ValueError("2C4a frozen-scale sidecar hash mismatch on load")
    if bool(payload.get("exact_accumulator_shadow_saved")):
        raise ValueError("2C4a sidecar must not save dense exact accumulator shadows")
    if bool(bounded_payload.get("dense_int16_accumulator_persisted")):
        raise ValueError("2C4a sidecar must not persist dense int16 accumulators")
    event_saved = bool(payload.get(EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY))
    event_flag = event_coded_live_carrier_enabled(enabled=event_coded_enabled)
    if event_saved:
        if w5_saved or w6_saved:
            raise ValueError(
                "2C4a sidecar cannot combine event-coded live carrier with W5/W6 acc payloads"
            )
        if not event_flag:
            from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
                RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV,
            )

            raise ValueError(
                "2C4a sidecar contains event-coded live carrier payload but "
                f"{RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV}=1 is not enabled"
            )
        packed = _packed_event_coded_from_roundtrip_payload(payload)
        carrier = hydrate_event_coded_live_carrier_from_packed(
            packed,
            demotion_band=int(payload.get(EVENT_CODED_DEMOTION_BAND_KEY, 1)),
        )
        return BoundedDeltaTensorState(
            state_key=str(payload["state_key"]),
            q_levels=q_levels,
            frozen_scale=frozen_scale,
            bounded_accumulator=bounded,
            exact_accumulator_shadow=None,
            bounded_accumulator_fresh_for_exact_shadow=False,
            event_coded_live_carrier=carrier,
        )
    if (
        w6_flag
        and not w6_saved
        and any(
            key in bounded_payload
            for key in (
                W6_BYTE_PACKED_PAYLOAD_KEY,
                W6_BYTE_PACKED_SCHEMA_KEY,
                W6_BYTE_PACKED_LOGICAL_SHAPE_KEY,
                W6_BYTE_PACKED_LOGICAL_NUMEL_KEY,
            )
        )
    ):
        raise ValueError("2C4a byte-packed metadata present without persisted flag")
    if (
        w5_flag
        and not w5_saved
        and any(
            key in bounded_payload
            for key in (
                W5_BYTE_PACKED_PAYLOAD_KEY,
                W5_BYTE_PACKED_SCHEMA_KEY,
                W5_BYTE_PACKED_LOGICAL_SHAPE_KEY,
                W5_BYTE_PACKED_LOGICAL_NUMEL_KEY,
            )
        )
    ):
        raise ValueError("2C4a W5 byte-packed metadata present without persisted flag")
    return BoundedDeltaTensorState(
        state_key=str(payload["state_key"]),
        q_levels=q_levels,
        frozen_scale=frozen_scale,
        bounded_accumulator=bounded,
        exact_accumulator_shadow=None,
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def build_trainer_sub2_authority_checkpoint_blob(
    model: torch.nn.Module,
    *,
    eligible_modules: Mapping[str, BitLinear],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    step: int = 0,
    byte_packed_enabled: bool | None = None,
    w5_byte_packed_enabled: bool | None = None,
    q_packed_enabled: bool | None = None,
    q_codec_selector: str | None = None,
) -> dict[str, Any]:
    """Build a 2C4a reconstructable sidecar checkpoint blob.

    Eligible BitLinear FP masters are deliberately excluded from model_state.
    The q+scale+bounded sidecar is the only authoritative eligible-weight state.
    """

    eligible_weight_keys = _eligible_weight_state_keys(eligible_modules)
    missing = set(eligible_modules) - set(tensor_states)
    if missing:
        raise ValueError(f"2C4a missing tensor states for eligible modules: {sorted(missing)}")
    raw_model_state = model.state_dict()
    model_state = {
        str(key): value.detach().cpu().clone()
        for key, value in raw_model_state.items()
        if str(key) not in set(eligible_weight_keys)
    }
    tensor_payloads = {
        str(key): _tensor_state_roundtrip_payload(
            tensor_states[str(key)],
            byte_packed_enabled=byte_packed_enabled,
            w5_byte_packed_enabled=w5_byte_packed_enabled,
            q_packed_enabled=q_packed_enabled,
            q_codec_selector=q_codec_selector,
        )
        for key in sorted(eligible_modules)
    }
    w6_byte_packed_saved = any(
        bool((payload.get("bounded_accumulator") or {}).get(W6_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY))
        for payload in tensor_payloads.values()
    )
    w5_byte_packed_saved = any(
        bool((payload.get("bounded_accumulator") or {}).get(W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY))
        for payload in tensor_payloads.values()
    )
    q_packed_saved = any(
        bool(payload.get(Q_TERNARY_BYTE_PACKED_PERSISTED_KEY)) for payload in tensor_payloads.values()
    )
    event_coded_saved = any(
        bool(payload.get(EVENT_CODED_LIVE_CARRIER_PERSISTED_KEY))
        for payload in tensor_payloads.values()
    )
    sidecar = {
        "schema_version": TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        "artifact_role": "trainer_sub2_authoritative_sidecar",
        "step": int(step),
        "eligible_state_keys": tuple(sorted(tensor_payloads)),
        "eligible_weight_state_keys": eligible_weight_keys,
        "tensor_payloads": tensor_payloads,
        "eligible_fp_masters_authoritative": False,
        "dense_int16_persistent_accumulator_saved": False,
        "w6_byte_packed_persistent_accumulator_saved": bool(w6_byte_packed_saved),
        "w5_byte_packed_persistent_accumulator_saved": bool(w5_byte_packed_saved),
        EVENT_CODED_LIVE_CARRIER_SAVED_KEY: bool(event_coded_saved),
        Q_TERNARY_BYTE_PACKED_PERSISTED_SAVED_KEY: bool(q_packed_saved),
        "normal_bitlinear_weight_forward_not_claimed": True,
    }
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(sidecar)
    blob = {
        "schema_version": TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        "artifact_role": "trainer_sub2_checkpoint_blob",
        "model_state": model_state,
        "trainer_sub2_authority": sidecar,
        "checkpoint_written": False,
        "gpu_launched": False,
        "optimizer_state_included": False,
    }
    blob["checkpoint_blob_sha256"] = _roundtrip_payload_sha256(blob)
    return blob


def load_trainer_sub2_authority_checkpoint_blob(
    model: torch.nn.Module,
    blob: Mapping[str, Any],
    *,
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device | str = "cpu",
    byte_packed_enabled: bool | None = None,
    w5_byte_packed_enabled: bool | None = None,
    q_packed_enabled: bool | None = None,
    event_coded_enabled: bool | None = None,
) -> dict[str, BoundedDeltaTensorState]:
    if blob.get("schema_version") != TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION:
        raise ValueError("2C4a checkpoint blob schema mismatch")
    model_state = dict(blob.get("model_state") or {})
    eligible_weight_keys = set(_eligible_weight_state_keys(eligible_modules))
    present_eligible = sorted(eligible_weight_keys.intersection(model_state))
    if present_eligible:
        raise ValueError(
            "2C4a raw state_dict eligible-weight fallback rejected: "
            f"{present_eligible}"
        )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing_set = set(str(key) for key in missing)
    if missing_set != eligible_weight_keys or unexpected:
        raise ValueError(
            "2C4a checkpoint load requires strict non-eligible model_state and "
            f"exact eligible-weight omissions; missing={sorted(missing_set)} "
            f"unexpected={list(unexpected)}"
        )
    sidecar = dict(blob.get("trainer_sub2_authority") or {})
    if sidecar.get("schema_version") != TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION:
        raise ValueError("2C4a sidecar schema mismatch")
    if bool(sidecar.get("dense_int16_persistent_accumulator_saved")):
        raise ValueError("2C4a sidecar must not save dense int16 persistent accumulators")
    if bool(sidecar.get("w6_byte_packed_persistent_accumulator_saved")) and not persistent_w6_byte_packed_enabled(
        enabled=byte_packed_enabled
    ):
        raise ValueError(
            "2C4a sidecar contains byte-packed W6 persistent accumulators but "
            f"{PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if bool(sidecar.get("w5_byte_packed_persistent_accumulator_saved")) and not persistent_w5_byte_packed_enabled(
        enabled=w5_byte_packed_enabled
    ):
        raise ValueError(
            "2C4a sidecar contains byte-packed W5 persistent accumulators but "
            f"{PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if bool(sidecar.get(Q_TERNARY_BYTE_PACKED_PERSISTED_SAVED_KEY)) and not persistent_q_ternary_byte_packed_enabled(
        enabled=q_packed_enabled
    ):
        raise ValueError(
            "2C4a sidecar contains byte-packed q persistent payloads but "
            f"{PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV}=1 is not enabled"
        )
    if bool(sidecar.get(EVENT_CODED_LIVE_CARRIER_SAVED_KEY)) and not event_coded_live_carrier_enabled(
        enabled=event_coded_enabled
    ):
        from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
            RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV,
        )

        raise ValueError(
            "2C4a sidecar contains event-coded live carrier payloads but "
            f"{RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV}=1 is not enabled"
        )
    declared_hash = str(sidecar.get("authoritative_state_payload_sha256"))
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    if declared_hash != _roundtrip_payload_sha256(sidecar_without_hash):
        raise ValueError("2C4a sidecar authoritative payload hash mismatch")
    tensor_payloads = dict(sidecar.get("tensor_payloads") or {})
    _assert_uniform_checkpoint_q_formats(tensor_payloads)
    states = {
        str(key): _state_from_roundtrip_payload(
            payload,
            byte_packed_enabled=byte_packed_enabled,
            w5_byte_packed_enabled=w5_byte_packed_enabled,
            q_packed_enabled=q_packed_enabled,
            event_coded_enabled=event_coded_enabled,
        )
        for key, payload in sorted(tensor_payloads.items())
    }
    if set(states) != set(eligible_modules):
        raise ValueError("2C4a sidecar state keys do not match eligible modules")
    if any(state.exact_accumulator_shadow is not None for state in states.values()):
        raise ValueError("2C4a loaded sidecar must not contain dense exact accumulator shadows")
    return {
        key: BoundedDeltaTensorState(
            state_key=state.state_key,
            q_levels=state.q_levels.to(device="cpu").contiguous(),
            frozen_scale=state.frozen_scale.to(device="cpu").contiguous(),
            bounded_accumulator=state.bounded_accumulator,
            exact_accumulator_shadow=None,
            bounded_accumulator_fresh_for_exact_shadow=False,
            event_coded_live_carrier=state.event_coded_live_carrier,
        )
        for key, state in states.items()
    }


P1_LIVE_CHECKPOINT_FORMAT = "p1_trainer_sub2_authority_live/v0"


@dataclass(frozen=True)
class TrainerSub2AuthorityLoadResult:
    authority_states: dict[str, BoundedDeltaTensorState] | None
    routing: str


def is_p1_live_sub2_checkpoint(ckpt: Mapping[str, Any]) -> bool:
    return (
        ckpt.get("checkpoint_format") == P1_LIVE_CHECKPOINT_FORMAT
        and isinstance(ckpt.get("trainer_sub2_authority"), dict)
    )


def reject_p1_live_checkpoint_format_mismatch(ckpt: Mapping[str, Any]) -> None:
    sidecar = ckpt.get("trainer_sub2_authority")
    if isinstance(sidecar, dict) and ckpt.get("checkpoint_format") != P1_LIVE_CHECKPOINT_FORMAT:
        raise ValueError(
            "P1 live checkpoint format mismatch: trainer_sub2_authority present "
            f"but checkpoint_format={ckpt.get('checkpoint_format')!r} "
            f"(expected {P1_LIVE_CHECKPOINT_FORMAT!r})"
        )


def install_persistent_sub2_eval_authority_on_parent(
    model: torch.nn.Module,
    states: Mapping[str, BoundedDeltaTensorState],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Install sidecar authority into BitLinear cached-inference path on parent_m only."""

    if set(states) != set(eligible_modules):
        raise ValueError("P1 install: state keys must match eligible modules")
    for key, module in sorted(eligible_modules.items()):
        if getattr(module, "_p1_persistent_eval_authority_installed", False):
            raise ValueError(f"P1 install: double-install on {key!r}")
        state = states[str(key)]
        if state.exact_accumulator_shadow is not None:
            state.bounded_decode_parity_report(fail_on_mismatch=True)
        w = state.materialized_weight(device=device, requires_grad=False)
        module._cached_weight = w.detach().contiguous()
        module._cached_active = True
        module._p1_persistent_eval_authority_installed = True
    model.eval()


def detach_persistent_sub2_eval_authority(
    eligible_modules: Mapping[str, BitLinear],
) -> None:
    for module in eligible_modules.values():
        if getattr(module, "_p1_persistent_eval_authority_installed", False):
            module.unfreeze()
            module._p1_persistent_eval_authority_installed = False


def load_train_checkpoint_into_model(
    model: torch.nn.Module,
    ckpt: Mapping[str, Any],
    *,
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    inference_only: bool,
    sub2_live_enabled: bool,
) -> TrainerSub2AuthorityLoadResult:
    reject_p1_live_checkpoint_format_mismatch(ckpt)
    if is_p1_live_sub2_checkpoint(ckpt):
        if not sub2_live_enabled:
            raise ValueError(
                "P1-format checkpoint requires --sub2-authority-live-checkpoint"
            )
        eligible = select_trainer_eligible_bitlinears(
            model,
            use_ternary_bulk=use_ternary_bulk,
            eligible_scope=eligible_scope,
        )
        states = load_trainer_sub2_authority_checkpoint_blob(
            model,
            ckpt,
            eligible_modules=eligible,
            device=device,
        )
        if inference_only:
            install_persistent_sub2_eval_authority_on_parent(
                model,
                states,
                eligible,
                device=device,
            )
        return TrainerSub2AuthorityLoadResult(authority_states=states, routing="p1_live")
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ValueError("checkpoint missing model_state")
    model.load_state_dict(model_state, strict=True)
    return TrainerSub2AuthorityLoadResult(authority_states=None, routing="legacy")


def save_trainer_sub2_live_checkpoint_envelope(
    model: torch.nn.Module,
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState] | None = None,
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    step: int = 0,
    config: Mapping[str, Any],
    source_pin: str,
    epoch: int = 0,
) -> dict[str, Any]:
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    states = (
        dict(tensor_states)
        if tensor_states is not None
        else derive_trainer_sub2_authority_states(eligible)
    )
    inner = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        step=int(step),
    )
    return {
        **inner,
        "checkpoint_format": P1_LIVE_CHECKPOINT_FORMAT,
        "config": dict(config),
        "step": int(step),
        "epoch": int(epoch),
        "source_pin": str(source_pin),
        "checkpoint_written": True,
    }


def _default_forward_output(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    if "x" not in batch:
        raise ValueError("2C4a default forward output requires batch['x']; pass forward_output_fn")
    return model(batch["x"])


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
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
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
    weighted_grad_by_key: dict[str, torch.Tensor] = {}
    weighted_grad_stats: dict[str, dict[str, Any]] = {}
    prior_training = bool(model.training)
    loss_finite = False
    try:
        _emit_phase("PHASE_START", "forward_backward")
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
                weighted_grad_by_key[key] = weighted_grad
                weighted_grad_stats[key] = {
                    "weighted_grad_shape": list(weighted_grad.shape),
                    "weighted_grad_nonzero_count": int((weighted_grad != 0).sum().item()),
                    "weighted_grad_finite": bool(torch.isfinite(weighted_grad).all().item()),
                }
    finally:
        model.train(prior_training)

    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_forward_backward(weighted_grad_by_key)
    _emit_phase("PHASE_END", "forward_backward")

    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=weighted_grad_by_key,
        q_levels_by_key={key: state.q_levels for key, state in states.items()},
        rank_spec=rank_spec,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )
    resolved_mode = str(path["resolved_mode"])
    sparse_events_by_key = dict(path["sparse_events_by_key"])
    for key, events in sparse_events_by_key.items():
        weighted_grad_stats[key]["sparse_vote_event_count"] = int(events.event_count())
        weighted_grad_stats[key]["sparse_vote_authority_mode"] = resolved_mode

    # fused_only: NO densify, NO dense oracle parity (oracle_on compartment only)
    _emit_phase("PHASE_START", "update")
    step_result = apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_update()
    _emit_phase("PHASE_END", "update")
    _emit_phase("PHASE_START", "emission")
    deferred_backlog_output_entry_count = sum(
        len(entries)
        for entries in step_result.deferred_backlog.values()
    )
    global_rate_cap_enabled = bool(step_result.global_summary.get("global_rate_cap_enabled"))
    trainer_builder_has_no_active_control_parameters = (
        trainer_local_update_builder_active_control_parameters() == ()
    )
    active_controls_inactive_proven = (
        not global_rate_cap_enabled
        and deferred_backlog_output_entry_count == 0
        and trainer_builder_has_no_active_control_parameters
    )
    proof_by_key = step_result.global_summary["candidate_local_update_proof_by_key"]
    parity_by_key: dict[str, Any] = {}
    if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON:
        dense_votes_for_oracle_proof = {
            key: _dense_votes_from_sparse_events(
                sparse_events_by_key[key],
                shape=states[key].q_levels.shape,
            )
            for key in sparse_events_by_key
        }
        parity_by_key = {
            key: _oracle_parity_proof(
                state_key=key,
                prior_state=states[key],
                next_state=step_result.tensor_states[key],
                votes=dense_votes_for_oracle_proof[key],
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
    total_sparse_events = sum(events.event_count() for events in sparse_events_by_key.values())
    q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
    parity_ok = (
        True
        if resolved_mode == SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY
        else all(bool(proof.get("parity_pass")) for proof in parity_by_key.values())
    )
    pass_receipt = bool(
        loss_finite
        and optimizer_checks.get("pass")
        and int(optimizer_checks.get("eligible_params_in_optimizer", -1)) == 0
        and int(optimizer_checks.get("eligible_optimizer_state_entries", -1)) == 0
        and total_sparse_events > 0
        and q_changed_count > 0
        and all(bool(proof.get("pass")) for proof in proof_by_key.values())
        and parity_ok
        and shadow_free_after
        and fp_masters_byte_identical
        and active_controls_inactive_proven
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
        default_off_trainer_active_controls_inactive_proven=active_controls_inactive_proven,
        global_cap_spec_passed=False,
        global_rate_cap_enabled=global_rate_cap_enabled,
        deferred_backlog_input_present=False,
        deferred_backlog_output_entry_count=deferred_backlog_output_entry_count,
        replay_ce_veto_maps_present=False,
        pc_aux_maps_present=False,
        pc_aux_mode_effective="not_enabled",
        front_c_identity_observer_present=False,
        candidate_mode_rejects_active_controls=True,
        trainer_builder_has_no_active_control_parameters=trainer_builder_has_no_active_control_parameters,
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
        transient_over2_tensors=_fused_or_oracle_transient_over2(resolved_mode),
        vote_projection_proof=_build_vote_projection_proof(
            rank_spec=rank_spec,
            update_spec=update_spec,
            resolved_mode=resolved_mode,
            total_sparse_events=int(total_sparse_events),
            # JSON-safe projection only (never raw oracle_only with SparseVoteEvents)
            oracle_only=path.get("oracle_only_serializable"),
            sparse_event_map_binding_sha256_by_key=path.get(
                "sparse_event_map_binding_sha256_by_key"
            ),
            sparse_event_count_by_key=path.get("sparse_event_count_by_key"),
            sparse_event_logical_shape_by_key=path.get(
                "sparse_event_logical_shape_by_key"
            ),
            s1_binding_interval_seconds=path.get("s1_binding_interval_seconds"),
        ),
        candidate_step_summary={
            "default_off_trainer_active_controls_inactive_proven": active_controls_inactive_proven,
            "global_cap_spec_passed": False,
            "global_rate_cap_enabled": global_rate_cap_enabled,
            "deferred_backlog_input_present": False,
            "deferred_backlog_output_entry_count": deferred_backlog_output_entry_count,
            "replay_ce_veto_maps_present": False,
            "pc_aux_maps_present": False,
            "pc_aux_mode_effective": "not_enabled",
            "front_c_identity_observer_present": False,
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
    _emit_phase("PHASE_END", "emission")
    return receipt


def build_trainer_sub2_authority_roundtrip_receipt(
    model: torch.nn.Module,
    *,
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    use_ternary_bulk: bool,
    forward_output_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor] | None = None,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    lr: float = 0.0,
    weight_decay: float = 0.0,
    step: int = 0,
    vote_update_spec: VoteUpdateSpec | None = None,
    poison_value: float = 17.0,
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> TrainerSub2AuthorityRoundtripReceipt:
    """Run the gated 2C4a checkpoint/resume/update poison-falsification proof."""

    output_fn = forward_output_fn or _default_forward_output
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        step=int(step),
    )
    eligible_weight_keys = _eligible_weight_state_keys(eligible)
    excluded = all(key not in blob["model_state"] for key in eligible_weight_keys)

    raw_fallback_rejected = False
    poisoned_blob = dict(blob)
    poisoned_model_state = dict(blob["model_state"])
    first_key = eligible_weight_keys[0]
    poisoned_model_state[first_key] = eligible[sorted(eligible)[0]].weight.detach().cpu().clone()
    poisoned_blob["model_state"] = poisoned_model_state
    try:
        raw_fresh = fresh_model_fn().to(device=device)
        raw_eligible = select_trainer_eligible_bitlinears(
            raw_fresh,
            use_ternary_bulk=use_ternary_bulk,
            eligible_scope=eligible_scope,
        )
        load_trainer_sub2_authority_checkpoint_blob(
            raw_fresh,
            poisoned_blob,
            eligible_modules=raw_eligible,
            device=device,
        )
    except ValueError as exc:
        raw_fallback_rejected = "raw state_dict eligible-weight fallback rejected" in str(exc)

    resumed_model = fresh_model_fn().to(device=device)
    resumed_eligible = select_trainer_eligible_bitlinears(
        resumed_model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    loaded_states = load_trainer_sub2_authority_checkpoint_blob(
        resumed_model,
        blob,
        eligible_modules=resumed_eligible,
        device=device,
    )
    reblob = build_trainer_sub2_authority_checkpoint_blob(
        resumed_model,
        eligible_modules=resumed_eligible,
        tensor_states=loaded_states,
        step=int(step),
    )
    roundtrip_hash_pass = (
        str(blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
        == str(reblob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
    )
    q_scale_bounded_hash_pass = all(
        tensor_sha256(states[key].q_levels) == tensor_sha256(loaded_states[key].q_levels)
        and tensor_sha256(states[key].frozen_scale) == tensor_sha256(loaded_states[key].frozen_scale)
        and _roundtrip_payload_sha256(
            _tensor_state_roundtrip_payload(states[key])["bounded_accumulator"]
        )
        == _roundtrip_payload_sha256(
            _tensor_state_roundtrip_payload(loaded_states[key])["bounded_accumulator"]
        )
        for key in sorted(states)
    )

    prior_training = bool(resumed_model.training)
    try:
        resumed_model.train(False)
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=False,
        ):
            expected_sidecar_output = output_fn(resumed_model, batch).detach().cpu()
        with torch.no_grad():
            for module in resumed_eligible.values():
                module.weight.fill_(float(poison_value))
        normal_poisoned_output = output_fn(resumed_model, batch).detach().cpu()
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=False,
        ):
            resumed_sidecar_output = output_fn(resumed_model, batch).detach().cpu()
    finally:
        resumed_model.train(prior_training)

    normal_poison_sensitivity = not torch.allclose(
        expected_sidecar_output,
        normal_poisoned_output,
        atol=1e-6,
        rtol=1e-6,
    )
    resumed_uses_sidecar = torch.allclose(
        expected_sidecar_output,
        resumed_sidecar_output,
        atol=1e-6,
        rtol=1e-6,
    )
    poisoned_bypass_falsified = bool(normal_poison_sensitivity and resumed_uses_sidecar)

    rank_spec = default_dry_run_rank_vote_spec()
    update_spec = vote_update_spec or _default_local_vote_update_spec()
    vote_specs_by_key = {key: update_spec for key in loaded_states}
    weighted_grad_by_key: dict[str, torch.Tensor] = {}
    loss_finite = False
    prior_training = bool(resumed_model.training)
    try:
        _emit_phase("PHASE_START", "forward_backward")
        resumed_model.train(True)
        resumed_model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=True,
        ) as handle:
            loss = forward_loss_fn(resumed_model, batch)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("2C4a forward_loss_fn must return a torch.Tensor loss")
            loss_to_backward = loss if loss.numel() == 1 else loss.mean()
            loss_finite = bool(torch.isfinite(loss_to_backward.detach()).item())
            loss_to_backward.backward()
            for key, state in sorted(loaded_states.items()):
                weighted_grad_by_key[key] = handle.weighted_grad(key)
    finally:
        resumed_model.train(prior_training)

    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_forward_backward(weighted_grad_by_key)
    _emit_phase("PHASE_END", "forward_backward")

    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=weighted_grad_by_key,
        q_levels_by_key={key: state.q_levels for key, state in loaded_states.items()},
        rank_spec=rank_spec,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )
    sparse_events_by_key = dict(path["sparse_events_by_key"])

    _emit_phase("PHASE_START", "update")
    step_result = apply_bounded_delta_vote_step(
        loaded_states,
        None,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_update()
    _emit_phase("PHASE_END", "update")
    _emit_phase("PHASE_START", "emission")
    post_blob = build_trainer_sub2_authority_checkpoint_blob(
        resumed_model,
        eligible_modules=resumed_eligible,
        tensor_states=step_result.tensor_states,
        step=int(step) + 1,
    )
    post_resume_mutated = (
        str(post_blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
        != str(blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
        and int(step_result.global_summary.get("q_changed_count", 0)) > 0
    )
    post_model = fresh_model_fn().to(device=device)
    post_eligible = select_trainer_eligible_bitlinears(
        post_model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    post_loaded_states = load_trainer_sub2_authority_checkpoint_blob(
        post_model,
        post_blob,
        eligible_modules=post_eligible,
        device=device,
    )
    post_reblob = build_trainer_sub2_authority_checkpoint_blob(
        post_model,
        eligible_modules=post_eligible,
        tensor_states=post_loaded_states,
        step=int(step) + 1,
    )
    post_payload_hash_roundtrip_pass = (
        str(post_blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
        == str(post_reblob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
    )
    post_survives_rebuild = (
        set(post_loaded_states) == set(step_result.tensor_states)
        and post_payload_hash_roundtrip_pass
    )
    shadow_free_loaded = all(state.exact_accumulator_shadow is None for state in loaded_states.values())
    shadow_free_post = all(state.exact_accumulator_shadow is None for state in step_result.tensor_states.values())
    total_sparse_events = sum(events.event_count() for events in sparse_events_by_key.values())
    dense_saved = bool(blob["trainer_sub2_authority"].get("dense_int16_persistent_accumulator_saved"))
    dense_loaded = any(state.exact_accumulator_shadow is not None for state in loaded_states.values())
    pass_receipt = bool(
        excluded
        and raw_fallback_rejected
        and roundtrip_hash_pass
        and q_scale_bounded_hash_pass
        and poisoned_bypass_falsified
        and loss_finite
        and total_sparse_events > 0
        and bool(step_result.global_summary.get("candidate_local_update_pass"))
        and post_resume_mutated
        and post_survives_rebuild
        and shadow_free_loaded
        and shadow_free_post
        and not dense_saved
        and not dense_loaded
    )
    # PLAN_v7 B2 / gate-2 D1: pure helper owns post_resume_update_proof assembly.
    # Frozen discriminator key names retained here for source-scan hostiles
    # (sparse_vote_authority_mode, sparse_vote_authority_only,
    # dense_vote_authority_skipped, votes_by_key_applied, transient_over2_tensors).
    post_resume_update_proof = assemble_b2_post_resume_update_proof(
        path=path,
        loss_finite=bool(loss_finite),
        total_sparse_events=int(total_sparse_events),
        step_result_global_summary=dict(step_result.global_summary),
        post_resume_mutated=bool(post_resume_mutated),
    )
    assert post_resume_update_proof["sparse_vote_authority_mode"] == str(path["resolved_mode"])
    assert post_resume_update_proof["sparse_vote_authority_only"] is True
    assert post_resume_update_proof["dense_vote_authority_skipped"] is True
    assert post_resume_update_proof["votes_by_key_applied"] is None
    assert "transient_over2_tensors" in post_resume_update_proof
    receipt = TrainerSub2AuthorityRoundtripReceipt(
        schema_version=TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        target_name=TRAINER_SUB2_ROUNDTRIP_TARGET_NAME,
        pass_receipt=pass_receipt,
        dry_run=True,
        gpu_launched=False,
        checkpoint_written=False,
        learner_update_called=True,
        optimizer_step_called=False,
        persistent_authority_state_roundtrip_pass=bool(roundtrip_hash_pass),
        trainer_state_mutation_uses_sub2_authority=bool(post_resume_mutated),
        resumed_forward_uses_sidecar_authority=bool(resumed_uses_sidecar),
        poisoned_fp_master_bypass_falsified=bool(poisoned_bypass_falsified),
        eligible_fp_masters_authoritative=False,
        eligible_fp_master_keys_excluded_from_authoritative_model_state=bool(excluded),
        raw_state_dict_eligible_weight_fallback_rejected=bool(raw_fallback_rejected),
        normal_bitlinear_weight_forward_not_claimed=True,
        dense_int16_persistent_accumulator_saved=False,
        dense_int16_persistent_accumulator_loaded=False,
        q_scale_sidecar_bounded_hash_roundtrip_pass=bool(q_scale_bounded_hash_pass),
        post_resume_update_mutated_resumed_sub2_authority=bool(
            post_resume_mutated and post_survives_rebuild
        ),
        update_law_quality_claim=False,
        learning_claim=False,
        optimizer_credit_state_resolved=False,
        credit_ranking_uninformative_update_law_pivot_deferred=True,
        trainer_entrypoint_can_construct_sub2_authority=True,
        trainer_entrypoint_uses_candidate=False,
        live_runtime_authority_converted=False,
        readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        broad_runtime_authority_converted=False,
        full_sub2_runtime_readiness_claim=False,
        use_ternary_bulk_required=True,
        use_ternary_bulk_observed=bool(use_ternary_bulk),
        eligible_scope=str(eligible_scope),
        eligible_module_count=len(eligible),
        eligible_state_keys=tuple(sorted(states)),
        eligible_weight_count=sum(int(state.q_levels.numel()) for state in states.values()),
        checkpoint_payload_summary={
            "schema_version": blob["schema_version"],
            "model_state_key_count": len(blob["model_state"]),
            "eligible_weight_state_keys": list(eligible_weight_keys),
            "eligible_weight_keys_excluded": bool(excluded),
            "authoritative_state_payload_sha256": blob["trainer_sub2_authority"][
                "authoritative_state_payload_sha256"
            ],
            "post_update_authoritative_state_payload_sha256": post_blob[
                "trainer_sub2_authority"
            ]["authoritative_state_payload_sha256"],
            "checkpoint_blob_sha256": blob["checkpoint_blob_sha256"],
            "checkpoint_written": False,
            "optimizer_state_included": False,
        },
        checkpoint_load_proof={
            "strict_noneligible_model_state_load": True,
            "missing_keys_exactly_eligible_weights": True,
            "raw_state_dict_eligible_weight_fallback_rejected": bool(raw_fallback_rejected),
            "loaded_exact_accumulator_shadow_count": 0,
            "post_update_payload_survives_rebuild": bool(post_survives_rebuild),
            "post_update_payload_hash_roundtrip_pass": bool(post_payload_hash_roundtrip_pass),
        },
        poison_forward_proof={
            "poison_value": float(poison_value),
            "normal_no_context_forward_changed_after_poison": bool(normal_poison_sensitivity),
            "resumed_context_forward_matches_sidecar_expected": bool(resumed_uses_sidecar),
            "expected_sidecar_output_sha256": _roundtrip_tensor_sha256(expected_sidecar_output),
            "normal_poisoned_output_sha256": _roundtrip_tensor_sha256(normal_poisoned_output),
            "resumed_sidecar_output_sha256": _roundtrip_tensor_sha256(resumed_sidecar_output),
        },
        post_resume_update_proof=post_resume_update_proof,
        proof_anchors=(
            "trainer_sub2_authority.py:build_trainer_sub2_authority_checkpoint_blob",
            "trainer_sub2_authority.py:load_trainer_sub2_authority_checkpoint_blob",
            "bounded_delta_learner.py:679",
            "bounded_delta_learner.py:1025",
            "bit_linear.py:108",
        ),
        non_claims=TRAINER_SUB2_ROUNDTRIP_NON_CLAIMS,
    )
    validate_trainer_sub2_authority_roundtrip_receipt(receipt)
    _emit_phase("PHASE_END", "emission")
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
    if not receipt.default_off_trainer_active_controls_inactive_proven:
        raise ValueError("2C3 active controls inactive proof did not pass")
    if receipt.global_cap_spec_passed:
        raise ValueError("2C3 inactive proof cannot pass a global cap spec")
    if receipt.global_rate_cap_enabled:
        raise ValueError("2C3 inactive proof cannot enable global cap")
    if receipt.deferred_backlog_input_present:
        raise ValueError("2C3 inactive proof cannot accept deferred backlog input")
    if receipt.deferred_backlog_output_entry_count != 0:
        raise ValueError("2C3 inactive proof cannot emit deferred backlog")
    if receipt.replay_ce_veto_maps_present:
        raise ValueError("2C3 inactive proof cannot pass replay CE veto maps")
    if receipt.pc_aux_maps_present:
        raise ValueError("2C3 inactive proof cannot pass PC auxiliary maps")
    if receipt.pc_aux_mode_effective != "not_enabled":
        raise ValueError("2C3 inactive proof cannot enable PC auxiliary mode")
    if receipt.front_c_identity_observer_present:
        raise ValueError("2C3 inactive proof cannot pass front-C identity observer")
    if not receipt.candidate_mode_rejects_active_controls:
        raise ValueError("2C3 inactive proof needs candidate-mode active-control rejection")
    if not receipt.trainer_builder_has_no_active_control_parameters:
        raise ValueError("2C3 inactive proof requires no trainer active-control parameters")
    candidate_summary = dict(receipt.candidate_step_summary)
    if not bool(candidate_summary.get("default_off_trainer_active_controls_inactive_proven")):
        raise ValueError("2C3 candidate summary did not prove inactive controls")
    if bool(candidate_summary.get("global_cap_spec_passed")):
        raise ValueError("2C3 candidate summary cannot pass a global cap spec")
    if bool(candidate_summary.get("global_rate_cap_enabled")):
        raise ValueError("2C3 candidate summary cannot enable global cap")
    if bool(candidate_summary.get("deferred_backlog_input_present")):
        raise ValueError("2C3 candidate summary cannot accept deferred backlog input")
    if int(candidate_summary.get("deferred_backlog_output_entry_count", -1)) != 0:
        raise ValueError("2C3 candidate summary cannot emit deferred backlog")
    if bool(candidate_summary.get("replay_ce_veto_maps_present")):
        raise ValueError("2C3 candidate summary cannot pass replay CE veto maps")
    if bool(candidate_summary.get("pc_aux_maps_present")):
        raise ValueError("2C3 candidate summary cannot pass PC auxiliary maps")
    if candidate_summary.get("pc_aux_mode_effective") != "not_enabled":
        raise ValueError("2C3 candidate summary cannot enable PC auxiliary mode")
    if bool(candidate_summary.get("front_c_identity_observer_present")):
        raise ValueError("2C3 candidate summary cannot pass front-C identity observer")
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


def validate_trainer_sub2_authority_roundtrip_receipt(
    receipt: TrainerSub2AuthorityRoundtripReceipt,
) -> None:
    if receipt.schema_version != TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION:
        raise ValueError("2C4a trainer authority roundtrip schema version mismatch")
    if receipt.target_name != TRAINER_SUB2_ROUNDTRIP_TARGET_NAME:
        raise ValueError("2C4a trainer authority roundtrip target name mismatch")
    if not receipt.dry_run or receipt.gpu_launched or receipt.checkpoint_written:
        raise ValueError("2C4a must stay dry-run/no GPU/no committed checkpoint")
    if not receipt.learner_update_called:
        raise ValueError("2C4a must exercise a post-resume local authority update")
    if receipt.optimizer_step_called:
        raise ValueError("2C4a must exit before optimizer step")
    required_true = {
        "persistent authority state roundtrip": receipt.persistent_authority_state_roundtrip_pass,
        "trainer state mutation uses sub2 authority": receipt.trainer_state_mutation_uses_sub2_authority,
        "resumed forward uses sidecar authority": receipt.resumed_forward_uses_sidecar_authority,
        "poisoned FP-master bypass falsified": receipt.poisoned_fp_master_bypass_falsified,
        "eligible FP master keys excluded": (
            receipt.eligible_fp_master_keys_excluded_from_authoritative_model_state
        ),
        "raw state_dict eligible fallback rejected": (
            receipt.raw_state_dict_eligible_weight_fallback_rejected
        ),
        "normal BitLinear weight forward not claimed": (
            receipt.normal_bitlinear_weight_forward_not_claimed
        ),
        "q/scale/bounded hash roundtrip": receipt.q_scale_sidecar_bounded_hash_roundtrip_pass,
        "post-resume authority mutation": (
            receipt.post_resume_update_mutated_resumed_sub2_authority
        ),
        "credit-ranking pivot deferred": (
            receipt.credit_ranking_uninformative_update_law_pivot_deferred
        ),
    }
    for label, value in required_true.items():
        if not bool(value):
            raise ValueError(f"2C4a missing required proof: {label}")
    forbidden_true = {
        "eligible FP masters authoritative": receipt.eligible_fp_masters_authoritative,
        "dense int16 persistent accumulator saved": (
            receipt.dense_int16_persistent_accumulator_saved
        ),
        "dense int16 persistent accumulator loaded": (
            receipt.dense_int16_persistent_accumulator_loaded
        ),
        "update law quality claim": receipt.update_law_quality_claim,
        "learning claim": receipt.learning_claim,
        "optimizer credit state resolved": receipt.optimizer_credit_state_resolved,
        "broad trainer_entrypoint_uses_candidate": receipt.trainer_entrypoint_uses_candidate,
        "live runtime conversion": receipt.live_runtime_authority_converted,
        "readiness row flip": receipt.readiness_row_flip_authorized,
        "broad runtime authority conversion": receipt.broad_runtime_authority_converted,
        "full-sub2 readiness claim": receipt.full_sub2_runtime_readiness_claim,
    }
    for label, value in forbidden_true.items():
        if bool(value):
            raise ValueError(f"2C4a forbidden claim/flag set: {label}")
    if receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("2C4a cannot authorize readiness-row surface names")
    if not receipt.use_ternary_bulk_required or not receipt.use_ternary_bulk_observed:
        raise ValueError("2C4a requires observed ternary bulk")
    if receipt.eligible_module_count <= 0 or receipt.eligible_weight_count <= 0:
        raise ValueError("2C4a needs at least one eligible BitLinear module")
    if bool(receipt.checkpoint_payload_summary.get("checkpoint_written")):
        raise ValueError("2C4a checkpoint payload must remain memory/tmp proof only")
    if not bool(receipt.checkpoint_payload_summary.get("eligible_weight_keys_excluded")):
        raise ValueError("2C4a checkpoint payload did not exclude eligible FP masters")
    if not bool(receipt.checkpoint_load_proof.get("strict_noneligible_model_state_load")):
        raise ValueError("2C4a strict non-eligible load proof missing")
    if not bool(receipt.checkpoint_load_proof.get("missing_keys_exactly_eligible_weights")):
        raise ValueError("2C4a load did not prove exact eligible-weight omissions")
    if int(receipt.checkpoint_load_proof.get("loaded_exact_accumulator_shadow_count", -1)) != 0:
        raise ValueError("2C4a loaded dense exact accumulator shadows")
    if not bool(receipt.checkpoint_load_proof.get("post_update_payload_hash_roundtrip_pass")):
        raise ValueError("2C4a post-update payload hash did not roundtrip after second load")
    poison = dict(receipt.poison_forward_proof)
    if not bool(poison.get("normal_no_context_forward_changed_after_poison")):
        raise ValueError("2C4a poison proof did not show normal FP-master sensitivity")
    if not bool(poison.get("resumed_context_forward_matches_sidecar_expected")):
        raise ValueError("2C4a poison proof did not show resumed sidecar authority")
    update = dict(receipt.post_resume_update_proof)
    if not bool(update.get("candidate_local_update_pass")):
        raise ValueError("2C4a post-resume candidate update did not pass")
    if int(update.get("total_sparse_vote_event_count", 0)) <= 0:
        raise ValueError("2C4a post-resume update needs sparse vote events")
    if int(update.get("q_changed_count", 0)) <= 0:
        raise ValueError("2C4a post-resume update must mutate q authority")
    if bool(update.get("candidate_dense_decode_used")):
        raise ValueError("2C4a candidate update cannot dense-decode as authority")
    if bool(update.get("candidate_dense_vote_authority_used")):
        raise ValueError("2C4a candidate update cannot use dense vote authority")
    if tuple(receipt.non_claims) != TRAINER_SUB2_ROUNDTRIP_NON_CLAIMS:
        raise ValueError("2C4a non-claims changed")
    if not receipt.pass_receipt:
        raise ValueError("2C4a trainer authority roundtrip proof did not pass")


TRAINER_SUB2_LIVE_CONVERSION_SCHEMA_VERSION = (
    "hrm_text_158_p1_trainer_sub2_authority/v0.live_conversion_proof"
)
TRAINER_SUB2_LIVE_CONVERSION_TARGET_NAME = (
    "p1_trainer_live_checkpoint_authority_conversion"
)
AUTHORIZED_P1B_SURFACE_TUPLE = (
    "persistent_qacc_authority",
    "dense_int16_persistent_accumulator_absence",
    "q_sidecar_vote_carrier",
)
AUTHORIZED_P1B_SURFACE_TUPLE_2ROW = (
    "persistent_qacc_authority",
    "dense_int16_persistent_accumulator_absence",
)
P1_LIVE_PARITY_ATOL = 1e-5
P1B_VOTE_SMOKE_STEP_BOUND = 1
TRAINER_SUB2_LIVE_CONVERSION_NON_CLAIMS = (
    "P1b proves live production checkpoint save/load routing + eval parent install only",
    "normal optimizer-resume/full training from P1 sidecar checkpoints is NOT proved",
    "readiness row flip authorizes persistent-lane surfaces only; main/diag stay blocked",
    "not learning, acquisition, throughput, GPU residency, banked .pt mutation, or broad runtime conversion",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _eligible_state_keys_sha256(keys: Sequence[str]) -> str:
    h = hashlib.sha256()
    for key in sorted(str(item) for item in keys):
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class TrainerSub2AuthorityLiveConversionReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    dry_run: bool
    gpu_launched: bool
    optimizer_step_called: bool
    checkpoint_written: bool
    checkpoint_written_to_banked_parent: bool
    learner_update_called: bool
    live_runtime_authority_converted: bool
    readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    source_commit_sha: str
    proof_command_argv: tuple[str, ...]
    checkpoint_format: str
    p1_envelope_sha256: str
    inner_authoritative_state_payload_sha256: str
    eligible_state_keys: tuple[str, ...]
    eligible_state_keys_sha256: str
    eligible_module_count: int
    load_routing_result: str
    dense_int16_persistent_accumulator_saved: bool
    dense_int16_persistent_accumulator_loaded: bool
    raw_state_dict_eligible_weight_fallback_rejected: bool
    cached_weight_parent_install_proven: bool
    parity_max_abs_diff_by_site: dict[str, float]
    parity_pass: bool
    vote_carrier_subproof_exercised: bool
    poisoned_fp_master_bypass_falsified: bool
    total_sparse_vote_event_count: int
    q_changed_count: int
    post_resume_update_mutated: bool
    authority_state_shadow_free_after: bool
    post_resume_payload_sha256_before: str
    post_resume_payload_sha256_after: str
    post_resume_payload_hash_roundtrip_pass: bool
    loss_finite: bool
    q_sidecar_vote_carrier_deferred: bool
    q_sidecar_deferred_reason: str
    normal_optimizer_resume_from_p1_sidecar_not_proved: bool
    full_training_authority_from_p1_sidecar_not_proved: bool
    learning_claim: bool
    acquisition_claim: bool
    full_sub2_runtime_readiness_claim: bool
    ready_for_main_science: bool
    ready_for_pre_full_stack_diagnostic: bool
    broad_runtime_authority_conversion: bool
    w6_parent_sha256_before: str
    w6_parent_sha256_after: str
    proof_anchors: tuple[str, ...]
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_row_flip_authorized_surface_names"] = list(
            self.readiness_row_flip_authorized_surface_names
        )
        payload["proof_command_argv"] = list(self.proof_command_argv)
        payload["eligible_state_keys"] = list(self.eligible_state_keys)
        payload["proof_anchors"] = list(self.proof_anchors)
        payload["non_claims"] = list(self.non_claims)
        return payload


def live_conversion_receipt_from_dict(payload: Mapping[str, Any]) -> TrainerSub2AuthorityLiveConversionReceipt:
    return TrainerSub2AuthorityLiveConversionReceipt(
        schema_version=str(payload["schema_version"]),
        target_name=str(payload["target_name"]),
        pass_receipt=bool(payload["pass_receipt"]),
        dry_run=bool(payload["dry_run"]),
        gpu_launched=bool(payload["gpu_launched"]),
        optimizer_step_called=bool(payload["optimizer_step_called"]),
        checkpoint_written=bool(payload["checkpoint_written"]),
        checkpoint_written_to_banked_parent=bool(payload["checkpoint_written_to_banked_parent"]),
        learner_update_called=bool(payload["learner_update_called"]),
        live_runtime_authority_converted=bool(payload["live_runtime_authority_converted"]),
        readiness_row_flip_authorized=bool(payload["readiness_row_flip_authorized"]),
        readiness_row_flip_authorized_surface_names=tuple(
            str(name) for name in payload["readiness_row_flip_authorized_surface_names"]
        ),
        source_commit_sha=str(payload["source_commit_sha"]),
        proof_command_argv=tuple(str(item) for item in payload["proof_command_argv"]),
        checkpoint_format=str(payload["checkpoint_format"]),
        p1_envelope_sha256=str(payload["p1_envelope_sha256"]),
        inner_authoritative_state_payload_sha256=str(
            payload["inner_authoritative_state_payload_sha256"]
        ),
        eligible_state_keys=tuple(str(key) for key in payload["eligible_state_keys"]),
        eligible_state_keys_sha256=str(payload["eligible_state_keys_sha256"]),
        eligible_module_count=int(payload["eligible_module_count"]),
        load_routing_result=str(payload["load_routing_result"]),
        dense_int16_persistent_accumulator_saved=bool(
            payload["dense_int16_persistent_accumulator_saved"]
        ),
        dense_int16_persistent_accumulator_loaded=bool(
            payload["dense_int16_persistent_accumulator_loaded"]
        ),
        raw_state_dict_eligible_weight_fallback_rejected=bool(
            payload["raw_state_dict_eligible_weight_fallback_rejected"]
        ),
        cached_weight_parent_install_proven=bool(payload["cached_weight_parent_install_proven"]),
        parity_max_abs_diff_by_site={
            str(key): float(value)
            for key, value in dict(payload["parity_max_abs_diff_by_site"]).items()
        },
        parity_pass=bool(payload["parity_pass"]),
        vote_carrier_subproof_exercised=bool(payload["vote_carrier_subproof_exercised"]),
        poisoned_fp_master_bypass_falsified=bool(payload["poisoned_fp_master_bypass_falsified"]),
        total_sparse_vote_event_count=int(payload["total_sparse_vote_event_count"]),
        q_changed_count=int(payload["q_changed_count"]),
        post_resume_update_mutated=bool(payload["post_resume_update_mutated"]),
        authority_state_shadow_free_after=bool(payload["authority_state_shadow_free_after"]),
        post_resume_payload_sha256_before=str(payload["post_resume_payload_sha256_before"]),
        post_resume_payload_sha256_after=str(payload["post_resume_payload_sha256_after"]),
        post_resume_payload_hash_roundtrip_pass=bool(
            payload["post_resume_payload_hash_roundtrip_pass"]
        ),
        loss_finite=bool(payload["loss_finite"]),
        q_sidecar_vote_carrier_deferred=bool(payload["q_sidecar_vote_carrier_deferred"]),
        q_sidecar_deferred_reason=str(payload["q_sidecar_deferred_reason"]),
        normal_optimizer_resume_from_p1_sidecar_not_proved=bool(
            payload["normal_optimizer_resume_from_p1_sidecar_not_proved"]
        ),
        full_training_authority_from_p1_sidecar_not_proved=bool(
            payload["full_training_authority_from_p1_sidecar_not_proved"]
        ),
        learning_claim=bool(payload["learning_claim"]),
        acquisition_claim=bool(payload["acquisition_claim"]),
        full_sub2_runtime_readiness_claim=bool(payload["full_sub2_runtime_readiness_claim"]),
        ready_for_main_science=bool(payload["ready_for_main_science"]),
        ready_for_pre_full_stack_diagnostic=bool(
            payload["ready_for_pre_full_stack_diagnostic"]
        ),
        broad_runtime_authority_conversion=bool(payload["broad_runtime_authority_conversion"]),
        w6_parent_sha256_before=str(payload.get("w6_parent_sha256_before", "")),
        w6_parent_sha256_after=str(payload.get("w6_parent_sha256_after", "")),
        proof_anchors=tuple(str(item) for item in payload.get("proof_anchors", ())),
        non_claims=tuple(str(item) for item in payload.get("non_claims", ())),
    )


def _p1_raw_fallback_rejected(
    *,
    blob: Mapping[str, Any],
    eligible: Mapping[str, BitLinear],
    fresh_model_fn: Callable[[], torch.nn.Module],
    use_ternary_bulk: bool,
    eligible_scope: str,
    device: torch.device | str,
) -> bool:
    eligible_weight_keys = _eligible_weight_state_keys(eligible)
    if not eligible_weight_keys:
        return False
    poisoned_blob = dict(blob)
    poisoned_model_state = dict(blob["model_state"])
    first_key = eligible_weight_keys[0]
    poisoned_model_state[first_key] = eligible[sorted(eligible)[0]].weight.detach().cpu().clone()
    poisoned_blob["model_state"] = poisoned_model_state
    try:
        raw_fresh = fresh_model_fn().to(device=device)
        raw_eligible = select_trainer_eligible_bitlinears(
            raw_fresh,
            use_ternary_bulk=use_ternary_bulk,
            eligible_scope=eligible_scope,
        )
        load_trainer_sub2_authority_checkpoint_blob(
            raw_fresh,
            poisoned_blob,
            eligible_modules=raw_eligible,
            device=device,
        )
    except ValueError as exc:
        return "raw state_dict eligible-weight fallback rejected" in str(exc)
    return False


def _run_live_p1_vote_carrier_subproof(
    *,
    resumed_model: torch.nn.Module,
    loaded_states: Mapping[str, BoundedDeltaTensorState],
    resumed_eligible: Mapping[str, BitLinear],
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    forward_output_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    use_ternary_bulk: bool,
    eligible_scope: str,
    device: torch.device | str,
    step: int,
    vote_update_spec: VoteUpdateSpec | None,
    poison_value: float,
    payload_sha_before: str,
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> dict[str, Any]:
    prior_training = bool(resumed_model.training)
    try:
        resumed_model.train(False)
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=False,
        ):
            expected_sidecar_output = forward_output_fn(resumed_model, batch).detach().cpu()
        with torch.no_grad():
            for module in resumed_eligible.values():
                module.weight.fill_(float(poison_value))
        normal_poisoned_output = forward_output_fn(resumed_model, batch).detach().cpu()
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=False,
        ):
            resumed_sidecar_output = forward_output_fn(resumed_model, batch).detach().cpu()
    finally:
        resumed_model.train(prior_training)

    poisoned_bypass_falsified = bool(
        not torch.allclose(expected_sidecar_output, normal_poisoned_output, atol=1e-6, rtol=1e-6)
        and torch.allclose(expected_sidecar_output, resumed_sidecar_output, atol=1e-6, rtol=1e-6)
    )

    rank_spec = default_dry_run_rank_vote_spec()
    update_spec = vote_update_spec or _default_local_vote_update_spec()
    vote_specs_by_key = {key: update_spec for key in loaded_states}
    weighted_grad_by_key: dict[str, torch.Tensor] = {}
    loss_finite = False
    prior_training = bool(resumed_model.training)
    try:
        _emit_phase("PHASE_START", "forward_backward")
        resumed_model.zero_grad(set_to_none=True)
        resumed_model.train(True)
        with trainer_authoritative_forward_context(
            resumed_eligible,
            loaded_states,
            device=device,
            requires_grad=True,
        ) as handle:
            loss = forward_loss_fn(resumed_model, batch)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("P1b forward_loss_fn must return a torch.Tensor loss")
            loss_to_backward = loss if loss.numel() == 1 else loss.mean()
            loss_finite = bool(torch.isfinite(loss_to_backward.detach()).item())
            loss_to_backward.backward()
            for key, state in sorted(loaded_states.items()):
                weighted_grad_by_key[key] = handle.weighted_grad(key)
    finally:
        resumed_model.zero_grad(set_to_none=True)
        resumed_model.train(prior_training)

    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_forward_backward(weighted_grad_by_key)
    _emit_phase("PHASE_END", "forward_backward")

    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=weighted_grad_by_key,
        q_levels_by_key={key: state.q_levels for key, state in loaded_states.items()},
        rank_spec=rank_spec,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )
    sparse_events_by_key = dict(path["sparse_events_by_key"])

    _emit_phase("PHASE_START", "update")
    step_result = apply_bounded_delta_vote_step(
        dict(loaded_states),
        None,
        vote_specs_by_key,
        candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    if _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS is not None:
        _ACTIVE_SPARSE_VOTE_EXECUTION_WITNESS.note_update()
    _emit_phase("PHASE_END", "update")
    _emit_phase("PHASE_START", "emission")
    post_blob = build_trainer_sub2_authority_checkpoint_blob(
        resumed_model,
        eligible_modules=resumed_eligible,
        tensor_states=step_result.tensor_states,
        step=int(step) + 1,
    )
    payload_sha_after = str(
        post_blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"]
    )
    post_resume_mutated = bool(
        payload_sha_after != str(payload_sha_before)
        and int(step_result.global_summary.get("q_changed_count", 0)) > 0
    )
    post_model = fresh_model_fn().to(device=device)
    post_eligible = select_trainer_eligible_bitlinears(
        post_model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    post_loaded_states = load_trainer_sub2_authority_checkpoint_blob(
        post_model,
        post_blob,
        eligible_modules=post_eligible,
        device=device,
    )
    post_reblob = build_trainer_sub2_authority_checkpoint_blob(
        post_model,
        eligible_modules=post_eligible,
        tensor_states=post_loaded_states,
        step=int(step) + 1,
    )
    roundtrip_pass = (
        str(post_blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
        == str(post_reblob["trainer_sub2_authority"]["authoritative_state_payload_sha256"])
    )
    shadow_free_after = all(
        state.exact_accumulator_shadow is None
        for state in step_result.tensor_states.values()
    )
    total_sparse_events = sum(events.event_count() for events in sparse_events_by_key.values())
    q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
    out = {
        "poisoned_fp_master_bypass_falsified": poisoned_bypass_falsified,
        "total_sparse_vote_event_count": int(total_sparse_events),
        "q_changed_count": q_changed_count,
        "post_resume_update_mutated": post_resume_mutated,
        "authority_state_shadow_free_after": shadow_free_after,
        "post_resume_payload_sha256_before": str(payload_sha_before),
        "post_resume_payload_sha256_after": str(payload_sha_after),
        "post_resume_payload_hash_roundtrip_pass": bool(roundtrip_pass),
        "loss_finite": bool(loss_finite),
    }
    # R2: B3 emission stays open through P1b receipt build/validate + landing
    # wrapper construct/validate; wrapper closes exactly once.
    return out


def compute_p1_parent_parity_max_abs_diff_by_site(
    *,
    legacy_checkpoint: Mapping[str, Any],
    p1_checkpoint: Mapping[str, Any],
    fresh_model_fn: Callable[[], torch.nn.Module],
    site_batches: Mapping[str, Mapping[str, Any]],
    forward_logits_fn: Callable[..., torch.Tensor],
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    atol: float = P1_LIVE_PARITY_ATOL,
) -> dict[str, float]:
    parent_ref = fresh_model_fn().to(device=device)
    load_train_checkpoint_into_model(
        parent_ref,
        legacy_checkpoint,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
        inference_only=True,
        sub2_live_enabled=False,
    )
    parent_ref.eval()

    parent_p1 = fresh_model_fn().to(device=device)
    load_train_checkpoint_into_model(
        parent_p1,
        p1_checkpoint,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
        inference_only=True,
        sub2_live_enabled=True,
    )
    parent_p1.eval()

    max_diffs: dict[str, float] = {}
    for site, batch in site_batches.items():
        logits_ref = forward_logits_fn(parent_ref, batch).detach().cpu()
        logits_p1 = forward_logits_fn(parent_p1, batch).detach().cpu()
        if not torch.isfinite(logits_ref).all() or not torch.isfinite(logits_p1).all():
            raise ValueError(f"P1 parity site {site!r} produced non-finite logits")
        max_abs_diff = float((logits_p1 - logits_ref).abs().max().item())
        if max_abs_diff > float(atol):
            raise ValueError(
                f"P1 parity site {site!r} failed: max_abs_diff={max_abs_diff} > {atol}"
            )
        max_diffs[str(site)] = max_abs_diff
    return max_diffs


def _vote_subproof_passes_three_row(vote: Mapping[str, Any]) -> bool:
    return bool(
        vote.get("loss_finite")
        and vote.get("poisoned_fp_master_bypass_falsified")
        and int(vote.get("total_sparse_vote_event_count", 0)) > 0
        and int(vote.get("q_changed_count", 0)) > 0
        and vote.get("post_resume_update_mutated")
        and vote.get("authority_state_shadow_free_after")
        and vote.get("post_resume_payload_hash_roundtrip_pass")
    )


def _build_trainer_sub2_authority_live_conversion_receipt_impl(
    *,
    p1_checkpoint: Mapping[str, Any],
    p1_envelope_bytes: bytes,
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    forward_output_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor] | None,
    parity_max_abs_diff_by_site: Mapping[str, float],
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    step: int = 0,
    source_commit_sha: str = "",
    proof_command_argv: Sequence[str] = (),
    vote_update_spec: VoteUpdateSpec | None = None,
    poison_value: float = 17.0,
    w6_parent_sha256_before: str = "",
    w6_parent_sha256_after: str = "",
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> TrainerSub2AuthorityLiveConversionReceipt:
    if not is_p1_live_sub2_checkpoint(p1_checkpoint):
        raise ValueError("P1b live conversion requires a P1 live checkpoint envelope")
    if int(P1B_VOTE_SMOKE_STEP_BOUND) != 1:
        raise ValueError("P1b vote smoke step bound must remain fixed at 1")

    output_fn = forward_output_fn or _default_forward_output
    inner_authority = dict(p1_checkpoint["trainer_sub2_authority"])
    payload_sha_before = str(inner_authority["authoritative_state_payload_sha256"])
    dense_saved = bool(inner_authority.get("dense_int16_persistent_accumulator_saved"))
    dense_loaded = False

    source_for_blob = fresh_model_fn().to(device=device)
    source_eligible = select_trainer_eligible_bitlinears(
        source_for_blob,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    raw_fallback_rejected = _p1_raw_fallback_rejected(
        blob=p1_checkpoint,
        eligible=source_eligible,
        fresh_model_fn=fresh_model_fn,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
    )

    resumed_model = fresh_model_fn().to(device=device)
    load_result = load_train_checkpoint_into_model(
        resumed_model,
        p1_checkpoint,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
        inference_only=False,
        sub2_live_enabled=True,
    )
    if load_result.routing != "p1_live" or load_result.authority_states is None:
        raise ValueError("P1b live conversion load did not route through p1_live")

    loaded_states = load_result.authority_states
    resumed_eligible = select_trainer_eligible_bitlinears(
        resumed_model,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
    )
    dense_loaded = any(state.exact_accumulator_shadow is not None for state in loaded_states.values())

    vote = _run_live_p1_vote_carrier_subproof(
        resumed_model=resumed_model,
        loaded_states=loaded_states,
        resumed_eligible=resumed_eligible,
        fresh_model_fn=fresh_model_fn,
        batch=batch,
        forward_loss_fn=forward_loss_fn,
        forward_output_fn=output_fn,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
        step=int(step),
        vote_update_spec=vote_update_spec,
        poison_value=float(poison_value),
        payload_sha_before=payload_sha_before,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )

    parity_map = {str(key): float(value) for key, value in parity_max_abs_diff_by_site.items()}
    required_sites = ("cache_builder", "main_kl", "retained_fallback")
    parity_pass = all(site in parity_map for site in required_sites) and all(
        parity_map[site] <= P1_LIVE_PARITY_ATOL for site in required_sites
    )
    cached_install_proven = bool(parity_pass)

    three_row = _vote_subproof_passes_three_row(vote)
    if three_row:
        authorized = AUTHORIZED_P1B_SURFACE_TUPLE
        deferred = False
        deferred_reason = ""
    else:
        authorized = AUTHORIZED_P1B_SURFACE_TUPLE_2ROW
        deferred = True
        if int(vote.get("q_changed_count", 0)) <= 0:
            deferred_reason = (
                "q_sidecar_vote_carrier deferred: bounded CPU smoke produced "
                f"q_changed_count={int(vote.get('q_changed_count', 0))} (required >0)"
            )
        else:
            deferred_reason = (
                "q_sidecar_vote_carrier deferred: vote-carrier subproof gates failed "
                f"(poison={vote.get('poisoned_fp_master_bypass_falsified')}, "
                f"mutated={vote.get('post_resume_update_mutated')}, "
                f"roundtrip={vote.get('post_resume_payload_hash_roundtrip_pass')})"
            )

    p1_envelope_sha256 = _sha256_bytes(p1_envelope_bytes)
    base_pass = bool(
        raw_fallback_rejected
        and not dense_saved
        and not dense_loaded
        and cached_install_proven
        and str(source_commit_sha).strip()
        and str(p1_envelope_sha256).strip()
    )
    pass_receipt = bool(base_pass)

    eligible_keys = tuple(sorted(loaded_states))
    receipt = TrainerSub2AuthorityLiveConversionReceipt(
        schema_version=TRAINER_SUB2_LIVE_CONVERSION_SCHEMA_VERSION,
        target_name=TRAINER_SUB2_LIVE_CONVERSION_TARGET_NAME,
        pass_receipt=pass_receipt,
        dry_run=True,
        gpu_launched=False,
        optimizer_step_called=False,
        checkpoint_written=True,
        checkpoint_written_to_banked_parent=False,
        learner_update_called=True,
        live_runtime_authority_converted=pass_receipt,
        readiness_row_flip_authorized=pass_receipt,
        readiness_row_flip_authorized_surface_names=authorized if pass_receipt else (),
        source_commit_sha=str(source_commit_sha),
        proof_command_argv=tuple(str(item) for item in proof_command_argv),
        checkpoint_format=P1_LIVE_CHECKPOINT_FORMAT,
        p1_envelope_sha256=p1_envelope_sha256,
        inner_authoritative_state_payload_sha256=payload_sha_before,
        eligible_state_keys=eligible_keys,
        eligible_state_keys_sha256=_eligible_state_keys_sha256(eligible_keys),
        eligible_module_count=len(eligible_keys),
        load_routing_result=str(load_result.routing),
        dense_int16_persistent_accumulator_saved=bool(dense_saved),
        dense_int16_persistent_accumulator_loaded=bool(dense_loaded),
        raw_state_dict_eligible_weight_fallback_rejected=bool(raw_fallback_rejected),
        cached_weight_parent_install_proven=bool(cached_install_proven),
        parity_max_abs_diff_by_site=parity_map,
        parity_pass=bool(parity_pass),
        vote_carrier_subproof_exercised=True,
        poisoned_fp_master_bypass_falsified=bool(vote["poisoned_fp_master_bypass_falsified"]),
        total_sparse_vote_event_count=int(vote["total_sparse_vote_event_count"]),
        q_changed_count=int(vote["q_changed_count"]),
        post_resume_update_mutated=bool(vote["post_resume_update_mutated"]),
        authority_state_shadow_free_after=bool(vote["authority_state_shadow_free_after"]),
        post_resume_payload_sha256_before=str(vote["post_resume_payload_sha256_before"]),
        post_resume_payload_sha256_after=str(vote["post_resume_payload_sha256_after"]),
        post_resume_payload_hash_roundtrip_pass=bool(
            vote["post_resume_payload_hash_roundtrip_pass"]
        ),
        loss_finite=bool(vote["loss_finite"]),
        q_sidecar_vote_carrier_deferred=bool(deferred),
        q_sidecar_deferred_reason=str(deferred_reason),
        normal_optimizer_resume_from_p1_sidecar_not_proved=True,
        full_training_authority_from_p1_sidecar_not_proved=True,
        learning_claim=False,
        acquisition_claim=False,
        full_sub2_runtime_readiness_claim=False,
        ready_for_main_science=False,
        ready_for_pre_full_stack_diagnostic=False,
        broad_runtime_authority_conversion=False,
        w6_parent_sha256_before=str(w6_parent_sha256_before),
        w6_parent_sha256_after=str(w6_parent_sha256_after),
        proof_anchors=(
            "trainer_sub2_authority.py:767",
            "trainer_sub2_authority.py:809",
            "trainer_sub2_authority.py:1418",
            "train_hrm_text_158.py:1551",
        ),
        non_claims=TRAINER_SUB2_LIVE_CONVERSION_NON_CLAIMS,
    )
    if pass_receipt:
        validate_trainer_sub2_authority_live_conversion_receipt(receipt)
    return receipt



def build_trainer_sub2_authority_live_conversion_receipt(
    *,
    p1_checkpoint: Mapping[str, Any],
    p1_envelope_bytes: bytes,
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    forward_output_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor] | None,
    parity_max_abs_diff_by_site: Mapping[str, float],
    use_ternary_bulk: bool,
    eligible_scope: str = "all-bitlinear",
    device: torch.device | str = "cpu",
    step: int = 0,
    source_commit_sha: str = "",
    proof_command_argv: Sequence[str] = (),
    vote_update_spec: VoteUpdateSpec | None = None,
    poison_value: float = 17.0,
    w6_parent_sha256_before: str = "",
    w6_parent_sha256_after: str = "",
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> TrainerSub2AuthorityLiveConversionReceipt:
    """Public P1b builder: keyword-only signature preserved; core once; returns ONLY p1b."""

    p1b, _subproof, _core_id = run_sparse_vote_authority_b3_core(
        p1_checkpoint=p1_checkpoint,
        p1_envelope_bytes=p1_envelope_bytes,
        fresh_model_fn=fresh_model_fn,
        batch=batch,
        forward_loss_fn=forward_loss_fn,
        forward_output_fn=forward_output_fn,
        parity_max_abs_diff_by_site=parity_max_abs_diff_by_site,
        use_ternary_bulk=use_ternary_bulk,
        eligible_scope=eligible_scope,
        device=device,
        step=step,
        source_commit_sha=source_commit_sha,
        proof_command_argv=proof_command_argv,
        vote_update_spec=vote_update_spec,
        poison_value=poison_value,
        w6_parent_sha256_before=w6_parent_sha256_before,
        w6_parent_sha256_after=w6_parent_sha256_after,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )
    # Public P1b path is not the B3 landing wrapper; close the emission pair
    # started by _run_live_p1_vote_carrier_subproof after full core+receipt work.
    # Landing wrapper path closes emission itself after landing validate instead.
    # Only one of these paths runs per call.
    _emit_phase("PHASE_END", "emission")
    return p1b



def _resolve_live_conversion_source_commit_sha() -> str:
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def validate_trainer_sub2_authority_live_conversion_receipt(
    receipt: TrainerSub2AuthorityLiveConversionReceipt,
    *,
    require_source_at_head: bool = True,
) -> None:
    if receipt.schema_version != TRAINER_SUB2_LIVE_CONVERSION_SCHEMA_VERSION:
        raise ValueError("P1b live conversion schema version mismatch")
    if receipt.target_name != TRAINER_SUB2_LIVE_CONVERSION_TARGET_NAME:
        raise ValueError("P1b live conversion target name mismatch")
    if not receipt.pass_receipt:
        raise ValueError("P1b live conversion proof did not pass")
    if not receipt.dry_run or receipt.gpu_launched or receipt.optimizer_step_called:
        raise ValueError("P1b must stay CPU dry-run with no optimizer step")
    if receipt.checkpoint_written_to_banked_parent:
        raise ValueError("P1b cannot write banked parent checkpoints")
    if not str(receipt.source_commit_sha).strip():
        raise ValueError("P1b missing source_commit_sha")
    if require_source_at_head:
        if receipt.source_commit_sha != _resolve_live_conversion_source_commit_sha():
            raise ValueError("P1b stale source_commit_sha")
    for field_name in (
        "p1_envelope_sha256",
        "inner_authoritative_state_payload_sha256",
        "eligible_state_keys_sha256",
    ):
        if not str(getattr(receipt, field_name)).strip():
            raise ValueError(f"P1b missing hash field {field_name}")
    if receipt.checkpoint_format != P1_LIVE_CHECKPOINT_FORMAT:
        raise ValueError("P1b checkpoint_format mismatch")
    if receipt.load_routing_result != "p1_live":
        raise ValueError("P1b load_routing_result must be p1_live")
    authorized = tuple(receipt.readiness_row_flip_authorized_surface_names)
    if authorized not in (AUTHORIZED_P1B_SURFACE_TUPLE, AUTHORIZED_P1B_SURFACE_TUPLE_2ROW):
        raise ValueError("P1b readiness_row_flip_authorized_surface_names mismatch")
    if not receipt.live_runtime_authority_converted or not receipt.readiness_row_flip_authorized:
        raise ValueError("P1b authorization flags must be true on pass")
    if receipt.dense_int16_persistent_accumulator_saved or receipt.dense_int16_persistent_accumulator_loaded:
        raise ValueError("P1b dense-int16 persistent accumulator flags must be false")
    if not receipt.raw_state_dict_eligible_weight_fallback_rejected:
        raise ValueError("P1b must prove raw eligible-weight fallback rejection")
    if not receipt.cached_weight_parent_install_proven or not receipt.parity_pass:
        raise ValueError("P1b cached parent install / parity proof failed")
    for site in ("cache_builder", "main_kl", "retained_fallback"):
        if site not in receipt.parity_max_abs_diff_by_site:
            raise ValueError(f"P1b missing parity site {site!r}")
        if float(receipt.parity_max_abs_diff_by_site[site]) > P1_LIVE_PARITY_ATOL:
            raise ValueError(f"P1b parity site {site!r} over threshold")
    if authorized == AUTHORIZED_P1B_SURFACE_TUPLE:
        if receipt.q_sidecar_vote_carrier_deferred:
            raise ValueError("P1b 3-row receipt cannot defer q_sidecar")
        if not _vote_subproof_passes_three_row(
            {
                "loss_finite": receipt.loss_finite,
                "poisoned_fp_master_bypass_falsified": receipt.poisoned_fp_master_bypass_falsified,
                "total_sparse_vote_event_count": receipt.total_sparse_vote_event_count,
                "q_changed_count": receipt.q_changed_count,
                "post_resume_update_mutated": receipt.post_resume_update_mutated,
                "authority_state_shadow_free_after": receipt.authority_state_shadow_free_after,
                "post_resume_payload_hash_roundtrip_pass": (
                    receipt.post_resume_payload_hash_roundtrip_pass
                ),
            }
        ):
            raise ValueError("P1b 3-row vote-carrier subproof gates failed")
    else:
        if not receipt.q_sidecar_vote_carrier_deferred:
            raise ValueError("P1b 2-row receipt must defer q_sidecar")
        if not str(receipt.q_sidecar_deferred_reason).strip():
            raise ValueError("P1b 2-row receipt missing q_sidecar_deferred_reason")
    forbidden_true = {
        "learning_claim": receipt.learning_claim,
        "acquisition_claim": receipt.acquisition_claim,
        "full_sub2_runtime_readiness_claim": receipt.full_sub2_runtime_readiness_claim,
        "ready_for_main_science": receipt.ready_for_main_science,
        "ready_for_pre_full_stack_diagnostic": receipt.ready_for_pre_full_stack_diagnostic,
        "broad_runtime_authority_conversion": receipt.broad_runtime_authority_conversion,
    }
    for label, value in forbidden_true.items():
        if bool(value):
            raise ValueError(f"P1b forbidden claim set: {label}")
    if not receipt.normal_optimizer_resume_from_p1_sidecar_not_proved:
        raise ValueError("P1b must carry optimizer-resume non-claim")
    if not receipt.full_training_authority_from_p1_sidecar_not_proved:
        raise ValueError("P1b must carry full-training non-claim")
    if tuple(receipt.non_claims) != TRAINER_SUB2_LIVE_CONVERSION_NON_CLAIMS:
        raise ValueError("P1b non-claims changed")
