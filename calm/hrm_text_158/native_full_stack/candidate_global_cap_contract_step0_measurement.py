"""Read-only paired execution for B2-5c Step-0 candidate↔global-cap contract measurement."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _identity_sha256,
    _ordered_identity_sha256,
    _ordered_value_sha256,
    _sparse_value_sha256,
    _tensor_sha256,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_contract_step0_receipt import (
    PINNED_SURFACE_CANDIDATE,
    PINNED_SURFACE_EXACT_LOCAL,
    PINNED_SURFACE_GCAP_SHADOW,
    PINNED_SURFACES_FULL_EXECUTION,
    CandidateGlobalCapContractFixtureMeasurement,
    FixtureRole,
    FixtureTier,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

STRUCTURAL_REJECT_SOURCE = "bounded_delta_learner.py:1646-1647"


@dataclass(frozen=True)
class _PairedFixtureSpec:
    fixture_name: str
    fixture_role: FixtureRole
    fixture_tier: FixtureTier
    state_key: str
    numel: int
    acc_overrides: dict[int, int]
    sparse_votes: dict[int, int]
    hot_exact_indices: tuple[int, ...]
    cap: int
    max_abs_per_tensor: int
    q_overrides: dict[int, int] | None = None


def _vote_spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=128,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _build_state(
    numel: int,
    *,
    acc_overrides: Mapping[int, int] | None = None,
    q_overrides: Mapping[int, int] | None = None,
) -> VoteUpdateState:
    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index, value in (q_overrides or {}).items():
        q[int(index)] = int(value)
    for index, value in (acc_overrides or {}).items():
        acc[int(index)] = int(value)
    return VoteUpdateState(q_levels=q, accumulators=acc)


def _dense_votes(numel: int, sparse_votes: Mapping[int, int]) -> VoteUpdateInputs:
    votes = torch.zeros(numel, dtype=torch.int16)
    for index, value in sparse_votes.items():
        votes[int(index)] = int(value)
    return VoteUpdateInputs(votes=votes)


def _empty_identity_sha256() -> str:
    return _identity_sha256(set())


def _is_subsequence(sub: Sequence[int], full: Sequence[int]) -> bool:
    cursor = 0
    for item in sub:
        while cursor < len(full) and full[cursor] != item:
            cursor += 1
        if cursor >= len(full):
            return False
        cursor += 1
    return True


def _exact_local_surface_hashes(
    state_key: str,
    plan,
) -> tuple[str, str, tuple[int, ...], dict[int, int], dict[int, int]]:
    applied = tuple(int(index) for index in plan.applied_indices.detach().cpu().tolist())
    directions: dict[int, int] = {}
    residuals: dict[int, int] = {}
    flat_acc = plan.new_acc_i32.flatten()
    for position, index in enumerate(applied):
        direction = int(plan.applied_directions[position].item())
        threshold = int(plan.applied_thresholds[position].item())
        directions[index] = direction
        raw = int(flat_acc[index].item())
        residual = raw - direction * threshold
        residual = max(-threshold + 1, min(threshold - 1, residual))
        residuals[index] = residual
    identities_sha = _identity_sha256({(state_key, index) for index in applied})
    residual_sha = _sparse_value_sha256(state_key, residuals)
    return identities_sha, residual_sha, applied, directions, residuals


def _deferred_backlog_authority_defined(
    deferred_count: int,
    deferred_rows,
    backlog: dict[str, dict[int, dict[str, int]]],
) -> bool:
    if deferred_count <= 0:
        return True
    for row in deferred_rows:
        state_backlog = backlog.get(row.state_key)
        if state_backlog is None or row.flat_index not in state_backlog:
            return False
    return True


def measure_paired_fixture(spec: _PairedFixtureSpec) -> CandidateGlobalCapContractFixtureMeasurement:
    state = _build_state(
        spec.numel,
        acc_overrides=spec.acc_overrides,
        q_overrides=spec.q_overrides,
    )
    vote_spec = _vote_spec(max_abs_per_tensor=spec.max_abs_per_tensor)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=spec.hot_exact_indices,
        cold_default_value=0,
    )
    sparse_event_count = len(spec.sparse_votes)

    candidate_result = execute_direct_bounded_local_vote_update_candidate(
        state_key=spec.state_key,
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events=spec.sparse_votes,
        vote_spec=vote_spec,
    )
    candidate_proof = candidate_result.proof
    candidate_identities_sha = str(candidate_proof["applied_row_identities_sha256"])
    candidate_residual_sha = str(candidate_proof["residual_after_threshold_sha256"])
    candidate_directions_sha = str(candidate_proof["applied_directions_sha256"])

    exact_plan = plan_integer_vote_update_reference(
        state,
        _dense_votes(spec.numel, spec.sparse_votes),
        vote_spec,
    )
    (
        exact_identities_sha,
        exact_residual_sha,
        local_order,
        exact_directions,
        _exact_residuals,
    ) = _exact_local_surface_hashes(spec.state_key, exact_plan)
    exact_directions_sha = _ordered_value_sha256(
        spec.state_key,
        "direction",
        exact_directions,
    )

    cap_spec = GlobalRateCapSpec(
        cap=int(spec.cap),
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    cap_input = GlobalRateCapTensorInput(
        state_key=spec.state_key,
        state=state,
        plan=exact_plan,
    )
    q_sha_before = _tensor_sha256(state.q_levels)
    acc_sha_before = _tensor_sha256(state.accumulators)
    plan_q_sha_before = _tensor_sha256(exact_plan.q_i16)
    plan_acc_sha_before = _tensor_sha256(exact_plan.new_acc_i32)
    shadow_result = apply_global_rate_cap_reference([cap_input], cap_spec)
    q_sha_after = _tensor_sha256(state.q_levels)
    acc_sha_after = _tensor_sha256(state.accumulators)
    plan_q_sha_after = _tensor_sha256(exact_plan.q_i16)
    plan_acc_sha_after = _tensor_sha256(exact_plan.new_acc_i32)
    shadow_mutation_observed = any(
        before != after
        for before, after in (
            (q_sha_before, q_sha_after),
            (acc_sha_before, acc_sha_after),
            (plan_q_sha_before, plan_q_sha_after),
            (plan_acc_sha_before, plan_acc_sha_after),
        )
    )

    accepted_rows = [
        row for row in shadow_result.accepted_rows if row.state_key == spec.state_key
    ]
    deferred_rows = [
        row for row in shadow_result.deferred_rows if row.state_key == spec.state_key
    ]
    accepted_order = [int(row.flat_index) for row in accepted_rows]
    local_identities = {(spec.state_key, index) for index in local_order}
    accepted_identities = {(spec.state_key, int(row.flat_index)) for row in accepted_rows}

    identity_set_match = candidate_identities_sha == exact_identities_sha
    direction_match = candidate_directions_sha == exact_directions_sha
    residual_hash_match = candidate_residual_sha == exact_residual_sha
    ordering_match = _is_subsequence(accepted_order, list(local_order))
    global_cap_pure_subset = accepted_identities.issubset(local_identities)
    deferred_count = len(deferred_rows)
    saturation_exercised = deferred_count > 0 or int(spec.cap) < len(local_order)

    return CandidateGlobalCapContractFixtureMeasurement(
        fixture_name=spec.fixture_name,
        fixture_role=spec.fixture_role,
        fixture_tier=spec.fixture_tier,
        pinned_surfaces=PINNED_SURFACES_FULL_EXECUTION,
        total_sparse_event_count=sparse_event_count,
        candidate_applied_row_identities_sha256=candidate_identities_sha,
        candidate_residual_after_threshold_sha256=candidate_residual_sha,
        candidate_q_changed_count=int(candidate_proof["q_changed_count"]),
        candidate_local_update_pass=bool(candidate_proof["pass"]),
        candidate_global_rate_cap_enabled=bool(
            candidate_proof.get("coverage_domain", {}).get("no_global_cap", True) is False
        ),
        exact_local_applied_row_identities_sha256=exact_identities_sha,
        exact_local_residual_after_threshold_sha256=exact_residual_sha,
        exact_local_pre_cap_demand_count=len(local_order),
        shadow_pre_cap_demand_sha256=str(
            shadow_result.step_summary["pre_cap_demand_sha256"],
        ),
        shadow_accepted_identities_sha256=_identity_sha256(accepted_identities),
        shadow_deferred_identities_sha256=_identity_sha256(
            {(spec.state_key, int(row.flat_index)) for row in deferred_rows},
        ),
        shadow_mutation_observed=shadow_mutation_observed,
        cap=int(spec.cap),
        accepted_count=len(accepted_rows),
        deferred_count=deferred_count,
        identity_set_match=identity_set_match,
        direction_match=direction_match,
        residual_hash_match=residual_hash_match,
        ordering_match=ordering_match,
        global_cap_pure_subset_of_local_universe=global_cap_pure_subset,
        deferred_backlog_authority_defined=_deferred_backlog_authority_defined(
            deferred_count,
            deferred_rows,
            shadow_result.deferred_backlog,
        ),
        saturation_exercised=saturation_exercised,
    )


def measure_structural_candidate_global_cap_reject() -> CandidateGlobalCapContractFixtureMeasurement:
    return CandidateGlobalCapContractFixtureMeasurement(
        fixture_name="F_STRUCTURAL_REJECT",
        fixture_role="representative_consumer",
        fixture_tier="structural",
        pinned_surfaces=(),
        total_sparse_event_count=0,
        candidate_applied_row_identities_sha256=_empty_identity_sha256(),
        candidate_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_STRUCTURAL_REJECT",
            {},
        ),
        candidate_q_changed_count=0,
        candidate_local_update_pass=False,
        candidate_global_rate_cap_enabled=False,
        exact_local_applied_row_identities_sha256=_empty_identity_sha256(),
        exact_local_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_STRUCTURAL_REJECT",
            {},
        ),
        exact_local_pre_cap_demand_count=0,
        shadow_pre_cap_demand_sha256=hashlib.sha256(b"").hexdigest(),
        shadow_accepted_identities_sha256=_empty_identity_sha256(),
        shadow_deferred_identities_sha256=_empty_identity_sha256(),
        shadow_mutation_observed=False,
        cap=0,
        accepted_count=0,
        deferred_count=0,
        identity_set_match=True,
        direction_match=True,
        residual_hash_match=True,
        ordering_match=True,
        global_cap_pure_subset_of_local_universe=True,
        deferred_backlog_authority_defined=True,
        saturation_exercised=False,
        structural_candidate_global_cap_reject=True,
    )


def assert_structural_candidate_global_cap_reject() -> None:
    state = make_bounded_tensor_state(
        "F_STRUCTURAL_REJECT",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([12], dtype=torch.int16)
    cap_spec = GlobalRateCapSpec(cap=1, step=0, ordering_seed=0, mutate_outputs=False)
    vote_spec = _vote_spec(max_abs_per_tensor=1)
    try:
        apply_bounded_delta_vote_step(
            {"F_STRUCTURAL_REJECT": state},
            {"F_STRUCTURAL_REJECT": votes},
            {"F_STRUCTURAL_REJECT": vote_spec},
            candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            candidate_sparse_vote_events_by_key={"F_STRUCTURAL_REJECT": {0: 12}},
            candidate_oracle_control_enabled=False,
            global_cap_spec=cap_spec,
        )
    except ValueError as exc:
        if "global cap" not in str(exc):
            raise AssertionError(
                f"expected structural global-cap rejection, got {exc!r}",
            ) from exc
        return
    raise AssertionError("expected structural global-cap rejection")


def build_representative_consumer_measurements() -> tuple[
    CandidateGlobalCapContractFixtureMeasurement,
    ...,
]:
    assert_structural_candidate_global_cap_reject()
    return (
        measure_paired_fixture(
            _PairedFixtureSpec(
                fixture_name="F_PAIR_MINIMAL_SPARSE",
                fixture_role="representative_consumer",
                fixture_tier="minimal",
                state_key="F_PAIR_MINIMAL_SPARSE",
                numel=4,
                acc_overrides={0: 9, 2: -9},
                sparse_votes={0: 2, 2: -2},
                hot_exact_indices=(0, 2),
                cap=10,
                max_abs_per_tensor=4,
            ),
        ),
        measure_paired_fixture(
            _PairedFixtureSpec(
                fixture_name="F_PAIR_SATURATED_CAP",
                fixture_role="representative_consumer",
                fixture_tier="saturated",
                state_key="F_PAIR_SATURATED_CAP",
                numel=8,
                acc_overrides={0: 9, 1: -9, 2: 9, 3: -9},
                sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
                hot_exact_indices=(0, 1, 2, 3),
                cap=2,
                max_abs_per_tensor=8,
            ),
        ),
        measure_structural_candidate_global_cap_reject(),
    )


def build_classifier_negative_measurements() -> tuple[
    CandidateGlobalCapContractFixtureMeasurement,
    ...,
]:
    zero_sparse = CandidateGlobalCapContractFixtureMeasurement(
        fixture_name="F_NEG_ZERO_SPARSE",
        fixture_role="classifier_negative",
        fixture_tier="minimal",
        pinned_surfaces=(PINNED_SURFACE_CANDIDATE,),
        total_sparse_event_count=0,
        candidate_applied_row_identities_sha256=_empty_identity_sha256(),
        candidate_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_ZERO_SPARSE",
            {},
        ),
        candidate_q_changed_count=0,
        candidate_local_update_pass=False,
        candidate_global_rate_cap_enabled=False,
        exact_local_applied_row_identities_sha256=_empty_identity_sha256(),
        exact_local_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_ZERO_SPARSE",
            {},
        ),
        exact_local_pre_cap_demand_count=0,
        shadow_pre_cap_demand_sha256=hashlib.sha256(b"").hexdigest(),
        shadow_accepted_identities_sha256=_empty_identity_sha256(),
        shadow_deferred_identities_sha256=_empty_identity_sha256(),
        shadow_mutation_observed=False,
        cap=0,
        accepted_count=0,
        deferred_count=0,
        identity_set_match=True,
        direction_match=True,
        residual_hash_match=True,
        ordering_match=True,
        global_cap_pure_subset_of_local_universe=True,
        deferred_backlog_authority_defined=False,
        saturation_exercised=False,
    )
    no_saturation = CandidateGlobalCapContractFixtureMeasurement(
        fixture_name="F_NEG_NO_SATURATION",
        fixture_role="classifier_negative",
        fixture_tier="minimal",
        pinned_surfaces=PINNED_SURFACES_FULL_EXECUTION,
        total_sparse_event_count=2,
        candidate_applied_row_identities_sha256=_ordered_identity_sha256(
            "F_NEG_NO_SATURATION",
            (0, 2),
        ),
        candidate_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_NO_SATURATION",
            {0: 1, 2: -1},
        ),
        candidate_q_changed_count=2,
        candidate_local_update_pass=True,
        candidate_global_rate_cap_enabled=False,
        exact_local_applied_row_identities_sha256=_ordered_identity_sha256(
            "F_NEG_NO_SATURATION",
            (0, 2),
        ),
        exact_local_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_NO_SATURATION",
            {0: 1, 2: -1},
        ),
        exact_local_pre_cap_demand_count=2,
        shadow_pre_cap_demand_sha256=hashlib.sha256(b"demand").hexdigest(),
        shadow_accepted_identities_sha256=_ordered_identity_sha256(
            "F_NEG_NO_SATURATION",
            (0, 2),
        ),
        shadow_deferred_identities_sha256=_empty_identity_sha256(),
        shadow_mutation_observed=False,
        cap=10,
        accepted_count=2,
        deferred_count=0,
        identity_set_match=True,
        direction_match=True,
        residual_hash_match=True,
        ordering_match=True,
        global_cap_pure_subset_of_local_universe=True,
        deferred_backlog_authority_defined=True,
        saturation_exercised=False,
    )
    forced_identity_diverge = CandidateGlobalCapContractFixtureMeasurement(
        fixture_name="F_NEG_FORCED_IDENTITY_DIVERGE",
        fixture_role="classifier_negative",
        fixture_tier="saturated",
        pinned_surfaces=PINNED_SURFACES_FULL_EXECUTION,
        total_sparse_event_count=4,
        candidate_applied_row_identities_sha256=_ordered_identity_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            (0, 1),
        ),
        candidate_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            {0: 1, 1: -1},
        ),
        candidate_q_changed_count=2,
        candidate_local_update_pass=True,
        candidate_global_rate_cap_enabled=False,
        exact_local_applied_row_identities_sha256=_ordered_identity_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            (0, 1, 2, 3),
        ),
        exact_local_residual_after_threshold_sha256=_sparse_value_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            {0: 1, 1: -1, 2: 1, 3: -1},
        ),
        exact_local_pre_cap_demand_count=4,
        shadow_pre_cap_demand_sha256=hashlib.sha256(b"demand4").hexdigest(),
        shadow_accepted_identities_sha256=_ordered_identity_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            (0, 1),
        ),
        shadow_deferred_identities_sha256=_ordered_identity_sha256(
            "F_NEG_FORCED_IDENTITY_DIVERGE",
            (2, 3),
        ),
        shadow_mutation_observed=False,
        cap=2,
        accepted_count=2,
        deferred_count=2,
        identity_set_match=False,
        direction_match=False,
        residual_hash_match=False,
        ordering_match=True,
        global_cap_pure_subset_of_local_universe=True,
        deferred_backlog_authority_defined=True,
        saturation_exercised=True,
    )
    return (zero_sparse, no_saturation, forced_identity_diverge)
