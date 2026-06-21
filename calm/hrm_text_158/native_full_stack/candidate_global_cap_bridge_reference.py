"""B2-5c Step-1a candidate+global-cap bridge reference (CPU/read-only, result-based)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _identity_sha256,
    _ordered_identity_sha256,
    _ordered_value_sha256,
    _sparse_value_sha256,
    _tensor_sha256,
    decode_bounded_accumulator_to_i16,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.candidate_global_cap_bridge_receipt import (
    CandidateGlobalCapBridgeFixtureMeasurement,
    MagnitudeRegime,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

STRUCTURAL_REJECT_SOURCE = "bounded_delta_learner.py:1646-1647"


@dataclass(frozen=True)
class BridgeFixtureSpec:
    fixture_name: str
    fixture_role: str
    state_key: str
    numel: int
    acc_overrides: dict[int, int]
    sparse_votes: dict[int, int]
    hot_exact_indices: tuple[int, ...]
    cap: int
    max_abs_per_tensor: int
    q_overrides: dict[int, int] | None = None


@dataclass(frozen=True)
class MaterializedBridgeArtifacts:
    state_key: str
    threshold: int
    applied_indices: tuple[int, ...]
    applied_directions: dict[int, int]
    applied_thresholds: dict[int, int]
    residual_after_threshold: dict[int, int]
    restored_support: torch.Tensor
    candidate_indices: tuple[int, ...]
    proof: dict[str, object]
    prior_q_flat: torch.Tensor
    next_q_levels: torch.Tensor
    next_bounded_summary: dict[str, object]


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


def _decode_support_int32(bounded) -> torch.Tensor:
    return decode_bounded_accumulator_to_i16(bounded).flatten().to(torch.int32)


def _crossing_universe(
    *,
    restored_support: torch.Tensor,
    prior_q: torch.Tensor,
    threshold: int,
) -> tuple[int, ...]:
    universe: list[int] = []
    for index in range(int(restored_support.numel())):
        support = int(restored_support[index].item())
        q_value = int(prior_q[index].item())
        if (support >= threshold and q_value < 1) or (support <= -threshold and q_value > -1):
            universe.append(index)
    return tuple(universe)


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


def verify_materialization_fidelity_lattice(
    artifacts: MaterializedBridgeArtifacts,
    *,
    tamper_index: int | None = None,
    tamper_delta: int = 0,
) -> None:
    proof = artifacts.proof
    state_key = artifacts.state_key
    threshold = artifacts.threshold
    restored = artifacts.restored_support.clone()
    residual_map = dict(artifacts.residual_after_threshold)
    if tamper_index is not None:
        index = int(tamper_index)
        restored[index] = int(restored[index].item()) + int(tamper_delta)
        if index in residual_map:
            residual_map[index] = int(residual_map[index]) + int(tamper_delta)

    expected_identities = str(proof["applied_row_identities_sha256"])
    actual_identities = _identity_sha256(
        {(state_key, index) for index in artifacts.applied_indices},
    )
    if actual_identities != expected_identities:
        raise ValueError("applied_row_identities_sha256 mismatch")

    expected_ordered = str(proof["ordered_applied_row_identities_sha256"])
    actual_ordered = _ordered_identity_sha256(state_key, artifacts.applied_indices)
    if actual_ordered != expected_ordered:
        raise ValueError("ordered_applied_row_identities_sha256 mismatch")

    expected_directions = str(proof["applied_directions_sha256"])
    actual_directions = _ordered_value_sha256(
        state_key,
        "direction",
        artifacts.applied_directions,
    )
    if actual_directions != expected_directions:
        raise ValueError("applied_directions_sha256 mismatch")

    expected_thresholds = str(proof["applied_thresholds_sha256"])
    actual_thresholds = _ordered_value_sha256(
        state_key,
        "threshold",
        artifacts.applied_thresholds,
    )
    if actual_thresholds != expected_thresholds:
        raise ValueError("applied_thresholds_sha256 mismatch")

    expected_residual = str(proof["residual_after_threshold_sha256"])
    actual_residual = _sparse_value_sha256(state_key, residual_map)
    if actual_residual != expected_residual:
        raise ValueError("residual_after_threshold_sha256 mismatch")

    expected_q = str(proof["candidate_q_sha256_after"])
    actual_q = _tensor_sha256(artifacts.next_q_levels)
    if actual_q != expected_q:
        raise ValueError("candidate_q_sha256_after mismatch")

    expected_summary = proof["bounded_accumulator_summary_after"]
    if artifacts.next_bounded_summary != expected_summary:
        raise ValueError("bounded_accumulator_summary_after mismatch")

    if int(proof["candidate_count"]) != len(artifacts.candidate_indices):
        raise ValueError("candidate_count mismatch")

    max_flips = int(proof["pre_veto_selected_flip_count"])
    ordered_candidates = sorted(
        artifacts.candidate_indices,
        key=lambda index: (-abs(int(restored[index].item())), int(index)),
    )
    if tuple(ordered_candidates[:max_flips]) != artifacts.applied_indices:
        raise ValueError("restored-support ordering mismatch vs applied_indices")


def materialize_bridge_artifacts_from_candidate_result(
    *,
    state_key: str,
    prior_state: VoteUpdateState,
    candidate_result,
    vote_spec: VoteUpdateSpec,
) -> MaterializedBridgeArtifacts:
    proof = candidate_result.proof
    if not bool(proof.get("pass")):
        raise ValueError("candidate proof pass required")
    if int(proof.get("event_vote_count", 0)) <= 0:
        raise ValueError("non-zero sparse event count required")
    coverage = proof.get("coverage_domain", {})
    if not isinstance(coverage, dict) or coverage.get("no_global_cap") is not True:
        raise ValueError("candidate coverage_domain.no_global_cap must be True")

    threshold = int(vote_spec.threshold_abs)
    prior_q_flat = prior_state.q_levels.flatten().to(torch.int8)
    next_q_flat = candidate_result.next_q_levels.flatten().to(torch.int8)
    changed = torch.nonzero(prior_q_flat != next_q_flat, as_tuple=False).flatten()
    if changed.numel() <= 0:
        raise ValueError("candidate must apply at least one q delta")
    applied_indices = tuple(sorted(int(index.item()) for index in changed))
    applied_directions: dict[int, int] = {}
    applied_thresholds: dict[int, int] = {}
    for index in applied_indices:
        direction = int(next_q_flat[index].item()) - int(prior_q_flat[index].item())
        if direction not in (-1, 1):
            raise ValueError(f"applied direction must be ±1 at index {index}")
        applied_directions[index] = direction
        applied_thresholds[index] = threshold

    decoded = _decode_support_int32(candidate_result.next_bounded_accumulator)
    residual_after_threshold = {
        index: int(decoded[index].item()) for index in applied_indices
    }
    restored_support = decoded.clone()
    for index in applied_indices:
        restored_support[index] = (
            int(residual_after_threshold[index])
            + int(applied_directions[index]) * threshold
        )

    candidate_indices = _crossing_universe(
        restored_support=restored_support,
        prior_q=prior_q_flat,
        threshold=threshold,
    )
    max_flips = int(proof["pre_veto_selected_flip_count"])
    ordered_candidates = sorted(
        candidate_indices,
        key=lambda index: (-abs(int(restored_support[index].item())), int(index)),
    )
    ordered_applied = tuple(ordered_candidates[:max_flips])
    if set(ordered_applied) != set(applied_indices):
        raise ValueError("q-delta applied set must match restored-support top-k")

    artifacts = MaterializedBridgeArtifacts(
        state_key=state_key,
        threshold=threshold,
        applied_indices=ordered_applied,
        applied_directions=applied_directions,
        applied_thresholds=applied_thresholds,
        residual_after_threshold=residual_after_threshold,
        restored_support=restored_support,
        candidate_indices=candidate_indices,
        proof=proof,
        prior_q_flat=prior_q_flat,
        next_q_levels=candidate_result.next_q_levels,
        next_bounded_summary=candidate_result.next_bounded_accumulator.to_dict(),
    )
    verify_materialization_fidelity_lattice(artifacts)
    return artifacts


def build_vote_update_plan_from_bridge_artifacts(
    *,
    prior_state: VoteUpdateState,
    artifacts: MaterializedBridgeArtifacts,
) -> VoteUpdatePlan:
    numel = int(prior_state.q_levels.numel())
    q_i16 = prior_state.q_levels.flatten().to(torch.int16).view_as(prior_state.q_levels)
    new_acc_i32 = artifacts.restored_support.view_as(prior_state.accumulators).to(torch.int32)
    empty_i64 = torch.zeros(0, dtype=torch.int64)
    empty_i16 = torch.zeros(0, dtype=torch.int16)
    empty_i32 = torch.zeros(0, dtype=torch.int32)
    applied_tensor = torch.tensor(artifacts.applied_indices, dtype=torch.int64)
    directions = torch.tensor(
        [artifacts.applied_directions[index] for index in artifacts.applied_indices],
        dtype=torch.int16,
    )
    thresholds = torch.tensor(
        [artifacts.applied_thresholds[index] for index in artifacts.applied_indices],
        dtype=torch.int32,
    )
    candidate_tensor = torch.tensor(artifacts.candidate_indices, dtype=torch.int64)
    return VoteUpdatePlan(
        q_i16=q_i16,
        new_acc_i32=new_acc_i32,
        candidate_indices=candidate_tensor,
        pre_veto_selected_indices=applied_tensor,
        applied_indices=applied_tensor,
        applied_directions=directions,
        applied_thresholds=thresholds,
        replay_ce_veto_indices=empty_i64,
        replay_veto_directions=empty_i16,
        replay_veto_thresholds=empty_i32,
        pc_aux_negative_indices=empty_i64,
        pc_aux_veto_indices=empty_i64,
        stats={
            "scope": "candidate_global_cap_bridge_materialized",
            "candidate_count": len(artifacts.candidate_indices),
            "applied_count": len(artifacts.applied_indices),
        },
    )


def _row_identities(result, state_key: str) -> tuple[list[int], list[int], str, str]:
    accepted = [
        int(row.flat_index)
        for row in result.accepted_rows
        if row.state_key == state_key
    ]
    deferred = [
        int(row.flat_index)
        for row in result.deferred_rows
        if row.state_key == state_key
    ]
    accepted_sha = _identity_sha256({(state_key, index) for index in accepted})
    deferred_sha = _identity_sha256({(state_key, index) for index in deferred})
    return accepted, deferred, accepted_sha, deferred_sha


def reference_exact_local_then_global_cap(
    *,
    state: VoteUpdateState,
    sparse_votes: Mapping[int, int],
    vote_spec: VoteUpdateSpec,
    cap_spec: GlobalRateCapSpec,
    state_key: str,
) -> tuple[VoteUpdatePlan, object]:
    oracle_plan = plan_integer_vote_update_reference(
        state,
        _dense_votes(state.q_levels.numel(), sparse_votes),
        vote_spec,
    )
    cap_input = GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=oracle_plan,
    )
    result = apply_global_rate_cap_reference([cap_input], cap_spec)
    return oracle_plan, result


def compose_candidate_global_cap_bridge_reference(
    *,
    state: VoteUpdateState,
    sparse_votes: Mapping[int, int],
    vote_spec: VoteUpdateSpec,
    cap_spec: GlobalRateCapSpec,
    state_key: str,
    bounded,
    candidate_result,
) -> tuple[VoteUpdatePlan, object, MaterializedBridgeArtifacts]:
    artifacts = materialize_bridge_artifacts_from_candidate_result(
        state_key=state_key,
        prior_state=state,
        candidate_result=candidate_result,
        vote_spec=vote_spec,
    )
    bridge_plan = build_vote_update_plan_from_bridge_artifacts(
        prior_state=state,
        artifacts=artifacts,
    )
    cap_input = GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=bridge_plan,
    )
    q_sha_before = _tensor_sha256(state.q_levels)
    acc_sha_before = _tensor_sha256(state.accumulators)
    bridge_result = apply_global_rate_cap_reference([cap_input], cap_spec)
    q_sha_after = _tensor_sha256(state.q_levels)
    acc_sha_after = _tensor_sha256(state.accumulators)
    if q_sha_before != q_sha_after or acc_sha_before != acc_sha_after:
        raise ValueError("global cap reference mutated live state")
    return bridge_plan, bridge_result, artifacts


def compare_bridge_vs_oracle_post_cap(
    *,
    state_key: str,
    bridge_result,
    oracle_result,
) -> tuple[bool, bool, bool, bool]:
    bridge_accepted, bridge_deferred, _, _ = _row_identities(bridge_result, state_key)
    oracle_accepted, oracle_deferred, _, _ = _row_identities(oracle_result, state_key)
    accepted_match = bridge_accepted == oracle_accepted
    deferred_match = bridge_deferred == oracle_deferred
    order_match = bridge_accepted == oracle_accepted
    counts_match = (
        len(bridge_accepted) == len(oracle_accepted)
        and len(bridge_deferred) == len(oracle_deferred)
    )
    equivalent = accepted_match and deferred_match and order_match and counts_match
    return equivalent, accepted_match, deferred_match, order_match and counts_match


def assert_structural_candidate_global_cap_reject() -> None:
    state = make_bounded_tensor_state(
        "F_STRUCTURAL_GUARD",
        torch.tensor([0], dtype=torch.int8),
        0.5,
        torch.zeros(1, dtype=torch.int16),
    )
    votes = torch.tensor([12], dtype=torch.int16)
    cap_spec = GlobalRateCapSpec(cap=1, step=0, ordering_seed=0, mutate_outputs=False)
    vote_spec = _vote_spec(max_abs_per_tensor=1)
    try:
        apply_bounded_delta_vote_step(
            {"F_STRUCTURAL_GUARD": state},
            {"F_STRUCTURAL_GUARD": votes},
            {"F_STRUCTURAL_GUARD": vote_spec},
            candidate_mode=ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            candidate_sparse_vote_events_by_key={"F_STRUCTURAL_GUARD": {0: 12}},
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


def run_bridge_fixture(spec: BridgeFixtureSpec) -> CandidateGlobalCapBridgeFixtureMeasurement:
    if not spec.sparse_votes:
        return CandidateGlobalCapBridgeFixtureMeasurement(
            fixture_name=spec.fixture_name,
            fixture_role=spec.fixture_role,  # type: ignore[arg-type]
            total_sparse_event_count=0,
            magnitude_regime="clip_boundary_reconciliation",
            add_back_clip_boundary_reconciliation=False,
            fidelity_lattice_pass=False,
            bridge_equivalent=False,
            accepted_identities_match=False,
            deferred_identities_match=False,
            accepted_order_match=False,
            cap_counts_match=False,
            step1a_novel_claim_materialization_fidelity=False,
            step1a_novel_claim_cap_api_composability=False,
            step1a_novel_claim_saturated_margin_ordering_identity=False,
            candidate_applied_row_identities_sha256=_identity_sha256(set()),
            bridge_accepted_identities_sha256=_identity_sha256(set()),
            oracle_accepted_identities_sha256=_identity_sha256(set()),
        )

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
    candidate_result = execute_direct_bounded_local_vote_update_candidate(
        state_key=spec.state_key,
        q_levels=state.q_levels,
        bounded_accumulator=bounded,
        sparse_vote_events=spec.sparse_votes,
        vote_spec=vote_spec,
    )
    cap_spec = GlobalRateCapSpec(
        cap=int(spec.cap),
        step=1,
        ordering_seed=0,
        mutate_outputs=False,
    )
    try:
        artifacts = materialize_bridge_artifacts_from_candidate_result(
            state_key=spec.state_key,
            prior_state=state,
            candidate_result=candidate_result,
            vote_spec=vote_spec,
        )
        fidelity_pass = True
    except ValueError:
        proof = candidate_result.proof
        return CandidateGlobalCapBridgeFixtureMeasurement(
            fixture_name=spec.fixture_name,
            fixture_role=spec.fixture_role,  # type: ignore[arg-type]
            total_sparse_event_count=len(spec.sparse_votes),
            magnitude_regime="clip_boundary_reconciliation",
            add_back_clip_boundary_reconciliation=True,
            fidelity_lattice_pass=False,
            bridge_equivalent=False,
            accepted_identities_match=False,
            deferred_identities_match=False,
            accepted_order_match=False,
            cap_counts_match=False,
            step1a_novel_claim_materialization_fidelity=False,
            step1a_novel_claim_cap_api_composability=False,
            step1a_novel_claim_saturated_margin_ordering_identity=False,
            candidate_applied_row_identities_sha256=str(
                proof.get("applied_row_identities_sha256", _identity_sha256(set())),
            ),
            bridge_accepted_identities_sha256=_identity_sha256(set()),
            oracle_accepted_identities_sha256=_identity_sha256(set()),
        )

    magnitude_regime = _magnitude_regime_for_applied(
        residual_after_threshold=artifacts.residual_after_threshold,
        threshold=artifacts.threshold,
    )
    clip_reconciliation = magnitude_regime == "clip_boundary_reconciliation"

    _, bridge_result, _ = compose_candidate_global_cap_bridge_reference(
        state=state,
        sparse_votes=spec.sparse_votes,
        vote_spec=vote_spec,
        cap_spec=cap_spec,
        state_key=spec.state_key,
        bounded=bounded,
        candidate_result=candidate_result,
    )
    _, oracle_result = reference_exact_local_then_global_cap(
        state=state,
        sparse_votes=spec.sparse_votes,
        vote_spec=vote_spec,
        cap_spec=cap_spec,
        state_key=spec.state_key,
    )
    bridge_accepted, _, bridge_acc_sha, bridge_def_sha = _row_identities(
        bridge_result,
        spec.state_key,
    )
    oracle_accepted, _, oracle_acc_sha, oracle_def_sha = _row_identities(
        oracle_result,
        spec.state_key,
    )
    equivalent, acc_match, def_match, order_counts_match = compare_bridge_vs_oracle_post_cap(
        state_key=spec.state_key,
        bridge_result=bridge_result,
        oracle_result=oracle_result,
    )
    if clip_reconciliation:
        equivalent = False
    saturated = int(spec.cap) < len(artifacts.candidate_indices)
    return CandidateGlobalCapBridgeFixtureMeasurement(
        fixture_name=spec.fixture_name,
        fixture_role=spec.fixture_role,  # type: ignore[arg-type]
        total_sparse_event_count=len(spec.sparse_votes),
        magnitude_regime=magnitude_regime,
        add_back_clip_boundary_reconciliation=clip_reconciliation,
        fidelity_lattice_pass=fidelity_pass,
        bridge_equivalent=equivalent and not clip_reconciliation,
        accepted_identities_match=acc_match,
        deferred_identities_match=def_match,
        accepted_order_match=bridge_accepted == oracle_accepted,
        cap_counts_match=order_counts_match,
        step1a_novel_claim_materialization_fidelity=fidelity_pass,
        step1a_novel_claim_cap_api_composability=fidelity_pass and bridge_result is not None,
        step1a_novel_claim_saturated_margin_ordering_identity=(
            fidelity_pass and saturated and equivalent and not clip_reconciliation
        ),
        candidate_applied_row_identities_sha256=str(
            candidate_result.proof["applied_row_identities_sha256"],
        ),
        bridge_accepted_identities_sha256=bridge_acc_sha,
        oracle_accepted_identities_sha256=oracle_acc_sha,
    )


def build_representative_bridge_measurements() -> tuple[
    CandidateGlobalCapBridgeFixtureMeasurement,
    ...,
]:
    assert_structural_candidate_global_cap_reject()
    return (
        run_bridge_fixture(
            BridgeFixtureSpec(
                fixture_name="F_BRIDGE_MINIMAL",
                fixture_role="representative_consumer",
                state_key="F_BRIDGE_MINIMAL",
                numel=4,
                acc_overrides={0: 9, 2: -9},
                sparse_votes={0: 2, 2: -2},
                hot_exact_indices=(0, 2),
                cap=10,
                max_abs_per_tensor=4,
            ),
        ),
        run_bridge_fixture(
            BridgeFixtureSpec(
                fixture_name="F_BRIDGE_SATURATED",
                fixture_role="representative_consumer",
                state_key="F_BRIDGE_SATURATED",
                numel=8,
                acc_overrides={0: 9, 1: -9, 2: 9, 3: -9},
                sparse_votes={0: 2, 1: -2, 2: 2, 3: -2},
                hot_exact_indices=(0, 1, 2, 3),
                cap=2,
                max_abs_per_tensor=8,
            ),
        ),
        CandidateGlobalCapBridgeFixtureMeasurement(
            fixture_name="F_STRUCTURAL_GUARD",
            fixture_role="representative_consumer",
            total_sparse_event_count=0,
            magnitude_regime="no_clip_exact_add_back",
            add_back_clip_boundary_reconciliation=False,
            fidelity_lattice_pass=False,
            bridge_equivalent=False,
            accepted_identities_match=False,
            deferred_identities_match=False,
            accepted_order_match=False,
            cap_counts_match=False,
            step1a_novel_claim_materialization_fidelity=False,
            step1a_novel_claim_cap_api_composability=False,
            step1a_novel_claim_saturated_margin_ordering_identity=False,
            candidate_applied_row_identities_sha256=_identity_sha256(set()),
            bridge_accepted_identities_sha256=_identity_sha256(set()),
            oracle_accepted_identities_sha256=_identity_sha256(set()),
            structural_candidate_global_cap_reject=True,
        ),
    )


def build_classifier_negative_bridge_measurements() -> tuple[
    CandidateGlobalCapBridgeFixtureMeasurement,
    ...,
]:
    zero_sparse = run_bridge_fixture(
        BridgeFixtureSpec(
            fixture_name="F_NEG_ZERO_SPARSE",
            fixture_role="classifier_negative",
            state_key="F_NEG_ZERO_SPARSE",
            numel=4,
            acc_overrides={0: 9},
            sparse_votes={},
            hot_exact_indices=(0,),
            cap=1,
            max_abs_per_tensor=4,
        ),
    )
    clip_boundary = run_bridge_fixture(
        BridgeFixtureSpec(
            fixture_name="F_CLIP_BOUNDARY",
            fixture_role="classifier_negative",
            state_key="F_CLIP_BOUNDARY",
            numel=4,
            acc_overrides={0: 18, 2: -18},
            sparse_votes={0: 2, 2: -2},
            hot_exact_indices=(0, 2),
            cap=10,
            max_abs_per_tensor=4,
        ),
    )
    return (zero_sparse, clip_boundary)
