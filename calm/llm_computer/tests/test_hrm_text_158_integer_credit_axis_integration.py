"""CPU tests for BR-3C-G integer credit-axis integration (frozen PLAN v2)."""
from __future__ import annotations

from dataclasses import replace

import hashlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
    ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
    AUDIT_NO_DENSE_INT_ACCUM,
    BRANCH_D_DENSE_LEAK,
    BRANCH_D_HIDDEN_FP_STRUCTURAL,
    BRANCH_D_INTEGER_VIABLE,
    BRANCH_D_PARTIAL_COVERAGE,
    BRANCH_D_RANKING_GAP,
    BRANCH_D_REPRESENTATION_LIMIT,
    BRANCH_D_WIRE_ONLY_REFERENCE_GAP,
    FORBIDDEN_INTEGRATION_RECEIPT_FIELDS,
    INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY,
    INTEGRATION_HASH_BYTE_ORDER,
    INTEGER_CREDIT_AXIS_INTEGRATION_NON_CLAIMS,
    _attribution_selected_for_moves,
    _cross_bind_ranking_tensors_from_bound_events,
    _numpy_little_endian_array,
    _recompute_integration_branch_id,
    build_integer_credit_axis_integration_receipt,
    build_streaming_sparse_attribution_subcontract_receipt,
    canonical_tensor_payload_sha256,
    candidate_dense_integer_dispatch_observation,
    classify_integer_native_optimizer_credit_path_branch,
    events_bit_identical,
    integer_credit_axis_integration_hard_false_snapshot,
    prove_integer_credit_axis_integration,
    prove_streaming_sparse_attribution_subcontract,
    prove_strict_integer_ranking_subcontract,
    streaming_sparse_attribution_from_captures,
    validate_integer_credit_axis_integration_receipt,
    validate_streaming_sparse_attribution_subcontract_receipt,
    validate_ranking_subcontract_receipt,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    BR_F_RANKING_INTEGER_EXACT,
    BR_F_RANKING_PRECISION_DIVERGENCE,
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
)


class _Tiny(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        self.proj = BitLinear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _dry_run_capture_fixture(
    *,
    in_features: int = 3,
    out_features: int = 2,
    seed: int = 158,
) -> tuple[dict, tuple[int, int], torch.Tensor]:
    torch.manual_seed(seed)
    model = _Tiny(in_features=in_features, out_features=out_features)
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)[:, :in_features]
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)[:, :out_features]
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    capture = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    q_levels_flat = state.q_levels.reshape(-1)
    return capture, weight_shape, q_levels_flat


def _green_integration_receipt():
    capture, weight_shape, q_levels_flat = _dry_run_capture_fixture()
    return prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="integration-green-v1",
        reference_oracle_run_id="ref-integration-green",
        candidate_run_id="candidate-integration-green",
        law_id=INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )


def test_same_candidate_events_object_binding() -> None:
    capture, weight_shape, q_levels_flat = _dry_run_capture_fixture()
    with candidate_dense_integer_dispatch_observation(weight_shape) as observer:
        candidate_events, sparse_metrics = streaming_sparse_attribution_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            weight_shape=weight_shape,
        )
    dispatch_obs = observer.observation()
    attribution_snapshot = build_streaming_sparse_attribution_subcontract_receipt(
        metrics=sparse_metrics,
        dispatch_observation=dispatch_obs,
        full_support_parity_pass=True,
        comparable_set_id="same-object-v1",
        reference_oracle_run_id="ref-same-object",
        candidate_run_id="candidate-same-object",
    )
    receipt = prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="same-object-v1",
        reference_oracle_run_id="ref-same-object",
        candidate_run_id="candidate-same-object",
    )
    bound = receipt.bound_candidate_attribution_events
    assert events_bit_identical(
        candidate_events,
        bound.as_integer_marginal_attribution_events(),
    )
    re_move_indices, re_moves, re_credit = _cross_bind_ranking_tensors_from_bound_events(
        bound,
        receipt.bound_q_levels_flat,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert torch.equal(re_move_indices, receipt.bound_projected_move_indices)
    assert torch.equal(re_moves, receipt.bound_projected_moves)
    assert torch.equal(re_credit, receipt.bound_credit_q31)
    validate_streaming_sparse_attribution_subcontract_receipt(attribution_snapshot)
    validate_integer_credit_axis_integration_receipt(receipt)


def test_embedded_attribution_snapshot_tamper_rejected() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(
        receipt,
        attribution_subcontract_snapshot=replace(
            receipt.attribution_subcontract_snapshot,
            candidate_dense_integer_scratch_observed=True,
            candidate_dense_integer_scratch_surfaces=(AUDIT_NO_DENSE_INT_ACCUM,),
        ),
    )
    with pytest.raises(ValueError):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_embedded_ranking_snapshot_tamper_rejected() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(
        receipt,
        ranking_subcontract_snapshot=replace(
            receipt.ranking_subcontract_snapshot,
            integer_vs_float_rank_mismatch_count=1,
            branch_id=BR_F_RANKING_PRECISION_DIVERGENCE,
            drop_in_float32_parity_pass=False,
            strict_integer_self_consistency_pass=False,
            precision_divergence_count=1,
        ),
    )
    with pytest.raises(ValueError):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_hash_byte_order_little_endian_normalized() -> None:
    tensor = torch.tensor([1, -2, 3], dtype=torch.int32)
    le_hash = canonical_tensor_payload_sha256(tensor)
    arr = tensor.detach().cpu().contiguous().numpy()
    be_arr = arr.astype(arr.dtype.newbyteorder(">"))
    le_from_be = _numpy_little_endian_array(torch.tensor(be_arr.astype(be_arr.dtype.newbyteorder("<")).copy()))
    meta = (
        f"{str(tensor.dtype)}|{tuple(int(x) for x in tensor.shape)}|{INTEGRATION_HASH_BYTE_ORDER}|"
    ).encode("utf-8")
    manual_hash = hashlib.sha256(meta + le_from_be.tobytes()).hexdigest()
    assert le_hash == manual_hash
    assert INTEGRATION_HASH_BYTE_ORDER == "little_endian"


def test_forged_composition_different_inputs_rejected() -> None:
    capture_a, weight_shape, q_a = _dry_run_capture_fixture(seed=158)
    capture_b, _, q_b = _dry_run_capture_fixture(seed=999)
    receipt_a = prove_integer_credit_axis_integration(
        capture_a["inputs"],
        capture_a["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_a,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="forge-a",
        reference_oracle_run_id="ref-forge-a",
        candidate_run_id="candidate-forge",
    )
    receipt_b = prove_integer_credit_axis_integration(
        capture_b["inputs"],
        capture_b["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_b,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="forge-b",
        reference_oracle_run_id="ref-forge-b",
        candidate_run_id="candidate-forge",
    )
    forged = replace(
        receipt_a,
        ranking_subcontract_snapshot=receipt_b.ranking_subcontract_snapshot,
        bound_projected_move_indices=receipt_b.bound_projected_move_indices,
        bound_projected_moves=receipt_b.bound_projected_moves,
        bound_credit_q31=receipt_b.bound_credit_q31,
        credit_q31_hash=receipt_b.credit_q31_hash,
        projected_move_indices_hash=receipt_b.projected_move_indices_hash,
        projected_moves_hash=receipt_b.projected_moves_hash,
    )
    with pytest.raises(ValueError, match="cross-bind|hash|bind"):
        validate_integer_credit_axis_integration_receipt(forged)


def test_data_hash_mismatch_rejected() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(receipt, credit_q31_hash="0" * 64)
    with pytest.raises(ValueError, match="credit_q31_hash mismatch"):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_comparable_run_id_mismatch_rejected() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(
        receipt,
        comparable_set_id="mismatched-comparable",
    )
    with pytest.raises(ValueError, match="comparable_set_id bind mismatch"):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_ranking_divergence_self_consistent_not_exact() -> None:
    green = _green_integration_receipt()
    ranking_snapshot = replace(
        green.ranking_subcontract_snapshot,
        branch_id=BR_F_RANKING_PRECISION_DIVERGENCE,
        drop_in_float32_parity_pass=False,
        strict_integer_self_consistency_pass=True,
        precision_divergence_count=1,
        bin_boundary_divergence_count=0,
        tie_group_divergence_count=0,
        measurement_invalid_count=0,
        representation_limit_count=0,
        partial_coverage_count=0,
        integer_vs_float_rank_mismatch_count=0,
        vote_mismatch_count=0,
    )
    gap_receipt = build_integer_credit_axis_integration_receipt(
        candidate_events=green.bound_candidate_attribution_events.as_integer_marginal_attribution_events(),
        q_levels_flat=green.bound_q_levels_flat,
        bound_projected_move_indices=green.bound_projected_move_indices,
        bound_projected_moves=green.bound_projected_moves,
        bound_credit_q31=green.bound_credit_q31,
        attribution_subcontract_snapshot=green.attribution_subcontract_snapshot,
        ranking_subcontract_snapshot=ranking_snapshot,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id=green.comparable_set_id,
        reference_oracle_run_id=green.reference_oracle_run_id,
        candidate_run_id=green.candidate_run_id,
        credit_law_id=CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    )
    assert gap_receipt.ranking_subcontract_pass is False
    assert gap_receipt.branch_id == BRANCH_D_RANKING_GAP
    validate_integer_credit_axis_integration_receipt(gap_receipt)


def test_dense_leak_precedence() -> None:
    branch = classify_integer_native_optimizer_credit_path_branch(
        candidate_alloc_guard_pass=False,
        candidate_dense_surfaces_observed=(AUDIT_NO_DENSE_INT_ACCUM,),
        candidate_dense_integer_scratch_observed=True,
        capture_transient_discriminator_pass=True,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
        attribution_subcontract_pass=True,
        ranking_subcontract_pass=True,
        comparable_set_complete=True,
    )
    assert branch == BRANCH_D_DENSE_LEAK


def test_hidden_fp_precedence() -> None:
    capture, weight_shape, q_levels_flat = _dry_run_capture_fixture()
    receipt = prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="hidden-fp-v1",
        reference_oracle_run_id="ref-hidden-fp",
        candidate_run_id="candidate-hidden-fp",
        capture_retained_fp_tensor_count=1,
    )
    assert receipt.branch_id == BRANCH_D_HIDDEN_FP_STRUCTURAL
    validate_integer_credit_axis_integration_receipt(receipt)


def test_reference_wire_only_cannot_branch_8() -> None:
    branch = classify_integer_native_optimizer_credit_path_branch(
        candidate_alloc_guard_pass=True,
        candidate_dense_surfaces_observed=(),
        candidate_dense_integer_scratch_observed=False,
        capture_transient_discriminator_pass=True,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        attribution_subcontract_pass=True,
        ranking_subcontract_pass=True,
        comparable_set_complete=True,
    )
    assert branch == BRANCH_D_WIRE_ONLY_REFERENCE_GAP
    assert branch != BRANCH_D_INTEGER_VIABLE


def test_attribution_fail_representation_limit() -> None:
    branch = classify_integer_native_optimizer_credit_path_branch(
        candidate_alloc_guard_pass=True,
        candidate_dense_surfaces_observed=(),
        candidate_dense_integer_scratch_observed=False,
        capture_transient_discriminator_pass=True,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
        attribution_subcontract_pass=False,
        ranking_subcontract_pass=True,
        comparable_set_complete=True,
    )
    assert branch == BRANCH_D_REPRESENTATION_LIMIT


def test_partial_coverage_branch() -> None:
    capture, weight_shape, q_levels_flat = _dry_run_capture_fixture()
    receipt = prove_integer_credit_axis_integration(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_spec=default_dry_run_rank_vote_spec(),
        comparable_set_id="partial-v1",
        reference_oracle_run_id="ref-partial",
        candidate_run_id="candidate-partial",
        partial_coverage_only=True,
        comparable_set_complete=True,
    )
    assert receipt.branch_id == BRANCH_D_PARTIAL_COVERAGE
    validate_integer_credit_axis_integration_receipt(receipt)


def test_all_green_integer_viable_cpu_evidence_only() -> None:
    receipt = _green_integration_receipt()
    assert receipt.branch_id == BRANCH_D_INTEGER_VIABLE
    assert receipt.integration_authority_level == INTEGRATION_AUTHORITY_CPU_EVIDENCE_ONLY
    assert receipt.attribution_subcontract_pass is True
    assert receipt.ranking_subcontract_pass is True
    assert receipt.ranking_subcontract_snapshot.branch_id == BR_F_RANKING_INTEGER_EXACT
    snapshot = integer_credit_axis_integration_hard_false_snapshot()
    for field in FORBIDDEN_INTEGRATION_RECEIPT_FIELDS:
        assert getattr(receipt, field) is False
        assert snapshot[field] is False
    assert receipt.fp_exception_caveat == OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT
    for claim in OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS:
        assert claim in INTEGER_CREDIT_AXIS_INTEGRATION_NON_CLAIMS
    validate_integer_credit_axis_integration_receipt(receipt)


def test_stored_branch_id_mismatch_rejected() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(receipt, branch_id=BRANCH_D_RANKING_GAP)
    with pytest.raises(ValueError, match="branch_id mismatch"):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_rank_bin_spec_hash_recomputed() -> None:
    receipt = _green_integration_receipt()
    tampered = replace(receipt, rank_bin_spec_hash="0" * 64)
    with pytest.raises(ValueError, match="rank_bin_spec_hash mismatch"):
        validate_integer_credit_axis_integration_receipt(tampered)


def test_attribution_selected_for_moves_matches_rank_votes_expression() -> None:
    capture, weight_shape, q_levels_flat = _dry_run_capture_fixture()
    with candidate_dense_integer_dispatch_observation(weight_shape):
        candidate_events, _ = streaming_sparse_attribution_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            weight_shape=weight_shape,
        )
    move_indices, _ = projected_moves_from_integer_attribution(candidate_events, q_levels_flat)
    selected = _attribution_selected_for_moves(candidate_events, move_indices)
    index_to_pos = {
        int(index): pos for pos, index in enumerate(candidate_events.flat_indices.tolist())
    }
    expected = torch.tensor(
        [
            int(candidate_events.attribution_q31[index_to_pos[int(index)]].item())
            for index in move_indices.tolist()
        ],
        dtype=torch.int32,
    )
    assert torch.equal(selected, expected)


def test_selector_helper_shared_by_validator_cross_bind() -> None:
    receipt = _green_integration_receipt()
    bound = receipt.bound_candidate_attribution_events
    events = bound.as_integer_marginal_attribution_events()
    selected = _attribution_selected_for_moves(events, receipt.bound_projected_move_indices)
    assert torch.equal(selected, _attribution_selected_for_moves(bound, receipt.bound_projected_move_indices))
