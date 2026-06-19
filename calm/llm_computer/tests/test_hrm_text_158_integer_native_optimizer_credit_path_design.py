"""CPU tests for native-integer optimizer credit path design contract (BR-3C-D v3)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
    ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
    AUDIT_NO_DENSE_INT_ACCUM,
    AUDIT_NO_DENSE_INT_ATTR,
    BRANCH_D_DENSE_LEAK,
    BRANCH_D_INTEGER_VIABLE,
    BRANCH_D_WIRE_ONLY_REFERENCE_GAP,
    EXECUTION_LANE_CANDIDATE,
    EXECUTION_LANE_REFERENCE_ORACLE,
    FORBIDDEN_DESIGN_RECEIPT_FIELDS,
    INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_NON_CLAIMS,
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
    build_design_receipt_from_attribution_lane_result,
    build_integer_native_optimizer_credit_path_design_receipt,
    integer_native_optimizer_credit_path_design_hard_false_snapshot,
    observe_current_wire_candidate_dense_integer_scratch,
    run_attribution_with_execution_lane,
    validate_integer_native_optimizer_credit_path_design_receipt,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _capture_fixture() -> tuple[dict, dict, torch.Tensor]:
    torch.manual_seed(158)
    model = _Tiny()
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
    states = {"proj": state}
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        states,
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    capture = handle.captures["proj"]
    return capture, states, handle


def test_reference_lane_allows_dense_integer_scratch():
    capture, _states, _handle = _capture_fixture()
    weight_shape = (2, 3)
    lane_result = run_attribution_with_execution_lane(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        execution_lane=EXECUTION_LANE_REFERENCE_ORACLE,
        reference_oracle_run_id="ref-oracle-run-001",
    )
    receipt = build_design_receipt_from_attribution_lane_result(
        lane_result,
        candidate_run_id="candidate-run-unused",
    )

    assert lane_result.attribution_subcontract_mode == ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL
    assert lane_result.events is not None
    assert lane_result.dense_scratch_observation.int64_accum_observed is True
    assert lane_result.candidate_alloc_guard_pass is True
    assert receipt.candidate_dense_integer_scratch_observed is False
    assert receipt.execution_lane == EXECUTION_LANE_REFERENCE_ORACLE
    assert receipt.reference_oracle_run_id == "ref-oracle-run-001"
    validate_integer_native_optimizer_credit_path_design_receipt(receipt)


def test_candidate_lane_rejects_dense_integer_scratch():
    capture, _states, _handle = _capture_fixture()
    lane_result = run_attribution_with_execution_lane(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=(2, 3),
        execution_lane=EXECUTION_LANE_CANDIDATE,
        candidate_run_id="candidate-run-001",
    )
    receipt = build_design_receipt_from_attribution_lane_result(
        lane_result,
        candidate_run_id="candidate-run-001",
    )

    assert lane_result.candidate_alloc_guard_pass is False
    assert lane_result.dense_scratch_observation.candidate_dense_integer_scratch_observed is True
    assert AUDIT_NO_DENSE_INT_ACCUM in lane_result.dense_scratch_observation.candidate_dense_integer_scratch_surfaces
    assert AUDIT_NO_DENSE_INT_ATTR in lane_result.dense_scratch_observation.candidate_dense_integer_scratch_surfaces
    assert receipt.candidate_dense_integer_scratch_observed is True
    assert receipt.candidate_alloc_guard_pass is False
    assert receipt.branch_id == BRANCH_D_DENSE_LEAK
    validate_integer_native_optimizer_credit_path_design_receipt(receipt)


def test_candidate_lane_classifies_current_wire_as_not_integer_viable():
    _capture, states, handle = _capture_fixture()
    rank_spec = default_dry_run_rank_vote_spec()
    observation = observe_current_wire_candidate_dense_integer_scratch(
        handle,
        states,
        rank_spec,
    )
    receipt = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_CANDIDATE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        candidate_run_id="candidate-wire-run-001",
        reference_oracle_run_id=None,
        candidate_alloc_guard_pass=False,
        candidate_dense_integer_scratch_observed=(
            observation.candidate_dense_integer_scratch_observed
        ),
        candidate_dense_integer_scratch_surfaces=(
            observation.candidate_dense_integer_scratch_surfaces
        ),
        attribution_subcontract_pass=True,
        wire_shape_only_pass=True,
    )

    assert observation.candidate_dense_integer_scratch_observed is True
    assert AUDIT_NO_DENSE_INT_ATTR in observation.candidate_dense_integer_scratch_surfaces
    assert receipt.branch_id == BRANCH_D_DENSE_LEAK
    validate_integer_native_optimizer_credit_path_design_receipt(receipt)


def test_reference_oracle_run_id_distinct_from_candidate():
    capture, _states, _handle = _capture_fixture()
    reference = run_attribution_with_execution_lane(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=(2, 3),
        execution_lane=EXECUTION_LANE_REFERENCE_ORACLE,
        reference_oracle_run_id="ref-oracle-run-abc",
    )
    candidate = run_attribution_with_execution_lane(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=(2, 3),
        execution_lane=EXECUTION_LANE_CANDIDATE,
        candidate_run_id="candidate-run-xyz",
    )
    ref_receipt = build_design_receipt_from_attribution_lane_result(
        reference,
        candidate_run_id="candidate-run-unused",
    )
    cand_receipt = build_design_receipt_from_attribution_lane_result(
        candidate,
        candidate_run_id="candidate-run-xyz",
    )

    assert ref_receipt.reference_oracle_run_id == "ref-oracle-run-abc"
    assert cand_receipt.candidate_run_id == "candidate-run-xyz"
    assert ref_receipt.reference_oracle_run_id != cand_receipt.candidate_run_id


@pytest.mark.parametrize("field", FORBIDDEN_DESIGN_RECEIPT_FIELDS)
def test_design_receipt_validator_rejects_forbidden_fields(field: str):
    receipt = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_REFERENCE_ORACLE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        candidate_run_id="candidate-run-forbidden",
        reference_oracle_run_id="ref-oracle-run-forbidden",
        candidate_alloc_guard_pass=True,
    )
    bad = replace(receipt, **{field: True})
    with pytest.raises(ValueError):
        validate_integer_native_optimizer_credit_path_design_receipt(bad)


def test_standing_non_claims_exact_tuple():
    receipt = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_REFERENCE_ORACLE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        candidate_run_id="candidate-run-non-claims",
        reference_oracle_run_id="ref-oracle-run-non-claims",
        candidate_alloc_guard_pass=True,
    )

    assert receipt.fp_exception_caveat == OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT
    assert receipt.non_claims[: len(OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS)] == (
        OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS
    )
    assert receipt.non_claims == INTEGER_NATIVE_OPTIMIZER_CREDIT_PATH_DESIGN_NON_CLAIMS
    assert integer_native_optimizer_credit_path_design_hard_false_snapshot() == {
        field: False for field in FORBIDDEN_DESIGN_RECEIPT_FIELDS
    }
    validate_integer_native_optimizer_credit_path_design_receipt(receipt)


def _forge_integer_viable_receipt() -> object:
    base = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_CANDIDATE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
        candidate_run_id="forge-base",
        reference_oracle_run_id="ref-forge",
        candidate_alloc_guard_pass=True,
        comparable_set_complete=True,
        attribution_subcontract_pass=True,
        ranking_subcontract_pass=True,
    )
    return replace(
        base,
        branch_id=BRANCH_D_INTEGER_VIABLE,
        candidate_dense_integer_scratch_observed=False,
        candidate_dense_surfaces_observed=(),
        candidate_dense_integer_scratch_surfaces=(),
    )


def test_branch8_validator_rejects_forged_integer_viable_missing_comparable_set():
    forged = replace(_forge_integer_viable_receipt(), comparable_set_complete=False)
    with pytest.raises(ValueError, match="comparable_set_complete"):
        validate_integer_native_optimizer_credit_path_design_receipt(forged)


def test_branch8_validator_rejects_forged_integer_viable_non_empty_dense_surfaces():
    forged = replace(
        _forge_integer_viable_receipt(),
        candidate_dense_surfaces_observed=(AUDIT_NO_DENSE_INT_ACCUM,),
    )
    with pytest.raises(ValueError, match="candidate_dense_surfaces_observed"):
        validate_integer_native_optimizer_credit_path_design_receipt(forged)


def test_branch8_validator_rejects_forged_integer_viable_non_empty_int_scratch_surfaces():
    forged = replace(
        _forge_integer_viable_receipt(),
        candidate_dense_integer_scratch_surfaces=(AUDIT_NO_DENSE_INT_ATTR,),
        candidate_dense_integer_scratch_observed=True,
    )
    with pytest.raises(ValueError, match="candidate_dense_integer_scratch_surfaces"):
        validate_integer_native_optimizer_credit_path_design_receipt(forged)


def test_validator_rejects_int_scratch_observed_without_surfaces():
    receipt = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_CANDIDATE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        candidate_run_id="consistency-run",
        reference_oracle_run_id=None,
        candidate_alloc_guard_pass=False,
        candidate_dense_integer_scratch_observed=True,
        candidate_dense_integer_scratch_surfaces=(AUDIT_NO_DENSE_INT_ACCUM, AUDIT_NO_DENSE_INT_ATTR),
    )
    bad = replace(receipt, candidate_dense_integer_scratch_surfaces=())
    with pytest.raises(ValueError, match="candidate_dense_integer_scratch_observed must match"):
        validate_integer_native_optimizer_credit_path_design_receipt(bad)


def test_validator_rejects_alloc_guard_pass_with_dense_surfaces():
    receipt = build_integer_native_optimizer_credit_path_design_receipt(
        execution_lane=EXECUTION_LANE_CANDIDATE,
        attribution_subcontract_mode=ATTRIBUTION_SUBCONTRACT_MODE_REFERENCE_DENSE_INTERNAL,
        candidate_run_id="alloc-guard-run",
        reference_oracle_run_id=None,
        candidate_alloc_guard_pass=False,
        candidate_dense_integer_scratch_observed=True,
        candidate_dense_integer_scratch_surfaces=(AUDIT_NO_DENSE_INT_ACCUM, AUDIT_NO_DENSE_INT_ATTR),
    )
    bad = replace(receipt, candidate_alloc_guard_pass=True)
    with pytest.raises(ValueError, match="candidate_alloc_guard_pass cannot be true"):
        validate_integer_native_optimizer_credit_path_design_receipt(bad)
