"""CPU tests for 3C-C1 optimizer credit-state proof contract + no-hidden-FP audit."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    build_optimizer_excluding_eligible_masters,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_AUDIT_PASS_CPU,
    BRANCH_3C_C_CAPTURE_LAUNDER,
    BRANCH_3C_C_DENSE_LEAK,
    BRANCH_3C_C_MEASUREMENT_INVALID,
    OBSERVATION_PROBE_MODE_ALLOC_GUARD,
    OBSERVATION_PROBE_MODE_STATIC,
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
    build_optimizer_credit_state_fail_closed_receipt,
    validate_optimizer_credit_state_fail_closed_receipt,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (
    AUDIT_CAPTURE_LAUNDER,
    DENSE_SURFACE_CREDIT,
    build_optimizer_credit_state_receipt_from_audit,
    run_integer_path_dense_surface_observation_with_alloc_guard,
    run_optimizer_credit_state_no_hidden_fp_audit,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _dry_run_fixture() -> tuple[dict, torch.Tensor, tuple[int, int], dict[str, Any]]:
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": tensor_state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        captures = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in tensor_state.q_levels.shape)
    return captures, q.reshape(-1), weight_shape, eligible


def test_default_receipt_v1_still_blocks_flip():
    receipt = build_optimizer_credit_state_fail_closed_receipt()
    validate_optimizer_credit_state_fail_closed_receipt(receipt)
    assert receipt.ready_to_flip is False
    assert receipt.optimizer_credit_state_sub2_claim is False
    assert receipt.optimizer_credit_state_resolved is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.real_native_integer_attribution_present is False
    assert receipt.real_native_integer_credit_ranking_present is False
    assert receipt.gpu_runtime_receipt_present is False
    assert receipt.cpu_reference_path_only is True
    assert receipt.audit_observation_complete is False


def test_ready_to_flip_rejects_without_gpu_receipt():
    with pytest.raises(ValueError, match="ready_to_flip cannot be true"):
        build_optimizer_credit_state_fail_closed_receipt(
            real_native_integer_attribution_present=True,
            real_native_integer_credit_ranking_present=True,
            no_hidden_bf16_fp_optimizer_state_proven=True,
            optimizer_state_eligible_exclusion_proven=True,
            no_hidden_fp_audit_branch_id=BRANCH_3C_C_AUDIT_PASS_CPU,
            no_hidden_fp_audit_receipt_sha256="a" * 64,
            audit_observation_complete=True,
            observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
            ready_to_flip=True,
        )


def test_cpu_reference_path_rejects_native_present_flags():
    with pytest.raises(ValueError, match="cpu_reference_path_only"):
        build_optimizer_credit_state_fail_closed_receipt(
            real_native_integer_attribution_present=True,
        )


def test_dense_fp_observed_blocks_native_present_claim():
    with pytest.raises(ValueError, match="dense_fp_intermediate_tensors_observed"):
        build_optimizer_credit_state_fail_closed_receipt(
            dense_fp_intermediate_tensors_observed=("credit",),
            real_native_integer_credit_ranking_present=True,
        )


def test_credit_capture_launder_audit_fail_closed():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=True,
        credit_capture_observed=True,
        native_attribution_claimed=True,
    )
    assert audit.branch_id == BRANCH_3C_C_CAPTURE_LAUNDER
    assert audit.fp_exception_laundering_detected is True
    assert audit.optimizer_state_eligible_exclusion_proven is False


def test_audit_opt_excl_passes_on_exclusion_checks():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=True,
    )
    receipt = build_optimizer_credit_state_receipt_from_audit(audit)
    assert audit.branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    assert receipt.optimizer_state_eligible_exclusion_proven is True
    assert receipt.no_hidden_bf16_fp_optimizer_state_proven is True
    assert receipt.no_hidden_fp_audit_branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    assert receipt.no_hidden_fp_audit_receipt_sha256
    assert receipt.ready_to_flip is False


def test_audit_dense_leak_detected():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(DENSE_SURFACE_CREDIT,),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_DENSE_LEAK


def test_empty_observed_without_completeness_is_measurement_invalid():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=False,
    )
    assert audit.branch_id == BRANCH_3C_C_MEASUREMENT_INVALID
    assert audit.optimizer_state_eligible_exclusion_proven is False
    receipt = build_optimizer_credit_state_receipt_from_audit(audit)
    assert receipt.optimizer_state_eligible_exclusion_proven is False
    assert receipt.no_hidden_fp_audit_branch_id == BRANCH_3C_C_MEASUREMENT_INVALID


def test_empty_observed_with_completeness_can_pass():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
        audit_observation_complete=True,
    )
    receipt = build_optimizer_credit_state_receipt_from_audit(audit)
    assert audit.branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    assert receipt.optimizer_state_eligible_exclusion_proven is True


def test_forged_exclusion_without_audit_pass_rejected():
    receipt = build_optimizer_credit_state_fail_closed_receipt()
    forged = replace(
        receipt,
        optimizer_state_eligible_exclusion_proven=True,
        no_hidden_fp_audit_branch_id=BRANCH_3C_C_MEASUREMENT_INVALID,
        no_hidden_fp_audit_receipt_sha256="b" * 64,
        audit_observation_complete=True,
        observation_probe_mode=OBSERVATION_PROBE_MODE_STATIC,
    )
    with pytest.raises(ValueError, match="optimizer_state_eligible_exclusion_proven requires"):
        validate_optimizer_credit_state_fail_closed_receipt(forged)


def test_invalid_probe_mode_is_measurement_invalid():
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks={
            "eligible_params_in_optimizer": 0,
            "eligible_optimizer_state_entries": 0,
        },
        observed_dense_surfaces=(),
        observation_probe_mode="unproven",
        audit_observation_complete=True,
    )
    assert audit.branch_id == BRANCH_3C_C_MEASUREMENT_INVALID


def test_alloc_guard_instrumented_integer_path_observation():
    captures, q_flat, weight_shape, eligible = _dry_run_fixture()
    observed, probe_mode = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    assert probe_mode == OBSERVATION_PROBE_MODE_ALLOC_GUARD
    assert observed == ()
    model = _Tiny()
    opt, checks = build_optimizer_excluding_eligible_masters(model, eligible, lr=0.0)
    assert opt is not None
    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks=checks,
        observed_dense_surfaces=observed,
        observation_probe_mode=probe_mode,
        audit_observation_complete=True,
    )
    receipt = build_optimizer_credit_state_receipt_from_audit(audit)
    assert audit.branch_id == BRANCH_3C_C_AUDIT_PASS_CPU
    assert receipt.integer_attribution_law_id == INTEGER_MARGINAL_ATTRIBUTION_LAW_ID
    assert receipt.integer_credit_ranking_law_id == CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0
    assert (
        receipt.parity_fixture_sha256
        == OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256
    )


def test_existing_v0_tests_remain_green():
    import calm.llm_computer.tests.test_hrm_text_158_optimizer_credit_state as legacy

    legacy.test_optimizer_credit_state_fail_closed_receipt_enumerates_dense_debt_without_flip()
    legacy.test_optimizer_credit_state_fail_closed_receipt_rejects_missing_unknown_and_laundering()
    legacy.test_optimizer_credit_state_receipt_rejects_drifted_contract_fields()
    legacy.test_native_full_stack_exports_optimizer_credit_state_contract_surface()
