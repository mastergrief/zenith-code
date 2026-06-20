"""B1 CPU fail-closed smoke for QaccApplyParityReceipt.

Tests A–L: validate anti-laundering guard behavior without GPU/CUDA.
"""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.qacc_apply_parity_receipt import (
    QACC_APPLY_B1_BLOCKED_REASON,
    QACC_APPLY_B1_NON_CLAIMS,
    QACC_APPLY_PARITY_RECEIPT_SCHEMA_VERSION,
    CPU_ORACLE_COMMIT_SHA_SHORT,
    QaccApplyParityReceipt,
    build_qacc_apply_parity_receipt,
    validate_qacc_apply_parity_receipt,
)


# --- Test A: default receipt passes validation ---
def test_a_default_receipt_passes_validation() -> None:
    """Test A: build with all defaults → no exception; validator accepts."""
    receipt = build_qacc_apply_parity_receipt()
    assert receipt.qacc_apply_parity_pass is False
    assert receipt.q_acc_apply_cpu_reference is True
    assert receipt.q_acc_apply_final_row_torch_cuda_reference is True
    assert receipt.q_acc_apply_native_hot_loop_kernel is False
    assert receipt.native_call_path_marker_present is False
    assert receipt.global_cap_gpu_native_marker_seen is False
    assert receipt.torch_cuda_ref_under_cap_rows_invoked is False
    assert receipt.liveness_fail is False
    assert receipt.blocked_reason == QACC_APPLY_B1_BLOCKED_REASON
    assert receipt.non_claims == QACC_APPLY_B1_NON_CLAIMS
    validate_qacc_apply_parity_receipt(receipt)


# --- Test B: parity_pass=True without native proof → REJECTED by builder ---
def test_b_parity_pass_true_without_native_rejected_by_builder() -> None:
    """Test B: builder raises ValueError when parity_pass=True."""
    with pytest.raises(ValueError, match="cannot mint qacc_apply_parity_pass=True"):
        build_qacc_apply_parity_receipt(qacc_apply_parity_pass=True)


# --- Test C: global_cap_gpu_native_marker_seen=True → REJECTED by builder ---
def test_c_global_cap_gpu_native_marker_true_rejected_by_builder() -> None:
    """Test C: builder raises ValueError when global_cap_gpu_native_marker_seen=True."""
    with pytest.raises(
        ValueError, match="global_cap_gpu_native_marker_seen=True is prohibited"
    ):
        build_qacc_apply_parity_receipt(global_cap_gpu_native_marker_seen=True)


# --- Test D: torch_cuda_ref_under_cap_rows_invoked=True → REJECTED by builder ---
def test_d_torch_cuda_ref_invoked_true_rejected_by_builder() -> None:
    """Test D: builder raises ValueError when torch_cuda_ref_under_cap_rows_invoked=True."""
    with pytest.raises(
        ValueError, match="torch_cuda_ref_under_cap_rows_invoked=True is prohibited"
    ):
        build_qacc_apply_parity_receipt(torch_cuda_ref_under_cap_rows_invoked=True)


# --- Test E: validate catches laundering (parity_pass=True, native_kernel=False) ---
def test_e_validate_rejects_laundered_receipt() -> None:
    """Test E: hand-crafted receipt with parity_pass=True triggers validator ValueError."""
    receipt = QaccApplyParityReceipt(
        qacc_apply_parity_pass=True,
        q_acc_apply_cpu_reference=False,
        q_acc_apply_native_hot_loop_kernel=False,
        blocked_reason=QACC_APPLY_B1_BLOCKED_REASON,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )
    with pytest.raises(ValueError, match="qacc_apply_parity_pass must be False"):
        validate_qacc_apply_parity_receipt(receipt)


# --- Test F: schema_version mismatch → REJECTED ---
def test_f_schema_version_mismatch_rejected() -> None:
    """Test F: wrong schema_version triggers validator ValueError."""
    receipt = QaccApplyParityReceipt(
        schema_version="wrong_schema",
        qacc_apply_parity_pass=False,
        q_acc_apply_cpu_reference=True,
        q_acc_apply_native_hot_loop_kernel=False,
        blocked_reason=QACC_APPLY_B1_BLOCKED_REASON,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )
    with pytest.raises(ValueError, match="schema_version mismatch"):
        validate_qacc_apply_parity_receipt(receipt)


# --- Test G: blocked_reason exact match ---
def test_g_blocked_reason_exact_match() -> None:
    """Test G: default receipt blocked_reason equals exact B1 constant."""
    receipt = build_qacc_apply_parity_receipt()
    assert receipt.blocked_reason == QACC_APPLY_B1_BLOCKED_REASON
    assert QACC_APPLY_B1_BLOCKED_REASON != ""
    validate_qacc_apply_parity_receipt(receipt)


# --- Test H: marker-present / no-kernel → REJECTED (AMENDMENT 1) ---
def test_h_marker_present_no_kernel_rejected_by_builder() -> None:
    """Test H: native_call_path_marker_present=True + native_hot_loop_kernel=False
    triggers builder ValueError (laundering trap)."""
    with pytest.raises(
        ValueError, match="native_call_path_marker_present=True requires"
    ):
        build_qacc_apply_parity_receipt(
            native_call_path_marker_present=True,
            q_acc_apply_native_hot_loop_kernel=False,
        )


def test_h_validate_rejects_marker_present_no_kernel() -> None:
    """Test H (validator path): hand-crafted receipt with marker_present=True,
    native_hot_loop_kernel=False triggers validator ValueError."""
    receipt = QaccApplyParityReceipt(
        qacc_apply_parity_pass=False,
        q_acc_apply_cpu_reference=True,
        q_acc_apply_native_hot_loop_kernel=False,
        native_call_path_marker_present=True,
        blocked_reason=QACC_APPLY_B1_BLOCKED_REASON,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )
    with pytest.raises(
        ValueError, match="native_call_path_marker_present must be False"
    ):
        validate_qacc_apply_parity_receipt(receipt)


# --- FIX 4 new tests: final_row=False, cpu_ref=False, native_hot_loop=True, liveness_fail=True ---

# Test I: final_row_torch_cuda_reference=False via builder → REJECTED
def test_i_final_row_false_rejected_by_builder() -> None:
    """FIX 4: final-row torch-CUDA reference must stay True in B1."""
    with pytest.raises(
        ValueError, match="q_acc_apply_final_row_torch_cuda_reference must be True"
    ):
        build_qacc_apply_parity_receipt(q_acc_apply_final_row_torch_cuda_reference=False)


def test_i_validate_rejects_final_row_false() -> None:
    """FIX 4 (validator path): hand-crafted receipt with final_row=False."""
    receipt = QaccApplyParityReceipt(
        qacc_apply_parity_pass=False,
        q_acc_apply_cpu_reference=True,
        q_acc_apply_final_row_torch_cuda_reference=False,
        q_acc_apply_native_hot_loop_kernel=False,
        blocked_reason=QACC_APPLY_B1_BLOCKED_REASON,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )
    with pytest.raises(
        ValueError, match="q_acc_apply_final_row_torch_cuda_reference must be True"
    ):
        validate_qacc_apply_parity_receipt(receipt)


# Test J: cpu_reference=False via builder → REJECTED
def test_j_cpu_reference_false_rejected_by_builder() -> None:
    """FIX 4: cpu_reference must stay True in B1."""
    with pytest.raises(
        ValueError, match="q_acc_apply_cpu_reference must be True"
    ):
        build_qacc_apply_parity_receipt(q_acc_apply_cpu_reference=False)


# Test K: native_hot_loop_kernel=True via builder → REJECTED
def test_k_native_hot_loop_kernel_true_rejected_by_builder() -> None:
    """FIX 4: native hot-loop kernel must stay False in B1."""
    with pytest.raises(
        ValueError, match="q_acc_apply_native_hot_loop_kernel must be False"
    ):
        build_qacc_apply_parity_receipt(q_acc_apply_native_hot_loop_kernel=True)


# Test L: liveness_fail=True → REJECTED (builder + validator paths)
def test_l_liveness_fail_true_rejected_by_builder() -> None:
    """FIX 4: liveness_fail must stay False in B1 (builder path)."""
    with pytest.raises(
        ValueError, match="liveness_fail must be False"
    ):
        build_qacc_apply_parity_receipt(liveness_fail=True)


def test_l_validate_rejects_liveness_fail_true() -> None:
    """FIX 4: liveness_fail must stay False in B1 (validator path)."""
    receipt = QaccApplyParityReceipt(
        qacc_apply_parity_pass=False,
        q_acc_apply_cpu_reference=True,
        q_acc_apply_final_row_torch_cuda_reference=True,
        q_acc_apply_native_hot_loop_kernel=False,
        liveness_fail=True,
        blocked_reason=QACC_APPLY_B1_BLOCKED_REASON,
        non_claims=QACC_APPLY_B1_NON_CLAIMS,
    )
    with pytest.raises(ValueError, match="liveness_fail must be False"):
        validate_qacc_apply_parity_receipt(receipt)


# --- Supplementary: builder validates before return (FIX 3 evidence) ---
def test_builder_calls_validate() -> None:
    """FIX 3 evidence: builder returns a receipt that survives validate()."""
    receipt = build_qacc_apply_parity_receipt()
    # Should not raise — proves builder calls validate before return
    validate_qacc_apply_parity_receipt(receipt)


# --- Supplementary: non_claims exact match ---
def test_non_claims_exact_match() -> None:
    receipt = build_qacc_apply_parity_receipt()
    assert receipt.non_claims == QACC_APPLY_B1_NON_CLAIMS
    validate_qacc_apply_parity_receipt(receipt)


# --- Supplementary: sha_short exact match ---
def test_cpu_oracle_sha_short_exact() -> None:
    receipt = build_qacc_apply_parity_receipt()
    assert receipt.cpu_oracle_commit_sha_short == CPU_ORACLE_COMMIT_SHA_SHORT
