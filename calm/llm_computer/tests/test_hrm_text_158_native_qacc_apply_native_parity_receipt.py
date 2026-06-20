"""B2-2a CPU fail-closed smoke for QaccApplyNativeParityReceipt.

Tests: blocked-state + pass-state + builder-cannot-mint + §11c enumeration +
half-pass + literal identity + token key-set + blocked-with-token.
"""
from __future__ import annotations

from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QACC_APPLY_B2_BLOCKED_REASON,
    QACC_APPLY_B2_NON_CLAIMS,
    QaccApplyNativeParityReceipt,
    QaccApplyNativeToken,
    build_qacc_apply_native_parity_receipt,
    validate_qacc_apply_native_parity_receipt,
    hash_qacc_apply_input_payloads,
    hash_qacc_apply_output_payloads,
    canonical_tensor_payload_sha256,
)

_LITERAL_SCHEMA = "hrm_text_158_qacc_apply_native_parity/v0.b2"
_LITERAL_B1_COMMIT = "ebfd6868ba8179226357cf93a3e06a9646d27751"


def _valid_mock_token() -> QaccApplyNativeToken:
    """Fixture with FULL key set per contract §3 / FIX 3."""
    return QaccApplyNativeToken(
        kernel_family="triton_qacc_apply",
        kernel_symbol="_qacc_apply_kernel_v0",
        kernel_source_sha256="a" * 64,
        wrapper_launch_nonce="deadbeef12345678",
        input_payload_hashes={
            "q_levels": "b" * 64,
            "new_accumulators": "c" * 64,
            "accepted_indices": "d" * 64,
            "accepted_directions": "e" * 64,
            "accepted_thresholds": "f" * 64,
            "replay_veto_indices": "g" * 64,
            "replay_veto_directions": "h" * 64,
            "replay_veto_thresholds": "i" * 64,
            "original_accumulators": "j" * 64,
            "mutate_outputs": "k" * 64,
        },
        output_payload_hashes={
            "q_levels": "l" * 64,
            "accumulators": "m" * 64,
        },
        backend="cuda",
        launch_time_ns=123456789,
    )


def _blocked_kwargs(**overrides: Any) -> dict:
    base = {
        "blocked_reason": QACC_APPLY_B2_BLOCKED_REASON,
        "non_claims": QACC_APPLY_B2_NON_CLAIMS,
    }
    base.update(overrides)
    return base


def _pass_kwargs(**overrides: Any) -> dict:
    tok = _valid_mock_token()
    base = dict(
        qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        q_acc_apply_cpu_reference=False,
        q_acc_apply_final_row_torch_cuda_reference=False,
        native_hot_loop_kernel=True,
        native_call_path_marker_present=True,
        exact_q_output_hash="o" * 64,
        exact_acc_output_hash="o" * 64,
        cpu_oracle_q_hash="o" * 64,
        cpu_oracle_acc_hash="o" * 64,
        wrapper_token=tok,
        blocked_reason=QACC_APPLY_B2_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_NON_CLAIMS,
    )
    base.update(overrides)
    return base


# =============================================================================
# Phase 1 — Literal identity (FIX 5, hard-coded, no import of expected value)
# =============================================================================

def test_literal_schema_version() -> None:
    receipt = build_qacc_apply_native_parity_receipt()
    # Hard-coded literal assertion — detects source truncation or drift
    assert receipt.schema_version == _LITERAL_SCHEMA


def test_literal_parent_b1_commit() -> None:
    receipt = build_qacc_apply_native_parity_receipt()
    assert receipt.parent_b1_commit == _LITERAL_B1_COMMIT


# =============================================================================
# Phase 2 — Blocked-state default
# =============================================================================

def test_blocked_default_validates() -> None:
    receipt = build_qacc_apply_native_parity_receipt()
    assert receipt.qacc_apply_parity_pass is False
    assert receipt.gpu_command_satisfied is False
    assert receipt.wrapper_token is None  # FIX 4: blocked = no token
    assert receipt.blocked_reason == QACC_APPLY_B2_BLOCKED_REASON
    assert receipt.non_claims == QACC_APPLY_B2_NON_CLAIMS
    validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 3 — Builder CANNOT mint pass (AMENDMENT 1)
# =============================================================================

def test_builder_rejects_parity_pass_true() -> None:
    with pytest.raises(ValueError, match="CANNOT mint qacc_apply_parity_pass=True"):
        build_qacc_apply_native_parity_receipt(qacc_apply_parity_pass=True)


def test_builder_rejects_gpu_command_satisfied_true() -> None:
    with pytest.raises(ValueError, match="CANNOT mint gpu_command_satisfied=True"):
        build_qacc_apply_native_parity_receipt(gpu_command_satisfied=True)


def test_builder_with_mock_token_still_blocked() -> None:
    receipt = build_qacc_apply_native_parity_receipt(
        wrapper_token=_valid_mock_token(),
        native_hot_loop_kernel=True,
        native_call_path_marker_present=True,
    )
    assert receipt.qacc_apply_parity_pass is False
    assert receipt.gpu_command_satisfied is False
    assert receipt.wrapper_token is None  # FIX 4: forced None


# =============================================================================
# Phase 4 — Validator-only pass fixture (NOT proof authority)
# =============================================================================

def test_validator_accepts_all_true_fixture() -> None:
    receipt = QaccApplyNativeParityReceipt(**_pass_kwargs())
    validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 5 — Half-pass states (named laundering hazard, direct validator)
# =============================================================================

def test_half_pass_parity_true_gpu_false() -> None:
    """parity=True + gpu=False triggers the single-truthy OR branch."""
    kw = _pass_kwargs()
    kw["gpu_command_satisfied"] = False
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="pass-state requires BOTH"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_half_pass_parity_false_gpu_true() -> None:
    """parity=False + gpu=True triggers the single-truthy OR branch."""
    kw = _pass_kwargs()
    kw["qacc_apply_parity_pass"] = False
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="pass-state requires BOTH"):
        validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 6 — Unconditional rejects (both states)
# =============================================================================

@pytest.mark.parametrize(
    "field, bad_value, match",
    [
        ("global_cap_gpu_native_marker_seen", True, "global_cap_gpu_native_marker_seen"),
        ("torch_cuda_ref_under_cap_rows_invoked", True, "torch_cuda_ref_under_cap_rows_invoked"),
        ("accepted_indices_unique", False, "accepted_indices_unique"),
        ("replay_indices_unique", False, "replay_indices_unique"),
        ("accepted_then_replay_order", False, "accepted_then_replay_order"),
        ("mutate_outputs_path", "False", "mutate_outputs_path"),
        ("parity_atol", 1e-6, "parity_atol"),
        ("parity_rtol", 1e-5, "parity_rtol"),
        ("blocked_reason", "wrong", "blocked_reason"),
        ("non_claims", ("wrong",), "non_claims"),
        ("schema_version", "wrong", "schema_version mismatch"),
        ("parent_b1_commit", "0000000", "parent_b1_commit mismatch"),
    ],
)
def test_unconditional_rejects(field: str, bad_value: Any, match: str) -> None:
    kw = _blocked_kwargs(**{field: bad_value})
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match=match):
        validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 7 — Blocked-state with token → reject (FIX 4)
# =============================================================================

def test_blocked_state_with_token_rejected() -> None:
    kw = _blocked_kwargs(wrapper_token=_valid_mock_token())
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="wrapper_token must be None"):
        validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 8 — Pass-state illegal values
# =============================================================================

def test_pass_cpu_reference_true() -> None:
    kw = _pass_kwargs(q_acc_apply_cpu_reference=True)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="q_acc_apply_cpu_reference"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_final_row_ref_true() -> None:
    kw = _pass_kwargs(q_acc_apply_final_row_torch_cuda_reference=True)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="q_acc_apply_final_row_torch_cuda_reference"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_native_hot_loop_false() -> None:
    kw = _pass_kwargs(native_hot_loop_kernel=False)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="native_hot_loop_kernel"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_marker_present_false() -> None:
    kw = _pass_kwargs(native_call_path_marker_present=False)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="native_call_path_marker_present"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_empty_q_output_hash() -> None:
    kw = _pass_kwargs(exact_q_output_hash="")
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="exact_q_output_hash"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_empty_acc_output_hash() -> None:
    kw = _pass_kwargs(exact_acc_output_hash="")
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="exact_acc_output_hash"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_q_hash_mismatch() -> None:
    kw = _pass_kwargs(cpu_oracle_q_hash="c" * 64)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="exact_q_output_hash"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_pass_acc_hash_mismatch() -> None:
    kw = _pass_kwargs(cpu_oracle_acc_hash="c" * 64)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="exact_acc_output_hash"):
        validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 9 — Token illegal values + key-set (FIX 3)
# =============================================================================

def test_token_none() -> None:
    kw = _pass_kwargs(wrapper_token=None)
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="wrapper_token"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_wrong_family() -> None:
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family="wrong", kernel_symbol="s", kernel_source_sha256="a" * 64,
        wrapper_launch_nonce="n", input_payload_hashes={"q_levels": "b" * 64},
        output_payload_hashes={"q_levels": "c" * 64, "accumulators": "d" * 64},
        backend="cuda", launch_time_ns=1,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="kernel_family"):
        validate_qacc_apply_native_parity_receipt(receipt)


@pytest.mark.parametrize("bad_key_set,match", [
    ({"q_levels": "b" * 64}, "input_payload_hashes keys"),  # missing 9 keys
    ({
        "q_levels": "b" * 64,
        "new_accumulators": "c" * 64,
        "accepted_indices": "d" * 64,
        "accepted_directions": "e" * 64,
        "accepted_thresholds": "f" * 64,
        "replay_veto_indices": "g" * 64,
        "replay_veto_directions": "h" * 64,
        "replay_veto_thresholds": "i" * 64,
        "original_accumulators": "j" * 64,
        "mutate_outputs": "k" * 64,
        "EXTRA": "z" * 64,
    }, "input_payload_hashes keys"),  # extra key
])
def test_token_input_key_set(bad_key_set: dict, match: str) -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=bad_key_set,
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match=match):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_output_key_set_missing() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes={"q_levels": "l" * 64},  # missing accumulators
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="output_payload_hashes keys"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_empty_symbol() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol="",
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="kernel_symbol"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_empty_source_sha() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256="", wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="kernel_source_sha256"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_empty_nonce() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256, wrapper_launch_nonce="",
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="wrapper_launch_nonce"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_input_hashes_not_dict() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=None,  # type: ignore[arg-type]
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="input_payload_hashes"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_output_hashes_not_dict() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=None,  # type: ignore[arg-type]
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="output_payload_hashes"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_backend_wrong() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=tok.output_payload_hashes,
        backend="cpu", launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="backend"):
        validate_qacc_apply_native_parity_receipt(receipt)


def test_token_launch_time_nonpositive() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes=tok.output_payload_hashes,
        backend=tok.backend, launch_time_ns=0,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="launch_time_ns"):
        validate_qacc_apply_native_parity_receipt(receipt)


# =============================================================================
# Phase 10 — Hash helper coverage (helper-breakage fix)
# =============================================================================

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    TOKEN_INPUT_PAYLOAD_KEYS,
    TOKEN_OUTPUT_PAYLOAD_KEYS,
)


def test_hash_helpers_all_keys_present_defaults() -> None:
    """Without optional args helper still emits all 10 input keys (b"" sentinel)."""
    input_hashes = hash_qacc_apply_input_payloads(
        q_levels_bytes=b"\x01\x02",
        new_accumulators_bytes=b"\x03\x04",
        accepted_indices_bytes=b"\x00\x01",
        accepted_directions_bytes=b"\x01\xff",
        accepted_thresholds_bytes=b"\x00\x00\x00\x0a",
        mutate_outputs=True,
    )
    assert set(input_hashes.keys()) == TOKEN_INPUT_PAYLOAD_KEYS
    for key in TOKEN_INPUT_PAYLOAD_KEYS:
        assert len(input_hashes[key]) == 64
    # Optional-absent keys hash b"" (deterministic sentinel)
    assert input_hashes["replay_veto_indices"] == canonical_tensor_payload_sha256(b"")


def test_hash_helpers_all_keys_present_with_optional() -> None:
    """With all optional args all 10 keys are present with real hashes."""
    input_hashes = hash_qacc_apply_input_payloads(
        q_levels_bytes=b"\x01",
        new_accumulators_bytes=b"\x02",
        accepted_indices_bytes=b"\x03",
        accepted_directions_bytes=b"\x04",
        accepted_thresholds_bytes=b"\x05",
        replay_veto_indices_bytes=b"\x06",
        replay_veto_directions_bytes=b"\x07",
        replay_veto_thresholds_bytes=b"\x08",
        original_accumulators_bytes=b"\x09",
        mutate_outputs=True,
    )
    assert set(input_hashes.keys()) == TOKEN_INPUT_PAYLOAD_KEYS
    assert input_hashes["replay_veto_indices"] != canonical_tensor_payload_sha256(b"")


def test_hash_outputs_exact_two_keys() -> None:
    output_hashes = hash_qacc_apply_output_payloads(
        q_levels_bytes=b"\x01\x02",
        accumulators_bytes=b"\x03\x04",
    )
    assert set(output_hashes.keys()) == TOKEN_OUTPUT_PAYLOAD_KEYS
    assert len(output_hashes["q_levels"]) == 64
    assert len(output_hashes["accumulators"]) == 64


def test_end_to_end_helper_produces_validator_accepted_token() -> None:
    """Helper → token → validate proves production path is not self-rejected."""
    input_hashes = hash_qacc_apply_input_payloads(
        q_levels_bytes=b"\x01",
        new_accumulators_bytes=b"\x02",
        accepted_indices_bytes=b"\x03",
        accepted_directions_bytes=b"\x04",
        accepted_thresholds_bytes=b"\x05",
        mutate_outputs=True,
    )
    output_hashes = hash_qacc_apply_output_payloads(
        q_levels_bytes=b"\x06", accumulators_bytes=b"\x07",
    )
    token = QaccApplyNativeToken(
        kernel_family="triton_qacc_apply",
        kernel_symbol="_qacc_apply_kernel_v0",
        kernel_source_sha256="a" * 64,
        wrapper_launch_nonce="nonce123",
        input_payload_hashes=input_hashes,
        output_payload_hashes=output_hashes,
        backend="cuda",
        launch_time_ns=1,
    )
    receipt = QaccApplyNativeParityReceipt(
        qacc_apply_parity_pass=True,
        gpu_command_satisfied=True,
        q_acc_apply_cpu_reference=False,
        q_acc_apply_final_row_torch_cuda_reference=False,
        native_hot_loop_kernel=True,
        native_call_path_marker_present=True,
        exact_q_output_hash=output_hashes["q_levels"],
        exact_acc_output_hash=output_hashes["accumulators"],
        cpu_oracle_q_hash=output_hashes["q_levels"],
        cpu_oracle_acc_hash=output_hashes["accumulators"],
        wrapper_token=token,
        blocked_reason=QACC_APPLY_B2_BLOCKED_REASON,
        non_claims=QACC_APPLY_B2_NON_CLAIMS,
    )
    validate_qacc_apply_native_parity_receipt(receipt)


def test_output_extra_key_rejected() -> None:
    tok = _valid_mock_token()
    kw = _pass_kwargs(wrapper_token=QaccApplyNativeToken(
        kernel_family=tok.kernel_family, kernel_symbol=tok.kernel_symbol,
        kernel_source_sha256=tok.kernel_source_sha256,
        wrapper_launch_nonce=tok.wrapper_launch_nonce,
        input_payload_hashes=tok.input_payload_hashes,
        output_payload_hashes={"q_levels": "l" * 64, "accumulators": "m" * 64, "EXTRA": "z" * 64},
        backend=tok.backend, launch_time_ns=tok.launch_time_ns,
    ))
    receipt = QaccApplyNativeParityReceipt(**kw)
    with pytest.raises(ValueError, match="output_payload_hashes keys"):
        validate_qacc_apply_native_parity_receipt(receipt)
