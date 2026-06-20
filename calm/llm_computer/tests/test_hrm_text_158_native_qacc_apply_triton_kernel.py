"""B2-2b CPU-only validation for native Triton q_acc_apply kernel + wrapper.

Tests: pure seams + input validation + row-validity preconditions + token
structure + hash-helper reuse + compare logic.  NO GPU launch.  NO receipt
minting.  triton.compile is honest-classified (compiled vs deferred).
"""
from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

_KERNEL_FILE = Path(__file__).parent.parent.parent / "hrm_text_158" / "native_full_stack" / "qacc_apply_triton_kernel.py"

# ---------------------------------------------------------------------------
# Partial import guard (module may fail import if Triton missing)
# ---------------------------------------------------------------------------
try:
    from calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel import (
        _MAX_GRID_ELEMENTS,
        _kernel_file_path_for_test,
        _kernel_source_sha256,
        _mint_qacc_apply_native_token,
        _qacc_apply_accepted_pass_kernel,
        _qacc_apply_replay_pass_kernel,
        apply_qacc_mutation_triton_native,
        compare_qacc_outputs,
    )
    _IMPORT_OK = True
except Exception as exc:
    _IMPORT_ERROR = str(exc)
    _IMPORT_OK = False
    _kernel_file_path_for_test = _KERNEL_FILE  # type: ignore
    _mint_qacc_apply_native_token = None  # type: ignore
    compare_qacc_outputs = None  # type: ignore

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QaccApplyNativeToken,
    hash_qacc_apply_input_payloads,
    hash_qacc_apply_output_payloads,
)


# =============================================================================
# Import / compilation baseline (B2-2b compile gate — honest classification)
# =============================================================================


def test_module_import_clean() -> None:
    """The module imports without errors (Triton may be absent, handled gracefully)."""
    assert _IMPORT_OK, f"Import failed: {_IMPORT_ERROR}"


def test_kernel_file_exists() -> None:
    """The kernel .py file is present on disk."""
    assert _KERNEL_FILE.exists(), f"Kernel file missing: {_KERNEL_FILE}"


def test_kernel_source_sha256_matches_live_file() -> None:
    """_kernel_source_sha256() == independent sha256 of the .py file."""
    assert _IMPORT_OK
    expected = hashlib.sha256(_KERNEL_FILE.read_bytes()).hexdigest()
    actual = _kernel_source_sha256()
    assert actual == expected, (
        f"kernel_source_sha256 mismatch:\nexpected {expected}\nactual   {actual}"
    )
    assert len(actual) == 64


_ACCEPTED_KERNEL_COMPILE_SIGNATURE = {
    "q_ptr": "*i8",
    "acc_ptr": "*i32",
    "accepted_ptr": "*i64",
    "accepted_dirs_ptr": "*i16",
    "accepted_thresh_ptr": "*i32",
    "n_accepted": "i32",
    "BLOCK": "constexpr",
}

_REPLAY_KERNEL_COMPILE_SIGNATURE = {
    "q_ptr": "*i8",
    "acc_ptr": "*i32",
    "replay_ptr": "*i64",
    "replay_dirs_ptr": "*i16",
    "replay_thresh_ptr": "*i32",
    "n_replay": "i32",
    "BLOCK": "constexpr",
}


def test_compile_attempt_honest_classification() -> None:
    """Compile both @triton.jit kernels via the Triton 3.6 ASTSource API.

    py_compile is always required.  A successful CPU-side compile (no launch)
    is a non-skipped B2-2b compile pass.  Only explicit GPU-target-unavailable
    environment signals may defer to B2-3; all other failures are blockers.
    """
    if not _IMPORT_OK:
        pytest.fail("Import failed — cannot assess compilation")

    py_compile.compile(str(_KERNEL_FILE), doraise=True)

    try:
        import triton
        from triton.compiler.compiler import ASTSource, GPUTarget
        from calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel import (
            _qacc_apply_accepted_pass_kernel as _acc_k,
            _qacc_apply_replay_pass_kernel as _rep_k,
        )

        target = GPUTarget(backend="cuda", arch=89, warp_size=32)
        triton.compile(
            ASTSource(
                fn=_acc_k,
                signature=_ACCEPTED_KERNEL_COMPILE_SIGNATURE,
                constexprs={"BLOCK": 128},
            ),
            target=target,
        )
        triton.compile(
            ASTSource(
                fn=_rep_k,
                signature=_REPLAY_KERNEL_COMPILE_SIGNATURE,
                constexprs={"BLOCK": 128},
            ),
            target=target,
        )
        result = "compiled_on_cpu"
    except Exception as exc:
        msg = str(exc).lower()
        defer_markers = (
            "no cuda device",
            "cuda device not available",
            "cuda is not available",
            "gpu target unavailable",
            "compute capability",
            "nvidia driver",
            "no gpu found",
        )
        if any(marker in msg for marker in defer_markers):
            result = "deferred_to_b2_3_gpu_env"
        else:
            pytest.fail(f"Unexpected compile error (not GPU-target-unavailable): {exc}")

    assert result in ("compiled_on_cpu", "deferred_to_b2_3_gpu_env")


# =============================================================================
# Pure seam: _mint_qacc_apply_native_token
# =============================================================================


def _make_tensors_for_mint_test() -> dict:
    """CPU tensors sufficient for _mint_ without a CUDA launch."""
    q = torch.tensor([[[-1, 0, 1, 0]]], dtype=torch.int8)
    acc = torch.tensor([[[0, 0, 0, 0]]], dtype=torch.int32)
    accepted_idx = torch.tensor([0, 2])
    accepted_dir = torch.tensor([1, -1], dtype=torch.int16)
    accepted_thresh = torch.tensor([3, 3], dtype=torch.int32)
    replay_idx = torch.tensor([1])
    replay_dir = torch.tensor([-1], dtype=torch.int16)
    replay_thresh = torch.tensor([2], dtype=torch.int32)
    q_out = torch.tensor([[[0, 0, 0, 0]]], dtype=torch.int8)
    acc_out = torch.tensor([[[-2, -1, 2, 0]]], dtype=torch.int32)
    return {
        "q_levels": q,
        "new_accumulators": acc,
        "accepted_indices": accepted_idx,
        "accepted_directions": accepted_dir,
        "accepted_thresholds": accepted_thresh,
        "replay_veto_indices": replay_idx,
        "replay_veto_directions": replay_dir,
        "replay_veto_thresholds": replay_thresh,
        "q_out": q_out,
        "acc_out": acc_out,
    }


def _call_mint(**overrides: Any) -> QaccApplyNativeToken:
    base = _make_tensors_for_mint_test()
    base.update(overrides)
    assert _mint_qacc_apply_native_token is not None
    return _mint_qacc_apply_native_token(**base)


def test_mint_token_produces_all_8_fields() -> None:
    """Token has exactly the 8 fields defined in QaccApplyNativeToken."""
    assert _IMPORT_OK
    tok = _call_mint()
    assert tok.kernel_family == "triton_qacc_apply"
    assert tok.kernel_symbol != ""
    assert tok.kernel_source_sha256 != ""
    assert len(tok.kernel_source_sha256) == 64
    assert tok.wrapper_launch_nonce != ""
    assert isinstance(tok.input_payload_hashes, dict)
    assert isinstance(tok.output_payload_hashes, dict)
    assert tok.backend == "cuda"
    assert tok.launch_time_ns > 0


def test_mint_input_payloads_exactly_10_keys() -> None:
    """input_payload_hashes keys == TOKEN_INPUT_PAYLOAD_KEYS (10)."""
    from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
        TOKEN_INPUT_PAYLOAD_KEYS,
    )
    assert _IMPORT_OK
    tok = _call_mint()
    assert set(tok.input_payload_hashes.keys()) == TOKEN_INPUT_PAYLOAD_KEYS


def test_mint_output_payloads_exactly_2_keys() -> None:
    """output_payload_hashes keys == TOKEN_OUTPUT_PAYLOAD_KEYS (2)."""
    from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
        TOKEN_OUTPUT_PAYLOAD_KEYS,
    )
    assert _IMPORT_OK
    tok = _call_mint()
    assert set(tok.output_payload_hashes.keys()) == TOKEN_OUTPUT_PAYLOAD_KEYS


def test_mint_input_hashes_byte_identical_to_b2_2a_helper() -> None:
    """Token input hashes match standalone B2-2a helper for same payloads."""
    assert _IMPORT_OK
    data = _make_tensors_for_mint_test()
    tok = _call_mint(**data)
    expected = hash_qacc_apply_input_payloads(
        q_levels_bytes=data["q_levels"].contiguous().numpy().tobytes(),
        new_accumulators_bytes=data["new_accumulators"].contiguous().numpy().tobytes(),
        accepted_indices_bytes=data["accepted_indices"].contiguous().numpy().tobytes(),
        accepted_directions_bytes=data["accepted_directions"].contiguous().numpy().tobytes(),
        accepted_thresholds_bytes=data["accepted_thresholds"].contiguous().numpy().tobytes(),
        replay_veto_indices_bytes=data["replay_veto_indices"].contiguous().numpy().tobytes(),
        replay_veto_directions_bytes=data["replay_veto_directions"].contiguous().numpy().tobytes(),
        replay_veto_thresholds_bytes=data["replay_veto_thresholds"].contiguous().numpy().tobytes(),
        original_accumulators_bytes=None,
        mutate_outputs=True,
    )
    assert tok.input_payload_hashes == expected


def test_mint_output_hashes_byte_identical_to_b2_2a_helper() -> None:
    """Token output hashes match standalone B2-2a helper for same payloads."""
    assert _IMPORT_OK
    data = _make_tensors_for_mint_test()
    tok = _call_mint(**data)
    expected = hash_qacc_apply_output_payloads(
        q_levels_bytes=data["q_out"].contiguous().numpy().tobytes(),
        accumulators_bytes=data["acc_out"].contiguous().numpy().tobytes(),
    )
    assert tok.output_payload_hashes == expected


def test_mint_kernel_source_sha256_matches_file() -> None:
    """Token kernel_source_sha256 == live file SHA."""
    assert _IMPORT_OK
    tok = _call_mint()
    expected = hashlib.sha256(_KERNEL_FILE.read_bytes()).hexdigest()
    assert tok.kernel_source_sha256 == expected


def test_mint_backend_cuda() -> None:
    """Token backend is exactly 'cuda'."""
    assert _IMPORT_OK
    tok = _call_mint()
    assert tok.backend == "cuda"


def test_mint_launch_time_ns_monotonic() -> None:
    """Two sequential mints have increasing launch_time_ns."""
    assert _IMPORT_OK
    import time
    tok1 = _call_mint()
    time.sleep(0.001)
    tok2 = _call_mint()
    assert tok2.launch_time_ns > tok1.launch_time_ns


def test_mint_nonce_unique_across_100_calls() -> None:
    """100 sequential nonces are all distinct."""
    assert _IMPORT_OK
    nonces = {n for n in (_call_mint().wrapper_launch_nonce for _ in range(100))}
    assert len(nonces) == 100, f"Nonce collision: {100 - len(nonces)} duplicates"


def test_mint_injected_nonce_preserved() -> None:
    """Injected nonce is preserved (not regenerated)."""
    assert _IMPORT_OK
    tok = _call_mint(wrapper_launch_nonce="injected-nonce-42")
    assert tok.wrapper_launch_nonce == "injected-nonce-42"


def test_mint_injected_launch_time_preserved() -> None:
    """Injected launch_time_ns is preserved."""
    assert _IMPORT_OK
    tok = _call_mint(launch_time_ns=777)
    assert tok.launch_time_ns == 777


# =============================================================================
# Pure seam: compare_qacc_outputs
# =============================================================================


def _q(size: int = 4) -> torch.Tensor:
    "CPU int8 q tensor."
    return torch.tensor([-1, 0, 1, 0][:size], dtype=torch.int8)


def _acc(size: int = 4) -> torch.Tensor:
    "CPU int32 acc tensor."
    return torch.tensor([0, 0, 0, 0][:size], dtype=torch.int32)


def test_compare_equal_passes_all() -> None:
    """Identical native + oracle → pass_all=True."""
    assert compare_qacc_outputs is not None
    q = _q()
    acc = _acc()
    result = compare_qacc_outputs(
        native_q=q, native_acc=acc, oracle_q=q, oracle_acc=acc
    )
    assert result["q_equal"] is True
    assert result["acc_equal"] is True
    assert result["q_hash_equal"] is True
    assert result["acc_hash_equal"] is True
    assert result["pass_all"] is True


def test_compare_q_mismatch_fails() -> None:
    """Different q → pass_all=False, q_equal=False."""
    assert compare_qacc_outputs is not None
    q1 = _q()
    q2 = q1.clone()
    q2[0] = 1
    result = compare_qacc_outputs(
        native_q=q1, native_acc=_acc(), oracle_q=q2, oracle_acc=_acc()
    )
    assert result["q_equal"] is False
    assert result["pass_all"] is False


def test_compare_acc_mismatch_fails() -> None:
    """Different acc → pass_all=False, acc_equal=False."""
    assert compare_qacc_outputs is not None
    a1 = _acc()
    a2 = a1.clone()
    a2[0] = 1
    result = compare_qacc_outputs(
        native_q=_q(), native_acc=a1, oracle_q=_q(), oracle_acc=a2
    )
    assert result["acc_equal"] is False
    assert result["pass_all"] is False


def test_compare_hash_mismatch_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash mismatch is reported when payload digests differ."""
    assert compare_qacc_outputs is not None
    counter = {"n": 0}

    def _fake_hash(_payload: bytes) -> str:
        counter["n"] += 1
        return f"hash_{counter['n']}"

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel.canonical_tensor_payload_sha256",
        _fake_hash,
    )
    result = compare_qacc_outputs(
        native_q=_q(),
        native_acc=_acc(),
        oracle_q=_q(),
        oracle_acc=_acc(),
    )
    assert result["q_equal"] is True
    assert result["q_hash_equal"] is False
    assert result["acc_hash_equal"] is False
    assert result["pass_all"] is False


# =============================================================================
# Wrapper input validation (CPU — no CUDA launch; device check on mocked CUDA)
# =============================================================================


@pytest.fixture
def cpu_trivial_inputs():
    return {
        "q_levels": torch.tensor([-1, 0, 1, 0], dtype=torch.int8),
        "new_accumulators": torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        "accepted_indices": torch.tensor([2]),
        "accepted_directions": torch.tensor([1], dtype=torch.int16),
        "accepted_thresholds": torch.tensor([3], dtype=torch.int32),
    }


def test_wrapper_invalid_q_dtype(cpu_trivial_inputs) -> None:
    """q_levels != int8 → ValueError."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    args["q_levels"] = args["q_levels"].to(torch.float32)
    with pytest.raises(ValueError, match="q_levels must be torch.int8"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_invalid_acc_dtype(cpu_trivial_inputs) -> None:
    """new_accumulators not int16/int32/int64 → ValueError."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    args["new_accumulators"] = torch.tensor([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="new_accumulators must be int16/int32/int64"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_shape_mismatch(cpu_trivial_inputs) -> None:
    """q_levels and new_accumulators shape mismatch → ValueError."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    args["new_accumulators"] = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(ValueError, match="identical shapes"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_device_mismatch(cpu_trivial_inputs) -> None:
    """q on CPU → ValueError (requires same CUDA device)."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    with pytest.raises(ValueError, match="CUDA device"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_mutate_outputs_false(cpu_trivial_inputs) -> None:
    """mutate_outputs=False → ValueError (B2-2b scope restriction)."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    args["q_levels"] = args["q_levels"].cuda() if torch.cuda.is_available() else args["q_levels"]
    args["new_accumulators"] = args["new_accumulators"].cuda() if torch.cuda.is_available() else args["new_accumulators"]
    with pytest.raises(ValueError, match="mutate_outputs=False"):
        apply_qacc_mutation_triton_native(**args, mutate_outputs=False)


def test_wrapper_negative_block_size(cpu_trivial_inputs) -> None:
    """block_size <= 0 → ValueError."""
    assert _IMPORT_OK
    args = dict(cpu_trivial_inputs)
    with pytest.raises(ValueError, match="block_size must be > 0"):
        apply_qacc_mutation_triton_native(**args, block_size=0)


# =============================================================================
# Row-validity preconditions (within-set uniqueness, index range, thresholds)
# =============================================================================


@pytest.fixture
def cuda_trivial_inputs():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        pytest.skip("CUDA not available")
    return {
        "q_levels": torch.tensor([-1, 0, 1, 0], dtype=torch.int8, device=dev),
        "new_accumulators": torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=dev),
        "accepted_indices": torch.tensor([2], device=dev),
        "accepted_directions": torch.tensor([1], dtype=torch.int16, device=dev),
        "accepted_thresholds": torch.tensor([3], dtype=torch.int32, device=dev),
    }


def test_wrapper_accepted_duplicate_rejected(cuda_trivial_inputs) -> None:
    """Duplicate accepted indices → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["accepted_indices"] = torch.tensor([2, 2], device=args["q_levels"].device)
    args["accepted_directions"] = torch.tensor([1, 1], dtype=torch.int16, device=args["q_levels"].device)
    args["accepted_thresholds"] = torch.tensor([3, 3], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="unique.*accepted"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_duplicate_rejected(cuda_trivial_inputs) -> None:
    """Duplicate replay indices → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([1, 1], device=args["q_levels"].device)
    args["replay_veto_directions"] = torch.tensor([-1, -1], dtype=torch.int16, device=args["q_levels"].device)
    args["replay_veto_thresholds"] = torch.tensor([2, 2], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="unique.*replay"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_accepted_out_of_range(cuda_trivial_inputs) -> None:
    """accepted_indices out of [0, numel) → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["accepted_indices"] = torch.tensor([99], device=args["q_levels"].device)
    with pytest.raises(ValueError, match="accepted_indices out of range"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_out_of_range(cuda_trivial_inputs) -> None:
    """replay_veto_indices out of [0, numel) → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([99], device=args["q_levels"].device)
    args["replay_veto_directions"] = torch.tensor([-1], dtype=torch.int16, device=args["q_levels"].device)
    args["replay_veto_thresholds"] = torch.tensor([2], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="replay_veto_indices out of range"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_accepted_direction_not_pm1(cuda_trivial_inputs) -> None:
    """accepted_directions not +1/-1 → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["accepted_directions"] = torch.tensor([0], dtype=torch.int16, device=args["q_levels"].device)
    with pytest.raises(ValueError, match=r"accepted_directions must be -1 or \+1"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_direction_not_pm1(cuda_trivial_inputs) -> None:
    """replay_veto_directions not +1/-1 → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([1], device=args["q_levels"].device)
    args["replay_veto_directions"] = torch.tensor([0], dtype=torch.int16, device=args["q_levels"].device)
    args["replay_veto_thresholds"] = torch.tensor([2], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match=r"replay_veto_directions must be -1 or \+1"):
        apply_qacc_mutation_triton_native(**args)


@pytest.mark.parametrize(
    ("field", "bad_tensor_factory", "match"),
    [
        (
            "accepted_indices",
            lambda dev: torch.tensor([0.0], dtype=torch.float32, device=dev),
            r"accepted_indices must be int32/int64",
        ),
        (
            "accepted_indices",
            lambda dev: torch.tensor([0], dtype=torch.int8, device=dev),
            r"accepted_indices must be int32/int64",
        ),
        (
            "accepted_directions",
            lambda dev: torch.tensor([1.0], dtype=torch.float32, device=dev),
            r"accepted_directions must be an integer tensor",
        ),
        (
            "accepted_directions",
            lambda dev: torch.tensor([True], dtype=torch.bool, device=dev),
            r"accepted_directions must be an integer tensor",
        ),
        (
            "accepted_thresholds",
            lambda dev: torch.tensor([3.0], dtype=torch.float32, device=dev),
            r"accepted_thresholds must be an integer tensor",
        ),
        (
            "accepted_thresholds",
            lambda dev: torch.tensor([3], dtype=torch.int8, device=dev),
            r"accepted_thresholds must be an integer tensor",
        ),
    ],
    ids=[
        "accepted_indices_float32",
        "accepted_indices_int8",
        "accepted_directions_float32",
        "accepted_directions_bool",
        "accepted_thresholds_float32",
        "accepted_thresholds_int8",
    ],
)
def test_wrapper_accepted_sparse_row_invalid_dtype(
    cuda_trivial_inputs,
    field: str,
    bad_tensor_factory,
    match: str,
) -> None:
    """Oracle-illegal accepted sparse-row dtypes are rejected before coercion."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args[field] = bad_tensor_factory(args["q_levels"].device)
    with pytest.raises(ValueError, match=match):
        apply_qacc_mutation_triton_native(**args)


@pytest.mark.parametrize(
    ("field", "bad_tensor_factory", "match"),
    [
        (
            "replay_veto_indices",
            lambda dev: torch.tensor([1.0], dtype=torch.float32, device=dev),
            r"replay_veto_indices must be int32/int64",
        ),
        (
            "replay_veto_indices",
            lambda dev: torch.tensor([1], dtype=torch.int8, device=dev),
            r"replay_veto_indices must be int32/int64",
        ),
        (
            "replay_veto_directions",
            lambda dev: torch.tensor([-1.0], dtype=torch.float32, device=dev),
            r"replay_veto_directions must be an integer tensor",
        ),
        (
            "replay_veto_directions",
            lambda dev: torch.tensor([True], dtype=torch.bool, device=dev),
            r"replay_veto_directions must be an integer tensor",
        ),
        (
            "replay_veto_thresholds",
            lambda dev: torch.tensor([2.0], dtype=torch.float32, device=dev),
            r"replay_veto_thresholds must be an integer tensor",
        ),
        (
            "replay_veto_thresholds",
            lambda dev: torch.tensor([2], dtype=torch.int8, device=dev),
            r"replay_veto_thresholds must be an integer tensor",
        ),
    ],
    ids=[
        "replay_indices_float32",
        "replay_indices_int8",
        "replay_directions_float32",
        "replay_directions_bool",
        "replay_thresholds_float32",
        "replay_thresholds_int8",
    ],
)
def test_wrapper_replay_sparse_row_invalid_dtype(
    cuda_trivial_inputs,
    field: str,
    bad_tensor_factory,
    match: str,
) -> None:
    """Oracle-illegal replay sparse-row dtypes are rejected before coercion."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    dev = args["q_levels"].device
    args["replay_veto_indices"] = torch.tensor([1], device=dev)
    args["replay_veto_directions"] = torch.tensor([-1], dtype=torch.int16, device=dev)
    args["replay_veto_thresholds"] = torch.tensor([2], dtype=torch.int32, device=dev)
    args[field] = bad_tensor_factory(dev)
    with pytest.raises(ValueError, match=match):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_accepted_length_mismatch(cuda_trivial_inputs) -> None:
    """accepted indices/directions/thresholds length mismatch → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["accepted_directions"] = torch.tensor([1, 1], dtype=torch.int16, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="accepted rows indices/directions/thresholds must have matching lengths"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_length_mismatch(cuda_trivial_inputs) -> None:
    """replay indices/directions/thresholds length mismatch → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([1, 1], device=args["q_levels"].device)
    args["replay_veto_directions"] = torch.tensor([-1], dtype=torch.int16, device=args["q_levels"].device)
    args["replay_veto_thresholds"] = torch.tensor([2], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="replay-veto rows indices/directions/thresholds must have matching lengths"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_partial_none_rejected(cuda_trivial_inputs) -> None:
    """Partial replay-veto tuple → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([1], device=args["q_levels"].device)
    with pytest.raises(ValueError, match="replay-veto rows require indices, directions, and thresholds"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_oversize_accepted_rejected(cuda_trivial_inputs) -> None:
    """accepted_count > _MAX_GRID_ELEMENTS → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    with patch(
        "calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel._MAX_GRID_ELEMENTS",
        2,
    ):
        args["accepted_indices"] = torch.tensor([0, 1, 2], device=args["q_levels"].device)
        args["accepted_directions"] = torch.tensor([1, 1, 1], dtype=torch.int16, device=args["q_levels"].device)
        args["accepted_thresholds"] = torch.tensor([3, 3, 3], dtype=torch.int32, device=args["q_levels"].device)
        with pytest.raises(ValueError, match="accepted_count 3 exceeds max grid elements 2"):
            apply_qacc_mutation_triton_native(**args)


def test_wrapper_oversize_replay_rejected(cuda_trivial_inputs) -> None:
    """replay_count > _MAX_GRID_ELEMENTS → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    with patch(
        "calm.hrm_text_158.native_full_stack.qacc_apply_triton_kernel._MAX_GRID_ELEMENTS",
        1,
    ):
        args["replay_veto_indices"] = torch.tensor([0, 1], device=args["q_levels"].device)
        args["replay_veto_directions"] = torch.tensor([-1, -1], dtype=torch.int16, device=args["q_levels"].device)
        args["replay_veto_thresholds"] = torch.tensor([2, 2], dtype=torch.int32, device=args["q_levels"].device)
        with pytest.raises(ValueError, match="replay_count 2 exceeds max grid elements 1"):
            apply_qacc_mutation_triton_native(**args)


def test_wrapper_accepted_threshold_not_positive(cuda_trivial_inputs) -> None:
    """accepted_thresholds <= 0 → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["accepted_thresholds"] = torch.tensor([0], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="accepted_thresholds must be > 0"):
        apply_qacc_mutation_triton_native(**args)


def test_wrapper_replay_threshold_not_positive(cuda_trivial_inputs) -> None:
    """replay_veto_thresholds <= 0 → ValueError."""
    assert _IMPORT_OK
    args = dict(cuda_trivial_inputs)
    args["replay_veto_indices"] = torch.tensor([1], device=args["q_levels"].device)
    args["replay_veto_directions"] = torch.tensor([-1], dtype=torch.int16, device=args["q_levels"].device)
    args["replay_veto_thresholds"] = torch.tensor([0], dtype=torch.int32, device=args["q_levels"].device)
    with pytest.raises(ValueError, match="replay_veto_thresholds must be > 0"):
        apply_qacc_mutation_triton_native(**args)


# =============================================================================
# Oversize guard
# =============================================================================


def test_oversize_constant_is_large() -> None:
    """_MAX_GRID_ELEMENTS is a sane large value."""
    assert _IMPORT_OK
    assert _MAX_GRID_ELEMENTS > 10**8


def test_oversize_constant_reachable_type(cuda_trivial_inputs) -> None:
    """The guard variable is importable and non-zero."""
    assert _IMPORT_OK
    assert isinstance(_MAX_GRID_ELEMENTS, int)


# =============================================================================
# Accumulator int32 intermediate guard (source-level, no GPU launch)
# =============================================================================


def test_kernel_accumulator_stores_preserve_int32_intermediate() -> None:
    """acc_ptr stores must keep int32 width across accepted→replay (oracle parity).

    Premature int16 narrowing between passes would break accepted∩replay overlap
    rows when clamped residuals exceed int16 range.  B2-3 must still prove runtime
    parity with threshold > 32768 overlap fixtures.
    """
    source = _KERNEL_FILE.read_text(encoding="utf-8")
    assert "acc_new.to(tl.int16)" not in source, (
        "accumulator stores must not narrow to int16 between passes"
    )
    acc_store_lines = [
        line.strip()
        for line in source.splitlines()
        if "tl.store(acc_ptr" in line
    ]
    assert len(acc_store_lines) == 2, (
        f"expected exactly 2 acc_ptr store sites, found {len(acc_store_lines)}"
    )
    for line in acc_store_lines:
        assert "acc_new.to(tl.int32)" in line, (
            f"acc_ptr store must use int32 width: {line!r}"
        )


# =============================================================================
# Sequential pass ordering seam
# =============================================================================


def test_kernel_handles_none() -> None:
    """Both JIT handles are either non-None or None together."""
    assert _IMPORT_OK
    assert (
        (_qacc_apply_accepted_pass_kernel is not None and _qacc_apply_replay_pass_kernel is not None)
        or (_qacc_apply_accepted_pass_kernel is None and _qacc_apply_replay_pass_kernel is None)
    )


def test_kernel_symbol_names_ordered_in_token() -> None:
    """Token kernel_symbol lists accepted before replay (oracle ordering)."""
    assert _IMPORT_OK
    tok = _call_mint()
    sym = tok.kernel_symbol
    assert sym.startswith("_qacc_apply_accepted_pass_kernel")
    assert "_qacc_apply_replay_pass_kernel" in sym


# =============================================================================
# Anti-launder: B2-2b does NOT mint pass / call builder / call validator
# =============================================================================


def test_wrapper_does_not_return_receipt() -> None:
    """apply_qacc_mutation_triton_native returns a 3-tuple, not a receipt."""
    assert _IMPORT_OK
    sig = apply_qacc_mutation_triton_native.__code__.co_names
    assert "QaccApplyNativeParityReceipt" not in sig


def test_mint_does_not_call_builder_or_validator() -> None:
    """_mint_ has no receipt/builder/validator references."""
    assert _IMPORT_OK
    import inspect
    src = inspect.getsource(_mint_qacc_apply_native_token)
    assert "build_qacc_apply_native_parity_receipt" not in src
    assert "validate_qacc_apply_native_parity_receipt" not in src
    assert "parity_pass" not in src.lower()
    assert "gpu_command_satisfied" not in src.lower()


def test_compare_has_no_receipt_logic() -> None:
    """compare_qacc_outputs has no receipt logic."""
    assert _IMPORT_OK
    import inspect
    src = inspect.getsource(compare_qacc_outputs)
    assert "receipt" not in src.lower()
    assert "parity_pass" not in src.lower()
    assert "gpu_command_satisfied" not in src.lower()


# =============================================================================
# Byte-level import: B1/B2-2a unmutated
# =============================================================================


def test_b1_b2_2a_unmutated() -> None:
    """B1 + B2-2a files have their committed SHA-256."""
    b1 = Path(__file__).parent.parent.parent / "hrm_text_158" / "native_full_stack" / "qacc_apply_parity_receipt.py"
    b2 = Path(__file__).parent.parent.parent / "hrm_text_158" / "native_full_stack" / "qacc_apply_native_parity_receipt.py"
    assert hashlib.sha256(b1.read_bytes()).hexdigest() == "99d612ce0810f2a592be99b71616473735caadc979b7f9f12b19cb63c72a8b59"
    assert hashlib.sha256(b2.read_bytes()).hexdigest() == "e07d9a380f8279e9ac5fe5fa0c0d3bdc46110ac6528112de17dd6365e6ff0996"
