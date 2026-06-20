"""B2-2b native Triton q_acc_apply kernel + wrapper.

Two sequential count-grid kernels (accepted-pass then replay-pass) for
race-free sparse q/acc mutation.  Pure non-launch seams for CPU testability.
Does NOT mint receipts, does NOT set parity_pass/gpu_command_satisfied.
Reuses B2-2a hash helpers (hash_qacc_apply_{input,output}_payloads).
"""  # noqa: E501
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

from calm.hrm_text_158.native_full_stack.qacc_apply_native_parity_receipt import (
    QaccApplyNativeToken,
    canonical_tensor_payload_sha256,
    hash_qacc_apply_input_payloads,
    hash_qacc_apply_output_payloads,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Kernel source SHA (computed once per import, file-not-in-memory drift guard)
# ---------------------------------------------------------------------------

_KERNEL_FILE = Path(__file__)
_kernel_file_path_for_test = _KERNEL_FILE


def _kernel_source_sha256() -> str:
    """SHA-256 of this .py file (kernel source drift guard)."""
    return hashlib.sha256(_KERNEL_FILE.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# @triton.jit  count-grid accepted-pass kernel
# ---------------------------------------------------------------------------
# Grid: 1D over accepted_count.  Each program = BLOCK consecutive accepted
# indices.  Within a program, each lane handles one accepted element.
#
# q flip:  clamp(q + direction, -1, 1)
# acc residual: clamp(acc - direction*threshold, -(threshold-1), threshold-1)
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _qacc_apply_accepted_pass_kernel(
        q_ptr,
        acc_ptr,
        accepted_ptr,
        accepted_dirs_ptr,
        accepted_thresh_ptr,
        n_accepted,
        BLOCK: tl.constexpr = 128,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_accepted

        # Load accepted element index
        idx = tl.load(accepted_ptr + offs, mask=mask, other=0).to(tl.int64)

        # Load q and acc at that element
        q = tl.load(q_ptr + idx, mask=mask, other=0).to(tl.int16)
        acc = tl.load(acc_ptr + idx, mask=mask, other=0).to(tl.int32)

        # Load direction and threshold
        dir_ = tl.load(accepted_dirs_ptr + offs, mask=mask, other=0).to(tl.int32)
        thresh = tl.load(accepted_thresh_ptr + offs, mask=mask, other=0).to(tl.int32)

        # q flip
        q_new = q + dir_.to(tl.int16)
        q_new = tl.where(q_new > 1, 1, q_new)
        q_new = tl.where(q_new < -1, -1, q_new)

        # acc residual
        residual = acc - dir_ * thresh
        lo = -(thresh - 1)
        hi = thresh - 1
        acc_new = tl.where(residual > hi, hi, residual)
        acc_new = tl.where(acc_new < lo, lo, acc_new)

        # Store back (in-place mutation)
        tl.store(q_ptr + idx, q_new.to(tl.int8), mask=mask)
        tl.store(acc_ptr + idx, acc_new.to(tl.int32), mask=mask)

    # -----------------------------------------------------------------------
    # @triton.jit  count-grid replay-pass kernel
    # -----------------------------------------------------------------------
    # Grid: 1D over replay_count.  Each program = BLOCK consecutive replay
    # indices.  q UNCHANGED; acc residual only.
    # -----------------------------------------------------------------------

    @triton.jit
    def _qacc_apply_replay_pass_kernel(
        q_ptr,
        acc_ptr,
        replay_ptr,
        replay_dirs_ptr,
        replay_thresh_ptr,
        n_replay,
        BLOCK: tl.constexpr = 128,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_replay

        # Load replay element index
        idx = tl.load(replay_ptr + offs, mask=mask, other=0).to(tl.int64)

        # Load acc at that element (q is NOT modified in replay)
        acc = tl.load(acc_ptr + idx, mask=mask, other=0).to(tl.int32)

        # Load direction and threshold
        dir_ = tl.load(replay_dirs_ptr + offs, mask=mask, other=0).to(tl.int32)
        thresh = tl.load(replay_thresh_ptr + offs, mask=mask, other=0).to(tl.int32)

        # acc residual only
        residual = acc - dir_ * thresh
        lo = -(thresh - 1)
        hi = thresh - 1
        acc_new = tl.where(residual > hi, hi, residual)
        acc_new = tl.where(acc_new < lo, lo, acc_new)

        # Store back (in-place, q untouched)
        tl.store(acc_ptr + idx, acc_new.to(tl.int32), mask=mask)

else:
    _qacc_apply_accepted_pass_kernel = None  # type: ignore[misc,assignment]
    _qacc_apply_replay_pass_kernel = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Pure non-launch seam: token minting
# ---------------------------------------------------------------------------


def _mint_qacc_apply_native_token(
    *,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    replay_veto_indices: torch.Tensor | None = None,
    replay_veto_directions: torch.Tensor | None = None,
    replay_veto_thresholds: torch.Tensor | None = None,
    original_accumulators: torch.Tensor | None = None,
    q_out: torch.Tensor,
    acc_out: torch.Tensor,
    wrapper_launch_nonce: str | None = None,
    launch_time_ns: int | None = None,
) -> QaccApplyNativeToken:
    """Mint a QaccApplyNativeToken by reusing the B2-2a hash helpers.

    All payload hashes are computed from the raw contiguous tensor bytes.
    This helper is PURE (no CUDA launch).
    """
    # Input hashes via B2-2a helpers
    input_hashes = hash_qacc_apply_input_payloads(
        q_levels_bytes=q_levels.contiguous().cpu().numpy().tobytes(),
        new_accumulators_bytes=new_accumulators.contiguous().cpu().numpy().tobytes(),
        accepted_indices_bytes=accepted_indices.contiguous().cpu().numpy().tobytes(),
        accepted_directions_bytes=accepted_directions.contiguous().cpu().numpy().tobytes(),
        accepted_thresholds_bytes=accepted_thresholds.contiguous().cpu().numpy().tobytes(),
        replay_veto_indices_bytes=(
            replay_veto_indices.contiguous().cpu().numpy().tobytes()
            if replay_veto_indices is not None
            else None
        ),
        replay_veto_directions_bytes=(
            replay_veto_directions.contiguous().cpu().numpy().tobytes()
            if replay_veto_directions is not None
            else None
        ),
        replay_veto_thresholds_bytes=(
            replay_veto_thresholds.contiguous().cpu().numpy().tobytes()
            if replay_veto_thresholds is not None
            else None
        ),
        original_accumulators_bytes=(
            original_accumulators.contiguous().cpu().numpy().tobytes()
            if original_accumulators is not None
            else None
        ),
        mutate_outputs=True,
    )

    # Output hashes via B2-2a helper
    output_hashes = hash_qacc_apply_output_payloads(
        q_levels_bytes=q_out.contiguous().cpu().numpy().tobytes(),
        accumulators_bytes=acc_out.contiguous().cpu().numpy().tobytes(),
    )

    return QaccApplyNativeToken(
        kernel_family="triton_qacc_apply",
        kernel_symbol="_qacc_apply_accepted_pass_kernel+_qacc_apply_replay_pass_kernel",
        kernel_source_sha256=_kernel_source_sha256(),
        wrapper_launch_nonce=wrapper_launch_nonce or str(uuid.uuid4()),
        input_payload_hashes=input_hashes,
        output_payload_hashes=output_hashes,
        backend="cuda",
        launch_time_ns=launch_time_ns or time.monotonic_ns(),
    )


# ---------------------------------------------------------------------------
# Pure non-launch seam: output comparison
# ---------------------------------------------------------------------------


def compare_qacc_outputs(
    *,
    native_q: torch.Tensor,
    native_acc: torch.Tensor,
    oracle_q: torch.Tensor,
    oracle_acc: torch.Tensor,
) -> dict[str, Any]:
    """Compare native (kernel) outputs against oracle (reference) outputs.

    Returns a result dict with:
        q_equal (bool), acc_equal (bool),
        q_hash_equal (bool), acc_hash_equal (bool),
        pass_all (bool)
    """
    q_equal = torch.equal(native_q, oracle_q)
    acc_equal = torch.equal(native_acc, oracle_acc)

    q_hash_equal = canonical_tensor_payload_sha256(
        native_q.contiguous().cpu().numpy().tobytes()
    ) == canonical_tensor_payload_sha256(
        oracle_q.contiguous().cpu().numpy().tobytes()
    )
    acc_hash_equal = canonical_tensor_payload_sha256(
        native_acc.contiguous().cpu().numpy().tobytes()
    ) == canonical_tensor_payload_sha256(
        oracle_acc.contiguous().cpu().numpy().tobytes()
    )

    return {
        "q_equal": bool(q_equal),
        "acc_equal": bool(acc_equal),
        "q_hash_equal": q_hash_equal,
        "acc_hash_equal": acc_hash_equal,
        "pass_all": bool(q_equal and acc_equal and q_hash_equal and acc_hash_equal),
    }


# ---------------------------------------------------------------------------
# Oversize guard constant
# ---------------------------------------------------------------------------

_MAX_GRID_ELEMENTS = 2**31 - 1  # Triton grid limit (practical)


def _validate_row_lengths(
    *,
    name: str,
    indices: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    if indices.numel() != directions.numel() or indices.numel() != thresholds.numel():
        raise ValueError(f"{name} indices/directions/thresholds must have matching lengths")


def _validate_indices_dtype(name: str, values: torch.Tensor) -> None:
    if values.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32/int64, got {values.dtype}")


def _validate_directions_dtype(name: str, values: torch.Tensor) -> None:
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor, got {values.dtype}")


def _validate_thresholds_dtype(name: str, values: torch.Tensor) -> None:
    if values.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor, got {values.dtype}")


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def apply_qacc_mutation_triton_native(
    *,
    q_levels: torch.Tensor,
    new_accumulators: torch.Tensor,
    accepted_indices: torch.Tensor,
    accepted_directions: torch.Tensor,
    accepted_thresholds: torch.Tensor,
    replay_veto_indices: torch.Tensor | None = None,
    replay_veto_directions: torch.Tensor | None = None,
    replay_veto_thresholds: torch.Tensor | None = None,
    original_accumulators: torch.Tensor | None = None,
    mutate_outputs: bool = True,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, QaccApplyNativeToken]:
    """Native Triton q_acc_apply mutation via two sequential count-grid kernels.

    Returns ``(q_out, acc_out, token)``.  Does NOT mint a receipt.
    Does NOT set parity_pass or gpu_command_satisfied.

    Accepts same interface as
    ``q_acc_apply_mutation_torch_cuda_reference_under_cap_rows``
    (vote_update.py:1200).
    """
    if _qacc_apply_accepted_pass_kernel is None:
        raise RuntimeError("Triton is not available")

    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    # ---- input validation (mirrors vote_update.py:1213-1228) ----
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    if new_accumulators.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError(
            f"new_accumulators must be int16/int32/int64, got {new_accumulators.dtype}"
        )
    if q_levels.shape != new_accumulators.shape:
        raise ValueError("q_levels and new_accumulators must have identical shapes")
    if q_levels.device.type != "cuda" or new_accumulators.device != q_levels.device:
        raise ValueError(
            "q_acc apply native requires q/new_acc tensors on the same CUDA device"
        )
    if not bool(mutate_outputs):
        if original_accumulators is None:
            raise ValueError("original_accumulators is required when mutate_outputs=False")
        if original_accumulators.dtype != torch.int16:
            raise ValueError(
                f"original_accumulators must be torch.int16, got {original_accumulators.dtype}"
            )
        if original_accumulators.shape != q_levels.shape:
            raise ValueError("original_accumulators shape must match q_levels")

    # B2-2b scope: mutate_outputs=True only (B2-1 §6)
    if not mutate_outputs:
        raise ValueError(
            "B2-2b wrapper does not support mutate_outputs=False; "
            "use the torch-CUDA reference path instead"
        )

    device = q_levels.device
    numel = int(q_levels.numel())

    _validate_indices_dtype("accepted_indices", accepted_indices)
    _validate_directions_dtype("accepted_directions", accepted_directions)
    _validate_thresholds_dtype("accepted_thresholds", accepted_thresholds)

    # Flatten and coerce accepted rows (mirrors vote_update.py coercion)
    accepted = accepted_indices.flatten().to(device=device, dtype=torch.int64).contiguous()
    accepted_dirs = accepted_directions.flatten().to(device=device, dtype=torch.int16).contiguous()
    accepted_thresh = accepted_thresholds.flatten().to(device=device, dtype=torch.int32).contiguous()

    _validate_row_lengths(
        name="accepted rows",
        indices=accepted,
        directions=accepted_dirs,
        thresholds=accepted_thresh,
    )

    # Coerce/validate directions
    if accepted_dirs.numel() > 0:
        invalid = (accepted_dirs != 1) & (accepted_dirs != -1)
        if bool(invalid.any().item()):
            raise ValueError("accepted_directions must be -1 or +1")

    # Coerce/validate thresholds
    if accepted_thresh.numel() > 0:
        if bool((accepted_thresh <= 0).any().item()):
            raise ValueError("accepted_thresholds must be > 0")

    # Validate accepted indices in range
    if accepted.numel() > 0:
        if not ((accepted >= 0) & (accepted < numel)).all().item():
            raise ValueError("accepted_indices out of range")

        if len(accepted.unique()) != accepted.numel():
            raise ValueError("accepted_indices must be unique within the accepted set")
    # Flatten replay rows
    replay_parts = (
        replay_veto_indices,
        replay_veto_directions,
        replay_veto_thresholds,
    )
    if any(part is not None for part in replay_parts):
        if any(part is None for part in replay_parts):
            raise ValueError(
                "replay-veto rows require indices, directions, and thresholds"
            )
        _validate_indices_dtype("replay_veto_indices", replay_veto_indices)
        _validate_directions_dtype("replay_veto_directions", replay_veto_directions)
        _validate_thresholds_dtype("replay_veto_thresholds", replay_veto_thresholds)
        replay = replay_veto_indices.flatten().to(device=device, dtype=torch.int64).contiguous()
        replay_dirs = replay_veto_directions.flatten().to(device=device, dtype=torch.int16).contiguous()
        replay_thresh = replay_veto_thresholds.flatten().to(device=device, dtype=torch.int32).contiguous()
    else:
        replay = torch.empty(0, dtype=torch.int64, device=device)
        replay_dirs = torch.empty(0, dtype=torch.int16, device=device)
        replay_thresh = torch.empty(0, dtype=torch.int32, device=device)

    _validate_row_lengths(
        name="replay-veto rows",
        indices=replay,
        directions=replay_dirs,
        thresholds=replay_thresh,
    )

    if replay.numel() > 0:
        if replay_dirs.numel() > 0:
            invalid = (replay_dirs != 1) & (replay_dirs != -1)
            if bool(invalid.any().item()):
                raise ValueError("replay_veto_directions must be -1 or +1")
        if replay_thresh.numel() > 0:
            if bool((replay_thresh <= 0).any().item()):
                raise ValueError("replay_veto_thresholds must be > 0")
        if not ((replay >= 0) & (replay < numel)).all().item():
            raise ValueError("replay_veto_indices out of range")
        if len(replay.unique()) != replay.numel():
            raise ValueError("replay_veto_indices must be unique within the replay set")

    # Oversize grid guard
    if accepted.numel() > _MAX_GRID_ELEMENTS:
        raise ValueError(
            f"accepted_count {accepted.numel()} exceeds max grid elements {_MAX_GRID_ELEMENTS}"
        )
    if replay.numel() > _MAX_GRID_ELEMENTS:
        raise ValueError(
            f"replay_count {replay.numel()} exceeds max grid elements {_MAX_GRID_ELEMENTS}"
        )

    # Prepare in-place output tensors (clones, since native mutates)
    q_work = q_levels.detach().clone().contiguous()
    acc_work = new_accumulators.flatten().to(torch.int32).detach().clone().contiguous()

    # ---- Launch accepted pass FIRST ----
    if accepted.numel() > 0:
        grid = (triton.cdiv(int(accepted.numel()), block_size),)
        _qacc_apply_accepted_pass_kernel[grid](
            q_work,
            acc_work,
            accepted,
            accepted_dirs,
            accepted_thresh,
            int(accepted.numel()),
            BLOCK=block_size,
        )

    # ---- Launch replay pass SECOND (same stream, sequential) ----
    if replay.numel() > 0:
        grid = (triton.cdiv(int(replay.numel()), block_size),)
        _qacc_apply_replay_pass_kernel[grid](
            q_work,
            acc_work,
            replay,
            replay_dirs,
            replay_thresh,
            int(replay.numel()),
            BLOCK=block_size,
        )

    # Cast outputs to target dtypes
    q_out = q_work.view_as(q_levels)
    acc_out = acc_work.view_as(new_accumulators).to(torch.int16)

    # Mint token (CPU, after kernel returns)
    token = _mint_qacc_apply_native_token(
        q_levels=q_levels,
        new_accumulators=new_accumulators,
        accepted_indices=accepted_indices,
        accepted_directions=accepted_directions,
        accepted_thresholds=accepted_thresholds,
        replay_veto_indices=replay_veto_indices,
        replay_veto_directions=replay_veto_directions,
        replay_veto_thresholds=replay_veto_thresholds,
        original_accumulators=original_accumulators,
        q_out=q_out,
        acc_out=acc_out,
    )

    return q_out, acc_out, token
