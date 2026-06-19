"""Native Triton integer credit-axis GPU kernel pipeline (BR-3C-H.1b v3.1)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt import (
    BR_H_GPU_DISPATCH_HELD,
    BR_H_GPU_KERNEL_MISSING,
    BR_H_NOT_KERNELIZED,
    RUN_GPU_CREDIT_AXIS_KERNEL_ENV,
    CreditAxisKernelBoundaryGuard,
    CreditAxisStageNativeEvidence,
    classify_credit_axis_gpu_prelaunch_branch,
    run_gpu_credit_axis_kernel_env_enabled,
    torch_cuda_reference_only_from_stage_evidence,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    GRAD_Q16_SCALE,
    INPUT_Q15_SCALE,
    INT32_MIN,
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    _attribution_rescale_shift_for_law,
    _quantize_to_int32,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1,
    CanonicalRankVoteBin,
    grouped_bisect_right_rank_positions_integer_abs,
    integer_abs_magnitude_i64,
    integer_rank_bin_bounds,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

CREDIT_AXIS_KERNEL_SEAM_NAME = "credit_axis_kernelized_sparse_pipeline_cuda"

S1_TILE_CAP = 256
S1_SUPPORTED_MAX = {
    "out_features": 1024,
    "in_features": 4096,
    "n_capture_pairs": 16,
    "batch": 64,
    "sequence": 64,
}
S1_EVENT_CAP = 1_048_576
S4_NATIVE_MAX = 1024
COMPACT_PREFIX_MAX = S1_EVENT_CAP

CREDIT_AXIS_KERNEL_MANIFEST_DIR = Path(__file__).resolve().parent / "credit_axis_kernel_manifest"

# Default CLEAN seam body (AST forbid-tests grep this slice only).
_DEFAULT_PIPELINE_SOURCE_SLICE_START = "def _run_integer_pipeline_cuda"
_DEFAULT_PIPELINE_FORBIDDEN_SYMBOLS = (
    "projected_moves_from_integer_attribution",
    "index_map",
    "masked_select",
    "nonzero",
    ".tolist(",
)


class CreditAxisKernelNotAvailable(RuntimeError):
    """Raised when the credit-axis GPU kernel cannot run."""


class CreditAxisShapeExceedsSupportedMax(CreditAxisKernelNotAvailable):
    """Raised when launch shape exceeds pre-registered bounds."""


@dataclass(frozen=True)
class CreditAxisKernelizedPipelineResult:
    flat_indices: torch.Tensor
    attribution_q31: torch.Tensor
    projected_move_indices: torch.Tensor
    projected_moves: torch.Tensor
    credit_q31: torch.Tensor
    sparse_vote_indices: torch.Tensor
    sparse_vote_values: torch.Tensor
    torch_cuda_reference_only: bool = False
    stage_native_evidence: CreditAxisStageNativeEvidence | None = None


def dense_compact_prefix_scan_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Python reference for §1.7.1 compaction contract tests (CPU, no launch)."""
    if keys.shape != values.shape or keys.dim() != 1:
        raise ValueError("keys and values must be same-shape 1-D tensors")
    keep = values != 0
    n = int(values.numel())
    exclusive_pos = torch.zeros(n, dtype=torch.int64)
    count = 0
    for i in range(n):
        exclusive_pos[i] = count
        if bool(keep[i].item()):
            count += 1
    if count == 0:
        return (
            torch.empty(0, dtype=keys.dtype),
            torch.empty(0, dtype=values.dtype),
            0,
        )
    out_keys = keys[keep].contiguous()
    out_values = values[keep].contiguous()
    return out_keys, out_values, count


def s1_row_major_compact_reference(
    row_attrs: list[torch.Tensor],
    *,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for S1 row-major compaction contract (§1.7.2)."""
    flat_parts: list[torch.Tensor] = []
    attr_parts: list[torch.Tensor] = []
    for row_index, attr_o in enumerate(row_attrs):
        keys, vals, _ = dense_compact_prefix_scan_reference(
            torch.arange(in_features, dtype=torch.int64) + row_index * in_features,
            attr_o.to(torch.int32),
        )
        if int(keys.numel()) > 0:
            flat_parts.append(keys)
            attr_parts.append(vals)
    if not flat_parts:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32)
    return torch.cat(flat_parts), torch.cat(attr_parts)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def credit_axis_kernel_module_built() -> bool:
    return _credit_axis_attribution_row_tile_kernel is not None


def classify_credit_axis_gpu_kernel_prelaunch_from_environment() -> str:
    return classify_credit_axis_gpu_prelaunch_branch(
        triton_available=_triton_available(),
        cuda_available=_cuda_available(),
        kernel_module_built=credit_axis_kernel_module_built(),
        seam_resolves_to_credit_axis_kernel=True,
        dispatch_env_enabled=run_gpu_credit_axis_kernel_env_enabled(),
    )


def _check_s1_supported_max(
    *,
    out_features: int,
    in_features: int,
    n_capture_pairs: int,
    batch: int,
    sequence: int,
) -> None:
    if out_features > S1_SUPPORTED_MAX["out_features"]:
        raise CreditAxisShapeExceedsSupportedMax("shape_exceeds_s1_supported_max:out_features")
    if in_features > S1_SUPPORTED_MAX["in_features"]:
        raise CreditAxisShapeExceedsSupportedMax("shape_exceeds_s1_supported_max:in_features")
    if n_capture_pairs > S1_SUPPORTED_MAX["n_capture_pairs"]:
        raise CreditAxisShapeExceedsSupportedMax("shape_exceeds_s1_supported_max:n_capture_pairs")
    if batch > S1_SUPPORTED_MAX["batch"]:
        raise CreditAxisShapeExceedsSupportedMax("shape_exceeds_s1_supported_max:batch")
    if sequence > S1_SUPPORTED_MAX["sequence"]:
        raise CreditAxisShapeExceedsSupportedMax("shape_exceeds_s1_supported_max:sequence")


def _as_bsi(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"{name} must have batch dimension")
    return tensor


def _boundary_quantize_captures(
    capture_inputs: Sequence[torch.Tensor],
    capture_grad_outputs: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    paired_inputs = list(capture_inputs[-len(capture_grad_outputs) :])
    grad_outputs_reversed = list(reversed(list(capture_grad_outputs)))
    input_q15_list: list[torch.Tensor] = []
    grad_q16_list: list[torch.Tensor] = []
    max_seq = 1
    for inp, grad_out in zip(paired_inputs, grad_outputs_reversed):
        input_bsi = _as_bsi(inp.detach().to(torch.float32), name="input")
        grad_out_bso = _as_bsi(grad_out.detach().to(torch.float32), name="grad_out")
        seq = int(input_bsi.shape[1]) if input_bsi.dim() > 2 else 1
        max_seq = max(max_seq, seq)
        input_q15_list.append(
            _quantize_to_int32(input_bsi, scale=INPUT_Q15_SCALE).to(device=device)
        )
        grad_q16_list.append(
            _quantize_to_int32(grad_out_bso, scale=GRAD_Q16_SCALE).to(device=device)
        )
    return input_q15_list, grad_q16_list, max_seq


if triton is not None:

    @triton.jit
    def _credit_axis_attribution_row_tile_kernel(
        INPUT_PTR,
        GRAD_PTR,
        ACC_PTR,
        ROW_INDEX,
        B,
        S,
        I,
        TILE_START,
        TILE_LEN,
        INPUT_STRIDE_B,
        INPUT_STRIDE_S,
        INPUT_STRIDE_I,
        GRAD_STRIDE_B,
        GRAD_STRIDE_S,
        GRAD_STRIDE_O,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < TILE_LEN
        acc = tl.zeros((BLOCK,), dtype=tl.int64)
        for b in range(B):
            for s in range(S):
                grad_scalar = tl.load(
                    GRAD_PTR
                    + b * GRAD_STRIDE_B
                    + s * GRAD_STRIDE_S
                    + ROW_INDEX * GRAD_STRIDE_O
                ).to(tl.int64)
                input_offs = (
                    b * INPUT_STRIDE_B
                    + s * INPUT_STRIDE_S
                    + TILE_START * INPUT_STRIDE_I
                    + offs * INPUT_STRIDE_I
                )
                input_vals = tl.load(INPUT_PTR + input_offs, mask=mask, other=0).to(tl.int64)
                acc += grad_scalar * input_vals
        tl.store(ACC_PTR + TILE_START + offs, acc, mask=mask)

    @triton.jit
    def _credit_axis_attribution_rescale_kernel(
        ACC_PTR,
        OUT_PTR,
        I,
        SHIFT: tl.constexpr,
        HALF: tl.constexpr,
    ):
        for i in range(I):
            val = tl.load(ACC_PTR + i).to(tl.int64)
            positive = val >= 0
            abs_val = tl.where(val < 0, -val, val)
            rounded = (abs_val + HALF) >> SHIFT
            rescaled = tl.where(positive, rounded, -rounded)
            tl.store(OUT_PTR + i, rescaled.to(tl.int32))

    @triton.jit
    def _credit_axis_row_nz_count_kernel(ROW_ATTR_PTR, ROW_NZ_PTR, I, ROW_INDEX):
        count = 0
        base = ROW_INDEX * I
        for i in range(I):
            if tl.load(ROW_ATTR_PTR + base + i) != 0:
                count += 1
        tl.store(ROW_NZ_PTR + ROW_INDEX, count)

    @triton.jit
    def _credit_axis_prefix_sum_exclusive_kernel(IN_PTR, OUT_PTR, N, MAX_N: tl.constexpr):
        running = 0
        for i in range(MAX_N):
            if i < N:
                tl.store(OUT_PTR + i, running)
                running += tl.load(IN_PTR + i)

    @triton.jit
    def _credit_axis_row_compact_scatter_kernel(
        ROW_ATTR_PTR,
        ROW_BASE_PTR,
        OUT_FLAT_PTR,
        OUT_ATTR_PTR,
        I,
        ROW_INDEX,
        IN_FEATURES: tl.constexpr,
    ):
        row_base = tl.load(ROW_BASE_PTR + ROW_INDEX)
        local_pos = 0
        global_base = ROW_INDEX * IN_FEATURES
        for i in range(IN_FEATURES):
            val = tl.load(ROW_ATTR_PTR + global_base + i)
            if val != 0:
                j = row_base + local_pos
                tl.store(OUT_FLAT_PTR + j, global_base + i)
                tl.store(OUT_ATTR_PTR + j, val)
                local_pos += 1

    @triton.jit
    def _credit_axis_project_moves_dense_kernel(
        FLAT_INDICES_PTR,
        ATTRIBUTION_PTR,
        Q_LEVELS_PTR,
        MOVE_DENSE_PTR,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        flat_idx = tl.load(FLAT_INDICES_PTR + offs, mask=mask, other=0).to(tl.int64)
        attr = tl.load(ATTRIBUTION_PTR + offs, mask=mask, other=0).to(tl.int32)
        q = tl.load(Q_LEVELS_PTR + flat_idx, mask=mask, other=0).to(tl.int8)
        move = tl.zeros((BLOCK,), dtype=tl.int8)
        q_neg = q < 0
        q_zero = q == 0
        q_pos = q > 0
        attr_neg = attr < 0
        attr_pos = attr > 0
        move = tl.where(q_neg & attr_neg, 1, move)
        move = tl.where(q_zero & attr_neg, 1, move)
        move = tl.where(q_zero & attr_pos, -1, move)
        move = tl.where(q_pos & attr_pos, -1, move)
        tl.store(MOVE_DENSE_PTR + offs, move, mask=mask)

    @triton.jit
    def _credit_axis_dense_compact_scatter_kernel(
        KEY_PTR,
        VALUE_PTR,
        OUT_KEY_PTR,
        OUT_VALUE_PTR,
        N,
        MAX_N: tl.constexpr,
    ):
        running = 0
        for i in range(MAX_N):
            if i < N:
                val = tl.load(VALUE_PTR + i)
                if val != 0:
                    tl.store(OUT_KEY_PTR + running, tl.load(KEY_PTR + i))
                    tl.store(OUT_VALUE_PTR + running, val)
                    running += 1

    @triton.jit
    def _credit_axis_gather_attribution_kernel(
        EVENT_FLAT_PTR,
        EVENT_ATTR_PTR,
        N_EVENTS,
        MOVE_FLAT_PTR,
        OUT_ATTR_PTR,
        N_MOVES,
        MAX_EVENTS: tl.constexpr,
    ):
        for i in range(N_MOVES):
            target = tl.load(MOVE_FLAT_PTR + i)
            attr = 0
            for j in range(MAX_EVENTS):
                if j < N_EVENTS:
                    if tl.load(EVENT_FLAT_PTR + j) == target:
                        attr = tl.load(EVENT_ATTR_PTR + j)
            tl.store(OUT_ATTR_PTR + i, attr)

    @triton.jit
    def _credit_axis_credit_q31_kernel(
        ATTRIBUTION_PTR,
        OUT_PTR,
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        attr = tl.load(ATTRIBUTION_PTR + offs, mask=mask, other=0).to(tl.int32)
        tl.store(OUT_PTR + offs, (-attr).to(tl.int32), mask=mask)

    @triton.jit
    def _credit_axis_grouped_bisect_right_rank_kernel(
        ABS_PTR,
        RANK_PTR,
        N,
        MAX_N: tl.constexpr,
    ):
        for i in range(MAX_N):
            if i < N:
                abs_i = tl.load(ABS_PTR + i)
                rank = 0
                for j in range(MAX_N):
                    if j < N:
                        abs_j = tl.load(ABS_PTR + j)
                        if abs_j <= abs_i:
                            rank += 1
                tl.store(RANK_PTR + i, rank)

    @triton.jit
    def _credit_axis_assign_bins_votes_kernel(
        RANK_PTR,
        MOVES_PTR,
        VOTES_PTR,
        N,
        LO_RANK: tl.constexpr,
        HI_LIMIT: tl.constexpr,
        VOTE_ABS: tl.constexpr,
        MAX_N: tl.constexpr,
    ):
        for i in range(MAX_N):
            if i < N:
                rank = tl.load(RANK_PTR + i)
                in_bin = (rank >= LO_RANK) & (rank < HI_LIMIT)
                move = tl.load(MOVES_PTR + i).to(tl.int16)
                vote = tl.where(in_bin, move * VOTE_ABS, tl.load(VOTES_PTR + i))
                tl.store(VOTES_PTR + i, vote)

else:
    _credit_axis_attribution_row_tile_kernel = None
    _credit_axis_attribution_rescale_kernel = None
    _credit_axis_row_nz_count_kernel = None
    _credit_axis_prefix_sum_exclusive_kernel = None
    _credit_axis_row_compact_scatter_kernel = None
    _credit_axis_project_moves_dense_kernel = None
    _credit_axis_dense_compact_scatter_kernel = None
    _credit_axis_gather_attribution_kernel = None
    _credit_axis_credit_q31_kernel = None
    _credit_axis_grouped_bisect_right_rank_kernel = None
    _credit_axis_assign_bins_votes_kernel = None

# Every @triton.jit kernel launched on the default native pipeline path (manifest must cover all).
DEFAULT_PIPELINE_TRITON_KERNEL_NAMES: tuple[str, ...] = (
    "_credit_axis_attribution_row_tile_kernel",
    "_credit_axis_attribution_rescale_kernel",
    "_credit_axis_row_nz_count_kernel",
    "_credit_axis_prefix_sum_exclusive_kernel",
    "_credit_axis_row_compact_scatter_kernel",
    "_credit_axis_project_moves_dense_kernel",
    "_credit_axis_dense_compact_scatter_kernel",
    "_credit_axis_gather_attribution_kernel",
    "_credit_axis_credit_q31_kernel",
    "_credit_axis_grouped_bisect_right_rank_kernel",
    "_credit_axis_assign_bins_votes_kernel",
)


def _launch_s1_attribution_triton(
    input_q15_list: list[torch.Tensor],
    grad_q16_list: list[torch.Tensor],
    *,
    out_features: int,
    in_features: int,
    law_id: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if _credit_axis_attribution_row_tile_kernel is None:
        raise CreditAxisKernelNotAvailable("triton kernels unavailable")
    device = input_q15_list[0].device
    shift = _attribution_rescale_shift_for_law(law_id)
    half = 1 << (int(shift) - 1)
    row_attrs = torch.empty(out_features * in_features, dtype=torch.int32, device=device)
    with CreditAxisKernelBoundaryGuard(fail_closed=True):
        for row_index in range(out_features):
            acc_o = torch.zeros(in_features, dtype=torch.int64, device=device)
            for input_q15, grad_q16 in zip(input_q15_list, grad_q16_list):
                batch = int(input_q15.shape[0])
                seq = int(input_q15.shape[1]) if input_q15.dim() > 2 else 1
                input_flat = input_q15.reshape(batch, seq, in_features).contiguous()
                grad_flat = grad_q16.reshape(batch, seq, out_features).contiguous()
                for tile_start in range(0, in_features, S1_TILE_CAP):
                    tile_len = min(S1_TILE_CAP, in_features - tile_start)
                    block = triton.next_power_of_2(tile_len)
                    _credit_axis_attribution_row_tile_kernel[(1,)](
                        input_flat,
                        grad_flat,
                        acc_o,
                        row_index,
                        batch,
                        seq,
                        in_features,
                        tile_start,
                        tile_len,
                        input_flat.stride(0),
                        input_flat.stride(1) if input_flat.dim() > 2 else 0,
                        input_flat.stride(-1),
                        grad_flat.stride(0),
                        grad_flat.stride(1) if grad_flat.dim() > 2 else 0,
                        grad_flat.stride(-1),
                        BLOCK=block,
                    )
            row_out = row_attrs[row_index * in_features : (row_index + 1) * in_features]
            _credit_axis_attribution_rescale_kernel[(1,)](
                acc_o,
                row_out,
                in_features,
                SHIFT=shift,
                HALF=half,
            )
        row_nz = torch.empty(out_features, dtype=torch.int32, device=device)
        for row_index in range(out_features):
            _credit_axis_row_nz_count_kernel[(1,)](
                row_attrs,
                row_nz,
                in_features,
                row_index,
            )
        row_base = torch.empty(out_features, dtype=torch.int32, device=device)
        _credit_axis_prefix_sum_exclusive_kernel[(1,)](
            row_nz,
            row_base,
            out_features,
            MAX_N=triton.next_power_of_2(max(out_features, 1)),
        )
        n_events = int(row_nz.sum().item())
        if n_events > S1_EVENT_CAP:
            raise CreditAxisKernelNotAvailable(BR_H_NOT_KERNELIZED)
        if n_events == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                True,
            )
        flat_indices = torch.empty(n_events, dtype=torch.int64, device=device)
        attribution_q31 = torch.empty(n_events, dtype=torch.int32, device=device)
        for row_index in range(out_features):
            _credit_axis_row_compact_scatter_kernel[(1,)](
                row_attrs,
                row_base,
                flat_indices,
                attribution_q31,
                in_features,
                row_index,
                IN_FEATURES=in_features,
            )
    return flat_indices, attribution_q31, True


def _launch_s2_project_and_compact_triton(
    flat_indices: torch.Tensor,
    attribution_q31: torch.Tensor,
    q_levels_flat: torch.Tensor,
    *,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    n_events = int(flat_indices.numel())
    if n_events == 0:
        device = flat_indices.device
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int8, device=device),
            True,
        )
    device = flat_indices.device
    q_levels = q_levels_flat.detach().reshape(-1).to(device=device, dtype=torch.int8)
    move_dense = torch.empty(n_events, dtype=torch.int8, device=device)
    with CreditAxisKernelBoundaryGuard(fail_closed=True):
        grid = (triton.cdiv(n_events, block),)
        _credit_axis_project_moves_dense_kernel[grid](
            flat_indices,
            attribution_q31,
            q_levels,
            move_dense,
            n_events,
            BLOCK=block,
        )
        n_moves = int((move_dense != 0).sum().item())
        if n_moves > S4_NATIVE_MAX:
            raise CreditAxisKernelNotAvailable(BR_H_NOT_KERNELIZED)
        if n_moves == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int8, device=device),
                True,
            )
        projected_move_indices = torch.empty(n_moves, dtype=torch.int64, device=device)
        projected_moves = torch.empty(n_moves, dtype=torch.int8, device=device)
        max_n = triton.next_power_of_2(max(n_events, 1))
        _credit_axis_dense_compact_scatter_kernel[(1,)](
            flat_indices,
            move_dense,
            projected_move_indices,
            projected_moves,
            n_events,
            MAX_N=min(max_n, COMPACT_PREFIX_MAX),
        )
    return projected_move_indices, projected_moves, True


def _launch_s3_gather_and_credit_triton(
    flat_indices: torch.Tensor,
    attribution_q31: torch.Tensor,
    projected_move_indices: torch.Tensor,
    *,
    block: int,
) -> tuple[torch.Tensor, bool]:
    n_moves = int(projected_move_indices.numel())
    n_events = int(flat_indices.numel())
    device = flat_indices.device
    if n_moves == 0:
        return torch.empty(0, dtype=torch.int32, device=device), True
    attr_sel = torch.empty(n_moves, dtype=torch.int32, device=device)
    max_events = triton.next_power_of_2(max(n_events, 1))
    with CreditAxisKernelBoundaryGuard(fail_closed=True):
        _credit_axis_gather_attribution_kernel[(1,)](
            flat_indices,
            attribution_q31,
            n_events,
            projected_move_indices,
            attr_sel,
            n_moves,
            MAX_EVENTS=min(max_events, COMPACT_PREFIX_MAX),
        )
        credit_q31 = torch.empty(n_moves, dtype=torch.int32, device=device)
        grid = (triton.cdiv(n_moves, block),)
        _credit_axis_credit_q31_kernel[grid](attr_sel, credit_q31, n_moves, BLOCK=block)
    return credit_q31, True


def _launch_s4_native_triton(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if _credit_axis_grouped_bisect_right_rank_kernel is None:
        raise CreditAxisKernelNotAvailable("triton kernels unavailable")
    n = int(projected_moves.numel())
    if n == 0:
        device = credit_q31.device
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int16, device=device),
            True,
        )
    if n > S4_NATIVE_MAX:
        raise CreditAxisKernelNotAvailable(BR_H_NOT_KERNELIZED)
    if bool((credit_q31 == INT32_MIN).any().item()):
        raise ValueError("credit_q31 contains INT32_MIN")
    device = credit_q31.device
    abs_i64 = credit_q31.detach().to(dtype=torch.int64, device=device).abs()
    rank_positions = torch.empty(n, dtype=torch.int64, device=device)
    max_n = min(triton.next_power_of_2(max(n, 1)), S4_NATIVE_MAX)
    votes = torch.zeros(n, dtype=torch.int16, device=device)
    with CreditAxisKernelBoundaryGuard(fail_closed=True):
        _credit_axis_grouped_bisect_right_rank_kernel[(1,)](
            abs_i64,
            rank_positions,
            n,
            MAX_N=max_n,
        )
        for canonical_bin in canonical_bins:
            lo_rank, hi_limit = integer_rank_bin_bounds(n, canonical_bin)
            _credit_axis_assign_bins_votes_kernel[(1,)](
                rank_positions,
                projected_moves,
                votes,
                n,
                LO_RANK=lo_rank,
                HI_LIMIT=hi_limit,
                VOTE_ABS=int(canonical_bin.vote_abs),
                MAX_N=max_n,
            )
    return flat_indices.contiguous(), votes.contiguous(), True


def _launch_s4_torch_sort_reference(
    credit_q31: torch.Tensor,
    projected_moves: torch.Tensor,
    flat_indices: torch.Tensor,
    canonical_bins: tuple[CanonicalRankVoteBin, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_i64 = integer_abs_magnitude_i64(credit_q31)
    rank_positions = grouped_bisect_right_rank_positions_integer_abs(abs_i64)
    votes = torch.zeros(int(projected_moves.numel()), dtype=torch.int16, device=projected_moves.device)
    for canonical_bin in canonical_bins:
        lo_rank, hi_limit = integer_rank_bin_bounds(int(projected_moves.numel()), canonical_bin)
        mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
        votes[mask] = (projected_moves[mask].to(torch.int16) * int(canonical_bin.vote_abs)).to(
            torch.int16
        )
    return flat_indices.contiguous(), votes.contiguous()


def _run_integer_pipeline_cuda(
    *,
    capture_inputs: Sequence[torch.Tensor],
    capture_grad_outputs: Sequence[torch.Tensor],
    weight_shape: tuple[int, int],
    q_levels_flat: torch.Tensor,
    rank_bin_spec_canonical: tuple[CanonicalRankVoteBin, ...],
    credit_law_id: str,
    use_torch_sort_s4: bool,
    block: int,
) -> CreditAxisKernelizedPipelineResult:
    if credit_law_id not in {CREDIT_LAW_NEG_ATTRIBUTION_Q31_V1, INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID}:
        raise ValueError(f"unsupported credit_law_id: {credit_law_id!r}")
    out_features, in_features = weight_shape
    paired_inputs = list(capture_inputs[-len(capture_grad_outputs) :])
    batch = int(paired_inputs[0].shape[0]) if paired_inputs else 0
    input_q15_list, grad_q16_list, sequence = _boundary_quantize_captures(
        capture_inputs,
        capture_grad_outputs,
        device=torch.device("cuda"),
    )
    _check_s1_supported_max(
        out_features=out_features,
        in_features=in_features,
        n_capture_pairs=len(capture_grad_outputs),
        batch=batch,
        sequence=sequence,
    )
    flat_indices, attribution_q31, s1_native = _launch_s1_attribution_triton(
        input_q15_list,
        grad_q16_list,
        out_features=out_features,
        in_features=in_features,
        law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    )
    projected_move_indices, projected_moves, s2_native = _launch_s2_project_and_compact_triton(
        flat_indices,
        attribution_q31,
        q_levels_flat,
        block=block,
    )
    credit_q31, s3_native = _launch_s3_gather_and_credit_triton(
        flat_indices,
        attribution_q31,
        projected_move_indices,
        block=block,
    )
    if use_torch_sort_s4:
        sparse_vote_indices, sparse_vote_values = _launch_s4_torch_sort_reference(
            credit_q31,
            projected_moves,
            projected_move_indices,
            rank_bin_spec_canonical,
        )
        s4_native = False
    else:
        sparse_vote_indices, sparse_vote_values, s4_native = _launch_s4_native_triton(
            credit_q31,
            projected_moves,
            projected_move_indices,
            rank_bin_spec_canonical,
        )
    stage_evidence = CreditAxisStageNativeEvidence(
        s1_native=s1_native,
        s2_native=s2_native,
        s3_native=s3_native,
        s4_native=s4_native,
    )
    torch_cuda_reference_only = torch_cuda_reference_only_from_stage_evidence(stage_evidence)
    return CreditAxisKernelizedPipelineResult(
        flat_indices=flat_indices,
        attribution_q31=attribution_q31,
        projected_move_indices=projected_move_indices,
        projected_moves=projected_moves,
        credit_q31=credit_q31,
        sparse_vote_indices=sparse_vote_indices,
        sparse_vote_values=sparse_vote_values,
        torch_cuda_reference_only=torch_cuda_reference_only,
        stage_native_evidence=stage_evidence,
    )


def credit_axis_kernelized_sparse_pipeline_cuda(
    *,
    capture_inputs: Sequence[torch.Tensor],
    capture_grad_outputs: Sequence[torch.Tensor],
    weight_shape: tuple[int, int],
    q_levels_flat: torch.Tensor,
    rank_bin_spec_canonical: tuple[CanonicalRankVoteBin, ...],
    credit_law_id: str,
    block: int = 256,
    use_torch_sort_s4: bool = False,
) -> CreditAxisKernelizedPipelineResult:
    """GPU kernelized credit-axis hot path (native Triton S1–S4)."""
    if not run_gpu_credit_axis_kernel_env_enabled():
        raise CreditAxisKernelNotAvailable(
            f"{RUN_GPU_CREDIT_AXIS_KERNEL_ENV}=1 required; "
            f"terminal branch {BR_H_GPU_DISPATCH_HELD}"
        )
    if not _triton_available() or not _cuda_available() or not credit_axis_kernel_module_built():
        raise CreditAxisKernelNotAvailable(
            f"{CREDIT_AXIS_KERNEL_SEAM_NAME} kernel unavailable; "
            f"terminal branch {BR_H_GPU_KERNEL_MISSING}"
        )
    try:
        classify_credit_axis_gpu_prelaunch_branch(
            triton_available=True,
            cuda_available=True,
            kernel_module_built=True,
            seam_resolves_to_credit_axis_kernel=True,
            dispatch_env_enabled=True,
        )
    except ValueError as exc:
        if "prelaunch checks passed" not in str(exc):
            raise
    else:
        raise CreditAxisKernelNotAvailable("unexpected prelaunch branch")
    return _run_integer_pipeline_cuda(
        capture_inputs=capture_inputs,
        capture_grad_outputs=capture_grad_outputs,
        weight_shape=weight_shape,
        q_levels_flat=q_levels_flat,
        rank_bin_spec_canonical=rank_bin_spec_canonical,
        credit_law_id=credit_law_id,
        use_torch_sort_s4=use_torch_sort_s4,
        block=block,
    )


def credit_axis_kernelized_sparse_pipeline_cuda_torch_sort_s4(
    **kwargs: object,
) -> CreditAxisKernelizedPipelineResult:
    """Debug/reference S4 path using torch.sort — CLEAN-ineligible."""
    return credit_axis_kernelized_sparse_pipeline_cuda(
        **kwargs,  # type: ignore[arg-type]
        use_torch_sort_s4=True,
    )


def default_pipeline_source_forbid_check() -> list[str]:
    """Return forbidden symbols present in default CLEAN seam body (AST gate)."""
    import inspect

    source = inspect.getsource(_run_integer_pipeline_cuda)
    return [sym for sym in _DEFAULT_PIPELINE_FORBIDDEN_SYMBOLS if sym in source]


def check_s1_shape_bounds_or_raise(
    *,
    out_features: int,
    in_features: int,
    n_capture_pairs: int,
    batch: int,
    sequence: int = 1,
) -> str | None:
    try:
        _check_s1_supported_max(
            out_features=out_features,
            in_features=in_features,
            n_capture_pairs=n_capture_pairs,
            batch=batch,
            sequence=sequence,
        )
    except CreditAxisShapeExceedsSupportedMax as exc:
        return str(exc)
    return None
