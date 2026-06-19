"""Credit-axis GPU kernel seam stub (BR-3C-H.1a CPU scaffold).

H.2 deferred: no Triton/CUDA kernel body. Callable always raises CreditAxisKernelNotAvailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_receipt import (
    BR_H_GPU_DISPATCH_HELD,
    BR_H_GPU_KERNEL_MISSING,
    RUN_GPU_CREDIT_AXIS_KERNEL_ENV,
    classify_credit_axis_gpu_prelaunch_branch,
    run_gpu_credit_axis_kernel_env_enabled,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CanonicalRankVoteBin,
)

CREDIT_AXIS_KERNEL_SEAM_NAME = "credit_axis_kernelized_sparse_pipeline_cuda"


class CreditAxisKernelNotAvailable(RuntimeError):
    """Raised when the credit-axis GPU kernel body is unavailable (H.1a stub)."""


@dataclass(frozen=True)
class CreditAxisKernelizedPipelineResult:
    flat_indices: torch.Tensor
    attribution_q31: torch.Tensor
    projected_move_indices: torch.Tensor
    projected_moves: torch.Tensor
    credit_q31: torch.Tensor
    sparse_vote_indices: torch.Tensor
    sparse_vote_values: torch.Tensor


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def classify_credit_axis_gpu_kernel_prelaunch_from_environment() -> str:
    return classify_credit_axis_gpu_prelaunch_branch(
        triton_available=_triton_available(),
        cuda_available=_cuda_available(),
        kernel_module_built=False,
        seam_resolves_to_credit_axis_kernel=True,
        dispatch_env_enabled=run_gpu_credit_axis_kernel_env_enabled(),
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
) -> CreditAxisKernelizedPipelineResult:
    """GPU kernelized credit-axis hot path (stub — H.1a raises NotAvailable)."""
    del block  # reserved for H.2 kernel launch tuning
    if not run_gpu_credit_axis_kernel_env_enabled():
        raise CreditAxisKernelNotAvailable(
            f"{RUN_GPU_CREDIT_AXIS_KERNEL_ENV}=1 required; "
            f"terminal branch {BR_H_GPU_DISPATCH_HELD}"
        )
    prelaunch = classify_credit_axis_gpu_kernel_prelaunch_from_environment()
    if prelaunch != BR_H_GPU_KERNEL_MISSING:
        raise CreditAxisKernelNotAvailable(
            f"credit-axis gpu prelaunch terminal branch {prelaunch}"
        )
    raise CreditAxisKernelNotAvailable(
        f"{CREDIT_AXIS_KERNEL_SEAM_NAME} kernel body not built (H.2 deferred); "
        f"terminal branch {BR_H_GPU_KERNEL_MISSING}"
    )
