"""B2-5a env-gated dispatcher for MARGIN selection scaffold vs reference.

Mirrors the B2-4 ``qacc_apply_composition_dispatch.py`` pattern: lane env
(``RUN_GPU_GLOBAL_RATE_CAP_ENV=1``) gates the GPU lane; scaffold env
(``HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD=1``) opts into the packed-key
CPU scaffold path.  The prior ``..._NATIVE`` env name is fail-closed.  Scaffold-on
without Triton import -> RuntimeError (no silent fallback).  The reference
fallback remains the default and its bodies are NOT mutated.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    DeviceGlobalRateCapSelectionResult,
    GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    select_global_rate_cap_rows_torch_cuda_reference,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCOPE,
    RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV,
    _TRITON_AVAILABLE,
    GlobalRateCapMarginSelectionFeasibilityNull,
    _reject_legacy_native_env,
    select_global_rate_cap_rows_margin_scaffold,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
    GlobalRateCapMarginSelectionFeasibilityReceipt,
)


def _require_lane_env() -> None:
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_GLOBAL_RATE_CAP_ENV}=1 is required and must only be set "
            "inside a granted gpu:0 resource lane"
        )


def _scaffold_routing_enabled() -> bool:
    return os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV) == "1"


def select_global_rate_cap_rows_under_margin(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    materialize_cpu_telemetry: bool = True,
) -> tuple[
    DeviceGlobalRateCapSelectionResult,
    GlobalRateCapMarginSelectionFeasibilityReceipt | None,
]:
    """Dispatch MARGIN selection to reference or CPU scaffold under env gates.

    Returns ``(selection, receipt_or_None)``: the scaffold path returns a
    feasibility-null receipt; the reference path returns ``None``.
    """

    _reject_legacy_native_env()
    _require_lane_env()
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)

    if _scaffold_routing_enabled():
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                f"{RUN_GPU_GLOBAL_RATE_CAP_SCAFFOLD_ENV}=1 requires Triton import; "
                "reference fallback is forbidden"
            )
        selection, receipt = select_global_rate_cap_rows_margin_scaffold(
            inputs,
            spec,
            tensor_offsets=offsets,
            deferred_backlog=deferred_backlog,
            scope=GLOBAL_RATE_CAP_MARGIN_SELECTION_SCAFFOLD_SCOPE,
        )
        return selection, receipt

    selection = select_global_rate_cap_rows_torch_cuda_reference(
        inputs,
        spec,
        tensor_offsets=offsets,
        deferred_backlog=deferred_backlog,
        materialize_cpu_telemetry=materialize_cpu_telemetry,
        scope=GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
    )
    return selection, None
