"""B2-5a″ Stage-B Path A wider single-block compose (global keys → one sort → one gather)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
except ImportError:
    triton = None  # type: ignore[assignment]

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget import (
    build_global_mechanism3_full_keys,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    _compute_packed_key_budget,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    KernelBufferProvenance,
    PADDING_SENTINEL,
    _kernel_source_sha256 as _banked_kernel_source_sha256,
    _margin_selection_gather_rows_kernel,
    compute_host_max_full_key_python_int,
    evaluate_padding_headroom,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_wider_single_block_triton_kernel import (
    WIDER_BITONIC_KERNEL_SYMBOL,
    WIDER_SINGLE_BLOCK_ROW_CEILING,
    WIDER_SINGLE_BLOCK_SORT_PADDED_N,
    _TRITON_AVAILABLE,
    _kernel_source_sha256 as _wider_kernel_source_sha256,
    launch_wider_bitonic_sort,
)

WIDER_SINGLE_BLOCK_COMPOSE_SEAM = "margin_selection_wider_single_block_compose"

_ROW_BUFFER_NAMES = (
    "row_state_ids",
    "row_flat_indices",
    "row_local_positions",
    "row_global_flat_indices",
    "row_abs_new_acc",
    "row_thresholds",
    "row_directions",
)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _pos_width_for_row_count(row_count: int) -> int:
    if row_count <= 0:
        return 1
    return max(1, (row_count - 1).bit_length())


class MarginSelectionWiderSingleBlockDeferred(RuntimeError):
    """Honest defer when row_count exceeds proven wider ceiling or budget fails."""


@dataclass(frozen=True)
class MarginSelectionWiderSingleBlockResult:
    row_count: int
    sort_padded_n: int
    pos_width: int
    host_max_full_key: int
    padding_headroom_ok: bool
    budget_infeasible: bool
    wider_single_block_regime: bool
    realistic_size_proven: bool
    sorted_keys: torch.Tensor
    row_state_ids: torch.Tensor
    row_flat_indices: torch.Tensor
    row_local_positions: torch.Tensor
    row_global_flat_indices: torch.Tensor
    row_abs_new_acc: torch.Tensor
    row_thresholds: torch.Tensor
    row_directions: torch.Tensor
    kernel_output_provenance: dict[str, KernelBufferProvenance]
    bitonic_kernel_symbol: str
    gather_kernel_symbol: str
    kernel_source_sha256: str
    compose_seam: str = WIDER_SINGLE_BLOCK_COMPOSE_SEAM


def _validate_sorted_keys_host(
    *,
    sorted_keys: torch.Tensor,
    n_rows: int,
    pos_width: int,
    host_max_full_key: int,
) -> None:
    if n_rows == 0:
        return
    pos_mask = (1 << pos_width) - 1
    head = sorted_keys[:n_rows].detach().cpu().to(torch.int64)
    if bool((head == PADDING_SENTINEL).any().item()):
        raise RuntimeError("padding sentinel appeared in first row_count sorted entries")
    decoded_pos = head & pos_mask
    for i in range(n_rows):
        if int(decoded_pos[i].item()) >= n_rows:
            raise RuntimeError(f"decoded original_pos out of range at i={i}")
    if int(head.max().item()) > host_max_full_key:
        raise RuntimeError("sorted key exceeds host_max_full_key proof bound")


def margin_selection_wider_single_block_compose(
    rows: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> MarginSelectionWiderSingleBlockResult:
    if triton is None or _margin_selection_gather_rows_kernel is None:
        raise RuntimeError("Triton required for wider single-block compose; no fallback")

    dev = device or rows["global_indices"].device
    global_indices = rows["global_indices"]
    abs_new_acc = rows["abs_new_acc"]
    row_count = int(global_indices.numel())

    if row_count == 0:
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        empty_i16 = torch.empty(0, dtype=torch.int16, device=dev)
        return MarginSelectionWiderSingleBlockResult(
            row_count=0,
            sort_padded_n=1,
            pos_width=1,
            host_max_full_key=0,
            padding_headroom_ok=True,
            budget_infeasible=False,
            wider_single_block_regime=False,
            realistic_size_proven=False,
            sorted_keys=empty,
            row_state_ids=empty,
            row_flat_indices=empty,
            row_local_positions=empty,
            row_global_flat_indices=empty,
            row_abs_new_acc=empty,
            row_thresholds=empty,
            row_directions=empty_i16,
            kernel_output_provenance={},
            bitonic_kernel_symbol="",
            gather_kernel_symbol="",
            kernel_source_sha256=_wider_kernel_source_sha256(),
        )

    if row_count > WIDER_SINGLE_BLOCK_ROW_CEILING:
        raise MarginSelectionWiderSingleBlockDeferred(
            f"row_count={row_count} > WIDER_CEILING={WIDER_SINGLE_BLOCK_ROW_CEILING}; "
            "fail-closed pending future size extension"
        )

    budget = _compute_packed_key_budget(
        global_flat_indices=global_indices,
        abs_new_acc=abs_new_acc,
    )
    pos_width = _pos_width_for_row_count(row_count)
    rank_bits = max(1, budget.max_abs_observed.bit_length()) if budget.max_abs_observed > 0 else 1
    full_pack_bits = rank_bits + budget.index_width + pos_width
    host_max_full_key = compute_host_max_full_key_python_int(
        abs_new_acc=abs_new_acc,
        global_flat_indices=global_indices,
        max_abs_observed=budget.max_abs_observed,
        index_width=budget.index_width,
        row_count=row_count,
    )
    headroom = evaluate_padding_headroom(
        host_max_full_key=host_max_full_key,
        full_pack_bits=full_pack_bits,
    )
    if headroom["budget_infeasible"]:
        raise MarginSelectionWiderSingleBlockDeferred(
            "padding headroom / full_pack_bits budget infeasible for wider single-block"
        )

    keys, pos_width = build_global_mechanism3_full_keys(
        abs_new_acc=abs_new_acc,
        global_flat_indices=global_indices,
        max_abs_observed=budget.max_abs_observed,
        index_width=budget.index_width,
        global_row_count=row_count,
        device=dev,
    )

    sort_padded_n = _next_power_of_2(row_count)
    if sort_padded_n > WIDER_SINGLE_BLOCK_SORT_PADDED_N:
        raise MarginSelectionWiderSingleBlockDeferred(
            f"sort_padded_n={sort_padded_n} exceeds proven width {WIDER_SINGLE_BLOCK_SORT_PADDED_N}"
        )

    keys_workspace = torch.full(
        (sort_padded_n,), PADDING_SENTINEL, dtype=torch.int64, device=dev
    )
    keys_workspace[:row_count] = keys
    sorted_keys = launch_wider_bitonic_sort(
        keys_workspace,
        n_rows=row_count,
        sort_padded_n=sort_padded_n,
        device=dev,
    )
    _validate_sorted_keys_host(
        sorted_keys=sorted_keys,
        n_rows=row_count,
        pos_width=pos_width,
        host_max_full_key=host_max_full_key,
    )

    out_state_ids = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_flat = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_local = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_global = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_abs = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_thresh = torch.empty(row_count, dtype=torch.int64, device=dev)
    out_dirs = torch.empty(row_count, dtype=torch.int16, device=dev)

    grid = (triton.cdiv(row_count, 128),) if triton is not None else (1,)
    gather_sym = _margin_selection_gather_rows_kernel.__name__
    _margin_selection_gather_rows_kernel[grid](
        sorted_keys,
        rows["state_ids"],
        rows["flat_indices"],
        rows["local_positions"],
        rows["global_indices"],
        rows["abs_new_acc"],
        rows["thresholds"],
        rows["directions"],
        out_state_ids,
        out_flat,
        out_local,
        out_global,
        out_abs,
        out_thresh,
        out_dirs,
        row_count,
        POS_WIDTH=pos_width,
        BLOCK=128,
    )

    banked_sha = _banked_kernel_source_sha256()
    wider_sha = _wider_kernel_source_sha256()
    provenance = {
        "sorted_keys": KernelBufferProvenance(
            buffer_role="sorted_keys",
            kernel_symbol=WIDER_BITONIC_KERNEL_SYMBOL,
            kernel_source_sha256=wider_sha,
        ),
        **{
            name: KernelBufferProvenance(
                buffer_role=name,
                kernel_symbol=gather_sym,
                kernel_source_sha256=banked_sha,
            )
            for name in _ROW_BUFFER_NAMES
        },
    }

    wider_regime = row_count > 1024
    return MarginSelectionWiderSingleBlockResult(
        row_count=row_count,
        sort_padded_n=sort_padded_n,
        pos_width=pos_width,
        host_max_full_key=host_max_full_key,
        padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
        budget_infeasible=False,
        wider_single_block_regime=wider_regime,
        realistic_size_proven=False,
        sorted_keys=sorted_keys,
        row_state_ids=out_state_ids,
        row_flat_indices=out_flat,
        row_local_positions=out_local,
        row_global_flat_indices=out_global,
        row_abs_new_acc=out_abs,
        row_thresholds=out_thresh,
        row_directions=out_dirs,
        kernel_output_provenance=provenance,
        bitonic_kernel_symbol=WIDER_BITONIC_KERNEL_SYMBOL,
        gather_kernel_symbol=gather_sym,
        kernel_source_sha256=wider_sha,
    )


__all__ = [
    "WIDER_SINGLE_BLOCK_COMPOSE_SEAM",
    "MarginSelectionWiderSingleBlockDeferred",
    "MarginSelectionWiderSingleBlockResult",
    "margin_selection_wider_single_block_compose",
]
