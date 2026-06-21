"""B2-5a′ Stage-2 (a) native single-block/TILE MARGIN-selection @triton.jit kernels.

Phase A: in-block bitonic sort on mechanism-3 full_key (whole int64 ascending).
Phase B: gather/write kernel reorders all 7 parallel row tensors via tl.load/tl.store.

NO tl.sort / torch.sort / torch.topk / torch advanced-indexing gather in the native
path.  row_count > BLOCK → fail-closed at host wrapper (no torch fallback).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GlobalRateCapMarginSelectionFeasibilityNull,
    _compute_packed_key_budget,
    _device_row_tensors_for_selection,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    TRITON_SINGLE_BLOCK_ROW_CEILING,
)

_KERNEL_FILE = Path(__file__)
_kernel_file_path_for_test = _KERNEL_FILE

INT64_MAX = (1 << 63) - 1
INT64_HEADROOM_LIMIT = INT64_MAX - 1
PADDING_SENTINEL = INT64_MAX

MULTIBLOCK_COMPOSITION_SEAM = "margin_selection_single_block_tile"

_ROW_BUFFER_NAMES = (
    "row_state_ids",
    "row_flat_indices",
    "row_local_positions",
    "row_global_flat_indices",
    "row_abs_new_acc",
    "row_thresholds",
    "row_directions",
)


def _kernel_source_sha256() -> str:
    return hashlib.sha256(_KERNEL_FILE.read_bytes()).hexdigest()


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _log2_ceil(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def bitonic_sort_single_writer_reference(
    keys: list[int],
    *,
    sort_padded_n: int,
    padding_sentinel: int = PADDING_SENTINEL,
    max_log2n: int = 10,
) -> list[int]:
    """Pure-Python single-writer bitonic network (same schedule as the Triton kernel).

    CPU reference model for gate-1 logic verification — NOT a native pass path.
    """

    arr = list(keys) + [padding_sentinel] * max(0, sort_padded_n - len(keys))
    if len(arr) != sort_padded_n:
        raise ValueError("sort_padded_n mismatch")
    for ki in range(1, max_log2n + 1):
        k = 1 << ki
        active_k = k <= sort_padded_n
        for ji in range(max_log2n):
            if ji < ki:
                j = 1 << (ki - ji - 1)
                next_arr = list(arr)
                for idx in range(sort_padded_n):
                    partner = idx ^ j
                    key_self = arr[idx]
                    key_partner = (
                        arr[partner] if partner < sort_padded_n else padding_sentinel
                    )
                    upward = (idx & k) == 0
                    is_low = idx < partner
                    keep_small = (upward and is_low) or ((not upward) and (not is_low))
                    lo = min(key_self, key_partner)
                    hi = max(key_self, key_partner)
                    if active_k:
                        next_arr[idx] = lo if keep_small else hi
                    else:
                        next_arr[idx] = key_self
                arr = next_arr
    return arr


def _pos_width_for_row_count(row_count: int) -> int:
    if row_count <= 0:
        return 1
    return max(1, (row_count - 1).bit_length())


def _packed_full_key_python_int(
    *,
    rank: int,
    global_flat_index: int,
    original_pos: int,
    index_width: int,
    pos_width: int,
) -> int:
    return (
        (int(rank) << (index_width + pos_width))
        | (int(global_flat_index) << pos_width)
        | int(original_pos)
    )


def compute_host_max_full_key_python_int(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    max_abs_observed: int,
    index_width: int,
    row_count: int,
) -> int:
    """Python-int headroom guard BEFORE any signed-int64 cast (overflow-safe)."""

    pos_width = _pos_width_for_row_count(row_count)
    host_max = 0
    for pos in range(row_count):
        rank = int(max_abs_observed) - int(abs_new_acc[pos].item())
        gfi = int(global_flat_indices[pos].item())
        key = _packed_full_key_python_int(
            rank=rank,
            global_flat_index=gfi,
            original_pos=pos,
            index_width=index_width,
            pos_width=pos_width,
        )
        if key > host_max:
            host_max = key
    return host_max


def evaluate_padding_headroom(
    *,
    host_max_full_key: int,
    full_pack_bits: int,
) -> dict[str, Any]:
    padding_headroom_ok = host_max_full_key <= INT64_HEADROOM_LIMIT
    budget_infeasible = (
        not padding_headroom_ok
        or host_max_full_key == PADDING_SENTINEL
        or full_pack_bits >= 63
    )
    return {
        "host_max_full_key": host_max_full_key,
        "padding_sentinel": PADDING_SENTINEL,
        "padding_headroom_ok": padding_headroom_ok,
        "full_pack_bits": full_pack_bits,
        "budget_infeasible": budget_infeasible,
    }


def _build_mechanism3_full_keys(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    max_abs_observed: int,
    index_width: int,
    row_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    pos_width = _pos_width_for_row_count(row_count)
    rank = int(max_abs_observed) - abs_new_acc.to(torch.int64)
    original_pos = torch.arange(row_count, device=device, dtype=torch.int64)
    keys = (
        (rank << (index_width + pos_width))
        | (global_flat_indices.to(torch.int64) << pos_width)
        | original_pos
    )
    return keys.to(torch.int64), pos_width


if triton is not None:

    @triton.jit
    def _margin_selection_bitonic_sort_kernel(
        keys_ptr,
        sorted_keys_ptr,
        n_rows,
        SORT_PADDED_N: tl.constexpr,
        PADDING_SENTINEL: tl.constexpr,
        LOG2N: tl.constexpr,
    ):
        offs = tl.arange(0, SORT_PADDED_N)
        for ki in tl.static_range(1, 11):
            k = 1 << ki
            active_k = k <= SORT_PADDED_N
            for ji in tl.static_range(10):
                if ji < ki:
                    j = 1 << (ki - ji - 1)
                    partner = offs ^ j
                    key_self = tl.load(keys_ptr + offs, mask=offs < SORT_PADDED_N, other=PADDING_SENTINEL)
                    key_partner = tl.load(
                        keys_ptr + partner, mask=partner < SORT_PADDED_N, other=PADDING_SENTINEL
                    )
                    upward = (offs & k) == 0
                    is_low = offs < partner
                    keep_small = (upward & is_low) | ((~upward) & (~is_low))
                    lo = tl.minimum(key_self, key_partner)
                    hi = tl.maximum(key_self, key_partner)
                    new_val = tl.where(keep_small, lo, hi)
                    tl.store(
                        keys_ptr + offs,
                        tl.where(active_k, new_val, key_self),
                        mask=offs < SORT_PADDED_N,
                    )

        sorted_keys = tl.load(keys_ptr + offs, mask=offs < SORT_PADDED_N, other=PADDING_SENTINEL)
        tl.store(sorted_keys_ptr + offs, sorted_keys, mask=offs < SORT_PADDED_N)

    @triton.jit
    def _margin_selection_gather_rows_kernel(
        sorted_keys_ptr,
        src_state_ids_ptr,
        src_flat_indices_ptr,
        src_local_positions_ptr,
        src_global_indices_ptr,
        src_abs_new_acc_ptr,
        src_thresholds_ptr,
        src_directions_ptr,
        dst_state_ids_ptr,
        dst_flat_indices_ptr,
        dst_local_positions_ptr,
        dst_global_indices_ptr,
        dst_abs_new_acc_ptr,
        dst_thresholds_ptr,
        dst_directions_ptr,
        n_rows,
        POS_WIDTH: tl.constexpr,
        BLOCK: tl.constexpr = 128,
    ):
        pos_mask = (1 << POS_WIDTH) - 1
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_rows

        sorted_key = tl.load(sorted_keys_ptr + offs, mask=mask, other=0).to(tl.int64)
        src_idx = sorted_key & pos_mask

        tl.store(
            dst_state_ids_ptr + offs,
            tl.load(src_state_ids_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_flat_indices_ptr + offs,
            tl.load(src_flat_indices_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_local_positions_ptr + offs,
            tl.load(src_local_positions_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_global_indices_ptr + offs,
            tl.load(src_global_indices_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_abs_new_acc_ptr + offs,
            tl.load(src_abs_new_acc_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_thresholds_ptr + offs,
            tl.load(src_thresholds_ptr + src_idx, mask=mask, other=0),
            mask=mask,
        )
        tl.store(
            dst_directions_ptr + offs,
            tl.load(src_directions_ptr + src_idx, mask=mask, other=0).to(tl.int16),
            mask=mask,
        )

else:
    _margin_selection_bitonic_sort_kernel = None  # type: ignore[misc,assignment]
    _margin_selection_gather_rows_kernel = None  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class KernelBufferProvenance:
    buffer_role: str
    kernel_symbol: str
    kernel_source_sha256: str


@dataclass(frozen=True)
class MarginSelectionSingleBlockTileResult:
    """Reusable single-block tile primitive for B2-5a'' multiblock composition."""

    row_count: int
    sort_padded_n: int
    pos_width: int
    host_max_full_key: int
    padding_headroom_ok: bool
    budget_infeasible: bool
    multiblock_deferred: bool
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
    tile_primitive_seam: str = MULTIBLOCK_COMPOSITION_SEAM


class MarginSelectionSingleBlockDeferred(RuntimeError):
    """Honest defer when row_count > BLOCK or budget/headroom fails."""


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


def margin_selection_single_block_tile(
    rows: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> MarginSelectionSingleBlockTileResult:
    """Host wrapper for the single-block native tile (sort + gather).

    Fail-closed when row_count > BLOCK, budget infeasible, or Triton absent.
    """

    if triton is None or _margin_selection_bitonic_sort_kernel is None:
        raise RuntimeError("Triton is required for native single-block tile; no fallback")

    dev = device or rows["global_indices"].device
    global_indices = rows["global_indices"]
    abs_new_acc = rows["abs_new_acc"]
    row_count = int(global_indices.numel())

    if row_count == 0:
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        empty_i16 = torch.empty(0, dtype=torch.int16, device=dev)
        return MarginSelectionSingleBlockTileResult(
            row_count=0,
            sort_padded_n=1,
            pos_width=1,
            host_max_full_key=0,
            padding_headroom_ok=True,
            budget_infeasible=False,
            multiblock_deferred=False,
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
            kernel_source_sha256=_kernel_source_sha256(),
        )

    if row_count > TRITON_SINGLE_BLOCK_ROW_CEILING:
        raise MarginSelectionSingleBlockDeferred(
            f"row_count={row_count} > BLOCK={TRITON_SINGLE_BLOCK_ROW_CEILING}; "
            "single-block tile deferred pending B2-5a'' multiblock merge"
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
        raise MarginSelectionSingleBlockDeferred(
            "padding headroom / full_pack_bits budget infeasible for single-block tile"
        )

    keys, pos_width = _build_mechanism3_full_keys(
        abs_new_acc=abs_new_acc,
        global_flat_indices=global_indices,
        max_abs_observed=budget.max_abs_observed,
        index_width=budget.index_width,
        row_count=row_count,
        device=dev,
    )

    sort_padded_n = _next_power_of_2(row_count)
    log2n = _log2_ceil(sort_padded_n)
    keys_workspace = torch.full(
        (sort_padded_n,), PADDING_SENTINEL, dtype=torch.int64, device=dev
    )
    keys_workspace[:row_count] = keys
    sorted_keys = torch.empty(sort_padded_n, dtype=torch.int64, device=dev)

    _margin_selection_bitonic_sort_kernel[(1,)](
        keys_workspace,
        sorted_keys,
        row_count,
        SORT_PADDED_N=sort_padded_n,
        PADDING_SENTINEL=PADDING_SENTINEL,
        LOG2N=log2n,
    )

    sorted_keys = sorted_keys[:row_count].contiguous()
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

    grid = (triton.cdiv(row_count, 128),)
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

    source_sha = _kernel_source_sha256()
    bitonic_sym = _margin_selection_bitonic_sort_kernel.__name__
    gather_sym = _margin_selection_gather_rows_kernel.__name__
    provenance = {
        name: KernelBufferProvenance(
            buffer_role=name,
            kernel_symbol=gather_sym,
            kernel_source_sha256=source_sha,
        )
        for name in _ROW_BUFFER_NAMES
    }

    return MarginSelectionSingleBlockTileResult(
        row_count=row_count,
        sort_padded_n=sort_padded_n,
        pos_width=pos_width,
        host_max_full_key=host_max_full_key,
        padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
        budget_infeasible=False,
        multiblock_deferred=False,
        sorted_keys=sorted_keys,
        row_state_ids=out_state_ids,
        row_flat_indices=out_flat,
        row_local_positions=out_local,
        row_global_flat_indices=out_global,
        row_abs_new_acc=out_abs,
        row_thresholds=out_thresh,
        row_directions=out_dirs,
        kernel_output_provenance=provenance,
        bitonic_kernel_symbol=bitonic_sym,
        gather_kernel_symbol=gather_sym,
        kernel_source_sha256=source_sha,
    )


__all__ = [
    "INT64_HEADROOM_LIMIT",
    "INT64_MAX",
    "MULTIBLOCK_COMPOSITION_SEAM",
    "PADDING_SENTINEL",
    "KernelBufferProvenance",
    "MarginSelectionSingleBlockDeferred",
    "MarginSelectionSingleBlockTileResult",
    "_kernel_file_path_for_test",
    "_kernel_source_sha256",
    "_margin_selection_bitonic_sort_kernel",
    "_margin_selection_gather_rows_kernel",
    "_TRITON_AVAILABLE",
    "bitonic_sort_single_writer_reference",
    "compute_host_max_full_key_python_int",
    "evaluate_padding_headroom",
    "margin_selection_single_block_tile",
]
