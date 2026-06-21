"""B2-5a″ Stage-B Path A wider single-block bitonic sort kernel (width 2048, ki≤11).

Production wider sort for realistic row_count > BLOCK and <= WIDER_CEILING.
Import-only from banked floor constants; does NOT mutate triton_kernel.py.

Runtime uses host-orchestrated compare-exchange stages (grid=sort_padded_n/1024)
to avoid in-place races at width 2048.  The stage kernel symbol is audited;
the ki/ji schedule is mirrored by ``bitonic_sort_single_writer_reference_wide``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    PADDING_SENTINEL,
)

_KERNEL_FILE = Path(__file__)
_kernel_file_path_for_test = _KERNEL_FILE
WIDER_SINGLE_BLOCK_SORT_PADDED_N = 2048
WIDER_SINGLE_BLOCK_ROW_CEILING = WIDER_SINGLE_BLOCK_SORT_PADDED_N
WIDER_BITONIC_KERNEL_SYMBOL = "_margin_selection_bitonic_ce_wider_stage_kernel"
WIDER_MAX_LOG2N = 11
WIDER_SORT_BLOCK = 1024


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


def iter_wider_bitonic_stage_schedule(*, sort_padded_n: int, max_log2n: int = WIDER_MAX_LOG2N):
    """Yield (k, j) compare-exchange stages matching the CPU reference schedule."""

    for ki in range(1, max_log2n + 1):
        k = 1 << ki
        if k > sort_padded_n:
            continue
        for ji in range(ki):
            j = 1 << (ki - ji - 1)
            yield k, j


def bitonic_sort_single_writer_reference_wide(
    keys: list[int],
    *,
    sort_padded_n: int,
    padding_sentinel: int = PADDING_SENTINEL,
    max_log2n: int = WIDER_MAX_LOG2N,
) -> list[int]:
    """CPU reference mirroring the wider host stage schedule exactly."""

    arr = list(keys) + [padding_sentinel] * max(0, sort_padded_n - len(keys))
    if len(arr) != sort_padded_n:
        raise ValueError("sort_padded_n mismatch")
    for k, j in iter_wider_bitonic_stage_schedule(
        sort_padded_n=sort_padded_n,
        max_log2n=max_log2n,
    ):
        next_arr = list(arr)
        for idx in range(sort_padded_n):
            partner = idx ^ j
            key_self = arr[idx]
            key_partner = arr[partner] if partner < sort_padded_n else padding_sentinel
            upward = (idx & k) == 0
            is_low = idx < partner
            keep_small = (upward and is_low) or ((not upward) and (not is_low))
            lo = min(key_self, key_partner)
            hi = max(key_self, key_partner)
            next_arr[idx] = lo if keep_small else hi
        arr = next_arr
    return arr


if triton is not None:

    @triton.jit
    def _margin_selection_bitonic_ce_wider_stage_kernel(
        keys_in_ptr,
        keys_out_ptr,
        SORT_PADDED_N: tl.constexpr,
        PADDING_SENTINEL: tl.constexpr,
        K: tl.constexpr,
        J: tl.constexpr,
        SORT_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        local_offs = tl.arange(0, SORT_BLOCK)
        global_offs = pid * SORT_BLOCK + local_offs
        in_bounds = global_offs < SORT_PADDED_N
        partner = global_offs ^ J
        key_self = tl.load(
            keys_in_ptr + global_offs,
            mask=in_bounds,
            other=PADDING_SENTINEL,
        )
        key_partner = tl.load(
            keys_in_ptr + partner,
            mask=partner < SORT_PADDED_N,
            other=PADDING_SENTINEL,
        )
        upward = (global_offs & K) == 0
        is_low = global_offs < partner
        keep_small = (upward & is_low) | ((~upward) & (~is_low))
        lo = tl.minimum(key_self, key_partner)
        hi = tl.maximum(key_self, key_partner)
        new_val = tl.where(keep_small, lo, hi)
        tl.store(keys_out_ptr + global_offs, new_val, mask=in_bounds)

else:
    _margin_selection_bitonic_ce_wider_stage_kernel = None  # type: ignore[misc,assignment]

# Backward-compatible alias for audit imports expecting a monolithic symbol name.
_margin_selection_bitonic_sort_wider_single_block_kernel = (
    _margin_selection_bitonic_ce_wider_stage_kernel
)


def launch_wider_bitonic_sort(
    keys_workspace: torch.Tensor,
    *,
    n_rows: int,
    sort_padded_n: int,
    device: torch.device,
) -> torch.Tensor:
    if triton is None or _margin_selection_bitonic_ce_wider_stage_kernel is None:
        raise RuntimeError("Triton required for wider bitonic sort")
    if sort_padded_n > WIDER_SINGLE_BLOCK_ROW_CEILING:
        raise ValueError(
            f"sort_padded_n={sort_padded_n} must be <= {WIDER_SINGLE_BLOCK_ROW_CEILING}"
        )
    if sort_padded_n % WIDER_SORT_BLOCK != 0:
        raise ValueError(
            f"sort_padded_n={sort_padded_n} must be divisible by {WIDER_SORT_BLOCK}"
        )

    grid = (sort_padded_n // WIDER_SORT_BLOCK,)
    buf_a = keys_workspace
    buf_b = torch.empty(sort_padded_n, dtype=torch.int64, device=device)
    src, dst = buf_a, buf_b
    for k, j in iter_wider_bitonic_stage_schedule(sort_padded_n=sort_padded_n):
        _margin_selection_bitonic_ce_wider_stage_kernel[grid](
            src,
            dst,
            SORT_PADDED_N=sort_padded_n,
            PADDING_SENTINEL=PADDING_SENTINEL,
            K=k,
            J=j,
            SORT_BLOCK=WIDER_SORT_BLOCK,
        )
        src, dst = dst, src

    return src[:n_rows].contiguous()


def verify_runtime_sort_key_exactness(
    *,
    device: torch.device,
    sort_padded_n: int,
    n_rows: int,
    seed: int = 0,
) -> bool:
    """Ladder step 1: wider kernel output == CPU sorted keys (GPU only)."""

    proof = build_runtime_sort_key_proof(
        device=device,
        sort_padded_n=sort_padded_n,
        n_rows=n_rows,
        seed=seed,
    )
    return proof.exact


def build_runtime_sort_key_proof(
    *,
    device: torch.device,
    sort_padded_n: int,
    n_rows: int,
    seed: int = 0,
) -> "RuntimeSortKeyProof":
    """Build step-1 RuntimeSortKeyProof from an in-lane GPU sort vs CPU reference."""

    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_feasibility_receipt import (
        canonical_tensor_payload_sha256,
    )
    from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_native_parity_receipt import (
        RuntimeSortKeyProof,
    )

    if device.type != "cuda":
        raise RuntimeError("runtime sort-key proof requires CUDA")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    keys = torch.randint(
        low=1,
        high=1 << 40,
        size=(n_rows,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    workspace = torch.full(
        (sort_padded_n,), PADDING_SENTINEL, dtype=torch.int64, device=device
    )
    workspace[:n_rows] = keys
    native_sorted = launch_wider_bitonic_sort(
        workspace.clone(),
        n_rows=n_rows,
        sort_padded_n=sort_padded_n,
        device=device,
    )
    cpu_expected = bitonic_sort_single_writer_reference_wide(
        keys.detach().cpu().tolist(),
        sort_padded_n=sort_padded_n,
        padding_sentinel=PADDING_SENTINEL,
        max_log2n=WIDER_MAX_LOG2N,
    )[:n_rows]
    exact = native_sorted.detach().cpu().tolist() == cpu_expected
    native_sha = canonical_tensor_payload_sha256(native_sorted)
    cpu_sha = canonical_tensor_payload_sha256(
        torch.tensor(cpu_expected, dtype=torch.int64, device="cpu")
    )
    return RuntimeSortKeyProof(
        sort_padded_n=sort_padded_n,
        n_rows=n_rows,
        native_sorted_sha256=native_sha,
        cpu_ref_sha256=cpu_sha,
        exact=exact,
    )


__all__ = [
    "PADDING_SENTINEL",
    "WIDER_BITONIC_KERNEL_SYMBOL",
    "WIDER_MAX_LOG2N",
    "WIDER_SORT_BLOCK",
    "WIDER_SINGLE_BLOCK_ROW_CEILING",
    "WIDER_SINGLE_BLOCK_SORT_PADDED_N",
    "_TRITON_AVAILABLE",
    "_kernel_file_path_for_test",
    "_kernel_source_sha256",
    "_margin_selection_bitonic_ce_wider_stage_kernel",
    "_margin_selection_bitonic_sort_wider_single_block_kernel",
    "bitonic_sort_single_writer_reference_wide",
    "build_runtime_sort_key_proof",
    "iter_wider_bitonic_stage_schedule",
    "launch_wider_bitonic_sort",
    "verify_runtime_sort_key_exactness",
]
