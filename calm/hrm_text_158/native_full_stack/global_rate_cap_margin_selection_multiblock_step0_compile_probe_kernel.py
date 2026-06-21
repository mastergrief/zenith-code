"""B2-5a″ Stage-A Step-0′ compile probe kernel (width 2048, ki≤11).

Used ONLY for compile/resource classification in Step-0′ — NOT for parity mint or
runtime sort-correctness claims.  Carries the same single-writer CE + ``if ji < ki``
guard discipline as the banked sort kernel at extended width.
"""
from __future__ import annotations

from pathlib import Path

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
PROBE_KERNEL_SYMBOL = "_margin_selection_bitonic_sort_probe_kernel_2048"
WIDER_PROBE_MAX_LOG2N = 11

if triton is not None:

    @triton.jit
    def _margin_selection_bitonic_sort_probe_kernel_2048(
        keys_ptr,
        sorted_keys_ptr,
        n_rows,
        SORT_PADDED_N: tl.constexpr,
        PADDING_SENTINEL: tl.constexpr,
        LOG2N: tl.constexpr,
    ):
        offs = tl.arange(0, SORT_PADDED_N)
        for ki in tl.static_range(1, 12):
            k = 1 << ki
            active_k = k <= SORT_PADDED_N
            for ji in tl.static_range(11):
                if ji < ki:
                    j = 1 << (ki - ji - 1)
                    partner = offs ^ j
                    key_self = tl.load(
                        keys_ptr + offs, mask=offs < SORT_PADDED_N, other=PADDING_SENTINEL
                    )
                    key_partner = tl.load(
                        keys_ptr + partner,
                        mask=partner < SORT_PADDED_N,
                        other=PADDING_SENTINEL,
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

        sorted_keys = tl.load(
            keys_ptr + offs, mask=offs < SORT_PADDED_N, other=PADDING_SENTINEL
        )
        tl.store(sorted_keys_ptr + offs, sorted_keys, mask=offs < SORT_PADDED_N)

else:
    _margin_selection_bitonic_sort_probe_kernel_2048 = None  # type: ignore[misc,assignment]


__all__ = [
    "PROBE_KERNEL_SYMBOL",
    "PADDING_SENTINEL",
    "WIDER_PROBE_MAX_LOG2N",
    "_TRITON_AVAILABLE",
    "_kernel_file_path_for_test",
    "_margin_selection_bitonic_sort_probe_kernel_2048",
]

_kernel_file_path_for_test = _KERNEL_FILE
