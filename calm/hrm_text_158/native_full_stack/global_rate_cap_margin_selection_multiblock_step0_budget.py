"""B2-5a″ Stage-A Step-0′ width classifier + budget receipt harness.

CPU budget @ realistic N, frozen assertions #1–#4, compile/resource probe at legal
power-of-two widths only.  Does NOT mint selection_parity_pass or claim runtime sort
correctness.
"""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    tensor_offsets_for_vote_update_states,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget_receipt import (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION,
    CompileProbeMeasurement,
    GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt,
    MultiblockStep0AssertionResults,
    MultiblockStep0Decision,
    MultiblockStep0FixtureMeasurement,
    OPTIONAL_BASELINE_SORT_PADDED_N,
    REALISTIC_ROW_COUNTS,
    WIDER_SINGLE_BLOCK_SORT_PADDED_N,
    build_global_rate_cap_margin_selection_multiblock_step0_budget_receipt,
    validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_compile_probe_kernel import (
    PROBE_KERNEL_SYMBOL,
    WIDER_PROBE_MAX_LOG2N,
    _TRITON_AVAILABLE,
    _margin_selection_bitonic_sort_probe_kernel_2048,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_packed_key_scaffold import (
    GlobalRateCapMarginSelectionFeasibilityNull,
    _compute_packed_key_budget,
    _device_row_tensors_for_selection,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget import (
    build_upper_bound_fixture_inputs,
    expected_row_count_upper_bound,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget_receipt import (
    TRITON_SINGLE_BLOCK_ROW_CEILING,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_triton_kernel import (
    PADDING_SENTINEL,
    _margin_selection_bitonic_sort_kernel,
    bitonic_sort_single_writer_reference,
    compute_host_max_full_key_python_int,
    evaluate_padding_headroom,
)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_length_non_negative(value: int) -> int:
    if value < 0:
        raise ValueError(f"bit_length input must be >= 0, got {value}")
    return max(1, int(value).bit_length()) if value > 0 else 1


def _pos_width_for_row_count(row_count: int) -> int:
    if row_count <= 0:
        return 1
    return max(1, (row_count - 1).bit_length())


def build_global_mechanism3_full_keys(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    max_abs_observed: int,
    index_width: int,
    global_row_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build packed full_keys with GLOBAL original_pos embedded in low bits."""

    pos_width = _pos_width_for_row_count(global_row_count)
    rank = int(max_abs_observed) - abs_new_acc.to(torch.int64)
    original_pos = torch.arange(global_row_count, device=device, dtype=torch.int64)
    keys = (
        (rank << (index_width + pos_width))
        | (global_flat_indices.to(torch.int64) << pos_width)
        | original_pos
    )
    return keys.to(torch.int64), pos_width


def build_realistic_fixture_inputs(*, target_row_count: int) -> list[GlobalRateCapTensorInput]:
    """Build upper-bound fixture inputs targeting an exact realistic row_count."""

    if target_row_count not in REALISTIC_ROW_COUNTS:
        raise ValueError(f"target_row_count must be one of {REALISTIC_ROW_COUNTS}")
    per_state = target_row_count // 256
    if per_state * 256 != target_row_count:
        raise ValueError(f"target_row_count {target_row_count} not divisible by 256")
    inputs = build_upper_bound_fixture_inputs(
        numel=256,
        max_abs_per_tensor=256,
        num_states=per_state,
    )
    expected = expected_row_count_upper_bound(
        numel=256,
        max_abs_per_tensor=256,
        num_states=per_state,
    )
    if expected != target_row_count:
        raise ValueError(
            f"fixture builder mismatch: expected {target_row_count}, got {expected}"
        )
    return inputs


def measure_multiblock_step0_fixture(
    *,
    fixture_name: str,
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    tensor_offsets: dict[str, int] | None = None,
    device: torch.device | None = None,
) -> MultiblockStep0FixtureMeasurement:
    dev = device or torch.device("cpu")
    offsets = dict(tensor_offsets or tensor_offsets_for_vote_update_states(inputs))
    rows = _device_row_tensors_for_selection(inputs, tensor_offsets=offsets, device=dev)
    row_count = int(rows["global_indices"].numel())
    cap = max(0, int(spec.cap))

    if row_count not in REALISTIC_ROW_COUNTS:
        raise ValueError(f"realistic fixture row_count must be in {REALISTIC_ROW_COUNTS}")

    try:
        budget = _compute_packed_key_budget(
            global_flat_indices=rows["global_indices"],
            abs_new_acc=rows["abs_new_acc"],
        )
    except GlobalRateCapMarginSelectionFeasibilityNull:
        return MultiblockStep0FixtureMeasurement(
            fixture_name=fixture_name,
            row_count=row_count,
            max_global_flat_index=-1,
            index_width=-1,
            max_abs_observed=-1,
            rank_bits=-1,
            pos_width=-1,
            full_pack_bits=-1,
            host_max_full_key=-1,
            sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
            int63_full_pack_ok=False,
            padding_headroom_ok=False,
            budget_infeasible=True,
            cap=cap,
        )

    index_width = budget.index_width
    max_abs_observed = budget.max_abs_observed
    rank_bits = _bit_length_non_negative(max_abs_observed)
    pos_width = _pos_width_for_row_count(row_count)
    full_pack_bits = rank_bits + index_width + pos_width
    host_max = compute_host_max_full_key_python_int(
        abs_new_acc=rows["abs_new_acc"],
        global_flat_indices=rows["global_indices"],
        max_abs_observed=max_abs_observed,
        index_width=index_width,
        row_count=row_count,
    )
    headroom = evaluate_padding_headroom(
        host_max_full_key=host_max,
        full_pack_bits=full_pack_bits,
    )
    return MultiblockStep0FixtureMeasurement(
        fixture_name=fixture_name,
        row_count=row_count,
        max_global_flat_index=budget.max_global_flat_index,
        index_width=index_width,
        max_abs_observed=max_abs_observed,
        rank_bits=rank_bits,
        pos_width=pos_width,
        full_pack_bits=full_pack_bits,
        host_max_full_key=host_max,
        sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
        int63_full_pack_ok=full_pack_bits <= 63,
        padding_headroom_ok=bool(headroom["padding_headroom_ok"]),
        budget_infeasible=bool(headroom["budget_infeasible"]),
        cap=cap,
    )


def _tile_sizes(row_count: int, *, block: int = TRITON_SINGLE_BLOCK_ROW_CEILING) -> list[int]:
    sizes: list[int] = []
    remaining = row_count
    while remaining > 0:
        chunk = min(block, remaining)
        sizes.append(chunk)
        remaining -= chunk
    return sizes


def _merge_tree_pairs(tile_sizes: list[int]) -> list[tuple[int, int]]:
    level = list(tile_sizes)
    pairs: list[tuple[int, int]] = []
    while len(level) > 1:
        next_level: list[int] = []
        idx = 0
        while idx < len(level):
            if idx + 1 < len(level):
                n_a, n_b = level[idx], level[idx + 1]
                pairs.append((n_a, n_b))
                next_level.append(n_a + n_b)
                idx += 2
            else:
                next_level.append(level[idx])
                idx += 1
        level = next_level
    return pairs


def check_merge_half_capacity(*, n_a: int, n_b: int) -> bool:
    total = n_a + n_b
    padded = _next_power_of_2(total)
    half = padded // 2
    return n_a <= half and n_b <= half


def assert_global_pos_permutation(
    *,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    max_abs_observed: int,
    index_width: int,
    global_row_count: int,
    device: torch.device,
) -> bool:
    keys, pos_width = build_global_mechanism3_full_keys(
        abs_new_acc=abs_new_acc,
        global_flat_indices=global_flat_indices,
        max_abs_observed=max_abs_observed,
        index_width=index_width,
        global_row_count=global_row_count,
        device=device,
    )
    pos_mask = (1 << pos_width) - 1
    decoded = [int(k.item()) & pos_mask for k in keys]
    return sorted(decoded) == list(range(global_row_count))


def assert_merge_half_capacity_for_row_count(row_count: int) -> bool:
    pairs = _merge_tree_pairs(_tile_sizes(row_count))
    return all(check_merge_half_capacity(n_a=a, n_b=b) for a, b in pairs)


def _half_capacity_stage_workspace(
    *,
    keys_a: list[int],
    keys_b: list[int],
    sort_padded_n: int,
    padding_sentinel: int,
) -> list[int]:
    n_a = len(keys_a)
    n_b = len(keys_b)
    half = sort_padded_n // 2
    if n_a > half or n_b > half:
        raise ValueError("half-capacity invariant violated")
    first_half = list(keys_a) + [padding_sentinel] * (half - n_a)
    second_half = [padding_sentinel] * (half - n_b) + list(reversed(keys_b))
    return first_half + second_half


def _merge_staged_via_bitonic_reference(
    workspace: list[int],
    *,
    sort_padded_n: int,
    valid_count: int,
    padding_sentinel: int = PADDING_SENTINEL,
) -> list[int]:
    sorted_ws = bitonic_sort_single_writer_reference(
        workspace,
        sort_padded_n=sort_padded_n,
        padding_sentinel=padding_sentinel,
        max_log2n=WIDER_PROBE_MAX_LOG2N,
    )
    return sorted_ws[:valid_count]


def assert_staging_layout_correctness() -> tuple[bool, bool]:
    """Returns (good_layout_ok, bad_layout_fails).

    ``bad_layout_fails`` is structural: the legacy tail-concat layout is not the
    required half-capacity staging shape (even if a full sort might mask it).
    """

    keys_a = [1, 4]
    keys_b = [2]
    good_ws = _half_capacity_stage_workspace(
        keys_a=keys_a,
        keys_b=keys_b,
        sort_padded_n=4,
        padding_sentinel=PADDING_SENTINEL,
    )
    assert good_ws == [1, 4, PADDING_SENTINEL, 2]
    merged_good = _merge_staged_via_bitonic_reference(
        good_ws, sort_padded_n=4, valid_count=3
    )
    good_ok = merged_good == sorted(keys_a + keys_b)

    bad_ws = [1, 4, 2, PADDING_SENTINEL]
    bad_fails = bad_ws != good_ws
    return good_ok, bad_fails


def assert_probe_width_pow2_for_bucket(*, row_count: int, sort_padded_n: int) -> bool:
    if sort_padded_n <= 0 or (sort_padded_n & (sort_padded_n - 1)) != 0:
        return False
    return sort_padded_n == _next_power_of_2(row_count)


def run_multiblock_step0_assertions(
    *,
    sample_fixture: MultiblockStep0FixtureMeasurement,
    abs_new_acc: torch.Tensor,
    global_flat_indices: torch.Tensor,
    device: torch.device,
) -> MultiblockStep0AssertionResults:
    global_ok = assert_global_pos_permutation(
        abs_new_acc=abs_new_acc,
        global_flat_indices=global_flat_indices,
        max_abs_observed=sample_fixture.max_abs_observed,
        index_width=sample_fixture.index_width,
        global_row_count=sample_fixture.row_count,
        device=device,
    )
    half_cap_ok = all(
        assert_merge_half_capacity_for_row_count(n) for n in REALISTIC_ROW_COUNTS
    )
    staging_ok, bad_staging_fails = assert_staging_layout_correctness()
    probe_ok = all(
        assert_probe_width_pow2_for_bucket(
            row_count=n,
            sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
        )
        for n in REALISTIC_ROW_COUNTS
    )
    return MultiblockStep0AssertionResults(
        global_pos_permutation_ok=global_ok,
        merge_half_capacity_ok=half_cap_ok,
        staging_layout_ok=staging_ok,
        bad_staging_layout_fails=bad_staging_fails,
        probe_width_pow2_ok=probe_ok,
    )


def _log2_ceil(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def _launch_compile_probe(
    *,
    sort_padded_n: int,
    n_rows_sample: int,
    kernel,
    kernel_symbol: str,
    device: torch.device,
) -> CompileProbeMeasurement:
    if kernel is None:
        return CompileProbeMeasurement(
            sort_padded_n=sort_padded_n,
            n_rows_sample=n_rows_sample,
            compile_probe_executed=False,
            compile_ok=False,
            kernel_symbol=kernel_symbol,
            error_detail="triton unavailable",
        )
    try:
        log2n = _log2_ceil(sort_padded_n)
        keys_workspace = torch.full(
            (sort_padded_n,), PADDING_SENTINEL, dtype=torch.int64, device=device
        )
        sorted_keys = torch.empty(sort_padded_n, dtype=torch.int64, device=device)
        kernel[(1,)](
            keys_workspace,
            sorted_keys,
            n_rows_sample,
            SORT_PADDED_N=sort_padded_n,
            PADDING_SENTINEL=PADDING_SENTINEL,
            LOG2N=log2n,
        )
        torch.cuda.synchronize(device)
        return CompileProbeMeasurement(
            sort_padded_n=sort_padded_n,
            n_rows_sample=n_rows_sample,
            compile_probe_executed=True,
            compile_ok=True,
            kernel_symbol=kernel_symbol,
        )
    except Exception as exc:  # noqa: BLE001 — compile probe records failure detail
        return CompileProbeMeasurement(
            sort_padded_n=sort_padded_n,
            n_rows_sample=n_rows_sample,
            compile_probe_executed=True,
            compile_ok=False,
            kernel_symbol=kernel_symbol,
            error_detail=str(exc),
        )


def run_compile_probes(*, device: torch.device | None = None) -> tuple[CompileProbeMeasurement, ...]:
    if not _TRITON_AVAILABLE or not torch.cuda.is_available():
        return (
            CompileProbeMeasurement(
                sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
                n_rows_sample=1280,
                compile_probe_executed=False,
                compile_ok=False,
                kernel_symbol=PROBE_KERNEL_SYMBOL,
                error_detail="cuda/triton unavailable",
            ),
            CompileProbeMeasurement(
                sort_padded_n=OPTIONAL_BASELINE_SORT_PADDED_N,
                n_rows_sample=512,
                compile_probe_executed=False,
                compile_ok=False,
                kernel_symbol="_margin_selection_bitonic_sort_kernel",
                error_detail="cuda/triton unavailable",
            ),
        )
    dev = device or torch.device("cuda")
    probes = [
        _launch_compile_probe(
            sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
            n_rows_sample=1280,
            kernel=_margin_selection_bitonic_sort_probe_kernel_2048,
            kernel_symbol=PROBE_KERNEL_SYMBOL,
            device=dev,
        ),
        _launch_compile_probe(
            sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
            n_rows_sample=1536,
            kernel=_margin_selection_bitonic_sort_probe_kernel_2048,
            kernel_symbol=PROBE_KERNEL_SYMBOL,
            device=dev,
        ),
        _launch_compile_probe(
            sort_padded_n=OPTIONAL_BASELINE_SORT_PADDED_N,
            n_rows_sample=512,
            kernel=_margin_selection_bitonic_sort_kernel,
            kernel_symbol="_margin_selection_bitonic_sort_kernel",
            device=dev,
        ),
    ]
    return tuple(probes)


def run_multiblock_step0_budget_suite(
    *,
    spec: GlobalRateCapSpec | None = None,
    device: torch.device | None = None,
    run_gpu_compile_probe: bool = True,
) -> GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt:
    dev = device or torch.device("cpu")
    cap_spec = spec or GlobalRateCapSpec(cap=1024, step=1)
    measurements: list[MultiblockStep0FixtureMeasurement] = []
    sample_rows = None
    for row_count in REALISTIC_ROW_COUNTS:
        inputs = build_realistic_fixture_inputs(target_row_count=row_count)
        measurement = measure_multiblock_step0_fixture(
            fixture_name=f"realistic_{row_count}",
            inputs=inputs,
            spec=cap_spec,
        )
        measurements.append(measurement)
        if row_count == 1280:
            offsets = tensor_offsets_for_vote_update_states(inputs)
            sample_rows = _device_row_tensors_for_selection(
                inputs, tensor_offsets=offsets, device=dev
            )

    assert sample_rows is not None
    assertion_results = run_multiblock_step0_assertions(
        sample_fixture=measurements[0],
        abs_new_acc=sample_rows["abs_new_acc"],
        global_flat_indices=sample_rows["global_indices"],
        device=dev,
    )
    compile_probes = (
        run_compile_probes(device=torch.device("cuda"))
        if run_gpu_compile_probe and _TRITON_AVAILABLE and torch.cuda.is_available()
        else (
            CompileProbeMeasurement(
                sort_padded_n=WIDER_SINGLE_BLOCK_SORT_PADDED_N,
                n_rows_sample=1280,
                compile_probe_executed=False,
                compile_ok=False,
                kernel_symbol=PROBE_KERNEL_SYMBOL,
                error_detail="compile probe skipped (cpu-only validation)",
            ),
            CompileProbeMeasurement(
                sort_padded_n=OPTIONAL_BASELINE_SORT_PADDED_N,
                n_rows_sample=512,
                compile_probe_executed=False,
                compile_ok=False,
                kernel_symbol="_margin_selection_bitonic_sort_kernel",
                error_detail="compile probe skipped (cpu-only validation)",
            ),
        )
    )
    receipt = build_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(
        fixture_measurements=tuple(measurements),
        compile_probes=compile_probes,
        assertion_results=assertion_results,
    )
    validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt(receipt)
    return receipt


__all__ = [
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_MULTIBLOCK_STEP0_BUDGET_SCHEMA_VERSION",
    "GlobalRateCapMarginSelectionMultiblockStep0BudgetReceipt",
    "MultiblockStep0Decision",
    "REALISTIC_ROW_COUNTS",
    "WIDER_SINGLE_BLOCK_SORT_PADDED_N",
    "assert_global_pos_permutation",
    "assert_merge_half_capacity_for_row_count",
    "assert_probe_width_pow2_for_bucket",
    "assert_staging_layout_correctness",
    "build_global_mechanism3_full_keys",
    "build_global_rate_cap_margin_selection_multiblock_step0_budget_receipt",
    "build_realistic_fixture_inputs",
    "check_merge_half_capacity",
    "measure_multiblock_step0_fixture",
    "run_compile_probes",
    "run_multiblock_step0_assertions",
    "run_multiblock_step0_budget_suite",
    "validate_global_rate_cap_margin_selection_multiblock_step0_budget_receipt",
]
