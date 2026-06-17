"""R5 wall-time gate — implemented dense carrier apply at prod-shaped density."""
from __future__ import annotations

import gc
import random
import time
import tracemalloc
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

STEP_BUDGET_S = 150.0
G_WALL_PROD_32KEY_TARGET_S = 30.0
G_RSS_PROD_VMRSS_DELTA_MB = 100.0
G_RSS_PROD_TRACEMALLOC_PEAK_MB = 100.0
PROD_NUMEL = 925_000
PROD_EVENTS = 601_250
KEYS_PER_STEP = 32


def _vmrss_mb() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found in /proc/self/status")


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(10, -127, 127, 4, 1.0)


def _make_state(numel: int) -> BoundedDeltaAccumulatorState:
    rng = random.Random(17)
    hot = tuple(sorted(rng.sample(range(numel), min(32, numel))))
    return BoundedDeltaAccumulatorState(
        logical_shape=(numel,),
        cold_default_value=0,
        hot_exact_indices=hot,
        hot_exact_values=tuple(5 for _ in hot),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="r5.walltime",
        raw_arrays_included=False,
    )


def _make_carrier(numel: int, events: int, *, seed: int) -> SparseVoteEvents:
    rng = random.Random(seed)
    events = min(events, numel)
    indices = torch.tensor(sorted(rng.sample(range(numel), events)), dtype=torch.int64)
    values = torch.randint(-5, 6, (events,), dtype=torch.int16)
    values[values == 0] = 1
    return SparseVoteEvents(indices=indices, values=values)


def test_r5_apply_walltime_prod_shaped_32key() -> None:
    gc.collect()
    q_levels = torch.zeros(PROD_NUMEL, dtype=torch.int8)
    t0 = time.perf_counter()
    for key_idx in range(KEYS_PER_STEP):
        execute_direct_bounded_local_vote_update_candidate(
            state_key=f"r5.prod.{key_idx}",
            q_levels=q_levels,
            bounded_accumulator=_make_state(PROD_NUMEL),
            sparse_vote_events=_make_carrier(PROD_NUMEL, PROD_EVENTS, seed=170 + key_idx),
            vote_spec=_spec(),
        )
    step32_s = time.perf_counter() - t0

    gc.collect()
    max_tracemalloc_peak_mb = 0.0
    max_vmrss_delta_mb = 0.0
    for key_idx in range(KEYS_PER_STEP):
        gc.collect()
        rss_before = _vmrss_mb()
        tracemalloc.start()
        execute_direct_bounded_local_vote_update_candidate(
            state_key=f"r5.rss.{key_idx}",
            q_levels=torch.zeros(PROD_NUMEL, dtype=torch.int8),
            bounded_accumulator=_make_state(PROD_NUMEL),
            sparse_vote_events=_make_carrier(PROD_NUMEL, PROD_EVENTS, seed=270 + key_idx),
            vote_spec=_spec(),
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _vmrss_mb()
        max_tracemalloc_peak_mb = max(max_tracemalloc_peak_mb, peak / (1024.0 * 1024.0))
        max_vmrss_delta_mb = max(max_vmrss_delta_mb, rss_after - rss_before)

    assert step32_s < STEP_BUDGET_S, f"step32 {step32_s:.3f}s exceeds {STEP_BUDGET_S}s budget"
    assert step32_s <= G_WALL_PROD_32KEY_TARGET_S, (
        f"step32 {step32_s:.3f}s exceeds comfort target {G_WALL_PROD_32KEY_TARGET_S}s"
    )
    assert max_vmrss_delta_mb < G_RSS_PROD_VMRSS_DELTA_MB, (
        f"VmRSS delta {max_vmrss_delta_mb:.1f}MB exceeds {G_RSS_PROD_VMRSS_DELTA_MB}MB"
    )
    assert max_tracemalloc_peak_mb < G_RSS_PROD_TRACEMALLOC_PEAK_MB, (
        f"tracemalloc peak {max_tracemalloc_peak_mb:.1f}MB exceeds "
        f"{G_RSS_PROD_TRACEMALLOC_PEAK_MB}MB (secondary)"
    )
