#!/usr/bin/env python3
"""R5 apply cost-model — measures implemented dense carrier path (step32 gate)."""
from __future__ import annotations

import argparse
import gc
import random
import sys
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
KEYS_PER_STEP = 32
G_WALL_PROD_32KEY_TARGET_S = 30.0
G_RSS_PROD_MB = 100.0
PROD_NUMEL = 925_000
PROD_EVENTS = 601_250


def _vmrss_mb() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found in /proc/self/status")


def vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(10, -127, 127, 4, 1.0)


def make_state(numel: int) -> BoundedDeltaAccumulatorState:
    rng = random.Random(17)
    hot = tuple(sorted(rng.sample(range(numel), min(32, numel))))
    return BoundedDeltaAccumulatorState(
        logical_shape=(numel,),
        cold_default_value=0,
        hot_exact_indices=hot,
        hot_exact_values=tuple(5 for _ in hot),
        cold_exception_indices=(),
        cold_exception_values=(),
        candidate_name="r5_cost_model",
        raw_arrays_included=False,
    )


def make_carrier(numel: int, events: int, *, seed: int) -> SparseVoteEvents:
    rng = random.Random(seed)
    events = min(events, numel)
    indices = torch.tensor(sorted(rng.sample(range(numel), events)), dtype=torch.int64)
    values = torch.randint(-5, 6, (events,), dtype=torch.int16)
    values[values == 0] = 1
    return SparseVoteEvents(indices=indices, values=values)


def bench_implemented_step32(numel: int, events: int) -> tuple[float, float, float]:
    total = 0.0
    for key_idx in range(KEYS_PER_STEP):
        carrier = make_carrier(numel, events, seed=17 + key_idx)
        state = make_state(numel)
        gc.collect()
        t0 = time.perf_counter()
        execute_direct_bounded_local_vote_update_candidate(
            state_key=f"k{key_idx}",
            q_levels=torch.zeros(numel, dtype=torch.int8),
            bounded_accumulator=state,
            sparse_vote_events=carrier,
            vote_spec=vote_spec(),
        )
        total += time.perf_counter() - t0

    max_tracemalloc_peak_mb = 0.0
    max_vmrss_delta_mb = 0.0
    for key_idx in range(KEYS_PER_STEP):
        carrier = make_carrier(numel, events, seed=17 + key_idx)
        state = make_state(numel)
        gc.collect()
        rss_before = _vmrss_mb()
        tracemalloc.start()
        execute_direct_bounded_local_vote_update_candidate(
            state_key=f"k{key_idx}",
            q_levels=torch.zeros(numel, dtype=torch.int8),
            bounded_accumulator=state,
            sparse_vote_events=carrier,
            vote_spec=vote_spec(),
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _vmrss_mb()
        max_tracemalloc_peak_mb = max(max_tracemalloc_peak_mb, peak / (1024.0 * 1024.0))
        max_vmrss_delta_mb = max(max_vmrss_delta_mb, rss_after - rss_before)
    return total, max_tracemalloc_peak_mb, max_vmrss_delta_mb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=PROD_NUMEL)
    parser.add_argument("--events", type=int, default=PROD_EVENTS)
    args = parser.parse_args()

    step32_s, tracemalloc_peak_mb, vmrss_delta_mb = bench_implemented_step32(
        args.numel, args.events
    )
    gate_pass = step32_s < STEP_BUDGET_S
    target_pass = step32_s <= G_WALL_PROD_32KEY_TARGET_S
    rss_pass = vmrss_delta_mb < G_RSS_PROD_MB

    print("R5_APPLY_COST_MODEL_IMPLEMENTED")
    print(f"numel={args.numel} events={args.events} keys={KEYS_PER_STEP}")
    print(f"implemented_step32_s={step32_s:.3f}")
    print(f"implemented_tracemalloc_peak_mb={tracemalloc_peak_mb:.3f}")
    print(f"implemented_vmrss_delta_mb={vmrss_delta_mb:.3f}")
    print(
        f"G-WALL-PROD-32KEY={'PASS' if gate_pass else 'FAIL'} "
        f"limit<{STEP_BUDGET_S}s target<={G_WALL_PROD_32KEY_TARGET_S}s "
        f"target_pass={target_pass}"
    )
    print(
        f"G-RSS-PROD={'PASS' if rss_pass else 'FAIL'} "
        f"vmrss_delta_limit<{G_RSS_PROD_MB}MB "
        f"(tracemalloc_peak_mb={tracemalloc_peak_mb:.3f} secondary)"
    )

    ok = gate_pass and target_pass and rss_pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
