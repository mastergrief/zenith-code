"""Fork-2 scale-smoke timing helpers + pure STOP/GO projection reducers."""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples))


def time_phase(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    sync: Callable[[], None] = cuda_sync,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
        sync()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"samples_ms": samples, "median_ms": median_ms(samples), "n": len(samples)}


def time_phase_with_setup(
    setup_fn: Callable[[], None],
    work_fn: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    sync: Callable[[], None] = cuda_sync,
) -> dict[str, Any]:
    """Reseed/setup OUTSIDE the timed region; time work_fn only."""
    for _ in range(int(warmup)):
        setup_fn()
        sync()
        work_fn()
        sync()
    samples: list[float] = []
    for _ in range(int(repeats)):
        setup_fn()
        sync()
        t0 = time.perf_counter()
        work_fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"samples_ms": samples, "median_ms": median_ms(samples), "n": len(samples)}


def project_observer_windows(
    *,
    select_ms: float,
    lifecycle_ms: float,
    update_full_ms: float,
    update_index_ms: float,
    writeback_ms: float,
    finalize_ms: float,
    steps: int,
) -> dict[str, float]:
    per_step_full = float(select_ms + lifecycle_ms + update_full_ms + writeback_ms)
    per_step_idx = float(select_ms + lifecycle_ms + update_index_ms + writeback_ms)
    window_full = per_step_full * int(steps) + float(finalize_ms)
    window_idx = per_step_idx * int(steps) + float(finalize_ms)
    return {
        "per_step_full_ms": per_step_full,
        "per_step_index_ms": per_step_idx,
        "window_full_ms": window_full,
        "window_index_ms": window_idx,
    }


def decide_stop_go(
    *,
    window_full_ms: float,
    window_index_ms: float,
    need_s: float,
    r6_status: str,
    full_mask_transfer: bool,
    publish_ok: bool,
) -> dict[str, Any]:
    index_only_correct = str(r6_status) == "PASS"
    projection_uses = "index_only" if index_only_correct else "full_h2d"
    go_idx = (window_index_ms / 1000.0) <= float(need_s) and not full_mask_transfer
    go_full = (window_full_ms / 1000.0) <= float(need_s) and not full_mask_transfer
    if not publish_ok:
        go_idx = False
        go_full = False
    if projection_uses == "index_only":
        stop_go = "GO" if go_idx else "RED_STOP"
        projected_s = window_index_ms / 1000.0
    else:
        stop_go = "GO" if go_full else "RED_STOP"
        projected_s = window_full_ms / 1000.0
    return {
        "index_only_correct": index_only_correct,
        "projection_uses": projection_uses,
        "go_idx": go_idx,
        "go_full": go_full,
        "disagreement_on_stop_go": go_idx != go_full,
        "stop_go": stop_go,
        "projected_observer_delta_s": projected_s,
    }


def lifecycle_projected_ms(
    *,
    process_ms: float,
    close_ms: float,
    roll_ms: float,
    full_sequence_ms: float,
) -> dict[str, float]:
    phase_sum = float(process_ms + close_ms + roll_ms)
    projected = max(float(full_sequence_ms), phase_sum)
    return {
        "lifecycle_sum_of_phase_medians": phase_sum,
        "lifecycle_full_per_step_sequence": float(full_sequence_ms),
        "lifecycle_projected": projected,
    }


def writeback_projected_ms(*medians_ms: float) -> dict[str, float | str]:
    vals = [float(x) for x in medians_ms]
    if not vals:
        raise ValueError("need at least one writeback median")
    # Caller labels order: one_arm, cross_arm_full_topk, ...
    labels = ("one_arm", "cross_arm_full_topk")
    idx = max(range(len(vals)), key=lambda i: vals[i])
    src = labels[idx] if idx < len(labels) else f"variant_{idx}"
    return {"writeback_projection_median_ms": vals[idx], "writeback_projection_uses": src}
