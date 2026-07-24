"""Frequency-weighted B-only phase timing reducer + classifier (Branch-A).

Pure accounting: invocations × median → window_contribution → normalized/step.
Dominator = argmax(window_contribution_ms). Raw-median selection is rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

PHASE_NAMES = (
    "process_pre",
    "close_before",
    "roll",
    "episode_snapshot",
    "publish",
    "finalize",
)

# Frozen cadence: publish when step % 25 == 0 or step == steps (loop contract).
PUBLISH_CADENCE_STEPS = 25
DEFAULT_FINALIZE_INVOCATIONS_PER_WINDOW = 1


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseTimer:
    """Optional per-phase sampler. When disabled, context managers are true no-ops."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if name not in PHASE_NAMES:
            raise ValueError(f"unknown phase {name!r}; expected one of {PHASE_NAMES}")
        cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.samples[name].append((time.perf_counter() - t0) * 1000.0)


def expected_publish_invocations(
    steps: int, *, cadence: int = PUBLISH_CADENCE_STEPS
) -> int:
    """Match screen_execution_loop publish gate: step % cadence == 0 or step == steps."""
    n = int(steps)
    if n <= 0:
        raise ValueError(f"steps must be >= 1, got {steps!r}")
    c = int(cadence)
    return sum(1 for step in range(1, n + 1) if step % c == 0 or step == n)


def invocations_per_window_for_phase(
    name: str,
    *,
    steps: int,
    publish_invocations: int | None = None,
    finalize_invocations: int = DEFAULT_FINALIZE_INVOCATIONS_PER_WINDOW,
) -> int:
    if name in ("process_pre", "close_before", "roll", "episode_snapshot"):
        return int(steps)
    if name == "publish":
        if publish_invocations is None:
            return expected_publish_invocations(steps)
        return int(publish_invocations)
    if name == "finalize":
        return int(finalize_invocations)
    raise ValueError(f"unknown phase {name!r}")


def build_phase_row(
    *,
    name: str,
    samples_ms: list[float],
    steps: int,
    publish_invocations: int | None = None,
    finalize_invocations: int = DEFAULT_FINALIZE_INVOCATIONS_PER_WINDOW,
) -> dict[str, Any]:
    if not samples_ms:
        raise ValueError(f"phase {name!r} has no samples (UNPRICEABLE)")
    if publish_invocations is None:
        publish_invocations = expected_publish_invocations(steps)
    inv = invocations_per_window_for_phase(
        name,
        steps=steps,
        publish_invocations=publish_invocations,
        finalize_invocations=finalize_invocations,
    )
    observed = len(samples_ms)
    if observed != int(inv):
        raise ValueError(
            f"phase {name!r} UNPRICEABLE: observed_samples={observed} "
            f"expected_invocations={inv} (steps={steps})"
        )
    median = float(statistics.median(samples_ms))
    window = float(median) * float(inv)
    normalized = window / float(steps)
    return {
        "name": name,
        "invocations_per_window": int(inv),
        "median_ms_per_invocation": median,
        "window_contribution_ms": window,
        "normalized_ms_per_step": normalized,
        "samples_ms": [float(x) for x in samples_ms],
        "n_samples": observed,
        "observed_matches_expected": True,
    }


def frequency_weighted_summary(
    samples_by_phase: Mapping[str, list[float]],
    *,
    steps: int,
    publish_invocations: int | None = None,
    finalize_invocations: int = DEFAULT_FINALIZE_INVOCATIONS_PER_WINDOW,
) -> dict[str, Any]:
    if publish_invocations is None:
        publish_invocations = expected_publish_invocations(steps)
    rows: list[dict[str, Any]] = []
    for name in PHASE_NAMES:
        if name not in samples_by_phase or not samples_by_phase[name]:
            raise ValueError(f"missing/empty samples for phase {name!r} (UNPRICEABLE)")
        rows.append(
            build_phase_row(
                name=name,
                samples_ms=list(samples_by_phase[name]),
                steps=steps,
                publish_invocations=publish_invocations,
                finalize_invocations=finalize_invocations,
            )
        )
    return {
        "steps": int(steps),
        "publish_invocations_expected": int(publish_invocations),
        "finalize_invocations_expected": int(finalize_invocations),
        "phases": rows,
        "sum_window_contribution_ms": float(
            sum(r["window_contribution_ms"] for r in rows)
        ),
        "sum_normalized_ms_per_step": float(
            sum(r["normalized_ms_per_step"] for r in rows)
        ),
    }


def select_dominator_by_window_contribution(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Binding dominator rule: argmax(window_contribution_ms)."""
    phases = list(summary["phases"])
    if not phases:
        raise ValueError("no phases")
    return max(phases, key=lambda r: float(r["window_contribution_ms"]))


def select_dominator_by_raw_median(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Forbidden classifier — exposed only so tests can assert rejection."""
    raise RuntimeError(
        "raw-median dominator selection is forbidden; "
        "use select_dominator_by_window_contribution"
    )


def normalized_reduction_ms_per_step(
    *,
    pre_row: Mapping[str, Any],
    post_row: Mapping[str, Any],
) -> float:
    return float(pre_row["normalized_ms_per_step"]) - float(
        post_row["normalized_ms_per_step"]
    )


def window_reduction_ms(
    *,
    pre_row: Mapping[str, Any],
    post_row: Mapping[str, Any],
) -> float:
    return float(pre_row["window_contribution_ms"]) - float(
        post_row["window_contribution_ms"]
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomic exclusive create — fails if path already exists (PRE immutability)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(target), flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {target}"
        ) from exc
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
    except Exception:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return sha256_bytes(body)


def assert_pre_immutable(pre_path: str | Path, expected_sha256: str) -> None:
    live = sha256_file(pre_path)
    if live != expected_sha256:
        raise AssertionError(
            f"PRE artifact mutated: expected {expected_sha256}, got {live}"
        )
