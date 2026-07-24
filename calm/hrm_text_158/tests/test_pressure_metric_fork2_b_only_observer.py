"""Branch-A frequency-weighted phase timing + PRE immutability + OFF-path tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_phase_timing import (
    PHASE_NAMES,
    PhaseTimer,
    assert_pre_immutable,
    build_phase_row,
    expected_publish_invocations,
    frequency_weighted_summary,
    select_dominator_by_raw_median,
    select_dominator_by_window_contribution,
    write_json_exclusive,
)
from calm.hrm_text_158.native_full_stack.screen_execution_loop import _run_phase

ROOT = Path(__file__).resolve().parents[3]


def _full_samples(*, steps: int) -> dict[str, list[float]]:
    pub_n = expected_publish_invocations(steps)
    return {
        "process_pre": [10.0] * steps,
        "close_before": [8.0] * steps,
        "roll": [2.0] * steps,
        "episode_snapshot": [1.0] * steps,
        "publish": [20.0] * pub_n,
        "finalize": [75.0],
    }


def test_frequency_weighted_accounting_mixed_invocations() -> None:
    # steps=5: process ×5 @ 10ms → 50 window / 10 per-step
    # finalize ×1 @ 75ms → 75 window / 15 per-step (raw median would wrongly win)
    steps = 5
    samples = _full_samples(steps=steps)
    samples["finalize"] = [75.0]
    summary = frequency_weighted_summary(samples, steps=steps)
    by_name = {r["name"]: r for r in summary["phases"]}
    assert by_name["process_pre"]["invocations_per_window"] == 5
    assert by_name["finalize"]["invocations_per_window"] == 1
    assert by_name["process_pre"]["window_contribution_ms"] == pytest.approx(50.0)
    assert by_name["finalize"]["window_contribution_ms"] == pytest.approx(75.0)
    assert by_name["process_pre"]["normalized_ms_per_step"] == pytest.approx(10.0)
    assert by_name["finalize"]["normalized_ms_per_step"] == pytest.approx(15.0)
    assert summary["sum_window_contribution_ms"] == pytest.approx(
        50 + 40 + 10 + 5 + 20 + 75
    )


def test_dominator_uses_window_contribution_not_raw_median() -> None:
    steps = 25
    samples = {
        "process_pre": [57.0] * steps,  # 57*25=1425 window
        "close_before": [10.0] * steps,
        "roll": [5.0] * steps,
        "episode_snapshot": [1.0] * steps,
        "publish": [20.0],
        "finalize": [75.0],  # raw median highest, but only 75 window
    }
    summary = frequency_weighted_summary(samples, steps=steps)
    dom = select_dominator_by_window_contribution(summary)
    assert dom["name"] == "process_pre"
    assert dom["window_contribution_ms"] == pytest.approx(57.0 * 25)


def test_raw_median_dominator_rejected() -> None:
    summary = frequency_weighted_summary(_full_samples(steps=5), steps=5)
    with pytest.raises(RuntimeError, match="raw-median"):
        select_dominator_by_raw_median(summary)


def test_phase_timer_disabled_is_noop() -> None:
    t = PhaseTimer(enabled=False)
    with t.time("process_pre"):
        pass
    assert t.samples == {}


def test_pre_exclusive_create_and_immutability(tmp_path: Path) -> None:
    path = tmp_path / "PRE.json"
    sha = write_json_exclusive(path, {"ok": True})
    assert_pre_immutable(path, sha)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_exclusive(path, {"ok": False})
    path.write_text(json.dumps({"ok": "mutated"}) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="PRE artifact mutated"):
        assert_pre_immutable(path, sha)


def test_build_phase_row_publish_once_per_window() -> None:
    row = build_phase_row(name="publish", samples_ms=[21.0], steps=25)
    assert row["invocations_per_window"] == 1
    assert row["window_contribution_ms"] == pytest.approx(21.0)
    assert row["normalized_ms_per_step"] == pytest.approx(21.0 / 25.0)


def test_observed_count_mismatch_missing_sample_unpriceable() -> None:
    samples = _full_samples(steps=25)
    samples["close_before"] = samples["close_before"][:-1]  # 24 vs expected 25
    with pytest.raises(ValueError, match=r"close_before.*UNPRICEABLE"):
        frequency_weighted_summary(samples, steps=25)


def test_observed_count_mismatch_extra_sample_unpriceable() -> None:
    samples = _full_samples(steps=25)
    samples["finalize"] = [1.0, 2.0]  # expected 1
    with pytest.raises(ValueError, match=r"finalize.*UNPRICEABLE"):
        frequency_weighted_summary(samples, steps=25)


def test_non_25_steps_publish_count_priced_correctly() -> None:
    # steps=50 → publish at 25 and 50 → 2 invocations
    steps = 50
    assert expected_publish_invocations(steps) == 2
    samples = _full_samples(steps=steps)
    summary = frequency_weighted_summary(samples, steps=steps)
    by_name = {r["name"]: r for r in summary["phases"]}
    assert by_name["publish"]["invocations_per_window"] == 2
    assert by_name["publish"]["n_samples"] == 2
    assert by_name["process_pre"]["invocations_per_window"] == 50
    # steps=10 → publish only at final step
    assert expected_publish_invocations(10) == 1
    summary10 = frequency_weighted_summary(_full_samples(steps=10), steps=10)
    assert summary10["publish_invocations_expected"] == 1


def test_non_25_steps_publish_count_mismatch_unpriceable() -> None:
    # steps=50 expects 2 publish samples; supplying 1 must hard-reject
    steps = 50
    samples = _full_samples(steps=steps)
    samples["publish"] = [20.0]  # only one
    with pytest.raises(ValueError, match=r"publish.*UNPRICEABLE"):
        frequency_weighted_summary(samples, steps=steps)


def test_run_phase_none_is_direct_call_zero_timer_sync() -> None:
    """Default OFF: direct fn() — no timer.time entry, no cuda_sync from timer path."""
    sync_calls = {"n": 0}

    def boom_sync() -> None:
        sync_calls["n"] += 1
        raise AssertionError("cuda_sync must not run on timer=None OFF path")

    class BoomTimer:
        def time(self, name: str):  # pragma: no cover
            raise AssertionError(
                f"timer.time({name!r}) must not run when timer is None"
            )

    state = {"n": 0}

    def fn() -> int:
        state["n"] += 1
        return 42

    with mock.patch(
        "calm.hrm_text_158.native_full_stack.pressure_metric_fork2_phase_timing.cuda_sync",
        boom_sync,
    ):
        out = _run_phase(None, "process_pre", fn)
    assert out == 42
    assert state["n"] == 1
    assert sync_calls["n"] == 0

    # Identity vs bare call (same side effects / return)
    left: list[int] = []
    right: list[int] = []
    assert _run_phase(None, "close_before", lambda: left.append(7) or 7) == 7
    assert (lambda: right.append(7) or 7)() == 7
    assert left == right == [7]

    # Explicit: BoomTimer is never consulted on the None branch
    _run_phase(None, "finalize", lambda: None)
    with pytest.raises(AssertionError, match="must not run"):
        _run_phase(BoomTimer(), "finalize", lambda: None)  # type: ignore[arg-type]


def test_run_phase_with_timer_records_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.pressure_metric_fork2_phase_timing.cuda_sync",
        lambda: None,
    )
    t = PhaseTimer(enabled=True)
    _run_phase(t, "process_pre", lambda: None)
    assert len(t.samples["process_pre"]) == 1


def test_warmup_default_none_passthrough_to_loop() -> None:
    """C1: default-None through warmup entrypoint reaches run_train_loop unchanged."""
    import calm.hrm_text_158.native_full_stack.pressure_metric_warmup_runtime as warm

    captured: dict[str, Any] = {}
    tiny = torch.zeros(2, 2, dtype=torch.int16)

    def fake_loop(**kwargs):  # type: ignore[no-untyped-def]
        captured["phase_timer"] = kwargs.get("phase_timer", "MISSING")
        return {
            "acc": {"w": tiny.clone()},
            "episode_start": {"w": torch.zeros_like(tiny, dtype=torch.int32)},
            "flip_count": {"w": torch.zeros_like(tiny, dtype=torch.int32)},
            "q_levels": {"w": tiny.clone()},
            "lifetimes": [],
            "credited_mass": 0,
            "n_flips": 0,
            "q_changed_count": 0,
            "n_applied_drains": 0,
            "excluded_hit_count": 0,
            "H_trajectory": [],
            "train_route_counters": {},
            "selection_frames": [],
            "sync_inventory_steps": [],
        }

    fake_rt = {
        "m": object(),
        "tok": object(),
        "eligible": ["w"],
        "q_levels": {"w": tiny.clone()},
        "frozen_scales": {"w": 1.0},
        "max_seq_len": 32,
        "sha_before": "parentsha",
        "scale_sha_before": "scalesha",
        "q_sha_before": "qsha",
    }

    with (
        mock.patch.object(warm, "run_hotpath_warmup_throwaway", return_value={}),
        mock.patch.object(warm, "load_and_patch_runtime", return_value=fake_rt),
        mock.patch.object(warm, "assert_q_levels_coupled"),
        mock.patch.object(
            warm,
            "build_phase1_probe_sets",
            return_value={"acquisition": [], "retention": []},
        ),
        mock.patch.object(warm, "build_pool", return_value=[]),
        mock.patch.object(warm, "cuda_sync"),
        mock.patch.object(warm, "run_train_loop", side_effect=fake_loop),
        mock.patch.object(warm, "sha256_file", return_value="parentsha"),
        mock.patch.object(warm, "hash_scale_dict", return_value="scalesha"),
        mock.patch.object(warm, "entropy_bits", return_value=0.0),
        mock.patch.object(warm, "lifetime_censored_frac", return_value=0.0),
    ):
        # phase_timer omitted → default None through production entrypoint
        warm.run_one_diagnostic_loop(
            ckpt_path="unused.pt",
            device="cpu",
            steps=1,
            batch=1,
            topk=1,
            telemetry=False,
            skip_probes=True,
            seed=0,
            warmup_enable=False,
        )
    assert captured.get("phase_timer") is None


def test_loop_has_no_nullcontext_off_path_seam() -> None:
    src = (
        ROOT / "calm/hrm_text_158/native_full_stack/screen_execution_loop.py"
    ).read_text(encoding="utf-8")
    assert "nullcontext" not in src
    assert "def _run_phase(" in src
    assert "phase_timer: Any | None = None" in src
    # Default OFF must not import the diagnostic timer module.
    assert "pressure_metric_fork2_phase_timing" not in src


def test_all_phase_names_covered() -> None:
    assert set(PHASE_NAMES) == {
        "process_pre",
        "close_before",
        "roll",
        "episode_snapshot",
        "publish",
        "finalize",
    }
