"""Regression: scalar JSON lines in run.log must not crash hygiene counters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.w7_launch_hygiene import (
    count_bounded_steps_starts_in_run_log,
)


def _bounded_steps_start_line() -> str:
    return json.dumps(
        {"event": "start", "phase": "bounded_steps", "step": 0},
        sort_keys=True,
    )


def _scalar_contamination_lines() -> list[str]:
    return [
        json.dumps("runtime-resource-failure"),
        json.dumps(42),
        "null",
    ]


def test_count_bounded_steps_starts_mixed_scalar_contamination(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    lines = _scalar_contamination_lines() + [_bounded_steps_start_line()]
    run_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert count_bounded_steps_starts_in_run_log(run_log) == 1


def test_count_bounded_steps_starts_scalar_only(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text("\n".join(_scalar_contamination_lines()) + "\n", encoding="utf-8")

    assert count_bounded_steps_starts_in_run_log(run_log) == 0


def test_count_bounded_steps_starts_missing_file(tmp_path: Path) -> None:
    assert count_bounded_steps_starts_in_run_log(tmp_path / "missing.log") == 0


def test_scale_smoke_hygiene_path_uses_helper_on_fixture(tmp_path: Path) -> None:
    """Scale-smoke assert path: run.log under cheap_complete_w7_on scratch."""
    scratch = tmp_path / "scale_smoke" / "cheap_complete_w7_on"
    scratch.mkdir(parents=True)
    run_log = scratch / "run.log"
    run_log.write_text(
        "\n".join(_scalar_contamination_lines() + [_bounded_steps_start_line()]) + "\n",
        encoding="utf-8",
    )

    assert count_bounded_steps_starts_in_run_log(run_log) == 1


@pytest.mark.parametrize("arm", ["int16_oracle_flag_off", "w7_dense_acc_treatment"])
def test_per_arm_hygiene_path_uses_helper_on_fixture(tmp_path: Path, arm: str) -> None:
    arm_root = tmp_path / arm
    arm_root.mkdir(parents=True)
    run_log = arm_root / "run.log"
    run_log.write_text(
        "\n".join(_scalar_contamination_lines() + [_bounded_steps_start_line()]) + "\n",
        encoding="utf-8",
    )

    assert count_bounded_steps_starts_in_run_log(run_log) == 1
