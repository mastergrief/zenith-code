"""Slice-1 telemetry honesty + confirmation RC capture tests."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import torch

from scripts.apply_arc2b_slice5_discovery_h25_launch_packet_v2 import (
    honest_confirmation_launch_rc,
)
from scripts.hrm_text_158_arc2b_slice5_discovery_live_postrun import (
    resolve_operational_ok,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    C2PhaseTimeout,
    PhaseProgress,
    is_terminal_liveness_breach,
)


def test_healthy_enter_does_not_stamp_liveness_failure(tmp_path: Path) -> None:
    last_active = tmp_path / "last_active_phase.json"
    progress = PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=30.0,
        last_active_phase_path=last_active,
        arm_faulthandler_timer=False,
    )
    with progress.phase("sparse_cap_apply", step=1):
        pass
    payload = json.loads(last_active.read_text(encoding="utf-8"))
    assert payload["guard_event"] == "cleared"
    assert payload["liveness_failure"] is False
    assert "failure_class" not in payload
    assert is_terminal_liveness_breach(payload) is False


def test_operational_ok_true_on_clean_cleared_last_active_phase(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scratch = run_root / "d_recompute_window_diagnostic"
    prelaunch = run_root / "prelaunch"
    scratch.mkdir(parents=True)
    prelaunch.mkdir(parents=True)
    (scratch / "last_active_phase.json").write_text(
        json.dumps(
            {
                "guard_event": "cleared",
                "liveness_failure": False,
                "phase": "bounded_steps",
            }
        ),
        encoding="utf-8",
    )
    (scratch / "receipt.json").write_text(
        json.dumps({"steps_completed": 25, "d_recompute_window_instrumentation_enabled": True}),
        encoding="utf-8",
    )
    (scratch / "recompute_window_log.jsonl").write_text(
        "\n".join(
            json.dumps({"replay_constants": {"decay_numerator": 1, "decay_denominator": 2}})
            for _ in range(25)
        )
        + "\n",
        encoding="utf-8",
    )
    (scratch / "live_carrier_snapshot.jsonl").write_text(
        json.dumps(
            {
                "live_carrier_bytes_exact": True,
                "events_bytes": 1,
                "backlog_bytes": 1,
                "hot_exact_bytes": 1,
                "metadata_bytes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (prelaunch / "confirmation_launch_rc.txt").write_text("0\n", encoding="utf-8")
    ok, details = resolve_operational_ok(run_root=run_root, steps_expected=25)
    assert ok is True
    assert details["reason"] == "ok"


def test_operational_ok_false_only_on_terminal_breach_not_enter(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    scratch = run_root / "d_recompute_window_diagnostic"
    prelaunch = run_root / "prelaunch"
    scratch.mkdir(parents=True)
    prelaunch.mkdir(parents=True)
    (scratch / "last_active_phase.json").write_text(
        json.dumps(
            {
                "guard_event": "enter",
                "phase": "sparse_cap_apply",
                "step": 12,
            }
        ),
        encoding="utf-8",
    )
    (scratch / "receipt.json").write_text(
        json.dumps({"steps_completed": 11, "d_recompute_window_instrumentation_enabled": True}),
        encoding="utf-8",
    )
    (scratch / "recompute_window_log.jsonl").write_text(
        json.dumps({"replay_constants": {"decay_numerator": 1, "decay_denominator": 2}}) + "\n",
        encoding="utf-8",
    )
    (scratch / "live_carrier_snapshot.jsonl").write_text(
        json.dumps(
            {
                "live_carrier_bytes_exact": True,
                "events_bytes": 1,
                "backlog_bytes": 1,
                "hot_exact_bytes": 1,
                "metadata_bytes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (prelaunch / "confirmation_launch_rc.txt").write_text("0\n", encoding="utf-8")
    ok, details = resolve_operational_ok(run_root=run_root, steps_expected=25)
    assert ok is False
    assert details["reason"] != "liveness_failure"


def test_heartbeat_invokes_stale_active_phase_guard(tmp_path: Path) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    last_active = tmp_path / "last_active_phase.json"
    progress = PhaseProgress(
        enabled=True,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=1.0,
        phase_heartbeat_interval_seconds=0.1,
        last_active_phase_path=last_active,
        arm_faulthandler_timer=False,
        clock=clock,
    )

    with pytest.raises(C2PhaseTimeout):
        with progress.phase("sparse_cap_apply", step=2):
            clock.now = 2.0
            time.sleep(0.25)

    payload = json.loads(last_active.read_text(encoding="utf-8"))
    assert payload["guard_event"] == "breach"
    assert payload["liveness_failure"] is True


def test_honest_confirmation_launch_rc_exits_with_probe_rc(tmp_path: Path) -> None:
    rc_path = tmp_path / "confirmation_launch_rc.txt"
    command = (
        "bash -c 'set +e; CONFIRMATION_RC=0; false; CONFIRMATION_RC=${PIPESTATUS[0]}; "
        f"printf \"%s\\n\" \"$CONFIRMATION_RC\" > {rc_path}; exit 0'"
    )
    fixed = honest_confirmation_launch_rc(command)
    assert "exit $CONFIRMATION_RC'" in fixed
    assert fixed.endswith("exit $CONFIRMATION_RC'")
    proc = subprocess.run(["bash", "-c", fixed], check=False)
    assert proc.returncode != 0
    assert rc_path.read_text(encoding="utf-8").strip() != "0"
