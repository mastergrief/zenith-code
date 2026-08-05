"""Bounded tests for --global-horizon four-surface plumb (A′ slice-3 phase-1).

Does NOT append to the 4336-line probe test god file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe_module
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_arg_parser,
    run_c2p1_probe,
)


def test_argparse_accepts_global_horizon_and_defaults_none() -> None:
    ns = build_arg_parser().parse_args([])
    assert ns.global_horizon is None
    ns2 = build_arg_parser().parse_args(["--global-horizon", "50", "--steps", "10"])
    assert ns2.global_horizon == 50
    assert ns2.steps == 10


@pytest.mark.parametrize(
    "global_horizon,steps",
    [
        (0, 1),
        (-1, 1),
        (20, 50),  # < steps
    ],
)
def test_entry_validation_rejects_invalid_before_parent_load(
    global_horizon: int, steps: int
) -> None:
    """Known-bad: load_parent_checkpoint must NEVER be called on invalid input."""
    with patch.object(
        probe_module, "load_parent_checkpoint", side_effect=AssertionError("parent load must not run")
    ) as load_mock:
        with patch.object(
            probe_module,
            "build_model_from_checkpoint",
            side_effect=AssertionError("model build must not run"),
        ) as build_mock:
            with pytest.raises(ValueError, match="global_horizon must be > 0 and >= steps"):
                run_c2p1_probe(
                    parent=Path("/tmp/nonexistent_parent.pt"),
                    parent_sha256="0" * 64,
                    scratch_root=Path("/tmp/a_prime_global_horizon_plumb_test"),
                    steps=steps,
                    global_horizon=global_horizon,
                    enabled=True,
                    device="cpu",
                )
    assert load_mock.call_count == 0
    assert build_mock.call_count == 0


@pytest.mark.parametrize(
    "global_horizon,steps",
    [
        (50, 50),  # == steps
        (50, 10),  # > steps
    ],
)
def test_entry_validation_accepts_valid_horizon_predicates(
    global_horizon: int, steps: int
) -> None:
    """Valid predicates pass the ENTRY gate; subsequent default-off may still fire.

    We only assert that ValueError for global_horizon is NOT raised. Parent path
    is invalid, so later failures are allowed after the entry gate.
    """
    with patch.object(
        probe_module,
        "load_parent_checkpoint",
        side_effect=RuntimeError("stop-after-entry-for-test"),
    ) as load_mock:
        with pytest.raises(RuntimeError, match="stop-after-entry-for-test"):
            run_c2p1_probe(
                parent=Path("/tmp/nonexistent_parent.pt"),
                parent_sha256="0" * 64,
                scratch_root=Path("/tmp/a_prime_global_horizon_plumb_test"),
                steps=steps,
                global_horizon=global_horizon,
                enabled=True,
                device="cpu",
            )
    assert load_mock.call_count == 1


def test_absent_flag_legacy_skips_entry_validation() -> None:
    """global_horizon=None must not raise the global_horizon ValueError."""
    with patch.object(
        probe_module,
        "load_parent_checkpoint",
        side_effect=RuntimeError("stop-after-entry-for-test"),
    ) as load_mock:
        with pytest.raises(RuntimeError, match="stop-after-entry-for-test"):
            run_c2p1_probe(
                parent=Path("/tmp/nonexistent_parent.pt"),
                parent_sha256="0" * 64,
                scratch_root=Path("/tmp/a_prime_global_horizon_plumb_test"),
                steps=50,
                global_horizon=None,
                enabled=True,
                device="cpu",
            )
    assert load_mock.call_count == 1


def test_main_forwards_global_horizon_kwarg() -> None:
    """Surface 2: main() passes args.global_horizon into run_c2p1_probe."""
    captured: dict[str, Any] = {}

    def _fake_run_c2p1_probe(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"synthetic": True, "ok": True}

    with patch.object(probe_module, "run_c2p1_probe", side_effect=_fake_run_c2p1_probe):
        with patch.object(
            probe_module,
            "maybe_enforce_phase3b_probe_import_byte_currency",
            create=True,
        ):
            # maybe_enforce is imported inside main from another module; patch run only.
            rc = probe_module.main(
                [
                    "--enable-bounded-delta-probe",
                    "--steps",
                    "10",
                    "--global-horizon",
                    "50",
                    "--scratch-root",
                    "/tmp/a_prime_global_horizon_plumb_test",
                    "--parent",
                    "/tmp/nonexistent_parent.pt",
                ]
            )
    assert rc == 0
    assert captured.get("global_horizon") == 50
    assert captured.get("steps") == 10


def test_run_bounded_delta_steps_receives_global_horizon_passthrough() -> None:
    """Surface 4: call site passes global_horizon into run_bounded_delta_steps.

    Unit-level: inspect source of run_c2p1_probe body for the keyword pass-through
    after the plumb (characterization of the four-surface activation).
    """
    src = Path(probe_module.__file__).read_text(encoding="utf-8")
    assert "global_horizon=global_horizon," in src
    assert "global_horizon=args.global_horizon," in src
    assert '"--global-horizon"' in src or "'--global-horizon'" in src
