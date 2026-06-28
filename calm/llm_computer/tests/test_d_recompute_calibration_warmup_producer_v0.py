from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_calibration_collector import (
    CALIBRATION_SURFACE_REPRESENTATIVE,
    CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_REPRESENTATIVE,
    MIN_REPRESENTATIVE_KEYS,
    build_stratified_selector_manifest,
)
from calm.llm_computer.tests.test_d_recompute_calibration_representative_coverage_v0 import (
    build_multi_key_eligible_tensor_states,
)
from scripts.hrm_text_158_d_recompute_calibration_prepass import (
    default_calibration_policy,
    run_calibration_prepass,
)
from scripts.hrm_text_158_d_recompute_calibration_warmup_producer import (
    DEFAULT_PARENT,
    DEFAULT_PARENT_SHA,
    run_calibration_warmup_producer,
)


def _mock_probe_runner_factory(
    *,
    observations_out: Path,
    parent_sha: str,
    warmup_steps: int = 5,
):
    states = build_multi_key_eligible_tensor_states(layers=6)
    manifest = build_stratified_selector_manifest(states)
    selected_keys = [entry.state_key for entry in manifest.entries]

    def _runner(argv, check=False, capture_output=True, text=True, env=None):
        payload = {
            "schema_version": CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
            "policy": default_calibration_policy(),
            "pre_warmup_banked_state_sha256": parent_sha,
            "calibration_surface": {
                "surface_id": CALIBRATION_SURFACE_REPRESENTATIVE,
                "selected_key_count": int(manifest.selected_key_count),
                "coverage_tier": str(manifest.coverage_tier),
                "representative_manifest_sha256": str(manifest.manifest_sha256),
                "manifest_spec": dict(manifest.manifest_spec),
            },
            "tensor_state_numel_by_key": {
                state_key: int(states[state_key].exact_accumulator_shadow.numel())
                for state_key in selected_keys
            },
            "steps": [
                {
                    "step": step,
                    "observations": {
                        state_key: {
                            "acc_values": [0]
                            * int(states[state_key].exact_accumulator_shadow.numel()),
                            "vote_values": [0]
                            * int(states[state_key].exact_accumulator_shadow.numel()),
                        }
                        for state_key in selected_keys
                    },
                }
                for step in range(1, int(warmup_steps) + 1)
            ],
        }
        observations_out.parent.mkdir(parents=True, exist_ok=True)
        observations_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return _runner


@pytest.mark.skipif(
    not Path(DEFAULT_PARENT).is_file(),
    reason="default parent checkpoint not present in workspace",
)
def test_warmup_producer_then_prepass_chain(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    observations_out = run_root / "prelaunch" / "calibration_warmup_observations.json"
    producer_receipt = run_calibration_warmup_producer(
        run_root=run_root,
        observations_out=observations_out,
        report_out=run_root / "prelaunch" / "calibration_warmup_producer_receipt.json",
        parent=Path(DEFAULT_PARENT),
        parent_sha256=DEFAULT_PARENT_SHA,
        warmup_steps=5,
        probe_runner=_mock_probe_runner_factory(
            observations_out=observations_out,
            parent_sha=DEFAULT_PARENT_SHA,
        ),
    )
    assert producer_receipt["pass"] is True
    assert producer_receipt["bit_exact_pre_warmup_parent_restored"] is True

    manifest_out = run_root / "prelaunch" / "calibrated_selector_manifest.json"
    prepass_receipt = run_calibration_prepass(
        warmup_observations_path=observations_out,
        manifest_out=manifest_out,
    )
    assert prepass_receipt["pass"] is True
    assert prepass_receipt["calibration_discard_contract_met"] is True
    assert prepass_receipt["coverage_gate_pass"] is True
    assert prepass_receipt["coverage_tier"] == COVERAGE_TIER_REPRESENTATIVE
    assert prepass_receipt["selected_key_count"] >= MIN_REPRESENTATIVE_KEYS
    assert manifest_out.is_file()
