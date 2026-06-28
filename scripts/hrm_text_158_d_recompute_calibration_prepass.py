#!/usr/bin/env python3
"""Finalize calibrated stratified selector manifest from bounded warmup observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_calibration_collector import (
    CALIBRATION_SURFACE_REPRESENTATIVE,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    CalibrationWarmupStep,
    COVERAGE_TIER_REPRESENTATIVE,
    DEFAULT_CALIBRATION_WARMUP_STEPS,
    MIN_REPRESENTATIVE_KEYS,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    build_calibrated_stratified_selector_manifest,
    save_stratified_selector_manifest,
)

CALIBRATION_POLICY_ID = "horizon_fixed_warmup_calibrated_v0"
CALIBRATION_PREPASS_SCHEMA = "hrm_text_158_d_recompute_calibration_prepass_receipt/v0"
CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA = (
    "hrm_text_158_d_recompute_calibration_warmup_observations/v0"
)


def default_calibration_policy() -> dict[str, Any]:
    return {
        "policy_id": CALIBRATION_POLICY_ID,
        "warmup_steps": DEFAULT_CALIBRATION_WARMUP_STEPS,
        "scoring_scheme": "stress_tail_score_v1",
        "candidate_superset": "same_family_eligible_bulk",
        "calibration_seed": 43,
        "support_order_seed": 43,
        "reset_semantics": "discard_calibration_state_before_measurement",
        "measurement_start_step": 1,
        "calibration_discarded_before_measurement": True,
    }


def calibration_warmup_step_to_dict(step: CalibrationWarmupStep) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for state_key, (acc_values, vote_values) in step.observations.items():
        observations[state_key] = {
            "acc_values": list(acc_values),
            "vote_values": list(vote_values),
        }
    return {"step": int(step.step), "observations": observations}


def calibration_warmup_step_from_dict(payload: Mapping[str, Any]) -> CalibrationWarmupStep:
    observations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for state_key, entry in dict(payload.get("observations") or {}).items():
        acc_values = tuple(int(value) for value in entry["acc_values"])
        vote_values = tuple(int(value) for value in entry["vote_values"])
        observations[str(state_key)] = (acc_values, vote_values)
    return CalibrationWarmupStep(step=int(payload["step"]), observations=observations)


def load_calibration_warmup_observations(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA:
        raise ValueError(
            f"unsupported warmup observations schema {schema!r}; "
            f"expected {CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA}"
        )
    steps = [
        calibration_warmup_step_from_dict(step_payload)
        for step_payload in payload.get("steps") or []
    ]
    return {
        "policy": dict(payload.get("policy") or default_calibration_policy()),
        "pre_warmup_banked_state_sha256": payload.get("pre_warmup_banked_state_sha256"),
        "calibration_surface": dict(payload.get("calibration_surface") or {}),
        "tensor_state_numel_by_key": {
            str(key): int(value)
            for key, value in dict(payload.get("tensor_state_numel_by_key") or {}).items()
        },
        "steps": steps,
    }


def _synthetic_tensor_state(*, state_key: str, numel: int, seed: int = 0) -> BoundedDeltaTensorState:
    generator = torch.Generator().manual_seed(int(seed))
    target = int(numel)
    if target <= 0:
        raise ValueError(f"numel must be positive for {state_key!r}, got {target}")
    rows = target
    cols = 1
    acc = torch.randint(-40, 40, (rows, cols), dtype=torch.int16, generator=generator)
    acc = acc.clamp(-120, 120)
    q = torch.zeros(rows, cols, dtype=torch.int8)
    return make_bounded_tensor_state(state_key, q, 1.0, acc)


def build_tensor_states_from_numel_map(
    numel_by_key: Mapping[str, int],
) -> dict[str, BoundedDeltaTensorState]:
    states: dict[str, BoundedDeltaTensorState] = {}
    for index, (state_key, numel) in enumerate(sorted(numel_by_key.items())):
        states[state_key] = _synthetic_tensor_state(
            state_key=str(state_key),
            numel=int(numel),
            seed=index + 17,
        )
    return states


def calibration_discard_contract_met(
    *,
    policy: Mapping[str, Any],
    manifest_spec: Mapping[str, Any],
) -> bool:
    return bool(policy.get("calibration_discarded_before_measurement")) and bool(
        manifest_spec.get("calibration_discarded_before_measurement")
    )


def representative_coverage_gate_met(
    *,
    coverage_tier: str,
    selected_key_count: int,
    policy: Mapping[str, Any],
    calibration_surface: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None]:
    if bool(policy.get("explicit_pilot_demotion")):
        return True, "explicit_pilot_demotion"
    if calibration_surface is not None:
        surface_id = str(calibration_surface.get("surface_id") or "")
        if surface_id != CALIBRATION_SURFACE_REPRESENTATIVE:
            return False, f"calibration_surface={surface_id!r}"
    if str(coverage_tier) != COVERAGE_TIER_REPRESENTATIVE:
        return False, f"coverage_tier={coverage_tier!r}"
    if int(selected_key_count) < int(MIN_REPRESENTATIVE_KEYS):
        return (
            False,
            f"selected_key_count={selected_key_count}<{MIN_REPRESENTATIVE_KEYS}",
        )
    return True, None


def run_calibration_prepass(
    *,
    warmup_observations_path: Path,
    manifest_out: Path,
    report_out: Path | None = None,
    replay_constants: ReplayConstants | None = None,
    require_representative_coverage: bool = True,
) -> dict[str, Any]:
    replay = replay_constants or default_production_replay_constants()
    loaded = load_calibration_warmup_observations(warmup_observations_path)
    policy = dict(loaded["policy"])
    calibration_samples: list[CalibrationWarmupStep] = list(loaded["steps"])
    numel_by_key = dict(loaded["tensor_state_numel_by_key"])
    if not numel_by_key:
        raise ValueError("warmup observations missing tensor_state_numel_by_key")
    if not calibration_samples:
        raise ValueError("warmup observations missing calibration steps")

    tensor_states = build_tensor_states_from_numel_map(numel_by_key)
    manifest_spec = {
        "policy_id": str(policy.get("policy_id") or CALIBRATION_POLICY_ID),
        "calibration_policy": policy,
        "min_keys": 12,
        "max_keys": 18,
    }
    manifest = build_calibrated_stratified_selector_manifest(
        tensor_states,
        calibration_samples=calibration_samples,
        manifest_spec=manifest_spec,
        replay_constants=replay,
        measurement_start_step=int(policy.get("measurement_start_step") or 1),
    )
    if manifest.manifest_spec.get("stress_tail_policy") != STRESS_TAIL_POLICY_HORIZON_FIXED:
        raise ValueError("calibrated manifest missing horizon-fixed stress-tail policy")
    save_stratified_selector_manifest(manifest, manifest_out)

    pre_warmup_sha = loaded.get("pre_warmup_banked_state_sha256")
    measurement_start_step = int(policy.get("measurement_start_step") or 1)
    discard_contract_met = calibration_discard_contract_met(
        policy=policy,
        manifest_spec=manifest.manifest_spec,
    )
    calibration_surface = dict(loaded.get("calibration_surface") or {})
    coverage_gate_pass, coverage_gate_reason = representative_coverage_gate_met(
        coverage_tier=str(manifest.coverage_tier),
        selected_key_count=int(manifest.selected_key_count),
        policy=policy,
        calibration_surface=calibration_surface or None,
    )
    if require_representative_coverage and not coverage_gate_pass:
        pass_ok = False
    else:
        pass_ok = bool(discard_contract_met and coverage_gate_pass)
    receipt = {
        "schema_version": CALIBRATION_PREPASS_SCHEMA,
        "policy": policy,
        "warmup_observations_path": str(warmup_observations_path),
        "manifest_out": str(manifest_out),
        "manifest_sha256": manifest.manifest_sha256,
        "coverage_tier": manifest.coverage_tier,
        "selected_key_count": manifest.selected_key_count,
        "calibration_surface": calibration_surface or None,
        "require_representative_coverage": bool(require_representative_coverage),
        "coverage_gate_pass": bool(coverage_gate_pass),
        "coverage_gate_reason": coverage_gate_reason,
        "measurement_start_step": measurement_start_step,
        "calibration_discarded_before_measurement": bool(
            policy.get("calibration_discarded_before_measurement")
        ),
        "calibration_discard_contract_met": discard_contract_met,
        "warmup_step_count": len(calibration_samples),
        "pre_warmup_banked_state_sha256": pre_warmup_sha,
        "bit_exact_pre_warmup_restore_required": True,
        "pass": pass_ok,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-observations", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = run_calibration_prepass(
        warmup_observations_path=args.warmup_observations,
        manifest_out=args.manifest_out,
        report_out=args.json_report,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if bool(receipt["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
