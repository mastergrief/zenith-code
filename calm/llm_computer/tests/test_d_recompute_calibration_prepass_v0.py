from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_calibration_collector import (
    CALIBRATION_SURFACE_REPRESENTATIVE,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_REPRESENTATIVE,
    MIN_REPRESENTATIVE_KEYS,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    build_stratified_selector_manifest,
    extract_calibration_observation,
    load_stratified_selector_manifest,
)
from scripts.hrm_text_158_d_recompute_calibration_prepass import (
    CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
    default_calibration_policy,
    run_calibration_prepass,
)


def _fresh_state(state_key: str, *, rows: int, cols: int, seed: int) -> BoundedDeltaTensorState:
    generator = torch.Generator().manual_seed(int(seed))
    acc = torch.randint(-40, 40, (rows, cols), dtype=torch.int16, generator=generator)
    acc = acc.clamp(-120, 120)
    q = torch.zeros(rows, cols, dtype=torch.int8)
    return make_bounded_tensor_state(state_key, q, 1.0, acc)


def build_multi_key_eligible_tensor_states(*, layers: int = 6) -> dict[str, BoundedDeltaTensorState]:
    states: dict[str, BoundedDeltaTensorState] = {}
    seed = 0
    for level in ("H", "L"):
        for layer_idx in range(layers):
            base = 8 + layer_idx
            for role_suffix, shape in (
                (".attn.gqkv_proj", (base, base + 1)),
                (".attn.o_proj", (base + 1, base)),
                (".mlp.gate_up_proj", (base, base + 2)),
                (".mlp.down_proj", (base + 2, base)),
            ):
                key = f"model.{level}_level.core.layers.{layer_idx}{role_suffix}"
                rows, cols = shape
                states[key] = _fresh_state(key, rows=rows, cols=cols, seed=seed)
                seed += 1
    return states


def _write_representative_warmup_observations(path: Path) -> None:
    states = build_multi_key_eligible_tensor_states(layers=6)
    replay = default_production_replay_constants()
    manifest = build_stratified_selector_manifest(states)
    selected_keys = [entry.state_key for entry in manifest.entries]
    steps = []
    for step in (1, 2, 3):
        observations: dict[str, dict[str, list[int]]] = {}
        for state_key in selected_keys:
            state = states[state_key]
            vote_tensor = torch.zeros(
                int(state.exact_accumulator_shadow.numel()),
                dtype=torch.int32,
            )
            acc_values, vote_values = extract_calibration_observation(
                state_key,
                state,
                vote_tensor,
                replay_constants=replay,
            )
            observations[state_key] = {
                "acc_values": list(acc_values),
                "vote_values": list(vote_values),
            }
        steps.append({"step": step, "observations": observations})
    payload = {
        "schema_version": CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
        "policy": default_calibration_policy(),
        "pre_warmup_banked_state_sha256": "abc123",
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
        "steps": steps,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_same_warmup_produces_same_manifest_sha(tmp_path: Path) -> None:
    warmup_path = tmp_path / "warmup.json"
    _write_representative_warmup_observations(warmup_path)
    manifest_a = tmp_path / "manifest_a.json"
    manifest_b = tmp_path / "manifest_b.json"
    receipt_a = run_calibration_prepass(
        warmup_observations_path=warmup_path,
        manifest_out=manifest_a,
    )
    receipt_b = run_calibration_prepass(
        warmup_observations_path=warmup_path,
        manifest_out=manifest_b,
    )
    assert receipt_a["manifest_sha256"] == receipt_b["manifest_sha256"]
    loaded = load_stratified_selector_manifest(manifest_a)
    assert loaded.manifest_spec["stress_tail_policy"] == STRESS_TAIL_POLICY_HORIZON_FIXED
    assert receipt_a["coverage_tier"] == COVERAGE_TIER_REPRESENTATIVE
    assert receipt_a["selected_key_count"] >= MIN_REPRESENTATIVE_KEYS


def test_calibration_discard_contract_met(tmp_path: Path) -> None:
    warmup_path = tmp_path / "warmup.json"
    _write_representative_warmup_observations(warmup_path)
    receipt = run_calibration_prepass(
        warmup_observations_path=warmup_path,
        manifest_out=tmp_path / "manifest.json",
    )
    assert receipt["measurement_start_step"] == 1
    assert receipt["calibration_discard_contract_met"] is True
    assert receipt["warmup_step_count"] == 3
    assert receipt["coverage_gate_pass"] is True
    assert receipt["pass"] is True


def test_calibration_discard_contract_requires_policy_flag() -> None:
    from scripts.hrm_text_158_d_recompute_calibration_prepass import (
        calibration_discard_contract_met,
    )

    assert calibration_discard_contract_met(
        policy={"calibration_discarded_before_measurement": True},
        manifest_spec={"calibration_discarded_before_measurement": True},
    )
    assert not calibration_discard_contract_met(
        policy={"calibration_discarded_before_measurement": False},
        manifest_spec={"calibration_discarded_before_measurement": True},
    )
