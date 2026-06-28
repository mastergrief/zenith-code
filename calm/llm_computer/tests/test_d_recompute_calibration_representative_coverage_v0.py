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
    CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
    CalibrationWarmupCollector,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    default_production_replay_constants,
    select_instrumentation_state_keys,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_PILOT,
    COVERAGE_TIER_REPRESENTATIVE,
    MIN_REPRESENTATIVE_KEYS,
    build_stratified_selector_manifest,
    extract_calibration_observation,
)
from scripts.hrm_text_158_d_recompute_calibration_prepass import (
    CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA as PREPASS_WARMUP_SCHEMA,
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
    states["model.embed_tokens.weight"] = _fresh_state(
        "model.embed_tokens.weight",
        rows=2,
        cols=2,
        seed=seed,
    )
    states["model.lm_head.weight"] = _fresh_state(
        "model.lm_head.weight",
        rows=2,
        cols=2,
        seed=seed + 1,
    )
    states["model.H_level.core.layers.0.norm.weight"] = _fresh_state(
        "model.H_level.core.layers.0.norm.weight",
        rows=2,
        cols=2,
        seed=seed + 2,
    )
    return states


def _replay_constants() -> ReplayConstants:
    return default_production_replay_constants()


def _votes_for_state(state: BoundedDeltaTensorState) -> torch.Tensor:
    numel = int(state.exact_accumulator_shadow.numel())
    return torch.zeros(numel, dtype=torch.int32)


def _write_legacy_two_smallest_warmup(path: Path, states: dict[str, BoundedDeltaTensorState]) -> None:
    replay = _replay_constants()
    legacy_keys = select_instrumentation_state_keys(states)
    assert len(legacy_keys) == 2
    steps = []
    for step in (1, 2, 3):
        observations: dict[str, dict[str, list[int]]] = {}
        for state_key in legacy_keys:
            state = states[state_key]
            acc_values, vote_values = extract_calibration_observation(
                state_key,
                state,
                _votes_for_state(state),
                replay_constants=replay,
            )
            observations[state_key] = {
                "acc_values": list(acc_values),
                "vote_values": list(vote_values),
            }
        steps.append({"step": step, "observations": observations})
    payload = {
        "schema_version": PREPASS_WARMUP_SCHEMA,
        "policy": default_calibration_policy(),
        "pre_warmup_banked_state_sha256": "abc123",
        "tensor_state_numel_by_key": {
            state_key: int(states[state_key].exact_accumulator_shadow.numel())
            for state_key in legacy_keys
        },
        "steps": steps,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_legacy_two_smallest_selector_is_not_representative() -> None:
    states = build_multi_key_eligible_tensor_states(layers=6)
    legacy_keys = select_instrumentation_state_keys(states)
    assert len(legacy_keys) == 2
    manifest = build_stratified_selector_manifest(states)
    assert manifest.coverage_tier == COVERAGE_TIER_REPRESENTATIVE
    assert manifest.selected_key_count >= MIN_REPRESENTATIVE_KEYS


def test_legacy_two_smallest_warmup_observations_fail_representative_coverage_gate(
    tmp_path: Path,
) -> None:
    states = build_multi_key_eligible_tensor_states(layers=6)
    warmup_path = tmp_path / "legacy_warmup.json"
    _write_legacy_two_smallest_warmup(warmup_path, states)
    receipt = run_calibration_prepass(
        warmup_observations_path=warmup_path,
        manifest_out=tmp_path / "manifest.json",
    )
    assert receipt["coverage_tier"] == COVERAGE_TIER_PILOT
    assert receipt["selected_key_count"] < MIN_REPRESENTATIVE_KEYS
    assert receipt["coverage_gate_pass"] is False
    assert receipt["pass"] is False


def test_collector_records_representative_surface_for_multi_key_eligible_bulk(
    tmp_path: Path,
) -> None:
    states = build_multi_key_eligible_tensor_states(layers=6)
    replay = _replay_constants()
    collector = CalibrationWarmupCollector(
        output_path=tmp_path / "warmup.json",
        pre_warmup_parent_sha256="abc123",
        policy=default_calibration_policy(),
    )
    votes_by_key = {state_key: _votes_for_state(state) for state_key, state in states.items()}
    for step in (1, 2, 3):
        collector.record_step(
            step=step,
            pre_update_states=states,
            votes_by_key=votes_by_key,
            replay_constants=replay,
        )
    output_path = collector.write()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA
    surface = payload["calibration_surface"]
    assert surface["surface_id"] == CALIBRATION_SURFACE_REPRESENTATIVE
    assert surface["selected_key_count"] >= MIN_REPRESENTATIVE_KEYS
    assert surface["coverage_tier"] == COVERAGE_TIER_REPRESENTATIVE
    assert len(payload["tensor_state_numel_by_key"]) >= MIN_REPRESENTATIVE_KEYS


def test_representative_warmup_observations_pass_coverage_gate(tmp_path: Path) -> None:
    states = build_multi_key_eligible_tensor_states(layers=6)
    replay = _replay_constants()
    collector = CalibrationWarmupCollector(
        output_path=tmp_path / "warmup.json",
        pre_warmup_parent_sha256="abc123",
        policy=default_calibration_policy(),
    )
    votes_by_key = {state_key: _votes_for_state(state) for state_key, state in states.items()}
    for step in (1, 2, 3):
        collector.record_step(
            step=step,
            pre_update_states=states,
            votes_by_key=votes_by_key,
            replay_constants=replay,
        )
    collector.write()
    receipt = run_calibration_prepass(
        warmup_observations_path=tmp_path / "warmup.json",
        manifest_out=tmp_path / "manifest.json",
    )
    assert receipt["coverage_tier"] == COVERAGE_TIER_REPRESENTATIVE
    assert receipt["selected_key_count"] >= MIN_REPRESENTATIVE_KEYS
    assert receipt["coverage_gate_pass"] is True
    assert receipt["pass"] is True
