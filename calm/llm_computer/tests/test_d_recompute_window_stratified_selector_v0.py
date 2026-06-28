from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    default_production_replay_constants,
    maybe_emit_d_recompute_window_step_records,
    select_instrumentation_state_keys,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_PILOT,
    COVERAGE_TIER_REPRESENTATIVE,
    MAX_REPRESENTATIVE_KEYS,
    MIN_REPRESENTATIVE_KEYS,
    STRESS_LANE_COUNT,
    UNIFORM_LANE_COUNT,
    _stress_tail_indices,
    _uniform_stride_indices,
    build_stratified_selector_manifest,
    load_stratified_selector_manifest,
    sample_lanes_for_key,
    save_stratified_selector_manifest,
    select_instrumentation_state_keys_from_manifest,
)


def _replay() -> ReplayConstants:
    return default_production_replay_constants()


def _fresh_state(
    state_key: str,
    *,
    rows: int,
    cols: int,
    seed: int,
) -> BoundedDeltaTensorState:
    generator = torch.Generator().manual_seed(int(seed))
    acc = torch.randint(-40, 40, (rows, cols), dtype=torch.int16, generator=generator)
    acc = acc.clamp(-120, 120)
    q = torch.zeros(rows, cols, dtype=torch.int8)
    return make_bounded_tensor_state(state_key, q, 1.0, acc)


def _stale_state_from_fresh(fresh: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    return BoundedDeltaTensorState(
        state_key=fresh.state_key,
        q_levels=fresh.q_levels,
        frozen_scale=fresh.frozen_scale,
        bounded_accumulator=fresh.bounded_accumulator,
        exact_accumulator_shadow=fresh.exact_accumulator_shadow.clone(),
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def _build_synthetic_tensor_states(*, layers: int = 6) -> dict[str, BoundedDeltaTensorState]:
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
                key = (
                    f"model.{level}_level.core.layers.{layer_idx}{role_suffix}"
                )
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


def test_manifest_is_deterministic_for_same_input() -> None:
    states = _build_synthetic_tensor_states(layers=6)
    first = build_stratified_selector_manifest(states)
    second = build_stratified_selector_manifest(states)
    assert first.manifest_sha256 == second.manifest_sha256
    assert [entry.state_key for entry in first.entries] == [
        entry.state_key for entry in second.entries
    ]


def test_exclude_rules_and_representative_key_count() -> None:
    states = _build_synthetic_tensor_states(layers=6)
    manifest = build_stratified_selector_manifest(states)
    selected = {entry.state_key for entry in manifest.entries}
    assert "model.embed_tokens.weight" not in selected
    assert "model.lm_head.weight" not in selected
    assert "model.H_level.core.layers.0.norm.weight" not in selected
    assert MIN_REPRESENTATIVE_KEYS <= len(selected) <= MAX_REPRESENTATIVE_KEYS
    assert manifest.coverage_tier == COVERAGE_TIER_REPRESENTATIVE


def test_lane_mix_uniform_plus_stress_without_overlap() -> None:
    states = _build_synthetic_tensor_states(layers=6)
    manifest = build_stratified_selector_manifest(states)
    for entry in manifest.entries:
        assert len(entry.uniform_lanes) == UNIFORM_LANE_COUNT
        assert len(entry.stress_tail_lanes) == STRESS_LANE_COUNT
        overlap = set(entry.uniform_lanes) & set(entry.stress_tail_lanes)
        assert not overlap
        assert len(entry.lane_indices) == UNIFORM_LANE_COUNT + STRESS_LANE_COUNT
        assert entry.stratum_weight > 0.0
    assert manifest.stratum_weights
    assert abs(sum(manifest.stratum_weights.values()) - 1.0) < 1e-9


def test_stress_tail_is_vote_sensitive_for_update_pressure_lanes() -> None:
    replay = _replay()
    state_key = "model.H_level.core.layers.0.attn.o_proj"
    state = _fresh_state(state_key, rows=8, cols=8, seed=17)
    numel = 64
    uniform = _uniform_stride_indices(numel, count=UNIFORM_LANE_COUNT)
    high_acc_lane = 5
    high_vote_lane = 6
    assert high_acc_lane not in uniform
    assert high_vote_lane not in uniform

    acc_values = [1] * numel
    acc_values[high_acc_lane] = 80
    acc_values[high_vote_lane] = 1
    state = replace(
        state,
        exact_accumulator_shadow=torch.tensor(acc_values, dtype=torch.int16).reshape(8, 8),
    )

    vote_values = [0] * numel
    vote_values[high_vote_lane] = 200

    acc_only_stress = _stress_tail_indices(
        state,
        numel=numel,
        exclude=set(uniform),
        count=1,
        replay_constants=replay,
        vote_values=None,
    )
    vote_aware_stress = _stress_tail_indices(
        state,
        numel=numel,
        exclude=set(uniform),
        count=1,
        replay_constants=replay,
        vote_values=vote_values,
    )

    assert acc_only_stress == [high_acc_lane]
    assert vote_aware_stress == [high_vote_lane]
    assert high_vote_lane not in _stress_tail_indices(
        state,
        numel=numel,
        exclude=set(uniform),
        count=1,
        replay_constants=replay,
        vote_values=None,
    )

    manifest = build_stratified_selector_manifest({state_key: state})
    entry = manifest.entries[0]
    live_lanes = sample_lanes_for_key(
        state,
        manifest_entry=entry,
        vote_values=vote_values,
        replay_constants=replay,
    )
    stress_live = [index for index in live_lanes if index not in set(entry.uniform_lanes)]
    assert high_vote_lane in stress_live


def test_pilot_floor_labels_shrunk_fixture(tmp_path: Path) -> None:
    states = _build_synthetic_tensor_states(layers=1)
    manifest = build_stratified_selector_manifest(
        states,
        manifest_spec={"min_keys": 4, "max_keys": 6},
    )
    assert manifest.coverage_tier == COVERAGE_TIER_PILOT
    path = tmp_path / "selector_manifest.json"
    save_stratified_selector_manifest(manifest, path)
    loaded = load_stratified_selector_manifest(path)
    assert loaded.manifest_sha256 == manifest.manifest_sha256


def test_select_from_manifest_and_emit_back_compat_without_manifest(tmp_path: Path) -> None:
    states = _build_synthetic_tensor_states(layers=6)
    manifest = build_stratified_selector_manifest(states)
    selected = select_instrumentation_state_keys_from_manifest(states, manifest)
    assert selected == [entry.state_key for entry in manifest.entries]

    small = _fresh_state("tiny.small", rows=2, cols=2, seed=99)
    large = _fresh_state("tiny.large", rows=2, cols=8, seed=100)
    default_states = {"tiny.large": large, "tiny.small": small}
    assert select_instrumentation_state_keys(default_states, max_keys=1) == ["tiny.small"]

    replay = _replay()
    stale = _stale_state_from_fresh(small)
    log_path = tmp_path / "recompute_window_log.jsonl"
    votes = {"tiny.small": torch.tensor([[1, -1, 2, 0]], dtype=torch.int32)}
    original = BoundedDeltaTensorState.decoded_accumulators

    def _spy_decoded_accumulators(
        self: BoundedDeltaTensorState,
        *,
        device: torch.device | str | None = None,
        rebuild_if_stale: bool = False,
    ) -> torch.Tensor:
        if rebuild_if_stale:
            pytest.fail("decoded_accumulators(rebuild_if_stale=True) called on hot loop")
        return original(self, device=device, rebuild_if_stale=rebuild_if_stale)

    with mock.patch.object(
        BoundedDeltaTensorState,
        "decoded_accumulators",
        _spy_decoded_accumulators,
    ):
        maybe_emit_d_recompute_window_step_records(
            enabled=True,
            log_path=log_path,
            step=1,
            pre_update_states={"tiny.small": stale},
            post_update_states={"tiny.small": stale},
            votes_by_key=votes,
            replay_constants=replay,
        )
        maybe_emit_d_recompute_window_step_records(
            enabled=True,
            log_path=log_path,
            step=2,
            pre_update_states=states,
            post_update_states=states,
            votes_by_key={
                key: torch.zeros(
                    int(state.q_levels.numel()),
                    dtype=torch.int32,
                )
                for key, state in states.items()
            },
            replay_constants=replay,
            selector_manifest=manifest,
        )
    assert log_path.is_file()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) >= 1 + len(manifest.entries)
