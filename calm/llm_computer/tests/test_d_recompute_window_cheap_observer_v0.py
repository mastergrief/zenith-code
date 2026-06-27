from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    DObserverOutOfRangeShadowError,
    DObserverShadowUnavailableError,
    ReplayConstants,
    _accumulator_i32_flat,
    _sample_accumulator_lanes,
    _sample_lane_indices,
    default_production_replay_constants,
    maybe_emit_d_recompute_window_step_records,
    select_instrumentation_state_keys,
)


def _replay() -> ReplayConstants:
    return default_production_replay_constants()


def _fresh_state(
  acc_values: list[list[int]],
) -> BoundedDeltaTensorState:
    acc = torch.tensor(acc_values, dtype=torch.int16)
    q = torch.zeros_like(acc, dtype=torch.int8)
    return make_bounded_tensor_state("tiny.proj", q, 1.0, acc)


def _stale_state_from_fresh(fresh: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    return BoundedDeltaTensorState(
        state_key=fresh.state_key,
        q_levels=fresh.q_levels,
        frozen_scale=fresh.frozen_scale,
        bounded_accumulator=fresh.bounded_accumulator,
        exact_accumulator_shadow=fresh.exact_accumulator_shadow.clone(),
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def test_shadow_in_range_matches_decoded_reference_fresh() -> None:
    replay = _replay()
    state = _fresh_state([[5, -9, 21, 88, -127, 127]])
    lane_indices = _sample_lane_indices(int(state.exact_accumulator_shadow.numel()))
    oracle = [int(_accumulator_i32_flat(state)[index].item()) for index in lane_indices]
    sampled = _sample_accumulator_lanes(state, lane_indices, replay_constants=replay)
    assert sampled == oracle


def test_shadow_in_range_matches_decoded_reference_stale() -> None:
    replay = _replay()
    fresh = _fresh_state([[5, -9, 21, 88, -127, 127]])
    stale = _stale_state_from_fresh(fresh)
    assert stale.bounded_accumulator_fresh_for_exact_shadow is False
    lane_indices = _sample_lane_indices(int(stale.exact_accumulator_shadow.numel()))
    oracle = [int(_accumulator_i32_flat(stale)[index].item()) for index in lane_indices]
    sampled = _sample_accumulator_lanes(stale, lane_indices, replay_constants=replay)
    assert sampled == oracle


@pytest.mark.parametrize("out_of_range_value", [140, -200])
def test_shadow_out_of_clamp_fails_closed(out_of_range_value: int) -> None:
    replay = _replay()
    state = _fresh_state([[0, out_of_range_value]])
    lane_indices = [1]
    with pytest.raises(DObserverOutOfRangeShadowError):
        _sample_accumulator_lanes(state, lane_indices, replay_constants=replay)


def test_shadow_absent_fail_closed() -> None:
    replay = _replay()
    fresh = _fresh_state([[1, 2, 3]])
    state = BoundedDeltaTensorState(
        state_key=fresh.state_key,
        q_levels=fresh.q_levels,
        frozen_scale=fresh.frozen_scale,
        bounded_accumulator=fresh.bounded_accumulator,
        exact_accumulator_shadow=None,
        bounded_accumulator_fresh_for_exact_shadow=False,
    )
    with pytest.raises(DObserverShadowUnavailableError):
        _sample_accumulator_lanes(state, [0], replay_constants=replay)


def test_shadow_invalid_dtype_fail_closed() -> None:
    replay = _replay()
    fresh = _fresh_state([[1, 2, 3]])
    state = object.__new__(BoundedDeltaTensorState)
    object.__setattr__(state, "state_key", fresh.state_key)
    object.__setattr__(state, "q_levels", fresh.q_levels)
    object.__setattr__(state, "frozen_scale", fresh.frozen_scale)
    object.__setattr__(state, "bounded_accumulator", fresh.bounded_accumulator)
    object.__setattr__(
        state,
        "exact_accumulator_shadow",
        fresh.exact_accumulator_shadow.to(torch.int32),
    )
    object.__setattr__(state, "bounded_accumulator_fresh_for_exact_shadow", True)
    object.__setattr__(state, "bounded_accumulator_rebuild_hot_exact_indices", None)
    object.__setattr__(state, "bounded_accumulator_rebuild_cold_default_value", None)
    object.__setattr__(state, "event_coded_live_carrier", None)
    with pytest.raises(DObserverShadowUnavailableError, match="torch.int16"):
        _sample_accumulator_lanes(state, [0], replay_constants=replay)


def test_stale_emit_never_calls_decoded_accumulators_rebuild(tmp_path: Path) -> None:
    replay = _replay()
    fresh = _fresh_state([[5, -9, 21, 88]])
    stale = _stale_state_from_fresh(fresh)
    states = {"tiny.proj": stale}
    votes = {
        "tiny.proj": torch.tensor([[1, -1, 2, 0]], dtype=torch.int32),
    }
    log_path = tmp_path / "recompute_window_log.jsonl"
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
            step=3,
            pre_update_states=states,
            post_update_states=states,
            votes_by_key=votes,
            replay_constants=replay,
        )
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8").strip()


def test_select_instrumentation_keys_never_rebuilds() -> None:
    small = _fresh_state([[1, 2]])
    large = _fresh_state([[3, 4, 5, 6, 7, 8, 9, 10]])
    stale_small = _stale_state_from_fresh(small)
    stale_large = _stale_state_from_fresh(large)
    states = {"large.proj": stale_large, "small.proj": stale_small}
    original = BoundedDeltaTensorState.decoded_accumulators

    def _spy_decoded_accumulators(
        self: BoundedDeltaTensorState,
        *,
        device: torch.device | str | None = None,
        rebuild_if_stale: bool = False,
    ) -> torch.Tensor:
        if rebuild_if_stale:
            pytest.fail("decoded_accumulators(rebuild_if_stale=True) called during key select")
        return original(self, device=device, rebuild_if_stale=rebuild_if_stale)

    with mock.patch.object(
        BoundedDeltaTensorState,
        "decoded_accumulators",
        _spy_decoded_accumulators,
    ):
        keys = select_instrumentation_state_keys(states, max_keys=1)
    assert keys == ["small.proj"]
