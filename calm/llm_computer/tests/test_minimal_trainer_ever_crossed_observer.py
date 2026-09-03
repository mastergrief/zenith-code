"""CPU characterization for the minimal_trainer frame-local ever-crossed observer.

In-memory synthetic 2-step run: the cumulative union is provably larger than
either step's candidate set. The production observer satisfies that assertion;
a deliberately broken observer (mask reset each step) fails the same one.

No filesystem I/O, no scratch, no hashing. The 1-step GPU smoke and the
state_dict/cost checks are run and emitted by gate-1, not by this file.

The observer is opt-in (`ever_crossed_observer_enabled`, default False), lives
only in loop-frame memory, and reads the producer's `candidate_indices`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.minimal_trainer.loop import (
    NEVER_CANDIDATE_STEP,
    PRODUCTION_CANDIDATE_WINDOWS,
    _ever_crossed_emission,
    _ever_crossed_masks_for_states,
    _last_candidate_steps_for_states,
    _observe_candidates,
    _or_accumulate_ever_crossed,
    _require_fresh_observation,
    _windowed_candidate_emission,
)

LEAF = "leaf_a"
NUMEL = 8
STEP_1_CANDIDATES = [0, 1]
STEP_2_CANDIDATES = [1, 2, 3]
UNION_SIZE = 4  # {0,1,2,3}: strictly larger than len(step1)=2 and len(step2)=3


def _fake_states(numel: int = NUMEL) -> dict[str, SimpleNamespace]:
    return {LEAF: SimpleNamespace(q_levels=torch.zeros(numel, dtype=torch.int8))}


def _observation(candidates: list[int]) -> dict[str, object]:
    """Shape mirrors _front_c_cloned_observation: plans_by_key -> .candidate_indices."""
    plan = SimpleNamespace(
        candidate_indices=torch.tensor(candidates, dtype=torch.int64)
    )
    return {"plans_by_key": {LEAF: plan}}


def _broken_or_accumulate(
    masks: dict[str, torch.Tensor], observation: dict[str, object]
) -> None:
    """Deliberately broken observer: resets the mask each step (no OR-accumulation)."""
    for key, plan in dict(observation["plans_by_key"]).items():  # type: ignore[arg-type]
        mask = masks[str(key)]
        mask.zero_()
        idx = plan.candidate_indices.detach().cpu().to(torch.int64).reshape(-1)
        if int(idx.numel()) > 0:
            mask[idx] = True


def _two_step_union_count(accumulate) -> int:
    masks = _ever_crossed_masks_for_states(_fake_states())
    accumulate(masks, _observation(STEP_1_CANDIDATES))
    accumulate(masks, _observation(STEP_2_CANDIDATES))
    emission = _ever_crossed_emission(masks)
    denominator = emission["ever_crossed_numel_by_key"][LEAF]
    assert denominator == NUMEL
    return round(emission["ever_crossed_fraction_by_key"][LEAF] * denominator)


def test_cpu_union_exceeds_either_step_known_good():
    """Production observer: cumulative union strictly exceeds both step sets."""
    crossed = _two_step_union_count(_or_accumulate_ever_crossed)
    assert crossed == UNION_SIZE
    assert crossed > len(STEP_1_CANDIDATES)
    assert crossed > len(STEP_2_CANDIDATES)


def test_cpu_broken_observer_fails_the_same_assertion():
    """Known-bad: mask reset each step cannot exceed the last step's set."""
    crossed = _two_step_union_count(_broken_or_accumulate)
    assert crossed == len(STEP_2_CANDIDATES)
    with pytest.raises(AssertionError):
        assert crossed > len(STEP_2_CANDIDATES)


def test_cpu_emission_carries_total_fraction_and_denominators():
    masks = _ever_crossed_masks_for_states(_fake_states())
    _or_accumulate_ever_crossed(masks, _observation(STEP_1_CANDIDATES))
    _or_accumulate_ever_crossed(masks, _observation(STEP_2_CANDIDATES))
    emission = _ever_crossed_emission(masks)
    assert set(emission) == {
        "ever_crossed_numel_by_key",
        "ever_crossed_fraction_by_key",
        "ever_crossed_fraction_total",
    }
    assert emission["ever_crossed_numel_by_key"] == {LEAF: NUMEL}
    assert emission["ever_crossed_fraction_total"] == pytest.approx(
        UNION_SIZE / NUMEL
    )


def test_cpu_empty_denominator_refuses():
    """Negative path for the fail-closed denominator guard, seen firing."""
    with pytest.raises(RuntimeError, match="empty-denominator"):
        _ever_crossed_emission({})


def test_cpu_masks_are_frame_local_bool_tensors():
    masks = _ever_crossed_masks_for_states(_fake_states())
    assert masks[LEAF].dtype is torch.bool
    assert not masks[LEAF].any()


# --- H2 windowed hot set: trailing-window candidate occupancy ------------------
#
# One frame-local int32 `last_candidate_step` per eligible leaf. Production
# emission is exactly W in PRODUCTION_CANDIDATE_WINDOWS; W=2 and W=3 below are
# test-internal windows read from that same state, never emitted fields.

LEAF_B = "leaf_b"
W_STEP_1 = [0]
W_STEP_2 = [1]
W_STEP_3 = [2, 3]


def _multi_states(numel_by_key: dict[str, int]) -> dict[str, SimpleNamespace]:
    return {
        key: SimpleNamespace(q_levels=torch.zeros(numel, dtype=torch.int8))
        for key, numel in numel_by_key.items()
    }


def _multi_observation(candidates_by_key: dict[str, list[int]]) -> dict[str, object]:
    return {
        "plans_by_key": {
            key: SimpleNamespace(
                candidate_indices=torch.tensor(candidates, dtype=torch.int64)
            )
            for key, candidates in candidates_by_key.items()
        }
    }


def _stamps_after(sequence: list[list[int]]) -> dict[str, torch.Tensor]:
    """Run `sequence` as steps 1..N against a fresh single-leaf stamp state."""
    stamps = _last_candidate_steps_for_states(_fake_states())
    for step, candidates in enumerate(sequence, start=1):
        _observe_candidates(None, stamps, _observation(candidates), step)
    return stamps


def _window_count(stamps: dict[str, torch.Tensor], step: int, window: int) -> int:
    row = _windowed_candidate_emission(stamps, step, (window,))[str(window)]
    assert row["numel_by_key"][LEAF] == NUMEL
    return row["count_by_key"][LEAF]


def test_cpu_windowed_w1_w2_w3_distinguish_recency_known_good():
    """W=1 is step 3 only; W=2 is steps 2-3; W=3 is the full 1-3 union."""
    stamps = _stamps_after([W_STEP_1, W_STEP_2, W_STEP_3])
    assert _window_count(stamps, 3, 1) == len(W_STEP_3)
    assert _window_count(stamps, 3, 2) == len(W_STEP_2) + len(W_STEP_3)
    assert _window_count(stamps, 3, 3) == len(W_STEP_1) + len(W_STEP_2) + len(W_STEP_3)


def test_cpu_never_updated_observer_fails_the_same_w1_assertion():
    """Known-bad: stamps that are never written cannot hold step 3's set."""
    stamps = _last_candidate_steps_for_states(_fake_states())
    assert _window_count(stamps, 3, 1) == 0
    with pytest.raises(AssertionError):
        assert _window_count(stamps, 3, 1) == len(W_STEP_3)


def test_cpu_windowed_emission_production_schema_and_totals():
    stamps = _stamps_after([W_STEP_1, W_STEP_2, W_STEP_3])
    rows = _windowed_candidate_emission(stamps, 3, PRODUCTION_CANDIDATE_WINDOWS)
    assert set(rows) == {"1", "10", "50"}
    for window, row in rows.items():
        assert set(row) == {
            "numel_by_key",
            "count_by_key",
            "fraction_by_key",
            "numel_total",
            "count_total",
            "fraction_total",
        }
        assert set(row["numel_by_key"]) == set(row["count_by_key"]) == {LEAF}
        assert row["numel_total"] == NUMEL
        assert row["count_total"] == row["count_by_key"][LEAF]
        assert row["fraction_total"] == pytest.approx(row["count_total"] / NUMEL)
        assert row["fraction_by_key"][LEAF] == pytest.approx(
            row["count_by_key"][LEAF] / NUMEL
        )
        assert isinstance(row["count_total"], int)
        assert window in {"1", "10", "50"}
    assert rows["1"]["count_total"] == len(W_STEP_3)
    # windows wider than the elapsed run reach the same 1-3 union, not more
    assert rows["10"]["count_total"] == rows["50"]["count_total"] == 4


def test_cpu_windowed_boundary_indices_update():
    stamps = _stamps_after([[0, NUMEL - 1]])
    assert _window_count(stamps, 1, 1) == 2


def test_cpu_negative_candidate_index_refuses_before_mutation():
    stamps = _last_candidate_steps_for_states(_fake_states())
    with pytest.raises(RuntimeError, match="candidate index out of range"):
        _observe_candidates(None, stamps, _observation([-1]), 1)
    assert int((stamps[LEAF] != NEVER_CANDIDATE_STEP).sum()) == 0


def test_cpu_upper_bound_candidate_index_refuses_before_mutation():
    stamps = _last_candidate_steps_for_states(_fake_states())
    with pytest.raises(RuntimeError, match="candidate index out of range"):
        _observe_candidates(None, stamps, _observation([NUMEL]), 1)
    assert int((stamps[LEAF] != NEVER_CANDIDATE_STEP).sum()) == 0


def test_cpu_later_invalid_leaf_leaves_all_leaves_unchanged():
    """Atomicity: an out-of-range index on any leaf blocks every write."""
    states = _multi_states({LEAF: NUMEL, LEAF_B: NUMEL})
    masks = _ever_crossed_masks_for_states(states)
    stamps = _last_candidate_steps_for_states(states)
    with pytest.raises(RuntimeError, match="candidate index out of range"):
        _observe_candidates(
            masks,
            stamps,
            _multi_observation({LEAF: [0, 1], LEAF_B: [NUMEL + 5]}),
            1,
        )
    for key in (LEAF, LEAF_B):
        assert not masks[key].any()
        assert int((stamps[key] != NEVER_CANDIDATE_STEP).sum()) == 0


def test_cpu_all_valid_multi_leaf_writes_exactly():
    states = _multi_states({LEAF: NUMEL, LEAF_B: NUMEL})
    masks = _ever_crossed_masks_for_states(states)
    stamps = _last_candidate_steps_for_states(states)
    _observe_candidates(
        masks, stamps, _multi_observation({LEAF: [0, 1], LEAF_B: [7]}), 4
    )
    assert int(masks[LEAF].sum()) == 2 and int(masks[LEAF_B].sum()) == 1
    assert int(stamps[LEAF][0]) == 4 and int(stamps[LEAF][2]) == NEVER_CANDIDATE_STEP
    assert int(stamps[LEAF_B][7]) == 4


def test_cpu_empty_candidates_no_op_and_repeats_idempotent():
    stamps = _stamps_after([[2, 2, 2], []])
    assert _window_count(stamps, 2, 2) == 1
    assert int(stamps[LEAF][2]) == 1


def test_cpu_reappearance_refreshes_last_step_and_expires_after_window():
    """Absent for two steps: outside W=1, inside W=3; reappearing refreshes."""
    stamps = _stamps_after([[5], [], []])
    assert _window_count(stamps, 3, 1) == 0
    assert _window_count(stamps, 3, 3) == 1
    _observe_candidates(None, stamps, _observation([5]), 4)
    assert int(stamps[LEAF][5]) == 4
    assert _window_count(stamps, 4, 1) == 1


def test_cpu_last_candidate_steps_are_frame_local_int32_sentinel_tensors():
    states = _fake_states()
    stamps = _last_candidate_steps_for_states(states)
    tensor = stamps[LEAF]
    assert tensor.dtype is torch.int32
    assert tensor.device.type == "cpu"
    assert tensor.dim() == 1 and tensor.numel() == NUMEL
    assert tensor.is_contiguous() and not tensor.is_sparse
    assert bool((tensor == NEVER_CANDIDATE_STEP).all())
    # frame-local: no aliasing of, or write-back into, the tensor-state operand
    assert tensor.data_ptr() != states[LEAF].q_levels.data_ptr()
    assert set(vars(states[LEAF])) == {"q_levels"}


def test_cpu_windowed_empty_denominator_refuses():
    with pytest.raises(RuntimeError, match="empty-denominator"):
        _windowed_candidate_emission({}, 1, PRODUCTION_CANDIDATE_WINDOWS)
    with pytest.raises(RuntimeError, match="empty-denominator"):
        _windowed_candidate_emission(
            _last_candidate_steps_for_states(_multi_states({LEAF: 0, LEAF_B: NUMEL})),
            1,
            PRODUCTION_CANDIDATE_WINDOWS,
        )


def test_cpu_window_below_one_refuses():
    with pytest.raises(RuntimeError, match="window must be >= 1"):
        _windowed_candidate_emission(_stamps_after([[0]]), 1, (0,))


def test_cpu_plan_key_mismatch_refuses_before_mutation():
    states = _multi_states({LEAF: NUMEL, LEAF_B: NUMEL})
    stamps = _last_candidate_steps_for_states(states)
    with pytest.raises(RuntimeError, match="key mismatch"):
        _observe_candidates(None, stamps, _multi_observation({LEAF: [0]}), 1)
    for key in (LEAF, LEAF_B):
        assert int((stamps[key] != NEVER_CANDIDATE_STEP).sum()) == 0


def test_cpu_one_producer_read_writes_both_observer_states():
    """Both frame-local states come from the same validated indices."""
    states = _fake_states()
    masks = _ever_crossed_masks_for_states(states)
    stamps = _last_candidate_steps_for_states(states)
    _observe_candidates(masks, stamps, _observation(W_STEP_1), 1)
    _observe_candidates(masks, stamps, _observation(W_STEP_3), 2)
    assert int(masks[LEAF].sum()) == len(W_STEP_1) + len(W_STEP_3)
    assert _window_count(stamps, 2, 1) == len(W_STEP_3)


def test_cpu_stale_observation_refuses_and_fresh_one_is_silent():
    """Known-bad: the observer did not fire this step; known-good: it did."""
    with pytest.raises(RuntimeError, match="did not fire on step"):
        _require_fresh_observation(2, 3)
    with pytest.raises(RuntimeError, match="did not fire on step"):
        _require_fresh_observation(None, 1)
    assert _require_fresh_observation(3, 3) is None
