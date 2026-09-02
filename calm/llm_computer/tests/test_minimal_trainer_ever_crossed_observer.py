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
    _ever_crossed_emission,
    _ever_crossed_masks_for_states,
    _or_accumulate_ever_crossed,
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
