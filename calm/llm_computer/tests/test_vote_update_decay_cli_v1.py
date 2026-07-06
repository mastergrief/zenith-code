"""CPU-static tests for vote-update decay CLI wiring in the acquisition probe."""

from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import ReplayConstants
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_arg_parser,
    resolve_probe_vote_update_spec,
)


def test_no_flags_default_decay_1_over_1_in_vote_spec_and_replay_constants() -> None:
    vote_spec = resolve_probe_vote_update_spec(
        max_abs_per_tensor=4096,
        confirmation_envelope="canonical_t10_prereg_v24",
    )
    assert vote_spec.decay_numerator == 1
    assert vote_spec.decay_denominator == 1
    replay = ReplayConstants.from_vote_update_spec(vote_spec)
    assert replay.decay_numerator == 1
    assert replay.decay_denominator == 1


def test_decay_flags_1_over_2_in_vote_spec_and_replay_constants() -> None:
    vote_spec = resolve_probe_vote_update_spec(
        max_abs_per_tensor=4096,
        confirmation_envelope="canonical_t10_prereg_v24",
        vote_update_decay_numerator=1,
        vote_update_decay_denominator=2,
    )
    assert vote_spec.decay_numerator == 1
    assert vote_spec.decay_denominator == 2
    replay = ReplayConstants.from_vote_update_spec(vote_spec)
    assert replay.decay_numerator == 1
    assert replay.decay_denominator == 2


def test_decay_denominator_zero_fail_closed() -> None:
    with pytest.raises(ValueError, match="decay_denominator"):
        resolve_probe_vote_update_spec(
            max_abs_per_tensor=4096,
            confirmation_envelope=None,
            vote_update_decay_numerator=1,
            vote_update_decay_denominator=0,
        )


def test_argparse_exposes_decay_flags() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--vote-update-decay-numerator",
            "1",
            "--vote-update-decay-denominator",
            "2",
        ]
    )
    assert args.vote_update_decay_numerator == 1
    assert args.vote_update_decay_denominator == 2
