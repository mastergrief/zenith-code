"""R4v ledger must stay q-unpacked even when global BASE3 codec env is set."""
from __future__ import annotations

import pytest
import torch

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV,
    Q_CODEC_SELECTOR_BASE3,
    _pack_q_for_checkpoint,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _make_event_coded_state() -> dict[str, object]:
    logical_numel = 256
    side = 16
    q = torch.zeros((side, side), dtype=torch.int8)
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        q,
        0.25,
        demotion_band=1,
    )
    vote_spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )
    votes = torch.zeros(logical_numel, dtype=torch.int16)
    votes[0] = 12
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes.view(side, side)},
        {"toy.proj": vote_spec},
    )
    return {"toy.proj": state}


def test_r4v_build_succeeds_under_base3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV, "1")
    ledger = probe.build_r4v_persistent_ledger_receipt(
        _make_event_coded_state(),
        event_coded_live_enabled=True,
    )
    assert ledger["enabled"] is True
    assert ledger["ledger_pass"] is True
    assert "content_sha256" in ledger


def test_save_time_base3_guard_still_raises_without_master_pack() -> None:
    q = torch.zeros(8, dtype=torch.int8)
    with pytest.raises(ValueError, match="base-3 q codec selector requires"):
        _pack_q_for_checkpoint(
            q,
            q_packed_enabled=False,
            q_codec_selector=Q_CODEC_SELECTOR_BASE3,
        )
