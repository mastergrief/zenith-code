"""Production wire contract for carrier growth sidecar (default-off + isolation)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
    CarrierGrowthCollector,
    maybe_emit_carrier_growth_step_record,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    EventCodedVoteUpdateState,
)


def test_carrier_growth_default_off_emits_nothing(tmp_path: Path) -> None:
    result = maybe_emit_carrier_growth_step_record(
        enabled=False,
        collector=None,
        optimizer_step_index=1,
        tensor_states={},
        votes_by_key={},
        tensor_stats_by_key={},
    )
    assert result is None
    assert not (tmp_path / "votes_emit" / "v1" / "carrier_growth").exists()


def test_carrier_growth_gated_on_writes_compact_sidecar(tmp_path: Path) -> None:
    key = "toy.proj"
    q = torch.zeros(64, dtype=torch.int8)
    state = make_event_coded_live_tensor_state(key, q, 1.0, demotion_band=1)
    vu = state.vote_update_state()
    assert isinstance(vu, EventCodedVoteUpdateState)
    votes = torch.zeros(64, dtype=torch.int16)
    votes[1] = 12
    carrier = vu.carrier
    carrier.apply_step(0, votes={1: 12})

    collector = CarrierGrowthCollector(tmp_path)
    encode_patch = (
        "calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec"
        ".encode_event_coded_acc_events"
    )
    with mock.patch(encode_patch) as encode_mock, mock.patch.object(
        EventCodedAccLiveState,
        "hot_packed_bytes",
        autospec=True,
    ) as hot_pack_mock:
        result = maybe_emit_carrier_growth_step_record(
            enabled=True,
            collector=collector,
            optimizer_step_index=1,
            tensor_states={key: state},
            votes_by_key={key: votes},
            tensor_stats_by_key={key: {"q_changed_count": 1, "global_rate_cap_accepted_count": 0}},
        )
        encode_mock.assert_not_called()
        hot_pack_mock.assert_not_called()

    assert result is not None
    step_path = Path(result["step_path"])
    assert step_path.exists()
    payload = step_path.read_text(encoding="utf-8")
    assert "carrier_growth" in str(step_path)
    assert '"compact":true' in payload.replace(" ", "")
    assert '"modules"' not in payload
