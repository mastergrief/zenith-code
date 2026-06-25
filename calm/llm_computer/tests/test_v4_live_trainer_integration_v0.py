"""V4-LIVE trainer-integration CPU tests (O1-O7 outbound authority)."""
from __future__ import annotations

import copy
import os

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    _merge_event_coded_cap_tensor_stats,
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
    make_event_coded_live_tensor_state,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV,
    C8_PERSISTENT_AUTHORITY_SCOPE_KEY,
    C8StepObservation,
    assert_c8_runtime_guards,
    carrier_content_sha256,
    measure_persistent_dense_accumulator_materialized_numel,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    select_trainer_eligible_bitlinears,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(4, 4, bias=False)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _tiny_v4_state(*, demotion_band: int = 1) -> dict[str, object]:
    q = torch.zeros((4, 4), dtype=torch.int8)
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        q,
        0.25,
        demotion_band=int(demotion_band),
    )
    return {"toy.proj": state}


def _votes_for_index(flat_index: int, magnitude: int = 12) -> torch.Tensor:
    votes = torch.zeros(16, dtype=torch.int16)
    votes[int(flat_index)] = int(magnitude)
    return votes.view(4, 4)


def test_resolve_live_acc_carrier_selector_mutex_v4_w5_w6_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV, raising=False)
    monkeypatch.delenv(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, raising=False)
    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, raising=False)
    assert resolve_live_acc_carrier_selector() == "none"
    monkeypatch.setenv(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV, "1")
    assert resolve_live_acc_carrier_selector() == "v4_live"
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(w6_enabled=True)
    monkeypatch.delenv(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV, raising=False)
    monkeypatch.setenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, "1")
    assert resolve_live_acc_carrier_selector() == "w6"


def test_c8_dense_accumulator_materialized_numel_zero_on_v4_path() -> None:
    states = _tiny_v4_state()
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(0)},
        {"toy.proj": _vote_spec()},
    )
    stats = result.tensor_stats["toy.proj"]
    assert int(stats["dense_accumulator_materialized_numel"]) == 0
    assert stats.get("live_authority") == "event_coded_live_carrier"
    assert int(stats["full_numel_flatten_count"]) >= 1
    assert int(stats["transient_dense_compute_numel"]) == 16
    assert stats.get(C8_PERSISTENT_AUTHORITY_SCOPE_KEY) is not None
    assert "transient O(numel) runtime buffers remain" in str(
        stats.get(C8_PERSISTENT_AUTHORITY_SCOPE_KEY)
    )


def test_c8_persistent_dense_authority_guard_trips_on_shadow() -> None:
    carrier = _tiny_v4_state()["toy.proj"].event_coded_live_carrier  # type: ignore[union-attr]
    assert carrier is not None
    observation = C8StepObservation()
    shadow = torch.zeros(16, dtype=torch.int16)
    with pytest.raises(ValueError, match="dense persistent accumulator authority"):
        assert_c8_runtime_guards(
            carrier,
            observation=observation,
            persistent_dense_accumulator_materialized_numel=measure_persistent_dense_accumulator_materialized_numel(
                exact_accumulator_shadow=shadow,
                event_coded_live_carrier=carrier,
                eligible_numel=16,
            ),
        )


def test_c8_full_numel_flatten_count_not_hardcoded_zero() -> None:
    from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
        EventCodedVoteUpdateState,
        apply_event_coded_integer_vote_update_reference,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateInputs

    states = _tiny_v4_state()
    prior = states["toy.proj"]
    vu = prior.vote_update_state()
    assert isinstance(vu, EventCodedVoteUpdateState)
    result = apply_event_coded_integer_vote_update_reference(
        vu,
        VoteUpdateInputs(votes=_votes_for_index(0)),
        _vote_spec(),
    )
    assert int(result.stats["full_numel_flatten_count"]) >= 1
    assert int(result.stats["transient_dense_compute_numel"]) == 16


def test_o1_multi_step_v4_carrier_consumed_by_next_step() -> None:
    states = _tiny_v4_state()
    spec = _vote_spec()
    step1 = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(0, magnitude=8)},
        {"toy.proj": spec},
        local_selection_ordering_step=0,
    )
    carrier_sha_1 = carrier_content_sha256(
        step1.tensor_states["toy.proj"].event_coded_live_carrier  # type: ignore[arg-type]
    )
    step2 = apply_bounded_delta_vote_step(
        step1.tensor_states,
        {"toy.proj": _votes_for_index(1, magnitude=8)},
        {"toy.proj": spec},
        local_selection_ordering_step=1,
    )
    next_state = step2.tensor_states["toy.proj"]
    assert next_state.event_coded_live_carrier is not None
    carrier_sha_2 = carrier_content_sha256(next_state.event_coded_live_carrier)
    assert carrier_sha_2 != carrier_sha_1


def test_o2_no_dense_eligible_numel_tensor_in_v4_next_states() -> None:
    states = _tiny_v4_state()
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(2)},
        {"toy.proj": _vote_spec()},
    )
    next_state = result.tensor_states["toy.proj"]
    assert next_state.exact_accumulator_shadow is None
    assert next_state.event_coded_live_carrier is not None
    eligible = int(next_state.q_levels.numel())
    assert eligible == 16


def test_o3_no_shadow_sha_as_live_authority_on_v4_path() -> None:
    states = _tiny_v4_state()
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(3)},
        {"toy.proj": _vote_spec()},
    )
    stats = result.tensor_stats["toy.proj"]
    assert stats.get("exact_accumulator_shadow_sha256_after") is None
    assert stats.get("event_coded_live_carrier_content_sha256_after") is not None
    assert stats.get("live_authority") == "event_coded_live_carrier"


def test_o4_checkpoint_v1_roundtrip_hydrates_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV, "1")
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    dense_states = derive_trainer_sub2_authority_states(eligible)
    states = {
        key: make_event_coded_live_tensor_state(
            key,
            state.q_levels,
            state.frozen_scale,
            demotion_band=1,
        )
        for key, state in dense_states.items()
    }
    carrier = states["proj"].event_coded_live_carrier
    assert carrier is not None
    carrier.apply_step(0, votes={0: 8})
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        step=1,
    )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    loaded = load_trainer_sub2_authority_checkpoint_blob(
        fresh,
        blob,
        eligible_modules=fresh_eligible,
        event_coded_enabled=True,
    )
    loaded_carrier = loaded["proj"].event_coded_live_carrier
    assert loaded_carrier is not None
    assert loaded["proj"].exact_accumulator_shadow is None
    payload = blob["trainer_sub2_authority"]["tensor_payloads"]["proj"]
    assert payload["event_coded_live_carrier_schema"] == EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1


def test_o5_global_cap_writes_through_event_coded_carrier() -> None:
    states = _tiny_v4_state()
    cap_spec = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(0)},
        {"toy.proj": _vote_spec()},
        global_cap_spec=cap_spec,
    )
    assert result.global_summary.get("event_coded_live_carrier_enabled") is True
    next_state = result.tensor_states["toy.proj"]
    assert next_state.event_coded_live_carrier is not None


def test_o5b_global_cap_event_coded_stats_include_post_cap_indices() -> None:
    states = _tiny_v4_state()
    votes = torch.zeros((4, 4), dtype=torch.int16)
    vote_flat = votes.view(-1)
    vote_flat[0] = 12
    vote_flat[1] = 12
    vote_flat[2] = 12
    cap_spec = GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True)
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": votes},
        {"toy.proj": _vote_spec()},
        global_cap_spec=cap_spec,
    )
    stats = result.tensor_stats["toy.proj"]
    assert stats.get("global_rate_cap_enabled") is True
    assert "post_veto_applied_indices" in stats
    post_cap_indices = stats["post_veto_applied_indices"]
    assert stats["post_veto_applied_flip_count"] == len(post_cap_indices)
    assert stats["post_veto_would_apply_pre_cap_count"] == 3
    assert len(post_cap_indices) == 1
    assert stats["post_veto_applied_flip_count"] != stats["post_veto_would_apply_pre_cap_count"]
    assert stats.get("live_authority") == "event_coded_live_carrier"
    assert stats.get("event_coded_live_carrier_content_sha256_after") is not None
    assert stats.get(C8_PERSISTENT_AUTHORITY_SCOPE_KEY) is not None


def test_merge_event_coded_cap_stats_prefers_post_cap_over_pre_cap_local() -> None:
    """C2 regression: blanket local_result.stats overlay must not clobber post-cap counts."""
    cap_stats = {
        "post_veto_applied_flip_count": 1,
        "post_veto_applied_indices": [7],
        "post_veto_would_apply_pre_cap_count": 4096,
        "global_rate_cap_enabled": True,
        "flip_count": 1,
    }
    local_stats = {
        "post_veto_applied_flip_count": 4096,
        "post_veto_would_apply_pre_cap_count": 4096,
        "live_authority": "event_coded_live_carrier",
        "event_coded_live_carrier_content_sha256_after": "abc123",
        C8_PERSISTENT_AUTHORITY_SCOPE_KEY: "event_coded_live_carrier_only",
        "transient_dense_compute_numel": 16,
    }
    merged = _merge_event_coded_cap_tensor_stats(cap_stats, local_stats)
    assert merged["post_veto_applied_flip_count"] == 1
    assert merged["post_veto_applied_indices"] == [7]
    assert merged["post_veto_would_apply_pre_cap_count"] == 4096
    assert merged["live_authority"] == "event_coded_live_carrier"
    assert merged["event_coded_live_carrier_content_sha256_after"] == "abc123"
    assert merged["exact_accumulator_shadow_sha256_after"] is None
    assert merged[C8_PERSISTENT_AUTHORITY_SCOPE_KEY] == "event_coded_live_carrier_only"
    assert merged["transient_dense_compute_numel"] == 16


def test_o6_disabled_dense_path_byte_stable() -> None:
    q = torch.zeros((2, 2), dtype=torch.int8)
    dense_state = make_bounded_tensor_state("toy.proj", q, 0.25)
    votes = torch.zeros(4, dtype=torch.int16).view(2, 2)
    spec = _vote_spec()
    before_sha = tensor_sha256(dense_state.q_levels)
    result = apply_bounded_delta_vote_step(
        {"toy.proj": dense_state},
        {"toy.proj": votes},
        {"toy.proj": spec},
    )
    next_state = result.tensor_states["toy.proj"]
    assert next_state.exact_accumulator_shadow is not None
    assert tensor_sha256(next_state.q_levels) == before_sha


def test_o7_four_point_unit_event_coded_outbound_not_shadow() -> None:
    states = _tiny_v4_state()
    prior = states["toy.proj"]
    result = apply_bounded_delta_vote_step(
        states,
        {"toy.proj": _votes_for_index(0, magnitude=9)},
        {"toy.proj": _vote_spec()},
    )
    next_state = result.tensor_states["toy.proj"]
    assert prior.event_coded_live_carrier is not None
    assert next_state.event_coded_live_carrier is not None
    assert next_state.exact_accumulator_shadow is None
    assert next_state.bounded_accumulator_fresh_for_exact_shadow is False


def test_dense_vote_update_rejects_event_coded_stub_accumulator() -> None:
    from calm.hrm_text_158.native_full_stack.vote_update import (
        VoteUpdateAccumulatorFormat,
        VoteUpdateInputs,
        VoteUpdateState,
        plan_integer_vote_update_reference,
    )

    q = torch.zeros((2, 2), dtype=torch.int8)
    state = VoteUpdateState(
        q_levels=q,
        accumulators=torch.zeros_like(q, dtype=torch.int16),
        accumulator_format=VoteUpdateAccumulatorFormat.EVENT_CODED_LIVE_CARRIER,
    )
    with pytest.raises(ValueError, match="forbidden on event-coded live carrier path"):
        plan_integer_vote_update_reference(
            state,
            VoteUpdateInputs(votes=torch.zeros_like(q, dtype=torch.int16)),
            _vote_spec(),
        )


def test_grep_persistent_state_budget_codec_import_free() -> None:
    from pathlib import Path

    text = Path(
        "calm/hrm_text_158/native_full_stack/persistent_state_budget.py"
    ).read_text(encoding="utf-8")
    assert "event_coded_acc_checkpoint_codec" not in text
