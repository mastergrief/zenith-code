"""Phase-1 Slice 2B global-rate-cap reference tests.

Slice 2B is CPU/reference control-flow glue by design. It consumes Slice 2A
VoteUpdatePlan objects and proves cap-bounded cross-tensor apply semantics
without trainer integration, GPU work, functional-veto probes, or bad-pressure
drain.
"""
from __future__ import annotations

import json

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    CAP_ORDERING_HASH_SEED,
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    CPU_GLUE_NOT_KERNEL_NOTE,
    DEFERRED_NON_SCOPE,
    DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE,
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GLOBAL_CAP_CONTRACT_OFF,
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    c1_banked_faithful_long_run_global_cap_contract,
    c1_banked_faithful_long_run_global_cap_for_step,
    named_global_cap_contract_receipt,
    resolve_named_global_cap_spec,
    scratch_s1_global_cap_contract,
    scratch_s1_global_cap_for_step,
    select_global_rate_cap_rows,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


def _spec(**kwargs) -> VoteUpdateSpec:
    base = dict(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    base.update(kwargs)
    return VoteUpdateSpec(**base)


def _state(q: list[int], acc: list[int]) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=torch.tensor(q, dtype=torch.int8),
        accumulators=torch.tensor(acc, dtype=torch.int16),
    )


def _inputs(votes: list[int], **kwargs) -> VoteUpdateInputs:
    converted = {}
    for name, value in kwargs.items():
        if value is None:
            converted[name] = None
        elif name.endswith("moves"):
            converted[name] = torch.tensor(value, dtype=torch.int8)
        else:
            converted[name] = torch.tensor(value, dtype=torch.int16)
    return VoteUpdateInputs(votes=torch.tensor(votes, dtype=torch.int16), **converted)


def _tensor_input(
    state_key: str,
    q: list[int],
    acc: list[int],
    votes: list[int],
    **vote_kwargs,
) -> GlobalRateCapTensorInput:
    state = _state(q, acc)
    inputs = _inputs(votes, **vote_kwargs)
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan,
        vote_inputs=inputs,
    )


def test_scratch_s1_global_cap_schedule_and_contract():
    assert scratch_s1_global_cap_for_step(1) == 512
    assert scratch_s1_global_cap_for_step(256) == 512
    assert scratch_s1_global_cap_for_step(257) == 1024
    assert scratch_s1_global_cap_for_step(512) == 1024
    assert scratch_s1_global_cap_for_step(9999) == 1024
    with pytest.raises(ValueError, match="step must be >=1"):
        scratch_s1_global_cap_for_step(0)

    contract = scratch_s1_global_cap_contract()
    assert contract["start"] == 512
    assert contract["max"] == 1024
    assert contract["anneal_step"] == 256
    assert "q_changed_count == global_rate_cap_applied_count" in contract["per_step_assertions"]


def test_c1_banked_faithful_long_run_contract_schedule_and_named_resolution():
    assert c1_banked_faithful_long_run_global_cap_for_step(1) == 512
    assert c1_banked_faithful_long_run_global_cap_for_step(2) == 512
    assert c1_banked_faithful_long_run_global_cap_for_step(3) == 256
    assert c1_banked_faithful_long_run_global_cap_for_step(9999) == 256
    with pytest.raises(ValueError, match="step must be >=1"):
        c1_banked_faithful_long_run_global_cap_for_step(0)

    contract = c1_banked_faithful_long_run_global_cap_contract()
    assert contract["name"] == C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    assert contract["finite_schedule_source"] == [512, 512, 256, 256]
    assert contract["long_run_translation"] == "steps 1..2 cap=512; steps >=3 cap=256"

    off_receipt = named_global_cap_contract_receipt(GLOBAL_CAP_CONTRACT_OFF)
    assert off_receipt["name"] == GLOBAL_CAP_CONTRACT_OFF
    assert off_receipt["enabled"] is False

    named_receipt = named_global_cap_contract_receipt(
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    )
    assert named_receipt["name"] == C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    assert named_receipt["active_runtime_control"] is False

    resolved = resolve_named_global_cap_spec(
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        step=4,
    )
    assert resolved is not None
    assert resolved.cap == 256
    assert resolved.step == 4
    assert resolve_named_global_cap_spec(GLOBAL_CAP_CONTRACT_OFF, step=4) is None


def test_ordering_modes_match_live_static_dispatch():
    a = _tensor_input(
        "A.synthetic",
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 30, 0, 30, 0, 30],
    )
    b = _tensor_input(
        "B.synthetic",
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [30, 0, 30, 0, 30],
    )
    offsets = {"A.synthetic": 0, "B.synthetic": 1000}

    margin_rows, margin_selected, _ = select_global_rate_cap_rows(
        [a, b],
        GlobalRateCapSpec(cap=3, step=100),
        tensor_offsets=offsets,
    )
    _, hash_selected, _ = select_global_rate_cap_rows(
        [a, b],
        GlobalRateCapSpec(
            cap=3,
            step=100,
            ordering_mode=GlobalRateCapOrderingMode.HASH_SHUFFLE,
            ordering_seed=CAP_ORDERING_HASH_SEED,
        ),
        tensor_offsets=offsets,
    )
    _, hash_selected_again, _ = select_global_rate_cap_rows(
        [a, b],
        GlobalRateCapSpec(
            cap=3,
            step=100,
            ordering_mode=GlobalRateCapOrderingMode.HASH_SHUFFLE,
            ordering_seed=CAP_ORDERING_HASH_SEED,
        ),
        tensor_offsets=offsets,
    )
    _, round_robin_selected, _ = select_global_rate_cap_rows(
        [a, b],
        GlobalRateCapSpec(cap=3, step=100, ordering_mode=GlobalRateCapOrderingMode.ROUND_ROBIN),
        tensor_offsets=offsets,
    )

    assert [row.global_flat_index for row in margin_rows] == [1, 3, 5, 1000, 1002, 1004]
    assert [row.global_flat_index for row in margin_selected] == [1, 3, 5]
    assert [row.global_flat_index for row in hash_selected] == [
        row.global_flat_index for row in hash_selected_again
    ]
    assert [row.global_flat_index for row in hash_selected] != [1, 3, 5]
    assert [row.global_flat_index for row in round_robin_selected] == [1, 1000, 3]


def test_cap_apply_distinguishes_2a_candidates_from_2b_applied_rows():
    item = _tensor_input(
        "synthetic.cap",
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [30, 30, 30, 0],
        replay_ce_veto_votes=[0, -1, 0, 0],
        replay_ce_veto_moves=[0, 0, 0, 0],
    )
    assert item.plan.pre_veto_selected_indices.tolist() == [0, 1, 2]
    assert item.plan.replay_ce_veto_indices.tolist() == [1]
    # 2A "applied" means local post-veto/pre-global-cap candidates.
    assert item.plan.applied_indices.tolist() == [0, 2]

    result = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets={"synthetic.cap": 0},
    )
    tensor = result.tensor_results[0]

    assert [row.flat_index for row in result.accepted_rows] == [0]
    assert [row.flat_index for row in result.deferred_rows] == [2]
    assert tensor.stats["two_b_input_name"] == (
        "2A applied_indices are local_post_veto_pre_global_cap_candidates"
    )
    assert tensor.stats["global_rate_cap_accepted_indices"] == [0]
    assert tensor.stats["global_rate_cap_deferred_indices"] == [2]

    # Accepted: q mutates and residual subtracts/clamps.
    assert int(tensor.q_levels[0].item()) == 1
    assert int(tensor.accumulators[0].item()) == 9
    # Replay-vetoed: residual subtracts/clamps, but q does not mutate.
    assert int(tensor.q_levels[1].item()) == 0
    assert int(tensor.accumulators[1].item()) == 9
    # Cap-deferred: no q mutation and no residual subtraction.
    assert int(tensor.q_levels[2].item()) == 0
    assert int(tensor.accumulators[2].item()) == 30
    assert result.step_summary["global_rate_cap_applied_count"] == 1
    assert result.step_summary["q_changed_count"] == 1
    assert result.step_summary["global_rate_cap_saturated"] is True
    assert result.deferred_backlog["synthetic.cap"][2]["defer_count"] == 1

    # Reference path returns copies; original state tensors are untouched.
    assert item.state.q_levels.tolist() == [0, 0, 0, 0]
    assert item.state.accumulators.tolist() == [0, 0, 0, 0]


def test_deferred_backlog_carries_and_clears_when_prior_row_is_accepted():
    item = _tensor_input(
        "synthetic.cap",
        [0, 0, 0],
        [0, 0, 0],
        [30, 30, 30],
    )
    first = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets={"synthetic.cap": 0},
    )
    assert sorted(first.deferred_backlog["synthetic.cap"]) == [1, 2]

    second = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=2, step=2),
        deferred_backlog=first.deferred_backlog,
        tensor_offsets={"synthetic.cap": 0},
    )

    assert second.step_summary["accepted_from_prior_deferred_count"] == 1
    assert sorted(second.deferred_backlog["synthetic.cap"]) == [2]
    assert first.deferred_backlog["synthetic.cap"][1]["defer_count"] == 1
    assert second.deferred_backlog["synthetic.cap"][2]["defer_count"] == 2


def test_mutate_outputs_false_freezes_returned_q_and_accumulators():
    item = _tensor_input(
        "synthetic.freeze",
        [0, 0, 0],
        [0, 0, 0],
        [30, 30, 30],
        replay_ce_veto_votes=[0, -1, 0],
        replay_ce_veto_moves=[0, 0, 0],
    )

    result = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=1, mutate_outputs=False),
        tensor_offsets={"synthetic.freeze": 0},
    )
    tensor = result.tensor_results[0]

    assert [row.flat_index for row in result.accepted_rows] == [0]
    assert item.plan.replay_ce_veto_indices.tolist() == [1]
    assert tensor.q_levels.tolist() == [0, 0, 0]
    assert tensor.accumulators.tolist() == [0, 0, 0]
    assert tensor.stats["ternary_mutation_frozen"] is True
    assert tensor.stats["global_rate_cap_accepted_indices"] == [0]
    assert tensor.stats["post_veto_applied_indices"] == []
    assert result.step_summary["global_rate_cap_accepted_count"] == 1
    assert result.step_summary["global_rate_cap_applied_count"] == 0
    assert result.step_summary["q_changed_count"] == 0


def test_deferred_non_scope_guards_and_cpu_glue_honesty():
    with pytest.raises(NotImplementedError, match="functional-window veto"):
        GlobalRateCapSpec(cap=1, step=1, functional_veto_policy="enabled").validate()
    with pytest.raises(NotImplementedError, match="bad-pressure drain"):
        GlobalRateCapSpec(cap=1, step=1, bad_pressure_drain_policy="enabled").validate()

    item = _tensor_input("synthetic.cpu_glue", [0], [0], [30])
    result = apply_global_rate_cap_reference([item], GlobalRateCapSpec(cap=1, step=1))

    assert result.step_summary["functional_veto_policy"] == DEFERRED_NON_SCOPE
    assert result.step_summary["bad_pressure_drain_policy"] == DEFERRED_NON_SCOPE
    assert result.step_summary["cpu_glue_not_kernel"] is True
    assert "no GPU receipt" in CPU_GLUE_NOT_KERNEL_NOTE


def test_defer_all_no_backfill_uses_same_pre_state_shadow_and_drops_mixed_class_accepts():
    item = _tensor_input(
        "synthetic.defer_all",
        [0, 0],
        [0, 0],
        [30, 30],
    )

    exact = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=3),
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        contract_name=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tensor_offsets={"synthetic.defer_all": 0},
    )
    defer_all = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=3),
        tie_rule_mode=DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE,
        contract_name=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tensor_offsets={"synthetic.defer_all": 0},
    )

    assert exact.step_summary["global_tie_rule_mode"] == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    assert defer_all.step_summary["global_tie_rule_mode"] == DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE
    assert defer_all.step_summary["global_rate_cap_contract_name"] == (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    )
    assert defer_all.step_summary["drop_exercised_basis"] == "same_step_same_pre_state_shadow"
    assert defer_all.step_summary["exact_shadow_full_demand_sha256"] == defer_all.step_summary["defer_full_demand_sha256"]
    assert defer_all.step_summary["mixed_class_count"] == 1
    assert defer_all.step_summary["mixed_class_row_count"] == 2
    assert defer_all.step_summary["dropped_mass_count"] == 1
    assert defer_all.step_summary["drop_exercised"] is True
    assert exact.step_summary["global_rate_cap_accepted_count"] == 1
    assert defer_all.step_summary["global_rate_cap_accepted_count"] == 0
    assert defer_all.step_summary["global_rate_cap_deferred_count"] == 2
    assert defer_all.step_summary["exact_shadow_accepted_count"] == 1
    assert defer_all.step_summary["defer_accepted_count"] == 0
    assert [row.flat_index for row in exact.accepted_rows] == [0]
    assert defer_all.accepted_rows == []
    assert sorted(row.flat_index for row in defer_all.deferred_rows) == [0, 1]
    json.dumps(defer_all.step_summary, sort_keys=True)
