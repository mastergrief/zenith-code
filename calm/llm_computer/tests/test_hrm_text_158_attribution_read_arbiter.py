"""Executed calibrations for the attribution-read reducer.

Authoritative calibration lives here, not in a design document.
Polarity (advisor class cure): known-bad FIRE = the world that SELECTS the
check's declared consequence; known-good SILENT = that consequence absent,
emitted fields asserted.

Borrowed instrument (license term 1786869820566-5d7d594b):
test_front_c_identity_observer_is_logging_only_and_cloned is imported and run
below. It is not wrapped and not reimplemented.
"""

from __future__ import annotations

from calm.hrm_text_158.native_full_stack.attribution_read_arbiter import (
    APPLIED,
    BRANCH_B,
    BRANCH_C,
    BRANCH_D,
    BRANCH_E,
    BRANCH_F,
    BRANCH_RESIDUAL_STOP,
    BRANCH_STOP,
    INVALID,
    REPRODUCED_LISTED_CARDINALITY,
    VETO_RESIDUAL,
    acc_clamped,
    apply_side_count_correct,
    classify,
    compose_front_c_observer,
    CONSUMED_INPUTS_INVENTORY_KEY,
    control_emission,
    is_untouched,
    row_type,
)
import calm.llm_computer.tests.test_hrm_text_158_native_front_c_live_identity_emission as _front_c_mod

N = REPRODUCED_LISTED_CARDINALITY


def _listed() -> list[int]:
    return list(range(N))


def _types(kind: str, indices: list[int]) -> dict[int, str]:
    return {i: kind for i in indices}


def _this_module_test_names() -> list[str]:
    return sorted(
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )


def test_suite_denominator_is_emitted_from_this_module() -> None:
    names = _this_module_test_names()
    assert len(names) == len(SUITE_TEST_NAMES)
    assert names == SUITE_TEST_NAMES


def test_row_type_APPLIED_fires() -> None:
    assert row_type(q_changed=True, acc_clamped=True) == APPLIED


def test_row_type_APPLIED_silent_on_q_only() -> None:
    assert row_type(q_changed=True, acc_clamped=False) != APPLIED
    assert row_type(q_changed=True, acc_clamped=False) == INVALID


def test_row_type_VETO_RESIDUAL_fires() -> None:
    assert row_type(q_changed=False, acc_clamped=True) == VETO_RESIDUAL


def test_row_type_INVALID_both_false_fires_stop() -> None:
    listed = _listed()
    types = _types(INVALID, listed)
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=types,
    )
    assert result["branch"] == BRANCH_STOP
    assert result["reason"] == "invalid_quadrant"


def test_row_type_INVALID_q_only_fires_stop() -> None:
    listed = _listed()
    types = {i: INVALID for i in listed}
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=types,
    )
    assert result["branch"] == BRANCH_STOP
    assert result["reason"] == "invalid_quadrant"


def test_row_type_INVALID_silent_when_all_applied() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] != BRANCH_STOP
    assert result["reason"] != "invalid_quadrant"
    assert result["examined_cardinality"] == N


def test_apply_side_count_correct_false_when_invalid_present() -> None:
    examined = {0, 1}
    assert (
        apply_side_count_correct(
            r_veto=set(),
            r_invalid={1},
            r_applied={0},
            examined=examined,
        )
        is False
    )


def test_apply_side_count_correct_true_when_all_applied() -> None:
    examined = {0, 1}
    assert (
        apply_side_count_correct(
            r_veto=set(),
            r_invalid=set(),
            r_applied=examined,
            examined=examined,
        )
        is True
    )


def test_B_fires_when_count_correct_and_mode_fed_disagrees() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(0,),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] == BRANCH_B
    assert result["reason"] == "mode_fed_disagrees"
    assert result["examined_cardinality"] == N


def test_B_silent_when_mode_fed_agrees() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] != BRANCH_B
    assert result["branch"] == BRANCH_E
    assert result["examined_cardinality"] == N


def test_E_fires_when_count_correct_and_mode_fed_agrees_and_949() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] == BRANCH_E
    assert result["reason"] == "comparison_ill_posed"


def test_E_silent_when_mode_fed_veto_nonempty() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(1,),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] != BRANCH_E
    assert result["branch"] == BRANCH_B


def test_D_fires_when_listed_cardinality_not_949() -> None:
    result = classify(
        listed=(),
        apply_applied=(),
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types={},
    )
    assert result["branch"] == BRANCH_D
    assert result["reason"] == "D_listed_cardinality_not_949"
    assert result["listed_cardinality"] == 0


def test_D_silent_on_reproduced_pair() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] != BRANCH_D
    assert result["listed_cardinality"] == N


def test_C_fires_when_listed_mixed() -> None:
    listed = _listed()
    types = {i: APPLIED if i < 10 else VETO_RESIDUAL for i in listed}
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types=types,
    )
    assert result["branch"] == BRANCH_C
    assert result["n_listed_APPLIED"] == 10
    assert result["n_listed_VETO_RESIDUAL"] == N - 10


def test_C_silent_when_listed_uniform_applied() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] != BRANCH_C
    assert result["branch"] == BRANCH_E
    assert result["examined_cardinality"] == N
    assert result["listed_cardinality"] == N


def test_F_fires_when_all_examined_veto_and_L_mode_empty() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types=_types(VETO_RESIDUAL, listed),
    )
    assert result["branch"] == BRANCH_F
    assert result["reason"] == "mode_contract_violation"
    assert result["R_VETO_cardinality"] == N


def test_F_silent_when_L_mode_nonempty_varies_the_conjunct() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(0,),
        mode_fed_applied=(),
        types=_types(VETO_RESIDUAL, listed),
    )
    assert result["branch"] != BRANCH_F
    assert result["branch"] == BRANCH_RESIDUAL_STOP


def test_F_silent_on_C_world() -> None:
    listed = _listed()
    types = {i: APPLIED if i < 10 else VETO_RESIDUAL for i in listed}
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types=types,
    )
    assert result["branch"] != BRANCH_F
    assert result["branch"] == BRANCH_C


def test_STOP_control_fires() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
        control_failing=(10_000,),
    )
    assert result["branch"] == BRANCH_STOP
    assert result["reason"] == "control_untouched_failed"


def test_STOP_control_silent_when_none_failing() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
        control_failing=(),
    )
    assert result["reason"] != "control_untouched_failed"


def test_STOP_listed_absent_fires() -> None:
    result = classify(
        listed=(5,),
        apply_applied=(1, 2),
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types={},
    )
    assert result["branch"] == BRANCH_STOP
    assert result["reason"] == "listed_absent_from_apply"


def test_STOP_listed_absent_silent_when_subset() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=0,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["reason"] != "listed_absent_from_apply"


def test_control_emission_budget_compact() -> None:
    indices = list(range(10_000))
    untouched = {i: True for i in indices}
    emitted = control_emission(
        control_indices=indices,
        untouched_by_index=untouched,
    )
    assert emitted["control_population_size"] == 10_000
    assert emitted["control_examined_count"] == 10_000
    assert emitted["control_failing_count"] == 0
    assert emitted["per_row_raw_emitted"] == 0
    assert emitted["control_extremal_failing_row"] is None
    assert len(emitted["control_hashes"]["untouched_mask_sha256"]) == 64


def test_control_emission_known_bad_would_be_per_row_dump() -> None:
    # The reducer refuses to emit per-row raw for the whole population.
    indices = list(range(100))
    emitted = control_emission(
        control_indices=indices,
        untouched_by_index={i: True for i in indices},
    )
    assert emitted["per_row_raw_emitted"] == 0
    assert "rows" not in emitted


def test_acc_clamped_and_untouched_identities() -> None:
    assert acc_clamped(
        acc_after=0, acc_pre_writeback=10, direction=1, threshold=10
    )
    assert not acc_clamped(
        acc_after=10, acc_pre_writeback=10, direction=1, threshold=10
    )
    assert is_untouched(q_before=0, q_after=0, acc_pre_writeback=4, acc_after=4)
    assert not is_untouched(q_before=0, q_after=1, acc_pre_writeback=4, acc_after=4)


def test_compose_never_replaces_existing_observer() -> None:
    seen: list[str] = []

    def existing(observation, **kwargs):
        seen.append("existing")
        return "kept"

    capture: list[object] = []
    composed = compose_front_c_observer(existing, capture)
    out = composed({"k": 1})
    assert out == "kept"
    assert seen == ["existing"]
    assert capture == [{"k": 1}]


def test_compose_installs_when_existing_is_none() -> None:
    capture: list[object] = []
    composed = compose_front_c_observer(None, capture)
    assert composed is not None
    assert composed({"k": 2}) is None
    assert capture == [{"k": 2}]
    assert compose_front_c_observer(None, None) is None


def _legacy_front_c_key_set() -> set[str]:
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        _front_c_cloned_observation,
    )

    return set(
        _front_c_cloned_observation(
            vote_update_states={},
            inputs_by_key={},
            vote_specs_by_key={},
            plans_by_key={},
            q_acc_by_key={},
            deferred_backlog=None,
            global_cap_used=False,
        ).keys()
    )


def _toy_apply_kwargs():
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        make_bounded_tensor_state,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

    state = make_bounded_tensor_state(
        "toy.weight",
        torch.zeros(8, dtype=torch.int8),
        1.0,
        torch.zeros(8, dtype=torch.int16),
    )
    votes = torch.zeros(8, dtype=torch.int16)
    votes[0] = 2
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )
    return (
        {"toy.weight": state},
        {"toy.weight": votes},
        {"toy.weight": spec},
    )


def test_dedicated_sink_off_keeps_legacy_front_c_surface_and_does_no_inventory_work() -> None:
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
    )

    calls = {"n": 0}
    original = learner.emit_object_inventory

    def counting_emit(obj, **kwargs):
        calls["n"] += 1
        return original(obj, **kwargs)

    learner.emit_object_inventory = counting_emit
    try:
        front_c: list[dict] = []
        states, votes, specs = _toy_apply_kwargs()
        apply_bounded_delta_vote_step(
            states,
            votes,
            specs,
            replay_ce_mode="telemetry",
            front_c_identity_observer=front_c.append,
        )
    finally:
        learner.emit_object_inventory = original
    assert calls["n"] == 0
    assert front_c
    assert set(front_c[0].keys()) == _legacy_front_c_key_set()
    assert CONSUMED_INPUTS_INVENTORY_KEY not in front_c[0]


def test_dedicated_sink_on_emits_inventory_and_leaves_coexisting_observer_legacy() -> None:
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as learner
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
    )

    calls = {"n": 0}
    original = learner.emit_object_inventory

    def counting_emit(obj, **kwargs):
        calls["n"] += 1
        return original(obj, **kwargs)

    learner.emit_object_inventory = counting_emit
    try:
        front_c: list[dict] = []
        sink: list[dict] = []
        states, votes, specs = _toy_apply_kwargs()
        apply_bounded_delta_vote_step(
            states,
            votes,
            specs,
            replay_ce_mode="telemetry",
            front_c_identity_observer=front_c.append,
            attribution_capture_sink=sink.append,
        )
    finally:
        learner.emit_object_inventory = original
    assert calls["n"] > 0
    assert front_c and sink
    legacy = _legacy_front_c_key_set()
    assert set(front_c[0].keys()) == legacy
    assert CONSUMED_INPUTS_INVENTORY_KEY not in front_c[0]
    assert set(sink[0].keys()) == legacy | {CONSUMED_INPUTS_INVENTORY_KEY}
    assert CONSUMED_INPUTS_INVENTORY_KEY in sink[0]


def test_dedicated_sink_preserves_apply_result_parity() -> None:
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
    )

    states, votes, specs = _toy_apply_kwargs()
    off = apply_bounded_delta_vote_step(states, votes, specs, replay_ce_mode="telemetry")
    front_c: list[dict] = []
    sink: list[dict] = []
    on = apply_bounded_delta_vote_step(
        states,
        votes,
        specs,
        replay_ce_mode="telemetry",
        front_c_identity_observer=front_c.append,
        attribution_capture_sink=sink.append,
    )
    assert off.to_compact_dict() == on.to_compact_dict()
    assert off.global_summary == on.global_summary


def test_borrowed_front_c_identity_observer_is_logging_only_and_cloned(tmp_path) -> None:
    # LICENSE TERM: run the borrowed instrument. Do not wrap. Do not reimplement.
    _front_c_mod.test_front_c_identity_observer_is_logging_only_and_cloned(tmp_path)


def test_control_emission_names_first_failing_and_extremal() -> None:
    indices = [0, 1, 2]
    untouched = {0: False, 1: False, 2: True}
    raw = {
        0: {"q_after": 0, "acc_after": 1},
        1: {"q_after": 1, "acc_after": 9},
        2: {"q_after": 0, "acc_after": 0},
    }
    emitted = control_emission(
        control_indices=indices,
        untouched_by_index=untouched,
        raw_by_index=raw,
        hash_series={
            "q_before_sha256": [0, 0, 0],
            "q_after_sha256": [0, 1, 0],
            "acc_before_sha256": [0, 0, 0],
            "acc_pre_writeback_sha256": [0, 0, 0],
            "acc_after_sha256": [1, 9, 0],
        },
    )
    assert emitted["control_first_failing_row"]["index"] == 0
    assert emitted["control_extremal_failing_row"]["index"] == 1
    assert emitted["per_row_raw_emitted"] == 2
    assert set(emitted["control_hashes"]) >= {
        "untouched_mask_sha256",
        "q_before_sha256",
        "q_after_sha256",
        "acc_before_sha256",
        "acc_pre_writeback_sha256",
        "acc_after_sha256",
    }


def test_wire_fed_observation_has_disjoint_apply_and_mode_fed_plans() -> None:
    """Operands from the actual wire over a venue-true observer observation."""
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
        make_bounded_tensor_state,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _attribution_read_receipt_from_observation,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

    state = make_bounded_tensor_state(
        "toy.weight",
        torch.zeros(8, dtype=torch.int8),
        1.0,
        torch.zeros(8, dtype=torch.int16),
    )
    votes = torch.zeros(8, dtype=torch.int16)
    votes[0] = 2
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )
    captured: list[dict] = []

    def observer(observation):
        captured.append(observation)

    apply_bounded_delta_vote_step(
        {"toy.weight": state},
        {"toy.weight": votes},
        {"toy.weight": spec},
        replay_ce_mode="telemetry",
        attribution_capture_sink=observer,
    )
    assert captured, "attribution sink did not produce a venue-true observation"
    receipt = _attribution_read_receipt_from_observation(
        captured[0],
        two_tier_carry_w6_enabled=False,
    )
    assert receipt["schema"] == "hrm_text_158_attribution_emission/v0"
    row = receipt["per_key"]["toy.weight"]
    assert "branch" not in row
    assert "reason" not in row
    assert "classify" not in row
    live = row["live_inputs"]
    assert "dataclasses_fields" in live
    assert "vars_keys" in live
    assert "replay_ce_mode" not in live["dataclasses_fields"]
    assert "_replay_ce_mode" in live["vars_keys"]
    consumed = row["consumed_inputs"]
    assert row["consumed_inputs_present"] is True
    assert consumed is not None
    assert "dataclasses_fields" in consumed
    assert "vars_keys" in consumed
    assert "replay_ce_mode" not in consumed["dataclasses_fields"]
    assert "_replay_ce_mode" in consumed["vars_keys"]
    assert "vote_active_flat_indices" in consumed["dataclasses_fields"]
    assert "sparse_vote_events" in consumed["dataclasses_fields"]
    assert "vote_active_flat_indices" in consumed["vars_keys"]
    assert "sparse_vote_events" in consumed["vars_keys"]
    assert "vote_active_flat_indices" in live["dataclasses_fields"]
    assert "sparse_vote_events" in live["dataclasses_fields"]
    assert "vote_active_flat_indices" in live["vars_keys"]
    assert "sparse_vote_events" in live["vars_keys"]
    from calm.hrm_text_158.native_full_stack.vote_update import (
        plan_integer_vote_update_reference,
    )

    apply_prov = row["apply_plan"]["planner_provenance"]
    assert apply_prov["planner_function"] == plan_integer_vote_update_reference.__name__
    assert apply_prov["planner_module"] == plan_integer_vote_update_reference.__module__
    for replan_name in ("modeless_veto", "mode_fed"):
        inv = row["replans"][replan_name]["inputs_inventory"]
        assert "replay_ce_mode" not in inv["dataclasses_fields"]
        assert "_replay_ce_mode" in inv["vars_keys"]
        assert "vote_active_flat_indices" in inv["dataclasses_fields"]
        assert "sparse_vote_events" in inv["dataclasses_fields"]
        replan_prov = row["replans"][replan_name]["planner_provenance"]
        assert replan_prov["planner_function"] == plan_integer_vote_update_reference.__name__
        assert replan_prov["planner_module"] == plan_integer_vote_update_reference.__module__
    ids = row["plan_ids"]
    identity_pairs = (
        ("apply", "mode_fed", ids["apply_plan_id"], ids["mode_fed_plan_id"]),
        ("apply", "listed", ids["apply_plan_id"], ids["listed_plan_id"]),
        ("listed", "mode_fed", ids["listed_plan_id"], ids["mode_fed_plan_id"]),
    )
    assert len(identity_pairs) == 3
    for left_name, right_name, left_id, right_id in identity_pairs:
        assert left_id != right_id, f"{left_name}_plan_id == {right_name}_plan_id"
    assert "listed" in row["input_hashes"]
    assert "mode_fed" in row["input_hashes"]
    assert "q_sha256" in row["input_hashes"]["listed"]
    assert "control" in row
    assert row["control"]["control_examined_count"] == row["control"]["control_population_size"]


def test_D_apply_veto_count_not_0_fires_at_reproduced_cardinality() -> None:
    listed = _listed()
    result = classify(
        listed=listed,
        apply_applied=listed,
        apply_veto_count=1,
        mode_fed_veto=(),
        mode_fed_applied=listed,
        types=_types(APPLIED, listed),
    )
    assert result["branch"] == BRANCH_D
    assert result["reason"] == "D_apply_veto_count_not_0"


def test_D_both_fires_when_cardinality_and_apply_veto_differ() -> None:
    result = classify(
        listed=(1, 2, 3),
        apply_applied=(1, 2, 3),
        apply_veto_count=7,
        mode_fed_veto=(),
        mode_fed_applied=(),
        types={1: APPLIED, 2: APPLIED, 3: APPLIED},
    )
    assert result["branch"] == BRANCH_D
    assert result["reason"] == "D_both"
    assert result["sub_reasons"] == [
        "D_listed_cardinality_not_949",
        "D_apply_veto_count_not_0",
    ]


def test_borrowed_hasher_distinguishes_different_vote_tensors() -> None:
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import tensor_sha256

    a = torch.zeros(8, dtype=torch.int16)
    b = torch.zeros(8, dtype=torch.int16)
    b[0] = 2
    assert tensor_sha256(a) != tensor_sha256(b)


def _scale_floor_observation(*, n: int, fail_control: bool) -> dict:
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
        make_bounded_tensor_state,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

    state = make_bounded_tensor_state(
        "toy.weight",
        torch.zeros(n, dtype=torch.int8),
        1.0,
        torch.zeros(n, dtype=torch.int16),
    )
    votes = torch.zeros(n, dtype=torch.int16)
    votes[:8] = 2
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=8,
    )
    captured: list[dict] = []

    def observer(observation):
        captured.append(observation)

    apply_bounded_delta_vote_step(
        {"toy.weight": state},
        {"toy.weight": votes},
        {"toy.weight": spec},
        replay_ce_mode="telemetry",
        attribution_capture_sink=observer,
    )
    assert captured, "scale-floor sink did not produce an observation"
    observation = captured[0]
    if fail_control:
        observation = dict(observation)
        q_acc_by_key = dict(observation["q_acc_by_key"])
        q_acc = dict(q_acc_by_key["toy.weight"])
        q_levels = q_acc["q_levels"].clone()
        q_levels[100] = 1
        q_acc["q_levels"] = q_levels
        q_acc_by_key["toy.weight"] = q_acc
        observation["q_acc_by_key"] = q_acc_by_key
    return observation


def _assert_scale_receipt_emitted(receipt: dict, *, expect_failing: bool) -> None:
    import json

    assert receipt["schema"] == "hrm_text_158_attribution_emission/v0"
    assert "branch" not in receipt
    payload = json.dumps(receipt)
    assert payload
    row = receipt["per_key"]["toy.weight"]
    assert row["consumed_inputs_present"] is True
    assert row["consumed_inputs"] is not None
    assert "dataclasses_fields" in row["consumed_inputs"]
    assert "vars_keys" in row["consumed_inputs"]
    assert "dataclasses_fields" in row["live_inputs"]
    assert "vars_keys" in row["live_inputs"]
    assert "modeless_veto" in row["replans"]
    assert "mode_fed" in row["replans"]
    hashes = row["control"]["control_hashes"]
    required = {
        "untouched_mask_sha256",
        "q_before_sha256",
        "q_after_sha256",
        "acc_before_sha256",
        "acc_pre_writeback_sha256",
        "acc_after_sha256",
    }
    assert set(hashes) >= required
    for name in required:
        assert isinstance(hashes[name], str)
        assert len(hashes[name]) == 64
    if expect_failing:
        assert row["control"]["control_failing_count"] >= 1
        assert row["control"]["control_first_failing_row"] is not None
    else:
        assert row["control"]["control_failing_count"] == 0


def test_control_scale_vectorized_at_measured_floor() -> None:
    import time
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _attribution_read_receipt_from_observation,
    )

    n = 682450
    started = time.perf_counter()
    clean_obs = _scale_floor_observation(n=n, fail_control=False)
    clean_receipt = _attribution_read_receipt_from_observation(
        clean_obs,
        two_tier_carry_w6_enabled=False,
    )
    fail_obs = _scale_floor_observation(n=n, fail_control=True)
    fail_receipt = _attribution_read_receipt_from_observation(
        fail_obs,
        two_tier_carry_w6_enabled=False,
    )
    elapsed = time.perf_counter() - started
    _assert_scale_receipt_emitted(clean_receipt, expect_failing=False)
    _assert_scale_receipt_emitted(fail_receipt, expect_failing=True)
    assert elapsed < 60.0


def test_argparse_exposes_emit_attribution_read() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser

    parser = build_arg_parser()
    off = parser.parse_args([])
    on = parser.parse_args(["--emit-attribution-read"])
    assert off.emit_attribution_read is False
    assert on.emit_attribution_read is True


def test_attribution_sidecar_survives_tolerated_attach_mismatch(tmp_path) -> None:
    import json
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _assert_tier_a_index_surface_count_consistency,
        attribution_read_sidecar_path,
        persist_attribution_read_sidecar,
    )

    record = {
        "schema": "hrm_text_158_attribution_emission/v0",
        "per_key": {"toy.weight": {"consumed_inputs_present": True}},
    }
    path = persist_attribution_read_sidecar(
        scratch_root=tmp_path,
        step=1,
        record=record,
    )
    _assert_tier_a_index_surface_count_consistency(
        "toy.weight",
        tensor_stats={"replay_ce_veto_count": 0},
        applied_indices=(),
    )
    assert path.is_file()
    assert path == attribution_read_sidecar_path(tmp_path, 1)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema"] == "hrm_text_158_attribution_emission/v0"
    assert "branch" not in parsed
    assert "reason" not in parsed


def test_attribution_sidecar_clean_world_and_per_step_keys(tmp_path) -> None:
    import json
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        persist_attribution_read_sidecar,
    )

    first = persist_attribution_read_sidecar(
        scratch_root=tmp_path,
        step=1,
        record={"schema": "hrm_text_158_attribution_emission/v0", "per_key": {"a": {}}},
    )
    second = persist_attribution_read_sidecar(
        scratch_root=tmp_path,
        step=2,
        record={"schema": "hrm_text_158_attribution_emission/v0", "per_key": {"b": {}}},
    )
    assert first.is_file()
    assert second.is_file()
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["per_key"] == {"a": {}}
    assert json.loads(second.read_text(encoding="utf-8"))["per_key"] == {"b": {}}


def test_abort_site_sidecar_survives_tolerated_mismatch(tmp_path) -> None:
    import json
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _assert_tier_a_index_surface_count_consistency,
        emit_control_arm_abort_site_row,
        persist_attribution_abort_site_sidecar,
        plan_vote_update_for_emit,
    )

    indices = list(range(949))
    compact_stats = {"replay_ce_veto_count": 0}
    row = emit_control_arm_abort_site_row(
        replay_ce_veto_indices=indices,
        compact_stats=compact_stats,
    )
    path = persist_attribution_abort_site_sidecar(
        scratch_root=tmp_path,
        step=1,
        record={
            "schema": "hrm_text_158_attribution_abort_site/v0",
            "per_key": {"toy.weight": row},
        },
    )
    _assert_tier_a_index_surface_count_consistency(
        "toy.weight",
        tensor_stats=compact_stats,
        applied_indices=(),
    )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema"] == "hrm_text_158_attribution_abort_site/v0"
    assert "branch" not in parsed
    toy = parsed["per_key"]["toy.weight"]
    assert toy["replay_ce_veto_indices"] == indices
    assert toy["compact_replay_ce_veto_count"] == 0
    assert toy["compact_replay_ce_veto_count_present"] is True
    assert toy["planner_provenance"]["control_arm_planner"] == plan_vote_update_for_emit.__name__
    assert toy["planner_provenance"]["control_arm_planner_module"] == plan_vote_update_for_emit.__module__


def test_attach_control_arm_production_path_writes_abort_site_sidecar(tmp_path) -> None:
    import json
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        make_bounded_tensor_state,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import (
        LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        VoteUpdatePlan,
        VoteUpdateSpec,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _attach_control_arm_index_surfaces_to_compact,
        _plan_integer_vote_update_for_control_arm_surfaces,
        attribution_abort_site_sidecar_path,
        plan_vote_update_for_emit,
    )

    state = make_bounded_tensor_state(
        "toy.weight",
        torch.zeros(8, dtype=torch.int8),
        1.0,
        torch.zeros(8, dtype=torch.int16),
    )
    votes = torch.zeros(8, dtype=torch.int16)
    votes[0] = 2
    spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )
    tensor_states = {"toy.weight": state}
    votes_by_key = {"toy.weight": votes}
    vote_specs = {"toy.weight": spec}
    wrapper_kwargs = dict(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs,
        replay_ce_veto_votes_by_key=None,
        replay_ce_veto_moves_by_key=None,
        pc_aux_votes_by_key=None,
        pc_aux_moves_by_key=None,
        pc_aux_mode="telemetry",
        replay_ce_mode="telemetry",
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        local_selection_ordering_seed=0,
        local_selection_ordering_step=1,
    )
    wrapper_plans = _plan_integer_vote_update_for_control_arm_surfaces(
        replay_ce_mode=wrapper_kwargs["replay_ce_mode"],
        **{k: v for k, v in wrapper_kwargs.items() if k != "replay_ce_mode"},
    )
    real_plan = wrapper_plans["toy.weight"]
    assert isinstance(real_plan, VoteUpdatePlan)
    expected_indices = [
        int(v) for v in real_plan.replay_ce_veto_indices.detach().cpu().tolist()
    ]
    mismatched_count = len(expected_indices) + 3
    compact = {
        "tensor_stats": {"toy.weight": {"replay_ce_veto_count": mismatched_count}},
        "global_summary": {},
    }
    attach_kwargs = dict(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs,
        replay_ce_veto_votes_by_key=None,
        replay_ce_veto_moves_by_key=None,
        pc_aux_votes_by_key=None,
        pc_aux_moves_by_key=None,
        pc_aux_mode="telemetry",
        replay_ce_mode="telemetry",
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        local_selection_ordering_seed=0,
        local_selection_ordering_step=1,
    )
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    _attach_control_arm_index_surfaces_to_compact(
        compact,
        replay_ce_mode=attach_kwargs["replay_ce_mode"],
        **{k: v for k, v in attach_kwargs.items() if k != "replay_ce_mode"},
        attribution_sidecar_dir=None,
        attribution_step=None,
    )
    assert list(off_dir.iterdir()) == []
    assert not attribution_abort_site_sidecar_path(tmp_path, 1).is_file()

    _attach_control_arm_index_surfaces_to_compact(
        compact,
        replay_ce_mode=attach_kwargs["replay_ce_mode"],
        **{k: v for k, v in attach_kwargs.items() if k != "replay_ce_mode"},
        attribution_sidecar_dir=tmp_path,
        attribution_step=1,
    )
    path = attribution_abort_site_sidecar_path(tmp_path, 1)
    assert path.is_file()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["schema"] == "hrm_text_158_attribution_abort_site/v0"
    assert "branch" not in parsed
    toy = parsed["per_key"]["toy.weight"]
    assert toy["replay_ce_veto_indices"] == expected_indices
    assert toy["compact_replay_ce_veto_count"] == mismatched_count
    assert toy["planner_provenance"]["control_arm_planner"] == plan_vote_update_for_emit.__name__
    assert toy["planner_provenance"]["control_arm_planner_module"] == plan_vote_update_for_emit.__module__


SUITE_TEST_NAMES = _this_module_test_names()
