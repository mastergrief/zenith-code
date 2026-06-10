from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
    REQUIRED_TRACE_ROW_FIELDS,
    VoteSpecParsed,
    build_required_field_inventory,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
    _stable_hash16,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_falsifier_battery import (
    BATTERY_CONTRACT_ID,
    EXPECTED_TRACE_HASH,
    F1_JACCARD_BAR,
    HELD_STEP_END,
    HELD_STEP_START,
    LABEL_CAP_PRIORITY_REQUIRES_FULL_MAGNITUDE,
    LABEL_CARRY_W6_FALSIFIERS_PASS,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    LABEL_SELECTION_MUST_STAY_TRANSIENT,
    LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD,
    MIN_HELD_QUALIFYING_STEPS,
    W_TEST,
    build_estimand_vacuity_guard,
    build_lane_maps,
    build_two_tier_carry_falsifier_battery,
    build_warmup_subthreshold_applies,
    classify_battery,
    evaluate_f1_step,
    evaluate_f2_step,
    evaluate_f3_step,
    is_held_step,
    jaccard_similarity,
    kendall_tau_b,
    run_falsifier_battery,
    verify_manifest_preflight,
)


def _vote_spec() -> VoteSpecParsed:
    return VoteSpecParsed(
        threshold_abs=CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
        decay_numerator=1,
        decay_denominator=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
    )


def _row(
    candidate_id: str,
    *,
    flat_index: int,
    pre_acc: int,
    vote: int,
    new_acc: int,
    q_level: int = 0,
    in_band: bool = True,
    local_loss_delta: float = -0.5,
) -> dict[str, object]:
    proposal_direction = 1 if int(new_acc) >= 0 else -1
    threshold = CANONICAL_VOTE_UPDATE_THRESHOLD_ABS
    return {
        "candidate_id": candidate_id,
        "flat_index": flat_index,
        "vote_value": vote,
        "pre_accumulator_i16": pre_acc,
        "new_acc_i32_signed": new_acc,
        "proposal_direction": proposal_direction,
        "current_q_level": q_level,
        "in_target_tie_band": in_band,
        "threshold_residual_signed": int(new_acc) - proposal_direction * threshold,
        "proximity_to_threshold": abs(abs(int(new_acc)) - threshold),
        "current_rank_position": flat_index,
        "local_loss_delta": local_loss_delta,
    }


def _step(
    step_index: int,
    rows: list[dict[str, object]],
    *,
    applied_flat_indices: list[int] | None = None,
    q_changed_count: int = 1,
) -> dict[str, object]:
    canonical = sorted(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "current_rank_position": int(row["current_rank_position"]),
                "local_loss_delta": float(row["local_loss_delta"]),
                "pre_accumulator_i16": int(row["pre_accumulator_i16"]),
                "new_acc_i32_signed": int(row["new_acc_i32_signed"]),
                "proximity_to_threshold": int(row["proximity_to_threshold"]),
            }
            for row in rows
        ],
        key=lambda item: str(item["candidate_id"]),
    )
    telemetry: dict[str, object] = {"q_changed_count": q_changed_count}
    if applied_flat_indices is not None:
        telemetry["applied_flip_flat_indices"] = list(applied_flat_indices)
    return {
        "optimizer_step_index": step_index,
        "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        "source_table_hash": _stable_hash16(canonical),
        "sampled_candidate_table": rows,
        "post_update_telemetry": telemetry,
    }


def _held_step_indices(count: int, *, start: int = HELD_STEP_START) -> list[int]:
    return list(range(start, start + count))


def _saturation_rows(flat_start: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(32):
        flat_index = flat_start + offset
        rows.append(
            _row(
                f"cand-{flat_index}",
                flat_index=flat_index,
                pre_acc=0,
                vote=40,
                new_acc=40,
            )
        )
    return rows


def test_f1_jaccard_edge_at_k_equals_two_pass_and_fail() -> None:
    vote_spec = _vote_spec()
    pass_rows = [
        _row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15),
        _row("b", flat_index=2, pre_acc=0, vote=12, new_acc=12),
    ]
    pass_step = _step(30, pass_rows, applied_flat_indices=[1, 2])
    lane_maps = build_lane_maps([pass_step], vote_spec=vote_spec)
    pass_result = evaluate_f1_step(
        pass_step,
        lane_maps=lane_maps,
        applied_candidate_ids_by_step={},
    )
    assert pass_result["qualifying"] is True
    assert pass_result["k"] == 2
    assert pass_result["jaccard"] >= F1_JACCARD_BAR
    assert pass_result["pass"] is True

    fail_rows = [
        _row("a", flat_index=1, pre_acc=0, vote=31, new_acc=31),
        _row("b", flat_index=2, pre_acc=0, vote=12, new_acc=12),
        _row("c", flat_index=3, pre_acc=0, vote=50, new_acc=50),
        _row("d", flat_index=4, pre_acc=0, vote=49, new_acc=49),
    ]
    fail_step = _step(31, fail_rows, applied_flat_indices=[1, 2])
    lane_maps_fail = build_lane_maps([fail_step], vote_spec=vote_spec)
    fail_result = evaluate_f1_step(
        fail_step,
        lane_maps=lane_maps_fail,
        applied_candidate_ids_by_step={},
    )
    assert fail_result["qualifying"] is True
    assert fail_result["k"] == 2
    assert fail_result["jaccard"] < F1_JACCARD_BAR
    assert fail_result["pass"] is False


def test_saturation_tie_fixture_triggers_f2_saturation_vacuous() -> None:
    vote_spec = _vote_spec()
    steps = [
        _step(step_index, _saturation_rows(), applied_flat_indices=[1, 2])
        for step_index in _held_step_indices(HELD_STEP_END - HELD_STEP_START + 1)
    ]
    battery = run_falsifier_battery(steps, vote_spec=vote_spec)
    f2 = battery["f2_rank_tau_b"]
    guard = battery["estimand_vacuity_guard"]
    assert f2["held_qualifying_steps"] == 0
    assert f2["disqualified_held_fraction"] > 0.80
    assert f2["vacuity_triggered"] is True
    assert guard["f2_saturation_vacuous"] is True
    assert battery["classifier"]["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert battery["classifier"]["matched_row"] == 1


def test_f3_lowest_flat_index_tiebreak_is_deterministic() -> None:
    vote_spec = _vote_spec()
    rows = [
        _row("a", flat_index=5, pre_acc=0, vote=50, new_acc=50),
        _row("b", flat_index=2, pre_acc=0, vote=50, new_acc=50),
        _row("c", flat_index=8, pre_acc=0, vote=12, new_acc=12),
    ]
    step = _step(30, rows)
    lane_maps = build_lane_maps([step], vote_spec=vote_spec)
    result = evaluate_f3_step(step, lane_maps=lane_maps)
    assert result["qualifying"] is True
    assert result["a_ref"] == 2
    assert result["a_test"] == 2
    assert result["pass"] is True


def test_classifier_precedence_maps_four_combos_and_screens() -> None:
    combos = [
        (False, True, True, LABEL_CAP_PRIORITY_REQUIRES_FULL_MAGNITUDE, 2),
        (True, False, True, LABEL_SELECTION_MUST_STAY_TRANSIENT, 3),
        (False, False, True, LABEL_SELECTION_MUST_STAY_TRANSIENT_BROAD, 4),
        (True, True, True, LABEL_CARRY_W6_FALSIFIERS_PASS, 5),
    ]
    for f1_pass, f2_pass, f3_pass, label, row in combos:
        result = classify_battery(
            f1_pass=f1_pass,
            f2_pass=f2_pass,
            f3_pass=f3_pass,
            vacuity_guard={},
            harness_failures=[],
        )
        assert result["primary_label"] == label
        assert result["matched_row"] == row

    screen = classify_battery(
        f1_pass=True,
        f2_pass=True,
        f3_pass=True,
        vacuity_guard={"f2_saturation_vacuous": True},
        harness_failures=[],
    )
    assert screen["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert screen["matched_row"] == 1


def _valid_manifest_base() -> dict[str, object]:
    return {
        "phase": "acc_width_sweep_v0",
        "storage_class": "durable_not_tmp",
        "exit_codes": {
            "b2c_replay": 0,
            "audit_v0": 0,
            "determinism_gate": 0,
            "acc_width_sweep_v0": 0,
        },
        "determinism_gate": {"pass": True, "trace_hash": EXPECTED_TRACE_HASH},
        "artifacts": [
            {"role": "stable_copy_00", "path": "/tmp/stable.ndjson"},
            {"role": "b2b_trace", "path": "/tmp/b2b.ndjson"},
            {"role": "capture_receipt", "path": "/tmp/capture.json"},
            {"role": "b2c_receipt", "path": "/tmp/b2c.json"},
            {"role": "audit_receipt", "path": "/tmp/audit.json"},
            {"role": "acc_width_receipt", "path": "/tmp/acc_width.json"},
        ],
    }


def test_manifest_bind_smoke_and_phase_guard_p3_exit_classes() -> None:
    base_manifest = _valid_manifest_base()
    ok = verify_manifest_preflight(base_manifest, fals_root="/tmp/fals")
    assert ok["passed"] is True
    assert ok["bound_paths"]["stable_copy_00"] == "/tmp/stable.ndjson"
    assert ok["bound_paths"]["b2b_trace"] == "/tmp/b2b.ndjson"
    assert ok["trace_hash"] == EXPECTED_TRACE_HASH

    rerun_manifest = {
        **base_manifest,
        "phase": "two_tier_falsifier_battery_v0",
        "exit_codes": {
            **base_manifest["exit_codes"],
            "two_tier_falsifier_battery_v0": 0,
        },
    }
    rerun = verify_manifest_preflight(rerun_manifest, fals_root="/tmp/fals")
    assert rerun["passed"] is True
    assert rerun["prior_own_phase_classification"] == "rerun_over_prior_success"

    launcher_manifest = {
        **rerun_manifest,
        "exit_codes": {
            **rerun_manifest["exit_codes"],
            "two_tier_falsifier_battery_v0": 127,
        },
    }
    launcher = verify_manifest_preflight(launcher_manifest, fals_root="/tmp/fals")
    assert launcher["passed"] is True
    assert launcher["prior_own_phase_classification"] == "launcher_failed_previous_attempt"

    stop_manifest = {
        **rerun_manifest,
        "exit_codes": {
            **rerun_manifest["exit_codes"],
            "two_tier_falsifier_battery_v0": 2,
        },
    }
    stop = verify_manifest_preflight(stop_manifest, fals_root="/tmp/fals")
    assert stop["passed"] is False
    assert "stop_for_review" in stop["failure_reasons"]
    assert stop["prior_own_phase_classification"] == "stop_for_review"


def test_manifest_preflight_fail_closed_on_missing_or_wrong_pins() -> None:
    base = _valid_manifest_base()

    missing_storage = dict(base)
    missing_storage.pop("storage_class")
    storage = verify_manifest_preflight(missing_storage)
    assert storage["passed"] is False
    assert "missing_storage_class" in storage["failure_reasons"]

    missing_det = dict(base)
    missing_det.pop("determinism_gate")
    det = verify_manifest_preflight(missing_det)
    assert det["passed"] is False
    assert "missing_determinism_gate" in det["failure_reasons"]

    wrong_hash = dict(base)
    wrong_hash["determinism_gate"] = {"pass": True, "trace_hash": "deadbeef00000000"}
    trace = verify_manifest_preflight(wrong_hash)
    assert trace["passed"] is False
    assert "trace_hash_mismatch" in trace["failure_reasons"]

    missing_acc_width = dict(base)
    missing_acc_width["artifacts"] = [
        entry
        for entry in base["artifacts"]
        if entry["role"] != "acc_width_receipt"
    ]
    acc = verify_manifest_preflight(missing_acc_width)
    assert acc["passed"] is False
    assert "missing_manifest_role:acc_width_receipt" in acc["failure_reasons"]

    missing_b2b = dict(base)
    missing_b2b["artifacts"] = [
        entry for entry in base["artifacts"] if entry["role"] != "b2b_trace"
    ]
    b2b = verify_manifest_preflight(missing_b2b)
    assert b2b["passed"] is False
    assert "missing_manifest_role:b2b_trace" in b2b["failure_reasons"]


def test_manifest_bind_with_production_role_names() -> None:
    base = _valid_manifest_base()
    prod_manifest = {
        **base,
        "artifacts": [
            {"role": "stable_copy_00", "path": "/tmp/stable.ndjson"},
            {"role": "b2b_trace", "path": "/tmp/b2b.ndjson"},
            {"role": "capture_receipt", "path": "/tmp/capture.json"},
            {"role": "b2c_receipt", "path": "/tmp/b2c.json"},
            {"role": "audit_receipt", "path": "/tmp/audit.json"},
            {"role": "acc_receipt", "path": "/tmp/acc_width.json"},
        ],
    }
    ok = verify_manifest_preflight(prod_manifest, fals_root="/tmp/fals")
    assert ok["passed"] is True
    assert ok["bound_paths"]["acc_width_receipt"] == "/tmp/acc_width.json"


def test_f1_insufficient_qualifying_routes_to_screen_row_one() -> None:
    vote_spec = _vote_spec()
    steps = [
        _step(
            step_index,
            [
                _row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15),
                _row("b", flat_index=2, pre_acc=0, vote=12, new_acc=12),
            ],
            applied_flat_indices=[1],
        )
        for step_index in _held_step_indices(5)
    ]
    battery = run_falsifier_battery(steps, vote_spec=vote_spec)
    assert battery["f1_cap_priority"]["held_qualifying_steps"] < MIN_HELD_QUALIFYING_STEPS
    assert battery["estimand_vacuity_guard"]["f1_insufficient_qualifying"] is True
    assert battery["classifier"]["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL


def test_f3_insufficient_qualifying_routes_to_screen_row_one() -> None:
    vote_spec = _vote_spec()
    steps = [
        _step(
            step_index,
            [_row("a", flat_index=1, pre_acc=0, vote=3, new_acc=3)],
            applied_flat_indices=[1],
        )
        for step_index in _held_step_indices(5)
    ]
    battery = run_falsifier_battery(steps, vote_spec=vote_spec)
    assert battery["f3_tiebreak"]["held_qualifying_steps"] < MIN_HELD_QUALIFYING_STEPS
    assert battery["estimand_vacuity_guard"]["f3_insufficient_qualifying"] is True
    assert battery["classifier"]["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL


def test_a2_boundary_tie_top_k_is_deterministic_under_abs_then_flat_index() -> None:
    vote_spec = _vote_spec()
    rows = [
        _row("a", flat_index=4, pre_acc=0, vote=50, new_acc=50),
        _row("b", flat_index=2, pre_acc=0, vote=50, new_acc=50),
        _row("c", flat_index=6, pre_acc=0, vote=50, new_acc=50),
        _row("d", flat_index=1, pre_acc=0, vote=12, new_acc=12),
    ]
    step = _step(30, rows, applied_flat_indices=[2, 4])
    lane_maps = build_lane_maps([step], vote_spec=vote_spec)
    result = evaluate_f1_step(
        step,
        lane_maps=lane_maps,
        applied_candidate_ids_by_step={},
    )
    assert result["qualifying"] is True
    assert result["k"] == 2
    assert result["o_ref_size"] == 2
    assert result["o_test_size"] == 2
    assert result["jaccard"] == 1.0


def test_f1_trace_policy_mismatch_when_k_exceeds_crossing_count() -> None:
    vote_spec = _vote_spec()
    rows = [_row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15)]
    step = _step(30, rows, applied_flat_indices=[1, 2, 3])
    lane_maps = build_lane_maps([step], vote_spec=vote_spec)
    result = evaluate_f1_step(
        step,
        lane_maps=lane_maps,
        applied_candidate_ids_by_step={},
    )
    assert result["qualifying"] is False
    assert result["skip_reason"] == "trace_policy_mismatch"

    guard = build_estimand_vacuity_guard(
        f1_summary={"held_qualifying_steps": 0},
        f2_summary={"vacuity_triggered": False},
        f3_summary={"held_qualifying_steps": 0},
        f1_trace_policy_mismatch_held=True,
    )
    classified = classify_battery(
        f1_pass=False,
        f2_pass=False,
        f3_pass=False,
        vacuity_guard=guard,
        harness_failures=[],
    )
    assert classified["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert guard["trace_policy_mismatch_held"] is True


def test_field_inventory_gate_lists_acc_width_row_fields(tmp_path: Path) -> None:
    steps = [
        _step(
            1,
            [_row("a", flat_index=1, pre_acc=0, vote=5, new_acc=5)],
            applied_flat_indices=[1],
        )
    ]
    inventory = build_required_field_inventory(steps)
    assert inventory["passed"] is True
    for field in REQUIRED_TRACE_ROW_FIELDS:
        assert field in inventory["present_fields"]

    trace_path = tmp_path / "stable.ndjson"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True),
                json.dumps(steps[0], sort_keys=True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(
        json.dumps(
            {
                "vote_update_spec": {
                    "threshold_abs": CANONICAL_VOTE_UPDATE_THRESHOLD_ABS,
                    "accumulator_clip_min": -127,
                    "accumulator_clip_max": 127,
                    "decay_numerator": 1,
                    "decay_denominator": 1,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    b2b_path = tmp_path / "b2b.ndjson"
    b2b_path.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("b2c.json", "audit.json", "acc_width.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    receipt = build_two_tier_carry_falsifier_battery(
        stable_trace_path=trace_path,
        b2b_trace_path=b2b_path,
        capture_receipt_path=capture_path,
        b2c_receipt_path=tmp_path / "b2c.json",
        audit_receipt_path=tmp_path / "audit.json",
        acc_width_receipt_path=tmp_path / "acc_width.json",
    )
    gate = receipt["field_inventory_gate"]
    assert gate["passed"] is True
    assert gate["required_fields"] == list(REQUIRED_TRACE_ROW_FIELDS)
    assert receipt["contract_id"] == BATTERY_CONTRACT_ID


def test_kendall_tau_b_uses_knight_tie_correction() -> None:
    tau_b, comparable, discordant = kendall_tau_b([1, 1, 2, 3], [1, 2, 2, 3])
    assert comparable > 0
    assert -1.0 <= tau_b <= 1.0
    assert discordant >= 0


def test_jaccard_edge_cases() -> None:
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_non_held_trace_policy_mismatch_not_row_one() -> None:
    """Non-held mismatch steps should not trigger trace_policy_mismatch_held, but should populate warmup block."""
    vote_spec = _vote_spec()
    non_held_step_index = 1
    rows = [_row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15)]
    step = _step(non_held_step_index, rows, applied_flat_indices=[1, 2, 3])
    lane_maps = build_lane_maps([step], vote_spec=vote_spec)
    result = evaluate_f1_step(
        step,
        lane_maps=lane_maps,
        applied_candidate_ids_by_step={},
    )
    assert result["qualifying"] is False
    assert result["skip_reason"] == "trace_policy_mismatch"
    assert not is_held_step(non_held_step_index)
    battery = run_falsifier_battery([step], vote_spec=vote_spec)
    assert battery["trace_policy_mismatch_any_step"] is True
    assert battery["estimand_vacuity_guard"]["trace_policy_mismatch_held"] is False
    assert len(battery["warmup_subthreshold_applies"]) == 1
    warmup = battery["warmup_subthreshold_applies"][0]
    assert warmup["step"] == non_held_step_index
    assert warmup["k"] == 3
    assert warmup["crossing_count"] == 1


def test_held_trace_policy_mismatch_still_row_one() -> None:
    """Held mismatch steps should trigger row 1 via vacuity guard."""
    vote_spec = _vote_spec()
    held_step_index = 30
    rows = [_row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15)]
    step = _step(held_step_index, rows, applied_flat_indices=[1, 2, 3])
    lane_maps = build_lane_maps([step], vote_spec=vote_spec)
    result = evaluate_f1_step(
        step,
        lane_maps=lane_maps,
        applied_candidate_ids_by_step={},
    )
    assert result["qualifying"] is False
    assert result["skip_reason"] == "trace_policy_mismatch"
    assert is_held_step(held_step_index)
    battery = run_falsifier_battery([step], vote_spec=vote_spec)
    assert battery["trace_policy_mismatch_any_step"] is True
    assert battery["estimand_vacuity_guard"]["trace_policy_mismatch_held"] is True
    assert battery["classifier"]["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert battery["classifier"]["matched_row"] == 1
    assert len(battery["warmup_subthreshold_applies"]) == 0


def test_warmup_block_schema() -> None:
    """Warmup block entries carry all required fields."""
    vote_spec = _vote_spec()
    non_held_step_index = 5
    rows = [
        _row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15),
        _row("b", flat_index=2, pre_acc=0, vote=12, new_acc=12),
    ]
    step = _step(non_held_step_index, rows, applied_flat_indices=[1, 2, 3])
    battery = run_falsifier_battery([step], vote_spec=vote_spec)
    assert len(battery["warmup_subthreshold_applies"]) == 1
    entry = battery["warmup_subthreshold_applies"][0]
    assert "step" in entry
    assert "k" in entry
    assert "crossing_count" in entry
    assert "applied" in entry
    assert "recompute_disagreements" in entry
    assert isinstance(entry["applied"], list)
    for applied_item in entry["applied"]:
        assert "flat_index" in applied_item
        assert "in_table" in applied_item
        if applied_item["in_table"]:
            assert "pre_accumulator_i16" in applied_item
            assert "vote_value" in applied_item
            assert "new_acc_w16_recomputed" in applied_item
            assert "new_acc_recorded" in applied_item
            assert "threshold_residual_signed" in applied_item
            assert "proximity_to_threshold" in applied_item


def test_v1_phase_and_rerun_handling() -> None:
    """v1 phase rerun classification routes to v1 exit code."""
    base_manifest = _valid_manifest_base()
    rerun_manifest_v1 = {
        **base_manifest,
        "phase": "two_tier_falsifier_battery_v1",
        "exit_codes": {
            **base_manifest["exit_codes"],
            "two_tier_falsifier_battery_v1": 0,
        },
    }
    rerun = verify_manifest_preflight(rerun_manifest_v1, fals_root="/tmp/fals")
    assert rerun["passed"] is True
    assert rerun["prior_own_phase_classification"] == "rerun_over_prior_success"
    launcher_manifest_v1 = {
        **rerun_manifest_v1,
        "exit_codes": {
            **rerun_manifest_v1["exit_codes"],
            "two_tier_falsifier_battery_v1": 127,
        },
    }
    launcher = verify_manifest_preflight(launcher_manifest_v1, fals_root="/tmp/fals")
    assert launcher["passed"] is True
    assert launcher["prior_own_phase_classification"] == "launcher_failed_previous_attempt"
    stop_manifest_v1 = {
        **rerun_manifest_v1,
        "exit_codes": {
            **rerun_manifest_v1["exit_codes"],
            "two_tier_falsifier_battery_v1": 3,
        },
    }
    stop = verify_manifest_preflight(stop_manifest_v1, fals_root="/tmp/fals")
    assert stop["passed"] is False
    assert "stop_for_review" in stop["failure_reasons"]
    assert stop["prior_own_phase_classification"] == "stop_for_review"
    v0_manifest = {
        **base_manifest,
        "phase": "two_tier_falsifier_battery_v0",
        "exit_codes": {
            **base_manifest["exit_codes"],
            "two_tier_falsifier_battery_v0": 0,
        },
    }
    v0 = verify_manifest_preflight(v0_manifest, fals_root="/tmp/fals")
    assert v0["passed"] is True


def _clean_pass_rows(flat_start: int = 1) -> list[dict[str, object]]:
    """32 rows with distinct new_acc 0..31 — inside W6 clip, so both lanes agree
    exactly: F1 Jaccard 1.0, F2 tau-b 1.0 (496 comparable pairs), F3 argmax agree."""
    return [
        _row(
            f"cand-{flat_start + offset}",
            flat_index=flat_start + offset,
            pre_acc=0,
            vote=offset,
            new_acc=offset,
        )
        for offset in range(32)
    ]


def test_non_held_mismatch_with_passing_held_support_is_not_row_one() -> None:
    """The amendment's critical behavior: a non-held trace-policy mismatch must
    NOT route to row 1 when the held split itself is clean — the battery must
    reach the normal F1/F2/F3 classification (row 5 for this all-pass fixture)."""
    vote_spec = _vote_spec()
    mismatch_step = _step(
        1,
        [_row("a", flat_index=1, pre_acc=0, vote=15, new_acc=15)],
        applied_flat_indices=[1, 2, 3],
    )
    held_steps = [
        _step(step_index, _clean_pass_rows(), applied_flat_indices=[30, 31])
        for step_index in _held_step_indices(HELD_STEP_END - HELD_STEP_START + 1)
    ]
    battery = run_falsifier_battery([mismatch_step] + held_steps, vote_spec=vote_spec)
    guard = battery["estimand_vacuity_guard"]
    assert battery["trace_policy_mismatch_any_step"] is True
    assert guard["trace_policy_mismatch_held"] is False
    assert guard["f1_insufficient_qualifying"] is False
    assert guard["f2_saturation_vacuous"] is False
    assert guard["f3_insufficient_qualifying"] is False
    assert len(battery["warmup_subthreshold_applies"]) == 1
    assert battery["f1_cap_priority"]["held_pass"] is True
    assert battery["f2_rank_tau_b"]["held_pass"] is True
    assert battery["f3_tiebreak"]["held_pass"] is True
    assert battery["classifier"]["matched_row"] != 1
    assert battery["classifier"]["matched_row"] == 5
