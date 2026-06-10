from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    ACC_WIDTH_RECORDED_ROW_SWEEP_SCHEMA_VERSION,
    LABEL_ACC_NOT_SHRINKABLE,
    LABEL_ACC_SHRINK_AGGRESSIVE,
    LABEL_ACC_SHRINK_PARTIAL,
    LABEL_ACC_SHRINK_TWO_TIER,
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    REQUIRED_TRACE_ROW_FIELDS,
    SCOPE_STATEMENT,
    VoteSpecParsed,
    assert_clip_bound_proof,
    assert_observed_clip_matches_declared,
    assert_trace_family_source_clip_is_pm127,
    build_acc_width_recorded_row_sweep,
    compose_vote_spec_from_production_sources,
    derive_threshold_abs_from_recorded_rows,
    resolve_vote_spec,
    build_required_field_inventory,
    classify_w_min_label,
    coarse_invariant_vs_reference,
    compute_w_min_headroom_safe,
    compute_w_min_invariant,
    decay_vote_clamp,
    effective_clip_bounds,
    headroom_passes,
    parse_vote_spec_from_capture_receipt,
    replay_width_lane,
    signed_w_max,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
    _stable_hash16,
)


def _vote_spec(
    *,
    clip_min: int = -127,
    clip_max: int = 127,
    threshold_abs: int = 20,
) -> VoteSpecParsed:
    return VoteSpecParsed(
        threshold_abs=threshold_abs,
        decay_numerator=1,
        decay_denominator=1,
        accumulator_clip_min=clip_min,
        accumulator_clip_max=clip_max,
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
    threshold_abs: int = 20,
) -> dict[str, object]:
    proposal_direction = 1 if int(new_acc) >= 0 else -1
    threshold = int(threshold_abs)
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
    return {
        "optimizer_step_index": step_index,
        "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        "source_table_hash": _stable_hash16(canonical),
        "sampled_candidate_table": rows,
        "post_update_telemetry": {"q_changed_count": q_changed_count},
    }


def _write_capture_receipt(
    path: Path,
    *,
    clip_min: int = -127,
    clip_max: int = 127,
    threshold_abs: int = 20,
) -> None:
    path.write_text(
        json.dumps(
            {
                "vote_update_spec": {
                    "threshold_abs": threshold_abs,
                    "accumulator_clip_min": clip_min,
                    "accumulator_clip_max": clip_max,
                    "decay_numerator": 1,
                    "decay_denominator": 1,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_trace(path: Path, steps: list[dict[str, object]]) -> None:
    lines = [json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True)]
    lines.extend(json.dumps(step, sort_keys=True) for step in steps)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_production_capture_receipt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "aux_vote_law": "fixed_rank_bucket_non_target_aux",
                "global_cap_contract": {"enabled": False, "name": "off"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _fixture_steps_bit_identical() -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for step_index in range(1, 4):
        oracle_id = f"oracle-{step_index}"
        rows = [
            _row(
                oracle_id,
                flat_index=1,
                pre_acc=10,
                vote=15,
                new_acc=25,
                local_loss_delta=-0.9,
            ),
            _row(
                f"decoy-{step_index}",
                flat_index=2,
                pre_acc=3,
                vote=1,
                new_acc=4,
                local_loss_delta=-0.1,
            ),
        ]
        steps.append(_step(step_index, rows))
    return steps


def test_field_inventory_gate_requires_each_trace_field_once() -> None:
    steps = _fixture_steps_bit_identical()
    inventory = build_required_field_inventory(steps)
    assert inventory["passed"] is True
    assert inventory["missing_fields"] == []
    for field in REQUIRED_TRACE_ROW_FIELDS:
        assert field in inventory["present_fields"]


def test_clip_bound_parse_and_assert_rejects_mismatched_fixture() -> None:
    vote_spec = _vote_spec(clip_min=-64, clip_max=64)
    assertion = assert_observed_clip_matches_declared(
        vote_spec,
        observed_clip_min=-127,
        observed_clip_max=127,
    )
    assert assertion["passed"] is False


def test_trace_family_source_clip_assertion_rejects_non_pm127_declared() -> None:
    vote_spec = _vote_spec(clip_min=-64, clip_max=64)
    assertion = assert_trace_family_source_clip_is_pm127(vote_spec)
    assert assertion["passed"] is False
    assert assertion["declared_passed"] is False


def test_clip_bound_proof_rejects_row_extrema_outside_declared_clip() -> None:
    vote_spec = _vote_spec()
    assertion = assert_clip_bound_proof(vote_spec, recorded_row_bounds=(3, 130))
    assert assertion["declared_passed"] is True
    assert assertion["row_extrema_within_declared"] is False
    assert assertion["passed"] is False


def test_non_pm127_capture_fails_prereg_semantics(tmp_path: Path) -> None:
    steps = _fixture_steps_bit_identical()
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    _write_trace(trace, steps)
    _write_capture_receipt(capture, clip_min=-64, clip_max=64)
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        width_grid=(16, 8),
    )
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "source_clip_not_pm127" in receipt["failure_reasons"]
    assert (
        receipt["source_semantics_prereg"][
            "global_clip_pm127_implies_w_ge_8_tautology_expected"
        ]
        is False
    )


def test_parse_vote_spec_reads_capture_receipt_clip_bounds() -> None:
    payload = {
        "vote_update_spec": {
            "threshold_abs": 20,
            "accumulator_clip_min": -127,
            "accumulator_clip_max": 127,
            "decay_numerator": 1,
            "decay_denominator": 1,
        }
    }
    parsed = parse_vote_spec_from_capture_receipt(payload)
    assert parsed.accumulator_clip_min == -127
    assert parsed.accumulator_clip_max == 127


def test_effective_clip_preserves_source_clip_for_w16_and_w8() -> None:
    assert effective_clip_bounds(16, -127, 127) == (-127, 127)
    assert effective_clip_bounds(8, -127, 127) == (-127, 127)


def test_effective_clip_tightens_below_w8() -> None:
    assert effective_clip_bounds(6, -127, 127) == (-31, 31)
    assert effective_clip_bounds(4, -127, 127) == (-7, 7)


def test_w16_lane_bit_identical_to_recorded_new_acc() -> None:
    steps = _fixture_steps_bit_identical()
    vote_spec = _vote_spec()
    applied = {int(step["optimizer_step_index"]): f"oracle-{step['optimizer_step_index']}" for step in steps}
    lane = replay_width_lane(
        steps,
        vote_spec=vote_spec,
        width=16,
        applied_candidate_ids_by_step=applied,
    )
    assert lane["bit_identical_to_recorded_new_acc"] is True
    assert lane["w16_mismatch_rows"] == []


def test_w8_lane_reference_identical_under_effective_clip() -> None:
    steps = _fixture_steps_bit_identical()
    vote_spec = _vote_spec()
    applied = {int(step["optimizer_step_index"]): f"oracle-{step['optimizer_step_index']}" for step in steps}
    lane16 = replay_width_lane(
        steps,
        vote_spec=vote_spec,
        width=16,
        applied_candidate_ids_by_step=applied,
    )
    lane8 = replay_width_lane(
        steps,
        vote_spec=vote_spec,
        width=8,
        applied_candidate_ids_by_step=applied,
    )
    invariance = coarse_invariant_vs_reference(lane8, reference_lane=lane16)
    assert invariance["coarse_crossing_invariant"] is True


def test_w4_lane_tightens_accumulator_relative_to_w16() -> None:
    vote_spec = _vote_spec(threshold_abs=1)
    rows = [
        _row(
            "tight",
            flat_index=1,
            pre_acc=100,
            vote=10,
            new_acc=110,
            q_level=0,
        )
    ]
    steps = [_step(1, rows, q_changed_count=0)]
    lane16 = replay_width_lane(steps, vote_spec=vote_spec, width=16, applied_candidate_ids_by_step={})
    lane4 = replay_width_lane(steps, vote_spec=vote_spec, width=4, applied_candidate_ids_by_step={})
    key = (1, 1)
    assert lane16["row_recomputed_new_acc"][key] == 110
    assert lane4["row_recomputed_new_acc"][key] == 7


def test_headroom_rule_arithmetic() -> None:
    assert headroom_passes(8, max_abs_acc_applied=63, headroom_factor=2.0) is True
    assert headroom_passes(8, max_abs_acc_applied=64, headroom_factor=2.0) is False
    assert signed_w_max(8) == 127


def test_classifier_disjointness_each_label_reachable() -> None:
    cases = [
        (4, True, LABEL_ACC_SHRINK_AGGRESSIVE),
        (8, True, LABEL_ACC_SHRINK_TWO_TIER),
        (12, True, LABEL_ACC_SHRINK_PARTIAL),
        (16, False, LABEL_ACC_NOT_SHRINKABLE),
    ]
    labels = set()
    for w_min, headroom, expected in cases:
        branch = classify_w_min_label(w_min, harness_failures=[], headroom_pass=headroom)
        assert branch["primary_label"] == expected
        labels.add(branch["primary_label"])
    assert labels == {
        LABEL_ACC_SHRINK_AGGRESSIVE,
        LABEL_ACC_SHRINK_TWO_TIER,
        LABEL_ACC_SHRINK_PARTIAL,
        LABEL_ACC_NOT_SHRINKABLE,
    }


def test_w16_non_bit_identical_triggers_harness_fail_label(
    tmp_path: Path,
) -> None:
    bad_rows = [
        _row("bad", flat_index=1, pre_acc=1, vote=1, new_acc=999, local_loss_delta=-1.0)
    ]
    steps = [_step(1, bad_rows)]
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    _write_trace(trace, steps)
    _write_capture_receipt(capture)
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        width_grid=(16, 8),
    )
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "w16_not_bit_identical_to_reference" in receipt["failure_reasons"]


def test_receipt_schema_completeness(tmp_path: Path) -> None:
    steps = _fixture_steps_bit_identical()
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    _write_trace(trace, steps)
    _write_capture_receipt(capture)
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        width_grid=(16, 8, 6, 4),
    )
    assert receipt["schema_version"] == ACC_WIDTH_RECORDED_ROW_SWEEP_SCHEMA_VERSION
    assert receipt["scope_statement"] == SCOPE_STATEMENT
    assert "source_semantics_prereg" in receipt
    assert receipt["source_semantics_prereg"]["not_measured_discovery"] is True
    assert receipt["vote_spec"] is not None
    assert receipt["width_results"]
    assert receipt["primary_label"] in {
        LABEL_ACC_SHRINK_AGGRESSIVE,
        LABEL_ACC_SHRINK_TWO_TIER,
        LABEL_ACC_SHRINK_PARTIAL,
        LABEL_ACC_NOT_SHRINKABLE,
        LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    }


def test_decay_vote_clamp_matches_recorded_values_for_fixture_rows() -> None:
    vote_spec = _vote_spec()
    for pre_acc, vote, expected in ((10, 15, 25), (3, 1, 4)):
        assert (
            decay_vote_clamp(
                pre_acc,
                vote,
                clip_min=vote_spec.accumulator_clip_min,
                clip_max=vote_spec.accumulator_clip_max,
                decay_numerator=1,
                decay_denominator=1,
            )
            == expected
        )


def test_parse_vote_spec_missing_raises() -> None:
    with pytest.raises(ValueError, match="vote_update_spec not found"):
        parse_vote_spec_from_capture_receipt({})


def test_classifier_harness_fail_on_disjoint_range_violation() -> None:
    branch = classify_w_min_label(5, harness_failures=[], headroom_pass=True)
    assert branch["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL


def test_w_min_headroom_safe_skips_invariant_low_width_without_headroom() -> None:
    reference_lane = {
        "bit_identical_to_recorded_new_acc": True,
        "row_crossings": {(1, 1): False},
        "max_abs_acc_applied_flips": 10,
    }
    lane_by_width = {
        16: reference_lane,
        2: {
            "bit_identical_to_recorded_new_acc": True,
            "row_crossings": {(1, 1): False},
            "max_abs_acc_applied_flips": 10,
        },
        8: {
            "bit_identical_to_recorded_new_acc": True,
            "row_crossings": {(1, 1): False},
            "max_abs_acc_applied_flips": 10,
        },
    }
    assert (
        compute_w_min_invariant((16, 8, 2), lane_by_width=lane_by_width, reference_width=16)
        == 2
    )
    assert (
        compute_w_min_headroom_safe(
            (16, 8, 2),
            lane_by_width=lane_by_width,
            reference_width=16,
            headroom_factor=2.0,
        )
        == 8
    )
    branch = classify_w_min_label(8, harness_failures=[], headroom_pass=True)
    assert branch["primary_label"] == LABEL_ACC_SHRINK_TWO_TIER


def test_derive_threshold_abs_from_recorded_rows_fixture() -> None:
    steps = _fixture_steps_bit_identical()
    threshold, provenance = derive_threshold_abs_from_recorded_rows(steps)
    assert threshold == 20
    assert provenance["threshold_source"] == "recorded_row_residual_proximity_relation"


def test_compose_vote_spec_from_production_capture_shape() -> None:
    steps = _fixture_steps_bit_identical()
    manifest = {"parameters": {"max_abs_per_tensor": 32}}
    vote_spec, provenance, failures = compose_vote_spec_from_production_sources(
        steps,
        manifest_payload=manifest,
    )
    assert failures == []
    assert vote_spec is not None
    assert vote_spec.threshold_abs == 20
    assert vote_spec.accumulator_clip_min == -127
    assert vote_spec.accumulator_clip_max == 127
    assert provenance["clip_source"] == "vote_update_source_at_pinned_head"
    assert provenance["max_abs_per_tensor"] == 32


def test_resolve_vote_spec_uses_composed_path_without_spec_block(
    tmp_path: Path,
) -> None:
    steps = _fixture_steps_bit_identical()
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace, steps)
    capture.write_text(
        json.dumps(
            {
                "aux_vote_law": "fixed_rank_bucket_non_target_aux",
                "global_cap_contract": {"enabled": False, "name": "off"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"parameters": {"max_abs_per_tensor": 32}}, sort_keys=True),
        encoding="utf-8",
    )
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        chain_manifest_path=manifest,
        width_grid=(16, 8),
    )
    assert receipt["vote_spec_provenance"]["parse_path"] == "composed_production_fallback"
    assert receipt["vote_spec"]["threshold_abs"] == 20
    assert receipt["vote_spec_provenance"]["max_abs_per_tensor"] == 32
    assert receipt["clip_bound_assertion"]["passed"] is True


def test_resolve_vote_spec_prefers_capture_spec_block_when_present() -> None:
    payload = {
        "vote_update_spec": {
            "threshold_abs": 20,
            "accumulator_clip_min": -127,
            "accumulator_clip_max": 127,
            "decay_numerator": 1,
            "decay_denominator": 1,
        }
    }
    steps = _fixture_steps_bit_identical()
    vote_spec, provenance, failures = resolve_vote_spec(payload, steps)
    assert failures == []
    assert vote_spec is not None
    assert provenance["parse_path"] == "capture_receipt_spec_block"


def test_coarse_invariant_uses_crossings_only_not_band_echo() -> None:
    reference_lane = {
        "row_crossings": {(1, 1): True},
        "recorded_band_membership_echo": {(1, 1): True},
    }
    lane = {
        "row_crossings": {(1, 1): True},
        "recorded_band_membership_echo": {(1, 1): False},
    }
    invariance = coarse_invariant_vs_reference(lane, reference_lane=reference_lane)
    assert invariance["coarse_crossing_invariant"] is True
    assert invariance["band_membership_scope"] == "recorded_echo_not_width_selector"


def test_composed_spec_path_succeeds_without_capture_spec_block(
    tmp_path: Path,
) -> None:
    steps = _fixture_steps_bit_identical()
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.json"
    _write_trace(trace, steps)
    _write_production_capture_receipt(capture)
    manifest.write_text(
        json.dumps({"parameters": {"max_abs_per_tensor": 32}}, sort_keys=True),
        encoding="utf-8",
    )
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        chain_manifest_path=manifest,
        width_grid=(16, 8),
    )
    provenance = receipt["vote_spec_provenance"]
    assert provenance["parse_path"] == "composed_production_fallback"
    assert provenance["clip_source"] == "vote_update_source_at_pinned_head"
    assert provenance["max_abs_source"] == "manifest_parameters"
    assert provenance["max_abs_per_tensor"] == 32
    assert provenance["threshold"]["threshold_source"] == (
        "recorded_row_residual_proximity_relation"
    )
    assert receipt["vote_spec"]["threshold_abs"] == 20
    assert receipt["clip_bound_assertion"]["passed"] is True
    assert "capture_receipt_parse_error" not in receipt["failure_reasons"]


def test_composed_spec_row_inconsistency_harness_fails(tmp_path: Path) -> None:
    steps = _fixture_steps_bit_identical()
    first_row = steps[0]["sampled_candidate_table"][0]
    first_row["proximity_to_threshold"] = 999
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    _write_trace(trace, steps)
    _write_production_capture_receipt(capture)
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        width_grid=(16, 8),
    )
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "threshold_derivation_fail" in receipt["failure_reasons"]
    assert receipt["vote_spec"] is None


def test_composed_non_pm127_clip_fails_prereg_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep."
        "vote_update_source_constants_at_pinned_head",
        lambda: {
            "accumulator_clip_min": -64,
            "accumulator_clip_max": 64,
            "decay_numerator": 1,
            "decay_denominator": 1,
        },
    )
    steps = _fixture_steps_bit_identical()
    trace = tmp_path / "trace.jsonl"
    capture = tmp_path / "capture.json"
    b2c = tmp_path / "b2c.json"
    audit = tmp_path / "audit.json"
    _write_trace(trace, steps)
    _write_production_capture_receipt(capture)
    b2c.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    receipt = build_acc_width_recorded_row_sweep(
        stable_trace_path=trace,
        capture_receipt_path=capture,
        b2c_receipt_path=b2c,
        audit_receipt_path=audit,
        width_grid=(16, 8),
    )
    assert receipt["primary_label"] == LABEL_SCREEN_HARNESS_OR_GATE_FAIL
    assert "source_clip_not_pm127" in receipt["failure_reasons"]
    assert (
        receipt["source_semantics_prereg"][
            "global_clip_pm127_implies_w_ge_8_tautology_expected"
        ]
        is False
    )
