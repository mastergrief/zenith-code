"""Compact rollup consumer contract for Phase 2 envelope projector."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
    COMPACT_ROLLUP_KEYS,
    build_carrier_growth_step_record_compact,
    phase2_oracle_required_sidecar_keys,
    project_best_combined_oracle_bpw,
)


def test_compact_rollup_keys_cover_phase2_oracle_inputs() -> None:
    required = phase2_oracle_required_sidecar_keys()
    assert required <= COMPACT_ROLLUP_KEYS


def test_project_best_combined_oracle_bpw_uses_rollup_only() -> None:
    rollup = {
        "module_count": 32,
        "event_count_after": 100,
        "hot_exact_row_count_after": 50,
        "backlog_count_after": 0,
        "new_crossing_events": 0,
        "events_on_q_locked_not_hot": 10,
        "est_events_payload_bytes": 400,
        "est_hot_exact_payload_bytes": 250,
        "est_saved_bytes_v5_clear": 40,
        "est_saved_bytes_v2_coalesce": 20,
        "sidecar_bytes": 512,
    }
    bpw = project_best_combined_oracle_bpw(
        rollup,
        eligible_weight_count=29360128,
        metadata_bytes=768,
        v1_max_hot_reduction_fraction=0.1,
    )
    assert bpw > 0.0


def test_compact_step_record_has_no_modules_array() -> None:
    module_rows = [
        {
            "state_key": "mod00",
            "logical_numel": 128,
            "event_count_after": 4,
            "hot_exact_row_count_after": 2,
            "backlog_count_after": 0,
            "new_crossing_events": 1,
            "demotion_on_crossing_count": 0,
            "demotion_on_decay_count": 0,
            "promotion_count": 0,
            "cap_accepted_rows": 0,
            "q_changed_rows": 0,
            "events_on_q_locked_not_hot": 1,
            "event_dup_lanes_gt1": 0,
            "event_dup_max_per_lane": 1,
            "event_dup_p95_per_lane": 1,
            "hot_carry_abs_bucket_counts": {"0": 0, "1": 1, "2_3": 1, "4_6": 0, "7_9": 0, "10_plus": 0},
            "hot_rows_vote_touched": 1,
            "hot_rows_in_proxy": 1,
            "hot_rows_in_backlog": 0,
            "est_events_payload_bytes": 16,
            "est_hot_exact_payload_bytes": 10,
            "est_saved_bytes_v5_clear": 4,
            "est_saved_bytes_v2_coalesce": 0,
            "active_lane_count": 1,
        }
    ]
    payload = build_carrier_growth_step_record_compact(
        optimizer_step_index=1,
        module_rows=module_rows,
    )
    assert "modules" not in payload
    assert payload["compact"] is True
    assert project_best_combined_oracle_bpw(
        payload["rollup"],
        eligible_weight_count=128,
    ) > 0.0
