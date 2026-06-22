"""CPU fixtures for the R6 pressure-source classifier probe (frozen v3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import pack_w6_lanes_to_bytes
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    build_r3_per_module_payload_rows,
    canonical_r3_packed_payload_content_sha256,
)
from calm.hrm_text_158.native_full_stack.r5_acc_term_measurement_probe import (
    cross_check_sidecar_against_receipt,
)
from calm.hrm_text_158.native_full_stack.r6_pressure_source_classifier_probe import (
    BRANCH_ARTIFACT_INSUFFICIENT,
    BRANCH_HARNESS_FAIL,
    BRANCH_INTRINSIC,
    BRANCH_PERSISTS,
    BRANCH_READ_PATH_FAIL,
    BRANCH_RELIEVED,
    HIGH_PRESSURE_ABS,
    NEXT_ACTION_BY_BRANCH,
    SPARSE_Q_RATIO_MAX,
    build_classifier_from_index,
    index_sidecar_records,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    append_headroom_wiring_sidecar_chunk,
)

BANKED_RUN_ROOT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "r4_1_q_bytepacked_tensorwide_seed43_20260622T164500Z"
)


def _record(step: int, acc: list[int], q: list[int], state_key: str = "mod0") -> dict[str, object]:
    return {
        "schema_version": "hrm_text_158_headroom_wiring_sidecar/v1",
        "step": step,
        "state_key": state_key,
        "accumulator_lanes": [int(v) for v in acc],
        "q_lanes": [int(v) for v in q],
    }


def _index_from_records(records: list[dict[str, object]]) -> dict[str, dict[int, dict[str, object]]]:
    index: dict[str, dict[int, dict[str, object]]] = {}
    for record in records:
        state_key = str(record["state_key"])
        step = int(record["step"])
        index.setdefault(state_key, {})[step] = record
    return index


def _all_lane_q_transition_ratio(records: list[dict[str, object]]) -> float:
    index = _index_from_records(records)
    q_total = 0
    lane_total = 0
    for state_key in index:
        steps = sorted(int(step) for step in index[state_key].keys())
        for prev_step, curr_step in zip(steps[:-1], steps[1:], strict=False):
            prev = index[state_key][prev_step]
            curr = index[state_key][curr_step]
            acc_prev = prev["accumulator_lanes"]
            q_prev = prev["q_lanes"]
            q_curr = curr["q_lanes"]
            assert isinstance(acc_prev, list)
            assert isinstance(q_prev, list)
            assert isinstance(q_curr, list)
            for q_before, q_after in zip(q_prev, q_curr, strict=True):
                lane_total += 1
                if q_before != q_after:
                    q_total += 1
    return float(q_total) / float(lane_total) if lane_total > 0 else 0.0


def _assert_adjacent_pair_surface(receipt: dict[str, object]) -> None:
    metrics = receipt["trajectory_metrics"]
    assert isinstance(metrics, dict)
    assert int(metrics["steps_observed"]) >= 1
    assert int(metrics["modules_observed"]) >= 1
    assert int(metrics["adjacent_pairs_total"]) >= 1
    pairs = metrics["adjacent_pairs"]
    assert isinstance(pairs, list)
    assert len(pairs) == int(metrics["adjacent_pairs_total"])
    pair = pairs[0]
    assert isinstance(pair, dict)
    for field in (
        "q_transition_count",
        "q_transition_fraction",
        "high_pressure_unchanged_q",
        "high_pressure_after_q_change",
        "pressure_mass_delta",
        "pair_content_hash",
    ):
        assert field in pair
    assert isinstance(pair["pair_content_hash"], str)
    assert len(pair["pair_content_hash"]) == 64


def _assert_frozen_strings(receipt: dict[str, object]) -> None:
    claims = receipt["explicit_non_claims"]
    assert isinstance(claims, list)
    assert "no_trainer_or_gpu" in claims
    assert "no_decision_surface_claim" in claims
    assert "no_decision_surface_claim_from_static_probe" not in claims


def _classify(
    records: list[dict[str, object]],
    *,
    harness_fail: bool = False,
    cross_check_pass: bool = True,
    cross_check_required: bool = False,
) -> dict[str, object]:
    index = _index_from_records(records)
    cross_check = {"cross_check_pass": cross_check_pass}
    return build_classifier_from_index(
        index=index,
        cross_check=cross_check,
        harness_fail=harness_fail,
        cross_check_required=cross_check_required,
    )


def test_fixture1_persists_growing_pressure_sparse_q() -> None:
    lanes = 64
    records: list[dict[str, object]] = []
    q = [0] * lanes
    for step, hp_lanes in enumerate((8, 16, 24, 32)):
        acc = [0] * lanes
        for lane in range(hp_lanes):
            acc[lane] = HIGH_PRESSURE_ABS
        records.append(_record(step, acc, q))
    receipt = _classify(records)

    branch = receipt["branch_selection"]
    metrics = receipt["trajectory_metrics"]
    assert branch["branch"] == BRANCH_PERSISTS
    assert branch["next_action"] == NEXT_ACTION_BY_BRANCH[BRANCH_PERSISTS]
    assert int(metrics["pressure_mass_trend_sign"]) >= 0
    assert float(metrics["persist_fraction"]) >= 0.50
    assert float(metrics["q_transition_mass_ratio"]) <= SPARSE_Q_RATIO_MAX
    _assert_adjacent_pair_surface(receipt)
    _assert_frozen_strings(receipt)


def test_fixture2_relieved_decreasing_pressure_mass() -> None:
    lanes = 64
    records: list[dict[str, object]] = []
    q = [1] * lanes
    hp_by_step = (40, 30, 20, 8)
    for step, hp_lanes in enumerate(hp_by_step):
        acc = [0] * lanes
        for lane in range(hp_lanes):
            acc[lane] = HIGH_PRESSURE_ABS + 1
        records.append(_record(step, acc, q))
    receipt = _classify(records)

    branch = receipt["branch_selection"]
    metrics = receipt["trajectory_metrics"]
    assert branch["branch"] == BRANCH_RELIEVED
    assert branch["next_action"] == NEXT_ACTION_BY_BRANCH[BRANCH_RELIEVED]
    assert int(metrics["pressure_mass_trend_sign"]) < 0
    assert float(metrics["relief_fraction"]) >= 0.25
    _assert_adjacent_pair_surface(receipt)


def test_fixture3_intrinsic_pressure_survives_q_flips() -> None:
    lanes = 64
    records: list[dict[str, object]] = []
    hp_lanes = 32
    for step in range(4):
        acc = [0] * lanes
        for lane in range(hp_lanes):
            acc[lane] = HIGH_PRESSURE_ABS + 2
        q = [(step + lane) % 3 for lane in range(lanes)]
        records.append(_record(step, acc, q))
    receipt = _classify(records)

    branch = receipt["branch_selection"]
    metrics = receipt["trajectory_metrics"]
    assert branch["branch"] == BRANCH_INTRINSIC
    assert branch["next_action"] == NEXT_ACTION_BY_BRANCH[BRANCH_INTRINSIC]
    assert int(metrics["intrinsic_pair_count"]) > 0
    assert float(metrics["intrinsic_persistence_fraction_mean"]) >= 0.50
    assert float(metrics["q_transition_mass_ratio"]) > SPARSE_Q_RATIO_MAX
    _assert_adjacent_pair_surface(receipt)


def test_fixture4_artifact_insufficient_single_step() -> None:
    lanes = 16
    acc = [HIGH_PRESSURE_ABS] * lanes
    q = [0] * lanes
    receipt = _classify([_record(0, acc, q)])

    assert receipt["branch_selection"]["branch"] == BRANCH_ARTIFACT_INSUFFICIENT
    assert int(receipt["trajectory_metrics"]["min_steps_per_module"]) < 2


def test_fixture5_harness_fail_malformed_sidecar_missing_q_lanes() -> None:
    import tempfile

    lanes = 16
    acc = [HIGH_PRESSURE_ABS] * lanes
    with tempfile.TemporaryDirectory() as tmp_dir:
        sidecar_path = Path(tmp_dir) / "headroom_wiring_sidecar.jsonl"
        malformed = {
            "schema_version": "hrm_text_158_headroom_wiring_sidecar/v1",
            "step": 0,
            "state_key": "mod0",
            "accumulator_lanes": acc,
        }
        sidecar_path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
        index = index_sidecar_records(sidecar_path)
        receipt = build_classifier_from_index(
            index=index,
            cross_check={"cross_check_pass": True},
            cross_check_required=False,
        )

    assert receipt["branch_selection"]["branch"] == BRANCH_HARNESS_FAIL
    failures = receipt.get("validation_failures", [])
    assert any("missing_field:q_lanes" in str(item) for item in failures)


def test_fixture5b_harness_fail_acc_q_length_mismatch() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        sidecar_path = Path(tmp_dir) / "headroom_wiring_sidecar.jsonl"
        malformed = {
            "schema_version": "hrm_text_158_headroom_wiring_sidecar/v1",
            "step": 0,
            "state_key": "mod0",
            "accumulator_lanes": [HIGH_PRESSURE_ABS] * 8,
            "q_lanes": [0] * 4,
        }
        sidecar_path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
        index = index_sidecar_records(sidecar_path)
        receipt = build_classifier_from_index(
            index=index,
            cross_check={"cross_check_pass": True},
            cross_check_required=False,
        )

    assert receipt["branch_selection"]["branch"] == BRANCH_HARNESS_FAIL
    failures = receipt.get("validation_failures", [])
    assert any("acc_q_length_mismatch" in str(item) for item in failures)


def test_pressure_mass_denominator_blocks_false_persists() -> None:
    """All-lane q ratio sparse, but pressure-mass ratio exceeds SPARSE_Q_RATIO_MAX."""
    lanes = 64
    records: list[dict[str, object]] = []
    hp_by_step = (8, 16, 24, 32)
    for step, hp_lanes in enumerate(hp_by_step):
        acc = [0] * lanes
        for lane in range(hp_lanes):
            acc[lane] = HIGH_PRESSURE_ABS
        q = [0] * lanes
        for cold_lane in range(3):
            q[hp_lanes + cold_lane] = step % 2
        records.append(_record(step, acc, q))

    all_lane_ratio = _all_lane_q_transition_ratio(records)
    assert all_lane_ratio <= SPARSE_Q_RATIO_MAX

    receipt = _classify(records)
    metrics = receipt["trajectory_metrics"]
    assert float(metrics["q_transition_mass_ratio"]) > SPARSE_Q_RATIO_MAX
    assert int(metrics["pressure_mass_trend_sign"]) >= 0
    assert float(metrics["persist_fraction"]) >= 0.50
    assert receipt["branch_selection"]["branch"] != BRANCH_PERSISTS


def test_source_step_high_pressure_unchanged_q_definition() -> None:
    lanes = 8
    acc0 = [HIGH_PRESSURE_ABS, HIGH_PRESSURE_ABS, 0, 0, 0, 0, 0, 0]
    q0 = [0, 1, 0, 0, 0, 0, 0, 0]
    acc1 = [HIGH_PRESSURE_ABS, HIGH_PRESSURE_ABS, 0, 0, 0, 0, 0, 0]
    q1 = [0, 0, 0, 0, 0, 0, 0, 0]
    records = [_record(0, acc0, q0), _record(1, acc1, q1)]
    receipt = _classify(records)
    pair = receipt["trajectory_metrics"]["adjacent_pairs"][0]
    assert pair["high_pressure_unchanged_q"] == 1
    assert pair["high_pressure_after_q_change"] == 1


def test_fixture6_read_path_fail_cross_check() -> None:
    lanes = 16
    acc = [HIGH_PRESSURE_ABS] * lanes
    q = [0] * lanes
    records = [_record(0, acc, q), _record(1, acc, q)]
    receipt = _classify(records, cross_check_pass=False, cross_check_required=True)

    assert receipt["branch_selection"]["branch"] == BRANCH_READ_PATH_FAIL


def test_empty_denominator_intrinsic_fail_closed() -> None:
    lanes = 16
    records: list[dict[str, object]] = []
    q = [0] * lanes
    for step in range(3):
        acc = [HIGH_PRESSURE_ABS] * lanes
        records.append(_record(step, acc, q))
    receipt = _classify(records)
    metrics = receipt["trajectory_metrics"]

    assert int(metrics["intrinsic_pair_count"]) == 0
    assert metrics["intrinsic_persistence_fraction_mean"] is None
    assert receipt["branch_selection"]["branch"] != BRANCH_INTRINSIC


def test_compact_only_no_raw_arrays() -> None:
    lanes = 8
    acc = [HIGH_PRESSURE_ABS, 0, 0, 0, 0, 0, 0, 0]
    q = [0] * lanes
    records = [_record(0, acc, q), _record(1, acc, q)]
    receipt = _classify(records)

    assert receipt["raw_arrays_included"] is False
    assert "proxy_not_proof" in receipt["explicit_non_claims"]
    assert "no_trainer_or_gpu" in receipt["explicit_non_claims"]
    assert "no_decision_surface_claim" in receipt["explicit_non_claims"]
    payload = json.dumps(receipt)
    assert "accumulator_lanes" not in payload


def test_sidecar_index_roundtrip_via_append_helper() -> None:
    lanes = 8
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        sidecar_path = Path(tmp_dir) / "headroom_wiring_sidecar.jsonl"
        append_headroom_wiring_sidecar_chunk(
            sidecar_path,
            step=0,
            state_key="tiny.proj",
            accumulator_lanes=[0] * lanes,
            q_lanes=[0] * lanes,
        )
        append_headroom_wiring_sidecar_chunk(
            sidecar_path,
            step=1,
            state_key="tiny.proj",
            accumulator_lanes=[HIGH_PRESSURE_ABS] * lanes,
            q_lanes=[1] * lanes,
        )
        index = index_sidecar_records(sidecar_path)
        assert len(index["tiny.proj"]) == 2


def test_terminal_cross_check_helper_matches_r5_rows() -> None:
    lanes = 16
    acc = torch.tensor([HIGH_PRESSURE_ABS] * lanes, dtype=torch.int16)
    modules = {"mod0": acc}
    rows, content_sha = _synthetic_rows(modules)
    cross_check = cross_check_sidecar_against_receipt(
        modules=modules,
        receipt_rows=rows,
        expected_content_sha256=content_sha,
    )
    assert cross_check["cross_check_pass"] is True


def _synthetic_rows(
    modules: dict[str, torch.Tensor],
) -> tuple[list[dict[str, object]], str]:
    state_keys = sorted(modules.keys())
    payloads = [pack_w6_lanes_to_bytes(modules[key]) for key in state_keys]
    rows = build_r3_per_module_payload_rows(state_keys, payloads)
    content_sha = canonical_r3_packed_payload_content_sha256(rows)
    return rows, content_sha


@pytest.mark.slow
def test_banked_integration_not_run_by_default() -> None:
    if not BANKED_RUN_ROOT.is_dir():
        pytest.skip("banked run_root not available on this host")
    from calm.hrm_text_158.native_full_stack.r6_pressure_source_classifier_probe import (
        build_classifier_probe_receipt,
    )

    receipt = build_classifier_probe_receipt(
        run_root=BANKED_RUN_ROOT,
        head_sha256="e501e89de18b7a8a929854fb85633d2751ff901f",
        expected_receipt_sha256="a569eebfe49d57670899edad66f73c9de814eb5e1537a8dfb081e2e24568be1b",
        expected_sidecar_sha256="682fcfe2a9792b18c04f51e903a4192f3ca9181d570dbe7026f238afd03d6e0f",
    )
    assert receipt["cross_check"]["cross_check_pass"] is True
