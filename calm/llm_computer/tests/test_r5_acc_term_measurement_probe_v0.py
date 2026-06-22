"""CPU fixtures for the R5 acc-term measurement probe (frozen v4)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import pack_w6_lanes_to_bytes
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    build_r3_per_module_payload_rows,
    canonical_r3_packed_payload_content_sha256,
)
from calm.hrm_text_158.native_full_stack.r5_acc_term_measurement_probe import (
    BRANCH_A1_DENSE_LOSSLESS,
    BRANCH_B_SPARSE_LOSSLESS_WINS,
    BRANCH_C_LOSSY_DECISION_PARITY,
    BRANCH_D_REPRESENTATION_LIMIT,
    PARITY_DECISION,
    PARITY_LOSSLESS,
    THRESHOLD_ABS,
    build_measurement_from_modules,
    cross_check_sidecar_against_receipt,
    exact_packed_bpw,
    extract_last_sidecar_records,
    modules_from_sidecar_records,
    next_gate_parity_type,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    append_headroom_wiring_sidecar_chunk,
)

BANKED_RUN_ROOT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "r4_1_q_bytepacked_tensorwide_seed43_20260622T164500Z"
)
BANKED_RECEIPT_SHA = (
    "a569eebfe49d57670899edad66f73c9de814eb5e1537a8dfb081e2e24568be1b"
)
BANKED_SIDECAR_SHA = (
    "682fcfe2a9792b18c04f51e903a4192f3ca9181d570dbe7026f238afd03d6e0f"
)
BANKED_CONTENT_SHA = (
    "db52ee9aaf3496b2c2c8fe9b2e95f20300e7f3cb0e90d10863470885a4ac46ca"
)


def _synthetic_rows(
    modules: dict[str, torch.Tensor],
) -> tuple[list[dict[str, object]], str]:
    state_keys = sorted(modules.keys())
    payloads = [pack_w6_lanes_to_bytes(modules[key]) for key in state_keys]
    rows = build_r3_per_module_payload_rows(state_keys, payloads)
    content_sha = canonical_r3_packed_payload_content_sha256(rows)
    return rows, content_sha


def _measure(
    acc: torch.Tensor,
    *,
    state_key: str = "mod0",
    observed_steps: dict[str, int] | None = None,
) -> dict[str, object]:
    shape = tuple(int(dim) for dim in acc.shape)
    modules = {state_key: acc.contiguous()}
    logical_shapes = {state_key: list(shape)}
    rows, content_sha = _synthetic_rows(modules)
    cross_check = cross_check_sidecar_against_receipt(
        modules=modules,
        receipt_rows=rows,
        expected_content_sha256=content_sha,
    )
    assert cross_check["cross_check_pass"] is True
    return build_measurement_from_modules(
        modules=modules,
        logical_shapes=logical_shapes,
        cross_check=cross_check,
        observed_max_step_per_module=observed_steps,
    )


def test_fixture1_synthetic_1024_lane_dense_frontier_and_thresholds() -> None:
    lanes = 1024
    values: list[int] = []
    pattern = [0, 7, -7, 10, -10, 31, -31]
    for index in range(lanes):
        values.append(pattern[index % len(pattern)])
    acc = torch.tensor(values, dtype=torch.int16).reshape(lanes)
    receipt = _measure(acc)

    assert receipt["raw_arrays_included"] is False
    dense = receipt["per_module"]["mod0"]["dense_frontier"]
    assert dense["widths"]["6"]["fit_boolean"] is True
    assert dense["widths"]["4"]["fit_boolean"] is False
    threshold = dense["threshold"]
    assert threshold["count_abs_gte_10"] > 0
    assert threshold["max_abs"] == 31
    assert threshold["margin_to_threshold"] == float(THRESHOLD_ABS - 31)


def test_fixture2_sparse_beats_dense_selects_branch_b() -> None:
    lanes = 1024
    values = [0] * 1000 + [1] * 24
    acc = torch.tensor(values, dtype=torch.int16).reshape(lanes)
    receipt = _measure(acc)

    branch = receipt["branch_selection"]
    assert branch["branch"] == BRANCH_B_SPARSE_LOSSLESS_WINS
    sparse_bpw = float(receipt["aggregate"]["sparse_projection"]["bounded_delta_acc_bits_per_weight"])
    min_lossless = int(branch["min_lossless_width"])
    best_dense = exact_packed_bpw(min_lossless, lanes)
    assert sparse_bpw <= best_dense - 0.25


def test_fixture3_step8_sidecar_cross_check_pass_without_step_ten_assertion() -> None:
    lanes = 64
    acc = torch.zeros(lanes, dtype=torch.int16)
    acc[0] = 3
    acc[1] = -2
    modules = {"tiny.proj": acc.contiguous()}
    logical_shapes = {"tiny.proj": [lanes]}
    rows, content_sha = _synthetic_rows(modules)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sidecar_path = Path(tmp_dir) / "headroom_wiring_sidecar.jsonl"
        append_headroom_wiring_sidecar_chunk(
            sidecar_path,
            step=7,
            state_key="tiny.proj",
            accumulator_lanes=[0] * lanes,
            q_lanes=[0] * lanes,
        )
        append_headroom_wiring_sidecar_chunk(
            sidecar_path,
            step=8,
            state_key="tiny.proj",
            accumulator_lanes=acc.reshape(-1).tolist(),
            q_lanes=[0] * lanes,
        )
        records, observed = extract_last_sidecar_records(sidecar_path)

    assert observed["tiny.proj"] == 8
    rebuilt = modules_from_sidecar_records(records, logical_shapes)
    cross_check = cross_check_sidecar_against_receipt(
        modules=rebuilt,
        receipt_rows=rows,
        expected_content_sha256=content_sha,
    )
    assert cross_check["cross_check_pass"] is True

    receipt = build_measurement_from_modules(
        modules=rebuilt,
        logical_shapes=logical_shapes,
        cross_check=cross_check,
        observed_max_step_per_module=observed,
    )
    assert receipt["observed_max_step_per_module"]["tiny.proj"] == 8
    assert receipt["observed_max_step_aggregate"] == 8
    assert receipt["raw_arrays_included"] is False


def test_fixture4_fits_w4_but_sparse_wins_branch_b_not_a1() -> None:
    lanes = 1024
    values = [0] * 1010 + [5] * 14
    acc = torch.tensor(values, dtype=torch.int16).reshape(lanes)
    receipt = _measure(acc)

    dense = receipt["aggregate"]["dense_frontier"]
    assert dense["widths"]["4"]["fit_boolean"] is True
    assert dense["min_lossless_width"] == 4

    branch = receipt["branch_selection"]
    assert branch["branch"] == BRANCH_B_SPARSE_LOSSLESS_WINS
    assert branch["branch"] != BRANCH_A1_DENSE_LOSSLESS.format(n=4)
    sparse_bpw = float(branch["sparse_bpw"])
    assert sparse_bpw <= exact_packed_bpw(4, lanes) - 0.25


def test_fixture5_a1_w4_when_sparse_does_not_beat_dense() -> None:
    lanes = 1024
    values = [0] * 884 + [5] * 140
    acc = torch.tensor(values, dtype=torch.int16).reshape(lanes)
    receipt = _measure(acc)

    dense = receipt["aggregate"]["dense_frontier"]
    assert dense["max_abs"] == 5
    assert dense["min_lossless_width"] == 4

    branch = receipt["branch_selection"]
    assert branch["branch"] == BRANCH_A1_DENSE_LOSSLESS.format(n=4)
    assert branch["a1_next_gate_parity_type"] == PARITY_DECISION
    sparse_bpw = float(branch["sparse_bpw"])
    assert sparse_bpw > exact_packed_bpw(4, lanes) - 0.25


def test_fixture6_w6_only_sparse_not_winning_selects_branch_d_not_c() -> None:
    lanes = 1024
    values = [18] * 512 + [-18] * 512
    acc = torch.tensor(values, dtype=torch.int16).reshape(lanes)
    receipt = _measure(acc)

    dense = receipt["aggregate"]["dense_frontier"]
    assert dense["min_lossless_width"] == 6

    branch = receipt["branch_selection"]
    assert branch["branch"] == BRANCH_D_REPRESENTATION_LIMIT
    assert branch["branch"] != BRANCH_C_LOSSY_DECISION_PARITY
    sparse_bpw = float(branch["sparse_bpw"])
    assert sparse_bpw > exact_packed_bpw(6, lanes) - 0.25
    assert "c_deferred_clip_details" in branch
    assert branch["c_deferred_clip_details"]["4"]["max_abs_after_clip"] <= 7
    assert "c_deferred_note" in branch


def test_next_gate_parity_type_formula_f6() -> None:
    assert next_gate_parity_type(5) == PARITY_LOSSLESS
    assert next_gate_parity_type(4) == PARITY_DECISION
    assert next_gate_parity_type(3) == PARITY_DECISION
    assert next_gate_parity_type(2) == PARITY_DECISION


def test_receipt_enforces_explicit_non_claims_and_no_raw_arrays() -> None:
    acc = torch.tensor([0, 1, -1, 2], dtype=torch.int16)
    receipt = _measure(acc)
    assert receipt["raw_arrays_included"] is False
    assert "no_raw_per_lane_arrays" in receipt["explicit_non_claims"]
    assert "no_decision_surface_claim_from_static_probe" in receipt["explicit_non_claims"]
    payload = json.dumps(receipt)
    assert "accumulator_lanes" not in payload


@pytest.mark.slow
def test_banked_integration_read_path_sha_pinned() -> None:
    if not BANKED_RUN_ROOT.is_dir():
        pytest.skip("banked run_root not available on this host")
    from calm.hrm_text_158.native_full_stack.r5_acc_term_measurement_probe import (
        build_measurement_probe_receipt,
    )

    receipt = build_measurement_probe_receipt(
        run_root=BANKED_RUN_ROOT,
        head_sha256="98a034fa7e18d21aa53c76ff77e5e4a238bff5c5",
        expected_receipt_sha256=BANKED_RECEIPT_SHA,
        expected_sidecar_sha256=BANKED_SIDECAR_SHA,
    )
    assert receipt["raw_arrays_included"] is False
    assert receipt["cross_check"]["cross_check_pass"] is True
    assert receipt["cross_check"]["built_content_sha256"] == BANKED_CONTENT_SHA
    assert receipt["branch_selection"]["branch"] not in {"HARNESS_FAIL", "READ_PATH_FAIL"}
