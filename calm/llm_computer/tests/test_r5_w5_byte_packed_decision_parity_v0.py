"""R5 W5 byte-packed decision-parity slice: codec, checkpoint, ledger, classifier."""
from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W5_SIGNED_MAX,
    W5_SIGNED_MIN,
    clip_then_pack_w5,
    clip_then_roundtrip_w5_tensor,
    pack_w5,
    pack_w5_lanes_to_bytes,
    unpack_w5,
    unpack_w5_lanes_from_bytes,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV,
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    apply_trainer_boundary_narrow_carrier,
    assert_no_dual_persistent_acc_byte_packing,
    persistent_w5_byte_packed_enabled,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    measure_r5_persistent_state_budget,
    pack_ternary_q_2bit_reference,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.s3bb_decision_parity import (
    CLASSIFIER_DECISION_MISMATCH,
    CLASSIFIER_DECISION_PARITY_OK,
    CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL,
    CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT,
    CLASSIFIER_HARNESS_OR_LIVENESS_FAIL,
    classify_s3bb_decision_parity_run,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
    MEASURED_STEPS_REQUIRED,
    WARMUP_STEPS,
    append_headroom_wiring_sidecar_chunk,
    compute_headroom_telemetry_from_accumulators,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY,
    W5_BYTE_PACKED_PAYLOAD_KEY,
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    select_trainer_eligible_bitlinears,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_r5_persistent_ledger_receipt,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _in_domain_w5_tensor(numel: int) -> torch.Tensor:
    values = torch.arange(numel, dtype=torch.int32) % (W5_SIGNED_MAX - W5_SIGNED_MIN + 1)
    values = values + W5_SIGNED_MIN
    return values.to(torch.int16).reshape(4, numel // 4)


def test_w5_codec_strict_clip_and_roundtrip() -> None:
    assert pack_w5(15) == pack_w5(W5_SIGNED_MAX)
    assert unpack_w5(pack_w5(-15)) == -15
    assert clip_then_pack_w5(100) == pack_w5(W5_SIGNED_MAX)
    assert clip_then_pack_w5(-100) == pack_w5(W5_SIGNED_MIN)
    acc = torch.tensor([[-16, 0, 15, 29]], dtype=torch.int16)
    clipped = clip_then_roundtrip_w5_tensor(acc)
    assert int(clipped.min()) >= W5_SIGNED_MIN
    assert int(clipped.max()) <= W5_SIGNED_MAX
    with pytest.raises(ValueError, match="pack_w5 requires"):
        pack_w5(16)


def _large_q_for_inclusive_gate() -> torch.Tensor:
    levels = torch.tensor([-1, 0, 1], dtype=torch.int8)
    idx = torch.arange(1024 * 1024, dtype=torch.long) % 3
    return levels[idx].view(1024, 1024).contiguous()


def test_w5_byte_pack_roundtrip_and_ledger_bytes() -> None:
    acc = _in_domain_w5_tensor(128)
    payload = pack_w5_lanes_to_bytes(acc)
    restored = unpack_w5_lanes_from_bytes(payload)
    torch.testing.assert_close(restored, acc, atol=0, rtol=0)
    assert payload.packed.dtype == torch.uint8
    assert payload.packed_data_bytes == math.ceil(int(acc.numel()) * 5 / 8)
    q = _large_q_for_inclusive_gate()
    decoded = torch.arange(q.numel(), dtype=torch.int32) % 31 - 15
    decoded = decoded.to(torch.int16).view(q.shape)
    payload_large = pack_w5_lanes_to_bytes(decoded)
    qstate = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(1.0, dtype=torch.float32),
    )
    q_packed = pack_ternary_q_2bit_reference(qstate.q_levels)
    report = measure_r5_persistent_state_budget([qstate], [q_packed], [payload_large])
    assert report.r5_acc_physical_bits_per_weight == pytest.approx(5.0, abs=0.25)
    assert report.r5_checkpoint_inclusive_physical_bits_per_weight <= 7.5
    assert report.r5_ledger_pass is True
    assert "NOT lossless" in report.receipt_statement


def test_w5_trainer_boundary_clip_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, "1")
    acc = torch.tensor([[20, -20, 3]], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc)
    assert int(out.min()) >= W5_SIGNED_MIN
    assert int(out.max()) <= W5_SIGNED_MAX


def test_checkpoint_w5_roundtrip_and_dual_persistence_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV, "1")
    assert persistent_w5_byte_packed_enabled()
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        w5_byte_packed_enabled=True,
    )
    assert blob["trainer_sub2_authority"]["w5_byte_packed_persistent_accumulator_saved"] is True
    payload = blob["trainer_sub2_authority"]["tensor_payloads"]["proj"]["bounded_accumulator"]
    assert payload[W5_BYTE_PACKED_ACCUMULATOR_PERSISTED_KEY] is True
    assert payload[W5_BYTE_PACKED_PAYLOAD_KEY].dtype == torch.uint8
    with pytest.raises(ValueError, match="dual persistent"):
        assert_no_dual_persistent_acc_byte_packing(
            {
                "w5_byte_packed_accumulator_persisted": True,
                "w6_byte_packed_accumulator_persisted": True,
                "dense_int16_accumulator_persisted": False,
            }
        )
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    loaded = load_trainer_sub2_authority_checkpoint_blob(
        fresh,
        blob,
        eligible_modules=fresh_eligible,
        w5_byte_packed_enabled=True,
    )
    for key in eligible:
        before = decode_bounded_accumulator_to_i16(states[key].bounded_accumulator)
        after = decode_bounded_accumulator_to_i16(loaded[key].bounded_accumulator)
        assert torch.equal(before, after)


def test_rejects_w5_checkpoint_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    monkeypatch.setenv(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV, "1")
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
        w5_byte_packed_enabled=True,
    )
    monkeypatch.delenv(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV, raising=False)
    fresh = _TinyTernary()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="byte-packed W5"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            blob,
            eligible_modules=fresh_eligible,
            w5_byte_packed_enabled=False,
        )


def test_build_r5_persistent_ledger_receipt_disabled_without_flags() -> None:
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    assert build_r5_persistent_ledger_receipt(
        states,
        q_packed_enabled=False,
        acc_w5_byte_packed_enabled=False,
    ) == {"enabled": False}


def _telemetry_block(acc_values: list[int]) -> dict:
    acc = torch.tensor(acc_values, dtype=torch.int16).reshape(1, -1)
    return compute_headroom_telemetry_from_accumulators(acc)


def _step_report(
    *,
    step_id: str,
    acc_values: list[int],
    q_values: list[int],
    applied_indices: list[int],
    q_sha_after: str = "sha_oracle",
    q_changed_count: int = 1,
    metrics: dict | None = None,
) -> dict:
    return {
        "step_id": step_id,
        "q_changed_count": q_changed_count,
        "metrics": metrics or {"loss": 0.5, "accuracy": 0.25},
        "headroom_telemetry": _telemetry_block(acc_values),
        "step_result": {
            "tensor_stats": {
                "tiny.proj": {
                    "applied_indices": applied_indices,
                    "q_sha256_after": q_sha_after,
                    "flip_count": len(applied_indices),
                }
            }
        },
    }


def _write_sidecar_pair(
    tmp_path: Path,
    *,
    steps: int,
    oracle_acc: list[int],
    treatment_acc: list[int],
    q_values: list[int],
) -> tuple[Path, Path]:
    oracle_path = tmp_path / "oracle_sidecar.jsonl"
    treatment_path = tmp_path / "treatment_sidecar.jsonl"
    for step in range(1, steps + 1):
        append_headroom_wiring_sidecar_chunk(
            oracle_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=oracle_acc,
            q_lanes=q_values,
        )
        append_headroom_wiring_sidecar_chunk(
            treatment_path,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=treatment_acc,
            q_lanes=q_values,
        )
    return oracle_path, treatment_path


def _parity_receipts_with_sidecars(
    tmp_path: Path,
    *,
    steps: int = MEASURED_STEPS_REQUIRED,
    treatment_w5: bool = True,
) -> tuple[dict, dict]:
    acc = [5, -9, 10]
    q = [0, 1, -1]
    oracle_sidecar, treatment_sidecar = _write_sidecar_pair(
        tmp_path,
        steps=steps,
        oracle_acc=acc,
        treatment_acc=acc,
        q_values=q,
    )
    step_reports: dict[str, dict] = {}
    for step in range(1, steps + 1):
        step_reports[str(step)] = _step_report(
            step_id=str(step),
            acc_values=acc,
            q_values=q,
            applied_indices=[0, 2],
            q_sha_after="same_sha",
            q_changed_count=2,
        )
    oracle = {
        "steps_completed": steps,
        "stop_reason": "",
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(oracle_sidecar),
        "receipt_emit_profile": "s3bb_headroom_diagnostic_slim",
    }
    treatment = {
        "steps_completed": steps,
        "stop_reason": "",
        "step_reports": copy.deepcopy(step_reports),
        "headroom_wiring_sidecar_path": str(treatment_sidecar),
        "receipt_emit_profile": "s3bb_headroom_diagnostic_slim",
        "persistent_accumulator_w5_byte_packed": bool(treatment_w5),
        "r5_persistent_ledger": {
            "enabled": True,
            "r5_ledger_pass": True,
        },
    }
    return oracle, treatment


def test_classifier_decision_parity_ok_despite_bit_equality_drift(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment_sidecar = Path(treatment["headroom_wiring_sidecar_path"])
    treatment_sidecar.unlink(missing_ok=True)
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            treatment_sidecar,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=[6, -9, 10],
            q_lanes=[0, 1, -1],
        )
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_PARITY_OK
    bit_diag = stats["bit_equality_diagnostics"]
    assert float(bit_diag.get("per_step_accumulator_l1_max_abs_delta", 0.0)) > 0.0


def test_classifier_decision_mismatch_on_crossing(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment_sidecar = Path(treatment["headroom_wiring_sidecar_path"])
    treatment_sidecar.unlink(missing_ok=True)
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            treatment_sidecar,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=[11, -9, 10],
            q_lanes=[0, 1, -1],
        )
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_MISMATCH


def test_classifier_decision_mismatch_on_applied_mask(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]["applied_indices"] = [1]
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_MISMATCH


def test_classifier_dynamics_drift_on_q_sha(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]["q_sha256_after"] = "drift"
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_FLIP_EQUIVALENT_DYNAMICS_DRIFT


def test_classifier_domain_fail_on_w5_out_of_domain(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment_sidecar = Path(treatment["headroom_wiring_sidecar_path"])
    treatment_sidecar.unlink(missing_ok=True)
    for step in range(1, MEASURED_STEPS_REQUIRED + 1):
        append_headroom_wiring_sidecar_chunk(
            treatment_sidecar,
            step=step,
            state_key="tiny.proj",
            accumulator_lanes=[5, -9, 16],
            q_lanes=[0, 1, -1],
        )
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DOMAIN_OR_HEADROOM_FAIL


def test_classifier_harness_fail_on_missing_w5_ledger(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment["r5_persistent_ledger"] = {"enabled": False}
    primary, _ = classify_s3bb_decision_parity_run(
        oracle,
        treatment,
        require_w5_ledger=True,
    )
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL


def test_classifier_harness_fail_on_short_run(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path, steps=3)
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL


def test_classifier_harness_fail_on_missing_tensor_stats(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    for receipt in (oracle, treatment):
        for step_report in receipt["step_reports"].values():
            step_report["step_result"] = {}
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["observable_coverage_pass"] is False


def test_classifier_harness_fail_on_module_key_asymmetric(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    treatment["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.other"] = {
        "applied_indices": [],
        "q_sha256_after": "sha",
    }
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["state_key_parity_failures"]


def test_classifier_harness_fail_on_applied_indices_absent(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    for receipt in (oracle, treatment):
        entry = receipt["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]
        entry.pop("applied_indices", None)
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["applied_indices_present_module_steps"] < stats[
        "observable_coverage"
    ]["compared_module_steps"]


def test_classifier_harness_fail_on_q_sha_absent_or_empty(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    for receipt in (oracle, treatment):
        entry = receipt["step_reports"]["4"]["step_result"]["tensor_stats"]["tiny.proj"]
        entry["q_sha256_after"] = ""
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["q_sha256_after_present_module_steps"] < stats[
        "observable_coverage"
    ]["compared_module_steps"]


def test_classifier_harness_fail_on_missing_final_metric(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    oracle["step_reports"]["10"]["metrics"] = {"accuracy": 0.25}
    treatment["step_reports"]["10"]["metrics"] = {"accuracy": 0.25}
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["observable_coverage_pass"] is False
    assert "loss" not in stats["observable_coverage"]["final_metric_coverage"]["oracle_present"]


def test_classifier_harness_fail_on_zero_compared_module_steps(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path, steps=2)
    primary, stats = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_HARNESS_OR_LIVENESS_FAIL
    assert stats["observable_coverage"]["shared_measured_step_count"] == 0
    assert stats["observable_coverage"]["compared_module_steps"] == 0


def test_classifier_allows_present_empty_applied_indices(tmp_path: Path) -> None:
    oracle, treatment = _parity_receipts_with_sidecars(tmp_path)
    for receipt in (oracle, treatment):
        for step_report in receipt["step_reports"].values():
            step_report["step_result"]["tensor_stats"]["tiny.proj"]["applied_indices"] = []
    primary, _ = classify_s3bb_decision_parity_run(oracle, treatment)
    assert primary == CLASSIFIER_DECISION_PARITY_OK
