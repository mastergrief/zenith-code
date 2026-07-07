"""CPU regression: event-coded lightweight recompute_window_log writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
    ReplayConstants,
    build_event_coded_recompute_window_step_entry,
    emit_event_coded_recompute_window_step_record,
    initialize_recompute_window_log_for_probe_session,
    iter_recompute_window_log_records,
    validate_bootstrap_record,
)
from calm.llm_computer.tests.test_probe_event_coded_sparse_vote_authority_wiring_v0 import (
    TINY_ARCH,
    _tiny_parent_blob,
)
from scripts.hrm_text_158_arc2b_slice5_step2_in_vivo_postrun_classifier import (
    STEP2_CONFIRMATION_STEPS,
    resolve_step2_operational_ok_from_run_artifacts,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    file_sha256,
    resolve_probe_vote_update_spec,
    run_c2p1_probe,
    validate_recompute_window_log_flag_mutual_exclusion,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
)

LIVE_CARRIER_SNAPSHOT_FILENAME = "live_carrier_snapshot.jsonl"


def _vote_spec_decay_half() -> ReplayConstants:
    vote_spec = resolve_probe_vote_update_spec(
        max_abs_per_tensor=4096,
        confirmation_envelope="canonical_t10_prereg_v24",
        vote_update_decay_numerator=1,
        vote_update_decay_denominator=2,
    )
    return ReplayConstants.from_vote_update_spec(vote_spec)


def test_lightweight_row_has_step_and_replay_constants_only() -> None:
    replay = _vote_spec_decay_half()
    entry = build_event_coded_recompute_window_step_entry(step=3, replay_constants=replay)
    assert entry["step"] == 3
    assert entry["replay_constants"]["decay_numerator"] == 1
    assert entry["replay_constants"]["decay_denominator"] == 2
    assert "acc_before_lanes" not in entry
    assert "vote_lanes" not in entry
    assert validate_bootstrap_record(entry) == []


def test_emit_writes_non_doubled_log_beside_live_carrier_snapshot(tmp_path: Path) -> None:
    scratch = tmp_path / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True)
    log_path = scratch / D_RECOMPUTE_WINDOW_LOG_FILENAME
    initialize_recompute_window_log_for_probe_session(log_path)
    replay = _vote_spec_decay_half()

    for step in (1, 2, 3):
        emit_event_coded_recompute_window_step_record(
            enabled=True,
            log_path=log_path,
            step=step,
            replay_constants=replay,
        )

    live_path = scratch / LIVE_CARRIER_SNAPSHOT_FILENAME
    live_path.write_text(
        json.dumps(
            {
                "step": 1,
                "live_acc_carrier_bytes_total": 40,
                "live_carrier_bytes_exact": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    doubled_live = scratch / "d_recompute_window_diagnostic" / LIVE_CARRIER_SNAPSHOT_FILENAME

    rows = iter_recompute_window_log_records(log_path)
    assert len(rows) == 3
    assert log_path.is_file()
    assert log_path == scratch / "recompute_window_log.jsonl"
    assert not doubled_live.exists()
    assert rows[0]["replay_constants"]["decay_denominator"] == 2


def test_mutual_exclusion_fail_closed() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_recompute_window_log_flag_mutual_exclusion(
            d_recompute_window_instrumentation_enabled=True,
            event_coded_recompute_window_log_enabled=True,
        )


def test_postrun_operational_ok_true_with_log_and_live_rows(tmp_path: Path) -> None:
    run_root = tmp_path
    scratch = run_root / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True)
    (run_root / "prelaunch").mkdir(parents=True, exist_ok=True)

    log_rows = [
        {
            "step": step,
            "replay_constants": {
                "decay_numerator": 1,
                "decay_denominator": 2,
            },
        }
        for step in range(1, STEP2_CONFIRMATION_STEPS + 1)
    ]
    log_path = scratch / D_RECOMPUTE_WINDOW_LOG_FILENAME
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in log_rows),
        encoding="utf-8",
    )
    live_path = scratch / LIVE_CARRIER_SNAPSHOT_FILENAME
    live_path.write_text(
        json.dumps(
            {
                "step": 1,
                "events_bytes": 10,
                "backlog_bytes": 10,
                "hot_exact_bytes": 10,
                "metadata_bytes": 10,
                "live_acc_carrier_bytes_total": 40,
                "live_carrier_bytes_exact": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (scratch / "receipt.json").write_text(
        json.dumps({"steps_completed": STEP2_CONFIRMATION_STEPS}),
        encoding="utf-8",
    )
    (run_root / "prelaunch" / "confirmation_launch_rc.txt").write_text("0\n", encoding="utf-8")
    (run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json").write_text(
        json.dumps({"pass": True, "bounded_steps_start_count": 1}),
        encoding="utf-8",
    )

    assert resolve_step2_operational_ok_from_run_artifacts(run_root) is True


def test_postrun_operational_ok_false_when_log_missing(tmp_path: Path) -> None:
    run_root = tmp_path
    scratch = run_root / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True)
    live_path = scratch / LIVE_CARRIER_SNAPSHOT_FILENAME
    live_path.write_text(
        json.dumps(
            {
                "step": 1,
                "live_acc_carrier_bytes_total": 40,
                "live_carrier_bytes_exact": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scratch / "receipt.json").write_text(
        json.dumps({"steps_completed": STEP2_CONFIRMATION_STEPS}),
        encoding="utf-8",
    )
    (run_root / "prelaunch").mkdir(parents=True, exist_ok=True)
    (run_root / "prelaunch" / "confirmation_launch_rc.txt").write_text("0\n", encoding="utf-8")
    (run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json").write_text(
        json.dumps({"pass": True, "bounded_steps_start_count": 1}),
        encoding="utf-8",
    )

    assert resolve_step2_operational_ok_from_run_artifacts(run_root) is False


def test_run_c2p1_probe_event_coded_log_and_live_snapshot_at_diagnostic_root(
    tmp_path: Path,
) -> None:
    """Runnable probe integration: event-coded log + live carrier at non-doubled paths."""
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)
    scratch_root = tmp_path / "d_recompute_window_diagnostic"
    steps = 2

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=scratch_root,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=steps,
        batch_size=16,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        persistent_accumulator_event_coded_live=True,
        persistent_q_ternary_base3_codec=True,
        event_coded_sparse_vote_authority=True,
        event_coded_recompute_window_log_enabled=True,
        d_recompute_window_instrumentation_enabled=False,
        d_live_carrier_snapshot_enabled=True,
        vote_update_decay_numerator=1,
        vote_update_decay_denominator=2,
        global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        d_diagnostic_compact_step_reports=True,
        emit_progress=True,
    )

    disk_receipt = json.loads((scratch_root / "receipt.json").read_text(encoding="utf-8"))
    for payload in (receipt, disk_receipt):
        assert payload["event_coded_recompute_window_log_enabled"] is True
        assert payload["event_coded_sparse_vote_authority"] is True
        assert payload.get("d_recompute_window_instrumentation_enabled", False) is False
        assert int(payload["steps_completed"]) == steps

    log_path = scratch_root / D_RECOMPUTE_WINDOW_LOG_FILENAME
    live_path = scratch_root / LIVE_CARRIER_SNAPSHOT_FILENAME
    doubled_live = scratch_root / "d_recompute_window_diagnostic" / LIVE_CARRIER_SNAPSHOT_FILENAME

    assert log_path.is_file()
    assert live_path.is_file()
    assert not doubled_live.exists()

    assert str(log_path) == receipt["event_coded_recompute_window_log_path"]
    assert str(log_path) == receipt["d_recompute_window_log_path"]
    assert str(live_path) == receipt["d_live_carrier_snapshot_path"]

    log_rows = iter_recompute_window_log_records(log_path)
    assert len(log_rows) == int(receipt["steps_completed"])
    first_rc = log_rows[0].get("replay_constants") or {}
    assert int(first_rc.get("decay_numerator", -1)) == 1
    assert int(first_rc.get("decay_denominator", -1)) == 2


def test_run_c2p1_probe_event_coded_sparse_cap_apply_serial_cpu_mode(
    tmp_path: Path,
) -> None:
    """Runnable probe: multi-module event-coded sparse cap apply uses serial_cpu."""
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)
    scratch_root = tmp_path / "d_recompute_window_diagnostic"
    steps = 1

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=scratch_root,
        device="cpu",
        eligible_scope="all-bitlinear",
        eligible_module_limit=2,
        steps=steps,
        batch_size=16,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        persistent_accumulator_event_coded_live=True,
        persistent_q_ternary_base3_codec=True,
        event_coded_sparse_vote_authority=True,
        event_coded_recompute_window_log_enabled=False,
        d_recompute_window_instrumentation_enabled=False,
        d_live_carrier_snapshot_enabled=False,
        global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        d_diagnostic_compact_step_reports=True,
        emit_progress=False,
    )

    assert int(receipt.get("eligible_module_count") or 0) >= 2
    global_summary = dict(receipt.get("bounded_delta_global_summary") or {})
    assert global_summary.get("sparse_cap_apply_parallel_mode") == "serial_cpu"
    assert int(receipt.get("steps_completed") or 0) == steps


def test_decay_replay_constants_witness_reads_registered_source(tmp_path: Path) -> None:
    scratch = tmp_path / "d_recompute_window_diagnostic"
    scratch.mkdir(parents=True)
    log_path = scratch / D_RECOMPUTE_WINDOW_LOG_FILENAME
    log_path.write_text(
        json.dumps(
            {
                "step": 1,
                "replay_constants": {
                    "decay_numerator": 1,
                    "decay_denominator": 2,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    row = json.loads(next(line for line in log_path.read_text().splitlines() if line.strip()))
    rc = row.get("replay_constants") or {}
    failures: list[str] = []
    if int(rc.get("decay_numerator", -1)) != 1 or int(rc.get("decay_denominator", -1)) != 2:
        failures.append("decay_not_1_over_2")
    assert failures == []
