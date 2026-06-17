"""R5 scratch liveness-audit parser fixture (watch-wrap real log)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

AUDIT_PATH = Path(
    "/home/gabe/.ai-room/scratch/racc_mode_b_arm_a_liveness_audit_1781550251902.py"
)
REAL_LOG = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "racc_mode_b_arm_a_20260617T092902Z/logs/racc_mode_b_arm_a.log"
)
# Frozen STEP-1 v2 §6 pinned set (14 phases for step 1/12 partial run).
PINNED_PHASES_14 = frozenset(
    {
        "STEP_START",
        "STEP_CE_BEFORE_START",
        "STEP_CE_BEFORE_DONE",
        "STEP_FORWARD_BACKWARD_START",
        "STEP_FORWARD_DONE",
        "STEP_BACKWARD_DONE",
        "STEP_VOTE_DONE",
        "REFERENCE_DEEPCOPY_START",
        "REFERENCE_DEEPCOPY_DONE",
        "RETENTION_CALLER_DEL_DONE",
        "RETENTION_GC_DONE",
        "SPARSE_EVENTS_START",
        "LIVE_OBJECT_LIVENESS_APPLY_START",
        "APPLY_START",
    }
)


def _load_audit():
    spec = importlib.util.spec_from_file_location("racc_liveness_audit_r5", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r5_watchwrap_parser_indexes_real_partial_log() -> None:
    audit = _load_audit()
    log_text = REAL_LOG.read_text(encoding="utf-8")
    by_phase = audit.parse_beacon_log(log_text)
    indexed_phases = set(by_phase)
    assert PINNED_PHASES_14.issubset(indexed_phases)
    assert len(PINNED_PHASES_14) == 14
    for phase in PINNED_PHASES_14:
        assert 0 in by_phase[phase]
    assert "APPLY_START" in by_phase


def test_r5_watchwrap_parser_sparse_events_done_fields() -> None:
    audit = _load_audit()
    log_text = REAL_LOG.read_text(encoding="utf-8")
    by_phase = audit.parse_beacon_log(log_text)
    sparse = by_phase["SPARSE_EVENTS_DONE"][0]
    assert sparse["sparse_carrier_path_used"] == "True"
    assert sparse["total_sparse_vote_event_count"] == "19299136"
