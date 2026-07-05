"""CPU tests for obmalloc site-emit dedup reset witness + CA receipt plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
    F3BWhyState0Branch,
    validate_receipt_schema,
)
from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
    _DEDUP_RESET_WITNESS,
    build_obmalloc_dedup_evidence_from_witness,
    obmalloc_site_emit_dedup_reset_witness,
    record_obmalloc_site_emit_dedup_reset,
    reset_obmalloc_site_emit_dedup_session,
)
from calm.llm_computer.tests.test_f3b_prereg_schema_v1 import _minimal_valid_f3b_receipt


def _reset_witness_module_state() -> None:
    reset_obmalloc_site_emit_dedup_session()
    _DEDUP_RESET_WITNESS["called"] = False
    _DEDUP_RESET_WITNESS["scope"] = None


def test_witness_false_until_record_called() -> None:
    _reset_witness_module_state()
    assert obmalloc_site_emit_dedup_reset_witness()["called"] is False
    evidence = build_obmalloc_dedup_evidence_from_witness()
    assert evidence["dedup_reset_called"] is False
    assert evidence["dedup_session_scope"] is None


def test_witness_true_only_after_record_obmalloc_site_emit_dedup_reset() -> None:
    _reset_witness_module_state()
    record_obmalloc_site_emit_dedup_reset("probe_subprocess")
    witness = obmalloc_site_emit_dedup_reset_witness()
    assert witness["called"] is True
    assert witness["scope"] == "probe_subprocess"
    evidence = build_obmalloc_dedup_evidence_from_witness()
    assert evidence == {
        "dedup_reset_called": True,
        "dedup_session_scope": "probe_subprocess",
    }


def test_reset_alone_does_not_set_witness() -> None:
    _reset_witness_module_state()
    reset_obmalloc_site_emit_dedup_session()
    assert obmalloc_site_emit_dedup_reset_witness()["called"] is False
    assert build_obmalloc_dedup_evidence_from_witness()["dedup_reset_called"] is False


def test_ca_receipt_transcribes_dedup_and_parent_from_probe_receipt(tmp_path: Path) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        build_ca_band_counter_confirmation_receipt,
    )

    marks_b = [
        {
            "event": "s1d7_band_counter_C4.S1d.7",
            "state_index": state_index,
            "s1d7_band_counters": {
                "byte_proxies": {
                    "band_a_bytes": 1,
                    "band_c_bytes": 2,
                    "band_e_bytes": 0,
                },
                "counts": {"crossing_indices_len": 1 if state_index == 0 else 0},
            },
        }
        for state_index in range(2)
    ]
    receipt = build_ca_band_counter_confirmation_receipt(
        confirmation_root=tmp_path,
        n_states=2,
        run_a={"wall_seconds": 1.0},
        run_b={
            "wall_seconds": 2.0,
            "eligible_module_limit": 2,
            "probe_receipt": {
                "dedup_reset_called": True,
                "dedup_session_scope": "probe_subprocess",
                "parent_hash_after": "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
            },
        },
        marks_b=marks_b,
        sampled_states=(0, 1),
    )
    assert receipt["dedup_reset_called"] is True
    assert receipt["dedup_session_scope"] == "probe_subprocess"
    assert receipt["parent_sha"] == (
        "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
    )


def test_ca_receipt_omits_true_dedup_when_probe_receipt_absent(tmp_path: Path) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        build_ca_band_counter_confirmation_receipt,
    )

    marks_b = [
        {
            "event": "s1d7_band_counter_C4.S1d.7",
            "state_index": 0,
            "s1d7_band_counters": {
                "byte_proxies": {"band_a_bytes": 1, "band_c_bytes": 2, "band_e_bytes": 0},
                "counts": {"crossing_indices_len": 1},
            },
        }
    ]
    receipt = build_ca_band_counter_confirmation_receipt(
        confirmation_root=tmp_path,
        n_states=1,
        run_a={"wall_seconds": 1.0},
        run_b={"wall_seconds": 1.0, "eligible_module_limit": 1},
        marks_b=marks_b,
        sampled_states=(0,),
    )
    assert "dedup_reset_called" not in receipt
    assert "dedup_session_scope" not in receipt


def test_f3b_schema_fail_closed_decisive_branch_without_parent_sha() -> None:
    receipt = _minimal_valid_f3b_receipt()
    receipt["f3b_branch"] = F3BWhyState0Branch.STATE0_IDENTITY_STRUCTURE.value
    receipt["parent_sha"] = None
    failures = validate_receipt_schema(receipt)
    assert any("missing:parent_sha_for_decisive_branch" in failure for failure in failures)
