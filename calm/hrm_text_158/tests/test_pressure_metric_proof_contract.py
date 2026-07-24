"""Proof-contract / LIVE amendment / per-index determinism tests (fork-2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
    ordered_selection_frame,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
    SOURCE_FILES,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_proof_contract import (
    ContractError,
    LIVE_AMENDMENT_RELPATH,
    applied_identity_sha256_from_frames,
    compare_per_index,
    load_live_amendment,
    per_index_all_equal,
    refuse_old_cpu_baseline_proof,
    validate_live_amendment_doc,
    validate_proof_against_live_amendment,
    validate_replicate_per_index_fields,
)

REPO = Path(__file__).resolve().parents[3]


def test_proof_py_under_500_loc() -> None:
    n = len(
        (REPO / "calm/hrm_text_158/native_full_stack/pressure_metric_proof.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert n < 500, n


def test_source_files_include_contract_and_bridge() -> None:
    assert any(p.endswith("pressure_metric_proof_contract.py") for p in SOURCE_FILES)
    assert any(p.endswith("pressure_metric_gpu_loop_bridge.py") for p in SOURCE_FILES)
    assert any(p.endswith("pressure_metric_gpu_lifecycle_derisk.py") for p in SOURCE_FILES)


def test_live_amendment_loads_and_binds() -> None:
    doc, digest = load_live_amendment(str(REPO))
    validate_live_amendment_doc(doc)
    assert len(digest) == 64
    assert (REPO / LIVE_AMENDMENT_RELPATH).is_file()


def test_refuse_old_cpu_baseline() -> None:
    with pytest.raises(ContractError):
        refuse_old_cpu_baseline_proof({"legacy_cpu_selection_baseline": True})
    with pytest.raises(ContractError):
        refuse_old_cpu_baseline_proof({"shared_gpu_baseline": False})
    with pytest.raises(ContractError):
        refuse_old_cpu_baseline_proof({"paired_baseline": {"variable": "cpu_argsort"}})


def test_wrong_amendment_sha_refused() -> None:
    doc, _digest = load_live_amendment(str(REPO))
    bad = {
        "proof_contract_amendment_sha256": "0" * 64,
        "shared_gpu_baseline": True,
        "overhead_denominator": doc["overhead_denominator"],
    }
    with pytest.raises(ContractError):
        validate_proof_against_live_amendment(bad, repo_root=str(REPO))


def _frame(step: int, idxs: list[int]) -> bytes:
    return ordered_selection_frame(
        step=step, ordered_flat_idx=torch.tensor(idxs, dtype=torch.int64)
    )


def test_ordered_selection_identity_negative_aggregate_equal_index_differ() -> None:
    """R-v2.2: equal counts/final mask, different intermediate selected indices → fail."""
    frames_a = [_frame(1, [0, 1, 2]), _frame(2, [3, 4, 5]), _frame(3, [0, 1, 2])]
    frames_b = [_frame(1, [3, 4, 5]), _frame(2, [0, 1, 2]), _frame(3, [0, 1, 2])]
    ha = applied_identity_sha256_from_frames(frames_a)
    hb = applied_identity_sha256_from_frames(frames_b)
    assert ha != hb

    flip = "a" * 64
    qh = "b" * 64
    row_a = {
        "flip_count_sha256": flip,
        "q_final_sha256": qh,
        "applied_identity_sha256": ha,
        "flip_count_equal": True,
        "q_final_equal": True,
        "applied_identity_equal": False,
        "measurements": {"n_flips": 10, "q_changed_count": 3, "credited_mass": 7},
    }
    row_b = {
        "flip_count_sha256": flip,
        "q_final_sha256": qh,
        "applied_identity_sha256": hb,
        "flip_count_equal": True,
        "q_final_equal": True,
        "applied_identity_equal": False,
        "measurements": {"n_flips": 10, "q_changed_count": 3, "credited_mass": 7},
    }
    assert row_a["measurements"] == row_b["measurements"]
    flags = compare_per_index(row_a, row_b)
    assert flags["flip_count_equal"] and flags["q_final_equal"]
    assert not flags["applied_identity_equal"]
    assert not per_index_all_equal(flags)

    replicates = {
        "AB_A": [row_a],
        "AB_B": [row_b],
        "BA_A": [row_a],
        "BA_B": [row_b],
    }
    with pytest.raises(ContractError):
        validate_replicate_per_index_fields(replicates)


def test_producer_accepted_false_when_per_index_mismatch() -> None:
    frames_a = [_frame(1, [1, 2]), _frame(2, [3, 4])]
    frames_b = [_frame(1, [3, 4]), _frame(2, [1, 2])]
    a = {
        "flip_count_sha256": "1" * 64,
        "q_final_sha256": "2" * 64,
        "applied_identity_sha256": applied_identity_sha256_from_frames(frames_a),
    }
    b = {
        "flip_count_sha256": "1" * 64,
        "q_final_sha256": "2" * 64,
        "applied_identity_sha256": applied_identity_sha256_from_frames(frames_b),
    }
    assert a["flip_count_sha256"] == b["flip_count_sha256"]
    per_index_ok = per_index_all_equal(compare_per_index(a, b))
    accepted = True and per_index_ok
    assert accepted is False


def test_live_artifact_schema_matches_file() -> None:
    raw = json.loads((REPO / LIVE_AMENDMENT_RELPATH).read_text(encoding="utf-8"))
    assert raw["status"] == "LIVE"
    assert raw["old_cpu_baseline_proof_semantics"] == "REFUSED"
