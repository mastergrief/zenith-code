"""Regression: probe receipts must stay bankable (no huge inline index arrays)."""
from __future__ import annotations

import json

from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
    RECEIPT_BANKABLE_MAX_BYTES,
    compact_probe_receipt_for_banking,
    find_raw_inline_index_violations,
    validate_bankable_probe_receipt,
)


def _synthetic_receipt_with_huge_indices(*, modules: int = 32, indices_per_module: int = 5000) -> dict:
    tensor_stats = {}
    for module_index in range(modules):
        tensor_stats[f"model.module_{module_index}"] = {
            "applied_indices": list(range(indices_per_module)),
            "pre_veto_selected_indices": list(range(indices_per_module)),
            "applied_flat_indices_hash16": "deadbeefdeadbeef",
        }
    return {
        "schema": "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0",
        "step_reports": {
            "1": {"tensor_stats": tensor_stats},
        },
    }


def test_compact_guard_removes_raw_inline_index_arrays() -> None:
    receipt = _synthetic_receipt_with_huge_indices()
    assert find_raw_inline_index_violations(receipt)

    compact_probe_receipt_for_banking(receipt)
    failures = validate_bankable_probe_receipt(receipt)
    assert failures == []
    assert len(json.dumps(receipt).encode("utf-8")) < RECEIPT_BANKABLE_MAX_BYTES

    stats = receipt["step_reports"]["1"]["tensor_stats"]["model.module_0"]
    assert "applied_indices" not in stats
    assert stats["applied_indices_summary"]["tier_a_index_surface_omitted"] is True
    assert stats["applied_indices_summary"]["len"] == 5000


def test_compact_guard_preserves_hash_summary_fields() -> None:
    receipt = _synthetic_receipt_with_huge_indices(modules=2, indices_per_module=128)
    compact_probe_receipt_for_banking(receipt)
    stats = receipt["step_reports"]["1"]["tensor_stats"]["model.module_0"]
    assert "applied_flat_indices_hash16" in stats["applied_indices_summary"]
