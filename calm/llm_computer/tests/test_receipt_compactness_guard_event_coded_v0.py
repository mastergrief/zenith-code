"""CPU battery: event-coded receipt compactness class rule + cap invariant."""
from __future__ import annotations

import json

import pytest

import copy

from calm.hrm_text_158.native_full_stack.event_coded_exact_geometry_receipt_validator_v0 import (
    EXACT_GEOMETRY_FAILURE_CLASSES,
    EXACT_GEOMETRY_LIVE_AUTHORITY,
    validate_event_coded_exact_geometry_receipt,
)
from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
    DECISIVE_Q_SNAPSHOT_KEY,
    RECEIPT_BANKABLE_MAX_BYTES,
    compact_probe_receipt_for_banking,
    estimate_receipt_json_bytes,
    find_raw_inline_index_violations,
    qualifies_as_raw_index_array,
    qualifies_as_raw_q_snapshot,
    validate_bankable_probe_receipt,
)
from scripts.a_prime_slice1_fidelity_core import (
    DEFAULT_PINNED_SUPPORTS,
    extract_prior_rates,
)


def test_bankable_cap_constant_unchanged() -> None:
    assert RECEIPT_BANKABLE_MAX_BYTES == 10 * 1024 * 1024


def test_class_rule_matches_indices_suffix_not_only_named_set() -> None:
    big = list(range(100))
    assert qualifies_as_raw_index_array("crossing_flat_indices", big)
    assert qualifies_as_raw_index_array("some_new_emitter_flat_indices", big)
    assert qualifies_as_raw_index_array("global_rate_cap_deferred_indices", big)
    assert not qualifies_as_raw_index_array("not_an_index_field", big)
    assert not qualifies_as_raw_index_array("crossing_flat_indices", list(range(10)))


def test_compact_removes_v4_observed_surfaces_and_q_snapshot() -> None:
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "mod": {
                            "crossing_flat_indices": list(range(200)),
                            "applied_flat_indices": list(range(150)),
                            DECISIVE_Q_SNAPSHOT_KEY: {str(i): 1 for i in range(100)},
                            "global_rate_cap_deferred_indices": list(range(300)),
                            "q_changed_count": 7,
                        }
                    }
                }
            }
        },
        "prior_audit": {
            "enabled": True,
            "per_support": {
                "L0b": {
                    "final": {"strict_exact": "230/230"},
                    "support_hash16": "89174273d21845bc",
                    "support_rows_expected": 230,
                },
                "math_a0": {
                    "final": {"strict_exact": "1255/1255"},
                    "support_hash16": "56e64266357b793d",
                    "support_rows_expected": 1255,
                },
            },
            "support_proofs": {
                "L0b": {
                    "support_hash16": "89174273d21845bc",
                    "expected_count": 230,
                },
                "math_a0": {
                    "support_hash16": "56e64266357b793d",
                    "expected_count": 1255,
                },
            },
        },
    }
    # known-bad pre-compact: raw surfaces present
    assert find_raw_inline_index_violations(receipt)
    compact_probe_receipt_for_banking(receipt)
    assert find_raw_inline_index_violations(receipt) == []
    stats = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["mod"]
    assert "crossing_flat_indices" not in stats
    assert "crossing_flat_indices_summary" in stats
    assert stats["crossing_flat_indices_summary"]["count"] == 200
    assert "applied_flat_indices" not in stats
    assert "global_rate_cap_deferred_indices" not in stats
    assert DECISIVE_Q_SNAPSHOT_KEY not in stats
    assert f"{DECISIVE_Q_SNAPSHOT_KEY}_summary" in stats
    assert stats["q_changed_count"] == 7
    # reducer-required prior_audit fields preserved
    rates = extract_prior_rates(receipt, pinned_supports=DEFAULT_PINNED_SUPPORTS)
    assert rates["ok"] is True
    assert rates["aggregate_total"] == 230 + 1255


def test_known_bad_size_trip_then_cure() -> None:
    huge = list(range(2_000_000))  # raw list ~ multi-MB
    receipt = {
        "blob": {"mystery_emitter_flat_indices": huge},
        "prior_audit": {"enabled": False},
    }
    failures = validate_bankable_probe_receipt(receipt)
    assert failures, "raw huge index list must fail validation"
    compact_probe_receipt_for_banking(receipt)
    failures2 = validate_bankable_probe_receipt(receipt)
    assert failures2 == []
    assert estimate_receipt_json_bytes(receipt) < RECEIPT_BANKABLE_MAX_BYTES


def test_decisive_q_snapshot_qualifier() -> None:
    assert qualifies_as_raw_q_snapshot(
        DECISIVE_Q_SNAPSHOT_KEY, {"0": 1, "1": -1, "2": 0}
    )
    assert not qualifies_as_raw_q_snapshot("other", {"0": 1})
    assert not qualifies_as_raw_q_snapshot(DECISIVE_Q_SNAPSHOT_KEY, {"a": 1})


def test_nested_list_of_list_escape_closed() -> None:
    """Known-bad: dict under list-of-list must fire pre-scan and clear post-compact."""

    nested_payload = list(range(100))
    receipt = {
        "outer": [[{"mystery_emitter_flat_indices": nested_payload}]],
        "prior_audit": {"enabled": False},
    }
    pre = find_raw_inline_index_violations(receipt)
    assert pre, "nested list-of-list raw index leaf must fail pre-scan"
    assert any("mystery_emitter_flat_indices" in item for item in pre)
    compact_probe_receipt_for_banking(receipt)
    post = find_raw_inline_index_violations(receipt)
    assert post == []
    leaf = receipt["outer"][0][0]
    assert "mystery_emitter_flat_indices" not in leaf
    assert "mystery_emitter_flat_indices_summary" in leaf
    assert leaf["mystery_emitter_flat_indices_summary"]["count"] == 100
    assert validate_bankable_probe_receipt(receipt) == []


def _known_good_exact_geometry_fixture(n_steps: int = 20) -> dict:
    """Minimal real-shape fixture: per-step global_summary + tensor_stats authority."""

    step_reports: dict = {}
    for i in range(1, n_steps + 1):
        step_reports[str(i)] = {
            "step_result": {
                "global_summary": {
                    "event_coded_live_carrier_enabled": True,
                    "global_rate_cap_enabled": False,
                    "q_changed_count": 0,
                },
                "tensor_stats": {
                    "model.H_level.core.layers.0.attn.gqkv_proj": {
                        "live_authority": EXACT_GEOMETRY_LIVE_AUTHORITY,
                        "q_changed_count": 0,
                    }
                },
            }
        }
    return {
        "steps_requested": n_steps,
        "steps_completed": n_steps,
        "step_reports": step_reports,
        "persistent_accumulator_event_coded_live": True,
        "event_coded_sparse_vote_authority": True,
        "bounded_delta_global_summary": {
            "event_coded_live_carrier_enabled": True,
            "global_rate_cap_enabled": False,
        },
        "device": "cuda:0",
        "gpu_launched": True,
        "gpu_launch_authorized": True,
        "forward_backward_update_executed": True,
        "device_guard": {
            "cuda_available": True,
            "pass": True,
            "device": "cuda:0",
            "device_type": "cuda",
        },
    }


def test_exact_geometry_known_good_silent() -> None:
    failures = validate_event_coded_exact_geometry_receipt(
        _known_good_exact_geometry_fixture()
    )
    assert failures == []


def test_exact_geometry_known_bad_mutation_battery() -> None:
    """Each declared geometry class: mutate known-good → check FIRES; good silent."""

    good = _known_good_exact_geometry_fixture()
    assert validate_event_coded_exact_geometry_receipt(good) == []

    def classes_of(receipt: dict) -> set[str]:
        return {f["class"] for f in validate_event_coded_exact_geometry_receipt(receipt)}

    observed_classes: set[str] = set()

    def expect(cls: str, receipt: dict) -> None:
        hit = classes_of(receipt)
        assert cls in hit, f"expected class {cls!r} fired, got {hit!r}"
        observed_classes.add(cls)

    # steps_requested
    bad_req = copy.deepcopy(good)
    bad_req["steps_requested"] = 19
    expect("steps_requested", bad_req)

    # short steps (steps_completed)
    short = copy.deepcopy(good)
    short["steps_completed"] = 19
    expect("steps_completed", short)

    # missing step key
    missing = copy.deepcopy(good)
    del missing["step_reports"]["7"]
    expect("step_reports_coverage", missing)

    # top-level event-coded live flag
    top_live = copy.deepcopy(good)
    top_live["persistent_accumulator_event_coded_live"] = False
    expect("toplevel_event_coded_live", top_live)

    # top-level sparse vote authority
    top_sparse = copy.deepcopy(good)
    top_sparse["event_coded_sparse_vote_authority"] = False
    expect("toplevel_sparse_vote_authority", top_sparse)

    # cap-enabled mid-step (1..19) while last-step BDGS still False
    cap_mid = copy.deepcopy(good)
    cap_mid["step_reports"]["5"]["step_result"]["global_summary"][
        "global_rate_cap_enabled"
    ] = True
    assert cap_mid["bounded_delta_global_summary"]["global_rate_cap_enabled"] is False
    expect("per_step_global_rate_cap", cap_mid)

    # live-carrier-disabled mid-step
    live_off = copy.deepcopy(good)
    live_off["step_reports"]["3"]["step_result"]["global_summary"][
        "event_coded_live_carrier_enabled"
    ] = False
    assert live_off["bounded_delta_global_summary"][
        "event_coded_live_carrier_enabled"
    ] is True
    expect("per_step_event_coded_live", live_off)

    # wrong live_authority in one tensor stat
    bad_auth = copy.deepcopy(good)
    bad_auth["step_reports"]["8"]["step_result"]["tensor_stats"][
        "model.H_level.core.layers.0.attn.gqkv_proj"
    ]["live_authority"] = "dense_vote_acc"
    expect("live_authority", bad_auth)

    # missing GPU evidence
    no_gpu = copy.deepcopy(good)
    no_gpu["gpu_launched"] = False
    expect("gpu_execution_evidence", no_gpu)

    # BDGS corroboration mismatch (per-step still clean)
    bdgs_bad = copy.deepcopy(good)
    bdgs_bad["bounded_delta_global_summary"]["global_rate_cap_enabled"] = True
    expect("bdgs_corroboration", bdgs_bad)

    # Completeness: every declared class has a durable negative calibration.
    assert observed_classes == set(EXACT_GEOMETRY_FAILURE_CLASSES), (
        f"battery missing classes={sorted(set(EXACT_GEOMETRY_FAILURE_CLASSES) - observed_classes)}; "
        f"extra={sorted(observed_classes - set(EXACT_GEOMETRY_FAILURE_CLASSES))}"
    )

    # known-good still silent after all mutations on copies
    assert validate_event_coded_exact_geometry_receipt(good) == []
