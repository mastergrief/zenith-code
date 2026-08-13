"""Regression: probe receipts must stay bankable (recursive compactness guard)."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from calm.hrm_text_158.native_full_stack.front_c_identity_emitter import (
    classify_front_c_identity_payload,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import _tensor_sha256
from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
    RECEIPT_BANKABLE_MAX_BYTES,
    ReceiptCompactnessCollisionError,
    canonical_int64_index_list_sha256_v1,
    compact_probe_receipt_for_banking,
    find_raw_inline_index_violations,
    validate_bankable_probe_receipt,
)


def _synthetic_receipt_with_huge_indices(
    *, modules: int = 32, indices_per_module: int = 5000
) -> dict:
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


def _nested_step_result_receipt(
    *,
    modules: int = 4,
    indices_per_module: int = 200,
    include_deferred: bool = True,
) -> dict:
    tensor_stats: dict = {}
    for module_index in range(modules):
        row = {
            "pre_veto_selected_indices": list(range(indices_per_module)),
            "post_veto_would_apply_pre_cap_indices": list(range(indices_per_module)),
            "applied_indices": list(range(min(32, indices_per_module))),
            "q_changed_count": 1,
        }
        if include_deferred:
            deferred = list(range(indices_per_module))
            row["global_rate_cap_deferred_indices"] = deferred
            row["global_rate_cap_deferred_indices_sha256"] = (
                canonical_int64_index_list_sha256_v1(deferred)
            )
            row["global_rate_cap_deferred_count"] = len(deferred)
        tensor_stats[f"model.module_{module_index}"] = row
    return {
        "schema": "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0",
        "step_reports": {
            "1": {"step_result": {"tensor_stats": tensor_stats}},
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
    assert stats["applied_indices_summary"]["count"] == 5000


def test_compact_guard_preserves_hash_summary_fields() -> None:
    receipt = _synthetic_receipt_with_huge_indices(modules=2, indices_per_module=128)
    compact_probe_receipt_for_banking(receipt)
    stats = receipt["step_reports"]["1"]["tensor_stats"]["model.module_0"]
    assert "applied_flat_indices_hash16" in stats["applied_indices_summary"]
    assert len(stats["applied_indices_summary"]["order_sensitive_content_hash16"]) == 16


def test_compact_guard_nested_step_result_tensor_stats_is_compacted_not_noop() -> None:
    """Characterization: live embed path is step_result.tensor_stats (was silent no-op)."""

    receipt = _nested_step_result_receipt(indices_per_module=200)
    assert find_raw_inline_index_violations(receipt)
    nested = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["model.module_0"]
    assert isinstance(nested["pre_veto_selected_indices"], list)
    assert len(nested["pre_veto_selected_indices"]) == 200

    compact_probe_receipt_for_banking(receipt)
    assert find_raw_inline_index_violations(receipt) == []
    stats = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["model.module_0"]
    assert "pre_veto_selected_indices" not in stats
    assert "global_rate_cap_deferred_indices" not in stats
    assert stats["pre_veto_selected_indices_summary"]["count"] == 200
    assert stats["global_rate_cap_deferred_indices_summary"]["count"] == 200
    assert "global_rate_cap_deferred_indices_sha256" in stats
    assert len(stats["global_rate_cap_deferred_indices_sha256"]) == 64


def test_recursive_coverage_vote_pressure_and_arbitrary_nesting() -> None:
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "pre_veto_selected_indices": list(range(100)),
                        }
                    },
                    "extra": {
                        "deeper": {
                            "global_rate_cap_accepted_indices": list(range(80)),
                        }
                    },
                },
                "vote_pressure": [
                    {
                        "state_key": "m0",
                        "replay_ce_veto_indices": list(range(90)),
                    }
                ],
            }
        }
    }
    assert find_raw_inline_index_violations(receipt)
    compact_probe_receipt_for_banking(receipt)
    assert find_raw_inline_index_violations(receipt) == []
    assert (
        "pre_veto_selected_indices_summary"
        in receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["m0"]
    )
    assert (
        "global_rate_cap_accepted_indices_summary"
        in receipt["step_reports"]["1"]["step_result"]["extra"]["deeper"]
    )
    assert (
        "replay_ce_veto_indices_summary"
        in receipt["step_reports"]["1"]["vote_pressure"][0]
    )


def test_small_ordinary_lists_preserved() -> None:
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "pre_veto_selected_indices": list(range(32)),
                            "ordinary_scores": list(range(200)),
                            "q_changed_count": 3,
                        }
                    }
                }
            }
        }
    }
    compact_probe_receipt_for_banking(receipt)
    stats = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["m0"]
    assert stats["pre_veto_selected_indices"] == list(range(32))
    assert stats["ordinary_scores"] == list(range(200))
    assert stats["q_changed_count"] == 3


def test_post_transform_independent_scan_hard_fails_if_raw_remains() -> None:
    receipt = _nested_step_result_receipt(indices_per_module=100)
    compact_probe_receipt_for_banking(receipt)
    assert validate_bankable_probe_receipt(receipt) == []
    receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["model.module_0"][
        "pre_veto_selected_indices"
    ] = list(range(100))
    failures = validate_bankable_probe_receipt(receipt)
    assert any("pre_veto_selected_indices" in item for item in failures)


def test_representative_20_step_receipt_under_10mib() -> None:
    step_reports = {}
    for step in range(1, 21):
        tensor_stats = {}
        for module_index in range(32):
            big = list(range(512))
            tensor_stats[f"model.module_{module_index}"] = {
                "pre_veto_selected_indices": list(big),
                "post_veto_would_apply_pre_cap_indices": list(big),
                "global_rate_cap_deferred_indices": list(big),
                "applied_indices": list(range(16)),
            }
        step_reports[str(step)] = {"step_result": {"tensor_stats": tensor_stats}}
    receipt = {"schema": "probe/v0", "step_reports": step_reports}
    compact_probe_receipt_for_banking(receipt)
    failures = validate_bankable_probe_receipt(receipt)
    assert failures == []
    size = len(json.dumps(receipt, separators=(",", ":")).encode("utf-8"))
    assert size < RECEIPT_BANKABLE_MAX_BYTES
    assert size < 1_000_000


def test_producer_equivalent_sha_matches_tensor_sha256_helper() -> None:
    for indices in ([], [0, 1, 2], list(range(1000)), list(range(0, 20000, 7))):
        guard = canonical_int64_index_list_sha256_v1(indices)
        producer = _tensor_sha256(torch.tensor(indices, dtype=torch.int64))
        assert guard == producer
        assert len(guard) == 64


def test_existing_matching_K_count_and_K_sha256_accepted() -> None:
    indices = list(range(100))
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "global_rate_cap_deferred_indices": list(indices),
                            "global_rate_cap_deferred_indices_count": 100,
                            "global_rate_cap_deferred_indices_sha256": (
                                canonical_int64_index_list_sha256_v1(indices)
                            ),
                        }
                    }
                }
            }
        }
    }
    compact_probe_receipt_for_banking(receipt)
    stats = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["m0"]
    assert stats["global_rate_cap_deferred_indices_count"] == 100
    assert stats["global_rate_cap_deferred_indices_sha256"] == (
        canonical_int64_index_list_sha256_v1(indices)
    )


def test_count_mismatch_hard_fails() -> None:
    indices = list(range(100))
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "global_rate_cap_deferred_indices": list(indices),
                            "global_rate_cap_deferred_indices_count": 99,
                        }
                    }
                }
            }
        }
    }
    with pytest.raises(ReceiptCompactnessCollisionError, match="count"):
        compact_probe_receipt_for_banking(receipt)


def test_full_hash_mismatch_hard_fails() -> None:
    indices = list(range(100))
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "global_rate_cap_deferred_indices": list(indices),
                            "global_rate_cap_deferred_indices_sha256": "0" * 64,
                        }
                    }
                }
            }
        }
    }
    with pytest.raises(ReceiptCompactnessCollisionError, match="sha256"):
        compact_probe_receipt_for_banking(receipt)


def test_hash16_never_accepted_as_full_K_sha256_equivalent() -> None:
    indices = list(range(100))
    from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import _sha16

    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "global_rate_cap_deferred_indices": list(indices),
                            "global_rate_cap_deferred_indices_sha256": _sha16(indices),
                        }
                    }
                }
            }
        }
    }
    with pytest.raises(ReceiptCompactnessCollisionError):
        compact_probe_receipt_for_banking(receipt)


def test_input_receipt_raw_lists_unchanged_under_transform_on_copy() -> None:
    receipt = _nested_step_result_receipt(indices_per_module=120)
    raw_ref = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["model.module_0"][
        "pre_veto_selected_indices"
    ]
    raw_snapshot = list(raw_ref)
    compact_probe_receipt_for_banking(receipt)
    assert raw_ref == raw_snapshot
    assert isinstance(raw_ref, list)
    assert "pre_veto_selected_indices" not in receipt["step_reports"]["1"]["step_result"][
        "tensor_stats"
    ]["model.module_0"]


def test_compact_guard_front_c_identity_signal_transitions_to_count_hash_not_neither() -> None:
    indices = list(range(100))
    payload_before = {
        "wrapper": {
            "global_rate_cap_deferred_indices": list(indices),
            "global_rate_cap_accepted_indices": list(range(80)),
        }
    }
    before = classify_front_c_identity_payload(payload_before)
    assert "global_rate_cap_deferred_indices" in before.observed_identity_keys
    assert "global_rate_cap_accepted_indices" in before.observed_identity_keys

    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "tensor_stats": {
                        "m0": {
                            "global_rate_cap_deferred_indices": list(indices),
                            "global_rate_cap_accepted_indices": list(range(80)),
                            "ordinary_small": [1, 2, 3],
                        }
                    }
                }
            }
        }
    }
    compact_probe_receipt_for_banking(receipt)
    after = classify_front_c_identity_payload(receipt)
    assert "global_rate_cap_deferred_indices" not in after.observed_identity_keys
    assert "global_rate_cap_accepted_indices" not in after.observed_identity_keys
    assert "global_rate_cap_deferred_indices_count" in after.observed_count_or_hash_keys
    assert "global_rate_cap_deferred_indices_sha256" in after.observed_count_or_hash_keys
    assert "global_rate_cap_accepted_indices_count" in after.observed_count_or_hash_keys
    assert "global_rate_cap_accepted_indices_sha256" in after.observed_count_or_hash_keys

    stats = receipt["step_reports"]["1"]["step_result"]["tensor_stats"]["m0"]
    assert stats["ordinary_small"] == [1, 2, 3]
    assert len(stats["global_rate_cap_deferred_indices_sha256"]) == 64
    assert len(stats["global_rate_cap_deferred_indices_summary"]["order_sensitive_content_hash16"]) == 16
    assert (
        stats["global_rate_cap_deferred_indices_summary"]["order_sensitive_content_hash16"]
        != stats["global_rate_cap_deferred_indices_sha256"]
    )


def test_decay_n50_phase_gate_only_two_phases() -> None:
    from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
        should_omit_tensor_stats_for_decay_n50,
    )

    assert should_omit_tensor_stats_for_decay_n50("ful-decay-stability-n50-control")
    assert should_omit_tensor_stats_for_decay_n50("ful-decay-stability-n50-treatment")
    assert not should_omit_tensor_stats_for_decay_n50("d-recompute-window-feasibility")
    assert not should_omit_tensor_stats_for_decay_n50("ful-decay-feasibility-shape-s2")


def test_omit_tensor_stats_replaces_module_map_and_keeps_operands() -> None:
    from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
        omit_step_result_tensor_stats,
        should_omit_tensor_stats_for_decay_n50,
    )

    global_summary = {
        "global_rate_cap_accepted_count": 12,
        "global_rate_cap_deferred_count": 4,
        "global_deferred_ratio": 0.25,
        "deferred_backlog_size": 7,
        "deferred_backlog_max_age_steps": 3,
        "global_rate_cap_ordering_summary": {"full_demand_count": 16},
    }
    tensor_stats = {
        "model.module_0": {"q_changed_count": 1, "applied_indices": [0, 1, 2]},
        "model.module_1": {"q_changed_count": 0},
    }
    receipt = {
        "step_reports": {
            "1": {
                "step_result": {
                    "global_summary": copy.deepcopy(global_summary),
                    "tensor_stats": copy.deepcopy(tensor_stats),
                    "vote_pressure": [{"state_key": "m0"}],
                }
            },
            "2": {
                "step_result": {
                    "global_summary": copy.deepcopy(global_summary),
                    "tensor_stats": copy.deepcopy(tensor_stats),
                    "vote_pressure": [{"state_key": "m0"}],
                }
            },
        },
        "prior_audit": {
            "final_reports": {"L0b": {"acquired": True}},
            "deltas": {
                "math_a0": {"no_new_broad_cluster": True},
                "L0b": {"no_new_broad_cluster": True, "new_strict_failure_count": 0},
                "L0c1": {"new_strict_failure_count": 0},
            },
        },
    }

    for step in receipt["step_reports"].values():
        original_ts = step["step_result"]["tensor_stats"]
        step["step_result"] = omit_step_result_tensor_stats(step["step_result"])
        stub = step["step_result"]["tensor_stats"]
        assert stub == {
            "omitted": True,
            "reason": "decay_stability_n50_receipt_cap",
            "n_modules": 2,
        }
        assert "model.module_0" not in stub
        assert original_ts == tensor_stats
        gs = step["step_result"]["global_summary"]
        assert gs == global_summary
        assert step["step_result"]["vote_pressure"] == [{"state_key": "m0"}]

    pa = receipt["prior_audit"]
    assert pa["final_reports"]["L0b"]["acquired"] is True
    assert pa["deltas"]["math_a0"]["no_new_broad_cluster"] is True
    assert pa["deltas"]["L0b"]["no_new_broad_cluster"] is True
    assert pa["deltas"]["L0b"]["new_strict_failure_count"] == 0
    assert pa["deltas"]["L0c1"]["new_strict_failure_count"] == 0

    assert should_omit_tensor_stats_for_decay_n50("ful-decay-stability-n50-control")
    assert should_omit_tensor_stats_for_decay_n50("ful-decay-stability-n50-treatment")
    for near_miss in (
        "d-recompute-window-feasibility",
        "ful-decay-feasibility-shape-s2",
        "ful-decay-stability-n50",
        "",
        "FUL-DECAY-STABILITY-N50-CONTROL",
        "ful-decay-stability-n50-control ",
        "ful-decay-stability-n50-control-extra",
    ):
        assert not should_omit_tensor_stats_for_decay_n50(near_miss)
