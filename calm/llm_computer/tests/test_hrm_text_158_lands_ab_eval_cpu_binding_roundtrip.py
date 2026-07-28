"""CPU roundtrip / B2 binding tests for LANDS-AB (IMPLEMENT_v10 split)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
    LandsAbReducerSchemaError,
    all_true_matrix,
    matrix_with,
    reduce_lands_ab_branch_strict,
)
from calm.llm_computer.tests.lands_ab_eval_test_helpers import (
    base_ok as _base_ok,
    write_real_cpu_row as _write_real_cpu_row,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_EQUIVALENT,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_VACUOUS,
    CANONICAL_CELL_KEYS,
)














def test_probe_b2_bare_changed_boolean_without_payload_values_rejected():
    """co_lead probe 3 + claude gate-1: bare changed / arbitrary distinct without twin → False."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_roundtrip,
    )
    production = {
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 2,
        "post_update_payload_changed": True,  # bare boolean
        "post_update_authoritative_state_payload_sha256": "",
        "pre_update_authoritative_state_payload_sha256": "",
    }
    compare = {"sparse_event_count": 2}
    bind = bind_production_to_twin_roundtrip(production=production, compare=compare)
    assert bind["production_sparse_matches_twin"] is False
    assert bind["payload_value_ok"] is False
    # arbitrary distinct pre/post WITHOUT twin_post → False (dead-code mutation path closed)
    production2 = {
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 2,
        "post_update_payload_changed": True,
        "pre_update_authoritative_state_payload_sha256": "1" * 64,
        "post_update_authoritative_state_payload_sha256": "2" * 64,
    }
    bind2 = bind_production_to_twin_roundtrip(production=production2, compare=compare)
    assert bind2["production_sparse_matches_twin"] is False
    assert bind2["payload_eq_mode"] == "twin_post_missing"
    # production post == twin post → pass
    compare3 = {
        "sparse_event_count": 2,
        "twin_post_authoritative_state_payload_sha256": "2" * 64,
    }
    bind3 = bind_production_to_twin_roundtrip(production=production2, compare=compare3)
    assert bind3["production_sparse_matches_twin"] is True
    assert bind3["payload_eq_mode"] == "production_post_equals_twin_post"
    # production post != twin post → fail
    compare4 = {
        "sparse_event_count": 2,
        "twin_post_authoritative_state_payload_sha256": "3" * 64,
    }
    bind4 = bind_production_to_twin_roundtrip(production=production2, compare=compare4)
    assert bind4["production_sparse_matches_twin"] is False
    assert bind4["payload_value_ok"] is False
