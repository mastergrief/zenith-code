"""Claim-authority safe construction — mode alone cannot claim; stamp unexported."""
from __future__ import annotations

import importlib
import inspect

from calm.hrm_text_158.native_full_stack import (
    forgotten_accum_training_equivalence_materialization as mat,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_measure import (
    apply_claim_coupling,
    stamp_mode_labels,
)


def test_materialization_all_excludes_stamp_authority_receipt():
    assert "stamp_authority_receipt" not in mat.__all__
    assert not hasattr(mat, "stamp_authority_receipt")


def test_claim_export_bypass_impossible_mode_formal_alone():
    out = apply_claim_coupling(
        {"status": "OK", "notes": {}},
        mode="formal",
    )
    assert out["claimable_science"] is False
    assert out["bankable"] is False
    assert "RULE_CONFLICT_UNRESOLVED" in out["notes"]["formal_claim_blockers"]
    assert out["run_kind"] == "FORMAL_SCIENCE"


def test_stamp_mode_labels_never_sets_claimable_true():
    labeled = stamp_mode_labels({"status": "OK"}, mode="formal")
    assert "claimable_science" not in labeled or labeled.get("claimable_science") is not True
    assert labeled.get("bankable") is not True


def test_all_formal_caller_reachability_census_stamp_or_successor():
    """Production callers must use bank_measure.apply_claim_coupling, not mat stamp."""

    launch = importlib.import_module(
        "calm.hrm_text_158.native_full_stack.forgotten_accum_run_arms_launch"
    )
    src = inspect.getsource(launch)
    assert "stamp_authority_receipt" not in src
    assert "apply_claim_coupling" in src
    assert launch.apply_claim_coupling is apply_claim_coupling
