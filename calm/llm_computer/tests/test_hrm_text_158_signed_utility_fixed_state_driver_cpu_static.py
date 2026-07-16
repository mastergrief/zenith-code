"""CPU-static driver characterization vs d1_r1 + authoritative-not-toy proofs (PLAN v5)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_driver import (
    AuthoritativeGpuDeferredError,
    authoritative_path_must_not_route_to_toy_source_pass,
    run_authoritative_fixed_state_signed_utility,
    run_developer_check_cpu_static,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    PRIVATE_TRUSTED_CORE,
    static_private_core_prohibition_pass,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    rehash_path,
)

CREDIT = Path("/home/gabe/claw-code-creditdir/transient_fp_credit")
FIXTURE = CREDIT / "fa_accounting_v2_post_seam_signed_utility_d1_r1_fixture_partition_v1.json"
D1_HARNESS = CREDIT / "fa_accounting_v2_post_seam_signed_utility_d1_r1_harness.py"
MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_driver.py"
WATCH = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/bin/watch-wrap")
VOTE = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm_text_158/native_full_stack/vote_update.py")


def _ff():
    return json.loads(FIXTURE.read_text())


def test_loc_budget_under_500():
    assert sum(1 for _ in MOD.open()) <= 420


def test_driver_static_private_core_prohibition():
    src = MOD.read_text(encoding="utf-8")
    assert static_private_core_prohibition_pass(src) is True
    assert PRIVATE_TRUSTED_CORE not in src.replace(
        "'_apply_integer_vote_update_from_frozen_plan' + '_trusted'", "PRIVATE_ASSEMBLED"
    ).replace(
        '"_apply_integer_vote_update_from_frozen_plan" + "_trusted"', "PRIVATE_ASSEMBLED"
    )


def test_d1_r1_characterization_parity_call_counts_and_classifier():
    diag = run_developer_check_cpu_static(_ff())
    assert diag["raw_holder_call_count"] == 1
    assert diag["eligible_state_key_count"] == 2
    assert diag["apply_integer_vote_update_from_frozen_plan_calls"] == 4  # 2*N
    assert diag["expected_internal_private_core_via_public"] == 4
    assert diag["mutation_parity"]["pass"] is True
    assert diag["noop_base_unchanged"] is True
    assert diag["classifier"] in {
        "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN",
        "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL",
    }
    assert diag["harness_static_private_core_prohibition_pass"] is True
    # d1_r1 harness bytes remain immutable forensic baseline
    assert (
        __import__("hashlib").sha256(D1_HARNESS.read_bytes()).hexdigest()
        == "386e6c6715269b5af8ec7e9477da0877bb3ba01e9db3cb13734fc0dd22612140"
    )


def test_call_count_formula_is_two_times_eligible_key_count():
    diag = run_developer_check_cpu_static(_ff())
    n = diag["eligible_state_key_count"]
    assert diag["apply_integer_vote_update_from_frozen_plan_calls"] == 2 * n


def test_authoritative_does_not_route_to_toy_and_is_deferred():
    src = MOD.read_text(encoding="utf-8")
    assert authoritative_path_must_not_route_to_toy_source_pass(src) is True
    packet = {
        "source_pins": {
            "watch_wrap": {"absolute_path": str(WATCH), "sha256": rehash_path(WATCH)},
            "vote_update": {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)},
        }
    }
    with pytest.raises(AuthoritativeGpuDeferredError, match="authoritative_gpu_deferred"):
        run_authoritative_fixed_state_signed_utility(packet)


def test_authoritative_source_excludes_toy_entrypoints():
    import ast

    src = MOD.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if getattr(n, "name", "") == "run_authoritative_fixed_state_signed_utility")
    body = ast.get_source_segment(src, fn) or ""
    for banned in (
        "run_developer_check_cpu_static(",
        "evaluate_cpu_static(",
        "synthetic_nll",
        "cpu_static_micro",
    ):
        assert banned not in body
