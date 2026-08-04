"""Argv-contract tests for A′ fidelity wrapper probe_argv (exemption geometry).

Move-only extraction from test_a_prime_slice1_fidelity_wrapper_exit.py (architecture
stop: exit module line-cap). Assertion bodies preserved.
"""
from __future__ import annotations

from pathlib import Path

def test_probe_argv_exemption_only_nondense_verdict():
    """N=50 nondense verdict alone carries aggregate bounded_steps exemption."""
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import (
        BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_CONTRACT,
        EXEMPTED_PROBE_ARMS,
        probe_argv,
    )

    scratch = Path("/tmp/a_prime_probe_argv_shape_scratch")
    flag = "--phase-timeout-exemption-contract"
    arms = [
        ("dense_screen", 20, False),
        ("nondense_screen", 20, True),
        ("dense_verdict", 50, False),
        ("nondense_verdict", 50, True),
    ]
    for name, steps, nd in arms:
        argv = probe_argv(steps, scratch, nd, arm_name=name)
        has_flag = flag in argv
        if name in EXEMPTED_PROBE_ARMS:
            assert has_flag, name
            assert argv[argv.index(flag) + 1] == (
                BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_CONTRACT
            )
        else:
            assert not has_flag, name
        # uniform nested-phase budget remains 120 under exemption (not a scalar bump)
        assert argv[argv.index("--phase-timeout-seconds") + 1] == "120"
        assert argv[argv.index("--max-silent-phase-seconds") + 1] == "600"
        assert argv[argv.index("--total-timeout-seconds") + 1] == "3600"
    assert EXEMPTED_PROBE_ARMS == frozenset({"nondense_verdict"})


def test_probe_argv_without_arm_name_has_no_exemption():
    """Legacy/default call shape must not silently widen any arm."""
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import probe_argv

    argv = probe_argv(50, Path("/tmp/x"), True)
    assert "--phase-timeout-exemption-contract" not in argv


def test_probe_argv_exempt_arm_geometry_mismatch_raises():
    """Name-only admission closed: wrong steps/nondense under exempt arm → ValueError."""
    import pytest
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import probe_argv

    scratch = Path("/tmp/a_prime_probe_argv_shape_scratch")
    arm = "nondense_verdict"
    # steps=20 (screen geometry) under exempt name
    with pytest.raises(ValueError, match="geometry mismatch"):
        probe_argv(20, scratch, True, arm_name=arm)
    # nondense=False under exempt name
    with pytest.raises(ValueError, match="geometry mismatch"):
        probe_argv(50, scratch, False, arm_name=arm)
    # both wrong
    with pytest.raises(ValueError, match="geometry mismatch"):
        probe_argv(20, scratch, False, arm_name=arm)
