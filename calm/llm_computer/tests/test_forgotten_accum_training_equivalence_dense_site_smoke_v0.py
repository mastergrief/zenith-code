"""Dense-site smoke CPU paths: non-vacuous ordinary/deferred twin (no GPU)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    LIVE_ACC_CARRIER_NONE,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    GLOBAL_CAP_CONTRACT,
    SMOKE_CPU_PREDICATES,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
    EXIT_EVENT_CODED_STOP,
    EXIT_INCONCLUSIVE_NO_CROSSING,
    EXIT_NO_AUTHORITY,
    EXIT_PASS,
    EXIT_PREDICATE_FAIL,
    RECEIPT_SCHEMA,
    TwinArmObservation,
    build_guaranteed_crossing_inputs,
    classify_ordinary_deferred_twin,
    classify_smoke_exit,
    cpu_checkable_predicate_eval,
    is_event_coded_carrier,
    resolve_carrier_for_smoke,
    run_ordinary_deferred_twin,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
    SmokeReceipt,
)

REPO = Path(__file__).resolve().parents[3]
RUN_PY = REPO / "scripts/forgotten_accum_training_equivalence_run.py"


def _arm(
    *,
    deferred: bool,
    applied: int,
    q_changed: int,
    acc_changed: bool,
    q_changed_hash: bool,
    demand: int,
    backlog_changed: bool = False,
) -> TwinArmObservation:
    return TwinArmObservation(
        deferred=deferred,
        q_hash_pre="q0",
        q_hash_post="q1" if q_changed_hash else "q0",
        acc_hash_pre="a0",
        acc_hash_post="a1" if acc_changed else "a0",
        backlog_sha_pre="b0",
        backlog_sha_post="b1" if backlog_changed else "b0",
        backlog_cardinality_pre=0,
        backlog_cardinality_post=1 if backlog_changed else 0,
        applied_count=applied,
        q_changed_count=q_changed,
        residual_writeback_count=0 if deferred else applied,
        crossing_demand=demand,
        cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=deferred,
    )


def test_cpu_smoke_predicate_catalog():
    assert len(SMOKE_CPU_PREDICATES) >= 4
    assert DENSE_LEGACY_CAP_SITE_ID
    assert GLOBAL_CAP_CONTRACT == "c1_banked_faithful_long_run_global_cap"
    assert RECEIPT_SCHEMA.endswith("/v2")


def test_legacy_vacuous_zeros_alone_insufficient():
    fails = cpu_checkable_predicate_eval(
        carrier_selector="none",
        cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=True,
        during_w_q_changed=0,
        during_w_applied=0,
    )
    assert "legacy_vacuous_deferred_zeros_insufficient_without_twin" in fails


def test_cpu_checkable_with_twin_pass():
    twin = run_ordinary_deferred_twin(device="cpu")
    assert twin.smoke_class == "PASS"
    fails = cpu_checkable_predicate_eval(
        carrier_selector="none",
        cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        twin=twin,
    )
    assert fails == []


def test_cpu_checkable_smoke_predicates_fail_on_event_coded():
    twin = run_ordinary_deferred_twin(device="cpu")
    fails = cpu_checkable_predicate_eval(
        carrier_selector="LIVE_ACC_CARRIER_V4_LIVE",
        cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        twin=twin,
    )
    assert "carrier_must_be_dense_legacy_not_event_coded" in fails
    assert is_event_coded_carrier("LIVE_ACC_CARRIER_V4_LIVE")


def test_guaranteed_crossing_twin_pass_on_cpu():
    twin = run_ordinary_deferred_twin(device="cpu")
    assert twin.smoke_class == "PASS"
    assert twin.ordinary.crossing_demand > 0
    assert twin.ordinary.applied_count > 0
    assert twin.ordinary.q_hash_pre != twin.ordinary.q_hash_post
    assert twin.deferred.applied_count == 0
    assert twin.deferred.q_changed_count == 0
    assert twin.deferred.q_hash_pre == twin.deferred.q_hash_post
    assert twin.deferred.acc_hash_pre != twin.deferred.acc_hash_post
    assert twin.deferred.backlog_sha_pre == twin.deferred.backlog_sha_post
    assert twin.deferred.cap_site_branch == DENSE_LEGACY_CAP_SITE_ID
    assert twin.deferred.flip_application_deferred is True


def test_classify_inconclusive_when_ordinary_no_crossing():
    ordinary = _arm(
        deferred=False,
        applied=0,
        q_changed=0,
        acc_changed=False,
        q_changed_hash=False,
        demand=0,
    )
    deferred = _arm(
        deferred=True,
        applied=0,
        q_changed=0,
        acc_changed=True,
        q_changed_hash=False,
        demand=0,
    )
    result = classify_ordinary_deferred_twin(ordinary, deferred)
    assert result.smoke_class == "INCONCLUSIVE_NO_CROSSING"
    receipt = SmokeReceipt(
        pass_fail="INCONCLUSIVE",
        smoke_class=result.smoke_class,
        failures=result.failures,
        carrier_selector="none",
        forgotten_accum_cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=True,
        during_w_q_changed=0,
        during_w_applied=0,
        head="h",
        tree="t",
        parent_sha256="p",
        argv_digest="a",
        device="cpu",
        global_cap_contract=GLOBAL_CAP_CONTRACT,
        predicates=tuple(SMOKE_CPU_PREDICATES),
        steps_run=0,
        twin=result.as_dict(),
    )
    assert classify_smoke_exit(receipt) == EXIT_INCONCLUSIVE_NO_CROSSING


def test_classify_fail_when_deferred_freezes_carry():
    ordinary = _arm(
        deferred=False,
        applied=2,
        q_changed=2,
        acc_changed=True,
        q_changed_hash=True,
        demand=3,
    )
    deferred = _arm(
        deferred=True,
        applied=0,
        q_changed=0,
        acc_changed=False,  # faux freeze
        q_changed_hash=False,
        demand=3,
    )
    result = classify_ordinary_deferred_twin(ordinary, deferred)
    assert result.smoke_class == "FAIL"
    assert "deferred_carry_did_not_change" in result.failures
    receipt = SmokeReceipt(
        pass_fail="FAIL",
        smoke_class=result.smoke_class,
        failures=result.failures,
        carrier_selector="none",
        forgotten_accum_cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=True,
        during_w_q_changed=0,
        during_w_applied=0,
        head="h",
        tree="t",
        parent_sha256="p",
        argv_digest="a",
        device="cpu",
        global_cap_contract=GLOBAL_CAP_CONTRACT,
        predicates=tuple(SMOKE_CPU_PREDICATES),
        steps_run=0,
        twin=result.as_dict(),
    )
    assert classify_smoke_exit(receipt) == EXIT_PREDICATE_FAIL


def test_receipt_schema_v2_includes_twin_fields():
    twin = run_ordinary_deferred_twin(device="cpu")
    receipt = SmokeReceipt(
        pass_fail="PASS",
        smoke_class=twin.smoke_class,
        failures=(),
        carrier_selector="none",
        forgotten_accum_cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=True,
        during_w_q_changed=0,
        during_w_applied=0,
        head="abc",
        tree="def",
        parent_sha256="9b4e",
        argv_digest="00",
        device="cuda:0",
        global_cap_contract=GLOBAL_CAP_CONTRACT,
        predicates=tuple(SMOKE_CPU_PREDICATES),
        steps_run=1,
        twin=twin.as_dict(),
        live_runner_cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
    )
    blob = receipt.as_dict()
    assert blob["schema"] == RECEIPT_SCHEMA
    for key in (
        "pass_fail",
        "smoke_class",
        "twin",
        "carrier_selector",
        "forgotten_accum_cap_site_branch",
        "during_w_q_changed",
        "during_w_applied",
        "head",
        "tree",
        "parent_sha256",
        "argv_digest",
        "device",
        "live_runner_cap_site_branch",
    ):
        assert key in blob
    assert "ordinary" in blob["twin"] and "deferred" in blob["twin"]
    assert blob["twin"]["ordinary"]["applied_count"] > 0
    assert classify_smoke_exit(receipt) == EXIT_PASS

    bad = SmokeReceipt(
        pass_fail="FAIL",
        smoke_class="FAIL",
        failures=("carrier_must_be_dense_legacy_not_event_coded",),
        carrier_selector="LIVE_ACC_CARRIER_V4_LIVE",
        forgotten_accum_cap_site_branch=DENSE_LEGACY_CAP_SITE_ID,
        flip_application_deferred=True,
        during_w_q_changed=0,
        during_w_applied=0,
        head="a",
        tree="b",
        parent_sha256="c",
        argv_digest="d",
        device="cuda:0",
        global_cap_contract=GLOBAL_CAP_CONTRACT,
        predicates=tuple(SMOKE_CPU_PREDICATES),
        steps_run=0,
        twin={},
    )
    assert classify_smoke_exit(bad) == EXIT_EVENT_CODED_STOP


def test_cli_refuses_without_gpu_authority():
    proc = subprocess.run(
        [sys.executable, str(RUN_PY), "smoke-dense-site", "--device", "cuda:0"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == EXIT_NO_AUTHORITY
    assert "REFUSED" in proc.stderr


def test_cli_smoke_predicates_subcommand():
    proc = subprocess.run(
        [sys.executable, str(RUN_PY), "smoke-predicates"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["DENSE_LEGACY_CAP_SITE_ID"] == DENSE_LEGACY_CAP_SITE_ID


def test_default_carrier_resolver_is_none_when_flags_off():
    assert resolve_carrier_for_smoke(force_event_coded=False) == LIVE_ACC_CARRIER_NONE
    assert resolve_carrier_for_smoke(force_event_coded=True) == "LIVE_ACC_CARRIER_V4_LIVE"


def test_clone_inputs_are_not_aliased():
    base = build_guaranteed_crossing_inputs(device="cpu")
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
        clone_cap_inputs,
    )

    a = clone_cap_inputs(base)
    b = clone_cap_inputs(base)
    assert a[0].state.q_levels.data_ptr() != b[0].state.q_levels.data_ptr()
    a[0].state.q_levels[0] = 1
    assert int(b[0].state.q_levels[0].item()) == 0


@pytest.mark.skip(reason="GPU dense-site smoke requires separate claude/test-operator authority")
def test_gpu_dense_site_smoke_NOT_RUN_IN_CPU_GATE():
    raise AssertionError("unreachable in CPU gate")
