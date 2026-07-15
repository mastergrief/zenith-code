"""Phase-1 adapter CPU suite. OLD→NEW: fold4_and_partial→fold4_post+u_e_raise;
18_arg_surface→18_bind+no_attrs+object_new+provenance; receipt_matrix→
dup+wrong+miss+zero; probe_reload→probe_override+facade_reload."""
from __future__ import annotations

import importlib
import inspect
import json
import pickle
import sys
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack import forgotten_accum_a_ledger_accounting_v2 as acct
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    make_success_apply_event,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
    EFFECTIVE_STAMP_KEY,
    RunnerContractRefuse,
    build_forgotten_accum_runner_contract,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ark_invoke import (
    invoke_arm_with_a_rk,
)
import calm.hrm_text_158.native_full_stack.forgotten_accum_a_ledger_accounting_v2_ark_adapter as adapter
from calm.hrm_text_158.native_full_stack.forgotten_accum_a_ledger_accounting_v2_ark_adapter import (
    AdapterAuthorityRefuse,
    FourArmARkCallInputs,
    run_closed_four_arm_accounting_v2_cpu,
)

REPO = Path(__file__).resolve().parents[3]
_NS = "calm/hrm_text_158/native_full_stack"
ADAPTER = REPO / f"{_NS}/forgotten_accum_a_ledger_accounting_v2_ark_adapter.py"
FACADE = REPO / f"{_NS}/forgotten_accum_a_ledger_accounting_v2.py"
CORE = REPO / _NS / ("forgotten_accum_a_ledger_accounting_v2" + "_core.py")
ARK = REPO / f"{_NS}/forgotten_accum_training_equivalence_ark_invoke.py"
DRIVER = REPO / f"{_NS}/forgotten_accum_training_equivalence_science_driver.py"
STEP_A = REPO / (
    "calm/llm_computer/tests/"
    "test_forgotten_accum_a_ledger_accounting_v2_ordered_event_consumer_v0.py"
)
EXTRACT = REPO / "calm/llm_computer/tests" / (
    "test_forgotten_accum_a_ledger_accounting_v2" + "_core_extraction_v0.py"
)
PINS = {
    FACADE: "2c8a76cb17a932abf7b21dc6af9d86901cfcd0fe34a3b7cdb2ee0713c6a97fab",
    CORE: "dceba582104f5bde6cb770802d0bcd80f7d416db1e9e856e5b773725cd5dfc5b",
    ARK: "3f6c71c56c79af3a6724ae57e2ba70e6e439c12d51a530f5b545a520a6d63e20",
    DRIVER: "b9ef6496532a60c882052751aff11608fc41c7f7f90bd66f82621d3836a92609",
    STEP_A: "4ca9a2f068bd9c697584dc20b29603d02ec0b1d450b1465426e84a65a18362de",
    EXTRACT: "0a686bfd40f91f1d29537989d1433aaec66c5ca5a4521badaa8df64dd28770e6",
}
_SEAM = "_forgotten_accum_acct_v2_phase1_probe_seam_box"
_FORBIDDEN = {
    "probe", "register", "pop", "clear", "mint", "_build_run_closed",
    "scope_stack", "_stack", "clear_top",
}


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loc(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _eff(horizon: int) -> dict:
    c = build_forgotten_accum_runner_contract(runway_steps=horizon)
    return {EFFECTIVE_STAMP_KEY: {
        **c.as_pins_dict(), "global_cap_resolved_spec_present": True,
        "within_arm_consistent": True,
    }}


def _nine(horizon: int):
    return ({}, _eff(horizon), {}, {}, "ok", 0, None, None, [])


def _appending_runner(*, raise_after_arm: str | None = None, call_count: list | None = None,
                      on_arm=None):
    def runner(*args, **kwargs):
        if call_count is not None:
            call_count.append(1)
        log = kwargs.get("ordered_apply_event_log")
        arm_id = str(kwargs.get("ordered_apply_event_arm_id"))
        steps = int(kwargs["steps"])
        start = int(kwargs.get("start_step", 1))
        if log is not None:
            for i in range(steps):
                log.append(make_success_apply_event(
                    seq=i, arm_id=arm_id, optimizer_step_id=start + i,
                    q_changed_count=1, tensor_state_key_count=1,
                ))
        if on_arm is not None:
            on_arm(arm_id, kwargs)
        if raise_after_arm is not None and arm_id == raise_after_arm:
            return ({}, {EFFECTIVE_STAMP_KEY: {"within_arm_consistent": False}},
                    {}, {}, "bad", 0, None, None, [])
        return _nine(int(kwargs["global_horizon"]))
    return runner


def _calls(runner, *, horizon: int = 4, Inputs=FourArmARkCallInputs) -> FourArmARkCallInputs:
    contract = build_forgotten_accum_runner_contract(runway_steps=horizon)
    arms = ("U", "E", "R0", "RW")
    return Inputs(
        runner=runner, model=object(), batch={}, device="cpu", eligible={},
        runner_contract=contract, rk=contract.as_runner_kwargs(),
        states_by_arm={a: {} for a in arms}, hook_by_arm={a: None for a in arms},
        backlog_by_arm={a: None for a in arms}, flip_by_arm={a: False for a in arms},
        schedule_by_arm={a: None for a in arms},
    )


def _valid(out):
    assert out["state"] == "VERIFIED_VALID"
    assert out["reason"] == "FOUR_ARM_CONJUNCTION_OK"


def _run(**kw):
    return run_closed_four_arm_accounting_v2_cpu(
        t_cut=2, runway_steps=4, rewarm_window_steps=1, **kw,
    )


def _patch_ark(monkeypatch, fn):
    monkeypatch.setattr(adapter, "invoke_arm_with_a_rk", fn)


def test_happy_path_241_verified_valid_data_only():
    out = _run(calls=_calls(_appending_runner()))
    _valid(out)
    assert out["claimable"] is False and out["runtime_proven"] is False
    assert out["bankable"] is False and out["forensic_only"] is True
    assert set(out["details"]["arm_results"]) == {"U", "E", "R0", "RW"}


def test_exact_envelope_provenance_and_attachment(monkeypatch):
    captured = {}
    real = acct.classify_four_arm_ordered_event_summaries

    def wrap(*, geometry, arm_summary_payloads, trusted_capabilities):
        captured["payloads"] = {k: dict(v) for k, v in arm_summary_payloads.items()}
        captured["caps"] = {k: type(v).__name__ for k, v in trusted_capabilities.items()}
        return real(
            geometry=geometry, arm_summary_payloads=arm_summary_payloads,
            trusted_capabilities=trusted_capabilities,
        )

    monkeypatch.setattr(acct, "classify_four_arm_ordered_event_summaries", wrap)
    _run(calls=_calls(_appending_runner()))
    for arm, env in captured["payloads"].items():
        assert set(env) == {acct.ATTACHMENT_KEY, "source_provenance"}
        assert env[acct.ATTACHMENT_KEY]["arm_id"] == arm
        assert env["source_provenance"] == dict(acct.REQUIRED_SOURCE_PROVENANCE)
        assert captured["caps"][arm] == "TrustedNormalSuccessCapability"


def test_unadmitted_and_malformed_refuse_before_ark():
    n: list = []
    with pytest.raises(AdapterAuthorityRefuse, match="GEOMETRY_REFUSED"):
        run_closed_four_arm_accounting_v2_cpu(
            t_cut=2, runway_steps=4, rewarm_window_steps=2,
            calls=_calls(_appending_runner(call_count=n)),
        )
    assert n == []
    with pytest.raises(AdapterAuthorityRefuse, match="GEOMETRY_REFUSED"):
        run_closed_four_arm_accounting_v2_cpu(
            t_cut="x", runway_steps=4, rewarm_window_steps=1,
            calls=_calls(_appending_runner(call_count=n)),
        )
    assert n == []
    contract = build_forgotten_accum_runner_contract(runway_steps=4)
    with pytest.raises(AdapterAuthorityRefuse, match="KEYSET"):
        run_closed_four_arm_accounting_v2_cpu(
            t_cut=2, runway_steps=4, rewarm_window_steps=1,
            calls=FourArmARkCallInputs(
                runner=_appending_runner(call_count=n), model=object(), batch={},
                device="cpu", eligible={}, runner_contract=contract,
                rk=contract.as_runner_kwargs(), states_by_arm={"U": {}, "E": {}},
                hook_by_arm={"U": None, "E": None, "R0": None, "RW": None},
                backlog_by_arm={"U": None, "E": None, "R0": None, "RW": None},
                flip_by_arm={"U": False, "E": False, "R0": False, "RW": False},
                schedule_by_arm={"U": None, "E": None, "R0": None, "RW": None},
            ),
        )
    assert n == []


def test_fold4_post_attachment_raise_zero_authority():
    with pytest.raises(RunnerContractRefuse):
        _run(calls=_calls(_appending_runner(raise_after_arm="U")))
    assert acct._capability_ok(
        object.__new__(acct.TrustedNormalSuccessCapability), arm_id="U",
    ) is False


def test_u_normal_e_raise_no_partial_result():
    with pytest.raises(RunnerContractRefuse):
        _run(calls=_calls(_appending_runner(raise_after_arm="E")))
    assert acct._capability_ok(
        object.__new__(acct.TrustedNormalSuccessCapability), arm_id="E",
    ) is False


def test_exact_18_arg_bind():
    sig = inspect.signature(invoke_arm_with_a_rk)
    assert len(sig.parameters) == 18
    g = acct.build_independent_expected_geometry(
        t_cut=2, runway_steps=4, rewarm_window_steps=1,
    )
    calls = _calls(_appending_runner())
    kwargs = dict(
        runner=calls.runner, model=calls.model, batch=calls.batch,
        states=calls.states_by_arm["U"], eligible=calls.eligible,
        device=calls.device, steps=int(g.arms["U"].steps),
        start_step=int(g.arms["U"].start_step), global_horizon=int(g.runway_steps),
        hook=None, backlog=None, flip=False, schedule=None, rk=dict(calls.rk),
        arm="U", log=[], runner_contract=calls.runner_contract, a_rk_receipts=[],
    )
    assert len(sig.bind(**kwargs).arguments) == 18


def test_no_module_authority_attrs_or_builder():
    assert not (_FORBIDDEN & (set(dir(adapter)) | set(adapter.__dict__)))
    assert "del _build_run_closed" in ADAPTER.read_text(encoding="utf-8")


def test_object_new_capability_outside_scope_untrusted():
    assert acct._capability_ok(
        object.__new__(acct.TrustedNormalSuccessCapability), arm_id="U",
    ) is False


def test_caller_provenance_injection_rejected():
    with pytest.raises(TypeError):
        FourArmARkCallInputs(  # type: ignore[call-arg]
            runner=_appending_runner(), model=object(), batch={}, device="cpu",
            eligible={}, runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
            rk={}, states_by_arm={}, hook_by_arm={}, backlog_by_arm={},
            flip_by_arm={}, schedule_by_arm={}, source_provenance={"forged": True},
        )


def test_deepcopy_json_pickle_no_capability_leak():
    out = _run(calls=_calls(_appending_runner()))
    assert "TrustedNormalSuccess" not in json.dumps(out)
    assert pickle.loads(pickle.dumps(out))["state"] == "VERIFIED_VALID"
    assert deepcopy(out)["state"] == "VERIFIED_VALID"


def test_nested_top_scope_preserves_outer(monkeypatch):
    depths: list[int] = []
    real = acct.classify_four_arm_ordered_event_summaries
    once = {"done": False}
    outer_probe_ids: list[int] = []

    def wrap(*, geometry, arm_summary_payloads, trusted_capabilities):
        if not once["done"]:
            once["done"] = True
            outer_probe_ids.append(id(acct._capability_ok))
            inner = _run(calls=_calls(_appending_runner()))
            depths.append(1)
            _valid(inner)
            assert id(acct._capability_ok) == outer_probe_ids[0]
        return real(
            geometry=geometry, arm_summary_payloads=arm_summary_payloads,
            trusted_capabilities=trusted_capabilities,
        )

    monkeypatch.setattr(acct, "classify_four_arm_ordered_event_summaries", wrap)
    _valid(_run(calls=_calls(_appending_runner())))
    assert depths == [1]


def test_duplicate_receipt_cardinality_refuse(monkeypatch):
    real = invoke_arm_with_a_rk

    def dup(*a, **k):
        real(*a, **k)
        k["a_rk_receipts"].append({"arm": k["arm"], acct.ATTACHMENT_KEY: {}})

    _patch_ark(monkeypatch, dup)
    with pytest.raises(AdapterAuthorityRefuse, match="CARDINALITY"):
        _run(calls=_calls(_appending_runner()))


def test_wrong_arm_receipt_refuse(monkeypatch):
    real = invoke_arm_with_a_rk

    def wrong(*a, **k):
        real(*a, **k)
        k["a_rk_receipts"][0]["arm"] = "X"

    _patch_ark(monkeypatch, wrong)
    with pytest.raises(AdapterAuthorityRefuse, match="WRONG_ARM"):
        _run(calls=_calls(_appending_runner()))


def test_missing_attachment_receipt_refuse(monkeypatch):
    real = invoke_arm_with_a_rk

    def miss(*a, **k):
        real(*a, **k)
        del k["a_rk_receipts"][0][acct.ATTACHMENT_KEY]

    _patch_ark(monkeypatch, miss)
    with pytest.raises(AdapterAuthorityRefuse, match="ATTACHMENT"):
        _run(calls=_calls(_appending_runner()))


def test_zero_receipt_cardinality_refuse(monkeypatch):
    real = invoke_arm_with_a_rk

    def zero(*a, **k):
        real(*a, **k)
        k["a_rk_receipts"].clear()

    _patch_ark(monkeypatch, zero)
    with pytest.raises(AdapterAuthorityRefuse, match="CARDINALITY"):
        _run(calls=_calls(_appending_runner()))


def test_unconditional_probe_override(monkeypatch):
    monkeypatch.setattr(acct, "_capability_ok", lambda *a, **k: False)
    _valid(_run(calls=_calls(_appending_runner())))
    assert acct._capability_ok(
        object.__new__(acct.TrustedNormalSuccessCapability), arm_id="U",
    ) is False


def test_facade_reload_recovery():
    importlib.reload(acct)
    _valid(_run(calls=_calls(_appending_runner())))


def test_two_thread_barrier_both_verified_valid():
    a_at_rw, b_at_u = threading.Event(), threading.Event()
    a_go, b_go, a_done = threading.Event(), threading.Event(), threading.Event()
    out: dict = {}
    timeline: list = []
    classify_guard = threading.Lock()

    def a_on(arm, _kwargs):
        timeline.append(("ark", "A", arm))
        if arm == "RW":
            a_at_rw.set()
            assert a_go.wait(30)

    def b_on(arm, _kwargs):
        timeline.append(("ark", "B", arm))
        if arm == "U":
            b_at_u.set()
            assert b_go.wait(30)

    real = acct.classify_four_arm_ordered_event_summaries

    def wrap(*, geometry, arm_summary_payloads, trusted_capabilities):
        assert classify_guard.acquire(blocking=False), "classify overlap"
        try:
            timeline.append(("classify", threading.get_ident()))
            return real(
                geometry=geometry, arm_summary_payloads=arm_summary_payloads,
                trusted_capabilities=trusted_capabilities,
            )
        finally:
            classify_guard.release()

    acct.classify_four_arm_ordered_event_summaries = wrap
    try:
        def thread_a():
            out["A"] = _run(calls=_calls(_appending_runner(on_arm=a_on)))
            a_done.set()

        def thread_b():
            out["B"] = _run(calls=_calls(_appending_runner(on_arm=b_on)))

        ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        assert a_at_rw.wait(30) and b_at_u.wait(30)
        a_go.set()
        assert a_done.wait(60)
        b_go.set()
        ta.join(60)
        tb.join(60)
    finally:
        acct.classify_four_arm_ordered_event_summaries = real
    _valid(out["A"])
    _valid(out["B"])
    assert not ta.is_alive() and not tb.is_alive()
    assert any(e[0] == "ark" for e in timeline)
    assert sum(1 for e in timeline if e[0] == "classify") == 2
    assert ("ark", "A", "RW") in timeline and ("ark", "B", "U") in timeline


def test_stale_vs_reloaded_concurrent_overlap_same_lock():
    stale = run_closed_four_arm_accounting_v2_cpu
    lock_before = sys.modules[_SEAM]["lock"]
    fresh_mod = importlib.reload(adapter)
    fresh = fresh_mod.run_closed_four_arm_accounting_v2_cpu
    assert lock_before is sys.modules[_SEAM]["lock"]
    assert set(sys.modules[_SEAM]) == {"lock"}
    Inputs = fresh_mod.FourArmARkCallInputs
    a_at_rw, b_at_u = threading.Event(), threading.Event()
    a_go, b_go, a_done = threading.Event(), threading.Event(), threading.Event()
    out: dict = {}

    def a_on(arm, _k):
        if arm == "RW":
            a_at_rw.set()
            assert a_go.wait(30)

    def b_on(arm, _k):
        if arm == "U":
            b_at_u.set()
            assert b_go.wait(30)

    def thread_a():
        out["A"] = stale(
            t_cut=2, runway_steps=4, rewarm_window_steps=1,
            calls=_calls(_appending_runner(on_arm=a_on), Inputs=Inputs),
        )
        a_done.set()

    def thread_b():
        out["B"] = fresh(
            t_cut=2, runway_steps=4, rewarm_window_steps=1,
            calls=_calls(_appending_runner(on_arm=b_on), Inputs=Inputs),
        )

    ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    assert a_at_rw.wait(30) and b_at_u.wait(30)
    a_go.set()
    assert a_done.wait(60)
    b_go.set()
    ta.join(60)
    tb.join(60)
    _valid(out["A"])
    _valid(out["B"])


def test_classify_exception_restores_callable_and_releases_lock(monkeypatch):
    prior = acct._capability_ok
    boom = RuntimeError("classify_boom_unique")
    real_cls = acct.classify_four_arm_ordered_event_summaries
    monkeypatch.setattr(
        acct, "classify_four_arm_ordered_event_summaries",
        lambda **_k: (_ for _ in ()).throw(boom),
    )
    with pytest.raises(RuntimeError) as ei:
        _run(calls=_calls(_appending_runner()))
    assert ei.value is boom
    assert acct._capability_ok is prior
    monkeypatch.setattr(acct, "classify_four_arm_ordered_event_summaries", real_cls)
    _valid(_run(calls=_calls(_appending_runner())))


def test_pins_and_budgets():
    for path, digest in PINS.items():
        assert _sha(path) == digest, path.name
    assert _sha(ADAPTER) == (
        "3f5c593bb0be38d59a17ed7c31f35de47cf74f4cbb45906802f70a7fe744b90f"
    )
    assert _loc(ADAPTER) <= 220
    assert _loc(Path(__file__)) <= 480
    assert max(len(l) for l in ADAPTER.read_text(encoding="utf-8").splitlines()) <= 119
    src = ADAPTER.read_text(encoding="utf-8")
    assert "setdefault" in src and _SEAM in src
