"""CPU-static tests for support-only characterization (D2c9 correction c5)."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import pytest, torch
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
    ARM_FORK_NAMES, AuthoritativeGpuHooks,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    AGG_FLOOR, MAX_COMPACT_TELEMETRY_BYTES, SKEW_MAX, assert_compact_json_nbytes,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_partition_leakage import (
    compute_partition_leakage_compact,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    FORMAL_SOURCE_PIN_BASENAMES, PinValidationError, WATCH_WRAP_HRM158_SHA256, rehash_path,
)
from calm.hrm_text_158.native_full_stack import signed_utility_fixed_state_support_only as so
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_support_only import (
    SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE, SUPPORT_DEGENERATE_BELOW_FLOOR, SUPPORT_ELIGIBLE,
    SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, TERMINAL_TAXONOMY, require_exact_40hex_commit,
    run_support_only_characterization,
)
from calm.llm_computer.tests import hrm_text_158_signed_utility_support_only_characterization as cli
REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
STACK = REPO / "calm/hrm_text_158/native_full_stack"
CLI = REPO / "calm/llm_computer/tests/hrm_text_158_signed_utility_support_only_characterization.py"
WATCH, MOD, VOTE = REPO / "bin/watch-wrap", STACK / "signed_utility_fixed_state_support_only.py", STACK / "vote_update.py"
FACADE = STACK / "signed_utility_fixed_state_creditdir_import_facade.py"
_IMM = ("parent_sha_pre", "parent_sha_post", "source_sha_pre", "source_sha_post",
        "launch_surface_sha_pre", "launch_surface_sha_post")
@dataclass
class _Plan:
    applied_indices: torch.Tensor; applied_directions: torch.Tensor; replay_veto_directions: torch.Tensor
    applied_thresholds: torch.Tensor; candidate_indices: torch.Tensor; pre_veto_selected_indices: torch.Tensor
    replay_ce_veto_indices: torch.Tensor; replay_veto_thresholds: torch.Tensor
    pc_aux_negative_indices: torch.Tensor; pc_aux_veto_indices: torch.Tensor
    q_i16: torch.Tensor; new_acc_i32: torch.Tensor
def _plan(*, idxs, dirs=None, q=None, new_acc=0, thr=10, qn=None):
    n = len(idxs); z = torch.tensor(idxs, dtype=torch.int64); empty = torch.zeros(0, dtype=torch.int64)
    d = torch.tensor(dirs if dirs is not None else [1] * n, dtype=torch.int16)
    qq = torch.tensor(q if q is not None else [0] * (qn or max(4, max(idxs) + 1 if idxs else 4)), dtype=torch.int16)
    return _Plan(applied_indices=z, applied_directions=d, replay_veto_directions=torch.zeros(0, dtype=torch.int16),
                 applied_thresholds=torch.full((n,), thr, dtype=torch.int32), candidate_indices=z.clone(),
                 pre_veto_selected_indices=z.clone(), replay_ce_veto_indices=empty.clone(),
                 replay_veto_thresholds=torch.zeros(0, dtype=torch.int32), pc_aux_negative_indices=empty.clone(),
                 pc_aux_veto_indices=empty.clone(), q_i16=qq, new_acc_i32=torch.zeros_like(qq, dtype=torch.int32))
def _clone(st):
    return SimpleNamespace(q_levels=st.q_levels.clone(), exact_accumulator_shadow=st.exact_accumulator_shadow.clone(),
                           frozen_scale=st.frozen_scale.clone(), state_key=st.state_key)
def _states(spec=None):
    spec = spec or {"k0": (0, 0, 0, 0)}; out = {}
    for k, q in spec.items():
        t = torch.tensor(q, dtype=torch.int8)
        out[k] = SimpleNamespace(q_levels=t, exact_accumulator_shadow=torch.zeros_like(t, dtype=torch.int16),
                                 frozen_scale=torch.tensor(1.0), state_key=k)
    return out
def build_support_only_test_packet(*, expected_head: str, **over):
    pins = {n: {"absolute_path": str(STACK / n), "sha256": rehash_path(STACK / n)} for n in FORMAL_SOURCE_PIN_BASENAMES}
    p = {"pin_mode": "cpu_static_di", "device": "cpu", "repo_root": str(REPO), "expected_head": expected_head,
         "parent_checkpoint": {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)}, "source_pins": pins,
         "cli_pin": {"absolute_path": str(CLI), "sha256": rehash_path(CLI)},
         "watch_wrap_pin": {"absolute_path": str(WATCH), "sha256": rehash_path(WATCH)}}
    p.update(over); return p
def _batch(n, start):
    return {"batch": {"x": torch.zeros(n, 1)}, "metadata": {
        "row_ids": [f"r{start+i}" for i in range(n)], "prompts": [f"p{start+i}" for i in range(n)],
        "targets": [f"t{start+i}" for i in range(n)], "response_tokens": [[start+i] for i in range(n)]}}
def _hooks(spies, *, plan_factory=None, qspec=None, leakage=None, capture_wrap=None):
    qspec = qspec or {"k0": (0, 0, 0, 0)}
    plan_factory = plan_factory or (lambda: {"k0": _plan(idxs=[0, 1, 2, 3])})
    def materialize(_p):
        spies["m"] = spies.get("m", 0) + 1; return SimpleNamespace(tensor_states=_states(qspec), eligible_modules={}, model=None)
    def rebuild(_b):
        spies["r"] = spies.get("r", 0) + 1; return [_batch(32, 0), _batch(32, 100), _batch(26, 200)]
    def leak_fn(b):
        spies["l"] = spies.get("l", 0) + 1
        return leakage if leakage is not None else compute_partition_leakage_compact(b)
    def fork(_b):
        spies["f"] = spies.get("f", 0) + 1
        return {k: {sk: _clone(sv) for sk, sv in _states(qspec).items()} for k in ARM_FORK_NAMES}
    def capture(_b, arms):
        spies["c"] = spies.get("c", 0) + 1; out = plan_factory(), {}, 1
        return capture_wrap(out) if capture_wrap else out
    def boom(*_a, **_k):
        spies["forbidden"] = spies.get("forbidden", 0) + 1; raise AssertionError("forbidden_hook_called")
    return AuthoritativeGpuHooks(materialize=materialize, rebuild_support_batches=rebuild, leakage_report=leak_fn,
                                 fork_arm_states=fork, capture_plans=capture, public_apply=boom, invert_plans=boom,
                                 eval_arm_nll=boom, phase_budgets={})
def _assert_imm(out):
    for k in _IMM: assert k in out
    assert out.get("claim_ceiling") == "support_eligibility_only" and out.get("estimand")
def _agg_hooks(zeros_k0, zeros_k1):
    def q(z): return tuple([0] * z + [1] * (10 - z))
    qspec = {"k0": q(zeros_k0), "k1": q(zeros_k1)}
    plans = {k: _plan(idxs=list(range(10)), q=list(qv)) for k, qv in qspec.items()}
    return _hooks({}, qspec=qspec, plan_factory=lambda: plans)
def test_taxonomy_hook_guard_cli_bootstrap_session(monkeypatch, tmp_path: Path):
    assert sum(1 for _ in MOD.open()) <= 246 and sum(1 for _ in CLI.open()) <= 200
    assert sum(1 for _ in Path(__file__).open()) <= 420 and "from calm" not in CLI.read_text().split("def main")[0]
    assert list(TERMINAL_TAXONOMY)[:1] == [SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE]
    out = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}))
    assert out["classifier"] == SUPPORT_ELIGIBLE; _assert_imm(out)
    monkeypatch.setattr(so, "build_live_hooks", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("hook_boom")))
    guarded = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40))
    assert guarded["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE and "hook_boom" in str(guarded.get("reason", ""))
    fac = cli._load_verified_facade(FACADE, rehash_path(FACADE)); assert Path(fac.__file__).resolve() == FACADE.resolve()
    spies = {}
    def _fake_run(p, **_k):
        spies["pkt"] = p
        return {"classifier": SUPPORT_ELIGIBLE, "route": ["via_session"], "estimand": cli.ESTIMAND,
                "claim_ceiling": "support_eligibility_only", **{k: (None if "parent" in k else {}) for k in _IMM}}
    so_ns = SimpleNamespace(__file__=str(STACK / "signed_utility_fixed_state_support_only.py"), run_support_only_characterization=_fake_run)
    ag_ns = SimpleNamespace(__file__=str(STACK / "signed_utility_fixed_state_authoritative_gpu.py"))
    @contextmanager
    def fake_session(expected):
        spies["expected"] = dict(expected)
        yield SimpleNamespace(support_only=so_ns, authoritative_gpu=ag_ns,
                              verified_paths_by_module={"support_only": so_ns.__file__, "authoritative_gpu": ag_ns.__file__})
    real_load = cli._load_verified_facade
    monkeypatch.setattr(cli, "_load_verified_facade", lambda p, s: (lambda m: setattr(m, "signed_utility_fixed_state_session", fake_session) or m)(real_load(p, s)))
    pkt = build_support_only_test_packet(expected_head="0" * 40)
    routed = cli._run_via_verified_session(pkt, self_file=CLI)
    assert routed["classifier"] == SUPPORT_ELIGIBLE and spies.get("pkt") is pkt and len(spies["expected"]) == 13
    copied = tmp_path / "copied_cli.py"; shutil.copy2(CLI, copied)
    with pytest.raises(RuntimeError, match="cli_pin_not_executing_file"): cli._bind_launch_identity(pkt, self_file=copied)
    bad_fac = tmp_path / FACADE.name; shutil.copy2(FACADE, bad_fac)
    bad_pkt = build_support_only_test_packet(expected_head="0" * 40)
    bad_pkt["source_pins"][FACADE.name] = {"absolute_path": str(bad_fac), "sha256": rehash_path(bad_fac)}
    with pytest.raises(RuntimeError, match="creditdir_import_facade_pin_not_repo_root"):
        cli._bind_launch_identity(bad_pkt, self_file=CLI)
    bad_ww = tmp_path / "watch-wrap"; bad_ww.write_bytes(WATCH.read_bytes())
    ww_pkt = build_support_only_test_packet(expected_head="0" * 40)
    ww_pkt["watch_wrap_pin"] = {"absolute_path": str(bad_ww), "sha256": rehash_path(bad_ww)}
    with pytest.raises(RuntimeError, match="watch_wrap_pin_not_repo_root"): cli._bind_launch_identity(ww_pkt, self_file=CLI)
    assert cli._bind_launch_identity(pkt, self_file=CLI) == FACADE.resolve()
def test_floors_skew_aggregate_boundaries_and_asymmetric(monkeypatch):
    out = run_support_only_characterization(
        build_support_only_test_packet(expected_head="0" * 40),
        hooks=_hooks({}, plan_factory=lambda: {"k0": _plan(idxs=[])}))
    assert out["classifier"] == SUPPORT_DEGENERATE_BELOW_FLOOR
    assert out["characterization"]["support_floors"]["skew_defined"] is False
    qspec = {"k0": (0, 0, 1, 1), "k1": (0, 1, 1, 1)}
    plans = {"k0": _plan(idxs=[0, 1, 2, 3], q=[0, 0, 1, 1]), "k1": _plan(idxs=[0, 1, 2, 3], q=[0, 1, 1, 1])}
    ok = run_support_only_characterization(
        build_support_only_test_packet(expected_head="0" * 40),
        hooks=_hooks({}, qspec=qspec, plan_factory=lambda: plans))
    assert abs(ok["characterization"]["support_floors"]["skew_observed"] - SKEW_MAX) < 1e-12
    assert ok["classifier"] == SUPPORT_ELIGIBLE
    plans2 = {"k0": _plan(idxs=[0, 1, 2, 3], q=[0, 0, 0, 1]), "k1": _plan(idxs=[0, 1, 2, 3], q=[0, 1, 1, 1])}
    bad_skew = run_support_only_characterization(
        build_support_only_test_packet(expected_head="0" * 40),
        hooks=_hooks({}, qspec={"k0": (0, 0, 0, 1), "k1": (0, 1, 1, 1)}, plan_factory=lambda: plans2))
    assert bad_skew["classifier"] == SUPPORT_DEGENERATE_BELOW_FLOOR
    at = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_agg_hooks(4, 3))
    assert at["classifier"] == SUPPORT_ELIGIBLE and abs(at["characterization"]["aggregate_retained_fraction"] - AGG_FLOOR) < 1e-12
    bel = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_agg_hooks(3, 3))
    assert bel["classifier"] == SUPPORT_DEGENERATE_BELOW_FLOOR and abs(bel["characterization"]["aggregate_retained_fraction"] - 0.30) < 1e-12
    def huge(_a, _b):
        return {}, {"applied_indices": [1, 2, 3], "blob": "z" * (MAX_COMPACT_TELEMETRY_BYTES + 16)}
    monkeypatch.setattr(so, "characterize_plans_bidirectional_legal", huge)
    big = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}))
    assert big["classifier"] == SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE
    diag = big["characterization"]
    assert diag.get("characterization_invalid") and diag.get("sha256_unavailable") is False and "blob" not in diag
def test_pins_launch_source_drift_and_oexcl_before_session(monkeypatch, tmp_path: Path):
    with pytest.raises(PinValidationError):
        require_exact_40hex_commit(REPO, "a" * 40)
    formal = build_support_only_test_packet(expected_head="a" * 40); formal["pin_mode"] = "formal"
    assert run_support_only_characterization(formal)["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE
    cli_tmp = tmp_path / "cli_pin.py"; cli_tmp.write_text("# pin\n")
    pkt = build_support_only_test_packet(expected_head="0" * 40)
    pkt["cli_pin"] = {"absolute_path": str(cli_tmp), "sha256": rehash_path(cli_tmp)}
    drift = run_support_only_characterization(
        pkt, hooks=_hooks({}, plan_factory=lambda: {"k0": _plan(idxs=[])},
                          capture_wrap=lambda o: (cli_tmp.write_text("# pin mutated\n") or o)))
    assert drift["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE and "launch_surface_sha_drift" in str(drift.get("reason", ""))
    src_tmp = tmp_path / "signed_utility_fixed_state_schema.py"; src_tmp.write_text("# src pin\n")
    spkt = build_support_only_test_packet(expected_head="0" * 40)
    spkt["source_pins"]["signed_utility_fixed_state_schema.py"] = {"absolute_path": str(src_tmp), "sha256": rehash_path(src_tmp)}
    sdrift = run_support_only_characterization(
        spkt, hooks=_hooks({}, plan_factory=lambda: {"k0": _plan(idxs=[])},
                           capture_wrap=lambda o: (src_tmp.write_text("# src mutated\n") or o)))
    assert sdrift["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE and "source_sha_drift" in str(sdrift.get("reason", ""))
    bad_packet = tmp_path / "bad.json"; bad_packet.write_text("{")
    rc = cli.main(["--packet", str(bad_packet), "--receipt", str(tmp_path / "r.json")])
    assert rc == 2; term = json.loads((tmp_path / "r.json").read_text())
    assert term["estimand"] == cli.ESTIMAND and term["claim_ceiling"] == "support_eligibility_only"
    assert_compact_json_nbytes(term, limit=256 * 1024, label="support_terminal")
    assert len(FORMAL_SOURCE_PIN_BASENAMES) == 14 and WATCH_WRAP_HRM158_SHA256.startswith("a19f1c5f")
    copied = tmp_path / "copied_cli.py"; shutil.copy2(CLI, copied)
    pkt_path = tmp_path / "pkt.json"; pkt_path.write_text(json.dumps(build_support_only_test_packet(expected_head="0" * 40)))
    proc = subprocess.run([sys.executable, str(copied), "--packet", str(pkt_path), "--receipt", str(tmp_path / "copy_r.json")],
                          cwd=str(REPO), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "cli_pin_not_executing_file" in (tmp_path / "copy_r.json").read_text()
    # O_EXCL collision before session + write-all under short os.write
    pre = tmp_path / "collide.json"; pre.write_text("{}\n"); calls: list[int] = []
    monkeypatch.setattr(cli, "_run_via_verified_session",
                        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("session")))
    assert cli.main(["--packet", str(bad_packet), "--receipt", str(pre)]) == 2 and calls == [] and pre.read_text() == "{}\n"
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[:7] if len(data) > 7 else data))
    short = tmp_path / "short.json"; assert cli.main(["--packet", str(bad_packet), "--receipt", str(short)]) == 2
    parsed = json.loads(short.read_text()); assert parsed["estimand"] == cli.ESTIMAND and "packet_load" in parsed["reason"]
    with pytest.raises(RuntimeError, match="receipt_write_zero_progress"):
        monkeypatch.setattr(os, "write", lambda fd, data: 0); fd = cli._reserve_receipt(tmp_path / "z.json")
        try: cli._write_reserved(fd, cli._canonical_fail("x"))
        finally: cli._close_reserved(fd)
def test_illegal_and_below_floor():
    bad = _plan(idxs=[0], dirs=[2], q=[0, 0, 0, 0])
    out = run_support_only_characterization(
        build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}, plan_factory=lambda: {"k0": bad}))
    assert out["classifier"] == SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE; _assert_imm(out)
    out2 = run_support_only_characterization(
        build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}, plan_factory=lambda: {"k0": _plan(idxs=[])}))
    assert out2["classifier"] == SUPPORT_DEGENERATE_BELOW_FLOOR and "per_key" in out2["characterization"]
def test_write_reserved_short_writes_complete_json(monkeypatch, tmp_path: Path):
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[:7] if len(data) > 7 else data))
    fd = cli._reserve_receipt(tmp_path / "w.json")
    try: cli._write_reserved(fd, cli._canonical_fail("short_ok"))
    finally: cli._close_reserved(fd)
    assert json.loads((tmp_path / "w.json").read_text())["reason"] == "short_ok"
def test_progress_emit_contract(capsys, monkeypatch, tmp_path: Path):
    import time
    pkt = build_support_only_test_packet(expected_head="0" * 40); spies = {}
    a = run_support_only_characterization(pkt, hooks=_hooks(spies))
    assert json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        run_support_only_characterization(pkt, hooks=_hooks({}), progress_sink=None), sort_keys=True, allow_nan=False)
    assert spies.get("forbidden", 0) == 0
    order = "MOD_PARSE_PACKET_PINS MOD_BUILD_LIVE_HOOKS MOD_PARENT_SHA_PRE MOD_MATERIALIZE MOD_REBUILD_BATCHES MOD_LEAKAGE MOD_FORK_ARMS MOD_CAPTURE_PLANS MOD_CHARACTERIZE MOD_VALIDATE_CHARACTERIZATION MOD_ENFORCE_FLOORS MOD_EMIT_TERMINAL".split()
    ev = []; run_support_only_characterization(pkt, hooks=_hooks({}), progress_sink=lambda s, e, r=None: ev.append((s, e)))
    assert [s for s, e in ev if e == "start"] == order
    assert all(ev[next(i for i, x in enumerate(ev) if x == (st, "start")) + 1] == (st, "done") for st in order)
    be = []
    def boom(step, edge, reason=None):
        be.append((step, edge))
        if step == "MOD_MATERIALIZE" and edge == "start": raise RuntimeError("sink_boom")
    fail = run_support_only_characterization(pkt, hooks=_hooks({}), progress_sink=boom)
    assert fail["reason"] == "progress_sink_failure" and not any(s == "MOD_MATERIALIZE" and e in ("done", "error") for s, e in be)
    ps = cli._build_progress_sink(time.monotonic_ns()); ps("CLI_PACKET_LOAD", "start", None); ps("CLI_PACKET_LOAD", "done", None)
    ms = [int(l.split("elapsed_ms=")[1].split()[0]) for l in capsys.readouterr().out.splitlines() if l.startswith("SUPPORT_PROGRESS ")]
    assert ms == sorted(ms) and ms[0] >= 0
    for args, ex in [(("NOT_A_STEP", "start", None), ValueError), (("CLI_PACKET_LOAD", "nope", None), ValueError),
                     (("CLI_PACKET_LOAD", "error", "x" * 300), ValueError), (("CLI_PACKET_LOAD", "error", ["arr"]), TypeError),
                     (("CLI_PACKET_LOAD", "error", "has NaN inside"), ValueError), (("CLI_PACKET_LOAD", "error", "a\nb"), ValueError)]:
        with pytest.raises(ex): ps(*args)
    ee = []
    deg = run_support_only_characterization(pkt, hooks=_hooks({}, plan_factory=lambda: {"k0": _plan(idxs=[])}),
                                            progress_sink=lambda s, e, r=None: ee.append((s, e)))
    assert deg["classifier"] == SUPPORT_DEGENERATE_BELOW_FLOOR and ("MOD_ENFORCE_FLOORS", "error") in ee
    def arm(fs, fe):
        calls = []
        def raw(step, edge, reason=None):
            calls.append((step, edge))
            if step == fs and edge == fe: raise RuntimeError("SINK_FAIL")
        return cli._guarded_sink(raw), calls
    r1 = tmp_path / "a1.json"; s1, c1 = arm("CLI_PACKET_LOAD", "start")
    fd1 = cli._step(s1, "CLI_RECEIPT_RESERVE", lambda: cli._reserve_receipt(r1))
    with pytest.raises(cli.ProgressSinkFailure): cli._step(s1, "CLI_PACKET_LOAD", lambda: json.loads("{}"))
    assert cli._fail_closed_sink(fd1) == 2 and json.loads(r1.read_text())["reason"] == "progress_sink_failure"
    assert not any(st == "CLI_RECEIPT_WRITE" for st, _ in c1)
    ok = {**cli._canonical_fail("x"), "classifier": SUPPORT_ELIGIBLE}; del ok["reason"]
    for name, fe in (("a2", "done"), ("a3", "start")):
        rp = tmp_path / f"{name}.json"; s, _ = arm("CLI_RECEIPT_WRITE", fe); fd = cli._step(s, "CLI_RECEIPT_RESERVE", lambda: cli._reserve_receipt(rp))
        with pytest.raises(cli.ProgressSinkFailure): cli._step(s, "CLI_RECEIPT_WRITE", lambda: cli._write_reserved(fd, ok))
        assert cli._fail_closed_sink(fd) == 2 and json.loads(rp.read_text())["reason"] == "progress_sink_failure"
    r4 = tmp_path / "a4.json"; held = []; s4, _ = arm("CLI_RECEIPT_RESERVE", "done")
    with pytest.raises(cli.ProgressSinkFailure):
        cli._step(s4, "CLI_RECEIPT_RESERVE", lambda: held.append(cli._reserve_receipt(r4)) or held[0])
    assert cli._fail_closed_sink(held[0]) == 2 and json.loads(r4.read_text())["reason"] == "progress_sink_failure"
    s5, c5 = arm("CLI_LAUNCH_IDENTITY_BIND", "error")
    with pytest.raises(cli.ProgressSinkFailure):
        cli._step(s5, "CLI_LAUNCH_IDENTITY_BIND", lambda: (_ for _ in ()).throw(RuntimeError("bind_boom")))
    assert ("CLI_LAUNCH_IDENTITY_BIND", "error") in c5
    monkeypatch.setattr(cli, "_build_progress_sink", lambda _t: (lambda step, edge, reason=None: (_ for _ in ()).throw(RuntimeError("SINK_FAIL"))))
    assert cli.main(["--packet", str(tmp_path / "p.json"), "--receipt", str(tmp_path / "nope.json")]) == 2 and not (tmp_path / "nope.json").exists()

_CAP = ("CAP_COMPUTE_GRADS", "CAP_VOTE_AUX", "CAP_SPEC", "CAP_APPLY_VOTE_STEP", "CAP_POST_RETURN_HOLDER_VALIDATION")
_CAP_FAIL_MSG = {c: f"fail_{c}" for c in _CAP}
def _live_cap_env(monkeypatch, *, fail_at=None):
    """Monkeypatch live capture deps. fail_at selects which CAP underlying op raises."""
    import calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu as ag
    import calm.hrm_text_158.native_full_stack.bounded_delta_learner as bdl
    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe
    apply_calls = []
    def grads(*_a, **_k):
        if fail_at == "CAP_COMPUTE_GRADS": raise RuntimeError(_CAP_FAIL_MSG[fail_at])
        return {"k0": 1}, 0.0, {}
    def votes(*_a, **_k):
        if fail_at == "CAP_VOTE_AUX": raise RuntimeError(_CAP_FAIL_MSG[fail_at])
        return {"k0": {}}, {}
    def spec(**_k):
        if fail_at == "CAP_SPEC": raise RuntimeError(_CAP_FAIL_MSG[fail_at])
        return {"spec": 1}
    def fake_apply(*_a, **k):
        apply_calls.append(dict(k))
        if fail_at == "CAP_APPLY_VOTE_STEP": raise RuntimeError(_CAP_FAIL_MSG[fail_at])
        if fail_at == "CAP_POST_RETURN_HOLDER_VALIDATION":
            return SimpleNamespace(tensor_states={})  # skip observer → post raises AuthoritativeGpuError
        k["front_c_identity_observer"]({"plans_by_key": {"k0": _plan(idxs=[0, 1, 2, 3])}, "tensor_states": {}})
        return SimpleNamespace(tensor_states={})
    monkeypatch.setattr(probe, "_compute_ce_weighted_grads", grads)
    monkeypatch.setattr(probe, "_weighted_grads_to_vote_aux_maps", votes)
    monkeypatch.setattr(probe, "resolve_probe_vote_update_spec", spec)
    monkeypatch.setattr(probe, "build_identity_full_support_batches",
                        lambda **_k: ([_batch(32, 0), _batch(32, 100), _batch(26, 200)], None))
    monkeypatch.setattr(bdl, "apply_bounded_delta_vote_step", fake_apply)
    monkeypatch.setattr(bdl, "canonical_acquisition_rank_vote_spec", lambda: {})
    model = SimpleNamespace(train=lambda: None, compute_train_extra_args=lambda *_a: {})
    bundle = SimpleNamespace(model=model, eligible_modules={"k0": None}, tok=None,
                             cfg=SimpleNamespace(max_seq_len=64), tensor_states={})
    return ag, bundle, apply_calls
def _live_capture_hooks(monkeypatch, *, fail_at=None, progress_sink=None):
    """DI early stages + real live capture_plans from build_live_hooks (shared _cap path)."""
    from dataclasses import replace
    ag_mod, bundle, apply_calls = _live_cap_env(monkeypatch, fail_at=fail_at)
    live = ag_mod.build_live_hooks({"device": "cpu", "allow_cpu_legacy_eval": True}, progress_sink=progress_sink)
    live.rebuild_support_batches(bundle)
    def capture(_b, arms):
        return live.capture_plans(bundle, {"capture_disposable": arms["capture_disposable"]})
    return replace(_hooks({}), capture_plans=capture), apply_calls, bundle
def test_d2c13_cap_progress_emit_matrix(monkeypatch):
    # DI path never calls build_live_hooks
    calls = []
    monkeypatch.setattr(so, "build_live_hooks", lambda *_a, **_k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("x")))
    assert run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}))["classifier"] == SUPPORT_ELIGIBLE
    assert calls == []
    with pytest.raises(ValueError, match="bad_step"):
        cli._build_progress_sink(0)("NOT_A_CAP", "start", None)
    for name in _CAP:
        cli._build_progress_sink(0)(name, "start", None); cli._build_progress_sink(0)(name, "done", None)
    # happy-path nested CAP order on live _cap
    ag_mod, bundle, apply_calls = _live_cap_env(monkeypatch)
    ev = []
    h = ag_mod.build_live_hooks(
        {"device": "cpu", "allow_cpu_legacy_eval": True},
        progress_sink=lambda step, edge, reason=None: so._progress(
            lambda s, e, r=None: ev.append((s, e)), step, edge, reason))
    h.rebuild_support_batches(bundle)
    plans, _st, hc = h.capture_plans(bundle, {"capture_disposable": {"k0": object()}})
    assert hc == 1 and "k0" in plans and len(apply_calls) == 1
    assert apply_calls[0].get("pc_aux_mode") == "telemetry" and apply_calls[0].get("two_tier_carry_w6_enabled") is False
    assert ev == [x for cap in _CAP for x in ((cap, "start"), (cap, "done"))]
    # (1) each CAP underlying failure via live _cap → start/error, no done, no later CAP; MOD error; reason preserved
    for i, cap in enumerate(_CAP):
        ev_u = []
        hooks, _, _ = _live_capture_hooks(
            monkeypatch, fail_at=cap,
            progress_sink=lambda s, e, r=None, _ev=ev_u: _ev.append((s, e)))
        out = run_support_only_characterization(
            build_support_only_test_packet(expected_head="0" * 40), hooks=hooks,
            progress_sink=lambda s, e, r=None, _ev=ev_u: _ev.append((s, e)))
        assert out["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE
        assert out["classifier"] != SUPPORT_ELIGIBLE
        if cap == "CAP_POST_RETURN_HOLDER_VALIDATION":
            assert out["reason"] == "raw_holder_call_count"
        else:
            assert out["reason"] == f"RuntimeError:{_CAP_FAIL_MSG[cap]}"
        assert (cap, "start") in ev_u and (cap, "error") in ev_u and (cap, "done") not in ev_u
        assert ("MOD_CAPTURE_PLANS", "error") in ev_u
        later = _CAP[i + 1:]
        assert not any(s in later for s, _ in ev_u)
    # (2) each CAP error-edge fail-once (underlying fails) → no later CAP/MOD raw; progress_sink_failure
    for cap in _CAP:
        raw_calls, failed = [], []
        def raw(step, edge, reason=None, _cap=cap):
            raw_calls.append((step, edge))
            if step == _cap and edge == "error" and not failed:
                failed.append(1); raise RuntimeError("raw_once")
        hooks, _, _ = _live_capture_hooks(
            monkeypatch, fail_at=cap,
            progress_sink=lambda step, edge, reason=None, _raw=raw: so._progress(_raw, step, edge, reason))
        out = run_support_only_characterization(
            build_support_only_test_packet(expected_head="0" * 40), hooks=hooks,
            progress_sink=lambda step, edge, reason=None, _raw=raw: so._progress(_raw, step, edge, reason))
        assert out["reason"] == "progress_sink_failure" and out["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE
        assert out["classifier"] != SUPPORT_ELIGIBLE and len(failed) == 1
        fail_idx = next(i for i, x in enumerate(raw_calls) if x == (cap, "error"))
        assert not any(s.startswith("CAP_") or s.startswith("MOD_") for s, _ in raw_calls[fail_idx + 1:])
    # (3) keep all-five start/done fail-once matrix on live capture_plans
    for cap, edge_fail in [(c, e) for e in ("start", "done") for c in _CAP]:
        raw_calls, failed = [], []
        def raw(step, edge, reason=None, _cap=cap, _ef=edge_fail):
            raw_calls.append((step, edge))
            if step == _cap and edge == _ef and not failed:
                failed.append(1); raise RuntimeError("raw_once")
        ag_i, bun, _ = _live_cap_env(monkeypatch)
        hi = ag_i.build_live_hooks(
            {"device": "cpu", "allow_cpu_legacy_eval": True},
            progress_sink=lambda step, edge, reason=None, _raw=raw: so._progress(_raw, step, edge, reason))
        hi.rebuild_support_batches(bun)
        with pytest.raises(so._ProgressSinkFailure):
            hi.capture_plans(bun, {"capture_disposable": {"k0": object()}})
        assert len(failed) == 1
        fail_idx = next(i for i, x in enumerate(raw_calls) if x == (cap, edge_fail))
        assert not any(s.startswith("CAP_") for s, _ in raw_calls[fail_idx + 1:])
