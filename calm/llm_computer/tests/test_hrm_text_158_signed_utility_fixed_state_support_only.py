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
    assert sum(1 for _ in MOD.open()) <= 220 and sum(1 for _ in CLI.open()) <= 180
    assert sum(1 for _ in Path(__file__).open()) <= 240 and "from calm" not in CLI.read_text().split("def main")[0]
    assert list(TERMINAL_TAXONOMY)[:1] == [SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE]
    out = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40), hooks=_hooks({}))
    assert out["classifier"] == SUPPORT_ELIGIBLE; _assert_imm(out)
    monkeypatch.setattr(so, "build_live_hooks", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("hook_boom")))
    guarded = run_support_only_characterization(build_support_only_test_packet(expected_head="0" * 40))
    assert guarded["classifier"] == SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE and "hook_boom" in str(guarded.get("reason", ""))
    fac = cli._load_verified_facade(FACADE, rehash_path(FACADE)); assert Path(fac.__file__).resolve() == FACADE.resolve()
    spies = {}
    def _fake_run(p):
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
