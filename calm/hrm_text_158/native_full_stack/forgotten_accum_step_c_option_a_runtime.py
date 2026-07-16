"""Step-C Option-A runtime facade (Path-injected; no hardcoded v1 occupied paths)."""
from __future__ import annotations
import hashlib, json, os, subprocess, time, uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from calm.hrm_text_158.native_full_stack.forgotten_accum_step_c_characterization_reducers import (
    MAX_PHYSICAL, MAX_RUNNER_ATTEMPTS, PHASE_BUDGETS, REQUIRED_KEYS, ReducerViolation,
    assert_keysets, assert_phase_budget, assert_planned_physical_cap, assert_runner_attempt_cap,
    assert_scratch_paths_equal, assert_scratch_pre_empty, exact_int_steps,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_step_c_mixed_residency_observer import (
    ObserverViolation, assert_mixed_residency,
)
T_CUT, RUNWAY, RW = 2, 4, 1
PASS_LABEL = "ACCOUNTING_V2_CHARACTERIZATION_VALID_AT_OPTION_A_GEOMETRY_2_4_1"
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
PREIMPLEMENTATION_HEAD_BASELINE = "200632f5c1f0ebf64aa0cf8f4b3ed012a3aef6cb"
EXPECTED_HEAD_ENV = "ACCT_V2_STEP_C_EXPECTED_HEAD"
_UNRESOLVED_HEAD = frozenset({
    "", "unresolved", "unresolved_post_commit", "post_commit_required", "tbd", "none", "null",
})
SOURCE_PINS = {
    "facade": "2c8a76cb17a932abf7b21dc6af9d86901cfcd0fe34a3b7cdb2ee0713c6a97fab",
    "core": "dceba582104f5bde6cb770802d0bcd80f7d416db1e9e856e5b773725cd5dfc5b",
    "adapter": "3f5c593bb0be38d59a17ed7c31f35de47cf74f4cbb45906802f70a7fe744b90f",
    "ark_invoke": "3f6c71c56c79af3a6724ae57e2ba70e6e439c12d51a530f5b545a520a6d63e20",
    "science_driver": "b9ef6496532a60c882052751aff11608fc41c7f7f90bd66f82621d3836a92609",
    "probe": "d39c3ead23f56edc36ec16409f38fed2129179723482b401ebac2a5aa6757701",
    "contracts": "19ae9024cce39b0edefe0f146036cb618369e1442eee6163ebd827a5d254f2cd",
    "observer": "cc99863eb4dba33cc4103f1892f1e364ef9a707f7d4242ab3c31e06bca77a613",
    "reducers": "9070b9e775ebea6ec1cb2a31ada45af493216e5244ccfebedb90e852fc49c135",
    "learner": "ef3eb981c23c5ba43bfe9e6a647e9a59b55824204f368efb3f0fcd1dfe92365d",
    "accumulator": "4ff08f849bfbc3beb4db7d6af4570b861e8555041422ccffa5946659b10208f6",
    "event_coded_carrier": "72d7a135832f979504f7c5eaab8d772cb16b7bc130351f5d002660af9f560b3c",
}
_NS = "calm/hrm_text_158/native_full_stack"
_PIN_FILES = {
    "facade": f"{_NS}/forgotten_accum_a_ledger_accounting_v2.py",
    "core": f"{_NS}/forgotten_accum_a_ledger_accounting_v2_core.py",
    "adapter": f"{_NS}/forgotten_accum_a_ledger_accounting_v2_ark_adapter.py",
    "ark_invoke": f"{_NS}/forgotten_accum_training_equivalence_ark_invoke.py",
    "science_driver": f"{_NS}/forgotten_accum_training_equivalence_science_driver.py",
    "contracts": f"{_NS}/forgotten_accum_training_equivalence_contracts.py",
    "observer": f"{_NS}/forgotten_accum_step_c_mixed_residency_observer.py",
    "reducers": f"{_NS}/forgotten_accum_step_c_characterization_reducers.py",
    "learner": f"{_NS}/bounded_delta_learner.py",
    "accumulator": f"{_NS}/bounded_delta_accumulator.py",
    "event_coded_carrier": f"{_NS}/event_coded_acc_live_carrier.py",
    "probe": "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
}
PIN_RELS = tuple((_PIN_FILES[k], k) for k in SOURCE_PINS)
class StepCBlock(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"[BLOCK] {code}" + (f": {detail}" if detail else ""))
def _map(exc: Exception) -> None:
    if isinstance(exc, (ObserverViolation, ReducerViolation)):
        raise StepCBlock(exc.code, exc.detail) from exc
    raise exc
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def list_scratch_files(scratch: Path) -> list[Path]:
    return [] if not scratch.exists() else sorted(p.resolve() for p in scratch.rglob("*") if p.is_file())
def assert_geometry_admitted(t_cut: int, runway: int, rewarm: int) -> None:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        is_option_a_admitted_characterization_geometry as _adm,
    )
    if not _adm(t_cut=t_cut, runway_steps=runway, rewarm_window_steps=rewarm):
        raise StepCBlock("UNADMITTED_GEOMETRY", f"{(t_cut, runway, rewarm)}")
def make_ctx(*, clock: Callable[[], float] = time.monotonic) -> SimpleNamespace:
    return SimpleNamespace(
        fork_complete=False, u_cut_source_fp=None, clock=clock,
        real_runner_attempt_count=0, planned_physical_update_count=0,
        cuda_model_forward_backward_completed_count=0, cpu_q_acc_apply_completed_count=0,
        observed_keysets={a: set() for a in REQUIRED_KEYS},
        model_checkpoint_materialization_call_count=0, parent_checkpoint_deserialize_count=0,
        resume_arm_call_count=0, resume_artifact_load_count=0, cut_fork_updates_during=0,
        cut_fork_duration_s=0.0, trainers_during_cut_fork=0, in_cut_fork=False, phase_durations={},
    )
def phase_check(ctx: SimpleNamespace, name: str, started: float) -> float:
    dur = float(ctx.clock()) - float(started)
    ctx.phase_durations[name] = dur
    try:
        assert_phase_budget(name=name, duration_s=dur)
    except ReducerViolation as exc:
        _map(exc)
    return dur
def wrap_runner(real: Callable[..., Any], ctx: SimpleNamespace) -> Callable[..., Any]:
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        if ctx.in_cut_fork:
            ctx.trainers_during_cut_fork += 1
            raise StepCBlock("CUT_FORK_TRAINER_UPDATE", "runner entered cut/fork")
        try:
            steps = exact_int_steps(kwargs.get("steps"))
            assert_planned_physical_cap(planned=int(ctx.planned_physical_update_count), steps=steps)
            nxt = int(ctx.real_runner_attempt_count) + 1
            assert_runner_attempt_cap(next_attempt=nxt)
        except ReducerViolation as exc:
            _map(exc)
        arm = str(kwargs.get("ordered_apply_event_arm_id") or "")
        if arm not in REQUIRED_KEYS:
            raise StepCBlock("ARM_ID_MISSING", arm)
        phase = f"arm_{arm}"
        print(f"[PROG] {phase} START", flush=True)
        t0 = ctx.clock()
        ctx.planned_physical_update_count = int(ctx.planned_physical_update_count) + steps
        ctx.real_runner_attempt_count = nxt
        try:
            return real(*args, **kwargs)
        finally:
            phase_check(ctx, phase, t0)
            print(f"[PROG] {phase} END", flush=True)
    return _wrapped
def record_step(ctx: SimpleNamespace, *, arm: str, step: int, model, batch, states) -> None:
    if ctx.in_cut_fork:
        ctx.cut_fork_updates_during += 1
        raise StepCBlock("CUT_FORK_TRAINER_UPDATE", "post-step during cut/fork")
    total = ctx.cuda_model_forward_backward_completed_count + 1
    if total > MAX_PHYSICAL:
        raise StepCBlock("PHYSICAL_UPDATES_OVERFLOW", str(total))
    try:
        assert_mixed_residency(model=model, batch=batch, states=states)
    except ObserverViolation as exc:
        _map(exc)
    keys = ctx.observed_keysets.setdefault(arm, set())
    if step in keys:
        raise StepCBlock("COUNTER_MISMATCH", f"duplicate {arm}:{step}")
    keys.add(step)
    ctx.cuda_model_forward_backward_completed_count = total
    ctx.cpu_q_acc_apply_completed_count = total
def mutate_holders_from_resume(*, state_holder, backlog_holder, resume, expected_fp, arm) -> None:
    meta = dict(resume.meta)
    if meta.get("pre_cut_source_sha256") != expected_fp:
        raise StepCBlock("FORK_SNAPSHOT_NOT_FROM_U_CUT", arm)
    need = {"E": "exact_preserve", "R0": "zero_strip", "RW": "zero_strip"}[arm]
    if str(meta.get("policy") or "") != need:
        raise StepCBlock("FORK_SNAPSHOT_NOT_FROM_U_CUT", f"policy {arm}")
    state_holder.clear(); state_holder.update(dict(resume.tensor_states))
    backlog_holder.clear()
    if resume.deferred_backlog is not None:
        backlog_holder.update(dict(resume.deferred_backlog))
    if not state_holder:
        raise StepCBlock("FORK_INCOMPLETE", f"empty states {arm}")
def run_cut_fork_at_u_step2(
    *, event, ctx, holders_states, holders_backlog, scratch, resume_fn, pre_cut_fp_fn, arm_id_cls,
) -> None:
    if int(event.step) != T_CUT:
        raise StepCBlock("FORK_SNAPSHOT_NOT_FROM_U_CUT", f"step={event.step}")
    print("[PROG] cut_fork_serialize_load START", flush=True)
    t0 = ctx.clock(); ctx.in_cut_fork = True
    try:
        fp = pre_cut_fp_fn(event.states, event.carry_backlog, None); ctx.u_cut_source_fp = fp
        for name in ("E", "R0", "RW"):
            if ctx.clock() - t0 > PHASE_BUDGETS["cut_fork_serialize_load"]:
                raise StepCBlock("CUT_FORK_PHASE_TIMEOUT", "mid")
            arm = arm_id_cls[name] if not isinstance(name, arm_id_cls) else name
            resume = resume_fn(
                arm=arm, live_states=event.states, live_backlog=event.carry_backlog,
                experiment_root=scratch, shared_pre_cut_source_sha256=fp,
            )
            ctx.resume_arm_call_count += 1; ctx.resume_artifact_load_count += 1
            mutate_holders_from_resume(
                state_holder=holders_states[name], backlog_holder=holders_backlog[name],
                resume=resume, expected_fp=fp, arm=name,
            )
        if ctx.trainers_during_cut_fork or ctx.cut_fork_updates_during:
            raise StepCBlock("CUT_FORK_TRAINER_UPDATE", "nonzero updates")
        ctx.fork_complete = True
    finally:
        ctx.in_cut_fork = False
        ctx.cut_fork_duration_s = phase_check(ctx, "cut_fork_serialize_load", t0)
        print("[PROG] cut_fork_serialize_load END", flush=True)
def gate_e_requires_fork(ctx: SimpleNamespace) -> None:
    if not ctx.fork_complete:
        raise StepCBlock("FORK_INCOMPLETE", "E before fork_complete")
def finalize_counters(ctx: SimpleNamespace, scratch: Path, resume_allow: tuple[Path, ...]) -> None:
    try:
        assert_keysets(ctx.observed_keysets)
        assert_scratch_paths_equal(got=list_scratch_files(scratch), allow=list(resume_allow))
    except ReducerViolation as exc:
        _map(exc)
    if ctx.real_runner_attempt_count != MAX_RUNNER_ATTEMPTS:
        raise StepCBlock("COUNTER_MISMATCH", "attempts")
    for label, got in (
        ("planned", ctx.planned_physical_update_count),
        ("cuda", ctx.cuda_model_forward_backward_completed_count),
        ("cpu", ctx.cpu_q_acc_apply_completed_count),
    ):
        if got != MAX_PHYSICAL:
            raise StepCBlock("PHYSICAL_UPDATES_NE_10", label)
    if ctx.resume_arm_call_count != 3 or ctx.resume_artifact_load_count != 3:
        raise StepCBlock("RESUME_ARM_CALL_COUNT_NE_3")
    if ctx.model_checkpoint_materialization_call_count != 1:
        raise StepCBlock("MODEL_CHECKPOINT_MATERIALIZATION_NE_1")
    if ctx.parent_checkpoint_deserialize_count != 1:
        raise StepCBlock("PARENT_CHECKPOINT_DESERIALIZE_NE_1")
def compose_hooks(
    *, ctx, model, batch, holders_states, holders_backlog, scratch, resume_fn, pre_cut_fp_fn, arm_id_cls,
):
    def u_hook(event):
        record_step(ctx, arm="U", step=int(event.step), model=model, batch=batch, states=event.states)
        if int(event.step) == T_CUT:
            run_cut_fork_at_u_step2(
                event=event, ctx=ctx, holders_states=holders_states, holders_backlog=holders_backlog,
                scratch=scratch, resume_fn=resume_fn, pre_cut_fp_fn=pre_cut_fp_fn, arm_id_cls=arm_id_cls,
            )
    def make(arm: str):
        def _h(event):
            if arm == "E":
                gate_e_requires_fork(ctx)
            record_step(ctx, arm=arm, step=int(event.step), model=model, batch=batch, states=event.states)
        return _h
    return {"U": u_hook, "E": make("E"), "R0": make("R0"), "RW": make("RW")}
def write_json_excl(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
def verify_log_identity(*, log_path: Path, frozen: Mapping[str, Any]) -> None:
    st = log_path.stat()
    if str(frozen.get("resolved_path")) != str(log_path.resolve()):
        raise StepCBlock("WRONG_LOG_IDENTITY", "resolved_path")
    if int(frozen["st_dev"]) != st.st_dev or int(frozen["st_ino"]) != st.st_ino:
        raise StepCBlock("WRONG_LOG_IDENTITY", "dev/ino")
    if int(frozen["mode"]) != (st.st_mode & 0o777):
        raise StepCBlock("WRONG_LOG_IDENTITY", "mode")
def verify_source_and_parent_pins(*, repo: Path, parent: Path) -> None:
    for rel, key in PIN_RELS:
        got = sha256_file(repo / rel)
        if got != SOURCE_PINS[key]:
            raise StepCBlock("SOURCE_OR_CHECKPOINT_DRIFT", f"{rel}:{got}")
    if sha256_file(parent) != PARENT_SHA:
        raise StepCBlock("SOURCE_OR_CHECKPOINT_DRIFT", "parent")
def resolve_expected_head(expected_head: str | None = None) -> str:
    raw = expected_head if expected_head is not None else os.environ.get(EXPECTED_HEAD_ENV, "")
    pin = str(raw or "").strip().lower()
    if pin in _UNRESOLVED_HEAD or len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise StepCBlock("HEAD_PIN_UNRESOLVED", f"expected_head={raw!r}")
    return pin
def verify_head_origin_pin(*, repo: Path, expected_head: str | None = None) -> str:
    pin = resolve_expected_head(expected_head)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/feature/hrm-text-1.58"], cwd=repo, text=True,
    ).strip()
    if head != pin or origin != pin or head != origin:
        raise StepCBlock("SOURCE_OR_CHECKPOINT_DRIFT", f"HEAD/origin {head}/{origin}!={pin}")
    return pin
def refuse_monitor_evidence_forge(path: Path) -> None:
    raise StepCBlock("HARNESS_MUST_NOT_FORGE_MONITOR_EVIDENCE", str(path))
def run_log_identity_preflight(
    *, log_path: Path, identity_path: Path, receipt_path: Path, monitor_path: Path,
    scratch: Path, repo: Path, parent: Path, check_git: bool = True,
    expected_head: str | None = None,
) -> dict[str, Any]:
    for p in (log_path, identity_path, receipt_path, monitor_path):
        if p.exists():
            raise StepCBlock("ARTIFACT_COLLISION", str(p))
    try:
        assert_scratch_pre_empty(list_scratch_files(scratch))
    except ReducerViolation as exc:
        _map(exc)
    if check_git:
        verify_head_origin_pin(repo=repo, expected_head=expected_head)
        verify_source_and_parent_pins(repo=repo, parent=parent)
    fd = os.open(str(log_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644); os.close(fd)
    st = log_path.stat()
    frozen = {
        "resolved_path": str(log_path.resolve()), "st_dev": st.st_dev, "st_ino": st.st_ino,
        "mode": st.st_mode & 0o777, "empty_sha256": hashlib.sha256(b"").hexdigest(),
        "launch_token": uuid.uuid4().hex,
    }
    write_json_excl(identity_path, frozen)
    verify_log_identity(log_path=log_path, frozen=frozen)
    return frozen
