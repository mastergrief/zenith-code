"""Shared run harness for HRM-158 bounded-delta measurement packets.

Source: `/home/gabe/claw-code-creditdir/transient_fp_credit/h2_windowed_hot_set/h2_windowed_hot_run_v1.frozen.py`
sha256 8f059a7c9101c920acff3ef68e8d97c38d14e7565aa680912dd5086c3992716a

Task 1788428215079-af9995e7, slice S1 (MOVE). Dispatch 1788439996018-8b9a8f0e.
ADVISOR_ROUTE: 1788439962383-bd9b8b20.

`threshold_abs` is threaded from the phase entry points into `run_loop` and the
emitted `run_env`. Task 1788456823866-aa9a873d.
ADVISOR_ROUTE: 1788456771491-42f1f60c.

Out of scope for this slice and authored in S2: the run-root / log /
`timeout.stderr` O_EXCL mint, the exclusivity probe, the pre-exec hash checks,
and the outer `timeout --verbose` classification.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
PARENT_REL = "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
MANIFEST_REL = "calm/hrm_text_158/manifests/chain_head_L0c1.json"
OBSERVER_COMMIT = "9738e321ddec3cf55dcca245e2bb2c64f247380e"
SEED = 17
ELIGIBLE_SCOPE = "all-bitlinear"
MAX_ABS_PER_TENSOR = 4096
SMOKE_STEPS = 10
K200_STEPS = 200
CADENCE_MODULUS = 10
SMOKE_WALL_CAP_SECONDS = 600
K200_WALL_CAP_SECONDS = 5400
WINDOW_KEYS = ("1", "10", "50")
STEP_ANCHOR = "step_reports[str(step)] = step_report"
SMOKE_OUT = "h2_windowed_hot_smoke.json"
TERMINAL_OUT = "h2_windowed_hot_terminal.json"
RUN_ROOT_ENV = "H2_WINDOWED_HOT_RUN_ROOT"
ENV_DIGEST_NAMES = ("PATH", "LD_LIBRARY_PATH")
ENV_NAME_PREFIXES = ("H2_", "PYTHON", "CUDA", "PYTORCH", "HRM_TEXT_158_")
ENV_NAME_DENY = ("KEY", "TOKEN", "SECRET", "PASSWORD", "OAUTH", "AUTH")


def _emit(line: str) -> None:
    print(line, flush=True)


_LAST_STOP: dict[str, Any] = {}


def _stop(code: str, detail: str) -> None:
    _LAST_STOP.clear()
    _LAST_STOP.update({"stop_code": code, "stop_detail": detail})
    _emit("[STOP] " + code + " :: " + detail)
    raise SystemExit(4)


def _run_root() -> Path:
    """Operator-supplied run root; absence is a named STOP, not a bare KeyError."""
    raw = os.environ.get(RUN_ROOT_ENV)
    if not raw:
        _stop("RUN_ROOT_UNSET", RUN_ROOT_ENV + " is unset or empty")
    root = Path(raw)
    if not root.is_dir():
        _stop("RUN_ROOT_MISSING", str(root))
    return root


def _driver_version() -> str:
    """No driver-version API exists in this torch build; read it from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception as exc:
        return "UNAVAILABLE:" + type(exc).__name__


def _run_env(device: Any, batch_size: int, threshold_abs: int) -> dict[str, Any]:
    import torch

    names = sorted(
        k
        for k in os.environ
        if k.startswith(ENV_NAME_PREFIXES) and not any(d in k.upper() for d in ENV_NAME_DENY)
    )
    return {
        "seed": SEED,
        "eligible_scope": ELIGIBLE_SCOPE,
        "max_abs_per_tensor": MAX_ABS_PER_TENSOR,
        "threshold_abs": int(threshold_abs),
        "batch_size_from_manifest": int(batch_size),
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "cuda_driver_version": _driver_version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allowlisted_env_names_NAMES_ONLY": names,
        "env_value_sha256_digests": {
            n: hashlib.sha256(os.environ.get(n, "").encode("utf-8")).hexdigest()
            for n in ENV_DIGEST_NAMES
        },
    }


def _executed_sources() -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256

    paths = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(Path(path)).startswith(str(REPO)):
            paths.add(str(Path(path)))
    return {
        "argv_verbatim": list(sys.argv),
        "executed_repo_sources_sha256": {p: file_sha256(p) for p in sorted(paths)},
    }


def _parent_identity() -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256

    manifest = json.loads((REPO / MANIFEST_REL).read_text())
    recorded = manifest["chain_head"]["ckpt_path"]
    fresh = file_sha256(REPO / PARENT_REL)
    pre = {
        "manifest_recorded_path": recorded,
        "batch_size_from_manifest": int(manifest["recipe"]["training"]["batch_size"]),
        "route_bound_path": PARENT_REL,
        "path_equal": recorded == PARENT_REL,
        "parent_sha_route_bound": PARENT_SHA,
        "parent_sha_fresh": fresh,
        "parent_sha_equal": fresh == PARENT_SHA,
    }
    if not (pre["path_equal"] and pre["parent_sha_equal"]):
        _stop("PARENT_IDENTITY_FAILED", json.dumps(pre, sort_keys=True))
    return pre


def _build(device: Any, batch_size: int):
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        build_identity_full_support_batches,
        build_model_from_checkpoint,
        derive_tensor_states_and_check_init_fidelity,
        load_parent_checkpoint,
        select_eligible_bitlinears,
    )

    ckpt, parent_sha_pre = load_parent_checkpoint(REPO / PARENT_REL, expected_sha256=PARENT_SHA)
    model, tok, _cfg = build_model_from_checkpoint(ckpt, device)
    support_batches, _proof = build_identity_full_support_batches(
        tok=tok,
        max_len=int(ckpt["config"]["max_seq_len"]),
        batch_size=int(batch_size),
        curriculum_seed=SEED,
        device=device,
    )
    eligible = select_eligible_bitlinears(model, eligible_scope=ELIGIBLE_SCOPE)
    states, fidelity = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)
    if fidelity.get("all_pass") is not True:
        _stop("INIT_FIDELITY_FAILED", "derive_tensor_states init fidelity did not pass")
    return model, support_batches, eligible, states, parent_sha_pre


def _run(
    model, support_batches, eligible, states, device: Any, steps: int, threshold_abs: int
):
    from calm.hrm_text_158.native_full_stack.global_rate_cap import (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    )
    from calm.hrm_text_158.native_full_stack.minimal_trainer.loop import run_loop

    return run_loop(
        model,
        support_batches[0]["batch"],
        states,
        eligible,
        device=device,
        steps=steps,
        require_q_change=False,
        max_abs_per_tensor=MAX_ABS_PER_TENSOR,
        threshold_abs=int(threshold_abs),
        support_batches=support_batches,
        r7_deferred_backlog_carry_enabled=True,
        global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        global_horizon=steps,
        ever_crossed_observer_enabled=True,
    )


def _check_flush(flushed: list[int], steps: int) -> None:
    """Refuse a phase whose per-step monitor did not fire once on every step.

    Named separately from `_phase` so its firing world is reachable without a
    GPU: a dead or misbound step monitor leaves this list short and must STOP
    rather than publish a phase whose in-flight emission never happened.
    """
    if list(flushed) != list(range(1, int(steps) + 1)):
        _stop(
            "STEP_FLUSH_INCOMPLETE",
            "flushed_count=" + str(len(flushed)) + " expected=" + str(steps)
            + " first_five=" + repr(list(flushed)[:5]),
        )


def _check_step_schema(step_reports: Mapping[str, Any], steps: int) -> None:
    """One presence gate for every field this packet later reads by index.

    Fails when a step row is missing, carries no `duration_seconds`, was not
    stamped by the observer on its own step, or does not carry exactly the
    pinned window set — each of which would otherwise surface as an incidental
    container exception instead of this named STOP.
    """
    bad: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        report = step_reports.get(str(step))
        windows = report.get("windowed_candidate_windows") if isinstance(report, dict) else None
        present = sorted(windows) if isinstance(windows, dict) else None
        observed = report.get("windowed_candidate_step") if isinstance(report, dict) else None
        duration = report.get("duration_seconds") if isinstance(report, dict) else None
        if present != sorted(WINDOW_KEYS) or observed != step or duration is None:
            bad[str(step)] = {
                "row_present": isinstance(report, dict),
                "windows_present": present,
                "observer_step": observed,
                "duration_present": duration is not None,
            }
    if bad:
        _stop(
            "STEP_SCHEMA_MISMATCH",
            json.dumps(
                {"expected_windows": sorted(WINDOW_KEYS), "steps": int(steps), "bad": bad},
                sort_keys=True,
            ),
        )


def _write(path: Path, payload: Mapping[str, Any]) -> str:
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import o_excl_write_json

    try:
        return o_excl_write_json(path, payload)
    except FileExistsError:
        _stop("TERMINAL_PATH_COLLISION", str(path))
        raise


def _device() -> Any:
    import torch

    if not torch.cuda.is_available():
        _stop("NO_CUDA_DEVICE", "packet requires a CUDA device")
    torch.manual_seed(SEED)
    return torch.device("cuda:0")


class _PhaseWallCapExceeded(Exception):
    """Raised in-process by SIGALRM when a phase exceeds its internal wall cap."""


def _arm_phase_cap(seconds: int, label: str) -> None:
    """Smallest reliable in-process wall cap: one interval timer, no thread or subprocess."""

    def _fire(signum: int, frame: Any) -> None:
        raise _PhaseWallCapExceeded(
            label + " exceeded internal wall cap " + str(int(seconds)) + "s"
        )

    signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))


def _disarm_phase_cap() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def _anchor_line(func: Any, anchor: str) -> int:
    """Resolve a UNIQUE source line; zero or multiple matches refuse."""
    lines, start_line = inspect.getsourcelines(func)
    matches = [i for i, line in enumerate(lines) if anchor in line]
    if len(matches) != 1:
        raise RuntimeError(
            "monitor_anchor_count=" + str(len(matches)) + " function=" + func.__name__
        )
    return start_line + matches[0]


@contextmanager
def _step_flush_monitor(run_loop: Any, phase: str, total: int, flushed: list[int]):
    """Flush each step's duration as it completes, and window rows on cadence.

    Every read here is presence-tolerant: an absent observer field emits its own
    `windows_present` row rather than raising inside the callback, leaving the
    named `_check_step_schema` STOP to own that failure after the run returns.
    """
    anchor = _anchor_line(run_loop, STEP_ANCHOR)
    tool_id = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(tool_id, "h2_windowed_step_monitor")

    def _cb(code: Any, line: int) -> None:
        if code is not run_loop.__code__ or line != anchor:
            return
        frame = sys._getframe(1)
        step = int(frame.f_locals["step"])
        report = frame.f_locals["step_report"]
        flushed.append(step)
        _emit(
            "[PROG] step_complete phase=" + phase + " step=" + str(step) + "_of_"
            + str(total) + " duration_seconds=" + repr(report.get("duration_seconds"))
        )
        if step % CADENCE_MODULUS:
            return
        windows = report.get("windowed_candidate_windows")
        windows = windows if isinstance(windows, dict) else {}
        _emit(
            "[PROG] window_step phase=" + phase + " step=" + str(step)
            + " observer_step=" + repr(report.get("windowed_candidate_step"))
            + " windows_present=" + json.dumps(sorted(windows))
        )
        for key in sorted(windows):
            row = windows[key] if isinstance(windows[key], dict) else {}
            _emit(
                "[PROG] window phase=" + phase + " step=" + str(step) + " W=" + key
                + " count_total=" + repr(row.get("count_total"))
                + " numel_total=" + repr(row.get("numel_total"))
                + " fraction_total=" + repr(row.get("fraction_total"))
                + " count_by_key=" + json.dumps(row.get("count_by_key"), sort_keys=True)
                + " numel_by_key=" + json.dumps(row.get("numel_by_key"), sort_keys=True)
                + " fraction_by_key=" + json.dumps(row.get("fraction_by_key"), sort_keys=True)
            )

    sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, _cb)
    sys.monitoring.set_local_events(tool_id, run_loop.__code__, sys.monitoring.events.LINE)
    try:
        yield
    finally:
        sys.monitoring.set_local_events(tool_id, run_loop.__code__, 0)
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.LINE, None)
        sys.monitoring.free_tool_id(tool_id)


def _phase(
    name: str,
    steps: int,
    cap_seconds: int,
    out_name: str,
    threshold_abs: int = 1,
) -> tuple[dict[str, Any], str]:
    """One protected phase: the cap covers material evidence through the write.

    Event order inside the armed interval is exactly device > identity > run_env >
    build > monitor > run > schema > parent_hash > payload > executed_sources >
    output_write > non-authoritative progress row, then the single `finally`
    disarms and only then does this function return.

    The AUTHORITY marker is NOT cap-protected and is not emitted here:
    `orchestrate` emits it after this function returns, so a cap firing at any
    bytecode before the return leaves a file and possibly an inner progress row
    but no authoritative marker.
    """
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256
    from calm.hrm_text_158.native_full_stack.minimal_trainer.loop import run_loop

    wall0 = time.perf_counter()
    _arm_phase_cap(cap_seconds, name)  # material evidence through O_EXCL write + inner progress
    try:
        device = _device()
        identity = _parent_identity()
        batch_size = int(identity["batch_size_from_manifest"])
        run_env = _run_env(device, batch_size, threshold_abs)
        _emit("[PROG] " + name + " run_env " + json.dumps(run_env, sort_keys=True))
        setup0 = time.perf_counter()
        model, support_batches, eligible, states, parent_sha_pre = _build(device, batch_size)
        setup_wall_seconds = time.perf_counter() - setup0
        _emit("[PROG] " + name + " build complete leaves=" + str(len(states)))
        flushed: list[int] = []
        with _step_flush_monitor(run_loop, name, steps, flushed):
            step_reports = _run(
                model, support_batches, eligible, states, device, steps, threshold_abs
            )[0]
        _check_flush(flushed, steps)
        _check_step_schema(step_reports, steps)
        durations = [float(step_reports[str(i)]["duration_seconds"]) for i in range(1, steps + 1)]
        parent_sha_post = file_sha256(REPO / PARENT_REL)
        if parent_sha_post != parent_sha_pre:
            _stop("PARENT_MUTATED_DURING_RUN", parent_sha_pre + " -> " + parent_sha_post)
        payload: dict[str, Any] = {
            "schema": "h2_windowed_hot_set/run_v1/" + name,
            "advisor_route": "1788381244526-49361a95",
            "dispatch": "1788428246697-d1d6b6ab",
            "dispatch_superseded_on_size_term": "1788427713764-2e2fe6ed",
            "advisor_route_amendment": "1788428202068-eeca5e7a",
            "h1_exemplar_sha256": (
                "93d65c97e2b3f865e435753528521d0301303165a67e8c97b28c147b86c26ffc"
            ),
            "observer_commit": OBSERVER_COMMIT,
            "phase": name,
            "steps_requested": steps,
            "steps_completed": len(durations),
            "windows_pinned": list(WINDOW_KEYS),
            "cadence_modulus_LOG_ROWS_ONLY": CADENCE_MODULUS,
            "run_env": run_env,
            "parent_identity_preflight": identity,
            "parent_sha_pre": parent_sha_pre,
            "parent_sha_post": parent_sha_post,
            "eligible_leaf_count": len(states),
            "step_seconds": durations,
            "setup_wall_seconds": setup_wall_seconds,
            "phase_wall_cap_seconds_INTERNAL": int(cap_seconds),
            "phase_wall_seconds_before_output_COMPUTED": time.perf_counter() - wall0,
            "step_reports_raw": step_reports,
            "emission_contract": (
                "raw observer rows only; this packet computes no window arithmetic, "
                "no branch, no flatness, no liveness verdict, no classifier"
            ),
            "admission_valid_only_with_authority_marker": (
                "this file is an admitted phase ONLY when the log carries "
                + ("[PROG] SMOKE_ADMITTED" if name == "smoke" else "[TERMINAL] RUN_COMPLETE")
                + " sha256=<this file's sha256>, emitted by orchestrate AFTER the "
                "phase returned and the wall cap was disarmed. The in-cap row "
                "[PROG] phase_payload_written is progress only and never admission "
                "proof; a file carrying PASS without the authority marker was "
                "interrupted before the phase returned"
            ),
        }
        if name == "smoke":
            payload["admission"] = (
                "PASS_TO_K200" if len(durations) == steps else "STOP_BEFORE_K200"
            )
        del model, support_batches, eligible, states
        payload["executed_sources_at_exit"] = _executed_sources()
        digest = _write(_run_root() / out_name, payload)
        # NON-AUTHORITATIVE progress only; never admission or terminal proof.
        _emit(
            "[PROG] phase_payload_written phase=" + name + " " + out_name
            + " sha256=" + digest
            + " phase_wall_seconds_before_return_COMPUTED=" + repr(time.perf_counter() - wall0)
        )
        result = (payload, digest)
    finally:
        _disarm_phase_cap()
    return result


def run_smoke(threshold_abs: int = 1) -> tuple[dict[str, Any], str | None]:
    """10 steps under the internal 600 s cap; the JSON is written INSIDE that cap.

    Returns (payload, digest). The digest is None on every failure path. Admission
    authority is the post-return `[PROG] SMOKE_ADMITTED` marker emitted by
    `orchestrate`, never this file's presence or the in-cap progress row.
    """
    failed: dict[str, Any] = {
        "schema": "h2_windowed_hot_set/run_v1/smoke",
        "phase": "smoke",
        "steps_requested": SMOKE_STEPS,
        "steps_completed": 0,
        "phase_wall_cap_seconds_INTERNAL": SMOKE_WALL_CAP_SECONDS,
        "admission": "STOP_BEFORE_K200",
        "published": False,
    }
    try:
        return _phase(
            "smoke", SMOKE_STEPS, SMOKE_WALL_CAP_SECONDS, SMOKE_OUT, threshold_abs
        )
    except _PhaseWallCapExceeded as exc:
        return dict(failed, cap_exceeded=str(exc)), None
    except SystemExit as exc:
        # EXPECTED classified `_stop` inside the phase; raw cause preserved. The
        # smoke boundary still exits 3 via orchestrate. Unexpected exceptions are
        # NOT caught and never become a false admission record.
        return (
            dict(
                failed,
                source_exit_code=exc.code,
                stop_code=_LAST_STOP.get("stop_code"),
                stop_detail=_LAST_STOP.get("stop_detail"),
            ),
            None,
        )


def run_k200(threshold_abs: int = 1) -> tuple[dict[str, Any], str]:
    """One uninterrupted run_loop(steps=200); terminal JSON written under the cap."""
    try:
        return _phase(
            "k200", K200_STEPS, K200_WALL_CAP_SECONDS, TERMINAL_OUT, threshold_abs
        )
    except _PhaseWallCapExceeded as exc:
        # Classified STOP, never a partial terminal accepted as complete.
        _stop(
            "K200_WALL_CAP_EXCEEDED",
            str(exc) + " (bound internal cap " + str(K200_WALL_CAP_SECONDS) + "s)",
        )
        raise


def _release_transient_device_state() -> None:
    """Drop Python/CUDA transients between phases; never touches the banked parent."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:  # emission only; never a silent swallow
        _emit("[PROG] phase_boundary cuda_release_skipped=" + type(exc).__name__)


def orchestrate(smoke_fn: Any, k200_fn: Any) -> int:
    """Single-process gate: on any smoke STOP, K=200 is unreachable in this flow."""
    smoke, smoke_digest = smoke_fn()
    if smoke.get("admission") != "PASS_TO_K200":
        _emit(
            "[STOP] SMOKE_ADMISSION_FAILED :: completed="
            + str(smoke.get("steps_completed"))
            + "_of_" + str(SMOKE_STEPS)
            + " cap_seconds=" + str(SMOKE_WALL_CAP_SECONDS)
            + " cap_exceeded=" + repr(smoke.get("cap_exceeded"))
        )
        raise SystemExit(3)
    # AUTHORITY marker; boundary is stated once, in the `_phase` docstring.
    _emit("[PROG] SMOKE_ADMITTED sha256=" + str(smoke_digest))
    _emit("[PROG] phase_boundary smoke_state_released objects=" + str(gc.collect()))
    _release_transient_device_state()
    _, terminal_digest = k200_fn()
    _emit("[TERMINAL] RUN_COMPLETE sha256=" + str(terminal_digest))
    return 0


def main() -> int:
    if len(sys.argv) != 1:
        _stop("BAD_ARGV", "expected no arguments; got " + repr(sys.argv[1:]))
    return orchestrate(run_smoke, run_k200)


