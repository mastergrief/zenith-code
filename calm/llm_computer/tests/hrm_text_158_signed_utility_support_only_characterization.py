"""Thin CLI: stdlib-only bootstrap → verified creditdir import facade session (D2c9/D2c10)."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from typing import Any, Callable, Mapping
ESTIMAND = "full_state_legal_subset_signed_direction_fixed_state_heldout_utility"
SUPPORT_ELIGIBLE = "SUPPORT_ELIGIBLE"
SUPPORT_INTEGRITY = "SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE"
MAX_RESULT, REASON_MAX_B, LINE_MAX_B = 256 * 1024, 256, 512
FACADE_BASENAME = "signed_utility_fixed_state_creditdir_import_facade.py"
FACADE_REL = Path("calm/hrm_text_158/native_full_stack") / FACADE_BASENAME
WATCH_REL = Path("bin/watch-wrap")
WATCH_WRAP_HRM158_SHA256 = "a19f1c5fe88fb3dcbf00ab442047576708f75272210e9a0cc94ed9369bf45d4b"
PROGRESS_STEPS = frozenset(("CLI_RECEIPT_RESERVE CLI_PACKET_LOAD CLI_LAUNCH_IDENTITY_BIND CLI_FACADE_LOAD "
    "CLI_VERIFIED_SESSION_ENTER CLI_MODULE_IDENTITY_CHECKS MOD_PARSE_PACKET_PINS MOD_BUILD_LIVE_HOOKS MOD_PARENT_SHA_PRE "
    "MOD_MATERIALIZE MOD_REBUILD_BATCHES MOD_LEAKAGE MOD_FORK_ARMS MOD_CAPTURE_PLANS MOD_CHARACTERIZE "
    "MOD_VALIDATE_CHARACTERIZATION MOD_ENFORCE_FLOORS MOD_EMIT_TERMINAL CLI_RECEIPT_WRITE").split())
PROGRESS_EDGES = frozenset({"start", "done", "error"})
_MOD_KEYS = ("reducers", "schema", "pin_validation", "phase_telemetry", "integrity_proofs", "partition_leakage",
            "arm_proofs", "legal_subset", "eval_contract", "authoritative_gpu", "support_only", "driver", "facade")
MODULE_BASENAME_TO_KEY = {f"signed_utility_fixed_state_{k}.py": k for k in _MOD_KEYS}
ProgressEmit = Callable[[str, str, str | None], None]
class ProgressSinkFailure(Exception): pass
def _canonical_fail(reason: str) -> dict[str, Any]:
    payload = {"schema": "support_only_terminal_v1", "estimand": ESTIMAND, "classifier": SUPPORT_INTEGRITY,
               "reason": reason, "route": [], "claim_ceiling": "support_eligibility_only",
               "parent_sha_pre": None, "parent_sha_post": None, "source_sha_pre": {}, "source_sha_post": {},
               "launch_surface_sha_pre": {}, "launch_surface_sha_post": {}}
    blob = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    if len(blob) > MAX_RESULT: raise RuntimeError("canonical_fail_oversized")
    return payload
def _reserve_receipt(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
def _write_reserved(fd: int, payload: Mapping[str, Any]) -> None:
    blob = json.dumps(dict(payload), indent=2, allow_nan=False, sort_keys=True).encode() + b"\n"
    if len(blob) > MAX_RESULT: raise RuntimeError("result_oversized")
    view = memoryview(blob); off = 0
    while off < len(view):
        try: n = os.write(fd, view[off:])
        except InterruptedError: continue
        if n <= 0: raise RuntimeError(f"receipt_write_zero_progress:{n}")
        off += n
    os.fsync(fd)
def _rewrite_progress_sink_failure(fd: int) -> None:
    os.ftruncate(fd, 0); os.lseek(fd, 0, os.SEEK_SET)
    _write_reserved(fd, _canonical_fail("progress_sink_failure"))
def _close_reserved(fd: int | None) -> None:
    if fd is not None:
        try: os.close(fd)
        except OSError: pass
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _require_pin(pin: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(pin, Mapping) or "absolute_path" not in pin or "sha256" not in pin:
        raise RuntimeError(f"{label}_pin_requires_absolute_path_and_sha256")
    path = Path(str(pin["absolute_path"])).resolve()
    if not path.is_file(): raise RuntimeError(f"{label}_pin_path_missing:{path}")
    digest = _sha(path)
    if digest != str(pin["sha256"]): raise RuntimeError(f"{label}_pin_sha_mismatch:{digest}!={pin['sha256']}")
    return path, digest
def _bind_launch_identity(packet: Mapping[str, Any], *, self_file: Path) -> Path:
    root = Path(str(packet.get("repo_root") or "")).resolve()
    if not root.is_dir(): raise RuntimeError(f"repo_root_missing:{root}")
    cli_path, cli_sha = _require_pin(packet.get("cli_pin"), label="cli")
    if cli_path != self_file.resolve():
        raise RuntimeError(f"cli_pin_not_executing_file:{cli_path}!={self_file.resolve()}")
    if cli_sha != _sha(self_file): raise RuntimeError("cli_pin_sha_ne_executing_bytes")
    pins = packet.get("source_pins")
    if not isinstance(pins, Mapping): raise RuntimeError("source_pins_missing")
    expected_facade = (root / FACADE_REL).resolve()
    facade_pin = next((pin for pin in pins.values()
                       if isinstance(pin, Mapping) and "absolute_path" in pin
                       and Path(str(pin["absolute_path"])).resolve() == expected_facade), None)
    if facade_pin is None: raise RuntimeError(f"creditdir_import_facade_pin_not_repo_root:{expected_facade}")
    facade_path, _ = _require_pin(facade_pin, label="facade")
    if facade_path != expected_facade: raise RuntimeError(f"facade_pin_path_mismatch:{facade_path}!={expected_facade}")
    ww_path, ww_sha = _require_pin(packet.get("watch_wrap_pin"), label="watch_wrap")
    expected_ww = (root / WATCH_REL).resolve()
    if ww_path != expected_ww: raise RuntimeError(f"watch_wrap_pin_not_repo_root:{ww_path}!={expected_ww}")
    if ww_sha != WATCH_WRAP_HRM158_SHA256: raise RuntimeError(f"watch_wrap_unexpected_sha:{ww_sha}")
    return facade_path
def _load_verified_facade(path: Path, expected_sha: str):
    resolved = Path(path).resolve()
    if _sha(resolved) != expected_sha: raise RuntimeError(f"facade_sha_mismatch:{resolved}")
    name = "_su_creditdir_import_facade_verified"
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None: raise RuntimeError("facade_spec_load_failed")
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(name); sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        if Path(getattr(mod, "__file__", "") or "").resolve() != resolved:
            raise RuntimeError(f"facade_file_identity_mismatch:{mod.__file__}!={resolved}")
        return mod
    except Exception:
        if prev is None: sys.modules.pop(name, None)
        else: sys.modules[name] = prev
        raise
def _expected_module_map(source_pins: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, pin in source_pins.items():
        if key == "head" or not isinstance(pin, Mapping): continue
        mod_key = MODULE_BASENAME_TO_KEY.get(Path(str(pin.get("absolute_path", ""))).name)
        if mod_key is None: continue
        out[mod_key] = str(pin["sha256"])
    if set(out) != set(MODULE_BASENAME_TO_KEY.values()):
        raise RuntimeError(f"module_pin_map_incomplete:{sorted(set(MODULE_BASENAME_TO_KEY.values()) - set(out))}")
    return out
def _build_progress_sink(t0_ns: int) -> ProgressEmit:
    def sink(step: str, edge: str, reason: str | None = None) -> None:
        if type(step) is not str or step not in PROGRESS_STEPS: raise ValueError(f"bad_step:{step!r}")
        if type(edge) is not str or edge not in PROGRESS_EDGES: raise ValueError(f"bad_edge:{edge!r}")
        if reason is not None:
            if type(reason) is not str: raise TypeError("reason_not_str")
            if "\n" in reason or "\r" in reason: raise ValueError("reason_has_newline")
            rb = reason.encode("utf-8")
            if len(rb) > REASON_MAX_B or b"NaN" in rb or b"Infinity" in rb: raise ValueError("reason_invalid")
        elapsed_ms = (time.monotonic_ns() - t0_ns) // 1_000_000
        if type(elapsed_ms) is not int or elapsed_ms < 0: raise ValueError("elapsed_ms_invalid")
        line = f"SUPPORT_PROGRESS step={step} edge={edge} elapsed_ms={elapsed_ms}" + (f" reason={reason}" if reason is not None else "")
        lb = line.encode("utf-8")
        if len(lb) > LINE_MAX_B or b"NaN" in lb or b"Infinity" in lb: raise ValueError("line_invalid")
        print(line, flush=True)
    return sink
def _guarded_sink(raw: ProgressEmit) -> ProgressEmit:
    dead = False
    def sink(step: str, edge: str, reason: str | None = None) -> None:
        nonlocal dead
        if dead: raise ProgressSinkFailure("progress_sink_dead")
        try: raw(step, edge, reason)
        except Exception as exc: dead = True; raise ProgressSinkFailure from exc
    return sink
def _step(sink: ProgressEmit | None, step: str, fn):
    if sink is not None: sink(step, "start", None)
    try: out = fn()
    except Exception as exc:
        if sink is not None: sink(step, "error", f"{type(exc).__name__}:{exc}")
        raise
    if sink is not None: sink(step, "done", None)
    return out
def _run_via_verified_session(packet: Mapping[str, Any], *, self_file: Path | None = None, progress_sink: ProgressEmit | None = None) -> dict[str, Any]:
    s, self_path = progress_sink, Path(self_file or __file__).resolve()
    facade_path = _step(s, "CLI_LAUNCH_IDENTITY_BIND", lambda: _bind_launch_identity(packet, self_file=self_path))
    pins = packet["source_pins"]
    facade_pin = next(pin for pin in pins.values() if isinstance(pin, Mapping) and Path(str(pin.get("absolute_path", ""))).resolve() == facade_path)
    facade = _step(s, "CLI_FACADE_LOAD", lambda: _load_verified_facade(facade_path, str(facade_pin["sha256"])))
    if s is not None: s("CLI_VERIFIED_SESSION_ENTER", "start", None)
    try: cm = facade.signed_utility_fixed_state_session(_expected_module_map(pins)); bundle = cm.__enter__()
    except Exception as exc:
        if s is not None: s("CLI_VERIFIED_SESSION_ENTER", "error", f"{type(exc).__name__}:{exc}")
        raise
    try:
        if s is not None: s("CLI_VERIFIED_SESSION_ENTER", "done", None)
        def _ident():
            so_p = Path(bundle.verified_paths_by_module["support_only"]).resolve(); ag_p = Path(bundle.verified_paths_by_module["authoritative_gpu"]).resolve()
            if Path(bundle.support_only.__file__).resolve() != so_p: raise RuntimeError("support_only_identity_mismatch")
            if Path(bundle.authoritative_gpu.__file__).resolve() != ag_p: raise RuntimeError("authoritative_gpu_identity_mismatch")
        _step(s, "CLI_MODULE_IDENTITY_CHECKS", _ident)
        return bundle.support_only.run_support_only_characterization(packet, progress_sink=s)
    finally: cm.__exit__(*sys.exc_info())
def _fail_closed_sink(fd: int | None) -> int:
    if fd is not None:
        try: _rewrite_progress_sink_failure(fd)
        except Exception: pass  # noqa: BLE001
        _close_reserved(fd)
    print("SUPPORT_FAIL progress_sink_failure", file=sys.stderr, flush=True); print(SUPPORT_INTEGRITY, flush=True); return 2
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hrm_text_158_signed_utility_support_only_characterization")
    parser.add_argument("--packet", required=True); parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv); receipt = Path(args.receipt); print("SUPPORT_BEGIN", flush=True)
    sink = _guarded_sink(_build_progress_sink(time.monotonic_ns())); fd = None; held: list[int] = []
    try: fd = _step(sink, "CLI_RECEIPT_RESERVE", lambda: held.append(_reserve_receipt(receipt)) or held[0])
    except FileExistsError:
        print("SUPPORT_FAIL oexcl_collision", file=sys.stderr, flush=True); return 2
    except ProgressSinkFailure: return _fail_closed_sink(held[0] if held else None)
    except Exception as exc:  # noqa: BLE001
        print(f"SUPPORT_FAIL receipt_reserve:{exc}", file=sys.stderr, flush=True); return 2
    try:
        try: packet = _step(sink, "CLI_PACKET_LOAD", lambda: json.loads(Path(args.packet).read_text(encoding="utf-8")))
        except ProgressSinkFailure: return _fail_closed_sink(fd)
        except Exception as exc:  # noqa: BLE001
            print(f"SUPPORT_FAIL packet_load:{exc}", file=sys.stderr, flush=True); result = _canonical_fail(f"packet_load:{exc}")
        else:
            try:
                result = _run_via_verified_session(packet, self_file=Path(__file__), progress_sink=sink)
                for step in result.get("route") or []: print(f"SUPPORT_PHASE_{str(step).upper()}", flush=True)
            except ProgressSinkFailure: return _fail_closed_sink(fd)
            except Exception as exc:  # noqa: BLE001
                result = _canonical_fail(f"{type(exc).__name__}:{exc}")
        try: _step(sink, "CLI_RECEIPT_WRITE", lambda: _write_reserved(fd, result))
        except ProgressSinkFailure: return _fail_closed_sink(fd)
        except Exception as exc:  # noqa: BLE001
            print(f"SUPPORT_FAIL receipt_write:{exc}", file=sys.stderr, flush=True); _close_reserved(fd); return 2
    except ProgressSinkFailure: return _fail_closed_sink(fd)
    except Exception as exc:  # noqa: BLE001
        print(f"SUPPORT_FAIL receipt_write:{exc}", file=sys.stderr, flush=True); _close_reserved(fd); return 2
    _close_reserved(fd); cls = str(result.get("classifier") or SUPPORT_INTEGRITY)
    print(cls, flush=True); return 0 if cls == SUPPORT_ELIGIBLE else 2
if __name__ == "__main__":
    raise SystemExit(main())
