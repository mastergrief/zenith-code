"""Thin CLI: stdlib-only bootstrap → verified creditdir import facade session (D2c9)."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
from typing import Any, Mapping

ESTIMAND = "full_state_legal_subset_signed_direction_fixed_state_heldout_utility"
SUPPORT_ELIGIBLE = "SUPPORT_ELIGIBLE"
SUPPORT_INTEGRITY = "SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE"
MAX_RESULT = 256 * 1024
FACADE_BASENAME = "signed_utility_fixed_state_creditdir_import_facade.py"
FACADE_REL = Path("calm/hrm_text_158/native_full_stack") / FACADE_BASENAME
WATCH_REL = Path("bin/watch-wrap")
WATCH_WRAP_HRM158_SHA256 = "a19f1c5fe88fb3dcbf00ab442047576708f75272210e9a0cc94ed9369bf45d4b"
MODULE_BASENAME_TO_KEY = {
    "signed_utility_fixed_state_reducers.py": "reducers",
    "signed_utility_fixed_state_schema.py": "schema",
    "signed_utility_fixed_state_pin_validation.py": "pin_validation",
    "signed_utility_fixed_state_phase_telemetry.py": "phase_telemetry",
    "signed_utility_fixed_state_integrity_proofs.py": "integrity_proofs",
    "signed_utility_fixed_state_partition_leakage.py": "partition_leakage",
    "signed_utility_fixed_state_arm_proofs.py": "arm_proofs",
    "signed_utility_fixed_state_legal_subset.py": "legal_subset",
    "signed_utility_fixed_state_eval_contract.py": "eval_contract",
    "signed_utility_fixed_state_authoritative_gpu.py": "authoritative_gpu",
    "signed_utility_fixed_state_support_only.py": "support_only",
    "signed_utility_fixed_state_driver.py": "driver",
    "signed_utility_fixed_state_facade.py": "facade",
}

def _canonical_fail(reason: str) -> dict[str, Any]:
    payload = {
        "schema": "support_only_terminal_v1", "estimand": ESTIMAND, "classifier": SUPPORT_INTEGRITY,
        "reason": reason, "route": [], "claim_ceiling": "support_eligibility_only",
        "parent_sha_pre": None, "parent_sha_post": None, "source_sha_pre": {}, "source_sha_post": {},
        "launch_surface_sha_pre": {}, "launch_surface_sha_post": {},
    }
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

def _close_reserved(fd: int | None) -> None:
    if fd is None: return
    try: os.close(fd)
    except OSError: pass

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

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

def _run_via_verified_session(packet: Mapping[str, Any], *, self_file: Path | None = None) -> dict[str, Any]:
    self_path = Path(self_file or __file__).resolve()
    facade_path = _bind_launch_identity(packet, self_file=self_path)
    pins = packet["source_pins"]
    facade_pin = next(pin for pin in pins.values()
                      if isinstance(pin, Mapping) and Path(str(pin.get("absolute_path", ""))).resolve() == facade_path)
    facade = _load_verified_facade(facade_path, str(facade_pin["sha256"]))
    with facade.signed_utility_fixed_state_session(_expected_module_map(pins)) as bundle:
        so_path = Path(bundle.verified_paths_by_module["support_only"]).resolve()
        ag_path = Path(bundle.verified_paths_by_module["authoritative_gpu"]).resolve()
        if Path(bundle.support_only.__file__).resolve() != so_path: raise RuntimeError("support_only_identity_mismatch")
        if Path(bundle.authoritative_gpu.__file__).resolve() != ag_path: raise RuntimeError("authoritative_gpu_identity_mismatch")
        return bundle.support_only.run_support_only_characterization(packet)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hrm_text_158_signed_utility_support_only_characterization")
    parser.add_argument("--packet", required=True); parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    receipt = Path(args.receipt); print("SUPPORT_BEGIN", flush=True)
    fd = None
    try:
        fd = _reserve_receipt(receipt)
    except FileExistsError:
        print("SUPPORT_FAIL oexcl_collision", file=sys.stderr, flush=True); return 2
    except Exception as exc:  # noqa: BLE001
        print(f"SUPPORT_FAIL receipt_reserve:{exc}", file=sys.stderr, flush=True); return 2
    try:
        try:
            packet = json.loads(Path(args.packet).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"SUPPORT_FAIL packet_load:{exc}", file=sys.stderr, flush=True)
            result = _canonical_fail(f"packet_load:{exc}")
        else:
            try:
                result = _run_via_verified_session(packet, self_file=Path(__file__))
                for step in result.get("route") or []: print(f"SUPPORT_PHASE_{str(step).upper()}", flush=True)
            except Exception as exc:  # noqa: BLE001
                result = _canonical_fail(f"{type(exc).__name__}:{exc}")
        _write_reserved(fd, result)
    except Exception as exc:  # noqa: BLE001
        print(f"SUPPORT_FAIL receipt_write:{exc}", file=sys.stderr, flush=True)
        _close_reserved(fd); return 2
    _close_reserved(fd)
    cls = str(result.get("classifier") or SUPPORT_INTEGRITY)
    print(cls, flush=True); return 0 if cls == SUPPORT_ELIGIBLE else 2

if __name__ == "__main__":
    raise SystemExit(main())
