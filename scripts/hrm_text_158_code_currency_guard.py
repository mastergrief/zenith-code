"""Phase-3b code-freshness guard: pycache invalidation + executed-code self-check."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

CODE_CURRENCY_MISMATCH_TERMINAL = "CODE_CURRENCY_MISMATCH_INCONCLUSIVE"
CODE_CURRENCY_MISMATCH_EXIT_CODE = 37
CODE_CURRENCY_EXECUTED_MISMATCH_TERMINAL = "CODE_CURRENCY_EXECUTED_MISMATCH_INCONCLUSIVE"
CODE_CURRENCY_GUARD_NOT_RUN_TERMINAL = "CODE_CURRENCY_GUARD_NOT_RUN_INCONCLUSIVE"

EXECUTED_GUARD_SCHEMA = "hrm_text_158_code_currency_executed_guard/v1"
IMPORT_BYTE_GUARD_SCHEMA = "hrm_text_158_code_currency_import_byte_check/v1"
EXECUTED_GUARD_PASSED_ENV = "HRM_TEXT_158_CODE_CURRENCY_EXECUTED_GUARD_PASSED"
EXECUTED_GUARD_PROOF_METHOD = "reachable_code_object_fingerprint_method_a"

SKIP_IMPORT_BYTE_CHECK_ENV = "HRM_TEXT_158_SKIP_CODE_CURRENCY_IMPORT_BYTE_CHECK"
IMPORT_BYTE_PINS_ENV = "HRM_TEXT_158_CODE_CURRENCY_IMPORT_BYTE_PINS"
OBMALLOC_EXPANDED_ENV = "HRM_TEXT_158_PROFILE_OBMALLOC_EXPANDED"
PROFILE_HOST_RSS_ENV = "HRM_TEXT_158_PROFILE_HOST_RSS"
PROFILE_TRACEMALLOC_ENV = "HRM_TEXT_158_PROFILE_TRACEMALLOC"
PROFILE_DEBUGMALLOCSTATS_ENV = "HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS"

PHASE3B_PINNED_SOURCE_FILES: dict[str, str] = {
    "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py": (
        "fe62aeacf4bcf541f11c60e985b1b9c149a8e3c36df175e2f04ce5c3a45f723a"
    ),
    "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py": (
        "5169e9f9152c39936140ef97c56d4dbf067775632581cb62d4d3a9debcf5c181"
    ),
    "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py": (
        "31ea9d1794ce3086b5204a7c3397b84e7cad9a960d0d4d568b4029523f0b6cc3"
    ),
    "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py": (
        "1eb3149091e2fddb39106c11e5bb8833325e90b64ca91ad4014663f1fea2d383"
    ),
    "calm/hrm_text_158/native_full_stack/host_tracemalloc_probe.py": (
        "2a2d9c623dafb5cf573e445cdf2fc30a8f8ff60cdabe0a771540e316830da5ab"
    ),
    "calm/hrm_text_158/native_full_stack/s1d7_tracemalloc_feasibility.py": (
        "76f2ddd28cac8b29c9ab451d2a69e256fee9ca15c75629e08a70029c472bf346"
    ),
    "scripts/hrm_text_158_slice5_v6i_oom_profile_attribution.py": (
        "c6da2c8221da6f88583a1d15aa1d9f9e751bc90bfab12d1129cf80e3ed51d141"
    ),
}

PHASE3B_PROBE_IMPORT_MODULE_BY_REL: dict[str, str] = {
    "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py": (
        "calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier"
    ),
    "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py": (
        "calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter"
    ),
    "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py": (
        "calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter"
    ),
    "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py": (
        "calm.hrm_text_158.native_full_stack.bounded_delta_learner"
    ),
    "calm/hrm_text_158/native_full_stack/host_tracemalloc_probe.py": (
        "calm.hrm_text_158.native_full_stack.host_tracemalloc_probe"
    ),
    "calm/hrm_text_158/native_full_stack/s1d7_tracemalloc_feasibility.py": (
        "calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility"
    ),
}

PINNED_PROBE_MODULE_NAMES: tuple[str, ...] = tuple(PHASE3B_PROBE_IMPORT_MODULE_BY_REL.values())

PHASE3B_PROBE_IMPORT_BYTE_PINS: dict[str, str] = {
    rel: PHASE3B_PINNED_SOURCE_FILES[rel]
    for rel in PHASE3B_PROBE_IMPORT_MODULE_BY_REL
}

PHASE3B_PYCACHE_INVALIDATION_PATHS: tuple[str, ...] = tuple(
    list(PHASE3B_PROBE_IMPORT_MODULE_BY_REL)
    + ["scripts/hrm_text_158_code_currency_guard.py"]
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def hash_file_bytes(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> str:
    payload = path.read_bytes() if read_bytes is None else read_bytes(path)
    return hashlib.sha256(payload).hexdigest()


def _serialize_co_const_value(value: Any) -> bytes:
    if isinstance(value, types.CodeType):
        raise TypeError("nested CodeType must be excluded from co_consts serialization")
    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"bool:" + (b"true" if value else b"false")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"float:" + repr(value).encode("ascii")
    if isinstance(value, complex):
        return b"complex:" + repr(value).encode("ascii")
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, tuple):
        parts = [_serialize_co_const_value(item) for item in value]
        return b"tuple:" + b"|".join(parts)
    if isinstance(value, frozenset):
        parts = sorted(_serialize_co_const_value(item) for item in value)
        return b"frozenset:" + b"|".join(parts)
    raise TypeError(f"unsupported co_const type: {type(value)!r}")


def _serialize_co_consts_non_code(code: types.CodeType) -> bytes:
    parts: list[bytes] = []
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            continue
        parts.append(_serialize_co_const_value(const))
    return b";".join(parts)


def _code_object_signature_co_consts_blind(code: types.CodeType) -> bytes:
    """Pre-fix fingerprint that ignored non-code co_consts (test-only baseline)."""
    hasher = hashlib.sha256()
    hasher.update(code.co_code)
    hasher.update(repr(code.co_names).encode("utf-8"))
    hasher.update(repr(code.co_varnames).encode("utf-8"))
    hasher.update(str(code.co_firstlineno).encode("utf-8"))
    qualname = getattr(code, "co_qualname", code.co_name)
    hasher.update(str(qualname).encode("utf-8"))
    return hasher.digest()


def _code_object_signature(code: types.CodeType) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_code_object_signature_co_consts_blind(code))
    hasher.update(_serialize_co_consts_non_code(code))
    return hasher.digest()


def _iter_nested_code_objects(code: types.CodeType) -> list[types.CodeType]:
    seen: set[types.CodeType] = set()
    ordered: list[types.CodeType] = []

    def visit(current: types.CodeType) -> None:
        if current in seen:
            return
        seen.add(current)
        ordered.append(current)
        for const in current.co_consts:
            if isinstance(const, types.CodeType):
                visit(const)

    visit(code)
    return ordered


def _iter_reachable_code_objects_from_module(module: types.ModuleType) -> list[types.CodeType]:
    seen: set[types.CodeType] = set()
    ordered: list[types.CodeType] = []

    def visit_code(code: types.CodeType) -> None:
        for nested in _iter_nested_code_objects(code):
            if nested in seen:
                continue
            seen.add(nested)
            ordered.append(nested)

    def visit_object(obj: Any) -> None:
        if inspect.isroutine(obj):
            code = getattr(obj, "__code__", None)
            if isinstance(code, types.CodeType):
                visit_code(code)
        elif inspect.isclass(obj):
            for attr in obj.__dict__.values():
                visit_object(attr)

    for value in module.__dict__.values():
        visit_object(value)
    return ordered


def fingerprint_reachable_code_objects(module: types.ModuleType) -> str:
    codes = _iter_reachable_code_objects_from_module(module)
    signatures = sorted(_code_object_signature(code) for code in codes)
    hasher = hashlib.sha256()
    for signature in signatures:
        hasher.update(signature)
    return hasher.hexdigest()


def expected_reachable_fingerprint_from_source(
    source: str,
    *,
    module_name: str = "pinned_expected_module",
) -> str:
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(source, "<pinned-source>", "exec"), module.__dict__)
        return fingerprint_reachable_code_objects(module)
    finally:
        sys.modules.pop(module_name, None)


def invalidate_pycache_for_source_files(
    repo_root: Path,
    rel_paths: Sequence[str],
) -> list[str]:
    removed: list[str] = []
    for rel in rel_paths:
        src = repo_root / rel
        if not src.is_file():
            continue
        cache_dir = src.parent / "__pycache__"
        if not cache_dir.is_dir():
            continue
        for pyc in sorted(cache_dir.glob(f"{src.stem}*.pyc")):
            pyc.unlink(missing_ok=True)
            removed.append(str(pyc))
    return removed


def import_module_for_rel_path(rel_path: str) -> Any:
    module_name = PHASE3B_PROBE_IMPORT_MODULE_BY_REL[str(rel_path)]
    return importlib.import_module(module_name)


def imported_module_file_path(module: Any) -> Path:
    file_path = getattr(module, "__file__", None)
    if not file_path:
        raise ValueError(f"module {module!r} has no __file__")
    return Path(file_path).resolve()


def pinned_modules_present_in_sys_modules(
    module_names: Sequence[str] = PINNED_PROBE_MODULE_NAMES,
) -> list[str]:
    return [name for name in module_names if name in sys.modules]


def verify_executed_code_for_pinned_modules(
    pins: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
    import_module_fn: Callable[[str], Any] = import_module_for_rel_path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for rel_path, expected_file_sha in pins.items():
        source_path = repo_root / rel_path
        source = source_path.read_text(encoding="utf-8")
        actual_file_sha = hash_file_bytes(source_path)
        if actual_file_sha != str(expected_file_sha):
            row = {
                "rel_path": str(rel_path),
                "kind": "source_file_sha_mismatch",
                "actual_sha256": actual_file_sha,
                "expected_sha256": str(expected_file_sha),
            }
            results.append(row)
            mismatches.append(row)
            continue
        expected_fp = expected_reachable_fingerprint_from_source(source)
        module = import_module_fn(str(rel_path))
        actual_fp = fingerprint_reachable_code_objects(module)
        row = {
            "rel_path": str(rel_path),
            "module_name": PHASE3B_PROBE_IMPORT_MODULE_BY_REL[str(rel_path)],
            "expected_executed_fingerprint": expected_fp,
            "actual_executed_fingerprint": actual_fp,
            "proof_method": EXECUTED_GUARD_PROOF_METHOD,
            "reachable_code_object_count": len(
                _iter_reachable_code_objects_from_module(module)
            ),
        }
        results.append(row)
        if actual_fp != expected_fp:
            mismatches.append({**row, "kind": "executed_fingerprint_mismatch"})
    return {
        "ok": not mismatches,
        "results": results,
        "mismatches": mismatches,
        "proof_method": EXECUTED_GUARD_PROOF_METHOD,
    }


def verify_imported_bytes_for_pinned_modules(
    pins: Mapping[str, str],
    *,
    enabled: bool = True,
    import_module_fn: Callable[[str], Any] = import_module_for_rel_path,
    hash_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {"skipped": True, "ok": True, "results": [], "mismatches": []}

    def _default_hash(path: Path) -> str:
        return hash_file_bytes(path)

    hasher = _default_hash if hash_fn is None else hash_fn
    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for rel_path, expected_sha in pins.items():
        module = import_module_fn(str(rel_path))
        imported_path = imported_module_file_path(module)
        actual_sha = hasher(imported_path)
        row = {
            "rel_path": str(rel_path),
            "imported_file": str(imported_path),
            "actual_sha256": actual_sha,
            "expected_sha256": str(expected_sha),
        }
        results.append(row)
        if actual_sha != str(expected_sha):
            mismatches.append(row)
    return {
        "skipped": False,
        "ok": not mismatches,
        "results": results,
        "mismatches": mismatches,
    }


def capture_probe_child_argv_records(
    post_script_cli_argv: Sequence[str] | None,
) -> dict[str, list[str]]:
    orig_argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    return {
        "child_orig_argv": orig_argv,
        "child_sys_argv": list(sys.argv),
        "child_post_script_cli_argv": list(post_script_cli_argv or []),
    }


def build_executed_guard_receipt(
    *,
    ok: bool,
    proof_method: str,
    executed_report: Mapping[str, Any],
    import_byte_report: Mapping[str, Any] | None,
    argv_records: Mapping[str, Sequence[str]],
    sys_modules_before: Sequence[str],
    sys_modules_after: Sequence[str],
    pycache_invalidated_paths: Sequence[str],
    fail_closed_terminal: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": EXECUTED_GUARD_SCHEMA,
        "ok": bool(ok),
        "proof_method": str(proof_method),
        "fail_closed_terminal": fail_closed_terminal,
        "process_exit_code": None if ok else CODE_CURRENCY_MISMATCH_EXIT_CODE,
        "mapped_terminal_code": None if ok else CODE_CURRENCY_MISMATCH_EXIT_CODE,
        "exit_code_agreement": True if ok else None,
        "child_orig_argv": list(argv_records.get("child_orig_argv") or []),
        "child_sys_argv": list(argv_records.get("child_sys_argv") or []),
        "child_post_script_cli_argv": list(
            argv_records.get("child_post_script_cli_argv") or []
        ),
        "env": {
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "HRM_TEXT_158_PROFILE_OBMALLOC_EXPANDED": os.environ.get(OBMALLOC_EXPANDED_ENV),
        },
        "sys_flags": {
            "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        },
        "sys_modules_before_guard": list(sys_modules_before),
        "sys_modules_after_guard": list(sys_modules_after),
        "guard_ran_before_pinned_imports": not bool(sys_modules_before),
        "pycache_invalidated_paths": list(pycache_invalidated_paths),
        "executed_code_fingerprints": list(executed_report.get("results") or []),
        "executed_code_mismatches": list(executed_report.get("mismatches") or []),
        "import_byte_fingerprints": (
            list(import_byte_report.get("results") or []) if import_byte_report else []
        ),
    }


def build_code_currency_mismatch_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": IMPORT_BYTE_GUARD_SCHEMA,
        "fail_closed_terminal": CODE_CURRENCY_MISMATCH_TERMINAL,
        "process_exit_code": CODE_CURRENCY_MISMATCH_EXIT_CODE,
        "mapped_terminal_code": CODE_CURRENCY_MISMATCH_EXIT_CODE,
        "exit_code_agreement": True,
        "mismatches": list(report.get("mismatches") or []),
        "checked_modules": list(report.get("results") or []),
    }


class CodeCurrencyMismatchError(SystemExit):
    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = dict(receipt)
        super().__init__(int(receipt.get("process_exit_code", CODE_CURRENCY_MISMATCH_EXIT_CODE)))


def enforce_import_byte_pins_or_fail_closed(
    pins: Mapping[str, str],
    *,
    enabled: bool = True,
    import_module_fn: Callable[[str], Any] = import_module_for_rel_path,
    hash_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    report = verify_imported_bytes_for_pinned_modules(
        pins,
        enabled=enabled,
        import_module_fn=import_module_fn,
        hash_fn=hash_fn,
    )
    if report.get("skipped"):
        return report
    if not report.get("ok"):
        raise CodeCurrencyMismatchError(build_code_currency_mismatch_receipt(report))
    return report


def load_import_byte_pins_from_env() -> dict[str, str] | None:
    raw = os.environ.get(IMPORT_BYTE_PINS_ENV, "").strip()
    if not raw:
        return None
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"{IMPORT_BYTE_PINS_ENV} must be a JSON object")
    return {str(k): str(v) for k, v in loaded.items()}


def resolve_probe_import_byte_pins() -> dict[str, str]:
    from_env = load_import_byte_pins_from_env()
    if from_env is not None:
        return from_env
    return dict(PHASE3B_PROBE_IMPORT_BYTE_PINS)


def _emit_guard_receipt(receipt: Mapping[str, Any]) -> None:
    print(json.dumps(dict(receipt), indent=2, sort_keys=True), flush=True)


def run_phase3b_probe_executed_code_currency_guard(
    *,
    argv: Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
    require_obmalloc_expanded: bool = True,
) -> int | None:
    if require_obmalloc_expanded and not _env_truthy(OBMALLOC_EXPANDED_ENV):
        return None
    if _env_truthy(SKIP_IMPORT_BYTE_CHECK_ENV):
        return None
    if _env_truthy(EXECUTED_GUARD_PASSED_ENV):
        return None

    argv_records = capture_probe_child_argv_records(argv)
    sys_modules_before = pinned_modules_present_in_sys_modules()
    if sys_modules_before:
        receipt = build_executed_guard_receipt(
            ok=False,
            proof_method=EXECUTED_GUARD_PROOF_METHOD,
            executed_report={"results": [], "mismatches": []},
            import_byte_report=None,
            argv_records=argv_records,
            sys_modules_before=sys_modules_before,
            sys_modules_after=list(sys_modules_before),
            pycache_invalidated_paths=[],
            fail_closed_terminal=CODE_CURRENCY_GUARD_NOT_RUN_TERMINAL,
        )
        receipt["process_exit_code"] = CODE_CURRENCY_MISMATCH_EXIT_CODE
        receipt["mapped_terminal_code"] = CODE_CURRENCY_MISMATCH_EXIT_CODE
        _emit_guard_receipt(receipt)
        return CODE_CURRENCY_MISMATCH_EXIT_CODE

    removed = invalidate_pycache_for_source_files(repo_root, PHASE3B_PYCACHE_INVALIDATION_PATHS)
    pins = resolve_probe_import_byte_pins()
    executed_report = verify_executed_code_for_pinned_modules(pins, repo_root=repo_root)
    import_byte_report = verify_imported_bytes_for_pinned_modules(pins, enabled=True)
    sys_modules_after = pinned_modules_present_in_sys_modules()
    ok = bool(executed_report.get("ok")) and bool(import_byte_report.get("ok"))
    fail_closed_terminal = None
    if not ok:
        fail_closed_terminal = (
            CODE_CURRENCY_EXECUTED_MISMATCH_TERMINAL
            if executed_report.get("mismatches")
            else CODE_CURRENCY_MISMATCH_TERMINAL
        )
    receipt = build_executed_guard_receipt(
        ok=ok,
        proof_method=EXECUTED_GUARD_PROOF_METHOD,
        executed_report=executed_report,
        import_byte_report=import_byte_report,
        argv_records=argv_records,
        sys_modules_before=sys_modules_before,
        sys_modules_after=sys_modules_after,
        pycache_invalidated_paths=removed,
        fail_closed_terminal=fail_closed_terminal,
    )
    if not ok:
        receipt["process_exit_code"] = CODE_CURRENCY_MISMATCH_EXIT_CODE
        receipt["mapped_terminal_code"] = CODE_CURRENCY_MISMATCH_EXIT_CODE
        _emit_guard_receipt(receipt)
        return CODE_CURRENCY_MISMATCH_EXIT_CODE

    _emit_guard_receipt(receipt)
    os.environ[EXECUTED_GUARD_PASSED_ENV] = "1"
    return None


def profile_callsite_tracemalloc_only_enabled() -> bool:
    # Env-only mirror of profile_tracemalloc_enabled (host_tracemalloc_probe.py:19-34)
    # and profile_debugmallocstats_enabled (host_allocator_probe.py:810-825).
    # If either probe fn gains non-env logic, update this predicate in the same commit.
    return (
        _env_truthy(PROFILE_HOST_RSS_ENV)
        and _env_truthy(PROFILE_TRACEMALLOC_ENV)
        and not _env_truthy(PROFILE_DEBUGMALLOCSTATS_ENV)
        and not _env_truthy(OBMALLOC_EXPANDED_ENV)
    )


def maybe_enforce_phase3b_probe_import_byte_currency() -> int | None:
    exit_code = run_phase3b_probe_executed_code_currency_guard(require_obmalloc_expanded=True)
    if exit_code is not None:
        return exit_code
    if profile_callsite_tracemalloc_only_enabled():
        return run_phase3b_probe_executed_code_currency_guard(require_obmalloc_expanded=False)
    return None


def prepare_phase3b_callsite_tracemalloc_launch_env(
    env: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    merged = dict(env)
    invalidate_pycache_for_source_files(repo_root, PHASE3B_PYCACHE_INVALIDATION_PATHS)
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    pin_payload = {
        rel: hash_file_bytes(repo_root / rel)
        for rel in PHASE3B_PROBE_IMPORT_MODULE_BY_REL
        if (repo_root / rel).is_file()
    }
    merged[IMPORT_BYTE_PINS_ENV] = json.dumps(pin_payload, sort_keys=True)
    merged.pop(EXECUTED_GUARD_PASSED_ENV, None)
    return merged


def prepare_phase3b_probe_launch_env(
    env: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
    expanded: bool,
) -> dict[str, str]:
    merged = dict(env)
    if not expanded:
        return merged
    invalidate_pycache_for_source_files(repo_root, PHASE3B_PYCACHE_INVALIDATION_PATHS)
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    pin_payload = {
        rel: hash_file_bytes(repo_root / rel)
        for rel in PHASE3B_PROBE_IMPORT_MODULE_BY_REL
        if (repo_root / rel).is_file()
    }
    merged[IMPORT_BYTE_PINS_ENV] = json.dumps(pin_payload, sort_keys=True)
    merged.pop(EXECUTED_GUARD_PASSED_ENV, None)
    return merged


def phase3b_probe_python_argv_prefix() -> list[str]:
    return ["-B"]


def phase3b_probe_script_path(*, expanded: bool) -> str:
    if expanded:
        return "scripts/hrm_text_158_bounded_delta_acquisition_probe_bootstrap.py"
    return "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
