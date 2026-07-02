"""Phase-3b code-freshness guard: pycache invalidation + imported-byte self-check."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

CODE_CURRENCY_MISMATCH_TERMINAL = "CODE_CURRENCY_MISMATCH_INCONCLUSIVE"
CODE_CURRENCY_MISMATCH_EXIT_CODE = 37

SKIP_IMPORT_BYTE_CHECK_ENV = "HRM_TEXT_158_SKIP_CODE_CURRENCY_IMPORT_BYTE_CHECK"
IMPORT_BYTE_PINS_ENV = "HRM_TEXT_158_CODE_CURRENCY_IMPORT_BYTE_PINS"
OBMALLOC_EXPANDED_ENV = "HRM_TEXT_158_PROFILE_OBMALLOC_EXPANDED"

PHASE3B_PINNED_SOURCE_FILES: dict[str, str] = {
    "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py": (
        "3286871f2906865ab69be676d290779fef72c088c750736f6761b5191a50d152"
    ),
    "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py": (
        "5169e9f9152c39936140ef97c56d4dbf067775632581cb62d4d3a9debcf5c181"
    ),
    "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py": (
        "31ea9d1794ce3086b5204a7c3397b84e7cad9a960d0d4d568b4029523f0b6cc3"
    ),
    "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py": (
        "b9d8f94cede2b31c695da09e37684fb5750b3607465d0cda2c45733258af83a9"
    ),
    "scripts/hrm_text_158_slice5_v6i_oom_profile_attribution.py": (
        "b559b75e0c68eacc08b795c05e0d21feee04d60674ae10b12d9edd7ef67bb901"
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
}

PHASE3B_PROBE_IMPORT_BYTE_PINS: dict[str, str] = {
    rel: PHASE3B_PINNED_SOURCE_FILES[rel]
    for rel in PHASE3B_PROBE_IMPORT_MODULE_BY_REL
}

PHASE3B_PYCACHE_INVALIDATION_PATHS: tuple[str, ...] = tuple(PHASE3B_PINNED_SOURCE_FILES)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def hash_file_bytes(path: Path, *, read_bytes: Callable[[Path], bytes] | None = None) -> str:
    payload = path.read_bytes() if read_bytes is None else read_bytes(path)
    return hashlib.sha256(payload).hexdigest()


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


def build_code_currency_mismatch_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hrm_text_158_code_currency_import_byte_check/v1",
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


def maybe_enforce_phase3b_probe_import_byte_currency() -> int | None:
    if not _env_truthy(OBMALLOC_EXPANDED_ENV):
        return None
    if _env_truthy(SKIP_IMPORT_BYTE_CHECK_ENV):
        return None
    pins = resolve_probe_import_byte_pins()
    try:
        report = enforce_import_byte_pins_or_fail_closed(pins, enabled=True)
    except CodeCurrencyMismatchError as exc:
        print(json.dumps(exc.receipt, indent=2, sort_keys=True), flush=True)
        return int(exc.receipt.get("process_exit_code", CODE_CURRENCY_MISMATCH_EXIT_CODE))
    print(
        json.dumps(
            {
                "schema": "hrm_text_158_code_currency_import_byte_check/v1",
                "ok": True,
                "checked_modules": report.get("results", []),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return None


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
    return merged


def phase3b_probe_python_argv_prefix() -> list[str]:
    return ["-B"]
