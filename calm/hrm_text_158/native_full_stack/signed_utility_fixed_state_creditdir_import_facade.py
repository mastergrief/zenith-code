"""One named creditdir import facade with fail-closed path/hash pins (PLAN v5)."""
from __future__ import annotations

import hashlib, importlib.util, sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping

REPO_ROOT = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
MODULE_REL_PATHS = {
    "reducers": "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_reducers.py",
    "schema": "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_schema.py",
    "pin_validation": "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_pin_validation.py",
    "driver": "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_driver.py",
    "facade": "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_facade.py",
}
MODULE_IMPORT_NAMES = {
    "reducers": "calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers",
    "schema": "calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema",
    "pin_validation": "calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation",
    "driver": "calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_driver",
    "facade": "calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_facade",
}
_LOAD_ORDER = ("reducers", "schema", "pin_validation", "driver", "facade")
FACADE_REASON = "creditdir must load signed-utility modules only via this facade with path+sha binding"


class ImportFacadeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModuleBundle:
    reducers: ModuleType
    schema: ModuleType
    pin_validation: ModuleType
    driver: ModuleType
    facade: ModuleType
    observed_sha256_by_module: Mapping[str, str]
    verified_paths_by_module: Mapping[str, str]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_expected_sha256_by_module(
    expected_sha256_by_module: Mapping[str, str], *, repo_root: Path = REPO_ROOT
) -> dict[str, tuple[str, Path]]:
    """Pre-import hash check against exact repo_root files; no import side effects."""
    if set(expected_sha256_by_module) != set(MODULE_REL_PATHS):
        raise ImportFacadeError(f"expected_keys_mismatch:{sorted(expected_sha256_by_module)}")
    out: dict[str, tuple[str, Path]] = {}
    root = Path(repo_root).resolve()
    for key, rel in MODULE_REL_PATHS.items():
        path = (root / rel).resolve()
        if not path.is_file():
            raise ImportFacadeError(f"module_path_missing:{key}:{path}")
        digest = _sha(path)
        if digest != str(expected_sha256_by_module[key]):
            raise ImportFacadeError(f"module_sha_mismatch:{key}:{digest}!={expected_sha256_by_module[key]}")
        out[key] = (digest, path)
    return out


def _purge() -> None:
    for name in MODULE_IMPORT_NAMES.values():
        sys.modules.pop(name, None)


def _load_verified(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportFacadeError(f"spec_from_file_location_failed:{name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    loaded = Path(getattr(mod, "__file__", "") or "").resolve()
    if loaded != path.resolve():
        raise ImportFacadeError(f"path_not_bound:{name}:{loaded}!={path.resolve()}")
    return mod


def load_signed_utility_fixed_state_modules(
    expected_sha256_by_module: Mapping[str, str], *, repo_root: Path = REPO_ROOT
) -> ModuleBundle:
    observed = verify_expected_sha256_by_module(expected_sha256_by_module, repo_root=repo_root)
    _purge()
    mods: dict[str, ModuleType] = {}
    try:
        for key in _LOAD_ORDER:
            mods[key] = _load_verified(MODULE_IMPORT_NAMES[key], observed[key][1])
    except Exception:
        _purge()
        raise
    return ModuleBundle(
        reducers=mods["reducers"], schema=mods["schema"], pin_validation=mods["pin_validation"],
        driver=mods["driver"], facade=mods["facade"],
        observed_sha256_by_module={k: observed[k][0] for k in MODULE_REL_PATHS},
        verified_paths_by_module={k: str(observed[k][1]) for k in MODULE_REL_PATHS},
    )


__all__ = [
    "FACADE_REASON", "ImportFacadeError", "MODULE_IMPORT_NAMES", "MODULE_REL_PATHS",
    "ModuleBundle", "REPO_ROOT", "load_signed_utility_fixed_state_modules",
    "verify_expected_sha256_by_module",
]
