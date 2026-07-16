"""Creditdir import facade: path+sha pins + clean-revision session (PLAN v3)."""
from __future__ import annotations

import hashlib, importlib.util, io, subprocess, sys, tarfile, tempfile, threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Mapping

REPO_ROOT = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
CLEAN_REVISION_PIN = "3a85d0e8705325dbb26a3bca28d3d0e1ac7af2e7"
CLEAN_TREE_PIN = "00e8f1c6c3538b35ab4b53f6b704acb2e7afb65b"
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
DRIFTED_TRANSITIVE_PROOF_SET = (
    "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py",
    "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py",
)
FACADE_REASON = "creditdir loads signed-utility only via clean-revision session + path/sha binding"
_LOCK = threading.Lock()
_ACTIVE = False


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


@dataclass(frozen=True)
class SessionBundle(ModuleBundle):
    snapshot_root: Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_expected_sha256_by_module(
    expected_sha256_by_module: Mapping[str, str], *, repo_root: Path = REPO_ROOT
) -> dict[str, tuple[str, Path]]:
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


def _purge_names(names: set[str]) -> None:
    for name in list(names):
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
    names = set(MODULE_IMPORT_NAMES.values())
    _purge_names(names)
    mods: dict[str, ModuleType] = {}
    try:
        for key in _LOAD_ORDER:
            mods[key] = _load_verified(MODULE_IMPORT_NAMES[key], observed[key][1])
    except Exception:
        _purge_names(names)
        raise
    return ModuleBundle(
        reducers=mods["reducers"], schema=mods["schema"], pin_validation=mods["pin_validation"],
        driver=mods["driver"], facade=mods["facade"],
        observed_sha256_by_module={k: observed[k][0] for k in MODULE_REL_PATHS},
        verified_paths_by_module={k: str(observed[k][1]) for k in MODULE_REL_PATHS},
    )


def _git_tree(repo: Path, revision: str) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{tree}}"],
        check=True, capture_output=True, text=True,
    )
    return p.stdout.strip()


def _validate_tar_member(member: tarfile.TarInfo, dest: Path) -> None:
    name = member.name.replace("\\", "/")
    if name.startswith("/") or any(p == ".." for p in Path(name).parts):
        raise ImportFacadeError(f"tar_unsafe:{name}")
    if member.issym() or member.islnk():
        raise ImportFacadeError(f"tar_link_forbidden:{name}")
    if not (member.isreg() or member.isdir()):
        raise ImportFacadeError(f"tar_special_forbidden:{name}")
    target = (dest / name).resolve()
    root = dest.resolve()
    if target != root and not str(target).startswith(str(root) + "/"):
        raise ImportFacadeError(f"tar_escape:{name}")


def _extract_archive(repo: Path, revision: str, dest: Path) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", revision],
        check=True, capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
        for member in tf.getmembers():
            _validate_tar_member(member, dest)
            tf.extract(member, path=dest, set_attrs=False, filter="data")


def _pkg_paths(items: list[tuple[str, ModuleType]]) -> dict[str, object]:
    return {k: getattr(m, "__path__", None) for k, m in items if m is not None}


def _restore_modules_exact(pre_items: list[tuple[str, ModuleType]], pre_paths: Mapping[str, object]) -> None:
    """Restore preexisting sys.modules identity and insertion order; drop session calm.* adds."""
    pre_map = dict(pre_items)
    leftovers = [(k, sys.modules[k]) for k in list(sys.modules) if k not in pre_map and not (k == "calm" or k.startswith("calm."))]
    sys.modules.clear()
    for k, mod in pre_items:
        sys.modules[k] = mod
        if mod is not None and k in pre_paths and pre_paths[k] is not None:
            mod.__path__ = pre_paths[k]  # type: ignore[attr-defined]
    for k, mod in leftovers:
        sys.modules[k] = mod


def _note(primary: BaseException, label: str, err: BaseException) -> None:
    if hasattr(primary, "add_note"):
        primary.add_note(f"{label}:{err!r}")


def _restore_and_cleanup(
    *,
    path_list: list[str],
    pre_path: list[str] | None,
    pre_items: list[tuple[str, ModuleType]] | None,
    pre_paths: dict[str, object] | None,
    tmp,
    primary: BaseException | None,
) -> None:
    """Always restore import state + cleanup tmp; notes aux errors on primary."""
    aux: BaseException | None = None
    try:
        if pre_path is not None:
            path_list[:] = pre_path
        if pre_items is not None and pre_paths is not None:
            _restore_modules_exact(pre_items, pre_paths)
    except BaseException as e:
        aux = e
    try:
        if tmp is not None:
            tmp.cleanup()
    except BaseException as e:
        if primary is not None:
            _note(primary, "session_cleanup_error", e)
        elif aux is None:
            aux = e
        else:
            _note(aux, "session_cleanup_error", e)
    if primary is not None:
        if aux is not None:
            _note(primary, "session_restore_error", aux)
    elif aux is not None:
        raise aux


@contextmanager
def signed_utility_fixed_state_session(
    expected_sha256_by_module: Mapping[str, str],
    *,
    repo_git_dir: Path = REPO_ROOT,
    revision: str = CLEAN_REVISION_PIN,
    expected_tree: str = CLEAN_TREE_PIN,
) -> Iterator[SessionBundle]:
    global _ACTIVE
    if not _LOCK.acquire(blocking=False):
        raise ImportFacadeError("session_busy")
    # Outermost release begins immediately after acquire (before _ACTIVE / snapshots).
    owned = False
    primary: BaseException | None = None
    path_list = sys.path
    pre_path: list[str] | None = None
    pre_items: list[tuple[str, ModuleType]] | None = None
    pre_paths: dict[str, object] | None = None
    tmp = None
    try:
        if _ACTIVE:
            raise ImportFacadeError("nested_session_forbidden")
        _ACTIVE = True
        owned = True
        pre_path = list(path_list)
        pre_items = list(sys.modules.items())
        pre_paths = _pkg_paths(pre_items)
        # One lifecycle guard covers all post-capture setup + yield.
        try:
            repo = Path(repo_git_dir).resolve()
            tree = _git_tree(repo, revision)
            if tree != expected_tree:
                raise ImportFacadeError(f"tree_pin_mismatch:{tree}!={expected_tree}")
            tmp = tempfile.TemporaryDirectory(prefix="su_fs_snap_")
            snap = Path(tmp.name).resolve()
            _extract_archive(repo, revision, snap)
            observed = verify_expected_sha256_by_module(expected_sha256_by_module, repo_root=snap)
            path_list[:] = [str(snap), *[p for p in pre_path if Path(p).resolve() != snap]]
            _purge_names({k for k in sys.modules if k == "calm" or k.startswith("calm.")})
            mods: dict[str, ModuleType] = {}
            for key in _LOAD_ORDER:
                mods[key] = _load_verified(MODULE_IMPORT_NAMES[key], observed[key][1])
            yield SessionBundle(
                reducers=mods["reducers"], schema=mods["schema"], pin_validation=mods["pin_validation"],
                driver=mods["driver"], facade=mods["facade"],
                observed_sha256_by_module={k: observed[k][0] for k in MODULE_REL_PATHS},
                verified_paths_by_module={k: str(observed[k][1]) for k in MODULE_REL_PATHS},
                snapshot_root=snap,
            )
        except BaseException as e:
            primary = e
            raise
        finally:
            _restore_and_cleanup(
                path_list=path_list, pre_path=pre_path, pre_items=pre_items,
                pre_paths=pre_paths, tmp=tmp, primary=primary,
            )
    finally:
        if owned:
            _ACTIVE = False
        _LOCK.release()


__all__ = [
    "CLEAN_REVISION_PIN", "CLEAN_TREE_PIN", "DRIFTED_TRANSITIVE_PROOF_SET", "FACADE_REASON",
    "ImportFacadeError", "MODULE_IMPORT_NAMES", "MODULE_REL_PATHS", "ModuleBundle", "REPO_ROOT",
    "SessionBundle", "load_signed_utility_fixed_state_modules", "signed_utility_fixed_state_session",
    "verify_expected_sha256_by_module",
]
