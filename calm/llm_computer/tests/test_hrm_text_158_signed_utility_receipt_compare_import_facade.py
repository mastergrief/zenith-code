"""CPU-static tests for dedicated receipt-compare facade loader (PLAN v16)."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack import (
    signed_utility_fixed_state_creditdir_import_facade as facade,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
COMPARE_REL = facade.RECEIPT_COMPARE_REL_PATH
COMPARE_NAME = facade.RECEIPT_COMPARE_IMPORT_NAME
COMPARE_PATH = REPO / COMPARE_REL


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _cleanup_compare_name():
    sys.modules.pop(COMPARE_NAME, None)
    yield
    sys.modules.pop(COMPARE_NAME, None)


def test_facade_loc_guard():
    lines = (REPO / "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_creditdir_import_facade.py").read_text().count("\n")
    assert lines <= 360


def test_helper_loc_guard():
    lines = COMPARE_PATH.read_text().count("\n")
    assert lines <= 150


def test_module_rel_paths_still_exactly_13():
    assert len(facade.MODULE_REL_PATHS) == 13
    assert len(facade.MODULE_IMPORT_NAMES) == 13
    assert COMPARE_REL not in facade.MODULE_REL_PATHS.values()
    assert "receipt_pre_post_compare" not in facade.MODULE_REL_PATHS
    assert COMPARE_NAME not in facade.MODULE_IMPORT_NAMES.values()


def test_verify_expected_still_keyset_strict():
    with pytest.raises(facade.ImportFacadeError, match="expected_keys_mismatch"):
        facade.verify_expected_sha256_by_module({"reducers": "0" * 64})


def test_clean_success_load():
    digest = _sha(COMPARE_PATH)
    mod = facade.load_receipt_pre_post_compare(digest, repo_root=REPO)
    assert sys.modules[COMPARE_NAME] is mod
    assert Path(mod.__file__).resolve() == COMPARE_PATH.resolve()
    assert callable(mod.pre_post_compare_hash) and callable(mod.pre_post_compare_git)
    r = mod.pre_post_compare_hash({"path": "/x", "sha256": "a" * 64}, {"path": "/x", "sha256": "a" * 64})
    assert r["pre_matches_post"] is True


def test_wrong_hash_absent_initially():
    assert COMPARE_NAME not in sys.modules
    with pytest.raises(facade.ImportFacadeError, match="module_sha_mismatch"):
        facade.load_receipt_pre_post_compare("0" * 64, repo_root=REPO)
    assert COMPARE_NAME not in sys.modules


def test_preseeded_wrong_hash_leaves_absent():
    stale = types.ModuleType(COMPARE_NAME)
    sys.modules[COMPARE_NAME] = stale
    with pytest.raises(facade.ImportFacadeError, match="module_sha_mismatch"):
        facade.load_receipt_pre_post_compare("0" * 64, repo_root=REPO)
    assert COMPARE_NAME not in sys.modules


def test_preseeded_missing_path_leaves_absent(tmp_path: Path):
    stale = types.ModuleType(COMPARE_NAME)
    sys.modules[COMPARE_NAME] = stale
    with pytest.raises(facade.ImportFacadeError, match="module_path_missing"):
        facade.load_receipt_pre_post_compare("0" * 64, repo_root=tmp_path)
    assert COMPARE_NAME not in sys.modules


def test_reload_after_file_change(tmp_path: Path):
    dest = tmp_path / COMPARE_REL
    dest.parent.mkdir(parents=True)
    body1 = "def pre_post_compare_hash(*a, **k):\n    return {'v': 1}\ndef pre_post_compare_git(*a, **k):\n    return {'v': 1}\n"
    dest.write_text(body1)
    d1 = _sha(dest)
    m1 = facade.load_receipt_pre_post_compare(d1, repo_root=tmp_path)
    assert m1.pre_post_compare_hash()["v"] == 1
    body2 = "def pre_post_compare_hash(*a, **k):\n    return {'v': 2}\ndef pre_post_compare_git(*a, **k):\n    return {'v': 2}\n"
    dest.write_text(body2)
    d2 = _sha(dest)
    m2 = facade.load_receipt_pre_post_compare(d2, repo_root=tmp_path)
    assert m2 is sys.modules[COMPARE_NAME]
    assert m2.pre_post_compare_hash()["v"] == 2
    assert m1 is not m2


def test_missing_public_api_leaves_absent(tmp_path: Path):
    dest = tmp_path / COMPARE_REL
    dest.parent.mkdir(parents=True)
    dest.write_text("X = 1\n")
    digest = _sha(dest)
    with pytest.raises(facade.ImportFacadeError, match="missing_public_api"):
        facade.load_receipt_pre_post_compare(digest, repo_root=tmp_path)
    assert COMPARE_NAME not in sys.modules


def test_preseeded_exec_failure_leaves_absent(tmp_path: Path):
    stale = types.ModuleType(COMPARE_NAME)
    sys.modules[COMPARE_NAME] = stale
    dest = tmp_path / COMPARE_REL
    dest.parent.mkdir(parents=True)
    dest.write_text("raise RuntimeError('exec_boom')\n")
    digest = _sha(dest)
    with pytest.raises(RuntimeError, match="exec_boom"):
        facade.load_receipt_pre_post_compare(digest, repo_root=tmp_path)
    assert COMPARE_NAME not in sys.modules


def test_preseeded_path_bind_failure_leaves_absent(tmp_path: Path):
    stale = types.ModuleType(COMPARE_NAME)
    sys.modules[COMPARE_NAME] = stale
    dest = tmp_path / COMPARE_REL
    dest.parent.mkdir(parents=True)
    dest.write_text(
        "def pre_post_compare_hash(*a, **k):\n    return {}\n"
        "def pre_post_compare_git(*a, **k):\n    return {}\n"
        "__file__ = '/nonexistent/path_bind_failure.py'\n"
    )
    digest = _sha(dest)
    with pytest.raises(facade.ImportFacadeError, match="path_not_bound"):
        facade.load_receipt_pre_post_compare(digest, repo_root=tmp_path)
    assert COMPARE_NAME not in sys.modules


def test_science_module_identities_unchanged_across_dedicated_load():
    # Capture any pre-existing science module objects (may be absent).
    before = {n: sys.modules.get(n) for n in facade.MODULE_IMPORT_NAMES.values()}
    facade.load_receipt_pre_post_compare(_sha(COMPARE_PATH), repo_root=REPO)
    after = {n: sys.modules.get(n) for n in facade.MODULE_IMPORT_NAMES.values()}
    assert before == after


def test_does_not_call_verify_expected(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("verify_expected_sha256_by_module must not be called")

    monkeypatch.setattr(facade, "verify_expected_sha256_by_module", boom)
    facade.load_receipt_pre_post_compare(_sha(COMPARE_PATH), repo_root=REPO)
