"""CPU-static tests for terminal-consumer facade loader (PLAN v28 Step A)."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack import (
    signed_utility_fixed_state_creditdir_import_facade as facade,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
CONSUMER_REL = facade.TERMINAL_CONSUMER_REL_PATH
CONSUMER_NAME = facade.TERMINAL_CONSUMER_IMPORT_NAME
CONSUMER_PATH = REPO / CONSUMER_REL
FACADE_PATH = REPO / "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_creditdir_import_facade.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _cleanup_consumer_name():
    sys.modules.pop(CONSUMER_NAME, None)
    yield
    sys.modules.pop(CONSUMER_NAME, None)


def test_facade_post_edit_loc_cap():
    assert len(FACADE_PATH.read_text().splitlines()) <= 500


def test_consumer_loc_cap():
    assert len(CONSUMER_PATH.read_text().splitlines()) <= 250


def test_module_rel_paths_untouched_by_terminal_consumer():
    assert len(facade.MODULE_REL_PATHS) == 13
    assert len(facade.MODULE_IMPORT_NAMES) == 13
    assert CONSUMER_REL not in facade.MODULE_REL_PATHS.values()
    assert "terminal_consumer" not in facade.MODULE_REL_PATHS
    assert CONSUMER_NAME not in facade.MODULE_IMPORT_NAMES.values()
    assert facade.RECEIPT_COMPARE_REL_PATH not in facade.MODULE_REL_PATHS.values()


def test_clean_success_load():
    digest = _sha(CONSUMER_PATH)
    mod = facade.load_terminal_receipt_consumer(digest, repo_root=REPO)
    assert sys.modules[CONSUMER_NAME] is mod
    assert Path(mod.__file__).resolve() == CONSUMER_PATH.resolve()
    assert callable(mod.support_trichotomy_from_bytes) and callable(mod.cross_check_pair_receipt)
    out = mod.support_trichotomy_from_bytes(None, exists=False, saw_begin=False)
    assert out["trichotomy_enum"] == "absent_pre_begin"


def test_wrong_hash_absent_initially():
    assert CONSUMER_NAME not in sys.modules
    with pytest.raises(facade.ImportFacadeError, match="module_sha_mismatch"):
        facade.load_terminal_receipt_consumer("0" * 64, repo_root=REPO)
    assert CONSUMER_NAME not in sys.modules


def test_preseeded_wrong_hash_leaves_absent():
    stale = types.ModuleType(CONSUMER_NAME)
    sys.modules[CONSUMER_NAME] = stale
    with pytest.raises(facade.ImportFacadeError, match="module_sha_mismatch"):
        facade.load_terminal_receipt_consumer("0" * 64, repo_root=REPO)
    assert CONSUMER_NAME not in sys.modules


def test_preseeded_missing_path_leaves_absent(tmp_path: Path):
    stale = types.ModuleType(CONSUMER_NAME)
    sys.modules[CONSUMER_NAME] = stale
    with pytest.raises(facade.ImportFacadeError, match="module_path_missing"):
        facade.load_terminal_receipt_consumer("0" * 64, repo_root=tmp_path)
    assert CONSUMER_NAME not in sys.modules


def test_reload_after_file_change(tmp_path: Path):
    dest = tmp_path / CONSUMER_REL
    dest.parent.mkdir(parents=True)
    body1 = (
        "def support_trichotomy_from_bytes(*a, **k):\n    return {'v': 1}\n"
        "def cross_check_pair_receipt(*a, **k):\n    return ['v1']\n"
    )
    dest.write_text(body1)
    m1 = facade.load_terminal_receipt_consumer(_sha(dest), repo_root=tmp_path)
    assert m1.support_trichotomy_from_bytes()["v"] == 1
    body2 = (
        "def support_trichotomy_from_bytes(*a, **k):\n    return {'v': 2}\n"
        "def cross_check_pair_receipt(*a, **k):\n    return ['v2']\n"
    )
    dest.write_text(body2)
    m2 = facade.load_terminal_receipt_consumer(_sha(dest), repo_root=tmp_path)
    assert m2 is sys.modules[CONSUMER_NAME]
    assert m2.support_trichotomy_from_bytes()["v"] == 2
    assert m1 is not m2


def test_missing_public_api_leaves_absent(tmp_path: Path):
    dest = tmp_path / CONSUMER_REL
    dest.parent.mkdir(parents=True)
    dest.write_text("X = 1\n")
    with pytest.raises(facade.ImportFacadeError, match="missing_public_api"):
        facade.load_terminal_receipt_consumer(_sha(dest), repo_root=tmp_path)
    assert CONSUMER_NAME not in sys.modules


def test_preseeded_exec_failure_leaves_absent(tmp_path: Path):
    stale = types.ModuleType(CONSUMER_NAME)
    sys.modules[CONSUMER_NAME] = stale
    dest = tmp_path / CONSUMER_REL
    dest.parent.mkdir(parents=True)
    dest.write_text("raise RuntimeError('exec_boom')\n")
    with pytest.raises(RuntimeError, match="exec_boom"):
        facade.load_terminal_receipt_consumer(_sha(dest), repo_root=tmp_path)
    assert CONSUMER_NAME not in sys.modules
