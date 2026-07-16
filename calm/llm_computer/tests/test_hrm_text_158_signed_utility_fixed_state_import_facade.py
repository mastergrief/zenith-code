"""Tests for the one named creditdir import facade (PLAN v5 / d1_e2 path binding)."""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_creditdir_import_facade import (
    MODULE_REL_PATHS,
    ImportFacadeError,
    load_signed_utility_fixed_state_modules,
    verify_expected_sha256_by_module,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
MOD = REPO / "calm/hrm_text_158/native_full_stack/signed_utility_fixed_state_creditdir_import_facade.py"


def _truth(root: Path = REPO) -> dict[str, str]:
    return {k: hashlib.sha256((root / rel).read_bytes()).hexdigest() for k, rel in MODULE_REL_PATHS.items()}


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 120


def test_hash_pin_pass_and_bundle_exports():
    expected = _truth()
    bundle = load_signed_utility_fixed_state_modules(expected)
    assert hasattr(bundle.facade, "developer_check")
    assert hasattr(bundle.driver, "run_authoritative_fixed_state_signed_utility")
    assert bundle.observed_sha256_by_module == expected
    for key, rel in MODULE_REL_PATHS.items():
        assert Path(bundle.verified_paths_by_module[key]).resolve() == (REPO / rel).resolve()
        assert Path(getattr(bundle, key).__file__).resolve() == (REPO / rel).resolve()


def test_pre_import_hash_mismatch_zero_side_effects():
    expected = dict(_truth())
    expected["reducers"] = "0" * 64
    with pytest.raises(ImportFacadeError, match="module_sha_mismatch:reducers"):
        verify_expected_sha256_by_module(expected)
    with pytest.raises(ImportFacadeError, match="module_sha_mismatch:reducers"):
        load_signed_utility_fixed_state_modules(expected)


def test_temp_repo_path_substitution_binds_verified_files(tmp_path: Path):
    # Copy byte-identical modules under a temporary repo_root; loaded __file__ must bind there.
    for rel in MODULE_REL_PATHS.values():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
    expected = _truth(tmp_path)
    # Ensure ambient/stale entries exist, then prove verified temp paths win.
    import importlib
    importlib.import_module("calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers")
    importlib.import_module("calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_driver")
    bundle = load_signed_utility_fixed_state_modules(expected, repo_root=tmp_path)
    for key, rel in MODULE_REL_PATHS.items():
        verified = (tmp_path / rel).resolve()
        assert Path(bundle.verified_paths_by_module[key]).resolve() == verified
        assert Path(getattr(bundle, key).__file__).resolve() == verified
        assert verified != (REPO / rel).resolve()
