"""Static TTIR manifest tests for BR-3C-H.1b credit-axis Triton kernels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.credit_axis_kernel_manifest import (
    DEFAULT_MANIFEST_DIR,
    FORBIDDEN_TTIR_PATTERNS,
    STAGE_KERNELS,
    _kernel_source_sha256,
    _scan_ttir_for_forbidden_float_ops,
    default_pipeline_triton_kernel_names,
    list_orphan_manifest_artifacts,
    list_stale_manifest_proof_artifacts,
    manifest_kernel_coverage_gap,
    verify_kernel_manifest_bundle_at_launch,
)

MANIFEST_DIR = DEFAULT_MANIFEST_DIR


@pytest.fixture(scope="module")
def manifest_bundle() -> dict:
  if not (MANIFEST_DIR / "manifest.json").is_file():
    pytest.fail("manifest bundle missing; run credit_axis_kernel_manifest emit first")
  return json.loads((MANIFEST_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_kernel_manifest_all_stages_integer_only(manifest_bundle: dict) -> None:
    assert set(manifest_bundle["stages"]) == set(STAGE_KERNELS)
    for stage, proof in manifest_bundle["stages"].items():
        assert proof["pass"] is True, stage
        ttir_path = MANIFEST_DIR / f"{stage}.ttir"
        assert ttir_path.is_file()
        violations = _scan_ttir_for_forbidden_float_ops(ttir_path.read_text(encoding="utf-8"))
        assert violations == []


def test_kernel_manifest_rejects_injected_float_ttir() -> None:
    injected = "module attributes {gpu.container_modules} {\n  sitofp %0 : i32 to f32\n}\n"
    violations = _scan_ttir_for_forbidden_float_ops(injected)
    assert any("sitofp" in v for v in violations)


def test_launch_rejects_stale_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = MANIFEST_DIR / "manifest.json"
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale = dict(bundle)
    stale["source_sha256"] = "0" * 64
    tmp_manifest = tmp_path / "manifest.json"
    tmp_manifest.write_text(json.dumps(stale, indent=2) + "\n", encoding="utf-8")
    (tmp_path / "manifest_bundle.sha256").write_text(
        __import__("hashlib").sha256(tmp_manifest.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(RuntimeError, match="source_sha256 stale"):
        verify_kernel_manifest_bundle_at_launch(tmp_path)


def test_manifest_source_sha256_matches_kernel_module() -> None:
    manifest = json.loads((MANIFEST_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == _kernel_source_sha256()


def test_forbidden_ttir_patterns_cover_float_ops() -> None:
    assert "fp32" in FORBIDDEN_TTIR_PATTERNS
    assert "sitofp" in FORBIDDEN_TTIR_PATTERNS


def test_manifest_covers_all_default_path_triton_kernels() -> None:
    launched = set(default_pipeline_triton_kernel_names())
    covered = set(STAGE_KERNELS.values())
    assert launched == covered, (
        f"coverage gap={sorted(launched - covered)} "
        f"extra manifest={sorted(covered - launched)}"
    )
    assert manifest_kernel_coverage_gap() == ()


def test_manifest_dir_has_no_orphan_or_stale_artifacts(manifest_bundle: dict) -> None:
    orphans = list_orphan_manifest_artifacts(MANIFEST_DIR, manifest_bundle)
    assert orphans == [], f"orphan manifest artifacts: {orphans}"
    stale = list_stale_manifest_proof_artifacts(MANIFEST_DIR, manifest_bundle)
    assert stale == [], f"stale manifest proof artifacts: {stale}"
    for stage in manifest_bundle["stages"]:
        assert (MANIFEST_DIR / f"{stage}.ttir").is_file(), stage
        assert (MANIFEST_DIR / f"{stage}.integer_only_proof.json").is_file(), stage
