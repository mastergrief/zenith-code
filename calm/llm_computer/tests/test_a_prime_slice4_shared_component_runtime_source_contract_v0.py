"""CPU-static unit battery for Rung-5 runtime-source contract (STEP-2)."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from scripts.a_prime_slice4_shared_component_runtime_source_contract_v0 import (
    ALGORITHM,
    FROZEN_HORIZON_PINS,
    FROZEN_RUNG3_TERMINAL_PIN,
    FROZEN_RUNG4_TERMINAL_PIN,
    MANIFEST_SCHEMA_ID,
    MINTED_BY,
    ORDERED_RUNTIME_PATHS,
    TASK_ID,
    admit_runtime_source_manifest,
    check_frozen_input_pins,
    compare_expected_observed_sha,
    ordered_concat_v0,
    rehash_runtime_files,
    validate_runtime_source,
)

ROOT = Path(__file__).resolve().parents[3]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _bytes_map() -> dict[str, bytes]:
    return {p: (ROOT / p).read_bytes() for p in ORDERED_RUNTIME_PATHS}


def _good_manifest(per_file: dict[str, str] | None = None) -> dict:
    if per_file is None:
        per_file = {p: _sha(b) for p, b in _bytes_map().items()}
    digest = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per_file)
    return {
        "schema_id": MANIFEST_SCHEMA_ID,
        "ordered_runtime_paths": list(ORDERED_RUNTIME_PATHS),
        "per_file_sha256": dict(per_file),
        "runtime_source_digest": digest,
        "algorithm": ALGORITHM,
        "implementation_content_digest": "a" * 64,
        "minted_by": MINTED_BY,
        "task_id": TASK_ID,
        "plan_revision_binding": "PLAN_v4 a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421",
    }


def test_nine_path_set():
    assert len(ORDERED_RUNTIME_PATHS) == 9
    assert ORDERED_RUNTIME_PATHS[0].endswith("shared_component_decomposition_schema_v0.py")
    assert ORDERED_RUNTIME_PATHS[7].endswith("residual_classification_reducer_v0.py")
    assert ORDERED_RUNTIME_PATHS[8].endswith("residual_classification_classifier_v0.py")
    assert ORDERED_RUNTIME_PATHS[-1].endswith("residual_classification_classifier_v0.py")


def test_compare_expected_observed_sha():
    h = "ab" * 32
    assert compare_expected_observed_sha(h, h) == (True, "ok")
    assert compare_expected_observed_sha(h, "cd" * 32)[0] is False
    assert compare_expected_observed_sha("nope", h)[0] is False


def test_ordered_concat_deterministic():
    per = {p: _sha(b"x" + p.encode()) for p in ORDERED_RUNTIME_PATHS}
    a = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per)
    assert a == ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per) and len(a) == 64
    assert ordered_concat_v0(list(ORDERED_RUNTIME_PATHS)[::-1], per) != a


def test_admit_known_good():
    m = _good_manifest()
    assert m["plan_revision_binding"] == "PLAN_v4 a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421"
    assert admit_runtime_source_manifest(m) == (True, "ok")


def test_admit_wrong_schema_and_path_set():
    m = _good_manifest(); m["schema_id"] = "wrong"
    assert admit_runtime_source_manifest(m)[0] is False
    m = _good_manifest(); m["ordered_runtime_paths"] = list(ORDERED_RUNTIME_PATHS)[:-1]
    assert admit_runtime_source_manifest(m)[0] is False
    m = _good_manifest(); m["per_file_sha256"] = dict(m["per_file_sha256"]); m["per_file_sha256"]["extra.py"] = "ab" * 32
    assert admit_runtime_source_manifest(m)[0] is False


def test_admit_digest_mismatch():
    m = _good_manifest(); m["runtime_source_digest"] = "ff" * 32
    assert admit_runtime_source_manifest(m)[0] is False


def test_validate_known_good_and_sha_mismatch():
    bmap = _bytes_map()
    per = {p: _sha(b) for p, b in bmap.items()}
    m = _good_manifest(per)
    exp = _sha(b"x")  # not used for map match
    # use actual expected equal
    import json
    raw = json.dumps(m, sort_keys=True).encode()
    exp = _sha(raw)
    ok, reason, omap, dig = validate_runtime_source(
        manifest_obj=m, expected_manifest_sha256=exp, observed_manifest_sha256=exp,
        worktree_root_or_readers=lambda p: bmap[p],
    )
    assert ok and omap == per and dig == m["runtime_source_digest"]
    ok2, reason2, _, _ = validate_runtime_source(
        manifest_obj=m, expected_manifest_sha256="ab" * 32, observed_manifest_sha256=exp,
        worktree_root_or_readers=lambda p: bmap[p],
    )
    assert ok2 is False and "mismatch" in reason2


@pytest.mark.parametrize("path_idx", [
    0,  # component schema
    4,  # densify schema
    6,  # residual schema
    8,  # ninth: residual_classification_classifier_v0
])
def test_validate_file_map_whole_equality(path_idx):
    """Whole-map per_file equality: any ordered-path byte flip → mismatch."""
    bmap = _bytes_map()
    per = {p: _sha(b) for p, b in bmap.items()}
    m = _good_manifest(per)
    flipped = dict(bmap)
    target = ORDERED_RUNTIME_PATHS[path_idx]
    flipped[target] = bmap[target] + b"\x00"
    ok, reason, _, _ = validate_runtime_source(
        manifest_obj=m, expected_manifest_sha256="cd" * 32, observed_manifest_sha256="cd" * 32,
        worktree_root_or_readers=lambda p: flipped[p],
    )
    assert ok is False and reason == "runtime_source_file_sha_mismatch"


def _pin_args():
    pkg_paths = {10: FROZEN_HORIZON_PINS["package/N10"]["path"], 20: FROZEN_HORIZON_PINS["package/N20"]["path"], 50: FROZEN_HORIZON_PINS["package/N50"]["path"]}
    out_paths = {10: FROZEN_HORIZON_PINS["out/N10"]["path"], 20: FROZEN_HORIZON_PINS["out/N20"]["path"], 50: FROZEN_HORIZON_PINS["out/N50"]["path"]}
    pkg_shas = {10: FROZEN_HORIZON_PINS["package/N10"]["sha256"], 20: FROZEN_HORIZON_PINS["package/N20"]["sha256"], 50: FROZEN_HORIZON_PINS["package/N50"]["sha256"]}
    out_shas = {10: FROZEN_HORIZON_PINS["out/N10"]["sha256"], 20: FROZEN_HORIZON_PINS["out/N20"]["sha256"], 50: FROZEN_HORIZON_PINS["out/N50"]["sha256"]}
    return dict(
        package_paths=pkg_paths, out_paths=out_paths, package_shas=pkg_shas, out_shas=out_shas,
        rung3_path=FROZEN_RUNG3_TERMINAL_PIN["path"], rung3_sha=FROZEN_RUNG3_TERMINAL_PIN["sha256"],
        rung4_path=FROZEN_RUNG4_TERMINAL_PIN["path"], rung4_sha=FROZEN_RUNG4_TERMINAL_PIN["sha256"],
    )


def test_frozen_pins_ok():
    assert check_frozen_input_pins(**_pin_args()) == (True, "ok")


def test_frozen_pins_horizon_path_and_sha():
    a = _pin_args(); a["package_paths"] = dict(a["package_paths"]); a["package_paths"][10] = "/wrong"
    assert check_frozen_input_pins(**a)[0] is False
    a = _pin_args(); a["package_shas"] = dict(a["package_shas"]); a["package_shas"][20] = "00" * 32
    assert check_frozen_input_pins(**a)[0] is False


def test_frozen_pins_rung3_rung4():
    a = _pin_args(); a["rung3_sha"] = "11" * 32
    ok, reason = check_frozen_input_pins(**a)
    assert ok is False and "rung3" in reason
    a = _pin_args(); a["rung4_path"] = "/nope"
    ok, reason = check_frozen_input_pins(**a)
    assert ok is False and "rung4" in reason


def test_rehash_runtime_files():
    bmap = _bytes_map()
    m = rehash_runtime_files(list(ORDERED_RUNTIME_PATHS), read_bytes=lambda p: bmap[p])
    assert set(m) == set(ORDERED_RUNTIME_PATHS)
    assert all(len(v) == 64 for v in m.values())


def _scripts_import_edges_from_paths(paths: tuple[str, ...] | list[str]) -> set[tuple[str, str]]:
    """AST: (src_rel, dst_module_relpath) for every repo-local scripts.* import edge."""
    import ast

    edges: set[tuple[str, str]] = set()
    for rel in paths:
        src = ROOT / rel
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts."):
                mod = node.module[len("scripts.") :]
                dst = f"scripts/{mod.replace('.', '/')}.py"
                edges.add((rel, dst))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("scripts."):
                        mod = alias.name[len("scripts.") :]
                        dst = f"scripts/{mod.replace('.', '/')}.py"
                        edges.add((rel, dst))
    return edges


def test_import_closure_ordered_runtime_set():
    """Every scripts.* import edge from ORDERED_RUNTIME_PATHS resolves INSIDE the set."""
    frozen = set(ORDERED_RUNTIME_PATHS)
    edges = _scripts_import_edges_from_paths(ORDERED_RUNTIME_PATHS)
    assert edges, "expected at least one scripts.* import edge"
    outside = sorted({dst for _, dst in edges if dst not in frozen})
    assert outside == [], f"import edges escape frozen set: {outside} edges={sorted(edges)}"


def test_import_closure_fails_when_ninth_missing():
    """Known-bad: synthetic 8-path set missing residual classifier → outside edge observed."""
    truncated = tuple(ORDERED_RUNTIME_PATHS[:-1])
    assert len(truncated) == 8
    assert not any(p.endswith("residual_classification_classifier_v0.py") for p in truncated)
    frozen = set(truncated)
    # edges from the *full* classifier still reference the ninth; evaluate edges among truncated only
    # Use component classifier (in truncated) which imports residual_classification_classifier_v0
    edges = _scripts_import_edges_from_paths(truncated)
    outside = sorted({dst for _, dst in edges if dst not in frozen})
    assert any(p.endswith("residual_classification_classifier_v0.py") for p in outside), (
        f"expected ninth path outside truncated set; outside={outside}"
    )
