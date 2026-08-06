"""STEP-2 tests: Rung-6 runtime-source contract (PLAN v6)."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import scripts.a_prime_slice4_count_standardization_runtime_source_contract_v0 as c

PLAN_BINDING = (
    "PLAN_v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900"
)
REPO = Path(__file__).resolve().parents[3]


def _sha_file(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def _good_manifest() -> dict:
    ordered = list(c.ORDERED_RUNTIME_PATHS)
    pf = {p: _sha_file(p) for p in ordered}
    digest = c.ordered_concat_v0(ordered, pf)
    return {
        "schema_id": c.MANIFEST_SCHEMA_ID,
        "algorithm": c.ALGORITHM,
        "minted_by": c.MINTED_BY,
        "task_id": c.TASK_ID,
        "plan_revision_binding": PLAN_BINDING,
        "implementation_content_digest": "a" * 64,
        "ordered_runtime_paths": ordered,
        "per_file_sha256": pf,
        "runtime_source_digest": digest,
    }


def test_admit_known_good():
    m = _good_manifest()
    ok, reason = c.admit_runtime_source_manifest(m)
    assert ok and reason == "ok"
    assert m["plan_revision_binding"] == PLAN_BINDING
    assert m["plan_revision_binding"] == c.PLAN_REVISION_BINDING


def test_ordered_concat_raw_bytes_not_hex_ascii():
    ordered = list(c.ORDERED_RUNTIME_PATHS)
    pf = {p: _sha_file(p) for p in ordered}
    raw = c.ordered_concat_v0(ordered, pf)
    # hex-ASCII concat would differ for multi-file sets
    ascii_join = hashlib.sha256("".join(pf[p] for p in ordered).encode()).hexdigest()
    assert raw != ascii_join
    assert len(raw) == 64


def test_expected_sha_mismatch_rejects():
    ok, reason = c.compare_expected_observed_sha("a" * 64, "b" * 64)
    assert not ok and reason == "runtime_source_manifest_sha_mismatch"
    ok, reason = c.compare_expected_observed_sha("nothex", "a" * 64)
    assert not ok and reason == "expected_sha_malformed"


def test_per_file_flip_fails_validate():
    m = _good_manifest()
    m_bytes = __import__("json").dumps(m, sort_keys=True).encode()
    observed = hashlib.sha256(m_bytes).hexdigest()
    # flip one file sha in manifest map vs worktree
    bad = copy.deepcopy(m)
    victim = c.ORDERED_RUNTIME_PATHS[0]
    bad["per_file_sha256"][victim] = "0" * 64
    bad["runtime_source_digest"] = c.ordered_concat_v0(
        list(c.ORDERED_RUNTIME_PATHS), bad["per_file_sha256"]
    )

    def reader(rel: str) -> bytes:
        return (REPO / rel).read_bytes()

    # admit would pass if digest matches flipped map, but rehash vs worktree fails
    ok, reason, _, _ = c.validate_runtime_source(
        manifest_obj=bad,
        expected_manifest_sha256=hashlib.sha256(
            __import__("json").dumps(bad, sort_keys=True).encode()
        ).hexdigest(),
        observed_manifest_sha256=hashlib.sha256(
            __import__("json").dumps(bad, sort_keys=True).encode()
        ).hexdigest(),
        read_bytes=reader,
    )
    assert not ok
    assert "runtime_source_file_sha_mismatch" in reason


def test_validate_known_good():
    m = _good_manifest()
    raw = __import__("json").dumps(m, sort_keys=True).encode()
    h = hashlib.sha256(raw).hexdigest()

    def reader(rel: str) -> bytes:
        return (REPO / rel).read_bytes()

    ok, reason, obs, dig = c.validate_runtime_source(
        manifest_obj=m,
        expected_manifest_sha256=h,
        observed_manifest_sha256=h,
        read_bytes=reader,
    )
    assert ok and reason == "ok"
    assert dig == m["runtime_source_digest"]
    assert obs is not None and len(obs) == 4


def test_rung5_pin():
    pin = c.FROZEN_RUNG5_TERMINAL_PIN
    ok, reason = c.check_rung5_terminal_pin(
        path=pin["path"], sha256=pin["sha256"]
    )
    assert ok and reason == "ok"
    ok, reason = c.check_rung5_terminal_pin(path="/wrong", sha256=pin["sha256"])
    assert not ok and "path_ne_pin" in reason
    ok, reason = c.check_rung5_terminal_pin(path=pin["path"], sha256="0" * 64)
    assert not ok and "sha_ne_pin" in reason


def test_four_path_set_exact():
    assert len(c.ORDERED_RUNTIME_PATHS) == 4
    assert c.ORDERED_RUNTIME_PATHS[0].endswith("schema_v0.py")
    assert c.ORDERED_RUNTIME_PATHS[1].endswith("reducer_v0.py")
    assert c.ORDERED_RUNTIME_PATHS[2].endswith("runtime_source_contract_v0.py")
    assert c.ORDERED_RUNTIME_PATHS[3].endswith("classifier_v0.py")


def test_line_cap():
    p = REPO / "scripts/a_prime_slice4_count_standardization_runtime_source_contract_v0.py"
    assert len(p.read_text().splitlines()) < 500


def test_no_shared_component_import():
    src = (
        REPO
        / "scripts/a_prime_slice4_count_standardization_runtime_source_contract_v0.py"
    ).read_text()
    assert "shared_component" not in src
    assert "residual_classification" not in src
    assert "support_split_residual" not in src
