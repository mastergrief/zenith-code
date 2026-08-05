"""CPU-static unit battery for runtime-source contract module (STEP-2)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.a_prime_slice4_support_split_runtime_source_contract_v0 import (
    ALGORITHM,
    FROZEN_HORIZON_PINS,
    FROZEN_TERMINAL_PIN,
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
        "plan_revision_binding": "v6_rung4_20260805",
    }


def test_compare_expected_observed_sha():
    h = "ab" * 32
    assert compare_expected_observed_sha(h, h) == (True, "ok")
    assert compare_expected_observed_sha(h, "cd" * 32)[0] is False
    assert compare_expected_observed_sha("nope", h)[0] is False
    assert compare_expected_observed_sha(h, "ZZ")[0] is False


def test_ordered_concat_deterministic():
    per = {p: _sha(b"x" + p.encode()) for p in ORDERED_RUNTIME_PATHS}
    a = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per)
    b = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), per)
    assert a == b and len(a) == 64
    # order matters
    rev = list(ORDERED_RUNTIME_PATHS)[::-1]
    c = ordered_concat_v0(rev, per)
    assert c != a


def test_admit_known_good():
    m = _good_manifest()
    assert admit_runtime_source_manifest(m) == (True, "ok")


def test_admit_wrong_schema_id():
    m = _good_manifest(); m["schema_id"] = "wrong"
    assert admit_runtime_source_manifest(m)[0] is False


def test_admit_path_set_missing_extra():
    m = _good_manifest()
    m["ordered_runtime_paths"] = list(ORDERED_RUNTIME_PATHS)[:-1]
    assert admit_runtime_source_manifest(m)[0] is False
    m = _good_manifest()
    m["per_file_sha256"] = dict(m["per_file_sha256"])
    m["per_file_sha256"]["extra.py"] = "ab" * 32
    # also fix ordered? key set mismatch
    assert admit_runtime_source_manifest(m)[0] is False


def test_admit_digest_mismatch():
    m = _good_manifest()
    m["runtime_source_digest"] = "ff" * 32
    assert admit_runtime_source_manifest(m)[0] is False


def test_rehash_and_validate_known_good():
    bmap = _bytes_map()
    per = {p: _sha(b) for p, b in bmap.items()}
    m = _good_manifest(per)
    raw = json.dumps(m, sort_keys=True).encode()
    exp = _sha(raw)
    ok, reason, omap, dig = validate_runtime_source(
        manifest_obj=m,
        expected_manifest_sha256=exp,
        observed_manifest_sha256=exp,
        worktree_root_or_readers=lambda p: bmap[p],
    )
    assert ok and reason == "ok" and omap == per and dig == m["runtime_source_digest"]


def test_validate_expected_sha_mismatch():
    m = _good_manifest()
    ok, reason, _, _ = validate_runtime_source(
        manifest_obj=m,
        expected_manifest_sha256="ab" * 32,
        observed_manifest_sha256="cd" * 32,
        worktree_root_or_readers=lambda p: b"",
    )
    assert ok is False and "mismatch" in reason


def test_validate_byte_flip_file_map():
    bmap = _bytes_map()
    per = {p: _sha(b) for p, b in bmap.items()}
    m = _good_manifest(per)
    # flip residual schema (index 4) while manifest lists old sha
    key = ORDERED_RUNTIME_PATHS[4]
    flipped = dict(bmap)
    flipped[key] = bmap[key] + b"\x00"
    ok, reason, omap, _ = validate_runtime_source(
        manifest_obj=m,
        expected_manifest_sha256="ab" * 32,
        observed_manifest_sha256="ab" * 32,
        worktree_root_or_readers=lambda p: flipped[p],
    )
    assert ok is False and reason == "runtime_source_file_sha_mismatch"
    assert omap is not None and omap[key] != per[key]


def test_validate_contract_module_byte_flip_index3():
    """Battery item 77: flip contract module (index 3); classifier sha unchanged."""
    bmap = _bytes_map()
    per = {p: _sha(b) for p, b in bmap.items()}
    m = _good_manifest(per)
    key = ORDERED_RUNTIME_PATHS[3]
    assert key.endswith("runtime_source_contract_v0.py")
    flipped = dict(bmap)
    flipped[key] = bmap[key] + b"\x01"
    ok, reason, omap, _ = validate_runtime_source(
        manifest_obj=m,
        expected_manifest_sha256="ab" * 32,
        observed_manifest_sha256="ab" * 32,
        worktree_root_or_readers=lambda p: flipped[p],
    )
    assert ok is False and reason == "runtime_source_file_sha_mismatch"
    assert omap is not None and omap[key] != per[key]
    # classifier path bytes unchanged in map
    clf = ORDERED_RUNTIME_PATHS[2]
    assert omap[clf] == per[clf]


def _good_pin_args():
    pkg_paths = {h: FROZEN_HORIZON_PINS[f"package/N{h}"]["path"] for h in (10, 20, 50)}
    out_paths = {h: FROZEN_HORIZON_PINS[f"out/N{h}"]["path"] for h in (10, 20, 50)}
    pkg_shas = {h: FROZEN_HORIZON_PINS[f"package/N{h}"]["sha256"] for h in (10, 20, 50)}
    out_shas = {h: FROZEN_HORIZON_PINS[f"out/N{h}"]["sha256"] for h in (10, 20, 50)}
    return dict(
        package_paths=pkg_paths,
        out_paths=out_paths,
        terminal_path=FROZEN_TERMINAL_PIN["path"],
        package_shas=pkg_shas,
        out_shas=out_shas,
        terminal_sha=FROZEN_TERMINAL_PIN["sha256"],
    )


def test_check_frozen_input_pins_known_good_silent():
    ok, reason = check_frozen_input_pins(**_good_pin_args())
    assert ok is True and reason == "ok"


@pytest.mark.parametrize("arm,h", [
    ("package", 10), ("package", 20), ("package", 50),
    ("out", 10), ("out", 20), ("out", 50),
])
def test_check_frozen_input_pins_sha_substitution(arm, h):
    kwargs = _good_pin_args()
    # correct path, wrong observed sha (tmp-fixture calibration; no creditdir mutate)
    if arm == "package":
        kwargs["package_shas"] = dict(kwargs["package_shas"])
        kwargs["package_shas"][h] = "ff" * 32
    else:
        kwargs["out_shas"] = dict(kwargs["out_shas"])
        kwargs["out_shas"][h] = "ff" * 32
    ok, reason = check_frozen_input_pins(**kwargs)
    assert ok is False
    assert reason == f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:sha_ne_pin"


def test_check_frozen_input_pins_path_mismatch():
    kwargs = _good_pin_args()
    kwargs["package_paths"] = dict(kwargs["package_paths"])
    kwargs["package_paths"][20] = "/tmp/not_the_pinned_package_N20.json"
    ok, reason = check_frozen_input_pins(**kwargs)
    assert ok is False and reason == "INSTRUMENT_OR_BIND_FAIL:package/N20:path_ne_pin"


def test_check_frozen_input_pins_terminal_sha_mismatch():
    kwargs = _good_pin_args()
    kwargs["terminal_sha"] = "aa" * 32
    ok, reason = check_frozen_input_pins(**kwargs)
    assert ok is False and reason == "AUTHORITY_BIND_FAIL:terminal_sha_ne_pin"


def test_check_frozen_input_pins_terminal_path_mismatch():
    kwargs = _good_pin_args()
    kwargs["terminal_path"] = "/tmp/fake_terminal.json"
    ok, reason = check_frozen_input_pins(**kwargs)
    assert ok is False and reason == "AUTHORITY_BIND_FAIL:terminal_path_ne_pin"


def test_valid_shaped_substitute_vs_activation_sha():
    """Internally consistent drifted manifest whose file sha != activation pin."""
    bmap = _bytes_map()
    # drift residual schema content in map
    key = ORDERED_RUNTIME_PATHS[4]
    drifted = dict(bmap)
    drifted[key] = bmap[key] + b"X"
    per = {p: _sha(b) for p, b in drifted.items()}
    m = _good_manifest(per)
    m_bytes = json.dumps(m, sort_keys=True).encode()
    drifted_sha = _sha(m_bytes)
    activation_sha = "11" * 32  # independent pin
    assert drifted_sha != activation_sha
    # compare fails before admit when expected != observed of loaded bytes
    ok, reason = compare_expected_observed_sha(activation_sha, drifted_sha)
    assert ok is False and reason == "runtime_source_manifest_sha_mismatch"


def test_line_cap_contract():
    p = ROOT / "scripts/a_prime_slice4_support_split_runtime_source_contract_v0.py"
    n = p.read_text().count("\n") + 1
    assert n < 500
