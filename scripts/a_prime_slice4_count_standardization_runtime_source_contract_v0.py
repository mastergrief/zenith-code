"""Pure runtime-source contract for Rung-6 count-standardization (STEP-2).

4-file ORDERED_CONCAT_V0 (RAW digest bytes), single Rung-5 terminal pin.
NO CLI/print/run-root writes. PLAN v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Mapping

MANIFEST_SCHEMA_ID = "a_prime_slice4_rung6_runtime_source_manifest/v0"
ALGORITHM = "ORDERED_CONCAT_V0"
MINTED_BY = "claude_gate1_step2_freeze"
TASK_ID = "1786004998450-f6569bd2"
PLAN_REVISION_BINDING = (
    "PLAN_v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900"
)

ORDERED_RUNTIME_PATHS: tuple[str, ...] = (
    "scripts/a_prime_slice4_count_standardization_schema_v0.py",
    "scripts/a_prime_slice4_count_standardization_reducer_v0.py",
    "scripts/a_prime_slice4_count_standardization_runtime_source_contract_v0.py",
    "scripts/a_prime_slice4_count_standardization_classifier_v0.py",
)

FROZEN_RUNG5_TERMINAL_PIN: dict[str, str] = {
    "path": (
        "/home/gabe/claw-code-creditdir/a_prime_slice4_rung5/"
        "run_component_v1/terminal_receipt.json"
    ),
    "sha256": "9b9939b52c6fa984582c93604d8033385bb6ecd154399d782da49ba013096a6c",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_exact_str(v: Any) -> bool:
    return type(v) is str


def _is_exact_list(v: Any) -> bool:
    return type(v) is list


def _is_exact_dict(v: Any) -> bool:
    return type(v) is dict


def compare_expected_observed_sha(
    expected_hex: str, observed_hex: str
) -> tuple[bool, str]:
    if not _is_exact_str(expected_hex) or not _HEX64.match(expected_hex):
        return False, "expected_sha_malformed"
    if not _is_exact_str(observed_hex) or not _HEX64.match(observed_hex):
        return False, "observed_sha_malformed"
    if expected_hex != observed_hex:
        return False, "runtime_source_manifest_sha_mismatch"
    return True, "ok"


def check_rung5_terminal_pin(
    *,
    path: Any,
    sha256: str,
    pin: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    p = dict(pin) if pin is not None else dict(FROZEN_RUNG5_TERMINAL_PIN)
    exp_path, exp_sha = p.get("path"), p.get("sha256")
    if not _is_exact_str(exp_path) or not _is_exact_str(exp_sha):
        return False, "INSTRUMENT_OR_BIND_FAIL:rung5:pin_shape"
    if str(path) != exp_path:
        return False, "INSTRUMENT_OR_BIND_FAIL:rung5:path_ne_pin"
    if not _is_exact_str(sha256) or not _HEX64.match(sha256):
        return False, "INSTRUMENT_OR_BIND_FAIL:rung5:sha_malformed"
    if sha256 != exp_sha:
        return False, "INSTRUMENT_OR_BIND_FAIL:rung5:sha_ne_pin"
    return True, "ok"


def ordered_concat_v0(
    ordered_paths: list[str], per_file_sha256: Mapping[str, str]
) -> str:
    """RAW digest-byte concat (Rung-5 parity) — not hex-ASCII."""
    parts = [bytes.fromhex(per_file_sha256[p]) for p in ordered_paths]
    return hashlib.sha256(b"".join(parts)).hexdigest()


def admit_runtime_source_manifest(obj: Any) -> tuple[bool, str]:
    if not _is_exact_dict(obj):
        return False, "manifest_not_dict"
    if obj.get("schema_id") != MANIFEST_SCHEMA_ID:
        return False, f"schema_id:{obj.get('schema_id')!r}"
    if obj.get("algorithm") != ALGORITHM:
        return False, f"algorithm:{obj.get('algorithm')!r}"
    if obj.get("minted_by") != MINTED_BY:
        return False, f"minted_by:{obj.get('minted_by')!r}"
    if obj.get("task_id") != TASK_ID:
        return False, f"task_id:{obj.get('task_id')!r}"
    prb = obj.get("plan_revision_binding")
    if not _is_exact_str(prb) or not prb:
        return False, "plan_revision_binding"
    if prb != PLAN_REVISION_BINDING:
        return False, "plan_revision_binding_ne_frozen"
    if not _is_exact_str(obj.get("implementation_content_digest")) or not _HEX64.match(
        obj["implementation_content_digest"]
    ):
        return False, "implementation_content_digest"
    paths = obj.get("ordered_runtime_paths")
    if not _is_exact_list(paths) or not all(_is_exact_str(x) for x in paths):
        return False, "ordered_runtime_paths_type"
    if list(paths) != list(ORDERED_RUNTIME_PATHS):
        return False, "ordered_runtime_paths_ne_frozen"
    pf = obj.get("per_file_sha256")
    if not _is_exact_dict(pf):
        return False, "per_file_sha256_type"
    if set(pf.keys()) != set(ORDERED_RUNTIME_PATHS):
        return False, "per_file_sha256_key_set"
    for p, h in pf.items():
        if not _is_exact_str(p) or not _is_exact_str(h) or not _HEX64.match(h):
            return False, f"per_file_sha256_entry:{p!r}"
    digest = obj.get("runtime_source_digest")
    if not _is_exact_str(digest) or not _HEX64.match(digest):
        return False, "runtime_source_digest_type"
    if digest != ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), pf):
        return False, "runtime_source_digest_mismatch"
    return True, "ok"


def rehash_runtime_files(
    ordered_paths: list[str],
    *,
    read_bytes: Callable[[str], bytes],
) -> dict[str, str]:
    return {p: hashlib.sha256(read_bytes(p)).hexdigest() for p in ordered_paths}


def validate_runtime_source(
    *,
    manifest_obj: Any,
    expected_manifest_sha256: str,
    observed_manifest_sha256: str,
    read_bytes: Callable[[str], bytes],
) -> tuple[bool, str, dict[str, str] | None, str | None]:
    """expected-sha check BEFORE admit; then rehash four files + ORDERED_CONCAT."""
    ok, reason = compare_expected_observed_sha(
        expected_manifest_sha256, observed_manifest_sha256
    )
    if not ok:
        return False, reason, None, None
    ok, reason = admit_runtime_source_manifest(manifest_obj)
    if not ok:
        return False, reason, None, None
    ordered = list(ORDERED_RUNTIME_PATHS)
    observed = rehash_runtime_files(ordered, read_bytes=read_bytes)
    expected_pf = manifest_obj["per_file_sha256"]
    for p in ordered:
        if observed[p] != expected_pf[p]:
            return False, f"runtime_source_file_sha_mismatch:{p}", None, None
    digest = ordered_concat_v0(ordered, observed)
    if digest != manifest_obj["runtime_source_digest"]:
        return False, "runtime_source_digest_recompute_mismatch", None, None
    return True, "ok", observed, digest
