"""Pure runtime-source contract for Rung-5 shared-component (STEP-2).

Manifest admission, expected-vs-observed sha, 9-file ORDERED_CONCAT_V0,
horizon+rung3+rung4 frozen pins. NO CLI/print/run-root/filesystem writes.
PLAN v4: a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Mapping

MANIFEST_SCHEMA_ID = "a_prime_slice4_rung5_runtime_source_manifest/v0"
ALGORITHM = "ORDERED_CONCAT_V0"
MINTED_BY = "claude_gate1_step2_freeze"
TASK_ID = "1785961037183-68ac27ec"

ORDERED_RUNTIME_PATHS: tuple[str, ...] = (
    "scripts/a_prime_slice4_shared_component_decomposition_schema_v0.py",
    "scripts/a_prime_slice4_shared_component_decomposition_reducer_v0.py",
    "scripts/a_prime_slice4_shared_component_decomposition_classifier_v0.py",
    "scripts/a_prime_slice4_shared_component_runtime_source_contract_v0.py",
    "scripts/a_prime_slice4_support_split_residual_densify_schema_v0.py",
    "scripts/a_prime_slice4_support_split_residual_densify_reducer_v0.py",
    "scripts/a_prime_slice4_residual_classification_schema_v0.py",
    "scripts/a_prime_slice4_residual_classification_reducer_v0.py",
    "scripts/a_prime_slice4_residual_classification_classifier_v0.py",
)

FROZEN_HORIZON_PINS: dict[str, dict[str, str]] = {
    "package/N10": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N10/c2p1_impl_cpu/receipt.json",
        "sha256": "217cf68650e56578c10c5c5e5a92d91eb81c2e95074d4e5726f76693efadfbe8",
    },
    "package/N20": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N20/c2p1_impl_cpu/receipt.json",
        "sha256": "b12700f39e9979661f3f7b38c51a17627ae279f2ce61eadffeea24f1f902326a",
    },
    "package/N50": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_protect/run_702cc34b/horizon_N50/c2p1_impl_cpu/receipt.json",
        "sha256": "ecffa277c677d509676bce62b93b7fab2225a8c7206d9553c2bbfbb1150de0a8",
    },
    "out/N10": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N10/c2p1_impl_cpu/receipt.json",
        "sha256": "9cd5fe63e509d9d81274b01f08bd9bae861012737282ffcebdb078f403432be3",
    },
    "out/N20": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N20/c2p1_impl_cpu/receipt.json",
        "sha256": "51d1b08df4b561343e8aaf396e84fcd776239b0f902fee203e4bd598ed338263",
    },
    "out/N50": {
        "path": "/home/gabe/claw-code-creditdir/a_prime_slice3_onset/run_fb2fcec5/horizon_N50/c2p1_impl_cpu/receipt.json",
        "sha256": "c5394cda16a1ec42cb938d39d305e1777e397e5a0b19726206bbb5bf54f9206d",
    },
}
FROZEN_RUNG3_TERMINAL_PIN: dict[str, str] = {
    "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_rung3/run_residual_v3/terminal_receipt.json",
    "sha256": "1a8e0fbcd682ad9c3e7c7e9b9dd52a4890b4529d6ac87439887a2ff8b66ab0a3",
}
FROZEN_RUNG4_TERMINAL_PIN: dict[str, str] = {
    "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_rung4/run_densify_v1/terminal_receipt.json",
    "sha256": "1aa7116fd59c591ea5845521e9f76907a4b717bab1a78555e6a82143226351eb",
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


def _pin_path_sha(
    *,
    got_path: Any,
    got_sha: str,
    pin: Mapping[str, str],
    fail_prefix: str,
) -> tuple[bool, str]:
    exp_path, exp_sha = pin.get("path"), pin.get("sha256")
    if not _is_exact_str(exp_path) or not _is_exact_str(exp_sha):
        return False, f"{fail_prefix}:pin_shape"
    if str(got_path) != exp_path:
        return False, f"{fail_prefix}:path_ne_pin"
    if not _is_exact_str(got_sha) or not _HEX64.match(got_sha):
        return False, f"{fail_prefix}:sha_malformed"
    if got_sha != exp_sha:
        return False, f"{fail_prefix}:sha_ne_pin"
    return True, "ok"


def check_frozen_input_pins(
    *,
    package_paths: Mapping[int, Any],
    out_paths: Mapping[int, Any],
    rung3_path: Any,
    rung4_path: Any,
    package_shas: Mapping[int, str],
    out_shas: Mapping[int, str],
    rung3_sha: str,
    rung4_sha: str,
    horizon_pins: Mapping[str, Mapping[str, str]] | None = None,
    rung3_pin: Mapping[str, str] | None = None,
    rung4_pin: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Pure path+sha pin check. No filesystem IO."""
    hp = dict(horizon_pins) if horizon_pins is not None else dict(FROZEN_HORIZON_PINS)
    r3 = dict(rung3_pin) if rung3_pin is not None else dict(FROZEN_RUNG3_TERMINAL_PIN)
    r4 = dict(rung4_pin) if rung4_pin is not None else dict(FROZEN_RUNG4_TERMINAL_PIN)
    for arm, paths, shas in (
        ("package", package_paths, package_shas),
        ("out", out_paths, out_shas),
    ):
        for h in (10, 20, 50):
            key = f"{arm}/N{h}"
            pin = hp.get(key)
            if not _is_exact_dict(pin):
                return False, f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:missing_pin"
            ok, reason = _pin_path_sha(
                got_path=paths[h],
                got_sha=shas[h],
                pin=pin,  # type: ignore[arg-type]
                fail_prefix=f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}",
            )
            if not ok:
                return False, reason
    ok, reason = _pin_path_sha(
        got_path=rung3_path, got_sha=rung3_sha, pin=r3, fail_prefix="AUTHORITY_BIND_FAIL:rung3"
    )
    if not ok:
        return False, reason
    ok, reason = _pin_path_sha(
        got_path=rung4_path, got_sha=rung4_sha, pin=r4, fail_prefix="AUTHORITY_BIND_FAIL:rung4"
    )
    if not ok:
        return False, reason
    return True, "ok"


def ordered_concat_v0(
    ordered_paths: list[str], per_file_sha256: Mapping[str, str]
) -> str:
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
    if not _is_exact_str(obj.get("plan_revision_binding")) or not obj.get(
        "plan_revision_binding"
    ):
        return False, "plan_revision_binding"
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
    worktree_root_or_readers: Any,
) -> tuple[bool, str, dict[str, str] | None, str | None]:
    ok, reason = compare_expected_observed_sha(
        expected_manifest_sha256, observed_manifest_sha256
    )
    if not ok:
        return False, reason, None, None
    ok, reason = admit_runtime_source_manifest(manifest_obj)
    if not ok:
        return False, reason, None, None
    if callable(worktree_root_or_readers):
        reader = worktree_root_or_readers
    else:
        root = worktree_root_or_readers

        def reader(rel: str) -> bytes:
            return (root / rel).read_bytes()  # type: ignore[operator]

    try:
        observed_map = rehash_runtime_files(list(ORDERED_RUNTIME_PATHS), read_bytes=reader)
    except Exception as e:
        return False, f"rehash_failed:{e}", None, None
    expected_map = dict(manifest_obj["per_file_sha256"])
    if observed_map != expected_map:  # whole-map exact-dict
        return False, "runtime_source_file_sha_mismatch", observed_map, None
    digest = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), observed_map)
    if digest != manifest_obj["runtime_source_digest"]:
        return False, "runtime_source_digest_recompute_ne", observed_map, digest
    return True, "ok", observed_map, digest
