"""Pure runtime-source contract for Rung-4 densify (STEP-2).

Manifest schema admission, expected-vs-observed sha, ORDERED_CONCAT_V0,
six-file rehash. NO CLI/print/run-root/terminal IO/filesystem writes.
PLAN v6: feea775c3b3bb1bee6f0d5775d4da783b09560b72b4a1b6cd8500af5f56329a9
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Mapping

MANIFEST_SCHEMA_ID = "a_prime_slice4_rung4_runtime_source_manifest/v0"
ALGORITHM = "ORDERED_CONCAT_V0"
MINTED_BY = "claude_gate1_step2_freeze"
TASK_ID = "1785954214314-1d129d5d"

ORDERED_RUNTIME_PATHS: tuple[str, ...] = (
    "scripts/a_prime_slice4_support_split_residual_densify_schema_v0.py",
    "scripts/a_prime_slice4_support_split_residual_densify_reducer_v0.py",
    "scripts/a_prime_slice4_support_split_residual_densify_classifier_v0.py",
    "scripts/a_prime_slice4_support_split_runtime_source_contract_v0.py",
    "scripts/a_prime_slice4_residual_classification_schema_v0.py",
    "scripts/a_prime_slice4_residual_classification_reducer_v0.py",
)

# PLAN v6 frozen_inputs.receipt_paths_and_sha256 + rung3_terminal_authority.
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
FROZEN_TERMINAL_PIN: dict[str, str] = {
    "path": "/home/gabe/claw-code-creditdir/a_prime_slice4_rung3/run_residual_v3/terminal_receipt.json",
    "sha256": "1a8e0fbcd682ad9c3e7c7e9b9dd52a4890b4529d6ac87439887a2ff8b66ab0a3",
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


def check_frozen_input_pins(
    *,
    package_paths: Mapping[int, Any],
    out_paths: Mapping[int, Any],
    terminal_path: Any,
    package_shas: Mapping[int, str],
    out_shas: Mapping[int, str],
    terminal_sha: str,
    horizon_pins: Mapping[str, Mapping[str, str]] | None = None,
    terminal_pin: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Pure path+sha pin check vs PLAN frozen inputs. No filesystem IO.

    Returns (ok, reason). Horizon mismatch → INSTRUMENT_OR_BIND_FAIL:...
    Terminal mismatch → AUTHORITY_BIND_FAIL:...
    """
    hp = dict(horizon_pins) if horizon_pins is not None else dict(FROZEN_HORIZON_PINS)
    tp = dict(terminal_pin) if terminal_pin is not None else dict(FROZEN_TERMINAL_PIN)
    for arm, paths, shas in (
        ("package", package_paths, package_shas),
        ("out", out_paths, out_shas),
    ):
        for h in (10, 20, 50):
            key = f"{arm}/N{h}"
            pin = hp.get(key)
            if not _is_exact_dict(pin):
                return False, f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:missing_pin"
            exp_path = pin.get("path")
            exp_sha = pin.get("sha256")
            if not _is_exact_str(exp_path) or not _is_exact_str(exp_sha):
                return False, f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:pin_shape"
            got_path = str(paths[h])
            got_sha = shas[h]
            if got_path != exp_path:
                return (
                    False,
                    f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:path_ne_pin",
                )
            if not _is_exact_str(got_sha) or not _HEX64.match(got_sha):
                return (
                    False,
                    f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:sha_malformed",
                )
            if got_sha != exp_sha:
                return (
                    False,
                    f"INSTRUMENT_OR_BIND_FAIL:{arm}/N{h}:sha_ne_pin",
                )
    exp_t_path = tp.get("path")
    exp_t_sha = tp.get("sha256")
    if not _is_exact_str(exp_t_path) or not _is_exact_str(exp_t_sha):
        return False, "AUTHORITY_BIND_FAIL:terminal_pin_shape"
    if str(terminal_path) != exp_t_path:
        return False, "AUTHORITY_BIND_FAIL:terminal_path_ne_pin"
    if not _is_exact_str(terminal_sha) or not _HEX64.match(terminal_sha):
        return False, "AUTHORITY_BIND_FAIL:terminal_sha_malformed"
    if terminal_sha != exp_t_sha:
        return False, "AUTHORITY_BIND_FAIL:terminal_sha_ne_pin"
    return True, "ok"


def ordered_concat_v0(
    ordered_paths: list[str], per_file_sha256: Mapping[str, str]
) -> str:
    parts: list[bytes] = []
    for p in ordered_paths:
        h = per_file_sha256[p]
        parts.append(bytes.fromhex(h))
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
    recomputed = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), pf)
    if digest != recomputed:
        return False, "runtime_source_digest_mismatch"
    return True, "ok"


def rehash_runtime_files(
    ordered_paths: list[str],
    *,
    read_bytes: Callable[[str], bytes],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in ordered_paths:
        out[p] = hashlib.sha256(read_bytes(p)).hexdigest()
    return out


def validate_runtime_source(
    *,
    manifest_obj: Any,
    expected_manifest_sha256: str,
    observed_manifest_sha256: str,
    worktree_root_or_readers: Any,
) -> tuple[bool, str, dict[str, str] | None, str | None]:
    """Full pure validation after expected-sha check on raw bytes.

    worktree_root_or_readers: Path-like root OR callable path->bytes.
    """
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
    if observed_map != expected_map:
        return False, "runtime_source_file_sha_mismatch", observed_map, None
    digest = ordered_concat_v0(list(ORDERED_RUNTIME_PATHS), observed_map)
    if digest != manifest_obj["runtime_source_digest"]:
        return False, "runtime_source_digest_recompute_ne", observed_map, digest
    return True, "ok", observed_map, digest
