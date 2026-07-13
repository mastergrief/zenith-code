"""CPU-pure §2b assertion reducers for forgotten-accum flip deferral.

Forbidden imports (enforced by characterization): torch.cuda, filesystem,
probe/launch glue, global_rate_cap.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY,
    RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0,
    DuringWTelemetry,
    WPlus1ReleaseRecord,
    backlog_cardinality,
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def backlog_content_sha256(backlog: Mapping[str, Any] | None) -> str:
    payload = backlog or {}
    # Normalize int keys that may appear as str after JSON roundtrip.
    normalized: dict[str, dict[str, Any]] = {}
    for state_key, by_index in payload.items():
        inner: dict[str, Any] = {}
        for flat_index, entry in dict(by_index).items():
            inner[str(int(flat_index))] = dict(entry)
        normalized[str(state_key)] = inner
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def assert_pre_W_seed_invariant(
    *,
    r0_acc_sha: str,
    rw_acc_sha: str,
    r0_backlog_sha: str,
    rw_backlog_sha: str,
    r0_backlog_cardinality: int,
    rw_backlog_cardinality: int,
) -> str:
    if r0_acc_sha != rw_acc_sha:
        raise AssertionError(
            f"{PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY}: acc sha mismatch "
            f"{r0_acc_sha} != {rw_acc_sha}"
        )
    if r0_backlog_sha != rw_backlog_sha:
        raise AssertionError(
            f"{PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY}: backlog sha mismatch"
        )
    if int(r0_backlog_cardinality) != int(rw_backlog_cardinality):
        raise AssertionError(
            f"{PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY}: backlog cardinality mismatch"
        )
    return PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY


def assert_during_W(telemetry: DuringWTelemetry, *, seed_backlog_sha: str) -> None:
    if telemetry.acc_hash_pre == telemetry.acc_hash_post:
        raise AssertionError("during_W: accumulator carry must CHANGE")
    if int(telemetry.flip_applied_count) != 0:
        raise AssertionError("during_W: flip_applied_count must be 0")
    if int(telemetry.threshold_residual_writeback_count) != 0:
        raise AssertionError("during_W: threshold_residual_writeback_count must be 0")
    if telemetry.backlog_hash != seed_backlog_sha:
        raise AssertionError(
            "during_W: authoritative backlog hash must stay FIXED "
            f"(got {telemetry.backlog_hash}, seed {seed_backlog_sha})"
        )
    if not telemetry.flip_application_deferred:
        raise AssertionError("during_W: flip_application_deferred must be True")
    if telemetry.cap_site_branch != DENSE_LEGACY_CAP_SITE_ID:
        raise AssertionError(
            f"during_W: cap site branch mismatch: {telemetry.cap_site_branch}"
        )


def assert_W_plus_1_anti_burst(record: WPlus1ReleaseRecord) -> None:
    if int(record.applied_count) > int(record.ordinary_cap):
        raise AssertionError(
            f"W+1: applied_count {record.applied_count} > ordinary_cap {record.ordinary_cap}"
        )
    if bool(record.special_backlog_flush):
        raise AssertionError("W+1: special_backlog_flush must be False")
    if record.release_path_id != RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0:
        raise AssertionError(
            f"W+1: release_path_id must be {RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0}, "
            f"got {record.release_path_id}"
        )


def assert_cap_site_branch_coverage(cap_site_branch: str) -> None:
    if cap_site_branch != DENSE_LEGACY_CAP_SITE_ID:
        raise AssertionError(
            "BRANCH_COVERAGE failed: expected "
            f"{DENSE_LEGACY_CAP_SITE_ID}, got {cap_site_branch}"
        )


def assert_backlog_unchanged(
    *,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> None:
    if backlog_content_sha256(before) != backlog_content_sha256(after):
        raise AssertionError("authoritative deferred backlog mutated unexpectedly")
    if backlog_cardinality(before) != backlog_cardinality(after):
        raise AssertionError("authoritative deferred backlog cardinality changed")
