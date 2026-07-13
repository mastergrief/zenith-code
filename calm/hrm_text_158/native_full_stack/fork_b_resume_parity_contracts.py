"""Fork B resume-parity pure contracts (schema / hashing / non-target snapshot).

No torch, no filesystem IO, no trainer_sub2, no GPU/launch imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

SCHEMA_ID = "fork_b_non_target_snapshot/v1"
CERTIFICATE_SCHEMA = "fork_b_resume_parity_certificate/v1"
CUTS_DEFAULT: tuple[int, ...] = (4, 16, 28)
K_DEFAULT = 4
Z_BINDING_CUT_T = 16
DENSE_SHADOW_FIELD_PERSISTENT_BPW = 0

GATE_BEARING_FIELDS: tuple[str, ...] = (
    "q_sha256_after",
    "applied_flat_indices_hash16",
    "votes_sha256",
    "global_rate_cap_accepted_indices_sha256",
    "global_rate_cap_deferred_indices_sha256",
    "global_rate_cap_applied_count",
    "flip_count",
    "q_changed_count",
    "applied_selection_score_p50",
    "applied_selection_score_p95",
)

# Frozen now (plan v2) — NOT deferrable to implement-time.
NON_TARGET_SNAPSHOT_SCHEMA_FIELDS: tuple[str, ...] = (
    "rng_states",
    "exact_future_batch_sample_ids",
    "loader_cursor",
    "rate_cap_backlog_schedule",
    "q_scales_weights_code_hash",
    "optimizer_empty_proof",
    "non_manipulated_manifest_fields",
)

# C vs S may differ ONLY in these manifest keys (plus declared accounting).
CS_MANIFEST_DIFF_ALLOWLIST: frozenset[str] = frozenset(
    {
        "bounded_accumulator",
        "hot_exact_indices",
        "hot_exact_values",
        "hot_exact_indices_sha256",
        "hot_exact_values_sha256",
        "cold_exception_indices",
        "cold_exception_values",
        "cold_exception_indices_sha256",
        "cold_exception_values_sha256",
        "cold_default_value",
        "s_accounting_metadata",
        "bounded_refresh_applied",
        "bounded_fresh_for_exact_shadow",
    }
)

BARRED_PROXIES: frozenset[str] = frozenset(
    {
        "vote_nonzero_count",
        "acc_abs_max_after_decay_vote",
    }
)

# Alias used by CLI probe scaffolding (same frozen schema id).
FORK_B_NON_TARGET_SNAPSHOT_SCHEMA = SCHEMA_ID


class ArmId(str, Enum):
    U = "U"
    F = "F"
    C = "C"
    S = "S"
    Z = "Z"


class PreScienceClass(str, Enum):
    INFRA_FAILURE = "INFRA_FAILURE"
    MISSING_OBSERVABLE = "MISSING_OBSERVABLE"
    NON_TARGET_STATE_MISMATCH = "NON_TARGET_STATE_MISMATCH"
    CONTROL_INVALID = "CONTROL_INVALID"


class ScienceLabel(str, Enum):
    CURRENT_PATH_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS = (
        "CURRENT_PATH_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS"
    )
    CURRENT_PATH_CUT_DEPENDENT = "CURRENT_PATH_CUT_DEPENDENT"
    REFRESHED_BOUNDED_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS = (
        "REFRESHED_BOUNDED_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS"
    )
    REFRESHED_BOUNDED_CUT_DEPENDENT = "REFRESHED_BOUNDED_CUT_DEPENDENT"
    TESTED_RECONSTRUCTIONS_INSUFFICIENT_AT_CUTS = (
        "TESTED_RECONSTRUCTIONS_INSUFFICIENT_AT_<CUTS>"
    )


def canonical_json_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def non_target_schema_field_set() -> frozenset[str]:
    return frozenset(NON_TARGET_SNAPSHOT_SCHEMA_FIELDS)


@dataclass(frozen=True)
class NonTargetSnapshot:
    """Run-local TEST EVIDENCE only — never checkpoint authority / persistent-bpw."""

    schema_id: str
    rng_states: dict[str, Any]
    exact_future_batch_sample_ids: tuple[Any, ...]
    loader_cursor: dict[str, Any]
    rate_cap_backlog_schedule: dict[str, Any]
    q_scales_weights_code_hash: dict[str, Any]
    optimizer_empty_proof: dict[str, Any]
    non_manipulated_manifest_fields: dict[str, Any]
    run_local_test_evidence_only: bool = True
    is_checkpoint_authority: bool = False
    contributes_persistent_bpw: bool = False

    def __post_init__(self) -> None:
        if self.schema_id != SCHEMA_ID:
            raise ValueError(f"schema_id must be {SCHEMA_ID}")
        if not self.run_local_test_evidence_only:
            raise ValueError("snapshot must remain run-local TEST EVIDENCE only")
        if self.is_checkpoint_authority:
            raise ValueError("snapshot must NEVER be checkpoint authority")
        if self.contributes_persistent_bpw:
            raise ValueError("snapshot must NEVER contribute persistent-bpw")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash_bundle(self) -> str:
        return canonical_json_sha256(self.to_dict())


def build_non_target_snapshot(
    *,
    rng_states: Mapping[str, Any],
    exact_future_batch_sample_ids: Sequence[Any],
    loader_cursor: Mapping[str, Any],
    rate_cap_backlog_schedule: Mapping[str, Any],
    q_scales_weights_code_hash: Mapping[str, Any],
    optimizer_empty_proof: Mapping[str, Any],
    non_manipulated_manifest_fields: Mapping[str, Any],
) -> NonTargetSnapshot:
    missing = [
        name
        for name in NON_TARGET_SNAPSHOT_SCHEMA_FIELDS
        if name
        not in {
            "rng_states",
            "exact_future_batch_sample_ids",
            "loader_cursor",
            "rate_cap_backlog_schedule",
            "q_scales_weights_code_hash",
            "optimizer_empty_proof",
            "non_manipulated_manifest_fields",
        }
    ]
    if missing:
        raise ValueError(f"schema drift: {missing}")
    return NonTargetSnapshot(
        schema_id=SCHEMA_ID,
        rng_states=dict(rng_states),
        exact_future_batch_sample_ids=tuple(exact_future_batch_sample_ids),
        loader_cursor=dict(loader_cursor),
        rate_cap_backlog_schedule=dict(rate_cap_backlog_schedule),
        q_scales_weights_code_hash=dict(q_scales_weights_code_hash),
        optimizer_empty_proof=dict(optimizer_empty_proof),
        non_manipulated_manifest_fields=dict(non_manipulated_manifest_fields),
    )


def assert_non_target_equality(
    snapshots: Mapping[str, NonTargetSnapshot],
) -> None:
    if not snapshots:
        raise ValueError("NON_TARGET_STATE_MISMATCH: empty snapshot map")
    hashes = {arm: snap.hash_bundle() for arm, snap in sorted(snapshots.items())}
    unique = set(hashes.values())
    if len(unique) != 1:
        raise ValueError(
            "NON_TARGET_STATE_MISMATCH: "
            + json.dumps(hashes, sort_keys=True)
        )


def _walk_collect_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, Mapping):
                out.update(_walk_collect_paths(child, path))
            else:
                out[path] = child
    else:
        out[prefix or "$"] = value
    return out


def manifests_equal_outside_allowlist(
    manifest_c: Mapping[str, Any],
    manifest_s: Mapping[str, Any],
    *,
    allowlist: frozenset[str] = CS_MANIFEST_DIFF_ALLOWLIST,
) -> tuple[bool, tuple[str, ...]]:
    """Return (ok, mismatched_paths). Allowlist matches leaf key names or path suffixes."""

    flat_c = _walk_collect_paths(manifest_c)
    flat_s = _walk_collect_paths(manifest_s)
    keys = sorted(set(flat_c) | set(flat_s))
    mismatched: list[str] = []
    for path in keys:
        leaf = path.rsplit(".", 1)[-1]
        if leaf in allowlist or any(token in path.split(".") for token in allowlist):
            continue
        if flat_c.get(path) != flat_s.get(path):
            mismatched.append(path)
    return (not mismatched, tuple(mismatched))


def assert_cs_manifests_or_mismatch(
    manifest_c: Mapping[str, Any],
    manifest_s: Mapping[str, Any],
) -> None:
    ok, mismatched = manifests_equal_outside_allowlist(manifest_c, manifest_s)
    if not ok:
        raise ValueError(
            "NON_TARGET_STATE_MISMATCH: C/S differ outside bounded-refresh/accounting "
            f"allowlist: {mismatched}"
        )


def parent_seed_scope_tag(
    *,
    parent_sha16: str,
    batch_seed: int,
    support_order_seed: int,
    ordering_seed: int,
    cuts: Sequence[int] = CUTS_DEFAULT,
    k: int = K_DEFAULT,
) -> str:
    cuts_s = "{" + ",".join(str(int(t)) for t in cuts) + "}"
    return (
        f"AT_TESTED_CUT_PARENT_SEED::{parent_sha16}::seed::"
        f"{batch_seed}/{support_order_seed}/{ordering_seed}::cuts::{cuts_s}::K::{int(k)}"
    )


def snapshot_not_loadable_as_checkpoint_authority(snapshot: NonTargetSnapshot) -> bool:
    return (
        snapshot.run_local_test_evidence_only
        and (not snapshot.is_checkpoint_authority)
        and (not snapshot.contributes_persistent_bpw)
    )
