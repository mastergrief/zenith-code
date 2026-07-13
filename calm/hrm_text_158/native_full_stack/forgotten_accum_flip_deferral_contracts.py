"""Contracts for forgotten-accumulator flip-application deferral (Phase-A).

Dense-legacy global-cap path only. Path-agnostic naming; production pin is
``DENSE_LEGACY_GLOBAL_CAP`` at ``bounded_delta_learner.apply_bounded_delta_vote_step``
→ ``apply_global_rate_cap_reference`` (learner :3547).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "forgotten_accum_flip_deferral_contracts/v1"

DENSE_LEGACY_CAP_SITE_ID = (
    "DENSE_LEGACY_apply_global_rate_cap_reference@"
    "bounded_delta_learner.apply_bounded_delta_vote_step"
)

RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0 = "ORDINARY_SELECTOR_SAME_AS_R0"
RELEASE_PATH_DEFERRED_W_NO_AUTHORITATIVE_RELEASE = "DEFERRED_W_NO_AUTHORITATIVE_RELEASE"

PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY = "PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY"


class FlipDeferralMode(str, Enum):
    OFF = "off"
    DURING_W = "during_W"


class CapSiteBranch(str, Enum):
    DENSE_LEGACY_GLOBAL_CAP = "DENSE_LEGACY_GLOBAL_CAP"
    EVENT_CODED_OUT_OF_SCOPE = "EVENT_CODED_OUT_OF_SCOPE"
    NON_CAP_NOT_PRODUCTION = "NON_CAP_NOT_PRODUCTION"


@dataclass(frozen=True)
class WBacklogLaw:
    """Frozen W/backlog law from IMPLEMENTATION_PLAN_v1_1 §2."""

    during_W_carry_updates: bool = True
    during_W_q_mutation: int = 0
    during_W_threshold_residual_writeback: int = 0
    during_W_flip_applied: int = 0
    during_W_authoritative_backlog_mutate: bool = False
    during_W_selector_cap_telemetry: str = "SHADOW_ONLY_NON_AUTHORITATIVE"
    at_W_plus_1_law: str = "PRESERVE_CARRY_THEN_ORDINARY_SELECTOR"
    at_W_plus_1_forbidden: tuple[str, ...] = (
        "preloaded_W_created_backlog",
        "special_backlog_flush",
        "synchronized_burst_release",
        "carry_drop_rebase",
    )


@dataclass(frozen=True)
class DuringWTelemetry:
    acc_hash_pre: str
    acc_hash_post: str
    q_hash: str
    backlog_hash: str
    backlog_cardinality: int
    flip_applied_count: int
    threshold_residual_writeback_count: int
    crossing_demand_count: int
    shadow_accepted_count: int
    shadow_deferred_count: int
    cap_site_branch: str
    flip_application_deferred: bool


@dataclass(frozen=True)
class WPlus1ReleaseRecord:
    pre_vote_carry_hash: str
    crossing_demand_count: int
    selected_count: int
    applied_count: int
    capped_count: int
    backlogged_count: int
    post_step_q_hash: str
    post_step_acc_hash: str
    release_path_id: str
    ordinary_cap: int
    special_backlog_flush: bool


def backlog_cardinality(backlog: Mapping[str, Mapping[int, Mapping[str, Any]]] | None) -> int:
    if not backlog:
        return 0
    return int(sum(len(by_index) for by_index in backlog.values()))
