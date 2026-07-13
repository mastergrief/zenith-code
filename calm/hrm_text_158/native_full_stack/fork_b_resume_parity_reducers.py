"""Fork B resume-parity pure reducers (classifier + S accounting + surfaces).

CPU-PURE: no filesystem / torch / GPU / trainer_sub2 / launch imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_contracts import (
    CERTIFICATE_SCHEMA,
    CUTS_DEFAULT,
    DENSE_SHADOW_FIELD_PERSISTENT_BPW,
    GATE_BEARING_FIELDS,
    PreScienceClass,
    ScienceLabel,
    Z_BINDING_CUT_T,
)


@dataclass(frozen=True)
class SAccountingLedger:
    cut_t: int
    pre_refresh_bounded_bits: int
    post_refresh_bounded_bits: int
    delta_bits: int
    schema_metadata_delta_bits: int
    fixed_size_packed_slab_delta_bits: int
    dense_shadow_field_persistent_bpw: int = DENSE_SHADOW_FIELD_PERSISTENT_BPW

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_s_accounting(
    *,
    cut_t: int,
    pre_refresh_bounded_bits: int,
    post_refresh_bounded_bits: int,
    schema_metadata_delta_bits: int = 0,
    fixed_size_packed_overwrite: bool = False,
) -> SAccountingLedger:
    # Fixed-size packed overwrite: slab incremental bpw delta is 0; charge metadata only.
    # Variable hot/cold: charge growth (post-pre) + metadata.
    if fixed_size_packed_overwrite:
        delta = int(schema_metadata_delta_bits)
        packed_slab_delta = 0
    else:
        delta = (
            int(post_refresh_bounded_bits)
            - int(pre_refresh_bounded_bits)
            + int(schema_metadata_delta_bits)
        )
        packed_slab_delta = 0
    return SAccountingLedger(
        cut_t=int(cut_t),
        pre_refresh_bounded_bits=int(pre_refresh_bounded_bits),
        post_refresh_bounded_bits=int(post_refresh_bounded_bits),
        delta_bits=int(delta),
        schema_metadata_delta_bits=int(schema_metadata_delta_bits),
        fixed_size_packed_slab_delta_bits=int(packed_slab_delta),
    )


def extract_comparison_surface(stats: Mapping[str, Any]) -> dict[str, Any]:
    surface = {key: stats.get(key) for key in GATE_BEARING_FIELDS}
    surface["q_sha256_before"] = stats.get("q_sha256_before")
    surface["post_rehydrate_or_live_acc_sha256"] = stats.get(
        "exact_accumulator_shadow_sha256_after"
    ) or stats.get("post_rehydrate_or_live_acc_sha256")
    return surface


def surfaces_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    keys = sorted(set(a) | set(b))
    return all(a.get(key) == b.get(key) for key in keys)


def z_decision_sensitive(
    *,
    z_surface: Mapping[str, Any],
    u_surface: Mapping[str, Any],
    f_surface: Mapping[str, Any] | None = None,
) -> bool:
    """True iff Z diverges from U (and F if provided) on a GATE-BEARING field."""

    def _diverges(other: Mapping[str, Any]) -> bool:
        for key in GATE_BEARING_FIELDS:
            if z_surface.get(key) != other.get(key):
                return True
        return False

    if not _diverges(u_surface):
        return False
    if f_surface is not None and not _diverges(f_surface):
        return False
    return True


@dataclass
class PerCutArmResult:
    cut_t: int
    arm: str
    comparison_surface_by_step: dict[str, dict[str, Any]] = field(default_factory=dict)
    pre_science: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_t": self.cut_t,
            "arm": self.arm,
            "comparison_surface_by_step": self.comparison_surface_by_step,
            "pre_science": self.pre_science,
            "notes": self.notes,
        }


@dataclass
class PerCutResult:
    cut_t: int
    f_matches_u: bool | None = None
    z_decision_sensitive: bool | None = None
    c_matches_u: bool | None = None
    s_matches_u: bool | None = None
    pre_science: str | None = None
    non_target_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _all_cuts_true(per_cut: Mapping[int, PerCutResult], attr: str) -> bool:
    if not per_cut:
        return False
    return all(getattr(result, attr) is True for result in per_cut.values())


def classify_terminal(
    *,
    per_cut: Mapping[int, PerCutResult],
    cuts: Sequence[int] = CUTS_DEFAULT,
    z_binding_cut_t: int = Z_BINDING_CUT_T,
    parent_seed_scope: str,
) -> dict[str, Any]:
    """Fail-closed classifier. Never emits dense-shadow necessity labels."""

    ordered = {int(t): per_cut[int(t)] for t in cuts if int(t) in per_cut}
    missing_cuts = [int(t) for t in cuts if int(t) not in per_cut]
    if missing_cuts:
        return {
            "schema": CERTIFICATE_SCHEMA,
            "pre_science": PreScienceClass.MISSING_OBSERVABLE.value,
            "science_label": None,
            "parent_seed_scope": parent_seed_scope,
            "missing_cuts": missing_cuts,
            "dense_shadow_field_persistent_bpw": DENSE_SHADOW_FIELD_PERSISTENT_BPW,
            "explicitly_not": ["dense_int16_shadow_necessity", "universal_C1_moot"],
        }

    # Pre-science precedence
    for result in ordered.values():
        if result.pre_science == PreScienceClass.INFRA_FAILURE.value:
            return _pre(PreScienceClass.INFRA_FAILURE, parent_seed_scope, ordered)
        if result.non_target_ok is False or (
            result.pre_science == PreScienceClass.NON_TARGET_STATE_MISMATCH.value
        ):
            return _pre(
                PreScienceClass.NON_TARGET_STATE_MISMATCH, parent_seed_scope, ordered
            )
        if result.pre_science == PreScienceClass.MISSING_OBSERVABLE.value:
            return _pre(PreScienceClass.MISSING_OBSERVABLE, parent_seed_scope, ordered)

    # F validity
    if any(result.f_matches_u is not True for result in ordered.values()):
        return _pre(PreScienceClass.CONTROL_INVALID, parent_seed_scope, ordered)

    # Z@t16 binding validates aggregate
    binding = ordered.get(int(z_binding_cut_t))
    if binding is None or binding.z_decision_sensitive is not True:
        return _pre(PreScienceClass.CONTROL_INVALID, parent_seed_scope, ordered)

    c_ok = {t: ordered[t].c_matches_u is True for t in ordered}
    s_ok = {t: ordered[t].s_matches_u is True for t in ordered}
    c_all = all(c_ok.values())
    s_all = all(s_ok.values())
    c_any = any(c_ok.values())
    s_any = any(s_ok.values())
    c_fail_cuts = sorted(t for t, ok in c_ok.items() if not ok)
    s_fail_cuts = sorted(t for t, ok in s_ok.items() if not ok)
    both_fail_cuts = sorted(t for t in ordered if (not c_ok[t]) and (not s_ok[t]))

    # Design v3 aggregate priority:
    # CURRENT all → REFRESHED all (C fails ≥1 & S all) →
    # INSUFFICIENT (cuts where C!=U AND S!=U) → CURRENT cut-dep → REFRESHED cut-dep
    if c_all:
        label = ScienceLabel.CURRENT_PATH_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS.value
    elif s_all:
        label = ScienceLabel.REFRESHED_BOUNDED_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS.value
    elif both_fail_cuts:
        label = (
            "TESTED_RECONSTRUCTIONS_INSUFFICIENT_AT_"
            + "+".join(str(t) for t in both_fail_cuts)
        )
    elif c_any:
        label = ScienceLabel.CURRENT_PATH_CUT_DEPENDENT.value
    elif s_any:
        label = ScienceLabel.REFRESHED_BOUNDED_CUT_DEPENDENT.value
    else:
        insufficient = sorted(set(c_fail_cuts) | set(s_fail_cuts)) or sorted(ordered)
        label = (
            "TESTED_RECONSTRUCTIONS_INSUFFICIENT_AT_"
            + "+".join(str(t) for t in insufficient)
        )

    return {
        "schema": CERTIFICATE_SCHEMA,
        "pre_science": None,
        "science_label": label,
        "parent_seed_scope": parent_seed_scope,
        "per_cut": {str(t): ordered[t].to_dict() for t in ordered},
        "c_fail_cuts": c_fail_cuts,
        "s_fail_cuts": s_fail_cuts,
        "dense_shadow_field_persistent_bpw": DENSE_SHADOW_FIELD_PERSISTENT_BPW,
        "explicitly_not": [
            "dense_int16_shadow_necessity",
            "universal_C1_moot",
            "SHADOW_NECESSARY_PERSISTENT",
        ],
        "z_binding_cut_t": int(z_binding_cut_t),
    }


def _pre(
    cls: PreScienceClass,
    parent_seed_scope: str,
    ordered: Mapping[int, PerCutResult],
) -> dict[str, Any]:
    return {
        "schema": CERTIFICATE_SCHEMA,
        "pre_science": cls.value,
        "science_label": None,
        "parent_seed_scope": parent_seed_scope,
        "per_cut": {str(t): ordered[t].to_dict() for t in ordered},
        "dense_shadow_field_persistent_bpw": DENSE_SHADOW_FIELD_PERSISTENT_BPW,
        "explicitly_not": ["dense_int16_shadow_necessity", "universal_C1_moot"],
    }
