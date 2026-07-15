from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    APPLY_OUTCOME_SUCCESS, ATTACHMENT_KEY, ExpectedIdentity, PRODUCER_LITERAL,
    VALIDATION_SCHEMA_ID, build_expected_identity_projection, identity_projection_sha256,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    is_option_a_admitted_characterization_geometry as _is_admitted_geom,
)
SCHEMA_ID = "forgotten_accum_a_ledger_accounting_v2/v0"
_ARM_ORDER = ("U", "E", "R0", "RW")
_ARM_KEYS = frozenset(_ARM_ORDER)
_HEX = frozenset("0123456789abcdef")
_ZERO_COUNT_KEYS = ("missing_count", "duplicate_count", "extra_count", "wrong_arm_count")
_REQUIRED_SUMMARY_FLAGS = MappingProxyType({
    "claimable": False, "bankable": False, "forensic_only": True, "runtime_proven": False,
})
V11_TERMINAL_FREEZE_SHA256 = "bdaffbd60311078cefd24d8efd3b19c6d16734b6834b026d14287d4692614635"
REQUIRED_SOURCE_PROVENANCE = MappingProxyType({
    "producer": PRODUCER_LITERAL, "apply_outcome": APPLY_OUTCOME_SUCCESS,
    "validation_schema_id": VALIDATION_SCHEMA_ID,
    "v11_terminal_freeze_sha256": V11_TERMINAL_FREEZE_SHA256,
})
class AccountingV2State(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED_INVALID = "VERIFIED_INVALID"
    VERIFIED_VALID = "VERIFIED_VALID"
IndependentArmGeometry = NamedTuple("IndependentArmGeometry", (
    ("arm_id", str), ("start_step", int), ("steps", int),
    ("expected_local_invocation", int), ("expected_post_cut", int),
))
IndependentExpectedGeometry = NamedTuple("IndependentExpectedGeometry", (
    ("t_cut", int), ("runway_steps", int), ("rewarm_window_steps", int),
    ("shared_prefix_once", int), ("physical_total", int), ("rw_rewarm_window", int),
    ("arms", Mapping[str, IndependentArmGeometry]),
))
@dataclass(frozen=True)
class AccountingV2Result:
    state: AccountingV2State
    reason: str
    schema_id: str = SCHEMA_ID
    details: Mapping[str, Any] | None = None
    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id, "state": self.state.value, "reason": self.reason,
            "details": dict(self.details or {}), "claimable": False, "bankable": False,
            "forensic_only": True, "runtime_proven": False,
        }
class TrustedNormalSuccessCapability:
    __slots__ = ()
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("TrustedNormalSuccessCapability is not caller-constructible")
def _as_int(value: Any) -> int | None:
    return value if type(value) is int else None
def _result(state: AccountingV2State, reason: str, details: Mapping[str, Any] | None = None) -> AccountingV2Result:
    return AccountingV2Result(state=state, reason=reason, details=details)
def _post_cut_count(*, start_step: int, steps: int, t_cut: int) -> int:
    start, end = start_step, start_step + steps
    first = max(start, t_cut + 1)
    return 0 if first >= end else end - first
def _rw_window_count(*, start_step: int, steps: int, t_cut: int, w: int) -> int:
    start, end = start_step, start_step + steps
    first, last = max(start, t_cut + 1), min(end - 1, t_cut + w)
    return 0 if first > last else last - first + 1
def _parse_geom_scalars(t_cut: Any, runway_steps: Any, rewarm_window_steps: Any):
    t, r, w = _as_int(t_cut), _as_int(runway_steps), _as_int(rewarm_window_steps)
    return None if t is None or r is None or w is None else (t, r, w)
def independent_expected_identity_sha256(expected: IndependentArmGeometry) -> str:
    return identity_projection_sha256(build_expected_identity_projection(ExpectedIdentity(
        arm_id=str(expected.arm_id), start_step=int(expected.start_step), steps=int(expected.steps),
    )))
def refuse_unadmitted_characterization_geometry(*, t_cut: Any, runway_steps: Any, rewarm_window_steps: Any) -> AccountingV2Result:
    parsed = _parse_geom_scalars(t_cut, runway_steps, rewarm_window_steps)
    if parsed is None:
        return _result(AccountingV2State.UNVERIFIED, "GEOMETRY_INPUT_MALFORMED", {
            "t_cut": t_cut, "runway_steps": runway_steps, "rewarm_window_steps": rewarm_window_steps,
            "arm_work": 0, "model_work": 0, "gpu_work": 0,
        })
    t, r, w = parsed
    admitted = _is_admitted_geom(t_cut=t, runway_steps=r, rewarm_window_steps=w)
    details = {"t_cut": t, "runway_steps": r, "rewarm_window_steps": w, "arm_work": 0, "model_work": 0, "gpu_work": 0}
    reason = "geometry_is_admitted_refuse_helper_not_applicable" if admitted else "UNADMITTED_CHARACTERIZATION_GEOMETRY"
    return _result(AccountingV2State.UNVERIFIED, reason, details)
def build_independent_expected_geometry(*, t_cut: Any, runway_steps: Any, rewarm_window_steps: Any) -> IndependentExpectedGeometry | AccountingV2Result:
    refuse = refuse_unadmitted_characterization_geometry(
        t_cut=t_cut, runway_steps=runway_steps, rewarm_window_steps=rewarm_window_steps,
    )
    if refuse.reason != "geometry_is_admitted_refuse_helper_not_applicable":
        return refuse
    t = int(refuse.details["t_cut"])
    r = int(refuse.details["runway_steps"])
    w = int(refuse.details["rewarm_window_steps"])
    post = r - t
    if post < w or t < 1 or r <= t:
        return _result(AccountingV2State.UNVERIFIED, "GEOMETRY_ARITHMETIC_INCONSISTENT", {
            "t_cut": t, "runway_steps": r, "rewarm_window_steps": w,
        })
    u = IndependentArmGeometry("U", 1, r, r, _post_cut_count(start_step=1, steps=r, t_cut=t))
    forks = {arm: IndependentArmGeometry(arm, t + 1, post, post, post) for arm in ("E", "R0", "RW")}
    arms = MappingProxyType({"U": u, **forks})
    physical = t + sum(a.expected_post_cut for a in arms.values())
    rw_window = _rw_window_count(
        start_step=forks["RW"].start_step, steps=forks["RW"].steps, t_cut=t, w=w,
    )
    return IndependentExpectedGeometry(t, r, w, t, physical, rw_window, arms)
def extract_attachment_summary(payload: Any) -> Mapping[str, Any] | None:
    if payload is None or not isinstance(payload, Mapping) or ATTACHMENT_KEY not in payload:
        return None
    inner = payload[ATTACHMENT_KEY]
    return inner if isinstance(inner, Mapping) else None
def _provenance_ok(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    prov = payload.get("source_provenance")
    return isinstance(prov, Mapping) and all(prov.get(k) == v for k, v in REQUIRED_SOURCE_PROVENANCE.items())
def _capability_ok(cap: Any, *, arm_id: str) -> bool:
    return False
def _geometries_equal(supplied: IndependentExpectedGeometry, canonical: IndependentExpectedGeometry) -> bool:
    if (supplied.t_cut, supplied.runway_steps, supplied.rewarm_window_steps, supplied.shared_prefix_once, supplied.physical_total, supplied.rw_rewarm_window) != (canonical.t_cut, canonical.runway_steps, canonical.rewarm_window_steps, canonical.shared_prefix_once, canonical.physical_total, canonical.rw_rewarm_window):
        return False
    if not isinstance(supplied.arms, Mapping) or frozenset(supplied.arms.keys()) != _ARM_KEYS:
        return False
    return all(supplied.arms.get(arm) == canonical.arms[arm] for arm in _ARM_ORDER)
def _resolve_canonical_geometry(geometry: Any) -> IndependentExpectedGeometry | AccountingV2Result:
    if not isinstance(geometry, IndependentExpectedGeometry):
        return _result(AccountingV2State.UNVERIFIED, "EXPECTED_GEOMETRY_NOT_INDEPENDENT")
    canonical = build_independent_expected_geometry(
        t_cut=geometry.t_cut, runway_steps=geometry.runway_steps,
        rewarm_window_steps=geometry.rewarm_window_steps,
    )
    if isinstance(canonical, AccountingV2Result) or not _geometries_equal(geometry, canonical):
        return _result(AccountingV2State.UNVERIFIED, "EXPECTED_GEOMETRY_NOT_INDEPENDENT")
    return canonical
def _arm_geometry_ok(expected: Any) -> bool:
    if not isinstance(expected, IndependentArmGeometry):
        return False
    vals = (expected.start_step, expected.steps, expected.expected_local_invocation, expected.expected_post_cut)
    return all(_as_int(v) is not None for v in vals) and isinstance(expected.arm_id, str)
def _check_int_field(summary: Mapping[str, Any], key: str, want: int | None, m: list[str]) -> int | None:
    got = _as_int(summary.get(key))
    if got is None:
        m.append(f"malformed_{key}")
        return None
    if want is not None and got != want:
        m.append(key)
    return got
def _summary_mismatches(summary: Mapping[str, Any], expected: IndependentArmGeometry, *, t_cut: int, rewarm_window_steps: int | None) -> list[str]:
    m: list[str] = []
    if str(summary.get("arm_id", "")) != str(expected.arm_id):
        m.append("arm_id")
    start = _check_int_field(summary, "start_step", int(expected.start_step), m)
    steps = _check_int_field(summary, "steps", int(expected.steps), m)
    _check_int_field(summary, "expected_count", int(expected.steps), m)
    for key in _ZERO_COUNT_KEYS:
        _check_int_field(summary, key, 0, m)
    if summary.get("reorder_detected") is not False:
        m.append("reorder_detected")
    if summary.get("sequence_exact_ok") is not True:
        m.append("sequence_exact_ok")
    for flag, want in _REQUIRED_SUMMARY_FLAGS.items():
        if summary.get(flag) is not want:
            m.append(f"authority_flag_{flag}")
    indep = independent_expected_identity_sha256(expected)
    for key, label in (
        ("expected_identity_projection_sha256", "expected_hash_not_independent"),
        ("observed_identity_projection_sha256", "observed_hash_not_independent"),
    ):
        value = summary.get(key)
        if not isinstance(value, str) or value != indep:
            m.append(label)
    fp = summary.get("full_payload_sha256")
    if not (isinstance(fp, str) and len(fp) == 64 and set(fp.lower()) <= _HEX):
        m.append("malformed_full_payload_sha256")
    _check_int_field(summary, "observed_count", int(expected.expected_local_invocation), m)
    if start is None or steps is None:
        return m
    if _post_cut_count(start_step=start, steps=steps, t_cut=t_cut) != int(expected.expected_post_cut):
        m.append("post_cut")
    if expected.arm_id == "RW" and rewarm_window_steps is not None:
        window = _rw_window_count(start_step=start, steps=steps, t_cut=t_cut, w=rewarm_window_steps)
        if window != rewarm_window_steps:
            m.append("rw_rewarm_window")
    return m
def classify_arm_ordered_event_summary(*, summary_payload: Any, expected: Any, trusted_normal_success: Any, t_cut: Any, rewarm_window_steps: Any = None) -> AccountingV2Result:
    if not _arm_geometry_ok(expected):
        return _result(AccountingV2State.UNVERIFIED, "EXPECTED_ARM_GEOMETRY_MALFORMED")
    if _as_int(t_cut) is None:
        return _result(AccountingV2State.UNVERIFIED, "GEOMETRY_INPUT_MALFORMED", {"arm_id": expected.arm_id})
    if rewarm_window_steps is not None and _as_int(rewarm_window_steps) is None:
        return _result(AccountingV2State.UNVERIFIED, "GEOMETRY_INPUT_MALFORMED", {"arm_id": expected.arm_id})
    arm = {"arm_id": expected.arm_id}
    if not isinstance(summary_payload, Mapping) or ATTACHMENT_KEY not in summary_payload:
        return _result(AccountingV2State.UNVERIFIED, "BARE_OR_ABSENT_ATTACHMENT_KEY", arm)
    if not _provenance_ok(summary_payload):
        return _result(AccountingV2State.UNVERIFIED, "ABSENT_OR_MISMATCHED_SOURCE_PROVENANCE", arm)
    summary = extract_attachment_summary(summary_payload)
    if summary is None:
        return _result(AccountingV2State.UNVERIFIED, "ABSENT_OR_LEGACY_SUMMARY", arm)
    if str(summary.get("schema_id", "")) != VALIDATION_SCHEMA_ID:
        return _result(AccountingV2State.UNVERIFIED, "UNSUPPORTED_SCHEMA", {
            "arm_id": expected.arm_id, "schema_id": summary.get("schema_id"),
        })
    rw = None if rewarm_window_steps is None else int(rewarm_window_steps)
    mismatches = _summary_mismatches(summary, expected, t_cut=int(t_cut), rewarm_window_steps=rw)
    if mismatches:
        return _result(AccountingV2State.VERIFIED_INVALID, "ELIGIBLE_SUMMARY_GEOMETRY_OR_SEQUENCE_MISMATCH", {
            "arm_id": expected.arm_id, "mismatches": mismatches,
        })
    if not _capability_ok(trusted_normal_success, arm_id=expected.arm_id):
        return _result(AccountingV2State.UNVERIFIED, "MISSING_OR_UNTRUSTED_NORMAL_SUCCESS_CAPABILITY", arm)
    return _result(AccountingV2State.VERIFIED_VALID, "CONJUNCTION_OK", arm)
def classify_four_arm_ordered_event_summaries(*, geometry: IndependentExpectedGeometry, arm_summary_payloads: Mapping[str, Any], trusted_capabilities: Mapping[str, Any]) -> AccountingV2Result:
    canonical = _resolve_canonical_geometry(geometry)
    if isinstance(canonical, AccountingV2Result):
        return canonical
    payloads_ok = isinstance(arm_summary_payloads, Mapping) and frozenset(arm_summary_payloads) == _ARM_KEYS
    caps_ok = isinstance(trusted_capabilities, Mapping) and frozenset(trusted_capabilities) == _ARM_KEYS
    if not payloads_ok or not caps_ok:
        return _result(AccountingV2State.UNVERIFIED, "ARM_KEYSET_NOT_EXACT")
    arm_results: dict[str, dict[str, Any]] = {}
    worst = AccountingV2State.VERIFIED_VALID
    for arm in _ARM_ORDER:
        one = classify_arm_ordered_event_summary(
            summary_payload=arm_summary_payloads.get(arm), expected=canonical.arms[arm],
            trusted_normal_success=trusted_capabilities.get(arm),
            t_cut=canonical.t_cut, rewarm_window_steps=canonical.rewarm_window_steps,
        )
        arm_results[arm] = one.as_dict()
        if one.state is AccountingV2State.VERIFIED_INVALID:
            worst = AccountingV2State.VERIFIED_INVALID
        elif one.state is AccountingV2State.UNVERIFIED and worst is AccountingV2State.VERIFIED_VALID:
            worst = AccountingV2State.UNVERIFIED
    details = {"arm_results": arm_results}
    if worst is AccountingV2State.VERIFIED_INVALID:
        return _result(AccountingV2State.VERIFIED_INVALID, "ANY_ARM_VERIFIED_INVALID", details)
    if worst is AccountingV2State.UNVERIFIED:
        return _result(AccountingV2State.UNVERIFIED, "ANY_ARM_UNVERIFIED_OR_TERMINAL_NON_SUCCESS", details)
    return _result(AccountingV2State.VERIFIED_VALID, "FOUR_ARM_CONJUNCTION_OK", details)
__all__ = [
    "ATTACHMENT_KEY", "AccountingV2Result", "AccountingV2State", "IndependentArmGeometry", "IndependentExpectedGeometry", "REQUIRED_SOURCE_PROVENANCE", "SCHEMA_ID",
    "TrustedNormalSuccessCapability", "V11_TERMINAL_FREEZE_SHA256", "build_independent_expected_geometry", "classify_arm_ordered_event_summary",
    "classify_four_arm_ordered_event_summaries", "extract_attachment_summary", "independent_expected_identity_sha256", "refuse_unadmitted_characterization_geometry",
]
