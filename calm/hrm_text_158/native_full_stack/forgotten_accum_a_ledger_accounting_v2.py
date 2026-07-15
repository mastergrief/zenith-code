from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    APPLY_OUTCOME_SUCCESS, ATTACHMENT_KEY, PRODUCER_LITERAL, VALIDATION_SCHEMA_ID,
)
from calm.hrm_text_158.native_full_stack import (
    forgotten_accum_a_ledger_accounting_v2_core as _core,
)
SCHEMA_ID = "forgotten_accum_a_ledger_accounting_v2/v0"
V11_TERMINAL_FREEZE_SHA256 = (
    "bdaffbd60311078cefd24d8efd3b19c6d16734b6834b026d14287d4692614635"
)
REQUIRED_SOURCE_PROVENANCE = MappingProxyType({
    "producer": PRODUCER_LITERAL,
    "apply_outcome": APPLY_OUTCOME_SUCCESS,
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
    ("shared_prefix_once", int), ("physical_total", int),
    ("rw_rewarm_window", int), ("arms", Mapping[str, IndependentArmGeometry]),
))
@dataclass(frozen=True)
class AccountingV2Result:
    state: AccountingV2State
    reason: str
    schema_id: str = SCHEMA_ID
    details: Mapping[str, Any] | None = None
    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id, "state": self.state.value,
            "reason": self.reason, "details": dict(self.details or {}),
            "claimable": False, "bankable": False,
            "forensic_only": True, "runtime_proven": False,
        }
class TrustedNormalSuccessCapability:
    __slots__ = ()
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("TrustedNormalSuccessCapability is not caller-constructible")
_U, _I, _V = AccountingV2State.UNVERIFIED, AccountingV2State.VERIFIED_INVALID, AccountingV2State.VERIFIED_VALID
def _capability_ok(cap: Any, *, arm_id: str) -> bool:
    return False
def _dec(state: AccountingV2State, reason: str, details: Mapping[str, Any] | None = None) -> AccountingV2Result:
    return AccountingV2Result(state=state, reason=reason, details=details)
def _geometry_from_facts(f: Mapping[str, Any]) -> IndependentExpectedGeometry:
    def _arm(x: Mapping[str, Any]) -> IndependentArmGeometry:
        return IndependentArmGeometry(
            str(x["arm_id"]), int(x["start_step"]), int(x["steps"]),
            int(x["expected_local_invocation"]), int(x["expected_post_cut"]),
        )
    arms = MappingProxyType({a: _arm(f["arms"][a]) for a in _core.ARM_ORDER})
    return IndependentExpectedGeometry(
        int(f["t_cut"]), int(f["runway_steps"]), int(f["rewarm_window_steps"]),
        int(f["shared_prefix_once"]), int(f["physical_total"]),
        int(f["rw_rewarm_window"]), arms,
    )
_EARLY = {
    "malformed": "GEOMETRY_INPUT_MALFORMED",
    "unadmitted": "UNADMITTED_CHARACTERIZATION_GEOMETRY",
}
def _from_geom_input(facts: Mapping[str, Any]) -> AccountingV2Result | None:
    reason = _EARLY.get(facts["status"])
    return None if reason is None else _dec(_U, reason, facts["details"])
def independent_expected_identity_sha256(expected: IndependentArmGeometry) -> str:
    return _core.identity_sha256(
        arm_id=expected.arm_id, start_step=expected.start_step, steps=expected.steps,
    )
def refuse_unadmitted_characterization_geometry(
    *, t_cut: Any, runway_steps: Any, rewarm_window_steps: Any,
) -> AccountingV2Result:
    facts = _core.geometry_input_facts(t_cut, runway_steps, rewarm_window_steps)
    if (early := _from_geom_input(facts)) is not None:
        return early
    d = facts["details"] if facts["status"] == "arithmetic_inconsistent" else facts
    return _dec(_U, "geometry_is_admitted_refuse_helper_not_applicable", {
        "t_cut": d["t_cut"], "runway_steps": d["runway_steps"],
        "rewarm_window_steps": d["rewarm_window_steps"],
        "arm_work": 0, "model_work": 0, "gpu_work": 0,
    })
def build_independent_expected_geometry(
    *, t_cut: Any, runway_steps: Any, rewarm_window_steps: Any,
) -> IndependentExpectedGeometry | AccountingV2Result:
    facts = _core.geometry_input_facts(t_cut, runway_steps, rewarm_window_steps)
    if (early := _from_geom_input(facts)) is not None:
        return early
    if facts["status"] == "arithmetic_inconsistent":
        return _dec(_U, "GEOMETRY_ARITHMETIC_INCONSISTENT", facts["details"])
    return _geometry_from_facts(_core.canonical_geometry_facts(
        t_cut=facts["t_cut"], runway_steps=facts["runway_steps"],
        rewarm_window_steps=facts["rewarm_window_steps"],
    ))
def extract_attachment_summary(payload: Any) -> Mapping[str, Any] | None:
    return _core.extract_attachment_summary(payload)
def classify_arm_ordered_event_summary(
    *, summary_payload: Any, expected: Any, trusted_normal_success: Any,
    t_cut: Any, rewarm_window_steps: Any = None,
) -> AccountingV2Result:
    if not isinstance(expected, IndependentArmGeometry):
        return _dec(_U, "EXPECTED_ARM_GEOMETRY_MALFORMED")
    e = expected
    facts = _core.evaluate_arm_summary_facts(
        summary_payload=summary_payload, arm_id=e.arm_id,
        start_step=e.start_step, steps=e.steps,
        expected_local_invocation=e.expected_local_invocation,
        expected_post_cut=e.expected_post_cut, t_cut=t_cut,
        rewarm_window_steps=rewarm_window_steps,
        expected_schema_id=VALIDATION_SCHEMA_ID,
        required_provenance=REQUIRED_SOURCE_PROVENANCE,
    )
    st, arm = facts["status"], {"arm_id": facts.get("arm_id")}
    if st == "arm_geom_malformed":
        return _dec(_U, "EXPECTED_ARM_GEOMETRY_MALFORMED")
    if st == "geom_input_malformed":
        return _dec(_U, "GEOMETRY_INPUT_MALFORMED", arm)
    if st == "precheck":
        return _dec(_U, facts["reason"], {**arm, **(facts["extra"] or {})})
    if st == "mismatch":
        return _dec(_I, "ELIGIBLE_SUMMARY_GEOMETRY_OR_SEQUENCE_MISMATCH", {
            **arm, "mismatches": facts["mismatches"],
        })
    if not _capability_ok(trusted_normal_success, arm_id=str(facts["arm_id"])):
        return _dec(_U, "MISSING_OR_UNTRUSTED_NORMAL_SUCCESS_CAPABILITY", arm)
    return _dec(_V, "CONJUNCTION_OK", arm)
def classify_four_arm_ordered_event_summaries(
    *, geometry: IndependentExpectedGeometry,
    arm_summary_payloads: Mapping[str, Any],
    trusted_capabilities: Mapping[str, Any],
) -> AccountingV2Result:
    if not isinstance(geometry, IndependentExpectedGeometry):
        return _dec(_U, "EXPECTED_GEOMETRY_NOT_INDEPENDENT")
    built = build_independent_expected_geometry(
        t_cut=geometry.t_cut, runway_steps=geometry.runway_steps,
        rewarm_window_steps=geometry.rewarm_window_steps,
    )
    geom_ok = not isinstance(built, AccountingV2Result) and _core.geometry_facts_match(
        _core.geometry_attr_facts(geometry), _core.geometry_attr_facts(built),
    )
    if not geom_ok:
        return _dec(_U, "EXPECTED_GEOMETRY_NOT_INDEPENDENT")
    if not _core.arm_maps_exact(arm_summary_payloads, trusted_capabilities):
        return _dec(_U, "ARM_KEYSET_NOT_EXACT")
    arm_results: dict[str, Any] = {}
    states: list[str] = []
    for arm in _core.ARM_ORDER:
        one = classify_arm_ordered_event_summary(
            summary_payload=arm_summary_payloads.get(arm), expected=built.arms[arm],
            trusted_normal_success=trusted_capabilities.get(arm),
            t_cut=built.t_cut, rewarm_window_steps=built.rewarm_window_steps,
        )
        arm_results[arm] = one.as_dict()
        states.append(one.state.value)
    worst = _core.worst_arm_state(states)
    details = {"arm_results": arm_results}
    if worst == "W_INVALID":
        return _dec(_I, "ANY_ARM_VERIFIED_INVALID", details)
    if worst == "W_UNVERIFIED":
        return _dec(_U, "ANY_ARM_UNVERIFIED_OR_TERMINAL_NON_SUCCESS", details)
    return _dec(_V, "FOUR_ARM_CONJUNCTION_OK", details)
__all__ = [
    "ATTACHMENT_KEY", "AccountingV2Result", "AccountingV2State",
    "IndependentArmGeometry", "IndependentExpectedGeometry",
    "REQUIRED_SOURCE_PROVENANCE", "SCHEMA_ID", "TrustedNormalSuccessCapability",
    "V11_TERMINAL_FREEZE_SHA256", "build_independent_expected_geometry",
    "classify_arm_ordered_event_summary", "classify_four_arm_ordered_event_summaries",
    "extract_attachment_summary", "independent_expected_identity_sha256",
    "refuse_unadmitted_characterization_geometry",
]
