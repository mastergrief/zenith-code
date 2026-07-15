"""Pure accounting-v2 helpers: authority-neutral facts only (no final VALID gate)."""
from __future__ import annotations
from types import MappingProxyType
from typing import Any, Mapping
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ATTACHMENT_KEY, ExpectedIdentity, build_expected_identity_projection, identity_projection_sha256,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    is_option_a_admitted_characterization_geometry as _is_admitted_geom,
)
ARM_ORDER = ("U", "E", "R0", "RW")
ARM_KEYS = frozenset(ARM_ORDER)
_HEX = frozenset("0123456789abcdef")
ZERO_COUNT_KEYS = ("missing_count", "duplicate_count", "extra_count", "wrong_arm_count")
REQUIRED_SUMMARY_FLAGS = MappingProxyType({
    "claimable": False, "bankable": False, "forensic_only": True, "runtime_proven": False,
})
def as_int(value: Any) -> int | None:
    return value if type(value) is int else None
def post_cut_count(*, start_step: int, steps: int, t_cut: int) -> int:
    start, end = start_step, start_step + steps
    first = max(start, t_cut + 1)
    return 0 if first >= end else end - first
def rw_window_count(*, start_step: int, steps: int, t_cut: int, w: int) -> int:
    start, end = start_step, start_step + steps
    first, last = max(start, t_cut + 1), min(end - 1, t_cut + w)
    return 0 if first > last else last - first + 1
def parse_geom_scalars(t_cut: Any, runway_steps: Any, rewarm_window_steps: Any):
    t, r, w = as_int(t_cut), as_int(runway_steps), as_int(rewarm_window_steps)
    return None if t is None or r is None or w is None else (t, r, w)
def identity_sha256(*, arm_id: str, start_step: int, steps: int) -> str:
    return identity_projection_sha256(build_expected_identity_projection(ExpectedIdentity(
        arm_id=str(arm_id), start_step=int(start_step), steps=int(steps),
    )))
def geometry_input_facts(t_cut: Any, runway_steps: Any, rewarm_window_steps: Any) -> dict[str, Any]:
    parsed = parse_geom_scalars(t_cut, runway_steps, rewarm_window_steps)
    if parsed is None:
        return {"status": "malformed", "details": {
            "t_cut": t_cut, "runway_steps": runway_steps, "rewarm_window_steps": rewarm_window_steps,
            "arm_work": 0, "model_work": 0, "gpu_work": 0,
        }}
    t, r, w = parsed
    details = {"t_cut": t, "runway_steps": r, "rewarm_window_steps": w, "arm_work": 0, "model_work": 0, "gpu_work": 0}
    if not _is_admitted_geom(t_cut=t, runway_steps=r, rewarm_window_steps=w):
        return {"status": "unadmitted", "details": details}
    if (r - t) < w or t < 1 or r <= t:
        return {"status": "arithmetic_inconsistent", "details": {"t_cut": t, "runway_steps": r, "rewarm_window_steps": w}}
    return {"status": "admitted", "t_cut": t, "runway_steps": r, "rewarm_window_steps": w}
def canonical_geometry_facts(*, t_cut: int, runway_steps: int, rewarm_window_steps: int) -> dict[str, Any]:
    t, r, w = int(t_cut), int(runway_steps), int(rewarm_window_steps)
    post = r - t
    u = {"arm_id": "U", "start_step": 1, "steps": r, "expected_local_invocation": r,
         "expected_post_cut": post_cut_count(start_step=1, steps=r, t_cut=t)}
    forks = {arm: {"arm_id": arm, "start_step": t + 1, "steps": post,
                   "expected_local_invocation": post, "expected_post_cut": post} for arm in ("E", "R0", "RW")}
    arms = {"U": u, **forks}
    physical = t + sum(a["expected_post_cut"] for a in arms.values())
    rw_window = rw_window_count(start_step=forks["RW"]["start_step"], steps=forks["RW"]["steps"], t_cut=t, w=w)
    return {"t_cut": t, "runway_steps": r, "rewarm_window_steps": w, "shared_prefix_once": t,
            "physical_total": physical, "rw_rewarm_window": rw_window, "arms": arms}
def extract_attachment_summary(payload: Any) -> Mapping[str, Any] | None:
    if payload is None or not isinstance(payload, Mapping) or ATTACHMENT_KEY not in payload:
        return None
    inner = payload[ATTACHMENT_KEY]
    return inner if isinstance(inner, Mapping) else None
def provenance_matches(payload: Any, required: Mapping[str, Any]) -> bool:
    if not isinstance(payload, Mapping):
        return False
    prov = payload.get("source_provenance")
    return isinstance(prov, Mapping) and all(prov.get(k) == v for k, v in required.items())
def arm_geometry_fields_ok(arm_id: Any, start_step: Any, steps: Any, local: Any, post: Any) -> bool:
    return isinstance(arm_id, str) and all(as_int(v) is not None for v in (start_step, steps, local, post))
def _check_int_field(summary: Mapping[str, Any], key: str, want: int | None, m: list[str]) -> int | None:
    got = as_int(summary.get(key))
    if got is None:
        m.append(f"malformed_{key}")
        return None
    if want is not None and got != want:
        m.append(key)
    return got
def summary_mismatches(
    summary: Mapping[str, Any], *, arm_id: str, start_step: int, steps: int,
    expected_local_invocation: int, expected_post_cut: int, t_cut: int, rewarm_window_steps: int | None,
) -> list[str]:
    m: list[str] = []
    if str(summary.get("arm_id", "")) != str(arm_id):
        m.append("arm_id")
    start = _check_int_field(summary, "start_step", int(start_step), m)
    got_steps = _check_int_field(summary, "steps", int(steps), m)
    _check_int_field(summary, "expected_count", int(steps), m)
    for key in ZERO_COUNT_KEYS:
        _check_int_field(summary, key, 0, m)
    if summary.get("reorder_detected") is not False:
        m.append("reorder_detected")
    if summary.get("sequence_exact_ok") is not True:
        m.append("sequence_exact_ok")
    for flag, want in REQUIRED_SUMMARY_FLAGS.items():
        if summary.get(flag) is not want:
            m.append(f"authority_flag_{flag}")
    indep = identity_sha256(arm_id=arm_id, start_step=start_step, steps=steps)
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
    _check_int_field(summary, "observed_count", int(expected_local_invocation), m)
    if start is None or got_steps is None:
        return m
    if post_cut_count(start_step=start, steps=got_steps, t_cut=t_cut) != int(expected_post_cut):
        m.append("post_cut")
    if arm_id == "RW" and rewarm_window_steps is not None:
        window = rw_window_count(start_step=start, steps=got_steps, t_cut=t_cut, w=rewarm_window_steps)
        if window != rewarm_window_steps:
            m.append("rw_rewarm_window")
    return m
def arm_maps_exact(payloads: Any, capabilities: Any) -> bool:
    return (isinstance(payloads, Mapping) and frozenset(payloads) == ARM_KEYS
            and isinstance(capabilities, Mapping) and frozenset(capabilities) == ARM_KEYS)
def geometry_facts_match(supplied: Mapping[str, Any], canonical: Mapping[str, Any]) -> bool:
    outer = ("t_cut", "runway_steps", "rewarm_window_steps", "shared_prefix_once", "physical_total", "rw_rewarm_window")
    if any(supplied.get(k) != canonical.get(k) for k in outer):
        return False
    arms = supplied.get("arms")
    if not isinstance(arms, Mapping) or frozenset(arms.keys()) != ARM_KEYS:
        return False
    keys = ("arm_id", "start_step", "steps", "expected_local_invocation", "expected_post_cut")
    for arm in ARM_ORDER:
        got, want = arms.get(arm), canonical["arms"][arm]
        if not isinstance(got, Mapping) or any(got.get(k) != want[k] for k in keys):
            return False
    return True
def arm_precheck_reason(*, summary_payload: Any, expected_schema_id: str, required_provenance: Mapping[str, Any]):
    if not isinstance(summary_payload, Mapping) or ATTACHMENT_KEY not in summary_payload:
        return ("BARE_OR_ABSENT_ATTACHMENT_KEY", None)
    if not provenance_matches(summary_payload, required_provenance):
        return ("ABSENT_OR_MISMATCHED_SOURCE_PROVENANCE", None)
    summary = extract_attachment_summary(summary_payload)
    if summary is None:
        return ("ABSENT_OR_LEGACY_SUMMARY", None)
    if str(summary.get("schema_id", "")) != expected_schema_id:
        return ("UNSUPPORTED_SCHEMA", {"schema_id": summary.get("schema_id")})
    return None

def geometry_attr_facts(geometry: Any) -> dict[str, Any]:
    """Duck-typed geometry -> facts mapping (no public type import)."""
    arms_obj = geometry.arms
    arms = {}
    for arm in arms_obj:
        item = arms_obj[arm]
        arms[arm] = {
            "arm_id": item.arm_id,
            "start_step": item.start_step,
            "steps": item.steps,
            "expected_local_invocation": item.expected_local_invocation,
            "expected_post_cut": item.expected_post_cut,
        }
    return {
        "t_cut": geometry.t_cut, "runway_steps": geometry.runway_steps,
        "rewarm_window_steps": geometry.rewarm_window_steps,
        "shared_prefix_once": geometry.shared_prefix_once, "physical_total": geometry.physical_total,
        "rw_rewarm_window": geometry.rw_rewarm_window, "arms": arms,
    }


def worst_arm_state(states: list[str]) -> str:
    # Authority-neutral tokens only. Facade maps to public enum/reasons.
    if "VERIFIED_INVALID" in states:
        return "W_INVALID"
    if "UNVERIFIED" in states:
        return "W_UNVERIFIED"
    return "W_ALL_OK"


def evaluate_arm_summary_facts(
    *,
    summary_payload: Any,
    arm_id: Any,
    start_step: Any,
    steps: Any,
    expected_local_invocation: Any,
    expected_post_cut: Any,
    t_cut: Any,
    rewarm_window_steps: Any,
    expected_schema_id: str,
    required_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Authority-neutral arm evaluation. Never returns final VALID."""
    if not arm_geometry_fields_ok(arm_id, start_step, steps, expected_local_invocation, expected_post_cut):
        return {"status": "arm_geom_malformed"}
    if as_int(t_cut) is None or (rewarm_window_steps is not None and as_int(rewarm_window_steps) is None):
        return {"status": "geom_input_malformed", "arm_id": arm_id}
    pre = arm_precheck_reason(
        summary_payload=summary_payload, expected_schema_id=expected_schema_id,
        required_provenance=required_provenance)
    if pre is not None:
        reason, extra = pre
        return {"status": "precheck", "reason": reason, "arm_id": arm_id, "extra": extra}
    summary = extract_attachment_summary(summary_payload)
    assert summary is not None
    rw = None if rewarm_window_steps is None else int(rewarm_window_steps)
    mismatches = summary_mismatches(
        summary, arm_id=str(arm_id), start_step=int(start_step), steps=int(steps),
        expected_local_invocation=int(expected_local_invocation),
        expected_post_cut=int(expected_post_cut), t_cut=int(t_cut), rewarm_window_steps=rw)
    if mismatches:
        return {"status": "mismatch", "arm_id": arm_id, "mismatches": mismatches}
    return {"status": "eligible", "arm_id": arm_id}
