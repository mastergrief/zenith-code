"""Step A: CPU adversarial matrix for accounting-v2 ordered-event summary consumer."""
from __future__ import annotations
import ast, hashlib, json
from pathlib import Path
from types import MappingProxyType
import pytest
from calm.hrm_text_158.native_full_stack import forgotten_accum_a_ledger_accounting_v2 as acct
from calm.hrm_text_158.native_full_stack.forgotten_accum_a_ledger_accounting_v2 import (
    AccountingV2State, IndependentArmGeometry, IndependentExpectedGeometry,
    REQUIRED_SOURCE_PROVENANCE, TrustedNormalSuccessCapability,
    build_independent_expected_geometry, classify_arm_ordered_event_summary,
    classify_four_arm_ordered_event_summaries, independent_expected_identity_sha256,
    refuse_unadmitted_characterization_geometry,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ATTACHMENT_KEY, ExpectedIdentity, make_success_apply_event,
    snapshot_ordered_apply_event_log, validate_ordered_apply_event_sequence,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    is_option_a_admitted_characterization_geometry,
)
REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "calm/hrm_text_158/native_full_stack/forgotten_accum_a_ledger_accounting_v2.py"
def _cap() -> TrustedNormalSuccessCapability:
    return object.__new__(TrustedNormalSuccessCapability)  # TEST-ONLY; trust via monkeypatch
@pytest.fixture
def trust_caps(monkeypatch):
    registry: dict[str, object] = {}
    monkeypatch.setattr(acct, "_capability_ok", lambda cap, *, arm_id: registry.get(str(arm_id)) is cap)
    def mint(arm_id: str):
        cap = _cap()
        registry[str(arm_id)] = cap
        return cap
    return mint
def _exact_summary(*, arm_id: str, start_step: int, steps: int) -> dict:
    events = [
        make_success_apply_event(
            seq=i, arm_id=arm_id, optimizer_step_id=start_step + i,
            q_changed_count=1, tensor_state_key_count=1,
        ) for i in range(steps)
    ]
    return validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events),
        ExpectedIdentity(arm_id=arm_id, start_step=start_step, steps=steps),
    )
def _envelope(summary: dict) -> dict:
    return {ATTACHMENT_KEY: summary, "source_provenance": dict(REQUIRED_SOURCE_PROVENANCE)}
def _geom_241():
    g = build_independent_expected_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=1)
    assert not isinstance(g, acct.AccountingV2Result)
    return g
def _exact_payloads(geometry) -> dict[str, dict]:
    return {
        arm: _envelope(_exact_summary(arm_id=ag.arm_id, start_step=ag.start_step, steps=ag.steps))
        for arm, ag in geometry.arms.items()
    }
def _four(geometry, payloads, caps):
    return classify_four_arm_ordered_event_summaries(
        geometry=geometry, arm_summary_payloads=payloads, trusted_capabilities=caps,
    )
def _one(payload, expected, cap, t_cut, rewarm=None):
    return classify_arm_ordered_event_summary(
        summary_payload=payload, expected=expected, trusted_normal_success=cap,
        t_cut=t_cut, rewarm_window_steps=rewarm,
    )
def _assert_geom_not_indep(result):
    assert result.state is AccountingV2State.UNVERIFIED
    assert result.reason == "EXPECTED_GEOMETRY_NOT_INDEPENDENT"
    assert result.state is not AccountingV2State.VERIFIED_VALID
def test_live_predicate_admits_241_refuses_242():
    assert is_option_a_admitted_characterization_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=1)
    assert not is_option_a_admitted_characterization_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=2)
def test_unadmitted_242_refuses_before_arm_work():
    refuse = refuse_unadmitted_characterization_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=2)
    assert refuse.state is AccountingV2State.UNVERIFIED
    assert refuse.reason == "UNADMITTED_CHARACTERIZATION_GEOMETRY"
    assert refuse.details["arm_work"] == refuse.details["model_work"] == refuse.details["gpu_work"] == 0
    built = build_independent_expected_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=2)
    assert isinstance(built, acct.AccountingV2Result)
    assert built.reason == "UNADMITTED_CHARACTERIZATION_GEOMETRY"
def test_241_geometry_arithmetic():
    g = _geom_241()
    assert (g.t_cut, g.runway_steps, g.rewarm_window_steps) == (2, 4, 1)
    assert g.shared_prefix_once == 2 and g.physical_total == 10 and g.rw_rewarm_window == 1
    assert g.arms["U"].expected_local_invocation == 4 and g.arms["U"].expected_post_cut == 2
    for arm in ("E", "R0", "RW"):
        assert g.arms[arm].start_step == 3
        assert g.arms[arm].expected_local_invocation == g.arms[arm].expected_post_cut == 2
def test_exact_241_four_arm_verified_valid(trust_caps):
    g = _geom_241()
    caps = {arm: trust_caps(arm) for arm in g.arms}
    result = _four(g, _exact_payloads(g), caps)
    assert result.state is AccountingV2State.VERIFIED_VALID
    assert result.reason == "FOUR_ARM_CONJUNCTION_OK"
    d = result.as_dict()
    assert d["claimable"] is False and d["runtime_proven"] is False
def test_production_capability_ok_always_false_without_monkeypatch():
    """Production guard is fail-closed; object.__new__ alone never grants VALID."""
    g = _geom_241()
    cap = _cap()
    try:
        object.__setattr__(cap, "_arm_id", "U")
    except Exception:
        pass
    one = _one(_envelope(_exact_summary(arm_id="U", start_step=1, steps=4)), g.arms["U"], cap, g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED
    assert one.reason == "MISSING_OR_UNTRUSTED_NORMAL_SUCCESS_CAPABILITY"
    assert one.state is not AccountingV2State.VERIFIED_VALID
    assert acct._capability_ok(cap, arm_id="U") is False
def test_missing_summary_unverified(trust_caps):
    g = _geom_241()
    one = _one(None, g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED and one.reason == "BARE_OR_ABSENT_ATTACHMENT_KEY"
def test_legacy_unsupported_schema_unverified(trust_caps):
    g = _geom_241()
    one = _one(_envelope({"schema_id": "legacy/v0", "arm_id": "U"}), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED and one.reason == "UNSUPPORTED_SCHEMA"
@pytest.mark.parametrize("forged", [True, "NORMAL_SUCCESS", {"kind": "NORMAL_SUCCESS"}])
def test_forged_capability_unverified(forged, trust_caps):
    g = _geom_241()
    one = _one(_envelope(_exact_summary(arm_id="U", start_step=1, steps=4)), g.arms["U"], forged, g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED
    assert one.reason == "MISSING_OR_UNTRUSTED_NORMAL_SUCCESS_CAPABILITY"
def test_eligible_corrupt_verified_invalid(trust_caps):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary["missing_count"] = 1
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID
def test_promoted_source_flags_verified_invalid(trust_caps):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary["claimable"] = True
    summary["bankable"] = True
    summary["forensic_only"] = False
    summary["runtime_proven"] = True
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID
    assert one.state is not AccountingV2State.VERIFIED_VALID
    assert "authority_flag_claimable" in one.details["mismatches"]
def test_any_arm_invalid_propagates(trust_caps):
    g = _geom_241()
    payloads = _exact_payloads(g)
    bad = dict(_exact_summary(arm_id="E", start_step=3, steps=2))
    bad["duplicate_count"] = 1
    payloads["E"] = _envelope(bad)
    result = _four(g, payloads, {arm: trust_caps(arm) for arm in g.arms})
    assert result.state is AccountingV2State.VERIFIED_INVALID
def test_any_arm_unverified_propagates(trust_caps):
    g = _geom_241()
    payloads = _exact_payloads(g)
    payloads["R0"] = _exact_summary(arm_id="R0", start_step=3, steps=2)
    result = _four(g, payloads, {arm: trust_caps(arm) for arm in g.arms})
    assert result.state is AccountingV2State.UNVERIFIED
def test_rw_window_mismatch_invalid(trust_caps):
    g = _geom_241()
    assert _one(
        _envelope(_exact_summary(arm_id="RW", start_step=3, steps=1)),
        g.arms["RW"], trust_caps("RW"), g.t_cut, g.rewarm_window_steps,
    ).state is AccountingV2State.VERIFIED_INVALID
def test_expected_never_copied_from_observed(trust_caps):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary["start_step"] = 99
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID and g.arms["U"].start_step == 1
def test_regression_public_fixture_mint_absent_from_production_module():
    assert not hasattr(acct, "fixture_only_mint_trusted_normal_success_capability")
    assert not hasattr(acct, "_TRUSTED_NORMAL_SUCCESS_BY_ARM")
    assert "_TRUSTED_NORMAL_SUCCESS_BY_ARM" not in acct.__all__
    with pytest.raises(TypeError):
        TrustedNormalSuccessCapability()  # type: ignore[misc]
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "mint" in node.name.lower():
            raise AssertionError(f"production mint function: {node.name}")
        if isinstance(node, ast.Attribute) and node.attr == "NORMAL_SUCCESS":
            raise AssertionError("NORMAL_SUCCESS attribute")
    for name in acct.__all__:
        assert "mint" not in name.lower()
def test_regression_bare_summary_without_attachment_key_never_valid(trust_caps):
    g = _geom_241()
    one = _one(_exact_summary(arm_id="U", start_step=1, steps=4), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED
    assert one.reason == "BARE_OR_ABSENT_ATTACHMENT_KEY"
def test_regression_forged_equal_hashes_never_valid(trust_caps):
    g = _geom_241()
    forged = {}
    for arm, ag in g.arms.items():
        s = dict(_exact_summary(arm_id=ag.arm_id, start_step=ag.start_step, steps=ag.steps))
        s["expected_identity_projection_sha256"] = s["observed_identity_projection_sha256"] = "forged"
        forged[arm] = _envelope(s)
    result = _four(g, forged, {arm: trust_caps(arm) for arm in g.arms})
    assert result.state is AccountingV2State.VERIFIED_INVALID
    assert independent_expected_identity_sha256(g.arms["U"]) != "forged"
def test_regression_absent_source_provenance_unverified(trust_caps):
    g = _geom_241()
    summary = _exact_summary(arm_id="U", start_step=1, steps=4)
    one = _one({ATTACHMENT_KEY: summary}, g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED
    two = _one({ATTACHMENT_KEY: summary, "source_provenance": {"producer": "forged"}}, g.arms["U"], trust_caps("U"), g.t_cut)
    assert two.state is AccountingV2State.UNVERIFIED
def test_capability_arm_mismatch_unverified(trust_caps):
    g = _geom_241()
    one = _one(_envelope(_exact_summary(arm_id="U", start_step=1, steps=4)), g.arms["U"], trust_caps("E"), g.t_cut)
    assert one.state is AccountingV2State.UNVERIFIED
@pytest.mark.parametrize("field,bad,needle", [
    ("start_step", "1", "malformed_start_step"),
    ("steps", "4", "malformed_steps"),
    ("missing_count", "0", "malformed_missing_count"),
    ("duplicate_count", 1.5, "malformed_duplicate_count"),
    ("extra_count", True, "malformed_extra_count"),
    ("wrong_arm_count", object(), "malformed_wrong_arm_count"),
    ("observed_count", object(), "malformed_observed_count"),
    ("expected_identity_projection_sha256", 123, "expected_hash_not_independent"),
    ("observed_identity_projection_sha256", None, "observed_hash_not_independent"),
    ("claimable", "false", "authority_flag_claimable"),
    ("forensic_only", 1, "authority_flag_forensic_only"),
])
def test_malformed_eligible_fields_verified_invalid_never_raise(trust_caps, field, bad, needle):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary[field] = bad
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID
    assert one.state is not AccountingV2State.VERIFIED_VALID
    assert needle in one.details["mismatches"]
def test_four_arm_non_mapping_inputs_unverified(trust_caps):
    g = _geom_241()
    caps = {arm: trust_caps(arm) for arm in g.arms}
    assert _four(g, ["not", "a", "mapping"], caps).state is AccountingV2State.UNVERIFIED
    assert _four(g, _exact_payloads(g), ["bad"]).state is AccountingV2State.UNVERIFIED
def test_wrong_expected_count_never_valid(trust_caps):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary["expected_count"] = 999
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID
    assert "expected_count" in one.details["mismatches"]
def test_malformed_full_payload_sha256_never_valid(trust_caps):
    g = _geom_241()
    summary = dict(_exact_summary(arm_id="U", start_step=1, steps=4))
    summary["full_payload_sha256"] = "not-a-64-hex"
    one = _one(_envelope(summary), g.arms["U"], trust_caps("U"), g.t_cut)
    assert one.state is AccountingV2State.VERIFIED_INVALID
    assert "malformed_full_payload_sha256" in one.details["mismatches"]
def test_extra_arm_key_never_valid(trust_caps):
    g = _geom_241()
    payloads = _exact_payloads(g)
    payloads["X"] = payloads["U"]
    caps = {arm: trust_caps(arm) for arm in g.arms}
    caps["X"] = trust_caps("U")
    result = _four(g, payloads, caps)
    assert result.state is AccountingV2State.UNVERIFIED
    assert result.reason == "ARM_KEYSET_NOT_EXACT"
    assert result.state is not AccountingV2State.VERIFIED_VALID
def test_missing_arm_key_never_valid(trust_caps):
    g = _geom_241()
    payloads = _exact_payloads(g)
    del payloads["RW"]
    result = _four(g, payloads, {arm: trust_caps(arm) for arm in ("U", "E", "R0")})
    assert result.state is AccountingV2State.UNVERIFIED
    assert result.reason == "ARM_KEYSET_NOT_EXACT"
@pytest.mark.parametrize("kwargs", [
    {"t_cut": "2", "runway_steps": 4, "rewarm_window_steps": 1},
    {"t_cut": 2, "runway_steps": True, "rewarm_window_steps": 1},
    {"t_cut": 2, "runway_steps": 4, "rewarm_window_steps": 1.0},
])
def test_malformed_geometry_scalars_unverified_never_raise(kwargs):
    refuse = refuse_unadmitted_characterization_geometry(**kwargs)
    assert refuse.state is AccountingV2State.UNVERIFIED
    assert refuse.reason == "GEOMETRY_INPUT_MALFORMED"
    built = build_independent_expected_geometry(**kwargs)
    assert isinstance(built, acct.AccountingV2Result)
    assert built.reason == "GEOMETRY_INPUT_MALFORMED"
def test_caller_geometry_extra_arm_unverified(trust_caps):
    g = _geom_241()
    bad_arms = dict(g.arms)
    bad_arms["X"] = IndependentArmGeometry("X", 3, 2, 2, 2)
    bad = IndependentExpectedGeometry(
        g.t_cut, g.runway_steps, g.rewarm_window_steps, g.shared_prefix_once,
        g.physical_total, g.rw_rewarm_window, bad_arms,
    )
    _assert_geom_not_indep(_four(bad, _exact_payloads(g), {arm: trust_caps(arm) for arm in g.arms}))
def test_forged_noncanonical_u_geometry_never_valid(trust_caps):
    """Admitted outer triple with forged U internals must not reach VERIFIED_VALID."""
    g = _geom_241()
    arms = dict(g.arms)
    arms["U"] = IndependentArmGeometry("U", 99, 4, 4, 4)
    forged = IndependentExpectedGeometry(
        g.t_cut, g.runway_steps, g.rewarm_window_steps, g.shared_prefix_once,
        12, g.rw_rewarm_window, arms,
    )
    payloads = {
        "U": _envelope(_exact_summary(arm_id="U", start_step=99, steps=4)),
        "E": _envelope(_exact_summary(arm_id="E", start_step=3, steps=2)),
        "R0": _envelope(_exact_summary(arm_id="R0", start_step=3, steps=2)),
        "RW": _envelope(_exact_summary(arm_id="RW", start_step=3, steps=2)),
    }
    _assert_geom_not_indep(_four(forged, payloads, {arm: trust_caps(arm) for arm in g.arms}))
@pytest.mark.parametrize("mutator", [
    lambda g, arms: arms.__setitem__("E", IndependentArmGeometry("E", 9, 2, 2, 2)),
    lambda g, arms: arms.__setitem__("R0", IndependentArmGeometry("R0", 3, 9, 9, 9)),
    lambda g, arms: arms.__setitem__("RW", IndependentArmGeometry("RW", 3, 2, 2, 9)),
])
def test_forged_fork_arm_geometry_never_valid(trust_caps, mutator):
    g = _geom_241()
    arms = dict(g.arms)
    mutator(g, arms)
    forged = IndependentExpectedGeometry(
        g.t_cut, g.runway_steps, g.rewarm_window_steps, g.shared_prefix_once,
        g.physical_total, g.rw_rewarm_window, arms,
    )
    _assert_geom_not_indep(_four(forged, _exact_payloads(g), {arm: trust_caps(arm) for arm in g.arms}))
@pytest.mark.parametrize("field,value", [
    ("shared_prefix_once", 9), ("physical_total", 99), ("rw_rewarm_window", 9),
])
def test_forged_derived_outer_fields_never_valid(trust_caps, field, value):
    g = _geom_241()
    kwargs = {
        "t_cut": g.t_cut, "runway_steps": g.runway_steps, "rewarm_window_steps": g.rewarm_window_steps,
        "shared_prefix_once": g.shared_prefix_once, "physical_total": g.physical_total,
        "rw_rewarm_window": g.rw_rewarm_window, "arms": g.arms,
    }
    kwargs[field] = value
    _assert_geom_not_indep(_four(
        IndependentExpectedGeometry(**kwargs), _exact_payloads(g),
        {arm: trust_caps(arm) for arm in g.arms},
    ))
@pytest.mark.parametrize("expected", [None, {"arm_id": "U"}, "U"])
def test_malformed_one_arm_expected_never_raises(expected, trust_caps):
    one = _one(_envelope(_exact_summary(arm_id="U", start_step=1, steps=4)), expected, trust_caps("U"), 2)
    assert one.state is AccountingV2State.UNVERIFIED
    assert one.reason == "EXPECTED_ARM_GEOMETRY_MALFORMED"
def test_malformed_one_arm_t_cut_never_raises(trust_caps):
    g = _geom_241()
    one = _one(_envelope(_exact_summary(arm_id="U", start_step=1, steps=4)), g.arms["U"], trust_caps("U"), "2")
    assert one.state is AccountingV2State.UNVERIFIED
    assert one.reason == "GEOMETRY_INPUT_MALFORMED"
def test_no_semicolon_packed_or_one_line_if_append_in_module():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for line in src.splitlines():
        assert ";" not in line
        stripped = line.lstrip()
        if stripped.startswith("if ") and "m.append" in stripped:
            raise AssertionError(f"one-line if/append compression: {line}")
def test_module_and_test_loc_budget():
    assert len(MODULE_PATH.read_text(encoding="utf-8").splitlines()) <= 250
    assert len(Path(__file__).read_text(encoding="utf-8").splitlines()) <= 400
def test_sha_helper_stable_for_fixture_identity():
    a = _exact_summary(arm_id="U", start_step=1, steps=4)
    b = _exact_summary(arm_id="U", start_step=1, steps=4)
    assert a["expected_identity_projection_sha256"] == b["expected_identity_projection_sha256"]
    assert a["expected_identity_projection_sha256"] == independent_expected_identity_sha256(_geom_241().arms["U"])
    assert len(hashlib.sha256((json.dumps(a, sort_keys=True) + "\n").encode()).hexdigest()) == 64
def test_required_provenance_is_immutable_mapping():
    assert isinstance(REQUIRED_SOURCE_PROVENANCE, MappingProxyType)
