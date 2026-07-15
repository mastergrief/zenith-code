"""Phase-0 extraction characterization: pure-core vs facade authority seam."""
from __future__ import annotations
import ast
import hashlib
import importlib
import inspect
import pickle
import subprocess
from pathlib import Path
from typing import Any
from calm.hrm_text_158.native_full_stack import forgotten_accum_a_ledger_accounting_v2 as acct
from calm.hrm_text_158.native_full_stack import forgotten_accum_a_ledger_accounting_v2_core as core
from calm.hrm_text_158.native_full_stack.forgotten_accum_a_ledger_accounting_v2 import (
    AccountingV2Result, AccountingV2State, IndependentArmGeometry,
    IndependentExpectedGeometry, TrustedNormalSuccessCapability,
    build_independent_expected_geometry, classify_four_arm_ordered_event_summaries,
    extract_attachment_summary,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ATTACHMENT_KEY, ExpectedIdentity, make_success_apply_event,
    snapshot_ordered_apply_event_log, validate_ordered_apply_event_sequence,
)
REPO = Path(__file__).resolve().parents[3]
NS = REPO / "calm/hrm_text_158/native_full_stack"
CORE_PATH = NS / "forgotten_accum_a_ledger_accounting_v2_core.py"
FACADE_PATH = NS / "forgotten_accum_a_ledger_accounting_v2.py"
STEP_A_TEST = REPO / "calm/llm_computer/tests/test_forgotten_accum_a_ledger_accounting_v2_ordered_event_consumer_v0.py"
ARK_PATH = NS / "forgotten_accum_training_equivalence_ark_invoke.py"
SCIENCE_DRIVER_PATH = NS / "forgotten_accum_training_equivalence_science_driver.py"
CORE_SHA = "dceba582104f5bde6cb770802d0bcd80f7d416db1e9e856e5b773725cd5dfc5b"
STEP_A_SHA = "4ca9a2f068bd9c697584dc20b29603d02ec0b1d450b1465426e84a65a18362de"
ARK_SHA = "3f6c71c56c79af3a6724ae57e2ba70e6e439c12d51a530f5b545a520a6d63e20"
SCIENCE_DRIVER_SHA = "b9ef6496532a60c882052751aff11608fc41c7f7f90bd66f82621d3836a92609"
LANDED = "ca4a4e28a580f09565b9566cc00a8274eb90cc5b"
FACADE_MOD = "calm.hrm_text_158.native_full_stack.forgotten_accum_a_ledger_accounting_v2"
FORBIDDEN_CORE = {
    "TrustedNormalSuccessCapability", "_capability_ok",
    "classify_arm_ordered_event_summary", "classify_four_arm_ordered_event_summaries",
    "AccountingV2State", "AccountingV2Result",
}
PUBLIC_FUNCS = [
    "build_independent_expected_geometry", "classify_arm_ordered_event_summary",
    "classify_four_arm_ordered_event_summaries", "extract_attachment_summary",
    "independent_expected_identity_sha256", "refuse_unadmitted_characterization_geometry",
]
PUBLIC_TYPES = [
    "AccountingV2Result", "AccountingV2State", "IndependentArmGeometry",
    "IndependentExpectedGeometry", "TrustedNormalSuccessCapability",
]
LANDED_SIGS = {
    "extract_attachment_summary": "(payload: 'Any') -> 'Mapping[str, Any] | None'",
    "independent_expected_identity_sha256": "(expected: 'IndependentArmGeometry') -> 'str'",
    "refuse_unadmitted_characterization_geometry": (
        "(*, t_cut: 'Any', runway_steps: 'Any', rewarm_window_steps: 'Any') "
        "-> 'AccountingV2Result'"
    ),
    "build_independent_expected_geometry": (
        "(*, t_cut: 'Any', runway_steps: 'Any', rewarm_window_steps: 'Any') -> "
        "'IndependentExpectedGeometry | AccountingV2Result'"
    ),
    "classify_arm_ordered_event_summary": (
        "(*, summary_payload: 'Any', expected: 'Any', trusted_normal_success: 'Any', "
        "t_cut: 'Any', rewarm_window_steps: 'Any' = None) -> 'AccountingV2Result'"
    ),
    "classify_four_arm_ordered_event_summaries": (
        "(*, geometry: 'IndependentExpectedGeometry', "
        "arm_summary_payloads: 'Mapping[str, Any]', "
        "trusted_capabilities: 'Mapping[str, Any]') -> 'AccountingV2Result'"
    ),
}
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _exact_summary(*, arm_id: str, start_step: int, steps: int) -> dict:
    events = [
        make_success_apply_event(
            seq=i, arm_id=arm_id, optimizer_step_id=start_step + i,
            q_changed_count=1, tensor_state_key_count=1,
        )
        for i in range(steps)
    ]
    return validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(events),
        ExpectedIdentity(arm_id=arm_id, start_step=start_step, steps=steps),
    )
def _envelope(summary: dict) -> dict:
    return {ATTACHMENT_KEY: summary, "source_provenance": dict(acct.REQUIRED_SOURCE_PROVENANCE)}
def _landed_facade_src() -> str:
    rel = "calm/hrm_text_158/native_full_stack/forgotten_accum_a_ledger_accounting_v2.py"
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{LANDED}:{rel}"], text=True,
    )
def _literal_exception_lines(tree: ast.AST, lines: list[str]) -> set[int]:
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    if lines[ln - 1].strip().startswith(("'", '"')):
                        allowed.add(ln)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    text = lines[ln - 1].strip().rstrip("),")
                    if text.startswith(("'", '"')):
                        allowed.add(ln)
    return allowed
def test_loc_budgets_and_style():
    core_src, facade_src = CORE_PATH.read_text(), FACADE_PATH.read_text()
    assert len(core_src.splitlines()) <= 220
    assert len(facade_src.splitlines()) <= 180
    assert ";" not in core_src and ";" not in facade_src
    tree = ast.parse(facade_src)
    lines = facade_src.splitlines()
    allowed = _literal_exception_lines(tree, lines)
    over = [(i + 1, len(l)) for i, l in enumerate(lines) if len(l) > 119 and i + 1 not in allowed]
    assert over == []
    public = set(acct.__all__)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in public:
                val = node.value
                if isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
                    assert val.value.id != "_core"
                if isinstance(val, ast.Name):
                    assert val.id != "_core"
def test_extract_attachment_summary_is_facade_functiondef():
    defs = [
        n for n in ast.parse(FACADE_PATH.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "extract_attachment_summary"
    ]
    assert len(defs) == 1
    assert extract_attachment_summary.__module__ == FACADE_MOD
    assert extract_attachment_summary.__name__ == "extract_attachment_summary"
    assert extract_attachment_summary is not core.extract_attachment_summary
    assert list(inspect.signature(extract_attachment_summary).parameters) == ["payload"]
    assert extract_attachment_summary.__annotations__["payload"] in {Any, "Any"}
def test_public_api_matches_landed_baseline():
    landed_all = None
    for node in ast.parse(_landed_facade_src()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    landed_all = ast.literal_eval(node.value)
    assert set(acct.__all__) == set(landed_all)
    for name in PUBLIC_FUNCS:
        live = getattr(acct, name)
        assert live.__module__ == FACADE_MOD and live.__name__ == name
        assert str(inspect.signature(live)) == LANDED_SIGS[name]
    for name in PUBLIC_TYPES:
        obj = getattr(acct, name)
        assert obj.__module__ == FACADE_MOD and obj.__name__ == name
    assert IndependentArmGeometry._fields == (
        "arm_id", "start_step", "steps", "expected_local_invocation", "expected_post_cut",
    )
    assert IndependentExpectedGeometry._fields == (
        "t_cut", "runway_steps", "rewarm_window_steps", "shared_prefix_once",
        "physical_total", "rw_rewarm_window", "arms",
    )
def test_public_types_facade_module_identity():
    for cls in (
        AccountingV2State, AccountingV2Result, IndependentArmGeometry,
        IndependentExpectedGeometry, TrustedNormalSuccessCapability,
    ):
        assert cls.__module__.endswith("forgotten_accum_a_ledger_accounting_v2")
    for banned in (
        "TrustedNormalSuccessCapability", "_capability_ok",
        "AccountingV2State", "AccountingV2Result",
    ):
        assert not hasattr(core, banned)
def test_core_byte_stable_and_zero_touch_pins():
    assert _sha(CORE_PATH) == CORE_SHA
    assert _sha(STEP_A_TEST) == STEP_A_SHA
    assert _sha(ARK_PATH) == ARK_SHA
    assert _sha(SCIENCE_DRIVER_PATH) == SCIENCE_DRIVER_SHA
def test_step_a_test_byte_identical_pin():
    assert _sha(STEP_A_TEST) == STEP_A_SHA
def test_monkeypatch_facade_capability_ok_still_drives_valid(monkeypatch):
    registry: dict[str, object] = {}
    monkeypatch.setattr(
        acct, "_capability_ok",
        lambda cap, *, arm_id: registry.get(str(arm_id)) is cap,
    )
    geom = build_independent_expected_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=1)
    assert isinstance(geom, IndependentExpectedGeometry)
    payloads, caps = {}, {}
    for arm, expected in geom.arms.items():
        summary = _exact_summary(
            arm_id=expected.arm_id, start_step=expected.start_step, steps=expected.steps,
        )
        payloads[arm] = _envelope(summary)
        cap = object.__new__(TrustedNormalSuccessCapability)
        registry[arm] = cap
        caps[arm] = cap
    result = classify_four_arm_ordered_event_summaries(
        geometry=geom, arm_summary_payloads=payloads, trusted_capabilities=caps,
    )
    assert result.state is AccountingV2State.VERIFIED_VALID
    assert result.reason == "FOUR_ARM_CONJUNCTION_OK"
def test_direct_core_cannot_finalize_valid():
    src = CORE_PATH.read_text()
    tree = ast.parse(src)
    names = {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
    }
    assigned = {
        t.id for n in tree.body if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }
    for banned in FORBIDDEN_CORE:
        assert banned not in names and banned not in assigned
        assert not hasattr(core, banned)
    assert "VERIFIED_VALID" not in src and "FOUR_ARM_CONJUNCTION_OK" not in src
    geom = build_independent_expected_geometry(t_cut=2, runway_steps=4, rewarm_window_steps=1)
    assert isinstance(geom, IndependentExpectedGeometry)
    expected = geom.arms["U"]
    summary = _exact_summary(
        arm_id=expected.arm_id, start_step=expected.start_step, steps=expected.steps,
    )
    facts = core.evaluate_arm_summary_facts(
        summary_payload=_envelope(summary),
        arm_id=expected.arm_id, start_step=expected.start_step, steps=expected.steps,
        expected_local_invocation=expected.expected_local_invocation,
        expected_post_cut=expected.expected_post_cut, t_cut=geom.t_cut,
        rewarm_window_steps=geom.rewarm_window_steps,
        expected_schema_id=acct.REQUIRED_SOURCE_PROVENANCE["validation_schema_id"],
        required_provenance=acct.REQUIRED_SOURCE_PROVENANCE,
    )
    assert facts["status"] == "eligible"
    assert not isinstance(facts, AccountingV2Result)
def test_core_never_imports_facade_or_adapter():
    src = CORE_PATH.read_text()
    assert "forgotten_accum_a_ledger_accounting_v2_ark_adapter" not in src
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "forgotten_accum_a_ledger_accounting_v2"
            assert not node.module.endswith("forgotten_accum_a_ledger_accounting_v2")
def test_caller_census_core_only_facade_and_this_test():
    hits = []
    needle = "forgotten_accum_a_ledger_accounting_v2_core"
    for path in (REPO / "calm").rglob("*.py"):
        if path.resolve() in {CORE_PATH.resolve(), Path(__file__).resolve()}:
            continue
        if needle in path.read_text(encoding="utf-8"):
            hits.append(str(path.relative_to(REPO)))
    assert hits == [
        "calm/hrm_text_158/native_full_stack/forgotten_accum_a_ledger_accounting_v2.py"
    ]
def test_golden_refuse_and_as_dict():
    refused = acct.refuse_unadmitted_characterization_geometry(
        t_cut=2, runway_steps=4, rewarm_window_steps=2,
    )
    assert refused.state is AccountingV2State.UNVERIFIED
    assert refused.reason == "UNADMITTED_CHARACTERIZATION_GEOMETRY"
    payload = refused.as_dict()
    assert payload["claimable"] is False and payload["bankable"] is False
    assert payload["forensic_only"] is True and payload["runtime_proven"] is False
def test_pickle_roundtrip_public_types():
    arm = IndependentArmGeometry("U", 1, 4, 4, 2)
    restored = pickle.loads(pickle.dumps(arm))
    assert restored == arm and type(restored) is IndependentArmGeometry
    refused = acct.refuse_unadmitted_characterization_geometry(
        t_cut=2, runway_steps=4, rewarm_window_steps=2,
    )
    assert pickle.loads(pickle.dumps(refused)) == refused
def test_import_reload_facade():
    reloaded = importlib.reload(acct)
    assert reloaded._capability_ok(object(), arm_id="U") is False
    assert reloaded.extract_attachment_summary.__module__.endswith(
        "forgotten_accum_a_ledger_accounting_v2"
    )
