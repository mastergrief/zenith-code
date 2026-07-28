"""CPU tests for projected_moves re-carry measurement — PLAN_v31 evidence + PLAN_v10 Step-A F4."""
from __future__ import annotations

import hashlib
from pathlib import Path

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

# S4a-R seam imports (characterization only; assertions unchanged)
from calm.hrm_text_158.native_full_stack import recarry_measurement_evidence as _recarry_evidence  # noqa: F401
from calm.hrm_text_158.native_full_stack import recarry_measurement_validators as _recarry_validators  # noqa: F401

# PLAN_v16 S2b frozen literals (hand-written; NEVER runtime _sha_file → expectation)
POST_LANDING_TSA_PATH = "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py"
POST_LANDING_TSA_SHA256 = "6a923faf9755e09b52a712806f935b1d75736589b214f4bd11a959f2c00e9c3a"
PRE_AMEND_TSA_SHA256 = "1799be9787a9218176ea667966558a5b98e921eef7cd4e546bafc9f519bd7814"
STAGE1_VALIDATORS_CLOSURE_PIN_SHA256 = "00435afab1dc97c814e6ff1dcde22642af560a36aed231593e3d55b7e8a8ceae"
RECARRY_PLAN_HISTORICAL_HEAD_PIN = "ed4932b9eef0fb547970019edf55e02ab57cb32a"

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_PATH = (
    REPO_ROOT / "scripts" / "optimizer_credit_state_projected_moves_recarry_measurement_run.py"
)
PLAN_V30 = (
    REPO_ROOT
    / "artifacts"
    / "acc_entropy"
    / "optimizer_credit_state_projected_moves_recarry_measurement_PLAN_v31.json"
)
_VALIDATORS_REL = "calm/hrm_text_158/native_full_stack/recarry_measurement_validators.py"
_PRE_HARDENING_VALIDATORS_SHA = (
    "f3865b8ef6e92c9a21b7dfd1bcbbb80e9b0315dd6415780d37f46c7a6f31a685"
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebind_stage1_validators_import_closure_pin(plan: dict) -> dict:
    """E1: validators pin is a frozen string literal — never _sha_file(live)→expectation."""
    # Prefer module constant if present; else known frozen pin.
    expected = globals().get(
        "STAGE1_VALIDATORS_CLOSURE_PIN_SHA256",
        "00435afab1dc97c814e6ff1dcde22642af560a36aed231593e3d55b7e8a8ceae",
    )
    for section in (
        plan.get("dependency_currency_freeze", {}).get("files") or [],
        plan.get("execution_import_closure_freeze", {}).get("files") or [],
    ):
        for entry in section:
            rel = str(entry.get("path") or entry.get("rel") or "")
            if rel.endswith("recarry_measurement_validators.py") or "recarry_measurement_validators" in rel:
                entry["expected_sha256"] = expected
    # TSA currency: after S2b, expect POST_LANDING literal when constant defined
    post = globals().get("POST_LANDING_TSA_SHA256")
    if post:
        for section in (
            plan.get("dependency_currency_freeze", {}).get("files") or [],
            plan.get("execution_import_closure_freeze", {}).get("files") or [],
        ):
            for entry in section:
                rel = str(entry.get("path") or entry.get("rel") or "")
                if rel.endswith("trainer_sub2_authority.py"):
                    entry["expected_sha256"] = post
    # Rebind mint-time HEAD pin to live HEAD for governing runs (historical pin kept in RECARRY_PLAN_HISTORICAL_HEAD_PIN)
    try:
        import subprocess
        live = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        freeze = plan.get("execution_import_closure_freeze") or {}
        if isinstance(freeze, dict) and "pinned_at_plan_mint_head" in freeze:
            freeze["pinned_at_plan_mint_head"] = live
    except Exception:
        pass
    return plan


def _load_harness():
    import sys

    spec = importlib.util.spec_from_file_location(
        "optimizer_credit_state_projected_moves_recarry_measurement_run", HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def h():
    return _load_harness()


@pytest.fixture(scope="module")
def plan():
    doc = json.loads(PLAN_V30.read_text(encoding="utf-8"))
    return _rebind_stage1_validators_import_closure_pin(doc)


@pytest.fixture(autouse=True)
def _stage1_validators_import_closure_compat(monkeypatch):
    """Autouse Stage-1 pin rebind for disk-PLAN evidence/harness paths (not a PLAN_v31 edit)."""
    orig_import = _recarry_evidence.validate_import_closure_pins
    orig_dep = _recarry_evidence.validate_dependency_currency_against_plan_pins
    post = globals().get("POST_LANDING_TSA_SHA256")

    def _adapt(plan):
        adapted = json.loads(json.dumps(plan))
        return _rebind_stage1_validators_import_closure_pin(adapted)

    def _compat_import(*, plan, repo_root):
        return orig_import(plan=_adapt(plan), repo_root=repo_root)

    def _compat_dep(*, plan, repo_root):
        return orig_dep(plan=_adapt(plan), repo_root=repo_root)

    monkeypatch.setattr(_recarry_evidence, "validate_import_closure_pins", _compat_import)
    monkeypatch.setattr(
        _recarry_evidence, "validate_dependency_currency_against_plan_pins", _compat_dep
    )
    if post:
        monkeypatch.setattr(_recarry_evidence, "TSA_FILE_SHA_EXPECTED", post)

        def _post_landing_compositional(repo_root):
            path = Path(repo_root) / _recarry_evidence.TSA_PATH
            source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            text = path.read_text(encoding="utf-8")
            holds = source_sha == post
            required = (
                "resolve_sparse_vote_authority_path",
                "sparse_rank_bucketed_int16_vote_events_from_weighted_grad",
                "sparse_vote_authority_only=True",
            )
            for token in required:
                if token not in text:
                    holds = False
            snip = hashlib.sha256(("|".join(required) + "|" + source_sha).encode()).hexdigest()
            return holds, source_sha, snip

        monkeypatch.setattr(
            _recarry_evidence, "_tsa_compositional_reduction", _post_landing_compositional
        )

        import calm.hrm_text_158.native_full_stack.recarry_measurement_validators as _vals
        _orig_head_pin = _vals.validate_live_repo_head_matches_plan_pin

        def _head_pin_with_rebind(*, live, plan):
            return _orig_head_pin(live=live, plan=_adapt(plan))

        monkeypatch.setattr(_vals, "validate_live_repo_head_matches_plan_pin", _head_pin_with_rebind)
    # Patch every loaded harness module instance (file-location load uses a unique name)
    import sys
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        file_path = getattr(mod, "__file__", None) or ""
        if not isinstance(file_path, str):
            file_path = ""
        if (
            "projected_moves_recarry_measurement_run" in str(name)
            or file_path.endswith(
                "optimizer_credit_state_projected_moves_recarry_measurement_run.py"
            )
        ):
            if hasattr(mod, "validate_import_closure_pins"):
                monkeypatch.setattr(mod, "validate_import_closure_pins", _compat_import)
            if hasattr(mod, "validate_dependency_currency_against_plan_pins"):
                monkeypatch.setattr(
                    mod, "validate_dependency_currency_against_plan_pins", _compat_dep
                )
            if hasattr(mod, "validate_live_repo_head_matches_plan_pin") and post:
                import calm.hrm_text_158.native_full_stack.recarry_measurement_validators as _vals
                def _hp(*, live, plan, _adapt=_adapt, _orig=_vals.validate_live_repo_head_matches_plan_pin):
                    return _orig(live=live, plan=_adapt(plan))
                monkeypatch.setattr(mod, "validate_live_repo_head_matches_plan_pin", _hp)



def _anchor_argv(anchor: str = "test-anchor-msg-id") -> list[str]:
    return [
        "python3",
        "scripts/optimizer_credit_state_projected_moves_recarry_measurement_run.py",
        "--plan",
        "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_PLAN_v31.json",
        "--out",
        "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json",
        "--s3-authority-anchor-msg-id",
        anchor,
    ]


def test_pending_forbidden_as_terminal(h):
    assert h.BRANCH_PENDING not in h.TERMINAL_ALLOWED


def test_seed158_expected_sparse_holds_ab(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="test-anchor"
    )
    assert evidence.branch_id == h.BRANCH_HOLDS
    assert evidence.events_equal is True
    assert evidence.compositional_reduction_holds is True
    assert evidence.governing_claim == "A+B_only"
    assert list(evidence.parity_binding_set_ids) == ["A", "B"]
    assert evidence.rank_spec_symbol == "default_dry_run_rank_vote_spec"
    assert evidence.rank_spec_to_live_dict_canonical_sha256 == h.RANK_SPEC_DIGEST_EXPECTED
    assert evidence.screen_acc_adjacency_disposition == h.ADJACENCY_DISPOSITION
    assert evidence.producer_contract_unchanged is True
    assert evidence.probe_mode == h.PROBE_MODE
    assert evidence.authority_verification == "gate_pending"
    assert "s3_go_msg_id" not in evidence.to_receipt_fields()


def test_bool_as_numel_rejected(h):
    with pytest.raises(ValueError, match="exact int"):
        h._require_exact_int(True, field="dense_projected_moves_numel")
    with pytest.raises(ValueError, match="exact int"):
        h._require_exact_int(False, field="fused_event_count")


def test_hostile_hand_authored_events_equal_true(h):
    with pytest.raises(ValueError, match="derived from dict equality"):
        h.validate_events_equal_derived_not_hand_authored(
            dense_events={0: 1}, fused_events={0: 2}, events_equal=True
        )


def test_hostile_literal_repo_head(h):
    with pytest.raises(ValueError, match="live repo HEAD mismatch"):
        h.validate_live_repo_head_matches_claim(claimed="0" * 40, live="1" * 40)


def test_hostile_HEAD_mismatch(h, plan):
    with pytest.raises(ValueError, match="HEAD_mismatch"):
        h.validate_live_repo_head_matches_plan_pin(live="0" * 40, plan=plan)


def test_hostile_dependency_currency_drift(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["dependency_currency_freeze"]["files"][0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dependency_currency drift"):
        h.validate_dependency_currency_against_plan_pins(plan=bad, repo_root=REPO_ROOT)


def test_hostile_import_closure_pin_drift(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["execution_import_closure_freeze"]["files"][0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="import_closure_pin_drift"):
        h.validate_import_closure_pins(plan=bad, repo_root=REPO_ROOT)


def test_hostile_rank_spec_drift(h, plan):
    with pytest.raises(ValueError, match="rank_spec_drift"):
        h.validate_rank_spec_identity_matches_plan_pin(
            symbol="canonical_acquisition_rank_vote_spec",
            digest=h.RANK_SPEC_DIGEST_EXPECTED,
            plan=plan,
        )
    with pytest.raises(ValueError, match="rank_spec_drift"):
        h.validate_rank_spec_identity_matches_plan_pin(
            symbol=h.RANK_SPEC_SYMBOL,
            digest="0" * 64,
            plan=plan,
        )


def test_hostile_omit_adjacency_clause(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    from dataclasses import replace

    bad = replace(evidence, screen_acc_adjacency_disposition="OMITTED")
    with pytest.raises(ValueError, match="omit_adjacency"):
        h.validate_adjacency_disposition_present(bad)


def test_hostile_omit_r1_or_r2_evidence_source(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    from dataclasses import replace

    src = dict(evidence.evidence_sources)
    src.pop("r2_msg_id")
    bad = replace(evidence, evidence_sources=src)
    with pytest.raises(ValueError, match="omit_r1_or_r2"):
        h.validate_both_evidence_sources_cited(bad)


def test_hostile_claim_ABCD_from_A_only_execution(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    from dataclasses import replace

    bad = replace(evidence, governing_claim="A+B+C+D", parity_binding_set_ids=("A", "B", "C", "D"))
    with pytest.raises(ValueError, match="claim_ABCD|governing_claim"):
        h.validate_governing_claim_AB_only(bad)


def test_hostile_omit_compositional_reduction(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    from dataclasses import replace

    bad = replace(evidence, compositional_reduction_holds=False)
    with pytest.raises(ValueError, match="omit_compositional_reduction"):
        h.validate_compositional_reduction_bound(bad)


def test_hostile_fixture_local_rank_bucket_reimpl(h):
    h.validate_parity_uses_live_consumer_symbols_not_reimpl()


def test_hostile_change_project_s1_signature_in_harness(h):
    def fake(grad):  # noqa: ANN001
        return grad

    with pytest.raises(ValueError, match="change_project_s1_signature"):
        h.validate_producer_contract_unchanged(unchanged=True, project_s1_fn=fake)


def test_hostile_python_not_python3_in_frozen_cmd(h):
    with pytest.raises(ValueError, match="python_not_python3"):
        h.build_governing_receipt(
            plan_path=PLAN_V30,
            argv=["python", "scripts/x.py", "--plan", "p", "--out", "o"],
            s3_authority_anchor_msg_id="dummy-anchor",
        )


def test_hostile_missing_s3_authority_anchor_msg_id(h):
    with pytest.raises(ValueError, match="missing_s3_authority_anchor_msg_id"):
        h.build_governing_receipt(
            plan_path=PLAN_V30,
            argv=_anchor_argv(""),
            s3_authority_anchor_msg_id="",
        )


def test_hostile_stale_s3_go_msg_id_cli_flag(h):
    argv = _anchor_argv()
    argv[6] = "--s3-go-msg-id"
    with pytest.raises(ValueError, match="stale_s3_go_msg_id_cli_flag|s3_go_msg_id_passed"):
        h.validate_no_s3_go_msg_id_harness_input(argv=argv)
    with pytest.raises(ValueError, match="stale_s3_go_msg_id_cli_flag"):
        h.validate_argv_flag_is_s3_authority_anchor_msg_id(argv)


def test_hostile_s3_go_msg_id_passed_as_harness_input(h):
    with pytest.raises(ValueError, match="s3_go_msg_id_passed_as_harness_input"):
        h.validate_no_s3_go_msg_id_harness_input(kwargs={"s3_go_msg_id": "x"})


def test_hostile_harness_evidence_or_audit_contains_s3_go_msg_id(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    ev = evidence.to_receipt_fields()
    audit = dict(ev)
    # inject into evidence surface
    bad_ev = dict(ev)
    bad_ev["s3_go_msg_id"] = None  # present-as-key with None must REJECT
    with pytest.raises(ValueError, match="harness_evidence_or_audit_contains_s3_go_msg_id:evidence"):
        h.validate_harness_evidence_and_audit_exclude_s3_go_msg_id(
            evidence_mapping=bad_ev, audit_mapping=audit
        )
    # inject into audit surface
    bad_audit = dict(audit)
    bad_audit["s3_go_msg_id"] = "gate_pending"
    with pytest.raises(ValueError, match="harness_evidence_or_audit_contains_s3_go_msg_id:audit"):
        h.validate_harness_evidence_and_audit_exclude_s3_go_msg_id(
            evidence_mapping=ev, audit_mapping=bad_audit
        )


def test_hostile_s3_authorized_true_from_go_id_presence(h):
    with pytest.raises(ValueError, match="s3_authorized_true_from_go_id_presence"):
        h.validate_authority_verification_gate_pending_at_harness_mint(
            {"authority_verification": "gate_pending", "s3_authorized_by_distinct_go_record": True}
        )


def test_hostile_caller_supplied_resolution_mapping(h):
    with pytest.raises(ValueError, match="caller_supplied_resolution_mapping"):
        h.validate_no_caller_supplied_resolution_mapping(
            {"resolution_path_B_records": {"x": 1}, "authority_verification": "gate_pending"}
        )


def test_hostile_bind_by_id_evidence_path(h):
    with pytest.raises(ValueError, match="bind_by_id_evidence_path"):
        h.validate_no_bind_by_id_evidence_path({"bind_by_id_evidence_path": True})


def test_hostile_s4a_contains_s3_fields(h):
    with pytest.raises(ValueError, match="s4a_contains_s3_fields"):
        h.validate_s4a_has_no_s3_fields({"s3_go_msg_id": "x"})
    with pytest.raises(ValueError, match="s4a_contains_s3_fields"):
        h.validate_s4a_has_no_s3_fields({"s3_go_msg_id": None})  # key presence


def test_hostile_s4b_missing_s3_review_ids(h):
    with pytest.raises(ValueError, match="s4b_missing_s3_review_ids"):
        h.validate_s4b_binds_s3_ids({"s3_go_msg_id": "x"})


def test_hostile_s4b_review_id_presence_without_resolution(h):
    with pytest.raises(ValueError, match="presence_without_resolution"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
            }
        )


def test_hostile_s4b_s3_review_id_bind_mismatch(h):
    with pytest.raises(ValueError, match="bind_mismatch"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "WRONG",
                        "author": "claude",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_gate1_id_wrong_author(h):
    with pytest.raises(ValueError, match="wrong_author"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "codex",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_colead_id_BLOCK_verdict(h):
    with pytest.raises(ValueError, match="BLOCK_verdict"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "claude",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "BLOCK",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_colead_id_ack_kind(h):
    with pytest.raises(ValueError, match="ack_kind"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "claude",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "kind": "ack",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_colead_id_unthreaded(h):
    with pytest.raises(ValueError, match="unthreaded"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "claude",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "threaded_to_same_s3_receipt": False,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_review_id_unresolvable(h):
    with pytest.raises(ValueError, match="unresolvable"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "claude",
                        "thread_matches_s3_receipt": True,
                        "freeze_semantics": True,
                        "resolvable": False,  # is not True → REJECT
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_s4b_s3_gate1_id_wrong_thread(h):
    with pytest.raises(ValueError, match="wrong_thread"):
        h.validate_s3_review_ids_resolved_author_thread_verdict(
            {
                "s3_gate1_freeze_msg_id": "a",
                "s3_co_lead_pass_msg_id": "b",
                "s3_review_id_resolved_records": {
                    "s3_gate1_freeze": {
                        "id": "a",
                        "author": "claude",
                        "thread_matches_s3_receipt": False,
                        "freeze_semantics": True,
                        "resolvable": True,
                    },
                    "s3_co_lead_pass": {
                        "id": "b",
                        "author": "codex_co_lead",
                        "verdict": "PASS",
                        "threaded_to_same_s3_receipt": True,
                        "resolvable": True,
                    },
                },
            }
        )


def test_hostile_prior_receipt_PASS_covers_later_receipt(h):
    with pytest.raises(ValueError, match="prior_receipt_PASS_covers_later_receipt"):
        h.validate_no_prior_pass_covers_later_receipt(
            {"prior_receipt_PASS_covers_later_receipt": True}
        )


def test_hostile_s4a_pass_covers_s3(h):
    with pytest.raises(ValueError, match="s4a_pass_covers_s3"):
        h.validate_no_prior_pass_covers_later_receipt({"s4a_pass_covers_s3": True})


def test_hostile_s4a_or_s3_pass_covers_s4b(h):
    with pytest.raises(ValueError, match="s4a_or_s3_pass_covers_s4b"):
        h.validate_no_prior_pass_covers_later_receipt({"s4a_or_s3_pass_covers_s4b": True})


def test_hostile_single_receipt_as_S3_pre_and_post():
    s4a = "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_IMPLEMENT_receipt_v2.json"
    s4b = "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_FINALIZE_receipt_v1.json"
    assert s4a != s4b


def test_hostile_s4b_used_as_s3_precondition(plan):
    assert plan["executor_split"]["phase_boundary_S3"]["required_s3_go_record"]["gates_on"].startswith(
        "S4a"
    )


def test_hostile_claim_ISRV_as_governing_parity(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    assert "C" not in evidence.parity_binding_set_ids
    assert evidence.reference_paths_classification["C_ISRV"]["dense_recarry_target"] is False


def test_hostile_claim_clean_porcelain_when_dirty(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    dirty = [
        "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py",
        "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py",
    ]
    for p in dirty:
        assert evidence.git_porcelain_over_closure.get(p) != "clean"


def test_hostile_pytest_after_oexcl_mint_ordering_constant(plan):
    assert plan["focused_pytest_command"]["order"].startswith("BEFORE")


def test_phase_dag_acyclic(h, plan):
    order = h.validate_phase_dag_acyclic(plan)
    assert order[0] == "S1_harness_evidence_type"
    assert order[-1] == "staging_commit_gate"


def test_text_vs_dag_edge_consistency(h, plan):
    h.validate_text_vs_dag_edge_consistency(plan)


def test_per_receipt_gate_rows_present(h, plan):
    h.validate_per_receipt_gate_rows_present(plan)


def test_hostile_s4b_before_s3_gate2_pass_stop_present(plan):
    stops = "\n".join(plan["stop_conditions"])
    assert "S4b minted before S3 co_lead PASS" in stops or "co_lead PASS" in stops


def test_hostile_s4b_requires_prior_weaker_disjunct(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["phase_dag_self_check"]["text_vs_dag_edge_consistency"]["weaker_disjuncts_remaining"] = 1
    with pytest.raises(ValueError, match="weaker_disjunct"):
        h.validate_text_vs_dag_edge_consistency(bad)


def test_evidence_receipt_field_equality_rejects_forged_branch(h, plan):
    evidence = h.collect_recarry_evidence(
        repo_root=REPO_ROOT, plan=plan, s3_authority_anchor_msg_id="a"
    )
    receipt = evidence.to_receipt_fields()
    receipt["events_equal"] = not evidence.events_equal
    with pytest.raises(ValueError, match="evidence↔receipt inequality"):
        h.validate_evidence_receipt_field_equality(evidence, receipt)


def test_hostile_oexcl_preexisting_refuses_without_overwrite(h, tmp_path):
    target = tmp_path / "preexisting.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        h.write_oexcl_receipt(target, {"ok": True})


def test_governing_receipt_anchor_only_excludes_go_id(h):
    argv = _anchor_argv("test-only-anchor-id-not-a-launch")
    argv[3] = argv[3].replace("v31", "v34")
    receipt = h.build_governing_receipt(
        plan_path=PLAN_V30.with_name(PLAN_V30.name.replace("v31", "v34")),
        argv=argv,
        s3_authority_anchor_msg_id="test-only-anchor-id-not-a-launch",
    )
    assert receipt["audit_branch_id"] == h.BRANCH_HOLDS
    assert receipt["events_equal"] is True
    assert receipt["s3_authority_anchor_msg_id"] == "test-only-anchor-id-not-a-launch"
    assert receipt["authority_verification"] == "gate_pending"
    assert "s3_go_msg_id" not in receipt
    assert receipt["plan_sha256"] == h.PLAN_SHA256_EXPECTED
    assert receipt["plan_revision"] == "v34"
    assert receipt["evidence_to_receipt_field_equality_validated"] is True
    assert receipt["transient_fp_debt_remains"] is True


def test_hostile_argv_template_deviation(h):
    argv = _anchor_argv("anchor")
    argv[3] = "artifacts/acc_entropy/wrong_plan.json"
    with pytest.raises(ValueError, match="argv_template_deviation"):
        h.build_governing_receipt(
            plan_path=PLAN_V30.with_name(PLAN_V30.name.replace("v31", "v34")), argv=argv, s3_authority_anchor_msg_id="anchor"
        )


def test_plan_derived_review_checklist_axis_8(plan):
    axes = plan["plan_derived_review_checklist"]["axes"]
    assert "8_forbidden_field_absent_from_produced_evidence_artifact" in axes


def test_harness_file_sha256_embedded_and_matches_runtime(h):
    argv = _anchor_argv("test-only-anchor-id-not-a-launch")
    argv[3] = argv[3].replace("v31", "v34")
    receipt = h.build_governing_receipt(
        plan_path=PLAN_V30.with_name(PLAN_V30.name.replace("v31", "v34")),
        argv=argv,
        s3_authority_anchor_msg_id="test-only-anchor-id-not-a-launch",
    )
    runtime = h.harness_file_sha256()
    assert receipt["harness_file_sha256"] == runtime
    assert "dry_exec" not in receipt


def test_validate_harness_sha_binding_fields_present_and_equal(h):
    sha = "abc123"
    h.validate_harness_sha_binding_fields_present_and_equal(
        fields={
            "anchor_harness_sha": sha,
            "go_harness_sha": sha,
            "terminal_harness_sha": sha,
            "audit_harness_file_sha256": sha,
            "gate1_freeze_harness_sha": sha,
        },
        required_keys=(
            "anchor_harness_sha",
            "go_harness_sha",
            "terminal_harness_sha",
            "audit_harness_file_sha256",
            "gate1_freeze_harness_sha",
        ),
    )
    with pytest.raises(ValueError, match="harness_sha_binding_mismatch"):
        h.validate_harness_sha_binding_fields_present_and_equal(
            fields={
                "anchor_harness_sha": sha,
                "go_harness_sha": "other",
                "terminal_harness_sha": sha,
                "audit_harness_file_sha256": sha,
                "gate1_freeze_harness_sha": sha,
            },
            required_keys=(
                "anchor_harness_sha",
                "go_harness_sha",
                "terminal_harness_sha",
                "audit_harness_file_sha256",
                "gate1_freeze_harness_sha",
            ),
        )
    with pytest.raises(ValueError, match="harness_sha_binding_missing"):
        h.validate_harness_sha_binding_fields_present_and_equal(
            fields={"anchor_harness_sha": sha},
            required_keys=("anchor_harness_sha", "go_harness_sha"),
        )


def test_hostile_dry_exec_out_equals_canonical_path(h, tmp_path):
    canonical = str((tmp_path / "governing.json").resolve())
    with pytest.raises(ValueError, match="dry_exec_out_equals_canonical_governing_path"):
        h.validate_dry_exec_out_not_canonical_governing_path(
            dry_exec_out=canonical, canonical_out=canonical
        )


def test_hostile_dry_exec_co_occurs_with_formal_run_marker(h):
    with pytest.raises(ValueError, match="dry_exec_co_occurs_with_formal_run_marker"):
        h.validate_dry_exec_not_with_formal_run_marker(
            dry_exec_out_present=True, formal_run_marker=True
        )


def test_hostile_formal_artifact_rejects_dry_exec_key(h):
    with pytest.raises(ValueError, match="formal_artifact_contains_dry_exec_key"):
        h.validate_formal_artifact_rejects_dry_exec_key({"dry_exec": True})


def test_dry_exec_out_scratch_path_cli(h, tmp_path):
    dry = tmp_path / "scratch_dry.json"
    canonical = (
        REPO_ROOT
        / "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json"
    )
    assert not dry.exists()
    # canonical may or may not exist; ensure we don't write it
    existed = canonical.exists()
    rc = h.main(
        [
            "--plan",
            str(PLAN_V30.with_name(PLAN_V30.name.replace("v31", "v34")).relative_to(REPO_ROOT)),
            "--out",
            "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json",
            "--s3-authority-anchor-msg-id",
            "dry-exec-test-anchor",
            "--dry-exec-out",
            str(dry),
        ]
    )
    assert rc == 0
    assert dry.exists()
    assert canonical.exists() is existed
    doc = json.loads(dry.read_text(encoding="utf-8"))
    assert doc.get("dry_exec") is True
    assert "harness_file_sha256" in doc
    assert "s3_go_msg_id" not in doc


def test_hostile_plan_anchor_go_citation_missing_harness_sha(h, plan):
    bad = json.loads(json.dumps(plan))
    reqs = bad["governing_runtime_command"]["authorization"]["requires"]
    reqs[1] = "persisted claude-authored S3 authority-anchor citing PLAN_v31 sha + argv_template_sha256"
    with pytest.raises(ValueError, match="anchor_go_citation_missing_harness_file_sha256"):
        h.validate_plan_anchor_go_citations_carry_harness_sha(bad)


def test_hostile_false_complete_s3_s4b_status(h, plan):
    bad = json.loads(json.dumps(plan))
    for step in bad["DEVELOPER_STEPS"]:
        if step.get("id") == "S0_v24_S3_execute_governing_and_bind":
            step["status"] = "COMPLETE_UNDER_V24_LINEAGE_SUPERSEDED"
            break
    with pytest.raises(ValueError, match="false_complete_s3_s4b_status"):
        h.validate_no_false_complete_s3_s4b_status(bad)


def test_hostile_active_ids_foreign_generation(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["DEVELOPER_STEPS"].append(
        {"id": "S0_v24_stale_active", "status": "ACTIVE", "actions": ["x"], "assertions": []}
    )
    with pytest.raises(ValueError, match="active_ids_set_mismatch"):
        h.validate_active_ids_exactly_current_generation(bad)


def test_hostile_lifecycle_prospective_dual_accept(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_v30_lifecycle"]["active_after"] = (
        "PLAN_v30 dual-accept + fresh v30 +1 implement (this dispatch carries both)"
    )
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_current_operative_oldgen_citation(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["governing_runtime_command"]["authorization"]["requires"][1] = (
        "persisted claude-authored S3 authority-anchor citing PLAN_v24 sha + argv_template_sha256"
    )
    with pytest.raises(ValueError, match="anchor_go_citation_stale_generation"):
        h.validate_plan_anchor_go_citations_carry_harness_sha(bad)


def test_hostile_active_ids_missing_s3_execute(h, plan):
    bad = json.loads(json.dumps(plan))
    for step in bad["DEVELOPER_STEPS"]:
        if step.get("id") == "S0_v31_S3_execute_governing_and_bind":
            step["status"] = "SUPERSEDED_DEFINITION_NEVER_EXECUTED"
            break
    with pytest.raises(ValueError, match="active_ids_set_mismatch"):
        h.validate_active_ids_exactly_current_generation(bad)


def test_hostile_active_ids_extra_current_prefix(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["DEVELOPER_STEPS"].append(
        {"id": "S0_v31_extra_active", "status": "ACTIVE", "actions": ["x"], "assertions": []}
    )
    with pytest.raises(ValueError, match="active_ids_set_mismatch"):
        h.validate_active_ids_exactly_current_generation(bad)


def test_hostile_forged_terminal_field_false_complete(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["bound_s3_terminal_receipt_msg_id"] = "forged-terminal"
    for step in bad["DEVELOPER_STEPS"]:
        if step.get("id") == "S0_v25_S3_execute_governing_and_bind":
            step["status"] = "COMPLETE_UNDER_V25_LINEAGE_SUPERSEDED"
            break
    with pytest.raises(ValueError, match="false_complete_s3_s4b_status"):
        h.validate_no_false_complete_s3_s4b_status(bad)


def test_hostile_lifecycle_wrong_dispatch_id_shape_valid(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_v30_lifecycle"]["active_after"] = (
        "PLAN_v30 minted under converged fast-path +1 implement dispatch 9999999999999-deadbeef; "
        "dual accept applies to the frozen artifact review, not the mint."
    )
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_action_class_dual_create_preserve(h, plan):
    bad = json.loads(json.dumps(plan))
    manifest = bad["active_write_manifest"]
    path = manifest["CREATE"][0]
    manifest.setdefault("PRESERVE_BYTES", []).append(path)
    with pytest.raises(ValueError, match="action_class_dual_create_preserve|action_class_conflict"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_supersedes_list_shaped(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["supersedes"] = ["plans", "note"]
    with pytest.raises(ValueError, match="supersedes_not_dict"):
        h.validate_supersedes_schema(bad)


def test_hostile_lifecycle_real_id_wrong_generation(h, plan):
    """G1: real registry id from wrong generation must REJECT (not mere membership)."""
    bad = json.loads(json.dumps(plan))
    wrong = plan["lifecycle_dispatch_id_registry"]["v24"]
    bad["frozen_preconditions"]["P4_write_surface_v30_lifecycle"]["active_after"] = (
        f"PLAN_v30 minted under converged fast-path +1 implement dispatch {wrong}; "
        "dual accept applies to the frozen artifact review, not the mint."
    )
    with pytest.raises(ValueError, match="dispatch_id_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_supersedes_gap_missing_generation(h, plan):
    """G2: monotonic but gapped lineage must REJECT."""
    bad = json.loads(json.dumps(plan))
    bad["supersedes"]["plans"] = [e for e in bad["supersedes"]["plans"] if e["rev"] not in ("v23", "v24")]
    with pytest.raises(ValueError, match="supersedes_revs_not_contiguous"):
        h.validate_supersedes_schema(bad)


def test_hostile_supersedes_entry_equal_current_generation(h, plan):
    """G2: entry >= current generation must REJECT."""
    bad = json.loads(json.dumps(plan))
    cur = bad["plan_revision"]
    bad["supersedes"]["plans"].append({"rev": cur, "sha256": "0" * 64, "preserved_bytes": True})
    with pytest.raises(ValueError, match="supersedes_entry_beyond_or_equal_current|supersedes_revs_not_contiguous"):
        h.validate_supersedes_schema(bad)



def test_hostile_developer_steps_duplicate_active_id(h, plan):
    """N5: duplicate ACTIVE step id must REJECT (set comparison must not collapse)."""
    bad = json.loads(json.dumps(plan))
    s3 = next(s for s in bad["DEVELOPER_STEPS"] if s["id"] == "S0_v30_S3_execute_governing_and_bind")
    bad["DEVELOPER_STEPS"].append(json.loads(json.dumps(s3)))
    with pytest.raises(ValueError, match="developer_steps_id_not_unique"):
        h.validate_developer_steps_ids_unique(bad)


def test_hostile_developer_steps_duplicate_id_conflicting_status(h, plan):
    """N5b: same id ACTIVE + SUPERSEDED coexisting must REJECT."""
    bad = json.loads(json.dumps(plan))
    bad["DEVELOPER_STEPS"].append(
        {
            "id": "S0_v30_S3_execute_governing_and_bind",
            "status": "SUPERSEDED_DEFINITION_NEVER_EXECUTED",
            "actions": [],
            "assertions": [],
        }
    )
    with pytest.raises(ValueError, match="developer_steps_id_not_unique"):
        h.validate_developer_steps_ids_unique(bad)


def test_hostile_action_class_within_list_duplicate_path(h, plan):
    """N6: duplicate path within a single action list must REJECT."""
    bad = json.loads(json.dumps(plan))
    path = bad["active_write_manifest"]["UPDATE"][0]
    bad["active_write_manifest"]["UPDATE"].append(path)
    with pytest.raises(ValueError, match="action_class_within_list_duplicate"):
        h.validate_single_action_class_per_path(bad)



def test_hostile_this_plan_phase_within_list_duplicate(h, plan):
    """I1: duplicate path inside frozen_write_surface.this_plan_phase.UPDATE must REJECT."""
    bad = json.loads(json.dumps(plan))
    upd = bad["frozen_write_surface"]["this_plan_phase"].setdefault("UPDATE", [])
    assert upd, "this_plan_phase.UPDATE must be non-empty in clean plan"
    upd.append(upd[0])
    with pytest.raises(ValueError, match="action_class_within_list_duplicate.*this_plan_phase"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_action_class_cross_surface_divergence(h, plan):
    """I1: two operative manifests assigning different classes to same path must REJECT."""
    bad = json.loads(json.dumps(plan))
    path = bad["active_write_manifest"]["UPDATE"][0]
    # Move path from UPDATE → CREATE on this_plan_phase only.
    tpp = bad["frozen_write_surface"]["this_plan_phase"]
    tpp["UPDATE"] = [p for p in tpp.get("UPDATE", []) if p != path]
    tpp.setdefault("CREATE", []).append(path)
    with pytest.raises(ValueError, match="action_class_cross_surface_not_exact|action_class_cross_surface_divergence"):
        h.validate_single_action_class_per_path(bad)



def test_hostile_action_class_surface_omission(h, plan):
    """R1: drop path from this_plan_phase (missing vs canonical) must REJECT."""
    bad = json.loads(json.dumps(plan))
    valp = "calm/hrm_text_158/native_full_stack/recarry_measurement_validators.py"
    bad["frozen_write_surface"]["this_plan_phase"]["UPDATE"] = [
        p for p in bad["frozen_write_surface"]["this_plan_phase"]["UPDATE"] if p != valp
    ]
    with pytest.raises(ValueError, match="action_class_cross_surface_not_exact"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_action_class_surface_extra_path(h, plan):
    """R1: extra path on an operative surface must REJECT."""
    bad = json.loads(json.dumps(plan))
    extra = "artifacts/acc_entropy/EXTRA_SHOULD_NOT_EXIST.py"
    bad["frozen_write_surface"]["this_plan_phase"]["UPDATE"].append(extra)
    with pytest.raises(ValueError, match="action_class_cross_surface_not_exact"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_plan_artifact_wrong_generation(h, plan):
    """R1: all-surfaces-consistent PLAN_v99 must REJECT identity check."""
    bad = json.loads(json.dumps(plan))
    wrong = "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_PLAN_v99.json"
    for _name, man in [
        ("awm", bad["active_write_manifest"]),
        ("tpp", bad["frozen_write_surface"]["this_plan_phase"]),
        ("fws", bad["frozen_write_surface"]["post_plus1_implement_bounded_v31_correction"]),
    ]:
        man["CREATE_THEN_PRESERVE"] = [wrong]
    # also ACTIVE fp blocks
    for key, block in bad["frozen_preconditions"].items():
        if isinstance(block, dict) and block.get("status") == "ACTIVE":
            if isinstance(block.get("manifest"), dict):
                block["manifest"]["CREATE_THEN_PRESERVE"] = [wrong]
            if "CREATE_THEN_PRESERVE" in block:
                block["CREATE_THEN_PRESERVE"] = [wrong]
    with pytest.raises(ValueError, match="plan_artifact_identity_mismatch"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_action_class_in_out_conflict(h, plan):
    """R1: IN↔OUT conflict for scoped file path must REJECT."""
    bad = json.loads(json.dumps(plan))
    p0 = bad["active_write_manifest"]["UPDATE"][0]
    for _n, man in [
        ("awm", bad["active_write_manifest"]),
        ("tpp", bad["frozen_write_surface"]["this_plan_phase"]),
        ("fws", bad["frozen_write_surface"]["post_plus1_implement_bounded_v31_correction"]),
    ]:
        man.setdefault("OUT", []).append(p0)
    for key, block in bad["frozen_preconditions"].items():
        if isinstance(block, dict) and block.get("status") == "ACTIVE":
            target = block["manifest"] if isinstance(block.get("manifest"), dict) else block
            if isinstance(target, dict) and any(k in target for k in ("UPDATE", "CREATE")):
                target.setdefault("OUT", []).append(p0)
    with pytest.raises(ValueError, match="action_class_in_out_conflict"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_lifecycle_wrong_plan_generation_text(h, plan):
    """R2: current leaf wrong PLAN_vN text with correct dispatch id must REJECT."""
    bad = json.loads(json.dumps(plan))
    did = plan["lifecycle_dispatch_id_registry"]["v30"]
    bad["frozen_preconditions"]["P4_write_surface_v30_lifecycle"]["active_after"] = (
        f"PLAN_v99 minted under converged fast-path +1 implement dispatch {did}; "
        "dual accept applies to the frozen artifact review, not the mint."
    )
    with pytest.raises(ValueError, match="plan_generation_text_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_lifecycle_historical_prospective(h, plan):
    """R2: historical leaf prospective dual-accept wording must REJECT (no exemption)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_v24_lifecycle"]["active_after"] = (
        "PLAN_v24 dual-accept + fresh v24 +1 implement (this dispatch carries both)"
    )
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_lifecycle_historical_generation_id_mismatch(h, plan):
    """R2: historical leaf generation/id mismatch must REJECT."""
    bad = json.loads(json.dumps(plan))
    wrong = plan["lifecycle_dispatch_id_registry"]["v25"]
    bad["frozen_preconditions"]["P4_write_surface_v24_lifecycle"]["active_after"] = (
        f"PLAN_v24 minted under converged fast-path +1 implement dispatch {wrong}; "
        "dual accept applies to the frozen artifact review, not the mint."
    )
    with pytest.raises(ValueError, match="dispatch_id_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_out_add_nonconflicting_path(h, plan):
    """R1: add nonconflicting OUT path on one surface → REJECT."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_write_surface"]["this_plan_phase"].setdefault("OUT", []).append("extra_out_path")
    with pytest.raises(ValueError, match="action_class_cross_surface_not_exact"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_out_delete_path_from_one_surface(h, plan):
    """R1: delete an OUT path from one surface → REJECT."""
    bad = json.loads(json.dumps(plan))
    out = bad["frozen_write_surface"]["this_plan_phase"]["OUT"]
    assert "push" in out
    bad["frozen_write_surface"]["this_plan_phase"]["OUT"] = [x for x in out if x != "push"]
    with pytest.raises(ValueError, match="action_class_cross_surface_not_exact"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_out_list_absent_on_surface(h, plan):
    """R1/F1: surface with OUT key absent → REJECT (absolute schema, not sentinel equality)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_write_surface"]["this_plan_phase"].pop("OUT", None)
    with pytest.raises(
        ValueError,
        match="required_list_absent|canonical_manifest_missing_required_list|action_class_cross_surface_not_exact",
    ):
        h.validate_single_action_class_per_path(bad)


def test_hostile_lifecycle_synonym_both_carried(h, plan):
    """R2: synonym 'dual accept and +1 implement are both carried by this dispatch' → REJECT."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v31"]["active_after"] = (
        "PLAN_v31 dual accept and +1 implement are both carried by this dispatch 1785162832443-e53d58be"
    )
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_lifecycle_extra_freetext_appended(h, plan):
    """R2: exact grammar + extra free-text → REJECT."""
    from calm.hrm_text_158.native_full_stack.recarry_measurement_validators import (
        _lifecycle_authority_grammar,
    )
    bad = json.loads(json.dumps(plan))
    g = _lifecycle_authority_grammar("v31", plan["lifecycle_dispatch_id_registry"]["v31"])
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v31"]["active_after"] = g + " (extra)"
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_lifecycle_grammar_wrong_id_or_generation(h, plan):
    """R2: correct grammar shape but wrong id/generation → REJECT."""
    from calm.hrm_text_158.native_full_stack.recarry_measurement_validators import (
        _lifecycle_authority_grammar,
    )
    bad = json.loads(json.dumps(plan))
    # wrong generation text with a real registry id from v30
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v31"]["active_after"] = (
        _lifecycle_authority_grammar("v30", plan["lifecycle_dispatch_id_registry"]["v30"])
    )
    with pytest.raises(ValueError, match="lifecycle_authority_cite_invalid"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


# --- PLAN_v5 Stage-1 structural hardening hostiles (F1/F2/F3; old-PASS → new-REJECT) ---


def test_hostile_remove_OUT_from_all_surfaces(h, plan):
    """F1: delete OUT from every operative surface → REJECT (was sentinel vacuity PASS)."""
    from calm.hrm_text_158.native_full_stack.recarry_measurement_validators import (
        _iter_operative_action_manifests,
    )

    bad = json.loads(json.dumps(plan))
    for _name, man in _iter_operative_action_manifests(bad):
        man.pop("OUT", None)
    with pytest.raises(
        ValueError,
        match="canonical_manifest_missing_required_list|required_list_absent|OUT",
    ):
        h.validate_single_action_class_per_path(bad)


def test_hostile_OUT_wrong_type_on_canonical(h, plan):
    """F1: canonical active_write_manifest.OUT as non-list → REJECT."""
    bad = json.loads(json.dumps(plan))
    bad["active_write_manifest"]["OUT"] = "not-a-list"
    with pytest.raises(ValueError, match="required_list_not_list:OUT"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_erase_this_plan_phase_class_fields(h, plan):
    """F2: erase action-class fields from this_plan_phase → REJECT (was hidden PASS)."""
    bad = json.loads(json.dumps(plan))
    tpp = bad["frozen_write_surface"]["this_plan_phase"]
    for key in (
        "UPDATE",
        "CREATE",
        "CREATE_THEN_PRESERVE",
        "PRESERVE_BYTES",
        "VALIDATE_ONLY",
        "OUT",
    ):
        tpp.pop(key, None)
    with pytest.raises(
        ValueError,
        match="required_operative_surface_absent|this_plan_phase|required_list_absent",
    ):
        h.validate_single_action_class_per_path(bad)


def test_hostile_delete_required_active_phase_block(h, plan):
    """F2: delete required active_phase_v{N} block → REJECT."""
    bad = json.loads(json.dumps(plan))
    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    n = revision[1:]
    del bad["frozen_preconditions"][f"P4_write_surface_active_phase_v{n}"]
    with pytest.raises(ValueError, match="required_operative_surface_absent:.*active_phase"):
        h.validate_single_action_class_per_path(bad)


def test_hostile_active_phase_note_to_ok(h, plan):
    """F3: version-addressable active_phase note='ok' → REJECT (was prose-skip PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v25"]["note"] = "ok"
    with pytest.raises(
        ValueError, match="lifecycle_grammar_mismatch|lifecycle_authority_cite_invalid"
    ):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_bounded_vN_note_to_ok(h, plan):
    """F3: version-addressable bounded_vN note='ok' → REJECT (was prose-skip PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_write_surface"]["post_plus1_implement_bounded_v20_correction"]["note"] = "ok"
    with pytest.raises(
        ValueError, match="lifecycle_grammar_mismatch|lifecycle_authority_cite_invalid"
    ):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_active_phase_note_none(h, plan):
    """F3 type-boundary: governed note=None → REJECT (was isinstance-continue PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v25"]["note"] = None
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_bounded_vN_note_int(h, plan):
    """F3 type-boundary: governed note=int → REJECT (was isinstance-continue PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_write_surface"]["post_plus1_implement_bounded_v20_correction"]["note"] = 7
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_lifecycle_active_after_none(h, plan):
    """F3 type-boundary: governed active_after=None → REJECT (was isinstance-continue PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_v24_lifecycle"]["active_after"] = None
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_active_phase_note_empty_dict(h, plan):
    """F3 Correction-2: governed note={} → REJECT (was scalar-walk hide PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v25"]["note"] = {}
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_bounded_vN_note_empty_list(h, plan):
    """F3 Correction-2: governed note=[] → REJECT (was scalar-walk hide PASS)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_write_surface"]["post_plus1_implement_bounded_v20_correction"]["note"] = []
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_hostile_active_phase_note_nonempty_dict(h, plan):
    """F3 Correction-2: governed note={child:...} → REJECT (child leaf names hide parent)."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v25"]["note"] = {"text": "ok"}
    with pytest.raises(ValueError, match="lifecycle_leaf_not_string"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_scope_preserve_nongoverned_container_note_accept(h, plan):
    """Non-governed top-level note={} stays ACCEPT (scope preservation)."""
    ok = json.loads(json.dumps(plan))
    ok["note"] = {}
    ok["unrelated_block"] = {"note": []}
    h.validate_lifecycle_authority_cites_dispatch(ok)


# ---------------------------------------------------------------------------
# F4 Step-A (PLAN_v10): mode-aware lifecycle grammar — PUBLIC entrypoint only
# All fixtures exercise validate_lifecycle_authority_cites_dispatch end-to-end.
# ---------------------------------------------------------------------------

PLAN_V31 = PLAN_V30  # alias: fixture path is PLAN_v31
PLAN_V32 = (
    REPO_ROOT
    / "artifacts"
    / "acc_entropy"
    / "optimizer_credit_state_projected_moves_recarry_measurement_PLAN_v32.json"
)
_F4_ROOM_ID = "1785168271320-9581a9c1"
_F4_ROOM_ID_ALT = "1785171120714-3ccd9924"


def _f4_grammar(gen: str, dispatch_id: str, mode: str) -> str:
    return _recarry_validators._lifecycle_authority_grammar(gen, dispatch_id, mode)


def _f4_set_gen_leaves(plan: dict, gen: str, text: str) -> None:
    for path, _value in _recarry_validators._walk_governed_lifecycle_key_candidates(plan):
        if _recarry_validators._lifecycle_path_generation(path) != gen:
            continue
        parts = path.lstrip("$").lstrip(".").split(".")
        cur: object = plan
        for p in parts[:-1]:
            assert isinstance(cur, dict)
            cur = cur[p]
        assert isinstance(cur, dict)
        cur[parts[-1]] = text


def _f4_delete_gen_leaves(plan: dict, gen: str) -> None:
    """Remove governed note/active_after values for gen (delete keys)."""
    targets = []
    for path, _value in _recarry_validators._walk_governed_lifecycle_key_candidates(plan):
        if _recarry_validators._lifecycle_path_generation(path) == gen:
            targets.append(path)
    for path in targets:
        parts = path.lstrip("$").lstrip(".").split(".")
        cur: object = plan
        for p in parts[:-1]:
            assert isinstance(cur, dict)
            cur = cur[p]
        assert isinstance(cur, dict)
        del cur[parts[-1]]


def test_f4_accept_full_plan_gate_mode_with_exact_full_plan_text(h, plan):
    """Positive: {id,mode:full_plan_gate} + exact full-plan text → ACCEPT (public path)."""
    ok = json.loads(json.dumps(plan))
    rid = ok["lifecycle_dispatch_id_registry"]["v31"]
    assert isinstance(rid, str)
    ok["lifecycle_dispatch_id_registry"]["v31"] = {"id": rid, "mode": "full_plan_gate"}
    text = _f4_grammar("v31", rid, "full_plan_gate")
    _f4_set_gen_leaves(ok, "v31", text)
    h.validate_lifecycle_authority_cites_dispatch(ok)


def test_f4_accept_fast_path_mode_carrying_with_exact_fast_path_text(h, plan):
    """Positive: {id,mode:fast_path} + exact fast-path text → ACCEPT (public path)."""
    ok = json.loads(json.dumps(plan))
    rid = ok["lifecycle_dispatch_id_registry"]["v31"]
    assert isinstance(rid, str)
    ok["lifecycle_dispatch_id_registry"]["v31"] = {"id": rid, "mode": "fast_path"}
    text = _f4_grammar("v31", rid, "fast_path")
    _f4_set_gen_leaves(ok, "v31", text)
    h.validate_lifecycle_authority_cites_dispatch(ok)


def test_f4_hostile_compat_plain_string_v31_still_accepts(h, plan):
    """Compat: immutable PLAN_v31 plain-string registry → ACCEPT."""
    doc = json.loads(PLAN_V31.read_text(encoding="utf-8"))
    assert _sha_file(PLAN_V31) == (
        "69041cbe5890c5b9d1af0c85ccdacbc6767147e2f263236629aca871939bf6ba"
    )
    h.validate_lifecycle_authority_cites_dispatch(doc)


def test_f4_hostile_compat_plain_string_v32_still_accepts_as_historical_grammar(h):
    """Compat: immutable PLAN_v32 plain-string registry → ACCEPT (historical grammar)."""
    doc = json.loads(PLAN_V32.read_text(encoding="utf-8"))
    assert _sha_file(PLAN_V32) == (
        "55417cff418f9625474e9818535def05355e33f3255fa6469c923c9ea6ca0cd7"
    )
    h.validate_lifecycle_authority_cites_dispatch(doc)


def test_f4_hostile_fast_path_text_under_full_plan_gate_mode(h, plan):
    bad = json.loads(json.dumps(plan))
    rid = bad["lifecycle_dispatch_id_registry"]["v31"]
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": rid, "mode": "full_plan_gate"}
    # leave leaves as fast_path text
    with pytest.raises(ValueError, match="lifecycle_mode_text_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_full_plan_text_under_fast_path_mode(h, plan):
    bad = json.loads(json.dumps(plan))
    rid = bad["lifecycle_dispatch_id_registry"]["v31"]
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": rid, "mode": "fast_path"}
    _f4_set_gen_leaves(bad, "v31", _f4_grammar("v31", rid, "full_plan_gate"))
    with pytest.raises(ValueError, match="lifecycle_mode_text_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_unknown_lifecycle_mode(h, plan):
    bad = json.loads(json.dumps(plan))
    rid = bad["lifecycle_dispatch_id_registry"]["v31"]
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": rid, "mode": "other_mode"}
    with pytest.raises(ValueError, match="lifecycle_mode_unknown"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_v33_plain_string(h):
    """v33+ plain-string registry entry → REJECT (legacy cutoff). Base = immutable v32."""
    bad = json.loads(PLAN_V32.read_text(encoding="utf-8"))
    bad["plan_revision"] = "v33"
    bad["lifecycle_dispatch_id_registry"]["v33"] = _F4_ROOM_ID
    bad.setdefault("frozen_preconditions", {})
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v33"] = {
        "note": _f4_grammar("v33", _F4_ROOM_ID, "fast_path")
    }
    with pytest.raises(ValueError, match="lifecycle_registry_legacy_string_beyond_v32"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_entry_non_mapping_non_string(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = 7
    with pytest.raises(ValueError, match="lifecycle_registry_entry_type"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_mapping_missing_id(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"mode": "fast_path"}
    with pytest.raises(ValueError, match="lifecycle_registry_missing_id"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_mapping_missing_mode(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": _F4_ROOM_ID}
    with pytest.raises(ValueError, match="lifecycle_registry_missing_mode"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_id_empty(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": "", "mode": "fast_path"}
    with pytest.raises(ValueError, match="lifecycle_registry_id_empty"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_id_non_string(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": 123, "mode": "fast_path"}
    with pytest.raises(ValueError, match="lifecycle_registry_id_type"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_id_malformed_shape(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {
        "id": "not-a-room-message-id",
        "mode": "fast_path",
    }
    with pytest.raises(ValueError, match="lifecycle_registry_id_shape"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_mode_empty_or_non_string(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {"id": _F4_ROOM_ID, "mode": ""}
    with pytest.raises(ValueError, match="lifecycle_registry_mode_type"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_unknown_mode(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {
        "id": _F4_ROOM_ID,
        "mode": "not_a_real_mode",
    }
    with pytest.raises(ValueError, match="lifecycle_mode_unknown"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_extra_key(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v31"] = {
        "id": _F4_ROOM_ID,
        "mode": "fast_path",
        "extra": True,
    }
    with pytest.raises(ValueError, match="lifecycle_registry_extra_key"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_unreferenced_registry_entry_non_string(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v99"] = 7
    with pytest.raises(ValueError, match="lifecycle_registry_entry_type"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_unreferenced_registry_entry_malformed_string_id(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v99"] = "not-a-room-message-id"
    with pytest.raises(
        ValueError,
        match="lifecycle_registry_legacy_string_beyond_v32|lifecycle_registry_id_shape",
    ):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_malformed_key_shape(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["vv25"] = _F4_ROOM_ID
    with pytest.raises(ValueError, match="lifecycle_registry_key_shape"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_legacy_plain_string_malformed_id_shape(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v25"] = "not-a-room-message-id"
    with pytest.raises(ValueError, match="lifecycle_registry_id_shape"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_unreferenced_wellformed_registry_v99_mapping(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v99"] = {
        "id": _F4_ROOM_ID,
        "mode": "fast_path",
    }
    with pytest.raises(ValueError, match="lifecycle_registry_generation_extra"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_leading_zero_alias_v031(h, plan):
    bad = json.loads(json.dumps(plan))
    bad["lifecycle_dispatch_id_registry"]["v031"] = {
        "id": _F4_ROOM_ID,
        "mode": "fast_path",
    }
    with pytest.raises(ValueError, match="lifecycle_registry_key_not_canonical"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_generation_missing_required(h, plan):
    bad = json.loads(json.dumps(plan))
    del bad["lifecycle_dispatch_id_registry"]["v25"]
    with pytest.raises(ValueError, match="lifecycle_registry_generation_missing"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_registry_only_generation_vs_governed(h, plan):
    """registry − governed direction: delete all governed leaves for v25."""
    bad = json.loads(json.dumps(plan))
    _f4_delete_gen_leaves(bad, "v25")
    with pytest.raises(ValueError, match="lifecycle_registry_governed_set_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)


def test_f4_hostile_governed_only_generation_vs_registry(h, plan):
    """governed − registry direction: inject governed leaf gen absent from registry."""
    bad = json.loads(json.dumps(plan))
    bad["frozen_preconditions"]["P4_write_surface_active_phase_v50"] = {
        "note": _f4_grammar("v50", _F4_ROOM_ID, "fast_path")
    }
    with pytest.raises(ValueError, match="lifecycle_registry_governed_set_mismatch"):
        h.validate_lifecycle_authority_cites_dispatch(bad)
