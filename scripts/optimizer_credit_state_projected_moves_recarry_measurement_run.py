#!/usr/bin/env python3
"""Projected-moves re-carry measurement thin harness (current-plan compatibility; governing generation = PLAN_REVISION constant / S4a-R thin harness).

CLI / path wiring / O_EXCL receipt mint only. Validators and evidence live in
importable modules. Dependency direction: this script -> modules (never reverse).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (  # noqa: E402
    compute_canonical_json_sha256,
)
from calm.hrm_text_158.native_full_stack.recarry_measurement_evidence import (  # noqa: E402
    FIXTURE_RECIPE_NAME,
    PROBE_MODE,
    PRODUCER_SWEEP_HIT_COUNT,
    RANK_SPEC_DIGEST_EXPECTED,
    RANK_SPEC_SYMBOL,
    ProjectedMovesRecarryMeasurementEvidence,
    collect_recarry_evidence,
    _git_porcelain_map,
    _live_repo_head,
    _sha_file,
    validate_dependency_currency_against_plan_pins,
    validate_import_closure_pins,
    validate_parity_uses_live_consumer_symbols_not_reimpl,
    validate_producer_contract_unchanged,
)

# Plan-version binding lives ONLY in this thin harness (not in pinned modules).
PLAN_SHA256_EXPECTED = (
    "0a47d2550b85cc1a26153e3439da38d08d45fb559665ce7b248ef1e6c736efee"
)
PLAN_REVISION = "v34"
ARGV_TEMPLATE_SHA256_EXPECTED = (
    "5a1faff46d7fa3dc2b69a9aa2be4e66abdad1c16656fce00845569778d8d2d24"
)
from calm.hrm_text_158.native_full_stack.recarry_measurement_validators import (  # noqa: E402
    ADJACENCY_DISPOSITION,
    ADJACENT_CALLER_PATHS,
    AUTHORITY_VERIFICATION_AT_MINT,
    BRANCH_HOLDS,
    BRANCH_INFEASIBLE,
    BRANCH_INVALID,
    BRANCH_PARITY_FAIL,
    BRANCH_PENDING,
    S3_FORBIDDEN_ON_S4A,
    TERMINAL_ALLOWED,
    _require_exact_int,
    classify_recarry_branch,
    validate_adjacency_disposition_present,
    validate_authority_verification_gate_pending_at_harness_mint,
    validate_argv_flag_is_s3_authority_anchor_msg_id,
    validate_argv_index7_equals_authority_anchor_not_go,
    validate_argv_matches_frozen_template_after_instantiation,
    validate_both_evidence_sources_cited,
    validate_compositional_reduction_bound,
    validate_evidence_receipt_field_equality,
    validate_events_equal_derived_not_hand_authored,
    validate_governing_claim_AB_only,
    validate_governing_claim_AB_only_semantic,
    validate_harness_evidence_and_audit_exclude_s3_go_msg_id,
    validate_live_repo_head_matches_claim,
    validate_live_repo_head_matches_plan_pin,
    validate_no_bind_by_id_evidence_path,
    validate_no_caller_supplied_resolution_mapping,
    validate_no_prior_pass_covers_later_receipt,
    validate_no_s3_go_msg_id_harness_input,
    validate_per_receipt_gate_rows_present,
    validate_phase_dag_acyclic,
    validate_porcelain_truthful,
    validate_rank_spec_identity_matches_plan_pin,
    validate_s3_authority_anchor_and_go_ids_distinct,
    validate_s3_authority_anchor_msg_id_present,
    validate_s3_review_cycle_complete_before_s4b,
    validate_s3_review_ids_resolved_author_thread_verdict,
    validate_s4a_has_no_s3_fields,
    validate_s4b_binds_s3_ids,
    validate_s4b_binds_s3_review_ids,
    validate_text_vs_dag_edge_consistency,
    validate_dry_exec_out_not_canonical_governing_path,
    validate_dry_exec_not_with_formal_run_marker,
    validate_dry_exec_artifact_marks_dry_exec,
    validate_formal_artifact_rejects_dry_exec_key,
    validate_harness_file_sha256_present_and_matches_runtime,
    validate_harness_sha_binding_fields_present_and_equal,
    validate_plan_anchor_go_citations_carry_harness_sha,
    validate_no_false_complete_s3_s4b_status,
    validate_developer_steps_ids_unique,
    validate_active_ids_exactly_current_generation,
    validate_lifecycle_authority_cites_dispatch,
    validate_single_action_class_per_path,
    validate_supersedes_schema,
)



def harness_file_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def validate_oexcl_open_x(*, out_path: Path, write_fn) -> None:
    with out_path.open("x", encoding="utf-8") as fh:
        write_fn(fh)


def build_governing_receipt(
    *,
    plan_path: Path,
    argv: list[str],
    s3_authority_anchor_msg_id: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    validate_no_s3_go_msg_id_harness_input(
        argv=argv, kwargs={"s3_authority_anchor_msg_id": s3_authority_anchor_msg_id}
    )
    if argv and argv[0] != "python3":
        raise ValueError("python_not_python3_in_frozen_cmd")
    if not s3_authority_anchor_msg_id or not str(s3_authority_anchor_msg_id).strip():
        raise ValueError("missing_s3_authority_anchor_msg_id")
    if s3_authority_anchor_msg_id == "<S3_AUTHORITY_ANCHOR_MSG_ID>":
        raise ValueError("literal placeholder launch forbidden")
    plan_sha = _sha_file(plan_path)
    if plan_sha != PLAN_SHA256_EXPECTED:
        raise ValueError(f"plan sha mismatch: expected={PLAN_SHA256_EXPECTED} actual={plan_sha}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_phase_dag_acyclic(plan)
    validate_text_vs_dag_edge_consistency(plan)
    validate_per_receipt_gate_rows_present(plan)
    validate_plan_anchor_go_citations_carry_harness_sha(plan)
    validate_no_false_complete_s3_s4b_status(plan)
    validate_developer_steps_ids_unique(plan)
    validate_active_ids_exactly_current_generation(plan)
    validate_lifecycle_authority_cites_dispatch(plan)
    validate_single_action_class_per_path(plan)
    validate_supersedes_schema(plan)
    validate_dependency_currency_against_plan_pins(plan=plan, repo_root=repo_root)
    validate_argv_matches_frozen_template_after_instantiation(
        argv=argv, plan=plan, s3_authority_anchor_msg_id=str(s3_authority_anchor_msg_id)
    )
    tmpl_sha = hashlib.sha256(
        json.dumps(
            plan["governing_runtime_command"]["argv_template"],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if tmpl_sha != ARGV_TEMPLATE_SHA256_EXPECTED:
        raise ValueError(f"argv_template_sha256 mismatch: {tmpl_sha}")

    live_head = _live_repo_head(repo_root)
    validate_live_repo_head_matches_plan_pin(live=live_head, plan=plan)

    evidence = collect_recarry_evidence(
        repo_root=repo_root,
        plan=plan,
        s3_authority_anchor_msg_id=str(s3_authority_anchor_msg_id),
    )
    validate_live_repo_head_matches_claim(claimed=evidence.repo_head_sha, live=live_head)
    validate_porcelain_truthful(
        claimed=evidence.git_porcelain_over_closure,
        observed=_git_porcelain_map(
            repo_root, list(evidence.git_porcelain_over_closure.keys())
        ),
    )

    receipt: dict[str, Any] = {
        "schema": "hrm_text_158_optimizer_credit_state_projected_moves_recarry_measurement/v1",
        "plan_path": str(plan_path.as_posix()),
        "plan_sha256": plan_sha,
        "plan_revision": PLAN_REVISION,
        **evidence.to_receipt_fields(),
        "runtime_command_argv": list(argv),
        "runtime_command_sha256": compute_canonical_json_sha256(list(argv)),
        "argv_template_sha256": tmpl_sha,
        "expected_branch_seed158": BRANCH_HOLDS,
        "claim_ceiling": {
            "may_claim": list(plan["claim_ceiling"]["may_claim"]),
            "must_not_claim": list(plan["claim_ceiling"]["must_not_claim"]),
            "transient_fp_debt_remains": True,
            "no_readiness_row_flip": True,
            "authorizes_readiness_row_flip": False,
            "authorizes_sub2_or_resolved": False,
        },
        "required_tokens": [
            "RECARRY_MEASUREMENT_ONLY",
            "GOVERNING_CLAIM_AB_ONLY",
            "COMPOSITIONAL_REDUCTION_BOUND",
            "RANK_SPEC_FROZEN_DEFAULT_DRY_RUN",
            "PENDING_FORBIDDEN_AS_TERMINAL",
            "TRANSIENT_FP_DEBT_REMAINS",
            "UNCHANGED_PRODUCER_CONSTRAINT",
            "AUTHORITY_VERIFICATION_GATE_PENDING",
            "S3_GO_MSG_ID_NOT_HARNESS_INPUT",
            "CREDITDIR_WRITES_OUT",
        ],
    }
    validate_s3_authority_anchor_msg_id_present(receipt)
    validate_authority_verification_gate_pending_at_harness_mint(receipt)
    validate_harness_evidence_and_audit_exclude_s3_go_msg_id(
        evidence_mapping=evidence.to_receipt_fields(), audit_mapping=receipt
    )
    validate_no_caller_supplied_resolution_mapping(receipt)
    validate_no_bind_by_id_evidence_path(receipt)
    validate_evidence_receipt_field_equality(evidence, receipt)
    receipt["evidence_to_receipt_field_equality_validated"] = True
    runtime_sha = harness_file_sha256()
    receipt["harness_file_sha256"] = runtime_sha
    validate_harness_file_sha256_present_and_matches_runtime(
        receipt=receipt, runtime_harness_file_sha256=runtime_sha
    )
    return receipt


def write_oexcl_receipt(out_path: Path, receipt: Mapping[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(fh) -> None:
        fh.write(json.dumps(dict(receipt), indent=2) + "\n")

    try:
        validate_oexcl_open_x(out_path=out_path, write_fn=_write)
    except FileExistsError as exc:
        raise FileExistsError(f"O_EXCL refused; receipt exists: {out_path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--s3-authority-anchor-msg-id",
        required=True,
        help="Persisted claude-authored S3 authority-anchor msg id (sole harness dynamic input)",
    )
    parser.add_argument(
        "--dry-exec-out",
        type=Path,
        default=None,
        help="Dry-exec only: write artifact here; formal --out stays canonical for argv/template checks",
    )
    parser.add_argument(
        "--formal-s3-run",
        action="store_true",
        default=False,
        help="Formal-run marker; incompatible with --dry-exec-out",
    )
    args = parser.parse_args(argv)
    raw = list(argv) if argv is not None else list(sys.argv[1:])
    validate_no_s3_go_msg_id_harness_input(argv=raw)

    dry_present = args.dry_exec_out is not None
    validate_dry_exec_not_with_formal_run_marker(
        dry_exec_out_present=dry_present, formal_run_marker=bool(args.formal_s3_run)
    )

    plan_arg = str(args.plan.as_posix())
    out_arg = str(args.out.as_posix())
    # Formal 8-arg argv always uses canonical --out (template geometry unchanged).
    argv_list = [
        "python3",
        "scripts/optimizer_credit_state_projected_moves_recarry_measurement_run.py",
        "--plan",
        plan_arg,
        "--out",
        out_arg,
        "--s3-authority-anchor-msg-id",
        str(args.s3_authority_anchor_msg_id),
    ]
    plan_path = args.plan if args.plan.is_absolute() else (REPO_ROOT / args.plan).resolve()
    canonical_out = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)
    if dry_present:
        dry_out = args.dry_exec_out if args.dry_exec_out.is_absolute() else (REPO_ROOT / args.dry_exec_out)
        validate_dry_exec_out_not_canonical_governing_path(
            dry_exec_out=str(Path(dry_out).resolve()),
            canonical_out=str(Path(canonical_out).resolve()),
        )
        write_path = dry_out
    else:
        write_path = canonical_out

    receipt = build_governing_receipt(
        plan_path=plan_path,
        argv=argv_list,
        s3_authority_anchor_msg_id=str(args.s3_authority_anchor_msg_id),
    )
    if dry_present:
        receipt["dry_exec"] = True
        validate_dry_exec_artifact_marks_dry_exec(receipt)
    else:
        validate_formal_artifact_rejects_dry_exec_key(receipt)

    write_oexcl_receipt(write_path, receipt)
    print(
        json.dumps(
            {
                "audit_branch_id": receipt["audit_branch_id"],
                "events_equal": receipt["events_equal"],
                "compositional_reduction_holds": receipt["compositional_reduction_holds"],
                "authority_verification": receipt["authority_verification"],
                "s3_authority_anchor_msg_id": receipt["s3_authority_anchor_msg_id"],
                "harness_file_sha256": receipt["harness_file_sha256"],
                "dry_exec": receipt.get("dry_exec", False),
                "out": str(write_path.as_posix()),
                "out_sha256": _sha_file(write_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
