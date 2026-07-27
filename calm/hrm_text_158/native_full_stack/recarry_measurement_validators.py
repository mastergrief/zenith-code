"""Pure validators for projected_moves re-carry measurement (PLAN_v17 S4a-R).

No filesystem, no CLI, no torch. Dependency direction: harness/evidence -> this module.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

AUTHORITY_VERIFICATION_AT_MINT = "gate_pending"

ADJACENCY_DISPOSITION = "UNCHANGED_PRODUCER_CONSTRAINT"
ADJACENT_CALLER_PATHS = (
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_loop_bridge.py:276",
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_selection_derisk.py:150",
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_selection_derisk.py:175",
    "calm/hrm_text_158/native_full_stack/oracle_screen_runner.py:430",
)

BRANCH_HOLDS = "BR-RECARRY-SPARSE-HOLDS-AB"
BRANCH_PARITY_FAIL = "BR-RECARRY-PARITY-FAIL-AB"
BRANCH_INFEASIBLE = "BR-RECARRY-INFEASIBLE"
BRANCH_INVALID = "BR-RECARRY-MEASUREMENT-INVALID"
BRANCH_PENDING = "BR-RECARRY-PENDING"

TERMINAL_ALLOWED = frozenset(
    {BRANCH_HOLDS, BRANCH_PARITY_FAIL, BRANCH_INFEASIBLE, BRANCH_INVALID}
)

S3_FORBIDDEN_ON_S4A = frozenset(
    {
        "s3_go_msg_id",
        "s3_authority_anchor_msg_id",
        "s3_terminal_receipt_msg_id",
        "s3_audit_receipt_sha256",
        "s3_gate1_freeze_msg_id",
        "s3_co_lead_pass_msg_id",
    }
)

def _require_exact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be exact int (bool rejected); got {type(value)}")
    return int(value)

def validate_events_equal_derived_not_hand_authored(
    *, dense_events: Mapping[Any, Any], fused_events: Mapping[Any, Any], events_equal: bool
) -> None:
    dense_i = {int(k): int(v) for k, v in dense_events.items()}
    fused_i = {int(k): int(v) for k, v in fused_events.items()}
    derived = dense_i == fused_i
    if bool(events_equal) != derived:
        raise ValueError(
            f"events_equal must be derived from dict equality; claimed={events_equal} derived={derived}"
        )

def validate_evidence_receipt_field_equality(
    evidence: Any, receipt: Mapping[str, Any]
) -> None:
    expected = evidence.to_receipt_fields()
    for key, value in expected.items():
        if key not in receipt:
            raise ValueError(f"receipt missing evidence-bound field: {key}")
        if receipt[key] != value:
            raise ValueError(
                f"evidence↔receipt inequality on {key}: evidence={value!r} receipt={receipt[key]!r}"
            )

def validate_live_repo_head_matches_claim(*, claimed: str, live: str) -> None:
    if claimed != live:
        raise ValueError(f"live repo HEAD mismatch: claimed={claimed} live={live}")

def validate_live_repo_head_matches_plan_pin(*, live: str, plan: Mapping[str, Any]) -> None:
    pinned = plan["execution_import_closure_freeze"]["pinned_at_plan_mint_head"]
    if live != pinned:
        raise ValueError(f"HEAD_mismatch: live={live} pinned={pinned}")

def validate_porcelain_truthful(
    *, claimed: Mapping[str, str], observed: Mapping[str, str]
) -> None:
    if dict(claimed) != dict(observed):
        raise ValueError(
            f"porcelain not truthful: claimed={dict(claimed)} observed={dict(observed)}"
        )

def validate_rank_spec_identity_matches_plan_pin(*, symbol: str, digest: str, plan: Mapping[str, Any]) -> None:
    frozen = plan["measurement_provenance"]["rank_spec_frozen"]
    if symbol != frozen["rank_spec_symbol"]:
        raise ValueError(f"rank_spec_drift: symbol {symbol} != {frozen['rank_spec_symbol']}")
    expected = frozen["constructed_spec_identity"]["to_live_dict_canonical_sha256"]
    if digest != expected:
        raise ValueError(f"rank_spec_drift: digest {digest} != {expected}")

def validate_adjacency_disposition_present(evidence: Any) -> None:
    if evidence.screen_acc_adjacency_disposition != ADJACENCY_DISPOSITION:
        raise ValueError("omit_adjacency_clause / disposition mismatch")
    if tuple(evidence.adjacent_caller_paths) != ADJACENT_CALLER_PATHS:
        raise ValueError("omit_adjacency_clause: adjacent_caller_paths mismatch")

def validate_both_evidence_sources_cited(evidence: Any) -> None:
    src = evidence.evidence_sources
    if "r1_msg_id" not in src or "r2_msg_id" not in src:
        raise ValueError("omit_r1_or_r2_evidence_source")

def validate_governing_claim_AB_only(evidence: Any) -> None:
    if evidence.governing_claim != "A+B_only":
        raise ValueError("claim_ABCD_from_A_only_execution / governing_claim")
    if list(evidence.parity_binding_set_ids) != ["A", "B"]:
        raise ValueError("claim_ABCD_from_A_only_execution")

def validate_governing_claim_AB_only_semantic(plan_or_receipt: Mapping[str, Any]) -> None:
    blob = json.dumps(plan_or_receipt)
    if re.search(r"governing[^\n]{0,80}A\+B\+C\+D", blob):
        raise ValueError("claim_ABCD_from_A_only_execution semantic")

def validate_compositional_reduction_bound(evidence: Any) -> None:
    if evidence.compositional_reduction_holds is not True:
        raise ValueError("omit_compositional_reduction / reduction_holds false")
    if not evidence.tsa_site_source_sha256 or not evidence.tsa_site_line_snippets_sha256:
        raise ValueError("omit_compositional_reduction: missing TSA hashes")

def validate_s3_authority_anchor_msg_id_present(mapping: Mapping[str, Any]) -> None:
    anchor = mapping.get("s3_authority_anchor_msg_id")
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError("missing_s3_authority_anchor_msg_id")

def validate_no_s3_go_msg_id_harness_input(
    *, argv: Sequence[str] | None = None, kwargs: Mapping[str, Any] | None = None
) -> None:
    if argv is not None and any(a == "--s3-go-msg-id" or str(a).startswith("--s3-go-msg-id=") for a in argv):
        raise ValueError("s3_go_msg_id_passed_as_harness_input")
    if kwargs is not None and "s3_go_msg_id" in kwargs:
        raise ValueError("s3_go_msg_id_passed_as_harness_input")

def validate_argv_flag_is_s3_authority_anchor_msg_id(argv: Sequence[str]) -> None:
    if len(argv) != 8:
        raise ValueError("argv_template_deviation: len!=8")
    if argv[6] == "--s3-go-msg-id":
        raise ValueError("stale_s3_go_msg_id_cli_flag")
    if argv[6] != "--s3-authority-anchor-msg-id":
        raise ValueError("validate_argv_flag_is_s3_authority_anchor_msg_id")

def validate_argv_index7_equals_authority_anchor_not_go(
    *, argv: Sequence[str], s3_authority_anchor_msg_id: str, s3_go_msg_id: str | None = None
) -> None:
    if len(argv) < 8:
        raise ValueError("argv_template_deviation")
    if argv[7] != s3_authority_anchor_msg_id:
        raise ValueError("wrong_s3_authority_anchor_msg_id")
    if s3_go_msg_id is not None and argv[7] == s3_go_msg_id:
        raise ValueError("argv_index7_equals_go_msg_id_self_reference")

def validate_argv_matches_frozen_template_after_instantiation(
    *, argv: Sequence[str], plan: Mapping[str, Any], s3_authority_anchor_msg_id: str
) -> None:
    template = list(plan["governing_runtime_command"]["argv_template"])
    expected = list(template)
    expected[7] = s3_authority_anchor_msg_id
    if list(argv) != expected:
        raise ValueError(f"argv_template_deviation: got={list(argv)!r} expected={expected!r}")
    validate_argv_flag_is_s3_authority_anchor_msg_id(argv)
    validate_argv_index7_equals_authority_anchor_not_go(
        argv=argv, s3_authority_anchor_msg_id=s3_authority_anchor_msg_id
    )

def validate_authority_verification_gate_pending_at_harness_mint(mapping: Mapping[str, Any]) -> None:
    if mapping.get("authority_verification") != AUTHORITY_VERIFICATION_AT_MINT:
        raise ValueError("authority_verification_not_gate_pending_at_mint")
    if mapping.get("s3_authorized_by_distinct_go_record") is True:
        raise ValueError("s3_authorized_true_from_go_id_presence")

def validate_harness_evidence_and_audit_exclude_s3_go_msg_id(
    *, evidence_mapping: Mapping[str, Any], audit_mapping: Mapping[str, Any]
) -> None:
    """KEY absence (not falsiness) on BOTH evidence and O_EXCL audit."""
    if "s3_go_msg_id" in evidence_mapping:
        raise ValueError("harness_evidence_or_audit_contains_s3_go_msg_id:evidence")
    if "s3_go_msg_id" in audit_mapping:
        raise ValueError("harness_evidence_or_audit_contains_s3_go_msg_id:audit")

def validate_no_caller_supplied_resolution_mapping(receipt: Mapping[str, Any]) -> None:
    if "caller_supplied_resolution_mapping" in receipt or receipt.get("resolution_path_B_records") is not None:
        raise ValueError("caller_supplied_resolution_mapping")

def validate_no_bind_by_id_evidence_path(receipt: Mapping[str, Any]) -> None:
    if receipt.get("bind_by_id_evidence_path") is True:
        raise ValueError("bind_by_id_evidence_path")
    if receipt.get("authority_verification") == "bind_by_id_validated":
        raise ValueError("bind_by_id_evidence_path")

def validate_s3_authority_anchor_and_go_ids_distinct(receipt: Mapping[str, Any]) -> None:
    if "s3_go_msg_id" not in receipt:
        return
    anchor = receipt.get("s3_authority_anchor_msg_id")
    go = receipt.get("s3_go_msg_id")
    if not isinstance(anchor, str) or not isinstance(go, str) or not anchor or not go or anchor == go:
        raise ValueError("anchor_and_go_ids_not_distinct")

def validate_s4a_has_no_s3_fields(receipt: Mapping[str, Any]) -> None:
    present = sorted(k for k in S3_FORBIDDEN_ON_S4A if k in receipt)
    if present:
        raise ValueError(f"s4a_contains_s3_fields: {present}")

def validate_s4b_binds_s3_ids(receipt: Mapping[str, Any]) -> None:
    required = [
        "s3_terminal_receipt_msg_id",
        "s3_authority_anchor_msg_id",
        "s3_go_msg_id",
        "s3_audit_receipt_sha256",
        "s3_gate1_freeze_msg_id",
        "s3_co_lead_pass_msg_id",
    ]
    missing = [k for k in required if k not in receipt or not receipt.get(k)]
    if missing:
        raise ValueError(f"s4b_missing_s3_review_ids: {missing}")
    validate_s3_authority_anchor_and_go_ids_distinct(receipt)

def validate_phase_dag_acyclic(plan: Mapping[str, Any]) -> list[str]:
    dag = plan["phase_dag_self_check"]
    nodes = list(dag["nodes"])
    edges = [tuple(e) for e in dag["edges"]]
    indeg = {n: 0 for n in nodes}
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        indeg[b] = indeg.get(b, 0) + 1
        indeg.setdefault(a, indeg.get(a, 0))
    queue = [n for n, d in indeg.items() if d == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in succ.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(order) != len(nodes):
        raise ValueError("phase DAG cyclic")
    return order

def validate_text_vs_dag_edge_consistency(plan: Mapping[str, Any]) -> None:
    rec = plan["phase_dag_self_check"]["text_vs_dag_edge_consistency"]
    if int(rec.get("weaker_disjuncts_remaining", 1)) != 0:
        raise ValueError("s4b_requires_prior_weaker_disjunct")
    for edge in rec["edges"]:
        if edge.get("weaker_disjunct_present") is True:
            raise ValueError("s4b_requires_prior_weaker_disjunct")
        if edge.get("consistency_ok") is not True:
            raise ValueError("text_vs_dag_edge_consistency failed")
        pre = " ".join(edge.get("B_operative_preconditions") or [])
        if re.search(r"\bin-flight\b|\bor pending\b", pre, re.I):
            raise ValueError("s4b_requires_prior_weaker_disjunct in edge text")

def validate_per_receipt_gate_rows_present(plan: Mapping[str, Any]) -> None:
    rows = plan["phase_dag_self_check"]["per_receipt_gate_rows"]
    for key in ("S4a_IMPLEMENT_receipt", "S3_terminal_audit_receipt", "S4b_FINALIZE_receipt"):
        if key not in rows:
            raise ValueError(f"missing per-receipt gate row: {key}")

def validate_no_prior_pass_covers_later_receipt(claim: Mapping[str, Any]) -> None:
    if claim.get("prior_receipt_PASS_covers_later_receipt") is True:
        raise ValueError("prior_receipt_PASS_covers_later_receipt")
    if claim.get("s4a_pass_covers_s3") is True:
        raise ValueError("s4a_pass_covers_s3")
    if claim.get("s4a_or_s3_pass_covers_s4b") is True:
        raise ValueError("s4a_or_s3_pass_covers_s4b")

def validate_s3_review_cycle_complete_before_s4b(receipt: Mapping[str, Any]) -> None:
    validate_s3_review_ids_resolved_author_thread_verdict(receipt)

def validate_s3_review_ids_resolved_author_thread_verdict(receipt: Mapping[str, Any]) -> None:
    """Fail-closed resolution of S3 review ids (not presence-only). Path A in-room only."""
    validate_no_caller_supplied_resolution_mapping(receipt)
    validate_no_bind_by_id_evidence_path(receipt)
    records = receipt.get("s3_review_id_resolved_records")
    if not isinstance(records, Mapping):
        raise ValueError("s4b_review_id_presence_without_resolution")
    freeze = records.get("s3_gate1_freeze")
    colead = records.get("s3_co_lead_pass")
    if not isinstance(freeze, Mapping) or not isinstance(colead, Mapping):
        raise ValueError("s4b_review_id_presence_without_resolution")
    freeze_id = receipt.get("s3_gate1_freeze_msg_id")
    colead_id = receipt.get("s3_co_lead_pass_msg_id")
    if not isinstance(freeze_id, str) or not freeze_id.strip() or not isinstance(colead_id, str) or not colead_id.strip():
        raise ValueError("s4b_missing_s3_review_ids")
    if freeze.get("id") != freeze_id:
        raise ValueError("s4b_s3_review_id_bind_mismatch_freeze")
    if colead.get("id") != colead_id:
        raise ValueError("s4b_s3_review_id_bind_mismatch_colead")
    if freeze.get("author") != "claude":
        raise ValueError("s4b_s3_gate1_id_wrong_author")
    if freeze.get("thread_matches_s3_receipt") is not True:
        raise ValueError("s4b_s3_gate1_id_wrong_thread")
    if freeze.get("freeze_semantics") is not True:
        raise ValueError("s4b_s3_gate1_id_wrong_thread")
    if colead.get("author") != "codex_co_lead":
        raise ValueError("s4b_s3_gate1_id_wrong_author")
    if colead.get("verdict") == "BLOCK":
        raise ValueError("s4b_s3_colead_id_BLOCK_verdict")
    if colead.get("kind") == "ack" or colead.get("is_ack") is True:
        raise ValueError("s4b_s3_colead_id_ack_kind")
    if colead.get("verdict") != "PASS":
        raise ValueError("s4b_s3_colead_id_BLOCK_verdict")
    if colead.get("threaded_to_same_s3_receipt") is not True:
        raise ValueError("s4b_s3_colead_id_unthreaded")
    if freeze.get("resolvable") is not True:
        raise ValueError("s4b_s3_review_id_unresolvable")
    if colead.get("resolvable") is not True:
        raise ValueError("s4b_s3_review_id_unresolvable")

def validate_s4b_binds_s3_review_ids(receipt: Mapping[str, Any]) -> None:
    validate_s4b_binds_s3_ids(receipt)
    validate_s3_review_ids_resolved_author_thread_verdict(receipt)

def classify_recarry_branch(
    *,
    events_equal: bool,
    compositional_reduction_holds: bool,
    measurement_valid: bool,
    infeasible: bool = False,
) -> str:
    if not measurement_valid:
        return BRANCH_INVALID
    if infeasible:
        return BRANCH_INFEASIBLE
    if not compositional_reduction_holds:
        return BRANCH_INVALID
    if events_equal:
        return BRANCH_HOLDS
    return BRANCH_PARITY_FAIL


def validate_dry_exec_out_not_canonical_governing_path(
    *,
    dry_exec_out: str,
    canonical_out: str,
) -> None:
    """Pure string equality on pre-normalized absolute paths (harness owns Path.resolve)."""
    if not isinstance(dry_exec_out, str) or not isinstance(canonical_out, str):
        raise ValueError("dry_exec_out_paths_must_be_normalized_strings")
    if dry_exec_out == canonical_out:
        raise ValueError("dry_exec_out_equals_canonical_governing_path")


def validate_dry_exec_not_with_formal_run_marker(
    *,
    dry_exec_out_present: bool,
    formal_run_marker: bool,
) -> None:
    if dry_exec_out_present is True and formal_run_marker is True:
        raise ValueError("dry_exec_co_occurs_with_formal_run_marker")


def validate_dry_exec_artifact_marks_dry_exec(receipt: Mapping[str, Any]) -> None:
    if "dry_exec" not in receipt:
        raise ValueError("dry_exec_artifact_missing_dry_exec_key")
    if receipt.get("dry_exec") is not True:
        raise ValueError("dry_exec_artifact_dry_exec_not_true")


def validate_formal_artifact_rejects_dry_exec_key(receipt: Mapping[str, Any]) -> None:
    if "dry_exec" in receipt:
        raise ValueError("formal_artifact_contains_dry_exec_key")


def validate_harness_file_sha256_present_and_matches_runtime(
    *,
    receipt: Mapping[str, Any],
    runtime_harness_file_sha256: str,
) -> None:
    if "harness_file_sha256" not in receipt:
        raise ValueError("harness_file_sha256_absent")
    observed = receipt.get("harness_file_sha256")
    if not isinstance(observed, str) or not observed.strip():
        raise ValueError("harness_file_sha256_absent")
    if observed != runtime_harness_file_sha256:
        raise ValueError("harness_file_sha256_runtime_mismatch")


def validate_harness_sha_binding_fields_present_and_equal(
    *,
    fields: Mapping[str, Any],
    required_keys: tuple[str, ...] | list[str],
) -> None:
    """Value-agnostic: all required keys present and equal (cycle-safe; no pinned harness sha)."""
    if not required_keys:
        raise ValueError("harness_sha_binding_required_keys_empty")
    values: list[str] = []
    for key in required_keys:
        if key not in fields:
            raise ValueError(f"harness_sha_binding_missing:{key}")
        val = fields.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"harness_sha_binding_missing:{key}")
        values.append(val)
    if len(set(values)) != 1:
        raise ValueError("harness_sha_binding_mismatch")


def _walk_plan_leaves(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_plan_leaves(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _walk_plan_leaves(value, f"{path}[{idx}]")
    else:
        yield path, obj


def _walk_governed_lifecycle_key_candidates(obj: Any, path: str = "$"):
    """Yield (path, value) for dict keys named note/active_after at CONTAINER-KEY level.

    Unlike _walk_plan_leaves, does NOT recurse into the value of a matched key — so a
    governed note={} / note=[] / note={...} presents as a candidate whose leaf name is
    note/active_after regardless of value type (scalar or container).
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}"
            if key in {"active_after", "note"}:
                yield child, value
                continue
            yield from _walk_governed_lifecycle_key_candidates(value, child)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _walk_governed_lifecycle_key_candidates(value, f"{path}[{idx}]")


def _is_metadata_surface(path: str) -> bool:
    lowered = path.lower()
    if "blocker_closures_" in lowered:
        return True
    if "_sweep_" in lowered:
        return True
    if "value_preview" in lowered:
        return True
    if "s3_closure_binding_v" in lowered:
        return True
    return False


def _path_developer_step_index(path: str) -> int | None:
    m = re.match(r"^\$\.DEVELOPER_STEPS\[(\d+)\]", path)
    if not m:
        return None
    return int(m.group(1))


def _container_status_for_path(plan: Mapping[str, Any], path: str) -> str | None:
    """Return status of the nearest status-bearing container for structural historical exclusion."""
    idx = _path_developer_step_index(path)
    if idx is not None:
        steps = plan.get("DEVELOPER_STEPS")
        if isinstance(steps, list) and 0 <= idx < len(steps) and isinstance(steps[idx], Mapping):
            return str(steps[idx].get("status") or "")
    # frozen_preconditions / frozen_write_surface blocks with status
    for key in ("frozen_preconditions", "frozen_write_surface"):
        block = plan.get(key)
        if not isinstance(block, Mapping):
            continue
        prefix = f"$.{key}."
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix) :]
        top = rest.split(".", 1)[0].split("[", 1)[0]
        sub = block.get(top)
        if isinstance(sub, Mapping) and "status" in sub:
            return str(sub.get("status") or "")
    return None


def _is_structurally_historical(plan: Mapping[str, Any], path: str) -> bool:
    if _is_metadata_surface(path):
        return True
    status = _container_status_for_path(plan, path)
    if status is None:
        return False
    if status.startswith("SUPERSEDED") or "SUPERSEDED" in status or status.startswith("COMPLETE_UNDER_"):
        return True
    if status == "PENDING_EXECUTION":
        # Never-executed superseded definitions may also be PENDING; treat non-ACTIVE as historical for citation.
        return True
    return False


def _is_anchor_go_citation_shape(text: str) -> bool:
    if not isinstance(text, str):
        return False
    has_plan = ("PLAN_v" in text and " sha" in text) or ("plan sha" in text.lower())
    has_tmpl = "argv_template_sha256" in text
    return has_plan and has_tmpl


def validate_plan_anchor_go_citations_carry_harness_sha(
    plan: Mapping[str, Any],
    *,
    registry_paths: Sequence[str] | None = None,
) -> list[str]:
    """Self-discovering: CURRENT operative citation leaves must cite current gen + harness_file_sha256.

    Historical exclusion is STRUCTURAL (superseded/historical container status), never by cited version.
    An old-gen citation on a current operative surface is itself a REJECT.
    """
    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    if not revision.startswith("v"):
        raise ValueError("plan_revision_missing")
    try:
        current_n = int(revision[1:])
    except ValueError as exc:
        raise ValueError("plan_revision_invalid") from exc
    discovered: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    for path, value in _walk_plan_leaves(plan):
        if _is_structurally_historical(plan, path):
            continue
        if not isinstance(value, str):
            continue
        if not _is_anchor_go_citation_shape(value):
            continue
        discovered.append(path)
        cited = [int(x) for x in re.findall(r"PLAN_v(\d+)", value)]
        if cited and max(cited) != current_n:
            stale.append(path)
            continue
        if f"PLAN_v{current_n}" not in value and "plan sha" not in value.lower():
            stale.append(path)
            continue
        if "harness_file_sha256" not in value:
            missing.append(path)
    if stale:
        raise ValueError("anchor_go_citation_stale_generation:" + ",".join(stale))
    if missing:
        raise ValueError(
            "anchor_go_citation_missing_harness_file_sha256:" + ",".join(missing)
        )
    if registry_paths is not None:
        registry_set = set(registry_paths)
        discovered_set = set(discovered)
        not_in_discovered = sorted(registry_set - discovered_set)
        if not_in_discovered:
            raise ValueError(
                "registry_paths_not_subset_of_discovered:" + ",".join(not_in_discovered)
            )
    return discovered


def _developer_step_is_s3_or_s4b(step: Mapping[str, Any]) -> bool:
    step_id = str(step.get("id") or "")
    return ("S3_execute" in step_id) or ("S4b_finalize" in step_id)


def validate_no_false_complete_s3_s4b_status(plan: Mapping[str, Any]) -> None:
    """S3/S4b definitions must never carry COMPLETE in a pre-S3 immutable plan (no terminal bypass)."""
    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    if not revision.startswith("v"):
        raise ValueError("plan_revision_missing")
    gen = revision
    steps = plan.get("DEVELOPER_STEPS")
    if not isinstance(steps, list):
        raise ValueError("DEVELOPER_STEPS_missing")
    bad: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if not _developer_step_is_s3_or_s4b(step):
            continue
        step_id = str(step.get("id") or "")
        status = str(step.get("status") or "")
        is_current_active = status == "ACTIVE" and (
            step_id.startswith(f"S0_{gen}_") or step_id.startswith(f"S0b_{gen}_")
        )
        if is_current_active:
            continue
        if "COMPLETE" in status:
            bad.append(f"{step_id}:{status}")
            continue
        if status in {"SUPERSEDED_DEFINITION_NEVER_EXECUTED", "PENDING_EXECUTION"}:
            continue
        if status == "ACTIVE":
            # Foreign ACTIVE handled by active-ids validator.
            continue
        bad.append(f"{step_id}:{status}")
    if bad:
        raise ValueError("false_complete_s3_s4b_status:" + ",".join(bad))


def validate_developer_steps_ids_unique(plan: Mapping[str, Any]) -> None:
    """Every DEVELOPER_STEPS[].id appears EXACTLY once (duplicate id REJECT regardless of statuses)."""
    steps = plan.get("DEVELOPER_STEPS")
    if not isinstance(steps, list):
        raise ValueError("DEVELOPER_STEPS_missing")
    seen: dict[str, int] = {}
    dupes: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("id") or "")
        if not step_id:
            raise ValueError("developer_step_id_empty")
        seen[step_id] = seen.get(step_id, 0) + 1
    for step_id, count in sorted(seen.items()):
        if count != 1:
            dupes.append(f"{step_id}:count={count}")
    if dupes:
        raise ValueError("developer_steps_id_not_unique:" + ",".join(dupes))


def validate_active_ids_exactly_current_generation(plan: Mapping[str, Any]) -> None:
    """ACTIVE set must equal the frozen exact current-id registry (both directions)."""
    registry = plan.get("active_id_registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("active_id_registry_missing")
    expected = {str(x) for x in registry}
    steps = plan.get("DEVELOPER_STEPS")
    if not isinstance(steps, list):
        raise ValueError("DEVELOPER_STEPS_missing")
    active_ids = {
        str(step.get("id"))
        for step in steps
        if isinstance(step, Mapping) and step.get("status") == "ACTIVE"
    }
    if active_ids != expected:
        missing = sorted(expected - active_ids)
        extra = sorted(active_ids - expected)
        raise ValueError(
            "active_ids_set_mismatch:"
            + f"missing={missing};extra={extra}"
        )


_DISPATCH_ID_RE = re.compile(r"dispatch\s+(\d{13}-[0-9a-f]+)", re.IGNORECASE)
_PROSPECTIVE_DUAL_RE = re.compile(
    r"this dispatch carries both",
    re.IGNORECASE,
)
_GEN_RE = re.compile(r"PLAN_v(\d+)|(?:^|[_\.])v(\d+)(?:[_\.]|$)", re.IGNORECASE)
_ROOM_MSG_ID_RE = re.compile(r"^\d{13}-[0-9a-f]+$")
_REGISTRY_KEY_SHAPE_RE = re.compile(r"^v\d+$")
_LIFECYCLE_LEGACY_PLAIN_STRING_MIN_GEN = 19
_LIFECYCLE_LEGACY_PLAIN_STRING_MAX_GEN = 32
_LIFECYCLE_MODES = frozenset({"fast_path", "full_plan_gate"})


def _lifecycle_authority_grammar(
    plan_generation: str, dispatch_id: str, mode: str = "fast_path"
) -> str:
    """Strict fail-closed chronology grammar (allowlist; only N and id vary).

    Exactly two templates: fast_path (historical byte-identical) and full_plan_gate.
    """
    n = plan_generation[1:] if plan_generation.startswith("v") else plan_generation
    if mode == "fast_path":
        return (
            f"PLAN_v{n} minted under converged fast-path +1 implement dispatch {dispatch_id}; "
            "dual accept applies to the frozen artifact review, not the mint."
        )
    if mode == "full_plan_gate":
        return (
            f"PLAN_v{n} minted under full-plan-gate dispatch {dispatch_id}; "
            "implementation authority arrives only via post-dual-accept persisted +1 implement."
        )
    raise ValueError(f"lifecycle_mode_unknown:{mode}")


def _plan_revision_generation(plan: Mapping[str, Any]) -> int:
    rev = plan.get("plan_revision")
    if not isinstance(rev, str) or not rev.startswith("v"):
        raise ValueError("plan_revision_missing")
    try:
        n = int(rev[1:])
    except ValueError as exc:
        raise ValueError("plan_revision_invalid") from exc
    if f"v{n}" != rev:
        # reject aliases like v031 at plan_revision level too
        raise ValueError("plan_revision_not_canonical")
    return n


def _lifecycle_path_generation(path: str) -> str | None:
    m = (
        re.search(r"P4_write_surface_v(\d+)(?:_|$)", path)
        or re.search(r"active_phase_v(\d+)", path)
        or re.search(r"bounded_v(\d+)_", path)
        or re.search(r"compatibility_v(\d+)", path)
        or re.search(r"write_surface_v(\d+)", path)
        or re.search(r"post_plus1_implement_bounded_v(\d+)_", path)
        or re.search(r"post_plus1_implement_bounded_compatibility_v(\d+)", path)
    )
    return f"v{m.group(1)}" if m else None


def _canonical_registry_key(key: str) -> str | None:
    if not isinstance(key, str) or not _REGISTRY_KEY_SHAPE_RE.fullmatch(key):
        return None
    digits = key[1:]
    try:
        n = int(digits)
    except ValueError:
        return None
    canonical = f"v{n}"
    return canonical if canonical == key else None


def _parse_lifecycle_registry_entry(gen: str, entry: Any) -> tuple[str, str]:
    """Return (dispatch_id, mode). Fail-closed per frozen schema."""
    # gen like v25 → int
    try:
        gen_n = int(str(gen)[1:])
    except Exception as exc:
        raise ValueError(f"lifecycle_registry_key_shape:{gen}") from exc

    if isinstance(entry, str):
        if not (
            _LIFECYCLE_LEGACY_PLAIN_STRING_MIN_GEN
            <= gen_n
            <= _LIFECYCLE_LEGACY_PLAIN_STRING_MAX_GEN
        ):
            raise ValueError(f"lifecycle_registry_legacy_string_beyond_v32:{gen}")
        if not _ROOM_MSG_ID_RE.fullmatch(entry):
            raise ValueError(f"lifecycle_registry_id_shape:{gen}")
        return entry, "fast_path"

    if not isinstance(entry, Mapping):
        raise ValueError(f"lifecycle_registry_entry_type:{gen}")

    keys = set(entry.keys())
    allowed = {"id", "mode"}
    extra = keys - allowed
    if extra:
        raise ValueError(f"lifecycle_registry_extra_key:{gen}:{sorted(extra)}")
    if "id" not in entry:
        raise ValueError(f"lifecycle_registry_missing_id:{gen}")
    if "mode" not in entry:
        raise ValueError(f"lifecycle_registry_missing_mode:{gen}")

    rid = entry.get("id")
    mode = entry.get("mode")
    if not isinstance(rid, str):
        raise ValueError(f"lifecycle_registry_id_type:{gen}")
    if rid == "":
        raise ValueError(f"lifecycle_registry_id_empty:{gen}")
    if not _ROOM_MSG_ID_RE.fullmatch(rid):
        raise ValueError(f"lifecycle_registry_id_shape:{gen}")
    if not isinstance(mode, str) or mode == "":
        raise ValueError(f"lifecycle_registry_mode_type:{gen}")
    if mode not in _LIFECYCLE_MODES:
        raise ValueError(f"lifecycle_mode_unknown:{mode}")
    return rid, mode


def _collect_governed_lifecycle_generations(plan: Mapping[str, Any]) -> set[str]:
    gens: set[str] = set()
    for path, _value in _walk_governed_lifecycle_key_candidates(plan):
        if not any(
            tok in path
            for tok in (
                "lifecycle",
                "write_surface",
                "P4_",
                "frozen_write_surface",
                "frozen_preconditions",
            )
        ):
            continue
        if _is_metadata_surface(path):
            continue
        gen = _lifecycle_path_generation(path)
        if gen is not None:
            gens.add(gen)
    return gens


def _validate_lifecycle_registry_structure(plan: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Registry-wide fail-closed parse + canonical membership BEFORE leaf matching.

    Returns gen → (dispatch_id, mode).
    """
    registry = plan.get("lifecycle_dispatch_id_registry")
    if not isinstance(registry, Mapping) or not registry:
        raise ValueError("lifecycle_dispatch_id_registry_missing")

    n = _plan_revision_generation(plan)
    expected_keys = {f"v{i}" for i in range(_LIFECYCLE_LEGACY_PLAIN_STRING_MIN_GEN, n + 1)}

    parsed: dict[str, tuple[str, str]] = {}
    observed_keys: set[str] = set()

    for key, entry in registry.items():
        if not isinstance(key, str) or not _REGISTRY_KEY_SHAPE_RE.fullmatch(key):
            raise ValueError(f"lifecycle_registry_key_shape:{key!r}")
        canonical = _canonical_registry_key(key)
        if canonical is None:
            raise ValueError(f"lifecycle_registry_key_not_canonical:{key}")
        observed_keys.add(canonical)
        parsed[canonical] = _parse_lifecycle_registry_entry(canonical, entry)

    missing = sorted(expected_keys - observed_keys, key=lambda g: int(g[1:]))
    extra = sorted(observed_keys - expected_keys, key=lambda g: int(g[1:]))
    if missing:
        raise ValueError(f"lifecycle_registry_generation_missing:{missing}")
    if extra:
        raise ValueError(f"lifecycle_registry_generation_extra:{extra}")

    governed = _collect_governed_lifecycle_generations(plan)
    if governed != observed_keys:
        # Keep a single error class; fixtures probe each direction via mutation shape.
        raise ValueError(
            "lifecycle_registry_governed_set_mismatch:"
            + f"registry_minus_governed={sorted(observed_keys - governed, key=lambda g: int(g[1:]))};"
            + f"governed_minus_registry={sorted(governed - observed_keys, key=lambda g: int(g[1:]))}"
        )
    return parsed


def validate_lifecycle_authority_cites_dispatch(plan: Mapping[str, Any]) -> None:
    """Lifecycle chronology leaves must MATCH the mode-aware strict grammar exactly.

    F3: candidate selection is structural-path-only (version-addressable active_after/note
    under write-surface namespaces). Applies to EVERY such leaf (current AND historical);
    historical leaves grammar-bind to path-derived registry[vN] (not current generation).
    Fail-closed allowlist: unrecognized / synonym / extra free-text → REJECT.

    F4: registry-wide structure/membership validation runs BEFORE leaf matching.
    Mode-carrying registry entries select the template; plain-string v19–v32 default fast_path.
    """
    parsed_registry = _validate_lifecycle_registry_structure(plan)
    bad: list[str] = []

    for path, value in _walk_governed_lifecycle_key_candidates(plan):
        if not any(
            tok in path
            for tok in (
                "lifecycle",
                "write_surface",
                "P4_",
                "frozen_write_surface",
                "frozen_preconditions",
            )
        ):
            continue
        if _is_metadata_surface(path):
            continue

        gen = _lifecycle_path_generation(path)
        if gen is None:
            continue
        if not isinstance(value, str):
            bad.append(f"lifecycle_leaf_not_string:{path}:type={type(value).__name__}")
            continue

        if gen not in parsed_registry:
            bad.append(f"dispatch_registry_missing_gen:{path}:{gen}")
            continue
        expected_id, mode = parsed_registry[gen]
        expected = _lifecycle_authority_grammar(gen, expected_id, mode)
        if value == expected:
            continue

        # Mode/text mismatch classification when the other template matches.
        other_mode = "full_plan_gate" if mode == "fast_path" else "fast_path"
        other_expected = _lifecycle_authority_grammar(gen, expected_id, other_mode)
        if value == other_expected:
            bad.append(f"lifecycle_mode_text_mismatch:{path}:declared={mode}")
            continue

        if "dual accept and +1 implement are both carried" in value.lower():
            bad.append(f"lifecycle_grammar_synonym:{path}")
        elif value.startswith(expected):
            bad.append(f"lifecycle_grammar_extra_text:{path}")
        else:
            named = re.findall(r"PLAN_v(\d+)", value)
            found_ids = _DISPATCH_ID_RE.findall(value)
            if named and f"v{named[0]}" != gen:
                bad.append(
                    f"plan_generation_text_mismatch:{path}:expected={gen}:found=v{named[0]}"
                )
            elif found_ids and found_ids[0].lower() != expected_id.lower():
                bad.append(
                    f"dispatch_id_mismatch:{path}:gen={gen}:expected={expected_id}:found={found_ids[0]}"
                )
            else:
                bad.append(f"lifecycle_grammar_mismatch:{path}")
    if bad:
        raise ValueError("lifecycle_authority_cite_invalid:" + ",".join(bad))



def _action_class_lists(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    classes = ("UPDATE", "CREATE", "CREATE_THEN_PRESERVE", "PRESERVE_BYTES", "VALIDATE_ONLY")
    out: dict[str, list[str]] = {}
    for cls in classes:
        items = manifest.get(cls) or []
        if not isinstance(items, list):
            continue
        out[cls] = [str(p) for p in items]
    return out


def _path_to_action_class(manifest: Mapping[str, Any], *, surface: str) -> dict[str, str]:
    """Validate one operative manifest surface; return path → single action class."""
    lists = _action_class_lists(manifest)
    path_to_classes: dict[str, list[str]] = {}
    for cls, str_items in lists.items():
        if len(str_items) != len(set(str_items)):
            dups = sorted({p for p in str_items if str_items.count(p) > 1})
            raise ValueError(
                f"action_class_within_list_duplicate:{surface}:{cls}:" + ",".join(dups)
            )
        for p in str_items:
            path_to_classes.setdefault(p, []).append(cls)
    conflicts = {p: cs for p, cs in path_to_classes.items() if len(set(cs)) > 1}
    if conflicts:
        raise ValueError(
            f"action_class_conflict:{surface}:" + ",".join(sorted(conflicts))
        )
    create = set(lists.get("CREATE", []))
    preserve = set(lists.get("PRESERVE_BYTES", []))
    ctp = set(lists.get("CREATE_THEN_PRESERVE", []))
    if create & preserve:
        raise ValueError(
            f"action_class_dual_create_preserve:{surface}:"
            + ",".join(sorted(create & preserve))
        )
    if (ctp & create) or (ctp & preserve):
        raise ValueError(
            f"action_class_conflict:{surface}:"
            + ",".join(sorted((ctp & create) | (ctp & preserve)))
        )
    # IN↔OUT conflict for scoped file paths
    out_items = manifest.get("OUT") or []
    if isinstance(out_items, list):
        out_set = {str(p) for p in out_items}
        in_set = set(path_to_classes)
        both = sorted(in_set & out_set)
        if both:
            raise ValueError(
                f"action_class_in_out_conflict:{surface}:" + ",".join(both)
            )
    return {p: cs[0] for p, cs in path_to_classes.items()}


_REQUIRED_ACTION_CLASS_KEYS = (
    "UPDATE",
    "CREATE",
    "CREATE_THEN_PRESERVE",
    "PRESERVE_BYTES",
    "VALIDATE_ONLY",
    "OUT",
)


def required_operative_surfaces(plan_revision: str) -> list[str]:
    """Explicit required operative write-manifest surfaces for generation N (fail-closed).

    Discovery of the required set MUST NOT depend on presence of action-class fields.
    """
    if not isinstance(plan_revision, str) or not plan_revision.startswith("v"):
        raise ValueError("plan_revision_missing")
    try:
        n = int(plan_revision[1:])
    except ValueError as exc:
        raise ValueError("plan_revision_invalid") from exc
    return [
        "active_write_manifest",
        f"frozen_preconditions.P4_write_surface_v{n}_lifecycle.manifest",
        f"frozen_preconditions.P4_write_surface_active_phase_v{n}",
        "frozen_write_surface.this_plan_phase",
        f"frozen_write_surface.post_plus1_implement_bounded_v{n}_correction",
    ]


def _resolve_dotted_mapping(
    root: Mapping[str, Any], dotted: str
) -> Mapping[str, Any] | None:
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, Mapping) else None


def _require_absolute_manifest_schema(
    manifest: Mapping[str, Any],
    *,
    surface: str,
    canonical: bool = False,
) -> None:
    """Absolute list-presence schema before cross-surface equality (F1)."""
    for cls in _REQUIRED_ACTION_CLASS_KEYS:
        if cls not in manifest:
            if canonical:
                raise ValueError(f"canonical_manifest_missing_required_list:{cls}")
            raise ValueError(f"required_list_absent:{surface}:{cls}")
        if not isinstance(manifest[cls], list):
            if canonical:
                raise ValueError(f"required_list_not_list:{cls}")
            raise ValueError(f"required_list_not_list:{surface}:{cls}")


def _iter_operative_action_manifests(
    plan: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Required operative surfaces (registry) plus additive ACTIVE surfaces with class fields."""
    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    required = required_operative_surfaces(revision)
    found: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()

    for name in required:
        block = _resolve_dotted_mapping(plan, name)
        if block is None:
            raise ValueError(f"required_operative_surface_absent:{name}")
        found.append((name, block))
        seen.add(name)

    # Additive: extra ACTIVE surfaces with class fields still validated if present.
    fp = plan.get("frozen_preconditions")
    if isinstance(fp, Mapping):
        for key, block in fp.items():
            if not isinstance(block, Mapping) or block.get("status") != "ACTIVE":
                continue
            if isinstance(block.get("manifest"), Mapping):
                name = f"frozen_preconditions.{key}.manifest"
                if name not in seen:
                    found.append((name, block["manifest"]))
                    seen.add(name)
                continue
            if any(k in block for k in _REQUIRED_ACTION_CLASS_KEYS):
                name = f"frozen_preconditions.{key}"
                if name not in seen:
                    found.append((name, block))
                    seen.add(name)
    fws = plan.get("frozen_write_surface")
    if isinstance(fws, Mapping):
        for key, block in fws.items():
            if not isinstance(block, Mapping):
                continue
            if key == "this_plan_phase":
                name = "frozen_write_surface.this_plan_phase"
                if name not in seen and any(k in block for k in _REQUIRED_ACTION_CLASS_KEYS):
                    found.append((name, block))
                    seen.add(name)
                continue
            if block.get("status") != "ACTIVE":
                continue
            if any(k in block for k in _REQUIRED_ACTION_CLASS_KEYS):
                name = f"frozen_write_surface.{key}"
                if name not in seen:
                    found.append((name, block))
                    seen.add(name)
    return found


def _canonical_action_class_map(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    """Normalized class → sorted-unique path list for exact-equality compare.

    R1/F1: OUT joins exactness at equal rank; absence is never a comparable sentinel.
    Caller must run _require_absolute_manifest_schema first.
    """
    out: dict[str, list[str]] = {}
    for cls in _REQUIRED_ACTION_CLASS_KEYS:
        items = manifest[cls]
        out[cls] = sorted({str(p) for p in items})
    return out


def validate_single_action_class_per_path(plan: Mapping[str, Any]) -> None:
    """Exact-equality across ALL operative manifests vs canonical active_write_manifest.

    R1/F1: absolute required-list schema then OUT joins exactness;
    missing/extra action path REJECT; sole CREATE_THEN_PRESERVE plan path must equal
    current PLAN_v{N}.json; CREATE receipt identity must equal current IMPLEMENT_receipt;
    IN↔OUT conflict REJECT. F2: required surfaces come from required_operative_surfaces.
    """
    surfaces = _iter_operative_action_manifests(plan)
    if not surfaces:
        raise ValueError("active_write_manifest_missing")
    awm = plan.get("active_write_manifest")
    if not isinstance(awm, Mapping):
        raise ValueError("active_write_manifest_missing")
    _require_absolute_manifest_schema(awm, surface="active_write_manifest", canonical=True)

    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    if not revision.startswith("v"):
        raise ValueError("plan_revision_missing")
    try:
        current_n = int(revision[1:])
    except ValueError as exc:
        raise ValueError("plan_revision_invalid") from exc
    expected_plan = (
        "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_"
        f"PLAN_v{current_n}.json"
    )
    # receipt_vK where K = N - 14 (lineage: v29→v15, v30→v16, …)
    expected_receipt = (
        "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_"
        f"IMPLEMENT_receipt_v{current_n - 14}.json"
    )

    # Per-surface absolute schema + internal consistency + collect maps
    canonical = _canonical_action_class_map(awm)
    for name, manifest in surfaces:
        if name != "active_write_manifest":
            _require_absolute_manifest_schema(manifest, surface=name, canonical=False)
        else:
            # already schema-checked as canonical above
            pass
        _path_to_action_class(manifest, surface=name)
        got = _canonical_action_class_map(manifest)
        if got != canonical:
            missing: list[str] = []
            extra: list[str] = []
            for cls in canonical:
                cset, gset = set(canonical[cls]), set(got[cls])
                for p in sorted(cset - gset):
                    missing.append(f"{cls}:{p}")
                for p in sorted(gset - cset):
                    extra.append(f"{cls}:{p}")
            parts = []
            if missing:
                parts.append("missing=" + ",".join(missing))
            if extra:
                parts.append("extra=" + ",".join(extra))
            raise ValueError(
                f"action_class_cross_surface_not_exact:{name}:" + ";".join(parts)
            )

        # Artifact identity on every surface
        ctp = [str(p) for p in (manifest.get("CREATE_THEN_PRESERVE") or [])]
        plan_paths = [p for p in ctp if "PLAN_v" in p]
        if len(plan_paths) != 1:
            raise ValueError(f"plan_create_then_preserve_count:{name}:{len(plan_paths)}")
        if plan_paths[0] != expected_plan:
            raise ValueError(
                f"plan_artifact_identity_mismatch:{name}:expected={expected_plan}:found={plan_paths[0]}"
            )
        create = [str(p) for p in (manifest.get("CREATE") or [])]
        receipt_paths = [p for p in create if "IMPLEMENT_receipt_v" in p]
        if len(receipt_paths) != 1:
            raise ValueError(
                f"receipt_create_count:{name}:{len(receipt_paths)}"
            )
        if receipt_paths[0] != expected_receipt:
            raise ValueError(
                f"receipt_artifact_identity_mismatch:{name}:expected={expected_receipt}:found={receipt_paths[0]}"
            )


def validate_supersedes_schema(plan: Mapping[str, Any]) -> None:
    """supersedes must be a dict with plans[] covering contiguous prior generations 1..(current-1)."""
    supersedes = plan.get("supersedes")
    if not isinstance(supersedes, Mapping):
        raise ValueError("supersedes_not_dict")
    if "plans" not in supersedes or "note" not in supersedes:
        raise ValueError("supersedes_missing_required_keys")
    plans = supersedes.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("supersedes_plans_empty")
    revision = str(plan.get("plan_revision") or plan.get("packet_revision") or "")
    if not revision.startswith("v"):
        raise ValueError("plan_revision_missing")
    try:
        current_n = int(revision[1:])
    except ValueError as exc:
        raise ValueError("plan_revision_invalid") from exc
    if current_n < 2:
        raise ValueError("plan_revision_too_small_for_supersedes")
    seen_revs: list[int] = []
    for i, entry in enumerate(plans):
        if not isinstance(entry, Mapping):
            raise ValueError(f"supersedes_entry_not_dict:{i}")
        if "rev" not in entry or "sha256" not in entry:
            raise ValueError(f"supersedes_entry_missing_keys:{i}")
        rev = str(entry.get("rev"))
        if not rev.startswith("v"):
            raise ValueError(f"supersedes_entry_rev_invalid:{i}:{rev}")
        try:
            n = int(rev[1:])
        except ValueError as exc:
            raise ValueError(f"supersedes_entry_rev_invalid:{i}:{rev}") from exc
        if n >= current_n:
            raise ValueError(f"supersedes_entry_beyond_or_equal_current:{i}:v{n}")
        if n < 1:
            raise ValueError(f"supersedes_entry_rev_invalid:{i}:{rev}")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"supersedes_entry_sha_invalid:{i}")
        seen_revs.append(n)
    if seen_revs != sorted(seen_revs):
        raise ValueError("supersedes_revs_not_monotonic")
    if len(seen_revs) != len(set(seen_revs)):
        raise ValueError("supersedes_revs_duplicate")
    expected = list(range(1, current_n))
    if sorted(seen_revs) != expected:
        missing = sorted(set(expected) - set(seen_revs))
        extra = sorted(set(seen_revs) - set(expected))
        raise ValueError(
            "supersedes_revs_not_contiguous:"
            + f"missing={missing};extra={extra};expected=1..{current_n - 1}"
        )

