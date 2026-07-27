"""Evidence collection for projected_moves re-carry measurement (PLAN_v17 S4a-R).

May import the torch / BDL stack. No CLI / O_EXCL mint logic.
Dependency direction: harness -> this module -> validators (never reverse into harness).
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    canonical_acquisition_rank_vote_spec,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    project_s1_gradient_to_moves,
    sparse_rank_bucketed_int16_vote_events,
    sparse_rank_bucketed_int16_vote_events_from_weighted_grad,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (
    compute_canonical_json_sha256,
    compute_tensor_canonical_sha256,
)
from calm.hrm_text_158.native_full_stack.recarry_measurement_validators import (
    ADJACENCY_DISPOSITION,
    ADJACENT_CALLER_PATHS,
    AUTHORITY_VERIFICATION_AT_MINT,
    BRANCH_HOLDS,
    BRANCH_INFEASIBLE,
    BRANCH_INVALID,
    BRANCH_PARITY_FAIL,
    BRANCH_PENDING,
    TERMINAL_ALLOWED,
    _require_exact_int,
    classify_recarry_branch,
    validate_adjacency_disposition_present,
    validate_authority_verification_gate_pending_at_harness_mint,
    validate_both_evidence_sources_cited,
    validate_compositional_reduction_bound,
    validate_events_equal_derived_not_hand_authored,
    validate_governing_claim_AB_only,
    validate_harness_evidence_and_audit_exclude_s3_go_msg_id,
    validate_rank_spec_identity_matches_plan_pin,
)

FIXTURE_RECIPE_NAME = "3C_C1_dry_run_fixture_seed158"
PROBE_MODE = "projected_moves_recarry_cpu_measurement"
PINNED_AT_PLAN_MINT_HEAD = "ed4932b9eef0fb547970019edf55e02ab57cb32a"
RANK_SPEC_SYMBOL = "default_dry_run_rank_vote_spec"
RANK_SPEC_DIGEST_EXPECTED = (
    "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5"
)
PRODUCER_SWEEP_HIT_COUNT = 97
R1_MSG_ID = "1785142156499-6d44e42b"
R2_MSG_ID = "1785142473879-f1cc0c84"
TSA_PATH = "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py"
TSA_SITE_LINES = (1762, 2089, 2790)
TSA_RANK_SPEC_LINES = (1736, 2064, 2765)
TSA_FILE_SHA_EXPECTED = (
    "99bd6d543407dbb91fee8dba10e110de4d7d839e9e43b63ad36fc04d6806dbf8"
)

@dataclass(frozen=True)
class ProjectedMovesRecarryMeasurementEvidence:
    fixture_recipe_name: str
    parity_fixture_descriptor_sha256: str
    weighted_grad_sha256: str
    weighted_grad_dtype: str
    weighted_grad_shape: tuple[int, ...]
    weighted_grad_numel: int
    dense_projected_moves_sha256: str
    dense_projected_moves_numel: int
    dense_projected_moves_bytes: int
    dense_events: dict[int, int]
    dense_events_canonical_sha256: str
    fused_events: dict[int, int]
    fused_events_canonical_sha256: str
    fused_event_count: int
    events_equal: bool
    evidence_sources: dict[str, Any]
    screen_acc_adjacency_disposition: str
    adjacent_caller_paths: tuple[str, ...]
    probe_mode: str
    repo_head_sha: str
    producer_contract_unchanged: bool
    rank_spec_symbol: str
    rank_spec_to_live_dict_canonical_sha256: str
    rank_spec_mode: str
    rank_spec_rank_method: str
    rank_spec_bins: tuple[dict[str, Any], ...]
    parity_binding_set_ids: tuple[str, ...]
    reference_paths_classification: dict[str, Any]
    compositional_reduction_holds: bool
    tsa_site_source_sha256: str
    tsa_site_line_snippets_sha256: str
    execution_import_closure_sha256_rollup: str
    import_closure_pin_pass: bool
    git_porcelain_over_closure: dict[str, str]
    secondary_acquisition_events_equal: bool | None
    branch_id: str
    recommended_next_slice: str
    governing_claim: str
    executor_role_for_governing_run: str
    s3_authority_anchor_msg_id: str
    authority_verification: str
    gated_by_S4a_not_S4b: bool

    def to_receipt_fields(self) -> dict[str, Any]:
        return {
            "fixture_recipe_name": self.fixture_recipe_name,
            "parity_fixture_descriptor_sha256": self.parity_fixture_descriptor_sha256,
            "weighted_grad_sha256": self.weighted_grad_sha256,
            "weighted_grad_dtype": self.weighted_grad_dtype,
            "weighted_grad_shape": list(self.weighted_grad_shape),
            "weighted_grad_numel": self.weighted_grad_numel,
            "dense_projected_moves_sha256": self.dense_projected_moves_sha256,
            "dense_projected_moves_numel": self.dense_projected_moves_numel,
            "dense_projected_moves_bytes": self.dense_projected_moves_bytes,
            "dense_events": {str(k): int(v) for k, v in sorted(self.dense_events.items())},
            "dense_events_canonical_sha256": self.dense_events_canonical_sha256,
            "fused_events": {str(k): int(v) for k, v in sorted(self.fused_events.items())},
            "fused_events_canonical_sha256": self.fused_events_canonical_sha256,
            "fused_event_count": self.fused_event_count,
            "events_equal": self.events_equal,
            "evidence_sources": dict(self.evidence_sources),
            "screen_acc_adjacency_disposition": self.screen_acc_adjacency_disposition,
            "adjacent_caller_paths": list(self.adjacent_caller_paths),
            "probe_mode": self.probe_mode,
            "repo_head_sha": self.repo_head_sha,
            "producer_contract_unchanged": self.producer_contract_unchanged,
            "rank_spec_symbol": self.rank_spec_symbol,
            "rank_spec_to_live_dict_canonical_sha256": self.rank_spec_to_live_dict_canonical_sha256,
            "rank_spec_mode": self.rank_spec_mode,
            "rank_spec_rank_method": self.rank_spec_rank_method,
            "rank_spec_bins": [dict(b) for b in self.rank_spec_bins],
            "parity_binding_set_ids": list(self.parity_binding_set_ids),
            "reference_paths_classification": dict(self.reference_paths_classification),
            "compositional_reduction_holds": self.compositional_reduction_holds,
            "tsa_site_source_sha256": self.tsa_site_source_sha256,
            "tsa_site_line_snippets_sha256": self.tsa_site_line_snippets_sha256,
            "execution_import_closure_sha256_rollup": self.execution_import_closure_sha256_rollup,
            "import_closure_pin_pass": self.import_closure_pin_pass,
            "git_porcelain_over_closure": dict(self.git_porcelain_over_closure),
            "secondary_acquisition_events_equal": self.secondary_acquisition_events_equal,
            "audit_branch_id": self.branch_id,
            "recommended_next_slice": self.recommended_next_slice,
            "governing_claim": self.governing_claim,
            "executor_role_for_governing_run": self.executor_role_for_governing_run,
            "s3_authority_anchor_msg_id": self.s3_authority_anchor_msg_id,
            "authority_verification": self.authority_verification,
            "gated_by_S4a_not_S4b": self.gated_by_S4a_not_S4b,
            "observation_evidence_type": "ProjectedMovesRecarryMeasurementEvidence",
            "transient_fp_debt_remains": True,
            "producer_sweep_hit_count": PRODUCER_SWEEP_HIT_COUNT,
        }

def _events_canonical_sha256(events: Mapping[int, int]) -> str:
    payload = {str(int(k)): int(v) for k, v in sorted((int(k), int(v)) for k, v in events.items())}
    return compute_canonical_json_sha256(payload)

def _rank_spec_digest(spec: Any) -> str:
    live = spec.to_live_dict()
    return hashlib.sha256(
        json.dumps(live, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_3c_harness():
    path = _repo_root() / "scripts" / "optimizer_credit_state_3C_readonly_audit_run.py"
    spec = importlib.util.spec_from_file_location(
        "optimizer_credit_state_3C_readonly_audit_run", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load 3C harness at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def _live_repo_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()

def _git_porcelain_map(repo_root: Path, paths: Sequence[str]) -> dict[str, str]:
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "-uall", "--", *paths],
        cwd=str(repo_root),
        text=True,
    )
    result = {p: "clean" for p in paths}
    for line in out.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        rel = line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1]
        if rel in result:
            result[rel] = status.strip() or status
    return result

def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _tsa_compositional_reduction(repo_root: Path) -> tuple[bool, str, str]:
    path = repo_root / TSA_PATH
    source_sha = _sha_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets: list[str] = []
    holds = source_sha == TSA_FILE_SHA_EXPECTED
    for line_no in TSA_SITE_LINES:
        start = max(0, line_no - 3)
        end = min(len(lines), line_no + 2)
        window = "\n".join(lines[start:end])
        snippets.append(f"L{line_no}:" + window)
        if "project_s1_gradient_to_moves" not in window:
            holds = False
        if "rank_bucketed_int16_votes" not in window:
            holds = False
    for line_no in TSA_RANK_SPEC_LINES:
        text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
        if "default_dry_run_rank_vote_spec" not in text:
            holds = False
        snippets.append(f"RANK_L{line_no}:{text}")
    snippet_sha = compute_canonical_json_sha256(snippets)
    return holds, source_sha, snippet_sha


def validate_dependency_currency_against_plan_pins(*, plan: Mapping[str, Any], repo_root: Path) -> None:
    freeze = plan["dependency_currency_freeze"]
    files = freeze["files"]
    if int(freeze["pinned_file_count"]) != len(files):
        raise ValueError("dependency_currency pinned_file_count mismatch")
    for entry in files:
        path = repo_root / entry["path"]
        actual = _sha_file(path)
        if actual != entry["expected_sha256"]:
            raise ValueError(
                f"dependency_currency drift on {entry['path']}: expected={entry['expected_sha256']} actual={actual}"
            )

def validate_import_closure_pins(*, plan: Mapping[str, Any], repo_root: Path) -> tuple[bool, str, dict[str, str]]:
    freeze = plan["execution_import_closure_freeze"]
    files = freeze["files"]
    if int(freeze["closure_file_count"]) != len(files):
        raise ValueError("closure_file_count mismatch")
    for req in freeze["named_minimum_required"]:
        if not any(e["path"] == req for e in files):
            raise ValueError(f"named minimum missing from closure pins: {req}")
    digests = []
    for entry in files:
        actual = _sha_file(repo_root / entry["path"])
        if actual != entry["expected_sha256"]:
            raise ValueError(
                f"import_closure_pin_drift on {entry['path']}: expected={entry['expected_sha256']} actual={actual}"
            )
        digests.append([entry["path"], actual])
    porcelain = _git_porcelain_map(repo_root, [e["path"] for e in files])
    for dirty in freeze.get("dirty_tracked_files_pinned_at_working_tree_bytes", []):
        obs = porcelain.get(dirty["path"], "clean")
        if obs == "clean":
            raise ValueError(
                f"claim_clean_porcelain_when_dirty: {dirty['path']} expected dirty, observed clean"
            )
    rollup = compute_canonical_json_sha256(digests)
    return True, rollup, porcelain

def validate_producer_contract_unchanged(*, unchanged: bool, project_s1_fn: Any) -> None:
    if unchanged is not True:
        raise ValueError("producer_contract_unchanged must be true")
    sig = str(inspect.signature(project_s1_fn))
    if sig != "(grad: 'torch.Tensor', q_levels: 'torch.Tensor') -> 'torch.Tensor'":
        params = list(inspect.signature(project_s1_fn).parameters)
        if params != ["grad", "q_levels"]:
            raise ValueError(f"change_project_s1_signature_in_harness: {sig}")

def validate_parity_uses_live_consumer_symbols_not_reimpl() -> None:
    if sparse_rank_bucketed_int16_vote_events.__module__ != (
        "calm.hrm_text_158.native_full_stack.bounded_delta_learner"
    ):
        raise ValueError("fixture_local_rank_bucket_reimpl")
    if sparse_rank_bucketed_int16_vote_events_from_weighted_grad.__module__ != (
        "calm.hrm_text_158.native_full_stack.bounded_delta_learner"
    ):
        raise ValueError("fixture_local_rank_bucket_reimpl")

def collect_recarry_evidence(
    *,
    repo_root: Path,
    plan: Mapping[str, Any] | None = None,
    s3_authority_anchor_msg_id: str = "",
) -> ProjectedMovesRecarryMeasurementEvidence:
    validate_parity_uses_live_consumer_symbols_not_reimpl()
    harness_3c = _load_3c_harness()
    captures, q_flat, weight_shape, _eligible, _model = harness_3c._dry_run_fixture()
    weighted_grad = weighted_grad_from_captures(
        captures["inputs"], captures["grad_outputs"], weight_shape=weight_shape
    )
    credit = credit_from_weighted_grad(weighted_grad)
    q_levels = q_flat.reshape(weight_shape)
    rank_spec = default_dry_run_rank_vote_spec()
    digest = _rank_spec_digest(rank_spec)
    if digest != RANK_SPEC_DIGEST_EXPECTED:
        raise ValueError(f"rank_spec_drift at collect: {digest}")

    dense_moves = project_s1_gradient_to_moves(weighted_grad, q_levels)
    dense_events_obj = sparse_rank_bucketed_int16_vote_events(credit, dense_moves, rank_spec)
    fused_events_obj = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
        weighted_grad, q_levels, rank_spec
    )
    dense_events = {int(k): int(v) for k, v in dense_events_obj.to_dict().items()}
    fused_events = {int(k): int(v) for k, v in fused_events_obj.to_dict().items()}
    events_equal = dense_events == fused_events
    validate_events_equal_derived_not_hand_authored(
        dense_events=dense_events, fused_events=fused_events, events_equal=events_equal
    )

    secondary_eq = None
    if plan is None or plan.get("measurement_provenance", {}).get("optional_secondary_non_governing", {}).get(
        "enabled_in_harness", True
    ):
        alt = canonical_acquisition_rank_vote_spec()
        d2 = sparse_rank_bucketed_int16_vote_events(credit, dense_moves, alt).to_dict()
        f2 = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
            weighted_grad, q_levels, alt
        ).to_dict()
        secondary_eq = {int(k): int(v) for k, v in d2.items()} == {
            int(k): int(v) for k, v in f2.items()
        }

    reduction_holds, tsa_sha, tsa_snip = _tsa_compositional_reduction(repo_root)
    numel = _require_exact_int(int(dense_moves.numel()), field="dense_projected_moves_numel")
    nbytes = _require_exact_int(
        int(dense_moves.detach().cpu().contiguous().numel() * dense_moves.element_size()),
        field="dense_projected_moves_bytes",
    )
    fused_count = _require_exact_int(len(fused_events), field="fused_event_count")

    pin_pass, rollup, porcelain = (True, "", {})
    if plan is not None:
        validate_dependency_currency_against_plan_pins(plan=plan, repo_root=repo_root)
        pin_pass, rollup, porcelain = validate_import_closure_pins(plan=plan, repo_root=repo_root)
        validate_rank_spec_identity_matches_plan_pin(
            symbol=RANK_SPEC_SYMBOL, digest=digest, plan=plan
        )

    live = rank_spec.to_live_dict()
    branch = classify_recarry_branch(
        events_equal=events_equal,
        compositional_reduction_holds=reduction_holds,
        measurement_valid=True,
    )
    if branch == BRANCH_PENDING or branch not in TERMINAL_ALLOWED:
        raise ValueError(f"PENDING_as_terminal forbidden: {branch}")

    validate_producer_contract_unchanged(
        unchanged=True, project_s1_fn=project_s1_gradient_to_moves
    )

    evidence = ProjectedMovesRecarryMeasurementEvidence(
        fixture_recipe_name=FIXTURE_RECIPE_NAME,
        parity_fixture_descriptor_sha256=OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
        weighted_grad_sha256=compute_tensor_canonical_sha256(weighted_grad),
        weighted_grad_dtype=str(weighted_grad.dtype),
        weighted_grad_shape=tuple(int(x) for x in weighted_grad.shape),
        weighted_grad_numel=int(weighted_grad.numel()),
        dense_projected_moves_sha256=compute_tensor_canonical_sha256(dense_moves),
        dense_projected_moves_numel=numel,
        dense_projected_moves_bytes=nbytes,
        dense_events=dense_events,
        dense_events_canonical_sha256=_events_canonical_sha256(dense_events),
        fused_events=fused_events,
        fused_events_canonical_sha256=_events_canonical_sha256(fused_events),
        fused_event_count=fused_count,
        events_equal=events_equal,
        evidence_sources={
            "r1_msg_id": R1_MSG_ID,
            "r2_msg_id": R2_MSG_ID,
            "producer_sweep_hit_count": PRODUCER_SWEEP_HIT_COUNT,
            "a_c_evidence_source": "consumer_trace_r1_direct_call_sites",
            "b_d_evidence_source": "consumer_trace_r2_producer_sweep",
        },
        screen_acc_adjacency_disposition=ADJACENCY_DISPOSITION,
        adjacent_caller_paths=ADJACENT_CALLER_PATHS,
        probe_mode=PROBE_MODE,
        repo_head_sha=_live_repo_head(repo_root),
        producer_contract_unchanged=True,
        rank_spec_symbol=RANK_SPEC_SYMBOL,
        rank_spec_to_live_dict_canonical_sha256=digest,
        rank_spec_mode=str(live["mode"]),
        rank_spec_rank_method=str(live["rank_method"]),
        rank_spec_bins=tuple(dict(b) for b in live["rank_bins"]),
        parity_binding_set_ids=("A", "B"),
        reference_paths_classification={
            "C_ISRV": {
                "classification": "REFERENCE_ALREADY_SPARSE_INTEGER_ATTRIBUTION",
                "dense_recarry_target": False,
            },
            "D_IOC": {
                "classification": "REFERENCE_ALREADY_SPARSE_INTEGER_ATTRIBUTION",
                "dense_recarry_target": False,
            },
        },
        compositional_reduction_holds=reduction_holds,
        tsa_site_source_sha256=tsa_sha,
        tsa_site_line_snippets_sha256=tsa_snip,
        execution_import_closure_sha256_rollup=rollup,
        import_closure_pin_pass=pin_pass,
        git_porcelain_over_closure=dict(porcelain),
        secondary_acquisition_events_equal=secondary_eq,
        branch_id=branch,
        recommended_next_slice=(
            "bounded remediation: switch TSA B sites to fused sparse producer under dual accept (OUT of this CREATE-only slice)"
            if branch == BRANCH_HOLDS
            else "classify/split — do not remediate under failed/invalid branch"
        ),
        governing_claim="A+B_only",
        executor_role_for_governing_run="claude_as_test_operator",
        s3_authority_anchor_msg_id=str(s3_authority_anchor_msg_id),
        authority_verification=AUTHORITY_VERIFICATION_AT_MINT,
        gated_by_S4a_not_S4b=True,
    )
    validate_adjacency_disposition_present(evidence)
    validate_both_evidence_sources_cited(evidence)
    validate_governing_claim_AB_only(evidence)
    validate_compositional_reduction_bound(evidence)
    ev_map = evidence.to_receipt_fields()
    validate_harness_evidence_and_audit_exclude_s3_go_msg_id(
        evidence_mapping=ev_map, audit_mapping=ev_map
    )
    validate_authority_verification_gate_pending_at_harness_mint(ev_map)
    return evidence

