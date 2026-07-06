"""Arc #2b Slice-5 in-vivo law-validation branch classifier (inert / not live-wired)."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

CLASSIFIER = "ARC2B_SLICE5_IN_VIVO_BRANCH_V1"

ANTI_OVERCLAIM_VERBATIM = (
    "Within the Slice-5 packet scope, in-vivo accumulator-carrier law validation "
    "classifies as one of the pre-registered branches. FORBIDDEN: sub-2 readiness, "
    "reduction eligibility, ~430MB bank pin, Fold-3B universalization, bank mutation, "
    "full-stack readiness, implementation readiness, main-science launch from Step-1 "
    "CPU prereg alone."
)

ALLOWED_CLAIM = (
    "Within the Slice-5 packet scope, in-vivo accumulator-carrier law validation "
    "classifies as one of the pre-registered branches."
)

RECEIPT_SCHEMA = "hrm_text_158_arc2b_slice5_in_vivo_law_validation_receipt/v1"
B1_CLASSIFIER_RECEIPT_SCHEMA = "hrm_text_158_d_recompute_window_classifier_receipt/v2"
PREFLIGHT_RECEIPT_SCHEMA = (
    "hrm_text_158_arc2b_slice5_feasibility_preflight_receipt/v1"
)
PREREG_PACKET_SCHEMA = (
    "hrm_text_158_arc2b_slice5_in_vivo_law_validation_prereg_packet/v1"
)

B1_RUN_ROOT = (
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "d_recompute_window_feasibility_seed43_43_2189e72017"
)
B1_RUN_ID = "2189e72017"
B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256 = (
    "5d6e0fea9c0f07600a40c36d6df73dc945296d5a684cafa26ba76b206bd7c996"
)
B1_RECORDED_MANIFEST_FILE_SHA256 = (
    "8a128d82473389b050015987320acc6677d8bc9186004b682a90ff588cf9bdf6"
)

PREREG_LAW_WINDOW_K = 180
PREREG_LAW_DECAY_NUM = 1
PREREG_LAW_DECAY_DEN = 2
DEFAULT_EFFECTIVE_ACC_BUDGET_BPW = 0.4
DEFAULT_TOLERANCE_BPW = 0.0
NUMEL_BASIS_SOURCE = "parent_checkpoint_tensor_state_numel"

CARRIER_BYTE_COMPONENTS: tuple[str, ...] = (
    "events_bytes",
    "backlog_bytes",
    "hot_exact_bytes",
    "metadata_bytes",
)

EVIDENCE_B1_OFFLINE_BRACKET = "b1_offline_bracket"
EVIDENCE_STEP2_GPU_LIVE_CARRIER = "step2_gpu_live_carrier"

OFFLINE_B1_ALLOWED_TERMINALS: frozenset[str] = frozenset(
    {
        "SLICE5_NO_VERDICT_OPERATIONAL",
        "SLICE5_NO_VERDICT_SCHEMA",
        "SLICE5_DIAGNOSTIC_BRACKET_ONLY",
        "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH",
        "SLICE5_INCONCLUSIVE_NO_LIVE_SNAPSHOT",
        "SLICE5_INCONCLUSIVE_INPUT_DRIFT",
        "SLICE5_INCONCLUSIVE_LOG_COVERAGE",
    }
)

STEP2_ONLY_TERMINALS: frozenset[str] = frozenset(
    {
        "D_NEEDS_UPDATE_LAW_REDESIGN",
        "SLICE5_IN_VIVO_LAW_BOUND",
    }
)

REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "schema",
    "task_id",
    "git_head_required",
    "evidence_source",
    "prereg_law_window_k",
    "prereg_law_decay_num",
    "prereg_law_decay_den",
    "runtime_decay_num",
    "runtime_decay_den",
    "runtime_window_k",
    "recorded_selector_internal_manifest_sha256",
    "manifest_binding_ok",
    "eligible_weight_numel",
    "numel_basis_source",
    "effective_acc_budget_bpw",
    "tolerance_bpw",
    "live_snapshot_present",
    "resume_generation",
    "offline_bracket_decision",
    "live_acc_carrier_bpw_max",
    "slice5_branch",
    "slice5_branch_inputs",
    "ready_for_main_science",
    "counts_as_sub2",
    "pre_full_stack_diagnostic",
    "autonomy_rung",
)

REQUIRED_BRANCH_INPUT_FIELDS: tuple[str, ...] = (
    "operational_ok",
    "schema_ok",
    "evidence_source",
    "prereg_law_window_k",
    "prereg_law_decay_num",
    "prereg_law_decay_den",
    "runtime_decay_num",
    "runtime_decay_den",
    "runtime_window_k",
    "recorded_selector_internal_manifest_sha256",
    "on_disk_selector_manifest_sha256",
    "manifest_binding_ok",
    "log_coverage_ok",
    "live_snapshot_present",
    "resume_generation",
    "offline_bracket_decision",
    "live_carrier_rows",
    "eligible_weight_numel",
    "effective_acc_budget_bpw",
    "tolerance_bpw",
)


class Arc2bSlice5InVivoBranch(StrEnum):
    NO_VERDICT_OPERATIONAL = "SLICE5_NO_VERDICT_OPERATIONAL"
    NO_VERDICT_SCHEMA = "SLICE5_NO_VERDICT_SCHEMA"
    INCONCLUSIVE_INPUT_DRIFT = "SLICE5_INCONCLUSIVE_INPUT_DRIFT"
    INCONCLUSIVE_SOURCE_MISMATCH = "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH"
    INCONCLUSIVE_LOG_COVERAGE = "SLICE5_INCONCLUSIVE_LOG_COVERAGE"
    INCONCLUSIVE_NO_LIVE_SNAPSHOT = "SLICE5_INCONCLUSIVE_NO_LIVE_SNAPSHOT"
    DIAGNOSTIC_BRACKET_ONLY = "SLICE5_DIAGNOSTIC_BRACKET_ONLY"
    NEEDS_UPDATE_LAW_REDESIGN = "D_NEEDS_UPDATE_LAW_REDESIGN"
    IN_VIVO_LAW_BOUND = "SLICE5_IN_VIVO_LAW_BOUND"


BRANCH_PRECEDENCE: tuple[Arc2bSlice5InVivoBranch, ...] = (
    Arc2bSlice5InVivoBranch.NO_VERDICT_OPERATIONAL,
    Arc2bSlice5InVivoBranch.NO_VERDICT_SCHEMA,
    Arc2bSlice5InVivoBranch.INCONCLUSIVE_INPUT_DRIFT,
    Arc2bSlice5InVivoBranch.INCONCLUSIVE_SOURCE_MISMATCH,
    Arc2bSlice5InVivoBranch.INCONCLUSIVE_LOG_COVERAGE,
    Arc2bSlice5InVivoBranch.INCONCLUSIVE_NO_LIVE_SNAPSHOT,
    Arc2bSlice5InVivoBranch.DIAGNOSTIC_BRACKET_ONLY,
    Arc2bSlice5InVivoBranch.NEEDS_UPDATE_LAW_REDESIGN,
    Arc2bSlice5InVivoBranch.IN_VIVO_LAW_BOUND,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def live_acc_carrier_bytes_total(snapshot: Mapping[str, Any]) -> int | None:
    if not isinstance(snapshot, Mapping):
        return None
    if snapshot.get("live_carrier_bytes_exact") is not True:
        return None
    component_values: list[int] = []
    for key in CARRIER_BYTE_COMPONENTS:
        raw = snapshot.get(key)
        if type(raw) is not int or raw < 0:
            return None
        component_values.append(raw)
    total = int(sum(component_values))
    recorded_total = snapshot.get("live_acc_carrier_bytes_total")
    if type(recorded_total) is int and recorded_total != total:
        return None
    return total


def compute_live_acc_bpw(
    *,
    live_acc_carrier_bytes_total: int,
    eligible_weight_numel: int,
) -> float:
    if int(eligible_weight_numel) <= 0:
        raise ValueError("eligible_weight_numel must be positive")
    return (float(int(live_acc_carrier_bytes_total)) * 8.0) / float(
        int(eligible_weight_numel)
    )


def compute_max_live_acc_bpw(
    *,
    live_carrier_rows: Sequence[Mapping[str, Any]],
    eligible_weight_numel: int,
) -> float | None:
    max_bpw: float | None = None
    for row in live_carrier_rows:
        total_bytes = live_acc_carrier_bytes_total(row)
        if total_bytes is None:
            continue
        bpw = compute_live_acc_bpw(
            live_acc_carrier_bytes_total=int(total_bytes),
            eligible_weight_numel=int(eligible_weight_numel),
        )
        max_bpw = bpw if max_bpw is None else max(max_bpw, bpw)
    return max_bpw


def strict_under_budget(
    *,
    observed_bpw: float,
    effective_acc_budget_bpw: float,
    tolerance_bpw: float = DEFAULT_TOLERANCE_BPW,
) -> bool:
    return float(observed_bpw) < (
        float(effective_acc_budget_bpw) - float(tolerance_bpw)
    )


def verify_manifest_binding(
    *,
    recorded_selector_internal_manifest_sha256: str | None,
    recorded_manifest_file_sha256: str | None,
    on_disk_manifest_path: Path | None,
    expected_selector_internal_manifest_sha256: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {
        "recorded_selector_internal_manifest_sha256": recorded_selector_internal_manifest_sha256,
        "recorded_manifest_file_sha256": recorded_manifest_file_sha256,
        "on_disk_manifest_path": str(on_disk_manifest_path) if on_disk_manifest_path else None,
    }
    if not recorded_selector_internal_manifest_sha256:
        details["reason"] = "missing_recorded_selector_internal_manifest_sha256"
        return False, details
    if (
        expected_selector_internal_manifest_sha256 is not None
        and recorded_selector_internal_manifest_sha256
        != expected_selector_internal_manifest_sha256
    ):
        details["reason"] = "recorded_selector_internal_manifest_sha256_mismatch"
        return False, details
    if on_disk_manifest_path is None or not on_disk_manifest_path.is_file():
        details["reason"] = "on_disk_manifest_missing"
        return False, details
    on_disk_file_sha256 = sha256_file(on_disk_manifest_path)
    details["on_disk_manifest_file_sha256"] = on_disk_file_sha256
    if recorded_manifest_file_sha256 and on_disk_file_sha256 != recorded_manifest_file_sha256:
        details["reason"] = "on_disk_manifest_file_sha256_mismatch"
        return False, details
    if on_disk_manifest_path.stat().st_size <= 0:
        details["reason"] = "on_disk_manifest_empty"
        return False, details
    details["reason"] = "ok"
    return True, details


def _coerce_int(value: Any) -> int | None:
    if type(value) is int:
        return value
    return None


def _decay_tuple(num: Any, den: Any) -> tuple[int, int] | None:
    coerced_num = _coerce_int(num)
    coerced_den = _coerce_int(den)
    if coerced_num is None or coerced_den is None or coerced_den <= 0:
        return None
    return (coerced_num, coerced_den)


def _law_decay_tuple(inputs: Mapping[str, Any]) -> tuple[int, int] | None:
    return _decay_tuple(
        inputs.get("prereg_law_decay_num"),
        inputs.get("prereg_law_decay_den"),
    )


def _runtime_decay_tuple(inputs: Mapping[str, Any]) -> tuple[int, int] | None:
    return _decay_tuple(
        inputs.get("runtime_decay_num"),
        inputs.get("runtime_decay_den"),
    )


def _runtime_window_k(inputs: Mapping[str, Any]) -> int | None:
    return _coerce_int(inputs.get("runtime_window_k"))


def _prereg_window_k(inputs: Mapping[str, Any]) -> int | None:
    return _coerce_int(inputs.get("prereg_law_window_k"))


def _source_matches_prereg_law(inputs: Mapping[str, Any]) -> bool:
    law_decay = _law_decay_tuple(inputs)
    runtime_decay = _runtime_decay_tuple(inputs)
    if law_decay is None or runtime_decay is None:
        return False
    if law_decay != runtime_decay:
        return False
    runtime_window_k = _runtime_window_k(inputs)
    prereg_window_k = _prereg_window_k(inputs)
    if runtime_window_k is not None and prereg_window_k is not None:
        if runtime_window_k != prereg_window_k:
            return False
    return True


def _pick_terminal(fired: Sequence[Arc2bSlice5InVivoBranch]) -> Arc2bSlice5InVivoBranch:
    unique: list[Arc2bSlice5InVivoBranch] = []
    for branch in fired:
        if branch not in unique:
            unique.append(branch)
    if not unique:
        return Arc2bSlice5InVivoBranch.INCONCLUSIVE_NO_LIVE_SNAPSHOT
    precedence_index = {branch: idx for idx, branch in enumerate(BRANCH_PRECEDENCE)}
    return min(unique, key=lambda branch: precedence_index[branch])


def _enforce_offline_b1_terminal_guard(
    *,
    evidence_source: str,
    terminal: Arc2bSlice5InVivoBranch,
    fired: list[Arc2bSlice5InVivoBranch],
) -> Arc2bSlice5InVivoBranch:
    if evidence_source != EVIDENCE_B1_OFFLINE_BRACKET:
        return terminal
    if terminal.value in OFFLINE_B1_ALLOWED_TERMINALS:
        return terminal
    fallback_candidates = [
        branch
        for branch in fired
        if branch.value in OFFLINE_B1_ALLOWED_TERMINALS
    ]
    if fallback_candidates:
        return _pick_terminal(fallback_candidates)
    return Arc2bSlice5InVivoBranch.DIAGNOSTIC_BRACKET_ONLY


def classify_arc2b_slice5_in_vivo_branch(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    fired: list[Arc2bSlice5InVivoBranch] = []
    evidence_source = str(inputs.get("evidence_source") or "")

    if inputs.get("operational_ok") is not True:
        fired.append(Arc2bSlice5InVivoBranch.NO_VERDICT_OPERATIONAL)
        terminal = Arc2bSlice5InVivoBranch.NO_VERDICT_OPERATIONAL
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "slice5_branch_inputs": dict(inputs),
            "autonomy_rung": "step1_cpu_prereg"
            if evidence_source == EVIDENCE_B1_OFFLINE_BRACKET
            else "step2_gpu_mechanism",
        }

    if inputs.get("schema_ok") is not True:
        fired.append(Arc2bSlice5InVivoBranch.NO_VERDICT_SCHEMA)
        terminal = Arc2bSlice5InVivoBranch.NO_VERDICT_SCHEMA
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "slice5_branch_inputs": dict(inputs),
            "autonomy_rung": "step1_cpu_prereg"
            if evidence_source == EVIDENCE_B1_OFFLINE_BRACKET
            else "step2_gpu_mechanism",
        }

    if inputs.get("manifest_binding_ok") is not True:
        fired.append(Arc2bSlice5InVivoBranch.INCONCLUSIVE_INPUT_DRIFT)

    source_matches = _source_matches_prereg_law(inputs)
    if not source_matches:
        fired.append(Arc2bSlice5InVivoBranch.INCONCLUSIVE_SOURCE_MISMATCH)

    if inputs.get("log_coverage_ok") is not True:
        fired.append(Arc2bSlice5InVivoBranch.INCONCLUSIVE_LOG_COVERAGE)

    if inputs.get("live_snapshot_present") is not True:
        fired.append(Arc2bSlice5InVivoBranch.INCONCLUSIVE_NO_LIVE_SNAPSHOT)

    if evidence_source == EVIDENCE_B1_OFFLINE_BRACKET:
        fired.append(Arc2bSlice5InVivoBranch.DIAGNOSTIC_BRACKET_ONLY)
        terminal = _pick_terminal(fired)
        terminal = _enforce_offline_b1_terminal_guard(
            evidence_source=evidence_source,
            terminal=terminal,
            fired=fired,
        )
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "slice5_branch_inputs": dict(inputs),
            "autonomy_rung": "step1_cpu_prereg",
        }

    if evidence_source == EVIDENCE_STEP2_GPU_LIVE_CARRIER:
        resume_generation = _coerce_int(inputs.get("resume_generation"))
        if resume_generation != 0:
            fired.append(Arc2bSlice5InVivoBranch.NO_VERDICT_SCHEMA)
        live_rows = inputs.get("live_carrier_rows") or []
        eligible_weight_numel = _coerce_int(inputs.get("eligible_weight_numel"))
        budget = inputs.get("effective_acc_budget_bpw")
        tolerance = inputs.get("tolerance_bpw", DEFAULT_TOLERANCE_BPW)
        max_bpw = None
        if isinstance(live_rows, Sequence) and eligible_weight_numel is not None:
            max_bpw = compute_max_live_acc_bpw(
                live_carrier_rows=live_rows,
                eligible_weight_numel=eligible_weight_numel,
            )
        if (
            resume_generation == 0
            and source_matches
            and inputs.get("live_snapshot_present") is True
            and max_bpw is not None
            and isinstance(budget, (int, float))
        ):
            if strict_under_budget(
                observed_bpw=float(max_bpw),
                effective_acc_budget_bpw=float(budget),
                tolerance_bpw=float(tolerance or 0.0),
            ):
                fired.append(Arc2bSlice5InVivoBranch.IN_VIVO_LAW_BOUND)
            else:
                fired.append(Arc2bSlice5InVivoBranch.NEEDS_UPDATE_LAW_REDESIGN)
        terminal = _pick_terminal(fired)
        return {
            "classifier": CLASSIFIER,
            "terminal_branch": terminal.value,
            "fired_branches": [branch.value for branch in fired],
            "slice5_branch_inputs": dict(inputs),
            "live_acc_carrier_bpw_max": max_bpw,
            "autonomy_rung": "step2_gpu_mechanism",
        }

    fired.append(Arc2bSlice5InVivoBranch.INCONCLUSIVE_SOURCE_MISMATCH)
    terminal = _pick_terminal(fired)
    return {
        "classifier": CLASSIFIER,
        "terminal_branch": terminal.value,
        "fired_branches": [branch.value for branch in fired],
        "slice5_branch_inputs": dict(inputs),
        "autonomy_rung": "step1_cpu_prereg",
    }


def build_branch_input_from_b1_classifier_receipt(
    receipt: Mapping[str, Any],
    *,
    operational_ok: bool = True,
    on_disk_manifest_path: Path | None = None,
    offline_bracket_decision: str | None = None,
) -> dict[str, Any]:
    input_hashes = dict(receipt.get("input_artifact_hashes") or {})
    manifest_entry = dict(
        input_hashes.get("prelaunch/calibrated_selector_manifest.json") or {}
    )
    recorded_internal_sha = manifest_entry.get("selector_internal_manifest_sha256")
    recorded_file_sha = manifest_entry.get("sha256")
    manifest_path = on_disk_manifest_path
    if manifest_path is None:
        run_root = Path(str(receipt.get("run_root") or B1_RUN_ROOT))
        manifest_path = run_root / "prelaunch" / "calibrated_selector_manifest.json"
    manifest_binding_ok, _details = verify_manifest_binding(
        recorded_selector_internal_manifest_sha256=(
            str(recorded_internal_sha) if recorded_internal_sha else None
        ),
        recorded_manifest_file_sha256=(
            str(recorded_file_sha) if recorded_file_sha else None
        ),
        on_disk_manifest_path=manifest_path,
        expected_selector_internal_manifest_sha256=B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256,
    )

    acc_sizing = dict(receipt.get("acc_sizing") or {})
    best_grid = dict(acc_sizing.get("best_grid_row") or {})
    in_vivo = dict(receipt.get("in_vivo_validation") or {})
    logged_surface = dict(in_vivo.get("logged_density_surface") or {})
    numel_by_key = dict(receipt.get("numel_by_key") or {})
    eligible_weight_numel = sum(int(value) for value in numel_by_key.values())

    # Runtime decay from B1 log is 1/1; law under test from sizing grid is 1/2.
    runtime_decay_num = 1
    runtime_decay_den = 1
    runtime_window_k = _coerce_int(best_grid.get("window_k")) or PREREG_LAW_WINDOW_K

    log_coverage_ok = bool(logged_surface.get("records_in_window")) and bool(
        logged_surface.get("steps_in_window")
    )
    if offline_bracket_decision is None:
        offline_bracket_decision = str(
            in_vivo.get("in_vivo_verdict")
            or receipt.get("primary_classifier")
            or "INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT"
        )

    return {
        "operational_ok": operational_ok,
        "schema_ok": receipt.get("schema") == B1_CLASSIFIER_RECEIPT_SCHEMA,
        "evidence_source": EVIDENCE_B1_OFFLINE_BRACKET,
        "prereg_law_window_k": PREREG_LAW_WINDOW_K,
        "prereg_law_decay_num": PREREG_LAW_DECAY_NUM,
        "prereg_law_decay_den": PREREG_LAW_DECAY_DEN,
        "runtime_decay_num": runtime_decay_num,
        "runtime_decay_den": runtime_decay_den,
        "runtime_window_k": runtime_window_k,
        "recorded_selector_internal_manifest_sha256": recorded_internal_sha,
        "on_disk_selector_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "manifest_binding_ok": manifest_binding_ok,
        "log_coverage_ok": log_coverage_ok,
        "live_snapshot_present": False,
        "resume_generation": None,
        "offline_bracket_decision": offline_bracket_decision,
        "live_carrier_rows": [],
        "eligible_weight_numel": int(eligible_weight_numel),
        "effective_acc_budget_bpw": float(
            acc_sizing.get("effective_acc_budget_bpw") or DEFAULT_EFFECTIVE_ACC_BUDGET_BPW
        ),
        "tolerance_bpw": DEFAULT_TOLERANCE_BPW,
    }


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_branch_input_from_step2_gpu_run(
    *,
    run_root: Path,
    operational_ok: bool = True,
    recorded_selector_internal_manifest_sha256: str | None = (
        B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
    ),
    recorded_manifest_file_sha256: str | None = B1_RECORDED_MANIFEST_FILE_SHA256,
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    hygiene_receipt: Mapping[str, Any] | None = None,
    probe_receipt: Mapping[str, Any] | None = None,
    eligible_weight_numel: int | None = None,
) -> dict[str, Any]:
    scratch = run_root / "d_recompute_window_diagnostic"
    log_path = scratch / "recompute_window_log.jsonl"
    live_path = scratch / "live_carrier_snapshot.jsonl"
    manifest_path = run_root / "prelaunch" / "calibrated_selector_manifest.json"

    log_rows = _load_jsonl_rows(log_path)
    replay_constants = dict(log_rows[0].get("replay_constants") or {}) if log_rows else {}
    runtime_decay_num = _coerce_int(replay_constants.get("decay_numerator"))
    runtime_decay_den = _coerce_int(replay_constants.get("decay_denominator"))
    runtime_window_k = PREREG_LAW_WINDOW_K

    live_rows = [
        row
        for row in _load_jsonl_rows(live_path)
        if row.get("live_carrier_bytes_exact") is True
    ]

    manifest_binding_ok, _details = verify_manifest_binding(
        recorded_selector_internal_manifest_sha256=recorded_selector_internal_manifest_sha256,
        recorded_manifest_file_sha256=recorded_manifest_file_sha256,
        on_disk_manifest_path=manifest_path,
        expected_selector_internal_manifest_sha256=recorded_selector_internal_manifest_sha256,
    )

    resume_generation: int | None = None
    if hygiene_receipt is not None:
        if (
            hygiene_receipt.get("pass") is True
            and hygiene_receipt.get("bounded_steps_start_count") == 1
        ):
            resume_generation = 0
    elif probe_receipt is not None and probe_receipt.get("resume_generation") == 0:
        resume_generation = 0

    if eligible_weight_numel is None and probe_receipt is not None:
        numel_by_key = dict(probe_receipt.get("numel_by_key") or {})
        if numel_by_key:
            eligible_weight_numel = sum(int(value) for value in numel_by_key.values())
    if eligible_weight_numel is None:
        eligible_weight_numel = 0

    steps_completed = 0
    if probe_receipt is not None:
        steps_completed = int(probe_receipt.get("steps_completed") or 0)
    log_coverage_ok = bool(log_rows) and steps_completed > 0 and len(log_rows) >= steps_completed

    return {
        "operational_ok": operational_ok,
        "schema_ok": True,
        "evidence_source": EVIDENCE_STEP2_GPU_LIVE_CARRIER,
        "prereg_law_window_k": PREREG_LAW_WINDOW_K,
        "prereg_law_decay_num": PREREG_LAW_DECAY_NUM,
        "prereg_law_decay_den": PREREG_LAW_DECAY_DEN,
        "runtime_decay_num": runtime_decay_num,
        "runtime_decay_den": runtime_decay_den,
        "runtime_window_k": runtime_window_k,
        "recorded_selector_internal_manifest_sha256": recorded_selector_internal_manifest_sha256,
        "on_disk_selector_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "manifest_binding_ok": manifest_binding_ok,
        "log_coverage_ok": log_coverage_ok,
        "live_snapshot_present": bool(live_rows),
        "resume_generation": resume_generation,
        "offline_bracket_decision": None,
        "live_carrier_rows": live_rows,
        "eligible_weight_numel": int(eligible_weight_numel),
        "effective_acc_budget_bpw": float(effective_acc_budget_bpw),
        "tolerance_bpw": DEFAULT_TOLERANCE_BPW,
    }


def build_adversarial_b1_offline_branch_input() -> dict[str, Any]:
    """Hostile-density B1 fixture: decay 1/1 runtime vs 1/2 law, no live snapshot."""
    return {
        "operational_ok": True,
        "schema_ok": True,
        "evidence_source": EVIDENCE_B1_OFFLINE_BRACKET,
        "prereg_law_window_k": PREREG_LAW_WINDOW_K,
        "prereg_law_decay_num": PREREG_LAW_DECAY_NUM,
        "prereg_law_decay_den": PREREG_LAW_DECAY_DEN,
        "runtime_decay_num": 1,
        "runtime_decay_den": 1,
        "runtime_window_k": PREREG_LAW_WINDOW_K,
        "recorded_selector_internal_manifest_sha256": (
            B1_RECORDED_SELECTOR_INTERNAL_MANIFEST_SHA256
        ),
        "on_disk_selector_manifest_sha256": B1_RECORDED_MANIFEST_FILE_SHA256,
        "manifest_binding_ok": True,
        "log_coverage_ok": True,
        "live_snapshot_present": False,
        "resume_generation": None,
        "offline_bracket_decision": "REAL_DENSITY_EXCEEDS_SUB2",
        "live_carrier_rows": [],
        "eligible_weight_numel": 18874368,
        "effective_acc_budget_bpw": DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
        "tolerance_bpw": DEFAULT_TOLERANCE_BPW,
        "adversarial_backlog_density_bpw": 0.7838234823258197,
        "adversarial_peak_backlog_depth": 130816,
    }


def validate_prereg_packet_schema(packet: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema") != PREREG_PACKET_SCHEMA:
        failures.append("schema_mismatch")
    for key in (
        "classifier",
        "branch_enum",
        "receipt_schema",
        "feasibility_preflight",
        "run_determinism",
        "step2_gpu_launch_scope",
        "b1_diagnostic_anchor",
        "carrier_byte_mapping",
        "offline_b1_contract",
        "claim_boundary",
        "decision_contract",
    ):
        if key not in packet:
            failures.append(f"missing:{key}")
    receipt_schema = packet.get("receipt_schema") or {}
    packet_required = receipt_schema.get("required_fields")
    if isinstance(packet_required, list):
        if set(REQUIRED_RECEIPT_FIELDS) != set(packet_required):
            failures.append("required_receipt_fields_packet_drift")
    offline_contract = packet.get("offline_b1_contract") or {}
    forbidden = offline_contract.get("forbidden_terminal_branches")
    if forbidden != sorted(STEP2_ONLY_TERMINALS):
        failures.append("offline_b1_forbidden_terminal_branches_drift")
    return failures


def validate_preflight_receipt_schema(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA:
        failures.append("schema_mismatch")
    for key in (
        "feasibility_verdict",
        "run_determinism_classification",
        "b1_diagnostic_anchor",
        "carrier_byte_mapping",
        "offline_b1_contract",
        "readiness_classification",
    ):
        if key not in receipt:
            failures.append(f"missing:{key}")
    if receipt.get("feasibility_verdict") not in {"EXISTS", "NEEDS_PATCH", "INFEASIBLE"}:
        failures.append("invalid_feasibility_verdict")
    if receipt.get("run_determinism_classification") not in {
        "DETERMINISTIC",
        "NON_DETERMINISTIC",
    }:
        failures.append("invalid_run_determinism_classification")
    readiness = receipt.get("readiness_classification") or {}
    if readiness.get("class") != "pre_full_stack_diagnostic":
        failures.append("readiness_class_not_pre_full_stack_diagnostic")
    flags = readiness.get("flags") or {}
    if flags.get("ready_for_main_science") is not False:
        failures.append("ready_for_main_science_not_false")
    if flags.get("counts_as_sub2") is not False:
        failures.append("counts_as_sub2_not_false")
    if flags.get("pre_full_stack_diagnostic") is not True:
        failures.append("pre_full_stack_diagnostic_not_true")
    return failures
