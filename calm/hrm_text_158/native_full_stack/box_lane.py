"""Box-lane chain packet helpers (code-currency preflight + science-chain watcher)."""
from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

EXIT_OK = 0
EXIT_CODE_CURRENCY_MISMATCH = 11
EXIT_ARTIFACT_RSYNC_MISMATCH = 12
EXIT_OVERLAP_FAILURE = 2

SCIENCE_CHAIN_CREDITDIR = "/home/gabe/claw-code-creditdir/transient_fp_credit"

DEFAULT_FLOOR_PINNED_FILES: tuple[tuple[str, str], ...] = (
    ("probe_cli", "scripts/probe_hrm_text_158.py"),
    ("acc_width_sweep", "scripts/hrm_text_158_acc_width_recorded_row_sweep.py"),
    ("selection_audit", "scripts/hrm_text_158_transient_selection_information_audit.py"),
    ("falsifier_battery", "scripts/hrm_text_158_two_tier_carry_falsifier_battery.py"),
    ("credit_bridge", "scripts/hrm_text_158_credit_bridge.py"),
    ("parallel_watcher", "scripts/parallel_audit_watcher.py"),
    ("science_harness", "calm/hrm_text_158/native_full_stack/optimizer_update_law_science.py"),
)

ANALYZER_PINNED_FILES: tuple[tuple[str, str], ...] = (
    ("selector_value_analysis", "calm/hrm_text_158/native_full_stack/selector_value_analysis.py"),
    ("selector_value_orchestrator", "scripts/hrm_text_158_selector_value_analysis.py"),
)

PHASE3_OBMALLOC_SURFACE_PINNED_FILES: tuple[tuple[str, str], ...] = (
    ("event_coded_acc_live_carrier", "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py"),
    ("sparse_cap_gpu_seam_adapter", "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py"),
    ("event_coded_vote_update_adapter", "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py"),
    ("bounded_delta_learner", "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py"),
    ("attribution_script", "scripts/hrm_text_158_slice5_v6i_oom_profile_attribution.py"),
    ("code_currency_guard", "scripts/hrm_text_158_code_currency_guard.py"),
    ("probe_bootstrap", "scripts/hrm_text_158_bounded_delta_acquisition_probe_bootstrap.py"),
)

CAPTURE_COMPLETE_RE = re.compile(
    r"capture_complete:\s+chain_id=(?P<chain_id>\S+)(?:\s+seed=(?P<seed>\d+))?",
)
PRODUCER_NEXT_START_RE = re.compile(
    r"producer_next_capture_start:\s+chain_id=(?P<chain_id>\S+)",
)
CONSUMER_AUDIT_START_RE = re.compile(
    r"consumer_audit_start:\s+chain_id=(?P<chain_id>\S+)",
)
CONSUMER_TERMINAL_RE = re.compile(
    r"consumer_terminal:\s+chain_id=(?P<chain_id>\S+)\s+status=(?P<status>\S+)",
)

ARTIFACT_TRANSPORT_SCHEMA = "hrm158_box_lane_artifact_transport/v1"
CONSENSUS_CONSUMER_AUDIT_SCHEMA = "hrm158_box_lane_consensus_consumer_audit/v1"


@dataclass(frozen=True)
class ChainArtifact:
    role: str
    rel_path: str
    optional: bool = False


def default_consensus_chain_artifacts(
    *,
    labels: Sequence[str] = ("S44", "S44_iso43", "S43"),
    arms: Sequence[str] = ("on", "off"),
    include_run_log: bool = True,
) -> list[ChainArtifact]:
    artifacts: list[ChainArtifact] = []
    for label in labels:
        for arm in arms:
            artifacts.append(ChainArtifact(f"{label}_{arm}_receipt", f"{label}/{arm}/receipt.json"))
            if include_run_log:
                artifacts.append(
                    ChainArtifact(
                        f"{label}_{arm}_run_log",
                        f"{label}/{arm}/run.log",
                        optional=True,
                    ),
                )
    return artifacts


def format_capture_complete_line(
    *,
    chain_id: str,
    code_currency_pass: bool,
    artifact_sha_verified: bool,
    ts: float,
    seed: int | None = None,
) -> str:
    seed_token = f" seed={seed}" if seed is not None else ""
    return (
        f"capture_complete: chain_id={chain_id}{seed_token} "
        f"code_currency_pass={'true' if code_currency_pass else 'false'} "
        f"artifact_sha_verified={'true' if artifact_sha_verified else 'false'} "
        f"ts={ts}"
    )


def format_consumer_audit_start_line(*, chain_id: str, ts: float) -> str:
    return f"consumer_audit_start: chain_id={chain_id} ts={ts}"


def format_consumer_terminal_line(*, chain_id: str, status: str, ts: float | None = None) -> str:
    if ts is not None:
        return f"consumer_terminal: chain_id={chain_id} status={status} ts={ts}"
    return f"consumer_terminal: chain_id={chain_id} status={status}"


def sync_chain_arm_artifacts(
    *,
    local_chain_root: Path,
    remote_chain_root: str,
    box: str,
    artifacts: Sequence[ChainArtifact],
    rsync_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    remote_sha_runner: Callable[[str, str], str | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    mismatches: list[str] = []
    synced_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        rel = artifact.rel_path.replace("\\", "/")
        local_path = local_chain_root / rel
        if not local_path.exists():
            if artifact.optional:
                synced_rows.append(
                    {
                        "role": artifact.role,
                        "rel_path": rel,
                        "optional": True,
                        "missing": True,
                        "skipped": True,
                    },
                )
                continue
            mismatches.append(f"missing:{rel}")
            synced_rows.append(
                {
                    "role": artifact.role,
                    "rel_path": rel,
                    "optional": artifact.optional,
                    "missing": True,
                },
            )
            continue
        producer_sha = sha256_file(local_path)
        remote_path = f"{box}:{remote_chain_root}/{rel}"
        rsync_cmd = [
            "rsync",
            "-az",
            "--mkpath",
            str(local_path),
            remote_path,
        ]
        try:
            rsync_runner(rsync_cmd)
            remote_sha = remote_sha_runner(box, f"{remote_chain_root}/{rel}")
        except subprocess.CalledProcessError as exc:
            cmd = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
            mismatches.append(f"rsync_transport_failure:{rel}")
            synced_rows.append(
                {
                    "role": artifact.role,
                    "rel_path": rel,
                    "optional": artifact.optional,
                    "producer_sha256": producer_sha,
                    "consumer_sha256": None,
                    "remote_sha256": None,
                    "rsync_ok": False,
                    "rsync_cmd": cmd,
                    "rsync_exit_code": int(exc.returncode) if exc.returncode is not None else None,
                },
            )
            continue
        ok = remote_sha == producer_sha
        if not ok:
            mismatches.append(f"artifact_sha_mismatch:{rel}")
        synced_rows.append(
            {
                "role": artifact.role,
                "rel_path": rel,
                "optional": artifact.optional,
                "producer_sha256": producer_sha,
                "consumer_sha256": remote_sha,
                "remote_sha256": remote_sha,
                "rsync_ok": ok,
                "rsync_cmd": rsync_cmd,
            },
        )
    return mismatches, synced_rows


def build_artifact_transport_manifest(
    *,
    chain_id: str,
    local_chain_root: Path,
    remote_chain_root: str,
    artifacts: Sequence[Mapping[str, Any]],
    mismatches: Sequence[str],
    sync_requested: bool,
) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_TRANSPORT_SCHEMA,
        "chain_id": chain_id,
        "local_chain_root": str(local_chain_root),
        "remote_chain_root": remote_chain_root,
        "sync_requested": sync_requested,
        "artifact_transport_pass": not mismatches,
        "mismatches": list(mismatches),
        "artifacts": list(artifacts),
    }


def _required_receipt_keys_present(receipt: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not isinstance(receipt.get("step_reports"), Mapping):
        issues.append("missing_step_reports")
    terminal = receipt.get("terminal_status")
    stop_reason = receipt.get("stop_reason")
    if not isinstance(terminal, Mapping) and not stop_reason:
        issues.append("missing_terminal_status_or_stop_reason")
    return issues


def audit_consensus_bounded_delta_consumer(
    chain_root: Path,
    *,
    primary_label: str = "S44",
    isolation_label: str = "S44_iso43",
    corroboration_label: str | None = None,
    consensus_mode: bool = False,
    transport_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.pressure_shape_agreement import (
        verify_pressure_shape_summary_preflight,
    )

    issues: list[str] = []
    primary_on_path = chain_root / primary_label / "on" / "receipt.json"
    isolation_on_path = chain_root / isolation_label / "on" / "receipt.json"
    receipts_checked: dict[str, Any] = {}

    receipt_specs: list[tuple[str, Path]] = [
        (f"{primary_label}_on", primary_on_path),
        (f"{isolation_label}_on", isolation_on_path),
    ]
    if consensus_mode:
        if corroboration_label is None:
            issues.append("missing_corroboration_label")
        else:
            corroboration_on_path = chain_root / corroboration_label / "on" / "receipt.json"
            receipt_specs.append((f"{corroboration_label}_on", corroboration_on_path))

    for label, path in receipt_specs:
        if not path.exists():
            issues.append(f"missing_receipt:{label}")
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"receipt_parse_error:{label}")
            continue
        receipts_checked[label] = {"path": str(path)}
        issues.extend(f"{label}:{issue}" for issue in _required_receipt_keys_present(receipt))
        preflight = verify_pressure_shape_summary_preflight(receipt, receipt_path=path)
        receipts_checked[label]["pressure_shape_preflight"] = preflight
        if not bool(preflight.get("pass")):
            issues.append(f"pressure_shape_preflight_fail:{label}")

    if transport_artifacts:
        for row in transport_artifacts:
            rel = row.get("rel_path", "?")
            is_optional = bool(row.get("optional"))
            is_missing = bool(row.get("missing") or row.get("skipped"))
            if is_optional and is_missing:
                continue
            if is_missing:
                issues.append(f"transport_missing_required_artifact:{rel}")
                continue
            producer_sha = row.get("producer_sha256")
            consumer_sha = row.get("consumer_sha256") or row.get("remote_sha256")
            if producer_sha is None or consumer_sha is None:
                issues.append(f"transport_missing_sha:{rel}")
            elif producer_sha != consumer_sha:
                issues.append(f"transport_sha_mismatch:{rel}")

    analysis_summary_path = chain_root / "analysis" / "selector_support_consensus_summary.json"
    if not analysis_summary_path.exists():
        analysis_summary_path = chain_root / "analysis" / "selector_support_invariance_summary.json"
    analysis_payload: dict[str, Any] | None = None
    if analysis_summary_path.exists():
        try:
            analysis_payload = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
            branch = (analysis_payload.get("branch_precedence_receipt") or {}).get("branch")
            if not branch:
                issues.append("analysis_missing_branch_precedence")
            if consensus_mode and analysis_summary_path.name == "selector_support_consensus_summary.json":
                identity = analysis_payload.get("consensus_identity") or {}
                if identity.get("intersection_core_fraction") is None:
                    issues.append("analysis_missing_intersection_core_fraction")
                receipt_inputs = (
                    (analysis_payload.get("branch_precedence_receipt") or {}).get("inputs") or {}
                )
                intersection = identity.get("intersection_core_fraction")
                held = receipt_inputs.get("held_median_topk_jaccard")
                if held is not None and intersection is not None and held == intersection:
                    issues.append("intersection_core_fraction_aliased_to_held_median")
        except json.JSONDecodeError:
            issues.append("analysis_summary_parse_error")

    return {
        "schema": CONSENSUS_CONSUMER_AUDIT_SCHEMA,
        "chain_root": str(chain_root),
        "consumer_scope": "consensus_bounded_delta_receipt_audit",
        "pass": not issues,
        "issues": issues,
        "receipts_checked": receipts_checked,
        "analysis_summary_present": analysis_summary_path.exists(),
        "analysis_branch": (
            (analysis_payload.get("branch_precedence_receipt") or {}).get("branch")
            if analysis_payload
            else None
        ),
    }


@dataclass(frozen=True)
class PinnedFile:
    role: str
    rel_path: str
    expected_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_manifest(path: Path | None) -> list[PinnedFile]:
    if path is None:
        return [PinnedFile(role, rel) for role, rel in DEFAULT_FLOOR_PINNED_FILES]
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") or payload
    pinned: list[PinnedFile] = []
    for entry in files:
        if isinstance(entry, Mapping):
            expected = entry.get("sha256")
            pinned.append(
                PinnedFile(
                    str(entry["role"]),
                    str(entry["rel_path"]),
                    expected_sha256=str(expected) if expected else None,
                )
            )
        else:
            raise ValueError("pinned manifest entries must be objects with role/rel_path")
    return pinned


CHAIN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_chain_id(chain_id: str) -> None:
    if not chain_id or not chain_id.strip():
        raise ValueError("chain_id must be a non-empty basename")
    if not CHAIN_ID_PATTERN.fullmatch(chain_id):
        raise ValueError("chain_id must match [A-Za-z0-9][A-Za-z0-9._-]*")


def chain_roots(chain_id: str, creditdir: str = SCIENCE_CHAIN_CREDITDIR) -> tuple[Path, Path]:
    validate_chain_id(chain_id)
    local_root = Path(creditdir) / chain_id
    remote_root = Path(creditdir) / chain_id
    return local_root, remote_root


def verify_head_triple(
    *,
    head_now: str,
    fetch_head: str,
    head_expected: str,
    require_fetch_head: bool = True,
) -> list[str]:
    issues: list[str] = []
    if head_now != head_expected:
        issues.append("HEAD_NOW_MISMATCH")
    if require_fetch_head and fetch_head != head_expected:
        issues.append("FETCH_HEAD_MISMATCH")
    return issues


def hash_pinned_files(repo_root: Path, pinned: Sequence[PinnedFile]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in pinned:
        path = repo_root / item.rel_path
        expected = item.expected_sha256
        if not path.exists():
            rows.append(
                {
                    "role": item.role,
                    "rel_path": item.rel_path,
                    "expected_sha256": expected,
                    "missing": True,
                }
            )
            continue
        producer_sha = sha256_file(path)
        producer_matches_expected: bool | None = None
        if expected is not None:
            producer_matches_expected = producer_sha == expected
        rows.append(
            {
                "role": item.role,
                "rel_path": item.rel_path,
                "expected_sha256": expected,
                "producer_sha256": producer_sha,
                "producer_matches_expected": producer_matches_expected,
                "bytes": path.stat().st_size,
                "missing": False,
            }
        )
    return rows


def verify_pinned_sha_expectations(pinned_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for row in pinned_rows:
        rel = str(row["rel_path"])
        if row.get("missing"):
            mismatches.append(f"missing:{rel}")
            continue
        expected = row.get("expected_sha256")
        if expected is None:
            continue
        producer_sha = row.get("producer_sha256")
        if producer_sha is None:
            mismatches.append(f"missing:{rel}")
            continue
        if producer_sha != expected or row.get("producer_matches_expected") is False:
            mismatches.append(f"producer_pin_mismatch:{rel}")
    return mismatches


def check_pinned_paths_clean(
    repo_root: Path,
    pinned: Sequence[PinnedFile],
    *,
    git_runner: Callable[..., str] | None = None,
) -> list[str]:
    runner = git_runner or run_git
    mismatches: list[str] = []
    for item in pinned:
        rel = item.rel_path.replace("\\", "/")
        output = runner(repo_root, "status", "--porcelain", "--", rel)
        if output.strip():
            mismatches.append(f"pinned_path_dirty:{rel}")
    return mismatches


def _compute_pinned_file_match(row: Mapping[str, Any], *, sync_requested: bool) -> bool:
    if row.get("missing"):
        return False
    expected = row.get("expected_sha256")
    if expected is not None:
        if row.get("producer_matches_expected") is not True:
            return False
        if not sync_requested:
            return True
        return row.get("remote_matches_expected") is True
    if sync_requested:
        return bool(row.get("rsync_ok"))
    return True


def probe_rsync_version() -> str:
    try:
        completed = subprocess.run(
            ["rsync", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    lines = (completed.stdout or "").splitlines()
    return lines[0] if lines else "unknown"


def sync_pinned_files(
    *,
    repo_root: Path,
    remote_repo: str,
    box: str,
    pinned_rows: Sequence[Mapping[str, Any]],
    rsync_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
    remote_sha_runner: Callable[[str, str], str | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    mismatches: list[str] = []
    synced_rows: list[dict[str, Any]] = []
    for row in pinned_rows:
        if row.get("missing"):
            mismatches.append(f"missing:{row['rel_path']}")
            synced_rows.append(dict(row))
            continue
        rel = str(row["rel_path"])
        producer_sha = str(row["producer_sha256"])
        local_path = repo_root / rel
        remote_path = f"{box}:{remote_repo}/{rel}"
        rsync_cmd = [
            "rsync",
            "-az",
            "--mkpath",
            str(local_path),
            remote_path,
        ]
        try:
            rsync_runner(rsync_cmd)
            remote_sha = remote_sha_runner(box, f"{remote_repo}/{rel}")
        except subprocess.CalledProcessError as exc:
            cmd = list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
            mismatches.append(f"rsync_transport_failure:{rel}")
            synced_rows.append(
                {
                    **dict(row),
                    "remote_sha256": None,
                    "rsync_ok": False,
                    "rsync_cmd": cmd,
                    "rsync_exit_code": int(exc.returncode) if exc.returncode is not None else None,
                }
            )
            continue
        ok = remote_sha == producer_sha
        if not ok:
            mismatches.append(f"remote_sha_mismatch:{rel}")
        expected = row.get("expected_sha256")
        remote_matches_expected: bool | None = None
        if expected is not None:
            remote_matches_expected = remote_sha == expected
            if not remote_matches_expected:
                mismatches.append(f"remote_pin_mismatch:{rel}")
        synced_rows.append(
            {
                **dict(row),
                "remote_sha256": remote_sha,
                "remote_matches_expected": remote_matches_expected,
                "rsync_ok": ok,
                "rsync_cmd": rsync_cmd,
            }
        )
    return mismatches, synced_rows


def build_code_currency_manifest(
    *,
    chain_id: str,
    head_expected: str,
    head_now: str,
    fetch_head: str,
    pinned_rows: Sequence[Mapping[str, Any]],
    dry_run: bool,
    sync_requested: bool,
    local_chain_root: Path,
    remote_chain_root: Path,
    remote_repo_root: str,
    mismatches: Sequence[str],
    rsync_version: str | None = None,
    remote_currency_check: str = "enforced",
) -> dict[str, Any]:
    receipt_files: list[dict[str, Any]] = []
    pin_enforcement = False
    for row in pinned_rows:
        entry = dict(row)
        if entry.get("expected_sha256") is not None:
            pin_enforcement = True
        if not sync_requested:
            entry.pop("remote_matches_expected", None)
        entry["match"] = _compute_pinned_file_match(entry, sync_requested=sync_requested)
        receipt_files.append(entry)
    payload: dict[str, Any] = {
        "schema": "hrm158_box_lane_code_currency_preflight/v1",
        "chain_id": chain_id,
        "head_sha": head_expected,
        "head_now": head_now,
        "fetch_head": fetch_head,
        "remote_currency_check": remote_currency_check,
        "dry_run": dry_run,
        "sync_requested": sync_requested,
        "code_currency_pass": not mismatches,
        "mismatches": list(mismatches),
        "n_files": len(pinned_rows),
        "files": receipt_files,
        "producer_host": socket.gethostname(),
        "consumer_host": "box",
        "local_chain_root": str(local_chain_root),
        "remote_chain_root": str(remote_chain_root),
        "remote_repo_root": remote_repo_root,
    }
    if rsync_version is not None:
        payload["rsync_version"] = rsync_version
    if pin_enforcement:
        payload["pin_enforcement"] = True
    return payload


def verify_artifact_manifest(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for entry in entries:
        producer_sha = entry.get("producer_sha256")
        consumer_sha = entry.get("consumer_sha256")
        if producer_sha is None or consumer_sha is None:
            mismatches.append(f"missing_sha:{entry.get('role', '?')}")
            continue
        if producer_sha != consumer_sha:
            mismatches.append(f"artifact_sha_mismatch:{entry.get('role', '?')}")
    return mismatches


def validate_receipt_residency(receipt: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    compute_lane = receipt.get("compute_lane")
    hot_loop = bool(receipt.get("hot_loop_residency_claim"))
    device_residency = bool(receipt.get("device_residency_claim"))
    hot_path_receipt = receipt.get("native_kernelized_hot_path_receipt")
    hot_path_receipt_ok = isinstance(hot_path_receipt, str) and bool(hot_path_receipt.strip())
    if compute_lane == "cpu_trace_analytics" and (hot_loop or device_residency):
        issues.append("cpu_phase_claims_device_or_hot_loop")
    if compute_lane == "cuda_probe" and hot_loop and not device_residency:
        issues.append("cuda_probe_hot_loop_without_device_residency")
    if hot_loop and not hot_path_receipt_ok:
        issues.append("hot_loop_claim_without_native_kernelized_hot_path_receipt")
    if receipt.get("device_residency_not_hot_loop_residency") is not True:
        issues.append("device_residency_not_hot_loop_residency_missing")
    return issues


@dataclass
class ChainCaptureState:
    chain_id: str
    seed: int | None = None
    code_currency_pass: bool = False
    artifact_sha_verified: bool = False
    consumer_started: bool = False
    consumer_terminal_status: str | None = None
    producer_capture_complete_ts: float | None = None
    producer_next_start_ts: float | None = None
    consumer_audit_start_ts: float | None = None
    quarantined: bool = False


@dataclass(frozen=True)
class OverlapVerdict:
    status: str
    issues: tuple[str, ...]
    overlap_seconds: float | None
    pipeline_eligible: bool
    verdict_eligible: bool


def classify_overlap(state: ChainCaptureState) -> OverlapVerdict:
    if state.quarantined:
        return OverlapVerdict(
            status="QUARANTINED_AFTER_CONSUMER_FAIL",
            issues=("prior_consumer_failure",),
            overlap_seconds=None,
            pipeline_eligible=False,
            verdict_eligible=False,
        )
    issues: list[str] = []
    if not state.code_currency_pass:
        issues.append("code_currency_not_passed")
    if not state.artifact_sha_verified:
        issues.append("artifact_sha_not_verified")
    if not state.consumer_started:
        issues.append("consumer_not_started")
    if state.consumer_terminal_status != "pass":
        issues.append("consumer_terminal_not_pass")
    if issues:
        return OverlapVerdict(
            status="INELIGIBLE",
            issues=tuple(issues),
            overlap_seconds=None,
            pipeline_eligible=False,
            verdict_eligible=False,
        )
    if (
        state.producer_next_start_ts is not None
        and state.consumer_audit_start_ts is not None
        and state.consumer_audit_start_ts < state.producer_next_start_ts
    ):
        overlap_seconds = state.producer_next_start_ts - state.consumer_audit_start_ts
        return OverlapVerdict(
            status="OVERLAP",
            issues=(),
            overlap_seconds=overlap_seconds,
            pipeline_eligible=True,
            verdict_eligible=True,
        )
    return OverlapVerdict(
        status="SERIAL_FALLBACK",
        issues=("consumer_started_after_next_capture",),
        overlap_seconds=None,
        pipeline_eligible=False,
        verdict_eligible=False,
    )


def process_science_chain_log(
    lines: Sequence[str],
    *,
    initial_states: Mapping[str, ChainCaptureState] | None = None,
) -> dict[str, ChainCaptureState]:
    states: dict[str, ChainCaptureState] = dict(initial_states or {})
    for line in lines:
        if m := CAPTURE_COMPLETE_RE.search(line):
            chain_id = m.group("chain_id")
            state = states.setdefault(chain_id, ChainCaptureState(chain_id=chain_id))
            state.seed = int(m.group("seed")) if m.group("seed") else state.seed
            if "code_currency_pass=true" in line:
                state.code_currency_pass = True
            if "artifact_sha_verified=true" in line:
                state.artifact_sha_verified = True
            if "ts=" in line:
                ts_token = re.search(r"ts=([0-9.]+)", line)
                if ts_token:
                    state.producer_capture_complete_ts = float(ts_token.group(1))
            continue
        if m := PRODUCER_NEXT_START_RE.search(line):
            chain_id = m.group("chain_id")
            state = states.setdefault(chain_id, ChainCaptureState(chain_id=chain_id))
            ts_token = re.search(r"ts=([0-9.]+)", line)
            if ts_token:
                state.producer_next_start_ts = float(ts_token.group(1))
            continue
        if m := CONSUMER_AUDIT_START_RE.search(line):
            chain_id = m.group("chain_id")
            state = states.setdefault(chain_id, ChainCaptureState(chain_id=chain_id))
            state.consumer_started = True
            ts_token = re.search(r"ts=([0-9.]+)", line)
            if ts_token:
                state.consumer_audit_start_ts = float(ts_token.group(1))
            continue
        if m := CONSUMER_TERMINAL_RE.search(line):
            chain_id = m.group("chain_id")
            state = states.setdefault(chain_id, ChainCaptureState(chain_id=chain_id))
            state.consumer_terminal_status = m.group("status")
            if m.group("status") != "pass":
                for other_id, other in states.items():
                    if other_id != chain_id and other.producer_next_start_ts is not None:
                        other.quarantined = True
    return states


def run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()
