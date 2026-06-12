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


@dataclass(frozen=True)
class PinnedFile:
    role: str
    rel_path: str


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
            pinned.append(PinnedFile(str(entry["role"]), str(entry["rel_path"])))
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
) -> list[str]:
    issues: list[str] = []
    if head_now != head_expected:
        issues.append("HEAD_NOW_MISMATCH")
    if fetch_head != head_expected:
        issues.append("FETCH_HEAD_MISMATCH")
    return issues


def hash_pinned_files(repo_root: Path, pinned: Sequence[PinnedFile]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in pinned:
        path = repo_root / item.rel_path
        if not path.exists():
            rows.append(
                {
                    "role": item.role,
                    "rel_path": item.rel_path,
                    "missing": True,
                }
            )
            continue
        rows.append(
            {
                "role": item.role,
                "rel_path": item.rel_path,
                "producer_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "missing": False,
            }
        )
    return rows


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
        synced_rows.append(
            {
                **dict(row),
                "remote_sha256": remote_sha,
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "hrm158_box_lane_code_currency_preflight/v1",
        "chain_id": chain_id,
        "head_sha": head_expected,
        "head_now": head_now,
        "fetch_head": fetch_head,
        "dry_run": dry_run,
        "sync_requested": sync_requested,
        "code_currency_pass": not mismatches,
        "mismatches": list(mismatches),
        "n_files": len(pinned_rows),
        "files": list(pinned_rows),
        "producer_host": socket.gethostname(),
        "consumer_host": "box",
        "local_chain_root": str(local_chain_root),
        "remote_chain_root": str(remote_chain_root),
        "remote_repo_root": remote_repo_root,
    }
    if rsync_version is not None:
        payload["rsync_version"] = rsync_version
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
