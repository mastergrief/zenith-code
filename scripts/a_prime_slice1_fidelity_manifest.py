"""Manifest candidate build + atomic publish for A′ slice1. Wrapper-only writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice1_fidelity_core import FINAL_BRANCHES, sha256_file

TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
MISMATCH_DIAGNOSTIC_NAME = "mismatch_diagnostic.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_command_statuses(run_root: Path) -> list[dict[str, Any]]:
    status_dir = run_root / "command_status"
    if not status_dir.is_dir():
        return []
    rows = []
    for p in sorted(status_dir.glob("*.json")):
        try:
            row = load_json(p)
            row["_status_path"] = str(p.relative_to(run_root))
            row["_status_sha256"] = sha256_file(p)
            rows.append(row)
        except Exception as e:
            rows.append(
                {
                    "_status_path": str(p.relative_to(run_root)),
                    "error": str(e),
                    "rc": 1,
                    "_status_sha256": sha256_file(p) if p.is_file() else None,
                }
            )
    return rows


def actual_status_set(run_root: Path) -> list[str]:
    status_dir = run_root / "command_status"
    if not status_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(run_root)) for p in status_dir.glob("*.json") if p.is_file()
    )


def collect_outputs(run_root: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    status_dir = run_root / "command_status"
    if status_dir.is_dir():
        for p in sorted(status_dir.glob("*.json")):
            outputs[str(p.relative_to(run_root))] = sha256_file(p)
    for rel in (
        "launch_preflight.json",
        "arm_DENSE_INT16_BASELINE/metrics.json",
        "arm_NONDENSE_CANDIDATE/metrics.json",
        "paired_probe_report.json",
        "non_q_bpw_receipt.json",
        "branch_verdict.json",
        "terminal_receipt.json",
    ):
        p = run_root / rel
        if p.is_file():
            outputs[rel] = sha256_file(p)
    return outputs


def build_manifest_payload(
    run_root: Path,
    *,
    branch: str,
    failing_status: Mapping[str, Any] | None = None,
    synthetic: bool = False,
    run_root_abs: str | None = None,
) -> dict[str, Any]:
    outputs = collect_outputs(run_root)
    ess = actual_status_set(run_root)
    return {
        "schema": "a_prime_slice1_terminal_manifest/v3",
        "branch": branch,
        "terminal_authority": "manifest+marker",
        "run_root": run_root_abs or str(run_root.resolve()),
        "outputs": outputs,
        "expected_status_set": ess,
        "failing_status": (
            {
                "path": failing_status.get("_status_path"),
                "sha256": failing_status.get("_status_sha256"),
                "rc": failing_status.get("rc"),
                "name": failing_status.get("name"),
            }
            if failing_status
            else None
        ),
        "synthetic": synthetic,
    }


def write_manifest_candidate(run_root: Path, payload: Mapping[str, Any]) -> Path:
    tmp = run_root / f"{TERMINAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def publish_manifest_atomic(temp_path: Path, final_path: Path) -> None:
    """Same-FS atomic rename. Caller must have fully validated candidate."""
    os.replace(str(temp_path), str(final_path))


def verify_published_manifest(
    final_path: Path,
    *,
    receipt_branch: str,
    expected_ess: list[str],
    expected_hashes: Mapping[str, str],
    candidate_sha256: str | None = None,
) -> tuple[bool, str]:
    if not final_path.is_file():
        return False, "final_manifest_absent"
    # published-byte identity with pre-rename candidate (when provided)
    if candidate_sha256 is not None:
        final_sha = sha256_file(final_path)
        if final_sha != candidate_sha256:
            return False, f"candidate_byte_mismatch:{final_sha}!={candidate_sha256}"
    try:
        payload = load_json(final_path)
    except Exception as e:
        return False, f"final_manifest_unparseable:{e}"
    if payload.get("branch") != receipt_branch:
        return False, f"branch {payload.get('branch')!r}!={receipt_branch!r}"
    if payload.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority_missing_or_wrong"
    if list(payload.get("expected_status_set") or []) != list(expected_ess):
        return False, "ess_mismatch"
    outs = payload.get("outputs") or {}
    for rel, exp in expected_hashes.items():
        if outs.get(rel) != exp:
            return False, f"hash_mismatch:{rel}"
    # re-hash disk outputs
    run_root = final_path.parent
    for rel, exp in outs.items():
        p = run_root / rel
        if not p.is_file():
            return False, f"missing_output:{rel}"
        if sha256_file(p) != exp:
            return False, f"stale_output_hash:{rel}"
    return True, "ok"


def snapshot_run_root(run_root: Path) -> dict[str, dict[str, int | str]]:
    """relpath -> {sha256, mtime_ns} for every file under run_root.

    Full zero-post-publish-writes denominator: inventory + content hash + mtime_ns.
    """
    out: dict[str, dict[str, int | str]] = {}
    for p in sorted(run_root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(run_root))] = {
                "sha256": sha256_file(p),
                "mtime_ns": int(st.st_mtime_ns),
            }
    return out


def write_mismatch_diagnostic(run_root: Path, details: Mapping[str, Any]) -> Path:
    """Verdict-free diagnostic — no FINAL_BRANCHES value as verdict branch."""
    path = run_root / MISMATCH_DIAGNOSTIC_NAME
    payload = {
        "schema": "a_prime_slice1_mismatch_diagnostic/v1",
        "verdict_free": True,
        "details": dict(details),
    }
    # deliberately no 'branch' field carrying FINAL_BRANCHES
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_candidate_against_receipt(
    payload: Mapping[str, Any],
    *,
    receipt_branch: str,
    rc: int,
    run_root: Path,
) -> tuple[bool, str]:
    from scripts.a_prime_slice1_fidelity_core import branch_matches_rc

    if receipt_branch not in FINAL_BRANCHES:
        return False, f"branch_not_final:{receipt_branch!r}"
    if not branch_matches_rc(receipt_branch, rc):
        return False, f"rc_branch_mismatch:rc={rc}:branch={receipt_branch}"
    if payload.get("branch") != receipt_branch:
        return False, "candidate_branch_ne_receipt"
    ess = list(payload.get("expected_status_set") or [])
    actual = actual_status_set(run_root)
    if ess != actual:
        return False, f"ess_ne_actual:{ess!r}!={actual!r}"
    outs = payload.get("outputs") or {}
    for rel, exp in outs.items():
        p = run_root / rel
        if not p.is_file() or sha256_file(p) != exp:
            return False, f"hash_fail:{rel}"
    return True, "ok"


def make_fixture_receipt(path: Path, final_fracs: dict[str, str]) -> None:
    """Synthetic prior_audit receipt for dry-synthetic-final arms."""
    pins = {
        "L0b": {
            "expected_count": 230,
            "expected_hash16": "89174273d21845bc",
            "builder_path": "x",
            "support_role": "true_prior",
        },
        "math_a0": {
            "expected_count": 1255,
            "expected_hash16": "56e64266357b793d",
            "builder_path": "y",
            "support_role": "true_prior",
        },
    }
    per_support: dict[str, Any] = {}
    support_proofs: dict[str, Any] = {}
    for name, pin in pins.items():
        frac = final_fracs[name]
        support_proofs[name] = {
            "support_hash16": pin["expected_hash16"],
            "expected_count": pin["expected_count"],
            "builder_path": pin["builder_path"],
            "support_role": pin["support_role"],
        }
        per_support[name] = {
            "support_hash16": pin["expected_hash16"],
            "support_rows_expected": pin["expected_count"],
            "builder_path": pin["builder_path"],
            "support_role": pin["support_role"],
            "start": {
                "strict_exact": f"{pin['expected_count']}/{pin['expected_count']}",
                "parsed_exact": f"{pin['expected_count']}/{pin['expected_count']}",
                "duration_seconds": 0.1,
            },
            "final": {
                "strict_exact": frac,
                "parsed_exact": frac,
                "duration_seconds": 0.1,
            },
            "delta": {},
        }
    prior = {
        "schema": "b1_prior_audit/v1",
        "enabled": True,
        "requested_supports": list(pins),
        "default_off": False,
        "support_proofs": support_proofs,
        "per_support": per_support,
        "start_reports": {},
        "final_reports": {},
        "deltas": {},
        "total_duration_seconds": 0.4,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"prior_audit": prior, "steps_completed": 50, "synthetic": True}, indent=2
        )
        + "\n"
    )


def finalize(
    run_root: Path,
    *,
    reduce_rc: int,
    synthetic: bool = False,
    inject_candidate_branch: str | None = None,
    inject_postpub_fail: bool = False,
    postpub_snapshot_out: dict[str, str] | None = None,
) -> int:
    """Candidate validate → atomic publish → byte-identity re-read → PACKET_TERMINAL.

    Validation authority is ALWAYS receipt["branch"] — inject_candidate_branch may
    only perturb the CANDIDATE so the equality chain catches the mismatch.

    Returns wrapper rc. State P: final ABSENT, nonzero, no marker.
    State Q: final MAY EXIST, nonzero, no marker, zero post-publish writes.
    """
    import sys

    run_root_abs = str(run_root.resolve())
    receipt_path = run_root / "terminal_receipt.json"
    final_path = run_root / TERMINAL_MANIFEST_NAME

    if not receipt_path.is_file():
        write_mismatch_diagnostic(
            run_root, {"reason": "missing_terminal_receipt", "reduce_rc": reduce_rc}
        )
        return 2

    receipt = load_json(receipt_path)
    # force_branch never replaces receipt authority
    branch = receipt.get("branch")
    if not isinstance(branch, str):
        write_mismatch_diagnostic(
            run_root, {"reason": "receipt_branch_missing", "reduce_rc": reduce_rc}
        )
        return 2

    # run_root mismatch → STATE P (no normalization rewrite)
    if receipt.get("run_root") != run_root_abs:
        write_mismatch_diagnostic(
            run_root,
            {
                "reason": "receipt_run_root_mismatch",
                "receipt_run_root": receipt.get("run_root"),
                "expected": run_root_abs,
                "state": "P",
            },
        )
        return 2

    # candidate may be perturbed for injection; receipt branch is validation authority
    cand_branch = (
        inject_candidate_branch if inject_candidate_branch is not None else branch
    )
    payload = build_manifest_payload(
        run_root,
        branch=cand_branch,
        synthetic=synthetic,
        run_root_abs=run_root_abs,
    )

    ok, reason = validate_candidate_against_receipt(
        payload, receipt_branch=branch, rc=reduce_rc, run_root=run_root
    )
    if not ok:
        write_mismatch_diagnostic(
            run_root,
            {
                "reason": reason,
                "receipt_branch": branch,
                "reduce_rc": reduce_rc,
                "state": "P",
            },
        )
        for p in run_root.glob(f"{TERMINAL_MANIFEST_NAME}.tmp.*"):
            try:
                p.unlink()
            except OSError:
                pass
        return 2 if reduce_rc in (0, 2) else (reduce_rc if reduce_rc else 2)

    tmp = write_manifest_candidate(run_root, payload)
    candidate_sha = sha256_file(tmp)
    try:
        publish_manifest_atomic(tmp, final_path)
    except OSError as e:
        write_mismatch_diagnostic(
            run_root, {"reason": f"publish_failed:{e}", "state": "P"}
        )
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if tmp.exists():
                tmp.unlink()
        return 2

    # capture post-rename inventory for zero-post-publish-writes bind
    post_rename_snap = snapshot_run_root(run_root)
    if postpub_snapshot_out is not None:
        postpub_snapshot_out.clear()
        postpub_snapshot_out.update(post_rename_snap)

    # Observer path OUTSIDE run_root — written on BOTH success and postpub-fail
    # so F10/F_postpub can bind post-rename inventory == post-exit inventory.
    snap_path = os.environ.get("A_PRIME_POSTPUB_SNAP_PATH", "").strip()
    if snap_path:
        Path(snap_path).write_text(
            json.dumps(post_rename_snap, indent=2, sort_keys=True) + "\n"
        )

    if inject_postpub_fail:
        # STATE Q: published, skip marker, nonzero, zero further writes into run_root.
        print(
            f"INCOMPLETE_FINALIZATION postpub_inject branch={branch}",
            file=sys.stderr,
            flush=True,
        )
        return 4

    ess = list(payload["expected_status_set"])
    hashes = dict(payload["outputs"])
    vok, vreason = verify_published_manifest(
        final_path,
        receipt_branch=branch,
        expected_ess=ess,
        expected_hashes=hashes,
        candidate_sha256=candidate_sha,
    )
    if not vok:
        print(f"INCOMPLETE_FINALIZATION {vreason}", file=sys.stderr, flush=True)
        return 4

    # verify-then-marker: only after published-byte verification
    print(f"PACKET_TERMINAL {branch}", flush=True)
    if branch == "LIVENESS_FAIL":
        return 3
    if branch == "INSTRUMENT_GAP":
        return 2
    return 0
