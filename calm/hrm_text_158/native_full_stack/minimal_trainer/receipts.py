#!/usr/bin/env python3
"""Minimal trainer receipt projection + pin rehash (S0)."""

# INVARIANT: this module never imports bounded_delta_loop*.py.

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

VOLATILE_KEYS = frozenset({"duration_seconds", "cuda_memory_snapshots"})
NINE = (
    "step_reports",
    "updater_config",
    "states",
    "audit_reports",
    "stop_reason",
    "steps_completed",
    "b2_full_verdict_state",
    "b2b_capture_receipt",
    "grad_proxy_ingress_crossing_eligible_count_by_step",
)

_SHA_SURFACE = {
    "step_reports": "step_reports_sha256",
    "updater_config": "updater_config_sha256",
    "audit_reports": "audit_reports_sha256",
    "b2_full_verdict_state": "b2_full_verdict_state_sha256",
    "b2b_capture_receipt": "b2b_capture_receipt_sha256",
    "grad_proxy_ingress_crossing_eligible_count_by_step": (
        "grad_proxy_ingress_crossing_eligible_count_by_step_sha256"
    ),
}


def _tensor_sha(t: Any) -> str:
    import torch

    arr = t.detach().contiguous().cpu()
    if arr.dtype == torch.bfloat16:
        arr = arr.to(torch.float32)
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def _states_hash(states: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(states):
        st = states[key]
        rec: dict[str, Any] = {
            "q": _tensor_sha(st.q_levels),
            "scale": _tensor_sha(st.frozen_scale),
        }
        if st.exact_accumulator_shadow is not None:
            rec["acc"] = _tensor_sha(st.exact_accumulator_shadow)
        out[str(key)] = rec
    return out


def _strip_volatile(obj: Any, dropped: list[str]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in VOLATILE_KEYS:
                dropped.append(k)
                continue
            out[k] = _strip_volatile(v, dropped)
        return out
    if isinstance(obj, list):
        return [_strip_volatile(x, dropped) for x in obj]
    if isinstance(obj, tuple):
        return [_strip_volatile(x, dropped) for x in obj]
    return obj


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _json_sha(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


strip_volatile = _strip_volatile
canonical = _canonical
json_sha = _json_sha


def _refuse_empty(obj: Any, *, label: str) -> None:
    if not obj:
        raise ValueError(f"empty-denominator: {label} is empty; refusing")


def project_named_surfaces(named: dict) -> tuple[dict, list]:
    _refuse_empty(named, label="named surfaces")
    dropped: list[str] = []
    projected = {name: strip_volatile(named[name], dropped) for name in NINE}
    if "duration_seconds" not in dropped:
        raise ValueError(
            "projection empty-denominator: duration_seconds not dropped from any "
            "of 9 surfaces; refusing receipt"
        )
    return projected, dropped


def compare_surfaces(
    baseline_surfaces: dict, candidate_surfaces: dict
) -> dict[str, Any]:
    _refuse_empty(baseline_surfaces, label="baseline_surfaces")
    _refuse_empty(candidate_surfaces, label="candidate_surfaces")
    rows: list[dict[str, Any]] = []
    for name in NINE:
        if name == "states":
            b = baseline_surfaces["states"]
            c = candidate_surfaces["states"]
        elif name in ("stop_reason", "steps_completed"):
            b = baseline_surfaces[name]
            c = candidate_surfaces[name]
        else:
            key = _SHA_SURFACE[name]
            b = baseline_surfaces[key]
            c = candidate_surfaces[key]
        rows.append({"name": name, "match": b == c, "baseline": b, "candidate": c})
    return {"rows": rows}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256sum_path(path: Path) -> str:
    proc = subprocess.run(
        ["sha256sum", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return ""
    return proc.stdout.split()[0]


def _git_show_bytes(repo_root: Path, rel: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:{rel}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"git show HEAD:{rel} failed rc={proc.returncode}; refusing"
        )
    return proc.stdout


def rehash_pins(baseline_receipt: dict, *, repo_root: Path) -> dict[str, Any]:
    source_hashes = baseline_receipt.get("source_hashes")
    _refuse_empty(source_hashes, label="source_hashes")
    out: dict[str, Any] = {}
    for key, entry in source_hashes.items():
        rel = entry["rel"]
        if key == "packet" or (isinstance(rel, str) and rel.startswith("/")):
            path = Path(rel)
            data = path.read_bytes()
            digest = _sha256_bytes(data)
            sum_line = _sha256sum_path(path)
            two_instrument_equal = digest == sum_line and bool(sum_line)
            matches_receipt = digest == entry["hashlib"] and sum_line == entry["sha256sum"]
            out[key] = {
                "hashlib": digest,
                "sha256sum": sum_line,
                "two_instrument_equal": two_instrument_equal,
                "matches_receipt": matches_receipt,
            }
            continue
        # Guard read set (statically enumerable): git show HEAD:{rel};
        # (repo_root / rel).read_bytes().
        head_bytes = _git_show_bytes(repo_root, rel)
        head_digest = _sha256_bytes(head_bytes)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(head_bytes)
            tmp_path = Path(tmp.name)
        try:
            head_sum = _sha256sum_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        worktree_path = repo_root / rel
        try:
            worktree_bytes = worktree_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"worktree read {worktree_path} failed: {exc}; refusing"
            ) from exc
        worktree_digest = _sha256_bytes(worktree_bytes)
        worktree_sum = _sha256sum_path(worktree_path)
        two_instrument_equal = (
            head_digest == head_sum
            and worktree_digest == worktree_sum
            and bool(head_sum)
            and bool(worktree_sum)
        )
        matches_receipt = (
            head_digest == entry["hashlib"]
            and head_sum == entry["sha256sum"]
            and worktree_digest == entry["hashlib"]
            and worktree_sum == entry["sha256sum"]
        )
        out[key] = {
            "hashlib": head_digest,
            "sha256sum": head_sum,
            "worktree_hashlib": worktree_digest,
            "worktree_sha256sum": worktree_sum,
            "head_eq_worktree": head_digest == worktree_digest,
            "two_instrument_equal": two_instrument_equal,
            "matches_receipt": matches_receipt,
        }
    return out


def pins_ok(result: dict) -> bool:
    # Bool consumer of the empty-denominator law. Producers raise via
    # _refuse_empty; this predicate cannot raise. Same law, different type.
    if not result:
        return False
    return all(v.get("matches_receipt") is True for v in result.values())


def load_baseline(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="minimal trainer receipt pins")
    parser.add_argument("--pins", action="store_true", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = load_baseline(args.baseline)
        result = rehash_pins(receipt, repo_root=args.repo_root)
    except Exception as exc:
        print(f"refusing: {type(exc).__name__}: {exc}")
        return 2
    if pins_ok(result):
        return 0
    failing = [k for k, v in result.items() if not v.get("matches_receipt")]
    print("pin mismatch:", " ".join(failing) if failing else "(empty pin set)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
