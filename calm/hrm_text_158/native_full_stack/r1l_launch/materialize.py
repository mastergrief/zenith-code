"""Phase file materialization + phase_manifest for R1-L launch-prep."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Mapping

from calm.hrm_text_158.native_full_stack.r1l_launch.freeze_digest import (
    content_digest_from_members,
)

PHASE_ORDER = ("S0", "S0b", "S1", "S2", "S3", "S4", "S5")
# Fixture / freeze identity digests over basenames with .sh (FIXTURE_CONTENT_DIGEST authority).
FROZEN_FIXTURE_CONTENT_DIGEST = (
    "fca61e87a6e34a73749080fc83b27f8d6b8991c7bcc82617adad91a6bb1ed859"
)


class PhaseFilePreflightError(Exception):
    """Phase file missing, foreign, or hash-mismatched vs manifest."""

    fail_class = "PHASE_FILE_PREFLIGHT_FAIL"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _basename_digest_map(members_by_phase: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    """Map declared basenames (S0.sh) -> sha256 for CONTENT_DIGEST."""
    out: dict[str, str] = {}
    for ph, rec in members_by_phase.items():
        base = f"{ph}.sh"
        out[base] = str(rec["sha256"])
    return out


def mint_phase_files(
    phases_dir: Path,
    source_shells: Mapping[str, str | bytes],
    *,
    phases: tuple[str, ...] = PHASE_ORDER,
) -> dict:
    """O_EXCL-mint phase scripts + phase_manifest.json under phases_dir.

    CONTENT_DIGEST is over basenames ``{phase}.sh`` (not bare phase ids), matching
    the frozen fixture identity ``FROZEN_FIXTURE_CONTENT_DIGEST`` when sources are
    the DEAD v13 fixture shells.
    """
    phases_dir = Path(phases_dir)
    phases_dir.mkdir(parents=True, exist_ok=True)
    members: dict[str, dict] = {}
    for ph in phases:
        if ph not in source_shells:
            raise KeyError(f"missing source shell for phase {ph}")
        raw = source_shells[ph]
        data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        dest = phases_dir / f"{ph}.sh"
        fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        got = dest.read_bytes()
        if got != data:
            raise PhaseFilePreflightError(f"byte mismatch after mint: {ph}")
        os.chmod(dest, 0o444)
        members[ph] = {
            "path": str(dest.resolve()),
            "sha256": _sha256_bytes(got),
            "mode": 0o444,
            "bytes": len(got),
            "basename": f"{ph}.sh",
        }
    digest_map = _basename_digest_map(members)
    manifest = {
        "schema": "r1l_phase_manifest/v1",
        "count": len(members),
        "phase_order": list(phases),
        "members": members,
        "CONTENT_DIGEST": content_digest_from_members(digest_map),
        "content_digest_key_form": "basename_with_dot_sh",
    }
    man_path = phases_dir / "phase_manifest.json"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(man_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(man_path, 0o444)
    os.chmod(phases_dir, 0o555)
    return manifest


def re_resolve_phase_manifest(
    manifest: Mapping[str, object],
    *,
    expected_phases: tuple[str, ...] = PHASE_ORDER,
) -> None:
    """Verify on-disk files match manifest. Fail-closed on incomplete/extra/renamed members.

    Metadata is required, not advisory: ``count`` and ``phase_order`` must be present
    and exact. A complete member set does not waive either check (member-set success
    must not mask reversed/absent declared execution order).
    """
    members = manifest.get("members")
    if not isinstance(members, dict) or not members:
        raise PhaseFilePreflightError("manifest members missing")

    expected_set = set(expected_phases)
    got_set = set(members.keys())
    if got_set != expected_set:
        raise PhaseFilePreflightError(
            f"member set mismatch got={sorted(got_set)} expected={sorted(expected_set)}"
        )

    # count REQUIRED: present, int, == len(members) == len(expected_phases)
    if "count" not in manifest:
        raise PhaseFilePreflightError("count missing")
    count = manifest["count"]
    if not isinstance(count, int) or isinstance(count, bool):
        raise PhaseFilePreflightError(f"count not int count={count!r}")
    if count != len(members):
        raise PhaseFilePreflightError(
            f"count mismatch count={count} len(members)={len(members)}"
        )
    if count != len(expected_phases):
        raise PhaseFilePreflightError(
            f"count mismatch count={count} len(expected_phases)={len(expected_phases)}"
        )

    # phase_order REQUIRED: present and exactly list(expected_phases), order-sensitive
    if "phase_order" not in manifest:
        raise PhaseFilePreflightError("phase_order missing")
    phase_order = manifest["phase_order"]
    if not isinstance(phase_order, list):
        raise PhaseFilePreflightError(f"phase_order not list phase_order={phase_order!r}")
    expected_list = list(expected_phases)
    if phase_order != expected_list:
        raise PhaseFilePreflightError(
            f"phase_order mismatch got={phase_order!r} expected={expected_list!r}"
        )

    for ph in expected_phases:
        rec = members[ph]
        path = Path(str(rec["path"]))
        if path.name != f"{ph}.sh":
            raise PhaseFilePreflightError(
                f"basename mismatch: phase={ph} path.name={path.name!r}"
            )
        if not path.is_file() or path.is_symlink():
            raise PhaseFilePreflightError(f"missing phase file: {ph} path={path}")
        data = path.read_bytes()
        got = _sha256_bytes(data)
        exp = str(rec["sha256"])
        if got != exp:
            raise PhaseFilePreflightError(f"hash mismatch: {ph} got={got} exp={exp}")
        mode = stat.S_IMODE(path.lstat().st_mode)
        exp_mode = int(rec.get("mode", 0o444))
        if mode != exp_mode:
            raise PhaseFilePreflightError(
                f"mode mismatch: {ph} got={oct(mode)} exp={oct(exp_mode)}"
            )
        if int(rec.get("bytes", -1)) != len(data):
            raise PhaseFilePreflightError(f"size mismatch: {ph}")

    if "CONTENT_DIGEST" in manifest:
        recomputed = content_digest_from_members(_basename_digest_map(members))
        if recomputed != str(manifest["CONTENT_DIGEST"]):
            raise PhaseFilePreflightError("CONTENT_DIGEST recompute mismatch")


def absolute_phase_paths(manifest: Mapping[str, object]) -> dict[str, str]:
    members = manifest["members"]  # type: ignore[index]
    return {str(ph): str(rec["path"]) for ph, rec in members.items()}  # type: ignore[union-attr]
