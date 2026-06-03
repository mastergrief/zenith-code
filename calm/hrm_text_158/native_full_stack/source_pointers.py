"""Hashed read-only source anchors for the Phase-0 native stack scaffold."""
from __future__ import annotations

from dataclasses import dataclass


ACTIVE_HRM_REPO_ROOT = "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"


@dataclass(frozen=True)
class SourcePointer:
    """A static source anchor, not a runtime dependency."""

    label: str
    root: str
    relative_path: str
    expected_sha256: str
    reason: str
    sha_kind: str = "file-content-sha256"
    runtime_dependency: bool = False
    lifecycle: str = "phase0_snapshot_anchor"
    anchored_as_of: str = ""
    reanchor_note: str = ""
    implementation_role: str = "source_pointer_only"

    @property
    def absolute_path(self) -> str:
        return f"{self.root.rstrip('/')}/{self.relative_path.lstrip('/')}"

    def validate_static(self) -> None:
        if self.sha_kind != "file-content-sha256":
            raise ValueError(f"{self.label}: sha_kind must be file-content-sha256")
        if self.runtime_dependency:
            raise ValueError(f"{self.label}: source pointers must not be runtime dependencies")
        if len(self.expected_sha256) != 64:
            raise ValueError(f"{self.label}: expected_sha256 must be 64 hex chars")
        int(self.expected_sha256, 16)


LIVE_S1_TRAINER_POINTER = SourcePointer(
    label="live_s1_c1353fd5_trainer",
    root=(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "1780347615017-1538f834/science_work/resume_global_s250_work"
    ),
    relative_path="transient_fp_credit_science_train.py",
    expected_sha256="c1353fd5837dd7661b0ef7e9fd87b55454c406ef3778f7a4fc004abcdc4e02ea",
    reason=(
        "Phase-0 read-only source pointer for authoritative q:int8, "
        "vote_acc:int16, frozen_scale:float32, attribution hooks, and "
        "FP-exception semantics observed in the live S1 trainer."
    ),
    anchored_as_of="S1 run 1780347615017-1538f834",
    reanchor_note=(
        "Post-terminal native work must re-read and refresh this pointer if the "
        "creditdir trainer file-content sha256 changes."
    ),
)

OLDER_7206_NON_ANCHOR_POINTER = SourcePointer(
    label="older_7206_non_anchor_trainer",
    root="/home/gabe/claw-code-creditdir/transient_fp_credit/1780327234329-cb920b83",
    relative_path="transient_fp_credit_science_train.py",
    expected_sha256="7206be4e7f020526756eceffd82267dbe3da293ba03442ec53350bd0c7e5c28a",
    reason=(
        "Historical structural background only. This file is not the live S1 "
        "anchor and must not define Phase-0 FP-exception, attribution, or "
        "state semantics."
    ),
    lifecycle="historical_non_anchor",
    anchored_as_of="older run 1780327234329-cb920b83",
    reanchor_note="Do not promote without a new plan gate and live-source read receipt.",
)

PHASE0_SOURCE_POINTERS = (LIVE_S1_TRAINER_POINTER,)
HISTORICAL_NON_ANCHOR_POINTERS = (OLDER_7206_NON_ANCHOR_POINTER,)
