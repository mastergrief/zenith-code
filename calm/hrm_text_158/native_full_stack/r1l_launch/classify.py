"""Pure terminal classification reducer for R1-L spawn topology (c).

No launch, GPU, or filesystem glue.
Absence of required evidence is never treated as success.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

EXIT_RE = re.compile(r"\[EXIT rc=(-?\d+)(?:\s|\])")


@dataclass(frozen=True)
class TerminalObservation:
    exit_rc: Optional[int]
    runner_pass_count: int
    last_nonempty_line: Optional[str]
    actual_log_sha256: Optional[str]
    projected_log_sha256: Optional[str]
    stderr_text: str = ""
    stdout_text: str = ""
    phase_file_preflight_ok: Optional[bool] = None


@dataclass(frozen=True)
class TerminalVerdict:
    status: str  # PASS | FAIL
    fail_class: Optional[str]
    reasons: tuple[str, ...]


def parse_exit_rc(*texts: str) -> Optional[int]:
    for text in texts:
        matches = list(EXIT_RE.finditer(text or ""))
        if matches:
            return int(matches[-1].group(1))
    return None


def count_runner_pass(log_text: str) -> int:
    return sum(1 for ln in (log_text or "").splitlines() if ln.strip() == "RUNNER_PASS")


def last_nonempty_line(log_text: str) -> Optional[str]:
    lines = [ln for ln in (log_text or "").splitlines() if ln.strip() != ""]
    return lines[-1] if lines else None


def classify_terminal(obs: TerminalObservation) -> TerminalVerdict:
    blob = (obs.stderr_text or "") + "\n" + (obs.stdout_text or "")
    reasons: list[str] = []

    # Preflight: only explicit True is success evidence; False and None both fail.
    if obs.phase_file_preflight_ok is False:
        return TerminalVerdict("FAIL", "PHASE_FILE_PREFLIGHT_FAIL", ("phase file preflight failed",))
    if obs.phase_file_preflight_ok is None:
        return TerminalVerdict(
            "FAIL",
            "PHASE_FILE_PREFLIGHT_ABSENT",
            ("phase_file_preflight_ok is None — absence is not success",),
        )

    if "PHASE_BUDGET_BREACH" in blob:
        return TerminalVerdict("FAIL", "PHASE_BUDGET_BREACH", ("per-phase timeout fired",))

    if "TERMINAL_MARKER_UNWRITABLE" in blob or "S5_TERMINAL_MARKER_UNWRITABLE" in blob:
        if obs.exit_rc is None or obs.exit_rc != 0 or obs.runner_pass_count == 0:
            return TerminalVerdict(
                "FAIL",
                "TERMINAL_MARKER_UNWRITABLE",
                ("append marker unwritable; success authority absent",),
            )

    if obs.exit_rc is None:
        return TerminalVerdict("FAIL", "WATCH_WRAP_CHILD_NONZERO_EXIT", ("missing [EXIT rc=]",))

    if obs.exit_rc != 0:
        cls = "WATCH_WRAP_CHILD_NONZERO_EXIT"
        if "PHASE_BUDGET_BREACH" in blob:
            cls = "PHASE_BUDGET_BREACH"
        reasons.append(f"exit_rc={obs.exit_rc}")
        if obs.runner_pass_count == 0:
            reasons.append("zero RUNNER_PASS")
        return TerminalVerdict("FAIL", cls, tuple(reasons))

    # exit_rc == 0 path — every required field must be present evidence
    if obs.runner_pass_count != 1:
        return TerminalVerdict(
            "FAIL",
            "TERMINAL_LOG_VERIFY_FAIL",
            (f"RUNNER_PASS count={obs.runner_pass_count} want 1",),
        )
    if obs.last_nonempty_line != "RUNNER_PASS":
        return TerminalVerdict(
            "FAIL",
            "TERMINAL_LOG_VERIFY_FAIL",
            (f"last line={obs.last_nonempty_line!r} want RUNNER_PASS",),
        )
    if obs.actual_log_sha256 is None:
        return TerminalVerdict(
            "FAIL",
            "TERMINAL_LOG_DIGEST_ABSENT",
            ("actual_log_sha256 is None — absence is not success",),
        )
    if obs.projected_log_sha256 is None:
        return TerminalVerdict(
            "FAIL",
            "TERMINAL_LOG_DIGEST_ABSENT",
            ("projected_log_sha256 is None — absence is not success",),
        )
    if obs.actual_log_sha256 != obs.projected_log_sha256:
        return TerminalVerdict(
            "FAIL",
            "TERMINAL_LOG_VERIFY_FAIL",
            ("actual != projected log digest",),
        )
    return TerminalVerdict(
        "PASS",
        None,
        ("exit_rc=0", "RUNNER_PASS count=1", "last line ok", "digests equal", "preflight True"),
    )
