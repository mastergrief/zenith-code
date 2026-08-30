#!/usr/bin/env python3
"""SessionStart hook: durable directive injector (all compaction/restart paths).

Cron/ScheduleWakeup heartbeats are session-only (in-memory) and do NOT survive a
process restart, and compaction can drop standing directives from context. This
on-disk hook is the durable seed for both:

- `startup` / `resume`: re-inject the auto-research charter + loop bootstrap
  (re-arm the 15-min heartbeat cron with Gabe's standing loop directive).
- `compact`: re-inject Gabe's standing directive verbatim.
- `clear` is an intentional reset (left alone).
"""

import json
import os
import subprocess
import sys


GABE_STANDING_LOOP = (
    "auto-research directive - full provenance to all peers (claude, "
    "co_lead, advisor, gate1_audit) - my judgement is deferred to advisor: route, direction, "
    "tier, budget-cap and cost-worth calls are advisor's and BIND, with me "
    "informed-not-asked and retaining override - no need to wait on me at any "
    "gates including pushes and gpu runs - keep going with whatever the "
    "recommended path based on the science - you \"claude\" will take on the "
    "role of test-operator using a shell/monitor with minimal polling until "
    "completion or error for maximum context efficiency - plan-dev is addressed "
    "with ai-room handle \"codex\" - if you are waiting on receipt from a worker "
    "or the gate-1 auditor (codex, codex_co_lead, or gate1_audit) follow up or "
    "diagnose if wedged. The standing "
    "direction-lead peer is addressed in-room as handle \"advisor\" (never as a "
    "subagent); it issues, renews, and kills route licenses and those rulings "
    "bind; it is also my portal - its verbatim-marked relays of my words carry "
    "my provenance. It is never an artifact reviewer and never gate-1 or "
    "gate-2. NOT deferred and never waivable: peer gates and persisted +1 "
    "records, the advisor's artifact bar, destructive/irreversible-action "
    "stops, and my risk-cost-goal seeding."
)

RESUME_DIRECTIVE = (
    "[auto-research orchestration — auto-injected on session start/resume]\n"
    "HRM-158 sub-2-bit arc is active. North star (STANDING CONTEXT, not a stop "
    "condition): achieve HRM-158 sub-2-bit (<2.0 bpw) training & persistent "
    "learning runtime; pragmatic working bar ≤2.5 bpw scale-inclusive "
    "(Gabe-locked two-tier; the term \"sub-2\" stays reserved for actual <2.0; "
    "≤2.5 does not satisfy the strict sub-2-first launch checker).\n"
    "Re-bootstrap the loop now:\n"
    "1) Heartbeat cron is session-local (in-memory; dies on process exit — THIS "
    "hook re-arms it each session). CronList to check; if absent, CronCreate "
    'cron="*/15 * * * *" recurring=true with Gabe\'s standing loop directive as '
    "the prompt (verbatim below). Push wakes (monitors, room messages) stay the "
    "primary signal; the cron is the fallback sweep + worker-wedge check.\n"
    "2) Run ai_room_resume_check and inspect active HRM-158 in_progress tasks — "
    "the board is canonical.\n"
    "3) Roles: codex_co_lead (science/mechanism/pivot/claim-boundary audit), "
    "plan-dev = ai-room handle \"codex\" (plan + implementation; NO subagents — "
    "it performs every edit, validation run, and receipt itself), gate1_audit "
    "(dedicated grok-backed gate-1 auditor: verification + freeze + external verdict "
    "ONLY — never dispatches, frames cures, or authors +1s; advisor topology "
    "ruling 1786976499508-54932946), and CLAUDE "
    "carries the test-operator role directly (shell + Monitor, minimal polling, "
    "foreground-log-watchwrap per shell_monitor.md). Claude orchestrates and "
    "authors all +1/commit/push/launch records; "
    "serialized review routing: worker material receipt → claude only (sink + "
    "framing) → gate1_audit gate-1 verify+freeze → "
    "frozen handoff → co_lead gate-2 validation/diff (LAST gate). Gabe gates are "
    "WAIVED by standing directive (including pushes and GPU runs) — but receipts, "
    "provenance to co_lead, anti-overclaim (candidate != final; no achievement "
    "claim without the matching ledger), and no-.pt-commits stay mandatory. The "
    "commit_precondition_colead_gate hook still fires, and PEER gates are "
    "never waived: it blocks until a fresh co_lead validation/diff PASS "
    "echoes the staged DIFF_DIGEST. Its CO_LEAD_GATE_OVERRIDE is "
    "claude-authorized and never unilateral, and it BINDS an actual co_lead "
    "PASS msg id together with the target-repo path and the 64-hex "
    "DIFF_DIGEST — it is not a route past a missing PASS.\n"
    "Gabe's standing loop directive (cron prompt, verbatim):\n"
    f"{GABE_STANDING_LOOP}\n"
    "Loop model: push (event wake) + 15-min cron sweep. Act once on actionable "
    "items (gate / co_lead audit / redrive / wedge follow-up / park-with-reason); "
    "observe silently otherwise."
)

COMPACT_DIRECTIVE = (
    "[standing directive — auto-injected post-compaction]\n"
    "remember: full provenance to all peers (claude, co_lead, advisor, gate1_audit), "
    "my judgement is deferred to advisor and its route rulings BIND, "
    "no need for AUQ's, "
    "auto-research directive, plan-dev carries plan and implementation via "
    "ai-room handle \"codex\" and spawns no subagents; claude carries the "
    "test-operator role directly "
    "(shell/monitor, minimal polling); gate-1 verification + freeze lives on "
    "the dedicated grok-backed auditor handle \"gate1_audit\" (advisor topology ruling "
    "1786976499508-54932946), with claude as sink/framing and sole +1 author. "
    "No need to recycle workers, their auto compact is fine. "
    "whatever is the recommended path, take it. No need to ask or wait on my "
    "gate — including pushes and gpu runs. If waiting on a worker receipt, "
    "follow up or diagnose if wedged.\n"
    f"Standing loop directive (15-min heartbeat cron prompt): {GABE_STANDING_LOOP}"
)

DIRECTIVES = {
    "startup": RESUME_DIRECTIVE,
    "resume": RESUME_DIRECTIVE,
    "compact": COMPACT_DIRECTIVE,
}

CO_LEAD_DIRECTIVE = (
    "[co_lead session — auto-injected on session start/compact]\n"
    "You are the CC-hosted codex_co_lead role session, NOT the main "
    "auto-research orchestrator; the standing auto-research/test-operator "
    "directive does not apply to you. Your wake route has no codex app-server "
    "(claude-arm lane): after any compaction or restart, wakes may have been "
    "dropped while idle. Drain now: run ai_room_resume_check, then "
    "ai_room_inbox, and answer any pending gate-2 review handoffs addressed "
    "to codex_co_lead before going idle."
)

GATE1_AUDIT_DIRECTIVE = (
    "[gate1_audit session — auto-injected on session start/compact]\n"
    "You are the CC-hosted gate1_audit role session (dedicated grok-backed gate-1 "
    "auditor; advisor topology ruling 1786976499508-54932946), NOT the main "
    "auto-research orchestrator; the standing auto-research/test-operator "
    "directive does not apply to you. Your lane is verification + freeze + "
    "external verdict ONLY, per .claude/agents/gate1-auditor.md. After any "
    "compaction or restart, wakes may have been dropped while idle. Drain "
    "now: run ai_room_resume_check, then ai_room_inbox, and answer any "
    "pending gate-1 verification handoffs addressed to gate1_audit before "
    "going idle."
)


def _session_name() -> str:
    """Hooks get no agent identity; read the tmux session name
    (claude_mcp_<channel>_<handle>_<ts>) for role detection."""
    pane = os.environ.get("TMUX_PANE")
    if not pane or not os.environ.get("TMUX"):
        return ""
    try:
        out = subprocess.run(
            ["tmux", "display-message", "-p", "-t", pane, "#S"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _is_co_lead_session() -> bool:
    """Detect the CC-hosted co_lead session from its tmux session name."""
    return "_codex_co_lead_" in _session_name()


def _is_gate1_audit_session() -> bool:
    """Detect the CC-hosted gate1_audit session from its tmux session name."""
    return "_gate1_audit_" in _session_name()


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except Exception:
        return 0

    if not isinstance(data, dict):
        return 0
    directive = DIRECTIVES.get(data.get("source"))
    if not directive:
        return 0
    if _is_co_lead_session():
        directive = CO_LEAD_DIRECTIVE
    elif _is_gate1_audit_session():
        directive = GATE1_AUDIT_DIRECTIVE

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": directive,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
