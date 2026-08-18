#!/usr/bin/env python3
"""SessionStart hook: re-bootstrap the HRM-158 auto-research loop on a new or
resumed session.

Cron/ScheduleWakeup heartbeats are session-only (in-memory) and do NOT survive a
process restart. This on-disk hook is the durable seed: on `startup`/`resume` it
re-injects the auto-research charter and instructs re-arming the single heartbeat
cron + a board check, so the loop auto-bootstraps on every (re)session.

Fires only on source in {startup, resume}; `compact` is handled by
post-compact-directive.py, `clear` is an intentional reset (left alone).
"""

import json
import sys


DIRECTIVE = (
    "[auto-research orchestration — auto-injected on session start/resume]\n"
    "HRM-158 sub-2-bit arc is active. North star (STANDING CONTEXT, not a stop "
    "condition): achieve HRM-158 sub-2-bit training & persistent learning runtime.\n"
    "Re-bootstrap the loop now:\n"
    "1) Heartbeat cron is session-local (in-memory; dies on process exit — THIS "
    "hook re-arms it each session). CronList to check; if absent, CronCreate "
    'cron="17,47 * * * *" recurring=true with the autonomous HRM-158 '
    "orchestration heartbeat prompt (resume_check -> act-once-or-park; observe "
    "silently while a worker runs; do not restate the goal or burn tokens).\n"
    "2) Run ai_room_resume_check and inspect active HRM-158 in_progress tasks — "
    "the board is canonical.\n"
    "3) Active codex roles: codex_co_lead (science/mechanism/pivot/claim-boundary), "
    "plan-dev (plan + implementation), test-operator (formal training/proof/test "
    "RUNS). Claude gates/orchestrates and routes mutating work to the current "
    "plan-dev worker (resolve the live handle from the board / peer_status, do not "
    "assume a prior session's handle). Serialized review routing: worker material "
    "receipt → claude gate-1 only → frozen handoff → co_lead gate-2 validation/diff "
    "(LAST gate). Commit precondition: commit_precondition_colead_gate requires a "
    "co_lead validation/diff PASS echoing the staged DIFF_DIGEST. Material gates "
    "(implement/commit/push/launch) stay explicit, persisted, and auditable; "
    "GPU/dyn200 autonomously gateable but still need receipts; no .pt commits; "
    "anti-overclaim (candidate != final; never claim achieved without the matching "
    "ledger).\n"
    "Loop model: push (event wake) + one ~30-min cron heartbeat. Act once on "
    "actionable items (gate / co_lead audit / redrive / park-with-reason); "
    "observe silently otherwise."
)

_SOURCES = {"startup", "resume"}


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except Exception:
        return 0

    if not isinstance(data, dict) or data.get("source") not in _SOURCES:
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": DIRECTIVE,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
