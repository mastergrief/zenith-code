#!/usr/bin/env python3
"""SessionStart hook: inject Gabe's standing directive after compaction only."""

import json
import sys


DIRECTIVE = (
    "[standing directive — auto-injected post-compaction]\n"
    "remember: full provenance with you and co_lead, no need for AUQ's, "
    "auto-research directive, plan-dev carries plan, implementation and "
    "executions/runs. No need to recycle workers, their auto compact is fine."
)


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        data = json.loads(raw)
    except Exception:
        return 0

    if not isinstance(data, dict) or data.get("source") != "compact":
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
