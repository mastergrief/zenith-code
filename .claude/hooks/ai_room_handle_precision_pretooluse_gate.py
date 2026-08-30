#!/usr/bin/env python3
"""PreToolUse hook: handle-precision gate on ai_room_post / ai_room_reply.

A repo-local, backend-agnostic reimplementation of the user-scope
``~/.ai-room/.claude/hooks/ai_room_handle_precision_pretooluse_gate.py``.
That hook is GLM-backend-scoped and draws its "known" set from the live
lease-stem zoo — which left two gaps that bit us in practice:

  1. ``codex`` is itself a live lease stem, so ``to=codex`` was silently
     accepted as a *valid* handle (passthrough) even when the intended
     recipient was ``codex_co_lead``. Wrong recipient, no warning.
  2. ``codex_co_lead`` is NOT a lease stem (only ``codex_co_lead_eval`` is),
     so ``to=codex_co`` / ``to=codex_co_lead`` would have been *mis-repaired*
     to ``codex_co_lead_eval`` — actively routed to the wrong peer.

This version adds a CANONICAL ROLE-HANDLE LAYER that takes precedence over
the lease zoo: ``codex_co_lead``, ``codex`` (plan-dev), ``claude``, ``gabe``,
``ai_room_supervisor``. Truncations of a role handle repair to the role,
never to an eval/lease stem. Role-aware ambiguity (e.g. ``cod`` matches both
``codex`` and ``codex_co_lead``) is denied, not silently routed.

Behaviour:
  - exact canonical role handle            -> passthrough
  - unambiguous prefix of ONE role handle   -> repair to that role
  - prefix of >=2 role handles              -> deny (ambiguous)
  - exact lease stem / unambiguous lease    -> passthrough / repair (zoo rules)
  - prefix matching role(s) AND lease(s)    -> role layer wins (repair to role
                                               if unambiguous, else deny)
  - genuine unknown handle (no matches)     -> passthrough (MCP validates shape)
  - non-str ``requires_response_from``, empty rrf, non-str list item -> deny
  - post-repair ``to`` != ``requires_response_from`` -> deny

Kill-switch: ``AI_ROOM_HANDLE_HOOK_DISABLE=1`` or a sibling
``.handle_hook_disabled`` sentinel file. Parse errors fail-open.

Wired FIRST in the ai_room_post|reply PreToolUse chain so the handle is
canonicalized before the wake-pairing / child-boundary / cross-thread gates
inspect it.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
SENTINEL = HOOK_DIR / ".handle_hook_disabled"
DEFAULT_ROOM_DIR = "/home/gabe/.ai-room"
DEFAULT_CHANNEL = "ai-room"

# Canonical role handles — the people/roles we actually address. These take
# precedence over the lease-stem zoo so a truncated role handle repairs to
# the role, not to an eval/scratch lease that happens to share a prefix.
CANONICAL_ROLES = frozenset(
    {
        "claude",            # this orchestrator
        "codex",             # plan-dev worker (ai-room handle "codex")
        "codex_co_lead",     # gate-2 reviewer / science-boundary owner
        "advisor",           # direction lead (interactive Fable session)
        "gate1_audit",       # gate-1 verification + freeze auditor
        "gabe",              # human direction owner
        "ai_room_supervisor",# pending-response watchdog (sender-only)
    }
)
# Adding advisor/gate1_audit makes two single-letter prefixes ambiguous that
# previously repaired: "a" (advisor | ai_room_supervisor) and "g" (gabe |
# gate1_audit). Denying those is the design — a prefix that names two roles
# is exactly the silent mis-route this layer exists to stop.

MATCHED_TOOLS = frozenset(
    {
        "mcp__ai-room__ai_room_post",
        "mcp__ai-room__ai_room_reply",
        # Task tools carry a `to` notify target that bypassed the gate: a
        # truncated handle in task_create/task_update notify was delivered
        # unrepaired (observed: task_update to=neural / to=plan-dev).
        # The `owner` field is NOT guarded — `to` only.
        "mcp__ai-room__ai_room_task_create",
        "mcp__ai-room__ai_room_task_update",
        "mcp__ai_room__ai_room_task_create",
        "mcp__ai_room__ai_room_task_update",
    }
)


def _disabled() -> bool:
    if os.environ.get("AI_ROOM_HANDLE_HOOK_DISABLE") == "1":
        return True
    return SENTINEL.is_file()


def _channel_leases_dir() -> Path:
    room_dir = os.environ.get("AI_ROOM_DIR", DEFAULT_ROOM_DIR)
    channel = os.environ.get("AI_ROOM_CHANNEL", DEFAULT_CHANNEL)
    return Path(room_dir) / "channels" / channel / "leases"


def _live_lease_handles() -> set[str]:
    handles: set[str] = set()
    leases = _channel_leases_dir()
    if not leases.is_dir():
        return handles
    for entry in leases.glob("*.json"):
        handles.add(entry.stem)
    return handles


def _known_handles() -> set[str]:
    """Roles first, then the lease zoo. Roles shadow same-named leases."""
    return set(CANONICAL_ROLES) | _live_lease_handles()


def _role_matches(value: str) -> list[str]:
    """Canonical roles that ``value`` is a strict prefix of (excluding exact)."""
    return sorted(r for r in CANONICAL_ROLES if r.startswith(value) and r != value)


def _lease_matches(value: str, known: set[str]) -> list[str]:
    """Lease stems (non-role) that ``value`` is a strict prefix of."""
    return sorted(
        h for h in known
        if h not in CANONICAL_ROLES and h.startswith(value) and h != value
    )


def _classify_unknown(value: str, known: set[str]) -> tuple[str, str | None]:
    """Classify an unknown handle string.

    Returns (action, detail). action in {repair, deny, passthrough}.
    Role layer wins over the lease zoo: if the prefix matches any role(s),
    the role layer decides (repair if exactly one role, deny if >=2) and
    leases are ignored — so ``codex_co`` -> ``codex_co_lead`` (the role),
    never ``codex_co_lead_eval`` (the lease).
    """
    roles = _role_matches(value)
    if len(roles) == 1:
        return "repair", roles[0]
    if len(roles) >= 2:
        return "deny", (
            f"ambiguous handle prefix {value!r}; canonical-role candidates: {roles}"
        )
    # No role match — fall through to the lease zoo.
    leases = _lease_matches(value, known)
    if len(leases) == 1:
        return "repair", leases[0]
    if len(leases) >= 2:
        return "deny", (
            f"ambiguous handle prefix {value!r}; lease candidates: {leases}"
        )
    return "passthrough", None


def _process_handles(
    tool_input: dict,
) -> tuple[str, dict | None, str, list[tuple[str, str]]]:
    """Validate and optionally repair handles in tool_input.

    Returns (outcome, updated_input, deny_reason, repairs) where outcome is
    ``allow`` or ``deny``. When outcome is ``allow`` with repairs,
    updated_input is a deep copy with corrected ``to`` / ``requires_response_from``.
    """
    known = _known_handles()
    repairs: list[tuple[str, str]] = []
    updated = copy.deepcopy(tool_input)

    to_val = tool_input.get("to")
    rrf_val = tool_input.get("requires_response_from")

    repaired_to: str | list | None = to_val
    repaired_rrf: str | None = rrf_val if isinstance(rrf_val, str) else rrf_val

    if to_val is not None:
        if isinstance(to_val, str):
            stripped = to_val.strip()
            if stripped and stripped not in known:
                action, detail = _classify_unknown(stripped, known)
                if action == "repair":
                    repairs.append((stripped, detail))
                    repaired_to = detail
                elif action == "deny":
                    return "deny", None, detail, repairs
            else:
                repaired_to = stripped if stripped else to_val
        elif isinstance(to_val, list):
            new_list: list = []
            for item in to_val:
                if not isinstance(item, str):
                    return (
                        "deny",
                        None,
                        f"unknown handle in `to` list: {item!r}; known: {sorted(known)}",
                        repairs,
                    )
                stripped = item.strip()
                if not stripped:
                    new_list.append(item)
                    continue
                if stripped in known:
                    new_list.append(stripped)
                    continue
                action, detail = _classify_unknown(stripped, known)
                if action == "repair":
                    repairs.append((stripped, detail))
                    new_list.append(detail)
                elif action == "deny":
                    return "deny", None, detail, repairs
                else:
                    new_list.append(stripped)
            repaired_to = new_list
        # None or other types: MCP handles shape validation; fail-open.

    if rrf_val is not None:
        if not isinstance(rrf_val, str):
            return (
                "deny",
                None,
                "`requires_response_from` must be a single handle string",
                repairs,
            )
        stripped_rrf = rrf_val.strip()
        if not stripped_rrf:
            return "deny", None, "`requires_response_from` is empty", repairs
        if stripped_rrf not in known:
            action, detail = _classify_unknown(stripped_rrf, known)
            if action == "repair":
                repairs.append((stripped_rrf, detail))
                repaired_rrf = detail
            elif action == "deny":
                return "deny", None, detail, repairs
            else:
                repaired_rrf = stripped_rrf
        else:
            repaired_rrf = stripped_rrf

        if isinstance(repaired_to, str):
            to_cmp = repaired_to.strip()
            if to_cmp and repaired_rrf != to_cmp:
                return (
                    "deny",
                    None,
                    (
                        f"`requires_response_from` ({repaired_rrf!r}) differs from "
                        f"`to` ({to_cmp!r}); the response-clearing peer must match "
                        "the addressed peer"
                    ),
                    repairs,
                )

    if repairs:
        if "to" in tool_input:
            updated["to"] = repaired_to
        if "requires_response_from" in tool_input:
            updated["requires_response_from"] = repaired_rrf
        return "allow", updated, "", repairs

    return "allow", None, "", repairs


def _emit_deny(reason: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


def _emit_repair(updated_input: dict, repairs: list[tuple[str, str]]) -> None:
    context = "repaired truncated handle(s): " + " ".join(
        f"{old}→{new}" for old, new in repairs
    )
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
            "additionalContext": context,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    if _disabled():
        return 0

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail-open on parse error

    tool_name = payload.get("tool_name")
    if tool_name not in MATCHED_TOOLS:
        return 0

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0  # fail-open; the MCP tool handles shape validation

    outcome, updated_input, deny_reason, repairs = _process_handles(tool_input)
    if outcome == "deny":
        _emit_deny(deny_reason)
        return 0

    if repairs and updated_input is not None:
        _emit_repair(updated_input, repairs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
