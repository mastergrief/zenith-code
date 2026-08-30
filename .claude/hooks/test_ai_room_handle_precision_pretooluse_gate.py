from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

HOOK_DIR = Path(__file__).resolve().parent
HOOK = HOOK_DIR / "ai_room_handle_precision_pretooluse_gate.py"
TOOL_NAME = "mcp__ai-room__ai_room_post"


def run_pretooluse(
    payload: dict,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["python3", str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=merged,
        check=False,
    )


def base_payload(tool_input: dict) -> dict:
    return {"tool_name": TOOL_NAME, "tool_input": tool_input}


def parse_stdout(result: subprocess.CompletedProcess[str]) -> dict | None:
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def write_leases(room: Path, handles: list[str]) -> None:
    leases = room / "channels" / "ai-room" / "leases"
    leases.mkdir(parents=True, exist_ok=True)
    for handle in handles:
        (leases / f"{handle}.json").write_text("{}", encoding="utf-8")


def env(room: Path) -> dict[str, str]:
    return {"AI_ROOM_DIR": str(room), "AI_ROOM_CHANNEL": "ai-room"}


@pytest.fixture
def room(tmp_path: Path) -> Path:
    path = tmp_path / "room"
    path.mkdir()
    return path


# --- canonical role-handle layer (the gap that bit us in practice) ---


def test_codex_co_repaired_to_role_not_eval_lease(room: Path) -> None:
    """codex_co must repair to the canonical role codex_co_lead, NOT to the
    codex_co_lead_eval lease stem (the mis-route the old GLM-scoped hook did)."""
    write_leases(room, ["codex_co_lead_eval", "codex_dev_harmonize"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "codex_co"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["to"] == "codex_co_lead"
    assert "codex_co→codex_co_lead" in out["hookSpecificOutput"]["additionalContext"]


def test_codex_co_lead_full_handle_passthrough(room: Path) -> None:
    """The full canonical role handle passes through unchanged even when a
    same-prefix eval lease exists (old hook would have mis-repaired it)."""
    write_leases(room, ["codex_co_lead_eval"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "codex_co_lead"}), env=env(room)
    )
    assert result.returncode == 0
    assert result.stdout == ""  # passthrough, no repair


def test_cod_ambiguous_across_two_roles_denied(room: Path) -> None:
    """cod is a strict prefix of both codex and codex_co_lead -> deny, not the
    old hook's silent passthrough to the codex lease stem."""
    write_leases(room, ["codex", "codex_co_lead_eval"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "cod"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "ambiguous" in reason
    assert "codex_co_lead" in reason


def test_codex_exact_role_passthrough(room: Path) -> None:
    """codex is a canonical role here (plan-dev worker handle), so it passes
    through unchanged — no repair attempted."""
    result = run_pretooluse(
        base_payload({"body": "x", "to": "codex"}), env=env(room)
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_role_layer_wins_over_lease_zoo(room: Path) -> None:
    """A prefix matching a role AND a lease repairs to the role, never the lease."""
    write_leases(room, ["codex_co_lead_eval", "codex_co_lead_other"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "codex_co_l"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == "codex_co_lead"


# --- classic prefix-truncation repair (parity with the old hook) ---


def test_cla_repaired_to_claude_preserves_body(room: Path) -> None:
    original_body = "dispatch body with preserved fields"
    tool_input = {
        "body": original_body,
        "to": "cla",
        "reply_to": "parent-msg-id",
        "response_deadline_secs": 300,
        "kind": "task_dispatch",
    }
    result = run_pretooluse(base_payload(tool_input), env=env(room))
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["to"] == "claude"
    assert updated["body"] == original_body
    assert updated["reply_to"] == "parent-msg-id"
    assert updated["response_deadline_secs"] == 300
    assert updated["kind"] == "task_dispatch"
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_gabe_prefix_repaired(room: Path) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": "gab"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == "gabe"


def test_ai_r_prefix_of_supervisor_repaired(room: Path) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": "ai_r"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == "ai_room_supervisor"


# --- lease-zoo repair still works for non-role prefixes ---


def test_neural_co_repaired_to_lease(room: Path) -> None:
    """A prefix that matches NO role but exactly one lease repairs to the lease."""
    write_leases(room, ["neural_co_lead"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "neural_co"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == "neural_co_lead"


def test_ambiguous_lease_prefix_denied(room: Path) -> None:
    write_leases(room, ["codex_dev_cache", "codex_dev_cache2"])
    result = run_pretooluse(
        base_payload({"body": "x", "to": "codex_dev_c"}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "ambiguous" in out["hookSpecificOutput"]["permissionDecisionReason"]


# --- genuine unknown + shape errors ---


def test_genuine_unknown_handle_passthrough(room: Path) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": "brand_new_role_xyz"}), env=env(room)
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_rrf_non_str_denied(room: Path) -> None:
    result = run_pretooluse(
        base_payload(
            {"body": "x", "to": "claude", "requires_response_from": ["claude"]}
        ),
        env=env(room),
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_to_list_non_str_denied(room: Path) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": ["gabe", 42]}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_rrf_mismatch_post_repair_denied(room: Path) -> None:
    result = run_pretooluse(
        base_payload(
            {"body": "x", "to": "cla", "requires_response_from": "gabe"}
        ),
        env=env(room),
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "differs from" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_rrf_repaired_alongside_to(room: Path) -> None:
    result = run_pretooluse(
        base_payload(
            {"body": "x", "to": "cla", "requires_response_from": "cla"}
        ),
        env=env(room),
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["to"] == "claude"
    assert updated["requires_response_from"] == "claude"


def test_to_list_element_repaired(room: Path) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": ["gabe", "cla"]}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == ["gabe", "claude"]


# --- non-matched tool / parse error fail-open ---


def test_non_matched_tool_passthrough(room: Path) -> None:
    result = run_pretooluse(
        {"tool_name": "mcp__ai-room__ai_room_inbox", "tool_input": {"limit": 5}},
        env=env(room),
    )
    assert result.returncode == 0
    assert result.stdout == ""


def test_malformed_json_fail_open(room: Path) -> None:
    result = subprocess.run(
        ["python3", str(HOOK)],
        input="{not json",
        text=True,
        capture_output=True,
        env=env(room),
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""


# --- advisor / gate1_audit joined CANONICAL_ROLES -----------------------------
#
# Both are live room handles that were absent from the canonical layer. With no
# lease on disk for either, a truncated `to` had NO match at all and fell to
# `passthrough` — the message left unrepaired for a handle that does not exist.
# Every arm below runs with an EMPTY room dir, so the role layer is the only
# possible source of a match and a green result cannot come from a lease.

import importlib.util as _ilu  # noqa: E402  (module-level fixture imports above)

_spec = _ilu.spec_from_file_location("_handle_gate", HOOK)
_gate = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gate)

PREVIOUS_ROLES = frozenset(
    {"claude", "codex", "codex_co_lead", "gabe", "ai_room_supervisor"}
)


@pytest.mark.parametrize(
    "truncated,expected",
    [("advis", "advisor"), ("gate1_aud", "gate1_audit")],
)
def test_new_role_prefix_repairs_with_no_lease_on_disk(
    room: Path, truncated: str, expected: str
) -> None:
    assert not (room / "channels").exists(), "arm must run lease-absent"
    result = run_pretooluse(
        base_payload({"body": "x", "to": truncated}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None, f"{truncated} was passed through unrepaired"
    assert out["hookSpecificOutput"]["updatedInput"]["to"] == expected
    assert (
        f"{truncated}→{expected}" in out["hookSpecificOutput"]["additionalContext"]
    )


@pytest.mark.parametrize("truncated", ["advis", "gate1_aud"])
def test_calibration_previous_role_set_leaves_them_unrepaired(
    truncated: str,
) -> None:
    """Runs the SHIPPED classifier against the role set as it stood before this
    change. It must return `passthrough` on the same inputs — otherwise the arm
    above is green over a predicate that could never have failed."""
    original = _gate.CANONICAL_ROLES
    try:
        _gate.CANONICAL_ROLES = PREVIOUS_ROLES
        before = _gate._classify_unknown(truncated, set(PREVIOUS_ROLES))
        assert before[0] == "passthrough", before
    finally:
        _gate.CANONICAL_ROLES = original
    after = _gate._classify_unknown(truncated, set(original))
    assert after[0] == "repair", after


@pytest.mark.parametrize("handle", ["advisor", "gate1_audit"])
def test_new_roles_exact_handle_passthrough(room: Path, handle: str) -> None:
    result = run_pretooluse(
        base_payload({"body": "x", "to": handle}), env=env(room)
    )
    assert result.returncode == 0
    assert result.stdout == "", result.stdout


@pytest.mark.parametrize(
    "prefix,candidates",
    [("a", ("advisor", "ai_room_supervisor")), ("g", ("gabe", "gate1_audit"))],
)
def test_prefixes_this_change_made_ambiguous_are_denied(
    room: Path, prefix: str, candidates: tuple[str, str]
) -> None:
    """Named, not incidental: both prefixes repaired before this change and now
    match two roles. Denying beats silently routing to whichever sorts first."""
    result = run_pretooluse(
        base_payload({"body": "x", "to": prefix}), env=env(room)
    )
    assert result.returncode == 0
    out = parse_stdout(result)
    assert out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "ambiguous" in reason
    for c in candidates:
        assert c in reason, reason


def test_every_canonical_role_is_addressable(room: Path) -> None:
    """Denominator comes from the hook's own frozenset, not a copy here.

    A role is addressable if some unambiguous strict prefix repairs to it, OR —
    when every strict prefix also prefixes another role — its exact handle
    passes through. `codex` is the second kind by construction: every prefix of
    it also prefixes `codex_co_lead`, so it is reachable only when spelled in
    full. That is the correct outcome, and asserting it stops a future role from
    silently becoming unaddressable.
    """
    roles = sorted(_gate.CANONICAL_ROLES)
    assert len(roles) >= 7, roles
    exact_only = []
    for role in roles:
        unambiguous = [
            role[:i]
            for i in range(1, len(role))
            if [r for r in roles if r.startswith(role[:i])] == [role]
        ]
        if unambiguous:
            result = run_pretooluse(
                base_payload({"body": "x", "to": unambiguous[0]}), env=env(room)
            )
            out = parse_stdout(result)
            assert out is not None, f"{role} prefix {unambiguous[0]!r} not repaired"
            assert out["hookSpecificOutput"]["updatedInput"]["to"] == role
        else:
            exact_only.append(role)
            result = run_pretooluse(
                base_payload({"body": "x", "to": role}), env=env(room)
            )
            assert result.returncode == 0
            assert result.stdout == "", f"{role} exact handle did not pass through"
    assert exact_only == ["codex"], (
        f"exact-only roles changed: {exact_only}. A role reachable ONLY by its "
        f"full spelling is a routing hazard; name it here deliberately."
    )
