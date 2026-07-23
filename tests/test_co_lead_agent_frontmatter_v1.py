"""Durable regression guard for .claude/agents/co-lead.md frontmatter.

The co-lead is the always-on READ-ONLY hard-gate reviewer (handle
codex_co_lead). This test is its least-privilege security boundary:
- name == co-lead
- exact 28-tool read-only set present (no grant_list — not on the live
  ai-room MCP surface; gate-2 finding)
- forbidden mutation/spawn/dispatch/commit tools ABSENT
- direct loader surface (no rendered sibling)

Run: PYTHONPATH=. python3 -m unittest tests.test_co_lead_agent_frontmatter_v1 -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / ".claude" / "agents" / "co-lead.md"

INTENDED_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "Bash",
    "mcp__ai-room__ai_room_ack",
    "mcp__ai-room__ai_room_deliveries",
    "mcp__ai-room__ai_room_doctor",
    "mcp__ai-room__ai_room_inbox",
    "mcp__ai-room__ai_room_cursor_commit",
    "mcp__ai-room__ai_room_peek",
    "mcp__ai-room__ai_room_peer_status",
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_provenance_lint",
    "mcp__ai-room__ai_room_read",
    "mcp__ai-room__ai_room_read_image",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai-room__ai_room_resource_lane_status",
    "mcp__ai-room__ai_room_resume_check",
    "mcp__ai-room__ai_room_scratch_delete",
    "mcp__ai-room__ai_room_scratch_get",
    "mcp__ai-room__ai_room_scratch_list",
    "mcp__ai-room__ai_room_scratch_set",
    "mcp__ai-room__ai_room_search",
    "mcp__ai-room__ai_room_status",
    "mcp__ai-room__ai_room_tail",
    "mcp__ai-room__ai_room_task_contract_lint",
    "mcp__ai-room__ai_room_task_list",
    "mcp__ai-room__ai_room_task_show",
]

# Read-only room tools that must always remain present.
REQUIRED_READONLY_TOOLS = [
    "mcp__ai-room__ai_room_read",
    "mcp__ai-room__ai_room_tail",
    "mcp__ai-room__ai_room_status",
    "mcp__ai-room__ai_room_resume_check",
    "mcp__ai-room__ai_room_peer_status",
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_reply",
]

FORBIDDEN_TOOLS = [
    # Mutating filesystem
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    # Board mutation
    "mcp__ai-room__ai_room_task_create",
    "mcp__ai-room__ai_room_task_start",
    "mcp__ai-room__ai_room_task_claim",
    "mcp__ai-room__ai_room_task_update",
    "mcp__ai-room__ai_room_task_complete",
    # Spawn / kill / dispatch
    "mcp__ai-room__ai_room_spawn_claude",
    "mcp__ai-room__ai_room_spawn_claudex",
    "mcp__ai-room__ai_room_kill_claude",
    "mcp__ai-room__ai_room_kill_claudex",
    "mcp__ai-room__ai_room_spawn_and_dispatch",
    "mcp__ai-room__ai_room_dispatch_run_claim",
    "mcp__ai-room__ai_room_dispatch_run_mark_started",
    "mcp__ai-room__ai_room_dispatch_run_mark_terminal",
    # Not on the live MCP surface (gate-2 finding) — must stay absent.
    "mcp__ai-room__ai_room_grant_list",
]


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise AssertionError("agent file missing opening frontmatter fence")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise AssertionError("agent file missing closing frontmatter fence")
    block = text[4:end]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _parse_tools(tools_line_value: str) -> list[str]:
    return [part.strip() for part in tools_line_value.split(",") if part.strip()]


class CoLeadAgentFrontmatterV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_path = AGENT_PATH
        cls.text = cls.agent_path.read_text(encoding="utf-8")
        cls.fields = _parse_frontmatter(cls.text)
        cls.tools = _parse_tools(cls.fields.get("tools", ""))

    def test_direct_loader_path_no_rendered_sibling(self) -> None:
        self.assertEqual(
            self.agent_path.resolve(),
            (REPO_ROOT / ".claude" / "agents" / "co-lead.md").resolve(),
        )
        self.assertTrue(self.agent_path.is_file())
        siblings = sorted(p.name for p in self.agent_path.parent.glob("co-lead*"))
        self.assertEqual(siblings, ["co-lead.md"])

    def test_name_is_co_lead(self) -> None:
        self.assertEqual(self.fields.get("name"), "co-lead")

    def test_exact_28_tool_set_present(self) -> None:
        self.assertEqual(len(self.tools), 28)
        self.assertEqual(self.tools, INTENDED_TOOLS)

    def test_required_readonly_tools_present(self) -> None:
        missing = [t for t in REQUIRED_READONLY_TOOLS if t not in self.tools]
        self.assertEqual(missing, [])

    def test_forbidden_tools_absent(self) -> None:
        present_forbidden = [t for t in FORBIDDEN_TOOLS if t in self.tools]
        self.assertEqual(present_forbidden, [])

    def test_no_grant_list_reference_in_body(self) -> None:
        # The tool is not on the live ai-room MCP surface (gate-2 finding);
        # neither frontmatter nor prose may instruct its use.
        self.assertNotIn("ai_room_grant_list", self.text)


if __name__ == "__main__":
    unittest.main()
