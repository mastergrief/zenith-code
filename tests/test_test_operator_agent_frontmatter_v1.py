"""Durable regression guard for .claude/agents/test-operator.md frontmatter.

Asserts CONFIG v6 EDIT_2 contract:
- model == haiku
- exact 15-tool least-privilege set present
- forbidden tools ABSENT
- direct loader surface (no rendered sibling)

Run: PYTHONPATH=. python3 -m unittest tests.test_test_operator_agent_frontmatter_v1 -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / ".claude" / "agents" / "test-operator.md"

INTENDED_TOOLS = [
    "Bash",
    "Read",
    "Glob",
    "Grep",
    "mcp__ai-room__ai_room_reply",
    "mcp__ai-room__ai_room_post",
    "mcp__ai-room__ai_room_status",
    "mcp__ai-room__ai_room_resume_check",
    "mcp__ai-room__ai_room_resource_lane_acquire",
    "mcp__ai-room__ai_room_resource_lane_release",
    "mcp__ai-room__ai_room_resource_lane_status",
    "mcp__ai-room__ai_room_dispatch_run_claim",
    "mcp__ai-room__ai_room_dispatch_run_status",
    "mcp__ai-room__ai_room_dispatch_run_mark_started",
    "mcp__ai-room__ai_room_dispatch_run_mark_terminal",
]

FORBIDDEN_TOOLS = [
    "mcp__ai-room__ai_room_ack",
    "mcp__ai-room__ai_room_read",
    "mcp__ai-room__ai_room_tail",
    "mcp__ai-room__ai_room_search",
    "mcp__ai-room__ai_room_inbox",
    "mcp__ai-room__ai_room_scratch_set",
    "mcp__ai-room__ai_room_scratch_get",
    "mcp__ai-room__ai_room_scratch_delete",
    "mcp__ai-room__ai_room_scratch_list",
    "mcp__ai-room__ai_room_task_list",
    "mcp__ai-room__ai_room_task_show",
    "mcp__ai-room__ai_room_spawn_claude",
    "mcp__ai-room__ai_room_spawn_claudex",
    "mcp__ai-room__ai_room_kill_claude",
    "mcp__ai-room__ai_room_kill_claudex",
    "mcp__ai-room__ai_room_spawn_and_dispatch",
    "mcp__ai-room__ai_room_task_create",
    "mcp__ai-room__ai_room_task_start",
    "mcp__ai-room__ai_room_task_claim",
    "mcp__ai-room__ai_room_task_update",
    "mcp__ai-room__ai_room_task_complete",
    "Edit",
    "Write",
    "MultiEdit",
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


class TestOperatorAgentFrontmatterV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_path = AGENT_PATH
        cls.text = cls.agent_path.read_text(encoding="utf-8")
        cls.fields = _parse_frontmatter(cls.text)
        cls.tools = _parse_tools(cls.fields.get("tools", ""))

    def test_direct_loader_path_no_rendered_sibling(self) -> None:
        self.assertEqual(
            self.agent_path.resolve(),
            (REPO_ROOT / ".claude" / "agents" / "test-operator.md").resolve(),
        )
        self.assertTrue(self.agent_path.is_file())
        # No generated/rendered peer surface beside the source agent md.
        agents_dir = self.agent_path.parent
        siblings = sorted(p.name for p in agents_dir.glob("test-operator*"))
        self.assertEqual(siblings, ["test-operator.md"])

    def test_model_is_haiku(self) -> None:
        self.assertEqual(self.fields.get("name"), "test-operator")
        self.assertEqual(self.fields.get("model"), "haiku")

    def test_exact_15_tool_set_present(self) -> None:
        self.assertEqual(len(self.tools), 15)
        self.assertEqual(self.tools, INTENDED_TOOLS)

    def test_forbidden_tools_absent(self) -> None:
        present_forbidden = [tool for tool in FORBIDDEN_TOOLS if tool in self.tools]
        self.assertEqual(present_forbidden, [])


if __name__ == "__main__":
    unittest.main()
