"""Durable regression guard for .claude/agents/fable-advisor.md frontmatter.

The advisor is a STANDING read-only peer with exactly one outbound tool. This
test is its least-privilege boundary:
- name == fable-advisor, model == fable (the hard stop rides on this key)
- exact 10-tool set (no ai_room_post -- reply-only is the whole containment
  design; ai_room_reply defaults to the parent's sender, which removes
  recipient selection as a surface)
- forbidden mutation/spawn/dispatch/board families ABSENT
- the PreToolUse guard is wired IN THIS FILE, matching ai_room_reply
- direct loader surface (no rendered sibling)

Run: PYTHONPATH=. python3 -m unittest tests.test_fable_advisor_agent_frontmatter_v1 -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / ".claude" / "agents" / "fable-advisor.md"
GUARD_BASENAME = "advisor_outbound_gate.py"

INTENDED_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "mcp__ai-room__ai_room_read",
    "mcp__ai-room__ai_room_tail",
    "mcp__ai-room__ai_room_search",
    "mcp__ai-room__ai_room_status",
    "mcp__ai-room__ai_room_inbox",
    "mcp__ai-room__ai_room_resume_check",
    "mcp__ai-room__ai_room_reply",
]

FORBIDDEN_TOOLS = [
    # The one that makes this agent reply-only. A tools allowlist does not by
    # itself forbid dispatch -- ai_room_reply accepts kind=task_dispatch -- but
    # removing post narrows the surface the guard has to close.
    "mcp__ai-room__ai_room_post",
    # Mutating filesystem / shell
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "Bash",
    # Board mutation
    "mcp__ai-room__ai_room_task_create",
    "mcp__ai-room__ai_room_task_start",
    "mcp__ai-room__ai_room_task_claim",
    "mcp__ai-room__ai_room_task_update",
    "mcp__ai-room__ai_room_task_complete",
    # Spawn / kill / dispatch
    "mcp__ai-room__ai_room_spawn_claude",
    "mcp__ai-room__ai_room_spawn_claudex",
    "mcp__ai-room__ai_room_spawn_and_dispatch",
    "mcp__ai-room__ai_room_kill_claude",
    "mcp__ai-room__ai_room_kill_claudex",
    "mcp__ai-room__ai_room_dispatch_run_claim",
    "mcp__ai-room__ai_room_dispatch_run_mark_started",
    "mcp__ai-room__ai_room_dispatch_run_mark_terminal",
    # Acknowledgement / resource lanes
    "mcp__ai-room__ai_room_ack",
    "mcp__ai-room__ai_room_resource_lane_acquire",
    "mcp__ai-room__ai_room_resource_lane_release",
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
        if line[0] in " \t-":  # nested mapping/sequence entry, not a top-level key
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields, block


def _parse_tools(tools_line_value: str) -> list[str]:
    return [part.strip() for part in tools_line_value.split(",") if part.strip()]


class FableAdvisorAgentFrontmatterV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_path = AGENT_PATH
        cls.text = cls.agent_path.read_text(encoding="utf-8")
        cls.fields, cls.block = _parse_frontmatter(cls.text)
        cls.tools = _parse_tools(cls.fields.get("tools", ""))

    def test_direct_loader_path_no_rendered_sibling(self) -> None:
        self.assertTrue(self.agent_path.is_file())
        siblings = sorted(p.name for p in self.agent_path.parent.glob("fable-advisor*"))
        self.assertEqual(siblings, ["fable-advisor.md"])

    def test_name_is_fable_advisor(self) -> None:
        self.assertEqual(self.fields.get("name"), "fable-advisor")

    def test_model_is_exactly_fable(self) -> None:
        # ai_room_spawn_claude has no model parameter, so this key is the ONLY
        # route to the model. The slice's hard stop is defined against it.
        self.assertEqual(self.fields.get("model"), "fable")

    def test_exact_10_tool_set_present(self) -> None:
        self.assertEqual(len(self.tools), 10)
        self.assertEqual(self.tools, INTENDED_TOOLS)

    def test_forbidden_tools_absent(self) -> None:
        present_forbidden = [t for t in FORBIDDEN_TOOLS if t in self.tools]
        self.assertEqual(present_forbidden, [])

    def test_ai_room_post_absent_everywhere(self) -> None:
        # Not merely absent from the tool list: the prose must not instruct its
        # use either, or a future reader will wire it back in.
        self.assertNotIn("ai_room_post", self.text)

    def test_pretooluse_guard_wired_agent_locally(self) -> None:
        # Agent-local by design: .claude/settings.json hooks are project-global
        # and would apply this guard to every session.
        self.assertIn("PreToolUse", self.block)
        self.assertIn("mcp__ai-room__ai_room_reply", self.block)
        self.assertIn(GUARD_BASENAME, self.block)

    def test_guard_file_exists(self) -> None:
        self.assertTrue((REPO_ROOT / ".claude" / "hooks" / GUARD_BASENAME).is_file())


if __name__ == "__main__":
    unittest.main()
