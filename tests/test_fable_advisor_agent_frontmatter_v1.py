"""Durable regression guard for .claude/agents/fable-advisor.md frontmatter.

The advisor is a STANDING read-only peer with exactly two outbound tools. This
test is its least-privilege boundary:
- name == fable-advisor, model == fable (the hard stop rides on this key)
- exact tool set, count carried by INTENDED_TOOLS alone
- forbidden mutation/spawn/dispatch/board families ABSENT
- EVERY outbound tool is covered by the PreToolUse guard matcher, wired IN THIS
  FILE
- direct loader surface (no rendered sibling)

Containment moved rather than loosened. It used to be "ai_room_post is absent,
so recipient selection does not exist as a surface." The advisor is the
direction lead and Gabe's portal, and requiring a solicitation before it could
speak made its own leash decide when Gabe could be heard, so post is now
present and the containment lives in advisor_outbound_gate.py: an initiation
must be addressed to `claude` alone, scalar, with a key allowlist of
{body, to, kind}. That is a predicate, not an absence, which is why the
outbound-coverage test below is load-bearing: adding an outbound tool without
extending the guard matcher would leave it ungated.

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
    "mcp__ai-room__ai_room_post",
    "CronCreate",
    "ScheduleWakeup",
]

# Tools that can emit into the room. Each MUST appear in the guard matcher; an
# outbound tool the matcher does not name is an ungated egress.
OUTBOUND_TOOLS = [
    "mcp__ai-room__ai_room_reply",
    "mcp__ai-room__ai_room_post",
]

FORBIDDEN_TOOLS = [
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

    def test_exact_tool_set_present(self) -> None:
        # The count is carried by INTENDED_TOOLS alone. It used to be asserted
        # here as a literal too, which is two authored copies of one fact with
        # nothing forcing agreement -- the defect class this repo removes by
        # deletion rather than by keeping both in sync.
        self.assertEqual(self.tools, INTENDED_TOOLS)

    def test_forbidden_tools_absent(self) -> None:
        present_forbidden = [t for t in FORBIDDEN_TOOLS if t in self.tools]
        self.assertEqual(present_forbidden, [])

    def test_every_outbound_tool_is_guard_covered(self) -> None:
        # The containment predicate: an outbound tool the matcher does not name
        # reaches the room ungated. Property, not spelling -- this fires for any
        # future outbound tool, not just post.
        granted = [t for t in OUTBOUND_TOOLS if t in self.tools]
        self.assertEqual(granted, OUTBOUND_TOOLS)
        # Bind the MATCHER lines, not the whole frontmatter block. Searching the
        # block is tautological: the tool name is in it because `tools:` granted
        # it, so the check passes with no matcher at all -- observed doing
        # exactly that before this line existed.
        matchers = " ".join(ln.split("matcher:", 1)[1]
                            for ln in self.block.splitlines() if "matcher:" in ln)
        uncovered = [t for t in granted if t not in matchers]
        self.assertEqual(uncovered, [], f"ungated egress; matcher line: {matchers!r}")

    def test_pretooluse_guard_wired_agent_locally(self) -> None:
        # Agent-local by design: .claude/settings.json hooks are project-global
        # and would apply this guard to every session.
        self.assertIn("PreToolUse", self.block)
        self.assertIn(GUARD_BASENAME, self.block)

    def test_guard_file_exists(self) -> None:
        self.assertTrue((REPO_ROOT / ".claude" / "hooks" / GUARD_BASENAME).is_file())


if __name__ == "__main__":
    unittest.main()
