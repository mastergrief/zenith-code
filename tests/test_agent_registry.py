"""Unit tests for AgentRegistry + the AgentCreate/Message/Get/List/Terminate tools.

Run: PYTHONPATH=. python3 -m unittest tests.test_agent_registry -v

Tests mock Agent.chat so they don't need a live llama-server.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.agent_registry import AgentRegistry, get_registry, reset_registry
from agents.tools import (
    TOOL_DEFINITIONS,
    _run_agent_create,
    _run_agent_get,
    _run_agent_list,
    _run_agent_message,
    _run_agent_terminate,
    execute_tool,
)


class _StubAgent:
    """Minimal stand-in for an Agent — has the attributes the registry reads."""
    def __init__(self, role="stub", model="m", history=None, todos=None):
        self.role = role
        self.model = model
        self.history = list(history or [])
        self.todos = list(todos or [])


class AgentRegistryDirectTests(unittest.TestCase):
    """Test AgentRegistry directly without going through the tool layer."""

    def setUp(self):
        self.reg = AgentRegistry()

    def test_register_with_explicit_id(self):
        aid = self.reg.register(_StubAgent("reviewer"), agent_id="reviewer_a")
        self.assertEqual(aid, "reviewer_a")
        self.assertEqual(len(self.reg), 1)

    def test_register_with_auto_id(self):
        aid = self.reg.register(_StubAgent("planner"))
        self.assertTrue(aid.startswith("agt_"))
        self.assertEqual(len(aid), 12)  # "agt_" + 8 hex
        self.assertEqual(len(self.reg), 1)

    def test_register_collision_raises(self):
        self.reg.register(_StubAgent(), agent_id="dup")
        with self.assertRaises(ValueError) as ctx:
            self.reg.register(_StubAgent(), agent_id="dup")
        self.assertIn("already exists", str(ctx.exception))

    def test_register_invalid_name_raises(self):
        # Names that should be rejected: capitals, spaces, leading dash/underscore,
        # special chars, too long, empty
        bad_names = ["BadCaps", "has space", "-leading", "_leading", "has!bang", "x" * 41, ""]
        for bad in bad_names:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.reg.register(_StubAgent(), agent_id=bad)

    def test_register_valid_names(self):
        good_names = ["a", "reviewer_a", "agent-1", "x123", "0_starts_with_digit",
                      "a" * 40]
        for good in good_names:
            with self.subTest(good=good):
                reg = AgentRegistry()  # fresh per case to avoid collisions
                reg.register(_StubAgent(), agent_id=good)

    def test_get_returns_registered_agent(self):
        aid = self.reg.register(_StubAgent("hunter"), agent_id="x")
        ra = self.reg.get(aid)
        self.assertIsNotNone(ra)
        self.assertEqual(ra.agent.role, "hunter")
        self.assertEqual(ra.name, "x")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.reg.get("nonexistent"))

    def test_terminate_removes(self):
        self.reg.register(_StubAgent(), agent_id="x")
        self.assertTrue(self.reg.terminate("x"))
        self.assertEqual(len(self.reg), 0)
        self.assertIsNone(self.reg.get("x"))

    def test_terminate_missing_returns_false(self):
        self.assertFalse(self.reg.terminate("nope"))

    def test_list_agents_returns_all(self):
        self.reg.register(_StubAgent("a"), agent_id="one")
        self.reg.register(_StubAgent("b"), agent_id="two")
        self.reg.register(_StubAgent("c"), agent_id="three")
        entries = self.reg.list_agents()
        self.assertEqual(len(entries), 3)
        ids = {e["id"] for e in entries}
        self.assertEqual(ids, {"one", "two", "three"})
        # Each entry has the expected keys
        for e in entries:
            self.assertIn("role", e)
            self.assertIn("history_len", e)
            self.assertIn("age_seconds", e)

    def test_list_agents_empty(self):
        self.assertEqual(self.reg.list_agents(), [])

    def test_clear(self):
        self.reg.register(_StubAgent(), agent_id="a")
        self.reg.register(_StubAgent(), agent_id="b")
        self.reg.clear()
        self.assertEqual(len(self.reg), 0)

    def test_touch_updates_last_active(self):
        import time as _time
        aid = self.reg.register(_StubAgent(), agent_id="x")
        ra = self.reg.get(aid)
        original = ra.last_active_at
        _time.sleep(0.01)
        self.reg.touch(aid)
        ra2 = self.reg.get(aid)
        self.assertGreater(ra2.last_active_at, original)


class RegistrySingletonTests(unittest.TestCase):
    def test_get_registry_returns_same_instance(self):
        reset_registry()
        r1 = get_registry()
        r2 = get_registry()
        self.assertIs(r1, r2)

    def test_reset_registry_creates_new(self):
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        self.assertIsNot(r1, r2)


class AgentCreateToolTests(unittest.TestCase):
    """Test the _run_agent_create helper end-to-end with mocked chat."""

    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_create_returns_agent_id_and_response(self):
        with patch("agents.agent.Agent.chat", return_value="initial response"):
            result = _run_agent_create(
                {"prompt": "do x", "subagent_type": "explore"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertIn("agent_id:", result)
        self.assertIn("initial response", result)
        # Auto-generated ID
        self.assertIn("agt_", result)

    def test_create_with_custom_name(self):
        with patch("agents.agent.Agent.chat", return_value="ok"):
            result = _run_agent_create(
                {"prompt": "do x", "name": "reviewer_a", "subagent_type": "explore"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertIn("reviewer_a", result)
        # Verify it's actually in the registry
        reg = get_registry()
        self.assertIsNotNone(reg.get("reviewer_a"))

    def test_create_with_role_override(self):
        captured = {}

        def chat_capture(self, *a, **kw):
            captured["agent"] = self
            return "ok"

        with patch("agents.agent.Agent.chat", new=chat_capture):
            _run_agent_create(
                {"prompt": "x", "role": "skeptical reviewer", "subagent_type": "explore"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertIn("skeptical reviewer", captured["agent"].system_prompt)

    def test_create_missing_prompt_returns_error(self):
        result = _run_agent_create(
            {"prompt": ""},
            parent_model="qwen3.5:4b", parent_backend="ollama",
        )
        self.assertTrue(result.startswith("Error:"))

    def test_create_duplicate_name_returns_error(self):
        with patch("agents.agent.Agent.chat", return_value="ok"):
            _run_agent_create(
                {"prompt": "x", "name": "dup"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
            result = _run_agent_create(
                {"prompt": "x", "name": "dup"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("already exists", result)

    def test_create_chat_failure_rolls_back_registration(self):
        with patch("agents.agent.Agent.chat", side_effect=RuntimeError("boom")):
            result = _run_agent_create(
                {"prompt": "x", "name": "doomed"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("boom", result)
        # Doomed agent should NOT be in the registry — rollback worked
        reg = get_registry()
        self.assertIsNone(reg.get("doomed"))

    def test_create_invalid_name_returns_error(self):
        with patch("agents.agent.Agent.chat", return_value="ok"):
            result = _run_agent_create(
                {"prompt": "x", "name": "Bad Name!"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
        self.assertTrue(result.startswith("Error:"))


class AgentMessageToolTests(unittest.TestCase):
    def setUp(self):
        reset_registry()
        # Pre-register an agent for messaging tests
        with patch("agents.agent.Agent.chat", return_value="initial"):
            _run_agent_create(
                {"prompt": "init", "name": "target"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )

    def tearDown(self):
        reset_registry()

    def test_message_runs_chat_on_target(self):
        with patch("agents.agent.Agent.chat", return_value="follow-up response") as mock:
            result = _run_agent_message({"agent_id": "target", "content": "what about X"})
        self.assertEqual(result, "follow-up response")
        mock.assert_called_once_with("what about X")

    def test_message_unknown_agent_returns_error(self):
        result = _run_agent_message({"agent_id": "ghost", "content": "hi"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("no agent registered", result)

    def test_message_missing_agent_id_returns_error(self):
        result = _run_agent_message({"content": "hi"})
        self.assertTrue(result.startswith("Error:"))

    def test_message_empty_content_returns_error(self):
        result = _run_agent_message({"agent_id": "target", "content": ""})
        self.assertTrue(result.startswith("Error:"))

    def test_message_chat_failure_returns_error(self):
        with patch("agents.agent.Agent.chat", side_effect=RuntimeError("boom")):
            result = _run_agent_message({"agent_id": "target", "content": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("boom", result)
        # Agent should still exist after the failure (we don't auto-remove on chat fail)
        reg = get_registry()
        self.assertIsNotNone(reg.get("target"))

    def test_message_appends_to_existing_history(self):
        """The same Agent instance should accumulate history across messages."""
        from agents.agent import Agent

        # Use a real Agent (not mocked) but mock the LLM call layer
        reset_registry()

        # Patch _call_ollama to return a canned response without network
        def fake_call(self, messages):
            return {"message": {"role": "assistant", "content": "echoed"}}

        with patch("agents.agent.Agent._call_ollama", new=fake_call):
            _run_agent_create(
                {"prompt": "first", "name": "stateful", "subagent_type": "general-purpose"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
            _run_agent_message({"agent_id": "stateful", "content": "second"})
            _run_agent_message({"agent_id": "stateful", "content": "third"})

        reg = get_registry()
        ra = reg.get("stateful")
        # 3 user turns + 3 assistant turns = 6 messages
        self.assertEqual(len(ra.agent.history), 6)
        user_messages = [m for m in ra.agent.history if m["role"] == "user"]
        # Note: each user message starts with /no_think because think defaults
        # depend on chat()'s think param. Let's just check there are 3.
        self.assertEqual(len(user_messages), 3)


class AgentGetToolTests(unittest.TestCase):
    def setUp(self):
        reset_registry()
        with patch("agents.agent.Agent.chat", return_value="initial response"):
            _run_agent_create(
                {"prompt": "investigate", "name": "spec_reviewer", "role": "spec reviewer"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )

    def tearDown(self):
        reset_registry()

    def test_get_returns_state_snapshot(self):
        result = _run_agent_get({"agent_id": "spec_reviewer"})
        self.assertIn("spec_reviewer", result)
        self.assertIn("role:", result)
        self.assertIn("history_len:", result)
        self.assertIn("age:", result)

    def test_get_unknown_agent_returns_error(self):
        result = _run_agent_get({"agent_id": "ghost"})
        self.assertTrue(result.startswith("Error:"))

    def test_get_missing_agent_id_returns_error(self):
        result = _run_agent_get({})
        self.assertTrue(result.startswith("Error:"))


class AgentListToolTests(unittest.TestCase):
    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_list_empty_returns_friendly_message(self):
        result = _run_agent_list({})
        self.assertIn("No agents registered", result)

    def test_list_returns_all_agents(self):
        with patch("agents.agent.Agent.chat", return_value="ok"):
            _run_agent_create(
                {"prompt": "a", "name": "agent_one"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )
            _run_agent_create(
                {"prompt": "b", "name": "agent_two"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )

        result = _run_agent_list({})
        self.assertIn("agent_one", result)
        self.assertIn("agent_two", result)
        self.assertIn("Registered agents (2)", result)


class AgentTerminateToolTests(unittest.TestCase):
    def setUp(self):
        reset_registry()
        with patch("agents.agent.Agent.chat", return_value="ok"):
            _run_agent_create(
                {"prompt": "x", "name": "doomed"},
                parent_model="qwen3.5:4b", parent_backend="ollama",
            )

    def tearDown(self):
        reset_registry()

    def test_terminate_removes_agent(self):
        result = _run_agent_terminate({"agent_id": "doomed"})
        self.assertIn("Terminated", result)
        self.assertIsNone(get_registry().get("doomed"))

    def test_terminate_unknown_returns_error(self):
        result = _run_agent_terminate({"agent_id": "ghost"})
        self.assertTrue(result.startswith("Error:"))

    def test_subsequent_message_to_terminated_agent_errors(self):
        _run_agent_terminate({"agent_id": "doomed"})
        result = _run_agent_message({"agent_id": "doomed", "content": "still there?"})
        self.assertTrue(result.startswith("Error:"))


class CrossValidationPatternTests(unittest.TestCase):
    """End-to-end test of the cross-validation pattern via execute_tool."""

    def setUp(self):
        reset_registry()

    def tearDown(self):
        reset_registry()

    def test_two_agents_can_coexist_and_be_messaged_independently(self):
        """The pattern: parent spawns 2 reviewers, sends each a message, both work independently."""
        responses = iter([
            "reviewer_a says: line 42 is buggy",     # AgentCreate(reviewer_a)
            "reviewer_b says: line 42 looks fine",   # AgentCreate(reviewer_b)
            "reviewer_a: on second look, you're right",  # AgentMessage(reviewer_a, ...)
            "reviewer_b: confirmed safe",             # AgentMessage(reviewer_b, ...)
        ])

        def chat_side_effect(*args, **kwargs):
            return next(responses)

        with patch("agents.agent.Agent.chat", side_effect=chat_side_effect):
            r1 = execute_tool("AgentCreate",
                              {"prompt": "review the auth code", "name": "reviewer_a"})
            r2 = execute_tool("AgentCreate",
                              {"prompt": "audit auth code skeptically", "name": "reviewer_b"})
            r3 = execute_tool("AgentMessage",
                              {"agent_id": "reviewer_a",
                               "content": "reviewer_b says line 42 looks fine, re-examine"})
            r4 = execute_tool("AgentMessage",
                              {"agent_id": "reviewer_b",
                               "content": "reviewer_a withdrew the claim — confirm?"})

        self.assertIn("reviewer_a", r1)
        self.assertIn("line 42 is buggy", r1)
        self.assertIn("reviewer_b", r2)
        self.assertIn("line 42 looks fine", r2)
        self.assertIn("on second look", r3)
        self.assertIn("confirmed safe", r4)

        # Both still in registry
        listed = execute_tool("AgentList", {})
        self.assertIn("reviewer_a", listed)
        self.assertIn("reviewer_b", listed)


class NewToolDefinitionsTests(unittest.TestCase):
    def test_total_tool_count_is_twenty(self):
        # 6 base + Agent + Sleep + WebFetch + WebSearch + AskUserQuestion
        # + TodoWrite + TodoRead + MultiEdit + list_directory
        # + AgentCreate + AgentMessage + AgentGet + AgentList + AgentTerminate
        # = 20
        self.assertEqual(len(TOOL_DEFINITIONS), 20)

    def test_all_five_registry_tools_in_definitions(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        for new in ("AgentCreate", "AgentMessage", "AgentGet", "AgentList", "AgentTerminate"):
            self.assertIn(new, names)

    def test_agent_create_schema(self):
        spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "AgentCreate")
        params = spec["function"]["parameters"]
        self.assertEqual(params["required"], ["prompt"])
        for opt in ("name", "subagent_type", "role"):
            self.assertIn(opt, params["properties"])

    def test_agent_message_schema(self):
        spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "AgentMessage")
        params = spec["function"]["parameters"]
        self.assertEqual(set(params["required"]), {"agent_id", "content"})


if __name__ == "__main__":
    unittest.main()
