"""Unit tests for the Agent tool — sub-agent spawning, isolation, allowlist filtering.

Run: PYTHONPATH=. python3 -m unittest tests.test_agent_tool -v

These tests mock Agent.chat so they don't need a live llama-server.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.tools import (
    ALLOWED_TOOLS_BY_SUBAGENT,
    TOOL_DEFINITIONS,
    _execute_agent_tool,
    execute_tool,
)
from agents.permissions import PermissionMode


class AgentToolDefinitionTests(unittest.TestCase):
    """The Agent tool spec is registered correctly."""

    def test_agent_tool_in_definitions(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("Agent", names)

    def test_agent_tool_schema_has_required_fields(self):
        spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "Agent")
        params = spec["function"]["parameters"]
        # Only `prompt` is required — `description` is optional and auto-derived.
        self.assertEqual(set(params["required"]), {"prompt"})
        self.assertIn("description", params["properties"])
        self.assertIn("subagent_type", params["properties"])
        # subagent_type enum lists all 4 types
        enum = params["properties"]["subagent_type"].get("enum", [])
        self.assertEqual(
            set(enum),
            {"explore", "plan", "verification", "general-purpose"},
        )

    def test_subagent_table_excludes_agent_tool(self):
        """Recursion guard: no subagent type can spawn more sub-agents."""
        for sub_type, (allowed, _mode) in ALLOWED_TOOLS_BY_SUBAGENT.items():
            self.assertNotIn(
                "Agent",
                allowed,
                f"{sub_type} must not allow Agent tool (would enable recursion)",
            )

    def test_explore_and_plan_are_readonly(self):
        for sub_type in ("explore", "plan"):
            _allowed, mode = ALLOWED_TOOLS_BY_SUBAGENT[sub_type]
            self.assertEqual(mode, PermissionMode.READ_ONLY)

    def test_verification_has_bash_but_no_writes(self):
        allowed, mode = ALLOWED_TOOLS_BY_SUBAGENT["verification"]
        self.assertIn("bash", allowed)
        self.assertNotIn("write_file", allowed)
        self.assertNotIn("edit_file", allowed)
        self.assertEqual(mode, PermissionMode.WORKSPACE_WRITE)


class AgentToolDispatchTests(unittest.TestCase):
    """`_execute_agent_tool` constructs sub-agents with the right wiring."""

    def _spawn_with_capture(self, args: dict, return_text: str = "SUBAGENT_DONE"):
        """Run _execute_agent_tool with Agent.chat patched; capture the spawned instance."""
        captured: dict = {}

        # Save the real Agent.__init__ before patching, so the spy can call it
        from agents.agent import Agent
        real_init = Agent.__init__

        def spy_init(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["instance"] = self

        with patch("agents.agent.Agent.__init__", spy_init), \
             patch("agents.agent.Agent.chat", return_value=return_text):
            result = _execute_agent_tool(
                args,
                parent_model="qwen3.5:4b",
                parent_backend="ollama",
            )
        return result, captured.get("instance")

    def test_explore_filters_to_readonly_tools(self):
        result, sub = self._spawn_with_capture(
            {"description": "look around", "prompt": "find foo", "subagent_type": "explore"}
        )
        self.assertEqual(
            sub.allowed_tool_names,
            {"read_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch"},
        )
        self.assertEqual(sub.permission_mode, PermissionMode.READ_ONLY)
        self.assertEqual(result, "SUBAGENT_DONE")

    def test_plan_filters_to_readonly_tools(self):
        _result, sub = self._spawn_with_capture(
            {"description": "design", "prompt": "plan x", "subagent_type": "plan"}
        )
        self.assertEqual(
            sub.allowed_tool_names,
            {"read_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch"},
        )
        self.assertEqual(sub.permission_mode, PermissionMode.READ_ONLY)

    def test_verification_includes_bash_and_sleep(self):
        _result, sub = self._spawn_with_capture(
            {"description": "run tests", "prompt": "pytest", "subagent_type": "verification"}
        )
        self.assertIn("bash", sub.allowed_tool_names)
        self.assertIn("Sleep", sub.allowed_tool_names)
        self.assertNotIn("write_file", sub.allowed_tool_names)
        self.assertEqual(sub.permission_mode, PermissionMode.WORKSPACE_WRITE)

    def test_general_purpose_has_all_safe_tools(self):
        _result, sub = self._spawn_with_capture(
            {"description": "do stuff", "prompt": "fix bug", "subagent_type": "general-purpose"}
        )
        self.assertEqual(
            sub.allowed_tool_names,
            {"bash", "read_file", "write_file", "edit_file", "grep", "list_files",
             "list_directory", "WebFetch", "WebSearch", "Sleep", "MultiEdit"},
        )
        self.assertNotIn("Agent", sub.allowed_tool_names)
        self.assertNotIn("AskUserQuestion", sub.allowed_tool_names)
        self.assertNotIn("TodoWrite", sub.allowed_tool_names)

    def test_unknown_subagent_type_falls_back_to_general_purpose(self):
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y", "subagent_type": "bogus-type"}
        )
        # general-purpose has 11 tools (6 base + list_directory + WebFetch
        # + WebSearch + Sleep + MultiEdit)
        self.assertEqual(len(sub.allowed_tool_names), 11)
        self.assertEqual(sub.permission_mode, PermissionMode.WORKSPACE_WRITE)

    def test_missing_subagent_type_defaults_to_general_purpose(self):
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y"}  # no subagent_type
        )
        self.assertEqual(len(sub.allowed_tool_names), 11)

    def test_subagent_type_case_insensitive(self):
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y", "subagent_type": "EXPLORE"}
        )
        self.assertEqual(
            sub.allowed_tool_names,
            {"read_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch"},
        )

    def test_subagent_inherits_parent_model(self):
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y", "subagent_type": "explore"}
        )
        self.assertEqual(sub.model, "qwen3.5:4b")
        self.assertEqual(sub.backend, "ollama")

    def test_subagent_has_extended_tool_round_budget(self):
        """Sub-agents get 32 rounds (Rust parity), not the 10-round default."""
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y", "subagent_type": "explore"}
        )
        self.assertEqual(sub.max_tool_rounds, 32)

    def test_subagent_has_isolated_empty_history(self):
        _result, sub = self._spawn_with_capture(
            {"description": "x", "prompt": "y", "subagent_type": "explore"}
        )
        # Fresh Agent — chat() was mocked so no messages were ever appended
        self.assertEqual(sub.history, [])

    def test_subagent_name_slugified_from_description(self):
        _result, sub = self._spawn_with_capture(
            {"description": "Find ALL the bugs!!", "prompt": "y", "subagent_type": "explore"}
        )
        self.assertEqual(sub.name, "find-all-the-bugs")

    def test_subagent_name_fallback_when_slug_empty(self):
        _result, sub = self._spawn_with_capture(
            {"description": "!@#$%", "prompt": "y"}
        )
        self.assertEqual(sub.name, "subagent")


class AgentToolValidationTests(unittest.TestCase):
    """Input validation errors are returned as strings, not raised."""

    def test_missing_description_is_allowed(self):
        """description is optional — derived from prompt[:60] when omitted."""
        captured: dict = {}
        from agents.agent import Agent
        real_init = Agent.__init__

        def spy_init(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["instance"] = self

        with patch("agents.agent.Agent.__init__", spy_init), \
             patch("agents.agent.Agent.chat", return_value="ok"):
            result = _execute_agent_tool(
                {"prompt": "find all the bugs in the code"},
                parent_model="qwen3.5:4b",
                parent_backend="ollama",
            )
        self.assertEqual(result, "ok")
        # name should be slugified from the prompt prefix
        self.assertEqual(captured["instance"].name, "find-all-the-bugs-in-the-code")

    def test_empty_description_falls_back_to_prompt(self):
        """description="" should also fall back to the prompt prefix."""
        captured: dict = {}
        from agents.agent import Agent
        real_init = Agent.__init__

        def spy_init(self, *a, **kw):
            real_init(self, *a, **kw)
            captured["instance"] = self

        with patch("agents.agent.Agent.__init__", spy_init), \
             patch("agents.agent.Agent.chat", return_value="ok"):
            _execute_agent_tool(
                {"description": "", "prompt": "investigate the auth module"},
                parent_model="qwen3.5:4b",
                parent_backend="ollama",
            )
        self.assertEqual(captured["instance"].name, "investigate-the-auth-module")

    def test_missing_prompt_returns_error(self):
        result = _execute_agent_tool(
            {"description": "x", "prompt": ""},
            parent_model="qwen3.5:4b",
            parent_backend="ollama",
        )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("prompt", result)

    def test_subagent_chat_failure_propagates_as_error(self):
        with patch("agents.agent.Agent.chat", side_effect=RuntimeError("kaboom")):
            result = _execute_agent_tool(
                {"description": "x", "prompt": "y"},
                parent_model="qwen3.5:4b",
                parent_backend="ollama",
            )
        self.assertIn("sub-agent failed", result)
        self.assertIn("kaboom", result)


class AgentHistoryIsolationTests(unittest.TestCase):
    """Sub-agents do NOT touch the parent's history."""

    def test_parent_history_unchanged_by_subagent(self):
        from agents.agent import Agent

        parent = Agent(name="parent", role="orchestrator", model="qwen3.5:4b", backend="ollama")
        parent.history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "do something"},
        ]
        before = list(parent.history)

        with patch("agents.agent.Agent.chat", return_value="ok"):
            _execute_agent_tool(
                {"description": "x", "prompt": "y", "subagent_type": "explore"},
                parent_model=parent.model,
                parent_backend=parent.backend,
            )

        self.assertEqual(parent.history, before, "parent history must not be mutated")


class FilteredToolDefinitionTests(unittest.TestCase):
    """Agent._filtered_tool_definitions and the system-prompt tool list."""

    def test_unfiltered_returns_all_tools(self):
        from agents.agent import Agent

        a = Agent(name="x", role="y", model="qwen3.5:4b", backend="ollama")
        names = {t["function"]["name"] for t in a._filtered_tool_definitions()}
        # Top-level agent — should see all 20 (6 base + Agent + Sleep + WebFetch
        # + WebSearch + AskUserQuestion + TodoWrite + TodoRead + MultiEdit
        # + list_directory + AgentCreate + AgentMessage + AgentGet + AgentList
        # + AgentTerminate)
        self.assertEqual(len(names), 20)
        self.assertIn("bash", names)
        self.assertIn("Agent", names)
        self.assertIn("Sleep", names)
        self.assertIn("WebFetch", names)
        self.assertIn("WebSearch", names)
        self.assertIn("AskUserQuestion", names)
        self.assertIn("TodoWrite", names)
        self.assertIn("TodoRead", names)
        self.assertIn("MultiEdit", names)
        self.assertIn("list_directory", names)

    def test_filtered_returns_only_allowed(self):
        from agents.agent import Agent

        a = Agent(name="x", role="y", model="qwen3.5:4b", backend="ollama")
        a.allowed_tool_names = {"read_file", "grep"}
        names = {t["function"]["name"] for t in a._filtered_tool_definitions()}
        self.assertEqual(names, {"read_file", "grep"})

    def test_system_prompt_lists_filtered_tools(self):
        from agents.agent import Agent

        a = Agent(name="x", role="y", model="qwen3.5:4b", backend="ollama")
        a.allowed_tool_names = {"read_file", "grep"}
        prompt = a.build_system_prompt()
        self.assertIn("read_file", prompt)
        self.assertIn("grep", prompt)
        # Filtered-out tools must NOT appear in the available-tools line
        # (we check the specific Available-tools line to avoid false positives
        # from the role/system-prompt body)
        avail_line = next(
            (line for line in prompt.split("\n") if line.startswith("Available tools:")),
            "",
        )
        self.assertNotIn("bash", avail_line)
        self.assertNotIn("write_file", avail_line)
        self.assertNotIn("Agent", avail_line)


class LlamacppArgumentParsingTests(unittest.TestCase):
    """Regression: _call_llamacpp must JSON-parse tool_call arguments to dict.

    The OpenAI-compatible API returns `arguments` as a JSON string. The streaming
    variant parses it (line ~432); the non-streaming variant must too, otherwise
    sub-agents (which call chat() without on_event → use the non-streaming path)
    would pass a string to tool dispatchers that do args["key"] → TypeError.
    """

    def test_non_streaming_parses_string_arguments_to_dict(self):
        from agents.agent import Agent
        import json as _json

        a = Agent(name="x", role="y", model="qwen3.5:4b", backend="llamacpp",
                  max_context_tokens=8192)
        # Mock the HTTP layer to return an OpenAI-style response with string args
        fake_response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"pattern": "agents/*.py"}',  # STRING
                        },
                    }],
                }
            }]
        }

        class FakeResp:
            def read(self_inner):
                return _json.dumps(fake_response).encode()
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): pass

        with patch.object(a, "_request_with_retry", return_value=FakeResp()):
            result = a._call_llamacpp([{"role": "user", "content": "x"}])

        tc = result["message"]["tool_calls"][0]
        self.assertIsInstance(tc["function"]["arguments"], dict)
        self.assertEqual(tc["function"]["arguments"], {"pattern": "agents/*.py"})

    def test_non_streaming_handles_empty_arguments(self):
        from agents.agent import Agent
        import json as _json

        a = Agent(name="x", role="y", model="qwen3.5:4b", backend="llamacpp",
                  max_context_tokens=8192)
        fake_response = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "list_files", "arguments": ""},
                    }],
                }
            }]
        }

        class FakeResp:
            def read(self_inner):
                return _json.dumps(fake_response).encode()
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): pass

        with patch.object(a, "_request_with_retry", return_value=FakeResp()):
            result = a._call_llamacpp([{"role": "user", "content": "x"}])

        tc = result["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["arguments"], {})


class ExecuteToolDispatchTests(unittest.TestCase):
    """The top-level execute_tool() routes Agent calls correctly."""

    def test_execute_tool_routes_agent_to_helper(self):
        with patch("agents.tools._execute_agent_tool", return_value="ROUTED") as mock:
            result = execute_tool(
                "Agent",
                {"description": "x", "prompt": "y"},
                parent_model="qwen3.5:4b",
                parent_backend="ollama",
            )
        self.assertEqual(result, "ROUTED")
        mock.assert_called_once()
        # Verify the helper got the parent context kwargs
        _args, kwargs = mock.call_args
        self.assertEqual(kwargs["parent_model"], "qwen3.5:4b")
        self.assertEqual(kwargs["parent_backend"], "ollama")

    def test_execute_tool_unknown_tool_returns_error(self):
        result = execute_tool("BogusTool", {})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("unknown tool", result)


if __name__ == "__main__":
    unittest.main()
