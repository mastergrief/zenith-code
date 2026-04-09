"""Unit tests for the 5 new tools: Sleep, WebFetch, WebSearch, AskUserQuestion, TodoWrite.

Run: PYTHONPATH=. python3 -m unittest tests.test_new_tools -v

These tests mock urllib.request.urlopen for network tools so they run offline.
"""
from __future__ import annotations

import io
import time
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from agents.tools import (
    ALLOWED_TOOLS_BY_SUBAGENT,
    TOOL_DEFINITIONS,
    _grep,
    _grep_with_python,
    _list_directory,
    _run_ask_user_question,
    _run_multi_edit,
    _run_sleep,
    _run_todo_read,
    _run_todo_write,
    _run_web_fetch,
    _run_web_search,
    execute_tool,
)


def _mock_http_response(body: bytes, content_type: str = "text/html"):
    """Build a context-manager-compatible mock for urllib.request.urlopen."""
    fake = MagicMock()
    fake.headers = {"Content-Type": content_type}
    fake.read = MagicMock(return_value=body)
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    return fake


# ── Sleep ─────────────────────────────────────────────────────────────────


class SleepTests(unittest.TestCase):
    def test_sleep_short_duration(self):
        start = time.monotonic()
        result = _run_sleep({"duration_ms": 50})
        elapsed_ms = (time.monotonic() - start) * 1000
        self.assertEqual(result, "Slept 50ms")
        self.assertGreaterEqual(elapsed_ms, 40)  # tolerance
        self.assertLess(elapsed_ms, 500)

    def test_sleep_zero(self):
        result = _run_sleep({"duration_ms": 0})
        self.assertEqual(result, "Slept 0ms")

    def test_sleep_negative_returns_error(self):
        result = _run_sleep({"duration_ms": -100})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn(">= 0", result)

    def test_sleep_over_cap_returns_error(self):
        result = _run_sleep({"duration_ms": 60001})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("60000", result)

    def test_sleep_missing_duration_returns_error(self):
        result = _run_sleep({})
        self.assertTrue(result.startswith("Error:"))

    def test_sleep_non_int_returns_error(self):
        result = _run_sleep({"duration_ms": "abc"})
        self.assertTrue(result.startswith("Error:"))


# ── WebFetch ──────────────────────────────────────────────────────────────


class WebFetchTests(unittest.TestCase):
    def test_fetch_html_strips_tags(self):
        html = b"<html><body><p>Hello <b>world</b></p></body></html>"
        with patch("urllib.request.urlopen", return_value=_mock_http_response(html)):
            result = _run_web_fetch({"url": "https://example.com/", "prompt": "what does it say"})
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        # No raw HTML tags in extracted content (allow `<` in headers/URLs)
        body = result.split("---\n", 1)[1]
        self.assertNotIn("<p>", body)
        self.assertNotIn("<b>", body)

    def test_fetch_strips_script_and_style(self):
        html = b"<html><head><style>body{color:red}</style></head><body><script>alert(1)</script><p>visible content</p></body></html>"
        with patch("urllib.request.urlopen", return_value=_mock_http_response(html)):
            result = _run_web_fetch({"url": "https://example.com/", "prompt": "x"})
        self.assertIn("visible content", result)
        self.assertNotIn("alert", result)
        self.assertNotIn("color:red", result)

    def test_fetch_truncates_large_body(self):
        # 50KB of plain text
        big = b"<html><body><p>" + b"x" * 50_000 + b"</p></body></html>"
        with patch("urllib.request.urlopen", return_value=_mock_http_response(big)):
            result = _run_web_fetch({"url": "https://example.com/", "prompt": "x"})
        # Total result should be capped — header is small, so body part ≤ ~8050
        self.assertLess(len(result), 9000)
        self.assertIn("truncated", result)

    def test_fetch_rejects_non_http_schemes(self):
        result = _run_web_fetch({"url": "file:///etc/passwd", "prompt": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("scheme", result)

    def test_fetch_rejects_no_host(self):
        result = _run_web_fetch({"url": "https:///foo", "prompt": "x"})
        self.assertTrue(result.startswith("Error:"))

    def test_fetch_rejects_disallowed_content_type(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_http_response(b"\x89PNG", content_type="image/png")):
            result = _run_web_fetch({"url": "https://example.com/img.png", "prompt": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("image/png", result)

    def test_fetch_includes_url_and_prompt_in_output(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_http_response(b"<html><body>hi</body></html>")):
            result = _run_web_fetch({
                "url": "https://example.com/page",
                "prompt": "what is the title",
            })
        self.assertTrue(result.startswith("URL: https://example.com/page"))
        self.assertIn("Prompt: what is the title", result)
        self.assertIn("---", result)

    def test_fetch_handles_http_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       "https://example.com/", 404, "Not Found", {}, None)):
            result = _run_web_fetch({"url": "https://example.com/", "prompt": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("404", result)

    def test_fetch_missing_url_returns_error(self):
        result = _run_web_fetch({"prompt": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("url", result)

    def test_fetch_missing_prompt_returns_error(self):
        result = _run_web_fetch({"url": "https://example.com/"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("prompt", result)

    def test_fetch_plain_text_passthrough(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_http_response(b"raw text content", content_type="text/plain")):
            result = _run_web_fetch({"url": "https://example.com/", "prompt": "x"})
        self.assertIn("raw text content", result)


# ── WebSearch ─────────────────────────────────────────────────────────────


# Realistic-ish DDG HTML structure
_DDG_FAKE_HTML = b"""
<html><body>
<div class="result">
  <a class="result__a" href="https://docs.python.org/3/library/re.html">Python re module</a>
  <a class="result__snippet" href="#">Regular expression operations in Python.</a>
</div>
<div class="result">
  <a class="result__a" href="https://github.com/python/cpython">CPython on GitHub</a>
  <a class="result__snippet" href="#">The CPython source code repository.</a>
</div>
<div class="result">
  <a class="result__a" href="https://stackoverflow.com/questions/python-regex">Stack Overflow regex tag</a>
  <a class="result__snippet" href="#">Q&A about Python regex.</a>
</div>
</body></html>
"""


class WebSearchTests(unittest.TestCase):
    def test_search_parses_ddg_results(self):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(_DDG_FAKE_HTML)):
            result = _run_web_search({"query": "python regex"})
        self.assertIn("Python re module", result)
        self.assertIn("CPython on GitHub", result)
        self.assertIn("Stack Overflow", result)
        self.assertIn("docs.python.org", result)
        self.assertIn("Regular expression", result)

    def test_search_allowed_domains_filter(self):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(_DDG_FAKE_HTML)):
            result = _run_web_search({
                "query": "python regex",
                "allowed_domains": ["python.org"],
            })
        self.assertIn("docs.python.org", result)
        self.assertNotIn("github.com", result)
        self.assertNotIn("stackoverflow.com", result)

    def test_search_blocked_domains_filter(self):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(_DDG_FAKE_HTML)):
            result = _run_web_search({
                "query": "python regex",
                "blocked_domains": ["stackoverflow.com"],
            })
        self.assertIn("docs.python.org", result)
        self.assertIn("github.com", result)
        self.assertNotIn("Stack Overflow", result)

    def test_search_short_query_rejected(self):
        result = _run_web_search({"query": "x"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("2 characters", result)

    def test_search_no_results_when_html_empty(self):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(b"<html></html>")):
            result = _run_web_search({"query": "abc"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("no results parsed", result)

    def test_search_handles_network_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            result = _run_web_search({"query": "python"})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("DuckDuckGo", result)


# ── AskUserQuestion ───────────────────────────────────────────────────────


class AskUserQuestionTests(unittest.TestCase):
    def test_ask_with_options(self):
        captured = {}

        def fake_ask(question, options):
            captured["q"] = question
            captured["o"] = options
            return "blue"

        result = _run_ask_user_question(
            {"question": "favorite color?", "options": ["red", "green", "blue"]},
            fake_ask,
        )
        self.assertEqual(result, "blue")
        self.assertEqual(captured["q"], "favorite color?")
        self.assertEqual(captured["o"], ["red", "green", "blue"])

    def test_ask_freeform_no_options(self):
        result = _run_ask_user_question(
            {"question": "what's your name?"},
            lambda q, o: "Gabe",
        )
        self.assertEqual(result, "Gabe")

    def test_ask_no_callback_returns_error(self):
        result = _run_ask_user_question(
            {"question": "anything?"},
            None,
        )
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("non-interactive", result)

    def test_ask_empty_question_returns_error(self):
        result = _run_ask_user_question({"question": ""}, lambda q, o: "x")
        self.assertTrue(result.startswith("Error:"))

    def test_ask_callback_exception_propagates_as_error(self):
        def bad(_q, _o):
            raise RuntimeError("ui broke")
        result = _run_ask_user_question({"question": "?"}, bad)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("ui broke", result)

    def test_ask_invalid_options_type_returns_error(self):
        result = _run_ask_user_question(
            {"question": "?", "options": "not a list"},
            lambda q, o: "x",
        )
        self.assertTrue(result.startswith("Error:"))


# ── TodoWrite ─────────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal stand-in for an Agent — only needs a `todos` attribute."""
    def __init__(self):
        self.todos: list[dict] = []


class TodoWriteTests(unittest.TestCase):
    def _todos(self):
        return [
            {"content": "Read the spec",  "activeForm": "Reading the spec",  "status": "completed"},
            {"content": "Draft the API",  "activeForm": "Drafting the API",  "status": "in_progress"},
            {"content": "Add tests",      "activeForm": "Adding tests",      "status": "pending"},
        ]

    def test_writes_todos_to_agent(self):
        agent = _FakeAgent()
        result = _run_todo_write({"todos": self._todos()}, agent)
        self.assertEqual(len(agent.todos), 3)
        self.assertEqual(agent.todos[0]["content"], "Read the spec")
        self.assertNotIn("Error", result)

    def test_replaces_existing_todos(self):
        agent = _FakeAgent()
        agent.todos = [{"content": "old", "activeForm": "old-ing", "status": "pending"}]
        _run_todo_write({"todos": self._todos()}, agent)
        self.assertEqual(len(agent.todos), 3)
        self.assertNotIn("old", [t["content"] for t in agent.todos])

    def test_renders_markdown_with_status_markers(self):
        agent = _FakeAgent()
        result = _run_todo_write({"todos": self._todos()}, agent)
        self.assertIn("[x]", result)
        self.assertIn("[~]", result)
        self.assertIn("[ ]", result)
        self.assertIn("1 pending", result)
        self.assertIn("1 in progress", result)
        self.assertIn("1 completed", result)

    def test_in_progress_uses_active_form(self):
        agent = _FakeAgent()
        result = _run_todo_write({"todos": self._todos()}, agent)
        self.assertIn("Drafting the API", result)  # gerund for in_progress
        self.assertIn("Read the spec", result)     # imperative for completed

    def test_validates_status_enum(self):
        agent = _FakeAgent()
        bad = [{"content": "x", "activeForm": "y", "status": "blocked"}]
        result = _run_todo_write({"todos": bad}, agent)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("status", result)

    def test_at_most_one_in_progress(self):
        agent = _FakeAgent()
        bad = [
            {"content": "a", "activeForm": "doing a", "status": "in_progress"},
            {"content": "b", "activeForm": "doing b", "status": "in_progress"},
        ]
        result = _run_todo_write({"todos": bad}, agent)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("at most one", result)

    def test_missing_required_field_returns_error(self):
        agent = _FakeAgent()
        bad = [{"content": "x", "status": "pending"}]  # missing activeForm
        result = _run_todo_write({"todos": bad}, agent)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("activeForm", result)

    def test_no_agent_returns_error(self):
        result = _run_todo_write({"todos": self._todos()}, None)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("agent", result)

    def test_todos_must_be_list(self):
        agent = _FakeAgent()
        result = _run_todo_write({"todos": "not a list"}, agent)
        self.assertTrue(result.startswith("Error:"))


# ── TOOL_DEFINITIONS / ALLOWED_TOOLS_BY_SUBAGENT regression checks ────────


class NewToolDefinitionTests(unittest.TestCase):
    def test_all_five_new_tools_in_definitions(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        for new_tool in ("Sleep", "WebFetch", "WebSearch", "AskUserQuestion", "TodoWrite"):
            self.assertIn(new_tool, names)

    def test_total_tool_count(self):
        # Updated by later batches — see NewToolDefinitionsBatch2Tests for the
        # current authoritative count.
        self.assertGreaterEqual(len(TOOL_DEFINITIONS), 12)


class SubagentAllowlistAdditionsTests(unittest.TestCase):
    def test_explore_now_includes_web_tools(self):
        allowed, _mode = ALLOWED_TOOLS_BY_SUBAGENT["explore"]
        self.assertIn("WebFetch", allowed)
        self.assertIn("WebSearch", allowed)

    def test_plan_now_includes_web_tools(self):
        allowed, _mode = ALLOWED_TOOLS_BY_SUBAGENT["plan"]
        self.assertIn("WebFetch", allowed)
        self.assertIn("WebSearch", allowed)

    def test_verification_includes_sleep(self):
        allowed, _mode = ALLOWED_TOOLS_BY_SUBAGENT["verification"]
        self.assertIn("Sleep", allowed)

    def test_general_purpose_includes_all_safe_new_tools(self):
        allowed, _mode = ALLOWED_TOOLS_BY_SUBAGENT["general-purpose"]
        self.assertIn("WebFetch", allowed)
        self.assertIn("WebSearch", allowed)
        self.assertIn("Sleep", allowed)

    def test_no_subagent_has_ask_user_question_or_todo_write(self):
        for sub_type, (allowed, _mode) in ALLOWED_TOOLS_BY_SUBAGENT.items():
            self.assertNotIn(
                "AskUserQuestion", allowed,
                f"{sub_type} should not include AskUserQuestion (parent-level)",
            )
            self.assertNotIn(
                "TodoWrite", allowed,
                f"{sub_type} should not include TodoWrite (parent-level)",
            )


# ── execute_tool dispatch routing ─────────────────────────────────────────


class ExecuteToolNewDispatchTests(unittest.TestCase):
    def test_execute_tool_routes_sleep(self):
        result = execute_tool("Sleep", {"duration_ms": 0})
        self.assertEqual(result, "Slept 0ms")

    def test_execute_tool_routes_webfetch(self):
        with patch("urllib.request.urlopen",
                   return_value=_mock_http_response(b"<html><body>x</body></html>")):
            result = execute_tool("WebFetch", {"url": "https://example.com/", "prompt": "x"})
        self.assertIn("URL: https://example.com/", result)

    def test_execute_tool_routes_websearch(self):
        with patch("urllib.request.urlopen", return_value=_mock_http_response(_DDG_FAKE_HTML)):
            result = execute_tool("WebSearch", {"query": "python"})
        self.assertIn("Python re module", result)

    def test_execute_tool_routes_askuserquestion(self):
        result = execute_tool(
            "AskUserQuestion",
            {"question": "ok?"},
            ask_user_fn=lambda q, o: "yes",
        )
        self.assertEqual(result, "yes")

    def test_execute_tool_routes_todowrite(self):
        agent = _FakeAgent()
        result = execute_tool(
            "TodoWrite",
            {"todos": [{"content": "do x", "activeForm": "doing x", "status": "pending"}]},
            agent=agent,
        )
        self.assertIn("[ ]", result)
        self.assertEqual(len(agent.todos), 1)


# ── Grep upgrade (ripgrep + Python fallback) ─────────────────────────────


import os
import tempfile


class GrepUpgradeTests(unittest.TestCase):
    """Verify the rg-backed grep, the Python fallback, and the new optional fields."""

    def setUp(self):
        # Build a temp tree with known content
        self.tmp = tempfile.mkdtemp(prefix="zenith-grep-test-")
        with open(os.path.join(self.tmp, "alpha.py"), "w") as f:
            f.write("def hello():\n    return 42\n\ndef world():\n    pass\n")
        with open(os.path.join(self.tmp, "beta.txt"), "w") as f:
            f.write("hello world\nhello there\n")
        with open(os.path.join(self.tmp, "gamma.md"), "w") as f:
            f.write("# Hello\nworld\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_python_fallback_finds_matches(self):
        result = _grep_with_python("hello", path=self.tmp)
        self.assertIn("alpha.py", result)
        self.assertIn("beta.txt", result)

    def test_python_fallback_case_insensitive(self):
        # "Hello" capitalized only in gamma.md (header)
        result = _grep_with_python("Hello", path=self.tmp, case_insensitive=True)
        self.assertIn("alpha.py", result)
        self.assertIn("beta.txt", result)
        self.assertIn("gamma.md", result)

    def test_python_fallback_case_sensitive(self):
        result = _grep_with_python("Hello", path=self.tmp, case_insensitive=False)
        # Only gamma.md has capitalized "Hello"
        self.assertIn("gamma.md", result)
        self.assertNotIn("alpha.py", result)

    def test_python_fallback_invalid_regex(self):
        result = _grep_with_python("(unclosed", path=self.tmp)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("invalid regex", result)

    def test_grep_uses_rg_when_available(self):
        # Force rg-backed path by ensuring cache has the path
        import agents.tools as t
        if t._find_rg() is None:
            self.skipTest("ripgrep not installed")
        result = _grep("def hello", path=self.tmp, file_type="py")
        self.assertIn("alpha.py", result)
        # Should NOT match the .txt or .md files (filetype filter)
        self.assertNotIn("beta.txt", result)
        self.assertNotIn("gamma.md", result)

    def test_grep_with_context(self):
        import agents.tools as t
        if t._find_rg() is None:
            self.skipTest("ripgrep not installed")
        result = _grep("return 42", path=self.tmp, context=1)
        # Context line "def hello():" should appear before the match
        self.assertIn("def hello", result)
        self.assertIn("return 42", result)

    def test_grep_word_match(self):
        import agents.tools as t
        if t._find_rg() is None:
            self.skipTest("ripgrep not installed")
        # "hell" should NOT match "hello" with word_match=True
        result = _grep("hell", path=self.tmp, word_match=True)
        self.assertEqual(result.strip(), "No matches found.")
        # But it should match "hell" as a word
        with open(os.path.join(self.tmp, "delta.txt"), "w") as f:
            f.write("the word hell appears here\n")
        result2 = _grep("hell", path=self.tmp, word_match=True)
        self.assertIn("delta.txt", result2)

    def test_grep_falls_back_when_rg_missing(self):
        """When _find_rg() returns None, grep uses the Python fallback."""
        import agents.tools as t
        original = t._RG_BIN_CACHE
        try:
            t._RG_BIN_CACHE = None  # force fallback
            result = _grep("hello", path=self.tmp)
            self.assertIn("alpha.py", result)
        finally:
            t._RG_BIN_CACHE = original

    def test_grep_no_matches(self):
        import agents.tools as t
        if t._find_rg() is None:
            self.skipTest("ripgrep not installed")
        result = _grep("zzz_no_such_pattern_zzz", path=self.tmp)
        self.assertIn("No matches found", result)


# ── TodoRead ──────────────────────────────────────────────────────────────


class TodoReadTests(unittest.TestCase):
    def test_read_returns_error_without_agent(self):
        result = _run_todo_read(None)
        self.assertTrue(result.startswith("Error:"))

    def test_read_empty_returns_friendly_message(self):
        agent = _FakeAgent()
        result = _run_todo_read(agent)
        self.assertIn("(no todos set yet", result)

    def test_read_returns_current_todos(self):
        agent = _FakeAgent()
        agent.todos = [
            {"content": "Build it", "activeForm": "Building it", "status": "in_progress"},
            {"content": "Test it", "activeForm": "Testing it", "status": "pending"},
        ]
        result = _run_todo_read(agent)
        self.assertIn("[~] Building it", result)
        self.assertIn("[ ] Test it", result)
        self.assertIn("1 pending, 1 in progress, 0 completed", result)

    def test_read_after_write_returns_same_render(self):
        """TodoRead and TodoWrite render identical output for the same state."""
        agent = _FakeAgent()
        todos = [
            {"content": "Step A", "activeForm": "Doing A", "status": "completed"},
            {"content": "Step B", "activeForm": "Doing B", "status": "pending"},
        ]
        write_result = _run_todo_write({"todos": todos}, agent)
        read_result = _run_todo_read(agent)
        self.assertEqual(write_result, read_result)


# ── MultiEdit ─────────────────────────────────────────────────────────────


class MultiEditTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(prefix="zenith-multiedit-", suffix=".txt", text=True)
        os.close(fd)
        with open(self.path, "w") as f:
            f.write("alpha\nbeta\ngamma\nalpha-second\n")

    def tearDown(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def _read(self) -> str:
        with open(self.path) as f:
            return f.read()

    def test_single_edit(self):
        result = _run_multi_edit(self.path, [
            {"old_string": "beta", "new_string": "BETA"},
        ])
        self.assertIn("Applied 1 edit", result)
        self.assertEqual(self._read(), "alpha\nBETA\ngamma\nalpha-second\n")

    def test_sequential_edits_each_sees_prior(self):
        # First edit narrows the duplicate "alpha" → "ALPHA" (replace_all),
        # second edit sees "ALPHA-second" and rewrites it.
        result = _run_multi_edit(self.path, [
            {"old_string": "alpha", "new_string": "ALPHA", "replace_all": True},
            {"old_string": "ALPHA-second", "new_string": "ALPHA-SECOND"},
        ])
        self.assertIn("Applied 2 edit", result)
        self.assertEqual(self._read(), "ALPHA\nbeta\ngamma\nALPHA-SECOND\n")

    def test_atomicity_no_writes_when_one_edit_fails(self):
        before = self._read()
        result = _run_multi_edit(self.path, [
            {"old_string": "beta", "new_string": "BETA"},
            {"old_string": "missing-string", "new_string": "x"},
        ])
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("not found", result)
        # File must be unchanged
        self.assertEqual(self._read(), before)

    def test_replace_all_required_for_duplicate(self):
        result = _run_multi_edit(self.path, [
            {"old_string": "alpha", "new_string": "X"},  # appears twice
        ])
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("appears 2 times", result)

    def test_replace_all_works(self):
        result = _run_multi_edit(self.path, [
            {"old_string": "alpha", "new_string": "X", "replace_all": True},
        ])
        self.assertIn("Applied 1 edit", result)
        # Both "alpha" and the "alpha" inside "alpha-second" replaced
        self.assertEqual(self._read(), "X\nbeta\ngamma\nX-second\n")

    def test_old_equals_new_rejected(self):
        result = _run_multi_edit(self.path, [
            {"old_string": "beta", "new_string": "beta"},
        ])
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("must differ", result)

    def test_missing_old_string_rejected(self):
        result = _run_multi_edit(self.path, [{"new_string": "x"}])
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("old_string", result)

    def test_empty_edits_rejected(self):
        result = _run_multi_edit(self.path, [])
        self.assertTrue(result.startswith("Error:"))

    def test_file_not_found(self):
        result = _run_multi_edit("/nonexistent/path/foo.txt", [
            {"old_string": "x", "new_string": "y"},
        ])
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("not found", result)

    def test_no_op_edits_reported(self):
        # All edits are valid but produce same final content as original
        # (e.g. swap two strings then swap back)
        before = self._read()
        result = _run_multi_edit(self.path, [
            {"old_string": "beta", "new_string": "BETA-temp"},
            {"old_string": "BETA-temp", "new_string": "beta"},
        ])
        self.assertIn("No changes", result)
        self.assertEqual(self._read(), before)


class ExecuteToolGrepUpgradeRoutingTests(unittest.TestCase):
    """execute_tool dispatches to TodoRead and MultiEdit correctly."""

    def test_execute_tool_routes_todoread(self):
        agent = _FakeAgent()
        agent.todos = [{"content": "x", "activeForm": "doing x", "status": "pending"}]
        result = execute_tool("TodoRead", {}, agent=agent)
        self.assertIn("[ ] x", result)

    def test_execute_tool_routes_multiedit(self):
        fd, path = tempfile.mkstemp(prefix="zenith-me-", text=True)
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("hello\nworld\n")
            result = execute_tool(
                "MultiEdit",
                {"path": path, "edits": [{"old_string": "hello", "new_string": "HI"}]},
            )
            self.assertIn("Applied 1 edit", result)
            with open(path) as f:
                self.assertEqual(f.read(), "HI\nworld\n")
        finally:
            os.unlink(path)


class NewToolDefinitionsBatch2Tests(unittest.TestCase):
    def test_total_tool_count_is_twenty(self):
        # See tests.test_agent_registry.NewToolDefinitionsTests for the
        # current authoritative count after Slice 1 (registry tools).
        self.assertGreaterEqual(len(TOOL_DEFINITIONS), 15)

    def test_todo_read_in_definitions(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("TodoRead", names)

    def test_multi_edit_in_definitions(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("MultiEdit", names)

    def test_list_directory_in_definitions(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("list_directory", names)

    def test_list_directory_schema_has_no_required_fields(self):
        """list_directory accepts no required args — defaults to cwd."""
        spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "list_directory")
        params = spec["function"]["parameters"]
        self.assertEqual(params["required"], [])
        self.assertIn("path", params["properties"])
        self.assertIn("show_hidden", params["properties"])

    def test_grep_schema_has_new_optional_fields(self):
        spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "grep")
        props = spec["function"]["parameters"]["properties"]
        for field in ("type", "context", "word_match", "case_insensitive"):
            self.assertIn(field, props, f"grep schema missing new field: {field}")
        # `pattern` still required, others still optional
        self.assertEqual(spec["function"]["parameters"]["required"], ["pattern"])


# ── list_directory ────────────────────────────────────────────────────────


class ListDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zenith-listdir-test-")
        # Build a known structure:
        #   tmp/
        #     subdir/
        #       inner.txt
        #     emptydir/
        #     alpha.txt
        #     beta.py
        #     .hidden_file
        #     .hidden_dir/
        os.makedirs(os.path.join(self.tmp, "subdir"))
        os.makedirs(os.path.join(self.tmp, "emptydir"))
        os.makedirs(os.path.join(self.tmp, ".hidden_dir"))
        with open(os.path.join(self.tmp, "subdir", "inner.txt"), "w") as f:
            f.write("inside")
        with open(os.path.join(self.tmp, "alpha.txt"), "w") as f:
            f.write("a")
        with open(os.path.join(self.tmp, "beta.py"), "w") as f:
            f.write("b")
        with open(os.path.join(self.tmp, ".hidden_file"), "w") as f:
            f.write("h")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_lists_dirs_first_then_files(self):
        result = _list_directory(self.tmp)
        lines = result.splitlines()
        # Header on first line
        self.assertIn("2 dirs", lines[0])
        self.assertIn("2 files", lines[0])
        # Directories should appear before files in body
        body = lines[1:]
        emptydir_idx = next(i for i, l in enumerate(body) if "emptydir/" in l)
        subdir_idx = next(i for i, l in enumerate(body) if "subdir/" in l)
        alpha_idx = next(i for i, l in enumerate(body) if "alpha.txt" in l)
        beta_idx = next(i for i, l in enumerate(body) if "beta.py" in l)
        self.assertLess(emptydir_idx, alpha_idx)
        self.assertLess(subdir_idx, alpha_idx)

    def test_dirs_have_trailing_slash(self):
        result = _list_directory(self.tmp)
        self.assertIn("subdir/", result)
        self.assertIn("emptydir/", result)
        # Files do NOT have trailing slash
        for line in result.splitlines()[1:]:
            stripped = line.strip()
            if stripped == "alpha.txt":
                break
        else:
            self.fail("alpha.txt should appear without trailing slash")

    def test_alphabetical_within_groups(self):
        result = _list_directory(self.tmp)
        lines = [l.strip() for l in result.splitlines()]
        # Within dirs: emptydir/ before subdir/
        self.assertLess(lines.index("emptydir/"), lines.index("subdir/"))
        # Within files: alpha.txt before beta.py
        self.assertLess(lines.index("alpha.txt"), lines.index("beta.py"))

    def test_hidden_files_excluded_by_default(self):
        result = _list_directory(self.tmp)
        self.assertNotIn(".hidden_file", result)
        self.assertNotIn(".hidden_dir", result)
        # But the count of hidden entries should be reported in the header
        self.assertIn("2 hidden", result)

    def test_show_hidden_includes_dotfiles(self):
        result = _list_directory(self.tmp, show_hidden=True)
        self.assertIn(".hidden_file", result)
        self.assertIn(".hidden_dir/", result)
        # Counts now include them
        self.assertIn("3 dirs", result)
        self.assertIn("3 files", result)

    def test_empty_directory(self):
        empty = os.path.join(self.tmp, "emptydir")
        result = _list_directory(empty)
        self.assertIn("(empty)", result)

    def test_nonexistent_path_returns_error(self):
        result = _list_directory("/zzz/does/not/exist")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("does not exist", result)

    def test_file_path_returns_error(self):
        result = _list_directory(os.path.join(self.tmp, "alpha.txt"))
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("not a directory", result)

    def test_default_path_is_cwd(self):
        # No args — defaults to "."
        result = _list_directory()
        self.assertNotIn("Error", result)

    def test_subagent_allowlists_include_list_directory(self):
        for sub_type, (allowed, _mode) in ALLOWED_TOOLS_BY_SUBAGENT.items():
            self.assertIn(
                "list_directory", allowed,
                f"{sub_type} should allow list_directory (read-only)",
            )

    def test_execute_tool_routes_list_directory(self):
        result = execute_tool("list_directory", {"path": self.tmp})
        self.assertIn("subdir/", result)
        self.assertIn("alpha.txt", result)


if __name__ == "__main__":
    unittest.main()
