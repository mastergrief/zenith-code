"""Tests for CALM v0.1 stream interceptor (Option B)."""

import pytest
from calm.interceptor import EventType, Interceptor


class TestBasicFlow:
    """CALM block detection and simple instruction execution."""

    def test_detect_calm_block(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 1\n</calm>")
        types = [e.type for e in events]
        assert EventType.CALM_START in types
        assert EventType.CALM_END in types
        assert EventType.EXECUTED in types

    def test_ignore_outside_calm(self):
        ic = Interceptor()
        events = ic.feed("some thinking text\npush 1\nmore text")
        assert events == []

    def test_simple_arithmetic(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 17\npush 23\nadd\n</calm>")
        executed = [e for e in events if e.type == EventType.EXECUTED]
        assert len(executed) == 3  # push, push, add
        assert executed[-1].actual_stack == [40]

    def test_multiple_instructions_accumulate(self):
        ic = Interceptor()
        ic.feed("<calm>\npush 1\n")
        events = ic.feed("push 2\nadd\n</calm>")
        executed = [e for e in events if e.type == EventType.EXECUTED]
        # Second feed sees: push 2, add
        assert len(executed) == 2
        assert executed[-1].actual_stack == [3]

    def test_comment_line(self):
        ic = Interceptor()
        events = ic.feed("<calm>\n\\ this is a comment\npush 1\n</calm>")
        comments = [e for e in events if e.type == EventType.COMMENT]
        assert len(comments) == 1

    def test_empty_calm_block(self):
        ic = Interceptor()
        events = ic.feed("<calm>\n</calm>")
        types = [e.type for e in events]
        assert types == [EventType.CALM_START, EventType.CALM_END]

    def test_word_definition_and_call(self):
        ic = Interceptor()
        events = ic.feed(
            "<calm>\n: double\ndup\nadd\n;\npush 7\ndouble\n</calm>"
        )
        executed = [e for e in events if e.type == EventType.EXECUTED]
        # Last executed should show stack = [14]
        assert executed[-1].actual_stack == [14]


class TestOptionB:
    """Stack claim validation (Option B — LLM owns stack state)."""

    def test_correct_claim(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 5 -> [5]\n</calm>")
        validated = [e for e in events if e.type == EventType.VALIDATED]
        assert len(validated) == 1
        assert validated[0].claimed_stack == [5]
        assert validated[0].actual_stack == [5]

    def test_mismatch_claim(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 5\npush 3\nadd -> [9]\n</calm>")
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert len(mismatches) == 1
        assert mismatches[0].claimed_stack == [9]
        assert mismatches[0].actual_stack == [8]
        assert "<error>" in mismatches[0].text
        assert "stack mismatch" in mismatches[0].text

    def test_empty_stack_claim(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 1\nemit -> []\n</calm>")
        validated = [e for e in events if e.type == EventType.VALIDATED]
        assert len(validated) == 1
        assert validated[0].actual_stack == []

    def test_multi_value_claim(self):
        ic = Interceptor()
        events = ic.feed(
            "<calm>\npush 1\npush 2\npush 3 -> [1, 2, 3]\n</calm>"
        )
        validated = [e for e in events if e.type == EventType.VALIDATED]
        assert len(validated) == 1
        assert validated[0].actual_stack == [1, 2, 3]

    def test_claim_with_string(self):
        ic = Interceptor()
        events = ic.feed('<calm>\npush "hi" -> ["hi"]\n</calm>')
        validated = [e for e in events if e.type == EventType.VALIDATED]
        assert len(validated) == 1

    def test_no_claim_no_validation(self):
        """Lines without -> [...] should execute but not validate."""
        ic = Interceptor()
        events = ic.feed("<calm>\npush 5\nadd\n</calm>")
        validated = [e for e in events if e.type == EventType.VALIDATED]
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert len(validated) == 0
        assert len(mismatches) == 0


class TestErrorHandling:
    """Runtime and parse errors produce <error> tags."""

    def test_stack_underflow(self):
        ic = Interceptor()
        events = ic.feed("<calm>\nadd\n</calm>")
        errors = [e for e in events if e.type == EventType.ERROR]
        assert len(errors) == 1
        assert "<error>" in errors[0].text
        assert "underflow" in errors[0].text

    def test_type_error(self):
        ic = Interceptor()
        events = ic.feed('<calm>\npush "a"\npush 1\nadd\n</calm>')
        errors = [e for e in events if e.type == EventType.ERROR]
        assert len(errors) == 1
        assert "numeric" in errors[0].text.lower() or "type" in errors[0].text.lower()

    def test_division_by_zero(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 10\npush 0\ndiv\n</calm>")
        errors = [e for e in events if e.type == EventType.ERROR]
        assert len(errors) == 1
        assert "zero" in errors[0].text


class TestPending:
    """<pending> placeholder — model defers prediction to the VM."""

    def test_pending_resolves(self):
        ic = Interceptor()
        events = ic.feed("<calm>\npush 5\npush 3\nadd -> <pending>\n</calm>")
        resolved = [e for e in events if e.type == EventType.RESOLVED]
        assert len(resolved) == 1
        assert resolved[0].actual_stack == [8]
        assert "-> [8]" in resolved[0].text

    def test_pending_no_mismatch(self):
        """Pending never produces a MISMATCH — it always resolves."""
        ic = Interceptor()
        events = ic.feed("<calm>\npush 99\npush 1\nadd -> <pending>\n</calm>")
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert len(mismatches) == 0

    def test_pending_mixed_with_claims(self):
        ic = Interceptor()
        events = ic.feed(
            "<calm>\npush 2 -> [2]\npush 3 -> [2, 3]\nmul -> <pending>\n</calm>"
        )
        validated = [e for e in events if e.type == EventType.VALIDATED]
        resolved = [e for e in events if e.type == EventType.RESOLVED]
        assert len(validated) == 2  # push 2, push 3
        assert len(resolved) == 1  # mul
        assert resolved[0].actual_stack == [6]


class TestStreamingTokens:
    """Simulate token-at-a-time feeding (partial lines)."""

    def test_partial_tag(self):
        ic = Interceptor()
        events = ic.feed("<cal")
        assert events == []
        events = ic.feed("m>\npush 1\n</calm>")
        types = [e.type for e in events]
        assert EventType.CALM_START in types
        assert EventType.EXECUTED in types
        assert EventType.CALM_END in types

    def test_token_by_token(self):
        ic = Interceptor()
        tokens = ["<", "calm", ">", "\n", "push", " ", "5", "\n", "</", "calm", ">"]
        all_events = []
        for t in tokens:
            all_events.extend(ic.feed(t))
        types = [e.type for e in all_events]
        assert EventType.CALM_START in types
        assert EventType.EXECUTED in types
        assert EventType.CALM_END in types

    def test_partial_instruction(self):
        ic = Interceptor()
        ic.feed("<calm>\npush")
        events = ic.feed(" 42\n</calm>")
        executed = [e for e in events if e.type == EventType.EXECUTED]
        assert len(executed) == 1
        assert executed[0].actual_stack == [42]

    def test_multiple_calm_blocks(self):
        ic = Interceptor()
        events = ic.feed(
            "thinking...<calm>\npush 1\n</calm>"
            " more thinking <calm>\npush 2\n</calm>"
        )
        starts = [e for e in events if e.type == EventType.CALM_START]
        ends = [e for e in events if e.type == EventType.CALM_END]
        assert len(starts) == 2
        assert len(ends) == 2

    def test_reset_between_blocks(self):
        """Each <calm> block gets a fresh VM state."""
        ic = Interceptor()
        events = ic.feed(
            "<calm>\npush 99\n</calm>"
            "<calm>\npush 1 -> [1]\n</calm>"
        )
        # The second block's claim is [1], not [99, 1]
        validated = [e for e in events if e.type == EventType.VALIDATED]
        assert len(validated) == 1
        assert validated[0].actual_stack == [1]
