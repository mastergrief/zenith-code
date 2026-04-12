"""Integration tests: grammar + interceptor + stack_vm working together."""

import subprocess
import tempfile

import pytest

from calm.grammar import generate_gbnf, write_gbnf
from calm.interceptor import EventType, Interceptor
from calm.stack_vm import run


# Path to llama.cpp GBNF validator (may not exist in CI).
GBNF_VALIDATOR = "/home/gabe/llama.cpp/build/bin/test-gbnf-validator"


def _validator_available():
    try:
        subprocess.run([GBNF_VALIDATOR], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _validate_gbnf(grammar_str: str, input_str: str) -> bool:
    """Run llama.cpp GBNF validator, return True if input is valid."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gbnf", delete=False) as gf:
        gf.write(grammar_str)
        gf.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as inf:
            inf.write(input_str)
            inf.flush()
            result = subprocess.run(
                [GBNF_VALIDATOR, gf.name, inf.name],
                capture_output=True, text=True, timeout=10,
            )
            return "valid" in result.stdout and "invalid" not in result.stdout


# -- Programs that exercise the full pipeline --

PROGRAMS = {
    "simple_add": {
        "calm": (
            "<calm>\n"
            "push 17\n"
            "push 23\n"
            "add -> [40]\n"
            "emit -> []\n"
            "halt\n"
            "</calm>"
        ),
        "expected_output": [40],
    },
    "fibonacci_step": {
        "calm": (
            "<calm>\n"
            "\\ compute fib(6) via repeated fib_step\n"
            ": fib_step\n"
            "over\n"
            "add\n"
            "swap\n"
            ";\n"
            "push 0\n"
            "push 1\n"
            "fib_step\n"
            "fib_step\n"
            "fib_step\n"
            "fib_step\n"
            "fib_step -> [5, 3]\n"
            "swap\n"
            "emit -> [3]\n"
            "halt\n"
            "</calm>"
        ),
        "expected_output": [5],
    },
    "multi_type": {
        "calm": (
            "<calm>\n"
            "push 42\n"
            'push "hello"\n'
            "push true\n"
            "push 3.14 -> [42, \"hello\", true, 3.14]\n"
            "</calm>"
        ),
        "expected_output": [],
    },
}


class TestGrammarAcceptance:
    """Generated grammar accepts all valid CALM programs."""

    @pytest.fixture
    def grammar(self):
        return generate_gbnf()

    @pytest.mark.skipif(not _validator_available(), reason="GBNF validator not found")
    @pytest.mark.parametrize("name", PROGRAMS.keys())
    def test_grammar_accepts(self, grammar, name):
        assert _validate_gbnf(grammar, PROGRAMS[name]["calm"])

    @pytest.mark.skipif(not _validator_available(), reason="GBNF validator not found")
    def test_grammar_rejects_invalid(self, grammar):
        invalid = "<calm>\n42 is not an instruction\n</calm>"
        assert not _validate_gbnf(grammar, invalid)


class TestInterceptorExecution:
    """Interceptor produces correct events and state for each program."""

    @pytest.mark.parametrize("name", PROGRAMS.keys())
    def test_execution(self, name):
        prog = PROGRAMS[name]
        ic = Interceptor()
        events = ic.feed(prog["calm"])

        # Must have start + end.
        types = [e.type for e in events]
        assert EventType.CALM_START in types
        assert EventType.CALM_END in types

        # No errors.
        errors = [e for e in events if e.type == EventType.ERROR]
        assert errors == [], f"unexpected errors: {[e.text for e in errors]}"

        # No mismatches.
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert mismatches == [], f"unexpected mismatches: {[e.text for e in mismatches]}"

        # Check output matches.
        assert ic.state.output == prog["expected_output"]


class TestStackVmAgreement:
    """Interceptor's VM state agrees with standalone stack_vm.run()."""

    @pytest.mark.parametrize("source,expected_stack", [
        ("push 10\npush 20\nadd", [30]),
        ("push 5\ndup\nmul", [25]),
        ("push 100\npush 3\nmod", [1]),
        ("push 1\npush 2\npush 3\nrot", [2, 3, 1]),
        ("push 10\nneg\nabs", [10]),
    ])
    def test_agreement(self, source, expected_stack):
        # Standalone VM.
        vm_state = run(source)
        assert vm_state.stack == expected_stack

        # Interceptor.
        ic = Interceptor()
        ic.feed(f"<calm>\n{source}\n</calm>")
        assert ic.state.stack == expected_stack


class TestOptionBEndToEnd:
    """Option B mismatch detection works across the full pipeline."""

    def test_deliberate_mismatch_detected(self):
        ic = Interceptor()
        events = ic.feed(
            "<calm>\n"
            "push 10\n"
            "push 10\n"
            "add -> [21]\n"  # Wrong! Should be 20.
            "</calm>"
        )
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert len(mismatches) == 1
        assert mismatches[0].claimed_stack == [21]
        assert mismatches[0].actual_stack == [20]

    def test_correct_claims_pass(self):
        ic = Interceptor()
        events = ic.feed(
            "<calm>\n"
            "push 2 -> [2]\n"
            "push 3 -> [2, 3]\n"
            "mul -> [6]\n"
            "push 1 -> [6, 1]\n"
            "add -> [7]\n"
            "</calm>"
        )
        validated = [e for e in events if e.type == EventType.VALIDATED]
        mismatches = [e for e in events if e.type == EventType.MISMATCH]
        assert len(validated) == 5
        assert len(mismatches) == 0
