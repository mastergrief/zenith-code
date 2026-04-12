"""Tests for the CALM wasm backend."""

import pytest

from calm.stack_vm import VMState, default_dispatcher, parse_program, CalmRuntimeError
from calm.backends import wasm_ops


@pytest.fixture
def dispatcher():
    d = default_dispatcher()
    wasm_ops.register(d)
    return d


def _run(dispatcher, source: str) -> VMState:
    state = VMState()
    for instr in parse_program(source):
        dispatcher.execute(state, instr)
    return state


class TestIntegerArithmetic:
    def test_add(self, dispatcher):
        assert _run(dispatcher, "push 17\npush 23\nwasm.add").stack == [40]

    def test_sub(self, dispatcher):
        assert _run(dispatcher, "push 100\npush 37\nwasm.sub").stack == [63]

    def test_mul(self, dispatcher):
        assert _run(dispatcher, "push 17\npush 23\nwasm.mul").stack == [391]

    def test_div(self, dispatcher):
        assert _run(dispatcher, "push 100\npush 4\nwasm.div").stack == [25]

    def test_mod(self, dispatcher):
        assert _run(dispatcher, "push 100\npush 3\nwasm.mod").stack == [1]

    def test_neg(self, dispatcher):
        assert _run(dispatcher, "push 42\nwasm.neg").stack == [-42]

    def test_abs(self, dispatcher):
        assert _run(dispatcher, "push -42\nwasm.abs").stack == [42]

    def test_pow(self, dispatcher):
        assert _run(dispatcher, "push 2\npush 10\nwasm.pow").stack == [1024]

    def test_gcd(self, dispatcher):
        assert _run(dispatcher, "push 391\npush 782\nwasm.gcd").stack == [391]

    def test_gcd_coprime(self, dispatcher):
        assert _run(dispatcher, "push 7\npush 13\nwasm.gcd").stack == [1]

    def test_large_mul(self, dispatcher):
        assert _run(dispatcher, "push 12345\npush 67890\nwasm.mul").stack == [838102050]

    def test_negative_operands(self, dispatcher):
        assert _run(dispatcher, "push -5\npush 3\nwasm.add").stack == [-2]
        assert _run(dispatcher, "push -5\npush -3\nwasm.mul").stack == [15]


class TestFloatArithmetic:
    def test_sqrt(self, dispatcher):
        assert _run(dispatcher, "push 1764\nwasm.sqrt").stack == [42.0]

    def test_sqrt_non_perfect(self, dispatcher):
        result = _run(dispatcher, "push 2\nwasm.sqrt").stack[0]
        assert abs(result - 1.41421356) < 0.0001

    def test_floor(self, dispatcher):
        assert _run(dispatcher, "push 3.7\nwasm.floor").stack == [3.0]

    def test_ceil(self, dispatcher):
        assert _run(dispatcher, "push 3.1\nwasm.ceil").stack == [4.0]

    def test_fabs(self, dispatcher):
        assert _run(dispatcher, "push -3.14\nwasm.fabs").stack == [3.14]


class TestAutoDispatch:
    """wasm.add/sub/mul/div auto-detect int vs float."""

    def test_int_stays_int(self, dispatcher):
        result = _run(dispatcher, "push 5\npush 3\nwasm.add").stack[0]
        assert isinstance(result, int)
        assert result == 8

    def test_float_stays_float(self, dispatcher):
        result = _run(dispatcher, "push 5.0\npush 3.0\nwasm.add").stack[0]
        assert isinstance(result, float)
        assert result == 8.0

    def test_mixed_promotes_to_float(self, dispatcher):
        result = _run(dispatcher, "push 5\npush 3.0\nwasm.mul").stack[0]
        assert isinstance(result, float)
        assert result == 15.0


class TestAgreementWithBuiltins:
    """Wasm backends must agree with stack_vm builtins."""

    @pytest.mark.parametrize("a,b", [
        (0, 0), (1, 1), (17, 23), (100, 37), (-5, 3), (999, 1),
        (42, 19), (12345, 67890),
    ])
    def test_add_agreement(self, dispatcher, a, b):
        wasm = _run(dispatcher, f"push {a}\npush {b}\nwasm.add").stack[0]
        builtin = _run(dispatcher, f"push {a}\npush {b}\nadd").stack[0]
        assert wasm == builtin

    @pytest.mark.parametrize("a,b", [
        (1, 1), (17, 23), (7, 8), (42, 19), (100, 3),
    ])
    def test_mul_agreement(self, dispatcher, a, b):
        wasm = _run(dispatcher, f"push {a}\npush {b}\nwasm.mul").stack[0]
        builtin = _run(dispatcher, f"push {a}\npush {b}\nmul").stack[0]
        assert wasm == builtin


class TestErrorHandling:
    def test_type_error(self, dispatcher):
        with pytest.raises(CalmRuntimeError, match="need numeric"):
            _run(dispatcher, 'push "hello"\npush 1\nwasm.add')

    def test_underflow(self, dispatcher):
        with pytest.raises(CalmRuntimeError, match="underflow"):
            _run(dispatcher, "push 1\nwasm.gcd")

    def test_div_by_zero(self, dispatcher):
        # wasm traps on integer division by zero
        with pytest.raises(CalmRuntimeError):
            _run(dispatcher, "push 10\npush 0\nwasm.div")
