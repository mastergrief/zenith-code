"""Tests for CALM triple modular redundancy verifier."""

import pytest

from calm.stack_vm import VMState, parse_program, CalmRuntimeError, Instruction, _pop_n
from calm.verifier import VerifiedDispatcher, make_verified_dispatcher
from calm.interceptor import EventType, Interceptor


class TestVerifiedDispatcher:
    """Core verifier behavior."""

    @pytest.fixture
    def vd(self):
        return make_verified_dispatcher()

    def _run(self, vd, source: str) -> VMState:
        state = VMState()
        for instr in parse_program(source):
            vd.execute(state, instr)
        return state

    def test_wasm_mul_verified(self, vd):
        self._run(vd, "push 17\npush 23\nwasm.mul")
        v = vd.last_verification
        assert v is not None
        assert v.unanimous
        assert v.primary.stack == [391]

    def test_wasm_gcd_quad(self, vd):
        """GCD has 4 lanes: wasm, math_ops, binary GCD, property proof."""
        self._run(vd, "push 391\npush 782\nwasm.gcd")
        v = vd.last_verification
        assert v.unanimous
        assert len(v.all_results) == 4
        for r in v.all_results:
            assert r.stack == [391], f"{r.name} got {r.stack}"

    def test_wasm_sqrt_quad(self, vd):
        """Sqrt has 4 lanes: wasm, math_ops, Newton's method, inverse proof."""
        self._run(vd, "push 1764\nwasm.sqrt")
        v = vd.last_verification
        assert v.unanimous
        assert len(v.all_results) == 4
        for r in v.all_results:
            assert r.stack == [42.0], f"{r.name} got {r.stack}"

    def test_wasm_pow_quad(self, vd):
        """Pow has 4 lanes: wasm, math_ops, squaring, inverse proof."""
        self._run(vd, "push 2\npush 10\nwasm.pow")
        v = vd.last_verification
        assert v.unanimous
        assert len(v.all_results) == 4
        for r in v.all_results:
            assert r.stack == [1024], f"{r.name} got {r.stack}"

    def test_math_gcd_cross_checked(self, vd):
        """math.gcd is also cross-checked against wasm."""
        self._run(vd, "push 12\npush 8\nmath.gcd")
        v = vd.last_verification
        assert v is not None
        assert v.unanimous
        assert v.primary.stack == [4]

    def test_builtins_no_verification(self, vd):
        """Builtins without shadows don't trigger verification."""
        self._run(vd, "push 1\npush 2\nadd")
        v = vd.last_verification
        assert v is None  # builtins have no shadows

    def test_no_verification_for_push(self, vd):
        self._run(vd, "push 42")
        assert vd.last_verification is None


class TestDivergenceDetection:
    """Verify that disagreements between lanes are caught."""

    def test_buggy_shadow_detected(self):
        from calm.stack_vm import default_dispatcher
        vd = VerifiedDispatcher()
        base = default_dispatcher()
        for name, fn in base.builtins.items():
            vd.register_builtin(name, fn)

        def buggy(state: VMState, instr: Instruction) -> None:
            _pop_n(state, 2, "buggy")
            state.stack.append(999)

        vd.register_shadow("add", "buggy", buggy)

        state = VMState()
        for instr in parse_program("push 1\npush 2\nadd"):
            vd.execute(state, instr)

        v = vd.last_verification
        assert not v.unanimous
        assert v.primary.stack == [3]
        assert v.shadows[0].stack == [999]

    def test_shadow_error_is_divergence(self):
        from calm.stack_vm import default_dispatcher
        vd = VerifiedDispatcher()
        base = default_dispatcher()
        for name, fn in base.builtins.items():
            vd.register_builtin(name, fn)

        def exploding(state: VMState, instr: Instruction) -> None:
            raise CalmRuntimeError("boom")

        vd.register_shadow("add", "exploding", exploding)

        state = VMState()
        for instr in parse_program("push 1\npush 2\nadd"):
            vd.execute(state, instr)

        v = vd.last_verification
        assert not v.unanimous
        assert v.primary.stack == [3]
        assert v.shadows[0].error == "boom"


class TestInterceptorIntegration:
    """Verifier events surface through the interceptor."""

    def test_verified_events_emitted(self):
        vd = make_verified_dispatcher()
        ic = Interceptor(dispatcher=vd)
        events = ic.feed("<calm>\npush 2\npush 10\nwasm.pow -> <pending>\n</calm>")

        verified = [e for e in events if e.type == EventType.VERIFIED]
        assert len(verified) >= 1
        assert "lanes agree" in verified[0].text

    def test_divergence_events_emitted(self):
        from calm.stack_vm import default_dispatcher
        vd = VerifiedDispatcher()
        base = default_dispatcher()
        for name, fn in base.builtins.items():
            vd.register_builtin(name, fn)

        def buggy(state: VMState, instr: Instruction) -> None:
            _pop_n(state, 2, "buggy")
            state.stack.append(-1)

        vd.register_shadow("add", "buggy", buggy)
        ic = Interceptor(dispatcher=vd)
        events = ic.feed("<calm>\npush 1\npush 2\nadd\n</calm>")

        divergence = [e for e in events if e.type == EventType.DIVERGENCE]
        assert len(divergence) == 1
        assert "DIVERGENCE" in divergence[0].text


class TestAgreementBattery:
    """Exhaustive agreement checks across all three-lane operations."""

    @pytest.fixture
    def vd(self):
        return make_verified_dispatcher()

    @pytest.mark.parametrize("a,b", [
        (0, 1), (1, 0), (7, 13), (12, 8), (100, 75),
        (391, 782), (1000, 1000), (17, 23), (48, 36),
    ])
    def test_gcd_triple_agreement(self, vd, a, b):
        state = VMState()
        for instr in parse_program(f"push {a}\npush {b}\nwasm.gcd"):
            vd.execute(state, instr)
        assert vd.last_verification.unanimous, (
            f"gcd({a},{b}) diverged: "
            + ", ".join(f"{r.name}={r.stack}" for r in vd.last_verification.all_results)
        )

    @pytest.mark.parametrize("n", [0, 1, 2, 4, 9, 16, 25, 100, 1764, 10000])
    def test_sqrt_triple_agreement(self, vd, n):
        state = VMState()
        for instr in parse_program(f"push {n}\nwasm.sqrt"):
            vd.execute(state, instr)
        assert vd.last_verification.unanimous, (
            f"sqrt({n}) diverged: "
            + ", ".join(f"{r.name}={r.stack}" for r in vd.last_verification.all_results)
        )

    @pytest.mark.parametrize("base,exp", [
        (2, 0), (2, 1), (2, 10), (2, 20), (3, 5), (10, 3), (7, 4),
    ])
    def test_pow_triple_agreement(self, vd, base, exp):
        state = VMState()
        for instr in parse_program(f"push {base}\npush {exp}\nwasm.pow"):
            vd.execute(state, instr)
        assert vd.last_verification.unanimous, (
            f"pow({base},{exp}) diverged: "
            + ", ".join(f"{r.name}={r.stack}" for r in vd.last_verification.all_results)
        )
