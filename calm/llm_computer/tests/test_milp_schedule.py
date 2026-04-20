"""R17 stub: MILP scheduler API surface (port pending).

Tests confirm the stub returns None when PuLP is absent — callers can
detect and fall back to greedy auto_schedule without try/except.
"""

from __future__ import annotations

from calm.llm_computer.gate_graph import GateGraph
from calm.llm_computer.milp_schedule import is_available, milp_schedule


def test_is_available_returns_bool():
    """is_available() returns a plain boolean."""
    result = is_available()
    assert isinstance(result, bool)


def test_milp_schedule_returns_none_when_pulp_absent():
    """When PuLP isn't installed, milp_schedule returns None so caller
    can fall back cleanly. No exception raised."""
    if is_available():
        # PuLP is installed — stub should still return None until port
        # lands. Confirm this is the case so we know to remove the test
        # when the actual port ships.
        pass
    g = GateGraph()
    result = milp_schedule(g)
    # Stub always returns None (whether PuLP absent OR port not landed)
    assert result is None


def test_milp_schedule_accepts_kwargs():
    """Signature matches the upstream — max_layers / max_ffn / log
    accepted without TypeError."""
    g = GateGraph()
    result = milp_schedule(g, max_layers=10, max_ffn=64, log=print)
    assert result is None
