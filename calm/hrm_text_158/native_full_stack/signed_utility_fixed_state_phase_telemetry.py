"""Phase begin/end timers and hard-budget receipts (PLAN v6 D1)."""
from __future__ import annotations

import time
from typing import Any, Mapping


class PhaseBudgetBreach(RuntimeError):
    def __init__(self, phase: str, elapsed_s: float, budget_s: float) -> None:
        super().__init__(f"phase_budget_breach:{phase}:{elapsed_s:.6f}>{budget_s:.6f}")
        self.phase = phase
        self.elapsed_s = float(elapsed_s)
        self.budget_s = float(budget_s)


class PhaseBudgetClock:
    """Emit PHASE_<NAME>_BEGIN/END markers and enforce hard budgets."""

    def __init__(self, budgets_secs: Mapping[str, float]) -> None:
        self._budgets = {str(k): float(v) for k, v in budgets_secs.items()}
        self._open: dict[str, float] = {}
        self.markers: dict[str, bool] = {}
        self.receipt: dict[str, Any] = {"phases": {}, "breaches": []}

    def begin(self, phase: str) -> str:
        name = str(phase)
        if name in self._open:
            raise RuntimeError(f"phase_already_open:{name}")
        self._open[name] = time.perf_counter()
        marker = f"PHASE_{name}_BEGIN"
        self.markers[marker] = True
        return marker

    def end(self, phase: str) -> str:
        name = str(phase)
        if name not in self._open:
            raise RuntimeError(f"phase_not_open:{name}")
        started = self._open.pop(name)
        elapsed = time.perf_counter() - started
        budget = self._budgets.get(name)
        self.receipt["phases"][name] = {"elapsed_s": elapsed, "budget_s": budget}
        marker = f"PHASE_{name}_END"
        self.markers[marker] = True
        if budget is not None and elapsed > budget:
            self.receipt["breaches"].append({"phase": name, "elapsed_s": elapsed, "budget_s": budget})
            raise PhaseBudgetBreach(name, elapsed, budget)
        return marker

    def markers_mapping(self) -> dict[str, bool]:
        return dict(self.markers)


def phase_marker_pair(phase: str) -> tuple[str, str]:
    return f"PHASE_{phase}_BEGIN", f"PHASE_{phase}_END"


__all__ = [
    "PhaseBudgetBreach",
    "PhaseBudgetClock",
    "phase_marker_pair",
]
