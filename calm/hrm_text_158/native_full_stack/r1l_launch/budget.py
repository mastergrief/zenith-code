"""Pure budget arithmetic for R1-L topology-(c) spawn bounds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

MONITOR_TIMEOUT_MS_MAX = 3_600_000
DEFAULT_KILL_AFTER_S = 60
DEFAULT_ORCH_MARGIN_S = 35
MARGIN_CAP_S = 480


@dataclass(frozen=True)
class BudgetPlan:
    per_phase: Mapping[str, int]
    total_seconds: int
    orchestration_margin_seconds: int
    outer_timeout_seconds: int
    kill_after_seconds: int
    child_wall_bound_seconds: int
    monitor_timeout_ms: int
    margin_cap_seconds: int = MARGIN_CAP_S

    def as_dict(self) -> dict:
        return {
            "per_phase": dict(self.per_phase),
            "total_seconds": self.total_seconds,
            "orchestration_margin_seconds": self.orchestration_margin_seconds,
            "outer_timeout_seconds": self.outer_timeout_seconds,
            "kill_after_seconds": self.kill_after_seconds,
            "child_wall_bound_seconds": self.child_wall_bound_seconds,
            "monitor_timeout_ms": self.monitor_timeout_ms,
            "margin_cap_seconds": self.margin_cap_seconds,
        }


def derive_budget_plan(
    per_phase: Mapping[str, int],
    *,
    orch_margin_s: int = DEFAULT_ORCH_MARGIN_S,
    kill_after_s: int = DEFAULT_KILL_AFTER_S,
    monitor_timeout_ms: int = 3_300_000,
) -> BudgetPlan:
    if orch_margin_s < 0 or kill_after_s < 0:
        raise ValueError("margins must be non-negative")
    if orch_margin_s > MARGIN_CAP_S:
        raise ValueError(f"orch_margin_s {orch_margin_s} exceeds cap {MARGIN_CAP_S}")
    cleaned = {str(k): int(v) for k, v in per_phase.items()}
    if any(v <= 0 for v in cleaned.values()):
        raise ValueError("per-phase budgets must be positive")
    total = sum(cleaned.values())
    outer = total + orch_margin_s
    child_wall = outer + kill_after_s
    if monitor_timeout_ms > MONITOR_TIMEOUT_MS_MAX:
        raise ValueError(
            f"monitor_timeout_ms {monitor_timeout_ms} exceeds ceiling {MONITOR_TIMEOUT_MS_MAX}"
        )
    if monitor_timeout_ms <= child_wall * 1000:
        raise ValueError(
            f"monitor_timeout_ms {monitor_timeout_ms} must exceed child_wall_ms {child_wall * 1000}"
        )
    if child_wall * 1000 >= MONITOR_TIMEOUT_MS_MAX:
        raise ValueError(
            f"child_wall {child_wall}s exceeds monitor ceiling; reduce budgets or margin"
        )
    return BudgetPlan(
        per_phase=cleaned,
        total_seconds=total,
        orchestration_margin_seconds=orch_margin_s,
        outer_timeout_seconds=outer,
        kill_after_seconds=kill_after_s,
        child_wall_bound_seconds=child_wall,
        monitor_timeout_ms=monitor_timeout_ms,
    )
