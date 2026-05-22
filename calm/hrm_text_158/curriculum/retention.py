"""Phase 3 cross-rung retention probe utility.

Per codex msg 1779458774209 + 1779457170889:

After each rung's K epochs, probe must run over:
- New rung held-out (acquisition signal)
- ALL prior rungs' held-out (retention signal)
- Canonical 17×23 (multiplication-rung mastery gate, fires after R3)

Retention delta = accuracy(prior_rung) at current ckpt - accuracy(prior_rung) at last_ckpt_of_that_rung.

Pass/fail gate (G2 from Step 0 design): no prior rung drops > 10% absolute.
"""
from __future__ import annotations

from calm.hrm_text_158.curriculum.schema import RungProbeResult


def compute_retention_deltas(
    current: RungProbeResult,
    prior: RungProbeResult | None,
) -> dict[str, float]:
    """Compute per-prior-rung retention deltas (accuracy change vs previous probe).

    Args:
        current: probe result from the CURRENT rung's ckpt (after K epochs of new rung)
        prior:   probe result from the PREVIOUS rung's ckpt (baseline for retention)
                 None means no prior rung exists (current rung is R0).

    Returns:
        dict[rung_name, delta] where delta = current.rung_accuracy[r] - prior.rung_accuracy[r]
        for every rung present in both current and prior. Positive = improvement;
        negative = forgetting.

    Notes:
        - If a rung is in `current.rung_accuracy` but NOT in `prior.rung_accuracy`,
          it's a newly-added rung (not a retention check) and is excluded from
          this delta dict.
        - If `prior` is None, returns {}.
    """
    if prior is None:
        return {}

    deltas: dict[str, float] = {}
    for rung_name, current_acc in current.rung_accuracy.items():
        prior_acc = prior.rung_accuracy.get(rung_name)
        if prior_acc is None:
            continue  # rung didn't exist in prior probe
        deltas[rung_name] = current_acc - prior_acc
    return deltas


def check_retention_gate(
    deltas: dict[str, float],
    threshold: float = -0.10,
) -> tuple[bool, list[str]]:
    """G2 gate: no prior rung drops > 10% absolute accuracy.

    Args:
        deltas: retention deltas (see compute_retention_deltas)
        threshold: minimum acceptable delta (default -0.10 = -10pp)

    Returns:
        (passed, violating_rungs)
        passed=False if any rung's delta < threshold.
    """
    violations = [r for r, d in deltas.items() if d < threshold]
    return (len(violations) == 0), violations
