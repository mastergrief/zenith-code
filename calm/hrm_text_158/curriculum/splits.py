"""Phase 3 curriculum splits + cross-rung invariant gate.

Per codex msg 1779458774209 + 1779460468673:
- Held-out splits NEVER seen in any rung's training data
- Cross-checked: held_out[i] ∩ train[j] = ∅ for all i ≠ j (and i == j)

Tested via set intersection on (question, expected) tuples.
"""
from __future__ import annotations

from calm.hrm_text_158.curriculum.generators import (
    RUNG_NAMES,
    make_rung_examples,
)


def build_rung_splits(
    n_train: int = 2000,
    n_held_out: int = 400,
    seed: int = 42,
    rungs: tuple[str, ...] = ("R0", "R1", "R1b1", "R1b2", "R2", "R3", "R4", "R5", "R6"),
) -> dict[str, dict[str, list[dict]]]:
    """Build train + held_out splits for all rungs.

    Returns:
        {"R0": {"train": [...], "held_out": [...]}, "R1": {...}, ...}

    R7 (GSM8k) excluded — served from load_gsm8k_splits.

    Default tuple is the ACTIVE CHAIN per codex msg 1779469638068:
    R1b is excluded by default because its A_plus_1 rows overlap R1b1's
    by construction (single-template R1b1 succeeds R1b's 3-template
    failure). R1b stays accessible by passing `rungs=("R0", "R1",
    "R1b1", "R1b", ...)` explicitly for diagnosis-only inspection,
    but assert_no_train_holdout_overlap WILL fail on a tuple containing
    both R1b and R1b1.
    """
    out: dict[str, dict[str, list[dict]]] = {}
    for rung in rungs:
        if rung == "R7":
            continue  # GSM8k served separately
        out[rung] = {
            "train": make_rung_examples(rung, n_train, seed=seed, split="train"),
            "held_out": make_rung_examples(rung, n_held_out, seed=seed, split="held_out"),
        }
    return out


def _row_key(row: dict) -> tuple[str, int]:
    """Canonical key for set membership across rungs."""
    return (row["question"], row["expected"])


def assert_no_train_holdout_overlap(splits: dict[str, dict[str, list[dict]]]) -> None:
    """Cross-rung invariant: NO row in any rung's held_out appears in
    ANY rung's train set. Raises AssertionError on violation.

    Per codex spec: prevents held-out leakage across the curriculum
    (e.g., R3's held-out 17×23 must not show up in any rung's train).
    """
    # Build the global train set across all rungs
    all_train_keys: set[tuple[str, int]] = set()
    for rung, splits_d in splits.items():
        for row in splits_d["train"]:
            all_train_keys.add(_row_key(row))

    # Now check each rung's held_out
    violations: list[tuple[str, dict]] = []
    for rung, splits_d in splits.items():
        for row in splits_d["held_out"]:
            if _row_key(row) in all_train_keys:
                violations.append((rung, row))

    if violations:
        # Show up to 5 to keep error message readable
        msg = "held_out ∩ train overlap detected:\n"
        for rung, row in violations[:5]:
            msg += f"  [{rung} held_out] {row['question']!r} -> {row['expected']} also in some rung's train\n"
        if len(violations) > 5:
            msg += f"  ... and {len(violations) - 5} more\n"
        msg += f"Total violations: {len(violations)}"
        raise AssertionError(msg)
