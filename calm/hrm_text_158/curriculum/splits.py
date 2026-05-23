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
    rungs: tuple[str, ...] = ("R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2", "R1b5", "R1b6", "R1b7", "R1b8", "R1b9", "R3", "R4", "R5", "R6"),
) -> dict[str, dict[str, list[dict]]]:
    """Build train + held_out splits for all rungs.

    Returns:
        {"R0": {"train": [...], "held_out": [...]}, "R1": {...}, ...}

    R7 (GSM8k) excluded — served from load_gsm8k_splits.

    Default tuple is the ACTIVE CHAIN per codex msgs 1779488238721-49f03cc9
    (R1b4v2 advance via seed=2 at b368b81) + 1779523412979-ff88b885 (R1b5
    K=4 carry-stratified added to chain) + 1779545956176-4a8cfc3e (R1b6
    K=5 carry-stratified added per gabe greenlight 1779545575582-7c52a912) +
    1779547753761-5711d790 (R1b7 K=6 carry-stratified added under durable
    gabe provenance relay 1779547541812) + 1779550489408-f40f66ab (R1b8
    K=7 carry-stratified added after R1b7 commit 682659b + A0 audit PASS) +
    1779554293017-3ba4b4ee (R1b9 K=8 carry-stratified added after R-C
    diagnostic PASS msg 1779554256972, parent = R1b3-repair candidate
    banked as new chain head with A0 1163/1164 strictly better than R1b8
    baseline 1161/1164) + 1779558351771-055c2265 (R1b10 K=9 PARKED as
    diagnosis-only after three failed promotion attempts from R1b9
    chain head; R1b10 supervision reliably destabilizes R1b2 K=-1
    subtraction; R1b10 stays reachable via explicit args for diagnosis
    but is OUT of default splits and exhaustive active rungs):

      R0 -> R1 -> R1b1 (K=1 plus) -> R1b2 (K=-1 minus) -> R1b3 (K=2 plus,
      PASSED at 175d327) -> R1b4v2 (K=3 plus, ADVANCED at b368b81 via
      seed=2 head) -> R1b5 (K=4 plus, carry-stratified, ADVANCED via
      seed=17) -> R1b6 (K=5 plus, carry-stratified, ADVANCED at 128b097
      via replay50_lr5e4) -> R1b7 (K=6 plus, carry-stratified, ADVANCED
      at 682659b via R1b2-retained chain) -> R1b8 (K=7 plus, ADVANCED
      at 1a14a09 via replay65_n10k_lr5e4) -> R1b9 (K=8 plus, banked as
      chain head via msg 1779556007032 from R1b3-repair parent with
      A0 1254/1255 strict + 1255/1255 parsed) -> [R1b10 K=9 PARKED] ->
      R3 -> ...

    Diagnosis-only and OUT of default: R1b2a (failed v1+v2 lowmult),
    R1b (legacy 3-template), R1b4 (K=3 v1 failed 7b53368 measurement
    bug — preserved immutable per codex 1779483673737), R2 (failed
    v1+v2 n_train=8000), R2a (failed v1 — variable B is the blocker).
    All reachable via explicit `rungs=` arg for diagnosis-only
    inspection. `assert_no_train_holdout_overlap` WILL fail on tuples
    containing overlapping pairs (R1b1+R1b, R1b2+R1b2a, R1b4+R1b4v2,
    R2a+R2, etc.).

    See `replay.py:DIAGNOSIS_ONLY_RUNGS` for the trainer-side
    auto-exclusion mechanism that complements this default.
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
