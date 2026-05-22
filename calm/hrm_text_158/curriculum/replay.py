"""Curriculum replay-prior resolution + DIAGNOSIS_ONLY_RUNGS handling.

Per codex msg 1779475162259-625114e1 + 1779475236781-c29da5e5 +
1779475454122-1512da3b structural-fix arc after R1b2a v2 lowmult
confounded fail (ddcc943):

Previous trainer derived priors positionally as `RUNG_NAMES[:cur_idx]`.
After R1b2/R1b/R1b2a were demoted to diagnosis-only in the active
chain, positional derivation would silently include failed rungs in
replay (e.g., R2 launch would replay R1b2a/R1b2/R1b). Bad.

This module exposes:

- `DIAGNOSIS_ONLY_RUNGS` — frozenset of rungs auto-excluded from
  positional prior derivation. R1b2 is INTENTIONALLY NOT included
  per codex msg 1779475454122 because R1b2 is the canonical retry
  target (R1b2_v2_replay50); excluding it would silently omit a
  successful retry from future R2 positional replay. Add R1b2 only
  if/when the retry also fails.

- `_resolve_prior_rungs(curriculum_rung, replay_rungs_arg, ...)` —
  pure helper that returns the resolved priors list. Used by
  `train_hrm_text_158.py` to derive replay priors for the curriculum
  branch. Validates:
    1. Override list non-empty (if provided)
    2. Each rung in RUNG_NAMES (unknown rejected)
    3. Current rung not in list (self rejected)
    4. No duplicates (overweighting prevented)
    5. R7 not in list (GSM8k served separately, not synthesized)
    6. No future rungs (index >= current rejected)
  Explicit override allowed to include diagnosis-only rungs (with
  WARN); positional default auto-excludes diagnosis-only AND R7.

Tests: `calm/llm_computer/tests/test_hrm_text_158_phase_a_wiring.py`
"""
from __future__ import annotations

from calm.hrm_text_158.curriculum.generators import RUNG_NAMES


# Diagnosis-only rungs are kept in RUNG_NAMES (and in generators for
# explicit access) but AUTOMATICALLY EXCLUDED from positional
# `--replay-rungs` derivation.
#
# Append a rung here ONLY when its retry/successor has shipped AND
# the failed variant should never participate in default replay.
# R1b2 is INTENTIONALLY NOT here per codex msg 1779475454122-1512da3b:
# R1b2 is the canonical retry target via R1b2_v2_replay50; if that
# retry passes, R1b2 should remain available for positional R2 replay.
DIAGNOSIS_ONLY_RUNGS: frozenset[str] = frozenset({"R1b2a", "R1b"})


# R7 (GSM8k) is generator-incompatible — served separately from
# load_gsm8k_splits in the trainer. Cannot appear in synthetic-rung
# replay regardless of position.
_NON_SYNTHETIC_RUNGS: frozenset[str] = frozenset({"R7"})


def _resolve_prior_rungs(
    curriculum_rung: str,
    replay_rungs_arg: str | None,
    *,
    rung_names: tuple[str, ...] = RUNG_NAMES,
    diagnosis_only: frozenset[str] = DIAGNOSIS_ONLY_RUNGS,
    warn_callback=None,
) -> list[str]:
    """Resolve prior_rungs for trainer's curriculum-replay mix.

    Args:
        curriculum_rung: The active rung being trained (must be in rung_names).
        replay_rungs_arg: Comma-separated explicit override (or None for
            positional derivation).
        rung_names: Tuple of all valid rung names (default RUNG_NAMES).
        diagnosis_only: Frozenset of rungs auto-excluded from positional
            derivation (default DIAGNOSIS_ONLY_RUNGS).
        warn_callback: Optional callable(msg) for diagnosis-only override
            WARN. Defaults to print() if None.

    Returns:
        List of rung names (deterministic order) to draw replay from.

    Raises:
        ValueError on validation failure (empty list, unknown rung, self,
        duplicate, R7, future rung).
    """
    if curriculum_rung not in rung_names:
        raise ValueError(
            f"curriculum_rung {curriculum_rung!r} not in rung_names; "
            f"valid: {rung_names}"
        )
    cur_idx = list(rung_names).index(curriculum_rung)

    if replay_rungs_arg is not None:
        # Explicit override path
        # Permit "R0, R1" with whitespace; reject "" or "R0,,R1" empty entries
        raw_entries = [r.strip() for r in replay_rungs_arg.split(",")]
        explicit_list = [r for r in raw_entries if r]
        # If raw_entries has empty strings but explicit_list is empty,
        # treat as empty input
        if not explicit_list:
            raise ValueError(
                "--replay-rungs cannot be empty; pass at least one rung name "
                "OR omit the flag to use positional derivation."
            )

        # Reject empty entries mid-list (e.g., "R0,,R1") — they indicate
        # malformed CLI argument
        if len(raw_entries) != len(explicit_list):
            raise ValueError(
                f"--replay-rungs contains empty entry (malformed comma list): "
                f"{replay_rungs_arg!r}"
            )

        # Reject duplicates (overweighting silently otherwise)
        seen: set[str] = set()
        dupes: list[str] = []
        for r in explicit_list:
            if r in seen:
                dupes.append(r)
            seen.add(r)
        if dupes:
            raise ValueError(
                f"--replay-rungs contains duplicate rungs (would silently "
                f"overweight): {dupes} in {explicit_list}"
            )

        # Reject unknown / R7 / self / future
        for r in explicit_list:
            if r not in rung_names:
                raise ValueError(
                    f"--replay-rungs entry {r!r} not in rung_names; "
                    f"valid: {rung_names}"
                )
            if r in _NON_SYNTHETIC_RUNGS:
                raise ValueError(
                    f"--replay-rungs cannot include {r!r} "
                    f"(generator-incompatible; served separately)."
                )
            if r == curriculum_rung:
                raise ValueError(
                    f"--replay-rungs cannot include current rung "
                    f"{curriculum_rung!r}; got {explicit_list}"
                )
            r_idx = list(rung_names).index(r)
            if r_idx >= cur_idx:
                raise ValueError(
                    f"--replay-rungs entry {r!r} (index {r_idx}) is at or "
                    f"after current rung {curriculum_rung!r} (index {cur_idx}); "
                    f"future rungs cannot be replay priors."
                )

        # Explicit override allowed to include diagnosis-only with WARN
        warn = warn_callback if warn_callback is not None else _default_warn
        for r in explicit_list:
            if r in diagnosis_only:
                warn(
                    f"--replay-rungs includes diagnosis-only rung {r!r}; "
                    f"operator override accepted but unusual (see "
                    f"DIAGNOSIS_ONLY_RUNGS in replay.py)."
                )
        return explicit_list

    # Positional default path: RUNG_NAMES[:cur_idx] minus diagnosis-only
    # minus non-synthetic (R7 etc.)
    positional = [
        r for r in rung_names[:cur_idx]
        if r not in diagnosis_only and r not in _NON_SYNTHETIC_RUNGS
    ]
    return positional


def _default_warn(msg: str) -> None:
    """Default warn callback prints to stdout with [hrm158] prefix."""
    print(f"[hrm158] WARN: {msg}", flush=True)
