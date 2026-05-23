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
    6. No future rungs by default (index >= current rejected);
       `allow_future_replay=True` is the explicit repair override
       that gates rule 6 only — all other validations still apply.
       WARN names each future rung accepted. Per codex msg
       1779548482300-05680b9d Option G after R1b6 commit 128b097
       baseline revealed R1b2=0.78 foundational-primitive gap.
  Explicit override allowed to include diagnosis-only rungs (with
  WARN); positional default auto-excludes diagnosis-only AND R7
  AND is unaffected by `allow_future_replay`.

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
# Status is RUNG-WISE, not architecture-wide — a rung listed here can
# be removed later if a future retry passes (e.g., R2 can drop out if
# a future full-R2 target acquires).
#
# Current set:
#   - R1b2a: failed both v1 (memorization) and v2 lowmult (confounded)
#   - R1b: legacy 3-template, superseded by R1b1 + R1b2 single-template
#   - R2: full teens ± failed v1 (n_train=6000) AND v2 (n_train=8000)
#         (codex msg 1779478819906-0e30503e). R2a was the addition-only
#         operator-split successor but ALSO failed v1 (0.045 at 558fcc1)
#         due to variable-B blocker (NOT operator mixing alone).
#   - R2a: addition-only teens variable-B failed v1 at 0.045 (worse
#          than R2). Per codex msg 1779479973262-6d7445d2: variable-B
#          itself is the structural blocker; the locked-piece pattern
#          is "constant-B single-template" (R1b1/R1b2 PASS). R1b3
#          (constant K=2 addition) is the next active rung.
#   - R1b4: constant K=3 addition v1 FAILED at 7b53368 (standard
#           0.885 < G1 0.90 by 0.015). Per codex msg 1779483673737-
#           20ff22ab the failure was a measurement bug (2-row
#           one_digit heldout sampled ~22× via rng.choice repeated
#           weighting), not architectural. R1b4 is preserved
#           immutable in RUNG_NAMES as failed-diagnostic provenance;
#           R1b4v2 (one_digit-exhaustive partition) is the active-
#           chain successor.
#
# R1b2 stays OUT (canonical retry that PASSED at c2686cc).
# R1b3 stays OUT (PASSED at 175d327 via v2 schedule).
# R1b4v2 stays OUT (ADVANCED at b368b81 via seed=2 head; canonical
#         R1b4v2 chain head per codex msg 1779488238721-49f03cc9).
# R1b5 stays OUT (active-chain target K=4 added per codex msg
#         1779523412979-ff88b885; ADVANCED via seed=17 head).
# R1b6 stays OUT (active-chain target K=5 added per codex msg
#         1779545956176-4a8cfc3e after gabe greenlight relay
#         1779545575582-7c52a912; ADVANCED via replay50_lr5e4 at commit
#         128b097 with R1b6=1.00 + all priors 1.00 + all keyed audits 9/9).
# R1b7 stays OUT (active-chain target K=6 added per codex msg
#         1779547753761-5711d790 under durable gabe provenance relay
#         1779547541812 of verbatim "you have full provenance until im
#         back"; current attempt). Positional priors when training R1b7
#         resolve to {R0, R1, R1b1, R1b2, R1b3, R1b4v2, R1b5, R1b6}.
DIAGNOSIS_ONLY_RUNGS: frozenset[str] = frozenset({"R1b2a", "R1b", "R1b4", "R2", "R2a"})


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
    allow_future_replay: bool = False,
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
        allow_future_replay: When True, skips the future-rung rejection
            in the EXPLICIT `--replay-rungs` path so a foundational-rung
            repair pass can include later-rung replay (e.g. repair R1b2
            with R1b3..R1b6 in replay to preserve mastery while lifting
            R1b2). Emits a WARN naming each future rung accepted. DOES
            NOT affect the positional default path. All other rejects
            (unknown, R7, self, duplicate, empty, malformed) still
            apply. Codex msg 1779548482300-05680b9d Option G after
            R1b6 commit 128b097 baseline revealed R1b2=0.78 pre-existing
            gap; durable gabe provenance relay 1779547541812.

    Returns:
        List of rung names (deterministic order) to draw replay from.

    Raises:
        ValueError on validation failure (empty list, unknown rung, self,
        duplicate, R7, or future rung when `allow_future_replay=False`).
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
            if r_idx >= cur_idx and not allow_future_replay:
                raise ValueError(
                    f"--replay-rungs entry {r!r} (index {r_idx}) is at or "
                    f"after current rung {curriculum_rung!r} (index {cur_idx}); "
                    f"future rungs cannot be replay priors. Pass "
                    f"--allow-future-replay to opt into future-rung replay "
                    f"for foundational-rung repair passes."
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

        # Future-rung WARN: when allow_future_replay=True and an entry is
        # >= cur_idx, name the rung explicitly so receipts make the
        # override visible (codex msg 1779548482300-05680b9d guardrail).
        if allow_future_replay:
            for r in explicit_list:
                r_idx = list(rung_names).index(r)
                if r_idx >= cur_idx:
                    warn(
                        f"--replay-rungs includes FUTURE rung {r!r} "
                        f"(index {r_idx} >= current rung {curriculum_rung!r} "
                        f"index {cur_idx}); --allow-future-replay override "
                        f"accepted. Use only for foundational-rung repair."
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
