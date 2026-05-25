# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HRM-Text-1.58 Fork: Progressive Checkpoint Curriculum

- Active fork target: **`hrm-158-base`**, a robust all-rounder native
  HRM-Text-1.58 checkpoint.
- Loss is response-only: prompt/instruction tokens are masked.
- Curriculum loop: start from the latest validated checkpoint, train
  one tiny capability block (rung) with replay over important prior
  rungs, promote only after sampled probes + A0 exhaustive
  finite-support audit + watch rows clear with no parent-relative
  cluster regression.
- **Bank gate**: acquire ≥90% / retain ≥95% per slice; bank the
  earliest save that clears (final has no privilege); on a miss
  **classify + split smaller** — don't stretch the run, bump LR, or
  add layers.
- **Bounded slices**: tight finite supports (~230) + ≤1500-step
  windows; full-density / exhaustive surfaces are progress metrics,
  not bank gates unless explicitly gated. **Stair-step into density** —
  bounded wrappers acquire cleanly (L0a/L0b), but swallowing or
  continuing+re-warming a full-density surface showed broad / rewarm
  regressions.
- **Retention (load-bearing)**: explicit replay + parent consistency +
  broad retained supports (L0b, math_a0); anchors are sentinels, not
  the primary mechanism.
- L0c lesson: the `<expr> equals what?` one-digit stratum transfers
  cleanly; failures concentrate in the two-digit / template-specific
  stratum — the same operands succeed under other wrappers, so it is a
  surface/template gap, NOT math capacity.
- Arc order: math-first, then language, then code. Specialists / MoE
  branch from robust base checkpoints, not from weak narrow experts.

Pointers:
- **Canonical active workflow** (bank gate, slice-size, recipe band,
  retention, failure classes, validation): `.claude/rules/hrm-158.md`.
- Repo manifest (multi-agent harness, CALM, AI-Room collab,
  vocabulary lock-in): `.claude/CLAUDE.md`.
- Curriculum rules (recipe, validation, failure-mode classification):
  `.claude/rules/training.md` §"HRM-Text-1.58 Fork".

## Detected stack
- Languages: Rust + Python.
- Frameworks: PyTorch (HRM-Text-1.58, CALM, llm_computer); none for the Rust workspace.

## Verification
- Run Rust verification from `rust/`: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`
- HRM-Text-1.58 fork tests: `PYTHONPATH=. python3 -m pytest calm/llm_computer/tests/test_hrm_text_158_curriculum.py calm/llm_computer/tests/test_exhaustive_supports.py -q`
- `src/` and `tests/` are both present; update both surfaces together when behavior changes.

## Repository shape
- `rust/` contains the Rust workspace and upstream claw-code Rust port (separate build).
- `calm/` contains the active CALM engine + LLM-Computer substrate + HRM-Text-1.58 native training/probe stack.
- `agents/` contains the Python multi-agent harness (terminal coding assistant).
- `src/` contains source files that should stay consistent with generated guidance and tests.
- `tests/` contains validation surfaces that should be reviewed alongside code changes.

## Working agreement
- Prefer small, reviewable changes; one round per commit with a before/after measurement table.
- `.pt` checkpoint artifacts are runtime/research outputs — commit code/tooling/docs/manifest receipts, NOT `.pt` by default.
- Keep shared defaults in `.claude.json`; reserve `.claude/settings.local.json` for machine-local overrides.
- Do not overwrite existing `CLAUDE.md` content automatically; update it intentionally when repo workflows change.
