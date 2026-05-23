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
- Arc order: math-first, then language, then code. Specialists / MoE
  branch from robust base checkpoints, not from weak narrow experts.

Pointers:
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
