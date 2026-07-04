# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HRM-Text-1.58 Fork: Progressive Checkpoint Curriculum

- Active fork target: **`hrm-158-base`**, a robust all-rounder native
  HRM-Text-1.58 checkpoint.
- Loss is response-only: prompt/instruction tokens are masked.
- Curriculum loop: start from the latest banked checkpoint, train one
  auditable finite-support capability slice (full-density when small enough
  to audit completely; bounded fallback otherwise) with replay over important
  prior rungs, promote only after sampled probes + A0 exhaustive
  finite-support audit + watch rows clear under the named gate semantics.
- **Bank gate** (gabe-locked): acquire ≥90% / retain ≥90% per slice; bank the
  earliest save that clears (final has no privilege); on a miss
  **classify + split smaller** — don't stretch the run, bump LR, or
  add layers.
- **Default slice** (gabe-locked, one atom): **auditable full-density finite
  support trained slow-safe** under the 90/90 bank gate; numeric recipe band →
  `.claude/rules/hrm-158.md` §"Recipe band" (no knob escalation on a miss).
  Full coverage of a small completely-auditable support DRIVES acquisition
  (banked identity 90/90) where sparse sub-sampling regressed to
  nearest-memorized retrieval. **Bounded stair-step is the FALLBACK** after a
  classified collision / oversized support; don't continue+re-warm a fragile
  dense surface.
- **Retention (load-bearing)**: explicit replay + parent consistency +
  broad retained supports (L0b, math_a0) + **direct close-sibling protection**
  when the target shares a template/emission surface (L0c1 lesson); anchors are
  sentinels, not the primary mechanism.
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
