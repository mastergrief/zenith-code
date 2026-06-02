# Architecture Discipline - Facades Before God Files

> Historical receipts: see `MEMORY/atlas/architecture_discipline_arc.md`.

## Core Rule

Research velocity is not permission to grow unbounded scripts. Logic that
survives beyond one throwaway run moves behind an importable facade, pure
reducer, or explicit contract before the next mechanism branch depends on it.

Task-local harnesses are allowed to move fast only while they stay thin:
CLI/path setup, launch wiring, artifact locations, and receipt emission.
Reusable behavior belongs in repo-tracked modules with tests.

## Stop Conditions

Any one of these turns the next mutating gate into a refactor gate first:

- A file exceeds 500 lines while mixing CLI, IO, training loop, telemetry,
  validation, or artifact contracts.
- A task-local copy needs a second semantic patch or becomes a second
  experiment's launch target.
- One file owns three or more state classes, such as authoritative train
  state, prior evidence, current-run telemetry, validation, or launcher state.
- A safe change requires understanding unrelated sections of the same file.

When tripped, post a seam map and a behavior-preserving extraction plan before
broad edits continue.

## Facade And Reimport Discipline

Canonical reusable logic lives in repo-tracked importable modules. Task-local
harnesses import those modules and remain thin orchestrators.

Promote on second use: when logic is reused by another experiment, run, or
harness, extract it before extending it again.

Dynamic repo imports are allowed only through one named import facade. The
facade records repo root, source path, expected hash or contract, and the
reason the dynamic import exists. Do not scatter dynamic imports through
training logic.

Helper reducers never import launch, loop, GPU, or filesystem glue. Dependency
direction is harness -> facades/reducers, never the reverse.

## Required Seams

Name an owner for each seam before broad edits:

- CLI, config, launch, and runtime resource setup.
- Artifact, manifest, hash, and resume contracts.
- Authoritative train-state, resume, rollback, and mutation ownership.
- Pure gate and status reducers: floors, coverage merge, rate pressure, trust,
  and terminal classification.
- Candidate generation and update policy.
- Telemetry/event schemas and consumer validation.

A seam without an owner blocks broad edits.

## Refactor Validation

Characterization tests come first. Capture current behavior before extraction,
especially for previously implicit state ownership.

Pure reducers use CPU-static/no-loop tests. Loop entry, checkpoint-load-and-step,
q/acc update, probes, and GPU-equivalence mechanics follow `workflow.md` section
`Full-GPU for trainer-loop work; CPU only for non-loop checks`.

Freeze a baseline packet before behavior-preserving refactors: source hash or
contract, exact command, fixture/probe set, expected artifact schemas, and
comparison rules. If equivalence fails, downstream science falls back to the
frozen baseline rather than rationalizing the refactor.

## Maintainable End State

Durable repo modules own reusable contracts, reducers, and facades. The harness
is a thin orchestrator.

One centralized import facade owns path, hash, and contract checks for dynamic
repo loading.

The q/acc mutation loop stays isolated and equivalence-protected until
characterization is strong enough to extract it safely.
