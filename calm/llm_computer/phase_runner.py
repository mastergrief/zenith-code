"""Phase runner — incremental sub-card training on one tensor.

The unified CHRLM architecture lets phases compound: phase N loads
phase N-1's .pt, freezes what N-1 trained, opens new trainable slots,
trains, regression-tests ALL prior gates, commits.

This module generalizes the rung-4 pattern into a reusable runner.
Each `Phase` declares:
  - what to freeze (so prior capabilities stay intact)
  - what to open for training (new trainable slot)
  - how to train (data + loss)
  - how to gate (pass threshold on held-out eval)

The runner:
  - Loads previous phase's checkpoint (or builds fresh at phase 0)
  - Applies all prior freezes + new phase's freezes
  - Runs training for `steps`
  - Tests the new phase's gate
  - Regression-tests every prior phase's gate (hard stop on any failure)
  - Saves checkpoint for phase N+1 to load

If any gate fails, the phase is aborted with a clear diagnostic and
the previous checkpoint is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn


@dataclass
class PhaseResult:
    phase_id: int
    name: str
    passed: bool
    gate_score: float
    min_threshold: float
    train_wallclock_s: float
    regression_scores: dict[str, float] = field(default_factory=dict)
    final_loss: float = 0.0
    note: str = ""


@dataclass
class Phase:
    """One phase of incremental training.

    Attributes:
        phase_id: ordinal (0, 1, 2, ...) — passes are expected in order.
        name: human-readable label (e.g. "compiled_adder", "echo_8plus_a").
        setup_fn: callable(model, prior_results) that prepares the model
            for this phase — installing compiled programs, setting up
            freezes, initializing new trainable params. Runs BEFORE
            training. Returns nothing.
        train_fn: callable(model, steps, lr, batch_size, seed) → (final_loss,
            wallclock_s). Runs the training loop for this phase. Only
            called if steps > 0 (compiled-only phases can set steps=0).
        gate_fn: callable(model) → float accuracy in [0, 1]. Evaluates
            THIS phase's capability.
        min_accuracy: pass threshold for the gate (e.g. 0.75).
        steps: number of training steps. 0 for compile-only phases.
        lr: learning rate.
        batch_size: minibatch size for train_fn.
        seed: RNG seed.
    """
    phase_id: int
    name: str
    setup_fn: Callable[[nn.Module, list[PhaseResult]], None]
    train_fn: Optional[Callable[..., tuple[float, float]]]
    gate_fn: Callable[[nn.Module], float]
    min_accuracy: float
    steps: int = 0
    lr: float = 1e-3
    batch_size: int = 8
    seed: int = 42


class PhaseRunner:
    """Sequences Phase instances on a single nn.Module, checkpointing
    between phases and hard-stopping on any regression.

    Usage:
        runner = PhaseRunner(
            build_model_fn=lambda: MultiStreamTransformer(cfg),
            checkpoint_dir=Path("/tmp/phases"),
        )
        result_0 = runner.run_phase(phase_0_compile_adder)
        result_1 = runner.run_phase(phase_1_echo_task)
        # After each phase: result.passed is True iff gate + all prior
        # regressions pass.
    """

    def __init__(
        self,
        build_model_fn: Callable[[], nn.Module],
        checkpoint_dir: Path,
    ):
        self._build_model_fn = build_model_fn
        self._ckpt_dir = Path(checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[PhaseResult] = []
        # gate_fn from each prior phase, keyed by phase name for
        # regression testing
        self._prior_gates: dict[str, Callable[[nn.Module], float]] = {}
        self._prior_thresholds: dict[str, float] = {}
        # Currently-loaded model (persists across phases within one
        # runner session; reload from ckpt if Python interpreter restarts)
        self._model: Optional[nn.Module] = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._build_model_fn()
        return self._model

    @property
    def history(self) -> list[PhaseResult]:
        return list(self._history)

    def _checkpoint_path(self, phase_id: int) -> Path:
        return self._ckpt_dir / f"phase_{phase_id:02d}.pt"

    def run_phase(self, phase: Phase) -> PhaseResult:
        """Execute one phase. Returns PhaseResult (passed=False if any
        gate fails; caller can inspect and decide what to do)."""
        import time

        # Phase ordinal check
        expected_id = len(self._history)
        if phase.phase_id != expected_id:
            raise ValueError(
                f"expected phase_id={expected_id}, got {phase.phase_id}"
            )

        model = self.model
        # Load previous checkpoint if present (and we're past phase 0)
        if phase.phase_id > 0:
            prev_ckpt = self._checkpoint_path(phase.phase_id - 1)
            if prev_ckpt.exists():
                model.load_state_dict(torch.load(prev_ckpt, weights_only=True))

        # Setup: freezes + installs + init. Phase-specific.
        phase.setup_fn(model, self._history)

        # Train (if applicable)
        final_loss = 0.0
        train_wallclock = 0.0
        if phase.steps > 0 and phase.train_fn is not None:
            t0 = time.time()
            final_loss, _ = phase.train_fn(
                model, phase.steps, phase.lr, phase.batch_size, phase.seed,
            )
            train_wallclock = time.time() - t0

        # Current phase's gate
        model.eval()
        gate_score = phase.gate_fn(model)
        current_pass = gate_score >= phase.min_accuracy

        # Regression test all prior phases
        regression_scores: dict[str, float] = {}
        regressions_pass = True
        for prior_name, gate_fn in self._prior_gates.items():
            score = gate_fn(model)
            regression_scores[prior_name] = score
            if score < self._prior_thresholds[prior_name]:
                regressions_pass = False

        overall_pass = current_pass and regressions_pass
        note = ""
        if not current_pass:
            note = (
                f"current gate {gate_score:.2%} < threshold "
                f"{phase.min_accuracy:.2%}"
            )
        elif not regressions_pass:
            failed = [
                (n, s, self._prior_thresholds[n])
                for n, s in regression_scores.items()
                if s < self._prior_thresholds[n]
            ]
            note = f"regression(s): {failed}"

        result = PhaseResult(
            phase_id=phase.phase_id,
            name=phase.name,
            passed=overall_pass,
            gate_score=gate_score,
            min_threshold=phase.min_accuracy,
            train_wallclock_s=train_wallclock,
            regression_scores=regression_scores,
            final_loss=final_loss,
            note=note,
        )

        # Save checkpoint only if passed
        if overall_pass:
            torch.save(model.state_dict(), self._checkpoint_path(phase.phase_id))
            self._history.append(result)
            self._prior_gates[phase.name] = phase.gate_fn
            self._prior_thresholds[phase.name] = phase.min_accuracy

        return result
