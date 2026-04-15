"""Rung 5 — phased incremental training demo.

Stacks sub-cards across a multi-stream unified tensor, training one at
a time with regression gates. Each phase compounds on the previous
checkpoint. If any prior capability regresses, the phase fails hard.

Phase ladder for this MVP:
  Phase 0: install compiled adder on `math` stream. Compile-only (no
           training). Gate: adder 16/16.
  Phase 1: train echo `8 + a` on `lm` stream head rows 8-15.
           Gate: echo >= 75%. Regression: adder 16/16.
  Phase 2: train echo `12 + b` on head rows 12-15 (reusing lm stream).
           Gate: echo-b >= 75%. Regression: adder 16/16, echo-a >= 75%.

Output: git-log-style before/after table per phase, regression scores
per prior gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.channel_masking import freeze_head_rows
from calm.llm_computer.multi_stream import (
    MultiStreamConfig, MultiStreamTransformer, StreamSpec,
    build_empty_multistream,
)
from calm.llm_computer.phase_runner import Phase, PhaseRunner, PhaseResult
from calm.llm_computer.unified_chrlm import (
    freeze_stream_embeddings, freeze_stream_layer,
    install_compiled_in_stream,
)
from scripts.experiment_fast_weights_fusion import build_adder_tiny_small2d


VOCAB = 20  # 0-7 adder, 8-11 echo-a, 12-15 echo-b, 16-19 spare
MAX_LEN = 4

_CFG = MultiStreamConfig(
    streams=(
        StreamSpec("math",   d_model=10, n_heads=5, d_ffn=14),
        StreamSpec("echo_a", d_model=16, n_heads=8, d_ffn=32),
        StreamSpec("echo_b", d_model=16, n_heads=8, d_ffn=32),
    ),
    n_layers=1, vocab_size=VOCAB, max_len=MAX_LEN, use_hard_max=True,
)


def _build_adder_src():
    return build_adder_tiny_small2d(target_layer=0, n_layers=1)


def _build_model():
    return build_empty_multistream(_CFG)


# ----- Gates -----

@torch.no_grad()
def gate_adder(model) -> float:
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1, :7]
            if int(logits.argmax().item()) == a + b:
                ok += 1
    return ok / 16.0


@torch.no_grad()
def _echo_gate(model, offset: int, target_fn) -> float:
    """Generic echo eval: logits[offset:offset+4] should pick target_fn(a,b)."""
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1, offset:offset + 4]
            if int(logits.argmax().item()) == target_fn(a, b):
                ok += 1
    return ok / 16.0


def gate_echo_a(model) -> float:
    return _echo_gate(model, offset=8, target_fn=lambda a, b: a)


def gate_echo_b(model) -> float:
    return _echo_gate(model, offset=12, target_fn=lambda a, b: b)


# ----- Phase setups -----

def _remove_prior_head_hooks(model):
    """Clear any head-row gradient hooks left over from previous phases.

    Hooks compound multiplicatively in PyTorch; stale hooks from prior
    phases zero out gradients on rows we want to train in the current
    phase. This clears the internal hook dict before each phase's
    freeze_head_rows call installs fresh ones.
    """
    w = model.head.weight
    # PyTorch tensors store hooks in an OrderedDict on the tensor
    if hasattr(w, "_backward_hooks") and w._backward_hooks is not None:
        w._backward_hooks.clear()


def setup_phase_0(model: MultiStreamTransformer, prior: list[PhaseResult]) -> None:
    """Install compiled adder on math stream, freeze it."""
    install_compiled_in_stream(
        model, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    freeze_stream_layer(model, "math", layer_idx=0)
    freeze_stream_embeddings(model, "math")


def setup_phase_1(model: MultiStreamTransformer, prior: list[PhaseResult]) -> None:
    """Train echo 8+a on echo_a stream + head rows 8-11.

    Keep math frozen. Freeze echo_b stream entirely (it's reserved for
    phase 2). Train echo_a stream + head rows 8-11.
    """
    freeze_stream_layer(model, "math", layer_idx=0)
    freeze_stream_embeddings(model, "math")
    # Freeze echo_b stream — phase 2's territory
    freeze_stream_layer(model, "echo_b", layer_idx=0)
    freeze_stream_embeddings(model, "echo_b")
    # Random init on echo_a stream (it was zero)
    with torch.no_grad():
        for p in model.streams["echo_a"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[8:12] == 0).all():
            model.head.weight[8:12].normal_(0, 0.02)
    # Head: only rows 8-11 trainable (clear stale hooks first)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 8)) + list(range(12, VOCAB)))


def setup_phase_2(model: MultiStreamTransformer, prior: list[PhaseResult]) -> None:
    """Train echo 12+b on echo_b stream + head rows 12-15.

    Phase 2 has its OWN stream body (echo_b) — physical isolation from
    phase 1's echo_a stream. Both coexist in the same tensor with zero
    interference. This is the L2 multi-stream payoff for phased
    training: each phase gets its own compute without touching priors.
    """
    # Freeze everything from prior phases
    freeze_stream_layer(model, "math", layer_idx=0)
    freeze_stream_embeddings(model, "math")
    freeze_stream_layer(model, "echo_a", layer_idx=0)
    freeze_stream_embeddings(model, "echo_a")
    # Init echo_b stream if zero
    with torch.no_grad():
        for p in model.streams["echo_b"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[12:16] == 0).all():
            model.head.weight[12:16].normal_(0, 0.02)
    # Head: only rows 12-15 trainable (clear stale hooks first)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 12)) + list(range(16, VOCAB)))


# ----- Training functions -----

def _train_echo(model, steps, lr, batch_size, seed,
                offset, target_fn) -> tuple[float, float]:
    import time
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = torch.Generator().manual_seed(seed)
    xs = torch.tensor([[a, b] for a in range(4) for b in range(4)],
                     dtype=torch.long)
    ys = torch.tensor([target_fn(a, b) for a in range(4) for b in range(4)],
                     dtype=torch.long)
    model.train()
    t0 = time.time()
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, 16, (batch_size,), generator=rng)
        logits = model(xs[idx])[:, 1, offset:offset + 4]
        loss = F.cross_entropy(logits, ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    model.eval()
    return last, time.time() - t0


def train_echo_a(model, steps, lr, batch_size, seed):
    return _train_echo(model, steps, lr, batch_size, seed,
                       offset=8, target_fn=lambda a, b: a)


def train_echo_b(model, steps, lr, batch_size, seed):
    return _train_echo(model, steps, lr, batch_size, seed,
                       offset=12, target_fn=lambda a, b: b)


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default="/tmp/minibench_phased")
    args = ap.parse_args()

    runner = PhaseRunner(
        build_model_fn=_build_model,
        checkpoint_dir=Path(args.ckpt_dir),
    )

    phases = [
        Phase(
            phase_id=0, name="compiled_adder",
            setup_fn=setup_phase_0,
            train_fn=None, steps=0,
            gate_fn=gate_adder, min_accuracy=1.0,
        ),
        Phase(
            phase_id=1, name="echo_a",
            setup_fn=setup_phase_1,
            train_fn=train_echo_a, steps=args.steps,
            lr=args.lr,
            gate_fn=gate_echo_a, min_accuracy=0.75,
        ),
        Phase(
            phase_id=2, name="echo_b",
            setup_fn=setup_phase_2,
            train_fn=train_echo_b, steps=args.steps,
            lr=args.lr,
            gate_fn=gate_echo_b, min_accuracy=0.75,
        ),
    ]

    print("=== minibench_phased — sub-card stacking via PhaseRunner ===",
          flush=True)
    print(f"  streams: {[f'{s.name}(d_model={s.d_model})' for s in _CFG.streams]}",
          flush=True)
    print(f"  vocab layout: 0-7 adder | 8-11 echo-a | 12-15 echo-b | 16-19 spare",
          flush=True)
    print(f"  checkpoints: {args.ckpt_dir}/phase_NN.pt", flush=True)
    print("", flush=True)

    for phase in phases:
        print(f"--- phase {phase.phase_id}: {phase.name} "
              f"({phase.steps} steps) ---", flush=True)
        result = runner.run_phase(phase)
        print(f"  gate:  {result.gate_score*100:.1f}% "
              f"(threshold {result.min_threshold*100:.0f}%) "
              f"{'PASS' if result.passed else 'FAIL'}", flush=True)
        if result.regression_scores:
            for prior_name, score in result.regression_scores.items():
                print(f"  regression [{prior_name}]: {score*100:.1f}%",
                      flush=True)
        if result.final_loss > 0:
            print(f"  final loss: {result.final_loss:.4f}", flush=True)
        if result.train_wallclock_s > 0:
            print(f"  wallclock: {result.train_wallclock_s:.2f}s", flush=True)
        if result.note:
            print(f"  note: {result.note}", flush=True)
        print("", flush=True)

        if not result.passed:
            print(f"=== STOP: phase {phase.phase_id} failed ===", flush=True)
            return

    print("=== summary ===", flush=True)
    print(f"  phases passed: {len(runner.history)}/{len(phases)}", flush=True)
    for r in runner.history:
        print(f"    phase {r.phase_id} ({r.name}): "
              f"gate={r.gate_score*100:.0f}% loss={r.final_loss:.3f} "
              f"ckpt saved", flush=True)
    print(f"\n  OVERALL: "
          f"{'PASS — all sub-cards compound in one tensor' if len(runner.history) == len(phases) else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
