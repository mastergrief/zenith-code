"""Rung 5 at 5-phase scale — real stack, real measurements.

Previously shipped: minibench_phased demonstrates 3-phase stacking (adder
→ echo_a → echo_b) at 16-example scale. Cute but proves nothing about
scaling.

This script pushes to 5 phases on a broader task set. The purpose is
NOT to show success — we expect regressions — but to EXPOSE where the
phased architecture actually breaks. Per the workflow: "run it to
failure, read the failure mode, fix the one wrong thing."

Phase ladder:
  Phase 0: compile adder onto math stream
  Phase 1: train echo_a (8+a) on echo_a stream
  Phase 2: train echo_b (12+b) on echo_b stream
  Phase 3: train sum_out (16+a+b) on sum_out stream — requires reading
           channels written by phase 0 (adder's step funcs 3-9 in math stream)
  Phase 4: train parity (18+((a+b) mod 2)) on parity stream — requires
           reading the adder output AND applying a boolean transform

Each phase must (a) pass its own gate AND (b) preserve all prior gates.

Running this answers: does the phased architecture scale past trivial
orthogonal tasks? Do compositional phases (3, 4) actually learn by
reading prior streams, or do they regress prior capabilities?
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


VOCAB = 32
# 0-7: adder vocab     (phase 0)
# 8-11: echo_a         (phase 1)
# 12-15: echo_b        (phase 2)
# 16-22: sum_out       (phase 3) — sum can be 0..6, offset 16 → 16..22
# 23-24: parity        (phase 4) — 0 or 1, offset 23
MAX_LEN = 4

_CFG = MultiStreamConfig(
    streams=(
        StreamSpec("math",    d_model=10, n_heads=5, d_ffn=14),
        StreamSpec("echo_a",  d_model=16, n_heads=8, d_ffn=32),
        StreamSpec("echo_b",  d_model=16, n_heads=8, d_ffn=32),
        StreamSpec("sum_out", d_model=16, n_heads=8, d_ffn=32),
        StreamSpec("parity",  d_model=16, n_heads=8, d_ffn=32),
    ),
    n_layers=1, vocab_size=VOCAB, max_len=MAX_LEN, use_hard_max=True,
)


def _build_adder_src():
    return build_adder_tiny_small2d(target_layer=0, n_layers=1)


def _build_model():
    return build_empty_multistream(_CFG)


def _remove_prior_head_hooks(model):
    w = model.head.weight
    if hasattr(w, "_backward_hooks") and w._backward_hooks is not None:
        w._backward_hooks.clear()


# ----- Gates (each returns accuracy 0..1) -----

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


def _gate_echo(model, offset, target_fn):
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1, offset:offset + 4]
            if int(logits.argmax().item()) == target_fn(a, b):
                ok += 1
    return ok / 16.0


@torch.no_grad()
def gate_echo_a(model) -> float:
    return _gate_echo(model, offset=8, target_fn=lambda a, b: a)


@torch.no_grad()
def gate_echo_b(model) -> float:
    return _gate_echo(model, offset=12, target_fn=lambda a, b: b)


@torch.no_grad()
def gate_sum_out(model) -> float:
    """At pos 1, vocab rows 16..22 should pick `a+b` (0..6)."""
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1, 16:23]
            if int(logits.argmax().item()) == a + b:
                ok += 1
    return ok / 16.0


@torch.no_grad()
def gate_parity(model) -> float:
    """At pos 1, vocab rows 23..24 should pick (a+b) mod 2."""
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = model(x)[0, 1, 23:25]
            if int(logits.argmax().item()) == (a + b) % 2:
                ok += 1
    return ok / 16.0


# ----- Setup functions -----

def _freeze_all_prior_streams_through(model, frozen_streams):
    for s in frozen_streams:
        freeze_stream_layer(model, s, layer_idx=0)
        freeze_stream_embeddings(model, s)


def setup_phase_0(model, prior):
    install_compiled_in_stream(
        model, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    _freeze_all_prior_streams_through(model, ["math"])


def setup_phase_1(model, prior):
    _freeze_all_prior_streams_through(model, ["math", "echo_b", "sum_out", "parity"])
    with torch.no_grad():
        for p in model.streams["echo_a"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[8:12] == 0).all():
            model.head.weight[8:12].normal_(0, 0.02)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 8)) + list(range(12, VOCAB)))


def setup_phase_2(model, prior):
    _freeze_all_prior_streams_through(model,
                                       ["math", "echo_a", "sum_out", "parity"])
    with torch.no_grad():
        for p in model.streams["echo_b"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[12:16] == 0).all():
            model.head.weight[12:16].normal_(0, 0.02)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 12)) + list(range(16, VOCAB)))


def setup_phase_3(model, prior):
    """Train sum_out to predict a+b at vocab rows 16-22. This is
    conceptually a 'read adder's output' task — the sum_out stream's
    own params + head rows 16-22 should learn to project from
    sum_out's own residual (which, being fed the same input tokens,
    should be able to compute a+b independently, or via pattern-match
    on adder's channels if joins were configured)."""
    _freeze_all_prior_streams_through(model,
                                       ["math", "echo_a", "echo_b", "parity"])
    with torch.no_grad():
        for p in model.streams["sum_out"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[16:23] == 0).all():
            model.head.weight[16:23].normal_(0, 0.02)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 16)) + list(range(23, VOCAB)))


def setup_phase_4(model, prior):
    """Train parity to predict (a+b) % 2 at rows 23-24."""
    _freeze_all_prior_streams_through(model,
                                       ["math", "echo_a", "echo_b", "sum_out"])
    with torch.no_grad():
        for p in model.streams["parity"].parameters():
            if (p == 0).all():
                p.normal_(0, 0.02)
        if (model.head.weight[23:25] == 0).all():
            model.head.weight[23:25].normal_(0, 0.02)
    _remove_prior_head_hooks(model)
    freeze_head_rows(model, rows=list(range(0, 23)) + list(range(25, VOCAB)))


# ----- Training loops -----

def _train_target(model, steps, lr, batch_size, seed,
                  offset, width, target_fn):
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
        logits = model(xs[idx])[:, 1, offset:offset + width]
        loss = F.cross_entropy(logits, ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    model.eval()
    return last, time.time() - t0


def train_echo_a(m, steps, lr, bs, seed):
    return _train_target(m, steps, lr, bs, seed, 8, 4, lambda a, b: a)


def train_echo_b(m, steps, lr, bs, seed):
    return _train_target(m, steps, lr, bs, seed, 12, 4, lambda a, b: b)


def train_sum_out(m, steps, lr, bs, seed):
    return _train_target(m, steps, lr, bs, seed, 16, 7, lambda a, b: a + b)


def train_parity(m, steps, lr, bs, seed):
    return _train_target(m, steps, lr, bs, seed, 23, 2, lambda a, b: (a + b) % 2)


# ----- Main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default="/tmp/minibench_5phase")
    args = ap.parse_args()

    runner = PhaseRunner(
        build_model_fn=_build_model,
        checkpoint_dir=Path(args.ckpt_dir),
    )
    phases = [
        Phase(0, "compiled_adder", setup_phase_0, None, gate_adder, 1.0, steps=0),
        Phase(1, "echo_a", setup_phase_1, train_echo_a, gate_echo_a, 0.75,
              steps=args.steps, lr=args.lr),
        Phase(2, "echo_b", setup_phase_2, train_echo_b, gate_echo_b, 0.75,
              steps=args.steps, lr=args.lr),
        Phase(3, "sum_out", setup_phase_3, train_sum_out, gate_sum_out, 0.75,
              steps=args.steps, lr=args.lr),
        Phase(4, "parity", setup_phase_4, train_parity, gate_parity, 0.75,
              steps=args.steps, lr=args.lr),
    ]

    print("=== minibench_5phase — 5 stacked sub-cards ===", flush=True)
    print(f"  streams: {[f'{s.name}(d={s.d_model})' for s in _CFG.streams]}",
          flush=True)
    print(f"  vocab: 32 — 0-7 adder | 8-11 echo_a | 12-15 echo_b | "
          f"16-22 sum | 23-24 parity", flush=True)
    print(f"  total params: {sum(p.numel() for p in _build_model().parameters()):,}",
          flush=True)
    print("", flush=True)

    for phase in phases:
        print(f"--- phase {phase.phase_id}: {phase.name} ({phase.steps} steps) ---",
              flush=True)
        r = runner.run_phase(phase)
        print(f"  gate: {r.gate_score*100:.1f}% "
              f"(threshold {r.min_threshold*100:.0f}%) "
              f"{'PASS' if r.passed else 'FAIL'}", flush=True)
        for prior_name, score in r.regression_scores.items():
            print(f"  regression [{prior_name}]: {score*100:.1f}%", flush=True)
        if r.final_loss > 0:
            print(f"  final loss: {r.final_loss:.4f}  "
                  f"wallclock: {r.train_wallclock_s:.2f}s", flush=True)
        if r.note:
            print(f"  note: {r.note}", flush=True)
        print("", flush=True)
        if not r.passed:
            print(f"=== STOP at phase {phase.phase_id} ===", flush=True)
            break

    print("=== summary ===", flush=True)
    print(f"  phases passed: {len(runner.history)}/{len(phases)}", flush=True)
    for r in runner.history:
        print(f"    [{r.phase_id}] {r.name:20s} gate={r.gate_score*100:5.1f}% "
              f"loss={r.final_loss:.3f}", flush=True)


if __name__ == "__main__":
    main()
