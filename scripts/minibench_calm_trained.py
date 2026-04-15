"""Train SubstrateHRLM on CALM-generated labels, eval with CALM verifier.

Closes the loop: the deterministic compute engine (CALM) produces labels
for our learned approximator (SubstrateHRLM), and verifies outputs at
eval time via the SAME engine. No hand-labeling, no hand-specified
target_fn's.

Pipeline:
  1. Build unified multi-stream tensor (compiled adder on math stream,
     trainable lm stream).
  2. Train lm stream on CALM-labeled data: samples from recipe_is_sum_prime
     (`is_prime(a+b)` → 0 or 1). CALM generates both prompts and labels.
  3. Gate: CALM-verified gate. Model prediction checked against
     safe_eval('is_prime(a+b)') for held-out samples.
  4. Compositional benchmark: held out some recipe combinations (random
     split), measure generalization gap.

If gap is small → composition emerges. If large → memorization.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.calm_training_bridge import (
    CALMOracle, calm_verified_gate, default_oracle,
    recipe_adder_small, recipe_is_sum_prime, recipe_sum_mod2,
)
from calm.llm_computer.channel_masking import freeze_head_rows
from calm.llm_computer.compositional_benchmark import (
    CompositionalBenchmark, report, split_leave_one_out,
)
from calm.llm_computer.multi_stream import (
    MultiStreamConfig, MultiStreamTransformer, StreamSpec,
    build_empty_multistream,
)
from calm.llm_computer.unified_chrlm import (
    freeze_stream_embeddings, freeze_stream_layer,
    install_compiled_in_stream,
)
from scripts.experiment_fast_weights_fusion import build_adder_tiny_small2d


# Output vocab partitioning:
#   0: False (for is_prime-like predictions)
#   1: True
#   2-8: sum tokens 0..6 (for adder output)
#   9: (a+b) % 2 = 0
#   10: (a+b) % 2 = 1
VOCAB = 16
MAX_LEN = 4


def _build_adder_src():
    return build_adder_tiny_small2d(target_layer=0, n_layers=1)


def _build_model():
    return build_empty_multistream(MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm", d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=VOCAB, max_len=MAX_LEN, use_hard_max=True,
    ))


def _prepare_model(seed: int = 42):
    torch.manual_seed(seed)
    model = _build_model()
    install_compiled_in_stream(
        model, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
    )
    # Random-init the lm stream
    with torch.no_grad():
        for p in model.streams["lm"].parameters():
            p.normal_(0.0, 0.02)
        # Train the ENTIRE head (0-15); all vocab used for task outputs
        model.head.weight.normal_(0.0, 0.02)
        # But keep adder's head slice (rows 2-8 get rebuilt from compiled
        # install; restore that for adder task)
    freeze_stream_layer(model, "math", layer_idx=0)
    freeze_stream_embeddings(model, "math")
    return model


def _prompt_to_tensor(prompt: tuple[int, ...], max_len: int = 4) -> torch.Tensor:
    """Pad prompt to max_len with zeros."""
    p = list(prompt)[:max_len]
    while len(p) < max_len:
        p.append(0)
    return torch.tensor([p], dtype=torch.long)


def predict(model, task, *, head_slice: slice):
    """Greedy predict: argmax over head_slice at position 1."""
    x = _prompt_to_tensor(task.prompt)
    model.eval()
    with torch.no_grad():
        logits = model(x)[0, 1, head_slice]
    return int(logits.argmax().item())


def train_on_recipe(
    model, oracle: CALMOracle, recipe_name: str,
    head_slice: slice,
    steps: int, lr: float, batch_size: int, seed: int,
):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = torch.Generator().manual_seed(seed)
    # Pre-generate a training pool
    pool = oracle.sample(recipe_name, n=256, seed=seed + 1)
    xs = torch.stack([_prompt_to_tensor(t.prompt)[0] for t in pool])
    ys = torch.tensor([t.answer for t in pool], dtype=torch.long)

    model.train()
    t0 = time.time()
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, len(pool), (batch_size,), generator=rng)
        logits = model(xs[idx])[:, 1, head_slice]
        loss = F.cross_entropy(logits, ys[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    model.eval()
    return last, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-path", default="/tmp/minibench_calm.pt")
    args = ap.parse_args()

    print("=== minibench_calm_trained — CALM as oracle ===", flush=True)

    # ---- Build model + install adder ----
    model = _prepare_model(args.seed)
    oracle = default_oracle()

    # Partial head freeze: rows 0-1 (is_prime output) are trainable;
    # rows 2-8 (adder sum) get maintained by adder's head contribution.
    # Rows 9-10 (sum_mod2) also trainable in a later phase if we add one.
    # Here we train only the is_sum_prime head (rows 0-1).

    # ---- Gate 1: pre-training baseline ----
    # Random init → random predictions
    pre_score = calm_verified_gate(
        predict_fn=lambda t: predict(model, t, head_slice=slice(0, 2)),
        recipe=recipe_is_sum_prime(), n_samples=50, seed=args.seed + 1,
    )
    print(f"\n  [gate] pre-training is_sum_prime: {pre_score*100:.1f}% "
          f"(random baseline ≈ 50% for binary)", flush=True)

    # ---- Train on CALM-labeled is_sum_prime ----
    print(f"\n--- training on CALM-labeled 'is_prime(a+b)' "
          f"({args.steps} steps) ---", flush=True)
    final_loss, wall = train_on_recipe(
        model, oracle, "calm_is_sum_prime",
        head_slice=slice(0, 2),
        steps=args.steps, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )
    print(f"  final_loss={final_loss:.4f}  wallclock={wall:.2f}s", flush=True)

    # ---- Gate 2: CALM-verified post-training ----
    post_score = calm_verified_gate(
        predict_fn=lambda t: predict(model, t, head_slice=slice(0, 2)),
        recipe=recipe_is_sum_prime(), n_samples=50, seed=args.seed + 2,
    )
    print(f"\n  [gate] post-training is_sum_prime: {post_score*100:.1f}% "
          f"(CALM-verified on held-out samples)", flush=True)

    # ---- Compositional benchmark: does composition generalize? ----
    # Build templates from CALM recipes.
    # Train set was is_sum_prime (a, b ∈ [0,3], composition of adder + is_prime).
    # Held-out: sum_mod2 (adder + modulo), a genuinely unseen composition.
    train_templates = [oracle.build_template("calm_is_sum_prime")]
    held_out_templates = [oracle.build_template("calm_sum_parity"),
                          oracle.build_template("calm_adder_small")]

    bench = CompositionalBenchmark(
        train_templates=train_templates,
        held_out_templates=held_out_templates,
        samples_per_template=20, seed=args.seed + 3,
    )

    def predict_for_task(task):
        # The vocab layout this model uses for its HEAD:
        #   rows 0-1: is_sum_prime output
        #   rows 2-8: adder output
        # For other templates, we don't have trained heads — predict
        # over relevant head_slice anyway and let the benchmark score it
        if task.cards_required == frozenset({"adder", "is_prime"}):
            return predict(model, task, head_slice=slice(0, 2))
        if task.cards_required == frozenset({"adder"}):
            return predict(model, task, head_slice=slice(2, 9)) - 2 \
                   + 0  # Actually adder outputs 0..6 natively
        # sum_parity (adder + modulo) — model has no trained head here
        # Predict over rows 0-1 (binary) anyway. Expected: poor perf
        # since training never saw this output location.
        if task.cards_required == frozenset({"adder", "modulo"}):
            return predict(model, task, head_slice=slice(0, 2))
        return -1  # unknown

    bench_result = bench.evaluate(predict_fn=predict_for_task)
    print("")
    print(report(bench_result), flush=True)

    # ---- Save ----
    torch.save(model.state_dict(), args.save_path)
    print(f"\n  saved: {args.save_path}", flush=True)

    # ---- Summary ----
    print("", flush=True)
    print("=== summary ===", flush=True)
    print(f"  pre-training gate:  {pre_score*100:.1f}%", flush=True)
    print(f"  post-training gate: {post_score*100:.1f}% "
          f"(Δ {(post_score-pre_score)*100:+.1f}pp)", flush=True)
    print(f"  train vs held-out gap: "
          f"{bench_result.generalization_gap*100:+.1f}pp "
          f"{'(generalizes)' if abs(bench_result.generalization_gap) < 0.3 else '(memorizes)'}",
          flush=True)


if __name__ == "__main__":
    main()
