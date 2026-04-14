"""Option 3 MVP — differentiable substrate stability test.

Take the compiled `adder_tiny` (1020 params, 16/16 exhaustive), enable
gradients on every weight, fine-tune with tiny LR on the same 16
in-distribution cases, and verify exhaustive accuracy stays 16/16.

If the compiled-then-fine-tuned model still passes, it proves:
  - Compiled weights can survive constrained fine-tuning.
  - Option 3 (differentiable substrate) is mechanically feasible.

If accuracy drops, we learn the brittleness threshold and know what
constraints (topology freeze / coefficient-only updates / LoRA
adapters) the real L6-phase-4 work needs.

Three regimes tested:
  1. Identity fine-tune — train on same 16 cases. Should preserve.
  2. Noise fine-tune    — add small Gaussian noise to weights, measure.
  3. Extension fine-tune — hide 4 of 16 cases, train on other 12, see
                           if model can still classify the hidden 4.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from calm.llm_computer.programs.adder_tiny import build_adder_tiny


def _exhaustive_accuracy(model) -> int:
    correct = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            correct += int(got == a + b)
    return correct


def _train_on_cases(model, cases, lr=1e-4, epochs=50):
    """Teacher-force the output. cases is a list of (a, b) pairs."""
    # Need softmax for gradients — temporarily turn off hard_max.
    orig = model.config.use_hard_max
    model.config.use_hard_max = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        total = 0.0
        for a, b in cases:
            x = torch.tensor([[a, b]], dtype=torch.long)
            target = torch.tensor([a + b], dtype=torch.long)
            logits = model(x)[:, 1, :]  # (1, vocab)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / max(len(cases), 1))
    model.config.use_hard_max = orig
    return losses


def test_identity_finetune():
    print("\n=== regime 1: identity fine-tune (train on all 16 cases) ===")
    model = build_adder_tiny(vocab_size=8, max_len=4)
    pre = _exhaustive_accuracy(model)
    all_cases = [(a, b) for a in range(4) for b in range(4)]
    losses = _train_on_cases(model, all_cases, lr=1e-4, epochs=20)
    post = _exhaustive_accuracy(model)
    print(f"  pre-finetune:  {pre}/16")
    print(f"  post-finetune: {post}/16")
    print(f"  loss trajectory: {[f'{l:.4f}' for l in losses[::4]]}")
    print(f"  VERDICT: {'PASS' if post == 16 else 'FAIL'}")
    return pre, post


def test_noise_perturbation():
    print("\n=== regime 2: weight noise injection ===")
    for sigma in (1e-4, 1e-3, 1e-2, 1e-1):
        model = build_adder_tiny(vocab_size=8, max_len=4)
        pre = _exhaustive_accuracy(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * sigma)
        post = _exhaustive_accuracy(model)
        print(f"  sigma={sigma:.0e}: {pre}/16 → {post}/16")


def test_extension_finetune():
    print("\n=== regime 3: extension fine-tune (12 seen, 4 held-out) ===")
    held_out = [(0, 3), (1, 2), (3, 1), (2, 2)]
    train_cases = [(a, b) for a in range(4) for b in range(4)
                   if (a, b) not in held_out]
    model = build_adder_tiny(vocab_size=8, max_len=4)
    pre_all = _exhaustive_accuracy(model)

    def _acc(cases):
        return sum(
            1 for a, b in cases
            if int(model(torch.tensor([[a, b]], dtype=torch.long))[0, 1].argmax().item()) == a + b
        )

    pre_seen = _acc(train_cases)
    pre_held = _acc(held_out)
    losses = _train_on_cases(model, train_cases, lr=1e-4, epochs=20)
    post_seen = _acc(train_cases)
    post_held = _acc(held_out)

    print(f"  pre:  seen {pre_seen}/12, held-out {pre_held}/4, total {pre_all}/16")
    print(f"  post: seen {post_seen}/12, held-out {post_held}/4")
    # The compiled adder already gets ALL cases right — so "generalization"
    # here is really "does fine-tuning on a subset preserve held-out perf?"
    print(f"  VERDICT: held-out preservation "
          f"{'OK' if post_held == pre_held else 'DEGRADED'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    print("[diff-substrate] loading compiled adder_tiny...")
    m = build_adder_tiny(vocab_size=8, max_len=4)
    print(f"[diff-substrate] base: {_exhaustive_accuracy(m)}/16, "
          f"{m.param_count():,} params")
    test_identity_finetune()
    test_noise_perturbation()
    test_extension_finetune()
