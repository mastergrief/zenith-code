"""Tests for per-card optimizer specs."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from calm.llm_computer.card_optimizers import (
    CardTrainingSpec, ce_loss, classifier_spec, contrastive_loss,
    contrastive_spec, frozen_compiled_spec, mse_loss,
    regressor_spec, reinforce_loss, reinforce_spec,
)


def _dummy_model():
    return nn.Linear(4, 3)


def test_ce_loss_basic():
    logits = torch.randn(2, 5)
    targets = torch.tensor([0, 3])
    loss = ce_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ce_loss_with_mask():
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_masked = ce_loss(logits, targets, mask=mask)
    loss_unmasked = ce_loss(logits, targets)
    # Different outputs (mask drops 2 elements)
    assert loss_masked.item() != loss_unmasked.item()


def test_mse_loss_zero_when_equal():
    x = torch.ones(3, 4)
    loss = mse_loss(x, x)
    assert loss.item() == 0.0


def test_mse_loss_positive_when_different():
    x = torch.ones(3, 4)
    y = torch.zeros(3, 4)
    loss = mse_loss(x, y)
    assert loss.item() == 1.0


def test_reinforce_loss_sign():
    # Positive reward with positive log_prob → negative loss (gradient
    # pushes log_prob higher)
    log_probs = torch.tensor([0.5])
    reward = torch.tensor([1.0])
    loss = reinforce_loss(log_probs, reward)
    assert loss.item() < 0
    # Positive reward with negative log_prob → positive loss
    loss_neg = reinforce_loss(torch.tensor([-0.5]), torch.tensor([1.0]))
    assert loss_neg.item() > 0


def test_contrastive_loss_prefers_positive():
    """Anchor close to positive, far from negatives → low loss."""
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])  # identical → cos=1
    negatives = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])  # orthogonal/opposite
    loss_good = contrastive_loss(anchor, positive, negatives)

    # Swap: anchor close to NEGATIVE — should give higher loss
    anchor2 = torch.tensor([[0.0, 1.0]])
    loss_bad = contrastive_loss(anchor2, positive, negatives)
    assert loss_bad > loss_good


def test_classifier_spec_builds_optimizer():
    spec = classifier_spec("cardA", lr=0.01)
    assert spec.name == "cardA"
    opt = spec.build_optimizer(_dummy_model())
    assert opt is not None
    assert isinstance(opt, torch.optim.AdamW)
    # Check LR
    assert opt.param_groups[0]["lr"] == 0.01


def test_regressor_spec_uses_mse():
    spec = regressor_spec("cardB")
    x = torch.tensor([1.0])
    y = torch.tensor([1.0])
    loss = spec.loss_fn(x, y)
    assert loss.item() == 0.0


def test_frozen_compiled_spec_returns_no_optimizer():
    spec = frozen_compiled_spec("adder")
    opt = spec.build_optimizer(_dummy_model())
    assert opt is None


def test_reinforce_spec_uses_sgd():
    spec = reinforce_spec("router", lr=0.1)
    opt = spec.build_optimizer(_dummy_model())
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.1


def test_contrastive_spec_temperature_plumbed():
    spec = contrastive_spec("memory", temperature=0.5)
    # Loss function should reference the stored temperature — validate by
    # comparing two temperatures on the same input.
    spec_hot = contrastive_spec("memory2", temperature=10.0)
    anchor = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([[0.5, 0.5]])
    neg = torch.tensor([[[-1.0, 0.0]]])
    l1 = spec.loss_fn(anchor, pos, neg)
    l2 = spec_hot.loss_fn(anchor, pos, neg)
    assert l1.item() != l2.item()


def test_custom_param_fn():
    """Can restrict optimizer to a subset of model params."""
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    # Only train the second layer
    spec = classifier_spec(
        "selective", lr=1e-3,
        param_fn=lambda m: list(m[1].parameters()),
    )
    opt = spec.build_optimizer(model)
    assert opt is not None
    # Optimizer should have exactly the second layer's params
    opt_params = set(id(p) for group in opt.param_groups for p in group["params"])
    expected = set(id(p) for p in model[1].parameters())
    assert opt_params == expected


def test_spec_with_no_trainable_params_returns_none():
    model = nn.Linear(4, 3)
    for p in model.parameters():
        p.requires_grad = False
    spec = classifier_spec("nothing")
    assert spec.build_optimizer(model) is None


if __name__ == "__main__":
    test_ce_loss_basic()
    print("[ok] ce_loss basic")
    test_ce_loss_with_mask()
    print("[ok] ce_loss with mask")
    test_mse_loss_zero_when_equal()
    print("[ok] mse_loss zero when equal")
    test_mse_loss_positive_when_different()
    print("[ok] mse_loss positive when different")
    test_reinforce_loss_sign()
    print("[ok] reinforce_loss sign conventions")
    test_contrastive_loss_prefers_positive()
    print("[ok] contrastive_loss favors close positive")
    test_classifier_spec_builds_optimizer()
    print("[ok] classifier_spec builds AdamW")
    test_regressor_spec_uses_mse()
    print("[ok] regressor_spec uses MSE")
    test_frozen_compiled_spec_returns_no_optimizer()
    print("[ok] frozen_compiled returns None")
    test_reinforce_spec_uses_sgd()
    print("[ok] reinforce_spec uses SGD")
    test_contrastive_spec_temperature_plumbed()
    print("[ok] contrastive temperature is tunable")
    test_custom_param_fn()
    print("[ok] custom param_fn restricts optimizer")
    test_spec_with_no_trainable_params_returns_none()
    print("[ok] no trainable params → None optimizer")
