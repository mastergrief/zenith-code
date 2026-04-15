"""Per-card optimizer + loss declaration.

Current phase runner: one AdamW + cross-entropy for everything. But
cards have fundamentally different training shapes:
  - Compiled: no training
  - Classifier: CE loss, dense gradient
  - Regressor: MSE loss
  - Selector/router: REINFORCE (discrete choice, no differentiable)
  - Memory/contrastive: InfoNCE-style

Uniform optimization treats them all the same, which is sub-optimal.
This module introduces `CardTrainingSpec` so phases can declare their
loss + optimizer preference.

MVP ships the declarative scaffold + 3 built-in specs (classifier,
regressor, frozen-compiled). More exotic specs (REINFORCE, contrastive)
can slot in via the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CardTrainingSpec:
    """Declarative training configuration for a card.

    Attributes:
        name: card identifier.
        loss_fn: callable(logits, targets, mask=None) → scalar loss.
        optimizer_cls: e.g., torch.optim.AdamW, torch.optim.SGD.
        optimizer_kwargs: lr, weight_decay, etc.
        trainable_param_fn: callable(model) → list of parameters this
            card trains. Default: all requires_grad params.
        mode: "train" | "frozen" | "eval_only". Controls whether
            `build_optimizer` returns None.
    """
    name: str
    loss_fn: Callable[..., torch.Tensor]
    optimizer_cls: type = torch.optim.AdamW
    optimizer_kwargs: dict = field(default_factory=dict)
    trainable_param_fn: Optional[Callable] = None
    mode: str = "train"

    def build_optimizer(self, model: nn.Module) -> Optional[torch.optim.Optimizer]:
        """Return an optimizer instance for this card's parameters,
        or None if the card shouldn't train."""
        if self.mode in ("frozen", "eval_only"):
            return None
        if self.trainable_param_fn is not None:
            params = list(self.trainable_param_fn(model))
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None
        kwargs = {"lr": 1e-3, **self.optimizer_kwargs}
        return self.optimizer_cls(params, **kwargs)


# ----- Built-in loss functions -----

def ce_loss(logits: torch.Tensor, targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy loss with optional per-element mask."""
    B, C = logits.shape[0], logits.shape[-1]
    flat_logits = logits.reshape(-1, C)
    flat_targets = targets.reshape(-1)
    per_elem = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    if mask is not None:
        per_elem = per_elem * mask.reshape(-1).to(per_elem.dtype)
        denom = mask.sum().clamp(min=1.0)
        return per_elem.sum() / denom
    return per_elem.mean()


def mse_loss(logits: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE for regressor cards."""
    diff = (logits - targets.to(logits.dtype)) ** 2
    if mask is not None:
        diff = diff * mask.to(diff.dtype)
        return diff.sum() / mask.sum().clamp(min=1.0)
    return diff.mean()


def reinforce_loss(
    log_probs: torch.Tensor, reward: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """REINFORCE (vanilla policy gradient): -E[log_prob * reward].

    Useful for cards that emit discrete routing decisions where the
    "correct" choice isn't easily labeled but can be scored
    post-hoc (e.g., did the downstream task succeed?).
    """
    loss = -log_probs * reward
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def contrastive_loss(
    anchor: torch.Tensor, positive: torch.Tensor,
    negatives: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE: pull anchor toward positive, push away from negatives."""
    # cosine similarities
    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1) / temperature
    pos_sim = cos(anchor, positive).unsqueeze(-1)       # (B, 1)
    neg_sim = cos(anchor.unsqueeze(1), negatives)       # (B, K)
    logits = torch.cat([pos_sim, neg_sim], dim=-1)       # (B, 1+K)
    targets = torch.zeros(logits.size(0), dtype=torch.long,
                          device=logits.device)
    return F.cross_entropy(logits, targets)


# ----- Presets -----

def classifier_spec(name: str, lr: float = 1e-3,
                    param_fn: Optional[Callable] = None) -> CardTrainingSpec:
    """Standard classifier: AdamW + CE."""
    return CardTrainingSpec(
        name=name, loss_fn=ce_loss,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr},
        trainable_param_fn=param_fn,
    )


def regressor_spec(name: str, lr: float = 1e-3,
                   param_fn: Optional[Callable] = None) -> CardTrainingSpec:
    """Regressor: AdamW + MSE."""
    return CardTrainingSpec(
        name=name, loss_fn=mse_loss,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr},
        trainable_param_fn=param_fn,
    )


def frozen_compiled_spec(name: str) -> CardTrainingSpec:
    """Compiled card — no training. build_optimizer returns None."""
    return CardTrainingSpec(
        name=name,
        loss_fn=lambda *a, **kw: torch.zeros(1),
        mode="frozen",
    )


def reinforce_spec(name: str, lr: float = 1e-3,
                   param_fn: Optional[Callable] = None) -> CardTrainingSpec:
    """Router card — policy gradient (REINFORCE)."""
    return CardTrainingSpec(
        name=name, loss_fn=reinforce_loss,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": lr},
        trainable_param_fn=param_fn,
    )


def contrastive_spec(name: str, lr: float = 1e-3, temperature: float = 0.1,
                     param_fn: Optional[Callable] = None) -> CardTrainingSpec:
    """Memory/contrastive card — InfoNCE."""
    def loss_with_temp(*args, **kw):
        return contrastive_loss(*args, temperature=temperature, **kw)
    return CardTrainingSpec(
        name=name, loss_fn=loss_with_temp,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": lr},
        trainable_param_fn=param_fn,
    )
