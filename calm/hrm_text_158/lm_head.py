"""HRM-Text-1.58 LMHead.

Source: sapientinc/HRM-Text SHA 056c4ec, `models/lm_head.py:18-74`.

Deviations recorded in RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md:
- D1.4: single-GPU only. No dist.all_reduce on loss_divisor. loss_divisor
  = masks.sum() locally.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from calm.hrm_text_158.config import LMHeadConfig
from calm.hrm_text_158.layers import LinearInit, ScaledEmbeddingInit


IGNORE_LABEL_ID = -100


def _packing_sequence_sum(x: Tensor, cu_seqlens: Optional[Tensor]) -> Optional[Tensor]:
    """Port of `models/common.py:16-18`. Returns sum per packed sequence.

    Only used when cu_seqlens is provided (training with packed sequences).
    For our mini-smoke (one sequence per row), pass cu_seqlens=None.
    """
    if cu_seqlens is None:
        return None
    c = F.pad(x.cumsum(0), (1, 0))
    return c[cu_seqlens[1:]] - c[cu_seqlens[:-1]]


class LMHead(nn.Module):
    """Embedding-in + linear-out head wrapping a Transformer-style model.

    Port of `sapientinc/HRM-Text/models/lm_head.py:18-74`. Single-GPU per D1.4.

    Delegations (per upstream `lm_head.py:22-25`):
    - `compute_train_extra_args` — proxied through; load-bearing for trainer
      to surface the `bp_steps` schedule on the wrapper.
    - `create_cache` — DEFERRED per D1.9 (RESEARCH/HRM-Text-1.58/01_DEVIATIONS.md).
      Phase 1 is training-only; autoregressive inference (and thus cache) lands
      in Phase 1 Slice 2 / probe.
    """
    def __init__(self, model: nn.Module, config: LMHeadConfig) -> None:
        super().__init__()
        self.model = model
        head_hint = self.model.head_hint
        self.embed_tokens = ScaledEmbeddingInit(
            config.vocab_size,
            head_hint["in"]["dim"],
            init_std=head_hint["in"]["init_std"],
        )
        self.lm_head = LinearInit(
            head_hint["out"]["dim"],
            config.vocab_size,
            bias=False,
            init_std=head_hint["out"]["init_std"],
        )

    def compute_train_extra_args(self, step: int, total_steps: int) -> dict:
        """Delegate to the wrapped model.

        Port of upstream `models/lm_head.py:24` pattern (which sets
        `self.compute_train_extra_args = self.model.compute_train_extra_args`
        at __init__). We use an explicit method so torch.nn.Module's
        __setattr__ doesn't fight us; result is identical."""
        return self.model.compute_train_extra_args(step, total_steps)

    def forward(
        self,
        carry: Any,
        batch: dict,
        **kwargs,
    ) -> Tuple[Any, Tensor] | Tuple[Any, Tensor, dict]:
        """Forward + optional loss/metrics.

        batch must contain "inputs". If "labels" present, returns (carry, loss, metrics).
        Else returns (carry, logits).
        """
        input_embedding = self.embed_tokens(batch["inputs"])
        seq_info = {k: v for k, v in batch.items() if k not in ("inputs", "labels", "cu_seqlens")}
        new_carry, hidden = self.model(carry, input_embedding, **seq_info, **kwargs)
        logits = self.lm_head(hidden)

        if "labels" in batch:
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID
            # CE in FP32 per upstream `lm_head.py:52`
            loss_sum = F.cross_entropy(
                logits.to(torch.float32).flatten(0, -2),
                labels.to(torch.long).flatten(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="sum",
            )
            # Single-GPU: local divisor only. D1.4.
            loss_divisor = masks.sum().to(torch.float32).clamp(min=1.0)
            loss = loss_sum / loss_divisor

            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                local_valid_counts = masks.sum()
                cu_seqlens = batch.get("cu_seqlens")
                seq_num_tokens_correct = _packing_sequence_sum(is_correct.flatten() & masks.flatten(), cu_seqlens)
                seq_num_valid_tokens = _packing_sequence_sum(masks.flatten(), cu_seqlens)
                if seq_num_valid_tokens is not None and seq_num_tokens_correct is not None:
                    seq_is_valid = seq_num_valid_tokens > 0
                    exact_correct = ((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum()
                    exact_total = seq_is_valid.sum()
                else:
                    # Per-row exact accuracy: all valid positions correct in this row
                    row_correct = ((is_correct | ~masks).all(dim=-1) & masks.any(dim=-1)).sum()
                    row_total = masks.any(dim=-1).sum()
                    exact_correct = row_correct
                    exact_total = row_total
                metrics = {
                    "loss": (loss_sum.detach(), local_valid_counts),
                    "accuracy": ((is_correct & masks).sum(), local_valid_counts),
                    "exact_accuracy": (exact_correct, exact_total),
                }
            return new_carry, loss, metrics

        return new_carry, logits
