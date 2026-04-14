"""Router HRM — tiny classifier over query domain.

Wraps `HRMEncoder` with mean pooling + linear classification head.
Purpose: decide which sub-specialist (math/nl/word/gsm/meta) should
handle a query. The HRM encoder already shipped in the codebase is
reused verbatim — only the head is new.

Target: ≥95% classification accuracy at h=16 (~8-12K params). At h=16
the full router is small enough to run on any CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from calm.hrm.model import HRMConfig, HRMEncoder


@dataclass
class RouterConfig:
    vocab_size: int = 80
    hidden_size: int = 16
    num_heads: int = 4
    L_layers: int = 1
    H_layers: int = 1
    max_seq_len: int = 384
    num_labels: int = 5

    def to_hrm_config(self) -> HRMConfig:
        return HRMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            L_layers=self.L_layers,
            H_layers=self.H_layers,
            max_seq_len=self.max_seq_len,
            # Decoder bits unused but HRMConfig requires them.
            decoder_layers=1,
            max_dec_len=2,
        )


class RouterHRM(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = HRMEncoder(cfg.to_hrm_config())
        self.head = nn.Linear(cfg.hidden_size, cfg.num_labels, bias=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """input_ids: (B, S). Returns logits (B, num_labels)."""
        memory = self.encoder(input_ids)  # (B, S, D)
        if attention_mask is None:
            pad_id = 0
            attention_mask = (input_ids != pad_id).float()
        mask = attention_mask.unsqueeze(-1)
        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
