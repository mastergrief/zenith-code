"""R51 tier-3 distillation student.

Hypothesis (R51): a ~1-5M-param Small2DTransformer-based student, trained
via MSE on (X_in, X_out) pairs captured at Gemma's L24 across 6 broad
domains, can reproduce L24's residual contribution well enough to install
as an additive CardSlot without degrading off-distribution capabilities
(R51.5 dual gate: >=80% on training distribution, >=95% off-distribution).

I/O contract:
    forward(x: [B, S, 2560] fp32) -> [B, S, 2560] fp32
    Same shape in and out. Input is Gemma's residual entering L24; output
    is the student's prediction of Gemma's L24 contribution at the same
    positions.

Parameter budget target: 1-5M total trainable params. The dominant term
is the projection pair (2 * 2560 * d_model); the Small2DTransformer core
adds a smaller share. Default d_model=128, n_layers=2 lands in budget.

Substrate compliance: d_head=2 invariant honored via Small2DConfig (asserted
in config, enforced by d_model % 2 == 0). Install path (R51.4): wrap this
module in a CardSlot at reserved residual channels; use_full_residual=True
so the student reads full [:,:,0:2560]; output_fn adds its 2560-d prediction
to Gemma's residual additively. Preservation masking on Gemma's native L24
contribution at the reserved channels keeps downstream layers reading the
student's output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class R51StudentConfig:
    d_io: int = 2560
    d_model: int = 128
    n_layers: int = 2
    d_ffn: int = 512
    max_len: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.d_model % 2 == 0, (
            f"d_model must be even for d_head=2 invariant, got {self.d_model}"
        )


class R51Student(nn.Module):
    def __init__(self, config: R51StudentConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_io, config.d_model, bias=False)
        self.in_norm = nn.LayerNorm(config.d_model)

        n_heads = config.d_model // 2
        core_cfg = Small2DConfig(
            vocab_size=1,
            d_model=config.d_model,
            n_heads=n_heads,
            n_layers=config.n_layers,
            d_ffn=config.d_ffn,
            max_len=config.max_len,
            use_hard_max=False,
        )
        self.core_cfg = core_cfg
        self.core = Small2DTransformer(core_cfg)

        self.pos = nn.Embedding(config.max_len, config.d_model)

        self.out_proj = nn.Linear(config.d_model, config.d_io, bias=False)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

    def _core_forward(self, h: torch.Tensor) -> torch.Tensor:
        B, S, _ = h.shape
        cfg = self.core_cfg
        for layer in range(cfg.n_layers):
            qkv = self.core.W_qkv[layer](h)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            attn = self.core._attention(q, k, v, hard_max=cfg.use_hard_max)
            attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            h = h + self.core.W_out[layer](attn)
            gate, val = self.core.ff_in[layer](h).chunk(2, dim=-1)
            h = h + self.core.ff_out[layer](F.relu(gate) * val)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        assert S <= self.config.max_len, (
            f"sequence length {S} exceeds max_len {self.config.max_len}"
        )
        h = self.in_proj(x)
        pos_idx = torch.arange(S, device=x.device)
        h = h + self.pos(pos_idx)
        h = self.in_norm(h)
        h = self._core_forward(h)
        y = self.out_proj(h)
        return y

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = R51StudentConfig()
    m = R51Student(cfg)
    print(f"params: {m.param_count():,} ({m.param_count()/1e6:.2f}M)")
    x = torch.randn(2, 16, 2560)
    y = m(x)
    assert y.shape == x.shape, f"{y.shape} vs {x.shape}"
    mag = y.abs().mean().item()
    assert mag < 0.1, f"untrained y too large: {mag}"
    print(f"zero-init output magnitude: {mag:.6f}")
    print("R51Student self-test PASS")
