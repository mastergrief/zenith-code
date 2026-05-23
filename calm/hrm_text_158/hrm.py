"""HRM-Text-1.58 Hierarchical Reasoning Model.

Source: sapientinc/HRM-Text SHA 056c4ec,
`models/baselines/hrm_nocarry_bp_warmup.py:26-100`.

Two-level recurrence with bp_warmup gradient schedule.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn

from calm.hrm_text_158.config import HierarchicalReasoningModelConfig, TransformerConfig
from calm.hrm_text_158.layers import trunc_normal_init_
from calm.hrm_text_158.transformer import Transformer


class HierarchicalReasoningModelRecurrentBlock(nn.Module):
    """Transformer with additive input injection.

    Port of `sapientinc/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py:26-43`.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.core = Transformer(config)

    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        # ADDITIVE input injection. Port of `hrm_nocarry_bp_warmup.py:43`.
        return self.core(hidden_states + input_injection, **kwargs)


def _config_with_override(
    base: HierarchicalReasoningModelConfig,
    override: dict,
) -> TransformerConfig:
    """Build a TransformerConfig from HRM config + override dict.

    Upstream pattern (hrm_nocarry_bp_warmup.py:56):
      TransformerConfig(**(config.model_dump() | config.H_override))
    We use dataclasses.replace to apply overrides only for fields that
    exist on TransformerConfig.
    """
    valid_fields = {f for f in TransformerConfig.__dataclass_fields__}
    filtered_override = {k: v for k, v in override.items() if k in valid_fields}
    return TransformerConfig(
        max_seq_len=base.max_seq_len,
        n_layers=base.n_layers,
        hidden_size=base.hidden_size,
        num_heads=base.num_heads,
        expansion=base.expansion,
        attn_type=base.attn_type,
        init_type=base.init_type,
        init_std=base.init_std,
        norm_type=base.norm_type,
        norm_eps=base.norm_eps,
        pos_emb_type=base.pos_emb_type,
        rope_theta=base.rope_theta,
        **filtered_override,
    )


class HierarchicalReasoningModel(nn.Module):
    """Two-level (H, L) recurrent reasoning model.

    Port of `sapientinc/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py:46-100`.
    """
    def __init__(self, config: HierarchicalReasoningModelConfig) -> None:
        super().__init__()
        # half_layers: split layers evenly between H and L
        # Port of `hrm_nocarry_bp_warmup.py:50-52`
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2"
            effective_n_layers = config.n_layers // 2
        else:
            effective_n_layers = config.n_layers

        # Build the per-level transformer configs.
        # L_level uses the base config (with halved n_layers if applicable).
        # H_level can override; upstream applies the override on top of base.
        base_per_level_cfg = TransformerConfig(
            max_seq_len=config.max_seq_len,
            n_layers=effective_n_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            attn_type=config.attn_type,
            init_type=config.init_type,
            init_std=config.init_std,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            pos_emb_type=config.pos_emb_type,
            rope_theta=config.rope_theta,
            use_ternary_bulk=config.use_ternary_bulk,  # D2.1: propagate to H/L per-level
        )
        # H gets the override
        h_cfg = replace(base_per_level_cfg, **{
            k: v for k, v in config.H_override.items()
            if k in TransformerConfig.__dataclass_fields__
        })
        self.H_level = HierarchicalReasoningModelRecurrentBlock(h_cfg)
        self.L_level = HierarchicalReasoningModelRecurrentBlock(base_per_level_cfg)

        # Recurrence config
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps

        self.hidden_size = config.hidden_size
        self.head_hint = self.H_level.core.head_hint  # LMHead init hint from H

        # L-level initial hidden state (persistent buffer, bf16 hardcoded upstream)
        # Port of `hrm_nocarry_bp_warmup.py:69`
        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0),
            persistent=True,
        )

    def forward(
        self,
        carry: Any,  # always None for nocarry variant
        x: Tensor,
        bp_steps: int = 2,
        **seq_info,
    ) -> Tuple[None, Tensor]:
        """HRM forward loop.

        Port of `hrm_nocarry_bp_warmup.py:75-91`. Selectively enables grad
        per bp_steps schedule.

        T2 γ1 (codex msg 1779530833485-eb9296ca): when `kv_cache` is in
        `seq_info`, threads (level, rec_idx) per iteration so Attention can
        key its cache by (level, rec_idx, layer_idx). When `kv_cache` is
        None (default), behavior is identical to the no-cache path.
        """
        # z_L starts from persistent buffer, broadcast across batch
        z_H = x
        z_L = self.zL_init.to(x.dtype).expand_as(x)
        # bp_steps allocation: H prioritized
        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps
        cache_active = seq_info.get("kv_cache") is not None
        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(
                    torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)
                ):
                    L_kwargs = seq_info
                    if cache_active:
                        L_kwargs = {
                            **seq_info,
                            "kv_cache_level": "L",
                            "kv_cache_rec_idx": k,
                        }
                    z_L = self.L_level(z_L, z_H, **L_kwargs)
            with torch.set_grad_enabled(
                torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)
            ):
                H_kwargs = seq_info
                if cache_active:
                    H_kwargs = {
                        **seq_info,
                        "kv_cache_level": "H",
                        "kv_cache_rec_idx": i,
                    }
                z_H = self.H_level(z_H, z_L, **H_kwargs)
        return None, z_H

    def compute_train_extra_args(self, step: int, total_steps: int) -> dict:
        """Return extra forward kwargs for current training step.

        Port of `hrm_nocarry_bp_warmup.py:93-97`. Linear ramp of bp_steps
        from bp_min_steps to bp_max_steps over bp_warmup_ratio * total_steps.
        """
        warmup_steps = total_steps * self.bp_warmup_ratio
        progress = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        bp_steps = self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps))
        return {"bp_steps": bp_steps}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        """No-carry variant; always None. Port of `hrm_nocarry_bp_warmup.py:99-100`."""
        return None
