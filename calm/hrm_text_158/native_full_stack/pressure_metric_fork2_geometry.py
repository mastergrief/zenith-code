"""Fork-2 de-risk geometry constants (smoke-only).

R5: real multi-arm geometry from run-3 ckpt (32 eligible BitLinear weights).
Plan flag N=18158319 was run-3 N_censored (events), NOT tensor numel.
Canonical flat-concat order matches _param_to_module / q_levels insertion order.
"""
from __future__ import annotations

import torch

CLIP = 127
INDEX_BITS = 32
INDEX_MASK = (1 << INDEX_BITS) - 1

RUN3_REAL_ARM_SHAPES: tuple[tuple[str, tuple[int, int]], ...] = tuple(
    (name, shape)
    for level in ("H_level", "L_level")
    for layer in range(4)
    for name, shape in (
        (f"model.{level}.core.layers.{layer}.attn.gqkv_proj.weight", (2048, 512)),
        (f"model.{level}.core.layers.{layer}.attn.o_proj.weight", (512, 512)),
        (f"model.{level}.core.layers.{layer}.mlp.gate_up_proj.weight", (3072, 512)),
        (f"model.{level}.core.layers.{layer}.mlp.down_proj.weight", (512, 1536)),
    )
)


def run3_total_numel() -> int:
    return int(sum(int(a * b) for _n, (a, b) in RUN3_REAL_ARM_SHAPES))


def make_zero_arms(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.int16,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, shape in RUN3_REAL_ARM_SHAPES:
        out[name] = torch.zeros(shape, device=device, dtype=dtype)
    return out


def arm_flat_offsets(arms: dict[str, torch.Tensor] | list[tuple[str, tuple[int, ...]]]) -> list[tuple[str, int, int]]:
    """Return (name, start, numel) in canonical iteration order."""
    if isinstance(arms, dict):
        items = list(arms.items())
        out: list[tuple[str, int, int]] = []
        off = 0
        for n, t in items:
            nn = int(t.numel())
            out.append((n, off, nn))
            off += nn
        return out
    out = []
    off = 0
    for n, shape in arms:
        nn = int(shape[0] * shape[1]) if len(shape) == 2 else int(torch.Size(shape).numel())
        out.append((n, off, nn))
        off += nn
    return out
