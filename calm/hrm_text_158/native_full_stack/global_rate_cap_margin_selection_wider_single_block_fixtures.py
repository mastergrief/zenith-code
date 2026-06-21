"""B2-5a″ Stage-B Path A fixture helpers (Stage-B-owned; does not mutate banked builders)."""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapTensorInput
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_multiblock_step0_budget import (
    REALISTIC_ROW_COUNTS,
    build_realistic_fixture_inputs,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_margin_selection_step0_budget import (
    build_vote_update_spec,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


def move_global_rate_cap_inputs_to_device(
    inputs: list[GlobalRateCapTensorInput],
    device: torch.device,
    *,
    max_abs_per_tensor: int = 256,
    fraction_per_tensor: float = 1.0,
) -> list[GlobalRateCapTensorInput]:
    """Rebuild synthetic vote-update inputs on ``device`` without touching banked builders."""

    spec = build_vote_update_spec(
        max_abs_per_tensor=max_abs_per_tensor,
        fraction_per_tensor=fraction_per_tensor,
    )
    vote_mag = max(spec.threshold_abs + 1, max_abs_per_tensor)
    moved: list[GlobalRateCapTensorInput] = []
    for inp in inputs:
        numel = int(inp.state.q_levels.numel())
        state = VoteUpdateState(
            q_levels=torch.zeros(numel, dtype=torch.int8, device=device),
            accumulators=torch.zeros(numel, dtype=torch.int16, device=device),
        )
        votes = torch.full((numel,), vote_mag, dtype=torch.int16, device=device)
        plan = plan_integer_vote_update_reference(
            state,
            VoteUpdateInputs(votes=votes),
            spec,
        )
        moved.append(
            GlobalRateCapTensorInput(
                state_key=inp.state_key,
                state=state,
                plan=plan,
            )
        )
    return moved


def build_realistic_fixture_inputs_on_device(
    *,
    target_row_count: int,
    device: torch.device,
    max_abs_per_tensor: int = 256,
) -> list[GlobalRateCapTensorInput]:
    if target_row_count not in REALISTIC_ROW_COUNTS:
        raise ValueError(f"target_row_count must be one of {REALISTIC_ROW_COUNTS}")
    cpu_inputs = build_realistic_fixture_inputs(target_row_count=target_row_count)
    return move_global_rate_cap_inputs_to_device(
        cpu_inputs,
        device,
        max_abs_per_tensor=max_abs_per_tensor,
    )


__all__ = [
    "build_realistic_fixture_inputs_on_device",
    "move_global_rate_cap_inputs_to_device",
]
