"""Fork B resume-parity arm state transforms (U/F/C/S/Z).

Torch/state ops only — no checkpoint IO / trainer_sub2 save-load.
"""

from __future__ import annotations

from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
    make_live_shadow_tensor_state,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_contracts import (
    canonical_json_sha256,
)


def clone_f_in_memory(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    """Test-only full-live-state clone. NOT checkpoint authority."""

    if state.exact_accumulator_shadow is None:
        raise ValueError("F requires live exact_accumulator_shadow")
    return BoundedDeltaTensorState(
        state_key=state.state_key,
        q_levels=state.q_levels.detach().cpu().clone().contiguous(),
        frozen_scale=state.frozen_scale.detach().cpu().clone().contiguous(),
        bounded_accumulator=state.bounded_accumulator,
        exact_accumulator_shadow=state.exact_accumulator_shadow.detach()
        .cpu()
        .clone()
        .contiguous(),
        bounded_accumulator_fresh_for_exact_shadow=bool(
            state.bounded_accumulator_fresh_for_exact_shadow
        ),
        bounded_accumulator_rebuild_hot_exact_indices=(
            state.bounded_accumulator_rebuild_hot_exact_indices
        ),
        bounded_accumulator_rebuild_cold_default_value=(
            state.bounded_accumulator_rebuild_cold_default_value
        ),
        event_coded_live_carrier=state.event_coded_live_carrier,
    )


def prepare_c_stale_for_save(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    """CURRENT path: keep stale bounded (fresh=False). Do not refresh."""

    if state.exact_accumulator_shadow is None:
        raise ValueError("C save requires live shadow to exist pre-save")
    # Intentional: leave bounded as-is (possibly stale).
    return state


def prepare_s_refresh_for_save(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    """Candidate path: rebuild bounded FROM live shadow before save."""

    if state.exact_accumulator_shadow is None:
        raise ValueError("S refresh requires live exact_accumulator_shadow")
    return state.with_fresh_bounded_accumulator()


def rehydrate_from_bounded(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    """Post-load: shadow=None → shadow_hat := decode(bounded)."""

    decoded = decode_bounded_accumulator_to_i16(state.bounded_accumulator)
    return BoundedDeltaTensorState(
        state_key=state.state_key,
        q_levels=state.q_levels.detach().cpu().contiguous(),
        frozen_scale=state.frozen_scale.detach().cpu().contiguous(),
        bounded_accumulator=state.bounded_accumulator,
        exact_accumulator_shadow=decoded,
        bounded_accumulator_fresh_for_exact_shadow=True,
        event_coded_live_carrier=None,
    )


def rehydrate_z_zeros(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    q = state.q_levels.detach().cpu().contiguous()
    zeros = torch.zeros_like(q, dtype=torch.int16)
    return make_bounded_tensor_state(
        state.state_key,
        q,
        state.frozen_scale,
        zeros,
        hot_exact_indices=(),
        cold_default_value=0,
    )


def evolve_shadow_one_step(
    state: BoundedDeltaTensorState,
    *,
    delta: int = 3,
) -> BoundedDeltaTensorState:
    """Toy live update for tests/smoke: bump a few coords; leave bounded stale."""

    if state.exact_accumulator_shadow is None:
        raise ValueError("evolve requires shadow")
    acc = state.exact_accumulator_shadow.detach().cpu().clone().contiguous().flatten()
    n = int(acc.numel())
    for idx in (0, 1, min(2, n - 1)):
        acc[idx] = int(acc[idx].item()) + int(delta)
    acc = acc.view_as(state.exact_accumulator_shadow)
    return make_live_shadow_tensor_state(state, state.q_levels, acc)


def estimate_bounded_bits(state: BoundedDeltaTensorState) -> int:
    bounded = state.bounded_accumulator
    # index(~25) + value(16) + flag(2) rough row cost for accounting tests
    rows = len(bounded.hot_exact_indices) + len(bounded.cold_exception_indices)
    return int(rows * 43) + 64  # + metadata floor


def comparison_stats_from_state(
    state: BoundedDeltaTensorState,
    *,
    step_tag: str,
    q_before: str | None = None,
) -> dict[str, Any]:
    """Synthetic gate-bearing surface for reducer/smoke fixtures (not science)."""

    shadow = state.exact_accumulator_shadow
    acc_sha = tensor_sha256(shadow) if shadow is not None else None
    q_after = tensor_sha256(state.q_levels)
    # Deterministic pseudo applied hash from acc+q
    applied = canonical_json_sha256({"q": q_after, "acc": acc_sha, "step": step_tag})[
        :16
    ]
    return {
        "q_sha256_before": q_before or q_after,
        "q_sha256_after": q_after,
        "applied_flat_indices_hash16": applied,
        "votes_sha256": canonical_json_sha256({"step": step_tag, "acc": acc_sha}),
        "global_rate_cap_accepted_indices_sha256": applied,
        "global_rate_cap_deferred_indices_sha256": canonical_json_sha256(
            {"def": applied}
        ),
        "global_rate_cap_applied_count": 1 if shadow is not None and int(shadow.abs().sum()) else 0,
        "flip_count": int(shadow.abs().sum().item()) if shadow is not None else 0,
        "q_changed_count": 0,
        "applied_selection_score_p50": float(
            shadow.float().abs().median().item() if shadow is not None and shadow.numel() else 0.0
        ),
        "applied_selection_score_p95": float(
            shadow.float().abs().quantile(0.95).item()
            if shadow is not None and shadow.numel()
            else 0.0
        ),
        "exact_accumulator_shadow_sha256_after": acc_sha,
    }
