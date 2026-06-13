"""C2.0 default-off bounded-delta learner seams.

This module ports the minimal S1 gradient -> move -> rank-bucket vote
semantics into the HRM-Text-1.58 repo without making creditdir a runtime
dependency. It is CPU/reference glue for the C2.0 learner integration gate:
projection, vote ranking, bounded accumulator update, authoritative forward
materialization, persistence metadata, and FP-master exclusion proof stay
separate so later GPU kernels can replace them independently.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import types
from typing import Any, Callable, Mapping, Sequence, Union

import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
    BoundedDeltaAccumulatorState,
    decode_bounded_accumulator_to_i16,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
    INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.source_pointers import (
    LIVE_S1_TRAINER_POINTER,
    SourcePointer,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
    assert_two_tier_threshold_receipt_consistent,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
)


RUN_BOUNDED_DELTA_LEARNER_ENV = "HRM_TEXT_158_RUN_BOUNDED_DELTA_LEARNER"
BOUNDED_DELTA_LEARNER_SCHEMA_VERSION = (
    "hrm_text_158_c2p0_bounded_delta_learner/v0.default_off_cpu_reference"
)
BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION = (
    "hrm_text_158_c2p0_bounded_delta_authoritative_state/v0"
)
S1_ORACLE_REANCHOR_SCHEMA_VERSION = "hrm_text_158_c2p0_s1_oracle_reanchor/v0"
S1_PROJECTION_LAW = "ported_s1_gradient_sign_to_ternary_move"
S1_RANK_BUCKET_VOTE_LAW = "ported_s1_rank_bucketed_integer_votes"
S1_SIGN_PRESSURE_VOTE_LAW = "rank_free_sign_pressure_constant_threshold_votes"
S1_INVERTED_SIGN_PRESSURE_VOTE_LAW = "inverted_rank_free_sign_pressure_constant_threshold_votes"
AUTHORITATIVE_STATE_SOURCE = "q_scale_bounded_delta_state_of_truth"
BOUNDED_UPDATE_ATTRIBUTION = "q_acc_backlog_changed_by_bounded_delta_vote_update_only"
DEFAULT_DRY_RUN_PARENT_HASH_BASIS = "no_parent_checkpoint_path_supplied_no_pt_touch"
FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION = (
    "hrm_text_158_front_c/v0.live_identity_observation_cloned_cpu"
)
AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY = "cpu_legacy"
AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT = "device_resident"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dict_without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in value.items() if v is not None}


@dataclass(frozen=True)
class RankVoteBin:
    lo_inclusive: float
    hi_exclusive: float
    vote_abs: int
    include_hi: bool = False

    def validate(self) -> None:
        if self.lo_inclusive < 0.0 or self.hi_exclusive > 1.0:
            raise ValueError("rank vote bin bounds must live in [0, 1]")
        if self.lo_inclusive >= self.hi_exclusive:
            raise ValueError("rank vote bin lo_inclusive must be < hi_exclusive")
        if self.vote_abs <= 0 or self.vote_abs > 32767:
            raise ValueError("rank vote_abs must fit positive int16")

    def to_dict(self) -> dict[str, int | float | bool]:
        out = asdict(self)
        out["vote_abs"] = int(out["vote_abs"])
        return out


@dataclass(frozen=True)
class RankVoteSpec:
    rank_bins: tuple[RankVoteBin, ...]
    mode: str = S1_RANK_BUCKET_VOTE_LAW
    rank_method: str = "grouped_bisect_right"

    def validate(self) -> None:
        if self.mode != S1_RANK_BUCKET_VOTE_LAW:
            raise ValueError(f"unsupported rank vote mode {self.mode!r}")
        if self.rank_method not in {"grouped_bisect_right", "searchsorted_reference"}:
            raise ValueError(f"unsupported rank method {self.rank_method!r}")
        if not self.rank_bins:
            raise ValueError("rank_bins must be non-empty")
        for item in self.rank_bins:
            item.validate()
        if self.rank_bins[0].lo_inclusive > 0.0:
            raise ValueError("rank bins must cover rank 0")
        if not self.rank_bins[-1].include_hi or self.rank_bins[-1].hi_exclusive < 1.0:
            raise ValueError("final rank bin must include rank 1.0")

    def to_live_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode,
            "rank_method": self.rank_method,
            "rank_bins": [item.to_dict() for item in self.rank_bins],
        }


def default_dry_run_rank_vote_spec() -> RankVoteSpec:
    """Small explicit C2.0 smoke spec; callers may pass the real prereg bins."""

    return RankVoteSpec(
        rank_bins=(
            RankVoteBin(0.0, 0.5, 1),
            RankVoteBin(0.5, 1.0, 4, include_hi=True),
        ),
    )


def _cpu_float32_rank_fraction(rank_position: int, count: int) -> float:
    rank = torch.tensor(rank_position, dtype=torch.int64).to(torch.float32)
    return float((rank / float(count)).item())


def _cpu_float32_scalar(value: float) -> float:
    return float(torch.tensor(float(value), dtype=torch.float32).item())


def _first_rank_position_matching(count: int, predicate: Any) -> int:
    lo = 1
    hi = count + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(_cpu_float32_rank_fraction(mid, count)):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _rank_bin_bounds(count: int, bin_spec: RankVoteBin) -> tuple[int, int]:
    lo = _cpu_float32_scalar(float(bin_spec.lo_inclusive))
    hi = _cpu_float32_scalar(float(bin_spec.hi_exclusive))
    lo_rank = _first_rank_position_matching(count, lambda rank: rank >= lo)
    if bool(bin_spec.include_hi):
        hi_limit = _first_rank_position_matching(count, lambda rank: rank > hi)
    else:
        hi_limit = _first_rank_position_matching(count, lambda rank: rank >= hi)
    return lo_rank, hi_limit


def _bisect_right_rank_positions_by_equal_value_group(abs_values: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(abs_values).all().item()):
        raise ValueError("rank bucket credit contains non-finite values")
    abs_bits = abs_values.contiguous().view(torch.int32)
    sorted_bits, order = torch.sort(abs_bits)
    n = int(abs_values.numel())
    group_start = torch.ones(n, dtype=torch.bool, device=abs_values.device)
    group_start[1:] = sorted_bits[1:] != sorted_bits[:-1]
    group_id = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    group_end = torch.ones(n, dtype=torch.bool, device=abs_values.device)
    group_end[:-1] = sorted_bits[:-1] != sorted_bits[1:]
    group_end_ranks = (torch.nonzero(group_end, as_tuple=False).flatten() + 1).to(torch.int64)
    rank_positions_sorted = group_end_ranks[group_id]
    rank_positions = torch.empty_like(rank_positions_sorted)
    rank_positions[order] = rank_positions_sorted
    return rank_positions


def project_s1_gradient_to_moves(grad: torch.Tensor, q_levels: torch.Tensor) -> torch.Tensor:
    """Port S1 `_project_fp_gradient_to_moves` exactly."""

    if grad.shape != q_levels.shape:
        raise ValueError("grad and q_levels must have identical shapes")
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    moves = torch.zeros_like(q_levels, dtype=torch.int8)
    moves[(q_levels < 0) & (grad < 0)] = 1
    moves[(q_levels == 0) & (grad < 0)] = 1
    moves[(q_levels == 0) & (grad > 0)] = -1
    moves[(q_levels > 0) & (grad > 0)] = -1
    return moves


def rank_bucketed_int16_votes(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    spec: RankVoteSpec,
) -> torch.Tensor:
    """Port S1 rank-bucketed integer vote mapping over explicit in-repo bins."""

    spec.validate()
    if credit.shape != projected_moves.shape:
        raise ValueError("credit/projected_moves tensor shape mismatch")
    flat_credit = credit.detach().flatten().to(torch.float32)
    flat_moves = projected_moves.detach().flatten().to(torch.int8)
    votes = torch.zeros_like(flat_moves, dtype=torch.int16)
    candidate_idx = torch.nonzero(flat_moves != 0, as_tuple=False).flatten()
    if candidate_idx.numel() == 0:
        return votes.view_as(projected_moves)
    abs_values = flat_credit[candidate_idx].abs()
    if spec.rank_method == "grouped_bisect_right":
        rank_positions = _bisect_right_rank_positions_by_equal_value_group(abs_values)
        ranks = None
    else:
        sorted_abs = torch.sort(abs_values).values
        ranks = torch.searchsorted(sorted_abs, abs_values, right=True).to(torch.float32) / float(
            candidate_idx.numel()
        )
        rank_positions = None
    vote_abs = torch.zeros(candidate_idx.numel(), dtype=torch.int16, device=flat_credit.device)
    matched = torch.zeros(candidate_idx.numel(), dtype=torch.bool, device=flat_credit.device)
    for item in spec.rank_bins:
        if spec.rank_method == "grouped_bisect_right":
            assert rank_positions is not None
            lo_rank, hi_limit = _rank_bin_bounds(int(candidate_idx.numel()), item)
            mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
        else:
            assert ranks is not None
            include_hi_mask = ranks <= float(item.hi_exclusive) if item.include_hi else torch.zeros_like(
                ranks,
                dtype=torch.bool,
            )
            mask = (ranks >= float(item.lo_inclusive)) & (
                (ranks < float(item.hi_exclusive)) | include_hi_mask
            )
        vote_abs[mask] = int(item.vote_abs)
        matched |= mask
    if not bool(matched.all().item()):
        raise ValueError("rank-bucket vote mapping left unmatched candidates")
    votes[candidate_idx] = (flat_moves[candidate_idx].to(torch.int16) * vote_abs).to(torch.int16)
    return votes.view_as(projected_moves)


def _as_bsi(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"{name} must be rank-2 or rank-3, got shape {tuple(tensor.shape)}")


def weighted_grad_from_captures(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
) -> torch.Tensor:
    """Reconstruct weight gradient from captured inputs and grad_outputs."""

    if not inputs or not grad_outputs:
        raise ValueError("inputs and grad_outputs must be non-empty")
    if len(inputs) < len(grad_outputs):
        raise ValueError("capture call-count mismatch")
    paired_inputs = inputs[-len(grad_outputs):]
    weight_dims = tuple(int(dim) for dim in weight_shape)
    if len(weight_dims) != 2:
        raise ValueError(f"weight_shape must be rank-2, got {weight_dims}")
    reference_input = _as_bsi(
        paired_inputs[0].detach().to(torch.float32),
        name="input",
    )
    weighted_grad = torch.zeros(
        weight_dims,
        dtype=torch.float32,
        device=reference_input.device,
    )
    for inp, grad_out in zip(paired_inputs, reversed(list(grad_outputs))):
        input_bsi = _as_bsi(inp.detach().to(torch.float32), name="input")
        grad_out_bso = _as_bsi(grad_out.detach().to(torch.float32), name="grad_out")
        if input_bsi.device != grad_out_bso.device:
            raise ValueError(
                "inputs and grad_outputs must share the same device for weighted_grad reconstruction"
            )
        weighted_grad += torch.einsum(
            "bso,bsi->oi",
            grad_out_bso,
            input_bsi,
        )
    return weighted_grad


# Chunk flat_indices before index_select to cap transient gather peak.
# peak_bytes ≈ 2 * B * S * chunk_size * 4 @ B=1,S=384 → ~100 MB for 32768.
# Pre-registered fallback if Stage-2b validation smoke still OOMs: 8192.
PROXY_GATHER_FLAT_INDEX_CHUNK_SIZE = 32768
PROXY_GATHER_FLAT_INDEX_CHUNK_SIZE_FALLBACK = 8192


def _accumulate_weighted_grad_proxy_chunk_from_captures(
    paired_inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    index_device: torch.device,
    proxies: torch.Tensor,
    diag_fisher: torch.Tensor,
) -> None:
    for inp, grad_out in zip(paired_inputs, reversed(list(grad_outputs))):
        input_bsi = _as_bsi(inp.detach().to(torch.float32), name="input")
        grad_out_bso = _as_bsi(grad_out.detach().to(torch.float32), name="grad_out")
        if input_bsi.device != grad_out_bso.device:
            raise ValueError(
                "inputs and grad_outputs must share the same device for candidate credit gathering"
            )
        if input_bsi.device != index_device:
            row_local = row_indices.to(device=input_bsi.device)
            col_local = col_indices.to(device=input_bsi.device)
        else:
            row_local = row_indices
            col_local = col_indices
        gathered_inputs = torch.index_select(input_bsi, dim=-1, index=col_local)
        gathered_grad_outputs = torch.index_select(grad_out_bso, dim=-1, index=row_local)
        proxies += (gathered_inputs * gathered_grad_outputs).sum(dim=(0, 1)).to(
            device=index_device
        )
        diag_fisher += (
            gathered_inputs.square() * gathered_grad_outputs.square()
        ).sum(dim=(0, 1)).to(device=index_device)


def candidate_weighted_grad_and_diag_fisher_proxies_from_captures(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    flat_indices: Union[Sequence[int], torch.Tensor],
    weight_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather candidate-local first-order credit and a diagonal-Fisher surrogate from one capture surface."""

    if not inputs or not grad_outputs:
        raise ValueError("inputs and grad_outputs must be non-empty")
    if len(inputs) < len(grad_outputs):
        raise ValueError("capture call-count mismatch")
    weight_dims = tuple(int(dim) for dim in weight_shape)
    if len(weight_dims) != 2:
        raise ValueError(f"weight_shape must be rank-2, got {weight_dims}")
    out_features, in_features = weight_dims
    if isinstance(flat_indices, torch.Tensor):
        flat_index_tensor = flat_indices.detach().flatten().to(dtype=torch.int64)
    else:
        flat_index_tensor = torch.tensor(
            [int(index) for index in flat_indices],
            dtype=torch.int64,
        )
    if flat_index_tensor.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32)
        return empty, empty.clone()
    paired_inputs = inputs[-len(grad_outputs):]
    reference_input = _as_bsi(
        paired_inputs[0].detach().to(torch.float32),
        name="input",
    )
    index_device = reference_input.device
    flat_index_tensor = flat_index_tensor.to(device=index_device)
    if torch.any(flat_index_tensor < 0).item() or torch.any(
        flat_index_tensor >= out_features * in_features
    ).item():
        raise ValueError("flat_indices must lie inside the flattened weight tensor")
    proxies = torch.zeros(
        (int(flat_index_tensor.numel()),),
        dtype=torch.float32,
        device=index_device,
    )
    diag_fisher = torch.zeros(
        (int(flat_index_tensor.numel()),),
        dtype=torch.float32,
        device=index_device,
    )
    chunk_size = int(PROXY_GATHER_FLAT_INDEX_CHUNK_SIZE)
    if chunk_size <= 0:
        raise ValueError("PROXY_GATHER_FLAT_INDEX_CHUNK_SIZE must be positive")
    for chunk_start in range(0, int(flat_index_tensor.numel()), chunk_size):
        chunk = flat_index_tensor[chunk_start : chunk_start + chunk_size]
        row_indices = torch.div(
            chunk,
            int(in_features),
            rounding_mode="floor",
        )
        col_indices = torch.remainder(chunk, int(in_features))
        _accumulate_weighted_grad_proxy_chunk_from_captures(
            paired_inputs,
            grad_outputs,
            row_indices=row_indices,
            col_indices=col_indices,
            index_device=index_device,
            proxies=proxies[chunk_start : chunk_start + int(chunk.numel())],
            diag_fisher=diag_fisher[chunk_start : chunk_start + int(chunk.numel())],
        )
    return proxies, diag_fisher


def candidate_weighted_grad_proxies_from_captures(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    flat_indices: Union[Sequence[int], torch.Tensor],
    weight_shape: Sequence[int],
) -> torch.Tensor:
    """Gather candidate-local first-order credit without reconstructing the full weight gradient."""

    proxies, _diag_fisher = candidate_weighted_grad_and_diag_fisher_proxies_from_captures(
        inputs,
        grad_outputs,
        flat_indices=flat_indices,
        weight_shape=weight_shape,
    )
    return proxies


def credit_from_weighted_grad(weighted_grad: torch.Tensor, *, scheme: str = "full_magnitude_ceiling") -> torch.Tensor:
    if scheme == "full_magnitude_ceiling":
        return -weighted_grad
    if scheme == "pow2_bucket":
        values = weighted_grad.to(torch.float32)
        out = torch.zeros_like(values)
        nonzero = values != 0
        if bool(nonzero.any().item()):
            abs_values = values[nonzero].abs()
            exponents = torch.log2(abs_values).round().clamp(-8.0, 8.0)
            out[nonzero] = values[nonzero].sign() * torch.pow(2.0, exponents)
        return -out
    raise ValueError(f"unsupported credit scheme {scheme!r}")


def sign_pressure_int16_votes(
    projected_moves: torch.Tensor,
    vote_spec: VoteUpdateSpec,
    *,
    inverted: bool = False,
) -> torch.Tensor:
    """Rank-free diagnostic votes: every nonzero move crosses by threshold_abs."""

    vote_spec.validate()
    if projected_moves.dtype != torch.int8:
        raise ValueError(f"projected_moves must be torch.int8, got {projected_moves.dtype}")
    threshold = int(vote_spec.threshold_abs)
    direction = -1 if bool(inverted) else 1
    flat_moves = projected_moves.detach().flatten().to(torch.int16)
    votes = torch.zeros_like(flat_moves, dtype=torch.int16)
    nonzero = flat_moves != 0
    votes[nonzero] = (flat_moves[nonzero] * int(direction) * threshold).to(torch.int16)
    return votes.view_as(projected_moves)


def _rank_bin_candidate_counts(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    spec: RankVoteSpec,
) -> tuple[int, list[int]]:
    """Count rank-bucket candidates per bin without materializing vote tensors."""

    spec.validate()
    if credit.shape != projected_moves.shape:
        raise ValueError("credit/projected_moves tensor shape mismatch")
    flat_credit = credit.detach().flatten().to(torch.float32)
    flat_moves = projected_moves.detach().flatten().to(torch.int8)
    candidate_idx = torch.nonzero(flat_moves != 0, as_tuple=False).flatten()
    candidate_count = int(candidate_idx.numel())
    bin_counts = [0] * len(spec.rank_bins)
    if candidate_count == 0:
        return candidate_count, bin_counts
    abs_values = flat_credit[candidate_idx].abs()
    if spec.rank_method == "grouped_bisect_right":
        rank_positions = _bisect_right_rank_positions_by_equal_value_group(abs_values)
        ranks = None
    else:
        sorted_abs = torch.sort(abs_values).values
        ranks = torch.searchsorted(sorted_abs, abs_values, right=True).to(torch.float32) / float(
            candidate_count
        )
        rank_positions = None
    for bin_index, item in enumerate(spec.rank_bins):
        if spec.rank_method == "grouped_bisect_right":
            assert rank_positions is not None
            lo_rank, hi_limit = _rank_bin_bounds(candidate_count, item)
            mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
        else:
            assert ranks is not None
            include_hi_mask = ranks <= float(item.hi_exclusive) if item.include_hi else torch.zeros_like(
                ranks,
                dtype=torch.bool,
            )
            mask = (ranks >= float(item.lo_inclusive)) & (
                (ranks < float(item.hi_exclusive)) | include_hi_mask
            )
        bin_counts[bin_index] = int(mask.sum().item())
    return candidate_count, bin_counts


def compact_pressure_shape_summary(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    spec: RankVoteSpec,
) -> dict[str, Any]:
    """Compact rank-bin occupancy mass fractions for pressure-shape agreement."""

    candidate_count, bin_counts = _rank_bin_candidate_counts(credit, projected_moves, spec)
    if candidate_count == 0:
        fractions = [0.0] * len(spec.rank_bins)
    else:
        fractions = [float(count) / float(candidate_count) for count in bin_counts]
    return {
        "schema": "hrm_text_158_pressure_shape_summary/v0",
        "rank_method": spec.rank_method,
        "rank_bins": [item.to_dict() for item in spec.rank_bins],
        "bin_occupancy_count": bin_counts,
        "bin_mass_fraction": fractions,
        "candidate_count": candidate_count,
        "raw_per_proposal_arrays_included": False,
    }


SIGNED_RANK_BIN_MASS_SCHEMA = "hrm_text_158_signed_rank_bin_mass/v0"


def _per_bin_signed_vote_mass(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    votes: torch.Tensor,
    spec: RankVoteSpec,
) -> tuple[list[float], list[float], float]:
    """Aggregate positive/negative vote mass per rank bin."""

    spec.validate()
    if credit.shape != projected_moves.shape or credit.shape != votes.shape:
        raise ValueError("credit/projected_moves/votes tensor shape mismatch")
    flat_credit = credit.detach().flatten().to(torch.float32)
    flat_moves = projected_moves.detach().flatten().to(torch.int8)
    flat_votes = votes.detach().flatten().to(torch.float32)
    candidate_idx = torch.nonzero(flat_moves != 0, as_tuple=False).flatten()
    pos_mass = [0.0] * len(spec.rank_bins)
    neg_mass = [0.0] * len(spec.rank_bins)
    if candidate_idx.numel() == 0:
        return pos_mass, neg_mass, 0.0
    abs_values = flat_credit[candidate_idx].abs()
    if spec.rank_method == "grouped_bisect_right":
        rank_positions = _bisect_right_rank_positions_by_equal_value_group(abs_values)
        ranks = None
    else:
        sorted_abs = torch.sort(abs_values).values
        ranks = torch.searchsorted(sorted_abs, abs_values, right=True).to(torch.float32) / float(
            candidate_idx.numel()
        )
        rank_positions = None
    vote_values = flat_votes[candidate_idx]
    for bin_index, item in enumerate(spec.rank_bins):
        if spec.rank_method == "grouped_bisect_right":
            assert rank_positions is not None
            lo_rank, hi_limit = _rank_bin_bounds(int(candidate_idx.numel()), item)
            mask = (rank_positions >= lo_rank) & (rank_positions < hi_limit)
        else:
            assert ranks is not None
            include_hi_mask = ranks <= float(item.hi_exclusive) if item.include_hi else torch.zeros_like(
                ranks,
                dtype=torch.bool,
            )
            mask = (ranks >= float(item.lo_inclusive)) & (
                (ranks < float(item.hi_exclusive)) | include_hi_mask
            )
        if not bool(mask.any().item()):
            continue
        bin_votes = vote_values[mask]
        pos_mass[bin_index] = float(bin_votes[bin_votes > 0.0].sum().item())
        neg_mass[bin_index] = float((-bin_votes[bin_votes < 0.0]).sum().item())
    total_abs = float(flat_votes.abs().sum().item())
    return pos_mass, neg_mass, total_abs


def compact_signed_rank_bin_mass_summary(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    spec: RankVoteSpec,
) -> dict[str, Any]:
    """Signed rank-bin mass on 2N pos/neg basis for branch-5 shadow arms."""

    votes = rank_bucketed_int16_votes(credit, projected_moves, spec)
    pos_mass, neg_mass, total_abs = _per_bin_signed_vote_mass(
        credit,
        projected_moves,
        votes,
        spec,
    )
    if total_abs <= 0.0:
        pos_fraction = [0.0] * len(spec.rank_bins)
        neg_fraction = [0.0] * len(spec.rank_bins)
        net_fraction = [0.0] * len(spec.rank_bins)
    else:
        pos_fraction = [float(value) / total_abs for value in pos_mass]
        neg_fraction = [float(value) / total_abs for value in neg_mass]
        net_fraction = [
            float(pos - neg) / total_abs for pos, neg in zip(pos_mass, neg_mass, strict=True)
        ]
    return {
        "schema": SIGNED_RANK_BIN_MASS_SCHEMA,
        "pos_bin_fraction": pos_fraction,
        "neg_bin_fraction": neg_fraction,
        "signed_bin_net_fraction": net_fraction,
        "total_abs_vote_mass": total_abs,
        "telemetry_only_net_fraction": True,
    }


def build_pressure_shape_summary_v1(
    credit: torch.Tensor,
    projected_moves: torch.Tensor,
    rank_spec: RankVoteSpec,
    *,
    a1_rank_spec: RankVoteSpec | None = None,
) -> dict[str, Any]:
    """Unsigned occupancy plus signed rank-bin mass and A1 counterfactual."""

    summary = compact_pressure_shape_summary(credit, projected_moves, rank_spec)
    summary["schema"] = "hrm_text_158_pressure_shape_summary/v1"
    primary_signed = compact_signed_rank_bin_mass_summary(credit, projected_moves, rank_spec)
    a1_spec = a1_rank_spec or rank_spec
    a1_signed = compact_signed_rank_bin_mass_summary(credit, projected_moves, a1_spec)
    summary["signed_rank_bin_mass"] = primary_signed
    summary["counterfactual_signed_rank_bin_mass"] = {
        "a1_order_matched": a1_signed,
        "order_matched_basis": "a1_emitted",
    }
    return summary


def compact_vote_pressure_summary(votes: torch.Tensor) -> dict[str, Any]:
    """Compact receipt metrics for vote pressure without raw per-proposal arrays."""

    flat = votes.detach().cpu().flatten().to(torch.int16)
    nonzero = flat[flat != 0]
    abs_values = nonzero.abs().to(torch.float32)
    summary: dict[str, Any] = {
        "vote_nonzero_count": int(nonzero.numel()),
        "vote_positive_count": int((nonzero > 0).sum().item()),
        "vote_negative_count": int((nonzero < 0).sum().item()),
        "raw_per_proposal_arrays_included": False,
    }
    if abs_values.numel() == 0:
        summary.update(
            {
                "vote_abs_min": 0,
                "vote_abs_median": 0.0,
                "vote_abs_max": 0,
            },
        )
        return summary
    summary.update(
        {
            "vote_abs_min": int(abs_values.min().item()),
            "vote_abs_median": float(abs_values.median().item()),
            "vote_abs_max": int(abs_values.max().item()),
        },
    )
    return summary


@dataclass(frozen=True)
class BoundedDeltaTensorState:
    state_key: str
    q_levels: torch.Tensor
    frozen_scale: torch.Tensor
    bounded_accumulator: BoundedDeltaAccumulatorState
    exact_accumulator_shadow: torch.Tensor | None
    bounded_accumulator_fresh_for_exact_shadow: bool = True
    bounded_accumulator_rebuild_hot_exact_indices: tuple[int, ...] | None = None
    bounded_accumulator_rebuild_cold_default_value: int | None = None

    def __post_init__(self) -> None:
        if not self.state_key:
            raise ValueError("state_key must be non-empty")
        if self.q_levels.dtype != torch.int8:
            raise ValueError(f"q_levels must be torch.int8, got {self.q_levels.dtype}")
        if self.frozen_scale.numel() != 1 or not self.frozen_scale.dtype.is_floating_point:
            raise ValueError("frozen_scale must be a floating scalar tensor")
        if tuple(self.bounded_accumulator.logical_shape) != tuple(self.q_levels.shape):
            raise ValueError("bounded accumulator shape must match q_levels")
        if self.exact_accumulator_shadow is not None:
            if self.exact_accumulator_shadow.dtype != torch.int16:
                raise ValueError(
                    "exact_accumulator_shadow must be torch.int16, got "
                    f"{self.exact_accumulator_shadow.dtype}"
                )
            if self.q_levels.shape != self.exact_accumulator_shadow.shape:
                raise ValueError("q_levels and exact_accumulator_shadow shapes must match")
        elif self.bounded_accumulator_fresh_for_exact_shadow:
            raise ValueError(
                "bounded_accumulator_fresh_for_exact_shadow cannot be true when no "
                "exact_accumulator_shadow is present"
            )

    def rebuild_hot_exact_indices(self) -> tuple[int, ...]:
        if self.bounded_accumulator_rebuild_hot_exact_indices is not None:
            return tuple(int(idx) for idx in self.bounded_accumulator_rebuild_hot_exact_indices)
        return tuple(int(idx) for idx in self.bounded_accumulator.hot_exact_indices)

    def rebuild_cold_default_value(self) -> int:
        if self.bounded_accumulator_rebuild_cold_default_value is not None:
            return int(self.bounded_accumulator_rebuild_cold_default_value)
        return int(self.bounded_accumulator.cold_default_value)

    def with_fresh_bounded_accumulator(self) -> BoundedDeltaTensorState:
        if self.exact_accumulator_shadow is None:
            raise ValueError("cannot rebuild bounded accumulator without exact_accumulator_shadow")
        return make_bounded_tensor_state(
            self.state_key,
            self.q_levels,
            self.frozen_scale,
            self.exact_accumulator_shadow,
            hot_exact_indices=self.rebuild_hot_exact_indices(),
            cold_default_value=self.rebuild_cold_default_value(),
        )

    def _fresh_state_for_bounded_parity(self) -> tuple[BoundedDeltaTensorState, bool]:
        if self.exact_accumulator_shadow is None:
            raise ValueError("bounded parity requires exact_accumulator_shadow")
        if self.bounded_accumulator_fresh_for_exact_shadow:
            return self, False
        return self.with_fresh_bounded_accumulator(), True

    def decoded_accumulators(
        self,
        *,
        device: torch.device | str | None = None,
        rebuild_if_stale: bool = False,
    ) -> torch.Tensor:
        if self.exact_accumulator_shadow is None:
            out = decode_bounded_accumulator_to_i16(self.bounded_accumulator)
            return out.to(device=device) if device is not None else out
        state = self
        if not self.bounded_accumulator_fresh_for_exact_shadow:
            if not rebuild_if_stale:
                raise ValueError(
                    "bounded accumulator is stale for exact_accumulator_shadow; "
                    "request rebuild_if_stale=True or use an explicit parity/checkpoint path",
                )
            state = self.with_fresh_bounded_accumulator()
        out = decode_bounded_accumulator_to_i16(state.bounded_accumulator)
        return out.to(device=device) if device is not None else out

    def bounded_decode_parity_report(self, *, fail_on_mismatch: bool = False) -> dict[str, Any]:
        parity_state, rebuilt = self._fresh_state_for_bounded_parity()
        decoded = decode_bounded_accumulator_to_i16(parity_state.bounded_accumulator)
        decoded_sha = tensor_sha256(decoded)
        shadow_sha = tensor_sha256(parity_state.exact_accumulator_shadow)
        matches = decoded_sha == shadow_sha
        if fail_on_mismatch and not matches:
            raise ValueError(
                f"bounded accumulator decode does not match exact shadow for {self.state_key}"
            )
        return {
            "bounded_decode_parity_checked": True,
            "bounded_accumulator_fresh_for_exact_shadow": True,
            "bounded_accumulator_rebuilt_for_parity": rebuilt,
            "bounded_accumulator_decoded_sha256": decoded_sha,
            "exact_accumulator_shadow_sha256": shadow_sha,
            "exact_shadow_matches_bounded_decode": matches,
        }

    def vote_update_state(self, *, device: torch.device | str | None = None) -> VoteUpdateState:
        if self.exact_accumulator_shadow is None:
            raise ValueError(
                "dense oracle/control vote_update_state is unavailable for a bounded-only "
                "candidate-authority tensor state"
            )
        accumulators = (
            self.exact_accumulator_shadow.to(device=device).contiguous()
            if device is not None else self.exact_accumulator_shadow.contiguous()
        )
        return VoteUpdateState(
            q_levels=self.q_levels.to(device=device).contiguous() if device is not None else self.q_levels,
            accumulators=accumulators,
        )

    def materialized_weight(
        self,
        *,
        device: torch.device | str = "cpu",
        requires_grad: bool,
    ) -> torch.Tensor:
        q = self.q_levels.to(device=device, dtype=torch.float32)
        scale = self.frozen_scale.to(device=device, dtype=torch.float32)
        weight = (q * scale).detach().clone()
        weight.requires_grad_(requires_grad)
        return weight

    def to_schema_dict(self, *, parity_check: bool = True) -> dict[str, Any]:
        summary_state = self
        rebuilt = False
        if parity_check and self.exact_accumulator_shadow is not None:
            summary_state, rebuilt = self._fresh_state_for_bounded_parity()
        out = {
            "state_key": summary_state.state_key,
            "shape": list(summary_state.q_levels.shape),
            "q_dtype": str(summary_state.q_levels.dtype),
            "q_sha256": tensor_sha256(summary_state.q_levels),
            "q_codec": "int8_levels_transitional_base3_pack_ready",
            "frozen_scale_dtype": str(summary_state.frozen_scale.dtype),
            "frozen_scale_value": float(summary_state.frozen_scale.detach().cpu().item()),
            "frozen_scale_law": "per_tensor_absmean_frozen_from_parent_qscale",
            "bounded_accumulator_schema": BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
            "bounded_accumulator": summary_state.bounded_accumulator.to_dict(),
            "bounded_accumulator_fresh_for_exact_shadow": bool(
                summary_state.bounded_accumulator_fresh_for_exact_shadow
            ),
            "bounded_accumulator_rebuilt_for_parity": bool(rebuilt),
            "live_authoritative_state_source": (
                "q_levels_plus_exact_accumulator_shadow"
                if summary_state.exact_accumulator_shadow is not None
                else "q_levels_plus_bounded_accumulator"
            ),
            "bounded_accumulator_authority": (
                (
                    "fresh_checkpoint_export_or_explicit_parity"
                    if summary_state.bounded_accumulator_fresh_for_exact_shadow
                    else "stale_optional_not_live_authority"
                )
                if summary_state.exact_accumulator_shadow is not None
                else "direct_candidate_authority"
            ),
            "exact_accumulator_shadow_available": bool(
                summary_state.exact_accumulator_shadow is not None
            ),
            "exact_accumulator_shadow_sha256": (
                tensor_sha256(summary_state.exact_accumulator_shadow)
                if summary_state.exact_accumulator_shadow is not None
                else None
            ),
            "bounded_decode_parity_checked": bool(
                parity_check and summary_state.exact_accumulator_shadow is not None
            ),
        }
        if parity_check and summary_state.exact_accumulator_shadow is not None:
            parity = summary_state.bounded_decode_parity_report(fail_on_mismatch=True)
            parity["bounded_accumulator_rebuilt_for_parity"] = bool(rebuilt)
            out.update(parity)
        return out


def _cold_exception_indices_for_exact_preservation(
    acc: torch.Tensor,
    *,
    hot_exact_indices: Sequence[int],
    cold_default_value: int,
) -> tuple[int, ...]:
    flat = acc.detach().cpu().flatten().to(torch.int16)
    hot = {int(idx) for idx in hot_exact_indices}
    return tuple(
        int(idx)
        for idx, value in enumerate(flat.tolist())
        if idx not in hot and int(value) != int(cold_default_value)
    )


def make_bounded_tensor_state(
    state_key: str,
    q_levels: torch.Tensor,
    frozen_scale: torch.Tensor | float,
    accumulators: torch.Tensor | None = None,
    *,
    hot_exact_indices: Sequence[int] = (),
    cold_default_value: int = 0,
) -> BoundedDeltaTensorState:
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must be torch.int8, got {q_levels.dtype}")
    q = q_levels.detach().cpu().contiguous()
    acc = (
        torch.zeros_like(q, dtype=torch.int16)
        if accumulators is None
        else accumulators.detach().cpu().to(torch.int16).contiguous()
    )
    if q.shape != acc.shape:
        raise ValueError("q_levels and accumulators must have identical shapes")
    scale = (
        torch.tensor(float(frozen_scale), dtype=torch.float32)
        if not isinstance(frozen_scale, torch.Tensor)
        else frozen_scale.detach().cpu().to(torch.float32).reshape(())
    )
    hot = tuple(int(idx) for idx in hot_exact_indices)
    cold_ex = _cold_exception_indices_for_exact_preservation(
        acc,
        hot_exact_indices=hot,
        cold_default_value=int(cold_default_value),
    )
    bounded = encode_budget_capped_hybrid_reference(
        VoteUpdateState(q_levels=q, accumulators=acc),
        hot_exact_indices=hot,
        cold_default_value=int(cold_default_value),
        cold_exception_indices=cold_ex,
    )
    return BoundedDeltaTensorState(
        state_key=state_key,
        q_levels=q,
        frozen_scale=scale,
        bounded_accumulator=bounded,
        exact_accumulator_shadow=acc,
    )


def make_live_shadow_tensor_state(
    prior_state: BoundedDeltaTensorState,
    q_levels: torch.Tensor,
    accumulators: torch.Tensor,
    *,
    hot_exact_indices: Sequence[int] | None = None,
    cold_default_value: int | None = None,
) -> BoundedDeltaTensorState:
    """Build the next live state without rebuilding the bounded accumulator."""

    q = q_levels.detach().cpu().to(torch.int8).contiguous()
    acc = accumulators.detach().cpu().to(torch.int16).contiguous()
    if q.shape != acc.shape:
        raise ValueError("q_levels and accumulators must have identical shapes")
    if tuple(prior_state.bounded_accumulator.logical_shape) != tuple(q.shape):
        raise ValueError("live shadow update cannot change bounded accumulator logical shape")
    scale = prior_state.frozen_scale.detach().cpu().to(torch.float32).reshape(())
    rebuild_hot = (
        tuple(int(idx) for idx in hot_exact_indices)
        if hot_exact_indices is not None
        else prior_state.rebuild_hot_exact_indices()
    )
    rebuild_default = (
        int(cold_default_value)
        if cold_default_value is not None
        else prior_state.rebuild_cold_default_value()
    )
    return BoundedDeltaTensorState(
        state_key=prior_state.state_key,
        q_levels=q,
        frozen_scale=scale,
        bounded_accumulator=prior_state.bounded_accumulator,
        exact_accumulator_shadow=acc,
        bounded_accumulator_fresh_for_exact_shadow=False,
        bounded_accumulator_rebuild_hot_exact_indices=rebuild_hot,
        bounded_accumulator_rebuild_cold_default_value=rebuild_default,
    )


def make_candidate_authority_tensor_state(
    prior_state: BoundedDeltaTensorState,
    q_levels: torch.Tensor,
    bounded_accumulator: BoundedDeltaAccumulatorState,
) -> BoundedDeltaTensorState:
    q = q_levels.detach().cpu().to(torch.int8).contiguous()
    if tuple(bounded_accumulator.logical_shape) != tuple(q.shape):
        raise ValueError("candidate bounded accumulator logical shape must match q_levels")
    scale = prior_state.frozen_scale.detach().cpu().to(torch.float32).reshape(())
    return BoundedDeltaTensorState(
        state_key=prior_state.state_key,
        q_levels=q,
        frozen_scale=scale,
        bounded_accumulator=bounded_accumulator,
        exact_accumulator_shadow=None,
        bounded_accumulator_fresh_for_exact_shadow=False,
        bounded_accumulator_rebuild_hot_exact_indices=tuple(
            int(idx) for idx in bounded_accumulator.hot_exact_indices
        ),
        bounded_accumulator_rebuild_cold_default_value=int(bounded_accumulator.cold_default_value),
    )


def ternarize_weight_to_q_scale(weight: torch.Tensor, *, scale_eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.detach().abs().mean().clamp(min=float(scale_eps)).to(torch.float32)
    q = (weight.detach().to(torch.float32) / scale).round().clamp(-1.0, 1.0).to(torch.int8)
    return q.cpu().contiguous(), scale.cpu().reshape(())


def derive_bounded_tensor_state_from_weight(
    state_key: str,
    weight: torch.Tensor,
    *,
    hot_exact_indices: Sequence[int] = (),
    cold_default_value: int = 0,
    scale_eps: float = 1e-5,
) -> BoundedDeltaTensorState:
    q, scale = ternarize_weight_to_q_scale(weight, scale_eps=scale_eps)
    return make_bounded_tensor_state(
        state_key,
        q,
        scale,
        hot_exact_indices=hot_exact_indices,
        cold_default_value=cold_default_value,
    )


@dataclass(frozen=True)
class AuthoritativeForwardHandle:
    current_weights: dict[str, torch.Tensor]
    captures: dict[str, dict[str, list[torch.Tensor]]]
    capture_enabled: bool
    capture_device_mode: str

    def weighted_grad(self, state_key: str) -> torch.Tensor:
        if state_key not in self.current_weights:
            raise KeyError(state_key)
        if not self.capture_enabled:
            raise RuntimeError(
                "weighted_grad is unavailable because "
                "authoritative_forward_context capture is disabled; use "
                "requires_grad=True when weighted gradients are needed"
            )
        capture = self.captures[state_key]
        if not capture["inputs"] or not capture["grad_outputs"]:
            raise RuntimeError(
                "weighted_grad requires captured inputs and grad_outputs; "
                "ensure the eligible module was invoked under a differentiable "
                "authoritative_forward_context"
            )
        return weighted_grad_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            weight_shape=tuple(self.current_weights[state_key].shape),
        )

    def candidate_weighted_grad_proxies(
        self,
        state_key: str,
        flat_indices: Union[Sequence[int], torch.Tensor],
    ) -> torch.Tensor:
        if state_key not in self.current_weights:
            raise KeyError(state_key)
        if not self.capture_enabled:
            raise RuntimeError(
                "candidate_weighted_grad_proxies is unavailable because "
                "authoritative_forward_context capture is disabled; use "
                "requires_grad=True when candidate-local weighted gradients are needed"
            )
        capture = self.captures[state_key]
        if not capture["inputs"] or not capture["grad_outputs"]:
            raise RuntimeError(
                "candidate_weighted_grad_proxies requires captured inputs and grad_outputs; "
                "ensure the eligible module was invoked under a differentiable "
                "authoritative_forward_context"
            )
        return candidate_weighted_grad_proxies_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            flat_indices=flat_indices,
            weight_shape=tuple(self.current_weights[state_key].shape),
        )

    def candidate_weighted_grad_and_diag_fisher_proxies(
        self,
        state_key: str,
        flat_indices: Union[Sequence[int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state_key not in self.current_weights:
            raise KeyError(state_key)
        if not self.capture_enabled:
            raise RuntimeError(
                "candidate_weighted_grad_and_diag_fisher_proxies is unavailable because "
                "authoritative_forward_context capture is disabled; use "
                "requires_grad=True when candidate-local credit signals are needed"
            )
        capture = self.captures[state_key]
        if not capture["inputs"] or not capture["grad_outputs"]:
            raise RuntimeError(
                "candidate_weighted_grad_and_diag_fisher_proxies requires captured inputs "
                "and grad_outputs; ensure the eligible module was invoked under a "
                "differentiable authoritative_forward_context"
            )
        return candidate_weighted_grad_and_diag_fisher_proxies_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            flat_indices=flat_indices,
            weight_shape=tuple(self.current_weights[state_key].shape),
        )


@contextmanager
def authoritative_forward_context(
    eligible_modules: Mapping[str, Any],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool,
    capture_device_mode: str = AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY,
) -> Any:
    missing = set(eligible_modules) - set(tensor_states)
    if missing:
        raise ValueError(f"missing tensor state for eligible modules: {sorted(missing)}")
    if capture_device_mode not in {
        AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY,
        AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT,
    }:
        raise ValueError(
            "capture_device_mode must be one of "
            f"{AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY!r}, "
            f"{AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT!r}"
        )
    originals = {state_key: module.forward for state_key, module in eligible_modules.items()}
    capture_enabled = bool(requires_grad)
    current_weights = {
        state_key: tensor_states[state_key].materialized_weight(
            device=device,
            requires_grad=requires_grad,
        )
        for state_key in eligible_modules
    }
    captures = {
        state_key: {"inputs": [], "grad_outputs": []}
        for state_key in eligible_modules
    }

    for state_key, module in eligible_modules.items():
        def _forward(self: Any, input: torch.Tensor, *, key: str = state_key) -> torch.Tensor:
            if capture_enabled:
                captured_input = input.detach()
                if capture_device_mode == AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY:
                    captured_input = captured_input.cpu()
                captures[key]["inputs"].append(captured_input)
            out = F.linear(input, current_weights[key], self.bias)
            if capture_enabled and out.requires_grad:
                out.register_hook(
                    lambda grad, capture_key=key: captures[capture_key]["grad_outputs"].append(
                        grad.detach().cpu()
                        if capture_device_mode == AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY
                        else grad.detach(),
                    )
                )
            return out

        module.forward = types.MethodType(_forward, module)
    try:
        yield AuthoritativeForwardHandle(
            current_weights=current_weights,
            captures=captures,
            capture_enabled=capture_enabled,
            capture_device_mode=capture_device_mode,
        )
    finally:
        for state_key, module in eligible_modules.items():
            module.forward = originals[state_key]


def build_optimizer_excluding_eligible_masters(
    model: Any,
    eligible_modules: Mapping[str, Any],
    *,
    lr: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[torch.optim.Optimizer | None, dict[str, Any]]:
    eligible_param_ids = set()
    for module in eligible_modules.values():
        module.weight.requires_grad_(True)
        eligible_param_ids.add(id(module.weight))
        if getattr(module, "bias", None) is not None:
            module.bias.requires_grad_(True)
            eligible_param_ids.add(id(module.bias))
    noneligible_params = [p for p in model.parameters() if id(p) not in eligible_param_ids]
    opt = (
        torch.optim.AdamW(noneligible_params, lr=float(lr), betas=(0.9, 0.95), weight_decay=float(weight_decay))
        if noneligible_params
        else None
    )
    opt_param_ids = (
        {id(p) for group in opt.param_groups for p in group["params"]}
        if opt is not None
        else set()
    )
    overlap = sorted(eligible_param_ids & opt_param_ids)
    eligible_with_state = []
    if opt is not None:
        for state_key, module in eligible_modules.items():
            if module.weight in opt.state:
                eligible_with_state.append(state_key)
    checks = {
        "eligible_param_count": len(eligible_param_ids),
        "optimizer_param_count": len(opt_param_ids),
        "eligible_params_in_optimizer": len(overlap),
        "eligible_optimizer_state_entries": len(eligible_with_state),
        "eligible_weight_requires_grad_for_transient_credit_capture": all(
            module.weight.requires_grad for module in eligible_modules.values()
        ),
        "optimizer_state_entries_total": len(opt.state) if opt is not None else 0,
        "optimizer_created": opt is not None,
        "pass": len(overlap) == 0 and not eligible_with_state,
    }
    return opt, checks


def snapshot_eligible_master_sha256(eligible_modules: Mapping[str, Any]) -> dict[str, str]:
    return {
        state_key: tensor_sha256(module.weight.detach())
        for state_key, module in eligible_modules.items()
    }


def prove_eligible_master_identity_after_optimizer_step(
    optimizer: torch.optim.Optimizer | None,
    eligible_modules: Mapping[str, Any],
    *,
    optimizer_checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = snapshot_eligible_master_sha256(eligible_modules)
    if optimizer is not None:
        optimizer.step()
    after = snapshot_eligible_master_sha256(eligible_modules)
    identity = before == after
    return {
        "schema": "hrm_text_158_c2p0_fp_master_identity_snapshot/v0",
        "eligible_master_sha256_before": before,
        "eligible_master_sha256_after": after,
        "eligible_master_identity_pass": identity,
        "optimizer_step_called": optimizer is not None,
        "optimizer_checks": dict(optimizer_checks or {}),
        "pass": bool(identity and (optimizer_checks or {}).get("pass", True)),
    }


@dataclass(frozen=True)
class BoundedDeltaLearnerStepResult:
    tensor_states: dict[str, BoundedDeltaTensorState]
    tensor_stats: dict[str, dict[str, Any]]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]
    global_summary: dict[str, Any]

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "schema": "hrm_text_158_c2p0_bounded_delta_step_result/v0.compact",
            "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
            "tensor_stats": self.tensor_stats,
            "deferred_backlog_entry_count": sum(len(v) for v in self.deferred_backlog.values()),
            "global_summary": self.global_summary,
            "tensor_state_key_count": len(self.tensor_states),
            "tensor_state_keys": sorted(self.tensor_states),
            "tensor_state_summaries_included": False,
        }

    def to_dict(self, *, parity_check: bool = True) -> dict[str, Any]:
        return {
            "schema": "hrm_text_158_c2p0_bounded_delta_step_result/v0",
            "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
            "tensor_stats": self.tensor_stats,
            "deferred_backlog_entry_count": sum(len(v) for v in self.deferred_backlog.values()),
            "global_summary": self.global_summary,
            "tensor_state_summaries_included": True,
            "tensor_state_summaries": {
                key: state.to_schema_dict(parity_check=parity_check)
                for key, state in sorted(self.tensor_states.items())
            },
        }


def _votes_sha(votes: torch.Tensor) -> str:
    return tensor_sha256(votes.to(torch.int16).contiguous())


def _two_tier_enabled_pin_sha256() -> str:
    pin_tuple = {
        "enabled": True,
        "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
        "threshold_abs": int(CROSSING_THRESHOLD_ABS),
        "seam_validator_module": "two_tier_transient_selection",
    }
    return _sha256_bytes(_canonical_json(pin_tuple).encode("utf-8"))


def _require_local_loss_delta_tensor(
    name: str,
    state_key: str,
    value: torch.Tensor,
    *,
    shape: torch.Size,
) -> torch.Tensor:
    if value.dtype != torch.float32:
        raise ValueError(f"{name}[{state_key!r}] local_loss_delta_bad_dtype: expected float32, got {value.dtype}")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name}[{state_key!r}] local_loss_delta_shape_mismatch: "
            f"got {tuple(value.shape)} expected {tuple(shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name}[{state_key!r}] local_loss_delta_non_finite")
    return value.detach().cpu().contiguous()


def _validate_optional_vote_map_keys(
    name: str,
    values_by_key: Mapping[str, torch.Tensor] | None,
    expected_keys: set[str],
) -> None:
    if values_by_key is None:
        return
    actual_keys = set(values_by_key)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{name} keys must match tensor_states exactly: "
            f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}"
        )


def _validate_candidate_sparse_vote_map_keys(
    name: str,
    values_by_key: Mapping[str, Mapping[int, int]] | None,
    expected_keys: set[str],
) -> None:
    if values_by_key is None:
        return
    actual_keys = set(values_by_key)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{name} keys must match tensor_states exactly: "
            f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}"
        )


def _coerce_optional_vote_map_tensor(
    name: str,
    values_by_key: Mapping[str, torch.Tensor] | None,
    state_key: str,
    *,
    dtype: torch.dtype,
    shape: torch.Size,
) -> torch.Tensor | None:
    if values_by_key is None:
        return None
    value = values_by_key[state_key]
    if value.dtype != dtype:
        raise ValueError(f"{name}[{state_key!r}] must be {dtype}, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(
            f"{name}[{state_key!r}] shape must match votes shape; "
            f"got {tuple(value.shape)} expected {tuple(shape)}"
        )
    return value.detach().cpu().contiguous()


def _coerce_candidate_sparse_vote_events(
    values_by_key: Mapping[str, Mapping[int, int]] | None,
    state_key: str,
) -> dict[int, int]:
    if values_by_key is None:
        raise ValueError(
            "candidate sparse vote events are required for the direct bounded "
            "candidate path; dense vote authority is unsupported there"
        )
    raw = values_by_key[state_key]
    out: dict[int, int] = {}
    for raw_index, raw_vote in raw.items():
        index = int(raw_index)
        vote = int(raw_vote)
        if vote != 0:
            out[index] = vote
    return out


def _clone_vote_update_state_for_front_c(state: VoteUpdateState) -> VoteUpdateState:
    return VoteUpdateState(
        q_levels=state.q_levels.detach().cpu().clone().contiguous(),
        accumulators=state.accumulators.detach().cpu().clone().contiguous(),
        q_format=state.q_format,
        accumulator_format=state.accumulator_format,
    )


def _clone_vote_update_inputs_for_front_c(inputs: VoteUpdateInputs) -> VoteUpdateInputs:
    def clone_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.detach().cpu().clone().contiguous()

    return VoteUpdateInputs(
        votes=inputs.votes.detach().cpu().clone().contiguous(),
        replay_ce_veto_votes=clone_optional(inputs.replay_ce_veto_votes),
        replay_ce_veto_moves=clone_optional(inputs.replay_ce_veto_moves),
        pc_aux_votes=clone_optional(inputs.pc_aux_votes),
        pc_aux_moves=clone_optional(inputs.pc_aux_moves),
        pc_aux_mode=inputs.pc_aux_mode,
        vote_format=inputs.vote_format,
        local_loss_delta=clone_optional(inputs.local_loss_delta),
    )


def _clone_vote_update_spec_for_front_c(spec: VoteUpdateSpec) -> VoteUpdateSpec:
    return VoteUpdateSpec(**asdict(spec))


def _clone_vote_update_plan_for_front_c(plan: VoteUpdatePlan) -> VoteUpdatePlan:
    return VoteUpdatePlan(
        q_i16=plan.q_i16.detach().cpu().clone().contiguous(),
        new_acc_i32=plan.new_acc_i32.detach().cpu().clone().contiguous(),
        candidate_indices=plan.candidate_indices.detach().cpu().clone().contiguous(),
        pre_veto_selected_indices=plan.pre_veto_selected_indices.detach().cpu().clone().contiguous(),
        applied_indices=plan.applied_indices.detach().cpu().clone().contiguous(),
        applied_directions=plan.applied_directions.detach().cpu().clone().contiguous(),
        applied_thresholds=plan.applied_thresholds.detach().cpu().clone().contiguous(),
        replay_ce_veto_indices=plan.replay_ce_veto_indices.detach().cpu().clone().contiguous(),
        replay_veto_directions=plan.replay_veto_directions.detach().cpu().clone().contiguous(),
        replay_veto_thresholds=plan.replay_veto_thresholds.detach().cpu().clone().contiguous(),
        pc_aux_negative_indices=plan.pc_aux_negative_indices.detach().cpu().clone().contiguous(),
        pc_aux_veto_indices=plan.pc_aux_veto_indices.detach().cpu().clone().contiguous(),
        stats=dict(plan.stats),
    )


def _clone_q_acc_result_for_front_c(
    q_levels: torch.Tensor,
    accumulators: torch.Tensor,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "q_levels": q_levels.detach().cpu().clone().contiguous(),
        "accumulators": accumulators.detach().cpu().clone().contiguous(),
        "stats": dict(stats),
    }


def _clone_backlog_for_front_c(
    backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> dict[str, dict[int, dict[str, int]]]:
    return {
        str(state_key): {
            int(flat_index): {
                str(field): int(value)
                for field, value in dict(entry).items()
            }
            for flat_index, entry in dict(by_index).items()
        }
        for state_key, by_index in dict(backlog or {}).items()
    }


def _front_c_cloned_observation(
    *,
    vote_update_states: Mapping[str, VoteUpdateState],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    plans_by_key: Mapping[str, VoteUpdatePlan],
    q_acc_by_key: Mapping[str, tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    global_cap_used: bool,
) -> dict[str, Any]:
    return {
        "schema": FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION,
        "global_cap_used": bool(global_cap_used),
        "states_by_key": {
            key: _clone_vote_update_state_for_front_c(state)
            for key, state in sorted(vote_update_states.items())
        },
        "inputs_by_key": {
            key: _clone_vote_update_inputs_for_front_c(inputs)
            for key, inputs in sorted(inputs_by_key.items())
        },
        "specs_by_key": {
            key: _clone_vote_update_spec_for_front_c(vote_specs_by_key[key])
            for key in sorted(vote_specs_by_key)
        },
        "plans_by_key": {
            key: _clone_vote_update_plan_for_front_c(plan)
            for key, plan in sorted(plans_by_key.items())
        },
        "q_acc_by_key": {
            key: _clone_q_acc_result_for_front_c(q_levels, accumulators, stats)
            for key, (q_levels, accumulators, stats) in sorted(q_acc_by_key.items())
        },
        "deferred_backlog": _clone_backlog_for_front_c(deferred_backlog),
        "live_mutation_inputs_exposed": False,
    }


def apply_bounded_delta_vote_step(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    *,
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None = None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None = None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None = None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None = None,
    pc_aux_mode: str = "telemetry",
    global_cap_spec: GlobalRateCapSpec | None = None,
    global_cap_tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    global_cap_contract_name: str | None = None,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    hot_exact_indices_by_key: Mapping[str, Sequence[int]] | None = None,
    cold_default_value: int | None = None,
    parity_check: bool = False,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None = None,
    two_tier_carry_w6_enabled: bool = False,
    front_c_identity_observer: Callable[[Mapping[str, Any]], object] | None = None,
    candidate_mode: str | None = None,
    candidate_sparse_vote_events_by_key: Mapping[str, Mapping[int, int]] | None = None,
    candidate_oracle_control_enabled: bool = True,
    local_selection_ordering_mode: str = LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    local_selection_ordering_seed: int = 0,
    local_selection_ordering_step: int = 0,
) -> BoundedDeltaLearnerStepResult:
    if set(tensor_states) != set(votes_by_key) or set(tensor_states) != set(vote_specs_by_key):
        raise ValueError("tensor_states, votes_by_key, and vote_specs_by_key must have identical keys")
    expected_keys = set(tensor_states)
    _validate_optional_vote_map_keys("replay_ce_veto_votes_by_key", replay_ce_veto_votes_by_key, expected_keys)
    _validate_optional_vote_map_keys("replay_ce_veto_moves_by_key", replay_ce_veto_moves_by_key, expected_keys)
    _validate_optional_vote_map_keys("pc_aux_votes_by_key", pc_aux_votes_by_key, expected_keys)
    _validate_optional_vote_map_keys("pc_aux_moves_by_key", pc_aux_moves_by_key, expected_keys)
    if two_tier_carry_w6_enabled:
        if str(local_selection_ordering_mode) != LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA:
            raise ValueError(
                "two_tier_carry_w6_enabled requires "
                f"local_selection_ordering_mode={LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA!r}, "
                f"got {local_selection_ordering_mode!r}"
            )
        if local_loss_delta_by_key is None:
            raise ValueError("local_loss_delta_by_key_required_when_two_tier_enabled")
        _validate_optional_vote_map_keys(
            "local_loss_delta_by_key",
            local_loss_delta_by_key,
            expected_keys,
        )
    _validate_candidate_sparse_vote_map_keys(
        "candidate_sparse_vote_events_by_key",
        candidate_sparse_vote_events_by_key,
        expected_keys,
    )
    if global_cap_spec is None and global_cap_tie_rule_mode != EXACT_GLOBAL_CAP_TIE_RULE_MODE:
        raise ValueError(
            "global_cap_tie_rule_mode requires an active global_cap_spec; "
            "non-global paths must stay exact_global_cap"
        )
    if candidate_mode is not None:
        if candidate_mode != ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE:
            raise ValueError(f"unsupported candidate_mode {candidate_mode!r}")
        if str(local_selection_ordering_mode) != LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX:
            raise ValueError("candidate_mode local vote-update proof does not cover alternate local ordering")
        if front_c_identity_observer is not None:
            raise ValueError("candidate_mode does not cover front_c live identity observation")
        if global_cap_spec is not None:
            raise ValueError("candidate_mode local vote-update proof does not cover global cap")
        if deferred_backlog is not None:
            raise ValueError("candidate_mode local vote-update proof does not cover deferred backlog")
        if (
            replay_ce_veto_votes_by_key is not None
            or replay_ce_veto_moves_by_key is not None
            or pc_aux_votes_by_key is not None
            or pc_aux_moves_by_key is not None
        ):
            raise ValueError("candidate_mode local vote-update proof does not cover replay/pc auxiliary paths")

        next_states: dict[str, BoundedDeltaTensorState] = {}
        tensor_stats: dict[str, dict[str, Any]] = {}
        proof_by_key: dict[str, dict[str, Any]] = {}
        for state_key, prior_state in sorted(tensor_states.items()):
            candidate_result = execute_direct_bounded_local_vote_update_candidate(
                state_key=state_key,
                q_levels=prior_state.q_levels,
                bounded_accumulator=prior_state.bounded_accumulator,
                sparse_vote_events=_coerce_candidate_sparse_vote_events(
                    candidate_sparse_vote_events_by_key,
                    state_key,
                ),
                vote_spec=vote_specs_by_key[state_key],
            )
            next_state = make_candidate_authority_tensor_state(
                prior_state,
                candidate_result.next_q_levels,
                candidate_result.next_bounded_accumulator,
            )
            proof = dict(candidate_result.proof)
            proof["candidate_dense_decode_used"] = False
            proof["candidate_accumulator_transient_over2_used"] = False
            proof["candidate_vote_transient_over2_used"] = False
            proof["candidate_dense_vote_authority_used"] = False
            proof["dense_oracle_control_used"] = False
            proof["oracle_dense_vote_sha256"] = None
            proof["oracle_q_sha256_after"] = None
            proof["oracle_acc_sha256_after"] = None
            proof["oracle_applied_row_identities_sha256"] = None
            proof["oracle_residual_after_threshold_sha256"] = None
            proof["parity_pass"] = None
            if candidate_oracle_control_enabled:
                dense_votes = votes_by_key[state_key].detach().cpu().to(torch.int16).contiguous()
                oracle_result = apply_integer_vote_update_reference(
                    prior_state.vote_update_state(),
                    VoteUpdateInputs(votes=dense_votes),
                    vote_specs_by_key[state_key],
                    local_selection_ordering_mode=str(local_selection_ordering_mode),
                    local_selection_ordering_seed=int(local_selection_ordering_seed),
                    local_selection_ordering_step=int(local_selection_ordering_step),
                )
                candidate_decode = decode_bounded_accumulator_to_i16(next_state.bounded_accumulator)
                oracle_applied = tuple(
                    int(index)
                    for index in oracle_result.plan.applied_indices.detach().cpu().to(torch.int64).tolist()
                )
                oracle_residuals = {
                    int(index): int(oracle_result.accumulators.flatten()[int(index)].item())
                    for index in oracle_applied
                }
                oracle_applied_hash = hashlib.sha256()
                for index in oracle_applied:
                    oracle_applied_hash.update(state_key.encode("utf-8"))
                    oracle_applied_hash.update(b":")
                    oracle_applied_hash.update(str(int(index)).encode("utf-8"))
                    oracle_applied_hash.update(b"\n")
                oracle_applied_sha = oracle_applied_hash.hexdigest()
                oracle_residual_hash = hashlib.sha256()
                for index, value in sorted(oracle_residuals.items()):
                    oracle_residual_hash.update(state_key.encode("utf-8"))
                    oracle_residual_hash.update(b":")
                    oracle_residual_hash.update(str(int(index)).encode("utf-8"))
                    oracle_residual_hash.update(b"=")
                    oracle_residual_hash.update(str(int(value)).encode("utf-8"))
                    oracle_residual_hash.update(b"\n")
                oracle_residual_sha = oracle_residual_hash.hexdigest()
                candidate_applied_sha = proof.get("applied_row_identities_sha256")
                candidate_residual_sha = proof.get("residual_after_threshold_sha256")
                parity_pass = (
                    tensor_sha256(next_state.q_levels) == tensor_sha256(oracle_result.q_levels)
                    and tensor_sha256(candidate_decode) == tensor_sha256(oracle_result.accumulators)
                    and candidate_applied_sha == oracle_applied_sha
                    and candidate_residual_sha == oracle_residual_sha
                )
                proof.update(
                    {
                        "dense_oracle_control_used": True,
                        "oracle_dense_vote_sha256": tensor_sha256(dense_votes),
                        "oracle_q_sha256_after": tensor_sha256(oracle_result.q_levels),
                        "oracle_acc_sha256_after": tensor_sha256(oracle_result.accumulators),
                        "candidate_bounded_decode_sha256_after": tensor_sha256(candidate_decode),
                        "oracle_applied_row_identities_sha256": oracle_applied_sha,
                        "oracle_residual_after_threshold_sha256": oracle_residual_sha,
                        "parity_pass": bool(parity_pass),
                    }
                )
                if bool(proof.get("pass")) and not parity_pass:
                    proof.update(
                        {
                            "pass": False,
                            "scoped_label": None,
                            "terminal_classification": INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
                            "scoped_physical_budget_claim": "not_applicable_domain_gap",
                            "domain_gap_dimension": "toy_oracle_parity",
                            "domain_gap_detail": (
                                "direct bounded local update diverged from the dense oracle "
                                "on q mutations, applied rows, residual-after-threshold, or "
                                "bounded decode parity within the claimed coverage domain"
                            ),
                        }
                    )

            next_states[state_key] = next_state
            tensor_stats[state_key] = {
                "state_key": state_key,
                "candidate_mode": candidate_mode,
                "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
                "q_sha256_before": tensor_sha256(prior_state.q_levels),
                "q_sha256_after": tensor_sha256(next_state.q_levels),
                "bounded_accumulator_fresh_for_exact_shadow": False,
                "bounded_accumulator_rebuilt_for_parity": False,
                "bounded_decode_parity_checked": False,
                "candidate_scoped_label": proof.get("scoped_label"),
                "candidate_terminal_classification": proof.get("terminal_classification"),
                "candidate_dense_decode_used": bool(proof.get("candidate_dense_decode_used")),
                "candidate_accumulator_transient_over2_used": bool(
                    proof.get("candidate_accumulator_transient_over2_used")
                ),
                "candidate_vote_transient_over2_used": bool(
                    proof.get("candidate_vote_transient_over2_used")
                ),
                "candidate_dense_vote_authority_used": bool(
                    proof.get("candidate_dense_vote_authority_used")
                ),
                "coverage_domain": dict(proof.get("coverage_domain") or {}),
                "dense_oracle_control_used": bool(proof.get("dense_oracle_control_used")),
                "candidate_local_update_pass": bool(proof.get("pass")),
                "candidate_local_update_domain_gap_dimension": proof.get("domain_gap_dimension"),
            }
            proof_by_key[state_key] = proof

        summary = {
            "global_rate_cap_enabled": False,
            "candidate_mode": candidate_mode,
            "q_changed_count": sum(
                int((proof.get("q_changed_count") or 0))
                for proof in proof_by_key.values()
            ),
            "candidate_local_update_pass": all(bool(proof.get("pass")) for proof in proof_by_key.values()),
            "candidate_local_update_proof_by_key": proof_by_key,
            "candidate_terminal_classifications": {
                key: proof.get("terminal_classification")
                for key, proof in sorted(proof_by_key.items())
            },
            "candidate_dense_decode_used": False,
            "candidate_accumulator_transient_over2_used": False,
            "candidate_vote_transient_over2_used": False,
            "candidate_dense_vote_authority_used": False,
        }
        return BoundedDeltaLearnerStepResult(
            tensor_states=next_states,
            tensor_stats=tensor_stats,
            deferred_backlog={},
            global_summary=summary,
        )

    hot_by_key = hot_exact_indices_by_key or {}
    vote_update_states: dict[str, VoteUpdateState] = {}
    inputs_by_key: dict[str, VoteUpdateInputs] = {}
    plans_by_key = {}
    cap_inputs: list[GlobalRateCapTensorInput] = []
    for state_key, state in sorted(tensor_states.items()):
        vu_state = state.vote_update_state()
        votes = votes_by_key[state_key].detach().cpu().to(torch.int16).contiguous()
        local_loss_delta = None
        if two_tier_carry_w6_enabled:
            local_loss_delta = _require_local_loss_delta_tensor(
                "local_loss_delta_by_key",
                state_key,
                local_loss_delta_by_key[state_key],
                shape=votes.shape,
            )
        inputs = VoteUpdateInputs(
            votes=votes,
            replay_ce_veto_votes=_coerce_optional_vote_map_tensor(
                "replay_ce_veto_votes_by_key",
                replay_ce_veto_votes_by_key,
                state_key,
                dtype=torch.int16,
                shape=votes.shape,
            ),
            replay_ce_veto_moves=_coerce_optional_vote_map_tensor(
                "replay_ce_veto_moves_by_key",
                replay_ce_veto_moves_by_key,
                state_key,
                dtype=torch.int8,
                shape=votes.shape,
            ),
            pc_aux_votes=_coerce_optional_vote_map_tensor(
                "pc_aux_votes_by_key",
                pc_aux_votes_by_key,
                state_key,
                dtype=torch.int16,
                shape=votes.shape,
            ),
            pc_aux_moves=_coerce_optional_vote_map_tensor(
                "pc_aux_moves_by_key",
                pc_aux_moves_by_key,
                state_key,
                dtype=torch.int8,
                shape=votes.shape,
            ),
            pc_aux_mode=pc_aux_mode,
            local_loss_delta=local_loss_delta,
        )
        spec = vote_specs_by_key[state_key]
        plan = plan_integer_vote_update_reference(
            vu_state,
            inputs,
            spec,
            local_selection_ordering_mode=str(local_selection_ordering_mode),
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
            two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        )
        vote_update_states[state_key] = vu_state
        inputs_by_key[state_key] = inputs
        plans_by_key[state_key] = plan
        cap_inputs.append(
            GlobalRateCapTensorInput(
                state_key=state_key,
                state=vu_state,
                plan=plan,
                vote_inputs=inputs,
            )
        )

    if global_cap_spec is not None:
        cap_result = apply_global_rate_cap_reference(
            cap_inputs,
            global_cap_spec,
            deferred_backlog=deferred_backlog,
            tie_rule_mode=global_cap_tie_rule_mode,
            contract_name=global_cap_contract_name,
        )
        q_acc_by_key = {
            item.state_key: (item.q_levels, item.accumulators, item.stats)
            for item in cap_result.tensor_results
        }
        backlog = cap_result.deferred_backlog
        summary = dict(cap_result.step_summary)
        summary["global_rate_cap_enabled"] = True
        if not two_tier_carry_w6_enabled:
            summary["local_selection_ordering_mode"] = str(local_selection_ordering_mode)
            summary["local_selection_ordering_seed"] = int(local_selection_ordering_seed)
            summary["local_selection_ordering_step"] = int(local_selection_ordering_step)
    else:
        q_acc_by_key = {}
        for state_key in sorted(tensor_states):
            result = apply_integer_vote_update_reference(
                vote_update_states[state_key],
                inputs_by_key[state_key],
                vote_specs_by_key[state_key],
                local_selection_ordering_mode=str(local_selection_ordering_mode),
                local_selection_ordering_seed=int(local_selection_ordering_seed),
                local_selection_ordering_step=int(local_selection_ordering_step),
                two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
            )
            q_acc_by_key[state_key] = (result.q_levels, result.accumulators, result.stats)
        backlog = deferred_backlog or {}
        summary = {
            "global_rate_cap_enabled": False,
            "q_changed_count": sum(
                int(stats.get("q_changed_count", 0))
                for _, _, stats in q_acc_by_key.values()
            ),
        }
        if not two_tier_carry_w6_enabled:
            summary["local_selection_ordering_mode"] = str(local_selection_ordering_mode)
            summary["local_selection_ordering_seed"] = int(local_selection_ordering_seed)
            summary["local_selection_ordering_step"] = int(local_selection_ordering_step)
    if two_tier_carry_w6_enabled:
        summary["two_tier_carry_w6_enabled"] = True
        summary["local_selection_ordering_mode"] = LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA
        summary["two_tier_enabled_pin_count"] = 1
        summary["two_tier_enabled_pin_sha256"] = _two_tier_enabled_pin_sha256()

    if front_c_identity_observer is not None:
        _ignored_observer_return = front_c_identity_observer(
            _front_c_cloned_observation(
                vote_update_states=vote_update_states,
                inputs_by_key=inputs_by_key,
                vote_specs_by_key=vote_specs_by_key,
                plans_by_key=plans_by_key,
                q_acc_by_key=q_acc_by_key,
                deferred_backlog=backlog,
                global_cap_used=global_cap_spec is not None,
            ),
        )
        del _ignored_observer_return

    next_states: dict[str, BoundedDeltaTensorState] = {}
    tensor_stats: dict[str, dict[str, Any]] = {}
    for state_key, prior_state in sorted(tensor_states.items()):
        q_out, acc_out, stats = q_acc_by_key[state_key]
        rebuilt_for_step_parity = False
        next_state = make_live_shadow_tensor_state(
            prior_state,
            q_out,
            acc_out,
            hot_exact_indices=hot_by_key.get(state_key),
            cold_default_value=cold_default_value,
        )
        if parity_check:
            next_state = next_state.with_fresh_bounded_accumulator()
            rebuilt_for_step_parity = True
        next_states[state_key] = next_state
        stats_out = {
            **dict(stats),
            "state_key": state_key,
            "projection_law": S1_PROJECTION_LAW,
            "vote_law": S1_RANK_BUCKET_VOTE_LAW,
            "votes_sha256": _votes_sha(votes_by_key[state_key]),
            "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
            "q_sha256_before": tensor_sha256(prior_state.q_levels),
            "q_sha256_after": tensor_sha256(q_out),
            "exact_accumulator_shadow_sha256_after": tensor_sha256(acc_out),
            "bounded_accumulator_fresh_for_exact_shadow": bool(
                next_state.bounded_accumulator_fresh_for_exact_shadow
            ),
            "bounded_accumulator_rebuilt_for_parity": bool(rebuilt_for_step_parity),
            "bounded_decode_parity_checked": bool(parity_check),
        }
        if parity_check:
            parity = next_state.bounded_decode_parity_report(fail_on_mismatch=True)
            stats_out["bounded_accumulator_decoded_sha256_after"] = parity[
                "bounded_accumulator_decoded_sha256"
            ]
            stats_out["bounded_decode_matches_exact_shadow"] = parity[
                "exact_shadow_matches_bounded_decode"
            ]
        if two_tier_carry_w6_enabled:
            assert_two_tier_threshold_receipt_consistent(stats_out)
        tensor_stats[state_key] = stats_out
    return BoundedDeltaLearnerStepResult(
        tensor_states=next_states,
        tensor_stats=tensor_stats,
        deferred_backlog=backlog,
        global_summary=summary,
    )


def reanchor_s1_oracle_hash(
    pointer: SourcePointer = LIVE_S1_TRAINER_POINTER,
    *,
    semantics_choice: str = "current_file_reanchored",
) -> dict[str, Any]:
    pointer.validate_static()
    current_sha = file_sha256(pointer.absolute_path)
    return {
        "schema": S1_ORACLE_REANCHOR_SCHEMA_VERSION,
        "label": pointer.label,
        "absolute_path": pointer.absolute_path,
        "expected_sha256": pointer.expected_sha256,
        "current_sha256": current_sha,
        "expected_matches_current": current_sha == pointer.expected_sha256,
        "reanchored": True,
        "semantics_choice": semantics_choice,
        "runtime_dependency": False,
        "source_pointer_role": pointer.implementation_role,
        "reanchor_note": pointer.reanchor_note,
    }


def build_authoritative_checkpoint_payload(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    *,
    step: int,
    updater_config: Mapping[str, Any],
    oracle_receipt: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    checkpoint_written: bool = False,
) -> dict[str, Any]:
    tensor_summaries = {
        key: state.to_schema_dict()
        for key, state in sorted(tensor_states.items())
    }
    state_summary_sha = _sha256_bytes(_canonical_json(tensor_summaries).encode("utf-8"))
    return _dict_without_none(
        {
            "schema": BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
            "artifact_role": "c2_bounded_delta_authoritative_train_state",
            "authoritative_state_source": AUTHORITATIVE_STATE_SOURCE,
            "step": int(step),
            "dry_run": bool(dry_run),
            "checkpoint_written": bool(checkpoint_written),
            "q_codec": "int8_levels_transitional_base3_pack_ready",
            "frozen_scale_law": "per_tensor_absmean_frozen_from_parent_qscale",
            "bounded_accumulator_schema": BOUNDED_DELTA_ACCUMULATOR_SCHEMA_VERSION,
            "updater_config_sha256": _sha256_bytes(
                _canonical_json(dict(updater_config)).encode("utf-8"),
            ),
            "authoritative_state_sha256": state_summary_sha,
            "tensor_summaries": tensor_summaries,
            "source_oracle_receipt": dict(oracle_receipt) if oracle_receipt is not None else None,
            "telemetry_proves_q_changes_from": BOUNDED_UPDATE_ATTRIBUTION,
        }
    )


def validate_authoritative_resume_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported bounded-delta checkpoint schema")
    if payload.get("artifact_role") != "c2_bounded_delta_authoritative_train_state":
        raise ValueError("resume requires c2 bounded-delta authoritative train state, not eval export")
    if payload.get("authoritative_state_source") != AUTHORITATIVE_STATE_SOURCE:
        raise ValueError("resume payload is not q+scale+bounded accumulator authoritative state")
    if payload.get("eval_export", False):
        raise ValueError("eval export is not resumable learner state")
    if "tensor_summaries" not in payload:
        raise ValueError("resume payload missing tensor_summaries")


@dataclass(frozen=True)
class BoundedDeltaDryRunReceipt:
    schema: str
    dry_run: bool
    device: str
    gpu_launched: bool
    checkpoint_written: bool
    first_forward_backward_update_finite: bool
    parent_hash_unchanged: bool
    parent_hash_basis: str
    projection_law: str
    vote_law: str
    bounded_update_attribution: str
    optimizer_identity_proof: dict[str, Any]
    oracle_receipt: dict[str, Any]
    step_result: dict[str, Any]
    checkpoint_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_default_off(enabled: bool | None) -> None:
    if enabled is True:
        return
    if os.environ.get(RUN_BOUNDED_DELTA_LEARNER_ENV) == "1":
        return
    raise RuntimeError(
        f"C2.0 bounded-delta learner is default-off; pass enabled=True or set {RUN_BOUNDED_DELTA_LEARNER_ENV}=1",
    )


def run_c2_bounded_delta_cpu_dry_run(
    *,
    enabled: bool | None = None,
    device: str = "cpu",
    oracle_pointer: SourcePointer = LIVE_S1_TRAINER_POINTER,
) -> BoundedDeltaDryRunReceipt:
    _assert_default_off(enabled)
    if device != "cpu":
        raise RuntimeError("C2.0 dry-run smoke is CPU-only in this gate; GPU launch is not allowed")

    from calm.hrm_text_158.bit_linear import BitLinear

    torch.manual_seed(158)

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(3, 2, bias=False)
            self.noneligible = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = _Tiny().to(device)
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    optimizer, checks = build_optimizer_excluding_eligible_masters(model, eligible)
    if optimizer is not None:
        model.noneligible.grad = torch.ones_like(model.noneligible)
    optimizer_proof = prove_eligible_master_identity_after_optimizer_step(
        optimizer,
        eligible,
        optimizer_checks=checks,
    )

    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": tensor_state},
        device=device,
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        weighted_grad = handle.weighted_grad("proj")
    credit = credit_from_weighted_grad(weighted_grad)
    moves = project_s1_gradient_to_moves(weighted_grad, tensor_state.q_levels)
    rank_spec = default_dry_run_rank_vote_spec()
    votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
    vote_spec = VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )
    step_result = apply_bounded_delta_vote_step(
        {"proj": tensor_state},
        {"proj": votes},
        {"proj": vote_spec},
        hot_exact_indices_by_key={"proj": tuple(range(int(q.numel())))},
    )
    oracle_receipt = reanchor_s1_oracle_hash(oracle_pointer)
    updater_config = {
        "rank_vote_spec": rank_spec.to_live_dict(),
        "vote_update_spec": asdict(vote_spec),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }
    checkpoint_payload = build_authoritative_checkpoint_payload(
        step_result.tensor_states,
        step=1,
        updater_config=updater_config,
        oracle_receipt=oracle_receipt,
        dry_run=True,
        checkpoint_written=False,
    )
    validate_authoritative_resume_payload(checkpoint_payload)
    finite = bool(torch.isfinite(loss).item()) and bool(torch.isfinite(weighted_grad).all().item())
    q_changed = int(step_result.global_summary.get("q_changed_count", 0))
    return BoundedDeltaDryRunReceipt(
        schema=BOUNDED_DELTA_LEARNER_SCHEMA_VERSION,
        dry_run=True,
        device=device,
        gpu_launched=False,
        checkpoint_written=False,
        first_forward_backward_update_finite=bool(finite and q_changed > 0),
        parent_hash_unchanged=True,
        parent_hash_basis=DEFAULT_DRY_RUN_PARENT_HASH_BASIS,
        projection_law=S1_PROJECTION_LAW,
        vote_law=S1_RANK_BUCKET_VOTE_LAW,
        bounded_update_attribution=BOUNDED_UPDATE_ATTRIBUTION,
        optimizer_identity_proof=optimizer_proof,
        oracle_receipt=oracle_receipt,
        step_result=step_result.to_dict(),
        checkpoint_payload=checkpoint_payload,
    )


__all__ = [
    "AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY",
    "AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT",
    "AUTHORITATIVE_STATE_SOURCE",
    "BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION",
    "BOUNDED_DELTA_LEARNER_SCHEMA_VERSION",
    "BOUNDED_UPDATE_ATTRIBUTION",
    "BoundedDeltaDryRunReceipt",
    "BoundedDeltaLearnerStepResult",
    "BoundedDeltaTensorState",
    "RankVoteBin",
    "RankVoteSpec",
    "RUN_BOUNDED_DELTA_LEARNER_ENV",
    "S1_INVERTED_SIGN_PRESSURE_VOTE_LAW",
    "S1_ORACLE_REANCHOR_SCHEMA_VERSION",
    "S1_PROJECTION_LAW",
    "S1_RANK_BUCKET_VOTE_LAW",
    "S1_SIGN_PRESSURE_VOTE_LAW",
    "apply_bounded_delta_vote_step",
    "authoritative_forward_context",
    "build_authoritative_checkpoint_payload",
    "build_optimizer_excluding_eligible_masters",
    "candidate_weighted_grad_and_diag_fisher_proxies_from_captures",
    "candidate_weighted_grad_proxies_from_captures",
    "compact_pressure_shape_summary",
    "compact_signed_rank_bin_mass_summary",
    "build_pressure_shape_summary_v1",
    "compact_vote_pressure_summary",
    "credit_from_weighted_grad",
    "default_dry_run_rank_vote_spec",
    "derive_bounded_tensor_state_from_weight",
    "file_sha256",
    "make_candidate_authority_tensor_state",
    "make_live_shadow_tensor_state",
    "make_bounded_tensor_state",
    "project_s1_gradient_to_moves",
    "prove_eligible_master_identity_after_optimizer_step",
    "rank_bucketed_int16_votes",
    "reanchor_s1_oracle_hash",
    "run_c2_bounded_delta_cpu_dry_run",
    "sign_pressure_int16_votes",
    "snapshot_eligible_master_sha256",
    "tensor_sha256",
    "ternarize_weight_to_q_scale",
    "validate_authoritative_resume_payload",
    "weighted_grad_from_captures",
]
