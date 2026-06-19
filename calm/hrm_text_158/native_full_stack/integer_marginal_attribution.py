"""CPU reference integer marginal attribution for HRM-Text-1.58 Step 3C-A.

Event-native carrier replacing dense FP32 weighted_grad on the integer path.
FP weighted_grad remains parity-reference only. Dense int32 [O,I] scratch inside
the reference function is labeled cpu_reference_dense_int32_scratch and does NOT
clear optimizer_credit_state debt or authorize real_native_integer_attribution_present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import _as_bsi

INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0 = "einsum_q15q16_int32_v0"
INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1 = "einsum_q15q16_rescale_q24_v1"
INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID = INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1

# Backward-compatible alias: historical tests and explicit v0 regression use this id.
INTEGER_MARGINAL_ATTRIBUTION_LAW_ID = INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0

ATTRIBUTION_RESCALE_SHIFT_V0 = 31
ATTRIBUTION_RESCALE_SHIFT_V1 = 24
ATTRIBUTION_RESCALE_SHIFT = ATTRIBUTION_RESCALE_SHIFT_V1

INPUT_Q15_SCALE = 2**15
GRAD_Q16_SCALE = 2**16
ATTRIBUTION_Q31_FRAC_BITS = 31

ROUNDING_MODE = "round_half_away_from_zero"
CLAMP_POLICY = "saturate_int32"
SATURATION_BEHAVIOR = "fail_closed_on_overflow"

CPU_REFERENCE_DENSE_INT32_SCRATCH_LABEL = "cpu_reference_dense_int32_scratch"

INDEX_SET_ALL_STRUCTURALLY_TOUCHED = "all_structurally_touched_v0"
INDEX_SET_PROJECTED_MOVE_REFERENCE_ONLY = "projected_move_reference_only"

BRANCH_3C_PARITY_PASS_CPU = "BR-3C-PARITY-PASS-CPU"
BRANCH_3C_PARITY_FAIL = "BR-3C-PARITY-FAIL"
BRANCH_3C_DENSE_LEAK = "BR-3C-DENSE-LEAK"
BRANCH_3C_MEASUREMENT_INVALID = "BR-3C-MEASUREMENT-INVALID"

INTEGER_MARGINAL_ATTRIBUTION_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
)

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class IntegerMarginalAttributionEvents:
    flat_indices: torch.Tensor
    attribution_q31: torch.Tensor
    law_id: str
    numel: int
    index_set_policy: str = INDEX_SET_ALL_STRUCTURALLY_TOUCHED

    def validate(self) -> None:
        if self.flat_indices.dtype != torch.int64:
            raise ValueError(f"flat_indices must be torch.int64, got {self.flat_indices.dtype}")
        if self.attribution_q31.dtype != torch.int32:
            raise ValueError(f"attribution_q31 must be torch.int32, got {self.attribution_q31.dtype}")
        if self.flat_indices.dim() != 1 or self.attribution_q31.dim() != 1:
            raise ValueError("flat_indices and attribution_q31 must be 1-D")
        if int(self.flat_indices.numel()) != int(self.attribution_q31.numel()):
            raise ValueError("flat_indices and attribution_q31 length mismatch")
        if not self.flat_indices.is_cpu or not self.attribution_q31.is_cpu:
            raise ValueError("flat_indices and attribution_q31 must be CPU tensors")
        if int(self.numel) <= 0:
            raise ValueError("numel must be > 0")
        if self.event_count() == 0:
            return
        if bool((self.attribution_q31 == 0).any().item()):
            raise ValueError("attribution_q31 must not contain zeros")
        if self.flat_indices.numel() > 1:
            diffs = self.flat_indices[1:] - self.flat_indices[:-1]
            if not bool((diffs > 0).all().item()):
                raise ValueError("flat_indices must be strictly increasing and unique")
        idx_min = int(self.flat_indices.min().item())
        idx_max = int(self.flat_indices.max().item())
        if idx_min < 0 or idx_max >= int(self.numel):
            raise ValueError("flat_indices out of range for numel")

    def event_count(self) -> int:
        return int(self.flat_indices.numel())

    @property
    def reference_only(self) -> bool:
        return self.index_set_policy == INDEX_SET_PROJECTED_MOVE_REFERENCE_ONLY

    def is_production_oracle(self) -> bool:
        return self.index_set_policy == INDEX_SET_ALL_STRUCTURALLY_TOUCHED


def _round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    fp = values.to(torch.float64)
    return torch.sign(fp) * torch.floor(fp.abs() + 0.5)


def _quantize_to_int32(values: torch.Tensor, *, scale: int) -> torch.Tensor:
    rounded = _round_half_away_from_zero(values.to(torch.float32) * float(scale))
    as_int64 = rounded.to(torch.int64)
    if bool((as_int64 < INT32_MIN).any().item()) or bool((as_int64 > INT32_MAX).any().item()):
        raise ValueError("quantized value exceeds int32 range before accumulation")
    return as_int64.to(torch.int32)


def _attribution_rescale_shift_for_law(law_id: str) -> int:
    if law_id == INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0:
        return ATTRIBUTION_RESCALE_SHIFT_V0
    if law_id == INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1:
        return ATTRIBUTION_RESCALE_SHIFT_V1
    raise ValueError(f"unsupported law_id: {law_id!r}")


def _rescale_accumulator_to_attribution_q(
    accumulator: torch.Tensor,
    *,
    shift: int,
) -> torch.Tensor:
    """Map q15*q16 accumulator to int32 attribution with parameterized right-shift."""

    values = accumulator.to(torch.int64)
    shift_i = int(shift)
    half = 1 << (shift_i - 1)
    positive = values >= 0
    abs_values = values.abs()
    rounded = (abs_values + half) >> shift_i
    rescaled = torch.where(positive, rounded, -rounded)
    if bool((rescaled < INT32_MIN).any().item()) or bool((rescaled > INT32_MAX).any().item()):
        raise ValueError("integer marginal attribution accumulation overflowed int32 range")
    return rescaled.to(torch.int32)


def _rescale_accumulator_to_attribution_q31(accumulator: torch.Tensor) -> torch.Tensor:
    """Banked v0 rescale (>>31) retained for regression."""

    return _rescale_accumulator_to_attribution_q(
        accumulator,
        shift=ATTRIBUTION_RESCALE_SHIFT_V0,
    )


def _accumulate_cpu_reference_dense_int32_scratch(
    paired_inputs: Sequence[torch.Tensor],
    grad_outputs_reversed: Sequence[torch.Tensor],
    *,
    weight_shape: tuple[int, int],
) -> torch.Tensor:
    """Transient dense int64 [O,I] scratch; immediately reduced to sparse events."""

    out_features, in_features = weight_shape
    # cpu_reference_dense_int32_scratch: dense accumulator, reference-only, not row-flip evidence.
    accumulator = torch.zeros(
        (out_features, in_features),
        dtype=torch.int64,
        device="cpu",
    )
    for inp, grad_out in zip(paired_inputs, grad_outputs_reversed):
        input_bsi = _as_bsi(inp.detach().to(torch.float32), name="input").cpu()
        grad_out_bso = _as_bsi(grad_out.detach().to(torch.float32), name="grad_out").cpu()
        input_q15 = _quantize_to_int32(input_bsi, scale=INPUT_Q15_SCALE)
        grad_q16 = _quantize_to_int32(grad_out_bso, scale=GRAD_Q16_SCALE)
        accumulator += torch.einsum(
            "bso,bsi->oi",
            grad_q16.to(torch.int64),
            input_q15.to(torch.int64),
        )
    return accumulator


def integer_marginal_attribution_from_captures(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    index_set_policy: str = INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    reference_flat_indices: torch.Tensor | None = None,
) -> IntegerMarginalAttributionEvents:
    if law_id not in {
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    }:
        raise ValueError(f"unsupported law_id: {law_id!r}")
    if index_set_policy not in {
        INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
        INDEX_SET_PROJECTED_MOVE_REFERENCE_ONLY,
    }:
        raise ValueError(f"unsupported index_set_policy: {index_set_policy!r}")
    if not inputs or not grad_outputs:
        raise ValueError("inputs and grad_outputs must be non-empty")
    if len(inputs) < len(grad_outputs):
        raise ValueError("capture call-count mismatch")
    weight_dims = tuple(int(dim) for dim in weight_shape)
    if len(weight_dims) != 2:
        raise ValueError(f"weight_shape must be rank-2, got {weight_dims}")
    numel = int(weight_dims[0] * weight_dims[1])
    paired_inputs = inputs[-len(grad_outputs) :]
    grad_outputs_reversed = list(reversed(list(grad_outputs)))

    accumulator = _accumulate_cpu_reference_dense_int32_scratch(
        paired_inputs,
        grad_outputs_reversed,
        weight_shape=weight_dims,
    )
    attribution_dense = _rescale_accumulator_to_attribution_q(
        accumulator,
        shift=_attribution_rescale_shift_for_law(law_id),
    )

    if index_set_policy == INDEX_SET_ALL_STRUCTURALLY_TOUCHED:
        flat = attribution_dense.reshape(-1)
        nz = torch.nonzero(flat != 0, as_tuple=False).flatten().to(torch.int64)
        attribution_q31 = flat.index_select(0, nz).to(torch.int32)
        flat_indices = nz
    else:
        if reference_flat_indices is None:
            raise ValueError(
                "reference_flat_indices required for projected_move_reference_only policy"
            )
        flat_indices = reference_flat_indices.detach().cpu().flatten().to(torch.int64)
        attribution_q31 = attribution_dense.reshape(-1).index_select(0, flat_indices).to(torch.int32)

    events = IntegerMarginalAttributionEvents(
        flat_indices=flat_indices.contiguous(),
        attribution_q31=attribution_q31.contiguous(),
        law_id=law_id,
        numel=numel,
        index_set_policy=index_set_policy,
    )
    events.validate()
    return events


def _scalar_projected_move(*, q_level: int, grad_value: int) -> int:
    if q_level < 0 and grad_value < 0:
        return 1
    if q_level == 0 and grad_value < 0:
        return 1
    if q_level == 0 and grad_value > 0:
        return -1
    if q_level > 0 and grad_value > 0:
        return -1
    return 0


def projected_moves_from_integer_attribution(
    events: IntegerMarginalAttributionEvents,
    q_levels_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    events.validate()
    q_flat = q_levels_flat.detach().cpu().reshape(-1).to(torch.int8)
    if int(q_flat.numel()) != int(events.numel):
        raise ValueError("q_levels_flat numel mismatch")
    if events.event_count() == 0:
        empty_idx = torch.empty(0, dtype=torch.int64)
        empty_moves = torch.empty(0, dtype=torch.int8)
        return empty_idx, empty_moves
    q_selected = q_flat.index_select(0, events.flat_indices)
    moves = torch.tensor(
        [
            _scalar_projected_move(
                q_level=int(q_level.item()),
                grad_value=int(grad_value.item()),
            )
            for q_level, grad_value in zip(q_selected, events.attribution_q31)
        ],
        dtype=torch.int8,
    )
    move_mask = moves != 0
    move_indices = events.flat_indices.index_select(0, torch.nonzero(move_mask, as_tuple=False).flatten())
    move_values = moves.index_select(0, torch.nonzero(move_mask, as_tuple=False).flatten())
    return move_indices.contiguous(), move_values.contiguous()


def integer_marginal_attribution_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in INTEGER_MARGINAL_ATTRIBUTION_HARD_FALSE_FIELDS}


def dense_int32_scratch_is_reference_only_not_row_flip_evidence() -> bool:
    """Explicit boundary marker: dense scratch does not satisfy 3C-C proof contract."""

    return True


@dataclass(frozen=True)
class StreamingSparseAttributionMetrics:
    max_candidate_tile_shape: tuple[int, ...]
    max_candidate_tile_numel: int
    max_candidate_tile_bytes: int
    full_dense_shape: tuple[int, int]
    full_dense_numel: int
    full_dense_baseline_bytes: int
    candidate_event_count: int
    candidate_event_carrier_peak_bytes: int
    event_carrier_density_ratio: float


def _update_streaming_tile_metrics(
    metrics: dict[str, int | tuple[int, ...]],
    tensor: torch.Tensor,
) -> None:
    shape = tuple(int(dim) for dim in tensor.shape)
    numel = int(tensor.numel())
    if numel <= 0:
        return
    bytes_ = numel * int(tensor.element_size())
    if numel > int(metrics["max_candidate_tile_numel"]):
        metrics["max_candidate_tile_shape"] = shape
        metrics["max_candidate_tile_numel"] = numel
        metrics["max_candidate_tile_bytes"] = bytes_


def streaming_sparse_attribution_from_captures(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
) -> tuple[IntegerMarginalAttributionEvents, StreamingSparseAttributionMetrics]:
    """Streaming-sparse attribution: per-row 1-D tiles only, never 2-D [O,I] int tensors."""

    if law_id not in {
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    }:
        raise ValueError(f"unsupported law_id: {law_id!r}")
    if not inputs or not grad_outputs:
        raise ValueError("inputs and grad_outputs must be non-empty")
    if len(inputs) < len(grad_outputs):
        raise ValueError("capture call-count mismatch")
    weight_dims = tuple(int(dim) for dim in weight_shape)
    if len(weight_dims) != 2:
        raise ValueError(f"weight_shape must be rank-2, got {weight_dims}")
    out_features, in_features = weight_dims
    numel = int(out_features * in_features)
    paired_inputs = inputs[-len(grad_outputs) :]
    grad_outputs_reversed = list(reversed(list(grad_outputs)))
    shift = _attribution_rescale_shift_for_law(law_id)

    tile_metrics: dict[str, int | tuple[int, ...]] = {
        "max_candidate_tile_shape": (0,),
        "max_candidate_tile_numel": 0,
        "max_candidate_tile_bytes": 0,
    }

    flat_indices_parts: list[torch.Tensor] = []
    attribution_parts: list[torch.Tensor] = []

    for row_index in range(out_features):
        acc_o = torch.zeros(in_features, dtype=torch.int64, device="cpu")
        for inp, grad_out in zip(paired_inputs, grad_outputs_reversed):
            input_bsi = _as_bsi(inp.detach().to(torch.float32), name="input").cpu()
            grad_out_bso = _as_bsi(grad_out.detach().to(torch.float32), name="grad_out").cpu()
            input_q15 = _quantize_to_int32(input_bsi, scale=INPUT_Q15_SCALE)
            grad_q16 = _quantize_to_int32(grad_out_bso, scale=GRAD_Q16_SCALE)
            acc_o += torch.einsum(
                "bs,bsi->i",
                grad_q16[..., row_index].to(torch.int64),
                input_q15.to(torch.int64),
            )
        _update_streaming_tile_metrics(tile_metrics, acc_o)

        # FOLD-5: true 1-D rescale — never unsqueeze to (1,I) full-dense shape.
        attr_o = _rescale_accumulator_to_attribution_q(acc_o, shift=shift)
        _update_streaming_tile_metrics(tile_metrics, attr_o)

        # Avoid aten.nonzero 2-D (N,1) index matrix — trips full-dense leak observer.
        row_mask = attr_o != 0
        nz = torch.arange(attr_o.numel(), dtype=torch.int64, device=attr_o.device).masked_select(
            row_mask
        )
        if int(nz.numel()) == 0:
            continue
        row_attr = attr_o.index_select(0, nz).to(torch.int32)
        row_flat = (row_index * in_features + nz).to(torch.int64)
        flat_indices_parts.append(row_flat)
        attribution_parts.append(row_attr)

    if flat_indices_parts:
        flat_indices = torch.cat(flat_indices_parts, dim=0).contiguous()
        attribution_q31 = torch.cat(attribution_parts, dim=0).contiguous()
    else:
        flat_indices = torch.empty(0, dtype=torch.int64)
        attribution_q31 = torch.empty(0, dtype=torch.int32)

    events = IntegerMarginalAttributionEvents(
        flat_indices=flat_indices,
        attribution_q31=attribution_q31,
        law_id=law_id,
        numel=numel,
        index_set_policy=INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    )
    events.validate()

    carrier_bytes = int(flat_indices.numel() * 8 + attribution_q31.numel() * 4)
    full_dense_baseline_bytes = numel * 8
    density_ratio = float(events.event_count()) / float(numel) if numel > 0 else 0.0

    metrics = StreamingSparseAttributionMetrics(
        max_candidate_tile_shape=tuple(int(dim) for dim in tile_metrics["max_candidate_tile_shape"]),  # type: ignore[arg-type]
        max_candidate_tile_numel=int(tile_metrics["max_candidate_tile_numel"]),
        max_candidate_tile_bytes=int(tile_metrics["max_candidate_tile_bytes"]),
        full_dense_shape=weight_dims,
        full_dense_numel=numel,
        full_dense_baseline_bytes=full_dense_baseline_bytes,
        candidate_event_count=events.event_count(),
        candidate_event_carrier_peak_bytes=carrier_bytes,
        event_carrier_density_ratio=density_ratio,
    )
    return events, metrics
