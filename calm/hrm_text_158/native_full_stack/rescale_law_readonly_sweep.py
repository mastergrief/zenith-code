"""Read-only rescale-shift sweep helpers for HRM-Text-1.58 Step 3C-A.

Shift-parameterized rank parity on captures without mutating production law.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
    INT32_MAX,
    INT32_MIN,
    IntegerMarginalAttributionEvents,
    _accumulate_cpu_reference_dense_int32_scratch,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
    _rank_positions_for_credit_values,
    compare_sparse_rank_to_fp_dense_reference,
    credit_q31_from_attribution,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    KeyProbeMetrics,
    MIN_MOVE_CANDIDATES,
    MIN_TIER_TOTAL_MOVE_CANDIDATES,
    PerCandidateParityRecord,
    RawKeyCapture,
    Tier2RawCaptureBundle,
    VERDICT_BROAD_HOLDS,
    VERDICT_NARROW_HOLDS,
    _aggregate_tier_rates,
    _compute_parity_rates_from_records,
    _count_fp_credit_nonzero,
    _fractional_diversity_count,
    _measurement_validity_for_key,
    _rank_group_count,
    build_trainer_16x16_capture_fixture,
    capture_tier2_checkpoint_raw_captures,
    classify_tier_parity_verdict,
    sha256_file,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    FROZEN_T2_ANCHOR_BATCH_SIZE,
    FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    FROZEN_T2_ANCHOR_CURRICULUM_SEED,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    derive_trainer_sub2_authority_states,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)

RESCALE_LAW_READONLY_SWEEP_SCHEMA_V1 = "hrm_text_158_3c_a_rescale_law_readonly_sweep/v1"
DUAL_TIER_SWEEP_SCHEMA_V1 = "hrm_text_158_3c_a_rescale_law_t1_t2_readonly_sweep/v1"

LAW_V0_SHIFT = 31
SELECTOR_SHIFTS_COARSEST_FIRST = (24, 16, 8)
SELECTOR_SHIFTS_FINER_CANDIDATES = (16, 8)
ALL_SWEEP_CANDIDATES = (
    ("law_v0", LAW_V0_SHIFT),
    ("rescale_q24", 24),
    ("rescale_q16", 16),
    ("rescale_q8", 8),
)

SELECTOR_OUTCOME_PIN_COARSEST_T2_BROAD = "PIN_COARSEST_T2_BROAD"
SELECTOR_OUTCOME_PIN_COARSEST_T2_NARROW_CAVEATED = "PIN_COARSEST_T2_NARROW_CAVEATED"
SELECTOR_OUTCOME_STOP_NO_SHIFT_CLEARS_VALID_T2 = "STOP_NO_SHIFT_CLEARS_VALID_T2"
SELECTOR_OUTCOME_STOP_RANK_STRUCTURE = "STOP_RANK_STRUCTURE"
SELECTOR_OUTCOME_PIN_CANDIDATE_REPORT_ONLY = "PIN_CANDIDATE_REPORT_ONLY"
SELECTOR_OUTCOME_STOP_TIER3_INT64_ROUTE = "STOP_TIER3_INT64_ROUTE"

ROUTING_STRENGTH_CLEAN = "CLEAN"
ROUTING_STRENGTH_CAVEATED = "CAVEATED"
ROUTING_STRENGTH_NONE = "NONE"

INTEGER_MARGINAL_ATTRIBUTION_REL = Path(
    "calm/hrm_text_158/native_full_stack/integer_marginal_attribution.py"
)


class RescaleSaturationError(ValueError):
    """Rescale shift produced int32 overflow on accumulator."""


@dataclass(frozen=True)
class SweepCandidateResult:
    candidate_id: str
    rescale_shift: int
    move_candidate_count: int
    rank_positions_match_rate: float
    events_match_rate: float
    fractional_collision_share_of_mismatches: float
    parity_verdict: str
    measurement_valid: bool
    rank_match_count: int
    event_match_count: int
    mismatch_count: int
    fractional_collision_mismatch_count: int
    saturation_fail_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rescale_shift": int(self.rescale_shift),
            "move_candidate_count": int(self.move_candidate_count),
            "rank_positions_match_rate": float(self.rank_positions_match_rate),
            "events_match_rate": float(self.events_match_rate),
            "fractional_collision_share_of_mismatches": float(
                self.fractional_collision_share_of_mismatches
            ),
            "parity_verdict": str(self.parity_verdict),
            "measurement_valid": bool(self.measurement_valid),
            "rank_match_count": int(self.rank_match_count),
            "event_match_count": int(self.event_match_count),
            "mismatch_count": int(self.mismatch_count),
            "fractional_collision_mismatch_count": int(
                self.fractional_collision_mismatch_count
            ),
            "saturation_fail_count": int(self.saturation_fail_count),
        }


@dataclass(frozen=True)
class ShiftParityKeyResult:
    state_key: str
    measurement_valid: bool
    validity_detail: dict[str, Any]
    move_candidate_count: int
    rank_positions_match_rate: float
    events_match_rate: float
    fractional_collision_share_of_mismatches: float
    saturation_failed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "measurement_valid": bool(self.measurement_valid),
            "validity_detail": dict(self.validity_detail),
            "move_candidate_count": int(self.move_candidate_count),
            "rank_positions_match_rate": float(self.rank_positions_match_rate),
            "events_match_rate": float(self.events_match_rate),
            "fractional_collision_share_of_mismatches": float(
                self.fractional_collision_share_of_mismatches
            ),
            "saturation_failed": bool(self.saturation_failed),
        }


@dataclass(frozen=True)
class TierShiftParityResult:
    tier_id: str
    rescale_shift: int
    candidate_id: str
    measurement_valid: bool
    parity_verdict: str
    rank_positions_match_rate: float
    events_match_rate: float
    fractional_collision_share_of_mismatches: float
    total_move_candidates: int
    valid_key_count: int
    saturation_fail_count: int
    per_key_results: tuple[ShiftParityKeyResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "rescale_shift": int(self.rescale_shift),
            "candidate_id": str(self.candidate_id),
            "measurement_valid": bool(self.measurement_valid),
            "parity_verdict": str(self.parity_verdict),
            "rank_positions_match_rate": float(self.rank_positions_match_rate),
            "events_match_rate": float(self.events_match_rate),
            "fractional_collision_share_of_mismatches": float(
                self.fractional_collision_share_of_mismatches
            ),
            "total_move_candidates": int(self.total_move_candidates),
            "valid_key_count": int(self.valid_key_count),
            "saturation_fail_count": int(self.saturation_fail_count),
            "per_key_results": [item.to_dict() for item in self.per_key_results],
        }


@dataclass(frozen=True)
class DualTierCandidateResult:
    candidate_id: str
    rescale_shift: int
    t1: SweepCandidateResult
    t2: TierShiftParityResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rescale_shift": int(self.rescale_shift),
            "t1": self.t1.to_dict(),
            "t2": self.t2.to_dict(),
        }


def integer_marginal_attribution_module_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / INTEGER_MARGINAL_ATTRIBUTION_REL


def rescale_accumulator_to_attribution_q(
    accumulator: torch.Tensor,
    *,
    shift: int,
) -> torch.Tensor:
    values = accumulator.to(torch.int64)
    half = 1 << (int(shift) - 1)
    positive = values >= 0
    abs_values = values.abs()
    rounded = (abs_values + half) >> int(shift)
    rescaled = torch.where(positive, rounded, -rounded)
    if bool((rescaled < INT32_MIN).any().item()) or bool((rescaled > INT32_MAX).any().item()):
        raise RescaleSaturationError(
            f"rescale shift={shift} produced values outside int32 range"
        )
    return rescaled.to(torch.int32)


def attribution_events_from_captures_with_shift(
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    *,
    weight_shape: Sequence[int],
    rescale_shift: int,
) -> IntegerMarginalAttributionEvents:
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
    attribution_dense = rescale_accumulator_to_attribution_q(
        accumulator,
        shift=int(rescale_shift),
    )
    flat = attribution_dense.reshape(-1)
    nz = torch.nonzero(flat != 0, as_tuple=False).flatten().to(torch.int64)
    events = IntegerMarginalAttributionEvents(
        flat_indices=nz.contiguous(),
        attribution_q31=flat.index_select(0, nz).to(torch.int32).contiguous(),
        law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID,
        numel=numel,
        index_set_policy=INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    )
    events.validate()
    return events


def build_full_parity_records_with_shift(
    *,
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    spec: RankVoteSpec,
    rescale_shift: int,
    credit_law_id: str = CREDIT_LAW_NEG_ATTRIBUTION_Q31_V0,
) -> list[PerCandidateParityRecord]:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    attribution_events = attribution_events_from_captures_with_shift(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
        rescale_shift=int(rescale_shift),
    )
    move_indices, moves = projected_moves_from_integer_attribution(
        attribution_events,
        q_levels_flat,
    )
    weighted_grad = weighted_grad_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
    )
    fp_credit = credit_from_weighted_grad(weighted_grad)
    fp_moves = project_s1_gradient_to_moves(weighted_grad, q_levels_flat.reshape(weight_dims))
    index_to_pos = {
        int(index): pos for pos, index in enumerate(attribution_events.flat_indices.tolist())
    }
    attribution_selected = torch.tensor(
        [
            int(attribution_events.attribution_q31[index_to_pos[int(index)]].item())
            for index in move_indices.tolist()
        ],
        dtype=torch.int32,
    )
    credit_q31 = credit_q31_from_attribution(attribution_selected, credit_law_id=credit_law_id)
    parity = compare_sparse_rank_to_fp_dense_reference(
        credit_q31,
        moves,
        move_indices,
        fp_credit,
        fp_moves,
        spec,
        credit_law_id=credit_law_id,
    )
    fp_votes_dense = rank_bucketed_int16_votes(fp_credit, fp_moves, spec)
    fp_reference_events = SparseVoteEvents.from_dense_votes(fp_votes_dense)
    int_events = parity.events
    fp_credit_sparse = (
        fp_credit.reshape(-1).index_select(0, move_indices).to(torch.float32)
        if int(move_indices.numel()) > 0
        else torch.empty(0, dtype=torch.float32)
    )
    int_rank = _rank_positions_for_credit_values(credit_q31.to(torch.float32), spec)
    fp_rank = _rank_positions_for_credit_values(fp_credit_sparse, spec)
    int_vote_by_index = {
        int(index): int(value)
        for index, value in zip(int_events.indices.tolist(), int_events.values.tolist())
    }
    fp_vote_by_index = {
        int(index): int(value)
        for index, value in zip(
            fp_reference_events.indices.tolist(),
            fp_reference_events.values.tolist(),
        )
    }
    records: list[PerCandidateParityRecord] = []
    for pos, flat_index_tensor in enumerate(move_indices.tolist()):
        flat_index = int(flat_index_tensor)
        fp_value = float(fp_credit_sparse[pos].item())
        int_credit = int(credit_q31[pos].item())
        fp_rank_pos = int(fp_rank[pos].item())
        int_rank_pos = int(int_rank[pos].item())
        fp_vote = int(fp_vote_by_index.get(flat_index, 0))
        int_vote = int(int_vote_by_index.get(flat_index, 0))
        rank_match = fp_rank_pos == int_rank_pos
        event_match = fp_vote == int_vote
        mismatch = not (rank_match and event_match)
        fractional_collision = mismatch and (
            abs(fp_value - round(fp_value)) > 1e-6
            or abs(fp_value - float(int_credit)) > 1e-3
        )
        records.append(
            PerCandidateParityRecord(
                flat_index=flat_index,
                fp_credit=fp_value,
                int_credit_q31=int_credit,
                fp_rank_position=fp_rank_pos,
                int_rank_position=int_rank_pos,
                fp_vote=fp_vote,
                int_vote=int_vote,
                rank_match=rank_match,
                event_match=event_match,
                fractional_collision_mismatch=fractional_collision,
            )
        )
    return records


def obtain_banked_probe_t1_captures() -> dict[str, Any]:
    fixture = build_trainer_16x16_capture_fixture(seed=158)
    model = fixture.model
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=fixture.use_ternary_bulk,
        eligible_scope=fixture.eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    prior_training = bool(model.training)
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device="cpu",
            requires_grad=True,
        ) as handle:
            out = model(fixture.batch["x"])
            loss = F.mse_loss(out, fixture.batch["target"])
            if not bool(torch.isfinite(loss.detach()).item()):
                raise ValueError("T1 fixture requires finite loss")
            loss.backward()
            key = "proj"
            capture = handle.captures[key]
            state = states[key]
            return {
                "state_key": key,
                "inputs": capture["inputs"],
                "grad_outputs": capture["grad_outputs"],
                "weight_shape": tuple(int(dim) for dim in state.q_levels.shape),
                "q_levels_flat": state.q_levels.reshape(-1),
                "fixture_seed": 158,
            }
    finally:
        model.train(prior_training)


def measure_shift_key_at_capture(
    *,
    state_key: str,
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    spec: RankVoteSpec,
    rescale_shift: int,
) -> ShiftParityKeyResult:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    captures_present = len(inputs) > 0 and len(grad_outputs) > 0
    captures_finite = True
    if captures_present:
        for tensor in list(inputs) + list(grad_outputs):
            captures_finite = captures_finite and bool(torch.isfinite(tensor).all().item())
    weighted_grad = weighted_grad_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
    )
    fp_credit = credit_from_weighted_grad(weighted_grad)
    saturation_failed = False
    try:
        records = build_full_parity_records_with_shift(
            inputs=inputs,
            grad_outputs=grad_outputs,
            weight_shape=weight_dims,
            q_levels_flat=q_levels_flat,
            spec=spec,
            rescale_shift=int(rescale_shift),
        )
    except RescaleSaturationError:
        records = []
        saturation_failed = True
    rates = _compute_parity_rates_from_records(records)
    move_count = len(records)
    if move_count > 0:
        move_indices = torch.tensor([item.flat_index for item in records], dtype=torch.int64)
        fp_credit_sparse = fp_credit.reshape(-1).index_select(0, move_indices)
        int_rank = _rank_positions_for_credit_values(
            torch.tensor([item.int_credit_q31 for item in records], dtype=torch.float32),
            spec,
        )
        rank_group_count = _rank_group_count(int_rank)
    else:
        fp_credit_sparse = torch.empty(0, dtype=torch.float32)
        rank_group_count = 0
    valid, validity_detail = _measurement_validity_for_key(
        captures_present=captures_present,
        captures_finite=captures_finite,
        move_candidate_count=move_count,
        fp_credit_nonzero_count=_count_fp_credit_nonzero(fp_credit),
        fractional_diversity=_fractional_diversity_count(fp_credit_sparse),
        rank_group_count=rank_group_count,
    )
    if saturation_failed:
        valid = False
        validity_detail = dict(validity_detail)
        validity_detail["saturation_failed"] = True
    return ShiftParityKeyResult(
        state_key=state_key,
        measurement_valid=valid,
        validity_detail=validity_detail,
        move_candidate_count=move_count,
        rank_positions_match_rate=float(rates["rank_positions_match_rate"]),
        events_match_rate=float(rates["events_match_rate"]),
        fractional_collision_share_of_mismatches=float(
            rates["fractional_collision_share_of_mismatches"]
        ),
        saturation_failed=saturation_failed,
    )


def aggregate_tier_from_shift_parity_key_results(
    *,
    tier_id: str,
    candidate_id: str,
    rescale_shift: int,
    per_key_results: Sequence[ShiftParityKeyResult],
) -> TierShiftParityResult:
    saturation_fail_count = sum(1 for item in per_key_results if item.saturation_failed)
    valid_key_count = sum(1 for item in per_key_results if item.measurement_valid)
    total_move_candidates = sum(item.move_candidate_count for item in per_key_results)
    tier_measurement_valid = (
        len(per_key_results) > 0
        and valid_key_count >= 1
        and total_move_candidates >= MIN_TIER_TOTAL_MOVE_CANDIDATES
    )
    if tier_measurement_valid:
        valid_metrics = {
            item.state_key: KeyProbeMetrics(
                state_key=item.state_key,
                measurement_valid=True,
                validity_detail=dict(item.validity_detail),
                move_candidate_count=item.move_candidate_count,
                rank_positions_match_rate=item.rank_positions_match_rate,
                events_match_rate=item.events_match_rate,
                fractional_collision_share_of_mismatches=item.fractional_collision_share_of_mismatches,
                branch_id="shift_sweep",
            )
            for item in per_key_results
            if item.measurement_valid
        }
        aggregate = _aggregate_tier_rates(valid_metrics)
        rank_rate = float(aggregate["rank_positions_match_rate"])
        event_rate = float(aggregate["events_match_rate"])
        frac_rate = float(aggregate["fractional_collision_share_of_mismatches"])
    else:
        rank_rate = 0.0
        event_rate = 0.0
        frac_rate = 0.0
    verdict = classify_tier_parity_verdict(
        tier_id=tier_id,
        measurement_valid=tier_measurement_valid,
        rank_positions_match_rate=rank_rate,
        events_match_rate=event_rate,
        fractional_collision_share_of_mismatches=frac_rate,
    )
    return TierShiftParityResult(
        tier_id=tier_id,
        rescale_shift=int(rescale_shift),
        candidate_id=str(candidate_id),
        measurement_valid=tier_measurement_valid,
        parity_verdict=str(verdict.parity_verdict),
        rank_positions_match_rate=rank_rate,
        events_match_rate=event_rate,
        fractional_collision_share_of_mismatches=frac_rate,
        total_move_candidates=int(total_move_candidates),
        valid_key_count=int(valid_key_count),
        saturation_fail_count=int(saturation_fail_count),
        per_key_results=tuple(per_key_results),
    )


def evaluate_shift_candidate(
    capture: Mapping[str, Any],
    *,
    candidate_id: str,
    rescale_shift: int,
    spec: RankVoteSpec,
) -> SweepCandidateResult:
    try:
        records = build_full_parity_records_with_shift(
            inputs=capture["inputs"],
            grad_outputs=capture["grad_outputs"],
            weight_shape=capture["weight_shape"],
            q_levels_flat=capture["q_levels_flat"],
            spec=spec,
            rescale_shift=int(rescale_shift),
        )
        saturation_fail_count = 0
    except RescaleSaturationError:
        records = []
        saturation_fail_count = 1
    rates = _compute_parity_rates_from_records(records)
    move_count = len(records)
    mismatches = [item for item in records if not (item.rank_match and item.event_match)]
    frac_mismatches = [item for item in mismatches if item.fractional_collision_mismatch]
    measurement_valid = move_count >= MIN_TIER_TOTAL_MOVE_CANDIDATES and saturation_fail_count == 0
    verdict = classify_tier_parity_verdict(
        tier_id="T1_sweep",
        measurement_valid=measurement_valid,
        rank_positions_match_rate=rates["rank_positions_match_rate"],
        events_match_rate=rates["events_match_rate"],
        fractional_collision_share_of_mismatches=rates[
            "fractional_collision_share_of_mismatches"
        ],
    )
    return SweepCandidateResult(
        candidate_id=str(candidate_id),
        rescale_shift=int(rescale_shift),
        move_candidate_count=move_count,
        rank_positions_match_rate=float(rates["rank_positions_match_rate"]),
        events_match_rate=float(rates["events_match_rate"]),
        fractional_collision_share_of_mismatches=float(
            rates["fractional_collision_share_of_mismatches"]
        ),
        parity_verdict=str(verdict.parity_verdict),
        measurement_valid=bool(measurement_valid),
        rank_match_count=int(sum(1 for item in records if item.rank_match)),
        event_match_count=int(sum(1 for item in records if item.event_match)),
        mismatch_count=int(len(mismatches)),
        fractional_collision_mismatch_count=int(len(frac_mismatches)),
        saturation_fail_count=int(saturation_fail_count),
    )


def measure_shift_candidate_t2(
    bundle: Tier2RawCaptureBundle,
    *,
    candidate_id: str,
    rescale_shift: int,
    spec: RankVoteSpec,
) -> TierShiftParityResult:
    per_key: list[ShiftParityKeyResult] = []
    for key in sorted(bundle.per_key_captures.keys()):
        capture = bundle.per_key_captures[key]
        per_key.append(
            measure_shift_key_at_capture(
                state_key=key,
                inputs=capture.inputs,
                grad_outputs=capture.grad_outputs,
                weight_shape=capture.weight_shape,
                q_levels_flat=capture.q_levels_flat,
                spec=spec,
                rescale_shift=int(rescale_shift),
            )
        )
    return aggregate_tier_from_shift_parity_key_results(
        tier_id="T2",
        candidate_id=candidate_id,
        rescale_shift=int(rescale_shift),
        per_key_results=per_key,
    )


def apply_v3_selector(
    results_by_shift: Mapping[int, SweepCandidateResult],
) -> dict[str, Any]:
    clearing_narrow: list[int] = []
    clearing_broad: list[int] = []
    for shift in SELECTOR_SHIFTS_COARSEST_FIRST:
        result = results_by_shift.get(int(shift))
        if result is None:
            continue
        if result.parity_verdict in {VERDICT_NARROW_HOLDS, VERDICT_BROAD_HOLDS}:
            clearing_narrow.append(int(shift))
        if result.parity_verdict == VERDICT_BROAD_HOLDS:
            clearing_broad.append(int(shift))
    if not clearing_narrow:
        return {
            "selector_outcome": SELECTOR_OUTCOME_STOP_TIER3_INT64_ROUTE,
            "selected_shift": None,
            "selected_candidate_id": None,
            "clears_narrow_holds": False,
            "clears_broad_holds": False,
            "rationale": (
                "No int32 shift in {24,16,8} cleared narrow_holds on full-256 T1; "
                "do not default-pin any shift."
            ),
        }
    selected_shift = max(clearing_narrow)
    candidate_id = {
        24: "rescale_q24",
        16: "rescale_q16",
        8: "rescale_q8",
    }[selected_shift]
    return {
        "selector_outcome": SELECTOR_OUTCOME_PIN_CANDIDATE_REPORT_ONLY,
        "selected_shift": int(selected_shift),
        "selected_candidate_id": candidate_id,
        "clears_narrow_holds": True,
        "clears_broad_holds": bool(clearing_broad),
        "coarsest_that_clears_shift": int(selected_shift),
        "rationale": (
            "Coarsest/highest-headroom shift among {24,16,8} clearing narrow_holds "
            "on full-256 T1 (production pin deferred to next +1)."
        ),
    }


def _shift_to_candidate_id(shift: int) -> str:
    return {24: "rescale_q24", 16: "rescale_q16", 8: "rescale_q8", 31: "law_v0"}[int(shift)]


def apply_valid_t2_selector(
    dual_results: Mapping[int, DualTierCandidateResult],
) -> dict[str, Any]:
    feasible: list[int] = []
    t2_valid_any = False
    t2_narrow_any = False
    for shift in SELECTOR_SHIFTS_FINER_CANDIDATES:
        item = dual_results.get(int(shift))
        if item is None:
            continue
        t1_broad = item.t1.parity_verdict == VERDICT_BROAD_HOLDS
        saturation_ok = item.t1.saturation_fail_count == 0 and item.t2.saturation_fail_count == 0
        if item.t2.measurement_valid:
            t2_valid_any = True
        if item.t2.parity_verdict in {VERDICT_NARROW_HOLDS, VERDICT_BROAD_HOLDS}:
            t2_narrow_any = True
        if t1_broad and item.t2.measurement_valid and saturation_ok:
            feasible.append(int(shift))

    if not feasible:
        if t2_valid_any and not t2_narrow_any:
            return {
                "selector_outcome": SELECTOR_OUTCOME_STOP_RANK_STRUCTURE,
                "selected_shift": None,
                "selected_candidate_id": None,
                "routing_strength": ROUTING_STRENGTH_NONE,
                "rationale": "T2 moves recovered but no shift clears narrow_holds (rank-structure failure).",
            }
        return {
            "selector_outcome": SELECTOR_OUTCOME_STOP_NO_SHIFT_CLEARS_VALID_T2,
            "selected_shift": None,
            "selected_candidate_id": None,
            "routing_strength": ROUTING_STRENGTH_NONE,
            "rationale": "No feasible shift clears T1 broad_holds + T2 measurement_valid + saturation.",
        }

    t2_broad = [
        shift
        for shift in feasible
        if dual_results[shift].t2.parity_verdict == VERDICT_BROAD_HOLDS
    ]
    if t2_broad:
        selected = max(t2_broad)
        return {
            "selector_outcome": SELECTOR_OUTCOME_PIN_COARSEST_T2_BROAD,
            "selected_shift": int(selected),
            "selected_candidate_id": _shift_to_candidate_id(selected),
            "routing_strength": ROUTING_STRENGTH_CLEAN,
            "rationale": "Coarsest shift among T2-broad-clearing feasible set (prefer-broad-then-coarsest).",
        }

    t2_narrow_only = [
        shift
        for shift in feasible
        if dual_results[shift].t2.parity_verdict in {VERDICT_NARROW_HOLDS, VERDICT_BROAD_HOLDS}
        and dual_results[shift].t2.parity_verdict != VERDICT_BROAD_HOLDS
    ]
    if t2_narrow_only:
        selected = max(t2_narrow_only)
        return {
            "selector_outcome": SELECTOR_OUTCOME_PIN_COARSEST_T2_NARROW_CAVEATED,
            "selected_shift": int(selected),
            "selected_candidate_id": _shift_to_candidate_id(selected),
            "routing_strength": ROUTING_STRENGTH_CAVEATED,
            "rationale": "Coarsest shift among T2-narrow-only feasible set (caveated routing).",
        }

    return {
        "selector_outcome": SELECTOR_OUTCOME_STOP_RANK_STRUCTURE,
        "selected_shift": None,
        "selected_candidate_id": None,
        "routing_strength": ROUTING_STRENGTH_NONE,
        "rationale": "Feasible shifts fail T2 narrow_holds.",
    }


def run_readonly_sweep(*, repo_root: Path | None = None) -> dict[str, Any]:
    module_path = integer_marginal_attribution_module_path(repo_root)
    sha_before = sha256_file(str(module_path))
    capture = obtain_banked_probe_t1_captures()
    spec = default_dry_run_rank_vote_spec()
    results: list[SweepCandidateResult] = []
    results_by_shift: dict[int, SweepCandidateResult] = {}
    for candidate_id, shift in ALL_SWEEP_CANDIDATES:
        result = evaluate_shift_candidate(
            capture,
            candidate_id=candidate_id,
            rescale_shift=int(shift),
            spec=spec,
        )
        results.append(result)
        if candidate_id != "law_v0":
            results_by_shift[int(shift)] = result
    sha_after = sha256_file(str(module_path))
    if sha_before != sha_after:
        raise RuntimeError("integer_marginal_attribution.py changed during read-only sweep")
    selector = apply_v3_selector(results_by_shift)
    return {
        "schema": RESCALE_LAW_READONLY_SWEEP_SCHEMA_V1,
        "pass_receipt": False,
        "fixture": {
            "probe_t1_bitlinear_16x16": True,
            "fixture_seed": int(capture["fixture_seed"]),
            "state_key": str(capture["state_key"]),
            "weight_shape": list(capture["weight_shape"]),
        },
        "integer_marginal_attribution_py_sha256_before": sha_before,
        "integer_marginal_attribution_py_sha256_after": sha_after,
        "production_module_unchanged": bool(sha_before == sha_after),
        "candidate_results": [item.to_dict() for item in results],
        "selector": selector,
        "frozen_thresholds": {
            "narrow_holds": {
                "rank_positions_match_rate_min": 0.80,
                "events_match_rate_min": 0.70,
            },
            "broad_holds": {
                "rank_positions_match_rate_min": 0.95,
                "events_match_rate_min": 0.90,
                "fractional_collision_share_max": 0.20,
            },
        },
    }


def run_dual_tier_readonly_sweep(
    *,
    checkpoint_path: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    module_path = integer_marginal_attribution_module_path(repo_root)
    sha_before = sha256_file(str(module_path))
    t1_capture = obtain_banked_probe_t1_captures()
    t2_bundle = capture_tier2_checkpoint_raw_captures(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
        curriculum_seed=FROZEN_T2_ANCHOR_CURRICULUM_SEED,
        batch_size=FROZEN_T2_ANCHOR_BATCH_SIZE,
    )
    spec = default_dry_run_rank_vote_spec()
    dual_results: list[DualTierCandidateResult] = []
    dual_by_shift: dict[int, DualTierCandidateResult] = {}
    for candidate_id, shift in ALL_SWEEP_CANDIDATES:
        t1 = evaluate_shift_candidate(
            t1_capture,
            candidate_id=candidate_id,
            rescale_shift=int(shift),
            spec=spec,
        )
        t2 = measure_shift_candidate_t2(
            t2_bundle,
            candidate_id=candidate_id,
            rescale_shift=int(shift),
            spec=spec,
        )
        item = DualTierCandidateResult(
            candidate_id=str(candidate_id),
            rescale_shift=int(shift),
            t1=t1,
            t2=t2,
        )
        dual_results.append(item)
        dual_by_shift[int(shift)] = item
    sha_after = sha256_file(str(module_path))
    if sha_before != sha_after:
        raise RuntimeError("integer_marginal_attribution.py changed during dual-tier sweep")
    selector = apply_valid_t2_selector(dual_by_shift)
    return {
        "schema": DUAL_TIER_SWEEP_SCHEMA_V1,
        "pass_receipt": False,
        "integer_marginal_attribution_py_sha256_before": sha_before,
        "integer_marginal_attribution_py_sha256_after": sha_after,
        "production_module_unchanged": bool(sha_before == sha_after),
        "anchor_checkpoint_sha256": FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
        "candidate_results": [item.to_dict() for item in dual_results],
        "selector": selector,
    }
