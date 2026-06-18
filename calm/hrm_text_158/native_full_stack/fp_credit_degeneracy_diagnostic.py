"""Read-only FP credit degeneracy grounding diagnostic for HRM-Text-1.58 Step 3C-A.

Multi-pool characterization of FP credit magnitude, tie structure, rank-fraction,
and vote-class balance on anchored real-HRM T2 captures. Parity evidence only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    RankVoteSpec,
    _bisect_right_rank_positions_by_equal_value_group,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    PerCandidateParityRecord,
    Tier2RawCaptureBundle,
    capture_tier2_checkpoint_raw_captures,
    realistic_gradient_parity_probe_hard_false_snapshot,
    sha256_file,
)
from calm.hrm_text_158.native_full_stack.rescale_law_readonly_sweep import (
    RescaleSaturationError,
    build_full_parity_records_with_shift,
    measure_shift_key_at_capture,
)
from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    FROZEN_T2_ANCHOR_BATCH_SIZE,
    FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    FROZEN_T2_ANCHOR_KEY_SET_SHA256,
    T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
    anchor_key_set_sha256,
)

FP_CREDIT_DEGENERACY_SCHEMA_V2 = "hrm_text_158_fp_credit_degeneracy/v2"
FP_CREDIT_DEGENERACY_TARGET_NAME = "fp_credit_degeneracy_ground"
FP_CREDIT_DEGENERACY_DESIGN_PACKET_SHA256 = (
    "ef0e1fa96c4d29f6796fd4930b4416924ee2ce5cbf5ad0be179f0ed970903ad8"
)

BR_FP_DEG_POOL_SELECTION_CONFOUND = "BR-FP-DEG-POOL-SELECTION-CONFOUND"
BR_FP_DEG_AMPLIFIED_FIXABLE = "BR-FP-DEG-AMPLIFIED-FIXABLE"
BR_FP_DEG_NO_PARITY_ENRICHMENT = "BR-FP-DEG-NO-PARITY-ENRICHMENT"
BR_FP_DEG_MEASUREMENT_INVALID = "BR-FP-DEG-MEASUREMENT-INVALID"
BR_FP_DEG_UNRESOLVED = "BR-FP-DEG-UNRESOLVED"

REGISTERED_BRANCH_IDS = (
    BR_FP_DEG_POOL_SELECTION_CONFOUND,
    BR_FP_DEG_AMPLIFIED_FIXABLE,
    BR_FP_DEG_NO_PARITY_ENRICHMENT,
    BR_FP_DEG_MEASUREMENT_INVALID,
    BR_FP_DEG_UNRESOLVED,
)

POOL_CONFOUND_PARITY_MIN = 0.85
PARITY_ENRICHMENT_MIN = 0.35
BY_CONSTRUCTION_ANCHOR_LO = 0.45
BY_CONSTRUCTION_ANCHOR_HI = 0.55
PARITY_LOOKUP_EPS = 1e-4
MIN_POOL_CANDIDATES = 10_000
PARITY_RESCALE_SHIFT = 16
HISTOGRAM_BIN_COUNT = 8

FP_CREDIT_DEGENERACY_NON_CLAIMS = (
    "fp credit degeneracy diagnostic is CPU read-only parity evidence only",
    "pass_receipt is always false; no production re-pin or 3C-GPU claim",
    "branch_id routes diagnostic interpretation only; not optimizer viability",
    "coverage metric bounds high-pass confound interpretation; not a classifier gate",
    "no vote-spec reopen, production attribution-law change, or GPU runtime",
)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _vote_abs(vote: int) -> int:
    return abs(int(vote))


@dataclass(frozen=True)
class FpCreditPoolMetrics:
    pool_id: str
    module_key: str | None
    move_candidates: int
    fp_vote4_count: int
    fp_vote1_count: int
    fp_vote0_count: int
    rank_rf_ge_0p5_count: int
    credit_min: float | None
    credit_p25: float | None
    credit_median: float | None
    credit_p75: float | None
    credit_max: float | None
    distinct_abs_groups: int
    largest_tie_group_size: int
    median_tie_group_size: float | None
    histogram_bins: tuple[int, ...]

    @property
    def fp_vote4_fraction(self) -> float | None:
        return _safe_rate(self.fp_vote4_count, self.move_candidates)

    @property
    def fp_vote1_fraction(self) -> float | None:
        return _safe_rate(self.fp_vote1_count, self.move_candidates)

    @property
    def fp_vote0_fraction(self) -> float | None:
        return _safe_rate(self.fp_vote0_count, self.move_candidates)

    @property
    def fraction_rf_ge_0p5(self) -> float | None:
        return _safe_rate(self.rank_rf_ge_0p5_count, self.move_candidates)

    @property
    def distinct_abs_groups_per_n(self) -> float | None:
        return _safe_rate(self.distinct_abs_groups, self.move_candidates)

    @property
    def largest_tie_group_fraction(self) -> float | None:
        return _safe_rate(self.largest_tie_group_size, self.move_candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "module_key": self.module_key,
            "move_candidates": int(self.move_candidates),
            "fp_vote4_count": int(self.fp_vote4_count),
            "fp_vote1_count": int(self.fp_vote1_count),
            "fp_vote0_count": int(self.fp_vote0_count),
            "rank_rf_ge_0p5_count": int(self.rank_rf_ge_0p5_count),
            "fp_vote4_fraction": self.fp_vote4_fraction,
            "fp_vote1_fraction": self.fp_vote1_fraction,
            "fp_vote0_fraction": self.fp_vote0_fraction,
            "fraction_rf_ge_0p5": self.fraction_rf_ge_0p5,
            "credit_min": self.credit_min,
            "credit_p25": self.credit_p25,
            "credit_median": self.credit_median,
            "credit_p75": self.credit_p75,
            "credit_max": self.credit_max,
            "distinct_abs_groups": int(self.distinct_abs_groups),
            "distinct_abs_groups_per_n": self.distinct_abs_groups_per_n,
            "largest_tie_group_size": int(self.largest_tie_group_size),
            "largest_tie_group_fraction": self.largest_tie_group_fraction,
            "median_tie_group_size": self.median_tie_group_size,
            "histogram_bins": list(self.histogram_bins),
        }


@dataclass(frozen=True)
class ParityEnrichmentMetrics:
    full_pool_fp4_fraction: float | None
    parity_fp4_balance: float | None
    full_pool_fp4_at_parity_indices: float | None
    parity_index_enrichment_over_full_pool: float | None
    integer_move_coverage_vs_full_fp: float | None
    parity_move_candidates: int
    full_pool_move_candidates: int
    consistency_anchor_pass: bool
    consistency_anchor_delta: float | None
    per_key_coverage: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_pool_fp4_fraction": self.full_pool_fp4_fraction,
            "parity_fp4_balance": self.parity_fp4_balance,
            "full_pool_fp4_at_parity_indices": self.full_pool_fp4_at_parity_indices,
            "parity_index_enrichment_over_full_pool": self.parity_index_enrichment_over_full_pool,
            "integer_move_coverage_vs_full_fp": self.integer_move_coverage_vs_full_fp,
            "parity_move_candidates": int(self.parity_move_candidates),
            "full_pool_move_candidates": int(self.full_pool_move_candidates),
            "consistency_anchor_pass": bool(self.consistency_anchor_pass),
            "consistency_anchor_delta": self.consistency_anchor_delta,
            "per_key_coverage": list(self.per_key_coverage),
        }


@dataclass(frozen=True)
class KeyFilterAudit:
    captured_key_count: int
    full_pool_key_count: int
    parity_valid_key_count: int
    parity_excluded_keys: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_key_count": int(self.captured_key_count),
            "full_pool_key_count": int(self.full_pool_key_count),
            "parity_valid_key_count": int(self.parity_valid_key_count),
            "parity_excluded_keys": list(self.parity_excluded_keys),
        }


@dataclass(frozen=True)
class FpCreditDegeneracyReceipt:
    schema: str
    target_name: str
    pass_receipt: bool
    hard_false: dict[str, bool]
    non_claims: tuple[str, ...]
    branch_id: str
    design_packet_sha256: str
    anchor: dict[str, Any]
    production_law_id: str
    full_pool: FpCreditPoolMetrics
    per_key_full_pool: tuple[FpCreditPoolMetrics, ...]
    parity_pool: FpCreditPoolMetrics
    per_key_parity_pool: tuple[FpCreditPoolMetrics, ...]
    enrichment: ParityEnrichmentMetrics
    key_filter_audit: KeyFilterAudit
    full_pool_fp4_anchor_in_band: bool
    intrinsic_via_full_pool_fp4_structurally_unreachable_under_locked_spec: bool
    bundle_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_name": self.target_name,
            "pass_receipt": bool(self.pass_receipt),
            "hard_false": dict(self.hard_false),
            "non_claims": list(self.non_claims),
            "branch_id": self.branch_id,
            "design_packet_sha256": self.design_packet_sha256,
            "anchor": dict(self.anchor),
            "production_law_id": self.production_law_id,
            "full_pool": self.full_pool.to_dict(),
            "per_key_full_pool": [item.to_dict() for item in self.per_key_full_pool],
            "parity_pool": self.parity_pool.to_dict(),
            "per_key_parity_pool": [item.to_dict() for item in self.per_key_parity_pool],
            "enrichment": self.enrichment.to_dict(),
            "key_filter_audit": self.key_filter_audit.to_dict(),
            "full_pool_fp4_fraction": self.enrichment.full_pool_fp4_fraction,
            "full_pool_fp4_anchor_in_band": bool(self.full_pool_fp4_anchor_in_band),
            "intrinsic_via_full_pool_fp4_structurally_unreachable_under_locked_spec": bool(
                self.intrinsic_via_full_pool_fp4_structurally_unreachable_under_locked_spec
            ),
            "bundle_provenance": dict(self.bundle_provenance),
        }


def _credit_quantiles(abs_values: torch.Tensor) -> tuple[float | None, ...]:
    if int(abs_values.numel()) == 0:
        return (None, None, None, None, None)
    sorted_vals, _ = torch.sort(abs_values)
    n = int(sorted_vals.numel())

    def _q(frac: float) -> float:
        if n == 1:
            return float(sorted_vals[0].item())
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return float(sorted_vals[idx].item())

    return (
        float(sorted_vals[0].item()),
        _q(0.25),
        _q(0.50),
        _q(0.75),
        float(sorted_vals[-1].item()),
    )


def _tie_structure(abs_values: torch.Tensor) -> tuple[int, int, float | None]:
    n = int(abs_values.numel())
    if n == 0:
        return 0, 0, None
    abs_bits = abs_values.contiguous().view(torch.int32)
    sorted_bits, _ = torch.sort(abs_bits)
    group_start = torch.ones(n, dtype=torch.bool)
    group_start[1:] = sorted_bits[1:] != sorted_bits[:-1]
    group_id = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    distinct_groups = int(group_id.max().item()) + 1
    group_sizes = torch.bincount(group_id, minlength=distinct_groups)
    largest = int(group_sizes.max().item())
    median_size = float(torch.median(group_sizes.to(torch.float32)).item())
    return distinct_groups, largest, median_size


def _log_histogram(abs_values: torch.Tensor, *, bin_count: int = HISTOGRAM_BIN_COUNT) -> tuple[int, ...]:
    n = int(abs_values.numel())
    if n == 0:
        return tuple(0 for _ in range(bin_count))
    max_val = float(abs_values.max().item())
    if max_val <= 0.0:
        counts = [0] * bin_count
        counts[0] = n
        return tuple(counts)
    log_max = math.log10(max_val)
    log_min = math.log10(max(max_val * 1e-12, float(abs_values[abs_values > 0].min().item()) if bool((abs_values > 0).any().item()) else max_val * 1e-12))
    if log_max <= log_min:
        counts = [0] * bin_count
        counts[-1] = n
        return tuple(counts)
    edges = [log_min + (log_max - log_min) * (idx + 1) / bin_count for idx in range(bin_count)]
    counts = [0] * bin_count
    positive = abs_values[abs_values > 0]
    if int(positive.numel()) == 0:
        counts[0] = n
        return tuple(counts)
    log_vals = torch.log10(positive)
    for val in log_vals.tolist():
        placed = False
        for idx, edge in enumerate(edges):
            if val <= edge:
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    zero_count = n - int(positive.numel())
    counts[0] += zero_count
    return tuple(counts)


def characterize_fp_credit_pool(
    *,
    pool_id: str,
    module_key: str | None,
    fp_credit: torch.Tensor,
    fp_moves: torch.Tensor,
    spec: RankVoteSpec,
) -> FpCreditPoolMetrics:
    """Vectorized per-key (or pooled) FP move-candidate pool characterization."""

    spec.validate()
    flat_credit = fp_credit.detach().reshape(-1).to(torch.float32)
    flat_moves = fp_moves.detach().reshape(-1).to(torch.int8)
    if flat_credit.numel() != flat_moves.numel():
        raise ValueError("fp_credit and fp_moves shape mismatch")
    candidate_idx = torch.nonzero(flat_moves != 0, as_tuple=False).flatten()
    n = int(candidate_idx.numel())
    if n == 0:
        return FpCreditPoolMetrics(
            pool_id=pool_id,
            module_key=module_key,
            move_candidates=0,
            fp_vote4_count=0,
            fp_vote1_count=0,
            fp_vote0_count=0,
            rank_rf_ge_0p5_count=0,
            credit_min=None,
            credit_p25=None,
            credit_median=None,
            credit_p75=None,
            credit_max=None,
            distinct_abs_groups=0,
            largest_tie_group_size=0,
            median_tie_group_size=None,
            histogram_bins=tuple(0 for _ in range(HISTOGRAM_BIN_COUNT)),
        )
    abs_values = flat_credit.index_select(0, candidate_idx).abs()
    rank_positions = _bisect_right_rank_positions_by_equal_value_group(abs_values)
    rf_ge = int((rank_positions.to(torch.float32) / float(n) >= 0.5).sum().item())
    fp_votes_dense = rank_bucketed_int16_votes(fp_credit, fp_moves, spec).reshape(-1)
    votes_at_candidates = fp_votes_dense.index_select(0, candidate_idx).to(torch.int16)
    vote_abs = votes_at_candidates.abs()
    vote4 = int((vote_abs == 4).sum().item())
    vote1 = int((vote_abs == 1).sum().item())
    vote0 = int((vote_abs == 0).sum().item())
    cmin, cp25, cmed, cp75, cmax = _credit_quantiles(abs_values)
    distinct, largest_tie, median_tie = _tie_structure(abs_values)
    hist = _log_histogram(abs_values)
    return FpCreditPoolMetrics(
        pool_id=pool_id,
        module_key=module_key,
        move_candidates=n,
        fp_vote4_count=vote4,
        fp_vote1_count=vote1,
        fp_vote0_count=vote0,
        rank_rf_ge_0p5_count=rf_ge,
        credit_min=cmin,
        credit_p25=cp25,
        credit_median=cmed,
        credit_p75=cp75,
        credit_max=cmax,
        distinct_abs_groups=distinct,
        largest_tie_group_size=largest_tie,
        median_tie_group_size=median_tie,
        histogram_bins=hist,
    )


def _aggregate_pool_metrics(
    *,
    pool_id: str,
    per_key: Sequence[FpCreditPoolMetrics],
) -> FpCreditPoolMetrics:
    move_candidates = sum(item.move_candidates for item in per_key)
    if move_candidates == 0:
        return FpCreditPoolMetrics(
            pool_id=pool_id,
            module_key=None,
            move_candidates=0,
            fp_vote4_count=0,
            fp_vote1_count=0,
            fp_vote0_count=0,
            rank_rf_ge_0p5_count=0,
            credit_min=None,
            credit_p25=None,
            credit_median=None,
            credit_p75=None,
            credit_max=None,
            distinct_abs_groups=0,
            largest_tie_group_size=0,
            median_tie_group_size=None,
            histogram_bins=tuple(0 for _ in range(HISTOGRAM_BIN_COUNT)),
        )
    mins = [item.credit_min for item in per_key if item.credit_min is not None]
    maxs = [item.credit_max for item in per_key if item.credit_max is not None]
    hist = [0] * HISTOGRAM_BIN_COUNT
    for item in per_key:
        for idx, count in enumerate(item.histogram_bins):
            hist[idx] += int(count)
    weighted_median_num = 0.0
    weighted_median_den = 0
    for item in per_key:
        if item.credit_median is not None and item.move_candidates > 0:
            weighted_median_num += float(item.credit_median) * item.move_candidates
            weighted_median_den += item.move_candidates
    return FpCreditPoolMetrics(
        pool_id=pool_id,
        module_key=None,
        move_candidates=move_candidates,
        fp_vote4_count=sum(item.fp_vote4_count for item in per_key),
        fp_vote1_count=sum(item.fp_vote1_count for item in per_key),
        fp_vote0_count=sum(item.fp_vote0_count for item in per_key),
        rank_rf_ge_0p5_count=sum(item.rank_rf_ge_0p5_count for item in per_key),
        credit_min=min(mins) if mins else None,
        credit_p25=None,
        credit_median=(weighted_median_num / weighted_median_den) if weighted_median_den else None,
        credit_p75=None,
        credit_max=max(maxs) if maxs else None,
        distinct_abs_groups=sum(item.distinct_abs_groups for item in per_key),
        largest_tie_group_size=max((item.largest_tie_group_size for item in per_key), default=0),
        median_tie_group_size=None,
        histogram_bins=tuple(hist),
    )


def _metrics_from_parity_records(
    *,
    pool_id: str,
    module_key: str | None,
    records: Sequence[PerCandidateParityRecord],
) -> FpCreditPoolMetrics:
    n = len(records)
    if n == 0:
        return characterize_fp_credit_pool(
            pool_id=pool_id,
            module_key=module_key,
            fp_credit=torch.zeros(1, dtype=torch.float32),
            fp_moves=torch.zeros(1, dtype=torch.int8),
            spec=default_dry_run_rank_vote_spec(),
        )
    credits = torch.tensor([float(record.fp_credit) for record in records], dtype=torch.float32)
    votes = torch.tensor([int(record.fp_vote) for record in records], dtype=torch.int16)
    abs_values = credits.abs()
    rank_positions = torch.tensor(
        [int(record.fp_rank_position) for record in records],
        dtype=torch.int64,
    )
    rf_ge = int((rank_positions.to(torch.float32) / float(n) >= 0.5).sum().item())
    vote_abs = votes.abs()
    vote4 = int((vote_abs == 4).sum().item())
    vote1 = int((vote_abs == 1).sum().item())
    vote0 = int((vote_abs == 0).sum().item())
    cmin, cp25, cmed, cp75, cmax = _credit_quantiles(abs_values)
    distinct, largest_tie, median_tie = _tie_structure(abs_values)
    hist = _log_histogram(abs_values)
    return FpCreditPoolMetrics(
        pool_id=pool_id,
        module_key=module_key,
        move_candidates=n,
        fp_vote4_count=vote4,
        fp_vote1_count=vote1,
        fp_vote0_count=vote0,
        rank_rf_ge_0p5_count=rf_ge,
        credit_min=cmin,
        credit_p25=cp25,
        credit_median=cmed,
        credit_p75=cp75,
        credit_max=cmax,
        distinct_abs_groups=distinct,
        largest_tie_group_size=largest_tie,
        median_tie_group_size=median_tie,
        histogram_bins=hist,
    )


def compute_parity_enrichment_metrics(
    *,
    full_pool: FpCreditPoolMetrics,
    parity_pool: FpCreditPoolMetrics,
    per_key_full: Sequence[FpCreditPoolMetrics],
    per_key_parity: Sequence[FpCreditPoolMetrics],
    parity_records_by_key: Mapping[str, Sequence[PerCandidateParityRecord]],
    dense_fp_votes_by_key: Mapping[str, torch.Tensor],
) -> ParityEnrichmentMetrics:
    full_pool_fp4 = full_pool.fp_vote4_fraction
    parity_fp4 = parity_pool.fp_vote4_fraction
    parity_at_indices_vote4 = 0
    parity_at_indices_total = 0
    for key, records in parity_records_by_key.items():
        dense_votes = dense_fp_votes_by_key[key].reshape(-1)
        for record in records:
            vote = int(dense_votes[int(record.flat_index)].item())
            parity_at_indices_total += 1
            if _vote_abs(vote) == 4:
                parity_at_indices_vote4 += 1
    full_pool_fp4_at_parity = _safe_rate(parity_at_indices_vote4, parity_at_indices_total)
    enrichment = None
    if parity_fp4 is not None and full_pool_fp4 is not None:
        enrichment = float(parity_fp4) - float(full_pool_fp4)
    coverage = _safe_rate(parity_pool.move_candidates, full_pool.move_candidates)
    delta = None
    if parity_fp4 is not None and full_pool_fp4_at_parity is not None:
        delta = abs(float(parity_fp4) - float(full_pool_fp4_at_parity))
    consistency_pass = (
        delta is not None
        and parity_at_indices_total > 0
        and delta <= PARITY_LOOKUP_EPS
    )
    per_key_coverage: list[dict[str, Any]] = []
    full_by_key = {item.module_key: item for item in per_key_full if item.module_key is not None}
    parity_by_key = {item.module_key: item for item in per_key_parity if item.module_key is not None}
    for key in sorted(full_by_key.keys()):
        full_item = full_by_key[key]
        parity_item = parity_by_key.get(key)
        parity_n = parity_item.move_candidates if parity_item is not None else 0
        per_key_coverage.append(
            {
                "module_key": key,
                "full_pool_move_candidates": int(full_item.move_candidates),
                "parity_move_candidates": int(parity_n),
                "integer_move_coverage_vs_full_fp": _safe_rate(parity_n, full_item.move_candidates),
            }
        )
    return ParityEnrichmentMetrics(
        full_pool_fp4_fraction=full_pool_fp4,
        parity_fp4_balance=parity_fp4,
        full_pool_fp4_at_parity_indices=full_pool_fp4_at_parity,
        parity_index_enrichment_over_full_pool=enrichment,
        integer_move_coverage_vs_full_fp=coverage,
        parity_move_candidates=int(parity_pool.move_candidates),
        full_pool_move_candidates=int(full_pool.move_candidates),
        consistency_anchor_pass=bool(consistency_pass),
        consistency_anchor_delta=delta,
        per_key_coverage=tuple(per_key_coverage),
    )


def full_pool_fp4_anchor_in_band(full_pool_fp4_fraction: float | None) -> bool:
    if full_pool_fp4_fraction is None:
        return False
    return (
        float(full_pool_fp4_fraction) >= BY_CONSTRUCTION_ANCHOR_LO
        and float(full_pool_fp4_fraction) <= BY_CONSTRUCTION_ANCHOR_HI
    )


def classify_fp_credit_degeneracy_branch(
    *,
    full_pool: FpCreditPoolMetrics,
    enrichment: ParityEnrichmentMetrics,
) -> str:
    full_vote4 = full_pool.fp_vote4_fraction
    parity_fp4 = enrichment.parity_fp4_balance
    enrichment_delta = enrichment.parity_index_enrichment_over_full_pool
    anchor_in_band = full_pool_fp4_anchor_in_band(full_vote4)

    if full_pool.move_candidates < MIN_POOL_CANDIDATES:
        return BR_FP_DEG_MEASUREMENT_INVALID
    if not enrichment.consistency_anchor_pass:
        return BR_FP_DEG_MEASUREMENT_INVALID
    if not anchor_in_band:
        return BR_FP_DEG_MEASUREMENT_INVALID

    if (
        parity_fp4 is not None
        and enrichment_delta is not None
        and parity_fp4 >= POOL_CONFOUND_PARITY_MIN
        and enrichment_delta >= PARITY_ENRICHMENT_MIN
    ):
        return BR_FP_DEG_POOL_SELECTION_CONFOUND

    if (
        parity_fp4 is not None
        and enrichment_delta is not None
        and parity_fp4 >= POOL_CONFOUND_PARITY_MIN
        and enrichment_delta < PARITY_ENRICHMENT_MIN
    ):
        return BR_FP_DEG_AMPLIFIED_FIXABLE

    if parity_fp4 is not None and parity_fp4 < POOL_CONFOUND_PARITY_MIN:
        return BR_FP_DEG_NO_PARITY_ENRICHMENT

    return BR_FP_DEG_UNRESOLVED


def validate_fp_credit_degeneracy_receipt(receipt: FpCreditDegeneracyReceipt) -> None:
    if receipt.schema != FP_CREDIT_DEGENERACY_SCHEMA_V2:
        raise ValueError("schema mismatch on fp credit degeneracy receipt")
    if receipt.target_name != FP_CREDIT_DEGENERACY_TARGET_NAME:
        raise ValueError("target_name mismatch on fp credit degeneracy receipt")
    if receipt.pass_receipt:
        raise ValueError("pass_receipt must remain false")
    for field_name, value in receipt.hard_false.items():
        if bool(value):
            raise ValueError(f"{field_name} must remain false on fp credit degeneracy receipt")
    if receipt.non_claims != FP_CREDIT_DEGENERACY_NON_CLAIMS:
        raise ValueError("non_claims must be exact on fp credit degeneracy receipt")
    if receipt.branch_id not in REGISTERED_BRANCH_IDS:
        raise ValueError(f"unknown branch_id {receipt.branch_id}")
    if receipt.design_packet_sha256 != FP_CREDIT_DEGENERACY_DESIGN_PACKET_SHA256:
        raise ValueError("design_packet_sha256 mismatch")


def _collect_key_metrics(
    bundle: Tier2RawCaptureBundle,
    *,
    spec: RankVoteSpec,
    rescale_shift: int,
) -> tuple[
    list[FpCreditPoolMetrics],
    list[FpCreditPoolMetrics],
    dict[str, list[PerCandidateParityRecord]],
    dict[str, torch.Tensor],
    KeyFilterAudit,
]:
    per_key_full: list[FpCreditPoolMetrics] = []
    per_key_parity: list[FpCreditPoolMetrics] = []
    parity_records_by_key: dict[str, list[PerCandidateParityRecord]] = {}
    dense_fp_votes_by_key: dict[str, torch.Tensor] = {}
    parity_excluded: list[dict[str, str]] = []
    captured_keys = sorted(bundle.per_key_captures.keys())
    for key in captured_keys:
        capture = bundle.per_key_captures[key]
        weighted_grad = weighted_grad_from_captures(
            capture.inputs,
            capture.grad_outputs,
            weight_shape=capture.weight_shape,
        )
        fp_credit = credit_from_weighted_grad(weighted_grad)
        q_levels = capture.q_levels_flat.reshape(capture.weight_shape)
        fp_moves = project_s1_gradient_to_moves(weighted_grad, q_levels)
        per_key_full.append(
            characterize_fp_credit_pool(
                pool_id="TRUE_FULL_FP_MOVE_POOL",
                module_key=key,
                fp_credit=fp_credit,
                fp_moves=fp_moves,
                spec=spec,
            )
        )
        key_result = measure_shift_key_at_capture(
            state_key=key,
            inputs=capture.inputs,
            grad_outputs=capture.grad_outputs,
            weight_shape=capture.weight_shape,
            q_levels_flat=capture.q_levels_flat,
            spec=spec,
            rescale_shift=int(rescale_shift),
        )
        if key_result.saturation_failed:
            parity_excluded.append({"module_key": key, "reason": "saturation_failed"})
            continue
        if not key_result.measurement_valid:
            parity_excluded.append({"module_key": key, "reason": "measurement_invalid"})
            continue
        try:
            records = build_full_parity_records_with_shift(
                inputs=capture.inputs,
                grad_outputs=capture.grad_outputs,
                weight_shape=capture.weight_shape,
                q_levels_flat=capture.q_levels_flat,
                spec=spec,
                rescale_shift=int(rescale_shift),
            )
        except RescaleSaturationError:
            parity_excluded.append({"module_key": key, "reason": "rescale_saturation"})
            continue
        fp_votes_dense = rank_bucketed_int16_votes(fp_credit, fp_moves, spec)
        dense_fp_votes_by_key[key] = fp_votes_dense
        parity_records_by_key[key] = records
        per_key_parity.append(
            _metrics_from_parity_records(
                pool_id="PARITY_INTEGER_FILTERED_POOL",
                module_key=key,
                records=records,
            )
        )
    audit = KeyFilterAudit(
        captured_key_count=len(captured_keys),
        full_pool_key_count=len(per_key_full),
        parity_valid_key_count=len(per_key_parity),
        parity_excluded_keys=tuple(parity_excluded),
    )
    return per_key_full, per_key_parity, parity_records_by_key, dense_fp_votes_by_key, audit


def run_anchored_fp_credit_degeneracy_diagnostic(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str | None = None,
    curriculum_seed: int = FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    batch_size: int = FROZEN_T2_ANCHOR_BATCH_SIZE,
    rank_spec: RankVoteSpec | None = None,
    rescale_shift: int = PARITY_RESCALE_SHIFT,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    blob_sha = checkpoint_sha256 or sha256_file(path)
    bundle = capture_tier2_checkpoint_raw_captures(
        checkpoint_path=str(path),
        checkpoint_sha256=str(blob_sha),
        curriculum_seed=int(curriculum_seed),
        batch_size=int(batch_size),
    )
    key_set_sha = anchor_key_set_sha256(sorted(bundle.per_key_captures.keys()))
    spec = rank_spec or default_dry_run_rank_vote_spec()
    per_key_full, per_key_parity, parity_records_by_key, dense_fp_votes_by_key, key_filter_audit = (
        _collect_key_metrics(
            bundle,
            spec=spec,
            rescale_shift=int(rescale_shift),
        )
    )
    full_pool = _aggregate_pool_metrics(pool_id="TRUE_FULL_FP_MOVE_POOL", per_key=per_key_full)
    parity_pool = _aggregate_pool_metrics(
        pool_id="PARITY_INTEGER_FILTERED_POOL",
        per_key=per_key_parity,
    )
    enrichment = compute_parity_enrichment_metrics(
        full_pool=full_pool,
        parity_pool=parity_pool,
        per_key_full=per_key_full,
        per_key_parity=per_key_parity,
        parity_records_by_key=parity_records_by_key,
        dense_fp_votes_by_key=dense_fp_votes_by_key,
    )
    branch_id = classify_fp_credit_degeneracy_branch(full_pool=full_pool, enrichment=enrichment)
    anchor_in_band = full_pool_fp4_anchor_in_band(enrichment.full_pool_fp4_fraction)
    receipt = FpCreditDegeneracyReceipt(
        schema=FP_CREDIT_DEGENERACY_SCHEMA_V2,
        target_name=FP_CREDIT_DEGENERACY_TARGET_NAME,
        pass_receipt=False,
        hard_false=realistic_gradient_parity_probe_hard_false_snapshot(),
        non_claims=FP_CREDIT_DEGENERACY_NON_CLAIMS,
        branch_id=branch_id,
        design_packet_sha256=FP_CREDIT_DEGENERACY_DESIGN_PACKET_SHA256,
        anchor={
            "checkpoint_path": str(path),
            "checkpoint_sha256": str(blob_sha),
            "curriculum_seed": int(curriculum_seed),
            "batch_size": int(batch_size),
            "key_count": len(bundle.per_key_captures),
            "key_set_sha256": str(key_set_sha),
            "frozen_checkpoint_sha256": FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
            "frozen_key_set_sha256": FROZEN_T2_ANCHOR_KEY_SET_SHA256,
            "capture_seam_id": T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
            "rescale_shift": int(rescale_shift),
        },
        production_law_id=INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
        full_pool=full_pool,
        per_key_full_pool=tuple(per_key_full),
        parity_pool=parity_pool,
        per_key_parity_pool=tuple(per_key_parity),
        enrichment=enrichment,
        key_filter_audit=key_filter_audit,
        full_pool_fp4_anchor_in_band=anchor_in_band,
        intrinsic_via_full_pool_fp4_structurally_unreachable_under_locked_spec=anchor_in_band,
        bundle_provenance=dict(bundle.provenance),
    )
    validate_fp_credit_degeneracy_receipt(receipt)
    return receipt.to_dict()
