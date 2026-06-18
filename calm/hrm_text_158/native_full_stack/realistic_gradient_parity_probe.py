"""CPU realistic-gradient parity probe for HRM-Text-1.58 Step 3C-B.

Measures whether integer sparse rank-bucket ordering holds against FP dense
reference on captures with fractional credit magnitudes. Gates 3C-GPU only;
does not flip optimizer_credit_state or authorize GPU runtime receipts.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    RankVoteSpec,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    integer_marginal_attribution_from_captures,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
    INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    compare_sparse_rank_to_fp_dense_reference,
    credit_q31_from_attribution,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    derive_trainer_sub2_authority_states,
    load_train_checkpoint_into_model,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)

REALISTIC_GRADIENT_PARITY_PROBE_SCHEMA_VERSION = (
    "hrm_text_158_realistic_gradient_parity_probe/v1_3"
)
REALISTIC_GRADIENT_PARITY_PROBE_TARGET_NAME = (
    "step3c_b_realistic_gradient_parity_probe"
)
PARITY_CONTRACT_MODE_REALISTIC_GRADIENT_ORDERING_PROBE = (
    "REALISTIC_GRADIENT_ORDERING_PROBE"
)

DEFAULT_T2_CHECKPOINT_REL = (
    "calm/hrm/checkpoints/hrm_text_158_tier_a_final_step01500.pt"
)
T2_CHECKPOINT_ENV_VAR = "HRM_TEXT_158_PROBE_CHECKPOINT"

MIN_MOVE_CANDIDATES = 8
MIN_FP_CREDIT_NONZERO = 4
MIN_FRACTIONAL_DIVERSITY = 3
FRACTIONAL_DIVERSITY_RELATIVE_BINS = 1000
MIN_RANK_GROUPS = 2
MIN_TIER_TOTAL_MOVE_CANDIDATES = 16
MAX_PER_CANDIDATE_RECORDS_PER_KEY = 64
MAX_FRACTIONAL_COLLISION_EXAMPLES = 5

VERDICT_BROAD_HOLDS = "broad_holds"
VERDICT_NARROW_HOLDS = "narrow_holds"
VERDICT_FRACTIONAL_COLLAPSE = "fractional_collapse"
VERDICT_MEASUREMENT_INVALID = "measurement_invalid"
VERDICT_INVESTIGATE = "investigate"

GPU_GATE_PROCEED = "proceed_3c_gpu"
GPU_GATE_PROCEED_NARROW = "proceed_3c_gpu_narrow"
GPU_GATE_REOPEN_3C_A = "reopen_3c_a_before_gpu"
GPU_GATE_INSUFFICIENT = "insufficient_evidence"
GPU_GATE_INVESTIGATE = "investigate"

CONCURRENCE_CONCUR = "concur"
CONCURRENCE_DISAGREE = "disagree"
CONCURRENCE_T2_ABSENT = "t2_absent"
CONCURRENCE_T2_REQUIRED_MISSING = "t2_required_missing"

REALISTIC_GRADIENT_PARITY_PROBE_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "gpu_runtime_receipt_present",
)

REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS = (
    "realistic-gradient parity probe is CPU ordering evidence only; no GPU runtime receipt",
    "pass_receipt is always false; probe does not flip optimizer_credit_state row",
    "parity_contract_mode=REALISTIC_GRADIENT_ORDERING_PROBE does not claim readiness or sub2 resolution",
    "T1 alone can block 3C-GPU; T1 broad without T2 when checkpoint exists is insufficient for broad proceed",
    "banked integer_sparse_rank_votes module is read-only consumer; probe observability is additive only",
    "no .pt write/commit; checkpoint load is read-only when T2 executes",
)

FORBIDDEN_FIXTURE_WEIGHT_SHAPE = (3, 2)


@dataclass(frozen=True)
class Trainer16x16CaptureFixture:
    model: torch.nn.Module
    batch: dict[str, torch.Tensor]
    eligible_scope: str = "all-bitlinear"
    use_ternary_bulk: bool = True

    def weight_shape(self) -> tuple[int, int]:
        proj = getattr(self.model, "proj", None)
        if not isinstance(proj, BitLinear):
            raise TypeError("fixture model must expose BitLinear proj")
        shape = tuple(int(dim) for dim in proj.weight.shape)
        if shape == FORBIDDEN_FIXTURE_WEIGHT_SHAPE:
            raise ValueError("BitLinear(3,2) fixture is forbidden for realistic-gradient probe")
        return shape


@dataclass(frozen=True)
class T2CheckpointDiscovery:
    checkpoint_path: str | None
    checkpoint_present: bool
    checked_paths: tuple[str, ...]
    absence_proof: str | None
    checkpoint_sha256: str | None


@dataclass(frozen=True)
class PerCandidateParityRecord:
    flat_index: int
    fp_credit: float
    int_credit_q31: int
    fp_rank_position: int
    int_rank_position: int
    fp_vote: int
    int_vote: int
    rank_match: bool
    event_match: bool
    fractional_collision_mismatch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "flat_index": int(self.flat_index),
            "fp_credit": float(self.fp_credit),
            "int_credit_q31": int(self.int_credit_q31),
            "fp_rank_position": int(self.fp_rank_position),
            "int_rank_position": int(self.int_rank_position),
            "fp_vote": int(self.fp_vote),
            "int_vote": int(self.int_vote),
            "rank_match": bool(self.rank_match),
            "event_match": bool(self.event_match),
            "fractional_collision_mismatch": bool(self.fractional_collision_mismatch),
        }


@dataclass(frozen=True)
class TierParityVerdict:
    tier_id: str
    measurement_valid: bool
    parity_verdict: str
    rank_positions_match_rate: float
    events_match_rate: float
    fractional_collision_share_of_mismatches: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "measurement_valid": bool(self.measurement_valid),
            "parity_verdict": str(self.parity_verdict),
            "rank_positions_match_rate": float(self.rank_positions_match_rate),
            "events_match_rate": float(self.events_match_rate),
            "fractional_collision_share_of_mismatches": float(
                self.fractional_collision_share_of_mismatches
            ),
        }


@dataclass
class KeyProbeMetrics:
    state_key: str
    measurement_valid: bool
    validity_detail: dict[str, Any]
    move_candidate_count: int
    rank_positions_match_rate: float
    events_match_rate: float
    fractional_collision_share_of_mismatches: float
    branch_id: str
    per_candidate_records: list[PerCandidateParityRecord] = field(default_factory=list)


@dataclass
class TierProbeResult:
    tier_id: str
    measurement_valid: bool
    measurement_validity_detail: dict[str, dict[str, Any]]
    tensor_key_sample_set: tuple[str, ...]
    per_key_metrics: dict[str, KeyProbeMetrics]
    aggregate_metrics: dict[str, Any]
    mismatch_clusters: dict[str, Any]
    fractional_collision_examples: list[dict[str, Any]]
    capture_provenance: dict[str, Any]


@dataclass(frozen=True)
class RawKeyCapture:
    inputs: tuple[torch.Tensor, ...]
    grad_outputs: tuple[torch.Tensor, ...]
    q_levels_flat: torch.Tensor
    weight_shape: tuple[int, int]


@dataclass(frozen=True)
class Tier2RawCaptureBundle:
    per_key_captures: dict[str, RawKeyCapture]
    per_key_states: dict[str, BoundedDeltaTensorState]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class RealisticGradientParityProbeReceipt:
    schema_version: str
    target_name: str
    parity_contract_mode: str
    pass_receipt: bool
    tiers_executed: tuple[str, ...]
    t2_checkpoint_present: bool
    t2_checkpoint_path: str | None
    t2_checkpoint_sha256: str | None
    t2_absence_proof: str | None
    tier_verdicts: dict[str, TierParityVerdict]
    t2_verdict: str | None
    t1_t2_concurrence: str
    hrm_representativeness_unconfirmed: bool
    capture_provenance: dict[str, Any]
    measurement_valid: bool
    measurement_validity_detail: dict[str, dict[str, dict[str, Any]]]
    tensor_key_sample_set: dict[str, tuple[str, ...]]
    per_key_metrics: dict[str, dict[str, Any]]
    per_candidate_records: dict[str, dict[str, list[dict[str, Any]]]]
    aggregate_metrics: dict[str, dict[str, Any]]
    mismatch_clusters: dict[str, dict[str, Any]]
    fractional_collision_examples: dict[str, list[dict[str, Any]]]
    parity_verdict: str
    gpu_gate_recommendation: str
    ready_to_flip: bool
    optimizer_credit_state_sub2_claim: bool
    optimizer_credit_state_resolved: bool
    readiness_row_flip_authorized: bool
    real_native_integer_attribution_present: bool
    real_native_integer_credit_ranking_present: bool
    gpu_runtime_receipt_present: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "parity_contract_mode": self.parity_contract_mode,
            "pass_receipt": self.pass_receipt,
            "tiers_executed": list(self.tiers_executed),
            "t2_checkpoint_present": self.t2_checkpoint_present,
            "t2_checkpoint_path": self.t2_checkpoint_path,
            "t2_checkpoint_sha256": self.t2_checkpoint_sha256,
            "t2_absence_proof": self.t2_absence_proof,
            "tier_verdicts": {
                key: value.to_dict() for key, value in sorted(self.tier_verdicts.items())
            },
            "t2_verdict": self.t2_verdict,
            "t1_t2_concurrence": self.t1_t2_concurrence,
            "hrm_representativeness_unconfirmed": self.hrm_representativeness_unconfirmed,
            "capture_provenance": dict(self.capture_provenance),
            "measurement_valid": self.measurement_valid,
            "measurement_validity_detail": self.measurement_validity_detail,
            "tensor_key_sample_set": {
                tier: list(keys) for tier, keys in sorted(self.tensor_key_sample_set.items())
            },
            "per_key_metrics": self.per_key_metrics,
            "per_candidate_records": self.per_candidate_records,
            "aggregate_metrics": self.aggregate_metrics,
            "mismatch_clusters": self.mismatch_clusters,
            "fractional_collision_examples": self.fractional_collision_examples,
            "parity_verdict": self.parity_verdict,
            "gpu_gate_recommendation": self.gpu_gate_recommendation,
            "ready_to_flip": self.ready_to_flip,
            "optimizer_credit_state_sub2_claim": self.optimizer_credit_state_sub2_claim,
            "optimizer_credit_state_resolved": self.optimizer_credit_state_resolved,
            "readiness_row_flip_authorized": self.readiness_row_flip_authorized,
            "real_native_integer_attribution_present": (
                self.real_native_integer_attribution_present
            ),
            "real_native_integer_credit_ranking_present": (
                self.real_native_integer_credit_ranking_present
            ),
            "gpu_runtime_receipt_present": self.gpu_runtime_receipt_present,
            "non_claims": list(self.non_claims),
        }


class _ProbeTinyTernary16x16(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def realistic_gradient_parity_probe_hard_false_snapshot() -> dict[str, bool]:
    return {field: False for field in REALISTIC_GRADIENT_PARITY_PROBE_HARD_FALSE_FIELDS}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "calm" / "hrm_text_158").is_dir() and (parent / "calm" / "hrm").is_dir():
            return parent
    raise RuntimeError("could not locate repo root for realistic-gradient parity probe")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_trainer_16x16_capture_fixture(
    *,
    seed: int = 158,
) -> Trainer16x16CaptureFixture:
    torch.manual_seed(int(seed))
    model = _ProbeTinyTernary16x16()
    with torch.no_grad():
        model.proj.weight.zero_()
        # Scale tail weights so integer q31 attribution survives rescale (≥4× vs 0.25 baseline).
        model.tail.weight.fill_(1.0)
        model.tail.bias.zero_()
    batch = {
        "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 4.0,
        "target": torch.randn(2, 4, dtype=torch.float32),
    }
    shape = tuple(int(dim) for dim in model.proj.weight.shape)
    if shape == FORBIDDEN_FIXTURE_WEIGHT_SHAPE:
        raise ValueError("BitLinear(3,2) fixture is forbidden")
    return Trainer16x16CaptureFixture(model=model, batch=batch)


def discover_t2_checkpoint(
    *,
    checkpoint_path: str | None = None,
) -> T2CheckpointDiscovery:
    checked: list[str] = []
    env_path = os.environ.get(T2_CHECKPOINT_ENV_VAR)
    candidates: list[str] = []
    if checkpoint_path is not None:
        candidates.append(str(checkpoint_path))
    if env_path:
        candidates.append(str(env_path))
    candidates.append(str(_repo_root() / DEFAULT_T2_CHECKPOINT_REL))
    resolved: str | None = None
    for candidate in candidates:
        checked.append(candidate)
        if Path(candidate).is_file():
            resolved = candidate
            break
    if resolved is None:
        absence = (
            "t2_checkpoint_absent; checked_paths="
            + ",".join(checked)
        )
        return T2CheckpointDiscovery(
            checkpoint_path=None,
            checkpoint_present=False,
            checked_paths=tuple(checked),
            absence_proof=absence,
            checkpoint_sha256=None,
        )
    return T2CheckpointDiscovery(
        checkpoint_path=resolved,
        checkpoint_present=True,
        checked_paths=tuple(checked),
        absence_proof=None,
        checkpoint_sha256=sha256_file(resolved),
    )


def _count_fp_credit_nonzero(fp_credit: torch.Tensor) -> int:
    return int(torch.count_nonzero(fp_credit.reshape(-1)).item())


def _fractional_diversity_count(fp_credit_sparse: torch.Tensor) -> int:
    if int(fp_credit_sparse.numel()) == 0:
        return 0
    abs_values = fp_credit_sparse.detach().cpu().to(torch.float32).abs()
    max_abs = float(abs_values.max().item()) if abs_values.numel() > 0 else 0.0
    if max_abs <= 0.0:
        return 0
    threshold = 0.05 * max_abs
    masked = abs_values[abs_values > threshold]
    if int(masked.numel()) == 0:
        return 0
    normalized = masked / max_abs
    quantized = torch.round(
        normalized * float(FRACTIONAL_DIVERSITY_RELATIVE_BINS)
    ) / float(FRACTIONAL_DIVERSITY_RELATIVE_BINS)
    return int(torch.unique(quantized).numel())


def _rank_group_count(rank_positions: torch.Tensor) -> int:
    if int(rank_positions.numel()) == 0:
        return 0
    return int(torch.unique(rank_positions.detach().cpu()).numel())


def _measurement_validity_for_key(
    *,
    captures_present: bool,
    captures_finite: bool,
    move_candidate_count: int,
    fp_credit_nonzero_count: int,
    fractional_diversity: int,
    rank_group_count: int,
) -> tuple[bool, dict[str, Any]]:
    detail = {
        "captures_present": bool(captures_present),
        "captures_finite": bool(captures_finite),
        "move_candidate_count": int(move_candidate_count),
        "fp_credit_nonzero_count": int(fp_credit_nonzero_count),
        "fractional_diversity": int(fractional_diversity),
        "rank_group_count": int(rank_group_count),
        "min_move_candidates": MIN_MOVE_CANDIDATES,
        "min_fp_credit_nonzero": MIN_FP_CREDIT_NONZERO,
        "min_fractional_diversity": MIN_FRACTIONAL_DIVERSITY,
        "min_rank_groups": MIN_RANK_GROUPS,
    }
    valid = (
        captures_present
        and captures_finite
        and move_candidate_count >= MIN_MOVE_CANDIDATES
        and fp_credit_nonzero_count >= MIN_FP_CREDIT_NONZERO
        and fractional_diversity >= MIN_FRACTIONAL_DIVERSITY
        and rank_group_count >= MIN_RANK_GROUPS
    )
    detail["measurement_valid"] = bool(valid)
    return valid, detail


def _compute_parity_rates_from_records(
    records: Sequence[PerCandidateParityRecord],
) -> dict[str, float]:
    total = len(records)
    if total == 0:
        return {
            "rank_positions_match_rate": 0.0,
            "events_match_rate": 0.0,
            "fractional_collision_share_of_mismatches": 0.0,
        }
    mismatches = [item for item in records if not (item.rank_match and item.event_match)]
    fractional_mismatches = [item for item in mismatches if item.fractional_collision_mismatch]
    return {
        "rank_positions_match_rate": float(
            sum(1 for item in records if item.rank_match) / total
        ),
        "events_match_rate": float(sum(1 for item in records if item.event_match) / total),
        "fractional_collision_share_of_mismatches": float(
            len(fractional_mismatches) / max(len(mismatches), 1)
        ),
    }


def build_per_candidate_parity_records(
    *,
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    spec: RankVoteSpec,
    attribution_law_id: str = INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    credit_law_id: str = INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
    max_records: int = MAX_PER_CANDIDATE_RECORDS_PER_KEY,
) -> tuple[list[PerCandidateParityRecord], dict[str, Any]]:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    attribution_events = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
        law_id=attribution_law_id,
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
    from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
        _rank_positions_for_credit_values,
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
    full_records: list[PerCandidateParityRecord] = []
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
        full_records.append(
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
    rate_summary = _compute_parity_rates_from_records(full_records)
    emitted_records = full_records[: int(max_records)]
    summary = {
        "branch_id": parity.branch_id,
        "rank_positions_match_rate": rate_summary["rank_positions_match_rate"],
        "events_match_rate": rate_summary["events_match_rate"],
        "fractional_collision_share_of_mismatches": rate_summary[
            "fractional_collision_share_of_mismatches"
        ],
        "move_candidate_count": int(move_indices.numel()),
        "full_record_count": len(full_records),
        "emitted_record_count": len(emitted_records),
    }
    return emitted_records, summary


def classify_tier_parity_verdict(
    *,
    tier_id: str,
    measurement_valid: bool,
    rank_positions_match_rate: float,
    events_match_rate: float,
    fractional_collision_share_of_mismatches: float,
) -> TierParityVerdict:
    if not measurement_valid:
        return TierParityVerdict(
            tier_id=tier_id,
            measurement_valid=False,
            parity_verdict=VERDICT_MEASUREMENT_INVALID,
            rank_positions_match_rate=float(rank_positions_match_rate),
            events_match_rate=float(events_match_rate),
            fractional_collision_share_of_mismatches=float(
                fractional_collision_share_of_mismatches
            ),
        )
    broad = (
        rank_positions_match_rate >= 0.95
        and events_match_rate >= 0.90
        and fractional_collision_share_of_mismatches < 0.20
    )
    narrow = (
        rank_positions_match_rate >= 0.80
        and events_match_rate >= 0.70
        and not broad
    )
    collapse = (
        rank_positions_match_rate < 0.80
        or fractional_collision_share_of_mismatches >= 0.50
    )
    if broad:
        verdict = VERDICT_BROAD_HOLDS
    elif collapse:
        verdict = VERDICT_FRACTIONAL_COLLAPSE
    elif narrow:
        verdict = VERDICT_NARROW_HOLDS
    else:
        verdict = VERDICT_FRACTIONAL_COLLAPSE
    return TierParityVerdict(
        tier_id=tier_id,
        measurement_valid=True,
        parity_verdict=verdict,
        rank_positions_match_rate=float(rank_positions_match_rate),
        events_match_rate=float(events_match_rate),
        fractional_collision_share_of_mismatches=float(
            fractional_collision_share_of_mismatches
        ),
    )


def _verdict_class(verdict: str) -> str:
    if verdict in {VERDICT_BROAD_HOLDS, VERDICT_NARROW_HOLDS}:
        return verdict
    if verdict == VERDICT_FRACTIONAL_COLLAPSE:
        return VERDICT_FRACTIONAL_COLLAPSE
    if verdict == VERDICT_MEASUREMENT_INVALID:
        return VERDICT_MEASUREMENT_INVALID
    return verdict


def classify_t1_t2_concurrence(
    *,
    t2_checkpoint_present: bool,
    tiers_executed: Sequence[str],
    tier_verdicts: Mapping[str, TierParityVerdict],
) -> str:
    executed = set(str(item) for item in tiers_executed)
    if not t2_checkpoint_present:
        return CONCURRENCE_T2_ABSENT
    if "T2" not in executed:
        return CONCURRENCE_T2_REQUIRED_MISSING
    t1 = tier_verdicts.get("T1")
    t2 = tier_verdicts.get("T2")
    if t1 is None or t2 is None:
        return CONCURRENCE_T2_REQUIRED_MISSING
    hold_classes = {VERDICT_BROAD_HOLDS, VERDICT_NARROW_HOLDS}
    if t1.parity_verdict in hold_classes and t2.parity_verdict in hold_classes:
        if _verdict_class(t1.parity_verdict) == _verdict_class(t2.parity_verdict):
            return CONCURRENCE_CONCUR
    return CONCURRENCE_DISAGREE


def classify_gpu_gate_recommendation(
    *,
    t2_checkpoint_present: bool,
    tiers_executed: Sequence[str],
    tier_verdicts: Mapping[str, TierParityVerdict],
    t1_t2_concurrence: str,
) -> tuple[str, str, bool]:
    t1 = tier_verdicts.get("T1")
    if t1 is None:
        return VERDICT_MEASUREMENT_INVALID, GPU_GATE_INSUFFICIENT, False
    if t1.parity_verdict == VERDICT_FRACTIONAL_COLLAPSE:
        return VERDICT_FRACTIONAL_COLLAPSE, GPU_GATE_REOPEN_3C_A, False
    if t1.parity_verdict == VERDICT_MEASUREMENT_INVALID:
        return VERDICT_MEASUREMENT_INVALID, GPU_GATE_INSUFFICIENT, False
    executed = set(str(item) for item in tiers_executed)
    if t2_checkpoint_present and "T2" not in executed:
        return VERDICT_MEASUREMENT_INVALID, GPU_GATE_INSUFFICIENT, False
    if t2_checkpoint_present and t1_t2_concurrence == CONCURRENCE_DISAGREE:
        return VERDICT_INVESTIGATE, GPU_GATE_INVESTIGATE, False
    if t2_checkpoint_present and t1_t2_concurrence == CONCURRENCE_CONCUR:
        t2 = tier_verdicts.get("T2")
        if t2 is None or not t2.measurement_valid:
            return VERDICT_MEASUREMENT_INVALID, GPU_GATE_INSUFFICIENT, False
        if (
            t1.parity_verdict == VERDICT_BROAD_HOLDS
            and t2.parity_verdict == VERDICT_BROAD_HOLDS
        ):
            return VERDICT_BROAD_HOLDS, GPU_GATE_PROCEED, False
        if (
            t1.parity_verdict == VERDICT_NARROW_HOLDS
            and t2.parity_verdict == VERDICT_NARROW_HOLDS
        ):
            return VERDICT_NARROW_HOLDS, GPU_GATE_PROCEED_NARROW, False
        return VERDICT_INVESTIGATE, GPU_GATE_INVESTIGATE, False
    if not t2_checkpoint_present:
        if t1.parity_verdict == VERDICT_BROAD_HOLDS:
            return VERDICT_BROAD_HOLDS, GPU_GATE_INSUFFICIENT, True
        if t1.parity_verdict == VERDICT_NARROW_HOLDS:
            return VERDICT_NARROW_HOLDS, GPU_GATE_PROCEED_NARROW, True
    return VERDICT_MEASUREMENT_INVALID, GPU_GATE_INSUFFICIENT, False


def _aggregate_tier_rates(per_key_metrics: Mapping[str, KeyProbeMetrics]) -> dict[str, float]:
    valid_keys = [item for item in per_key_metrics.values() if item.measurement_valid]
    if not valid_keys:
        return {
            "rank_positions_match_rate": 0.0,
            "events_match_rate": 0.0,
            "fractional_collision_share_of_mismatches": 0.0,
            "total_move_candidates": 0,
        }
    total_moves = sum(int(item.move_candidate_count) for item in valid_keys)
    rank_weighted = sum(
        item.rank_positions_match_rate * item.move_candidate_count for item in valid_keys
    )
    event_weighted = sum(
        item.events_match_rate * item.move_candidate_count for item in valid_keys
    )
    frac_weighted = sum(
        item.fractional_collision_share_of_mismatches * item.move_candidate_count
        for item in valid_keys
    )
    return {
        "rank_positions_match_rate": float(rank_weighted / max(total_moves, 1)),
        "events_match_rate": float(event_weighted / max(total_moves, 1)),
        "fractional_collision_share_of_mismatches": float(
            frac_weighted / max(total_moves, 1)
        ),
        "total_move_candidates": int(total_moves),
    }


def _probe_key_from_captures(
    *,
    state_key: str,
    inputs: Sequence[torch.Tensor],
    grad_outputs: Sequence[torch.Tensor],
    weight_shape: Sequence[int],
    q_levels_flat: torch.Tensor,
    spec: RankVoteSpec,
) -> KeyProbeMetrics:
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
    fp_moves = project_s1_gradient_to_moves(weighted_grad, q_levels_flat.reshape(weight_dims))
    attribution_events = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_dims,
    )
    move_indices, moves = projected_moves_from_integer_attribution(
        attribution_events,
        q_levels_flat,
    )
    move_count = int(move_indices.numel())
    if move_count > 0:
        fp_credit_sparse = fp_credit.reshape(-1).index_select(0, move_indices)
    else:
        fp_credit_sparse = torch.empty(0, dtype=torch.float32)
    from calm.hrm_text_158.native_full_stack.integer_sparse_rank_votes import (
        _rank_positions_for_credit_values,
    )

    rank_positions = _rank_positions_for_credit_values(fp_credit_sparse.to(torch.float32), spec)
    valid, validity_detail = _measurement_validity_for_key(
        captures_present=captures_present,
        captures_finite=captures_finite,
        move_candidate_count=move_count,
        fp_credit_nonzero_count=_count_fp_credit_nonzero(fp_credit),
        fractional_diversity=_fractional_diversity_count(fp_credit_sparse),
        rank_group_count=_rank_group_count(rank_positions),
    )
    records, summary = build_per_candidate_parity_records(
        inputs=inputs,
        grad_outputs=grad_outputs,
        weight_shape=weight_dims,
        q_levels_flat=q_levels_flat,
        spec=spec,
    )
    return KeyProbeMetrics(
        state_key=state_key,
        measurement_valid=valid,
        validity_detail=validity_detail,
        move_candidate_count=int(summary["move_candidate_count"]),
        rank_positions_match_rate=float(summary["rank_positions_match_rate"]),
        events_match_rate=float(summary["events_match_rate"]),
        fractional_collision_share_of_mismatches=float(
            summary["fractional_collision_share_of_mismatches"]
        ),
        branch_id=str(summary["branch_id"]),
        per_candidate_records=records,
    )


def _finalize_tier_probe_result(
    *,
    tier_id: str,
    per_key_metrics: dict[str, KeyProbeMetrics],
    capture_provenance: dict[str, Any],
) -> TierProbeResult:
    validity_detail = {
        key: dict(item.validity_detail) for key, item in sorted(per_key_metrics.items())
    }
    aggregate = _aggregate_tier_rates(per_key_metrics)
    valid_key_count = sum(1 for item in per_key_metrics.values() if item.measurement_valid)
    tier_measurement_valid = (
        len(per_key_metrics) > 0
        and valid_key_count >= 1
        and int(aggregate["total_move_candidates"]) >= MIN_TIER_TOTAL_MOVE_CANDIDATES
    )
    examples: list[dict[str, Any]] = []
    mismatch_clusters: dict[str, Any] = {}
    for key, metrics in sorted(per_key_metrics.items()):
        mismatches = [
            record.to_dict()
            for record in metrics.per_candidate_records
            if record.fractional_collision_mismatch
        ]
        if mismatches:
            mismatch_clusters[key] = {"count": len(mismatches), "sample": mismatches[:3]}
        for record in metrics.per_candidate_records:
            if record.fractional_collision_mismatch and len(examples) < MAX_FRACTIONAL_COLLISION_EXAMPLES:
                examples.append({"state_key": key, **record.to_dict()})
    return TierProbeResult(
        tier_id=tier_id,
        measurement_valid=tier_measurement_valid,
        measurement_validity_detail=validity_detail,
        tensor_key_sample_set=tuple(sorted(per_key_metrics.keys())),
        per_key_metrics=per_key_metrics,
        aggregate_metrics=aggregate,
        mismatch_clusters=mismatch_clusters,
        fractional_collision_examples=examples,
        capture_provenance=capture_provenance,
    )


def run_tier1_trainer_16x16_capture(
    *,
    fixture: Trainer16x16CaptureFixture | None = None,
    rank_spec: RankVoteSpec | None = None,
    device: torch.device | str = "cpu",
) -> TierProbeResult:
    probe_fixture = fixture or build_trainer_16x16_capture_fixture()
    spec = rank_spec or default_dry_run_rank_vote_spec()
    model = probe_fixture.model
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=probe_fixture.use_ternary_bulk,
        eligible_scope=probe_fixture.eligible_scope,
    )
    states = derive_trainer_sub2_authority_states(eligible)
    prior_training = bool(model.training)
    per_key_metrics: dict[str, KeyProbeMetrics] = {}
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device=device,
            requires_grad=True,
        ) as handle:
            out = model(probe_fixture.batch["x"])
            loss = F.mse_loss(out, probe_fixture.batch["target"])
            if not bool(torch.isfinite(loss.detach()).item()):
                raise ValueError("tier1 probe requires finite loss")
            loss.backward()
            for key, state in states.items():
                capture = handle.captures[key]
                per_key_metrics[key] = _probe_key_from_captures(
                    state_key=key,
                    inputs=capture["inputs"],
                    grad_outputs=capture["grad_outputs"],
                    weight_shape=tuple(state.q_levels.shape),
                    q_levels_flat=state.q_levels.reshape(-1),
                    spec=spec,
                )
    finally:
        model.train(prior_training)
    provenance = {
        "tier": "T1",
        "fixture": "trainer_16x16_capture_fixture",
        "weight_shape": list(probe_fixture.weight_shape()),
        "module_file": inspect.getfile(build_trainer_16x16_capture_fixture),
    }
    return _finalize_tier_probe_result(
        tier_id="T1",
        per_key_metrics=per_key_metrics,
        capture_provenance=provenance,
    )


def capture_tier2_checkpoint_raw_captures(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str | None = None,
    device: torch.device | str = "cpu",
    curriculum_seed: int = 158,
    batch_size: int = 4,
) -> Tier2RawCaptureBundle:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    blob_sha = checkpoint_sha256 or sha256_file(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        build_identity_full_batch,
        build_model_from_checkpoint,
    )

    torch_device = torch.device(device)
    model, tok, _cfg = build_model_from_checkpoint(ckpt, torch_device)
    load_result = load_train_checkpoint_into_model(
        model,
        ckpt,
        use_ternary_bulk=bool(ckpt.get("config", {}).get("use_ternary_bulk", False)),
        eligible_scope="all-bitlinear",
        device=torch_device,
        inference_only=False,
        sub2_live_enabled=False,
    )
    eligible = select_trainer_eligible_bitlinears(
        model,
        use_ternary_bulk=bool(ckpt.get("config", {}).get("use_ternary_bulk", False)),
        eligible_scope="all-bitlinear",
    )
    if load_result.authority_states is not None:
        states = load_result.authority_states
    else:
        states = derive_trainer_sub2_authority_states(eligible)
    batch, _batch_proof = build_identity_full_batch(
        tok=tok,
        max_len=int(ckpt.get("config", {}).get("max_len", 32)),
        batch_size=int(batch_size),
        curriculum_seed=int(curriculum_seed),
        device=torch_device,
    )
    extras = model.compute_train_extra_args(step=0, total_steps=1)
    prior_training = bool(model.training)
    per_key_captures: dict[str, RawKeyCapture] = {}
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device=torch_device,
            requires_grad=True,
        ) as handle:
            _carry, loss, _metrics = model(None, dict(batch), **extras)
            if not bool(torch.isfinite(loss.detach()).item()):
                raise ValueError("tier2 probe requires finite loss")
            loss.backward()
            for key, state in states.items():
                capture = handle.captures[key]
                per_key_captures[key] = RawKeyCapture(
                    inputs=tuple(capture["inputs"]),
                    grad_outputs=tuple(capture["grad_outputs"]),
                    q_levels_flat=state.q_levels.reshape(-1),
                    weight_shape=tuple(int(dim) for dim in state.q_levels.shape),
                )
    finally:
        model.train(prior_training)
    provenance = {
        "tier": "T2",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(blob_sha),
        "routing": str(load_result.routing),
        "batch_size": int(batch_size),
        "curriculum_seed": int(curriculum_seed),
        "enriched_capture": True,
        "attribution_law_id": INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
        "credit_law_id": INTEGER_SPARSE_RANK_PRODUCTION_CREDIT_LAW_ID,
        "tensor_keys_probed": sorted(per_key_captures.keys()),
        "capture_seam_id": "capture_tier2_checkpoint_raw_captures",
    }
    return Tier2RawCaptureBundle(
        per_key_captures=per_key_captures,
        per_key_states=dict(states),
        provenance=provenance,
    )


def run_tier2_checkpoint_capture(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str | None = None,
    rank_spec: RankVoteSpec | None = None,
    device: torch.device | str = "cpu",
    curriculum_seed: int = 158,
    batch_size: int = 4,
) -> TierProbeResult:
    bundle = capture_tier2_checkpoint_raw_captures(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
        curriculum_seed=curriculum_seed,
        batch_size=batch_size,
    )
    spec = rank_spec or default_dry_run_rank_vote_spec()
    per_key_metrics: dict[str, KeyProbeMetrics] = {}
    for key, capture in bundle.per_key_captures.items():
        per_key_metrics[key] = _probe_key_from_captures(
            state_key=key,
            inputs=capture.inputs,
            grad_outputs=capture.grad_outputs,
            weight_shape=capture.weight_shape,
            q_levels_flat=capture.q_levels_flat,
            spec=spec,
        )
    return _finalize_tier_probe_result(
        tier_id="T2",
        per_key_metrics=per_key_metrics,
        capture_provenance=bundle.provenance,
    )


def validate_realistic_gradient_parity_probe_receipt(
    receipt: RealisticGradientParityProbeReceipt,
) -> None:
    if receipt.schema_version != REALISTIC_GRADIENT_PARITY_PROBE_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch on realistic-gradient parity probe receipt")
    if receipt.target_name != REALISTIC_GRADIENT_PARITY_PROBE_TARGET_NAME:
        raise ValueError("target_name mismatch on realistic-gradient parity probe receipt")
    if receipt.parity_contract_mode != PARITY_CONTRACT_MODE_REALISTIC_GRADIENT_ORDERING_PROBE:
        raise ValueError("parity_contract_mode mismatch on realistic-gradient parity probe receipt")
    if receipt.pass_receipt:
        raise ValueError("pass_receipt must remain false on realistic-gradient parity probe receipt")
    for field_name in REALISTIC_GRADIENT_PARITY_PROBE_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field_name)):
            raise ValueError(f"{field_name} must remain false on realistic-gradient parity probe receipt")
    if receipt.non_claims != REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS:
        raise ValueError("non_claims must be exact on realistic-gradient parity probe receipt")
    if receipt.t2_checkpoint_present and "T2" not in receipt.tiers_executed:
        if receipt.gpu_gate_recommendation.startswith("proceed_3c_gpu"):
            raise ValueError("proceed_3c_gpu* forbidden when T2 required but not executed")
    if receipt.gpu_gate_recommendation == GPU_GATE_PROCEED:
        t1 = receipt.tier_verdicts.get("T1")
        t2 = receipt.tier_verdicts.get("T2")
        if not receipt.t2_checkpoint_present:
            raise ValueError("proceed_3c_gpu requires checkpoint present")
        if receipt.t1_t2_concurrence != CONCURRENCE_CONCUR:
            raise ValueError("proceed_3c_gpu requires t1_t2_concurrence=concur")
        if t1 is None or t2 is None:
            raise ValueError("proceed_3c_gpu requires T1 and T2 tier verdicts")
        if t1.parity_verdict != VERDICT_BROAD_HOLDS or t2.parity_verdict != VERDICT_BROAD_HOLDS:
            raise ValueError("proceed_3c_gpu requires both tiers broad_holds")
    if receipt.gpu_gate_recommendation == GPU_GATE_PROCEED_NARROW:
        allowed_absent_narrow = (
            not receipt.t2_checkpoint_present
            and receipt.hrm_representativeness_unconfirmed
            and receipt.tier_verdicts.get("T1") is not None
            and receipt.tier_verdicts["T1"].parity_verdict == VERDICT_NARROW_HOLDS
        )
        allowed_concur_narrow = (
            receipt.t2_checkpoint_present
            and receipt.t1_t2_concurrence == CONCURRENCE_CONCUR
            and receipt.tier_verdicts.get("T1") is not None
            and receipt.tier_verdicts.get("T2") is not None
            and receipt.tier_verdicts["T1"].parity_verdict == VERDICT_NARROW_HOLDS
            and receipt.tier_verdicts["T2"].parity_verdict == VERDICT_NARROW_HOLDS
        )
        if not (allowed_absent_narrow or allowed_concur_narrow):
            raise ValueError("proceed_3c_gpu_narrow gate preconditions failed")
    if receipt.gpu_gate_recommendation.startswith("proceed_3c_gpu"):
        if receipt.t2_checkpoint_present and receipt.t1_t2_concurrence != CONCURRENCE_CONCUR:
            raise ValueError("proceed_3c_gpu* forbidden when checkpoint present without concurrence")
        for tier_id in receipt.tiers_executed:
            verdict = receipt.tier_verdicts.get(tier_id)
            if verdict is None or not verdict.measurement_valid:
                raise ValueError("proceed_3c_gpu* forbidden when any executed tier is measurement-invalid")


def run_realistic_gradient_parity_probe(
    *,
    checkpoint_path: str | None = None,
    rank_spec: RankVoteSpec | None = None,
    device: torch.device | str = "cpu",
    run_t2: bool | None = None,
    t2_batch_size: int = 4,
) -> RealisticGradientParityProbeReceipt:
    discovery = discover_t2_checkpoint(checkpoint_path=checkpoint_path)
    tier_results: dict[str, TierProbeResult] = {}
    tiers_executed: list[str] = []
    tier1 = run_tier1_trainer_16x16_capture(rank_spec=rank_spec, device=device)
    tier_results["T1"] = tier1
    tiers_executed.append("T1")
    should_run_t2 = discovery.checkpoint_present if run_t2 is None else bool(run_t2)
    if should_run_t2:
        if not discovery.checkpoint_present or discovery.checkpoint_path is None:
            raise FileNotFoundError("T2 requested but checkpoint absent")
        tier2 = run_tier2_checkpoint_capture(
            checkpoint_path=discovery.checkpoint_path,
            checkpoint_sha256=discovery.checkpoint_sha256,
            rank_spec=rank_spec,
            device=device,
            batch_size=int(t2_batch_size),
        )
        tier_results["T2"] = tier2
        tiers_executed.append("T2")
    tier_verdicts: dict[str, TierParityVerdict] = {}
    for tier_id, result in tier_results.items():
        aggregate = result.aggregate_metrics
        tier_verdicts[tier_id] = classify_tier_parity_verdict(
            tier_id=tier_id,
            measurement_valid=result.measurement_valid,
            rank_positions_match_rate=float(aggregate["rank_positions_match_rate"]),
            events_match_rate=float(aggregate["events_match_rate"]),
            fractional_collision_share_of_mismatches=float(
                aggregate["fractional_collision_share_of_mismatches"]
            ),
        )
    concurrence = classify_t1_t2_concurrence(
        t2_checkpoint_present=discovery.checkpoint_present,
        tiers_executed=tiers_executed,
        tier_verdicts=tier_verdicts,
    )
    parity_verdict, gpu_gate, hrm_unconfirmed = classify_gpu_gate_recommendation(
        t2_checkpoint_present=discovery.checkpoint_present,
        tiers_executed=tiers_executed,
        tier_verdicts=tier_verdicts,
        t1_t2_concurrence=concurrence,
    )
    aggregate_measurement_valid = all(
        tier_verdicts[tier_id].measurement_valid for tier_id in tiers_executed
    )
    hard_false = realistic_gradient_parity_probe_hard_false_snapshot()
    receipt = RealisticGradientParityProbeReceipt(
        schema_version=REALISTIC_GRADIENT_PARITY_PROBE_SCHEMA_VERSION,
        target_name=REALISTIC_GRADIENT_PARITY_PROBE_TARGET_NAME,
        parity_contract_mode=PARITY_CONTRACT_MODE_REALISTIC_GRADIENT_ORDERING_PROBE,
        pass_receipt=False,
        tiers_executed=tuple(tiers_executed),
        t2_checkpoint_present=discovery.checkpoint_present,
        t2_checkpoint_path=discovery.checkpoint_path,
        t2_checkpoint_sha256=discovery.checkpoint_sha256,
        t2_absence_proof=discovery.absence_proof,
        tier_verdicts=tier_verdicts,
        t2_verdict=tier_verdicts["T2"].parity_verdict if "T2" in tier_verdicts else None,
        t1_t2_concurrence=concurrence,
        hrm_representativeness_unconfirmed=hrm_unconfirmed,
        capture_provenance={
            tier: result.capture_provenance for tier, result in tier_results.items()
        },
        measurement_valid=aggregate_measurement_valid,
        measurement_validity_detail={
            tier: result.measurement_validity_detail for tier, result in tier_results.items()
        },
        tensor_key_sample_set={
            tier: result.tensor_key_sample_set for tier, result in tier_results.items()
        },
        per_key_metrics={
            tier: {
                key: {
                    "measurement_valid": item.measurement_valid,
                    "validity_detail": item.validity_detail,
                    "move_candidate_count": item.move_candidate_count,
                    "rank_positions_match_rate": item.rank_positions_match_rate,
                    "events_match_rate": item.events_match_rate,
                    "fractional_collision_share_of_mismatches": (
                        item.fractional_collision_share_of_mismatches
                    ),
                    "branch_id": item.branch_id,
                }
                for key, item in sorted(result.per_key_metrics.items())
            }
            for tier, result in tier_results.items()
        },
        per_candidate_records={
            tier: {
                key: [record.to_dict() for record in item.per_candidate_records]
                for key, item in sorted(result.per_key_metrics.items())
            }
            for tier, result in tier_results.items()
        },
        aggregate_metrics={
            tier: dict(result.aggregate_metrics) for tier, result in tier_results.items()
        },
        mismatch_clusters={
            tier: dict(result.mismatch_clusters) for tier, result in tier_results.items()
        },
        fractional_collision_examples={
            tier: list(result.fractional_collision_examples)
            for tier, result in tier_results.items()
        },
        parity_verdict=parity_verdict,
        gpu_gate_recommendation=gpu_gate,
        ready_to_flip=hard_false["ready_to_flip"],
        optimizer_credit_state_sub2_claim=hard_false["optimizer_credit_state_sub2_claim"],
        optimizer_credit_state_resolved=hard_false["optimizer_credit_state_resolved"],
        readiness_row_flip_authorized=hard_false["readiness_row_flip_authorized"],
        real_native_integer_attribution_present=hard_false[
            "real_native_integer_attribution_present"
        ],
        real_native_integer_credit_ranking_present=hard_false[
            "real_native_integer_credit_ranking_present"
        ],
        gpu_runtime_receipt_present=hard_false["gpu_runtime_receipt_present"],
        non_claims=REALISTIC_GRADIENT_PARITY_PROBE_NON_CLAIMS,
    )
    validate_realistic_gradient_parity_probe_receipt(receipt)
    return receipt
