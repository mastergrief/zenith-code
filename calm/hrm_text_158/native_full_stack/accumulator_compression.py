"""C1.0 accumulator-compression feasibility and decision contract.

This module is deliberately a contract/feasibility surface, not an accumulator
encoder. It computes the strict physical ledger lower bound and registers the
learner-loop decisions that a future compressed accumulator representation must
preserve before any C2 dynamics run is meaningful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Sequence

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PACKED_TERNARY_METADATA_BYTES_PER_DIM,
    PACKED_TERNARY_METADATA_HEADER_BYTES,
    TARGET_PHYSICAL_BITS_PER_WEIGHT,
    _bits_per_weight as _persistent_bits_per_weight,
)


ACCUMULATOR_COMPRESSION_CONTRACT_SCHEMA_VERSION = (
    "hrm_text_158_accumulator_compression_contract/v0.feasibility_semantics"
)
ACCUMULATOR_COMPRESSION_CONTRACT_LABEL = (
    "accumulator_compression_feasibility_semantic_contract_no_encoder"
)
NO_ENCODER_STATUS = "contract_feasibility_only_no_accumulator_encoder"
FIXED_Q_ACCUMULATOR_ONLY_NULL = (
    "physical sub-2 is impossible via accumulator compression alone under the "
    "current fixed 2-bit q payload when q metadata/scale are positive"
)
JOINT_Q_ENTROPY_PREVIEW = (
    "joint q-entropy plus accumulator compression preview only; q-code "
    "metadata/padding/grouping overhead must have its own ledger before any "
    "physical sub-2 achievement claim"
)
IDENTITY_INT16_BASELINE_NAME = "identity_int16_baseline_not_compressed"


class CandidateClassification(str, Enum):
    """Machine-checkable taxonomy for future accumulator candidates."""

    BIT_EXACT = "bit_exact"
    DECISION_EXACT = "decision_exact"
    BOUNDED_DELTA_WITH_REPORT = "bounded_delta_with_report"
    NOT_SAME_LEARNER = "not_same_learner"


@dataclass(frozen=True)
class AccumulatorFeasibilityReport:
    """Strict physical persistent-state ledger for an accumulator candidate."""

    schema_version: str
    label: str
    regime_name: str
    target_bits_per_weight: float
    eligible_weight_count: int
    q_packed_data_bits_per_weight: float
    q_packed_padding_bits: int
    q_packed_padding_bits_per_weight: float
    q_packed_metadata_bits: int
    q_packed_metadata_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    frozen_scale_bits: int
    frozen_scale_bits_per_weight: float
    accumulator_bits_per_weight: float
    remaining_accumulator_budget_bits_per_weight: float
    packed_inclusive_physical_bits_per_weight: float
    target_achieved_with_reported_ledger: bool
    claimable_physical_sub2: bool
    accumulator_only_sub2_possible_under_current_q: bool
    q_entropy_code_overhead_accounted: bool
    joint_q_entropy_route_status: str
    fixed_q_null_statement: str
    status: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticDecisionDimension:
    """One required learner-loop decision a compressed accumulator may perturb."""

    surface: str
    name: str
    source_anchor: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAssessment:
    """Classification of a future accumulator representation against C1.0 gates."""

    candidate_name: str
    classification: CandidateClassification | str
    covered_decision_dimensions: tuple[str, ...]
    compressed_representation: bool
    bounded_delta_hypothesis: str | None = None
    guardrail: str | None = None
    preserved_information: tuple[str, ...] = ()
    sub2_persistent_strategy: str | None = None
    note: str = ""

    @property
    def normalized_classification(self) -> CandidateClassification:
        return CandidateClassification(self.classification)

    @property
    def missing_decision_dimensions(self) -> tuple[str, ...]:
        covered = set(self.covered_decision_dimensions)
        return tuple(name for name in required_decision_dimension_names() if name not in covered)

    @property
    def c2_eligible_by_default(self) -> bool:
        if not self.compressed_representation:
            return False
        if self.missing_decision_dimensions:
            return False
        return self.normalized_classification in {
            CandidateClassification.BIT_EXACT,
            CandidateClassification.DECISION_EXACT,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "classification": self.normalized_classification.value,
            "covered_decision_dimensions": list(self.covered_decision_dimensions),
            "missing_decision_dimensions": list(self.missing_decision_dimensions),
            "compressed_representation": bool(self.compressed_representation),
            "c2_eligible_by_default": bool(self.c2_eligible_by_default),
            "bounded_delta_hypothesis": self.bounded_delta_hypothesis,
            "guardrail": self.guardrail,
            "preserved_information": list(self.preserved_information),
            "sub2_persistent_strategy": self.sub2_persistent_strategy,
            "note": self.note,
        }


@dataclass(frozen=True)
class AccumulatorCompressionContractReport:
    """Compact C1.0 report: ledger rows plus semantic/taxonomy guardrails."""

    schema_version: str
    label: str
    status: str
    feasibility_rows: tuple[AccumulatorFeasibilityReport, ...]
    semantic_decision_dimensions: tuple[SemanticDecisionDimension, ...]
    candidate_taxonomy: tuple[str, ...]
    identity_baseline: CandidateAssessment
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "status": self.status,
            "feasibility_rows": [row.to_dict() for row in self.feasibility_rows],
            "semantic_decision_dimensions": [
                item.to_dict() for item in self.semantic_decision_dimensions
            ],
            "candidate_taxonomy": list(self.candidate_taxonomy),
            "identity_baseline": self.identity_baseline.to_dict(),
            "non_claims": list(self.non_claims),
        }


def packed_ternary_metadata_bits_for_shape(logical_shape: Sequence[int]) -> int:
    """Return the source-ledger metadata bits for one packed q tensor."""

    if len(tuple(logical_shape)) == 0:
        raise ValueError("logical_shape must have at least one dimension")
    for dim in logical_shape:
        if int(dim) <= 0:
            raise ValueError(f"logical_shape dims must be positive, got {tuple(logical_shape)}")
    metadata_bytes = PACKED_TERNARY_METADATA_HEADER_BYTES + (
        PACKED_TERNARY_METADATA_BYTES_PER_DIM * len(tuple(logical_shape))
    )
    return int(metadata_bytes * 8)


def packed_2bit_payload_bits_and_padding(logical_numel: int) -> tuple[int, int]:
    """Return actual packed payload bits and diagnostic padding bits."""

    numel = int(logical_numel)
    if numel <= 0:
        raise ValueError(f"logical_numel must be > 0, got {logical_numel}")
    payload_bits = int(math.ceil(numel / 4.0) * 8)
    ideal_bits = int(numel * 2)
    return payload_bits, int(payload_bits - ideal_bits)


def evaluate_accumulator_feasibility(
    *,
    regime_name: str,
    eligible_weight_count: int,
    q_packed_data_bits_per_weight: float,
    q_packed_metadata_bits: int,
    frozen_scale_bits: int,
    accumulator_bits_per_weight: float,
    q_packed_padding_bits: int = 0,
    target_bits_per_weight: float = TARGET_PHYSICAL_BITS_PER_WEIGHT,
    q_entropy_code_overhead_accounted: bool = True,
) -> AccumulatorFeasibilityReport:
    """Evaluate the exact source lower bound: target - q_packed_total - scale.

    ``q_packed_data_bits_per_weight`` is the actual packed byte payload per
    eligible weight, so it already physically includes any padding. Padding is
    retained separately as a diagnostic and must not be double-counted.
    """

    if not regime_name:
        raise ValueError("regime_name must be non-empty")
    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError(f"eligible_weight_count must be > 0, got {eligible_weight_count}")
    if target_bits_per_weight <= 0.0:
        raise ValueError("target_bits_per_weight must be > 0")
    if q_packed_data_bits_per_weight < 0.0:
        raise ValueError("q_packed_data_bits_per_weight must be >= 0")
    if q_packed_metadata_bits < 0:
        raise ValueError("q_packed_metadata_bits must be >= 0")
    if q_packed_padding_bits < 0:
        raise ValueError("q_packed_padding_bits must be >= 0")
    if frozen_scale_bits < 0:
        raise ValueError("frozen_scale_bits must be >= 0")
    if accumulator_bits_per_weight < 0.0:
        raise ValueError("accumulator_bits_per_weight must be >= 0")

    padding_bpw = _persistent_bits_per_weight(q_packed_padding_bits, eligible)
    if padding_bpw > q_packed_data_bits_per_weight:
        raise ValueError("diagnostic padding bits cannot exceed packed payload bits")
    metadata_bpw = _persistent_bits_per_weight(q_packed_metadata_bits, eligible)
    scale_bpw = _persistent_bits_per_weight(frozen_scale_bits, eligible)

    # Mirrors persistent_state_budget.measure_persistent_state_budget lines
    # 272-277: q_packed_total is actual payload data plus metadata, then scale.
    q_packed_total_bpw = float(q_packed_data_bits_per_weight) + metadata_bpw
    remaining_acc_budget = (
        float(target_bits_per_weight) - q_packed_total_bpw - scale_bpw
    )
    inclusive_bpw = q_packed_total_bpw + scale_bpw + float(accumulator_bits_per_weight)
    target_achieved = inclusive_bpw < float(target_bits_per_weight)
    q_entropy_preview_required = (
        q_packed_data_bits_per_weight < TARGET_PHYSICAL_BITS_PER_WEIGHT
        and not bool(q_entropy_code_overhead_accounted)
    )

    return AccumulatorFeasibilityReport(
        schema_version=ACCUMULATOR_COMPRESSION_CONTRACT_SCHEMA_VERSION,
        label=ACCUMULATOR_COMPRESSION_CONTRACT_LABEL,
        regime_name=regime_name,
        target_bits_per_weight=float(target_bits_per_weight),
        eligible_weight_count=eligible,
        q_packed_data_bits_per_weight=float(q_packed_data_bits_per_weight),
        q_packed_padding_bits=int(q_packed_padding_bits),
        q_packed_padding_bits_per_weight=padding_bpw,
        q_packed_metadata_bits=int(q_packed_metadata_bits),
        q_packed_metadata_bits_per_weight=metadata_bpw,
        q_packed_total_bits_per_weight=q_packed_total_bpw,
        frozen_scale_bits=int(frozen_scale_bits),
        frozen_scale_bits_per_weight=scale_bpw,
        accumulator_bits_per_weight=float(accumulator_bits_per_weight),
        remaining_accumulator_budget_bits_per_weight=remaining_acc_budget,
        packed_inclusive_physical_bits_per_weight=inclusive_bpw,
        target_achieved_with_reported_ledger=target_achieved,
        claimable_physical_sub2=bool(target_achieved and not q_entropy_preview_required),
        accumulator_only_sub2_possible_under_current_q=remaining_acc_budget > 0.0,
        q_entropy_code_overhead_accounted=bool(q_entropy_code_overhead_accounted),
        joint_q_entropy_route_status=(
            JOINT_Q_ENTROPY_PREVIEW
            if q_entropy_preview_required
            else "ledger_accounted_or_fixed_q"
        ),
        fixed_q_null_statement=(
            FIXED_Q_ACCUMULATOR_ONLY_NULL
            if remaining_acc_budget <= 0.0
            else "positive accumulator budget remains under the reported q ledger"
        ),
        status=NO_ENCODER_STATUS,
    )


def build_fixed_2bit_q_regime(
    *,
    regime_name: str,
    logical_shapes: Sequence[Sequence[int]],
    scale_count: int,
    accumulator_bits_per_weight: float = 16.0,
) -> AccumulatorFeasibilityReport:
    """Build a fixed 2-bit q regime from tensor shapes and source metadata constants."""

    if not logical_shapes:
        raise ValueError("logical_shapes must be non-empty")
    if int(scale_count) < 0:
        raise ValueError("scale_count must be >= 0")

    eligible = 0
    payload_bits = 0
    padding_bits = 0
    metadata_bits = 0
    for shape in logical_shapes:
        numel = _numel(shape)
        tensor_payload_bits, tensor_padding_bits = packed_2bit_payload_bits_and_padding(numel)
        eligible += numel
        payload_bits += tensor_payload_bits
        padding_bits += tensor_padding_bits
        metadata_bits += packed_ternary_metadata_bits_for_shape(shape)

    return evaluate_accumulator_feasibility(
        regime_name=regime_name,
        eligible_weight_count=eligible,
        q_packed_data_bits_per_weight=_persistent_bits_per_weight(payload_bits, eligible),
        q_packed_padding_bits=padding_bits,
        q_packed_metadata_bits=metadata_bits,
        frozen_scale_bits=int(scale_count) * 32,
        accumulator_bits_per_weight=float(accumulator_bits_per_weight),
        q_entropy_code_overhead_accounted=True,
    )


def default_fixed_q_feasibility_table() -> tuple[AccumulatorFeasibilityReport, ...]:
    """Compute the named C1.0 fixed-q lower-bound regimes."""

    realistic_shape = (4096, 4096)
    return (
        build_fixed_2bit_q_regime(
            regime_name="tiny_two_projection_fixture_fixed_2bit_q",
            logical_shapes=((8, 16), (4, 8)),
            scale_count=2,
        ),
        build_fixed_2bit_q_regime(
            regime_name="prior_large_fixture_fixed_2bit_q",
            logical_shapes=((128, 128),),
            scale_count=1,
        ),
        build_fixed_2bit_q_regime(
            regime_name="illustrative_4096x4096_one_tensor_one_scale_fixed_2bit_q",
            logical_shapes=(realistic_shape,),
            scale_count=1,
        ),
        build_fixed_2bit_q_regime(
            regime_name="illustrative_4096x4096_one_tensor_per_row_scale_fixed_2bit_q",
            logical_shapes=(realistic_shape,),
            scale_count=realistic_shape[0],
        ),
    )


def joint_q_entropy_preview(
    *,
    regime_name: str,
    eligible_weight_count: int,
    q_entropy_bits_per_weight: float,
    q_packed_metadata_bits: int,
    frozen_scale_bits: int,
    accumulator_bits_per_weight: float,
    q_packed_padding_bits: int = 0,
) -> AccumulatorFeasibilityReport:
    """Evaluate a q<2 preview while marking q-code overhead as unproven."""

    return evaluate_accumulator_feasibility(
        regime_name=regime_name,
        eligible_weight_count=eligible_weight_count,
        q_packed_data_bits_per_weight=float(q_entropy_bits_per_weight),
        q_packed_padding_bits=q_packed_padding_bits,
        q_packed_metadata_bits=q_packed_metadata_bits,
        frozen_scale_bits=frozen_scale_bits,
        accumulator_bits_per_weight=accumulator_bits_per_weight,
        q_entropy_code_overhead_accounted=False,
    )


def validate_accumulator_feasibility_report(
    report: AccumulatorFeasibilityReport,
    *,
    claimed_physical_sub2_achieved: bool = False,
) -> None:
    """Reject fixed-q or preview-only physical sub-2 overclaims."""

    recomputed_q_total = (
        report.q_packed_data_bits_per_weight + report.q_packed_metadata_bits_per_weight
    )
    if not math.isclose(report.q_packed_total_bits_per_weight, recomputed_q_total, abs_tol=1e-12):
        raise ValueError("q_packed_total must be packed payload data plus metadata only")
    recomputed_remaining = (
        report.target_bits_per_weight
        - report.q_packed_total_bits_per_weight
        - report.frozen_scale_bits_per_weight
    )
    if not math.isclose(
        report.remaining_accumulator_budget_bits_per_weight,
        recomputed_remaining,
        abs_tol=1e-12,
    ):
        raise ValueError("remaining accumulator budget must match source lower-bound formula")
    recomputed_inclusive = (
        report.q_packed_total_bits_per_weight
        + report.frozen_scale_bits_per_weight
        + report.accumulator_bits_per_weight
    )
    if not math.isclose(
        report.packed_inclusive_physical_bits_per_weight,
        recomputed_inclusive,
        abs_tol=1e-12,
    ):
        raise ValueError("inclusive physical bits/weight must include q, scale, and accumulator")
    recomputed_target = recomputed_inclusive < report.target_bits_per_weight
    if bool(report.target_achieved_with_reported_ledger) != bool(recomputed_target):
        raise ValueError("target flag must be computed from the reported inclusive ledger")
    if claimed_physical_sub2_achieved and not report.claimable_physical_sub2:
        raise ValueError(
            "physical sub-2 claim is not allowed for an over-target fixed-q or "
            "preview-only q-entropy ledger"
        )


def semantic_decision_surface_contract() -> tuple[SemanticDecisionDimension, ...]:
    """Required decision dimensions for C1 accumulator candidate classification."""

    return (
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="candidate_mask",
            source_anchor="vote_update.plan_integer_vote_update_reference:candidates",
            description="which flat rows cross threshold and can become local candidates",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="direction",
            source_anchor="vote_update.plan_integer_vote_update_reference:directions",
            description="sign of each threshold-crossing update",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="threshold_crossing",
            source_anchor="vote_update.plan_integer_vote_update_reference:new_acc_i32",
            description="threshold_abs crossing after decay, vote addition, and clipping",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="truncating_decay",
            source_anchor="vote_update.plan_integer_vote_update_reference:torch.div_trunc",
            description="integer decay with truncation toward zero",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="clip",
            source_anchor="vote_update.plan_integer_vote_update_reference:clamp",
            description="accumulator clip_min/clip_max effects before selection",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="residual_after_threshold",
            source_anchor="vote_update.apply_integer_vote_update_reference:residual",
            description="threshold subtraction and clamp after an applied flip",
        ),
        SemanticDecisionDimension(
            surface="vote_preplan",
            name="replay_veto_residual_rows",
            source_anchor="vote_update.apply_integer_vote_update_reference:replay_ce_veto_indices",
            description="replay-veto rows consume residual without q mutation",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="abs_new_acc_ranking",
            source_anchor="selection_topk/global_rate_cap:abs_new_acc",
            description="absolute new-accumulator score used for local and global priority",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="flat_index_tie_ordering",
            source_anchor="selection_topk:composite lower flat index tie-break",
            description="local equal-score tie-break by lower flat index",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="cross_state_global_flat_ordering",
            source_anchor="global_rate_cap:global_flat_index",
            description="cross-tensor tie ordering by global flat index",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="accepted_rows",
            source_anchor="global_rate_cap:accepted_rows",
            description="rows accepted by the global cap and allowed to mutate q/acc",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="deferred_rows",
            source_anchor="global_rate_cap:deferred_rows",
            description="rows deferred by the global cap with no threshold residual applied",
        ),
        SemanticDecisionDimension(
            surface="selection_global_cap",
            name="backlog_carry",
            source_anchor="global_rate_cap:deferred_backlog",
            description="deferred backlog carry/clear behavior across steps",
        ),
        SemanticDecisionDimension(
            surface="q_acc_apply",
            name="final_q_changes",
            source_anchor="vote_update/global_rate_cap:q_i16 mutation",
            description="final ternary q mutation count and row identities",
        ),
        SemanticDecisionDimension(
            surface="q_acc_apply",
            name="accumulator_residuals",
            source_anchor="vote_update/global_rate_cap:new_acc_i32 residuals",
            description="post-apply accumulator residual values after accepted and vetoed rows",
        ),
        SemanticDecisionDimension(
            surface="q_acc_apply",
            name="step_to_step_state_hashes",
            source_anchor="full_loop_receipt:state_input_hashes/state_output_hashes",
            description="state hash continuity across consecutive learner-loop steps",
        ),
    )


def required_decision_dimension_names() -> tuple[str, ...]:
    return tuple(item.name for item in semantic_decision_surface_contract())


def validate_candidate_assessment(assessment: CandidateAssessment) -> None:
    """Validate taxonomy, required decision coverage, and C2 eligibility claims."""

    classification = assessment.normalized_classification
    missing = assessment.missing_decision_dimensions
    if missing:
        raise ValueError(
            "classified accumulator candidates must cover every required "
            f"decision dimension; missing={missing}"
        )
    if classification == CandidateClassification.BOUNDED_DELTA_WITH_REPORT:
        if not assessment.bounded_delta_hypothesis or not assessment.guardrail:
            raise ValueError("bounded_delta candidates require a hypothesis and guardrail")
        if not assessment.preserved_information:
            raise ValueError("bounded_delta candidates require preserved_information")
        if not assessment.sub2_persistent_strategy:
            raise ValueError("bounded_delta candidates require a sub2_persistent_strategy")


def candidate_assessment(
    *,
    candidate_name: str,
    classification: CandidateClassification | str,
    covered_decision_dimensions: Iterable[str],
    compressed_representation: bool,
    bounded_delta_hypothesis: str | None = None,
    guardrail: str | None = None,
    preserved_information: Iterable[str] = (),
    sub2_persistent_strategy: str | None = None,
    note: str = "",
) -> CandidateAssessment:
    assessment = CandidateAssessment(
        candidate_name=candidate_name,
        classification=CandidateClassification(classification),
        covered_decision_dimensions=tuple(covered_decision_dimensions),
        compressed_representation=bool(compressed_representation),
        bounded_delta_hypothesis=bounded_delta_hypothesis,
        guardrail=guardrail,
        preserved_information=tuple(str(item) for item in preserved_information),
        sub2_persistent_strategy=sub2_persistent_strategy,
        note=note,
    )
    validate_candidate_assessment(assessment)
    return assessment


def identity_int16_baseline_assessment() -> CandidateAssessment:
    """The exact int16 path is a baseline/control, not compression progress."""

    return candidate_assessment(
        candidate_name=IDENTITY_INT16_BASELINE_NAME,
        classification=CandidateClassification.BIT_EXACT,
        covered_decision_dimensions=required_decision_dimension_names(),
        compressed_representation=False,
        note="bit-exact control path; not a compressed representation and not C2 progress",
    )


def accumulator_compression_contract_report() -> AccumulatorCompressionContractReport:
    """Return the compact C1.0 contract report without raw per-weight arrays."""

    return AccumulatorCompressionContractReport(
        schema_version=ACCUMULATOR_COMPRESSION_CONTRACT_SCHEMA_VERSION,
        label=ACCUMULATOR_COMPRESSION_CONTRACT_LABEL,
        status=NO_ENCODER_STATUS,
        feasibility_rows=default_fixed_q_feasibility_table(),
        semantic_decision_dimensions=semantic_decision_surface_contract(),
        candidate_taxonomy=tuple(item.value for item in CandidateClassification),
        identity_baseline=identity_int16_baseline_assessment(),
        non_claims=(
            "no accumulator encoder",
            "no trainer/live-run integration",
            "no acquisition or stability dynamics claim",
            "no .pt or creditdir mutation",
            "no physical sub-2 achievement under fixed 2-bit q",
            "compact report only; no raw per-weight arrays",
        ),
    )


def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        dim_i = int(dim)
        if dim_i <= 0:
            raise ValueError(f"logical_shape dims must be positive, got {tuple(shape)}")
        out *= dim_i
    return int(out)
