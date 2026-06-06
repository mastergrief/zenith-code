"""Strict sub-2 native-birth scaffold reporting for HRM-Text-1.58.

This slice is intentionally scaffold-first and fail-closed. It does NOT claim
an executable sub-2 learner. Instead it emits:

- the binding persistent candidate-path ledger subtotal,
- explicit off-path controls/blockers,
- adjacent transient/runtime ledgers kept separate from persistent authority,
- the dense-baseline parity contract for the first executable sub-2 proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
    HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
    INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
    bounded_delta_admission_contract,
    bounded_delta_candidate_assessment,
)
from calm.hrm_text_158.native_full_stack.fp_exceptions import (
    HIDDEN_FP_LEARNER_FAIL_STATE,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    base3_q_entropy_ledger_for_shapes,
    base3_q_storage_orthogonality_report,
)


STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION = (
    "hrm_text_158_strict_sub2_candidate_runtime_scaffold/v0"
)
STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL = (
    "strict_sub2_candidate_runtime_scaffold"
)
STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME = (
    "strict_sub2_candidate_runtime_scaffold"
)

RUNTIME_STATE_AUTHORITY_DENSE_CONTROL = "dense_control"
RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY = "sub2_scaffold_only"
RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE = "sub2_candidate_executable"

LEDGER_CLASS_LEQ2 = "<=2_bits"
LEDGER_CLASS_EXECUTABLE = "sub2_candidate_executable"
LEDGER_CLASS_NOT_YET = "not_yet_in_candidate_runtime"

PERSISTENT_CANDIDATE_SECTION = "persistent_candidate"
OFF_PATH_CONTROL_SECTION = "off_path_control"
ADJACENT_RUNTIME_SECTION = "adjacent_runtime"

ACQUISITION_GATE_DEFERRED = "deferred_until_after_parity_non_regression"


@dataclass(frozen=True)
class StrictSub2ScaffoldRow:
    name: str
    section: str
    classification: str
    in_candidate_authority: bool
    counted_in_physical_persistent_bpw: bool
    bits_per_weight: float | None
    blocker: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictSub2CandidateRuntimeScaffoldReport:
    schema_version: str
    label: str
    target_name: str
    runtime_state_authority: str
    pass_report: bool
    eligible_module_count: int
    eligible_weight_count: int
    physical_persistent_bpw: float
    physical_persistent_target_bpw: float
    physical_persistent_target_pass: bool
    physical_persistent_interpretation: str
    candidate_runtime_complete: bool
    candidate_authority_row_names: tuple[str, ...]
    blocker_names: tuple[str, ...]
    persistent_candidate_rows: tuple[StrictSub2ScaffoldRow, ...]
    off_path_control_rows: tuple[StrictSub2ScaffoldRow, ...]
    adjacent_runtime_rows: tuple[StrictSub2ScaffoldRow, ...]
    q_storage_orthogonality: dict[str, Any]
    parity_contract: dict[str, Any]
    acquisition_gate: dict[str, Any]
    hot_loop_residency: dict[str, Any]
    hidden_fp_learner_fail_state: str
    scoped_candidate_proof: dict[str, Any] | None
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "target_name": self.target_name,
            "runtime_state_authority": self.runtime_state_authority,
            "pass": bool(self.pass_report),
            "eligible_module_count": int(self.eligible_module_count),
            "eligible_weight_count": int(self.eligible_weight_count),
            "physical_persistent_bpw": float(self.physical_persistent_bpw),
            "physical_persistent_target_bpw": float(self.physical_persistent_target_bpw),
            "physical_persistent_target_pass": bool(self.physical_persistent_target_pass),
            "physical_persistent_interpretation": self.physical_persistent_interpretation,
            "candidate_runtime_complete": bool(self.candidate_runtime_complete),
            "candidate_authority_row_names": list(self.candidate_authority_row_names),
            "blocker_names": list(self.blocker_names),
            "persistent_candidate_rows": [row.to_dict() for row in self.persistent_candidate_rows],
            "off_path_control_rows": [row.to_dict() for row in self.off_path_control_rows],
            "adjacent_runtime_rows": [row.to_dict() for row in self.adjacent_runtime_rows],
            "q_storage_orthogonality": dict(self.q_storage_orthogonality),
            "parity_contract": dict(self.parity_contract),
            "acquisition_gate": dict(self.acquisition_gate),
            "hot_loop_residency": dict(self.hot_loop_residency),
            "hidden_fp_learner_fail_state": self.hidden_fp_learner_fail_state,
            "scoped_candidate_proof": (
                None if self.scoped_candidate_proof is None else dict(self.scoped_candidate_proof)
            ),
            "non_claims": list(self.non_claims),
        }


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(dim) for dim in shape)
    if not out or any(dim <= 0 for dim in out):
        raise ValueError(f"eligible module shapes must be non-empty positive tuples, got {shape!r}")
    return out


def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def _scale_bits_per_weight(scale_count: int, eligible_weight_count: int) -> float:
    return float(scale_count * 32) / float(eligible_weight_count)


def _max_activation_bits_per_value(activation_paid_bits_ledger: Mapping[str, Any]) -> float | None:
    surfaces = activation_paid_bits_ledger.get("surfaces")
    if not isinstance(surfaces, Sequence) or not surfaces:
        return None
    values = []
    for row in surfaces:
        if not isinstance(row, Mapping):
            continue
        paid = row.get("paid_bits_per_value")
        if paid is None:
            continue
        values.append(float(paid))
    return max(values) if values else None


def _all_rows(
    report: StrictSub2CandidateRuntimeScaffoldReport,
) -> tuple[StrictSub2ScaffoldRow, ...]:
    return (
        report.persistent_candidate_rows
        + report.off_path_control_rows
        + report.adjacent_runtime_rows
    )


def validate_strict_sub2_candidate_runtime_scaffold_report(
    report: StrictSub2CandidateRuntimeScaffoldReport,
) -> None:
    if report.schema_version != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION:
        raise ValueError("strict sub-2 scaffold schema version mismatch")
    if report.label != STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL:
        raise ValueError("strict sub-2 scaffold label mismatch")
    if report.target_name != STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME:
        raise ValueError("strict sub-2 scaffold target name mismatch")
    if report.runtime_state_authority not in {
        RUNTIME_STATE_AUTHORITY_DENSE_CONTROL,
        RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
        RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE,
    }:
        raise ValueError("unknown runtime_state_authority")
    if report.runtime_state_authority != RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY:
        raise ValueError("this slice must emit runtime_state_authority=sub2_scaffold_only")
    allowed = {LEDGER_CLASS_LEQ2, LEDGER_CLASS_EXECUTABLE, LEDGER_CLASS_NOT_YET}
    for row in _all_rows(report):
        if row.classification not in allowed:
            raise ValueError(f"unknown ledger row classification for {row.name!r}")
        if row.counted_in_physical_persistent_bpw and not row.in_candidate_authority:
            raise ValueError(f"{row.name!r} cannot count toward persistent bpw outside candidate authority")
        if row.in_candidate_authority and row.classification == LEDGER_CLASS_NOT_YET:
            raise ValueError(f"{row.name!r} cannot be in candidate authority while blocked")
        if row.counted_in_physical_persistent_bpw:
            if row.bits_per_weight is None:
                raise ValueError(f"{row.name!r} counted in persistent bpw must disclose bits_per_weight")
            if float(row.bits_per_weight) >= 2.0:
                raise ValueError(f"{row.name!r} candidate-authority row exceeds the strict sub-2 limit")
    recomputed_bpw = sum(
        float(row.bits_per_weight)
        for row in report.persistent_candidate_rows
        if row.counted_in_physical_persistent_bpw
    )
    if abs(recomputed_bpw - float(report.physical_persistent_bpw)) > 1e-12:
        raise ValueError("physical_persistent_bpw must equal the counted candidate-authority subtotal")
    if bool(report.physical_persistent_target_pass) != (float(report.physical_persistent_bpw) < float(report.physical_persistent_target_bpw)):
        raise ValueError("physical_persistent_target_pass must be computed from the binding subtotal")
    if report.candidate_runtime_complete:
        raise ValueError("this scaffold slice must stay non-executable/candidate_runtime_complete=false")
    if report.acquisition_gate.get("status") != ACQUISITION_GATE_DEFERRED:
        raise ValueError("acquisition must be explicitly deferred in the scaffold-first slice")
    serialized = str(report.to_dict())
    if "justified_fp_exception" in serialized:
        raise ValueError("justified_fp_exception labels are forbidden in the candidate path")
    if report.hot_loop_residency.get("qacc_kernelized") is not False:
        raise ValueError("this slice must still disclose qacc_kernelized=false")
    hot = report.hot_loop_residency.get("hot_loop_residency", {})
    if hot.get("qacc_update_over_64") != "cpu_reference":
        raise ValueError("this slice must still disclose qacc_update_over_64=cpu_reference")
    if report.hidden_fp_learner_fail_state != HIDDEN_FP_LEARNER_FAIL_STATE:
        raise ValueError("hidden FP learner fail state must be preserved verbatim")
    if not bool(report.pass_report):
        raise ValueError("scaffold report pass flag must reflect a valid fail-closed scaffold, not executable success")
    if report.scoped_candidate_proof is not None:
        proof = dict(report.scoped_candidate_proof)
        if proof.get("surface") != "accumulator_substitute":
            raise ValueError("scoped candidate proof must stay on accumulator_substitute only")
        if proof.get("runtime_state_authority_after") != RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY:
            raise ValueError("scoped candidate proof must leave runtime_state_authority scaffold-only")
        if bool(proof.get("candidate_dense_decode_used")):
            raise ValueError("scoped candidate proof cannot use dense decode on the candidate path")
        if bool(proof.get("candidate_accumulator_transient_over2_used")):
            raise ValueError("scoped candidate proof cannot use >2-bit accumulator transients")
        if bool(proof.get("candidate_vote_transient_over2_used")):
            raise ValueError("scoped candidate proof cannot use dense vote transients")
        if bool(proof.get("candidate_dense_vote_authority_used")):
            raise ValueError("scoped candidate proof cannot use dense vote authority")
        if proof.get("q_storage_physical_budget_covered_by_scoped_proof") is not False:
            raise ValueError("scoped candidate proof must not claim q-storage physical budget coverage")
        if proof.get("frozen_scale_physical_budget_covered_by_scoped_proof") is not False:
            raise ValueError("scoped candidate proof must not claim frozen-scale physical budget coverage")
        if not isinstance(proof.get("coverage_domain"), Mapping):
            raise ValueError("scoped candidate proof must disclose its coverage domain")
        terminal = proof.get("terminal_classification")
        if not bool(proof.get("pass")) and terminal != INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP:
            raise ValueError("negative scoped candidate proof must land as the intrinsic domain-gap null")
        if terminal not in {
            ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
            ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2,
            INTRINSIC_BOUNDED_UPDATE_DOMAIN_GAP,
        }:
            raise ValueError("scoped candidate proof terminal classification is unknown")
        if bool(proof.get("pass")):
            if not isinstance(proof.get("storage_projection"), Mapping):
                raise ValueError("positive scoped candidate proof must disclose storage_projection")
            accumulator_bpw = float(
                proof["storage_projection"].get("bounded_delta_acc_bits_per_weight")
            )
            if proof.get("scoped_label") == ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE:
                if accumulator_bpw >= 2.0:
                    raise ValueError(
                        "positive scoped candidate proof with the physical local-vote label "
                        "must validate storage_projection bounded_delta_acc_bits_per_weight < 2"
                    )
                if proof.get("scoped_physical_budget_claim") != "physical_sub2_budgeted":
                    raise ValueError(
                        "physical local-vote label must explicitly claim physical_sub2_budgeted"
                    )
            elif proof.get("scoped_label") == ALGORITHMIC_LOCAL_VOTE_UPDATE_EXECUTABLE_NOT_PHYSICAL_SUB2:
                if proof.get("scoped_physical_budget_claim") != "algorithmic_only_not_physical_sub2":
                    raise ValueError(
                        "algorithmic-only local-vote label must explicitly reject physical-sub2 interpretation"
                    )
            else:
                raise ValueError("positive scoped candidate proof uses an unknown positive label")
        persistent_rows = {
            row.name: row
            for row in report.persistent_candidate_rows
        }
        accumulator_row = persistent_rows["accumulator_substitute"]
        if accumulator_row.in_candidate_authority or accumulator_row.classification != LEDGER_CLASS_NOT_YET:
            raise ValueError(
                "scoped candidate proof cannot silently promote the full accumulator row; "
                "it stays blocked/not-yet until broader decision dimensions are covered"
            )


def build_strict_sub2_candidate_runtime_scaffold(
    *,
    eligible_module_shapes: Mapping[str, Sequence[int]],
    activation_paid_bits_ledger: Mapping[str, Any],
    live_both_gate: Mapping[str, Any],
    hot_loop_residency: Mapping[str, Any],
    candidate_name: str = HYBRID_HOT_EXACT_COLD_DEFAULT_CANDIDATE,
) -> StrictSub2CandidateRuntimeScaffoldReport:
    if not eligible_module_shapes:
        raise ValueError("eligible_module_shapes must be non-empty")
    ordered_shapes = tuple(
        _shape_tuple(eligible_module_shapes[name])
        for name in sorted(eligible_module_shapes)
    )
    eligible_weight_count = sum(_numel(shape) for shape in ordered_shapes)
    scale_count = len(ordered_shapes)
    q_storage = base3_q_entropy_ledger_for_shapes(
        regime_name="strict_sub2_candidate_q_storage_shape_ledger",
        logical_shapes=ordered_shapes,
        scale_count=0,
        accumulator_bits_per_weight=0.0,
    )
    q_orthogonality = base3_q_storage_orthogonality_report().to_dict()
    admission_contract = bounded_delta_admission_contract(candidate_name=candidate_name)
    candidate_assessment = bounded_delta_candidate_assessment(candidate_name=candidate_name)
    scale_bpw = _scale_bits_per_weight(scale_count, eligible_weight_count)
    activation_bits = _max_activation_bits_per_value(activation_paid_bits_ledger)
    activation_packable = bool(activation_paid_bits_ledger.get("pass"))
    kv_uncovered = "kv_cache.append_update" in tuple(live_both_gate.get("not_covered", ()))
    hot_loop = dict(hot_loop_residency)
    qacc_kernelized = bool(hot_loop.get("qacc_kernelized"))
    qacc_hot_loop = dict(hot_loop.get("hot_loop_residency", {}))
    qacc_cpu_reference = (
        qacc_hot_loop.get("qacc_update_over_64") == "cpu_reference"
        or qacc_hot_loop.get("qacc_vote_selection") == "cpu_reference"
        or qacc_hot_loop.get("qacc_apply_vote_step") == "cpu_reference"
    )

    persistent_candidate_rows = (
        StrictSub2ScaffoldRow(
            name="q_storage",
            section=PERSISTENT_CANDIDATE_SECTION,
            classification=LEDGER_CLASS_LEQ2,
            in_candidate_authority=True,
            counted_in_physical_persistent_bpw=True,
            bits_per_weight=float(q_storage.q_packed_total_bits_per_weight),
            blocker=False,
            rationale=(
                "Base-3 q storage is the current strict candidate-authority row. "
                "Its ledger is shape-derived and orthogonal to accumulator progress."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="accumulator_substitute",
            section=PERSISTENT_CANDIDATE_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=True,
            rationale=(
                "Default slot is the bounded-delta candidate "
                f"{candidate_name!r}, but this slice keeps it scaffold/reference-only: "
                "the adapter/oracle path has a real admission contract and capacity "
                "hypothesis, yet no executable materialize/update/collapse runtime."
            ),
        ),
    )
    off_path_control_rows = (
        StrictSub2ScaffoldRow(
            name="frozen_scales_fp32_metadata",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=float(scale_bpw),
            blocker=True,
            rationale=(
                "Frozen scales are >2 physical bits per component and therefore "
                "must remain off-path controls/blockers until a strict <=2-bit "
                "replacement exists."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="dense_int16_accumulator_control",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=16.0,
            blocker=True,
            rationale="Dense int16 accumulator is the banked control/baseline only, never candidate authority.",
        ),
        StrictSub2ScaffoldRow(
            name="fp_shell_and_noneligible_fp_controls",
            section=OFF_PATH_CONTROL_SECTION,
            classification=LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=True,
            rationale=(
                "FP shell, lm_head/embeddings/norms, and other >2-bit or non-eligible "
                "tensors remain off-path controls only; hidden FP learning stays a hard fail state."
            ),
        ),
    )
    adjacent_runtime_rows = (
        StrictSub2ScaffoldRow(
            name="activations_and_residual_runtime_packability",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_LEQ2 if activation_packable else LEDGER_CLASS_NOT_YET,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=activation_bits,
            blocker=not activation_packable,
            rationale=(
                "Adjacent runtime activation surfaces stay outside persistent authority, "
                "but must be tracked beside it. This row is sourced from the harness "
                "activation paid-bits ledger."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="attention_kv_append_update",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_NOT_YET if kv_uncovered else LEDGER_CLASS_EXECUTABLE,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=kv_uncovered,
            rationale=(
                "KV append/update remains explicit uncovered/estimator-only in the current "
                "runtime gate and therefore blocks a complete candidate runtime."
            ),
        ),
        StrictSub2ScaffoldRow(
            name="qacc_hot_loop_residency",
            section=ADJACENT_RUNTIME_SECTION,
            classification=LEDGER_CLASS_NOT_YET if (qacc_cpu_reference or not qacc_kernelized) else LEDGER_CLASS_EXECUTABLE,
            in_candidate_authority=False,
            counted_in_physical_persistent_bpw=False,
            bits_per_weight=None,
            blocker=bool(qacc_cpu_reference or not qacc_kernelized),
            rationale=(
                "The standing speed blocker remains the qacc hot loop: vote selection/apply/"
                "update are still CPU-reference today, so hot-loop GPU residency is not yet in the candidate runtime."
            ),
        ),
    )

    blocker_names = tuple(
        row.name for row in (
            persistent_candidate_rows + off_path_control_rows + adjacent_runtime_rows
        ) if row.blocker
    )
    candidate_authority_row_names = tuple(
        row.name for row in persistent_candidate_rows if row.in_candidate_authority
    )
    physical_persistent_bpw = sum(
        float(row.bits_per_weight)
        for row in persistent_candidate_rows
        if row.counted_in_physical_persistent_bpw
    )
    parity_contract = {
        "candidate_name": candidate_name,
        "candidate_assessment": candidate_assessment.to_dict(),
        "preserved_information": list(admission_contract.preserved_information),
        "capacity_hypothesis": admission_contract.capacity_hypothesis,
        "sub2_persistent_strategy": admission_contract.sub2_persistent_strategy,
        "exact_surfaces": list(admission_contract.exact_surfaces),
        "allowed_divergence_contract": admission_contract.allowed_divergence_contract,
        "dense_baseline_non_regression_required": True,
        "executable_proof_before_acquisition": [
            "physical_persistent_bpw < 2.0",
            "dense int16 accumulator absent from candidate persistent authority",
            "hidden FP learner absent",
            "q storage orthogonality preserved",
            "q_changed identities/counts parity contract present",
            "accepted/deferred/backlog/frontier guard surfaces declared",
        ],
        "adapter_oracle_only": True,
        "acquisition_not_used_as_first_gate": True,
    }
    acquisition_gate = {
        "status": ACQUISITION_GATE_DEFERRED,
        "support_name": "L0c2-K2-addition-120",
        "reason": (
            "fork-b scaffold-first: the first executable proof is strict ledger + "
            "runtime authority + dense-baseline parity/non-regression, not acquisition"
        ),
    }

    report = StrictSub2CandidateRuntimeScaffoldReport(
        schema_version=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION,
        label=STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL,
        target_name=STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME,
        runtime_state_authority=RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY,
        pass_report=True,
        eligible_module_count=len(ordered_shapes),
        eligible_weight_count=eligible_weight_count,
        physical_persistent_bpw=float(physical_persistent_bpw),
        physical_persistent_target_bpw=2.0,
        physical_persistent_target_pass=bool(physical_persistent_bpw < 2.0),
        physical_persistent_interpretation=(
            "candidate-authority subtotal only; fail-closed scaffold excludes blockers/"
            "off-path controls from the binding persistent learner-state sum"
        ),
        candidate_runtime_complete=False,
        candidate_authority_row_names=candidate_authority_row_names,
        blocker_names=blocker_names,
        persistent_candidate_rows=persistent_candidate_rows,
        off_path_control_rows=off_path_control_rows,
        adjacent_runtime_rows=adjacent_runtime_rows,
        q_storage_orthogonality=q_orthogonality,
        parity_contract=parity_contract,
        acquisition_gate=acquisition_gate,
        hot_loop_residency=hot_loop,
        hidden_fp_learner_fail_state=HIDDEN_FP_LEARNER_FAIL_STATE,
        scoped_candidate_proof=None,
        non_claims=(
            "scaffold-only; no sub-2 learner achieved claim",
            "runtime_state_authority stays sub2_scaffold_only in this slice",
            "no executable bounded-delta authority until a real materialize/update/collapse path exists",
            "no acquisition or retention claim in this slice",
        ),
    )
    validate_strict_sub2_candidate_runtime_scaffold_report(report)
    return report


def attach_strict_sub2_scoped_candidate_proof(
    report: StrictSub2CandidateRuntimeScaffoldReport,
    *,
    scoped_candidate_proof: Mapping[str, Any],
) -> StrictSub2CandidateRuntimeScaffoldReport:
    updated = replace(
        report,
        scoped_candidate_proof=dict(scoped_candidate_proof),
    )
    validate_strict_sub2_candidate_runtime_scaffold_report(updated)
    return updated


__all__ = [
    "ACQUISITION_GATE_DEFERRED",
    "LEDGER_CLASS_EXECUTABLE",
    "LEDGER_CLASS_LEQ2",
    "LEDGER_CLASS_NOT_YET",
    "RUNTIME_STATE_AUTHORITY_DENSE_CONTROL",
    "RUNTIME_STATE_AUTHORITY_SUB2_CANDIDATE_EXECUTABLE",
    "RUNTIME_STATE_AUTHORITY_SUB2_SCAFFOLD_ONLY",
    "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_LABEL",
    "STRICT_SUB2_CANDIDATE_RUNTIME_SCAFFOLD_SCHEMA_VERSION",
    "STRICT_SUB2_CANDIDATE_RUNTIME_TARGET_NAME",
    "StrictSub2CandidateRuntimeScaffoldReport",
    "StrictSub2ScaffoldRow",
    "attach_strict_sub2_scoped_candidate_proof",
    "build_strict_sub2_candidate_runtime_scaffold",
    "validate_strict_sub2_candidate_runtime_scaffold_report",
]
