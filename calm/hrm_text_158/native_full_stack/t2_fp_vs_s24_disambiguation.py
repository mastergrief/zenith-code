"""Read-only T2 FP-vs-S24 disambiguation diagnostic for HRM-Text-1.58.

Classifies banked T2 zero-moves as shift-too-coarse (CLASS_A) or input-too-easy
(CLASS_B) on anchored captures from capture_tier2_checkpoint_raw_captures.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    project_s1_gradient_to_moves,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    IntegerMarginalAttributionEvents,
    _accumulate_cpu_reference_dense_int32_scratch,
    _rescale_accumulator_to_attribution_q,
    integer_marginal_attribution_from_captures,
    projected_moves_from_integer_attribution,
)
from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    RawKeyCapture,
    Tier2RawCaptureBundle,
    capture_tier2_checkpoint_raw_captures,
    sha256_file,
)

T2_FP_VS_S24_DISAMBIGUATION_SCHEMA_VERSION = "hrm_text_158_t2_fp_vs_s24_disambiguation/v1"
T2_FP_VS_S24_DISAMBIGUATION_TARGET_NAME = "t2_fp_vs_s24_disambiguation"

FROZEN_T2_ANCHOR_CHECKPOINT_SHA256 = (
    "7055aed07d309d0762dbeccdc0703bd24de24fb7604166b752bc3ee11e9a406f"
)
FROZEN_T2_ANCHOR_KEY_SET_SHA256 = (
    "7e4571ae7ece7dea86de255b98bdc8b097e6335f0dbe65cbbca919ecdcff5aec"
)
FROZEN_T2_ANCHOR_KEY_COUNT = 32
FROZEN_T2_ANCHOR_CURRICULUM_SEED = 158
FROZEN_T2_ANCHOR_BATCH_SIZE = 4
T2_DISAMBIGUATION_CAPTURE_SEAM_ID = "capture_tier2_checkpoint_raw_captures"

BRANCH_SHIFT_TOO_COARSE = "BR-T2-DISAMBIG-SHIFT-TOO-COARSE"
BRANCH_INPUT_TOO_EASY = "BR-T2-DISAMBIG-INPUT-TOO-EASY"
BRANCH_INCONSISTENT = "BR-T2-DISAMBIG-INCONSISTENT"
BRANCH_UNRESOLVED = "BR-T2-DISAMBIG-UNRESOLVED"

RECOMMEND_REOPEN_SHIFT_FINER = "reopen_shift_finer"
RECOMMEND_FIX_T2_INPUT = "fix_t2_input"
RECOMMEND_INVESTIGATE_CAPTURE = "investigate_capture"
RECOMMEND_UNRESOLVED_ENRICH = "unresolved_enrich"

LSB_S24 = 2.0 ** (24 - 31)
LSB_S16 = 2.0 ** (16 - 31)
HYPOTHETICAL_S16_RESCALE_SHIFT = 16

T2_FP_VS_S24_DISAMBIGUATION_HARD_FALSE_FIELDS = (
    "ready_to_flip",
    "optimizer_credit_state_sub2_claim",
    "optimizer_credit_state_resolved",
    "readiness_row_flip_authorized",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "gpu_runtime_receipt_present",
)

T2_FP_VS_S24_DISAMBIGUATION_NON_CLAIMS = (
    "t2 fp-vs-s24 disambiguation is CPU read-only diagnostic evidence only",
    "pass_receipt is always false; diagnostic does not flip optimizer_credit_state row",
    "classification requires anchored capture seam and self-consistency with banked T2",
    "no production S=24 law change, GPU runtime, readiness flip, or checkpoint mutation",
)


@dataclass(frozen=True)
class T2KeyDisambiguationMetrics:
    state_key: str
    fp_move_count: int
    s24_move_count: int
    wg_abs_min: float
    wg_abs_median: float
    wg_abs_max: float
    wg_below_lsb_s24: int
    wg_below_lsb_s16: int
    wg_exact_zero: int
    wg_above_lsb_s16: int
    total_weight_elements: int
    hypothetical_s16_move_count: int

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class T2FpVsS24DisambiguationReceipt:
    schema_version: str
    target_name: str
    pass_receipt: bool
    anchor_checkpoint_sha256: str
    anchor_curriculum_seed: int
    anchor_batch_size: int
    anchor_key_count: int
    anchor_key_set_sha256: str
    anchor_capture_seam_id: str
    anchor_precondition_pass: bool
    banked_s24_move_total: int
    self_consistency_pass: bool
    per_key_metrics: tuple[T2KeyDisambiguationMetrics, ...]
    keys_with_fp_moves_gt0: int
    keys_with_s24_moves_gt0: int
    keys_fp_gt0_s24_eq0: int
    keys_fp_gt0_s24_eq0_fraction: float
    keys_both_zero: int
    wg_global_below_s24_nonzero: int
    wg_global_above_s16: int
    hypothetical_s16_move_total: int
    global_wg_abs_max: float
    exact_zero_fraction_all_elements: float
    branch_id: str
    recommended_next_slice: str
    ready_to_flip: bool
    optimizer_credit_state_sub2_claim: bool
    optimizer_credit_state_resolved: bool
    readiness_row_flip_authorized: bool
    real_native_integer_attribution_present: bool
    real_native_integer_credit_ranking_present: bool
    gpu_runtime_receipt_present: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            field.name: getattr(self, field.name)
            if field.name != "per_key_metrics"
            else [item.to_dict() for item in self.per_key_metrics]
            for field in fields(self)
        }
        payload["non_claims"] = list(self.non_claims)
        return payload


def anchor_key_set_sha256(keys: Sequence[str]) -> str:
    blob = "\n".join(sorted(keys))
    return hashlib.sha256(blob.encode()).hexdigest()


def _count_nonzero_moves(moves: torch.Tensor) -> int:
    return int(torch.count_nonzero(moves).item())


def _wg_distribution_metrics(weighted_grad: torch.Tensor) -> dict[str, float | int]:
    flat = weighted_grad.detach().cpu().reshape(-1).to(torch.float32)
    total = int(flat.numel())
    if total == 0:
        return {
            "wg_abs_min": 0.0,
            "wg_abs_median": 0.0,
            "wg_abs_max": 0.0,
            "wg_below_lsb_s24": 0,
            "wg_below_lsb_s16": 0,
            "wg_exact_zero": 0,
            "wg_above_lsb_s16": 0,
            "total_weight_elements": 0,
        }
    abs_vals = flat.abs()
    nonzero_mask = abs_vals > 0
    nonzero = abs_vals[nonzero_mask]
    return {
        "wg_abs_min": float(abs_vals.min().item()),
        "wg_abs_median": float(abs_vals.median().item()),
        "wg_abs_max": float(abs_vals.max().item()),
        "wg_below_lsb_s24": int((nonzero < LSB_S24).sum().item()) if nonzero.numel() else 0,
        "wg_below_lsb_s16": int((nonzero < LSB_S16).sum().item()) if nonzero.numel() else 0,
        "wg_exact_zero": int((abs_vals == 0).sum().item()),
        "wg_above_lsb_s16": int((abs_vals >= LSB_S16).sum().item()),
        "total_weight_elements": total,
    }


def attribution_events_with_rescale_shift(
    capture: RawKeyCapture,
    *,
    rescale_shift: int,
) -> IntegerMarginalAttributionEvents:
    weight_dims = capture.weight_shape
    paired_inputs = capture.inputs[-len(capture.grad_outputs) :]
    grad_outputs_reversed = list(reversed(list(capture.grad_outputs)))
    accumulator = _accumulate_cpu_reference_dense_int32_scratch(
        paired_inputs,
        grad_outputs_reversed,
        weight_shape=weight_dims,
    )
    attribution_dense = _rescale_accumulator_to_attribution_q(
        accumulator,
        shift=int(rescale_shift),
    )
    flat = attribution_dense.reshape(-1)
    nz = torch.nonzero(flat != 0, as_tuple=False).flatten().to(torch.int64)
    numel = int(weight_dims[0] * weight_dims[1])
    events = IntegerMarginalAttributionEvents(
        flat_indices=nz.contiguous(),
        attribution_q31=flat.index_select(0, nz).to(torch.int32).contiguous(),
        law_id=INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
        numel=numel,
        index_set_policy=INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    )
    events.validate()
    return events


def compute_key_disambiguation_metrics(capture: RawKeyCapture) -> T2KeyDisambiguationMetrics:
    weight_dims = capture.weight_shape
    weighted_grad = weighted_grad_from_captures(
        capture.inputs,
        capture.grad_outputs,
        weight_shape=weight_dims,
    )
    fp_moves = project_s1_gradient_to_moves(
        weighted_grad,
        capture.q_levels_flat.reshape(weight_dims),
    )
    s24_events = integer_marginal_attribution_from_captures(
        capture.inputs,
        capture.grad_outputs,
        weight_shape=weight_dims,
        law_id=INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    )
    _s24_indices, s24_moves = projected_moves_from_integer_attribution(
        s24_events,
        capture.q_levels_flat,
    )
    s16_events = attribution_events_with_rescale_shift(
        capture,
        rescale_shift=HYPOTHETICAL_S16_RESCALE_SHIFT,
    )
    _s16_indices, s16_moves = projected_moves_from_integer_attribution(
        s16_events,
        capture.q_levels_flat,
    )
    wg_stats = _wg_distribution_metrics(weighted_grad)
    return T2KeyDisambiguationMetrics(
        state_key="",
        fp_move_count=_count_nonzero_moves(fp_moves),
        s24_move_count=_count_nonzero_moves(s24_moves),
        wg_abs_min=float(wg_stats["wg_abs_min"]),
        wg_abs_median=float(wg_stats["wg_abs_median"]),
        wg_abs_max=float(wg_stats["wg_abs_max"]),
        wg_below_lsb_s24=int(wg_stats["wg_below_lsb_s24"]),
        wg_below_lsb_s16=int(wg_stats["wg_below_lsb_s16"]),
        wg_exact_zero=int(wg_stats["wg_exact_zero"]),
        wg_above_lsb_s16=int(wg_stats["wg_above_lsb_s16"]),
        total_weight_elements=int(wg_stats["total_weight_elements"]),
        hypothetical_s16_move_count=_count_nonzero_moves(s16_moves),
    )


def evaluate_anchor_precondition(
    bundle: Tier2RawCaptureBundle,
    *,
    expected_checkpoint_sha256: str = FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    expected_key_set_sha256: str = FROZEN_T2_ANCHOR_KEY_SET_SHA256,
    expected_key_count: int = FROZEN_T2_ANCHOR_KEY_COUNT,
    expected_curriculum_seed: int = FROZEN_T2_ANCHOR_CURRICULUM_SEED,
    expected_batch_size: int = FROZEN_T2_ANCHOR_BATCH_SIZE,
    expected_capture_seam_id: str = T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
) -> tuple[bool, dict[str, Any]]:
    provenance = bundle.provenance
    key_list = sorted(bundle.per_key_captures.keys())
    observed_key_sha = anchor_key_set_sha256(key_list)
    anchor_fields = {
        "anchor_checkpoint_sha256": str(provenance.get("checkpoint_sha256", "")),
        "anchor_curriculum_seed": int(provenance.get("curriculum_seed", -1)),
        "anchor_batch_size": int(provenance.get("batch_size", -1)),
        "anchor_key_count": len(key_list),
        "anchor_key_set_sha256": observed_key_sha,
        "anchor_capture_seam_id": str(provenance.get("capture_seam_id", "")),
    }
    anchor_precondition_pass = (
        anchor_fields["anchor_checkpoint_sha256"] == expected_checkpoint_sha256
        and anchor_fields["anchor_curriculum_seed"] == expected_curriculum_seed
        and anchor_fields["anchor_batch_size"] == expected_batch_size
        and anchor_fields["anchor_key_count"] == expected_key_count
        and anchor_fields["anchor_key_set_sha256"] == expected_key_set_sha256
        and anchor_fields["anchor_capture_seam_id"] == expected_capture_seam_id
    )
    anchor_fields["anchor_precondition_pass"] = anchor_precondition_pass
    return anchor_precondition_pass, anchor_fields


def build_disambiguation_metrics_from_bundle(
    bundle: Tier2RawCaptureBundle,
) -> list[T2KeyDisambiguationMetrics]:
    metrics: list[T2KeyDisambiguationMetrics] = []
    for key in sorted(bundle.per_key_captures.keys()):
        item = compute_key_disambiguation_metrics(bundle.per_key_captures[key])
        metrics.append(
            T2KeyDisambiguationMetrics(
                state_key=key,
                fp_move_count=item.fp_move_count,
                s24_move_count=item.s24_move_count,
                wg_abs_min=item.wg_abs_min,
                wg_abs_median=item.wg_abs_median,
                wg_abs_max=item.wg_abs_max,
                wg_below_lsb_s24=item.wg_below_lsb_s24,
                wg_below_lsb_s16=item.wg_below_lsb_s16,
                wg_exact_zero=item.wg_exact_zero,
                wg_above_lsb_s16=item.wg_above_lsb_s16,
                total_weight_elements=item.total_weight_elements,
                hypothetical_s16_move_count=item.hypothetical_s16_move_count,
            )
        )
    return metrics


def aggregate_disambiguation_metrics(
    per_key_metrics: Sequence[T2KeyDisambiguationMetrics],
) -> dict[str, float | int]:
    keys_with_fp_moves_gt0 = sum(1 for item in per_key_metrics if item.fp_move_count > 0)
    keys_with_s24_moves_gt0 = sum(1 for item in per_key_metrics if item.s24_move_count > 0)
    keys_fp_gt0_s24_eq0 = sum(
        1 for item in per_key_metrics if item.fp_move_count > 0 and item.s24_move_count == 0
    )
    keys_both_zero = sum(
        1 for item in per_key_metrics if item.fp_move_count == 0 and item.s24_move_count == 0
    )
    key_count = len(per_key_metrics)
    total_elements = sum(item.total_weight_elements for item in per_key_metrics)
    exact_zero_total = sum(item.wg_exact_zero for item in per_key_metrics)
    return {
        "keys_with_fp_moves_gt0": keys_with_fp_moves_gt0,
        "keys_with_s24_moves_gt0": keys_with_s24_moves_gt0,
        "keys_fp_gt0_s24_eq0": keys_fp_gt0_s24_eq0,
        "keys_fp_gt0_s24_eq0_fraction": float(keys_fp_gt0_s24_eq0 / max(key_count, 1)),
        "keys_both_zero": keys_both_zero,
        "wg_global_below_s24_nonzero": sum(item.wg_below_lsb_s24 for item in per_key_metrics),
        "wg_global_above_s16": sum(item.wg_above_lsb_s16 for item in per_key_metrics),
        "hypothetical_s16_move_total": sum(
            item.hypothetical_s16_move_count for item in per_key_metrics
        ),
        "global_wg_abs_max": max((item.wg_abs_max for item in per_key_metrics), default=0.0),
        "exact_zero_fraction_all_elements": float(
            exact_zero_total / max(total_elements, 1)
        ),
        "banked_s24_move_total": sum(item.s24_move_count for item in per_key_metrics),
    }


def _frozen_anchor_fields_match(receipt: T2FpVsS24DisambiguationReceipt | Mapping[str, Any]) -> bool:
    if isinstance(receipt, T2FpVsS24DisambiguationReceipt):
        getter = lambda name: getattr(receipt, name)
    else:
        getter = receipt.get
    return (
        getter("anchor_checkpoint_sha256") == FROZEN_T2_ANCHOR_CHECKPOINT_SHA256
        and int(getter("anchor_curriculum_seed")) == FROZEN_T2_ANCHOR_CURRICULUM_SEED
        and int(getter("anchor_batch_size")) == FROZEN_T2_ANCHOR_BATCH_SIZE
        and int(getter("anchor_key_count")) == FROZEN_T2_ANCHOR_KEY_COUNT
        and getter("anchor_key_set_sha256") == FROZEN_T2_ANCHOR_KEY_SET_SHA256
        and getter("anchor_capture_seam_id") == T2_DISAMBIGUATION_CAPTURE_SEAM_ID
    )


def _validate_recomputed_aggregates(
    receipt: T2FpVsS24DisambiguationReceipt,
    aggregates: Mapping[str, float | int],
) -> None:
    int_fields = (
        "keys_with_fp_moves_gt0",
        "keys_with_s24_moves_gt0",
        "keys_fp_gt0_s24_eq0",
        "keys_both_zero",
        "wg_global_below_s24_nonzero",
        "wg_global_above_s16",
        "hypothetical_s16_move_total",
        "banked_s24_move_total",
    )
    for field_name in int_fields:
        claimed = int(getattr(receipt, field_name))
        recomputed = int(aggregates[field_name])
        if claimed != recomputed:
            raise ValueError(f"{field_name} mismatch with per_key_metrics recomputation")
    if receipt.keys_fp_gt0_s24_eq0_fraction != float(aggregates["keys_fp_gt0_s24_eq0_fraction"]):
        raise ValueError(
            "keys_fp_gt0_s24_eq0_fraction mismatch with per_key_metrics recomputation"
        )
    if receipt.global_wg_abs_max != float(aggregates["global_wg_abs_max"]):
        raise ValueError("global_wg_abs_max mismatch with per_key_metrics recomputation")
    if receipt.exact_zero_fraction_all_elements != float(
        aggregates["exact_zero_fraction_all_elements"]
    ):
        raise ValueError(
            "exact_zero_fraction_all_elements mismatch with per_key_metrics recomputation"
        )


def anchor_gates_pass(receipt: T2FpVsS24DisambiguationReceipt | Mapping[str, Any]) -> bool:
    if isinstance(receipt, T2FpVsS24DisambiguationReceipt):
        banked_total = int(receipt.banked_s24_move_total)
        self_consistency_pass = bool(receipt.self_consistency_pass)
    else:
        banked_total = int(receipt.get("banked_s24_move_total", -1))
        self_consistency_pass = bool(receipt.get("self_consistency_pass"))
    return bool(
        _frozen_anchor_fields_match(receipt)
        and self_consistency_pass
        and banked_total == 0
    )


def classify_t2_disambiguation(
    receipt: T2FpVsS24DisambiguationReceipt,
) -> tuple[str, str]:
    if not anchor_gates_pass(receipt):
        return BRANCH_INCONSISTENT, RECOMMEND_INVESTIGATE_CAPTURE

    if (
        receipt.keys_fp_gt0_s24_eq0 >= 1
        and receipt.wg_global_below_s24_nonzero > receipt.wg_global_above_s16
        and receipt.hypothetical_s16_move_total > 0
    ):
        return BRANCH_SHIFT_TOO_COARSE, RECOMMEND_REOPEN_SHIFT_FINER

    if receipt.keys_with_fp_moves_gt0 == 0 and (
        receipt.global_wg_abs_max < LSB_S16
        or receipt.exact_zero_fraction_all_elements > 0.95
    ):
        return BRANCH_INPUT_TOO_EASY, RECOMMEND_FIX_T2_INPUT

    return BRANCH_UNRESOLVED, RECOMMEND_UNRESOLVED_ENRICH


def validate_t2_disambiguation_receipt(receipt: T2FpVsS24DisambiguationReceipt) -> None:
    if receipt.schema_version != T2_FP_VS_S24_DISAMBIGUATION_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch on t2 fp-vs-s24 disambiguation receipt")
    if receipt.target_name != T2_FP_VS_S24_DISAMBIGUATION_TARGET_NAME:
        raise ValueError("target_name mismatch on t2 fp-vs-s24 disambiguation receipt")
    if receipt.pass_receipt:
        raise ValueError("pass_receipt must remain false on t2 fp-vs-s24 disambiguation receipt")
    for field_name in T2_FP_VS_S24_DISAMBIGUATION_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field_name)):
            raise ValueError(
                f"{field_name} must remain false on t2 fp-vs-s24 disambiguation receipt"
            )
    if receipt.non_claims != T2_FP_VS_S24_DISAMBIGUATION_NON_CLAIMS:
        raise ValueError("non_claims must be exact on t2 fp-vs-s24 disambiguation receipt")
    if not _frozen_anchor_fields_match(receipt):
        raise ValueError("frozen anchor field mismatch on t2 fp-vs-s24 disambiguation receipt")
    if receipt.anchor_precondition_pass != _frozen_anchor_fields_match(receipt):
        raise ValueError(
            "anchor_precondition_pass does not match re-derived frozen anchor check"
        )
    if len(receipt.per_key_metrics) != FROZEN_T2_ANCHOR_KEY_COUNT:
        raise ValueError("per_key_metrics length must equal frozen anchor key count")
    if receipt.anchor_key_count != len(receipt.per_key_metrics):
        raise ValueError("anchor_key_count must match per_key_metrics length")
    aggregates = aggregate_disambiguation_metrics(receipt.per_key_metrics)
    _validate_recomputed_aggregates(receipt, aggregates)
    if receipt.self_consistency_pass != (int(aggregates["banked_s24_move_total"]) == 0):
        raise ValueError("self_consistency_pass must match banked_s24_move_total==0")
    branch_id, recommended = classify_t2_disambiguation(receipt)
    if receipt.branch_id != branch_id:
        raise ValueError("branch_id mismatch with classify_t2_disambiguation")
    if receipt.recommended_next_slice != recommended:
        raise ValueError("recommended_next_slice mismatch with classify_t2_disambiguation")
    if receipt.branch_id in {BRANCH_SHIFT_TOO_COARSE, BRANCH_INPUT_TOO_EASY}:
        if not anchor_gates_pass(receipt):
            raise ValueError("CLASS_A/B unreachable without anchor gates pass")


def run_t2_fp_vs_s24_disambiguation(
    *,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> T2FpVsS24DisambiguationReceipt:
    resolved_path = checkpoint_path or DEFAULT_T2_CHECKPOINT_REL
    if not Path(resolved_path).is_file():
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / DEFAULT_T2_CHECKPOINT_REL
        if candidate.is_file():
            resolved_path = str(candidate)
        else:
            raise FileNotFoundError(resolved_path)
    observed_sha = sha256_file(resolved_path)
    bundle = capture_tier2_checkpoint_raw_captures(
        checkpoint_path=resolved_path,
        checkpoint_sha256=observed_sha,
        device=device,
        curriculum_seed=FROZEN_T2_ANCHOR_CURRICULUM_SEED,
        batch_size=FROZEN_T2_ANCHOR_BATCH_SIZE,
    )
    anchor_precondition_pass, anchor_fields = evaluate_anchor_precondition(bundle)
    per_key_metrics = build_disambiguation_metrics_from_bundle(bundle)
    aggregates = aggregate_disambiguation_metrics(per_key_metrics)
    self_consistency_pass = int(aggregates["banked_s24_move_total"]) == 0
    hard_false = {field_name: False for field_name in T2_FP_VS_S24_DISAMBIGUATION_HARD_FALSE_FIELDS}
    draft = T2FpVsS24DisambiguationReceipt(
        schema_version=T2_FP_VS_S24_DISAMBIGUATION_SCHEMA_VERSION,
        target_name=T2_FP_VS_S24_DISAMBIGUATION_TARGET_NAME,
        pass_receipt=False,
        anchor_checkpoint_sha256=str(anchor_fields["anchor_checkpoint_sha256"]),
        anchor_curriculum_seed=int(anchor_fields["anchor_curriculum_seed"]),
        anchor_batch_size=int(anchor_fields["anchor_batch_size"]),
        anchor_key_count=int(anchor_fields["anchor_key_count"]),
        anchor_key_set_sha256=str(anchor_fields["anchor_key_set_sha256"]),
        anchor_capture_seam_id=str(anchor_fields["anchor_capture_seam_id"]),
        anchor_precondition_pass=anchor_precondition_pass,
        banked_s24_move_total=int(aggregates["banked_s24_move_total"]),
        self_consistency_pass=self_consistency_pass,
        per_key_metrics=tuple(per_key_metrics),
        keys_with_fp_moves_gt0=int(aggregates["keys_with_fp_moves_gt0"]),
        keys_with_s24_moves_gt0=int(aggregates["keys_with_s24_moves_gt0"]),
        keys_fp_gt0_s24_eq0=int(aggregates["keys_fp_gt0_s24_eq0"]),
        keys_fp_gt0_s24_eq0_fraction=float(aggregates["keys_fp_gt0_s24_eq0_fraction"]),
        keys_both_zero=int(aggregates["keys_both_zero"]),
        wg_global_below_s24_nonzero=int(aggregates["wg_global_below_s24_nonzero"]),
        wg_global_above_s16=int(aggregates["wg_global_above_s16"]),
        hypothetical_s16_move_total=int(aggregates["hypothetical_s16_move_total"]),
        global_wg_abs_max=float(aggregates["global_wg_abs_max"]),
        exact_zero_fraction_all_elements=float(aggregates["exact_zero_fraction_all_elements"]),
        branch_id=BRANCH_INCONSISTENT,
        recommended_next_slice=RECOMMEND_INVESTIGATE_CAPTURE,
        non_claims=T2_FP_VS_S24_DISAMBIGUATION_NON_CLAIMS,
        **hard_false,
    )
    branch_id, recommended = classify_t2_disambiguation(draft)
    return T2FpVsS24DisambiguationReceipt(
        schema_version=draft.schema_version,
        target_name=draft.target_name,
        pass_receipt=False,
        anchor_checkpoint_sha256=draft.anchor_checkpoint_sha256,
        anchor_curriculum_seed=draft.anchor_curriculum_seed,
        anchor_batch_size=draft.anchor_batch_size,
        anchor_key_count=draft.anchor_key_count,
        anchor_key_set_sha256=draft.anchor_key_set_sha256,
        anchor_capture_seam_id=draft.anchor_capture_seam_id,
        anchor_precondition_pass=draft.anchor_precondition_pass,
        banked_s24_move_total=draft.banked_s24_move_total,
        self_consistency_pass=draft.self_consistency_pass,
        per_key_metrics=draft.per_key_metrics,
        keys_with_fp_moves_gt0=draft.keys_with_fp_moves_gt0,
        keys_with_s24_moves_gt0=draft.keys_with_s24_moves_gt0,
        keys_fp_gt0_s24_eq0=draft.keys_fp_gt0_s24_eq0,
        keys_fp_gt0_s24_eq0_fraction=draft.keys_fp_gt0_s24_eq0_fraction,
        keys_both_zero=draft.keys_both_zero,
        wg_global_below_s24_nonzero=draft.wg_global_below_s24_nonzero,
        wg_global_above_s16=draft.wg_global_above_s16,
        hypothetical_s16_move_total=draft.hypothetical_s16_move_total,
        global_wg_abs_max=draft.global_wg_abs_max,
        exact_zero_fraction_all_elements=draft.exact_zero_fraction_all_elements,
        branch_id=branch_id,
        recommended_next_slice=recommended,
        non_claims=draft.non_claims,
        **hard_false,
    )
