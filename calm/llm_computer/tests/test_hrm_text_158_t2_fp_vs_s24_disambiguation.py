"""Tests for T2 FP-vs-S24 disambiguation diagnostic."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.realistic_gradient_parity_probe import (
    DEFAULT_T2_CHECKPOINT_REL,
    RawKeyCapture,
    Tier2RawCaptureBundle,
    discover_t2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.t2_fp_vs_s24_disambiguation import (
    BRANCH_INCONSISTENT,
    BRANCH_INPUT_TOO_EASY,
    BRANCH_SHIFT_TOO_COARSE,
    BRANCH_UNRESOLVED,
    FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
    FROZEN_T2_ANCHOR_KEY_COUNT,
    FROZEN_T2_ANCHOR_KEY_SET_SHA256,
    LSB_S16,
    RECOMMEND_FIX_T2_INPUT,
    RECOMMEND_INVESTIGATE_CAPTURE,
    RECOMMEND_REOPEN_SHIFT_FINER,
    RECOMMEND_UNRESOLVED_ENRICH,
    T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
    T2_FP_VS_S24_DISAMBIGUATION_HARD_FALSE_FIELDS,
    T2_FP_VS_S24_DISAMBIGUATION_NON_CLAIMS,
    T2_FP_VS_S24_DISAMBIGUATION_SCHEMA_VERSION,
    T2_FP_VS_S24_DISAMBIGUATION_TARGET_NAME,
    T2FpVsS24DisambiguationReceipt,
    T2KeyDisambiguationMetrics,
    anchor_key_set_sha256,
    classify_t2_disambiguation,
    evaluate_anchor_precondition,
    run_t2_fp_vs_s24_disambiguation,
    validate_t2_disambiguation_receipt,
)


def _metric(
    *,
    state_key: str = "proj",
    fp_move_count: int = 0,
    s24_move_count: int = 0,
    wg_abs_max: float = 0.0,
    wg_below_lsb_s24: int = 0,
    wg_above_lsb_s16: int = 0,
    wg_exact_zero: int = 100,
    total_weight_elements: int = 100,
    hypothetical_s16_move_count: int = 0,
) -> T2KeyDisambiguationMetrics:
    return T2KeyDisambiguationMetrics(
        state_key=state_key,
        fp_move_count=fp_move_count,
        s24_move_count=s24_move_count,
        wg_abs_min=0.0,
        wg_abs_median=wg_abs_max / 2.0,
        wg_abs_max=wg_abs_max,
        wg_below_lsb_s24=wg_below_lsb_s24,
        wg_below_lsb_s16=0,
        wg_exact_zero=wg_exact_zero,
        wg_above_lsb_s16=wg_above_lsb_s16,
        total_weight_elements=total_weight_elements,
        hypothetical_s16_move_count=hypothetical_s16_move_count,
    )


def _pad_metrics_to_frozen_count(
    metrics: tuple[T2KeyDisambiguationMetrics, ...],
) -> tuple[T2KeyDisambiguationMetrics, ...]:
    padded = list(metrics)
    while len(padded) < FROZEN_T2_ANCHOR_KEY_COUNT:
        padded.append(_metric(state_key=f"pad_{len(padded)}"))
    return tuple(padded[:FROZEN_T2_ANCHOR_KEY_COUNT])


def _receipt_from_metrics(
    metrics: tuple[T2KeyDisambiguationMetrics, ...],
    *,
    anchor_precondition_pass: bool = True,
    banked_s24_move_total: int = 0,
    self_consistency_pass: bool = True,
    anchor_key_set_sha256_value: str = FROZEN_T2_ANCHOR_KEY_SET_SHA256,
    anchor_capture_seam_id: str = T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
    pad_to_frozen_key_count: bool = True,
) -> T2FpVsS24DisambiguationReceipt:
    if pad_to_frozen_key_count:
        metrics = _pad_metrics_to_frozen_count(metrics)
    keys_with_fp_moves_gt0 = sum(1 for item in metrics if item.fp_move_count > 0)
    keys_with_s24_moves_gt0 = sum(1 for item in metrics if item.s24_move_count > 0)
    keys_fp_gt0_s24_eq0 = sum(
        1 for item in metrics if item.fp_move_count > 0 and item.s24_move_count == 0
    )
    keys_both_zero = sum(
        1 for item in metrics if item.fp_move_count == 0 and item.s24_move_count == 0
    )
    total_elements = sum(item.total_weight_elements for item in metrics)
    exact_zero_total = sum(item.wg_exact_zero for item in metrics)
    hard_false = {field_name: False for field_name in T2_FP_VS_S24_DISAMBIGUATION_HARD_FALSE_FIELDS}
    draft = T2FpVsS24DisambiguationReceipt(
        schema_version=T2_FP_VS_S24_DISAMBIGUATION_SCHEMA_VERSION,
        target_name=T2_FP_VS_S24_DISAMBIGUATION_TARGET_NAME,
        pass_receipt=False,
        anchor_checkpoint_sha256=FROZEN_T2_ANCHOR_CHECKPOINT_SHA256,
        anchor_curriculum_seed=158,
        anchor_batch_size=4,
        anchor_key_count=FROZEN_T2_ANCHOR_KEY_COUNT,
        anchor_key_set_sha256=anchor_key_set_sha256_value,
        anchor_capture_seam_id=anchor_capture_seam_id,
        anchor_precondition_pass=anchor_precondition_pass,
        banked_s24_move_total=banked_s24_move_total,
        self_consistency_pass=self_consistency_pass,
        per_key_metrics=metrics,
        keys_with_fp_moves_gt0=keys_with_fp_moves_gt0,
        keys_with_s24_moves_gt0=keys_with_s24_moves_gt0,
        keys_fp_gt0_s24_eq0=keys_fp_gt0_s24_eq0,
        keys_fp_gt0_s24_eq0_fraction=float(keys_fp_gt0_s24_eq0 / max(len(metrics), 1)),
        keys_both_zero=keys_both_zero,
        wg_global_below_s24_nonzero=sum(item.wg_below_lsb_s24 for item in metrics),
        wg_global_above_s16=sum(item.wg_above_lsb_s16 for item in metrics),
        hypothetical_s16_move_total=sum(item.hypothetical_s16_move_count for item in metrics),
        global_wg_abs_max=max((item.wg_abs_max for item in metrics), default=0.0),
        exact_zero_fraction_all_elements=float(exact_zero_total / max(total_elements, 1)),
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


def test_anchor_key_set_sha256_matches_frozen_list():
    keys = [
        "model.H_level.core.layers.0.attn.gqkv_proj",
        "model.H_level.core.layers.0.attn.o_proj",
        "model.L_level.core.layers.3.mlp.gate_up_proj",
    ]
    assert anchor_key_set_sha256(keys) == anchor_key_set_sha256(list(reversed(keys)))


def test_classify_class_a_when_anchor_gates_pass():
    metrics = (
        _metric(
            state_key="k1",
            fp_move_count=3,
            s24_move_count=0,
            wg_below_lsb_s24=50,
            wg_above_lsb_s16=1,
            hypothetical_s16_move_count=2,
        ),
        _metric(state_key="k2"),
    )
    receipt = _receipt_from_metrics(metrics)
    assert receipt.branch_id == BRANCH_SHIFT_TOO_COARSE
    assert receipt.recommended_next_slice == RECOMMEND_REOPEN_SHIFT_FINER
    validate_t2_disambiguation_receipt(receipt)


def test_classify_class_b_when_anchor_gates_pass():
    metrics = (_metric(state_key="k1", wg_abs_max=LSB_S16 / 10.0, wg_exact_zero=99),)
    receipt = _receipt_from_metrics(metrics)
    assert receipt.branch_id == BRANCH_INPUT_TOO_EASY
    assert receipt.recommended_next_slice == RECOMMEND_FIX_T2_INPUT
    validate_t2_disambiguation_receipt(receipt)


def test_class_b_parens_negative_fp_moves_block_class_b():
    metrics = (
        _metric(
            state_key="k1",
            fp_move_count=2,
            wg_abs_max=LSB_S16 / 100.0,
            wg_exact_zero=99,
            total_weight_elements=100,
        ),
    )
    receipt = _receipt_from_metrics(metrics)
    assert receipt.branch_id != BRANCH_INPUT_TOO_EASY


def test_self_consistency_negative_emits_class_c():
    metrics = (_metric(state_key="k1", s24_move_count=1),)
    receipt = _receipt_from_metrics(
        metrics,
        banked_s24_move_total=1,
        self_consistency_pass=False,
    )
    assert receipt.branch_id == BRANCH_INCONSISTENT
    assert receipt.recommended_next_slice == RECOMMEND_INVESTIGATE_CAPTURE


def test_key_set_hash_negative_emits_class_c():
    metrics = (_metric(state_key="k1"),)
    receipt = _receipt_from_metrics(
        metrics,
        anchor_key_set_sha256_value="deadbeef",
    )
    assert receipt.branch_id == BRANCH_INCONSISTENT


def test_checkpoint_sha_negative_via_anchor_precondition():
    bundle = Tier2RawCaptureBundle(
        per_key_captures={"k1": _raw_capture_fixture()},
        per_key_states={},
        provenance={
            "checkpoint_sha256": "wrong",
            "curriculum_seed": 158,
            "batch_size": 4,
            "capture_seam_id": T2_DISAMBIGUATION_CAPTURE_SEAM_ID,
        },
    )
    passed, fields = evaluate_anchor_precondition(bundle)
    assert not passed
    assert fields["anchor_precondition_pass"] is False


def test_seam_id_negative_blocks_class_a_b():
    metrics = (
        _metric(
            state_key="k1",
            fp_move_count=1,
            wg_below_lsb_s24=10,
            hypothetical_s16_move_count=1,
        ),
    )
    receipt = _receipt_from_metrics(metrics, anchor_capture_seam_id="reimplemented_capture")
    assert receipt.branch_id == BRANCH_INCONSISTENT


def test_unresolved_when_mixed_pattern():
    metrics = (
        _metric(state_key="k1", fp_move_count=1, wg_below_lsb_s24=1),
        _metric(state_key="k2", wg_abs_max=LSB_S16 / 10.0),
    )
    receipt = _receipt_from_metrics(metrics)
    assert receipt.branch_id == BRANCH_UNRESOLVED
    assert receipt.recommended_next_slice == RECOMMEND_UNRESOLVED_ENRICH


def test_validator_rejects_forged_anchor_precondition_pass_wrong_checkpoint_sha():
    receipt = _receipt_from_metrics(
        (
            _metric(
                state_key="k1",
                fp_move_count=1,
                wg_below_lsb_s24=10,
                hypothetical_s16_move_count=1,
            ),
        )
    )
    forged = replace(
        receipt,
        anchor_checkpoint_sha256="deadbeef",
        anchor_precondition_pass=True,
    )
    with pytest.raises(ValueError, match="frozen anchor field mismatch"):
        validate_t2_disambiguation_receipt(forged)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("anchor_curriculum_seed", 999),
        ("anchor_batch_size", 8),
        ("anchor_key_count", 31),
    ],
)
def test_validator_rejects_forged_anchor_fields(field_name, field_value):
    receipt = _receipt_from_metrics((_metric(state_key="k1"),))
    forged = replace(receipt, anchor_precondition_pass=True, **{field_name: field_value})
    with pytest.raises(ValueError, match="frozen anchor field mismatch"):
        validate_t2_disambiguation_receipt(forged)


def test_validator_rejects_forged_aggregate_inconsistent_with_per_key_metrics():
    receipt = _receipt_from_metrics(
        (
            _metric(
                state_key="k1",
                fp_move_count=3,
                s24_move_count=0,
                wg_below_lsb_s24=50,
                wg_above_lsb_s16=1,
                hypothetical_s16_move_count=2,
            ),
        )
    )
    forged = replace(receipt, keys_fp_gt0_s24_eq0=999)
    with pytest.raises(ValueError, match="keys_fp_gt0_s24_eq0 mismatch"):
        validate_t2_disambiguation_receipt(forged)


def test_validator_rejects_forged_global_wg_abs_max():
    receipt = _receipt_from_metrics((_metric(state_key="k1", wg_abs_max=0.001),))
    forged = replace(receipt, global_wg_abs_max=9.999)
    with pytest.raises(ValueError, match="global_wg_abs_max mismatch"):
        validate_t2_disambiguation_receipt(forged)


def _raw_capture_fixture() -> RawKeyCapture:
    inputs = (torch.randn(2, 4, 8, requires_grad=True),)
    grad_outputs = (torch.randn(2, 4, 16),)
    q_levels = torch.randint(-1, 2, (16, 8), dtype=torch.int8)
    return RawKeyCapture(
        inputs=inputs,
        grad_outputs=grad_outputs,
        q_levels_flat=q_levels.reshape(-1),
        weight_shape=(16, 8),
    )


@pytest.mark.skipif(
    not Path(DEFAULT_T2_CHECKPOINT_REL).is_file()
    and not (
        Path(__file__).resolve().parents[3] / DEFAULT_T2_CHECKPOINT_REL
    ).is_file(),
    reason="default T2 checkpoint absent on disk",
)
def test_live_read_only_disambiguation_when_checkpoint_present():
    discovery = discover_t2_checkpoint()
    assert discovery.checkpoint_present
    receipt = run_t2_fp_vs_s24_disambiguation(checkpoint_path=str(discovery.checkpoint_path))
    validate_t2_disambiguation_receipt(receipt)
    assert receipt.anchor_precondition_pass
    assert receipt.self_consistency_pass
    assert receipt.anchor_key_count == 32
    assert receipt.anchor_key_set_sha256 == FROZEN_T2_ANCHOR_KEY_SET_SHA256
    assert len(receipt.per_key_metrics) == 32
    assert receipt.banked_s24_move_total == 0
