"""Explicit FP exception registry for the Phase-0 native stack scaffold."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FPException:
    name: str
    path_or_tensor: str
    dtype: str
    lifecycle: str
    budget: str
    proof_gate: str
    rationale: str
    sunset_condition: str


FP_EXCEPTION_REGISTRY = (
    FPException(
        name="frozen_scale_authoritative_state",
        path_or_tensor="train_state._scales[*]",
        dtype="float32",
        lifecycle="permanent_budgeted_metadata",
        budget="one float32 scalar per eligible BitLinear state key",
        proof_gate="finite positive scale hash observed in live c1353fd5",
        rationale="Scale is the declared FP metadata paired with q:int8.",
        sunset_condition="none unless a later native design replaces per-tensor scale.",
    ),
    FPException(
        name="non_eligible_hrm_tensors",
        path_or_tensor="embeddings/norms/lm_head/zL_or_zH_init",
        dtype="float32_or_model_default",
        lifecycle="permanent_out_of_scope",
        budget="outside eligible-bulk native learner-state budget",
        proof_gate="D2.2 scope test and attribution non-regression floor",
        rationale="These tensors are not part of the ternary eligible bulk.",
        sunset_condition="requires a new scope gate to move non-eligible tensors native.",
    ),
    FPException(
        name="transient_authoritative_forward_materialization",
        path_or_tensor="q_levels.float32 * frozen_scale.float32",
        dtype="float32",
        lifecycle="transient_proof_eval_only",
        budget="not persisted; bounded by future eval/export receipt",
        proof_gate="state hash stable before/after proof; no FP optimizer state",
        rationale="Decode and attribution need a faithful q*scale evaluation context.",
        sunset_condition="native eval kernel proves equivalent without FP materialization.",
    ),
    FPException(
        name="credit_capture_tensors",
        path_or_tensor="captured inputs and grad_outputs",
        dtype="float32",
        lifecycle="transient_attribution_only",
        budget="bounded by hook receipt; never authoritative learner state",
        proof_gate="observed c1353fd5 hook strata and capture integrity",
        rationale="Attribution compares gradients to q/vote changes without making FP the learner.",
        sunset_condition="native integer attribution proof replaces FP capture.",
    ),
    FPException(
        name="structural_bitlinear_fp_master",
        path_or_tensor="BitLinear.weight",
        dtype="float32_or_bfloat16",
        lifecycle="sunset_diagnostic_placeholder",
        budget="excluded from eligible-bulk native learner state",
        proof_gate="optimizer exclusion plus authoritative state hash excludes FP masters",
        rationale="Current repo BitLinear stores FP master weights; native path must not learn through them.",
        sunset_condition="eligible-bulk native module removes or freezes structural FP masters.",
    ),
    FPException(
        name="current_ttrain_b_fp_master_path",
        path_or_tensor="calm/hrm_text_158/ternary_train_kernel.py",
        dtype="float32",
        lifecycle="comparison_surface_only",
        budget="not accepted as native full-stack learner state",
        proof_gate="hidden-FP-learner fail-state check",
        rationale="Existing TTrain-B is a useful seam and parity surface, not the end-state.",
        sunset_condition="native q/vote update path passes attribution and decode/EOS gates.",
    ),
)

HIDDEN_FP_LEARNER_FAIL_STATE = (
    "Fail if eligible-bulk learning is carried by hidden FP masters, Adam moments, "
    "cached inference weights, native_train flags, or any unbudgeted FP learner state."
)
