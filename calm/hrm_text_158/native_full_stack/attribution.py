"""Frozen-q attribution hook contracts for Phase-0."""
from __future__ import annotations

from dataclasses import dataclass


LIVE_C1353FD5_OBSERVATIONS = {
    "source_pointer_label": "live_s1_c1353fd5_trainer",
    "sha256": "c1353fd5837dd7661b0ef7e9fd87b55454c406ef3778f7a4fc004abcdc4e02ea",
    "observed_in": "live c1353fd5",
    "expected_grad_enabled_invocation_strata": 160,
    "expected_schedule_excluded_no_grad": 96,
    "strata_constant_lines": "36-37",
    "strata_assertion_lines": "4228-4253",
    "hook_lines": "913-936",
    "state_lines": "494-551",
    "state_components": ("q:int8", "acc:int16", "frozen_scale:float32"),
}


@dataclass(frozen=True)
class AttributionHookPoint:
    name: str
    source_anchor: str
    hook_timing: str
    captured_tensors: tuple[str, ...]
    invariants: tuple[str, ...]
    decode_eos_tie_in: str
    integrity_gate: str


@dataclass(frozen=True)
class AttributionIntegrityCheck:
    name: str
    required: bool
    evidence: str


ATTRIBUTION_HOOK_POINTS = (
    AttributionHookPoint(
        name="bitlinear_forward_input_capture",
        source_anchor="live_s1_c1353fd5_trainer:913-936",
        hook_timing="forward_hook",
        captured_tensors=("inputs[0].detach().cpu()",),
        invariants=(
            "module_name resolves from state_key",
            "source hash equals live c1353fd5",
            "capture is proof-bound and not learner state",
        ),
        decode_eos_tie_in="capture must be tied to decode/EOS non-regression before banking",
        integrity_gate="observed c1353fd5 hook path plus source hash",
    ),
    AttributionHookPoint(
        name="bitlinear_backward_grad_output_capture",
        source_anchor="live_s1_c1353fd5_trainer:913-936",
        hook_timing="full_backward_hook",
        captured_tensors=("grad_output[0].detach().cpu()",),
        invariants=(
            "grad_outputs present for each paired credit call",
            "160 grad-enabled invocation strata observed in c1353fd5",
            "96 schedule-excluded no-grad invocations observed in c1353fd5",
        ),
        decode_eos_tie_in="credit proof must connect q/vote movement to decode/EOS behavior",
        integrity_gate="actual strata equal expected 160/96 in proof receipt",
    ),
    AttributionHookPoint(
        name="authoritative_state_hash_check",
        source_anchor="live_s1_c1353fd5_trainer:494-551",
        hook_timing="before_and_after_proof",
        captured_tensors=("q:int8", "acc:int16", "frozen_scale:float32"),
        invariants=(
            "q contains only -1, 0, 1",
            "accumulator dtype is int16",
            "frozen scale is finite positive float32",
        ),
        decode_eos_tie_in="state hash anchors the q state used by decode/EOS probes",
        integrity_gate="authoritative_train_state schema/hash receipt",
    ),
)

ATTRIBUTION_INTEGRITY_CHECKS = (
    AttributionIntegrityCheck(
        name="source_hash_matches_c1353fd5",
        required=True,
        evidence="separate read-only sha256sum receipt",
    ),
    AttributionIntegrityCheck(
        name="cached_and_native_train_flags_inactive",
        required=True,
        evidence="BitLinear runtime flags must not provide hidden FP learner path",
    ),
    AttributionIntegrityCheck(
        name="decode_eos_non_regression_floor",
        required=True,
        evidence="finite logits, EOS stop, parsed/exact fields",
    ),
    AttributionIntegrityCheck(
        name="acquisition_tracked_not_banked",
        required=True,
        evidence="first native addition records trend without 90/90 bank claim",
    ),
)
