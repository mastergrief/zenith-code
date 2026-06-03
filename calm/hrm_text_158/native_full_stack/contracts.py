"""Subsystem contracts for the native-full-stack Phase-0 scaffold."""
from __future__ import annotations

from dataclasses import dataclass


IMPLEMENTATION_STATUS_SKELETON_ONLY = "skeleton_only"

PROJECTION_GROUPS = (
    "attn.gqkv.gate",
    "attn.gqkv.query",
    "attn.gqkv.key",
    "attn.gqkv.value",
    "attn.o",
    "mlp.gate_up.gate",
    "mlp.gate_up.up",
    "mlp.down",
)


@dataclass(frozen=True)
class SubsystemContract:
    name: str
    source_file: str
    projection_group: str
    state_key_pattern: str
    existing_role: str
    native_target_role: str
    proof_gate: str
    implementation_status: str = IMPLEMENTATION_STATUS_SKELETON_ONLY


SUBSYSTEM_CONTRACTS = (
    SubsystemContract(
        name="attention_gqkv_gate",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="attn.gqkv.gate",
        state_key_pattern="*.attn.gqkv_proj.weight[group=gate]",
        existing_role="BitLinear forward-ternary gate projection slice",
        native_target_role="native ternary gate projection seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="attention_gqkv_query",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="attn.gqkv.query",
        state_key_pattern="*.attn.gqkv_proj.weight[group=query]",
        existing_role="BitLinear forward-ternary query projection slice",
        native_target_role="native ternary query projection seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="attention_gqkv_key",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="attn.gqkv.key",
        state_key_pattern="*.attn.gqkv_proj.weight[group=key]",
        existing_role="BitLinear forward-ternary key projection slice",
        native_target_role="native ternary key projection seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="attention_gqkv_value",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="attn.gqkv.value",
        state_key_pattern="*.attn.gqkv_proj.weight[group=value]",
        existing_role="BitLinear forward-ternary value projection slice",
        native_target_role="native ternary value projection seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="attention_output",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="attn.o",
        state_key_pattern="*.attn.o_proj.weight",
        existing_role="BitLinear forward-ternary attention output projection",
        native_target_role="native ternary attention output seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="mlp_gate_up_gate",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="mlp.gate_up.gate",
        state_key_pattern="*.mlp.gate_up_proj.weight[group=gate]",
        existing_role="BitLinear forward-ternary SwiGLU gate slice",
        native_target_role="native ternary MLP gate seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="mlp_gate_up_up",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="mlp.gate_up.up",
        state_key_pattern="*.mlp.gate_up_proj.weight[group=up]",
        existing_role="BitLinear forward-ternary SwiGLU up slice",
        native_target_role="native ternary MLP up seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="mlp_down",
        source_file="calm/hrm_text_158/layers.py",
        projection_group="mlp.down",
        state_key_pattern="*.mlp.down_proj.weight",
        existing_role="BitLinear forward-ternary MLP down projection",
        native_target_role="native ternary MLP down seam",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
    ),
    SubsystemContract(
        name="authoritative_q_levels",
        source_file="live_s1_c1353fd5_trainer",
        projection_group="eligible_bulk_state",
        state_key_pattern="train_state._q_tensors[*]",
        existing_role="observed live S1 q:int8 ternary learner state",
        native_target_role="persistent native q state",
        proof_gate="state hash plus non-frozen-q proof",
    ),
    SubsystemContract(
        name="integer_vote_accumulators",
        source_file="live_s1_c1353fd5_trainer",
        projection_group="eligible_bulk_state",
        state_key_pattern="train_state._accumulators[*]",
        existing_role="observed live S1 acc:int16 vote state",
        native_target_role="persistent native vote accumulator state",
        proof_gate="state hash plus accumulator range invariant",
    ),
    SubsystemContract(
        name="frozen_scales",
        source_file="live_s1_c1353fd5_trainer",
        projection_group="eligible_bulk_state",
        state_key_pattern="train_state._scales[*]",
        existing_role="observed live S1 frozen_scale:float32",
        native_target_role="budgeted frozen scale metadata",
        proof_gate="finite positive scale hash",
    ),
    SubsystemContract(
        name="credit_capture_hooks",
        source_file="live_s1_c1353fd5_trainer",
        projection_group="attribution",
        state_key_pattern="_attach_credit_hooks(capture_state_keys)",
        existing_role="forward input and backward grad_output capture",
        native_target_role="frozen-q attribution evidence seam",
        proof_gate="160/96 invocation strata observed in c1353fd5",
    ),
    SubsystemContract(
        name="authoritative_forward_context",
        source_file="live_s1_c1353fd5_trainer",
        projection_group="forward_eval",
        state_key_pattern="q_levels.float32 * frozen_scale.float32",
        existing_role="transient q*scale materialization for proof/eval",
        native_target_role="native eval/export facade seam",
        proof_gate="no persisted FP eligible-bulk learner state",
    ),
    SubsystemContract(
        name="decode_eos_probe_surface",
        source_file="scripts/probe_hrm_text_158.py",
        projection_group="validation",
        state_key_pattern="decode/eos/finite/exact probe fields",
        existing_role="decode-EOS correctness and finite-logit probe surface",
        native_target_role="non-regression floor for first native addition",
        proof_gate="decode/EOS smoke non-regression",
    ),
)
