"""C2.1 real-model bounded-delta acquisition probe harness.

Default-off harness for graduating the C2.0 bounded-delta learner from the toy
BitLinear fixture to the real HRM-Text model wiring. This script deliberately
separates implementation validation from GPU launch validation: CPU-safe
step-0 checks are allowed under the C2.1 implementation gate, while CUDA
forward-level fidelity and acquisition dynamics require separate +1 LAUNCH
gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer
from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.curriculum import (
    BROAD_NORMALIZER_VERSION,
    BroadTokenizer,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (
    build_l0c2k1_identity_full_support,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
    BOUNDED_DELTA_LEARNER_SCHEMA_VERSION,
    BOUNDED_UPDATE_ATTRIBUTION,
    S1_PROJECTION_LAW,
    S1_RANK_BUCKET_VOTE_LAW,
    apply_bounded_delta_vote_step,
    authoritative_forward_context,
    build_authoritative_checkpoint_payload,
    build_optimizer_excluding_eligible_masters,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    derive_bounded_tensor_state_from_weight,
    file_sha256,
    project_s1_gradient_to_moves,
    prove_eligible_master_identity_after_optimizer_step,
    rank_bucketed_int16_votes,
    tensor_sha256,
    validate_authoritative_resume_payload,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from scripts.train_hrm_text_158 import HrmTextGsm8kDataset


RUN_C2_ACQUISITION_PROBE_ENV = "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"
RUN_C2_GPU_LAUNCH_ENV = "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"
C2P1_HARNESS_SCHEMA_VERSION = "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0"
FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL = 1e-3
FORWARD_LEVEL_INIT_FIDELITY_TOLERANCE_REASON = (
    "Native BitLinear training forward materializes q*scale through the "
    "float32 STE expression weight + detach(q*scale - weight), while the "
    "bounded authoritative path uses direct q*scale; mathematically identical "
    "weights can differ by float32 operation order and recurrent-stack "
    "propagation in downstream logits/loss."
)
DEFAULT_PARENT_SHA256 = (
    "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
)
DEFAULT_PARENT = (
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_"
    "pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
IDENTITY_FULL_RUNG = "L0c2-K1-identity-2digit-full"
HISTORICAL_IDENTITY_CONTROL = {
    "control_role": "historical_positive_acquirability_control_not_same_harness_paired_int16",
    "receipt_msg_id": "1779747988676-247047ce",
    "surface": "L0C2K1IDENTITYFULL",
    "strict_exact": "90/90",
    "step": 1500,
    "sha256": "8f23d6b41102873babe712e66bd2a4f6da976b39fc5c06c4fd7fbd697e86ffec",
    "parent_sha256": DEFAULT_PARENT_SHA256,
}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()[:16]


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "inputs": torch.stack([b["inputs"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "sep_positions": torch.stack([b["sep_position"] for b in batch], dim=0),
        "is_prior": torch.stack([b["is_prior"] for b in batch], dim=0),
    }


def _tensor_scalar(value: Any) -> int | float | bool | str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected scalar tensor")
        item = value.detach().cpu().item()
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        if isinstance(item, float):
            return float(item)
        return str(item)
    return value


def _metrics_to_dict(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "logits":
            continue
        if isinstance(value, tuple):
            out[key] = [_tensor_scalar(item) for item in value]
        else:
            out[key] = _tensor_scalar(value)
    return out


def assert_default_off(enabled: bool | None) -> None:
    if enabled is True:
        return
    if os.environ.get(RUN_C2_ACQUISITION_PROBE_ENV) == "1":
        return
    raise RuntimeError(
        "C2.1 acquisition probe is default-off; pass "
        "--enable-bounded-delta-probe or set "
        f"{RUN_C2_ACQUISITION_PROBE_ENV}=1"
    )


def guard_gpu_launch(device: torch.device, *, allow_gpu_launch: bool) -> None:
    if device.type != "cuda":
        return
    if allow_gpu_launch and os.environ.get(RUN_C2_GPU_LAUNCH_ENV) == "1":
        return
    raise RuntimeError(
        "CUDA execution is outside the C2.1 implementation gate. Pass "
        "--allow-gpu-launch AND set "
        f"{RUN_C2_GPU_LAUNCH_ENV}=1 only after a persisted +1 LAUNCH."
    )


def load_parent_checkpoint(path: Path, *, expected_sha256: str | None) -> tuple[dict, str]:
    actual_sha = file_sha256(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(
            f"parent sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in ckpt or "model_state" not in ckpt:
        raise ValueError("parent checkpoint must contain config and model_state")
    return ckpt, actual_sha


def tokenizer_from_checkpoint_config(config: Mapping[str, Any]) -> Any:
    normalizer = config["gsm8k_normalizer_version"]
    if normalizer == BROAD_NORMALIZER_VERSION:
        tok = BroadTokenizer()
        if list(config["gsm8k_char_vocab"]) != tok.vocab_as_list():
            raise ValueError("BroadTokenizer checkpoint vocab mismatch")
        return tok
    return Gsm8kTokenizer.from_metadata(
        vocab_list=config["gsm8k_char_vocab"],
        normalizer_version=normalizer,
    )


def model_config_from_checkpoint_config(config: Mapping[str, Any]) -> HierarchicalReasoningModelConfig:
    return HierarchicalReasoningModelConfig(
        max_seq_len=int(config["max_seq_len"]),
        n_layers=int(config["n_layers"]),
        hidden_size=int(config["hidden_size"]),
        num_heads=int(config["num_heads"]),
        expansion=float(config["expansion"]),
        H_cycles=int(config["H_cycles"]),
        L_cycles=int(config["L_cycles"]),
        half_layers=bool(config["half_layers"]),
        bp_warmup_ratio=float(config["bp_warmup_ratio"]),
        bp_min_steps=int(config["bp_min_steps"]),
        bp_max_steps=int(config["bp_max_steps"]),
        norm_type=config.get("norm_type", "pre"),
        norm_eps=float(config.get("norm_eps", 1e-6)),
        rope_theta=config.get("rope_theta", 10000.0),
        attn_type=config.get("attn_type", "prefixlm"),
        init_type=config.get("init_type", "lecun_normal"),
        pos_emb_type=config.get("pos_emb_type", "rope"),
        use_ternary_bulk=bool(config.get("use_ternary_bulk", False)),
    )


def build_model_from_checkpoint(ckpt: Mapping[str, Any], device: torch.device) -> tuple[LMHead, Any, HierarchicalReasoningModelConfig]:
    config = ckpt["config"]
    tok = tokenizer_from_checkpoint_config(config)
    cfg = model_config_from_checkpoint_config(config)
    model = LMHead(
        HierarchicalReasoningModel(cfg),
        LMHeadConfig(vocab_size=int(config["vocab_size"])),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, tok, cfg


def build_identity_full_batch(
    *,
    tok: Any,
    max_len: int,
    batch_size: int,
    curriculum_seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = make_rung_examples(
        IDENTITY_FULL_RUNG,
        n=90,
        seed=int(curriculum_seed),
        split="train",
    )
    dataset = HrmTextGsm8kDataset(rows, tok, max_len=max_len, curriculum_rung=IDENTITY_FULL_RUNG)
    if len(dataset) < batch_size:
        raise RuntimeError(
            f"identity-full dataset has {len(dataset)} usable rows, need batch_size={batch_size}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate,
    )
    batch = next(iter(loader))
    inputs = batch["inputs"].to(device)
    labels = batch["labels"].to(device)
    sep_positions = batch["sep_positions"].to(device)
    B, L = inputs.shape
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    model_batch = {
        "inputs": inputs,
        "labels": labels,
        "sep_positions": sep_positions,
        "position_ids": position_ids,
    }
    proof = {
        "rung": IDENTITY_FULL_RUNG,
        "seed": int(curriculum_seed),
        "requested_rows": 90,
        "usable_rows": len(dataset),
        "dropped_rows": int(dataset.n_dropped),
        "batch_shape": {
            "inputs": list(inputs.shape),
            "labels": list(labels.shape),
            "sep_positions": list(sep_positions.shape),
        },
        "first_questions": [row["question"] for row in rows[: min(3, len(rows))]],
    }
    return model_batch, proof


def identity_full_support_control_proof(curriculum_seed: int) -> dict[str, Any]:
    support17 = build_l0c2k1_identity_full_support(17)[IDENTITY_FULL_RUNG]
    support_for_seed = build_l0c2k1_identity_full_support(int(curriculum_seed))[IDENTITY_FULL_RUNG]
    train_rows = make_rung_examples(
        IDENTITY_FULL_RUNG,
        n=90,
        seed=int(curriculum_seed),
        split="train",
    )
    support_qe = {(q, e) for q, e, _bucket in support_for_seed}
    train_qe = {(row["question"], row["expected"]) for row in train_rows}
    support_match = len(support_qe) == 90 and support_qe == train_qe
    seed_independent = support17 == support_for_seed
    return {
        "schema": "hrm_text_158_c2p1_identity_full_control_support_proof/v0",
        "historical_control": dict(HISTORICAL_IDENTITY_CONTROL),
        "support_rows": len(support_for_seed),
        "support_hash16": _sha16(support_for_seed),
        "seed": int(curriculum_seed),
        "seed17_support_hash16": _sha16(support17),
        "seed_independent_support": bool(seed_independent),
        "train_rows_qe_match_support": bool(support_match),
        "same_harness_paired_int16_control": False,
        "inline_control_required": not (support_match and seed_independent),
        "null_escalation_rule": (
            "If C2.2 null is ambiguous after C2.1 telemetry, add an inline "
            "same-harness int16/dense-acc control."
        ),
    }


def select_eligible_bitlinears(model: torch.nn.Module, *, eligible_scope: str) -> dict[str, BitLinear]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }
    if not modules:
        raise RuntimeError("model has no BitLinear modules; parent is not ternary-bulk")
    if eligible_scope == "first-bitlinear":
        first_key = sorted(modules)[0]
        return {first_key: modules[first_key]}
    if eligible_scope == "all-bitlinear":
        return dict(sorted(modules.items()))
    raise ValueError(f"unsupported eligible_scope {eligible_scope!r}")


def native_ternary_effective_weight(module: BitLinear) -> torch.Tensor:
    scale = module.weight.detach().abs().mean().clamp(min=module._SCALE_EPS)
    q = (module.weight.detach().to(torch.float32) / scale).round().clamp(-1.0, 1.0)
    return (q * scale.to(torch.float32)).detach().cpu().contiguous()


def derive_tensor_states_and_check_init_fidelity(
    eligible_modules: Mapping[str, BitLinear],
    *,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor_states = {}
    module_reports = {}
    all_pass = True
    for state_key, module in sorted(eligible_modules.items()):
        state = derive_bounded_tensor_state_from_weight(
            state_key,
            module.weight.detach(),
            scale_eps=module._SCALE_EPS,
        )
        bounded_effective = state.materialized_weight(device="cpu", requires_grad=False)
        native_effective = native_ternary_effective_weight(module)
        diff = (bounded_effective - native_effective).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        module_pass = max_abs_diff <= float(threshold)
        all_pass = all_pass and module_pass
        tensor_states[state_key] = state
        module_reports[state_key] = {
            "shape": list(module.weight.shape),
            "q_sha256": tensor_sha256(state.q_levels),
            "frozen_scale": float(state.frozen_scale.item()),
            "native_effective_sha256": tensor_sha256(native_effective),
            "bounded_effective_sha256": tensor_sha256(bounded_effective),
            "max_abs_diff": max_abs_diff,
            "threshold": float(threshold),
            "pass": bool(module_pass),
        }
    report = {
        "schema": "hrm_text_158_c2p1_weight_level_init_fidelity/v0",
        "threshold": float(threshold),
        "module_count": len(module_reports),
        "all_pass": bool(all_pass),
        "modules": module_reports,
    }
    return tensor_states, report


def _capture_eligible_module_outputs(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    eligible_modules: Mapping[str, BitLinear],
    extras: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, list[torch.Tensor]]]:
    captures = {state_key: [] for state_key in eligible_modules}
    handles = []
    for state_key, module in eligible_modules.items():

        def _capture_output(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            key: str = state_key,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "eligible BitLinear forward output telemetry requires "
                    f"torch.Tensor output for {key}, got {type(output).__name__}"
                )
            captures[key].append(output.detach().to(torch.float32).cpu())

        handles.append(module.register_forward_hook(_capture_output))
    try:
        _carry, loss, metrics = model(
            None,
            dict(batch),
            return_logits=True,
            **extras,
        )
    finally:
        for handle in handles:
            handle.remove()
    return loss, metrics, captures


def compare_module_output_fidelity(
    native_outputs: Mapping[str, list[torch.Tensor]],
    bounded_outputs: Mapping[str, list[torch.Tensor]],
    *,
    threshold: float,
    eligible_scope: str,
) -> dict[str, Any]:
    module_reports = {}
    all_pass = True
    for state_key in sorted(native_outputs):
        native_items = native_outputs[state_key]
        bounded_items = bounded_outputs.get(state_key, [])
        counts_match = len(native_items) == len(bounded_items)
        invoked = len(native_items) > 0
        aligned_count = min(len(native_items), len(bounded_items))
        max_abs_diff = 0.0
        allclose = counts_match and invoked
        shape_mismatch_count = 0
        first_output_shape = None
        for native_item, bounded_item in zip(native_items, bounded_items):
            if first_output_shape is None:
                first_output_shape = list(native_item.shape)
            if native_item.shape != bounded_item.shape:
                shape_mismatch_count += 1
                allclose = False
                continue
            diff = (bounded_item - native_item).abs()
            item_max = float(diff.max().item()) if diff.numel() else 0.0
            max_abs_diff = max(max_abs_diff, item_max)
            allclose = allclose and bool(
                torch.allclose(
                    bounded_item,
                    native_item,
                    atol=float(threshold),
                    rtol=0.0,
                )
            )
        module_pass = bool(
            counts_match
            and invoked
            and shape_mismatch_count == 0
            and allclose
        )
        all_pass = all_pass and module_pass
        module_reports[state_key] = {
            "native_invocation_count": len(native_items),
            "bounded_invocation_count": len(bounded_items),
            "invocation_count": len(native_items) if counts_match else None,
            "aligned_invocation_count": aligned_count,
            "first_output_shape": first_output_shape,
            "shape_mismatch_count": shape_mismatch_count,
            "max_abs_diff": max_abs_diff,
            "threshold": float(threshold),
            "rtol": 0.0,
            "allclose": bool(allclose),
            "pass": module_pass,
        }
    missing_bounded_keys = sorted(set(bounded_outputs) - set(native_outputs))
    all_pass = all_pass and not missing_bounded_keys
    return {
        "schema": "hrm_text_158_c2p1_module_output_init_fidelity/v0",
        "eligible_scope": eligible_scope,
        "threshold": float(threshold),
        "rtol": 0.0,
        "module_count": len(module_reports),
        "eligible_modules": sorted(native_outputs),
        "missing_bounded_only_modules": missing_bounded_keys,
        "all_pass": bool(all_pass),
        "modules": module_reports,
    }


def compute_forward_level_init_fidelity(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    threshold: float,
    eligible_scope: str,
    total_steps: int,
) -> dict[str, Any]:
    was_training = model.training
    schedule_total_steps = max(1, int(total_steps))
    extras = model.compute_train_extra_args(0, schedule_total_steps)

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        native_loss, native_metrics, native_module_outputs = _capture_eligible_module_outputs(
            model,
            batch,
            eligible_modules,
            extras,
        )
        native_logits = native_metrics["logits"].detach().to(torch.float32).cpu()
        native_loss_cpu = native_loss.detach().to(torch.float32).cpu()

        with authoritative_forward_context(
            eligible_modules,
            tensor_states,
            device=device,
            requires_grad=False,
        ):
            bounded_loss, bounded_metrics, bounded_module_outputs = _capture_eligible_module_outputs(
                model,
                batch,
                eligible_modules,
                extras,
            )
        bounded_logits = bounded_metrics["logits"].detach().to(torch.float32).cpu()
        bounded_loss_cpu = bounded_loss.detach().to(torch.float32).cpu()
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    requested_threshold_f = float(threshold)
    threshold_f = max(requested_threshold_f, FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL)
    logits_diff = (bounded_logits - native_logits).abs()
    logits_max_abs_diff = float(logits_diff.max().item()) if logits_diff.numel() else 0.0
    loss_abs_diff = float((bounded_loss_cpu - native_loss_cpu).abs().item())
    max_abs_diff = max(logits_max_abs_diff, loss_abs_diff)
    logits_allclose = bool(torch.allclose(bounded_logits, native_logits, atol=threshold_f, rtol=0.0))
    loss_allclose = bool(torch.allclose(bounded_loss_cpu, native_loss_cpu, atol=threshold_f, rtol=0.0))
    module_output_fidelity = compare_module_output_fidelity(
        native_module_outputs,
        bounded_module_outputs,
        threshold=threshold_f,
        eligible_scope=eligible_scope,
    )
    module_output_max_abs_diff = max(
        (
            float(item["max_abs_diff"])
            for item in module_output_fidelity["modules"].values()
        ),
        default=0.0,
    )
    max_abs_diff = max(max_abs_diff, module_output_max_abs_diff)
    module_outputs_allclose = bool(module_output_fidelity["all_pass"])
    passed = bool(logits_allclose and loss_allclose and module_outputs_allclose)
    report = {
        "schema": "hrm_text_158_c2p1_forward_level_init_fidelity/v0",
        "status": "computed",
        "threshold_requested": requested_threshold_f,
        "threshold": threshold_f,
        "threshold_reason": (
            FORWARD_LEVEL_INIT_FIDELITY_TOLERANCE_REASON
            if threshold_f > requested_threshold_f
            else "caller supplied threshold"
        ),
        "rtol": 0.0,
        "eligible_scope": eligible_scope,
        "eligible_module_count": len(eligible_modules),
        "eligible_modules": sorted(eligible_modules),
        "schedule_step": 0,
        "schedule_total_steps": schedule_total_steps,
        "bp_steps": int(extras["bp_steps"]),
        "logits_shape": list(native_logits.shape),
        "logits_max_abs_diff": logits_max_abs_diff,
        "loss_abs_diff": loss_abs_diff,
        "module_output_max_abs_diff": module_output_max_abs_diff,
        "max_abs_diff": max_abs_diff,
        "logits_allclose": logits_allclose,
        "loss_allclose": loss_allclose,
        "module_outputs_allclose": module_outputs_allclose,
        "module_output_fidelity": module_output_fidelity,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(
            "forward-level init-fidelity allclose failed: "
            f"max_abs_diff={max_abs_diff} threshold={threshold_f}"
        )
    return report


def default_vote_update_spec(max_abs_per_tensor: int) -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=int(max_abs_per_tensor),
    )


def run_bounded_delta_steps(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    steps: int,
    require_q_change: bool,
    max_abs_per_tensor: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if steps <= 0:
        return {}, {}, dict(tensor_states)
    model.train()
    rank_spec = default_dry_run_rank_vote_spec()
    vote_spec = default_vote_update_spec(max_abs_per_tensor)
    vote_specs = {key: vote_spec for key in tensor_states}
    optimizer, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model,
        eligible_modules,
        lr=0.0,
        weight_decay=0.0,
    )
    states = dict(tensor_states)
    step_reports: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        extras = model.compute_train_extra_args(step, max(1, int(steps)))
        with authoritative_forward_context(
            eligible_modules,
            states,
            device=device,
            requires_grad=True,
        ) as handle:
            _carry, loss, metrics = model(None, dict(batch), **extras)
            loss.backward()
            weighted_grads = {
                key: handle.weighted_grad(key)
                for key in states
            }
        votes_by_key = {}
        finite_weighted_grad = True
        for key, weighted_grad in weighted_grads.items():
            finite_weighted_grad = finite_weighted_grad and bool(torch.isfinite(weighted_grad).all().item())
            credit = credit_from_weighted_grad(weighted_grad)
            moves = project_s1_gradient_to_moves(weighted_grad, states[key].q_levels)
            votes_by_key[key] = rank_bucketed_int16_votes(credit, moves, rank_spec)
        step_result = apply_bounded_delta_vote_step(states, votes_by_key, vote_specs)
        states = step_result.tensor_states
        q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
        if require_q_change and q_changed_count <= 0:
            raise RuntimeError("bounded-delta step produced no q movement under --require-q-change")
        identity_proof = prove_eligible_master_identity_after_optimizer_step(
            optimizer,
            eligible_modules,
            optimizer_checks=optimizer_checks,
        )
        step_reports[str(step)] = {
            "loss": float(loss.detach().cpu().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "weighted_grad_finite": bool(finite_weighted_grad),
            "metrics": _metrics_to_dict(metrics),
            "bp_steps": int(extras["bp_steps"]),
            "q_changed_count": q_changed_count,
            "step_result": step_result.to_dict(),
            "optimizer_identity_proof": identity_proof,
        }
    updater_config = {
        "rank_vote_spec": rank_spec.to_live_dict(),
        "vote_update_spec": asdict(vote_spec),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }
    return step_reports, updater_config, states


def prove_step0_optimizer_identity(
    model: LMHead,
    eligible_modules: Mapping[str, BitLinear],
) -> dict[str, Any]:
    optimizer, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model,
        eligible_modules,
        lr=0.0,
        weight_decay=0.0,
    )
    return prove_eligible_master_identity_after_optimizer_step(
        optimizer,
        eligible_modules,
        optimizer_checks=optimizer_checks,
    )


def cuda_memory_stats_device_arg(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"CUDA memory stats require a cuda device, got {device}")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def reset_cuda_memory_stats(device: torch.device) -> int:
    stats_device = cuda_memory_stats_device_arg(device)
    torch.cuda.set_device(stats_device)
    torch.cuda.reset_peak_memory_stats(stats_device)
    return stats_device


def cuda_memory_receipt(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "cuda_final_allocated_bytes": None,
        }
    stats_device = cuda_memory_stats_device_arg(device)
    torch.cuda.set_device(stats_device)
    return {
        "device": str(device),
        "cuda_memory_stats_device": int(stats_device),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(stats_device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(stats_device)),
        "cuda_final_allocated_bytes": int(torch.cuda.memory_allocated(stats_device)),
    }


def run_c2p1_probe(
    *,
    parent: Path,
    parent_sha256: str | None = DEFAULT_PARENT_SHA256,
    scratch_root: Path,
    phase: str = "c2p1-real-model-smoke",
    device: str = "cpu",
    eligible_scope: str = "first-bitlinear",
    steps: int = 0,
    batch_size: int = 1,
    max_len: int | None = None,
    curriculum_seed: int = 17,
    init_fidelity_atol: float = 0.0,
    require_q_change: bool = False,
    max_abs_per_tensor: int = 4096,
    enabled: bool | None = None,
    allow_gpu_launch: bool = False,
) -> dict[str, Any]:
    assert_default_off(enabled)
    torch_device = torch.device(device)
    guard_gpu_launch(torch_device, allow_gpu_launch=allow_gpu_launch)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if torch_device.type == "cuda":
        reset_cuda_memory_stats(torch_device)

    ckpt, parent_hash_before = load_parent_checkpoint(parent, expected_sha256=parent_sha256)
    model, tok, cfg = build_model_from_checkpoint(ckpt, torch_device)
    model_batch, batch_proof = build_identity_full_batch(
        tok=tok,
        max_len=int(max_len or ckpt["config"]["max_seq_len"]),
        batch_size=int(batch_size),
        curriculum_seed=int(curriculum_seed),
        device=torch_device,
    )
    support_proof = identity_full_support_control_proof(int(curriculum_seed))
    eligible = select_eligible_bitlinears(model, eligible_scope=eligible_scope)
    tensor_states, init_fidelity = derive_tensor_states_and_check_init_fidelity(
        eligible,
        threshold=float(init_fidelity_atol),
    )
    if not init_fidelity["all_pass"]:
        raise RuntimeError("weight-level init-fidelity allclose failed")

    forward_init_fidelity = compute_forward_level_init_fidelity(
        model,
        model_batch,
        tensor_states,
        eligible,
        device=torch_device,
        threshold=float(init_fidelity_atol),
        eligible_scope=eligible_scope,
        total_steps=int(steps),
    )

    step0_optimizer_identity_proof = None
    if int(steps) <= 0:
        step0_optimizer_identity_proof = prove_step0_optimizer_identity(model, eligible)

    step_reports, updater_config, final_states = run_bounded_delta_steps(
        model,
        model_batch,
        tensor_states,
        eligible,
        device=torch_device,
        steps=int(steps),
        require_q_change=bool(require_q_change),
        max_abs_per_tensor=int(max_abs_per_tensor),
    )
    if not updater_config:
        updater_config = {
            "rank_vote_spec": default_dry_run_rank_vote_spec().to_live_dict(),
            "vote_update_spec": asdict(default_vote_update_spec(max_abs_per_tensor)),
            "projection_law": S1_PROJECTION_LAW,
            "vote_law": S1_RANK_BUCKET_VOTE_LAW,
        }
    checkpoint_payload = build_authoritative_checkpoint_payload(
        final_states,
        step=int(steps),
        updater_config=updater_config,
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
    )
    validate_authoritative_resume_payload(checkpoint_payload)
    parent_hash_after = file_sha256(parent)
    parent_hash_unchanged = parent_hash_before == parent_hash_after
    if not parent_hash_unchanged:
        raise RuntimeError("parent checkpoint hash changed during C2.1 probe")
    receipt = {
        "schema": C2P1_HARNESS_SCHEMA_VERSION,
        "c2p0_schema": BOUNDED_DELTA_LEARNER_SCHEMA_VERSION,
        "bounded_delta_checkpoint_schema": BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
        "phase": phase,
        "implementation_gpu_validation_split": True,
        "gpu_launch_authorized": bool(torch_device.type == "cuda"),
        "gpu_launched": bool(torch_device.type == "cuda"),
        "device": str(torch_device),
        "dry_run": True,
        "checkpoint_written": False,
        "creditdir_mutated": False,
        "banked_pt_mutated": False,
        "parent": str(parent),
        "parent_hash_before": parent_hash_before,
        "parent_hash_after": parent_hash_after,
        "parent_hash_unchanged": parent_hash_unchanged,
        "model_config": {
            "max_seq_len": int(cfg.max_seq_len),
            "n_layers": int(cfg.n_layers),
            "hidden_size": int(cfg.hidden_size),
            "num_heads": int(cfg.num_heads),
            "H_cycles": int(cfg.H_cycles),
            "L_cycles": int(cfg.L_cycles),
            "half_layers": bool(cfg.half_layers),
            "use_ternary_bulk": bool(cfg.use_ternary_bulk),
        },
        "batch": batch_proof,
        "identity_full_control": support_proof,
        "eligible_scope": eligible_scope,
        "eligible_module_count": len(eligible),
        "eligible_modules": sorted(eligible),
        "weight_level_init_fidelity": init_fidelity,
        "forward_level_init_fidelity": forward_init_fidelity,
        "steps_requested": int(steps),
        "forward_backward_update_executed": bool(steps > 0),
        "step0_optimizer_identity_proof": step0_optimizer_identity_proof,
        "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
        "step_reports": step_reports,
        "checkpoint_payload": checkpoint_payload,
        "memory": cuda_memory_receipt(torch_device),
    }
    receipt_path = scratch_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Default-off C2.1 real-model bounded-delta probe harness."
    )
    ap.add_argument("--enable-bounded-delta-probe", action="store_true")
    ap.add_argument("--allow-gpu-launch", action="store_true")
    ap.add_argument("--phase", default="c2p1-real-model-smoke")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--parent", type=Path, default=Path(DEFAULT_PARENT))
    ap.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    ap.add_argument("--scratch-root", type=Path, default=Path("/tmp/hrm158_c2_gpu_probe/c2p1_impl_cpu"))
    ap.add_argument("--curriculum-seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--eligible-scope", choices=["first-bitlinear", "all-bitlinear"], default="first-bitlinear")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--require-q-change", action="store_true")
    ap.add_argument("--max-abs-per-tensor", type=int, default=4096)
    ap.add_argument("--init-fidelity-atol", type=float, default=0.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_c2p1_probe(
        parent=args.parent,
        parent_sha256=args.parent_sha256,
        scratch_root=args.scratch_root,
        phase=args.phase,
        device=args.device,
        eligible_scope=args.eligible_scope,
        steps=args.steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        curriculum_seed=args.curriculum_seed,
        init_fidelity_atol=args.init_fidelity_atol,
        require_q_change=args.require_q_change,
        max_abs_per_tensor=args.max_abs_per_tensor,
        enabled=args.enable_bounded_delta_probe,
        allow_gpu_launch=args.allow_gpu_launch,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
