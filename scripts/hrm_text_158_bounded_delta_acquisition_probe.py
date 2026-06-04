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
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID
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
C2P2_TRAJECTORY_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_acquisition_trajectory/v0"
C2P2_AUDIT_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_strict_exact_audit/v0"
C2P2_SUPPORT_CYCLER_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_support_cycler/v0"
C2P2_TIMING_SCHEMA_VERSION = "hrm_text_158_c2p2_calibration_timing_summary/v0"
C2P2_PHASE_TELEMETRY_SCHEMA_VERSION = "hrm_text_158_c2p2_phase_telemetry/v0"
C2P2_DEVICE_GUARD_SCHEMA_VERSION = "hrm_text_158_c2p2_device_guard/v0"
C2P2_STRICT_EXACT_TARGET = 90
C2P2_DEFAULT_MAX_STEPS_HARD = 1500
C2P2_NULL_TAXONOMY = (
    "no-q-move",
    "q-move-no-accuracy",
    "partial-acquisition-plateau",
    "nonfinite",
    "instability-divergence",
    "audit-mismatch",
    "runtime-resource-failure",
)
C2P2_NULL_ESCALATION_RULE = (
    "If C2.2 returns a classified null, escalate inside the same harness with "
    "an inline int16/dense-acc control; historical receipt 1779747988676 is "
    "context only, not a same-harness paired control."
)
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


def identity_full_rows(curriculum_seed: int) -> list[dict[str, Any]]:
    return make_rung_examples(
        IDENTITY_FULL_RUNG,
        n=C2P2_STRICT_EXACT_TARGET,
        seed=int(curriculum_seed),
        split="train",
    )


def _identity_full_usable_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tok: Any,
    max_len: int,
) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ids, _sep_pos = tok.encode_example(row["question"], row["expected"])
        if len(ids) <= int(max_len):
            item = dict(row)
            item["_support_index"] = int(index)
            usable.append(item)
    return usable


def _model_batch_from_collated(
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = batch["inputs"].to(device)
    labels = batch["labels"].to(device)
    sep_positions = batch["sep_positions"].to(device)
    batch_size, seq_len = inputs.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return {
        "inputs": inputs,
        "labels": labels,
        "sep_positions": sep_positions,
        "position_ids": position_ids,
    }


def _support_sample_hash(row: Mapping[str, Any]) -> str:
    return _sha16(
        {
            "question": row["question"],
            "expected": row["expected"],
            "support_index": int(row.get("_support_index", -1)),
        }
    )


def _support_batch_metadata(
    *,
    batch_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sample_hashes = [_support_sample_hash(row) for row in rows]
    row_ids = [
        f"{int(row.get('_support_index', index))}:{sample_hash}"
        for index, (row, sample_hash) in enumerate(zip(rows, sample_hashes))
    ]
    return {
        "batch_index": int(batch_index),
        "row_count": len(rows),
        "row_ids": row_ids,
        "sample_hashes": sample_hashes,
        "batch_content_hash16": _sha16(sample_hashes),
        "first_question": rows[0]["question"] if rows else None,
        "last_question": rows[-1]["question"] if rows else None,
    }


def build_identity_full_support_batches(
    *,
    tok: Any,
    max_len: int,
    batch_size: int,
    curriculum_seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows = identity_full_rows(int(curriculum_seed))
    usable_rows = _identity_full_usable_rows(rows, tok=tok, max_len=int(max_len))
    dataset = HrmTextGsm8kDataset(
        rows,
        tok,
        max_len=int(max_len),
        curriculum_rung=IDENTITY_FULL_RUNG,
    )
    if len(dataset) != len(usable_rows):
        raise RuntimeError(
            "identity-full metadata/tensor row mismatch: "
            f"dataset={len(dataset)} usable_metadata={len(usable_rows)}"
        )
    if not dataset:
        raise RuntimeError("identity-full dataset has no usable rows")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate,
    )
    support_batches: list[dict[str, Any]] = []
    row_offset = 0
    for batch_index, collated in enumerate(loader):
        row_count = int(collated["inputs"].shape[0])
        batch_rows = usable_rows[row_offset: row_offset + row_count]
        metadata = _support_batch_metadata(batch_index=batch_index, rows=batch_rows)
        support_batches.append(
            {
                "batch": _model_batch_from_collated(collated, device=device),
                "metadata": {
                    **metadata,
                    "row_start": int(row_offset),
                    "row_end_exclusive": int(row_offset + row_count),
                },
            }
        )
        row_offset += row_count
    distinct_batch_hashes = {
        item["metadata"]["batch_content_hash16"]
        for item in support_batches
    }
    proof = {
        "schema": C2P2_SUPPORT_CYCLER_SCHEMA_VERSION,
        "rung": IDENTITY_FULL_RUNG,
        "seed": int(curriculum_seed),
        "requested_rows": C2P2_STRICT_EXACT_TARGET,
        "usable_rows": len(dataset),
        "dropped_rows": int(dataset.n_dropped),
        "batch_size": int(batch_size),
        "batch_count": len(support_batches),
        "distinct_batch_count": len(distinct_batch_hashes),
        "has_at_least_two_distinct_batches": len(distinct_batch_hashes) >= 2,
        "covers_full_support": len(dataset) == C2P2_STRICT_EXACT_TARGET and row_offset == len(dataset),
        "support_content_hash16": _sha16(
            [
                batch["metadata"]["batch_content_hash16"]
                for batch in support_batches
            ]
        ),
        "first_questions": [row["question"] for row in rows[: min(3, len(rows))]],
        "batch_metadata": [
            item["metadata"]
            for item in support_batches
        ],
    }
    return support_batches, proof


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


def _synchronize_timing_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_start(device: torch.device) -> float:
    _synchronize_timing_device(device)
    return time.perf_counter()


def _timing_duration_seconds(start: float, device: torch.device) -> float:
    _synchronize_timing_device(device)
    return max(0.0, float(time.perf_counter() - start))


def _timeout_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0.0:
        return None
    return timeout


class C2PhaseTimeout(RuntimeError):
    def __init__(
        self,
        *,
        phase: str,
        bound_kind: str,
        duration_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self.payload = {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "event": "phase_timeout",
            "phase": str(phase),
            "bound_kind": str(bound_kind),
            "duration_seconds": float(duration_seconds),
            "timeout_seconds": float(timeout_seconds),
        }
        super().__init__(json.dumps(self.payload, sort_keys=True))


def enforce_phase_bound(
    *,
    phase: str,
    duration_seconds: float,
    timeout_seconds: float | int | None,
    bound_kind: str,
) -> None:
    timeout = _timeout_or_none(timeout_seconds)
    if timeout is not None and float(duration_seconds) > timeout:
        raise C2PhaseTimeout(
            phase=phase,
            bound_kind=bound_kind,
            duration_seconds=float(duration_seconds),
            timeout_seconds=timeout,
        )


def assert_probe_device_ready(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": C2P2_DEVICE_GUARD_SCHEMA_VERSION,
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type != "cuda":
        report["pass"] = True
        return report
    if not torch.cuda.is_available():
        report["pass"] = False
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    stats_device = cuda_memory_stats_device_arg(device)
    torch.cuda.set_device(stats_device)
    current_device = int(torch.cuda.current_device())
    report.update(
        {
            "cuda_device_index": stats_device,
            "cuda_current_device": current_device,
            "pass": current_device == stats_device,
        }
    )
    if not report["pass"]:
        raise RuntimeError(
            "CUDA current device does not match requested probe device: "
            f"current={current_device} requested={stats_device}"
        )
    return report


class PhaseProgress:
    def __init__(
        self,
        *,
        enabled: bool,
        device: torch.device,
        phase_timeout_seconds: float | int | None = None,
        total_timeout_seconds: float | int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.phase_timeout_seconds = _timeout_or_none(phase_timeout_seconds)
        self.total_timeout_seconds = _timeout_or_none(total_timeout_seconds)
        self.clock = clock
        self.started_at = float(self.clock())
        self.events: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return bool(
            self.enabled
            or self.phase_timeout_seconds is not None
            or self.total_timeout_seconds is not None
        )

    def _elapsed(self) -> float:
        return max(0.0, float(self.clock() - self.started_at))

    def mark(self, phase: str, event: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "phase": str(phase),
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            **fields,
        }
        if self.active:
            self.events.append(payload)
        if self.enabled:
            print(json.dumps(payload, sort_keys=True), flush=True)
        return payload

    def _check_total_bound(self, phase: str) -> None:
        enforce_phase_bound(
            phase=phase,
            duration_seconds=self._elapsed(),
            timeout_seconds=self.total_timeout_seconds,
            bound_kind="total",
        )

    @contextmanager
    def phase(self, phase: str, **fields: Any) -> Any:
        phase_start = float(self.clock())
        self.mark(phase, "start", **fields)
        try:
            yield
        except Exception as exc:
            self.mark(
                phase,
                "error",
                duration_seconds=max(0.0, float(self.clock() - phase_start)),
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        duration = max(0.0, float(self.clock() - phase_start))
        try:
            enforce_phase_bound(
                phase=phase,
                duration_seconds=duration,
                timeout_seconds=self.phase_timeout_seconds,
                bound_kind="phase",
            )
            self._check_total_bound(phase)
        except C2PhaseTimeout as exc:
            timeout_fields = {
                key: value
                for key, value in exc.payload.items()
                if key not in {"schema", "phase", "event"}
            }
            self.mark(phase, "timeout", **timeout_fields, **fields)
            raise
        self.mark(phase, "end", duration_seconds=duration, **fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "phase_timeout_seconds": self.phase_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "event_count": len(self.events),
            "events": list(self.events),
        }


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _numeric_report_items(
    reports: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (str(key), reports[key])
        for key in sorted(reports, key=lambda item: int(item))
    ]


def build_timing_summary(
    *,
    step_reports: Mapping[str, Any],
    audit_reports: Mapping[str, Any],
    total_run_duration_seconds: float,
) -> dict[str, Any]:
    step_duration_by_step = {
        step: float(report["duration_seconds"])
        for step, report in _numeric_report_items(step_reports)
        if "duration_seconds" in report
    }
    audit_duration_by_step = {
        step: float(report["duration_seconds"])
        for step, report in _numeric_report_items(audit_reports)
        if "duration_seconds" in report
    }
    step_durations = list(step_duration_by_step.values())
    audit_durations = list(audit_duration_by_step.values())
    return {
        "schema": C2P2_TIMING_SCHEMA_VERSION,
        "total_run_duration_seconds": float(total_run_duration_seconds),
        "step_report_count": len(step_reports),
        "step_timing_count": len(step_duration_by_step),
        "step_duration_seconds": step_durations,
        "step_duration_seconds_by_step": step_duration_by_step,
        "median_step_duration_seconds": _median_or_none(step_durations),
        "total_step_duration_seconds": float(sum(step_durations)),
        "audit_report_count": len(audit_reports),
        "audit_timing_count": len(audit_duration_by_step),
        "audit_duration_seconds": audit_durations,
        "audit_duration_seconds_by_step": audit_duration_by_step,
        "audit_overhead_seconds_by_step": audit_duration_by_step,
        "median_audit_duration_seconds": _median_or_none(audit_durations),
        "total_audit_duration_seconds": float(sum(audit_durations)),
    }


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
    support_batches, proof = build_identity_full_support_batches(
        tok=tok,
        max_len=int(max_len),
        batch_size=int(batch_size),
        curriculum_seed=int(curriculum_seed),
        device=device,
    )
    if not support_batches:
        raise RuntimeError(
            f"identity-full dataset has no usable rows for batch_size={batch_size}"
        )
    first_batch = support_batches[0]
    first_model_batch = first_batch["batch"]
    first_metadata = first_batch["metadata"]
    batch_proof = {
        **proof,
        "selected_batch_index": 0,
        "selected_batch_metadata": first_metadata,
        "batch_shape": {
            "inputs": list(first_model_batch["inputs"].shape),
            "labels": list(first_model_batch["labels"].shape),
            "sep_positions": list(first_model_batch["sep_positions"].shape),
        },
    }
    return first_model_batch, batch_proof


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


def _parse_single_int(text: str) -> int | None:
    match = re.fullmatch(r"\s*(-?\d+)\s*", text)
    if match is None:
        return None
    return int(match.group(1))


def _decode_valid_tokens(tok: Any, token_ids: torch.Tensor) -> str:
    return tok.decode([int(token) for token in token_ids.detach().cpu().tolist()])


def score_strict_exact_and_parsed_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tok: Any,
    row_offset: int = 0,
    include_row_results: bool = False,
    max_failure_examples: int = 5,
) -> dict[str, Any]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "expected logits=(B,L,V) and labels=(B,L), got "
            f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "logits/labels sequence shape mismatch: "
            f"logits={tuple(logits.shape[:2])} labels={tuple(labels.shape)}"
        )
    pred_ids = torch.argmax(logits.detach(), dim=-1).to("cpu")
    labels_cpu = labels.detach().to("cpu")
    masks = labels_cpu != IGNORE_LABEL_ID
    row_has_labels = masks.any(dim=-1)
    strict_per_row = ((pred_ids == labels_cpu) | ~masks).all(dim=-1) & row_has_labels
    row_total = int(row_has_labels.sum().item())
    strict_count = int(strict_per_row.sum().item())
    parsed_count = 0
    failure_examples: list[dict[str, Any]] = []
    row_results: list[dict[str, Any]] = []
    for local_index in range(labels_cpu.shape[0]):
        if not bool(row_has_labels[local_index].item()):
            continue
        mask = masks[local_index]
        expected_text = _decode_valid_tokens(tok, labels_cpu[local_index][mask])
        predicted_text = _decode_valid_tokens(tok, pred_ids[local_index][mask])
        expected_value = _parse_single_int(expected_text)
        predicted_value = _parse_single_int(predicted_text)
        parsed_exact = (
            expected_value is not None
            and predicted_value is not None
            and expected_value == predicted_value
        )
        parsed_count += int(parsed_exact)
        row_result = {
            "row_index": int(row_offset + local_index),
            "strict_exact": bool(strict_per_row[local_index].item()),
            "parsed_exact": bool(parsed_exact),
            "expected_text": expected_text,
            "predicted_text": predicted_text,
            "expected_value": expected_value,
            "predicted_value": predicted_value,
        }
        if include_row_results:
            row_results.append(row_result)
        if (
            (not row_result["strict_exact"] or not row_result["parsed_exact"])
            and len(failure_examples) < int(max_failure_examples)
        ):
            failure_examples.append(row_result)
    report = {
        "strict_exact_count": strict_count,
        "strict_exact_total": row_total,
        "strict_exact": f"{strict_count}/{row_total}",
        "strict_exact_pct": float(strict_count / row_total) if row_total else 0.0,
        "parsed_exact_count": parsed_count,
        "parsed_exact_total": row_total,
        "parsed_exact": f"{parsed_count}/{row_total}",
        "parsed_exact_pct": float(parsed_count / row_total) if row_total else 0.0,
        "strict_exact_and_parsed_independent": True,
        "failure_examples": failure_examples,
    }
    if include_row_results:
        report["row_results"] = row_results
    return report


def aggregate_identity_full_audit_batch_reports(
    *,
    step: int,
    batch_reports: Sequence[Mapping[str, Any]],
    bp_steps: int,
) -> dict[str, Any]:
    strict_metric_count = sum(
        int(report["metric_strict"]["count"])
        for report in batch_reports
    )
    strict_metric_total = sum(
        int(report["metric_strict"]["total"])
        for report in batch_reports
    )
    strict_recomputed_count = sum(
        int(report["strict_recomputed"]["count"])
        for report in batch_reports
    )
    strict_recomputed_total = sum(
        int(report["strict_recomputed"]["total"])
        for report in batch_reports
    )
    parsed_count = sum(
        int(report["parsed"]["count"])
        for report in batch_reports
    )
    parsed_total = sum(
        int(report["parsed"]["total"])
        for report in batch_reports
    )
    loss_values = [
        float(report["loss"])
        for report in batch_reports
        if report.get("loss") is not None
    ]
    strict_recompute_mismatch = (
        strict_metric_count != strict_recomputed_count
        or strict_metric_total != strict_recomputed_total
    )
    audit_mismatch = (
        strict_recompute_mismatch
        or strict_metric_total != parsed_total
    )
    audited_hashes = [
        report["metadata"]["batch_content_hash16"]
        for report in batch_reports
    ]
    return {
        "schema": C2P2_AUDIT_SCHEMA_VERSION,
        "step": int(step),
        "support_rows_expected": C2P2_STRICT_EXACT_TARGET,
        "support_rows_audited": strict_metric_total,
        "audit_batch_count": len(batch_reports),
        "audited_batch_content_hashes": audited_hashes,
        "audited_distinct_batch_count": len(set(audited_hashes)),
        "strict_exact_count": strict_metric_count,
        "strict_exact_total": strict_metric_total,
        "strict_exact": f"{strict_metric_count}/{strict_metric_total}",
        "strict_exact_pct": (
            float(strict_metric_count / strict_metric_total)
            if strict_metric_total
            else 0.0
        ),
        "strict_exact_recomputed_from_logits_count": strict_recomputed_count,
        "strict_exact_recomputed_from_logits_total": strict_recomputed_total,
        "strict_exact_recompute_matches_metric": not strict_recompute_mismatch,
        "parsed_exact_count": parsed_count,
        "parsed_exact_total": parsed_total,
        "parsed_exact": f"{parsed_count}/{parsed_total}",
        "parsed_exact_pct": (
            float(parsed_count / parsed_total)
            if parsed_total
            else 0.0
        ),
        "strict_exact_and_parsed_independent": True,
        "acquired": (
            strict_metric_count == C2P2_STRICT_EXACT_TARGET
            and strict_metric_total == C2P2_STRICT_EXACT_TARGET
        ),
        "loss_mean": (
            float(sum(loss_values) / len(loss_values))
            if loss_values
            else None
        ),
        "bp_steps": int(bp_steps),
        "audit_mismatch": bool(audit_mismatch),
        "batch_reports": list(batch_reports),
    }


def audit_identity_full_support(
    model: LMHead,
    audit_batches: Sequence[Mapping[str, Any]],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    tok: Any,
    device: torch.device,
    step: int,
    total_steps: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    batch_reports: list[dict[str, Any]] = []
    try:
        extras = model.compute_train_extra_args(int(step), max(1, int(total_steps)))
        with torch.no_grad():
            with authoritative_forward_context(
                eligible_modules,
                tensor_states,
                device=device,
                requires_grad=False,
            ):
                for batch_item in audit_batches:
                    metadata = batch_item["metadata"]
                    batch = batch_item["batch"]
                    _carry, loss, metrics = model(
                        None,
                        dict(batch),
                        return_logits=True,
                        **extras,
                    )
                    metric_exact = metrics["exact_accuracy"]
                    metric_count = int(_tensor_scalar(metric_exact[0]))
                    metric_total = int(_tensor_scalar(metric_exact[1]))
                    score = score_strict_exact_and_parsed_from_logits(
                        metrics["logits"],
                        batch["labels"],
                        tok=tok,
                        row_offset=int(metadata["row_start"]),
                    )
                    batch_reports.append(
                        {
                            "metadata": metadata,
                            "loss": float(loss.detach().cpu().item()),
                            "metrics": _metrics_to_dict(metrics),
                            "metric_strict": {
                                "count": metric_count,
                                "total": metric_total,
                                "strict_exact": f"{metric_count}/{metric_total}",
                            },
                            "strict_recomputed": {
                                "count": int(score["strict_exact_count"]),
                                "total": int(score["strict_exact_total"]),
                                "strict_exact": score["strict_exact"],
                            },
                            "parsed": {
                                "count": int(score["parsed_exact_count"]),
                                "total": int(score["parsed_exact_total"]),
                                "parsed_exact": score["parsed_exact"],
                            },
                            "failure_examples": score["failure_examples"],
                        }
                    )
    finally:
        model.train(was_training)
    return aggregate_identity_full_audit_batch_reports(
        step=int(step),
        batch_reports=batch_reports,
        bp_steps=int(extras["bp_steps"]),
    )


def _step_q_changed_total(step_reports: Mapping[str, Any]) -> int:
    return sum(
        int(report.get("q_changed_count", 0))
        for report in step_reports.values()
    )


def classify_c2p2_null(
    *,
    audit_reports: Mapping[str, Any],
    step_reports: Mapping[str, Any],
    support_cycler_proof: Mapping[str, Any],
) -> str | None:
    if not audit_reports:
        return None
    ordered = [
        audit_reports[key]
        for key in sorted(audit_reports, key=lambda item: int(item))
    ]
    final = ordered[-1]
    if final.get("acquired") is True:
        return None
    if (
        any(report.get("audit_mismatch") for report in ordered)
        or final.get("strict_exact_total") != C2P2_STRICT_EXACT_TARGET
        or not support_cycler_proof.get("covers_full_support", False)
    ):
        return "audit-mismatch"
    if any(
        report.get("loss_finite") is False
        or report.get("weighted_grad_finite") is False
        for report in step_reports.values()
    ):
        return "nonfinite"
    if _step_q_changed_total(step_reports) <= 0:
        return "no-q-move"
    baseline = audit_reports.get("0", ordered[0])
    if int(final["strict_exact_count"]) > int(baseline["strict_exact_count"]):
        return "partial-acquisition-plateau"
    return "q-move-no-accuracy"


def build_acquisition_trajectory(
    *,
    audit_enabled: bool,
    audit_reports: Mapping[str, Any],
    step_reports: Mapping[str, Any],
    support_cycler_proof: Mapping[str, Any],
    audit_interval: int,
    stop_on_strict_exact: bool,
    max_steps_hard: int,
    stop_reason: str,
    timing_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timing_payload = dict(timing_summary) if timing_summary is not None else None
    if not audit_enabled:
        return {
            "schema": C2P2_TRAJECTORY_SCHEMA_VERSION,
            "enabled": False,
            "reason": "audit_interval<=0 and stop_on_strict_exact disabled",
            "timing_summary": timing_payload,
            "null_taxonomy": list(C2P2_NULL_TAXONOMY),
            "null_escalation_rule": C2P2_NULL_ESCALATION_RULE,
        }
    ordered_steps = sorted(audit_reports, key=lambda item: int(item))
    ordered_audits = [audit_reports[key] for key in ordered_steps]
    baseline = audit_reports.get("0", ordered_audits[0] if ordered_audits else None)
    final = ordered_audits[-1] if ordered_audits else None
    trained_batch_hashes = [
        report["support_batch"]["batch_content_hash16"]
        for report in step_reports.values()
        if "support_batch" in report
    ]
    acquired = bool(final and final.get("acquired"))
    null_class = classify_c2p2_null(
        audit_reports=audit_reports,
        step_reports=step_reports,
        support_cycler_proof=support_cycler_proof,
    )
    baseline_count = int(baseline["strict_exact_count"]) if baseline else 0
    final_count = int(final["strict_exact_count"]) if final else 0
    final_total = int(final["strict_exact_total"]) if final else 0
    return {
        "schema": C2P2_TRAJECTORY_SCHEMA_VERSION,
        "enabled": True,
        "recipe": {
            "stability": "OUT",
            "replay": "OUT/pure-rung",
            "audit_interval": int(audit_interval),
            "stop_on_strict_exact": bool(stop_on_strict_exact),
            "max_steps_hard": int(max_steps_hard),
            "steps_upper_bound": C2P2_DEFAULT_MAX_STEPS_HARD,
            "acquire_gate": f"{C2P2_STRICT_EXACT_TARGET}/{C2P2_STRICT_EXACT_TARGET} strict-exact",
        },
        "timing_summary": timing_payload,
        "acquisition_definition": (
            "Acquisition is current-q exhaustive strict-exact "
            "90/90 on the identity-full support; step-0 M/90 is the parent "
            "baseline denominator, not a bank/pass verdict."
        ),
        "support_cycler_proof": support_cycler_proof,
        "support_cycler_distinctness": {
            "trained_batch_content_hashes": trained_batch_hashes,
            "trained_distinct_batch_count": len(set(trained_batch_hashes)),
            "trained_at_least_two_distinct_batches": len(set(trained_batch_hashes)) >= 2,
            "audited_distinct_batch_count": (
                int(final["audited_distinct_batch_count"])
                if final
                else 0
            ),
            "audited_at_least_two_distinct_batches": (
                int(final["audited_distinct_batch_count"]) >= 2
                if final
                else False
            ),
        },
        "audit_steps": [int(step) for step in ordered_steps],
        "audits": {
            str(step): audit_reports[str(step)]
            for step in ordered_steps
        },
        "baseline_strict_exact_at_step0": (
            {
                "strict_exact_count": int(baseline["strict_exact_count"]),
                "strict_exact_total": int(baseline["strict_exact_total"]),
                "strict_exact": baseline["strict_exact"],
                "strict_exact_pct": baseline["strict_exact_pct"],
                "parsed_exact_count": int(baseline["parsed_exact_count"]),
                "parsed_exact_total": int(baseline["parsed_exact_total"]),
                "parsed_exact": baseline["parsed_exact"],
            }
            if baseline
            else None
        ),
        "final_audit": (
            {
                "step": int(final["step"]),
                "strict_exact_count": final_count,
                "strict_exact_total": final_total,
                "strict_exact": final["strict_exact"],
                "strict_exact_pct": final["strict_exact_pct"],
                "parsed_exact_count": int(final["parsed_exact_count"]),
                "parsed_exact_total": int(final["parsed_exact_total"]),
                "parsed_exact": final["parsed_exact"],
                "acquired": bool(final["acquired"]),
            }
            if final
            else None
        ),
        "baseline_to_final_delta": final_count - baseline_count,
        "baseline_to_target_delta_remaining": C2P2_STRICT_EXACT_TARGET - final_count,
        "stop_reason": stop_reason,
        "acquisition_verdict": "acquired" if acquired else "no_acquisition_verdict",
        "null_attribution_class": null_class,
        "null_taxonomy": list(C2P2_NULL_TAXONOMY),
        "null_escalation_rule": C2P2_NULL_ESCALATION_RULE,
        "total_q_changed_count": _step_q_changed_total(step_reports),
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
    support_batches: Sequence[Mapping[str, Any]] | None = None,
    audit_callback: Callable[[int, Mapping[str, Any]], dict[str, Any]] | None = None,
    audit_interval: int = 0,
    stop_on_strict_exact: bool = False,
    phase_progress: PhaseProgress | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, int]:
    model.train()
    progress = phase_progress or PhaseProgress(enabled=False, device=device)
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
    audit_reports: dict[str, Any] = {}
    if support_batches:
        step_batches = list(support_batches)
    else:
        row_count = int(batch["inputs"].shape[0])
        step_batches = [
            {
                "batch": batch,
                "metadata": {
                    "batch_index": 0,
                    "row_start": 0,
                    "row_end_exclusive": row_count,
                    "row_count": row_count,
                    "row_ids": [],
                    "sample_hashes": [],
                    "batch_content_hash16": _sha16(
                        {
                            "legacy_single_batch_shape": list(batch["inputs"].shape),
                            "row_count": row_count,
                        }
                    ),
                    "legacy_single_batch": True,
                },
            }
        ]
    if not step_batches:
        raise RuntimeError("bounded-delta step loop requires at least one support batch")

    def maybe_audit(step: int, *, final: bool = False) -> bool:
        if audit_callback is None:
            return False
        if int(step) == 0:
            audit_phase = "audit0"
        elif final:
            audit_phase = "final_audit"
        else:
            audit_phase = "audit"
        with progress.phase(audit_phase, step=int(step)):
            audit_timing_start = _timing_start(device)
            audit_report = dict(audit_callback(int(step), states))
            audit_report["duration_seconds"] = _timing_duration_seconds(
                audit_timing_start,
                device,
            )
        audit_reports[str(step)] = audit_report
        return (
            bool(stop_on_strict_exact)
            and bool(audit_reports[str(step)].get("acquired"))
        )

    if maybe_audit(0):
        updater_config = {
            "rank_vote_spec": rank_spec.to_live_dict(),
            "vote_update_spec": asdict(vote_spec),
            "projection_law": S1_PROJECTION_LAW,
            "vote_law": S1_RANK_BUCKET_VOTE_LAW,
        }
        return (
            step_reports,
            updater_config,
            states,
            audit_reports,
            "strict_exact_acquired_step0",
            0,
        )
    if steps <= 0:
        updater_config = {
            "rank_vote_spec": rank_spec.to_live_dict(),
            "vote_update_spec": asdict(vote_spec),
            "projection_law": S1_PROJECTION_LAW,
            "vote_law": S1_RANK_BUCKET_VOTE_LAW,
        }
        return step_reports, updater_config, states, audit_reports, "no_steps", 0

    stop_reason = "max_steps_completed"
    steps_completed = 0
    for step in range(1, int(steps) + 1):
        with progress.phase("step", step=int(step)):
            step_timing_start = _timing_start(device)
            batch_item = step_batches[(step - 1) % len(step_batches)]
            step_batch = batch_item["batch"]
            step_batch_metadata = batch_item["metadata"]
            model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            extras = model.compute_train_extra_args(step, max(1, int(steps)))
            with progress.phase("step_forward_backward", step=int(step)):
                with authoritative_forward_context(
                    eligible_modules,
                    states,
                    device=device,
                    requires_grad=True,
                ) as handle:
                    _carry, loss, metrics = model(None, dict(step_batch), **extras)
                    loss.backward()
                    weighted_grads = {
                        key: handle.weighted_grad(key)
                        for key in states
                    }
            with progress.phase("step_update", step=int(step)):
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
                step_duration_seconds = _timing_duration_seconds(
                    step_timing_start,
                    device,
                )
                step_reports[str(step)] = {
                    "loss": float(loss.detach().cpu().item()),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "weighted_grad_finite": bool(finite_weighted_grad),
                    "duration_seconds": step_duration_seconds,
                    "metrics": _metrics_to_dict(metrics),
                    "bp_steps": int(extras["bp_steps"]),
                    "q_changed_count": q_changed_count,
                    "support_batch": dict(step_batch_metadata),
                    "step_result": step_result.to_compact_dict(),
                    "optimizer_identity_proof": identity_proof,
                }
        steps_completed = step
        if (
            audit_callback is not None
            and int(audit_interval) > 0
            and step % int(audit_interval) == 0
            and maybe_audit(step)
        ):
            stop_reason = "strict_exact_acquired_stop_fast"
            break
    if audit_callback is not None and str(steps_completed) not in audit_reports:
        if maybe_audit(steps_completed, final=True):
            stop_reason = "strict_exact_acquired_final"
    updater_config = {
        "rank_vote_spec": rank_spec.to_live_dict(),
        "vote_update_spec": asdict(vote_spec),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }
    return (
        step_reports,
        updater_config,
        states,
        audit_reports,
        stop_reason,
        steps_completed,
    )


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
    audit_interval: int = 0,
    stop_on_strict_exact: bool = False,
    max_steps_hard: int = C2P2_DEFAULT_MAX_STEPS_HARD,
    emit_progress: bool = False,
    phase_timeout_seconds: float = 0.0,
    total_timeout_seconds: float = 0.0,
    enabled: bool | None = None,
    allow_gpu_launch: bool = False,
) -> dict[str, Any]:
    assert_default_off(enabled)
    if int(max_steps_hard) <= 0:
        raise ValueError("max_steps_hard must be positive")
    if int(steps) > int(max_steps_hard):
        raise ValueError(
            f"steps={int(steps)} exceeds max_steps_hard={int(max_steps_hard)}"
        )
    if int(audit_interval) < 0:
        raise ValueError("audit_interval must be non-negative")
    torch_device = torch.device(device)
    guard_gpu_launch(torch_device, allow_gpu_launch=allow_gpu_launch)
    device_guard = assert_probe_device_ready(torch_device)
    phase_progress = PhaseProgress(
        enabled=bool(emit_progress),
        device=torch_device,
        phase_timeout_seconds=float(phase_timeout_seconds),
        total_timeout_seconds=float(total_timeout_seconds),
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    if torch_device.type == "cuda":
        with phase_progress.phase("cuda_memory_reset"):
            reset_cuda_memory_stats(torch_device)
    run_timing_start = _timing_start(torch_device)

    with phase_progress.phase("load"):
        ckpt, parent_hash_before = load_parent_checkpoint(parent, expected_sha256=parent_sha256)
    with phase_progress.phase("build_model"):
        model, tok, cfg = build_model_from_checkpoint(ckpt, torch_device)
    with phase_progress.phase("support_build"):
        support_batches, support_cycler_proof = build_identity_full_support_batches(
            tok=tok,
            max_len=int(max_len or ckpt["config"]["max_seq_len"]),
            batch_size=int(batch_size),
            curriculum_seed=int(curriculum_seed),
            device=torch_device,
        )
    model_batch = support_batches[0]["batch"]
    batch_proof = {
        **support_cycler_proof,
        "selected_batch_index": 0,
        "selected_batch_metadata": support_batches[0]["metadata"],
        "batch_shape": {
            "inputs": list(model_batch["inputs"].shape),
            "labels": list(model_batch["labels"].shape),
            "sep_positions": list(model_batch["sep_positions"].shape),
        },
    }
    with phase_progress.phase("support_control"):
        support_control_proof = identity_full_support_control_proof(int(curriculum_seed))
    with phase_progress.phase("select_eligible"):
        eligible = select_eligible_bitlinears(model, eligible_scope=eligible_scope)
    with phase_progress.phase("state_init"):
        tensor_states, init_fidelity = derive_tensor_states_and_check_init_fidelity(
            eligible,
            threshold=float(init_fidelity_atol),
        )
    if not init_fidelity["all_pass"]:
        raise RuntimeError("weight-level init-fidelity allclose failed")

    with phase_progress.phase("forward_fidelity"):
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
        with phase_progress.phase("step0_optimizer_identity"):
            step0_optimizer_identity_proof = prove_step0_optimizer_identity(model, eligible)

    audit_enabled = int(audit_interval) > 0 or bool(stop_on_strict_exact)

    def audit_callback(step: int, states: Mapping[str, Any]) -> dict[str, Any]:
        return audit_identity_full_support(
            model,
            support_batches,
            states,
            eligible,
            tok=tok,
            device=torch_device,
            step=int(step),
            total_steps=max(1, int(steps)),
        )

    with phase_progress.phase("bounded_steps"):
        (
            step_reports,
            updater_config,
            final_states,
            audit_reports,
            stop_reason,
            steps_completed,
        ) = run_bounded_delta_steps(
            model,
            model_batch,
            tensor_states,
            eligible,
            device=torch_device,
            steps=int(steps),
            require_q_change=bool(require_q_change),
            max_abs_per_tensor=int(max_abs_per_tensor),
            support_batches=support_batches,
            audit_callback=audit_callback if audit_enabled else None,
            audit_interval=int(audit_interval),
            stop_on_strict_exact=bool(stop_on_strict_exact),
            phase_progress=phase_progress,
        )
    if not updater_config:
        updater_config = {
            "rank_vote_spec": default_dry_run_rank_vote_spec().to_live_dict(),
            "vote_update_spec": asdict(default_vote_update_spec(max_abs_per_tensor)),
            "projection_law": S1_PROJECTION_LAW,
            "vote_law": S1_RANK_BUCKET_VOTE_LAW,
        }
    with phase_progress.phase("checkpoint_payload"):
        checkpoint_payload = build_authoritative_checkpoint_payload(
            final_states,
            step=int(steps_completed),
            updater_config=updater_config,
            oracle_receipt=None,
            dry_run=True,
            checkpoint_written=False,
        )
        validate_authoritative_resume_payload(checkpoint_payload)
    with phase_progress.phase("parent_hash_after"):
        parent_hash_after = file_sha256(parent)
    parent_hash_unchanged = parent_hash_before == parent_hash_after
    if not parent_hash_unchanged:
        raise RuntimeError("parent checkpoint hash changed during C2.1 probe")
    total_run_duration_seconds = _timing_duration_seconds(
        run_timing_start,
        torch_device,
    )
    timing_summary = build_timing_summary(
        step_reports=step_reports,
        audit_reports=audit_reports,
        total_run_duration_seconds=total_run_duration_seconds,
    )
    receipt = {
        "schema": C2P1_HARNESS_SCHEMA_VERSION,
        "c2p0_schema": BOUNDED_DELTA_LEARNER_SCHEMA_VERSION,
        "bounded_delta_checkpoint_schema": BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
        "phase": phase,
        "implementation_gpu_validation_split": True,
        "gpu_launch_authorized": bool(torch_device.type == "cuda"),
        "gpu_launched": bool(torch_device.type == "cuda"),
        "device": str(torch_device),
        "device_guard": device_guard,
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
        "identity_full_control": support_control_proof,
        "support_cycler": support_cycler_proof,
        "eligible_scope": eligible_scope,
        "eligible_module_count": len(eligible),
        "eligible_modules": sorted(eligible),
        "weight_level_init_fidelity": init_fidelity,
        "forward_level_init_fidelity": forward_init_fidelity,
        "steps_requested": int(steps),
        "steps_completed": int(steps_completed),
        "max_steps_hard": int(max_steps_hard),
        "audit_interval": int(audit_interval),
        "stop_on_strict_exact": bool(stop_on_strict_exact),
        "stop_reason": stop_reason,
        "forward_backward_update_executed": bool(steps > 0),
        "step0_optimizer_identity_proof": step0_optimizer_identity_proof,
        "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
        "step_reports": step_reports,
        "audit_reports": audit_reports,
        "timing_summary": timing_summary,
        "acquisition_trajectory": build_acquisition_trajectory(
            audit_enabled=audit_enabled,
            audit_reports=audit_reports,
            step_reports=step_reports,
            support_cycler_proof=support_cycler_proof,
            audit_interval=int(audit_interval),
            stop_on_strict_exact=bool(stop_on_strict_exact),
            max_steps_hard=int(max_steps_hard),
            stop_reason=stop_reason,
            timing_summary=timing_summary,
        ),
        "checkpoint_payload": checkpoint_payload,
        "memory": cuda_memory_receipt(torch_device),
        "phase_telemetry": phase_progress.to_dict(),
    }
    receipt_path = scratch_root / "receipt.json"
    phase_progress.mark("receipt_write", "start", path=str(receipt_path))
    receipt["phase_telemetry"] = phase_progress.to_dict()
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    phase_progress.mark("receipt_write", "end", path=str(receipt_path))
    receipt["phase_telemetry"] = phase_progress.to_dict()
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
    ap.add_argument("--audit-interval", type=int, default=0)
    ap.add_argument("--stop-on-strict-exact", action="store_true")
    ap.add_argument("--max-steps-hard", type=int, default=C2P2_DEFAULT_MAX_STEPS_HARD)
    ap.add_argument("--emit-progress", action="store_true")
    ap.add_argument("--phase-timeout-seconds", type=float, default=0.0)
    ap.add_argument("--total-timeout-seconds", type=float, default=0.0)
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
        audit_interval=args.audit_interval,
        stop_on_strict_exact=args.stop_on_strict_exact,
        max_steps_hard=args.max_steps_hard,
        emit_progress=args.emit_progress,
        phase_timeout_seconds=args.phase_timeout_seconds,
        total_timeout_seconds=args.total_timeout_seconds,
        enabled=args.enable_bounded_delta_probe,
        allow_gpu_launch=args.allow_gpu_launch,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
