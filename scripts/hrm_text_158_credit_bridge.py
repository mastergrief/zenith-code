#!/usr/bin/env python3
"""Read-only HRM-Text-1.58 magnitude-aware credit bridge diagnostic.

This diagnostic compares strict, coarse-magnitude, groupwise-scale, and full
magnitude credit signals against the magnitude-aware STE master-weight
gradient, after each is projected onto the same admissible one-step ternary
moves. It is intentionally read-only: no optimizer, no checkpoint save, and
checkpoint SHA-256 must match before/after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn


TASK_ID = "1780239430784-b2c90d8d"
DEFAULT_CKPT = Path(
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_"
    "pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
DEFAULT_OUT_DIR = Path(f"/tmp/claw-code-creditdir/hrm_bridge_credit/{TASK_ID}")
DEFAULT_PUBLIC_OUT_DIR = Path(f"/home/gabe/claw-code-creditdir/hrm_bridge_credit/{TASK_ID}")
IGNORE_LABEL_ID = -100
GROUP_ORDER = (
    "attn.gqkv.gate",
    "attn.gqkv.query",
    "attn.gqkv.key",
    "attn.gqkv.value",
    "attn.o",
    "mlp.gate_up.gate",
    "mlp.gate_up.up",
    "mlp.down",
)
CREDIT_VARIANTS = ("strict", "pow2_bucket", "fp16_groupwise", "full_magnitude_ceiling")
NULL_BACKEND_CPU_LOCKED = "cpu_locked"
NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY = "cpu_sampler_gpu_aggregation_replay"
NULL_BACKEND_GPU_NATIVE_COUNTS_PMF = "gpu_native_counts_pmf"
NULL_BACKENDS = (
    NULL_BACKEND_CPU_LOCKED,
    NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
    NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
)
DEFAULT_NULL_BACKEND = NULL_BACKEND_CPU_LOCKED
DEFAULT_NULL_SPEEDUP_FLOOR = 1.25
DEFAULT_NULL_PROFILE_WARMUPS = 1
DEFAULT_NULL_PROFILE_REPEATS = 5
DEFAULT_NULL_SPEED_CPU_REPEATS = 1
DEFAULT_NULL_SPEED_CANDIDATE_REPEATS = 5
DEFAULT_NULL_SPEED_MAX_INVOCATIONS_PER_VARIANT = 3
NULL_DISTRIBUTIONAL_ABS_TOL = 0.01
GPU_NATIVE_PMF_TV_BOUND = 1e-5
GPU_NATIVE_PMF_CDF_BOUND = 1e-5
GPU_NATIVE_PMF_CAPTURED_MASS_EPS = 1e-6
GPU_NATIVE_PMF_FULL_SUPPORT_GUARD = 65_536
GPU_NATIVE_REFERENCE_SCIPY_MAX_POINTS = 4096
GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET = 32_000_000
GPU_NATIVE_REFERENCE_CHUNK_CELL_BUDGET = 1_000_000
GPU_NATIVE_BOUNDED_SAMPLE_COUNT = 65_536
GPU_NATIVE_BOUNDED_SAMPLE_CONFIDENCE = 0.999
GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND = 0.05
GPU_NATIVE_BOUNDED_SAMPLE_SEED = 15817
GPU_NATIVE_STAGE2_MAX_SECONDS = 600.0
GPU_NATIVE_CURRENT_CPU_LOWER_BOUND_SECONDS = 1526.0
GPU_NATIVE_PRIOR_CPU_LOWER_BOUND_SECONDS = 6600.0
NULL_SAMPLER_BOUND_FRACTION = 0.60
STRICT_REPRODUCTION_EXPECTED = 0.508805533381048
STRICT_REPRODUCTION_TOL = 0.01
POW2_EXP_MIN = -24
POW2_EXP_MAX = 15
POW2_ROUND_MODE = "round"
FP16_GROUPWISE_GROUP_SIZE = 128
FP16_GROUPWISE_SCALE_STAT = "mean_abs"
CREDIT_TERMINAL_LABELS = (
    "pow2_magnitude_sufficient",
    "fp16_groupwise_credit_sufficient_proxy",
    "tested_lowbit_magnitude_insufficient",
    "diagnostic_reference_invalid",
    "integrity_failure",
)
NULL_PARITY_TERMINAL_LABELS = (
    "gpu_null_parity_exact_default_enabled",
    "gpu_null_parity_exact_speedup_insufficient_cpu_default_retained",
    "gpu_null_parity_exact_sampler_bound_deferred",
    "gpu_null_parity_fail",
    "diagnostic_reference_invalid",
    "integrity_failure",
)
GPU_NATIVE_NULL_TERMINAL_LABELS = (
    "gpu_native_null_parity_default_enabled",
    "gpu_native_null_parity_explicit_validated_default_deferred",
    "gpu_native_null_parity_speedup_insufficient_cpu_default_retained",
    "gpu_native_null_parity_fail",
    "diagnostic_reference_invalid",
    "integrity_failure",
)
REFERENCE_PMF_FUNCTION = "_joint_match_pmf_reference_scipy_vectorized_sparse"
CANDIDATE_PMF_FUNCTION = "_joint_match_pmf_gpu_windowed_sparse"
TARGET_RE = re.compile(
    r"^model\.(?P<level>[HL])_level\.core\.layers\.(?P<layer>\d+)\."
    r"(?P<block>attn|mlp)\.(?P<proj>gqkv_proj|o_proj|gate_up_proj|down_proj)$"
)


class DiagnosticInvalid(RuntimeError):
    """Raised when the diagnostic reference is invalid before terminal scoring."""


class IntegrityFailure(RuntimeError):
    """Raised when the checkpoint/read-only invariants fail."""


@dataclass(frozen=True)
class Bars:
    global_floor: float = 0.65
    family_floor: float = 0.60
    stratum_floor: float = 0.55
    global_null_margin: float = 0.10
    family_null_margin: float = 0.05
    stratum_null_margin: float = 0.03
    min_projected_denom: int = 4096
    min_active_outputs: int = 128
    min_q0_projected_denom_plausible: int = 1024
    min_q0_projected_denom_valid: int = 256
    route_p01_floor: int = 16
    route_median_floor: int = 64
    family_dead_rate_route_death: float = 0.01
    stratum_dead_rate_route_death: float = 0.05


@dataclass(frozen=True)
class TargetInfo:
    name: str
    level: str
    layer: int
    proj: str
    module: nn.Module


@dataclass(frozen=True)
class GroupInfo:
    target: TargetInfo
    group: str
    start: int
    end: int

    @property
    def key_base(self) -> str:
        return f"{self.target.level}.layer{self.target.layer}.{self.group}"


@dataclass(frozen=True)
class InvocationKey:
    level: str
    rec_idx: int
    layer: int
    group: str

    @property
    def label(self) -> str:
        return f"{self.level}.rec{self.rec_idx}.layer{self.layer}.{self.group}"

    @property
    def family_label(self) -> str:
        return f"{self.level}.{self.group}"

    @property
    def aggregate64_label(self) -> str:
        return f"{self.level}.layer{self.layer}.{self.group}"


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, values: Tensor) -> None:
        if values.numel() == 0:
            return
        v = values.detach().to(torch.float32)
        self.count += int(v.numel())
        self.total += float(v.sum().item())
        self.total_sq += float((v * v).sum().item())

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None, "cv": None}
        mean = self.total / self.count
        var = max(0.0, self.total_sq / self.count - mean * mean)
        std = math.sqrt(var)
        cv = std / mean if mean != 0 else None
        return {"count": self.count, "mean": mean, "std": std, "cv": cv}


@dataclass
class InvocationAggregate:
    key: InvocationKey
    module_name: str
    group_start: int
    group_end: int
    in_features: int
    variant_credits: dict[str, Tensor] = field(default_factory=dict)
    weighted_grad: Tensor | None = None
    active_inputs: Tensor | None = None
    active_outputs: Tensor | None = None
    backward_calls: int = 0
    prefix_active_positions: int = 0
    response_active_positions: int = 0
    prefix_active_output_elements: int = 0
    response_active_output_elements: int = 0
    input_abs: RunningMoments = field(default_factory=RunningMoments)
    grad_abs: RunningMoments = field(default_factory=RunningMoments)

    def accumulate(
        self,
        *,
        variant_credits: dict[str, Tensor],
        weighted_grad: Tensor,
        active_inputs: Tensor,
        active_outputs: Tensor,
        prefix_active_positions: int,
        response_active_positions: int,
        prefix_active_output_elements: int,
        response_active_output_elements: int,
        input_abs_values: Tensor,
        grad_abs_values: Tensor,
    ) -> None:
        credit_cpu = {name: value.detach().cpu() for name, value in variant_credits.items()}
        weighted_cpu = weighted_grad.detach().to(torch.float32).cpu()
        active_inputs_cpu = active_inputs.detach().to(torch.bool).cpu()
        active_outputs_cpu = active_outputs.detach().to(torch.bool).cpu()

        if self.weighted_grad is None:
            self.weighted_grad = torch.zeros_like(weighted_cpu)
            self.active_inputs = torch.zeros_like(active_inputs_cpu)
            self.active_outputs = torch.zeros_like(active_outputs_cpu)
        assert self.weighted_grad is not None
        assert self.active_inputs is not None
        assert self.active_outputs is not None

        for name, value in credit_cpu.items():
            if name not in self.variant_credits:
                self.variant_credits[name] = torch.zeros_like(value)
            self.variant_credits[name] += value
        self.weighted_grad += weighted_cpu
        self.active_inputs |= active_inputs_cpu
        self.active_outputs |= active_outputs_cpu
        self.backward_calls += 1
        self.prefix_active_positions += prefix_active_positions
        self.response_active_positions += response_active_positions
        self.prefix_active_output_elements += prefix_active_output_elements
        self.response_active_output_elements += response_active_output_elements
        self.input_abs.update(input_abs_values.detach().cpu())
        self.grad_abs.update(grad_abs_values.detach().cpu())


@dataclass
class ScheduleExcluded:
    key: InvocationKey
    module_name: str
    reason: str


@dataclass(frozen=True)
class BucketCounts:
    fp_pos: int
    fp_neg: int
    int_pos: int
    int_neg: int
    int_zero: int

    @property
    def total(self) -> int:
        return self.fp_pos + self.fp_neg


@dataclass
class CountAccumulator:
    label: str
    denom: int = 0
    agree: int = 0
    buckets_global: list[BucketCounts] = field(default_factory=list)
    buckets_rowq: list[BucketCounts] = field(default_factory=list)
    q_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    raw_dir_denom: int = 0
    raw_dir_disagree: int = 0
    raw_dir_integer_zero: int = 0
    active_output_count: int = 0
    dead_active_output_count: int = 0
    route_counts: list[int] = field(default_factory=list)
    current_route_counts: list[int] = field(default_factory=list)
    prefix_active_positions: int = 0
    response_active_positions: int = 0
    prefix_active_output_elements: int = 0
    response_active_output_elements: int = 0

    def merge(self, other: "CountAccumulator") -> None:
        self.denom += other.denom
        self.agree += other.agree
        self.buckets_global.extend(other.buckets_global)
        self.buckets_rowq.extend(other.buckets_rowq)
        for q, stats in other.q_stats.items():
            mine = self.q_stats.setdefault(q, {"denom": 0, "agree": 0})
            mine["denom"] += stats["denom"]
            mine["agree"] += stats["agree"]
        self.raw_dir_denom += other.raw_dir_denom
        self.raw_dir_disagree += other.raw_dir_disagree
        self.raw_dir_integer_zero += other.raw_dir_integer_zero
        self.active_output_count += other.active_output_count
        self.dead_active_output_count += other.dead_active_output_count
        self.route_counts.extend(other.route_counts)
        self.current_route_counts.extend(other.current_route_counts)
        self.prefix_active_positions += other.prefix_active_positions
        self.response_active_positions += other.response_active_positions
        self.prefix_active_output_elements += other.prefix_active_output_elements
        self.response_active_output_elements += other.response_active_output_elements


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_with_sha(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = stable_json(obj).encode("utf-8")
    path.write_bytes(data)
    digest = sha256_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def project_fp_gradient_to_moves(grad: Tensor, q_levels: Tensor) -> Tensor:
    """Project magnitude-aware gradient descent onto one-step ternary moves."""
    moves = torch.zeros_like(q_levels, dtype=torch.int8)
    moves[(q_levels < 0) & (grad < 0)] = 1
    moves[(q_levels == 0) & (grad < 0)] = 1
    moves[(q_levels == 0) & (grad > 0)] = -1
    moves[(q_levels > 0) & (grad > 0)] = -1
    return moves


def project_integer_credit_to_moves(credit: Tensor, q_levels: Tensor) -> Tensor:
    """Project strict integer credit onto one-step ternary moves."""
    moves = torch.zeros_like(q_levels, dtype=torch.int8)
    moves[(q_levels <= 0) & (credit > 0)] = 1
    moves[(q_levels >= 0) & (credit < 0)] = -1
    moves[(q_levels <= -1) & (moves < 0)] = 0
    moves[(q_levels >= 1) & (moves > 0)] = 0
    return moves


def strict_sign_credit(grad_chunk: Tensor, input_tensor: Tensor) -> Tensor:
    """Magnitude-free signed credit before projection."""
    return -torch.einsum("bso,bsi->oi", grad_chunk.to(torch.float32).sign(), input_tensor.to(torch.float32).sign())


def pow2_bucket_values(values: Tensor, *, exp_min: int = POW2_EXP_MIN, exp_max: int = POW2_EXP_MAX) -> Tensor:
    """Map values to sign plus rounded/clipped base-2 exponent buckets."""
    values_f = values.to(torch.float32)
    out = torch.zeros_like(values_f)
    nonzero = values_f != 0
    if bool(nonzero.any().item()):
        abs_values = values_f[nonzero].abs()
        exponents = torch.log2(abs_values).round().clamp(float(exp_min), float(exp_max))
        out[nonzero] = values_f[nonzero].sign() * torch.exp2(exponents)
    return out


def pow2_bucket_credit(grad_chunk: Tensor, input_tensor: Tensor) -> Tensor:
    """Coarse exponent-magnitude credit; magnitudes weight signed credit before projection."""
    return -torch.einsum("bso,bsi->oi", pow2_bucket_values(grad_chunk), pow2_bucket_values(input_tensor))


def signed_fp16_groupwise_mean_abs(
    values: Tensor,
    *,
    group_size: int = FP16_GROUPWISE_GROUP_SIZE,
) -> Tensor:
    """Replace each nonzero element magnitude with one fp16 mean-abs group scale."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    values_f = values.to(torch.float32)
    signed = torch.empty_like(values_f)
    features = values_f.shape[-1]
    for start in range(0, features, group_size):
        end = min(start + group_size, features)
        chunk = values_f[..., start:end]
        scale = chunk.abs().mean(dim=-1, keepdim=True).to(torch.float16).to(torch.float32)
        signed[..., start:end] = chunk.sign() * scale
    return signed


def fp16_groupwise_credit(
    grad_chunk: Tensor,
    input_tensor: Tensor,
    *,
    group_size: int = FP16_GROUPWISE_GROUP_SIZE,
) -> Tensor:
    """Groupwise-scale credit; scales weight signed credit before projection."""
    signed_grad = signed_fp16_groupwise_mean_abs(grad_chunk, group_size=group_size)
    signed_input = signed_fp16_groupwise_mean_abs(input_tensor, group_size=group_size)
    return -torch.einsum("bso,bsi->oi", signed_grad, signed_input)


def ternary_levels(weight: Tensor, eps: float = 1e-5) -> Tensor:
    scale = weight.detach().abs().mean().clamp(min=eps)
    return (weight.detach() / scale).round().clamp(-1.0, 1.0).to(torch.int8)


def _slice_groups(target: TargetInfo) -> list[GroupInfo]:
    out_features = int(target.module.weight.shape[0])
    if target.proj == "gqkv_proj":
        width = out_features // 4
        if width * 4 != out_features:
            raise DiagnosticInvalid(f"{target.name}: gqkv out_features not divisible by 4")
        names = ("attn.gqkv.gate", "attn.gqkv.query", "attn.gqkv.key", "attn.gqkv.value")
        return [GroupInfo(target, name, i * width, (i + 1) * width) for i, name in enumerate(names)]
    if target.proj == "gate_up_proj":
        width = out_features // 2
        if width * 2 != out_features:
            raise DiagnosticInvalid(f"{target.name}: gate_up out_features not divisible by 2")
        return [
            GroupInfo(target, "mlp.gate_up.gate", 0, width),
            GroupInfo(target, "mlp.gate_up.up", width, 2 * width),
        ]
    if target.proj == "o_proj":
        return [GroupInfo(target, "attn.o", 0, out_features)]
    if target.proj == "down_proj":
        return [GroupInfo(target, "mlp.down", 0, out_features)]
    raise DiagnosticInvalid(f"unsupported projection {target.name}")


def expected_grad_rec_indices(level: str, *, h_cycles: int = 2, l_cycles: int = 3, bp_steps: int = 5) -> set[int]:
    h_bp_steps = min(h_cycles, bp_steps - 1)
    l_bp_steps = bp_steps - h_bp_steps
    if level == "H":
        return set(range(h_cycles - h_bp_steps, h_cycles))
    total_l = h_cycles * l_cycles
    return set(range(total_l - l_bp_steps, total_l))


def expected_forward_calls(level: str, *, h_cycles: int = 2, l_cycles: int = 3) -> int:
    return h_cycles if level == "H" else h_cycles * l_cycles


def find_target_bitlinears(model: nn.Module) -> list[TargetInfo]:
    from calm.hrm_text_158.bit_linear import BitLinear

    targets: list[TargetInfo] = []
    for name, module in model.named_modules():
        match = TARGET_RE.match(name)
        if not match:
            continue
        if not isinstance(module, BitLinear):
            raise DiagnosticInvalid(f"{name} is {type(module).__name__}, expected BitLinear")
        targets.append(
            TargetInfo(
                name=name,
                level=match.group("level"),
                layer=int(match.group("layer")),
                proj=match.group("proj"),
                module=module,
            )
        )
    targets.sort(key=lambda t: (t.level, t.layer, t.proj, t.name))
    return targets


def assert_runtime_bitlinear_flags(targets: Iterable[TargetInfo]) -> None:
    bad: list[str] = []
    for target in targets:
        module = target.module
        if getattr(module, "_cached_active", None) is not False:
            bad.append(f"{target.name} _cached_active={getattr(module, '_cached_active', None)!r}")
        if getattr(module, "_cached_weight", None) is not None:
            bad.append(f"{target.name} _cached_weight is not None")
        if getattr(module, "_native_train_active", None) is not False:
            bad.append(f"{target.name} _native_train_active={getattr(module, '_native_train_active', None)!r}")
    if bad:
        raise DiagnosticInvalid("cached/native BitLinear invariant failed: " + "; ".join(bad))


class CreditHookTracker:
    def __init__(self, targets: list[TargetInfo], *, bp_steps: int) -> None:
        self.targets = targets
        self.bp_steps = bp_steps
        self.group_infos = {target.name: _slice_groups(target) for target in targets}
        self.call_counts: dict[str, int] = {}
        self.aggregates: dict[InvocationKey, InvocationAggregate] = {}
        self.schedule_excluded: dict[tuple[InvocationKey, str], ScheduleExcluded] = {}
        self.current_sep_positions: Tensor | None = None
        self.handles: list[Any] = []

    def install(self) -> None:
        for target in self.targets:
            self.handles.append(target.module.register_forward_hook(self._make_forward_hook(target)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def begin_batch(self, sep_positions: Tensor) -> None:
        self.call_counts = {target.name: 0 for target in self.targets}
        self.current_sep_positions = sep_positions.detach()

    def assert_batch_forward_complete(self) -> None:
        missing: list[str] = []
        for target in self.targets:
            expected = expected_forward_calls(target.level)
            seen = self.call_counts.get(target.name, 0)
            if seen != expected:
                missing.append(f"{target.name}: seen {seen}, expected {expected}")
        if missing:
            raise DiagnosticInvalid("unexpected BitLinear forward-call counts: " + "; ".join(missing[:8]))

    def _make_forward_hook(self, target: TargetInfo):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
            if self.current_sep_positions is None:
                raise DiagnosticInvalid("CreditHookTracker.begin_batch was not called")
            if not inputs or not isinstance(inputs[0], Tensor):
                raise DiagnosticInvalid(f"{target.name}: missing tensor input")
            rec_idx = self.call_counts.get(target.name, 0)
            self.call_counts[target.name] = rec_idx + 1
            input_tensor = inputs[0].detach()
            schedule_grad = rec_idx in expected_grad_rec_indices(target.level, bp_steps=self.bp_steps)
            if not output.requires_grad:
                for group in self.group_infos[target.name]:
                    key = InvocationKey(target.level, rec_idx, target.layer, group.group)
                    reason = "schedule_excluded_no_grad" if not schedule_grad else "unexpected_no_grad"
                    self.schedule_excluded[(key, target.name)] = ScheduleExcluded(key, target.name, reason)
                return
            if not schedule_grad:
                raise DiagnosticInvalid(
                    f"{target.name} rec_idx={rec_idx} has grad but is schedule-excluded for bp_steps={self.bp_steps}"
                )

            sep_positions = self.current_sep_positions.detach()

            def grad_hook(grad: Tensor) -> Tensor:
                self._process_grad(target, rec_idx, input_tensor, grad.detach(), sep_positions)
                return grad

            output.register_hook(grad_hook)

        return hook

    def _aggregate_for(self, key: InvocationKey, target: TargetInfo, group: GroupInfo) -> InvocationAggregate:
        agg = self.aggregates.get(key)
        if agg is None:
            agg = InvocationAggregate(
                key=key,
                module_name=target.name,
                group_start=group.start,
                group_end=group.end,
                in_features=int(target.module.weight.shape[1]),
            )
            self.aggregates[key] = agg
        return agg

    def _process_grad(
        self,
        target: TargetInfo,
        rec_idx: int,
        input_tensor: Tensor,
        grad_output: Tensor,
        sep_positions: Tensor,
    ) -> None:
        inp_f = input_tensor.to(torch.float32)
        input_sign = inp_f.sign()
        input_nonzero = input_sign != 0
        B, S, _ = inp_f.shape
        pos = torch.arange(S, device=grad_output.device).unsqueeze(0).expand(B, -1)
        sep = sep_positions.to(device=grad_output.device).view(-1, 1)
        prefix_mask = pos < sep
        response_mask = ~prefix_mask
        for group in self.group_infos[target.name]:
            grad_chunk = grad_output[..., group.start:group.end].to(torch.float32)
            grad_sign = grad_chunk.sign()
            active_out = grad_sign != 0
            credit_f = strict_sign_credit(grad_chunk, inp_f)
            rounded = credit_f.round()
            max_round_err = float((credit_f - rounded).abs().max().item()) if credit_f.numel() else 0.0
            if max_round_err > 1e-3:
                raise DiagnosticInvalid(
                    f"{target.name} rec_idx={rec_idx} {group.group}: integer credit lost exactness "
                    f"(max round err {max_round_err:.6g})"
                )
            weighted_grad = torch.einsum("bso,bsi->oi", grad_chunk, inp_f)
            variant_credits = {
                "strict": rounded.to(torch.int32),
                "pow2_bucket": pow2_bucket_credit(grad_chunk, inp_f),
                "fp16_groupwise": fp16_groupwise_credit(grad_chunk, inp_f),
            }
            active_inputs = torch.einsum(
                "bso,bsi->oi",
                active_out.to(torch.float32),
                input_nonzero.to(torch.float32),
            ) > 0
            active_outputs = active_out.any(dim=(0, 1))

            active_pos = active_out.any(dim=2)
            prefix_active_positions = int((active_pos & prefix_mask).sum().item())
            response_active_positions = int((active_pos & response_mask).sum().item())
            prefix_active_output_elements = int((active_out & prefix_mask.unsqueeze(-1)).sum().item())
            response_active_output_elements = int((active_out & response_mask.unsqueeze(-1)).sum().item())
            input_abs_values = inp_f[input_nonzero].abs()
            grad_abs_values = grad_chunk[grad_chunk != 0].abs()

            key = InvocationKey(target.level, rec_idx, target.layer, group.group)
            self._aggregate_for(key, target, group).accumulate(
                variant_credits=variant_credits,
                weighted_grad=weighted_grad,
                active_inputs=active_inputs,
                active_outputs=active_outputs,
                prefix_active_positions=prefix_active_positions,
                response_active_positions=response_active_positions,
                prefix_active_output_elements=prefix_active_output_elements,
                response_active_output_elements=response_active_output_elements,
                input_abs_values=input_abs_values,
                grad_abs_values=grad_abs_values,
            )


def _bucket_counts(fp_moves: Tensor, int_moves: Tensor, mask: Tensor) -> BucketCounts | None:
    if int(mask.sum().item()) == 0:
        return None
    fp = fp_moves[mask]
    im = int_moves[mask]
    return BucketCounts(
        fp_pos=int((fp > 0).sum().item()),
        fp_neg=int((fp < 0).sum().item()),
        int_pos=int((im > 0).sum().item()),
        int_neg=int((im < 0).sum().item()),
        int_zero=int((im == 0).sum().item()),
    )


def row_q_bucket_counts(fp_moves: Tensor, int_moves: Tensor, q_levels: Tensor, denom_mask: Tensor) -> list[BucketCounts]:
    buckets: list[BucketCounts] = []
    rows = fp_moves.shape[0]
    for row in range(rows):
        row_mask = denom_mask[row]
        if not bool(row_mask.any().item()):
            continue
        for q in (-1, 0, 1):
            mask = row_mask & (q_levels[row] == q)
            bucket = _bucket_counts(fp_moves[row], int_moves[row], mask)
            if bucket is not None and bucket.total > 0:
                buckets.append(bucket)
    return buckets


def _summarize_null_scores(
    scores: np.ndarray,
    *,
    backend: str,
    timing_seconds: dict[str, float] | None = None,
    aggregation_device: str | None = None,
) -> dict[str, float | str | dict[str, float] | None]:
    if scores.size == 0:
        out: dict[str, float | str | dict[str, float] | None] = {"mean": 0.0, "p95": 0.0, "p99": 0.0}
    else:
        out = {
            "mean": float(scores.mean()),
            "p95": float(np.quantile(scores, 0.95, method="higher")),
            "p99": float(np.quantile(scores, 0.99, method="higher")),
        }
    out["backend"] = backend
    if aggregation_device is not None:
        out["aggregation_device"] = aggregation_device
    if timing_seconds is not None:
        out["timing_seconds"] = timing_seconds
    return out


def _simulate_permutation_null_cpu_locked(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
    profile: bool = False,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    if not buckets:
        timing = {"total": time.perf_counter() - total_start} if profile else None
        return _summarize_null_scores(
            np.asarray([], dtype=np.float64),
            backend=NULL_BACKEND_CPU_LOCKED,
            timing_seconds=timing,
        )
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(permutations):
        matches = 0
        total = 0
        for b in buckets:
            n = b.total
            if n <= 0:
                continue
            k_pos = min(b.int_pos, n)
            x_pos_match = int(rng.hypergeometric(b.fp_pos, n - b.fp_pos, k_pos))
            fp_neg_consumed_by_pos = k_pos - x_pos_match
            remaining_n = n - k_pos
            remaining_fp_neg = max(0, b.fp_neg - fp_neg_consumed_by_pos)
            k_neg = min(b.int_neg, remaining_n)
            x_neg_match = 0
            if remaining_n > 0 and k_neg > 0:
                x_neg_match = int(
                    rng.hypergeometric(remaining_fp_neg, remaining_n - remaining_fp_neg, k_neg)
                )
            matches += x_pos_match + x_neg_match
            total += n
        scores.append(matches / total if total else 0.0)
    timing = {"total": time.perf_counter() - total_start} if profile else None
    return _summarize_null_scores(
        np.asarray(scores, dtype=np.float64),
        backend=NULL_BACKEND_CPU_LOCKED,
        timing_seconds=timing,
    )


def _sample_bucket_match_matrix_cpu_locked(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay the locked numpy hypergeometric draw order into a match matrix."""
    start = time.perf_counter()
    totals = np.asarray([b.total for b in buckets], dtype=np.int64)
    matches = np.zeros((permutations, len(buckets)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for perm_idx in range(permutations):
        for bucket_idx, b in enumerate(buckets):
            n = b.total
            if n <= 0:
                continue
            k_pos = min(b.int_pos, n)
            x_pos_match = int(rng.hypergeometric(b.fp_pos, n - b.fp_pos, k_pos))
            fp_neg_consumed_by_pos = k_pos - x_pos_match
            remaining_n = n - k_pos
            remaining_fp_neg = max(0, b.fp_neg - fp_neg_consumed_by_pos)
            k_neg = min(b.int_neg, remaining_n)
            x_neg_match = 0
            if remaining_n > 0 and k_neg > 0:
                x_neg_match = int(
                    rng.hypergeometric(remaining_fp_neg, remaining_n - remaining_fp_neg, k_neg)
                )
            matches[perm_idx, bucket_idx] = x_pos_match + x_neg_match
    return matches, totals, time.perf_counter() - start


def _torch_device_name(device_name: str | None) -> str:
    if device_name is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise DiagnosticInvalid("requested cuda null aggregation but torch.cuda.is_available() is false")
    return device_name


def _sync_for_timing(device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()


def _bucket_totals_metadata(buckets: list[BucketCounts]) -> dict[str, Any]:
    totals = np.asarray([b.total for b in buckets], dtype=np.int64)
    return {
        "bucket_count": int(totals.size),
        "sum_total": int(totals.sum()) if totals.size else 0,
        "ordered_totals_sha256": sha256_bytes(totals.tobytes()),
    }


def _new_support_policy_stats() -> dict[str, Any]:
    return {
        "distribution_count": 0,
        "full_support_bucket_count": 0,
        "mass_trimmed_bucket_count": 0,
        "max_support_size": 0,
        "max_omitted_mass": 0.0,
    }


def _record_support_policy(
    stats: dict[str, Any],
    *,
    legal_size: Tensor,
    support_size: Tensor,
    omitted_mass: Tensor,
) -> None:
    legal_cpu = legal_size.detach().cpu().to(torch.int64)
    support_cpu = support_size.detach().cpu().to(torch.int64)
    omitted_cpu = omitted_mass.detach().cpu().to(torch.float64)
    if legal_cpu.numel() == 0:
        return
    stats["distribution_count"] += int(legal_cpu.numel())
    stats["full_support_bucket_count"] += int((support_cpu >= legal_cpu).sum().item())
    stats["mass_trimmed_bucket_count"] += int((support_cpu < legal_cpu).sum().item())
    stats["max_support_size"] = max(stats["max_support_size"], int(support_cpu.max().item()))
    stats["max_omitted_mass"] = max(stats["max_omitted_mass"], float(omitted_cpu.max().item()))


def _torch_logcomb(n: Tensor, k: Tensor) -> Tensor:
    n = n.to(dtype=torch.float64)
    k = k.to(dtype=torch.float64)
    return torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)


def _hypergeom_logpmf_torch(successes: Tensor, failures: Tensor, draws: Tensor, support: Tensor) -> Tensor:
    total = successes + failures
    return (
        _torch_logcomb(successes, support)
        + _torch_logcomb(failures, draws - support)
        - _torch_logcomb(total, draws)
    )


def _hypergeom_initial_center_half(successes: Tensor, failures: Tensor, draws: Tensor) -> tuple[Tensor, Tensor]:
    total = (successes + failures).to(torch.float64)
    draws_f = draws.to(torch.float64)
    successes_f = successes.to(torch.float64)
    probs = torch.where(total > 0, successes_f / total, torch.zeros_like(total))
    mean = draws_f * probs
    finite_population = torch.where(
        total > 1,
        (total - draws_f).clamp_min(0.0) / (total - 1.0),
        torch.zeros_like(total),
    )
    var = draws_f * probs * (1.0 - probs) * finite_population
    std = torch.sqrt(var.clamp_min(0.0))
    center = torch.round(mean).to(torch.int64)
    half = torch.ceil(torch.maximum(std * 4.0, torch.full_like(std, 8.0))).to(torch.int64)
    return center, half


def _select_hypergeom_window_scalar(
    *,
    successes: int,
    failures: int,
    draws: int,
    device_name: str,
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
    stats: dict[str, Any] | None = None,
) -> tuple[Tensor, Tensor, float]:
    if draws <= 0 or successes <= 0:
        support = torch.zeros(1, dtype=torch.int64, device=device_name)
        probs = torch.ones(1, dtype=torch.float64, device=device_name)
        if stats is not None:
            one = torch.ones(1, dtype=torch.int64, device=device_name)
            zero = torch.zeros(1, dtype=torch.float64, device=device_name)
            _record_support_policy(stats, legal_size=one, support_size=one, omitted_mass=zero)
        return support, probs, 0.0
    if failures <= 0:
        support = torch.tensor([draws], dtype=torch.int64, device=device_name)
        probs = torch.ones(1, dtype=torch.float64, device=device_name)
        if stats is not None:
            one = torch.ones(1, dtype=torch.int64, device=device_name)
            zero = torch.zeros(1, dtype=torch.float64, device=device_name)
            _record_support_policy(stats, legal_size=one, support_size=one, omitted_mass=zero)
        return support, probs, 0.0

    lo = max(0, draws - failures)
    hi = min(draws, successes)
    if hi < lo:
        raise DiagnosticInvalid(
            f"invalid hypergeometric support successes={successes} failures={failures} draws={draws}"
        )
    legal_size = hi - lo + 1
    if legal_size <= support_guard:
        low, high = lo, hi
    else:
        s_t = torch.tensor([successes], dtype=torch.int64, device=device_name)
        f_t = torch.tensor([failures], dtype=torch.int64, device=device_name)
        d_t = torch.tensor([draws], dtype=torch.int64, device=device_name)
        center_t, half_t = _hypergeom_initial_center_half(s_t, f_t, d_t)
        center = max(lo, min(hi, int(center_t.item())))
        half = min(int(half_t.item()), max(1, (support_guard - 1) // 2))
        while True:
            low = max(lo, center - half)
            high = min(hi, center + half)
            support_size = high - low + 1
            if support_size > support_guard:
                raise DiagnosticInvalid(
                    f"gpu PMF support guard exceeded support_size={support_size} guard={support_guard}"
                )
            support = torch.arange(low, high + 1, dtype=torch.int64, device=device_name)
            logp = _hypergeom_logpmf_torch(s_t, f_t, d_t, support)
            log_mass = torch.logsumexp(logp, dim=0)
            omitted = max(0.0, 1.0 - float(torch.exp(log_mass).item()))
            if omitted <= captured_mass_eps or (low == lo and high == hi):
                break
            if support_size >= support_guard:
                raise DiagnosticInvalid(
                    "gpu PMF captured-mass window could not meet epsilon "
                    f"omitted={omitted:.6g} eps={captured_mass_eps} guard={support_guard}"
                )
            half = min(max(half * 2 + 1, half + 8), max(1, (support_guard - 1) // 2))

    support = torch.arange(low, high + 1, dtype=torch.int64, device=device_name)
    s_t = torch.tensor(successes, dtype=torch.int64, device=device_name)
    f_t = torch.tensor(failures, dtype=torch.int64, device=device_name)
    d_t = torch.tensor(draws, dtype=torch.int64, device=device_name)
    logp = _hypergeom_logpmf_torch(s_t, f_t, d_t, support)
    log_mass = torch.logsumexp(logp, dim=0)
    probs = torch.exp(logp - log_mass)
    omitted = 0.0 if support.numel() == legal_size else max(0.0, 1.0 - float(torch.exp(log_mass).item()))
    if omitted > captured_mass_eps:
        raise DiagnosticInvalid(
            f"gpu PMF omitted mass {omitted:.6g} exceeds epsilon {captured_mass_eps}"
        )
    if stats is not None:
        _record_support_policy(
            stats,
            legal_size=torch.tensor([legal_size], dtype=torch.int64, device=device_name),
            support_size=torch.tensor([support.numel()], dtype=torch.int64, device=device_name),
            omitted_mass=torch.tensor([omitted], dtype=torch.float64, device=device_name),
        )
    return support, probs, omitted


def _sample_hypergeom_scalar_gpu(
    *,
    successes: int,
    failures: int,
    draws: int,
    sample_count: int,
    device_name: str,
    generator: torch.Generator,
    stats: dict[str, Any],
) -> Tensor:
    support, probs, _ = _select_hypergeom_window_scalar(
        successes=successes,
        failures=failures,
        draws=draws,
        device_name=device_name,
        stats=stats,
    )
    cdf = torch.cumsum(probs, dim=0)
    u = torch.rand(sample_count, dtype=torch.float64, device=device_name, generator=generator)
    idx = torch.clamp((cdf.unsqueeze(0) < u.unsqueeze(1)).sum(dim=1), max=support.numel() - 1)
    return support[idx].to(torch.int64)


def _sample_hypergeom_batched_gpu(
    *,
    successes: Tensor,
    failures: Tensor,
    draws: Tensor,
    device_name: str,
    generator: torch.Generator,
    stats: dict[str, Any],
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
) -> Tensor:
    successes = successes.to(device=device_name, dtype=torch.int64)
    failures = failures.to(device=device_name, dtype=torch.int64)
    draws = draws.to(device=device_name, dtype=torch.int64)
    if successes.numel() == 0:
        return successes

    legal_lo = torch.maximum(torch.zeros_like(draws), draws - failures)
    legal_hi = torch.minimum(draws, successes)
    if bool((legal_hi < legal_lo).any().item()):
        raise DiagnosticInvalid("invalid batched hypergeometric support")
    legal_size = legal_hi - legal_lo + 1
    deterministic = legal_size <= 1
    out = legal_lo.clone()
    active = ~deterministic
    if not bool(active.any().item()):
        _record_support_policy(
            stats,
            legal_size=legal_size,
            support_size=torch.ones_like(legal_size),
            omitted_mass=torch.zeros_like(legal_size, dtype=torch.float64),
        )
        return out

    center, half = _hypergeom_initial_center_half(successes, failures, draws)
    center = torch.minimum(torch.maximum(center, legal_lo), legal_hi)
    max_half = max(1, (support_guard - 1) // 2)
    half = torch.minimum(half, torch.full_like(half, max_half))
    full_support = legal_size <= support_guard
    half = torch.where(full_support, torch.maximum(center - legal_lo, legal_hi - center), half)

    while True:
        low = torch.maximum(legal_lo, center - half)
        high = torch.minimum(legal_hi, center + half)
        support_size = high - low + 1
        max_width = int(support_size[active].max().item())
        if max_width > support_guard:
            raise DiagnosticInvalid(
                f"gpu PMF batched support guard exceeded support_size={max_width} guard={support_guard}"
            )
        offsets = torch.arange(max_width, dtype=torch.int64, device=device_name).unsqueeze(0)
        support = low.unsqueeze(1) + offsets
        valid = offsets < support_size.unsqueeze(1)
        safe_support = torch.where(valid, support, low.unsqueeze(1))
        logp = _hypergeom_logpmf_torch(
            successes.unsqueeze(1),
            failures.unsqueeze(1),
            draws.unsqueeze(1),
            safe_support,
        )
        logp = torch.where(valid, logp, torch.full_like(logp, -torch.inf))
        log_mass = torch.logsumexp(logp, dim=1)
        omitted = (1.0 - torch.exp(log_mass)).clamp_min(0.0)
        covers_full = (low == legal_lo) & (high == legal_hi)
        done = deterministic | covers_full | (omitted <= captured_mass_eps)
        if bool((done | ~active).all().item()):
            break
        stuck = active & ~done & (support_size >= support_guard)
        if bool(stuck.any().item()):
            worst = float(omitted[stuck].max().item())
            raise DiagnosticInvalid(
                "gpu PMF captured-mass batched window could not meet epsilon "
                f"omitted={worst:.6g} eps={captured_mass_eps} guard={support_guard}"
            )
        half = torch.where(done, half, torch.minimum(half * 2 + 1, torch.full_like(half, max_half)))

    _record_support_policy(stats, legal_size=legal_size, support_size=support_size, omitted_mass=omitted)
    probs = torch.exp(logp - log_mass.unsqueeze(1))
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    cdf = torch.cumsum(probs, dim=1)
    u = torch.rand(successes.shape[0], 1, dtype=torch.float64, device=device_name, generator=generator)
    idx = torch.clamp((cdf < u).sum(dim=1), max=max_width - 1)
    sampled = support.gather(1, idx.unsqueeze(1)).squeeze(1).to(torch.int64)
    out = torch.where(deterministic, out, sampled)
    return out


def _simulate_permutation_null_gpu_native_counts_pmf(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
    aggregation_device: str | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    device_name = _torch_device_name(aggregation_device)
    bucket_metadata = _bucket_totals_metadata(buckets)
    if not buckets:
        timing = {"total": time.perf_counter() - total_start} if profile else None
        out = _summarize_null_scores(
            np.asarray([], dtype=np.float64),
            backend=NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
            timing_seconds=timing,
            aggregation_device=device_name,
        )
        out["input_bucket_metadata"] = bucket_metadata
        out["candidate_batching_metadata"] = bucket_metadata
        out["support_policy"] = _new_support_policy_stats()
        return out

    generator = torch.Generator(device=device_name)
    generator.manual_seed(seed)
    support_stats = _new_support_policy_stats()
    _sync_for_timing(device_name)
    sample_start = time.perf_counter()
    matches = torch.zeros(permutations, dtype=torch.int64, device=device_name)
    total = int(sum(b.total for b in buckets))
    for b in buckets:
        n = b.total
        if n <= 0:
            continue
        k_pos = min(b.int_pos, n)
        x_pos = _sample_hypergeom_scalar_gpu(
            successes=b.fp_pos,
            failures=n - b.fp_pos,
            draws=k_pos,
            sample_count=permutations,
            device_name=device_name,
            generator=generator,
            stats=support_stats,
        )
        fp_neg_consumed_by_pos = k_pos - x_pos
        remaining_n = n - k_pos
        remaining_fp_neg = (torch.full_like(x_pos, b.fp_neg) - fp_neg_consumed_by_pos).clamp_min(0)
        k_neg = min(b.int_neg, remaining_n)
        if remaining_n > 0 and k_neg > 0:
            x_neg = _sample_hypergeom_batched_gpu(
                successes=remaining_fp_neg,
                failures=torch.full_like(remaining_fp_neg, remaining_n) - remaining_fp_neg,
                draws=torch.full_like(remaining_fp_neg, k_neg),
                device_name=device_name,
                generator=generator,
                stats=support_stats,
            )
        else:
            x_neg = torch.zeros_like(x_pos)
        matches += x_pos + x_neg
    _sync_for_timing(device_name)
    gpu_sampler_seconds = time.perf_counter() - sample_start
    scores = matches.detach().cpu().numpy().astype(np.float64) / float(total) if total else np.zeros(permutations, dtype=np.float64)
    timing = None
    if profile:
        timing = {
            "gpu_sampler": gpu_sampler_seconds,
            "aggregation": 0.0,
            "total": time.perf_counter() - total_start,
        }
    out = _summarize_null_scores(
        scores,
        backend=NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
        timing_seconds=timing,
        aggregation_device=device_name,
    )
    out["input_bucket_metadata"] = bucket_metadata
    out["candidate_batching_metadata"] = bucket_metadata
    out["support_policy"] = support_stats
    return out


def _hypergeom_logpmf_float(successes: int, failures: int, draws: int, x: int) -> float:
    total = successes + failures
    if x < max(0, draws - failures) or x > min(draws, successes):
        return -math.inf
    return (
        math.lgamma(successes + 1)
        - math.lgamma(x + 1)
        - math.lgamma(successes - x + 1)
        + math.lgamma(failures + 1)
        - math.lgamma(draws - x + 1)
        - math.lgamma(failures - (draws - x) + 1)
        - math.lgamma(total + 1)
        + math.lgamma(draws + 1)
        + math.lgamma(total - draws + 1)
    )


def _hypergeom_support_bounds(successes: int, failures: int, draws: int) -> tuple[int, int]:
    lo = max(0, draws - failures)
    hi = min(draws, successes)
    if hi < lo:
        raise DiagnosticInvalid(
            f"invalid hypergeometric support successes={successes} failures={failures} draws={draws}"
        )
    return lo, hi


def _logsumexp_float(values: list[float]) -> float:
    if not values:
        return -math.inf
    peak = max(values)
    if not math.isfinite(peak):
        return peak
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def _reference_initial_center_half(successes: int, failures: int, draws: int) -> tuple[int, int]:
    total = successes + failures
    if total <= 0:
        return 0, 1
    prob = successes / total
    mean = draws * prob
    finite_population = ((total - draws) / (total - 1)) if total > 1 else 0.0
    var = draws * prob * (1.0 - prob) * max(0.0, finite_population)
    std = math.sqrt(max(0.0, var))
    return int(round(mean)), int(math.ceil(max(std * 4.0, 8.0)))


def _new_scipy_cross_check_summary() -> dict[str, Any]:
    return {
        "library": "scipy.stats.hypergeom.pmf",
        "available": None,
        "max_points_per_distribution": GPU_NATIVE_REFERENCE_SCIPY_MAX_POINTS,
        "checked_distribution_count": 0,
        "skipped_distribution_count": 0,
        "max_abs_delta": 0.0,
        "failures": [],
        "fallback_reasons": [],
    }


def _record_scipy_cross_check(
    summary: dict[str, Any],
    *,
    successes: int,
    failures: int,
    draws: int,
    support: np.ndarray,
    label: str,
) -> None:
    if support.size == 0:
        return
    try:
        from scipy.stats import hypergeom  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on scipy-missing hosts.
        summary["available"] = False
        reason = f"scipy_unavailable:{type(exc).__name__}:{exc}"
        if reason not in summary["fallback_reasons"]:
            summary["fallback_reasons"].append(reason)
        summary["skipped_distribution_count"] += 1
        return

    summary["available"] = True
    support_i = support.astype(np.int64, copy=False)
    if support_i.size > GPU_NATIVE_REFERENCE_SCIPY_MAX_POINTS:
        idx = np.linspace(0, support_i.size - 1, GPU_NATIVE_REFERENCE_SCIPY_MAX_POINTS, dtype=np.int64)
        check_support = np.unique(support_i[idx])
        summary["fallback_reasons"].append(
            f"{label}:spot_checked_{check_support.size}_of_{support_i.size}_support_points"
        )
    else:
        check_support = support_i

    scipy_pmf = np.asarray(
        hypergeom.pmf(check_support, successes + failures, successes, draws),
        dtype=np.float64,
    )
    reference_pmf = np.asarray(
        [math.exp(_hypergeom_logpmf_float(successes, failures, draws, int(x))) for x in check_support],
        dtype=np.float64,
    )
    delta = np.abs(scipy_pmf - reference_pmf)
    max_abs = float(delta.max()) if delta.size else 0.0
    summary["checked_distribution_count"] += 1
    summary["max_abs_delta"] = max(float(summary["max_abs_delta"]), max_abs)
    if not np.all(np.isfinite(scipy_pmf)) or max_abs > 1e-10:
        summary["failures"].append({"label": label, "max_abs_delta": max_abs})


def _hypergeom_pmf_scipy_vectorized(successes: int, failures: int, draws: int, support: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import hypergeom  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on scipy-missing hosts.
        raise DiagnosticInvalid(f"scipy hypergeom reference unavailable: {type(exc).__name__}: {exc}") from exc
    return np.asarray(
        hypergeom.pmf(support.astype(np.int64, copy=False), successes + failures, successes, draws),
        dtype=np.float64,
    )


def _select_hypergeom_window_reference(
    *,
    successes: int,
    failures: int,
    draws: int,
    support_guard: int,
    captured_mass_eps: float,
    scipy_summary: dict[str, Any] | None,
    scipy_label: str | None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if draws <= 0 or successes <= 0:
        support = np.asarray([0], dtype=np.int64)
        probs = np.asarray([1.0], dtype=np.float64)
        if scipy_summary is not None and scipy_label is not None:
            _record_scipy_cross_check(
                scipy_summary,
                successes=successes,
                failures=failures,
                draws=draws,
                support=support,
                label=scipy_label,
            )
        return support, probs, 0.0, 1
    if failures <= 0:
        support = np.asarray([draws], dtype=np.int64)
        probs = np.asarray([1.0], dtype=np.float64)
        if scipy_summary is not None and scipy_label is not None:
            _record_scipy_cross_check(
                scipy_summary,
                successes=successes,
                failures=failures,
                draws=draws,
                support=support,
                label=scipy_label,
            )
        return support, probs, 0.0, 1

    lo, hi = _hypergeom_support_bounds(successes, failures, draws)
    legal_size = hi - lo + 1
    if legal_size <= support_guard:
        low, high = lo, hi
    else:
        center, half = _reference_initial_center_half(successes, failures, draws)
        center = max(lo, min(hi, center))
        half = min(half, max(1, (support_guard - 1) // 2))
        while True:
            low = max(lo, center - half)
            high = min(hi, center + half)
            support_size = high - low + 1
            if support_size > support_guard:
                raise DiagnosticInvalid(
                    f"reference PMF support guard exceeded support_size={support_size} guard={support_guard}"
                )
            support = np.arange(low, high + 1, dtype=np.int64)
            raw = _hypergeom_pmf_scipy_vectorized(successes, failures, draws, support)
            mass = float(raw.sum())
            if mass <= 0.0 or not math.isfinite(mass):
                raise DiagnosticInvalid(
                    "reference scipy PMF produced invalid captured mass "
                    f"mass={mass} successes={successes} failures={failures} draws={draws}"
                )
            omitted = max(0.0, 1.0 - mass)
            if omitted <= captured_mass_eps or (low == lo and high == hi):
                break
            if support_size >= support_guard:
                raise DiagnosticInvalid(
                    "reference PMF captured-mass window could not meet epsilon "
                    f"omitted={omitted:.6g} eps={captured_mass_eps} guard={support_guard}"
                )
            half = min(max(half * 2 + 1, half + 8), max(1, (support_guard - 1) // 2))

    support = np.arange(low, high + 1, dtype=np.int64)
    raw = _hypergeom_pmf_scipy_vectorized(successes, failures, draws, support)
    mass = float(raw.sum())
    if mass <= 0.0 or not math.isfinite(mass):
        raise DiagnosticInvalid(
            "reference scipy PMF produced invalid captured mass "
            f"mass={mass} successes={successes} failures={failures} draws={draws}"
        )
    probs = (raw / mass).astype(np.float64, copy=False)
    omitted = 0.0 if support.size == legal_size else max(0.0, 1.0 - mass)
    if omitted > captured_mass_eps:
        raise DiagnosticInvalid(
            f"reference PMF omitted mass {omitted:.6g} exceeds epsilon {captured_mass_eps}"
        )
    if scipy_summary is not None and scipy_label is not None:
        _record_scipy_cross_check(
            scipy_summary,
            successes=successes,
            failures=failures,
            draws=draws,
            support=support,
            label=scipy_label,
        )
    return support, probs, omitted, legal_size


def _conditional_cross_check_values(x_pos_support: np.ndarray) -> set[int]:
    values = [int(x) for x in x_pos_support.tolist()]
    if len(values) <= 16:
        return set(values)
    return {values[0], values[len(values) // 2], values[-1]}


def _normalize_sparse_pmf(pmf: dict[int, float]) -> dict[int, float]:
    total = sum(pmf.values())
    if total <= 0.0:
        raise DiagnosticInvalid("sparse PMF has zero mass")
    return {k: float(v / total) for k, v in pmf.items() if v > 0.0}


def _add_dense_window(
    acc: np.ndarray | None,
    offset: int | None,
    keys: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, int]:
    if keys.size == 0:
        if acc is None or offset is None:
            return np.zeros(0, dtype=np.float64), 0
        return acc, offset
    low = int(keys.min())
    high = int(keys.max())
    if acc is None or offset is None or acc.size == 0:
        new_acc = np.zeros(high - low + 1, dtype=np.float64)
        np.add.at(new_acc, keys.astype(np.int64, copy=False) - low, values)
        return new_acc, low

    old_low = offset
    old_high = offset + acc.size - 1
    if low < old_low or high > old_high:
        new_low = min(low, old_low)
        new_high = max(high, old_high)
        new_acc = np.zeros(new_high - new_low + 1, dtype=np.float64)
        new_acc[old_low - new_low : old_high - new_low + 1] = acc
        acc = new_acc
        offset = new_low
    np.add.at(acc, keys.astype(np.int64, copy=False) - offset, values)
    return acc, offset


def _dense_window_to_sparse(acc: np.ndarray, offset: int) -> dict[int, float]:
    if acc.size == 0:
        raise DiagnosticInvalid("dense PMF window has zero size")
    idx = np.flatnonzero(acc > 0.0)
    if idx.size == 0:
        raise DiagnosticInvalid("dense PMF window has zero positive mass")
    keys = idx + offset
    values = acc[idx]
    return {int(k): float(v) for k, v in zip(keys.tolist(), values.tolist(), strict=True)}


def _reference_bounded_sample_seed(bucket: BucketCounts) -> int:
    seed = GPU_NATIVE_BOUNDED_SAMPLE_SEED
    for value in (bucket.fp_pos, bucket.fp_neg, bucket.int_pos, bucket.int_neg, bucket.int_zero):
        seed = (seed * 1_315_423_911 + int(value) + 0x9E3779B9) & 0xFFFFFFFF
    return int(seed)


def _dkw_epsilon(sample_count: int, confidence: float) -> float:
    if sample_count <= 0:
        raise DiagnosticInvalid("bounded sampled reference sample_count must be positive")
    if not 0.0 < confidence < 1.0:
        raise DiagnosticInvalid("bounded sampled reference confidence must be in (0, 1)")
    return math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * sample_count))


def _reference_joint_work_estimate(
    bucket: BucketCounts,
    x_pos_support: np.ndarray,
    *,
    k_pos: int,
    k_neg: int,
    remaining_n: int,
    support_guard: int,
) -> dict[str, Any]:
    if x_pos_support.size == 0:
        return {
            "x_pos_window_size": 0,
            "max_conditional_window_upper_bound": 0,
            "sum_conditional_window_upper_bound": 0,
            "estimated_joint_cells": 0,
            "budget": GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
        }
    xp = x_pos_support.astype(np.int64, copy=False)
    successes = np.maximum(0, int(bucket.fp_neg) - (int(k_pos) - xp))
    failures = int(remaining_n) - successes
    lo = np.maximum(0, int(k_neg) - failures)
    hi = np.minimum(int(k_neg), successes)
    if bool(np.any(hi < lo)):
        raise DiagnosticInvalid(f"invalid conditional support estimate for bucket={bucket}")
    legal_size = hi - lo + 1
    window_upper = np.minimum(legal_size, int(support_guard)).astype(np.int64, copy=False)
    return {
        "x_pos_window_size": int(xp.size),
        "max_conditional_window_upper_bound": int(window_upper.max(initial=0)),
        "sum_conditional_window_upper_bound": int(window_upper.sum(dtype=np.int64)),
        "estimated_joint_cells": int(window_upper.sum(dtype=np.int64)),
        "budget": GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
    }


def _reference_initial_center_half_array(
    successes: np.ndarray,
    failures: np.ndarray,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = successes.astype(np.float64, copy=False) + failures.astype(np.float64, copy=False)
    probs = np.divide(
        successes.astype(np.float64, copy=False),
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )
    draws_f = float(draws)
    mean = draws_f * probs
    finite_population = np.divide(
        np.maximum(total - draws_f, 0.0),
        total - 1.0,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 1.0,
    )
    var = draws_f * probs * (1.0 - probs) * finite_population
    center = np.rint(mean).astype(np.int64)
    half = np.ceil(np.maximum(np.sqrt(np.maximum(var, 0.0)) * 4.0, 8.0)).astype(np.int64)
    return center, half


def _select_hypergeom_windows_reference_batched(
    *,
    successes: np.ndarray,
    failures: np.ndarray,
    draws: int,
    support_guard: int,
    captured_mass_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.stats import hypergeom  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only on scipy-missing hosts.
        raise DiagnosticInvalid(f"scipy hypergeom reference unavailable: {type(exc).__name__}: {exc}") from exc

    successes = successes.astype(np.int64, copy=False)
    failures = failures.astype(np.int64, copy=False)
    if successes.size == 0:
        empty_i = np.zeros((0, 0), dtype=np.int64)
        empty_f = np.zeros((0, 0), dtype=np.float64)
        empty_b = np.zeros((0, 0), dtype=bool)
        empty_v = np.zeros(0, dtype=np.float64)
        empty_s = np.zeros(0, dtype=np.int64)
        return empty_i, empty_f, empty_b, empty_v, empty_s, empty_s

    lo = np.maximum(0, int(draws) - failures)
    hi = np.minimum(int(draws), successes)
    if bool(np.any(hi < lo)):
        raise DiagnosticInvalid("invalid batched reference hypergeometric support")
    legal_size = (hi - lo + 1).astype(np.int64, copy=False)
    center, half = _reference_initial_center_half_array(successes, failures, draws)
    center = np.minimum(np.maximum(center, lo), hi)
    max_half = max(1, (int(support_guard) - 1) // 2)
    half = np.minimum(half, max_half)
    full_support = legal_size <= int(support_guard)
    half = np.where(full_support, np.maximum(center - lo, hi - center), half)

    while True:
        low = np.maximum(lo, center - half)
        high = np.minimum(hi, center + half)
        support_size = (high - low + 1).astype(np.int64, copy=False)
        max_width = int(support_size.max(initial=0))
        if max_width > int(support_guard):
            raise DiagnosticInvalid(
                f"reference PMF batched support guard exceeded support_size={max_width} guard={support_guard}"
            )
        offsets = np.arange(max_width, dtype=np.int64)
        support = low[:, None] + offsets[None, :]
        valid = offsets[None, :] < support_size[:, None]
        safe_support = np.where(valid, support, low[:, None])
        raw = np.asarray(
            hypergeom.pmf(
                safe_support,
                (successes + failures)[:, None],
                successes[:, None],
                int(draws),
            ),
            dtype=np.float64,
        )
        raw = np.where(valid, raw, 0.0)
        mass = raw.sum(axis=1)
        if bool(np.any((mass <= 0.0) | ~np.isfinite(mass))):
            raise DiagnosticInvalid("reference scipy PMF produced invalid batched captured mass")
        omitted = np.maximum(0.0, 1.0 - mass)
        covers_full = (low == lo) & (high == hi)
        done = covers_full | (omitted <= captured_mass_eps)
        if bool(np.all(done)):
            probs = raw / mass[:, None]
            return support, probs, valid, omitted.astype(np.float64), legal_size, support_size
        stuck = (~done) & (support_size >= int(support_guard))
        if bool(np.any(stuck)):
            worst = float(omitted[stuck].max(initial=0.0))
            raise DiagnosticInvalid(
                "reference PMF captured-mass batched window could not meet epsilon "
                f"omitted={worst:.6g} eps={captured_mass_eps} guard={support_guard}"
            )
        half = np.where(done, half, np.minimum(half * 2 + 1, max_half))


def _joint_match_pmf_reference_bounded_sampled_sparse(
    bucket: BucketCounts,
    *,
    joint_work: dict[str, Any],
    sample_count: int = GPU_NATIVE_BOUNDED_SAMPLE_COUNT,
    confidence: float = GPU_NATIVE_BOUNDED_SAMPLE_CONFIDENCE,
    seed_salt: int = 0,
    reference_mode: str = "bounded_sampled",
    fallback_reason: str = "joint_work_budget_exceeded",
) -> dict[str, Any]:
    n = bucket.total
    if n <= 0:
        return {
            "pmf": {0: 1.0},
            "omitted_mass_bound": 0.0,
            "max_support_size": 1,
            "materialized_window_size": 1,
            "max_stage_window_size": 1,
            "reference_mode": reference_mode,
            "fallback_flag": True,
            "fallback_reason": fallback_reason,
            "joint_work": joint_work,
            "bounded_sample": {
                "sample_count": sample_count,
                "confidence": confidence,
                "dkw_epsilon": _dkw_epsilon(sample_count, confidence),
                "sampling_cdf_bound": GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND,
            },
            "scipy_cross_check": _new_scipy_cross_check_summary(),
        }

    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    seed = (_reference_bounded_sample_seed(bucket) + int(seed_salt)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    x_pos = rng.hypergeometric(
        ngood=int(bucket.fp_pos),
        nbad=max(0, int(n - bucket.fp_pos)),
        nsample=int(k_pos),
        size=int(sample_count),
    ).astype(np.int64, copy=False)
    remaining_fp_neg = np.maximum(0, int(bucket.fp_neg) - (int(k_pos) - x_pos)).astype(np.int64, copy=False)
    remaining_failures = np.maximum(0, int(remaining_n) - remaining_fp_neg).astype(np.int64, copy=False)
    x_neg = rng.hypergeometric(
        ngood=remaining_fp_neg,
        nbad=remaining_failures,
        nsample=int(k_neg),
    ).astype(np.int64, copy=False)
    totals = x_pos + x_neg
    offset = int(totals.min(initial=0))
    counts = np.bincount((totals - offset).astype(np.int64, copy=False))
    probs = counts.astype(np.float64) / float(sample_count)
    pmf = {int(idx + offset): float(value) for idx, value in enumerate(probs.tolist()) if value > 0.0}
    scipy_summary = _new_scipy_cross_check_summary()
    scipy_summary["available"] = True
    scipy_summary["fallback_reasons"].append("bounded_sampled_reference_no_scipy_distribution_cross_check")
    dkw = _dkw_epsilon(sample_count, confidence)
    return {
        "pmf": pmf,
        "omitted_mass_bound": 0.0,
        "max_support_size": _bucket_joint_support_span(bucket),
        "materialized_window_size": int(len(probs)),
        "max_stage_window_size": int(len(probs)),
        "reference_mode": reference_mode,
        "fallback_flag": True,
        "fallback_reason": fallback_reason,
        "joint_work": joint_work,
        "bounded_sample": {
            "sample_count": int(sample_count),
            "confidence": float(confidence),
            "dkw_epsilon": float(dkw),
            "sampling_cdf_bound": GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND,
            "seed": seed,
        },
        "scipy_cross_check": scipy_summary,
    }


def _joint_match_pmf_reference_scalar_loop_sparse(
    bucket: BucketCounts,
    *,
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
) -> dict[str, Any]:
    n = bucket.total
    scipy_summary = _new_scipy_cross_check_summary()
    if n <= 0:
        return {
            "pmf": {0: 1.0},
            "omitted_mass_bound": 0.0,
            "max_support_size": 1,
            "materialized_window_size": 1,
            "max_stage_window_size": 1,
            "reference_mode": "scalar_loop_exact",
            "fallback_flag": False,
            "scipy_cross_check": scipy_summary,
        }

    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    x_pos_support, x_pos_probs, x_pos_omitted, x_pos_legal = _select_hypergeom_window_reference(
        successes=bucket.fp_pos,
        failures=n - bucket.fp_pos,
        draws=k_pos,
        support_guard=support_guard,
        captured_mass_eps=captured_mass_eps,
        scipy_summary=scipy_summary,
        scipy_label="x_pos",
    )
    check_conditionals = _conditional_cross_check_values(x_pos_support)
    skipped_conditionals = max(0, int(x_pos_support.size) - len(check_conditionals))
    if skipped_conditionals:
        scipy_summary["skipped_distribution_count"] += skipped_conditionals
        scipy_summary["fallback_reasons"].append(
            f"x_neg_conditionals_sampled_{len(check_conditionals)}_of_{int(x_pos_support.size)}"
        )

    acc: np.ndarray | None = None
    acc_offset: int | None = None
    max_cond_omitted = 0.0
    max_support_size = int(x_pos_legal)
    max_window_size = int(x_pos_support.size)
    for xp, p_pos in zip(x_pos_support.tolist(), x_pos_probs.tolist(), strict=True):
        remaining_fp_neg = max(0, bucket.fp_neg - (k_pos - int(xp)))
        scipy_label = f"x_neg_given_x_pos_{int(xp)}" if int(xp) in check_conditionals else None
        x_neg_support, x_neg_probs, x_neg_omitted, x_neg_legal = _select_hypergeom_window_reference(
            successes=remaining_fp_neg,
            failures=remaining_n - remaining_fp_neg,
            draws=k_neg,
            support_guard=support_guard,
            captured_mass_eps=captured_mass_eps,
            scipy_summary=scipy_summary,
            scipy_label=scipy_label,
        )
        max_cond_omitted = max(max_cond_omitted, float(x_neg_omitted))
        max_support_size = max(max_support_size, int(x_neg_legal))
        max_window_size = max(max_window_size, int(x_neg_support.size))
        acc, acc_offset = _add_dense_window(
            acc,
            acc_offset,
            x_neg_support.astype(np.int64, copy=False) + int(xp),
            x_neg_probs.astype(np.float64, copy=False) * float(p_pos),
        )

    if acc is None or acc_offset is None:
        raise DiagnosticInvalid(f"reference PMF produced no support for bucket={bucket}")
    pmf = _dense_window_to_sparse(acc, acc_offset)

    return {
        "pmf": _normalize_sparse_pmf(pmf),
        "omitted_mass_bound": float(x_pos_omitted + max_cond_omitted),
        "max_support_size": max_support_size,
        "materialized_window_size": int(acc.size),
        "max_stage_window_size": max_window_size,
        "reference_mode": "scalar_loop_exact",
        "fallback_flag": False,
        "scipy_cross_check": scipy_summary,
    }


def _joint_match_pmf_reference_scipy_vectorized_sparse(
    bucket: BucketCounts,
    *,
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
    joint_work_budget: int = GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
    chunk_cell_budget: int = GPU_NATIVE_REFERENCE_CHUNK_CELL_BUDGET,
) -> dict[str, Any]:
    n = bucket.total
    scipy_summary = _new_scipy_cross_check_summary()
    if n <= 0:
        return {
            "pmf": {0: 1.0},
            "omitted_mass_bound": 0.0,
            "max_support_size": 1,
            "materialized_window_size": 1,
            "max_stage_window_size": 1,
            "reference_mode": "vectorized_chunked_exact",
            "fallback_flag": False,
            "joint_work": {
                "estimated_joint_cells": 0,
                "joint_work_budget": int(joint_work_budget),
                "chunk_cell_budget": int(chunk_cell_budget),
            },
            "scipy_cross_check": scipy_summary,
        }

    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    x_pos_support, x_pos_probs, x_pos_omitted, x_pos_legal = _select_hypergeom_window_reference(
        successes=bucket.fp_pos,
        failures=n - bucket.fp_pos,
        draws=k_pos,
        support_guard=support_guard,
        captured_mass_eps=captured_mass_eps,
        scipy_summary=scipy_summary,
        scipy_label="x_pos",
    )
    check_conditionals = _conditional_cross_check_values(x_pos_support)
    skipped_conditionals = max(0, int(x_pos_support.size) - len(check_conditionals))
    if skipped_conditionals:
        scipy_summary["skipped_distribution_count"] += skipped_conditionals
        scipy_summary["fallback_reasons"].append(
            f"x_neg_conditionals_sampled_{len(check_conditionals)}_of_{int(x_pos_support.size)}"
        )

    joint_work = _reference_joint_work_estimate(
        bucket,
        x_pos_support,
        k_pos=k_pos,
        k_neg=k_neg,
        remaining_n=remaining_n,
        support_guard=support_guard,
    )
    joint_work["joint_work_budget"] = int(joint_work_budget)
    joint_work["chunk_cell_budget"] = int(chunk_cell_budget)
    if int(joint_work["estimated_joint_cells"]) > int(joint_work_budget):
        scipy_summary["fallback_reasons"].append(
            f"joint_work_budget_exceeded:{joint_work['estimated_joint_cells']}>{joint_work_budget}"
        )
        out = _joint_match_pmf_reference_bounded_sampled_sparse(
            bucket,
            joint_work=joint_work,
        )
        out["scipy_cross_check"] = scipy_summary
        return out

    acc: np.ndarray | None = None
    acc_offset: int | None = None
    max_cond_omitted = 0.0
    max_support_size = int(x_pos_legal)
    max_window_size = int(x_pos_support.size)
    cond_width_upper = max(1, int(joint_work["max_conditional_window_upper_bound"]))
    start = 0
    chunk_count = 0
    while start < int(x_pos_support.size):
        remaining = int(x_pos_support.size) - start
        rows = max(1, min(remaining, int(chunk_cell_budget) // cond_width_upper))
        while rows > 1 and rows * cond_width_upper > int(chunk_cell_budget):
            rows = max(1, rows // 2)
        stop = start + rows
        xp_chunk = x_pos_support[start:stop].astype(np.int64, copy=False)
        p_pos_chunk = x_pos_probs[start:stop].astype(np.float64, copy=False)
        remaining_fp_neg = np.maximum(0, int(bucket.fp_neg) - (int(k_pos) - xp_chunk)).astype(
            np.int64,
            copy=False,
        )
        remaining_failures = (int(remaining_n) - remaining_fp_neg).astype(np.int64, copy=False)
        x_neg_support, x_neg_probs, valid, x_neg_omitted, x_neg_legal, x_neg_support_size = (
            _select_hypergeom_windows_reference_batched(
                successes=remaining_fp_neg,
                failures=remaining_failures,
                draws=k_neg,
                support_guard=support_guard,
                captured_mass_eps=captured_mass_eps,
            )
        )
        max_cond_omitted = max(max_cond_omitted, float(x_neg_omitted.max(initial=0.0)))
        max_support_size = max(max_support_size, int(x_neg_legal.max(initial=0)))
        max_window_size = max(max_window_size, int(x_neg_support_size.max(initial=0)))
        keys = x_neg_support + xp_chunk[:, None]
        values = x_neg_probs * p_pos_chunk[:, None]
        flat_valid = valid.reshape(-1)
        acc, acc_offset = _add_dense_window(
            acc,
            acc_offset,
            keys.reshape(-1)[flat_valid].astype(np.int64, copy=False),
            values.reshape(-1)[flat_valid].astype(np.float64, copy=False),
        )
        for row_idx, xp in enumerate(xp_chunk.tolist()):
            if int(xp) not in check_conditionals:
                continue
            row_support = x_neg_support[row_idx, : int(x_neg_support_size[row_idx])]
            _record_scipy_cross_check(
                scipy_summary,
                successes=int(remaining_fp_neg[row_idx]),
                failures=int(remaining_failures[row_idx]),
                draws=k_neg,
                support=row_support,
                label=f"x_neg_given_x_pos_{int(xp)}",
            )
        start = stop
        chunk_count += 1

    if acc is None or acc_offset is None:
        raise DiagnosticInvalid(f"reference PMF produced no support for bucket={bucket}")
    pmf = _dense_window_to_sparse(acc, acc_offset)

    return {
        "pmf": _normalize_sparse_pmf(pmf),
        "omitted_mass_bound": float(x_pos_omitted + max_cond_omitted),
        "max_support_size": max_support_size,
        "materialized_window_size": int(acc.size),
        "max_stage_window_size": max_window_size,
        "reference_mode": "vectorized_chunked_exact",
        "fallback_flag": False,
        "joint_work": {
            **joint_work,
            "chunk_count": chunk_count,
            "chunking_mode": "batched_conditional_scatter",
        },
        "scipy_cross_check": scipy_summary,
    }


def _joint_match_pmf_gpu_windowed_sparse(
    bucket: BucketCounts,
    *,
    device_name: str,
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
    stats: dict[str, Any] | None = None,
    joint_work_budget: int = GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
) -> dict[str, Any]:
    n = bucket.total
    local_stats = stats if stats is not None else _new_support_policy_stats()
    if n <= 0:
        return {
            "pmf": {0: 1.0},
            "omitted_mass_bound": 0.0,
            "materialized_window_size": 1,
            "max_stage_window_size": 1,
            "support_policy": local_stats,
            "candidate_mode": "torch_windowed_exact",
        }

    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    x_pos_support, x_pos_probs, x_pos_omitted = _select_hypergeom_window_scalar(
        successes=bucket.fp_pos,
        failures=n - bucket.fp_pos,
        draws=k_pos,
        device_name=device_name,
        support_guard=support_guard,
        captured_mass_eps=captured_mass_eps,
        stats=local_stats,
    )
    x_pos_np = x_pos_support.detach().cpu().numpy().astype(np.int64, copy=False)
    joint_work = _reference_joint_work_estimate(
        bucket,
        x_pos_np,
        k_pos=k_pos,
        k_neg=k_neg,
        remaining_n=remaining_n,
        support_guard=support_guard,
    )
    joint_work["joint_work_budget"] = int(joint_work_budget)
    if int(joint_work["estimated_joint_cells"]) > int(joint_work_budget):
        out = _joint_match_pmf_reference_bounded_sampled_sparse(
            bucket,
            joint_work=joint_work,
            seed_salt=7919,
            reference_mode="bounded_sampled_candidate",
            fallback_reason="candidate_joint_work_budget_exceeded",
        )
        out["support_policy"] = local_stats
        out["candidate_mode"] = "bounded_sampled"
        return out
    acc: np.ndarray | None = None
    acc_offset: int | None = None
    max_cond_omitted = 0.0
    max_window_size = int(x_pos_support.numel())
    for xp, p_pos in zip(x_pos_support.detach().cpu().tolist(), x_pos_probs.detach().cpu().tolist(), strict=True):
        remaining_fp_neg = max(0, bucket.fp_neg - (k_pos - int(xp)))
        x_neg_support, x_neg_probs, x_neg_omitted = _select_hypergeom_window_scalar(
            successes=remaining_fp_neg,
            failures=remaining_n - remaining_fp_neg,
            draws=k_neg,
            device_name=device_name,
            support_guard=support_guard,
            captured_mass_eps=captured_mass_eps,
            stats=local_stats,
        )
        max_cond_omitted = max(max_cond_omitted, float(x_neg_omitted))
        x_neg_np = x_neg_support.detach().cpu().numpy().astype(np.int64, copy=False)
        max_window_size = max(max_window_size, int(x_neg_np.size))
        acc, acc_offset = _add_dense_window(
            acc,
            acc_offset,
            x_neg_np + int(xp),
            x_neg_probs.detach().cpu().numpy().astype(np.float64, copy=False) * float(p_pos),
        )

    if acc is None or acc_offset is None:
        raise DiagnosticInvalid(f"candidate PMF produced no support for bucket={bucket}")
    pmf = _dense_window_to_sparse(acc, acc_offset)

    return {
        "pmf": _normalize_sparse_pmf(pmf),
        "omitted_mass_bound": float(x_pos_omitted + max_cond_omitted),
        "materialized_window_size": int(acc.size),
        "max_stage_window_size": max_window_size,
        "support_policy": local_stats,
        "candidate_mode": "torch_windowed_exact",
        "joint_work": joint_work,
    }


def _sparse_pmf_distance_metrics(
    reference: dict[int, float],
    candidate: dict[int, float],
    *,
    omitted_mass_bound: float,
) -> dict[str, float]:
    keys = sorted(set(reference) | set(candidate))
    ref_cdf = 0.0
    cand_cdf = 0.0
    tv_core = 0.0
    max_cdf_core = 0.0
    max_pmf = 0.0
    for key in keys:
        ref = float(reference.get(key, 0.0))
        cand = float(candidate.get(key, 0.0))
        diff = abs(ref - cand)
        tv_core += diff
        max_pmf = max(max_pmf, diff)
        ref_cdf += ref
        cand_cdf += cand
        max_cdf_core = max(max_cdf_core, abs(ref_cdf - cand_cdf))
    tv_core *= 0.5
    return {
        "tv_distance_core": tv_core,
        "max_cdf_delta_core": max_cdf_core,
        "max_pmf_delta": max_pmf,
        "omitted_mass_bound": omitted_mass_bound,
        "tv_distance": tv_core + omitted_mass_bound,
        "max_cdf_delta": max_cdf_core + omitted_mass_bound,
    }


def _joint_match_pmf_reference(bucket: BucketCounts) -> np.ndarray:
    n = bucket.total
    if n <= 0:
        return np.asarray([1.0], dtype=np.float64)
    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    max_total = k_pos + k_neg
    pmf = np.zeros(max_total + 1, dtype=np.float64)
    xp_lo = max(0, k_pos - bucket.fp_neg)
    xp_hi = min(k_pos, bucket.fp_pos)
    for xp in range(xp_lo, xp_hi + 1):
        p_pos = math.exp(_hypergeom_logpmf_float(bucket.fp_pos, bucket.fp_neg, k_pos, xp))
        remaining_fp_neg = max(0, bucket.fp_neg - (k_pos - xp))
        xn_lo = max(0, k_neg - (remaining_n - remaining_fp_neg))
        xn_hi = min(k_neg, remaining_fp_neg)
        for xn in range(xn_lo, xn_hi + 1):
            p_neg = math.exp(
                _hypergeom_logpmf_float(
                    remaining_fp_neg,
                    remaining_n - remaining_fp_neg,
                    k_neg,
                    xn,
                )
            )
            pmf[xp + xn] += p_pos * p_neg
    total = pmf.sum()
    if total <= 0:
        raise DiagnosticInvalid(f"reference PMF has zero mass for bucket={bucket}")
    return pmf / total


def _joint_match_pmf_gpu_windowed(
    bucket: BucketCounts,
    *,
    device_name: str,
    support_guard: int = GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
    captured_mass_eps: float = GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
    stats: dict[str, Any] | None = None,
) -> np.ndarray:
    n = bucket.total
    if n <= 0:
        return np.asarray([1.0], dtype=np.float64)
    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    max_total = k_pos + k_neg
    pmf = np.zeros(max_total + 1, dtype=np.float64)
    local_stats = stats if stats is not None else _new_support_policy_stats()
    x_pos_support, x_pos_probs, _ = _select_hypergeom_window_scalar(
        successes=bucket.fp_pos,
        failures=n - bucket.fp_pos,
        draws=k_pos,
        device_name=device_name,
        support_guard=support_guard,
        captured_mass_eps=captured_mass_eps,
        stats=local_stats,
    )
    for xp, p_pos in zip(x_pos_support.detach().cpu().tolist(), x_pos_probs.detach().cpu().tolist(), strict=True):
        remaining_fp_neg = max(0, bucket.fp_neg - (k_pos - int(xp)))
        x_neg_support, x_neg_probs, _ = _select_hypergeom_window_scalar(
            successes=remaining_fp_neg,
            failures=remaining_n - remaining_fp_neg,
            draws=k_neg,
            device_name=device_name,
            support_guard=support_guard,
            captured_mass_eps=captured_mass_eps,
            stats=local_stats,
        )
        for xn, p_neg in zip(x_neg_support.detach().cpu().tolist(), x_neg_probs.detach().cpu().tolist(), strict=True):
            pmf[int(xp) + int(xn)] += float(p_pos) * float(p_neg)
    total = pmf.sum()
    if total <= 0:
        raise DiagnosticInvalid(f"gpu windowed PMF has zero mass for bucket={bucket}")
    return pmf / total


def _pmf_distance_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    size = max(reference.size, candidate.size)
    ref = np.zeros(size, dtype=np.float64)
    cand = np.zeros(size, dtype=np.float64)
    ref[: reference.size] = reference
    cand[: candidate.size] = candidate
    diff = np.abs(ref - cand)
    return {
        "tv_distance": float(0.5 * diff.sum()),
        "max_cdf_delta": float(np.abs(np.cumsum(ref) - np.cumsum(cand)).max()) if size else 0.0,
        "max_pmf_delta": float(diff.max()) if size else 0.0,
    }


def _bucket_joint_support_span(bucket: BucketCounts) -> int:
    n = bucket.total
    if n <= 0:
        return 1
    k_pos = min(bucket.int_pos, n)
    remaining_n = n - k_pos
    k_neg = min(bucket.int_neg, remaining_n)
    return k_pos + k_neg + 1


def _bucket_skew_tail_score(bucket: BucketCounts) -> float:
    n = max(1, bucket.total)
    fp_balance = abs((bucket.fp_pos / n) - 0.5)
    int_move = (bucket.int_pos + bucket.int_neg) / n
    int_balance = abs((bucket.int_pos / max(1, bucket.int_pos + bucket.int_neg)) - 0.5)
    zero_rate = bucket.int_zero / n
    return float(fp_balance + int_balance + zero_rate + abs(int_move - 0.5))


def _bucket_manifest(bucket: BucketCounts) -> dict[str, Any]:
    return {
        "fp_pos": bucket.fp_pos,
        "fp_neg": bucket.fp_neg,
        "int_pos": bucket.int_pos,
        "int_neg": bucket.int_neg,
        "int_zero": bucket.int_zero,
        "total": bucket.total,
        "joint_support_span": _bucket_joint_support_span(bucket),
        "skew_tail_score": _bucket_skew_tail_score(bucket),
    }


def _null_item_bucket_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        counts: CountAccumulator = item["counts"]
        q0_denom = counts.q_stats.get("0", {}).get("denom", 0)
        q_denoms = {q: stats.get("denom", 0) for q, stats in counts.q_stats.items()}
        mixed_q = sum(1 for denom in q_denoms.values() if denom > 0) >= 2
        for null_kind, buckets, _seed in _null_item_runs(item, null_seed=0):
            for bucket_idx, bucket in enumerate(buckets):
                records.append(
                    {
                        "variant": item["variant"],
                        "level": item["level"],
                        "label": item["label"],
                        "null_kind": null_kind,
                        "bucket_idx": bucket_idx,
                        "bucket": bucket,
                        "q0_denom": q0_denom,
                        "q_denoms": q_denoms,
                        "mixed_q": mixed_q,
                        "denominator": bucket.total,
                        "support_size": _bucket_joint_support_span(bucket),
                        "skew_tail_score": _bucket_skew_tail_score(bucket),
                    }
                )
    return records


def collect_real_analytic_pmf_fixtures(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    max_invocations_per_variant: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = collect_null_profile_items(
        variant_count_sets,
        max_invocations_per_variant=max_invocations_per_variant,
    )
    records = _null_item_bucket_records(items)
    if not records:
        raise DiagnosticInvalid("real analytic PMF corpus has no bucket records")

    winners: dict[str, dict[str, Any]] = {
        "max_denominator": max(records, key=lambda r: r["denominator"]),
        "max_support_size": max(records, key=lambda r: r["support_size"]),
        "max_q0_denominator": max(records, key=lambda r: r["q0_denom"]),
        "skew_tail_heavy": max(records, key=lambda r: r["skew_tail_score"]),
        "global_permutation": max(
            (r for r in records if r["null_kind"] == "global_permutation"),
            key=lambda r: r["denominator"],
        ),
        "row_q_preserving": max(
            (r for r in records if r["null_kind"] == "row_q_preserving"),
            key=lambda r: r["denominator"],
        ),
    }

    deduped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    fixture_reasons: dict[tuple[str, str, str, str, int], list[str]] = {}
    for reason, record in winners.items():
        key = (
            record["variant"],
            record["level"],
            record["label"],
            record["null_kind"],
            int(record["bucket_idx"]),
        )
        deduped[key] = record
        fixture_reasons.setdefault(key, []).append(reason)

    fixtures: list[dict[str, Any]] = []
    for key, record in sorted(deduped.items()):
        reasons = fixture_reasons[key]
        fixtures.append(
            {
                "name": "real_" + "_".join(reasons),
                "source": "real_full_subset",
                "reasons": reasons,
                "variant": record["variant"],
                "level": record["level"],
                "label": record["label"],
                "null_kind": record["null_kind"],
                "bucket_idx": record["bucket_idx"],
                "q_level": "mixed" if record["mixed_q"] else "single_or_empty",
                "q0_denom": record["q0_denom"],
                "q_denoms": record["q_denoms"],
                "bucket": record["bucket"],
                "support_guard": GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
            }
        )

    manifest = {
        "item_count": len(items),
        "bucket_record_count": len(records),
        "required_real_winner_names": sorted(winners),
        "real_winners": {
            reason: {
                "variant": record["variant"],
                "level": record["level"],
                "label": record["label"],
                "null_kind": record["null_kind"],
                "bucket_idx": record["bucket_idx"],
                "q0_denom": record["q0_denom"],
                "q_denoms": record["q_denoms"],
                "bucket": _bucket_manifest(record["bucket"]),
            }
            for reason, record in sorted(winners.items())
        },
        "deduped_fixture_count": len(fixtures),
        "null_kind_coverage": sorted({record["null_kind"] for record in records}),
        "max_denominator": max(record["denominator"] for record in records),
        "max_support_size": max(record["support_size"] for record in records),
        "max_q0_denominator": max(record["q0_denom"] for record in records),
    }
    return fixtures, manifest


def analytic_pmf_fixture_corpus(
    variant_count_sets: dict[str, dict[str, Any]] | None = None,
    *,
    max_invocations_per_variant: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    synthetic = [
        {
            "name": "q_neg_small_full_support",
            "source": "synthetic",
            "q_level": -1,
            "bucket": BucketCounts(fp_pos=6, fp_neg=4, int_pos=5, int_neg=3, int_zero=2),
            "support_guard": GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
        },
        {
            "name": "q_zero_skew_tail_full_support",
            "source": "synthetic",
            "q_level": 0,
            "bucket": BucketCounts(fp_pos=95, fp_neg=5, int_pos=80, int_neg=10, int_zero=10),
            "support_guard": GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
        },
        {
            "name": "q_pos_large_forced_mass_trim",
            "source": "synthetic",
            "q_level": 1,
            "bucket": BucketCounts(fp_pos=700, fp_neg=500, int_pos=650, int_neg=300, int_zero=250),
            "support_guard": 128,
        },
    ]
    manifest: dict[str, Any] = {
        "synthetic_fixture_count": len(synthetic),
        "real_full_subset_required": variant_count_sets is not None,
    }
    if variant_count_sets is None:
        return synthetic, manifest
    real, real_manifest = collect_real_analytic_pmf_fixtures(
        variant_count_sets,
        max_invocations_per_variant=max_invocations_per_variant,
    )
    manifest["real_full_subset"] = real_manifest
    return synthetic + real, manifest


def run_analytic_pmf_parity(
    device_name: str | None = None,
    variant_count_sets: dict[str, dict[str, Any]] | None = None,
    *,
    max_invocations_per_variant: int = 16,
    reference_joint_work_budget: int = GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
    reference_chunk_cell_budget: int = GPU_NATIVE_REFERENCE_CHUNK_CELL_BUDGET,
) -> dict[str, Any]:
    device = _torch_device_name(device_name)
    corpus, corpus_manifest = analytic_pmf_fixture_corpus(
        variant_count_sets,
        max_invocations_per_variant=max_invocations_per_variant,
    )
    fixtures: list[dict[str, Any]] = []
    max_tv = 0.0
    max_cdf = 0.0
    max_omitted = 0.0
    scipy_failures: list[dict[str, Any]] = []
    for fixture in corpus:
        print(
            "[credit-bridge] stage1 analytic fixture="
            f"{fixture['name']} source={fixture.get('source')} null_kind={fixture.get('null_kind')}",
            flush=True,
        )
        stats = _new_support_policy_stats()
        reference = _joint_match_pmf_reference_scipy_vectorized_sparse(
            fixture["bucket"],
            support_guard=int(fixture["support_guard"]),
            captured_mass_eps=GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
            joint_work_budget=reference_joint_work_budget,
            chunk_cell_budget=reference_chunk_cell_budget,
        )
        candidate = _joint_match_pmf_gpu_windowed_sparse(
            fixture["bucket"],
            device_name=device,
            support_guard=int(fixture["support_guard"]),
            captured_mass_eps=GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
            stats=stats,
            joint_work_budget=reference_joint_work_budget,
        )
        omitted_bound = float(reference["omitted_mass_bound"]) + float(candidate["omitted_mass_bound"])
        metrics = _sparse_pmf_distance_metrics(
            reference["pmf"],
            candidate["pmf"],
            omitted_mass_bound=omitted_bound,
        )
        max_tv = max(max_tv, metrics["tv_distance"])
        max_cdf = max(max_cdf, metrics["max_cdf_delta"])
        max_omitted = max(max_omitted, omitted_bound, float(stats["max_omitted_mass"]))
        scipy_cross_check = reference["scipy_cross_check"]
        if scipy_cross_check["failures"]:
            scipy_failures.append({"fixture": fixture["name"], "failures": scipy_cross_check["failures"]})
        exact_reference_modes = {"exact_windowed_scipy", "vectorized_chunked_exact", "streaming_cdf_exact"}
        bounded_sample = reference.get("bounded_sample")
        exact_certified = (
            reference["reference_mode"] in exact_reference_modes
            and metrics["tv_distance"] <= GPU_NATIVE_PMF_TV_BOUND
            and metrics["max_cdf_delta"] <= GPU_NATIVE_PMF_CDF_BOUND
            and reference["max_stage_window_size"] <= fixture["support_guard"]
            and candidate["max_stage_window_size"] <= fixture["support_guard"]
            and not scipy_cross_check["failures"]
        )
        sampling_aware_metrics = None
        bounded_certified = False
        candidate_bounded_sample = candidate.get("bounded_sample")
        if reference["reference_mode"] == "bounded_sampled" and bounded_sample is not None:
            candidate_dkw = (
                float(candidate_bounded_sample["dkw_epsilon"])
                if candidate_bounded_sample is not None
                else 0.0
            )
            sampling_cdf_delta_bound = (
                metrics["max_cdf_delta_core"]
                + float(bounded_sample["dkw_epsilon"])
                + candidate_dkw
                + float(candidate["omitted_mass_bound"])
            )
            sampling_aware_metrics = {
                "max_cdf_delta_core": metrics["max_cdf_delta_core"],
                "reference_dkw_epsilon": float(bounded_sample["dkw_epsilon"]),
                "candidate_dkw_epsilon": candidate_dkw,
                "candidate_omitted_mass_bound": float(candidate["omitted_mass_bound"]),
                "sampling_cdf_delta_bound": sampling_cdf_delta_bound,
                "sampling_cdf_bound": float(bounded_sample["sampling_cdf_bound"]),
                "confidence": float(bounded_sample["confidence"]),
            }
            bounded_certified = (
                sampling_cdf_delta_bound <= float(bounded_sample["sampling_cdf_bound"])
                and not scipy_cross_check["failures"]
            )
        q0_structured = fixture["q_level"] == 0 or int(fixture.get("q0_denom") or 0) > 0
        fixtures.append(
            {
                "name": fixture["name"],
                "source": fixture.get("source"),
                "reasons": fixture.get("reasons", []),
                "variant": fixture.get("variant"),
                "level": fixture.get("level"),
                "label": fixture.get("label"),
                "null_kind": fixture.get("null_kind"),
                "q_level": fixture["q_level"],
                "q0_denom": fixture.get("q0_denom"),
                "bucket": _bucket_manifest(fixture["bucket"]),
                "support_guard": fixture["support_guard"],
                "metrics": metrics,
                "support_policy": stats,
                "reference_pmf_function": REFERENCE_PMF_FUNCTION,
                "candidate_pmf_function": CANDIDATE_PMF_FUNCTION,
                "reference_candidate_independent": REFERENCE_PMF_FUNCTION != CANDIDATE_PMF_FUNCTION,
                "reference_mode": reference["reference_mode"],
                "fallback_flag": bool(reference.get("fallback_flag", False)),
                "fallback_reason": reference.get("fallback_reason"),
                "joint_work": reference.get("joint_work"),
                "bounded_sample": bounded_sample,
                "candidate_mode": candidate.get("candidate_mode", "torch_windowed_exact"),
                "candidate_bounded_sample": candidate_bounded_sample,
                "sampling_aware_metrics": sampling_aware_metrics,
                "reference_materialized_window_size": reference["materialized_window_size"],
                "candidate_materialized_window_size": candidate["materialized_window_size"],
                "reference_max_stage_window_size": reference["max_stage_window_size"],
                "candidate_max_stage_window_size": candidate["max_stage_window_size"],
                "scipy_cross_check": scipy_cross_check,
                "q0_structured": q0_structured,
                "stage1_exact_certified": exact_certified,
                "bounded_certified": bounded_certified,
                "science_unblock_eligible": exact_certified or bounded_certified,
                "pass": exact_certified or bounded_certified,
            }
        )
    exact_backend_certified = all(item["stage1_exact_certified"] for item in fixtures)
    bounded_items = [item for item in fixtures if item["reference_mode"] == "bounded_sampled"]
    bounded_reference_certified = bool(bounded_items) and all(item["bounded_certified"] for item in bounded_items)
    q0_exact_coverage = any(item["q0_structured"] and item["stage1_exact_certified"] for item in fixtures)
    explicit_backend_validated_for_science = (
        all(item["science_unblock_eligible"] for item in fixtures)
        and q0_exact_coverage
        and not scipy_failures
    )
    default_flip_eligible = exact_backend_certified
    passed = explicit_backend_validated_for_science
    return {
        "device": device,
        "primary_math_guard": True,
        "reference_pmf_function": REFERENCE_PMF_FUNCTION,
        "candidate_pmf_function": CANDIDATE_PMF_FUNCTION,
        "reference_candidate_independent": REFERENCE_PMF_FUNCTION != CANDIDATE_PMF_FUNCTION,
        "tv_bound": GPU_NATIVE_PMF_TV_BOUND,
        "cdf_bound": GPU_NATIVE_PMF_CDF_BOUND,
        "captured_mass_epsilon": GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
        "joint_work_budget": reference_joint_work_budget,
        "chunk_cell_budget": reference_chunk_cell_budget,
        "bounded_sample_count": GPU_NATIVE_BOUNDED_SAMPLE_COUNT,
        "bounded_sample_confidence": GPU_NATIVE_BOUNDED_SAMPLE_CONFIDENCE,
        "bounded_sample_cdf_bound": GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND,
        "scipy_cross_check_required": True,
        "scipy_cross_check_failure_count": len(scipy_failures),
        "scipy_cross_check_failures": scipy_failures[:20],
        "corpus_manifest": corpus_manifest,
        "max_tv_distance": max_tv,
        "max_cdf_delta": max_cdf,
        "max_omitted_mass": max_omitted,
        "exact_backend_certified": exact_backend_certified,
        "bounded_reference_certified": bounded_reference_certified,
        "explicit_backend_validated_for_science": explicit_backend_validated_for_science,
        "default_flip_eligible": default_flip_eligible,
        "q0_exact_coverage": {
            "present": q0_exact_coverage,
            "fixture_names": [
                item["name"] for item in fixtures if item["q0_structured"] and item["stage1_exact_certified"]
            ],
        },
        "fallback_fixture_count": len(bounded_items),
        "pass": passed,
        "fixtures": fixtures,
    }


def _aggregate_match_matrix_with_torch(
    matches: np.ndarray,
    totals: np.ndarray,
    *,
    device_name: str,
) -> tuple[np.ndarray, float]:
    if matches.size == 0 or totals.size == 0:
        return np.asarray([], dtype=np.float64), 0.0
    _sync_for_timing(device_name)
    start = time.perf_counter()
    match_t = torch.as_tensor(matches, dtype=torch.int64, device=device_name)
    per_perm = match_t.sum(dim=1).cpu().numpy().astype(np.int64)
    _sync_for_timing(device_name)
    elapsed = time.perf_counter() - start
    total = int(totals.sum())
    scores = per_perm.astype(np.float64) / float(total) if total else np.zeros(matches.shape[0], dtype=np.float64)
    return scores, elapsed


def _simulate_permutation_null_cpu_sampler_gpu_aggregation_replay(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
    aggregation_device: str | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    device_name = _torch_device_name(aggregation_device)
    matches, totals, sample_seconds = _sample_bucket_match_matrix_cpu_locked(
        buckets,
        permutations=permutations,
        seed=seed,
    )
    scores, aggregation_seconds = _aggregate_match_matrix_with_torch(matches, totals, device_name=device_name)
    timing = None
    if profile:
        timing = {
            "cpu_sampler": sample_seconds,
            "aggregation": aggregation_seconds,
            "total": time.perf_counter() - total_start,
        }
    return _summarize_null_scores(
        scores,
        backend=NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
        timing_seconds=timing,
        aggregation_device=device_name,
    )


def simulate_permutation_null(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
    backend: str = DEFAULT_NULL_BACKEND,
    aggregation_device: str | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    if backend == NULL_BACKEND_CPU_LOCKED:
        return _simulate_permutation_null_cpu_locked(
            buckets,
            permutations=permutations,
            seed=seed,
            profile=profile,
        )
    if backend == NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY:
        return _simulate_permutation_null_cpu_sampler_gpu_aggregation_replay(
            buckets,
            permutations=permutations,
            seed=seed,
            aggregation_device=aggregation_device,
            profile=profile,
        )
    if backend == NULL_BACKEND_GPU_NATIVE_COUNTS_PMF:
        return _simulate_permutation_null_gpu_native_counts_pmf(
            buckets,
            permutations=permutations,
            seed=seed,
            aggregation_device=aggregation_device,
            profile=profile,
        )
    raise DiagnosticInvalid(f"unknown null backend: {backend}")


def deterministic_seed(base_seed: int, label: str, offset: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{offset}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_counts_for_invocation(
    key: InvocationKey,
    agg: InvocationAggregate,
    q_levels: Tensor,
    *,
    variant: str,
) -> tuple[CountAccumulator, dict[str, Any]]:
    if agg.weighted_grad is None or agg.active_inputs is None or agg.active_outputs is None:
        raise DiagnosticInvalid(f"{key.label}: no backward aggregate captured")
    q = q_levels[agg.group_start:agg.group_end].cpu()
    weighted = agg.weighted_grad.cpu()
    if variant == "full_magnitude_ceiling":
        credit = -weighted
    else:
        if variant not in agg.variant_credits:
            raise DiagnosticInvalid(f"{key.label}: missing credit variant {variant}")
        credit = agg.variant_credits[variant].cpu()
    active_inputs = agg.active_inputs.cpu()
    active_outputs = agg.active_outputs.cpu()

    fp_moves = project_fp_gradient_to_moves(weighted, q)
    int_moves = project_integer_credit_to_moves(credit, q)
    denom_mask = fp_moves != 0
    agree_mask = denom_mask & (fp_moves == int_moves)

    counts = CountAccumulator(label=key.label)
    counts.denom = int(denom_mask.sum().item())
    counts.agree = int(agree_mask.sum().item())
    global_bucket = _bucket_counts(fp_moves, int_moves, denom_mask)
    if global_bucket is not None:
        counts.buckets_global.append(global_bucket)
    counts.buckets_rowq.extend(row_q_bucket_counts(fp_moves, int_moves, q, denom_mask))

    for q_value in (-1, 0, 1):
        qm = denom_mask & (q == q_value)
        q_label = str(q_value)
        counts.q_stats[q_label] = {
            "denom": int(qm.sum().item()),
            "agree": int((qm & (fp_moves == int_moves)).sum().item()),
        }

    weighted_dir = -weighted.sign()
    credit_dir = credit.sign()
    raw_denom = weighted_dir != 0
    counts.raw_dir_denom = int(raw_denom.sum().item())
    counts.raw_dir_disagree = int((raw_denom & (credit_dir != 0) & (weighted_dir != credit_dir)).sum().item())
    counts.raw_dir_integer_zero = int((raw_denom & (credit_dir == 0)).sum().item())

    admissible_routes = active_inputs & ((q != 0) | (int_moves != 0))
    current_routes = active_inputs & (q != 0)
    route_counts = admissible_routes.sum(dim=1).to(torch.int64)
    current_counts = current_routes.sum(dim=1).to(torch.int64)
    active_route_counts = route_counts[active_outputs]
    active_current_counts = current_counts[active_outputs]
    counts.active_output_count = int(active_outputs.sum().item())
    counts.dead_active_output_count = int((active_outputs & (route_counts == 0)).sum().item())
    counts.route_counts = [int(v) for v in active_route_counts.tolist()]
    counts.current_route_counts = [int(v) for v in active_current_counts.tolist()]
    counts.prefix_active_positions = agg.prefix_active_positions
    counts.response_active_positions = agg.response_active_positions
    counts.prefix_active_output_elements = agg.prefix_active_output_elements
    counts.response_active_output_elements = agg.response_active_output_elements

    detail = {
        "variant": variant,
        "invocation_label": key.label,
        "family": key.family_label,
        "aggregate64": key.aggregate64_label,
        "module_name": agg.module_name,
        "group_rows": [agg.group_start, agg.group_end],
        "backward_calls": agg.backward_calls,
        "input_abs": agg.input_abs.as_dict(),
        "grad_abs": agg.grad_abs.as_dict(),
    }
    return counts, detail


def summarize_counts(
    counts: CountAccumulator,
    *,
    bars: Bars,
    level: str,
    null_permutations: int,
    null_seed: int,
    null_backend: str = DEFAULT_NULL_BACKEND,
    null_aggregation_device: str | None = None,
    profile_null: bool = False,
) -> dict[str, Any]:
    agreement = counts.agree / counts.denom if counts.denom else 0.0
    global_null = simulate_permutation_null(
        counts.buckets_global,
        permutations=null_permutations,
        seed=deterministic_seed(null_seed, counts.label, 1),
        backend=null_backend,
        aggregation_device=null_aggregation_device,
        profile=profile_null,
    )
    rowq_null = simulate_permutation_null(
        counts.buckets_rowq,
        permutations=null_permutations,
        seed=deterministic_seed(null_seed, counts.label, 2),
        backend=null_backend,
        aggregation_device=null_aggregation_device,
        profile=profile_null,
    )
    stronger_p99 = max(global_null["p99"], rowq_null["p99"])
    if level == "global":
        threshold = max(stronger_p99 + bars.global_null_margin, bars.global_floor)
    elif level == "family":
        threshold = max(stronger_p99 + bars.family_null_margin, bars.family_floor)
    else:
        threshold = max(stronger_p99 + bars.stratum_null_margin, bars.stratum_floor)

    route_counts = np.asarray(counts.route_counts, dtype=np.float64)
    current_counts = np.asarray(counts.current_route_counts, dtype=np.float64)
    route_summary = {
        "active_output_count": counts.active_output_count,
        "dead_active_output_count": counts.dead_active_output_count,
        "dead_active_output_rate": (
            counts.dead_active_output_count / counts.active_output_count
            if counts.active_output_count
            else 0.0
        ),
        "admissible_p01": int(np.quantile(route_counts, 0.01, method="lower")) if route_counts.size else 0,
        "admissible_median": int(np.median(route_counts)) if route_counts.size else 0,
        "current_only_p01": int(np.quantile(current_counts, 0.01, method="lower")) if current_counts.size else 0,
        "current_only_median": int(np.median(current_counts)) if current_counts.size else 0,
    }
    q_stats: dict[str, Any] = {}
    for q, stats in sorted(counts.q_stats.items(), key=lambda kv: int(kv[0])):
        denom = stats["denom"]
        q_stats[q] = {
            "denom": denom,
            "agree": stats["agree"],
            "agreement": stats["agree"] / denom if denom else None,
        }

    return {
        "label": counts.label,
        "level": level,
        "agreement": agreement,
        "agree": counts.agree,
        "denom": counts.denom,
        "threshold": threshold,
        "pass_sign": counts.denom > 0 and agreement >= threshold,
        "null": {
            "global_permutation": global_null,
            "row_q_preserving": rowq_null,
            "stronger_p99": stronger_p99,
        },
        "q_level": q_stats,
        "route": route_summary,
        "token_position_coverage": {
            "prefix_active_positions": counts.prefix_active_positions,
            "response_active_positions": counts.response_active_positions,
            "prefix_active_output_elements": counts.prefix_active_output_elements,
            "response_active_output_elements": counts.response_active_output_elements,
        },
        "fp_non_tautology": {
            "raw_direction_denom": counts.raw_dir_denom,
            "raw_direction_disagree": counts.raw_dir_disagree,
            "raw_direction_integer_zero": counts.raw_dir_integer_zero,
            "raw_direction_disagree_rate": (
                counts.raw_dir_disagree / counts.raw_dir_denom if counts.raw_dir_denom else 0.0
            ),
            "raw_direction_integer_zero_rate": (
                counts.raw_dir_integer_zero / counts.raw_dir_denom if counts.raw_dir_denom else 0.0
            ),
        },
    }


def _select_evenly_spaced(items: list[tuple[str, CountAccumulator]], limit: int) -> list[tuple[str, CountAccumulator]]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indexes = np.linspace(0, len(items) - 1, num=limit, dtype=np.int64)
    selected: list[tuple[str, CountAccumulator]] = []
    seen: set[int] = set()
    for idx in indexes.tolist():
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    return selected


def collect_null_profile_items(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    max_invocations_per_variant: int,
) -> list[dict[str, Any]]:
    """Choose a deterministic mixed-q subset covering global, all families, and invocations."""
    items: list[dict[str, Any]] = []
    for variant in CREDIT_VARIANTS:
        count_set = variant_count_sets[variant]
        items.append({"variant": variant, "level": "global", "label": "global", "counts": count_set["global"]})
        for label, counts in sorted(count_set["families"].items()):
            items.append({"variant": variant, "level": "family", "label": label, "counts": counts})

        invocation_items = sorted(count_set["invocations"].items())
        selected = _select_evenly_spaced(invocation_items, max_invocations_per_variant)
        # Force q=0 denominator extremes into the fixed subset so q=0 revival coverage is explicit.
        by_q0 = sorted(
            invocation_items,
            key=lambda kv: kv[1].q_stats.get("0", {}).get("denom", 0),
        )
        for extra in [by_q0[0], by_q0[len(by_q0) // 2], by_q0[-1]]:
            if extra not in selected:
                selected.append(extra)
        for label, counts in sorted(selected):
            items.append({"variant": variant, "level": "invocation", "label": label, "counts": counts})
    return items


def _null_item_runs(item: dict[str, Any], *, null_seed: int) -> list[tuple[str, list[BucketCounts], int]]:
    counts: CountAccumulator = item["counts"]
    return [
        ("global_permutation", counts.buckets_global, deterministic_seed(null_seed, counts.label, 1)),
        ("row_q_preserving", counts.buckets_rowq, deterministic_seed(null_seed, counts.label, 2)),
    ]


def _run_null_backend_items(
    items: list[dict[str, Any]],
    *,
    backend: str,
    permutations: int,
    null_seed: int,
    aggregation_device: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    outputs: dict[str, dict[str, Any]] = {}
    timing_totals = {"cpu_sampler": 0.0, "gpu_sampler": 0.0, "aggregation": 0.0, "reported_total": 0.0}
    wall_start = time.perf_counter()
    for item in items:
        for null_kind, buckets, seed in _null_item_runs(item, null_seed=null_seed):
            input_metadata = _bucket_totals_metadata(buckets)
            out = simulate_permutation_null(
                buckets,
                permutations=permutations,
                seed=seed,
                backend=backend,
                aggregation_device=aggregation_device,
                profile=True,
            )
            out.setdefault("input_bucket_metadata", input_metadata)
            key = f"{item['variant']}::{item['level']}::{item['label']}::{null_kind}"
            outputs[key] = out
            timing = out.get("timing_seconds") or {}
            timing_totals["cpu_sampler"] += float(timing.get("cpu_sampler", 0.0))
            timing_totals["gpu_sampler"] += float(timing.get("gpu_sampler", 0.0))
            timing_totals["aggregation"] += float(timing.get("aggregation", 0.0))
            timing_totals["reported_total"] += float(timing.get("total", 0.0))
    timing_totals["wall_total"] = time.perf_counter() - wall_start
    return outputs, timing_totals


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def collect_null_speed_items(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    max_invocations_per_variant: int,
) -> list[dict[str, Any]]:
    """Small, q0-aware invocation subset used only for bounded CPU speed evidence."""
    items: list[dict[str, Any]] = []
    limit = max(1, max_invocations_per_variant)
    for variant in CREDIT_VARIANTS:
        invocation_items = sorted(variant_count_sets[variant]["invocations"].items())
        selected = _select_evenly_spaced(invocation_items, limit)
        by_q0 = sorted(
            invocation_items,
            key=lambda kv: kv[1].q_stats.get("0", {}).get("denom", 0),
        )
        for extra in [by_q0[0], by_q0[len(by_q0) // 2], by_q0[-1]]:
            if extra not in selected:
                selected.append(extra)
        for label, counts in sorted(selected):
            items.append({"variant": variant, "level": "invocation", "label": label, "counts": counts})
    return items


def _item_key(item: dict[str, Any], null_kind: str) -> str:
    return f"{item['variant']}::{item['level']}::{item['label']}::{null_kind}"


def _level_threshold(p99: float, level: str, bars: Bars) -> float:
    if level == "global":
        return max(p99 + bars.global_null_margin, bars.global_floor)
    if level == "family":
        return max(p99 + bars.family_null_margin, bars.family_floor)
    return max(p99 + bars.stratum_null_margin, bars.stratum_floor)


def _compare_distributional_null_outputs(
    cpu_outputs: dict[str, dict[str, Any]],
    candidate_outputs: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    bars: Bars,
    abs_tol: float,
) -> dict[str, Any]:
    summary_failures: list[dict[str, Any]] = []
    bucket_metadata_failures: list[dict[str, Any]] = []
    gating_failures: list[dict[str, Any]] = []
    if set(cpu_outputs) != set(candidate_outputs):
        missing_cpu = sorted(set(candidate_outputs) - set(cpu_outputs))
        missing_candidate = sorted(set(cpu_outputs) - set(candidate_outputs))
        return {
            "pass": False,
            "summary_failures": [{"reason": "key_mismatch", "missing_cpu": missing_cpu, "missing_candidate": missing_candidate}],
            "bucket_metadata_failures": [],
            "gating_failures": [],
            "abs_tol": abs_tol,
        }

    for key in sorted(cpu_outputs):
        cpu = cpu_outputs[key]
        candidate = candidate_outputs[key]
        deltas = {field: abs(float(cpu[field]) - float(candidate[field])) for field in ("mean", "p95", "p99")}
        if any(delta > abs_tol for delta in deltas.values()):
            summary_failures.append(
                {
                    "key": key,
                    "cpu": {k: cpu[k] for k in ("mean", "p95", "p99")},
                    "candidate": {k: candidate[k] for k in ("mean", "p95", "p99")},
                    "deltas": deltas,
                }
            )

        cpu_meta = cpu.get("input_bucket_metadata")
        candidate_input = candidate.get("input_bucket_metadata")
        candidate_batched = candidate.get("candidate_batching_metadata")
        if cpu_meta != candidate_input or candidate_input != candidate_batched:
            bucket_metadata_failures.append(
                {
                    "key": key,
                    "cpu_input": cpu_meta,
                    "candidate_input": candidate_input,
                    "candidate_batched": candidate_batched,
                }
            )

    for item in items:
        counts: CountAccumulator = item["counts"]
        denom = counts.denom
        agreement = counts.agree / denom if denom else 0.0
        cpu_p99 = max(
            float(cpu_outputs[_item_key(item, "global_permutation")]["p99"]),
            float(cpu_outputs[_item_key(item, "row_q_preserving")]["p99"]),
        )
        candidate_p99 = max(
            float(candidate_outputs[_item_key(item, "global_permutation")]["p99"]),
            float(candidate_outputs[_item_key(item, "row_q_preserving")]["p99"]),
        )
        cpu_threshold = _level_threshold(cpu_p99, item["level"], bars)
        candidate_threshold = _level_threshold(candidate_p99, item["level"], bars)
        cpu_decision = denom > 0 and agreement >= cpu_threshold
        candidate_decision = denom > 0 and agreement >= candidate_threshold
        if cpu_decision != candidate_decision:
            gating_failures.append(
                {
                    "variant": item["variant"],
                    "level": item["level"],
                    "label": item["label"],
                    "agreement": agreement,
                    "cpu_threshold": cpu_threshold,
                    "candidate_threshold": candidate_threshold,
                    "cpu_decision": cpu_decision,
                    "candidate_decision": candidate_decision,
                    "threshold_delta": abs(cpu_threshold - candidate_threshold),
                }
            )

    return {
        "pass": not summary_failures and not bucket_metadata_failures and not gating_failures,
        "summary_failures": summary_failures[:20],
        "bucket_metadata_failures": bucket_metadata_failures[:20],
        "gating_failures": gating_failures[:20],
        "summary_failure_count": len(summary_failures),
        "bucket_metadata_failure_count": len(bucket_metadata_failures),
        "gating_failure_count": len(gating_failures),
        "abs_tol": abs_tol,
        "compared_fields": ["mean", "p95", "p99"],
    }


def build_full_subset_bucket_metadata_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    max_denominator = 0
    max_support_size = 0
    max_q0_denom = 0
    for item in items:
        counts: CountAccumulator = item["counts"]
        q0_denom = counts.q_stats.get("0", {}).get("denom", 0)
        max_q0_denom = max(max_q0_denom, q0_denom)
        for null_kind, buckets, _seed in _null_item_runs(item, null_seed=0):
            key = _item_key(item, null_kind)
            bucket_metadata = _bucket_totals_metadata(buckets)
            bucket_max_denominator = max((bucket.total for bucket in buckets), default=0)
            bucket_max_support = max((_bucket_joint_support_span(bucket) for bucket in buckets), default=0)
            max_denominator = max(max_denominator, bucket_max_denominator)
            max_support_size = max(max_support_size, bucket_max_support)
            entries[key] = {
                "variant": item["variant"],
                "level": item["level"],
                "label": item["label"],
                "null_kind": null_kind,
                "bucket_metadata": bucket_metadata,
                "bucket_max_denominator": bucket_max_denominator,
                "bucket_max_support_size": bucket_max_support,
                "q0_denom": q0_denom,
                "q_denoms": {q: stats.get("denom", 0) for q, stats in counts.q_stats.items()},
            }
    return {
        "item_count": len(items),
        "entry_count": len(entries),
        "entries": entries,
        "null_kind_coverage": sorted({entry["null_kind"] for entry in entries.values()}),
        "max_denominator": max_denominator,
        "max_support_size": max_support_size,
        "max_q0_denominator": max_q0_denom,
    }


def candidate_full_subset_metadata_guard(
    candidate_outputs: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    max_omitted = 0.0
    max_support_size = 0
    entries = manifest["entries"]
    if set(candidate_outputs) != set(entries):
        failures.append(
            {
                "reason": "key_mismatch",
                "missing_candidate": sorted(set(entries) - set(candidate_outputs)),
                "unexpected_candidate": sorted(set(candidate_outputs) - set(entries)),
            }
        )
    for key, expected in entries.items():
        out = candidate_outputs.get(key)
        if out is None:
            continue
        expected_meta = expected["bucket_metadata"]
        input_meta = out.get("input_bucket_metadata")
        batched_meta = out.get("candidate_batching_metadata")
        if input_meta != expected_meta or batched_meta != expected_meta:
            failures.append(
                {
                    "reason": "bucket_metadata_mismatch",
                    "key": key,
                    "expected": expected_meta,
                    "candidate_input": input_meta,
                    "candidate_batched": batched_meta,
                }
            )
        support_policy = out.get("support_policy") or {}
        max_omitted = max(max_omitted, float(support_policy.get("max_omitted_mass", 0.0)))
        max_support_size = max(max_support_size, int(support_policy.get("max_support_size", 0)))
    if max_omitted > GPU_NATIVE_PMF_CAPTURED_MASS_EPS:
        failures.append(
            {
                "reason": "support_policy_omitted_mass_exceeded",
                "max_omitted_mass": max_omitted,
                "epsilon": GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
            }
        )
    return {
        "pass": not failures,
        "failure_count": len(failures),
        "failures": failures[:20],
        "max_omitted_mass": max_omitted,
        "max_support_size": max_support_size,
    }


def _tiny_empirical_item_coverage(item: dict[str, Any]) -> dict[str, Any]:
    counts: CountAccumulator = item["counts"]
    q_denoms = {q: stats.get("denom", 0) for q, stats in counts.q_stats.items()}
    nonzero_q = sorted(q for q, denom in q_denoms.items() if denom > 0)
    buckets = counts.buckets_global + counts.buckets_rowq
    return {
        "q_denoms": q_denoms,
        "nonzero_q": nonzero_q,
        "has_q0": q_denoms.get("0", 0) > 0,
        "mixed_q": len(nonzero_q) >= 2,
        "bucket_count": len(buckets),
        "sum_total": sum(bucket.total for bucket in buckets),
        "max_bucket_total": max((bucket.total for bucket in buckets), default=0),
        "max_support_size": max((_bucket_joint_support_span(bucket) for bucket in buckets), default=0),
    }


def collect_tiny_empirical_items(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    max_items: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[str, Any], dict[str, Any]]] = []
    for variant in CREDIT_VARIANTS:
        for label, counts in sorted(variant_count_sets[variant]["invocations"].items()):
            item = {"variant": variant, "level": "invocation", "label": label, "counts": counts}
            coverage = _tiny_empirical_item_coverage(item)
            coverage_score = int(coverage["has_q0"]) + int(coverage["mixed_q"])
            cost = int(coverage["sum_total"]) + int(coverage["max_support_size"])
            candidates.append(((-coverage_score, cost, coverage["max_bucket_total"], len(candidates)), item, coverage))
    if not candidates:
        return [], {"selected_count": 0, "reason": "no_invocation_candidates"}
    selected = sorted(candidates, key=lambda row: row[0])[: max(1, max_items)]
    items = [item for _key, item, _coverage in selected]
    selected_manifest = [
        {
            "variant": item["variant"],
            "level": item["level"],
            "label": item["label"],
            "coverage": coverage,
        }
        for _key, item, coverage in selected
    ]
    return items, {
        "selected_count": len(items),
        "max_items": max_items,
        "selection_rule": "prefer q0+mixed-q coverage, then smallest CPU-cost score",
        "selected": selected_manifest,
    }


def write_stage_artifact(args: argparse.Namespace, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(args.out_dir) / name
    digest = write_json_with_sha(path, payload)
    public_path = None
    if args.public_out_dir:
        public_dir = Path(args.public_out_dir)
        public_dir.mkdir(parents=True, exist_ok=True)
        public_path = public_dir / name
        public_path.write_bytes(path.read_bytes())
        public_path.with_suffix(public_path.suffix + ".sha256").write_text(
            path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return {"path": str(path), "sha256": digest, "public_path": str(public_path) if public_path else None}


def build_cited_cpu_speed_gate(stage2_wall_seconds: float, required_speedup_floor: float) -> dict[str, Any]:
    evidence = [
        {
            "name": "current_gpu_native_stop",
            "elapsed_wall_seconds_lower_bound": GPU_NATIVE_CURRENT_CPU_LOWER_BOUND_SECONDS,
            "receipt_msg": "1780242593805-dfaeaec5",
            "subset_relation_to_stage2": "same current collect_null_profile_items full subset; stopped inside CPU oracle prefix, therefore subset of Stage-2 full intended null subset",
            "valid_for_ratio": True,
        },
        {
            "name": "prior_hybrid_stop",
            "elapsed_wall_seconds_lower_bound": GPU_NATIVE_PRIOR_CPU_LOWER_BOUND_SECONDS,
            "receipt_msg": "1780239060814-46159e2d",
            "subset_relation_to_stage2": "same slice-1 shared-label parity/profile subset family; conservative CPU lower-bound corroboration",
            "valid_for_ratio": True,
        },
    ]
    valid_bounds = [item["elapsed_wall_seconds_lower_bound"] for item in evidence if item["valid_for_ratio"]]
    selected_lower_bound = max(valid_bounds) if valid_bounds else 0.0
    speedup = (selected_lower_bound / stage2_wall_seconds) if stage2_wall_seconds > 0 else 0.0
    return {
        "required_speedup_floor": required_speedup_floor,
        "stage2_candidate_full_wall_seconds": stage2_wall_seconds,
        "selected_cpu_lower_bound_seconds": selected_lower_bound,
        "candidate_speedup_lower_bound": speedup,
        "passes_speed_gate": bool(valid_bounds) and speedup >= required_speedup_floor,
        "evidence": evidence,
        "subset_containment_required": True,
        "non_comparable_subsets_invalidate_ratio": True,
    }


def run_null_parity_profile(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
    bars: Bars,
    prereg_path: Path,
    prereg_sha: str,
) -> tuple[str, dict[str, Any]]:
    items = collect_null_profile_items(
        variant_count_sets,
        max_invocations_per_variant=args.null_parity_max_invocations_per_variant,
    )
    metadata_manifest = build_full_subset_bucket_metadata_manifest(items)
    analytic_parity = run_analytic_pmf_parity(
        args.null_aggregation_device,
        variant_count_sets,
        max_invocations_per_variant=args.null_parity_max_invocations_per_variant,
    )
    stage1 = {
        "stage": 1,
        "name": "correctness_first",
        "prereg": {"path": str(prereg_path), "sha256": prereg_sha},
        "source_parent_commit": "e7aa7fe12c6f52297478b457e2743e959e71137f",
        "reference_pmf_function": REFERENCE_PMF_FUNCTION,
        "candidate_pmf_function": CANDIDATE_PMF_FUNCTION,
        "reference_candidate_independent": REFERENCE_PMF_FUNCTION != CANDIDATE_PMF_FUNCTION,
        "analytic_pmf": analytic_parity,
        "bucket_metadata_manifest": metadata_manifest,
        "exact_backend_certified": bool(analytic_parity["exact_backend_certified"]),
        "bounded_reference_certified": bool(analytic_parity["bounded_reference_certified"]),
        "explicit_backend_validated_for_science": bool(
            analytic_parity["explicit_backend_validated_for_science"]
        ),
        "default_flip_eligible": bool(analytic_parity["default_flip_eligible"]),
        "q0_exact_coverage": analytic_parity["q0_exact_coverage"],
        "pass": bool(analytic_parity["pass"])
        and REFERENCE_PMF_FUNCTION != CANDIDATE_PMF_FUNCTION
        and set(metadata_manifest["null_kind_coverage"]) == {"global_permutation", "row_q_preserving"},
    }
    stage1["artifact"] = write_stage_artifact(args, "gpu_native_stage1_correctness.json", stage1)

    candidate_outputs, candidate_timing = _run_null_backend_items(
        items,
        backend=NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
        permutations=args.null_permutations,
        null_seed=args.null_seed,
        aggregation_device=args.null_aggregation_device,
    )
    stage2_guard = candidate_full_subset_metadata_guard(candidate_outputs, metadata_manifest)
    stage2_wall = float(candidate_timing["wall_total"])
    stage2_under_t = stage2_wall <= args.gpu_native_stage2_max_seconds
    explicit_backend_validated = bool(stage1["pass"]) and bool(stage2_guard["pass"]) and stage2_under_t
    stage2 = {
        "stage": 2,
        "name": "candidate_only_full_subset_operational_unblock",
        "backend": NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
        "cpu_oracle_exercised": False,
        "full_subset_item_count": len(items),
        "wall_seconds": stage2_wall,
        "max_seconds": args.gpu_native_stage2_max_seconds,
        "completed_under_t": stage2_under_t,
        "timing": candidate_timing,
        "metadata_guard": stage2_guard,
        "explicit_backend_validated_for_science": explicit_backend_validated,
    }
    stage2["artifact"] = write_stage_artifact(args, "gpu_native_stage2_candidate_full.json", stage2)

    tiny_items, tiny_manifest = collect_tiny_empirical_items(variant_count_sets, max_items=1)
    if explicit_backend_validated and tiny_items:
        tiny_cpu_outputs, tiny_cpu_timing = _run_null_backend_items(
            tiny_items,
            backend=NULL_BACKEND_CPU_LOCKED,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )
        tiny_candidate_outputs, tiny_candidate_timing = _run_null_backend_items(
            tiny_items,
            backend=NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )
        empirical_parity = _compare_distributional_null_outputs(
            tiny_cpu_outputs,
            tiny_candidate_outputs,
            tiny_items,
            bars=bars,
            abs_tol=args.null_parity_abs_tol,
        )
        stage3_status = "pass" if empirical_parity["pass"] else "empirical_confirm_fail_or_inconclusive_default_deferred"
        stage3_timings = {"cpu_locked": tiny_cpu_timing, "candidate": tiny_candidate_timing}
    else:
        empirical_parity = {
            "pass": False,
            "skipped": True,
            "reason": "stage1_stage2_not_validated_or_no_tiny_item",
            "abs_tol": args.null_parity_abs_tol,
        }
        stage3_status = "skipped_default_deferred"
        stage3_timings = {}
    stage3 = {
        "stage": 3,
        "name": "tiny_empirical_confirm_default_blocking_only",
        "semantics": (
            "A fail/timeout blocks default and triggers investigation, but does not erase "
            "Stage1+2 explicit-backend validation unless it reports a concrete analytic-proof counterexample."
        ),
        "tiny_subset": tiny_manifest,
        "empirical": empirical_parity,
        "timings": stage3_timings,
        "concrete_analytic_counterexample": False,
        "status": stage3_status,
    }
    stage3["artifact"] = write_stage_artifact(args, "gpu_native_stage3_tiny_empirical.json", stage3)

    speed_gate = build_cited_cpu_speed_gate(stage2_wall, args.null_speedup_floor)
    q0_denoms = [
        item["counts"].q_stats.get("0", {}).get("denom", 0)
        for item in items
        if item["level"] == "invocation"
    ]
    default_route_proof = {
        "required_before_default_flip": True,
        "default_backend": DEFAULT_NULL_BACKEND,
        "candidate_backend": NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
        "null_backend_explicit": bool(getattr(args, "null_backend_explicit", False)),
        "no_null_backend_cli_route_exercised": not bool(getattr(args, "null_backend_explicit", False)),
        "passes": (
            DEFAULT_NULL_BACKEND == NULL_BACKEND_GPU_NATIVE_COUNTS_PMF
            and args.null_backend == DEFAULT_NULL_BACKEND
            and not bool(getattr(args, "null_backend_explicit", False))
        ),
        "failure_reason": None,
    }
    if not default_route_proof["passes"]:
        default_route_proof["failure_reason"] = "default backend is not the GPU-native candidate or CLI overrode --null-backend"
    stage4 = {
        "stage": 4,
        "name": "default_ceremony_deferrable",
        "speed_gate": speed_gate,
        "default_cli_route_proof": default_route_proof,
        "cpu_oracle_reentry_allowed": False,
        "passes": bool(speed_gate["passes_speed_gate"]) and bool(default_route_proof["passes"]) and bool(empirical_parity["pass"]),
    }
    stage4["artifact"] = write_stage_artifact(args, "gpu_native_stage4_default_ceremony.json", stage4)

    if not stage1["pass"] or not stage2_guard["pass"] or bool(stage3["concrete_analytic_counterexample"]):
        terminal = "gpu_native_null_parity_fail"
    elif explicit_backend_validated and bool(empirical_parity["pass"]) and bool(stage4["passes"]):
        terminal = "gpu_native_null_parity_default_enabled"
    elif explicit_backend_validated and not bool(speed_gate["passes_speed_gate"]) and bool(empirical_parity["pass"]):
        terminal = "gpu_native_null_parity_speedup_insufficient_cpu_default_retained"
    elif explicit_backend_validated:
        terminal = "gpu_native_null_parity_explicit_validated_default_deferred"
    else:
        terminal = "gpu_native_null_parity_speedup_insufficient_cpu_default_retained"

    profile = {
        "terminal": terminal,
        "backend_names": {
            "oracle": NULL_BACKEND_CPU_LOCKED,
            "candidate": NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
            "default": DEFAULT_NULL_BACKEND,
        },
        "explicit_backend_validated_for_science": explicit_backend_validated,
        "speed_gate": speed_gate,
        "benchmark_protocol": {
            "cpu_oracle_full_subset_repeats": 0,
            "candidate_full_subset_repeats": 1,
            "speed_cpu_repeats": 0,
            "speed_candidate_repeats": 0,
            "permutations": args.null_permutations,
            "aggregation_device": _torch_device_name(args.null_aggregation_device),
            "stage2_max_seconds": args.gpu_native_stage2_max_seconds,
        },
        "subset": {
            "item_count": len(items),
            "max_invocations_per_variant": args.null_parity_max_invocations_per_variant,
            "q0_invocation_denom_min": min(q0_denoms) if q0_denoms else None,
            "q0_invocation_denom_max": max(q0_denoms) if q0_denoms else None,
            "levels": sorted({item["level"] for item in items}),
        },
        "parity": {
            "standard": "Stage1 analytic PMF primary guard plus Stage3 tiny empirical confirm",
            "pass": bool(stage1["pass"]) and (bool(empirical_parity["pass"]) or explicit_backend_validated),
            "analytic_pmf": analytic_parity,
            "empirical": empirical_parity,
        },
        "timings": {
            "full_subset_candidate_wall_seconds": [candidate_timing["wall_total"]],
            "full_subset_candidate_gpu_sampler_seconds": [candidate_timing["gpu_sampler"]],
        },
        "support_policy": {
            "captured_mass_epsilon": GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
            "full_support_guard": GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
            "full_subset_candidate": {
                "max_omitted_mass": max(
                    (
                        float(out.get("support_policy", {}).get("max_omitted_mass", 0.0))
                        for out in candidate_outputs.values()
                    ),
                    default=0.0,
                ),
                "max_support_size": max(
                    (
                        int(out.get("support_policy", {}).get("max_support_size", 0))
                        for out in candidate_outputs.values()
                    ),
                    default=0,
                ),
            },
        },
        "default_cli_route_proof": default_route_proof,
        "stages": {
            "stage1_correctness": stage1,
            "stage2_candidate_full": stage2,
            "stage3_tiny_empirical": stage3,
            "stage4_default_ceremony": stage4,
        },
        "bars": bars.__dict__,
    }
    return terminal, profile


def terminal_from_summaries(
    *,
    global_summary: dict[str, Any],
    family_summaries: dict[str, dict[str, Any]],
    invocation_summaries: dict[str, dict[str, Any]],
    bars: Bars,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    invalid = False
    mixed = False
    route_death = False
    credit_mismatch = False

    for label, summary in invocation_summaries.items():
        if summary["denom"] < bars.min_projected_denom:
            invalid = True
            reasons.append(f"{label}: projected denominator {summary['denom']} < {bars.min_projected_denom}")
        if summary["route"]["active_output_count"] < bars.min_active_outputs:
            invalid = True
            reasons.append(
                f"{label}: active outputs {summary['route']['active_output_count']} < {bars.min_active_outputs}"
            )
        q0 = summary["q_level"].get("0", {})
        q0_denom = q0.get("denom", 0) or 0
        if q0_denom < bars.min_q0_projected_denom_valid:
            invalid = True
            reasons.append(
                f"{label}: q=0 projected denominator {q0_denom} < {bars.min_q0_projected_denom_valid}"
            )
        elif q0_denom < bars.min_q0_projected_denom_plausible:
            mixed = True
            reasons.append(
                f"{label}: q=0 projected denominator {q0_denom} < {bars.min_q0_projected_denom_plausible}"
            )
        if not summary["pass_sign"]:
            mixed = True
            reasons.append(
                f"{label}: sign agreement {summary['agreement']:.4f} < threshold {summary['threshold']:.4f}"
            )
        route = summary["route"]
        if route["dead_active_output_rate"] >= bars.stratum_dead_rate_route_death:
            route_death = True
            reasons.append(
                f"{label}: dead-active-output rate {route['dead_active_output_rate']:.4f} >= "
                f"{bars.stratum_dead_rate_route_death}"
            )
        elif route["dead_active_output_count"] > 0 or route["admissible_p01"] < bars.route_p01_floor:
            mixed = True
            reasons.append(
                f"{label}: route weakness dead={route['dead_active_output_count']} "
                f"p01={route['admissible_p01']}"
            )
        if route["admissible_p01"] == 0:
            route_death = True
            reasons.append(f"{label}: p01 admissible routes is zero")
        if route["admissible_median"] < bars.route_median_floor:
            mixed = True
            reasons.append(
                f"{label}: median admissible routes {route['admissible_median']} < {bars.route_median_floor}"
            )

    if not global_summary["pass_sign"]:
        credit_mismatch = True
        reasons.append(
            f"global: sign agreement {global_summary['agreement']:.4f} < "
            f"threshold {global_summary['threshold']:.4f}"
        )
    for label, summary in family_summaries.items():
        if not summary["pass_sign"]:
            credit_mismatch = True
            reasons.append(
                f"{label}: family sign agreement {summary['agreement']:.4f} < "
                f"threshold {summary['threshold']:.4f}"
            )
        if summary["route"]["dead_active_output_rate"] >= bars.family_dead_rate_route_death:
            route_death = True
            reasons.append(
                f"{label}: family dead-active-output rate "
                f"{summary['route']['dead_active_output_rate']:.4f} >= {bars.family_dead_rate_route_death}"
            )

    if invalid:
        return "diagnostic_reference_invalid", reasons
    if route_death:
        return "route_death_persists_at_scale", reasons
    if credit_mismatch:
        return "credit_direction_mismatch", reasons
    if mixed:
        return "mixed_requires_scale_hedge", reasons
    return "locked_bars_pass", reasons


def variant_locked_bars_pass(variant_result: dict[str, Any]) -> bool:
    return variant_result["bar_terminal"] == "locked_bars_pass"


def terminal_from_variant_results(variant_results: dict[str, dict[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    for variant, variant_result in variant_results.items():
        bar_terminal = variant_result["bar_terminal"]
        if bar_terminal in {"diagnostic_reference_invalid", "route_death_persists_at_scale"}:
            reasons.append(f"{variant}: {bar_terminal}")
            reasons.extend(f"{variant}: {reason}" for reason in variant_result["bar_reasons"][:20])
            checks["invalid_variant"] = variant
            return "diagnostic_reference_invalid", reasons, checks

    strict_agreement = float(variant_results["strict"]["global"]["agreement"])
    strict_delta = abs(strict_agreement - STRICT_REPRODUCTION_EXPECTED)
    strict_passes = variant_locked_bars_pass(variant_results["strict"])
    checks["strict_reproduction"] = {
        "expected": STRICT_REPRODUCTION_EXPECTED,
        "tolerance_abs": STRICT_REPRODUCTION_TOL,
        "observed": strict_agreement,
        "delta_abs": strict_delta,
        "within_tolerance": strict_delta <= STRICT_REPRODUCTION_TOL,
        "locked_bars_pass": strict_passes,
    }
    if strict_delta > STRICT_REPRODUCTION_TOL:
        reasons.append(
            f"strict: global agreement {strict_agreement:.6f} deviates from committed "
            f"{STRICT_REPRODUCTION_EXPECTED:.6f} by {strict_delta:.6f} > {STRICT_REPRODUCTION_TOL}"
        )
        return "diagnostic_reference_invalid", reasons, checks
    if strict_passes:
        reasons.append("strict: reproduction sentinel crossed locked bars; this indicates harness drift")
        return "diagnostic_reference_invalid", reasons, checks

    if not variant_locked_bars_pass(variant_results["full_magnitude_ceiling"]):
        reasons.append("full_magnitude_ceiling: failed locked bars; FP reference/projection is suspect")
        reasons.extend(variant_results["full_magnitude_ceiling"]["bar_reasons"][:50])
        return "diagnostic_reference_invalid", reasons, checks

    if variant_locked_bars_pass(variant_results["pow2_bucket"]):
        reasons.append("pow2_bucket: passed locked bars while strict reproduced the committed failing baseline")
        return "pow2_magnitude_sufficient", reasons, checks

    if variant_locked_bars_pass(variant_results["fp16_groupwise"]):
        reasons.append("fp16_groupwise: passed locked bars while strict/pow2 did not")
        return "fp16_groupwise_credit_sufficient_proxy", reasons, checks

    reasons.append("pow2_bucket and fp16_groupwise both failed locked bars while full magnitude ceiling passed")
    return "tested_lowbit_magnitude_insufficient", reasons, checks


def assert_schedule_excluded_no_grad_count(excluded: list[dict[str, Any]], *, expected: int = 96) -> None:
    if len(excluded) != expected:
        raise DiagnosticInvalid(f"schedule_excluded_no_grad count {len(excluded)} != {expected}")
    bad = [item for item in excluded if item.get("reason") != "schedule_excluded_no_grad"]
    if bad:
        labels = ", ".join(str(item.get("label")) for item in bad[:5])
        raise DiagnosticInvalid(f"schedule_excluded entries have non-matching reasons: {labels}")


def _build_model_from_ckpt(ckpt: dict[str, Any], device: str):
    from calm.hrm_text_158.config import HierarchicalReasoningModelConfig, LMHeadConfig
    from calm.hrm_text_158.curriculum.broad_tokenizer import BROAD_NORMALIZER_VERSION, BroadTokenizer
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer
    from calm.hrm_text_158.hrm import HierarchicalReasoningModel
    from calm.hrm_text_158.lm_head import LMHead

    config = ckpt["config"]
    normalizer_version = config["gsm8k_normalizer_version"]
    if normalizer_version == BROAD_NORMALIZER_VERSION:
        tok = BroadTokenizer()
        if list(config["gsm8k_char_vocab"]) != tok.vocab_as_list():
            raise DiagnosticInvalid("BroadTokenizer vocab mismatch against checkpoint config")
    else:
        tok = Gsm8kTokenizer.from_metadata(
            vocab_list=config["gsm8k_char_vocab"],
            normalizer_version=normalizer_version,
        )
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=config["max_seq_len"],
        n_layers=config["n_layers"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        expansion=config["expansion"],
        H_cycles=config["H_cycles"],
        L_cycles=config["L_cycles"],
        half_layers=config["half_layers"],
        bp_warmup_ratio=config["bp_warmup_ratio"],
        bp_min_steps=config["bp_min_steps"],
        bp_max_steps=config["bp_max_steps"],
        norm_type=config["norm_type"],
        norm_eps=config["norm_eps"],
        rope_theta=config["rope_theta"],
        attn_type=config["attn_type"],
        init_type=config["init_type"],
        pos_emb_type=config["pos_emb_type"],
        use_ternary_bulk=config.get("use_ternary_bulk", False),
    )
    hrm = HierarchicalReasoningModel(cfg)
    model = LMHead(hrm, LMHeadConfig(vocab_size=config["vocab_size"])).to(device)
    load_result = model.load_state_dict(ckpt["model_state"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise DiagnosticInvalid(
            f"state_dict mismatch missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, tok, config


def load_l0c1_rows(seed: int) -> list[tuple[str, int, str]]:
    from calm.hrm_text_158.curriculum.language_supports import build_l0c1_support

    support = build_l0c1_support(seed=seed)
    rows = support["L0c1"]
    if len(rows) != 121:
        raise DiagnosticInvalid(f"L0c1 support count {len(rows)} != 121")
    return rows


def collate_encoded_rows(
    rows: list[tuple[str, int, str]],
    tok: Any,
    *,
    device: str,
) -> dict[str, Tensor]:
    encoded: list[tuple[list[int], int]] = [tok.encode_example(q, expected) for q, expected, _ in rows]
    max_len = max(len(ids) for ids, _ in encoded)
    if max_len < 2:
        raise DiagnosticInvalid("encoded batch max_len < 2")
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    seps: list[int] = []
    for ids, sep_pos in encoded:
        ids_padded = list(ids) + [tok.pad_id] * (max_len - len(ids))
        ids_t = torch.tensor(ids_padded, dtype=torch.long)
        inp = ids_t[:-1].contiguous()
        lab = torch.full_like(inp, IGNORE_LABEL_ID)
        lab[sep_pos:] = ids_t[sep_pos + 1 :]
        for pos in range(sep_pos + 1, len(ids)):
            if ids[pos] == tok.eos_id:
                if pos < lab.shape[0]:
                    lab[pos:] = IGNORE_LABEL_ID
                break
        inputs.append(inp)
        labels.append(lab)
        seps.append(sep_pos)
    input_t = torch.stack(inputs, dim=0).to(device)
    label_t = torch.stack(labels, dim=0).to(device)
    sep_t = torch.tensor(seps, dtype=torch.long, device=device)
    B, L = input_t.shape
    pos_t = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    return {"inputs": input_t, "labels": label_t, "sep_positions": sep_t, "position_ids": pos_t}


def batched(rows: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_prereg(
    *,
    args: argparse.Namespace,
    ckpt_path: Path,
    checkpoint_sha256_before: str,
) -> dict[str, Any]:
    bars = Bars()
    return {
        "task_id": TASK_ID,
        "created_unix": int(time.time()),
        "source_parent_commit": "e7aa7fe12c6f52297478b457e2743e959e71137f",
        "dispatch_msg": "1780239448199-2d583c60",
        "plan_msg": "1780239762459-42eb99eb",
        "fold_msgs": [
            "1780239779212-e70eee07",
            "1780239791830-38c1c154",
            "1780239855284-d6b2d25d",
            "1780239871440-8ec2d087",
            "1780239882455-bb9eb191",
            "1780239896457-6e47e248",
            "1780239900650-9a218a1d",
        ],
        "runner_watcher_provenance_msg": "1780233326665-f3e3d2fa",
        "implement_gate_msg": "1780239930504-cf146583",
        "implement_gate_confirmation_msg": "1780239968234-f3c18df3",
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256_before": checkpoint_sha256_before,
            "read_only_invariant": "sha256_before must equal sha256_after; no optimizer/no param write/no .pt writes",
        },
        "run": {
            "support": "L0c1 full finite support",
            "support_seed": args.support_seed,
            "bp_steps": args.bp_steps,
            "batch_size": args.batch_size,
            "device": args.device,
            "null_permutations": args.null_permutations,
            "null_seed": args.null_seed,
            "max_rows": args.max_rows,
        },
        "null_backend": {
            "default": DEFAULT_NULL_BACKEND,
            "oracle": NULL_BACKEND_CPU_LOCKED,
            "candidate": NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
            "intended_default_if_all_gates_pass": NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
            "candidate_explicit_name": (
                "GPU-native counts-PMF inverse-CDF sampler for the finite-population "
                "two-stage null; distributional parity to cpu_locked, not bit-identical RNG replay"
            ),
            "aggregation_device": _torch_device_name(args.null_aggregation_device),
            "profile_only": args.null_parity_profile_only,
            "speedup_floor": args.null_speedup_floor,
            "empirical_abs_tolerance": args.null_parity_abs_tol,
            "analytic_tv_bound": GPU_NATIVE_PMF_TV_BOUND,
            "analytic_cdf_bound": GPU_NATIVE_PMF_CDF_BOUND,
            "captured_mass_epsilon": GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
            "full_support_guard": GPU_NATIVE_PMF_FULL_SUPPORT_GUARD,
            "reference_joint_work_budget": GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET,
            "reference_chunk_cell_budget": GPU_NATIVE_REFERENCE_CHUNK_CELL_BUDGET,
            "bounded_sample_count": GPU_NATIVE_BOUNDED_SAMPLE_COUNT,
            "bounded_sample_confidence": GPU_NATIVE_BOUNDED_SAMPLE_CONFIDENCE,
            "bounded_sample_cdf_bound": GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND,
            "stage1_certification_tiers": {
                "exact_backend_certified": "true only when every accepted fixture/path clears the exact 1e-5 TV/CDF envelope",
                "bounded_reference_certified": "true when a tier-2 bounded-sampled bucket clears its predeclared sampling-aware CDF bound",
                "explicit_backend_validated_for_science": (
                    "true when all non-fallback fixtures clear exact bounds, q0 exact coverage is present, "
                    "and any downgrade is only a flagged bounded-sampled giant bucket within its bound"
                ),
                "default_flip_eligible": "false under any tier-2 bounded fallback until exact/default-route proof is supplied",
            },
            "reference_pmf_function": REFERENCE_PMF_FUNCTION,
            "candidate_pmf_function": CANDIDATE_PMF_FUNCTION,
            "reference_candidate_independent_required": True,
            "scipy_cross_check_required_where_feasible": True,
            "stage1_real_corpus_required_winners": [
                "max_denominator",
                "max_support_size",
                "max_q0_denominator",
                "skew_tail_heavy",
                "global_permutation",
                "row_q_preserving",
            ],
            "stage2_candidate_full_max_seconds": args.gpu_native_stage2_max_seconds,
            "stage2_unblock_condition": (
                "Stage1 analytic correctness pass + Stage2 candidate-only full subset wall<=T "
                "+ support/bucket/batching metadata guards pass"
            ),
            "stage3_semantics": (
                "tiny CPU empirical confirm blocks default and triggers investigation on fail/timeout, "
                "but does not erase Stage1+2 explicit-backend validation unless it reports a concrete "
                "analytic-proof counterexample"
            ),
            "cited_cpu_lower_bound_seconds": GPU_NATIVE_CURRENT_CPU_LOWER_BOUND_SECONDS,
            "cited_cpu_lower_bound_corroboration_seconds": GPU_NATIVE_PRIOR_CPU_LOWER_BOUND_SECONDS,
            "cited_cpu_subset_containment_required": True,
            "speed_cpu_repeats": args.null_speed_cpu_repeats,
            "speed_candidate_repeats": args.null_speed_candidate_repeats,
            "speed_cpu_denominator_label": "no live CPU speed denominator; cited CPU lower-bound subset containment only",
            "speed_max_invocations_per_variant": args.null_speed_max_invocations_per_variant,
            "sampler_bound_fraction_threshold": NULL_SAMPLER_BOUND_FRACTION,
            "parity_max_invocations_per_variant": args.null_parity_max_invocations_per_variant,
            "default_cli_route_proof_condition": (
                "cpu_locked remains the committed default through this diagnostic. Only after analytic parity, "
                "candidate-only full-subset operational proof, empirical/default ceremony, cited speed, "
                "and an explicit no---null-backend route proof all pass "
                "may a final diff flip DEFAULT_NULL_BACKEND to gpu_native_counts_pmf."
            ),
        },
        "bars": bars.__dict__,
        "strict_reproduction": {
            "expected_global_agreement": STRICT_REPRODUCTION_EXPECTED,
            "tolerance_abs": STRICT_REPRODUCTION_TOL,
            "terminal_on_deviation": "diagnostic_reference_invalid",
            "sentinel_only": True,
        },
        "null_label_scheme": "slice1_shared_seed_labels",
        "null_label_scheme_detail": (
            "all variants reuse global/family/invocation CountAccumulator labels from slice 1; "
            "variant names do not enter deterministic null seeds"
        ),
        "credit_variants": {
            "order": list(CREDIT_VARIANTS),
            "null_seed_label_scheme": "all variants reuse the slice-1 labels (global/family/invocation) so deterministic null draws stay shared",
            "strict": {
                "formula": "-sum_t sign(dL/dy_i) * sign(x_j)",
                "role": "reproduction sentinel only; never a success path",
            },
            "pow2_bucket": {
                "formula": "-sum_t pow2(dL/dy_i) * pow2(x_j)",
                "pow2": {
                    "round_mode": POW2_ROUND_MODE,
                    "exp_min": POW2_EXP_MIN,
                    "exp_max": POW2_EXP_MAX,
                    "zero_behavior": "zero remains zero",
                },
                "magnitude_enters": "weights signed credit before projection",
            },
            "fp16_groupwise": {
                "formula": "-sum_t (sign(dL/dy_i)*g_scale[t,out_group(i)]) * (sign(x_j)*x_scale[t,in_group(j)])",
                "group_size": FP16_GROUPWISE_GROUP_SIZE,
                "scale_stat": FP16_GROUPWISE_SCALE_STAT,
                "scale_dtype": "fp16 quantized then fp32 accumulated",
                "partition": "local within already-sliced projection group; no q/k/v/gate/up/down crossing",
                "zero_behavior": "all-zero groups produce scale 0; no epsilon floor",
                "magnitude_enters": "weights signed credit before projection",
                "proxy_only": True,
            },
            "full_magnitude_ceiling": {
                "formula": "-weighted_grad",
                "role": "sanity ceiling; failure makes terminal diagnostic_reference_invalid",
            },
        },
        "locked_tightenings": [
            "recurrence-aware 160 gradient-enabled invocation strata at bp_steps=5; schedule-excluded no-grad L invocations listed separately",
            "credit denominator uses all local positions with nonzero BitLinear output grad, with prefix/response breakdown",
            "sign agreement stratified by q=-1/q=0/q=+1; q=0 projected denominator gates plausible",
            "row/output-channel-preserving q-bucket null added; score against stronger p99 null",
            "absolute floors raised: global>=0.65, family>=0.60, stratum/invocation>=0.55",
            "FP reference is the reused magnitude-aware STE master-weight gradient; all variants share denominators/projection/nulls",
            "all variants reuse slice-1 null seed labels; variant names do not alter deterministic null draws",
            "strict variant is a reproduction sentinel expected at 0.508805533381048 +/- 0.01, never a success path",
            "pow2 and fp16_groupwise magnitudes weight signed credit before projection, not tie-only resolution",
            "fp16_groupwise uses local group size 128, mean(abs), fp16-quantized scales, no epsilon floor",
            "full_magnitude_ceiling must pass locked bars or terminal is diagnostic_reference_invalid",
            "assert len(schedule_excluded_no_grad)==96",
            "assert BitLinear cached/native flags false/None; no freeze_for_inference or enable_native_train",
            "F1: analytic PMF parity is the primary math guard and is persisted before any CPU sampling: TV distance <=1e-5 and max CDF delta <=1e-5 per fixture/stage and joint x_pos+conditional-x_neg total-match distribution vs independent exact scipy/lgamma hypergeometric reference; reference_pmf_function and candidate_pmf_function must differ; scipy cross-check runs where feasible or records explicit fallback; real full-subset-derived extremes named for max denominator, max support size, max q0 denominator, skew/tail-heavy, and both null kinds; fp64 where needed; fail closed, no ad-hoc tolerance loosening",
            "F2: captured-mass truncation epsilon=1e-6 per bucket/draw stage; full legal support when <= guard; CDF-quantile/captured-mass window for oversized supports; fail closed to diagnostic_reference_invalid if guard cannot meet epsilon; report support_policy",
            "F2b: reference joint accumulation preserves x_neg|x_pos dependence via conditional weighted scatter, never independent convolution; predeclared joint-work budget self-limits giant fixtures; bounded-sampled fallback has a separate sampling-aware CDF bound and cannot be mistaken for exact/default certification",
            "F3: empirical gating parity is tiny confirm/default ceremony only: prefer q0+mixed-q actual-data invocation, <=1 invocation, hard record mean/p95/p99 abs delta <=0.01 and derived threshold/gating decision per item/null_kind; fail/timeout blocks default and triggers investigation but does not erase Stage1+2 explicit-backend validation unless it reports a concrete analytic-proof counterexample",
            "F4: row-q hard proof in the sampling path: same bucket_count, same sum(total), and same ordered bucket-total vector/hash per item/null_kind before and after candidate batching; any coalescing/flattening is a parity fail/diagnostic even if p99 matches",
            "F5: no live CPU speed denominator: Stage2 candidate-only full intended null subset must complete under T=600s for explicit science use; default speed proof uses cited CPU lower-bound evidence only when subset containment is stated, >=1.25x lower-bound speedup, and no---null-backend default CLI proof after a future flip; non-comparable subsets invalidate the speed ratio",
            "F6: cpu_locked never removed (oracle/fallback); terminal set locked; two-file scope; checkpoint sha before==after; no train/optimizer/param/.pt; prereg sha-pinned before screen; /tmp+box artifacts only",
        ],
        "credit_terminal_labels": list(CREDIT_TERMINAL_LABELS),
        "legacy_null_terminal_labels": list(NULL_PARITY_TERMINAL_LABELS),
        "terminal_labels": list(GPU_NATIVE_NULL_TERMINAL_LABELS),
    }


def load_or_write_prereg(args: argparse.Namespace, ckpt_path: Path, ckpt_sha: str) -> tuple[dict[str, Any], Path, str]:
    if args.prereg_path:
        prereg_path = Path(args.prereg_path)
        data = prereg_path.read_bytes()
        prereg = json.loads(data.decode("utf-8"))
        prereg_sha = sha256_bytes(data)
        expected = prereg.get("checkpoint", {}).get("sha256_before")
        if expected != ckpt_sha:
            raise IntegrityFailure(
                f"prereg checkpoint sha {expected} != current before-run sha {ckpt_sha}"
            )
        return prereg, prereg_path, prereg_sha
    out_dir = Path(args.out_dir)
    prereg_path = out_dir / "prereg.json"
    prereg = build_prereg(args=args, ckpt_path=ckpt_path, checkpoint_sha256_before=ckpt_sha)
    prereg_sha = write_json_with_sha(prereg_path, prereg)
    return prereg, prereg_path, prereg_sha


def verify_weight_grad_reconstruction(model: nn.Module, tracker: CreditHookTracker) -> dict[str, Any]:
    by_module: dict[str, Tensor] = {}
    for agg in tracker.aggregates.values():
        if agg.weighted_grad is None:
            continue
        full = by_module.setdefault(
            agg.module_name,
            torch.zeros_like(dict(model.named_modules())[agg.module_name].weight.detach().cpu(), dtype=torch.float32),
        )
        full[agg.group_start:agg.group_end] += agg.weighted_grad
    checks: dict[str, Any] = {}
    modules = dict(model.named_modules())
    max_abs = 0.0
    for name, weighted in by_module.items():
        grad = modules[name].weight.grad
        if grad is None:
            raise DiagnosticInvalid(f"{name}: missing autograd weight.grad for FP-reference check")
        diff = (weighted - grad.detach().to(torch.float32).cpu()).abs()
        module_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, module_max)
        checks[name] = {"max_abs_diff": module_max}
    return {"max_abs_diff": max_abs, "per_module": checks}


def run_diagnostic(args: argparse.Namespace) -> tuple[str, dict[str, Any], Path]:
    repo_root = Path.cwd()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    if not ckpt_path.exists():
        raise DiagnosticInvalid(f"checkpoint not found: {ckpt_path}")

    ckpt_sha_before = sha256_file(ckpt_path)
    prereg, prereg_path, prereg_sha = load_or_write_prereg(args, ckpt_path, ckpt_sha_before)
    print(f"[credit-bridge] prereg_path={prereg_path}", flush=True)
    print(f"[credit-bridge] prereg_sha256={prereg_sha}", flush=True)
    if args.prereg_only:
        return "prereg_only", {"prereg": prereg, "prereg_sha256": prereg_sha}, prereg_path

    if args.device == "cuda" and not torch.cuda.is_available():
        raise DiagnosticInvalid("requested cuda but torch.cuda.is_available() is false")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model, tok, ckpt_config = _build_model_from_ckpt(ckpt, args.device)
    targets = find_target_bitlinears(model)
    if len(targets) != 32:
        raise DiagnosticInvalid(f"target BitLinear count {len(targets)} != 32")
    assert_runtime_bitlinear_flags(targets)

    rows = load_l0c1_rows(args.support_seed)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    total_valid = 0
    for batch_rows in batched(rows, args.batch_size):
        batch = collate_encoded_rows(batch_rows, tok, device=args.device)
        total_valid += int((batch["labels"] != IGNORE_LABEL_ID).sum().item())
    if total_valid <= 0:
        raise DiagnosticInvalid("support has zero valid response labels")

    model.zero_grad(set_to_none=True)
    tracker = CreditHookTracker(targets, bp_steps=args.bp_steps)
    tracker.install()
    losses: list[float] = []
    try:
        for batch_idx, batch_rows in enumerate(batched(rows, args.batch_size), start=1):
            batch = collate_encoded_rows(batch_rows, tok, device=args.device)
            tracker.begin_batch(batch["sep_positions"])
            _carry, loss, metrics = model(None, batch, bp_steps=args.bp_steps)
            if not torch.isfinite(loss).item():
                raise DiagnosticInvalid(f"non-finite loss at batch {batch_idx}: {loss}")
            valid = int((batch["labels"] != IGNORE_LABEL_ID).sum().item())
            scaled_loss = loss * (valid / total_valid)
            losses.append(float(loss.detach().cpu().item()))
            tracker.assert_batch_forward_complete()
            scaled_loss.backward()
            if batch_idx % max(1, args.progress_every) == 0:
                print(
                    f"[credit-bridge] processed batch {batch_idx} rows={min(batch_idx * args.batch_size, len(rows))}/"
                    f"{len(rows)} loss={losses[-1]:.6f}",
                    flush=True,
                )
    finally:
        tracker.remove()

    grad_check = verify_weight_grad_reconstruction(model, tracker)
    if grad_check["max_abs_diff"] > args.grad_reconstruction_atol:
        raise DiagnosticInvalid(
            f"FP reference reconstruction max_abs_diff {grad_check['max_abs_diff']:.6g} > "
            f"{args.grad_reconstruction_atol}"
        )

    q_by_module = {target.name: ternary_levels(target.module.weight).cpu() for target in targets}
    bars = Bars()
    expected_invocation_count = 160
    if len(tracker.aggregates) != expected_invocation_count:
        raise DiagnosticInvalid(
            f"gradient-enabled invocation strata {len(tracker.aggregates)} != {expected_invocation_count}"
        )

    excluded = [
        {"label": item.key.label, "module_name": item.module_name, "reason": item.reason}
        for item in tracker.schedule_excluded.values()
    ]
    assert_schedule_excluded_no_grad_count(excluded, expected=96)

    variant_results: dict[str, dict[str, Any]] = {}
    variant_count_sets: dict[str, dict[str, Any]] = {}
    for variant in CREDIT_VARIANTS:
        print(f"[credit-bridge] scoring variant={variant}", flush=True)
        invocation_counts: dict[str, CountAccumulator] = {}
        invocation_details: dict[str, dict[str, Any]] = {}
        family_counts: dict[str, CountAccumulator] = {}
        aggregate64_counts: dict[str, CountAccumulator] = {}
        global_counts = CountAccumulator(label="global")
        for key, agg in sorted(tracker.aggregates.items(), key=lambda kv: kv[0].label):
            q = q_by_module[agg.module_name]
            counts, detail = build_counts_for_invocation(key, agg, q, variant=variant)
            invocation_counts[key.label] = counts
            invocation_details[key.label] = detail
            family_counts.setdefault(
                key.family_label,
                CountAccumulator(label=key.family_label),
            ).merge(counts)
            aggregate64_counts.setdefault(
                key.aggregate64_label,
                CountAccumulator(label=key.aggregate64_label),
            ).merge(counts)
            global_counts.merge(counts)

        variant_count_sets[variant] = {
            "global": global_counts,
            "families": family_counts,
            "aggregate64": aggregate64_counts,
            "invocations": invocation_counts,
        }
        if args.null_parity_profile_only:
            continue

        print(f"[credit-bridge] summarizing variant={variant} invocation/family/global nulls", flush=True)
        invocation_summaries = {
            label: summarize_counts(
                counts,
                bars=bars,
                level="invocation",
                null_permutations=args.null_permutations,
                null_seed=args.null_seed,
                null_backend=args.null_backend,
                null_aggregation_device=args.null_aggregation_device,
            )
            | invocation_details[label]
            for label, counts in invocation_counts.items()
        }
        family_summaries = {
            label: summarize_counts(
                counts,
                bars=bars,
                level="family",
                null_permutations=args.null_permutations,
                null_seed=args.null_seed,
                null_backend=args.null_backend,
                null_aggregation_device=args.null_aggregation_device,
            )
            for label, counts in sorted(family_counts.items())
        }
        aggregate64_summaries = {
            label: summarize_counts(
                counts,
                bars=bars,
                level="aggregate64",
                null_permutations=args.null_permutations,
                null_seed=args.null_seed,
                null_backend=args.null_backend,
                null_aggregation_device=args.null_aggregation_device,
            )
            for label, counts in sorted(aggregate64_counts.items())
        }
        global_summary = summarize_counts(
            global_counts,
            bars=bars,
            level="global",
            null_permutations=args.null_permutations,
            null_seed=args.null_seed,
            null_backend=args.null_backend,
            null_aggregation_device=args.null_aggregation_device,
        )
        bar_terminal, bar_reasons = terminal_from_summaries(
            global_summary=global_summary,
            family_summaries=family_summaries,
            invocation_summaries=invocation_summaries,
            bars=bars,
        )
        variant_results[variant] = {
            "bar_terminal": bar_terminal,
            "bar_reasons": bar_reasons[:200],
            "global": global_summary,
            "families": family_summaries,
            "aggregate64": aggregate64_summaries,
            "invocations": invocation_summaries,
        }

    if args.null_parity_profile_only:
        print("[credit-bridge] running null parity/profile subset", flush=True)
        terminal, null_profile = run_null_parity_profile(
            variant_count_sets,
            args=args,
            bars=bars,
            prereg_path=prereg_path,
            prereg_sha=prereg_sha,
        )
        ckpt_sha_after = sha256_file(ckpt_path)
        if ckpt_sha_after != ckpt_sha_before:
            raise IntegrityFailure(
                f"checkpoint hash changed before={ckpt_sha_before} after={ckpt_sha_after}"
            )
        result = {
            "task_id": TASK_ID,
            "terminal": terminal,
            "terminal_reasons": [],
            "checkpoint": {
                "path": str(ckpt_path),
                "sha256_before": ckpt_sha_before,
                "sha256_after": ckpt_sha_after,
                "unchanged": ckpt_sha_before == ckpt_sha_after,
            },
            "prereg": {"path": str(prereg_path), "sha256": prereg_sha},
            "support": {
                "surface": "L0c1",
                "seed": args.support_seed,
                "rows": len(rows),
                "total_valid_response_labels": total_valid,
                "loss_mean": float(np.mean(losses)) if losses else None,
            },
            "model": {
                "hidden_size": ckpt_config.get("hidden_size"),
                "n_layers": ckpt_config.get("n_layers"),
                "half_layers": ckpt_config.get("half_layers"),
                "H_cycles": ckpt_config.get("H_cycles"),
                "L_cycles": ckpt_config.get("L_cycles"),
                "bp_steps": args.bp_steps,
                "target_bitlinear_count": len(targets),
            },
            "bars": bars.__dict__,
            "variant_order": list(CREDIT_VARIANTS),
            "null_parity_profile": null_profile,
            "schedule_excluded_no_grad": excluded,
            "fp_reference_reconstruction": grad_check,
            "read_only_invariants": {
                "no_optimizer": True,
                "no_torch_save": True,
                "no_param_write_intended": True,
                "cached_native_flags_asserted": True,
            },
        }
        return terminal, result, prereg_path

    terminal, reasons, terminal_checks = terminal_from_variant_results(variant_results)

    ckpt_sha_after = sha256_file(ckpt_path)
    if ckpt_sha_after != ckpt_sha_before:
        raise IntegrityFailure(
            f"checkpoint hash changed before={ckpt_sha_before} after={ckpt_sha_after}"
        )

    result = {
        "task_id": TASK_ID,
        "terminal": terminal,
        "terminal_reasons": reasons[:200],
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256_before": ckpt_sha_before,
            "sha256_after": ckpt_sha_after,
            "unchanged": ckpt_sha_before == ckpt_sha_after,
        },
        "prereg": {"path": str(prereg_path), "sha256": prereg_sha},
        "support": {
            "surface": "L0c1",
            "seed": args.support_seed,
            "rows": len(rows),
            "total_valid_response_labels": total_valid,
            "loss_mean": float(np.mean(losses)) if losses else None,
        },
        "model": {
            "hidden_size": ckpt_config.get("hidden_size"),
            "n_layers": ckpt_config.get("n_layers"),
            "half_layers": ckpt_config.get("half_layers"),
            "H_cycles": ckpt_config.get("H_cycles"),
            "L_cycles": ckpt_config.get("L_cycles"),
            "bp_steps": args.bp_steps,
            "target_bitlinear_count": len(targets),
        },
        "bars": bars.__dict__,
        "variant_order": list(CREDIT_VARIANTS),
        "terminal_checks": terminal_checks,
        "variants": variant_results,
        "global": variant_results["strict"]["global"],
        "families": variant_results["strict"]["families"],
        "aggregate64": variant_results["strict"]["aggregate64"],
        "invocations": variant_results["strict"]["invocations"],
        "schedule_excluded_no_grad": excluded,
        "fp_reference_reconstruction": grad_check,
        "read_only_invariants": {
            "no_optimizer": True,
            "no_torch_save": True,
            "no_param_write_intended": True,
            "cached_native_flags_asserted": True,
        },
    }
    return terminal, result, prereg_path


def mirror_artifacts(out_dir: Path, public_out_dir: Path) -> None:
    if not public_out_dir:
        return
    public_out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*"):
        if path.is_file():
            target = public_out_dir / path.name
            target.write_bytes(path.read_bytes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Checkpoint path, relative to cwd or absolute.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Artifact directory.")
    parser.add_argument("--public-out-dir", default=str(DEFAULT_PUBLIC_OUT_DIR), help="Optional mirror artifact directory.")
    parser.add_argument("--prereg-path", default=None, help="Existing prereg JSON to verify/use.")
    parser.add_argument("--prereg-only", action="store_true", help="Write prereg JSON/SHA and exit before model load.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--support-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bp-steps", type=int, default=5)
    parser.add_argument("--null-permutations", type=int, default=256)
    parser.add_argument("--null-seed", type=int, default=17)
    parser.add_argument("--null-backend", choices=NULL_BACKENDS, default=DEFAULT_NULL_BACKEND)
    parser.add_argument(
        "--null-aggregation-device",
        choices=("cuda", "cpu"),
        default=None,
        help="Torch device for cpu_sampler_gpu_aggregation_replay aggregation; default prefers cuda when available.",
    )
    parser.add_argument(
        "--null-parity-profile-only",
        action="store_true",
        help="Run the preregistered CPU-null vs candidate parity/profile subset and exit.",
    )
    parser.add_argument("--null-profile-warmups", type=int, default=DEFAULT_NULL_PROFILE_WARMUPS)
    parser.add_argument("--null-profile-repeats", type=int, default=DEFAULT_NULL_PROFILE_REPEATS)
    parser.add_argument("--null-speedup-floor", type=float, default=DEFAULT_NULL_SPEEDUP_FLOOR)
    parser.add_argument("--null-speed-cpu-repeats", type=int, default=DEFAULT_NULL_SPEED_CPU_REPEATS)
    parser.add_argument("--null-speed-candidate-repeats", type=int, default=DEFAULT_NULL_SPEED_CANDIDATE_REPEATS)
    parser.add_argument("--null-speed-max-invocations-per-variant", type=int, default=DEFAULT_NULL_SPEED_MAX_INVOCATIONS_PER_VARIANT)
    parser.add_argument("--null-parity-abs-tol", type=float, default=NULL_DISTRIBUTIONAL_ABS_TOL)
    parser.add_argument("--null-parity-max-invocations-per-variant", type=int, default=16)
    parser.add_argument("--gpu-native-stage2-max-seconds", type=float, default=GPU_NATIVE_STAGE2_MAX_SECONDS)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; default uses all 121 rows.")
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--grad-reconstruction-atol", type=float, default=2e-3)
    args = parser.parse_args(raw_argv)
    args.null_backend_explicit = any(
        arg == "--null-backend" or arg.startswith("--null-backend=")
        for arg in raw_argv
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "credit_bridge_result.json"
    try:
        terminal, result, prereg_path = run_diagnostic(args)
        if terminal == "prereg_only":
            print(f"[credit-bridge] prereg_only complete path={prereg_path}", flush=True)
            if args.public_out_dir:
                mirror_artifacts(out_dir, Path(args.public_out_dir))
            return 0
        result_sha = write_json_with_sha(result_path, result)
        if args.public_out_dir:
            mirror_artifacts(out_dir, Path(args.public_out_dir))
        print(f"[credit-bridge] terminal={terminal}", flush=True)
        print(f"[credit-bridge] result_path={result_path}", flush=True)
        print(f"[credit-bridge] result_sha256={result_sha}", flush=True)
        ok_terminals = {
            "pow2_magnitude_sufficient",
            "fp16_groupwise_credit_sufficient_proxy",
            "tested_lowbit_magnitude_insufficient",
            "gpu_null_parity_exact_default_enabled",
            "gpu_null_parity_exact_speedup_insufficient_cpu_default_retained",
            "gpu_null_parity_exact_sampler_bound_deferred",
            "gpu_native_null_parity_default_enabled",
            "gpu_native_null_parity_explicit_validated_default_deferred",
            "gpu_native_null_parity_speedup_insufficient_cpu_default_retained",
        }
        semantic_fail_terminals = {"gpu_null_parity_fail", "gpu_native_null_parity_fail"}
        if terminal in ok_terminals:
            return 0
        if terminal in semantic_fail_terminals:
            return 1
        return 2
    except IntegrityFailure as exc:
        payload = {"task_id": TASK_ID, "terminal": "integrity_failure", "error": str(exc)}
        write_json_with_sha(result_path, payload)
        print(f"[credit-bridge] integrity_failure: {exc}", file=sys.stderr, flush=True)
        return 3
    except DiagnosticInvalid as exc:
        payload = {"task_id": TASK_ID, "terminal": "diagnostic_reference_invalid", "error": str(exc)}
        write_json_with_sha(result_path, payload)
        print(f"[credit-bridge] diagnostic_reference_invalid: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
