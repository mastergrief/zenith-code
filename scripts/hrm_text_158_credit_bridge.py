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


TASK_ID = "1780231236796-49ed823a"
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
NULL_BACKENDS = (NULL_BACKEND_CPU_LOCKED, NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY)
DEFAULT_NULL_BACKEND = NULL_BACKEND_CPU_LOCKED
DEFAULT_NULL_SPEEDUP_FLOOR = 1.25
DEFAULT_NULL_PROFILE_WARMUPS = 1
DEFAULT_NULL_PROFILE_REPEATS = 5
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
    timing_totals = {"cpu_sampler": 0.0, "aggregation": 0.0, "reported_total": 0.0}
    wall_start = time.perf_counter()
    for item in items:
        for null_kind, buckets, seed in _null_item_runs(item, null_seed=null_seed):
            out = simulate_permutation_null(
                buckets,
                permutations=permutations,
                seed=seed,
                backend=backend,
                aggregation_device=aggregation_device,
                profile=True,
            )
            key = f"{item['variant']}::{item['level']}::{item['label']}::{null_kind}"
            outputs[key] = out
            timing = out.get("timing_seconds") or {}
            timing_totals["cpu_sampler"] += float(timing.get("cpu_sampler", 0.0))
            timing_totals["aggregation"] += float(timing.get("aggregation", 0.0))
            timing_totals["reported_total"] += float(timing.get("total", 0.0))
    timing_totals["wall_total"] = time.perf_counter() - wall_start
    return outputs, timing_totals


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _compare_null_outputs(cpu_outputs: dict[str, dict[str, Any]], candidate_outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if set(cpu_outputs) != set(candidate_outputs):
        missing_cpu = sorted(set(candidate_outputs) - set(cpu_outputs))
        missing_candidate = sorted(set(cpu_outputs) - set(candidate_outputs))
        return [{"reason": "key_mismatch", "missing_cpu": missing_cpu, "missing_candidate": missing_candidate}]
    for key in sorted(cpu_outputs):
        cpu = cpu_outputs[key]
        candidate = candidate_outputs[key]
        deltas = {field: abs(float(cpu[field]) - float(candidate[field])) for field in ("mean", "p95", "p99")}
        if any(delta != 0.0 for delta in deltas.values()):
            failures.append({"key": key, "cpu": {k: cpu[k] for k in ("mean", "p95", "p99")}, "candidate": {k: candidate[k] for k in ("mean", "p95", "p99")}, "deltas": deltas})
    return failures


def run_null_parity_profile(
    variant_count_sets: dict[str, dict[str, Any]],
    *,
    args: argparse.Namespace,
    bars: Bars,
) -> tuple[str, dict[str, Any]]:
    items = collect_null_profile_items(
        variant_count_sets,
        max_invocations_per_variant=args.null_parity_max_invocations_per_variant,
    )
    for _ in range(args.null_profile_warmups):
        _run_null_backend_items(
            items,
            backend=NULL_BACKEND_CPU_LOCKED,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )
        _run_null_backend_items(
            items,
            backend=NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )

    cpu_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    for _ in range(args.null_profile_repeats):
        cpu_outputs, cpu_timing = _run_null_backend_items(
            items,
            backend=NULL_BACKEND_CPU_LOCKED,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )
        candidate_outputs, candidate_timing = _run_null_backend_items(
            items,
            backend=NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            permutations=args.null_permutations,
            null_seed=args.null_seed,
            aggregation_device=args.null_aggregation_device,
        )
        cpu_runs.append({"outputs": cpu_outputs, "timing": cpu_timing})
        candidate_runs.append({"outputs": candidate_outputs, "timing": candidate_timing})

    parity_failures = _compare_null_outputs(cpu_runs[0]["outputs"], candidate_runs[0]["outputs"]) if cpu_runs else []
    cpu_totals = [run["timing"]["wall_total"] for run in cpu_runs]
    candidate_totals = [run["timing"]["wall_total"] for run in candidate_runs]
    cpu_median = _median(cpu_totals)
    candidate_median = _median(candidate_totals)
    speedup = (cpu_median / candidate_median) if candidate_median > 0 else 0.0
    candidate_sampler_median = _median([run["timing"]["cpu_sampler"] for run in candidate_runs])
    candidate_sampler_fraction = candidate_sampler_median / candidate_median if candidate_median > 0 else 0.0
    q0_denoms = [
        item["counts"].q_stats.get("0", {}).get("denom", 0)
        for item in items
        if item["level"] == "invocation"
    ]
    if parity_failures:
        terminal = "gpu_null_parity_fail"
    elif speedup >= args.null_speedup_floor and DEFAULT_NULL_BACKEND == NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY:
        terminal = "gpu_null_parity_exact_default_enabled"
    elif candidate_sampler_fraction >= NULL_SAMPLER_BOUND_FRACTION:
        terminal = "gpu_null_parity_exact_sampler_bound_deferred"
    else:
        terminal = "gpu_null_parity_exact_speedup_insufficient_cpu_default_retained"

    profile = {
        "terminal": terminal,
        "backend_names": {
            "oracle": NULL_BACKEND_CPU_LOCKED,
            "candidate": NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            "default": DEFAULT_NULL_BACKEND,
        },
        "speed_gate": {
            "required_speedup_floor": args.null_speedup_floor,
            "cpu_locked_median_wall_seconds": cpu_median,
            "candidate_median_wall_seconds": candidate_median,
            "candidate_speedup": speedup,
            "candidate_cpu_sampler_fraction": candidate_sampler_fraction,
            "sampler_bound_fraction_threshold": NULL_SAMPLER_BOUND_FRACTION,
            "passes_speed_gate": speedup >= args.null_speedup_floor,
        },
        "benchmark_protocol": {
            "warmups": args.null_profile_warmups,
            "repeats": args.null_profile_repeats,
            "permutations": args.null_permutations,
            "aggregation_device": _torch_device_name(args.null_aggregation_device),
        },
        "subset": {
            "item_count": len(items),
            "max_invocations_per_variant": args.null_parity_max_invocations_per_variant,
            "q0_invocation_denom_min": min(q0_denoms) if q0_denoms else None,
            "q0_invocation_denom_max": max(q0_denoms) if q0_denoms else None,
            "levels": sorted({item["level"] for item in items}),
        },
        "parity": {
            "exact": not parity_failures,
            "failure_count": len(parity_failures),
            "failures": parity_failures[:20],
            "compared_fields": ["mean", "p95", "p99"],
        },
        "timings": {
            "cpu_locked_wall_seconds": cpu_totals,
            "candidate_wall_seconds": candidate_totals,
            "candidate_cpu_sampler_seconds": [run["timing"]["cpu_sampler"] for run in candidate_runs],
            "candidate_aggregation_seconds": [run["timing"]["aggregation"] for run in candidate_runs],
        },
        "default_cli_route_proof": {
            "required_before_default_flip": True,
            "default_backend": DEFAULT_NULL_BACKEND,
            "candidate_backend": NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            "not_applicable_reason": (
                None
                if DEFAULT_NULL_BACKEND == NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY
                else "cpu_locked remains default in this prereg/run"
            ),
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
        "source_parent_commit": "df7123b517f60f8d6c53d6360b31a8f37b880bd3",
        "dispatch_msg": "1780231252415-76819bf7",
        "plan_msg": "1780231521606-4d2ee333",
        "fold_msgs": [
            "1780231376585-ea0cf8ec",
            "1780231412346-5465c051",
            "1780231574654-b062b304",
            "1780231604003-fb0b20ba",
            "1780231616659-2912699c",
            "1780231671991-39ea93c5",
            "1780231687391-80e43239",
        ],
        "runner_watcher_provenance_msg": "1780226246648-6ee162bb",
        "implement_gate_msg": "1780231644991-683e9530",
        "implement_gate_confirmation_msg": "1780231671991-39ea93c5",
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
            "candidate": NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            "candidate_explicit_name": (
                "CPU/numpy hypergeometric sampler plus Torch aggregation replay; "
                "the sampler has NOT moved to GPU in this slice"
            ),
            "aggregation_device": _torch_device_name(args.null_aggregation_device),
            "profile_only": args.null_parity_profile_only,
            "speedup_floor": args.null_speedup_floor,
            "profile_warmups": args.null_profile_warmups,
            "profile_repeats": args.null_profile_repeats,
            "sampler_bound_fraction_threshold": NULL_SAMPLER_BOUND_FRACTION,
            "parity_max_invocations_per_variant": args.null_parity_max_invocations_per_variant,
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
            "Fold A: no from-scratch GPU hypergeometric/gpu_sampler in this slice; sampling-bound profile stops and defers sampler dispatch",
            "Fold B: bucket construction + aggregation parity must be bit-identical; CPU/GPU replay consume the exact same draw tensor/fixture",
            "Fold C: default flip requires bit-identical parity plus >=1.25x median total-null speedup over cpu_locked, with 1 warmup + 5 measured repeats",
            "Fold D: locked null parity terminal set has explicit default-enabled, speedup-insufficient, sampler-bound-deferred, fail, invalid, integrity outcomes",
            "Fold E: CPU/numpy sampler + GPU aggregation path must be named cpu_sampler_gpu_aggregation_replay in prereg/receipts",
            "Fold F: cpu_locked remains oracle/fallback; two-file scope; checkpoint sha before==after; no train/optimizer/param/.pt; prereg before screen",
            "Fold G: before any default flip, prove the default CLI path itself routes through cpu_sampler_gpu_aggregation_replay and matches explicit --null-backend cpu_locked on the prereg subset",
        ],
        "credit_terminal_labels": list(CREDIT_TERMINAL_LABELS),
        "terminal_labels": list(NULL_PARITY_TERMINAL_LABELS),
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
        terminal, null_profile = run_null_parity_profile(variant_count_sets, args=args, bars=bars)
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
    parser.add_argument("--null-parity-max-invocations-per-variant", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; default uses all 121 rows.")
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--grad-reconstruction-atol", type=float, default=2e-3)
    return parser.parse_args(argv)


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
        }
        semantic_fail_terminals = {"gpu_null_parity_fail"}
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
