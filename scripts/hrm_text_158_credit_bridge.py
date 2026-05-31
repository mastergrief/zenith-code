#!/usr/bin/env python3
"""Read-only HRM-Text-1.58 strict-integer credit bridge diagnostic.

This diagnostic compares a magnitude-free integer credit signal against the
magnitude-aware STE master-weight gradient, after both are projected onto the
same admissible one-step ternary moves. It is intentionally read-only: no
optimizer, no checkpoint save, and checkpoint SHA-256 must match before/after.
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


TASK_ID = "1780222249885-f8e55be0"
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
    credit: Tensor | None = None
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
        credit: Tensor,
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
        credit_cpu = credit.detach().to(torch.int32).cpu()
        weighted_cpu = weighted_grad.detach().to(torch.float32).cpu()
        active_inputs_cpu = active_inputs.detach().to(torch.bool).cpu()
        active_outputs_cpu = active_outputs.detach().to(torch.bool).cpu()

        if self.credit is None:
            self.credit = torch.zeros_like(credit_cpu)
            self.weighted_grad = torch.zeros_like(weighted_cpu)
            self.active_inputs = torch.zeros_like(active_inputs_cpu)
            self.active_outputs = torch.zeros_like(active_outputs_cpu)
        assert self.credit is not None
        assert self.weighted_grad is not None
        assert self.active_inputs is not None
        assert self.active_outputs is not None

        self.credit += credit_cpu
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
            credit_f = -torch.einsum("bso,bsi->oi", grad_sign, input_sign)
            rounded = credit_f.round()
            max_round_err = float((credit_f - rounded).abs().max().item()) if credit_f.numel() else 0.0
            if max_round_err > 1e-3:
                raise DiagnosticInvalid(
                    f"{target.name} rec_idx={rec_idx} {group.group}: integer credit lost exactness "
                    f"(max round err {max_round_err:.6g})"
                )
            weighted_grad = torch.einsum("bso,bsi->oi", grad_chunk, inp_f)
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
                credit=rounded.to(torch.int32),
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


def simulate_permutation_null(
    buckets: list[BucketCounts],
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if not buckets:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
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
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95, method="higher")),
        "p99": float(np.quantile(arr, 0.99, method="higher")),
    }


def deterministic_seed(base_seed: int, label: str, offset: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{offset}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def build_counts_for_invocation(
    key: InvocationKey,
    agg: InvocationAggregate,
    q_levels: Tensor,
) -> tuple[CountAccumulator, dict[str, Any]]:
    if agg.credit is None or agg.weighted_grad is None or agg.active_inputs is None or agg.active_outputs is None:
        raise DiagnosticInvalid(f"{key.label}: no backward aggregate captured")
    q = q_levels[agg.group_start:agg.group_end].cpu()
    credit = agg.credit.cpu()
    weighted = agg.weighted_grad.cpu()
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
    integer_dir = credit.sign()
    raw_denom = weighted_dir != 0
    counts.raw_dir_denom = int(raw_denom.sum().item())
    counts.raw_dir_disagree = int((raw_denom & (integer_dir != 0) & (weighted_dir != integer_dir)).sum().item())
    counts.raw_dir_integer_zero = int((raw_denom & (integer_dir == 0)).sum().item())

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
        "label": key.label,
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
) -> dict[str, Any]:
    agreement = counts.agree / counts.denom if counts.denom else 0.0
    global_null = simulate_permutation_null(
        counts.buckets_global,
        permutations=null_permutations,
        seed=deterministic_seed(null_seed, counts.label, 1),
    )
    rowq_null = simulate_permutation_null(
        counts.buckets_rowq,
        permutations=null_permutations,
        seed=deterministic_seed(null_seed, counts.label, 2),
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
    return "strict_integer_bridge_plausible", reasons


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
        "dispatch_msg": "1780222274361-cb110bbd",
        "plan_msg": "1780222832795-c45aaeaf",
        "bars_update_msg": "1780223015674-af9b7aa5",
        "implement_gate_msg": "1780223042896-c36934e2",
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
        "bars": bars.__dict__,
        "locked_tightenings": [
            "recurrence-aware 160 gradient-enabled invocation strata at bp_steps=5; schedule-excluded no-grad L invocations listed separately",
            "credit denominator uses all local positions with nonzero BitLinear output grad, with prefix/response breakdown",
            "sign agreement stratified by q=-1/q=0/q=+1; q=0 projected denominator gates plausible",
            "row/output-channel-preserving q-bucket null added; score against stronger p99 null",
            "absolute floors raised: global>=0.65, family>=0.60, stratum/invocation>=0.55",
            "FP reference is magnitude-aware STE master-weight gradient sign; integer credit is sum-of-signs, with cancellation diagnostics",
            "assert BitLinear cached/native flags false/None; no freeze_for_inference or enable_native_train",
        ],
        "terminal_labels": [
            "strict_integer_bridge_plausible",
            "credit_direction_mismatch",
            "route_death_persists_at_scale",
            "mixed_requires_scale_hedge",
            "diagnostic_reference_invalid",
            "integrity_failure",
        ],
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
    invocation_counts: dict[str, CountAccumulator] = {}
    invocation_details: dict[str, dict[str, Any]] = {}
    family_counts: dict[str, CountAccumulator] = {}
    aggregate64_counts: dict[str, CountAccumulator] = {}
    global_counts = CountAccumulator(label="global")
    for key, agg in sorted(tracker.aggregates.items(), key=lambda kv: kv[0].label):
        q = q_by_module[agg.module_name]
        counts, detail = build_counts_for_invocation(key, agg, q)
        invocation_counts[key.label] = counts
        invocation_details[key.label] = detail
        family_counts.setdefault(key.family_label, CountAccumulator(label=key.family_label)).merge(counts)
        aggregate64_counts.setdefault(key.aggregate64_label, CountAccumulator(label=key.aggregate64_label)).merge(counts)
        global_counts.merge(counts)

    expected_invocation_count = 160
    if len(invocation_counts) != expected_invocation_count:
        raise DiagnosticInvalid(
            f"gradient-enabled invocation strata {len(invocation_counts)} != {expected_invocation_count}"
        )

    invocation_summaries = {
        label: summarize_counts(
            counts,
            bars=bars,
            level="invocation",
            null_permutations=args.null_permutations,
            null_seed=args.null_seed,
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
        )
        for label, counts in sorted(aggregate64_counts.items())
    }
    global_summary = summarize_counts(
        global_counts,
        bars=bars,
        level="global",
        null_permutations=args.null_permutations,
        null_seed=args.null_seed,
    )
    terminal, reasons = terminal_from_summaries(
        global_summary=global_summary,
        family_summaries=family_summaries,
        invocation_summaries=invocation_summaries,
        bars=bars,
    )

    ckpt_sha_after = sha256_file(ckpt_path)
    if ckpt_sha_after != ckpt_sha_before:
        raise IntegrityFailure(
            f"checkpoint hash changed before={ckpt_sha_before} after={ckpt_sha_after}"
        )

    excluded = [
        {"label": item.key.label, "module_name": item.module_name, "reason": item.reason}
        for item in tracker.schedule_excluded.values()
    ]
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
        "global": global_summary,
        "families": family_summaries,
        "aggregate64": aggregate64_summaries,
        "invocations": invocation_summaries,
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
        return 0 if terminal in {"strict_integer_bridge_plausible", "mixed_requires_scale_hedge", "credit_direction_mismatch", "route_death_persists_at_scale"} else 2
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
