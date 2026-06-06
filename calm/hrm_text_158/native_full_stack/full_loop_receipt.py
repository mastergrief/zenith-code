"""Default-off native full-loop reference-stitch engineering receipt.

This bridge stitches the Phase-1 qscale, vote/update, selection, global-cap,
q/acc apply, and q-pack accounting seams in one tiny deterministic loop. It is
an engineering receipt only: no live trainer entrypoint, no optimizer, no
checkpoint, no creditdir mutation, no acquisition/retention claim, and no
native/custom-kernel speed claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapOrderingMode,
    GlobalRateCapResult,
    GlobalRateCapRow,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
    GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
    RUN_GPU_GLOBAL_RATE_CAP_ENV,
    apply_global_rate_cap_torch_cuda_reference_under_margin,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
    PersistentStateBudgetReport,
    measure_persistent_state_budget,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import (
    QScaleWeightState,
    qscale_linear_reference,
)
from calm.hrm_text_158.native_full_stack.selection_topk import (
    select_pre_veto_candidates_from_plan,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    RUN_GPU_Q_ACC_APPLY_ENV,
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
    VoteUpdateState,
    q_acc_apply_mutation_torch_cuda_reference_under_cap_rows,
    plan_integer_vote_update_reference,
)


RUN_GPU_FULL_LOOP_RECEIPT_ENV = "HRM_TEXT_158_RUN_GPU_FULL_LOOP_RECEIPT"
FULL_LOOP_RECEIPT_ARTIFACT_ENV = "HRM_TEXT_158_FULL_LOOP_RECEIPT_ARTIFACT"
DEFAULT_FULL_LOOP_RECEIPT_ARTIFACT_PATH = (
    "artifacts/hrm_text_158_native_full_loop_receipts/"
    "native_full_loop_engineering_receipt_gpu.json"
)
FULL_LOOP_RECEIPT_SCHEMA_VERSION = "hrm_text_158_native_full_loop_receipt/v0.reference_stitch"
NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL = (
    "native_full_loop_engineering_receipt_reference_stitch_only"
)
NATIVE_FULL_LOOP_REFERENCE_STITCH_SCOPE = "native_full_loop_reference_stitch_default_off"
TINY_TWO_PROJECTION_FIXTURE_NAME = "tiny_two_projection_vote_cap_fixture"
TINY_LOOP_STEP_COUNT = 2
TINY_LOOP_GLOBAL_CAP = 2
QSCALE_REFERENCE_MATERIALIZATION_CAVEAT = (
    "qscale_linear_reference materializes q.float() * scale before F.linear; "
    "wall-clock is reference-harness timing, not a native/custom-kernel speed claim."
)
GLOBAL_CAP_CPU_GLUE_CAVEAT = (
    "global cap remains apply_global_rate_cap_reference CPU/control-flow glue "
    f"when {RUN_GPU_GLOBAL_RATE_CAP_ENV} is unset; explicit per-step backend "
    "fields identify opt-in GPU-cap receipt rows."
)
CAP_ACCEPTED_ROWS_PROVENANCE = "apply_global_rate_cap_reference.accepted_rows"
GPU_CAP_ACCEPTED_ROWS_PROVENANCE = "device_selection_result.rows_by_state"
GLOBAL_CAP_CPU_GLUE_SCOPE = "apply_global_rate_cap_reference_cpu_glue"
GLOBAL_CAP_BACKEND_CAVEAT = (
    "global cap backend is explicit per step: legacy CPU/reference glue when "
    f"{RUN_GPU_GLOBAL_RATE_CAP_ENV} is unset; MARGIN-only torch-CUDA reference "
    "selection/apply-chain when set; CPU oracle parity is retained in both modes."
)
ALLOCATOR_DELTA_TELEMETRY_CAVEAT = (
    "no_leak_alloc_delta_bytes is allocator/headroom telemetry, not a zero-leak proof"
)
NEXT_PHYSICAL_SUB2_FORK = (
    "accumulator/vote-state compression toward ternary-hybrid/event-coded/sparse "
    "representation"
)

# Prior large-fixture context only. The tiny receipt computes its own ledger from
# measure_persistent_state_budget and must not use this as the tiny pass metric.
PRIOR_LARGE_FIXTURE_REFERENCE = {
    "scope": "prior_large_fixture_reference_only_not_tiny_loop_expected_value",
    "packed_inclusive_physical_bits_per_weight": 18.017578,
    "required_acc_bits_per_weight_for_sub2_physical_q_with_scale_and_metadata": -0.017578,
    "target_achieved": False,
    "receipt_statement": PHYSICAL_SUB2_NOT_ACHIEVED_STATEMENT,
}

_TENSOR_ORDER = ("proj_in", "proj_out")


@dataclass(frozen=True)
class TinyTwoProjectionVoteCapFixture:
    """Static tiny fixture used by CPU/default-off tests and the GPU receipt."""

    name: str
    qscale_states: dict[str, QScaleWeightState]
    accumulators: dict[str, torch.Tensor]
    input_tensor: torch.Tensor
    cap: int
    step_count: int

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(state.q_levels.shape) for key, state in self.qscale_states.items()}

    @property
    def eligible_weight_count(self) -> int:
        return int(sum(state.q_levels.numel() for state in self.qscale_states.values()))


@dataclass(frozen=True)
class NativeFullLoopStepReceipt:
    step: int
    label: str
    scope: str
    qscale_backend: str
    qscale_materialization_caveat: str
    step_duration_seconds: float
    state_input_hashes: dict[str, str]
    state_output_hashes: dict[str, str]
    qscale_output_hashes: dict[str, str]
    qscale_outputs_finite: bool
    selection_backend_by_state: dict[str, str]
    pre_cap_demand_count: int
    global_rate_cap_cap: int
    accepted_count: int
    deferred_count: int
    accepted_count_by_state: dict[str, int]
    deferred_count_by_state: dict[str, int]
    accepted_state_keys: tuple[str, ...]
    cap_single_tensor_winner_state_key: str | None
    q_changed_count: int
    q_changed_count_by_state: dict[str, int]
    global_rate_cap_saturated: bool
    cap_provenance_source: str
    global_cap_backend: str
    global_cap_scope: str
    global_cap_gpu_enabled: bool
    global_cap_cpu_oracle_retained: bool
    global_cap_counts_match_cpu_oracle: bool
    global_cap_backlog_matches_cpu_oracle: bool
    global_cap_backlog_keys_match_cpu_oracle: bool
    global_cap_deferred_backlog_size: int
    global_cap_deferred_backlog_max_age_steps: int
    global_cap_deferred_backlog_max_defer_count: int
    local_plan_rows_used_as_cap_acceptance: bool
    parity_cuda_matches_cpu_global_cap_oracle: bool
    cpu_oracle_mutate_outputs: bool
    step_consumes_state_mutated_by_prior_step: bool
    budget: dict[str, int | float | bool | str]


@dataclass(frozen=True)
class NativeFullLoopEngineeringReceipt:
    schema_version: str
    label: str
    scope: str
    fixture_name: str
    step_count: int
    tensor_shapes: dict[str, tuple[int, ...]]
    eligible_weight_count: int
    default_off_env: str
    q_acc_apply_env: str
    global_cap_gpu_env: str
    qscale_materialization_caveat: str
    global_cap_cpu_glue_caveat: str
    global_cap_backend_caveat: str
    acquisition_claim: bool
    retention_claim: bool
    native_custom_kernel_speed_claim: bool
    prior_large_fixture_reference: dict[str, int | float | bool | str]
    next_physical_sub2_science_fork: str
    step_receipts: tuple[NativeFullLoopStepReceipt, ...]
    terminal_budget: dict[str, int | float | bool | str]
    wall_clock_total_seconds: float
    wall_clock_per_step_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    free_memory_bytes: int
    total_memory_bytes: int
    allocated_before_bytes: int
    allocated_after_cleanup_bytes: int
    no_leak_alloc_delta_bytes: int
    allocator_delta_caveat: str
    artifact_path: str | None
    artifact_hygiene: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tiny_two_projection_vote_cap_fixture(
    *,
    device: torch.device | str = "cpu",
) -> TinyTwoProjectionVoteCapFixture:
    """Build the exact tiny two-state fixture named in the implementation gate."""

    device = torch.device(device)
    q_in = _ternary_q_with_zeroed_rows((8, 16), zero_flat_indices=(0, 1, 2, 3, 4, 5), device=device)
    q_out = _ternary_q_with_zeroed_rows((4, 8), zero_flat_indices=(0, 1, 2, 3), device=device)
    states = {
        "proj_in": QScaleWeightState(
            q_levels=q_in,
            scale=torch.tensor(0.125, dtype=torch.float32, device=device),
        ),
        "proj_out": QScaleWeightState(
            q_levels=q_out,
            scale=torch.tensor(0.25, dtype=torch.float32, device=device),
        ),
    }
    accumulators = {
        key: torch.zeros_like(state.q_levels, dtype=torch.int16, device=device)
        for key, state in states.items()
    }
    input_tensor = (
        torch.arange(32, dtype=torch.float32, device=device).view(2, 16) - 15.5
    ) / 17.0
    return TinyTwoProjectionVoteCapFixture(
        name=TINY_TWO_PROJECTION_FIXTURE_NAME,
        qscale_states=states,
        accumulators=accumulators,
        input_tensor=input_tensor.contiguous(),
        cap=TINY_LOOP_GLOBAL_CAP,
        step_count=TINY_LOOP_STEP_COUNT,
    )


def measure_tiny_two_projection_fixture_budget(
    *,
    device: torch.device | str = "cpu",
) -> PersistentStateBudgetReport:
    """Measure the tiny fixture ledger; this is not the prior large fixture."""

    fixture = tiny_two_projection_vote_cap_fixture(device=device)
    return measure_persistent_state_budget(
        list(fixture.qscale_states.values()),
        list(fixture.accumulators.values()),
    )


def tiny_full_loop_vote_update_spec() -> VoteUpdateSpec:
    """Return the fixed per-step vote/update spec used by the tiny receipt stitch."""

    return VoteUpdateSpec(
        threshold_abs=2,
        accumulator_clip_min=-32768,
        accumulator_clip_max=32767,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=4,
        fraction_per_tensor=1.0,
    )


def tiny_full_loop_votes_for_step(
    step: int,
    *,
    device: torch.device | str = "cpu",
    repeat_cycle: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the tiny receipt vote pattern, optionally cycling past step 2."""

    device = torch.device(device)
    normalized_step = _normalize_tiny_loop_pattern_step(step, repeat_cycle=repeat_cycle)
    return _votes_for_step(normalized_step, device=device)


def run_native_full_loop_engineering_receipt(
    *,
    device: torch.device | str = "cuda",
    artifact_path: str | Path | None = None,
) -> NativeFullLoopEngineeringReceipt:
    """Run the default-off CUDA reference-stitch loop and return compact proof data."""

    device = _require_gpu_receipt_env(device)
    if artifact_path is None and os.environ.get(FULL_LOOP_RECEIPT_ARTIFACT_ENV):
        artifact_path = os.environ[FULL_LOOP_RECEIPT_ARTIFACT_ENV]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    step_receipts, terminal_budget, tensor_shapes, eligible_weight_count = _execute_tiny_loop(device)
    torch.cuda.synchronize(device)
    wall_clock_total = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_memory, total_memory = torch.cuda.mem_get_info(device)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    allocated_after_cleanup = int(torch.cuda.memory_allocated(device))

    artifact_path_str = str(artifact_path) if artifact_path is not None else None
    receipt = NativeFullLoopEngineeringReceipt(
        schema_version=FULL_LOOP_RECEIPT_SCHEMA_VERSION,
        label=NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL,
        scope=NATIVE_FULL_LOOP_REFERENCE_STITCH_SCOPE,
        fixture_name=TINY_TWO_PROJECTION_FIXTURE_NAME,
        step_count=TINY_LOOP_STEP_COUNT,
        tensor_shapes=tensor_shapes,
        eligible_weight_count=eligible_weight_count,
        default_off_env=RUN_GPU_FULL_LOOP_RECEIPT_ENV,
        q_acc_apply_env=RUN_GPU_Q_ACC_APPLY_ENV,
        global_cap_gpu_env=RUN_GPU_GLOBAL_RATE_CAP_ENV,
        qscale_materialization_caveat=QSCALE_REFERENCE_MATERIALIZATION_CAVEAT,
        global_cap_cpu_glue_caveat=GLOBAL_CAP_CPU_GLUE_CAVEAT,
        global_cap_backend_caveat=GLOBAL_CAP_BACKEND_CAVEAT,
        acquisition_claim=False,
        retention_claim=False,
        native_custom_kernel_speed_claim=False,
        prior_large_fixture_reference=PRIOR_LARGE_FIXTURE_REFERENCE,
        next_physical_sub2_science_fork=NEXT_PHYSICAL_SUB2_FORK,
        step_receipts=step_receipts,
        terminal_budget=terminal_budget,
        wall_clock_total_seconds=wall_clock_total,
        wall_clock_per_step_seconds=wall_clock_total / float(TINY_LOOP_STEP_COUNT),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        free_memory_bytes=int(free_memory),
        total_memory_bytes=int(total_memory),
        allocated_before_bytes=allocated_before,
        allocated_after_cleanup_bytes=allocated_after_cleanup,
        no_leak_alloc_delta_bytes=allocated_after_cleanup - allocated_before,
        allocator_delta_caveat=ALLOCATOR_DELTA_TELEMETRY_CAVEAT,
        artifact_path=artifact_path_str,
        artifact_hygiene=(
            "compact runtime proof only: shapes, hashes, counters, ledgers, timings; "
            "no raw tensor arrays or per-weight dumps; allocator delta is telemetry, "
            "not a zero-leak proof; do not stage in commit"
        ),
    )
    if artifact_path is not None:
        write_native_full_loop_receipt_artifact(receipt, artifact_path)
    return receipt


def write_native_full_loop_receipt_artifact(
    receipt: NativeFullLoopEngineeringReceipt,
    path: str | Path,
) -> Path:
    """Write the compact receipt JSON artifact; callers must keep it untracked."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _require_gpu_receipt_env(device: torch.device | str) -> torch.device:
    if os.environ.get(RUN_GPU_FULL_LOOP_RECEIPT_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_FULL_LOOP_RECEIPT_ENV}=1 is required and must only be set "
            "inside a granted gpu:0 resource lane"
        )
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        raise RuntimeError(
            f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required for the q/acc apply seam and "
            "must only be set inside the same granted gpu:0 resource lane"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native full-loop engineering receipt")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("native full-loop engineering receipt requires a CUDA device")
    return device


def _execute_tiny_loop(
    device: torch.device,
) -> tuple[tuple[NativeFullLoopStepReceipt, ...], dict[str, Any], dict[str, tuple[int, ...]], int]:
    fixture = tiny_two_projection_vote_cap_fixture(device=device)
    qscale_states = dict(fixture.qscale_states)
    accumulators = dict(fixture.accumulators)
    input_tensor = fixture.input_tensor
    spec = tiny_full_loop_vote_update_spec()
    tensor_offsets = _tensor_offsets(qscale_states)
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None
    previous_output_hashes: dict[str, str] | None = None
    step_receipts: list[NativeFullLoopStepReceipt] = []

    for step in range(1, fixture.step_count + 1):
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        state_input_hashes = _state_hashes(qscale_states, accumulators)

        with torch.no_grad():
            hidden = qscale_linear_reference(input_tensor, qscale_states["proj_in"])
            logits = qscale_linear_reference(hidden, qscale_states["proj_out"])
        qscale_output_hashes = {
            "proj_in_hidden": _tensor_sha256(hidden),
            "proj_out_logits": _tensor_sha256(logits),
        }
        qscale_outputs_finite = bool(torch.isfinite(hidden).all().item() and torch.isfinite(logits).all().item())

        plans = _plans_for_step(qscale_states, accumulators, spec, step=step)
        selection_backend_by_state = {
            key: select_pre_veto_candidates_from_plan(plan, threshold_abs=spec.threshold_abs).backend
            for key, plan in plans.items()
        }
        cap_inputs = [
            GlobalRateCapTensorInput(
                state_key=key,
                state=VoteUpdateState(
                    q_levels=qscale_states[key].q_levels,
                    accumulators=accumulators[key],
                ),
                plan=plans[key],
            )
            for key in _TENSOR_ORDER
        ]
        cpu_oracle = apply_global_rate_cap_reference(
            cap_inputs,
            GlobalRateCapSpec(
                cap=fixture.cap,
                step=step,
                ordering_mode=GlobalRateCapOrderingMode.MARGIN,
                mutate_outputs=True,
            ),
            deferred_backlog=deferred_backlog,
            tensor_offsets=tensor_offsets,
        )
        cap_step = _run_cap_path_for_step(
            qscale_states=qscale_states,
            accumulators=accumulators,
            cap_inputs=cap_inputs,
            cpu_oracle=cpu_oracle,
            tensor_offsets=tensor_offsets,
            incoming_deferred_backlog=deferred_backlog,
            cap=fixture.cap,
            step=step,
        )
        qscale_states = cap_step.qscale_states
        accumulators = cap_step.accumulators
        deferred_backlog = cap_step.deferred_backlog
        state_output_hashes = _state_hashes(qscale_states, accumulators)
        budget = measure_persistent_state_budget(
            list(qscale_states.values()),
            list(accumulators.values()),
        ).to_dict()
        torch.cuda.synchronize(device)

        nonzero_winners = [
            key for key, count in cap_step.accepted_count_by_state.items() if count > 0
        ]
        step_receipts.append(
            NativeFullLoopStepReceipt(
                step=step,
                label=NATIVE_FULL_LOOP_ENGINEERING_RECEIPT_LABEL,
                scope=NATIVE_FULL_LOOP_REFERENCE_STITCH_SCOPE,
                qscale_backend=device.type,
                qscale_materialization_caveat=QSCALE_REFERENCE_MATERIALIZATION_CAVEAT,
                step_duration_seconds=time.perf_counter() - step_started,
                state_input_hashes=state_input_hashes,
                state_output_hashes=state_output_hashes,
                qscale_output_hashes=qscale_output_hashes,
                qscale_outputs_finite=qscale_outputs_finite,
                selection_backend_by_state=selection_backend_by_state,
                pre_cap_demand_count=cap_step.pre_cap_demand_count,
                global_rate_cap_cap=cap_step.global_rate_cap_cap,
                accepted_count=cap_step.accepted_count,
                deferred_count=cap_step.deferred_count,
                accepted_count_by_state=cap_step.accepted_count_by_state,
                deferred_count_by_state=cap_step.deferred_count_by_state,
                accepted_state_keys=cap_step.accepted_state_keys,
                cap_single_tensor_winner_state_key=nonzero_winners[0] if len(nonzero_winners) == 1 else None,
                q_changed_count=int(sum(cap_step.q_changed_count_by_state.values())),
                q_changed_count_by_state=cap_step.q_changed_count_by_state,
                global_rate_cap_saturated=cap_step.global_rate_cap_saturated,
                cap_provenance_source=cap_step.cap_provenance_source,
                global_cap_backend=cap_step.global_cap_backend,
                global_cap_scope=cap_step.global_cap_scope,
                global_cap_gpu_enabled=cap_step.global_cap_gpu_enabled,
                global_cap_cpu_oracle_retained=True,
                global_cap_counts_match_cpu_oracle=cap_step.counts_match_cpu_oracle,
                global_cap_backlog_matches_cpu_oracle=cap_step.backlog_matches_cpu_oracle,
                global_cap_backlog_keys_match_cpu_oracle=cap_step.backlog_keys_match_cpu_oracle,
                global_cap_deferred_backlog_size=cap_step.deferred_backlog_summary[
                    "deferred_backlog_size"
                ],
                global_cap_deferred_backlog_max_age_steps=cap_step.deferred_backlog_summary[
                    "deferred_backlog_max_age_steps"
                ],
                global_cap_deferred_backlog_max_defer_count=cap_step.deferred_backlog_summary[
                    "deferred_backlog_max_defer_count"
                ],
                local_plan_rows_used_as_cap_acceptance=False,
                parity_cuda_matches_cpu_global_cap_oracle=cap_step.parity_ok,
                cpu_oracle_mutate_outputs=True,
                step_consumes_state_mutated_by_prior_step=(
                    previous_output_hashes is not None
                    and state_input_hashes == previous_output_hashes
                    and any(state_input_hashes[key] != step_receipts[-1].state_input_hashes[key] for key in _TENSOR_ORDER)
                ),
                budget=budget,
            )
        )
        previous_output_hashes = state_output_hashes

    terminal_budget = measure_persistent_state_budget(
        list(qscale_states.values()),
        list(accumulators.values()),
    ).to_dict()
    return (
        tuple(step_receipts),
        terminal_budget,
        fixture.tensor_shapes,
        fixture.eligible_weight_count,
    )


def _plans_for_step(
    qscale_states: dict[str, QScaleWeightState],
    accumulators: dict[str, torch.Tensor],
    spec: VoteUpdateSpec,
    *,
    step: int,
) -> dict[str, VoteUpdatePlan]:
    votes = _votes_for_step(step, device=qscale_states["proj_in"].q_levels.device)
    return {
        key: plan_integer_vote_update_reference(
            VoteUpdateState(
                q_levels=qscale_states[key].q_levels,
                accumulators=accumulators[key],
            ),
            VoteUpdateInputs(votes=votes[key]),
            spec,
        )
        for key in _TENSOR_ORDER
    }


def _apply_cap_rows_on_cuda(
    *,
    qscale_states: dict[str, QScaleWeightState],
    accumulators: dict[str, torch.Tensor],
    plans: dict[str, VoteUpdatePlan],
    cap_result: GlobalRateCapResult,
    cpu_oracle: GlobalRateCapResult,
    device: torch.device,
) -> tuple[dict[str, QScaleWeightState], dict[str, torch.Tensor], bool, dict[str, int]]:
    accepted_rows_by_state = _rows_by_state(cap_result.accepted_rows)
    oracle_by_state = {result.state_key: result for result in cpu_oracle.tensor_results}
    out_states: dict[str, QScaleWeightState] = {}
    out_accumulators: dict[str, torch.Tensor] = {}
    parity_ok = True
    q_changed_by_state: dict[str, int] = {}

    for key in _TENSOR_ORDER:
        plan = plans[key]
        accepted, directions, thresholds = _accepted_row_tensors(
            plan,
            accepted_rows_by_state.get(key, ()),
            device=device,
        )
        result = q_acc_apply_mutation_torch_cuda_reference_under_cap_rows(
            q_levels=qscale_states[key].q_levels,
            new_accumulators=plan.new_acc_i32,
            accepted_indices=accepted,
            accepted_directions=directions,
            accepted_thresholds=thresholds,
            replay_veto_indices=plan.replay_ce_veto_indices,
            replay_veto_directions=plan.replay_veto_directions,
            replay_veto_thresholds=plan.replay_veto_thresholds,
            mutate_outputs=True,
            original_accumulators=accumulators[key],
        )
        oracle = oracle_by_state[key]
        q_match = torch.equal(result.q_levels.detach().cpu(), oracle.q_levels.detach().cpu())
        acc_match = torch.equal(result.accumulators.detach().cpu(), oracle.accumulators.detach().cpu())
        parity_ok = parity_ok and q_match and acc_match
        out_states[key] = QScaleWeightState(
            q_levels=result.q_levels,
            scale=qscale_states[key].scale,
            format=qscale_states[key].format,
        )
        out_accumulators[key] = result.accumulators
        q_changed_by_state[key] = int(result.stats["q_changed_count"])

    return out_states, out_accumulators, parity_ok, q_changed_by_state


@dataclass(frozen=True)
class _FullLoopCapStepResult:
    qscale_states: dict[str, QScaleWeightState]
    accumulators: dict[str, torch.Tensor]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]
    pre_cap_demand_count: int
    global_rate_cap_cap: int
    accepted_count: int
    deferred_count: int
    accepted_count_by_state: dict[str, int]
    deferred_count_by_state: dict[str, int]
    accepted_state_keys: tuple[str, ...]
    q_changed_count_by_state: dict[str, int]
    global_rate_cap_saturated: bool
    cap_provenance_source: str
    global_cap_backend: str
    global_cap_scope: str
    global_cap_gpu_enabled: bool
    parity_ok: bool
    counts_match_cpu_oracle: bool
    backlog_matches_cpu_oracle: bool
    backlog_keys_match_cpu_oracle: bool
    deferred_backlog_summary: dict[str, int]


def _run_cap_path_for_step(
    *,
    qscale_states: dict[str, QScaleWeightState],
    accumulators: dict[str, torch.Tensor],
    cap_inputs: list[GlobalRateCapTensorInput],
    cpu_oracle: GlobalRateCapResult,
    tensor_offsets: dict[str, int],
    incoming_deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
    cap: int,
    step: int,
) -> _FullLoopCapStepResult:
    spec_frozen = GlobalRateCapSpec(
        cap=cap,
        step=step,
        ordering_mode=GlobalRateCapOrderingMode.MARGIN,
        mutate_outputs=False,
    )
    spec_mutating = GlobalRateCapSpec(
        cap=cap,
        step=step,
        ordering_mode=GlobalRateCapOrderingMode.MARGIN,
        mutate_outputs=True,
    )
    cpu_selection = apply_global_rate_cap_reference(
        cap_inputs,
        spec_frozen,
        deferred_backlog=incoming_deferred_backlog,
        tensor_offsets=tensor_offsets,
    )
    if os.environ.get(RUN_GPU_GLOBAL_RATE_CAP_ENV) == "1":
        return _run_gpu_cap_path_for_step(
            qscale_states=qscale_states,
            cap_inputs=cap_inputs,
            cpu_selection=cpu_selection,
            cpu_oracle=cpu_oracle,
            spec=spec_mutating,
            tensor_offsets=tensor_offsets,
            incoming_deferred_backlog=incoming_deferred_backlog,
        )
    out_states, out_accumulators, parity_ok, q_changed_by_state = _apply_cap_rows_on_cuda(
        qscale_states=qscale_states,
        accumulators=accumulators,
        plans={item.state_key: item.plan for item in cap_inputs},
        cap_result=cpu_selection,
        cpu_oracle=cpu_oracle,
        device=next(iter(qscale_states.values())).q_levels.device,
    )
    return _cap_step_result_from_cpu_selection(
        qscale_states=out_states,
        accumulators=out_accumulators,
        active_backlog=cpu_selection.deferred_backlog,
        cpu_selection=cpu_selection,
        cpu_oracle=cpu_oracle,
        q_changed_count_by_state=q_changed_by_state,
        parity_ok=parity_ok,
        step=step,
    )


def _run_gpu_cap_path_for_step(
    *,
    qscale_states: dict[str, QScaleWeightState],
    cap_inputs: list[GlobalRateCapTensorInput],
    cpu_selection: GlobalRateCapResult,
    cpu_oracle: GlobalRateCapResult,
    spec: GlobalRateCapSpec,
    tensor_offsets: dict[str, int],
    incoming_deferred_backlog: dict[str, dict[int, dict[str, int]]] | None,
) -> _FullLoopCapStepResult:
    gpu_cap = apply_global_rate_cap_torch_cuda_reference_under_margin(
        cap_inputs,
        spec,
        tensor_offsets=tensor_offsets,
        deferred_backlog=incoming_deferred_backlog,
    )
    out_states, out_accumulators = _states_from_tensor_results(
        qscale_states,
        gpu_cap.tensor_results,
    )
    parity_ok = _tensor_results_match_oracle(gpu_cap.tensor_results, cpu_oracle)
    q_changed_by_state = {
        result.state_key: int(result.stats["q_changed_count"])
        for result in gpu_cap.tensor_results
    }
    accepted_by_state = {
        key: int(gpu_cap.selection.rows_by_state[key].accepted_indices.numel())
        for key in _TENSOR_ORDER
    }
    deferred_by_state = {
        key: int(gpu_cap.selection.rows_by_state[key].deferred_indices.numel())
        for key in _TENSOR_ORDER
    }
    active_backlog = gpu_cap.selection.deferred_backlog
    return _FullLoopCapStepResult(
        qscale_states=out_states,
        accumulators=out_accumulators,
        deferred_backlog=active_backlog,
        pre_cap_demand_count=int(gpu_cap.stats["global_pre_cap_would_apply_count"]),
        global_rate_cap_cap=int(gpu_cap.stats["global_rate_cap_cap"]),
        accepted_count=int(gpu_cap.stats["global_rate_cap_accepted_count"]),
        deferred_count=int(gpu_cap.stats["global_rate_cap_deferred_count"]),
        accepted_count_by_state=accepted_by_state,
        deferred_count_by_state=deferred_by_state,
        accepted_state_keys=tuple(row[0] for row in gpu_cap.selection.accepted_rows_as_tuples()),
        q_changed_count_by_state=q_changed_by_state,
        global_rate_cap_saturated=bool(gpu_cap.stats["global_rate_cap_saturated"]),
        cap_provenance_source=GPU_CAP_ACCEPTED_ROWS_PROVENANCE,
        global_cap_backend="cuda",
        global_cap_scope=GLOBAL_RATE_CAP_TORCH_CUDA_REFERENCE_SCOPE,
        global_cap_gpu_enabled=True,
        parity_ok=parity_ok,
        counts_match_cpu_oracle=_counts_match_cpu_oracle(
            accepted_by_state,
            deferred_by_state,
            cpu_oracle,
        )
        and _counts_match_cpu_oracle(
            accepted_by_state,
            deferred_by_state,
            cpu_selection,
        ),
        backlog_matches_cpu_oracle=_backlog_summary(active_backlog, step=spec.step)
        == _backlog_summary(cpu_oracle.deferred_backlog, step=spec.step),
        backlog_keys_match_cpu_oracle=_backlog_keys(active_backlog)
        == _backlog_keys(cpu_oracle.deferred_backlog),
        deferred_backlog_summary=_backlog_summary(active_backlog, step=spec.step),
    )


def _cap_step_result_from_cpu_selection(
    *,
    qscale_states: dict[str, QScaleWeightState],
    accumulators: dict[str, torch.Tensor],
    active_backlog: dict[str, dict[int, dict[str, int]]],
    cpu_selection: GlobalRateCapResult,
    cpu_oracle: GlobalRateCapResult,
    q_changed_count_by_state: dict[str, int],
    parity_ok: bool,
    step: int,
) -> _FullLoopCapStepResult:
    accepted_by_state = _row_count_by_state(cpu_selection.accepted_rows)
    deferred_by_state = _row_count_by_state(cpu_selection.deferred_rows)
    return _FullLoopCapStepResult(
        qscale_states=qscale_states,
        accumulators=accumulators,
        deferred_backlog=active_backlog,
        pre_cap_demand_count=int(cpu_selection.step_summary["global_pre_cap_would_apply_count"]),
        global_rate_cap_cap=int(cpu_selection.step_summary["global_rate_cap_cap"]),
        accepted_count=int(cpu_selection.step_summary["global_rate_cap_accepted_count"]),
        deferred_count=int(cpu_selection.step_summary["global_rate_cap_deferred_count"]),
        accepted_count_by_state=accepted_by_state,
        deferred_count_by_state=deferred_by_state,
        accepted_state_keys=tuple(row.state_key for row in cpu_selection.accepted_rows),
        q_changed_count_by_state=q_changed_count_by_state,
        global_rate_cap_saturated=bool(cpu_selection.step_summary["global_rate_cap_saturated"]),
        cap_provenance_source=CAP_ACCEPTED_ROWS_PROVENANCE,
        global_cap_backend="cpu_reference_glue",
        global_cap_scope=GLOBAL_CAP_CPU_GLUE_SCOPE,
        global_cap_gpu_enabled=False,
        parity_ok=parity_ok,
        counts_match_cpu_oracle=_counts_match_cpu_oracle(
            accepted_by_state,
            deferred_by_state,
            cpu_oracle,
        ),
        backlog_matches_cpu_oracle=_backlog_summary(active_backlog, step=step)
        == _backlog_summary(cpu_oracle.deferred_backlog, step=step),
        backlog_keys_match_cpu_oracle=_backlog_keys(active_backlog)
        == _backlog_keys(cpu_oracle.deferred_backlog),
        deferred_backlog_summary=_backlog_summary(
            active_backlog,
            step=step,
        ),
    )


def _states_from_tensor_results(
    qscale_states: dict[str, QScaleWeightState],
    tensor_results: list[Any],
) -> tuple[dict[str, QScaleWeightState], dict[str, torch.Tensor]]:
    out_states: dict[str, QScaleWeightState] = {}
    out_accumulators: dict[str, torch.Tensor] = {}
    result_by_key = {result.state_key: result for result in tensor_results}
    for key in _TENSOR_ORDER:
        result = result_by_key[key]
        out_states[key] = QScaleWeightState(
            q_levels=result.q_levels,
            scale=qscale_states[key].scale,
            format=qscale_states[key].format,
        )
        out_accumulators[key] = result.accumulators
    return out_states, out_accumulators


def _tensor_results_match_oracle(
    tensor_results: list[Any],
    cpu_oracle: GlobalRateCapResult,
) -> bool:
    oracle_by_state = {result.state_key: result for result in cpu_oracle.tensor_results}
    for result in tensor_results:
        oracle = oracle_by_state[result.state_key]
        if not torch.equal(result.q_levels.detach().cpu(), oracle.q_levels.detach().cpu()):
            return False
        if not torch.equal(result.accumulators.detach().cpu(), oracle.accumulators.detach().cpu()):
            return False
    return True


def _counts_match_cpu_oracle(
    accepted_by_state: dict[str, int],
    deferred_by_state: dict[str, int],
    cpu_oracle: GlobalRateCapResult,
) -> bool:
    return (
        accepted_by_state == _row_count_by_state(cpu_oracle.accepted_rows)
        and deferred_by_state == _row_count_by_state(cpu_oracle.deferred_rows)
    )


def _backlog_summary(
    backlog: dict[str, dict[int, dict[str, int]]],
    *,
    step: int,
) -> dict[str, int]:
    entries = [entry for by_index in backlog.values() for entry in by_index.values()]
    if not entries:
        return {
            "deferred_backlog_size": 0,
            "deferred_backlog_max_age_steps": 0,
            "deferred_backlog_max_defer_count": 0,
        }
    return {
        "deferred_backlog_size": len(entries),
        "deferred_backlog_max_age_steps": max(
            int(step) - int(entry["first_step"]) for entry in entries
        ),
        "deferred_backlog_max_defer_count": max(
            int(entry["defer_count"]) for entry in entries
        ),
    }


def _backlog_keys(
    backlog: dict[str, dict[int, dict[str, int]]],
) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(sorted(int(index) for index in by_index))
        for key, by_index in sorted(backlog.items())
    }


def _accepted_row_tensors(
    plan: VoteUpdatePlan,
    rows: tuple[GlobalRateCapRow, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int16, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    return (
        torch.tensor([row.flat_index for row in rows], dtype=torch.int64, device=device),
        torch.tensor(
            [int(plan.applied_directions[row.local_pos].item()) for row in rows],
            dtype=torch.int16,
            device=device,
        ),
        torch.tensor(
            [int(plan.applied_thresholds[row.local_pos].item()) for row in rows],
            dtype=torch.int32,
            device=device,
        ),
    )


def _normalize_tiny_loop_pattern_step(step: int, *, repeat_cycle: bool) -> int:
    step_i = int(step)
    if step_i < 1:
        raise ValueError(f"tiny full-loop step must be >= 1, got {step}")
    if step_i <= TINY_LOOP_STEP_COUNT:
        return step_i
    if not repeat_cycle:
        raise ValueError(f"tiny full-loop fixture has exactly two steps, got {step}")
    return ((step_i - 1) % TINY_LOOP_STEP_COUNT) + 1


def _votes_for_step(step: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    votes_in = torch.zeros((8, 16), dtype=torch.int16, device=device)
    votes_out = torch.zeros((4, 8), dtype=torch.int16, device=device)
    flat_in = votes_in.flatten()
    flat_out = votes_out.flatten()
    if step == 1:
        flat_in[0] = 6
        flat_in[1] = 4
        flat_in[2] = -3
        flat_in[3] = 2
        flat_out[0] = -7
        flat_out[1] = 5
        flat_out[2] = 2
    elif step == 2:
        # Indices 0 are saturated by step 1, so these high votes prove that the
        # second step consumes the mutated state instead of replaying step 1.
        flat_in[0] = 8
        flat_in[1] = 6
        flat_in[4] = -5
        flat_in[5] = 3
        flat_out[0] = -9
        flat_out[1] = 7
        flat_out[3] = -4
    else:
        raise ValueError(f"tiny full-loop fixture has exactly two steps, got {step}")
    return {"proj_in": votes_in.contiguous(), "proj_out": votes_out.contiguous()}


def _ternary_q_with_zeroed_rows(
    shape: tuple[int, int],
    *,
    zero_flat_indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    numel = int(shape[0] * shape[1])
    q = ((torch.arange(numel, device=device) % 3) - 1).to(torch.int8)
    if zero_flat_indices:
        q[torch.tensor(zero_flat_indices, dtype=torch.int64, device=device)] = 0
    return q.view(shape).contiguous()


def _tensor_offsets(qscale_states: dict[str, QScaleWeightState]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for key in _TENSOR_ORDER:
        offsets[key] = cursor
        cursor += int(qscale_states[key].q_levels.numel())
    return offsets


def _rows_by_state(rows: list[GlobalRateCapRow]) -> dict[str, tuple[GlobalRateCapRow, ...]]:
    grouped: dict[str, list[GlobalRateCapRow]] = {key: [] for key in _TENSOR_ORDER}
    for row in rows:
        grouped.setdefault(row.state_key, []).append(row)
    return {key: tuple(value) for key, value in grouped.items()}


def _row_count_by_state(rows: list[GlobalRateCapRow]) -> dict[str, int]:
    grouped = {key: 0 for key in _TENSOR_ORDER}
    for row in rows:
        grouped[row.state_key] = grouped.get(row.state_key, 0) + 1
    return grouped


def _state_hashes(
    qscale_states: dict[str, QScaleWeightState],
    accumulators: dict[str, torch.Tensor],
) -> dict[str, str]:
    return {
        key: _state_sha256(qscale_states[key], accumulators[key])
        for key in _TENSOR_ORDER
    }


def _state_sha256(state: QScaleWeightState, accumulators: torch.Tensor) -> str:
    h = hashlib.sha256()
    _hash_tensor(h, state.q_levels)
    _hash_tensor(h, accumulators)
    _hash_tensor(h, state.scale)
    return h.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    _hash_tensor(h, tensor)
    return h.hexdigest()


def _hash_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
