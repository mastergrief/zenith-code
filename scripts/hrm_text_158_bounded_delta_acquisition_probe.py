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
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import statistics
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
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
from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports
from calm.hrm_text_158.curriculum.language_supports import (
    _l0b_support,
    build_l0c1_support,
    build_l0c2k1_identity_full_support,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    DEFER_ALL_NO_BACKFILL_TIE_RULE_MODE,
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GLOBAL_CAP_CONTRACT_OFF,
    GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
    GLOBAL_TIE_RULE_MODES,
    GlobalRateCapOrderingMode,
    GlobalRateCapSpec,
    named_global_cap_contract_receipt,
    resolve_named_global_cap_spec,
)
from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    R7_SIDECAR_FILENAME,
    append_step_chunk,
    build_step_chunk,
    optional_selection_scores_from_step_result_compact,
    pressure_mass_from_tensor_states,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_live_carrier_snapshot import (
    emit_live_carrier_snapshots_for_probe_step,
    initialize_live_carrier_snapshot_log,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
    ReplayConstants,
    emit_event_coded_recompute_window_step_record,
    initialize_recompute_window_log_for_probe_session,
    maybe_emit_d_recompute_window_step_records,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_receipt_compact import (
    compact_d_diagnostic_step_result,
    should_apply_d_diagnostic_receipt_compaction,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_calibration_collector import (
    CalibrationWarmupCollector,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    StratifiedSelectorManifest,
    load_stratified_selector_manifest,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
    BOUNDED_DELTA_LEARNER_SCHEMA_VERSION,
    BOUNDED_UPDATE_ATTRIBUTION,
    S1_INVERTED_SIGN_PRESSURE_VOTE_LAW,
    S1_PROJECTION_LAW,
    S1_RANK_BUCKET_VOTE_LAW,
    S1_SIGN_PRESSURE_VOTE_LAW,
    VoteUpdateInputs,
    apply_bounded_delta_vote_step,
    authoritative_forward_context,
    build_authoritative_checkpoint_payload,
    build_optimizer_excluding_eligible_masters,
    compact_pressure_shape_summary,
    build_pressure_shape_summary_v1,
    compact_vote_pressure_summary,
    compact_sparse_vote_pressure_summary,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    derive_bounded_tensor_state_from_weight,
    file_sha256,
    make_bounded_tensor_state,
    make_event_coded_live_tensor_state,
    project_s1_gradient_to_moves,
    prove_eligible_master_identity_after_optimizer_step,
    rank_bucketed_int16_votes,
    rank_bucketed_int16_votes_and_sparse_events,
    sign_pressure_int16_votes,
    sign_pressure_int16_votes_and_sparse_events,
    sparse_rank_bucketed_int16_vote_events,
    sparse_rank_bucketed_int16_vote_events_from_weighted_grad,
    sparse_sign_pressure_int16_vote_events,
    tensor_sha256,
    validate_authoritative_resume_payload,
)
from calm.hrm_text_158.native_full_stack.front_c_live_identity_emission import (
    FrontCLiveIdentityCollector,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_CAPTURE_RECEIPT_KIND,
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
    TRACE_TEMPORALITY_SEQUENTIAL_OPTIMIZER_STEPS,
    TRACKING_SCOPE_OPTIMIZER_STEP_TRAJECTORY,
)
from calm.hrm_text_158.native_full_stack.b2b_capture_receipt_emission import (
    finalize_b2b_capture_receipt,
    rewrite_b2b_trace_with_receipt_emissions,
)
from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
    DRIFT_AUDIT_STEP_INTERVAL,
    POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
    assert_local_loss_delta_proxy_coverage,
    build_grad_proxy_local_loss_delta_by_key,
    count_w6_t10_crossing_eligible_from_votes,
    crossing_count_by_state_key_from_votes,
    run_proxy_oracle_drift_audit,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    capture_cuda_phase_memory_snapshot,
    install_probe_cuda_memory_snapshot_jsonl,
    ORACLE_SCREEN_MODE_CHOICES,
    ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT,
    ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE,
    ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
    ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
    ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR,
    _build_oracle_candidate_universe,
    _evaluate_sampled_candidates_for_activation_credit_oracle,
    capture_b2b_sequential_pre_update_step,
    run_activation_credit_measurement_oracle_screen,
    run_activation_credit_scale_smoke_oracle_screen,
    run_credit_ranking_pivot_measurement_oracle_screen,
    run_candidate_set_viability_oracle_screen,
    run_within_tie_band_discriminator_oracle_screen,
)
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ACTIVATION_CREDIT_STDERR_PATH_ENV,
    ACTIVATION_CREDIT_STDOUT_PATH_ENV,
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    ARM_INVERTED_SIGN_PRESSURE,
    FIXED_RANK_BUCKET_NON_TARGET_AUX,
    ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES,
    ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    TIE_POLICY_CURRENT_MARGIN_INDEX,
    TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
    oracle_screen_budget_max_seconds,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    pack_w5_lanes_to_bytes,
    pack_w6_lanes_to_bytes,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV,
    c8_runtime_guard_stats,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV,
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV,
    persistent_w5_byte_packed_enabled,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.w7_dense_acc_in_vivo_confirmation import (
    CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24,
    resolve_confirmation_envelope,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    R3_ARTIFACT_BYTES_SEMANTICS_ACTUAL_PAYLOAD,
    build_r3_per_module_payload_rows,
    build_r4_per_module_q_rows,
    measure_r3_persistent_state_budget,
    measure_r4_persistent_state_budget,
    measure_r4b_persistent_state_budget,
    measure_r4v_event_coded_acc_budget,
    measure_r5_persistent_state_budget,
    pack_ternary_q_2bit_reference,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    pack_ternary_q_base3_5perbyte_reference,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import QScaleWeightState
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV,
    PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV,
    Q_CODEC_SELECTOR_2BIT,
    Q_CODEC_SELECTOR_BASE3,
    persistent_q_ternary_base3_codec_enabled,
    persistent_q_ternary_byte_packed_enabled,
    persistent_w6_byte_packed_enabled,
    resolve_q_codec_selector,
)
from calm.hrm_text_158.native_full_stack.two_tier_transient_selection import (
    FORBIDDEN_PERSIST_SELECTOR_SURFACES,
    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
    crossing_eligible_flat_indices,
)
from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
    compact_probe_receipt_for_banking,
    validate_bankable_probe_receipt,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    HEADROOM_WIRING_SIDECAR_FILENAME,
    HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION,
    RECEIPT_EMIT_PROFILE_CHOICES,
    RECEIPT_EMIT_PROFILE_FULL,
    RECEIPT_EMIT_PROFILE_SLIM,
    S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE,
    SNAPSHOT_MODE_AGGREGATE_ONLY,
    SNAPSHOT_MODE_FULL,
    attach_s3bb_headroom_telemetry_to_step_report,
    initialize_headroom_wiring_sidecar_for_probe_session,
    run_vote_materialization_with_s3bb_boundary_catch,
)
from calm.hrm_text_158.native_full_stack.vote_update_emit_routing import (
    plan_vote_update_for_emit,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED,
    VoteUpdateSpec,
    plan_integer_vote_update_reference,
)
from scripts.train_hrm_text_158 import HrmTextGsm8kDataset


RUN_C2_ACQUISITION_PROBE_ENV = "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE"
RUN_C2_GPU_LAUNCH_ENV = "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"
C2P1_HARNESS_SCHEMA_VERSION = "hrm_text_158_c2p1_real_model_bounded_delta_probe/v0"
C2P2_TRAJECTORY_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_acquisition_trajectory/v0"
C2P2_AUDIT_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_strict_exact_audit/v0"
C2P2_SUPPORT_CYCLER_SCHEMA_VERSION = "hrm_text_158_c2p2_identity_full_support_cycler/v0"
C2P2_TIMING_SCHEMA_VERSION = "hrm_text_158_c2p2_calibration_timing_summary/v0"
C2P2_PHASE_TELEMETRY_SCHEMA_VERSION = "hrm_text_158_c2p2_phase_telemetry/v0"
PHASE_MILESTONE_COUNTER_SCHEMA = "hrm_text_158_phase_milestone_counter/v1"
MILESTONE_SPARSE_CAP_SUB_PHASE_IDS = frozenset({
    "cap_selection_cpu_copy",
    "post_cap_apply_sync",
    "boundary_normalize",
})
SPARSE_CAP_SUB_PHASE_MILESTONE_KINDS = {
    "cap_selection_cpu_copy": "cap_reference_cpu_shim_done",
    "post_cap_apply_sync": "module_cap_sync_done",
    "boundary_normalize": "module_boundary_normalize_done",
}
SPARSE_CAP_SUB_PHASE_JSONL_NAMES = {
    "cap_selection_cpu_copy": "sparse_cap_apply_cap_selection_cpu_copy.jsonl",
    "post_cap_apply_sync": "sparse_cap_apply_post_cap_apply_sync.jsonl",
    "boundary_normalize": "sparse_cap_apply_boundary_normalize.jsonl",
}
MILESTONE_BUDGETED_PHASE_IDS = frozenset({
    "step_forward_backward",
    "sparse_vote_construction",
    "sparse_cap_apply",
    "live_carrier_snapshot_emit",
    "artifact_flush",
})
PROBE_PHASE_TO_MILESTONE_PHASE_ID = {
    "step_forward_backward": "step_forward_backward",
    "sparse_vote_construction": "sparse_vote_construction",
    "sparse_cap_apply": "sparse_cap_apply",
    "live_carrier_snapshot_emit": "live_carrier_snapshot_emit",
    "receipt_write": "artifact_flush",
}
PHASE_BUDGET_INTERRUPT_AUTHORITY_SCHEMA = (
    "hrm_text_158_phase_budget_interrupt_authority/v1"
)
# C4.S1 Phase-3 first-milestone wall budgets are REPORT-ONLY progress telemetry.
# Sole fail-closed interrupt authority for active-phase silence is the faulthandler
# guard armed from max_silent_phase_seconds (default 600s on GPU launch).
PHASE3_C4S1_FIRST_MILESTONE_REPORT_ONLY_BUDGET_SECONDS: dict[str, float] = {
    "forward_backward": 90.0,
    "optimizer_update": 120.0,
    "sparse_cap_apply_emission": 180.0,
    "artifact_flush": 60.0,
}
PHASE3_C4S1_MILESTONE_TO_PROBE_PHASE_IDS: dict[str, tuple[str, ...]] = {
    "forward_backward": ("step_forward_backward",),
    "optimizer_update": ("sparse_vote_construction", "step_update"),
    "sparse_cap_apply_emission": ("sparse_cap_apply",),
    "artifact_flush": ("receipt_write", "artifact_flush"),
}
PROFILE_HOST_RSS_ENV = "HRM_TEXT_158_PROFILE_HOST_RSS"
PROFILE_HOST_RSS_LIVE_RESIDENT_ENV = "HRM_TEXT_158_PROFILE_HOST_RSS_LIVE_RESIDENT"
PROFILE_TORCH_CPU_CENSUS_ENV = "HRM_TEXT_158_PROFILE_TORCH_CPU_CENSUS"
PROFILE_ALLOCATOR_NATIVE_ENV = "HRM_TEXT_158_PROFILE_ALLOCATOR_NATIVE"
PROFILE_ALLOCATOR_HOST_CACHE_DIAG_ENV = "HRM_TEXT_158_PROFILE_ALLOCATOR_HOST_CACHE_DIAG"
PROFILE_ALLOC_HOOK_ENV = "HRM_TEXT_158_PROFILE_ALLOC_HOOK"
PROFILE_TRACEMALLOC_ENV = "HRM_TEXT_158_PROFILE_TRACEMALLOC"
PROFILE_DEBUGMALLOCSTATS_ENV = "HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS"
PROFILE_OBMALLOC_SITE_BRACKETS_ENV = "HRM_TEXT_158_PROFILE_OBMALLOC_SITE_BRACKETS"
PROFILE_OBMALLOC_EXPANDED_ENV = "HRM_TEXT_158_PROFILE_OBMALLOC_EXPANDED"
OBMALLOC_EXPANDED_SAMPLED_STATES_ENV = "HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES"
PROFILE_HOST_RSS_LIVE_RESIDENT_DROP_GIB = 1.0
PROFILE_HOST_RSS_SCHEMA = "hrm_text_158_profile_host_rss_mark/v1"
PROFILE_HOST_RSS_SUBPHASE_SCHEMA = "hrm_text_158_profile_host_rss_mark/v2"
PROFILE_HOST_RSS_CENSUS_SCHEMA = "hrm_text_158_profile_host_rss_mark/v3"
PROFILE_HOST_RSS_ALLOCATOR_SCHEMA = "hrm_text_158_profile_host_rss_mark/v4"
PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA = "hrm_text_158_profile_host_rss_mark/v5"
PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA = "hrm_text_158_profile_host_rss_mark/v6"
PROFILE_HOST_RSS_TRIANGULATION_SCHEMA = "hrm_text_158_profile_host_rss_mark/v7"
PROFILE_HOST_RSS_OBMALLOC_SCHEMA = "hrm_text_158_profile_host_rss_mark/v8"
PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA = "hrm_text_158_profile_host_rss_mark/v9"
PROFILE_S1D7_TRACEMALLOC_SITE_SCHEMA = "hrm_text_158_s1d7_tracemalloc_site/v1"
PROFILE_S1D7_BAND_COUNTER_SITE_SCHEMA = "hrm_text_158_s1d7_band_counter_site/v1"
PROFILE_HOST_RSS_SUBPHASE_IDS = frozenset({
    "C1_vote_plan_build",
    "C2_cap_input_assembly",
    "C3_gpu_cap_selection",
    "C4_gpu_cap_apply_sync",
    "C5_next_state_materialize",
    "C6_deferred_backlog_telemetry",
})
PROFILE_HOST_RSS_PHASES = frozenset({
    "step_forward_backward",
    "step_update",
    "sparse_vote_construction",
    "sparse_cap_apply",
    "live_carrier_snapshot_emit",
    "receipt_write",
})
HOST_RSS_PROFILE_JSONL_NAME = "host_rss_profile.jsonl"
C2P2_DEVICE_GUARD_SCHEMA_VERSION = "hrm_text_158_c2p2_device_guard/v0"
C2P2_FAULTHANDLER_SCHEMA_VERSION = "hrm_text_158_faulthandler_guard/v0"
B1_PRIOR_AUDIT_SCHEMA_VERSION = "hrm_text_158_b1_prior_support_audit/v0"
B1_PRIOR_SUPPORT_SCHEMA_VERSION = "hrm_text_158_b1_prior_support_adapter/v0"
B1_PRIOR_AUDIT_DELTA_SCHEMA_VERSION = "hrm_text_158_b1_prior_support_delta/v0"
B2_RETAINED_SUPPORT_SCHEMA_VERSION = "hrm_text_158_b2_retained_support_vote_aux/v0"
B2_FULL_VERDICT_SCHEMA_VERSION = "hrm_text_158_b2_full_retention_verdict/v0"
B2_RETAINED_SUPPORTS: tuple[str, ...] = ("L0b", "math_a0")
B2_FULL_STOP_SUPPORTS: tuple[str, ...] = ("L0b", "math_a0")
B2_PC_AUX_MODES: tuple[str, ...] = ("telemetry", "veto")
SCIENCE_ARM_CHOICES: tuple[str, ...] = (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER,
    ARM_INVERTED_SIGN_PRESSURE,
)
SCIENCE_LOCAL_SELECTION_ORDERING_SEED = 17
C2P2_STRICT_EXACT_TARGET = 90
C2P2_DEFAULT_MAX_STEPS_HARD = 1500
C2P2_DEFAULT_GPU_SILENT_PHASE_TIMEOUT_SECONDS = 300.0
C2P2_DEFAULT_PHASE_HEARTBEAT_INTERVAL_SECONDS = 30.0
C2P2_LONGEST_QUIET_PHASE_REFERENCE_SECONDS = 80.0
C2P2_MIN_WATCH_WRAP_HEARTBEAT_SECONDS = 120.0
C2P2_PROBE_STDOUT_LIVENESS_SCHEMA_VERSION = "hrm_text_158_probe_stdout_liveness/v1"
PHASE_TIMEOUT_EXEMPTION_SCHEMA_VERSION = "hrm_text_158_phase_timeout_exemption/v0"
PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF = "off"
B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1 = (
    "b2b_bounded_steps_aggregate_timeout_exemption_v1"
)
PHASE_TIMEOUT_EXEMPTION_CONTRACT_CHOICES = (
    PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF,
    B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1,
)
BOUNDED_STEPS_AGGREGATE_PHASE = "bounded_steps"
PROBE_RUN_LOG_NAME = "run.log"
PROBE_EXIT_CODE_ARTIFACT_NAME = "probe.exit_code.txt"
PARENT_CHECKPOINT_POSTHASH_ARTIFACT_NAME = "parent_checkpoint_posthash.json"
C2P2_PARENT_CHECKPOINT_POSTHASH_SCHEMA_VERSION = (
    "hrm_text_158_parent_checkpoint_posthash/v0"
)
C2P2_NULL_TAXONOMY = (
    "no-q-move",
    "q-move-no-accuracy",
    "partial-acquisition-plateau",
    "nonfinite",
    "instability-divergence",
    "audit-mismatch",
    "runtime-resource-failure",
)


class _MirrorTextStream:
    def __init__(
        self,
        *streams: Any,
        fileno_stream: Any | None = None,
    ) -> None:
        self._streams = streams
        self._fileno_stream = fileno_stream
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def fileno(self) -> int:
        self.flush()
        if self._fileno_stream is None:
            raise AttributeError("_MirrorTextStream has no fileno delegate")
        return int(self._fileno_stream.fileno())

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def writable(self) -> bool:
        return True


def install_probe_durable_run_log(scratch_root: Path) -> Path:
    """Mirror stdout/stderr to ``$RUN_ROOT/run.log`` for the probe process lifetime."""

    log_path = Path(scratch_root) / PROBE_RUN_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _MirrorTextStream(
        sys.stdout,
        log_file,
        fileno_stream=log_file,
    )
    sys.stderr = _MirrorTextStream(
        sys.stderr,
        log_file,
        fileno_stream=log_file,
    )
    return log_path


def _write_liveness_stack_dump(
    *,
    dump_path: Path,
    guard_event: str,
    phase: str,
    payload: Mapping[str, Any],
) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("w", encoding="utf-8") as dump_file:
        dump_file.write(f"guard_event={guard_event}\n")
        dump_file.write(f"phase={phase}\n")
        dump_file.write(json.dumps(dict(payload), sort_keys=True, default=str))
        dump_file.write("\n")
        faulthandler.dump_traceback(file=dump_file, all_threads=True)


@contextmanager
def activation_credit_env_log_capture() -> Any:
    stdout_path = os.environ.get(ACTIVATION_CREDIT_STDOUT_PATH_ENV)
    stderr_path = os.environ.get(ACTIVATION_CREDIT_STDERR_PATH_ENV)
    if not stdout_path and not stderr_path:
        yield
        return
    with ExitStack() as stack:
        stdout_target = sys.stdout
        stderr_target = sys.stderr
        if stdout_path:
            stdout_file = Path(stdout_path)
            stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stack.enter_context(
                stdout_file.open("w", encoding="utf-8", buffering=1)
            )
            stdout_target = _MirrorTextStream(
                sys.stdout,
                stdout_handle,
                fileno_stream=stdout_handle,
            )
        if stderr_path:
            stderr_file = Path(stderr_path)
            stderr_file.parent.mkdir(parents=True, exist_ok=True)
            stderr_handle = stack.enter_context(
                stderr_file.open("w", encoding="utf-8", buffering=1)
            )
            stderr_target = _MirrorTextStream(
                sys.stderr,
                stderr_handle,
                fileno_stream=stderr_handle,
            )
        stack.enter_context(redirect_stdout(stdout_target))
        stack.enter_context(redirect_stderr(stderr_target))
        yield
C2P2_NULL_ESCALATION_RULE = (
    "If C2.2 returns a classified null, escalate inside the same harness with "
    "an inline int16/dense-acc control; historical receipt 1779747988676 is "
    "context only, not a same-harness paired control."
)
GLOBAL_CAP_CONTRACT_CHOICES = (
    GLOBAL_CAP_CONTRACT_OFF,
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
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
B1_PRIOR_AUDIT_SUPPORTS: tuple[str, ...] = ("L0b", "math_a0", "L0c1")
B1_PRIOR_AUDIT_FIXED_SEED = 17
B1_PRIOR_AUDIT_PINS: dict[str, dict[str, Any]] = {
    "L0b": {
        "expected_count": 230,
        "expected_hash16": "89174273d21845bc",
        "builder_path": "calm.hrm_text_158.curriculum.language_supports._l0b_support",
        "support_role": "true_prior",
    },
    "math_a0": {
        "expected_count": 1255,
        "expected_hash16": "56e64266357b793d",
        "builder_path": "calm.hrm_text_158.curriculum.exhaustive_supports.build_exhaustive_supports",
        "support_role": "true_prior",
    },
    "L0c1": {
        "expected_count": 121,
        "expected_hash16": "7bc8cd771daab878",
        "builder_path": "calm.hrm_text_158.curriculum.language_supports.build_l0c1_support",
        "support_role": "close_wrapper_report_only",
    },
}
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


def _support_order_flat_row_ids(support_batches: Sequence[Mapping[str, Any]]) -> list[str]:
    row_ids: list[str] = []
    for batch in support_batches:
        metadata = batch.get("metadata", {})
        row_ids.extend(str(row_id) for row_id in metadata.get("row_ids", ()))
    return row_ids


def _support_ordered_traversal_hash16(
    support_batches: Sequence[Mapping[str, Any]],
) -> str:
    return _sha16(_support_order_flat_row_ids(support_batches))


def _support_order_invariant_multiset_hash16(
    support_batches: Sequence[Mapping[str, Any]],
) -> str:
    return _sha16(sorted(_support_order_flat_row_ids(support_batches)))


def build_support_order_trajectory_proof(
    original_support_batches: Sequence[Mapping[str, Any]],
    traversed_support_batches: Sequence[Mapping[str, Any]],
    *,
    support_order_seed: int | None,
) -> dict[str, Any]:
    original_ordered = _support_ordered_traversal_hash16(original_support_batches)
    traversed_ordered = _support_ordered_traversal_hash16(traversed_support_batches)
    original_invariant = _support_order_invariant_multiset_hash16(original_support_batches)
    traversed_invariant = _support_order_invariant_multiset_hash16(traversed_support_batches)
    return {
        "support_order_seed": None if support_order_seed is None else int(support_order_seed),
        "support_order_permutation_enabled": support_order_seed is not None,
        "support_order_original_ordered_traversal_hash16": original_ordered,
        "support_order_permuted_ordered_traversal_hash16": traversed_ordered,
        "support_order_original_invariant_multiset_hash16": original_invariant,
        "support_order_permuted_invariant_multiset_hash16": traversed_invariant,
        "support_order_changed": original_ordered != traversed_ordered,
        "support_content_unchanged": original_invariant == traversed_invariant,
        "support_content_unchanged_basis": "order_invariant_multiset_hash16",
        "legacy_support_content_hash16_semantics": "ordered_batch_hashes_order_sensitive",
        "ordered_support_content_hash16_is_invariant": False,
        "support_order_original_first_row_ids": _support_order_flat_row_ids(original_support_batches)[:8],
        "support_order_permuted_first_row_ids": _support_order_flat_row_ids(traversed_support_batches)[:8],
    }


def _permute_support_batches(
    support_batches: Sequence[dict[str, Any]],
    *,
    support_order_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(support_order_seed))
    permuted_indices = list(range(len(support_batches)))
    rng.shuffle(permuted_indices)
    permuted: list[dict[str, Any]] = []
    for new_index, original_index in enumerate(permuted_indices):
        item = support_batches[original_index]
        metadata = dict(item["metadata"])
        metadata["original_batch_index"] = int(metadata.get("batch_index", original_index))
        metadata["batch_index"] = int(new_index)
        metadata["support_order_position"] = int(new_index)
        metadata["support_order_original_position"] = int(original_index)
        permuted.append(
            {
                **item,
                "metadata": metadata,
            }
        )
    return permuted


def build_identity_full_support_batches(
    *,
    tok: Any,
    max_len: int,
    batch_size: int,
    curriculum_seed: int,
    device: torch.device,
    support_order_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if support_order_seed is not None and int(support_order_seed) < 0:
        raise ValueError("support_order_seed must be non-negative when set")
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
    original_support_batches = list(support_batches)
    if support_order_seed is not None:
        support_batches = _permute_support_batches(
            support_batches,
            support_order_seed=int(support_order_seed),
        )
    support_order_proof = build_support_order_trajectory_proof(
        original_support_batches,
        support_batches,
        support_order_seed=support_order_seed,
    )
    if (
        support_order_seed is not None
        and len(support_batches) > 1
        and not support_order_proof["support_order_changed"]
    ):
        raise RuntimeError(
            "support-order permutation did not change traversal order; "
            f"seed={int(support_order_seed)} batch_count={len(support_batches)}"
        )
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
        **support_order_proof,
        "first_questions": [row["question"] for row in rows[: min(3, len(rows))]],
        "batch_metadata": [
            item["metadata"]
            for item in support_batches
        ],
    }
    return support_batches, proof


def parse_prior_audit_supports(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    unknown = [part for part in parts if part not in B1_PRIOR_AUDIT_SUPPORTS]
    if unknown:
        raise ValueError(
            f"unknown prior audit support(s) {unknown}; valid: {B1_PRIOR_AUDIT_SUPPORTS}"
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"duplicate prior audit support(s): {parts}")
    return tuple(parts)


def parse_b2_retained_supports(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    unknown = [part for part in parts if part not in B2_RETAINED_SUPPORTS]
    if unknown:
        raise ValueError(
            f"unknown B2 retained support(s) {unknown}; valid: {B2_RETAINED_SUPPORTS}. "
            "L0c1 is report-only in B2.0."
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"duplicate B2 retained support(s): {parts}")
    return tuple(parts)


def make_front_c_identity_observer_for_step(
    front_c_identity_collector: FrontCLiveIdentityCollector | None,
    *,
    step: int,
    total_steps: int,
) -> Callable[[Mapping[str, Any]], None] | None:
    if front_c_identity_collector is None:
        return None
    if not front_c_identity_collector.should_collect_step(
        int(step),
        total_steps=int(total_steps),
    ):
        return None

    def front_c_identity_observer(
        observation: Mapping[str, Any],
        *,
        observed_step: int = int(step),
    ) -> None:
        front_c_identity_collector.record_step_observation(
            step=observed_step,
            observation=observation,
            collect=True,
        )

    return front_c_identity_observer


def _prior_support_sorted_rows(name: str, curriculum_seed: int) -> list[tuple[str, int, str]]:
    if name == "L0b":
        rows = [(q, e, source_rung) for (q, e, source_rung) in _l0b_support(int(curriculum_seed))]
    elif name == "math_a0":
        rows = [
            (q, e, rung)
            for rung, pairs in build_exhaustive_supports().items()
            for (q, e) in pairs
        ]
    elif name == "L0c1":
        rows = [
            (q, e, bucket)
            for _surface, pairs in build_l0c1_support(int(curriculum_seed)).items()
            for (q, e, bucket) in pairs
        ]
    else:
        raise ValueError(
            f"unknown prior audit support {name!r}; valid: {B1_PRIOR_AUDIT_SUPPORTS}"
        )
    return sorted(rows, key=lambda row: (row[2], row[0], row[1]))


def build_prior_audit_support_rows(
    name: str,
    *,
    run_curriculum_seed: int,
    use_fixed_audit_seed: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pins = B1_PRIOR_AUDIT_PINS[name]
    construction_seed = (
        B1_PRIOR_AUDIT_FIXED_SEED if use_fixed_audit_seed else int(run_curriculum_seed)
    )
    sorted_rows = _prior_support_sorted_rows(name, int(construction_seed))
    row_count = len(sorted_rows)
    support_hash16 = hashlib.sha256(repr(sorted_rows).encode("utf-8")).hexdigest()[:16]
    if row_count != int(pins["expected_count"]) or support_hash16 != pins["expected_hash16"]:
        raise RuntimeError(
            f"prior audit support pin mismatch for {name}: "
            f"count/hash {row_count}/{support_hash16} != "
            f"{pins['expected_count']}/{pins['expected_hash16']}"
        )
    rows = [
        {
            "question": question,
            "expected": expected,
            "rung": name,
            "source_rung": source_rung,
            "prior_audit_support": name,
            "_support_index": index,
        }
        for index, (question, expected, source_rung) in enumerate(sorted_rows)
    ]
    source_counts: dict[str, int] = {}
    for _question, _expected, source_rung in sorted_rows:
        source_counts[source_rung] = source_counts.get(source_rung, 0) + 1
    proof = {
        "schema": B1_PRIOR_SUPPORT_SCHEMA_VERSION,
        "support": name,
        "support_role": pins["support_role"],
        "audit_seed": int(construction_seed),
        "run_curriculum_seed": int(run_curriculum_seed),
        "builder_path": pins["builder_path"],
        "row_count": row_count,
        "expected_count": int(pins["expected_count"]),
        "support_hash16": support_hash16,
        "expected_hash16": pins["expected_hash16"],
        "pinned_count_hash_pass": True,
        "source_bucket_counts": dict(sorted(source_counts.items())),
        "report_only": True,
        "direct_kl": False,
        "replay_pc": "OUT",
        "target_parent_kl": False,
    }
    if name == "L0c1":
        proof["close_wrapper_report_only"] = {
            "direct_kl": False,
            "replay_pc": "OUT",
            "target_parent_kl": False,
        }
    return rows, proof


def build_prior_audit_support_batches(
    *,
    support: str,
    tok: Any,
    max_len: int,
    batch_size: int,
    run_curriculum_seed: int,
    device: torch.device,
    use_fixed_audit_seed: bool = False,
) -> dict[str, Any]:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows, proof = build_prior_audit_support_rows(
        support,
        run_curriculum_seed=int(run_curriculum_seed),
        use_fixed_audit_seed=use_fixed_audit_seed,
    )
    usable_rows = _identity_full_usable_rows(rows, tok=tok, max_len=int(max_len))
    dataset = HrmTextGsm8kDataset(
        rows,
        tok,
        max_len=int(max_len),
        curriculum_rung=support,
    )
    if len(dataset) != len(usable_rows):
        raise RuntimeError(
            f"prior audit metadata/tensor row mismatch for {support}: "
            f"dataset={len(dataset)} usable_metadata={len(usable_rows)}"
        )
    if len(dataset) != int(proof["expected_count"]):
        raise RuntimeError(
            f"prior audit support {support} dropped rows under max_len={max_len}: "
            f"usable={len(dataset)} expected={proof['expected_count']}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=_collate,
    )
    batches: list[dict[str, Any]] = []
    row_offset = 0
    for batch_index, collated in enumerate(loader):
        row_count = int(collated["inputs"].shape[0])
        batch_rows = usable_rows[row_offset: row_offset + row_count]
        source_buckets = [str(row["source_rung"]) for row in batch_rows]
        source_counts: dict[str, int] = {}
        for source_bucket in source_buckets:
            source_counts[source_bucket] = source_counts.get(source_bucket, 0) + 1
        metadata = _support_batch_metadata(batch_index=batch_index, rows=batch_rows)
        batches.append(
            {
                "batch": _model_batch_from_collated(collated, device=device),
                "metadata": {
                    **metadata,
                    "support": support,
                    "row_start": int(row_offset),
                    "row_end_exclusive": int(row_offset + row_count),
                    "source_buckets": source_buckets,
                    "source_bucket_counts": dict(sorted(source_counts.items())),
                },
            }
        )
        row_offset += row_count
    distinct_batch_hashes = {
        item["metadata"]["batch_content_hash16"]
        for item in batches
    }
    proof = {
        **proof,
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "distinct_batch_count": len(distinct_batch_hashes),
        "support_content_hash16": _sha16(
            [
                batch["metadata"]["batch_content_hash16"]
                for batch in batches
            ]
        ),
        "batch_metadata": [
            item["metadata"]
            for item in batches
        ],
    }
    return {"support": support, "batches": batches, "proof": proof}


def build_prior_audit_support_sets(
    supports: Sequence[str],
    *,
    tok: Any,
    max_len: int,
    batch_size: int,
    run_curriculum_seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    return {
        support: build_prior_audit_support_batches(
            support=support,
            tok=tok,
            max_len=int(max_len),
            batch_size=int(batch_size),
            run_curriculum_seed=int(run_curriculum_seed),
            device=device,
            use_fixed_audit_seed=True,
        )
        for support in supports
    }


def build_b2_retained_support_sets(
    supports: Sequence[str],
    *,
    tok: Any,
    max_len: int,
    support_batch_sizes: Mapping[str, int],
    curriculum_seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    retained: dict[str, dict[str, Any]] = {}
    for support in supports:
        batch_size = int(support_batch_sizes[support])
        if batch_size <= 0:
            raise ValueError(f"B2 retained support {support} batch size must be positive")
        support_set = build_prior_audit_support_batches(
            support=support,
            tok=tok,
            max_len=int(max_len),
            batch_size=batch_size,
            run_curriculum_seed=int(curriculum_seed),
            device=device,
        )
        proof = dict(support_set["proof"])
        proof.update(
            {
                "schema": B2_RETAINED_SUPPORT_SCHEMA_VERSION,
                "support_role": "retained_true_prior",
                "report_only": False,
                "replay_ce_veto": True,
                "pc_aux_eligible": True,
                "target_parent_kl": False,
                "l0c1_report_only_b2": True,
            }
        )
        support_set = dict(support_set)
        support_set["proof"] = proof
        retained[support] = support_set
    return retained


def new_b2_full_coverage_tracker(
    rows_total_by_support: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Track disjoint support-pass coverage for B2-full retention verdicts."""
    tracker: dict[str, dict[str, Any]] = {}
    for support, rows_total in rows_total_by_support.items():
        total = int(rows_total)
        if total <= 0:
            raise ValueError(f"B2-full coverage rows_total must be positive for {support}")
        tracker[str(support)] = {
            "rows_total": total,
            "rows_seen_total": 0,
            "rows_seen_unique": set(),
            "current_cycle_row_ids": set(),
            "coverage_cycles": 0,
        }
    return tracker


def update_b2_full_coverage_tracker(
    tracker: dict[str, dict[str, Any]],
    *,
    support: str,
    row_ids: Sequence[str],
) -> dict[str, Any]:
    if support not in tracker:
        raise KeyError(f"B2-full coverage tracker missing support {support!r}")
    item = tracker[support]
    rows_total = int(item["rows_total"])
    current_cycle = item["current_cycle_row_ids"]
    for row_id in row_ids:
        normalized = str(row_id)
        item["rows_seen_total"] = int(item["rows_seen_total"]) + 1
        item["rows_seen_unique"].add(normalized)
        current_cycle.add(normalized)
        if len(current_cycle) >= rows_total:
            item["coverage_cycles"] = int(item["coverage_cycles"]) + 1
            item["current_cycle_row_ids"] = set()
            current_cycle = item["current_cycle_row_ids"]
    return snapshot_b2_full_coverage_tracker(tracker)[support]


def snapshot_b2_full_coverage_tracker(
    tracker: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for support, item in sorted(tracker.items()):
        unique_rows = item.get("rows_seen_unique", set())
        current_cycle = item.get("current_cycle_row_ids", set())
        cycles = int(item.get("coverage_cycles", 0))
        rows_total = int(item["rows_total"])
        snapshot[support] = {
            # Legacy names kept for B2 receipt readers.
            "rows_seen": len(unique_rows),
            "rows_total": rows_total,
            "coverage_cycle_complete": cycles >= 1,
            # B2-full first-class cycle accounting.
            "rows_seen_unique": len(unique_rows),
            "rows_seen_total": int(item.get("rows_seen_total", 0)),
            "rows_seen_current_cycle": len(current_cycle),
            "coverage_cycles": cycles,
            "coverage_gate_met": cycles >= 1,
        }
    return snapshot


def b2_full_coverage_cycles(
    coverage_by_support: Mapping[str, Mapping[str, Any]],
    support: str,
) -> int:
    coverage = coverage_by_support.get(support, {})
    return int(coverage.get("coverage_cycles", 0))


def b2_full_coverage_gate_met(
    coverage_by_support: Mapping[str, Mapping[str, Any]],
    *,
    support: str = "math_a0",
    required_cycles: int = 1,
) -> bool:
    return b2_full_coverage_cycles(coverage_by_support, support) >= int(required_cycles)


def b2_full_target_gate_met(
    target_audit: Mapping[str, Any],
    *,
    target_count: int = C2P2_STRICT_EXACT_TARGET,
) -> bool:
    if target_audit.get("acquired") is True:
        return True
    if "strict_exact_count" not in target_audit:
        return False
    return int(target_audit["strict_exact_count"]) >= int(target_count)


def new_b2_full_verdict_state() -> dict[str, Any]:
    return {
        "schema": B2_FULL_VERDICT_SCHEMA_VERSION,
        "enabled": True,
        "coverage_support": "math_a0",
        "required_coverage_cycles": 1,
        "stop_supports": list(B2_FULL_STOP_SUPPORTS),
        "report_only_supports": ["L0c1"],
        "first_audited_target_ge_90": None,
        "first_covered_target_ge_90": None,
        "terminal": None,
        "snapshot_steps": {},
        "prior_audit_executions": [],
        "prior_audit_count": 0,
        "audit_export_paths": {},
        "combined_stop": {
            "triggered": False,
            "step": None,
            "reason": None,
        },
        "verdict": "pending",
    }


def b2_full_required_snapshot_names(
    state: Mapping[str, Any],
    *,
    target_audit: Mapping[str, Any],
    coverage_by_support: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not b2_full_target_gate_met(target_audit):
        return []
    names: list[str] = []
    if state.get("first_audited_target_ge_90") is None:
        names.append("first_audited_target_ge_90")
    if (
        state.get("first_covered_target_ge_90") is None
        and b2_full_coverage_gate_met(coverage_by_support)
    ):
        names.append("first_covered_target_ge_90")
    return names


def build_b2_full_prior_snapshot(
    *,
    snapshot_name: str,
    step: int,
    target_audit: Mapping[str, Any],
    coverage_by_support: Mapping[str, Mapping[str, Any]],
    start_reports: Mapping[str, Mapping[str, Any]],
    current_reports: Mapping[str, Mapping[str, Any]],
    stop_supports: Sequence[str] = B2_FULL_STOP_SUPPORTS,
    report_only_supports: Sequence[str] = ("L0c1",),
) -> dict[str, Any]:
    deltas = {
        support: build_prior_audit_delta(
            support=support,
            start_report=start_reports[support],
            final_report=current_reports[support],
        )
        for support in current_reports
        if support in start_reports
    }
    stop_status = {
        support: bool(deltas.get(support, {}).get("no_new_broad_cluster", False))
        for support in stop_supports
        if support in deltas
    }
    stop_supports_present = all(support in stop_status for support in stop_supports)
    retained_true_priors_pass = bool(stop_supports_present and all(stop_status.values()))
    math_cycles = b2_full_coverage_cycles(coverage_by_support, "math_a0")
    l0b_cycles = b2_full_coverage_cycles(coverage_by_support, "L0b")
    target_gate = b2_full_target_gate_met(target_audit)
    coverage_gate = b2_full_coverage_gate_met(coverage_by_support)
    return {
        "schema": B2_FULL_VERDICT_SCHEMA_VERSION,
        "snapshot_name": snapshot_name,
        "step": int(step),
        "target_gate_met": bool(target_gate),
        "target_audit": dict(target_audit),
        "coverage_gate_met": bool(coverage_gate),
        "math_a0_coverage_cycles": int(math_cycles),
        "l0b_coverage_cycles": int(l0b_cycles),
        "coverage_by_support": {
            support: dict(coverage)
            for support, coverage in coverage_by_support.items()
        },
        "prior_report_summary": {
            support: {
                "strict_exact": report.get("strict_exact"),
                "strict_exact_count": report.get("strict_exact_count"),
                "parsed_exact": report.get("parsed_exact"),
                "parsed_exact_count": report.get("parsed_exact_count"),
                "duration_seconds": report.get("duration_seconds"),
            }
            for support, report in current_reports.items()
        },
        "deltas": deltas,
        "stop_supports": list(stop_supports),
        "stop_support_status": stop_status,
        "report_only_supports": list(report_only_supports),
        "retained_true_priors_no_new_broad_cluster": retained_true_priors_pass,
        "combined_stop_pass": bool(
            target_gate and coverage_gate and retained_true_priors_pass
        ),
    }


def record_b2_full_prior_snapshot(
    state: dict[str, Any],
    *,
    snapshot_names: Sequence[str],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    names = [str(name) for name in snapshot_names]
    if not names:
        return state
    execution = {
        "step": int(snapshot["step"]),
        "snapshots": names,
    }
    state.setdefault("prior_audit_executions", []).append(execution)
    state["prior_audit_count"] = len(state["prior_audit_executions"])
    snapshot_steps = state.setdefault("snapshot_steps", {})
    for name in names:
        named_snapshot = dict(snapshot)
        named_snapshot["snapshot_name"] = name
        state[name] = named_snapshot
        snapshot_steps[name] = int(snapshot["step"])
        if name == "first_covered_target_ge_90" and named_snapshot["combined_stop_pass"]:
            state["combined_stop"] = {
                "triggered": True,
                "step": int(snapshot["step"]),
                "reason": "b2_full_target_coverage_retain_pass",
            }
    return state


def summarize_b2_full_prior_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    deltas = {
        support: {
            "broad_cluster_classification": delta.get("broad_cluster_classification"),
            "no_new_broad_cluster": delta.get("no_new_broad_cluster"),
            "new_strict_failure_count": delta.get("new_strict_failure_count"),
            "new_parsed_failure_count": delta.get("new_parsed_failure_count"),
            "parent_baseline_vs_final": delta.get("parent_baseline_vs_final"),
        }
        for support, delta in snapshot.get("deltas", {}).items()
    }
    return {
        "snapshot_name": snapshot.get("snapshot_name"),
        "step": snapshot.get("step"),
        "target_gate_met": snapshot.get("target_gate_met"),
        "coverage_gate_met": snapshot.get("coverage_gate_met"),
        "math_a0_coverage_cycles": snapshot.get("math_a0_coverage_cycles"),
        "l0b_coverage_cycles": snapshot.get("l0b_coverage_cycles"),
        "retained_true_priors_no_new_broad_cluster": snapshot.get(
            "retained_true_priors_no_new_broad_cluster"
        ),
        "combined_stop_pass": snapshot.get("combined_stop_pass"),
        "deltas": deltas,
    }


def finalize_b2_full_verdict_state(
    state: dict[str, Any],
    *,
    terminal_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    record_b2_full_prior_snapshot(
        state,
        snapshot_names=["terminal"],
        snapshot=terminal_snapshot,
    )
    state["math_a0_coverage_cycles"] = int(
        terminal_snapshot.get("math_a0_coverage_cycles", 0)
    )
    state["l0b_coverage_cycles"] = int(
        terminal_snapshot.get("l0b_coverage_cycles", 0)
    )
    first = state.get("first_audited_target_ge_90")
    terminal = state.get("terminal")
    if first is None:
        verdict = "no-target-acquisition"
    elif (
        terminal
        and terminal.get("target_gate_met")
        and terminal.get("coverage_gate_met")
        and terminal.get("retained_true_priors_no_new_broad_cluster")
    ):
        verdict = "RETAINS"
    elif terminal and not terminal.get("target_gate_met"):
        verdict = "acquire-then-forget"
    else:
        verdict = "no-retain"
    state["verdict"] = verdict
    return state


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
        **fields: Any,
    ) -> None:
        self.payload = {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "event": "phase_timeout",
            "phase": str(phase),
            "bound_kind": str(bound_kind),
            "duration_seconds": float(duration_seconds),
            "timeout_seconds": float(timeout_seconds),
            **fields,
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


def build_phase_budget_interrupt_authority_contract(
    *,
    silent_phase_timeout_seconds: float | int | None = None,
    max_silent_phase_seconds: float | int | None = None,
) -> dict[str, Any]:
    """Encode C4: milestone budgets are report-only; faulthandler silent guard interrupts."""
    interrupt_seconds = _timeout_or_none(
        silent_phase_timeout_seconds
        if silent_phase_timeout_seconds is not None
        else max_silent_phase_seconds
    )
    return {
        "schema": PHASE_BUDGET_INTERRUPT_AUTHORITY_SCHEMA,
        "first_milestone_budgets_report_only": True,
        "first_milestone_budget_seconds": dict(
            PHASE3_C4S1_FIRST_MILESTONE_REPORT_ONLY_BUDGET_SECONDS
        ),
        "interrupt_authority": "faulthandler_silent_phase_guard",
        "interrupt_timeout_seconds": interrupt_seconds,
        "phase_timeout_scalar_is_aggregate_cap_not_milestone_interrupt": True,
        "milestone_budget_breach_triggers_interrupt": False,
    }


def evaluate_first_milestone_budget_report_only(
    milestone_phase_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Report milestone budget breach without raising or arming fail-closed interrupt."""
    budget = PHASE3_C4S1_FIRST_MILESTONE_REPORT_ONLY_BUDGET_SECONDS.get(
        str(milestone_phase_id)
    )
    if budget is None:
        return {
            "milestone_phase_id": str(milestone_phase_id),
            "budget_seconds": None,
            "elapsed_seconds": float(elapsed_seconds),
            "budget_breached": False,
            "report_only": True,
            "triggers_interrupt": False,
            "known_milestone": False,
        }
    breached = float(elapsed_seconds) > float(budget)
    return {
        "milestone_phase_id": str(milestone_phase_id),
        "budget_seconds": float(budget),
        "elapsed_seconds": float(elapsed_seconds),
        "budget_breached": breached,
        "report_only": True,
        "triggers_interrupt": False,
        "known_milestone": True,
    }


def first_milestone_budget_breach_triggers_interrupt() -> bool:
    return False


def resolve_max_silent_phase_seconds(
    *,
    allow_gpu_launch: bool,
    max_silent_phase_seconds: float | int | None,
) -> float | None:
    if max_silent_phase_seconds is not None:
        return _timeout_or_none(max_silent_phase_seconds)
    if bool(allow_gpu_launch):
        return C2P2_DEFAULT_GPU_SILENT_PHASE_TIMEOUT_SECONDS
    return None


def resolve_phase_heartbeat_seconds(
    *,
    emit_progress: bool,
    phase_heartbeat_seconds: float | int | None,
) -> float | None:
    if phase_heartbeat_seconds is not None:
        return _timeout_or_none(phase_heartbeat_seconds)
    if bool(emit_progress):
        return C2P2_DEFAULT_PHASE_HEARTBEAT_INTERVAL_SECONDS
    return None


def recommended_watch_wrap_heartbeat_seconds() -> float:
    return max(
        float(C2P2_MIN_WATCH_WRAP_HEARTBEAT_SECONDS),
        float(C2P2_LONGEST_QUIET_PHASE_REFERENCE_SECONDS) * 1.5,
    )


def build_probe_stdout_liveness_receipt(
    *,
    emit_progress: bool,
    phase_heartbeat_seconds: float | int | None,
    watch_wrap_heartbeat_seconds: float | int | None = None,
) -> dict[str, Any]:
    resolved_heartbeat = resolve_phase_heartbeat_seconds(
        emit_progress=bool(emit_progress),
        phase_heartbeat_seconds=phase_heartbeat_seconds,
    )
    watch_wrap_budget = (
        float(watch_wrap_heartbeat_seconds)
        if watch_wrap_heartbeat_seconds is not None
        else recommended_watch_wrap_heartbeat_seconds()
    )
    return {
        "schema": C2P2_PROBE_STDOUT_LIVENESS_SCHEMA_VERSION,
        "emit_progress": bool(emit_progress),
        "phase_heartbeat_seconds": resolved_heartbeat,
        "longest_quiet_phase_reference_seconds": float(
            C2P2_LONGEST_QUIET_PHASE_REFERENCE_SECONDS
        ),
        "longest_quiet_phase_name": "proxy_oracle_drift_audit",
        "watch_wrap_heartbeat_seconds": watch_wrap_budget,
        "watch_wrap_heartbeat_exceeds_longest_quiet_phase": bool(
            watch_wrap_budget > float(C2P2_LONGEST_QUIET_PHASE_REFERENCE_SECONDS)
        ),
        "intra_phase_heartbeat_covers_long_phases": bool(
            emit_progress
            and resolved_heartbeat is not None
            and float(resolved_heartbeat)
            < float(C2P2_LONGEST_QUIET_PHASE_REFERENCE_SECONDS)
        ),
    }


def validate_probe_stdout_liveness_config(
    receipt: Mapping[str, Any],
) -> None:
    watch_wrap_budget = float(receipt["watch_wrap_heartbeat_seconds"])
    longest_quiet = float(receipt["longest_quiet_phase_reference_seconds"])
    if watch_wrap_budget <= longest_quiet:
        raise ValueError(
            "probe stdout liveness config invalid: watch_wrap_heartbeat_seconds "
            f"{watch_wrap_budget:g} must exceed longest quiet phase "
            f"{longest_quiet:g}"
        )
    if bool(receipt["emit_progress"]):
        heartbeat = receipt.get("phase_heartbeat_seconds")
        if heartbeat is None or float(heartbeat) <= 0.0:
            raise ValueError(
                "probe stdout liveness config invalid: emit_progress requires "
                "positive phase_heartbeat_seconds"
            )


def resolve_phase_timeout_exemptions(*, contract: str) -> frozenset[str]:
    contract_name = str(contract)
    if contract_name == PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF:
        return frozenset()
    if contract_name == B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1:
        return frozenset({BOUNDED_STEPS_AGGREGATE_PHASE})
    raise ValueError(
        "phase_timeout_exemption_contract must be one of "
        f"{PHASE_TIMEOUT_EXEMPTION_CONTRACT_CHOICES}, got {contract_name!r}"
    )


def _canonical_exempt_phase_hash(exempt_phases: Sequence[str]) -> str:
    canonical = json.dumps(
        sorted(str(phase) for phase in exempt_phases),
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_phase_timeout_exemption_receipt(
    *,
    contract: str,
    phase_timeout_seconds: float | int | None,
    silent_phase_timeout_seconds: float | int | None,
    total_timeout_seconds: float | int | None,
) -> dict[str, Any]:
    contract_name = str(contract)
    exempt_phases = sorted(resolve_phase_timeout_exemptions(contract=contract_name))
    enabled = bool(exempt_phases)
    receipt: dict[str, Any] = {
        "schema": PHASE_TIMEOUT_EXEMPTION_SCHEMA_VERSION,
        "enabled": enabled,
        "contract": (
            PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF if not enabled else contract_name
        ),
        "exempt_phases": exempt_phases,
        "exempt_phase_count": len(exempt_phases),
        "exempt_phase_hash": _canonical_exempt_phase_hash(exempt_phases),
        "default_on": False,
        "requires_explicit_launch_command_entry": True,
        "phase_timeout_seconds_must_be_positive": enabled,
        "silent_phase_guard_required": enabled,
        "total_timeout_required": enabled,
        "claim_boundary": (
            "exempts aggregate bounded_steps post-exit duration only; "
            "nested phases and silent/total liveness remain fail-closed"
        ),
    }
    if enabled:
        receipt["phase_timeout_seconds"] = _timeout_or_none(phase_timeout_seconds)
        receipt["silent_phase_timeout_seconds"] = _timeout_or_none(
            silent_phase_timeout_seconds
        )
        receipt["total_timeout_seconds"] = _timeout_or_none(total_timeout_seconds)
    return receipt


def validate_b2b_phase_timeout_launch_requirements(
    *,
    b2b_sequential_capture_enabled: bool,
    phase_timeout_seconds: float | int,
    phase_timeout_exemption_contract: str,
    total_timeout_seconds: float | int,
    silent_phase_timeout_seconds: float | int | None,
    allow_gpu_launch: bool,
    max_silent_phase_seconds: float | int | None,
) -> None:
    """Validate B2b launch timeout requirements after WS-C exemption infra lands.

    Post-WS-C B2b packets must not use packet-0-style ``--phase-timeout-seconds 0``.
    Launch packets must state a positive scalar phase cap and declare which
    first-milestone nested-phase budget that cap covers.
    """
    if not b2b_sequential_capture_enabled:
        return

    resolved_silent = resolve_max_silent_phase_seconds(
        allow_gpu_launch=allow_gpu_launch,
        max_silent_phase_seconds=max_silent_phase_seconds,
    )
    effective_silent = (
        silent_phase_timeout_seconds
        if silent_phase_timeout_seconds is not None
        else resolved_silent
    )
    contract = str(phase_timeout_exemption_contract)
    scalar_timeout = float(phase_timeout_seconds)
    total_timeout = float(total_timeout_seconds)

    if scalar_timeout <= 0.0:
        raise ValueError(
            "b2b sequential capture requires a positive --phase-timeout-seconds "
            "(packet-0-style --phase-timeout-seconds 0 is invalid post-WS-C; "
            "re-draft with positive scalar cap + "
            f"--phase-timeout-exemption-contract={B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1} "
            "for aggregate bounded_steps only). Launch packets must also declare "
            "which first-milestone nested-phase budget the scalar cap covers "
            "(e.g. step, step_update, b2b_sequential_capture)."
        )
    if contract != B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1:
        raise ValueError(
            "b2b sequential capture requires "
            f"--phase-timeout-exemption-contract={B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1} "
            "when using the aggregate bounded_steps exemption; "
            f"got {contract!r}. Post-WS-C, packet-0-style --phase-timeout-seconds 0 "
            "without the named contract is invalid."
        )
    if _timeout_or_none(effective_silent) is None:
        raise ValueError(
            "b2b sequential capture requires an effective silent-phase guard "
            "(positive --max-silent-phase-seconds or --allow-gpu-launch default)."
        )
    if _timeout_or_none(total_timeout) is None:
        raise ValueError(
            "b2b sequential capture requires a positive --total-timeout-seconds."
        )


def register_probe_faulthandler(
    *,
    run_log_path: Path | None = None,
    enable_fn: Callable[..., Any] | None = None,
    register_fn: Callable[..., Any] | None = None,
    is_enabled_fn: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    enable = enable_fn or faulthandler.enable
    register = register_fn or faulthandler.register
    is_enabled = is_enabled_fn or faulthandler.is_enabled
    report: dict[str, Any] = {
        "schema": C2P2_FAULTHANDLER_SCHEMA_VERSION,
        "enabled_before": bool(is_enabled()),
        "signals": {},
        "traceback_target": "stderr",
    }
    traceback_file: Any = sys.stderr
    if run_log_path is not None:
        traceback_file = Path(run_log_path).open("a", encoding="utf-8", buffering=1)
        report["traceback_target"] = str(run_log_path)
    if run_log_path is not None or not report["enabled_before"]:
        try:
            enable(file=traceback_file, all_threads=True)
        except Exception as exc:
            report["enable_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise RuntimeError("failed to enable faulthandler for probe") from exc
    report["enabled_after"] = bool(is_enabled())

    sigquit = getattr(signal, "SIGQUIT", None)
    if sigquit is None:
        report["signals"]["SIGQUIT"] = {"status": "unavailable"}
    else:
        def _probe_sigquit_handler(signum: int, frame: Any) -> None:
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:
                pass
            flush_probe_terminal_artifacts(
                exit_code=128 + int(signum),
                flush_reason="sigquit",
            )
            raise SystemExit(128 + int(signum))

        try:
            signal.signal(sigquit, _probe_sigquit_handler)
            report["signals"]["SIGQUIT"] = {
                "status": "registered",
                "signal": int(sigquit),
                "handler": "probe_sigquit_flush_then_exit",
            }
        except Exception as exc:
            report["signals"]["SIGQUIT"] = {
                "status": "failed",
                "signal": int(sigquit),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    sigabrt = getattr(signal, "SIGABRT", None)
    report["signals"]["SIGABRT"] = (
        {"status": "unavailable"}
        if sigabrt is None
        else {
            "status": "handled_by_faulthandler_enable",
            "signal": int(sigabrt),
        }
    )
    return report


def profile_host_rss_enabled() -> bool:
    return os.environ.get(PROFILE_HOST_RSS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def profile_host_rss_live_resident_enabled() -> bool:
    return os.environ.get(PROFILE_HOST_RSS_LIVE_RESIDENT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def profile_torch_cpu_census_enabled() -> bool:
    return profile_host_rss_enabled() and os.environ.get(
        PROFILE_TORCH_CPU_CENSUS_ENV, ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def profile_tracemalloc_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        profile_tracemalloc_enabled as _enabled,
    )

    return _enabled()


def profile_debugmallocstats_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        profile_debugmallocstats_enabled as _enabled,
    )

    return _enabled()


def profile_obmalloc_site_brackets_enabled() -> bool:
    if not profile_debugmallocstats_enabled():
        return False
    return os.environ.get(PROFILE_OBMALLOC_SITE_BRACKETS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def profile_s1d7_tracemalloc_site_enabled() -> bool:
    if not profile_host_rss_enabled():
        return False
    if not profile_tracemalloc_enabled():
        return False
    if profile_debugmallocstats_enabled():
        return False
    return True


def profile_s1d7_tracemalloc_full_trace_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        profile_s1d7_tracemalloc_full_trace_enabled as _enabled,
    )

    return _enabled()


def profile_s1d7_band_counter_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        profile_s1d7_band_counter_enabled as _enabled,
    )

    return _enabled()


def profile_s1d7_band_counter_only_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        profile_s1d7_band_counter_only_enabled as _enabled,
    )

    return _enabled()


def profile_obmalloc_expanded_enabled() -> bool:
    if not profile_obmalloc_site_brackets_enabled():
        return False
    return os.environ.get(PROFILE_OBMALLOC_EXPANDED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def assert_profile_tracemalloc_debugmallocstats_mutual_exclusion() -> None:
    if profile_tracemalloc_enabled() and profile_debugmallocstats_enabled():
        payload = {
            "event": "profile_env_mutual_exclusion_abort",
            "tracemalloc_env": PROFILE_TRACEMALLOC_ENV,
            "debugmallocstats_env": PROFILE_DEBUGMALLOCSTATS_ENV,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        os.abort()


def profile_allocator_native_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        profile_allocator_native_enabled as _enabled,
    )

    return _enabled()


def profile_allocator_host_cache_diag_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        profile_allocator_host_cache_diag_enabled as _enabled,
    )

    return _enabled()


def profile_alloc_hook_enabled() -> bool:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        profile_alloc_hook_enabled as _enabled,
    )

    return _enabled()


def gc_collect_and_malloc_trim() -> None:
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _proc_kib_field(path: Path, key_prefix: str) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(key_prefix):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except Exception:
        return None
    return None


def _proc_self_resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"pid": int(os.getpid())}
    try:
        times = os.times()
        snapshot["process_cpu_user_seconds"] = float(times.user)
        snapshot["process_cpu_system_seconds"] = float(times.system)
    except Exception as exc:
        snapshot["process_cpu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        status_path = Path("/proc/self/status")
        snapshot["rss_kib"] = _proc_kib_field(status_path, "VmRSS:")
        snapshot["hwm_kib"] = _proc_kib_field(status_path, "VmHWM:")
    except Exception as exc:
        snapshot["rss_error"] = f"{type(exc).__name__}: {exc}"
    try:
        rollup_path = Path("/proc/self/smaps_rollup")
        snapshot["pss_kib"] = _proc_kib_field(rollup_path, "Pss:")
        private_clean = _proc_kib_field(rollup_path, "Private_Clean:")
        private_dirty = _proc_kib_field(rollup_path, "Private_Dirty:")
        if private_clean is not None and private_dirty is not None:
            snapshot["uss_kib"] = int(private_clean) + int(private_dirty)
    except Exception as exc:
        snapshot["memory_rollup_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _append_host_rss_profile_mark(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True))
        handle.write("\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)




def validate_recompute_window_log_flag_mutual_exclusion(
    *,
    d_recompute_window_instrumentation_enabled: bool,
    event_coded_recompute_window_log_enabled: bool,
) -> None:
    if bool(d_recompute_window_instrumentation_enabled) and bool(
        event_coded_recompute_window_log_enabled
    ):
        raise ValueError(
            "d_recompute_window_instrumentation_enabled and "
            "event_coded_recompute_window_log_enabled are mutually exclusive"
        )


def _validate_event_coded_sparse_vote_authority_config(
    *,
    event_coded_sparse_vote_authority: bool,
    persistent_accumulator_event_coded_live: bool,
    two_tier_carry_w6_enabled: bool,
    b2b_sequential_capture_enabled: bool,
    votes_emit_enabled: bool,
    carrier_growth_enabled: bool,
    d_recompute_window_instrumentation_enabled: bool,
    d_recompute_calibration_warmup_out: Path | None,
) -> None:
    if not bool(event_coded_sparse_vote_authority):
        return
    if not bool(persistent_accumulator_event_coded_live):
        raise ValueError(
            "--event-coded-sparse-vote-authority requires "
            "--persistent-accumulator-event-coded-live"
        )
    incompatible: list[str] = []
    if bool(two_tier_carry_w6_enabled):
        incompatible.append("--two-tier-carry-w6")
    if bool(b2b_sequential_capture_enabled):
        incompatible.append("--b2b-sequential-capture")
    if bool(votes_emit_enabled):
        incompatible.append("--votes-emit-enabled")
    if bool(carrier_growth_enabled):
        incompatible.append("--carrier-growth-enabled")
    if bool(d_recompute_window_instrumentation_enabled):
        incompatible.append("--d-recompute-window-instrumentation")
    if d_recompute_calibration_warmup_out is not None:
        incompatible.append("--d-recompute-calibration-warmup-out")
    if incompatible:
        raise ValueError(
            "event_coded_sparse_vote_authority incompatible with: "
            + ", ".join(incompatible)
        )


_PROBE_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV,
    PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV,
    PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV,
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV,
    PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV,
    PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV,
)


def _snapshot_probe_runtime_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _PROBE_RUNTIME_ENV_KEYS}


def _restore_probe_runtime_env(snapshot: Mapping[str, str | None]) -> None:
    for key in _PROBE_RUNTIME_ENV_KEYS:
        value = snapshot.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class PhaseMilestoneEmitter:
    def __init__(
        self,
        root: Path,
        *,
        enabled: bool,
        device: torch.device,
    ) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.device = device
        self._counters: dict[str, int] = {}

    def record_phase_complete(
        self,
        phase_id: str,
        *,
        optimizer_step_index: int | None,
        elapsed_since_phase_enter_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        if str(phase_id) not in MILESTONE_BUDGETED_PHASE_IDS:
            return
        counter = int(self._counters.get(str(phase_id), 0)) + 1
        self._counters[str(phase_id)] = counter
        out_path = self.root / "liveness_milestones" / f"{phase_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": PHASE_MILESTONE_COUNTER_SCHEMA,
            "phase_id": str(phase_id),
            "optimizer_step_index": optimizer_step_index,
            "milestone_counter": counter,
            "milestone_kind": "phase_complete",
            "device": str(self.device),
            "elapsed_since_phase_enter_seconds": round(
                float(elapsed_since_phase_enter_seconds),
                6,
            ),
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def record_sparse_cap_sub_phase(
        self,
        sub_phase_id: str,
        *,
        optimizer_step_index: int | None,
        milestone_kind: str,
        elapsed_since_phase_enter_seconds: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        sub_phase = str(sub_phase_id)
        if sub_phase not in MILESTONE_SPARSE_CAP_SUB_PHASE_IDS:
            return
        counter_key = f"sparse_cap_apply_{sub_phase}"
        counter = int(self._counters.get(counter_key, 0)) + 1
        self._counters[counter_key] = counter
        jsonl_name = SPARSE_CAP_SUB_PHASE_JSONL_NAMES[sub_phase]
        out_path = self.root / "liveness_milestones" / jsonl_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": PHASE_MILESTONE_COUNTER_SCHEMA,
            "phase_id": "sparse_cap_apply",
            "parent_phase_id": "sparse_cap_apply",
            "sub_phase_id": sub_phase,
            "optimizer_step_index": optimizer_step_index,
            "milestone_counter": counter,
            "milestone_kind": str(milestone_kind),
            "device": str(self.device),
            "elapsed_since_phase_enter_seconds": round(
                float(elapsed_since_phase_enter_seconds),
                6,
            ),
        }
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def ensure_sparse_cap_subphase_contract(
        self,
        *,
        optimizer_step_index: int,
        required_sub_phase_ids: tuple[str, ...] = (
            "cap_selection_cpu_copy",
            "post_cap_apply_sync",
            "boundary_normalize",
        ),
    ) -> dict[str, Any]:
        """Post-step validator: record which sparse_cap subphase jsonl are present."""
        if not self.enabled:
            return {"pass": True, "skipped": True}
        present: dict[str, bool] = {}
        for sub_phase in required_sub_phase_ids:
            jsonl_name = SPARSE_CAP_SUB_PHASE_JSONL_NAMES[str(sub_phase)]
            path = self.root / "liveness_milestones" / jsonl_name
            rows = []
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            step_rows = [
                row
                for row in rows
                if int(row.get("optimizer_step_index", -1)) == int(optimizer_step_index)
            ]
            present[str(sub_phase)] = bool(step_rows)
        contract_path = self.root / "liveness_milestones" / "sparse_cap_subphase_contract.jsonl"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "hrm_text_158_sparse_cap_subphase_contract/v1",
            "optimizer_step_index": int(optimizer_step_index),
            "present": present,
            "pass": all(present.values()),
        }
        with contract_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record


def run_submilestone_emit_contract_witness() -> dict[str, Any]:
    """Static preflight: sub-phase milestone hooks present and O(1)-safe."""
    probe_src = Path(__file__).read_text(encoding="utf-8")
    learner_path = Path("calm/hrm_text_158/native_full_stack/bounded_delta_learner.py")
    learner_src = learner_path.read_text(encoding="utf-8")
    failures: list[str] = []
    for sub_phase_id in sorted(MILESTONE_SPARSE_CAP_SUB_PHASE_IDS):
        jsonl_name = SPARSE_CAP_SUB_PHASE_JSONL_NAMES[sub_phase_id]
        if jsonl_name not in probe_src:
            failures.append(f"missing_jsonl_path_template:{sub_phase_id}")
        if sub_phase_id not in learner_src:
            failures.append(f"missing_learner_sub_phase_marker:{sub_phase_id}")
    required_probe_markers = [
        "def record_sparse_cap_sub_phase",
        "parent_phase_id",
        "sparse_cap_submilestone_emit",
    ]
    for marker in required_probe_markers:
        if marker not in probe_src:
            failures.append(f"missing_probe_marker:{marker}")
    required_learner_markers = [
        "def _emit_sparse_cap_submilestone",
        "sparse_cap_submilestone_cap_selection_recorded",
        "sparse_cap_submilestone_post_cap_sync_recorded",
        "sparse_cap_submilestone_boundary_normalize_recorded",
    ]
    for marker in required_learner_markers:
        if marker not in learner_src:
            failures.append(f"missing_learner_marker:{marker}")
    if "from scripts.hrm_text_158_bounded_delta_acquisition_probe" in learner_src:
        failures.append("learner_must_not_import_probe")
    hook_regions = [
        probe_src.split("def record_sparse_cap_sub_phase", 1)[-1].split("def run_submilestone_emit_contract_witness", 1)[0],
        learner_src.split("def _emit_sparse_cap_submilestone", 1)[-1].split("def _apply_bounded_delta_vote_step_event_coded_live", 1)[0],
    ]
    for region_name, region in zip(("probe_hook", "learner_hook"), hook_regions, strict=True):
        if ".tolist(" in region:
            failures.append(f"forbidden_tolist_in_{region_name}")
        if "torch.zeros(" in region and "numel" in region:
            failures.append(f"forbidden_zeros_numel_in_{region_name}")
    return {
        "schema": "hrm_text_158_slice5_submilestone_emit_contract_witness/v1",
        "pass": not failures,
        "failures": failures,
        "sub_phase_ids": sorted(MILESTONE_SPARSE_CAP_SUB_PHASE_IDS),
    }


@dataclass
class ProbeTerminalFlushContext:
    scratch_root: Path
    parent_checkpoint_path: Path
    parent_hash_before: str
    _exit_code_written: bool = False
    _posthash_written: bool = False


_PROBE_TERMINAL_FLUSH_CTX: ProbeTerminalFlushContext | None = None


def set_probe_terminal_flush_context(ctx: ProbeTerminalFlushContext) -> None:
    global _PROBE_TERMINAL_FLUSH_CTX
    _PROBE_TERMINAL_FLUSH_CTX = ctx


def clear_probe_terminal_flush_context() -> None:
    global _PROBE_TERMINAL_FLUSH_CTX
    _PROBE_TERMINAL_FLUSH_CTX = None


def _probe_run_root_from_scratch(scratch_root: Path) -> Path:
    return scratch_root.parent


def _parent_checkpoint_posthash_path(scratch_root: Path) -> Path:
    return (
        _probe_run_root_from_scratch(scratch_root)
        / "prelaunch"
        / PARENT_CHECKPOINT_POSTHASH_ARTIFACT_NAME
    )


def flush_probe_terminal_artifacts(
    *,
    exit_code: int,
    flush_reason: str,
) -> None:
    ctx = _PROBE_TERMINAL_FLUSH_CTX
    if ctx is None:
        return
    scratch_root = Path(ctx.scratch_root)
    run_root = _probe_run_root_from_scratch(scratch_root)
    exit_path = run_root / PROBE_EXIT_CODE_ARTIFACT_NAME
    if not ctx._exit_code_written:
        exit_path.parent.mkdir(parents=True, exist_ok=True)
        exit_path.write_text(f"{int(exit_code)}\n", encoding="utf-8")
        ctx._exit_code_written = True
    if not ctx._posthash_written:
        parent_hash_after = file_sha256(ctx.parent_checkpoint_path)
        payload = {
            "schema": C2P2_PARENT_CHECKPOINT_POSTHASH_SCHEMA_VERSION,
            "parent_checkpoint_path": str(ctx.parent_checkpoint_path),
            "parent_sha256": str(parent_hash_after),
            "parent_hash_before": str(ctx.parent_hash_before),
            "parent_hash_after": str(parent_hash_after),
            "parent_hash_unchanged": str(ctx.parent_hash_before)
            == str(parent_hash_after),
            "flush_reason": str(flush_reason),
        }
        _write_json_atomic(_parent_checkpoint_posthash_path(scratch_root), payload)
        ctx._posthash_written = True


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
        silent_phase_timeout_seconds: float | int | None = None,
        phase_heartbeat_interval_seconds: float | int | None = None,
        last_active_phase_path: Path | None = None,
        arm_faulthandler_timer: bool = True,
        phase_timeout_exemptions: frozenset[str] | set[str] | None = None,
        phase_timeout_exemption_contract: str = PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF,
        milestone_emitter: PhaseMilestoneEmitter | None = None,
        host_rss_profile_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.phase_timeout_seconds = _timeout_or_none(phase_timeout_seconds)
        self.total_timeout_seconds = _timeout_or_none(total_timeout_seconds)
        self.silent_phase_timeout_seconds = _timeout_or_none(silent_phase_timeout_seconds)
        self.phase_heartbeat_interval_seconds = _timeout_or_none(
            phase_heartbeat_interval_seconds
        )
        self.last_active_phase_path = last_active_phase_path
        self.arm_faulthandler_timer = bool(arm_faulthandler_timer)
        self.phase_timeout_exemption_contract = str(phase_timeout_exemption_contract)
        self.phase_timeout_exemptions = frozenset(
            str(phase) for phase in (phase_timeout_exemptions or ())
        )
        self.milestone_emitter = milestone_emitter
        self.host_rss_profile_path = (
            Path(host_rss_profile_path) if host_rss_profile_path is not None else None
        )
        self.clock = clock
        self.started_at = float(self.clock())
        self.events: list[dict[str, Any]] = []
        self._phase_stack: list[dict[str, Any]] = []
        self._last_active_phase_payload: dict[str, Any] | None = None
        self._live_heartbeat_threads: list[threading.Thread] = []
        self._ring_sampler = None
        self._ring_jsonl_path: Path | None = None
        if last_active_phase_path is not None:
            from calm.hrm_text_158.native_full_stack.phase_stack_ring_sampler import (
                PhaseStackRingSampler,
                phase_stack_ring_sampler_enabled,
            )

            if phase_stack_ring_sampler_enabled():
                self._ring_sampler = PhaseStackRingSampler()
                self._ring_jsonl_path = (
                    Path(last_active_phase_path).parent / "liveness_stack_ring.jsonl"
                )

    @property
    def active(self) -> bool:
        return bool(
            self.enabled
            or self.phase_timeout_seconds is not None
            or self.total_timeout_seconds is not None
            or self.silent_phase_timeout_seconds is not None
        )

    def _elapsed(self) -> float:
        return max(0.0, float(self.clock() - self.started_at))

    def _phase_timeout_for(self, phase: str) -> float | None:
        if str(phase) in self.phase_timeout_exemptions:
            return None
        return self.phase_timeout_seconds

    def _emit_host_rss_profile_mark(
        self,
        phase: str,
        event: str,
        fields: Mapping[str, Any],
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if str(phase) not in PROFILE_HOST_RSS_PHASES:
            return
        resource_snapshot = _proc_self_resource_snapshot()
        mark = {
            "schema": PROFILE_HOST_RSS_SCHEMA,
            "phase": str(phase),
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
        }
        for key in ("step",):
            if key in fields:
                mark[key] = fields[key]
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_host_rss_subphase_mark(
        self,
        *,
        parent_phase: str,
        sub_phase_id: str,
        event: str,
        fields: Mapping[str, Any],
        allocation_dims: Mapping[str, Any] | None = None,
        measurement_perturbed: bool = False,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if str(sub_phase_id) not in PROFILE_HOST_RSS_SUBPHASE_IDS:
            return
        resource_snapshot = _proc_self_resource_snapshot()
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_SUBPHASE_SCHEMA,
            "phase": str(parent_phase),
            "parent_phase": str(parent_phase),
            "sub_phase": str(sub_phase_id),
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": bool(measurement_perturbed),
        }
        for key in ("step", "optimizer_step_index"):
            if key in fields:
                mark[key] = fields[key]
        if allocation_dims:
            mark["allocation_dims"] = dict(allocation_dims)
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_torch_cpu_census_subphase_mark(
        self,
        *,
        parent_phase: str,
        sub_phase_id: str,
        event: str,
        fields: Mapping[str, Any],
        allocation_site_id: str,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_torch_cpu_census_enabled():
            return
        if str(sub_phase_id) not in PROFILE_HOST_RSS_SUBPHASE_IDS:
            return
        from calm.hrm_text_158.native_full_stack.host_torch_census import (
            torch_cpu_tensor_census,
        )

        resource_snapshot = _proc_self_resource_snapshot()
        census = torch_cpu_tensor_census(top_k=10)
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "phase": str(parent_phase),
            "parent_phase": str(parent_phase),
            "sub_phase": str(sub_phase_id),
            "event": str(event),
            "allocation_site_id": str(allocation_site_id),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": True,
            "torch_census": census,
        }
        for key in ("step", "optimizer_step_index", "state_index", "state_bucket"):
            if key in fields:
                mark[key] = fields[key]
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_allocator_native_subphase_mark(
        self,
        *,
        parent_phase: str,
        sub_phase_id: str,
        event: str,
        fields: Mapping[str, Any],
        allocation_site_id: str,
        allocation_dims: Mapping[str, Any] | None = None,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_allocator_native_enabled():
            return
        if str(sub_phase_id) not in PROFILE_HOST_RSS_SUBPHASE_IDS:
            return
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            snapshot_allocator_probe,
        )

        resource_snapshot = _proc_self_resource_snapshot()
        probe_snapshot = snapshot_allocator_probe()
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_ALLOCATOR_SCHEMA,
            "phase": str(parent_phase),
            "parent_phase": str(parent_phase),
            "sub_phase": str(sub_phase_id),
            "event": str(event),
            "allocation_site_id": str(allocation_site_id),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": True,
            "allocator_probe": probe_snapshot,
        }
        for key in ("step", "optimizer_step_index", "state_index", "state_bucket"):
            if key in fields:
                mark[key] = fields[key]
        if allocation_dims is not None:
            mark["allocation_dims"] = dict(allocation_dims)
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_allocator_site_mark(
        self,
        *,
        site_id: str,
        event_suffix: str,
        origin_file: str,
        origin_line: int,
        fields: Mapping[str, Any],
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_allocator_native_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            snapshot_allocator_probe,
        )

        resource_snapshot = _proc_self_resource_snapshot()
        probe_snapshot = snapshot_allocator_probe()
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": f"allocator_site_{site_id}_{event_suffix}",
            "site_id": str(site_id),
            "origin_file": str(origin_file),
            "origin_line": int(origin_line),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": True,
            "allocator_probe": probe_snapshot,
        }
        for key in ("step", "optimizer_step_index", "state_index"):
            if key in fields:
                mark[key] = fields[key]
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_obmalloc_site_bracket_mark(
        self,
        *,
        site_id: str,
        event_suffix: str,
        origin_file: str,
        origin_line: int,
        fields: Mapping[str, Any],
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_obmalloc_site_brackets_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            read_debugmallocstats,
        )

        resource_snapshot = _proc_self_resource_snapshot()
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": f"obmalloc_site_{site_id}_{event_suffix}",
            "site_id": str(site_id),
            "origin_file": str(origin_file),
            "origin_line": int(origin_line),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": True,
            "debugmallocstats": read_debugmallocstats(),
        }
        for key in ("step", "optimizer_step_index", "state_index"):
            if key in fields:
                mark[key] = fields[key]
        if "sampled_states" in fields:
            mark["sampled_states"] = list(fields["sampled_states"])
        stats = dict(mark.get("debugmallocstats") or {})
        if profile_obmalloc_expanded_enabled():
            mark["obmalloc_expanded"] = True
            if stats.get("arena_bytes") is not None:
                mark["arena_bytes_forcing"] = int(stats["arena_bytes"])
            if stats.get("bytes_in_allocated_blocks") is not None:
                mark["allocated_blocks_holding"] = int(stats["bytes_in_allocated_blocks"])
        if str(site_id) == "C4.S1d.7" and profile_tracemalloc_enabled():
            from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
                begin_s1d7_tracemalloc_bracket,
                end_s1d7_tracemalloc_bracket,
            )
            from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
                take_tracemalloc_snapshot_dict,
            )

            if str(event_suffix) == "pre":
                begin_s1d7_tracemalloc_bracket(depth=50)
                mark["s1d7_tracemalloc"] = take_tracemalloc_snapshot_dict()
            else:
                try:
                    mark["s1d7_tracemalloc"] = take_tracemalloc_snapshot_dict()
                finally:
                    end_s1d7_tracemalloc_bracket()
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_s1d7_band_counter_site_mark(
        self,
        *,
        origin_file: str,
        origin_line: int,
        counters: Mapping[str, Any],
        fields: Mapping[str, Any],
        measurement_contract: str | None = None,
        event_encoded_bytes_delta_source: str | None = None,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_s1d7_band_counter_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
            profile_s1d7_tracemalloc_site_enabled,
        )
        from calm.hrm_text_158.native_full_stack.s1d7_band_counter import (
            S1D7_BAND_COUNTER_EVENT,
            S1D7_BAND_COUNTER_SITE_SCHEMA,
            S1D7_SITE_ID,
        )

        resource_snapshot = _proc_self_resource_snapshot()
        mark: dict[str, Any] = {
            "schema": S1D7_BAND_COUNTER_SITE_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": str(S1D7_BAND_COUNTER_EVENT),
            "site_id": str(S1D7_SITE_ID),
            "origin_file": str(origin_file),
            "origin_line": int(origin_line),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "tracemalloc_only": profile_s1d7_tracemalloc_site_enabled(),
            "tracemalloc_diagnostic": False,
            "band_counter_only": True,
            "s1d7_band_counters": dict(counters),
        }
        if measurement_contract is not None:
            mark["measurement_contract"] = str(measurement_contract)
        if event_encoded_bytes_delta_source is not None:
            mark["event_encoded_bytes_delta_source"] = str(event_encoded_bytes_delta_source)
        for key in ("step", "optimizer_step_index", "state_index"):
            if key in fields:
                mark[key] = fields[key]
        if "sampled_states" in fields:
            mark["sampled_states"] = list(fields["sampled_states"])
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_s1d7_tracemalloc_site_mark(
        self,
        *,
        event_suffix: str,
        origin_file: str,
        origin_line: int,
        fields: Mapping[str, Any],
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_s1d7_tracemalloc_full_trace_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
            begin_s1d7_tracemalloc_bracket,
            end_s1d7_tracemalloc_bracket,
        )
        from calm.hrm_text_158.native_full_stack.s1d7_tracemalloc_feasibility import (
            S1D7_SITE_ID,
            S1D7_TRACEMALLOC_POST_EVENT,
            S1D7_TRACEMALLOC_PRE_EVENT,
            take_tracemalloc_snapshot_dict,
        )

        event = (
            S1D7_TRACEMALLOC_PRE_EVENT
            if str(event_suffix) == "pre"
            else S1D7_TRACEMALLOC_POST_EVENT
        )
        resource_snapshot = _proc_self_resource_snapshot()
        if str(event_suffix) == "pre":
            begin_s1d7_tracemalloc_bracket(depth=50)
            snapshot = take_tracemalloc_snapshot_dict()
        else:
            try:
                snapshot = take_tracemalloc_snapshot_dict()
            finally:
                end_s1d7_tracemalloc_bracket()
        mark: dict[str, Any] = {
            "schema": PROFILE_S1D7_TRACEMALLOC_SITE_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": str(event),
            "site_id": str(S1D7_SITE_ID),
            "origin_file": str(origin_file),
            "origin_line": int(origin_line),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "tracemalloc_only": True,
            "tracemalloc_diagnostic": True,
            "s1d7_tracemalloc": snapshot,
        }
        for key in ("step", "optimizer_step_index", "state_index"):
            if key in fields:
                mark[key] = fields[key]
        if "sampled_states" in fields:
            mark["sampled_states"] = list(fields["sampled_states"])
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_alloc_hook_mark(
        self,
        *,
        event: str,
        fields: Mapping[str, Any],
        allocation_dims: Mapping[str, Any] | None = None,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_alloc_hook_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
            arm_hook,
            disarm_hook,
            flush_live_ranges,
            flush_stats,
            hook_vma_ranges,
            prefault_hook,
            reset_aggregation_window,
        )
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            snapshot_allocator_probe,
        )

        stats_path = self.host_rss_profile_path.parent / "alloc_hook_stats.json"
        live_ranges_path = self.host_rss_profile_path.parent / "alloc_hook_live_ranges.json"
        stats: dict[str, Any] = {}
        live_ranges: list[dict[str, Any]] = []
        if str(event) == "alloc_hook_C4_enter":
            prefault_hook()
            arm_hook()
            reset_aggregation_window()
        elif str(event) == "alloc_hook_C4_exit":
            live_flush = flush_live_ranges(live_ranges_path)
            live_ranges = list(live_flush.get("live_ranges") or [])
            stats = flush_stats(stats_path)
            disarm_hook()
        elif str(event).startswith("alloc_hook_"):
            stats = flush_stats(stats_path)
        exclude = hook_vma_ranges(stats) if stats else []
        resource_snapshot = _proc_self_resource_snapshot()
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": resource_snapshot,
            "measurement_perturbed": True,
            "alloc_hook_stats": stats,
            "allocator_probe": snapshot_allocator_probe(exclude_hook_vmas=exclude),
        }
        if live_ranges:
            mark["live_ranges"] = live_ranges
            mark["live_ranges_path"] = str(live_ranges_path)
        for key in ("step", "optimizer_step_index", "state_index", "state_bucket", "status"):
            if key in fields:
                mark[key] = fields[key]
        if allocation_dims is not None:
            mark["allocation_dims"] = dict(allocation_dims)
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_triangulation_boundary_mark(
        self,
        event: str,
        *,
        fields: Mapping[str, Any],
        measurement_perturbed: bool = False,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_tracemalloc_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            read_debugmallocstats,
        )
        from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
            ensure_tracemalloc_started,
            snapshot_tracemalloc,
        )

        ensure_tracemalloc_started(depth=25)
        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_TRIANGULATION_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": _proc_self_resource_snapshot(),
            "measurement_perturbed": bool(measurement_perturbed),
            "tracemalloc": snapshot_tracemalloc(),
            "debugmallocstats": read_debugmallocstats(),
        }
        for key in (
            "step",
            "optimizer_step_index",
            "state_index",
            "state_bucket",
            "allocation_site_id",
            "sub_phase_id",
        ):
            if key in fields:
                mark[key] = fields[key]
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def _emit_obmalloc_boundary_mark(
        self,
        event: str,
        *,
        fields: Mapping[str, Any],
        measurement_perturbed: bool = False,
    ) -> None:
        if self.host_rss_profile_path is None:
            return
        if not profile_debugmallocstats_enabled():
            return
        from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
            read_debugmallocstats,
        )

        mark: dict[str, Any] = {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "phase": "sparse_cap_apply",
            "parent_phase": "sparse_cap_apply",
            "event": str(event),
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "resource_snapshot": _proc_self_resource_snapshot(),
            "measurement_perturbed": bool(measurement_perturbed),
            "debugmallocstats": read_debugmallocstats(),
        }
        for key in (
            "step",
            "optimizer_step_index",
            "state_index",
            "state_bucket",
            "allocation_site_id",
            "sub_phase_id",
        ):
            if key in fields:
                mark[key] = fields[key]
        _append_host_rss_profile_mark(self.host_rss_profile_path, mark)

    def make_host_rss_subphase_emitter(
        self,
        *,
        step: int,
    ) -> Callable[..., None] | None:
        if self.host_rss_profile_path is None:
            return None

        def emit(
            event: str,
            *,
            sub_phase_id: str,
            optimizer_step_index: int,
            allocation_dims: Mapping[str, Any] | None = None,
            measurement_perturbed: bool = False,
            allocation_site_id: str | None = None,
            state_index: int | None = None,
            state_bucket: int | None = None,
        ) -> None:
            if str(event).startswith("census_"):
                if allocation_site_id is None:
                    return
                self._emit_torch_cpu_census_subphase_mark(
                    parent_phase="sparse_cap_apply",
                    sub_phase_id=str(sub_phase_id),
                    event=str(event),
                    fields={
                        "step": int(step),
                        "optimizer_step_index": int(optimizer_step_index),
                        **(
                            {"state_index": int(state_index)}
                            if state_index is not None
                            else {}
                        ),
                        **(
                            {"state_bucket": int(state_bucket)}
                            if state_bucket is not None
                            else {}
                        ),
                    },
                    allocation_site_id=str(allocation_site_id),
                )
                return
            if str(event).startswith("allocator_"):
                if allocation_site_id is None:
                    return
                self._emit_allocator_native_subphase_mark(
                    parent_phase="sparse_cap_apply",
                    sub_phase_id=str(sub_phase_id),
                    event=str(event),
                    fields={
                        "step": int(step),
                        "optimizer_step_index": int(optimizer_step_index),
                        **(
                            {"state_index": int(state_index)}
                            if state_index is not None
                            else {}
                        ),
                        **(
                            {"state_bucket": int(state_bucket)}
                            if state_bucket is not None
                            else {}
                        ),
                    },
                    allocation_site_id=str(allocation_site_id),
                    allocation_dims=allocation_dims,
                )
                if profile_alloc_hook_enabled():
                    hook_event = str(event).replace("allocator_", "alloc_hook_", 1)
                    self._emit_alloc_hook_mark(
                        event=hook_event,
                        fields={
                            "step": int(step),
                            "optimizer_step_index": int(optimizer_step_index),
                            **(
                                {"state_index": int(state_index)}
                                if state_index is not None
                                else {}
                            ),
                            **(
                                {"state_bucket": int(state_bucket)}
                                if state_bucket is not None
                                else {}
                            ),
                        },
                        allocation_dims=allocation_dims,
                    )
                return
            self._emit_host_rss_subphase_mark(
                parent_phase="sparse_cap_apply",
                sub_phase_id=str(sub_phase_id),
                event=str(event),
                fields={
                    "step": int(step),
                    "optimizer_step_index": int(optimizer_step_index),
                },
                allocation_dims=allocation_dims,
                measurement_perturbed=bool(measurement_perturbed),
            )

        def site_emit(
            site_id: str,
            event_suffix: str,
            *,
            origin_file: str,
            origin_line: int,
            optimizer_step_index: int,
            state_index: int,
        ) -> None:
            site_fields = {
                "step": int(step),
                "optimizer_step_index": int(optimizer_step_index),
                "state_index": int(state_index),
            }
            sampled_states = getattr(emit, "_obmalloc_expanded_sampled_states", None)
            if sampled_states is None:
                sampled_states = getattr(self, "_obmalloc_expanded_sampled_states", None)
            if sampled_states is not None:
                site_fields["sampled_states"] = list(sampled_states)
            self._emit_allocator_site_mark(
                site_id=str(site_id),
                event_suffix=str(event_suffix),
                origin_file=str(origin_file),
                origin_line=int(origin_line),
                fields=site_fields,
            )
            self._emit_obmalloc_site_bracket_mark(
                site_id=str(site_id),
                event_suffix=str(event_suffix),
                origin_file=str(origin_file),
                origin_line=int(origin_line),
                fields=site_fields,
            )
            if str(site_id) == "C4.S1d.7":
                if profile_s1d7_tracemalloc_full_trace_enabled():
                    self._emit_s1d7_tracemalloc_site_mark(
                        event_suffix=str(event_suffix),
                        origin_file=str(origin_file),
                        origin_line=int(origin_line),
                        fields=site_fields,
                    )

        def band_counter_emit(
            *,
            origin_file: str,
            origin_line: int,
            counters: Mapping[str, Any],
            optimizer_step_index: int,
            state_index: int,
            measurement_contract: str | None = None,
            event_encoded_bytes_delta_source: str | None = None,
        ) -> None:
            site_fields = {
                "step": int(step),
                "optimizer_step_index": int(optimizer_step_index),
                "state_index": int(state_index),
            }
            sampled_states = getattr(emit, "_obmalloc_expanded_sampled_states", None)
            if sampled_states is None:
                sampled_states = getattr(self, "_obmalloc_expanded_sampled_states", None)
            if sampled_states is not None:
                site_fields["sampled_states"] = list(sampled_states)
            self._emit_s1d7_band_counter_site_mark(
                origin_file=str(origin_file),
                origin_line=int(origin_line),
                counters=dict(counters),
                fields=site_fields,
                measurement_contract=measurement_contract,
                event_encoded_bytes_delta_source=event_encoded_bytes_delta_source,
            )

        emit.site_emit = site_emit  # type: ignore[attr-defined]
        emit.band_counter_emit = band_counter_emit  # type: ignore[attr-defined]

        def triangulation_emit(
            event: str,
            *,
            sub_phase_id: str,
            optimizer_step_index: int,
            allocation_site_id: str,
            state_index: int | None = None,
            state_bucket: int | None = None,
        ) -> None:
            self._emit_triangulation_boundary_mark(
                str(event),
                fields={
                    "step": int(step),
                    "optimizer_step_index": int(optimizer_step_index),
                    "sub_phase_id": str(sub_phase_id),
                    "allocation_site_id": str(allocation_site_id),
                    **(
                        {"state_index": int(state_index)}
                        if state_index is not None
                        else {}
                    ),
                    **(
                        {"state_bucket": int(state_bucket)}
                        if state_bucket is not None
                        else {}
                    ),
                },
                measurement_perturbed=True,
            )

        emit.triangulation_emit = triangulation_emit  # type: ignore[attr-defined]

        def obmalloc_emit(
            event: str,
            *,
            sub_phase_id: str,
            optimizer_step_index: int,
            allocation_site_id: str,
            state_index: int | None = None,
            state_bucket: int | None = None,
        ) -> None:
            self._emit_obmalloc_boundary_mark(
                str(event),
                fields={
                    "step": int(step),
                    "optimizer_step_index": int(optimizer_step_index),
                    "sub_phase_id": str(sub_phase_id),
                    "allocation_site_id": str(allocation_site_id),
                    **(
                        {"state_index": int(state_index)}
                        if state_index is not None
                        else {}
                    ),
                    **(
                        {"state_bucket": int(state_bucket)}
                        if state_bucket is not None
                        else {}
                    ),
                },
                measurement_perturbed=True,
            )

        emit.obmalloc_emit = obmalloc_emit  # type: ignore[attr-defined]
        return emit

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

    def _active_phase_payload(
        self,
        record: Mapping[str, Any],
        *,
        guard_event: str,
    ) -> dict[str, Any]:
        budget = self.silent_phase_timeout_seconds
        guard_started_at = float(record.get("guard_started_at", self.clock()))
        payload = {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "event": "active_phase_guard",
            "guard_event": str(guard_event),
            "phase": str(record["phase"]),
            "elapsed_since_start_seconds": self._elapsed(),
            "active_phase_elapsed_seconds": max(0.0, float(self.clock() - guard_started_at)),
            "budget_seconds": budget,
            "breach_after_silent_seconds": budget,
            "device": str(self.device),
            "pid": int(os.getpid()),
            "failure_class": "LIVENESS_FAILURE",
            "stack_source": "faulthandler.dump_traceback_later",
            "fail_closed_termination": "faulthandler_exit_true",
            "resource_snapshot": _proc_self_resource_snapshot(),
            **dict(record.get("fields", {})),
        }
        return payload

    def _write_last_active_phase(self, payload: Mapping[str, Any]) -> None:
        if self.last_active_phase_path is None:
            return
        try:
            _write_json_atomic(self.last_active_phase_path, payload)
        except Exception as exc:
            raise RuntimeError(
                f"failed to write last-active-phase record: {self.last_active_phase_path}"
            ) from exc

    def _cancel_faulthandler_timer(self) -> None:
        if self.silent_phase_timeout_seconds is None or not self.arm_faulthandler_timer:
            return
        faulthandler.cancel_dump_traceback_later()

    def _arm_current_phase(self, record: dict[str, Any], *, guard_event: str) -> None:
        if self.silent_phase_timeout_seconds is None:
            return
        record["guard_started_at"] = float(self.clock())
        payload = self._active_phase_payload(record, guard_event=guard_event)
        self._last_active_phase_payload = payload
        self._write_last_active_phase(payload)
        if self.last_active_phase_path is not None:
            try:
                _write_liveness_stack_dump(
                    dump_path=self.last_active_phase_path.parent / "liveness_stack_dump.txt",
                    guard_event=str(guard_event),
                    phase=str(record["phase"]),
                    payload=payload,
                )
            except Exception:
                pass
        if not self.arm_faulthandler_timer:
            return
        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                float(self.silent_phase_timeout_seconds),
                repeat=False,
                exit=True,
            )
        except Exception as exc:
            raise RuntimeError("failed to arm silent phase faulthandler guard") from exc

    def _enter_phase(self, phase: str, phase_start: float, fields: Mapping[str, Any]) -> None:
        record: dict[str, Any] = {
            "phase": str(phase),
            "fields": dict(fields),
            "phase_started_at": float(phase_start),
            "guard_started_at": float(phase_start),
        }
        self._phase_stack.append(record)
        try:
            self._arm_current_phase(record, guard_event="enter")
        except Exception:
            self._phase_stack.pop()
            self._cancel_faulthandler_timer()
            raise
        if self._ring_sampler is not None:
            self._ring_sampler.start(str(phase), flush_path=self._ring_jsonl_path)

    def _pop_phase_from_stack(self, phase: str) -> None:
        if self._phase_stack and self._phase_stack[-1]["phase"] == str(phase):
            self._phase_stack.pop()
            return
        for index in range(len(self._phase_stack) - 1, -1, -1):
            if self._phase_stack[index]["phase"] == str(phase):
                del self._phase_stack[index]
                break

    def _exit_phase_stack(self, phase: str) -> None:
        # Nested exit cancels the inner faulthandler timer before re-arming the
        # parent phase guard (resume), so no stale dump_traceback_later survives.
        if self._ring_sampler is not None:
            self._ring_sampler.stop()
        self._pop_phase_from_stack(phase)
        self._cancel_faulthandler_timer()
        if self._phase_stack:
            self._arm_current_phase(self._phase_stack[-1], guard_event="resume")
            if self._ring_sampler is not None:
                self._ring_sampler.start(
                    str(self._phase_stack[-1]["phase"]),
                    flush_path=self._ring_jsonl_path,
                )
        elif self._ring_sampler is not None and self._ring_jsonl_path is not None:
            self._ring_sampler.flush_jsonl(self._ring_jsonl_path)

    @property
    def live_heartbeat_thread_count(self) -> int:
        self._live_heartbeat_threads = [
            thread for thread in self._live_heartbeat_threads if thread.is_alive()
        ]
        return len(self._live_heartbeat_threads)

    def _register_heartbeat_thread(self, thread: threading.Thread) -> None:
        self._live_heartbeat_threads.append(thread)

    def _unregister_heartbeat_thread(self, thread: threading.Thread) -> None:
        while thread in self._live_heartbeat_threads:
            self._live_heartbeat_threads.remove(thread)

    def _run_phase_heartbeat_loop(
        self,
        phase: str,
        fields: Mapping[str, Any],
        phase_started_at: float,
        stop_event: threading.Event,
    ) -> None:
        interval = self.phase_heartbeat_interval_seconds
        if interval is None or float(interval) <= 0.0:
            return
        while not stop_event.wait(timeout=float(interval)):
            if not self._phase_stack or self._phase_stack[-1]["phase"] != str(phase):
                return
            elapsed = max(0.0, float(self.clock() - phase_started_at))
            self.mark(
                phase,
                "heartbeat",
                active_phase_elapsed_seconds=elapsed,
                **fields,
            )

    def _exit_phase(self, phase: str) -> None:
        self._exit_phase_stack(phase)

    def _cleared_active_phase_payload(self, phase: str) -> dict[str, Any]:
        return {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "event": "active_phase_guard",
            "guard_event": "cleared",
            "phase_status": "completed",
            "phase": str(phase),
            "liveness_failure": False,
            "elapsed_since_start_seconds": self._elapsed(),
            "device": str(self.device),
            "pid": int(os.getpid()),
            "resource_snapshot": _proc_self_resource_snapshot(),
        }

    def _write_cleared_last_active_phase_if_idle(self, phase: str) -> None:
        if self._phase_stack:
            return
        if self.last_active_phase_path is None:
            return
        payload = self._cleared_active_phase_payload(phase)
        self._last_active_phase_payload = payload
        self._write_last_active_phase(payload)

    def check_stale_active_phase(self) -> dict[str, Any] | None:
        if self.silent_phase_timeout_seconds is None or not self._phase_stack:
            return None
        record = self._phase_stack[-1]
        guard_started_at = float(record.get("guard_started_at", record["phase_started_at"]))
        silent_seconds = max(0.0, float(self.clock() - guard_started_at))
        budget = float(self.silent_phase_timeout_seconds)
        if silent_seconds <= budget:
            return None
        fields = dict(record.get("fields", {}))
        raise C2PhaseTimeout(
            phase=str(record["phase"]),
            bound_kind="silent_phase",
            duration_seconds=silent_seconds,
            timeout_seconds=budget,
            failure_class="LIVENESS_FAILURE",
            active_phase=str(record["phase"]),
            silent_seconds=silent_seconds,
            pid=int(os.getpid()),
            stack_source="faulthandler.dump_traceback_later",
            fail_closed_termination="faulthandler_exit_true",
            **fields,
        )

    @contextmanager
    def phase(self, phase: str, **fields: Any) -> Any:
        phase_start = float(self.clock())
        self._enter_phase(phase, phase_start, fields)
        self.mark(phase, "start", **fields)
        self._emit_host_rss_profile_mark(phase, "enter", fields)
        stop_event = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        if (
            self.enabled
            and self.phase_heartbeat_interval_seconds is not None
            and float(self.phase_heartbeat_interval_seconds) > 0.0
        ):
            heartbeat_thread = threading.Thread(
                target=self._run_phase_heartbeat_loop,
                args=(phase, fields, phase_start, stop_event),
                daemon=True,
                name=f"phase-heartbeat-{phase}",
            )
            heartbeat_thread.start()
            self._register_heartbeat_thread(heartbeat_thread)
        try:
            yield
        except Exception as exc:
            if heartbeat_thread is not None:
                stop_event.set()
                heartbeat_thread.join(timeout=1.0)
                self._unregister_heartbeat_thread(heartbeat_thread)
            self._exit_phase(phase)
            self.mark(
                phase,
                "error",
                duration_seconds=max(0.0, float(self.clock() - phase_start)),
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        finally:
            if heartbeat_thread is not None:
                stop_event.set()
                heartbeat_thread.join(timeout=1.0)
                self._unregister_heartbeat_thread(heartbeat_thread)
        duration = max(0.0, float(self.clock() - phase_start))
        self._exit_phase(phase)
        try:
            enforce_phase_bound(
                phase=phase,
                duration_seconds=duration,
                timeout_seconds=self._phase_timeout_for(phase),
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
        end_fields = dict(fields)
        if (
            str(phase) in self.phase_timeout_exemptions
            and self.phase_timeout_seconds is not None
            and duration > float(self.phase_timeout_seconds)
        ):
            end_fields["phase_timeout_exempted"] = True
            end_fields["scalar_phase_timeout_seconds"] = self.phase_timeout_seconds
        self.mark(phase, "end", duration_seconds=duration, **end_fields)
        self._emit_host_rss_profile_mark(phase, "exit", end_fields)
        if self.milestone_emitter is not None:
            milestone_phase_id = PROBE_PHASE_TO_MILESTONE_PHASE_ID.get(str(phase))
            if milestone_phase_id is not None:
                step_index = fields.get("step")
                optimizer_step_index = int(step_index) if step_index is not None else None
                self.milestone_emitter.record_phase_complete(
                    milestone_phase_id,
                    optimizer_step_index=optimizer_step_index,
                    elapsed_since_phase_enter_seconds=duration,
                )
        self._write_cleared_last_active_phase_if_idle(phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "phase_timeout_seconds": self.phase_timeout_seconds,
            "phase_timeout_exemption_contract": self.phase_timeout_exemption_contract,
            "phase_timeout_exemptions": sorted(self.phase_timeout_exemptions),
            "total_timeout_seconds": self.total_timeout_seconds,
            "silent_phase_timeout_seconds": self.silent_phase_timeout_seconds,
            "phase_heartbeat_interval_seconds": self.phase_heartbeat_interval_seconds,
            "last_active_phase_path": (
                None
                if self.last_active_phase_path is None
                else str(self.last_active_phase_path)
            ),
            "faulthandler_timer_enabled": bool(self.arm_faulthandler_timer),
            "last_active_phase": self._last_active_phase_payload,
            "event_count": len(self.events),
            "events": list(self.events),
        }


def build_receipt_terminal_status(
    *,
    stop_reason: str,
    steps_completed: int,
    steps_requested: int,
    producer_clean_completion: bool = True,
) -> dict[str, Any]:
    return {
        "stop_reason": str(stop_reason),
        "steps_completed": int(steps_completed),
        "steps_requested": int(steps_requested),
        "producer_clean_completion": bool(producer_clean_completion),
        "planned_return_code": 0,
        "classification_source": "receipt_fields+wrapper_rc+stdout_phase_end",
    }


def _attach_obmalloc_dedup_evidence(receipt: dict[str, Any]) -> None:
    """Transcribe dedup fields from the reset call witness (never hardcoded true)."""
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        build_obmalloc_dedup_evidence_from_witness,
    )

    evidence = build_obmalloc_dedup_evidence_from_witness()
    receipt["dedup_reset_called"] = evidence.get("dedup_reset_called")
    receipt["dedup_session_scope"] = evidence.get("dedup_session_scope")


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


def _audit_failure_summary(
    score: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    exact_key: str,
) -> dict[str, Any]:
    row_ids = list(metadata.get("row_ids", []))
    source_buckets = list(metadata.get("source_buckets", []))
    row_start = int(metadata.get("row_start", 0))
    failure_row_ids: list[str] = []
    sources_by_row_id: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for row_result in score.get("row_results", []):
        if bool(row_result[exact_key]):
            continue
        row_index = int(row_result["row_index"])
        local_index = row_index - row_start
        row_id = (
            str(row_ids[local_index])
            if 0 <= local_index < len(row_ids)
            else f"{row_index}:missing-row-id"
        )
        source_bucket = (
            str(source_buckets[local_index])
            if 0 <= local_index < len(source_buckets)
            else "unknown"
        )
        failure_row_ids.append(row_id)
        sources_by_row_id[row_id] = source_bucket
        source_counts[source_bucket] = source_counts.get(source_bucket, 0) + 1
    return {
        "failure_row_ids": failure_row_ids,
        "failure_sources_by_row_id": sources_by_row_id,
        "failure_source_counts": dict(sorted(source_counts.items())),
    }


def _merge_failure_summaries(
    batch_reports: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    row_ids: list[str] = []
    sources_by_row_id: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for report in batch_reports:
        for row_id in report.get(f"{prefix}_failure_row_ids", []):
            row_ids.append(str(row_id))
        for row_id, source_bucket in report.get(f"{prefix}_failure_sources_by_row_id", {}).items():
            sources_by_row_id[str(row_id)] = str(source_bucket)
        for source_bucket, count in report.get(f"{prefix}_failure_source_counts", {}).items():
            source_counts[str(source_bucket)] = source_counts.get(str(source_bucket), 0) + int(count)
    return {
        f"{prefix}_failure_row_ids": sorted(row_ids),
        f"{prefix}_failure_sources_by_row_id": dict(sorted(sources_by_row_id.items())),
        f"{prefix}_failure_source_counts": dict(sorted(source_counts.items())),
    }


def aggregate_prior_audit_batch_reports(
    *,
    support: str,
    support_proof: Mapping[str, Any],
    phase: str,
    step: int,
    batch_reports: Sequence[Mapping[str, Any]],
    bp_steps: int,
    duration_seconds: float,
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
        or strict_metric_total != int(support_proof["expected_count"])
    )
    audited_hashes = [
        report["metadata"]["batch_content_hash16"]
        for report in batch_reports
    ]
    strict_failures = _merge_failure_summaries(batch_reports, prefix="strict")
    parsed_failures = _merge_failure_summaries(batch_reports, prefix="parsed")
    return {
        "schema": B1_PRIOR_AUDIT_SCHEMA_VERSION,
        "phase": phase,
        "step": int(step),
        "support": support,
        "support_role": support_proof["support_role"],
        "support_proof": dict(support_proof),
        "support_rows_expected": int(support_proof["expected_count"]),
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
            strict_metric_count == int(support_proof["expected_count"])
            and strict_metric_total == int(support_proof["expected_count"])
        ),
        "loss_mean": (
            float(sum(loss_values) / len(loss_values))
            if loss_values
            else None
        ),
        "bp_steps": int(bp_steps),
        "duration_seconds": float(duration_seconds),
        "audit_mismatch": bool(audit_mismatch),
        "report_only": True,
        "direct_kl": False,
        "replay_pc": "OUT",
        "target_parent_kl": False,
        **strict_failures,
        **parsed_failures,
        "batch_reports": list(batch_reports),
    }


def audit_prior_support(
    model: LMHead,
    prior_support_set: Mapping[str, Any],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    tok: Any,
    device: torch.device,
    phase: str,
    step: int,
    total_steps: int,
) -> dict[str, Any]:
    support = str(prior_support_set["support"])
    support_batches = list(prior_support_set["batches"])
    support_proof = dict(prior_support_set["proof"])
    was_training = model.training
    model.eval()
    batch_reports: list[dict[str, Any]] = []
    timing_start = _timing_start(device)
    try:
        extras = model.compute_train_extra_args(int(step), max(1, int(total_steps)))
        with torch.no_grad():
            with authoritative_forward_context(
                eligible_modules,
                tensor_states,
                device=device,
                requires_grad=False,
            ):
                for batch_item in support_batches:
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
                        include_row_results=True,
                    )
                    strict_failure_summary = _audit_failure_summary(
                        score,
                        metadata,
                        exact_key="strict_exact",
                    )
                    parsed_failure_summary = _audit_failure_summary(
                        score,
                        metadata,
                        exact_key="parsed_exact",
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
                            "strict_failure_row_ids": strict_failure_summary["failure_row_ids"],
                            "strict_failure_sources_by_row_id": strict_failure_summary[
                                "failure_sources_by_row_id"
                            ],
                            "strict_failure_source_counts": strict_failure_summary[
                                "failure_source_counts"
                            ],
                            "parsed_failure_row_ids": parsed_failure_summary["failure_row_ids"],
                            "parsed_failure_sources_by_row_id": parsed_failure_summary[
                                "failure_sources_by_row_id"
                            ],
                            "parsed_failure_source_counts": parsed_failure_summary[
                                "failure_source_counts"
                            ],
                        }
                    )
    finally:
        model.train(was_training)
    duration_seconds = _timing_duration_seconds(timing_start, device)
    return aggregate_prior_audit_batch_reports(
        support=support,
        support_proof=support_proof,
        phase=phase,
        step=int(step),
        batch_reports=batch_reports,
        bp_steps=int(extras["bp_steps"]),
        duration_seconds=duration_seconds,
    )


def audit_prior_support_sets(
    model: LMHead,
    prior_support_sets: Mapping[str, Mapping[str, Any]],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    tok: Any,
    device: torch.device,
    phase: str,
    step: int,
    total_steps: int,
) -> dict[str, Any]:
    return {
        support: audit_prior_support(
            model,
            prior_support_sets[support],
            tensor_states,
            eligible_modules,
            tok=tok,
            device=device,
            phase=phase,
            step=int(step),
            total_steps=int(total_steps),
        )
        for support in prior_support_sets
    }


def _new_failure_source_counts(
    final_report: Mapping[str, Any],
    new_row_ids: Sequence[str],
    *,
    prefix: str,
) -> dict[str, int]:
    source_by_row = final_report.get(f"{prefix}_failure_sources_by_row_id", {})
    counts: dict[str, int] = {}
    for row_id in new_row_ids:
        source = str(source_by_row.get(row_id, "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def build_prior_audit_delta(
    *,
    support: str,
    start_report: Mapping[str, Any],
    final_report: Mapping[str, Any],
) -> dict[str, Any]:
    start_strict_failures = set(start_report.get("strict_failure_row_ids", []))
    final_strict_failures = set(final_report.get("strict_failure_row_ids", []))
    start_parsed_failures = set(start_report.get("parsed_failure_row_ids", []))
    final_parsed_failures = set(final_report.get("parsed_failure_row_ids", []))
    new_strict_failures = sorted(final_strict_failures - start_strict_failures)
    new_parsed_failures = sorted(final_parsed_failures - start_parsed_failures)
    strict_source_counts = _new_failure_source_counts(
        final_report,
        new_strict_failures,
        prefix="strict",
    )
    parsed_source_counts = _new_failure_source_counts(
        final_report,
        new_parsed_failures,
        prefix="parsed",
    )
    max_new_cluster_count = max(
        [0, *strict_source_counts.values(), *parsed_source_counts.values()]
    )
    broad_cluster_threshold_rows = 3
    no_new_broad_cluster = max_new_cluster_count < broad_cluster_threshold_rows
    return {
        "schema": B1_PRIOR_AUDIT_DELTA_SCHEMA_VERSION,
        "support": support,
        "parent_baseline_step": int(start_report["step"]),
        "final_step": int(final_report["step"]),
        "parent_baseline_vs_final": {
            "baseline_strict_exact": start_report["strict_exact"],
            "final_strict_exact": final_report["strict_exact"],
            "strict_exact_count_delta": int(final_report["strict_exact_count"])
            - int(start_report["strict_exact_count"]),
            "baseline_parsed_exact": start_report["parsed_exact"],
            "final_parsed_exact": final_report["parsed_exact"],
            "parsed_exact_count_delta": int(final_report["parsed_exact_count"])
            - int(start_report["parsed_exact_count"]),
        },
        "new_strict_failure_count": len(new_strict_failures),
        "new_strict_failure_row_ids": new_strict_failures,
        "new_strict_failure_source_counts": strict_source_counts,
        "new_parsed_failure_count": len(new_parsed_failures),
        "new_parsed_failure_row_ids": new_parsed_failures,
        "new_parsed_failure_source_counts": parsed_source_counts,
        "broad_cluster_threshold_rows": broad_cluster_threshold_rows,
        "max_new_failure_cluster_count": max_new_cluster_count,
        "no_new_broad_cluster": bool(no_new_broad_cluster),
        "broad_cluster_classification": (
            "no-new-broad-cluster"
            if no_new_broad_cluster
            else "broad-cluster"
        ),
    }


def build_prior_audit_receipt(
    *,
    requested_supports: Sequence[str],
    support_sets: Mapping[str, Mapping[str, Any]],
    start_reports: Mapping[str, Mapping[str, Any]],
    final_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not requested_supports:
        return {
            "schema": B1_PRIOR_AUDIT_SCHEMA_VERSION,
            "enabled": False,
            "requested_supports": [],
            "default_off": True,
            "prior_batches_fed_to_bounded_steps": False,
            "direct_kl": False,
            "replay_pc": "OUT",
            "target_parent_kl": False,
        }
    deltas = {
        support: build_prior_audit_delta(
            support=support,
            start_report=start_reports[support],
            final_report=final_reports[support],
        )
        for support in requested_supports
    }
    per_support = {
        support: {
            "support_hash16": support_sets[support]["proof"]["support_hash16"],
            "support_rows_expected": support_sets[support]["proof"]["expected_count"],
            "builder_path": support_sets[support]["proof"]["builder_path"],
            "support_role": support_sets[support]["proof"]["support_role"],
            "start": {
                "strict_exact": start_reports[support]["strict_exact"],
                "parsed_exact": start_reports[support]["parsed_exact"],
                "duration_seconds": start_reports[support]["duration_seconds"],
            },
            "final": {
                "strict_exact": final_reports[support]["strict_exact"],
                "parsed_exact": final_reports[support]["parsed_exact"],
                "duration_seconds": final_reports[support]["duration_seconds"],
            },
            "delta": deltas[support],
        }
        for support in requested_supports
    }
    total_duration_seconds = sum(
        float(report["duration_seconds"])
        for report in list(start_reports.values()) + list(final_reports.values())
    )
    return {
        "schema": B1_PRIOR_AUDIT_SCHEMA_VERSION,
        "enabled": True,
        "requested_supports": list(requested_supports),
        "default_off": False,
        "prior_batches_fed_to_bounded_steps": False,
        "direct_kl": False,
        "replay_pc": "OUT",
        "target_parent_kl": False,
        "support_proofs": {
            support: support_sets[support]["proof"]
            for support in requested_supports
        },
        "start_reports": dict(start_reports),
        "final_reports": dict(final_reports),
        "deltas": deltas,
        "per_support": per_support,
        "total_duration_seconds": float(total_duration_seconds),
    }


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


def update_strict_exact_stop_state(
    *,
    step: int,
    audit_report: Mapping[str, Any],
    stop_on_strict_exact: bool,
    matched_continued_training_horizon_steps: int,
    first_strict_exact_step: int | None,
) -> tuple[int | None, str | None]:
    if not bool(stop_on_strict_exact) or not bool(audit_report.get("acquired")):
        return first_strict_exact_step, None
    first_step = int(step) if first_strict_exact_step is None else int(first_strict_exact_step)
    horizon = int(matched_continued_training_horizon_steps)
    if horizon < 0:
        raise ValueError("matched_continued_training_horizon_steps must be non-negative")
    if int(step) >= first_step + horizon:
        token = (
            "strict_exact_acquired"
            if horizon == 0
            else "strict_exact_acquired_matched_horizon"
        )
        return first_step, token
    return first_step, None


def build_acquisition_trajectory(
    *,
    audit_enabled: bool,
    audit_reports: Mapping[str, Any],
    step_reports: Mapping[str, Any],
    support_cycler_proof: Mapping[str, Any],
    audit_interval: int,
    stop_on_strict_exact: bool,
    matched_continued_training_horizon_steps: int,
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
            "matched_continued_training_horizon_steps": int(
                matched_continued_training_horizon_steps
            ),
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


def apply_eligible_module_limit(
    eligible: Mapping[str, BitLinear],
    *,
    eligible_scope: str,
    eligible_module_limit: int | None,
) -> dict[str, BitLinear]:
    """Take the first N sorted module keys (prefix of all-bitlinear eligible set)."""

    if eligible_module_limit is None:
        return dict(eligible)
    if eligible_scope != "all-bitlinear":
        raise ValueError(
            "eligible_module_limit requires eligible_scope=all-bitlinear; "
            f"got {eligible_scope!r}"
        )
    limit = int(eligible_module_limit)
    if limit <= 0:
        raise ValueError(f"eligible_module_limit must be positive; got {limit}")
    keys = sorted(eligible.keys())[:limit]
    return {key: eligible[key] for key in keys}


def build_eligible_scale_receipt_fields(
    eligible: Mapping[str, BitLinear],
    *,
    eligible_scope: str,
    eligible_module_limit: int | None,
    eligible_full_count: int,
) -> dict[str, Any]:
    from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
        resolve_obmalloc_expanded_sampled_states,
    )

    n_c4_states = len(eligible)
    sampled = sorted(resolve_obmalloc_expanded_sampled_states(n_c4_states))
    return {
        "eligible_scope": str(eligible_scope),
        "eligible_module_limit": (
            int(eligible_module_limit) if eligible_module_limit is not None else None
        ),
        "eligible_full_module_count": int(eligible_full_count),
        "eligible_module_count": int(n_c4_states),
        "eligible_modules": sorted(eligible),
        "n_c4_states": int(n_c4_states),
        "sampled_states": sampled,
    }


def build_r3_persistent_ledger_receipt(
    tensor_states: Mapping[str, Any],
    *,
    byte_packed_enabled: bool,
) -> dict[str, Any]:
    """Emit byte-derived R3 ledger fields for eligible tensor states."""

    if not bool(byte_packed_enabled):
        return {"enabled": False}
    state_keys: list[str] = []
    qscale_states: list[QScaleWeightState] = []
    packed_payloads = []
    for state_key, state in sorted(tensor_states.items()):
        state_keys.append(str(state_key))
        qscale_states.append(
            QScaleWeightState(
                q_levels=state.q_levels.detach().cpu().contiguous(),
                scale=state.frozen_scale.detach().cpu().contiguous(),
            )
        )
        decoded_i16 = state.decoded_accumulators(rebuild_if_stale=True)
        packed_payloads.append(pack_w6_lanes_to_bytes(decoded_i16))
    per_module_rows = build_r3_per_module_payload_rows(state_keys, packed_payloads)
    ledger = measure_r3_persistent_state_budget(
        qscale_states,
        packed_payloads,
        state_keys=state_keys,
    )
    return {
        "enabled": True,
        "eligible_module_count": len(tensor_states),
        "persistent_accumulator_w6_byte_packed": True,
        "r3_artifact_bytes_semantics": R3_ARTIFACT_BYTES_SEMANTICS_ACTUAL_PAYLOAD,
        "r3_per_module_payload_rows": per_module_rows,
        **ledger.to_dict(),
    }


def build_r4_persistent_ledger_receipt(
    tensor_states: Mapping[str, Any],
    *,
    q_packed_enabled: bool,
    acc_byte_packed_enabled: bool,
) -> dict[str, Any]:
    """Emit byte-derived R4 ledger fields for packed-q + W6-packed acc states."""

    if not bool(q_packed_enabled) or not bool(acc_byte_packed_enabled):
        return {"enabled": False}
    state_keys: list[str] = []
    qscale_states: list[QScaleWeightState] = []
    packed_q_payloads = []
    packed_acc_payloads = []
    for state_key, state in sorted(tensor_states.items()):
        state_keys.append(str(state_key))
        q_levels = state.q_levels.detach().cpu().contiguous()
        qscale_states.append(
            QScaleWeightState(
                q_levels=q_levels,
                scale=state.frozen_scale.detach().cpu().contiguous(),
            )
        )
        packed_q_payloads.append(pack_ternary_q_2bit_reference(q_levels))
        decoded_i16 = state.decoded_accumulators(rebuild_if_stale=True)
        packed_acc_payloads.append(pack_w6_lanes_to_bytes(decoded_i16))
    q_rows = build_r4_per_module_q_rows(state_keys, packed_q_payloads)
    acc_rows = build_r3_per_module_payload_rows(state_keys, packed_acc_payloads)
    ledger = measure_r4_persistent_state_budget(
        qscale_states,
        packed_q_payloads,
        packed_acc_payloads,
        state_keys=state_keys,
    )
    return {
        "enabled": True,
        "eligible_module_count": len(tensor_states),
        "persistent_q_ternary_byte_packed": True,
        "persistent_accumulator_w6_byte_packed": True,
        "r4_per_module_q_rows": q_rows,
        "r4_per_module_acc_rows": acc_rows,
        **ledger.to_dict(),
    }


def build_r4b_persistent_ledger_receipt(
    tensor_states: Mapping[str, Any],
    *,
    q_packed_enabled: bool,
    acc_byte_packed_enabled: bool,
    q_codec_selector: str | None = None,
) -> dict[str, Any]:
    """Emit byte-derived R4b ledger fields from serialized SAVE-path sidecar payloads."""

    if not bool(q_packed_enabled) or not bool(acc_byte_packed_enabled):
        return {"enabled": False}
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        _tensor_state_roundtrip_payload,
        packed_q_state_from_roundtrip_q_payload,
        packed_w6_acc_payload_from_roundtrip_bounded_payload,
    )

    selector = resolve_q_codec_selector(q_codec_selector=q_codec_selector)
    state_keys: list[str] = []
    qscale_states: list[QScaleWeightState] = []
    packed_q_payloads = []
    packed_acc_payloads = []
    for state_key, state in sorted(tensor_states.items()):
        state_keys.append(str(state_key))
        q_levels = state.q_levels.detach().cpu().contiguous()
        qscale_states.append(
            QScaleWeightState(
                q_levels=q_levels,
                scale=state.frozen_scale.detach().cpu().contiguous(),
            )
        )
        roundtrip_payload = _tensor_state_roundtrip_payload(
            state,
            byte_packed_enabled=bool(acc_byte_packed_enabled),
            q_packed_enabled=bool(q_packed_enabled),
            q_codec_selector=selector,
        )
        packed_q_payloads.append(packed_q_state_from_roundtrip_q_payload(roundtrip_payload))
        packed_acc_payloads.append(
            packed_w6_acc_payload_from_roundtrip_bounded_payload(
                dict(roundtrip_payload.get("bounded_accumulator") or {})
            )
        )
    q_rows = build_r4_per_module_q_rows(state_keys, packed_q_payloads)
    acc_rows = build_r3_per_module_payload_rows(state_keys, packed_acc_payloads)
    ledger = measure_r4b_persistent_state_budget(
        qscale_states,
        packed_q_payloads,
        packed_acc_payloads,
        state_keys=state_keys,
    )
    return {
        "enabled": True,
        "eligible_module_count": len(tensor_states),
        "persistent_q_ternary_byte_packed": True,
        "persistent_q_ternary_base3_codec": selector == Q_CODEC_SELECTOR_BASE3,
        "q_codec_selector": selector,
        "persistent_accumulator_w6_byte_packed": True,
        "r4b_per_module_q_rows": q_rows,
        "r4b_per_module_acc_rows": acc_rows,
        **ledger.to_dict(),
    }


def build_r4v_persistent_ledger_receipt(
    tensor_states: Mapping[str, Any],
    *,
    event_coded_live_enabled: bool,
) -> dict[str, Any]:
    """Emit byte-derived R4v ledger fields from serialized SAVE-path event-coded payloads."""

    if not bool(event_coded_live_enabled):
        return {"enabled": False}
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        _packed_event_coded_from_roundtrip_payload,
        _tensor_state_roundtrip_payload,
    )

    state_keys: list[str] = []
    qscale_states: list[QScaleWeightState] = []
    event_payloads = []
    per_module_rows: list[dict[str, int | float | str]] = []
    for state_key, state in sorted(tensor_states.items()):
        if state.event_coded_live_carrier is None:
            raise ValueError(
                f"{state_key} missing event_coded_live_carrier for R4v ledger receipt"
            )
        state_keys.append(str(state_key))
        q_levels = state.q_levels.detach().cpu().contiguous()
        qscale_states.append(
            QScaleWeightState(
                q_levels=q_levels,
                scale=state.frozen_scale.detach().cpu().contiguous(),
            )
        )
        roundtrip_payload = _tensor_state_roundtrip_payload(
            state,
            byte_packed_enabled=False,
            w5_byte_packed_enabled=False,
            q_packed_enabled=False,
            q_codec_selector=Q_CODEC_SELECTOR_2BIT,
        )
        packed = _packed_event_coded_from_roundtrip_payload(roundtrip_payload)
        event_payloads.append(packed)
        events_bytes = int(packed.events_packed.numel())
        backlog_bytes = int(packed.backlog_packed.numel())
        hot_exact_bytes = int(packed.hot_exact_packed.numel())
        logical_numel = int(packed.logical_numel)
        per_module_rows.append(
            {
                "state_key": str(state_key),
                "logical_numel": logical_numel,
                "r4v_actual_events_payload_bytes": events_bytes,
                "r4v_actual_backlog_payload_bytes": backlog_bytes,
                "r4v_actual_hot_exact_payload_bytes": hot_exact_bytes,
            }
        )
    ledger = measure_r4v_event_coded_acc_budget(
        qscale_states,
        event_payloads,
        state_keys=state_keys,
    )
    return {
        "enabled": True,
        "eligible_module_count": len(tensor_states),
        "persistent_accumulator_event_coded_live": True,
        "r4v_per_module_acc_rows": per_module_rows,
        **ledger.to_dict(),
        "ledger_pass": bool(ledger.r4v_ledger_pass),
        "content_sha256": str(ledger.r4v_event_payload_content_sha256),
    }


def build_r5_persistent_ledger_receipt(
    tensor_states: Mapping[str, Any],
    *,
    q_packed_enabled: bool,
    acc_w5_byte_packed_enabled: bool,
) -> dict[str, Any]:
    """Emit byte-derived R5 ledger fields for packed-q + W5-packed acc (decision-parity)."""

    if not bool(q_packed_enabled) or not bool(acc_w5_byte_packed_enabled):
        return {"enabled": False}
    state_keys: list[str] = []
    qscale_states: list[QScaleWeightState] = []
    packed_q_payloads = []
    packed_acc_payloads = []
    for state_key, state in sorted(tensor_states.items()):
        state_keys.append(str(state_key))
        q_levels = state.q_levels.detach().cpu().contiguous()
        qscale_states.append(
            QScaleWeightState(
                q_levels=q_levels,
                scale=state.frozen_scale.detach().cpu().contiguous(),
            )
        )
        packed_q_payloads.append(pack_ternary_q_2bit_reference(q_levels))
        decoded_i16 = state.decoded_accumulators(rebuild_if_stale=True)
        packed_acc_payloads.append(pack_w5_lanes_to_bytes(decoded_i16))
    q_rows = build_r4_per_module_q_rows(state_keys, packed_q_payloads)
    acc_rows = build_r3_per_module_payload_rows(state_keys, packed_acc_payloads)
    ledger = measure_r5_persistent_state_budget(
        qscale_states,
        packed_q_payloads,
        packed_acc_payloads,
        state_keys=state_keys,
    )
    return {
        "enabled": True,
        "eligible_module_count": len(tensor_states),
        "persistent_q_ternary_byte_packed": True,
        "persistent_accumulator_w5_byte_packed": True,
        "r5_per_module_q_rows": q_rows,
        "r5_per_module_acc_rows": acc_rows,
        **ledger.to_dict(),
    }


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


def resolve_probe_vote_update_spec(
    *,
    max_abs_per_tensor: int,
    confirmation_envelope: str | None,
    vote_update_decay_numerator: int | None = None,
    vote_update_decay_denominator: int | None = None,
) -> VoteUpdateSpec:
    envelope = resolve_confirmation_envelope(confirmation_envelope)
    if envelope is not None:
        vote_spec = envelope.vote_update_spec(max_abs_per_tensor=int(max_abs_per_tensor))
    else:
        vote_spec = default_vote_update_spec(int(max_abs_per_tensor))
    if (
        vote_update_decay_numerator is not None
        or vote_update_decay_denominator is not None
    ):
        decay_num = 1 if vote_update_decay_numerator is None else int(
            vote_update_decay_numerator
        )
        decay_den = 1 if vote_update_decay_denominator is None else int(
            vote_update_decay_denominator
        )
        vote_spec = replace(
            vote_spec,
            decay_numerator=decay_num,
            decay_denominator=decay_den,
        )
    vote_spec.validate()
    return vote_spec


def _zero_weighted_grad_sums(
    tensor_states: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(state.q_levels, dtype=torch.float32)
        for key, state in tensor_states.items()
    }


def _add_weighted_grads_in_place(
    destination: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
) -> None:
    for key, value in source.items():
        destination[key] = destination[key] + value.detach().cpu().to(torch.float32)


def _weighted_grads_all_finite(weighted_grads: Mapping[str, torch.Tensor]) -> bool:
    return all(
        bool(torch.isfinite(value).all().item())
        for value in weighted_grads.values()
    )


def _weighted_grads_to_vote_aux_maps(
    weighted_grads: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    rank_spec: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    votes_by_key: dict[str, torch.Tensor] = {}
    moves_by_key: dict[str, torch.Tensor] = {}
    for key, weighted_grad in weighted_grads.items():
        moves = project_s1_gradient_to_moves(weighted_grad, tensor_states[key].q_levels)
        credit = credit_from_weighted_grad(weighted_grad)
        votes_by_key[key] = rank_bucketed_int16_votes(credit, moves, rank_spec)
        moves_by_key[key] = moves.detach().cpu().to(torch.int8).contiguous()
    return votes_by_key, moves_by_key


def _science_arm_vote_law(science_arm: str) -> str:
    arm = str(science_arm)
    if arm in {ARM_A0_RANK_BUCKET_CURRENT, ARM_A1_RANK_BUCKET_ORDER_MATCHED}:
        return S1_RANK_BUCKET_VOTE_LAW
    if arm in {ARM_B_RANK_FREE_SIGN_PRESSURE, ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER}:
        return S1_SIGN_PRESSURE_VOTE_LAW
    if arm == ARM_INVERTED_SIGN_PRESSURE:
        return S1_INVERTED_SIGN_PRESSURE_VOTE_LAW
    raise ValueError(f"unknown science arm {science_arm!r}")


def _science_arm_tie_policy(science_arm: str) -> str:
    arm = str(science_arm)
    if arm == ARM_A0_RANK_BUCKET_CURRENT:
        return TIE_POLICY_CURRENT_MARGIN_INDEX
    if arm == ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER:
        return TIE_POLICY_CURRENT_MARGIN_INDEX
    if arm in {
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
        ARM_INVERTED_SIGN_PRESSURE,
    }:
        return TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    raise ValueError(f"unknown science arm {science_arm!r}")


def _science_local_selection_ordering_mode(science_arm: str) -> str:
    arm = str(science_arm)
    if arm == ARM_A0_RANK_BUCKET_CURRENT:
        return LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX
    if arm == ARM_C_RANK_FREE_SIGN_CURRENT_MARGIN_ORDER:
        return LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX
    if arm in {
        ARM_A1_RANK_BUCKET_ORDER_MATCHED,
        ARM_B_RANK_FREE_SIGN_PRESSURE,
        ARM_INVERTED_SIGN_PRESSURE,
    }:
        return LOCAL_SELECTION_ORDER_DETERMINISTIC_HASH_MATCHED
    raise ValueError(f"unknown science arm {science_arm!r}")


def _science_global_cap_spec_for_arm(
    global_cap_spec: GlobalRateCapSpec | None,
    *,
    science_arm: str,
) -> GlobalRateCapSpec | None:
    if global_cap_spec is None or str(science_arm) == ARM_A0_RANK_BUCKET_CURRENT:
        return global_cap_spec
    return GlobalRateCapSpec(
        cap=int(global_cap_spec.cap),
        step=int(global_cap_spec.step),
        ordering_mode=GlobalRateCapOrderingMode.HASH_SHUFFLE,
        ordering_seed=int(global_cap_spec.ordering_seed),
        functional_veto_policy=global_cap_spec.functional_veto_policy,
        bad_pressure_drain_policy=global_cap_spec.bad_pressure_drain_policy,
        mutate_outputs=bool(global_cap_spec.mutate_outputs),
    )


def _weighted_grads_to_science_arm_votes(
    weighted_grads: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    *,
    rank_spec: Any,
    vote_spec: VoteUpdateSpec,
    science_arm: str,
    sparse_events_out: dict[str, Any] | None = None,
    sparse_construction_only: bool = False,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any], bool]:
    if str(science_arm) not in SCIENCE_ARM_CHOICES:
        raise ValueError(f"science_arm must be one of {SCIENCE_ARM_CHOICES}, got {science_arm!r}")
    votes_by_key: dict[str, torch.Tensor] = {}
    sparse_events_by_key: dict[str, Any] = {}
    pressure_by_key: dict[str, Any] = {}
    finite_weighted_grad = True
    inverted = str(science_arm) == ARM_INVERTED_SIGN_PRESSURE
    if bool(sparse_construction_only) and len(weighted_grads) > 1:
        import os
        from concurrent.futures import ThreadPoolExecutor

        def _init_sparse_worker() -> None:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        def _sparse_events_for_key(item: tuple[str, torch.Tensor]) -> tuple[str, Any]:
            key, weighted_grad = item
            with torch.inference_mode():
                if str(science_arm) in {ARM_A0_RANK_BUCKET_CURRENT, ARM_A1_RANK_BUCKET_ORDER_MATCHED}:
                    sparse_events = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
                        weighted_grad,
                        tensor_states[key].q_levels,
                        rank_spec,
                    )
                else:
                    moves = project_s1_gradient_to_moves(weighted_grad, tensor_states[key].q_levels)
                    sparse_events = sparse_sign_pressure_int16_vote_events(
                        moves,
                        vote_spec,
                        inverted=inverted,
                    )
            return key, sparse_events

        cpu_workers = os.cpu_count() or 4
        max_workers = min(cpu_workers, len(weighted_grads))
        with ThreadPoolExecutor(max_workers=max_workers, initializer=_init_sparse_worker) as pool:
            for key, sparse_events in pool.map(
                _sparse_events_for_key,
                weighted_grads.items(),
                chunksize=max(1, (len(weighted_grads) + max_workers - 1) // max_workers),
            ):
                sparse_events_by_key[key] = sparse_events
                pressure_by_key[key] = {
                    "state_key": key,
                    "science_arm": str(science_arm),
                    "vote_law": _science_arm_vote_law(str(science_arm)),
                    "tie_policy_id": _science_arm_tie_policy(str(science_arm)),
                    "vote_nonzero_count": int(sparse_events.event_count()),
                    "raw_per_proposal_arrays_included": False,
                }
    else:
        for key, weighted_grad in weighted_grads.items():
            finite_weighted_grad = finite_weighted_grad and bool(torch.isfinite(weighted_grad).all().item())
            if str(science_arm) in {ARM_A0_RANK_BUCKET_CURRENT, ARM_A1_RANK_BUCKET_ORDER_MATCHED}:
                if bool(sparse_construction_only):
                    sparse_events_by_key[key] = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
                        weighted_grad,
                        tensor_states[key].q_levels,
                        rank_spec,
                    )
                else:
                    moves = project_s1_gradient_to_moves(weighted_grad, tensor_states[key].q_levels)
                    credit = credit_from_weighted_grad(weighted_grad)
                    votes, sparse_events_by_key[key] = rank_bucketed_int16_votes_and_sparse_events(
                        credit,
                        moves,
                        rank_spec,
                    )
                    votes_by_key[key] = votes
            else:
                moves = project_s1_gradient_to_moves(weighted_grad, tensor_states[key].q_levels)
                if bool(sparse_construction_only):
                    sparse_events_by_key[key] = sparse_sign_pressure_int16_vote_events(
                        moves,
                        vote_spec,
                        inverted=inverted,
                    )
                else:
                    votes, sparse_events_by_key[key] = sign_pressure_int16_votes_and_sparse_events(
                        moves,
                        vote_spec,
                        inverted=inverted,
                    )
                    votes_by_key[key] = votes
            if bool(sparse_construction_only):
                pressure_entry = {
                    "state_key": key,
                    "science_arm": str(science_arm),
                    "vote_law": _science_arm_vote_law(str(science_arm)),
                    "tie_policy_id": _science_arm_tie_policy(str(science_arm)),
                    "vote_nonzero_count": int(sparse_events_by_key[key].event_count()),
                    "raw_per_proposal_arrays_included": False,
                }
            else:
                pressure_entry = {
                    "state_key": key,
                    "science_arm": str(science_arm),
                    "vote_law": _science_arm_vote_law(str(science_arm)),
                    "tie_policy_id": _science_arm_tie_policy(str(science_arm)),
                    **compact_vote_pressure_summary(votes_by_key[key]),
                }
                if str(science_arm) in {ARM_A0_RANK_BUCKET_CURRENT, ARM_A1_RANK_BUCKET_ORDER_MATCHED}:
                    pressure_entry["pressure_shape_summary"] = build_pressure_shape_summary_v1(
                        credit,
                        moves,
                        rank_spec,
                    )
            pressure_by_key[key] = pressure_entry
    if sparse_events_out is not None:
        sparse_events_out.clear()
        sparse_events_out.update(sparse_events_by_key)
    if bool(sparse_construction_only):
        return None, pressure_by_key, finite_weighted_grad
    return votes_by_key, pressure_by_key, finite_weighted_grad


def _compute_ce_weighted_grads(
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    extras: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, Mapping[str, Any]]:
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible_modules,
        tensor_states,
        device=device,
        requires_grad=True,
    ) as handle:
        _carry, loss, metrics = model(None, dict(batch), **extras)
        loss.backward()
        weighted_grads = {
            key: handle.weighted_grad(key)
            for key in tensor_states
        }
    model.zero_grad(set_to_none=True)
    return weighted_grads, loss.detach(), metrics


def _parent_consistency_kl(
    child_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temp: float = 1.0,
) -> torch.Tensor:
    mask = labels != IGNORE_LABEL_ID
    if not bool(mask.any().item()):
        return child_logits.new_zeros(())
    temp_f = float(temp)
    child_logp = F.log_softmax(child_logits[mask] / temp_f, dim=-1)
    parent_logp = F.log_softmax(parent_logits[mask] / temp_f, dim=-1)
    parent_p = parent_logp.exp()
    return F.kl_div(child_logp, parent_p, reduction="batchmean") * (temp_f ** 2)


def _compute_pc_weighted_grads(
    model: LMHead,
    parent_model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    *,
    device: torch.device,
    extras: Mapping[str, Any],
    weight: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    model.zero_grad(set_to_none=True)
    parent_model.eval()
    with torch.no_grad():
        _parent_carry, _parent_loss, parent_metrics = parent_model(
            None,
            dict(batch),
            return_logits=True,
            **extras,
        )
        parent_logits = parent_metrics["logits"].detach()
    with authoritative_forward_context(
        eligible_modules,
        tensor_states,
        device=device,
        requires_grad=True,
    ) as handle:
        _child_carry, _child_loss, child_metrics = model(
            None,
            dict(batch),
            return_logits=True,
            **extras,
        )
        kl = _parent_consistency_kl(
            child_metrics["logits"],
            parent_logits,
            batch["labels"],
        )
        (float(weight) * kl).backward()
        weighted_grads = {
            key: handle.weighted_grad(key)
            for key in tensor_states
        }
    model.zero_grad(set_to_none=True)
    return weighted_grads, kl.detach()


def _empty_b2_step_aux_receipt() -> dict[str, Any]:
    return {
        "enabled": False,
        "support_batches": [],
        "coverage_by_support": {},
        "replay_ce_veto_generated": False,
        "pc_aux_generated": False,
    }


def build_b2_retention_receipt(
    *,
    requested_supports: Sequence[str],
    support_sets: Mapping[str, Mapping[str, Any]],
    step_reports: Mapping[str, Mapping[str, Any]],
    pc_aux_mode: str,
    parent_consistency_weight: float,
) -> dict[str, Any]:
    if not requested_supports:
        return {
            "schema": B2_RETAINED_SUPPORT_SCHEMA_VERSION,
            "enabled": False,
            "default_off": True,
            "requested_supports": [],
            "prior_batches_fed_to_bounded_steps": False,
            "replay_ce_veto": False,
            "pc_aux_mode": str(pc_aux_mode),
            "pc_aux_enabled": False,
            "target_parent_kl": False,
            "l0c1_report_only_b2": True,
        }
    replay_veto_count = 0
    pc_aux_negative_count = 0
    pc_aux_veto_count = 0
    post_veto_acceptance_ratios: list[float] = []
    coverage_by_support: dict[str, Any] = {}
    for report in step_reports.values():
        step_result = report.get("step_result", {})
        for stats in step_result.get("tensor_stats", {}).values():
            replay_veto_count += int(stats.get("replay_ce_veto_count", 0))
            pc_aux_negative_count += int(stats.get("pc_aux_negative_count", 0))
            pc_aux_veto_count += int(stats.get("pc_aux_veto_count", 0))
            post_veto_acceptance_ratios.append(
                float(stats.get("post_veto_acceptance_ratio_pre_cap", 0.0))
            )
        for support, coverage in report.get("b2_retained_support", {}).get(
            "coverage_by_support",
            {},
        ).items():
            coverage_by_support[support] = coverage
    return {
        "schema": B2_RETAINED_SUPPORT_SCHEMA_VERSION,
        "enabled": True,
        "default_off": False,
        "requested_supports": list(requested_supports),
        "prior_batches_fed_to_bounded_steps": True,
        "replay_ce_veto": True,
        "pc_aux_mode": str(pc_aux_mode),
        "pc_aux_enabled": float(parent_consistency_weight) > 0.0,
        "parent_consistency_weight": float(parent_consistency_weight),
        "target_parent_kl": False,
        "target_rows_excluded_from_pc": True,
        "l0c1_report_only_b2": True,
        "support_proofs": {
            support: support_sets[support]["proof"]
            for support in requested_supports
        },
        "coverage_by_support": coverage_by_support,
        "replay_ce_veto_count": replay_veto_count,
        "pc_aux_negative_count": pc_aux_negative_count,
        "pc_aux_veto_count": pc_aux_veto_count,
        "post_veto_acceptance_ratio_pre_cap_min": (
            min(post_veto_acceptance_ratios)
            if post_veto_acceptance_ratios
            else None
        ),
        "post_veto_acceptance_ratio_pre_cap_max": (
            max(post_veto_acceptance_ratios)
            if post_veto_acceptance_ratios
            else None
        ),
    }


def _init_b2b_sequential_trace_capture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_b2b_sequential_trace_step(path: Path, step_record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(step_record), sort_keys=True) + "\n")


def build_b2b_sequential_capture_receipt(
    *,
    capture_out: Path,
    steps_captured: int,
    min_steps_for_verdict: int,
    trace_hashes: Sequence[str],
    parent_hash_unchanged: bool,
    max_sampled_candidates: int,
) -> dict[str, Any]:
    trace_hash = _sha16(list(trace_hashes)) if trace_hashes else None
    verdict_eligible = int(steps_captured) >= int(min_steps_for_verdict)
    return {
        "receipt_kind": B2B_SEQUENTIAL_CAPTURE_RECEIPT_KIND,
        "proof_side": "b2b_sequential_within_tie_band_capture",
        "pre_full_stack_diagnostic_only": True,
        "measurement_only_pre_full_stack_diagnostic": True,
        "runtime_readiness_claim": False,
        "training_or_acquisition_claim": False,
        "full_sub2_claim": False,
        "q_mutation_applied_to_model": True,
        "accumulator_arm_algorithmic_proxy_not_physical_sub2": True,
        "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        "trace_temporality": TRACE_TEMPORALITY_SEQUENTIAL_OPTIMIZER_STEPS,
        "tracking_scope": TRACKING_SCOPE_OPTIMIZER_STEP_TRAJECTORY,
        "capture_out": str(capture_out),
        "optimizer_steps_captured": int(steps_captured),
        "min_steps_for_verdict": int(min_steps_for_verdict),
        "verdict_eligible": bool(verdict_eligible),
        "trace_hash": trace_hash,
        "max_sampled_candidates": int(max_sampled_candidates),
        "candidate_apply_policy": "full_vote_planned_candidate_force_apply_v1",
        "b2b_oracle_estimand": "full_vote_planned_candidate_marginal",
        "cross_comparable_to_single_step_oracle_screen": False,
        "estimand_non_comparable_to_single_step_sparse_singleton_oracle": True,
        "checkpoint_written": False,
        "creditdir_mutated": False,
        "banked_pt_mutated": False,
        "parent_hash_unchanged": bool(parent_hash_unchanged),
        "pt_writes_allowed": False,
    }


TIER_A_PROBE_RECEIPT_INDEX_SURFACE_KEYS: frozenset[str] = frozenset(
    {
        "pre_veto_selected_indices",
        "applied_indices",
        "post_veto_would_apply_pre_cap_indices",
        "replay_ce_veto_indices",
    }
)

# Receipt-only audit telemetry for cap-window identity/quality (Stage B).
# Non-authoritative: must not feed selection, gating, or runtime decisions.
# See two_tier_transient_selection.FORBIDDEN_PERSIST_SELECTOR_SURFACES.
CAP_WINDOW_AUDIT_SURFACE_KEYS: frozenset[str] = frozenset(
    {
        "applied_flat_indices_hash16",
        "top8_flat_indices_hash16",
        "top64_flat_indices_hash16",
        "top4096_flat_indices_hash16",
        "cap_window_jaccard_vs_prior_step",
        "applied_selection_score_p50",
        "applied_selection_score_p95",
        "applied_selection_score_semantics",
        "cap_window_audit_non_authoritative",
    }
)


def _sorted_flat_indices_hash16(indices: Sequence[int]) -> str:
    return _sha16([int(value) for value in sorted(indices)])


def _cap_window_jaccard_vs_prior(
    applied_indices: Sequence[int],
    prior_applied_indices: Sequence[int] | None,
) -> float | None:
    if prior_applied_indices is None:
        return None
    current = {int(value) for value in applied_indices}
    prior = {int(value) for value in prior_applied_indices}
    if not current and not prior:
        return 1.0
    union = current | prior
    if not union:
        return 1.0
    return float(len(current & prior)) / float(len(union))


def _within_arm_score_quantiles(
    values: Sequence[float],
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(float(value) for value in values)
    p50 = float(statistics.median(ordered))
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95)))
    if len(ordered) > 1 and p95_index == 0:
        p95_index = len(ordered) - 1
    p95 = float(ordered[p95_index])
    return p50, p95


def _applied_selection_scores_for_plan(
    plan: Any,
    *,
    local_loss_delta: torch.Tensor | None,
) -> tuple[float | None, float | None, str]:
    applied = [
        int(value) for value in plan.applied_indices.detach().cpu().tolist()
    ]
    if not applied:
        return None, None, "empty_applied_set"
    if local_loss_delta is not None:
        delta = local_loss_delta.detach().cpu().to(torch.float32).flatten()
        scores = [float(delta[int(index)].item()) for index in applied]
        return (
            *_within_arm_score_quantiles(scores),
            "local_loss_delta_at_applied_flat_index",
        )
    new_acc = plan.new_acc_i32.detach().cpu().flatten()
    scores = [float(abs(int(new_acc[int(index)].item()))) for index in applied]
    return (
        *_within_arm_score_quantiles(scores),
        "abs_new_acc_at_applied_flat_index",
    )


def _attach_cap_window_audit_surfaces(
    step_result_compact: Mapping[str, Any],
    *,
    plans_by_key: Mapping[str, Any],
    prior_applied_by_state_key: Mapping[str, Sequence[int]],
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None,
    optimizer_step_index: int,
) -> dict[str, Any]:
    """Attach non-authoritative cap-window audit sketches to step_result compact."""

    compact = dict(step_result_compact)
    tensor_stats = {
        state_key: dict(stats)
        for state_key, stats in dict(compact.get("tensor_stats", {})).items()
    }
    for state_key, plan in sorted(plans_by_key.items()):
        stats = dict(tensor_stats[state_key])
        applied_indices = [
            int(value) for value in plan.applied_indices.detach().cpu().tolist()
        ]
        pre_veto_indices = [
            int(value)
            for value in plan.pre_veto_selected_indices.detach().cpu().tolist()
        ]
        local_loss_delta = (
            local_loss_delta_by_key[state_key]
            if local_loss_delta_by_key is not None
            else None
        )
        score_p50, score_p95, score_semantics = _applied_selection_scores_for_plan(
            plan,
            local_loss_delta=local_loss_delta,
        )
        stats["applied_flat_indices_hash16"] = _sorted_flat_indices_hash16(
            applied_indices
        )
        stats["top8_flat_indices_hash16"] = _sorted_flat_indices_hash16(
            pre_veto_indices[:8]
        )
        stats["top64_flat_indices_hash16"] = _sorted_flat_indices_hash16(
            pre_veto_indices[:64]
        )
        stats["top4096_flat_indices_hash16"] = _sorted_flat_indices_hash16(
            pre_veto_indices[:4096]
        )
        stats["cap_window_jaccard_vs_prior_step"] = _cap_window_jaccard_vs_prior(
            applied_indices,
            prior_applied_by_state_key.get(state_key),
        )
        stats["applied_selection_score_p50"] = score_p50
        stats["applied_selection_score_p95"] = score_p95
        stats["applied_selection_score_semantics"] = score_semantics
        stats["cap_window_audit_non_authoritative"] = True
        stats["cap_window_audit_optimizer_step_index"] = int(optimizer_step_index)
        tensor_stats[state_key] = stats
    compact["tensor_stats"] = tensor_stats
    compact["cap_window_audit_non_authoritative"] = True
    compact["cap_window_audit_forbidden_persistent_authority_surfaces"] = list(
        FORBIDDEN_PERSIST_SELECTOR_SURFACES
    )
    return compact


def _update_prior_applied_by_state_key(
    prior_applied_by_state_key: dict[str, list[int]],
    step_result_compact: Mapping[str, Any],
) -> None:
    for state_key, stats in dict(step_result_compact.get("tensor_stats", {})).items():
        applied = stats.get("applied_indices")
        if applied is not None:
            prior_applied_by_state_key[state_key] = [int(value) for value in applied]


def _optional_step_vote_tensor(
    values_by_key: Mapping[str, torch.Tensor] | None,
    state_key: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if values_by_key is None:
        return None
    return values_by_key[state_key].detach().cpu().to(dtype).contiguous()


def _bounded_delta_vote_step_two_tier_kwargs(
    *,
    two_tier_carry_w6_enabled: bool,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None,
) -> dict[str, Any]:
    if not two_tier_carry_w6_enabled:
        return {}
    if local_loss_delta_by_key is None:
        raise ValueError("local_loss_delta_by_key required when two_tier_carry_w6_enabled")
    return {
        "two_tier_carry_w6_enabled": True,
        "local_loss_delta_by_key": local_loss_delta_by_key,
    }


def resolve_r7_deferred_backlog_vote_step_kwargs(
    *,
    r7_deferred_backlog_carry_enabled: bool,
    carry_backlog: dict[str, dict[int, dict[str, int]]] | None,
) -> dict[str, Any]:
    if not r7_deferred_backlog_carry_enabled:
        if carry_backlog is not None:
            raise ValueError(
                "r7_deferred_backlog_carry_enabled is false but carry_backlog is non-None"
            )
        return {}
    return {"deferred_backlog": carry_backlog}


def _materialize_selector_rows_for_crossing_coverage(
    *,
    votes: torch.Tensor,
    state: Any,
) -> list[dict[str, Any]]:
    vu_state = state.vote_update_state()
    q_levels = vu_state.q_levels.flatten()
    accumulators = vu_state.accumulators.flatten()
    vote_flat = votes.flatten()
    return [
        {
            "flat_index": int(flat_index),
            "vote_value": int(vote_flat[flat_index].item()),
            "pre_accumulator_i16": int(accumulators[flat_index].item()),
            "current_q_level": int(q_levels[flat_index].item()),
        }
        for flat_index in range(int(q_levels.numel()))
    ]


def _assert_local_loss_delta_crossing_coverage(
    *,
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    sampled_candidates: Sequence[Mapping[str, Any]],
    budget_exceeded: bool,
    candidate_count: int,
    sampled_count: int,
    max_sampled_candidates: int,
) -> None:
    if budget_exceeded:
        raise ValueError(
            "local_loss_delta_incomplete_candidate_coverage: "
            f"oracle_screen_budget_exceeded "
            f"max_sampled_candidates={int(max_sampled_candidates)}"
        )
    measured = {
        (str(candidate["state_key"]), int(candidate["flat_index"]))
        for candidate in sampled_candidates
    }
    missing: list[str] = []
    for state_key, state in sorted(tensor_states.items()):
        rows = _materialize_selector_rows_for_crossing_coverage(
            votes=votes_by_key[state_key],
            state=state,
        )
        for flat_index in crossing_eligible_flat_indices(rows):
            if (state_key, flat_index) not in measured:
                missing.append(f"{state_key}:{flat_index}")
    if missing:
        raise ValueError(
            "local_loss_delta_incomplete_candidate_coverage: "
            f"unmeasured_crossing_eligible_rows={missing} "
            f"candidate_count={int(candidate_count)} "
            f"sampled_count={int(sampled_count)} "
            f"max_sampled_candidates={int(max_sampled_candidates)}"
        )


def _assert_local_loss_delta_proxy_crossing_coverage(
    *,
    local_loss_delta_by_key: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
) -> None:
    assert_local_loss_delta_proxy_coverage(
        local_loss_delta_by_key=local_loss_delta_by_key,
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    )


def _zero_fill_non_crossing_unmeasured_local_loss_deltas(
    *,
    local_loss_delta_by_key: dict[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
) -> None:
    # Non-crossing rows never enter two-tier selection; only they may remain
    # zero-filled when unmeasured. Crossing-eligible rows must be measured
    # before this helper runs (_assert_local_loss_delta_crossing_coverage).
    for state_key, tensor in local_loss_delta_by_key.items():
        rows = _materialize_selector_rows_for_crossing_coverage(
            votes=votes_by_key[state_key],
            state=tensor_states[state_key],
        )
        crossing_eligible = set(crossing_eligible_flat_indices(rows))
        view = tensor.view(-1)
        for flat_index in range(int(view.numel())):
            if flat_index not in crossing_eligible:
                view[flat_index] = 0.0
            elif not torch.isfinite(view[flat_index]).item():
                raise ValueError(
                    "local_loss_delta_incomplete_candidate_coverage: "
                    f"unmeasured_crossing_eligible_row state_key={state_key!r} "
                    f"flat_index={int(flat_index)}"
                )


def _build_local_loss_delta_by_key_from_activation_credit_oracle(
    *,
    model: LMHead,
    batch: Mapping[str, torch.Tensor],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, BitLinear],
    device: torch.device,
    extras: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    max_abs_per_tensor: int,
    max_sampled_candidates: int,
    phase_progress: PhaseProgress | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    universe = _build_oracle_candidate_universe(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        max_abs_per_tensor=int(max_abs_per_tensor),
        extras=extras,
        max_sampled_candidates=int(max_sampled_candidates),
        phase_progress=phase_progress,
    )
    (
        sampled_candidates,
        _oracle_top,
        budget_exceeded,
        _elapsed,
        activation_credit_oracle_receipt,
    ) = _evaluate_sampled_candidates_for_activation_credit_oracle(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible_modules,
        device=device,
        extras=extras,
        votes_by_key=universe["votes_by_key"],
        candidate_by_id=universe["candidate_by_id"],
        sampled_ids=universe["sampled_ids"],
        baseline_loss=float(universe["baseline_loss"]),
        base_spec=universe["base_spec"],
        one_flip_spec=universe["one_flip_spec"],
        max_seconds=oracle_screen_budget_max_seconds(int(max_sampled_candidates)),
        phase_progress=phase_progress,
    )
    _assert_local_loss_delta_crossing_coverage(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        sampled_candidates=sampled_candidates,
        budget_exceeded=bool(budget_exceeded),
        candidate_count=len(universe["candidate_by_id"]),
        sampled_count=len(sampled_candidates),
        max_sampled_candidates=int(max_sampled_candidates),
    )
    local_loss_delta_by_key: dict[str, torch.Tensor] = {
        state_key: torch.full(votes.shape, float("nan"), dtype=torch.float32)
        for state_key, votes in votes_by_key.items()
    }
    for candidate in sampled_candidates:
        state_key = str(candidate["state_key"])
        flat_index = int(candidate["flat_index"])
        local_loss_delta_by_key[state_key].view(-1)[flat_index] = float(
            candidate["local_loss_delta"]
        )
    _zero_fill_non_crossing_unmeasured_local_loss_deltas(
        local_loss_delta_by_key=local_loss_delta_by_key,
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
    )
    return (
        {
            state_key: tensor.detach().cpu().contiguous()
            for state_key, tensor in local_loss_delta_by_key.items()
        },
        dict(activation_credit_oracle_receipt),
    )


def _plan_integer_vote_update_for_tier_a_surfaces(
    *,
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_mode: str,
    local_loss_delta_by_key: Mapping[str, torch.Tensor],
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
) -> dict[str, Any]:
    plans_by_key: dict[str, Any] = {}
    for state_key, state in sorted(tensor_states.items()):
        vu_state = state.vote_update_state()
        votes = votes_by_key[state_key].detach().cpu().to(torch.int16).contiguous()
        inputs = VoteUpdateInputs(
            votes=votes,
            replay_ce_veto_votes=_optional_step_vote_tensor(
                replay_ce_veto_votes_by_key,
                state_key,
                dtype=torch.int16,
            ),
            replay_ce_veto_moves=_optional_step_vote_tensor(
                replay_ce_veto_moves_by_key,
                state_key,
                dtype=torch.int8,
            ),
            pc_aux_votes=_optional_step_vote_tensor(
                pc_aux_votes_by_key,
                state_key,
                dtype=torch.int16,
            ),
            pc_aux_moves=_optional_step_vote_tensor(
                pc_aux_moves_by_key,
                state_key,
                dtype=torch.int8,
            ),
            pc_aux_mode=str(pc_aux_mode),
            local_loss_delta=local_loss_delta_by_key[state_key].detach().cpu().contiguous(),
        )
        plans_by_key[state_key] = plan_integer_vote_update_reference(
            vu_state,
            inputs,
            vote_specs_by_key[state_key],
            local_selection_ordering_mode=LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA,
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
            two_tier_carry_w6_enabled=True,
        )
    return plans_by_key


def _plan_integer_vote_update_for_control_arm_surfaces(
    *,
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_mode: str,
    local_selection_ordering_mode: str,
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
) -> dict[str, Any]:
    plans_by_key: dict[str, Any] = {}
    for state_key, state in sorted(tensor_states.items()):
        vu_state = state.vote_update_state()
        votes = votes_by_key[state_key].detach().cpu().to(torch.int16).contiguous()
        inputs = VoteUpdateInputs(
            votes=votes,
            replay_ce_veto_votes=_optional_step_vote_tensor(
                replay_ce_veto_votes_by_key,
                state_key,
                dtype=torch.int16,
            ),
            replay_ce_veto_moves=_optional_step_vote_tensor(
                replay_ce_veto_moves_by_key,
                state_key,
                dtype=torch.int8,
            ),
            pc_aux_votes=_optional_step_vote_tensor(
                pc_aux_votes_by_key,
                state_key,
                dtype=torch.int16,
            ),
            pc_aux_moves=_optional_step_vote_tensor(
                pc_aux_moves_by_key,
                state_key,
                dtype=torch.int8,
            ),
            pc_aux_mode=str(pc_aux_mode),
        )
        plans_by_key[state_key] = plan_vote_update_for_emit(
            vu_state,
            inputs,
            vote_specs_by_key[state_key],
            local_selection_ordering_mode=str(local_selection_ordering_mode),
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(local_selection_ordering_step),
            two_tier_carry_w6_enabled=False,
        )
    return plans_by_key


def _attach_control_arm_index_surfaces_to_compact(
    step_result_compact: Mapping[str, Any],
    *,
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_mode: str,
    local_selection_ordering_mode: str,
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
) -> dict[str, Any]:
    compact = dict(step_result_compact)
    tensor_stats = {
        state_key: dict(stats)
        for state_key, stats in dict(compact.get("tensor_stats", {})).items()
    }
    plans_by_key = _plan_integer_vote_update_for_control_arm_surfaces(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
        pc_aux_votes_by_key=pc_aux_votes_by_key,
        pc_aux_moves_by_key=pc_aux_moves_by_key,
        pc_aux_mode=str(pc_aux_mode),
        local_selection_ordering_mode=str(local_selection_ordering_mode),
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
    )
    global_rate_cap_enabled = bool(
        dict(compact.get("global_summary", {})).get("global_rate_cap_enabled")
    )
    for state_key, plan in sorted(plans_by_key.items()):
        stats = dict(tensor_stats[state_key])
        replay_ce_veto_indices = [
            int(value) for value in plan.replay_ce_veto_indices.detach().cpu().tolist()
        ]
        applied_indices = [
            int(value) for value in plan.applied_indices.detach().cpu().tolist()
        ]
        cap_enabled = bool(stats.get("global_rate_cap_enabled", global_rate_cap_enabled))
        _assert_tier_a_index_surface_count_consistency(
            state_key,
            tensor_stats=stats,
            replay_ce_veto_indices=replay_ce_veto_indices,
            applied_indices=applied_indices,
            global_rate_cap_enabled=cap_enabled,
        )
        stats["pre_veto_selected_indices"] = [
            int(value)
            for value in plan.pre_veto_selected_indices.detach().cpu().tolist()
        ]
        if cap_enabled:
            stats["post_veto_would_apply_pre_cap_indices"] = list(applied_indices)
            stats["applied_indices"] = [
                int(value) for value in stats["post_veto_applied_indices"]
            ]
        else:
            stats["applied_indices"] = applied_indices
        stats["replay_ce_veto_indices"] = replay_ce_veto_indices
        tensor_stats[state_key] = stats
    compact["tensor_stats"] = tensor_stats
    return compact


def _assert_tier_a_index_surface_count_consistency(
    state_key: str,
    *,
    tensor_stats: Mapping[str, Any],
    replay_ce_veto_indices: Sequence[int],
    applied_indices: Sequence[int],
    global_rate_cap_enabled: bool = False,
) -> None:
    stats = dict(tensor_stats)
    cap_enabled = bool(stats.get("global_rate_cap_enabled", global_rate_cap_enabled))
    if "replay_ce_veto_count" in stats:
        expected = int(stats["replay_ce_veto_count"])
        actual = len(replay_ce_veto_indices)
        if actual != expected:
            raise ValueError(
                "tier_a_staging_index_surface_replay_ce_veto_count_mismatch: "
                f"state_key={state_key!r}, replay_ce_veto_indices_len={actual}, "
                f"replay_ce_veto_count={expected}"
            )
    if cap_enabled:
        if "post_veto_would_apply_pre_cap_count" in stats:
            expected_pre_cap = int(stats["post_veto_would_apply_pre_cap_count"])
            actual_pre_cap = len(applied_indices)
            if actual_pre_cap != expected_pre_cap:
                raise ValueError(
                    "tier_a_staging_index_surface_post_veto_would_apply_pre_cap_count_mismatch: "
                    f"state_key={state_key!r}, applied_indices_len={actual_pre_cap}, "
                    f"post_veto_would_apply_pre_cap_count={expected_pre_cap}"
                )
        if "post_veto_applied_flip_count" in stats:
            expected_post_cap = int(stats["post_veto_applied_flip_count"])
            if "post_veto_applied_indices" not in stats:
                raise ValueError(
                    "tier_a_staging_index_surface_post_veto_applied_indices_missing: "
                    f"state_key={state_key!r}, post_veto_applied_flip_count={expected_post_cap}"
                )
            post_veto_applied_indices = stats["post_veto_applied_indices"]
            actual_post_cap = len(post_veto_applied_indices)
            if actual_post_cap != expected_post_cap:
                raise ValueError(
                    "tier_a_staging_index_surface_post_veto_applied_flip_count_mismatch: "
                    f"state_key={state_key!r}, post_veto_applied_indices_len={actual_post_cap}, "
                    f"post_veto_applied_flip_count={expected_post_cap}"
                )
    elif "post_veto_applied_flip_count" in stats:
        expected = int(stats["post_veto_applied_flip_count"])
        actual = len(applied_indices)
        if actual != expected:
            raise ValueError(
                "tier_a_staging_index_surface_post_veto_applied_flip_count_mismatch: "
                f"state_key={state_key!r}, applied_indices_len={actual}, "
                f"post_veto_applied_flip_count={expected}"
            )


def _attach_tier_a_staging_index_surfaces_to_compact(
    step_result_compact: Mapping[str, Any],
    *,
    tensor_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    replay_ce_veto_votes_by_key: Mapping[str, torch.Tensor] | None,
    replay_ce_veto_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_votes_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_moves_by_key: Mapping[str, torch.Tensor] | None,
    pc_aux_mode: str,
    local_loss_delta_by_key: Mapping[str, torch.Tensor],
    local_selection_ordering_seed: int,
    local_selection_ordering_step: int,
) -> dict[str, Any]:
    compact = dict(step_result_compact)
    tensor_stats = {
        state_key: dict(stats)
        for state_key, stats in dict(compact.get("tensor_stats", {})).items()
    }
    plans_by_key = _plan_integer_vote_update_for_tier_a_surfaces(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
        pc_aux_votes_by_key=pc_aux_votes_by_key,
        pc_aux_moves_by_key=pc_aux_moves_by_key,
        pc_aux_mode=str(pc_aux_mode),
        local_loss_delta_by_key=local_loss_delta_by_key,
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        local_selection_ordering_step=int(local_selection_ordering_step),
    )
    global_rate_cap_enabled = bool(
        dict(compact.get("global_summary", {})).get("global_rate_cap_enabled")
    )
    for state_key, plan in sorted(plans_by_key.items()):
        stats = dict(tensor_stats[state_key])
        replay_ce_veto_indices = [
            int(value) for value in plan.replay_ce_veto_indices.detach().cpu().tolist()
        ]
        applied_indices = [
            int(value) for value in plan.applied_indices.detach().cpu().tolist()
        ]
        cap_enabled = bool(stats.get("global_rate_cap_enabled", global_rate_cap_enabled))
        _assert_tier_a_index_surface_count_consistency(
            state_key,
            tensor_stats=stats,
            replay_ce_veto_indices=replay_ce_veto_indices,
            applied_indices=applied_indices,
            global_rate_cap_enabled=cap_enabled,
        )
        stats["pre_veto_selected_indices"] = [
            int(value) for value in plan.pre_veto_selected_indices.detach().cpu().tolist()
        ]
        if cap_enabled:
            stats["post_veto_would_apply_pre_cap_indices"] = list(applied_indices)
            stats["applied_indices"] = [
                int(value) for value in stats["post_veto_applied_indices"]
            ]
        else:
            stats["applied_indices"] = applied_indices
        stats["replay_ce_veto_indices"] = replay_ce_veto_indices
        tensor_stats[state_key] = stats
    compact["tensor_stats"] = tensor_stats
    return compact


def _strip_tier_a_probe_receipt_extensions(step_result_compact: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(step_result_compact)
    tensor_stats = {}
    for state_key, stats in dict(compact.get("tensor_stats", {})).items():
        stripped = {
            key: value
            for key, value in dict(stats).items()
            if key not in TIER_A_PROBE_RECEIPT_INDEX_SURFACE_KEYS
        }
        tensor_stats[state_key] = stripped
    compact["tensor_stats"] = tensor_stats
    return compact


def harness_wire_cpu_validation_self_check() -> dict[str, Any]:
    parser = build_arg_parser()
    flag_action = next(
        action
        for action in parser._actions
        if "--two-tier-carry-w6-enabled" in getattr(action, "option_strings", ())
    )
    assert flag_action.default is False
    assert _bounded_delta_vote_step_two_tier_kwargs(
        two_tier_carry_w6_enabled=False,
        local_loss_delta_by_key=None,
    ) == {}
    fixture_compact = {
        "schema": "hrm_text_158_c2p0_bounded_delta_step_result/v0.compact",
        "tensor_stats": {
            "toy.proj": {
                "replay_ce_veto_count": 0,
                "post_veto_applied_flip_count": 1,
                "q_changed_count": 1,
            }
        },
        "global_summary": {
            "global_rate_cap_enabled": False,
            "q_changed_count": 1,
            "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
        },
    }
    on_path = _attach_tier_a_staging_index_surfaces_to_compact(
        fixture_compact,
        tensor_states={
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0], dtype=torch.int8),
                0.5,
                torch.zeros(1, dtype=torch.int16),
            )
        },
        votes_by_key={"toy.proj": torch.tensor([12], dtype=torch.int16)},
        vote_specs_by_key={
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=1,
            )
        },
        replay_ce_veto_votes_by_key=None,
        replay_ce_veto_moves_by_key=None,
        pc_aux_votes_by_key=None,
        pc_aux_moves_by_key=None,
        pc_aux_mode="telemetry",
        local_loss_delta_by_key={"toy.proj": torch.tensor([-0.1], dtype=torch.float32)},
        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
        local_selection_ordering_step=1,
    )
    assert "pre_veto_selected_indices" in on_path["tensor_stats"]["toy.proj"]
    stripped = _strip_tier_a_probe_receipt_extensions(on_path)
    assert stripped == fixture_compact
    assert _strip_tier_a_probe_receipt_extensions(fixture_compact) == fixture_compact
    mismatch_fixture = {
        **fixture_compact,
        "tensor_stats": {
            "toy.proj": {
                **fixture_compact["tensor_stats"]["toy.proj"],
                "replay_ce_veto_count": 99,
            }
        },
    }
    mismatch_kwargs = {
        "tensor_states": {
            "toy.proj": make_bounded_tensor_state(
                "toy.proj",
                torch.tensor([0], dtype=torch.int8),
                0.5,
                torch.zeros(1, dtype=torch.int16),
            )
        },
        "votes_by_key": {"toy.proj": torch.tensor([12], dtype=torch.int16)},
        "vote_specs_by_key": {
            "toy.proj": VoteUpdateSpec(
                threshold_abs=10,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=1,
            )
        },
        "replay_ce_veto_votes_by_key": None,
        "replay_ce_veto_moves_by_key": None,
        "pc_aux_votes_by_key": None,
        "pc_aux_moves_by_key": None,
        "pc_aux_mode": "telemetry",
        "local_loss_delta_by_key": {"toy.proj": torch.tensor([-0.1], dtype=torch.float32)},
        "local_selection_ordering_seed": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
        "local_selection_ordering_step": 1,
    }
    try:
        _attach_tier_a_staging_index_surfaces_to_compact(
            mismatch_fixture,
            **mismatch_kwargs,
        )
    except ValueError as exc:
        assert "tier_a_staging_index_surface_replay_ce_veto_count_mismatch" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for mismatched tier_a replay_ce_veto_count"
        )
    coverage_state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(2, dtype=torch.int16),
    )
    try:
        _assert_local_loss_delta_crossing_coverage(
            tensor_states={"toy.proj": coverage_state},
            votes_by_key={"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
            sampled_candidates=[
                {
                    "state_key": "toy.proj",
                    "flat_index": 0,
                    "local_loss_delta": -0.1,
                }
            ],
            budget_exceeded=False,
            candidate_count=2,
            sampled_count=1,
            max_sampled_candidates=1,
        )
    except ValueError as exc:
        assert "local_loss_delta_incomplete_candidate_coverage" in str(exc)
        assert "toy.proj:1" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for incomplete crossing-eligible coverage"
        )
    try:
        _assert_local_loss_delta_crossing_coverage(
            tensor_states={"toy.proj": coverage_state},
            votes_by_key={"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
            sampled_candidates=[
                {
                    "state_key": "toy.proj",
                    "flat_index": 0,
                    "local_loss_delta": -0.1,
                },
                {
                    "state_key": "toy.proj",
                    "flat_index": 1,
                    "local_loss_delta": -0.2,
                },
            ],
            budget_exceeded=True,
            candidate_count=2,
            sampled_count=2,
            max_sampled_candidates=1,
        )
    except ValueError as exc:
        assert "local_loss_delta_incomplete_candidate_coverage" in str(exc)
        assert "oracle_screen_budget_exceeded" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for oracle_screen_budget_exceeded"
        )
    _assert_local_loss_delta_crossing_coverage(
        tensor_states={"toy.proj": coverage_state},
        votes_by_key={"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
        sampled_candidates=[
            {
                "state_key": "toy.proj",
                "flat_index": 0,
                "local_loss_delta": -0.1,
            },
            {
                "state_key": "toy.proj",
                "flat_index": 1,
                "local_loss_delta": -0.2,
            },
        ],
        budget_exceeded=False,
        candidate_count=2,
        sampled_count=2,
        max_sampled_candidates=2,
    )
    _assert_local_loss_delta_crossing_coverage(
        tensor_states={"toy.proj": coverage_state},
        votes_by_key={"toy.proj": torch.tensor([12, 1], dtype=torch.int16)},
        sampled_candidates=[
            {
                "state_key": "toy.proj",
                "flat_index": 0,
                "local_loss_delta": -0.1,
            }
        ],
        budget_exceeded=False,
        candidate_count=2,
        sampled_count=1,
        max_sampled_candidates=1,
    )
    non_crossing_delta_by_key = {
        "toy.proj": torch.full((2,), float("nan"), dtype=torch.float32)
    }
    non_crossing_delta_by_key["toy.proj"][0] = -0.1
    _zero_fill_non_crossing_unmeasured_local_loss_deltas(
        local_loss_delta_by_key=non_crossing_delta_by_key,
        tensor_states={"toy.proj": coverage_state},
        votes_by_key={"toy.proj": torch.tensor([12, 1], dtype=torch.int16)},
    )
    assert torch.isclose(
        non_crossing_delta_by_key["toy.proj"][0],
        torch.tensor(-0.1, dtype=torch.float32),
    ).item()
    assert float(non_crossing_delta_by_key["toy.proj"][1].item()) == 0.0
    proxy_delta_by_key = {
        "toy.proj": torch.full((2,), float("nan"), dtype=torch.float32)
    }
    proxy_delta_by_key["toy.proj"][0] = -0.1
    try:
        _assert_local_loss_delta_proxy_crossing_coverage(
            local_loss_delta_by_key=proxy_delta_by_key,
            tensor_states={"toy.proj": coverage_state},
            votes_by_key={"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
        )
    except ValueError as exc:
        assert "local_loss_delta_proxy_incomplete_coverage" in str(exc)
        assert "toy.proj:1" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for incomplete proxy crossing-eligible coverage"
        )
    proxy_delta_by_key["toy.proj"][1] = -0.2
    _assert_local_loss_delta_proxy_crossing_coverage(
        local_loss_delta_by_key=proxy_delta_by_key,
        tensor_states={"toy.proj": coverage_state},
        votes_by_key={"toy.proj": torch.tensor([12, 12], dtype=torch.int16)},
    )
    return {
        "argparse_default_off": True,
        "off_path_kwargs_empty": True,
        "on_only_receipt_extensions_stripped_to_fixture": True,
        "tier_a_index_surface_count_consistency_fail_closed": True,
        "local_loss_delta_crossing_coverage_fail_closed": True,
        "local_loss_delta_full_crossing_coverage_proceeds": True,
        "local_loss_delta_non_crossing_unmeasured_proceeds": True,
        "local_loss_delta_proxy_crossing_coverage_fail_closed": True,
        "local_loss_delta_proxy_full_crossing_coverage_proceeds": True,
        "tier_a_index_surface_keys": sorted(TIER_A_PROBE_RECEIPT_INDEX_SURFACE_KEYS),
    }


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
    b2_retained_support_sets: Mapping[str, Mapping[str, Any]] | None = None,
    b2_parent_model: LMHead | None = None,
    b2_parent_consistency_weight: float = 0.0,
    b2_pc_aux_mode: str = "telemetry",
    audit_callback: Callable[[int, Mapping[str, Any]], dict[str, Any]] | None = None,
    audit_interval: int = 0,
    stop_on_strict_exact: bool = False,
    matched_continued_training_horizon_steps: int = 0,
    global_cap_contract: str = GLOBAL_CAP_CONTRACT_OFF,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    science_arm: str = ARM_A0_RANK_BUCKET_CURRENT,
    b2_full_verdict_mode: bool = False,
    b2_full_prior_snapshot_callback: Callable[
        [int, Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, Any]]],
        dict[str, Any],
    ] | None = None,
    b2_full_audit_export_callback: Callable[
        [
            int,
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Mapping[str, Any]],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
        str | None,
    ] | None = None,
    front_c_identity_collector: FrontCLiveIdentityCollector | None = None,
    phase_progress: PhaseProgress | None = None,
    b2b_sequential_capture_enabled: bool = False,
    b2b_sequential_capture_out: Path | None = None,
    b2b_sequential_min_steps_for_verdict: int = 50,
    b2b_sequential_max_sampled_candidates: int = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    two_tier_carry_w6_enabled: bool = False,
    oracle_screen_max_sampled_candidates: int = ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    phase: str = "c2p1-real-model-smoke",
    snapshot_mode: str = SNAPSHOT_MODE_FULL,
    headroom_wiring_sidecar_path: Path | None = None,
    r7_cap_defer_pressure_instrumentation_enabled: bool = False,
    r7_deferred_backlog_carry_enabled: bool = False,
    r7_cap_defer_pressure_sidecar_path: Path | None = None,
    d_recompute_window_instrumentation_enabled: bool = False,
    d_recompute_window_log_path: Path | None = None,
    d_recompute_selector_manifest: StratifiedSelectorManifest | None = None,
    event_coded_recompute_window_log_enabled: bool = False,
    d_live_carrier_snapshot_enabled: bool = False,
    d_live_carrier_snapshot_path: Path | None = None,
    receipt_emit_profile: str = RECEIPT_EMIT_PROFILE_FULL,
    d_diagnostic_compact_step_reports: bool = False,
    calibration_warmup_collector: CalibrationWarmupCollector | None = None,
    votes_emit_enabled: bool = False,
    votes_emit_root: Path | None = None,
    carrier_growth_enabled: bool = False,
    confirmation_envelope: str | None = None,
    event_coded_sparse_vote_authority: bool = False,
    vote_update_decay_numerator: int | None = None,
    vote_update_decay_denominator: int | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    int,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[int],
]:
    model.train()
    if str(science_arm) not in SCIENCE_ARM_CHOICES:
        raise ValueError(f"science_arm must be one of {SCIENCE_ARM_CHOICES}, got {science_arm!r}")
    progress = phase_progress or PhaseProgress(enabled=False, device=device)
    envelope = resolve_confirmation_envelope(confirmation_envelope)
    vote_spec = resolve_probe_vote_update_spec(
        max_abs_per_tensor=int(max_abs_per_tensor),
        confirmation_envelope=confirmation_envelope,
        vote_update_decay_numerator=vote_update_decay_numerator,
        vote_update_decay_denominator=vote_update_decay_denominator,
    )
    rank_spec = (
        envelope.rank_spec if envelope is not None else default_dry_run_rank_vote_spec()
    )
    vote_specs = {key: vote_spec for key in tensor_states}
    updater_config = {
        "rank_vote_spec": rank_spec.to_live_dict(),
        "vote_update_spec": asdict(vote_spec),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": _science_arm_vote_law(str(science_arm)),
        "science_arm": str(science_arm),
        "target_vote_law": _science_arm_vote_law(str(science_arm)),
        "target_tie_policy_id": _science_arm_tie_policy(str(science_arm)),
        "local_selection_ordering_mode": _science_local_selection_ordering_mode(str(science_arm)),
        "local_selection_ordering_seed": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
        "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
        "default_rank_bucket_path_unchanged": str(science_arm) == ARM_A0_RANK_BUCKET_CURRENT,
    }
    optimizer, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model,
        eligible_modules,
        lr=0.0,
        weight_decay=0.0,
    )
    states = dict(tensor_states)
    votes_emit_collector = None
    if bool(votes_emit_enabled):
        if votes_emit_root is None:
            raise ValueError("votes_emit_enabled requires votes_emit_root")
        from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
            VotesEmitCollector,
        )

        votes_emit_collector = VotesEmitCollector(Path(votes_emit_root))
    carrier_growth_collector = None
    if bool(carrier_growth_enabled):
        if not bool(votes_emit_enabled) or votes_emit_root is None:
            raise ValueError(
                "carrier_growth_enabled requires votes_emit_enabled and votes_emit_root"
            )
        from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
            CarrierGrowthCollector,
        )

        carrier_growth_collector = CarrierGrowthCollector(Path(votes_emit_root))
    step_reports: dict[str, Any] = {}
    audit_reports: dict[str, Any] = {}
    grad_proxy_ingress_crossing_eligible_count_by_step: list[int] = []
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
    retained_support_sets = dict(b2_retained_support_sets or {})
    retained_coverage: dict[str, set[str]] = {
        support: set()
        for support in retained_support_sets
    }
    b2_full_coverage_tracker = (
        new_b2_full_coverage_tracker(
            {
                support: int(support_set["proof"]["expected_count"])
                for support, support_set in retained_support_sets.items()
            }
        )
        if b2_full_verdict_mode and retained_support_sets
        else None
    )
    b2_full_verdict_state = (
        new_b2_full_verdict_state()
        if b2_full_verdict_mode
        else None
    )
    if retained_support_sets and float(b2_parent_consistency_weight) > 0.0 and b2_parent_model is None:
        raise ValueError("B2 parent-consistency aux requires a frozen parent model")
    first_strict_exact_step: int | None = None
    b2b_trace_hashes: list[str] = []
    if b2b_sequential_capture_enabled:
        if b2b_sequential_capture_out is None:
            raise ValueError(
                "b2b sequential capture requires --b2b-sequential-capture-out"
            )
        if (
            int(b2b_sequential_max_sampled_candidates)
            != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
        ):
            raise ValueError(
                "b2b sequential capture requires max_sampled_candidates == 32"
            )
        _init_b2b_sequential_trace_capture(Path(b2b_sequential_capture_out))

    def maybe_audit(step: int, *, final: bool = False) -> str | None:
        nonlocal first_strict_exact_step
        if audit_callback is None:
            return None
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
        if b2_full_verdict_mode:
            if b2_full_verdict_state is None:
                raise RuntimeError("B2-full verdict state was not initialized")
            coverage_by_support = (
                snapshot_b2_full_coverage_tracker(b2_full_coverage_tracker)
                if b2_full_coverage_tracker is not None
                else {}
            )
            audit_report["b2_full_coverage_by_support"] = coverage_by_support
            snapshot_names = b2_full_required_snapshot_names(
                b2_full_verdict_state,
                target_audit=audit_report,
                coverage_by_support=coverage_by_support,
            )
            if snapshot_names:
                if b2_full_prior_snapshot_callback is None:
                    raise RuntimeError("B2-full verdict mode requires a prior snapshot callback")
                with progress.phase("b2_full_prior_snapshot", step=int(step)):
                    prior_snapshot = b2_full_prior_snapshot_callback(
                        int(step),
                        states,
                        audit_report,
                        coverage_by_support,
                    )
                record_b2_full_prior_snapshot(
                    b2_full_verdict_state,
                    snapshot_names=snapshot_names,
                    snapshot=prior_snapshot,
                )
            b2_full_verdict_state["coverage_by_support"] = coverage_by_support
            b2_full_verdict_state["math_a0_coverage_cycles"] = b2_full_coverage_cycles(
                coverage_by_support,
                "math_a0",
            )
            b2_full_verdict_state["l0b_coverage_cycles"] = b2_full_coverage_cycles(
                coverage_by_support,
                "L0b",
            )
            if b2_full_audit_export_callback is not None:
                with progress.phase("b2_full_audit_export", step=int(step)):
                    export_path = b2_full_audit_export_callback(
                        int(step),
                        states,
                        audit_report,
                        coverage_by_support,
                        b2_full_verdict_state,
                        updater_config,
                    )
                if export_path:
                    b2_full_verdict_state.setdefault("audit_export_paths", {})[
                        str(step)
                    ] = str(export_path)
            if b2_full_verdict_state.get("combined_stop", {}).get("triggered"):
                return "b2_full_target_coverage_retain_stop"
            return None
        first_strict_exact_step, stop_token = update_strict_exact_stop_state(
            step=int(step),
            audit_report=audit_report,
            stop_on_strict_exact=bool(stop_on_strict_exact),
            matched_continued_training_horizon_steps=int(
                matched_continued_training_horizon_steps
            ),
            first_strict_exact_step=first_strict_exact_step,
        )
        return stop_token

    if front_c_identity_collector is not None:
        with progress.phase("front_c_identity_step0", step=0):
            front_c_identity_collector.record_step0(states)

    initial_stop = maybe_audit(0)
    if initial_stop:
        return (
            step_reports,
            updater_config,
            states,
            audit_reports,
            "strict_exact_acquired_step0"
            if initial_stop == "strict_exact_acquired"
            else initial_stop,
            0,
            b2_full_verdict_state,
            None,
            grad_proxy_ingress_crossing_eligible_count_by_step,
        )
    if steps <= 0:
        return (
            step_reports,
            updater_config,
            states,
            audit_reports,
            "no_steps",
            0,
            b2_full_verdict_state,
            None,
            grad_proxy_ingress_crossing_eligible_count_by_step,
        )

    stop_reason = "max_steps_completed"
    steps_completed = 0
    prior_applied_by_state_key: dict[str, list[int]] = {}
    carry_backlog: dict[str, dict[int, dict[str, int]]] | None = None
    prior_pressure_mass: int | None = None
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
                weighted_grads, loss, metrics = _compute_ce_weighted_grads(
                    model,
                    step_batch,
                    states,
                    eligible_modules,
                    device=device,
                    extras=extras,
                )
            b2_step_receipt = _empty_b2_step_aux_receipt()
            replay_ce_veto_votes_by_key = None
            replay_ce_veto_moves_by_key = None
            pc_aux_votes_by_key = None
            pc_aux_moves_by_key = None
            aux_weighted_grad_finite = True
            if retained_support_sets:
                replay_grad_sums = _zero_weighted_grad_sums(states)
                pc_grad_sums = _zero_weighted_grad_sums(states)
                support_receipts: list[dict[str, Any]] = []
                pc_aux_generated = False
                with progress.phase("b2_retained_aux", step=int(step)):
                    for support, support_set in sorted(retained_support_sets.items()):
                        support_batches_for_name = list(support_set["batches"])
                        support_batch_item = support_batches_for_name[
                            (step - 1) % len(support_batches_for_name)
                        ]
                        support_batch = support_batch_item["batch"]
                        support_metadata = dict(support_batch_item["metadata"])
                        row_ids = [str(row_id) for row_id in support_metadata.get("row_ids", [])]
                        if b2_full_coverage_tracker is not None:
                            update_b2_full_coverage_tracker(
                                b2_full_coverage_tracker,
                                support=support,
                                row_ids=row_ids,
                            )
                        else:
                            retained_coverage[support].update(row_ids)
                        ce_grads, ce_loss, ce_metrics = _compute_ce_weighted_grads(
                            model,
                            support_batch,
                            states,
                            eligible_modules,
                            device=device,
                            extras=extras,
                        )
                        _add_weighted_grads_in_place(replay_grad_sums, ce_grads)
                        ce_finite = bool(torch.isfinite(ce_loss).item())
                        ce_grad_finite = _weighted_grads_all_finite(ce_grads)
                        pc_kl_value = None
                        pc_finite = True
                        pc_grad_finite = True
                        if float(b2_parent_consistency_weight) > 0.0:
                            assert b2_parent_model is not None
                            pc_grads, pc_kl = _compute_pc_weighted_grads(
                                model,
                                b2_parent_model,
                                support_batch,
                                states,
                                eligible_modules,
                                device=device,
                                extras=extras,
                                weight=float(b2_parent_consistency_weight),
                            )
                            _add_weighted_grads_in_place(pc_grad_sums, pc_grads)
                            pc_kl_value = float(pc_kl.cpu().item())
                            pc_finite = bool(torch.isfinite(pc_kl).item())
                            pc_grad_finite = _weighted_grads_all_finite(pc_grads)
                            pc_aux_generated = True
                        aux_weighted_grad_finite = (
                            aux_weighted_grad_finite
                            and ce_finite
                            and ce_grad_finite
                            and pc_finite
                            and pc_grad_finite
                        )
                        support_receipts.append(
                            {
                                "support": support,
                                "batch_metadata": support_metadata,
                                "replay_ce_loss": float(ce_loss.cpu().item()),
                                "replay_ce_loss_finite": ce_finite,
                                "replay_ce_weighted_grad_finite": ce_grad_finite,
                                "replay_ce_metrics": _metrics_to_dict(ce_metrics),
                                "pc_kl": pc_kl_value,
                                "pc_kl_finite": pc_finite,
                                "pc_weighted_grad_finite": pc_grad_finite,
                            }
                        )
                replay_ce_veto_votes_by_key, replay_ce_veto_moves_by_key = _weighted_grads_to_vote_aux_maps(
                    replay_grad_sums,
                    states,
                    rank_spec,
                )
                if pc_aux_generated:
                    pc_aux_votes_by_key, pc_aux_moves_by_key = _weighted_grads_to_vote_aux_maps(
                        pc_grad_sums,
                        states,
                        rank_spec,
                    )
                if b2_full_coverage_tracker is not None:
                    coverage_by_support = snapshot_b2_full_coverage_tracker(
                        b2_full_coverage_tracker
                    )
                else:
                    coverage_by_support = {
                        support: {
                            "rows_seen": len(retained_coverage[support]),
                            "rows_total": int(support_set["proof"]["expected_count"]),
                            "coverage_cycle_complete": (
                                len(retained_coverage[support])
                                >= int(support_set["proof"]["expected_count"])
                            ),
                        }
                        for support, support_set in sorted(retained_support_sets.items())
                    }
                b2_step_receipt = {
                    "enabled": True,
                    "support_batches": support_receipts,
                    "coverage_by_support": coverage_by_support,
                    "replay_ce_veto_generated": True,
                    "pc_aux_generated": pc_aux_generated,
                    "pc_aux_mode": str(b2_pc_aux_mode),
                    "parent_consistency_weight": float(b2_parent_consistency_weight),
                    "target_parent_kl": False,
                    "target_rows_excluded_from_pc": True,
                }
            with progress.phase("step_update", step=int(step)):
                sparse_events_by_key: dict[str, Any] = {}
                with progress.phase("sparse_vote_construction", step=int(step)):
                    votes_by_key, vote_pressure_by_key, finite_weighted_grad = _weighted_grads_to_science_arm_votes(
                        weighted_grads,
                        states,
                        rank_spec=rank_spec,
                        vote_spec=vote_spec,
                        science_arm=str(science_arm),
                        sparse_events_out=sparse_events_by_key,
                        sparse_construction_only=bool(event_coded_sparse_vote_authority),
                    )
                front_c_identity_observer = make_front_c_identity_observer_for_step(
                    front_c_identity_collector,
                    step=int(step),
                    total_steps=int(steps),
                )
                global_cap_spec = resolve_named_global_cap_spec(
                    str(global_cap_contract),
                    step=int(step),
                )
                effective_global_cap_spec = _science_global_cap_spec_for_arm(
                    global_cap_spec,
                    science_arm=str(science_arm),
                )

                b2b_step_capture: dict[str, Any] | None = None
                if b2b_sequential_capture_enabled:
                    with progress.phase("b2b_sequential_capture", step=int(step)):
                        b2b_step_capture = capture_b2b_sequential_pre_update_step(
                            model=model,
                            batch=step_batch,
                            tensor_states=states,
                            eligible_modules=eligible_modules,
                            device=device,
                            extras=extras,
                            votes_by_key=votes_by_key,
                            baseline_loss=float(loss.detach().cpu().item()),
                            optimizer_step_index=int(step),
                            max_abs_per_tensor=int(max_abs_per_tensor),
                            max_sampled_candidates=int(
                                b2b_sequential_max_sampled_candidates
                            ),
                            max_seconds=oracle_screen_budget_max_seconds(
                                int(b2b_sequential_max_sampled_candidates)
                            ),
                            source_kind=SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
                            phase_progress=progress,
                        )

                pre_apply_states = states
                local_loss_delta_by_key = None
                grad_proxy_ingress_receipt: dict[str, Any] | None = None
                proxy_oracle_drift_receipt: dict[str, Any] | None = None
                step_cuda_memory_snapshots: list[dict[str, Any]] = []
                if two_tier_carry_w6_enabled:
                    crossing_eligible_count = count_w6_t10_crossing_eligible_from_votes(
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                    )
                    crossing_count_by_state_key = crossing_count_by_state_key_from_votes(
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                    )
                    grad_proxy_ingress_crossing_eligible_count_by_step.append(
                        int(crossing_eligible_count)
                    )
                    if device.type == "cuda":
                        step_cuda_memory_snapshots.append(
                            capture_cuda_phase_memory_snapshot(
                                device,
                                label="pre_two_tier_grad_proxy_ingress",
                                optimizer_step_index=int(step),
                            )
                        )
                    with progress.phase(
                        "two_tier_grad_proxy_ingress",
                        step=int(step),
                        grad_proxy_ingress_crossing_eligible_count=int(
                            crossing_eligible_count
                        ),
                        crossing_count_by_state_key=dict(crossing_count_by_state_key),
                    ):
                        (
                            local_loss_delta_by_key,
                            grad_proxy_ingress_receipt,
                        ) = build_grad_proxy_local_loss_delta_by_key(
                            model=model,
                            batch=step_batch,
                            tensor_states=pre_apply_states,
                            eligible_modules=eligible_modules,
                            device=device,
                            extras=extras,
                            votes_by_key=votes_by_key,
                            max_abs_per_tensor=int(max_abs_per_tensor),
                            population_mode=POPULATION_MODE_FULL_CROSSING_ELIGIBLE,
                            phase_progress=progress,
                            optimizer_step_index=int(step),
                        )
                    if device.type == "cuda":
                        step_cuda_memory_snapshots.append(
                            capture_cuda_phase_memory_snapshot(
                                device,
                                label="post_two_tier_grad_proxy_ingress",
                                optimizer_step_index=int(step),
                            )
                        )
                    if grad_proxy_ingress_receipt is not None:
                        step_cuda_memory_snapshots.extend(
                            list(
                                grad_proxy_ingress_receipt.get("cuda_memory_snapshots")
                                or []
                            )
                        )
                    if int(step) % int(DRIFT_AUDIT_STEP_INTERVAL) == 0:
                        if device.type == "cuda":
                            step_cuda_memory_snapshots.append(
                                capture_cuda_phase_memory_snapshot(
                                    device,
                                    label="pre_proxy_oracle_drift_audit",
                                    optimizer_step_index=int(step),
                                )
                            )
                        with progress.phase("proxy_oracle_drift_audit", step=int(step)):
                            proxy_oracle_drift_receipt = run_proxy_oracle_drift_audit(
                                model=model,
                                batch=step_batch,
                                tensor_states=pre_apply_states,
                                eligible_modules=eligible_modules,
                                device=device,
                                extras=extras,
                                votes_by_key=votes_by_key,
                                local_loss_delta_by_key=local_loss_delta_by_key,
                                baseline_loss=float(loss.detach().cpu().item()),
                                max_abs_per_tensor=int(max_abs_per_tensor),
                                optimizer_step_index=int(step),
                            )
                        if device.type == "cuda":
                            step_cuda_memory_snapshots.append(
                                capture_cuda_phase_memory_snapshot(
                                    device,
                                    label="post_proxy_oracle_drift_audit",
                                    optimizer_step_index=int(step),
                                )
                            )
                two_tier_vote_step_kwargs = _bounded_delta_vote_step_two_tier_kwargs(
                    two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
                    local_loss_delta_by_key=local_loss_delta_by_key,
                )
                step_selection_ordering_mode = (
                    LOCAL_SELECTION_ORDER_TRANSIENT_LOCAL_LOSS_DELTA
                    if two_tier_carry_w6_enabled
                    else _science_local_selection_ordering_mode(str(science_arm))
                )
                s3bb_boundary_step_report: dict[str, Any] = {}

                def _materialize_bounded_delta_vote_step() -> Any:
                    apply_votes_by_key = (
                        None
                        if bool(event_coded_sparse_vote_authority)
                        else votes_by_key
                    )
                    return apply_bounded_delta_vote_step(
                        states,
                        apply_votes_by_key,
                        vote_specs,
                        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
                        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
                        pc_aux_votes_by_key=pc_aux_votes_by_key,
                        pc_aux_moves_by_key=pc_aux_moves_by_key,
                        pc_aux_mode=str(b2_pc_aux_mode),
                        global_cap_spec=effective_global_cap_spec,
                        global_cap_tie_rule_mode=str(tie_rule_mode),
                        global_cap_contract_name=(
                            str(global_cap_contract)
                            if effective_global_cap_spec is not None
                            else None
                        ),
                        local_selection_ordering_mode=step_selection_ordering_mode,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                        local_selection_ordering_step=int(step),
                        front_c_identity_observer=front_c_identity_observer,
                        candidate_sparse_vote_events_by_key=sparse_events_by_key,
                        event_coded_sparse_vote_authority=bool(
                            event_coded_sparse_vote_authority
                        ),
                        sparse_cap_submilestone_emit=(
                            progress.milestone_emitter
                            if bool(event_coded_sparse_vote_authority)
                            and progress.milestone_emitter is not None
                            else None
                        ),
                        host_rss_subphase_emit=progress.make_host_rss_subphase_emitter(
                            step=int(step),
                        ),
                        **two_tier_vote_step_kwargs,
                        **resolve_r7_deferred_backlog_vote_step_kwargs(
                            r7_deferred_backlog_carry_enabled=bool(
                                r7_deferred_backlog_carry_enabled
                            ),
                            carry_backlog=carry_backlog,
                        ),
                    )

                def _invoke_bounded_delta_vote_step_materialize() -> Any:
                    if bool(event_coded_sparse_vote_authority):
                        with progress.phase("sparse_cap_apply", step=int(step)):
                            return _materialize_bounded_delta_vote_step()
                    return _materialize_bounded_delta_vote_step()

                if votes_emit_collector is not None:
                    from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
                        maybe_emit_votes_step_record,
                    )

                    maybe_emit_votes_step_record(
                        root=Path(votes_emit_root),
                        enabled=True,
                        optimizer_step_index=int(step),
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        vote_specs_by_key=vote_specs,
                        max_abs_per_tensor=int(max_abs_per_tensor),
                        collector=votes_emit_collector,
                        two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
                        local_loss_delta_by_key=local_loss_delta_by_key,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                    )

                if str(phase) == S3BB_W6_HEADROOM_DIAGNOSTIC_PHASE:
                    materialization = run_vote_materialization_with_s3bb_boundary_catch(
                        phase=str(phase),
                        step_report=s3bb_boundary_step_report,
                        materialize=_invoke_bounded_delta_vote_step_materialize,
                    )
                    if materialization.terminated:
                        breach_duration_seconds = _timing_duration_seconds(
                            step_timing_start,
                            device,
                        )
                        step_reports[str(step)] = {
                            "loss": float(loss.detach().cpu().item()),
                            "loss_finite": bool(torch.isfinite(loss).item()),
                            "weighted_grad_finite": bool(finite_weighted_grad),
                            "aux_weighted_grad_finite": bool(aux_weighted_grad_finite),
                            "duration_seconds": breach_duration_seconds,
                            "metrics": _metrics_to_dict(metrics),
                            "bp_steps": int(extras["bp_steps"]),
                            "q_changed_count": 0,
                            "science_arm": str(science_arm),
                            "target_vote_law": _science_arm_vote_law(str(science_arm)),
                            "target_tie_policy_id": _science_arm_tie_policy(str(science_arm)),
                            "local_selection_ordering_mode": step_selection_ordering_mode,
                            "local_selection_ordering_seed": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                            "local_selection_ordering_step": int(step),
                            "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
                            "vote_pressure": vote_pressure_by_key,
                            "support_batch": dict(step_batch_metadata),
                            "b2_retained_support": b2_step_receipt,
                            "step_result": {"headroom_breach": True},
                            "optimizer_identity_proof": {"headroom_breach": True},
                        }
                        attach_s3bb_headroom_telemetry_to_step_report(
                            step_reports[str(step)],
                            phase=str(phase),
                            post_update_states=pre_apply_states,
                            snapshot_mode=str(snapshot_mode),
                            headroom_wiring_sidecar_path=headroom_wiring_sidecar_path,
                            step=int(step),
                        )
                        caught_telemetry = s3bb_boundary_step_report.get(
                            "headroom_telemetry"
                        ) or {}
                        if bool(caught_telemetry.get("boundary_value_error_caught")):
                            telemetry = step_reports[str(step)]["headroom_telemetry"]
                            telemetry["boundary_value_error_caught"] = True
                            telemetry["would_strict_raise_step"] = True
                            telemetry["strict_raise_count"] = 1
                        stop_reason = str(
                            materialization.stop_reason or "headroom_breach"
                        )
                        steps_completed = step
                        break
                    step_result = materialization.value
                else:
                    step_result = _invoke_bounded_delta_vote_step_materialize()
                if (
                    bool(event_coded_sparse_vote_authority)
                    and progress.milestone_emitter is not None
                ):
                    progress.milestone_emitter.ensure_sparse_cap_subphase_contract(
                        optimizer_step_index=int(step),
                    )
                states = step_result.tensor_states
                if carrier_growth_collector is not None:
                    from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
                        maybe_emit_carrier_growth_step_record,
                    )

                    maybe_emit_carrier_growth_step_record(
                        enabled=True,
                        collector=carrier_growth_collector,
                        optimizer_step_index=int(step),
                        tensor_states=states,
                        votes_by_key=votes_by_key,
                        tensor_stats_by_key=step_result.tensor_stats,
                    )
                if r7_deferred_backlog_carry_enabled:
                    carry_backlog = step_result.deferred_backlog
                q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
                if b2b_sequential_capture_enabled and b2b_step_capture is not None:
                    assert b2b_sequential_capture_out is not None
                    trace_record = dict(b2b_step_capture)
                    trace_record["post_update_telemetry"] = {
                        "q_changed_count": q_changed_count,
                    }
                    _append_b2b_sequential_trace_step(
                        Path(b2b_sequential_capture_out),
                        trace_record,
                    )
                    b2b_trace_hashes.append(str(b2b_step_capture["source_table_hash"]))
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
                step_result_compact = step_result.to_compact_dict()
                if two_tier_carry_w6_enabled:
                    assert local_loss_delta_by_key is not None
                    tier_a_plans_by_key = _plan_integer_vote_update_for_tier_a_surfaces(
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        vote_specs_by_key=vote_specs,
                        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
                        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
                        pc_aux_votes_by_key=pc_aux_votes_by_key,
                        pc_aux_moves_by_key=pc_aux_moves_by_key,
                        pc_aux_mode=str(b2_pc_aux_mode),
                        local_loss_delta_by_key=local_loss_delta_by_key,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                        local_selection_ordering_step=int(step),
                    )
                    step_result_compact = _attach_tier_a_staging_index_surfaces_to_compact(
                        step_result_compact,
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        vote_specs_by_key=vote_specs,
                        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
                        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
                        pc_aux_votes_by_key=pc_aux_votes_by_key,
                        pc_aux_moves_by_key=pc_aux_moves_by_key,
                        pc_aux_mode=str(b2_pc_aux_mode),
                        local_loss_delta_by_key=local_loss_delta_by_key,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                        local_selection_ordering_step=int(step),
                    )
                    step_result_compact = _attach_cap_window_audit_surfaces(
                        step_result_compact,
                        plans_by_key=tier_a_plans_by_key,
                        prior_applied_by_state_key=prior_applied_by_state_key,
                        local_loss_delta_by_key=local_loss_delta_by_key,
                        optimizer_step_index=int(step),
                    )
                elif not bool(event_coded_sparse_vote_authority):
                    control_plans_by_key = _plan_integer_vote_update_for_control_arm_surfaces(
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        vote_specs_by_key=vote_specs,
                        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
                        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
                        pc_aux_votes_by_key=pc_aux_votes_by_key,
                        pc_aux_moves_by_key=pc_aux_moves_by_key,
                        pc_aux_mode=str(b2_pc_aux_mode),
                        local_selection_ordering_mode=step_selection_ordering_mode,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                        local_selection_ordering_step=int(step),
                    )
                    step_result_compact = _attach_control_arm_index_surfaces_to_compact(
                        step_result_compact,
                        tensor_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        vote_specs_by_key=vote_specs,
                        replay_ce_veto_votes_by_key=replay_ce_veto_votes_by_key,
                        replay_ce_veto_moves_by_key=replay_ce_veto_moves_by_key,
                        pc_aux_votes_by_key=pc_aux_votes_by_key,
                        pc_aux_moves_by_key=pc_aux_moves_by_key,
                        pc_aux_mode=str(b2_pc_aux_mode),
                        local_selection_ordering_mode=step_selection_ordering_mode,
                        local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                        local_selection_ordering_step=int(step),
                    )
                    step_result_compact = _attach_cap_window_audit_surfaces(
                        step_result_compact,
                        plans_by_key=control_plans_by_key,
                        prior_applied_by_state_key=prior_applied_by_state_key,
                        local_loss_delta_by_key=None,
                        optimizer_step_index=int(step),
                    )
                else:
                    step_result_compact = dict(step_result_compact)
                    step_result_compact[
                        "control_arm_index_surfaces_skipped_sparse_authority"
                    ] = True
                _update_prior_applied_by_state_key(
                    prior_applied_by_state_key,
                    step_result_compact,
                )
                if should_apply_d_diagnostic_receipt_compaction(
                    phase=str(phase),
                    receipt_emit_profile=str(receipt_emit_profile),
                    d_diagnostic_compact_step_reports=bool(
                        d_diagnostic_compact_step_reports
                    ),
                ):
                    step_result_compact = compact_d_diagnostic_step_result(
                        step_result_compact
                    )
                if (
                    r7_cap_defer_pressure_instrumentation_enabled
                    and r7_cap_defer_pressure_sidecar_path is not None
                ):
                    pressure_mass = pressure_mass_from_tensor_states(pre_apply_states)
                    pressure_mass_delta = (
                        int(pressure_mass) - int(prior_pressure_mass)
                        if prior_pressure_mass is not None
                        else None
                    )
                    append_step_chunk(
                        r7_cap_defer_pressure_sidecar_path,
                        build_step_chunk(
                            step=int(step),
                            global_summary=step_result.global_summary,
                            pressure_mass=pressure_mass,
                            pressure_mass_delta=pressure_mass_delta,
                            optional_selection_scores=(
                                optional_selection_scores_from_step_result_compact(
                                    step_result_compact
                                )
                            ),
                        ),
                    )
                    prior_pressure_mass = pressure_mass
                if (
                    d_recompute_window_instrumentation_enabled
                    and d_recompute_window_log_path is not None
                ):
                    maybe_emit_d_recompute_window_step_records(
                        enabled=True,
                        log_path=d_recompute_window_log_path,
                        step=int(step),
                        pre_update_states=pre_apply_states,
                        post_update_states=states,
                        votes_by_key=votes_by_key,
                        replay_constants=ReplayConstants.from_vote_update_spec(vote_spec),
                        global_summary=step_result.global_summary,
                        selector_manifest=d_recompute_selector_manifest,
                    )
                if (
                    event_coded_recompute_window_log_enabled
                    and d_recompute_window_log_path is not None
                ):
                    emit_event_coded_recompute_window_step_record(
                        enabled=True,
                        log_path=d_recompute_window_log_path,
                        step=int(step),
                        replay_constants=ReplayConstants.from_vote_update_spec(vote_spec),
                    )
                if d_live_carrier_snapshot_enabled and d_live_carrier_snapshot_path is not None:
                    if bool(event_coded_sparse_vote_authority):
                        with progress.phase("live_carrier_snapshot_emit", step=int(step)):
                            emit_live_carrier_snapshots_for_probe_step(
                                enabled=True,
                                log_path=d_live_carrier_snapshot_path,
                                step=int(step),
                                post_update_states=states,
                            )
                    else:
                        emit_live_carrier_snapshots_for_probe_step(
                            enabled=True,
                            log_path=d_live_carrier_snapshot_path,
                            step=int(step),
                            post_update_states=states,
                        )
                if calibration_warmup_collector is not None:
                    calibration_warmup_collector.record_step(
                        step=int(step),
                        pre_update_states=pre_apply_states,
                        votes_by_key=votes_by_key,
                        replay_constants=ReplayConstants.from_vote_update_spec(vote_spec),
                    )
                if device.type == "cuda":
                    step_cuda_memory_snapshots.append(
                        capture_cuda_phase_memory_snapshot(
                            device,
                            label="post_step_update",
                            optimizer_step_index=int(step),
                        )
                    )
                step_reports[str(step)] = {
                    "loss": float(loss.detach().cpu().item()),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "weighted_grad_finite": bool(finite_weighted_grad),
                    "aux_weighted_grad_finite": bool(aux_weighted_grad_finite),
                    "duration_seconds": step_duration_seconds,
                    "metrics": _metrics_to_dict(metrics),
                    "bp_steps": int(extras["bp_steps"]),
                    "q_changed_count": q_changed_count,
                    "science_arm": str(science_arm),
                    "target_vote_law": _science_arm_vote_law(str(science_arm)),
                    "target_tie_policy_id": _science_arm_tie_policy(str(science_arm)),
                    "local_selection_ordering_mode": step_result.global_summary.get(
                        "local_selection_ordering_mode",
                        step_selection_ordering_mode,
                    ),
                    "local_selection_ordering_seed": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
                    "local_selection_ordering_step": int(step),
                    "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
                    "vote_pressure": vote_pressure_by_key,
                    "support_batch": dict(step_batch_metadata),
                    "b2_retained_support": b2_step_receipt,
                    "step_result": step_result_compact,
                    "optimizer_identity_proof": identity_proof,
                }
                if grad_proxy_ingress_receipt is not None:
                    step_reports[str(step)]["grad_proxy_ingress"] = (
                        grad_proxy_ingress_receipt
                    )
                if device.type == "cuda":
                    step_reports[str(step)]["cuda_memory_snapshots"] = list(
                        step_cuda_memory_snapshots
                    )
                if proxy_oracle_drift_receipt is not None:
                    step_reports[str(step)]["proxy_oracle_drift"] = (
                        proxy_oracle_drift_receipt
                    )
                if b2b_step_capture is not None:
                    step_reports[str(step)]["b2b_sequential_capture"] = {
                        "capture_side": b2b_step_capture["capture_side"],
                        "candidate_apply_policy": b2b_step_capture["candidate_apply_policy"],
                        "source_table_hash": b2b_step_capture["source_table_hash"],
                        "pre_update_state_hash": b2b_step_capture["pre_update_state_hash"],
                        "sampled_candidate_count": b2b_step_capture["sampled_candidate_count"],
                        "sparse_singleton_identity_checked_count": b2b_step_capture[
                            "sparse_singleton_identity_checked_count"
                        ],
                        "sparse_singleton_identity_drift_count": b2b_step_capture[
                            "sparse_singleton_identity_drift_count"
                        ],
                        "post_update_q_changed_count": q_changed_count,
                    }
                attach_s3bb_headroom_telemetry_to_step_report(
                    step_reports[str(step)],
                    phase=str(phase),
                    post_update_states=states,
                    snapshot_mode=str(snapshot_mode),
                    headroom_wiring_sidecar_path=headroom_wiring_sidecar_path,
                    step=int(step),
                )
        steps_completed = step
        if (
            audit_callback is not None
            and int(audit_interval) > 0
            and step % int(audit_interval) == 0
        ):
            stop_token = maybe_audit(step)
            if stop_token:
                stop_reason = (
                    "strict_exact_acquired_stop_fast"
                    if stop_token == "strict_exact_acquired"
                    else stop_token
                )
                break
    if audit_callback is not None and str(steps_completed) not in audit_reports:
        stop_token = maybe_audit(steps_completed, final=True)
        if stop_token:
            stop_reason = (
                "strict_exact_acquired_final"
                if stop_token == "strict_exact_acquired"
                else stop_token
            )
    b2b_capture_receipt: dict[str, Any] | None = None
    if b2b_sequential_capture_enabled and b2b_sequential_capture_out is not None:
        rewrite_b2b_trace_with_receipt_emissions(Path(b2b_sequential_capture_out))
        b2b_capture_receipt = finalize_b2b_capture_receipt(
            build_b2b_sequential_capture_receipt(
                capture_out=Path(b2b_sequential_capture_out),
                steps_captured=len(b2b_trace_hashes),
                min_steps_for_verdict=int(b2b_sequential_min_steps_for_verdict),
                trace_hashes=b2b_trace_hashes,
                parent_hash_unchanged=True,
                max_sampled_candidates=int(b2b_sequential_max_sampled_candidates),
            )
        )
    return (
        step_reports,
        updater_config,
        states,
        audit_reports,
        stop_reason,
        steps_completed,
        b2_full_verdict_state,
        b2b_capture_receipt,
        grad_proxy_ingress_crossing_eligible_count_by_step,
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


def _rss_bytes_self() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _build_checkpoint_payload_with_phase_telemetry(
    phase_progress: PhaseProgress,
    tensor_states: Mapping[str, Any],
    *,
    step: int,
    updater_config: Mapping[str, Any],
    oracle_receipt: Mapping[str, Any] | None = None,
    dry_run: bool = True,
    checkpoint_written: bool = False,
) -> dict[str, Any]:
    peak_rss_bytes = _rss_bytes_self()
    peak_at_tensor_key: str | None = None
    export_event_toggle = {"start": True}

    def sample_rss(*, tensor_key: str | None = None) -> None:
        nonlocal peak_rss_bytes, peak_at_tensor_key
        current = _rss_bytes_self()
        if current >= peak_rss_bytes:
            peak_rss_bytes = current
            if tensor_key is not None:
                peak_at_tensor_key = tensor_key
        fields: dict[str, Any] = {
            "rss_bytes": current,
            "rss_peak_bytes": peak_rss_bytes,
        }
        if peak_at_tensor_key is not None:
            fields["rss_peak_at_tensor_key"] = peak_at_tensor_key
        phase_progress.mark("checkpoint_payload", "rss_sample", **fields)

    def on_tensor_export(tensor_key: str, tensor_index: int, tensor_count: int) -> None:
        if export_event_toggle["start"]:
            phase_progress.mark(
                "checkpoint_payload",
                "checkpoint_tensor_export_start",
                tensor_key=tensor_key,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
            )
            sample_rss(tensor_key=tensor_key)
            export_event_toggle["start"] = False
            return
        phase_progress.mark(
            "checkpoint_payload",
            "checkpoint_tensor_export_done",
            tensor_key=tensor_key,
            tensor_index=tensor_index,
            tensor_count=tensor_count,
        )
        sample_rss(tensor_key=tensor_key)
        export_event_toggle["start"] = True

    sample_rss()
    payload = build_authoritative_checkpoint_payload(
        tensor_states,
        step=int(step),
        updater_config=updater_config,
        oracle_receipt=oracle_receipt,
        dry_run=bool(dry_run),
        checkpoint_written=bool(checkpoint_written),
        on_tensor_export=on_tensor_export,
    )
    sample_rss()
    return payload


def _maybe_log_checkpoint_states_dump(
    dump_path: Path | None,
    *,
    tensor_states: Mapping[str, Any],
) -> None:
    if dump_path is None:
        return
    print(
        json.dumps(
            {
                "schema": "hrm_text_158_checkpoint_states_dump_deferred/v0",
                "requested_path": str(dump_path),
                "tensor_count": len(tensor_states),
                "bytes_written": 0,
                "reason": "capture_gate_required_before_runtime_state_dump",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_c2p1_probe(
    *,
    parent: Path,
    parent_sha256: str | None = DEFAULT_PARENT_SHA256,
    scratch_root: Path,
    phase: str = "c2p1-real-model-smoke",
    device: str = "cpu",
    eligible_scope: str = "first-bitlinear",
    eligible_module_limit: int | None = None,
    steps: int = 0,
    batch_size: int = 1,
    max_len: int | None = None,
    curriculum_seed: int = 17,
    support_order_seed: int | None = None,
    init_fidelity_atol: float = 0.0,
    require_q_change: bool = False,
    max_abs_per_tensor: int = 4096,
    audit_interval: int = 0,
    stop_on_strict_exact: bool = False,
    matched_continued_training_horizon_steps: int = 0,
    max_steps_hard: int = C2P2_DEFAULT_MAX_STEPS_HARD,
    emit_progress: bool = False,
    phase_heartbeat_seconds: float | None = None,
    phase_timeout_seconds: float = 0.0,
    total_timeout_seconds: float = 0.0,
    max_silent_phase_seconds: float | None = None,
    phase_timeout_exemption_contract: str = PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF,
    enabled: bool | None = None,
    allow_gpu_launch: bool = False,
    global_cap_contract: str = GLOBAL_CAP_CONTRACT_OFF,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    prior_audit_supports: str | Sequence[str] | None = None,
    b2_retained_supports: str | Sequence[str] | None = None,
    b2_parent_consistency_weight: float = 0.0,
    b2_pc_aux_mode: str = "telemetry",
    b2_full_verdict_mode: bool = False,
    b2_l0b_batch_size: int = 8,
    b2_math_a0_batch_size: int = 16,
    front_c_identity_emission_artifact: Path | None = None,
    front_c_identity_emission_interval: int = 0,
    front_c_independent_oracle: bool = False,
    science_arm: str = ARM_A0_RANK_BUCKET_CURRENT,
    oracle_screen_mode: str | None = None,
    oracle_screen_max_sampled_candidates: int = ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
    b2b_sequential_capture_enabled: bool = False,
    b2b_sequential_capture_out: Path | None = None,
    b2b_sequential_min_steps_for_verdict: int = 50,
    b2b_sequential_max_sampled_candidates: int = PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
    two_tier_carry_w6_enabled: bool = False,
    checkpoint_states_dump: Path | None = None,
    receipt_emit_profile: str = RECEIPT_EMIT_PROFILE_FULL,
    persistent_accumulator_w6_byte_packed: bool = False,
    persistent_accumulator_w5_byte_packed: bool = False,
    persistent_q_ternary_byte_packed: bool = False,
    persistent_q_ternary_base3_codec: bool = False,
    r7_cap_defer_pressure_instrumentation_enabled: bool = False,
    r7_deferred_backlog_carry_enabled: bool = False,
    d_recompute_window_instrumentation_enabled: bool = False,
    d_recompute_selector_manifest_path: Path | None = None,
    event_coded_recompute_window_log_enabled: bool = False,
    d_live_carrier_snapshot_enabled: bool = False,
    d_diagnostic_compact_step_reports: bool = False,
    d_recompute_calibration_warmup_out: Path | None = None,
    votes_emit_enabled: bool = False,
    votes_emit_root: Path | None = None,
    carrier_growth_enabled: bool = False,
    persistent_accumulator_event_coded_live: bool = False,
    event_coded_live_demotion_band: int = 1,
    confirmation_envelope: str | None = None,
    dense_accumulator_w7_clip: bool = False,
    dense_accumulator_w8_clip: bool = False,
    event_coded_sparse_vote_authority: bool = False,
    vote_update_decay_numerator: int | None = None,
    vote_update_decay_denominator: int | None = None,
) -> dict[str, Any]:
    oracle_screen_budget = int(oracle_screen_max_sampled_candidates)
    if oracle_screen_budget not in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES:
        raise ValueError(
            "oracle_screen_max_sampled_candidates must be one of "
            f"{ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES}"
        )
    validate_recompute_window_log_flag_mutual_exclusion(
        d_recompute_window_instrumentation_enabled=bool(
            d_recompute_window_instrumentation_enabled
        ),
        event_coded_recompute_window_log_enabled=bool(
            event_coded_recompute_window_log_enabled
        ),
    )
    assert_default_off(enabled)
    assert_profile_tracemalloc_debugmallocstats_mutual_exclusion()
    if bool(persistent_accumulator_w6_byte_packed) and bool(persistent_accumulator_w5_byte_packed):
        raise ValueError(
            "persistent_accumulator_w6_byte_packed and persistent_accumulator_w5_byte_packed "
            "are mutually exclusive"
        )
    if bool(dense_accumulator_w7_clip) and (
        bool(persistent_accumulator_w6_byte_packed) or bool(persistent_accumulator_w5_byte_packed)
    ):
        raise ValueError(
            "dense_accumulator_w7_clip is mutually exclusive with W5/W6 byte-packed accumulators"
        )
    if bool(dense_accumulator_w8_clip) and (
        bool(persistent_accumulator_w6_byte_packed) or bool(persistent_accumulator_w5_byte_packed)
    ):
        raise ValueError(
            "dense_accumulator_w8_clip is mutually exclusive with W5/W6 byte-packed accumulators"
        )
    narrow_clip_count = sum(
        bool(flag)
        for flag in (
            dense_accumulator_w7_clip,
            dense_accumulator_w8_clip,
        )
    )
    if narrow_clip_count > 1:
        raise ValueError(
            "dense_accumulator_w7_clip and dense_accumulator_w8_clip are mutually exclusive"
        )
    _probe_runtime_env_snapshot = _snapshot_probe_runtime_env()
    try:
        if bool(persistent_accumulator_event_coded_live):
            if bool(persistent_accumulator_w6_byte_packed) or bool(
                persistent_accumulator_w5_byte_packed
            ) or bool(dense_accumulator_w7_clip) or bool(dense_accumulator_w8_clip):
                raise ValueError(
                    "persistent_accumulator_event_coded_live is mutually exclusive with "
                    "W5/W6 byte-packed accumulators and W7/W8 clip boundaries"
                )
            resolve_live_acc_carrier_selector(
                v4_enabled=True,
                w5_enabled=bool(persistent_accumulator_w5_byte_packed),
                w6_enabled=bool(persistent_accumulator_w6_byte_packed),
                w7_enabled=bool(dense_accumulator_w7_clip),
                w8_enabled=bool(dense_accumulator_w8_clip),
            )
            os.environ[RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV] = "1"
        elif RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV in os.environ:
            os.environ.pop(RUN_EVENT_CODED_ACC_LIVE_CARRIER_ENV, None)
        if bool(persistent_accumulator_w6_byte_packed):
            os.environ[PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV] = "1"
        elif PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV in os.environ:
            os.environ.pop(PERSISTENT_ACCUMULATOR_W6_BYTE_PACKED_ENV, None)
        if bool(persistent_accumulator_w5_byte_packed):
            os.environ[PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV] = "1"
            os.environ[RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV] = "1"
            os.environ.pop(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, None)
        elif PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV in os.environ:
            os.environ.pop(PERSISTENT_ACCUMULATOR_W5_BYTE_PACKED_ENV, None)
            os.environ.pop(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, None)
        if bool(dense_accumulator_w7_clip):
            os.environ[RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV] = "1"
            os.environ.pop(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, None)
            os.environ.pop(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, None)
            os.environ.pop(RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV, None)
        elif RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV in os.environ:
            os.environ.pop(RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV, None)
        if bool(dense_accumulator_w8_clip):
            os.environ[RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV] = "1"
            os.environ.pop(RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV, None)
            os.environ.pop(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, None)
            os.environ.pop(RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV, None)
        elif RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV in os.environ:
            os.environ.pop(RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV, None)
        if bool(persistent_q_ternary_byte_packed):
            os.environ[PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV] = "1"
        elif PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV in os.environ:
            os.environ.pop(PERSISTENT_Q_TERNARY_BYTE_PACKED_ENV, None)
        if bool(persistent_q_ternary_base3_codec):
            os.environ[PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV] = "1"
        elif PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV in os.environ:
            os.environ.pop(PERSISTENT_Q_TERNARY_BASE3_CODEC_ENV, None)
        if oracle_screen_mode is not None and str(oracle_screen_mode) not in ORACLE_SCREEN_MODE_CHOICES:
            raise ValueError(
                f"oracle_screen_mode must be one of {ORACLE_SCREEN_MODE_CHOICES}, got {oracle_screen_mode!r}"
            )
        if oracle_screen_mode is None and str(science_arm) not in SCIENCE_ARM_CHOICES:
            raise ValueError(f"science_arm must be one of {SCIENCE_ARM_CHOICES}, got {science_arm!r}")
        if int(max_steps_hard) <= 0:
            raise ValueError("max_steps_hard must be positive")
        if int(steps) > int(max_steps_hard):
            raise ValueError(
                f"steps={int(steps)} exceeds max_steps_hard={int(max_steps_hard)}"
            )
        if int(audit_interval) < 0:
            raise ValueError("audit_interval must be non-negative")
        if int(matched_continued_training_horizon_steps) < 0:
            raise ValueError("matched_continued_training_horizon_steps must be non-negative")
        if support_order_seed is not None and int(support_order_seed) < 0:
            raise ValueError("support_order_seed must be non-negative when set")
        global_cap_contract_receipt = named_global_cap_contract_receipt(str(global_cap_contract))
        if str(tie_rule_mode) not in GLOBAL_TIE_RULE_MODES:
            raise ValueError(
                f"tie_rule_mode must be one of {GLOBAL_TIE_RULE_MODES}, got {tie_rule_mode!r}"
            )
        if (
            str(global_cap_contract) == GLOBAL_CAP_CONTRACT_OFF
            and str(tie_rule_mode) != EXACT_GLOBAL_CAP_TIE_RULE_MODE
        ):
            raise ValueError(
                "tie_rule_mode requires a non-off global_cap_contract; "
                "use exact_global_cap when global_cap_contract=off"
            )
        requested_prior_audit_supports = parse_prior_audit_supports(prior_audit_supports)
        requested_b2_retained_supports = parse_b2_retained_supports(b2_retained_supports)
        if b2_full_verdict_mode:
            missing_retained = [
                support
                for support in B2_FULL_STOP_SUPPORTS
                if support not in requested_b2_retained_supports
            ]
            if missing_retained:
                raise ValueError(
                    "B2-full verdict mode requires retained supports "
                    f"{B2_FULL_STOP_SUPPORTS}; missing {tuple(missing_retained)}"
                )
            required_prior_supports = (*B2_FULL_STOP_SUPPORTS, "L0c1")
            missing_prior = [
                support
                for support in required_prior_supports
                if support not in requested_prior_audit_supports
            ]
            if missing_prior:
                raise ValueError(
                    "B2-full verdict mode requires prior audit supports "
                    f"{required_prior_supports}; missing {tuple(missing_prior)}"
                )
            if int(audit_interval) <= 0:
                raise ValueError("B2-full verdict mode requires audit_interval > 0")
        if b2_pc_aux_mode not in B2_PC_AUX_MODES:
            raise ValueError(f"b2_pc_aux_mode must be one of {B2_PC_AUX_MODES}, got {b2_pc_aux_mode!r}")
        if float(b2_parent_consistency_weight) < 0.0:
            raise ValueError("b2_parent_consistency_weight must be non-negative")
        if front_c_identity_emission_artifact is not None:
            if int(steps) <= 0:
                raise ValueError("Front-C identity emission requires steps > 0")
            if int(front_c_identity_emission_interval) < 0:
                raise ValueError("front_c_identity_emission_interval must be non-negative")
            required_front_c_prior = ("L0b", "math_a0", "L0c1")
            missing_front_c_prior = [
                support
                for support in required_front_c_prior
                if support not in requested_prior_audit_supports
            ]
            if missing_front_c_prior:
                raise ValueError(
                    "Front-C identity emission requires prior audit supports "
                    f"{required_front_c_prior}; missing {tuple(missing_front_c_prior)}"
                )
        if oracle_screen_mode is not None:
            if int(steps) != 1:
                raise ValueError("oracle_screen_mode requires steps=1 because the screen evaluates one support batch")
            if int(batch_size) <= 0:
                raise ValueError("oracle_screen_mode requires batch_size > 0")
            if requested_prior_audit_supports:
                raise ValueError("oracle_screen_mode does not support prior_audit_supports")
            if requested_b2_retained_supports:
                raise ValueError("oracle_screen_mode does not support b2_retained_supports")
            if float(b2_parent_consistency_weight) != 0.0:
                raise ValueError("oracle_screen_mode does not support parent-consistency auxiliary paths")
            if str(global_cap_contract) != GLOBAL_CAP_CONTRACT_OFF:
                raise ValueError("oracle_screen_mode requires global_cap_contract=off")
            if str(tie_rule_mode) != EXACT_GLOBAL_CAP_TIE_RULE_MODE:
                raise ValueError("oracle_screen_mode requires tie_rule_mode=exact_global_cap")
            if bool(stop_on_strict_exact):
                raise ValueError("oracle_screen_mode does not support stop_on_strict_exact")
            if int(matched_continued_training_horizon_steps) != 0:
                raise ValueError("oracle_screen_mode requires matched_continued_training_horizon_steps=0")
            if bool(b2_full_verdict_mode):
                raise ValueError("oracle_screen_mode does not support b2_full_verdict_mode")
            if front_c_identity_emission_artifact is not None or bool(front_c_independent_oracle):
                raise ValueError("oracle_screen_mode does not support Front-C identity emission")
        if b2b_sequential_capture_enabled:
            if oracle_screen_mode is not None:
                raise ValueError(
                    "b2b sequential capture cannot run together with oracle_screen_mode"
                )
            if int(steps) <= 0:
                raise ValueError("b2b sequential capture requires steps > 0")
            if int(b2b_sequential_min_steps_for_verdict) <= 0:
                raise ValueError("b2b sequential min steps for verdict must be positive")
            if (
                int(b2b_sequential_max_sampled_candidates)
                != PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES
            ):
                raise ValueError(
                    "b2b sequential capture requires max_sampled_candidates == 32"
                )
            if b2b_sequential_capture_out is None:
                b2b_sequential_capture_out = scratch_root / "b2b_sequential_trace.ndjson"
        phase_timeout_exemptions = resolve_phase_timeout_exemptions(
            contract=str(phase_timeout_exemption_contract)
        )
        validate_b2b_phase_timeout_launch_requirements(
            b2b_sequential_capture_enabled=bool(b2b_sequential_capture_enabled),
            phase_timeout_seconds=float(phase_timeout_seconds),
            phase_timeout_exemption_contract=str(phase_timeout_exemption_contract),
            total_timeout_seconds=float(total_timeout_seconds),
            silent_phase_timeout_seconds=max_silent_phase_seconds,
            allow_gpu_launch=bool(allow_gpu_launch),
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        oracle_screen_max_seconds = oracle_screen_budget_max_seconds(oracle_screen_budget)
        b2_support_batch_sizes = {
            "L0b": int(b2_l0b_batch_size),
            "math_a0": int(b2_math_a0_batch_size),
        }
        torch_device = torch.device(device)
        guard_gpu_launch(torch_device, allow_gpu_launch=allow_gpu_launch)
        device_guard = assert_probe_device_ready(torch_device)
        scratch_root.mkdir(parents=True, exist_ok=True)
        if str(receipt_emit_profile) not in RECEIPT_EMIT_PROFILE_CHOICES:
            raise ValueError(
                f"receipt_emit_profile must be one of {RECEIPT_EMIT_PROFILE_CHOICES}, "
                f"got {receipt_emit_profile!r}"
            )
        slim_receipt_emit = str(receipt_emit_profile) == RECEIPT_EMIT_PROFILE_SLIM
        snapshot_mode = (
            SNAPSHOT_MODE_AGGREGATE_ONLY if slim_receipt_emit else SNAPSHOT_MODE_FULL
        )
        headroom_wiring_sidecar_path = (
            scratch_root / HEADROOM_WIRING_SIDECAR_FILENAME if slim_receipt_emit else None
        )
        if headroom_wiring_sidecar_path is not None:
            initialize_headroom_wiring_sidecar_for_probe_session(headroom_wiring_sidecar_path)
        r7_cap_defer_pressure_sidecar_path = (
            scratch_root / R7_SIDECAR_FILENAME
            if bool(r7_cap_defer_pressure_instrumentation_enabled)
            else None
        )
        d_recompute_window_log_path = (
            scratch_root / D_RECOMPUTE_WINDOW_LOG_FILENAME
            if bool(d_recompute_window_instrumentation_enabled)
            or bool(event_coded_recompute_window_log_enabled)
            else None
        )
        if d_recompute_window_log_path is not None:
            initialize_recompute_window_log_for_probe_session(d_recompute_window_log_path)
        d_live_carrier_snapshot_path = (
            Path(scratch_root) / "live_carrier_snapshot.jsonl"
            if bool(d_live_carrier_snapshot_enabled)
            else None
        )
        if d_live_carrier_snapshot_path is not None:
            initialize_live_carrier_snapshot_log(d_live_carrier_snapshot_path)
        run_log_path = install_probe_durable_run_log(scratch_root)
        cuda_memory_snapshots_jsonl_path = install_probe_cuda_memory_snapshot_jsonl(
            scratch_root
        )
        faulthandler_report = register_probe_faulthandler(run_log_path=run_log_path)
        silent_phase_timeout_seconds = resolve_max_silent_phase_seconds(
            allow_gpu_launch=bool(allow_gpu_launch),
            max_silent_phase_seconds=max_silent_phase_seconds,
        )
        phase_timeout_exemption_receipt = build_phase_timeout_exemption_receipt(
            contract=str(phase_timeout_exemption_contract),
            phase_timeout_seconds=float(phase_timeout_seconds),
            silent_phase_timeout_seconds=silent_phase_timeout_seconds,
            total_timeout_seconds=float(total_timeout_seconds),
        )
        phase_budget_interrupt_authority = build_phase_budget_interrupt_authority_contract(
            silent_phase_timeout_seconds=silent_phase_timeout_seconds,
            max_silent_phase_seconds=silent_phase_timeout_seconds,
        )
        last_active_phase_path = scratch_root / "last_active_phase.json"
        resolved_phase_heartbeat_seconds = resolve_phase_heartbeat_seconds(
            emit_progress=bool(emit_progress),
            phase_heartbeat_seconds=phase_heartbeat_seconds,
        )
        stdout_liveness_receipt = build_probe_stdout_liveness_receipt(
            emit_progress=bool(emit_progress),
            phase_heartbeat_seconds=resolved_phase_heartbeat_seconds,
        )
        validate_probe_stdout_liveness_config(stdout_liveness_receipt)
        milestone_emitter = PhaseMilestoneEmitter(
            scratch_root,
            enabled=bool(event_coded_sparse_vote_authority),
            device=torch_device,
        )
        host_rss_profile_path = (
            scratch_root / HOST_RSS_PROFILE_JSONL_NAME
            if profile_host_rss_enabled()
            else None
        )
        if profile_alloc_hook_enabled():
            from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import prefault_hook

            prefault_hook()
        phase_progress = PhaseProgress(
            enabled=bool(emit_progress),
            device=torch_device,
            phase_timeout_seconds=float(phase_timeout_seconds),
            total_timeout_seconds=float(total_timeout_seconds),
            silent_phase_timeout_seconds=silent_phase_timeout_seconds,
            phase_heartbeat_interval_seconds=resolved_phase_heartbeat_seconds,
            last_active_phase_path=last_active_phase_path,
            phase_timeout_exemptions=phase_timeout_exemptions,
            phase_timeout_exemption_contract=str(phase_timeout_exemption_contract),
            milestone_emitter=milestone_emitter,
            host_rss_profile_path=host_rss_profile_path,
        )
        if torch_device.type == "cuda":
            with phase_progress.phase("cuda_memory_reset"):
                reset_cuda_memory_stats(torch_device)
        run_timing_start = _timing_start(torch_device)
    
        with phase_progress.phase("load"):
            ckpt, parent_hash_before = load_parent_checkpoint(parent, expected_sha256=parent_sha256)
        set_probe_terminal_flush_context(
            ProbeTerminalFlushContext(
                scratch_root=scratch_root,
                parent_checkpoint_path=Path(parent),
                parent_hash_before=str(parent_hash_before),
            )
        )
        with phase_progress.phase("build_model"):
            model, tok, cfg = build_model_from_checkpoint(ckpt, torch_device)
        b2_parent_model = None
        if requested_b2_retained_supports and float(b2_parent_consistency_weight) > 0.0:
            with phase_progress.phase("b2_parent_model_build"):
                b2_parent_model, _parent_tok, _parent_cfg = build_model_from_checkpoint(ckpt, torch_device)
                b2_parent_model.eval()
                for param in b2_parent_model.parameters():
                    param.requires_grad_(False)
        with phase_progress.phase("support_build"):
            support_batches, support_cycler_proof = build_identity_full_support_batches(
                tok=tok,
                max_len=int(max_len or ckpt["config"]["max_seq_len"]),
                batch_size=int(batch_size),
                curriculum_seed=int(curriculum_seed),
                device=torch_device,
                support_order_seed=support_order_seed,
            )
        prior_support_sets: dict[str, dict[str, Any]] = {}
        if requested_prior_audit_supports:
            with phase_progress.phase("prior_support_build"):
                prior_support_sets = build_prior_audit_support_sets(
                    requested_prior_audit_supports,
                    tok=tok,
                    max_len=int(max_len or ckpt["config"]["max_seq_len"]),
                    batch_size=int(batch_size),
                    run_curriculum_seed=int(curriculum_seed),
                    device=torch_device,
                )
        b2_retained_support_sets: dict[str, dict[str, Any]] = {}
        if requested_b2_retained_supports:
            with phase_progress.phase("b2_retained_support_build"):
                b2_retained_support_sets = build_b2_retained_support_sets(
                    requested_b2_retained_supports,
                    tok=tok,
                    max_len=int(max_len or ckpt["config"]["max_seq_len"]),
                    support_batch_sizes=b2_support_batch_sizes,
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
            eligible_full = select_eligible_bitlinears(model, eligible_scope=eligible_scope)
            eligible = apply_eligible_module_limit(
                eligible_full,
                eligible_scope=eligible_scope,
                eligible_module_limit=eligible_module_limit,
            )
            eligible_scale_fields = build_eligible_scale_receipt_fields(
                eligible,
                eligible_scope=eligible_scope,
                eligible_module_limit=eligible_module_limit,
                eligible_full_count=len(eligible_full),
            )
        with phase_progress.phase("state_init"):
            tensor_states, init_fidelity = derive_tensor_states_and_check_init_fidelity(
                eligible,
                threshold=float(init_fidelity_atol),
            )
            if bool(persistent_accumulator_event_coded_live):
                tensor_states = {
                    state_key: make_event_coded_live_tensor_state(
                        state_key,
                        state.q_levels,
                        state.frozen_scale,
                        demotion_band=int(event_coded_live_demotion_band),
                    )
                    for state_key, state in tensor_states.items()
                }
        if not init_fidelity["all_pass"]:
            raise RuntimeError("weight-level init-fidelity allclose failed")
    
        d_recompute_selector_manifest: StratifiedSelectorManifest | None = None
        if d_recompute_selector_manifest_path is not None:
            d_recompute_selector_manifest = load_stratified_selector_manifest(
                d_recompute_selector_manifest_path
            )
    
        calibration_warmup_collector: CalibrationWarmupCollector | None = None
        if d_recompute_calibration_warmup_out is not None:
            if not d_recompute_window_instrumentation_enabled:
                raise ValueError(
                    "d_recompute_calibration_warmup_out requires "
                    "d_recompute_window_instrumentation_enabled"
                )
            from scripts.hrm_text_158_d_recompute_calibration_prepass import (
                default_calibration_policy,
            )
    
            warmup_observations_path = Path(d_recompute_calibration_warmup_out)
            if not warmup_observations_path.is_absolute():
                warmup_observations_path = scratch_root.parent / warmup_observations_path
            calibration_warmup_collector = CalibrationWarmupCollector(
                output_path=warmup_observations_path,
                pre_warmup_parent_sha256=str(parent_hash_before),
                policy=default_calibration_policy(),
            )
    
        _validate_event_coded_sparse_vote_authority_config(
            event_coded_sparse_vote_authority=bool(event_coded_sparse_vote_authority),
            persistent_accumulator_event_coded_live=bool(
                persistent_accumulator_event_coded_live
            ),
            two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
            b2b_sequential_capture_enabled=bool(b2b_sequential_capture_enabled),
            votes_emit_enabled=bool(votes_emit_enabled),
            carrier_growth_enabled=bool(carrier_growth_enabled),
            d_recompute_window_instrumentation_enabled=bool(
                d_recompute_window_instrumentation_enabled
            ),
            d_recompute_calibration_warmup_out=d_recompute_calibration_warmup_out,
        )
    
        front_c_identity_collector = None
        if front_c_identity_emission_artifact is not None:
            artifact_path = Path(front_c_identity_emission_artifact)
            if not artifact_path.is_absolute():
                artifact_path = scratch_root / artifact_path
            front_c_identity_collector = FrontCLiveIdentityCollector(
                artifact_path=artifact_path,
                emission_interval=int(front_c_identity_emission_interval),
                audit_interval=int(audit_interval),
                independent_oracle_compare=bool(front_c_independent_oracle),
            )
    
        with phase_progress.phase("forward_fidelity"):
            if profile_alloc_hook_enabled() or profile_tracemalloc_enabled():
                skip_reason = (
                    "alloc_hook_attribution_fixture"
                    if profile_alloc_hook_enabled()
                    else "tracemalloc_attribution_fixture"
                )
                forward_init_fidelity = {
                    "schema": "hrm_text_158_c2p1_weight_level_init_fidelity/v0",
                    "skipped": True,
                    "skip_reason": skip_reason,
                }
            else:
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
        if oracle_screen_mode is not None:
            extras = model.compute_train_extra_args(1, max(1, int(steps)))
            if str(oracle_screen_mode) == ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY:
                oracle_screen = run_candidate_set_viability_oracle_screen(
                    model=model,
                    batch=model_batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=torch_device,
                    max_abs_per_tensor=int(max_abs_per_tensor),
                    extras=extras,
                    max_sampled_candidates=oracle_screen_budget,
                    max_seconds=oracle_screen_max_seconds,
                    phase_progress=phase_progress,
                )
            elif str(oracle_screen_mode) == ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT:
                oracle_screen = run_credit_ranking_pivot_measurement_oracle_screen(
                    model=model,
                    batch=model_batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=torch_device,
                    max_abs_per_tensor=int(max_abs_per_tensor),
                    extras=extras,
                    max_sampled_candidates=oracle_screen_budget,
                    max_seconds=oracle_screen_max_seconds,
                    phase_progress=phase_progress,
                )
            elif str(oracle_screen_mode) == ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR:
                oracle_screen = run_within_tie_band_discriminator_oracle_screen(
                    model=model,
                    batch=model_batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=torch_device,
                    max_abs_per_tensor=int(max_abs_per_tensor),
                    extras=extras,
                    max_sampled_candidates=oracle_screen_budget,
                    max_seconds=oracle_screen_max_seconds,
                    phase_progress=phase_progress,
                )
            elif str(oracle_screen_mode) == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE:
                oracle_screen = run_activation_credit_scale_smoke_oracle_screen(
                    model=model,
                    batch=model_batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=torch_device,
                    max_abs_per_tensor=int(max_abs_per_tensor),
                    extras=extras,
                    max_sampled_candidates=oracle_screen_budget,
                    max_seconds=oracle_screen_max_seconds,
                    phase_progress=phase_progress,
                )
            elif str(oracle_screen_mode) == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT:
                oracle_screen = run_activation_credit_measurement_oracle_screen(
                    model=model,
                    batch=model_batch,
                    tensor_states=tensor_states,
                    eligible_modules=eligible,
                    device=torch_device,
                    max_abs_per_tensor=int(max_abs_per_tensor),
                    extras=extras,
                    max_sampled_candidates=oracle_screen_budget,
                    max_seconds=oracle_screen_max_seconds,
                    phase_progress=phase_progress,
                )
            else:
                raise RuntimeError(f"unsupported oracle screen mode {oracle_screen_mode!r}")
            with phase_progress.phase("checkpoint_payload"):
                checkpoint_payload = build_authoritative_checkpoint_payload(
                    tensor_states,
                    step=0,
                    updater_config={
                        "oracle_screen_mode": str(oracle_screen_mode),
                        "projection_law": S1_PROJECTION_LAW,
                        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
                    },
                    oracle_receipt=None,
                    dry_run=True,
                    checkpoint_written=False,
                )
                validate_authoritative_resume_payload(checkpoint_payload)
            with phase_progress.phase("parent_hash_after"):
                parent_hash_after = file_sha256(parent)
            parent_hash_unchanged = parent_hash_before == parent_hash_after
            if not parent_hash_unchanged:
                raise RuntimeError("parent checkpoint hash changed during oracle screen")
            total_run_duration_seconds = _timing_duration_seconds(
                run_timing_start,
                torch_device,
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
                "faulthandler": faulthandler_report,
                "silent_phase_guard": {
                    "default_on_with_allow_gpu_launch": True,
                    "allow_gpu_launch": bool(allow_gpu_launch),
                    "max_silent_phase_seconds": silent_phase_timeout_seconds,
                    "last_active_phase_path": str(last_active_phase_path),
                    "fail_closed_mechanism": "faulthandler.dump_traceback_later(exit=True)",
                },
                "stdout_liveness": stdout_liveness_receipt,
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
                **eligible_scale_fields,
                "weight_level_init_fidelity": init_fidelity,
                "forward_level_init_fidelity": forward_init_fidelity,
                "steps_requested": int(steps),
                "steps_completed": int(steps),
                "max_steps_hard": int(max_steps_hard),
                "audit_interval": int(audit_interval),
                "stop_on_strict_exact": False,
                "matched_continued_training_horizon_steps": 0,
                "global_cap_contract": global_cap_contract_receipt,
                "tie_rule_mode": str(tie_rule_mode),
                "science_arm": None,
                "oracle_screen_mode": str(oracle_screen_mode),
                "target_vote_law": S1_RANK_BUCKET_VOTE_LAW,
                "target_tie_policy_id": TIE_POLICY_CURRENT_MARGIN_INDEX,
                "local_selection_ordering_mode": LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
                "local_selection_ordering_seed": 17,
                "local_selection_ordering_step": 1,
                "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
                "default_rank_bucket_path_unchanged": True,
                "stop_reason": "oracle_screen_completed",
                "forward_backward_update_executed": False,
                "step0_optimizer_identity_proof": prove_step0_optimizer_identity(model, eligible),
                "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
                "step_reports": {},
                "audit_reports": {},
                "prior_audit": build_prior_audit_receipt(
                    requested_supports=(),
                    support_sets={},
                    start_reports={},
                    final_reports={},
                ),
                "b2_retention": build_b2_retention_receipt(
                    requested_supports=(),
                    support_sets={},
                    step_reports={},
                    pc_aux_mode=str(b2_pc_aux_mode),
                    parent_consistency_weight=0.0,
                ),
                "front_c_identity_emission": {"enabled": False},
                "timing_summary": {
                    "schema": C2P2_TIMING_SCHEMA_VERSION,
                    "step_reports": {},
                    "audit_reports": {},
                    "total_run_duration_seconds": total_run_duration_seconds,
                },
                "acquisition_trajectory": build_acquisition_trajectory(
                    audit_enabled=False,
                    audit_reports={},
                    step_reports={},
                    support_cycler_proof=support_cycler_proof,
                    audit_interval=0,
                    stop_on_strict_exact=False,
                    matched_continued_training_horizon_steps=0,
                    max_steps_hard=int(max_steps_hard),
                    stop_reason="oracle_screen_completed",
                    timing_summary={
                        "schema": C2P2_TIMING_SCHEMA_VERSION,
                        "step_reports": {},
                        "audit_reports": {},
                        "total_run_duration_seconds": total_run_duration_seconds,
                    },
                ),
                "checkpoint_payload": checkpoint_payload,
                "memory": cuda_memory_receipt(torch_device),
                "oracle_screen": oracle_screen,
                "branch_classification": oracle_screen["branch_classification"],
                "phase_telemetry": phase_progress.to_dict(),
            }
            receipt_path = scratch_root / "receipt.json"
            receipt["receipt_path"] = str(receipt_path)
            receipt["run_log_path"] = str(run_log_path)
            receipt["cuda_memory_snapshots_jsonl_path"] = str(
                cuda_memory_snapshots_jsonl_path
            )
            receipt["terminal_status"] = build_receipt_terminal_status(
                stop_reason=str(receipt["stop_reason"]),
                steps_completed=int(receipt["steps_completed"]),
                steps_requested=int(receipt["steps_requested"]),
            )
            with phase_progress.phase("receipt_write", path=str(receipt_path)):
                receipt["phase_telemetry"] = phase_progress.to_dict()
                _attach_obmalloc_dedup_evidence(receipt)
                receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            receipt["phase_telemetry"] = phase_progress.to_dict()
            return receipt
    
        prior_audit_start_reports: dict[str, dict[str, Any]] = {}
        if prior_support_sets:
            with phase_progress.phase("prior_audit0"):
                prior_audit_start_reports = audit_prior_support_sets(
                    model,
                    prior_support_sets,
                    tensor_states,
                    eligible,
                    tok=tok,
                    device=torch_device,
                    phase="prior_audit0",
                    step=0,
                    total_steps=max(1, int(steps)),
                )
    
        step0_optimizer_identity_proof = None
        if int(steps) <= 0:
            with phase_progress.phase("step0_optimizer_identity"):
                step0_optimizer_identity_proof = prove_step0_optimizer_identity(model, eligible)
    
        audit_enabled = (
            int(audit_interval) > 0
            or bool(stop_on_strict_exact)
            or bool(b2_full_verdict_mode)
        )
    
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
    
        b2_full_prior_snapshot_callback = None
        b2_full_audit_export_callback = None
        if b2_full_verdict_mode:
    
            def b2_full_prior_snapshot_callback(
                step: int,
                states: Mapping[str, Any],
                target_audit: Mapping[str, Any],
                coverage_by_support: Mapping[str, Mapping[str, Any]],
            ) -> dict[str, Any]:
                current_reports = audit_prior_support_sets(
                    model,
                    prior_support_sets,
                    states,
                    eligible,
                    tok=tok,
                    device=torch_device,
                    phase="b2_full_prior_snapshot",
                    step=int(step),
                    total_steps=max(1, int(steps)),
                )
                return build_b2_full_prior_snapshot(
                    snapshot_name="runtime_prior_snapshot",
                    step=int(step),
                    target_audit=target_audit,
                    coverage_by_support=coverage_by_support,
                    start_reports=prior_audit_start_reports,
                    current_reports=current_reports,
                )
    
            def b2_full_audit_export_callback(
                step: int,
                states: Mapping[str, Any],
                target_audit: Mapping[str, Any],
                coverage_by_support: Mapping[str, Mapping[str, Any]],
                verdict_state: Mapping[str, Any],
                updater_config: Mapping[str, Any],
            ) -> str | None:
                checkpoint_payload = build_authoritative_checkpoint_payload(
                    states,
                    step=int(step),
                    updater_config=updater_config,
                    oracle_receipt=None,
                    dry_run=True,
                    checkpoint_written=False,
                )
                validate_authoritative_resume_payload(checkpoint_payload)
                parent_hash_current = file_sha256(parent)
                audit_dir = scratch_root / "audits" / f"step_{int(step):04d}"
                audit_dir.mkdir(parents=True, exist_ok=True)
                audit_path = audit_dir / "summary.json"
                summary = {
                    "schema": B2_FULL_VERDICT_SCHEMA_VERSION,
                    "artifact_role": "b2_full_audit_summary",
                    "step": int(step),
                    "dry_run": True,
                    "checkpoint_written": False,
                    "checkpoint_artifact_written": False,
                    "pt_artifact_written": False,
                    "parent_hash_before": parent_hash_before,
                    "parent_hash_current": parent_hash_current,
                    "parent_hash_unchanged": parent_hash_current == parent_hash_before,
                    "checkpoint_payload_summary": {
                        "schema": checkpoint_payload["schema"],
                        "artifact_role": checkpoint_payload["artifact_role"],
                        "step": checkpoint_payload["step"],
                        "dry_run": checkpoint_payload["dry_run"],
                        "checkpoint_written": checkpoint_payload["checkpoint_written"],
                        "authoritative_state_sha256": checkpoint_payload[
                            "authoritative_state_sha256"
                        ],
                        "updater_config_sha256": checkpoint_payload[
                            "updater_config_sha256"
                        ],
                        "tensor_summary_count": len(checkpoint_payload["tensor_summaries"]),
                    },
                    "target_audit": dict(target_audit),
                    "coverage_by_support": {
                        support: dict(coverage)
                        for support, coverage in coverage_by_support.items()
                    },
                    "math_a0_coverage_cycles": b2_full_coverage_cycles(
                        coverage_by_support,
                        "math_a0",
                    ),
                    "l0b_coverage_cycles": b2_full_coverage_cycles(
                        coverage_by_support,
                        "L0b",
                    ),
                    "verdict_summary": {
                        "prior_audit_count": verdict_state.get("prior_audit_count", 0),
                        "snapshot_steps": dict(verdict_state.get("snapshot_steps", {})),
                        "combined_stop": dict(verdict_state.get("combined_stop", {})),
                        "first_audited_target_ge_90": summarize_b2_full_prior_snapshot(
                            verdict_state.get("first_audited_target_ge_90")
                        ),
                        "first_covered_target_ge_90": summarize_b2_full_prior_snapshot(
                            verdict_state.get("first_covered_target_ge_90")
                        ),
                    },
                }
                audit_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                return str(audit_path)
    
        b2b_capture_receipt: dict[str, Any] | None = None
        with phase_progress.phase("bounded_steps"):
            (
                step_reports,
                updater_config,
                final_states,
                audit_reports,
                stop_reason,
                steps_completed,
                b2_full_verdict_state,
                b2b_capture_receipt,
                grad_proxy_ingress_crossing_eligible_count_by_step,
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
                b2_retained_support_sets=b2_retained_support_sets,
                b2_parent_model=b2_parent_model,
                b2_parent_consistency_weight=float(b2_parent_consistency_weight),
                b2_pc_aux_mode=str(b2_pc_aux_mode),
                audit_callback=audit_callback if audit_enabled else None,
                audit_interval=int(audit_interval),
                stop_on_strict_exact=bool(stop_on_strict_exact),
                matched_continued_training_horizon_steps=int(
                    matched_continued_training_horizon_steps
                ),
                global_cap_contract=str(global_cap_contract),
                tie_rule_mode=str(tie_rule_mode),
                science_arm=str(science_arm),
                b2_full_verdict_mode=bool(b2_full_verdict_mode),
                b2_full_prior_snapshot_callback=b2_full_prior_snapshot_callback,
                b2_full_audit_export_callback=b2_full_audit_export_callback,
                front_c_identity_collector=front_c_identity_collector,
                phase_progress=phase_progress,
                b2b_sequential_capture_enabled=bool(b2b_sequential_capture_enabled),
                b2b_sequential_capture_out=b2b_sequential_capture_out,
                b2b_sequential_min_steps_for_verdict=int(b2b_sequential_min_steps_for_verdict),
                b2b_sequential_max_sampled_candidates=int(
                    b2b_sequential_max_sampled_candidates
                ),
                two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
                oracle_screen_max_sampled_candidates=int(
                    oracle_screen_max_sampled_candidates
                ),
                phase=str(phase),
                snapshot_mode=str(snapshot_mode),
                headroom_wiring_sidecar_path=headroom_wiring_sidecar_path,
                r7_cap_defer_pressure_instrumentation_enabled=bool(
                    r7_cap_defer_pressure_instrumentation_enabled
                ),
                r7_deferred_backlog_carry_enabled=bool(r7_deferred_backlog_carry_enabled),
                r7_cap_defer_pressure_sidecar_path=r7_cap_defer_pressure_sidecar_path,
                d_recompute_window_instrumentation_enabled=bool(
                    d_recompute_window_instrumentation_enabled
                ),
                d_recompute_window_log_path=d_recompute_window_log_path,
                d_recompute_selector_manifest=d_recompute_selector_manifest,
                event_coded_recompute_window_log_enabled=bool(
                    event_coded_recompute_window_log_enabled
                ),
                d_live_carrier_snapshot_enabled=bool(d_live_carrier_snapshot_enabled),
                d_live_carrier_snapshot_path=d_live_carrier_snapshot_path,
                receipt_emit_profile=str(receipt_emit_profile),
                d_diagnostic_compact_step_reports=bool(d_diagnostic_compact_step_reports),
                calibration_warmup_collector=calibration_warmup_collector,
                votes_emit_enabled=bool(votes_emit_enabled),
                votes_emit_root=votes_emit_root,
                carrier_growth_enabled=bool(carrier_growth_enabled),
                confirmation_envelope=confirmation_envelope,
                event_coded_sparse_vote_authority=bool(event_coded_sparse_vote_authority),
                vote_update_decay_numerator=vote_update_decay_numerator,
                vote_update_decay_denominator=vote_update_decay_denominator,
            )
        if calibration_warmup_collector is not None:
            with phase_progress.phase("calibration_warmup_observations_write"):
                calibration_warmup_collector.write()
        prior_audit_final_reports: dict[str, dict[str, Any]] = {}
        if prior_support_sets:
            with phase_progress.phase("prior_final_audit", step=int(steps_completed)):
                prior_audit_final_reports = audit_prior_support_sets(
                    model,
                    prior_support_sets,
                    final_states,
                    eligible,
                    tok=tok,
                    device=torch_device,
                    phase="prior_final_audit",
                    step=int(steps_completed),
                    total_steps=max(1, int(steps)),
                )
        if not updater_config:
            vote_spec = resolve_probe_vote_update_spec(
                max_abs_per_tensor=int(max_abs_per_tensor),
                confirmation_envelope=confirmation_envelope,
                vote_update_decay_numerator=vote_update_decay_numerator,
                vote_update_decay_denominator=vote_update_decay_denominator,
            )
            envelope = resolve_confirmation_envelope(confirmation_envelope)
            rank_spec = (
                envelope.rank_spec if envelope is not None else default_dry_run_rank_vote_spec()
            )
            updater_config = {
                "rank_vote_spec": rank_spec.to_live_dict(),
                "vote_update_spec": asdict(vote_spec),
                "projection_law": S1_PROJECTION_LAW,
                "vote_law": S1_RANK_BUCKET_VOTE_LAW,
            }
        if slim_receipt_emit:
            checkpoint_payload = {
                "checkpoint_payload_omitted": True,
                "reason": RECEIPT_EMIT_PROFILE_SLIM,
                "schema": BOUNDED_DELTA_CHECKPOINT_SCHEMA_VERSION,
                "dry_run": True,
                "tensor_count": len(final_states),
            }
        else:
            with phase_progress.phase("checkpoint_payload"):
                _maybe_log_checkpoint_states_dump(
                    checkpoint_states_dump,
                    tensor_states=final_states,
                )
                checkpoint_payload = _build_checkpoint_payload_with_phase_telemetry(
                    phase_progress,
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
        if b2b_capture_receipt is not None:
            b2b_capture_receipt["parent_hash_unchanged"] = bool(parent_hash_unchanged)
        total_run_duration_seconds = _timing_duration_seconds(
            run_timing_start,
            torch_device,
            )
        timing_summary = build_timing_summary(
            step_reports=step_reports,
            audit_reports=audit_reports,
            total_run_duration_seconds=total_run_duration_seconds,
        )
        prior_audit_receipt = build_prior_audit_receipt(
            requested_supports=requested_prior_audit_supports,
            support_sets=prior_support_sets,
            start_reports=prior_audit_start_reports,
            final_reports=prior_audit_final_reports,
        )
        b2_retention_receipt = build_b2_retention_receipt(
            requested_supports=requested_b2_retained_supports,
            support_sets=b2_retained_support_sets,
            step_reports=step_reports,
            pc_aux_mode=str(b2_pc_aux_mode),
            parent_consistency_weight=float(b2_parent_consistency_weight),
        )
        b2_full_verdict_receipt = None
        if b2_full_verdict_mode:
            if b2_full_verdict_state is None:
                raise RuntimeError("B2-full verdict mode did not return verdict state")
            if not audit_reports:
                raise RuntimeError("B2-full verdict mode requires at least one target audit")
            final_audit_step = str(
                max(int(step) for step in audit_reports)
            )
            coverage_by_support = b2_full_verdict_state.get("coverage_by_support", {})
            terminal_snapshot = build_b2_full_prior_snapshot(
                snapshot_name="terminal",
                step=int(steps_completed),
                target_audit=audit_reports[final_audit_step],
                coverage_by_support=coverage_by_support,
                start_reports=prior_audit_start_reports,
                current_reports=prior_audit_final_reports,
            )
            b2_full_verdict_receipt = finalize_b2_full_verdict_state(
                b2_full_verdict_state,
                terminal_snapshot=terminal_snapshot,
            )
        front_c_identity_emission_receipt: dict[str, Any] = {"enabled": False}
        if front_c_identity_collector is not None:
            with phase_progress.phase("front_c_identity_artifact", step=int(steps_completed)):
                front_c_identity_emission_receipt = {
                    "enabled": True,
                    **front_c_identity_collector.finalize(
                        audit_reports=audit_reports,
                        prior_audit_start_reports=prior_audit_start_reports,
                        prior_audit_final_reports=prior_audit_final_reports,
                        steps_completed=int(steps_completed),
                        stop_reason=stop_reason,
                    ),
                }
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
            "faulthandler": faulthandler_report,
            "silent_phase_guard": {
                "default_on_with_allow_gpu_launch": True,
                "allow_gpu_launch": bool(allow_gpu_launch),
                "max_silent_phase_seconds": silent_phase_timeout_seconds,
                "last_active_phase_path": str(last_active_phase_path),
                "fail_closed_mechanism": "faulthandler.dump_traceback_later(exit=True)",
            },
            "stdout_liveness": stdout_liveness_receipt,
            "phase_timeout_exemption": phase_timeout_exemption_receipt,
            "phase_budget_interrupt_authority": phase_budget_interrupt_authority,
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
            **eligible_scale_fields,
            "weight_level_init_fidelity": init_fidelity,
            "forward_level_init_fidelity": forward_init_fidelity,
            "steps_requested": int(steps),
            "steps_completed": int(steps_completed),
            "max_steps_hard": int(max_steps_hard),
            "audit_interval": int(audit_interval),
            "stop_on_strict_exact": bool(stop_on_strict_exact),
            "matched_continued_training_horizon_steps": int(
                matched_continued_training_horizon_steps
            ),
            "global_cap_contract": global_cap_contract_receipt,
            "tie_rule_mode": str(tie_rule_mode),
            "science_arm": str(science_arm),
            "target_vote_law": _science_arm_vote_law(str(science_arm)),
            "target_tie_policy_id": _science_arm_tie_policy(str(science_arm)),
            "local_selection_ordering_mode": _science_local_selection_ordering_mode(str(science_arm)),
            "local_selection_ordering_seed": SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
            "aux_vote_law": FIXED_RANK_BUCKET_NON_TARGET_AUX,
            "default_rank_bucket_path_unchanged": str(science_arm) == ARM_A0_RANK_BUCKET_CURRENT,
            "stop_reason": stop_reason,
            "forward_backward_update_executed": bool(steps > 0),
            "step0_optimizer_identity_proof": step0_optimizer_identity_proof,
            "bounded_update_attribution": BOUNDED_UPDATE_ATTRIBUTION,
            "step_reports": step_reports,
            "audit_reports": audit_reports,
            "prior_audit": prior_audit_receipt,
            "b2_retention": b2_retention_receipt,
            "front_c_identity_emission": front_c_identity_emission_receipt,
            "timing_summary": timing_summary,
            "acquisition_trajectory": build_acquisition_trajectory(
                audit_enabled=audit_enabled,
                audit_reports=audit_reports,
                step_reports=step_reports,
                support_cycler_proof=support_cycler_proof,
                audit_interval=int(audit_interval),
                stop_on_strict_exact=bool(stop_on_strict_exact),
                matched_continued_training_horizon_steps=int(
                    matched_continued_training_horizon_steps
                ),
                max_steps_hard=int(max_steps_hard),
                stop_reason=stop_reason,
                timing_summary=timing_summary,
            ),
            "checkpoint_payload": checkpoint_payload,
            "memory": cuda_memory_receipt(torch_device),
            "phase_telemetry": phase_progress.to_dict(),
            "b2b_sequential_capture": b2b_capture_receipt,
            "receipt_emit_profile": str(receipt_emit_profile),
            "persistent_accumulator_w6_byte_packed": bool(persistent_accumulator_w6_byte_packed),
            "persistent_accumulator_w5_byte_packed": bool(persistent_accumulator_w5_byte_packed),
            "dense_accumulator_w7_clip": bool(dense_accumulator_w7_clip),
            "dense_accumulator_w8_clip": bool(dense_accumulator_w8_clip),
            "persistent_accumulator_event_coded_live": bool(
                persistent_accumulator_event_coded_live
            ),
            "event_coded_live_demotion_band": int(event_coded_live_demotion_band),
            "r3_persistent_ledger": build_r3_persistent_ledger_receipt(
                final_states,
                byte_packed_enabled=bool(persistent_accumulator_w6_byte_packed),
            ),
            "persistent_q_ternary_byte_packed": bool(persistent_q_ternary_byte_packed),
            "persistent_q_ternary_base3_codec": bool(persistent_q_ternary_base3_codec),
            "r4_persistent_ledger": build_r4_persistent_ledger_receipt(
                final_states,
                q_packed_enabled=bool(persistent_q_ternary_byte_packed),
                acc_byte_packed_enabled=bool(persistent_accumulator_w6_byte_packed),
            ),
            "r4b_persistent_ledger": build_r4b_persistent_ledger_receipt(
                final_states,
                q_packed_enabled=bool(persistent_q_ternary_byte_packed),
                acc_byte_packed_enabled=bool(persistent_accumulator_w6_byte_packed),
                q_codec_selector=(
                    Q_CODEC_SELECTOR_BASE3
                    if bool(persistent_q_ternary_base3_codec)
                    else "2bit"
                ),
            ),
            "r5_persistent_ledger": build_r5_persistent_ledger_receipt(
                final_states,
                q_packed_enabled=bool(persistent_q_ternary_byte_packed),
                acc_w5_byte_packed_enabled=bool(persistent_accumulator_w5_byte_packed),
            ),
            "r4v_persistent_ledger": build_r4v_persistent_ledger_receipt(
                final_states,
                event_coded_live_enabled=bool(persistent_accumulator_event_coded_live),
            ),
        }
        if bool(event_coded_sparse_vote_authority):
            receipt["event_coded_sparse_vote_authority"] = True
            receipt["control_arm_index_surfaces_skipped_sparse_authority"] = True
            last_step_key = str(int(steps_completed))
            last_step = dict(step_reports.get(last_step_key) or {})
            step_result_body = dict(last_step.get("step_result") or {})
            global_summary = dict(step_result_body.get("global_summary") or {})
            receipt["bounded_delta_global_summary"] = global_summary
            if C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY in global_summary:
                receipt["C8_TRANSIENT_DENSE_COMPUTE_NUMEL"] = int(
                    global_summary[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY]
                )
    
        envelope = resolve_confirmation_envelope(confirmation_envelope)
        if envelope is not None:
            receipt.update(envelope.receipt_fields())
        if slim_receipt_emit:
            assert headroom_wiring_sidecar_path is not None
            receipt["headroom_wiring_sidecar_path"] = str(headroom_wiring_sidecar_path)
            receipt["headroom_wiring_sidecar_schema"] = HEADROOM_WIRING_SIDECAR_SCHEMA_VERSION
        if r7_cap_defer_pressure_instrumentation_enabled:
            assert r7_cap_defer_pressure_sidecar_path is not None
            receipt["r7_cap_defer_pressure_instrumentation_enabled"] = True
            receipt["r7_cap_defer_pressure_sidecar_path"] = str(
                r7_cap_defer_pressure_sidecar_path
            )
        if d_recompute_window_instrumentation_enabled:
            assert d_recompute_window_log_path is not None
            receipt["d_recompute_window_instrumentation_enabled"] = True
            receipt["d_recompute_window_log_path"] = str(d_recompute_window_log_path)
        if event_coded_recompute_window_log_enabled:
            assert d_recompute_window_log_path is not None
            receipt["event_coded_recompute_window_log_enabled"] = True
            receipt["event_coded_recompute_window_log_path"] = str(
                d_recompute_window_log_path
            )
            receipt["d_recompute_window_log_path"] = str(d_recompute_window_log_path)
        if d_live_carrier_snapshot_enabled:
            assert d_live_carrier_snapshot_path is not None
            receipt["d_live_carrier_snapshot_enabled"] = True
            receipt["d_live_carrier_snapshot_path"] = str(d_live_carrier_snapshot_path)
        if d_recompute_window_instrumentation_enabled:
            if d_recompute_selector_manifest is not None:
                receipt["d_recompute_selector_manifest_sha256"] = str(
                    d_recompute_selector_manifest.manifest_sha256
                )
                if d_recompute_selector_manifest_path is not None:
                    receipt["d_recompute_selector_manifest_path"] = str(
                        d_recompute_selector_manifest_path
                    )
        if calibration_warmup_collector is not None:
            receipt["d_recompute_calibration_warmup_observations_path"] = str(
                calibration_warmup_collector.output_path
            )
            receipt["pre_warmup_banked_state_sha256"] = str(parent_hash_before)
            receipt["calibration_discarded_before_measurement"] = True
        if r7_deferred_backlog_carry_enabled:
            receipt["r7_deferred_backlog_carry_enabled"] = True
        if b2_full_verdict_mode:
            assert b2_full_verdict_receipt is not None
            receipt.update(
                {
                    "b2_full_verdict_mode": True,
                    "b2_full_retention_verdict": b2_full_verdict_receipt,
                    "math_a0_coverage_cycles": b2_full_verdict_receipt.get(
                        "math_a0_coverage_cycles"
                    ),
                    "l0b_coverage_cycles": b2_full_verdict_receipt.get(
                        "l0b_coverage_cycles"
                    ),
                }
        )
        if two_tier_carry_w6_enabled:
            receipt.update(
                {
                    "two_tier_carry_w6_enabled": True,
                    "harness_wire_tier_a_index_surface_keys": sorted(
                        TIER_A_PROBE_RECEIPT_INDEX_SURFACE_KEYS
                    ),
                    "grad_proxy_ingress_enabled": True,
                    "grad_proxy_ingress_estimand": (
                        "first_order_grad_proxy_weighted_local_loss_delta"
                    ),
                    "grad_proxy_ingress_population_mode": (
                        POPULATION_MODE_FULL_CROSSING_ELIGIBLE
                    ),
                    "grad_proxy_ingress_crossing_eligible_count_by_step": list(
                        grad_proxy_ingress_crossing_eligible_count_by_step
                    ),
                }
            )
        receipt["harness_wire_cap_window_audit_surface_keys"] = sorted(
            CAP_WINDOW_AUDIT_SURFACE_KEYS
        )
        receipt["cap_window_audit_non_authoritative"] = True
        receipt["cap_window_audit_forbidden_persistent_authority_surfaces"] = list(
            FORBIDDEN_PERSIST_SELECTOR_SURFACES
        )
        receipt_path = scratch_root / "receipt.json"
        receipt["receipt_path"] = str(receipt_path)
        receipt["run_log_path"] = str(run_log_path)
        receipt["cuda_memory_snapshots_jsonl_path"] = str(
            cuda_memory_snapshots_jsonl_path
        )
        receipt["terminal_status"] = build_receipt_terminal_status(
            stop_reason=str(stop_reason),
            steps_completed=int(steps_completed),
            steps_requested=int(steps),
        )
        with phase_progress.phase("receipt_write", path=str(receipt_path)):
            receipt["phase_telemetry"] = phase_progress.to_dict()
            _attach_obmalloc_dedup_evidence(receipt)
            if not slim_receipt_emit:
                compact_probe_receipt_for_banking(receipt)
                compactness_failures = validate_bankable_probe_receipt(receipt)
                if compactness_failures:
                    raise RuntimeError(
                        "probe receipt failed bankable compactness guard: "
                        + "; ".join(compactness_failures)
                    )
            if slim_receipt_emit:
                receipt_text = json.dumps(receipt, separators=(",", ":"), sort_keys=True)
            else:
                receipt_text = json.dumps(receipt, indent=2, sort_keys=True)
            receipt_path.write_text(receipt_text, encoding="utf-8")
        receipt["phase_telemetry"] = phase_progress.to_dict()
        return receipt
    finally:
        _restore_probe_runtime_env(_probe_runtime_env_snapshot)


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
    ap.add_argument(
        "--support-order-seed",
        type=int,
        default=None,
        help=(
            "Default-off support traversal permutation seed. When set, the "
            "identity-full support set is preserved but the cyclic step-batch "
            "trajectory order changes, with ordered and order-invariant hashes "
            "recorded in the receipt."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--eligible-scope", choices=["first-bitlinear", "all-bitlinear"], default="first-bitlinear")
    ap.add_argument(
        "--eligible-module-limit",
        type=int,
        default=None,
        help=(
            "When set with --eligible-scope all-bitlinear, take the first N sorted "
            "BitLinear module keys (prefix of the full eligible set)."
        ),
    )
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--require-q-change", action="store_true")
    ap.add_argument("--max-abs-per-tensor", type=int, default=4096)
    ap.add_argument("--init-fidelity-atol", type=float, default=0.0)
    ap.add_argument("--audit-interval", type=int, default=0)
    ap.add_argument("--stop-on-strict-exact", action="store_true")
    ap.add_argument(
        "--matched-continued-training-horizon-steps",
        type=int,
        default=0,
        help=(
            "When --stop-on-strict-exact is enabled, continue the same training "
            "recipe for this many additional steps after first acquisition "
            "before stopping. Default 0 stops immediately on first acquire."
        ),
    )
    ap.add_argument(
        "--global-cap-contract",
        choices=GLOBAL_CAP_CONTRACT_CHOICES,
        default=GLOBAL_CAP_CONTRACT_OFF,
        help=(
            "Opt-in global cap contract. "
            f"`{GLOBAL_CAP_CONTRACT_OFF}` preserves the legacy local per-tensor path; "
            f"`{C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME}` enables the "
            "banked-faithful long-run global cap used by the c1 tie-rule probe."
        ),
    )
    ap.add_argument(
        "--tie-rule-mode",
        choices=GLOBAL_TIE_RULE_MODES,
        default=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        help=(
            "Tie-rule mode inside an active global-cap contract. "
            "Default exact_global_cap keeps rows[:cap] unchanged; "
            "defer_all_no_backfill defers mixed-class oracle-accepted rows without backfill."
        ),
    )
    ap.add_argument(
        "--science-arm",
        choices=SCIENCE_ARM_CHOICES,
        default=ARM_A0_RANK_BUCKET_CURRENT,
        help=(
            "Default-off optimizer/update-law science arm. A0 preserves the "
            "current rank-bucket/current-ordering path; A1/B/inverted are "
            "diagnostic-only arms for the Step-1 science packet."
        ),
    )
    ap.add_argument(
        "--oracle-screen-mode",
        choices=ORACLE_SCREEN_MODE_CHOICES,
        default=None,
        help=(
            "Optional narrow oracle-screen mode. Keeps the candidate-set-viability "
            "runner off the generic science-arm path."
        ),
    )
    ap.add_argument(
        "--oracle-screen-max-sampled-candidates",
        type=int,
        choices=ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES,
        default=ORACLE_SCREEN_FEASIBILITY_MAX_SAMPLED_CANDIDATES,
        help=(
            "Closed-set oracle-screen sample budget. The runtime max-seconds tier "
            "is derived from this value and pinned fail-closed."
        ),
    )
    ap.add_argument(
        "--two-tier-carry-w6-enabled",
        action="store_true",
        help=(
            "Opt-in harness wire for two-tier carry W6: builds per-step "
            "local_loss_delta_by_key from the activation-credit oracle screen "
            "and passes two_tier_carry_w6_enabled into apply_bounded_delta_vote_step. "
            "Default off preserves legacy receipt surfaces."
        ),
    )
    ap.add_argument(
        "--b2b-sequential-within-tie-band-capture",
        action="store_true",
        help=(
            "Capture pre-update same-vote within_tie_band tables across real "
            "optimizer steps into an NDJSON trace (no oracle_screen_mode)."
        ),
    )
    ap.add_argument(
        "--b2b-sequential-capture-out",
        type=Path,
        default=None,
        help="Output NDJSON trace path for B2b sequential capture.",
    )
    ap.add_argument(
        "--b2b-sequential-min-steps-for-verdict",
        type=int,
        default=50,
        help="Minimum captured optimizer steps before B2c replay verdict is allowed.",
    )
    ap.add_argument(
        "--b2b-sequential-max-sampled-candidates",
        type=int,
        default=PIVOT_MEASUREMENT_REQUIRED_MAX_SAMPLED_CANDIDATES,
        help="Per-step within_tie_band sample budget for B2b capture (must be 32).",
    )
    ap.add_argument(
        "--prior-audit-supports",
        default="",
        help=(
            "Comma-separated read-only prior supports to audit before/after "
            f"bounded steps. Valid: {','.join(B1_PRIOR_AUDIT_SUPPORTS)}. "
            "Default off."
        ),
    )
    ap.add_argument(
        "--b2-retained-supports",
        default="",
        help=(
            "Comma-separated B2 retained true-prior supports that generate "
            "replay-CE veto aux votes during bounded steps. Valid: "
            f"{','.join(B2_RETAINED_SUPPORTS)}. L0c1 remains report-only in B2.0. "
            "Default off."
        ),
    )
    ap.add_argument(
        "--b2-parent-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "B2 retained-support parent-consistency KL weight. When >0, "
            "builds a frozen parent from --parent and generates PC aux votes "
            "on retained supports only; target rows are never parent-KL'd. "
            "Default 0.0."
        ),
    )
    ap.add_argument(
        "--b2-pc-aux-mode",
        choices=B2_PC_AUX_MODES,
        default="telemetry",
        help=(
            "B2 PC aux mode. telemetry records negative PC aux only; veto "
            "also masks PC-negative candidate flips after replay veto. "
            "Default telemetry."
        ),
    )
    ap.add_argument(
        "--b2-full-verdict-mode",
        action="store_true",
        help=(
            "Enable B2-full retention verdict accounting: disjoint retained-support "
            "coverage cycles, first target>=0.90 prior snapshots, covered-stop "
            "snapshot, terminal verdict, and scratch audit summaries. Default off."
        ),
    )
    ap.add_argument("--b2-l0b-batch-size", type=int, default=8)
    ap.add_argument("--b2-math-a0-batch-size", type=int, default=16)
    ap.add_argument(
        "--front-c-identity-emission-artifact",
        type=Path,
        default=None,
        help=(
            "Default-off Front-C path-b identity artifact path. When set, "
            "the probe records cloned CPU learner observations and writes one "
            "self-contained artifact validated by the Stage-1a adapter."
        ),
    )
    ap.add_argument(
        "--front-c-identity-emission-interval",
        type=int,
        default=0,
        help=(
            "Record Front-C identity rows at this step interval, plus step 1 "
            "and the requested terminal step. Audit-interval rows are always "
            "recorded when --audit-interval is set so acquired audit rows cannot "
            "be silently skipped. Default 0 records step 1, audit rows, and terminal."
        ),
    )
    ap.add_argument(
        "--front-c-independent-oracle",
        action="store_true",
        help=(
            "Emit collected-row independent exact-reference oracle receipts for "
            "Front-C identity rows without enabling the legacy full-active-hash "
            "oracle control."
        ),
    )
    ap.add_argument(
        "--checkpoint-states-dump",
        type=Path,
        default=None,
        help=(
            "Default-off diagnostic path for post-step tensor state capture. "
            "Deferred to a later capture gate; no binary artifact is written "
            "in the B1+B2+A1-pre implement slice."
        ),
    )
    ap.add_argument("--max-steps-hard", type=int, default=C2P2_DEFAULT_MAX_STEPS_HARD)
    ap.add_argument("--emit-progress", action="store_true")
    ap.add_argument(
        "--phase-heartbeat-seconds",
        type=float,
        default=None,
        help=(
            "Emit intra-phase stdout heartbeat JSON every N seconds when "
            "--emit-progress is set. Defaults to "
            f"{C2P2_DEFAULT_PHASE_HEARTBEAT_INTERVAL_SECONDS:g}s."
        ),
    )
    ap.add_argument("--phase-timeout-seconds", type=float, default=0.0)
    ap.add_argument("--total-timeout-seconds", type=float, default=0.0)
    ap.add_argument(
        "--phase-timeout-exemption-contract",
        choices=PHASE_TIMEOUT_EXEMPTION_CONTRACT_CHOICES,
        default=PHASE_TIMEOUT_EXEMPTION_CONTRACT_OFF,
        help=(
            "Default-off named exemption for aggregate bounded_steps phase timeout "
            "only. When enabled, nested phases keep the scalar "
            "--phase-timeout-seconds; silent-phase and total timeouts remain "
            "fail-closed. B2b durable launches require positive "
            "--phase-timeout-seconds plus "
            f"{B2B_BOUNDED_STEPS_AGGREGATE_TIMEOUT_EXEMPTION_V1}; "
            "packet-0-style --phase-timeout-seconds 0 is invalid post-WS-C. "
            "Launch packets must declare which first-milestone nested-phase "
            "budget the scalar cap covers."
        ),
    )
    ap.add_argument(
        "--persistent-accumulator-w6-byte-packed",
        action="store_true",
        help=(
            "Default-off checkpoint seam: persist vote accumulators as real "
            "uint8 W6 byte payloads (separate from trainer-boundary W6 flag)."
        ),
    )
    ap.add_argument(
        "--persistent-accumulator-w5-byte-packed",
        action="store_true",
        help=(
            "Default-off treatment-only checkpoint seam: persist vote accumulators "
            "as real uint8 W5 byte payloads (decision-parity lane; mutually exclusive "
            "with --persistent-accumulator-w6-byte-packed)."
        ),
    )
    ap.add_argument(
        "--persistent-q-ternary-byte-packed",
        action="store_true",
        help=(
            "Default-off checkpoint seam: persist q levels as real uint8 "
            "ternary byte payloads (codec selected by base-3 selector; in-step int8 hot path retained)."
        ),
    )
    ap.add_argument(
        "--persistent-q-ternary-base3-codec",
        action="store_true",
        help=(
            "Default-off checkpoint seam: when q byte-packing is enabled, use the "
            "base-3 5-per-byte codec instead of the legacy 2-bit reference codec."
        ),
    )
    ap.add_argument(
        "--receipt-emit-profile",
        choices=list(RECEIPT_EMIT_PROFILE_CHOICES),
        default=RECEIPT_EMIT_PROFILE_FULL,
        help=(
            "Receipt emission profile. Use s3bb_headroom_diagnostic_slim for "
            "aggregate-only receipt.json plus chunked headroom_wiring_sidecar.jsonl."
        ),
    )
    ap.add_argument(
        "--max-silent-phase-seconds",
        type=float,
        default=None,
        help=(
            "Fail-closed active-phase liveness budget. Defaults to "
            f"{C2P2_DEFAULT_GPU_SILENT_PHASE_TIMEOUT_SECONDS:g}s when "
            "--allow-gpu-launch is present; pass 0 only when the launch gate "
            "explicitly authorizes running without this guard."
        ),
    )
    ap.add_argument(
        "--r7-cap-defer-pressure-instrumentation",
        action="store_true",
        help=(
            "Default-off R7 cap/defer pressure instrumentation. When enabled, "
            "append compact per-step chunks to r7_cap_defer_pressure_sidecar.jsonl."
        ),
    )
    ap.add_argument(
        "--d-recompute-window-instrumentation",
        action="store_true",
        help=(
            "Default-off D recompute-window instrumentation. When enabled, append "
            "bounded per-step vote/acc/q snapshots to recompute_window_log.jsonl."
        ),
    )
    ap.add_argument(
        "--event-coded-recompute-window-log",
        action="store_true",
        help=(
            "Default-off event-coded-compatible lightweight recompute_window_log "
            "writer. Emits step + replay_constants rows only (no D lane sampling). "
            "Compatible with --event-coded-sparse-vote-authority; mutually "
            "exclusive with --d-recompute-window-instrumentation."
        ),
    )
    ap.add_argument(
        "--d-recompute-selector-manifest",
        type=Path,
        default=None,
        help=(
            "Optional deterministic stratified selector manifest JSON for D "
            "recompute-window instrumentation. When set, emit uses manifest "
            "keys/lanes instead of the default smallest-numel selector."
        ),
    )
    ap.add_argument(
        "--d-recompute-calibration-warmup-out",
        type=Path,
        default=None,
        help=(
            "Session-boundary path for bounded D-ON calibration warmup observations "
            "JSON. Requires --d-recompute-window-instrumentation. Warmup state is "
            "discarded at session end; parent checkpoint file is not mutated."
        ),
    )
    ap.add_argument(
        "--d-live-carrier-snapshot",
        action="store_true",
        help=(
            "Default-off Slice-5 live carrier byte snapshot instrumentation. When "
            "enabled with event-coded live carrier, append per-step exact carrier "
            "byte surfaces to live_carrier_snapshot.jsonl under scratch_root."
        ),
    )
    ap.add_argument(
        "--d-diagnostic-compact-step-reports",
        action="store_true",
        help=(
            "Default-off D-feasibility receipt compaction. When enabled with "
            "--phase d-recompute-window-feasibility and slim receipt emit, drop "
            "raw per-module tensor_stats arrays before receipt write."
        ),
    )
    ap.add_argument(
        "--votes-emit-enabled",
        action="store_true",
        help=(
            "Default-off per-step votes observables sidecar emission for "
            "dynamics-proof instrumentation."
        ),
    )
    ap.add_argument(
        "--carrier-growth-enabled",
        action="store_true",
        help=(
            "Default-off diagnostic carrier growth summary sidecar. Requires "
            "--votes-emit-enabled; writes to votes_emit/v1/carrier_growth/ only."
        ),
    )
    ap.add_argument(
        "--persistent-accumulator-event-coded-live",
        action="store_true",
        help=(
            "Default-off V4-LIVE event-coded accumulator carrier on the trainer "
            "vote-update path (mutually exclusive with W5/W6 narrow carriers)."
        ),
    )
    ap.add_argument(
        "--event-coded-live-demotion-band",
        type=int,
        default=1,
        help="Demotion band for V4-LIVE event-coded carrier (default 1).",
    )
    ap.add_argument(
        "--event-coded-sparse-vote-authority",
        action="store_true",
        help=(
            "Default-off M4 sparse-authority hot path: SparseVoteEvents construction "
            "+ event_coded_sparse_vote_authority apply (requires "
            "--persistent-accumulator-event-coded-live). OFF = byte-identical dense."
        ),
    )
    ap.add_argument(
        "--r7-deferred-backlog-carry",
        action="store_true",
        help=(
            "Default-off cross-step deferred_backlog carry on the existing "
            "non-candidate global-cap path. Required for R7 age falsifiability."
        ),
    )
    ap.add_argument(
        "--confirmation-envelope",
        choices=[CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24],
        default=None,
        help=(
            "Opt-in confirmation envelope for W7 in-vivo runs. "
            f"{CONFIRMATION_ENVELOPE_CANONICAL_T10_PREREG_V24} wires T=10 + prereg vote bins."
        ),
    )
    ap.add_argument(
        "--dense-accumulator-w7-clip",
        action="store_true",
        help=(
            "Enable W7 clip-only dense accumulator trainer boundary (±63 via effective_clip_bounds). "
            "Mutually exclusive with W5/W6 byte-pack, W8 clip, and V4 event-coded carrier."
        ),
    )
    ap.add_argument(
        "--dense-accumulator-w8-clip",
        action="store_true",
        help=(
            "Enable W8 clip-only dense accumulator trainer boundary (±127 source-clip-lossless). "
            "Mutually exclusive with W5/W6 byte-pack, W7 clip, and V4 event-coded carrier."
        ),
    )
    ap.add_argument(
        "--vote-update-decay-numerator",
        type=int,
        default=None,
        help=(
            "Optional vote-update decay numerator override (default-off: leaves 1/1). "
            "When set with --vote-update-decay-denominator, wires into the live vote_spec "
            "and ReplayConstants emission."
        ),
    )
    ap.add_argument(
        "--vote-update-decay-denominator",
        type=int,
        default=None,
        help=(
            "Optional vote-update decay denominator override (default-off: leaves 1/1). "
            "Must be > 0 when set; fail-closed via VoteUpdateSpec.validate()."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    from scripts.hrm_text_158_code_currency_guard import (
        maybe_enforce_phase3b_probe_import_byte_currency,
    )

    currency_exit = maybe_enforce_phase3b_probe_import_byte_currency()
    if currency_exit is not None:
        return int(currency_exit)
    args = build_arg_parser().parse_args(argv)
    receipt = run_c2p1_probe(
        parent=args.parent,
        parent_sha256=args.parent_sha256,
        scratch_root=args.scratch_root,
        phase=args.phase,
        device=args.device,
        eligible_scope=args.eligible_scope,
        eligible_module_limit=args.eligible_module_limit,
        steps=args.steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        curriculum_seed=args.curriculum_seed,
        support_order_seed=args.support_order_seed,
        init_fidelity_atol=args.init_fidelity_atol,
        require_q_change=args.require_q_change,
        max_abs_per_tensor=args.max_abs_per_tensor,
        audit_interval=args.audit_interval,
        stop_on_strict_exact=args.stop_on_strict_exact,
        matched_continued_training_horizon_steps=args.matched_continued_training_horizon_steps,
        global_cap_contract=args.global_cap_contract,
        tie_rule_mode=args.tie_rule_mode,
        prior_audit_supports=args.prior_audit_supports,
        b2_retained_supports=args.b2_retained_supports,
        b2_parent_consistency_weight=args.b2_parent_consistency_weight,
        b2_pc_aux_mode=args.b2_pc_aux_mode,
        b2_full_verdict_mode=args.b2_full_verdict_mode,
        b2_l0b_batch_size=args.b2_l0b_batch_size,
        b2_math_a0_batch_size=args.b2_math_a0_batch_size,
        front_c_identity_emission_artifact=args.front_c_identity_emission_artifact,
        front_c_identity_emission_interval=args.front_c_identity_emission_interval,
        front_c_independent_oracle=args.front_c_independent_oracle,
        science_arm=args.science_arm,
        oracle_screen_mode=args.oracle_screen_mode,
        oracle_screen_max_sampled_candidates=args.oracle_screen_max_sampled_candidates,
        b2b_sequential_capture_enabled=args.b2b_sequential_within_tie_band_capture,
        b2b_sequential_capture_out=args.b2b_sequential_capture_out,
        b2b_sequential_min_steps_for_verdict=args.b2b_sequential_min_steps_for_verdict,
        b2b_sequential_max_sampled_candidates=args.b2b_sequential_max_sampled_candidates,
        two_tier_carry_w6_enabled=args.two_tier_carry_w6_enabled,
        checkpoint_states_dump=args.checkpoint_states_dump,
        max_steps_hard=args.max_steps_hard,
        emit_progress=args.emit_progress,
        phase_heartbeat_seconds=args.phase_heartbeat_seconds,
        phase_timeout_seconds=args.phase_timeout_seconds,
        total_timeout_seconds=args.total_timeout_seconds,
        max_silent_phase_seconds=args.max_silent_phase_seconds,
        phase_timeout_exemption_contract=args.phase_timeout_exemption_contract,
        enabled=args.enable_bounded_delta_probe,
        allow_gpu_launch=args.allow_gpu_launch,
        receipt_emit_profile=args.receipt_emit_profile,
        persistent_accumulator_w6_byte_packed=bool(
            args.persistent_accumulator_w6_byte_packed
            or persistent_w6_byte_packed_enabled()
        ),
        persistent_accumulator_w5_byte_packed=bool(
            args.persistent_accumulator_w5_byte_packed
            or persistent_w5_byte_packed_enabled()
        ),
        persistent_q_ternary_byte_packed=bool(
            args.persistent_q_ternary_byte_packed
            or persistent_q_ternary_byte_packed_enabled()
        ),
        persistent_q_ternary_base3_codec=bool(
            args.persistent_q_ternary_base3_codec
            or persistent_q_ternary_base3_codec_enabled()
        ),
        r7_cap_defer_pressure_instrumentation_enabled=bool(
            args.r7_cap_defer_pressure_instrumentation
        ),
        r7_deferred_backlog_carry_enabled=bool(args.r7_deferred_backlog_carry),
        d_recompute_window_instrumentation_enabled=bool(
            args.d_recompute_window_instrumentation
        ),
        d_recompute_selector_manifest_path=args.d_recompute_selector_manifest,
        event_coded_recompute_window_log_enabled=bool(
            args.event_coded_recompute_window_log
        ),
        d_diagnostic_compact_step_reports=bool(args.d_diagnostic_compact_step_reports),
        d_recompute_calibration_warmup_out=args.d_recompute_calibration_warmup_out,
        d_live_carrier_snapshot_enabled=bool(args.d_live_carrier_snapshot),
        votes_emit_enabled=bool(args.votes_emit_enabled),
        votes_emit_root=(
            Path(args.scratch_root) if bool(args.votes_emit_enabled) else None
        ),
        carrier_growth_enabled=bool(args.carrier_growth_enabled),
        persistent_accumulator_event_coded_live=bool(
            args.persistent_accumulator_event_coded_live
        ),
        event_coded_live_demotion_band=int(args.event_coded_live_demotion_band),
        event_coded_sparse_vote_authority=bool(args.event_coded_sparse_vote_authority),
        confirmation_envelope=args.confirmation_envelope,
        dense_accumulator_w7_clip=bool(args.dense_accumulator_w7_clip),
        dense_accumulator_w8_clip=bool(args.dense_accumulator_w8_clip),
        vote_update_decay_numerator=args.vote_update_decay_numerator,
        vote_update_decay_denominator=args.vote_update_decay_denominator,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    flush_probe_terminal_artifacts(exit_code=0, flush_reason="normal_completion")
    return 0


def _cli_main(argv: list[str] | None = None) -> int:
    exit_code = 1
    with activation_credit_env_log_capture():
        try:
            exit_code = int(main(argv))
            return exit_code
        except Exception:
            traceback.print_exc()
            flush_probe_terminal_artifacts(
                exit_code=1,
                flush_reason="uncaught_exception",
            )
            return 1
        finally:
            flush_probe_terminal_artifacts(
                exit_code=exit_code,
                flush_reason="cli_finally",
            )


if __name__ == "__main__":
    raise SystemExit(_cli_main())
