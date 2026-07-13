"""Dense-legacy global-cap apply with optional flip-application deferral.

SOLE owner of deferred-mode semantics over the production cap path.
Default ``flip_application_deferred=False`` delegates byte/state/BACKLOG-identically
to ``apply_global_rate_cap_reference``.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    RELEASE_PATH_DEFERRED_W_NO_AUTHORITATIVE_RELEASE,
    RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0,
    DuringWTelemetry,
    WPlus1ReleaseRecord,
    backlog_cardinality,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
    backlog_content_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    GlobalRateCapResult,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    GlobalRateCapTensorResult,
    apply_global_rate_cap_reference,
    select_global_rate_cap_rows,
    tensor_offsets_for_vote_update_states,
    validate_global_tie_rule_mode,
    _deferred_age_summary,
    _row_global_index_sha,
    _rows_by_key,
    _safe_ratio,
    _tensor_sha256,
    DEFERRED_NON_SCOPE,
    CPU_GLUE_NOT_KERNEL_NOTE,
)


def _sha_tensor(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def apply_global_rate_cap_with_optional_flip_deferral(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
    tie_rule_mode: str = EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    contract_name: str | None = None,
    event_coded_sparse_cap_enabled: bool = False,
    flip_application_deferred: bool = False,
) -> GlobalRateCapResult:
    """Apply dense-legacy global cap, optionally deferring flip application.

    When ``flip_application_deferred`` is False, this is an exact delegate to
    ``apply_global_rate_cap_reference`` (default-off / backward compatible).

    When True (during-W law):
    - accumulator carry updates to ``plan.new_acc_i32`` (vote rebuild)
    - q mutation + threshold-residual write-back remain zero
    - authoritative deferred backlog is NOT created/mutated from suppressed crossings
    - selector/cap accepted/deferred rows are recorded as SHADOW-ONLY telemetry
    """

    if event_coded_sparse_cap_enabled:
        raise ValueError(
            "forgotten-accum flip deferral Phase-A covers DENSE_LEGACY only; "
            "event_coded_sparse_cap_enabled is out of scope"
        )

    if not bool(flip_application_deferred):
        # Exact delegate — byte/state/BACKLOG identical to HEAD apply_global_rate_cap_reference.
        return apply_global_rate_cap_reference(
            inputs,
            spec,
            deferred_backlog=deferred_backlog,
            tensor_offsets=tensor_offsets,
            tie_rule_mode=tie_rule_mode,
            contract_name=contract_name,
            event_coded_sparse_cap_enabled=False,
        )

    # --- DURING-W deferred path ---
    spec.validate()
    if not bool(spec.mutate_outputs):
        raise ValueError(
            "flip_application_deferred=True forbids mutate_outputs=False faux-defer; "
            "use the explicit deferred facade path"
        )
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    tie_mode = validate_global_tie_rule_mode(tie_rule_mode)
    rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        inputs,
        spec,
        tensor_offsets=offsets,
        event_coded_sparse_cap_enabled=False,
    )
    # Authoritative backlog: deep-copy INPUT only — do NOT ingest deferred_rows.
    backlog = copy.deepcopy(deferred_backlog or {})
    seed_backlog_sha = backlog_content_sha256(backlog)

    accepted_by_key = _rows_by_key(accepted_rows)
    deferred_by_key = _rows_by_key(deferred_rows)

    tensor_results: list[GlobalRateCapTensorResult] = []
    total_q_changed = 0
    for item in inputs:
        plan = item.plan
        q_out = item.state.q_levels.detach().clone().contiguous()
        # Carry rebuilds from the planned post-vote accumulator; NO residual write-back.
        acc_out = (
            plan.new_acc_i32.flatten()
            .to(torch.int32)
            .view_as(item.state.accumulators)
            .to(torch.int16)
            .contiguous()
        )
        q_changed = int((q_out != item.state.q_levels).sum().item())
        total_q_changed += q_changed
        accepted_set = accepted_by_key.get(item.state_key, set())
        deferred_set = deferred_by_key.get(item.state_key, set())
        shadow_accepted = sorted(int(x) for x in accepted_set)
        shadow_deferred = sorted(int(x) for x in deferred_set)
        stats = dict(plan.stats)
        stats.update(
            {
                "scope": "forgotten_accum_flip_deferral_dense_legacy",
                "global_rate_cap_enabled": True,
                "global_rate_cap_cap": int(spec.cap),
                "flip_application_deferred": True,
                "forgotten_accum_cap_site_branch": DENSE_LEGACY_CAP_SITE_ID,
                "ternary_mutation_enabled": False,
                "flip_count": 0,
                "post_veto_applied_flip_count": 0,
                "threshold_residual_writeback_count": 0,
                "global_rate_cap_would_accept_count": len(shadow_accepted),
                "global_rate_cap_accepted_count": len(shadow_accepted),
                "global_rate_cap_applied_count": 0,
                "global_rate_cap_deferred_count": len(shadow_deferred),
                "global_rate_cap_accepted_indices": list(shadow_accepted),
                "global_rate_cap_deferred_indices": list(shadow_deferred),
                "post_veto_applied_indices": [],
                "shadow_only_selector_cap_telemetry": True,
                "q_changed_count": q_changed,
                "release_path_id": RELEASE_PATH_DEFERRED_W_NO_AUTHORITATIVE_RELEASE,
            }
        )
        tensor_results.append(
            GlobalRateCapTensorResult(
                state_key=item.state_key,
                q_levels=q_out,
                accumulators=acc_out,
                stats=stats,
            )
        )

    if total_q_changed != 0:
        raise RuntimeError("deferred path invariant violated: q_changed_count != 0")
    if backlog_content_sha256(backlog) != seed_backlog_sha:
        raise RuntimeError("deferred path invariant violated: backlog mutated")

    age_summary = _deferred_age_summary(backlog, step=spec.step)
    accepted_count = len(accepted_rows)
    deferred_count = len(deferred_rows)
    step_summary: dict[str, Any] = {
        "global_rate_cap_enabled": True,
        "global_rate_cap_cap": int(spec.cap),
        "global_rate_cap_ordering_mode": spec.normalized_ordering_mode.value,
        "global_rate_cap_ordering_seed": int(spec.ordering_seed),
        "functional_veto_policy": DEFERRED_NON_SCOPE,
        "bad_pressure_drain_policy": DEFERRED_NON_SCOPE,
        "cpu_glue_not_kernel": True,
        "cpu_glue_not_kernel_note": CPU_GLUE_NOT_KERNEL_NOTE,
        "ternary_mutation_enabled": False,
        "ternary_mutation_frozen": True,
        "flip_application_deferred": True,
        "forgotten_accum_cap_site_branch": DENSE_LEGACY_CAP_SITE_ID,
        "global_pre_cap_would_apply_count": len(rows),
        "global_rate_cap_accepted_count": accepted_count,
        "global_rate_cap_applied_count": 0,
        "global_rate_cap_deferred_count": deferred_count,
        "global_rate_cap_saturated": len(rows) > int(spec.cap),
        "global_rate_cap_fill_ratio": _safe_ratio(0, int(spec.cap)),
        "global_deferred_ratio": _safe_ratio(deferred_count, len(rows)),
        "accepted_from_prior_deferred_count": 0,
        "accepted_fresh_count": 0,
        "q_changed_count": 0,
        "threshold_residual_writeback_count": 0,
        "shadow_only_selector_cap_telemetry": True,
        "release_path_id": RELEASE_PATH_DEFERRED_W_NO_AUTHORITATIVE_RELEASE,
        "special_backlog_flush": False,
        "pre_cap_demand_sha256": _row_global_index_sha(rows),
        "global_tie_rule_mode": tie_mode,
        "exact_shadow_full_demand_sha256": _row_global_index_sha(rows),
        "exact_shadow_accepted_sha256": _row_global_index_sha(accepted_rows),
        "exact_shadow_deferred_sha256": _row_global_index_sha(deferred_rows),
        **age_summary,
    }
    if contract_name is not None:
        step_summary["global_rate_cap_contract_name"] = str(contract_name)

    return GlobalRateCapResult(
        tensor_results=tensor_results,
        step_summary=step_summary,
        rows=rows,
        accepted_rows=accepted_rows,
        deferred_rows=deferred_rows,
        deferred_backlog=backlog,
    )


def build_during_W_telemetry(
    *,
    acc_hash_pre: str,
    result: GlobalRateCapResult,
) -> DuringWTelemetry:
    acc_hash_post = hashlib.sha256(
        b"|".join(
            _sha_tensor(tr.accumulators).encode()
            for tr in sorted(result.tensor_results, key=lambda x: x.state_key)
        )
    ).hexdigest()
    q_hash = hashlib.sha256(
        b"|".join(
            _sha_tensor(tr.q_levels).encode()
            for tr in sorted(result.tensor_results, key=lambda x: x.state_key)
        )
    ).hexdigest()
    return DuringWTelemetry(
        acc_hash_pre=acc_hash_pre,
        acc_hash_post=acc_hash_post,
        q_hash=q_hash,
        backlog_hash=backlog_content_sha256(result.deferred_backlog),
        backlog_cardinality=backlog_cardinality(result.deferred_backlog),
        flip_applied_count=int(result.step_summary.get("global_rate_cap_applied_count", 0)),
        threshold_residual_writeback_count=int(
            result.step_summary.get("threshold_residual_writeback_count", 0)
        ),
        crossing_demand_count=int(result.step_summary.get("global_pre_cap_would_apply_count", 0)),
        shadow_accepted_count=int(result.step_summary.get("global_rate_cap_accepted_count", 0)),
        shadow_deferred_count=int(result.step_summary.get("global_rate_cap_deferred_count", 0)),
        cap_site_branch=str(result.step_summary.get("forgotten_accum_cap_site_branch", "")),
        flip_application_deferred=bool(result.step_summary.get("flip_application_deferred")),
    )


def build_W_plus_1_release_record(
    *,
    pre_vote_carry_hash: str,
    result: GlobalRateCapResult,
) -> WPlus1ReleaseRecord:
    q_hash = hashlib.sha256(
        b"|".join(
            _sha_tensor(tr.q_levels).encode()
            for tr in sorted(result.tensor_results, key=lambda x: x.state_key)
        )
    ).hexdigest()
    acc_hash = hashlib.sha256(
        b"|".join(
            _sha_tensor(tr.accumulators).encode()
            for tr in sorted(result.tensor_results, key=lambda x: x.state_key)
        )
    ).hexdigest()
    return WPlus1ReleaseRecord(
        pre_vote_carry_hash=pre_vote_carry_hash,
        crossing_demand_count=int(result.step_summary.get("global_pre_cap_would_apply_count", 0)),
        selected_count=int(result.step_summary.get("global_rate_cap_accepted_count", 0)),
        applied_count=int(result.step_summary.get("global_rate_cap_applied_count", 0)),
        capped_count=int(result.step_summary.get("global_rate_cap_cap", 0)),
        backlogged_count=int(result.step_summary.get("global_rate_cap_deferred_count", 0)),
        post_step_q_hash=q_hash,
        post_step_acc_hash=acc_hash,
        release_path_id=str(
            result.step_summary.get(
                "release_path_id", RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0
            )
        ),
        ordinary_cap=int(result.step_summary.get("global_rate_cap_cap", 0)),
        special_backlog_flush=bool(result.step_summary.get("special_backlog_flush", False)),
    )


__all__ = [
    "apply_global_rate_cap_with_optional_flip_deferral",
    "build_during_W_telemetry",
    "build_W_plus_1_release_record",
]
