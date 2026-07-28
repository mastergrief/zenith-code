"""CUDA production-site measurement (IMPLEMENT_v12).

Native builder phase relay via tsa._PHASE_EMITTER capture (no nested suppress
around builders). Exact transition-proof equality via production_post_state.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
    fold_native_builder_phases_plus_flush,
    install_capturing_phase_emitter,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
    classify_phase_topology,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
    bind_production_to_twin_landing,
    bind_production_to_twin_local_update,
    bind_production_to_twin_roundtrip,
    extract_landing_binding,
    extract_local_update_binding,
    extract_roundtrip_binding,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
    _suppress_nested_phase_emissions,
    capture_weighted_grad_by_key,
    measure_from_production_capture,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    clone_prior_states,
    run_twin_apply_compare,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    build_sparse_vote_authority_landing_receipt,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    derive_trainer_sub2_authority_states,
    save_trainer_sub2_live_checkpoint_envelope,
    select_trainer_eligible_bitlinears,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_oracle_sites import (  # noqa: F401
    measure_oracle_at_production_site,
)


def _bind_and_return(
    obs: dict[str, Any],
    events: list[dict[str, Any]],
    node_id: str,
    *,
    flush_work_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Attach phase fold result; partial/synthesized → fixture-fail science polarity.

    On native_complete: emit measurement-owned flush to memory+JSONL with work_fn
    enclosed (IMPLEMENT_v14 formal flush transport).
    """
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
        emit_measurement_owned_flush,
        fold_native_builder_phases_plus_flush,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
        APPLICABILITY_MAP,
    )

    fold = fold_native_builder_phases_plus_flush(
        events, node_id=node_id, open_starts={}
    )
    events = list(fold["events"])
    if fold.get("needs_measurement_flush"):
        emit_measurement_owned_flush(
            events,
            node_id=node_id,
            open_starts={},
            work_fn=flush_work_fn,
        )
    topo = classify_phase_topology(
        events, expected_node_id=node_id, require_enforcer_fields=True
    )
    out = dict(obs)
    out["phase_topology"] = topo
    out["phase_events"] = events
    out["phase_stream_class"] = fold["phase_stream_class"]
    out["phase_stream_anomaly"] = bool(fold["phase_stream_anomaly"])
    out["phase_events_synthesized"] = bool(fold["phase_events_synthesized"])
    science_block = bool(
        fold["phase_stream_anomaly"]
        or fold["phase_events_synthesized"]
        or topo.get("good_topology") is not True
    )
    if science_block:
        out["fixture_contract_raw_fail"] = True
        try:
            out["measured_surfaces"] = recompute_surface_cells_from_primitives(
                gating_row=str(out.get("gating_row") or node_id),
                metrics=dict(out.get("metrics") or {}),
                key_universe=list(out.get("key_universe") or []),
                fixture_contract_raw_fail=True,
            )
        except Exception:
            row = str(out.get("gating_row") or node_id)
            out["measured_surfaces"] = {
                s: False for s in APPLICABILITY_MAP.get(row, ())
            }
        met = dict(out.get("metrics") or {})
        met["phase_stream_class"] = fold["phase_stream_class"]
        met["phase_stream_anomaly"] = bool(fold["phase_stream_anomaly"])
        met["phase_events_synthesized"] = bool(fold["phase_events_synthesized"])
        out["metrics"] = met
    elif not topo.get("good_topology"):
        raise ValueError(f"malformed_phase_topology:{topo}")
    return out




def _cuda_sync(device: torch.device | str) -> None:
    if str(getattr(device, "type", device)) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_b1_local_update_site(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    device: torch.device | str,
) -> dict[str, Any]:
    node_id = "G_CUDA_B1_APPLY"
    events: list[dict[str, Any]] = []
    open_starts: dict[str, float] = {}
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)

    # Twin WG capture is measurement-owned and must NOT pollute native topology:
    # suppress only this pre-capture, never the named builder.
    with _suppress_nested_phase_emissions():
        weighted = capture_weighted_grad_by_key(
            model=model,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            states=states,
            eligible=eligible,
            device=device,
        )
    _cuda_sync(device)

    # Named builder owns native forward_backward/update/emission via _emit_phase.
    with install_capturing_phase_emitter(node_id, events, open_starts):
        receipt = build_trainer_sub2_authority_local_update_receipt(
            model=model,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
    _cuda_sync(device)
    prod = extract_local_update_binding(receipt)
    proof_by_key = dict(
        (getattr(receipt, "candidate_step_summary", None) or {}).get(
            "candidate_local_update_proof_by_key"
        )
        or {}
    )
    # Evidence reduction OUTSIDE builder native phases
    obs = measure_from_production_capture(
        gating_row=node_id,
        prior_states=states,
        weighted_grad_by_key=weighted,
        device=str(getattr(device, "type", device)),
        site_tag="B1_local_update",
        production_site="build_trainer_sub2_authority_local_update_receipt",
        phase_events=[],
        builder_receipt_pass=bool(prod["builder_receipt_pass"]),
        receipt_proof_by_key=proof_by_key,
        production_event_count=int(prod.get("total_sparse_vote_event_count", -1)),
        production_q_changed_count=int(prod.get("q_changed_count", -1)),
    )
    obs = dict(obs)
    met = dict(obs["metrics"])
    met["receipt_extract"] = {
        "production_post_q_sha256_by_key": prod.get("production_post_q_sha256_by_key"),
        "receipt_logical_acc_ro_observable": prod.get("logical_acc_ro_observable"),
    }
    obs["metrics"] = met
    return _bind_and_return(obs, events, node_id, flush_work_fn=lambda: _cuda_sync(device))


def measure_b2_roundtrip_site(
    *,
    model: torch.nn.Module,
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    forward_output_fn: Callable,
    device: torch.device | str,
) -> dict[str, Any]:
    node_id = "G_CUDA_B2_APPLY"
    events: list[dict[str, Any]] = []
    open_starts: dict[str, float] = {}
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)

    with _suppress_nested_phase_emissions():
        weighted = capture_weighted_grad_by_key(
            model=model,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            states=states,
            eligible=eligible,
            device=device,
        )
    _cuda_sync(device)

    with install_capturing_phase_emitter(node_id, events, open_starts):
        receipt = build_trainer_sub2_authority_roundtrip_receipt(
            model=model,
            fresh_model_fn=fresh_model_fn,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            forward_output_fn=forward_output_fn,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
    _cuda_sync(device)
    prod = extract_roundtrip_binding(receipt)
    compare = run_twin_apply_compare(
        prior_states=clone_prior_states(states),
        weighted_grad_by_key={k: v.detach().cpu() for k, v in weighted.items()},
    )
    bind = bind_production_to_twin_roundtrip(production=prod, compare=compare)
    obs = measure_from_production_capture(
        gating_row=node_id,
        prior_states=states,
        weighted_grad_by_key=weighted,
        device=str(getattr(device, "type", device)),
        site_tag="B2_roundtrip",
        production_site="build_trainer_sub2_authority_roundtrip_receipt",
        phase_events=[],
        builder_receipt_pass=bool(prod["builder_receipt_pass"]),
        production_sparse_matches_twin=bool(bind["production_sparse_matches_twin"]),
        production_event_count=int(prod.get("total_sparse_vote_event_count", -1)),
        production_q_changed_count=int(prod.get("q_changed_count", -1)),
    )
    obs = dict(obs)
    met = dict(obs["metrics"])
    met["production_binding"] = bind
    met["post_update_authoritative_state_payload_sha256"] = prod[
        "post_update_authoritative_state_payload_sha256"
    ]
    met["pre_update_authoritative_state_payload_sha256"] = prod[
        "pre_update_authoritative_state_payload_sha256"
    ]
    met["twin_post_authoritative_state_payload_sha256"] = compare.get(
        "twin_post_authoritative_state_payload_sha256", ""
    )
    obs["metrics"] = met
    return _bind_and_return(obs, events, node_id, flush_work_fn=lambda: _cuda_sync(device))


def measure_b3_landing_site(
    *,
    model: torch.nn.Module,
    fresh_model_fn: Callable[[], torch.nn.Module],
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    forward_output_fn: Callable,
    device: torch.device | str,
) -> dict[str, Any]:
    node_id = "G_CUDA_B3_APPLY"
    events: list[dict[str, Any]] = []
    open_starts: dict[str, float] = {}
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)

    with _suppress_nested_phase_emissions():
        weighted = capture_weighted_grad_by_key(
            model=model,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            states=states,
            eligible=eligible,
            device=device,
        )
        wg_sha = {k: tensor_sha256(v.detach().cpu()) for k, v in weighted.items()}
    _cuda_sync(device)

    with install_capturing_phase_emitter(node_id, events, open_starts):
        p1 = save_trainer_sub2_live_checkpoint_envelope(
            model,
            use_ternary_bulk=True,
            step=0,
            config={"proof": "lands_ab_b3"},
            source_pin="lands_ab_b3",
            epoch=0,
        )
        landing = build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="lands_ab_b3",
            p1_checkpoint=p1,
            p1_envelope_bytes=b"{}",
            fresh_model_fn=fresh_model_fn,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            forward_output_fn=forward_output_fn,
            parity_max_abs_diff_by_site={
                "cache_builder": 0.0,
                "main_kl": 0.0,
                "retained_fallback": 0.0,
            },
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
    _cuda_sync(device)
    prod = extract_landing_binding(landing)
    compare = run_twin_apply_compare(
        prior_states=clone_prior_states(states),
        weighted_grad_by_key={k: v.detach().cpu() for k, v in weighted.items()},
    )
    bind = bind_production_to_twin_landing(
        production=prod, compare=compare, capture_wg_sha_by_key=wg_sha
    )
    obs = measure_from_production_capture(
        gating_row=node_id,
        prior_states=states,
        weighted_grad_by_key=weighted,
        device=str(getattr(device, "type", device)),
        site_tag="B3_landing",
        production_site="build_sparse_vote_authority_landing_receipt",
        phase_events=[],
        builder_receipt_pass=bool(prod["builder_receipt_pass"]),
        production_sparse_matches_twin=bool(bind["production_sparse_matches_twin"]),
        production_event_count=int(prod.get("total_sparse_vote_event_count", -1)),
        production_q_changed_count=int(prod.get("q_changed_count", -1)),
    )
    obs = dict(obs)
    met = dict(obs["metrics"])
    met["production_binding"] = bind
    met["post_update_payload_sha256"] = prod["post_update_payload_sha256"]
    met["p1b_pass_receipt"] = prod["p1b_pass_receipt"]
    met["weighted_grad_capture_sha256_by_key"] = prod[
        "weighted_grad_capture_sha256_by_key"
    ]
    met["twin_post_authoritative_state_payload_sha256"] = compare.get(
        "twin_post_authoritative_state_payload_sha256", ""
    )
    obs["metrics"] = met
    return _bind_and_return(obs, events, node_id, flush_work_fn=lambda: _cuda_sync(device))
