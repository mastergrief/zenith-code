"""CUDA oracle-site measurement (IMPLEMENT_v8 extract from cuda_sites).

Named builders invoked under ORACLE_ON; no fused-only oracle path.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
    make_raw_row_observation,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
    fold_native_builder_phases_plus_flush,
    install_capturing_phase_emitter,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
    classify_phase_topology,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
    extract_oracle_from_builder_receipt,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
    _suppress_nested_phase_emissions,
    capture_weighted_grad_by_key,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    events_maps_equal,
    require_canonical_rank_spec,
    two_branch_dense_votes,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    _sparse_vote_events,
    build_trainer_sub2_authority_local_update_receipt,
    derive_trainer_sub2_authority_states,
    resolve_sparse_vote_authority_path,
    select_trainer_eligible_bitlinears,
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


def measure_oracle_at_production_site(
    *,
    gating_row: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    device: torch.device | str,
    site_tag: str,
    production_site: str,
    site_runner: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Oracle row: named builder MUST run ORACLE_ON and return a receipt.

    site_runner, when provided, must return the builder receipt (not None) and
    itself pass sparse_vote_authority_mode=ORACLE_ON. Failures are not swallowed.
    """
    node_id = gating_row
    events: list[dict[str, Any]] = []
    open_starts: dict[str, float] = {}
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    keys = sorted(states.keys())

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

    builder_exc: str | None = None
    builder_receipt: Any = None
    with install_capturing_phase_emitter(node_id, events, open_starts):
        try:
            if site_runner is not None:
                builder_receipt = site_runner()
                if builder_receipt is None:
                    raise ValueError(
                        "site_runner_returned_none: oracle named builder must return receipt"
                    )
            else:
                builder_receipt = build_trainer_sub2_authority_local_update_receipt(
                    model=model,
                    batch=batch,
                    forward_loss_fn=forward_loss_fn,
                    use_ternary_bulk=True,
                    device=device,
                    sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
                )
        except Exception as exc:
            builder_exc = f"{type(exc).__name__}:{exc}"
            builder_receipt = None

    rank_spec = require_canonical_rank_spec()
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key={k: weighted[k] for k in keys},
        q_levels_by_key={k: states[k].q_levels.detach().cpu() for k in keys},
        rank_spec=rank_spec,
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    )

    named = (
        extract_oracle_from_builder_receipt(builder_receipt)
        if builder_receipt is not None
        else {
            "resolved_mode": "",
            "oracle_only": {},
            "builder_receipt_pass": False,
            "events_equal_by_key": {},
            "events_equal_fused_vs_dense_derived": None,
        }
    )
    # Prefer landing subproof / roundtrip proof / local vpp
    if builder_receipt is not None and not named.get("resolved_mode"):
        vpp = dict(getattr(builder_receipt, "vote_projection_proof", None) or {})
        named["resolved_mode"] = str(vpp.get("sparse_vote_authority_mode") or "")
        if vpp.get("oracle_only"):
            named["oracle_only"] = dict(vpp["oracle_only"])
        # landing envelope
        sub = dict(getattr(builder_receipt, "sparse_vote_authority_subproof", None) or {})
        if sub.get("sparse_vote_authority_mode"):
            named["resolved_mode"] = str(sub.get("sparse_vote_authority_mode") or "")
            if sub.get("oracle_only"):
                named["oracle_only"] = dict(sub["oracle_only"])
                named["events_equal_by_key"] = {
                    str(k): bool(v)
                    for k, v in dict(
                        (sub.get("oracle_only") or {}).get("events_equal_by_key") or {}
                    ).items()
                }
                fused_vs = (sub.get("oracle_only") or {}).get(
                    "events_equal_fused_vs_dense_derived"
                )
                if fused_vs is not None:
                    named["events_equal_fused_vs_dense_derived"] = bool(fused_vs)
        # roundtrip
        proof = dict(getattr(builder_receipt, "post_resume_update_proof", None) or {})
        if proof.get("sparse_vote_authority_mode"):
            named["resolved_mode"] = str(proof.get("sparse_vote_authority_mode") or "")
            if proof.get("oracle_only"):
                named["oracle_only"] = dict(proof["oracle_only"])

    # re-extract events after mode fixups
    if builder_receipt is not None:
        re_named = extract_oracle_from_builder_receipt(builder_receipt)
        # merge richer fields from re_named if named was partial
        if re_named.get("resolved_mode"):
            named["resolved_mode"] = re_named["resolved_mode"]
        if re_named.get("oracle_only"):
            named["oracle_only"] = re_named["oracle_only"]
        if re_named.get("events_equal_by_key"):
            named["events_equal_by_key"] = re_named["events_equal_by_key"]
        if re_named.get("events_equal_fused_vs_dense_derived") is not None:
            named["events_equal_fused_vs_dense_derived"] = re_named[
                "events_equal_fused_vs_dense_derived"
            ]
        named["builder_receipt_pass"] = bool(re_named.get("builder_receipt_pass"))

    oracle_mode_on_named_site = bool(
        builder_exc is None
        and builder_receipt is not None
        and named.get("resolved_mode") == "oracle_on"
    )

    # NAMED-ONLY provenance (IMPLEMENT_v10): never rescue via generic path_oracle
    events_equal_by_key = dict(named.get("events_equal_by_key") or {})
    named_map_present = bool(events_equal_by_key)
    fused_vs = named.get("events_equal_fused_vs_dense_derived")
    independent_ok = True
    # independent two-branch recompute is diagnostic only; does NOT fill named map
    for k in keys:
        dens = two_branch_dense_votes(
            weighted[k], states[k].q_levels.detach().cpu(), rank_spec
        )
        derived = _sparse_vote_events(dens["votes"])
        fused = path["sparse_events_by_key"][k]
        if not events_maps_equal(fused, derived):
            independent_ok = False
    prim = (
        named_map_present
        and set(events_equal_by_key.keys()) == set(keys)
        and all(events_equal_by_key.get(k) for k in keys)
    )
    builder_pass = bool(named.get("builder_receipt_pass")) and builder_exc is None
    # missing named map → fixture-contract / S5 false (PLAN_v6 semantics)
    fixture_fail = (
        builder_exc is not None
        or not keys
        or not named_map_present
        or fused_vs is None
    )
    s5 = bool(
        (not fixture_fail)
        and prim
        and independent_ok
        and fused_vs is True
        and oracle_mode_on_named_site
        and builder_pass
    )
    if not events_equal_by_key:
        # schema requires key universe; named-absent → all False (not path_oracle TRUE)
        events_equal_by_key = {str(k): False for k in keys}
        fused_vs_for_metrics = False
    else:
        fused_vs_for_metrics = bool(fused_vs) if fused_vs is not None else False
    obs = make_raw_row_observation(
        gating_row=gating_row,
        device=str(getattr(device, "type", device)),
        measured_surfaces={"s5": s5},
        metrics={
            "site_tag": site_tag,
            "resolved_mode": named.get("resolved_mode") or path.get("resolved_mode"),
            "events_equal_fused_vs_dense_derived": fused_vs_for_metrics,
            "events_equal_by_key": events_equal_by_key,
            "independent_two_branch_recompute_ok": independent_ok,
            "dense_derived_provenance": "two_branch_parallel_dense_vote_derivation",
            "d1_densify_from_sparse_used": False,
            "sparse_vote_authority_mode": "oracle_on",
            "votes_by_key_applied": None,
            "builder_receipt_pass": builder_pass,
            "oracle_mode_on_named_site": oracle_mode_on_named_site,
            "production_site": production_site,
            "builder_exception": builder_exc,
            "named_builder_oracle_only_keys": sorted(
                (named.get("oracle_only") or {}).keys()
            ),
            "named_builder_returned_receipt": builder_receipt is not None,
            "named_events_equal_map_present": named_map_present,
            "path_oracle_fallback_used": False,
        },
        key_universe=keys,
        fixture_contract_raw_fail=fixture_fail,
        synthetic_only=False,
        site_tag=site_tag,
        production_site=production_site,
    )
    _cuda_sync(device)
    return _bind_and_return(obs, events, node_id, flush_work_fn=lambda: _cuda_sync(device))
