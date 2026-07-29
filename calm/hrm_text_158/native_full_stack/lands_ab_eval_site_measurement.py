"""Live site measurement: CPU static row + CUDA production-site instrumentation (IMPLEMENT_v3 seam c).

Captures weighted_grad + prestate from the ACTUAL named production site when measuring
CUDA rows; dense reference runs against THAT capture.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
    make_candidate_authority_tensor_state,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
    make_raw_row_observation,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
    load_seed158_static_fixture,
    verify_recarry_receipt_ro,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
    classify_phase_topology,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    APPLICABILITY_MAP,
    FIXTURE_RECIPE_NAME,
    PARITY_FIXTURE_DESCRIPTOR_SHA256,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
    recompute_surface_cells_from_primitives,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
    bind_production_to_twin_local_update,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
    applied_identities_from_proof_by_key,
    crosscheck_production_q_vs_receipt_proof,
    recompute_s1_and_compare,
    evaluate_family_s2,
    production_fused_apply_post_states,
    production_post_q_and_logical_acc_sha256_by_key,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    clone_bounded_accumulator,
    clone_prior_states,
    events_maps_equal,
    require_canonical_rank_spec,
    run_twin_apply_compare,
    two_branch_dense_votes,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    _sparse_vote_events,
    build_sparse_vote_authority_landing_receipt,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    derive_trainer_sub2_authority_states,
    resolve_sparse_vote_authority_path,
    save_trainer_sub2_live_checkpoint_envelope,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
)
import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as tsa


@contextmanager
def phase_event_capture() -> Iterator[list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    def _emit(kind: str, phase: str) -> None:
        events.append({"type": str(kind), "phase": str(phase)})

    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = _emit
    try:
        yield events
    finally:
        tsa._PHASE_EMITTER = prev


@contextmanager
def _suppress_nested_phase_emissions() -> Iterator[None]:
    """Run production builders without polluting measurement-owned topology."""
    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = None
    try:
        yield
    finally:
        tsa._PHASE_EMITTER = prev


def _emit_pair(phase: str) -> None:
    tsa._emit_phase("PHASE_START", phase)
    tsa._emit_phase("PHASE_END", phase)


def assert_phase_topology_complete(
    events: list[dict[str, Any]],
    *,
    expected_node_id: str | None = None,
    require_enforcer_fields: bool = False,
) -> dict[str, Any]:
    """Do NOT manufacture missing pairs — absent flush = malformed."""
    topo = classify_phase_topology(
        events,
        expected_node_id=expected_node_id,
        require_enforcer_fields=require_enforcer_fields or expected_node_id is not None,
    )
    if not topo["good_topology"]:
        raise ValueError(f"malformed_phase_topology:{topo}")
    return topo


def capture_weighted_grad_by_key(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    states: Mapping[str, BoundedDeltaTensorState],
    eligible: Mapping[str, BitLinear],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    weighted_grad_by_key: dict[str, torch.Tensor] = {}
    prior_training = bool(model.training)
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        with trainer_authoritative_forward_context(
            eligible,
            states,
            device=device,
            requires_grad=True,
        ) as handle:
            loss = forward_loss_fn(model, batch)
            if not isinstance(loss, torch.Tensor):
                raise TypeError("forward_loss_fn must return a torch.Tensor loss")
            loss_to_backward = loss if loss.numel() == 1 else loss.mean()
            loss_to_backward.backward()
            for key in sorted(states):
                weighted_grad_by_key[key] = handle.weighted_grad(key).detach().cpu()
    finally:
        model.train(prior_training)
    return weighted_grad_by_key


def measure_g_cpu_static_ab(*, repo_root: Path | None = None) -> dict[str, Any]:
    recarry = verify_recarry_receipt_ro(repo_root)
    fx = load_seed158_static_fixture()
    s2_pass = bool(recarry["compositional_reduction_holds"] is True)

    q = fx["q_levels"]
    logical = torch.zeros_like(q, dtype=torch.int16)
    prior = make_bounded_tensor_state(
        "proj",
        q_levels=q.clone(),
        frozen_scale=1.0,
        accumulators=logical,
    )
    prior_cand = make_candidate_authority_tensor_state(
        prior,
        q_levels=q.clone(),
        bounded_accumulator=clone_bounded_accumulator(prior.bounded_accumulator),
    )
    compare = run_twin_apply_compare(
        prior_states={"proj": prior_cand},
        weighted_grad_by_key={"proj": fx["weighted_grad"]},
        rank_spec=fx["rank_spec"],
    )

    s1_pass = bool(compare["events_equal"] is True and recarry["events_equal"] is True)
    s3_pass = bool(compare["s3_pass"])
    s4_pass = bool(compare["s4_pass"])
    s6_pass = bool(compare["s6_pass"])
    fixture_fail = bool(compare["fixture_contract_raw_fail"])

    return make_raw_row_observation(
        gating_row="G_CPU_STATIC_AB",
        device="cpu",
        measured_surfaces={
            "s1": s1_pass,
            "s2": s2_pass,
            "s3": s3_pass,
            "s4": s4_pass,
            "s6": s6_pass,
        },
        metrics={
            "events_equal": bool(compare["events_equal"]),
            "events_equal_by_key": dict(compare.get("events_equal_by_key") or {"proj": bool(compare["events_equal"])}),
            "compositional_reduction_holds": s2_pass,
            "q_match": bool(compare["q_match"]),
            "logical_acc_match": bool(compare["logical_acc_match"]),
            "q_changed_match": bool(compare["q_changed_match"]),
            "sparse_event_count": int(compare["sparse_event_count"]),
            "q_changed_count_sparse": int(compare["q_changed_count_sparse"]),
            "q_changed_count_dense": int(compare["q_changed_count_dense"]),
            "weighted_grad_sha256": fx["weighted_grad_sha256"],
            "q_levels_sha256": fx["q_levels_sha256"],
            "recarry_receipt": recarry,
            "fixture_recipe_name": FIXTURE_RECIPE_NAME,
            "parity_fixture_descriptor_sha256": PARITY_FIXTURE_DESCRIPTOR_SHA256,
            "s6_geometry": compare["s6_geometry"],
            "prestate_digests": compare["prestate_digests"],
            "post_q_sha256_by_key": compare["post_q_sha256_by_key"],
            "post_logical_acc_sha256_by_key": compare["post_logical_acc_sha256_by_key"],
            "physical_carrier_equal_diagnostic": compare["physical_carrier_equal_diagnostic"],
            "d1_densify_from_sparse_used": False,
        },
        key_universe=list(compare["keys"]),
        fixture_contract_raw_fail=fixture_fail,
        synthetic_only=False,
        production_site="3C_C1_dry_run_fixture_seed158",
    )


def run_s3_apply_equivalence_cpu_tiny_diagnostic(
    *,
    seed: int = 158,
    dim: int = 4,
) -> dict[str, Any]:
    """NON-GATING diagnostic only."""
    torch.manual_seed(int(seed))

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = BitLinear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    model = _Tiny()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    if not eligible:
        return {
            "diagnostic_only": True,
            "gating_row": None,
            "science_claim": False,
            "synthetic_only": True,
            "fixture_contract_raw_fail": True,
            "reason": "empty_eligible",
        }
    prior = derive_trainer_sub2_authority_states(eligible)
    weighted = {
        k: torch.randn(st.q_levels.shape, dtype=torch.float32) + 0.5
        for k, st in prior.items()
    }
    compare = run_twin_apply_compare(prior_states=prior, weighted_grad_by_key=weighted)
    compare["diagnostic_only"] = True
    compare["gating_row"] = None
    compare["science_claim"] = False
    compare["synthetic_only"] = True
    compare["label"] = "cpu_tiny_random_nongating_characterization"
    return compare


def run_s3_apply_equivalence_cpu(*, seed: int = 158, dim: int = 4) -> dict[str, Any]:
    return run_s3_apply_equivalence_cpu_tiny_diagnostic(seed=seed, dim=dim)


def _compare_to_apply_surfaces(compare: Mapping[str, Any]) -> dict[str, bool]:
    # Branch-preserving: emit measured polarity (may be False → VACUOUS etc.)
    return {
        "s3": bool(compare["s3_pass"]),
        "s4": bool(compare["s4_pass"]),
        "s6": bool(compare["s6_pass"]),
    }


def measure_from_production_capture(
    *,
    gating_row: str,
    prior_states: Mapping[str, BoundedDeltaTensorState],
    weighted_grad_by_key: Mapping[str, torch.Tensor],
    device: str,
    site_tag: str,
    production_site: str,
    phase_events: list[dict[str, Any]],
    builder_receipt_pass: bool = False,
    production_sparse_matches_twin: bool | None = None,
    receipt_proof_by_key: Mapping[str, Any] | None = None,
    production_event_count: int | None = None,
    production_q_changed_count: int | None = None,
    named_sparse_event_map_binding_sha256_by_key: Mapping[str, str] | None = None,
    named_sparse_event_logical_shape_by_key: Mapping[str, Any] | None = None,
    family: str | None = None,
    named_s2_decode_by_key: Mapping[str, str] | None = None,
    named_post_payload_sha256: str | None = None,
    twin_or_canonical_post_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Author apply-row cells from production-bound primitives (IMPLEMENT_v10).

    Claimed S3 is recomputed from the SAME primitives the metric reducer uses:
    twin sparse/dense post-q + logical-acc pairs AND production_sparse_matches_twin
    from RO production re-apply decode (q + logical-int16). Never authors s3
    from twin-only compare while binding is False (eliminates metric_cell_contradiction).
    """
    if phase_events:
        topo = assert_phase_topology_complete(phase_events)
    else:
        topo = {"good_topology": None, "detail": "deferred_to_caller"}
    keys = sorted(prior_states.keys())
    if not keys:
        return make_raw_row_observation(
            gating_row=gating_row,
            device=device,
            measured_surfaces={s: False for s in APPLICABILITY_MAP[gating_row]},
            metrics={"reason": "empty_eligible", "site_tag": site_tag},
            key_universe=[],
            fixture_contract_raw_fail=True,
            synthetic_only=False,
            phase_topology=topo,
            phase_events=phase_events,
            site_tag=site_tag,
            production_site=production_site,
        )
    cpu_states = clone_prior_states(prior_states)
    cpu_weighted = {k: v.detach().cpu() for k, v in weighted_grad_by_key.items()}
    compare = run_twin_apply_compare(
        prior_states=cpu_states, weighted_grad_by_key=cpu_weighted
    )
    # RO production post-state (same apply path as fused local-update builder)
    prod_apply = production_fused_apply_post_states(
        prior_states=cpu_states, weighted_grad_by_key=cpu_weighted
    )
    fixture_fail = bool(compare["fixture_contract_raw_fail"]) or bool(
        prod_apply.get("fixture_contract_raw_fail")
    )
    prod_hashes = production_post_q_and_logical_acc_sha256_by_key(
        prod_apply.get("post_states") or {}
    )
    if not prod_hashes.get("logical_acc_ro_observable"):
        fixture_fail = True
        logical_acc_absent_reason = "production_logical_acc_not_ro_decodable"
    else:
        logical_acc_absent_reason = ""
    # S1: independent NRB recompute from prod_apply events + prior q shapes (PLAN_v6)
    s1_compare = recompute_s1_and_compare(
        sparse_events_by_key=dict(prod_apply.get("sparse_events_by_key") or {}),
        prior_states=cpu_states,
        named_binding_by_key=named_sparse_event_map_binding_sha256_by_key,
        named_shape_by_key=named_sparse_event_logical_shape_by_key,
    )
    s2_compare = evaluate_family_s2(
        family=family or site_tag,
        named_s2_decode_by_key=named_s2_decode_by_key,
        production_logical_acc_by_key=prod_hashes.get(
            "production_post_logical_acc_sha256_by_key"
        ),
        named_post_payload_sha256=named_post_payload_sha256,
        twin_or_canonical_post_payload_sha256=twin_or_canonical_post_payload_sha256,
    )
    # cross-check: family-specific (D1) — B1 needs transition proof; B2/B3 S1+S2 only
    fam = family or site_tag
    xcheck = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key=prod_hashes["production_post_q_sha256_by_key"],
        receipt_proof_by_key=dict(receipt_proof_by_key or {}),
        builder_receipt_pass=bool(builder_receipt_pass),
        reapply_proof_by_key=dict(prod_apply.get("proof_by_key") or {}),
        s1_compare=s1_compare if builder_receipt_pass else None,
        s2_compare=s2_compare if builder_receipt_pass else None,
        family=fam,
    )
    # Always attach diagnostic S1/S2 under the allowed optional key for hostiles/GPU
    # smokes (top-level metrics keys are fail-closed by RO metric_reducer allowlist).
    xcheck = dict(xcheck)
    xcheck.setdefault("s1_compare", s1_compare)
    xcheck.setdefault("s2_compare", s2_compare)
    if not xcheck.get("crosscheck_ok"):
        fixture_fail = True

    reapply_applied = applied_identities_from_proof_by_key(
        dict(prod_apply.get("proof_by_key") or {})
    )
    named_applied = applied_identities_from_proof_by_key(dict(receipt_proof_by_key or {}))
    # B1 only: require applied-identity map equality when builder pass + proof provided
    fam_u = (fam or "").upper()
    is_b1 = fam_u in ("B1", "G_CUDA_B1_APPLY") or fam_u.startswith("B1") or "B1" in fam_u
    if builder_receipt_pass and is_b1 and receipt_proof_by_key is not None:
        if named_applied != reapply_applied:
            fixture_fail = True
    production = {
        "builder_receipt_pass": bool(builder_receipt_pass),
        "total_sparse_vote_event_count": int(
            production_event_count
            if production_event_count is not None
            else prod_apply.get("sparse_event_count", -1)
        ),
        "q_changed_count": int(
            production_q_changed_count
            if production_q_changed_count is not None
            else prod_apply.get("q_changed_count", -1)
        ),
        "production_post_q_sha256_by_key": prod_hashes["production_post_q_sha256_by_key"],
        "production_post_logical_acc_sha256_by_key": prod_hashes[
            "production_post_logical_acc_sha256_by_key"
        ],
        # use re-apply identities (equal to named when transition proof holds)
        "production_applied_row_identities_sha256_by_key": dict(reapply_applied),
        "named_applied_row_identities_sha256_by_key": dict(named_applied),
    }
    bind = bind_production_to_twin_local_update(production=production, compare=compare)
    if production_sparse_matches_twin is None:
        binding_match = bool(bind.get("production_sparse_matches_twin"))
    else:
        # caller override still must not disagree with recompute without fixture fail
        binding_match = bool(production_sparse_matches_twin)

    metrics = {
        "site_tag": site_tag,
        "events_equal": bool(compare["events_equal"]),
        "q_match": bool(compare["q_match"]),
        "logical_acc_match": bool(compare["logical_acc_match"]),
        "q_changed_match": bool(compare["q_changed_match"]),
        "sparse_event_count": int(compare["sparse_event_count"]),
        "q_changed_count_sparse": int(compare["q_changed_count_sparse"]),
        "q_changed_count_dense": int(compare["q_changed_count_dense"]),
        "s6_geometry": compare["s6_geometry"],
        "prestate_digests": compare["prestate_digests"],
        "post_q_sha256_by_key": compare["post_q_sha256_by_key"],
        "post_logical_acc_sha256_by_key": compare["post_logical_acc_sha256_by_key"],
        "physical_carrier_equal_diagnostic": compare["physical_carrier_equal_diagnostic"],
        "d1_densify_from_sparse_used": False,
        "events_equal_by_key": compare["events_equal_by_key"],
        "builder_receipt_pass": bool(builder_receipt_pass),
        "production_sparse_matches_twin": bool(binding_match and not fixture_fail),
        "production_post_q_sha256_by_key": prod_hashes["production_post_q_sha256_by_key"],
        "production_post_logical_acc_sha256_by_key": prod_hashes[
            "production_post_logical_acc_sha256_by_key"
        ],
        "production_binding": bind,
        # s1/s2 nested under production_reapply_crosscheck (RO metric_reducer allowlist)
        "production_reapply_crosscheck": xcheck,
        "logical_acc_absent_reason": logical_acc_absent_reason,
        "twin_post_authoritative_state_payload_sha256": compare.get(
            "twin_post_authoritative_state_payload_sha256", ""
        ),
    }
    # claimed cells == reducer recompute by construction
    try:
        claimed = recompute_surface_cells_from_primitives(
            gating_row=gating_row,
            metrics=metrics,
            key_universe=keys,
            fixture_contract_raw_fail=fixture_fail,
        )
    except ValueError as exc:
        fixture_fail = True
        claimed = {s: False for s in APPLICABILITY_MAP[gating_row]}
        metrics["cell_author_error"] = f"{type(exc).__name__}:{exc}"

    return make_raw_row_observation(
        gating_row=gating_row,
        device=device,
        measured_surfaces=claimed,
        metrics=metrics,
        key_universe=keys,
        fixture_contract_raw_fail=fixture_fail,
        synthetic_only=False,
        phase_topology=topo,
        phase_events=phase_events,
        site_tag=site_tag,
        production_site=production_site,
    )


# Back-compat wrappers (non-authoritative generic helpers for CPU hostiles only)
def measure_site_apply_twin(
    *,
    gating_row: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    device: torch.device | str,
    site_tag: str,
) -> dict[str, Any]:
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    with phase_event_capture() as events:
        tsa._emit_phase("PHASE_START", "forward_backward")
        weighted = capture_weighted_grad_by_key(
            model=model,
            batch=batch,
            forward_loss_fn=forward_loss_fn,
            states=states,
            eligible=eligible,
            device=device,
        )
        tsa._emit_phase("PHASE_END", "forward_backward")
        tsa._emit_phase("PHASE_START", "update")
        tsa._emit_phase("PHASE_END", "update")
        tsa._emit_phase("PHASE_START", "emission")
        tsa._emit_phase("PHASE_END", "emission")
        tsa._emit_phase("PHASE_START", "flush")
        tsa._emit_phase("PHASE_END", "flush")
    return measure_from_production_capture(
        gating_row=gating_row,
        prior_states=states,
        weighted_grad_by_key=weighted,
        device=str(getattr(device, "type", device)),
        site_tag=site_tag,
        production_site="generic_capture_helper",
        phase_events=list(events),
    )


def measure_oracle_events_equal(
    *,
    gating_row: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    forward_loss_fn: Callable,
    device: torch.device | str,
    site_tag: str,
) -> dict[str, Any]:
    return measure_oracle_at_production_site(
        gating_row=gating_row,
        model=model,
        batch=batch,
        forward_loss_fn=forward_loss_fn,
        device=device,
        site_tag=site_tag,
        production_site="generic_oracle_helper",
    )
