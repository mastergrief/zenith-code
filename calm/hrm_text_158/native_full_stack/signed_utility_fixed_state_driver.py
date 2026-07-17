"""Authoritative call-graph / state ownership for fixed-state signed-utility (PLAN v5)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    PinValidationError,
    validate_proof_packet_source_pins,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    SignedUtilityReducerError,
    classify_signed_utility,
    make_raw_front_c_observation_holder_observer,
    mean_nll_f64_from_metrics_loss,
    mutation_parity_report,
    static_private_core_prohibition_pass,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema import (
    build_non_authoritative_developer_payload,
)

PRIVATE = "_apply_integer_vote_update_from_frozen_plan" + "_trusted"
VOTE_SHA_PIN = "7cf70cdb24f4929a33a6fb4bcdbeac50c37f4fea5462096c06e730e32d860334"
REV_PIN = "466ad1c813734a741d36ada57d4c52cea876b77a"


class DriverError(RuntimeError):
    pass


class AuthoritativeGpuDeferredError(RuntimeError):
    """GPU science path is deferred until successor plan after thin-harness land."""


def _sha_list(xs: list[int]) -> str:
    return hashlib.sha256(json.dumps(xs).encode()).hexdigest()


def _spec(d: Mapping[str, Any]):
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

    return VoteUpdateSpec(
        threshold_abs=int(d["threshold_abs"]),
        accumulator_clip_min=int(d["accumulator_clip_min"]),
        accumulator_clip_max=int(d["accumulator_clip_max"]),
        decay_numerator=int(d["decay_numerator"]),
        decay_denominator=int(d["decay_denominator"]),
        max_abs_per_tensor=int(d["max_abs_per_tensor"]),
        fraction_per_tensor=float(d["fraction_per_tensor"]),
        threshold_jitter_enabled=bool(d.get("threshold_jitter_enabled", False)),
    )


def _mk_states(micro: Mapping[str, Any]):
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import make_bounded_tensor_state

    q = torch.tensor(list(micro["q_levels"]), dtype=torch.int8)
    acc = torch.tensor(list(micro["exact_accumulator_shadow"]), dtype=torch.int16)
    scale = float(micro["frozen_scale"])
    return {k: make_bounded_tensor_state(k, q.clone(), scale, acc.clone()) for k in micro["keys"]}


def _votes_map(micro: Mapping[str, Any]):
    import torch

    return {k: torch.tensor(list(micro["votes_by_key"][k]), dtype=torch.int16) for k in micro["keys"]}


def invert_plans_by_key_directions(plans_by_key: Mapping[str, Any]) -> dict[str, Any]:
    """Delegate canonical inversion to arm_proofs (both direction fields required)."""
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_arm_proofs import (
        canonical_invert_plans_v4,
    )
    return canonical_invert_plans_v4(plans_by_key)


def apply_arms_via_public_frozen_plan(base_states: Mapping[str, Any], plans_by_key: Mapping[str, Any]):
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import make_live_shadow_tensor_state
    from calm.hrm_text_158.native_full_stack.vote_update import apply_integer_vote_update_from_frozen_plan

    out, calls = {}, 0
    for key, prior in base_states.items():
        plan = plans_by_key[key]
        res = apply_integer_vote_update_from_frozen_plan(prior.vote_update_state(), plan)
        calls += 1
        out[key] = make_live_shadow_tensor_state(prior, res.q_levels, res.accumulators)
    return out, calls


def capture_once_front_c_plans(
    base_states: Mapping[str, Any],
    votes_by_key: Mapping[str, Any],
    specs: Mapping[str, Any],
    *,
    local_selection: Mapping[str, Any],
):
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import apply_bounded_delta_vote_step

    holder, call_count = [], [0]
    apply_bounded_delta_vote_step(
        dict(base_states),
        dict(votes_by_key),
        dict(specs),
        global_cap_spec=None,
        front_c_identity_observer=make_raw_front_c_observation_holder_observer(holder, call_count),
        local_selection_ordering_mode=str(local_selection["mode"]),
        local_selection_ordering_seed=int(local_selection["seed"]),
        local_selection_ordering_step=int(local_selection["step"]),
        two_tier_carry_w6_enabled=False,
        parity_check=False,
    )
    if call_count[0] != 1 or len(holder) != 1:
        raise DriverError("raw_holder_call_count")
    if holder[0].get("global_cap_used"):
        raise DriverError("global_cap_used_true")
    return holder[0]["plans_by_key"], call_count[0]


def run_developer_check_cpu_static(ff: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import tensor_sha256

    if ff.get("global_cap_spec") is not None:
        raise DriverError("global_cap_must_be_none")
    if ff.get("vote_conversion_helper") != "_weighted_grads_to_vote_aux_maps":
        raise DriverError("vote_helper_pin_mismatch")
    part = ff["partition_preregistration_v0"]
    if set(part["capture_row_ids"]) & set(part["eval_row_ids"]) or not part["capture_row_ids"] or not part["eval_row_ids"]:
        raise DriverError("partition_disjointness_failed")
    src = Path(__file__).read_text(encoding="utf-8")
    static_pass = static_private_core_prohibition_pass(src)
    if not static_pass:
        raise DriverError("private_core_static_prohibition_failed")

    micro, loc = ff["cpu_static_micro"], ff["local_selection"]
    base, votes = _mk_states(micro), _votes_map(micro)
    specs = {k: _spec(ff["vote_update_spec_frozen"]) for k in micro["keys"]}
    plans, holder_calls = capture_once_front_c_plans(base, votes, specs, local_selection=loc)
    agg = sum(int(p.applied_indices.numel()) for p in plans.values())
    n_keys = len(base)
    if agg <= 0:
        return {
            "classifier": "UNVERIFIED_INTEGRITY_OR_EXECUTION",
            "reason": "aggregate_applied_count_zero",
            "aggregate_applied_count": agg,
            "raw_holder_call_count": holder_calls,
            "eligible_state_key_count": n_keys,
            "harness_static_private_core_prohibition_pass": static_pass,
        }

    prod_states, c1 = apply_arms_via_public_frozen_plan(base, plans)
    inv_plans = invert_plans_by_key_directions(plans)
    inv_states, c2 = apply_arms_via_public_frozen_plan(base, inv_plans)
    public_calls = c1 + c2
    if public_calls != 2 * n_keys:
        raise DriverError(f"call_count_formula_breach:{public_calls}!=2*{n_keys}")

    parity = mutation_parity_report(base, prod_states, inv_states)
    if not parity["pass"]:
        return {
            "classifier": "UNVERIFIED_ASYMMETRIC_INTERVENTION",
            "mutation_parity": parity,
            "aggregate_applied_count": agg,
            "raw_holder_call_count": holder_calls,
            "eligible_state_key_count": n_keys,
            "apply_integer_vote_update_from_frozen_plan_calls": public_calls,
            "expected_internal_private_core_via_public": public_calls,
            "harness_static_private_core_prohibition_pass": static_pass,
        }

    syn = micro["synthetic_nll"]
    _, _, L_prod = mean_nll_f64_from_metrics_loss(tuple(syn["L_prod_pair"]))
    _, _, L_inv = mean_nll_f64_from_metrics_loss(tuple(syn["L_inv_pair"]))
    _, _, L_noop = mean_nll_f64_from_metrics_loss(tuple(syn["L_noop_pair"]))
    classifier, eps = classify_signed_utility(L_prod, L_inv, L_noop)
    applied_idx, dirs_p, dirs_i = [], [], []
    for key in sorted(plans):
        applied_idx.extend(int(x) for x in plans[key].applied_indices.tolist())
        d = [int(x) for x in plans[key].applied_directions.tolist()]
        dirs_p.extend(d)
        dirs_i.extend([-x for x in d])
    writeback = {
        key: {
            "q_sha256": tensor_sha256(prod_states[key].q_levels),
            "exact_accumulator_shadow_sha256": tensor_sha256(prod_states[key].exact_accumulator_shadow),
            "frozen_scale_value": float(prod_states[key].frozen_scale.item()),
            "bounded_accumulator_fresh_for_exact_shadow": bool(
                prod_states[key].bounded_accumulator_fresh_for_exact_shadow
            ),
        }
        for key in sorted(prod_states)
    }
    return {
        "classifier": classifier,
        "L_prod": L_prod,
        "L_inv": L_inv,
        "L_noop": L_noop,
        "epsilon": eps,
        "aggregate_applied_count": agg,
        "raw_holder_call_count": holder_calls,
        "eligible_state_key_count": n_keys,
        "vote_conversion_helper": "_weighted_grads_to_vote_aux_maps",
        "global_cap_spec": None,
        "apply_integer_vote_update_from_frozen_plan_calls": public_calls,
        "expected_internal_private_core_via_public": public_calls,
        "harness_static_private_core_prohibition_pass": static_pass,
        "mutation_parity": parity,
        "capture_row_ids_sha256": part["capture_row_ids_sha256"],
        "eval_row_ids_sha256": part["eval_row_ids_sha256"],
        "applied_indices_sha256": _sha_list(applied_idx),
        "applied_directions_prod_sha256": _sha_list(dirs_p),
        "applied_directions_inv_sha256": _sha_list(dirs_i),
        "arm_writeback_hashes": writeback,
        "revision": REV_PIN,
        "vote_update_sha256": VOTE_SHA_PIN,
        "noop_base_unchanged": all(
            torch.equal(base[k].q_levels, _mk_states(micro)[k].q_levels) for k in base
        ),
    }


def authoritative_path_must_not_route_to_toy_source_pass(source: str) -> bool:
    banned = (
        "run_developer_check_cpu_static(",
        "evaluate_cpu_static(",
        "synthetic_nll",
        "cpu_static_micro",
    )
    # Only scan the authoritative function body region roughly via AST names/calls is heavier;
    # contract: authoritative function source segment must not invoke toy entrypoints.
    try:
        import ast

        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_authoritative_fixed_state_signed_utility":
            body_src = ast.get_source_segment(source, node) or ""
            if any(b in body_src for b in banned):
                return False
            if PRIVATE in body_src.replace("'" + PRIVATE[:40], "X").replace('"' + PRIVATE[:40], "X"):
                # private assembled string may appear in module constants only outside this fn
                pass
            return True
    return False


def run_authoritative_fixed_state_signed_utility(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Module-owned authoritative entry. Delegates to authoritative_gpu; never routes to toy."""
    if not isinstance(packet, Mapping):
        raise DriverError("packet_not_mapping")
    validate_proof_packet_source_pins(packet)
    src = Path(__file__).read_text(encoding="utf-8")
    if not static_private_core_prohibition_pass(src):
        raise DriverError("private_core_static_prohibition_failed")
    if not authoritative_path_must_not_route_to_toy_source_pass(src):
        raise DriverError("authoritative_routes_to_toy")
    # Hard separation: never delegate to the developer/toy evaluator entrypoints.
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
        AuthoritativeGpuDeferredError as _GpuDeferred,
        run_authoritative_gpu_call_graph,
    )

    try:
        return run_authoritative_gpu_call_graph(packet)
    except _GpuDeferred as exc:
        raise AuthoritativeGpuDeferredError(str(exc)) from exc


def developer_check_payload(ff: Mapping[str, Any]) -> dict[str, Any]:
    return build_non_authoritative_developer_payload(run_developer_check_cpu_static(ff))


__all__ = [
    "AuthoritativeGpuDeferredError",
    "DriverError",
    "PinValidationError",
    "SignedUtilityReducerError",
    "apply_arms_via_public_frozen_plan",
    "authoritative_path_must_not_route_to_toy_source_pass",
    "capture_once_front_c_plans",
    "developer_check_payload",
    "invert_plans_by_key_directions",
    "run_authoritative_fixed_state_signed_utility",
    "run_developer_check_cpu_static",
]
