"""Thin authoritative GPU orchestrator (D2c3 S4)."""
from __future__ import annotations
import ast, hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_arm_proofs import (
    ArmProofError, arm_hash_map, calibrate_capture_vs_public, canonical_invert_plans_v4,
    hash_current_weights_tensors, mutable_arms_for_isolation, run_isolation_sentinel_checkpoint,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_eval_contract import deterministic_eval_contract
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_integrity_proofs import (
    INTEGRITY, IntegrityProofError, terminal_precedence_classify, untouched_sentinel_report,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    ESTIMAND_NAME, LegalSubsetError, MAX_AUTHORITATIVE_RESULT_BYTES, MAX_COMPACT_TELEMETRY_BYTES,
    assert_compact_json_nbytes, characterize_plans_bidirectional_legal, enforce_legal_subset_support_floors,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_partition_leakage import (
    PartitionLeakageError, compute_partition_leakage_compact, surface_values,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_phase_telemetry import (
    PhaseBudgetBreach, PhaseBudgetClock,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    PinValidationError, rehash_path, require_formal_source_pin_basenames,
    require_head_equals_upstream_pin, validate_proof_packet_source_pins,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import (
    classify_signed_utility, epsilon_from_noop, make_raw_front_c_observation_holder_observer,
    mean_nll_f64_from_metrics_loss, mutation_parity_report, static_private_core_prohibition_pass,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema import (
    REQUIRED_PHASE_MARKER_NAMES, SCHEMA_PREFLIGHT, SCHEMA_SCIENCE, SCHEMA_UNVERIFIED,
    validate_authoritative_result_payload_v3,
)
CALL_GRAPH_STEPS_V6 = (
    "parse_packet_live_rehash_pins", "parent_sha_pre_materialize", "rebuild_support_batches_bind_leakage",
    "fork_clones_from_materialized_base", "causal_storage_isolation_baseline", "capture_backward_vote",
    "calibrate_capture_vs_public_apply", "apply_writeback_prod_inv_noop_parity", "post_apply_isolation_sentinels",
    "eval_nll_three_arm_plus_noop_repeat", "post_eval_isolation_sentinels", "emit_in_memory_payload")
ARM_FORK_NAMES = ("base", "prod", "inv", "noop", "calibration_shadow", "parent", "capture_disposable")
INVERT_DIR_FIELDS = ("applied_directions", "replay_veto_directions")
FORMAL_PHASE_BUDGETS = {"MATERIALIZE": 120.0, "CAPTURE_BACKWARD_VOTE": 180.0, "THREE_ARM_APPLY_WRITEBACK": 60.0,
                        "THREE_ARM_EVAL_NLL": 480.0, "EMIT_FLUSH": 30.0}
SMOKE_PHASE_BUDGETS = {"MATERIALIZE": 120.0, "CAPTURE": 120.0, "CALIBRATE_EVAL": 60.0}
SMOKE_SUBPHASE_TO_ENVELOPE = {"MATERIALIZE": "MATERIALIZE", "CAPTURE_BACKWARD_VOTE": "CAPTURE",
    "THREE_ARM_APPLY_WRITEBACK": "CALIBRATE_EVAL", "THREE_ARM_EVAL_NLL": "CALIBRATE_EVAL", "EMIT_FLUSH": "CALIBRATE_EVAL"}
class AuthoritativeGpuError(RuntimeError): pass
class AuthoritativeGpuDeferredError(RuntimeError): pass
def isolate_fork_arm_state(state: Any):
    if getattr(state, "event_coded_live_carrier", None) is not None:
        raise AuthoritativeGpuError("unsupported_event_coded_live_carrier_in_signed_utility_fork")
    from dataclasses import replace as _dc_replace
    from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_arm_ops import clone_f_in_memory
    cloned = clone_f_in_memory(state)
    if cloned.event_coded_live_carrier is not None:
        raise AuthoritativeGpuError("unsupported_event_coded_live_carrier_in_signed_utility_fork")
    return _dc_replace(cloned, bounded_accumulator=_dc_replace(cloned.bounded_accumulator))
@dataclass
class AuthoritativeGpuHooks:
    materialize: Callable[[Mapping[str, Any]], Any]
    rebuild_support_batches: Callable[[Any], list[dict[str, Any]]]
    leakage_report: Callable[[list[dict[str, Any]]], dict[str, Any]]
    fork_arm_states: Callable[[Any], dict[str, Any]]
    capture_plans: Callable[[Any, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any], int]]
    public_apply: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Mapping[str, Any], int]]
    invert_plans: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    eval_arm_nll: Callable[[str, Mapping[str, Any], Any, Sequence[Any]], tuple[float, int, float, str]]
    phase_budgets: Mapping[str, float] = field(default_factory=lambda: dict(FORMAL_PHASE_BUDGETS))
def authoritative_gpu_source_must_not_route_to_toy(source: str) -> bool:
    banned = ("run_developer_check_cpu_static(", "evaluate_cpu_static(", "synthetic_nll", "cpu_static_micro")
    try: tree = ast.parse(source)
    except SyntaxError: return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_authoritative_gpu_call_graph":
            return not any(b in (ast.get_source_segment(source, node) or "") for b in banned)
    return False
def call_graph_steps() -> Sequence[str]: return CALL_GRAPH_STEPS_V6
def phase_budgets_for_packet(packet: Mapping[str, Any]) -> dict[str, float]:
    return dict(SMOKE_PHASE_BUDGETS if packet.get("smoke_mode") else FORMAL_PHASE_BUDGETS)
def resolve_capture_device_mode(packet: Mapping[str, Any], device: str) -> str:
    if str(device).startswith("cuda"): return "device_resident"
    if packet.get("allow_cpu_legacy_eval") is True: return "cpu_legacy"
    raise AuthoritativeGpuError("cpu_eval_requires_allow_cpu_legacy_eval")
def _validated(payload: dict[str, Any]) -> dict[str, Any]:
    validate_authoritative_result_payload_v3(payload); return payload
def _preflight(reason, stage, observed, expected, route):
    return _validated({"schema": SCHEMA_PREFLIGHT, "classifier": INTEGRITY, "failed_stage": stage,
                       "observed": observed, "expected": expected, "ts_utc": "now", "reason": reason, "route": list(route)})
def _unverified(reason, stage, markers, parent_pre, compact, route, *, asymmetric=False):
    return _validated({"schema": SCHEMA_UNVERIFIED,
                       "classifier": "UNVERIFIED_ASYMMETRIC_INTERVENTION" if asymmetric else INTEGRITY,
                       "reason": reason, "failed_stage": stage, "phase_markers": dict(markers),
                       "parent_sha256_pre": parent_pre, "compact_diagnostics": dict(compact), "route": list(route)})
def _validate_packet_pins(packet: Mapping[str, Any]) -> None:
    validate_proof_packet_source_pins(packet); require_formal_source_pin_basenames(packet)
    mode = str(packet.get("pin_mode") or ("smoke" if packet.get("smoke_mode") else "formal"))
    if mode == "cpu_static_di": return
    expected, root = packet.get("expected_head"), packet.get("repo_root")
    head_pin = (packet.get("source_pins") or {}).get("head")
    if isinstance(head_pin, Mapping):
        expected = expected or head_pin.get("sha256") or head_pin.get("sha")
    if mode in {"formal", "smoke"}:
        if not root or not expected: raise PinValidationError(f"{mode}_head_or_repo_root_missing")
        require_head_equals_upstream_pin(root, str(expected))

def build_live_hooks(packet: Mapping[str, Any], progress_sink: Callable[[str, str, str | None], None] | None = None) -> AuthoritativeGpuHooks:
    import torch
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import materialize_run_arms_live_bundle
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_driver import apply_arms_via_public_frozen_plan
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step, authoritative_forward_context, canonical_acquisition_rank_vote_spec)
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        SCIENCE_LOCAL_SELECTION_ORDERING_SEED, _compute_ce_weighted_grads, _weighted_grads_to_vote_aux_maps,
        build_identity_full_support_batches, resolve_probe_vote_update_spec)
    st: dict[str, Any] = {}; device = str(packet.get("device", "cuda:0")); cap_mode = resolve_capture_device_mode(packet, device)
    def _cap(step, fn):
        if progress_sink is not None: progress_sink(step, "start", None)
        try: out = fn()
        except Exception as exc:
            if progress_sink is not None: progress_sink(step, "error", f"{type(exc).__name__}:{exc}")
            raise
        if progress_sink is not None: progress_sink(step, "done", None)
        return out
    def materialize(p):
        parent = p["parent_checkpoint"]
        return materialize_run_arms_live_bundle(
            parent_path=parent["absolute_path"], expected_parent_sha256=parent["sha256"],
            device=device, eligible_scope="all-bitlinear", batch_size=32, curriculum_seed=43)
    def rebuild(bundle):
        batches, _ = build_identity_full_support_batches(
            tok=bundle.tok, max_len=int(getattr(bundle.cfg, "max_seq_len", 64) or 64),
            batch_size=32, curriculum_seed=43, support_order_seed=None, device=torch.device(device))
        sizes = [len(b["metadata"]["row_ids"]) for b in batches]
        if len(batches) != 3 or sizes != [32, 32, 26]: raise AuthoritativeGpuError(f"support_batch_shape:{sizes}")
        for b in batches:
            meta = b["metadata"]; meta["normalized_prompt_hashes"] = surface_values(b, "normalized_prompt_hash")
            meta["normalized_target_hashes"] = surface_values(b, "normalized_target_hash")
            meta["response_token_hashes"] = surface_values(b, "response_token_hash")
        st["batches"] = batches; return batches
    def fork(bundle):
        base = bundle.tensor_states
        return {n: {str(k): isolate_fork_arm_state(v) for k, v in sorted(base.items())} for n in ARM_FORK_NAMES}
    def capture(bundle, arms):
        model, eligible, cap = bundle.model, bundle.eligible_modules, arms["capture_disposable"]
        def _grads():
            model.train(); torch.manual_seed(43); extras = model.compute_train_extra_args(0, 1)
            return _compute_ce_weighted_grads(
                model, st["batches"][0]["batch"], cap, eligible, device=torch.device(device), extras=extras)
        grads, _loss, _m = _cap("CAP_COMPUTE_GRADS", _grads)
        votes, _moves = _cap("CAP_VOTE_AUX", lambda: _weighted_grads_to_vote_aux_maps(
            grads, cap, rank_spec=canonical_acquisition_rank_vote_spec()))
        spec = _cap("CAP_SPEC", lambda: resolve_probe_vote_update_spec(
            max_abs_per_tensor=4096, confirmation_envelope="canonical_t10_prereg_v24",
            vote_update_decay_numerator=None, vote_update_decay_denominator=None))
        def _apply():
            holder, cc = [], [0]
            step = apply_bounded_delta_vote_step(
                dict(cap), dict(votes), {k: spec for k in cap}, global_cap_spec=None,
                front_c_identity_observer=make_raw_front_c_observation_holder_observer(holder, cc),
                two_tier_carry_w6_enabled=False, parity_check=False, replay_ce_veto_votes_by_key=None,
                replay_ce_veto_moves_by_key=None, pc_aux_votes_by_key=None, pc_aux_moves_by_key=None,
                pc_aux_mode="telemetry", local_selection_ordering_mode="current_abs_new_acc_then_index",
                local_selection_ordering_seed=int(SCIENCE_LOCAL_SELECTION_ORDERING_SEED), local_selection_ordering_step=0)
            return step, holder, cc
        step, holder, cc = _cap("CAP_APPLY_VOTE_STEP", _apply)
        def _post():
            if cc[0] != 1 or len(holder) != 1: raise AuthoritativeGpuError("raw_holder_call_count")
            plans = holder[0]["plans_by_key"]
            return plans, (getattr(step, "tensor_states", None) or holder[0].get("tensor_states")), cc[0]
        return _cap("CAP_POST_RETURN_HOLDER_VALIDATION", _post)
    def eval_arm(arm, states, bundle, eval_batches):
        model, eligible = bundle.model, bundle.eligible_modules
        model.eval(); num, den = 0.0, 0
        with torch.no_grad(), deterministic_eval_contract(device=device):
            with authoritative_forward_context(
                eligible, states, device=torch.device(device), requires_grad=False, capture_device_mode=cap_mode,
            ) as ctx:
                w_hash = hash_current_weights_tensors(ctx.current_weights)
                extras = model.compute_train_extra_args(0, 1)
                for batch in eval_batches:
                    _c, _loss, metrics = model(None, dict(batch), **extras)
                    n, d, _m = mean_nll_f64_from_metrics_loss(tuple(metrics["loss"]))
                    num += float(n); den += int(d)
        if den < 1: raise AuthoritativeGpuError(f"nll_denominator_lt_1:{arm}")
        return num, den, num / float(den), w_hash
    return AuthoritativeGpuHooks(
        materialize=materialize, rebuild_support_batches=rebuild, leakage_report=compute_partition_leakage_compact,
        fork_arm_states=fork, capture_plans=capture, public_apply=apply_arms_via_public_frozen_plan,
        invert_plans=canonical_invert_plans_v4, eval_arm_nll=eval_arm, phase_budgets=phase_budgets_for_packet(packet))

def run_authoritative_gpu_call_graph(packet: Mapping[str, Any], *, hooks: AuthoritativeGpuHooks | None = None) -> dict[str, Any]:
    if not isinstance(packet, Mapping): raise AuthoritativeGpuError("packet_not_mapping")
    if packet.get("authoritative_deferred", True) is True:
        raise AuthoritativeGpuDeferredError("authoritative_gpu_deferred_until_successor_plan_after_thin_harness_land")
    route: list[str] = []; markers = {n: False for n in REQUIRED_PHASE_MARKER_NAMES}; parent_pre = None
    smoke = bool(packet.get("smoke_mode")); h = hooks if hooks is not None else build_live_hooks(packet)
    budgets = phase_budgets_for_packet(packet); clock = PhaseBudgetClock(budgets)
    try:
        _validate_packet_pins(packet); route.append("parse_packet_live_rehash_pins")
    except PinValidationError as exc:
        return _preflight(str(exc), "source_pins", {"error": str(exc)}, {"source_pins": "formal+head"}, route)
    src = Path(__file__).read_text(encoding="utf-8")
    if not static_private_core_prohibition_pass(src) or not authoritative_gpu_source_must_not_route_to_toy(src):
        return _preflight("static_source_guard", "source_static", {}, {}, route)
    try:
        clock.begin("MATERIALIZE"); markers["PHASE_MATERIALIZE_BEGIN"] = True
        parent = packet.get("parent_checkpoint") or {}
        if "absolute_path" in parent and "sha256" in parent:
            parent_pre = rehash_path(parent["absolute_path"])
            if parent_pre != str(parent["sha256"]):
                return _preflight("parent_sha_mismatch", "parent_hash", {"got": parent_pre}, {"sha256": parent["sha256"]}, route)
        route.append("parent_sha_pre_materialize"); bundle = h.materialize(packet); batches = h.rebuild_support_batches(bundle)
        leak = h.leakage_report(batches)
        if leak.get("pass") is not True:
            markers["PHASE_MATERIALIZE_END"] = True
            return _unverified("leakage_overlap", "partition", markers, parent_pre, leak, route)
        route.append("rebuild_support_batches_bind_leakage"); arms = h.fork_arm_states(bundle)
        if "capture_disposable" not in arms:
            return _unverified("capture_disposable_missing", "fork", markers, parent_pre, {}, route)
        route.append("fork_clones_from_materialized_base")
        run_isolation_sentinel_checkpoint(mutable_arms_for_isolation(arms), base=arms.get("base"), label="baseline")
        untouched_keys = ("base", "parent", "prod", "inv", "noop", "calibration_shadow")
        before = arm_hash_map(arms, untouched_keys); route.append("causal_storage_isolation_baseline")
        clock.end("MATERIALIZE"); markers["PHASE_MATERIALIZE_END"] = True
        clock.begin("CAPTURE" if smoke else "CAPTURE_BACKWARD_VOTE"); markers["PHASE_CAPTURE_BACKWARD_VOTE_BEGIN"] = True
        plans, capture_states, holder_calls = h.capture_plans(bundle, arms); route.append("capture_backward_vote")
        untouched_sentinel_report(before=before, after=arm_hash_map(arms, untouched_keys), required_unchanged=untouched_keys)
        if holder_calls != 1:
            return _unverified("holder_call_count", "capture", markers, parent_pre, {"calls": holder_calls}, route)
        cal_pre = arm_hash_map(arms, ("calibration_shadow",))
        before_cal = arm_hash_map(arms, ("base", "parent", "prod", "inv", "noop"))
        calib, _ = h.public_apply(arms["calibration_shadow"], plans)
        cal_post = arm_hash_map(arms, ("calibration_shadow",))
        untouched_sentinel_report(before=cal_pre, after=cal_post, required_unchanged=("calibration_shadow",))
        untouched_sentinel_report(before=before_cal, after=arm_hash_map(arms, ("base", "parent", "prod", "inv", "noop")),
                                  required_unchanged=("base", "parent", "prod", "inv", "noop"))
        if not isinstance(capture_states, Mapping):
            return _unverified("capture_states_missing", "calibration", markers, parent_pre, {}, route)
        cal = calibrate_capture_vs_public(capture_states, calib)
        if cal.get("pass") is not True:
            return _unverified("calibration_state_mismatch", "calibration", markers, parent_pre, cal, route)
        cal["calibration_input_hash_pre"] = cal_pre["calibration_shadow"]
        cal["calibration_input_hash_post"] = cal_post["calibration_shadow"]
        cal["calibration_input_unchanged"] = True
        route.append("calibrate_capture_vs_public_apply")
        agg = sum(int(getattr(p, "applied_indices").numel()) for p in plans.values())
        if agg <= 0:
            markers["PHASE_CAPTURE_BACKWARD_VOTE_END"] = True
            return _unverified("aggregate_applied_count_zero", "capture", markers, parent_pre, {"agg": agg}, route)
        capture_only = packet.get("capture_only_diagnostic") is True
        cap_phase = "CAPTURE" if smoke else "CAPTURE_BACKWARD_VOTE"
        try:
            clock.end(cap_phase)
        except PhaseBudgetBreach as exc:
            if not capture_only:
                raise
            markers["PHASE_CAPTURE_BACKWARD_VOTE_END"] = True
            markers["CAPTURE_ONLY_DIAGNOSTIC_TERMINAL"] = True
            markers["calibration_included"] = True
            return _unverified(
                str(exc), "capture_only_diagnostic", markers, parent_pre,
                {"capture_only_diagnostic": True, "calibration_included": True,
                 "CAPTURE_ONLY_DIAGNOSTIC_TERMINAL": True,
                 "phase_clock_elapsed_seconds": float(exc.elapsed_s),
                 "phase_budget_seconds": float(exc.budget_s),
                 "cap_progress_envelope_seconds_localization_only": None,
                 "cap_steps_seconds_localization_only": None,
                 "m1_branch_id": packet.get("m1_branch_id"),
                 "PHASE_THREE_ARM_APPLY_WRITEBACK_BEGIN": False,
                 "PHASE_THREE_ARM_EVAL_NLL_BEGIN": False,
                 "THREE_ARM_APPLY_WRITEBACK_BEGIN": False, "EVAL_NLL_BEGIN": False}, route)
        markers["PHASE_CAPTURE_BACKWARD_VOTE_END"] = True
        if capture_only:
            markers["CAPTURE_ONLY_DIAGNOSTIC_TERMINAL"] = True
            markers["calibration_included"] = True
            pe = clock.receipt["phases"].get(cap_phase) or {}
            return _unverified(
                f"capture_only_diagnostic_complete:{cap_phase}:{float(pe.get('elapsed_s', 0.0)):.6f}",
                "capture_only_diagnostic", markers, parent_pre,
                {"capture_only_diagnostic": True, "calibration_included": True,
                 "CAPTURE_ONLY_DIAGNOSTIC_TERMINAL": True,
                 "phase_clock_elapsed_seconds": pe.get("elapsed_s"),
                 "phase_budget_seconds": pe.get("budget_s"),
                 "cap_progress_envelope_seconds_localization_only": None,
                 "cap_steps_seconds_localization_only": None,
                 "m1_branch_id": packet.get("m1_branch_id"),
                 "PHASE_THREE_ARM_APPLY_WRITEBACK_BEGIN": False,
                 "PHASE_THREE_ARM_EVAL_NLL_BEGIN": False,
                 "THREE_ARM_APPLY_WRITEBACK_BEGIN": False, "EVAL_NLL_BEGIN": False}, route)
        if smoke: clock.begin("CALIBRATE_EVAL")
        else: clock.begin("THREE_ARM_APPLY_WRITEBACK")
        markers["PHASE_THREE_ARM_APPLY_WRITEBACK_BEGIN"] = True
        before_apply = arm_hash_map(arms, ("base", "parent", "noop"))
        try: filtered_plans, legal_subset = characterize_plans_bidirectional_legal(arms["prod"], plans)
        except LegalSubsetError as exc:
            markers["PHASE_THREE_ARM_APPLY_WRITEBACK_END"] = True
            return _unverified(str(exc), "legal_subset", markers, parent_pre, {"error": str(exc)}, route)
        assert_compact_json_nbytes(legal_subset, limit=MAX_COMPACT_TELEMETRY_BYTES, label="legal_subset")
        route.append("legal_subset_filter")
        try: enforce_legal_subset_support_floors(legal_subset)
        except LegalSubsetError as exc:
            markers["PHASE_THREE_ARM_APPLY_WRITEBACK_END"] = True
            return _unverified(str(exc), "legal_subset", markers, parent_pre, legal_subset, route)
        prod, c1 = h.public_apply(arms["prod"], filtered_plans)
        inv, c2 = h.public_apply(arms["inv"], h.invert_plans(filtered_plans))
        noop = arms["noop"]; arms["prod"], arms["inv"], arms["noop"] = prod, inv, noop
        untouched_sentinel_report(before=before_apply, after=arm_hash_map(arms, ("base", "parent", "noop")),
                                  required_unchanged=("base", "parent", "noop"))
        if c1 + c2 != 2 * len(arms["base"]):
            return _unverified("call_count_formula", "apply", markers, parent_pre, {"c1": c1, "c2": c2}, route)
        parity = mutation_parity_report(arms["base"], prod, inv)
        assert_compact_json_nbytes(parity, limit=MAX_COMPACT_TELEMETRY_BYTES, label="mutation_parity")
        route.append("apply_writeback_prod_inv_noop_parity")
        run_isolation_sentinel_checkpoint(mutable_arms_for_isolation(arms), base=arms["base"], label="post_apply")
        route.append("post_apply_isolation_sentinels")
        if not parity.get("pass"):
            markers["PHASE_THREE_ARM_APPLY_WRITEBACK_END"] = True
            return _unverified("mutation_parity_fail", "parity", markers, parent_pre, parity, route, asymmetric=True)
        markers["PHASE_THREE_ARM_APPLY_WRITEBACK_END"] = True
        if not smoke:
            clock.end("THREE_ARM_APPLY_WRITEBACK"); clock.begin("THREE_ARM_EVAL_NLL")
        markers["PHASE_THREE_ARM_EVAL_NLL_BEGIN"] = True
        eval_batches = [batches[i]["batch"] for i in (1, 2)]; nll, weights = {}, {}
        for arm in ("prod", "inv", "noop", "noop_repeat"):
            states = arms["noop"] if arm == "noop_repeat" else arms[arm]
            pre = arm_hash_map(arms, ("base", "parent", "prod", "inv", "noop", "calibration_shadow"))
            run_isolation_sentinel_checkpoint(mutable_arms_for_isolation(arms), base=arms["base"], label=f"pre_eval_{arm}")
            num, den, mean, w_hash = h.eval_arm_nll(arm, states, bundle, eval_batches)
            untouched_sentinel_report(
                before=pre, after=arm_hash_map(arms, ("base", "parent", "prod", "inv", "noop", "calibration_shadow")),
                required_unchanged=("base", "parent", "prod", "inv", "noop", "calibration_shadow"))
            nll[arm] = {"numerator_f64": float(num), "denominator": int(den), "mean": float(mean)}; weights[arm] = w_hash
        route.append("eval_nll_three_arm_plus_noop_repeat")
        run_isolation_sentinel_checkpoint(mutable_arms_for_isolation(arms), base=arms["base"], label="post_eval")
        route.append("post_eval_isolation_sentinels"); markers["PHASE_THREE_ARM_EVAL_NLL_END"] = True
        if not smoke:
            clock.end("THREE_ARM_EVAL_NLL"); clock.begin("EMIT_FLUSH")
        markers["PHASE_EMIT_FLUSH_BEGIN"] = True
        Lp, Li, L0, Lr = (nll[a]["mean"] for a in ("prod", "inv", "noop", "noop_repeat")); eps = epsilon_from_noop(L0)
        if abs(L0 - Lr) >= eps:
            markers["PHASE_EMIT_FLUSH_END"] = True
            return _unverified("noop_repeat_drift_crosses_epsilon", "eval", markers, parent_pre,
                               {"L_noop": L0, "L_noop_repeat": Lr, "epsilon": eps}, route)
        if weights["noop"] != weights["noop_repeat"]:
            markers["PHASE_EMIT_FLUSH_END"] = True
            return _unverified("noop_repeat_weight_hash_mismatch", "eval", markers, parent_pre,
                               {"noop": weights["noop"], "noop_repeat": weights["noop_repeat"]}, route)
        parent_post = rehash_path(parent["absolute_path"]) if parent.get("absolute_path") else parent_pre
        if parent_post != parent_pre:
            return _unverified("parent_sha_drift", "emit", markers, parent_pre, {"parent_sha256_post": parent_post}, route)
        cls, _ = classify_signed_utility(Lp, Li, L0)
        term = terminal_precedence_classify(integrity_failure=False, asymmetry_failure=False, empty_applied=False, science_classifier=cls)
        eval_ids = list(batches[1]["metadata"]["row_ids"]) + list(batches[2]["metadata"]["row_ids"])
        payload = {
            "schema": SCHEMA_SCIENCE, "classifier": term, "estimand": ESTIMAND_NAME, "legal_subset": legal_subset,
            "L_prod": Lp, "L_inv": Li, "L_noop": L0, "L_noop_repeat": Lr, "epsilon": eps, "nll_per_arm": nll,
            "parent_sha256_pre": parent_pre, "parent_sha256_post": parent_post,
            "phase_markers": {n: True for n in REQUIRED_PHASE_MARKER_NAMES},
            "apply_integer_vote_update_from_frozen_plan_calls": int(c1 + c2),
            "eligible_state_key_count": int(len(arms["base"])), "observer_public_apply_calibration": cal,
            "current_weights_sha256_by_arm": weights,
            "eval_row_ids_sha256": hashlib.sha256(json.dumps(eval_ids).encode()).hexdigest(),
            "eval_batch_count": 2, "leakage_report_compact": leak, "mutation_parity": parity,
            "terminal_precedence_path": ["integrity_clear", "asymmetry_clear", "science"],
            "route": route + ["emit_in_memory_payload"], "hooks_injected": hooks is not None, "phase_budgets": budgets,
        }
        assert_compact_json_nbytes(payload, limit=MAX_AUTHORITATIVE_RESULT_BYTES, label="authoritative_result")
        markers["PHASE_EMIT_FLUSH_END"] = True
        clock.end("CALIBRATE_EVAL" if smoke else "EMIT_FLUSH"); route.append("emit_in_memory_payload")
        return _validated(payload)
    except (IntegrityProofError, ArmProofError, PartitionLeakageError, LegalSubsetError) as exc:
        return _unverified(str(exc), "integrity", markers, parent_pre, {"error": type(exc).__name__}, route)
    except Exception as exc:  # noqa: BLE001
        if parent_pre is None: return _preflight(str(exc), "execution", {"error": type(exc).__name__}, {}, route)
        return _unverified(str(exc), "execution", markers, parent_pre, {"error": type(exc).__name__}, route)

def run_one_step_smoke(packet: Mapping[str, Any], *, hooks: AuthoritativeGpuHooks | None = None) -> dict[str, Any]:
    p = dict(packet); p["authoritative_deferred"] = False; p["smoke_mode"] = True; p.setdefault("pin_mode", "smoke")
    result = run_authoritative_gpu_call_graph(p, hooks=hooks)
    result["claim_ceiling"] = "implementation_correctness_only"; validate_authoritative_result_payload_v3(result); return result

__all__ = [
    "ARM_FORK_NAMES", "AuthoritativeGpuDeferredError", "AuthoritativeGpuError", "AuthoritativeGpuHooks",
    "CALL_GRAPH_STEPS_V6", "FORMAL_PHASE_BUDGETS", "INVERT_DIR_FIELDS", "PinValidationError",
    "SMOKE_PHASE_BUDGETS", "SMOKE_SUBPHASE_TO_ENVELOPE", "authoritative_gpu_source_must_not_route_to_toy",
    "build_live_hooks", "calibrate_capture_vs_public", "call_graph_steps", "canonical_invert_plans_v4",
    "compute_partition_leakage_compact", "deterministic_eval_contract", "hash_current_weights_tensors",
    "isolate_fork_arm_state", "phase_budgets_for_packet", "resolve_capture_device_mode",
    "run_authoritative_gpu_call_graph", "run_isolation_sentinel_checkpoint", "run_one_step_smoke"]
