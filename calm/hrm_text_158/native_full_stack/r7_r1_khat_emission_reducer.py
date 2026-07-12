"""R1 K_hat emission reducer (pure CPU core). per_step is SSOT; aggregates derived."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from calm.hrm_text_158.native_full_stack.r7_b2_table2_trajectory_sufficiency_reducer import (
    B2ReduceResult, OVERALL_INSUFFICIENT, OVERALL_INVALID, OVERALL_SUFFICIENT, reduce_b2_trajectory,
)
DEFAULT_K_GRID = (2, 4, 8, 12, 16)
FRACTION_ABS_TOL = 1e-9
OUTCOME_INVALID_OBSERVATION = "INVALID_OBSERVATION"
OUTCOME_INVALID_COMPARISON_INPUT = "INVALID_COMPARISON_INPUT"
OUTCOME_NO_CANDIDATE_NONVACUOUS = "NO_CANDIDATE_NONVACUOUS"
OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2 = "NO_CANDIDATE_INSUFFICIENT_B2"
OUTCOME_CANDIDATE_ONLY = "CANDIDATE_ONLY"
OUTCOME_NO_FREEZE_DISAGREEMENT = "NO_FREEZE_DISAGREEMENT"
OUTCOME_FREEZE_OK = "FREEZE_OK"
ROLE_ACCEPTED_PRIMARY = "accepted_primary"
ROLE_FRESH_REPLICATE = "fresh_replicate"
R1_APPROVED_DIFF_PATHS = (
    "calm/hrm_text_158/native_full_stack/r7_r1_khat_emission_reducer.py",
    "calm/hrm_text_158/native_full_stack/r7_r1_khat_emission_reducer_cli.py",
    "calm/llm_computer/tests/test_r7_r1_khat_emission_reducer_v0.py",
)
@dataclass(frozen=True, slots=True)
class PerStepKSnapshot:
    step: int; denominator: int; ordered_K_eligible_counts: tuple[int, ...]
    derived_fractions: tuple[float, ...]; closures: tuple[bool, ...]
@dataclass(frozen=True, slots=True)
class PerKAggregate:
    k: int; min_eligible_count: int; max_eligible_count: int; min_fraction: float
    mean_fraction: float; any_zero: bool; all_closures_ok: bool; feasible: bool
@dataclass(frozen=True, slots=True)
class CliffDiagnostic:
    k_hat: int | None; k_next: int | None; k_hat_feasible: bool | None
    k_next_feasible: bool | None; cliff_holds: bool | None
    k_hat_min_count: int | None; k_hat_min_fraction: float | None
    k_next_min_count: int | None; k_next_min_fraction: float | None
@dataclass(frozen=True, slots=True)
class R1KhatResult:
    overall: str; b2_overall: str | None; N: int; W: int; k_grid: tuple[int, ...]
    final_four_ends: tuple[int, ...]; S_ss: tuple[int, ...]; per_step: tuple[PerStepKSnapshot, ...]
    per_k: tuple[PerKAggregate, ...]; denominator_min: int | None; denominator_max: int | None
    denominator_constant: bool | None; nesting_invariant_passed: bool | None
    table3_structural_passed: bool | None; k_hat: int | None; cliff: CliffDiagnostic
    failure_locus: str | None
@dataclass(frozen=True, slots=True)
class ScienceSourcePins:
    census: str; learner: str; probe: str; parent_pt: str; b2_reducer_core: str
@dataclass(frozen=True, slots=True)
class ObservationProvenance:
    role: str; launch_gate_msg_id: str; launch_packet_sha: str; nonce_or_run_id: str
    scratch_root: str; sidecar_sha256: str; sidecar_path: str; observation_HEAD: str
    science_source_pins: ScienceSourcePins; argv_semantic_family_digest: str
    N: int; W: int; k_grid: tuple[int, ...]
    role_anchor_b2_terminal_receipt_sha256: str | None = None
    role_anchor_b2_bookend_amendment: str | None = None
    role_anchor_original_launch_nonce: str | None = None
    role_anchor_accepted_sidecar_sha256: str | None = None
    role_anchor_replicate_launch_gate_msg_id: str | None = None
    role_anchor_replicate_terminal_receipt_sha256: str | None = None
@dataclass(frozen=True, slots=True)
class AnalysisProvenance:
    r1_design_sha256: str; landed_r1_reducer_core_sha256: str; landed_r1_cli_sha256: str
    landed_r1_test_sha256: str; analysis_HEAD: str; packet_contract_lineage: str
@dataclass(frozen=True, slots=True)
class R1RunEnvelope:
    observation_provenance: ObservationProvenance; analysis_provenance: AnalysisProvenance
    r1_result: R1KhatResult; derived_S_ss: tuple[int, ...]
@dataclass(frozen=True, slots=True)
class ActivationDeltaProof:
    primary_observation_HEAD: str; replicate_observation_HEAD: str; approved_r1_commit_sha: str
    git_diff_paths: tuple[str, ...]; science_pins_unchanged: bool; operator_attestation_note: str
@dataclass(frozen=True, slots=True)
class R1FreezeCompareResult:
    overall: str; freeze_eligible_k_hat: int | None; primary_k_hat: int | None
    replicate_k_hat: int | None; hard_check_failures: tuple[str, ...]
    science_mismatch_reasons: tuple[str, ...]; activation_delta_applied: bool

def _empty_cliff() -> CliffDiagnostic:
    return CliffDiagnostic(None, None, None, None, None, None, None, None, None)

def _empty_result(*, overall: str, b2_overall: str | None, N: int, W: int,
                  k_grid: tuple[int, ...], failure_locus: str | None, **kw: Any) -> R1KhatResult:
    return R1KhatResult(
        overall=overall, b2_overall=b2_overall, N=N, W=W, k_grid=k_grid,
        final_four_ends=kw.get("final_four_ends", ()), S_ss=kw.get("S_ss", ()),
        per_step=kw.get("per_step", ()), per_k=(), denominator_min=None, denominator_max=None,
        denominator_constant=None, nesting_invariant_passed=kw.get("nesting_invariant_passed"),
        table3_structural_passed=kw.get("table3_structural_passed"), k_hat=None,
        cliff=_empty_cliff(), failure_locus=failure_locus,
    )

def _norm_body(per_k: Mapping[Any, Any], k: int) -> Mapping[str, Any] | None:
    v = per_k[k] if k in per_k else per_k.get(str(k))
    return v if isinstance(v, Mapping) else None

def _cint(v: Any) -> int | None:
    return v if type(v) is int else None

def _frac_ok(stored: Any, count: int, denom: int) -> bool:
    if denom <= 0: return False
    if stored is None: return True
    if type(stored) is bool or not (type(stored) is int or type(stored) is float): return False
    return abs(float(stored) - count / denom) <= FRACTION_ABS_TOL

def _validate_final_four(ff: Any, *, W: int) -> tuple[tuple[int, ...], tuple[int, ...], str | None]:
    if ff is None: return (), (), "final_four_available_none"
    if not isinstance(ff, (tuple, list)) or len(ff) != 4: return (), (), "final_four_len_ne_4"
    ends = []
    for w in ff:
        end = getattr(w, "end_step", None)
        if type(end) is not int: return (), (), "final_four_end_not_int"
        ends.append(end)
    if len(set(ends)) != 4: return (), (), "final_four_dup_ends"
    if any(ends[i] >= ends[i + 1] for i in range(3)): return (), (), "final_four_not_strictly_increasing"
    supports = []
    for end in ends:
        support = set(range(end - W + 1, end + 1))
        if len(support) != W: return (), (), "final_four_invalid_W8_support"
        supports.append(support)
    union = set().union(*supports); u = sorted(union)
    if not u or u != list(range(u[0], u[-1] + 1)): return (), (), "final_four_union_noncontiguous"
    return tuple(ends), tuple(u), None

def _build_per_step(rows_by_step: Mapping[int, Mapping[str, Any]], S_ss: Sequence[int],
                    k_grid: Sequence[int]) -> tuple[tuple[PerStepKSnapshot, ...], str | None]:
    out: list[PerStepKSnapshot] = []
    for step in S_ss:
        row = rows_by_step.get(step)
        if row is None: return (), f"missing_step_{step}"
        per_k = (row.get("table3") or {}).get("per_k") or {}
        if not isinstance(per_k, Mapping): return (), f"per_k_not_mapping_step_{step}"
        present = []
        for key in per_k.keys():
            if type(key) is int: present.append(key)
            elif type(key) is str and key.isdigit(): present.append(int(key))
            else: return (), f"extra_or_bad_k_key_step_{step}"
        if sorted(present) != list(k_grid) or len(present) != len(k_grid):
            return (), f"k_grid_mismatch_step_{step}"
        counts: list[int] = []; fracs: list[float] = []; closures: list[bool] = []; denom0 = None
        for k in k_grid:
            body = _norm_body(per_k, k)
            if body is None: return (), f"missing_k_{k}_step_{step}"
            count, denom, closure = _cint(body.get("eligible_count")), _cint(body.get("current_deferred_candidate_denominator")), body.get("eligibility_closure_ok")
            if count is None or denom is None or denom <= 0 or type(closure) is not bool:
                return (), f"bad_fields_k_{k}_step_{step}"
            if denom0 is None: denom0 = denom
            elif denom != denom0: return (), f"unequal_denom_step_{step}"
            if not closure: return (), f"closure_false_k_{k}_step_{step}"
            if not _frac_ok(body.get("eligible_fraction_of_deferred"), count, denom):
                return (), f"fraction_mismatch_k_{k}_step_{step}"
            counts.append(count); fracs.append(count / denom); closures.append(True)
        assert denom0 is not None
        out.append(PerStepKSnapshot(step, denom0, tuple(counts), tuple(fracs), tuple(closures)))
    return tuple(out), None

def _nesting_ok(per_step: Sequence[PerStepKSnapshot]) -> bool:
    for s in per_step:
        c = s.ordered_K_eligible_counts
        if any(c[i] < c[i + 1] for i in range(len(c) - 1)): return False
    return True

def derive_aggregates(per_step: Sequence[PerStepKSnapshot], k_grid: Sequence[int]
                      ) -> tuple[tuple[PerKAggregate, ...], int | None, int | None, bool | None]:
    if not per_step: return (), None, None, None
    denoms = [s.denominator for s in per_step]; dmin, dmax = min(denoms), max(denoms)
    aggs = []
    for idx, k in enumerate(k_grid):
        counts = [s.ordered_K_eligible_counts[idx] for s in per_step]
        fracs = [s.derived_fractions[idx] for s in per_step]
        closures = [s.closures[idx] for s in per_step]
        min_c, max_c = min(counts), max(counts); min_f = min(fracs); mean_f = sum(fracs) / len(fracs)
        any_zero = any(c == 0 for c in counts); all_ok = all(closures)
        feasible = (min_c >= 1) and (mean_f > 0.0) and all_ok and (not all(c == 0 for c in counts))
        aggs.append(PerKAggregate(k, min_c, max_c, min_f, mean_f, any_zero, all_ok, feasible))
    return tuple(aggs), dmin, dmax, dmin == dmax

def derive_cliff(per_k: Sequence[PerKAggregate], k_grid: Sequence[int], k_hat: int | None) -> CliffDiagnostic:
    if k_hat is None: return _empty_cliff()
    by_k = {a.k: a for a in per_k}; hat = by_k.get(k_hat)
    if hat is None: return _empty_cliff()
    try: idx = list(k_grid).index(k_hat)
    except ValueError: return _empty_cliff()
    k_next = k_grid[idx + 1] if idx + 1 < len(k_grid) else None
    nxt = by_k.get(k_next) if k_next is not None else None
    k_next_feas = None if k_next is None else bool(nxt.feasible) if nxt else False
    return CliffDiagnostic(k_hat, k_next, bool(hat.feasible), k_next_feas,
                           True if k_next is None else (not bool(k_next_feas)),
                           hat.min_eligible_count, hat.min_fraction,
                           None if nxt is None else nxt.min_eligible_count,
                           None if nxt is None else nxt.min_fraction)

def _k_hat_from_aggs(per_k: Sequence[PerKAggregate]) -> int | None:
    feas = [a.k for a in per_k if a.feasible]
    return max(feas) if feas else None

def validate_result_consistency(result: R1KhatResult) -> tuple[bool, tuple[str, ...]]:
    fails: list[str] = []
    if result.overall in (OUTCOME_INVALID_OBSERVATION, OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2) and not result.per_step:
        if result.k_hat is not None: fails.append("early_short_circuit_has_k_hat")
        if result.per_k: fails.append("early_short_circuit_has_per_k")
        return (not fails), tuple(fails)
    if not result.per_step: return False, ("missing_per_step",)
    if tuple(s.step for s in result.per_step) != result.S_ss: fails.append("per_step_S_ss_mismatch")
    if any(len(s.ordered_K_eligible_counts) != len(result.k_grid) for s in result.per_step):
        fails.append("per_step_width_mismatch")
    derived_aggs, dmin, dmax, dconst = derive_aggregates(result.per_step, result.k_grid)
    if result.per_k != derived_aggs: fails.append("per_k_not_derived_from_per_step")
    if (result.denominator_min, result.denominator_max, result.denominator_constant) != (dmin, dmax, dconst):
        fails.append("denominator_summary_mismatch")
    expect_hat = _k_hat_from_aggs(derived_aggs)
    if result.overall == OUTCOME_CANDIDATE_ONLY and (result.k_hat != expect_hat or expect_hat is None):
        fails.append("k_hat_mismatch")
    if result.overall == OUTCOME_NO_CANDIDATE_NONVACUOUS and (result.k_hat is not None or expect_hat is not None):
        fails.append("nonvacuous_k_hat_mismatch")
    if result.cliff != derive_cliff(derived_aggs, result.k_grid, result.k_hat):
        fails.append("cliff_not_derived_from_per_step")
    return (not fails), tuple(fails)

def reduce_r1_khat_emission(rows: Sequence[Mapping[str, Any]], *, N: int = 32, W: int = 8,
                            k_grid: Sequence[int] = DEFAULT_K_GRID) -> R1KhatResult:
    grid = tuple(int(k) for k in k_grid)
    b2: B2ReduceResult = reduce_b2_trajectory(rows, N=N, W=W)
    b2o = b2.verdicts.overall
    if b2o == OVERALL_INVALID or not b2.integrity_gate.passed:
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus="b2_invalid_or_integrity")
    if b2o == OVERALL_INSUFFICIENT:
        return _empty_result(overall=OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus="b2_insufficient")
    if b2o != OVERALL_SUFFICIENT:
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus="b2_unexpected_overall")
    ends, S_ss, ff_err = _validate_final_four(b2.rolling_mean_W8.final_four_available, W=W)
    if ff_err is not None:
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus=ff_err)
    rows_by_step = {row.get("step"): row for row in rows if isinstance(row, Mapping) and type(row.get("step")) is int}
    per_step, struct_err = _build_per_step(rows_by_step, S_ss, grid)  # type: ignore[arg-type]
    if struct_err is not None:
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus=struct_err)
    if not _nesting_ok(per_step):
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus="nesting_breach",
                             final_four_ends=ends, S_ss=S_ss, per_step=per_step, table3_structural_passed=True, nesting_invariant_passed=False)
    per_k, dmin, dmax, dconst = derive_aggregates(per_step, grid)
    k_hat = _k_hat_from_aggs(per_k)
    overall = OUTCOME_CANDIDATE_ONLY if k_hat is not None else OUTCOME_NO_CANDIDATE_NONVACUOUS
    result = R1KhatResult(overall, b2o, N, W, grid, ends, S_ss, per_step, per_k, dmin, dmax, dconst, True, True, k_hat, derive_cliff(per_k, grid, k_hat), None)
    ok, loci = validate_result_consistency(result)
    if not ok:
        return _empty_result(overall=OUTCOME_INVALID_OBSERVATION, b2_overall=b2o, N=N, W=W, k_grid=grid, failure_locus="consistency:" + ",".join(loci))
    return result

def _ne(s: str | None) -> bool: return isinstance(s, str) and len(s) > 0

def _pins_eq(a: ScienceSourcePins, b: ScienceSourcePins) -> bool:
    return (a.census, a.learner, a.probe, a.parent_pt, a.b2_reducer_core) == (b.census, b.learner, b.probe, b.parent_pt, b.b2_reducer_core)

def _analysis_eq(a: AnalysisProvenance, b: AnalysisProvenance) -> bool:
    return (a.r1_design_sha256, a.landed_r1_reducer_core_sha256, a.landed_r1_cli_sha256, a.landed_r1_test_sha256, a.analysis_HEAD, a.packet_contract_lineage) == (
        b.r1_design_sha256, b.landed_r1_reducer_core_sha256, b.landed_r1_cli_sha256, b.landed_r1_test_sha256, b.analysis_HEAD, b.packet_contract_lineage)

def _primary_ok(o: ObservationProvenance) -> tuple[bool, str | None]:
    if o.role != ROLE_ACCEPTED_PRIMARY: return True, None
    if not all(map(_ne, (o.role_anchor_b2_terminal_receipt_sha256, o.role_anchor_b2_bookend_amendment, o.role_anchor_original_launch_nonce, o.role_anchor_accepted_sidecar_sha256))):
        return False, "primary_anchors_missing"
    if o.role_anchor_original_launch_nonce != o.nonce_or_run_id: return False, "primary_nonce_anchor_mismatch"
    if o.role_anchor_accepted_sidecar_sha256 != o.sidecar_sha256: return False, "primary_sidecar_anchor_mismatch"
    return True, None

def _replicate_ok(o: ObservationProvenance) -> tuple[bool, str | None]:
    if o.role != ROLE_FRESH_REPLICATE: return True, None
    if not (_ne(o.role_anchor_replicate_launch_gate_msg_id) and _ne(o.role_anchor_replicate_terminal_receipt_sha256)):
        return False, "replicate_anchors_missing"
    return True, None

def _delta_ok(primary: ObservationProvenance, replicate: ObservationProvenance, proof: ActivationDeltaProof | None) -> tuple[bool, str | None, bool]:
    if primary.observation_HEAD == replicate.observation_HEAD: return True, None, False
    if proof is None: return False, "activation_delta_required", False
    if proof.primary_observation_HEAD != primary.observation_HEAD: return False, "activation_delta_primary_HEAD_mismatch", False
    if proof.replicate_observation_HEAD != replicate.observation_HEAD: return False, "activation_delta_replicate_HEAD_mismatch", False
    if not proof.science_pins_unchanged: return False, "activation_delta_science_pins_changed", False
    if tuple(proof.git_diff_paths) != R1_APPROVED_DIFF_PATHS: return False, "activation_delta_paths_not_exact_r1_set", False
    if not _ne(proof.approved_r1_commit_sha): return False, "activation_delta_commit_missing", False
    return True, None, True

def evaluate_r1_final_freeze(env_a: R1RunEnvelope, env_b: R1RunEnvelope, *,
                             activation_delta_proof: ActivationDeltaProof | None = None) -> R1FreezeCompareResult:
    hard: list[str] = []
    by_role = {env_a.observation_provenance.role: env_a, env_b.observation_provenance.role: env_b}
    if set(by_role) != {ROLE_ACCEPTED_PRIMARY, ROLE_FRESH_REPLICATE} or len(by_role) != 2:
        hard.append("roles_not_exact_pair")
    primary, replicate = by_role.get(ROLE_ACCEPTED_PRIMARY), by_role.get(ROLE_FRESH_REPLICATE)
    if primary is None or replicate is None:
        return R1FreezeCompareResult(OUTCOME_INVALID_COMPARISON_INPUT, None, None, None, tuple(hard or ["missing_role"]), (), False)
    po, ro = primary.observation_provenance, replicate.observation_provenance
    if not (_ne(po.nonce_or_run_id) and _ne(ro.nonce_or_run_id) and po.nonce_or_run_id != ro.nonce_or_run_id): hard.append("nonce_not_distinct")
    if not (_ne(po.sidecar_sha256) and _ne(ro.sidecar_sha256) and po.sidecar_sha256 != ro.sidecar_sha256): hard.append("sidecar_sha_not_distinct")
    ok_p, err_p = _primary_ok(po)
    if not ok_p: hard.append(err_p or "primary_anchors")
    ok_r, err_r = _replicate_ok(ro)
    if not ok_r: hard.append(err_r or "replicate_anchors")
    if _ne(po.role_anchor_replicate_launch_gate_msg_id) and not _ne(po.role_anchor_b2_terminal_receipt_sha256):
        hard.append("role_swap_primary_looks_replicate")
    if _ne(ro.role_anchor_b2_terminal_receipt_sha256) and not _ne(ro.role_anchor_replicate_launch_gate_msg_id):
        hard.append("role_swap_replicate_looks_primary")
    if not _pins_eq(po.science_source_pins, ro.science_source_pins): hard.append("science_pins_mismatch")
    if po.argv_semantic_family_digest != ro.argv_semantic_family_digest: hard.append("argv_semantic_mismatch")
    if (po.N, po.W, po.k_grid) != (ro.N, ro.W, ro.k_grid): hard.append("N_W_grid_mismatch")
    if primary.derived_S_ss != primary.r1_result.S_ss or replicate.derived_S_ss != replicate.r1_result.S_ss:
        hard.append("derived_S_ss_ne_result")
    if not _analysis_eq(primary.analysis_provenance, replicate.analysis_provenance): hard.append("analysis_provenance_mismatch")
    delta_ok, delta_err, delta_applied = _delta_ok(po, ro, activation_delta_proof)
    if not delta_ok: hard.append(delta_err or "activation_delta")
    for label, env in (("primary", primary), ("replicate", replicate)):
        ok, loci = validate_result_consistency(env.r1_result)
        if not ok: hard.append(f"consistency_{label}:" + ",".join(loci))
    pr, rr = primary.r1_result, replicate.r1_result
    if hard:
        return R1FreezeCompareResult(OUTCOME_INVALID_COMPARISON_INPUT, None, pr.k_hat, rr.k_hat, tuple(hard), (), delta_applied)
    if pr.overall == OUTCOME_INVALID_OBSERVATION or rr.overall == OUTCOME_INVALID_OBSERVATION:
        return R1FreezeCompareResult(OUTCOME_INVALID_COMPARISON_INPUT, None, pr.k_hat, rr.k_hat, ("invalid_observation_input",), (), delta_applied)
    # Healthy non-candidates: do not require S_ss equality (insufficient/empty has no S_ss).
    if pr.overall != OUTCOME_CANDIDATE_ONLY or rr.overall != OUTCOME_CANDIDATE_ONLY:
        return R1FreezeCompareResult(OUTCOME_NO_FREEZE_DISAGREEMENT, None, pr.k_hat, rr.k_hat, (), ("non_candidate_overall",), delta_applied)
    science: list[str] = []
    if primary.derived_S_ss != replicate.derived_S_ss or pr.S_ss != rr.S_ss:
        science.append("S_ss_mismatch")
    if pr.k_hat != rr.k_hat: science.append("k_hat_mismatch")
    k = pr.k_hat
    def _feas(res: R1KhatResult, kk: int) -> bool:
        try: idx = list(res.k_grid).index(kk)
        except ValueError: return False
        return all(s.ordered_K_eligible_counts[idx] >= 1 for s in res.per_step)
    if k is None or not _feas(pr, k) or not _feas(rr, k): science.append("not_feasible_every_S_ss_step")
    if not (bool(pr.cliff.cliff_holds) and bool(rr.cliff.cliff_holds)): science.append("cliff_mismatch")
    if science:
        return R1FreezeCompareResult(OUTCOME_NO_FREEZE_DISAGREEMENT, None, pr.k_hat, rr.k_hat, (), tuple(science), delta_applied)
    return R1FreezeCompareResult(OUTCOME_FREEZE_OK, pr.k_hat, pr.k_hat, rr.k_hat, (), (), delta_applied)

def to_json_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, R1KhatResult):
        if obj.per_step:
            per_k, dmin, dmax, dconst = derive_aggregates(obj.per_step, obj.k_grid)
            cliff = derive_cliff(per_k, obj.k_grid, obj.k_hat)
        else:
            per_k, dmin, dmax, dconst, cliff = (), None, None, None, _empty_cliff()
        return {"overall": obj.overall, "b2_overall": obj.b2_overall, "N": obj.N, "W": obj.W,
                "k_grid": list(obj.k_grid), "final_four_ends": list(obj.final_four_ends), "S_ss": list(obj.S_ss),
                "per_step": [{"step": s.step, "denominator": s.denominator,
                              "ordered_K_eligible_counts": list(s.ordered_K_eligible_counts),
                              "derived_fractions": list(s.derived_fractions), "closures": list(s.closures)} for s in obj.per_step],
                "per_k": [{"k": a.k, "min_eligible_count": a.min_eligible_count, "max_eligible_count": a.max_eligible_count,
                           "min_fraction": a.min_fraction, "mean_fraction": a.mean_fraction, "any_zero": a.any_zero,
                           "all_closures_ok": a.all_closures_ok, "feasible": a.feasible} for a in per_k],
                "denominator_min": dmin, "denominator_max": dmax, "denominator_constant": dconst,
                "nesting_invariant_passed": obj.nesting_invariant_passed, "table3_structural_passed": obj.table3_structural_passed,
                "k_hat": obj.k_hat,
                "cliff": {"k_hat": cliff.k_hat, "k_next": cliff.k_next, "k_hat_feasible": cliff.k_hat_feasible,
                          "k_next_feasible": cliff.k_next_feasible, "cliff_holds": cliff.cliff_holds,
                          "k_hat_min_count": cliff.k_hat_min_count, "k_hat_min_fraction": cliff.k_hat_min_fraction,
                          "k_next_min_count": cliff.k_next_min_count, "k_next_min_fraction": cliff.k_next_min_fraction},
                "failure_locus": obj.failure_locus}
    if isinstance(obj, R1FreezeCompareResult):
        return {"overall": obj.overall, "freeze_eligible_k_hat": obj.freeze_eligible_k_hat,
                "primary_k_hat": obj.primary_k_hat, "replicate_k_hat": obj.replicate_k_hat,
                "hard_check_failures": list(obj.hard_check_failures),
                "science_mismatch_reasons": list(obj.science_mismatch_reasons),
                "activation_delta_applied": obj.activation_delta_applied}
    if isinstance(obj, R1RunEnvelope):
        o, a = obj.observation_provenance, obj.analysis_provenance
        return {"observation_provenance": {
            "role": o.role, "launch_gate_msg_id": o.launch_gate_msg_id, "launch_packet_sha": o.launch_packet_sha,
            "nonce_or_run_id": o.nonce_or_run_id, "scratch_root": o.scratch_root, "sidecar_sha256": o.sidecar_sha256,
            "sidecar_path": o.sidecar_path, "observation_HEAD": o.observation_HEAD,
            "science_source_pins": {"census": o.science_source_pins.census, "learner": o.science_source_pins.learner,
                                    "probe": o.science_source_pins.probe, "parent_pt": o.science_source_pins.parent_pt,
                                    "b2_reducer_core": o.science_source_pins.b2_reducer_core},
            "argv_semantic_family_digest": o.argv_semantic_family_digest, "N": o.N, "W": o.W, "k_grid": list(o.k_grid),
            "role_anchor_b2_terminal_receipt_sha256": o.role_anchor_b2_terminal_receipt_sha256,
            "role_anchor_b2_bookend_amendment": o.role_anchor_b2_bookend_amendment,
            "role_anchor_original_launch_nonce": o.role_anchor_original_launch_nonce,
            "role_anchor_accepted_sidecar_sha256": o.role_anchor_accepted_sidecar_sha256,
            "role_anchor_replicate_launch_gate_msg_id": o.role_anchor_replicate_launch_gate_msg_id,
            "role_anchor_replicate_terminal_receipt_sha256": o.role_anchor_replicate_terminal_receipt_sha256},
            "analysis_provenance": {"r1_design_sha256": a.r1_design_sha256,
                                    "landed_r1_reducer_core_sha256": a.landed_r1_reducer_core_sha256,
                                    "landed_r1_cli_sha256": a.landed_r1_cli_sha256,
                                    "landed_r1_test_sha256": a.landed_r1_test_sha256,
                                    "analysis_HEAD": a.analysis_HEAD, "packet_contract_lineage": a.packet_contract_lineage},
            "r1_result": to_json_dict(obj.r1_result), "derived_S_ss": list(obj.derived_S_ss)}
    raise TypeError(f"unsupported type {type(obj)!r}")
