"""Thin read-only loader + CLI for the R1 K_hat emission reducer.

Dependency: this module -> r7_r1_khat_emission_reducer (+ optional B2 CLI loader).
Never imported by the pure core.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from calm.hrm_text_158.native_full_stack.r7_b2_reducer_cli import load_sidecar_jsonl
from calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer import (
    OUTCOME_CANDIDATE_ONLY,
    OUTCOME_FREEZE_OK,
    OUTCOME_INVALID_COMPARISON_INPUT,
    OUTCOME_INVALID_OBSERVATION,
    OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2,
    OUTCOME_NO_CANDIDATE_NONVACUOUS,
    OUTCOME_NO_FREEZE_DISAGREEMENT,
    ActivationDeltaProof,
    AnalysisProvenance,
    ObservationProvenance,
    R1FreezeCompareResult,
    R1KhatResult,
    R1RunEnvelope,
    ScienceSourcePins,
    evaluate_r1_final_freeze,
    reduce_r1_khat_emission,
    to_json_dict,
)

_EXIT0 = {
    OUTCOME_CANDIDATE_ONLY,
    OUTCOME_FREEZE_OK,
    OUTCOME_NO_FREEZE_DISAGREEMENT,
    OUTCOME_NO_CANDIDATE_NONVACUOUS,
    OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2,
}
_EXIT3 = {OUTCOME_INVALID_OBSERVATION, OUTCOME_INVALID_COMPARISON_INPUT}


def _cli_exit(overall: str) -> int:
    if overall in _EXIT0:
        return 0
    if overall in _EXIT3:
        return 3
    return 3


def build_science_pins(d: dict[str, Any]) -> ScienceSourcePins:
    return ScienceSourcePins(
        census=str(d["census"]),
        learner=str(d["learner"]),
        probe=str(d["probe"]),
        parent_pt=str(d["parent_pt"]),
        b2_reducer_core=str(d["b2_reducer_core"]),
    )


def build_observation_provenance(d: dict[str, Any]) -> ObservationProvenance:
    return ObservationProvenance(
        role=str(d["role"]),
        launch_gate_msg_id=str(d["launch_gate_msg_id"]),
        launch_packet_sha=str(d["launch_packet_sha"]),
        nonce_or_run_id=str(d["nonce_or_run_id"]),
        scratch_root=str(d["scratch_root"]),
        sidecar_sha256=str(d["sidecar_sha256"]),
        sidecar_path=str(d["sidecar_path"]),
        observation_HEAD=str(d["observation_HEAD"]),
        science_source_pins=build_science_pins(d["science_source_pins"]),
        argv_semantic_family_digest=str(d["argv_semantic_family_digest"]),
        N=int(d["N"]),
        W=int(d["W"]),
        k_grid=tuple(int(x) for x in d["k_grid"]),
        role_anchor_b2_terminal_receipt_sha256=d.get("role_anchor_b2_terminal_receipt_sha256"),
        role_anchor_b2_bookend_amendment=d.get("role_anchor_b2_bookend_amendment"),
        role_anchor_original_launch_nonce=d.get("role_anchor_original_launch_nonce"),
        role_anchor_accepted_sidecar_sha256=d.get("role_anchor_accepted_sidecar_sha256"),
        role_anchor_replicate_launch_gate_msg_id=d.get("role_anchor_replicate_launch_gate_msg_id"),
        role_anchor_replicate_terminal_receipt_sha256=d.get("role_anchor_replicate_terminal_receipt_sha256"),
    )


def build_analysis_provenance(d: dict[str, Any]) -> AnalysisProvenance:
    return AnalysisProvenance(
        r1_design_sha256=str(d["r1_design_sha256"]),
        landed_r1_reducer_core_sha256=str(d["landed_r1_reducer_core_sha256"]),
        landed_r1_cli_sha256=str(d["landed_r1_cli_sha256"]),
        landed_r1_test_sha256=str(d["landed_r1_test_sha256"]),
        analysis_HEAD=str(d["analysis_HEAD"]),
        packet_contract_lineage=str(d["packet_contract_lineage"]),
    )


def _per_step_from_json(items: list[dict[str, Any]]):
    from calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer import PerStepKSnapshot
    return tuple(
        PerStepKSnapshot(
            step=int(s["step"]),
            denominator=int(s["denominator"]),
            ordered_K_eligible_counts=tuple(int(x) for x in s["ordered_K_eligible_counts"]),
            derived_fractions=tuple(float(x) for x in s["derived_fractions"]),
            closures=tuple(bool(x) for x in s["closures"]),
        )
        for s in items
    )


def _cliff_from_json(d: dict[str, Any]):
    from calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer import CliffDiagnostic
    return CliffDiagnostic(
        d.get("k_hat"), d.get("k_next"), d.get("k_hat_feasible"), d.get("k_next_feasible"),
        d.get("cliff_holds"), d.get("k_hat_min_count"), d.get("k_hat_min_fraction"),
        d.get("k_next_min_count"), d.get("k_next_min_fraction"),
    )


def _per_k_from_json(items: list[dict[str, Any]]):
    from calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer import PerKAggregate
    return tuple(
        PerKAggregate(
            k=int(a["k"]), min_eligible_count=int(a["min_eligible_count"]),
            max_eligible_count=int(a["max_eligible_count"]), min_fraction=float(a["min_fraction"]),
            mean_fraction=float(a["mean_fraction"]), any_zero=bool(a["any_zero"]),
            all_closures_ok=bool(a["all_closures_ok"]), feasible=bool(a["feasible"]),
        )
        for a in items
    )


def r1_result_from_json(d: dict[str, Any]) -> R1KhatResult:
    return R1KhatResult(
        overall=str(d["overall"]),
        b2_overall=d.get("b2_overall"),
        N=int(d["N"]), W=int(d["W"]),
        k_grid=tuple(int(x) for x in d["k_grid"]),
        final_four_ends=tuple(int(x) for x in d.get("final_four_ends") or ()),
        S_ss=tuple(int(x) for x in d.get("S_ss") or ()),
        per_step=_per_step_from_json(list(d.get("per_step") or [])),
        per_k=_per_k_from_json(list(d.get("per_k") or [])),
        denominator_min=d.get("denominator_min"),
        denominator_max=d.get("denominator_max"),
        denominator_constant=d.get("denominator_constant"),
        nesting_invariant_passed=d.get("nesting_invariant_passed"),
        table3_structural_passed=d.get("table3_structural_passed"),
        k_hat=d.get("k_hat"),
        cliff=_cliff_from_json(d.get("cliff") or {}),
        failure_locus=d.get("failure_locus"),
    )


def envelope_from_json(d: dict[str, Any]) -> R1RunEnvelope:
    return R1RunEnvelope(
        observation_provenance=build_observation_provenance(d["observation_provenance"]),
        analysis_provenance=build_analysis_provenance(d["analysis_provenance"]),
        r1_result=r1_result_from_json(d["r1_result"]),
        derived_S_ss=tuple(int(x) for x in d["derived_S_ss"]),
    )


def activation_delta_from_json(d: dict[str, Any]) -> ActivationDeltaProof:
    return ActivationDeltaProof(
        primary_observation_HEAD=str(d["primary_observation_HEAD"]),
        replicate_observation_HEAD=str(d["replicate_observation_HEAD"]),
        approved_r1_commit_sha=str(d["approved_r1_commit_sha"]),
        git_diff_paths=tuple(str(x) for x in d["git_diff_paths"]),
        science_pins_unchanged=bool(d["science_pins_unchanged"]),
        operator_attestation_note=str(d.get("operator_attestation_note") or ""),
    )


def build_run_envelope(
    *,
    rows: list[dict[str, Any]],
    observation: dict[str, Any],
    analysis: dict[str, Any],
    N: int = 32,
    W: int = 8,
) -> R1RunEnvelope:
    result = reduce_r1_khat_emission(rows, N=N, W=W)
    return R1RunEnvelope(
        observation_provenance=build_observation_provenance(observation),
        analysis_provenance=build_analysis_provenance(analysis),
        r1_result=result,
        derived_S_ss=result.S_ss,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r7_r1_khat_emission_reducer_cli")
    sub = parser.add_subparsers(dest="mode")
    p_reduce = sub.add_parser("reduce")
    p_reduce.add_argument("sidecar_path")
    p_reduce.add_argument("--pretty", action="store_true")
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("envelope_a")
    p_compare.add_argument("envelope_b")
    p_compare.add_argument("--activation-delta-proof")
    p_compare.add_argument("--pretty", action="store_true")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return 2 if code != 0 else 0

    if args.mode is None:
        print("usage error: mode required (reduce|compare)", file=sys.stderr)
        return 2

    try:
        if args.mode == "reduce":
            rows = load_sidecar_jsonl(args.sidecar_path)
            result = reduce_r1_khat_emission(rows)
            body = to_json_dict(result)
            overall = result.overall
        else:
            env_a = envelope_from_json(json.loads(Path(args.envelope_a).read_text(encoding="utf-8")))
            env_b = envelope_from_json(json.loads(Path(args.envelope_b).read_text(encoding="utf-8")))
            proof = None
            if args.activation_delta_proof:
                proof = activation_delta_from_json(
                    json.loads(Path(args.activation_delta_proof).read_text(encoding="utf-8"))
                )
            result = evaluate_r1_final_freeze(env_a, env_b, activation_delta_proof=proof)
            body = to_json_dict(result)
            overall = result.overall
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"io_or_parse_error: {exc}", file=sys.stderr)
        return 2

    if args.pretty:
        print(json.dumps(body, indent=2, sort_keys=True))
    else:
        print(json.dumps(body, separators=(",", ":"), sort_keys=True))
    return _cli_exit(overall)


if __name__ == "__main__":
    raise SystemExit(main())
