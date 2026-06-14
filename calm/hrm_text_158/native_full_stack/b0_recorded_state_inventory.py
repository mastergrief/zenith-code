"""B0 CPU-only recorded-state inventory + vote-acc prize-sizing wrapper.

Reuses ``build_acc_width_recorded_row_sweep`` — no new width replay engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.acc_width_recorded_row_sweep import (
    LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
    MIN_NON_DEGENERATE_THRESHOLD_ABS,
    build_acc_width_recorded_row_sweep,
)
from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    _file_sha256,
)

B0_SCHEMA_VERSION = "hrm_text_158_b0_recorded_state_inventory_vote_acc_prize_sizing/v0"
B0_SLICE_ID = "B0_recorded_state_inventory_vote_acc_prize_sizing_v0"
B0_MULTI_TRACE_SCHEMA_VERSION = (
    "hrm_text_158_b0_multi_trace_recorded_state_inventory_vote_acc_prize_sizing/v1"
)
B0_MULTI_TRACE_SLICE_ID = "B0_multi_trace_recorded_state_inventory_vote_acc_prize_sizing_v1"

BRANCH_HARNESS_OR_SCOPE_FAIL = "HARNESS_OR_SCOPE_FAIL"
BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM = "MEASUREMENT_STATE_EXISTS_AND_HEADROOM"
BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM = "MEASUREMENT_STATE_EXISTS_NO_HEADROOM"

MEASUREMENT_SHAPE_BRANCH_PRECEDENCE: tuple[str, ...] = (
    BRANCH_HARNESS_OR_SCOPE_FAIL,
    BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM,
    BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM,
)

CROSS_TRACE_HARNESS_FAIL = "HARNESS_FAIL"
CROSS_TRACE_TRACE_DEPENDENT_HEADROOM = "TRACE_DEPENDENT_HEADROOM"
CROSS_TRACE_HOLDS_ACROSS_TRACES = "HOLDS_ACROSS_TRACES"

CROSS_TRACE_BRANCH_PRECEDENCE: tuple[str, ...] = (
    CROSS_TRACE_HARNESS_FAIL,
    CROSS_TRACE_TRACE_DEPENDENT_HEADROOM,
    CROSS_TRACE_HOLDS_ACROSS_TRACES,
)

SOURCE_KIND_RECORDED_ROW = "recorded_row"
SOURCE_KIND_STABLE_TRACE = "stable_trace"
SOURCE_KIND_FULL_TENSOR_SNAPSHOT = "full_tensor_snapshot"
SOURCE_KIND_MISSING = "missing"

CREDITDIR_ROOT = Path("/home/gabe/claw-code-creditdir/transient_fp_credit")

THRESHOLD_MISMATCH_ID = "threshold_row_derivation_mismatch"

FROZEN_SWEEP_FINGERPRINT_KEYS = (
    "w_min",
    "w_min_headroom_safe",
    "w_min_invariant",
    "primary_label",
    "headroom_pass",
    "max_abs_acc_applied_flips",
    "max_abs_replayed_candidate_stream",
)

EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "deferred_backlog NOT serialized",
    "full_tensor_persistent_state pressure NOT claimed",
    "direction_flip STABILITY NOT claimed",
    "training_or_acquisition NOT claimed",
    "runtime_readiness_claim false",
    "full_sub2_claim false",
    "tensor_wide_deferred true",
    "single_trace true",
    "no science banking beyond measurement-shape inventory on this bundle",
)

MULTI_TRACE_EXPLICIT_NON_CLAIMS: tuple[str, ...] = (
    "deferred_backlog NOT serialized",
    "full_tensor_persistent_state pressure NOT claimed",
    "direction_flip STABILITY NOT claimed",
    "training_or_acquisition NOT claimed",
    "runtime_readiness_claim false",
    "full_sub2_claim false",
    "tensor_wide_deferred true",
    "multi_trace true",
    "no_gpu_dynamics true",
    "no science banking beyond measurement-shape inventory on these bundles",
)

MULTI_TRACE_CLAIM_BOUNDARY = (
    "measurement-shape inventory + prize-sizing generalization only; cross-trace "
    "branch result, whether HOLDS or TRACE_DEPENDENT, does not close the int16 "
    "vote-acc reduction lane — it only sizes/generalizes recorded-row headroom"
)


@dataclass(frozen=True)
class B0BundleSpec:
    capture_id: str
    chain_root: Path
    stable_trace_relpath: str
    capture_receipt_relpath: str
    b2c_receipt_relpath: str
    audit_receipt_relpath: str
    chain_manifest_relpath: str
    frozen_acc_width_receipt_relpath: str
    expected_input_shas: Mapping[str, str]
    frozen_acc_width_receipt_sha256: str
    parent_checkpoint_relpath: str
    parent_sha256: str
    b2b_trace_ndjson_relpath: str = "b2b_seed44/b2b_sequential_trace.ndjson"

    def path(self, relpath: str) -> Path:
        return self.chain_root / relpath


B0_CAPTURE2_BUNDLE = B0BundleSpec(
    capture_id="capture2",
    chain_root=CREDITDIR_ROOT / "b2b_recapture_20260610T204129Z",
    stable_trace_relpath="b2c_replay/stable_copy/source_00_27eaa270f16395d7.json",
    capture_receipt_relpath="b2b_seed44/receipt.json",
    b2c_receipt_relpath="b2c_replay/b2c_final_temporal_verdict_receipt.json",
    audit_receipt_relpath="audit_v0/transient_selection_information_audit_v0_receipt.json",
    chain_manifest_relpath="artifact_manifest.json",
    frozen_acc_width_receipt_relpath=(
        "baseline_b0/acc_width_sweep/acc_width_recorded_row_sweep_v0_receipt.json"
    ),
    expected_input_shas={
        "stable_trace": "27eaa270f16395d7e89fe6d2028176d358661dcda8409398c3cd75cf33dfe0be",
        "capture_receipt": "9aa6085b210f118fa0febac74d761332ecb919a6c2107587c34a97fbe0b99bf6",
        "b2c_receipt": "17e09a87d01875a560a43215252729b9d1f47f7679642e5a67899b987a7a9c27",
        "audit_receipt": "da611458bcf1b13a21675162a7ab11332e820dadfee81b3520f84349d8379d57",
    },
    frozen_acc_width_receipt_sha256=(
        "6556c9a7e7fb37deccb54bdef1711e2fd33c20b4d11df4e9390d895e834cfbd6"
    ),
    parent_checkpoint_relpath=(
        "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_"
        "pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
    ),
    parent_sha256="9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
)

B0_TRACE1_BUNDLE = B0BundleSpec(
    capture_id="trace1",
    chain_root=CREDITDIR_ROOT / "b2b_recapture_20260610T145044Z",
    stable_trace_relpath="b2c_replay/stable_copy/source_00_2556fcd31e592c6c.json",
    capture_receipt_relpath="b2b_seed43/receipt.json",
    b2c_receipt_relpath="b2c_replay/b2c_final_temporal_verdict_receipt.json",
    audit_receipt_relpath="audit_v0/transient_selection_information_audit_v0_receipt.json",
    chain_manifest_relpath="artifact_manifest.json",
    frozen_acc_width_receipt_relpath=(
        "baseline_b0/acc_width_sweep/acc_width_recorded_row_sweep_v0_receipt.json"
    ),
    expected_input_shas={
        "stable_trace": "2556fcd31e592c6c59e4784f9b2afe4171e3770b87c05f2b01586e5151fe2d28",
        "capture_receipt": "b0c0b06443772921e26e3642ed9499306648712f883c460b4aa74ca2ab61c452",
        "b2c_receipt": "cbe3dc8afb5be7ec606de1ca6006c92e084b107c795041f0e982023e3876b2ba",
        "audit_receipt": "098649204e17c0a274f2191855867aa149ef16e2d635427c4885fdf5f0b093fe",
    },
    frozen_acc_width_receipt_sha256=(
        "3e3157af6857b91adc2578449fbd0c19ebc24c6f87bfbdbd28958757ae8389ef"
    ),
    parent_checkpoint_relpath="",
    parent_sha256="",
    b2b_trace_ndjson_relpath="b2b_seed43/b2b_sequential_trace.ndjson",
)

B0_MULTI_TRACE_BUNDLE_SPECS: tuple[B0BundleSpec, ...] = (
    B0_TRACE1_BUNDLE,
    B0_CAPTURE2_BUNDLE,
)


def bundle_sweep_inputs_available(spec: B0BundleSpec) -> bool:
    required = (
        spec.path(spec.stable_trace_relpath),
        spec.path(spec.capture_receipt_relpath),
        spec.path(spec.b2c_receipt_relpath),
        spec.path(spec.audit_receipt_relpath),
    )
    return all(path.is_file() for path in required)


def _inventory_entry(
    *,
    artifact_id: str,
    relpath: str | None,
    role: str,
    source_kind: str,
    sufficient_for_b0: bool,
    notes: str,
    expected_sha256: str | None = None,
    chain_root: Path | None = None,
) -> dict[str, Any]:
    path_obj = chain_root / relpath if chain_root is not None and relpath else None
    exists = path_obj.is_file() if path_obj is not None else False
    observed_sha256: str | None = None
    sha_match: bool | None = None
    if exists and path_obj is not None:
        observed_sha256 = _file_sha256(path_obj)
        if expected_sha256 is not None:
            sha_match = observed_sha256 == expected_sha256
    return {
        "artifact_id": artifact_id,
        "relpath": relpath,
        "path": str(path_obj) if path_obj is not None else None,
        "role": role,
        "source_kind": source_kind if exists else SOURCE_KIND_MISSING,
        "exists": exists,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "sha256_match": sha_match,
        "sufficient_for_b0": bool(sufficient_for_b0 and exists),
        "notes": notes,
    }


def build_preregistered_source_inventory(spec: B0BundleSpec) -> list[dict[str, Any]]:
    root = spec.chain_root
    return [
        _inventory_entry(
            artifact_id="stable_trace_canonical",
            relpath=spec.stable_trace_relpath,
            role="stable_trace",
            source_kind=SOURCE_KIND_STABLE_TRACE,
            sufficient_for_b0=True,
            notes="canonical JSON stable copy consumed by sweep",
            expected_sha256=spec.expected_input_shas["stable_trace"],
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="b2b_trace_ndjson",
            relpath=spec.b2b_trace_ndjson_relpath,
            role="b2b_trace",
            source_kind=SOURCE_KIND_STABLE_TRACE,
            sufficient_for_b0=True,
            notes="ndjson transport variant; same hash as stable_copy when present",
            expected_sha256=spec.expected_input_shas["stable_trace"],
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="capture_receipt",
            relpath=spec.capture_receipt_relpath,
            role="capture_receipt",
            source_kind=SOURCE_KIND_RECORDED_ROW,
            sufficient_for_b0=True,
            notes="vote-spec / capture metadata",
            expected_sha256=spec.expected_input_shas["capture_receipt"],
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="b2c_receipt",
            relpath=spec.b2c_receipt_relpath,
            role="b2c_receipt",
            source_kind=SOURCE_KIND_RECORDED_ROW,
            sufficient_for_b0=True,
            notes="temporal verdict pin",
            expected_sha256=spec.expected_input_shas["b2c_receipt"],
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="audit_receipt",
            relpath=spec.audit_receipt_relpath,
            role="audit_receipt",
            source_kind=SOURCE_KIND_RECORDED_ROW,
            sufficient_for_b0=True,
            notes="audit pin for teacher-forced applied-candidate reconstruction",
            expected_sha256=spec.expected_input_shas["audit_receipt"],
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="chain_manifest",
            relpath=spec.chain_manifest_relpath,
            role="chain_manifest",
            source_kind=SOURCE_KIND_RECORDED_ROW,
            sufficient_for_b0=True,
            notes="optional composed vote-spec fallback via manifest parameters",
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="acc_width_existing_receipt",
            relpath=spec.frozen_acc_width_receipt_relpath,
            role="frozen_acc_width_receipt",
            source_kind=SOURCE_KIND_RECORDED_ROW,
            sufficient_for_b0=True,
            notes="frozen measurement receipt for fingerprint compare",
            expected_sha256=spec.frozen_acc_width_receipt_sha256,
            chain_root=root,
        ),
        _inventory_entry(
            artifact_id="parent_checkpoint",
            relpath=spec.parent_checkpoint_relpath,
            role="parent_checkpoint_reference",
            source_kind=SOURCE_KIND_MISSING,
            sufficient_for_b0=False,
            notes="manifest parent_sha256 reference only; not serialized in bundle",
            expected_sha256=spec.parent_sha256,
            chain_root=root,
        ),
        {
            "artifact_id": "tensor_wide_persistent_qacc",
            "relpath": None,
            "path": None,
            "role": "full_tensor_persistent_state",
            "source_kind": SOURCE_KIND_MISSING,
            "exists": False,
            "expected_sha256": None,
            "observed_sha256": None,
            "sha256_match": None,
            "sufficient_for_b0": False,
            "notes": "no dense int16 persistent accumulator tensor capture in bundle",
        },
    ]


def sweep_inputs_missing_from_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> list[str]:
    missing: list[str] = []
    required_ids = (
        "stable_trace_canonical",
        "capture_receipt",
        "b2c_receipt",
        "audit_receipt",
    )
    by_id = {str(entry["artifact_id"]): entry for entry in inventory}
    for artifact_id in required_ids:
        entry = by_id.get(artifact_id)
        if entry is None or not entry.get("exists"):
            missing.append(artifact_id)
        elif entry.get("sha256_match") is False:
            missing.append(f"{artifact_id}:sha_mismatch")
    return missing


def _vacuity_guard_assessment(threshold_abs: int | None) -> dict[str, Any]:
    if threshold_abs is None:
        return {
            "threshold_abs": None,
            "min_non_degenerate_threshold_abs": MIN_NON_DEGENERATE_THRESHOLD_ABS,
            "passes_vacuity_guard": False,
            "reason": "threshold_abs_unresolved",
        }
    passes = int(threshold_abs) >= MIN_NON_DEGENERATE_THRESHOLD_ABS
    return {
        "threshold_abs": int(threshold_abs),
        "min_non_degenerate_threshold_abs": MIN_NON_DEGENERATE_THRESHOLD_ABS,
        "passes_vacuity_guard": passes,
        "reason": (
            "threshold_abs_ge_min"
            if passes
            else "estimand_vacuous_threshold_below_min"
        ),
    }


def extract_threshold_mismatch_hazard(
    sweep_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = dict(sweep_receipt.get("vote_spec_provenance") or {})
    mismatch = dict(
        provenance.get("threshold_row_derivation_mismatch")
        or provenance.get("threshold_crosscheck")
        or {}
    )
    vote_spec = dict(sweep_receipt.get("vote_spec") or {})
    derived = mismatch.get("derived_threshold_abs")
    expected = mismatch.get("expected_threshold_abs")
    is_mismatch = (
        mismatch.get("threshold_crosscheck") == THRESHOLD_MISMATCH_ID
        or provenance.get("threshold_crosscheck", {}).get("threshold_crosscheck")
        == THRESHOLD_MISMATCH_ID
    )
    derived_vacuity = _vacuity_guard_assessment(
        int(derived) if derived is not None else None
    )
    attested_vacuity = _vacuity_guard_assessment(
        vote_spec.get("threshold_abs")
        if vote_spec.get("threshold_abs") is not None
        else expected
    )
    return {
        "hazard_id": THRESHOLD_MISMATCH_ID,
        "present": bool(is_mismatch),
        "surfaced_loudly": bool(mismatch.get("surfaced_loudly", is_mismatch)),
        "expected_threshold_abs": expected,
        "derived_threshold_abs_from_rows": derived,
        "vote_spec_replay_threshold_abs": vote_spec.get("threshold_abs"),
        "authoritative_for_replay": vote_spec.get("threshold_abs"),
        "do_not_bank_row_derived_threshold": True,
        "row_provenance": mismatch.get("row_provenance"),
        "estimand_vacuity_guard": {
            "min_non_degenerate_threshold_abs": MIN_NON_DEGENERATE_THRESHOLD_ABS,
            "derived_threshold_assessment": derived_vacuity,
            "attested_threshold_assessment": attested_vacuity,
            "corrected_plan_note": (
                "derived=1 FAILS vacuity guard (1 < min=2); attested=10 PASSES — "
                "reinforces row-derived threshold is NOT authoritative"
            ),
        },
        "interpretation": (
            "replay uses attested vote_spec.threshold_abs; row-derived threshold "
            "from residual/proximity relation is surfaced but NOT adopted"
        ),
    }


def emit_measurement_shape_branch(
    sweep_receipt: Mapping[str, Any],
    *,
    inventory_missing: Sequence[str],
) -> dict[str, Any]:
    failure_reasons = list(sweep_receipt.get("failure_reasons") or [])
    integrity = dict(sweep_receipt.get("input_integrity") or {})
    field_gate = dict(sweep_receipt.get("field_inventory_gate") or {})
    harness_failures: list[str] = list(inventory_missing)

    if not integrity.get("passed", False):
        harness_failures.append("input_integrity_fail")
    if not field_gate.get("passed", False):
        harness_failures.append("field_inventory_gate_fail")
    if sweep_receipt.get("vote_spec") is None:
        harness_failures.append("vote_spec_unresolved")
    if sweep_receipt.get("primary_label") == LABEL_SCREEN_HARNESS_OR_GATE_FAIL:
        harness_failures.extend(failure_reasons or ["screen_harness_or_gate_fail"])
    for token in failure_reasons:
        if token.startswith("estimand_vacuous") or token in {
            "w16_not_bit_identical_to_reference",
            "w8_not_reference_invariant",
        }:
            harness_failures.append(token)

    headroom_pass = bool(sweep_receipt.get("headroom_pass"))
    w_min_headroom_safe = sweep_receipt.get("w_min_headroom_safe")
    max_abs_applied = int(sweep_receipt.get("max_abs_acc_applied_flips") or 0)

    if harness_failures:
        primary_branch = BRANCH_HARNESS_OR_SCOPE_FAIL
    elif (
        headroom_pass
        and w_min_headroom_safe is not None
        and max_abs_applied > 0
    ):
        primary_branch = BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
    else:
        primary_branch = BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM

    return {
        "primary_branch": primary_branch,
        "branch_precedence": list(MEASUREMENT_SHAPE_BRANCH_PRECEDENCE),
        "harness_failures": harness_failures,
        "headroom_pass": headroom_pass,
        "w_min_headroom_safe": w_min_headroom_safe,
        "max_abs_acc_applied_flips": max_abs_applied,
        "acc_shrink_primary_label": sweep_receipt.get("primary_label"),
    }


def compare_sweep_fingerprint(
    live_receipt: Mapping[str, Any],
    frozen_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: dict[str, dict[str, Any]] = {}
    for key in FROZEN_SWEEP_FINGERPRINT_KEYS:
        live_value = live_receipt.get(key)
        frozen_value = frozen_receipt.get(key)
        if live_value != frozen_value:
            mismatches[key] = {"live": live_value, "frozen": frozen_value}
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "compared_keys": list(FROZEN_SWEEP_FINGERPRINT_KEYS),
    }


def run_b0_recorded_state_inventory_vote_acc_prize_sizing(
    *,
    bundle_spec: B0BundleSpec = B0_CAPTURE2_BUNDLE,
    frozen_receipt_path: Path | None = None,
) -> dict[str, Any]:
    inventory = build_preregistered_source_inventory(bundle_spec)
    inventory_missing = sweep_inputs_missing_from_inventory(inventory)

    sweep_receipt: dict[str, Any] | None = None
    sweep_error: str | None = None
    if not inventory_missing:
        try:
            sweep_receipt = build_acc_width_recorded_row_sweep(
                stable_trace_path=bundle_spec.path(bundle_spec.stable_trace_relpath),
                capture_receipt_path=bundle_spec.path(
                    bundle_spec.capture_receipt_relpath
                ),
                b2c_receipt_path=bundle_spec.path(bundle_spec.b2c_receipt_relpath),
                audit_receipt_path=bundle_spec.path(bundle_spec.audit_receipt_relpath),
                expected_shas=dict(bundle_spec.expected_input_shas),
                chain_manifest_path=bundle_spec.path(bundle_spec.chain_manifest_relpath),
            )
        except Exception as exc:  # pragma: no cover - defensive harness surface
            sweep_error = f"{type(exc).__name__}:{exc}"
            inventory_missing.append(f"sweep_build_error:{sweep_error}")

    if sweep_receipt is None:
        sweep_receipt = {
            "input_integrity": {"passed": False, "failure_reasons": inventory_missing},
            "field_inventory_gate": {"passed": False},
            "vote_spec": None,
            "failure_reasons": inventory_missing,
            "primary_label": LABEL_SCREEN_HARNESS_OR_GATE_FAIL,
            "headroom_pass": False,
            "w_min_headroom_safe": None,
            "max_abs_acc_applied_flips": 0,
        }

    threshold_hazard = extract_threshold_mismatch_hazard(sweep_receipt)
    measurement_shape = emit_measurement_shape_branch(
        sweep_receipt,
        inventory_missing=inventory_missing,
    )

    frozen_path = frozen_receipt_path or bundle_spec.path(
        bundle_spec.frozen_acc_width_receipt_relpath
    )
    frozen_receipt: dict[str, Any] | None = None
    frozen_fingerprint: dict[str, Any] | None = None
    if frozen_path.is_file():
        frozen_receipt = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_fingerprint = compare_sweep_fingerprint(sweep_receipt, frozen_receipt)

    prize_sizing = {
        "w_min": sweep_receipt.get("w_min"),
        "w_min_headroom_safe": sweep_receipt.get("w_min_headroom_safe"),
        "w_min_invariant": sweep_receipt.get("w_min_invariant"),
        "headroom_pass": sweep_receipt.get("headroom_pass"),
        "headroom_factor": sweep_receipt.get("headroom_factor"),
        "max_abs_acc_applied_flips": sweep_receipt.get("max_abs_acc_applied_flips"),
        "max_abs_replayed_candidate_stream": sweep_receipt.get(
            "max_abs_replayed_candidate_stream"
        ),
        "acc_shrink_primary_label": sweep_receipt.get("primary_label"),
    }

    readiness_embedded = dict(sweep_receipt.get("readiness_current_repo") or {})
    readiness_fixtures = {
        "embedded_current_repo_scaffold": {
            "fixture_id": "current_repo_scaffold",
            "ready_for_pre_full_stack_diagnostic": readiness_embedded.get(
                "ready_for_pre_full_stack_diagnostic"
            ),
            "ready_for_main_science": readiness_embedded.get("ready_for_main_science"),
            "blocker_surface_names": readiness_embedded.get("blocker_surface_names"),
        },
        "consensus_prelaunch_reference": {
            "fixture_id": "pre_full_stack_diagnostic",
            "cited_in": (
                "artifacts/consensus_prep/selector_support_consensus_v0_launch_packet.json"
            ),
            "ready_for_pre_full_stack_diagnostic": True,
            "ready_for_main_science": False,
            "main_science_blockers": ["activations_residuals"],
            "note": "reference only — not recomputed in B0 wrapper",
        },
        "neither_is_launch_pass": True,
    }

    return {
        "schema_version": B0_SCHEMA_VERSION,
        "slice_id": B0_SLICE_ID,
        "capture_id": bundle_spec.capture_id,
        "chain_root": str(bundle_spec.chain_root),
        "reuse_verdict": "REUSE_EXISTING_SWEEP_NO_NEW_MEASUREMENT_ENGINE",
        "source_inventory": inventory,
        "bundle_sufficiency_verdict": (
            "recorded_row class ONLY — sufficient for inventory/prize-sizing; "
            "NOT GPU/tensor-wide/training claims"
        ),
        "sweep_receipt": sweep_receipt,
        "threshold_mismatch_hazard": threshold_hazard,
        "measurement_shape_branch": measurement_shape,
        "prize_sizing": prize_sizing,
        "frozen_fingerprint_compare": frozen_fingerprint,
        "readiness_fixtures": readiness_fixtures,
        "explicit_non_claims": list(EXPLICIT_NON_CLAIMS),
        "claim_boundary": dict(sweep_receipt.get("claim_boundary") or {}),
    }


def _per_trace_harness_failed(trace_result: Mapping[str, Any]) -> bool:
    shape = dict(trace_result.get("measurement_shape_branch") or {})
    if shape.get("primary_branch") == BRANCH_HARNESS_OR_SCOPE_FAIL:
        return True
    fingerprint = trace_result.get("frozen_fingerprint_compare")
    if fingerprint is not None and not bool(fingerprint.get("passed")):
        return True
    return False


def emit_cross_trace_branch_classifier(
    per_trace_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    harness_failures: list[str] = []
    for trace_result in per_trace_results:
        capture_id = str(trace_result.get("capture_id", "unknown"))
        if _per_trace_harness_failed(trace_result):
            harness_failures.append(capture_id)

    if harness_failures:
        return {
            "primary_branch": CROSS_TRACE_HARNESS_FAIL,
            "branch_precedence": list(CROSS_TRACE_BRANCH_PRECEDENCE),
            "harness_failures": harness_failures,
            "trace_count": len(per_trace_results),
        }

    headroom_flags = [
        bool(
            dict(trace_result.get("measurement_shape_branch") or {}).get(
                "headroom_pass"
            )
        )
        for trace_result in per_trace_results
    ]
    w_min_headroom_values = [
        dict(trace_result.get("prize_sizing") or {}).get("w_min_headroom_safe")
        for trace_result in per_trace_results
    ]
    w_min_values = [
        dict(trace_result.get("prize_sizing") or {}).get("w_min")
        for trace_result in per_trace_results
    ]
    per_trace_branches = [
        dict(trace_result.get("measurement_shape_branch") or {}).get("primary_branch")
        for trace_result in per_trace_results
    ]

    headroom_diverges = len(set(headroom_flags)) > 1
    w_min_headroom_diverges = len(set(w_min_headroom_values)) > 1
    w_min_diverges = len(set(w_min_values)) > 1

    if headroom_diverges or w_min_headroom_diverges or w_min_diverges:
        return {
            "primary_branch": CROSS_TRACE_TRACE_DEPENDENT_HEADROOM,
            "branch_precedence": list(CROSS_TRACE_BRANCH_PRECEDENCE),
            "harness_failures": [],
            "trace_count": len(per_trace_results),
            "headroom_pass_by_trace": headroom_flags,
            "w_min_headroom_safe_by_trace": w_min_headroom_values,
            "w_min_by_trace": w_min_values,
        }

    all_headroom_branches = all(
        branch == BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM
        for branch in per_trace_branches
    )
    all_headroom_pass = all(headroom_flags)
    if all_headroom_branches and all_headroom_pass:
        return {
            "primary_branch": CROSS_TRACE_HOLDS_ACROSS_TRACES,
            "branch_precedence": list(CROSS_TRACE_BRANCH_PRECEDENCE),
            "harness_failures": [],
            "trace_count": len(per_trace_results),
            "w_min_headroom_safe": w_min_headroom_values[0] if w_min_headroom_values else None,
            "headroom_pass": True,
            "per_trace_measurement_shape_branches": per_trace_branches,
        }

    return {
        "primary_branch": CROSS_TRACE_TRACE_DEPENDENT_HEADROOM,
        "branch_precedence": list(CROSS_TRACE_BRANCH_PRECEDENCE),
        "harness_failures": [],
        "trace_count": len(per_trace_results),
        "per_trace_measurement_shape_branches": per_trace_branches,
        "reason": "harness_clean_but_not_uniform_headroom_branch",
    }


def run_b0_multi_trace_recorded_state_inventory(
    *,
    bundle_specs: Sequence[B0BundleSpec] = B0_MULTI_TRACE_BUNDLE_SPECS,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    for spec in bundle_specs:
        trace_result = run_b0_recorded_state_inventory_vote_acc_prize_sizing(
            bundle_spec=spec
        )
        traces.append(
            {
                "capture_id": spec.capture_id,
                "chain_root": str(spec.chain_root),
                "schema_version": trace_result["schema_version"],
                "slice_id": trace_result["slice_id"],
                "measurement_shape_branch": trace_result["measurement_shape_branch"],
                "prize_sizing": trace_result["prize_sizing"],
                "frozen_fingerprint_compare": trace_result.get(
                    "frozen_fingerprint_compare"
                ),
                "threshold_mismatch_hazard": trace_result["threshold_mismatch_hazard"],
                "single_trace_receipt": trace_result,
            }
        )

    cross_trace = emit_cross_trace_branch_classifier(traces)
    return {
        "schema_version": B0_MULTI_TRACE_SCHEMA_VERSION,
        "slice_id": B0_MULTI_TRACE_SLICE_ID,
        "mode": "multi_trace",
        "reuse_verdict": "REUSE_EXISTING_SWEEP_NO_NEW_MEASUREMENT_ENGINE",
        "traces": traces,
        "cross_trace_branch_classifier": cross_trace,
        "bundle_sufficiency_verdict": (
            "recorded_row class ONLY — sufficient for inventory/prize-sizing; "
            "NOT GPU/tensor-wide/training claims"
        ),
        "explicit_non_claims": list(MULTI_TRACE_EXPLICIT_NON_CLAIMS),
        "claim_boundary": MULTI_TRACE_CLAIM_BOUNDARY,
    }


__all__ = [
    "B0_CAPTURE2_BUNDLE",
    "B0_MULTI_TRACE_BUNDLE_SPECS",
    "B0_MULTI_TRACE_SCHEMA_VERSION",
    "B0_MULTI_TRACE_SLICE_ID",
    "B0_SCHEMA_VERSION",
    "B0_SLICE_ID",
    "B0_TRACE1_BUNDLE",
    "BRANCH_HARNESS_OR_SCOPE_FAIL",
    "BRANCH_MEASUREMENT_STATE_EXISTS_AND_HEADROOM",
    "BRANCH_MEASUREMENT_STATE_EXISTS_NO_HEADROOM",
    "CROSS_TRACE_HARNESS_FAIL",
    "CROSS_TRACE_HOLDS_ACROSS_TRACES",
    "CROSS_TRACE_TRACE_DEPENDENT_HEADROOM",
    "build_preregistered_source_inventory",
    "bundle_sweep_inputs_available",
    "compare_sweep_fingerprint",
    "emit_cross_trace_branch_classifier",
    "emit_measurement_shape_branch",
    "extract_threshold_mismatch_hazard",
    "run_b0_multi_trace_recorded_state_inventory",
    "run_b0_recorded_state_inventory_vote_acc_prize_sizing",
]
