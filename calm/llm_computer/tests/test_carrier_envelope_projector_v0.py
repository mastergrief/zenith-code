"""Unit tests for the joint-drain envelope projector facade."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.carrier_envelope_projector import (
    CLASSIFICATION_NOT_REACHABLE,
    PATH_B_STRUCTURALLY_NOT_SUB2,
    REACHABLE_ORACLE_BPW_THRESHOLD,
    TerminalAnchors,
    build_envelope_verdict_artifact,
    build_residual_hot_reduction,
    build_verdict_from_manifest_path,
    canonical_json,
    classify_envelope_verdict,
    load_band_constants_from_sweep,
    load_terminal_anchors_from_manifest,
    project_transform_bpw,
    required_hot_reduction_fraction_for_bpw,
    rollup_dependent_not_applicable,
    run_band_sweep,
    sha256_payload,
    sub2_budget_bytes,
)
from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING,
)
from calm.hrm_text_158.native_full_stack.carrier_growth_summary import (
    project_best_combined_oracle_bpw,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    REPO_ROOT
    / "artifacts/consensus_prep/"
    "v4_live_phase_a_diagnostic_tier1_run_2189e72004_evidence_manifest.json"
)


@pytest.fixture()
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def anchors(manifest: dict) -> TerminalAnchors:
    return load_terminal_anchors_from_manifest(manifest)


@pytest.fixture()
def band_constants():
    return load_band_constants_from_sweep(run_band_sweep())


def test_manifest_2189e72004_optimistic_upper_bound_not_reachable(
    anchors: TerminalAnchors,
    band_constants,
) -> None:
    transforms = project_transform_bpw(anchors, band_constants)
    residual = build_residual_hot_reduction(anchors, band_constants)
    assert transforms.baseline == pytest.approx(85.10070174080985, rel=0, abs=1e-6)
    assert transforms.hot_floor_only == pytest.approx(31.316748210362025, rel=0, abs=1e-6)
    assert transforms.events_floor_only == pytest.approx(53.78416279384068, rel=0, abs=1e-6)
    assert transforms.optimistic_upper_bound == pytest.approx(
        transforms.hot_floor_only, rel=0, abs=1e-9
    )
    assert transforms.optimistic_upper_bound > float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING)
    assert transforms.optimistic_upper_bound > float(REACHABLE_ORACLE_BPW_THRESHOLD)
    assert classify_envelope_verdict(
        transforms, residual=residual, rollup_present=False
    ) == CLASSIFICATION_NOT_REACHABLE
    assert residual.flip_to_2p0_bpw_fraction_gt == pytest.approx(0.93614, rel=0, abs=1e-3)
    assert residual.flip_to_1p75_bpw_fraction_gt == pytest.approx(0.94413, rel=0, abs=1e-3)
    assert residual.synthetic_available_fraction == 0.0


def test_optimistic_upper_bound_monotonic_in_savings(
    anchors: TerminalAnchors,
    band_constants,
) -> None:
    transforms = project_transform_bpw(anchors, band_constants)
    assert transforms.baseline >= transforms.events_floor_only
    assert transforms.events_floor_only >= transforms.hot_floor_only
    assert transforms.hot_floor_only == pytest.approx(
        transforms.optimistic_upper_bound, rel=0, abs=1e-9
    )


def test_band_fraction_zero_when_p95_identical(band_constants) -> None:
    assert band_constants.max_hot_reduction_fraction == 0.0
    assert band_constants.qualifying_bands == (1, 2, 3)
    assert len(band_constants.rows) == 6


def test_best_combined_oracle_known_rollup() -> None:
    rollup = {
        "est_events_payload_bytes": 400,
        "est_hot_exact_payload_bytes": 250,
        "est_saved_bytes_v5_clear": 40,
        "est_saved_bytes_v2_coalesce": 20,
        "events_on_q_locked_not_hot": 10,
    }
    bpw = project_best_combined_oracle_bpw(
        rollup,
        eligible_weight_count=128,
        metadata_bytes=64,
        v1_max_hot_reduction_fraction=0.1,
    )
    assert bpw == pytest.approx(40.5625, rel=0, abs=1e-6)


def test_verdict_artifact_schema_roundtrip(anchors: TerminalAnchors, band_constants) -> None:
    transforms = project_transform_bpw(anchors, band_constants)
    residual = build_residual_hot_reduction(anchors, band_constants)
    artifact = build_envelope_verdict_artifact(
        anchors=anchors,
        band=band_constants,
        transforms=transforms,
        residual=residual,
        manifest_sha256="abc123",
        head_commit="a6ec875018a6128688082c8acbcf2478339b5f1b",
        parent_hash="9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        packet_sha="packetsha",
        rollup_present=False,
    )
    assert artifact["classification"] == CLASSIFICATION_NOT_REACHABLE
    assert artifact["path"] == PATH_B_STRUCTURALLY_NOT_SUB2
    assert "TBD" not in canonical_json(artifact)
    assert artifact["transform_bpw"]["best_combined_oracle"]["status"] == "not_applicable"
    assert artifact["transform_bpw_basis"]
    assert artifact["band_sweep"]["rows"]
    assert artifact["residual_hot_reduction"]["synthetic_available_fraction"] == 0.0
    assert sha256_payload(artifact)


def test_build_verdict_from_manifest_path_matches_cli_shape() -> None:
    verdict = build_verdict_from_manifest_path(
        manifest_path=str(MANIFEST_PATH),
        head_commit="a6ec875018a6128688082c8acbcf2478339b5f1b",
    )
    assert verdict["classification"] == CLASSIFICATION_NOT_REACHABLE
    assert verdict["terminal_anchors"]["run_id"] == "2189e72004"
    assert verdict["sub2_budget"]["bytes"] == sub2_budget_bytes(
        eligible_weight_count=29360128
    )


def test_required_hot_reduction_fraction_matches_residual(anchors: TerminalAnchors) -> None:
    flip_2 = required_hot_reduction_fraction_for_bpw(
        hot_bytes=anchors.hot_bytes,
        metadata_bytes=anchors.metadata_bytes,
        eligible_weight_count=anchors.eligible_weight_count,
        target_bpw=float(R4V_ACC_PHYSICAL_BITS_PER_WEIGHT_CEILING),
    )
    assert flip_2 == pytest.approx(0.93614, rel=0, abs=1e-3)


def test_manifest_only_rollup_dependent_transforms_not_applicable(
    anchors: TerminalAnchors,
    band_constants,
) -> None:
    transforms = project_transform_bpw(anchors, band_constants, rollup=None)
    artifact = build_envelope_verdict_artifact(
        anchors=anchors,
        band=band_constants,
        transforms=transforms,
        residual=build_residual_hot_reduction(anchors, band_constants),
        manifest_sha256="abc123",
        head_commit="a6ec875",
        parent_hash="9b4e311a",
        rollup_present=False,
    )
    marker = rollup_dependent_not_applicable()
    for key in ("V2_coalesce", "V5_stable_q_clear", "V5_max", "best_combined_oracle"):
        assert artifact["transform_bpw"][key] == marker
    for key in ("baseline", "optimistic_upper_bound", "hot_floor_only"):
        assert isinstance(artifact["transform_bpw"][key], float)


def test_rollup_present_transform_formulas_exact_values() -> None:
    from calm.hrm_text_158.native_full_stack.carrier_envelope_projector import (
        BandSweepConstants,
    )

    anchors = TerminalAnchors(
        run_id="synthetic",
        events_bytes=1000,
        hot_bytes=500,
        backlog_bytes=10,
        metadata_bytes=64,
        eligible_weight_count=128,
        terminal_inclusive_bpw=0.0,
    )
    rollup = {
        "est_events_payload_bytes": 400,
        "est_hot_exact_payload_bytes": 250,
        "est_saved_bytes_v5_clear": 40,
        "est_saved_bytes_v2_coalesce": 20,
        "events_on_q_locked_not_hot": 10,
    }
    band = BandSweepConstants(
        qualifying_bands=(1, 2, 3),
        hot_p95_by_band={1: 2.0},
        max_hot_reduction_fraction=0.1,
        rows=(),
    )
    transforms = project_transform_bpw(anchors, band, rollup=rollup)
    weights = 128
    metadata = 64
    hot_v1 = 225
    assert transforms.v2_coalesce == pytest.approx(
        ((400 - 20) + 250 + 10 + metadata) * 8 / weights, rel=0, abs=1e-9
    )
    assert transforms.v5_stable_q_clear == pytest.approx(
        ((400 - 40) + 250 + 10 + metadata) * 8 / weights, rel=0, abs=1e-9
    )
    assert transforms.v5_max == pytest.approx(
        ((400 - 40) + 250 + 10 + metadata) * 8 / weights, rel=0, abs=1e-9
    )
    assert transforms.v1_band_b == pytest.approx(
        (400 + hot_v1 + 10 + metadata) * 8 / weights, rel=0, abs=1e-9
    )
    assert transforms.best_combined_oracle == pytest.approx(
        ((400 - 40) + hot_v1 + 10 + metadata) * 8 / weights, rel=0, abs=1e-9
    )
    assert transforms.v2_coalesce > transforms.v5_stable_q_clear
    assert transforms.v1_band_b > transforms.best_combined_oracle
