"""Hostile + identity tests for BST composition audit — IMPORT harness symbols only (PLAN_v7)."""
from __future__ import annotations

import importlib

import pytest

import scripts.backward_saved_tensors_composition_audit_run as harness

SavedTensorEventEvidence = harness.SavedTensorEventEvidence
BackwardSavedTensorCompositionEvidence = harness.BackwardSavedTensorCompositionEvidence
freeze_events = harness.freeze_events
build_composition_evidence_from_frozen_fixture = (
    harness.build_composition_evidence_from_frozen_fixture
)
compute_events_sha256 = harness.compute_events_sha256
compute_partition_sha256 = harness.compute_partition_sha256
validate_event_schema_exact = harness.validate_event_schema_exact
validate_events_sha256 = harness.validate_events_sha256
validate_partition_sha256 = harness.validate_partition_sha256
validate_evidence_receipt_field_equality = harness.validate_evidence_receipt_field_equality
validate_live_repo_head_matches_claim = harness.validate_live_repo_head_matches_claim
validate_dependency_currency_against_plan_pins = (
    harness.validate_dependency_currency_against_plan_pins
)
validate_observed_device_uniformity = harness.validate_observed_device_uniformity
validate_device_census_covers_every_event = harness.validate_device_census_covers_every_event
classify_composition = harness.classify_composition
evidence_to_receipt_fields = harness.evidence_to_receipt_fields
count_helper_campaigns_in_harness_source = harness.count_helper_campaigns_in_harness_source
BRANCH_FP_DENSE = harness.BRANCH_FP_DENSE
BACKEND_EXACT = harness.BACKEND_EXACT


def _ev(
    *,
    dtype: str = "torch.float32",
    shape=(2,),
    numel: int = 2,
    requires_grad: bool = True,
    device: str = "cpu",
) -> dict:
    return {
        "dtype": dtype,
        "shape": list(shape) if not isinstance(shape, list) else shape,
        "numel": numel,
        "requires_grad": requires_grad,
        "device": device,
    }


def test_symbol_identity_evidence_class_is_harness_object() -> None:
    reloaded = importlib.import_module(
        "scripts.backward_saved_tensors_composition_audit_run"
    )
    assert SavedTensorEventEvidence is reloaded.SavedTensorEventEvidence
    assert BackwardSavedTensorCompositionEvidence is (
        reloaded.BackwardSavedTensorCompositionEvidence
    )
    assert freeze_events is reloaded.freeze_events
    assert build_composition_evidence_from_frozen_fixture is (
        reloaded.build_composition_evidence_from_frozen_fixture
    )
    assert validate_event_schema_exact is reloaded.validate_event_schema_exact
    assert validate_evidence_receipt_field_equality is (
        reloaded.validate_evidence_receipt_field_equality
    )
    assert validate_dependency_currency_against_plan_pins is (
        reloaded.validate_dependency_currency_against_plan_pins
    )


def test_shape_stored_as_immutable_tuple() -> None:
    ev = SavedTensorEventEvidence(
        dtype="torch.float32",
        shape=(2, 16, 32),
        numel=1024,
        requires_grad=True,
        device="cpu",
    )
    assert isinstance(ev.shape, tuple)
    canonical = ev.to_canonical_dict()
    assert isinstance(canonical["shape"], list)
    assert canonical["shape"] == [2, 16, 32]
    assert canonical["device"] == "cpu"


def test_single_helper_campaign_source_guard() -> None:
    assert count_helper_campaigns_in_harness_source() == 1


def test_freeze_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing"):
        freeze_events(
            [{"dtype": "torch.float32", "shape": [2], "numel": 2, "requires_grad": True}]
        )


def test_freeze_rejects_extra_field() -> None:
    with pytest.raises(ValueError, match="extra"):
        freeze_events([_ev() | {"extra": 1}])


def test_freeze_rejects_requires_grad_non_bool() -> None:
    bad = _ev()
    bad["requires_grad"] = 1  # type: ignore[assignment]
    with pytest.raises(ValueError, match="requires_grad"):
        freeze_events([bad])


def test_bool_as_int_in_shape_rejected() -> None:
    bad = _ev(shape=[True, 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="shape entry"):
        freeze_events([bad])


def test_bool_as_int_in_numel_rejected() -> None:
    bad = _ev()
    bad["numel"] = True  # type: ignore[assignment]
    with pytest.raises(ValueError, match="numel"):
        freeze_events([bad])


def test_events_sha_order_drift() -> None:
    a = freeze_events([_ev(shape=[2], numel=2), _ev(shape=[3], numel=3, requires_grad=False)])
    b = freeze_events([_ev(shape=[3], numel=3, requires_grad=False), _ev(shape=[2], numel=2)])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_events_sha_dtype_drift() -> None:
    a = freeze_events([_ev(dtype="torch.float32")])
    b = freeze_events([_ev(dtype="torch.float16")])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_events_sha_shape_drift() -> None:
    a = freeze_events([_ev(shape=[2])])
    b = freeze_events([_ev(shape=[3], numel=2)])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_events_sha_numel_drift() -> None:
    a = freeze_events([_ev(numel=2)])
    b = freeze_events([_ev(numel=3)])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_events_sha_requires_grad_drift() -> None:
    a = freeze_events([_ev(requires_grad=True)])
    b = freeze_events([_ev(requires_grad=False)])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_events_sha_device_drift() -> None:
    a = freeze_events([_ev(device="cpu")])
    b = freeze_events([_ev(device="cuda:0")])
    assert compute_events_sha256(a) != compute_events_sha256(b)


def test_hand_authored_partition_rejected_by_equality() -> None:
    evidence = build_composition_evidence_from_frozen_fixture()
    receipt = evidence_to_receipt_fields(evidence)
    receipt["observed_internal_payload_tensor_count"] = 0
    with pytest.raises(ValueError, match="inequality"):
        validate_evidence_receipt_field_equality(evidence, receipt)


def test_literal_device_rejected_by_equality() -> None:
    evidence = build_composition_evidence_from_frozen_fixture()
    receipt = evidence_to_receipt_fields(evidence)
    # Inject a literal that disagrees with observed census-derived device.
    receipt["device"] = "cuda:0"
    with pytest.raises(ValueError, match="inequality"):
        validate_evidence_receipt_field_equality(evidence, receipt)


def test_head_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="repo_head live mismatch"):
        validate_live_repo_head_matches_claim(
            claimed="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            live="9963f5b8fea6084d13b76b3849a90b189791c2e2",
        )


def test_dependency_currency_drift_rejected() -> None:
    pins = list(harness.DEPENDENCY_CURRENCY_PINS)
    bad = list(pins)
    path, _ = bad[0]
    bad[0] = (path, "0" * 64)
    with pytest.raises(ValueError, match="dependency-currency mismatch"):
        validate_dependency_currency_against_plan_pins(pins=bad)


def test_partial_device_census_rejected() -> None:
    events = freeze_events([_ev(device="cpu"), _ev(device="cpu")])
    with pytest.raises(ValueError, match="does not cover every event"):
        validate_device_census_covers_every_event(events, {"cpu": 1})


def test_nonuniform_device_census_measurement_invalid() -> None:
    events = freeze_events([_ev(device="cpu"), _ev(device="cuda:0")])
    with pytest.raises(ValueError, match="observed-device disagreement"):
        validate_observed_device_uniformity(events, expected_event_count=2)


def test_default_device_source_rejected_documented() -> None:
    """Hostile: torch.get_default_device is forbidden as a device source in harness source."""
    src = (harness.REPO_ROOT / "scripts/backward_saved_tensors_composition_audit_run.py").read_text(
        encoding="utf-8"
    )
    assert "get_default_device" not in src
    assert "OBSERVED_FROM_EVERY_EVENT" in src


def test_no_superseded_plan_authority_in_runtime_source() -> None:
    src = (harness.REPO_ROOT / "scripts/backward_saved_tensors_composition_audit_run.py").read_text(
        encoding="utf-8"
    )
    assert "PLAN_v4" not in src
    assert "PLAN_v5" not in src
    assert "PLAN_v6" not in src
    assert "3e4843b1984fb792caf0652e2c0c1bf04fc61fe1d632922c0be0e2e78d12cae6" not in src
    assert harness.PLAN_SHA256_EXPECTED.startswith("ebbc5ecd")
    assert harness.PLAN_REVISION == "v7"


def test_live_fixture_classifies_fp_dense() -> None:
    evidence = build_composition_evidence_from_frozen_fixture()
    assert evidence.deep_frozen_snapshot is True
    assert evidence.events_container_type == "tuple[SavedTensorEventEvidence, ...]"
    assert isinstance(evidence.events, tuple)
    assert all(isinstance(e.shape, tuple) for e in evidence.events)
    assert all(isinstance(e.device, str) for e in evidence.events)
    assert evidence.saved_tensor_count == 180
    assert evidence.observed_boundary_tensor_count == 30
    assert evidence.observed_checkpoint_dummy_tensor_count == 0
    assert evidence.observed_internal_payload_tensor_count == 150
    assert evidence.device_census == {"cpu": 180}
    assert evidence.device == "cpu"
    assert evidence.device_binding_mode == "OBSERVED_FROM_EVERY_EVENT"
    assert evidence.backend == BACKEND_EXACT
    assert classify_composition(evidence) == BRANCH_FP_DENSE
    validate_events_sha256(evidence)
    validate_partition_sha256(evidence)
    validate_device_census_covers_every_event(evidence.events, evidence.device_census)


def test_no_mutable_list_reachable_from_evidence() -> None:
    evidence = build_composition_evidence_from_frozen_fixture()
    assert isinstance(evidence.events, tuple)
    assert isinstance(evidence.device_census_items, tuple)
    with pytest.raises(Exception):
        evidence.events = ()  # type: ignore[misc]


def test_test_local_duplicate_evidence_symbol_forbidden_documented() -> None:
    assert SavedTensorEventEvidence is harness.SavedTensorEventEvidence
    assert (
        BackwardSavedTensorCompositionEvidence
        is harness.BackwardSavedTensorCompositionEvidence
    )


def test_partition_sha_stable() -> None:
    a = compute_partition_sha256(
        observed_boundary_tensor_count=30,
        observed_checkpoint_dummy_tensor_count=0,
        observed_internal_payload_tensor_count=150,
        saved_tensor_count=180,
    )
    b = compute_partition_sha256(
        observed_boundary_tensor_count=30,
        observed_checkpoint_dummy_tensor_count=0,
        observed_internal_payload_tensor_count=150,
        saved_tensor_count=180,
    )
    assert a == b


def test_dependency_currency_pins_live_match() -> None:
    observed = validate_dependency_currency_against_plan_pins()
    assert len(observed) == 7
