"""CPU tests for BR-3C-E streaming-sparse attribution subcontract (Step-0 + Step-1)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.integer_marginal_attribution import (
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
    INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
    INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    integer_marginal_attribution_from_captures,
    streaming_sparse_attribution_from_captures,
)
from calm.hrm_text_158.native_full_stack.integer_native_optimizer_credit_path_design import (
    ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE,
    AUDIT_NO_DENSE_INT_ACCUM,
    AUDIT_NO_DENSE_INT_ATTR,
    CandidateDenseIntegerDispatchObservation,
    CandidateDenseIntegerDispatchObserver,
    FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS,
    OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT,
    STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS,
    build_streaming_sparse_attribution_subcontract_receipt,
    candidate_dense_integer_dispatch_observation,
    events_bit_identical,
    prove_streaming_sparse_attribution_subcontract,
    streaming_sparse_attribution_subcontract_hard_false_snapshot,
    validate_streaming_sparse_attribution_subcontract_receipt,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS,
)


class _Tiny(torch.nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        self.proj = BitLinear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _synthetic_captures(
    *,
    batch: int = 1,
    seq: int = 2,
    out_features: int = 4,
    in_features: int = 8,
    seed: int = 158,
) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, int]]:
    torch.manual_seed(seed)
    inputs = [torch.randn(batch, seq, in_features)]
    grad_outputs = [torch.randn(batch, seq, out_features)]
    return inputs, grad_outputs, (out_features, in_features)


def _dry_run_capture_fixture(
    *,
    in_features: int = 3,
    out_features: int = 2,
) -> tuple[dict, tuple[int, int]]:
    torch.manual_seed(158)
    model = _Tiny(in_features=in_features, out_features=out_features)
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)[:, :in_features]
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)[:, :out_features]
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    capture = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in state.q_levels.shape)
    return capture, weight_shape


# --- Step-0 adversarial observer tests (must pass before parity is interpreted) ---


@pytest.mark.parametrize("o,i", [(4, 8), (1, 6)])
def test_observer_trips_int32_add(o: int, i: int) -> None:
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        a = torch.zeros(o, i, dtype=torch.int32)
        _ = a + torch.ones(o, i, dtype=torch.int32)
    obs = observer.observation()
    assert obs.candidate_dense_integer_scratch_observed is True
    assert AUDIT_NO_DENSE_INT_ATTR in obs.candidate_dense_integer_scratch_surfaces


def test_observer_trips_int32_full() -> None:
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.full((o, i), 1, dtype=torch.int32)
    assert observer.observation().int32_attr_observed is True


def test_observer_trips_int32_ones_mul() -> None:
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.ones(o, i, dtype=torch.int32) * 3
    assert observer.observation().int32_attr_observed is True


def test_observer_trips_int32_where() -> None:
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        cond = torch.ones(o, i, dtype=torch.bool)
        a = torch.zeros(o, i, dtype=torch.int32)
        b = torch.ones(o, i, dtype=torch.int32)
        _ = torch.where(cond, a, b)
    assert observer.observation().int32_attr_observed is True


def test_observer_trips_int64_accum() -> None:
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.zeros(o, i, dtype=torch.int64)
    obs = observer.observation()
    assert obs.int64_accum_observed is True
    assert AUDIT_NO_DENSE_INT_ACCUM in obs.candidate_dense_integer_scratch_surfaces


def test_observer_trips_transpose_i_o_shape() -> None:
    """FOLD-1: [I,O] transpose with numel==O*I must trip (not only exact (O,I))."""
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.zeros(i, o, dtype=torch.int32)
    assert observer.observation().int32_attr_observed is True


def test_observer_trips_nested_tuple_output() -> None:
    """FOLD-2: full 2-D int tensor inside tuple-returning op must trip."""
    o, i = 4, 8
    full = torch.zeros(o, i, dtype=torch.int32)
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.sort(full, dim=1)
    assert observer.observation().int32_attr_observed is True


def test_observer_allows_1d_carrier() -> None:
    o, i = 4, 8
    with candidate_dense_integer_dispatch_observation((o, i)) as observer:
        _ = torch.zeros(o * i, dtype=torch.int32)
    assert observer.observation().candidate_dense_integer_scratch_observed is False


def test_observer_allows_reference_lane_off() -> None:
    o, i = 4, 8
    _ = torch.zeros(o, i, dtype=torch.int32)
    observer = CandidateDenseIntegerDispatchObserver((o, i))
    assert observer.observation().candidate_dense_integer_scratch_observed is False


def test_current_dense_helper_still_trips() -> None:
    inputs, grad_outputs, weight_shape = _synthetic_captures()
    with candidate_dense_integer_dispatch_observation(weight_shape) as observer:
        integer_marginal_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_shape,
        )
    assert observer.observation().candidate_dense_integer_scratch_observed is True


# --- Step-1 streaming parity ---


@pytest.mark.parametrize(
    "law_id",
    [
        INTEGER_MARGINAL_ATTRIBUTION_PRODUCTION_LAW_ID,
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V0,
        INTEGER_MARGINAL_ATTRIBUTION_LAW_ID_V1,
    ],
)
def test_streaming_matches_dense_oracle_full_support(law_id: str) -> None:
    inputs, grad_outputs, weight_shape = _synthetic_captures()
    oracle = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_shape,
        law_id=law_id,
        index_set_policy=INDEX_SET_ALL_STRUCTURALLY_TOUCHED,
    )
    with candidate_dense_integer_dispatch_observation(weight_shape) as observer:
        candidate, metrics = streaming_sparse_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_shape,
            law_id=law_id,
        )
    assert events_bit_identical(oracle, candidate)
    assert observer.observation().candidate_dense_integer_scratch_observed is False
    assert metrics.max_candidate_tile_numel <= weight_shape[1]
    assert metrics.max_candidate_tile_bytes < metrics.full_dense_baseline_bytes


def test_streaming_o1_edge_no_full_2d_materialization() -> None:
    """FOLD-5: O==1 uses true 1-D rescale; observer must stay clean; parity holds."""
    inputs, grad_outputs, weight_shape = _synthetic_captures(
        out_features=1,
        in_features=6,
        seed=901,
    )
    assert weight_shape == (1, 6)
    oracle = integer_marginal_attribution_from_captures(
        inputs,
        grad_outputs,
        weight_shape=weight_shape,
    )
    with candidate_dense_integer_dispatch_observation(weight_shape) as observer:
        candidate, metrics = streaming_sparse_attribution_from_captures(
            inputs,
            grad_outputs,
            weight_shape=weight_shape,
        )
    assert events_bit_identical(oracle, candidate)
    assert observer.observation().candidate_dense_integer_scratch_observed is False
    assert metrics.max_candidate_tile_shape == (6,)


def test_streaming_dry_run_capture_fixture_parity() -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    oracle = integer_marginal_attribution_from_captures(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
    )
    with candidate_dense_integer_dispatch_observation(weight_shape) as observer:
        candidate, _metrics = streaming_sparse_attribution_from_captures(
            capture["inputs"],
            capture["grad_outputs"],
            weight_shape=weight_shape,
        )
    assert events_bit_identical(oracle, candidate)
    assert observer.observation().candidate_dense_integer_scratch_observed is False


def test_prove_streaming_subcontract_receipt_green() -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    receipt = prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="dry-run-fixture-v1",
        reference_oracle_run_id="ref-oracle-001",
        candidate_run_id="candidate-001",
    )
    assert receipt.attribution_subcontract_mode == ATTRIBUTION_SUBCONTRACT_MODE_STREAMING_SPARSE
    assert receipt.full_support_parity_pass is True
    assert receipt.candidate_dense_integer_scratch_observed is False
    assert receipt.max_candidate_tile_bytes < receipt.full_dense_baseline_bytes
    assert receipt.reference_oracle_run_id != receipt.candidate_run_id
    validate_streaming_sparse_attribution_subcontract_receipt(receipt)


def test_receipt_tile_peak_separate_from_carrier_density() -> None:
    """FOLD-4: tile-peak win and carrier density are both recorded, not conflated."""
    capture, weight_shape = _dry_run_capture_fixture()
    receipt = prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="density-separation-v1",
        reference_oracle_run_id="ref-oracle-002",
        candidate_run_id="candidate-002",
    )
    assert receipt.max_candidate_tile_bytes < receipt.full_dense_baseline_bytes
    assert receipt.event_carrier_density_ratio >= 0.0
    assert receipt.candidate_event_carrier_peak_bytes > 0


def test_receipt_non_claims_exact_superset_of_standing_tuple() -> None:
    for claim in OPTIMIZER_CREDIT_STATE_FAIL_CLOSED_NON_CLAIMS:
        assert claim in STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS
    capture, weight_shape = _dry_run_capture_fixture()
    receipt = prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="non-claims-v1",
        reference_oracle_run_id="ref-oracle-003",
        candidate_run_id="candidate-003",
    )
    assert receipt.fp_exception_caveat == OPTIMIZER_CREDIT_STATE_FP_EXCEPTION_CAVEAT
    assert receipt.non_claims == STREAMING_SPARSE_ATTRIBUTION_SUBCONTRACT_NON_CLAIMS


def test_receipt_forbidden_flags_default_false() -> None:
    snapshot = streaming_sparse_attribution_subcontract_hard_false_snapshot()
    for field in FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS:
        assert snapshot[field] is False


@pytest.mark.parametrize("field", FORBIDDEN_STREAMING_SPARSE_SUBCONTRACT_FIELDS)
def test_receipt_validator_rejects_forbidden_true(field: str) -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    receipt = prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="forbidden-v1",
        reference_oracle_run_id="ref-oracle-004",
        candidate_run_id="candidate-004",
    )
    with pytest.raises(ValueError, match=field):
        validate_streaming_sparse_attribution_subcontract_receipt(
            replace(receipt, **{field: True})
        )


def test_receipt_validator_rejects_dense_scratch_observed() -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    receipt = prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="dense-scratch-v1",
        reference_oracle_run_id="ref-oracle-005",
        candidate_run_id="candidate-005",
    )
    with pytest.raises(ValueError, match="candidate_dense_integer_scratch_observed"):
        validate_streaming_sparse_attribution_subcontract_receipt(
            replace(
                receipt,
                candidate_dense_integer_scratch_observed=True,
                candidate_dense_integer_scratch_surfaces=(AUDIT_NO_DENSE_INT_ACCUM,),
            )
        )


def _green_streaming_subcontract_receipt():
    capture, weight_shape = _dry_run_capture_fixture()
    return prove_streaming_sparse_attribution_subcontract(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
        comparable_set_id="tamper-resistance-v1",
        reference_oracle_run_id="ref-oracle-tamper",
        candidate_run_id="candidate-tamper",
    )


def test_builder_rejects_dispatch_observation_shape_mismatch() -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    _, metrics = streaming_sparse_attribution_from_captures(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
    )
    wrong_obs = CandidateDenseIntegerDispatchObservation(
        weight_shape=(1, 1),
        full_dense_numel=1,
        int64_accum_observed=False,
        int32_attr_observed=False,
        max_candidate_tile_shape=(0,),
        max_candidate_tile_numel=0,
        max_candidate_tile_bytes=0,
    )
    with pytest.raises(ValueError, match="dispatch_observation.weight_shape"):
        build_streaming_sparse_attribution_subcontract_receipt(
            metrics=metrics,
            dispatch_observation=wrong_obs,
            full_support_parity_pass=True,
            comparable_set_id="shape-mismatch-v1",
            reference_oracle_run_id="ref-oracle-shape",
            candidate_run_id="candidate-shape",
        )


def test_builder_rejects_dispatch_observation_numel_mismatch() -> None:
    capture, weight_shape = _dry_run_capture_fixture()
    _, metrics = streaming_sparse_attribution_from_captures(
        capture["inputs"],
        capture["grad_outputs"],
        weight_shape=weight_shape,
    )
    wrong_obs = CandidateDenseIntegerDispatchObservation(
        weight_shape=metrics.full_dense_shape,
        full_dense_numel=metrics.full_dense_numel + 1,
        int64_accum_observed=False,
        int32_attr_observed=False,
        max_candidate_tile_shape=(0,),
        max_candidate_tile_numel=0,
        max_candidate_tile_bytes=0,
    )
    with pytest.raises(ValueError, match="dispatch_observation.full_dense_numel"):
        build_streaming_sparse_attribution_subcontract_receipt(
            metrics=metrics,
            dispatch_observation=wrong_obs,
            full_support_parity_pass=True,
            comparable_set_id="numel-mismatch-v1",
            reference_oracle_run_id="ref-oracle-numel",
            candidate_run_id="candidate-numel",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("full_dense_numel", 0, "full_dense_numel must be > 0"),
        ("full_dense_numel", 999, "prod\\(full_dense_shape\\)"),
        ("full_dense_baseline_bytes", 1, "full_dense_baseline_bytes"),
        ("candidate_event_count", 999999, "candidate_event_count"),
        ("candidate_event_carrier_peak_bytes", 1, "candidate_event_carrier_peak_bytes"),
        ("event_carrier_density_ratio", 0.5, "event_carrier_density_ratio"),
        ("max_candidate_tile_bytes", 1, "max_candidate_tile_bytes"),
        ("max_candidate_tile_numel", 999, "prod\\(max_candidate_tile_shape\\)"),
    ],
)
def test_receipt_validator_rejects_malformed_metric_invariants(
    field: str,
    value: object,
    match: str,
) -> None:
    receipt = _green_streaming_subcontract_receipt()
    with pytest.raises(ValueError, match=match):
        validate_streaming_sparse_attribution_subcontract_receipt(
            replace(receipt, **{field: value})
        )
