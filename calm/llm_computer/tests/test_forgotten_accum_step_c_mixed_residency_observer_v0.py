"""CPU-static characterization tests for Step-C mixed-residency observer."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_step_c_mixed_residency_observer import (
    ObserverViolation,
    assert_mixed_residency,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    assert_carrier_preflight,
)


class FakeCudaDevice:
    type = "cuda"
    index = 0


class FakeCpuDevice:
    type = "cpu"
    index = None


class FakeCudaTensor:
    def __init__(self, shape=(2, 3)) -> None:
        self.shape = shape
        self.dtype = torch.int64
        self.device = FakeCudaDevice()


class FakeCpuTensor:
    def __init__(self, shape=(2, 3)) -> None:
        self.shape = shape
        self.dtype = torch.int8
        self.device = FakeCpuDevice()


class FakeCudaModel:
    device = FakeCudaDevice()

    def parameters(self):
        yield SimpleNamespace(device=FakeCudaDevice())


def _real_state(*, shadow: bool = True) -> BoundedDeltaTensorState:
    q = torch.zeros((2, 3), dtype=torch.int8)
    acc = BoundedDeltaAccumulatorState(
        logical_shape=(2, 3),
        cold_default_value=0,
        hot_exact_indices=(),
        hot_exact_values=(),
    )
    shadow_t = torch.zeros((2, 3), dtype=torch.int16) if shadow else None
    return BoundedDeltaTensorState(
        state_key="toy",
        q_levels=q,
        frozen_scale=torch.tensor(1.0),
        bounded_accumulator=acc,
        exact_accumulator_shadow=shadow_t,
        bounded_accumulator_fresh_for_exact_shadow=bool(shadow),
        event_coded_live_carrier=None,
    )


def _cuda_batch() -> dict:
    return {
        "inputs": FakeCudaTensor(),
        "labels": FakeCudaTensor(),
        "sep_positions": FakeCudaTensor((2,)),
        "position_ids": FakeCudaTensor(),
    }


def test_positive_real_qacc_and_fake_cuda_model_batch_pass() -> None:
    assert_mixed_residency(
        model=FakeCudaModel(),
        batch=_cuda_batch(),
        states={"toy": _real_state()},
    )


@pytest.mark.parametrize(
    "batch,match",
    [
        ({}, "empty"),
        ({"labels": FakeCudaTensor()}, "missing"),
        (
            {
                "inputs": object(),
                "labels": FakeCudaTensor(),
                "sep_positions": FakeCudaTensor((2,)),
                "position_ids": FakeCudaTensor(),
            },
            "bad required",
        ),
        (
            {
                "inputs": FakeCpuTensor(),
                "labels": FakeCudaTensor(),
                "sep_positions": FakeCudaTensor((2,)),
                "position_ids": FakeCudaTensor(),
            },
            "not cuda",
        ),
        (
            {
                "inputs": FakeCudaTensor(),
                "labels": FakeCudaTensor(),
                "sep_positions": FakeCudaTensor((2,)),
                "position_ids": FakeCudaTensor(),
                "extra": {"nested": FakeCpuTensor()},
            },
            "not cuda",
        ),
        (
            SimpleNamespace(
                device=FakeCudaDevice(),
                data={
                    "inputs": FakeCudaTensor(),
                    "labels": FakeCudaTensor(),
                    "sep_positions": FakeCudaTensor((2,)),
                    "position_ids": FakeCudaTensor(),
                },
            ),
            "Mapping",
        ),
        (
            {
                "input_ids": FakeCudaTensor(),
                "labels": FakeCudaTensor(),
                "sep_positions": FakeCudaTensor((2,)),
                "position_ids": FakeCudaTensor(),
            },
            "missing",
        ),
    ],
)
def test_batch_negatives(batch, match: str) -> None:
    with pytest.raises(ObserverViolation, match="CUDA_LOOP_FAILURE") as exc:
        assert_mixed_residency(model=FakeCudaModel(), batch=batch, states={"toy": _real_state()})
    assert match.lower() in str(exc.value).lower() or match in str(exc.value)


def test_container_device_shortcut_rejected_when_batch_is_mapping_with_device_attr() -> None:
    class MappingWithDevice(dict):
        device = FakeCudaDevice()

    batch = MappingWithDevice(_cuda_batch())
    # still validates leaves; strip inputs to prove device attr is not authority
    del batch["inputs"]
    with pytest.raises(ObserverViolation, match="CUDA_LOOP_FAILURE"):
        assert_mixed_residency(model=FakeCudaModel(), batch=batch, states={"toy": _real_state()})


def test_q_levels_cuda_marked_double_rejected() -> None:
    st = _real_state()
    object.__setattr__(st, "q_levels", FakeCudaTensor(st.q_levels.shape))  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="CPU_APPLY_MISLABELED_AS_GPU"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_shadow_cuda_marked_double_rejected() -> None:
    st = _real_state()
    object.__setattr__(st, "exact_accumulator_shadow", FakeCudaTensor(st.q_levels.shape))  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="CPU_APPLY_MISLABELED_AS_GPU"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_q_levels_missing_device_invalid() -> None:
    st = _real_state()
    object.__setattr__(st, "q_levels", SimpleNamespace(dtype=torch.int8))  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="CPU_APPLY_MISLABELED_AS_GPU"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_bounded_accumulator_tensor_substitute_rejected() -> None:
    st = _real_state()
    object.__setattr__(st, "bounded_accumulator", torch.zeros(1))  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="QACC_CARRIER_TYPE_MISMATCH"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_bounded_accumulator_wrong_mapping_rejected() -> None:
    st = _real_state()
    object.__setattr__(st, "bounded_accumulator", {"device": FakeCudaDevice()})  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="QACC_CARRIER_TYPE_MISMATCH"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_event_coded_carrier_forbidden_without_scratch_walk() -> None:
    st = _real_state()
    carrier = SimpleNamespace(
        _hot_list_sync_scratch_indices_i64=FakeCudaTensor(),
        _hot_list_sync_scratch_values_i8=FakeCudaTensor(),
    )
    object.__setattr__(st, "event_coded_live_carrier", carrier)  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="EVENT_CODED_CARRIER_FORBIDDEN"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": st})


def test_assert_carrier_preflight_refuses_event_coded_flags() -> None:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        CARRIER_NONE,
        ELIGIBLE_SCOPE,
        GLOBAL_CAP_CONTRACT,
    )

    with pytest.raises(ValueError, match="event-coded"):
        assert_carrier_preflight(
            live_acc_carrier_selector=CARRIER_NONE,
            global_cap_contract=GLOBAL_CAP_CONTRACT,
            eligible_scope=ELIGIBLE_SCOPE,
            event_coded_flags_present=True,
        )


def test_empty_states_fail_closed() -> None:
    with pytest.raises(ObserverViolation, match="QACC_STATES_EMPTY"):
        assert_mixed_residency(model=FakeCudaModel(), batch=_cuda_batch(), states={})


def test_unrecognized_state_not_skipped() -> None:
    with pytest.raises(ObserverViolation, match="QACC_STATE_TYPE_MISMATCH"):
        assert_mixed_residency(
            model=FakeCudaModel(),
            batch=_cuda_batch(),
            states={"bad": object()},
        )

def _state_with_acc(acc: BoundedDeltaAccumulatorState) -> BoundedDeltaTensorState:
    return BoundedDeltaTensorState(
        state_key="toy", q_levels=torch.zeros(1, dtype=torch.int8),
        frozen_scale=torch.tensor(1.0), exact_accumulator_shadow=torch.zeros(1, dtype=torch.int16),
        bounded_accumulator=acc, event_coded_live_carrier=None,
        bounded_accumulator_fresh_for_exact_shadow=True,
    )

def test_real_acc_nested_tensor_in_hot_values_rejected() -> None:
    bad = BoundedDeltaAccumulatorState(
        logical_shape=(1,), cold_default_value=0, hot_exact_indices=(0,),
        hot_exact_values=(torch.tensor(1),),  # type: ignore[arg-type]
    )
    with pytest.raises(ObserverViolation, match="QACC_CARRIER_TYPE_MISMATCH"):
        assert_mixed_residency(
            model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": _state_with_acc(bad)},
        )

def test_real_acc_wrong_cold_default_type_rejected() -> None:
    bad = BoundedDeltaAccumulatorState(
        logical_shape=(1,), cold_default_value=0, hot_exact_indices=(), hot_exact_values=(),
    )
    object.__setattr__(bad, "cold_default_value", "0")  # type: ignore[misc]
    with pytest.raises(ObserverViolation, match="QACC_CARRIER_TYPE_MISMATCH"):
        assert_mixed_residency(
            model=FakeCudaModel(), batch=_cuda_batch(), states={"toy": _state_with_acc(bad)},
        )
