"""Pure Step-C mixed-residency observer (CPU-static; no I/O/GPU/launch)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import BoundedDeltaTensorState

REQUIRED_BATCH_KEYS = ("inputs", "labels", "sep_positions", "position_ids")
_ACC_FIELDS = ("logical_shape", "cold_default_value", "hot_exact_indices", "hot_exact_values")
_ACC_OPT = ("cold_exception_indices", "cold_exception_values")


class ObserverViolation(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code, self.detail = str(code), str(detail)
        super().__init__(f"{self.code}" + (f": {detail}" if detail else ""))


def _dev_type(obj: Any) -> str | None:
    device = getattr(obj, "device", None)
    return None if device is None else str(getattr(device, "type", "") or "")


def _tensor_like(obj: Any) -> bool:
    return hasattr(obj, "device") and hasattr(obj, "dtype")


def _tensor_leaves(node: Any, path: str = "") -> list[tuple[str, Any]]:
    if _tensor_like(node):
        return [(path or "<root>", node)]
    out: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            out.extend(_tensor_leaves(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for i, v in enumerate(node):
            out.extend(_tensor_leaves(v, f"{path}[{i}]"))
    return out


def _nested_tensor(node: Any) -> bool:
    return bool(_tensor_leaves(node))


def _int_tuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and all(type(x) is int for x in obj) and not _nested_tensor(obj)


def _host_acc(obj: Any) -> bool:
    # No isinstance short-circuit — real BoundedDeltaAccumulatorState must still validate fields.
    if _tensor_like(obj) or not all(hasattr(obj, n) for n in _ACC_FIELDS):
        return False
    if type(obj.cold_default_value) is not int or not _int_tuple(obj.logical_shape):
        return False
    if not _int_tuple(obj.hot_exact_indices) or not _int_tuple(obj.hot_exact_values):
        return False
    for name in _ACC_OPT:
        if hasattr(obj, name) and not _int_tuple(getattr(obj, name)):
            return False
    return not any(_nested_tensor(getattr(obj, n)) for n in _ACC_FIELDS)


def _state_ok(obj: Any) -> bool:
    if isinstance(obj, BoundedDeltaTensorState):
        return True
    return all(hasattr(obj, n) for n in (
        "q_levels", "bounded_accumulator", "exact_accumulator_shadow", "event_coded_live_carrier"
    ))


def _cpu_tensor(obj: Any, label: str) -> None:
    if not _tensor_like(obj):
        raise ObserverViolation("CPU_APPLY_MISLABELED_AS_GPU", f"{label} not tensor")
    dt = _dev_type(obj)
    if dt is None:
        raise ObserverViolation("CPU_APPLY_MISLABELED_AS_GPU", f"{label} missing device")
    if dt != "cpu":
        raise ObserverViolation("CPU_APPLY_MISLABELED_AS_GPU", label)


def assert_mixed_residency(*, model: Any, batch: Any, states: Any) -> None:
    mdev = getattr(model, "device", None)
    if mdev is None and hasattr(model, "parameters"):
        mdev = next(model.parameters()).device
    if str(getattr(mdev, "type", "")) != "cuda":
        raise ObserverViolation("CUDA_LOOP_FAILURE", "model not cuda")
    if not isinstance(batch, Mapping) or not batch:
        raise ObserverViolation("CUDA_LOOP_FAILURE", "batch empty or not Mapping")
    for key in REQUIRED_BATCH_KEYS:
        if key not in batch:
            raise ObserverViolation("CUDA_LOOP_FAILURE", f"missing key {key}")
        leaf = batch[key]
        if not _tensor_like(leaf) or _dev_type(leaf) is None:
            raise ObserverViolation("CUDA_LOOP_FAILURE", f"bad required leaf {key}")
        if _dev_type(leaf) != "cuda":
            raise ObserverViolation("CUDA_LOOP_FAILURE", f"required leaf not cuda {key}")
    for path, tensor in _tensor_leaves(batch):
        dt = _dev_type(tensor)
        if dt is None or dt != "cuda":
            raise ObserverViolation("CUDA_LOOP_FAILURE", f"tensor leaf not cuda {path}")
    if not isinstance(states, Mapping):
        raise ObserverViolation("QACC_STATE_TYPE_MISMATCH", "states not Mapping")
    if not states:
        raise ObserverViolation("QACC_STATES_EMPTY", "states empty")
    for key, state in states.items():
        if not _state_ok(state):
            raise ObserverViolation("QACC_STATE_TYPE_MISMATCH", str(key))
        if getattr(state, "event_coded_live_carrier", None) is not None:
            raise ObserverViolation("EVENT_CODED_CARRIER_FORBIDDEN", str(key))
        _cpu_tensor(state.q_levels, f"{key}.q_levels")
        if state.exact_accumulator_shadow is not None:
            _cpu_tensor(state.exact_accumulator_shadow, f"{key}.exact_accumulator_shadow")
        if not _host_acc(state.bounded_accumulator):
            raise ObserverViolation("QACC_CARRIER_TYPE_MISMATCH", f"{key}.bounded_accumulator")


__all__ = ["REQUIRED_BATCH_KEYS", "ObserverViolation", "assert_mixed_residency"]
