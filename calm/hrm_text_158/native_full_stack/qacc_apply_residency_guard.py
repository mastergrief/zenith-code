"""B2-4 apply-input row residency guard for composed q_acc_apply paths.

Uses TorchDispatchMode to block device→host transfers on registered apply-input
row tensors inside the guarded selection→apply window.  Structural provenance
requires apply-input tensors to be registered views of device rows_by_state.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING:
    from calm.hrm_text_158.native_full_stack.global_rate_cap_gpu import (
        DeviceGlobalRateCapStateRows,
    )

_ACTIVE_GUARD: ContextVar[QAccApplyInputRowGuard | None] = ContextVar(
    "qacc_apply_residency_guard",
    default=None,
)

_CPU_TRANSFER_OPS = frozenset(
    {
        torch.ops.aten._to_copy.default,
        torch.ops.aten.copy_.default,
    }
)


class QAccApplyResidencyViolation(RuntimeError):
    """Raised when apply-input row residency or provenance is violated."""


@dataclass(frozen=True)
class QAccApplyResidencyReport:
    cpu_selected_rows_materialized_before_q_acc_apply: bool
    python_row_lists_materialized_before_q_acc_apply: bool
    accepted_row_source: str

    def to_dict(self) -> dict[str, bool | str]:
        return {
            "cpu_selected_rows_materialized_before_q_acc_apply": bool(
                self.cpu_selected_rows_materialized_before_q_acc_apply
            ),
            "python_row_lists_materialized_before_q_acc_apply": bool(
                self.python_row_lists_materialized_before_q_acc_apply
            ),
            "accepted_row_source": self.accepted_row_source,
        }


def _storage_data_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _tensor_touches_registered_storage(
    tensor: torch.Tensor,
    registered_storage_ptrs: frozenset[int],
) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    try:
        return _storage_data_ptr(tensor) in registered_storage_ptrs
    except RuntimeError:
        return False


class QAccApplyInputRowGuard(TorchDispatchMode):
    """Guard registered apply-input row tensors during selection→apply."""

    def __init__(self) -> None:
        super().__init__()
        self._registered_storage_ptrs: set[int] = set()
        self._registered_tensor_ids: set[int] = set()
        self._rows_by_state_keys: set[str] = set()
        self.cpu_transfer_violation = False
        self.python_list_violation = False

    def register_apply_input_row_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("register_apply_input_row_tensor requires a torch.Tensor")
        if tensor.device.type != "cuda":
            raise QAccApplyResidencyViolation(
                "apply-input row tensors must be CUDA device-resident"
            )
        self._registered_storage_ptrs.add(_storage_data_ptr(tensor))
        self._registered_tensor_ids.add(id(tensor))
        return tensor

    def register_device_state_rows_provenance(
        self,
        state_rows: DeviceGlobalRateCapStateRows,
    ) -> DeviceGlobalRateCapStateRows:
        self._rows_by_state_keys.add(state_rows.state_key)
        for field_name in (
            "accepted_indices",
            "accepted_directions",
            "accepted_thresholds",
            "accepted_global_flat_indices",
            "deferred_indices",
            "deferred_directions",
            "deferred_thresholds",
            "deferred_global_flat_indices",
        ):
            self.register_apply_input_row_tensor(getattr(state_rows, field_name))
        return state_rows

    def register_replay_veto_provenance(
        self,
        replay_veto_indices: torch.Tensor,
        replay_veto_directions: torch.Tensor,
        replay_veto_thresholds: torch.Tensor,
    ) -> None:
        if replay_veto_indices.numel() == 0:
            return
        self.register_apply_input_row_tensor(replay_veto_indices)
        self.register_apply_input_row_tensor(replay_veto_directions)
        self.register_apply_input_row_tensor(replay_veto_thresholds)

    def register_dispatch_probe_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Register a tensor for dispatch interception tests without CUDA requirement."""

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("register_dispatch_probe_tensor requires a torch.Tensor")
        self._registered_storage_ptrs.add(_storage_data_ptr(tensor))
        self._registered_tensor_ids.add(id(tensor))
        return tensor

    def validate_apply_input_row_provenance(self, *tensors: torch.Tensor) -> None:
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            if tensor.device.type != "cuda":
                raise QAccApplyResidencyViolation(
                    "unregistered apply-input row tensor is not CUDA device-resident"
                )
            storage_ptr = _storage_data_ptr(tensor)
            if storage_ptr not in self._registered_storage_ptrs:
                self.python_list_violation = True
                raise QAccApplyResidencyViolation(
                    "apply-input row tensor is not a registered device rows_by_state view"
                )

    def residency_report(self) -> QAccApplyResidencyReport:
        return QAccApplyResidencyReport(
            cpu_selected_rows_materialized_before_q_acc_apply=bool(
                self.cpu_transfer_violation
            ),
            python_row_lists_materialized_before_q_acc_apply=bool(
                self.python_list_violation
            ),
            accepted_row_source="device_selection_result.rows_by_state",
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        registered_ptrs = frozenset(self._registered_storage_ptrs)
        if func in _CPU_TRANSFER_OPS:
            target_device = kwargs.get("device")
            if target_device is not None and torch.device(target_device).type == "cpu":
                for arg in args:
                    if _tensor_touches_registered_storage(arg, registered_ptrs):
                        self.cpu_transfer_violation = True
                        raise QAccApplyResidencyViolation(
                            "registered apply-input row tensor CPU transfer blocked"
                        )
        return func(*args, **kwargs)


@contextmanager
def composition_apply_residency_guard() -> Iterator[QAccApplyInputRowGuard]:
    guard = QAccApplyInputRowGuard()
    token = _ACTIVE_GUARD.set(guard)
    try:
        with guard:
            yield guard
    finally:
        _ACTIVE_GUARD.reset(token)


def active_residency_guard() -> QAccApplyInputRowGuard | None:
    return _ACTIVE_GUARD.get()


def build_apply_inputs_from_python_list_legacy(
    *,
    flat_indices: list[int],
    directions: list[int],
    thresholds: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy host→device python-list apply-input build (adversarial test only)."""

    return (
        torch.tensor(flat_indices, dtype=torch.int64, device=device),
        torch.tensor(directions, dtype=torch.int16, device=device),
        torch.tensor(thresholds, dtype=torch.int32, device=device),
    )
