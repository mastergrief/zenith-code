"""B2-4 GPU composition exact-parity proof (UNEXECUTED at B2-4-impl receipt).

Formal pass mint awaits test-operator GPU launch. Missing CUDA/env/Triton → pytest.fail.
"""
from __future__ import annotations

import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.qacc_apply_composition_dispatch import (
    RUN_GPU_Q_ACC_APPLY_NATIVE_ENV,
)
from calm.hrm_text_158.native_full_stack.vote_update import RUN_GPU_Q_ACC_APPLY_ENV

GPU_PARITY_MODULE_STATUS = "UNEXECUTED"


@pytest.fixture(autouse=True)
def _require_gpu_composition_env() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA device required for B2-4 composition GPU parity proof")
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_ENV) != "1":
        pytest.fail(f"{RUN_GPU_Q_ACC_APPLY_ENV}=1 is required for composition GPU proof")
    if os.environ.get(RUN_GPU_Q_ACC_APPLY_NATIVE_ENV) != "1":
        pytest.fail(
            f"{RUN_GPU_Q_ACC_APPLY_NATIVE_ENV}=1 is required for composition GPU proof"
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Triton required for composition GPU proof: {exc}")


def test_gpu_composition_parity_module_declared_unexecuted_on_cpu_box() -> None:
    """Receipt label only — impl validation does not claim GPU pass."""

    assert GPU_PARITY_MODULE_STATUS == "UNEXECUTED"


def test_gpu_composition_exact_parity_placeholder() -> None:
    pytest.fail(
        "B2-4-impl leaves composition GPU parity UNEXECUTED; test-operator owns launch"
    )
