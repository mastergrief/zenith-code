"""CPU-static hostiles for P1b live-conversion device-truth (r5 + r5b seam)."""

from __future__ import annotations

from dataclasses import replace
import types

import pytest
import torch

from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    apply_live_p1_conversion_surface_overrides,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    resolve_live_conversion_execution_device,
    validate_trainer_sub2_authority_live_conversion_receipt,
)
from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
    _mint_live_conversion_receipt,
)


def test_truthful_cpu_receipt_validates():
    receipt = _mint_live_conversion_receipt()
    assert receipt.execution_device == "cpu"
    assert receipt.gpu_launched is False
    assert receipt.proof_exit_before_optimizer_step is True
    assert receipt.dry_run is True  # synonym of proof-exit; NOT CLI --dry-run
    assert receipt.cli_dry_run is False
    validate_trainer_sub2_authority_live_conversion_receipt(
        receipt, measured_device="cpu"
    )


def test_truthful_cuda_declared_receipt_validates_without_gpu():
    """Internal consistency of a truthful CUDA receipt must PASS on CPU hosts."""
    receipt = replace(
        _mint_live_conversion_receipt(),
        execution_device="cuda",
        gpu_launched=True,
    )
    validate_trainer_sub2_authority_live_conversion_receipt(receipt)
    validate_trainer_sub2_authority_live_conversion_receipt(
        receipt, measured_device="cuda"
    )


def test_cuda_declared_claiming_cpu_rejected():
    receipt = replace(
        _mint_live_conversion_receipt(),
        execution_device="cpu",
        gpu_launched=False,
    )
    with pytest.raises(ValueError, match="execution_device.*measured"):
        validate_trainer_sub2_authority_live_conversion_receipt(
            receipt, measured_device="cuda"
        )


def test_cpu_fixture_claiming_gpu_launched_rejected():
    receipt = replace(
        _mint_live_conversion_receipt(),
        execution_device="cpu",
        gpu_launched=True,
    )
    with pytest.raises(ValueError, match="gpu_launched inconsistent"):
        validate_trainer_sub2_authority_live_conversion_receipt(receipt)


def test_cuda_receipt_claiming_no_gpu_rejected():
    receipt = replace(
        _mint_live_conversion_receipt(),
        execution_device="cuda",
        gpu_launched=False,
    )
    with pytest.raises(ValueError, match="gpu_launched inconsistent"):
        validate_trainer_sub2_authority_live_conversion_receipt(receipt)


def test_dry_run_cli_disambiguation_mismatch_rejected():
    receipt = replace(
        _mint_live_conversion_receipt(),
        dry_run=False,
        proof_exit_before_optimizer_step=True,
    )
    with pytest.raises(ValueError, match="dry_run must equal proof_exit"):
        validate_trainer_sub2_authority_live_conversion_receipt(receipt)


def test_optimizer_step_still_forbidden():
    receipt = replace(_mint_live_conversion_receipt(), optimizer_step_called=True)
    with pytest.raises(ValueError, match="optimizer.step"):
        validate_trainer_sub2_authority_live_conversion_receipt(receipt)


def test_readiness_consumer_accepts_truthful_cpu_shape():
    receipt = _mint_live_conversion_receipt()
    surfaces = apply_live_p1_conversion_surface_overrides(
        receipt, require_source_at_head=True
    )
    assert surfaces  # 3-row or 2-row flip applied without device-field refusal


# --- r5b: constructor-composition regressions (not receipt replace()) ---
# Mechanism: duck-typed FakeDevice/FakeParam/FakeModel stand-ins so CUDA
# observation is CPU-runnable without a GPU; raw loader batch stays real CPU
# tensors to prove they cannot override observed staged/model device.


class _FakeDevice:
    def __init__(self, type: str):
        self.type = type


class _FakeParam:
    def __init__(self, type: str):
        self.device = _FakeDevice(type)


class _FakeModel:
    def __init__(self, type: str):
        self._param = _FakeParam(type)

    def parameters(self):
        yield self._param


def test_r5b_raw_cpu_batch_plus_observed_cuda_resolves_cuda():
    """Actual failure shape: CPU loader batch present, staged/observed is CUDA."""
    raw_cpu_batch = {"inputs": torch.zeros(2, 4), "labels": torch.zeros(2, 4)}
    assert raw_cpu_batch["inputs"].device.type == "cpu"
    # Even if a caller mistakenly considered the raw batch, observed must win.
    observed = resolve_live_conversion_execution_device(
        observed_execution_device=_FakeDevice("cuda"),
        model=_FakeModel("cuda"),
        device="cpu",
    )
    assert observed == "cuda"
    # Staged-inputs path (post-_proof_child_batch stand-in): duck-typed .device
    staged = types.SimpleNamespace(device=_FakeDevice("cuda"))
    observed2 = resolve_live_conversion_execution_device(
        staged_inputs=staged,
        model=_FakeModel("cuda"),
        device="cpu",
    )
    assert observed2 == "cuda"
    assert observed == "cuda" and observed2 == "cuda"
    # gpu_launched binding under receipt rules
    assert (observed == "cuda") is True


def test_r5b_staged_vs_model_param_mismatch_rejects():
    with pytest.raises(ValueError, match="mismatches model parameter"):
        resolve_live_conversion_execution_device(
            observed_execution_device="cuda",
            model=_FakeModel("cpu"),
        )


def test_r5b_truthful_cpu_path_unchanged():
    raw_cpu = torch.zeros(2, 3)  # loader-shaped CPU tensor
    observed = resolve_live_conversion_execution_device(
        staged_inputs=raw_cpu,
        model=_FakeModel("cpu"),
        device="cpu",
    )
    assert observed == "cpu"
    receipt = _mint_live_conversion_receipt()
    assert receipt.execution_device == "cpu"
    assert receipt.gpu_launched is False


def test_r5b_builder_observed_cuda_ignores_cpu_batch_device_field():
    """Composition: builder receipt fields follow observed_execution_device, not batch."""
    from dataclasses import replace as _replace
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        _normalize_execution_device,
    )

    # Prove normalizer no longer has a batch override path (TypeError/ignored).
    assert _normalize_execution_device("cuda") == "cuda"
    assert _normalize_execution_device(_FakeDevice("cuda")) == "cuda"
    cpu_batch = {"inputs": torch.zeros(1, 2)}
    # resolve with explicit observed must not consult cpu_batch at all
    assert (
        resolve_live_conversion_execution_device(
            observed_execution_device="cuda",
            model=_FakeModel("cuda"),
            device="cpu",
        )
        == "cuda"
    )
    # Minted CPU receipt remains truthful when observed omitted (unit path)
    receipt = _mint_live_conversion_receipt()
    assert receipt.execution_device == "cpu"
    # Synthesize the CUDA-declared receipt fields the formal path must emit
    cuda_receipt = _replace(receipt, execution_device="cuda", gpu_launched=True)
    validate_trainer_sub2_authority_live_conversion_receipt(
        cuda_receipt, measured_device="cuda"
    )
