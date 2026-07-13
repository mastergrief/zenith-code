"""CPU tests for W6/W4 clip-only dense-acc trainer boundary wiring (R8).

Covers co_lead T1–T9: defaults, mutex, exact clips, no-pack spy, selector
identity, W8 regression, real-write smoke, and fresh-process env isolation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    LIVE_ACC_CARRIER_NONE,
    LIVE_ACC_CARRIER_W4,
    LIVE_ACC_CARRIER_W6_CLIP_ONLY,
    LIVE_ACC_CARRIER_W8,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W4_SIGNED_MAX,
    W4_SIGNED_MIN,
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
    clip_then_roundtrip_w8_tensor,
    clip_to_w4_tensor,
    clip_to_w6_tensor,
    pack_w6_tensor,
    strict_roundtrip_w6_tensor,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    NARROW_BOUNDARY_NONE,
    NARROW_BOUNDARY_W4_CLIP_ONLY,
    NARROW_BOUNDARY_W6_CLIP_ONLY,
    NARROW_BOUNDARY_W8,
    RUN_NARROW_CARRIER_W4_CLIP_ONLY_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION_ENV,
    apply_trainer_boundary_narrow_carrier,
    narrow_carrier_w4_clip_only_enabled,
    narrow_carrier_w6_clip_only_enabled,
    resolve_narrow_carrier_boundary_selection,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    effective_clip_bounds,
    signed_w_max,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "scripts" / "hrm_text_158_bounded_delta_acquisition_probe.py"


def test_t3_effective_clips_exact() -> None:
    assert signed_w_max(8) == 127
    assert signed_w_max(6) == 31
    assert signed_w_max(4) == 7
    assert effective_clip_bounds(8, -127, 127) == (-127, 127)
    assert effective_clip_bounds(6, -127, 127) == (-31, 31)
    assert effective_clip_bounds(4, -127, 127) == (-7, 7)


def test_t1_t8_default_off_byte_shape_identity() -> None:
    acc = torch.tensor([80, -80, 12], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc)
    assert torch.equal(out, acc)
    assert out.dtype == torch.int16
    assert out.shape == acc.shape
    assert resolve_narrow_carrier_boundary_selection() == NARROW_BOUNDARY_NONE
    assert resolve_live_acc_carrier_selector() == LIVE_ACC_CARRIER_NONE


def test_t9_w6_clip_only_real_write_smoke() -> None:
    acc = torch.tensor([80, -80, 12], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w6_clip_only_enabled=True)
    assert out.tolist() == [W6_SIGNED_MAX, W6_SIGNED_MIN, 12]
    assert torch.equal(out, clip_to_w6_tensor(acc))


def test_t9_w4_clip_only_real_write_smoke() -> None:
    acc = torch.tensor([80, -80, 3], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w4_clip_only_enabled=True)
    assert out.tolist() == [W4_SIGNED_MAX, W4_SIGNED_MIN, 3]
    assert torch.equal(out, clip_to_w4_tensor(acc))


def test_t4_no_pack_spy_w6_clip_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"strict": 0, "pack": 0}

    def _strict(acc: torch.Tensor) -> torch.Tensor:
        calls["strict"] += 1
        return strict_roundtrip_w6_tensor(acc)

    def _pack(acc: torch.Tensor) -> torch.Tensor:
        calls["pack"] += 1
        return pack_w6_tensor(acc)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration.strict_roundtrip_w6_tensor",
        _strict,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.narrow_accumulator_codec.pack_w6_tensor",
        _pack,
    )
    acc = torch.tensor([80, -80], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w6_clip_only_enabled=True)
    assert out.tolist() == [31, -31]
    assert calls == {"strict": 0, "pack": 0}


def test_t4_no_pack_spy_w4_clip_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"strict": 0, "pack": 0}

    def _strict(acc: torch.Tensor) -> torch.Tensor:
        calls["strict"] += 1
        return strict_roundtrip_w6_tensor(acc)

    def _pack(acc: torch.Tensor) -> torch.Tensor:
        calls["pack"] += 1
        return pack_w6_tensor(acc)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration.strict_roundtrip_w6_tensor",
        _strict,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.narrow_accumulator_codec.pack_w6_tensor",
        _pack,
    )
    acc = torch.tensor([80, -80], dtype=torch.int16)
    out = apply_trainer_boundary_narrow_carrier(acc, w4_clip_only_enabled=True)
    assert out.tolist() == [7, -7]
    assert calls == {"strict": 0, "pack": 0}


def test_t2_mutex_all_pairs() -> None:
    acc = torch.tensor([10], dtype=torch.int16)
    pairs = [
        {"w6_clip_only_enabled": True, "w4_clip_only_enabled": True},
        {"w6_clip_only_enabled": True, "w7_enabled": True},
        {"w6_clip_only_enabled": True, "w8_enabled": True},
        {"w6_clip_only_enabled": True, "w6_enabled": True},
        {"w6_clip_only_enabled": True, "w5_enabled": True},
        {"w4_clip_only_enabled": True, "w7_enabled": True},
        {"w4_clip_only_enabled": True, "w8_enabled": True},
        {"w4_clip_only_enabled": True, "w6_enabled": True},
        {"w4_clip_only_enabled": True, "w5_enabled": True},
    ]
    for kwargs in pairs:
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_trainer_boundary_narrow_carrier(acc, **kwargs)
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_narrow_carrier_boundary_selection(**kwargs)
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_live_acc_carrier_selector(**kwargs)


def test_t2_mutex_with_v4() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(v4_enabled=True, w6_clip_only_enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_live_acc_carrier_selector(v4_enabled=True, w4_clip_only_enabled=True)
    with pytest.raises(ValueError, match="disabled when V4-LIVE"):
        apply_trainer_boundary_narrow_carrier(
            torch.tensor([1], dtype=torch.int16),
            w6_clip_only_enabled=True,
            v4_enabled=True,
        )


def test_t6_w8_bitexact_incumbent() -> None:
    acc = torch.tensor([200, -200, 33], dtype=torch.int16)
    expected = clip_then_roundtrip_w8_tensor(acc)
    out = apply_trainer_boundary_narrow_carrier(acc, w8_enabled=True)
    assert torch.equal(out, expected)
    assert resolve_narrow_carrier_boundary_selection(w8_enabled=True) == NARROW_BOUNDARY_W8
    assert resolve_live_acc_carrier_selector(w8_enabled=True) == LIVE_ACC_CARRIER_W8


def test_t7_runtime_telemetry_same_width_selection() -> None:
    assert (
        resolve_narrow_carrier_boundary_selection(w6_clip_only_enabled=True)
        == NARROW_BOUNDARY_W6_CLIP_ONLY
    )
    assert (
        resolve_live_acc_carrier_selector(w6_clip_only_enabled=True)
        == LIVE_ACC_CARRIER_W6_CLIP_ONLY
    )
    assert (
        resolve_narrow_carrier_boundary_selection(w4_clip_only_enabled=True)
        == NARROW_BOUNDARY_W4_CLIP_ONLY
    )
    assert resolve_live_acc_carrier_selector(w4_clip_only_enabled=True) == LIVE_ACC_CARRIER_W4


def test_env_flags_enable_clip_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION_ENV, "1")
    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, raising=False)
    assert narrow_carrier_w6_clip_only_enabled() is True
    acc = torch.tensor([80], dtype=torch.int16)
    assert apply_trainer_boundary_narrow_carrier(acc).tolist() == [31]

    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION_ENV, raising=False)
    monkeypatch.setenv(RUN_NARROW_CARRIER_W4_CLIP_ONLY_TRAINER_INTEGRATION_ENV, "1")
    assert narrow_carrier_w4_clip_only_enabled() is True
    assert apply_trainer_boundary_narrow_carrier(acc).tolist() == [7]


def test_t5_probe_help_registers_flags() -> None:
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0
    assert "--dense-accumulator-w6-clip" in proc.stdout
    assert "--dense-accumulator-w4-clip" in proc.stdout
    assert "--dense-accumulator-w8-clip" in proc.stdout


def test_fresh_process_probe_mutex_w6_w8_fails_before_trainer() -> None:
    """Illegal pair must fail-closed in a fresh process (no env leakage)."""

    code = r"""
import os, sys
sys.path.insert(0, os.environ['REPO_ROOT'])
from pathlib import Path
from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_c2p1_probe
os.environ['HRM_TEXT_158_RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION'] = '1'
try:
    run_c2p1_probe(
        parent=Path('calm/hrm/checkpoints/dummy.pt'),
        parent_sha256='0' * 64,
        scratch_root=Path('/tmp/w6_w4_clip_mutex_probe'),
        phase='w8-dense-acc-in-vivo-confirmation',
        device='cpu',
        eligible_scope='all-bitlinear',
        steps=1,
        batch_size=1,
        max_steps_hard=1,
        dense_accumulator_w6_clip=True,
        dense_accumulator_w8_clip=True,
        enabled=True,
    )
except ValueError as exc:
    print('MUTEX_OK', 'mutually exclusive' in str(exc).lower())
    raise SystemExit(0)
print('MUTEX_MISS')
raise SystemExit(2)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "REPO_ROOT": str(REPO_ROOT), "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MUTEX_OK True" in proc.stdout


def test_fresh_process_env_isolation_w6_clip_clears_stale_pack() -> None:
    code = r"""
import os, sys
sys.path.insert(0, os.environ['REPO_ROOT'])
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION_ENV,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    NARROW_BOUNDARY_W6_CLIP_ONLY,
    apply_trainer_boundary_narrow_carrier,
    resolve_narrow_carrier_boundary_selection,
)
import torch
# Stale pack env then clear-all-then-set-one contract (matches probe arming).
os.environ[RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV] = '1'
for key in (
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W5_TRAINER_INTEGRATION',
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION',
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION',
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W4_CLIP_ONLY_TRAINER_INTEGRATION',
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W7_TRAINER_INTEGRATION',
    'HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION',
):
    os.environ.pop(key, None)
os.environ[RUN_NARROW_CARRIER_W6_CLIP_ONLY_TRAINER_INTEGRATION_ENV] = '1'
assert os.environ.get(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV) is None
assert resolve_narrow_carrier_boundary_selection() == NARROW_BOUNDARY_W6_CLIP_ONLY
out = apply_trainer_boundary_narrow_carrier(torch.tensor([80], dtype=torch.int16))
assert out.tolist() == [31]
print('ISOLATION_OK')
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "REPO_ROOT": str(REPO_ROOT), "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ISOLATION_OK" in proc.stdout


def test_probe_flag_mutex_via_run_fn() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_c2p1_probe

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_c2p1_probe(
            parent=Path("calm/hrm/checkpoints/dummy.pt"),
            parent_sha256="0" * 64,
            scratch_root=Path("/tmp/w6_w4_clip_mutex"),
            phase="w8-dense-acc-in-vivo-confirmation",
            device="cpu",
            eligible_scope="all-bitlinear",
            steps=1,
            batch_size=1,
            max_steps_hard=1,
            dense_accumulator_w6_clip=True,
            dense_accumulator_w4_clip=True,
            enabled=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_c2p1_probe(
            parent=Path("calm/hrm/checkpoints/dummy.pt"),
            parent_sha256="0" * 64,
            scratch_root=Path("/tmp/w6_w4_clip_mutex2"),
            phase="w8-dense-acc-in-vivo-confirmation",
            device="cpu",
            eligible_scope="all-bitlinear",
            steps=1,
            batch_size=1,
            max_steps_hard=1,
            dense_accumulator_w6_clip=True,
            persistent_accumulator_w6_byte_packed=True,
            enabled=True,
        )
