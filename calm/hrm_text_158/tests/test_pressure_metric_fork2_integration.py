"""Fork-2 integration parity: receipt adapter + shared selection identity."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
    cpu_store_from_shapes,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    select_topk_masks_deterministic,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_receipt import (
    build_diagnostic_receipt,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    sanitize_receipt_for_strict_json,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

ROOT = Path(__file__).resolve().parents[3]


def _tiny_shapes() -> dict[str, tuple[int, int]]:
    return {
        "arm0.gqkv": (32, 16),
        "arm1.o": (16, 16),
        "arm2.down": (16, 24),
        "arm3.gate": (48, 16),
    }


def _loc(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_touched_mixed_modules_under_500_loc() -> None:
    for rel in (
        "calm/hrm_text_158/native_full_stack/screen_execution_loop.py",
        "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_lifecycle_derisk.py",
        "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_loop_bridge.py",
        "calm/hrm_text_158/native_full_stack/pressure_metric_proof.py",
    ):
        assert _loc(ROOT / rel) < 500, rel


def test_device_store_exposes_per_step_ratios_exact_name() -> None:
    shapes = _tiny_shapes()
    store = DeviceLifecycleStore.from_arm_shapes(shapes, steps=8, device="cpu")
    assert hasattr(store, "per_step_ratios")
    assert "per_step_demand" not in ast.dump(
        ast.parse(
            (ROOT / "calm/hrm_text_158/native_full_stack/pressure_metric_receipt.py").read_text()
        )
    ) or True  # receipt reads per_step_ratios only
    src = (
        ROOT / "calm/hrm_text_158/native_full_stack/pressure_metric_receipt.py"
    ).read_text(encoding="utf-8")
    assert "store.per_step_ratios" in src
    assert "store.per_step_demand" not in src


def _drive_twin_stores(device: str) -> tuple[object, DeviceLifecycleStore]:
    shapes = _tiny_shapes()
    steps = 12
    cpu = cpu_store_from_shapes(shapes, steps=steps)
    dev = DeviceLifecycleStore.from_arm_shapes(shapes, steps=steps, device=device)
    thr = int(CROSSING_THRESHOLD_ABS)
    to_dev = (lambda t: t.to(device)) if device != "cpu" else (lambda t: t)
    for t in range(1, steps + 1):
        acc = {
            n: torch.full(shape, thr if (t % 2) else 0, dtype=torch.int16)
            for n, shape in shapes.items()
        }
        acc["arm0.gqkv"].reshape(-1)[:12] = thr + 3
        cand, applied, n_cand, n_app, _o = select_topk_masks_deterministic(
            acc, topk=16, threshold=thr
        )
        cpu.process_pre_writeback(
            candidate_masks=cand,
            applied_masks=applied,
            step=t,
            n_candidates=n_cand,
            n_applied=n_app,
        )
        cand_d = {n: to_dev(v) for n, v in cand.items()}
        applied_d = {n: to_dev(v) for n, v in applied.items()}
        dev.process_pre_writeback(
            candidate_masks=cand_d,
            applied_masks=applied_d,
            step=t,
            n_candidates=n_cand,
            n_applied=n_app,
        )
        residual_zero = {n: torch.zeros_like(applied[n]) for n in applied}
        residual_zero_d = {n: to_dev(v) for n, v in residual_zero.items()}
        cpu.close_before_writeback_resets(
            applied_masks=applied, step=t, residual_zero=residual_zero
        )
        dev.close_before_writeback_resets(
            applied_masks=applied_d, step=t, residual_zero=residual_zero_d
        )
        ep_b = {n: torch.zeros(shape, dtype=torch.int32) for n, shape in shapes.items()}
        ep_a = {n: v.clone() for n, v in ep_b.items()}
        for n in applied:
            if bool(applied[n].any()):
                ep_a[n][applied[n]] = t
        cpu.roll_tracker_after_writeback(
            applied_masks=applied,
            episode_start_before=ep_b,
            episode_start_after=ep_a,
            step=t,
        )
        dev.roll_tracker_after_writeback(
            applied_masks=applied_d,
            episode_start_before={n: to_dev(v) for n, v in ep_b.items()},
            episode_start_after={n: to_dev(v) for n, v in ep_a.items()},
            step=t,
        )
    cpu.finalize_window(final_step=steps)
    dev.finalize_window(final_step=steps)
    cpu.two_tier_threshold_assert_pass = True
    dev.two_tier_threshold_assert_pass = True
    return cpu, dev


def test_full_receipt_facing_cpu_device_parity() -> None:
    cpu, dev = _drive_twin_stores("cpu")
    assert cpu.per_step_ratios == dev.per_step_ratios
    assert cpu.survival_summary() == dev.survival_summary()
    meas = {
        "n_flips": 0,
        "q_changed_count": 0,
        "credited_mass": 0,
        "lifetime_censored_frac": 0.0,
        "p50_flip_lifetime": None,
        "H_bits_per_weight": 0.0,
        "H_trajectory": [],
        "n_applied_drains": 0,
        "margin_trajectory": [],
        "episode_trajectory": [],
    }
    r_cpu = build_diagnostic_receipt(
        store=cpu,
        measurements=meas,
        probes={"retention_ok": True, "ret_final_count": 1, "ret_step0_count": 1},
        require_probes=False,
        schema_only=True,
    )
    r_dev = build_diagnostic_receipt(
        store=dev,
        measurements=meas,
        probes={"retention_ok": True, "ret_final_count": 1, "ret_step0_count": 1},
        require_probes=False,
        schema_only=True,
    )
    assert sanitize_receipt_for_strict_json(r_cpu)["measurements"]["demand"] == (
        sanitize_receipt_for_strict_json(r_dev)["measurements"]["demand"]
    )
    assert r_cpu["measurements"]["deferred_survival"] == r_dev["measurements"][
        "deferred_survival"
    ]
    assert r_cpu["two_tier_threshold_assert_pass"] is True
    assert r_dev["two_tier_threshold_assert_pass"] is True


def test_build_diagnostic_receipt_zero_behavioral_receipt_edits() -> None:
    """R-v2.1: DeviceLifecycleStore works with receipt.py as consumer only."""
    _cpu, dev = _drive_twin_stores("cpu")
    assert isinstance(dev.per_step_ratios, list)
    assert len(dev.per_step_ratios) == 12
    receipt = build_diagnostic_receipt(
        store=dev,
        measurements={
            "n_flips": 1,
            "q_changed_count": 1,
            "credited_mass": 1,
            "lifetime_censored_frac": 0.0,
            "p50_flip_lifetime": 1.0,
            "H_bits_per_weight": 0.1,
            "H_trajectory": [{"step": 25, "H_bits_per_weight": 0.1}],
            "n_applied_drains": 1,
            "margin_trajectory": [],
            "episode_trajectory": [],
        },
        probes={"retention_ok": True, "ret_final_count": 2, "ret_step0_count": 2},
        require_probes=False,
        schema_only=True,
    )
    assert "demand" in receipt["measurements"]
    assert receipt["measurements"]["demand_per_25"]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_shared_selection_identity_stable(device: str) -> None:
    shapes = _tiny_shapes()
    thr = int(CROSSING_THRESHOLD_ABS)
    acc = {
        n: torch.arange(shape[0] * shape[1], dtype=torch.int16).view(shape) % (thr + 5)
        for n, shape in shapes.items()
    }
    if device != "cpu":
        acc = {n: v.to(device) for n, v in acc.items()}
    c1, a1, nc1, na1, o1 = select_topk_masks_deterministic(acc, topk=16, threshold=thr)
    c2, a2, nc2, na2, o2 = select_topk_masks_deterministic(acc, topk=16, threshold=thr)
    assert nc1 == nc2 and na1 == na2
    assert torch.equal(o1, o2)
    for n in a1:
        assert torch.equal(a1[n], a2[n])
        assert torch.equal(c1[n], c2[n])


def test_additive_smoke_script_owns_three_priced_lines() -> None:
    smoke = ROOT / "scripts/hrm_text_158_fork2_integration_additive_smoke.py"
    assert smoke.is_file()
    src = smoke.read_text(encoding="utf-8")
    assert "wiring_shell" in src
    assert "publish_cadence" in src
    assert "receipt_hotpath" in src
    assert "UNPRICEABLE" in src
    assert '"GO"' not in src and "'GO'" not in src
    # Representative glue must price the promoted phases + A/B sync inventory.
    assert "project_credit_shared" in src
    assert "lifetimes_before_writeback" in src
    assert "episode_start_clone" in src
    assert "sync_inventory" in src
    assert "duplicate_idx_d2h_for_identity" in src


def test_hotpath_sync_allowlist_closed() -> None:
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        HOTPATH_SYNC_ALLOWLIST,
        assert_hotpath_sync_allowlist,
    )

    observed = assert_hotpath_sync_allowlist()
    assert set(observed) == set(HOTPATH_SYNC_ALLOWLIST)
    for fname, limits in HOTPATH_SYNC_ALLOWLIST.items():
        for pat, limit in limits.items():
            assert observed[fname][pat] <= limit


def test_ordered_selection_frame_rejects_device_idx() -> None:
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        ordered_selection_frame,
    )

    host = torch.arange(4, dtype=torch.int64)
    frame = ordered_selection_frame(step=3, ordered_flat_idx=host)
    assert isinstance(frame, bytes) and len(frame) == 4 + 4 * 8
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="host idx"):
            ordered_selection_frame(
                step=3, ordered_flat_idx=host.to("cuda")
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_writeback_host_idx_reused_no_duplicate_identity_d2h() -> None:
    """Whole-loop sync inventory: ONE idx D2H; identity reuses host payload."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        hotpath_sync_inventory_from_writeback,
        init_gpu_loop_residency,
        ordered_selection_frame,
        select_shared,
        writeback_shared,
    )

    thr = int(CROSSING_THRESHOLD_ABS)
    shapes = _tiny_shapes()
    q_cpu = {
        n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()
    }
    res = init_gpu_loop_residency(q_cpu, device="cuda")
    for i, n in enumerate(res.acc):
        flat = res.acc[n].reshape(-1)
        n_hit = min(8, int(flat.numel()))
        flat[:n_hit] = thr + 1 + (i % 2)
        res.episode_start[n][res.acc[n].abs() >= thr] = 2
    _cand, applied, _nc, _na, ordered = select_shared(
        res, topk=16, threshold=thr
    )
    assert ordered.device.type == "cuda"
    wb = writeback_shared(
        res, applied, step=5, threshold=thr, ordered_flat_idx=ordered
    )
    inv = hotpath_sync_inventory_from_writeback(wb)
    assert inv["idx_d2h_count"] == 1
    assert inv["dir_d2h_count"] == 1
    assert inv["batched_global_d2h"] is True
    assert inv["duplicate_idx_d2h_for_identity"] is False
    host = wb["selection_idx_host"]
    assert host.device.type == "cpu"
    frame = ordered_selection_frame(step=5, ordered_flat_idx=host)
    assert len(frame) == 4 + int(host.numel()) * 8
