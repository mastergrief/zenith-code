"""Fork-2 de-risk packet tests (selection, update, full lifecycle, R4/R6, sync audit)."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_geometry import (
    RUN3_REAL_ARM_SHAPES,
    run3_total_numel,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk import (
    DeviceLifecycleStore,
    cpu_store_from_shapes,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_selection_derisk import (
    composite_rank_key,
    cpu_oracle_project_and_update,
    project_and_update_acc_episode,
    select_topk_masks_deterministic,
    writeback_bridge_cpu_q,
    writeback_cpu_oracle,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

R6_TEST_ID = "test_r6_index_only_q_shadow_25step_invariant"


def _tiny_shapes() -> dict[str, tuple[int, int]]:
    return {
        "arm0.gqkv": (32, 16),
        "arm1.o": (16, 16),
        "arm2.down": (16, 24),
        "arm3.gate": (48, 16),
    }


def _assert_store_equiv(cpu, dev) -> None:
    assert cpu.aggregates.as_dict() == dev.aggregates_as_dict()
    for n in cpu.first_deferral_step:
        assert torch.equal(cpu.first_deferral_step[n], dev.first_deferral_step[n].cpu())
        assert torch.equal(
            cpu.applied_after_deferral_step[n], dev.applied_after_deferral_step[n].cpu()
        )
        assert torch.equal(cpu.episode_generation[n], dev.episode_generation[n].cpu())


def _run_full_lifecycle_parity(device: str) -> None:
    """Shared adversarial fixture: frozen CPU oracle vs DeviceLifecycleStore(device)."""
    shapes = _tiny_shapes()
    steps = 25
    cpu = cpu_store_from_shapes(shapes, steps=steps)
    dev = DeviceLifecycleStore.from_arm_shapes(shapes, steps=steps, device=device)
    thr = int(CROSSING_THRESHOLD_ABS)
    to_dev = (lambda t: t.to(device)) if device != "cpu" else (lambda t: t)

    for t in range(1, 8):
        acc = {
            n: torch.full(shape, thr if (t % 2) else 0, dtype=torch.int16)
            for n, shape in shapes.items()
        }
        acc["arm0.gqkv"].reshape(-1)[:10] = thr + 2
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
        _assert_store_equiv(cpu, dev)

    applied = {n: torch.zeros(shape, dtype=torch.bool) for n, shape in shapes.items()}
    for n in shapes:
        open_m = cpu.first_deferral_step[n] > 0
        if bool(open_m.any()):
            idxs = torch.nonzero(open_m.reshape(-1), as_tuple=False).flatten()[:4]
            applied[n].reshape(-1)[idxs] = True
    residual_zero = {
        n: torch.zeros(shape, dtype=torch.bool) for n, shape in shapes.items()
    }
    a0 = applied["arm0.gqkv"].reshape(-1)
    idxs = torch.nonzero(a0, as_tuple=False).flatten()
    if idxs.numel() >= 2:
        residual_zero["arm0.gqkv"].reshape(-1)[idxs[0]] = True

    cpu.close_before_writeback_resets(
        applied_masks=applied, step=8, residual_zero=residual_zero
    )
    dev.close_before_writeback_resets(
        applied_masks={n: to_dev(v) for n, v in applied.items()},
        step=8,
        residual_zero={n: to_dev(v) for n, v in residual_zero.items()},
    )
    _assert_store_equiv(cpu, dev)

    ep_before = {n: torch.zeros(shape, dtype=torch.int32) for n, shape in shapes.items()}
    ep_after = {n: t.clone() for n, t in ep_before.items()}
    for n, m in applied.items():
        ep_before[n][m] = 3
        ep_after[n][m] = 8
    for n, m in applied.items():
        cpu.first_deferral_step[n][m] = 2
        cpu.applied_after_deferral_step[n][m] = 0
        dev.first_deferral_step[n][m] = 2
        dev.applied_after_deferral_step[n][m] = 0
        cpu.episode_generation[n][m] = 1
        dev.episode_generation[n][m] = 1
    cpu.roll_tracker_after_writeback(
        applied_masks=applied,
        episode_start_before=ep_before,
        episode_start_after=ep_after,
        step=8,
    )
    dev.roll_tracker_after_writeback(
        applied_masks={n: to_dev(v) for n, v in applied.items()},
        episode_start_before={n: to_dev(v) for n, v in ep_before.items()},
        episode_start_after={n: to_dev(v) for n, v in ep_after.items()},
        step=8,
    )
    _assert_store_equiv(cpu, dev)
    for n, m in applied.items():
        if bool(m.any()):
            assert int(cpu.episode_generation[n][m].max().item()) >= 2

    cpu.finalize_window(final_step=steps)
    dev.finalize_window(final_step=steps)
    _assert_store_equiv(cpu, dev)
    for n in shapes:
        assert int((cpu.first_deferral_step[n] > 0).sum().item()) == 0


def test_run3_geometry_is_multi_arm_not_flat():
    assert len(RUN3_REAL_ARM_SHAPES) == 32
    assert run3_total_numel() == 29_360_128
    assert run3_total_numel() != 18_158_319


def test_r1_composite_key_ordered_sequence_abs_desc_index_asc():
    abs_acc = torch.tensor([5, 5, 5, 4, 5], dtype=torch.int64)
    idx = torch.arange(5, dtype=torch.int64)
    keys = composite_rank_key(abs_acc, idx)
    order = torch.argsort(keys, descending=True)
    assert order.tolist() == [0, 1, 2, 4, 3]


def test_select_topk_ordered_sequence_tie_saturated_cpu():
    shapes = _tiny_shapes()
    acc = {n: torch.zeros(shape, dtype=torch.int16) for n, shape in shapes.items()}
    thr = int(CROSSING_THRESHOLD_ABS)
    flat = acc["arm0.gqkv"].reshape(-1)
    flat[:] = thr
    flat[10] = thr + 2
    flat[3] = thr + 2
    _c, _a, n_cand, n_app, ordered = select_topk_masks_deterministic(
        acc, topk=8, threshold=thr
    )
    assert n_cand == flat.numel()
    assert n_app == 8
    abs_all = torch.cat([a.abs().reshape(-1) for a in acc.values()])
    cross = torch.nonzero(abs_all >= thr, as_tuple=False).flatten()
    keys = composite_rank_key(abs_all[cross], cross)
    expect = cross[torch.argsort(keys, descending=True)[:8]]
    assert ordered.tolist() == expect.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_select_topk_cpu_cuda_identical_index_sets_and_order():
    shapes = _tiny_shapes()
    thr = int(CROSSING_THRESHOLD_ABS)
    acc_cpu = {n: torch.full(shape, thr, dtype=torch.int16) for n, shape in shapes.items()}
    acc_cpu["arm0.gqkv"].reshape(-1)[::17] = thr + 3
    acc_cuda = {n: t.cuda() for n, t in acc_cpu.items()}
    *_r, ord_cpu = select_topk_masks_deterministic(acc_cpu, topk=64, threshold=thr)
    *_r2, ord_cuda = select_topk_masks_deterministic(acc_cuda, topk=64, threshold=thr)
    assert ord_cpu.tolist() == ord_cuda.cpu().tolist()


def test_b1_update_seam_parity_vs_cpu_oracle():
    shapes = _tiny_shapes()
    step = 3
    q = {n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()}
    grads = {n: torch.randn(shape, dtype=torch.float32) for n, shape in shapes.items()}
    acc = {n: torch.zeros(shape, dtype=torch.int16) for n, shape in shapes.items()}
    ep = {n: torch.zeros(shape, dtype=torch.int32) for n, shape in shapes.items()}
    shadow = {n: q[n].clone() for n in q}
    na, ne, mv, _sync = project_and_update_acc_episode(
        grads=grads,
        q_auth_cpu=q,
        q_shadow=shadow,
        acc=acc,
        episode_start=ep,
        step=step,
        q_shadow_mode="index_only",
    )
    oa, oe, om = cpu_oracle_project_and_update(
        grads_cpu=grads, q_cpu=q, acc_cpu=acc, episode_cpu=ep, step=step
    )
    for n in shapes:
        assert torch.equal(na[n], oa[n])
        assert torch.equal(ne[n], oe[n])
        assert torch.equal(mv[n], om[n])


def test_r2_writeback_bridge_matches_production_oracle_cpu():
    shapes = _tiny_shapes()
    acc = {n: torch.randint(-20, 21, shape, dtype=torch.int16) for n, shape in shapes.items()}
    ep = {n: torch.zeros(shape, dtype=torch.int32) for n, shape in shapes.items()}
    q = {n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()}
    masks = {n: torch.zeros(shape, dtype=torch.bool) for n, shape in shapes.items()}
    masks["arm0.gqkv"].reshape(-1)[:8] = True
    acc_o = {n: t.clone() for n, t in acc.items()}
    ep_o = {n: t.clone() for n, t in ep.items()}
    q_o = {n: t.clone() for n, t in q.items()}
    writeback_bridge_cpu_q(
        acc=acc, episode_start=ep, q_auth_cpu=q, q_shadow=None, applied_masks=masks, step=5
    )
    oa, oe, oq = writeback_cpu_oracle(
        acc_cpu=acc_o, episode_cpu=ep_o, q_cpu=q_o, applied_masks_cpu=masks, step=5
    )
    for n in shapes:
        assert torch.equal(acc[n], oa[n])
        assert torch.equal(ep[n], oe[n])
        assert torch.equal(q[n], oq[n])


def test_b2_full_lifecycle_phases_parity_vs_cpu_oracle():
    """All phases on CPU device-store vs frozen CPU oracle."""
    _run_full_lifecycle_parity("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_b2_full_lifecycle_phases_parity_vs_cpu_oracle_cuda():
    """CUDA DeviceLifecycleStore vs frozen CPU oracle (copy at assert boundaries only)."""
    _run_full_lifecycle_parity("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_b2_cross_arm_topk_sync_audit_batched_one_d2h():
    """Equal-demand across arms — ONE global idx/dir D2H; ≤2 publishes."""
    shapes = {f"arm{i}": (64, 8) for i in range(8)}
    device = torch.device("cuda:0")
    thr = int(CROSSING_THRESHOLD_ABS)
    acc = {
        n: torch.full(shape, thr, dtype=torch.int16, device=device)
        for n, shape in shapes.items()
    }
    for i, n in enumerate(shapes):
        acc[n].reshape(-1)[i] = thr + 1
    cand, applied, n_cand, n_app, ordered = select_topk_masks_deterministic(
        acc, topk=8, threshold=thr
    )
    arms_hit = sum(int(m.any().item()) for m in applied.values())
    assert arms_hit >= 2, "fixture must exercise cross-arm applied"
    q_cpu = {n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()}
    q_shadow = {n: t.to(device) for n, t in q_cpu.items()}
    acc_w = {
        n: torch.randint(-20, 21, shape, dtype=torch.int16, device=device)
        for n, shape in shapes.items()
    }
    ep = {n: torch.zeros(shape, dtype=torch.int32, device=device) for n, shape in shapes.items()}
    stats = writeback_bridge_cpu_q(
        acc=acc_w,
        episode_start=ep,
        q_auth_cpu=q_cpu,
        q_shadow=q_shadow,
        applied_masks=applied,
        step=3,
        refresh_shadow_index_only=True,
    )
    assert stats["batched_global_d2h"] is True
    assert stats["scalar_item_publishes"] <= 2, stats
    assert stats["n_arms_with_applied"] >= 2
    # Structural kill of per-arm multiplier: exactly ONE idx + ONE dir D2H.
    assert stats["idx_d2h_count"] == 1
    assert stats["dir_d2h_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_r2_batched_writeback_matches_production_oracle_cuda():
    shapes = _tiny_shapes()
    device = torch.device("cuda:0")
    thr = int(CROSSING_THRESHOLD_ABS)
    acc_cpu = {
        n: torch.randint(-20, 21, shape, dtype=torch.int16) for n, shape in shapes.items()
    }
    ep_cpu = {n: torch.zeros(shape, dtype=torch.int32) for n, shape in shapes.items()}
    q_cpu = {n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()}
    masks_cpu = {n: torch.zeros(shape, dtype=torch.bool) for n, shape in shapes.items()}
    # Spread applied across arms
    masks_cpu["arm0.gqkv"].reshape(-1)[:4] = True
    masks_cpu["arm1.o"].reshape(-1)[:3] = True
    masks_cpu["arm2.down"].reshape(-1)[:2] = True
    acc = {n: t.to(device) for n, t in acc_cpu.items()}
    ep = {n: t.to(device) for n, t in ep_cpu.items()}
    q = {n: t.clone() for n, t in q_cpu.items()}
    masks = {n: t.to(device) for n, t in masks_cpu.items()}
    q_shadow = {n: t.to(device) for n, t in q.items()}
    writeback_bridge_cpu_q(
        acc=acc,
        episode_start=ep,
        q_auth_cpu=q,
        q_shadow=q_shadow,
        applied_masks=masks,
        step=5,
    )
    oa, oe, oq = writeback_cpu_oracle(
        acc_cpu={n: t.clone() for n, t in acc_cpu.items()},
        episode_cpu={n: t.clone() for n, t in ep_cpu.items()},
        q_cpu={n: t.clone() for n, t in q_cpu.items()},
        applied_masks_cpu=masks_cpu,
        step=5,
    )
    for n in shapes:
        assert torch.equal(acc[n].cpu(), oa[n])
        assert torch.equal(ep[n].cpu(), oe[n])
        assert torch.equal(q[n], oq[n])
        assert torch.equal(q_shadow[n].cpu(), oq[n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_r6_index_only_q_shadow_25step_invariant():
    shapes = _tiny_shapes()
    device = torch.device("cuda:0")
    q_cpu = {n: torch.randint(-1, 2, shape, dtype=torch.int8) for n, shape in shapes.items()}
    q_shadow = {n: t.to(device) for n, t in q_cpu.items()}
    acc = {n: torch.zeros(shape, dtype=torch.int16, device=device) for n, shape in shapes.items()}
    ep = {n: torch.zeros(shape, dtype=torch.int32, device=device) for n, shape in shapes.items()}
    thr = int(CROSSING_THRESHOLD_ABS)
    for step in range(1, 26):
        grads = {
            n: torch.randn(shape, dtype=torch.float32, device=device)
            for n, shape in shapes.items()
        }
        na, ne, _mv, _s = project_and_update_acc_episode(
            grads=grads,
            q_auth_cpu=q_cpu,
            q_shadow=q_shadow,
            acc=acc,
            episode_start=ep,
            step=step,
            q_shadow_mode="index_only",
        )
        acc, ep = na, ne
        _c, applied, _nc, _na, _o = select_topk_masks_deterministic(
            acc, topk=8, threshold=thr
        )
        writeback_bridge_cpu_q(
            acc=acc,
            episode_start=ep,
            q_auth_cpu=q_cpu,
            q_shadow=q_shadow,
            applied_masks=applied,
            step=step,
            refresh_shadow_index_only=True,
        )
        for n in shapes:
            assert torch.equal(q_shadow[n].cpu(), q_cpu[n]), f"shadow drift at step {step}"
