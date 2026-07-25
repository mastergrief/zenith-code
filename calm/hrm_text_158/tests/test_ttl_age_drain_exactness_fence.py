"""CPU exactness fence for `apply_ttl_age_drain` (F2 / arm2_ttl_age_drain).

Pins live semantics from forgetting_laws.apply_ttl_age_drain:
  age = step - episode_start
  drain iff (episode_start > 0) AND (age > ttl)   # strict >; age==ttl preserved
  drained indices: acc:=0 and episode_start:=0

Also measures the A_ttl vs Z_full_suppression_like discrimination boundary:
  age-conditional zeroing preserves age<=T episodes; age-independent wipe does not.
"""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    apply_decay_leak,
    apply_ttl_age_drain,
)

TTL_PREREG = 32


def _exact_oracle(
    acc: list[int],
    episode_start: list[int],
    *,
    step: int,
    ttl: int = TTL_PREREG,
) -> tuple[list[int], list[int], list[bool]]:
    """Pure-Python oracle for force-zero any active episode with age > T."""
    new_acc: list[int] = []
    new_ep: list[int] = []
    drained: list[bool] = []
    for a, ep in zip(acc, episode_start, strict=True):
        age = int(step) - int(ep)
        old = (int(ep) > 0) and (age > int(ttl))
        if old:
            new_acc.append(0)
            new_ep.append(0)
            drained.append(True)
        else:
            new_acc.append(int(a))
            new_ep.append(int(ep))
            drained.append(False)
    return new_acc, new_ep, drained


def _age_independent_wipe(
    acc: torch.Tensor,
    episode_start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-class contrast: zero every active episode regardless of age."""
    active = episode_start > 0
    new_acc = acc.clone()
    new_ep = episode_start.clone()
    new_acc[active] = 0
    new_ep[active] = 0
    return new_acc, new_ep


def test_ttl_oracle_matches_impl_on_boundary_fixture() -> None:
    step = 200
    ages = [1, 16, 31, 32, 33, 40, 100]
    acc = torch.tensor([7, -5, 3, 9, -2, 4, 1], dtype=torch.int16)
    ep = torch.tensor([step - a for a in ages], dtype=torch.int32)
    # all episode_starts stay > 0 (step - age >= 100 > 0)
    assert all(int(x) > 0 for x in ep.tolist())
    want_acc, want_ep, drained = _exact_oracle(
        acc.tolist(), ep.tolist(), step=step, ttl=TTL_PREREG
    )
    got_acc, got_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=TTL_PREREG)
    assert got_acc.tolist() == want_acc
    assert got_ep.tolist() == want_ep
    # age==ttl preserved; age==ttl+1 drained
    assert drained == [False, False, False, False, True, True, True]
    assert got_acc[3].item() == 9 and got_ep[3].item() == step - 32
    assert got_acc[4].item() == 0 and got_ep[4].item() == 0


def test_inactive_episode_start_zero_never_drains() -> None:
    """True-negative: episode_start==0 is inactive even when step is huge."""
    step = 10_000
    acc = torch.tensor([11, -9, 5], dtype=torch.int16)
    ep = torch.zeros(3, dtype=torch.int32)
    got_acc, got_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=TTL_PREREG)
    want_acc, want_ep, drained = _exact_oracle(
        acc.tolist(), ep.tolist(), step=step, ttl=TTL_PREREG
    )
    assert drained == [False, False, False]
    assert got_acc.tolist() == want_acc == acc.tolist()
    assert got_ep.tolist() == want_ep == [0, 0, 0]


def test_exhaustive_small_grid_zero_mismatches() -> None:
    """Exhaustive CPU sweep over a finite support (dual-oracle: impl == pure Python)."""
    mismatches = 0
    cases = 0
    ttl = TTL_PREREG
    for step in range(0, 80):
        for ep0 in range(0, 80):
            for aval in (-127, -1, 0, 1, 63, 127):
                acc = torch.tensor([aval], dtype=torch.int16)
                ep = torch.tensor([ep0], dtype=torch.int32)
                got_acc, got_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=ttl)
                want_acc, want_ep, _ = _exact_oracle(
                    [aval], [ep0], step=step, ttl=ttl
                )
                cases += 1
                if got_acc.tolist() != want_acc or got_ep.tolist() != want_ep:
                    mismatches += 1
                    if mismatches <= 5:
                        raise AssertionError(
                            f"mismatch step={step} ep={ep0} aval={aval}: "
                            f"got=({got_acc.tolist()},{got_ep.tolist()}) "
                            f"want=({want_acc},{want_ep})"
                        )
    assert cases == 80 * 80 * 6
    assert mismatches == 0


def test_a_ttl_vs_z_discrimination_measurement() -> None:
    """A/Z boundary observable: young (age<=T) preserved under TTL; wiped under Z.

    Measurement:
      young_preserved_ttl = (post_ttl[young] == pre[young])
      young_wiped_z       = (post_z[young] == 0)
      old_drained_ttl     = (post_ttl[old] == 0)
    A_ttl requires young_preserved_ttl AND old_drained_ttl.
    Z_full_suppression_like is age-independent wipe (young also zeroed).
    """
    step = 200
    ttl = TTL_PREREG
    # one young (age=10 <= 32), one old (age=50 > 32)
    ages = [10, 50]
    acc = torch.tensor([17, -19], dtype=torch.int16)
    ep = torch.tensor([step - a for a in ages], dtype=torch.int32)

    ttl_acc, ttl_ep = apply_ttl_age_drain(acc, ep, step=step, ttl=ttl)
    z_acc, z_ep = _age_independent_wipe(acc, ep)

    young_preserved_ttl = bool(ttl_acc[0].item() == 17 and ttl_ep[0].item() == step - 10)
    old_drained_ttl = bool(ttl_acc[1].item() == 0 and ttl_ep[1].item() == 0)
    young_wiped_z = bool(z_acc[0].item() == 0 and z_ep[0].item() == 0)
    old_wiped_z = bool(z_acc[1].item() == 0 and z_ep[1].item() == 0)

    assert young_preserved_ttl and old_drained_ttl, "A_ttl age-conditional failed"
    assert young_wiped_z and old_wiped_z, "Z age-independent wipe failed"
    # Discrimination: same pre-state → different young post-state across operators
    assert ttl_acc[0].item() != z_acc[0].item()
    assert young_preserved_ttl and young_wiped_z


def test_decay_leak_is_age_independent_contrast() -> None:
    """Soft-decay (F1) changes young and old alike — must not be mixed into F2."""
    step = 200
    acc = torch.tensor([32, 32], dtype=torch.int16)
    ep = torch.tensor([step - 10, step - 50], dtype=torch.int32)  # young, old
    ttl_acc, _ = apply_ttl_age_drain(acc, ep, step=step, ttl=TTL_PREREG)
    decayed = apply_decay_leak(acc, lam=1.0 / 32.0)
    # TTL: young preserved at 32, old zeroed
    assert ttl_acc.tolist() == [32, 0]
    # decay: both shrink (age-independent) — forbidden mix-in signal for F2
    assert decayed.tolist() == [31, 31]
    assert decayed[0].item() != ttl_acc[0].item() or decayed[1].item() != ttl_acc[1].item()


def test_default_ttl_is_prereg_32() -> None:
    step = 100
    acc = torch.tensor([5, 6], dtype=torch.int16)
    ep = torch.tensor([step - 32, step - 33], dtype=torch.int32)
    got_acc, got_ep = apply_ttl_age_drain(acc, ep, step=step)  # default ttl
    assert got_acc.tolist() == [5, 0]
    assert got_ep.tolist() == [step - 32, 0]
