"""Pure reducers for the vote-lifetime screen (classify-before-build).

Bound by PLAN_v1 sha 3b3848c8… + riders 1/2/3:
- episode age semantics (rider-2)
- executable classifier order (rider-2)
- lifetime_censored_frac episode-count formula (rider-3)
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

EPS = 1e-9
CREDITED_MASS_VACUOUS = 10_000
N_FLIPS_VACUOUS = 1000
CENSOR_FRAC_GUARD = 0.50
P50_CENSOR_GUARD = 32

LIFETIME_BIN_LABELS = (
    "1",
    "2",
    "3-4",
    "5-8",
    "9-16",
    "17-32",
    "33-64",
    "65-128",
    "129-150",
)

AGE_BIN_EDGES = (
    (0, 1, "0-1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 10_000, "65+"),
)

SURVIVAL_M = (1, 2, 5, 10, 20, 40, 80)

FAMILY_F1 = "F1_decay_leak"
FAMILY_F2 = "F2_ttl_age_drain"
FAMILY_F3 = "F3_sparse_hot_forgettable_cold"
FAMILY_F4 = "F4_underspecified_need_longer_or_coupled_q_motion"


def lifetime_bin_label(lifetime_steps: int) -> str:
    if lifetime_steps <= 1:
        return "1"
    if lifetime_steps == 2:
        return "2"
    if lifetime_steps <= 4:
        return "3-4"
    if lifetime_steps <= 8:
        return "5-8"
    if lifetime_steps <= 16:
        return "9-16"
    if lifetime_steps <= 32:
        return "17-32"
    if lifetime_steps <= 64:
        return "33-64"
    if lifetime_steps <= 128:
        return "65-128"
    return "129-150"


def age_bin_label(age: int) -> str:
    for lo, hi, lab in AGE_BIN_EDGES:
        if lo <= age <= hi:
            return lab
    return "65+"


def lifetime_censored_frac(n_flips: int, n_censored_active_episodes: int) -> float:
    """Rider-3: episode-count censor fraction (not never-convert mass)."""
    return float(n_censored_active_episodes) / float(
        max(1, int(n_flips) + int(n_censored_active_episodes))
    )


def update_episode_starts(
    prev_acc: torch.Tensor,
    new_acc: torch.Tensor,
    episode_start: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Apply rider-2 episode start/reset after move+clamp, before drain.

    - zero→nonzero: start episode at `step`
    - return to exactly zero: clear episode (0)
    - sign reversal: start new episode at `step`
    - else: keep prior episode_start
    """
    if prev_acc.shape != new_acc.shape or prev_acc.shape != episode_start.shape:
        raise ValueError("prev_acc, new_acc, episode_start shapes must match")
    out = episode_start.clone()
    prev_z = prev_acc == 0
    new_z = new_acc == 0
    # return to zero
    out = torch.where(new_z, torch.zeros_like(out), out)
    # zero → nonzero
    start_mask = prev_z & (~new_z)
    out = torch.where(start_mask, torch.full_like(out, int(step)), out)
    # sign reversal (both nonzero, signs differ)
    sign_flip = (~prev_z) & (~new_z) & (prev_acc.sign() != new_acc.sign())
    out = torch.where(sign_flip, torch.full_like(out, int(step)), out)
    return out


def apply_drain_resets(
    acc: torch.Tensor,
    episode_start: torch.Tensor,
    drain_mask: torch.Tensor,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Drain selected weights: record lifetimes, zero acc, clear episodes."""
    if acc.shape != episode_start.shape or acc.shape != drain_mask.shape:
        raise ValueError("acc, episode_start, drain_mask shapes must match")
    lifetimes: list[int] = []
    if bool(drain_mask.any()):
        starts = episode_start[drain_mask]
        # Only record when an episode was active (start > 0).
        active = starts > 0
        if bool(active.any()):
            ages = (int(step) - starts[active].to(torch.int64)).tolist()
            lifetimes.extend(int(x) for x in ages)
        acc = acc.clone()
        episode_start = episode_start.clone()
        acc[drain_mask] = 0
        episode_start[drain_mask] = 0
    return acc, episode_start, lifetimes


def histogram_lifetimes(lifetimes: Sequence[int]) -> dict[str, int]:
    hist = {lab: 0 for lab in LIFETIME_BIN_LABELS}
    for lt in lifetimes:
        hist[lifetime_bin_label(int(lt))] += 1
    return hist


def lifetime_quantiles(lifetimes: Sequence[int]) -> dict[str, float | None]:
    if not lifetimes:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90", "p99")}
    t = torch.tensor(sorted(int(x) for x in lifetimes), dtype=torch.float64)
    qs = torch.quantile(
        t,
        torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.99], dtype=torch.float64),
        interpolation="linear",
    )
    keys = ("p10", "p25", "p50", "p75", "p90", "p99")
    return {k: float(v) for k, v in zip(keys, qs.tolist())}


def classify_forgetting_family(
    *,
    n_flips: int,
    credited_mass: float,
    p50_flip_lifetime: float | None,
    never_convert_frac: float,
    age_rate_old: float,
    age_rate_young: float,
    lifetime_censored_frac_value: float,
) -> dict[str, Any]:
    """Deterministic eval order from rider-2 + rider-3 censor field."""
    out: dict[str, Any] = {
        "eps": EPS,
        "eval_order": [
            "R4_vacuous",
            "censor_guard",
            "R1_R2_R3_independent",
            "R5_priority_or_F4",
        ],
        "predicates": {},
        "family": FAMILY_F4,
        "multi_match": False,
        "stop_reason": None,
    }

    # (0) R4 vacuous guard
    if int(n_flips) < N_FLIPS_VACUOUS or float(credited_mass) < CREDITED_MASS_VACUOUS:
        out["predicates"]["R4_vacuous"] = True
        out["family"] = FAMILY_F4
        out["stop_reason"] = "R4_vacuous"
        return out
    out["predicates"]["R4_vacuous"] = False

    p50 = float("nan") if p50_flip_lifetime is None else float(p50_flip_lifetime)

    # (1) Censor guard uses lifetime_censored_frac ONLY (rider-3)
    censor_hit = (
        float(lifetime_censored_frac_value) >= CENSOR_FRAC_GUARD
        and p50 >= float(P50_CENSOR_GUARD)
    )
    out["predicates"]["censor_guard"] = censor_hit
    if censor_hit:
        out["family"] = FAMILY_F4
        out["stop_reason"] = "censor_guard"
        return out

    ratio = float(age_rate_old) / max(EPS, float(age_rate_young))
    out["age_rate_ratio_old_over_young"] = ratio

    r1 = (
        p50 <= 8.0
        and ratio <= 0.25
        and float(never_convert_frac) >= 0.50
    )
    r2 = p50 >= 32.0 and ratio >= 0.60
    r3 = float(never_convert_frac) >= 0.70 and p50 <= 16.0
    out["predicates"]["R1"] = bool(r1)
    out["predicates"]["R2"] = bool(r2)
    out["predicates"]["R3"] = bool(r3)

    matches = []
    if r1:
        matches.append(FAMILY_F2)
    if r2:
        matches.append(FAMILY_F1)
    if r3:
        matches.append(FAMILY_F3)

    if not matches:
        out["family"] = FAMILY_F4
        out["stop_reason"] = "no_rule_match"
        return out
    if len(matches) == 1:
        out["family"] = matches[0]
        out["stop_reason"] = "single_match"
        return out

    # R5 priority F3 > F2 > F1
    out["multi_match"] = True
    for fam in (FAMILY_F3, FAMILY_F2, FAMILY_F1):
        if fam in matches:
            out["family"] = fam
            out["stop_reason"] = "R5_priority"
            return out
    out["family"] = FAMILY_F4
    out["stop_reason"] = "R5_fallback"
    return out


def never_convert_metrics(
    final_acc: torch.Tensor,
    flip_count_per_weight: torch.Tensor,
) -> dict[str, float]:
    """Measurement (b): never-flipped final mass — NOT the censor fraction."""
    abs_acc = final_acc.abs().to(torch.float64)
    total = float(abs_acc.sum().clamp(min=1.0))
    never = flip_count_per_weight == 0
    drainable = float(abs_acc[never].sum())
    return {
        "never_convert_frac": drainable / total,
        "never_flipped_final_abs_mass": drainable,
        "final_abs_mass": total,
        "never_flipped_weight_frac": float(never.float().mean()),
    }


def count_censored_active_episodes(
    final_acc: torch.Tensor,
    episode_start: torch.Tensor,
) -> int:
    """Rider-3: nonzero acc alive at window end with an active episode."""
    return int(((final_acc != 0) & (episode_start > 0)).sum().item())


def empty_hazard_table() -> dict[str, dict[str, float | int]]:
    return {
        lab: {"flips": 0, "exposure": 0, "rate": 0.0}
        for _, _, lab in AGE_BIN_EDGES
    }


def accumulate_hazard(
    table: dict[str, dict[str, float | int]],
    ages: torch.Tensor,
    flip_mask: torch.Tensor,
) -> None:
    """Add exposure for all ages; flips for flip_mask rows."""
    if ages.numel() == 0:
        return
    for lo, hi, lab in AGE_BIN_EDGES:
        in_bin = (ages >= lo) & (ages <= hi)
        table[lab]["exposure"] = int(table[lab]["exposure"]) + int(in_bin.sum())
        table[lab]["flips"] = int(table[lab]["flips"]) + int(
            (in_bin & flip_mask).sum()
        )
    for lab, row in table.items():
        exp = max(1, int(row["exposure"]))
        row["rate"] = float(row["flips"]) / float(exp)


def age_rate_partition(
    table: Mapping[str, Mapping[str, float | int]],
    *,
    young_max: int = 8,
    old_min: int = 33,
) -> tuple[float, float]:
    young_f = young_e = old_f = old_e = 0
    for lo, hi, lab in AGE_BIN_EDGES:
        flips = int(table[lab]["flips"])
        exp = int(table[lab]["exposure"])
        if hi <= young_max:
            young_f += flips
            young_e += exp
        if lo >= old_min:
            old_f += flips
            old_e += exp
    young_rate = float(young_f) / float(max(1, young_e))
    old_rate = float(old_f) / float(max(1, old_e))
    return old_rate, young_rate


def empty_survival_table() -> dict[str, dict[str, dict[str, int]]]:
    # age_bin -> M -> {at_risk, event_ge_m}
    return {
        lab: {str(m): {"at_risk": 0, "ge_m": 0} for m in SURVIVAL_M}
        for _, _, lab in AGE_BIN_EDGES
    }


def accumulate_survival(
    table: dict[str, dict[str, dict[str, int]]],
    ages: torch.Tensor,
    abs_acc: torch.Tensor,
) -> None:
    if ages.numel() == 0:
        return
    for lo, hi, lab in AGE_BIN_EDGES:
        in_bin = (ages >= lo) & (ages <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        vals = abs_acc[in_bin]
        for m in SURVIVAL_M:
            table[lab][str(m)]["at_risk"] += n
            table[lab][str(m)]["ge_m"] += int((vals >= m).sum())


def survival_fractions(
    table: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for lab, by_m in table.items():
        out[lab] = {}
        for m, row in by_m.items():
            at_risk = int(row["at_risk"])
            if at_risk <= 0:
                out[lab][m] = None
            else:
                out[lab][m] = float(row["ge_m"]) / float(at_risk)
    return out
