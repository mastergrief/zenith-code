"""Phase-1 probe registry + train-exclusion helpers (PLAN_v9).

Extracted behavior-preservingly from forgetting_mechanism_screen_reducers.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Sequence

Row3 = tuple[str, int, str]

# PLAN_v9 probe registry pins
MATH_A0_PARENT_SUPPORT_HASH = "56e64266357b793d"
IDENTITY_PARENT_SUPPORT_HASH = "bf43ff7354b64c4e"
ACQUISITION_SELECTION_SHA256 = (
    "b73f532bc17facfba93426b6c9cdff7e1e0cfc244eb2b0c5a57c2a8c1489237b"
)
IDENTITY_SELECTION_SHA256 = (
    "3a137eb6e99784e1b0d5e26deb174f816d816607c38e00262dbeebde96b743f2"
)
ACQ_N = 64
RET_MATH_N = 32
RET_ID_N = 32
ACQ_SHUFFLE_SEED = 7
RET_SHUFFLE_SEED = 11


def compact_rows_sha256(rows: Sequence[Row3]) -> str:
    """PLAN_v9 selection sha: sha256(json.dumps(rows, separators=(',',':')))."""
    return hashlib.sha256(
        json.dumps(list(rows), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parent_support_hash16(rows: Sequence[Row3]) -> str:
    """Match train_hrm_text_158._retained_support hash (repr → sha256[:16])."""
    return hashlib.sha256(repr(list(rows)).encode("utf-8")).hexdigest()[:16]


def load_math_a0_rows() -> list[Row3]:
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )

    rows: list[Row3] = [
        (q, e, rung)
        for rung, pairs in build_exhaustive_supports().items()
        for (q, e) in pairs
    ]
    return sorted(rows, key=lambda r: (r[2], r[0], r[1]))


def load_identity_full_rows(*, seed: int = 17) -> list[Row3]:
    from calm.hrm_text_158.curriculum.language_supports import (
        build_l0c2k1_identity_full_support,
    )

    rows: list[Row3] = [
        (q, e, bucket)
        for _surface, pairs in build_l0c2k1_identity_full_support(seed).items()
        for (q, e, bucket) in pairs
    ]
    return sorted(rows, key=lambda r: (r[2], r[0], r[1]))


def select_acquisition_rows(
    math_rows: Sequence[Row3], *, n: int = ACQ_N, seed: int = ACQ_SHUFFLE_SEED
) -> list[Row3]:
    idx = list(range(len(math_rows)))
    random.Random(int(seed)).shuffle(idx)
    return [math_rows[i] for i in idx[: int(n)]]


def select_retention_math_rows(
    math_rows: Sequence[Row3],
    acquisition: Sequence[Row3],
    *,
    n: int = RET_MATH_N,
    seed: int = RET_SHUFFLE_SEED,
) -> list[Row3]:
    acq_set = set(acquisition)
    idx = list(range(len(math_rows)))
    random.Random(int(seed)).shuffle(idx)
    out: list[Row3] = []
    for i in idx:
        row = math_rows[i]
        if row in acq_set:
            continue
        out.append(row)
        if len(out) >= int(n):
            break
    if len(out) < int(n):
        raise RuntimeError(
            f"retention math_a0 selection undersized: got {len(out)} need {n}"
        )
    return out


def select_retention_identity_rows(
    id_rows: Sequence[Row3], *, n: int = RET_ID_N, seed: int = RET_SHUFFLE_SEED
) -> list[Row3]:
    idx = list(range(len(id_rows)))
    random.Random(int(seed)).shuffle(idx)
    return [id_rows[i] for i in idx[: int(n)]]


def build_phase1_probe_sets() -> dict[str, Any]:
    """Build acquisition + retention sets; fail-closed on prereg sha mismatch."""
    math_rows = load_math_a0_rows()
    id_rows = load_identity_full_rows(seed=17)
    math_hash = parent_support_hash16(math_rows)
    id_hash = parent_support_hash16(id_rows)
    if math_hash != MATH_A0_PARENT_SUPPORT_HASH:
        raise RuntimeError(
            f"math_a0 parent hash mismatch: {math_hash} != {MATH_A0_PARENT_SUPPORT_HASH}"
        )
    if id_hash != IDENTITY_PARENT_SUPPORT_HASH:
        raise RuntimeError(
            f"identity parent hash mismatch: {id_hash} != {IDENTITY_PARENT_SUPPORT_HASH}"
        )

    acquisition = select_acquisition_rows(math_rows)
    acq_sha = compact_rows_sha256(acquisition)
    if acq_sha != ACQUISITION_SELECTION_SHA256:
        raise RuntimeError(
            f"acquisition selection_sha256 mismatch: {acq_sha} != "
            f"{ACQUISITION_SELECTION_SHA256}"
        )

    ret_math = select_retention_math_rows(math_rows, acquisition)
    ret_id = select_retention_identity_rows(id_rows)
    id_sha = compact_rows_sha256(ret_id)
    if id_sha != IDENTITY_SELECTION_SHA256:
        raise RuntimeError(
            f"identity selection_sha256 mismatch: {id_sha} != "
            f"{IDENTITY_SELECTION_SHA256}"
        )
    if set(ret_math) & set(acquisition):
        raise RuntimeError("retention math_a0 not disjoint from acquisition")

    retention = list(ret_math) + list(ret_id)
    return {
        "acquisition": acquisition,
        "retention": retention,
        "retention_math_a0": ret_math,
        "retention_identity": ret_id,
        "acquisition_selection_sha256": acq_sha,
        "identity_selection_sha256": id_sha,
        "math_a0_parent_support_hash": math_hash,
        "identity_parent_support_hash": id_hash,
        "acquisition_n": len(acquisition),
        "retention_n": len(retention),
    }


def sample_batch_excluding_acquisition(
    pool: Sequence[Row3],
    *,
    batch: int,
    rng: random.Random,
    acquisition_set: set[Row3],
    max_draws: int = 10_000,
) -> tuple[list[Row3], int]:
    """Reject+resample any training row in the acquisition set.

    Returns (batch_rows, excluded_hit_count).
    """
    if not pool:
        raise ValueError("empty training pool")
    out: list[Row3] = []
    excluded = 0
    draws = 0
    while len(out) < int(batch):
        draws += 1
        if draws > int(max_draws):
            raise RuntimeError(
                "train-exclusion resampling exceeded max_draws "
                f"({max_draws}); pool too small or acquisition covers pool"
            )
        row = pool[rng.randrange(len(pool))]
        if row in acquisition_set:
            excluded += 1
            continue
        out.append(row)
    return out, excluded
