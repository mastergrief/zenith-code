"""Correctness and speedup tests for HullKVCache."""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

import torch

from calm.llm_computer.hull_cache import HullKVCache


def _linear_argmax(points: List[Tuple[float, float, torch.Tensor]],
                   q: Tuple[float, float]) -> torch.Tensor:
    """Baseline: linear scan argmax."""
    best = None
    best_score = float("-inf")
    for (kx, ky, v) in points:
        s = q[0] * kx + q[1] * ky
        if s > best_score:
            best_score = s
            best = v
    assert best is not None
    return best


def test_parabolic_keys_exact_lookup():
    """Per the paper's 2D-head construction, k_j = (2j, -j²) queried with
    q = (i, 1) selects exactly position i. Cache should agree."""
    cache = HullKVCache()
    N = 100
    points = []
    for j in range(N):
        k = (float(2 * j), float(-(j * j)))
        v = torch.tensor([float(j)])
        cache.insert(k, v)
        points.append((k[0], k[1], v))

    for i in range(N):
        q = (float(i), 1.0)
        got = cache.query(q).item()
        assert got == float(i), f"parabolic lookup at i={i}: got {got}"


def test_random_queries_match_linear_scan():
    """On random 2D keys + random query directions, hull cache must match
    linear-scan argmax exactly (integer keys → no float tolerance needed)."""
    random.seed(42)
    cache = HullKVCache()
    points = []
    for i in range(1000):
        kx, ky = random.randint(-100, 100), random.randint(-100, 100)
        v = torch.tensor([float(i)])
        cache.insert((float(kx), float(ky)), v)
        points.append((float(kx), float(ky), v))

    for _ in range(500):
        theta = random.uniform(0, 2 * math.pi)
        q = (math.cos(theta), math.sin(theta))
        hull_v = cache.query(q).item()
        linear_v = _linear_argmax(points, q).item()
        # Compare SCORES, not values: ties on score are allowed to pick
        # different indices.
        hull_score = q[0] * (0.0) + q[1] * (0.0)  # placeholder; real below
        # Rebuild: find the score for hull_v's point
        def _score_of(v_id: float) -> float:
            for (kx, ky, v) in points:
                if v.item() == v_id:
                    return q[0] * kx + q[1] * ky
            return float("-inf")
        hull_score = _score_of(hull_v)
        linear_score = _score_of(linear_v)
        assert abs(hull_score - linear_score) < 1e-9, \
            f"score mismatch: hull={hull_score} linear={linear_score} (q={q})"


def test_speedup_vs_linear():
    """On t=2000 2D points with 1000 queries, HullKVCache should be
    noticeably faster than linear scan. Gate: ≥ 3× (conservative; this
    prototype uses a linear walk over hull vertices, not a proper log-n
    search — so the speedup comes from hull size << total points)."""
    random.seed(123)
    N = 2000
    Q = 1000
    cache = HullKVCache()
    points = []
    for i in range(N):
        kx, ky = random.randint(-1000, 1000), random.randint(-1000, 1000)
        v = torch.tensor([float(i)])
        cache.insert((float(kx), float(ky)), v)
        points.append((float(kx), float(ky), v))

    queries = [
        (math.cos(theta), math.sin(theta))
        for theta in [random.uniform(0, 2 * math.pi) for _ in range(Q)]
    ]

    # Warm up JIT / attribute lookup.
    for q in queries[:5]:
        cache.query(q)
        _linear_argmax(points, q)

    t0 = time.perf_counter()
    for q in queries:
        cache.query(q)
    t_hull = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        _linear_argmax(points, q)
    t_linear = time.perf_counter() - t0

    hull_sz = cache.hull_size()
    speedup = t_linear / t_hull
    print(
        f"\n[hull_cache] N={N} Q={Q} hull sizes={hull_sz} "
        f"hull={t_hull*1000:.1f}ms linear={t_linear*1000:.1f}ms "
        f"speedup={speedup:.1f}x"
    )
    assert speedup >= 3.0, f"hull cache speedup {speedup:.1f}x < 3x gate"


if __name__ == "__main__":
    test_parabolic_keys_exact_lookup()
    print("[ok] parabolic_keys_exact_lookup")
    test_random_queries_match_linear_scan()
    print("[ok] random_queries_match_linear_scan")
    test_speedup_vs_linear()
    print("[ok] speedup_vs_linear")
