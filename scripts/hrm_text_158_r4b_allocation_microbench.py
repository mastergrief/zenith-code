#!/usr/bin/env python3
"""CPU allocation microbench for r4b sparse/apply/proof materialization."""
from __future__ import annotations

import argparse
import hashlib
import random
import resource
import time
import tracemalloc
from typing import Any, Callable

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    _execute_direct_bounded_local_vote_update_reference_3936d74,
    encode_budget_capped_hybrid_reference,
    execute_direct_bounded_local_vote_update_candidate,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import _sparse_vote_events
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec, VoteUpdateState

R4A_SPARSE_MB = 766.0
R4A_APPLY_MB = 355.0


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
        fraction_per_tensor=1.0,
    )


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss) / 1024.0


def _measure_phase(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    tracemalloc.start()
    rss_before = _rss_mb()
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_mb()
    return {
        "phase": label,
        "wall_ms": round(elapsed_ms, 3),
        "tracemalloc_peak_bytes": int(peak),
        "rss_before_mb": round(rss_before, 3),
        "rss_after_mb": round(rss_after, 3),
        "rss_delta_mb": round(rss_after - rss_before, 3),
        "result": result,
    }


def _proof_materialization(result) -> str:
    proof = dict(result.proof)
    return hashlib.sha256(repr(sorted(proof.items())).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=29_600_000)
    parser.add_argument("--sparse-count", type=int, default=400)
    parser.add_argument("--keys", type=int, default=32)
    args = parser.parse_args()

    rng = random.Random(17)
    numel = int(args.numel)
    keys = int(args.keys)
    hot = tuple(rng.sample(range(numel), min(32, numel)))
    sparse_indices = rng.sample(range(numel), min(int(args.sparse_count), numel))
    votes = torch.zeros(numel, dtype=torch.int16)
    for index in sparse_indices:
        votes[index] = int(rng.randint(-6, 6) or 1)

    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.zeros(numel, dtype=torch.int16)
    for index in hot:
        acc[index] = 12
    state = VoteUpdateState(q_levels=q, accumulators=acc)
    bounded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=hot,
        cold_default_value=0,
    )
    sparse_dict = {
        int(i): int(votes[int(i)].item())
        for i in sparse_indices
        if int(votes[int(i)].item()) != 0
    }
    sparse_carrier = SparseVoteEvents.from_dict(sparse_dict)
    apply_kwargs = dict(
        state_key="r4b.microbench",
        q_levels=q,
        bounded_accumulator=bounded,
        vote_spec=_spec(),
    )

    def sparse_reference() -> dict[int, int]:
        flat = votes.detach().cpu().to(torch.int16).flatten()
        return {
            int(index): int(flat[int(index)].item())
            for index in torch.nonzero(flat != 0, as_tuple=False).flatten().tolist()
        }

    sparse_ref = _measure_phase("SPARSE_REFERENCE", sparse_reference)
    sparse_new = _measure_phase("SPARSE_CANDIDATE", lambda: _sparse_vote_events(votes))

    apply_ref = _measure_phase(
        "APPLY_REFERENCE",
        lambda: _execute_direct_bounded_local_vote_update_reference_3936d74(
            sparse_vote_events=sparse_dict,
            **apply_kwargs,
        ),
    )
    apply_new = _measure_phase(
        "APPLY_CANDIDATE",
        lambda: execute_direct_bounded_local_vote_update_candidate(
            sparse_vote_events=sparse_carrier,
            **apply_kwargs,
        ),
    )
    proof_ref = _measure_phase(
        "PROOF_REFERENCE",
        lambda: _proof_materialization(apply_ref["result"]),
    )
    proof_new = _measure_phase(
        "PROOF_CANDIDATE",
        lambda: _proof_materialization(apply_new["result"]),
    )

    def reduction_fraction(ref_bytes: int, new_bytes: int) -> float:
        if ref_bytes <= 0:
            return 0.0
        return round(1.0 - (float(new_bytes) / float(ref_bytes)), 6)

    def time_ratio(ref_ms: float, new_ms: float) -> float:
        if ref_ms <= 0:
            return 0.0
        return round(float(new_ms) / float(ref_ms), 6)

    sparse_ref_peak = int(sparse_ref["tracemalloc_peak_bytes"])
    sparse_new_peak = int(sparse_new["tracemalloc_peak_bytes"])
    apply_ref_peak = int(apply_ref["tracemalloc_peak_bytes"])
    apply_new_peak = int(apply_new["tracemalloc_peak_bytes"])
    proof_ref_peak = int(proof_ref["tracemalloc_peak_bytes"])
    proof_new_peak = int(proof_new["tracemalloc_peak_bytes"])

    sparse_ref_total_mb = (sparse_ref_peak * keys) / (1024.0 * 1024.0)
    sparse_new_total_mb = (sparse_new_peak * keys) / (1024.0 * 1024.0)
    apply_ref_total_mb = (apply_ref_peak * keys) / (1024.0 * 1024.0)
    apply_new_total_mb = (apply_new_peak * keys) / (1024.0 * 1024.0)

    report = {
        "numel": numel,
        "keys": keys,
        "r4a_baseline_sparse_mb": R4A_SPARSE_MB,
        "r4a_baseline_apply_mb": R4A_APPLY_MB,
        "sparse_reference": {k: sparse_ref[k] for k in sparse_ref if k != "result"},
        "sparse_candidate": {k: sparse_new[k] for k in sparse_new if k != "result"},
        "sparse_peak_byte_reduction_fraction": reduction_fraction(sparse_ref_peak, sparse_new_peak),
        "sparse_total_estimated_mb_reference": round(sparse_ref_total_mb, 3),
        "sparse_total_estimated_mb_candidate": round(sparse_new_total_mb, 3),
        "sparse_vs_r4a_estimated_reduction_fraction": round(
            1.0 - (sparse_new_total_mb / R4A_SPARSE_MB), 6
        ),
        "sparse_wall_ms_ratio_candidate_over_reference": time_ratio(
            float(sparse_ref["wall_ms"]), float(sparse_new["wall_ms"])
        ),
        "apply_reference": {k: apply_ref[k] for k in apply_ref if k != "result"},
        "apply_candidate": {k: apply_new[k] for k in apply_new if k != "result"},
        "apply_peak_byte_reduction_fraction": reduction_fraction(apply_ref_peak, apply_new_peak),
        "apply_total_estimated_mb_reference": round(apply_ref_total_mb, 3),
        "apply_total_estimated_mb_candidate": round(apply_new_total_mb, 3),
        "apply_vs_r4a_estimated_reduction_fraction": round(
            1.0 - (apply_new_total_mb / R4A_APPLY_MB), 6
        ),
        "apply_wall_ms_ratio_candidate_over_reference": time_ratio(
            float(apply_ref["wall_ms"]), float(apply_new["wall_ms"])
        ),
        "proof_reference": {k: proof_ref[k] for k in proof_ref if k != "result"},
        "proof_candidate": {k: proof_new[k] for k in proof_new if k != "result"},
        "proof_peak_byte_reduction_fraction": reduction_fraction(proof_ref_peak, proof_new_peak),
        "proof_wall_ms_ratio_candidate_over_reference": time_ratio(
            float(proof_ref["wall_ms"]), float(proof_new["wall_ms"])
        ),
    }
    print(report)
    sparse_pass = report["sparse_peak_byte_reduction_fraction"] >= 0.80
    return 0 if sparse_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
