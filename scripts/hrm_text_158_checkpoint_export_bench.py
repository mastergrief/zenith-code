#!/usr/bin/env python3
"""CPU-only checkpoint export bench for healthy-baseline RSS/wall measurement."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (  # noqa: E402
    build_authoritative_checkpoint_payload,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    S1_PROJECTION_LAW,
    S1_RANK_BUCKET_VOTE_LAW,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec  # noqa: E402


def _rss_bytes_self() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _default_updater_config() -> dict[str, Any]:
    return {
        "rank_vote_spec": default_dry_run_rank_vote_spec().to_live_dict(),
        "vote_update_spec": asdict(
            VoteUpdateSpec(
                threshold_abs=1,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=4096,
            )
        ),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }


def _build_synthetic_all_bitlinear_states() -> dict[str, Any]:
    states: dict[str, Any] = {}
    for module_index in range(32):
        numel = 2048 * 512
        q = torch.zeros(numel, dtype=torch.int8)
        acc = torch.zeros(numel, dtype=torch.int16)
        hot = tuple(range(0, numel, max(1, numel // 64)))
        states[f"module_{module_index:02d}"] = make_bounded_tensor_state(
            f"module_{module_index:02d}",
            q,
            0.25,
            acc,
            hot_exact_indices=hot,
            cold_default_value=0,
        )
    return states


def _build_synthetic_cold_exception_stress_states() -> tuple[dict[str, Any], int]:
    numel = 2048 * 512
    q = torch.zeros(numel, dtype=torch.int8)
    acc = torch.full((numel,), 7, dtype=torch.int16)
    cold_stride = max(1, numel // 50000)
    for idx in range(0, numel, cold_stride):
        acc[idx] = int((idx % 5) - 2)
    hot = tuple(range(0, min(128, numel)))
    state = make_bounded_tensor_state(
        "cold_exception_stress",
        q,
        0.5,
        acc,
        hot_exact_indices=hot,
        cold_default_value=7,
    )
    cold_exception_count = len(
        [
            idx
            for idx, value in enumerate(acc.flatten().tolist())
            if idx not in set(hot) and int(value) != 7
        ]
    )
    return {"cold_exception_stress": state}, cold_exception_count


def run_export_bench(
    tensor_states: Mapping[str, Any],
    *,
    states_source: str,
    cold_exception_count: int | None = None,
) -> dict[str, Any]:
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("checkpoint export bench requires CPU-only execution")

    updater_config = _default_updater_config()
    start = time.perf_counter()
    peak_rss = _rss_bytes_self()
    payload = build_authoritative_checkpoint_payload(
        tensor_states,
        step=10,
        updater_config=updater_config,
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
    )
    wall_seconds = time.perf_counter() - start
    peak_rss = max(peak_rss, _rss_bytes_self())

    summaries = payload["tensor_summaries"]
    hot_path_sites_exercised = ["A", "B", "C", "D"]
    for summary in summaries.values():
        if summary.get("bounded_decode_parity_checked") is not True:
            raise RuntimeError("bench must exercise parity/decode path (site A/C)")
        if not summary.get("q_sha256"):
            raise RuntimeError("bench must exercise tensor_sha256 (site B)")
    if not payload.get("authoritative_state_sha256"):
        raise RuntimeError("bench must exercise _canonical_json summaries (site D)")

    receipt: dict[str, Any] = {
        "schema": "hrm_text_158_checkpoint_export_bench_receipt/v1",
        "states_source": states_source,
        "tensor_count": len(tensor_states),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss,
        "host": os.uname().nodename,
        "hot_path_sites_exercised": hot_path_sites_exercised,
        "dry_run": True,
    }
    if cold_exception_count is not None:
        receipt["cold_exception_count"] = int(cold_exception_count)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-all-bitlinear", action="store_true")
    group.add_argument("--synthetic-cold-exception-stress", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.synthetic_all_bitlinear:
        states = _build_synthetic_all_bitlinear_states()
        receipt = run_export_bench(states, states_source="synthetic_all_bitlinear")
    else:
        states, cold_exception_count = _build_synthetic_cold_exception_stress_states()
        receipt = run_export_bench(
            states,
            states_source="synthetic_cold_exception_stress",
            cold_exception_count=cold_exception_count,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
