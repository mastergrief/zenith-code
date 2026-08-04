#!/usr/bin/env python3
"""Exact replayable GPU one-step smoke: event-coded live no-cap (+ with-cap).

Path proof for defect-cycle NameError at
bounded_delta_learner._apply_bounded_delta_vote_step_event_coded_live no-cap
branch (global_cap_spec is None). Loop-entry on cuda:0. Fail-closed.

Replay (normal — expect rc0 + SMOKE_OK):
  cd /mnt/c/Users/gabes/projects/claw-code-hrm-text-158
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
    python3 -u -B scripts/smoke_event_coded_no_cap_gpu_onestep_v0.py

Known-bad calibration (expect rc!=0 + SMOKE_FAIL):
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
    python3 -u -B scripts/smoke_event_coded_no_cap_gpu_onestep_v0.py \\
      --calibrate-known-bad
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )


def _tiny(device: torch.device):
    q = torch.zeros((4, 4), dtype=torch.int8, device=device)
    st = make_event_coded_live_tensor_state("toy.proj", q, 0.25, demotion_band=1)
    return {"toy.proj": st}


def _votes(device: torch.device, idx: int = 0, mag: int = 12) -> torch.Tensor:
    v = torch.zeros(16, dtype=torch.int16, device=device)
    v[int(idx)] = int(mag)
    return v.view(4, 4)


def _row(result, *, branch: str) -> dict:
    return {
        "branch": branch,
        "global_rate_cap_enabled": result.global_summary.get("global_rate_cap_enabled"),
        "event_coded_live_carrier_enabled": result.global_summary.get(
            "event_coded_live_carrier_enabled"
        ),
        "live_authority": result.tensor_stats["toy.proj"].get("live_authority"),
        "q_device": str(result.tensor_states["toy.proj"].q_levels.device),
    }


def _check(row: dict, *, expect_cap: bool) -> str | None:
    """Return failure reason or None if row is good."""
    if row["global_rate_cap_enabled"] is not expect_cap:
        return (
            f"{row['branch']}:global_rate_cap_enabled="
            f"{row['global_rate_cap_enabled']!r} expected {expect_cap}"
        )
    if row["event_coded_live_carrier_enabled"] is not True:
        return (
            f"{row['branch']}:event_coded_live_carrier_enabled="
            f"{row['event_coded_live_carrier_enabled']!r} expected True"
        )
    if row["live_authority"] != "event_coded_live_carrier":
        return (
            f"{row['branch']}:live_authority={row['live_authority']!r} "
            "expected event_coded_live_carrier"
        )
    # carrier-design contract: q authority stays CPU after event-coded apply
    if row["q_device"] != "cpu":
        return f"{row['branch']}:q_device={row['q_device']!r} expected cpu"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--calibrate-known-bad",
        action="store_true",
        help="Invert one expected assertion so the fail-closed path is observed",
    )
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("SMOKE_FAIL cuda_unavailable", file=sys.stderr, flush=True)
        return 2
    device = torch.device("cuda:0")
    print(
        json.dumps(
            {"device": str(device), "cuda_name": torch.cuda.get_device_name(0)},
            sort_keys=True,
        ),
        flush=True,
    )

    r_no = apply_bounded_delta_vote_step(
        _tiny(device),
        {"toy.proj": _votes(device)},
        {"toy.proj": _vote_spec()},
        local_selection_ordering_step=0,
    )
    row_no = _row(r_no, branch="no_cap")
    print(json.dumps(row_no, sort_keys=True), flush=True)

    r_cap = apply_bounded_delta_vote_step(
        _tiny(device),
        {"toy.proj": _votes(device)},
        {"toy.proj": _vote_spec()},
        global_cap_spec=GlobalRateCapSpec(cap=1, step=1, mutate_outputs=True),
        local_selection_ordering_step=0,
    )
    row_cap = _row(r_cap, branch="with_cap")
    print(json.dumps(row_cap, sort_keys=True), flush=True)

    if args.calibrate_known_bad:
        # Known-bad: invert no_cap global_rate_cap expectation so assert fires.
        reason = _check(row_no, expect_cap=True)
        if reason is None:
            # Real no_cap has cap=False; inverted expect_cap=True must fail.
            reason = "calibrate_known_bad_did_not_fire"
        print(f"SMOKE_FAIL {reason}", file=sys.stderr, flush=True)
        return 3

    for row, expect_cap in ((row_no, False), (row_cap, True)):
        reason = _check(row, expect_cap=expect_cap)
        if reason is not None:
            print(f"SMOKE_FAIL {reason}", file=sys.stderr, flush=True)
            return 1

    print("SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
