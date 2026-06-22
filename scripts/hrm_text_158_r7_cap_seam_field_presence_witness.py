#!/usr/bin/env python3
"""CPU field-presence witness for R7 cap/defer instrumentation (no trainer loop)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    field_presence_from_step_summary,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )


def _tensor_input(
    state_key: str,
    q: list[int],
    acc: list[int],
    votes: list[int],
) -> GlobalRateCapTensorInput:
    state = VoteUpdateState(
        q_levels=torch.tensor(q, dtype=torch.int8),
        accumulators=torch.tensor(acc, dtype=torch.int16),
    )
    inputs = VoteUpdateInputs(votes=torch.tensor(votes, dtype=torch.int16))
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan,
        vote_inputs=inputs,
    )


def build_field_presence_witness() -> dict[str, object]:
    item = _tensor_input(
        "synthetic.cap",
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [30, 30, 30, 0],
    )
    result = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets={"synthetic.cap": 0},
    )
    witness = field_presence_from_step_summary(result.step_summary)
    witness["cpu_no_model_forward"] = True
    witness["cpu_no_trainer_loop"] = True
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit the R7 cap-seam field-presence witness JSON.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    witness = build_field_presence_witness()
    payload = json.dumps(witness, indent=args.indent, sort_keys=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(payload + "\n", encoding="utf-8")
    if not bool(witness.get("field_presence_pass")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
