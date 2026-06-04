"""Thin JSON/in-memory harness for Front-C CPU/static reducers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.front_c_projection import (
    FrontCProjectionReport,
    build_front_c_projection_report,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    Base3QEntropyLedgerRow,
    base3_q_entropy_ledger_for_shapes,
)


def base3_q_ledger_from_front_c_artifact(payload: Mapping[str, Any]) -> Base3QEntropyLedgerRow:
    """Build a gate-valid q ledger from explicit physical base3 shape metadata."""

    q_payload = payload.get("q_ledger")
    if not isinstance(q_payload, Mapping):
        raise ValueError("Front-C artifact must include q_ledger shape metadata")
    logical_shapes = q_payload.get("logical_shapes")
    scale_count = q_payload.get("scale_count")
    if logical_shapes is None or scale_count is None:
        raise ValueError("q_ledger must include logical_shapes and scale_count")
    return base3_q_entropy_ledger_for_shapes(
        regime_name=str(q_payload.get("regime_name", "front_c_artifact_base3_q")),
        logical_shapes=logical_shapes,
        scale_count=int(scale_count),
        accumulator_bits_per_weight=float(q_payload.get("accumulator_bits_per_weight", 0.0)),
    )


def _reject_bounded_nonclaim_artifact(payload: Mapping[str, Any]) -> None:
    derivation = payload.get("decision_path_derivation")
    if not isinstance(derivation, Mapping):
        return
    scope = str(derivation.get("identity_emission_scope", ""))
    full_identity = derivation.get("full_identity_emission_claimed", None)
    full_sparse = derivation.get("full_sparse_equivalence_claimed", None)
    if (
        scope.startswith("bounded_")
        or full_identity is False
        or full_sparse is False
    ):
        raise ValueError(
            "bounded/non-claim Front-C identity artifacts cannot build a claimable projection report"
        )


def front_c_report_from_mapping(
    payload: Mapping[str, Any],
    *,
    q_ledger_row: Base3QEntropyLedgerRow | None = None,
) -> FrontCProjectionReport:
    """Normalize a future B2 audit artifact and compute the compact Front-C report."""

    _reject_bounded_nonclaim_artifact(payload)
    timeline = payload.get("timeline")
    if not isinstance(timeline, Sequence) or isinstance(timeline, (str, bytes)):
        raise ValueError("Front-C artifact must include a timeline sequence")
    dense_path = payload.get("dense_decision_path")
    sparse_path = payload.get("sparse_decision_path")
    if not isinstance(dense_path, Mapping) or not isinstance(sparse_path, Mapping):
        raise ValueError("Front-C artifact must include dense_decision_path and sparse_decision_path")
    q_row = q_ledger_row if q_ledger_row is not None else base3_q_ledger_from_front_c_artifact(payload)
    return build_front_c_projection_report(
        timeline_steps=timeline,
        q_ledger_row=q_row,
        dense_decision_path=dense_path,
        sparse_decision_path=sparse_path,
        value_bits_per_row=int(payload.get("value_bits_per_row", 16)),
        flag_bits_per_row=int(payload.get("flag_bits_per_row", 2)),
        tensor_metadata_bits=int(payload.get("tensor_metadata_bits", 0)),
        bucket_metadata_bits=int(payload.get("bucket_metadata_bits", 0)),
        scale_metadata_bits=int(payload.get("scale_metadata_bits", 0)),
        guardrail_metadata_bits=int(payload.get("guardrail_metadata_bits", 0)),
        event_delta_count=int(payload.get("event_delta_count", 0)),
    )


def front_c_report_from_json(
    path: str | Path,
    *,
    q_ledger_row: Base3QEntropyLedgerRow | None = None,
) -> FrontCProjectionReport:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError("Front-C JSON artifact must contain an object")
    return front_c_report_from_mapping(payload, q_ledger_row=q_ledger_row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute a CPU/static Front-C projection report")
    parser.add_argument("artifact", help="JSON Front-C timeline artifact")
    args = parser.parse_args(argv)
    report = front_c_report_from_json(args.artifact)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
