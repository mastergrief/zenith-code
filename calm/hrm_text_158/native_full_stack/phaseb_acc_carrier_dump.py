"""Phase B observation-only compact acc-carrier dump helper.

Reads eligible bulk acc (BULK_SUFFIXES × layers), emits compact JSON dumps.
ZERO law / transfer / hot_h mutation — observation only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.forgetting_laws import entropy_bits

BULK_SUFFIXES: tuple[str, ...] = (
    "gqkv_proj.weight",
    "o_proj.weight",
    "gate_up_proj.weight",
    "down_proj.weight",
)
PHASEB_N_ELIGIBLE_PRODUCTION = 29_360_128
PHASEB_SNAPSHOT_STEPS: tuple[int, ...] = (25, 50, 75, 100, 125, 150)
SNAPSHOTS_JSONL = "acc_carrier_snapshots.jsonl"
SUMMARY_JSON = "acc_carrier_dump_summary.json"
RUN_RECEIPT_JSON = "dump_run_receipt.json"


def select_eligible_bulk_acc(
    acc_by_name: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fail-closed filter to BULK_SUFFIXES-ending keys."""
    if not isinstance(acc_by_name, Mapping) or not acc_by_name:
        raise ValueError("phaseb dump: acc_by_name missing or empty")
    out = {
        str(k): v
        for k, v in acc_by_name.items()
        if str(k).endswith(BULK_SUFFIXES)
    }
    if not out:
        raise ValueError(
            "phaseb dump: zero tensors matched BULK_SUFFIXES "
            f"{BULK_SUFFIXES!r}"
        )
    return out


def _value_abs_histogram(flat: torch.Tensor) -> dict[str, int]:
    abs_i = flat.abs().to(torch.int64)
    vals, counts = torch.unique(abs_i, return_counts=True)
    return {str(int(v.item())): int(c.item()) for v, c in zip(vals, counts)}


def _top_h_multiset_sha256(flat: torch.Tensor) -> str:
    """SHA256 of sorted (|v|, count) multiset over NONZERO |acc| values."""
    nz = flat[flat != 0].abs().to(torch.int64)
    if int(nz.numel()) == 0:
        payload = "[]"
    else:
        vals, counts = torch.unique(nz, return_counts=True)
        pairs = sorted(
            (int(v.item()), int(c.item())) for v, c in zip(vals, counts)
        )
        payload = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def m2_empirical_bpw(*, n_nonzero: int, n_eligible: int) -> float:
    n = int(n_eligible)
    if n <= 0:
        raise ValueError("phaseb dump: n_eligible must be positive")
    return float(int(n_nonzero)) * (math.log2(float(n)) + 8.0) / float(n)


def build_compact_snapshot(
    step: int,
    acc_by_name: Mapping[str, torch.Tensor],
    *,
    expected_n_eligible: int | None = PHASEB_N_ELIGIBLE_PRODUCTION,
) -> dict[str, Any]:
    """Build one contract-schema snapshot; fail-closed on shape/key issues."""
    bulk = select_eligible_bulk_acc(acc_by_name)
    flat = torch.cat([t.detach().reshape(-1).cpu() for t in bulk.values()])
    n_eligible = int(flat.numel())
    if expected_n_eligible is not None and n_eligible != int(expected_n_eligible):
        raise ValueError(
            f"phaseb dump: n_eligible={n_eligible} != "
            f"expected_n_eligible={int(expected_n_eligible)}"
        )
    n_nonzero = int((flat != 0).sum().item())
    if n_nonzero > 8192:
        # Law break sentinel for ARM3 sparse_hot after apply — surface as D later;
        # helper still records but tags the anomaly for fail-closed consumers.
        pass
    h = float(entropy_bits(flat))
    snap = {
        "step": int(step),
        "n_eligible": int(n_eligible),
        "n_nonzero": int(n_nonzero),
        "H_bits_per_weight": h,
        "value_abs_histogram": _value_abs_histogram(flat),
        "top_h_multiset_sha256": _top_h_multiset_sha256(flat),
        "M2_empirical_bpw": m2_empirical_bpw(
            n_nonzero=n_nonzero, n_eligible=n_eligible
        ),
        "acc_side_scale_bits_bpw": 0.0,
    }
    _assert_snapshot_schema(snap)
    return snap


def _assert_snapshot_schema(snap: Mapping[str, Any]) -> None:
    required = (
        "step",
        "n_eligible",
        "n_nonzero",
        "H_bits_per_weight",
        "value_abs_histogram",
        "top_h_multiset_sha256",
        "M2_empirical_bpw",
        "acc_side_scale_bits_bpw",
    )
    for k in required:
        if k not in snap:
            raise ValueError(f"phaseb dump: snapshot missing key {k!r}")
    if not isinstance(snap["value_abs_histogram"], dict):
        raise ValueError("phaseb dump: value_abs_histogram must be dict")
    if snap["acc_side_scale_bits_bpw"] != 0.0:
        raise ValueError("phaseb dump: acc_side_scale_bits_bpw must be 0.0")
    if not isinstance(snap["top_h_multiset_sha256"], str) or len(
        snap["top_h_multiset_sha256"]
    ) != 64:
        raise ValueError("phaseb dump: top_h_multiset_sha256 must be 64-hex")


def _write_o_excl(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


class PhaseBAccCarrierDumpWriter:
    """Append-only compact dump writer for a single run directory."""

    def __init__(self, dump_dir: str | Path, *, expected_n_eligible: int | None = PHASEB_N_ELIGIBLE_PRODUCTION):
        self.dump_dir = Path(dump_dir)
        self.expected_n_eligible = expected_n_eligible
        self.snapshots: list[dict[str, Any]] = []
        self._jsonl_path = self.dump_dir / SNAPSHOTS_JSONL
        self._summary_path = self.dump_dir / SUMMARY_JSON
        self._receipt_path = self.dump_dir / RUN_RECEIPT_JSON
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        for p in (self._jsonl_path, self._summary_path, self._receipt_path):
            if p.exists():
                raise FileExistsError(
                    f"phaseb dump: refuse overwrite existing {p}"
                )

    def record_snapshot(
        self, step: int, acc_by_name: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        snap = build_compact_snapshot(
            step,
            acc_by_name,
            expected_n_eligible=self.expected_n_eligible,
        )
        self.snapshots.append(snap)
        with self._jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snap, sort_keys=True) + "\n")
        return snap

    def finalize(self, run_receipt: Mapping[str, Any]) -> dict[str, Any]:
        if not self.snapshots:
            raise ValueError("phaseb dump: finalize with zero snapshots")
        for key in (
            "geometry",
            "parent_sha256",
            "batch_rng_base",
            "arm",
        ):
            if key not in run_receipt:
                raise ValueError(f"phaseb dump: run_receipt missing {key!r}")
        summary = {
            "schema": "compact_measurement_dumps_v1",
            "n_snapshots": len(self.snapshots),
            "snapshot_steps": [int(s["step"]) for s in self.snapshots],
            "final": dict(self.snapshots[-1]),
            "trajectory_H": [
                {"step": int(s["step"]), "H_bits_per_weight": float(s["H_bits_per_weight"])}
                for s in self.snapshots
            ],
            "trajectory_n_nonzero": [
                {"step": int(s["step"]), "n_nonzero": int(s["n_nonzero"])}
                for s in self.snapshots
            ],
        }
        receipt = {
            **dict(run_receipt),
            "schema": "phaseb_dump_run_receipt_v1",
            "dump_dir": str(self.dump_dir),
            "files": {
                "snapshots_jsonl": SNAPSHOTS_JSONL,
                "final_summary_json": SUMMARY_JSON,
                "run_receipt_json": RUN_RECEIPT_JSON,
            },
            "n_snapshots": len(self.snapshots),
            "BULK_SUFFIXES": list(BULK_SUFFIXES),
            "expected_n_eligible": self.expected_n_eligible,
        }
        _write_o_excl(self._summary_path, json.dumps(summary, indent=2) + "\n")
        _write_o_excl(self._receipt_path, json.dumps(receipt, indent=2) + "\n")
        return {"summary": summary, "receipt": receipt}
