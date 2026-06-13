"""Compact pressure-shape agreement across paired probe receipts."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    PRIMARY_STEP_MAX,
    PRIMARY_STEP_MIN,
)

SCHEMA_VERSION = "hrm_text_158_pressure_shape_agreement/v0"
PRESSURE_SHAPE_SCHEMA = "hrm_text_158_pressure_shape_summary/v0"
PRESSURE_SHAPE_SCHEMA_V1 = "hrm_text_158_pressure_shape_summary/v1"
ACCEPTED_PRESSURE_SHAPE_SCHEMAS = frozenset(
    {PRESSURE_SHAPE_SCHEMA, PRESSURE_SHAPE_SCHEMA_V1},
)

BRANCH4_MEDIAN_THRESHOLD = 0.80
BRANCH4_P10_THRESHOLD = 0.60
BRANCH4_MIN_COMPARABLE_MODULES = 8


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cosine vectors must share length")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left_norm == right_norm else 0.0
    return dot / (left_norm * right_norm)


def _step_in_primary_window(step_key: str) -> bool:
    step = int(step_key)
    return PRIMARY_STEP_MIN <= step <= PRIMARY_STEP_MAX


def _extract_shape_vector(entry: Mapping[str, Any]) -> list[float] | None:
    summary = entry.get("pressure_shape_summary")
    if not isinstance(summary, Mapping):
        return None
    if summary.get("schema") not in ACCEPTED_PRESSURE_SHAPE_SCHEMAS:
        return None
    fractions = summary.get("bin_mass_fraction")
    if not isinstance(fractions, list) or not fractions:
        return None
    if summary.get("raw_per_proposal_arrays_included") is not False:
        return None
    return [float(value) for value in fractions]


def extract_module_pressure_vectors(
    receipt: Mapping[str, Any],
    *,
    step_min: int = PRIMARY_STEP_MIN,
    step_max: int = PRIMARY_STEP_MAX,
) -> dict[str, list[float]]:
    """Mean rank-bin mass fractions per module across the primary step window."""

    accum: dict[str, list[list[float]]] = {}
    for step_key, step_entry in receipt.get("step_reports", {}).items():
        step = int(step_key)
        if step < step_min or step > step_max:
            continue
        vote_pressure = step_entry.get("vote_pressure")
        if not isinstance(vote_pressure, Mapping):
            continue
        for state_key, pressure_entry in vote_pressure.items():
            if not isinstance(pressure_entry, Mapping):
                continue
            vector = _extract_shape_vector(pressure_entry)
            if vector is None:
                continue
            accum.setdefault(str(state_key), []).append(vector)
    out: dict[str, list[float]] = {}
    for state_key, vectors in accum.items():
        if not vectors:
            continue
        width = len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            continue
        out[state_key] = [
            float(sum(vector[index] for vector in vectors) / len(vectors))
            for index in range(width)
        ]
    return out


def compare_module_vectors(
    left_vectors: Mapping[str, Sequence[float]],
    right_vectors: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    per_module: dict[str, float] = {}
    for state_key in sorted(set(left_vectors) & set(right_vectors)):
        left = left_vectors[state_key]
        right = right_vectors[state_key]
        if len(left) != len(right):
            continue
        per_module[state_key] = cosine_similarity(left, right)
    cosines = list(per_module.values())
    if not cosines:
        return {
            "median_module_cosine": None,
            "p10_module_cosine": None,
            "n_comparable_modules": 0,
            "per_module_cosine": per_module,
            "computable": False,
        }
    ordered = sorted(cosines)
    p10_index = max(0, int(math.ceil(0.10 * len(ordered))) - 1)
    return {
        "median_module_cosine": float(statistics.median(ordered)),
        "p10_module_cosine": float(ordered[p10_index]),
        "n_comparable_modules": len(ordered),
        "per_module_cosine": per_module,
        "computable": True,
    }


def branch4_pressure_agreement_established(summary: Mapping[str, Any]) -> bool:
    if not bool(summary.get("computable")):
        return False
    median = summary.get("median_module_cosine")
    p10 = summary.get("p10_module_cosine")
    count = int(summary.get("n_comparable_modules") or 0)
    if median is None or p10 is None:
        return False
    return (
        float(median) >= BRANCH4_MEDIAN_THRESHOLD
        and float(p10) >= BRANCH4_P10_THRESHOLD
        and count >= BRANCH4_MIN_COMPARABLE_MODULES
    )


def build_pressure_shape_agreement(
    *,
    left_receipt: Mapping[str, Any],
    right_receipt: Mapping[str, Any],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_vectors = extract_module_pressure_vectors(left_receipt)
    right_vectors = extract_module_pressure_vectors(right_receipt)
    comparison = compare_module_vectors(left_vectors, right_vectors)
    return {
        "schema": SCHEMA_VERSION,
        "left_label": left_label,
        "right_label": right_label,
        "step_window": {
            "min": PRIMARY_STEP_MIN,
            "max": PRIMARY_STEP_MAX,
        },
        "thresholds": {
            "branch4_median_module_cosine": BRANCH4_MEDIAN_THRESHOLD,
            "branch4_p10_module_cosine": BRANCH4_P10_THRESHOLD,
            "branch4_min_comparable_modules": BRANCH4_MIN_COMPARABLE_MODULES,
        },
        "branch4_pressure_agreement_established": branch4_pressure_agreement_established(
            comparison,
        ),
        **comparison,
    }


def receipt_has_pressure_shape_summary(
    receipt: Mapping[str, Any],
    *,
    step_min: int = PRIMARY_STEP_MIN,
    step_max: int = PRIMARY_STEP_MAX,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    step_reports = receipt.get("step_reports")
    if not isinstance(step_reports, Mapping) or not step_reports:
        return False, ["missing_step_reports"]
    for step_key, step_entry in step_reports.items():
        if not _step_in_primary_window(step_key):
            continue
        if int(step_key) < step_min or int(step_key) > step_max:
            continue
        vote_pressure = step_entry.get("vote_pressure")
        if not isinstance(vote_pressure, Mapping) or not vote_pressure:
            issues.append(f"step_{step_key}:missing_vote_pressure")
            continue
        for state_key, pressure_entry in vote_pressure.items():
            if not isinstance(pressure_entry, Mapping):
                issues.append(f"step_{step_key}:{state_key}:invalid_vote_pressure_entry")
                continue
            vector = _extract_shape_vector(pressure_entry)
            if vector is None:
                issues.append(f"step_{step_key}:{state_key}:missing_pressure_shape_summary")
    return len(issues) == 0, issues


def verify_pressure_shape_summary_preflight(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    ok, issues = receipt_has_pressure_shape_summary(receipt)
    payload: dict[str, Any] = {
        "schema": "hrm_text_158_pressure_shape_summary_preflight/v0",
        "pass": ok,
        "failure_branch": None if ok else "missing_pressure_shape_summary",
        "issues": issues,
        "step_window": {
            "min": PRIMARY_STEP_MIN,
            "max": PRIMARY_STEP_MAX,
        },
        "required_fields": [
            "vote_pressure[*].pressure_shape_summary.schema (v0|v1)",
            "vote_pressure[*].pressure_shape_summary.bin_mass_fraction",
            "vote_pressure[*].pressure_shape_summary.raw_per_proposal_arrays_included=false",
        ],
        "accepted_pressure_shape_schemas": sorted(ACCEPTED_PRESSURE_SHAPE_SCHEMAS),
    }
    if receipt_path is not None:
        payload["receipt_path"] = str(receipt_path)
        payload["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return payload


def load_receipt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
