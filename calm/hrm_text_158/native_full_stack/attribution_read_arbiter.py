"""Pure attribution-read reducer. No launch, GPU, or filesystem glue.

Branch space, arbiter property, and first-match order are frozen route data
(ADVISOR_ROUTE 1786830304690-97c77ac5 + rides through 1786869653249-6569b931).
This module executes them. It does not re-derive them.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
import hashlib
from typing import Any, Callable, Iterable, Mapping

CONSUMED_INPUTS_INVENTORY_KEY = "consumed_inputs_inventory_by_key"

APPLIED = "APPLIED"
VETO_RESIDUAL = "VETO_RESIDUAL"
INVALID = "INVALID"

BRANCH_STOP = "STOP"
BRANCH_D = "D"
BRANCH_C = "C"
BRANCH_F = "F"
BRANCH_B = "B"
BRANCH_E = "E"
BRANCH_RESIDUAL_STOP = "residual_STOP"

REPRODUCED_LISTED_CARDINALITY = 949
REPRODUCED_APPLY_VETO_COUNT = 0

FIRST_MATCH_ORDER = (
    BRANCH_STOP,
    BRANCH_D,
    BRANCH_C,
    BRANCH_F,
    BRANCH_B,
    BRANCH_E,
    BRANCH_RESIDUAL_STOP,
)


def row_type(*, q_changed: bool, acc_clamped: bool) -> str:
    if q_changed and acc_clamped:
        return APPLIED
    if (not q_changed) and acc_clamped:
        return VETO_RESIDUAL
    return INVALID


def acc_clamped(
    *,
    acc_after: int,
    acc_pre_writeback: int,
    direction: int,
    threshold: int,
) -> bool:
    residual = int(acc_pre_writeback) - int(direction) * int(threshold)
    low = -int(threshold) + 1
    high = int(threshold) - 1
    clamped = min(max(residual, low), high)
    return int(acc_after) == clamped


def is_untouched(
    *,
    q_before: int,
    q_after: int,
    acc_pre_writeback: int,
    acc_after: int,
) -> bool:
    return int(q_after) == int(q_before) and int(acc_after) == int(acc_pre_writeback)


def sha256_of_ints(values: Iterable[int]) -> str:
    payload = ",".join(str(int(v)) for v in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def control_emission(
    *,
    control_indices: Iterable[int],
    untouched_by_index: Mapping[int, bool],
    raw_by_index: Mapping[int, Mapping[str, Any]] | None = None,
    hash_series: Mapping[str, Iterable[int]] | None = None,
    population_size: int | None = None,
    untouched_count: int | None = None,
    precomputed_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Examine the whole control population. Emit compact fields only.

    per_row_raw_emitted counts emitted row FIELDS (first_failing and
    extremal). Two fields carrying the same row is still two emissions.
    """
    indices = [int(i) for i in control_indices]
    failing = [i for i in indices if not bool(untouched_by_index[i])]
    first_failing = None
    extremal = None
    if failing and raw_by_index is not None:
        first_failing = {"index": failing[0], **dict(raw_by_index[failing[0]])}
        chosen = max(
            failing,
            key=lambda i: abs(int(raw_by_index[i].get("acc_after", 0)))
            + abs(int(raw_by_index[i].get("q_after", 0))),
        )
        extremal = {"index": chosen, **dict(raw_by_index[chosen])}
    hashes = {
        "untouched_mask_sha256": sha256_of_ints(
            int(bool(untouched_by_index[i])) for i in indices
        )
    }
    if hash_series is not None:
        for name, series in hash_series.items():
            hashes[name] = sha256_of_ints(series)
    if precomputed_hashes is not None:
        hashes.update({str(k): str(v) for k, v in precomputed_hashes.items()})
    size = int(population_size) if population_size is not None else len(indices)
    untouched_n = (
        int(untouched_count) if untouched_count is not None else size - len(failing)
    )
    return {
        "control_population_size": size,
        "control_examined_count": size,
        "control_untouched_count": untouched_n,
        "control_failing_count": len(failing),
        "control_failing_indices": failing,
        "control_first_failing_row": first_failing,
        "control_extremal_failing_row": extremal,
        "control_hashes": hashes,
        "per_row_raw_emitted": sum(1 for row in (first_failing, extremal) if row is not None),
    }


def apply_side_count_correct(
    *,
    r_veto: set[int],
    r_invalid: set[int],
    r_applied: set[int],
    examined: set[int],
) -> bool:
    return (not r_veto) and (not r_invalid) and examined == r_applied


def classify(
    *,
    listed: Iterable[int],
    apply_applied: Iterable[int],
    apply_veto_count: int,
    mode_fed_veto: Iterable[int],
    mode_fed_applied: Iterable[int],
    types: Mapping[int, str],
    control_failing: Iterable[int] = (),
) -> dict[str, Any]:
    listed_s = {int(i) for i in listed}
    applied_s = {int(i) for i in apply_applied}
    mode_veto_s = {int(i) for i in mode_fed_veto}
    mode_applied_s = {int(i) for i in mode_fed_applied}
    failing_s = {int(i) for i in control_failing}
    listed_absent = listed_s - applied_s
    examined = listed_s | (applied_s - listed_s)

    if failing_s:
        return _stop("control_untouched_failed", listed_s, examined)
    if listed_absent:
        return _stop("listed_absent_from_apply", listed_s, examined)

    r_applied = {i for i in examined if types[i] == APPLIED}
    r_veto = {i for i in examined if types[i] == VETO_RESIDUAL}
    r_invalid = {i for i in examined if types[i] == INVALID}
    if r_invalid:
        return _stop("invalid_quadrant", listed_s, examined)

    apply_veto = int(apply_veto_count)
    if (len(listed_s), apply_veto) != (
        REPRODUCED_LISTED_CARDINALITY,
        REPRODUCED_APPLY_VETO_COUNT,
    ):
        sub = []
        if len(listed_s) != REPRODUCED_LISTED_CARDINALITY:
            sub.append("D_listed_cardinality_not_949")
        if apply_veto != REPRODUCED_APPLY_VETO_COUNT:
            sub.append("D_apply_veto_count_not_0")
        return {
            "branch": BRANCH_D,
            "reason": "D_both" if len(sub) == 2 else sub[0],
            "sub_reasons": sub,
            **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
        }

    n_listed_applied = sum(1 for i in listed_s if types[i] == APPLIED)
    n_listed_veto = sum(1 for i in listed_s if types[i] == VETO_RESIDUAL)
    if n_listed_applied > 0 and n_listed_veto > 0:
        return {
            "branch": BRANCH_C,
            "reason": "listed_mixed",
            "n_listed_APPLIED": n_listed_applied,
            "n_listed_VETO_RESIDUAL": n_listed_veto,
            **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
        }

    # F keeps L_mode-empty as a discriminator; tests vary it.
    if examined and examined == r_veto and (not r_applied) and (not mode_veto_s):
        return {
            "branch": BRANCH_F,
            "reason": "mode_contract_violation",
            **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
        }

    count_ok = apply_side_count_correct(
        r_veto=r_veto,
        r_invalid=r_invalid,
        r_applied=r_applied,
        examined=examined,
    )
    mode_fed_agrees = mode_veto_s == r_veto and mode_applied_s == r_applied
    if count_ok and not mode_fed_agrees:
        return {
            "branch": BRANCH_B,
            "reason": "mode_fed_disagrees",
            **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
        }
    if count_ok and mode_fed_agrees and len(listed_s) == REPRODUCED_LISTED_CARDINALITY:
        return {
            "branch": BRANCH_E,
            "reason": "comparison_ill_posed",
            **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
        }
    return {
        "branch": BRANCH_RESIDUAL_STOP,
        "reason": "residual_unclassified",
        **_sets(listed_s, examined, r_applied, r_veto, r_invalid),
    }


def _stop(reason: str, listed: set[int], examined: set[int]) -> dict[str, Any]:
    return {
        "branch": BRANCH_STOP,
        "reason": reason,
        "listed_cardinality": len(listed),
        "examined_cardinality": len(examined),
    }


def _sets(
    listed: set[int],
    examined: set[int],
    r_applied: set[int],
    r_veto: set[int],
    r_invalid: set[int],
) -> dict[str, Any]:
    return {
        "listed_cardinality": len(listed),
        "examined_cardinality": len(examined),
        "R_APPLIED_cardinality": len(r_applied),
        "R_VETO_cardinality": len(r_veto),
        "R_INVALID_cardinality": len(r_invalid),
    }


def emit_raw_value(val: Any, *, tensor_hasher: Callable[[Any], str]) -> Any:
    if val is None:
        return None
    if hasattr(val, "dtype") and hasattr(val, "shape"):
        return {
            "kind": "tensor",
            "sha256": tensor_hasher(val),
            "shape": list(val.shape),
            "dtype": str(val.dtype),
        }
    if isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "value") and not isinstance(val, (bytes, bytearray)):
        return {"kind": "enum", "value": str(val.value)}
    if isinstance(val, Mapping):
        return {str(k): emit_raw_value(v, tensor_hasher=tensor_hasher) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [emit_raw_value(v, tensor_hasher=tensor_hasher) for v in val]
    return {"kind": "unparsed", "type": type(val).__name__}


def emit_object_inventory(obj: Any, *, tensor_hasher: Callable[[Any], str]) -> dict[str, Any]:
    """Both enumerations; the caller does not choose between them."""
    typ = type(obj)
    try:
        field_names = [f.name for f in dataclass_fields(typ)]
    except TypeError:
        field_names = []
    var_keys = sorted(vars(obj).keys()) if hasattr(obj, "__dict__") else []
    names = sorted(set(field_names) | set(var_keys))
    values: dict[str, Any] = {}
    for name in names:
        if hasattr(obj, name):
            values[name] = emit_raw_value(getattr(obj, name), tensor_hasher=tensor_hasher)
    return {
        "type": f"{typ.__module__}.{typ.__qualname__}",
        "dataclasses_fields": field_names,
        "vars_keys": var_keys,
        "emitted_names": sorted(values),
        "values": values,
    }


def compose_front_c_observer(
    existing: Any,
    extra_capture: list[Any] | None,
) -> Any:
    """Compose a list-capture with an existing Front-C observer. Never replace.

    Attribution capture does not use this function.
    If extra_capture is None, return existing unchanged (may be None).
    """
    if extra_capture is None:
        return existing

    def composed(observation: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        extra_capture.append(observation)
        if existing is not None:
            return existing(observation, *args, **kwargs)
        return None

    return composed
