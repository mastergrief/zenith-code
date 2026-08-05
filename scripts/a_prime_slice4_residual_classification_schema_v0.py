"""A′ slice-4 Rung-3 residual classification schema (STEP-1).

Exact-type predicates, integer arithmetic, constants, projection admission.
No branch rules, no IO/CLI. Imports nothing local.
PLAN v6: bddf41c768b48efc4346618c3c9f9f8f5285eacfcf1159de728c5b4743fb96ca
"""
from __future__ import annotations

from typing import Any, Mapping

HORIZONS: tuple[int, ...] = (10, 20, 50)
ARMS: tuple[str, ...] = ("package", "out")
SUPPORTS: tuple[str, ...] = ("L0b", "math_a0")
EXPECTED_CARDINALITY: dict[str, int] = {"L0b": 230, "math_a0": 1255}

MIN_BUCKET_DENOMINATOR_ROWS = 20
# coverage: rows 4/5, buckets 1/2 — integer comparisons only

PREEMPTING_ONLY: tuple[str, ...] = (
    "IDENTITY_BIND_FAIL",
    "INSTRUMENT_OR_BIND_FAIL",
)

OVERLAP_PER_SUPPORT = ("STABLE_CORE", "PARTIAL_CORE", "CHURNED", "DEGENERATE_EMPTY")
OVERLAP_COMPOSITE = (
    "STABLE_CORE",
    "PARTIAL_CORE",
    "CHURNED",
    "SPLIT_SUPPORTS",
    "DEGENERATE_EMPTY",
)
NINE_CELL_TABLE: dict[tuple[str, str], str] = {
    ("STABLE_CORE", "STABLE_CORE"): "STABLE_CORE",
    ("STABLE_CORE", "PARTIAL_CORE"): "PARTIAL_CORE",
    ("STABLE_CORE", "CHURNED"): "SPLIT_SUPPORTS",
    ("PARTIAL_CORE", "STABLE_CORE"): "PARTIAL_CORE",
    ("PARTIAL_CORE", "PARTIAL_CORE"): "PARTIAL_CORE",
    ("PARTIAL_CORE", "CHURNED"): "SPLIT_SUPPORTS",
    ("CHURNED", "STABLE_CORE"): "SPLIT_SUPPORTS",
    ("CHURNED", "PARTIAL_CORE"): "SPLIT_SUPPORTS",
    ("CHURNED", "CHURNED"): "CHURNED",
}

RESCUE_PER_SUPPORT = ("TRANSIENT", "PERSISTENT", "NONMONOTONE_RESCUE", "MIXED")
RESCUE_COMPOSITE = (
    "TRANSIENT",
    "PERSISTENT",
    "NONMONOTONE_RESCUE",
    "MIXED",
    "SPLIT_SUPPORTS",
)
CL_LABELS = ("CL_TRANSIENT", "CL_PERSISTENT", "CL_NONMONOTONE", "CL_MIXED")

RESIDUAL_PER_SUPPORT = (
    "STRATIFIED",
    "UNIFORM",
    "METADATA_INSUFFICIENT",
    "DEGENERATE_NO_SURVIVORS",
)
RESIDUAL_COMPOSITE = (
    "STRATIFIED",
    "UNIFORM",
    "SPLIT_SUPPORTS",
    "METADATA_INSUFFICIENT",
    "DEGENERATE_NO_SURVIVORS",
)

Q3_BRANCH_ARM = "package"
Q3_BRANCH_HORIZON = 50

REQUIRED_CLAIM_BOUNDARY: dict[str, bool] = {
    "receipts_only_descriptive": True,
    "no_cause": True,
    "no_individual_replay_vs_pc_attribution": True,
    "no_mechanism_mint": True,
    "no_carrier_claim": True,
    "no_readiness_acquisition_bank": True,
    "pre_carrier": True,
}


def is_exact_bool(v: Any) -> bool:
    return type(v) is bool


def is_exact_int(v: Any) -> bool:
    return type(v) is int


def is_exact_str(v: Any) -> bool:
    return type(v) is str


def is_exact_list(v: Any) -> bool:
    return type(v) is list


def is_exact_dict(v: Any) -> bool:
    return type(v) is dict


def is_exact_set(v: Any) -> bool:
    return type(v) is set


def ceil_0_70(n: int) -> int:
    """ceil(0.70 * n) via integer-only (7*n+9)//10. No float path."""
    if not is_exact_int(n) or n < 0:
        raise ValueError(f"ceil_0_70_bad_n:{n!r}")
    return (7 * n + 9) // 10


def j_ge_0_8(intersect: int, union: int) -> bool:
    """J >= 4/5 ⇔ |∩|*5 >= |∪|*4."""
    if not (is_exact_int(intersect) and is_exact_int(union)):
        raise ValueError("j_ge_0_8_types")
    if union < 0 or intersect < 0 or intersect > union:
        raise ValueError("j_ge_0_8_domain")
    if union == 0:
        raise ValueError("j_ge_0_8_empty_union")
    return intersect * 5 >= union * 4


def j_le_0_3(intersect: int, union: int) -> bool:
    """J <= 3/10 ⇔ |∩|*10 <= |∪|*3."""
    if not (is_exact_int(intersect) and is_exact_int(union)):
        raise ValueError("j_le_0_3_types")
    if union < 0 or intersect < 0 or intersect > union:
        raise ValueError("j_le_0_3_domain")
    if union == 0:
        raise ValueError("j_le_0_3_empty_union")
    return intersect * 10 <= union * 3


def enrichment_ge_1_5(
    bucket_surv: int, bucket_rows: int, support_surv: int, support_rows: int
) -> bool:
    """(bs/br)/(ss/sr) >= 3/2 ⇔ bs*sr*2 >= br*ss*3."""
    for x in (bucket_surv, bucket_rows, support_surv, support_rows):
        if not is_exact_int(x) or x < 0:
            raise ValueError("enrichment_ge_types")
    if bucket_rows <= 0 or support_rows <= 0 or support_surv <= 0:
        raise ValueError("enrichment_ge_domain")
    return bucket_surv * support_rows * 2 >= bucket_rows * support_surv * 3


def enrichment_le_0_5(
    bucket_surv: int, bucket_rows: int, support_surv: int, support_rows: int
) -> bool:
    """(bs/br)/(ss/sr) <= 1/2 ⇔ bs*sr*2 <= br*ss*1."""
    for x in (bucket_surv, bucket_rows, support_surv, support_rows):
        if not is_exact_int(x) or x < 0:
            raise ValueError("enrichment_le_types")
    if bucket_rows <= 0 or support_rows <= 0 or support_surv <= 0:
        raise ValueError("enrichment_le_domain")
    return bucket_surv * support_rows * 2 <= bucket_rows * support_surv * 1


def coverage_rows_ok(eligible_rows: int, support_rows: int) -> bool:
    """eligible/support >= 4/5 ⇔ eligible*5 >= support*4."""
    if not (is_exact_int(eligible_rows) and is_exact_int(support_rows)):
        raise ValueError("coverage_rows_types")
    if support_rows <= 0 or eligible_rows < 0 or eligible_rows > support_rows:
        raise ValueError("coverage_rows_domain")
    return eligible_rows * 5 >= support_rows * 4


def coverage_buckets_ok(eligible_buckets: int, nonempty_buckets: int) -> bool:
    """eligible/nonempty >= 1/2 ⇔ eligible*2 >= nonempty."""
    if not (is_exact_int(eligible_buckets) and is_exact_int(nonempty_buckets)):
        raise ValueError("coverage_buckets_types")
    if nonempty_buckets < 0 or eligible_buckets < 0 or eligible_buckets > nonempty_buckets:
        raise ValueError("coverage_buckets_domain")
    if nonempty_buckets == 0:
        return False
    return eligible_buckets * 2 >= nonempty_buckets


def row_id_suffix(row_id: str) -> str:
    if not is_exact_str(row_id) or ":" not in row_id:
        raise ValueError(f"row_id_bad:{row_id!r}")
    return row_id.rsplit(":", 1)[-1]


def admit_horizon_view(view: Any, *, support: str) -> list[str]:
    """Admit one support×arm×horizon projection. Returns reasons (empty=ok)."""
    reasons: list[str] = []
    if not is_exact_dict(view):
        return ["view_not_dict"]
    for key in (
        "row_ids",
        "sample_hashes",
        "source_buckets",
        "strict_failure_row_ids",
        "support_rows_audited",
    ):
        if key not in view:
            reasons.append(f"missing_{key}")
    if reasons:
        return reasons
    row_ids = view["row_ids"]
    sample_hashes = view["sample_hashes"]
    source_buckets = view["source_buckets"]
    fails = view["strict_failure_row_ids"]
    audited = view["support_rows_audited"]
    if not is_exact_list(row_ids) or not all(is_exact_str(x) for x in row_ids):
        reasons.append("row_ids_type")
    if not is_exact_list(sample_hashes) or not all(is_exact_str(x) for x in sample_hashes):
        reasons.append("sample_hashes_type")
    if not is_exact_list(source_buckets) or not all(is_exact_str(x) for x in source_buckets):
        reasons.append("source_buckets_type")
    if not is_exact_list(fails) or not all(is_exact_str(x) for x in fails):
        reasons.append("strict_failure_row_ids_type")
    if not is_exact_int(audited):
        reasons.append("support_rows_audited_type")
    if reasons:
        return reasons
    # Length checks BEFORE any index into parallel lists (IDENTITY-class).
    if len(row_ids) != len(sample_hashes):
        reasons.append(
            f"length_mismatch_sample_hashes:{len(sample_hashes)}!={len(row_ids)}"
        )
    if len(row_ids) != len(source_buckets):
        reasons.append(
            f"length_mismatch_source_buckets:{len(source_buckets)}!={len(row_ids)}"
        )
    if len(row_ids) != audited:
        reasons.append(f"len_row_ids_ne_audited:{len(row_ids)}!={audited}")
    exp = EXPECTED_CARDINALITY.get(support)
    if exp is not None and audited != exp:
        reasons.append(f"audited_ne_expected:{audited}!={exp}")
    if len(set(row_ids)) != len(row_ids):
        reasons.append("duplicate_row_ids")
    # Fail closed on structural length mismatch — never index misaligned lists.
    if reasons:
        return reasons
    for i, rid in enumerate(row_ids):
        try:
            suf = row_id_suffix(rid)
        except ValueError:
            reasons.append(f"row_id_form:{rid!r}")
            continue
        if suf != sample_hashes[i]:
            reasons.append(f"suffix_ne_sample_hash:{rid}")
        if not is_exact_str(source_buckets[i]) or source_buckets[i] == "":
            reasons.append(f"empty_bucket:{rid}")
    fail_set = set(fails)
    if len(fail_set) != len(fails):
        reasons.append("duplicate_failure_ids")
    universe = set(row_ids)
    for fid in fails:
        if fid not in universe:
            reasons.append(f"failure_not_in_universe:{fid}")
    return reasons


def extract_universe(view: Mapping[str, Any]) -> list[str]:
    return list(view["row_ids"])


def extract_bucket_map(view: Mapping[str, Any]) -> dict[str, str]:
    """Requires admitted view: len(row_ids)==len(source_buckets)."""
    row_ids = view["row_ids"]
    source_buckets = view["source_buckets"]
    if len(row_ids) != len(source_buckets):
        raise ValueError("extract_bucket_map_length_mismatch")
    return {rid: source_buckets[i] for i, rid in enumerate(row_ids)}


def extract_survivors(view: Mapping[str, Any]) -> set[str]:
    universe = set(view["row_ids"])
    fails = set(view["strict_failure_row_ids"])
    return universe - fails
