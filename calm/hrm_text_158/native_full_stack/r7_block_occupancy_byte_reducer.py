"""R7 block-occupancy offline byte reducer (pure CPU core; no IO/CLI/GPU).

Consumes chunk['block_occupancy_B64'] + B2 S_ss + Q-companion geometry.
EV 35af31ef applies to optimistic_union_projected_acc_bpw only.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.r7_b2_table2_trajectory_sufficiency_reducer import (
    reduce_b2_trajectory,
)
from calm.hrm_text_158.native_full_stack.r7_block_occupancy_b64 import (
    BINARY_ENCODING,
    DEFAULT_B,
    DEFAULT_K,
    SCHEMA_VERSION as OCC_SCHEMA,
    _eoe_block_ids_from_bitmap,
    _eoe_set_sha256,
)

CENSUS_SCHEMA = "hrm_text_158_r7_selective_drain_eligibility_census_step_chunk/v1"
CENSUS_OK = "OK"
E1_SCHEMA_STR = "hrm_text_158_r7_e1_block_bitmap_packed_nonzero_w8_snapshot/v1"
ACC_BUDGET_BPW = 0.4
Q_HELD_BPW = 1.6
BOUND_CLASS = "DIAGNOSTIC_PRIOR"
E1_ACC_HEADER_FIELDS: tuple[tuple[str, int], ...] = (
    ("schema_version_len_u8", 1),
    ("schema_version_utf8", len(E1_SCHEMA_STR.encode("utf-8"))),
    ("eligible_numel_u64", 8),
    ("block_size_u32", 4),
    ("n_blocks_u32", 4),
    ("n_active_blocks_u32", 4),
    ("nnz_u64", 8),
    ("k_star_u32", 4),
    ("drain_generation_u64", 8),
    ("payload_dtype_u8", 1),
    ("bitmap_nbytes_u32", 4),
    ("payload_nbytes_u64", 8),
    ("content_sha256", 32),
)
HEADER_BYTES_REAL = sum(w for _, w in E1_ACC_HEADER_FIELDS)
assert HEADER_BYTES_REAL == 147 and E1_ACC_HEADER_FIELDS[1][1] == 61

MISSING_OBSERVABLES = "MISSING_OBSERVABLES"
PROCEED_TO_M1_SCREEN = "PROCEED_TO_M1_SCREEN"
DEPRIORITIZE_M1 = "DEPRIORITIZE_M1"
EV_CLASSIFIER_SHA256 = "35af31ef322130257fab0efd35250f1e79a0c2f4238242dc17efa6d80dd7d21e"
EV_LITERALS = {
    MISSING_OBSERVABLES: "either seed fails schema/domain/hash/closure/byte-projection integrity.",
    PROCEED_TO_M1_SCREEN: (
        "BOTH receipts valid AND at least ONE seed has exact "
        "optimistic_union_projected_acc_bpw < 0.4."
    ),
    DEPRIORITIZE_M1: "BOTH valid seeds have exact optimistic-union projected_acc_bpw >= 0.4.",
}


@dataclass(frozen=True, slots=True)
class QCompanionModule:
    state_key: str
    logical_numel: int


@dataclass(frozen=True, slots=True)
class QCompanionGeometry:
    modules: tuple[QCompanionModule, ...]

    def sorted_modules(self) -> tuple[QCompanionModule, ...]:
        return tuple(sorted(self.modules, key=lambda m: m.state_key))


def companion_from_numel_map(numel_by_key: Mapping[str, int]) -> QCompanionGeometry:
    return QCompanionGeometry(
        tuple(QCompanionModule(str(k), int(v)) for k, v in sorted(numel_by_key.items()))
    )


@dataclass(frozen=True, slots=True)
class ByteProjection:
    nnz: int
    bitmap_bytes: int
    payload_bytes: int
    header_bytes: int
    content_bytes: int
    aligned_bytes: int
    metadata_alignment_bytes: int
    projected_acc_bpw: float
    inclusive_bpw: float
    margin: float
    strict_lt_0_4: bool


@dataclass(frozen=True, slots=True)
class PerStepReport:
    step: int
    fully_eoe_fraction: float
    optimistic_nnz: int
    pessimistic_nnz: int
    projected_acc_bpw_optimistic: float
    projected_acc_bpw_pessimistic: float
    margin_optimistic: float
    margin_pessimistic: float


@dataclass(frozen=True, slots=True)
class ByteReduceResult:
    overall: str
    failure_locus: str | None
    N: int
    S_ss: tuple[int, ...]
    anchor_step: int | None
    eligible_weights: int | None
    n_blocks_global: int | None
    header_bytes: int
    zeroable_weights: int | None
    optimistic_nnz_union: int | None
    pessimistic_nnz_union: int | None
    eligible_nonzero_unknown: bool | None
    optimistic_union_projected_acc_bpw: float | None
    pessimistic_union_projected_acc_bpw: float | None
    inclusive_bpw_optimistic: float | None
    inclusive_bpw_pessimistic: float | None
    margin_optimistic: float | None
    margin_pessimistic: float | None
    strict_lt_0_4_optimistic: bool | None
    bound_class_label: str
    per_step: tuple[PerStepReport, ...]
    opt_projection: ByteProjection | None
    pess_projection: ByteProjection | None


@dataclass(frozen=True, slots=True)
class EvVerdict:
    outcome: str
    primary_overall: str
    independent_overall: str
    primary_optimistic_bpw: float | None
    independent_optimistic_bpw: float | None
    classifier_sha256: str
    bound_class_label: str


def header_field_enumeration() -> tuple[tuple[str, int], ...]:
    return E1_ACC_HEADER_FIELDS


def _missing(locus: str, *, N: int, S_ss: tuple[int, ...] = (), anchor: int | None = None) -> ByteReduceResult:
    return ByteReduceResult(
        MISSING_OBSERVABLES, locus, N, S_ss, anchor, None, None, HEADER_BYTES_REAL,
        None, None, None, None, None, None, None, None, None, None, None, BOUND_CLASS,
        (), None, None,
    )


def _n_blocks(n: int, B: int = DEFAULT_B) -> int:
    return (n + B - 1) // B


def _tail_len(n: int, B: int = DEFAULT_B) -> int:
    return n % B


def _block_len(n: int, b: int, B: int = DEFAULT_B) -> int:
    return min(B, n - b * B)


def _project(nnz: int, *, bitmap_bytes: int, eligible_weights: int) -> ByteProjection:
    payload = int(nnz)
    content = HEADER_BYTES_REAL + bitmap_bytes + payload
    aligned = ((content + 7) // 8) * 8
    bpw = (aligned * 8) / float(eligible_weights)
    return ByteProjection(
        payload, bitmap_bytes, payload, HEADER_BYTES_REAL, content, aligned,
        aligned - content, bpw, Q_HELD_BPW + bpw, bpw - ACC_BUDGET_BPW, bpw < ACC_BUDGET_BPW,
    )


def _b64(field: object) -> bytes | None:
    if not isinstance(field, str):
        return None
    try:
        return base64.b64decode(field, validate=True)
    except Exception:
        return None


def _validate_final_four(ff: Any, *, W: int) -> tuple[tuple[int, ...], str | None]:
    if ff is None:
        return (), "final_four_available_none"
    if not isinstance(ff, (tuple, list)) or len(ff) != 4:
        return (), "final_four_len_ne_4"
    ends: list[int] = []
    for w in ff:
        end = getattr(w, "end_step", None)
        if type(end) is not int:
            return (), "final_four_end_not_int"
        ends.append(end)
    if len(set(ends)) != 4 or any(ends[i] >= ends[i + 1] for i in range(3)):
        return (), "final_four_ends_invalid"
    supports = [set(range(e - W + 1, e + 1)) for e in ends]
    if any(len(s) != W for s in supports):
        return (), "final_four_invalid_W"
    union = sorted(set().union(*supports))
    if not union or union != list(range(union[0], union[-1] + 1)):
        return (), "final_four_union_noncontiguous"
    return tuple(union), None


def _parse_occ_state(ps: Mapping[str, Any], *, expect_numel: int) -> tuple[dict[str, Any], str | None]:
    sk = ps.get("state_key")
    if not isinstance(sk, str) or not sk:
        return {}, "bad_state_key"
    numel, n_blocks, tail = ps.get("logical_numel"), ps.get("n_blocks"), ps.get("tail_len")
    if type(numel) is not int or type(n_blocks) is not int or type(tail) is not int:
        return {}, f"noncanonical_geometry:{sk}"
    if numel != expect_numel:
        return {}, f"companion_numel_mismatch:{sk}"
    if n_blocks != _n_blocks(numel) or tail != _tail_len(numel):
        return {}, f"geometry_derive_mismatch:{sk}"
    elig, nzn, empty, bitmap = (
        _b64(ps.get("per_block_eligible_u8_b64")),
        _b64(ps.get("per_block_noneligible_nonzero_u8_b64")),
        _b64(ps.get("per_block_empty_u8_b64")),
        _b64(ps.get("fully_eoe_block_bitmap_b64")),
    )
    if elig is None or nzn is None or empty is None or bitmap is None:
        return {}, f"b64_decode_fail:{sk}"
    if len(elig) != n_blocks or len(nzn) != n_blocks or len(empty) != n_blocks:
        return {}, f"u8_len_mismatch:{sk}"
    if len(bitmap) != (n_blocks + 7) // 8:
        return {}, f"bitmap_len_mismatch:{sk}"
    eoe_ids = _eoe_block_ids_from_bitmap(bitmap, n_blocks)
    if ps.get("fully_eoe_set_sha256") != _eoe_set_sha256(eoe_ids):
        return {}, f"eoe_sha_mismatch:{sk}"
    fc = ps.get("fully_eoe_count")
    if ps.get("set_hash_ok") is not True or type(fc) is not int or fc != len(eoe_ids):
        return {}, f"eoe_meta_mismatch:{sk}"
    for b in range(n_blocks):
        if elig[b] + nzn[b] + empty[b] != _block_len(numel, b):
            return {}, f"count_closure:{sk}:{b}"
        if bool(bitmap[b >> 3] & (1 << (b & 7))) != (nzn[b] == 0):
            return {}, f"eoe_bit_closure:{sk}:{b}"
    return {
        "state_key": sk, "logical_numel": numel, "n_blocks": n_blocks, "tail_len": tail,
        "elig": elig, "nzn": nzn, "empty": empty, "bitmap": bitmap, "eoe_ids": set(eoe_ids),
    }, None


def _residual_nnz(parsed: Mapping[str, Any], residual: set[int]) -> tuple[int, int]:
    elig, nzn = parsed["elig"], parsed["nzn"]
    opt = pess = 0
    for b in residual:
        opt += int(nzn[b])
        pess += int(elig[b]) + int(nzn[b])
    return opt, pess


def reduce_block_occupancy_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    companion: QCompanionGeometry,
    N: int = 32,
    W: int = 8,
) -> ByteReduceResult:
    mods = companion.sorted_modules()
    if not mods:
        return _missing("companion_empty", N=N)
    keys = [m.state_key for m in mods]
    if len(keys) != len(set(keys)):
        return _missing("companion_duplicate_keys", N=N)
    if any(type(m.logical_numel) is not int or m.logical_numel <= 0 for m in mods):
        return _missing("companion_bad_numel", N=N)
    expect = {m.state_key: m.logical_numel for m in mods}
    eligible_weights = sum(expect.values())
    n_blocks_global = sum(_n_blocks(n) for n in expect.values())
    bitmap_bytes = (n_blocks_global + 7) // 8

    b2 = reduce_b2_trajectory(rows, N=N, W=W)
    if not b2.integrity_gate.passed:
        return _missing("b2_integrity_fail", N=N)
    S_ss, err = _validate_final_four(b2.rolling_mean_W8.final_four_available, W=W)
    if err is not None:
        return _missing(f"s_ss:{err}", N=N)
    anchor = max(S_ss)

    ordinary = [r for r in rows if isinstance(r, Mapping) and r.get("schema_version") == CENSUS_SCHEMA]
    if len(ordinary) != N:
        return _missing("ordinary_count", N=N, S_ss=S_ss, anchor=anchor)
    by_step: dict[int, Mapping[str, Any]] = {}
    for r in ordinary:
        step = r.get("step")
        if type(step) is not int or step in by_step:
            return _missing("step_index", N=N, S_ss=S_ss, anchor=anchor)
        by_step[step] = r
    if sorted(by_step) != list(range(1, N + 1)):
        return _missing("steps_not_1_n", N=N, S_ss=S_ss, anchor=anchor)

    per_step_parsed: dict[int, dict[str, dict[str, Any]]] = {}
    reports: list[PerStepReport] = []
    for step in range(1, N + 1):
        row = by_step[step]
        if row.get("census_status") != CENSUS_OK:
            return _missing(f"census_status:{step}", N=N, S_ss=S_ss, anchor=anchor)
        occ = row.get("block_occupancy_B64")
        if not isinstance(occ, Mapping):
            return _missing(f"occupancy_absent:{step}", N=N, S_ss=S_ss, anchor=anchor)
        if (
            occ.get("schema_version") != OCC_SCHEMA or occ.get("B") != DEFAULT_B
            or occ.get("k") != DEFAULT_K or occ.get("event_coded_live") is not False
            or occ.get("binary_encoding") != BINARY_ENCODING
        ):
            return _missing(f"occupancy_header:{step}", N=N, S_ss=S_ss, anchor=anchor)
        per_state = occ.get("per_state")
        if not isinstance(per_state, list):
            return _missing(f"per_state_not_list:{step}", N=N, S_ss=S_ss, anchor=anchor)
        seen: set[str] = set()
        parsed_by_key: dict[str, dict[str, Any]] = {}
        for ps in per_state:
            if not isinstance(ps, Mapping):
                return _missing(f"per_state_row:{step}", N=N, S_ss=S_ss, anchor=anchor)
            sk = ps.get("state_key")
            if not isinstance(sk, str) or sk in seen:
                return _missing(f"per_state_key:{step}", N=N, S_ss=S_ss, anchor=anchor)
            seen.add(sk)
            if sk not in expect:
                return _missing(f"extra_state:{sk}:{step}", N=N, S_ss=S_ss, anchor=anchor)
            parsed, perr = _parse_occ_state(ps, expect_numel=expect[sk])
            if perr is not None:
                return _missing(f"{perr}:{step}", N=N, S_ss=S_ss, anchor=anchor)
            parsed_by_key[sk] = parsed
        if seen != set(expect):
            return _missing(f"missing_state:{step}", N=N, S_ss=S_ss, anchor=anchor)
        opt_t = pess_t = eoe_count = 0
        for sk in keys:
            p = parsed_by_key[sk]
            eoe_count += len(p["eoe_ids"])
            o, pe = _residual_nnz(p, set(range(p["n_blocks"])) - p["eoe_ids"])
            opt_t += o
            pess_t += pe
        if opt_t > pess_t:
            return _missing(f"bound_invariant:{step}", N=N, S_ss=S_ss, anchor=anchor)
        op, pp = (
            _project(opt_t, bitmap_bytes=bitmap_bytes, eligible_weights=eligible_weights),
            _project(pess_t, bitmap_bytes=bitmap_bytes, eligible_weights=eligible_weights),
        )
        reports.append(PerStepReport(
            step, eoe_count / float(n_blocks_global), opt_t, pess_t,
            op.projected_acc_bpw, pp.projected_acc_bpw, op.margin, pp.margin,
        ))
        per_step_parsed[step] = parsed_by_key

    eoe_union = {sk: set() for sk in keys}
    for step in S_ss:
        for sk in keys:
            eoe_union[sk] |= per_step_parsed[step][sk]["eoe_ids"]
    opt_u = pess_u = zeroable = residual_elig = 0
    for sk in keys:
        p = per_step_parsed[anchor][sk]
        residual = set(range(p["n_blocks"])) - eoe_union[sk]
        o, pe = _residual_nnz(p, residual)
        opt_u += o
        pess_u += pe
        for b in eoe_union[sk]:
            zeroable += _block_len(p["logical_numel"], b)
        for b in residual:
            residual_elig += int(p["elig"][b])
    if opt_u > pess_u:
        return _missing("bound_invariant_union", N=N, S_ss=S_ss, anchor=anchor)
    op = _project(opt_u, bitmap_bytes=bitmap_bytes, eligible_weights=eligible_weights)
    pp = _project(pess_u, bitmap_bytes=bitmap_bytes, eligible_weights=eligible_weights)
    return ByteReduceResult(
        "OK", None, N, S_ss, anchor, eligible_weights, n_blocks_global, HEADER_BYTES_REAL,
        zeroable, opt_u, pess_u, residual_elig > 0, op.projected_acc_bpw, pp.projected_acc_bpw,
        op.inclusive_bpw, pp.inclusive_bpw, op.margin, pp.margin, op.strict_lt_0_4, BOUND_CLASS,
        tuple(reports), op, pp,
    )


def apply_ev_classifier(primary: ByteReduceResult, independent: ByteReduceResult) -> EvVerdict:
    if primary.overall != "OK" or independent.overall != "OK":
        return EvVerdict(
            MISSING_OBSERVABLES, primary.overall, independent.overall,
            primary.optimistic_union_projected_acc_bpw, independent.optimistic_union_projected_acc_bpw,
            EV_CLASSIFIER_SHA256, BOUND_CLASS,
        )
    p, i = primary.optimistic_union_projected_acc_bpw, independent.optimistic_union_projected_acc_bpw
    assert p is not None and i is not None
    outcome = PROCEED_TO_M1_SCREEN if (p < ACC_BUDGET_BPW or i < ACC_BUDGET_BPW) else DEPRIORITIZE_M1
    return EvVerdict(outcome, primary.overall, independent.overall, p, i, EV_CLASSIFIER_SHA256, BOUND_CLASS)


def to_json_dict(result: ByteReduceResult) -> dict[str, Any]:
    def _proj(p: ByteProjection | None) -> dict[str, Any] | None:
        if p is None:
            return None
        return {
            "nnz": p.nnz, "bitmap_bytes": p.bitmap_bytes, "payload_bytes": p.payload_bytes,
            "header_bytes": p.header_bytes, "content_bytes": p.content_bytes,
            "aligned_bytes": p.aligned_bytes, "metadata_alignment_bytes": p.metadata_alignment_bytes,
            "projected_acc_bpw": p.projected_acc_bpw, "inclusive_bpw": p.inclusive_bpw,
            "margin": p.margin, "strict_lt_0_4": p.strict_lt_0_4,
        }

    return {
        "overall": result.overall, "failure_locus": result.failure_locus, "N": result.N,
        "S_ss": list(result.S_ss), "anchor_step": result.anchor_step,
        "eligible_weights": result.eligible_weights, "n_blocks_global": result.n_blocks_global,
        "header_bytes": result.header_bytes, "zeroable_weights": result.zeroable_weights,
        "optimistic_nnz_union": result.optimistic_nnz_union,
        "pessimistic_nnz_union": result.pessimistic_nnz_union,
        "eligible_nonzero_unknown": result.eligible_nonzero_unknown,
        "optimistic_union_projected_acc_bpw": result.optimistic_union_projected_acc_bpw,
        "pessimistic_union_projected_acc_bpw": result.pessimistic_union_projected_acc_bpw,
        "inclusive_bpw_optimistic": result.inclusive_bpw_optimistic,
        "inclusive_bpw_pessimistic": result.inclusive_bpw_pessimistic,
        "margin_optimistic": result.margin_optimistic, "margin_pessimistic": result.margin_pessimistic,
        "strict_lt_0_4_optimistic": result.strict_lt_0_4_optimistic,
        "bound_class_label": result.bound_class_label,
        "opt_projection": _proj(result.opt_projection), "pess_projection": _proj(result.pess_projection),
        "per_step": [
            {
                "step": s.step, "fully_eoe_fraction": s.fully_eoe_fraction,
                "optimistic_nnz": s.optimistic_nnz, "pessimistic_nnz": s.pessimistic_nnz,
                "projected_acc_bpw_optimistic": s.projected_acc_bpw_optimistic,
                "projected_acc_bpw_pessimistic": s.projected_acc_bpw_pessimistic,
                "margin_optimistic": s.margin_optimistic, "margin_pessimistic": s.margin_pessimistic,
            }
            for s in result.per_step
        ],
    }


def ev_to_json_dict(v: EvVerdict) -> dict[str, Any]:
    return {
        "outcome": v.outcome, "primary_overall": v.primary_overall,
        "independent_overall": v.independent_overall,
        "primary_optimistic_bpw": v.primary_optimistic_bpw,
        "independent_optimistic_bpw": v.independent_optimistic_bpw,
        "classifier_sha256": v.classifier_sha256, "bound_class_label": v.bound_class_label,
        "ev_literals": dict(EV_LITERALS),
    }
