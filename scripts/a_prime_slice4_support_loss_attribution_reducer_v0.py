"""A′ slice-4 Rung-0 pure reducer: factorized support-loss surface ATTRIBUTION.

Plan authority: A_prime_slice4_cause_localization_PLAN_v2.json
  sha bdefb0180d26dc2a65926edd43c4ed8280ce608ad7b3d88e6d264887f3f3e295

No torch/GPU. Branch math uses N1 start-survivor denominators only (230/1254).
Absolute-count share is descriptive secondary only — never a branch input.

Cycle-2: no detached receipt_objs override without byte-proof; no expected_source_shas
override; frozen terminal sha always required in production bind path.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 35, 50)
CLIFFS: tuple[tuple[int, int], ...] = ((5, 10), (10, 20))
ENDPOINT_N = 20

SUPPORT_ROWS_EXPECTED: dict[str, int] = {"L0b": 230, "math_a0": 1255}
START_SURVIVOR_DENOMINATORS: dict[str, int] = {"L0b": 230, "math_a0": 1254}

ENDPOINT_CO_COLLAPSE_OWN_LOSS = 0.90
MATERIAL_OWN_LOSS_ON_CLIFF = 0.20
ENRICHMENT_RATIO = 1.15

REQUIRED_TERMINAL_BRANCH = "NONMONOTONE_OR_MULTI_CLIFF"
REQUIRED_TERMINAL_AUTHORITY = "manifest+marker"
FROZEN_TERMINAL_SHA256 = (
    "f587cee02fad40cdf5296f37c7625e7fb263cff723ff3798fb09099095153998"
)

ENDPOINT_VALUES = ("CO_COLLAPSE", "PARTIAL", "CONTAINED")
CLIFF_VALUES = (
    "L0B_ENRICHED_BOTH",
    "MATH_A0_ENRICHED_BOTH",
    "CLIFF_SPECIFIC",
    "BALANCED",
    "SUB_THRESHOLD",
)


def parse_strict_exact_count(spec: str) -> int:
    if not isinstance(spec, str) or "/" not in spec:
        raise ValueError(f"bad strict_exact spec: {spec!r}")
    num, _den = spec.split("/", 1)
    return int(num)


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def extract_final_strict_count(receipt: Mapping[str, Any], support: str) -> int:
    pa = receipt.get("prior_audit") or {}
    deltas = pa.get("deltas") or {}
    d = deltas.get(support) or {}
    pbvf = d.get("parent_baseline_vs_final") or {}
    spec = pbvf.get("final_strict_exact")
    if spec is None:
        raise ValueError(f"missing parent_baseline_vs_final.final_strict_exact for {support}")
    return parse_strict_exact_count(spec)


def extract_protection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    pa = receipt.get("prior_audit") or {}
    return {
        "replay_pc": pa.get("replay_pc"),
        "direct_kl": pa.get("direct_kl"),
        "prior_batches_fed_to_bounded_steps": pa.get("prior_batches_fed_to_bounded_steps"),
    }


def check_per_step_global_horizon(receipt: Mapping[str, Any], n: int) -> list[str]:
    reasons: list[str] = []
    reports = receipt.get("step_reports") or {}
    if not isinstance(reports, dict) or not reports:
        reasons.append(f"N{n}:missing_step_reports")
        return reasons
    for step in range(1, n + 1):
        key = str(step)
        if key not in reports:
            reasons.append(f"N{n}:missing_step_report_{step}")
            continue
        gh = reports[key].get("global_horizon")
        if gh != 50:
            reasons.append(f"N{n}:step_{step}_global_horizon={gh!r}")
    return reasons


def own_baseline_loss_rate(lost: int, baseline: int) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return lost / baseline


def cliff_enrichment(own_l0b: float, own_math: float) -> tuple[bool, bool, bool]:
    both = own_l0b >= MATERIAL_OWN_LOSS_ON_CLIFF and own_math >= MATERIAL_OWN_LOSS_ON_CLIFF
    l0e = own_l0b >= MATERIAL_OWN_LOSS_ON_CLIFF and own_l0b >= ENRICHMENT_RATIO * own_math
    me = own_math >= MATERIAL_OWN_LOSS_ON_CLIFF and own_math >= ENRICHMENT_RATIO * own_l0b
    return l0e, me, both


def classify_endpoint_profile(ep_l0b: float, ep_math: float) -> str:
    l_hi = ep_l0b >= ENDPOINT_CO_COLLAPSE_OWN_LOSS
    m_hi = ep_math >= ENDPOINT_CO_COLLAPSE_OWN_LOSS
    if l_hi and m_hi:
        return "CO_COLLAPSE"
    if l_hi ^ m_hi:
        return "PARTIAL"
    return "CONTAINED"


def classify_cliff_profile(
    cliff_flags: list[tuple[bool, bool, bool]],
) -> str:
    if len(cliff_flags) != 2:
        raise ValueError("expected exactly two cliffs")
    (c1_l0, c1_m, c1_both), (c2_l0, c2_m, c2_both) = cliff_flags
    if c1_l0 and c2_l0:
        return "L0B_ENRICHED_BOTH"
    if c1_m and c2_m:
        return "MATH_A0_ENRICHED_BOTH"
    c1_enr = c1_l0 or c1_m
    c2_enr = c2_l0 or c2_m
    disagree = (c1_l0 and c2_m) or (c1_m and c2_l0)
    exactly_one_cliff_enriched = c1_enr ^ c2_enr
    if disagree or exactly_one_cliff_enriched:
        return "CLIFF_SPECIFIC"
    if c1_both and c2_both and not (c1_l0 or c1_m or c2_l0 or c2_m):
        return "BALANCED"
    return "SUB_THRESHOLD"


def classify_from_counts(
    counts: Mapping[int, Mapping[str, int]],
    *,
    survivor_denoms: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Classify from integer final-count tables. Rates derived only here."""
    B = dict(survivor_denoms or START_SURVIVOR_DENOMINATORS)
    b_l0b = int(B["L0b"])
    b_math = int(B["math_a0"])
    if b_math == 1255:
        raise ValueError("math_a0 survivor denom must not be 1255 in branch math")
    if b_math == SUPPORT_ROWS_EXPECTED["math_a0"]:
        raise ValueError("refusing support_rows_expected as branch denom for math_a0")

    def cnt(n: int, s: str) -> int:
        return int(counts[n][s])

    def lost(a: int, b: int, s: str) -> int:
        return cnt(a, s) - cnt(b, s)

    def own(a: int, b: int, s: str) -> float:
        base = b_l0b if s == "L0b" else b_math
        return own_baseline_loss_rate(lost(a, b, s), base)

    def endpoint(s: str) -> float:
        base = b_l0b if s == "L0b" else b_math
        return own_baseline_loss_rate(base - cnt(ENDPOINT_N, s), base)

    cliff_metrics: list[dict[str, Any]] = []
    cliff_flags: list[tuple[bool, bool, bool]] = []
    for a, b in CLIFFS:
        ol = own(a, b, "L0b")
        om = own(a, b, "math_a0")
        flags = cliff_enrichment(ol, om)
        cliff_flags.append(flags)
        total_drop = lost(a, b, "L0b") + lost(a, b, "math_a0")
        cliff_metrics.append(
            {
                "from": a,
                "to": b,
                "L0b_lost": lost(a, b, "L0b"),
                "math_a0_lost": lost(a, b, "math_a0"),
                "L0b_own_baseline_loss_rate": ol,
                "math_a0_own_baseline_loss_rate": om,
                "abs_share_L0b": (lost(a, b, "L0b") / total_drop) if total_drop else None,
                "abs_share_math_a0": (lost(a, b, "math_a0") / total_drop) if total_drop else None,
                "l0b_enriched": flags[0],
                "math_enriched": flags[1],
                "both_material": flags[2],
            }
        )

    ep_l = endpoint("L0b")
    ep_m = endpoint("math_a0")
    endpoint_profile = classify_endpoint_profile(ep_l, ep_m)
    cliff_profile = classify_cliff_profile(cliff_flags)
    branch = f"{endpoint_profile}__{cliff_profile}"

    return {
        "branch": branch,
        "endpoint_profile": endpoint_profile,
        "cliff_profile": cliff_profile,
        "survivor_denominators": {"L0b": b_l0b, "math_a0": b_math},
        "support_rows_expected": dict(SUPPORT_ROWS_EXPECTED),
        "counts": {str(n): dict(counts[n]) for n in sorted(counts)},
        "endpoint": {
            "n": ENDPOINT_N,
            "L0b_own_loss": ep_l,
            "math_a0_own_loss": ep_m,
        },
        "cliffs": cliff_metrics,
        "claim_boundary": {
            "attribution_only": True,
            "pre_cause": True,
            "pre_carrier": True,
            "absolute_share_not_branch_input": True,
        },
    }


def _canonical_json_obj(obj: Any) -> Any:
    """Normalize JSON object for equality after loads (dicts sorted keys recurse)."""
    return json.loads(json.dumps(obj, sort_keys=True))


def bind_and_extract(
    *,
    terminal: Mapping[str, Any],
    terminal_sha256: str,
    receipt_bytes_by_n: Mapping[int, bytes],
    require_frozen_terminal_sha: bool = True,
) -> dict[str, Any]:
    """Same-byte authority then attribution.

    Classifies ONLY objects parsed from hashed raw receipt bytes.
    Governing source_shas come ONLY from terminal.source_shas.
    """
    reasons: list[str] = []

    if require_frozen_terminal_sha and terminal_sha256 != FROZEN_TERMINAL_SHA256:
        reasons.append(f"terminal_sha_mismatch:{terminal_sha256}")
    if terminal.get("branch") != REQUIRED_TERMINAL_BRANCH:
        reasons.append(f"terminal_branch={terminal.get('branch')!r}")
    if terminal.get("terminal_authority") != REQUIRED_TERMINAL_AUTHORITY:
        reasons.append(f"terminal_authority={terminal.get('terminal_authority')!r}")
    if terminal.get("synthetic") is not False:
        reasons.append(f"synthetic={terminal.get('synthetic')!r}")

    src = terminal.get("source_shas") or {}
    if not isinstance(src, Mapping):
        reasons.append("terminal_source_shas_not_mapping")
        src = {}
    for n in HORIZONS:
        key = f"input/N{n}"
        if key not in src:
            reasons.append(f"missing_source_sha_key:{key}")

    details = (terminal.get("classification") or {}).get("details") or {}
    counts_by_n_raw = details.get("counts_by_n") or {}
    counts_by_n = {str(k): int(v) for k, v in counts_by_n_raw.items()}

    objs: dict[int, Any] = {}
    bound_shas: dict[str, str] = {}

    for n in HORIZONS:
        key = f"input/N{n}"
        raw = receipt_bytes_by_n.get(n)
        if raw is None:
            reasons.append(f"missing_receipt_bytes:N{n}")
            continue
        got = sha256_hex(raw)
        bound_shas[key] = got
        exp = src.get(key)
        if exp is not None and got != exp:
            reasons.append(f"source_sha_mismatch:{key}")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as e:
            reasons.append(f"parse_fail:N{n}:{e}")
            continue
        # production path: always use object parsed from hashed raw bytes
        objs[n] = parsed

        rec = objs[n]
        if rec.get("steps_completed") != n:
            reasons.append(f"N{n}:steps_completed={rec.get('steps_completed')!r}")
        if rec.get("parent_hash_unchanged") is not True:
            reasons.append(f"N{n}:parent_hash_unchanged={rec.get('parent_hash_unchanged')!r}")
        reasons.extend(check_per_step_global_horizon(rec, n))

        prot = extract_protection(rec)
        if prot.get("replay_pc") != "OUT":
            reasons.append(f"N{n}:replay_pc={prot.get('replay_pc')!r}")
        if prot.get("direct_kl") is not False:
            reasons.append(f"N{n}:direct_kl={prot.get('direct_kl')!r}")
        if prot.get("prior_batches_fed_to_bounded_steps") is not False:
            reasons.append(
                f"N{n}:prior_batches_fed_to_bounded_steps="
                f"{prot.get('prior_batches_fed_to_bounded_steps')!r}"
            )

    count_table: dict[int, dict[str, int]] = {}
    for n, rec in objs.items():
        try:
            l0 = extract_final_strict_count(rec, "L0b")
            m = extract_final_strict_count(rec, "math_a0")
        except Exception as e:
            reasons.append(f"N{n}:count_extract:{e}")
            continue
        count_table[n] = {"L0b": l0, "math_a0": m}
        if str(n) in counts_by_n and l0 + m != int(counts_by_n[str(n)]):
            reasons.append(f"N{n}:compose_fail sum={l0+m} expected={counts_by_n[str(n)]}")

    if 1 in count_table:
        if count_table[1]["L0b"] != START_SURVIVOR_DENOMINATORS["L0b"]:
            reasons.append(
                f"N1_survivor_L0b={count_table[1]['L0b']}!="
                f"{START_SURVIVOR_DENOMINATORS['L0b']}"
            )
        if count_table[1]["math_a0"] != START_SURVIVOR_DENOMINATORS["math_a0"]:
            reasons.append(
                f"N1_survivor_math_a0={count_table[1]['math_a0']}!="
                f"{START_SURVIVOR_DENOMINATORS['math_a0']}"
            )

    if reasons:
        return {
            "branch": "INSTRUMENT_OR_BIND_FAIL",
            "endpoint_profile": None,
            "cliff_profile": None,
            "instrument_fail": True,
            "reasons": reasons,
            "source_shas": bound_shas,
            "counts": {str(n): dict(count_table[n]) for n in sorted(count_table)},
            "claim_boundary": {
                "attribution_only": True,
                "pre_cause": True,
                "pre_carrier": True,
            },
        }

    for need in (5, 10, 20):
        if need not in count_table:
            return {
                "branch": "INSTRUMENT_OR_BIND_FAIL",
                "endpoint_profile": None,
                "cliff_profile": None,
                "instrument_fail": True,
                "reasons": [f"missing_counts:N{need}"],
                "source_shas": bound_shas,
                "claim_boundary": {
                    "attribution_only": True,
                    "pre_cause": True,
                    "pre_carrier": True,
                },
            }

    result = classify_from_counts(
        count_table, survivor_denoms=START_SURVIVOR_DENOMINATORS
    )
    result["instrument_fail"] = False
    result["reasons"] = []
    result["source_shas"] = bound_shas
    return result


def classify_suite_from_paths(
    *,
    terminal_path: str,
    receipt_paths: Mapping[int, str],
) -> dict[str, Any]:
    from pathlib import Path

    tpath = Path(terminal_path)
    traw = tpath.read_bytes()
    terminal = json.loads(traw.decode("utf-8"))
    tsha = sha256_hex(traw)
    rbytes: dict[int, bytes] = {}
    for n, p in receipt_paths.items():
        rbytes[int(n)] = Path(p).read_bytes()
    return bind_and_extract(
        terminal=terminal,
        terminal_sha256=tsha,
        receipt_bytes_by_n=rbytes,
        require_frozen_terminal_sha=True,
    )
