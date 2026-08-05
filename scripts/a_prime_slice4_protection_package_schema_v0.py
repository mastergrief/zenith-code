"""A′ slice-4 Rung-2 schema/admission primitives (cycle-5 seam).

Plan v5 addendum: schema ONLY — exact-type predicates, parse, binding/geometry/
parent/horizon/count-domain checks, envelope validators. No branch rules, no IO/CLI.
Imports nothing local.
"""
from __future__ import annotations

from typing import Any, Mapping

HORIZONS: tuple[int, ...] = (10, 20, 50)
START_SURVIVOR_DENOMINATORS: dict[str, int] = {"L0b": 230, "math_a0": 1254}
SUPPORT_ROWS_EXPECTED: dict[str, int] = {"L0b": 230, "math_a0": 1255}
FINAL_STRICT_DEN: dict[str, int] = {"L0b": 230, "math_a0": 1255}
BASELINE_STRICT: dict[str, str] = {"L0b": "230/230", "math_a0": "1254/1255"}
COUNT_DOMAIN_MAX: dict[str, int] = {"L0b": 230, "math_a0": 1255}

FROZEN_OUT_TERMINAL_SHA256 = (
    "f587cee02fad40cdf5296f37c7625e7fb263cff723ff3798fb09099095153998"
)
PARENT_SHA_EXPECTED = (
    "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
)
REQUIRED_RUN_GEOMETRY: dict[str, Any] = {
    "device": "cuda:0",
    "eligible_scope": "first-bitlinear",
    "local_selection_ordering_seed": 17,
    "banked_pt_mutated": False,
}
REQUIRED_PACKAGE_BINDING: dict[str, Any] = {
    "requested_supports": ["L0b", "math_a0"],
    "replay_ce_veto": True,
    "pc_aux_enabled": True,
    "parent_consistency_weight": 1.0,
    "pc_aux_mode": "veto",
    "prior_batches_fed_to_bounded_steps": True,
    "target_parent_kl": False,
    "target_rows_excluded_from_pc": True,
}
REQUIRED_CLAIM_BOUNDARY: dict[str, bool] = {
    "protection_package_sensitivity_only": True,
    "pre_carrier": True,
    "pre_mechanism_mint": True,
    "absolute_share_not_branch_input": True,
    "no_individual_replay_vs_pc_causal": True,
}
REQUIRED_OUT_AUTHORITY: dict[str, Any] = {
    "sha256": FROZEN_OUT_TERMINAL_SHA256,
    "branch": "NONMONOTONE_OR_MULTI_CLIFF",
    "terminal_authority": "manifest+marker",
    "synthetic": False,
}


def is_exact_bool(v: Any) -> bool:
    return type(v) is bool


def is_exact_int(v: Any) -> bool:
    return type(v) is int


def is_exact_str(v: Any) -> bool:
    return type(v) is str


def is_exact_float(v: Any) -> bool:
    return type(v) is float


def is_exact_list(v: Any) -> bool:
    return type(v) is list


def is_exact_dict(v: Any) -> bool:
    return type(v) is dict


def is_exact_number(v: Any) -> bool:
    return type(v) in (int, float)


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def parse_strict_exact_count(spec: Any, *, expected_den: int) -> int:
    """Exact num/den; digit-only; den must match. No float coercion."""
    if not is_exact_str(spec) or "/" not in spec:
        raise ValueError(f"bad_strict_exact_spec:{spec!r}")
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"bad_strict_exact_parts:{spec!r}")
    num_s, den_s = parts
    if not num_s.isdigit() or not den_s.isdigit():
        raise ValueError(f"bad_strict_exact_digits:{spec!r}")
    num = 0
    for ch in num_s:
        num = num * 10 + (ord(ch) - 48)
    den = 0
    for ch in den_s:
        den = den * 10 + (ord(ch) - 48)
    if den != expected_den:
        raise ValueError(f"strict_exact_den_mismatch:{spec!r}:expected_{expected_den}")
    return num


def check_package_binding(binding: Any) -> list[str]:
    reasons: list[str] = []
    if not is_exact_dict(binding):
        return ["package_binding_not_dict"]
    exp = REQUIRED_PACKAGE_BINDING
    got_supports = binding.get("requested_supports")
    if not is_exact_list(got_supports):
        reasons.append(f"requested_supports_not_list={got_supports!r}")
    elif not all(is_exact_str(x) for x in got_supports):
        reasons.append(f"requested_supports_not_all_str={got_supports!r}")
    elif sorted(got_supports) != sorted(exp["requested_supports"]):
        reasons.append(f"requested_supports={got_supports!r}")
    for key in (
        "replay_ce_veto",
        "pc_aux_enabled",
        "prior_batches_fed_to_bounded_steps",
        "target_parent_kl",
        "target_rows_excluded_from_pc",
    ):
        got = binding.get(key)
        if not is_exact_bool(got) or got is not exp[key]:
            reasons.append(f"{key}={got!r}")
    mode = binding.get("pc_aux_mode")
    if not is_exact_str(mode) or mode != exp["pc_aux_mode"]:
        reasons.append(f"pc_aux_mode={mode!r}")
    w = binding.get("parent_consistency_weight")
    if not is_exact_number(w) or float(w) != float(exp["parent_consistency_weight"]):
        reasons.append(f"parent_consistency_weight={w!r}")
    return reasons


def check_run_geometry(receipt: Mapping[str, Any], n: int) -> list[str]:
    reasons: list[str] = []
    device = receipt.get("device")
    if not is_exact_str(device) or device != REQUIRED_RUN_GEOMETRY["device"]:
        reasons.append(f"N{n}:device={device!r}")
    scope = receipt.get("eligible_scope")
    if not is_exact_str(scope) or scope != REQUIRED_RUN_GEOMETRY["eligible_scope"]:
        reasons.append(f"N{n}:eligible_scope={scope!r}")
    seed = receipt.get("local_selection_ordering_seed")
    if not is_exact_int(seed) or seed != REQUIRED_RUN_GEOMETRY["local_selection_ordering_seed"]:
        reasons.append(f"N{n}:local_selection_ordering_seed={seed!r}")
    mutated = receipt.get("banked_pt_mutated")
    if not is_exact_bool(mutated) or mutated is not REQUIRED_RUN_GEOMETRY["banked_pt_mutated"]:
        reasons.append(f"N{n}:banked_pt_mutated={mutated!r}")
    return reasons


def check_parent_pins(receipt: Mapping[str, Any], n: int) -> list[str]:
    reasons: list[str] = []
    unchanged = receipt.get("parent_hash_unchanged")
    if not is_exact_bool(unchanged) or unchanged is not True:
        reasons.append(f"N{n}:parent_hash_unchanged={unchanged!r}")
    before = receipt.get("parent_hash_before")
    after = receipt.get("parent_hash_after")
    if not is_exact_str(before):
        reasons.append(f"N{n}:parent_hash_before_missing_or_type={before!r}")
    elif before != PARENT_SHA_EXPECTED:
        reasons.append(f"N{n}:parent_hash_before={before!r}")
    if not is_exact_str(after):
        reasons.append(f"N{n}:parent_hash_after_missing_or_type={after!r}")
    elif after != PARENT_SHA_EXPECTED:
        reasons.append(f"N{n}:parent_hash_after={after!r}")
    return reasons


def check_per_step_global_horizon(receipt: Mapping[str, Any], n: int) -> list[str]:
    reasons: list[str] = []
    reports = receipt.get("step_reports")
    if not is_exact_dict(reports) or not reports:
        return [f"N{n}:missing_step_reports"]
    for step in range(1, n + 1):
        key = str(step)
        if key not in reports:
            reasons.append(f"N{n}:missing_step_report_{step}")
            continue
        entry = reports[key]
        if not is_exact_dict(entry):
            reasons.append(f"N{n}:step_{step}_not_dict")
            continue
        gh = entry.get("global_horizon")
        if not is_exact_int(gh) or gh != 50:
            reasons.append(f"N{n}:step_{step}_global_horizon={gh!r}")
    return reasons


def check_count_domain(l0b: Any, math_a0: Any, n: int) -> list[str]:
    reasons: list[str] = []
    for name, val, mx in (
        ("L0b", l0b, COUNT_DOMAIN_MAX["L0b"]),
        ("math_a0", math_a0, COUNT_DOMAIN_MAX["math_a0"]),
    ):
        if not is_exact_int(val):
            reasons.append(f"N{n}:{name}_not_int={val!r}")
            continue
        if val < 0 or val > mx:
            reasons.append(f"N{n}:{name}_out_of_domain={val}")
    return reasons


def extract_package_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    b2 = receipt.get("b2_retention") if is_exact_dict(receipt.get("b2_retention")) else {}
    pa = receipt.get("prior_audit") if is_exact_dict(receipt.get("prior_audit")) else {}
    return {
        "requested_supports": list(b2.get("requested_supports") or []),
        "replay_ce_veto": b2.get("replay_ce_veto"),
        "pc_aux_enabled": b2.get("pc_aux_enabled"),
        "parent_consistency_weight": b2.get("parent_consistency_weight"),
        "pc_aux_mode": b2.get("pc_aux_mode"),
        "prior_batches_fed_to_bounded_steps": b2.get(
            "prior_batches_fed_to_bounded_steps",
            pa.get("prior_batches_fed_to_bounded_steps"),
        ),
        "target_parent_kl": b2.get("target_parent_kl", pa.get("target_parent_kl")),
        "target_rows_excluded_from_pc": b2.get("target_rows_excluded_from_pc"),
    }


def extract_support_counts(receipt: Mapping[str, Any], support: str) -> int:
    pa = receipt.get("prior_audit")
    if not is_exact_dict(pa):
        raise ValueError(f"prior_audit_missing:{support}")
    deltas = pa.get("deltas")
    if not is_exact_dict(deltas):
        raise ValueError(f"deltas_missing:{support}")
    d = deltas.get(support)
    if not is_exact_dict(d):
        raise ValueError(f"support_delta_missing:{support}")
    pbvf = d.get("parent_baseline_vs_final")
    if not is_exact_dict(pbvf):
        raise ValueError(f"pbvf_missing:{support}")
    final = parse_strict_exact_count(
        pbvf.get("final_strict_exact"), expected_den=FINAL_STRICT_DEN[support]
    )
    base_spec = pbvf.get("baseline_strict_exact")
    if base_spec != BASELINE_STRICT[support]:
        raise ValueError(f"baseline_mismatch:{support}:{base_spec!r}")
    return final


def extract_final_strict_count(receipt: Mapping[str, Any], support: str) -> int:
    return extract_support_counts(receipt, support)


def admit_count_table(counts: Mapping[Any, Any]) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for n in HORIZONS:
        if n not in counts and str(n) not in counts:
            raise ValueError(f"missing counts N={n}")
        entry = counts[n] if n in counts else counts[str(n)]
        if not is_exact_dict(entry):
            raise ValueError(f"counts_N{n}_not_dict")
        l0, m = entry.get("L0b"), entry.get("math_a0")
        fails = check_count_domain(l0, m, n)
        if fails:
            raise ValueError(";".join(fails))
        out[n] = {"L0b": l0, "math_a0": m}
    return out


def build_out_authority(
    out_terminal: Mapping[str, Any], out_terminal_sha256: str
) -> dict[str, Any]:
    return {
        "sha256": out_terminal_sha256,
        "branch": out_terminal.get("branch"),
        "terminal_authority": out_terminal.get("terminal_authority"),
        "synthetic": out_terminal.get("synthetic"),
    }


def validate_source_shas_envelope(
    src: Any, *, bind_shas: Mapping[str, str] | None = None, require_out: bool = True
) -> tuple[bool, str]:
    """Mandatory authority: non-empty mapping; out pin when required. No 'if present'."""
    if not isinstance(src, Mapping):
        return False, "source_shas_not_mapping"
    if len(src) == 0:
        return False, "source_shas_empty"
    if require_out:
        if "out/terminal" not in src:
            return False, "source_shas_missing:out/terminal"
        if src.get("out/terminal") != FROZEN_OUT_TERMINAL_SHA256:
            return False, f"source_shas_out_terminal_ne_frozen:{src.get('out/terminal')!r}"
        if not is_exact_str(src.get("out/terminal")):
            return False, "source_shas_out_terminal_type"
    if bind_shas is not None:
        for k, v in bind_shas.items():
            if k not in src:
                return False, f"source_shas_missing_bind:{k}"
            if src.get(k) != v:
                return False, f"source_shas_ne_bind:{k}"
            if not is_exact_str(src.get(k)):
                return False, f"source_shas_type:{k}"
    return True, "ok"


def validate_out_authority_envelope(oa: Any) -> tuple[bool, str]:
    """Mandatory out_authority — absence fails."""
    if not isinstance(oa, Mapping):
        return False, "out_authority_missing"
    if oa.get("sha256") != FROZEN_OUT_TERMINAL_SHA256:
        return False, f"out_authority_sha={oa.get('sha256')!r}"
    if oa.get("branch") != REQUIRED_OUT_AUTHORITY["branch"]:
        return False, f"out_authority_branch={oa.get('branch')!r}"
    if oa.get("terminal_authority") != REQUIRED_OUT_AUTHORITY["terminal_authority"]:
        return False, f"out_authority_terminal_authority={oa.get('terminal_authority')!r}"
    if not is_exact_bool(oa.get("synthetic")) or oa.get("synthetic") is not False:
        return False, f"out_authority_synthetic={oa.get('synthetic')!r}"
    return True, "ok"


def validate_claim_boundary_envelope(
    receipt: Mapping[str, Any], cls: Mapping[str, Any]
) -> tuple[bool, str]:
    if "claim_boundary" not in receipt:
        return False, "top_claim_boundary_missing"
    top_b = receipt.get("claim_boundary")
    if not isinstance(top_b, Mapping):
        return False, "top_claim_boundary_not_mapping"
    top_b_d = dict(top_b)
    emb = cls.get("claim_boundary")
    if not isinstance(emb, Mapping) or top_b_d != dict(emb):
        return False, "top_claim_boundary_ne_embedded"
    if top_b_d != REQUIRED_CLAIM_BOUNDARY:
        return False, "top_claim_boundary_required_values"
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        if not is_exact_bool(top_b_d.get(k)) or top_b_d.get(k) is not v:
            return False, f"claim_boundary_type:{k}"
    return True, "ok"
