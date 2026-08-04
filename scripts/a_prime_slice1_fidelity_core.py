"""Pure CPU core for A′ slice1 retained-credit fidelity instruments.

No git spawn, no terminal print, no manifest IO.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

FRAC_RE = re.compile(r"^(\d+)/(\d+)$")
FINAL_BRANCHES = frozenset(
    {
        "LIVENESS_FAIL",
        "FIDELITY_COLLAPSE",
        "PAIRED_ACHIEVED_FIDELITY_AT_N",
        "INSTRUMENT_GAP",
    }
)
# wrapper/process rc → allowed receipt.branch values (operational map)
RC_BRANCH_MAP: dict[int, frozenset[str]] = {
    0: frozenset({"PAIRED_ACHIEVED_FIDELITY_AT_N", "FIDELITY_COLLAPSE"}),
    2: frozenset({"INSTRUMENT_GAP"}),
    3: frozenset({"LIVENESS_FAIL"}),
}
DEFAULT_PINNED_SUPPORTS = {
    "L0b": {"expected_count": 230, "expected_hash16": "89174273d21845bc"},
    "math_a0": {"expected_count": 1255, "expected_hash16": "56e64266357b793d"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_strict_exact_fraction(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    m = FRAC_RE.match(value.strip())
    if not m:
        return None
    c, t = int(m.group(1)), int(m.group(2))
    if t <= 0 or c < 0 or c > t:
        return None
    return c, t


def extract_prior_rates(
    receipt: Mapping[str, Any],
    *,
    pinned_supports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "source": "receipt.prior_audit.per_support.final.strict_exact",
        "schema_cite": "probe.py:build_prior_audit_receipt:4523-4591; strict_exact:3970",
        "supports": {},
        "aggregate_count": None,
        "aggregate_total": None,
        "aggregate_exact_rate": None,
        "pin_errors": [],
    }
    prior = receipt.get("prior_audit")
    if not isinstance(prior, dict) or not prior.get("enabled"):
        out["pin_errors"].append("prior_audit.missing_or_disabled")
        return out
    per = prior.get("per_support")
    if not isinstance(per, dict):
        out["pin_errors"].append("prior_audit.per_support.missing")
        return out
    proofs = prior.get("support_proofs") if isinstance(prior.get("support_proofs"), dict) else {}

    sum_c = 0
    sum_t = 0
    for name, pin in pinned_supports.items():
        node = per.get(name)
        if not isinstance(node, dict):
            out["pin_errors"].append(f"{name}:missing_per_support")
            continue
        final = node.get("final")
        if not isinstance(final, dict):
            out["pin_errors"].append(f"{name}:missing_final")
            continue
        parsed = parse_strict_exact_fraction(final.get("strict_exact"))
        if parsed is None:
            out["pin_errors"].append(
                f"{name}:strict_exact_unparseable_or_invalid:{final.get('strict_exact')!r}"
            )
            continue
        c, t = parsed
        exp_n = int(pin["expected_count"])
        exp_h = str(pin["expected_hash16"])
        got_h = node.get("support_hash16")
        got_n = node.get("support_rows_expected")
        proof = proofs.get(name) if isinstance(proofs, dict) else None
        if got_h is None and isinstance(proof, dict):
            got_h = proof.get("support_hash16")
        if got_n is None and isinstance(proof, dict):
            got_n = proof.get("expected_count")
        if got_h is None:
            out["pin_errors"].append(f"{name}:support_hash16_absent")
        elif str(got_h) != exp_h:
            out["pin_errors"].append(f"{name}:hash {got_h}!={exp_h}")
        if got_n is None:
            out["pin_errors"].append(f"{name}:support_rows_expected_absent")
        elif int(got_n) != exp_n:
            out["pin_errors"].append(f"{name}:count {got_n}!={exp_n}")
        if t != exp_n:
            out["pin_errors"].append(f"{name}:fraction_total {t}!={exp_n}")
        out["supports"][name] = {
            "strict_exact": f"{c}/{t}",
            "count": c,
            "total": t,
            "exact_rate": c / t,
            "support_hash16": got_h,
            "support_rows_expected": got_n,
        }
        sum_c += c
        sum_t += t

    if out["pin_errors"] or sum_t <= 0 or set(out["supports"]) != set(pinned_supports):
        return out
    out["ok"] = True
    out["aggregate_count"] = sum_c
    out["aggregate_total"] = sum_t
    out["aggregate_exact_rate"] = sum_c / sum_t
    return out


def extract_non_q_bpw(receipt: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("non_q_bpw", "non_q_persistent_bpw", "persistent_non_q_bpw"):
        if key in receipt and receipt[key] is not None:
            return {"status": "measured", "field": key, "value": float(receipt[key])}
    return {
        "status": "INSTRUMENT_GAP",
        "field": None,
        "value": None,
        "observed": "probe emits no non_q bpw field",
        "budget_claimable": False,
    }


def any_liveness_fail(statuses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for s in statuses:
        rc = s.get("rc")
        if s.get("timeout") or s.get("oom") or (rc is not None and int(rc) != 0):
            return s
    return None


def classify_branch(
    *,
    dense_prior: Mapping[str, Any],
    nondense_prior: Mapping[str, Any],
    delta_collapse: float,
) -> tuple[str, float | None]:
    if not dense_prior.get("ok") or not nondense_prior.get("ok"):
        return "INSTRUMENT_GAP", None
    d = float(dense_prior["aggregate_exact_rate"])
    n = float(nondense_prior["aggregate_exact_rate"])
    delta = d - n
    if delta > float(delta_collapse):
        return "FIDELITY_COLLAPSE", delta
    return "PAIRED_ACHIEVED_FIDELITY_AT_N", delta


def branch_matches_rc(branch: str, rc: int) -> bool:
    allowed = RC_BRANCH_MAP.get(int(rc))
    if allowed is None:
        return False
    return branch in allowed


def branch_rc_for(branch: str) -> int | None:
    for rc, branches in RC_BRANCH_MAP.items():
        if branch in branches:
            return rc
    return None
