"""RO production-outcome binding from builder receipts (IMPLEMENT_v9).

No TSA edits. Exact hash-VALUE binding for post-q AND logical-int16 acc when
RO-observable; fail-closed when production does not emit logical-acc sha.
B2/B3 require production payload VALUE == twin canonical sidecar sha
(authoritative_sidecar_payload_sha256). Mutation-only / existence-only
cannot pass. B3 also binds WG capture VALUES.
"""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.lands_ab_eval_authoritative_payload import (  # noqa: F401
    authoritative_sidecar_payload_sha256,
    eligible_weight_state_keys_from_state_keys,
)


def _sha64(v: Any) -> str | None:
    if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
        return v
    return None



def extract_local_update_binding(receipt: Any) -> dict[str, Any]:
    """B1 local-update RO observables: post-q + logical-acc (when present).

    Logical-acc sources (RO only, no TSA edit):
      1. exact_local_parity_proof_by_key[*].candidate_bounded_decode_sha256_after
         (oracle_on parity path)
      2. candidate_local_update_proof_by_key[*].candidate_bounded_decode_sha256_after
         (when dense-oracle control filled it)
    Fused-only path typically has q only → logical-acc map empty → bind fails closed.
    """
    proof = dict(getattr(receipt, "candidate_step_summary", None) or {})
    by_key = dict(proof.get("candidate_local_update_proof_by_key") or {})
    parity = dict(getattr(receipt, "exact_local_parity_proof_by_key", None) or {})
    post_q: dict[str, str] = {}
    post_acc: dict[str, str] = {}
    applied_ids: dict[str, str] = {}
    q_changed_ids: dict[str, str] = {}
    for k, p in sorted(by_key.items()):
        if not isinstance(p, Mapping):
            continue
        ks = str(k)
        cq = _sha64(p.get("candidate_q_sha256_after"))
        if cq:
            post_q[ks] = cq
        # fused proof rarely has decode; still try
        ca = _sha64(p.get("candidate_bounded_decode_sha256_after"))
        if ca:
            post_acc[ks] = ca
        aid = _sha64(
            p.get("applied_row_identities_sha256")
            or p.get("ordered_applied_row_identities_sha256")
        )
        if aid:
            applied_ids[ks] = aid
        qid = _sha64(p.get("q_changed_identities_sha256"))
        if qid:
            q_changed_ids[ks] = qid
    # parity path (oracle_on) is the reliable logical-acc RO surface
    for k, p in sorted(parity.items()):
        if not isinstance(p, Mapping):
            continue
        ks = str(k)
        cq = _sha64(p.get("candidate_q_sha256_after"))
        if cq:
            post_q[ks] = cq
        ca = _sha64(p.get("candidate_bounded_decode_sha256_after")) or _sha64(
            p.get("oracle_acc_sha256_after")
        )
        if ca:
            post_acc[ks] = ca
    return {
        "builder_receipt_pass": bool(getattr(receipt, "pass_receipt", False)),
        "total_sparse_vote_event_count": int(
            getattr(receipt, "total_sparse_vote_event_count", -1)
        ),
        "q_changed_count": int(getattr(receipt, "q_changed_count", -1)),
        "production_post_q_sha256_by_key": post_q,
        "production_post_logical_acc_sha256_by_key": post_acc,
        "production_applied_row_identities_sha256_by_key": applied_ids,
        "production_q_changed_identities_sha256_by_key": q_changed_ids,
        "sparse_vote_authority_mode": str(
            (getattr(receipt, "vote_projection_proof", None) or {}).get(
                "sparse_vote_authority_mode", ""
            )
        ),
        "exact_local_parity_proof_by_key": parity,
        "logical_acc_ro_observable": bool(post_acc) and set(post_acc.keys()) == set(post_q.keys())
        if post_q
        else False,
    }


def extract_roundtrip_binding(receipt: Any) -> dict[str, Any]:
    """B2 roundtrip RO observables — payload VALUE hashes (not mere change)."""
    proof = dict(getattr(receipt, "post_resume_update_proof", None) or {})
    cps = dict(getattr(receipt, "checkpoint_payload_summary", None) or {})
    pre = _sha64(cps.get("authoritative_state_payload_sha256")) or ""
    post = _sha64(cps.get("post_update_authoritative_state_payload_sha256")) or ""
    return {
        "builder_receipt_pass": bool(getattr(receipt, "pass_receipt", False))
        and bool(proof.get("sparse_vote_authority_only") is True)
        and bool(getattr(receipt, "post_resume_update_mutated_resumed_sub2_authority", False)),
        "total_sparse_vote_event_count": int(proof.get("total_sparse_vote_event_count", -1)),
        "q_changed_count": int(proof.get("q_changed_count", -1)),
        "post_update_authoritative_state_payload_sha256": post,
        "pre_update_authoritative_state_payload_sha256": pre,
        # retained for diagnostics only — NOT a match criterion
        "post_update_payload_changed": bool(pre and post and pre != post),
        "sparse_vote_authority_mode": str(proof.get("sparse_vote_authority_mode") or ""),
        "post_resume_update_proof": proof,
    }


def extract_landing_binding(landing: Any) -> dict[str, Any]:
    """B3 landing RO observables — p1b + core identity + WG VALUE map.

    Landing evidence comes from p1b / sparse_vote_authority_subproof /
    core_execution_identity only — no generic fallback laundering.
    """
    p1b = getattr(landing, "p1b_live_conversion_receipt", None)
    core = dict(getattr(landing, "core_execution_identity", None) or {})
    sub = dict(getattr(landing, "sparse_vote_authority_subproof", None) or {})
    p1b_pass = bool(getattr(p1b, "pass_receipt", False)) if p1b is not None else False
    wg_raw = dict(core.get("weighted_grad_capture_sha256_by_key") or {})
    wg: dict[str, str] = {}
    for k, v in wg_raw.items():
        s = _sha64(v)
        if s:
            wg[str(k)] = s
    return {
        "builder_receipt_pass": p1b_pass,
        "p1b_pass_receipt": p1b_pass,
        "total_sparse_vote_event_count": int(
            getattr(p1b, "total_sparse_vote_event_count", -1) if p1b is not None else -1
        ),
        "q_changed_count": int(getattr(p1b, "q_changed_count", -1) if p1b is not None else -1),
        "post_update_payload_sha256": _sha64(core.get("post_update_payload_sha256")) or "",
        "weighted_grad_capture_sha256_by_key": wg,
        "path_resolved_mode": str(core.get("path_resolved_mode") or ""),
        "sparse_vote_authority_mode": str(sub.get("sparse_vote_authority_mode") or ""),
        "oracle_only": dict(sub.get("oracle_only") or {}),
        "slice_readiness_claim": bool(getattr(landing, "slice_readiness_claim", True)),
    }


def twin_sparse_post_q_by_key(compare: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in dict(compare.get("post_q_sha256_by_key") or {}).items():
        if isinstance(v, Mapping) and isinstance(v.get("sparse"), str):
            s = _sha64(v["sparse"])
            if s:
                out[str(k)] = s
    return out


def twin_sparse_post_logical_acc_by_key(compare: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in dict(compare.get("post_logical_acc_sha256_by_key") or {}).items():
        if isinstance(v, Mapping) and isinstance(v.get("sparse"), str):
            s = _sha64(v["sparse"])
            if s:
                out[str(k)] = s
    return out


def bind_production_to_twin_local_update(
    *,
    production: Mapping[str, Any],
    compare: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-key production post-q AND logical-acc VALUES vs twin sparse.

    Fail-closed when production logical-acc is missing (fused path without
    decode sha). Count-only equality never passes. Q-only match without
    logical-acc VALUE equality → production_sparse_matches_twin False.
    """
    twin_q = twin_sparse_post_q_by_key(compare)
    twin_acc = twin_sparse_post_logical_acc_by_key(compare)
    prod_q = dict(production.get("production_post_q_sha256_by_key") or {})
    prod_acc = dict(production.get("production_post_logical_acc_sha256_by_key") or {})

    if not twin_q or not twin_acc:
        return {
            "production_sparse_matches_twin": False,
            "reason": "twin_missing_post_q_or_logical_acc",
            "per_key_q_equal": {},
            "per_key_logical_acc_equal": {},
            "hash_ok": False,
            "logical_acc_ok": False,
            "counts_ok": False,
            "identity_ok": False,
        }
    if set(prod_q.keys()) != set(twin_q.keys()) or not prod_q:
        return {
            "production_sparse_matches_twin": False,
            "reason": "post_q_key_set_mismatch_or_empty",
            "per_key_q_equal": {},
            "per_key_logical_acc_equal": {},
            "hash_ok": False,
            "logical_acc_ok": False,
            "counts_ok": False,
            "identity_ok": False,
            "production_post_q_sha256_by_key": prod_q,
            "production_post_logical_acc_sha256_by_key": prod_acc,
            "twin_sparse_post_q_sha256_by_key": twin_q,
            "twin_sparse_post_logical_acc_sha256_by_key": twin_acc,
        }

    # logical-acc must be RO-observable on production for every key
    if set(prod_acc.keys()) != set(prod_q.keys()) or not prod_acc:
        return {
            "production_sparse_matches_twin": False,
            "reason": "production_logical_acc_not_ro_observable",
            "per_key_q_equal": {k: prod_q[k] == twin_q[k] for k in sorted(prod_q)},
            "per_key_logical_acc_equal": {},
            "hash_ok": False,
            "logical_acc_ok": False,
            "counts_ok": False,
            "identity_ok": False,
            "production_post_q_sha256_by_key": prod_q,
            "production_post_logical_acc_sha256_by_key": prod_acc,
            "twin_sparse_post_q_sha256_by_key": twin_q,
            "twin_sparse_post_logical_acc_sha256_by_key": twin_acc,
        }
    if set(prod_acc.keys()) != set(twin_acc.keys()):
        return {
            "production_sparse_matches_twin": False,
            "reason": "logical_acc_key_set_mismatch",
            "per_key_q_equal": {k: prod_q[k] == twin_q[k] for k in sorted(prod_q)},
            "per_key_logical_acc_equal": {},
            "hash_ok": False,
            "logical_acc_ok": False,
            "counts_ok": False,
            "identity_ok": False,
            "production_post_q_sha256_by_key": prod_q,
            "production_post_logical_acc_sha256_by_key": prod_acc,
            "twin_sparse_post_q_sha256_by_key": twin_q,
            "twin_sparse_post_logical_acc_sha256_by_key": twin_acc,
        }

    per_q = {k: prod_q[k] == twin_q[k] for k in sorted(prod_q)}
    per_acc = {k: prod_acc[k] == twin_acc[k] for k in sorted(prod_acc)}
    counts_ok = int(production.get("total_sparse_vote_event_count", -1)) == int(
        compare.get("sparse_event_count", -2)
    ) and int(production.get("q_changed_count", -1)) == int(
        compare.get("q_changed_count_sparse", -2)
    )
    hash_ok = all(per_q.values()) and len(per_q) > 0
    logical_acc_ok = all(per_acc.values()) and len(per_acc) > 0
    applied = dict(production.get("production_applied_row_identities_sha256_by_key") or {})
    identity_ok = int(production.get("q_changed_count", 0)) == 0 or (
        len(applied) == len(prod_q)
        and all(isinstance(v, str) and len(v) == 64 for v in applied.values())
    )
    return {
        "production_sparse_matches_twin": bool(
            production.get("builder_receipt_pass") is True
            and hash_ok
            and logical_acc_ok
            and counts_ok
            and identity_ok
        ),
        "per_key_q_equal": per_q,
        "per_key_logical_acc_equal": per_acc,
        "counts_ok": counts_ok,
        "hash_ok": hash_ok,
        "logical_acc_ok": logical_acc_ok,
        "identity_ok": identity_ok,
        "production_post_q_sha256_by_key": prod_q,
        "production_post_logical_acc_sha256_by_key": prod_acc,
        "twin_sparse_post_q_sha256_by_key": twin_q,
        "twin_sparse_post_logical_acc_sha256_by_key": twin_acc,
        "production_applied_row_identities_sha256_by_key": applied,
    }


def bind_production_to_twin_roundtrip(
    *,
    production: Mapping[str, Any],
    compare: Mapping[str, Any],
) -> dict[str, Any]:
    """B2: production post payload VALUE must equal twin canonical sidecar sha.

    Live branch is always production_post_equals_twin_post. Twin must emit
    twin_post_authoritative_state_payload_sha256 (canonical TSA serializer on
    twin post states). Mutation-only / bare-changed / arbitrary distinct pre/post
    without twin equality CANNOT pass.
    """
    counts_ok = int(production.get("total_sparse_vote_event_count", -1)) == int(
        compare.get("sparse_event_count", -2)
    )
    pre = _sha64(production.get("pre_update_authoritative_state_payload_sha256"))
    post = _sha64(production.get("post_update_authoritative_state_payload_sha256"))
    twin_post = _sha64(compare.get("twin_post_authoritative_state_payload_sha256"))
    payload_values_present = pre is not None and post is not None
    bare_changed = bool(production.get("post_update_payload_changed")) and not payload_values_present
    if twin_post is None:
        payload_value_ok = False
        payload_eq_mode = "twin_post_missing"
        reason = "twin_post_authoritative_state_payload_sha256_required"
    elif post is None:
        payload_value_ok = False
        payload_eq_mode = "production_post_equals_twin_post"
        reason = "production_post_payload_missing"
    else:
        payload_value_ok = post == twin_post
        payload_eq_mode = "production_post_equals_twin_post"
        reason = "payload_values_equal" if payload_value_ok else "payload_value_mismatch"
    return {
        "production_sparse_matches_twin": bool(
            production.get("builder_receipt_pass") is True
            and counts_ok
            and payload_value_ok
            and not bare_changed
        ),
        "counts_ok": counts_ok,
        "payload_value_ok": payload_value_ok,
        "payload_values_present": payload_values_present,
        "payload_eq_mode": payload_eq_mode,
        "reason": reason,
        "pre_update_authoritative_state_payload_sha256": pre or "",
        "post_update_authoritative_state_payload_sha256": post or "",
        "twin_post_authoritative_state_payload_sha256": twin_post or "",
        "bare_changed_without_values_rejected": bare_changed,
    }


def bind_production_to_twin_landing(
    *,
    production: Mapping[str, Any],
    compare: Mapping[str, Any],
    capture_wg_sha_by_key: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """B3: p1b.pass_receipt + post_update_payload VALUE==twin sidecar + WG VALUES.

    post_update_payload_sha256 (production p1b post-resume sidecar) must equal
    compare.twin_post_authoritative_state_payload_sha256. Existence-only
    (64-char present) is not sufficient. WG key-set-only match rejected.
    """
    p1b_ok = bool(production.get("p1b_pass_receipt") is True)
    payload = _sha64(production.get("post_update_payload_sha256"))
    twin_post = _sha64(compare.get("twin_post_authoritative_state_payload_sha256"))
    if payload is None or twin_post is None:
        payload_ok = False
        payload_reason = (
            "twin_post_missing" if twin_post is None else "production_post_payload_missing"
        )
    else:
        payload_ok = payload == twin_post
        payload_reason = "payload_values_equal" if payload_ok else "payload_value_mismatch"
    counts_ok = int(production.get("total_sparse_vote_event_count", -1)) == int(
        compare.get("sparse_event_count", -2)
    ) and int(production.get("q_changed_count", -1)) == int(
        compare.get("q_changed_count_sparse", -2)
    )
    wg_prod = {
        str(k): s
        for k, v in dict(production.get("weighted_grad_capture_sha256_by_key") or {}).items()
        if (s := _sha64(v))
    }
    wg_cap = {
        str(k): s
        for k, v in dict(capture_wg_sha_by_key or {}).items()
        if (s := _sha64(v))
    }
    if not wg_prod or not wg_cap or set(wg_prod.keys()) != set(wg_cap.keys()):
        wg_ok = False
        per_wg: dict[str, bool] = {}
        wg_reason = "wg_key_set_mismatch_or_empty"
    else:
        per_wg = {k: wg_prod[k] == wg_cap[k] for k in sorted(wg_prod)}
        wg_ok = all(per_wg.values())
        wg_reason = "wg_values_equal" if wg_ok else "wg_value_mismatch"
    return {
        "production_sparse_matches_twin": bool(
            p1b_ok and payload_ok and counts_ok and wg_ok
        ),
        "p1b_ok": p1b_ok,
        "payload_ok": payload_ok,
        "payload_reason": payload_reason,
        "counts_ok": counts_ok,
        "wg_ok": wg_ok,
        "wg_reason": wg_reason,
        "per_key_wg_equal": per_wg,
        "post_update_payload_sha256": payload or "",
        "twin_post_authoritative_state_payload_sha256": twin_post or "",
        "production_weighted_grad_capture_sha256_by_key": wg_prod,
        "capture_weighted_grad_capture_sha256_by_key": wg_cap,
    }


def extract_oracle_from_builder_receipt(receipt: Any) -> dict[str, Any]:
    """Pull oracle_only / mode from named builder when ORACLE_ON.

    Landing: sparse_vote_authority_subproof only (no generic fallback).
    Roundtrip: post_resume_update_proof.oracle_only.
    Local update: vote_projection_proof.
    """
    vpp = dict(getattr(receipt, "vote_projection_proof", None) or {})
    mode = str(vpp.get("sparse_vote_authority_mode") or "")
    oracle = dict(vpp.get("oracle_only") or {})
    builder_pass = bool(getattr(receipt, "pass_receipt", False))

    # roundtrip
    proof = dict(getattr(receipt, "post_resume_update_proof", None) or {})
    if proof.get("oracle_only") or proof.get("sparse_vote_authority_mode"):
        if proof.get("oracle_only"):
            oracle = dict(proof.get("oracle_only") or {})
        mode = str(proof.get("sparse_vote_authority_mode") or mode)
        if "pass" in proof:
            builder_pass = builder_pass and bool(proof.get("sparse_vote_authority_only") is True)

    # landing envelope — subproof is the named evidence surface
    sub = dict(getattr(receipt, "sparse_vote_authority_subproof", None) or {})
    if sub:
        mode = str(sub.get("sparse_vote_authority_mode") or mode)
        if sub.get("oracle_only"):
            oracle = dict(sub.get("oracle_only") or {})
        p1b = getattr(receipt, "p1b_live_conversion_receipt", None)
        if p1b is not None:
            builder_pass = bool(getattr(p1b, "pass_receipt", False))

    return {
        "resolved_mode": mode,
        "oracle_only": oracle,
        "builder_receipt_pass": builder_pass,
        "events_equal_by_key": {
            str(k): bool(v)
            for k, v in dict(oracle.get("events_equal_by_key") or {}).items()
        },
        "events_equal_fused_vs_dense_derived": bool(
            oracle.get("events_equal_fused_vs_dense_derived")
        )
        if "events_equal_fused_vs_dense_derived" in oracle
        else None,
    }
