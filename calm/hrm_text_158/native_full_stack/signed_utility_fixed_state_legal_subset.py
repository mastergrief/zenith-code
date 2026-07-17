from __future__ import annotations
import hashlib, json, struct
from dataclasses import replace
from typing import Any, Mapping

ESTIMAND_NAME = "full_state_legal_subset_signed_direction_fixed_state_heldout_utility"
CLAIM_CEILING = "full_state_legal_subset_signed_direction_utility_only"
SLICE_FIELDS = ("applied_indices", "applied_directions", "applied_thresholds")
PRESERVE_FIELDS = (
    "q_i16", "new_acc_i32", "candidate_indices", "pre_veto_selected_indices",
    "replay_ce_veto_indices", "replay_veto_directions", "replay_veto_thresholds",
    "pc_aux_negative_indices", "pc_aux_veto_indices",
    "event_coded_sparse_active_idx", "event_coded_sparse_post_active_i32",
)
PER_KEY_FLOOR, AGG_FLOOR, SKEW_MAX = 0.25, 0.35, 2.0
TAG_RETAINED, TAG_DROPPED, TAG_APPLIED = b"RETAINED_v1", b"DROPPED_v1", b"APPLIED_v1"
REASON_FULL_STATE = 2
MAX_COMPACT_TELEMETRY_BYTES, MAX_AUTHORITATIVE_RESULT_BYTES = 64 * 1024, 256 * 1024
_EXACT_BANNED = {
    "changed_prod", "changed_inv", "dq_prod", "dq_inv", "applied_indices", "applied_directions",
    "applied_thresholds", "candidate_indices", "pre_veto_selected_indices", "replay_ce_veto_indices",
    "replay_veto_directions", "replay_veto_thresholds", "pc_aux_negative_indices", "pc_aux_veto_indices",
}
class LegalSubsetError(RuntimeError): pass

def clamp_ternary_i16(q: int, d: int) -> int:
    import torch
    return int((torch.tensor([int(q)], dtype=torch.int16) + torch.tensor([int(d)], dtype=torch.int16)).clamp(-1, 1).item())

def clamp_acc_residual(new_acc: int, d: int, thr: int) -> int:
    import torch
    if int(d) not in (-1, 1) or not (1 <= int(thr) <= 32767):
        raise LegalSubsetError("illegal_direction_or_threshold")
    acc = torch.tensor([int(new_acc)], dtype=torch.int32)
    dd, tt = torch.tensor([int(d)], dtype=torch.int32), torch.tensor([int(thr)], dtype=torch.int32)
    return int(torch.minimum(torch.maximum(acc - dd * tt, -tt + 1), tt - 1).to(torch.int16).item())

def simulate_public_apply_at_index(*, prior_q, prior_acc, plan_q, plan_new_acc, d, thr):
    if int(plan_q) != int(prior_q): raise LegalSubsetError("plan_q_i16_not_bound_to_prior_q")
    return clamp_ternary_i16(plan_q, d), clamp_acc_residual(plan_new_acc, d, thr)

def is_full_state_bidirectionally_legal(*, prior_q, prior_acc, plan_q, plan_new_acc, d, thr) -> bool:
    qp, ap = simulate_public_apply_at_index(
        prior_q=prior_q, prior_acc=prior_acc, plan_q=plan_q, plan_new_acc=plan_new_acc, d=d, thr=thr)
    qm, am = simulate_public_apply_at_index(
        prior_q=prior_q, prior_acc=prior_acc, plan_q=plan_q, plan_new_acc=plan_new_acc, d=-d, thr=thr)
    if abs(qp - prior_q) != abs(qm - prior_q) or abs(ap - prior_acc) != abs(am - prior_acc): return False
    if (qp != prior_q) != (qm != prior_q) or (ap != prior_acc) != (am != prior_acc): return False
    return bool(qp != prior_q or ap != prior_acc)

def _pack_key(key: str) -> bytes:
    kb = key.encode("utf-8")
    if len(kb) > 65535: raise LegalSubsetError("key_utf8_overflow")
    return struct.pack("<H", len(kb)) + kb

def encode_retained_record(key, index, direction, threshold):
    if int(direction) not in (-1, 1) or not (1 <= int(threshold) <= 32767): raise LegalSubsetError("retained_field_range")
    return _pack_key(key) + struct.pack("<qbh", int(index), int(direction), int(threshold))
def encode_dropped_record(key, index, direction, reason):
    if int(direction) not in (-1, 1) or not (-128 <= int(reason) <= 127): raise LegalSubsetError("dropped_field_range")
    return _pack_key(key) + struct.pack("<qbb", int(index), int(direction), int(reason))
def encode_applied_record(key, index, direction):
    if int(direction) not in (-1, 1): raise LegalSubsetError("direction_out_of_int8")
    return _pack_key(key) + struct.pack("<qb", int(index), int(direction))

def _le_hash(tag: bytes, records: list[bytes]) -> str:
    h = hashlib.sha256(); h.update(tag); h.update(struct.pack("<Q", len(records)))
    for rec in records: h.update(rec)
    return h.hexdigest()

def _canonical_sort(records: list[tuple[bytes, int, bytes]]) -> list[bytes]:
    seen: set[tuple[bytes, int]] = set(); out: list[bytes] = []
    for kb, ix, rec in sorted(records, key=lambda t: (t[0], t[1])):
        if (kb, ix) in seen: raise LegalSubsetError("duplicate_key_index_record")
        seen.add((kb, ix)); out.append(rec)
    return out

def _slice_plan(plan: Any, keep_mask: list[bool]) -> Any:
    idx = [i for i, k in enumerate(keep_mask) if k]
    kwargs: dict[str, Any] = {n: getattr(plan, n)[idx].contiguous() for n in SLICE_FIELDS}
    for name in PRESERVE_FIELDS:
        if hasattr(plan, name): kwargs[name] = getattr(plan, name)
    if hasattr(plan, "stats"): kwargs["stats"] = dict(getattr(plan, "stats"))
    if hasattr(plan, "__dataclass_fields__"):
        return replace(plan, **{k: v for k, v in kwargs.items() if k in plan.__dataclass_fields__})
    data = dict(getattr(plan, "__dict__", {})); data.update(kwargs)
    return type(plan)(**data)

def _require_empty_replay_veto(plan: Any, key: str) -> int:
    import torch
    specs = (
        ("replay_ce_veto_indices", torch.int64),
        ("replay_veto_directions", torch.int16),
        ("replay_veto_thresholds", torch.int32),
    )
    for name, dt in specs:
        t = getattr(plan, name, None)
        if t is None: raise LegalSubsetError("replay_veto_nonempty")
        x = t.detach().cpu()
        if int(x.ndim) != 1: raise LegalSubsetError(f"plan_rank_mismatch:{name}:{key}:ndim={int(x.ndim)}")
        if x.dtype != dt: raise LegalSubsetError(f"plan_dtype_mismatch:{name}:{key}:{x.dtype}")
        if int(x.numel()) != 0: raise LegalSubsetError("replay_veto_nonempty")
    return 0

def _cpu_vec1(t: Any, dtype, name: str, key: str):
    x = t.detach().cpu()
    if int(x.ndim) != 1: raise LegalSubsetError(f"plan_rank_mismatch:{name}:{key}:ndim={int(x.ndim)}")
    if x.dtype != dtype: raise LegalSubsetError(f"plan_dtype_mismatch:{name}:{key}:{x.dtype}")
    return x

def _cpu_qacc(t: Any, dtype, shape, name: str, key: str):
    x = t.detach().cpu()
    if x.dtype != dtype: raise LegalSubsetError(f"plan_dtype_mismatch:{name}:{key}:{x.dtype}")
    if tuple(x.shape) != tuple(shape): raise LegalSubsetError(f"plan_shape_mismatch:{name}:{key}")
    return x.reshape(-1)

def _bind_preserved_index_vecs(plan: Any, key: str) -> None:
    import torch
    for name in ("candidate_indices", "pre_veto_selected_indices", "pc_aux_negative_indices", "pc_aux_veto_indices"):
        if hasattr(plan, name):
            _cpu_vec1(getattr(plan, name), torch.int64, name, key)

def characterize_plans_bidirectional_legal(prior_states: Mapping[str, Any], plans_by_key: Mapping[str, Any]):
    import torch
    if set(prior_states) != set(plans_by_key): raise LegalSubsetError("legal_subset_key_mismatch")
    retained_raw: list[tuple[bytes, int, bytes]] = []; dropped_raw: list[tuple[bytes, int, bytes]] = []
    applied_raw: list[tuple[bytes, int, bytes]] = []; per_key: dict[str, Any] = {}; boundary: dict[str, int] = {}
    fractions: list[float] = []; orig_total = retained_total = replay_veto_total = 0
    out: dict[str, Any] = {}; all_nonempty, fail_reason = True, None
    for key in sorted(plans_by_key, key=lambda s: s.encode("utf-8")):
        plan, st = plans_by_key[key], prior_states[key]
        replay_veto_total += _require_empty_replay_veto(plan, key)
        _bind_preserved_index_vecs(plan, key)
        prior_q_t = st.q_levels.detach().cpu().to(torch.int16)
        prior_acc_t = st.exact_accumulator_shadow.detach().cpu().to(torch.int32)
        plan_q = _cpu_qacc(plan.q_i16, torch.int16, prior_q_t.shape, "q_i16", key)
        plan_acc = _cpu_qacc(plan.new_acc_i32, torch.int32, prior_acc_t.shape, "new_acc_i32", key)
        prior_q, prior_acc = prior_q_t.reshape(-1), prior_acc_t.reshape(-1)
        if not bool((prior_q == plan_q).all().item()):
            raise LegalSubsetError(f"plan_q_i16_not_bound_to_prior_q:{key}")
        indices = _cpu_vec1(plan.applied_indices, torch.int64, "applied_indices", key)
        dirs = _cpu_vec1(plan.applied_directions, torch.int16, "applied_directions", key)
        thr = _cpu_vec1(plan.applied_thresholds, torch.int32, "applied_thresholds", key)
        n = int(indices.numel())
        if int(dirs.numel()) != n or int(thr.numel()) != n:
            raise LegalSubsetError(f"legal_subset_field_len_mismatch:{key}")
        keep: list[bool] = []; retained = dropped = 0; kb = key.encode("utf-8")
        for i in range(n):
            ix, d, t = int(indices[i].item()), int(dirs[i].item()), int(thr[i].item())
            if d not in (-1, 1): raise LegalSubsetError(f"direction_not_pm1:{key}:{d}")
            if t < 1 or t > 32767: raise LegalSubsetError(f"threshold_not_signed_i16_positive:{key}:{t}")
            if ix < 0 or ix >= int(prior_q.numel()): raise LegalSubsetError(f"legal_subset_index_oor:{key}:{ix}")
            pq, pa = int(prior_q[ix].item()), int(prior_acc[ix].item())
            nq, na = int(plan_q[ix].item()), int(plan_acc[ix].item())
            boundary[f"q{pq}_acc{pa}_d{d}"] = int(boundary.get(f"q{pq}_acc{pa}_d{d}", 0)) + 1
            applied_raw.append((kb, ix, encode_applied_record(key, ix, d)))
            if is_full_state_bidirectionally_legal(prior_q=pq, prior_acc=pa, plan_q=nq, plan_new_acc=na, d=d, thr=t):
                keep.append(True); retained += 1
                retained_raw.append((kb, ix, encode_retained_record(key, ix, d, t)))
            else:
                keep.append(False); dropped += 1
                dropped_raw.append((kb, ix, encode_dropped_record(key, ix, d, REASON_FULL_STATE)))
        frac = float(retained) / float(n) if n else 0.0; fractions.append(frac)
        orig_total += n; retained_total += retained
        if retained < 1: all_nonempty = False
        if frac + 1e-15 < PER_KEY_FLOOR: fail_reason = fail_reason or "legal_subset_support_degenerate:per_key"
        per_key[key] = {"original_count": n, "retained_count": retained, "dropped_count": dropped, "retained_fraction": frac}
        out[key] = _slice_plan(plan, keep)
    agg_frac = float(retained_total) / float(orig_total) if orig_total else 0.0
    if fractions and min(fractions) > 0: skew_defined, skew = True, float(max(fractions) / min(fractions))
    else: skew_defined, skew = False, None
    floors_pass, fail = True, fail_reason
    if fail: floors_pass = False
    elif agg_frac + 1e-15 < AGG_FLOOR: floors_pass, fail = False, "legal_subset_support_degenerate:aggregate_floor"
    elif skew_defined and skew is not None and skew > SKEW_MAX + 1e-15:
        floors_pass, fail = False, "legal_subset_support_degenerate:skew"
    sf = {"per_key_min": PER_KEY_FLOOR, "aggregate_min": AGG_FLOOR, "skew_max": SKEW_MAX,
          "skew_observed": skew, "skew_defined": bool(skew_defined), "pass": bool(floors_pass)}
    if fail: sf["fail_reason"] = fail
    receipt = {
        "estimand": ESTIMAND_NAME, "claim_ceiling": CLAIM_CEILING,
        "original_applied_total": int(orig_total), "retained_total": int(retained_total),
        "dropped_total": int(orig_total - retained_total), "aggregate_retained_fraction": float(agg_frac),
        "all_keys_nonempty": bool(all_nonempty), "replay_veto_total": int(replay_veto_total), "per_key": per_key,
        "boundary_q_acc_by_direction_counts": {k: int(v) for k, v in sorted(boundary.items())},
        "retained_stream_sha256": _le_hash(TAG_RETAINED, _canonical_sort(retained_raw)),
        "dropped_stream_sha256": _le_hash(TAG_DROPPED, _canonical_sort(dropped_raw)),
        "applied_plan_index_direction_sha256": _le_hash(TAG_APPLIED, _canonical_sort(applied_raw)),
        "support_floors": sf,
    }
    return out, receipt

def enforce_legal_subset_support_floors(receipt: Mapping[str, Any]) -> None:
    floors = receipt.get("support_floors") if isinstance(receipt, Mapping) else None
    if not isinstance(floors, Mapping) or floors.get("pass") is not True:
        raise LegalSubsetError((floors.get("fail_reason") if isinstance(floors, Mapping) else None)
                               or "legal_subset_support_degenerate")

def filter_plans_bidirectional_legal(prior_states, plans_by_key):
    out, receipt = characterize_plans_bidirectional_legal(prior_states, plans_by_key)
    enforce_legal_subset_support_floors(receipt); return out, receipt

def assert_compact_json_nbytes(obj: Mapping[str, Any], *, limit: int, label: str) -> int:
    n = len(json.dumps(obj, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    if n > int(limit): raise LegalSubsetError(f"compact_artifact_overflow:{label}:{n}>{limit}")
    return n

def payload_has_raw_index_arrays(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(v, (list, tuple)):
                if k in _EXACT_BANNED or k.endswith(("_indices", "_directions", "_thresholds")): return True
                if k.startswith("dq_") or k in {"changed_prod", "changed_inv"}: return True
            if payload_has_raw_index_arrays(v): return True
    elif isinstance(obj, list):
        return any(payload_has_raw_index_arrays(x) for x in obj)
    return False

__all__ = [
    "AGG_FLOOR", "CLAIM_CEILING", "ESTIMAND_NAME", "LegalSubsetError", "MAX_AUTHORITATIVE_RESULT_BYTES",
    "MAX_COMPACT_TELEMETRY_BYTES", "PER_KEY_FLOOR", "PRESERVE_FIELDS", "REASON_FULL_STATE", "SKEW_MAX",
    "SLICE_FIELDS", "TAG_APPLIED", "TAG_DROPPED", "TAG_RETAINED", "assert_compact_json_nbytes",
    "characterize_plans_bidirectional_legal", "clamp_acc_residual", "clamp_ternary_i16",
    "encode_applied_record", "encode_dropped_record", "encode_retained_record",
    "enforce_legal_subset_support_floors", "filter_plans_bidirectional_legal",
    "is_full_state_bidirectionally_legal", "payload_has_raw_index_arrays", "simulate_public_apply_at_index",
]
