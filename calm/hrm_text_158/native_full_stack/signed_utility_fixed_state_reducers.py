"""Pure reducers for fixed-state signed-utility diagnostic (PLAN v5 + D2c7 compact)."""
from __future__ import annotations

import ast
import hashlib
import struct
from typing import Any, Callable, Mapping

PRIVATE_TRUSTED_CORE = "_apply_integer_vote_update_from_frozen_plan" + "_trusted"


class SignedUtilityReducerError(RuntimeError):
    pass


def epsilon_from_noop(L_noop: float) -> float:
    return 1e-7 * max(1.0, abs(float(L_noop)))


def classify_signed_utility(L_prod: float, L_inv: float, L_noop: float) -> tuple[str, float]:
    eps = epsilon_from_noop(L_noop)
    if (L_prod + eps < L_inv) and (L_prod + eps < L_noop):
        return "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN", eps
    return "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL", eps


def mean_nll_f64_from_metrics_loss(loss_pair: tuple[Any, Any]) -> tuple[float, int, float]:
    import torch

    loss_sum, local_valid_counts = loss_pair
    num = float(torch.as_tensor(loss_sum).detach().cpu().to(torch.float64).item())
    den = int(torch.as_tensor(local_valid_counts).detach().cpu().item())
    if den < 1:
        raise SignedUtilityReducerError("nll_denominator_lt_1")
    return num, den, num / float(den)


def _idx_sha(indices: list[int]) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(indices)))
    for i in sorted(indices):
        h.update(struct.pack("<q", int(i)))
    return h.hexdigest()


def _quantiles(vals: list[int]) -> dict[str, int]:
    if not vals:
        return {"p0": 0, "p50": 0, "p100": 0}
    s = sorted(int(v) for v in vals)
    mid = s[len(s) // 2]
    return {"p0": int(s[0]), "p50": int(mid), "p100": int(s[-1])}


def _carrier_parity(base: Mapping[str, Any], prod: Mapping[str, Any], inv: Mapping[str, Any], attr: str) -> dict[str, Any]:
    import torch

    ok, details = True, {}
    prod_total = inv_total = inter_total = prod_only = inv_only = 0
    all_abs = True
    for key in sorted(base):
        b = getattr(base[key], attr).to(torch.int32).reshape(-1)
        p = getattr(prod[key], attr).to(torch.int32).reshape(-1)
        i = getattr(inv[key], attr).to(torch.int32).reshape(-1)
        if tuple(b.shape) != tuple(p.shape) or tuple(b.shape) != tuple(i.shape):
            raise SignedUtilityReducerError(f"parity_shape_mismatch:{attr}:{key}")
        dq_p = (p - b).tolist()
        dq_i = (i - b).tolist()
        ch_p = [idx for idx, v in enumerate(dq_p) if int(v) != 0]
        ch_i = [idx for idx, v in enumerate(dq_i) if int(v) != 0]
        inter = sorted(set(ch_p) & set(ch_i))
        po, io = sorted(set(ch_p) - set(ch_i)), sorted(set(ch_i) - set(ch_p))
        abs_ok = all(abs(int(dq_p[idx])) == abs(int(dq_i[idx])) for idx in range(len(dq_p)))
        key_ok = ch_p == ch_i and abs_ok
        ok = ok and key_ok
        all_abs = all_abs and abs_ok
        prod_total += len(ch_p); inv_total += len(ch_i)
        inter_total += len(inter); prod_only += len(po); inv_only += len(io)
        details[key] = {
            "numel": int(b.numel()), "shape": list(getattr(base[key], attr).shape),
            "prod_changed_count": len(ch_p), "inv_changed_count": len(ch_i),
            "intersection_count": len(inter), "prod_only_count": len(po), "inv_only_count": len(io),
            "abs_equal": bool(abs_ok),
            "prod_changed_sha256": _idx_sha(ch_p), "inv_changed_sha256": _idx_sha(ch_i),
            "prod_abs_delta_quantiles": _quantiles([abs(int(dq_p[j])) for j in ch_p]),
            "inv_abs_delta_quantiles": _quantiles([abs(int(dq_i[j])) for j in ch_i]),
        }
    return {
        "pass": bool(ok),
        "per_key": details,
        "aggregate": {
            "key_count": len(base), "prod_changed_total": prod_total, "inv_changed_total": inv_total,
            "symmetric_intersection_total": inter_total, "prod_only_total": prod_only, "inv_only_total": inv_only,
            "inverse_changed_is_subset_of_production_changed_for_all_keys": inv_only == 0,
            "all_abs_equal": bool(all_abs),
        },
    }


def mutation_parity_report(base: Mapping[str, Any], prod: Mapping[str, Any], inv: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    base_keys, prod_keys, inv_keys = set(base), set(prod), set(inv)
    if base_keys != prod_keys or base_keys != inv_keys:
        raise SignedUtilityReducerError(
            f"parity_key_mismatch:base={sorted(base_keys)}:prod={sorted(prod_keys)}:inv={sorted(inv_keys)}"
        )
    for key in base_keys:
        for arm in (base, prod, inv):
            st = arm[key]
            if not hasattr(st, "q_levels") or not hasattr(st, "exact_accumulator_shadow") or not hasattr(st, "frozen_scale"):
                raise SignedUtilityReducerError(f"parity_missing_carrier:{key}")
    q_rep = _carrier_parity(base, prod, inv, "q_levels")
    acc_rep = _carrier_parity(base, prod, inv, "exact_accumulator_shadow")
    scale_ok, scale_details = True, {}
    for key in sorted(base_keys):
        b = base[key].frozen_scale.detach().cpu().contiguous()
        p = prod[key].frozen_scale.detach().cpu().contiguous()
        i = inv[key].frozen_scale.detach().cpu().contiguous()
        bh = hashlib.sha256(b.numpy().tobytes()).hexdigest()
        ph = hashlib.sha256(p.numpy().tobytes()).hexdigest()
        ih = hashlib.sha256(i.numpy().tobytes()).hexdigest()
        ok = (
            tuple(b.shape) == tuple(p.shape) == tuple(i.shape)
            and b.dtype == p.dtype == i.dtype
            and bh == ph == ih
        )
        scale_ok = scale_ok and ok
        scale_details[key] = {
            "pass": ok, "shape": list(b.shape), "dtype": str(b.dtype).replace("torch.", ""),
            "base_sha256": bh, "prod_sha256": ph, "inv_sha256": ih,
        }
    scale_rep = {"pass": bool(scale_ok), "per_key": scale_details}
    return {
        "pass": bool(q_rep["pass"] and acc_rep["pass"] and scale_rep["pass"]),
        "q_levels": q_rep,
        "exact_accumulator_shadow": acc_rep,
        "frozen_scale": scale_rep,
        "per_key": q_rep["per_key"],
    }


def static_private_core_prohibition_pass(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    banned = PRIVATE_TRUSTED_CORE
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == banned or (alias.asname or "") == banned:
                    return False
        if isinstance(node, ast.Name) and node.id == banned:
            return False
        if isinstance(node, ast.Attribute) and node.attr == banned:
            return False
    return True


def make_raw_front_c_observation_holder_observer(
    holder: list,
    call_count: list[int],
    delegate: Callable[[Mapping[str, Any]], object] | None = None,
) -> Callable[[Mapping[str, Any]], None]:
    def observer(observation: Mapping[str, Any]) -> None:
        if holder:
            raise SignedUtilityReducerError("raw_holder_second_call")
        holder.append(dict(observation))
        call_count[0] += 1
        if delegate is not None:
            delegate(observation)

    return observer


__all__ = [
    "PRIVATE_TRUSTED_CORE",
    "SignedUtilityReducerError",
    "classify_signed_utility",
    "epsilon_from_noop",
    "make_raw_front_c_observation_holder_observer",
    "mean_nll_f64_from_metrics_loss",
    "mutation_parity_report",
    "static_private_core_prohibition_pass",
]
