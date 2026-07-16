"""Pure reducers for fixed-state signed-utility diagnostic (PLAN v5 extraction)."""
from __future__ import annotations

import ast
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


def mutation_parity_report(base: Mapping[str, Any], prod: Mapping[str, Any], inv: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    base_keys, prod_keys, inv_keys = set(base), set(prod), set(inv)
    if base_keys != prod_keys or base_keys != inv_keys:
        raise SignedUtilityReducerError(
            f"parity_key_mismatch:base={sorted(base_keys)}:prod={sorted(prod_keys)}:inv={sorted(inv_keys)}"
        )
    ok, details = True, {}
    for key in sorted(base_keys):
        b = base[key].q_levels.to(torch.int16)
        p = prod[key].q_levels.to(torch.int16)
        i = inv[key].q_levels.to(torch.int16)
        if tuple(b.shape) != tuple(p.shape) or tuple(b.shape) != tuple(i.shape):
            raise SignedUtilityReducerError(
                f"parity_shape_mismatch:{key}:{tuple(b.shape)}!={tuple(p.shape)}!={tuple(i.shape)}"
            )
        # Flatten before changed-index / abs-delta compare (production q may be rank>=2).
        dq_p = (p - b).reshape(-1).tolist()
        dq_i = (i - b).reshape(-1).tolist()
        ch_p = [idx for idx, v in enumerate(dq_p) if v != 0]
        ch_i = [idx for idx, v in enumerate(dq_i) if v != 0]
        abs_ok = all(abs(dq_p[idx]) == abs(dq_i[idx]) for idx in range(len(dq_p)))
        ok = ok and ch_p == ch_i and abs_ok
        details[key] = {
            "changed_prod": ch_p,
            "changed_inv": ch_i,
            "abs_equal": abs_ok,
            "numel": len(dq_p),
            "shape": list(b.shape),
        }
    return {"pass": ok, "per_key": details}


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
