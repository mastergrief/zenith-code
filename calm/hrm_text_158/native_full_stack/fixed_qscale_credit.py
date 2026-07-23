"""Fixed-qscale credit seam for the forgetting-mechanism screen (PLAN_v9).

Extracted behavior-preservingly from forgetting_mechanism_screen_reducers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CreditGradStore:
    """Step-scoped credit_grads + route counters (PLAN_v9 lifecycle)."""

    credit_grads: dict[str, torch.Tensor] = field(default_factory=dict)
    n_fixed_qscale_forwards: int = 0
    n_bitlinear_dynamic_forwards: int = 0
    n_eligible_keys: int = 0
    n_credit_grads_present: int = 0
    begun: bool = False
    # Ephemeral scalar leaf so FixedQScale stays on the autograd graph when HRM
    # activations are detached (carry detach). NOT a Parameter; not optimized;
    # recreated each begin_credit_step (no persistent FP trainable state).
    graph_anchor: Optional[torch.Tensor] = None

    def begin_credit_step(self, eligible_names: Sequence[str] | None = None) -> None:
        self.credit_grads.clear()
        self.n_fixed_qscale_forwards = 0
        self.n_bitlinear_dynamic_forwards = 0
        self.n_eligible_keys = int(len(eligible_names) if eligible_names is not None else 0)
        self.n_credit_grads_present = 0
        self.begun = True
        self.graph_anchor = torch.zeros((), dtype=torch.float32, requires_grad=True)

    def require_begun(self) -> None:
        if not self.begun:
            raise RuntimeError(
                "begin_credit_step() was not called before FixedQScaleLinearWithCredit "
                "(fail-closed non-carry invariant)"
            )

    def add_dW(self, name: str, dW: torch.Tensor) -> None:
        self.require_begun()
        if name not in self.credit_grads:
            self.credit_grads[name] = dW.detach().clone()
        else:
            self.credit_grads[name] = self.credit_grads[name] + dW.detach()

    def snapshot_and_mark(self) -> dict[str, torch.Tensor]:
        self.require_begun()
        out = {k: v.detach().clone() for k, v in self.credit_grads.items()}
        self.n_credit_grads_present = len(out)
        return out

    def assert_route_completeness(self, eligible_names: Sequence[str]) -> None:
        missing = [n for n in eligible_names if n not in self.credit_grads]
        if missing:
            raise RuntimeError(f"route-completeness: missing credit_grads for {missing[:5]}")
        if int(self.n_bitlinear_dynamic_forwards) != 0:
            raise RuntimeError(
                f"route-completeness: n_bitlinear_dynamic_forwards="
                f"{self.n_bitlinear_dynamic_forwards} (must be 0)"
            )
        for n, g in self.credit_grads.items():
            if not torch.isfinite(g).all():
                raise RuntimeError(f"credit_grads[{n}] has non-finite values")
            if not bool((g != 0).any()):
                raise RuntimeError(f"credit_grads[{n}] is all-zero (nonzero required)")


_GLOBAL_CREDIT_STORE = CreditGradStore()


def get_credit_store() -> CreditGradStore:
    return _GLOBAL_CREDIT_STORE


def begin_credit_step(eligible_names: Sequence[str] | None = None) -> CreditGradStore:
    store = get_credit_store()
    store.begin_credit_step(eligible_names)
    return store


def flattened_nd_dW(grad_output: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """PLAN_v9: dW = grad_output.reshape(-1,out).T @ act.reshape(-1,in)."""
    if grad_output.shape[:-1] != act.shape[:-1]:
        raise ValueError(
            f"grad_output/act leading dims mismatch: {tuple(grad_output.shape)} vs "
            f"{tuple(act.shape)}"
        )
    out_features = int(grad_output.shape[-1])
    in_features = int(act.shape[-1])
    go = grad_output.reshape(-1, out_features)
    ac = act.reshape(-1, in_features)
    return go.T @ ac


class FixedQScaleLinearWithCredit(torch.autograd.Function):
    """Forward == qscale_linear_reference; backward accumulates flattened ND dW.

    ``credit_anchor`` is a step-local requires_grad scalar (not a Parameter) so
    the Function stays connected when HRM activations are carry-detached — the
    regime where BitLinear still got weight.grad via its Parameter leaf but a
    pure q*scale Function would otherwise never see backward.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        q_levels: torch.Tensor,
        frozen_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        credit_anchor: torch.Tensor,
        name: str,
        store_token: int,
    ) -> torch.Tensor:
        store = get_credit_store()
        store.require_begun()
        store.n_fixed_qscale_forwards += 1
        q_f = q_levels.to(dtype=torch.float32)
        scale = frozen_scale.to(dtype=torch.float32).reshape(())
        weight = q_f * scale
        ctx.save_for_backward(input, q_f, scale, bias if bias is not None else torch.tensor([]))
        ctx.has_bias = bias is not None
        ctx.name = str(name)
        ctx.store_token = int(store_token)
        out = F.linear(input, weight, bias)
        # Keep grad_fn alive when act.requires_grad is False (HRM carry detach).
        anchor = credit_anchor.to(device=out.device, dtype=out.dtype)
        return out + anchor * 0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, q_f, scale, bias_or_empty = ctx.saved_tensors
        name = ctx.name
        store = get_credit_store()
        store.require_begun()
        dW = flattened_nd_dW(grad_output, input)
        store.add_dW(name, dW)
        # grad w.r.t. input through effective weight
        weight = q_f * scale
        grad_input = grad_output @ weight
        grad_q = None  # q_levels not a leaf requiring grad
        grad_scale = None
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
        # credit_anchor grad unused (None); name/store_token non-tensors
        return grad_input, grad_q, grad_scale, grad_bias, None, None, None


def fixed_qscale_linear_with_credit(
    input: torch.Tensor,
    q_levels: torch.Tensor,
    frozen_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    name: str,
) -> torch.Tensor:
    store = get_credit_store()
    store.require_begun()
    if store.graph_anchor is None:
        store.graph_anchor = torch.zeros((), dtype=torch.float32, requires_grad=True)
    return FixedQScaleLinearWithCredit.apply(
        input,
        q_levels,
        frozen_scale,
        bias,
        store.graph_anchor,
        name,
        id(store),
    )


def qscale_reference_weight(q_levels: torch.Tensor, frozen_scale: torch.Tensor) -> torch.Tensor:
    return q_levels.to(torch.float32) * frozen_scale.to(torch.float32).reshape(())



def bitlinear_absmean_quantize(
    master: torch.Tensor, *, scale_eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = master.abs().mean().clamp(min=scale_eps)
    w_q = (master / scale).round().clamp(-1.0, 1.0)
    return w_q * scale, scale


def mechanical_dynamic_scale_diverges(
    q: torch.Tensor,
    frozen_scale: float,
    x: torch.Tensor,
) -> dict[str, Any]:
    """Prove BitLinear(q*s) differs from F.linear(x, q*s) by construction."""
    s = float(frozen_scale)
    W_fixed = q.to(torch.float32) * s
    W_dyn, scale_dyn = bitlinear_absmean_quantize(W_fixed)
    y_fixed = F.linear(x, W_fixed)
    y_dyn = F.linear(x, W_dyn)
    return {
        "scale_dyn": float(scale_dyn.item()),
        "frozen_scale": s,
        "scale_equal": bool(torch.equal(scale_dyn.cpu(), torch.tensor(s))),
        "W_allclose": bool(torch.allclose(W_dyn, W_fixed)),
        "Y_allclose": bool(torch.allclose(y_dyn, y_fixed)),
        "diverges": (
            not torch.equal(scale_dyn.cpu(), torch.tensor(s))
            and not torch.allclose(W_dyn, W_fixed)
            and not torch.allclose(y_dyn, y_fixed)
        ),
    }


def cumulative_q_transitions(
    q_before: torch.Tensor, q_after: torch.Tensor, applied_mask: torch.Tensor
) -> int:
    """Count transitions on applied indices (for cumulative counter)."""
    return int((q_before[applied_mask] != q_after[applied_mask]).sum().item())


def snapshot_route_counters(store: CreditGradStore) -> dict[str, int]:
    """Capture credit-step route counters (call BEFORE any probe begin_credit_step)."""
    return {
        "n_fixed_qscale_forwards": int(store.n_fixed_qscale_forwards),
        "n_bitlinear_dynamic_forwards": int(store.n_bitlinear_dynamic_forwards),
        "n_eligible_keys": int(store.n_eligible_keys),
        "n_credit_grads_present": int(store.n_credit_grads_present),
    }
