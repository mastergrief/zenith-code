"""Phase-1 closed production adapter: wrap real A-RK; mint only after normal return."""
from __future__ import annotations

import inspect
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack import forgotten_accum_a_ledger_accounting_v2 as acct
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ark_invoke import (
    invoke_arm_with_a_rk,
)

_ARM_ORDER = ("U", "E", "R0", "RW")
_ARK_SIG = inspect.signature(invoke_arm_with_a_rk)
_PROBE_SEAM_BOX = "_forgotten_accum_acct_v2_phase1_probe_seam_box"


def _probe_seam_lock() -> threading.RLock:
    """Process-sticky RLock shared across adapter reload generations."""
    box = sys.modules.setdefault(_PROBE_SEAM_BOX, {"lock": threading.RLock()})
    return box["lock"]


class AdapterAuthorityRefuse(RuntimeError):
    """Fail-closed adapter refuse (geometry/keyset/receipt cardinality)."""


@dataclass(frozen=True)
class FourArmARkCallInputs:
    runner: Any
    model: Any
    batch: Any
    device: Any
    eligible: Any
    runner_contract: Any
    rk: Mapping[str, Any]
    states_by_arm: Mapping[str, Any]
    hook_by_arm: Mapping[str, Any]
    backlog_by_arm: Mapping[str, Any]
    flip_by_arm: Mapping[str, Any]
    schedule_by_arm: Mapping[str, Any]


def _exact_arm_maps(*maps: Mapping[str, Any]) -> None:
    need = set(_ARM_ORDER)
    for m in maps:
        if not isinstance(m, Mapping) or set(m) != need:
            raise AdapterAuthorityRefuse("FOUR_ARM_INPUT_KEYSET_MISMATCH")


def _build_run_closed():
    scope_stack: ContextVar[list[dict[str, Any]] | None] = ContextVar(
        "acct_v2_phase1_scope_stack", default=None,
    )

    def _top() -> dict[str, Any] | None:
        stack = scope_stack.get()
        return stack[-1] if stack else None

    def probe(cap: Any, *, arm_id: str) -> bool:
        scope = _top()
        return scope is not None and scope.get(str(arm_id)) is cap

    def register(arm_id: str, cap: Any) -> None:
        scope = _top()
        if scope is None:
            raise AdapterAuthorityRefuse("NO_LIVE_SCOPE_FOR_REGISTER")
        scope[str(arm_id)] = cap

    def clear_top() -> None:
        scope = _top()
        if scope is not None:
            scope.clear()

    def mint():
        return object.__new__(acct.TrustedNormalSuccessCapability)

    def run_closed_four_arm_accounting_v2_cpu(
        *,
        t_cut: Any,
        runway_steps: Any,
        rewarm_window_steps: Any,
        calls: FourArmARkCallInputs,
    ) -> dict[str, Any]:
        need = (
            "runner", "model", "batch", "device", "eligible", "runner_contract",
            "rk", "states_by_arm", "hook_by_arm", "backlog_by_arm", "flip_by_arm",
            "schedule_by_arm",
        )
        if any(not hasattr(calls, name) for name in need):
            raise AdapterAuthorityRefuse("CALLS_NOT_FOUR_ARM_INPUTS")
        created = False
        stack = scope_stack.get()
        if stack is None:
            stack = []
            scope_stack.set(stack)
            created = True
        stack.append({})
        try:
            geometry, envelopes, caps = _run_arms(
                t_cut=t_cut, runway_steps=runway_steps,
                rewarm_window_steps=rewarm_window_steps, calls=calls,
                register=register, mint=mint,
            )
            lock = _probe_seam_lock()
            lock.acquire()
            try:
                prev_ok = acct._capability_ok
                acct._capability_ok = probe
                try:
                    result = acct.classify_four_arm_ordered_event_summaries(
                        geometry=geometry,
                        arm_summary_payloads=MappingProxyType(envelopes),
                        trusted_capabilities=MappingProxyType(caps),
                    )
                    return result.as_dict()
                finally:
                    acct._capability_ok = prev_ok
            finally:
                lock.release()
        finally:
            clear_top()
            if stack:
                stack.pop()
            if created:
                scope_stack.set(None)

    return run_closed_four_arm_accounting_v2_cpu


def _run_arms(*, t_cut, runway_steps, rewarm_window_steps, calls, register, mint):
    _exact_arm_maps(
        calls.states_by_arm, calls.hook_by_arm, calls.backlog_by_arm,
        calls.flip_by_arm, calls.schedule_by_arm,
    )
    geometry = acct.build_independent_expected_geometry(
        t_cut=t_cut, runway_steps=runway_steps,
        rewarm_window_steps=rewarm_window_steps,
    )
    if isinstance(geometry, acct.AccountingV2Result):
        raise AdapterAuthorityRefuse(f"GEOMETRY_REFUSED:{geometry.reason}")
    envelopes: dict[str, Any] = {}
    caps: dict[str, Any] = {}
    for arm in _ARM_ORDER:
        arm_geo = geometry.arms[arm]
        receipts: list[dict[str, Any]] = []
        inv_log: list[Any] = []
        kwargs = dict(
            runner=calls.runner, model=calls.model, batch=calls.batch,
            states=calls.states_by_arm[arm], eligible=calls.eligible,
            device=calls.device, steps=int(arm_geo.steps),
            start_step=int(arm_geo.start_step),
            global_horizon=int(geometry.runway_steps),
            hook=calls.hook_by_arm[arm], backlog=calls.backlog_by_arm[arm],
            flip=calls.flip_by_arm[arm], schedule=calls.schedule_by_arm[arm],
            rk=dict(calls.rk), arm=arm, log=inv_log,
            runner_contract=calls.runner_contract, a_rk_receipts=receipts,
        )
        _ARK_SIG.bind(**kwargs)
        invoke_arm_with_a_rk(**kwargs)
        if len(receipts) != 1:
            raise AdapterAuthorityRefuse("RECEIPT_CARDINALITY_NOT_ONE")
        row = receipts[0]
        if row.get("arm") != arm:
            raise AdapterAuthorityRefuse("RECEIPT_WRONG_ARM")
        summary = row.get(acct.ATTACHMENT_KEY)
        if not isinstance(summary, Mapping):
            raise AdapterAuthorityRefuse("RECEIPT_ATTACHMENT_MISSING")
        cap = mint()
        register(arm, cap)
        envelopes[arm] = {
            acct.ATTACHMENT_KEY: summary,
            "source_provenance": dict(acct.REQUIRED_SOURCE_PROVENANCE),
        }
        caps[arm] = cap
    return geometry, envelopes, caps


run_closed_four_arm_accounting_v2_cpu = _build_run_closed()
del _build_run_closed

__all__ = [
    "AdapterAuthorityRefuse",
    "FourArmARkCallInputs",
    "run_closed_four_arm_accounting_v2_cpu",
]
