"""Pure ForgottenAccum runner contract (A-RK) — no probe/GPU/launch imports.

Fail-closed: missing/defaulted/mismatched science pins => RUNNER_CONTRACT_INVALID.
Effective values must come from consumer-site stamps (A-EFF), never entry kwargs alone.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    GLOBAL_CAP_CONTRACT,
)

RUNNER_CONTRACT_INVALID = "RUNNER_CONTRACT_INVALID"
EXIT_RUN_ARMS_RUNNER_CONTRACT = 25
EFFECTIVE_STAMP_KEY = "forgotten_accum_runner_contract_effective"
PIN_NAMES = (
    "global_horizon",
    "global_cap_contract",
    "max_abs_per_tensor",
    "r7_deferred_backlog_carry_enabled",
    "require_q_change",
)


class RunnerContractRefuse(ValueError):
    """Raised when A-RK three-way / bind / pin checks fail."""

    def __init__(self, message: str):
        super().__init__(f"{RUNNER_CONTRACT_INVALID}: {message}")


@dataclass(frozen=True)
class ForgottenAccumRunnerContract:
    """Indivisible five-pin runner contract for U/E/R0/RW."""

    global_horizon: int
    global_cap_contract: str
    max_abs_per_tensor: int
    r7_deferred_backlog_carry_enabled: bool
    require_q_change: bool
    runway_steps: int
    derivation_rule: str = "global_horizon = absolute final optimizer step of THIS run (= runway_steps)"

    def as_pins_dict(self) -> dict[str, Any]:
        return {
            "global_horizon": int(self.global_horizon),
            "global_cap_contract": str(self.global_cap_contract),
            "max_abs_per_tensor": int(self.max_abs_per_tensor),
            "r7_deferred_backlog_carry_enabled": bool(self.r7_deferred_backlog_carry_enabled),
            "require_q_change": bool(self.require_q_change),
        }

    def as_runner_kwargs(self) -> dict[str, Any]:
        return dict(self.as_pins_dict())


def build_forgotten_accum_runner_contract(*, runway_steps: int) -> ForgottenAccumRunnerContract:
    """Build the frozen five-pin contract. No science-relevant defaults on the path."""

    horizon = int(runway_steps)
    return ForgottenAccumRunnerContract(
        global_horizon=horizon,
        global_cap_contract=str(GLOBAL_CAP_CONTRACT),
        max_abs_per_tensor=4096,
        r7_deferred_backlog_carry_enabled=True,
        require_q_change=False,
        runway_steps=horizon,
    )


def contract_digest(contract: ForgottenAccumRunnerContract) -> str:
    blob = json.dumps(asdict(contract), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def bind_real_run_bounded_delta_steps(
    fn: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> inspect.BoundArguments:
    """bind() — NOT bind_partial. Does not apply_defaults before pin presence check."""

    return inspect.signature(fn).bind(*args, **dict(kwargs))


def assert_pins_explicit_in_bound(bound: inspect.BoundArguments) -> None:
    missing = [n for n in PIN_NAMES if n not in bound.arguments]
    if missing:
        raise RunnerContractRefuse(
            f"science pins omitted from invocation (would inherit signature defaults): {missing}"
        )


def pins_from_bound(bound: inspect.BoundArguments) -> dict[str, Any]:
    assert_pins_explicit_in_bound(bound)
    return {
        "global_horizon": int(bound.arguments["global_horizon"]),
        "global_cap_contract": str(bound.arguments["global_cap_contract"]),
        "max_abs_per_tensor": int(bound.arguments["max_abs_per_tensor"]),
        "r7_deferred_backlog_carry_enabled": bool(
            bound.arguments["r7_deferred_backlog_carry_enabled"]
        ),
        "require_q_change": bool(bound.arguments["require_q_change"]),
    }


def extract_effective_from_runner_return(runner_return: Any) -> dict[str, Any]:
    """Pull A-EFF stamp from run_bounded_delta_steps return tuple element updater_config."""

    if not isinstance(runner_return, tuple) or len(runner_return) < 2:
        raise RunnerContractRefuse(
            "runner return is not the expected tuple (step_reports, updater_config, ...)"
        )
    updater_config = runner_return[1]
    if not isinstance(updater_config, Mapping):
        raise RunnerContractRefuse("updater_config missing/non-mapping in runner return")
    if EFFECTIVE_STAMP_KEY not in updater_config:
        raise RunnerContractRefuse(
            f"missing {EFFECTIVE_STAMP_KEY!r} — A-EFF stamp absent; cannot prove effective leg"
        )
    stamp = dict(updater_config[EFFECTIVE_STAMP_KEY])
    if not bool(stamp.get("within_arm_consistent", False)):
        raise RunnerContractRefuse(
            f"within-arm A-EFF observations inconsistent: {stamp.get('inconsistency')}"
        )
    required = list(PIN_NAMES) + ["global_cap_resolved_spec_present"]
    missing = [k for k in required if k not in stamp]
    if missing:
        raise RunnerContractRefuse(f"A-EFF stamp incomplete, missing {missing}")
    return stamp


def _pin_view(pins: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "global_horizon": int(pins["global_horizon"]),
        "global_cap_contract": (
            None
            if pins["global_cap_contract"] is None
            else str(pins["global_cap_contract"])
        ),
        "max_abs_per_tensor": int(pins["max_abs_per_tensor"]),
        "r7_deferred_backlog_carry_enabled": bool(pins["r7_deferred_backlog_carry_enabled"]),
        "require_q_change": bool(pins["require_q_change"]),
    }


def assert_three_way_equal(
    *,
    requested: ForgottenAccumRunnerContract | Mapping[str, Any],
    bound_pins: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    """Requested == bound == effective (A-EFF). Cap absent-spec must not equal requested name."""

    req = (
        requested.as_pins_dict()
        if isinstance(requested, ForgottenAccumRunnerContract)
        else _pin_view(requested)
    )
    bnd = _pin_view(bound_pins)
    eff_pins = {
        "global_horizon": int(effective["global_horizon"]),
        "global_cap_contract": (
            None
            if effective["global_cap_contract"] is None
            else str(effective["global_cap_contract"])
        ),
        "max_abs_per_tensor": int(effective["max_abs_per_tensor"]),
        "r7_deferred_backlog_carry_enabled": bool(
            effective["r7_deferred_backlog_carry_enabled"]
        ),
        "require_q_change": bool(effective["require_q_change"]),
    }
    if not bool(effective.get("global_cap_resolved_spec_present", False)):
        # Resolver returned no spec: effective cap name MUST be None and cannot
        # match a requested non-None banked contract (decisive A-EFF negative).
        if eff_pins["global_cap_contract"] is not None:
            raise RunnerContractRefuse(
                "resolved_spec_present=False but effective global_cap_contract is not None"
            )
        if req["global_cap_contract"] is not None:
            raise RunnerContractRefuse(
                "requested cap present but effective resolved_spec_present=False "
                f"(requested={req['global_cap_contract']!r}, effective=None)"
            )
    if req != bnd:
        raise RunnerContractRefuse(f"requested!=bound pins: requested={req} bound={bnd}")
    if req != eff_pins:
        raise RunnerContractRefuse(
            f"requested!=effective pins: requested={req} effective={eff_pins}"
        )
    return {
        "requested": req,
        "bound": bnd,
        "effective": eff_pins,
        "global_cap_resolved_spec_present": bool(
            effective["global_cap_resolved_spec_present"]
        ),
        "three_way_equality_pass": True,
    }


def assert_horizon_matches_runway(
    *, observed_horizon: int, runway_steps: int, arm: str
) -> None:
    if int(observed_horizon) != int(runway_steps):
        raise RunnerContractRefuse(
            f"horizon spy mismatch arm={arm}: observed={observed_horizon} "
            f"runway_steps={runway_steps}"
        )


def finalize_a_eff_stamp_from_observations(
    *,
    horizon_obs: Sequence[int],
    cap_name_obs: Sequence[Any],
    cap_resolved_obs: Sequence[bool],
    max_abs_per_tensor: int,
    r7_consumed: bool,
    require_q_change_consumed: bool,
) -> dict[str, Any]:
    """Pure helper to form the A-EFF stamp (also usable from tests)."""

    inconsistency: list[str] = []
    if not horizon_obs:
        inconsistency.append("empty horizon observations")
        horizon = None
    elif len(set(int(x) for x in horizon_obs)) != 1:
        inconsistency.append(f"horizon varies within arm: {list(horizon_obs)}")
        horizon = int(horizon_obs[-1])
    else:
        horizon = int(horizon_obs[0])

    if not cap_name_obs:
        inconsistency.append("empty cap name observations")
        cap_name = None
    elif len({(None if x is None else str(x)) for x in cap_name_obs}) != 1:
        inconsistency.append(f"cap name varies within arm: {list(cap_name_obs)}")
        cap_name = None if cap_name_obs[-1] is None else str(cap_name_obs[-1])
    else:
        cap_name = None if cap_name_obs[0] is None else str(cap_name_obs[0])

    if not cap_resolved_obs:
        inconsistency.append("empty cap resolved-spec observations")
        resolved = False
    elif len(set(bool(x) for x in cap_resolved_obs)) != 1:
        inconsistency.append(f"cap resolved_spec_present varies: {list(cap_resolved_obs)}")
        resolved = bool(cap_resolved_obs[-1])
    else:
        resolved = bool(cap_resolved_obs[0])

    return {
        "global_horizon": horizon,
        "global_cap_contract": cap_name,
        "global_cap_resolved_spec_present": resolved,
        "max_abs_per_tensor": int(max_abs_per_tensor),
        "r7_deferred_backlog_carry_enabled": bool(r7_consumed),
        "require_q_change": bool(require_q_change_consumed),
        "within_arm_consistent": not inconsistency,
        "inconsistency": inconsistency or None,
    }


__all__ = [
    "RUNNER_CONTRACT_INVALID",
    "EXIT_RUN_ARMS_RUNNER_CONTRACT",
    "EFFECTIVE_STAMP_KEY",
    "PIN_NAMES",
    "RunnerContractRefuse",
    "ForgottenAccumRunnerContract",
    "build_forgotten_accum_runner_contract",
    "contract_digest",
    "bind_real_run_bounded_delta_steps",
    "assert_pins_explicit_in_bound",
    "pins_from_bound",
    "extract_effective_from_runner_return",
    "assert_three_way_equal",
    "assert_horizon_matches_runway",
    "finalize_a_eff_stamp_from_observations",
]
