"""Run-arms launch seam: authority matrix, materialize wire, A-RK runner kwargs.

Extracted from scripts/forgotten_accum_training_equivalence_run.py so run.py stays
a thin parser/adapter under the no-growth ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
    EXIT_RUN_ARMS_RUNNER_CONTRACT,
    RUNNER_CONTRACT_INVALID,
    RunnerContractRefuse,
    build_forgotten_accum_runner_contract,
    contract_digest,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_measure import (
    BankInputsRefuse,
    apply_claim_coupling,
    refuse_formal_unresolved_policy,
    synthetic_ledger_notes,
)

EXIT_RUN_ARMS_NO_AUTHORITY = 20
EXIT_RUN_ARMS_PREFLIGHT = 21
EXIT_RUN_ARMS_CONTROL_INVALID = 22
EXIT_RUN_ARMS_FAILURE = 23
EXIT_RUN_ARMS_IDENTITY = 24
EXIT_RUN_ARMS_BANK_INPUTS = 26


def resolve_run_arms_authority(args: argparse.Namespace) -> str:
    allow = bool(getattr(args, "allow_gpu_launch", False))
    smoke = bool(getattr(args, "i_have_claude_run_arms_smoke_authority", False))
    formal = bool(getattr(args, "formal_science", False))
    if smoke and formal:
        return "mutex"
    if allow and smoke and not formal:
        return "smoke"
    if allow and formal and not smoke:
        return "formal"
    return "refuse"


def run_arms_kwargs_from_args(args: argparse.Namespace) -> dict:
    """Pure argv→driver kwargs mapping (CPU-testable; no GPU)."""

    mode = resolve_run_arms_authority(args)
    return {
        "experiment_root": Path(args.scratch_root),
        "parent_sha256": str(args.parent_sha256),
        "live_acc_carrier_selector": str(args.live_acc_carrier_selector),
        "global_cap_contract": str(args.global_cap_contract),
        "eligible_scope": str(args.eligible_scope),
        "event_coded_flags_present": bool(args.event_coded_flags_present),
        "t_cut": int(args.t_cut),
        "runway_steps": int(args.runway_steps),
        "W": int(args.W),
        "developer_validation": mode != "formal",
        "device": str(args.device),
        "parent": Path(args.parent) if args.parent else None,
        "allow_gpu_launch": bool(args.allow_gpu_launch),
        "formal_science": bool(args.formal_science),
        "run_arms_smoke_authority": bool(
            getattr(args, "i_have_claude_run_arms_smoke_authority", False)
        ),
        "authority_mode": mode,
    }


def _bank_refuse_receipt(exc: BankInputsRefuse, *, mode: str) -> dict:
    notes = {
        "bank_section": getattr(exc, "kind", "UNRESOLVED_POLICY"),
        "bank_refuse_kind": getattr(exc, "kind", "UNRESOLVED_POLICY"),
    }
    notes.update(synthetic_ledger_notes())
    receipt = {
        "status": "FAILURE",
        "fail_closed_class": "BANK_INPUTS_INVALID",
        "error": str(exc),
        "science_label": None,
        "claimable_science": False,
        "bankable": False,
        "bank_receipts": None,
        "notes": notes,
    }
    return apply_claim_coupling(receipt, mode=mode)


def launch_run_arms(args: argparse.Namespace) -> tuple[dict, int]:
    """Authority already checked. Formal refuses BEFORE materialize/load."""

    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
        materialize_run_arms_live_bundle,
    )
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
        assert_carrier_preflight,
        run_forgotten_accum_training_equivalence_arms,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        run_bounded_delta_steps,
    )

    mode = resolve_run_arms_authority(args)
    if mode not in ("smoke", "formal"):
        raise RuntimeError(f"launch_run_arms called without authority (mode={mode})")
    # Scope (A): formal refuse BEFORE model load / materialize / GPU work.
    if mode == "formal":
        try:
            refuse_formal_unresolved_policy(bank_inputs=None)
        except BankInputsRefuse as exc:
            return _bank_refuse_receipt(exc, mode=mode), EXIT_RUN_ARMS_BANK_INPUTS

    assert_carrier_preflight(
        live_acc_carrier_selector=str(args.live_acc_carrier_selector),
        global_cap_contract=str(args.global_cap_contract),
        eligible_scope=str(args.eligible_scope),
        event_coded_flags_present=bool(args.event_coded_flags_present),
    )
    assert callable(run_forgotten_accum_training_equivalence_arms)
    assert callable(run_bounded_delta_steps)

    runner_contract = build_forgotten_accum_runner_contract(
        runway_steps=int(args.runway_steps)
    )
    bundle = materialize_run_arms_live_bundle(
        parent_path=Path(args.parent),
        expected_parent_sha256=str(args.parent_sha256),
        device=str(args.device),
        eligible_scope=str(args.eligible_scope),
    )
    driver_kwargs: dict[str, Any] = dict(
        runner=run_bounded_delta_steps,
        model=bundle.model,
        batch=bundle.batch,
        tensor_states=bundle.tensor_states,
        eligible_modules=bundle.eligible_modules,
        device=bundle.device,
        experiment_root=Path(args.scratch_root),
        parent_sha256=str(args.parent_sha256),
        live_acc_carrier_selector=str(args.live_acc_carrier_selector),
        global_cap_contract=str(args.global_cap_contract),
        eligible_scope=str(args.eligible_scope),
        event_coded_flags_present=bool(args.event_coded_flags_present),
        t_cut=int(args.t_cut),
        runway_steps=int(args.runway_steps),
        W=int(args.W),
        cadence_saver=bundle.cadence_saver,
        developer_validation=(mode == "smoke"),
        config=dict(bundle.config),
        runner_contract=runner_contract,
    )
    if mode == "smoke":
        t_cut = int(args.t_cut)
        runway = int(args.runway_steps)
        driver_kwargs["save_cadence"] = tuple(
            sorted({s for s in (t_cut, runway) if 1 <= int(s) <= int(runway)})
        )

    result = run_forgotten_accum_training_equivalence_arms(**driver_kwargs)
    receipt = result.as_dict() if hasattr(result, "as_dict") else dict(result)
    receipt["identity_inventory"] = bundle.identity_inventory
    receipt["runner_contract_requested_digest"] = contract_digest(runner_contract)
    receipt["runner_contract_pins"] = runner_contract.as_pins_dict()
    # Claim/mode labels live in bank_measure (safe-by-construction).
    receipt = apply_claim_coupling(receipt, mode=mode)
    if receipt.get("fail_closed_class") == RUNNER_CONTRACT_INVALID:
        return receipt, EXIT_RUN_ARMS_RUNNER_CONTRACT
    if receipt.get("fail_closed_class") == "BANK_INPUTS_INVALID":
        return receipt, EXIT_RUN_ARMS_BANK_INPUTS
    if receipt.get("fail_closed_class") == "CONTROL_INVALID":
        return receipt, EXIT_RUN_ARMS_CONTROL_INVALID
    if receipt.get("status") == "REFUSED":
        return receipt, EXIT_RUN_ARMS_PREFLIGHT
    if receipt.get("status") != "OK":
        return receipt, EXIT_RUN_ARMS_FAILURE
    return receipt, 0


def _emit_json(payload: dict, out_path: Path | None) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fail_payload(error: str, *, status: str = "FAILURE", fail=None) -> dict:
    out = {
        "status": status,
        "error": error,
        "science_label": None,
        "claimable_science": False,
        "bankable": False,
    }
    if fail is not None:
        out["fail_closed_class"] = fail
    return out


def cmd_run_arms(args: argparse.Namespace, argv: list[str]) -> int:
    mode = resolve_run_arms_authority(args)
    if mode == "mutex":
        print(
            "REFUSED: SMOKE_FORMAL_MUTUAL_EXCLUSION — "
            "--i-have-claude-run-arms-smoke-authority and --formal-science are mutually exclusive.",
            file=sys.stderr,
        )
        return EXIT_RUN_ARMS_NO_AUTHORITY
    if mode == "refuse":
        print(
            "REFUSED: run-arms requires --allow-gpu-launch AND exactly one of "
            "--i-have-claude-run-arms-smoke-authority (smoke) or --formal-science (formal).",
            file=sys.stderr,
        )
        return EXIT_RUN_ARMS_NO_AUTHORITY
    out_path = Path(args.receipt_out) if args.receipt_out else None
    try:
        receipt, code = launch_run_arms(args)
    except RunnerContractRefuse as exc:
        _emit_json(_fail_payload(str(exc), fail=RUNNER_CONTRACT_INVALID), out_path)
        return EXIT_RUN_ARMS_RUNNER_CONTRACT
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("PREFLIGHT_REFUSE"):
            _emit_json({"status": "REFUSED", "error": msg, "science_label": None}, out_path)
            return EXIT_RUN_ARMS_PREFLIGHT
        if msg.startswith(RUNNER_CONTRACT_INVALID):
            _emit_json(_fail_payload(msg, fail=RUNNER_CONTRACT_INVALID), out_path)
            return EXIT_RUN_ARMS_RUNNER_CONTRACT
        if msg.startswith("IDENTITY_REFUSE"):
            _emit_json(_fail_payload(msg, status="IDENTITY_REFUSED"), out_path)
            return EXIT_RUN_ARMS_IDENTITY
        raise
    except Exception as exc:  # noqa: BLE001
        from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
            IdentityRefuse,
        )
        if isinstance(exc, IdentityRefuse) or str(exc).startswith("IDENTITY_REFUSE"):
            _emit_json(_fail_payload(str(exc), status="IDENTITY_REFUSED"), out_path)
            return EXIT_RUN_ARMS_IDENTITY
        if isinstance(exc, RunnerContractRefuse) or str(exc).startswith(
            RUNNER_CONTRACT_INVALID
        ):
            _emit_json(_fail_payload(str(exc), fail=RUNNER_CONTRACT_INVALID), out_path)
            return EXIT_RUN_ARMS_RUNNER_CONTRACT
        payload = _fail_payload(f"{type(exc).__name__}: {exc}")
        payload["traceback"] = traceback.format_exc()
        _emit_json(payload, out_path)
        return EXIT_RUN_ARMS_FAILURE
    _emit_json(receipt, out_path)
    return int(code)


__all__ = [
    "EXIT_RUN_ARMS_NO_AUTHORITY",
    "EXIT_RUN_ARMS_PREFLIGHT",
    "EXIT_RUN_ARMS_CONTROL_INVALID",
    "EXIT_RUN_ARMS_FAILURE",
    "EXIT_RUN_ARMS_IDENTITY",
    "EXIT_RUN_ARMS_RUNNER_CONTRACT",
    "EXIT_RUN_ARMS_BANK_INPUTS",
    "apply_claim_coupling",
    "resolve_run_arms_authority",
    "run_arms_kwargs_from_args",
    "launch_run_arms",
    "cmd_run_arms",
]
