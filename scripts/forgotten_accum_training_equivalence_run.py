#!/usr/bin/env python3
"""Thin CLI for forgotten-accum training-equivalence (Phase-B + run-arms).

`smoke-dense-site` is a real device smoke behind
`--i-have-claude-gpu-smoke-authority` (claude/test-operator runs it).

`run-arms` is the 4-arm driver entrypoint. Default refuse. Authority matrix:
- smoke: `--allow-gpu-launch` AND `--i-have-claude-run-arms-smoke-authority`
  AND NOT `--formal-science`
- formal: `--allow-gpu-launch` AND `--formal-science` AND NOT smoke flag
Both set => refuse (mutual exclusion).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _repo_root() -> Path:
    return REPO_ROOT


def cmd_manifests(_: argparse.Namespace) -> int:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_arms import (
        all_else_identical_manifests,
    )

    print(json.dumps(all_else_identical_manifests(), indent=2, sort_keys=True))
    return 0


def cmd_smoke_predicates(_: argparse.Namespace) -> int:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        DENSE_LEGACY_CAP_SITE_ID,
        GLOBAL_CAP_CONTRACT,
        SMOKE_CPU_PREDICATES,
    )

    print(
        json.dumps(
            {
                "predicates": list(SMOKE_CPU_PREDICATES),
                "DENSE_LEGACY_CAP_SITE_ID": DENSE_LEGACY_CAP_SITE_ID,
                "global_cap_contract": GLOBAL_CAP_CONTRACT,
                "gpu_execution": "requires --i-have-claude-gpu-smoke-authority",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_smoke_dense_site(args: argparse.Namespace, argv: list[str]) -> int:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
        EXIT_NO_AUTHORITY,
    )

    if not bool(args.i_have_claude_gpu_smoke_authority):
        print(
            "REFUSED: GPU dense-site smoke requires --i-have-claude-gpu-smoke-authority "
            "(claude/test-operator separate authority).",
            file=sys.stderr,
        )
        return EXIT_NO_AUTHORITY

    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        PARENT_SHA256_FULL,
    )
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
        DEFAULT_PARENT_RELPATH,
        EXIT_EVENT_CODED_STOP,
        EXIT_INFRA_FAIL,
        run_dense_site_device_smoke,
    )

    repo = _repo_root()
    parent = Path(args.parent) if args.parent else (repo / DEFAULT_PARENT_RELPATH)
    expected = str(args.parent_sha256 or PARENT_SHA256_FULL)
    out_path = Path(args.receipt_out) if args.receipt_out else None

    try:
        receipt, code = run_dense_site_device_smoke(
            repo_root=repo,
            parent_path=parent,
            expected_parent_sha256=expected,
            device=str(args.device),
            include_deferred_second_step=not bool(args.skip_deferred_step),
            argv_for_digest=list(argv),
            run_live_runner_branch_probe=not bool(
                getattr(args, "skip_live_runner_branch_probe", False)
            ),
        )
    except Exception as exc:  # noqa: BLE001 — emit infra receipt
        err = {
            "schema": "forgotten_accum_dense_site_smoke_receipt/v2",
            "pass_fail": "FAIL",
            "smoke_class": "FAIL",
            "failures": ["infra_exception"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "device": str(args.device),
            "parent": str(parent),
        }
        print(json.dumps(err, indent=2, sort_keys=True))
        if out_path is not None:
            out_path.write_text(json.dumps(err, indent=2, sort_keys=True) + "\n")
        return EXIT_INFRA_FAIL

    blob = receipt.as_dict()
    print(json.dumps(blob, indent=2, sort_keys=True))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    if code == EXIT_EVENT_CODED_STOP:
        print("STOP: event-coded / non-dense carrier", file=sys.stderr)
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
        EXIT_INCONCLUSIVE_NO_CROSSING,
    )

    if code == EXIT_INCONCLUSIVE_NO_CROSSING:
        print("INCONCLUSIVE_NO_CROSSING: ordinary twin lacked qualifying demand", file=sys.stderr)
    return int(code)


EXIT_RUN_ARMS_NO_AUTHORITY = 20
EXIT_RUN_ARMS_PREFLIGHT = 21
EXIT_RUN_ARMS_CONTROL_INVALID = 22
EXIT_RUN_ARMS_FAILURE = 23
EXIT_RUN_ARMS_IDENTITY = 24


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


def launch_run_arms(args: argparse.Namespace) -> tuple[dict, int]:
    """Authority already checked. Materialize live bundle then run science_driver."""

    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
        IdentityRefuse,
        assert_formal_canonical_params,
        materialize_run_arms_live_bundle,
        stamp_authority_receipt,
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
    if mode == "formal":
        assert_formal_canonical_params(
            t_cut=int(args.t_cut), runway_steps=int(args.runway_steps), W=int(args.W)
        )

    assert_carrier_preflight(
        live_acc_carrier_selector=str(args.live_acc_carrier_selector),
        global_cap_contract=str(args.global_cap_contract),
        eligible_scope=str(args.eligible_scope),
        event_coded_flags_present=bool(args.event_coded_flags_present),
    )
    assert callable(run_forgotten_accum_training_equivalence_arms)
    assert callable(run_bounded_delta_steps)

    bundle = materialize_run_arms_live_bundle(
        parent_path=Path(args.parent),
        expected_parent_sha256=str(args.parent_sha256),
        device=str(args.device),
        eligible_scope=str(args.eligible_scope),
    )

    driver_kwargs = dict(
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
    receipt = stamp_authority_receipt(receipt, mode=mode)
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
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("PREFLIGHT_REFUSE"):
            _emit_json({"status": "REFUSED", "error": msg, "science_label": None}, out_path)
            return EXIT_RUN_ARMS_PREFLIGHT
        if msg.startswith("IDENTITY_REFUSE"):
            _emit_json(
                {
                    "status": "IDENTITY_REFUSED",
                    "error": msg,
                    "science_label": None,
                    "claimable_science": False,
                    "bankable": False,
                },
                out_path,
            )
            return EXIT_RUN_ARMS_IDENTITY
        raise
    except Exception as exc:  # noqa: BLE001
        from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
            IdentityRefuse,
        )

        if isinstance(exc, IdentityRefuse) or str(exc).startswith("IDENTITY_REFUSE"):
            _emit_json(
                {
                    "status": "IDENTITY_REFUSED",
                    "error": str(exc),
                    "science_label": None,
                    "claimable_science": False,
                    "bankable": False,
                },
                out_path,
            )
            return EXIT_RUN_ARMS_IDENTITY
        _emit_json(
            {
                "status": "FAILURE",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "science_label": None,
            },
            out_path,
        )
        return EXIT_RUN_ARMS_FAILURE

    _emit_json(receipt, out_path)
    return int(code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgotten_accum_training_equivalence_run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifests", help="Print all-else-identical arm manifests")
    sub.add_parser(
        "smoke-predicates",
        help="Print CPU-checkable dense-site smoke predicates",
    )
    smoke = sub.add_parser(
        "smoke-dense-site",
        help="GPU dense-site 1-step smoke (requires claude GPU authority flag)",
    )
    smoke.add_argument(
        "--i-have-claude-gpu-smoke-authority",
        action="store_true",
        help="Required to run device smoke (separate claude/test-operator authority)",
    )
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--parent", default=None, help="Parent checkpoint path")
    smoke.add_argument(
        "--parent-sha256",
        default=None,
        help="Expected parent sha256 (default: frozen 9b4e311a…)",
    )
    smoke.add_argument(
        "--receipt-out",
        default=None,
        help="Optional path to write terminal receipt JSON",
    )
    smoke.add_argument(
        "--skip-deferred-step",
        action="store_true",
        help="Ignored under receipt/v2 (non-vacuous twin always runs both arms)",
    )
    smoke.add_argument(
        "--skip-live-runner-branch-probe",
        action="store_true",
        help="Skip the live run_bounded_delta_steps dense-site probe (twin-only)",
    )

    arms = sub.add_parser(
        "run-arms",
        help="U/E/R0/RW science/smoke driver (requires launch authority flags)",
    )
    arms.add_argument("--allow-gpu-launch", action="store_true", default=False)
    arms.add_argument("--formal-science", action="store_true", default=False)
    arms.add_argument(
        "--i-have-claude-run-arms-smoke-authority",
        action="store_true",
        default=False,
        help="Distinct smoke authority; mutually exclusive with --formal-science",
    )
    arms.add_argument("--device", default="cuda:0")
    arms.add_argument("--parent", required=True)
    arms.add_argument("--parent-sha256", required=True)
    arms.add_argument("--scratch-root", required=True)
    arms.add_argument("--live-acc-carrier-selector", default="NONE")
    arms.add_argument(
        "--global-cap-contract", default="c1_banked_faithful_long_run_global_cap"
    )
    arms.add_argument("--eligible-scope", default="all-bitlinear")
    arms.add_argument("--event-coded-flags-present", action="store_true", default=False)
    arms.add_argument("--t-cut", type=int, default=500)
    arms.add_argument("--runway-steps", type=int, default=1500)
    arms.add_argument("--W", type=int, default=32)
    arms.add_argument("--receipt-out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)
    if args.cmd == "manifests":
        return cmd_manifests(args)
    if args.cmd == "smoke-predicates":
        return cmd_smoke_predicates(args)
    if args.cmd == "smoke-dense-site":
        return cmd_smoke_dense_site(args, argv_list)
    if args.cmd == "run-arms":
        return cmd_run_arms(args, argv_list)
    parser.error(f"unknown cmd {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
