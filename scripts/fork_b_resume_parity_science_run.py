#!/usr/bin/env python3
"""Fork B resume-parity SCIENCE CLI — fail-closed formal-launch entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RECEIPT_SCHEMA = "fork_b_resume_parity_science_cli/v2"
STUB_TOKEN = "CLI_WIRED_AWAITING_LAUNCH_PACKET"
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
PARENT_PATH = (
    "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_"
    "anchorsv1r3_from_L0b_final_step01500.pt"
)
SCRATCH_TEMPLATE = (
    "/home/gabe/claw-code-creditdir/transient_fp_credit/fork_b_formal_{LAUNCH_NONCE}"
)
ARGV_TEMPLATE: list[str] = [
    "python", "-u", "scripts/fork_b_resume_parity_science_run.py",
    "--allow-gpu-launch", "--formal-science", "--eligible-scope", "all-bitlinear",
    "--batch-size", "1",
    "--parent", PARENT_PATH, "--parent-sha256", PARENT_SHA256,
    "--scratch-root", SCRATCH_TEMPLATE, "--device", "cuda:0",
    "--cuts", "4,16,28", "--k-steps", "4", "--steps", "32",
    "--global-horizon", "32", "--batch-seed", "44", "--support-order-seed", "43",
    "--ordering-seed", "17", "--launch-nonce", "{LAUNCH_NONCE}",
]
ARGV_TEMPLATE_SHA256 = "06bcc9072bbf798373b05b87b86e48be89412bd6e5af65336e053a68792ce354"
CLEAN_ARCHIVE_HEAD_SHA = "f86c7ec109dc260ffcc9657714def194b708101d"
CLEAN_ARCHIVE_TREE_SHA = "1d592f18cf0ac518079b3434ab550389d3e5c6a2"
FORMAL_CUTS = (4, 16, 28)
RUNNER_IDENTITY = (
    "scripts.hrm_text_158_bounded_delta_acquisition_probe.run_bounded_delta_steps"
)

# Test hooks (monkeypatch targets).
_sha256_file: Callable[[Path], str] | None = None
_run_certificate: Callable[..., Any] | None = None
_load_parent_and_run: Callable[..., Any] | None = None

def canonical_argv_sha256(argv: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(argv), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

def concrete_argv_from_nonce(nonce: str) -> list[str]:
    return [t.replace("{LAUNCH_NONCE}", str(nonce)) for t in ARGV_TEMPLATE]

def sha256_file(path: Path) -> str:
    if _sha256_file is not None:
        return _sha256_file(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fork B 5-arm resume-parity science run")
    p.add_argument("--allow-gpu-launch", action="store_true", default=False)
    p.add_argument("--formal-science", action="store_true", default=False)
    p.add_argument("--eligible-scope", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuts", default="4,16,28")
    p.add_argument("--k-steps", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--global-horizon", type=int, default=32)
    p.add_argument("--batch-seed", type=int, default=44)
    p.add_argument("--support-order-seed", type=int, default=43)
    p.add_argument("--ordering-seed", type=int, default=17)
    p.add_argument("--launch-nonce", default=None)
    p.add_argument("--argv-template-sha256", default=None)
    p.add_argument("--concrete-argv-sha256", default=None)
    return p

def _parse_cuts(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if str(x).strip()]

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path

def _base_receipt(*, args: argparse.Namespace, started: str, pre_science: str | None,
                  status: str, science_label: Any = None, terminal: Any = None,
                  parent_before: str | None = None, parent_after: str | None = None,
                  parent_unchanged: bool | None = None, error: str | None = None) -> dict[str, Any]:
    nonce = str(args.launch_nonce or "")
    concrete = concrete_argv_from_nonce(nonce) if nonce else []
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "launch_nonce": nonce or None,
        "argv_template_sha256": ARGV_TEMPLATE_SHA256,
        "concrete_argv_sha256": canonical_argv_sha256(concrete) if concrete else None,
        "clean_archive_head_sha": CLEAN_ARCHIVE_HEAD_SHA,
        "clean_archive_tree_sha": CLEAN_ARCHIVE_TREE_SHA,
        "parent_path": str(args.parent),
        "parent_sha256_before": parent_before,
        "parent_sha256_after": parent_after,
        "parent_unchanged": parent_unchanged,
        "cuts": _parse_cuts(args.cuts),
        "k_steps": int(args.k_steps),
        "steps": int(args.steps),
        "batch_seed": int(args.batch_seed),
        "support_order_seed": int(args.support_order_seed),
        "ordering_seed": int(args.ordering_seed),
        "global_horizon": int(args.global_horizon),
        "eligible_scope": args.eligible_scope,
        "batch_size": getattr(args, "batch_size", None),
        "formal_science": bool(args.formal_science),
        "developer_validation": False if args.formal_science else None,
        "require_strict_f_equals_u": True if args.formal_science else None,
        "require_z_gate_break": True if args.formal_science else None,
        "runner_identity": RUNNER_IDENTITY,
        "terminal": terminal,
        "science_label": science_label,
        "pre_science": pre_science,
        "scratch_root": str(args.scratch_root),
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "error": error,
    }

def _fail(args: argparse.Namespace, *, pre_science: str, started: str,
          parent_before: str | None = None, parent_after: str | None = None,
          parent_unchanged: bool | None = None, error: str | None = None,
          terminal: Any = None, exit_code: int = 1) -> int:
    receipt = _base_receipt(
        args=args, started=started, pre_science=pre_science, status="FAILED",
        parent_before=parent_before, parent_after=parent_after,
        parent_unchanged=parent_unchanged, error=error, terminal=terminal,
    )
    assert receipt["status"] != STUB_TOKEN
    path = _atomic_write_json(Path(args.scratch_root) / "fork_b_science_cli_receipt.json", receipt)
    print(json.dumps({"receipt_path": str(path), "science_label": None, "pre_science": pre_science}))
    return exit_code

def _frozen_tuple_preflight(args: argparse.Namespace) -> str | None:
    if not args.formal_science:
        return "ALLOW_WITHOUT_FORMAL"
    if not args.launch_nonce or not str(args.launch_nonce).strip():
        return "EMPTY_LAUNCH_NONCE"
    if args.eligible_scope != "all-bitlinear":
        return "FROZEN_TUPLE_MISMATCH"
    if args.batch_size is None or int(args.batch_size) != 1:
        return "FROZEN_TUPLE_MISMATCH"
    if int(args.global_horizon) != 32:
        return "FROZEN_TUPLE_MISMATCH"
    if tuple(_parse_cuts(args.cuts)) != FORMAL_CUTS:
        return "FROZEN_TUPLE_MISMATCH"
    if int(args.k_steps) != 4 or int(args.steps) != 32:
        return "FROZEN_TUPLE_MISMATCH"
    if (int(args.batch_seed), int(args.support_order_seed), int(args.ordering_seed)) != (44, 43, 17):
        return "FROZEN_TUPLE_MISMATCH"
    if str(args.parent_sha256) != PARENT_SHA256:
        return "FROZEN_TUPLE_MISMATCH"
    if str(args.parent) != PARENT_PATH:
        return "FROZEN_TUPLE_MISMATCH"
    expected_template = ARGV_TEMPLATE_SHA256
    expected_concrete = canonical_argv_sha256(concrete_argv_from_nonce(str(args.launch_nonce)))
    if not args.argv_template_sha256 or str(args.argv_template_sha256) != expected_template:
        return "ARGV_HASH_MISMATCH"
    if not args.concrete_argv_sha256 or str(args.concrete_argv_sha256) != expected_concrete:
        return "ARGV_HASH_MISMATCH"
    expected_scratch = SCRATCH_TEMPLATE.replace("{LAUNCH_NONCE}", str(args.launch_nonce))
    if str(args.scratch_root) != expected_scratch:
        return "FROZEN_TUPLE_MISMATCH"
    return None

def _default_load_and_run(args: argparse.Namespace, *, parent_before: str) -> dict[str, Any]:
    import torch
    from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_science_driver import (
        FORMAL_GLOBAL_HORIZON,
        run_fork_b_resume_parity_certificate,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        build_identity_full_support_batches,
        build_model_from_checkpoint,
        derive_tensor_states_and_check_init_fidelity,
        run_bounded_delta_steps,
        select_eligible_bitlinears,
    )

    cert = _run_certificate or run_fork_b_resume_parity_certificate
    device = torch.device(str(args.device))
    ckpt = torch.load(Path(args.parent), map_location="cpu", weights_only=False)
    model, tok, _cfg = build_model_from_checkpoint(ckpt, device)
    support_batches, _proof = build_identity_full_support_batches(
        tok=tok, max_len=int(getattr(_cfg, "max_seq_len", 64) or 64),
        batch_size=int(args.batch_size), curriculum_seed=int(args.ordering_seed), device=device,
    )
    eligible = select_eligible_bitlinears(model, eligible_scope="all-bitlinear")
    states, report = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)
    if not report.get("all_pass", False):
        raise RuntimeError(f"init fidelity failed: {report}")
    runner_kwargs = {
        "global_horizon": 32,
        "batch_size": int(args.batch_size),
        "global_cap_contract": C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        "max_abs_per_tensor": 4096,
        "r7_deferred_backlog_carry_enabled": True,
        "require_q_change": False,
        "eligible_scope": "all-bitlinear",
    }
    assert FORMAL_GLOBAL_HORIZON == 32
    assert run_bounded_delta_steps.__module__.endswith(
        "hrm_text_158_bounded_delta_acquisition_probe"
    )
    result = cert(
        runner=run_bounded_delta_steps,
        model=model,
        batch=support_batches[0]["batch"],
        tensor_states=states,
        eligible_modules=eligible,
        device=device,
        scratch_root=Path(args.scratch_root) / "certificate",
        parent_sha16=parent_before[:16],
        batch_seed=int(args.batch_seed),
        support_order_seed=int(args.support_order_seed),
        ordering_seed=int(args.ordering_seed),
        cuts=list(FORMAL_CUTS),
        k_steps=4,
        total_steps=32,
        support_batches=support_batches,
        runner_kwargs=runner_kwargs,
        developer_validation=False,
        require_strict_f_equals_u=True,
        require_z_gate_break=True,
        global_horizon=32,
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    payload["_spy_bindings"] = {
        "developer_validation": False,
        "require_strict_f_equals_u": True,
        "require_z_gate_break": True,
        "runner_identity": RUNNER_IDENTITY,
        "global_horizon": 32,
        "runner_kwargs": runner_kwargs,
        "eligible_scope": "all-bitlinear",
        "batch_size": int(args.batch_size),
        "support_batch_size": int(args.batch_size),
        "cuts": list(FORMAL_CUTS),
        "k_steps": 4,
        "steps": 32,
        "batch_seed": int(args.batch_seed),
        "support_order_seed": int(args.support_order_seed),
        "ordering_seed": int(args.ordering_seed),
    }
    return payload

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = _utc_now()
    if not args.allow_gpu_launch:
        print(
            "REFUSED: Fork B science CLI requires --allow-gpu-launch "
            "(no implicit GPU; formal launch is a separate gate)",
            file=sys.stderr,
        )
        return 2

    Path(args.scratch_root).mkdir(parents=True, exist_ok=True)
    pre = _frozen_tuple_preflight(args)
    if pre is not None:
        return _fail(args, pre_science=pre, started=started)

    parent_path = Path(args.parent)
    parent_before: str | None = None
    parent_after: str | None = None
    if not parent_path.is_file():
        return _fail(args, pre_science="PARENT_REHASH_MISMATCH", started=started,
                     error=f"parent missing: {parent_path}")
    parent_before = sha256_file(parent_path)
    if parent_before != str(args.parent_sha256) or parent_before != PARENT_SHA256:
        return _fail(
            args, pre_science="PARENT_REHASH_MISMATCH", started=started,
            parent_before=parent_before,
        )

    loader = _load_parent_and_run or _default_load_and_run
    try:
        cert_out = loader(args, parent_before=parent_before)
    except Exception as exc:  # noqa: BLE001 — atomic failure receipt required
        parent_after = sha256_file(parent_path) if parent_path.is_file() else None
        return _fail(
            args, pre_science="CERTIFICATE_EXCEPTION", started=started,
            parent_before=parent_before, parent_after=parent_after,
            parent_unchanged=(parent_after == parent_before) if parent_after else None,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    parent_after = sha256_file(parent_path) if parent_path.is_file() else None
    unchanged = bool(parent_after == parent_before) if parent_after is not None else False
    if not unchanged:
        return _fail(
            args, pre_science="PARENT_REHASH_MISMATCH", started=started,
            parent_before=parent_before, parent_after=parent_after,
            parent_unchanged=False, terminal=cert_out.get("terminal"),
        )

    receipt = _base_receipt(
        args=args, started=started, pre_science=cert_out.get("pre_science"),
        status="COMPLETE", science_label=cert_out.get("science_label"),
        terminal=cert_out.get("terminal"), parent_before=parent_before,
        parent_after=parent_after, parent_unchanged=True,
    )
    receipt["certificate_notes"] = cert_out.get("notes")
    receipt["spy_bindings"] = cert_out.get("_spy_bindings")
    if STUB_TOKEN in json.dumps(receipt):
        return _fail(args, pre_science="STUB_TOKEN_LEAK", started=started,
                     parent_before=parent_before, parent_after=parent_after,
                     parent_unchanged=True, error="stub token forbidden on success")
    path = _atomic_write_json(
        Path(args.scratch_root) / "fork_b_science_cli_receipt.json", receipt,
    )
    print(json.dumps({
        "receipt_path": str(path),
        "science_label": receipt.get("science_label"),
        "pre_science": receipt.get("pre_science"),
        "status": "COMPLETE",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
