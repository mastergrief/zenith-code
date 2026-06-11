#!/usr/bin/env python3
"""Read-only W6/T=10 grad-proxy audit (slice 3a, step 1 only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.grad_proxy_audit import (
    GRAD_PROXY_AUDIT_RECEIPT_NAME,
    GradProxyAuditAborted,
    resolve_launch_sha,
    run_grad_proxy_audit_step1,
    write_grad_proxy_audit_receipt,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    DEFAULT_PARENT,
    DEFAULT_PARENT_SHA256,
    build_identity_full_support_batches,
    build_model_from_checkpoint,
    derive_tensor_states_and_check_init_fidelity,
    load_parent_checkpoint,
    select_eligible_bitlinears,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only M-A precursor: W6/T=10 grad-proxy vs shadow comparator "
            "audit at optimizer step 1."
        )
    )
    ap.add_argument(
        "--grad-proxy-audit-only",
        action="store_true",
        help="Required gate flag: run the read-only grad-proxy audit and exit.",
    )
    ap.add_argument("--parent", type=Path, default=Path(DEFAULT_PARENT))
    ap.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA256)
    ap.add_argument("--artifact-dir", type=Path, required=False)
    ap.add_argument("--max-audit-candidates", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eligible-scope", choices=["first-bitlinear", "all-bitlinear"], default="first-bitlinear")
    ap.add_argument("--curriculum-seed", type=int, default=44)
    ap.add_argument("--support-order-seed", type=int, default=44)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--max-abs-per-tensor", type=int, default=4096)
    ap.add_argument("--launch-sha", default=None)
    return ap


def run_grad_proxy_audit(args: argparse.Namespace) -> dict:
    if not bool(args.grad_proxy_audit_only):
        raise ValueError("--grad-proxy-audit-only is required for this entrypoint")
    if args.artifact_dir is None:
        raise ValueError("--artifact-dir is required")
    device = torch.device(str(args.device))
    ckpt, parent_sha = load_parent_checkpoint(
        Path(args.parent),
        expected_sha256=str(args.parent_sha256),
    )
    model, tok, cfg = build_model_from_checkpoint(ckpt, device)
    max_len = int(args.max_len) if args.max_len is not None else int(cfg.max_seq_len)
    support_batches, support_proof = build_identity_full_support_batches(
        tok=tok,
        max_len=max_len,
        batch_size=int(args.batch_size),
        curriculum_seed=int(args.curriculum_seed),
        device=device,
        support_order_seed=int(args.support_order_seed),
    )
    if not support_batches:
        raise RuntimeError("grad-proxy audit requires at least one support batch")
    batch = support_batches[0]["batch"]
    eligible = select_eligible_bitlinears(model, eligible_scope=str(args.eligible_scope))
    tensor_states, init_report = derive_tensor_states_and_check_init_fidelity(
        eligible,
        threshold=0.0,
    )
    if not init_report["all_pass"]:
        raise RuntimeError("init fidelity failed for grad-proxy audit parent")
    model.train()
    extras = model.compute_train_extra_args(1, 1)
    launch_sha = str(args.launch_sha or resolve_launch_sha())
    receipt = run_grad_proxy_audit_step1(
        model=model,
        batch=batch,
        tensor_states=tensor_states,
        eligible_modules=eligible,
        device=device,
        extras=extras,
        max_abs_per_tensor=int(args.max_abs_per_tensor),
        max_audit_candidates=int(args.max_audit_candidates),
        launch_sha=launch_sha,
    )
    receipt["parent_sha256"] = parent_sha
    receipt["support_proof"] = support_proof
    receipt_path = write_grad_proxy_audit_receipt(
        receipt,
        artifact_dir=Path(args.artifact_dir),
    )
    receipt["receipt_path"] = receipt_path
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = run_grad_proxy_audit(args)
    except GradProxyAuditAborted as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"wrote {GRAD_PROXY_AUDIT_RECEIPT_NAME} -> {receipt['receipt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
