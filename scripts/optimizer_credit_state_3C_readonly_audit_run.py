#!/usr/bin/env python3
"""Governing A+B harness for 3C read-only classification audit (PLAN_v5).

Consumes IntegerPathDenseSurfaceObservationEvidence directly into the receipt.
FORBIDS recomputing projected_moves_from_integer_attribution for receipt fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    build_optimizer_excluding_eligible_masters,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_AUDIT_PENDING,
    BRANCH_3C_C_DENSE_LEAK,
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (
    assert_eligible_modules_owned_by_model,
    compute_canonical_json_sha256,
    compute_optimizer_credit_state_no_hidden_fp_audit_receipt_sha256,
    compute_tensor_canonical_sha256,
    run_integer_path_dense_surface_observation_with_alloc_guard,
    run_optimizer_credit_state_no_hidden_fp_audit,
    validate_evidence_receipt_field_equality,
)

FIXTURE_RECIPE_NAME = "3C_C1_dry_run_fixture_seed158"
PLAN_SHA256_EXPECTED = (
    "f6668f38ae19cd169c6186897aea6519203775dffbad0f65f5a31e165cc20159"
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _dry_run_fixture() -> tuple[
    dict, torch.Tensor, tuple[int, int], dict[str, Any], torch.nn.Module
]:
    """Exact seed-158 recipe; returns owning model with eligible modules."""
    torch.manual_seed(158)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {"proj": model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        "proj",
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {"proj": tensor_state},
        device="cpu",
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        captures = handle.captures["proj"]
    weight_shape = tuple(int(dim) for dim in tensor_state.q_levels.shape)
    return captures, q.reshape(-1), weight_shape, eligible, model


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tensor_sequence(tensors: Sequence[torch.Tensor]) -> str:
    digests = [compute_tensor_canonical_sha256(t) for t in tensors]
    return compute_canonical_json_sha256(digests)


def build_governing_receipt(
    *,
    plan_path: Path,
    argv: list[str],
) -> dict[str, Any]:
    plan_sha = _sha_file(plan_path)
    if plan_sha != PLAN_SHA256_EXPECTED:
        raise ValueError(
            f"plan sha mismatch: expected={PLAN_SHA256_EXPECTED} actual={plan_sha}"
        )

    captures, q_flat, weight_shape, eligible, model = _dry_run_fixture()
    # Fold-B same-model ownership: optimizer_checks MUST use the fixture owner.
    assert_eligible_modules_owned_by_model(model, eligible)
    _opt, optimizer_checks = build_optimizer_excluding_eligible_masters(
        model, eligible, lr=0.0
    )

    # Single observation call — evidence is the sole source for projected_moves_* .
    # Do NOT call projected_moves_from_integer_attribution again for receipt fields.
    evidence = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    if evidence.probe_mode != "alloc_guard_instrumented":
        raise ValueError(f"unexpected probe_mode: {evidence.probe_mode}")

    audit = run_optimizer_credit_state_no_hidden_fp_audit(
        optimizer_checks=optimizer_checks,
        observed_dense_surfaces=evidence.observed_surfaces,
        observation_probe_mode=evidence.probe_mode,
        audit_observation_complete=True,
    )
    if audit.branch_id == BRANCH_3C_C_AUDIT_PENDING:
        raise ValueError("PENDING forbidden as terminal")

    pev = evidence.projected_moves_evidence
    observed_surfaces_sha256 = compute_canonical_json_sha256(
        list(evidence.observed_surfaces)
    )
    optimizer_checks_payload = {
        "eligible_params_in_optimizer": int(
            optimizer_checks["eligible_params_in_optimizer"]
        ),
        "eligible_optimizer_state_entries": int(
            optimizer_checks["eligible_optimizer_state_entries"]
        ),
        "pass": bool(optimizer_checks.get("pass", False)),
    }
    optimizer_checks_sha256 = compute_canonical_json_sha256(optimizer_checks_payload)
    runtime_command_sha256 = compute_canonical_json_sha256(list(argv))
    audit_receipt_sha256 = (
        compute_optimizer_credit_state_no_hidden_fp_audit_receipt_sha256(audit)
    )

    receipt: dict[str, Any] = {
        "schema": (
            "hrm_text_158_optimizer_credit_state_3C_readonly_classification_audit/v1"
        ),
        "plan_path": str(plan_path.as_posix()),
        "plan_sha256": plan_sha,
        "fixture_recipe_name": FIXTURE_RECIPE_NAME,
        "parity_fixture_descriptor_sha256": (
            OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256
        ),
        "captures_inputs_sha256": _hash_tensor_sequence(captures["inputs"]),
        "captures_grad_outputs_sha256": _hash_tensor_sequence(
            captures["grad_outputs"]
        ),
        "q_flat_sha256": compute_tensor_canonical_sha256(q_flat),
        "weight_shape": list(weight_shape),
        # Bound FROM evidence — never recomputed.
        "projected_moves_sha256": pev.projected_moves_sha256,
        "projected_moves_numel": pev.projected_moves_numel,
        "projected_moves_dtype": pev.projected_moves_dtype,
        "projected_moves_shape": list(pev.projected_moves_shape),
        "move_indices_sha256": pev.move_indices_sha256,
        "move_indices_dtype": pev.move_indices_dtype,
        "move_indices_numel": pev.move_indices_numel,
        "observed_dense_surfaces": list(evidence.observed_surfaces),
        "observed_surfaces_sha256": observed_surfaces_sha256,
        "optimizer_checks": optimizer_checks_payload,
        "optimizer_checks_sha256": optimizer_checks_sha256,
        "probe_mode": evidence.probe_mode,
        "audit_branch_id": audit.branch_id,
        "audit_receipt_sha256": audit_receipt_sha256,
        "runtime_command_argv": list(argv),
        "runtime_command_sha256": runtime_command_sha256,
        "fold_C_deferred": True,
        "observation_evidence_type": "IntegerPathDenseSurfaceObservationEvidence",
        "projected_moves_sha256_sourced_from_evidence_not_recomputed": True,
        "optimizer_checks_model_ownership": (
            "same_fixture_model: eligible modules are params of the returned "
            "_dry_run_fixture() model; assert_eligible_modules_owned_by_model "
            "enforced before build_optimizer_excluding_eligible_masters"
        ),
        "expected_branch_seed158": BRANCH_3C_C_DENSE_LEAK,
        "claim_ceiling": {
            "may_claim": [
                "CPU measurement-bound observations over four debt objects via "
                "frozen harness"
            ],
            "must_not_claim": [
                "readiness-row flip",
                "sub2 / resolved=true",
                "full_sub2_runtime_ready_for_science",
            ],
        },
        "required_tokens": [
            "AUDIT_NO_DENSE_PROJECTED_MOVES",
            "DENSE_LEAK_EXPECTED_SEED158",
            "PENDING_FORBIDDEN_AS_TERMINAL",
            "FOLD_C_DEFERRED",
            "MEASUREMENT_BINDING_PRESENT",
            "HOSTILE_HAND_AUTHORED_EMPTY_OBSERVATIONS_REJECT",
            "HOSTILE_ROW_FLIP_REJECT",
            "EVIDENCE_SEAM_PRESENT",
            "EVIDENCE_RECEIPT_EQUALITY",
            "HOSTILE_RECOMPUTE_REJECT",
            "HOSTILE_HASH_DRIFT_REJECT",
            "CANONICAL_HASH_CONTRACT",
        ],
    }

    # Full field census must pass BEFORE emitting the validated boolean.
    validate_evidence_receipt_field_equality(evidence, receipt)
    receipt["evidence_to_receipt_field_equality_validated"] = True

    if audit.branch_id != BRANCH_3C_C_DENSE_LEAK:
        raise ValueError(
            f"seed-158 expected BR-3C-C-DENSE-LEAK, got {audit.branch_id}"
        )

    return receipt


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Governing 3C readonly classification audit (PLAN_v5)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv_list[1:])

    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan

    frozen_argv = [
        "python3",
        "scripts/optimizer_credit_state_3C_readonly_audit_run.py",
        "--out",
        str(args.out),
        "--plan",
        str(args.plan),
    ]

    receipt = build_governing_receipt(plan_path=plan_path, argv=frozen_argv)

    if out_path.exists():
        raise FileExistsError(f"O_EXCL: receipt already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with open(out_path, "x", encoding="utf-8") as fh:
        fh.write(payload)

    sys.stdout.write(
        json.dumps(
            {
                "audit_branch_id": receipt["audit_branch_id"],
                "projected_moves_numel": receipt["projected_moves_numel"],
                "out": str(out_path),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
