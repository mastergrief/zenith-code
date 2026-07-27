#!/usr/bin/env python3
"""Fold C drain-path classification audit harness (PLAN_v3).

Measurement-only: seed-158 fixture → weighted_grad → credit → FP vs integer
projection compare + dense-surface bind → taxonomy branch. No drain fix,
no readiness/sub2 flip, no creditdir writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (  # noqa: E402
    credit_from_weighted_grad,
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (  # noqa: E402
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state_no_hidden_fp_audit import (  # noqa: E402
    compute_canonical_json_sha256,
    compute_tensor_canonical_sha256,
    run_integer_path_dense_surface_observation_with_alloc_guard,
)
from hrm_text_158_credit_bridge import (  # noqa: E402
    project_fp_gradient_to_moves,
    project_integer_credit_to_moves,
)

FIXTURE_RECIPE_NAME = "3C_C1_dry_run_fixture_seed158"
CREDIT_SCHEME = "full_magnitude_ceiling"
PROBE_MODE = "fold_c_drain_path_cpu_projection"
PLAN_SHA256_EXPECTED = (
    "b5c6409cf0a566f54395b6452c867f6de2a84ab9292096e7fb3e3b9d2630bcbe"
)
BACKEND_DECLARED_CONTRACT = "pytorch_eager_cpu"

BRANCH_PARITY_HOLDS_DENSE_REMAINS = "BR-FOLD-C-PARITY-HOLDS-DENSE-REMAINS"
BRANCH_FRACTIONAL_COLLISION = "BR-FOLD-C-FRACTIONAL-COLLISION"
BRANCH_MEASUREMENT_INVALID = "BR-FOLD-C-MEASUREMENT-INVALID"
BRANCH_NATIVE_INCOMPLETE = "BR-FOLD-C-NATIVE-PATH-SIGNAL-INCOMPLETE"
BRANCH_PENDING = "BR-FOLD-C-PENDING"

TERMINAL_ALLOWED = frozenset(
    {
        BRANCH_PARITY_HOLDS_DENSE_REMAINS,
        BRANCH_FRACTIONAL_COLLISION,
        BRANCH_MEASUREMENT_INVALID,
        BRANCH_NATIVE_INCOMPLETE,
    }
)

RECOMMENDED_NEXT = {
    BRANCH_PARITY_HOLDS_DENSE_REMAINS: (
        "dense-surface elimination / re-carriering measurement contract "
        "(NOT rotor ride-along; NOT flip)"
    ),
    BRANCH_FRACTIONAL_COLLISION: (
        "update-law pivot measurement contract "
        "(credit_ranking_update_law_pivot_deferred debt anchor)"
    ),
    BRANCH_MEASUREMENT_INVALID: "remint or repair measurement seam",
    BRANCH_NATIVE_INCOMPLETE: "incomplete native-path proof — do not invent discharge",
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_3c_harness():
    path = REPO_ROOT / "scripts" / "optimizer_credit_state_3C_readonly_audit_run.py"
    spec = importlib.util.spec_from_file_location(
        "optimizer_credit_state_3C_readonly_audit_run", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load 3C harness at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _device_str_from_tensor(tensor: torch.Tensor) -> str:
    """Observed device from EXACT tensor — never torch.get_default_device()."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected torch.Tensor for observed_device derivation")
    return str(tensor.device)


def _require_exact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be exact int (bool rejected); got {type(value)}")
    return int(value)


@dataclass(frozen=True)
class FoldCDrainPathEvidence:
    fixture_recipe_name: str
    parity_fixture_descriptor_sha256: str
    weighted_grad_sha256: str
    weighted_grad_dtype: str
    weighted_grad_shape: tuple[int, ...]
    weighted_grad_numel: int
    credit_sha256: str
    credit_dtype: str
    credit_shape: tuple[int, ...]
    credit_numel: int
    credit_scheme: str
    fp_moves_sha256: str
    integer_moves_sha256: str
    mismatch_count: int
    mismatch_frac: float
    observed_dense_surfaces: tuple[str, ...]
    observed_surfaces_sha256: str
    probe_mode: str
    repo_head_sha: str
    observed_device: str
    observed_devices_per_tensor: tuple[tuple[str, str], ...]
    branch_id: str
    recommended_next_slice: str

    def to_receipt_fields(self) -> dict[str, Any]:
        return {
            "fixture_recipe_name": self.fixture_recipe_name,
            "parity_fixture_descriptor_sha256": self.parity_fixture_descriptor_sha256,
            "weighted_grad_sha256": self.weighted_grad_sha256,
            "weighted_grad_dtype": self.weighted_grad_dtype,
            "weighted_grad_shape": list(self.weighted_grad_shape),
            "weighted_grad_numel": self.weighted_grad_numel,
            "credit_sha256": self.credit_sha256,
            "credit_dtype": self.credit_dtype,
            "credit_shape": list(self.credit_shape),
            "credit_numel": self.credit_numel,
            "credit_scheme": self.credit_scheme,
            "fp_moves_sha256": self.fp_moves_sha256,
            "integer_moves_sha256": self.integer_moves_sha256,
            "mismatch_count": self.mismatch_count,
            "mismatch_frac": self.mismatch_frac,
            "observed_dense_surfaces": list(self.observed_dense_surfaces),
            "observed_surfaces_sha256": self.observed_surfaces_sha256,
            "probe_mode": self.probe_mode,
            "repo_head_sha": self.repo_head_sha,
            "observed_device": self.observed_device,
            "observed_devices_per_tensor": {
                k: v for k, v in self.observed_devices_per_tensor
            },
            "audit_branch_id": self.branch_id,
            "recommended_next_slice": self.recommended_next_slice,
            "observation_evidence_type": "FoldCDrainPathEvidence",
        }


def validate_mismatch_count_exact_int(value: Any) -> int:
    return _require_exact_int(value, field="mismatch_count")


def validate_credit_scheme_exact(scheme: str) -> None:
    if scheme != CREDIT_SCHEME:
        raise ValueError(
            f"credit_scheme drift: expected={CREDIT_SCHEME!r} actual={scheme!r}"
        )


def validate_observed_device_uniformity(
    devices: Mapping[str, str],
) -> str:
    if not devices:
        raise ValueError("observed device map empty")
    values = sorted(set(devices.values()))
    if len(values) != 1:
        raise ValueError(
            f"device non-uniform across bound tensors: {dict(devices)}"
        )
    return values[0]


def validate_observed_device_from_bound_tensors_not_literal(
    *,
    observed_device: str,
    devices_from_tensors: Mapping[str, str],
) -> None:
    uniform = validate_observed_device_uniformity(devices_from_tensors)
    if observed_device != uniform:
        raise ValueError(
            "observed_device must equal uniformity result from bound tensors; "
            f"claimed={observed_device!r} from_tensors={uniform!r}"
        )


def validate_live_repo_head_matches_claim(*, claimed: str, live: str) -> None:
    if claimed != live:
        raise ValueError(
            f"live repo HEAD mismatch: claimed={claimed} live={live}"
        )


def validate_dependency_currency_against_plan_pins(
    *,
    plan: Mapping[str, Any],
    repo_root: Path,
) -> None:
    freeze = plan["dependency_currency_freeze"]
    files = freeze["files"]
    if int(freeze["pinned_file_count"]) != len(files):
        raise ValueError("dependency_currency pinned_file_count mismatch")
    for entry in files:
        path = repo_root / entry["path"]
        actual = _sha_file(path)
        expected = entry["expected_sha256"]
        if actual != expected:
            raise ValueError(
                f"dependency_currency drift on {entry['path']}: "
                f"expected={expected} actual={actual}"
            )


def validate_observed_dense_surfaces_sha256(
    surfaces: Sequence[str], claimed: str
) -> None:
    actual = compute_canonical_json_sha256(list(surfaces))
    if actual != claimed:
        raise ValueError(
            f"observed_surfaces_sha256 mismatch: claimed={claimed} actual={actual}"
        )


def validate_fold_c_evidence_receipt_field_equality(
    evidence: FoldCDrainPathEvidence,
    receipt: Mapping[str, Any],
) -> None:
    expected = evidence.to_receipt_fields()
    for key, value in expected.items():
        if key not in receipt:
            raise ValueError(f"receipt missing evidence-bound field: {key}")
        if receipt[key] != value:
            raise ValueError(
                f"evidence↔receipt inequality on {key}: "
                f"evidence={value!r} receipt={receipt[key]!r}"
            )


def classify_fold_c_branch(
    *,
    mismatch_count: int,
    observed_dense_surfaces: Sequence[str],
    measurement_valid: bool,
    native_path_signal_incomplete: bool = False,
) -> str:
    mismatch_count = validate_mismatch_count_exact_int(mismatch_count)
    if not measurement_valid:
        return BRANCH_MEASUREMENT_INVALID
    if native_path_signal_incomplete:
        return BRANCH_NATIVE_INCOMPLETE
    if mismatch_count > 0:
        return BRANCH_FRACTIONAL_COLLISION
    if "projected_moves" in set(observed_dense_surfaces):
        return BRANCH_PARITY_HOLDS_DENSE_REMAINS
    return BRANCH_MEASUREMENT_INVALID


def _live_repo_head(repo_root: Path) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        text=True,
    )
    return out.strip()


def collect_fold_c_evidence(
    *,
    repo_root: Path,
    credit_scheme: str = CREDIT_SCHEME,
) -> FoldCDrainPathEvidence:
    validate_credit_scheme_exact(credit_scheme)
    harness_3c = _load_3c_harness()
    captures, q_flat, weight_shape, _eligible, _model = harness_3c._dry_run_fixture()

    weighted_grad = weighted_grad_from_captures(
        captures["inputs"],
        captures["grad_outputs"],
        weight_shape=weight_shape,
    )
    credit = credit_from_weighted_grad(weighted_grad, scheme=credit_scheme)
    q = q_flat.reshape(weight_shape).to(torch.int8).reshape(-1)
    fp_moves = project_fp_gradient_to_moves(weighted_grad.reshape(-1), q)
    integer_moves = project_integer_credit_to_moves(credit.reshape(-1), q)

    # Dense surfaces from SAME fixture observation (single call).
    dense_ev = run_integer_path_dense_surface_observation_with_alloc_guard(
        captures=captures,
        weight_shape=weight_shape,
        q_flat=q_flat,
    )
    observed_surfaces = tuple(dense_ev.observed_surfaces)
    observed_surfaces_sha256 = compute_canonical_json_sha256(list(observed_surfaces))

    mismatch_mask = fp_moves != integer_moves
    mismatch_count = validate_mismatch_count_exact_int(int(mismatch_mask.sum().item()))
    mismatch_frac = float(mismatch_count) / float(int(fp_moves.numel()))

    devices = {
        "weighted_grad": _device_str_from_tensor(weighted_grad),
        "credit": _device_str_from_tensor(credit),
        "fp_moves": _device_str_from_tensor(fp_moves),
        "integer_moves": _device_str_from_tensor(integer_moves),
    }
    observed_device = validate_observed_device_uniformity(devices)
    validate_observed_device_from_bound_tensors_not_literal(
        observed_device=observed_device,
        devices_from_tensors=devices,
    )

    repo_head = _live_repo_head(repo_root)
    branch_id = classify_fold_c_branch(
        mismatch_count=mismatch_count,
        observed_dense_surfaces=observed_surfaces,
        measurement_valid=True,
    )
    if branch_id == BRANCH_PENDING or branch_id not in TERMINAL_ALLOWED:
        raise ValueError(f"PENDING/non-terminal forbidden: {branch_id}")

    return FoldCDrainPathEvidence(
        fixture_recipe_name=FIXTURE_RECIPE_NAME,
        parity_fixture_descriptor_sha256=(
            OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256
        ),
        weighted_grad_sha256=compute_tensor_canonical_sha256(weighted_grad),
        weighted_grad_dtype=str(weighted_grad.dtype),
        weighted_grad_shape=tuple(int(d) for d in weighted_grad.shape),
        weighted_grad_numel=int(weighted_grad.numel()),
        credit_sha256=compute_tensor_canonical_sha256(credit),
        credit_dtype=str(credit.dtype),
        credit_shape=tuple(int(d) for d in credit.shape),
        credit_numel=int(credit.numel()),
        credit_scheme=credit_scheme,
        fp_moves_sha256=compute_tensor_canonical_sha256(fp_moves),
        integer_moves_sha256=compute_tensor_canonical_sha256(integer_moves),
        mismatch_count=mismatch_count,
        mismatch_frac=mismatch_frac,
        observed_dense_surfaces=observed_surfaces,
        observed_surfaces_sha256=observed_surfaces_sha256,
        probe_mode=PROBE_MODE,
        repo_head_sha=repo_head,
        observed_device=observed_device,
        observed_devices_per_tensor=tuple(sorted(devices.items())),
        branch_id=branch_id,
        recommended_next_slice=RECOMMENDED_NEXT[branch_id],
    )


def build_governing_receipt(
    *,
    plan_path: Path,
    argv: list[str],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    plan_sha = _sha_file(plan_path)
    if plan_sha != PLAN_SHA256_EXPECTED:
        raise ValueError(
            f"plan sha mismatch: expected={PLAN_SHA256_EXPECTED} actual={plan_sha}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_dependency_currency_against_plan_pins(plan=plan, repo_root=repo_root)

    evidence = collect_fold_c_evidence(repo_root=repo_root)
    validate_live_repo_head_matches_claim(
        claimed=evidence.repo_head_sha,
        live=_live_repo_head(repo_root),
    )
    validate_observed_dense_surfaces_sha256(
        evidence.observed_dense_surfaces, evidence.observed_surfaces_sha256
    )
    validate_credit_scheme_exact(evidence.credit_scheme)

    receipt: dict[str, Any] = {
        "schema": "hrm_text_158_optimizer_credit_state_fold_c_drain_path_audit/v1",
        "plan_path": str(plan_path.as_posix()),
        "plan_sha256": plan_sha,
        "plan_revision": "v3",
        **evidence.to_receipt_fields(),
        "backend": {
            "field_type": "declared_contract",
            "value": BACKEND_DECLARED_CONTRACT,
        },
        "runtime_command_argv": list(argv),
        "runtime_command_sha256": compute_canonical_json_sha256(list(argv)),
        "expected_branch_seed158": BRANCH_PARITY_HOLDS_DENSE_REMAINS,
        "transient_fp_debt_remains": True,
        "claim_ceiling": {
            "may_claim": list(plan["claim_ceiling"]["may_claim"]),
            "must_not_claim": list(plan["claim_ceiling"]["must_not_claim"]),
            "transient_fp_debt_remains": True,
            "no_readiness_row_flip": True,
            "authorizes_readiness_row_flip": False,
            "authorizes_sub2_or_resolved": False,
        },
        "required_tokens": [
            "FOLD_C_DRAIN_PATH_MEASUREMENT_ONLY",
            "OBSERVED_DEVICE_FROM_BOUND_TENSORS",
            "BACKEND_DECLARED_CONTRACT",
            "PENDING_FORBIDDEN_AS_TERMINAL",
            "TRANSIENT_FP_DEBT_REMAINS",
            "NO_READINESS_ROW_FLIP",
            "EVIDENCE_SEAM_PRESENT",
            "EVIDENCE_RECEIPT_EQUALITY",
            "HOSTILE_LITERAL_DEVICE_REJECT",
            "HOSTILE_RECOMPUTE_REJECT",
            "UPDATE_EMPTY",
            "CREDITDIR_WRITES_OUT",
        ],
    }
    validate_fold_c_evidence_receipt_field_equality(evidence, receipt)
    receipt["evidence_to_receipt_field_equality_validated"] = True
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    argv_list = [
        "python",
        "scripts/optimizer_credit_state_fold_c_drain_path_audit_run.py",
        "--plan",
        str(args.plan.as_posix()),
        "--out",
        str(args.out.as_posix()),
    ]
    plan_path = args.plan
    if not plan_path.is_absolute():
        plan_path = (REPO_ROOT / plan_path).resolve()
    receipt = build_governing_receipt(plan_path=plan_path, argv=argv_list)

    out_path = args.out
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if out_path.exists():
        raise FileExistsError(f"O_EXCL refused; receipt exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_branch_id": receipt["audit_branch_id"],
                "mismatch_count": receipt["mismatch_count"],
                "observed_device": receipt["observed_device"],
                "transient_fp_debt_remains": receipt["transient_fp_debt_remains"],
                "out": str(out_path.as_posix()),
                "out_sha256": _sha_file(out_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
