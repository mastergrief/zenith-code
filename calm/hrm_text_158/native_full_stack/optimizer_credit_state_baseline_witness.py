"""Read-only CPU baseline inventory + witness for optimizer_credit_state (3C-C).

Produces baseline FP debt inventory and external feasibility witnesses only.
Does NOT set BR-3C-C-AUDIT-PASS-CPU, row flip, or GPU receipt flags.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    build_optimizer_excluding_eligible_masters,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    project_s1_gradient_to_moves,
    prove_eligible_master_identity_after_optimizer_step,
    rank_bucketed_int16_votes,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    BRANCH_3C_C_CAPTURE_LAUNDER,
    BRANCH_3C_C_MEASUREMENT_INVALID,
    BRANCH_3C_C_OPT_EXCL_FAIL,
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR,
    OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
)

OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_SCHEMA_VERSION = (
    "hrm_text_158_optimizer_credit_state_baseline_witness/v1"
)
OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_TARGET_NAME = (
    "optimizer_credit_state_baseline_witness_cpu"
)

BRANCH_BASELINE_WITNESS_A_GREEN = "BR-3C-C-BASELINE-WITNESS-A-GREEN"
BRANCH_FEASIBILITY_WITNESS_FAIL = "BR-3C-C-FEASIBILITY-WITNESS-FAIL"

REGISTERED_BASELINE_WITNESS_BRANCH_IDS = frozenset(
    {
        BRANCH_BASELINE_WITNESS_A_GREEN,
        BRANCH_3C_C_OPT_EXCL_FAIL,
        BRANCH_3C_C_CAPTURE_LAUNDER,
        BRANCH_FEASIBILITY_WITNESS_FAIL,
        BRANCH_3C_C_MEASUREMENT_INVALID,
    }
)

INVENTORY_ANCHOR_WEIGHTED_GRAD = "weighted_grad"
INVENTORY_ANCHOR_CREDIT = "credit"
INVENTORY_ANCHOR_PROJECTED_MOVES = "projected_moves"
INVENTORY_ANCHOR_DENSE_RANK_VOTES = "dense_rank_votes_before_sparse_event_extraction"

INVENTORY_ANCHOR_NAMES = (
    INVENTORY_ANCHOR_WEIGHTED_GRAD,
    INVENTORY_ANCHOR_CREDIT,
    INVENTORY_ANCHOR_PROJECTED_MOVES,
    INVENTORY_ANCHOR_DENSE_RANK_VOTES,
)

# projected_moves is inventory-only — not an AUDIT-NO-DENSE-* alloc-guard surface.
ALLOC_GUARD_DENSE_SURFACE_NAMES = frozenset(
    {
        INVENTORY_ANCHOR_WEIGHTED_GRAD,
        INVENTORY_ANCHOR_CREDIT,
        INVENTORY_ANCHOR_DENSE_RANK_VOTES,
    }
)

BASELINE_WITNESS_NON_CLAIMS = (
    "baseline_current_debt_inventory observes expected FP dense debt on the current path; not a pass",
    "optimizer_state_exclusion_observation is an observation only; not optimizer_state_eligible_exclusion_proven",
    "projection_equivalence_feasibility_witness is external CPU feasibility evidence only",
    "does not set BR-3C-C-AUDIT-PASS-CPU or authorize optimizer_credit_state row flip",
    "does not launch GPU, mutate .pt artifacts, or claim persistent carrier-width reduction",
    "two_tier_persistent_state_ledger.py names the 3-ledger accounting source only",
)

FORBIDDEN_RECEIPT_FIELDS = (
    "br_3c_c_audit_pass_cpu",
    "optimizer_state_eligible_exclusion_proven",
    "optimizer_credit_state_sub2_claim",
    "readiness_row_flip_authorized",
    "gpu_runtime_receipt_present",
    "real_native_integer_attribution_present",
    "real_native_integer_credit_ranking_present",
    "ready_to_flip",
    "persistent_carrier_width_claim",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_transient_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _tensor_inventory_row(
    anchor_name: str,
    tensor: torch.Tensor,
    *,
    lifetime: str,
    module_key: str,
) -> dict[str, Any]:
    return {
        "module_key": module_key,
        "anchor_name": anchor_name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "peak_transient_bytes": _peak_transient_bytes(tensor),
        "lifetime": lifetime,
        "device": str(tensor.device),
    }


def _import_credit_bridge():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "hrm_text_158_credit_bridge.py"
    module_name = "_optimizer_credit_state_baseline_witness_credit_bridge"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import credit bridge from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class DebtInventoryModuleRow:
    module_key: str
    anchors: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_key": self.module_key,
            "anchors": list(self.anchors),
        }


@dataclass(frozen=True)
class BaselineCurrentDebtInventory:
    inventory_complete: bool
    per_module_rows: tuple[DebtInventoryModuleRow, ...]
    observed_anchor_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_complete": self.inventory_complete,
            "per_module_rows": [row.to_dict() for row in self.per_module_rows],
            "observed_anchor_names": list(self.observed_anchor_names),
        }


@dataclass(frozen=True)
class OptimizerStateExclusionObservation:
    observation_holds: bool
    checkpoint_sha256_before: str
    checkpoint_sha256_after: str
    eligible_state_summary: dict[str, Any]
    capture_laundering_signal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_state_exclusion_observation": True,
            "observation_holds": self.observation_holds,
            "checkpoint_sha256_before": self.checkpoint_sha256_before,
            "checkpoint_sha256_after": self.checkpoint_sha256_after,
            "eligible_state_summary": dict(self.eligible_state_summary),
            "capture_laundering_signal": self.capture_laundering_signal,
        }


@dataclass(frozen=True)
class ProjectionEquivalenceFeasibilityWitness:
    passed: bool
    cases_run: int
    zero_revival_exercised: bool
    checkpoint_sha256_before: str
    checkpoint_sha256_after: str
    mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_equivalence_feasibility_witness": True,
            "passed": self.passed,
            "cases_run": self.cases_run,
            "zero_revival_exercised": self.zero_revival_exercised,
            "checkpoint_sha256_before": self.checkpoint_sha256_before,
            "checkpoint_sha256_after": self.checkpoint_sha256_after,
            "mismatch_count": self.mismatch_count,
        }


@dataclass(frozen=True)
class OptimizerCreditStateBaselineWitnessReceipt:
    schema_version: str
    target_name: str
    fixture_descriptor: dict[str, str]
    baseline_current_debt_inventory: BaselineCurrentDebtInventory
    optimizer_state_exclusion_observation: OptimizerStateExclusionObservation
    projection_equivalence_feasibility_witness: ProjectionEquivalenceFeasibilityWitness
    branch_classifier: str
    branch_evidence: str
    non_claims: tuple[str, ...]
    br_3c_c_audit_pass_cpu: bool = False
    optimizer_state_eligible_exclusion_proven: bool = False
    optimizer_credit_state_sub2_claim: bool = False
    readiness_row_flip_authorized: bool = False
    gpu_runtime_receipt_present: bool = False
    real_native_integer_attribution_present: bool = False
    real_native_integer_credit_ranking_present: bool = False
    ready_to_flip: bool = False
    persistent_carrier_width_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "fixture_descriptor": dict(self.fixture_descriptor),
            "baseline_current_debt_inventory": self.baseline_current_debt_inventory.to_dict(),
            "optimizer_state_exclusion_observation": (
                self.optimizer_state_exclusion_observation.to_dict()
            ),
            "projection_equivalence_feasibility_witness": (
                self.projection_equivalence_feasibility_witness.to_dict()
            ),
            "branch_classifier": self.branch_classifier,
            "branch_evidence": self.branch_evidence,
            "non_claims": list(self.non_claims),
            **{field: getattr(self, field) for field in FORBIDDEN_RECEIPT_FIELDS},
        }


def collect_baseline_current_debt_inventory(
    *,
    module_key: str = "proj",
    device: str = "cpu",
) -> BaselineCurrentDebtInventory:
    from calm.hrm_text_158.bit_linear import BitLinear

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(3, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    torch.manual_seed(158)
    model = _Tiny().to(device)
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = {module_key: model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        module_key,
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32, device=device)
    target = torch.tensor([[2.0, -1.0]], dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)
    with authoritative_forward_context(
        eligible,
        {module_key: tensor_state},
        device=device,
        requires_grad=True,
    ) as handle:
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        weighted_grad = handle.weighted_grad(module_key)
    credit = credit_from_weighted_grad(weighted_grad)
    projected_moves = project_s1_gradient_to_moves(weighted_grad, tensor_state.q_levels)
    rank_spec = default_dry_run_rank_vote_spec()
    dense_rank_votes = rank_bucketed_int16_votes(credit, projected_moves, rank_spec)

    anchors = (
        _tensor_inventory_row(
            INVENTORY_ANCHOR_WEIGHTED_GRAD,
            weighted_grad,
            lifetime="step_local_fp_path",
            module_key=module_key,
        ),
        _tensor_inventory_row(
            INVENTORY_ANCHOR_CREDIT,
            credit,
            lifetime="step_local_fp_path",
            module_key=module_key,
        ),
        _tensor_inventory_row(
            INVENTORY_ANCHOR_PROJECTED_MOVES,
            projected_moves,
            lifetime="step_local_fp_path",
            module_key=module_key,
        ),
        _tensor_inventory_row(
            INVENTORY_ANCHOR_DENSE_RANK_VOTES,
            dense_rank_votes,
            lifetime="step_local_fp_path",
            module_key=module_key,
        ),
    )
    observed = tuple(row["anchor_name"] for row in anchors)
    inventory_complete = observed == INVENTORY_ANCHOR_NAMES and all(
        int(row["peak_transient_bytes"]) > 0 for row in anchors
    )
    return BaselineCurrentDebtInventory(
        inventory_complete=inventory_complete,
        per_module_rows=(DebtInventoryModuleRow(module_key=module_key, anchors=anchors),),
        observed_anchor_names=observed,
    )


def run_optimizer_state_exclusion_observation(
    *,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> OptimizerStateExclusionObservation:
    from calm.hrm_text_158.bit_linear import BitLinear

    owned = checkpoint_path is None
    if checkpoint_path is None:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        handle.write(b"optimizer_credit_state_baseline_witness_checkpoint_v1")
        handle.flush()
        handle.close()
        checkpoint_path = Path(handle.name)
    sha_before = sha256_file(checkpoint_path)

    torch.manual_seed(158)
    module_key = "proj"

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(3, 2, bias=False)
            self.noneligible = torch.nn.Parameter(torch.ones((), dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = _Tiny().to(device)
    eligible = {module_key: model.proj}
    q = torch.zeros_like(model.proj.weight.detach(), dtype=torch.int8)
    tensor_state = make_bounded_tensor_state(
        module_key,
        q,
        torch.tensor(1.0, dtype=torch.float32),
        hot_exact_indices=tuple(range(int(q.numel()))),
    )
    optimizer, checks = build_optimizer_excluding_eligible_masters(model, eligible)
    if optimizer is not None:
        model.noneligible.grad = torch.ones_like(model.noneligible)
    identity_proof = prove_eligible_master_identity_after_optimizer_step(
        optimizer,
        eligible,
        optimizer_checks=checks,
    )
    vote_sidecar = tensor_state.decoded_accumulators()
    vote_sidecar_finite = bool(torch.isfinite(vote_sidecar.to(torch.float32)).all().item())
    q_sidecar_finite = bool(torch.isfinite(tensor_state.q_levels.to(torch.float32)).all().item())
    capture_laundering_signal = bool(
        checks.get("eligible_params_in_optimizer", -1) != 0
        and checks.get("eligible_weight_requires_grad_for_transient_credit_capture", False)
    )
    observation_holds = bool(
        checks.get("pass", False)
        and identity_proof.get("eligible_master_identity_pass", False)
        and vote_sidecar_finite
        and q_sidecar_finite
    )
    sha_after = sha256_file(checkpoint_path)
    if owned:
        checkpoint_path.unlink(missing_ok=True)
    return OptimizerStateExclusionObservation(
        observation_holds=observation_holds and sha_before == sha_after,
        checkpoint_sha256_before=sha_before,
        checkpoint_sha256_after=sha_after,
        eligible_state_summary={
            "optimizer_checks": dict(checks),
            "eligible_master_identity_proof": identity_proof,
            "vote_sidecar_finite": vote_sidecar_finite,
            "q_sidecar_finite": q_sidecar_finite,
            "ledger_reference": "two_tier_persistent_state_ledger.py:22-46",
        },
        capture_laundering_signal=capture_laundering_signal,
    )


def run_projection_equivalence_feasibility_witness(
    *,
    checkpoint_path: Path | None = None,
) -> ProjectionEquivalenceFeasibilityWitness:
    owned = checkpoint_path is None
    if checkpoint_path is None:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        handle.write(b"projection_equivalence_feasibility_witness_v1")
        handle.flush()
        handle.close()
        checkpoint_path = Path(handle.name)
    sha_before = sha256_file(checkpoint_path)

    bridge = _import_credit_bridge()
    cases = (
        (
            torch.tensor([[-1, -1, 0, 0, 1, 1]], dtype=torch.int8),
            torch.tensor([[-2.0, 3.0, -0.5, 0.25, -4.0, 6.0]]),
            torch.tensor([[4, -3, 5, -6, 7, -8]], dtype=torch.int32),
            True,
        ),
        (
            torch.tensor([[-1, 0, 1]], dtype=torch.int8),
            torch.tensor([[-2.0, 3.0, 4.0]]),
            None,
            False,
        ),
    )
    mismatch_count = 0
    zero_revival_exercised = False
    for q, grad, credit, zero_case in cases:
        fp_moves = bridge.project_fp_gradient_to_moves(grad, q)
        if credit is None:
            int_moves = bridge.project_integer_credit_to_moves(-grad, q)
        else:
            int_moves = bridge.project_integer_credit_to_moves(credit, q)
            zero_revival_exercised = zero_revival_exercised or zero_case
        if not torch.equal(fp_moves, int_moves):
            mismatch_count += 1

    sha_after = sha256_file(checkpoint_path)
    if owned:
        checkpoint_path.unlink(missing_ok=True)
    return ProjectionEquivalenceFeasibilityWitness(
        passed=mismatch_count == 0 and sha_before == sha_after,
        cases_run=len(cases),
        zero_revival_exercised=zero_revival_exercised,
        checkpoint_sha256_before=sha_before,
        checkpoint_sha256_after=sha_after,
        mismatch_count=mismatch_count,
    )


def classify_optimizer_credit_state_baseline_witness_branch(
    *,
    inventory: BaselineCurrentDebtInventory,
    exclusion: OptimizerStateExclusionObservation,
    projection: ProjectionEquivalenceFeasibilityWitness,
) -> tuple[str, str]:
    if not inventory.inventory_complete:
        return (
            BRANCH_3C_C_MEASUREMENT_INVALID,
            "baseline_current_debt_inventory incomplete or unbounded",
        )
    if projection.cases_run <= 0:
        return (
            BRANCH_3C_C_MEASUREMENT_INVALID,
            "projection_equivalence_feasibility_witness did not run bounded cases",
        )
    if not projection.passed:
        return (
            BRANCH_FEASIBILITY_WITNESS_FAIL,
            f"projection witness mismatch_count={projection.mismatch_count}",
        )
    if not exclusion.observation_holds:
        if exclusion.capture_laundering_signal:
            return (
                BRANCH_3C_C_CAPTURE_LAUNDER,
                "optimizer_state_exclusion_observation failed with capture-laundering signal",
            )
        return (
            BRANCH_3C_C_OPT_EXCL_FAIL,
            "optimizer_state_exclusion_observation failed on eligible optimizer exclusion",
        )
    return (
        BRANCH_BASELINE_WITNESS_A_GREEN,
        "inventory complete + exclusion observation holds + projection witness passed",
    )


def validate_optimizer_credit_state_baseline_witness_receipt(
    receipt: OptimizerCreditStateBaselineWitnessReceipt,
) -> None:
    if receipt.schema_version != OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_SCHEMA_VERSION:
        raise ValueError("unsupported baseline witness schema_version")
    if receipt.target_name != OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_TARGET_NAME:
        raise ValueError("unexpected baseline witness target_name")
    if receipt.branch_classifier not in REGISTERED_BASELINE_WITNESS_BRANCH_IDS:
        raise ValueError(f"unknown branch_classifier {receipt.branch_classifier!r}")
    for field in FORBIDDEN_RECEIPT_FIELDS:
        if bool(getattr(receipt, field)):
            raise ValueError(f"forbidden receipt field set true: {field}")
    if tuple(receipt.baseline_current_debt_inventory.observed_anchor_names) != INVENTORY_ANCHOR_NAMES:
        raise ValueError("inventory missing required anchor names")
    for required in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS[:4]:
        if required not in receipt.baseline_current_debt_inventory.observed_anchor_names:
            raise ValueError(f"missing inventory anchor {required!r}")
    if (
        receipt.optimizer_state_exclusion_observation.checkpoint_sha256_before
        != receipt.optimizer_state_exclusion_observation.checkpoint_sha256_after
    ):
        raise ValueError("exclusion observation checkpoint sha mismatch")
    if (
        receipt.projection_equivalence_feasibility_witness.checkpoint_sha256_before
        != receipt.projection_equivalence_feasibility_witness.checkpoint_sha256_after
    ):
        raise ValueError("projection witness checkpoint sha mismatch")
    classified, _ = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=receipt.baseline_current_debt_inventory,
        exclusion=receipt.optimizer_state_exclusion_observation,
        projection=receipt.projection_equivalence_feasibility_witness,
    )
    if classified != receipt.branch_classifier:
        raise ValueError(
            "branch_classifier does not match component evidence: "
            f"receipt={receipt.branch_classifier!r} recomputed={classified!r}"
        )


def build_optimizer_credit_state_baseline_witness_receipt(
    *,
    inventory: BaselineCurrentDebtInventory,
    exclusion: OptimizerStateExclusionObservation,
    projection: ProjectionEquivalenceFeasibilityWitness,
) -> OptimizerCreditStateBaselineWitnessReceipt:
    branch, evidence = classify_optimizer_credit_state_baseline_witness_branch(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    return OptimizerCreditStateBaselineWitnessReceipt(
        schema_version=OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_SCHEMA_VERSION,
        target_name=OPTIMIZER_CREDIT_STATE_BASELINE_WITNESS_TARGET_NAME,
        fixture_descriptor=dict(OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR),
        baseline_current_debt_inventory=inventory,
        optimizer_state_exclusion_observation=exclusion,
        projection_equivalence_feasibility_witness=projection,
        branch_classifier=branch,
        branch_evidence=evidence,
        non_claims=BASELINE_WITNESS_NON_CLAIMS,
    )


def run_optimizer_credit_state_baseline_witness(
    *,
    checkpoint_path: Path | None = None,
) -> OptimizerCreditStateBaselineWitnessReceipt:
    inventory = collect_baseline_current_debt_inventory()
    exclusion = run_optimizer_state_exclusion_observation(checkpoint_path=checkpoint_path)
    projection = run_projection_equivalence_feasibility_witness(checkpoint_path=checkpoint_path)
    receipt = build_optimizer_credit_state_baseline_witness_receipt(
        inventory=inventory,
        exclusion=exclusion,
        projection=projection,
    )
    validate_optimizer_credit_state_baseline_witness_receipt(receipt)
    return receipt


def compute_optimizer_credit_state_baseline_witness_receipt_sha256(
    receipt: OptimizerCreditStateBaselineWitnessReceipt,
) -> str:
    return hashlib.sha256(_canonical_json(receipt.to_dict()).encode("utf-8")).hexdigest()
