"""B2-5a MARGIN-selection FEASIBILITY-NULL receipt + packed-key scaffold token.

SCOPE = frozen plan 1781984159546-c894af21, converted to a COMMITTED NULL after
claude gate-1 BLOCK + claude/co_lead direction convergence (1781986822527):
the frozen plan's "native Triton selection" cannot be implemented within this
slice because the standard Triton sort primitives (tl.sort / sort_impl /
tl.topk at triton/language/standard.py:423-470) are value-returning,
block-local, static-shape, and return NO argsort/permutation. A native MARGIN
selection requires a per-row permutation over parallel row tensors (state_ids,
flat_indices, abs, thresholds, directions) — there is no drop-in native
argsort-permutation, and a custom @triton.jit argsort-with-permutation kernel
is its own pre-registered slice (B2-5a'), NOT in-frozen-scope.

The packed-key scan via torch.topk-on-device is RETAINED ONLY as a clearly
labeled NON-NATIVE CPU scaffold that proves the packed-key ORDERING
reproduces the oracle tie-break (DESC abs, ASC global_flat_index). It is
scaffold evidence for the packed-key math, NOT a native path and NOT a GPU
pass. selection_parity_pass is PERMANENTLY False; no native pass is mintable.
index_width_bit_budget_pass may be True (the packed-key bit-budget IS proven),
but it is scoped to the packed-key math, not to native kernelization.

No q/acc mutation, no ledger edit, no global_cap_margin_only_reference flip.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import uuid

GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_SCHEMA_VERSION = (
    "hrm_text_158_global_rate_cap_margin_selection_feasibility/v0.b2_5a_null"
)
GLOBAL_RATE_CAP_MARGIN_SELECTION_CPU_ORACLE_COMMIT_SHA_SHORT = "2b45cf0"

GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NULL_FINDING = (
    "Triton standard sort primitives (tl.sort/sort_impl/tl.topk at "
    "triton/language/standard.py:423-470) are value-returning, block-local, "
    "static-shape, and return NO argsort/permutation. A native MARGIN "
    "selection requires a per-row permutation over parallel row tensors "
    "(state_ids, flat_indices, abs, thresholds, directions) — there is no "
    "drop-in native argsort-permutation. torch.topk-on-device is "
    "device-resident but NOT kernelized (NATIVE_KERNELIZED_HOT_PATH_DEVICE_"
    "LAUNDERING_CAVEAT) and does NOT advance the global_cap native axis over "
    "the existing torch.argsort-stable CUDA reference. Native MARGIN "
    "selection requires a custom @triton.jit argsort-with-permutation "
    "kernel = B2-5a', NOT in-frozen-scope."
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS: tuple[str, ...] = (
    "B2-5a is a COMMITTED FEASIBILITY NULL; selection_parity_pass is PERMANENTLY False",
    "B2-5a does NOT implement a native @triton.jit kernel",
    "B2-5a does NOT mint a native selection pass on GPU or CPU",
    "B2-5a does NOT advance the global_cap native axis (no native kernelization)",
    "B2-5a torch.topk scaffold is NON-NATIVE CPU evidence for packed-key ordering, NOT a native path",
    "B2-5a does NOT flip global_cap_margin_only_reference",
    "B2-5a does NOT touch the deferred-backlog ledger",
    "B2-5a does NOT mutate q_levels / new_acc_i32 / accumulators",
    "B2-5a does NOT claim readiness / native-hot-path / hot_loop_resident",
    "B2-5a does NOT claim optimizer_credit_state",
    "B2-5a does NOT claim a full-loop integration result",
    "B2-5a native-pass via custom @triton.jit argsort is re-scoped to B2-5a'",
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_NON_CLAIMS = (
    GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS
)


@dataclass(frozen=True)
class GlobalRateCapSelectionScaffoldToken:
    """Wrapper-coupled token binding the CPU-scaffold packed-key ordering
    output to the scaffold invocation that produced it. NOT a native launch
    token; it couples the scaffold's input/output byte hashes for audit."""

    scaffold_invocation_nonce: str
    selection_input_sha256: str
    ordered_output_sha256: str
    accepted_output_sha256: str
    deferred_output_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "scaffold_invocation_nonce": self.scaffold_invocation_nonce,
            "selection_input_sha256": self.selection_input_sha256,
            "ordered_output_sha256": self.ordered_output_sha256,
            "accepted_output_sha256": self.accepted_output_sha256,
            "deferred_output_sha256": self.deferred_output_sha256,
        }


_FAIL_CLOSED_NATIVE_ALIAS_MSG = (
    "B2-5a committed null: prior native-named symbol removed/fail-closed; "
    "use scaffold/feasibility names only"
)


class _FailClosedNativeTypeAlias:
    """Prior native-named type alias; fail-closed on use."""

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(_FAIL_CLOSED_NATIVE_ALIAS_MSG)


GlobalRateCapSelectionNativeToken = _FailClosedNativeTypeAlias


@dataclass(frozen=True)
class GlobalRateCapMarginSelectionFeasibilityReceipt:
    schema_version: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_SCHEMA_VERSION
    cpu_oracle_commit_sha_short: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_CPU_ORACLE_COMMIT_SHA_SHORT
    selection_parity_pass: bool = False
    index_width_bit_budget_pass: bool = False
    feasibility_null: bool = True
    feasibility_null_finding: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NULL_FINDING
    observed_max_abs_observed: int = -1
    observed_index_width: int = -1
    observed_max_global_flat_index: int = -1
    empty_branch_taken: bool = False
    row_count: int = -1
    accepted_count: int = -1
    deferred_count: int = -1
    scaffold_parity_cases: tuple[str, ...] = ()
    scaffold_failed_cases: tuple[str, ...] = ()
    negative_offset_reject_evidence: bool = False
    non_claims: tuple[str, ...] = GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NON_CLAIMS
    token: GlobalRateCapSelectionScaffoldToken | None = None
    caveats: tuple[str, ...] = ()


GlobalRateCapMarginSelectionNativeParityReceipt = _FailClosedNativeTypeAlias


def build_global_rate_cap_margin_selection_feasibility_receipt(
    *,
    selection_parity_pass: bool = False,
    index_width_bit_budget_pass: bool = False,
    feasibility_null: bool = True,
    feasibility_null_finding: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_NULL_FINDING,
    observed_max_abs_observed: int = -1,
    observed_index_width: int = -1,
    observed_max_global_flat_index: int = -1,
    empty_branch_taken: bool = False,
    row_count: int = -1,
    accepted_count: int = -1,
    deferred_count: int = -1,
    scaffold_parity_cases: tuple[str, ...] = (),
    scaffold_failed_cases: tuple[str, ...] = (),
    negative_offset_reject_evidence: bool = False,
    token: GlobalRateCapSelectionScaffoldToken | None = None,
    caveats: tuple[str, ...] = (),
) -> GlobalRateCapMarginSelectionFeasibilityReceipt:
    """Null/scaffold builder. selection_parity_pass PERMANENTLY False — no
    native pass is mintable on this slice. Mutating / ledger / optimizer /
    native-kernel claims are forbidden regardless of inputs."""

    if selection_parity_pass:
        raise ValueError(
            "B2-5a is a committed feasibility null; selection_parity_pass "
            "cannot be True (no native @triton.jit kernel exists)"
        )

    if feasibility_null and selection_parity_pass:
        raise ValueError(
            "feasibility_null=True cannot coexist with selection_parity_pass=True"
        )

    if row_count == 0 and not empty_branch_taken:
        raise ValueError(
            "row_count=0 must carry empty_branch_taken=True (not a fabricated budget)"
        )

    if row_count == 0 and observed_max_abs_observed != -1:
        raise ValueError(
            "empty branch must not record a fabricated observed_max_abs_observed"
        )

    return GlobalRateCapMarginSelectionFeasibilityReceipt(
        selection_parity_pass=selection_parity_pass,
        index_width_bit_budget_pass=index_width_bit_budget_pass,
        feasibility_null=feasibility_null,
        feasibility_null_finding=feasibility_null_finding,
        observed_max_abs_observed=observed_max_abs_observed,
        observed_index_width=observed_index_width,
        observed_max_global_flat_index=observed_max_global_flat_index,
        empty_branch_taken=empty_branch_taken,
        row_count=row_count,
        accepted_count=accepted_count,
        deferred_count=deferred_count,
        scaffold_parity_cases=tuple(scaffold_parity_cases),
        scaffold_failed_cases=tuple(scaffold_failed_cases),
        negative_offset_reject_evidence=negative_offset_reject_evidence,
        token=token,
        caveats=tuple(caveats),
    )


def build_global_rate_cap_margin_selection_native_parity_receipt(*args, **kwargs):
    """Prior native-named builder removed/fail-closed on B2-5a null."""
    raise RuntimeError(_FAIL_CLOSED_NATIVE_ALIAS_MSG)


def validate_global_rate_cap_margin_selection_feasibility_receipt(
    receipt: GlobalRateCapMarginSelectionFeasibilityReceipt,
) -> None:
    """Validate the null/scaffold receipt. Raises ValueError on any
    invariant breach. Forbidden-claim checks are structural, not advisory."""

    if receipt.schema_version != GLOBAL_RATE_CAP_MARGIN_SELECTION_FEASIBILITY_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")

    if receipt.selection_parity_pass:
        raise ValueError(
            "B2-5a is a committed feasibility null; selection_parity_pass "
            "must be PERMANENTLY False"
        )

    if receipt.feasibility_null and receipt.selection_parity_pass:
        raise ValueError("feasibility_null and selection_parity_pass are mutually exclusive")

    if receipt.row_count == 0 and not receipt.empty_branch_taken:
        raise ValueError("row_count=0 requires empty_branch_taken=True")

    if receipt.row_count == 0 and receipt.observed_max_abs_observed != -1:
        raise ValueError("empty branch must not record a fabricated observed_max_abs_observed")

    if receipt.row_count == 0 and receipt.observed_index_width != -1:
        raise ValueError("empty branch must not record a fabricated observed_index_width")

    if receipt.row_count == 0 and receipt.observed_max_global_flat_index != -1:
        raise ValueError(
            "empty branch must not record a fabricated observed_max_global_flat_index"
        )

    if receipt.row_count < 0:
        raise ValueError("row_count must be >= 0 (use 0 for the empty branch)")

    if receipt.row_count > 0:
        if receipt.observed_max_abs_observed < 0:
            raise ValueError("non-empty scaffold must record observed_max_abs_observed >= 0")
        if receipt.observed_index_width < 1:
            raise ValueError("non-empty scaffold must record observed_index_width >= 1")
        if receipt.observed_max_global_flat_index < 0:
            raise ValueError(
                "non-empty scaffold must record observed_max_global_flat_index >= 0"
            )

    if receipt.token is not None and not isinstance(
        receipt.token, GlobalRateCapSelectionScaffoldToken
    ):
        raise ValueError("token must be a scaffold token or None")


def validate_global_rate_cap_margin_selection_native_parity_receipt(*args, **kwargs):
    """Prior native-named validator removed/fail-closed on B2-5a null."""
    raise RuntimeError(_FAIL_CLOSED_NATIVE_ALIAS_MSG)


def new_selection_token(
    *,
    selection_input_sha256: str,
    ordered_output_sha256: str,
    accepted_output_sha256: str,
    deferred_output_sha256: str,
) -> GlobalRateCapSelectionScaffoldToken:
    """Mint a fresh scaffold-coupled token with a uuid4 invocation nonce."""

    return GlobalRateCapSelectionScaffoldToken(
        scaffold_invocation_nonce=uuid.uuid4().hex,
        selection_input_sha256=selection_input_sha256,
        ordered_output_sha256=ordered_output_sha256,
        accepted_output_sha256=accepted_output_sha256,
        deferred_output_sha256=deferred_output_sha256,
    )


def canonical_tensor_payload_sha256(tensor) -> str:
    """Canonical byte-hash for a device/CPU tensor payload
    (detach().cpu().contiguous().view(-1).numpy().tobytes())."""

    view = tensor.detach().cpu().contiguous().view(-1)
    if view.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(view.numpy().tobytes()).hexdigest()
