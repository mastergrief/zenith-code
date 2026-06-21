"""B2-5a′ Stage-2 native MARGIN-selection parity receipt (single-block scoped).

``selection_parity_pass=True`` is mintable ONLY after GPU exact-parity proof in the
single-block regime (row_count <= BLOCK).  Realistic-size native global selection
remains BLOCKED pending B2-5a'' multiblock merge.
"""
from __future__ import annotations

from dataclasses import dataclass
import uuid

GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION = (
    "hrm_text_158_global_rate_cap_margin_selection_native_parity/v0.b2_5a_prime_stage2"
)

GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_NON_CLAIMS: tuple[str, ...] = (
    "B2-5a′ Stage-2 selection_parity_pass is scoped to single-block row_count<=BLOCK ONLY",
    "B2-5a′ Stage-2 does NOT claim realistic-size native global selection (BLOCKED pending B2-5a'')",
    "B2-5a′ Stage-2 does NOT flip global_cap_margin_only_reference",
    "B2-5a′ Stage-2 does NOT touch the deferred-backlog ledger",
    "B2-5a′ Stage-2 does NOT claim readiness / optimizer_credit_state / full-loop",
    "B2-5a′ Stage-2 row_count>BLOCK defers honestly (no torch/CPU fallback)",
    "B2-5a′ Stage-2 native path forbids torch sort/gather permutation of the 7 row buffers",
)

ROW_TENSOR_BUFFER_ROLES: tuple[str, ...] = (
    "row_state_ids",
    "row_flat_indices",
    "row_local_positions",
    "row_global_flat_indices",
    "row_abs_new_acc",
    "row_thresholds",
    "row_directions",
)


@dataclass(frozen=True)
class NativeSelectionParityProof:
    """Explicit oracle-comparison proof required to mint selection_parity_pass=True."""

    parity_ok: bool = False
    ordered_global_indices_match: bool = False
    accepted_positions_match: bool = False
    deferred_positions_match: bool = False

    def validate_for_pass_mint(self) -> None:
        if not self.parity_ok:
            raise ValueError("parity_ok must be True to mint selection_parity_pass")
        if not self.ordered_global_indices_match:
            raise ValueError("ordered_global_indices_match required for pass mint")
        if not self.accepted_positions_match:
            raise ValueError("accepted_positions_match required for pass mint")
        if not self.deferred_positions_match:
            raise ValueError("deferred_positions_match required for pass mint")


@dataclass(frozen=True)
class GlobalRateCapSelectionNativeToken:
    native_invocation_nonce: str
    bitonic_kernel_symbol: str
    gather_kernel_symbol: str
    kernel_source_sha256: str
    selection_input_sha256: str
    ordered_output_sha256: str
    accepted_output_sha256: str
    deferred_output_sha256: str
    backend: str = "cuda"

    def to_dict(self) -> dict[str, str]:
        return {
            "native_invocation_nonce": self.native_invocation_nonce,
            "bitonic_kernel_symbol": self.bitonic_kernel_symbol,
            "gather_kernel_symbol": self.gather_kernel_symbol,
            "kernel_source_sha256": self.kernel_source_sha256,
            "selection_input_sha256": self.selection_input_sha256,
            "ordered_output_sha256": self.ordered_output_sha256,
            "accepted_output_sha256": self.accepted_output_sha256,
            "deferred_output_sha256": self.deferred_output_sha256,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class GlobalRateCapMarginSelectionNativeParityReceipt:
    schema_version: str = GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION
    selection_parity_pass: bool = False
    single_block_regime: bool = False
    multiblock_deferred: bool = False
    row_count: int = -1
    block: int = 1024
    sort_padded_n: int = -1
    pos_width: int = -1
    host_max_full_key: int = -1
    padding_sentinel: int = (1 << 63) - 1
    padding_headroom_ok: bool = False
    full_pack_bits: int = -1
    budget_infeasible: bool = False
    native_path_audit_pass: bool = False
    post_kernel_torch_permutation_detected: bool = False
    kernel_output_buffers_emitted: bool = False
    tile_primitive_seam: str = "margin_selection_single_block_tile"
    parity_cases: tuple[str, ...] = ()
    failed_cases: tuple[str, ...] = ()
    parity_proof: NativeSelectionParityProof | None = None
    non_claims: tuple[str, ...] = GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_NON_CLAIMS
    token: GlobalRateCapSelectionNativeToken | None = None
    caveats: tuple[str, ...] = ()


def build_global_rate_cap_margin_selection_native_parity_receipt(
    *,
    selection_parity_pass: bool = False,
    single_block_regime: bool = False,
    multiblock_deferred: bool = False,
    row_count: int = -1,
    block: int = 1024,
    sort_padded_n: int = -1,
    pos_width: int = -1,
    host_max_full_key: int = -1,
    padding_sentinel: int = (1 << 63) - 1,
    padding_headroom_ok: bool = False,
    full_pack_bits: int = -1,
    budget_infeasible: bool = False,
    native_path_audit_pass: bool = False,
    post_kernel_torch_permutation_detected: bool = False,
    kernel_output_buffers_emitted: bool = False,
    tile_primitive_seam: str = "margin_selection_single_block_tile",
    parity_cases: tuple[str, ...] = (),
    failed_cases: tuple[str, ...] = (),
    parity_proof: NativeSelectionParityProof | None = None,
    token: GlobalRateCapSelectionNativeToken | None = None,
    caveats: tuple[str, ...] = (),
) -> GlobalRateCapMarginSelectionNativeParityReceipt:
    if selection_parity_pass and parity_proof is None:
        raise ValueError(
            "selection_parity_pass=True requires parity_proof from oracle comparison"
        )
    if selection_parity_pass and parity_proof is not None:
        parity_proof.validate_for_pass_mint()
    if selection_parity_pass and not single_block_regime:
        raise ValueError(
            "selection_parity_pass=True requires single_block_regime=True (row_count<=BLOCK)"
        )
    if selection_parity_pass and multiblock_deferred:
        raise ValueError("selection_parity_pass=True incompatible with multiblock_deferred")
    if selection_parity_pass and budget_infeasible:
        raise ValueError("selection_parity_pass=True incompatible with budget_infeasible")
    if selection_parity_pass and not native_path_audit_pass:
        raise ValueError("selection_parity_pass=True requires native_path_audit_pass=True")
    if selection_parity_pass and post_kernel_torch_permutation_detected:
        raise ValueError(
            "selection_parity_pass=True incompatible with post_kernel_torch_permutation_detected"
        )
    if selection_parity_pass and not kernel_output_buffers_emitted:
        raise ValueError(
            "selection_parity_pass=True requires kernel_output_buffers_emitted=True"
        )
    if selection_parity_pass and token is None:
        raise ValueError("selection_parity_pass=True requires a native token")

    return GlobalRateCapMarginSelectionNativeParityReceipt(
        selection_parity_pass=selection_parity_pass,
        single_block_regime=single_block_regime,
        multiblock_deferred=multiblock_deferred,
        row_count=row_count,
        block=block,
        sort_padded_n=sort_padded_n,
        pos_width=pos_width,
        host_max_full_key=host_max_full_key,
        padding_sentinel=padding_sentinel,
        padding_headroom_ok=padding_headroom_ok,
        full_pack_bits=full_pack_bits,
        budget_infeasible=budget_infeasible,
        native_path_audit_pass=native_path_audit_pass,
        post_kernel_torch_permutation_detected=post_kernel_torch_permutation_detected,
        kernel_output_buffers_emitted=kernel_output_buffers_emitted,
        tile_primitive_seam=tile_primitive_seam,
        parity_cases=tuple(parity_cases),
        failed_cases=tuple(failed_cases),
        parity_proof=parity_proof,
        token=token,
        caveats=tuple(caveats),
    )


def validate_global_rate_cap_margin_selection_native_parity_receipt(
    receipt: GlobalRateCapMarginSelectionNativeParityReceipt,
) -> None:
    if receipt.schema_version != GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")

    if receipt.selection_parity_pass and not receipt.single_block_regime:
        raise ValueError("selection_parity_pass requires single_block_regime")
    if receipt.selection_parity_pass and receipt.multiblock_deferred:
        raise ValueError("selection_parity_pass incompatible with multiblock_deferred")
    if receipt.selection_parity_pass and receipt.budget_infeasible:
        raise ValueError("selection_parity_pass incompatible with budget_infeasible")
    if receipt.selection_parity_pass and not receipt.native_path_audit_pass:
        raise ValueError("selection_parity_pass requires native_path_audit_pass")
    if receipt.selection_parity_pass and receipt.post_kernel_torch_permutation_detected:
        raise ValueError("selection_parity_pass incompatible with post_kernel_torch_permutation")
    if receipt.selection_parity_pass and not receipt.kernel_output_buffers_emitted:
        raise ValueError("selection_parity_pass requires kernel_output_buffers_emitted")
    if receipt.selection_parity_pass and receipt.token is None:
        raise ValueError("selection_parity_pass requires token")

    if receipt.selection_parity_pass and receipt.parity_proof is None:
        raise ValueError("selection_parity_pass requires parity_proof")
    if receipt.selection_parity_pass and receipt.parity_proof is not None:
        receipt.parity_proof.validate_for_pass_mint()

    if receipt.row_count == 0 and receipt.selection_parity_pass:
        raise ValueError("empty row_count cannot mint selection_parity_pass on GPU path")

    if receipt.row_count > receipt.block and receipt.selection_parity_pass:
        raise ValueError("row_count > block cannot mint selection_parity_pass")


def new_native_selection_token(
    *,
    bitonic_kernel_symbol: str,
    gather_kernel_symbol: str,
    kernel_source_sha256: str,
    selection_input_sha256: str,
    ordered_output_sha256: str,
    accepted_output_sha256: str,
    deferred_output_sha256: str,
    backend: str = "cuda",
) -> GlobalRateCapSelectionNativeToken:
    return GlobalRateCapSelectionNativeToken(
        native_invocation_nonce=uuid.uuid4().hex,
        bitonic_kernel_symbol=bitonic_kernel_symbol,
        gather_kernel_symbol=gather_kernel_symbol,
        kernel_source_sha256=kernel_source_sha256,
        selection_input_sha256=selection_input_sha256,
        ordered_output_sha256=ordered_output_sha256,
        accepted_output_sha256=accepted_output_sha256,
        deferred_output_sha256=deferred_output_sha256,
        backend=backend,
    )


def apply_native_selection_parity_proof(
    receipt: GlobalRateCapMarginSelectionNativeParityReceipt,
    *,
    parity_proof: NativeSelectionParityProof,
    token: GlobalRateCapSelectionNativeToken,
) -> GlobalRateCapMarginSelectionNativeParityReceipt:
    """Rebuild receipt with oracle parity proof — sole path to mint selection_parity_pass."""

    patched = build_global_rate_cap_margin_selection_native_parity_receipt(
        selection_parity_pass=True,
        single_block_regime=receipt.single_block_regime,
        multiblock_deferred=receipt.multiblock_deferred,
        row_count=receipt.row_count,
        block=receipt.block,
        sort_padded_n=receipt.sort_padded_n,
        pos_width=receipt.pos_width,
        host_max_full_key=receipt.host_max_full_key,
        padding_sentinel=receipt.padding_sentinel,
        padding_headroom_ok=receipt.padding_headroom_ok,
        full_pack_bits=receipt.full_pack_bits,
        budget_infeasible=receipt.budget_infeasible,
        native_path_audit_pass=receipt.native_path_audit_pass,
        post_kernel_torch_permutation_detected=receipt.post_kernel_torch_permutation_detected,
        kernel_output_buffers_emitted=receipt.kernel_output_buffers_emitted,
        tile_primitive_seam=receipt.tile_primitive_seam,
        parity_cases=receipt.parity_cases,
        failed_cases=receipt.failed_cases,
        parity_proof=parity_proof,
        token=token,
        caveats=receipt.caveats,
    )
    validate_global_rate_cap_margin_selection_native_parity_receipt(patched)
    return patched


__all__ = [
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_NON_CLAIMS",
    "GLOBAL_RATE_CAP_MARGIN_SELECTION_NATIVE_PARITY_SCHEMA_VERSION",
    "GlobalRateCapMarginSelectionNativeParityReceipt",
    "GlobalRateCapSelectionNativeToken",
    "NativeSelectionParityProof",
    "ROW_TENSOR_BUFFER_ROLES",
    "apply_native_selection_parity_proof",
    "build_global_rate_cap_margin_selection_native_parity_receipt",
    "new_native_selection_token",
    "validate_global_rate_cap_margin_selection_native_parity_receipt",
]
