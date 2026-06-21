"""B2-5c Step-1b-(3a) candidate global-cap trainer wiring receipt (CPU/read-only)."""
from __future__ import annotations

from dataclasses import dataclass

CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_SCHEMA_VERSION = (
    "hrm_text_158_candidate_global_cap_trainer_wiring/v0.b2_5c_step1b_3a"
)

CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_NON_CLAIMS: tuple[str, ...] = (
    "B2-5c Step-1b-(3a) is CPU trainer wiring behind default-off flag only",
    "B2-5c Step-1b-(3a) does NOT run native MARGIN selection or GPU proof",
    "B2-5c Step-1b-(3a) does NOT mint selection_parity_pass",
    "B2-5c Step-1b-(3a) does NOT flip global_cap_margin_only_reference",
    "B2-5c Step-1b-(3a) does NOT flip optimizer_credit_state / readiness rows",
    "B2-5c Step-1b-(3a) does NOT claim B-lite native shape compatibility (3b owns GPU)",
    "composition_path_default_active remains False when flag default-off",
)

CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_HARD_FALSE_FIELDS: tuple[str, ...] = (
    "selection_parity_pass",
    "native_selector_wired",
    "readiness_flip_authorized",
    "global_cap_margin_only_reference_flipped",
    "optimizer_credit_state_sub2_claim",
    "b_lite_native_shape_compat_proven",
    "gpu_wired_trainer_parity_proven",
)


@dataclass(frozen=True)
class CandidateGlobalCapTrainerWiringReceipt:
    schema_version: str
    composition_path_exists_in_code: bool
    composition_path_default_active: bool
    trainer_candidate_global_cap_composition_active: bool
    flag_default_off_guard_preserved: bool
    seam_module_byte_frozen: bool
    selection_parity_pass: bool
    native_selector_wired: bool
    readiness_flip_authorized: bool
    global_cap_margin_only_reference_flipped: bool
    optimizer_credit_state_sub2_claim: bool
    b_lite_native_shape_compat_proven: bool
    gpu_wired_trainer_parity_proven: bool
    non_claims: tuple[str, ...]


def build_candidate_global_cap_trainer_wiring_receipt(
    *,
    composition_path_exists_in_code: bool,
    composition_path_default_active: bool,
    trainer_candidate_global_cap_composition_active: bool,
    flag_default_off_guard_preserved: bool,
    seam_module_byte_frozen: bool,
) -> CandidateGlobalCapTrainerWiringReceipt:
    receipt = CandidateGlobalCapTrainerWiringReceipt(
        schema_version=CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_SCHEMA_VERSION,
        composition_path_exists_in_code=bool(composition_path_exists_in_code),
        composition_path_default_active=bool(composition_path_default_active),
        trainer_candidate_global_cap_composition_active=bool(
            trainer_candidate_global_cap_composition_active,
        ),
        flag_default_off_guard_preserved=bool(flag_default_off_guard_preserved),
        seam_module_byte_frozen=bool(seam_module_byte_frozen),
        selection_parity_pass=False,
        native_selector_wired=False,
        readiness_flip_authorized=False,
        global_cap_margin_only_reference_flipped=False,
        optimizer_credit_state_sub2_claim=False,
        b_lite_native_shape_compat_proven=False,
        gpu_wired_trainer_parity_proven=False,
        non_claims=CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_NON_CLAIMS,
    )
    validate_candidate_global_cap_trainer_wiring_receipt(receipt)
    return receipt


def validate_candidate_global_cap_trainer_wiring_receipt(
    receipt: CandidateGlobalCapTrainerWiringReceipt,
) -> None:
    if receipt.schema_version != CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_SCHEMA_VERSION:
        raise ValueError("trainer wiring receipt schema_version mismatch")
    for field_name in CANDIDATE_GLOBAL_CAP_TRAINER_WIRING_HARD_FALSE_FIELDS:
        if bool(getattr(receipt, field_name)):
            raise ValueError(f"trainer wiring hard-false field must be False: {field_name}")
