"""LANDS-AB evaluation schema constants (PLAN_v6 / IMPLEMENT_v2).

Pure constants + key-set helpers. No IO, no GPU, no torch.
"""
from __future__ import annotations

from typing import Final

BRANCH_SCOPE_CREEP: Final = "BR-LANDS-AB-SCOPE-CREEP-STOP"
BRANCH_FIXTURE_CONTRACT_FAIL: Final = "BR-LANDS-AB-FIXTURE-CONTRACT-FAIL"
BRANCH_VACUOUS: Final = "BR-LANDS-AB-VACUOUS"
BRANCH_DIVERGENT_EVENT: Final = "BR-LANDS-AB-DIVERGENT-EVENT"
BRANCH_DIVERGENT_APPLY: Final = "BR-LANDS-AB-DIVERGENT-APPLY"
BRANCH_DIVERGENT_ORACLE_LIVE: Final = "BR-LANDS-AB-DIVERGENT-ORACLE-LIVE"
BRANCH_EQUIVALENT: Final = "BR-LANDS-AB-EQUIVALENT"

PRIORITY_ORDER: Final[tuple[str, ...]] = (
    BRANCH_SCOPE_CREEP,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_VACUOUS,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_ORACLE_LIVE,
    BRANCH_EQUIVALENT,
)

# Frozen applicability map: gating_row -> applicable surfaces (PLAN_v6)
APPLICABILITY_MAP: Final[dict[str, tuple[str, ...]]] = {
    "G_CPU_STATIC_AB": ("s1", "s2", "s3", "s4", "s6"),
    "G_CUDA_B1_APPLY": ("s3", "s4", "s6"),
    "G_CUDA_B2_APPLY": ("s3", "s4", "s6"),
    "G_CUDA_B3_APPLY": ("s3", "s4", "s6"),
    "G_CUDA_ORACLE_B1": ("s5",),
    "G_CUDA_ORACLE_B2": ("s5",),
    "G_CUDA_ORACLE_B3": ("s5",),
}

GATING_ROWS: Final[tuple[str, ...]] = tuple(APPLICABILITY_MAP.keys())

SURFACE_S4_SITES: Final[tuple[str, ...]] = (
    "G_CPU_STATIC_AB",
    "G_CUDA_B1_APPLY",
    "G_CUDA_B2_APPLY",
    "G_CUDA_B3_APPLY",
)


def cell_key(gating_row: str, surface: str) -> str:
    return f"{gating_row}/{surface}"


CANONICAL_CELL_KEYS: Final[tuple[str, ...]] = tuple(
    sorted(
        cell_key(row, surf)
        for row, surfs in APPLICABILITY_MAP.items()
        for surf in surfs
    )
)

assert len(CANONICAL_CELL_KEYS) == 17

FORBIDDEN_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "s1_pass",
        "s2_pass",
        "s3_pass",
        "s4_pass",
        "s5_pass",
        "s6_pass",
        "site_s4_pass",
        "gating_row_pass",
        "vacuous",
        "divergent_event",
        "divergent_apply",
        "divergent_oracle_live",
        "gating_rows_all_pass",
        "fixture_contract_fail",
        "branch_id",
        "terminal_branch",
        "equivalent",
    }
)

REQUIRED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"scope_creep", "fixture_contract_raw_fail", "surface_pass_by_row"}
)

# --- IMPLEMENT_v2 evidence-bound evaluation contract ---

PLAN_V6_PATH: Final = (
    "artifacts/acc_entropy/"
    "optimizer_credit_state_sparse_vote_authority_LANDS_AB_EVAL_PLAN_v6.json"
)
PLAN_V6_SHA256: Final = (
    "93645d31ea8a0cb0f89cfc4f1aedd38190a47f18433b6bc67c9b5d98da7093c5"
)
TASK_ID: Final = "1785244883605-c7c0b0a3"
PLUS1_IMPLEMENT_MSG_ID: Final = "1785247579082-220d1149"
IMPLEMENT_V2_DISPATCH_MSG_ID: Final = "1785248661917-5c70950f"

FIXTURE_RECIPE_NAME: Final = "3C_C1_dry_run_fixture_seed158"
PARITY_FIXTURE_DESCRIPTOR_SHA256: Final = (
    "fdc186f780cfdcbe5db72ee04bef49628173a65106e1843d21593c047290314c"
)
RECARRY_RECEIPT_PATH: Final = (
    "artifacts/acc_entropy/"
    "optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json"
)
RECARRY_RECEIPT_SHA256: Final = (
    "783f279986ebaa9bd7d170b5996146a319e9c8f1980939ec8ee49ac4b5d5db2f"
)
RANK_SPEC_SYMBOL: Final = "default_dry_run_rank_vote_spec"
RANK_SPEC_DIGEST_EXPECTED: Final = (
    "6c109e0482292edf72d3cc4ada6bda0840e67e8dbfac4ad7fd64d353602806a5"
)

RAW_ROW_OBSERVATION_SCHEMA: Final = "lands_ab_raw_row_observation_v1"
EVAL_RECEIPT_SCHEMA: Final = "lands_ab_eval_receipt_v2"
DIAGNOSTIC_RECEIPT_SCHEMA: Final = "lands_ab_eval_diagnostic_v1"

PHASE_ORDER: Final[tuple[str, ...]] = (
    "forward_backward",
    "update",
    "emission",
    "flush",
)

CLAIM_CEILING: Final[dict[str, bool]] = {
    "LANDS_AB": False,
    "science_claim": False,
    "full_sub2_runtime_ready_for_science": False,
    "equivalent_minted": False,
}

REQUIRED_RAW_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "gating_row",
        "device",
        "measured_surfaces",
        "metrics",
        "key_universe",
        "key_universe_sha256",
        "rank_spec_digest",
        "fixture_contract_raw_fail",
        "science_claim",
        "synthetic_only",
    }
)

REQUIRED_EVAL_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "plan_sha256",
        "plan_path",
        "source_pins",
        "required_key_set",
        "required_key_set_sha256",
        "raw_row_artifacts",
        "surface_pass_by_row",
        "scope_creep",
        "fixture_contract_raw_fail",
        "reducer_output",
        "claim_ceiling",
        "science_claim",
        "synthetic_only",
        "caveats",
    }
)
