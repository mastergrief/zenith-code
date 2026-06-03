"""Phase-0 memory/compute ledger skeleton."""
from __future__ import annotations

from dataclasses import dataclass


LEDGER_SCHEMA_VERSION = "hrm_text_158_native_full_stack_ledger/v0.phase0"
PENDING_S1_TERMINAL = "pending_s1_terminal"


@dataclass(frozen=True)
class LedgerRow:
    subsystem: str
    current_b4_measured: str
    target_precision_storage: str
    learner_role: str
    peak_budget: str
    proof_gate: str
    gap: str
    status: str = PENDING_S1_TERMINAL


PHASE0_LEDGER_ROWS = (
    LedgerRow(
        subsystem="eligible_bulk_projections",
        current_b4_measured=PENDING_S1_TERMINAL,
        target_precision_storage="q:int8 ternary + acc:int16 + frozen_scale:float32",
        learner_role="native eligible-bulk learner state",
        peak_budget="sub-2-bit persistent state target; exact peak pending S1 terminal sizing",
        proof_gate="frozen-q attribution integrity plus decode/EOS non-regression",
        gap="No native q/k/v/o/gate/up/down kernel implementation in Phase-0.",
    ),
    LedgerRow(
        subsystem="structural_bitlinear_fp_parameters",
        current_b4_measured=PENDING_S1_TERMINAL,
        target_precision_storage="sunset diagnostic FP placeholders only",
        learner_role="must not carry eligible-bulk learning in native path",
        peak_budget="excluded from authoritative eligible-bulk learner-state budget",
        proof_gate="state hash excludes FP masters; optimizer excludes eligible FP params",
        gap="Current repo BitLinear keeps FP master parameters outside the native end-state.",
    ),
    LedgerRow(
        subsystem="credit_capture",
        current_b4_measured=PENDING_S1_TERMINAL,
        target_precision_storage="transient FP inputs/grad_outputs for proof only",
        learner_role="attribution evidence, not persisted learner state",
        peak_budget="bounded by hook receipt; no Phase-0 allocation claim",
        proof_gate="observed c1353fd5 hook strata and state-hash receipt",
        gap="Phase-0 records hook contracts only; no capture execution.",
    ),
    LedgerRow(
        subsystem="authoritative_forward_eval",
        current_b4_measured=PENDING_S1_TERMINAL,
        target_precision_storage="transient q*scale float32 materialization",
        learner_role="evaluation/export context, not learning state",
        peak_budget="transient only; no persisted eligible-bulk FP learner",
        proof_gate="source pointer hash plus attribution integrity checks",
        gap="Native eval/export facade remains unimplemented pre-terminal.",
    ),
    LedgerRow(
        subsystem="decode_eos_smoke",
        current_b4_measured=PENDING_S1_TERMINAL,
        target_precision_storage="probe metrics schema",
        learner_role="correctness/non-regression floor",
        peak_budget="CPU/static schema now; future probe wall-time/headroom receipt",
        proof_gate="finite logits, EOS stop, exact/parsed/non-regression fields",
        gap="Phase-0 does not run probes or bank acquisition.",
    ),
)
