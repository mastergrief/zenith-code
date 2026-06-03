"""Phase-0 native-full-stack scaffold for HRM-Text-1.58.

This package is intentionally contract-only. It records the native ternary
stack seams, source anchors, ledger rows, FP exceptions, attribution hooks, and
acceptance metrics without implementing kernels, trainer entrypoints, or live
run integration.
"""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.attribution import (
    ATTRIBUTION_HOOK_POINTS,
    ATTRIBUTION_INTEGRITY_CHECKS,
    LIVE_C1353FD5_OBSERVATIONS,
    AttributionHookPoint,
    AttributionIntegrityCheck,
)
from calm.hrm_text_158.native_full_stack.contracts import (
    IMPLEMENTATION_STATUS_SKELETON_ONLY,
    PROJECTION_GROUPS,
    SUBSYSTEM_CONTRACTS,
    SubsystemContract,
)
from calm.hrm_text_158.native_full_stack.fp_exceptions import (
    FP_EXCEPTION_REGISTRY,
    FPException,
)
from calm.hrm_text_158.native_full_stack.ledger import (
    LEDGER_SCHEMA_VERSION,
    PHASE0_LEDGER_ROWS,
    LedgerRow,
)
from calm.hrm_text_158.native_full_stack.metrics import (
    ACCEPTANCE_METRICS,
    AcceptanceMetric,
    first_class_metric_names,
)
from calm.hrm_text_158.native_full_stack.qscale_linear import (
    INT8_LEVELS_TRANSITIONAL_NOTE,
    QScaleLinearConfig,
    QScaleWeightFormat,
    QScaleWeightState,
    qscale_linear_reference,
    qscale_linear_triton,
    validate_qscale_weight_state,
)
from calm.hrm_text_158.native_full_stack.source_pointers import (
    ACTIVE_HRM_REPO_ROOT,
    HISTORICAL_NON_ANCHOR_POINTERS,
    LIVE_S1_TRAINER_POINTER,
    PHASE0_SOURCE_POINTERS,
    SourcePointer,
)

__all__ = [
    "ACTIVE_HRM_REPO_ROOT",
    "ACCEPTANCE_METRICS",
    "ATTRIBUTION_HOOK_POINTS",
    "ATTRIBUTION_INTEGRITY_CHECKS",
    "AttributionHookPoint",
    "AttributionIntegrityCheck",
    "AcceptanceMetric",
    "FP_EXCEPTION_REGISTRY",
    "FPException",
    "HISTORICAL_NON_ANCHOR_POINTERS",
    "IMPLEMENTATION_STATUS_SKELETON_ONLY",
    "INT8_LEVELS_TRANSITIONAL_NOTE",
    "LEDGER_SCHEMA_VERSION",
    "LIVE_C1353FD5_OBSERVATIONS",
    "LIVE_S1_TRAINER_POINTER",
    "LedgerRow",
    "PHASE0_LEDGER_ROWS",
    "PHASE0_SOURCE_POINTERS",
    "PROJECTION_GROUPS",
    "QScaleLinearConfig",
    "QScaleWeightFormat",
    "QScaleWeightState",
    "SUBSYSTEM_CONTRACTS",
    "SourcePointer",
    "SubsystemContract",
    "first_class_metric_names",
    "qscale_linear_reference",
    "qscale_linear_triton",
    "validate_qscale_weight_state",
]
