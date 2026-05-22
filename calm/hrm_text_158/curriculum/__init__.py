"""HRM-Text-1.58 Phase 3 curriculum infrastructure.

Per task #51, board task `1779460303130-742c8cbd`, codex msg
1779460698439 (Phase 3 Step 0 +1 with A1 byte-level UTF-8 + 7 guardrails).

This package contains DATA + PROBE infrastructure ONLY. No model code,
no training launch. Step 0 dry-run scope.
"""
from calm.hrm_text_158.curriculum.broad_tokenizer import (
    BROAD_NORMALIZER_VERSION,
    BroadTokenizer,
)
from calm.hrm_text_158.curriculum.schema import RungProbeResult
from calm.hrm_text_158.curriculum.generators import (
    RUNG_NAMES,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.splits import (
    build_rung_splits,
    assert_no_train_holdout_overlap,
)
from calm.hrm_text_158.curriculum.retention import compute_retention_deltas
from calm.hrm_text_158.curriculum.ckpt_compat import (
    validate_load_from_ckpt_compat,
)

__all__ = [
    "BROAD_NORMALIZER_VERSION",
    "BroadTokenizer",
    "RungProbeResult",
    "RUNG_NAMES",
    "make_rung_examples",
    "build_rung_splits",
    "assert_no_train_holdout_overlap",
    "compute_retention_deltas",
    "validate_load_from_ckpt_compat",
]
