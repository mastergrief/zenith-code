"""R51 tier-3 distillation: broad prompt bank + Small2DTransformer student."""

from calm.llm_computer.r51.prompt_bank import (  # noqa: F401
    build_broad_corpus,
    sample_code,
    sample_creative,
    sample_factual,
    sample_multi_step_arith,
    sample_single_step_arith,
    sample_translation,
)
from calm.llm_computer.r51.student import R51Student, R51StudentConfig  # noqa: F401
