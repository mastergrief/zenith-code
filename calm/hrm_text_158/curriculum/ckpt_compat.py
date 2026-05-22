"""Phase 3 --load-from ckpt compatibility validation.

Per codex msg 1779460421099 + 1779460698439: each rung loads PRIOR rung's
best ckpt. --load-from must HARD FAIL on:
- vocab mismatch (gsm8k_char_vocab list differs)
- normalizer_version mismatch
- use_ternary_bulk mismatch
- architecture mismatch (hidden_size, n_layers, num_heads, H/L_cycles, etc.)

Per codex msg 1779460698439: --load-from loads MODEL_STATE ONLY. Optimizer
state + LR schedule reset per rung. Phase 2 GSM8k-tokenizer ckpts are
NOT load-from parents for Phase 3 (their vocab differs from BroadTokenizer's).
"""
from __future__ import annotations

from typing import Any


# Architecture fields that must match exactly between rungs
_ARCH_FIELDS = (
    "hidden_size",
    "n_layers",
    "num_heads",
    "H_cycles",
    "L_cycles",
    "half_layers",
    "expansion",
    "max_seq_len",
    "attn_type",
    "init_type",
    "norm_type",
)


def validate_load_from_ckpt_compat(
    loaded_ckpt_config: dict,
    current_cfg: Any,
    current_vocab_list: list[str],
    current_normalizer_version: str,
) -> None:
    """Validate that a loaded ckpt's config is compatible with current config.

    Args:
        loaded_ckpt_config: the `config` dict from torch.load(ckpt)["config"]
        current_cfg: HierarchicalReasoningModelConfig in use for this rung
        current_vocab_list: current tokenizer's `vocab_as_list()` output
        current_normalizer_version: current tokenizer's `normalizer_version`

    Raises:
        ValueError on any mismatch (vocab / normalizer / ternary / arch field).
        Each mismatch surfaces the specific drift to make diagnosis fast.
    """
    # Vocab — full list equality (id order matters)
    loaded_vocab = loaded_ckpt_config.get("gsm8k_char_vocab")
    if loaded_vocab is None:
        raise ValueError(
            "--load-from ckpt missing 'gsm8k_char_vocab' field; cannot verify "
            "tokenizer compatibility. Phase 2 ckpts have this field; cross-arc "
            "ckpts may not."
        )
    if list(loaded_vocab) != list(current_vocab_list):
        # Surface first divergence
        n_loaded = len(loaded_vocab)
        n_current = len(current_vocab_list)
        if n_loaded != n_current:
            detail = f"vocab size differs: loaded={n_loaded} current={n_current}"
        else:
            first_diff = next(
                (
                    f"idx {i}: loaded={a!r} current={b!r}"
                    for i, (a, b) in enumerate(zip(loaded_vocab, current_vocab_list))
                    if a != b
                ),
                "size matches but mapping differs",
            )
            detail = f"vocab content differs ({first_diff})"
        raise ValueError(
            f"--load-from vocab mismatch:\n  {detail}\n"
            f"Rung curriculum requires fixed tokenizer across the entire chain. "
            f"If you're loading a Phase 2 GSM8k-tokenizer ckpt into a Phase 3 "
            f"BroadTokenizer model, that's intentional and not supported — "
            f"Phase 3 R0 trains from random init with broad vocab."
        )

    # Normalizer version
    loaded_normalizer = loaded_ckpt_config.get("gsm8k_normalizer_version")
    if loaded_normalizer != current_normalizer_version:
        raise ValueError(
            f"--load-from normalizer mismatch: loaded={loaded_normalizer!r} "
            f"current={current_normalizer_version!r}"
        )

    # Ternary flag — must match (can't switch FP <-> ternary mid-chain)
    loaded_ternary = loaded_ckpt_config.get("use_ternary_bulk", False)
    current_ternary = getattr(current_cfg, "use_ternary_bulk", False)
    if loaded_ternary != current_ternary:
        raise ValueError(
            f"--load-from ternary flag mismatch: loaded={loaded_ternary} "
            f"current={current_ternary}. Curriculum cannot switch FP<->ternary mid-chain."
        )

    # Architecture fields — exact match required
    for field in _ARCH_FIELDS:
        if field not in loaded_ckpt_config:
            raise ValueError(
                f"--load-from ckpt missing arch field {field!r}; cannot verify shape compatibility"
            )
        loaded_val = loaded_ckpt_config[field]
        current_val = getattr(current_cfg, field)
        if loaded_val != current_val:
            raise ValueError(
                f"--load-from {field!r} mismatch: loaded={loaded_val} current={current_val}"
            )
