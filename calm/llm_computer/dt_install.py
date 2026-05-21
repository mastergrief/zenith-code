"""DT (delta-transducer) install scaffold for Gemma CardSlot.

Loads a trained DT checkpoint from `dt_code_skel_best.pt` and wires
it into Gemma's residual stream via the CardSlot + VerificationHook
pattern proven in R22 MQAR (commit `9691e06`, threshold 14.5).

This is the next-step scaffold after `scripts/train_code_dt.py` lands
a checkpoint. Not run against live daemon yet — waiting on stable
daemon + validated autoreg rate.

Usage:
    from calm.llm_computer.dt_install import install_code_dt
    slot, hook = install_code_dt(
        gemma, tokenizer,
        checkpoint_path="calm/hrm/checkpoints/dt_code_skel_best.pt",
        layer_idx=30,
        ch_off=2400,
    )
    # Gemma.generate now biased toward emitting `def fn(args):` on code prompts

Pattern follows R22 (`delta_rule.md` §R22 install):
  - CardSlot at L30 with preserve=False
  - write_margin = min_margin = 14.5 (4-gate default)
  - DT's copy-augmented output biases Gemma's decode via VerificationHook
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch


def load_dt_checkpoint(
    path: str | Path,
    device: str = "cuda",
):
    """Load a trained DT checkpoint. Returns (model, config).

    Checkpoint layout (from train_code_dt.py):
      {
        "model_state": state_dict,
        "config": {vocab_size, max_len, d_model, n_heads, n_layers,
                   d_ffn, n_copy_heads, use_chunkwise},
        "epoch": int,
        "val_autoreg": float,
        "n_train": int, "n_val": int,
      }
    """
    from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # HRM-Text-derived flags: defaults preserve backward-compat for old
    # checkpoints (which don't include these keys). New checkpoints saved
    # via the training scripts MUST include them so the architecture round-trips.
    model = build_copy_augmented_delta(
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ffn=cfg["d_ffn"],
        n_copy_heads=cfg["n_copy_heads"],
        copy_gate_bias_init=cfg.get("copy_gate_bias_init", -2.0),
        use_chunkwise=cfg.get("use_chunkwise", False),
        n_iterations=cfg.get("n_iterations", 1),
        use_loop_index=cfg.get("use_loop_index", False),
        use_input_injection=cfg.get("use_input_injection", False),
        use_gated_attention=cfg.get("use_gated_attention", False),
        use_z_init=cfg.get("use_z_init", False),
        use_lecun_init=cfg.get("use_lecun_init", False),
        use_prefix_lm=cfg.get("use_prefix_lm", False),
        h_cycles=cfg.get("h_cycles", 1),
        use_h_rmsnorm=cfg.get("use_h_rmsnorm", False),
        use_short_conv=cfg.get("use_short_conv", False),
        use_h_layer_stack=cfg.get("use_h_layer_stack", False),
        use_halt_head=cfg.get("use_halt_head", False),
        use_carry=cfg.get("use_carry", False),
        use_pre_rmsnorm=cfg.get("use_pre_rmsnorm", False),
    ).to(device)
    # `chunk_size` is config-only (not a builder kwarg) — restore after build
    # so non-default chunk sizes round-trip. Default 32 matches DeltaNetConfig.
    model.config.chunk_size = cfg.get("chunk_size", 32)
    # `train_pt_delta_mqar.py` historically saves under "model_state_dict";
    # `train_code_dt.py` saves under "model_state". Accept either.
    state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError(
            f"checkpoint {path!s} has neither 'model_state' nor 'model_state_dict'"
        )
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def build_dt_adapter(model, tokenizer):
    """Build the `card_input_fn` adapter that converts Gemma's residual
    at the injection layer into a DT input sequence.

    The adapter extracts the Gemma prompt (decoded from the token IDs
    seen so far in this generation), char-tokenizes via the code-DT
    vocab, and prepends <bos> + <sep>. Returns a tensor ready for
    `model.forward(input_ids)`.

    Not yet wired — needs the full Gemma-substrate wire-up analogous
    to `r22_install_mqar_card.py::parse_mqar_prompt`.
    """
    from calm.hrm.code_dt_data import _CODE_CHAR_TO_ID, code_tokenize

    sep = _CODE_CHAR_TO_ID["<sep>"]

    def adapter(residual_at_layer, prompt_text: str) -> torch.Tensor:
        """Convert a Gemma prompt to DT input IDs.
        Args:
            residual_at_layer: (B, S, d_model) Gemma residual — unused here
                since we use the prompt text directly, but kept for
                interface compat with future attention-based adapters.
            prompt_text: the Gemma prompt text for this batch element.
        Returns:
            (1, L) tensor of DT token IDs.
        """
        prefix = code_tokenize(prompt_text, add_bos=True, add_eos=False) + [sep]
        # Truncate if too long for DT max_len
        if len(prefix) > model.config.max_len - 40:  # reserve for skeleton gen
            prefix = [prefix[0]] + prefix[-(model.config.max_len - 41):]
        return torch.tensor([prefix], dtype=torch.long,
                            device=next(model.parameters()).device)

    return adapter


def build_dt_writer(model, tokenizer):
    """Build the `output_fn` writer that converts DT output log-probs
    to a residual-channel vector to write into Gemma's reserved channels.

    The writer greedy-decodes a short skeleton prefix from DT, maps the
    chars back to Gemma BPE tokens via a small embedding lookup, and
    returns the sequence-averaged token embedding to write at the slot's
    reserved channels.

    Placeholder for now — the actual wiring needs the CardSlot protocol
    which expects a callable returning (d_card,) fp32 vector per forward.
    """
    # TODO: implement once daemon stable for test iteration.
    def writer(dt_output_log_probs):
        # For now: simple argmax → char → average over short prefix.
        argmax_ids = dt_output_log_probs.argmax(dim=-1)[0].tolist()
        return argmax_ids  # Consumer decides what to do with char ids

    return writer


def install_code_dt(
    gemma,
    tokenizer,
    checkpoint_path: str | Path = "calm/hrm/checkpoints/dt_code_skel_best.pt",
    layer_idx: int = 30,
    ch_off: int = 2400,
    d_card: int = 80,
    write_margin: float = 14.5,
    min_margin: float = 14.5,
) -> Tuple[object, object]:
    """End-to-end: load DT checkpoint, install as CardSlot at layer_idx,
    register VerificationHook. Mirrors R22 MQAR install pattern.

    NOT INVOKED ON LIVE DAEMON YET — pending stable daemon + validated
    autoreg rate on the DT checkpoint. This function is the scaffold
    that glues the pieces together for the eventual wire-up.

    Returns (CardSlot, VerificationHook). Caller is responsible for
    retaining references to prevent GC, and for calling `detach()` on
    the slot when done.
    """
    from calm.llm_computer.gemma_substrate import CardSlot, VerificationHook

    dt_model, ckpt = load_dt_checkpoint(checkpoint_path)
    print(f"[dt-install] loaded DT checkpoint "
          f"val_autoreg={ckpt.get('val_autoreg', 'n/a')} "
          f"from epoch {ckpt.get('epoch', '?')}")

    adapter = build_dt_adapter(dt_model, tokenizer)
    writer = build_dt_writer(dt_model, tokenizer)

    slot = CardSlot(
        layer_idx=layer_idx, ch_off=ch_off, card=dt_model, d_card=d_card,
        card_input_fn=adapter, use_full_residual=True, output_fn=writer,
    )
    slot.attach(gemma, preserve=False, write_margin=write_margin)

    hook = VerificationHook(slot, boost=50.0, min_margin=min_margin)
    gemma.verification_hooks.append(hook)

    return slot, hook
