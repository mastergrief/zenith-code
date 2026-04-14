"""SubstrateHRM — Small2DTransformer-based seq2seq for NL → expression.

Option 2 MVP of "compile HRM into the substrate." The existing HRM uses
RoPE + RMSNorm + GLU + nested recurrence — none of those primitives are
in the compiled LLM-Computer substrate. This model uses ONLY the
substrate's primitives (learned positional embedding, softmax/hard-max
attention, ReGLU-shape FFN via GLU with ReLU gate) so its trained
weights *live on the same runtime* as compiled programs.

Decoder-only, GPT-style: input is `<bos> NL_tokens <sep> expression
<eos>`, trained with teacher forcing on a causal mask. Loss is only on
the expression positions (the prefix is free-input).

Inference: prompt with `<bos> NL_tokens <sep>`, decode autoregressively
until `<eos>`.

Intentionally decoder-only: keeps the compile target identical to
`Small2DTransformer` so the next step (fusing with compiled arithmetic
layers) can share the exact residual stream.
"""

from __future__ import annotations

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


def build_substrate_hrm(vocab_size: int = 80, d_model: int = 64,
                         n_heads: int = 32, n_layers: int = 4,
                         d_ffn: int = 128, max_len: int = 128,
                         use_hard_max: bool = False) -> Small2DTransformer:
    """Initialize a substrate-native seq2seq model for training.

    Defaults picked to match HRM's ~48K param budget: d_model=64,
    n_heads=32 (d_head=2), n_layers=4, d_ffn=128 → ~160K params.
    Softmax attention by default (gradients need to flow).
    """
    cfg = Small2DConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        use_hard_max=use_hard_max,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return Small2DTransformer(cfg)
