"""S0 gates for rdt-v2 train-script flag plumbing.

Codex audit `1779311799556-48ff0f15` / `1779311984856-b8487eb1` named the
smallest-S0 assertion as:
    CLI/config -> model.config -> forward uses >=1 non-inert flag
    -> checkpoint reload preserves the same flags

These tests cover the build-time and round-trip half of that gate. The
CLI half is covered structurally — train_pt_delta_mqar.py and
train_code_dt.py now pass each flag as a kwarg through to
`build_copy_augmented_delta`, so a `m.config.<flag>` check after build is
sufficient.
"""
from __future__ import annotations

import torch

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.dt_install import load_dt_checkpoint


_RDTV2_FLAGS_DEFAULT_FALSE = (
    "use_loop_index",
    "use_input_injection",
    "use_gated_attention",
    "use_z_init",
    "use_lecun_init",
    "use_prefix_lm",
    "use_h_rmsnorm",
    "use_short_conv",
    "use_h_layer_stack",
    "use_halt_head",
    "use_carry",
)


def _tiny_kwargs(**overrides):
    kw = dict(
        vocab_size=82, d_model=64, n_heads=32, n_layers=2,
        d_ffn=64, max_len=32, n_copy_heads=4,
    )
    kw.update(overrides)
    return kw


def test_build_threads_flags_through_to_config():
    """S0a smoke: every CLI flag the train scripts now expose maps to a
    `m.config.<flag>` attribute that reflects the kwarg value passed at
    build time.
    """
    torch.manual_seed(0)
    m = build_copy_augmented_delta(
        **_tiny_kwargs(
            use_chunkwise=True,
            n_iterations=2,
            use_loop_index=True,
            use_input_injection=True,
            use_gated_attention=True,
            use_z_init=True,
            use_lecun_init=True,
            use_prefix_lm=True,
            h_cycles=2,
            use_h_rmsnorm=True,
            use_short_conv=True,
            use_h_layer_stack=True,
            use_halt_head=True,
            use_carry=True,
        ),
    )
    cfg = m.config
    assert cfg.use_chunkwise is True
    assert cfg.n_iterations == 2
    assert cfg.h_cycles == 2
    for flag in _RDTV2_FLAGS_DEFAULT_FALSE:
        assert getattr(cfg, flag) is True, f"flag {flag} did not thread through"


def test_baseline_defaults_match_pre_flag_behavior():
    """Behavior with no flags set must match the pre-S0 train-script
    defaults (chunkwise on, everything else off). Guards against
    accidental default-flip in `build_copy_augmented_delta`.
    """
    m = build_copy_augmented_delta(**_tiny_kwargs(use_chunkwise=True))
    cfg = m.config
    assert cfg.use_chunkwise is True
    assert cfg.n_iterations == 1
    assert cfg.h_cycles == 1
    for flag in _RDTV2_FLAGS_DEFAULT_FALSE:
        assert getattr(cfg, flag) is False, f"flag {flag} unexpectedly True at baseline"


def test_h_cycles_and_h_layer_stack_changes_forward_output():
    """S0c sensitivity gate (codex named these two flags as the smallest
    non-inert example). With h_cycles=2 + use_h_layer_stack=True, the
    H boundary runs a full extra layer stack — output MUST differ from
    the h_cycles=1 baseline on the same input.
    """
    idx = torch.tensor([[1, 2, 3, 4, 5, 3, 6, 7, 8, 9]], dtype=torch.long)

    torch.manual_seed(42)
    m_baseline = build_copy_augmented_delta(**_tiny_kwargs(use_chunkwise=True))
    torch.manual_seed(42)
    m_h = build_copy_augmented_delta(
        **_tiny_kwargs(
            use_chunkwise=True,
            h_cycles=2,
            use_h_layer_stack=True,
        ),
    )

    m_baseline.eval()
    m_h.eval()
    with torch.no_grad():
        out_baseline = m_baseline(idx)
        out_h = m_h(idx)

    assert out_baseline.shape == out_h.shape
    # Flag-on output must NOT be bit-identical to baseline. If this
    # passes-by-coincidence, h_cycles + h_layer_stack are inert at this
    # site — investigate the forward path.
    assert not torch.allclose(out_baseline, out_h), (
        "h_cycles=2 + use_h_layer_stack=True produced identical output to baseline; "
        "flag is not causally wired into forward."
    )


def test_checkpoint_roundtrip_preserves_flags_off(tmp_path):
    """S0d round-trip floor: a save-then-load of a default-off model must
    leave every flag False. Guards against load defaults leaking flags
    that the save side didn't explicitly persist.
    """
    torch.manual_seed(1)
    m = build_copy_augmented_delta(**_tiny_kwargs(use_chunkwise=True))

    ckpt_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state": m.state_dict(),
            "config": {
                "vocab_size": 82, "max_len": 32, "d_model": 64,
                "n_heads": 32, "n_layers": 2, "d_ffn": 64,
                "n_copy_heads": 4,
                "use_chunkwise": True,
                "n_iterations": 1, "h_cycles": 1,
                "copy_gate_bias_init": -2.0,
            },
        },
        ckpt_path,
    )

    m2, _ = load_dt_checkpoint(ckpt_path, device="cpu")
    cfg = m2.config
    assert cfg.use_chunkwise is True
    assert cfg.n_iterations == 1
    assert cfg.h_cycles == 1
    for flag in _RDTV2_FLAGS_DEFAULT_FALSE:
        assert getattr(cfg, flag) is False, f"flag {flag} leaked to True after reload"


def test_checkpoint_roundtrip_preserves_non_default_chunkwise_and_chunk_size(tmp_path):
    """S0d gate (codex audit `1779312497386`): the edited CLI now exposes
    `--no-chunkwise` and `--chunk-size`. Save + load must round-trip BOTH —
    a `--no-chunkwise` run reloading as `use_chunkwise=True` would silently
    change the trained forward path.
    """
    torch.manual_seed(3)
    m = build_copy_augmented_delta(
        **_tiny_kwargs(use_chunkwise=False),
    )
    m.config.chunk_size = 16

    ckpt_path = tmp_path / "no_chunkwise.pt"
    torch.save(
        {
            "model_state": m.state_dict(),
            "config": {
                "vocab_size": 82, "max_len": 32, "d_model": 64,
                "n_heads": 32, "n_layers": 2, "d_ffn": 64,
                "n_copy_heads": 4,
                "use_chunkwise": False,
                "chunk_size": 16,
                "copy_gate_bias_init": -2.0,
            },
        },
        ckpt_path,
    )

    m2, _ = load_dt_checkpoint(ckpt_path, device="cpu")
    assert m2.config.use_chunkwise is False
    assert m2.config.chunk_size == 16


def test_checkpoint_roundtrip_accepts_mqar_state_dict_key(tmp_path):
    """`train_pt_delta_mqar.py` saves under `model_state_dict`;
    `train_code_dt.py` saves under `model_state`. Per codex audit
    `1779312497386`, `load_dt_checkpoint` must accept either so MQAR
    checkpoints (and any future GSM8k entrypoint following either
    convention) reload without an external rewrite step.
    """
    torch.manual_seed(4)
    m = build_copy_augmented_delta(**_tiny_kwargs(use_chunkwise=True))
    ckpt_path = tmp_path / "mqar_key.pt"
    torch.save(
        {
            "model_state_dict": m.state_dict(),  # MQAR convention
            "config": {
                "vocab_size": 82, "max_len": 32, "d_model": 64,
                "n_heads": 32, "n_layers": 2, "d_ffn": 64,
                "n_copy_heads": 4,
                "use_chunkwise": True, "chunk_size": 32,
                "copy_gate_bias_init": -2.0,
            },
        },
        ckpt_path,
    )

    m2, _ = load_dt_checkpoint(ckpt_path, device="cpu")
    assert m2.config.use_chunkwise is True


def test_checkpoint_roundtrip_preserves_full_flag_bundle(tmp_path):
    """S0d round-trip gate at codex's locked first-card config. Saves a
    model with the Core-H/L bundle (10 ON / 4 OFF / prefix_lm dropped)
    and verifies every flag survives reload.
    """
    on_flags = dict(
        use_chunkwise=True,
        n_iterations=2,
        h_cycles=2,
        use_loop_index=True,
        use_input_injection=True,
        use_gated_attention=True,
        use_z_init=True,
        use_lecun_init=True,
        use_h_rmsnorm=True,
        use_short_conv=True,
        use_h_layer_stack=True,
    )
    torch.manual_seed(2)
    m = build_copy_augmented_delta(**_tiny_kwargs(**on_flags))

    ckpt_path = tmp_path / "core_hl.pt"
    cfg_dict = {
        "vocab_size": 82, "max_len": 32, "d_model": 64,
        "n_heads": 32, "n_layers": 2, "d_ffn": 64,
        "n_copy_heads": 4,
        "copy_gate_bias_init": -2.0,
        **on_flags,
        # Explicit OFF (matches codex's first-card config).
        "use_prefix_lm": False,
        "use_halt_head": False,
        "use_carry": False,
    }
    torch.save({"model_state": m.state_dict(), "config": cfg_dict}, ckpt_path)

    m2, _ = load_dt_checkpoint(ckpt_path, device="cpu")
    cfg = m2.config
    for k, v in on_flags.items():
        assert getattr(cfg, k) == v, (
            f"flag {k} did not survive reload: expected {v!r}, got {getattr(cfg, k)!r}"
        )
    assert cfg.use_prefix_lm is False
    assert cfg.use_halt_head is False
    assert cfg.use_carry is False
