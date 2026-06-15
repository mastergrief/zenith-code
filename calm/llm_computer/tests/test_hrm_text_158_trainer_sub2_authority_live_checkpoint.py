"""P1a live checkpoint dual-consumer load + production-lifecycle parity tests."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    load_train_checkpoint_into_model,
    save_trainer_sub2_live_checkpoint_envelope,
    select_trainer_eligible_bitlinears,
)

W6_PARENT_PATH = (
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_"
    "anchorsv1r3_from_L0b_final_step01500.pt"
)
W6_PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
P1_PARITY_ATOL = 1e-5


def _tiny_hrm_config(**overrides) -> HierarchicalReasoningModelConfig:
    base = dict(
        max_seq_len=32,
        n_layers=2,
        hidden_size=32,
        num_heads=2,
        expansion=4,
        H_cycles=2,
        L_cycles=2,
        half_layers=True,
        bp_warmup_ratio=0.0,
        bp_min_steps=2,
        bp_max_steps=2,
        use_ternary_bulk=True,
    )
    base.update(overrides)
    return HierarchicalReasoningModelConfig(**base)


def _make_ternary_lmhead(seed: int = 158) -> LMHead:
    torch.manual_seed(seed)
    hrm = HierarchicalReasoningModel(_tiny_hrm_config())
    return LMHead(hrm, LMHeadConfig(vocab_size=98))


def _fresh_copy(m: LMHead) -> LMHead:
    hrm = HierarchicalReasoningModel(_tiny_hrm_config())
    return LMHead(hrm, LMHeadConfig(vocab_size=98))


def _legacy_checkpoint(model: LMHead) -> dict:
    return {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "config": {"use_ternary_bulk": True},
        "step": 0,
        "epoch": 0,
        "source_pin": "test",
    }


def _p1_envelope(model: LMHead) -> dict:
    return save_trainer_sub2_live_checkpoint_envelope(
        model,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
        step=0,
        config={"use_ternary_bulk": True},
        source_pin="test",
        epoch=0,
    )


def _fixed_batches(device: str = "cpu") -> dict[str, dict]:
    torch.manual_seed(2026)
    main_inputs = torch.randint(0, 98, (2, 16), dtype=torch.long, device=device)
    main_sep = torch.tensor([4, 6], dtype=torch.long, device=device)
    main_pos = torch.arange(16, dtype=torch.long, device=device).unsqueeze(0).expand(2, -1)
    main_batch = {
        "inputs": main_inputs,
        "sep_positions": main_sep,
        "position_ids": main_pos,
    }

    retained_inputs = torch.randint(0, 98, (3, 14), dtype=torch.long, device=device)
    retained_sep = torch.tensor([3, 5, 7], dtype=torch.long, device=device)
    retained_pos = torch.arange(14, dtype=torch.long, device=device).unsqueeze(0).expand(3, -1)
    retained_labels = torch.full((3, 14), -100, dtype=torch.long, device=device)
    retained_labels[:, -1] = torch.randint(0, 98, (3,), device=device)
    retained_batch = {
        "inputs": retained_inputs,
        "sep_positions": retained_sep,
        "position_ids": retained_pos,
        "labels": retained_labels,
    }
    retained_forward_batch = {
        "inputs": retained_inputs,
        "sep_positions": retained_sep,
        "position_ids": retained_pos,
    }
    return {
        "main_kl": main_batch,
        "retained_fallback": retained_batch,
        "cache_builder": retained_forward_batch,
    }


def _forward_logits(model: LMHead, batch: dict, *, bp_steps: int | None = None) -> torch.Tensor:
    total_steps = 10
    extras = model.compute_train_extra_args(0, total_steps)
    if bp_steps is not None:
        extras = dict(extras)
        extras["bp_steps"] = int(bp_steps)
    model.eval()
    with torch.no_grad():
        if "labels" in batch:
            _carry, _loss, metrics = model(None, batch, return_logits=True, **extras)
            return metrics["logits"].detach().cpu()
        _carry, logits = model(None, batch, **extras)
        return logits.detach().cpu()


def _parity_gate(logits_p1: torch.Tensor, logits_ref: torch.Tensor) -> float:
    assert torch.isfinite(logits_p1).all()
    assert torch.isfinite(logits_ref).all()
    max_abs_diff = float((logits_p1 - logits_ref).abs().max().item())
    assert torch.allclose(logits_p1, logits_ref, rtol=0.0, atol=P1_PARITY_ATOL), (
        f"max_abs_diff={max_abs_diff}"
    )
    return max_abs_diff


@pytest.fixture
def parity_models():
    source = _make_ternary_lmhead()
    legacy_ckpt = _legacy_checkpoint(source)
    p1_ckpt = _p1_envelope(source)

    parent_ref = _fresh_copy(source)
    load_train_checkpoint_into_model(
        parent_ref,
        legacy_ckpt,
        use_ternary_bulk=True,
        inference_only=True,
        sub2_live_enabled=False,
    )
    parent_ref.eval()

    parent_p1 = _fresh_copy(source)
    load_train_checkpoint_into_model(
        parent_p1,
        p1_ckpt,
        use_ternary_bulk=True,
        inference_only=True,
        sub2_live_enabled=True,
    )
    parent_p1.eval()

    train_m = _fresh_copy(source)
    train_result = load_train_checkpoint_into_model(
        train_m,
        p1_ckpt,
        use_ternary_bulk=True,
        inference_only=False,
        sub2_live_enabled=True,
    )
    assert train_result.routing == "p1_live"
    return parent_ref, parent_p1, train_m


def test_p1_v12_dual_consumer_production_parent_forward(parity_models):
    parent_ref, parent_p1, _train_m = parity_models
    batches = _fixed_batches()
    with torch.no_grad():
        logits = _forward_logits(parent_p1, batches["main_kl"])
    assert torch.isfinite(logits).all()
    # sanity: P1 path should match legacy reference on main batch
    logits_ref = _forward_logits(parent_ref, batches["main_kl"])
    _parity_gate(logits, logits_ref)


def test_p1_v13_production_lifecycle_parity_all_three_sites(parity_models):
    parent_ref, parent_p1, _train_m = parity_models
    batches = _fixed_batches()
    max_diffs: dict[str, float] = {}

    logits_ref_main = _forward_logits(parent_ref, batches["main_kl"])
    logits_p1_main = _forward_logits(parent_p1, batches["main_kl"])
    max_diffs["main_kl"] = _parity_gate(logits_p1_main, logits_ref_main)

    logits_ref_retained = _forward_logits(parent_ref, batches["retained_fallback"])
    logits_p1_retained = _forward_logits(parent_p1, batches["retained_fallback"])
    max_diffs["retained_fallback"] = _parity_gate(logits_p1_retained, logits_ref_retained)

    logits_ref_cache = _forward_logits(parent_ref, batches["cache_builder"], bp_steps=2)
    logits_p1_cache = _forward_logits(parent_p1, batches["cache_builder"], bp_steps=2)
    max_diffs["cache_builder"] = _parity_gate(logits_p1_cache, logits_ref_cache)

    assert max_diffs["main_kl"] <= P1_PARITY_ATOL
    assert max_diffs["retained_fallback"] <= P1_PARITY_ATOL
    assert max_diffs["cache_builder"] <= P1_PARITY_ATOL


def test_p1_v14_legacy_w6_dual_consumer_flag_off():
    repo_root = Path(__file__).resolve().parents[3]
    ckpt_path = repo_root / W6_PARENT_PATH
    if not ckpt_path.is_file():
        pytest.skip(f"W6 parent checkpoint not present at {ckpt_path}")
    actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert actual_sha == W6_PARENT_SHA

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_blob = loaded["config"]
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=cfg_blob["max_seq_len"],
        n_layers=cfg_blob["n_layers"],
        hidden_size=cfg_blob["hidden_size"],
        num_heads=cfg_blob["num_heads"],
        expansion=cfg_blob["expansion"],
        H_cycles=cfg_blob["H_cycles"],
        L_cycles=cfg_blob["L_cycles"],
        half_layers=cfg_blob["half_layers"],
        bp_warmup_ratio=cfg_blob["bp_warmup_ratio"],
        bp_min_steps=cfg_blob["bp_min_steps"],
        bp_max_steps=cfg_blob["bp_max_steps"],
        use_ternary_bulk=cfg_blob.get("use_ternary_bulk", False),
    )
    vocab_size = int(cfg_blob["vocab_size"])

    child = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=vocab_size))
    parent = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=vocab_size))

    child_result = load_train_checkpoint_into_model(
        child,
        loaded,
        use_ternary_bulk=bool(cfg.use_ternary_bulk),
        inference_only=False,
        sub2_live_enabled=False,
    )
    parent_result = load_train_checkpoint_into_model(
        parent,
        loaded,
        use_ternary_bulk=bool(cfg.use_ternary_bulk),
        inference_only=True,
        sub2_live_enabled=False,
    )
    assert child_result.routing == "legacy"
    assert parent_result.routing == "legacy"
    assert child_result.authority_states is None
    assert parent_result.authority_states is None

    post_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert post_sha == W6_PARENT_SHA


def test_p1_install_uses_cached_weight_not_master_weight(parity_models):
    _parent_ref, parent_p1, _train_m = parity_models
    eligible = select_trainer_eligible_bitlinears(parent_p1, use_ternary_bulk=True)
    first = eligible[sorted(eligible)[0]]
    assert first._cached_active is True
    assert first._cached_weight is not None
    assert getattr(first, "_p1_persistent_eval_authority_installed", False) is True
    poisoned = copy.deepcopy(parent_p1)
    poisoned_eligible = select_trainer_eligible_bitlinears(poisoned, use_ternary_bulk=True)
    with torch.no_grad():
        for module in poisoned_eligible.values():
            module.weight.fill_(999.0)
    batches = _fixed_batches()
    logits_clean = _forward_logits(parent_p1, batches["main_kl"])
    logits_poisoned = _forward_logits(poisoned, batches["main_kl"])
    assert torch.allclose(logits_clean, logits_poisoned, rtol=0.0, atol=P1_PARITY_ATOL)
