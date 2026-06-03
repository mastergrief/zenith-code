"""CPU/static tests for lossless HRM activation-relief policy."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID
from calm.hrm_text_158.native_full_stack import (
    DEFERRED_GPU_MEASUREMENT_NOTE,
    MODE_LOSSLESS_RECOMPUTE,
    MODE_LOSSY_ACTIVATION_STORAGE,
    REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS,
    TERMINAL_DEPENDENT_TUNING_FIELDS,
    TIER1_LOSSLESS_RECOMPUTE,
    TIER2_LOSSY_ACTIVATION_STORAGE_DEFERRED,
    ActivationReliefPolicy,
    normalize_activation_relief_policy,
    recurrence_checkpoint_decisions,
    validate_activation_relief_measurement,
)


def _tiny_config() -> HierarchicalReasoningModelConfig:
    return HierarchicalReasoningModelConfig(
        max_seq_len=32,
        n_layers=2,
        hidden_size=32,
        num_heads=2,
        expansion=4,
        H_cycles=2,
        L_cycles=3,
        half_layers=True,
    )


def _make_model_pair(seed: int = 2026) -> tuple[LMHead, LMHead]:
    torch.manual_seed(seed)
    base = LMHead(HierarchicalReasoningModel(_tiny_config()), LMHeadConfig(vocab_size=98))
    clone = LMHead(HierarchicalReasoningModel(_tiny_config()), LMHeadConfig(vocab_size=98))
    clone.load_state_dict(base.state_dict())
    return base, clone


def _batch() -> dict:
    torch.manual_seed(1337)
    B, S = 2, 16
    ids = torch.randint(0, 98, (B, S), dtype=torch.long)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
    labels = torch.full((B, S), IGNORE_LABEL_ID, dtype=torch.long)
    labels[0, 6:11] = torch.randint(1, 98, (5,))
    labels[1, 8:13] = torch.randint(1, 98, (5,))
    return {"inputs": ids, "sep_positions": sep, "position_ids": pos, "labels": labels}


def _loss_logits_and_named_grads(
    model: LMHead,
    *,
    activation_relief_policy=None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss, metrics = model(
        None,
        _batch(),
        return_logits=True,
        bp_steps=5,
        activation_relief_policy=activation_relief_policy,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = {
        name: (param.grad.detach().clone() if param.grad is not None else None)
        for name, param in model.named_parameters()
    }
    return loss.detach(), metrics["logits"].detach(), grads


def _assert_named_grads_identical(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> None:
    assert set(left) == set(right)
    for name in left:
        if left[name] is None or right[name] is None:
            assert left[name] is None and right[name] is None, f"grad presence mismatch for {name}"
        else:
            torch.testing.assert_close(left[name], right[name], atol=0.0, rtol=0.0)


def test_policy_contract_is_lossless_first_and_terminal_sizing_is_abstract():
    policy = ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE).validate()

    assert policy.enabled is True
    assert policy.tier == TIER1_LOSSLESS_RECOMPUTE
    assert policy.use_reentrant is False
    assert policy.preserve_rng_state is True
    assert "gpu:0" in DEFERRED_GPU_MEASUREMENT_NOTE
    assert {
        "profile_batch_candidates",
        "profile_sequence_lengths",
        "profile_bp_steps",
        "resource_lane_device",
    } <= set(TERMINAL_DEPENDENT_TUNING_FIELDS)

    lossy = ActivationReliefPolicy(mode=MODE_LOSSY_ACTIVATION_STORAGE)
    assert lossy.tier == TIER2_LOSSY_ACTIVATION_STORAGE_DEFERRED
    with pytest.raises(NotImplementedError, match="Tier-2/deferred"):
        lossy.validate()
    with pytest.raises(ValueError, match="use_reentrant=False"):
        ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE, use_reentrant=True).validate()
    with pytest.raises(ValueError, match="preserve_rng_state=True"):
        ActivationReliefPolicy(
            mode=MODE_LOSSLESS_RECOMPUTE,
            preserve_rng_state=False,
        ).validate()


def test_policy_is_consumed_at_hrm_boundary_and_not_leaked_to_levels():
    hrm = HierarchicalReasoningModel(_tiny_config())
    seen: list[dict] = []

    original_l = hrm.L_level.forward
    original_h = hrm.H_level.forward

    def l_wrapper(hidden_states, input_injection, **kwargs):
        seen.append(dict(kwargs))
        return original_l(hidden_states, input_injection, **kwargs)

    def h_wrapper(hidden_states, input_injection, **kwargs):
        seen.append(dict(kwargs))
        return original_h(hidden_states, input_injection, **kwargs)

    hrm.L_level.forward = l_wrapper  # type: ignore[method-assign]
    hrm.H_level.forward = h_wrapper  # type: ignore[method-assign]
    x = torch.randn(1, 8, 32)
    sep = torch.tensor([3], dtype=torch.long)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)

    hrm(None, x, bp_steps=2, sep_positions=sep, position_ids=pos, activation_relief_policy=False)

    assert seen
    assert not any("activation_relief_policy" in kwargs for kwargs in seen)


def test_checkpoint_decisions_only_target_grad_enabled_recurrence_calls():
    policy = ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE)

    bp2 = recurrence_checkpoint_decisions(policy, H_cycles=2, L_cycles=3, bp_steps=2)
    assert [(d.level, d.rec_idx) for d in bp2 if d.checkpoint] == [("L", 5), ("H", 1)]
    assert [(d.level, d.rec_idx) for d in bp2 if d.scheduled_grad_enabled] == [
        ("L", 5),
        ("H", 1),
    ]

    bp5 = recurrence_checkpoint_decisions(policy, H_cycles=2, L_cycles=3, bp_steps=5)
    assert [(d.level, d.rec_idx) for d in bp5 if d.checkpoint] == [
        ("H", 0),
        ("L", 3),
        ("L", 4),
        ("L", 5),
        ("H", 1),
    ]

    no_outer_grad = recurrence_checkpoint_decisions(
        policy,
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
        outer_grad_enabled=False,
    )
    assert not any(d.checkpoint for d in no_outer_grad)
    assert not any(d.scheduled_grad_enabled for d in no_outer_grad)

    eval_mode = recurrence_checkpoint_decisions(
        policy,
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
        module_training=False,
    )
    assert any(d.scheduled_grad_enabled for d in eval_mode)
    assert not any(d.checkpoint for d in eval_mode)


def test_kv_cache_with_enabled_policy_rejects_before_cache_side_effects():
    hrm = HierarchicalReasoningModel(_tiny_config())
    hrm.train()
    x = torch.randn(1, 8, 32, requires_grad=True)
    sep = torch.tensor([3], dtype=torch.long)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)

    class _Cache:
        calls = 0

        def update(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("kv_cache.update must not be reached")

    cache = _Cache()
    with pytest.raises(ValueError, match="kv_cache is present"):
        hrm(
            None,
            x,
            bp_steps=5,
            sep_positions=sep,
            position_ids=pos,
            kv_cache=cache,
            activation_relief_policy=MODE_LOSSLESS_RECOMPUTE,
        )
    assert cache.calls == 0


def test_measurement_contract_rejects_memory_only_receipts():
    memory_only = {
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    with pytest.raises(ValueError, match="wall_clock_per_step_seconds"):
        validate_activation_relief_measurement(memory_only)

    complete = {
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "wall_clock_per_step_seconds": 0.25,
        "max_safe_batch_size": 8,
        "effective_exposure_per_step": 2048,
    }
    validate_activation_relief_measurement(complete)
    assert set(REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS) <= set(complete)


def test_default_off_and_enabled_recompute_are_bit_identical_in_loss_logits_and_all_named_grads():
    baseline, explicit_off = _make_model_pair()
    loss_a, logits_a, grads_a = _loss_logits_and_named_grads(baseline)
    loss_b, logits_b, grads_b = _loss_logits_and_named_grads(
        explicit_off,
        activation_relief_policy=normalize_activation_relief_policy(False),
    )

    torch.testing.assert_close(loss_a, loss_b, atol=0.0, rtol=0.0)
    torch.testing.assert_close(logits_a, logits_b, atol=0.0, rtol=0.0)
    _assert_named_grads_identical(grads_a, grads_b)

    default_off, recompute = _make_model_pair()
    loss_off, logits_off, grads_off = _loss_logits_and_named_grads(default_off)
    loss_recompute, logits_recompute, grads_recompute = _loss_logits_and_named_grads(
        recompute,
        activation_relief_policy=ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE),
    )

    torch.testing.assert_close(loss_off, loss_recompute, atol=0.0, rtol=0.0)
    torch.testing.assert_close(logits_off, logits_recompute, atol=0.0, rtol=0.0)
    _assert_named_grads_identical(grads_off, grads_recompute)
