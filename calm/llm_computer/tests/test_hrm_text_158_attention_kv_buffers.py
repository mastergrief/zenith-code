"""CPU/static tests for HRM attention/KV buffer accounting."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
)
from calm.hrm_text_158.kv_cache import KVCache
from calm.hrm_text_158.native_full_stack import (
    ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES,
    ATTENTION_KV_FAIL_CLOSED_NON_CLAIMS,
    ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
    ATTENTION_KV_FAIL_CLOSED_TARGET_NAME,
    ATTENTION_KV_GQA_REPEAT_CAVEAT,
    ATTENTION_KV_PREFIX_LM_MASK_CAVEAT,
    ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES,
    ATTENTION_KV_RUNTIME_CACHE_SCOPE,
    ATTENTION_KV_BUFFER_SCHEMA_VERSION,
    MATERIALIZED_ALLOCATION_ONLY,
    MODE_ATTENTION_KV_OFF,
    MODE_ATTENTION_KV_RUNTIME_CACHE_ACCOUNTING,
    MODE_LOSSLESS_ATTENTION_KV_EVICTION,
    MODE_LOSSLESS_ATTENTION_KV_OFFLOAD,
    MODE_LOSSY_ATTENTION_KV_COMPRESSION,
    REQUIRED_ATTENTION_KV_MEASUREMENT_FIELDS,
    RUNTIME_CACHE_NOT_STATE_DICT,
    SDPA_WORKSPACE_GPU_MEASURED_DEFERRED,
    TIER1_LOSSLESS_ATTENTION_KV_RELIEF_DEFERRED,
    TIER2_LOSSY_ATTENTION_KV_COMPRESSION_DEFERRED,
    AttentionKVBufferSpec,
    attention_kv_dtype_nbytes,
    build_attention_kv_fail_closed_receipt,
    estimate_attention_kv_buffers,
    validate_attention_kv_fail_closed_receipt,
    validate_attention_kv_measurement,
    validate_attention_kv_mode,
    visible_attention_memory_estimate,
)


def _spec(**kwargs) -> AttentionKVBufferSpec:
    base = dict(
        batch_size=1,
        query_seq_len=8,
        attention_kv_seq_len=8,
        max_seq_len=64,
        hidden_size=512,
        head_dim=128,
        num_heads=4,
        num_kv_heads=4,
        dtype=torch.float32,
        H_cycles=2,
        L_cycles=3,
        layers_per_level=4,
        attn_type="prefixlm",
    )
    base.update(kwargs)
    return AttentionKVBufferSpec(**base)


def _complete_receipt(estimate) -> dict:
    visible = estimate.visible_attention
    return {
        "peak_allocated_bytes": estimate.kv_cache_total_bytes + visible.materialized_allocated_bytes,
        "peak_reserved_bytes": estimate.kv_cache_total_bytes + visible.materialized_allocated_bytes + 4096,
        "wall_clock_per_step_seconds": 0.25,
        "max_safe_batch_size": 8,
        "effective_exposure_per_step": 2048,
        "attention_kv_buffer_schema_version": ATTENTION_KV_BUFFER_SCHEMA_VERSION,
        "kv_cache_key_count": estimate.key_schedule.total_key_count,
        "per_key_k_bytes": estimate.per_key_k_bytes,
        "per_key_v_bytes": estimate.per_key_v_bytes,
        "kv_cache_total_bytes": estimate.kv_cache_total_bytes,
        "gqkv_projection_bytes": visible.gqkv_projection_bytes,
        "prefix_lm_mask_bytes": visible.prefix_lm_mask_bytes,
        "gqa_repeated_kv_bytes": visible.gqa_repeated_kv_bytes,
        "visible_attention_allocated_bytes": visible.materialized_allocated_bytes,
        "view_logical_bytes": visible.view_logical_bytes,
        "materialized_allocation_policy": MATERIALIZED_ALLOCATION_ONLY,
        "runtime_cache_persistence": RUNTIME_CACHE_NOT_STATE_DICT,
        "sdpa_workspace_caveat": SDPA_WORKSPACE_GPU_MEASURED_DEFERRED,
    }


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


def _attention_kv_live_tensor_events():
    torch.manual_seed(2027)
    hrm = HierarchicalReasoningModel(_tiny_config())
    hrm.train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)
    events: list[dict[str, object]] = []

    def seam(family: str, tensor: torch.Tensor, **_: object) -> torch.Tensor:
        if family in ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES:
            events.append(
                {
                    "family": family,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "requires_grad": bool(tensor.requires_grad),
                }
            )
        return tensor

    hrm(
        None,
        x,
        bp_steps=5,
        sep_positions=sep,
        position_ids=pos,
        activation_codec_seam=seam,
    )
    return events


def test_default_kv_cache_formula_matches_48_mib_contract():
    estimate = estimate_attention_kv_buffers(_spec(max_seq_len=384))

    assert estimate.schema_version == ATTENTION_KV_BUFFER_SCHEMA_VERSION
    assert estimate.key_schedule.l_key_count == 24
    assert estimate.key_schedule.h_key_count == 8
    assert estimate.key_schedule.total_key_count == 32
    assert estimate.per_key_k_bytes == 4 * 384 * 128 * 4
    assert estimate.per_key_v_bytes == 4 * 384 * 128 * 4
    assert estimate.per_key_total_bytes == 1_572_864
    assert estimate.kv_cache_total_bytes == 48 * 1024 * 1024
    assert estimate.runtime_cache_persistence == RUNTIME_CACHE_NOT_STATE_DICT


def test_dtype_mapping_batch_scaling_and_invalid_specs():
    assert attention_kv_dtype_nbytes(torch.float32) == 4
    assert attention_kv_dtype_nbytes("torch.float16") == 2
    assert attention_kv_dtype_nbytes("bfloat16") == 2

    b1 = estimate_attention_kv_buffers(_spec(batch_size=1, dtype="float16"))
    b3 = estimate_attention_kv_buffers(_spec(batch_size=3, dtype="float16"))
    assert b3.kv_cache_total_bytes == 3 * b1.kv_cache_total_bytes

    with pytest.raises(ValueError, match="unsupported attention/KV dtype"):
        attention_kv_dtype_nbytes("complex64")
    with pytest.raises(ValueError, match="hidden_size must equal"):
        estimate_attention_kv_buffers(_spec(hidden_size=1024))
    with pytest.raises(ValueError, match="num_heads must be divisible"):
        estimate_attention_kv_buffers(_spec(num_heads=6, num_kv_heads=4, hidden_size=768))


def test_estimator_matches_live_kv_cache_total_memory_bytes_oracle():
    spec = _spec(
        max_seq_len=8,
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_kv_heads=2,
        query_seq_len=1,
        attention_kv_seq_len=1,
    )
    estimate = estimate_attention_kv_buffers(spec)
    cache = KVCache(
        max_seq_len=spec.max_seq_len,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
        dtype=torch.float32,
        device="cpu",
        batch_size=spec.batch_size,
    )
    new_k = torch.zeros(spec.batch_size, spec.num_kv_heads, 1, spec.head_dim)
    new_v = torch.zeros_like(new_k)
    for rec_idx in range(spec.H_cycles * spec.L_cycles):
        for layer_idx in range(spec.layers_per_level):
            cache.update("L", rec_idx, layer_idx, new_k, new_v)
    for rec_idx in range(spec.H_cycles):
        for layer_idx in range(spec.layers_per_level):
            cache.update("H", rec_idx, layer_idx, new_k, new_v)

    assert cache.num_buffers() == estimate.key_schedule.total_key_count == 32
    assert cache.total_memory_bytes() == estimate.kv_cache_total_bytes == 16_384


def test_visible_attention_accounting_separates_allocations_from_views():
    spec = _spec(
        batch_size=2,
        query_seq_len=5,
        attention_kv_seq_len=7,
        max_seq_len=16,
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_kv_heads=2,
        dtype=torch.float16,
    )
    visible = visible_attention_memory_estimate(spec)

    assert visible.gqkv_projection_bytes == 2 * 5 * (2 * 4 + 2 * 2) * 4 * 2
    assert visible.prefix_lm_mask_bytes == 2 * 5 * 5
    assert visible.gqa_repeated_kv_bytes == 2 * 2 * 4 * 7 * 4 * 2
    assert visible.split_view_logical_bytes == visible.gqkv_projection_bytes
    assert visible.transpose_view_logical_bytes == 2 * 5 * (4 + 2 * 2) * 4 * 2
    assert visible.materialized_allocated_bytes == (
        visible.gqkv_projection_bytes
        + visible.prefix_lm_mask_bytes
        + visible.gqa_repeated_kv_bytes
    )
    assert visible.view_logical_bytes > 0
    assert visible.materialized_allocated_bytes != (
        visible.gqkv_projection_bytes
        + visible.prefix_lm_mask_bytes
        + visible.gqa_repeated_kv_bytes
        + visible.view_logical_bytes
    )
    assert visible.sdpa_workspace_caveat == SDPA_WORKSPACE_GPU_MEASURED_DEFERRED

    decode_visible = visible_attention_memory_estimate(spec.__class__(**{
        **spec.__dict__,
        "attn_type": "cached_decode",
    }))
    assert decode_visible.prefix_lm_mask_bytes == 0


def test_measurement_validator_rejects_memory_only_and_alias_only_receipts():
    estimate = estimate_attention_kv_buffers(_spec())
    memory_only = {
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
    }
    with pytest.raises(ValueError, match="wall_clock_per_step_seconds"):
        validate_attention_kv_measurement(memory_only)

    alias_only = _complete_receipt(estimate)
    alias_only["probe_wall_clock_seconds"] = alias_only.pop("wall_clock_per_step_seconds")
    with pytest.raises(ValueError, match="wall_clock_per_step_seconds"):
        validate_attention_kv_measurement(alias_only)

    receipt = _complete_receipt(estimate)
    receipt["probe_wall_clock_seconds"] = 0.1
    validate_attention_kv_measurement(receipt)
    assert set(REQUIRED_ATTENTION_KV_MEASUREMENT_FIELDS) <= set(receipt)


def test_measurement_validator_rejects_schema_caveat_and_total_mismatches():
    estimate = estimate_attention_kv_buffers(_spec())
    receipt = _complete_receipt(estimate)

    wrong_schema = dict(receipt, attention_kv_buffer_schema_version="old")
    with pytest.raises(ValueError, match="attention_kv_buffer_schema_version"):
        validate_attention_kv_measurement(wrong_schema)

    wrong_policy = dict(receipt, materialized_allocation_policy="count_views_too")
    with pytest.raises(ValueError, match="materialized_allocation_policy"):
        validate_attention_kv_measurement(wrong_policy)

    wrong_persistence = dict(receipt, runtime_cache_persistence="state_dict_buffer")
    with pytest.raises(ValueError, match="runtime_cache_persistence"):
        validate_attention_kv_measurement(wrong_persistence)

    wrong_sdpa = dict(receipt, sdpa_workspace_caveat="exact_static_workspace")
    with pytest.raises(ValueError, match="sdpa_workspace_caveat"):
        validate_attention_kv_measurement(wrong_sdpa)

    wrong_kv_total = dict(receipt, kv_cache_total_bytes=receipt["kv_cache_total_bytes"] + 1)
    with pytest.raises(ValueError, match="kv_cache_total_bytes"):
        validate_attention_kv_measurement(wrong_kv_total)

    wrong_visible_total = dict(
        receipt,
        visible_attention_allocated_bytes=(
            receipt["visible_attention_allocated_bytes"] + receipt["view_logical_bytes"]
        ),
    )
    with pytest.raises(ValueError, match="visible_attention_allocated_bytes"):
        validate_attention_kv_measurement(wrong_visible_total)

    bad_reserved = dict(receipt, peak_reserved_bytes=receipt["peak_allocated_bytes"] - 1)
    with pytest.raises(ValueError, match="peak_reserved_bytes"):
        validate_attention_kv_measurement(bad_reserved)


def test_attention_kv_modes_are_runtime_accounting_or_deferred():
    assert validate_attention_kv_mode(MODE_ATTENTION_KV_OFF) == MODE_ATTENTION_KV_OFF
    assert (
        validate_attention_kv_mode(MODE_ATTENTION_KV_RUNTIME_CACHE_ACCOUNTING)
        == MODE_ATTENTION_KV_RUNTIME_CACHE_ACCOUNTING
    )

    for mode in (MODE_LOSSLESS_ATTENTION_KV_OFFLOAD, MODE_LOSSLESS_ATTENTION_KV_EVICTION):
        with pytest.raises(NotImplementedError, match=TIER1_LOSSLESS_ATTENTION_KV_RELIEF_DEFERRED):
            validate_attention_kv_mode(mode)

    with pytest.raises(NotImplementedError, match=TIER2_LOSSY_ATTENTION_KV_COMPRESSION_DEFERRED):
        validate_attention_kv_mode(MODE_LOSSY_ATTENTION_KV_COMPRESSION)


def test_attention_kv_fail_closed_receipt_enumerates_qkv_allowlist_without_flip():
    receipt = build_attention_kv_fail_closed_receipt(
        seam_events=_attention_kv_live_tensor_events()
    )
    validate_attention_kv_fail_closed_receipt(receipt)

    observed_counts = {
        observation.family: observation.observed_count
        for observation in receipt.observed_families
    }
    observed_dtypes = {
        observation.family: observation.dtypes
        for observation in receipt.observed_families
    }

    assert receipt.schema_version == ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION
    assert receipt.target_name == ATTENTION_KV_FAIL_CLOSED_TARGET_NAME
    assert receipt.allowed_observed_families == ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES
    assert receipt.required_observed_families == ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES
    assert receipt.allowed_observed_families == (
        "attn.gqkv.query_post_rope",
        "attn.gqkv.key_post_rope",
        "attn.gqkv.value",
    )
    assert receipt.attention_kv_attention_buffers_sub2_claim is False
    assert receipt.ready_to_flip is False
    assert receipt.real_sub2_representation_present is False
    assert receipt.lossless_eviction_or_offload_proof_present is False
    assert receipt.compression_proof_present is False
    assert receipt.fidelity_acquisition_revalidation_present is False
    assert receipt.no_hidden_bf16_authority_proven is False
    assert receipt.gpu_memory_throughput_receipt_present is False
    assert observed_counts == {
        "attn.gqkv.query_post_rope": 8,
        "attn.gqkv.key_post_rope": 8,
        "attn.gqkv.value": 8,
    }
    assert len(set(observed_counts.values())) == 1
    assert set(observed_dtypes) == set(ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES)
    assert all(dtypes == ("torch.float32",) for dtypes in observed_dtypes.values())
    assert receipt.caveats.prefix_lm_mask_caveat == ATTENTION_KV_PREFIX_LM_MASK_CAVEAT
    assert receipt.caveats.gqa_repeat_caveat == ATTENTION_KV_GQA_REPEAT_CAVEAT
    assert receipt.caveats.sdpa_workspace_caveat == SDPA_WORKSPACE_GPU_MEASURED_DEFERRED
    assert receipt.caveats.runtime_cache_persistence == RUNTIME_CACHE_NOT_STATE_DICT
    assert receipt.caveats.runtime_cache_training_bypass is True
    assert receipt.caveats.runtime_cache_state_dict_claim is False
    assert receipt.caveats.materialized_allocation_policy == MATERIALIZED_ALLOCATION_ONLY
    assert receipt.caveats.runtime_cache_scope == ATTENTION_KV_RUNTIME_CACHE_SCOPE
    assert any("blocker evidence" in non_claim for non_claim in receipt.non_claims)
    assert any("PrefixLM" in non_claim for non_claim in receipt.non_claims)
    assert any("GQA" in non_claim for non_claim in receipt.non_claims)
    assert any("SDPA" in non_claim for non_claim in receipt.non_claims)
    assert any("KVCache" in non_claim for non_claim in receipt.non_claims)
    assert any(".pt" in non_claim for non_claim in receipt.non_claims)


def test_attention_kv_fail_closed_receipt_rejects_missing_unknown_claims_and_compression():
    events = _attention_kv_live_tensor_events()

    with pytest.raises(ValueError, match="missing required observed families"):
        build_attention_kv_fail_closed_receipt(
            seam_events=[
                event for event in events if event["family"] != "attn.gqkv.value"
            ]
        )

    with pytest.raises(ValueError, match="Step 3B allowlist"):
        build_attention_kv_fail_closed_receipt(
            seam_events=[
                *events,
                {
                    "family": "residual.post_attn",
                    "shape": (2, 16, 32),
                    "dtype": "torch.float32",
                    "device": "cpu",
                    "requires_grad": True,
                },
            ]
        )

    with pytest.raises(ValueError, match="requires real representation"):
        build_attention_kv_fail_closed_receipt(
            seam_events=events,
            attention_kv_attention_buffers_sub2_claim=True,
        )

    with pytest.raises(ValueError, match="generic compression/lossy"):
        build_attention_kv_fail_closed_receipt(
            seam_events=events,
            generic_lossy_or_compression_wording=True,
        )

    with pytest.raises(ValueError, match="requires real representation"):
        build_attention_kv_fail_closed_receipt(
            seam_events=events,
            attention_kv_attention_buffers_sub2_claim=True,
            compression_proof_present=True,
            no_hidden_bf16_authority_proven=True,
            gpu_memory_throughput_receipt_present=True,
            ready_to_flip=True,
        )


def test_native_full_stack_exports_attention_kv_contract_surface():
    import calm.hrm_text_158.native_full_stack as native_full_stack

    for name in (
        "ATTENTION_KV_BUFFER_SCHEMA_VERSION",
        "ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION",
        "ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES",
        "ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES",
        "ATTENTION_KV_FAIL_CLOSED_NON_CLAIMS",
        "AttentionKVBufferSpec",
        "AttentionKVFailClosedReceipt",
        "estimate_attention_kv_buffers",
        "build_attention_kv_fail_closed_receipt",
        "validate_attention_kv_fail_closed_receipt",
        "validate_attention_kv_measurement",
        "SDPA_WORKSPACE_GPU_MEASURED_DEFERRED",
        "MATERIALIZED_ALLOCATION_ONLY",
    ):
        assert hasattr(native_full_stack, name)
