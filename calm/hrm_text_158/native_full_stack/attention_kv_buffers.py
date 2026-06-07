"""CPU/static attention/KV buffer accounting for HRM-Text-1.58.

This slice is contract/estimator only. It accounts for visible materialized
attention surfaces plus the runtime KV-cache allocation contract, and it keeps
framework-internal SDPA workspace as GPU-measured/deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


ATTENTION_KV_BUFFER_SCHEMA_VERSION = (
    "hrm_text_158_attention_kv_buffers/v0.contract_estimator"
)

MODE_ATTENTION_KV_OFF = "off"
MODE_ATTENTION_KV_RUNTIME_CACHE_ACCOUNTING = "runtime_kv_cache_accounting"
MODE_LOSSLESS_ATTENTION_KV_OFFLOAD = "lossless_attention_kv_offload"
MODE_LOSSLESS_ATTENTION_KV_EVICTION = "lossless_attention_kv_eviction"
MODE_LOSSY_ATTENTION_KV_COMPRESSION = "lossy_attention_kv_compression"

TIER1_LOSSLESS_ATTENTION_KV_RELIEF_DEFERRED = (
    "tier1_lossless_attention_kv_relief_deferred"
)
TIER2_LOSSY_ATTENTION_KV_COMPRESSION_DEFERRED = (
    "tier2_lossy_attention_kv_compression_deferred"
)

MATERIALIZED_ALLOCATION_ONLY = "materialized_allocation_only"
RUNTIME_CACHE_NOT_STATE_DICT = "runtime_only_not_state_dict"
SDPA_WORKSPACE_GPU_MEASURED_DEFERRED = "sdpa_workspace_gpu_measured_deferred"

ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_attention_kv_fail_closed/v0.live_tensor_seams"
)
ATTENTION_KV_FAIL_CLOSED_TARGET_NAME = "step3b_attention_kv_fail_closed"
ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES = (
    "attn.gqkv.query_post_rope",
    "attn.gqkv.key_post_rope",
    "attn.gqkv.value",
)
ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES = ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES
ATTENTION_KV_BLOCKED_REASON = (
    "fail-closed attention/KV live-tensor harness only; live BF16/FP q/k/v "
    "tensor seams plus estimator/deferred runtime cache surfaces are observed "
    "and no real sub2/eviction/offload/compression/no-hidden-BF16/GPU "
    "memory-throughput proof is present"
)
ATTENTION_KV_PREFIX_LM_MASK_CAVEAT = (
    "PrefixLM mask is a materialized attention surface, not an observed-family "
    "substitute or sub2 proof"
)
ATTENTION_KV_GQA_REPEAT_CAVEAT = (
    "GQA repeated K/V tensors are materialized attention surfaces, not an "
    "observed-family substitute or sub2 proof"
)
ATTENTION_KV_RUNTIME_CACHE_SCOPE = (
    "runtime KVCache is inference/probe-only, not state_dict, and the training "
    "path bypasses cached attention"
)
ATTENTION_KV_FAIL_CLOSED_NON_CLAIMS = (
    "attention q/k/v live-tensor observation is not learning, acquisition, retention, or throughput",
    "observer callbacks returning BF16/FP q/k/v tensors are blocker evidence, not sub2 credit",
    "PrefixLM mask and GQA repeated K/V surfaces are caveats/non-claims, not observed-family substitutes",
    "SDPA workspace remains GPU-measured/deferred and is not proven by this CPU receipt",
    "runtime KVCache is inference/probe-only, not state_dict, and training bypasses cached attention",
    "this receipt does not launch GPU, prove memory/throughput relief, write checkpoints, or mutate .pt artifacts",
)

REQUIRED_ATTENTION_KV_MEASUREMENT_FIELDS = (
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "wall_clock_per_step_seconds",
    "max_safe_batch_size",
    "effective_exposure_per_step",
    "attention_kv_buffer_schema_version",
    "kv_cache_key_count",
    "per_key_k_bytes",
    "per_key_v_bytes",
    "kv_cache_total_bytes",
    "gqkv_projection_bytes",
    "prefix_lm_mask_bytes",
    "gqa_repeated_kv_bytes",
    "visible_attention_allocated_bytes",
    "view_logical_bytes",
    "materialized_allocation_policy",
    "runtime_cache_persistence",
    "sdpa_workspace_caveat",
)

_DTYPE_BYTE_WIDTHS = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}
_DTYPE_BY_NAME = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def normalize_attention_kv_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Return a supported torch dtype for attention/KV byte accounting."""

    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        if dtype_name in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[dtype_name]
        valid = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(f"unsupported attention/KV dtype {dtype!r}; valid={valid}")
    try:
        if dtype in _DTYPE_BYTE_WIDTHS:
            return dtype
    except TypeError:
        pass
    raise ValueError(f"unsupported attention/KV dtype {dtype!r}")


def attention_kv_dtype_nbytes(dtype: torch.dtype | str) -> int:
    """Byte width for dtypes accepted by the attention/KV estimator."""

    return _DTYPE_BYTE_WIDTHS[normalize_attention_kv_dtype(dtype)]


@dataclass(frozen=True)
class AttentionKVBufferSpec:
    batch_size: int
    query_seq_len: int
    attention_kv_seq_len: int
    max_seq_len: int
    hidden_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    dtype: torch.dtype | str
    H_cycles: int
    L_cycles: int
    layers_per_level: int
    attn_type: str = "prefixlm"

    def validate(self) -> "AttentionKVBufferSpec":
        for name, value in (
            ("batch_size", self.batch_size),
            ("query_seq_len", self.query_seq_len),
            ("attention_kv_seq_len", self.attention_kv_seq_len),
            ("max_seq_len", self.max_seq_len),
            ("hidden_size", self.hidden_size),
            ("head_dim", self.head_dim),
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("H_cycles", self.H_cycles),
            ("L_cycles", self.L_cycles),
            ("layers_per_level", self.layers_per_level),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        normalize_attention_kv_dtype(self.dtype)
        if self.attention_kv_seq_len > self.max_seq_len:
            raise ValueError("attention_kv_seq_len must be <= max_seq_len")
        if self.query_seq_len > self.max_seq_len:
            raise ValueError("query_seq_len must be <= max_seq_len")
        if self.attn_type not in {"prefixlm", "causal", "cached_decode"}:
            raise ValueError(f"unsupported attention type {self.attn_type!r}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        return self

    @property
    def dtype_bytes(self) -> int:
        return attention_kv_dtype_nbytes(self.dtype)

    @property
    def total_gqkv_heads(self) -> int:
        return 2 * self.num_heads + 2 * self.num_kv_heads

    @property
    def kv_repeat(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass(frozen=True)
class AttentionKVKeySchedule:
    H_cycles: int
    L_cycles: int
    layers_per_level: int
    l_key_count: int
    h_key_count: int
    total_key_count: int


@dataclass(frozen=True)
class VisibleAttentionMemoryEstimate:
    gqkv_projection_bytes: int
    prefix_lm_mask_bytes: int
    gqa_repeated_kv_bytes: int
    split_view_logical_bytes: int
    transpose_view_logical_bytes: int
    view_logical_bytes: int
    materialized_allocated_bytes: int
    materialized_allocation_policy: str
    sdpa_workspace_caveat: str


@dataclass(frozen=True)
class AttentionKVBufferEstimate:
    schema_version: str
    spec: AttentionKVBufferSpec
    dtype_name: str
    dtype_bytes: int
    key_schedule: AttentionKVKeySchedule
    per_key_k_bytes: int
    per_key_v_bytes: int
    per_key_total_bytes: int
    kv_cache_total_bytes: int
    runtime_cache_persistence: str
    visible_attention: VisibleAttentionMemoryEstimate


@dataclass(frozen=True)
class AttentionKVLiveTensorFamilyObservation:
    family: str
    observed_count: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    requires_grad_values: tuple[bool, ...]
    mechanism: str = "observer_returns_original_tensor"

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "observed_count": self.observed_count,
            "shapes": [list(shape) for shape in self.shapes],
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "requires_grad_values": list(self.requires_grad_values),
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True)
class AttentionKVCaveatSummary:
    prefix_lm_mask_caveat: str
    gqa_repeat_caveat: str
    sdpa_workspace_caveat: str
    runtime_cache_persistence: str
    runtime_cache_training_bypass: bool
    runtime_cache_state_dict_claim: bool
    materialized_allocation_policy: str
    runtime_cache_scope: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prefix_lm_mask_caveat": self.prefix_lm_mask_caveat,
            "gqa_repeat_caveat": self.gqa_repeat_caveat,
            "sdpa_workspace_caveat": self.sdpa_workspace_caveat,
            "runtime_cache_persistence": self.runtime_cache_persistence,
            "runtime_cache_training_bypass": self.runtime_cache_training_bypass,
            "runtime_cache_state_dict_claim": self.runtime_cache_state_dict_claim,
            "materialized_allocation_policy": self.materialized_allocation_policy,
            "runtime_cache_scope": self.runtime_cache_scope,
        }


@dataclass(frozen=True)
class AttentionKVFailClosedReceipt:
    schema_version: str
    target_name: str
    allowed_observed_families: tuple[str, ...]
    required_observed_families: tuple[str, ...]
    attention_kv_attention_buffers_sub2_claim: bool
    real_sub2_representation_present: bool
    lossless_eviction_or_offload_proof_present: bool
    compression_proof_present: bool
    fidelity_acquisition_revalidation_present: bool
    no_hidden_bf16_authority_proven: bool
    gpu_memory_throughput_receipt_present: bool
    generic_lossy_or_compression_wording: bool
    ready_to_flip: bool
    blocked_reason: str
    observed_families: tuple[AttentionKVLiveTensorFamilyObservation, ...]
    caveats: AttentionKVCaveatSummary
    smallest_missing_proof: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "allowed_observed_families": list(self.allowed_observed_families),
            "required_observed_families": list(self.required_observed_families),
            "attention_kv_attention_buffers_sub2_claim": (
                self.attention_kv_attention_buffers_sub2_claim
            ),
            "real_sub2_representation_present": self.real_sub2_representation_present,
            "lossless_eviction_or_offload_proof_present": (
                self.lossless_eviction_or_offload_proof_present
            ),
            "compression_proof_present": self.compression_proof_present,
            "fidelity_acquisition_revalidation_present": (
                self.fidelity_acquisition_revalidation_present
            ),
            "no_hidden_bf16_authority_proven": self.no_hidden_bf16_authority_proven,
            "gpu_memory_throughput_receipt_present": (
                self.gpu_memory_throughput_receipt_present
            ),
            "generic_lossy_or_compression_wording": (
                self.generic_lossy_or_compression_wording
            ),
            "ready_to_flip": self.ready_to_flip,
            "blocked_reason": self.blocked_reason,
            "observed_families": [
                observation.to_dict() for observation in self.observed_families
            ],
            "caveats": self.caveats.to_dict(),
            "smallest_missing_proof": self.smallest_missing_proof,
            "non_claims": list(self.non_claims),
        }


def attention_kv_key_schedule(
    *,
    H_cycles: int,
    L_cycles: int,
    layers_per_level: int,
) -> AttentionKVKeySchedule:
    """Return the HRM KV-cache key counts split by L/H level."""

    for name, value in (
        ("H_cycles", H_cycles),
        ("L_cycles", L_cycles),
        ("layers_per_level", layers_per_level),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    l_key_count = H_cycles * L_cycles * layers_per_level
    h_key_count = H_cycles * layers_per_level
    return AttentionKVKeySchedule(
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        layers_per_level=layers_per_level,
        l_key_count=l_key_count,
        h_key_count=h_key_count,
        total_key_count=l_key_count + h_key_count,
    )


def visible_attention_memory_estimate(
    spec: AttentionKVBufferSpec,
) -> VisibleAttentionMemoryEstimate:
    """Estimate visible materialized attention buffers and logical views.

    Split tensors and transposes are logical/view surfaces in the current code.
    They are tracked separately so receipts cannot double-count them as
    materialized allocations.
    """

    spec = spec.validate()
    dtype_bytes = spec.dtype_bytes
    gqkv_projection_bytes = int(
        spec.batch_size
        * spec.query_seq_len
        * spec.total_gqkv_heads
        * spec.head_dim
        * dtype_bytes
    )
    prefix_lm_mask_bytes = 0
    if spec.attn_type == "prefixlm":
        prefix_lm_mask_bytes = int(spec.batch_size * spec.query_seq_len * spec.query_seq_len)
    gqa_repeated_kv_bytes = 0
    if spec.num_kv_heads != spec.num_heads:
        gqa_repeated_kv_bytes = int(
            2
            * spec.batch_size
            * spec.num_heads
            * spec.attention_kv_seq_len
            * spec.head_dim
            * dtype_bytes
        )
    split_view_logical_bytes = gqkv_projection_bytes
    transpose_view_logical_bytes = int(
        spec.batch_size
        * spec.query_seq_len
        * (spec.num_heads + 2 * spec.num_kv_heads)
        * spec.head_dim
        * dtype_bytes
    )
    materialized = gqkv_projection_bytes + prefix_lm_mask_bytes + gqa_repeated_kv_bytes
    return VisibleAttentionMemoryEstimate(
        gqkv_projection_bytes=gqkv_projection_bytes,
        prefix_lm_mask_bytes=prefix_lm_mask_bytes,
        gqa_repeated_kv_bytes=gqa_repeated_kv_bytes,
        split_view_logical_bytes=split_view_logical_bytes,
        transpose_view_logical_bytes=transpose_view_logical_bytes,
        view_logical_bytes=split_view_logical_bytes + transpose_view_logical_bytes,
        materialized_allocated_bytes=materialized,
        materialized_allocation_policy=MATERIALIZED_ALLOCATION_ONLY,
        sdpa_workspace_caveat=SDPA_WORKSPACE_GPU_MEASURED_DEFERRED,
    )


def estimate_attention_kv_buffers(
    spec: AttentionKVBufferSpec,
) -> AttentionKVBufferEstimate:
    """Estimate runtime KV-cache bytes plus visible attention lower bounds."""

    spec = spec.validate()
    dtype = normalize_attention_kv_dtype(spec.dtype)
    schedule = attention_kv_key_schedule(
        H_cycles=spec.H_cycles,
        L_cycles=spec.L_cycles,
        layers_per_level=spec.layers_per_level,
    )
    per_key_k_bytes = int(
        spec.batch_size * spec.num_kv_heads * spec.max_seq_len * spec.head_dim * spec.dtype_bytes
    )
    per_key_v_bytes = per_key_k_bytes
    per_key_total_bytes = per_key_k_bytes + per_key_v_bytes
    return AttentionKVBufferEstimate(
        schema_version=ATTENTION_KV_BUFFER_SCHEMA_VERSION,
        spec=spec,
        dtype_name=str(dtype).removeprefix("torch."),
        dtype_bytes=spec.dtype_bytes,
        key_schedule=schedule,
        per_key_k_bytes=per_key_k_bytes,
        per_key_v_bytes=per_key_v_bytes,
        per_key_total_bytes=per_key_total_bytes,
        kv_cache_total_bytes=schedule.total_key_count * per_key_total_bytes,
        runtime_cache_persistence=RUNTIME_CACHE_NOT_STATE_DICT,
        visible_attention=visible_attention_memory_estimate(spec),
    )


def validate_attention_kv_mode(mode: str) -> str:
    """Validate named attention/KV modes for this contract slice."""

    if mode in {MODE_ATTENTION_KV_OFF, MODE_ATTENTION_KV_RUNTIME_CACHE_ACCOUNTING}:
        return mode
    if mode in {MODE_LOSSLESS_ATTENTION_KV_OFFLOAD, MODE_LOSSLESS_ATTENTION_KV_EVICTION}:
        raise NotImplementedError(
            "lossless attention/KV offload/eviction is "
            f"{TIER1_LOSSLESS_ATTENTION_KV_RELIEF_DEFERRED}; "
            "this slice is contract/accounting only"
        )
    if mode == MODE_LOSSY_ATTENTION_KV_COMPRESSION:
        raise NotImplementedError(
            "lossy or low-bit attention/KV compression is "
            f"{TIER2_LOSSY_ATTENTION_KV_COMPRESSION_DEFERRED} and needs "
            "acquisition re-validation before any claim"
        )
    raise ValueError(f"unknown attention/KV mode: {mode!r}")


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _shape_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of integer dimensions")
    shape = tuple(int(dim) for dim in value)
    if not shape:
        raise ValueError(f"{field_name} must be non-empty")
    return shape


def _summarize_attention_kv_live_tensor_families(
    seam_events: Sequence[Mapping[str, object]],
) -> tuple[AttentionKVLiveTensorFamilyObservation, ...]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        family: [] for family in ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES
    }
    for event in seam_events:
        family = event.get("family", event.get("name"))
        if family not in grouped:
            raise ValueError(
                "attention/KV receipt observed families must be exactly the "
                f"Step 3B allowlist {ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES!r}; "
                f"got {family!r}"
            )
        grouped[str(family)].append(event)

    missing = [family for family, events in grouped.items() if not events]
    if missing:
        raise ValueError(
            "attention/KV receipt missing required observed families: "
            + ", ".join(missing)
        )

    observations: list[AttentionKVLiveTensorFamilyObservation] = []
    for family in ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES:
        events = grouped[family]
        observations.append(
            AttentionKVLiveTensorFamilyObservation(
                family=family,
                observed_count=len(events),
                shapes=tuple(
                    sorted(
                        {
                            _shape_tuple(
                                event.get("shape", ()),
                                field_name=f"{family}.shape",
                            )
                            for event in events
                        }
                    )
                ),
                dtypes=tuple(
                    sorted(
                        {
                            _require_nonempty_string(
                                event.get("dtype", ""),
                                field_name=f"{family}.dtype",
                            )
                            for event in events
                        }
                    )
                ),
                devices=tuple(
                    sorted(
                        {
                            _require_nonempty_string(
                                event.get("device", ""),
                                field_name=f"{family}.device",
                            )
                            for event in events
                        }
                    )
                ),
                requires_grad_values=tuple(
                    sorted({bool(event.get("requires_grad", False)) for event in events})
                ),
            )
        )
    return tuple(observations)


def build_attention_kv_fail_closed_receipt(
    *,
    seam_events: Sequence[Mapping[str, object]],
    attention_kv_attention_buffers_sub2_claim: bool = False,
    real_sub2_representation_present: bool = False,
    lossless_eviction_or_offload_proof_present: bool = False,
    compression_proof_present: bool = False,
    fidelity_acquisition_revalidation_present: bool = False,
    no_hidden_bf16_authority_proven: bool = False,
    gpu_memory_throughput_receipt_present: bool = False,
    generic_lossy_or_compression_wording: bool = False,
    ready_to_flip: bool = False,
    smallest_missing_proof: str = (
        "real attention/KV sub2 representation or lossless eviction/offload or "
        "compression proof with fidelity/acquisition revalidation, plus "
        "no-hidden-BF16 authority proof and GPU memory/throughput receipt"
    ),
) -> AttentionKVFailClosedReceipt:
    """Build the Step 3B fail-closed attention/KV blocker receipt."""

    receipt = AttentionKVFailClosedReceipt(
        schema_version=ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
        target_name=ATTENTION_KV_FAIL_CLOSED_TARGET_NAME,
        allowed_observed_families=ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES,
        required_observed_families=ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES,
        attention_kv_attention_buffers_sub2_claim=bool(
            attention_kv_attention_buffers_sub2_claim
        ),
        real_sub2_representation_present=bool(real_sub2_representation_present),
        lossless_eviction_or_offload_proof_present=bool(
            lossless_eviction_or_offload_proof_present
        ),
        compression_proof_present=bool(compression_proof_present),
        fidelity_acquisition_revalidation_present=bool(
            fidelity_acquisition_revalidation_present
        ),
        no_hidden_bf16_authority_proven=bool(no_hidden_bf16_authority_proven),
        gpu_memory_throughput_receipt_present=bool(
            gpu_memory_throughput_receipt_present
        ),
        generic_lossy_or_compression_wording=bool(generic_lossy_or_compression_wording),
        ready_to_flip=bool(ready_to_flip),
        blocked_reason=ATTENTION_KV_BLOCKED_REASON,
        observed_families=_summarize_attention_kv_live_tensor_families(seam_events),
        caveats=AttentionKVCaveatSummary(
            prefix_lm_mask_caveat=ATTENTION_KV_PREFIX_LM_MASK_CAVEAT,
            gqa_repeat_caveat=ATTENTION_KV_GQA_REPEAT_CAVEAT,
            sdpa_workspace_caveat=SDPA_WORKSPACE_GPU_MEASURED_DEFERRED,
            runtime_cache_persistence=RUNTIME_CACHE_NOT_STATE_DICT,
            runtime_cache_training_bypass=True,
            runtime_cache_state_dict_claim=False,
            materialized_allocation_policy=MATERIALIZED_ALLOCATION_ONLY,
            runtime_cache_scope=ATTENTION_KV_RUNTIME_CACHE_SCOPE,
        ),
        smallest_missing_proof=_require_nonempty_string(
            smallest_missing_proof,
            field_name="smallest_missing_proof",
        ),
        non_claims=ATTENTION_KV_FAIL_CLOSED_NON_CLAIMS,
    )
    validate_attention_kv_fail_closed_receipt(receipt)
    return receipt


def validate_attention_kv_fail_closed_receipt(
    receipt: AttentionKVFailClosedReceipt,
) -> None:
    if receipt.schema_version != ATTENTION_KV_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION:
        raise ValueError("attention/KV fail-closed receipt schema mismatch")
    if receipt.target_name != ATTENTION_KV_FAIL_CLOSED_TARGET_NAME:
        raise ValueError("attention/KV fail-closed receipt target mismatch")
    if receipt.allowed_observed_families != ATTENTION_KV_ALLOWED_OBSERVED_FAMILIES:
        raise ValueError("attention/KV allowed observed families must be exact")
    if receipt.required_observed_families != ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES:
        raise ValueError("attention/KV required observed families must be exact")
    observed_names = tuple(observation.family for observation in receipt.observed_families)
    if observed_names != ATTENTION_KV_REQUIRED_OBSERVED_FAMILIES:
        raise ValueError("attention/KV observed families must match required q/k/v set")
    counts = tuple(observation.observed_count for observation in receipt.observed_families)
    if any(count <= 0 for count in counts):
        raise ValueError("attention/KV q/k/v families must all be observed")
    if len(set(counts)) != 1:
        raise ValueError("attention/KV q/k/v observed counts must have parity")
    for observation in receipt.observed_families:
        if observation.mechanism != "observer_returns_original_tensor":
            raise ValueError("Step 3B accepts only observer-returned original tensors")
        if not observation.shapes or not observation.dtypes or not observation.devices:
            raise ValueError(f"{observation.family} is missing tensor metadata")

    caveats = receipt.caveats
    if caveats.prefix_lm_mask_caveat != ATTENTION_KV_PREFIX_LM_MASK_CAVEAT:
        raise ValueError("attention/KV receipt must carry the PrefixLM mask caveat")
    if caveats.gqa_repeat_caveat != ATTENTION_KV_GQA_REPEAT_CAVEAT:
        raise ValueError("attention/KV receipt must carry the GQA repeat caveat")
    if caveats.sdpa_workspace_caveat != SDPA_WORKSPACE_GPU_MEASURED_DEFERRED:
        raise ValueError("attention/KV receipt must keep SDPA workspace GPU-deferred")
    if caveats.runtime_cache_persistence != RUNTIME_CACHE_NOT_STATE_DICT:
        raise ValueError("attention/KV receipt must keep runtime KVCache out of state_dict")
    if caveats.runtime_cache_training_bypass is not True:
        raise ValueError("attention/KV receipt must state training bypasses cached attention")
    if caveats.runtime_cache_state_dict_claim is not False:
        raise ValueError("attention/KV receipt must not claim KVCache state_dict authority")
    if caveats.materialized_allocation_policy != MATERIALIZED_ALLOCATION_ONLY:
        raise ValueError("attention/KV receipt must keep materialized-allocation caveat")
    if caveats.runtime_cache_scope != ATTENTION_KV_RUNTIME_CACHE_SCOPE:
        raise ValueError("attention/KV receipt must carry the runtime cache scope caveat")

    compression_gate = (
        receipt.compression_proof_present
        and receipt.fidelity_acquisition_revalidation_present
    )
    representation_gate = (
        receipt.real_sub2_representation_present
        or receipt.lossless_eviction_or_offload_proof_present
        or compression_gate
    )
    required_proofs = (
        representation_gate
        and receipt.no_hidden_bf16_authority_proven
        and receipt.gpu_memory_throughput_receipt_present
    )
    if receipt.generic_lossy_or_compression_wording and not (
        compression_gate
        and receipt.no_hidden_bf16_authority_proven
        and receipt.gpu_memory_throughput_receipt_present
    ):
        raise ValueError(
            "generic compression/lossy wording requires explicit fidelity/acquisition "
            "revalidation plus no-hidden-BF16 and GPU memory/throughput proof"
        )
    if receipt.attention_kv_attention_buffers_sub2_claim and not (
        required_proofs and receipt.ready_to_flip
    ):
        raise ValueError(
            "attention_kv_attention_buffers_sub2_claim requires real representation "
            "or lossless eviction/offload or compression+fidelity proof, plus "
            "no-hidden-BF16, GPU memory/throughput proof, and ready_to_flip=True"
        )
    if receipt.ready_to_flip and not (
        receipt.attention_kv_attention_buffers_sub2_claim and required_proofs
    ):
        raise ValueError("ready_to_flip cannot be true without all attention/KV proof gates")
    if receipt.blocked_reason != ATTENTION_KV_BLOCKED_REASON:
        raise ValueError("attention/KV blocked reason must be exact")
    if receipt.non_claims != ATTENTION_KV_FAIL_CLOSED_NON_CLAIMS:
        raise ValueError("attention/KV receipt non-claims must be exact")


def _require_numeric(receipt: Mapping[str, object], field: str) -> int | float:
    value = receipt[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value!r}")
    return value


def validate_attention_kv_measurement(receipt: Mapping[str, object]) -> None:
    """Validate future attention/KV receipts.

    A valid receipt must include canonical resource metrics and attention/KV
    schema fields together. Probe-specific timing aliases are additive only and
    never substitute for `wall_clock_per_step_seconds`.
    """

    missing = [
        field
        for field in REQUIRED_ATTENTION_KV_MEASUREMENT_FIELDS
        if field not in receipt
    ]
    if missing:
        raise ValueError(
            "attention/KV measurement missing required fields: " + ", ".join(missing)
        )
    if receipt["attention_kv_buffer_schema_version"] != ATTENTION_KV_BUFFER_SCHEMA_VERSION:
        raise ValueError(
            "attention_kv_buffer_schema_version must equal "
            f"{ATTENTION_KV_BUFFER_SCHEMA_VERSION!r}"
        )
    if receipt["materialized_allocation_policy"] != MATERIALIZED_ALLOCATION_ONLY:
        raise ValueError(
            "materialized_allocation_policy must equal "
            f"{MATERIALIZED_ALLOCATION_ONLY!r}"
        )
    if receipt["runtime_cache_persistence"] != RUNTIME_CACHE_NOT_STATE_DICT:
        raise ValueError(
            "runtime_cache_persistence must equal "
            f"{RUNTIME_CACHE_NOT_STATE_DICT!r}"
        )
    if receipt["sdpa_workspace_caveat"] != SDPA_WORKSPACE_GPU_MEASURED_DEFERRED:
        raise ValueError(
            "sdpa_workspace_caveat must equal "
            f"{SDPA_WORKSPACE_GPU_MEASURED_DEFERRED!r}"
        )

    numeric = {
        field: _require_numeric(receipt, field)
        for field in REQUIRED_ATTENTION_KV_MEASUREMENT_FIELDS
        if field
        not in {
            "attention_kv_buffer_schema_version",
            "materialized_allocation_policy",
            "runtime_cache_persistence",
            "sdpa_workspace_caveat",
        }
    }
    for positive_field in (
        "wall_clock_per_step_seconds",
        "max_safe_batch_size",
        "effective_exposure_per_step",
        "kv_cache_key_count",
        "per_key_k_bytes",
        "per_key_v_bytes",
        "kv_cache_total_bytes",
        "gqkv_projection_bytes",
        "visible_attention_allocated_bytes",
    ):
        if numeric[positive_field] <= 0:
            raise ValueError(f"{positive_field} must be > 0")
    if numeric["peak_reserved_bytes"] < numeric["peak_allocated_bytes"]:
        raise ValueError("peak_reserved_bytes must be >= peak_allocated_bytes")

    expected_kv_total = numeric["kv_cache_key_count"] * (
        numeric["per_key_k_bytes"] + numeric["per_key_v_bytes"]
    )
    if numeric["kv_cache_total_bytes"] != expected_kv_total:
        raise ValueError(
            "kv_cache_total_bytes must equal "
            "kv_cache_key_count * (per_key_k_bytes + per_key_v_bytes)"
        )
    expected_visible = (
        numeric["gqkv_projection_bytes"]
        + numeric["prefix_lm_mask_bytes"]
        + numeric["gqa_repeated_kv_bytes"]
    )
    if numeric["visible_attention_allocated_bytes"] != expected_visible:
        raise ValueError(
            "visible_attention_allocated_bytes must equal materialized "
            "gqkv_projection_bytes + prefix_lm_mask_bytes + gqa_repeated_kv_bytes"
        )
