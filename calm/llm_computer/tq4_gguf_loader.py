"""Load our tq4 GGUF files directly into Tq4Tensor — no re-quantization.

The GGUF format stores tq4 blocks in the exact layout our `Tq4Tensor`
uses: 132 bytes per 256-element block (128 `qs` + 2 fp16 `d` + 2 pad).
We can extract raw bytes from the file and wrap them in our substrate
classes — no dequant, no re-quant, lossless reload of our own
quantization work.

The upstream `gguf` Python library doesn't know about our custom
TurboQuant types (TQ3_K256=42, TQ3_K512=43, TQ4_K256=44). This module
monkey-patches the enum + quant-size table at import time so
`GGUFReader` can read our files without modification.

Scope:
- `patch_gguf_for_turboquant()` — extend gguf library enum at import
- `read_turboquant_gguf(path)` — open a GGUF, return a reader
- `extract_tq4_tensor(reader, name)` → Tq4Tensor
- `load_gemma4_e4b_stream(gguf_path, ...)` — skeleton loader that
  walks the Gemma 4 E4B layer structure (GQA + SWA alternation).
  Returns a dict of (layer_name -> loaded tensor) — actual wiring
  into GemmaStream requires SWA-aware config extensions (deferred).
"""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch


_PATCHED = False
_BLOCK_BYTES_TQ4 = 132
_BLOCK_ELEMENTS_TQ4 = 256


def patch_gguf_for_turboquant() -> None:
    """Extend the gguf library enum + quant-size table with our custom
    TurboQuant types so GGUFReader can read our GGUF files.

    Idempotent — safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        import gguf
        import gguf.constants as gc
        import gguf.gguf_reader as gr
    except ImportError as e:
        raise ImportError(
            "gguf library required for tq4 GGUF loading. "
            "Install via: pip install gguf"
        ) from e
    orig_members = {m.name: m.value for m in gguf.GGMLQuantizationType}
    for name, val in [("TQ3_K256", 42), ("TQ3_K512", 43), ("TQ4_K256", 44)]:
        if name not in orig_members:
            orig_members[name] = val
    patched = IntEnum("GGMLQuantizationType", orig_members)
    gguf.GGMLQuantizationType = patched
    gc.GGMLQuantizationType = patched
    gr.GGMLQuantizationType = patched
    # Block sizes for our custom types
    if hasattr(gc, "GGML_QUANT_SIZES"):
        gc.GGML_QUANT_SIZES.setdefault(patched.TQ4_K256, (256, 132))
        gc.GGML_QUANT_SIZES.setdefault(patched.TQ3_K256, (256, 98))
        gc.GGML_QUANT_SIZES.setdefault(patched.TQ3_K512, (512, 194))
    _PATCHED = True


def read_turboquant_gguf(path: str | Path):
    """Open a GGUF containing TurboQuant-quantized tensors. Returns
    a patched `GGUFReader`."""
    patch_gguf_for_turboquant()
    import gguf
    return gguf.GGUFReader(str(path))


def _get_ggml_type_id(tensor) -> int:
    """Extract the numeric type id from a gguf tensor descriptor."""
    t = tensor.tensor_type
    # Both IntEnum value and raw int work here
    return int(t.value) if hasattr(t, "value") else int(t)


def extract_tq4_tensor(reader, tensor_name: str):
    """Read raw tq4 bytes for `tensor_name`, split into (qs, d) and
    wrap as a Tq4Tensor.

    Returns:
        Tq4Tensor with .qs (n_blocks, 128) uint8 and .d (n_blocks,)
        float32. .shape is the logical tensor shape from GGUF.

    Raises:
        KeyError: if tensor not found.
        ValueError: if tensor is not tq4_k256 type or byte count mismatches.
    """
    # Import here to avoid circular dependency
    from calm.llm_computer.tq4_torch import HEAD_DIM, Tq4Tensor

    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise KeyError(f"tensor {tensor_name!r} not in GGUF")

    type_id = _get_ggml_type_id(tensor)
    if type_id != 44:  # TQ4_K256
        raise ValueError(
            f"{tensor_name!r} is type {tensor.tensor_type.name} "
            f"(id {type_id}), expected TQ4_K256 (id 44)"
        )

    # Tensor data is a uint8 numpy array already
    data = tensor.data  # (n_blocks * 132,) uint8
    logical_shape = tuple(int(d) for d in tensor.shape)
    total_elements = int(np.prod(logical_shape))
    expected_n_blocks = total_elements // _BLOCK_ELEMENTS_TQ4
    expected_bytes = expected_n_blocks * _BLOCK_BYTES_TQ4
    if data.nbytes != expected_bytes:
        raise ValueError(
            f"tensor {tensor_name}: got {data.nbytes} bytes, "
            f"expected {expected_bytes} for {expected_n_blocks} blocks"
        )

    # Reshape to (n_blocks, 132) and split
    blocks = data.reshape(expected_n_blocks, _BLOCK_BYTES_TQ4)
    qs_np = blocks[:, :128].copy()                          # (n_blocks, 128)
    d_bytes = blocks[:, 128:130].tobytes()                  # fp16 scales
    d_fp16 = np.frombuffer(d_bytes, dtype=np.float16)       # (n_blocks,)
    d_fp32 = d_fp16.astype(np.float32)

    qs = torch.from_numpy(qs_np)
    d = torch.from_numpy(d_fp32.copy())
    return Tq4Tensor(qs=qs, d=d, shape=logical_shape)


def ggml_gemma_tensor_map(n_layers: int) -> dict[str, dict[str, str]]:
    """Map logical layer names → ggml tensor names for Gemma 4 E4B.

    GGML naming uses `blk.N.attn_q.weight` style (vs HF's
    `model.layers.N.self_attn.q_proj.weight`).
    """
    result = {
        "_meta": {
            "token_embd":       "token_embd.weight",
            "output_norm":      "output_norm.weight",
            "rope_freqs":       "rope_freqs.weight",
            # Per-layer token embeddings (Gemma 4 specific)
            "per_layer_embd":   "per_layer_token_embd.weight",
            "per_layer_proj":   "per_layer_model_proj.weight",
            "per_layer_norm":   "per_layer_proj_norm.weight",
        }
    }
    for i in range(n_layers):
        result[f"layer_{i}"] = {
            "q":              f"blk.{i}.attn_q.weight",
            "k":              f"blk.{i}.attn_k.weight",
            "v":              f"blk.{i}.attn_v.weight",
            "o":              f"blk.{i}.attn_output.weight",
            "gate":           f"blk.{i}.ffn_gate.weight",
            "up":             f"blk.{i}.ffn_up.weight",
            "down":           f"blk.{i}.ffn_down.weight",
            "attn_norm":      f"blk.{i}.attn_norm.weight",
            "ffn_norm":       f"blk.{i}.ffn_norm.weight",
            "attn_q_norm":    f"blk.{i}.attn_q_norm.weight",
            "attn_k_norm":    f"blk.{i}.attn_k_norm.weight",
            "post_attn_norm": f"blk.{i}.post_attention_norm.weight",
            "inp_gate":       f"blk.{i}.inp_gate.weight",
            "layer_out_scale": f"blk.{i}.layer_output_scale.weight",
        }
    return result


def extract_fp_tensor(reader, tensor_name: str) -> torch.Tensor:
    """Read an F16/F32/Q6_K etc tensor from GGUF as a torch tensor.

    For uncommon types (Q6_K embeddings), we rely on gguf library's
    auto-dequant via `tensor.data` which returns float32 numpy.
    """
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise KeyError(f"tensor {tensor_name!r} not in GGUF")
    arr = tensor.data
    # For F16/F32 the data is already typed correctly; for quantized
    # the gguf library returns raw bytes and dequant needs manual work.
    # Since our main quantized path is tq4 (handled above), and norms
    # are always F32, we assume F32/F16 here.
    if arr.dtype == np.uint8 and _get_ggml_type_id(tensor) in (14, 15):
        # Q6_K etc - we can't easily dequant here without llama.cpp
        raise NotImplementedError(
            f"dequant of {tensor.tensor_type.name} not implemented; "
            f"re-quantize embeddings to F16 or implement Q6_K dequant"
        )
    t_tensor = torch.from_numpy(np.ascontiguousarray(arr))
    # GGUF shape is column-major for 2D; reshape to match
    shape = tuple(int(d) for d in tensor.shape)
    return t_tensor.reshape(shape)


def summarize_gguf(reader) -> dict:
    """Return a dict summary of tensor counts by type and key dims.

    Useful for debugging what's in a GGUF before committing to load it.
    """
    type_counts: dict[str, int] = {}
    for t in reader.tensors:
        name = t.tensor_type.name
        type_counts[name] = type_counts.get(name, 0) + 1
    meta = {}
    for f in reader.fields.values():
        if any(k in f.name for k in [
            "block_count", "embedding_length", "head_count",
            "feed_forward", "rope.freq", "context_length",
            "key_length", "value_length",
        ]):
            try:
                val = f.parts[-1].tolist()
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                meta[f.name] = val
            except Exception:
                pass
    return {
        "n_tensors": len(reader.tensors),
        "type_counts": type_counts,
        "metadata": meta,
    }
