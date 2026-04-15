"""Load the bit-exact Pi rotation matrix from llama.cpp's
turboquant_tables.h for GGUF compatibility.

Our `tq4_torch.py:build_pi()` generates a mathematically-equivalent
but BYTE-DIFFERENT Pi matrix (different RNG from the C reference).
Loading existing tq4 GGUFs requires the C-reference Pi.

This module parses `turboquant_tables.h` once, extracts the 65536
fp32 constants of TQ3_K256_PI (which is aliased as TQ4_K256_PI),
and caches them as a numpy file for fast subsequent loads.

The parse is slow (~200ms to read and tokenize 41K lines of C) but
runs once per environment; the cache path is configurable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch


DEFAULT_HEADER_PATH = Path(
    os.environ.get(
        "ZENITH_TQ4_HEADER",
        "/home/gabe/llama.cpp/ggml/src/turboquant_tables.h",
    )
)
DEFAULT_CACHE_PATH = Path(
    os.environ.get(
        "ZENITH_TQ4_PI_CACHE",
        "/tmp/tq4_k256_pi.npy",
    )
)


def parse_pi_from_header(header_path: Path = DEFAULT_HEADER_PATH) -> np.ndarray:
    """Parse TQ3_K256_PI (65536 floats) out of the C header file.

    Looks for the array definition `static const float TQ3_K256_PI[65536] = {`
    and parses float literals (including `e-XX` exponent forms and `f`
    suffix) until the closing `};`.
    """
    if not header_path.exists():
        raise FileNotFoundError(
            f"turboquant_tables.h not found at {header_path}. "
            f"Set ZENITH_TQ4_HEADER env var or pass header_path."
        )
    # Float regex: signed mantissa + optional exponent + optional f suffix
    float_re = re.compile(
        r"-?\d+\.\d+(?:[eE][+\-]?\d+)?f?|-?\d+f?"
    )
    in_pi = False
    values: list[float] = []
    with open(header_path) as fh:
        for line in fh:
            stripped = line.strip()
            if not in_pi:
                if "TQ3_K256_PI[65536]" in stripped and "{" in stripped:
                    in_pi = True
                continue
            if "};" in stripped:
                # final partial line
                for m in float_re.finditer(stripped):
                    v = m.group(0).rstrip("f")
                    values.append(float(v))
                break
            for m in float_re.finditer(stripped):
                v = m.group(0).rstrip("f")
                values.append(float(v))
    expected = 256 * 256
    if len(values) != expected:
        raise ValueError(
            f"parsed {len(values)} floats, expected {expected}"
        )
    arr = np.array(values, dtype=np.float32).reshape(256, 256)
    return arr


def load_c_reference_pi(
    header_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Return the C-reference Pi matrix as a (256, 256) fp32 torch tensor.

    On first call, parses turboquant_tables.h and caches to a .npy file.
    Subsequent calls read the cache (~1 ms). The cache can be disabled
    or given a custom path.
    """
    header = Path(header_path) if header_path else DEFAULT_HEADER_PATH
    cache = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if use_cache and cache.exists():
        try:
            arr = np.load(cache)
            if arr.shape == (256, 256) and arr.dtype == np.float32:
                return torch.from_numpy(arr)
        except Exception:
            pass
    arr = parse_pi_from_header(header)
    if use_cache:
        try:
            np.save(cache, arr)
        except OSError:
            pass  # cache dir not writable; OK
    return torch.from_numpy(arr)


def verify_pi_is_orthogonal(
    pi: torch.Tensor, atol: float = 1e-4,
) -> bool:
    """Check Pi @ Pi.T ≈ I."""
    identity = torch.eye(pi.size(0), dtype=pi.dtype)
    return torch.allclose(pi @ pi.T, identity, atol=atol)
