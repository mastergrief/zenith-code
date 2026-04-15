"""Tests for loading C-reference Pi matrix from turboquant_tables.h."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.llm_computer.tq4_pi_loader import (
    DEFAULT_HEADER_PATH, load_c_reference_pi, parse_pi_from_header,
    verify_pi_is_orthogonal,
)


HEADER_AVAILABLE = DEFAULT_HEADER_PATH.exists()


@pytest.mark.skipif(not HEADER_AVAILABLE, reason="header file not present")
def test_parse_pi_produces_256x256():
    arr = parse_pi_from_header()
    assert arr.shape == (256, 256)
    assert arr.dtype.name == "float32"


@pytest.mark.skipif(not HEADER_AVAILABLE, reason="header file not present")
def test_parse_pi_is_orthogonal():
    arr = parse_pi_from_header()
    pi = torch.from_numpy(arr)
    assert verify_pi_is_orthogonal(pi), (
        "parsed Pi is not orthogonal within tolerance"
    )


@pytest.mark.skipif(not HEADER_AVAILABLE, reason="header file not present")
def test_load_c_reference_pi_returns_tensor():
    pi = load_c_reference_pi(use_cache=False)
    assert isinstance(pi, torch.Tensor)
    assert pi.shape == (256, 256)


@pytest.mark.skipif(not HEADER_AVAILABLE, reason="header file not present")
def test_cache_round_trip(tmp_path):
    """First load writes cache; second load reads cache (fast path)."""
    cache = tmp_path / "pi.npy"
    pi1 = load_c_reference_pi(cache_path=cache, use_cache=True)
    assert cache.exists()
    pi2 = load_c_reference_pi(cache_path=cache, use_cache=True)
    assert torch.equal(pi1, pi2)


@pytest.mark.skipif(not HEADER_AVAILABLE, reason="header file not present")
def test_first_row_matches_known_values():
    """Sanity: first few floats must match what we saw in the header."""
    arr = parse_pi_from_header()
    # From the header file, first three values of row 0:
    # 1.203470230e-01f, 9.858401120e-02f, 5.262503400e-02f
    assert abs(arr[0, 0] - 1.203470230e-01) < 1e-7
    assert abs(arr[0, 1] - 9.858401120e-02) < 1e-7
    assert abs(arr[0, 2] - 5.262503400e-02) < 1e-7


def test_missing_header_raises():
    """If header path doesn't exist, should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_pi_from_header(Path("/does/not/exist/turboquant_tables.h"))


def test_verify_orthogonal_helper():
    """Helper correctly identifies orthogonal and non-orthogonal matrices."""
    ortho = torch.eye(4)
    non_ortho = torch.ones(4, 4)
    assert verify_pi_is_orthogonal(ortho)
    assert not verify_pi_is_orthogonal(non_ortho)


if __name__ == "__main__":
    if HEADER_AVAILABLE:
        test_parse_pi_produces_256x256()
        print("[ok] parse Pi produces 256x256")
        test_parse_pi_is_orthogonal()
        print("[ok] parsed Pi is orthogonal")
        test_load_c_reference_pi_returns_tensor()
        print("[ok] load_c_reference_pi returns tensor")
        test_first_row_matches_known_values()
        print("[ok] first row matches known values")
    else:
        print("[SKIP] header not available on this machine")
    test_missing_header_raises()
    print("[ok] missing header raises")
    test_verify_orthogonal_helper()
    print("[ok] verify_pi_is_orthogonal helper")
