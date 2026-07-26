"""STATIC marker schema/order parser tests (synthetic logs only)."""
from __future__ import annotations

import re

REQUIRED_MARKERS_IN_ORDER = [
    "[P1B_PHASE] model_build_start",
    "[P1B_PHASE] model_build_end",
    "[P1B_PHASE] forward_backward_start",
    "[P1B_PHASE] forward_backward_end",
    "[P1B_PHASE] vote_apply_start",
    "[P1B_PHASE] vote_apply_end",
    "[P1B_PHASE] checkpoint_roundtrip_start",
    "[P1B_PHASE] checkpoint_roundtrip_end",
    "[P1B_PHASE] receipt_mint_start",
    "[P1B_PHASE] receipt_mint_end",
    "[P1B_PHASE] TERMINAL_OK",
]

_MARKER_RE = re.compile(r"^\[P1B_PHASE\]\s+(\S+)\s*$")


def parse_p1b_markers(log_text: str) -> list[str]:
    out: list[str] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        m = _MARKER_RE.match(line)
        if m:
            out.append(f"[P1B_PHASE] {m.group(1)}")
    return out


def assert_required_marker_order(markers: list[str]) -> None:
    filtered = [m for m in markers if m in REQUIRED_MARKERS_IN_ORDER]
    assert filtered == REQUIRED_MARKERS_IN_ORDER, filtered


def test_required_markers_in_order_from_synthetic_log():
    synthetic = "\n".join(
        [
            "noise before",
            "[P1B_PHASE] model_build_start",
            "[hrm158] params: 1",
            "[P1B_PHASE] model_build_end",
            "[P1B_PHASE] forward_backward_start",
            "[P1B_PHASE] forward_backward_end",
            "[P1B_PHASE] vote_apply_start",
            "[P1B_PHASE] vote_apply_end",
            "[P1B_PHASE] checkpoint_roundtrip_start",
            "[P1B_PHASE] checkpoint_roundtrip_end",
            "[P1B_PHASE] receipt_mint_start",
            "[P1B_PHASE] receipt_mint_end",
            "[P1B_PHASE] TERMINAL_OK",
            "noise after",
        ]
    )
    markers = parse_p1b_markers(synthetic)
    assert_required_marker_order(markers)


def test_out_of_order_markers_detected():
    synthetic = "\n".join(
        [
            "[P1B_PHASE] model_build_start",
            "[P1B_PHASE] forward_backward_start",  # skipped model_build_end
            "[P1B_PHASE] model_build_end",
        ]
    )
    markers = parse_p1b_markers(synthetic)
    filtered = [m for m in markers if m in REQUIRED_MARKERS_IN_ORDER]
    assert filtered != REQUIRED_MARKERS_IN_ORDER[: len(filtered)] or filtered[1] != (
        "[P1B_PHASE] model_build_end"
    )


def test_marker_prefix_constant():
    assert all(m.startswith("[P1B_PHASE] ") for m in REQUIRED_MARKERS_IN_ORDER)
