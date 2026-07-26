"""receipt_mint_end emits only after successful O_EXCL mint."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_receipt_mint_end_ordering_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Failing O_EXCL leaves phase open; success emits end only after write."""
    from scripts.p1b_o_excl_copy import write_bytes_o_excl

    markers: list[str] = []

    def fake_emit(marker: str) -> None:
        markers.append(marker)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.trainer_sub2_authority.emit_p1b_phase",
        fake_emit,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import emit_p1b_phase

    # Failure path: pre-existing target → no end.
    markers.clear()
    out = tmp_path / "r1.json"
    out.write_bytes(b"x")
    emit_p1b_phase("receipt_mint_start")
    try:
        write_bytes_o_excl(out, b"{}")
        emit_p1b_phase("receipt_mint_end")
    except FileExistsError:
        pass
    assert markers == ["receipt_mint_start"]

    # Success path: end after write.
    markers.clear()
    out2 = tmp_path / "r2.json"
    emit_p1b_phase("receipt_mint_start")
    write_bytes_o_excl(out2, b"{}")
    emit_p1b_phase("receipt_mint_end")
    assert markers == ["receipt_mint_start", "receipt_mint_end"]
    assert out2.is_file()
