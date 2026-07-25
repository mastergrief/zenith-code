"""CPU tests for Phase B observation-only compact dump helper."""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack import screen_execution_loop as sel
from calm.hrm_text_158.native_full_stack.forgetting_laws import entropy_bits
from calm.hrm_text_158.native_full_stack.phaseb_acc_carrier_dump import (
    BULK_SUFFIXES,
    PhaseBAccCarrierDumpWriter,
    build_compact_snapshot,
    m2_empirical_bpw,
    select_eligible_bulk_acc,
)


def _synth_bulk_acc() -> dict[str, torch.Tensor]:
    # 4 tensors × 8 elems = 32 eligible; known hot-set of 3 nonzeros.
    t0 = torch.zeros(8, dtype=torch.int16)
    t0[0] = 5
    t0[1] = -5
    t0[2] = 10
    return {
        "layers.0.gqkv_proj.weight": t0,
        "layers.0.o_proj.weight": torch.zeros(8, dtype=torch.int16),
        "layers.0.gate_up_proj.weight": torch.zeros(8, dtype=torch.int16),
        "layers.0.down_proj.weight": torch.zeros(8, dtype=torch.int16),
        "layers.0.unrelated.bias": torch.ones(4, dtype=torch.int16),  # filtered out
    }


def test_select_eligible_bulk_filters_suffixes_and_fail_closed() -> None:
    bulk = select_eligible_bulk_acc(_synth_bulk_acc())
    assert set(bulk) == {
        "layers.0.gqkv_proj.weight",
        "layers.0.o_proj.weight",
        "layers.0.gate_up_proj.weight",
        "layers.0.down_proj.weight",
    }
    with pytest.raises(ValueError, match="zero tensors matched"):
        select_eligible_bulk_acc({"foo.bias": torch.zeros(2)})


def test_build_compact_snapshot_known_entropy_and_m2() -> None:
    acc = _synth_bulk_acc()
    snap = build_compact_snapshot(25, acc, expected_n_eligible=32)
    flat = torch.cat([t.reshape(-1) for t in select_eligible_bulk_acc(acc).values()])
    assert snap["step"] == 25
    assert snap["n_eligible"] == 32
    assert snap["n_nonzero"] == 3
    assert snap["H_bits_per_weight"] == pytest.approx(entropy_bits(flat))
    assert snap["M2_empirical_bpw"] == pytest.approx(
        m2_empirical_bpw(n_nonzero=3, n_eligible=32)
    )
    assert snap["acc_side_scale_bits_bpw"] == 0.0
    assert snap["value_abs_histogram"]["0"] == 29
    assert snap["value_abs_histogram"]["5"] == 2
    assert snap["value_abs_histogram"]["10"] == 1
    assert len(snap["top_h_multiset_sha256"]) == 64
    # Exact M2 formula check
    expected_m2 = 3.0 * (math.log2(32.0) + 8.0) / 32.0
    assert snap["M2_empirical_bpw"] == pytest.approx(expected_m2)


def test_expected_n_eligible_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_eligible="):
        build_compact_snapshot(25, _synth_bulk_acc(), expected_n_eligible=29360128)


def test_writer_o_excl_and_schema_files(tmp_path: Path) -> None:
    w = PhaseBAccCarrierDumpWriter(tmp_path, expected_n_eligible=32)
    w.record_snapshot(25, _synth_bulk_acc())
    w.record_snapshot(50, _synth_bulk_acc())
    out = w.finalize(
        {
            "geometry": "150/8/1024/cpu/arm3_sparse_hot_forgettable_cold",
            "parent_sha256": "0" * 64,
            "batch_rng_base": 1000,
            "arm": "arm3_sparse_hot_forgettable_cold",
        }
    )
    assert (tmp_path / "acc_carrier_snapshots.jsonl").exists()
    assert (tmp_path / "acc_carrier_dump_summary.json").exists()
    assert (tmp_path / "dump_run_receipt.json").exists()
    lines = (tmp_path / "acc_carrier_snapshots.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert out["summary"]["n_snapshots"] == 2
    with pytest.raises(FileExistsError):
        PhaseBAccCarrierDumpWriter(tmp_path, expected_n_eligible=32)


def test_finalize_missing_receipt_key_fail_closed(tmp_path: Path) -> None:
    w = PhaseBAccCarrierDumpWriter(tmp_path / "d2", expected_n_eligible=32)
    w.record_snapshot(25, _synth_bulk_acc())
    with pytest.raises(ValueError, match="run_receipt missing"):
        w.finalize({"geometry": "x", "parent_sha256": "y", "batch_rng_base": 1000})


def test_run_train_loop_signature_default_dump_off() -> None:
    sig = inspect.signature(sel.run_train_loop)
    assert sig.parameters["phaseb_acc_dump_dir"].default is None


def test_cli_help_lists_phaseb_flag() -> None:
    import argparse
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "scripts" / "hrm_text_158_forgetting_mechanism_screen.py"
    # Parse just the argparse block via running --help subprocess-free:
    # import module would execute side effects; inspect source instead.
    src = path.read_text(encoding="utf-8")
    assert "--phaseb-acc-dump-dir" in src
    assert "Default OFF" in src or "default OFF" in src.lower() or "Default OFF" in src


def test_flag_off_means_no_writer_import_path() -> None:
    # Absent/None dump dir must not create dump files; exercised by ensuring
    # PhaseBAccCarrierDumpWriter is only constructed when dir is truthy in source.
    src = Path(sel.__file__).read_text(encoding="utf-8")
    assert "phaseb_acc_dump_dir" in src
    assert "if phaseb_acc_dump_dir:" in src
    assert "if phaseb_writer is not None:" in src


def test_bulk_suffixes_match_phase_a_four_tuple() -> None:
    assert BULK_SUFFIXES == (
        "gqkv_proj.weight",
        "o_proj.weight",
        "gate_up_proj.weight",
        "down_proj.weight",
    )
    assert len(BULK_SUFFIXES) == 4
