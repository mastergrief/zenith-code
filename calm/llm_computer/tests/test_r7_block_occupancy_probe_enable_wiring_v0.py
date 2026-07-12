"""Probe occupancy-enable wiring: parser + default-off + real-write + reducer."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
    tensor_states_use_event_coded_live_carrier,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import GlobalRateCapSpec
from calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census import (
    ObserverContinuityTracker,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts" / "hrm_text_158_bounded_delta_acquisition_probe.py"
FLAG = "--r7-block-occupancy-B64"
RECEIPT_OCC_KEY = "r7_block_occupancy_B64_enabled"
# Nondeterministic / path-like keys excluded from shape comparison when present.
_RECEIPT_SHAPE_EXCLUDE = frozenset(
    {
        "receipt_path",
        "run_log_path",
        "cuda_memory_snapshots_jsonl_path",
        "r7_selective_drain_eligibility_census_sidecar_path",
        "r7_cap_defer_pressure_sidecar_path",
        "headroom_wiring_sidecar_path",
        "d_recompute_window_log_path",
        "event_coded_recompute_window_log_path",
        "d_live_carrier_snapshot_path",
        "d_recompute_selector_manifest_path",
        "d_recompute_calibration_warmup_observations_path",
        "phase_telemetry",
    }
)


def _probe_text() -> str:
    return PROBE.read_text(encoding="utf-8")


def _apply_probe_occupancy_receipt_echo(
    receipt: dict, *, r7_block_occupancy_B64_enabled: bool
) -> dict:
    """Replay the landed probe receipt echo (:9363-9364) against a receipt dict.

    Source authority: the gated block must exist verbatim in the probe; we then
    execute that same gate semantics. Default-off must leave the key ABSENT.
    """
    text = _probe_text()
    # Exactly one gated receipt assignment (not an unconditional write).
    gated = re.findall(
        r"if r7_block_occupancy_B64_enabled:\n"
        r"\s+receipt\[\"r7_block_occupancy_B64_enabled\"\] = True",
        text,
    )
    assert len(gated) == 1, gated
    assert text.count('receipt["r7_block_occupancy_B64_enabled"]') == 1
    # Execute the same gate the probe uses (emitted receipt-dict authority).
    if r7_block_occupancy_B64_enabled:
        receipt["r7_block_occupancy_B64_enabled"] = True
    return receipt


def _canonical_receipt_shape(receipt: dict) -> frozenset[str]:
    return frozenset(k for k in receipt.keys() if k not in _RECEIPT_SHAPE_EXCLUDE)


def _dense_fixture():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([0, 0], dtype=torch.int8),
        0.5,
        torch.zeros(2, dtype=torch.int16),
    )
    votes = torch.tensor([12, 12], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=2,
    )
    cap = GlobalRateCapSpec(cap=1, step=0)
    return state, votes, spec, cap


def test_a_parser_and_help_exact_token():
    parser = build_arg_parser()
    absent = parser.parse_args([])
    assert absent.r7_block_occupancy_B64 is False
    present = parser.parse_args([FLAG])
    assert present.r7_block_occupancy_B64 is True
    help_text = parser.format_help()
    assert FLAG in help_text
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0
    assert FLAG in proc.stdout


def test_b_default_off_sidecar_field_absent(tmp_path: Path):
    state, votes, spec, cap = _dense_fixture()
    assert tensor_states_use_event_coded_live_carrier({"toy.proj": state}) is False
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "off.jsonl"
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        global_cap_spec=cap,
        deferred_backlog=pre,
        local_selection_ordering_step=0,
        r7_selective_drain_eligibility_census_enabled=True,
        r7_selective_drain_eligibility_census_tracker=tr,
        r7_selective_drain_eligibility_census_sidecar_path=sidecar,
        r7_block_occupancy_B64_enabled=False,
    )
    assert sidecar.exists()
    lines = [ln for ln in sidecar.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "block_occupancy_B64" not in row


def test_c_enabled_real_write_field_present(tmp_path: Path):
    state, votes, spec, cap = _dense_fixture()
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "on.jsonl"
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        global_cap_spec=cap,
        deferred_backlog=pre,
        local_selection_ordering_step=0,
        r7_selective_drain_eligibility_census_enabled=True,
        r7_selective_drain_eligibility_census_tracker=tr,
        r7_selective_drain_eligibility_census_sidecar_path=sidecar,
        r7_block_occupancy_B64_enabled=True,
    )
    assert sidecar.exists()
    lines = [ln for ln in sidecar.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert "block_occupancy_B64" in row
    occ = row["block_occupancy_B64"]
    assert isinstance(occ, dict)
    assert occ.get("schema_version")
    assert occ.get("B") == 64
    assert occ.get("event_coded_live") is False


def test_d_reducer_consumes_enabled_smoke_sidecar(tmp_path: Path):
    state, votes, spec, cap = _dense_fixture()
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "smoke.jsonl"
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes},
        {"toy.proj": spec},
        global_cap_spec=cap,
        deferred_backlog=pre,
        local_selection_ordering_step=0,
        r7_selective_drain_eligibility_census_enabled=True,
        r7_selective_drain_eligibility_census_tracker=tr,
        r7_selective_drain_eligibility_census_sidecar_path=sidecar,
        r7_block_occupancy_B64_enabled=True,
    )
    row = json.loads(sidecar.read_text().strip().splitlines()[0])
    assert "block_occupancy_B64" in row
    companion = tmp_path / "companion.json"
    companion.write_text(json.dumps({"toy.proj": 2}))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "calm.hrm_text_158.native_full_stack.r7_block_occupancy_byte_reducer_cli",
            "reduce",
            str(sidecar),
            str(companion),
            "--N",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    body = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert body, proc.stderr
    payload = json.loads(body)
    # Must not be the absent-instrumentation MISSING path (field was present).
    assert "block_occupancy_B64" in row
    if payload.get("overall") == "MISSING_OBSERVABLES":
        reasons = str(payload.get("integrity_failures") or payload.get("errors") or "")
        assert "absent" not in reasons.lower()
        assert "missing_block_occupancy" not in reasons.lower()
    # Prefer healthy reduce; N=1 may be INSUFFICIENT/other honest class — not absent.
    assert proc.returncode in (0, 3)


def test_e_probe_source_threads_kwarg_to_vote_step():
    text = _probe_text()
    assert 'ap.add_argument(\n        "--r7-block-occupancy-B64"' in text or (
        f'"{FLAG}"' in text or f"'{FLAG}'" in text
    )
    assert "r7_block_occupancy_B64_enabled=bool(args.r7_block_occupancy_B64)" in text
    assert "r7_block_occupancy_B64_enabled=bool(\n                            r7_block_occupancy_B64_enabled\n                        )" in text or (
        "r7_block_occupancy_B64_enabled=bool(r7_block_occupancy_B64_enabled)" in text
    )
    # Launcher must not embed occupancy math / serialization.
    forbidden = (
        "build_block_occupancy_B64",
        "fully_eoe_set_sha256",
        "projected_acc_bpw",
    )
    for tok in forbidden:
        assert tok not in text


def test_f_default_off_receipt_key_absent_and_shape_parity():
    """Positive proof: default-off receipt NEVER emits r7_block_occupancy_B64_enabled.

    Replays the landed probe gated echo (:9363-9364). Default-off must leave the
    receipt key-set unchanged (no new keys from the flag machinery). Enabled adds
    exactly that one key. Exclusions for path-like nondeterministic keys are
    enumerated in _RECEIPT_SHAPE_EXCLUDE.
    """
    baseline = {
        "steps_completed": 0,
        "steps_requested": 1,
        "stop_reason": "ok",
        "r7_selective_drain_eligibility_census_enabled": True,
        "r7_selective_drain_eligibility_census_sidecar_path": "/tmp/census.jsonl",
        "receipt_path": "/tmp/receipt.json",
    }
    baseline_shape = _canonical_receipt_shape(baseline)

    off_receipt = dict(baseline)
    _apply_probe_occupancy_receipt_echo(
        off_receipt, r7_block_occupancy_B64_enabled=False
    )
    assert RECEIPT_OCC_KEY not in off_receipt
    assert _canonical_receipt_shape(off_receipt) == baseline_shape
    assert set(off_receipt.keys()) == set(baseline.keys())

    on_receipt = dict(baseline)
    _apply_probe_occupancy_receipt_echo(
        on_receipt, r7_block_occupancy_B64_enabled=True
    )
    assert on_receipt.get(RECEIPT_OCC_KEY) is True
    assert _canonical_receipt_shape(on_receipt) == baseline_shape | {RECEIPT_OCC_KEY}
    assert set(on_receipt.keys()) - set(baseline.keys()) == {RECEIPT_OCC_KEY}


def test_g_default_off_sidecar_chunk_keyset_unchanged_by_occupancy(tmp_path: Path):
    """Real-write: off chunk keys ⊇ baseline; on adds exactly block_occupancy_B64."""
    state, votes, spec, cap = _dense_fixture()
    pre = {"toy.proj": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}

    def _one(*, enabled: bool, name: str) -> set[str]:
        tr = ObserverContinuityTracker()
        tr.reset()
        sidecar = tmp_path / name
        apply_bounded_delta_vote_step(
            {"toy.proj": state},
            {"toy.proj": votes},
            {"toy.proj": spec},
            global_cap_spec=cap,
            deferred_backlog=pre,
            local_selection_ordering_step=0,
            r7_selective_drain_eligibility_census_enabled=True,
            r7_selective_drain_eligibility_census_tracker=tr,
            r7_selective_drain_eligibility_census_sidecar_path=sidecar,
            r7_block_occupancy_B64_enabled=enabled,
        )
        row = json.loads(sidecar.read_text().strip().splitlines()[0])
        return set(row.keys())

    off_keys = _one(enabled=False, name="off.jsonl")
    on_keys = _one(enabled=True, name="on.jsonl")
    assert "block_occupancy_B64" not in off_keys
    assert "block_occupancy_B64" in on_keys
    # Default-off shape is the enabled shape minus the occupancy field only.
    assert on_keys - off_keys == {"block_occupancy_B64"}
    assert off_keys <= on_keys
