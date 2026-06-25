"""CPU tests for V4-LIVE Phase-A parity witness (anti-vacuous)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    observed_surfaces_dict,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_PERSISTENT_AUTHORITY_SCOPE_KEY,
    C8_PERSISTENT_AUTHORITY_SCOPE_VALUE,
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.v4_live_phase_a_parity_witness import (
    run_phase_a_parity_witness,
)
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VOTES_EMIT_SECTION6_CONTRACT_FIELDS,
)


def _write_phase_bundle(
    tmp_path: Path,
    *,
    gpu_surfaces_by_step: dict[int, dict[str, object]],
    votes_by_step: dict[int, dict[str, dict[int, int]]],
    carrier_sha_by_step: dict[int, str] | None = None,
    ledger_pass: bool = True,
    include_r4v_ledger: bool = True,
    r4v_ledger_override: dict[str, object] | None = None,
    logical_numel: int = 16,
    mismatch_step: int | None = None,
    include_sparse_votes: bool = True,
    include_c8_stats: bool = True,
) -> Path:
    phase_root = tmp_path / "phase_a"
    emit_root = phase_root / "votes_emit" / "v1" / "per_step"
    emit_root.mkdir(parents=True, exist_ok=True)
    step_reports: dict[str, object] = {}
    for step_index in sorted(votes_by_step):
        votes = votes_by_step[step_index]
        emit_payload = {
            "schema_version": "hrm_text_158_votes_emit/v0",
            "optimizer_step_index": int(step_index),
            "applied_flat_indices_hash": "hash",
            "cap_order_summary": {"ordering_mode": "current_margin"},
            "pre_update_state_hash": "pre",
        }
        if include_sparse_votes:
            emit_payload["sparse_vote_inputs_by_state_key"] = {
                state_key: {str(k): int(v) for k, v in lane_votes.items()}
                for state_key, lane_votes in votes.items()
            }
        for field in VOTES_EMIT_SECTION6_CONTRACT_FIELDS:
            emit_payload.setdefault(field, f"{field}-value")
        (emit_root / f"{int(step_index):05d}.json").write_text(
            json.dumps(emit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        carrier = EventCodedAccLiveState(logical_numel=int(logical_numel), demotion_band=1)
        for prior_step in range(int(step_index)):
            carrier.apply_step(prior_step, votes=votes_by_step[prior_step]["toy.proj"])
        carrier.apply_step(int(step_index), votes=votes["toy.proj"])
        cpu_surfaces = observed_surfaces_dict(carrier.step_records[-1])
        gpu_surfaces = dict(gpu_surfaces_by_step[step_index]["toy.proj"])  # type: ignore[index]
        if mismatch_step is not None and int(step_index) == int(mismatch_step):
            gpu_surfaces = {
                **gpu_surfaces,
                "applied_flat_indices": [999],
            }
        cpu_sha = carrier_content_sha256(carrier)
        gpu_sha = (
            carrier_sha_by_step.get(int(step_index), cpu_sha)
            if carrier_sha_by_step is not None
            else cpu_sha
        )
        stats_payload: dict[str, object] = {
            "logical_numel": int(logical_numel),
            "dense_accumulator_materialized_numel": 0,
            "v4_live_observed_surfaces": gpu_surfaces,
            "event_coded_live_carrier_content_sha256_after": gpu_sha,
        }
        if include_c8_stats:
            stats_payload[C8_PERSISTENT_AUTHORITY_SCOPE_KEY] = C8_PERSISTENT_AUTHORITY_SCOPE_VALUE
            stats_payload[C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY] = int(logical_numel)
        step_reports[str(step_index)] = {
            "step_result": {
                "tensor_stats": {
                    "toy.proj": stats_payload,
                }
            }
        }
    receipt: dict[str, object] = {"step_reports": step_reports}
    if include_r4v_ledger:
        receipt["r4v_persistent_ledger"] = (
            dict(r4v_ledger_override)
            if r4v_ledger_override is not None
            else {"ledger_pass": bool(ledger_pass)}
        )
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return phase_root


def test_phase_a_parity_witness_passes_on_matching_bundle(tmp_path: Path) -> None:
    votes = {
        0: {"toy.proj": {0: 8}},
        1: {"toy.proj": {1: 8}},
    }
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    surfaces_by_step: dict[int, dict[str, object]] = {}
    for step_index, lane_votes in votes.items():
        carrier.apply_step(int(step_index), votes=lane_votes["toy.proj"])
        surfaces_by_step[int(step_index)] = {
            "toy.proj": observed_surfaces_dict(carrier.step_records[-1]),
        }
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step=surfaces_by_step,
        votes_by_step=votes,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is True
    assert verdict["decisive_surface_diff_count"] == 0
    assert verdict["per_step_mismatches"] == []
    assert verdict["states_compared_count"] >= 1


def test_phase_a_parity_witness_fail_closed_missing_sparse_vote_inputs(
    tmp_path: Path,
) -> None:
    votes = {0: {"toy.proj": {0: 8}}}
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes=votes[0]["toy.proj"])
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step=votes,
        include_sparse_votes=False,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert verdict["states_compared_count"] == 0
    assert verdict["sparse_vote_inputs_missing_steps"] == [0]


def test_phase_a_parity_witness_fail_closed_zero_compared_states(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "phase_a"
    emit_root = phase_root / "votes_emit" / "v1" / "per_step"
    emit_root.mkdir(parents=True, exist_ok=True)
    emit_payload = {
        "schema_version": "hrm_text_158_votes_emit/v0",
        "optimizer_step_index": 0,
        "applied_flat_indices_hash": "hash",
        "cap_order_summary": {"ordering_mode": "current_margin"},
        "pre_update_state_hash": "pre",
        "sparse_vote_inputs_by_state_key": {"toy.proj": {}},
    }
    (emit_root / "00000.json").write_text(
        json.dumps(emit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "step_reports": {"0": {"step_result": {"tensor_stats": {}}}},
        "r4v_persistent_ledger": {"ledger_pass": True},
    }
    (phase_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert verdict["states_compared_count"] == 0
    assert verdict["zero_vote_inputs_steps"] == [0]


def test_phase_a_parity_witness_fail_closed_missing_c8_scope(tmp_path: Path) -> None:
    votes = {0: {"toy.proj": {0: 8}}}
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes=votes[0]["toy.proj"])
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step=votes,
        include_c8_stats=False,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert any(
        failure.get("reason") == "missing_or_invalid_c8_persistent_authority_scope"
        for failure in verdict["c8_persistent_failures"]
    )


def test_phase_a_parity_witness_fail_closed_missing_transient_numel(
    tmp_path: Path,
) -> None:
    votes = {0: {"toy.proj": {0: 8}}}
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes=votes[0]["toy.proj"])
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step=votes,
    )
    receipt = json.loads((phase_root / "receipt.json").read_text(encoding="utf-8"))
    stats = receipt["step_reports"]["0"]["step_result"]["tensor_stats"]["toy.proj"]
    stats.pop(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY, None)
    (phase_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert any(
        failure.get("reason") == "missing_or_non_numeric_transient_dense_compute_numel"
        for failure in verdict["c8_persistent_failures"]
    )


def test_phase_a_parity_witness_fails_on_mismatched_bundle(tmp_path: Path) -> None:
    votes = {
        0: {"toy.proj": {0: 8}},
        1: {"toy.proj": {1: 8}},
    }
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    surfaces_by_step: dict[int, dict[str, object]] = {}
    for step_index, lane_votes in votes.items():
        carrier.apply_step(int(step_index), votes=lane_votes["toy.proj"])
        surfaces_by_step[int(step_index)] = {
            "toy.proj": observed_surfaces_dict(carrier.step_records[-1]),
        }
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step=surfaces_by_step,
        votes_by_step=votes,
        mismatch_step=1,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert verdict["decisive_surface_diff_count"] >= 1
    assert verdict["per_step_mismatches"]


def test_phase_a_parity_witness_cli_end_to_end(tmp_path: Path) -> None:
    import subprocess
    import sys

    votes = {0: {"toy.proj": {0: 9}}}
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes=votes[0]["toy.proj"])
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step=votes,
    )
    out_path = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/hrm_text_158_v4_live_phase_a_parity_witness.py",
            "--phase-root",
            str(phase_root),
            "--json-out",
            str(out_path),
            "--demotion-band",
            "1",
        ],
        cwd="/mnt/c/Users/gabes/projects/claw-code-hrm-text-158",
        env={**dict(__import__("os").environ), "PYTHONPATH": "."},
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["phase_a_parity_pass"] is True


def test_phase_a_parity_witness_fail_closed_without_r4v_ledger(tmp_path: Path) -> None:
    votes = {0: {"toy.proj": {0: 8}}}
    carrier = EventCodedAccLiveState(logical_numel=16, demotion_band=1)
    carrier.apply_step(0, votes=votes[0]["toy.proj"])
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step=votes,
        include_r4v_ledger=False,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is False
    assert verdict["measure_r4v_ledger_pass"] is None


def test_phase_a_parity_witness_passes_with_probe_r4v_ledger_receipt(
    tmp_path: Path,
) -> None:
    import torch

    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        apply_bounded_delta_vote_step,
        make_event_coded_live_tensor_state,
    )
    from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec

    logical_numel = 256
    side = 16
    q = torch.zeros((side, side), dtype=torch.int8)
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        q,
        0.25,
        demotion_band=1,
    )
    vote_spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=16,
    )
    votes = torch.zeros(logical_numel, dtype=torch.int16)
    votes[0] = 12
    apply_bounded_delta_vote_step(
        {"toy.proj": state},
        {"toy.proj": votes.view(side, side)},
        {"toy.proj": vote_spec},
    )
    r4v_ledger = probe.build_r4v_persistent_ledger_receipt(
        {"toy.proj": state},
        event_coded_live_enabled=True,
    )
    assert r4v_ledger["enabled"] is True
    assert r4v_ledger["ledger_pass"] is True
    assert "content_sha256" in r4v_ledger

    carrier = EventCodedAccLiveState(logical_numel=logical_numel, demotion_band=1)
    carrier.apply_step(0, votes={0: 12})
    surfaces = {"toy.proj": observed_surfaces_dict(carrier.step_records[-1])}
    phase_root = _write_phase_bundle(
        tmp_path,
        gpu_surfaces_by_step={0: surfaces},
        votes_by_step={0: {"toy.proj": {0: 12}}},
        r4v_ledger_override=r4v_ledger,
        logical_numel=logical_numel,
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is True
    assert verdict["measure_r4v_ledger_pass"] is True


V4_LAUNCH_PACKET_MAIN = Path(
    "artifacts/consensus_prep/v4_live_trainer_integration_gpu_launch_packet_v1.json"
)
V4_LAUNCH_PACKET_COMPANION = Path(
    "artifacts/consensus_prep/v4_live_trainer_integration_gpu_launch_packet_v1_replay_commands.json"
)


@pytest.mark.parametrize(
    "artifact_path",
    [V4_LAUNCH_PACKET_MAIN, V4_LAUNCH_PACKET_COMPANION],
    ids=["v4_main_packet", "v4_companion_replay_commands"],
)
def test_v4_launch_artifacts_both_json_load(artifact_path: Path) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload


def test_v4_launch_packet_contract_and_bindings() -> None:
    import hashlib

    packet = json.loads(V4_LAUNCH_PACKET_MAIN.read_text(encoding="utf-8"))
    companion = json.loads(V4_LAUNCH_PACKET_COMPANION.read_text(encoding="utf-8"))
    comp_sha = hashlib.sha256(V4_LAUNCH_PACKET_COMPANION.read_bytes()).hexdigest()
    assert packet["artifact_binding_dag"]["replay_commands_sha256"] == comp_sha
    assert packet["packet_sha"] not in {"", "TBD_AT_GATE"}
    packet_copy = dict(packet)
    packet_copy["packet_sha"] = ""
    expected_packet_sha = hashlib.sha256(
        json.dumps(packet_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert packet["packet_sha"] == expected_packet_sha

    phase_a = packet["phase_plan"]["phase_a_gpu_correctness_smoke"]
    witness_cmd = str(phase_a["parity_witness_command"])
    assert "hrm_text_158_v4_live_phase_a_parity_witness.py" in witness_cmd
    assert phase_a["parity_witness_pass_assertion"] == "phase_a_parity_pass==true"
    phase_b = packet["phase_plan"]["phase_b_live_dynamics"]
    assert phase_b["only_after_phase_a_parity_witness_pass"] is True

    phase_a_cmd = str(companion["phase_a_gpu_smoke_command_template"])
    phase_b_cmd = str(companion["phase_b_live_dynamics_command_template"])
    for cmd in (phase_a_cmd, phase_b_cmd):
        assert "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1" in cmd
        assert "--persistent-accumulator-event-coded-live" in cmd
        assert "--event-coded-live-demotion-band" in cmd
        assert "--votes-emit-enabled" in cmd
        assert "--allow-gpu-launch" in cmd
        assert "--science-arm A0_rank_bucket_current_ordering" in cmd
        assert "--science-arm V4" not in cmd
    forbidden = set(companion.get("forbidden_flags", []))
    assert "--expect-ready" in forbidden
    assert "--two-tier-carry-w6-enabled" in forbidden
    assert "--r7-deferred-backlog-carry" in forbidden
    assert "--persistent-accumulator-w6-byte-packed" in forbidden
    assert "--persistent-accumulator-w5-byte-packed" in forbidden
    assertions = companion["sub2_readiness_assertions"]
    assert assertions["ready_for_pre_full_stack_diagnostic"] is True
    assert assertions["ready_for_main_science"] is False
    assert assertions["main_science_launch_blocked"] is True
    assert "--expect-ready" not in str(companion["sub2_readiness_command"])

    required_gpu_env = packet["fail_closed_constraints"]["required_gpu_launch_env"]
    assert required_gpu_env == {"HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1"}

    # code_pins currency: per-file shas are authoritative; head is advisory
    code_pins = packet["code_pins"]
    assert code_pins["preflight_authority"] == "file_content_sha256_at_head"
    assert code_pins["head_required_advisory"] is True
    assert code_pins["head_mismatch_fail_closed"] is False
    assert code_pins["file_content_sha_mismatch_fail_closed"] is True
    assert code_pins["head_required"] == "2abfd51110689e81a821ee83219f7081145435fd"
    pinned_paths = list(code_pins["file_content_sha256_at_head"].keys())
    assert any(p.endswith("v4_live_phase_a_parity_witness.py") for p in pinned_paths)
    assert "scripts/hrm_text_158_v4_live_phase_a_parity_witness.py" in code_pins[
        "file_content_sha256_at_head"
    ]
    assert "scripts/hrm_text_158_v4_file_content_pin_witness.py" in code_pins[
        "file_content_sha256_at_head"
    ]
    assert (
        code_pins["file_content_sha256_at_head"][
            "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
        ]
        == "57f3570470c48b8811e4da0caf74397a64b7d2d7135c11b7691dea59f3625ccd"
    )
    assert len(code_pins["file_content_sha256_at_head"]) == 17
    assert set(code_pins["pinned_surfaces"]) == set(
        code_pins["file_content_sha256_at_head"].keys()
    )
    for rel_path, expected_sha in code_pins["file_content_sha256_at_head"].items():
        on_disk_sha = hashlib.sha256(Path(rel_path).read_bytes()).hexdigest()
        assert on_disk_sha == expected_sha, f"stale pin for {rel_path}: {expected_sha} != {on_disk_sha}"

    assert companion.get("preflight_authority") == "file_content_sha256_at_head"
    assert "file_content_pin_witness_command" in companion
    assert companion.get("head_pin_advisory") is True
    witness_cmd = str(companion["file_content_pin_witness_command"])
    assert "hrm_text_158_v4_file_content_pin_witness.py" in witness_cmd
    assert "--packet" in witness_cmd
    assert "--json-out" in witness_cmd
    import py_compile

    py_compile.compile(
        "scripts/hrm_text_158_v4_file_content_pin_witness.py",
        doraise=True,
    )


def test_file_content_pin_witness_passes_when_pins_match_disk(tmp_path: Path) -> None:
    import subprocess

    out_path = tmp_path / "prelaunch" / "file_content_pin_witness.json"
    proc = subprocess.run(
        [
            "python3",
            "scripts/hrm_text_158_v4_file_content_pin_witness.py",
            "--packet",
            str(V4_LAUNCH_PACKET_MAIN),
            "--json-out",
            str(out_path),
        ],
        cwd=Path("."),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["file_content_pin_pass"] is True
    assert payload["all_match"] is True
    assert payload["mismatch_count"] == 0
    assert out_path.is_file()


def test_file_content_pin_witness_fails_on_tampered_pin(tmp_path: Path) -> None:
    from scripts.hrm_text_158_v4_file_content_pin_witness import run_file_content_pin_witness

    packet = json.loads(V4_LAUNCH_PACKET_MAIN.read_text(encoding="utf-8"))
    pins = dict(packet["code_pins"]["file_content_sha256_at_head"])
    first_path = sorted(pins.keys())[0]
    pins[first_path] = "0" * 64
    out_path = tmp_path / "prelaunch" / "file_content_pin_witness.json"
    payload = run_file_content_pin_witness(
        packet_path=V4_LAUNCH_PACKET_MAIN,
        json_out=out_path,
        pins_override=pins,
    )
    assert payload["file_content_pin_pass"] is False
    assert payload["all_match"] is False
    assert payload["mismatch_count"] >= 1


def test_file_content_pin_witness_cli_fails_on_tampered_pin(tmp_path: Path) -> None:
    import subprocess

    packet = json.loads(V4_LAUNCH_PACKET_MAIN.read_text(encoding="utf-8"))
    tampered = dict(packet)
    pins = dict(packet["code_pins"]["file_content_sha256_at_head"])
    first_path = sorted(pins.keys())[0]
    pins[first_path] = "f" * 64
    tampered["code_pins"]["file_content_sha256_at_head"] = pins
    tampered_packet = tmp_path / "tampered_packet.json"
    tampered_packet.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_path = tmp_path / "prelaunch" / "file_content_pin_witness.json"
    proc = subprocess.run(
        [
            "python3",
            "scripts/hrm_text_158_v4_file_content_pin_witness.py",
            "--packet",
            str(tampered_packet),
            "--json-out",
            str(out_path),
        ],
        cwd=Path("."),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["file_content_pin_pass"] is False
