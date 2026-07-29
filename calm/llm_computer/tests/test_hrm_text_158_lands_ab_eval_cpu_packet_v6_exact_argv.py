"""IMPLEMENT packet_v6: exact-argv dry-exec + revision-neutral operative binding.

Gate-2 BLOCK 1785315799719-cdb71d80: v5 operative_packet_binding.meaning still said
"this packet_v4"; forbidden list omitted v4.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_FIXTURE_CONTRACT_FAIL,
    GATING_ROWS,
)
from calm.llm_computer.tests.lands_ab_eval_test_helpers import (
    make_cuda_fixture_fail_obs,
    write_real_cpu_row,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKET_V6_REL = (
    "artifacts/acc_entropy/"
    "optimizer_credit_state_sparse_vote_authority_LANDS_AB_EVAL_launch_packet_v6.json"
)
PACKET_V6_SHA256 = (
    "6a97928392384d1769164c8c6bc3ef86c6474deb5086153ccf5d17dcb66f9722"
)
DEAD_PACKET_SHA_FIELD = re.compile(r"^packet_v[0-9]+_sha256$")
THIS_PACKET_VN = re.compile(r"this packet_v[0-9]+")


def _load_packet_v6() -> dict:
    path = REPO_ROOT / PACKET_V6_REL
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PACKET_V6_SHA256, "packet_v6_sha_mismatch"
    return json.loads(raw.decode("utf-8"))


def _argv_flag(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    return argv[argv.index(flag) + 1]


def _seed_seven_row_fixture(run_root: Path) -> None:
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (
        o_excl_write_json,
        runtime_scratch_raw_path,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    write_real_cpu_row(run_root)
    for row in GATING_ROWS:
        if row == "G_CPU_STATIC_AB":
            continue
        obs = make_cuda_fixture_fail_obs(row, key="lin")
        p = runtime_scratch_raw_path(
            scratch_dir=run_root, gating_row=row, run_nonce=uuid.uuid4().hex[:8]
        )
        o_excl_write_json(p, obs)


def test_packet_v6_terminal_fields_bind_operative_not_dead_revision():
    pkt = _load_packet_v6()
    fields = pkt["terminal_classification"]["terminal_receipt_required_fields"]
    assert "operative_packet_path_and_sha256" in fields
    assert not any(DEAD_PACKET_SHA_FIELD.fullmatch(f) for f in fields)


def test_packet_v6_operative_packet_binding_is_revision_neutral():
    """Entire operative_packet_binding subtree: no this packet_vN; generic forbid rule."""
    pkt = _load_packet_v6()
    opb = pkt["terminal_classification"]["operative_packet_binding"]
    blob = json.dumps(opb, sort_keys=True)
    assert not THIS_PACKET_VN.search(blob), blob
    # meaning revision-neutral
    assert "this packet_v" not in opb["meaning"]
    assert "persisted go record" in opb["meaning"] or "go record" in opb["meaning"]
    # forbidden rule covers all dead revisions generically
    forbid_blob = json.dumps(opb.get("forbidden_as_required_live_fields", []))
    regex = opb.get("forbidden_required_live_field_regex", "")
    assert "packet_v[0-9]+_sha256" in forbid_blob or re.search(
        r"packet_v\[0-9\]\+_sha256", regex
    )
    # no dead revision described as operative inside this subtree
    assert not re.search(r"this packet_v[1-5]\b", blob)
    # lineage_policy lists dead revs; forbidden rule must cover them
    dead_revs = [
        k for k, v in (pkt.get("lineage_policy") or {}).items() if v.get("status") == "DEAD"
    ]
    assert dead_revs  # at least v1..v5
    # generic regex covers any packet_vN_sha256
    assert re.fullmatch(r"\^?packet_v\[0-9\]\+_sha256\$?", regex) or "packet_v[0-9]+_sha256" in forbid_blob


def test_packet_v6_enforcer_path_aliases_match_argv_and_stale_key_absent():
    pkt = _load_packet_v6()
    cuda_rows = 0
    for row in pkt["row_commands"]:
        gr = row["gating_row"]
        inv = row["invocation"]
        assert "enforcer_receipt_template" not in inv
        if not gr.startswith("G_CUDA"):
            continue
        cuda_rows += 1
        argv = inv["argv_template"]
        enf = _argv_flag(argv, "--enforcer-receipt")
        phase = _argv_flag(argv, "--phase-events-jsonl")
        assert enf is not None and phase is not None
        assert inv["enforcer_receipt_path_template"] == enf
        assert inv["terminal_collection_enforcer_receipt_path"] == enf
        assert inv["phase_events_jsonl_template"] == phase
        assert inv["terminal_collection_phase_events_jsonl_path"] == phase
        pre = inv.get("preflight_must_not_exist") or []
        assert enf in pre and phase in pre
        assert not any(re.search(r"enforcer_receipt_.*-<nonce>\.json$", p) for p in pre)
    assert cuda_rows == 6


def test_packet_v6_exact_argv_consumer_reaches_fixture_contract_fail(tmp_path: Path):
    pkt = _load_packet_v6()
    inv = pkt["evidence_consumer_command"]["invocation"]
    argv = list(inv["argv_template"])
    assert argv[:4] == ["timeout", "120", "python3", "-c"]
    script = argv[4]
    assert "sha256" in script and "path" in script
    assert "paths={gr: str(" not in script

    run_root = tmp_path / "test-operator" / uuid.uuid4().hex
    template_root = pkt["runtime_scratch"]["run_root_template"]
    assert "<nonce>" in template_root
    script_live = script.replace(template_root, str(run_root))
    assert "<nonce>" not in script_live

    _seed_seven_row_fixture(run_root)

    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["LANDS_AB_RUN_ROOT"] = str(run_root)

    r = subprocess.run(
        [sys.executable, "-c", script_live],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines, r.stdout
    summary = json.loads(lines[-1])
    assert summary["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL
    assert summary["science_claim"] is False
    assert summary["LANDS_AB"] is False

    man_path = run_root / "raw_artifact_manifest.json"
    eval_path = run_root / "eval_receipt.json"
    assert man_path.is_file() and eval_path.is_file()
    assert summary["manifest_sha256"] == hashlib.sha256(man_path.read_bytes()).hexdigest()
    assert summary["eval_receipt_sha256"] == hashlib.sha256(eval_path.read_bytes()).hexdigest()

    man = json.loads(man_path.read_text())
    assert set(man["raw_artifact_paths"].keys()) == set(GATING_ROWS)
    for _gr, entry in man["raw_artifact_paths"].items():
        assert set(entry.keys()) >= {"path", "sha256"}
        assert len(entry["sha256"]) == 64
        p = Path(entry["path"])
        assert p.is_file()
        assert entry["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()

    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import o_excl_write_text

    with pytest.raises(FileExistsError):
        o_excl_write_text(man_path, "{}")
    with pytest.raises(FileExistsError):
        o_excl_write_text(eval_path, "{}")


def test_packet_v6_consumer_script_rejects_path_string_entries_if_rewound():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        build_eval_receipt_from_raw_artifacts,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
        DEFAULT_SOURCE_PINS,
    )

    with pytest.raises(ValueError, match="raw_artifact_entry_incomplete|raw_artifact"):
        build_eval_receipt_from_raw_artifacts(
            raw_artifact_paths={row: "/tmp/not-a-map" for row in GATING_ROWS},  # type: ignore[dict-item]
            source_pins=DEFAULT_SOURCE_PINS,
            required_key_set=["proj"],
        )
