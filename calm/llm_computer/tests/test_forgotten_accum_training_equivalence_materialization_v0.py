"""CPU identity/materialization characterization (A-CFG + complete inventory)."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    PARENT_SHA256_FULL,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_dense_site_smoke import (
    DEFAULT_PARENT_RELPATH,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_materialization import (
    IdentityRefuse,
    assert_complete_eligible_inventory,
    assert_no_default_applied,
    assert_use_ternary_bulk_true,
    inventory_model_config_field_provenance,
    load_verified_parent_checkpoint,
    materialize_run_arms_live_bundle,
    run_readonly_actual_parent_preflight,
    verify_parent_bytes_sha256,
)

REPO = Path(__file__).resolve().parents[3]


def _tiny_ckpt_from_real() -> dict:
    ckpt = torch.load(
        REPO / DEFAULT_PARENT_RELPATH, map_location="cpu", weights_only=False
    )
    # Keep a deep copy for mutation fixtures; tests that need a full model use real parent.
    return copy.deepcopy(ckpt)


def test_verify_parent_sha_before_deserialize(tmp_path: Path):
    src = REPO / DEFAULT_PARENT_RELPATH
    blob = src.read_bytes()
    bad = tmp_path / "bad.pt"
    bad.write_bytes(blob + b"x")
    with pytest.raises(IdentityRefuse, match="pre-load"):
        verify_parent_bytes_sha256(bad, PARENT_SHA256_FULL)


def test_a_cfg_default_applied_refuses():
    cfg = {"max_seq_len": 8, "n_layers": 1, "hidden_size": 8, "num_heads": 1,
           "expansion": 1.0, "H_cycles": 1, "L_cycles": 1, "half_layers": False,
           "bp_warmup_ratio": 0.0, "bp_min_steps": 1, "bp_max_steps": 1,
           "use_ternary_bulk": True}
    # rope_theta absent => DEFAULT_APPLIED
    prov = inventory_model_config_field_provenance(cfg)
    assert prov["rope_theta"]["source"] == "DEFAULT_APPLIED"
    with pytest.raises(IdentityRefuse, match="DEFAULT_APPLIED"):
        assert_no_default_applied(prov)


def test_use_ternary_bulk_absent_and_false_refuse():
    with pytest.raises(IdentityRefuse, match="missing"):
        assert_use_ternary_bulk_true({})
    with pytest.raises(IdentityRefuse, match="explicit True"):
        assert_use_ternary_bulk_true({"use_ternary_bulk": False})


def test_complete_inventory_empty_and_partial_refuse():
    class _M(torch.nn.Module):
        pass

    model = _M()
    with pytest.raises(IdentityRefuse, match="> 0"):
        assert_complete_eligible_inventory(
            model=model, eligible_modules={}, tensor_states={}, eligible_scope="all-bitlinear"
        )


def test_pre_post_parent_hash_law():
    path = REPO / DEFAULT_PARENT_RELPATH
    ckpt, pre, post = load_verified_parent_checkpoint(
        path, expected_sha256=PARENT_SHA256_FULL
    )
    assert pre == post == PARENT_SHA256_FULL
    assert "config" in ckpt and "model_state" in ckpt


def test_materialize_fixture_use_ternary_bulk_absent(tmp_path: Path, monkeypatch):
    ckpt = _tiny_ckpt_from_real()
    del ckpt["config"]["use_ternary_bulk"]
    path = tmp_path / "no_ternary.pt"
    torch.save(ckpt, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def fake_load(p, *, expected_sha256):
        return ckpt, expected_sha256

    monkeypatch.setattr(
        "scripts.hrm_text_158_bounded_delta_acquisition_probe.load_parent_checkpoint",
        fake_load,
    )
    # Still hash-check against provided sha of mutated file.
    with pytest.raises(IdentityRefuse, match="use_ternary_bulk|DEFAULT_APPLIED"):
        materialize_run_arms_live_bundle(
            parent_path=path, expected_parent_sha256=sha, device="cpu"
        )


def test_materialize_fixture_rope_theta_absent(tmp_path: Path, monkeypatch):
    ckpt = _tiny_ckpt_from_real()
    del ckpt["config"]["rope_theta"]
    path = tmp_path / "no_rope.pt"
    torch.save(ckpt, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    def fake_load(p, *, expected_sha256):
        return ckpt, expected_sha256

    monkeypatch.setattr(
        "scripts.hrm_text_158_bounded_delta_acquisition_probe.load_parent_checkpoint",
        fake_load,
    )
    with pytest.raises(IdentityRefuse, match="DEFAULT_APPLIED|rope_theta"):
        materialize_run_arms_live_bundle(
            parent_path=path, expected_parent_sha256=sha, device="cpu"
        )


def test_readonly_actual_parent_preflight():
    receipt = run_readonly_actual_parent_preflight(
        repo_root=REPO,
        parent_relpath=DEFAULT_PARENT_RELPATH,
        expected_parent_sha256=PARENT_SHA256_FULL,
        device="cpu",
    )
    assert receipt["status"] == "OK"
    inv = receipt["identity_inventory"]
    assert inv["parent_sha256_pre_load"] == PARENT_SHA256_FULL
    assert inv["parent_sha256_post_load_echo"] == PARENT_SHA256_FULL
    assert receipt["eligible_module_count"] > 0
    assert receipt["fidelity_all_pass"] is True
    prov = inv["model_config_field_provenance"]
    assert all(v["source"] == "checkpoint" for v in prov.values())
    assert prov["use_ternary_bulk"]["value"] is True
    assert inv["build_identity_full_support_batches"].endswith(":778")
