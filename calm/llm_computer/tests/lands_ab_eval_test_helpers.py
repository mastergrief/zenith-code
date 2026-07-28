"""Shared LANDS-AB CPU test helpers (IMPLEMENT_v12 dedupe)."""
from __future__ import annotations

import uuid
from pathlib import Path

from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import all_true_matrix


def base_ok(**over):
    p = {
        "scope_creep": False,
        "fixture_contract_raw_fail": False,
        "surface_pass_by_row": all_true_matrix(),
    }
    p.update(over)
    return p


def write_real_cpu_row(scratch: Path):
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
        measure_g_cpu_static_ab,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    obs = measure_g_cpu_static_ab()
    path = runtime_scratch_raw_path(
        scratch_dir=scratch, gating_row="G_CPU_STATIC_AB", run_nonce=uuid.uuid4().hex[:8]
    )
    sha = o_excl_write_json(path, obs)
    return obs, sha, path
