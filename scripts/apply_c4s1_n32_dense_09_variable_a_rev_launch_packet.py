#!/usr/bin/env python3
"""Apply Fold-3B Variable A reversed-order mechanism-diagnosis GPU launch packet."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
REV14_DRAFT = (
    REPO / "artifacts/consensus_prep/c4s1_phase3_gpu_callsite_acceptance_launch_packet_v1_draft.json"
)
REV14_REPLAY = (
    REPO
    / "artifacts/consensus_prep/c4s1_phase3_gpu_callsite_acceptance_launch_packet_v1_replay_commands.json"
)
DRAFT = (
    REPO
    / "artifacts/consensus_prep/c4s1_n32_dense_09_variable_a_rev_launch_packet_v1_draft.json"
)
REPLAY = (
    REPO
    / "artifacts/consensus_prep/c4s1_n32_dense_09_variable_a_rev_launch_packet_v1_replay_commands.json"
)
HEAD = "bd23cc9f3dd8e2dfc1245e80f970c2a5baaf1888"
SCIENCE_HEAD = "bd23cc9f3dd8e2dfc1245e80f970c2a5baaf1888"
ACTIVE_TASK_ID = "1782633464140-b85ec12a"
UPSTREAM_TASK_ID = "1782633464140-b85ec12a"
RUN_ROOT = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_VARIABLE_A_REV_V1"
)
RUN_ID = "C4S1_PHASE3_N32_DENSE_09_VARIABLE_A_REV_V1"
N_STATES = 32
PACKET_REVISION = "v1_n32_dense_09_variable_a_rev_bd23cc9"
FOLD1_SPEC = (
    "artifacts/measurement_closeout/c4s1d7_dense_09_structural_fork_resolution_spec.md"
)
PREREG_PACKET_PATH = (
    "artifacts/consensus_prep/c4s1_fold3b_step1_prereg_packet_v1_draft.json"
)
IDENTITY_INERTNESS_WRAPPER_PATH = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_IDENTITY_INERTNESS_V1/"
    "prelaunch/ca_confirmation_wrapper_receipt.json"
)
MECHANISM_RECEIPT_NAME = "fold3b_variable_a_mechanism_diagnosis_receipt.json"
VARIABLE_ID = "A_order_only"
CONTROL_REASON = "order_only_perturbation"
F3B_DECISIVE_BRANCHES = frozenset(
    {
        "F3B_STATE0_IDENTITY_STRUCTURE",
        "F3B_MEASUREMENT_ORDER_ARTIFACT",
    }
)
RSS_FALLBACK_GIB = 6.5
DENSE_SAMPLED_STATES_ENV = "HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES"
DENSE_SAMPLED_STATES_VALUE = "0,1,2,3,4,5,6,7,8,9"
DENSE_SAMPLED_STATES_LIST = list(range(10))
DENSE_ORDER_ENV = "HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER"
DENSE_ORDER_VALUE = "9,8,7,6,5,4,3,2,1,0"
DENSE_ORDER_LIST = list(range(9, -1, -1))
EXPECTED_MARK_COUNT = len(DENSE_SAMPLED_STATES_LIST)
EXPECTED_EFFECTIVE_VISIT_ORDER = list(range(9, -1, -1)) + list(range(10, 32))

PINS: dict[str, str] = {
    "scripts/hrm_text_158_slice5_v6i_oom_profile_attribution.py": (
        "33340dffc28f45fee02a7204802f489e4810337a0ab9ed64e56df628817bdaeb"
    ),
    "scripts/hrm_text_158_code_currency_guard.py": (
        "c8aae32b7125c3e683ff026e82fe776d04f6ff4b28cc32ed17e9ca0ca356cb0c"
    ),
    "scripts/hrm_text_158_bounded_delta_acquisition_probe.py": (
        "c3a7a4dbdc1e14d3ff631c0922b3f7b65c5a0c94594e54a7c398ac3afbc2a797"
    ),
    "scripts/hrm_text_158_bounded_delta_acquisition_probe_bootstrap.py": (
        "c7e5ab2283ba14f26db2fb0e4f3892aab786a60302c90d3e7e98b5393c02b27f"
    ),
    "scripts/box_lane_code_currency_preflight.py": (
        "08bc8d13fbac548aea49d90db1af252d1cbde4abe867287f127f228f59ab1ecc"
    ),
    "calm/hrm_text_158/native_full_stack/box_lane.py": (
        "987a9a48e9a841c64ee35b3266541b98c593ab3523db1af70e42163c93ed7744"
    ),
    "scripts/hrm_text_158_r7_resource_lane_acquire.py": (
        "c69fd07f9416f15bb2fdc0c6d11b4c10d85c145c57d015e929653ac7f25df027"
    ),
    "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py": (
        "624f3ac39945cf5ef5caa5a249a4a168a1071b3057ef4469f038d565540aa412"
    ),
    "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py": (
        "af20de289b2b22e18ea1bf169cb20b567343c62b26b7ab782d90e624f7aa520f"
    ),
    "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py": (
        "ab79f8bcb1abc800e3f5217247b187f83455c60b5a46427d1d3ae7a04eb865d1"
    ),
    "calm/hrm_text_158/native_full_stack/host_tracemalloc_probe.py": (
        "4a680f248569f93d5b665ff9581eaf5c3e227b65037fd1da3305a0b49a945cf1"
    ),
    "calm/hrm_text_158/native_full_stack/s1d7_band_counter.py": (
        "1f2eabb8c3bc9c7b50745b7b73f92cacbd9484c6d1a910df6b933c2bf8693688"
    ),
    "calm/hrm_text_158/native_full_stack/s1d7_tracemalloc_feasibility.py": (
        "f1af399e75968a2752431abd556919bbfa61bd9d352a70d6b88f2bb541142794"
    ),
    "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py": (
        "a0dc750edd98a64dca629e2989ca9ec646b44f7074a0ecbb9408840bff3f5c11"
    ),
    "calm/hrm_text_158/native_full_stack/f3b_why_state0_branch.py": (
        "21ad6a9b0d74e1c47ec6e228aad1beefd5e2daadd3ea861aa47ee50746344afb"
    ),
}

F3B_CLASSIFIER_REL = "calm/hrm_text_158/native_full_stack/f3b_why_state0_branch.py"

# Every code_pins field (except git_head) maps to a launch-executed repo path.
# test_slice5 is intentionally omitted — not imported/executed by the GPU run.
CODE_PIN_FIELD_TO_REL: dict[str, str] = {
    "bounded_delta_learner_sha256": (
        "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py"
    ),
    "event_coded_acc_live_carrier_sha256": (
        "calm/hrm_text_158/native_full_stack/event_coded_acc_live_carrier.py"
    ),
    "event_coded_vote_update_adapter_sha256": (
        "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py"
    ),
    "sparse_cap_gpu_seam_adapter_sha256": (
        "calm/hrm_text_158/native_full_stack/sparse_cap_gpu_seam_adapter.py"
    ),
    "host_tracemalloc_probe_sha256": (
        "calm/hrm_text_158/native_full_stack/host_tracemalloc_probe.py"
    ),
    "s1d7_band_counter_sha256": "calm/hrm_text_158/native_full_stack/s1d7_band_counter.py",
    "s1d7_tracemalloc_feasibility_sha256": (
        "calm/hrm_text_158/native_full_stack/s1d7_tracemalloc_feasibility.py"
    ),
    "hrm_text_158_bounded_delta_acquisition_probe_sha256": (
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
    ),
    "bounded_delta_acquisition_probe_sha256": (
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
    ),
    "hrm_text_158_slice5_v6i_oom_profile_attribution_sha256": (
        "scripts/hrm_text_158_slice5_v6i_oom_profile_attribution.py"
    ),
    "hrm_text_158_code_currency_guard_sha256": (
        "scripts/hrm_text_158_code_currency_guard.py"
    ),
    "probe_bootstrap_sha256": (
        "scripts/hrm_text_158_bounded_delta_acquisition_probe_bootstrap.py"
    ),
    "r7_resource_lane_acquire_sha256": (
        "scripts/hrm_text_158_r7_resource_lane_acquire.py"
    ),
    "box_lane_code_currency_preflight_sha256": (
        "scripts/box_lane_code_currency_preflight.py"
    ),
    "box_lane_sha256": "calm/hrm_text_158/native_full_stack/box_lane.py",
    "f3b_why_state0_branch_sha256": (
        "calm/hrm_text_158/native_full_stack/f3b_why_state0_branch.py"
    ),
}

# fold-2d-b: science pins verified @ git_head_required (fold-2a source baseline FDC).
# Launch-executed infra tooling may advance in packet-only commits; verify on-disk.
INFRA_PIN_RELS_ON_DISK: frozenset[str] = frozenset(
    {
        "scripts/box_lane_code_currency_preflight.py",
        "calm/hrm_text_158/native_full_stack/box_lane.py",
    }
)
INFRA_CODE_PIN_FIELDS_ON_DISK: frozenset[str] = frozenset(
    {
        "box_lane_code_currency_preflight_sha256",
        "box_lane_sha256",
    }
)
BOX_LANE_REL = "calm/hrm_text_158/native_full_stack/box_lane.py"

RSS_FALLBACK_GIB = 6.5
FALLBACK_N_STATES = 16
OLD_CALLSITE_RUN_ID = "C4S1_PHASE3_GPU_CALLSITE_V1"
FAIL_CLOSED_B_ARM_EXIT_CODES = frozenset({37, -6})
SAMPLED_STATES_RULE = (
    "dense [0..9] via HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES env override; "
    "reversed order [9..0] via HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER"
)


def compute_expected_sampled_states(n_states: int) -> frozenset[int]:
    """Dense decider: contiguous early states 0..9 (env override at launch)."""
    n = int(n_states)
    if n <= 0:
        return frozenset()
    if n < EXPECTED_MARK_COUNT:
        raise ValueError(
            f"dense-[0..9] decider requires n_states>={EXPECTED_MARK_COUNT}; got {n}"
        )
    return frozenset(DENSE_SAMPLED_STATES_LIST)


def expected_mark_event_count(n_states: int) -> int:
    _ = int(n_states)
    return EXPECTED_MARK_COUNT


def evaluate_feasibility_subsample_fallback_trigger(
    primary_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Packet-only trigger: resource evidence (timeout/RSS), never exit 37/-6 fail-closed."""
    peak_rss = primary_receipt.get("peak_rss_gib")
    b = primary_receipt.get("runs", {}).get("B", {})
    exit_code = int(b.get("exit_code", 0) or 0)
    timeout_breach = bool(b.get("subprocess_timeout_expired"))
    rss_breach = peak_rss is not None and float(peak_rss) > RSS_FALLBACK_GIB
    fail_closed_exit = exit_code in FAIL_CLOSED_B_ARM_EXIT_CODES
    fallback = (rss_breach or timeout_breach) and not fail_closed_exit
    return {
        "exit_code": exit_code,
        "fail_closed_exit": fail_closed_exit,
        "fallback": fallback,
        "rss_breach": rss_breach,
        "timeout_breach": timeout_breach,
    }


CA_CONFIRMATION_HEREDOC = r"""import json, sys
from pathlib import Path
from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
    ca_confirmation_wrapper_exit_code,
    orchestrate_ca_confirmation_with_fallback,
)
from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import (
    build_branch_input_contract_from_ca_receipt,
    classify_f3b_why_state0_branch,
)

DENSE_SAMPLED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DENSE_ORDER = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
EXPECTED_EFFECTIVE = DENSE_ORDER + list(range(10, 32))
IDENTITY_INERTNESS_WRAPPER_PATH = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_IDENTITY_INERTNESS_V1/"
    "prelaunch/ca_confirmation_wrapper_receipt.json"
)
MECHANISM_RECEIPT_NAME = "fold3b_variable_a_mechanism_diagnosis_receipt.json"
WRAPPER_RECEIPT_NAME = "ca_confirmation_wrapper_receipt.json"
GIT_HEAD_REQUIRED = "bd23cc9f3dd8e2dfc1245e80f970c2a5baaf1888"


def compute_primary_dense_fork_readable(wrapper: dict) -> bool:
    try:
        primary = (wrapper or {}).get("runs", {}).get("primary")
        if not isinstance(primary, dict):
            return False
        sampled = primary.get("sampled_states")
        if list(sampled) != DENSE_SAMPLED:
            return False
        if int(primary.get("s1d7_band_counter_mark_count", -1)) != 10:
            return False
        checks = primary.get("checks") or {}
        if checks.get("s1d7_band_counter_mark_count_eq_sampled_state_count") is not True:
            return False
        per_state = primary.get("per_state") or []
        if not isinstance(per_state, list):
            return False
        covered = {
            int(row.get("state_index"))
            for row in per_state
            if isinstance(row, dict) and row.get("state_index") is not None
        }
        if covered != set(DENSE_SAMPLED):
            return False
        b_run = (primary.get("runs") or {}).get("B") or {}
        if int(b_run.get("exit_code", -1)) != 0:
            return False
        if bool(b_run.get("subprocess_timeout_expired")):
            return False
        if primary.get("infra_ok") is not True:
            return False
        return True
    except Exception:
        return False


def _reversed_order_provenance_ok(primary: dict) -> bool:
    try:
        sampled_set = list(primary.get("sampled_state_set") or [])
        if sampled_set != list(DENSE_SAMPLED):
            return False
        sampled_order = list(primary.get("sampled_state_order") or [])
        if sampled_order != list(DENSE_ORDER):
            return False
        if primary.get("order_control_active") is not True:
            return False
        if primary.get("order_perturbation_kind") != "sampled_block_order_perturbation":
            return False
        effective = list(primary.get("effective_visit_order") or [])
        if effective != list(EXPECTED_EFFECTIVE):
            return False
        rank_map = primary.get("order_rank_by_semantic_state") or {}
        for idx, state in enumerate(DENSE_ORDER):
            if int(rank_map.get(str(state), -1)) != idx:
                return False
        per_state = primary.get("per_state") or []
        for row in per_state:
            if not isinstance(row, dict):
                continue
            state_index = row.get("state_index")
            if state_index is None:
                continue
            if int(state_index) not in set(DENSE_SAMPLED):
                continue
            if row.get("semantic_state_id") != state_index:
                return False
        return True
    except Exception:
        return False


def _identity_inertness_precondition_ok() -> bool:
    try:
        wrapper_path = Path(IDENTITY_INERTNESS_WRAPPER_PATH)
        if not wrapper_path.is_file():
            return False
        wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
        return bool(wrapper.get("identity_inertness_proven") is True)
    except Exception:
        return False


def build_mechanism_diagnosis_receipt(wrapper: dict, run_root: Path) -> dict:
    primary = (wrapper or {}).get("runs", {}).get("primary")
    fallback = (wrapper or {}).get("runs", {}).get("fallback")
    wrapper_path = str(run_root / "prelaunch" / WRAPPER_RECEIPT_NAME)
    primary_path = None
    fallback_path = None
    if isinstance(primary, dict):
        primary_path = primary.get("receipt_path")
    if isinstance(fallback, dict):
        fallback_path = fallback.get("receipt_path")

    identity_inert = _identity_inertness_precondition_ok()
    operational_ok = not bool(wrapper.get("abnormal_exit"))
    if not isinstance(primary, dict):
        branch_inputs = {
            "operational_ok": False,
            "schema_ok": False,
            "variable_id": "A_order_only",
            "control_reason": "order_only_perturbation",
            "identity_order_inertness_proven": identity_inert,
        }
        classified = classify_f3b_why_state0_branch(branch_inputs)
    else:
        branch_inputs = build_branch_input_contract_from_ca_receipt(
            primary,
            variable_id="A_order_only",
            control_reason="order_only_perturbation",
            identity_order_inertness_proven=identity_inert,
            operational_ok=operational_ok and compute_primary_dense_fork_readable(wrapper),
        )
        if not _reversed_order_provenance_ok(primary):
            branch_inputs["schema_ok"] = False
        classified = classify_f3b_why_state0_branch(branch_inputs)

    per_state = list((primary or {}).get("per_state") or [])
    semantic_state_id = [
        int(row.get("semantic_state_id", row.get("state_index")))
        for row in per_state
        if isinstance(row, dict) and row.get("state_index") is not None
    ]
    return {
        "schema": "hrm_text_158_fold3b_mechanism_diagnosis_receipt/v1",
        "classifier": classified.get("classifier"),
        "f3b_branch": classified.get("terminal_branch"),
        "f3b_terminal_branch": classified.get("terminal_branch"),
        "f3b_fired_branches": classified.get("fired_branches"),
        "f3b_branch_inputs": classified.get("f3b_branch_inputs"),
        "variable_id": "A_order_only",
        "control_reason": "order_only_perturbation",
        "sampled_state_set": list(DENSE_SAMPLED),
        "sampled_state_order": list(DENSE_ORDER),
        "order_rank_by_semantic_state": (primary or {}).get("order_rank_by_semantic_state") or {},
        "semantic_state_id": semantic_state_id,
        "per_state": per_state,
        "dedup_reset_called": (primary or {}).get("dedup_reset_called"),
        "dedup_session_scope": (primary or {}).get("dedup_session_scope"),
        "wrapper_path": wrapper_path,
        "primary_receipt_path": primary_path,
        "fallback_receipt_path": fallback_path,
        "science_verdict_source": wrapper.get("science_verdict_source"),
        "parent_sha": (primary or {}).get("parent_sha"),
        "git_head_required": GIT_HEAD_REQUIRED,
        "identity_order_inertness_proven": identity_inert,
        "identity_inertness_wrapper_citation_path": IDENTITY_INERTNESS_WRAPPER_PATH,
        "primary_dense_fork_readable": compute_primary_dense_fork_readable(wrapper),
        "reversed_order_provenance_ok": _reversed_order_provenance_ok(primary) if isinstance(primary, dict) else False,
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }


run_root = Path(sys.argv[1])
wrapper_receipt_path = run_root / "prelaunch" / WRAPPER_RECEIPT_NAME
mechanism_receipt_path = run_root / "prelaunch" / MECHANISM_RECEIPT_NAME
wrapper_receipt_path.parent.mkdir(parents=True, exist_ok=True)

try:
    wrapper = orchestrate_ca_confirmation_with_fallback(run_root)
except Exception as exc:
    wrapper = {
        "schema": "hrm_text_158_ca_confirmation_wrapper_receipt/v1",
        "abnormal_exit": True,
        "abnormal_exit_error": f"{type(exc).__name__}: {exc}",
        "runs": {"primary": None},
        "science_verdict_source": "primary",
        "fallback_trigger": None,
        "primary_receipt_path": None,
        "primary_dense_fork_readable": False,
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }

if not isinstance(wrapper, dict):
    wrapper = {
        "schema": "hrm_text_158_ca_confirmation_wrapper_receipt/v1",
        "abnormal_exit": True,
        "runs": {"primary": None},
        "science_verdict_source": "primary",
        "fallback_trigger": None,
        "primary_receipt_path": None,
        "primary_dense_fork_readable": False,
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
    }

primary = (wrapper.get("runs") or {}).get("primary")
if isinstance(primary, dict):
    wrapper["primary_receipt_path"] = primary.get("receipt_path")
else:
    wrapper["primary_receipt_path"] = None

mechanism_receipt = build_mechanism_diagnosis_receipt(wrapper, run_root)
mechanism_receipt_path.write_text(
    json.dumps(mechanism_receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

wrapper["identity_inertness_wrapper_citation_path"] = IDENTITY_INERTNESS_WRAPPER_PATH
wrapper["identity_order_inertness_proven"] = bool(
    mechanism_receipt.get("identity_order_inertness_proven")
)
wrapper["primary_dense_fork_readable"] = bool(
    mechanism_receipt.get("primary_dense_fork_readable")
)
wrapper["f3b_terminal_branch"] = mechanism_receipt.get("f3b_terminal_branch")
wrapper["f3b_branch_inputs"] = mechanism_receipt.get("f3b_branch_inputs")
wrapper["mechanism_diagnosis_receipt_path"] = str(mechanism_receipt_path)
wrapper["ready_for_main_science"] = False
wrapper["counts_as_sub2"] = False
wrapper["pre_full_stack_diagnostic"] = True

wrapper_receipt_path.write_text(
    json.dumps(wrapper, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

science_receipt = (
    wrapper.get("runs", {}).get("fallback")
    if wrapper.get("science_verdict_source") == "fallback"
    else wrapper.get("runs", {}).get("primary")
)
if not isinstance(science_receipt, dict):
    science_receipt = {"infra_ok": False, "terminal_branch": "INFRA_NULL"}

print(json.dumps(wrapper, indent=2, sort_keys=True))
raise SystemExit(ca_confirmation_wrapper_exit_code(science_receipt))
"""

SCIENCE_BRANCHES = (
    "CA_PERSISTS",
    "CA_MIXED",
    "CA_DILUTES",
    "INSUFFICIENT_CB_STATES",
    "INFRA_NULL",
    "FEASIBILITY_SUBSAMPLE",
)

CONFIRMATION_GATES = [
    "observer_guard_clear",
    "tracemalloc_perturbed_false",
    "eligible_module_limit_eq_n_states",
    "s1d7_band_counter_mark_count_eq_sampled_state_count",
    "tracemalloc_mark_count_eq_0",
    "b_profile_mark_count_gt_0",
    "no_profile_env_mutual_exclusion_abort",
    "no_tracemalloc_perturbed_inconclusive",
    "infra_not_null",
]


def sha256_git_blob(commit: str, rel: str) -> str:
    data = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{rel}"])
    return hashlib.sha256(data).hexdigest()


def sha256_disk_file(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def pin_expected_sha256(rel: str, *, science_head: str = SCIENCE_HEAD) -> str:
    if rel in INFRA_PIN_RELS_ON_DISK:
        return sha256_disk_file(rel)
    return sha256_git_blob(science_head, rel)


def refresh_launch_pins() -> None:
    """Refresh infra tooling pins from on-disk bytes before packet regen."""
    for rel in INFRA_PIN_RELS_ON_DISK:
        PINS[rel] = sha256_disk_file(rel)


def refresh_science_pins_from_head() -> None:
    """Refresh science pins from git blob @ SCIENCE_HEAD before packet regen."""
    for rel in PINS:
        if rel in INFRA_PIN_RELS_ON_DISK:
            continue
        PINS[rel] = sha256_git_blob(SCIENCE_HEAD, rel)


def ca_confirmation_command() -> str:
    return (
        f"set -euo pipefail; cd {REPO}; RUN_ROOT={RUN_ROOT}; "
        f'mkdir -p "$RUN_ROOT/prelaunch" "$RUN_ROOT/callsite_band_counter_a" '
        f'"$RUN_ROOT/callsite_band_counter_b" "$RUN_ROOT/postrun"; '
        f'rm -f "$RUN_ROOT/run_nonce.txt" "$RUN_ROOT/exit_code.txt"; '
        f'RUN_NONCE="$(git -C {REPO} rev-parse HEAD)-$(date -u +%Y%m%dT%H%M%SZ)-$$"; '
        f'printf \'%s\\n\' "$RUN_NONCE" > "$RUN_ROOT/run_nonce.txt"; '
        f"export PYTHONPATH=.; "
        f"export {DENSE_SAMPLED_STATES_ENV}={DENSE_SAMPLED_STATES_VALUE}; "
        f"export {DENSE_ORDER_ENV}={DENSE_ORDER_VALUE}; "
        f'cleanup_lane_release() {{ PYTHONPATH=. python3 -c "from pathlib import Path; '
        f"from scripts.hrm_text_158_r7_resource_lane_release import release_resource_lane; "
        f'import json; print(json.dumps(release_resource_lane(Path(\'{RUN_ROOT}\'))))" '
        f'>>"$RUN_ROOT/postrun/lane_release_trap.log" 2>&1 || true; }}; '
        f"trap cleanup_lane_release EXIT ERR INT TERM; "
        f"timeout 3600 python3 - \"$RUN_ROOT\" <<'PY'\n"
        f"{CA_CONFIRMATION_HEREDOC}\n"
        "PY"
    )


def build_preflight_command() -> str:
    pin_lines = []
    for rel, expected in sorted(PINS.items()):
        pin_lines.append(f"    '{rel}': '{expected}',")
    pins_block = "\n".join(pin_lines)
    return (
        "set -euo pipefail; "
        f"cd {REPO}; "
        f"HEAD=$(git rev-parse HEAD); git merge-base --is-ancestor '{HEAD}' \"$HEAD\"; "
        f"PARENT={REPO}/calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt; "
        "PARENT_SHA=$(sha256sum \"$PARENT\" | awk '{print $1}'); "
        "test -f \"$PARENT\"; "
        "test \"$PARENT_SHA\" = '9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec'; "
        "PYTHONPATH=. python3 - <<'PY'\n"
        "import hashlib, subprocess, sys\n"
        "from pathlib import Path\n"
        f"REPO = Path('{REPO}')\n"
        "PINS = {\n"
        f"{pins_block}\n"
        "}\n"
        "for rel, expected in PINS.items():\n"
        "    actual = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()\n"
        "    if actual != expected:\n"
        "        print(f'PIN_MISMATCH {rel}: got {actual} expected {expected}', file=sys.stderr)\n"
        "        raise SystemExit(11)\n"
        "print('preflight_pins_ok')\n"
        "PY\n"
        f"BASELINE='{IDENTITY_INERTNESS_WRAPPER_PATH}'; "
        f'test -f "$BASELINE"; '
        f"PYTHONPATH=. python3 - <<'PY'\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"wrapper_path = Path('{IDENTITY_INERTNESS_WRAPPER_PATH}')\n"
        "if not wrapper_path.is_file():\n"
        "    print('IDENTITY_WRAPPER_MISSING', file=sys.stderr)\n"
        "    raise SystemExit(12)\n"
        "wrapper = json.loads(wrapper_path.read_text(encoding='utf-8'))\n"
        "if wrapper.get('identity_inertness_proven') is not True:\n"
        "    print('IDENTITY_PRECONDITION_NOT_PROVEN', file=sys.stderr)\n"
        "    raise SystemExit(12)\n"
        "print('preflight_identity_inertness_ok')\n"
        "PY\n"
        f"RUN_ROOT={RUN_ROOT}; mkdir -p \"$RUN_ROOT/prelaunch\"; "
        f"PYTHONPATH=. python3 scripts/box_lane_code_currency_preflight.py --chain-id {RUN_ID.lower()} "
        f"--head-expected {HEAD} --skip-fetch --allow-descendant-head "
        f"--include-phase3-obmalloc-surfaces "
        f'--output "$RUN_ROOT/prelaunch/box_code_currency_preflight.json"; '
        'python3 -c "import json;d=json.load(open(\'$RUN_ROOT/prelaunch/box_code_currency_preflight.json\')); '
        "assert d['code_currency_pass']; assert d.get('n_files',0)>=10\"; "
        f"cd {REPO}; PYTHONPATH=. python3 - <<'PY'\n"
        "import json\n"
        "from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (\n"
        "    dry_check_callsite_b_prime_b_arm_launch_composition,\n"
        ")\n"
        "receipt = dry_check_callsite_b_prime_b_arm_launch_composition()\n"
        "print(json.dumps({'launch_composition_dry_check_ok': receipt.get('ok')}, sort_keys=True))\n"
        "if not receipt.get('ok'):\n"
        "    raise SystemExit(43)\n"
        "PY"
    )


def verify_code_pins_against_commit(draft: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    head = str(draft.get("git_head_required", HEAD))
    code_pins = draft.get("code_pins", {})
    if code_pins.get("git_head") != head:
        failures.append("code_pins:git_head_mismatch")
    for field, rel in CODE_PIN_FIELD_TO_REL.items():
        if field not in code_pins:
            failures.append(f"code_pins:missing:{field}")
            continue
        pinned = str(code_pins[field])
        if field in INFRA_CODE_PIN_FIELDS_ON_DISK:
            actual = sha256_disk_file(rel)
        else:
            actual = sha256_git_blob(SCIENCE_HEAD, rel)
        if pinned != actual:
            failures.append(f"code_pins:mismatch:{field}")
    for key in code_pins:
        if key in {"git_head", "code_pins_note"}:
            continue
        if key not in CODE_PIN_FIELD_TO_REL:
            failures.append(f"code_pins:unmapped:{key}")
    if "test_slice5_v6i_oom_profile_attribution_sha256" in code_pins:
        failures.append("code_pins:stale_test_pin_present")
    return failures


def dry_check_launch_composition_command() -> str:
    return (
        f"set -euo pipefail; cd {REPO}; PYTHONPATH=. python3 - <<'PY'\n"
        "import json\n"
        "from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (\n"
        "    dry_check_callsite_b_prime_b_arm_launch_composition,\n"
        ")\n"
        "receipt = dry_check_callsite_b_prime_b_arm_launch_composition()\n"
        "print(json.dumps(receipt, indent=2, sort_keys=True))\n"
        "if not receipt.get('ok'):\n"
        "    raise SystemExit(43)\n"
        "PY"
    )


def build_band_counter_b_arm_env_toggles() -> dict[str, str]:
    return {
        "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
        "HRM_TEXT_158_PROFILE_HOST_RSS": "1",
        "HRM_TEXT_158_PROFILE_TRACEMALLOC": "0",
        "HRM_TEXT_158_PROFILE_S1D7_BAND_COUNTER_ONLY": "1",
        "HRM_TEXT_158_PROFILE_DEBUGMALLOCSTATS": "0",
        "HRM_TEXT_158_PROFILE_OBMALLOC_SITE_BRACKETS": "0",
        "HRM_TEXT_158_PROFILE_OBMALLOC_EXPANDED": "0",
        DENSE_SAMPLED_STATES_ENV: DENSE_SAMPLED_STATES_VALUE,
        DENSE_ORDER_ENV: DENSE_ORDER_VALUE,
        "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
        "HRM_TEXT_158_RUN_GPU_GLOBAL_RATE_CAP": "1",
        "HRM_TEXT_158_RUN_GPU_Q_ACC_APPLY": "1",
    }


def build_code_pins() -> dict[str, str]:
    pins = {"git_head": HEAD}
    for field, rel in CODE_PIN_FIELD_TO_REL.items():
        pins[field] = PINS[rel]
    return pins


def _update_pin_blocks(draft: dict[str, Any]) -> None:
    for rel, pin in PINS.items():
        key = rel.split("/")[-1].replace(".py", "")
        if key == "hrm_text_158_slice5_v6i_oom_profile_attribution":
            key = "attribution_script"
        elif key == "hrm_text_158_bounded_delta_acquisition_probe":
            key = "bounded_delta_acquisition_probe"
        elif key == "hrm_text_158_bounded_delta_acquisition_probe_bootstrap":
            key = "probe_bootstrap"
        elif key == "hrm_text_158_code_currency_guard":
            key = "code_currency_guard"
        elif key == "hrm_text_158_r7_resource_lane_acquire":
            key = "r7_resource_lane_acquire"
        elif key == "event_coded_acc_live_carrier":
            key = "event_coded_acc_live_carrier"
        elif key == "event_coded_vote_update_adapter":
            key = "event_coded_vote_update_adapter"
        elif key == "sparse_cap_gpu_seam_adapter":
            key = "sparse_cap_gpu_seam_adapter"
        elif key == "host_tracemalloc_probe":
            key = "host_tracemalloc_probe"
        elif key == "s1d7_band_counter":
            key = "s1d7_band_counter"
        elif key == "s1d7_tracemalloc_feasibility":
            key = "s1d7_tracemalloc_feasibility"
        elif key == "bounded_delta_learner":
            key = "bounded_delta_learner"
        elif key == "f3b_why_state0_branch":
            key = "f3b_why_state0_branch"
        elif key == "box_lane_code_currency_preflight":
            key = "box_lane_code_currency_preflight"
        elif key == "box_lane":
            key = "box_lane"
        if "box_preflight_role_pins" in draft and key in draft["box_preflight_role_pins"]:
            draft["box_preflight_role_pins"][key]["sha256"] = pin
            draft["box_preflight_role_pins"][key]["rel_path"] = rel
    draft["code_pins"] = build_code_pins()
    draft["code_pins_note"] = (
        "test_slice5 omitted (not launch-executed). git_head_required=bd23cc9 is the "
        "order-control patch science baseline (FDC); descendant HEAD allowed via merge-base "
        "ancestor check. Science code_pins verified @ git_head_required via git show. "
        "Launch-executed infra pins (on-disk): box_lane.py implements descendant-head "
        "accept decision; box_lane_code_currency_preflight.py imports box_lane at module "
        "load."
    )


def build_execution_order() -> list[str]:
    return [
        "dispatch_run_claim",
        "preflight",
        "parent_checkpoint_rehash_before",
        "install_lane_release_trap_cleanup_only",
        "ca_confirmation_primary_or_fallback",
        "parent_checkpoint_rehash_after",
        "resource_lane_release_cleanup_only",
    ]


def build_phase_budgets_and_watcher() -> dict[str, Any]:
    return {
        "convention": (
            "fold-2b confirmation budgets: max_silent(900) < phase(1800) < total(3600); "
            "milestone budgets report-only"
        ),
        "heartbeat_interval_seconds": 30,
        "heartbeat_supplementary_only": True,
        "interrupt_authority": {
            "first_milestone_budgets_report_only": True,
            "interrupt_authority": "faulthandler_silent_phase_guard",
            "interrupt_timeout_seconds": 900.0,
            "milestone_budget_breach_triggers_interrupt": False,
            "schema": "hrm_text_158_phase_budget_interrupt_authority/v1",
        },
        "liveness_contract": {
            "coherence": "max_silent(900) < phase(1800) < total(3600)",
            "max_silent_phase_seconds": 900,
            "per_arm_subprocess_timeout_seconds": 1800,
            "phase_heartbeat_seconds": 30,
            "phase_timeout_seconds": 1800,
            "total_timeout_seconds": 3600,
            "watcher_stall_seconds": 900,
        },
        "max_silent_phase_seconds": 900,
        "phase_timeout_seconds": 1800,
        "stop_conditions": [
            "executed guard receipt missing or guard_ran_before_pinned_imports false",
            "ca_confirmation subprocess exit 37 (infra_null / currency fail-closed)",
            "ca_confirmation subprocess exit 42 (unknown terminal_branch)",
            "no phase_heartbeat within watcher_stall_seconds(900)",
            "parent checkpoint sha256 mismatch pre/post",
            "peak_rss_gib > 6.5 on primary n=32 triggers n=16 FEASIBILITY_SUBSAMPLE fallback",
            "primary runs.B.subprocess_timeout_expired triggers n=16 FEASIBILITY_SUBSAMPLE fallback",
            "primary exit_code 37/-6 MUST NOT trigger fallback (fail-closed INFRA_NULL)",
        ],
        "total_timeout_seconds": 3600,
        "watcher_liveness_fail_regex": (
            "LIVENESS_FAIL_KERNELIZED_BUT_STALLED|phase_milestone_stall|"
            "LIVENESS_FAIL_TOTAL_TIMEOUT|total_timeout|LIVENESS_FAILURE|LIVENESS_FAIL"
        ),
    }


def build_code_pin_coverage() -> dict[str, Any]:
    executed_surfaces = sorted(PINS.keys())
    return {
        "box_preflight": {
            "note": "box attests file bytes on disk; executed guard attests imported bytecode",
            "surfaces": [
                "DEFAULT_FLOOR_PINNED_FILES + PHASE3_OBMALLOC_SURFACE_PINNED_FILES (10 roles)"
            ],
            "via": "box_lane_code_currency_preflight.py --include-phase3-obmalloc-surfaces",
        },
        "executed_code_guard": {
            "surfaces": executed_surfaces,
            "via": (
                "PHASE3B_PINNED_SOURCE_FILES + bootstrap executed guard "
                "(test_slice5 intentionally omitted — not launch-executed)"
            ),
        },
    }


def build_f3b_mechanism_acceptance_contract() -> dict[str, Any]:
    return {
        "field": "f3b_terminal_branch",
        "classifier": "F3B_WHY_STATE0_BRANCH_V1",
        "variable_id": VARIABLE_ID,
        "control_reason": CONTROL_REASON,
        "identity_inertness_wrapper_citation_path": IDENTITY_INERTNESS_WRAPPER_PATH,
        "true_iff_all": [
            "identity_order_inertness_proven == true (accepted identity wrapper citation)",
            "primary_dense_fork_readable == true (Q4 salvage on completed n=32 primary)",
            "sampled_state_set == [0,1,2,3,4,5,6,7,8,9]",
            "sampled_state_order == [9,8,7,6,5,4,3,2,1,0]",
            "order_control_active == true on reversed primary receipt",
            "order_perturbation_kind == sampled_block_order_perturbation",
            "effective_visit_order == [9..0]+[10..31] (reversed sampled block + numeric tail)",
            "order_rank_by_semantic_state[str(9-i)] == i for i in 0..9",
            "semantic_state_id == state_index per per_state row over [0..9]",
            "dedup_reset_called == true AND dedup_session_scope valid (probe_subprocess)",
            "classify_f3b_why_state0_branch(f3b_branch_inputs).terminal_branch emitted",
        ],
        "decision_rule": {
            "F3B_STATE0_IDENTITY_STRUCTURE": "semantic state0 remains sole crossing-bearing state under reversed order",
            "F3B_MEASUREMENT_ORDER_ARTIFACT": "first-measured semantic state (9 under reversed order) becomes CB",
        },
        "decisive_branches": sorted(F3B_DECISIVE_BRANCHES),
        "non_decisive_branches": [
            "F3B_NO_VERDICT_OPERATIONAL",
            "F3B_NO_VERDICT_SCHEMA",
            "F3B_SAMPLE_SET_OR_ELIGIBILITY_ARTIFACT",
            "F3B_MARKING_OR_DEDUP_ARTIFACT",
            "F3B_MIXED_OR_INCONCLUSIVE",
        ],
        "n50_equivalent_semantics": (
            "DETERMINISTIC: this n=32 dense reversed primary IS the N=50-equivalent verdict arm "
            "(single deterministic receipt; no stochastic repeats)"
        ),
        "n20_screen_policy": (
            "N=20 screen deliberately NOT bundled — identity-inertness run 1783255165644 already "
            "exercised n=32 dense CA machinery end-to-end CLEAN; reversed run changes only visit "
            "sequence. Explicit N=20 screen tier remains available via separate dispatch if liveness "
            "surprise occurs."
        ),
        "fold3b_budget": "counts 1 launch toward Fold-3B 4-launch/140-unit budget",
        "negative_gate": (
            "MUST NOT use compute_identity_inertness_proven or baseline element-wise crossing "
            "comparison as science gate — that would false-reject measurement-order artifacts"
        ),
        "when_false": "Variable B remains deferred",
    }


def build_primary_dense_fork_readable_contract() -> dict[str, Any]:
    return {
        "field": "primary_dense_fork_readable",
        "true_iff_all": [
            "wrapper.runs.primary.sampled_states == [0,1,2,3,4,5,6,7,8,9]",
            "wrapper.runs.primary.s1d7_band_counter_mark_count == 10",
            "wrapper.runs.primary.checks.s1d7_band_counter_mark_count_eq_sampled_state_count == true",
            "wrapper.runs.primary.per_state covers state_index 0..9 (all ten rows)",
            "wrapper.runs.primary.runs.B.exit_code == 0",
            "wrapper.runs.primary.runs.B.subprocess_timeout_expired == false",
            "wrapper.runs.primary.infra_ok == true",
        ],
        "when_true": (
            "Read structural-vs-sampling fork from wrapper.runs.primary.per_state EVEN IF "
            "fallback_trigger.rss_breach==true and science_verdict_source==fallback. "
            "Completed primary with peak_rss>6.5 is fork-readable; NOT mere FEASIBILITY_SUBSAMPLE."
        ),
        "when_false": (
            "Primary is partial telemetry only; dense n=16 fallback is the science source. "
            "If fallback also lacks full dense coverage or clean B-arm exit → "
            "feasibility/inconclusive."
        ),
    }


def build_ca_branch_outcomes() -> dict[str, str]:
    base = {
        "CA_DILUTES": (
            "crossing-bearing W/P partition: neither crossing-weighted nor per-state "
            "(C+A) share meets ca_share_min — informative null; fork read from per_state "
            "when primary_dense_fork_readable"
        ),
        "CA_MIXED": (
            "crossing-weighted (C+A) share ok XOR per-state ok — mixed persistence signal; "
            "fork read from per_state when primary_dense_fork_readable"
        ),
        "CA_PERSISTS": (
            "both crossing-weighted and per-state (C+A) shares meet ca_share_min — "
            "fold-1 precondition ONLY; NOT reduction eligibility"
        ),
        "FEASIBILITY_SUBSAMPLE": (
            f"primary n=32 dense RSS>{RSS_FALLBACK_GIB} GiB or runs.B.subprocess_timeout_expired "
            f"→ rerun n={FALLBACK_N_STATES} dense-[0..9]; exit 37/-6 NEVER fallback; "
            "terminal_branch via classify(feasibility_subsample=True) ONLY when "
            "primary_dense_fork_readable==false"
        ),
        "INFRA_NULL": (
            "infra_ok false or guard/observer fail-closed — no science branch; abnormal B-arm "
            "exit fail-closed"
        ),
        "INSUFFICIENT_CB_STATES": (
            "fewer than 2 crossing-bearing states — informative null; if only state 0 CB → "
            "structural (state0-only); fork read from per_state when primary_dense_fork_readable"
        ),
        "PRIMARY_DENSE_FORK_READABLE": (
            "completed primary dense-[0..9] with full mark coverage and clean B-arm exit — "
            "fork evidence authoritative even when RSS fallback fires post-primary"
        ),
    }
    return base


def build_proof_artifacts() -> dict[str, str]:
    confirmation_root = (
        f"{RUN_ROOT}/prelaunch/callsite_band_counter_ca_confirmation"
    )
    primary_receipt = (
        f"{confirmation_root}/callsite_band_counter_ca_confirmation_receipt.json"
    )
    fallback_receipt = (
        f"{RUN_ROOT}/feasibility_subsample_n16/prelaunch/"
        "callsite_band_counter_ca_confirmation/"
        "callsite_band_counter_ca_confirmation_receipt.json"
    )
    return {
        "ca_confirmation_receipt": primary_receipt,
        "ca_confirmation_wrapper_receipt": (
            f"{RUN_ROOT}/prelaunch/ca_confirmation_wrapper_receipt.json"
        ),
        "b_host_rss_profile": f"{confirmation_root}/callsite_band_counter_b/host_rss_profile.jsonl",
        "b_probe_stream_log": f"{confirmation_root}/callsite_band_counter_b/probe_stream.log",
        "exit_code_monitor_optional": (
            "NOT written by ca_confirmation (command rm -f only); monitor/watcher may "
            "optionally record shell exit — terminal acceptance is f3b_terminal_branch based"
        ),
        "lane_release_trap_log": f"{RUN_ROOT}/postrun/lane_release_trap.log",
        "lane_release_trap_note": (
            "CLEANUP-ONLY; no_holding_file at RUN_ROOT is expected/normal — NOT protection proof"
        ),
        "primary_runs_a_resource_lane_holding": (
            f"{primary_receipt}#runs.A.resource_lane_holding"
        ),
        "primary_runs_a_resource_lane_release": (
            f"{primary_receipt}#runs.A.resource_lane_release"
        ),
        "primary_runs_b_resource_lane_holding": (
            f"{primary_receipt}#runs.B.resource_lane_holding"
        ),
        "primary_runs_b_resource_lane_release": (
            f"{primary_receipt}#runs.B.resource_lane_release"
        ),
        "fallback_runs_b_resource_lane_holding": (
            f"{fallback_receipt}#runs.B.resource_lane_holding"
        ),
        "fallback_runs_b_resource_lane_release": (
            f"{fallback_receipt}#runs.B.resource_lane_release"
        ),
        "terminal_lane_free_witness": (
            "ai_room_resource_lane_status(lane=gpu:hrm-text-158) held=false — "
            "test-operator supplies at terminal receipt; NOT written by ca_confirmation"
        ),
        "prelaunch_code_currency": f"{RUN_ROOT}/prelaunch/box_code_currency_preflight.json",
        "feasibility_subsample_receipt": fallback_receipt,
        "primary_receipt_path": primary_receipt,
        "primary_per_state": f"{primary_receipt}#per_state",
        "fallback_trigger": (
            f"{RUN_ROOT}/prelaunch/ca_confirmation_wrapper_receipt.json#fallback_trigger"
        ),
        "primary_dense_fork_readable": (
            f"{RUN_ROOT}/prelaunch/ca_confirmation_wrapper_receipt.json#primary_dense_fork_readable"
        ),
        "f3b_mechanism_diagnosis_receipt": (
            f"{RUN_ROOT}/prelaunch/{MECHANISM_RECEIPT_NAME}"
        ),
        "f3b_terminal_branch": (
            f"{RUN_ROOT}/prelaunch/{MECHANISM_RECEIPT_NAME}#f3b_terminal_branch"
        ),
    }


def build_consumption_matrix() -> dict[str, Any]:
    return {
        "A_freshness_guard": {
            "bootstrap_launch": (
                "python3 -B probe_bootstrap.py entry; band_counter_only env; "
                "guard runs before pinned imports"
            ),
            "box_preflight": (
                "box_lane_code_currency_preflight.py --include-phase3-obmalloc-surfaces"
            ),
            "guard_entry": (
                "bootstrap.main → run_phase3b_probe_executed_code_currency_guard BEFORE "
                "bounded_delta_acquisition_probe import (band_counter_only)"
            ),
            "pycache_invalidation": (
                "prepare_phase3b_band_counter_only_launch_env + guard-internal invalidation"
            ),
        },
        "B_confirmation_receipt": {
            "builder": "run_callsite_band_counter_ca_confirmation + classify_ca_band_counter_confirmation",
            "science_gate": "f3b_terminal_branch via F3B_WHY_STATE0_BRANCH_V1 (NOT receipt.ok / infra_ok / identity_inertness_proven)",
            "fork_read_gate": "primary_dense_fork_readable (completed primary salvage)",
            "fallback": (
                f"POST-primary RSS>{RSS_FALLBACK_GIB} GiB or runs.B.subprocess_timeout_expired on "
                f"n=32 dense → n={FALLBACK_N_STATES} dense-[0..9]; exit 37/-6 fail-closed (no fallback)"
            ),
            "launch_wrapper_reporting": (
                "wrapper receipt MUST surface runs.primary.per_state, primary_receipt_path, "
                "fallback_trigger, primary_dense_fork_readable; per-fixture lane "
                "acquire/release inside slice5 is the sole lock boundary (no outer RUN_ROOT acquire)"
            ),
        },
    }


def build_monitor_section() -> dict[str, Any]:
    confirmation_root = (
        f"{RUN_ROOT}/prelaunch/callsite_band_counter_ca_confirmation"
    )
    return {
        "primary_tail_log": f"{confirmation_root}/callsite_band_counter_b/probe_stream.log",
        "probe_stream_log_paths": [
            f"{confirmation_root}/callsite_band_counter_a/probe_stream.log",
            f"{confirmation_root}/callsite_band_counter_b/probe_stream.log",
        ],
        "run_root": RUN_ROOT,
        "scratch_root": f"{confirmation_root}/callsite_band_counter_b",
    }


STALE_RENDER_PATTERNS: tuple[tuple[str, str], ...] = (
    (OLD_CALLSITE_RUN_ID, "stale_old_run_root"),
    ("callsite_band_counter_scale_smoke", "stale_scale_smoke_flow"),
    ("fixture_callsite_b_prime_gpu_run", "stale_b_prime_fixture_flow"),
    ("classifier_extract_command", "stale_classifier_extract"),
    ("executed_guard_receipt_audit_command", "stale_guard_audit_command"),
    ("terminal_currency_check_command", "stale_terminal_currency_check"),
    ("postrun_receipt_aggregate_command", "stale_postrun_aggregate"),
    ("S1D7_CALL_SITE_CANDIDATE", "stale_candidate_c_branch"),
    ("S1D3_INT32_LANES_CONFIRMED", "stale_s1d3_branch"),
    ("S1F1_CONFIRMED", "stale_s1f1_branch"),
    ("RL-S1d", "stale_reduction_route_s1d"),
    ("RL-S1f", "stale_reduction_route_s1f"),
    ("test_slice5_v6i_oom_profile_attribution_v1.py", "stale_test_in_coverage"),
    ("9977166", "stale_rev14_head"),
    ("c0371f18331059df0f91d793d119a0cf8010ec54", "stale_pre_fold2d_head"),
    ("s1d7_band_counter_mark_count_eq_n_states", "stale_mark_count_eq_n_states_gate"),
    ('"phase_timeout_seconds": 2280', "stale_rev14_phase_budget"),
    ('"total_timeout_seconds": 5400', "stale_rev14_total_budget"),
    ('"max_silent_phase_seconds": 600', "stale_rev14_max_silent"),
    ('"interrupt_timeout_seconds": 600.0', "stale_rev14_interrupt"),
    (f"({N_STATES} sampled states)", "stale_n_states_as_sampled_count_wording"),
    (f'"new_mark_events": {N_STATES}', "stale_new_mark_events_eq_n_states"),
    ("4-point sample", "stale_4point_sample_wording"),
    ("{0, n//3, 2n//3, n-1}", "stale_default_sampler_rule"),
)


def build_process_exit_precedence() -> dict[str, Any]:
    return {
        "ca_confirmation_exits": {
            "37": "infra_null / currency fail-closed / guard not proven",
            "42": "unknown terminal_branch (not a valid CA confirmation science branch)",
        },
        "description": (
            "fold-2b CA confirmation fail-closed exits; science read from terminal_branch "
            "after infra_ok"
        ),
        "subprocess_pre_measurement": {
            "enforced_by": (
                "probe_bootstrap.py entry → maybe_enforce before probe import (band_counter_only)"
            ),
            "exit_code": 37,
            "order": 1,
            "rule": (
                "executed-code currency guard MUST fail-close exit 37 BEFORE phase telemetry "
                "if freshness cannot be proven"
            ),
            "terminals": [
                "CODE_CURRENCY_GUARD_NOT_RUN_INCONCLUSIVE",
                "CODE_CURRENCY_EXECUTED_MISMATCH_INCONCLUSIVE",
                "CODE_CURRENCY_MISMATCH_INCONCLUSIVE",
            ],
        },
    }


def build_forbidden() -> list[str]:
    return [
        "mutate banked .pt checkpoint",
        "pass --mirror-durable-attribution at launch",
        "v6i-unpark",
        "claim reduction implementation eligibility from this confirmation run",
        "stage/commit/push from test-operator",
        f"edit code beyond {HEAD[:7]} pins",
        "launch conflated profiler envs (TRACEMALLOC=1 on band-counter-only subprocess)",
        "set DEBUGMALLOCSTATS and TRACEMALLOC on same subprocess",
        "treat FEASIBILITY_SUBSAMPLE as reduction eligibility",
        "legacy callsite classifier or candidate-band routing from this packet",
    ]


def scrub_stale_rev14_fields(draft: dict[str, Any]) -> None:
    for key in (
        "postrun_receipt_aggregate_command",
        "launch_robustness_at_18505c4",
        "perturbation_rules",
        "consumption_assertions",
        "probe_argv_b_arm",
        "dispatch_msg_id",
        "dispatch_msg_id_authority",
        "dispatch_msg_id_launch_substitution",
    ):
        draft.pop(key, None)


def _build_band_counter_enable_contract(n_states: int) -> dict[str, Any]:
    sampled = compute_expected_sampled_states(n_states)
    expected_marks = len(sampled)
    return {
        "schema": "hrm_text_158_s1d7_band_counter_site/v1",
        "eligible_module_limit": n_states,
        "sampled_states_rule": SAMPLED_STATES_RULE,
        "sampled_states": sorted(sampled),
        "expected_sampled_state_count": expected_marks,
        "expected_mark_events": expected_marks,
        "events": (
            f"s1d7_band_counter_site_C4.S1d.7_post "
            f"({expected_marks} sampled states @ eligible_module_limit={n_states})"
        ),
        "env": (
            "HRM_TEXT_158_PROFILE_S1D7_BAND_COUNTER_ONLY=1 + HOST_RSS=1 + TRACEMALLOC=0 + "
            f"{DENSE_SAMPLED_STATES_ENV}={DENSE_SAMPLED_STATES_VALUE} + "
            f"{DENSE_ORDER_ENV}={DENSE_ORDER_VALUE}"
        ),
        "new_mark_events": expected_marks,
        "tracemalloc_site_events": 0,
        "measurement_contract": "static_pre_append_v1",
        "n_states": n_states,
    }


def verify_whole_render_sweep(draft: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    blob = json.dumps({"draft": draft, "replay": replay}, sort_keys=True)
    for pattern, label in STALE_RENDER_PATTERNS:
        if pattern in blob:
            failures.append(f"whole_render:{label}")
    coverage = draft.get("code_pin_coverage", {}).get("executed_code_guard", {}).get(
        "surfaces", []
    )
    if "calm/llm_computer/tests/test_slice5_v6i_oom_profile_attribution_v1.py" in coverage:
        failures.append("whole_render:test_in_executed_code_guard_surfaces")
    if draft.get("execution_order") != build_execution_order():
        failures.append("whole_render:execution_order_mismatch")
    pbw = draft.get("phase_budgets_and_watcher", {})
    if pbw.get("total_timeout_seconds") != 3600:
        failures.append("whole_render:phase_budget_total_not_3600")
    if pbw.get("phase_timeout_seconds") != 1800:
        failures.append("whole_render:phase_budget_phase_not_1800")
    liveness = pbw.get("liveness_contract", {})
    if liveness.get("max_silent_phase_seconds") != 900:
        failures.append("whole_render:liveness_max_silent_not_900")
    outcomes = draft.get("proof", {}).get("branch_outcomes", {})
    for stale_key in (
        "S1D7_CALL_SITE_CANDIDATE_A_CROSSING_INDICES",
        "S1D3_INT32_LANES_CONFIRMED",
        "S1F1_CONFIRMED",
    ):
        if stale_key in outcomes:
            failures.append(f"whole_render:stale_branch_outcome:{stale_key}")
    for required in SCIENCE_BRANCHES:
        if required not in outcomes:
            failures.append(f"whole_render:missing_branch_outcome:{required}")
    if "classifier_extract_command" in draft.get("proof", {}):
        failures.append("whole_render:proof_still_has_classifier_extract")
    bce = draft.get("child_emission_contract", {}).get("band_counter_enable", {})
    eligible = int(bce.get("eligible_module_limit", bce.get("n_states", N_STATES)) or N_STATES)
    expected_marks = expected_mark_event_count(eligible)
    if int(bce.get("new_mark_events", -1)) != expected_marks:
        failures.append("whole_render:new_mark_events_not_sampled_count")
    if int(bce.get("expected_mark_events", -1)) != expected_marks:
        failures.append("whole_render:expected_mark_events_not_sampled_count")
    if bce.get("new_mark_events") == eligible and eligible != expected_marks:
        failures.append("whole_render:new_mark_events_conflates_n_states")
    return failures


def build_draft() -> dict[str, Any]:
    draft = copy.deepcopy(json.loads(REV14_DRAFT.read_text(encoding="utf-8")))
    draft["packet_revision"] = PACKET_REVISION
    draft["git_head_required"] = HEAD
    draft["schema_version"] = (
        "hrm_text_158_c4s1_n32_dense_09_variable_a_rev_launch_packet/v1"
    )
    draft["run_id"] = RUN_ID
    draft["task_id"] = ACTIVE_TASK_ID
    draft["replay_commands_artifact"] = str(REPLAY.relative_to(REPO)).replace("\\", "/")
    draft["run_root_template"] = "/home/gabe/hrm158_c4s1_phase3_gpu_gate/{run_id}"
    draft["design_binding"] = {
        "fold1_spec_sha256": hashlib.sha256((REPO / FOLD1_SPEC).read_bytes()).hexdigest(),
        "fold2_design_msg_id": "1783186409528",
        "fold2a_source_commit": SCIENCE_HEAD,
        "chosen_option": "n32_dense_09_variable_a_reversed_order_mechanism_diagnosis",
        "prereg_packet_path": PREREG_PACKET_PATH,
        "identity_inertness_terminal_receipt_msg_id": "1783255165644",
        "identity_inertness_co_lead_pass_msg_id": "1783255317949",
    }
    draft["acceptance_gate_order"] = [
        "CURRENCY_OK (executed guard + box preflight)",
        (
            "INFRA_OK (eligible_module_limit==n_states, mark_count==len(sampled_states), "
            "observer clear, tracemalloc_mark_count==0)"
        ),
        "SCIENCE_BRANCH (terminal_branch classifier — NOT receipt.ok)",
        "CA_PERSISTS satisfies precondition ONLY — NOT reduction eligibility",
    ]
    draft["decision_contract"] = {
        "chosen_path": (
            f"Fold-3B Variable A order-only perturbation via "
            "run_callsite_band_counter_ca_confirmation @ bd23cc9 n_states=32; "
            "HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATES=0..9 + "
            f"HRM_TEXT_158_OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER={DENSE_ORDER_VALUE}; "
            "classifier F3B_WHY_STATE0_BRANCH_V1 mechanism verdict"
        ),
        "forbidden": [
            "direct probe.py launch without bootstrap for band_counter_only B-arm",
            "TRACEMALLOC=1 on band_counter_only confirmation subprocess",
            "reading receipt.ok or infra_ok or command exit as f3b_terminal_branch",
            "using compute_identity_inertness_proven as science gate",
            "candidate-C single-band resolution claim from this run",
            "outer RUN_ROOT lane acquire (double-acquire deadlock)",
        ],
        "forbidden_at_launch": [
            "launch without box code-currency preflight",
            "launch without identity_inertness_wrapper preflight check",
            "launch with claude pre-holding gpu:hrm-text-158 lane (per-fixture self-acquire)",
            "direct probe.py launch without bootstrap for band_counter_only B-arm",
            "TRACEMALLOC=1 band_counter_only subprocess",
            "outer r7_resource_lane_acquire at RUN_ROOT",
        ],
    }
    draft["bounded_steps_budget"] = {
        "fixture_a_arm_subprocess_timeout_seconds": 900,
        "fixture_b_arm_subprocess_timeout_seconds": 1800,
        "liveness_coherence": "max_silent(900) < phase(1800) < total(3600)",
        "max_silent_phase_seconds": 900,
        "max_steps_hard": 1,
        "phase_heartbeat_seconds": 30,
        "phase_timeout_seconds": 1800,
        "total_timeout_seconds": 3600,
    }
    draft["child_emission_contract"] = {
        "env_toggles_b_arm": build_band_counter_b_arm_env_toggles(),
        "env_source": (
            "B-prime band-counter-only: tracemalloc=False, band_counter_only=True; "
            "bootstrap -B guard-ordering; dense sampled_states via shell export + env_toggles"
        ),
        "band_counter_enable": _build_band_counter_enable_contract(N_STATES),
        "line_split_surfaces": {
            "tracemalloc_classifier_physical_bracket": "[909,955]",
            "tracemalloc_classifier_candidate_bands": "A910 / C941-952 / E914-917",
            "band_counter_logical_marker": "event_coded_acc_live_carrier.py:895",
            "band_counter_logical_candidate": "event_coded_acc_live_carrier.py:896",
        },
    }
    draft["ca_confirmation_objective"] = {
        "science_question": (
            "Fold-3B Variable A: under reversed ORDER=[9..0] with unchanged SET=[0..9], "
            "does crossing-bearing support follow semantic state0 identity or measurement order?"
        ),
        "dense_sampled_states": DENSE_SAMPLED_STATES_LIST,
        "dense_sampled_state_order": DENSE_ORDER_LIST,
        "dense_env_override": DENSE_SAMPLED_STATES_VALUE,
        "dense_order_env_override": DENSE_ORDER_VALUE,
        "identity_inertness_wrapper_citation_path": IDENTITY_INERTNESS_WRAPPER_PATH,
        "prereg_packet_path": PREREG_PACKET_PATH,
        "variable_id": VARIABLE_ID,
        "control_reason": CONTROL_REASON,
        "cb_share_min": 0.80,
        "crossing_bearing_predicate": "crossing_indices_len > 0",
        "terminal_branches": list(SCIENCE_BRANCHES),
        "science_gate_field": "f3b_terminal_branch",
        "fork_read_gate_field": "primary_dense_fork_readable",
        "not_science_gate_fields": ["ok", "infra_ok", "terminal_branch", "identity_inertness_proven"],
        "feasibility_subsample_trigger": {
            "peak_rss_gib_gt": RSS_FALLBACK_GIB,
            "or_subprocess_timeout_expired": True,
            "exclude_exit_codes": [37, -6],
            "fallback_n_states": FALLBACK_N_STATES,
            "fallback_dense_env": DENSE_SAMPLED_STATES_VALUE,
            "fallback_dense_order_env": DENSE_ORDER_VALUE,
            "post_primary_only": True,
            "executable_via": (
                "packet wrapper: evaluate_ca_confirmation_fallback_trigger on COMPLETED "
                "primary receipt (slice5:6634-6652); NOT in-run interrupt; "
                "primary_dense_fork_readable salvage when primary completes cleanly"
            ),
        },
        "primary_dense_fork_readable": build_primary_dense_fork_readable_contract(),
        "f3b_mechanism_acceptance": build_f3b_mechanism_acceptance_contract(),
        "diagnostic_flags": {
            "ready_for_main_science": False,
            "counts_as_sub2": False,
            "pre_full_stack_diagnostic": True,
            "diagnostic_reason": (
                "Fold-3B Variable A mechanism diagnosis — order-only perturbation decider"
            ),
            "inside_fold3b_budget": True,
            "n20_screen_bundled": False,
        },
        "anti_overclaim": (
            "Within the Fold-3B packet scope, state0-only crossing support classifies as one "
            "of the pre-registered branches. FORBIDDEN: candidate-C, CA/reduction eligibility, "
            "W/P, ~430MB bank pin, universal all-state census, bank mutation, sub-2 readiness, "
            "full-stack readiness, implementation readiness"
        ),
    }
    draft["perturbation_risk_classification"] = {
        "ca_confirmation_mandatory": {
            "command": "proof.ca_confirmation_command",
            "exit_code_fail_infra": 37,
            "exit_code_fail_unknown_branch": 42,
            "gates": CONFIRMATION_GATES,
            "science_branches": list(SCIENCE_BRANCHES),
            "science_gate": "f3b_terminal_branch",
            "ordering": (
                "PRIMARY Variable A reversed-order run; acceptance read from "
                "f3b_terminal_branch via mechanism diagnosis receipt NOT ok/exit"
            ),
        },
    }
    draft.pop("banked_reconcile_provenance", None)
    for stale_key in (
        "callsite_acceptance_objective",
        "event_total_envelope_shift",
        "multi_pair_event_watch",
        "phase3b_instrumentation_at_77c5a5a",
        "packet_impl_delta_at_plus1",
        "upstream_chain",
    ):
        draft.pop(stale_key, None)
    scrub_stale_rev14_fields(draft)
    draft["packet_id"] = RUN_ID.lower()
    draft["forbidden"] = build_forbidden()
    draft["execution_order"] = build_execution_order()
    draft["phase_budgets_and_watcher"] = build_phase_budgets_and_watcher()
    draft["code_pin_coverage"] = build_code_pin_coverage()
    draft["consumption_matrix"] = build_consumption_matrix()
    monitor = build_monitor_section()
    draft["monitor"] = monitor
    draft["probe_stream_log_paths"] = list(monitor["probe_stream_log_paths"])
    draft["process_exit_precedence"] = build_process_exit_precedence()
    proof = {}
    proof["ca_confirmation_command"] = ca_confirmation_command()
    proof["primary_command"] = ca_confirmation_command()
    proof["callsite_currency_dry_check_command"] = dry_check_launch_composition_command()
    proof["artifacts"] = build_proof_artifacts()
    proof["branch_outcomes"] = build_ca_branch_outcomes()
    proof["description"] = (
        f"Fold-3B Variable A reversed-order mechanism diagnosis via "
        "orchestrate_ca_confirmation_with_fallback import heredoc; "
        f"ORDER={DENSE_ORDER_VALUE} + SET={DENSE_SAMPLED_STATES_VALUE}; "
        f"POST-primary RSS>{RSS_FALLBACK_GIB} GiB or timeout triggers n={FALLBACK_N_STATES} "
        "reversed fallback; f3b_terminal_branch via F3B_WHY_STATE0_BRANCH_V1 classifier"
    )
    proof["interpretation_rule"] = (
        "Acceptance: parse f3b_terminal_branch from fold3b_variable_a_mechanism_diagnosis_receipt "
        "via classify_f3b_why_state0_branch — NOT ok/infra_ok/command exit/identity_inertness_proven. "
        "Decisive branches: F3B_STATE0_IDENTITY_STRUCTURE or F3B_MEASUREMENT_ORDER_ARTIFACT. "
        "Reversed order provenance checked for CORRECTNESS on Variable A run. "
        "primary_dense_fork_readable salvage applies under Q4 RSS breach. "
        "identity_order_inertness_proven precondition cites accepted identity wrapper. "
        "Partial/abnormal exit fail-closed. Launch wrapper MUST surface f3b_terminal_branch, "
        "mechanism_diagnosis_receipt_path, runs.primary.per_state, primary_receipt_path, "
        "fallback_trigger, primary_dense_fork_readable, identity_inertness_wrapper_citation_path, "
        "ready_for_main_science=false, counts_as_sub2=false, pre_full_stack_diagnostic=true."
    )
    proof["primary_dense_fork_readable"] = build_primary_dense_fork_readable_contract()
    proof["f3b_mechanism_acceptance"] = build_f3b_mechanism_acceptance_contract()
    proof["identity_inertness_wrapper_citation_path"] = IDENTITY_INERTNESS_WRAPPER_PATH
    proof["prereg_packet_path"] = PREREG_PACKET_PATH
    proof["launch_wrapper_receipt_required_fields"] = [
        "runs.primary.per_state",
        "primary_receipt_path",
        "mechanism_diagnosis_receipt_path",
        "f3b_terminal_branch",
        "f3b_branch_inputs",
        "identity_inertness_wrapper_citation_path",
        "identity_order_inertness_proven",
        "fallback_trigger",
        "primary_dense_fork_readable",
        "science_verdict_source",
        "ready_for_main_science",
        "counts_as_sub2",
        "pre_full_stack_diagnostic",
    ]
    proof["pass_criteria"] = {
        "identity_inertness_preflight_exists": (
            f"preflight asserts identity_inertness_wrapper_citation_path exists and "
            "identity_inertness_proven==true"
        ),
        "dense_env_export": (
            f"ca_confirmation_command exports {DENSE_SAMPLED_STATES_ENV}="
            f"{DENSE_SAMPLED_STATES_VALUE} AND {DENSE_ORDER_ENV}={DENSE_ORDER_VALUE} "
            "AND env_toggles_b_arm matches both (REVERSED order on both surfaces)"
        ),
        "mark_count_eq_sampled_states": (
            f"s1d7_band_counter_mark_count==len(sampled_states)=={EXPECTED_MARK_COUNT} "
            f"(dense [0..9] at n={N_STATES}; same reversed env+order on n={FALLBACK_N_STATES} fallback)"
        ),
        "f3b_terminal_branch": (
            "classifier terminal_branch from fold3b_variable_a_mechanism_diagnosis_receipt; "
            "decisive when F3B_STATE0_IDENTITY_STRUCTURE or F3B_MEASUREMENT_ORDER_ARTIFACT"
        ),
        "terminal_acceptance": (
            "terminal launch acceptance parses f3b_terminal_branch from "
            "fold3b_variable_a_mechanism_diagnosis_receipt.json (or wrapper f3b_terminal_branch) — "
            "NOT ok / infra_ok / terminal_branch / command exit 0 / identity_inertness_proven alone"
        ),
        "order_provenance_correctness": (
            "sampled_state_set==[0,1,2,3,4,5,6,7,8,9] AND "
            "sampled_state_order==[9,8,7,6,5,4,3,2,1,0] AND "
            "order_control_active=true AND effective_visit_order=[9..0]+[10..31] AND "
            "order_perturbation_kind=sampled_block_order_perturbation AND "
            "order_rank_by_semantic_state[str(9-i)]==i AND semantic_state_id==state_index"
        ),
        "dedup_evidence_required": (
            "dedup_reset_called==true AND dedup_session_scope==probe_subprocess on mechanism receipt; "
            "fail-closed to F3B_NO_VERDICT_SCHEMA or F3B_MARKING_OR_DEDUP_ARTIFACT if absent"
        ),
        "primary_dense_fork_readable": (
            "TRUE iff completed primary has sampled_states=[0..9], mark_count==10, "
            "per_state covers 0..9, runs.B exit_code==0, no timeout, infra_ok==true"
        ),
        "abort_fail_closed": (
            "partial JSONL, nonzero B-arm exit, timeout, or mark-coverage failure "
            "CANNOT decide mechanism branch; launch acceptance fail-closed on abnormal exit"
        ),
        "negative_science_gate": (
            "MUST NOT use compute_identity_inertness_proven or baseline crossing-vector "
            "comparison as science acceptance"
        ),
        "tracemalloc_mark_count_zero": "s1d7_tracemalloc_mark_count==0",
        "infra_ok_required": "infra_ok==true for launch validity",
        "feasibility_subsample_executable": (
            f"POST-primary: n={N_STATES} reversed dense → RSS/timeout gate (NOT exit 37/-6) → "
            f"n={FALLBACK_N_STATES} reversed dense + classify(feasibility_subsample=True) only when "
            "primary_dense_fork_readable==false"
        ),
        "wrapper_reporting": (
            "terminal launch wrapper receipt surfaces f3b_terminal_branch, "
            "mechanism_diagnosis_receipt_path, runs.primary.per_state, primary_receipt_path, "
            "fallback_trigger, primary_dense_fork_readable, identity_inertness_wrapper_citation_path, "
            "diagnostic flags"
        ),
        "resource_lane_per_fixture_only": (
            "slice5 run_fixture_* paths self-acquire/release gpu:hrm-text-158 per fixture; "
            "ca_confirmation_command MUST NOT outer-acquire at RUN_ROOT (double-acquire deadlock)"
        ),
        "resource_lane_release_trap_cleanup_only": (
            "EXIT trap is CLEANUP-ONLY at RUN_ROOT; no-ops when RUN_ROOT holds no lane; "
            "NOT protection proof"
        ),
        "diagnostic_flags": (
            "ready_for_main_science=false, counts_as_sub2=false, pre_full_stack_diagnostic=true"
        ),
        "n20_screen_policy": (
            "N=20 screen deliberately NOT bundled; identity-inertness run already validated "
            "n=32 dense machinery; this run IS the N=50-equivalent verdict arm"
        ),
        "fold3b_budget": "counts 1 launch toward Fold-3B 4-launch/140-unit budget",
    }
    proof["feasibility_subsample_classification"] = {
        "path": "(a) EXECUTABLE-VIA-PACKET",
        "grounding": (
            "run_callsite_band_counter_ca_confirmation accepts n_states; "
            "classify_ca_band_counter_confirmation accepts feasibility_subsample=True; "
            "runner hardcodes feasibility_subsample=False — packet wrapper orchestrates "
            "pilot RSS/budget gate then applies classifier override on n=16 fallback"
        ),
        "runner_commit": SCIENCE_HEAD,
    }
    draft["proof"] = proof
    draft["provenance"] = {
        "upstream_task_id": UPSTREAM_TASK_ID,
        "upstream_task_id_note": "Arc #2b historical upstream; active task_id is top-level",
        "fold1_commit": "d981582",
        "fold2_design_dual_accept": ["1783186596758", "1783186683459"],
        "fold2a_source_commit": HEAD,
        "gabe_standing_directive_verbatim": (
            "auto-research directive - full provenance to you and co_lead - no need to wait "
            "on me at any gates including pushes and gpu runs"
        ),
    }
    draft["resource_lane_release_trap"] = {
        "description": (
            "CLEANUP-ONLY shell trap on EXIT/ERR/INT/TERM; no-ops at RUN_ROOT when no outer "
            "lane held; per-fixture acquire/release inside slice5 is the sole lock boundary"
        ),
        "log_artifact": f"{RUN_ROOT}/postrun/lane_release_trap.log",
    }
    _update_pin_blocks(draft)
    preflight = draft.setdefault("preflight", {})
    preflight["command"] = build_preflight_command()
    preflight["pass_criterion"] = (
        f"exit 0; git merge-base --is-ancestor {HEAD[:7]} HEAD; parent sha match; "
        "all pinned surfaces hash-match and clean; identity_inertness_wrapper exists+proven; "
        "box code-currency pass with --allow-descendant-head; band_counter_only currency dry-check ok"
    )
    for stale_text_key in ("forbidden_at_launch", "forbidden"):
        items = draft.get("decision_contract", {}).get(stale_text_key, [])
        draft["decision_contract"][stale_text_key] = [
            item.replace("9977166", HEAD[:7]) for item in items
        ]
    if isinstance(draft.get("launch_constraints"), list):
        draft["launch_constraints"] = [
            item.replace("9977166", HEAD[:7]) for item in draft["launch_constraints"]
        ]
    return draft


def build_replay(draft: dict[str, Any]) -> dict[str, Any]:
    cmd = ca_confirmation_command()
    preflight_cmd = build_preflight_command()
    dry = draft.get("proof", {}).get("callsite_currency_dry_check_command", dry_check_launch_composition_command())
    return {
        "packet_revision": PACKET_REVISION,
        "git_head_required": HEAD,
        "run_id": RUN_ID,
        "task_id": ACTIVE_TASK_ID,
        "upstream_task_id": UPSTREAM_TASK_ID,
        "b_arm_env_toggles": build_band_counter_b_arm_env_toggles(),
        "ca_confirmation": cmd,
        "primary_command": cmd,
        "replay_sequence": ["box_preflight", "ca_confirmation"],
        "commands": {
            "box_preflight": preflight_cmd,
            "ca_confirmation": cmd,
            "callsite_currency_dry_check": dry,
        },
        "consumption_matrix": {
            "A_freshness_guard": {
                "bootstrap_launch": (
                    "python3 -B probe_bootstrap.py entry; band_counter_only env; "
                    "guard runs before pinned imports"
                ),
                "guard_entry": (
                    "bootstrap.main → run_phase3b_probe_executed_code_currency_guard BEFORE "
                    "bounded_delta_acquisition_probe import (band_counter_only)"
                ),
                "pycache_invalidation": (
                    "prepare_phase3b_band_counter_only_launch_env + guard-internal invalidation"
                ),
            }
        },
        "consumption_assertions": [
            {
                "id": "A2_bootstrap_argv",
                "assert": "child argv includes -B and probe_bootstrap.py for band_counter_only",
            },
            {
                "id": "science_terminal_branch_not_ok",
                "assert": (
                    "science verdict read from receipt.terminal_branch; "
                    "receipt.ok/infra_ok are infra-only"
                ),
            },
        ],
    }


def verify_import_invocation(draft: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("primary_command", "ca_confirmation_command"):
        cmd = draft.get("proof", {}).get(key, "")
        if "orchestrate_ca_confirmation_with_fallback" not in cmd:
            failures.append(f"draft:proof.{key}_missing_orchestrator")
        if "ca_confirmation_wrapper_exit_code" not in cmd:
            failures.append(f"draft:proof.{key}_missing_wrapper_exit_helper")
        if "s1d7_band_counter_mark_count_eq_n_states" in cmd:
            failures.append(f"draft:proof.{key}_stale_mark_count_gate")
        if 'receipt.get("ok")' in cmd and "NOT receipt" not in cmd:
            failures.append(f"draft:proof.{key}_uses_ok_as_science")
    primary = draft.get("proof", {}).get("primary_command", "")
    if "run_callsite_band_counter_scale_smoke" in primary:
        failures.append("draft:primary_command_still_scale_smoke")
    return failures


def verify_science_gate_prose(draft: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    obj = draft.get("ca_confirmation_objective", {})
    if obj.get("science_gate_field") != "f3b_terminal_branch":
        failures.append("draft:science_gate_field_not_f3b_terminal_branch")
    if "identity_inertness_proven" not in (obj.get("not_science_gate_fields") or []):
        failures.append("draft:not_science_gate_fields_missing_identity_inertness_proven")
    mandatory = draft.get("perturbation_risk_classification", {}).get(
        "ca_confirmation_mandatory", {}
    )
    if mandatory.get("science_gate") != "f3b_terminal_branch":
        failures.append("draft:ca_confirmation_mandatory_science_gate_wrong")
    if "band_counter_dominance_ok" in str(mandatory.get("gates", [])):
        failures.append("draft:still_has_c_only_dominance_gate")
    return failures


def verify_infra_render(draft: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected = build_band_counter_b_arm_env_toggles()
    if draft.get("child_emission_contract", {}).get("env_toggles_b_arm") != expected:
        failures.append("draft:env_toggles_b_arm_mismatch")
    if replay.get("b_arm_env_toggles") != expected:
        failures.append("replay:b_arm_env_toggles_mismatch")
    if replay.get("b_arm_env_toggles", {}).get("HRM_TEXT_158_PROFILE_TRACEMALLOC") != "0":
        failures.append("replay:tracemalloc_not_zero")
    primary = draft.get("proof", {}).get("primary_command", "")
    if "orchestrate_ca_confirmation_with_fallback" not in primary:
        failures.append("draft:primary_missing_ca_orchestrator")
    budget = draft.get("bounded_steps_budget", {})
    if budget.get("max_silent_phase_seconds") != 900:
        failures.append("draft:max_silent_not_900")
    if budget.get("phase_timeout_seconds") != 1800:
        failures.append("draft:phase_timeout_not_1800")
    if budget.get("total_timeout_seconds") != 3600:
        failures.append("draft:total_timeout_not_3600")
    return failures


def verify_f3b_mechanism_acceptance_contract(draft: dict[str, Any]) -> list[str]:
    """CPU-static: identity precondition + prereg alignment + negative gate."""
    failures: list[str] = []
    wrapper_path = Path(IDENTITY_INERTNESS_WRAPPER_PATH)
    if not wrapper_path.is_file():
        failures.append("variable_a:identity_wrapper_missing_on_disk")
        return failures
    try:
        wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"variable_a:identity_wrapper_parse_error:{type(exc).__name__}")
        return failures
    if wrapper.get("identity_inertness_proven") is not True:
        failures.append("variable_a:identity_precondition_not_proven")
    acceptance = draft.get("proof", {}).get("f3b_mechanism_acceptance", {})
    if acceptance.get("field") != "f3b_terminal_branch":
        failures.append("variable_a:acceptance_field_not_f3b_terminal_branch")
    if acceptance.get("variable_id") != VARIABLE_ID:
        failures.append("variable_a:acceptance_variable_id_mismatch")
    if "negative_gate" not in str(acceptance):
        failures.append("variable_a:acceptance_missing_negative_gate")
    if "n20_screen_policy" not in str(acceptance):
        failures.append("variable_a:acceptance_missing_n20_screen_policy")
    obj = draft.get("ca_confirmation_objective", {})
    if obj.get("identity_inertness_wrapper_citation_path") != IDENTITY_INERTNESS_WRAPPER_PATH:
        failures.append("variable_a:objective_identity_wrapper_path_mismatch")
    flags = obj.get("diagnostic_flags", {})
    if flags.get("inside_fold3b_budget") is not True:
        failures.append("variable_a:diagnostic_not_inside_fold3b_budget")
    if flags.get("n20_screen_bundled") is not False:
        failures.append("variable_a:diagnostic_n20_screen_should_not_be_bundled")
    if flags.get("ready_for_main_science") is not False:
        failures.append("variable_a:diagnostic_ready_for_main_science_not_false")
    return failures


def verify_variable_a_packet_contract(draft: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    toggles = build_band_counter_b_arm_env_toggles()
    if toggles.get(DENSE_SAMPLED_STATES_ENV) != DENSE_SAMPLED_STATES_VALUE:
        failures.append("variable_a:env_toggles_missing_sampled_states")
    if toggles.get(DENSE_ORDER_ENV) != DENSE_ORDER_VALUE:
        failures.append("variable_a:env_toggles_missing_reversed_order")
    cmd = draft.get("proof", {}).get("ca_confirmation_command", "")
    if f"export {DENSE_SAMPLED_STATES_ENV}={DENSE_SAMPLED_STATES_VALUE}" not in cmd:
        failures.append("variable_a:ca_confirmation_command_missing_set_export")
    if f"export {DENSE_ORDER_ENV}={DENSE_ORDER_VALUE}" not in cmd:
        failures.append("variable_a:ca_confirmation_command_missing_order_export")
    if replay.get("primary_command") != cmd:
        failures.append("variable_a:replay_primary_command_mismatch")
    render_blob = json.dumps({"draft": draft, "replay": replay}, sort_keys=True)
    if DENSE_ORDER_VALUE not in render_blob:
        failures.append("variable_a:render_missing_reversed_order_value")
    bce = draft.get("child_emission_contract", {}).get("band_counter_enable", {})
    if bce.get("sampled_states") != DENSE_SAMPLED_STATES_LIST:
        failures.append("variable_a:band_counter_enable_sampled_states")
    if int(bce.get("expected_mark_events", -1)) != EXPECTED_MARK_COUNT:
        failures.append("variable_a:expected_mark_events_not_10")
    proof = draft.get("proof", {})
    if "primary_dense_fork_readable" not in proof:
        failures.append("variable_a:proof_missing_primary_dense_fork_readable")
    if "f3b_mechanism_acceptance" not in proof:
        failures.append("variable_a:proof_missing_f3b_mechanism_acceptance")
    if "launch_wrapper_receipt_required_fields" not in proof:
        failures.append("variable_a:proof_missing_wrapper_reporting_fields")
    required_fields = proof.get("launch_wrapper_receipt_required_fields", [])
    for field in (
        "f3b_terminal_branch",
        "mechanism_diagnosis_receipt_path",
        "identity_inertness_wrapper_citation_path",
        "ready_for_main_science",
        "counts_as_sub2",
        "pre_full_stack_diagnostic",
    ):
        if field not in required_fields:
            failures.append(f"variable_a:proof_missing_wrapper_field:{field}")
    if "identity_inertness_proven" in required_fields:
        failures.append("variable_a:proof_forbidden_identity_inertness_as_wrapper_gate")
    forbidden_lane = {
        "resource_lane_holding_acquired",
        "resource_lane_canonical_name",
        "resource_lane_token_present",
        "resource_lane_released",
    }
    if forbidden_lane.intersection(required_fields):
        failures.append("variable_a:proof_forbidden_outer_lane_reporting_fields")
    pass_criteria = proof.get("pass_criteria", {})
    if "f3b_terminal_branch" not in pass_criteria:
        failures.append("variable_a:pass_criteria_missing_f3b_terminal_branch")
    if "negative_science_gate" not in pass_criteria:
        failures.append("variable_a:pass_criteria_missing_negative_science_gate")
    if "order_provenance_correctness" not in pass_criteria:
        failures.append("variable_a:pass_criteria_missing_order_provenance")
    if "terminal_acceptance" not in pass_criteria:
        failures.append("variable_a:pass_criteria_missing_terminal_acceptance")
    terminal_acceptance = pass_criteria.get("terminal_acceptance", "")
    if "f3b_terminal_branch" not in str(terminal_acceptance):
        failures.append("variable_a:terminal_acceptance_missing_f3b_terminal_branch")
    if "identity_inertness_proven==true" in str(pass_criteria.get("terminal_acceptance", "")):
        failures.append("variable_a:terminal_acceptance_must_not_use_identity_inertness")
    order_prov = pass_criteria.get("order_provenance_correctness", "")
    if "[9,8,7,6,5,4,3,2,1,0]" not in str(order_prov):
        failures.append("variable_a:order_provenance_missing_reversed_order")
    acceptance_true_iff = draft.get("proof", {}).get("f3b_mechanism_acceptance", {}).get(
        "true_iff_all", []
    )
    if not any("[9,8,7,6,5,4,3,2,1,0]" in str(item) for item in acceptance_true_iff):
        failures.append("variable_a:acceptance_missing_reversed_order")
    obj = draft.get("ca_confirmation_objective", {})
    if obj.get("science_gate_field") != "f3b_terminal_branch":
        failures.append("variable_a:objective_science_gate_not_f3b")
    if draft.get("task_id") != ACTIVE_TASK_ID:
        failures.append("variable_a:draft_task_id_not_active_slice")
    if replay.get("task_id") != ACTIVE_TASK_ID:
        failures.append("variable_a:replay_task_id_not_active_slice")
    preflight_cmd = draft.get("preflight", {}).get("command", "")
    if IDENTITY_INERTNESS_WRAPPER_PATH not in preflight_cmd:
        failures.append("variable_a:preflight_missing_identity_wrapper_path")
    if "preflight_identity_inertness_ok" not in preflight_cmd:
        failures.append("variable_a:preflight_missing_identity_precondition_check")
    return failures


def verify_dense_packet_contract(draft: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    return verify_variable_a_packet_contract(draft, replay)


def verify_no_outer_lane_acquire(draft: dict[str, Any]) -> list[str]:
    """Assert ca_confirmation does NOT outer-acquire lane at RUN_ROOT (per-fixture only)."""
    failures: list[str] = []
    cmd = draft.get("proof", {}).get("ca_confirmation_command", "")
    if "hrm_text_158_r7_resource_lane_acquire.py" in cmd:
        failures.append("lane_acquire:forbidden_outer_acquire_script")
    if "prelaunch/resource_lane_holding.json" in cmd:
        failures.append("lane_acquire:forbidden_outer_holding_test")
    return failures


STALE_OUTER_LANE_ARTIFACT_PATHS: tuple[str, ...] = (
    "prelaunch/resource_lane_holding.json",
    "post_gpu/resource_lane_release_witness.json",
)


def verify_no_stale_outer_lane_manifest(draft: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    """Fail if rendered manifest/pass_criteria/required-fields reference impossible outer-lane files."""
    failures: list[str] = []
    proof = draft.get("proof", {})
    scan_targets: list[tuple[str, Any]] = [
        ("proof.artifacts", proof.get("artifacts", {})),
        ("proof.pass_criteria", proof.get("pass_criteria", {})),
        ("proof.launch_wrapper_receipt_required_fields", proof.get("launch_wrapper_receipt_required_fields", [])),
        ("proof.f3b_mechanism_acceptance", proof.get("f3b_mechanism_acceptance", {})),
        ("resource_lane_release_trap", draft.get("resource_lane_release_trap", {})),
        ("replay", replay),
    ]
    for label, obj in scan_targets:
        blob = json.dumps(obj, sort_keys=True)
        for stale_path in STALE_OUTER_LANE_ARTIFACT_PATHS:
            if stale_path in blob:
                failures.append(f"lane_manifest:stale_outer_lane_path:{label}:{stale_path}")
    artifacts = proof.get("artifacts", {})
    for required_key in (
        "primary_runs_a_resource_lane_holding",
        "primary_runs_b_resource_lane_holding",
        "lane_release_trap_log",
        "terminal_lane_free_witness",
        "f3b_mechanism_diagnosis_receipt",
        "f3b_terminal_branch",
    ):
        if required_key not in artifacts:
            failures.append(f"lane_manifest:missing_required_artifact:{required_key}")
    return failures


def verify_classifier_launch_executed_pin() -> list[str]:
    """Fail-closed: wrapper heredoc executes F3B classifier → module must be pinned+matching."""
    failures: list[str] = []
    heredoc = CA_CONFIRMATION_HEREDOC
    import_markers = (
        "from calm.hrm_text_158.native_full_stack.f3b_why_state0_branch import",
        "classify_f3b_why_state0_branch",
        "build_branch_input_contract_from_ca_receipt",
    )
    if not all(marker in heredoc for marker in import_markers):
        failures.append("classifier_pin:heredoc_missing_f3b_import_or_classify")
        return failures

    def _classifier_pin_ok(pins: dict[str, str]) -> bool:
        if F3B_CLASSIFIER_REL not in pins:
            return False
        return pins[F3B_CLASSIFIER_REL] == sha256_disk_file(F3B_CLASSIFIER_REL)

    if not _classifier_pin_ok(PINS):
        failures.append("classifier_pin:missing_or_stale_in_pins")
    field = "f3b_why_state0_branch_sha256"
    if CODE_PIN_FIELD_TO_REL.get(field) != F3B_CLASSIFIER_REL:
        failures.append("classifier_pin:code_pin_field_not_mapped")
    # Negative invariant: dropping the pin must fail the same check.
    pins_without = {k: v for k, v in PINS.items() if k != F3B_CLASSIFIER_REL}
    if _classifier_pin_ok(pins_without):
        failures.append("classifier_pin:negative_test_dropping_pin_still_passes")
    return failures


def verify_wrapper_receipt_sink_executable() -> list[str]:
    """Assert heredoc writes mechanism + wrapper receipts and routes F3B classifier."""
    failures: list[str] = []
    heredoc = CA_CONFIRMATION_HEREDOC
    if "ca_confirmation_wrapper_receipt.json" not in heredoc:
        failures.append("wrapper_sink:missing_receipt_path")
    if MECHANISM_RECEIPT_NAME not in heredoc:
        failures.append("wrapper_sink:missing_mechanism_receipt_name")
    if "build_mechanism_diagnosis_receipt" not in heredoc:
        failures.append("wrapper_sink:missing_build_mechanism_fn")
    if "classify_f3b_why_state0_branch" not in heredoc:
        failures.append("wrapper_sink:missing_classify_fn")
    if "build_branch_input_contract_from_ca_receipt" not in heredoc:
        failures.append("wrapper_sink:missing_build_branch_inputs_fn")
    if "compute_primary_dense_fork_readable" not in heredoc:
        failures.append("wrapper_sink:missing_fork_readable_fn")
    if "f3b_terminal_branch" not in heredoc:
        failures.append("wrapper_sink:missing_f3b_terminal_branch_field")
    if "mechanism_diagnosis_receipt_path" not in heredoc:
        failures.append("wrapper_sink:missing_mechanism_receipt_path_field")
    if "identity_inertness_wrapper_citation_path" not in heredoc:
        failures.append("wrapper_sink:missing_identity_wrapper_citation")
    if "wrapper_receipt_path.write_text" not in heredoc:
        failures.append("wrapper_sink:missing_wrapper_write_text")
    if "mechanism_receipt_path.write_text" not in heredoc:
        failures.append("wrapper_sink:missing_mechanism_write_text")
    if "compute_identity_inertness_proven" in heredoc:
        failures.append("wrapper_sink:forbidden_identity_inertness_science_gate")
    if "_crossing_vector_over_dense" in heredoc:
        failures.append("wrapper_sink:forbidden_baseline_crossing_compare")
    if "baseline_vec.get(state)" in heredoc:
        failures.append("wrapper_sink:forbidden_elementwise_baseline_compare")
    if "_reversed_order_provenance_ok" not in heredoc:
        failures.append("wrapper_sink:missing_reversed_order_provenance_fn")
    predicates = (
        "sampled_states",
        "s1d7_band_counter_mark_count",
        "s1d7_band_counter_mark_count_eq_sampled_state_count",
        "per_state",
        "state_index",
        "exit_code",
        "subprocess_timeout_expired",
        "infra_ok",
        "order_control_active",
        "effective_visit_order",
        "order_rank_by_semantic_state",
        "semantic_state_id",
        "sampled_state_set",
        "sampled_state_order",
        "dedup_reset_called",
        "dedup_session_scope",
        "variable_id",
        "control_reason",
    )
    for pred in predicates:
        if pred not in heredoc:
            failures.append(f"wrapper_sink:missing_predicate:{pred}")
    if "DENSE_ORDER = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]" not in heredoc:
        failures.append("wrapper_sink:missing_reversed_dense_order_constant")
    if "abnormal_exit" not in heredoc:
        failures.append("wrapper_sink:missing_abnormal_exit_marker")
    for diagnostic_flag in (
        "ready_for_main_science",
        "counts_as_sub2",
        "pre_full_stack_diagnostic",
    ):
        if diagnostic_flag not in heredoc:
            failures.append(f"wrapper_sink:missing_diagnostic_flag:{diagnostic_flag}")
    for forbidden_lane_field in (
        "read_resource_lane_launch_reporting",
        "resource_lane_holding_acquired",
        "resource_lane_canonical_name",
        "resource_lane_token_present",
        "resource_lane_released",
    ):
        if forbidden_lane_field in heredoc:
            failures.append(f"wrapper_sink:forbidden_outer_lane_field:{forbidden_lane_field}")
    return failures


def verify_wrapper_heredoc_contract(draft: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    cmd = draft.get("proof", {}).get("ca_confirmation_command", "")
    for required in (
        "orchestrate_ca_confirmation_with_fallback",
        "ca_confirmation_wrapper_exit_code",
        "science_verdict_source",
    ):
        if required not in cmd:
            failures.append(f"draft:wrapper_heredoc_missing:{required}")
    if "s1d7_band_counter_mark_count_eq_n_states" in cmd:
        failures.append("draft:wrapper_heredoc_stale_mark_count_gate")
    if "SystemExit(42)" in cmd and "ca_confirmation_wrapper_exit_code" not in cmd:
        failures.append("draft:wrapper_heredoc_hardcoded_exit_42")
    return failures


def verify_feasibility_subsample_trigger_safety() -> list[str]:
    """Synthetic CPU assertions: fallback trigger must not mask exit 37/-6 fail-closed."""
    failures: list[str] = []

    def _case(name: str, receipt: dict[str, Any], expect_fallback: bool) -> None:
        result = evaluate_feasibility_subsample_fallback_trigger(receipt)
        if result["fallback"] != expect_fallback:
            failures.append(
                f"trigger_safety:{name}:expected_fallback={expect_fallback}:got={result}"
            )

    _case(
        "exit_37_no_fallback",
        {
            "peak_rss_gib": 3.0,
            "terminal_branch": "INFRA_NULL",
            "infra_ok": False,
            "runs": {"B": {"exit_code": 37, "subprocess_timeout_expired": False}},
        },
        False,
    )
    _case(
        "exit_m6_no_fallback",
        {
            "peak_rss_gib": 3.0,
            "terminal_branch": "INFRA_NULL",
            "infra_ok": False,
            "runs": {"B": {"exit_code": -6, "subprocess_timeout_expired": False}},
        },
        False,
    )
    _case(
        "exit_37_timeout_cooccur_no_fallback",
        {
            "peak_rss_gib": 3.0,
            "terminal_branch": "INFRA_NULL",
            "infra_ok": False,
            "runs": {"B": {"exit_code": 37, "subprocess_timeout_expired": True}},
        },
        False,
    )
    _case(
        "timeout_triggers_fallback",
        {
            "peak_rss_gib": 3.0,
            "terminal_branch": "INFRA_NULL",
            "infra_ok": False,
            "runs": {"B": {"exit_code": 1, "subprocess_timeout_expired": True}},
        },
        True,
    )
    _case(
        "rss_triggers_fallback",
        {
            "peak_rss_gib": 7.0,
            "terminal_branch": "CA_PERSISTS",
            "infra_ok": True,
            "runs": {"B": {"exit_code": 0, "subprocess_timeout_expired": False}},
        },
        True,
    )
    _case(
        "clean_n32_no_fallback",
        {
            "peak_rss_gib": 3.0,
            "terminal_branch": "CA_PERSISTS",
            "infra_ok": True,
            "runs": {"B": {"exit_code": 0, "subprocess_timeout_expired": False}},
        },
        False,
    )
    return failures


def verify_box_lane_infra_pin_contract(draft: dict[str, Any]) -> list[str]:
    """box_lane.py must be pinned before preflight subprocess imports it."""
    failures: list[str] = []
    if BOX_LANE_REL not in PINS:
        failures.append("pins:missing_box_lane")
    code_pins = draft.get("code_pins", {})
    if "box_lane_sha256" not in code_pins:
        failures.append("code_pins:missing_box_lane_sha256")
    elif str(code_pins["box_lane_sha256"]) != PINS.get(BOX_LANE_REL):
        failures.append("code_pins:box_lane_sha256_mismatch_pins")
    preflight_cmd = draft.get("preflight", {}).get("command", "")
    if BOX_LANE_REL not in preflight_cmd:
        failures.append("preflight:inline_pins_missing_box_lane")
    note = str(draft.get("code_pins_note", ""))
    if "not packet-pinned" in note or "NOT packet-pinned" in note:
        failures.append("code_pins_note:stale_box_lane_unpinned_wording")
    render_blob = json.dumps(draft, sort_keys=True)
    if "not packet-pinned" in render_blob:
        failures.append("render:stale_box_lane_unpinned_wording")
    pinned_rels = sorted(PINS.keys())
    if BOX_LANE_REL in pinned_rels:
        preflight_rel = "scripts/box_lane_code_currency_preflight.py"
        if preflight_rel in pinned_rels:
            if pinned_rels.index(BOX_LANE_REL) > pinned_rels.index(preflight_rel):
                failures.append("preflight:inline_pins_box_lane_after_preflight_script")
    return failures


def verify_launch_pins_on_disk(pins: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for rel, expected in pins.items():
        actual = sha256_disk_file(rel)
        if actual != expected:
            failures.append(f"launch_pin_mismatch:{rel}")
    return failures


def verify_science_surfaces_dirty_vs_baseline(draft: dict[str, Any]) -> list[str]:
    """Fail if any science pin's on-disk bytes differ from git blob @ git_head_required."""
    failures: list[str] = []
    science_head = str(draft.get("git_head_required", HEAD))
    for rel in PINS:
        if rel in INFRA_PIN_RELS_ON_DISK:
            continue
        disk_sha = sha256_disk_file(rel)
        baseline_sha = sha256_git_blob(science_head, rel)
        if disk_sha != baseline_sha:
            failures.append(f"science_surface_dirty:{rel}")
        if PINS[rel] != baseline_sha:
            failures.append(f"science_pin_stale_vs_baseline:{rel}")
    return failures


def verify_real_launch_dry_check() -> list[str]:
    failures: list[str] = []
    try:
        import sys

        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
            dry_check_callsite_b_prime_b_arm_launch_composition,
        )

        receipt = dry_check_callsite_b_prime_b_arm_launch_composition()
        if not receipt.get("ok"):
            failures.append("dry_check:launch_composition_not_ok")
        checks = receipt.get("guard_receipt_checks") or {}
        if not checks.get("guard_ran_before_pinned_imports"):
            failures.append("dry_check:guard_ran_before_pinned_imports_false")
    except Exception as exc:
        failures.append(f"dry_check:exception:{type(exc).__name__}")
    return failures


def self_verify(draft: dict[str, Any], replay: dict[str, Any]) -> None:
    pins_ok = all(pin_expected_sha256(rel) == pin for rel, pin in PINS.items())
    code_pin_failures = verify_code_pins_against_commit(draft)
    science_dirty = verify_science_surfaces_dirty_vs_baseline(draft)
    pin_ok = pins_ok and not code_pin_failures and not science_dirty
    stale: list[str] = []
    if draft.get("git_head_required") != HEAD:
        stale.append("draft:git_head_mismatch")
    if not pins_ok:
        stale.append("pins_mismatch_commit")
    stale.extend(code_pin_failures)
    stale.extend(science_dirty)
    preflight_cmd = draft.get("preflight", {}).get("command", "")
    if HEAD not in preflight_cmd:
        stale.append("preflight:stale_head_ref")
    if "merge-base --is-ancestor" not in preflight_cmd:
        stale.append("preflight:missing_ancestor_head_check")
    if "--allow-descendant-head" not in preflight_cmd:
        stale.append("preflight:missing_allow_descendant_head")
    if replay.get("commands", {}).get("box_preflight") != preflight_cmd:
        stale.append("replay:box_preflight_mismatch")
    stale.extend(verify_import_invocation(draft))
    stale.extend(verify_science_gate_prose(draft))
    stale.extend(verify_infra_render(draft, replay))
    stale.extend(verify_real_launch_dry_check())
    stale.extend(verify_f3b_mechanism_acceptance_contract(draft))
    stale.extend(verify_dense_packet_contract(draft, replay))
    stale.extend(verify_no_outer_lane_acquire(draft))
    stale.extend(verify_no_stale_outer_lane_manifest(draft, replay))
    stale.extend(verify_wrapper_receipt_sink_executable())
    stale.extend(verify_classifier_launch_executed_pin())
    stale.extend(verify_wrapper_heredoc_contract(draft))
    stale.extend(verify_feasibility_subsample_trigger_safety())
    stale.extend(verify_box_lane_infra_pin_contract(draft))
    stale.extend(verify_whole_render_sweep(draft, replay))
    mandatory = draft.get("perturbation_risk_classification", {}).get(
        "ca_confirmation_mandatory", {}
    )
    if len(mandatory.get("gates", [])) != len(CONFIRMATION_GATES):
        stale.append("draft:confirmation_gate_count_wrong")
    render_blob = json.dumps({"draft": draft, "replay": replay}, sort_keys=True)
    for pat in (
        "s1d7_band_counter_mark_count_eq_4",
        "s1d7_band_counter_mark_count_eq_n_states",
        "band_counter_dominance_ok",
        "call_site_status_resolved",
        "s1d7_call_site_candidate_eq_c",
        "run_callsite_band_counter_scale_smoke",
        '"HRM_TEXT_158_PROFILE_TRACEMALLOC": "1"',
        OLD_CALLSITE_RUN_ID,
    ):
        if pat in render_blob:
            stale.append(f"stale_render_pattern:{pat}")
    print(
        f"self_verify git_head_required=={HEAD[:7]} "
        f"pins_match_commit={pin_ok} stale_refs={stale or 'none'}"
    )
    if stale:
        raise SystemExit(1)


def main() -> None:
    refresh_launch_pins()
    refresh_science_pins_from_head()
    draft = build_draft()
    replay = build_replay(draft)
    DRAFT.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPLAY.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    self_verify(draft, replay)
    print(f"wrote {DRAFT}")
    print(f"wrote {REPLAY}")


if __name__ == "__main__":
    main()
