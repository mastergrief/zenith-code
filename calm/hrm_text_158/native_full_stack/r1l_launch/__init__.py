"""R1-L launch/materialization/freeze facade (CPU characterization surface)."""
from __future__ import annotations

from calm.hrm_text_158.native_full_stack.r1l_launch.argv import (
    assert_suffix_equals_child,
    build_child_timeout_argv,
    build_run_phase_bash_c,
    build_watch_wrap_spawn_argv,
    render_argv_shell,
    spawn_suffix_after_double_dash,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.budget import (
    MONITOR_TIMEOUT_MS_MAX,
    BudgetPlan,
    derive_budget_plan,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.classify import (
    TerminalObservation,
    TerminalVerdict,
    classify_terminal,
    count_runner_pass,
    last_nonempty_line,
    parse_exit_rc,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.freeze_digest import (
    content_digest_from_member_records,
    content_digest_from_members,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.materialize import (
    FROZEN_FIXTURE_CONTENT_DIGEST,
    PHASE_ORDER,
    PhaseFilePreflightError,
    absolute_phase_paths,
    mint_phase_files,
    re_resolve_phase_manifest,
)

__all__ = [
    "MONITOR_TIMEOUT_MS_MAX",
    "FROZEN_FIXTURE_CONTENT_DIGEST",
    "PHASE_ORDER",
    "BudgetPlan",
    "PhaseFilePreflightError",
    "TerminalObservation",
    "TerminalVerdict",
    "absolute_phase_paths",
    "assert_suffix_equals_child",
    "build_child_timeout_argv",
    "build_run_phase_bash_c",
    "build_watch_wrap_spawn_argv",
    "classify_terminal",
    "content_digest_from_member_records",
    "content_digest_from_members",
    "count_runner_pass",
    "derive_budget_plan",
    "last_nonempty_line",
    "mint_phase_files",
    "parse_exit_rc",
    "re_resolve_phase_manifest",
    "render_argv_shell",
    "spawn_suffix_after_double_dash",
]
