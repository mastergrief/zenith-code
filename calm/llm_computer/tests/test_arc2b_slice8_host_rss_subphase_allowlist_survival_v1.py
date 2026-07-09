"""Survival test: every learner host-RSS observer mark must be allowlisted.

Closes the instrumentation-null gap where C2b_cpu_reference_cap_apply was
wrapped in bounded_delta_learner but filtered by PROFILE_HOST_RSS_SUBPHASE_IDS
before host_rss_profile persistence.

Path resolution uses Path(bounded_delta_learner.__file__) so the assertion is
location-robust (not Path(__file__).parents[N]).
"""

from __future__ import annotations

import re
from pathlib import Path

import calm.hrm_text_158.native_full_stack.bounded_delta_learner as bounded_delta_learner
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    PROFILE_HOST_RSS_SUBPHASE_IDS,
)


def _scope_wrap_subphase_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for match in re.finditer(r"with _host_rss_subphase_scope\((.*?)\):", text, re.S):
        for sub_phase_id in re.findall(
            r'sub_phase_id\s*=\s*"([^"]+)"', match.group(1)
        ):
            ids.add(sub_phase_id)
    return ids


def test_all_host_rss_subphase_scope_ids_are_allowlisted() -> None:
    learner_path = Path(bounded_delta_learner.__file__).resolve()
    assert learner_path.is_file(), f"learner path missing: {learner_path}"
    ids = _scope_wrap_subphase_ids(learner_path.read_text(encoding="utf-8"))
    assert ids, "expected at least one _host_rss_subphase_scope wrap"
    missing = sorted(ids - set(PROFILE_HOST_RSS_SUBPHASE_IDS))
    assert not missing, (
        "observer sub_phase_id(s) wrapped in bounded_delta_learner but NOT in "
        f"PROFILE_HOST_RSS_SUBPHASE_IDS (would be silently filtered): {missing}"
    )
