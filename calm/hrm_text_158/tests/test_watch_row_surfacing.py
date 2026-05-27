"""Tests for watch-row surfacing in probe_hrm_text_158 (codex tooling slice
msg 1779691883270 / 1779691762976: accepted-exception policy needs MECHANICAL
visibility, not prose). Covers the pure helpers — merge from ckpt
`config.watch_rows`, grep-able line format, and the non-blocking aggregate —
including parsed-ok, parsed-fail, missing/no watch rows, and backward-compat.
No model / no GPU.
"""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_spec = importlib.util.spec_from_file_location(
    "_probe_hrm_text_158", os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)

merge_watch_rows = _probe.merge_watch_rows
format_watch_line = _probe.format_watch_line
watch_aggregate = _probe.watch_aggregate

# The banked accepted-exception row (config.watch_rows on the L0c1 chain head).
_R1B2 = {"key": "r1b2:10_minus_1", "question": "what is 10 minus 1?",
         "expected": 9, "source_rung": "R1b2"}


# --------------------------------------------------------------------------- #
# merge_watch_rows — sourcing from ckpt config + dedup + missing handling
# --------------------------------------------------------------------------- #

def test_merge_sources_config_rows():
    # The whole point: a probe with no --watch-rows-json still picks up the
    # accepted exception from the chain-head config.
    assert merge_watch_rows([], [_R1B2]) == [_R1B2]


def test_merge_dedup_passed_wins():
    passed = [{"key": "passed", "question": "what is 10 minus 1?", "expected": 9}]
    merged = merge_watch_rows(passed, [_R1B2])
    assert len(merged) == 1 and merged[0]["key"] == "passed"


def test_merge_union_distinct_rows():
    other = {"key": "r1b2:11_minus_1", "question": "what is 11 minus 1?", "expected": 10}
    merged = merge_watch_rows([_R1B2], [other])
    assert {m["question"] for m in merged} == {"what is 10 minus 1?", "what is 11 minus 1?"}


def test_merge_missing_config_is_passed_only():
    # No / empty config.watch_rows must not drop the --watch-rows-json rows.
    assert merge_watch_rows([_R1B2], None) == [_R1B2]
    assert merge_watch_rows([_R1B2], []) == [_R1B2]


def test_merge_no_rows_is_empty():
    assert merge_watch_rows(None, None) == []


# --------------------------------------------------------------------------- #
# format_watch_line — grep-able, covers parsed-fail (the accepted exception)
# --------------------------------------------------------------------------- #

def test_format_parsed_fail_row():
    line = format_watch_line({
        "key": "r1b2:10_minus_1", "question": "what is 10 minus 1?",
        "expected": 9, "decoded": "0", "parsed": 0, "parsed_ok": False})
    assert line.startswith("[probe-watch] ")
    assert "r1b2:10_minus_1" in line
    assert "expected=9" in line and "decoded='0'" in line and "parsed_ok=False" in line


def test_format_parsed_ok_row():
    line = format_watch_line({
        "key": "r1b2:11_minus_1", "question": "what is 11 minus 1?",
        "expected": 10, "decoded": "10", "parsed": 10, "parsed_ok": True})
    assert "parsed_ok=True" in line and "expected=10" in line


def test_format_includes_source_rung_when_present():
    # Accepted exception keyed to its rung — mechanical, not prose.
    line = format_watch_line({
        "key": "r1b2:10_minus_1", "question": "what is 10 minus 1?",
        "expected": 9, "decoded": "0", "parsed": 0, "parsed_ok": False,
        "source_rung": "R1b2"})
    assert "source_rung=R1b2" in line


def test_format_omits_source_rung_when_absent():
    line = format_watch_line({
        "key": "x", "question": "q?", "expected": 1,
        "decoded": "1", "parsed": 1, "parsed_ok": True})
    assert "source_rung" not in line


# --------------------------------------------------------------------------- #
# watch_aggregate — non-blocking; empty is vacuously ok (backward-compat)
# --------------------------------------------------------------------------- #

def test_aggregate_mixed():
    agg = watch_aggregate([{"parsed_ok": True}, {"parsed_ok": False}, {"parsed_ok": True}])
    assert agg == {"n_total": 3, "n_parsed_ok": 2, "all_parsed_ok": False}


def test_aggregate_all_ok():
    assert watch_aggregate([{"parsed_ok": True}])["all_parsed_ok"] is True


def test_aggregate_empty_vacuously_ok():
    # Backward-compat: a probe with no watch rows is NOT a failure.
    assert watch_aggregate([]) == {"n_total": 0, "n_parsed_ok": 0, "all_parsed_ok": True}


def test_aggregate_single_accepted_exception_not_all_ok():
    # The r1b2:10_minus_1-only case: reported, parsed_ok false, but the
    # aggregate is informational (the audit itself does not fail on it).
    assert watch_aggregate([{"parsed_ok": False}]) == {
        "n_total": 1, "n_parsed_ok": 0, "all_parsed_ok": False}


# --------------------------------------------------------------------------- #
# L0c watch transform + parametrized-audit defaults (codex msg 1779693537447):
# the math-surface config watch row maps to the L0c surface so it decodes in
# the exhaustive-L0c audit instead of falsely reporting NOT_IN_ACTIVE.
# --------------------------------------------------------------------------- #

def test_l0c_watch_transform_maps_math_to_l0c():
    out = _probe._l0c_watch_transform({
        "key": "r1b2:10_minus_1", "question": "what is 10 minus 1?",
        "expected": 9, "source_rung": "R1b2"})
    assert out["question"] == "10 minus 1 equals what?"
    # key / expected / source_rung preserved.
    assert out["key"] == "r1b2:10_minus_1" and out["expected"] == 9
    assert out["source_rung"] == "R1b2"


def test_l0c_watch_transform_passthrough_non_math():
    # Already-L0c (or foreign) questions pass through unchanged — matched
    # directly against the L0c support or reported NOT_IN_ACTIVE by design.
    row = {"key": "x", "question": "10 minus 1 equals what?", "expected": 9}
    assert _probe._l0c_watch_transform(row) == row


def test_l0c_exhaustive_audit_mutex_fails_fast():
    # CLI mutex must fail BEFORE ckpt load (codex msg 1779694143993): a
    # co-passed audit mode must error, not silently win via dispatch order.
    import subprocess
    p = subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts", "probe_hrm_text_158.py"),
         "--ckpt-path", "/nonexistent_ckpt_for_mutex_test.pt",
         "--l0c-exhaustive-audit", "--language-supports"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _REPO}, timeout=120)
    combined = p.stdout + p.stderr
    assert p.returncode != 0, combined
    assert "--l0c-exhaustive-audit conflicts with" in combined, combined
    assert "--language-supports" in combined, combined
    # fail-fast: never reached ckpt load (no torch.load FileNotFoundError).
    assert "No such file" not in combined and "FileNotFound" not in combined, combined


def test_probe_exhaustive_defaults_are_math_a0():
    # math-A0 behavior is the default: no support_builder/transform override,
    # label "probe-exhaustive" (so the watcher's existing grep still matches).
    import inspect
    sig = inspect.signature(_probe.probe_exhaustive_finite_supports)
    assert sig.parameters["support_builder"].default is None
    assert sig.parameters["expected_aggregate"].default is None
    assert sig.parameters["label"].default == "probe-exhaustive"
    assert sig.parameters["watch_row_transform"].default is None


# --------------------------------------------------------------------------- #
# Watcher durability: the math_a0 grep pattern surfaces watch-row lines so
# accepted exceptions appear in producer/consumer logs (codex fix 2).
# --------------------------------------------------------------------------- #

def _watcher_modes():
    _wspec = importlib.util.spec_from_file_location(
        "_parallel_audit_watcher",
        os.path.join(_REPO, "scripts", "parallel_audit_watcher.py"))
    _watcher = importlib.util.module_from_spec(_wspec)
    _wspec.loader.exec_module(_watcher)
    return _watcher, dict((name, grep) for name, _flags, grep in _watcher._AUDIT_MODES), \
        dict((name, flags) for name, flags, _grep in _watcher._AUDIT_MODES)


def test_watcher_math_a0_pattern_surfaces_watch_rows():
    import re
    _watcher, pats, _flags = _watcher_modes()
    pat = pats["math_a0"]
    # Still matches the exhaustive aggregate (no regression)...
    assert re.search(pat, "[probe-exhaustive] AGGREGATE strict=1254/1255 = 0.9992")
    # ...and now the watch-row line + aggregate.
    assert re.search(pat, "[probe-watch] r1b2:10_minus_1 'what is 10 minus 1?' "
                          "expected=9 decoded='0' parsed=0 parsed_ok=False source_rung=R1b2")
    assert re.search(pat, "[probe-watch] WATCH AGGREGATE parsed_ok=0/1")


# --------------------------------------------------------------------------- #
# F.3b: the l0c_exhaustive audit mode is wired with its own flag + grep, and
# math_a0 stays present (report-only l0c1 continuity is asserted separately).
# --------------------------------------------------------------------------- #

def test_watcher_l0c_exhaustive_mode_wired():
    _watcher, _pats, flags = _watcher_modes()
    names = [name for name, _f, _g in _watcher._AUDIT_MODES]
    assert "l0c_exhaustive" in names
    assert flags["l0c_exhaustive"] == ["--l0c-exhaustive-audit"]
    # Report-only continuity: l0c1 + math_a0 modes are retained.
    assert "l0c1" in names and "math_a0" in names


def test_watcher_l0c_exhaustive_pattern_surfaces_aggregate_and_watch():
    import re
    _watcher, pats, _flags = _watcher_modes()
    pat = pats["l0c_exhaustive"]
    # The exhaustive-L0c aggregate (label baked by the parametrized probe)...
    assert re.search(pat, "[probe-l0c-exhaustive] AGGREGATE strict=1130/1255 = 0.9004")
    # ...and the SAME accepted exception, mapped to the L0c surface.
    assert re.search(pat, "[probe-watch] r1b2:10_minus_1 '10 minus 1 equals what?' "
                          "expected=9 decoded='0' parsed=0 parsed_ok=False source_rung=R1b2")
    assert re.search(pat, "[probe-watch] WATCH AGGREGATE parsed_ok=0/1")


def test_watcher_l0c_exhaustive_pattern_excludes_math_a0_aggregate():
    # The two exhaustive modes are distinguishable on their AGGREGATE lines
    # (shared [probe-watch] lines are by design — same accepted exception):
    # each mode's grep must NOT swallow the other's aggregate.
    import re
    _watcher, pats, _flags = _watcher_modes()
    math_a0_agg = "[probe-exhaustive] AGGREGATE strict=1254/1255 = 0.9992"
    l0cx_agg = "[probe-l0c-exhaustive] AGGREGATE strict=1130/1255 = 0.9004"
    assert not re.search(pats["l0c_exhaustive"], math_a0_agg)
    assert not re.search(pats["math_a0"], l0cx_agg)


def test_watcher_l0c2k1_identity_mode_wired_and_printed():
    _watcher, pats, flags = _watcher_modes()
    names = [name for name, _f, _g in _watcher._AUDIT_MODES]
    assert "l0c2k1identity" in names
    assert flags["l0c2k1identity"] == ["--l0c2k1-identity-audit"]
    assert pats["l0c2k1identity"] == r"L0C2K1IDENTITY AGGREGATE"

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    band_line = next(
        line for line in src.splitlines()
        if "for name in (" in line and "l0c2k1identityfull" in line and "l0c2k3" in line
    )
    assert "l0c2k1identity" in band_line


def test_watcher_l0c2k1_identity_pattern_excludes_k1_and_edge():
    import re
    _watcher, pats, _flags = _watcher_modes()
    k1_agg = "[probe-language] L0C2K1 AGGREGATE strict=29/29 = 1.0000"
    edge_agg = "[probe-language] L0C2K1EDGE AGGREGATE strict=65/65 = 1.0000"
    identity_agg = "[probe-language] L0C2K1IDENTITY AGGREGATE strict=90/90 = 1.0000"

    assert re.search(pats["l0c2k1identity"], identity_agg)
    assert not re.search(pats["l0c2k1"], identity_agg)
    assert not re.search(pats["l0c2k1edge"], identity_agg)
    assert not re.search(pats["l0c2k1identity"], k1_agg)
    assert not re.search(pats["l0c2k1identity"], edge_agg)


# --------------------------------------------------------------------------- #
# F.4d-identity-full: the full-density 90/90 coverage audit mode is wired with
# its own flag + grep, printed in the per-step consumer summary, and its
# aggregate token does NOT cross-match the sparse identity / K1 / K1EDGE modes.
# --------------------------------------------------------------------------- #

def test_watcher_l0c2k1_identity_full_mode_wired_and_printed():
    _watcher, pats, flags = _watcher_modes()
    names = [name for name, _f, _g in _watcher._AUDIT_MODES]
    assert "l0c2k1identityfull" in names
    assert flags["l0c2k1identityfull"] == ["--l0c2k1-identity-full-audit"]
    assert pats["l0c2k1identityfull"] == r"L0C2K1IDENTITYFULL AGGREGATE"
    # The sparse identity mode is retained (no regression).
    assert "l0c2k1identity" in names

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    # Printed in the per-step consumer band grouping.
    band_line = next(
        line for line in src.splitlines()
        if "for name in (" in line and "l0c2k1identity" in line and "l0c2k3" in line
    )
    assert "l0c2k1identityfull" in band_line


def test_watcher_l0c2k1_identity_full_pattern_excludes_sparse_and_kband():
    import re
    _watcher, pats, _flags = _watcher_modes()
    k1_agg = "[probe-language] L0C2K1 AGGREGATE strict=29/29 = 1.0000"
    edge_agg = "[probe-language] L0C2K1EDGE AGGREGATE strict=65/65 = 1.0000"
    identity_agg = "[probe-language] L0C2K1IDENTITY AGGREGATE strict=90/90 = 1.0000"
    full_agg = "[probe-language] L0C2K1IDENTITYFULL AGGREGATE strict=90/90 = 1.0000"

    # The full pattern matches only its own aggregate.
    assert re.search(pats["l0c2k1identityfull"], full_agg)
    assert not re.search(pats["l0c2k1identityfull"], identity_agg)
    assert not re.search(pats["l0c2k1identityfull"], k1_agg)
    assert not re.search(pats["l0c2k1identityfull"], edge_agg)
    # And no other mode's pattern swallows the full aggregate (trailing-space
    # anchor: "L0C2K1IDENTITY " never matches "L0C2K1IDENTITYFULL ").
    assert not re.search(pats["l0c2k1identity"], full_agg)
    assert not re.search(pats["l0c2k1"], full_agg)
    assert not re.search(pats["l0c2k1edge"], full_agg)


# --------------------------------------------------------------------------- #
# STEP 1 K2 addition surfaces: trainable addition-full and trained-OUT
# heldout-50s diagnostic are separate watcher modes and do not cross-match the
# legacy L0C2K2 aggregate token.
# --------------------------------------------------------------------------- #

def test_watcher_l0c2k2_addition_modes_wired_and_printed():
    _watcher, pats, flags = _watcher_modes()
    names = [name for name, _f, _g in _watcher._AUDIT_MODES]
    assert "l0c2k2additionfull" in names
    assert "l0c2k2addition50s" in names
    assert "l0c2k2addition60stracetrain" in names
    assert "l0c2k2addition60straceheld" in names
    assert "l0c2k2additionheldout50s" in names
    assert "l0c2k2additionheldout60s" in names
    assert flags["l0c2k2additionfull"] == ["--l0c2k2-addition-full-audit"]
    assert flags["l0c2k2addition50s"] == ["--l0c2k2-addition-50s-audit"]
    assert flags["l0c2k2addition60stracetrain"] == [
        "--l0c2k2-addition-60s-trace-train-audit", "--max-gen", "128",
    ]
    assert flags["l0c2k2addition60straceheld"] == [
        "--l0c2k2-addition-60s-trace-held-audit", "--max-gen", "128",
    ]
    assert flags["l0c2k2additionheldout50s"] == ["--l0c2k2-addition-heldout-50s-audit"]
    assert flags["l0c2k2additionheldout60s"] == ["--l0c2k2-addition-heldout-60s-audit"]
    assert pats["l0c2k2additionfull"] == r"L0C2K2ADDITIONFULL AGGREGATE"
    assert pats["l0c2k2addition50s"] == r"L0C2K2ADDITION50S AGGREGATE"
    assert pats["l0c2k2addition60stracetrain"] == r"L0C2K2ADDITION60STRACETRAIN AGGREGATE"
    assert pats["l0c2k2addition60straceheld"] == r"L0C2K2ADDITION60STRACEHELD AGGREGATE"
    assert pats["l0c2k2additionheldout50s"] == r"L0C2K2ADDITIONHELDOUT50S AGGREGATE"
    assert pats["l0c2k2additionheldout60s"] == r"L0C2K2ADDITIONHELDOUT60S AGGREGATE"

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert "l0c2k2additionfull" in src
    assert "l0c2k2addition50s" in src
    assert "l0c2k2addition60stracetrain" in src
    assert "l0c2k2addition60straceheld" in src
    assert "l0c2k2additionheldout50s" in src
    assert "l0c2k2additionheldout60s" in src
    assert "alias-only/non-gating" in src
    assert "TRACE_TRAIN_BANK_GATE" in src
    assert "TRACE_HELD_RECOMBINATION_BANK_GATE" in src

    held_line = "[probe-language] L0C2K2ADDITIONHELDOUT50S AGGREGATE strict=80/80 = 1.0000"
    probe_label = _probe.language_aggregate_label("l0c2k2additionheldout50s")
    assert "LEGACY_ALIAS_ONLY_NON_GATING" in probe_label
    assert "same rows as L0C2K2ADDITION50S" in probe_label
    assert "not transfer" in probe_label
    assert _probe.language_aggregate_label("l0c2k2addition50s") == "L0C2K2ADDITION50S AGGREGATE"
    assert "DIAGNOSTIC_NON_GATING" in _probe.language_aggregate_label("l0c2k2additionheldout60s")

    summary = _watcher._summary_aggregate("l0c2k2additionheldout50s", held_line)
    assert summary.startswith(held_line)
    assert "LEGACY_ALIAS_ONLY_NON_GATING" in summary
    assert "same rows as L0C2K2ADDITION50S" in summary
    assert "not transfer" in summary
    assert _watcher._summary_aggregate("l0c2k2additionheldout50s", summary) == summary

    held60_line = "[probe-language] L0C2K2ADDITIONHELDOUT60S AGGREGATE strict=80/80 = 1.0000"
    held60_summary = _watcher._summary_aggregate("l0c2k2additionheldout60s", held60_line)
    assert "DIAGNOSTIC_NON_GATING" in held60_summary
    assert "not gate" in held60_summary

    trace_line = "[probe-language] L0C2K2ADDITION60STRACETRAIN AGGREGATE exact_trace=60/60 = 1.0000"
    trace_summary = _watcher._summary_aggregate("l0c2k2addition60stracetrain", trace_line)
    assert "TRACE_TRAIN_BANK_GATE" in trace_summary
    assert "not retained/parent-KL" in trace_summary


def test_watcher_l0c2k2_addition_patterns_do_not_cross_match():
    import re
    _watcher, pats, _flags = _watcher_modes()
    legacy = "[probe-language] L0C2K2 AGGREGATE strict=79/79 = 1.0000"
    full = "[probe-language] L0C2K2ADDITIONFULL AGGREGATE strict=240/240 = 1.0000"
    fifties = "[probe-language] L0C2K2ADDITION50S AGGREGATE strict=80/80 = 1.0000"
    trace_train = "[probe-language] L0C2K2ADDITION60STRACETRAIN AGGREGATE exact_trace=60/60 = 1.0000"
    trace_held = "[probe-language] L0C2K2ADDITION60STRACEHELD AGGREGATE exact_trace=20/20 = 1.0000"
    held = "[probe-language] L0C2K2ADDITIONHELDOUT50S AGGREGATE strict=80/80 = 1.0000"
    held60 = "[probe-language] L0C2K2ADDITIONHELDOUT60S AGGREGATE strict=80/80 = 1.0000"

    assert re.search(pats["l0c2k2"], legacy)
    assert re.search(pats["l0c2k2additionfull"], full)
    assert re.search(pats["l0c2k2addition50s"], fifties)
    assert re.search(pats["l0c2k2addition60stracetrain"], trace_train)
    assert re.search(pats["l0c2k2addition60straceheld"], trace_held)
    assert re.search(pats["l0c2k2additionheldout50s"], held)
    assert re.search(pats["l0c2k2additionheldout60s"], held60)
    assert not re.search(pats["l0c2k2"], full)
    assert not re.search(pats["l0c2k2"], fifties)
    assert not re.search(pats["l0c2k2"], trace_train)
    assert not re.search(pats["l0c2k2"], trace_held)
    assert not re.search(pats["l0c2k2"], held)
    assert not re.search(pats["l0c2k2"], held60)
    assert not re.search(pats["l0c2k2additionfull"], legacy)
    assert not re.search(pats["l0c2k2addition50s"], legacy)
    assert not re.search(pats["l0c2k2additionheldout50s"], legacy)
    assert not re.search(pats["l0c2k2additionheldout60s"], legacy)
    assert not re.search(pats["l0c2k2additionfull"], held)
    assert not re.search(pats["l0c2k2additionfull"], fifties)
    assert not re.search(pats["l0c2k2additionheldout50s"], full)
    assert not re.search(pats["l0c2k2additionheldout50s"], fifties)
    assert not re.search(pats["l0c2k2addition50s"], full)
    assert not re.search(pats["l0c2k2addition50s"], held)
    assert not re.search(pats["l0c2k2addition50s"], held60)
    assert not re.search(pats["l0c2k2addition50s"], trace_train)
    assert not re.search(pats["l0c2k2addition50s"], trace_held)
    assert not re.search(pats["l0c2k2addition60stracetrain"], trace_held)
    assert not re.search(pats["l0c2k2addition60straceheld"], trace_train)
    assert not re.search(pats["l0c2k2additionheldout60s"], full)
    assert not re.search(pats["l0c2k2additionheldout60s"], fifties)
    assert not re.search(pats["l0c2k2additionheldout60s"], held)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("watch-row-surfacing tests: PASS")
