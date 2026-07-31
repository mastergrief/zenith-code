#!/usr/bin/env python3
"""LANDS-AB packet dry-exec tool (PLAN_v6 Phase A): packet↔manifest bind + content validation + structural preflight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hex64(val: Any, *, field: str) -> str:
    if not isinstance(val, str) or len(val) != 64 or any(c not in "0123456789abcdef" for c in val):
        raise ValueError(f"{field} must be lowercase 64-hex")
    return val


def _require_hex40(val: Any, *, field: str) -> str:
    if not isinstance(val, str) or len(val) != 40 or any(c not in "0123456789abcdef" for c in val):
        raise ValueError(f"{field} must be 40-hex commit sha")
    return val


PLACEHOLDER_RE = re.compile(r"(\.\.\.|…|<nonce>|<HEAD_A|TODO|FIXME|placeholder)", re.I)

LIVE_BRANCH_IDS = {
    "BR-LANDS-AB-SCOPE-CREEP-STOP",
    "BR-LANDS-AB-FIXTURE-CONTRACT-FAIL",
    "BR-LANDS-AB-VACUOUS",
    "BR-LANDS-AB-DIVERGENT-EVENT",
    "BR-LANDS-AB-DIVERGENT-APPLY",
    "BR-LANDS-AB-DIVERGENT-ORACLE-LIVE",
    "BR-LANDS-AB-EQUIVALENT",
}

GATING_ROWS = [
    "G_CPU_STATIC_AB",
    "G_CUDA_B1_APPLY",
    "G_CUDA_B2_APPLY",
    "G_CUDA_B3_APPLY",
    "G_CUDA_ORACLE_B1",
    "G_CUDA_ORACLE_B2",
    "G_CUDA_ORACLE_B3",
]

# Frozen per-row pytest node mapping from packet_v6 (F2).
CUDA_PYTEST_NODE_BY_ROW = {
    "G_CUDA_B1_APPLY": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_b1_apply_twin_s3_s4_s6"
    ),
    "G_CUDA_B2_APPLY": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_b2_apply_twin_s3_s4_s6"
    ),
    "G_CUDA_B3_APPLY": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_b3_apply_twin_s3_s4_s6"
    ),
    "G_CUDA_ORACLE_B1": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_oracle_b1_events_equal"
    ),
    "G_CUDA_ORACLE_B2": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_oracle_b2_events_equal"
    ),
    "G_CUDA_ORACLE_B3": (
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
        "::test_gpu_live_lands_ab_oracle_b3_events_equal"
    ),
}

REQUIRED_PHASE_BUDGET_NAMES = ("forward_backward", "update", "emission", "flush")
ENFORCER_SCRIPT_SUBSTR = "sparse_live_carrier_gpu_phase_budget_enforcer.py"
NONCE_TOKENS = ("<nonce>", "<run_local_nonce>")

# H2: mandatory formal-execution source set (subset of generator MANDATORY_ALWAYS)
MANDATORY_EXECUTION_SOURCE_SET = (
    "scripts/lands_ab_eval_run.py",
    "scripts/lands_ab_plan_v4_characterization.py",
    "scripts/lands_ab_packet_dry_exec.py",
    "scripts/lands_ab_science_source_manifest.py",
    "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py",
    "bin/watch-wrap",
    "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_twin_apply.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_oracle_sites.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_site_measurement.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_production_post_state.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_production_binding.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_schema.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_branch_reducer.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_fixture_source.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_phase_jsonl.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_phase_topology.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_measurement.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_metric_reducer.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_evidence_contract.py",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_runtime_io.py",
)

# H2: external system binaries allowed as formal argv tokens without manifest pins
EXTERNAL_SYSTEM_BINARY_ALLOWLIST = frozenset({"python3", "timeout"})


def load_runner_parser(repo: Path):
    """Bind to LIVE scripts/lands_ab_eval_run.py ArgumentParser build (no independent emulation).

    Mechanism (actual): AST-extract the ArgumentParser construction statements
    from live main() up to (excluding) parse_args, then exec that prefix under a
    controlled namespace so the returned parser object is built FROM live pinned
    bytes. Live file sha256 is captured as the binding pin; mode choices are
    fail-closed against the live set. Does NOT execute main()'s science body.
    """
    import ast

    path = repo / "scripts" / "lands_ab_eval_run.py"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    src = path.read_text(encoding="utf-8")
    live_sha = sha256_file(path)

    # Extract the exact ArgumentParser construction block from live main() via AST
    # and exec it under a controlled namespace so preflight uses live field names/
    # choices/defaults without running science (main body after parse_args).
    tree = ast.parse(src, filename=str(path))
    main_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_fn = node
            break
    if main_fn is None:
        raise RuntimeError("live runner missing main()")
    # Collect statements from start of main until (and excluding) parse_args call.
    build_stmts: list[ast.stmt] = []
    for stmt in main_fn.body:
        if isinstance(stmt, ast.Assign):
            # stop once we hit args = ap.parse_args(...)
            if any(
                isinstance(t, ast.Name) and t.id == "args" for t in stmt.targets
            ) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute) and func.attr == "parse_args":
                    break
        build_stmts.append(stmt)
        # also break if expression call parse_args
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "parse_args":
                build_stmts.pop()
                break
    if not build_stmts:
        raise RuntimeError("could not extract live ArgumentParser build from main")
    mod = ast.Module(body=build_stmts, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, Any] = {"argparse": argparse}
    code = compile(mod, str(path), "exec")
    exec(code, ns, ns)
    ap = ns.get("ap")
    if not isinstance(ap, argparse.ArgumentParser):
        raise RuntimeError("live main did not bind ArgumentParser to name 'ap'")
    # surface proof: live mode choices present
    actions = {a.dest: a for a in ap._actions if getattr(a, "dest", None)}
    mode = actions.get("mode")
    if mode is None or set(getattr(mode, "choices", ()) or ()) != {
        "cpu-static-ab",
        "cpu-s3-char",
        "reducer-smoke",
    }:
        raise RuntimeError(f"live mode choices mismatch: {getattr(mode, 'choices', None)}")
    return ap, src, live_sha


def _require_watch_wrap_flag_value(argv: list[str], flag: str) -> str:
    if flag not in argv:
        raise ValueError(f"watch-wrap missing {flag}")
    i = argv.index(flag)
    if i + 1 >= len(argv) or not str(argv[i + 1]).strip() or str(argv[i + 1]).startswith("-"):
        raise ValueError(f"watch-wrap {flag} requires non-empty value")
    return str(argv[i + 1])


def _require_finite_positive(val: str, *, field: str) -> float:
    """F4: finite AND >0 (reject 0/negative/inf/nan/non-numeric)."""
    try:
        num = float(val)
    except Exception as e:
        raise ValueError(f"{field} not numeric: {val!r}") from e
    if num != num or num in (float("inf"), float("-inf")) or num <= 0:
        raise ValueError(f"{field} must be finite and >0: {val!r}")
    return num


def _template_norm(path_s: str) -> str:
    """Normalize path string while preserving template tokens as path segments."""
    s = str(path_s).replace("\\", "/")
    for tok in NONCE_TOKENS:
        s = s.replace(tok, tok.strip("<>").upper())  # <nonce> -> NONCE
    # collapse // and resolve . / .. without requiring existence
    parts: list[str] = []
    abs_form = s.startswith("/")
    for part in s.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts and parts[-1] not in ("NONCE", "RUN_LOCAL_NONCE"):
                parts.pop()
            continue
        parts.append(part)
    out = "/".join(parts)
    return ("/" + out) if abs_form else out


def _is_descendant(child: str, parent: str) -> bool:
    c = _template_norm(child)
    p = _template_norm(parent)
    if not p or not c:
        return False
    return c == p or c.startswith(p.rstrip("/") + "/")


def _resolve_under_repo(path_s: str, *, repo: Path) -> tuple[Path, str]:
    """Resolve absolute or relative path; require under repo. Returns (abs_path, rel_posix)."""
    repo_r = repo.resolve()
    p = Path(path_s)
    if p.is_absolute():
        abs_p = p.resolve()
    else:
        abs_p = (repo_r / p).resolve()
    try:
        rel = abs_p.relative_to(repo_r).as_posix()
    except ValueError as e:
        raise ValueError(f"path outside repo: {path_s}") from e
    return abs_p, rel


def _parse_budget_token(tok: str) -> tuple[str, float]:
    if "=" not in tok:
        raise ValueError(f"budget token must be name=value, got {tok!r}")
    name, _, rest = tok.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"empty budget name in {tok!r}")
    val = _require_finite_positive(rest.strip(), field=f"budget[{name}]")
    return name, val


def _flag_occurrences(argv: list[str], flag: str) -> list[int]:
    return [i for i, a in enumerate(argv) if a == flag]


def _require_flag_once_in_open_interval(
    argv: list[str],
    flag: str,
    *,
    lo: int,
    hi: int,
    gr: str,
    segment: str,
) -> tuple[int, str]:
    """G3/G4: flag+value exactly once with both indices in (lo, hi)."""
    idxs = _flag_occurrences(argv, flag)
    if len(idxs) != 1:
        raise ValueError(f"{gr} {flag} must appear exactly once (got {len(idxs)})")
    i = idxs[0]
    if not (lo < i < hi):
        raise ValueError(f"{gr} {flag} must sit inside {segment} segment (index {i} not in ({lo},{hi}))")
    if i + 1 >= hi or i + 1 >= len(argv):
        raise ValueError(f"{gr} {flag} value must sit inside {segment} segment")
    val = str(argv[i + 1])
    if not val.strip() or val.startswith("-"):
        raise ValueError(f"{gr} {flag} requires non-empty non-flag value inside {segment}")
    # ensure no other copy of the value-slot confusion: already unique flag
    return i, val


def _validate_ordered_cuda_chain(argv: list[str], *, gr: str) -> dict[str, Any]:
    """G1–G4 + F1/F2: exact process-segment ownership for CUDA rows."""
    if len(argv) < 10 or argv[0] != "timeout":
        raise ValueError(f"{gr} CUDA argv must start with timeout <budget>")
    _require_finite_positive(argv[1], field=f"{gr} outer timeout")

    ww_idxs = [
        i
        for i, a in enumerate(argv)
        if a == "bin/watch-wrap" or a.endswith("/watch-wrap") or a.endswith("watch-wrap")
    ]
    if not ww_idxs:
        raise ValueError(f"{gr} missing bin/watch-wrap")
    if len(ww_idxs) != 1:
        raise ValueError(f"{gr} watch-wrap must appear exactly once")
    ww_i = ww_idxs[0]
    if ww_i != 2:
        raise ValueError(f"{gr} watch-wrap must immediately follow timeout <budget> (index 2), got {ww_i}")

    # G1: exactly two `--` delimiters
    dash_idxs = [i for i, a in enumerate(argv) if a == "--"]
    if len(dash_idxs) != 2:
        raise ValueError(f"{gr} CUDA chain requires exactly two `--` delimiters, got {len(dash_idxs)}")
    d0, d1 = dash_idxs[0], dash_idxs[1]
    if not (ww_i < d0 < d1):
        raise ValueError(f"{gr} delimiter order invalid vs watch-wrap index")

    # G2: first child begins exactly python3 <enforcer_script>
    if d0 + 2 >= d1:
        raise ValueError(f"{gr} enforcer child segment too short")
    if argv[d0 + 1] != "python3":
        raise ValueError(f"{gr} enforcer child must begin with python3 immediately after first `--`")
    enf_tok = argv[d0 + 2]
    if enf_tok != f"scripts/{ENFORCER_SCRIPT_SUBSTR}":
        raise ValueError(
            f"{gr} enforcer child must be exactly scripts/{ENFORCER_SCRIPT_SUBSTR} immediately after python3, got {enf_tok!r}"
        )
    enf_i = d0 + 2

    # G2: second child begins exactly python3 -m pytest <frozen-node>
    if d1 + 4 > len(argv):
        raise ValueError(f"{gr} pytest child segment too short")
    if argv[d1 + 1] != "python3":
        raise ValueError(f"{gr} pytest child must begin with python3 immediately after second `--`")
    if argv[d1 + 2] != "-m" or argv[d1 + 3] != "pytest":
        raise ValueError(f"{gr} pytest child must be `python3 -m pytest` immediately after second `--`")
    node = str(argv[d1 + 4])
    if not node.strip() or node.startswith("-"):
        raise ValueError(f"{gr} missing pytest node after `python3 -m pytest`")
    expected = CUDA_PYTEST_NODE_BY_ROW.get(gr)
    if expected is None:
        raise ValueError(f"{gr} not in frozen CUDA pytest node map")
    if node != expected:
        raise ValueError(f"{gr} pytest node {node!r} != frozen {expected!r}")
    m_i = d1 + 2

    # G3: watch-wrap flags only in (ww_i, d0)
    for fl in ("--error", "--progress", "--success", "--heartbeat"):
        _require_flag_once_in_open_interval(argv, fl, lo=ww_i, hi=d0, gr=gr, segment="watch-wrap")
    hb = _require_flag_once_in_open_interval(argv, "--heartbeat", lo=ww_i, hi=d0, gr=gr, segment="watch-wrap")[1]
    _require_finite_positive(hb, field=f"{gr} --heartbeat")

    # G4: enforcer flags/budgets only in (d0, d1)
    phase_argv = _require_flag_once_in_open_interval(
        argv, "--phase-events-jsonl", lo=d0, hi=d1, gr=gr, segment="enforcer"
    )[1]
    enf_argv = _require_flag_once_in_open_interval(
        argv, "--enforcer-receipt", lo=d0, hi=d1, gr=gr, segment="enforcer"
    )[1]
    node_id = _require_flag_once_in_open_interval(
        argv, "--expected-node-id", lo=d0, hi=d1, gr=gr, segment="enforcer"
    )[1]
    if node_id != gr:
        raise ValueError(f"{gr} --expected-node-id must equal gating_row")

    budget_idxs = _flag_occurrences(argv, "--budget")
    if len(budget_idxs) != 4:
        raise ValueError(f"{gr} requires exactly four --budget flags, got {len(budget_idxs)}")
    budget_map: dict[str, float] = {}
    for bi in budget_idxs:
        if not (d0 < bi < d1) or bi + 1 >= d1:
            raise ValueError(f"{gr} --budget must sit entirely inside enforcer segment")
        bname, bval = _parse_budget_token(str(argv[bi + 1]))
        if bname in budget_map:
            raise ValueError(f"{gr} duplicate phase budget name {bname!r}")
        budget_map[bname] = bval
    if set(budget_map) != set(REQUIRED_PHASE_BUDGET_NAMES):
        raise ValueError(
            f"{gr} phase budget names {sorted(budget_map)} != {list(REQUIRED_PHASE_BUDGET_NAMES)}"
        )

    # no enforcer/watch flags leaked into other segments (uniqueness already enforced)
    return {
        "watch_wrap_i": ww_i,
        "dash0": d0,
        "dash1": d1,
        "enforcer_i": enf_i,
        "pytest_m_i": m_i,
        "pytest_node": node,
        "phase_argv": phase_argv,
        "enf_argv": enf_argv,
        "budget_map": budget_map,
    }


def _row_hard_timeout_seconds(rc: dict, inv: dict) -> float | None:
    """G6 hard-bound selection: inv.hard_timeout_seconds, else row.hard_timeout_seconds, else row.duration_budget_seconds."""
    for src in (inv.get("hard_timeout_seconds"), rc.get("hard_timeout_seconds"), rc.get("duration_budget_seconds")):
        if src is None or src == "":
            continue
        return float(src)
    return None


def _is_repo_path_token(tok: str) -> bool:
    """True if argv token is a repo-relative path that must be manifest-covered (H2)."""
    if not tok or tok in EXTERNAL_SYSTEM_BINARY_ALLOWLIST:
        return False
    if tok.startswith("-"):
        return False
    if tok.startswith("/"):
        return False  # absolute runtime/tmp paths are not science-source pins
    # template tokens in runtime paths
    if "<nonce>" in tok or "<run_local_nonce>" in tok:
        return False
    # common repo path shapes
    if tok.startswith(("scripts/", "calm/", "bin/", "artifacts/")):
        return True
    if tok.endswith((".py", ".sh")) and "/" in tok:
        return True
    if "::" in tok and tok.startswith("calm/"):
        # pytest nodeid calm/.../file.py::test
        return True
    return False


def _repo_path_from_token(tok: str) -> str:
    """Normalize pytest nodeid to file path for coverage."""
    if "::" in tok:
        return tok.split("::", 1)[0]
    return tok


def structural_preflight_row(argv: list[str], *, repo: Path, runner_parser) -> None:
    if not argv or not isinstance(argv, list) or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("row argv must be non-empty list[str]")
    joined = " ".join(argv)
    # strip allowed path template tokens before placeholder ban
    tmp = joined
    for tok in ("<nonce>", "<run_local_nonce>"):
        tmp = tmp.replace(tok, "NONCE")
    if "..." in tmp or "…" in tmp or "TODO" in tmp.upper() or "FIXME" in tmp.upper() or "placeholder" in tmp.lower():
        raise ValueError(f"placeholder/ellipsis in argv: {argv}")
    # CPU runner rows
    if any(a.endswith("lands_ab_eval_run.py") or a == "scripts/lands_ab_eval_run.py" for a in argv):
        try:
            idx = next(i for i, a in enumerate(argv) if a.endswith("lands_ab_eval_run.py"))
        except StopIteration:
            raise ValueError("runner path not found")
        runner_args = argv[idx + 1 :]
        runner_parser.parse_args(runner_args)
        return
    # CUDA/watch-wrap structure-only (D3.8)
    if "bin/watch-wrap" in argv or any(a.endswith("watch-wrap") for a in argv):
        err = _require_watch_wrap_flag_value(argv, "--error")
        prog = _require_watch_wrap_flag_value(argv, "--progress")
        # require timeout/hard-bound and success/stop semantics when present as flags
        for fl in ("--timeout", "--hard-timeout", "--success", "--stop-on"):
            if fl in argv:
                _require_watch_wrap_flag_value(argv, fl)
        if not err.strip() or not prog.strip():
            raise ValueError("watch-wrap empty error/progress")
        return



HEX40_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])")
# Q3: any isolated hex40/hex64 token, whether standalone leaf or embedded in prose.
# 64 first so the longer token wins. Callers lowercase before matching.
HEX_TOKEN_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{64}|[0-9a-f]{40})(?![0-9a-f])")
def _require_packet_bool(value: Any, *, field: str, where: str) -> bool:
    """Q13: a packet-supplied ASSERTION field must be a JSON boolean. TYPE FIRST, then
    value -- never identity alone.

    An identity test (`v is True`) silently skips every non-bool, so `1`, `1.0`,
    `"true"`, `"True"`, `[1]` all read as "no assertion" to the validator while any
    consumer evaluating truthiness reads them as asserting one. That divergence between
    the validator's guarantee and a consumer's reading is the failure this rejects.

    Strict bool is the only COHERENT rule here, and the deciding case is `"false"`:
    in Python a non-empty string is TRUTHY, so a truthiness-based rule would raise
    "claims non-source head" on a packet that literally said false, while skipping the
    falsy `0` -- two semantically identical negations treated oppositely by Python
    truthiness rather than by what the packet means.

    Declared strictness: present-but-`null` and `0` RAISE rather than reading as absent.
    In an assertion field those are far more likely an authoring defect than an
    intentional non-claim, and the fail-closed direction is the safe one. Absence and
    literal `false` remain clean -- a packet is entitled to assert nothing.

    `isinstance(v, bool)` is correct despite `bool` subclassing `int`: literal `1` is
    `int`, not `bool`, so it is rejected.
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"I1/Q13 {where} assertion field {field!r} must be a JSON boolean "
            f"(true/false), got {type(value).__name__} {value!r}. Assertion fields are "
            f"read for truthiness by consumers, so a non-boolean would let the "
            f"validator and a consumer disagree about whether a claim was made."
        )
    return value


# Q11: maximal [0-9a-f] runs of >= 7 chars inside a `self_check` `pinned_to_*` key.
# 7 is git's shortest conventional head abbreviation, so anything at or above it is
# a head-shaped claim. Deliberately NO lookbehind/lookahead guard: an adjacent
# non-hex letter (e.g. `..._xdeadbeefcafe`) must not shield the run from the check,
# and greedy matching already yields maximal runs.
_PIN_HEX_TOKEN_RE = re.compile(r"[0-9a-f]{7,}")
# A git commit head is at most 40 hex chars, so a longer run under `pinned_to_*`
# cannot be a head under any abbreviation -- it is a head-only contract violation
# rather than a wrong head. Classified by LENGTH ALONE, never by membership in the
# reference set: coupling this diagnostic to `known_refs` would reintroduce
# name-conferred trust from the exemption side.
_PIN_HEAD_MAX_LEN = 40
# Packet revision tokens only (underscore-safe: dry_exec_v16 is NOT a packet rev).
PACKET_REV_TOKEN_RE = re.compile(
    r"packet_v(\d+)(?![0-9])|(?<![A-Za-z0-9_])v(\d+)(?![A-Za-z0-9_])",
    re.I,
)
# Exact operative plan for this LANDS-AB consumer-adapt family (L3).
EXPECTED_OPERATIVE_PLAN_REL = (
    "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
    "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v6.json"
)
EXPECTED_OPERATIVE_PLAN_SHA256 = (
    "e421aecdf1cc4b9a94d118a0563e7d8ac8516f97da998d38af3b8c60ac88a41c"
)
# M/O-series: key hints may IDENTIFY a candidate lineage object, never grant DEAD.
# DEAD exemption is schema-based + CONJUNCTIVE + identity-bound (O1).
LINEAGE_KEY_HINTS = (
    "dead",
    "lineage",
    "historical",
    "superseded",
    "packet_v1",
    "packet_v2",
    "packet_v3",
    "packet_v4",
    "packet_v5",
    "packet_v6",
    "packet_v7",
    "implement_v16",
    "implement_v17",
    "snapshot_v16",
    "snapshot_v17",
    "parent_structural_null",
    "parent_null",
)
# Independent status markers (do_not_activate is NEVER a status marker — M2/M3).
_INDEPENDENT_DEAD_STATUS = frozenset(
    {
        "dead",
        "dead_immutable",
        "historical",
        "superseded",
    }
)
_PACKET_V_IN_KEY_RE = re.compile(r"packet_v(\d+)", re.I)
# Q1: CLOSED set of pins locations where a lineage identity may be DECLARED.
# This RESTRICTS candidate locations (fail-closed); it never GRANTS exemption —
# the grant comes from disk binding below. Restriction-by-allowlist and
# grant-by-hint are opposite directions: an unlisted key is simply live-scanned.
_LINEAGE_PIN_KEY_ALLOWED = (
    re.compile(r"^packet_v\d+_dead_lineage$", re.I),
    re.compile(r"^lineage_policy_packet_v\d+$", re.I),
    re.compile(r"^historical_eval_PLAN_v\d+$", re.I),
    re.compile(r"^historical_packet_fixture_notes$", re.I),
)
# Q2/Q7: only a lineage object's DISK-VERIFIED identity (`path` + `sha256`) may seed
# the known-reference set. No field is trusted by key name — key-name-shaped trust
# was the recurring injection route. See `_collect_validated_lineage_identities`.


def _is_allowed_lineage_pin_key(key: str) -> bool:
    """Q1 closed-location gate for lineage identity DECLARATION sites."""
    k = str(key).strip()
    return any(rx.match(k) for rx in _LINEAGE_PIN_KEY_ALLOWED)


# Q4: an admitted lineage pin may contain ONLY these fields, so a disk-verified
# pin cannot also smuggle freeform live content behind its own exemption.
_LINEAGE_PIN_SCHEMA_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "status",
        "do_not_activate",
        "dead_lineage",
        "historical",
        "superseded",
        "block_msg_id",
        "defect",
        "do_not_modify_bytes",
    }
)
_LINEAGE_PIN_SCHEMA_FIELD_RE = re.compile(
    r"^commit_as_lineage_with_packet_v\d+$", re.I
)
# The SINGLE named freeform-historical carrier. Its identity is still disk-bound;
# it is exempt from the Q4 field-schema check only. RESIDUAL (recorded, not hidden):
# freeform content inside this one location is not token-scanned. Preferred
# long-term fix is to move that prose into the referenced on-disk artifact so the
# packet carries only (path, sha256) — recommended for packet_v8.
_FREEFORM_LINEAGE_CARRIER_KEYS = frozenset({"historical_packet_fixture_notes"})


def _lineage_pin_schema_violations(pk: str, pv: dict) -> list[str]:
    """Q4: unknown fields inside an admitted (non-carrier) lineage pin."""
    if str(pk).strip().lower() in _FREEFORM_LINEAGE_CARRIER_KEYS:
        return []
    bad: list[str] = []
    for k in pv.keys():
        ks = str(k)
        if ks in _LINEAGE_PIN_SCHEMA_FIELDS:
            continue
        if _LINEAGE_PIN_SCHEMA_FIELD_RE.match(ks):
            continue
        bad.append(ks)
    return sorted(bad)


def _is_lineage_key_hint(key: str) -> bool:
    """Candidate detector only — NEVER grants DEAD exemption (M1)."""
    kl = str(key).lower()
    return any(h in kl for h in LINEAGE_KEY_HINTS)


# Back-compat alias; not an exemption grant.
#
# INTENTIONALLY UNUSED -- verified zero callers. Do NOT wire this (or
# `_is_lineage_key_hint`) into any exemption, admission, or skip path. Despite the
# name, it decides by SUBSTRING KEY-NAME MATCH, which is exactly the
# trust-conferred-by-key-name class removed across Q5-Q11: a key whose name merely
# mentions a dead revision proves nothing about the VALUE it carries. DEAD lineage is
# determined from independent status on the object itself -- see
# `_has_independent_dead_status` (used at the two live sites) -- and the M1 hostiles
# lock that behaviour. Kept rather than deleted so this warning survives at the
# location where the mistake would be attempted.
def _is_dead_lineage_key(key: str) -> bool:
    return _is_lineage_key_hint(key)


def _normalize_status_token(raw: Any) -> str:
    return str(raw or "").lower().replace(" ", "_").replace("-", "_")


def _has_independent_dead_status(obj: dict) -> bool:
    """True only for independent status/typed markers — never do_not_activate alone (M2/M3)."""
    if obj.get("dead_lineage") is True:
        return True
    if obj.get("historical") is True:
        return True
    if obj.get("superseded") is True:
        return True
    st = _normalize_status_token(obj.get("status"))
    return st in _INDEPENDENT_DEAD_STATUS


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").lstrip("./")


def _canonical_repo_relpath(rel: str, *, repo: Path) -> str | None:
    """
    Q5: return the repo-relative path ONLY when `rel` is already canonical and
    contained. Returns None (fail-closed) otherwise.

    Both legs matter for reference injection:
      * containment — `artifacts/../../../tmp/x.json` still satisfies
        `startswith("artifacts/")`, so a bare prefix test lets a pin bind a file
        OUTSIDE the repo and thereby seed the known-reference set with its path
        string and sha (demonstrated escape: a foreign `..._PLAN_v3.json` path
        admitted past L2/J4).
      * canonicality — a path that resolves inside the repo but carries `..`
        segments is an ALIAS of a real file, so accepting it would let one file
        legitimize many distinct reference strings.
    """
    rel_n = _norm_path(rel)
    if not rel_n or rel_n.startswith("/"):
        return None
    if Path(rel_n).is_absolute():
        return None
    parts = rel_n.split("/")
    if any(seg in ("", ".", "..") for seg in parts):
        return None
    repo_r = repo.resolve()
    fp = (repo_r / rel_n).resolve()
    try:
        got = fp.relative_to(repo_r).as_posix()
    except ValueError:
        return None
    if got != rel_n:
        return None
    return got


def _collect_validated_lineage_identities(
    pins: dict[str, Any], *, rev_norm: str, repo: Path
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    """
    Q1: lineage identity is DISK-BOUND, not self-asserted.

    An identity is admitted only when ALL of these hold:
      (a) declared at a CLOSED allowlisted pins location (restriction, not grant),
      (b) do_not_activate is True,
      (c) an independent status/typed DEAD marker (never do_not_activate itself),
      (d) path + sha256 present, sha256 is hex64,
      (e) the path resolves UNDER the repo AND sha256(disk bytes) == declared sha256.

    (e) is the load-bearing leg: a packet author cannot mint authority for an
    artifact that does not exist with exactly those bytes. Only typed reference
    fields of an admitted object seed the known-reference set (Q2).
    """
    identities: set[tuple[str, str]] = set()
    refs: set[str] = set()
    exempt_paths: set[str] = set()
    if not isinstance(pins, dict):
        return identities, refs, exempt_paths
    repo_res = repo.resolve()
    for pk, pv in pins.items():
        if not isinstance(pv, dict):
            continue
        if not _is_allowed_lineage_pin_key(str(pk)):
            continue
        if pv.get("do_not_activate") is not True:
            continue
        if not _has_independent_dead_status(pv):
            continue
        path = pv.get("path")
        sha = pv.get("sha256")
        if not isinstance(path, str) or not isinstance(sha, str):
            continue
        path_n = _norm_path(path)
        sha_n = sha.lower()
        # non-current revision identity when packet_vN is present
        m = _PACKET_V_IN_KEY_RE.search(str(pk)) or _PACKET_V_IN_KEY_RE.search(path_n)
        if m:
            other = m.group(1)
            if other == rev_norm:
                continue
        try:
            sha_n = _require_hex64(sha_n, field=f"pins.{pk}.sha256")
        except ValueError:
            continue
        # (e) disk binding — canonical under-repo resolution + exact byte hash.
        # Q5: canonicality is required too, so one artifact cannot legitimize
        # multiple aliased reference strings via `..` segments.
        canon = _canonical_repo_relpath(path_n, repo=repo_res)
        if canon is None:
            continue
        path_n = canon
        fp = repo_res / canon
        if not fp.is_file():
            continue
        try:
            if sha256_file(fp) != sha_n:
                continue
        except OSError:
            continue
        # Q4: a disk-verified pin must not also carry freeform live content.
        bad_fields = _lineage_pin_schema_violations(str(pk), pv)
        if bad_fields:
            raise ValueError(
                f"Q4 lineage pin 'pins.{pk}' carries non-schema field(s) "
                f"{bad_fields} — exemption covers the closed lineage schema only"
            )
        identities.add((path_n, sha_n))
        exempt_paths.add(f"pins.{pk}")
        # Q6/Q7: seed the known-reference set from the pin's DISK-VERIFIED IDENTITY
        # ONLY — the `path` and `sha256` that just passed leg (e). Every other field
        # of a pin is self-asserted and unverified, so trusting it lets an admitted
        # pin introduce arbitrary values into the reference set and thereby exempt
        # unrelated LIVE content elsewhere in the tree.
        #
        # This is ONE rule for ALL admitted pins, not a carrier special case. Two
        # demonstrated escape routes motivated collapsing them:
        #   * the Q4-schema-exempt freeform carrier could add any `*_path` /
        #     `*_sha256` field, so unscanned content inside it became a way to
        #     exempt live content outside it;
        #   * `block_msg_id` is BOTH schema-allowed on ordinary pins AND
        #     reference-shaped by key name, so it admitted an arbitrary hex64 that
        #     then exempted a live token. A room message id is not a content hash
        #     and must never bind one.
        # Key-name-shaped trust was the recurring defect; disk-verified identity is
        # the only thing that earns reference authority.
        for _v in (path_n, str(path), sha_n):
            if isinstance(_v, str) and _v.strip():
                refs.add(_v.strip())
                refs.add(_v.strip().lower())
    return identities, refs, exempt_paths


def _enter_dead(
    child_path: str,
    parent_dead: bool,
    *,
    exempt_paths: set[str],
) -> bool:
    """
    Q1': exemption is LOCATION-bound, never value-bound.

    A (path, sha256) pair is copyable, so deciding exemption by pair membership
    lets any object anywhere inherit a legitimate pin's exemption by restating its
    pair. Only the exact tree locations admitted by
    `_collect_validated_lineage_identities` (and their descendants) are exempt.
    """
    if parent_dead:
        return True
    return child_path in exempt_paths


def _walk_collect_hex40(
    obj: Any,
    *,
    path: str,
    out: list[tuple[str, str, bool, str]],
    dead: bool = False,
    exempt_paths: set[str] | None = None,
) -> None:
    idset = exempt_paths if exempt_paths is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            _walk_collect_hex40(
                v,
                path=p,
                out=out,
                dead=_enter_dead(p, dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_collect_hex40(
                v,
                path=f"{path}[{i}]",
                out=out,
                dead=_enter_dead(f"{path}[{i}]", dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, str):
        for m in HEX40_RE.finditer(obj.lower()):
            out.append((path, m.group(1), dead, obj))


def _walk_collect_hex_tokens(
    obj: Any,
    *,
    path: str,
    out: list[tuple[str, str, bool]],
    dead: bool = False,
    exempt_paths: set[str] | None = None,
) -> None:
    """Q3: collect every isolated hex40/hex64 token, standalone or embedded in prose."""
    idset = exempt_paths if exempt_paths is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            _walk_collect_hex_tokens(
                v,
                path=p,
                out=out,
                dead=_enter_dead(p, dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_collect_hex_tokens(
                v,
                path=f"{path}[{i}]",
                out=out,
                dead=_enter_dead(f"{path}[{i}]", dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, str):
        for m in HEX_TOKEN_RE.finditer(obj.lower()):
            out.append((path, m.group(1), dead))


def _walk_strings_dead_aware(
    obj: Any,
    *,
    path: str,
    out: list[tuple[str, str, bool]],
    dead: bool = False,
    exempt_paths: set[str] | None = None,
) -> None:
    idset = exempt_paths if exempt_paths is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            _walk_strings_dead_aware(
                v,
                path=p,
                out=out,
                dead=_enter_dead(p, dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_strings_dead_aware(
                v,
                path=f"{path}[{i}]",
                out=out,
                dead=_enter_dead(f"{path}[{i}]", dead, exempt_paths=idset),
                exempt_paths=idset,
            )
    elif isinstance(obj, str):
        out.append((path, obj, dead))


def _validate_i_series_consistency(
    packet: dict[str, Any],
    *,
    man_path_set: set[str],
    man_sha_by_path: dict[str, str],
    source_commit: str,
    repo: Path,
    expected_operative_plan_path: str = EXPECTED_OPERATIVE_PLAN_REL,
    expected_operative_plan_sha256: str = EXPECTED_OPERATIVE_PLAN_SHA256,
) -> None:
    """I1–I5 + J1–J5 + L1–L3 + O1/O2 structural authority vs HEAD_A binding + manifest."""
    src = source_commit.lower()

    # ---- I3 + L1 structured packet authority (NO prose/synonym vocabulary) ----
    # Resolve revision first so O1 identities can require non-current revision.
    rev = str(packet.get("packet_revision") or "").strip()
    if not rev:
        raise ValueError("I3 packet_revision required")
    rev_norm = rev.lower().lstrip("v")
    opr = packet.get("operative_packet_revision")
    if opr is None or str(opr).strip() == "":
        raise ValueError(
            "L1 operative_packet_revision required (must equal packet_revision)"
        )
    opr_norm = str(opr).strip().lower().lstrip("v")
    if opr_norm != rev_norm:
        raise ValueError(
            f"L1 operative_packet_revision={opr!r} must equal packet_revision={rev!r}"
        )
    path_s = str(packet.get("path") or "")
    if path_s:
        if rev_norm and f"v{rev_norm}" not in path_s.lower() and rev.lower() not in path_s.lower():
            raise ValueError(
                f"I3 path does not match packet_revision: path={path_s!r} rev={rev!r}"
            )

    # ---- pins required early for O1 identities + I2 ----
    pins = packet.get("pins")
    if not isinstance(pins, dict):
        raise ValueError("I2 pins object required for pin-vs-manifest binding")

    # J1 first: every packet_vN lineage pin must dna + independent status + path/sha256 identity.
    lineage_key_re = re.compile(r"packet_v(\d+)", re.I)
    for pk, pv in pins.items():
        m = lineage_key_re.search(str(pk))
        if not m:
            path_hint = ""
            if isinstance(pv, dict):
                path_hint = str(pv.get("path") or "")
            m = lineage_key_re.search(path_hint) or re.search(
                r"launch_packet_v(\d+)", path_hint, re.I
            )
            if not m:
                continue
        other = m.group(1)
        if other == rev_norm:
            continue
        if not isinstance(pv, dict):
            raise ValueError(
                f"J1 packet lineage pin {pk!r} must be object with do_not_activate=true"
            )
        if pv.get("do_not_activate") is not True:
            raise ValueError(
                f"J1 packet lineage pin {pk!r} (v{other}) missing do_not_activate=true"
            )
        if not _has_independent_dead_status(pv):
            raise ValueError(
                f"J1 packet lineage pin {pk!r} missing independent DEAD/historical status "
                f"(do_not_activate alone is insufficient)"
            )
        # O1 identity leg on pin objects themselves
        if not isinstance(pv.get("path"), str) or not isinstance(pv.get("sha256"), str):
            raise ValueError(
                f"J1/O1 packet lineage pin {pk!r} requires path+sha256 identity fields"
            )
        try:
            _require_hex64(str(pv["sha256"]).lower(), field=f"pins.{pk}.sha256")
        except ValueError as e:
            raise ValueError(f"J1/O1 {e}") from e

    # O1/O2: validated lineage identities + exact reference value set
    lineage_identities, known_refs, exempt_pin_paths = _collect_validated_lineage_identities(
        pins, rev_norm=rev_norm, repo=repo
    )
    # ---- O2/Q8: authority_chain values are NOT trusted by KEY NAME ----
    # A value here joins the known-reference set ONLY when it is independently
    # verifiable on its own terms: the family-frozen operative plan identity (path
    # or sha256), or the source commit this packet is being validated against.
    #
    # Being named `plan_sha256` / `plan_path` / `head_a` confers nothing. The prior
    # code seeded exactly those five keys UNCONDITIONALLY — no disk binding, no
    # comparison against the operative plan — and three of them are bound by no
    # later check, so a changed value was admitted and then exempted LIVE content
    # elsewhere in the tree. Measured rc=0 with rc=2 controls (g1/g2/g3).
    #
    # This is the SAME defect class as Q7 on pins (trust conferred by key name),
    # on the one route the Q7 unification did not cover. The rule is now uniform
    # across both routes: only independently verifiable values earn reference
    # authority. Note the previous comment here claimed this route "cannot teach
    # the known-reference set" — it did; a comment asserting a property the code
    # lacks is how this survived several rounds, so the invariant is now enforced
    # by the code below rather than described above it.
    auth_pre = packet.get("authority_chain")
    _seed_known_refs_from_authority_chain(
        known_refs,
        auth_pre,
        expected_operative_plan_path=expected_operative_plan_path,
        expected_operative_plan_sha256=expected_operative_plan_sha256,
        src=src,
    )
    # ---- Q9: harvest is strictly DOWNSTREAM of validation ----
    # The operative-plan family (`operative_plan_id` / `_sha256` and the
    # `operative_adaptation_*` alternative naming) is NO LONGER harvested from the
    # raw packet. L3/J5 requires the resolved operative id/sha to equal
    # expected_plan_rel / expected_plan_sha EXACTLY, so a legitimate packet's values
    # are byte-identical to the frozen constants already admitted just above — the
    # raw harvest was redundant when legitimate and was the admission surface when
    # not. Removing all four names (not just the two that lacked a reaching check)
    # keeps the two namings symmetric by construction instead of equalising them.
    #
    # `path` and `source_commit_sha` stay: each has a reaching named check (I3
    # revision-match, and the 40-hex + source-commit compare respectively).
    for k in ("path", "source_commit_sha"):
        v = packet.get(k)
        if isinstance(v, str) and v.strip():
            known_refs.add(v.strip())
            known_refs.add(v.strip().lower())
    # expected operative plan frozen constants always known
    known_refs.add(EXPECTED_OPERATIVE_PLAN_REL)
    known_refs.add(EXPECTED_OPERATIVE_PLAN_SHA256)

    # O2: expand known-reference value set from validated path+sha pins / maps (exact values only).
    # _add_ref / _disk_sha_ok / _check_path_sha are module-level (A1 Tier-2 lift).
    # pin/map expansion is module-level (A2 P2 pure peel).
    _expand_known_refs_from_validated_pins(
        known_refs,
        pins,
        exempt_pin_paths=exempt_pin_paths,
        man_path_set=man_path_set,
        man_sha_by_path=man_sha_by_path,
        repo=repo,
        packet=packet,
    )

    # NOTE (Q3'): the universal Q3/I1 hex-token backstop runs LAST, after every
    # specific structural check below. It is a catch-all, and a catch-all that ran
    # first would preempt the specific diagnoses (I2 / L1 / I5 / L3 / J4 / L2) for
    # any mutation that also happens to introduce a foreign token — masking whether
    # those checks still work at all. Ordering is safe because `known_refs` and
    # `exempt_pin_paths` are FINAL before this point (seeded only above); every
    # later use is a read-only membership test, so deferring the scan cannot widen
    # what it accepts. See `_q3_i1_unbound_token_backstop` at the end of this
    # function.

    # boolean self_check pins claiming an old head
    #
    # Q11: the trigger is a RULE, not an enumeration. The retired form fired only on
    # an exact hex40 token or one of two hardcoded literals ("a258f314"/"95097a8d"),
    # so the shorthand git actually writes for a head (7-12 chars) satisfied neither
    # and passed silently -- a narrow trigger standing in for the invariant. A
    # `pinned_to_*` key asserting True is a claim ABOUT THE SOURCE HEAD, so every
    # hex-shaped token it carries must be a prefix of `src`.
    #
    # PREFIX, not containment: the retired `suffix not in src` leg skipped any suffix
    # that appeared anywhere in `src`, so a run matching mid-sha was accepted. A
    # mid-sha run is not a legitimate abbreviation of a head.
    #
    # Both retired literals remain caught by the rule (each is a non-prefix hex run
    # of length >= 7), so deleting the enumeration loses no coverage -- see the
    # control hostiles in the test module.
    #
    # Deliberate fail-CLOSED tradeoff: a prose token that happens to be spelled from
    # [0-9a-f] and is >= 7 chars (e.g. "acceded") would raise here. That direction is
    # safe (it rejects rather than admits) and no faithful key is affected -- the
    # faithful packet's only pin key is `pinned_to_source_commit`, whose longest
    # contiguous [0-9a-f] run is 2 chars. Claim-integrity only: nothing in this block
    # enters `known_refs`.
    #
    # CONTRACT (explicit, so the strict rule is intentional and not accidental):
    # `pinned_to_*` is HEAD-ONLY. The prefix asserts "this packet is pinned to source
    # head <x>", so the only legitimate payload is the source commit or an
    # abbreviation of it. A legitimate NON-head sha under this prefix -- e.g.
    # `pinned_to_manifest_<manifest sha256>` -- is malformed BY DESIGN, not merely
    # unrecognised, and gets its own diagnostic below so that a later round does not
    # "repair" a confusing rejection by widening the rule. Non-head values are pinned
    # through the typed path/sha structures that carry disk binding, not through a
    # boolean key name. That distinction is the whole point: a boolean key cannot be
    # verified against disk, so it may only restate a value verified elsewhere.
    # Q14a (gate-2 finding F1): `self_check` must BE a mapping. The retired form guarded
    # entry with `isinstance(sc, dict)` and had no else-branch, so `self_check: [true]` or
    # `self_check: "all_good"` skipped the entire block silently -- a packet could drop
    # every assertion this section enforces just by changing the container type.
    sc = packet.get("self_check")
    if sc is not None and not isinstance(sc, dict):
        raise ValueError(
            f"I1/Q14 self_check must be a JSON object, got {type(sc).__name__}. A "
            f"non-mapping self_check would skip every pin and assertion check in this "
            f"section rather than failing, so it is malformed by contract."
        )
    if isinstance(sc, dict):
        for k, v in sc.items():
            kl = str(k).lower()
            # Q13a (blocker B1v11-P2a), depth: REJECT non-scalar, do not recurse.
            # `sc.items()` walks one level, so a pin-shaped key nested under
            # `self_check.nested`, `self_check.nested[0]` or `self_check.a.b` was never
            # reached. Rejecting a non-scalar value closes all of those WITHOUT a
            # traversal to write or get wrong, and it is fail-closed. Nested
            # `self_check` would be a schema change with its own gate, not something
            # this guard should silently enable.
            if isinstance(v, (dict, list)):
                raise ValueError(
                    f"I1/Q13 self_check value for {k!r} must be a scalar, got "
                    f"{type(v).__name__}. Nested self_check containers are not part of "
                    f"the schema and would hide pin-shaped keys from this check."
                )
            # Q14b (gate-2 finding F1): the assertion contract covers the WHOLE mapping,
            # not just pin-shaped keys. This validation is hoisted ABOVE the pin branch
            # because the retired order reached `continue` first for every non-pin key,
            # so `expected_branch_classifier_determined: "false"` and
            # `some_assertion: 0` / `: null` were admitted unvalidated. Every
            # `self_check` entry is now exactly a JSON bool, or the packet is malformed.
            #
            # Q13a: a pin is an ASSERTION field, so it must be a real boolean. The
            # retired `v is not True` identity test skipped 1 / 1.0 / "true" / "True" /
            # "yes" / "1" / [1] as "not asserted", which reopened the pre-marker route
            # by a different dimension.
            _require_packet_bool(v, field=str(k), where="self_check")
            # Q14c (gate-2 finding F2, cure B): ADMISSION no longer depends on marker
            # POSITION. The retired predicate required `pinned_to_` (with trailing
            # underscore) or a `pinned_to` prefix, so `deadbeefcafe_pinned_to` -- marker
            # at the END, carrying a foreign head -- was discharged unscanned. F2 is an
            # ADMISSION defect, not a spelling defect: the Q12 whole-key scan already
            # judges position correctly once a key is admitted, so the cure removes the
            # exemption rather than adding a second position contract that would have to
            # agree with the scan. (A marker-at-start GRAMMAR was the originally
            # dispatched cure; it was superseded because it rejects
            # `<src_prefix>_pinned_to_source_commit`, a legitimate claim and a frozen
            # acceptance positive. Ratified at 1785415760948-52adea6f.)
            if "pinned_to" not in kl:
                continue
            # Q14d (option 2): a key whose name is ONLY the marker asserts pinning with
            # no referent. Under the assertion-field contract enforced mapping-wide
            # above, a claim that names nothing is malformed by the same logic that makes
            # a non-bool value malformed. Closes the one row the superseded grammar would
            # have caught, so admission-based cure (B) is strictly no weaker than (A).
            if not kl.replace("pinned_to", "").strip("_"):
                raise ValueError(
                    f"I1/Q14 self_check pin key {k!r} names no referent: the key is only "
                    f"the pin marker, so it asserts pinning without identifying a head. "
                    f"Use pinned_to_<head-or-descriptor>."
                )
            if v is not True:
                continue
            # Q12 (blocker B1v10-P1): scan the WHOLE key, never a post-marker
            # remainder. The retired form was `kl.split("pinned_to", 1)[-1]`, which
            # discarded everything BEFORE the first marker while the admission test
            # above still treated the key as a pin -- so a head-shaped run placed
            # ahead of the marker (`deadbeefcafe_pinned_to_source_commit`) was never
            # examined, at any length including hex40/hex64. The rule was correct;
            # its INPUT was truncated before it ran. There is no second line of
            # defence here: the Q3/I1 token backstop walks VALUES, not `self_check`
            # key names. Scanning `kl` whole costs nothing -- the marker literal
            # itself contains no [0-9a-f] run of length >= 7.
            for _pin_m in _PIN_HEX_TOKEN_RE.finditer(kl):
                _tok = _pin_m.group(0)
                if src.startswith(_tok):
                    continue
                if len(_tok) > _PIN_HEAD_MAX_LEN:
                    raise ValueError(
                        f"I1/Q11 self_check pin is head-only by contract, but "
                        f"{k} carries a {len(_tok)}-char hex token which cannot be "
                        f"a commit head (max {_PIN_HEAD_MAX_LEN}): {_tok!r}. Pin "
                        f"non-head values through a typed path/sha structure that "
                        f"binds to disk, not through a boolean key name."
                    )
                raise ValueError(
                    f"I1/Q11 self_check pin claims non-source head: {k} "
                    f"(hex-shaped token {_tok!r} is not a prefix of the "
                    f"source commit)"
                )

    # ---- I2: pins.runner_and_harness_shas + TSA/BDL-like RO pins vs manifest ----
    man_map = man_sha_by_path

    rhs = pins.get("runner_and_harness_shas")
    if isinstance(rhs, dict):
        if not rhs:
            raise ValueError("I2 pins.runner_and_harness_shas empty")
        for rel, sha in rhs.items():
            _check_path_sha(
                man_map, man_path_set, str(rel), str(sha), where="pins.runner_and_harness_shas"
            )
    SOURCE_PIN_PREFIXES = (
        "calm/",
        "scripts/",
        "bin/",
        "rust/",
    )
    for pk, pv in pins.items():
        # Skip only Q1-admitted (disk-bound, allowlisted-location) DEAD lineage pins
        if f"pins.{pk}" in exempt_pin_paths:
            continue
        if not isinstance(pv, dict):
            continue
        if "path" not in pv or "sha256" not in pv:
            continue
        rel = str(pv["path"]).replace("\\", "/").lstrip("./")
        if rel.startswith("artifacts/"):
            continue
        if not rel.startswith(SOURCE_PIN_PREFIXES):
            continue
        _check_path_sha(
            man_map, man_path_set, rel, str(pv["sha256"]), where=f"pins.{pk}"
        )

    # L1: ANY non-current packet_vN / bare vN token outside identity-bound DEAD FAILS.
    # O2: exact known-ref whole-string equality only (no key/shape carve-out).
    rev_str_hits: list[tuple[str, str, bool]] = []
    _walk_strings_dead_aware(
        packet, path="", out=rev_str_hits, exempt_paths=exempt_pin_paths
    )
    for path, s, dead in rev_str_hits:
        if dead:
            continue
        if s.strip() in known_refs or s.strip().lower() in known_refs:
            continue
        for m in PACKET_REV_TOKEN_RE.finditer(s):
            other = m.group(1) or m.group(2)
            if other == rev_norm:
                continue
            raise ValueError(
                f"L1/J2 non-current packet_v{other} reference outside DEAD lineage at {path}"
            )

    # (O2's whole-string hex check is subsumed by the Q3/I1 token scan above, which
    # covers standalone AND embedded tokens under one value-bound rule.)

    # ---- I4 + J3: structured stop/branch contract ----
    branch_ids = packet.get("branch_ids") or packet.get("PRIORITY_ORDER") or []
    if not isinstance(branch_ids, (list, tuple)) or not branch_ids:
        raise ValueError("I4 branch_ids/PRIORITY_ORDER required")
    branch_set = {str(x) for x in branch_ids}
    allow = packet.get("terminal_branch_allow_set")
    if not isinstance(allow, (list, tuple)) or not allow:
        raise ValueError(
            "J3 terminal_branch_allow_set required (must equal branch_ids exactly)"
        )
    allow_set = {str(x) for x in allow}
    if allow_set != branch_set:
        raise ValueError(
            "J3 terminal_branch_allow_set must equal branch_ids exactly (set equality)"
        )
    # procedural stop reasons may not reference branch ids
    proc = packet.get("procedural_stop_reasons")
    if proc is not None:
        if not isinstance(proc, (list, tuple)):
            raise ValueError("J3 procedural_stop_reasons must be a list when present")
        for item in proc:
            sl = str(item)
            for bid in branch_set:
                if bid in sl or bid.lower() in sl.lower():
                    raise ValueError(
                        f"J3 procedural_stop_reasons must not reference branch id {bid}"
                    )
    # conservative prose check on stop_conditions
    stops = packet.get("stop_conditions")
    stop_text_parts: list[str] = []
    if isinstance(stops, list):
        stop_text_parts.extend(str(x) for x in stops)
    elif isinstance(stops, dict):
        stop_text_parts.append(json.dumps(stops))
    elif isinstance(stops, str):
        stop_text_parts.append(stops)
    stop_blob = "\n".join(stop_text_parts)
    stop_low = stop_blob.lower()
    if stop_blob:
        # any preregistered branch id co-occurring with forbidden/DEVIATION/"only ... may terminate"
        for bid in branch_set:
            bl = bid.lower()
            if bl not in stop_low:
                continue
            # window per occurrence
            for m in re.finditer(re.escape(bl), stop_low):
                lo = max(0, m.start() - 80)
                hi = min(len(stop_low), m.end() + 80)
                win = stop_low[lo:hi]
                if any(
                    x in win
                    for x in (
                        "forbidden",
                        "deviation",
                        "may terminate",
                        "only ",
                        "must not",
                        "!=",
                    )
                ):
                    raise ValueError(
                        f"J3/I4 stop_conditions restricts preregistered branch {bid}"
                    )

    # ---- I5 + L3: exact operative PLAN binding (path+sha frozen for this family) ----
    expected_plan_rel = str(expected_operative_plan_path).replace("\\", "/").lstrip("./")
    try:
        expected_plan_sha = _require_hex64(
            str(expected_operative_plan_sha256).lower(),
            field="expected_operative_plan_sha256",
        )
    except ValueError as e:
        raise ValueError(f"L3 {e}") from e

    # HAZARD (Q9) — `or` precedence over alternative key namings silently DROPS the
    # sibling. When the first name is present the second is never read here, so it
    # reaches no comparison at all. Combined with a harvest that enumerated BOTH
    # names, that asymmetry admitted a value nothing ever checked. The harvest is now
    # downstream of validation, and the sibling is explicitly reconciled below, so a
    # present-but-contradictory alternative name cannot pass unexamined. Any future
    # alternative naming added here MUST be reconciled the same way — the `or` chain
    # resolves a value, it does not validate the inputs it skipped.
    operative_id = packet.get("operative_plan_id") or packet.get("operative_adaptation_plan_id")
    operative_sha = packet.get("operative_plan_sha256") or packet.get("operative_adaptation_plan_sha256")
    for _prim, _alt in (
        ("operative_plan_id", "operative_adaptation_plan_id"),
        ("operative_plan_sha256", "operative_adaptation_plan_sha256"),
    ):
        _pv, _av = packet.get(_prim), packet.get(_alt)
        if isinstance(_pv, str) and isinstance(_av, str) and _pv.strip() and _av.strip():
            if _pv.strip() != _av.strip():
                raise ValueError(
                    f"I5/Q9 {_prim} and {_alt} both present and disagree: "
                    f"{_pv.strip()!r} != {_av.strip()!r} — the alternative naming must "
                    f"not carry a second, unvalidated identity"
                )
    auth = packet.get("authority_chain")
    if isinstance(auth, dict):
        operative_id = operative_id or auth.get("operative_plan_id") or auth.get("plan_path")
        operative_sha = operative_sha or auth.get("operative_plan_sha256") or auth.get("plan_sha256")
    if not operative_id or not operative_sha:
        raise ValueError(
            "I5 operative plan id+sha required (operative_plan_id/operative_plan_sha256 or authority_chain.plan_*)"
        )
    try:
        op_sha = _require_hex64(str(operative_sha).lower(), field="operative_plan_sha256")
    except ValueError as e:
        raise ValueError(f"I5 {e}") from e
    op_id = str(operative_id).replace("\\", "/").lstrip("./")
    if op_id != expected_plan_rel:
        raise ValueError(
            f"L3/J5 operative_plan_id must be EXACT {expected_plan_rel!r}, got {op_id!r}"
        )
    if op_sha != expected_plan_sha:
        raise ValueError(
            f"L3/J5 operative_plan_sha256 must be EXACT {expected_plan_sha}, got {op_sha}"
        )
    # reject DEAD plan lineage basenames even when path is wrong (defense in depth)
    base = Path(op_id).name.lower()
    if re.search(r"consumer_adapt_rerun_plan_v[1-5]\.json$", base):
        raise ValueError(f"L3/J5 DEAD plan lineage rejected: {op_id}")
    if ".." in Path(op_id).parts or op_id.startswith("/"):
        raise ValueError(f"J5 operative_plan_id must be repo-relative, got {op_id!r}")
    op_path = (repo / op_id).resolve()
    try:
        op_path.relative_to(repo.resolve())
    except ValueError as e:
        raise ValueError(f"J5 operative_plan_id escapes repo: {op_id}") from e
    if not op_path.is_file():
        raise ValueError(f"J5 operative plan missing on disk: {op_id}")
    live_op_sha = sha256_file(op_path)
    if live_op_sha != op_sha:
        raise ValueError(
            f"J5 operative_plan_sha256 mismatch for {op_id}: packet={op_sha} disk={live_op_sha}"
        )
    if live_op_sha != expected_plan_sha:
        raise ValueError(
            f"L3 disk rehash of operative plan != expected: disk={live_op_sha} expected={expected_plan_sha}"
        )

    # Q10: the operative plan's own revision number, so a mention of the OPERATIVE
    # revision is distinguishable from a mention of any other revision.
    _m_op_rev = re.search(r"plan_v(\d+)", expected_plan_rel.lower())
    _operative_rev = _m_op_rev.group(1) if _m_op_rev else None

    # L2/J4: FULL-tree recursive plan-reference scan (identity-bound DEAD only)
    bare_hits: list[tuple[str, str, bool]] = []
    _walk_strings_dead_aware(
        packet, path="", out=bare_hits, exempt_paths=exempt_pin_paths
    )
    for path, s, dead in bare_hits:
        if dead:
            continue
        # O2 value-bound: exact known-ref whole-string equality only
        if s.strip() in known_refs or s.strip().lower() in known_refs:
            continue
        # Skip the operative plan identity fields — but ONLY when the value really IS
        # that identity.
        #
        # Q9: this skip list is a SECOND key-name-shaped exemption, parallel to
        # known_refs admission and NOT covered by enumerating the seed sites. Keying
        # it on the key name alone left a foreign plan reference completely
        # unexamined under `operative_adaptation_plan_id`: it is not admitted to
        # known_refs (harvest removed above), L3/J5's `or` never compares the
        # sibling, and the hex backstop sees no token inside a path string — so the
        # skip was the whole of its escape. Value-checking here matches how
        # `plan_path`/`plan_sha256` are already handled just below.
        if (
            path.endswith("operative_plan_id")
            or path.endswith("operative_adaptation_plan_id")
            or path.endswith("operative_plan_sha256")
            or path.endswith("operative_adaptation_plan_sha256")
        ):
            _sv = s.strip()
            if _sv.replace("\\", "/").lstrip("./") == expected_plan_rel:
                continue
            if _sv.lower() == expected_plan_sha:
                continue
            # not the expected identity -> fall through to the normal scan
        if path.endswith("plan_path") or path.endswith("plan_sha256"):
            # Q10: EQUALITY, not containment. `expected_plan_rel in s` skipped any
            # value that merely CONTAINED the operative identity, so appending a
            # claim to it carried a foreign plan reference through unexamined
            # (e.g. "<expected rel> superseding PLAN_v3 which stays operative").
            # A skip must require the value to BE the identity, not to mention it.
            _sv = s.strip()
            if _sv.replace("\\", "/").lstrip("./") == expected_plan_rel:
                continue
            if _sv.lower() == expected_plan_sha:
                continue
        sl = s.lower()
        if "plan_v" not in sl and "re-hash plan" not in sl and "consumer_adapt_rerun_plan" not in sl:
            continue
        # Mentioning the operative plan is allowed — carrying a SECOND plan revision
        # alongside it is not.
        #
        # Q10 (second containment site): `expected_plan_rel in sl` alone meant any
        # string that merely mentioned the operative plan was waved through, so an
        # appended foreign claim rode along on the mention
        # (e.g. "<expected rel> superseding PLAN_v3 which stays operative").
        # Equality is too strict here — legitimate prose does reference the plan
        # inside a sentence — so instead remove the sanctioned identity and require
        # the REMAINDER to name no other plan revision.
        if expected_plan_rel.lower() in sl or expected_plan_sha in sl:
            _resid = sl.replace(expected_plan_rel.lower(), " ").replace(
                expected_plan_sha, " "
            )
            # Naming the OPERATIVE revision is legitimate (a faithful preflight row
            # reads "re-hash operative CONSUMER_ADAPT_RERUN PLAN_v6 -> <sha>"), so
            # only a DIFFERENT revision in the remainder is disqualifying.
            _other_revs = [
                g for g in re.findall(r"plan_v(\d+)", _resid) if g != _operative_rev
            ]
            if not _other_revs:
                continue
            # a different plan revision is named alongside -> fall through and flag
        # historical/eval qualifier still required for non-operative PLAN mentions
        if any(
            q in sl
            for q in (
                "historical",
                "eval plan",
                "eval_plan",
                "historical_eval",
                "dead lineage",
                "dead_lineage",
                "non-operative",
                "non_operative",
            )
        ):
            continue
        raise ValueError(
            f"J4/L2 ambiguous/foreign plan reference at {path} "
            f"(need DEAD/historical marker or exact operative PLAN_v6)"
        )

    # ---- Q3/I1 (LAST): EVERY hex40/hex64 token in the live tree must be bound ----
    # No key-name context gate (that was the I1 fail-open) and no whole-string-only
    # restriction (that was the embedded-token fail-open). A token escapes ONLY by
    # being the source commit or an exact member of the known-reference set, which
    # is itself seeded only from disk-verified lineage objects and typed identity
    # fields. Tokens inside identity-bound DEAD subtrees remain exempt.
    # Runs LAST by design (Q3'): it is the residual backstop for tokens that no
    # specific check binds, so specific checks keep their own diagnoses.
    tok_hits: list[tuple[str, str, bool]] = []
    _walk_collect_hex_tokens(
        packet, path="", out=tok_hits, exempt_paths=exempt_pin_paths
    )
    for path, hx, dead in tok_hits:
        if dead:
            continue
        if hx == src:
            continue
        if hx in known_refs:
            continue
        raise ValueError(
            f"Q3/I1 unbound hex token outside known-reference set at {path}: {hx}"
        )


def _seed_known_refs_from_authority_chain(
    known_refs: set,
    auth_pre: object,
    *,
    expected_operative_plan_path: str,
    expected_operative_plan_sha256: str,
    src: str,
) -> None:
    if not isinstance(auth_pre, dict):
        return
    _auth_plan_rel = str(expected_operative_plan_path).replace("\\", "/").lstrip("./")
    _auth_plan_sha = str(expected_operative_plan_sha256).lower()
    _auth_verifiable = {
        _auth_plan_rel,
        _auth_plan_rel.lower(),
        _auth_plan_sha,
        src,
    }
    for _k, _v in auth_pre.items():
        if not isinstance(_v, str):
            continue
        _t = _v.strip()
        if not _t:
            continue
        if _t in _auth_verifiable or _t.lower() in _auth_verifiable:
            known_refs.add(_t)
            known_refs.add(_t.lower())


def _expand_known_refs_from_validated_pins(
    known_refs: set,
    pins: dict,
    *,
    exempt_pin_paths: set,
    man_path_set: set,
    man_sha_by_path: dict,
    repo: Path,
    packet: dict,
) -> None:
    for pk, pv in pins.items():
        if isinstance(pv, dict) and isinstance(pv.get("path"), str) and isinstance(
            pv.get("sha256"), str
        ):
            rel = _norm_path(str(pv["path"]))
            sha = str(pv["sha256"]).lower()
            if f"pins.{pk}" in exempt_pin_paths:
                _add_ref(known_refs, rel)
                _add_ref(known_refs, sha)
            elif rel.startswith("artifacts/") and _disk_sha_ok(repo, rel, sha):
                _add_ref(known_refs, rel)
                _add_ref(known_refs, sha)
            elif (
                rel.startswith(("calm/", "scripts/", "bin/", "rust/"))
                and rel in man_path_set
                and man_sha_by_path.get(rel, "").lower() == sha
            ):
                _add_ref(known_refs, rel)
                _add_ref(known_refs, sha)
        elif isinstance(pv, dict) and pk in (
            "runner_and_harness_shas",
            "default_source_pins_for_consumer",
        ):
            for rel, sha in pv.items():
                rel_n = _norm_path(str(rel))
                sha_n = str(sha).lower()
                if (
                    rel_n in man_path_set
                    and man_sha_by_path.get(rel_n, "").lower() == sha_n
                ):
                    _add_ref(known_refs, rel_n)
                    _add_ref(known_refs, sha_n)
                elif rel_n.startswith("artifacts/") and _disk_sha_ok(repo, rel_n, sha_n):
                    _add_ref(known_refs, rel_n)
                    _add_ref(known_refs, sha_n)
    for k in (
        "science_source_manifest_sha256",
        "generator_script_sha256",
        "dry_exec_tool_sha256",
        "source_commit_sha",
    ):
        _add_ref(known_refs, packet.get(k))


def _build_dry_exec_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LANDS-AB packet dry-exec")
    ap.add_argument("--packet", required=True)
    ap.add_argument("--verify-source-manifest", required=True)
    ap.add_argument("--expected-source-commit", required=True)
    ap.add_argument(
        "--expected-operative-plan-path",
        default=EXPECTED_OPERATIVE_PLAN_REL,
        help="L3 exact operative plan path (this packet family default frozen)",
    )
    ap.add_argument(
        "--expected-operative-plan-sha256",
        default=EXPECTED_OPERATIVE_PLAN_SHA256,
        help="L3 exact operative plan sha256 (this packet family default frozen)",
    )
    ap.add_argument("--repo-root", default=".")
    return ap


def _add_ref(known_refs: set, val: object) -> None:
    if isinstance(val, str) and val.strip():
        known_refs.add(val.strip())
        known_refs.add(val.strip().lower())


def _disk_sha_ok(repo: Path, rel: str, sha: str) -> bool:
    # Q5: containment + canonicality BEFORE any disk read, so a traversal pin
    # cannot bind an out-of-repo file and inject its path/sha into known_refs.
    rel_n = _canonical_repo_relpath(rel, repo=repo)
    if rel_n is None:
        return False
    fp = repo.resolve() / rel_n
    if not fp.is_file():
        return False
    try:
        return sha256_file(fp) == str(sha).lower()
    except Exception:
        return False


def _check_path_sha(
    man_map: dict, man_path_set: set, rel: str, sha: str, *, where: str
) -> None:
    rel_n = str(rel).replace("\\", "/").lstrip("./")
    if rel_n not in man_path_set:
        raise ValueError(f"I2 pin path absent from manifest: {rel_n} ({where})")
    exp = man_map[rel_n]
    got = str(sha).lower()
    if got != exp:
        raise ValueError(
            f"I2 pin sha != manifest for {rel_n} at {where}: pin={got} manifest={exp}"
        )


def _flag_val(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        return None
    return str(argv[i + 1])


def _is_write_path_token(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    # only treat path-like tokens (absolute or relative with separators / extension)
    if not (s.startswith("/") or s.startswith("./") or "/" in s or s.endswith((".json", ".jsonl", ".pt", ".pth", ".log", ".txt"))):
        return False
    low = s.lower()
    if low.endswith(".pt") or low.endswith(".pth"):
        return True
    if "/checkpoints/" in low or "calm/hrm/checkpoints" in low:
        return True
    if re.search(r"(^|/)banked(_|/|-|$)", low):
        return True
    return False


def _scan_write_paths_from_row(inv: dict, argv: list[str], env: dict) -> list[str]:
    paths: list[str] = []
    for key in (
        "raw_obs_path_template",
        "out_path_template",
        "terminal_collection_raw_obs_path_template",
        "enforcer_receipt_path_template",
        "terminal_collection_enforcer_receipt_path",
        "phase_events_jsonl_template",
    ):
        v = inv.get(key)
        if isinstance(v, str):
            paths.append(v)
    for a in argv:
        if isinstance(a, str):
            paths.append(a)
    for v in env.values():
        if isinstance(v, str):
            paths.append(v)
    return paths


import types

# Value-based control-flow sentinel (NOT an Exception — census-neutral).
# Leaf hosts return int exit codes; R3 children may return this to mean loop-continue.
_DRY_EXEC_CONTINUE = object()

def _dry_exec_validate_row_env_paths_raw(ctx):
    if not isinstance(ctx.rc, dict):
        print('error: row_command not object', file=sys.stderr)
        return 2
    ctx.gr = ctx.rc.get('gating_row')
    if ctx.gr not in GATING_ROWS or ctx.gr in ctx.seen_rows:
        print(f'error: bad/dup gating_row {ctx.gr!r}', file=sys.stderr)
        return 2
    ctx.seen_rows.add(ctx.gr)
    ctx.inv = ctx.rc.get('invocation') or {}
    if not isinstance(ctx.inv, dict):
        print(f'error: {ctx.gr} invocation must be object', file=sys.stderr)
        return 2
    ctx.argv = ctx.inv.get('argv_template') or ctx.inv.get('argv')
    if not isinstance(ctx.argv, list) or not ctx.argv or (not all((isinstance(x, str) for x in ctx.argv))):
        print(f'error: missing argv for {ctx.gr}', file=sys.stderr)
        return 2
    ctx.argv_key = json.dumps(ctx.argv)
    if ctx.argv_key in ctx.seen_argv:
        print(f'error: duplicate argv for {ctx.gr}', file=sys.stderr)
        return 2
    ctx.seen_argv.add(ctx.argv_key)
    for ctx.tok in ctx.argv:
        if _is_repo_path_token(ctx.tok):
            ctx.formal_argv_repo_paths.add(_repo_path_from_token(ctx.tok))
    ctx.env = ctx.inv.get('env_required') or {}
    if not isinstance(ctx.env, dict):
        print(f'error: {ctx.gr} env_required must be object', file=sys.stderr)
        return 2
    for ctx.req_env in ('PYTHONPATH', 'LANDS_AB_RUNTIME_SCRATCH', 'LANDS_AB_RUN_ROOT'):
        if ctx.req_env not in ctx.env or not str(ctx.env.get(ctx.req_env) or '').strip():
            print(f'error: {ctx.gr} missing env {ctx.req_env}', file=sys.stderr)
            return 2
    ctx.scratch = str(ctx.env['LANDS_AB_RUNTIME_SCRATCH']).strip()
    ctx.run_root = str(ctx.env['LANDS_AB_RUN_ROOT']).strip()
    if not any((tok in ctx.run_root for tok in NONCE_TOKENS)):
        print(f"error: {ctx.gr} LANDS_AB_RUN_ROOT missing frozen nonce template token (<{NONCE_TOKENS[0].strip('<>')}> or similar): {ctx.run_root!r}", file=sys.stderr)
        return 2
    if not _is_descendant(ctx.run_root, ctx.scratch):
        print(f'error: {ctx.gr} LANDS_AB_RUN_ROOT must be descendant of LANDS_AB_RUNTIME_SCRATCH', file=sys.stderr)
        return 2
    if str(ctx.run_root).startswith('/') and _is_descendant(ctx.run_root.replace('<nonce>', 'NONCE').replace('<run_local_nonce>', 'RUN_LOCAL_NONCE'), str(ctx.repo)):
        print(f'error: {ctx.gr} LANDS_AB_RUN_ROOT must not be under repo root', file=sys.stderr)
        return 2
    if str(ctx.scratch).startswith('/') and _is_descendant(ctx.scratch, str(ctx.repo)):
        print(f'error: {ctx.gr} LANDS_AB_RUNTIME_SCRATCH must not be under repo root', file=sys.stderr)
        return 2
    if ctx.scratch != ctx.pkt_scratch or ctx.run_root != ctx.pkt_run_root:
        print(f'error: {ctx.gr} env scratch/run_root must equal packet runtime_scratch.env_binding', file=sys.stderr)
        return 2
    if ctx.shared_scratch is None:
        ctx.shared_scratch, ctx.shared_run_root = (ctx.scratch, ctx.run_root)
    elif ctx.scratch != ctx.shared_scratch or ctx.run_root != ctx.shared_run_root:
        print(f'error: {ctx.gr} scratch/run_root differs from other gating rows (one formal run)', file=sys.stderr)
        return 2
    for ctx.tok in _scan_write_paths_from_row(ctx.inv, list(ctx.argv), ctx.env):
        if _is_write_path_token(ctx.tok):
            print(f'error: forbidden write path token in {ctx.gr}: {ctx.tok}', file=sys.stderr)
            return 2
    ctx.is_cuda = str(ctx.gr).startswith('G_CUDA')
    ctx.raw_aliases = {'raw_obs_path_template': ctx.inv.get('raw_obs_path_template'), 'terminal_collection_raw_obs_path_template': ctx.inv.get('terminal_collection_raw_obs_path_template'), 'out_path_template': ctx.inv.get('out_path_template')}
    ctx.present_raw = {k: str(v) for k, v in ctx.raw_aliases.items() if isinstance(v, str) and v.strip()}
    if 'raw_obs_path_template' not in ctx.present_raw:
        print(f'error: {ctx.gr} missing required raw_obs_path_template', file=sys.stderr)
        return 2
    ctx.raw = ctx.present_raw['raw_obs_path_template']
    for ctx.k, ctx.v in ctx.present_raw.items():
        if ctx.v != ctx.raw:
            print(f'error: {ctx.gr} raw alias {ctx.k} != raw_obs_path_template', file=sys.stderr)
            return 2
    if ctx.raw in ctx.seen_raw:
        print(f'error: duplicate raw path {ctx.raw}', file=sys.stderr)
        return 2
    ctx.seen_raw.add(ctx.raw)
    if not _is_descendant(ctx.raw, ctx.run_root):
        print(f'error: {ctx.gr} raw_obs path must be descendant of LANDS_AB_RUN_ROOT', file=sys.stderr)
        return 2
    if str(ctx.raw).startswith('/') and _is_descendant(ctx.raw.replace('<nonce>', 'NONCE').replace('<run_local_nonce>', 'RUN_LOCAL_NONCE'), str(ctx.repo)):
        print(f'error: {ctx.gr} raw_obs path must not be under repo root', file=sys.stderr)
        return 2
    ctx.raw_base = Path(str(ctx.raw).replace('<nonce>', 'N').replace('<run_local_nonce>', 'R')).name
    if not ctx.raw_base.startswith(f'lands_ab_raw_obs_{ctx.gr}_'):
        print(f'error: {ctx.gr} raw basename must start with lands_ab_raw_obs_{ctx.gr}_ (harvest pattern)', file=sys.stderr)
        return 2

def _dry_exec_validate_row_cpu_cuda_budgets(ctx):
    if ctx.raw_base in ctx.seen_raw_basenames:
        print(f'error: {ctx.gr} duplicate raw basename {ctx.raw_base}', file=sys.stderr)
        return 2
    ctx.seen_raw_basenames.add(ctx.raw_base)
    if ctx.argv[0] != 'timeout' or len(ctx.argv) < 3:
        print(f'error: {ctx.gr} argv must start with timeout <budget>', file=sys.stderr)
        return 2
    try:
        ctx.outer_to = _require_finite_positive(ctx.argv[1], field=f'{ctx.gr} outer timeout')
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    ctx.hard = _row_hard_timeout_seconds(ctx.rc, ctx.inv)
    if ctx.hard is None:
        print(f'error: {ctx.gr} missing duration_budget_seconds/hard_timeout_seconds for G6 bind', file=sys.stderr)
        return 2
    if float(ctx.outer_to) != float(ctx.hard):
        print(f'error: {ctx.gr} outer timeout {ctx.outer_to} != row hard bound {ctx.hard} (rule: argv[1] == inv.hard_timeout_seconds or row.duration_budget_seconds)', file=sys.stderr)
        return 2
    if not ctx.is_cuda:
        try:
            structural_preflight_row(list(ctx.argv), repo=ctx.repo, runner_parser=ctx.runner_parser)
        except Exception as e:
            print(f'error: structural preflight {ctx.gr}: {e}', file=sys.stderr)
            return 2
        ctx.out_val = _flag_val(list(ctx.argv), '--out')
        if ctx.out_val != ctx.raw:
            print(f'error: {ctx.gr} --out must equal raw_obs_path_template', file=sys.stderr)
            return 2
        return _DRY_EXEC_CONTINUE
    try:
        ctx.chain = _validate_ordered_cuda_chain(list(ctx.argv), gr=str(ctx.gr))
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    ctx.phase_argv = ctx.chain['phase_argv']
    ctx.enf_argv = ctx.chain['enf_argv']
    ctx.budget_map = ctx.chain['budget_map']
    ctx.pb_decl = ctx.inv.get('phase_budgets_seconds')
    if isinstance(ctx.pb_decl, dict):
        if set(ctx.pb_decl.keys()) != set(REQUIRED_PHASE_BUDGET_NAMES):
            print(f'error: {ctx.gr} declared phase_budgets_seconds keys mismatch required set', file=sys.stderr)
            return 2
        for ctx.name in REQUIRED_PHASE_BUDGET_NAMES:
            try:
                ctx.decl_v = float(ctx.pb_decl[ctx.name])
            except Exception:
                print(f'error: {ctx.gr} declared budget {ctx.name} not numeric', file=sys.stderr)
                return 2
            if float(ctx.budget_map[ctx.name]) != float(ctx.decl_v):
                print(f'error: {ctx.gr} argv budget {ctx.name}={ctx.budget_map[ctx.name]} != declared {ctx.decl_v}', file=sys.stderr)
                return 2
    for ctx.label, ctx.pth in (('phase', ctx.phase_argv), ('enforcer', ctx.enf_argv)):
        if not _is_descendant(ctx.pth, ctx.run_root):
            print(f'error: {ctx.gr} {ctx.label} path must be descendant of LANDS_AB_RUN_ROOT', file=sys.stderr)
            return 2
        if str(ctx.pth).startswith('/') and _is_descendant(ctx.pth.replace('<nonce>', 'NONCE').replace('<run_local_nonce>', 'RUN_LOCAL_NONCE'), str(ctx.repo)):
            print(f'error: {ctx.gr} {ctx.label} path must not be under repo root', file=sys.stderr)
            return 2
        ctx.base = Path(str(ctx.pth).replace('<nonce>', 'N').replace('<run_local_nonce>', 'R')).name
        if ctx.base in ctx.seen_runtime_basenames:
            print(f'error: {ctx.gr} duplicate runtime basename {ctx.base}', file=sys.stderr)
            return 2
        ctx.seen_runtime_basenames.add(ctx.base)
    ctx.phase_aliases = {'phase_events_jsonl_template': ctx.inv.get('phase_events_jsonl_template'), 'terminal_collection_phase_events_jsonl_path': ctx.inv.get('terminal_collection_phase_events_jsonl_path')}
    for ctx.k, ctx.v in ctx.phase_aliases.items():
        if isinstance(ctx.v, str) and ctx.v.strip() and (str(ctx.v) != ctx.phase_argv):
            print(f'error: {ctx.gr} phase alias {ctx.k} != --phase-events-jsonl', file=sys.stderr)
            return 2
    ctx.enf_aliases = {'enforcer_receipt_path_template': ctx.inv.get('enforcer_receipt_path_template'), 'terminal_collection_enforcer_receipt_path': ctx.inv.get('terminal_collection_enforcer_receipt_path')}
    ctx.present_enf = {k: str(v) for k, v in ctx.enf_aliases.items() if isinstance(v, str) and v.strip()}
    if not ctx.present_enf:
        print(f'error: {ctx.gr} missing enforcer metadata path', file=sys.stderr)
        return 2
    for ctx.k, ctx.v in ctx.present_enf.items():
        if ctx.v != ctx.enf_argv:
            print(f'error: {ctx.gr} enforcer alias {ctx.k} != --enforcer-receipt argv', file=sys.stderr)
            return 2
    ctx.phase_env = ctx.env.get('SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL')
    if not ctx.phase_env:
        print(f'error: {ctx.gr} missing phase env', file=sys.stderr)
        return 2
    if ctx.phase_env != ctx.phase_argv:
        print(f'error: {ctx.gr} phase env != --phase-events-jsonl argv', file=sys.stderr)
        return 2
    if ctx.phase_env in ctx.seen_phase:
        print(f'error: duplicate phase {ctx.phase_env}', file=sys.stderr)
        return 2
    ctx.seen_phase.add(ctx.phase_env)
    if ctx.enf_argv in ctx.seen_enforcer:
        print(f'error: duplicate enforcer {ctx.enf_argv}', file=sys.stderr)
        return 2
    ctx.seen_enforcer.add(ctx.enf_argv)

def _dry_exec_validate_row_commands(ctx):
    for rc in ctx.row_commands:
        ctx.rc = rc
        r = _dry_exec_validate_row_env_paths_raw(ctx)
        if r is _DRY_EXEC_CONTINUE:
            continue
        if r is not None:
            return r
        r = _dry_exec_validate_row_cpu_cuda_budgets(ctx)
        if r is _DRY_EXEC_CONTINUE:
            continue
        if r is not None:
            return r

def _dry_exec_bind_sources(ctx):
    try:
        ctx.man_cli, ctx.man_cli_rel = _resolve_under_repo(ctx.args.verify_source_manifest, repo=ctx.repo)
    except ValueError as e:
        print(f'error: manifest outside repo ({e})', file=sys.stderr)
        return 2
    try:
        ctx.packet = json.loads(ctx.packet_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'error: packet load: {e}', file=sys.stderr)
        return 2
    ctx.required = ['science_source_manifest_path', 'science_source_manifest_sha256', 'source_commit_sha', 'generator_script_path', 'generator_script_sha256', 'dry_exec_tool_path', 'dry_exec_tool_sha256']
    for ctx.f in ctx.required:
        if ctx.f not in ctx.packet:
            print(f'error: packet missing binding field {ctx.f}', file=sys.stderr)
            return 2
    try:
        ctx.man_path = str(ctx.packet['science_source_manifest_path'])
        ctx.man_sha = _require_hex64(ctx.packet['science_source_manifest_sha256'], field='science_source_manifest_sha256')
        ctx.src_commit = _require_hex40(str(ctx.packet['source_commit_sha']).lower(), field='source_commit_sha')
        ctx.gen_path = str(ctx.packet['generator_script_path'])
        ctx.gen_sha = _require_hex64(ctx.packet['generator_script_sha256'], field='generator_script_sha256')
        ctx.dry_path = str(ctx.packet['dry_exec_tool_path'])
        ctx.dry_sha = _require_hex64(ctx.packet['dry_exec_tool_sha256'], field='dry_exec_tool_sha256')
        ctx.exp = _require_hex40(str(ctx.args.expected_source_commit).lower(), field='expected_source_commit')
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    try:
        ctx._man_pkt_abs, ctx.man_path_norm = _resolve_under_repo(ctx.man_path, repo=ctx.repo)
    except ValueError as e:
        print(f'error: packet.science_source_manifest_path outside repo ({e})', file=sys.stderr)
        return 2
    if ctx.man_cli_rel != ctx.man_path_norm:
        print(f'error: CLI manifest path {ctx.man_cli_rel!r} != packet.science_source_manifest_path {ctx.man_path_norm!r}', file=sys.stderr)
        return 2
    if ctx.exp != ctx.src_commit:
        print('error: --expected-source-commit != packet.source_commit_sha', file=sys.stderr)
        return 2
    if not ctx.man_cli.is_file():
        print(f'error: missing manifest file {ctx.man_cli}', file=sys.stderr)
        return 2
    ctx.live_man_sha = sha256_file(ctx.man_cli)
    if ctx.live_man_sha != ctx.man_sha:
        print('error: manifest file sha != packet.science_source_manifest_sha256', file=sys.stderr)
        return 2
    if ctx.gen_path != 'scripts/lands_ab_science_source_manifest.py':
        print(f'error: generator_script_path must be scripts/lands_ab_science_source_manifest.py, got {ctx.gen_path!r}', file=sys.stderr)
        return 2
    if ctx.dry_path != 'scripts/lands_ab_packet_dry_exec.py':
        print(f'error: dry_exec_tool_path must be scripts/lands_ab_packet_dry_exec.py, got {ctx.dry_path!r}', file=sys.stderr)
        return 2
    for ctx.rel, ctx.expect in ((ctx.gen_path, ctx.gen_sha), (ctx.dry_path, ctx.dry_sha)):
        ctx.fp = ctx.repo / ctx.rel
        if not ctx.fp.is_file():
            print(f'error: missing pinned tool/script {ctx.rel}', file=sys.stderr)
            return 2
        if sha256_file(ctx.fp) != ctx.expect:
            print(f'error: sha mismatch for {ctx.rel}', file=sys.stderr)
            return 2
    try:
        ctx.man = json.loads(ctx.man_cli.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'error: manifest JSON: {e}', file=sys.stderr)
        return 2
    if ctx.man.get('schema') != 'LANDS_AB_science_source_manifest/v1':
        print(f"error: manifest schema must be LANDS_AB_science_source_manifest/v1, got {ctx.man.get('schema')!r}", file=sys.stderr)
        return 2
    ctx.entries = ctx.man.get('entries')
    if not isinstance(ctx.entries, list) or not ctx.entries:
        print('error: manifest entries missing', file=sys.stderr)
        return 2
    if ctx.man.get('n_entries') != len(ctx.entries):
        print('error: manifest n_entries != len(entries)', file=sys.stderr)
        return 2
    ctx.paths = []
    for ctx.ent in ctx.entries:
        if not isinstance(ctx.ent, dict) or 'path' not in ctx.ent or 'sha256' not in ctx.ent:
            print('error: bad manifest entry', file=sys.stderr)
            return 2
        ctx.rel = ctx.ent['path']
        if not isinstance(ctx.rel, str) or '..' in Path(ctx.rel).parts or ctx.rel.startswith('/'):
            print(f'error: bad/escape path {ctx.rel!r}', file=sys.stderr)
            return 2
        try:
            ctx.h = _require_hex64(ctx.ent['sha256'], field=f'manifest[{ctx.rel}].sha256')
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return 2
        ctx.paths.append(ctx.rel)
        ctx.fp = ctx.repo / ctx.rel
        if not ctx.fp.is_file():
            print(f'error: missing source path {ctx.rel}', file=sys.stderr)
            return 2
        if sha256_file(ctx.fp) != ctx.h:
            print(f'error: disk sha mismatch {ctx.rel}', file=sys.stderr)
            return 2
    if ctx.paths != sorted(ctx.paths) or len(ctx.paths) != len(set(ctx.paths)):
        print('error: manifest paths not sorted unique', file=sys.stderr)
        return 2
    ctx.man_path_set = set(ctx.paths)
    ctx.missing_mand = [p for p in MANDATORY_EXECUTION_SOURCE_SET if p not in ctx.man_path_set]
    if ctx.missing_mand:
        print(f'error: missing mandatory source: {ctx.missing_mand[0]}', file=sys.stderr)
        return 2

def _dry_exec_validate_packet_contract(ctx):
    ctx.schema = str(ctx.packet.get('schema') or '')
    if not ctx.schema.startswith('LANDS_AB_EVAL_launch_packet'):
        print(f'error: bad packet schema {ctx.schema!r}', file=sys.stderr)
        return 2
    ctx.rows = ctx.packet.get('gating_rows_exact') or ctx.packet.get('gating_rows')
    if list(ctx.rows) != GATING_ROWS and set(ctx.rows) != set(GATING_ROWS):
        if set(ctx.rows) != set(GATING_ROWS) or len(ctx.rows) != 7:
            print('error: gating_rows must be exact 7-tuple set', file=sys.stderr)
            return 2
    ctx.row_commands = ctx.packet.get('row_commands')
    if not isinstance(ctx.row_commands, list) or len(ctx.row_commands) != 7:
        print('error: row_commands must be list of 7', file=sys.stderr)
        return 2
    if 'claim_ceiling' not in ctx.packet:
        print('error: claim_ceiling required', file=sys.stderr)
        return 2
    ctx.cc = ctx.packet.get('claim_ceiling')
    if not isinstance(ctx.cc, dict):
        print('error: claim_ceiling must be object', file=sys.stderr)
        return 2
    for ctx.k in ('LANDS_AB', 'science_claim', 'equivalent_minted', 'full_sub2_runtime_ready_for_science'):
        if ctx.k not in ctx.cc:
            print(f'error: claim_ceiling missing field {ctx.k}', file=sys.stderr)
            return 2
        if ctx.cc.get(ctx.k) is not False:
            print(f'error: claim_ceiling.{ctx.k} must be false', file=sys.stderr)
            return 2
    if 'science_claim' in ctx.packet:
        try:
            ctx._sci = _require_packet_bool(ctx.packet.get('science_claim'), field='science_claim', where='packet')
        except ValueError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2
        if ctx._sci is True:
            print('error: science_claim true forbidden', file=sys.stderr)
            return 2
    ctx.LIVE_ORDER = ['BR-LANDS-AB-SCOPE-CREEP-STOP', 'BR-LANDS-AB-FIXTURE-CONTRACT-FAIL', 'BR-LANDS-AB-VACUOUS', 'BR-LANDS-AB-DIVERGENT-EVENT', 'BR-LANDS-AB-DIVERGENT-APPLY', 'BR-LANDS-AB-DIVERGENT-ORACLE-LIVE', 'BR-LANDS-AB-EQUIVALENT']
    ctx.po = ctx.packet.get('PRIORITY_ORDER') or ctx.packet.get('priority_order')
    if not isinstance(ctx.po, (list, tuple)) or list(ctx.po) != ctx.LIVE_ORDER:
        print('error: PRIORITY_ORDER must echo exact live order', file=sys.stderr)
        return 2
    ctx.bids = ctx.packet.get('branch_ids')
    if not isinstance(ctx.bids, (list, tuple)) or set(ctx.bids) != set(ctx.LIVE_ORDER) or len(ctx.bids) != 7:
        print('error: branch_ids must be exact live 7-set', file=sys.stderr)
        return 2
    ctx.FROZEN_EXECUTOR_ROLE = 'claude_as_test_operator'
    ctx.ex = ctx.packet.get('executor')
    if not isinstance(ctx.ex, dict):
        print('error: executor required object', file=sys.stderr)
        return 2
    ctx.role = str(ctx.ex.get('role') or '')
    if ctx.role != ctx.FROZEN_EXECUTOR_ROLE:
        print(f'error: executor.role must be exactly {ctx.FROZEN_EXECUTOR_ROLE!r}, got {ctx.role!r}', file=sys.stderr)
        return 2
    ctx.forb = ctx.ex.get('forbidden_for_plan_dev')
    if not isinstance(ctx.forb, (list, tuple)) or not ctx.forb:
        print('error: executor.forbidden_for_plan_dev required non-empty', file=sys.stderr)
        return 2
    if not any(('formal' in str(x).lower() for x in ctx.forb)):
        print('error: executor.forbidden_for_plan_dev must prohibit formal run', file=sys.stderr)
        return 2
    if ctx.ex.get('one_terminal_receipt') is not True and ctx.packet.get('one_terminal_receipt') is not True:
        print('error: one_terminal_receipt must be true', file=sys.stderr)
        return 2
    ctx.rs = ctx.packet.get('runtime_scratch')
    if not isinstance(ctx.rs, dict):
        print('error: runtime_scratch required object', file=sys.stderr)
        return 2
    ctx.harvest = ctx.rs.get('harvest_exactly_one_raw_obs')
    if not isinstance(ctx.harvest, dict):
        print('error: runtime_scratch.harvest_exactly_one_raw_obs must be object', file=sys.stderr)
        return 2
    if ctx.harvest.get('applies_to_each_of_7_gating_rows') is not True:
        print('error: harvest applies_to_each_of_7_gating_rows must be true', file=sys.stderr)
        return 2
    ctx.helper = ctx.harvest.get('helper')
    if not isinstance(ctx.helper, str) or 'harvest_exactly_one_raw_obs' not in ctx.helper:
        print('error: harvest.helper must name harvest_exactly_one_raw_obs', file=sys.stderr)
        return 2
    ctx.zom = str(ctx.harvest.get('zero_or_multiple') or '').upper()
    if ctx.zom not in ('STOP', 'FAIL', 'ERROR'):
        print('error: harvest.zero_or_multiple must be STOP semantics', file=sys.stderr)
        return 2

def _dry_exec_validate_execution_order(ctx):
    ctx.REQUIRED_LIFECYCLE = {'preflight', 'activate_fresh_nonce_run_root_must_not_exist', 'harvest_exactly_one_per_row', 'EVIDENCE_CONSUMER', 'terminal_classification_vs_prereg', 'FORMAL_RUNTIME_CREATE_terminal_receipt'}
    ctx.eo = ctx.packet.get('execution_order')
    if not isinstance(ctx.eo, (list, tuple)) or not ctx.eo:
        print('error: execution_order required non-empty list', file=sys.stderr)
        return 2
    ctx.eo_list = [str(x) for x in ctx.eo]
    ctx.positions = []
    for ctx.gr in GATING_ROWS:
        if ctx.eo_list.count(ctx.gr) != 1:
            print(f'error: execution_order must contain gating row {ctx.gr} exactly once', file=sys.stderr)
            return 2
        ctx.positions.append(ctx.eo_list.index(ctx.gr))
    if ctx.positions != sorted(ctx.positions):
        print('error: gating rows in execution_order not in required order', file=sys.stderr)
        return 2
    ctx.missing_life = ctx.REQUIRED_LIFECYCLE - set(ctx.eo_list)
    if ctx.missing_life:
        print(f'error: execution_order missing lifecycle steps {sorted(ctx.missing_life)}', file=sys.stderr)
        return 2

def _dry_exec_bind_runtime_and_order(ctx):
    ctx.runner_parser, ctx._, ctx.live_runner_sha = load_runner_parser(ctx.repo)
    ctx._ = ctx.live_runner_sha
    ctx.seen_phase: set[str] = set()
    ctx.seen_enforcer: set[str] = set()
    ctx.seen_rows: set[str] = set()
    ctx.seen_argv: set[str] = set()
    ctx.seen_raw: set[str] = set()
    ctx.seen_raw_basenames: set[str] = set()
    ctx.seen_runtime_basenames: set[str] = set()
    ctx.shared_scratch: str | None = None
    ctx.shared_run_root: str | None = None
    ctx.formal_argv_repo_paths: set[str] = set()
    ctx.rs_bind = (ctx.packet.get('runtime_scratch') or {}).get('env_binding') or {}
    if not isinstance(ctx.rs_bind, dict):
        print('error: runtime_scratch.env_binding must be object', file=sys.stderr)
        return 2
    ctx.pkt_scratch = str(ctx.rs_bind.get('LANDS_AB_RUNTIME_SCRATCH') or '').strip()
    ctx.pkt_run_root = str(ctx.rs_bind.get('LANDS_AB_RUN_ROOT') or '').strip()
    if not ctx.pkt_scratch or not ctx.pkt_run_root:
        print('error: runtime_scratch.env_binding missing LANDS_AB_RUNTIME_SCRATCH/RUN_ROOT', file=sys.stderr)
        return 2

def _dry_exec_validate_rowset_invariants(ctx):
    if ctx.seen_rows != set(GATING_ROWS):
        print('error: row_commands incomplete vs GATING_ROWS', file=sys.stderr)
        return 2
    if len(ctx.seen_argv) != 7 or len(ctx.seen_raw) != 7:
        print('error: argv/raw paths not unique 7', file=sys.stderr)
        return 2
    ctx.cuda_rows = [g for g in GATING_ROWS if g.startswith('G_CUDA')]
    if len(ctx.seen_phase) != len(ctx.cuda_rows) or len(ctx.seen_enforcer) != len(ctx.cuda_rows):
        print('error: phase/enforcer path count mismatch vs CUDA rows', file=sys.stderr)
        return 2
    for ctx.rel in sorted(ctx.formal_argv_repo_paths):
        if ctx.rel not in ctx.man_path_set:
            print(f'error: formal argv path not covered by manifest: {ctx.rel}', file=sys.stderr)
            return 2

def _dry_exec_run_i_series(ctx):
    try:
        _validate_i_series_consistency(ctx.packet, man_path_set=ctx.man_path_set, man_sha_by_path={e['path']: _require_hex64(e['sha256'], field=f"manifest[{e['path']}].sha256") for e in ctx.entries}, source_commit=ctx.src_commit, repo=ctx.repo, expected_operative_plan_path=ctx.args.expected_operative_plan_path, expected_operative_plan_sha256=ctx.args.expected_operative_plan_sha256)
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

def main(argv: list[str] | None = None) -> int:
    ctx = types.SimpleNamespace()
    ctx.argv = argv
    ctx.ap = _build_dry_exec_arg_parser()
    ctx.args = ctx.ap.parse_args(ctx.argv)
    ctx.repo = Path(ctx.args.repo_root).resolve()
    ctx.packet_path = Path(ctx.args.packet)
    if not ctx.packet_path.is_absolute():
        ctx.packet_path = ctx.repo / ctx.packet_path
    _rc = _dry_exec_bind_sources(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_validate_packet_contract(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_validate_execution_order(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_bind_runtime_and_order(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_validate_row_commands(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_validate_rowset_invariants(ctx)
    if _rc is not None:
        return _rc
    _rc = _dry_exec_run_i_series(ctx)
    if _rc is not None:
        return _rc
    print('PACKET_DRY_EXEC_OK')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
