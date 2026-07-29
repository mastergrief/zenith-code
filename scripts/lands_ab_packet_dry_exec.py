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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LANDS-AB packet dry-exec")
    ap.add_argument("--packet", required=True)
    ap.add_argument("--verify-source-manifest", required=True)
    ap.add_argument("--expected-source-commit", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    packet_path = Path(args.packet)
    if not packet_path.is_absolute():
        packet_path = repo / packet_path
    # F7: resolve absolute AND relative manifests; reject any resolved path outside repo
    # (including relative `../` escapes). Normalize before equality later.
    try:
        man_cli, man_cli_rel = _resolve_under_repo(args.verify_source_manifest, repo=repo)
    except ValueError as e:
        print(f"error: manifest outside repo ({e})", file=sys.stderr)
        return 2

    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"error: packet load: {e}", file=sys.stderr)
        return 2

    # G2 required binding fields
    required = [
        "science_source_manifest_path",
        "science_source_manifest_sha256",
        "source_commit_sha",
        "generator_script_path",
        "generator_script_sha256",
        "dry_exec_tool_path",
        "dry_exec_tool_sha256",
    ]
    for f in required:
        if f not in packet:
            print(f"error: packet missing binding field {f}", file=sys.stderr)
            return 2
    try:
        man_path = str(packet["science_source_manifest_path"])
        man_sha = _require_hex64(packet["science_source_manifest_sha256"], field="science_source_manifest_sha256")
        src_commit = _require_hex40(str(packet["source_commit_sha"]).lower(), field="source_commit_sha")
        gen_path = str(packet["generator_script_path"])
        gen_sha = _require_hex64(packet["generator_script_sha256"], field="generator_script_sha256")
        dry_path = str(packet["dry_exec_tool_path"])
        dry_sha = _require_hex64(packet["dry_exec_tool_sha256"], field="dry_exec_tool_sha256")
        exp = _require_hex40(str(args.expected_source_commit).lower(), field="expected_source_commit")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # F7: normalize packet-relative (or absolute) manifest path under repo before equality
    try:
        _man_pkt_abs, man_path_norm = _resolve_under_repo(man_path, repo=repo)
    except ValueError as e:
        print(f"error: packet.science_source_manifest_path outside repo ({e})", file=sys.stderr)
        return 2
    if man_cli_rel != man_path_norm:
        print(
            f"error: CLI manifest path {man_cli_rel!r} != packet.science_source_manifest_path {man_path_norm!r}",
            file=sys.stderr,
        )
        return 2
    if exp != src_commit:
        print("error: --expected-source-commit != packet.source_commit_sha", file=sys.stderr)
        return 2
    if not man_cli.is_file():
        print(f"error: missing manifest file {man_cli}", file=sys.stderr)
        return 2
    live_man_sha = sha256_file(man_cli)
    if live_man_sha != man_sha:
        print("error: manifest file sha != packet.science_source_manifest_sha256", file=sys.stderr)
        return 2
    # E-G: frozen tool paths EXACT then self-rehash
    if gen_path != "scripts/lands_ab_science_source_manifest.py":
        print(f"error: generator_script_path must be scripts/lands_ab_science_source_manifest.py, got {gen_path!r}", file=sys.stderr)
        return 2
    if dry_path != "scripts/lands_ab_packet_dry_exec.py":
        print(f"error: dry_exec_tool_path must be scripts/lands_ab_packet_dry_exec.py, got {dry_path!r}", file=sys.stderr)
        return 2
    for rel, expect in ((gen_path, gen_sha), (dry_path, dry_sha)):
        fp = repo / rel
        if not fp.is_file():
            print(f"error: missing pinned tool/script {rel}", file=sys.stderr)
            return 2
        if sha256_file(fp) != expect:
            print(f"error: sha mismatch for {rel}", file=sys.stderr)
            return 2

    # load manifest entries
    try:
        man = json.loads(man_cli.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"error: manifest JSON: {e}", file=sys.stderr)
        return 2
    if man.get("schema") != "LANDS_AB_science_source_manifest/v1":
        print(f"error: manifest schema must be LANDS_AB_science_source_manifest/v1, got {man.get('schema')!r}", file=sys.stderr)
        return 2
    entries = man.get("entries")
    if not isinstance(entries, list) or not entries:
        print("error: manifest entries missing", file=sys.stderr)
        return 2
    if man.get("n_entries") != len(entries):
        print("error: manifest n_entries != len(entries)", file=sys.stderr)
        return 2
    paths = []
    for ent in entries:
        if not isinstance(ent, dict) or "path" not in ent or "sha256" not in ent:
            print("error: bad manifest entry", file=sys.stderr)
            return 2
        rel = ent["path"]
        if not isinstance(rel, str) or ".." in Path(rel).parts or rel.startswith("/"):
            print(f"error: bad/escape path {rel!r}", file=sys.stderr)
            return 2
        try:
            h = _require_hex64(ent["sha256"], field=f"manifest[{rel}].sha256")
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        paths.append(rel)
        fp = repo / rel
        if not fp.is_file():
            print(f"error: missing source path {rel}", file=sys.stderr)
            return 2
        if sha256_file(fp) != h:
            print(f"error: disk sha mismatch {rel}", file=sys.stderr)
            return 2
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        print("error: manifest paths not sorted unique", file=sys.stderr)
        return 2

    # H2: mandatory execution-source set must be present
    man_path_set = set(paths)
    missing_mand = [p for p in MANDATORY_EXECUTION_SOURCE_SET if p not in man_path_set]
    if missing_mand:
        print(
            f"error: missing mandatory source: {missing_mand[0]}",
            file=sys.stderr,
        )
        return 2

    # packet schema / rows
    schema = str(packet.get("schema") or "")
    if not schema.startswith("LANDS_AB_EVAL_launch_packet"):
        print(f"error: bad packet schema {schema!r}", file=sys.stderr)
        return 2
    rows = packet.get("gating_rows_exact") or packet.get("gating_rows")
    if list(rows) != GATING_ROWS and set(rows) != set(GATING_ROWS):
        # require exact set of 7
        if set(rows) != set(GATING_ROWS) or len(rows) != 7:
            print("error: gating_rows must be exact 7-tuple set", file=sys.stderr)
            return 2
    row_commands = packet.get("row_commands")
    if not isinstance(row_commands, list) or len(row_commands) != 7:
        print("error: row_commands must be list of 7", file=sys.stderr)
        return 2

    # claim ceiling REQUIRED all-false
    if "claim_ceiling" not in packet:
        print("error: claim_ceiling required", file=sys.stderr)
        return 2
    cc = packet.get("claim_ceiling")
    if not isinstance(cc, dict):
        print("error: claim_ceiling must be object", file=sys.stderr)
        return 2
    for k in (
        "LANDS_AB",
        "science_claim",
        "equivalent_minted",
        "full_sub2_runtime_ready_for_science",
    ):
        if k not in cc:
            print(f"error: claim_ceiling missing field {k}", file=sys.stderr)
            return 2
        if cc.get(k) is not False:
            print(f"error: claim_ceiling.{k} must be false", file=sys.stderr)
            return 2
    if packet.get("science_claim") is True:
        print("error: science_claim true forbidden", file=sys.stderr)
        return 2

    # branch authority REQUIRED
    LIVE_ORDER = [
        "BR-LANDS-AB-SCOPE-CREEP-STOP",
        "BR-LANDS-AB-FIXTURE-CONTRACT-FAIL",
        "BR-LANDS-AB-VACUOUS",
        "BR-LANDS-AB-DIVERGENT-EVENT",
        "BR-LANDS-AB-DIVERGENT-APPLY",
        "BR-LANDS-AB-DIVERGENT-ORACLE-LIVE",
        "BR-LANDS-AB-EQUIVALENT",
    ]
    po = packet.get("PRIORITY_ORDER") or packet.get("priority_order")
    if not isinstance(po, (list, tuple)) or list(po) != LIVE_ORDER:
        print("error: PRIORITY_ORDER must echo exact live order", file=sys.stderr)
        return 2
    bids = packet.get("branch_ids")
    if not isinstance(bids, (list, tuple)) or set(bids) != set(LIVE_ORDER) or len(bids) != 7:
        print("error: branch_ids must be exact live 7-set", file=sys.stderr)
        return 2

    # E-A: executor exact frozen spelling only
    FROZEN_EXECUTOR_ROLE = "claude_as_test_operator"
    ex = packet.get("executor")
    if not isinstance(ex, dict):
        print("error: executor required object", file=sys.stderr)
        return 2
    role = str(ex.get("role") or "")
    if role != FROZEN_EXECUTOR_ROLE:
        print(f"error: executor.role must be exactly {FROZEN_EXECUTOR_ROLE!r}, got {role!r}", file=sys.stderr)
        return 2
    forb = ex.get("forbidden_for_plan_dev")
    if not isinstance(forb, (list, tuple)) or not forb:
        print("error: executor.forbidden_for_plan_dev required non-empty", file=sys.stderr)
        return 2
    if not any("formal" in str(x).lower() for x in forb):
        print("error: executor.forbidden_for_plan_dev must prohibit formal run", file=sys.stderr)
        return 2
    if ex.get("one_terminal_receipt") is not True and packet.get("one_terminal_receipt") is not True:
        print("error: one_terminal_receipt must be true", file=sys.stderr)
        return 2

    # E-C harvest structural object
    rs = packet.get("runtime_scratch")
    if not isinstance(rs, dict):
        print("error: runtime_scratch required object", file=sys.stderr)
        return 2
    harvest = rs.get("harvest_exactly_one_raw_obs")
    if not isinstance(harvest, dict):
        print("error: runtime_scratch.harvest_exactly_one_raw_obs must be object", file=sys.stderr)
        return 2
    if harvest.get("applies_to_each_of_7_gating_rows") is not True:
        print("error: harvest applies_to_each_of_7_gating_rows must be true", file=sys.stderr)
        return 2
    helper = harvest.get("helper")
    if not isinstance(helper, str) or "harvest_exactly_one_raw_obs" not in helper:
        print("error: harvest.helper must name harvest_exactly_one_raw_obs", file=sys.stderr)
        return 2
    zom = str(harvest.get("zero_or_multiple") or "").upper()
    if zom not in ("STOP", "FAIL", "ERROR"):
        print("error: harvest.zero_or_multiple must be STOP semantics", file=sys.stderr)
        return 2

    # E-B execution_order subsequence + required lifecycle steps
    REQUIRED_LIFECYCLE = {
        "preflight",
        "activate_fresh_nonce_run_root_must_not_exist",
        "harvest_exactly_one_per_row",
        "EVIDENCE_CONSUMER",
        "terminal_classification_vs_prereg",
        "FORMAL_RUNTIME_CREATE_terminal_receipt",
    }
    eo = packet.get("execution_order")
    if not isinstance(eo, (list, tuple)) or not eo:
        print("error: execution_order required non-empty list", file=sys.stderr)
        return 2
    eo_list = [str(x) for x in eo]
    # seven gating rows exactly once each, in order as subsequence
    positions = []
    for gr in GATING_ROWS:
        if eo_list.count(gr) != 1:
            print(f"error: execution_order must contain gating row {gr} exactly once", file=sys.stderr)
            return 2
        positions.append(eo_list.index(gr))
    if positions != sorted(positions):
        print("error: gating rows in execution_order not in required order", file=sys.stderr)
        return 2
    missing_life = REQUIRED_LIFECYCLE - set(eo_list)
    if missing_life:
        print(f"error: execution_order missing lifecycle steps {sorted(missing_life)}", file=sys.stderr)
        return 2
    # seven-only list rejected by missing lifecycle above

    # helpers for path extraction
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

    runner_parser, _, live_runner_sha = load_runner_parser(repo)
    _ = live_runner_sha
    seen_phase: set[str] = set()
    seen_enforcer: set[str] = set()
    seen_rows: set[str] = set()
    seen_argv: set[str] = set()
    seen_raw: set[str] = set()
    seen_raw_basenames: set[str] = set()
    seen_runtime_basenames: set[str] = set()
    shared_scratch: str | None = None
    shared_run_root: str | None = None
    formal_argv_repo_paths: set[str] = set()

    # G7 packet env binding (required for one formal run)
    rs_bind = (packet.get("runtime_scratch") or {}).get("env_binding") or {}
    if not isinstance(rs_bind, dict):
        print("error: runtime_scratch.env_binding must be object", file=sys.stderr)
        return 2
    pkt_scratch = str(rs_bind.get("LANDS_AB_RUNTIME_SCRATCH") or "").strip()
    pkt_run_root = str(rs_bind.get("LANDS_AB_RUN_ROOT") or "").strip()
    if not pkt_scratch or not pkt_run_root:
        print("error: runtime_scratch.env_binding missing LANDS_AB_RUNTIME_SCRATCH/RUN_ROOT", file=sys.stderr)
        return 2

    for rc in row_commands:
        if not isinstance(rc, dict):
            print("error: row_command not object", file=sys.stderr)
            return 2
        gr = rc.get("gating_row")
        if gr not in GATING_ROWS or gr in seen_rows:
            print(f"error: bad/dup gating_row {gr!r}", file=sys.stderr)
            return 2
        seen_rows.add(gr)
        inv = rc.get("invocation") or {}
        if not isinstance(inv, dict):
            print(f"error: {gr} invocation must be object", file=sys.stderr)
            return 2
        argv = inv.get("argv_template") or inv.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            print(f"error: missing argv for {gr}", file=sys.stderr)
            return 2
        argv_key = json.dumps(argv)
        if argv_key in seen_argv:
            print(f"error: duplicate argv for {gr}", file=sys.stderr)
            return 2
        seen_argv.add(argv_key)
        # H2: collect formal argv repo-path tokens for later coverage
        for tok in argv:
            if _is_repo_path_token(tok):
                formal_argv_repo_paths.add(_repo_path_from_token(tok))

        env = inv.get("env_required") or {}
        if not isinstance(env, dict):
            print(f"error: {gr} env_required must be object", file=sys.stderr)
            return 2
        for req_env in ("PYTHONPATH", "LANDS_AB_RUNTIME_SCRATCH", "LANDS_AB_RUN_ROOT"):
            if req_env not in env or not str(env.get(req_env) or "").strip():
                print(f"error: {gr} missing env {req_env}", file=sys.stderr)
                return 2

        # F5/F6 + G7: run-root / scratch / nonce shape
        scratch = str(env["LANDS_AB_RUNTIME_SCRATCH"]).strip()
        run_root = str(env["LANDS_AB_RUN_ROOT"]).strip()
        if not any(tok in run_root for tok in NONCE_TOKENS):
            print(
                f"error: {gr} LANDS_AB_RUN_ROOT missing frozen nonce template token "
                f"(<{NONCE_TOKENS[0].strip('<>')}> or similar): {run_root!r}",
                file=sys.stderr,
            )
            return 2
        if not _is_descendant(run_root, scratch):
            print(
                f"error: {gr} LANDS_AB_RUN_ROOT must be descendant of LANDS_AB_RUNTIME_SCRATCH",
                file=sys.stderr,
            )
            return 2
        if str(run_root).startswith("/") and _is_descendant(
            run_root.replace("<nonce>", "NONCE").replace("<run_local_nonce>", "RUN_LOCAL_NONCE"),
            str(repo),
        ):
            print(f"error: {gr} LANDS_AB_RUN_ROOT must not be under repo root", file=sys.stderr)
            return 2
        if str(scratch).startswith("/") and _is_descendant(scratch, str(repo)):
            print(f"error: {gr} LANDS_AB_RUNTIME_SCRATCH must not be under repo root", file=sys.stderr)
            return 2
        # G7 one formal run: equal across rows + packet env_binding
        if scratch != pkt_scratch or run_root != pkt_run_root:
            print(
                f"error: {gr} env scratch/run_root must equal packet runtime_scratch.env_binding",
                file=sys.stderr,
            )
            return 2
        if shared_scratch is None:
            shared_scratch, shared_run_root = scratch, run_root
        elif scratch != shared_scratch or run_root != shared_run_root:
            print(f"error: {gr} scratch/run_root differs from other gating rows (one formal run)", file=sys.stderr)
            return 2

        # E-I write-path scan on this row's executable paths only
        for tok in _scan_write_paths_from_row(inv, list(argv), env):
            if _is_write_path_token(tok):
                print(f"error: forbidden write path token in {gr}: {tok}", file=sys.stderr)
                return 2

        is_cuda = str(gr).startswith("G_CUDA")

        # G8: raw aliases — all present must agree; canonical is required raw_obs_path_template
        raw_aliases = {
            "raw_obs_path_template": inv.get("raw_obs_path_template"),
            "terminal_collection_raw_obs_path_template": inv.get("terminal_collection_raw_obs_path_template"),
            "out_path_template": inv.get("out_path_template"),
        }
        present_raw = {k: str(v) for k, v in raw_aliases.items() if isinstance(v, str) and v.strip()}
        if "raw_obs_path_template" not in present_raw:
            print(f"error: {gr} missing required raw_obs_path_template", file=sys.stderr)
            return 2
        raw = present_raw["raw_obs_path_template"]
        for k, v in present_raw.items():
            if v != raw:
                print(f"error: {gr} raw alias {k} != raw_obs_path_template", file=sys.stderr)
                return 2
        if raw in seen_raw:
            print(f"error: duplicate raw path {raw}", file=sys.stderr)
            return 2
        seen_raw.add(raw)
        if not _is_descendant(raw, run_root):
            print(f"error: {gr} raw_obs path must be descendant of LANDS_AB_RUN_ROOT", file=sys.stderr)
            return 2
        if str(raw).startswith("/") and _is_descendant(
            raw.replace("<nonce>", "NONCE").replace("<run_local_nonce>", "RUN_LOCAL_NONCE"),
            str(repo),
        ):
            print(f"error: {gr} raw_obs path must not be under repo root", file=sys.stderr)
            return 2
        # G9: harvest pattern lands_ab_raw_obs_<gating_row>_*.json
        raw_base = Path(str(raw).replace("<nonce>", "N").replace("<run_local_nonce>", "R")).name
        if not raw_base.startswith(f"lands_ab_raw_obs_{gr}_"):
            print(
                f"error: {gr} raw basename must start with lands_ab_raw_obs_{gr}_ (harvest pattern)",
                file=sys.stderr,
            )
            return 2
        if raw_base in seen_raw_basenames:
            print(f"error: {gr} duplicate raw basename {raw_base}", file=sys.stderr)
            return 2
        seen_raw_basenames.add(raw_base)

        # G6: outer timeout == row hard bound (rule stated: == hard_timeout if present else duration_budget)
        if argv[0] != "timeout" or len(argv) < 3:
            print(f"error: {gr} argv must start with timeout <budget>", file=sys.stderr)
            return 2
        try:
            outer_to = _require_finite_positive(argv[1], field=f"{gr} outer timeout")
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        hard = _row_hard_timeout_seconds(rc, inv)
        if hard is None:
            print(f"error: {gr} missing duration_budget_seconds/hard_timeout_seconds for G6 bind", file=sys.stderr)
            return 2
        if float(outer_to) != float(hard):
            print(
                f"error: {gr} outer timeout {outer_to} != row hard bound {hard} "
                f"(rule: argv[1] == inv.hard_timeout_seconds or row.duration_budget_seconds)",
                file=sys.stderr,
            )
            return 2

        if not is_cuda:
            try:
                structural_preflight_row(list(argv), repo=repo, runner_parser=runner_parser)
            except Exception as e:
                print(f"error: structural preflight {gr}: {e}", file=sys.stderr)
                return 2
            out_val = _flag_val(list(argv), "--out")
            if out_val != raw:
                print(f"error: {gr} --out must equal raw_obs_path_template", file=sys.stderr)
                return 2
            continue

        # ---- CUDA rows: G1–G5 process segments + F2 node + budgets ----
        try:
            chain = _validate_ordered_cuda_chain(list(argv), gr=str(gr))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        phase_argv = chain["phase_argv"]
        enf_argv = chain["enf_argv"]
        budget_map = chain["budget_map"]

        # G5: declared phase_budgets_seconds values must equal argv budgets exactly
        pb_decl = inv.get("phase_budgets_seconds")
        if isinstance(pb_decl, dict):
            if set(pb_decl.keys()) != set(REQUIRED_PHASE_BUDGET_NAMES):
                print(f"error: {gr} declared phase_budgets_seconds keys mismatch required set", file=sys.stderr)
                return 2
            for name in REQUIRED_PHASE_BUDGET_NAMES:
                try:
                    decl_v = float(pb_decl[name])
                except Exception:
                    print(f"error: {gr} declared budget {name} not numeric", file=sys.stderr)
                    return 2
                if float(budget_map[name]) != float(decl_v):
                    print(
                        f"error: {gr} argv budget {name}={budget_map[name]} != declared {decl_v}",
                        file=sys.stderr,
                    )
                    return 2

        # F5/G8: phase + enforcer under run_root; alias equality
        for label, pth in (("phase", phase_argv), ("enforcer", enf_argv)):
            if not _is_descendant(pth, run_root):
                print(f"error: {gr} {label} path must be descendant of LANDS_AB_RUN_ROOT", file=sys.stderr)
                return 2
            if str(pth).startswith("/") and _is_descendant(
                pth.replace("<nonce>", "NONCE").replace("<run_local_nonce>", "RUN_LOCAL_NONCE"),
                str(repo),
            ):
                print(f"error: {gr} {label} path must not be under repo root", file=sys.stderr)
                return 2
            base = Path(str(pth).replace("<nonce>", "N").replace("<run_local_nonce>", "R")).name
            if base in seen_runtime_basenames:
                print(f"error: {gr} duplicate runtime basename {base}", file=sys.stderr)
                return 2
            seen_runtime_basenames.add(base)

        phase_aliases = {
            "phase_events_jsonl_template": inv.get("phase_events_jsonl_template"),
            "terminal_collection_phase_events_jsonl_path": inv.get("terminal_collection_phase_events_jsonl_path"),
        }
        for k, v in phase_aliases.items():
            if isinstance(v, str) and v.strip() and str(v) != phase_argv:
                print(f"error: {gr} phase alias {k} != --phase-events-jsonl", file=sys.stderr)
                return 2
        enf_aliases = {
            "enforcer_receipt_path_template": inv.get("enforcer_receipt_path_template"),
            "terminal_collection_enforcer_receipt_path": inv.get("terminal_collection_enforcer_receipt_path"),
        }
        present_enf = {k: str(v) for k, v in enf_aliases.items() if isinstance(v, str) and v.strip()}
        if not present_enf:
            print(f"error: {gr} missing enforcer metadata path", file=sys.stderr)
            return 2
        for k, v in present_enf.items():
            if v != enf_argv:
                print(f"error: {gr} enforcer alias {k} != --enforcer-receipt argv", file=sys.stderr)
                return 2

        phase_env = env.get("SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL")
        if not phase_env:
            print(f"error: {gr} missing phase env", file=sys.stderr)
            return 2
        if phase_env != phase_argv:
            print(f"error: {gr} phase env != --phase-events-jsonl argv", file=sys.stderr)
            return 2
        if phase_env in seen_phase:
            print(f"error: duplicate phase {phase_env}", file=sys.stderr)
            return 2
        seen_phase.add(phase_env)
        if enf_argv in seen_enforcer:
            print(f"error: duplicate enforcer {enf_argv}", file=sys.stderr)
            return 2
        seen_enforcer.add(enf_argv)

    if seen_rows != set(GATING_ROWS):
        print("error: row_commands incomplete vs GATING_ROWS", file=sys.stderr)
        return 2
    if len(seen_argv) != 7 or len(seen_raw) != 7:
        print("error: argv/raw paths not unique 7", file=sys.stderr)
        return 2
    cuda_rows = [g for g in GATING_ROWS if g.startswith("G_CUDA")]
    if len(seen_phase) != len(cuda_rows) or len(seen_enforcer) != len(cuda_rows):
        print("error: phase/enforcer path count mismatch vs CUDA rows", file=sys.stderr)
        return 2

    # H2: every formal argv repo-path token covered by manifest (or was allowlisted already)
    for rel in sorted(formal_argv_repo_paths):
        if rel not in man_path_set:
            print(f"error: formal argv path not covered by manifest: {rel}", file=sys.stderr)
            return 2

    print("PACKET_DRY_EXEC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
