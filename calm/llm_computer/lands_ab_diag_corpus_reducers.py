"""Pure reducers for the LANDS-AB dry-exec diagnostic-class corpus (A0).

Extracted from scripts/lands_ab_dry_exec_diag_corpus.py under Phase E of
A0_SUCCESSOR_PLAN_v2 (sha f381736d…): behavior-preserving, no semantic change.

Seam contract (architecture_discipline.md §Required Seams — "pure gate and status
reducers"): every function here takes plain data and returns plain data. This module
performs NO filesystem IO, NO subprocess execution, NO dynamic import, and imports no
launch/IO/GPU glue. It must never import the harness or the sources facade.

Dependency direction: harness -> reducers -> stdlib. Never the reverse.
"""
from __future__ import annotations

import ast
import copy
import re

V4_SHA = "b011fb91dc61593ef27081c50fdbb75962ff7c237eb818d23daa97df5bdaaa8c"
SPEC_KEYS = {"axis", "ops", "cli", "manifest", "extra_args", "packet_raw"}
OP_KINDS = {"set", "delete", "list_pop", "list_append", "reverse",
            "filter_out_entry_paths", "resync_n_entries"}
TOKEN_RE = re.compile(r"^([A-Z][0-9]+[a-z]?(?:/[A-Z][0-9]+[a-z]?)*)\b")


# --------------------------------------------------------------------- normalization

def normalize(text: str, repo_str: str) -> str:
    """Canonicalize one stderr line: repo path, tmp path, and sha shapes."""
    out = text.replace(repo_str, "<REPO>")
    out = re.sub(r"/tmp/[A-Za-z0-9_]+", "<TMP>", out)
    out = re.sub(r"[0-9a-f]{64}", "<SHA64>", out)
    out = re.sub(r"[0-9a-f]{40}", "<SHA40>", out)
    return re.sub(r"^error:\s*", "", out)


# --------------------------------------------------------------------- mutation grammar

def _get(doc, path):
    for key in path:
        doc = doc[key]
    return doc


def apply_ops(doc, ops):
    """Execute the declarative mutation grammar (see ROWS.json row_schema)."""
    for op in ops or []:
        kind = op["op"]
        if kind == "set":
            _get(doc, op["path"][:-1])[op["path"][-1]] = copy.deepcopy(op["value"])
        elif kind == "delete":
            _get(doc, op["path"][:-1]).pop(op["path"][-1], None)
        elif kind == "list_pop":
            _get(doc, op["path"]).pop(op.get("index", -1))
        elif kind == "list_append":
            _get(doc, op["path"]).append(copy.deepcopy(op["value"]))
        elif kind == "reverse":
            parent = _get(doc, op["path"][:-1])
            parent[op["path"][-1]] = list(reversed(_get(doc, op["path"])))
        elif kind == "filter_out_entry_paths":
            parent = _get(doc, op["path"][:-1])
            key = op["path"][-1]
            parent[key] = [e for e in _get(doc, op["path"]) if e.get("path") not in op["values"]]
        elif kind == "resync_n_entries":
            doc["n_entries"] = len(doc["entries"])
        else:
            raise ValueError(f"unknown mutation op: {kind}")
    return doc


# --------------------------------------------------------------------- schema preflight

def _tokens_of(rows) -> set:
    return {r["expected_class_token"] for r in rows if r["expected_class_token"] != "NONE"}


def preflight_schema(corpus: dict) -> list:
    """Fixed schema + ledger contract. Pure; no IO, no import, no subprocess."""
    fails = []

    def bad(code, **kw):
        fails.append(dict(code=code, **kw))

    rows = corpus.get("rows")
    if not isinstance(rows, list) or not rows:
        return [dict(code="rows_missing")]
    required = corpus.get("row_schema", {}).get("required_fields")
    if not isinstance(required, list) or not required:
        return [dict(code="required_fields_missing")]

    missing = {}
    for row in rows:
        for field in required:
            if field not in row:
                missing[field] = missing.get(field, 0) + 1
    if missing:
        bad("required_field_absent", fields=missing, of_rows=len(rows),
            required_count=len(required))

    ids = [r.get("row_id") for r in rows]
    if len(set(ids)) != len(rows):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        bad("row_id_not_unique", unique=len(set(ids)), rows=len(rows), duplicates=dupes[:10])
    sites = [r.get("intended_site_id") for r in rows]
    if len(set(sites)) != len(rows):
        dupes = sorted({s for s in sites if sites.count(s) > 1})
        bad("intended_site_not_unique", unique=len(set(sites)), rows=len(rows),
            duplicates=dupes[:10])

    for row in rows:
        spec = row.get("mutation_spec")
        if not isinstance(spec, dict):
            bad("mutation_spec_not_object", row_id=row.get("row_id"))
            continue
        extra = set(spec) - SPEC_KEYS
        if extra:
            bad("mutation_spec_unknown_key", row_id=row.get("row_id"), keys=sorted(extra))
        for op in spec.get("ops") or []:
            if not isinstance(op, dict) or op.get("op") not in OP_KINDS:
                bad("mutation_op_invalid", row_id=row.get("row_id"), op=op)
            elif not isinstance(op.get("path"), list):
                bad("mutation_op_path_invalid", row_id=row.get("row_id"), op=op.get("op"))

    # evidence tier <-> Ledger-B eligibility must be exact in BOTH directions
    cross = [r["row_id"] for r in rows
             if bool(r.get("independent_correctness_eligible")) != (r.get("evidence_tier") == "preregistered")]
    if cross:
        bad("tier_ledger_crossover", rows=cross[:10], count=len(cross))

    ledgers = corpus.get("ledgers") or {}
    a_rows = [r for r in rows if r.get("baseline_equivalence_eligible")]
    b_rows = [r for r in rows if r.get("independent_correctness_eligible")]
    for name, subset in (("A_equivalence", a_rows), ("B_independent_correctness", b_rows)):
        led = ledgers.get(name)
        if not isinstance(led, dict):
            bad("ledger_missing", ledger=name)
            continue
        if led.get("rows") != len(subset):
            bad("ledger_row_count", ledger=name, declared=led.get("rows"), actual=len(subset))
        toks = _tokens_of(subset)
        if set(led.get("class_tokens") or []) != toks:
            bad("ledger_token_set", ledger=name,
                declared_only=sorted(set(led.get("class_tokens") or []) - toks),
                actual_only=sorted(toks - set(led.get("class_tokens") or [])))
        if led.get("n_class_tokens") != len(toks):
            bad("ledger_token_count", ledger=name, declared=led.get("n_class_tokens"),
                actual=len(toks))

    census = corpus.get("census") or {}
    residual = corpus.get("residual_sites")
    n_res = len(residual) if isinstance(residual, list) else None
    if census.get("rows") != len(rows):
        bad("census_rows", declared=census.get("rows"), actual=len(rows))
    if census.get("residual") != n_res:
        bad("census_residual", declared=census.get("residual"), actual=n_res)
    if n_res is not None and census.get("total") != len(rows) + n_res:
        bad("census_reconciliation", declared=census.get("total"),
            actual=len(rows) + n_res)

    gen = (corpus.get("generated_from") or {}).get("sha256")
    if gen != V4_SHA:
        bad("generated_from_identity", expected=V4_SHA, got=gen)
    return fails


# --------------------------------------------------------------------- ledgers

def check_observed_vs_prereg(rows, observed):
    """PREREG ledger: live vs ROWS.json, on rc AND class token AND normalized msg_key.

    Three FIELDS from two SOURCES (live, prereg). This is not a baseline comparison --
    see check_observed_vs_baseline for the independent second reference.
    """
    failures = []
    for row in rows:
        got = observed.get(row["row_id"])
        if got is None:
            failures.append({"row_id": row["row_id"], "why": "no observation"})
            continue
        token = TOKEN_RE.match(got["msg_key"])
        got_token = token.group(1) if token else "NONE"
        if got["rc"] != row["expected_rc"]:
            failures.append({"row_id": row["row_id"], "why": "rc", "expected": row["expected_rc"], "got": got["rc"]})
        elif got["msg_key"] != row["expected_msg_key"]:
            failures.append({"row_id": row["row_id"], "why": "msg_key", "expected": row["expected_msg_key"], "got": got["msg_key"]})
        elif got_token != row["expected_class_token"]:
            failures.append({"row_id": row["row_id"], "why": "class_token", "expected": row["expected_class_token"], "got": got_token})
    return failures


def check_observed_vs_baseline(rows, observed, baseline):
    """BASELINE ledger: live vs the committed HEAD-9f471b3 observation map.

    Independent of ROWS: refreshing an expectation in ROWS cannot mask a behavior
    change here, which is the whole point of committing the baseline.
    """
    failures = []
    bmap = (baseline or {}).get("map") or {}
    for row in rows:
        rid = row["row_id"]
        got, want = observed.get(rid), bmap.get(rid)
        if want is None:
            failures.append({"row_id": rid, "why": "absent_from_baseline"})
        elif got is None:
            failures.append({"row_id": rid, "why": "no observation"})
        elif got.get("rc") != want.get("rc"):
            failures.append({"row_id": rid, "why": "rc", "baseline": want.get("rc"), "got": got.get("rc")})
        elif got.get("msg_key") != want.get("msg_key"):
            failures.append({"row_id": rid, "why": "msg_key", "baseline": want.get("msg_key"), "got": got.get("msg_key")})
    return failures


def check_determinism(first, second):
    diffs = []
    for key in sorted(set(first) | set(second)):
        if first.get(key) != second.get(key):
            diffs.append({"row_id": key, "first": first.get(key), "second": second.get(key)})
    return diffs


def check_tokens(rows, observed):
    """Bidirectional: every declared token observed, every observed token declared."""
    declared = _tokens_of(rows)
    seen = set()
    for row in rows:
        got = observed.get(row["row_id"]) or {}
        match = TOKEN_RE.match(got.get("msg_key", ""))
        if match:
            seen.add(match.group(1))
    return {"declared_not_observed": sorted(declared - seen), "observed_not_declared": sorted(seen - declared)}


def check_vacuity(observed, threshold=0.90):
    counts = {}
    for value in observed.values():
        counts[value["msg_key"]] = counts.get(value["msg_key"], 0) + 1
    if not counts:
        return {"verdict": "CORPUS_VACUOUS_FAIL", "reason": "no observations"}
    top = max(counts.values())
    share = top / len(observed)
    verdict = "CORPUS_VACUOUS_FAIL" if share > threshold else "OK"
    return {"verdict": verdict, "max_identical_share": round(share, 4), "distinct_msg_keys": len(counts)}


# --------------------------------------------------------------------- AST reducers

def census_from_source(source: str):
    """Re-derive the diagnostic-site census from tool SOURCE TEXT (fail-closed on drift)."""
    tree = ast.parse(source)
    prints, raises = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            # The module entry point raises SystemExit(main()); that is process exit,
            # not a diagnostic emission, and the frozen census excludes it.
            exc = node.exc
            name = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", None)
            if name == "SystemExit":
                continue
            raises += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if any(getattr(k, "arg", None) == "file" for k in node.keywords):
                prints += 1
    return {"stderr_prints": prints, "raises": raises, "total": prints + raises}


def ast_structure_report(source: str, step: str, allowlist: dict):
    """Per-step symbol-size / closure allowlist report over tool SOURCE TEXT."""
    tree = ast.parse(source)
    allowed = (allowlist.get("per_step", {}).get(step) or {}).get("allowed_over_150", [])
    over, closures, violations = [], [], []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lines = (node.end_lineno or node.lineno) - node.lineno + 1
        if lines > 150:
            over.append({"symbol": node.name, "lines": lines})
            if node.name not in allowed:
                violations.append({"symbol": node.name, "reason": "over_150_not_allowlisted"})
        nested = [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not node]
        if nested:
            closures.append({"symbol": node.name, "count": len(nested)})
    if step != "A0" and closures:
        violations += [{"symbol": c["symbol"], "reason": "nested_closure"} for c in closures]
    return {"check_id": "A0_AST_ALLOWLIST", "step": step, "over_150": over,
            "nested_closures": closures, "violations": violations,
            "verdict": "FAIL" if violations else "PASS"}


# --------------------------------------------------------------------- verdict

def acceptance_verdict(result: dict, baseline_required: bool) -> bool:
    """Fold every ledger into the single pass/fail decision."""
    tokens = result["token_rule"]
    ok = (not result["prereg_ledger_failures"] and not result["determinism_pair_diffs"]
          and not tokens["declared_not_observed"] and not tokens["observed_not_declared"]
          and result["vacuity"]["verdict"] == "OK"
          and result["structure"]["verdict"] == "PASS" and result["census_identity_ok"])
    if baseline_required:
        ok = ok and result["baseline_ledger_failures"] == []
    return ok
