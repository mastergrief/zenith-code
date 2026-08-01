#!/usr/bin/env python3
"""Deterministic LANDS-AB science source-set manifest generator (PLAN_v6 Phase A)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_imports(py_path: Path) -> set[str]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


def module_to_relpath(mod: str, *, repo: Path) -> Path | None:
    # only local calm/hrm_text_158/native_full_stack and scripts/lands_ab*
    if mod.startswith("calm.hrm_text_158.native_full_stack."):
        rel = Path(mod.replace(".", "/") + ".py")
        if (repo / rel).is_file():
            return rel
        pkg = Path(mod.replace(".", "/")) / "__init__.py"
        if (repo / pkg).is_file():
            return pkg
    if mod == "scripts.lands_ab_eval_run" or mod.startswith("scripts.lands_ab"):
        rel = Path(mod.replace(".", "/") + ".py")
        if (repo / rel).is_file():
            return rel
    return None


def walk(entry_rel: str, *, repo: Path, root_package: str) -> set[str]:
    seen: set[str] = set()
    stack = [entry_rel]
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        path = repo / rel
        if not path.is_file():
            continue
        seen.add(rel)
        if not rel.endswith(".py"):
            continue
        for mod in _parse_imports(path):
            if not (mod.startswith(root_package) or mod.startswith("scripts.lands_ab")):
                continue
            mpath = module_to_relpath(mod, repo=repo)
            if mpath is not None:
                stack.append(mpath.as_posix())
    return seen


# Non-tool force-includes. MUST NOT contain dry-exec tool modules —
# those enter only via DRY_EXEC_TOOL_MODULE_SET union (O3 pure).
MANDATORY_ALWAYS_BASE: tuple[str, ...] = (
    "scripts/lands_ab_eval_run.py",
    "scripts/lands_ab_plan_v4_characterization.py",
    "scripts/lands_ab_science_source_manifest.py",
    # H1: formal CUDA execution surfaces (repo-tracked executables)
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


def load_tool_module_set(*, owner_path: Path | None = None) -> tuple[str, ...]:
    """Load DRY_EXEC_TOOL_MODULE_SET from the O3 owner module (path importlib)."""
    if owner_path is None:
        owner_path = Path(__file__).resolve().parent / "lands_ab_dry_exec_tool_module_set.py"
    spec = importlib.util.spec_from_file_location(
        "lands_ab_dry_exec_tool_module_set", owner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool module set from {owner_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tool_set = getattr(mod, "DRY_EXEC_TOOL_MODULE_SET", None)
    if not isinstance(tool_set, tuple) or not tool_set:
        raise RuntimeError("DRY_EXEC_TOOL_MODULE_SET missing or empty")
    return tool_set


def mandatory_always_union(
    base: tuple[str, ...] | list[str],
    tool_set: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Generator-owned force-include construction (pure; testable)."""
    return tuple(sorted(set(base) | set(tool_set)))


def generator_force_include_result(
    *,
    owner_path: Path | None = None,
    base: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Live generator-owned MANDATORY_ALWAYS value after BASE ∪ tool-set."""
    b = MANDATORY_ALWAYS_BASE if base is None else tuple(base)
    return mandatory_always_union(b, load_tool_module_set(owner_path=owner_path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LANDS-AB science source manifest")
    ap.add_argument("--entry", action="append", default=[], help="entry relative path (repeatable)")
    ap.add_argument("--root-package", default="calm.hrm_text_158.native_full_stack")
    ap.add_argument("--also-include", action="append", default=[], help="force-include relative path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    if not args.entry:
        print("error: at least one --entry required", file=sys.stderr)
        return 2
    paths: set[str] = set()
    for e in args.entry:
        ep = Path(e)
        if ep.is_absolute():
            try:
                e = ep.resolve().relative_to(repo).as_posix()
            except ValueError:
                print(f"error: entry outside repo: {e}", file=sys.stderr)
                return 2
        if ".." in Path(e).parts:
            print(f"error: escape path: {e}", file=sys.stderr)
            return 2
        if not (repo / e).is_file():
            print(f"error: missing entry: {e}", file=sys.stderr)
            return 2
        paths |= walk(e, repo=repo, root_package=args.root_package)
        paths.add(e)
    for a in args.also_include:
        if ".." in Path(a).parts:
            print(f"error: escape path: {a}", file=sys.stderr)
            return 2
        if not (repo / a).is_file():
            print(f"error: missing also-include: {a}", file=sys.stderr)
            return 2
        if a in paths:
            # duplicates fail-closed for hostile suite? allow de-dupe by set; explicit dups in also-include OK
            pass
        paths.add(a)
    # detect duplicate listing in also-include list itself
    if len(args.also_include) != len(set(args.also_include)):
        print("error: duplicate also-include", file=sys.stderr)
        return 2
    # O3 pure: force-include = non-tool BASE ∪ DRY_EXEC_TOOL_MODULE_SET (owner-set union).
    # Dry path must NOT appear in MANDATORY_ALWAYS_BASE; it enters only via the union.
    try:
        owner_path = repo / "scripts" / "lands_ab_dry_exec_tool_module_set.py"
        MANDATORY_ALWAYS = generator_force_include_result(owner_path=owner_path)
    except Exception as exc:
        print(f"error: tool module set load: {exc}", file=sys.stderr)
        return 2
    for rel in MANDATORY_ALWAYS:
        if (repo / rel).is_file():
            paths.add(rel)
        else:
            print(f"error: mandatory path missing: {rel}", file=sys.stderr)
            return 2

    sorted_paths = sorted(paths)
    entries = []
    for rel in sorted_paths:
        path = repo / rel
        if not path.is_file():
            print(f"error: missing path: {rel}", file=sys.stderr)
            return 2
        entries.append({"path": rel, "sha256": sha256_file(path)})
    # ensure sorted unique
    if sorted_paths != sorted(set(sorted_paths)):
        print("error: unsorted or non-unique", file=sys.stderr)
        return 2
    payload = {
        "schema": "LANDS_AB_science_source_manifest/v1",
        "repo_root_note": str(repo),
        "entries": entries,
        "n_entries": len(entries),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(data)
    print("SCIENCE_SOURCE_MANIFEST_OK")
    print(f"n_entries={len(entries)} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
